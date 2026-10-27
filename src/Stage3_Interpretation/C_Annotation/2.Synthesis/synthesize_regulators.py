"""Read each regulator's per-program findings together with Claude: are they consistent across programs, and what
do they add up to? The model-assisted half of 2.Synthesis (build_ledgers.py is the deterministic half).

    python synthesize_regulators.py --run_dir <out_dir>/<run_name>/Annotation

Reads <run_dir>/Synthesis/{regulator_reports.jsonl, findings.jsonl, regulator_effects.tsv} (run build_ledgers.py
first). One call per regulator cited as support in findings of at least --min_programs programs: the prompt holds
the regulator's effect (sign, log2FC) on every program it significantly moves and every finding that cites it.
Writes <run_dir>/Synthesis/Regulator_synthesis/<REGULATOR>.json:

    {"regulator", "consistency": consistent | context_dependent | contradictory | insufficient,
     "claims": [{claim, kind, finding_ids, programs, confidence, caveat}], "review": [{finding_id, why}],
     "dropped", "input_sha256", "model", "prompts", "created"}

Finding ids outside the regulator's findings are removed (listed in dropped), and a claim left resting on fewer
than 2 programs is dropped whole. A cached answer is reused while its input (the regulator's findings and effects)
is unchanged. Run records go to Regulator_synthesis/_runs/<REGULATOR>.json (api/manifest.py). Claude answers through a
Claude Code session by default (--llm_backend session, two passes; no API cost); --llm_backend api needs
ANTHROPIC_API_KEY. API spend is logged per regulator under <run_dir>/Budget/regulator_<REGULATOR>.json.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "api")]
from budget import add_budget_args, attach, resolve_budget_dir  # noqa: E402
from manifest import StepRun  # noqa: E402
from prompts import load as load_prompt, record as prompt_record  # noqa: E402

SYSTEM = load_prompt("regulator_synthesis/system.md")
TASK = load_prompt("regulator_synthesis/task.md")


class ScreenClaim(BaseModel):
    claim: str
    kind: Literal["shared_mechanism", "context_dependent", "contradiction", "broad_effect"]
    finding_ids: List[str]
    confidence: Literal["high", "medium", "low"]
    caveat: str


class ReviewFlag(BaseModel):
    finding_id: str
    why: str


class RegulatorSynthesis(BaseModel):
    consistency: Literal["consistent", "context_dependent", "contradictory", "insufficient"]
    claims: List[ScreenClaim]
    review: List[ReviewFlag]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def regulator_input(report, effects, findings):
    """(prompt evidence text, {finding_id: finding}) for one regulator."""
    cited = {c["finding_id"]: findings[c["finding_id"]] for c in report["cited_in"]}
    lines = [f"# REGULATOR {report['regulator']}",
             f"Significant in {report['n_programs']} of {report['of_programs']} programs"
             + (f" (flags: {', '.join(report['flags'])})" if report["flags"] else "") + ".",
             "", "## Effect of its knockdown on each program it significantly moves"]
    for e in effects:
        lines.append(f"- {e['program']} {e['label']}: log2FC {float(e['log2fc']):+.2f} (adj p {float(e['adj_pval']):.2g}) "
                     f"-> {e['role']}")
    lines += ["", "## Findings that cite it (one program each)"]
    for fid, f in cited.items():
        kind = f["kind"] + (f", {f['mechanism_kind']}" if f.get("mechanism_kind") else "")
        lines.append(f"- {fid} ({kind}; {f['confidence']} confidence): {f['claim']}")
        if f.get("support_genes"):
            lines.append(f"  support genes: {', '.join(f['support_genes'])}")
    return "\n".join(lines), cited


def clean(result, cited):
    """Answer as stored: finding ids outside cited removed; claims left on fewer than 2 programs dropped."""
    dropped = {"finding_ids": set(), "claims": []}
    claims = []
    for c in result.claims:
        ids = list(dict.fromkeys(i.strip() for i in c.finding_ids))
        dropped["finding_ids"] |= {i for i in ids if i not in cited}
        ids = [i for i in ids if i in cited]
        programs = sorted({i.split(":")[0] for i in ids}, key=lambda p: int(p[1:]))
        if len(programs) < 2:
            dropped["claims"].append(c.claim)
            continue
        claims.append({"claim": c.claim.strip(), "kind": c.kind, "finding_ids": ids, "programs": programs,
                       "confidence": c.confidence, "caveat": c.caveat.strip()})
    review = [{"finding_id": r.finding_id.strip(), "why": r.why.strip()} for r in result.review]
    dropped["finding_ids"] |= {r["finding_id"] for r in review if r["finding_id"] not in cited}
    review = [r for r in review if r["finding_id"] in cited]
    return {"consistency": result.consistency, "claims": claims, "review": review,
            "dropped": {k: sorted(v) if isinstance(v, set) else v for k, v in dropped.items() if v}}


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_parser():
    p = argparse.ArgumentParser(description="Claude reads each regulator's per-program findings together.")
    p.add_argument("--run_dir", required=True, help="Folder holding Synthesis/ (run build_ledgers.py first).")
    p.add_argument("--synthesis_dir", default=None, help="build_ledgers.py output. Default: <run_dir>/Synthesis.")
    p.add_argument("--out_dir", default=None, help="Output folder. Default: <synthesis_dir>/Regulator_synthesis.")
    p.add_argument("--regulators", nargs="+", default=None, help="Regulators to synthesize (default: every regulator cited in --min_programs programs).")
    p.add_argument("--min_programs", type=int, default=2, help="Synthesize a regulator once findings of at least this many programs cite it.")
    p.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id.")
    p.add_argument("--effort", default="medium", help="Effort level: low | medium | high | xhigh | max.")
    p.add_argument("--max_tokens", type=int, default=6000, help="Answer token limit.")
    p.add_argument("--overwrite", action="store_true", help="Redo regulators whose cached answer was made for the same input.")
    p.add_argument("--llm_backend", default="session", choices=["api", "session"], help="Default session (no API cost). api = Anthropic API (ANTHROPIC_API_KEY); session = write each prompt to --session_dir and read the answer written there by a Claude Code session (api/session_llm.py; run twice: export, then import).")
    p.add_argument("--session_dir", default=None, help="Request / answer folder for --llm_backend session. Default: <run_dir>/Session_LLM.")
    p.add_argument("--session_model", default="claude-code-session", help="Model name recorded for session answers.")
    add_budget_args(p, reserve_default=0.0)
    return p


def main():
    from env import load_env, make_handler
    args = build_parser().parse_args()
    load_env()
    run_dir = Path(args.run_dir).resolve()
    syn = Path(args.synthesis_dir or run_dir / "Synthesis")
    if not (syn / "regulator_reports.jsonl").exists():
        sys.exit(f"no {syn / 'regulator_reports.jsonl'}: run build_ledgers.py first")
    out_dir = Path(args.out_dir or syn / "Regulator_synthesis")
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = {f["finding_id"]: f for f in read_jsonl(syn / "findings.jsonl")}
    with (syn / "regulator_effects.tsv").open() as fh:
        effects = list(csv.DictReader(fh, delimiter="\t"))
    reports = [r for r in read_jsonl(syn / "regulator_reports.jsonl")
               if (args.regulators is None and len({c["finding_id"].split(":")[0] for c in r["cited_in"]}) >= args.min_programs)
               or (args.regulators is not None and r["regulator"] in args.regulators)]
    if not reports:
        print("[done] no regulator is cited in findings of enough programs")
        return 0

    resolve_budget_dir(args, run_dir)
    if args.llm_backend == "session":
        from session_llm import session_llm
        handler = session_llm(args.session_dir or run_dir / "Session_LLM", "regulator_synthesis", args.session_model)
    else:
        handler = make_handler(args.model)
    model = getattr(handler, "model_label", args.model)
    prompts = prompt_record(SYSTEM, TASK)

    n_fail = 0
    for report in reports:
        reg = report["regulator"]
        text, cited = regulator_input(report, [e for e in effects if e["regulator"] == reg], findings)
        input_sha = hashlib.sha256(text.encode()).hexdigest()
        out = out_dir / f"{safe(reg)}.json"
        budget = attach(handler, args, f"regulator_{safe(reg)}", "regulator_synthesis")
        run = StepRun(out_dir, safe(reg), "regulator_synthesis", model, prompts, budget)
        if out.exists() and not args.overwrite and json.loads(out.read_text()).get("input_sha256") == input_sha:
            print(f"  {reg}: cached ({out})")
            run.finish("cached")
            continue
        handler.key = safe(reg)
        result = handler.get_json(SYSTEM.text, f"{text}\n\n---\n\n{TASK.text}", RegulatorSynthesis,
                                  effort=args.effort, max_tokens=args.max_tokens)
        if result is None:
            if getattr(handler, "last_status", None) == "pending":
                run.finish("pending_session")
                continue
            print(f"  {reg}: [fail] no answer returned")
            run.finish("failed", errors=[getattr(handler, "last_error", None) or "no answer returned"])
            n_fail += 1
            continue
        record = {"regulator": reg, **clean(result, cited), "n_findings": len(cited), "input_sha256": input_sha,
                  "model": model, "prompts": prompts, "created": datetime.now().isoformat(timespec="seconds")}
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        run.finish("completed", {"consistency": record["consistency"], "n_claims": len(record["claims"]),
                                 "n_review": len(record["review"]),
                                 "dropped": {k: len(v) for k, v in record["dropped"].items()}})
        print(f"  {reg}: {record['consistency']}, {len(record['claims'])} claim(s), {len(record['review'])} to review"
              + (f", dropped {record['dropped']}" if record["dropped"] else ""))
    print(f"[done] {len(reports) - n_fail}/{len(reports)} regulator(s) -> {out_dir}")
    if args.llm_backend == "session":
        handler.report()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
