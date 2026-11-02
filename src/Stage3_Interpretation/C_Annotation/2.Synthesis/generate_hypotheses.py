"""Read every program of the screen together with Claude and propose its top hypotheses: the screen-level end of
2.Synthesis, shown as the last section of the viewer.

    python generate_hypotheses.py --run_dir <out_dir>/<run_name>/Annotation

Reads <run_dir>/Synthesis/{findings.jsonl, regulator_reports.jsonl} (run build_ledgers.py first) and each program's
label.json. One call for the whole screen: the prompt holds every program's label and function sentence, every
finding (the interpretation slots and modules, with their support genes and regulators) and every regulator
significant in any program (the perturbed regulators) with its effect signs. Writes <run_dir>/Synthesis/hypotheses.json:

    {"hypotheses": [{rank, title, claim, programs, regulators, finding_ids, rationale, experiment, confidence,
                     caveat}], "labels": {P<k>: label}, "dropped", "n_requested", "input_sha256", "model",
     "prompts", "created"}

Finding ids not in the run are removed (listed in dropped); programs are those the finding ids belong to. A
hypothesis is dropped whole when it names a regulator that is not a perturbed regulator significant in the run (a
program gene described as a knockdown) or is left with no finding. These hypotheses come from one reading of the
annotations: they are not tested against the data (perturbseq-hypotheses-skill does that). A cached answer is
reused while its input is unchanged. The run record goes to Synthesis/_runs/hypotheses.json (api/manifest.py) and
API spend to <run_dir>/Budget/hypotheses.json. --llm_backend session answers through a Claude Code session instead
(two passes, no API cost).
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "api"), str(HERE.parent / "0.Prepare")]
from budget import add_budget_args, attach, resolve_budget_dir  # noqa: E402
from manifest import StepRun  # noqa: E402
from program_dir import list_programs  # noqa: E402
from prompts import load as load_prompt, record as prompt_record  # noqa: E402

SYSTEM = load_prompt("hypotheses/system.md")
TASK = load_prompt("hypotheses/task.md")


class Hypothesis(BaseModel):
    title: str
    claim: str
    programs: List[str]
    regulators: List[str]
    finding_ids: List[str]
    rationale: str
    experiment: str
    confidence: Literal["high", "medium", "low"]
    caveat: str


class ScreenHypotheses(BaseModel):
    hypotheses: List[Hypothesis]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def screen_input(meta, labels, findings, reports):
    """Prompt evidence text for the whole screen."""
    lines = [f"# SCREEN: CRISPRi Perturb-seq in {meta.get('cell_type') or 'unknown cell type'} "
             f"({meta.get('organism') or 'human'}), {len(labels)} programs",
             "", "## Programs (label: function)"]
    lines += [f"- {pid} {lab['program_name']}: {lab.get('function_sentence', '')}" for pid, lab in labels.items()]
    lines += ["", "## Findings (per-program interpretation claims and modules)"]
    for f in findings:
        lines.append(f"- {f['finding_id']} [{f['confidence']}]: {f['claim']}")
        support = [("genes", f.get("support_genes")), ("regulators", f.get("support_regulators")),
                   ("terms", f.get("support_terms"))]
        support = "; ".join(f"{k}: {', '.join(v)}" for k, v in support if v)
        if support:
            lines.append(f"  {support}")
    lines += ["", "## Perturbed regulators: every knocked-down gene significant in at least one program "
              "(knockdown log2FC: - = activates, + = represses)"]
    for r in sorted(reports, key=lambda r: (-r["n_programs"], r["regulator"])):
        flags = f" [{', '.join(r['flags'])}]" if r["flags"] else ""
        lines.append(f"- {r['regulator']} ({r['n_programs']}/{r['of_programs']} programs{flags}): "
                     f"activates {'; '.join(r['activator_of']) or '-'} | represses {'; '.join(r['repressor_of']) or '-'}")
    return "\n".join(lines)


def clean(result, finding_ids, regulators, n):
    """Answer as stored: unknown finding ids removed, programs from the finding ids; hypotheses naming an unknown
    regulator or left without a finding dropped; at most n kept."""
    dropped = {"finding_ids": set(), "regulators": set(), "hypotheses": []}
    out = []
    for h in result.hypotheses:
        ids = list(dict.fromkeys(i.strip() for i in h.finding_ids))
        dropped["finding_ids"] |= {i for i in ids if i not in finding_ids}
        ids = [i for i in ids if i in finding_ids]
        regs = list(dict.fromkeys(r.strip() for r in h.regulators))
        unknown = {r for r in regs if r not in regulators}
        dropped["regulators"] |= unknown
        if unknown or not ids:
            dropped["hypotheses"].append(h.title)
            continue
        programs = sorted({i.split(":")[0] for i in ids}, key=lambda p: int(p[1:]))
        out.append({"rank": len(out) + 1, "title": h.title.strip(), "claim": h.claim.strip(), "programs": programs,
                    "regulators": regs, "finding_ids": ids, "rationale": h.rationale.strip(),
                    "experiment": h.experiment.strip(), "confidence": h.confidence, "caveat": h.caveat.strip()})
        if len(out) == n:
            break
    return out, {k: sorted(v) if isinstance(v, set) else v for k, v in dropped.items() if v}


def build_parser():
    p = argparse.ArgumentParser(description="Claude reads every program together and proposes the screen's top hypotheses.")
    p.add_argument("--run_dir", required=True, help="Annotation run folder holding Programs/ and Synthesis/ (run build_ledgers.py first).")
    p.add_argument("--synthesis_dir", default=None, help="build_ledgers.py output. Default: <run_dir>/Synthesis.")
    p.add_argument("--out", default=None, help="Output JSON. Default: <synthesis_dir>/hypotheses.json.")
    p.add_argument("--n_hypotheses", type=int, default=10, help="Hypotheses asked for (fewer are returned when the evidence does not support more).")
    p.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id.")
    p.add_argument("--effort", default="high", help="Effort level: low | medium | high | xhigh | max.")
    p.add_argument("--max_tokens", type=int, default=32000, help="Answer token limit (thinking at --effort high counts toward it).")
    p.add_argument("--overwrite", action="store_true", help="Redo even when the cached answer was made for the same input.")
    p.add_argument("--llm_backend", default="api", choices=["api", "session"], help="api = Anthropic API (ANTHROPIC_API_KEY, C_Annotation/.env); session = write the prompt to --session_dir and read the answer written there by a Claude Code session (api/session_llm.py; run twice: export, then import).")
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
    if not (syn / "findings.jsonl").exists():
        sys.exit(f"no {syn / 'findings.jsonl'}: run build_ledgers.py first")
    out = Path(args.out or syn / "hypotheses.json")
    meta_path = run_dir / "Programs" / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    labels = {}
    for d in list_programs(run_dir):
        if (d / "label.json").exists():
            labels[d.name] = json.loads((d / "label.json").read_text())
    findings = read_jsonl(syn / "findings.jsonl")
    reports = read_jsonl(syn / "regulator_reports.jsonl")
    text = screen_input(meta, labels, findings, reports)
    prompt = f"{text}\n\n---\n\n{TASK.text}\n\nNumber of hypotheses requested: {args.n_hypotheses}."
    input_sha = hashlib.sha256(prompt.encode()).hexdigest()

    resolve_budget_dir(args, run_dir)
    if args.llm_backend == "session":
        from session_llm import session_llm
        handler = session_llm(args.session_dir or run_dir / "Session_LLM", "hypotheses", args.session_model)
    else:
        handler = make_handler(args.model)
    model = getattr(handler, "model_label", args.model)
    prompts = prompt_record(SYSTEM, TASK)
    budget = attach(handler, args, "hypotheses", "hypotheses")
    run = StepRun(syn, "hypotheses", "hypotheses", model, prompts, budget)
    if out.exists() and not args.overwrite and json.loads(out.read_text()).get("input_sha256") == input_sha:
        print(f"[done] cached ({out})")
        run.finish("cached")
        return 0
    print(f"  prompt: {len(labels)} programs, {len(findings)} findings, {len(prompt):,} characters")
    handler.key = "hypotheses"
    result = handler.get_json(SYSTEM.text, prompt, ScreenHypotheses, effort=args.effort, max_tokens=args.max_tokens)
    if result is None:
        if getattr(handler, "last_status", None) == "pending":
            run.finish("pending_session")
            handler.report()
            return 0
        run.finish("failed", errors=[getattr(handler, "last_error", None) or "no answer returned"])
        sys.exit("[fail] no answer returned")
    hypotheses, dropped = clean(result, {f["finding_id"] for f in findings}, {r["regulator"] for r in reports},
                                args.n_hypotheses)
    record = {"hypotheses": hypotheses, "labels": {pid: lab["program_name"] for pid, lab in labels.items()},
              "dropped": dropped, "n_requested": args.n_hypotheses, "input_sha256": input_sha, "model": model,
              "prompts": prompts, "created": datetime.now().isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    run.finish("completed", {"n_hypotheses": len(hypotheses), "dropped": {k: len(v) for k, v in dropped.items()}})
    for h in hypotheses:
        print(f"  {h['rank']}. [{h['confidence']}] {h['title']} ({', '.join(h['programs'])})")
    print(f"[done] {len(hypotheses)} hypotheses -> {out}" + (f"; dropped {dropped}" if dropped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
