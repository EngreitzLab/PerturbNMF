"""Name each program and describe its function in one sentence, with Claude.

Reads '<info_dir>/P<k>.json' (any AGeneTic bundle: PerturbNMF_Info, Gene_info_extended or
Literature_info_extended) and writes '<out_dir>/P<k>.json':

    {"program_id", "program_name", "function_sentence", "model", "source", "created"}

The prompt holds the top-loading and distinctive genes (with MyGene names when present), the
regulators with their log2FC sign, the GO terms and, when 1.1.Search_literature has run, the
literature summary. The input bundles are never modified. Cached labels are reused unless
--overwrite. Env: ANTHROPIC_API_KEY (or C_Annotation/.env).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "1.Search"))
sys.path.insert(0, str(HERE.parent / "api"))
from html_report import gene_name, load_bundles, select_regulators  # noqa: E402
from budget import add_budget_args, attach  # noqa: E402
from src import load_env, make_handler  # noqa: E402

SYSTEM_PROMPT = (
    "You are a cell biologist naming gene programs found by consensus NMF on a CRISPRi Perturb-seq screen. "
    "Use only the evidence given. Be specific to the strongest shared function; avoid generic labels "
    "such as 'housekeeping' unless the evidence offers nothing better."
)


class ProgramLabel(BaseModel):
    program_name: str
    function_sentence: str


def gene_line(entry):
    g = gene_name(entry)
    name = (entry.get("gene_info", {}).get("MyGene", {}).get("name") if isinstance(entry, dict) else None)
    return f"{g} ({name})" if name else g


def literature_summary(bundle):
    """Summary text of every literature_search task, joined."""
    return "\n\n".join(t["summary"] for t in bundle.get("literature_search", {}).values()
                       if isinstance(t, dict) and t.get("summary"))


def build_prompt(bundle, top_regulator=6):
    regs = select_regulators(bundle, top_regulator)
    reg_lines = [f"{r['gene']}: log2FC {r['log2fc']:+.2f} on knockdown ({r['condition']}, adj p {r['adj_pval']:.2g}) -> "
                 + ("activator (knockdown lowers the program)" if r["log2fc"] < 0 else "repressor (knockdown raises the program)")
                 for r in regs if r.get("log2fc") is not None]
    parts = [
        f"Program {bundle.get('program_id')} in {bundle.get('organism', '')} {bundle.get('cell_type', '')}.",
        "Top-loading genes:\n" + "\n".join(f"- {gene_line(e)}" for e in bundle.get("program_genes", [])),
        "Distinctive genes (high in this program, low in others):\n"
        + "\n".join(f"- {gene_line(e)}" for e in bundle.get("distinctive_genes", [])),
        "Significant perturbation regulators:\n" + ("\n".join(f"- {x}" for x in reg_lines) or "- none"),
        "Enriched GO terms:\n" + "\n".join(f"- {t}" for t in bundle.get("GO", [])),
    ]
    lit = literature_summary(bundle)
    if lit:
        parts.append("Literature summary (from retrieved papers):\n" + lit)
    parts.append("Return program_name: a short title-case name of at most 8 words. "
                 "Return function_sentence: exactly one sentence (at most 40 words) on the program's biological function "
                 "in this cell type, naming 2-4 anchor genes.")
    return "\n\n".join(parts)


def build_parser():
    p = argparse.ArgumentParser(description="Write a Claude-generated name and one-sentence function per program.")

    # IO
    p.add_argument("--info_dir", required=True, help="Folder of P<k>.json bundles (Literature_info_extended_PerturbNMF_Info recommended).")
    p.add_argument("--out_dir", default=None, help="Output folder. Default: <info_dir>/../Program_label.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids, space separated (e.g. 1 2 3). Default: every P<k>.json in --info_dir.")
    p.add_argument("--top_regulator", type=int, default=6, help="Regulators passed to the prompt (best adj_pval over conditions).")

    # LLM
    p.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id.")
    p.add_argument("--effort", default="medium", help="Effort level: low | medium | high | xhigh | max.")
    p.add_argument("--overwrite", action="store_true", help="Re-label programs that already have a label file.")
    p.add_argument("--llm_backend", default="api", choices=["api", "session"], help="api = Anthropic API (ANTHROPIC_API_KEY); session = write each prompt to --session_dir and read the answer written there by a Claude Code session (api/session_llm.py; run twice: export, then import).")
    p.add_argument("--session_dir", default=None, help="Request / answer folder for --llm_backend session.")
    p.add_argument("--session_model", default="claude-code-session", help="Model name recorded for session answers.")
    add_budget_args(p, reserve_default=0.3)
    return p


def main():
    args = build_parser().parse_args()
    load_env()
    bundles = load_bundles(args.info_dir, args.programs)
    out_dir = Path(args.out_dir or Path(args.info_dir).resolve().parent / "Program_label")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.llm_backend == "session":
        from session_llm import session_llm
        handler = session_llm(args.session_dir, "label", args.session_model)
    else:
        handler = make_handler(args.model)
    model = getattr(handler, "model_label", args.model)

    n_fail = 0
    for pid, bundle in bundles.items():
        out = out_dir / f"P{pid}.json"
        if out.exists() and not args.overwrite:
            print(f"  P{pid}: cached ({out})")
            continue
        attach(handler, args, f"P{pid}", "label")
        handler.key = f"P{pid}"
        label = handler.get_json(SYSTEM_PROMPT, build_prompt(bundle, args.top_regulator), ProgramLabel, effort=args.effort)
        if label is None:
            if args.llm_backend == "session":
                continue
            print(f"  P{pid}: [fail] no label returned")
            n_fail += 1
            continue
        out.write_text(json.dumps({
            "program_id": f"P{pid}",
            "program_name": label.program_name.strip(),
            "function_sentence": label.function_sentence.strip(),
            "model": model,
            "source": str(Path(args.info_dir).resolve() / f"P{pid}.json"),
            "created": datetime.now().isoformat(timespec="seconds"),
        }, indent=2, ensure_ascii=False))
        print(f"  P{pid}: {label.program_name}")
    print(f"[done] {len(bundles) - n_fail}/{len(bundles)} program(s) -> {out_dir}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
