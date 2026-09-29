"""Ad-hoc paper-qa over one folder of PDFs (no search; PDFs come from 1.Search/search_literature).

Two modes:

  # Free-form ranked-evidence report
  python run_qa.py --pdf_dir <Literature_search>/KIAA1429__ZNF10 \
      --question "How is VIRMA related to ZNF10?" --report_out report.md --json_out report.json

  # Gene-pair classification (same call curate_evidence.py makes per pair)
  python run_qa.py --pdf_dir <Literature_search>/KIAA1429__ZNF10 --pair ZNF10 KIAA1429 \
      --cell_type "endothelial" --json_out pair.json

For whole programs use curate_evidence.py instead.
"""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qa  # noqa: E402
from curate_evidence import load_config, load_env, load_search_log  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("run_qa")


def run_question(args, settings) -> None:
    session = asyncio.run(qa.answer(args.question, args.pdf_dir, settings))
    contexts = sorted(session.contexts, key=lambda c: -(c.score or -1))
    payload = {
        "question": args.question,
        "answer": session.answer,
        "n_pdfs": len(list(args.pdf_dir.glob("*.pdf"))),
        "contexts": [
            {"score": c.score, "summary": c.context,
             "citation": getattr(c.text.doc, "citation", None),
             "title": getattr(c.text.doc, "title", None)}
            for c in contexts
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        logger.info("Wrote %s", args.json_out)
    report = _format_report(payload)
    if args.report_out:
        Path(args.report_out).write_text(report)
        logger.info("Wrote %s", args.report_out)
    print(report)


def _format_report(payload: dict) -> str:
    lines = [f"# {payload['question']}", "", "## Answer", payload["answer"], "",
             f"## Ranked evidence ({len(payload['contexts'])} contexts, {payload['n_pdfs']} PDFs)"]
    for i, c in enumerate(payload["contexts"], 1):
        cite = c.get("citation") or c.get("title") or "unknown source"
        lines.append(f"{i}. [score {c['score']}] {cite}\n   {c['summary']}")
    return "\n".join(lines)


def run_pair(args, settings) -> None:
    gene_a, gene_b = args.pair
    log = load_search_log(args.pdf_dir) or {}
    metas = log.get("result", {}).get("papers", [])
    result = asyncio.run(qa.classify_pair(gene_a, gene_b, args.pdf_dir, settings,
                                          cell_type=args.cell_type, metas=metas))
    text = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text)
        logger.info("Wrote %s (category=%s)", args.json_out, result.get("category"))
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="paper-qa over a local folder of PDFs")
    parser.add_argument("--pdf_dir", required=True, type=Path, help="Folder of PDFs (e.g. Literature_search/<A>__<B>).")
    parser.add_argument("--config", default=str(HERE / "config.yaml"), help="paper-qa settings (qa: block).")
    parser.add_argument("--question", help="Free-form question to answer over the PDFs")
    parser.add_argument("--pair", nargs=2, metavar=("GENE_A", "GENE_B"),
                        help="Two gene symbols to classify a relationship between")
    parser.add_argument("--cell_type", default="", help="Optional cell-type context for pair mode")
    parser.add_argument("--json_out", help="Where to write the JSON result")
    parser.add_argument("--report_out", help="(question mode) Markdown report path")
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        parser.error(f"--pdf_dir not found: {args.pdf_dir}")
    load_env()
    settings = qa.build_settings(load_config(args.config))
    if args.pair:
        run_pair(args, settings)
    elif args.question:
        run_question(args, settings)
    else:
        parser.error("Provide either --question or --pair")


if __name__ == "__main__":
    main()
