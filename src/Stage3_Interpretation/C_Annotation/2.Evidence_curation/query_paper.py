"""Answer questions with paper-qa over one folder of PDFs and write the answers into a program JSON.

  python query_paper.py --program_json <lit_dir>/P1.json --pdf_dir <Literature_search>/brief__Q1 \
      --question "Does VIRMA regulate ZNF10 in endothelial cells?" "Which regulator drives P1?"

Only the PDFs directly in --pdf_dir are read (not subfolders). Each question's answer is cached as
'<cache_dir>/<key>.json', key = sha1(question + PDF filenames)[:12], so a re-run reuses it until the
question or the PDF set changes (or --overwrite). The PDFs are indexed once per run, and only when at
least one question is not cached.

Answers go to bundle['literature_evidence']['questions'][<key>]; answers already there are kept.
"""
import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "1.Search")]

import yaml  # noqa: E402

import qa  # noqa: E402
from src import load_env  # noqa: E402

BLOCK = "literature_evidence"


def question_key(question, pdf_names):
    text = question.strip() + "\n" + "\n".join(pdf_names)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def load_metas(pdf_dir):
    """Paper list from '<pdf_dir>/search_log.json' (run_literature_search), used for PMIDs / URLs."""
    path = Path(pdf_dir) / "search_log.json"
    return json.loads(path.read_text()).get("result", {}).get("papers", []) if path.exists() else []


def read_cache(path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return result if result.get("n_pdfs", 0) else None   # an earlier run parsed no PDF; retry it


async def answer_all(questions, pdf_dir, cache_dir, settings, overwrite):
    """{key: (result, ran paper-qa?)} for every question."""
    pdf_names = [p.name for p in qa.list_pdfs(pdf_dir)]
    keys = {q: question_key(q, pdf_names) for q in questions}
    out, docs, metas = {}, None, None
    for q, key in keys.items():
        cache = cache_dir / f"{key}.json"
        result = None if overwrite else read_cache(cache)
        ran = result is None
        if not pdf_names:
            result, ran = {"question": q, "verdict": None, "n_pdfs": 0, "status": "no_literature"}, False
        elif ran:
            if docs is None:
                print(f"[index] {len(pdf_names)} PDF(s) in {pdf_dir}")
                docs, metas = await qa.build_docs(pdf_dir, settings), load_metas(pdf_dir)
            result = await qa.answer_question(q, docs, settings, metas=metas)
            result["status"] = "answered" if result["n_pdfs"] else "no_parsed_pdf"
            result["created"] = datetime.now().isoformat(timespec="seconds")
            if result["n_pdfs"]:   # cache only when paper-qa indexed at least one PDF
                cache.write_text(json.dumps(result, indent=2))
        out[key] = (result, ran)
        print(f"  {key}: {result.get('status')} verdict={result.get('verdict')} "
              f"({'paper-qa run' if ran else 'cached'}) | {q}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description="paper-qa over one PDF folder; answers are cached and written into the program JSON.")

    # IO
    p.add_argument("--program_json", required=True, type=Path, help="Program bundle, e.g. Literature_info_extended_PerturbNMF_Info/P1.json.")
    p.add_argument("--pdf_dir", required=True, type=Path, help="Folder of PDFs (only *.pdf directly inside it), e.g. Literature_search/brief__Q1.")
    p.add_argument("--out_json", default=None, type=Path, help="Where to write the bundle. Default: --program_json (updated in place).")
    p.add_argument("--cache_dir", default=None, type=Path, help="Per-question answer cache. Default: <pdf_dir>/paperqa_cache.")
    p.add_argument("--config", default=str(HERE / "config.yaml"), help="paper-qa settings (qa: block).")

    # questions
    p.add_argument("--question", nargs="+", required=True, help="One or more questions, each quoted.")
    p.add_argument("--overwrite", action="store_true", help="Re-run paper-qa for questions that already have a cached answer.")
    return p


def main():
    args = build_parser().parse_args()
    if not args.program_json.exists():
        raise FileNotFoundError(f"Program JSON not found: {args.program_json}")
    if not args.pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF folder not found: {args.pdf_dir}")
    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    settings = qa.build_settings(cfg)
    cache_dir = args.cache_dir or args.pdf_dir / "paperqa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    questions = list(dict.fromkeys(q.strip() for q in args.question if q.strip()))
    print(f"[{args.program_json.name}] {len(questions)} question(s)")
    results = asyncio.run(answer_all(questions, args.pdf_dir, cache_dir, settings, args.overwrite))

    bundle = json.loads(args.program_json.read_text())
    block = bundle.setdefault(BLOCK, {})
    answers = block.setdefault("questions", {})
    for key, (result, _) in results.items():
        answers[key] = {**result, "pdf_dir": str(args.pdf_dir.resolve()), "cache_file": str(cache_dir / f"{key}.json")}
    counts = {}
    for r in answers.values():
        counts[str(r.get("verdict"))] = counts.get(str(r.get("verdict")), 0) + 1
    block.update(params={"qa": cfg.get("qa", {})}, n_questions=len(answers), n_by_verdict=counts)

    out = args.out_json or args.program_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    n_ran = sum(ran for _, ran in results.values())
    print(f"[done] {len(results)} answer(s) ({n_ran} paper-qa run(s)) -> {out} ({BLOCK}.questions); cache -> {cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
