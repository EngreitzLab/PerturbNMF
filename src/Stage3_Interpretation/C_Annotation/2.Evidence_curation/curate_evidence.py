"""Curate literature evidence: OmniPath JSON + paper-qa over the literature PDFs.

Bundles are read from '<lit_dir>/P<k>.json' (Literature_info_extended_PerturbNMF_Info, written by
1.1.Search_literature) when present, else '<info_dir>/P<k>.json'; results are written to '<lit_dir>'.

Pairs -- for every OmniPath-tested pair of a program:

  found: true            -> evidence taken from the OmniPath JSON (no paper-qa run)
  found: false + PDFs    -> paper-qa classify_pair over '<literature_dir>/<A>__<B>/*.pdf',
                            cached as '<evidence_dir>/<A>__<B>.json' (reused on re-runs)
  found: false, no PDFs  -> status no_literature

Questions -- for every literature_plan.<plan>.curation_questions entry (llm_query_agent.py):
  paper-qa answer_question over the PDFs of its linked query folders
  ('<literature_dir>/<plan>__<Q>/') plus '<lit_dir>/gene_pdfs/<GENE>/' for C7 cell-type questions,
  cached as '<evidence_dir>/questions/<plan>__<C>.json'.

Pair results go to gene_interactions.Literature, question results to literature_evidence (other
blocks are kept), plus '<evidence_dir>/meta.json'.
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENETIC_DIR = HERE.parent
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

import qa  # noqa: E402
from evidence_cache import cache_path, pair_key, read_result, write_result  # noqa: E402

SOURCE = "Literature"
LIT_FOLDER = "Literature_info_extended_PerturbNMF_Info"


# loaders
def load_env(path=AGENETIC_DIR / ".env"):
    """Read KEY=VALUE lines from AGeneTic/.env into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def load_search_log(pair_dir):
    path = Path(pair_dir) / "search_log.json"
    return json.loads(path.read_text()) if path.exists() else None


# evidence
def omnipath_evidence(pair):
    """Evidence record of a pair OmniPath found, straight from its JSON."""
    return {
        "status": "omnipath",
        "source": "OmniPath",
        "interaction_categories": pair["interaction_categories"],
        "directions": pair["directions"],
        "signs": pair["signs"],
        "resources": pair["resources"],
        "n_references": pair["n_references"],
        "pmids": pair["pmids"],
    }


def paperqa_evidence(result, evidence_file, n_pdfs, search_log_file, web_hits_file):
    """Evidence record of a pair from its classify_pair result."""
    if result["n_pdfs"] == 0:
        status = "no_parsed_pdf"
    elif result["category"] is None:
        status = "no_relation"
    else:
        status = "classified"
    return {
        "status": status,
        "source": "paper-qa",
        "category": result["category"],
        "directed": result["directed"],
        "short_excerpt": result["short_excerpt"],
        "citation": {k: result.get(k) for k in ("title", "year", "doi", "url", "pmids")},
        "n_pdfs": n_pdfs,
        "n_pdfs_parsed": result["n_pdfs"],
        "evidence_file": str(evidence_file) if evidence_file.exists() else None,
        "search_log_file": str(search_log_file),
        "web_hits_file": str(web_hits_file) if web_hits_file.exists() else None,
    }


async def curate_pair(pair, cell_type, lit_dir, evidence_dir, settings, overwrite):
    """(evidence record, ran paper-qa?) for one OmniPath pair."""
    if pair["found"]:
        return omnipath_evidence(pair), False
    gene, reg = pair["gene"], pair["regulator"]
    pair_dir = Path(lit_dir) / pair_key(gene, reg)
    n_pdfs = len(list(pair_dir.glob("*.pdf")))
    log = load_search_log(pair_dir)
    if log is None or n_pdfs == 0:
        return {"status": "no_literature" if log else "not_searched", "source": None,
                "category": None, "n_pdfs": n_pdfs}, False

    evidence_file = cache_path(evidence_dir, gene, reg)
    result = None if overwrite else read_result(evidence_dir, gene, reg)
    if result is not None and result.get("n_pdfs", 0) == 0:
        result = None   # an earlier run could not index any PDF (e.g. API error); retry it
    ran = result is None
    if ran:
        metas = log.get("result", {}).get("papers", [])
        result = await qa.classify_pair(gene, reg, pair_dir, settings, cell_type=cell_type, metas=metas)
        if result["n_pdfs"]:   # cache only when paper-qa indexed at least one PDF
            write_result(evidence_dir, result)
    return paperqa_evidence(result, evidence_file, n_pdfs, pair_dir / "search_log.json",
                            pair_dir / "web_hits.json"), ran


async def curate_program(bundle, args, settings):
    omnipath = bundle.get("gene_interactions", {}).get("OmniPath")
    if omnipath is None:
        raise ValueError(f"{bundle.get('program_id')}: no gene_interactions.OmniPath block "
                         "(run 1.Search/1.0.Search_database/search_gene_interaction/search_OmniPath.py first)")
    cell_type = args.cell_type if args.cell_type is not None else bundle.get("cell_type", "")
    pairs, n_ran = {}, 0
    for name, pair in omnipath["pairs"].items():
        record, ran = await curate_pair(pair, cell_type, args.literature_dir, args.evidence_dir,
                                        settings, args.overwrite)
        pairs[name] = {"gene": pair["gene"], "regulator": pair["regulator"],
                       "query_category": pair["query_category"], **record}
        n_ran += ran
        print(f"    {name}: {record['status']}"
              f"{' category=' + str(record.get('category')) if record.get('source') == 'paper-qa' else ''}"
              f"{' (paper-qa run)' if ran else ''}")
    counts = {}
    for p in pairs.values():
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    block = {"params": {"cell_type": cell_type, "qa": args.qa_params},
             "n_pairs": len(pairs), "n_by_status": counts, "pairs": pairs}
    return block, n_ran


def question_pdf_dirs(question, lit_dir, literature_dir):
    dirs = [Path(literature_dir) / qid.replace(":", "__") for qid in question["query_ids"]]
    if question["question_type"] == "C7_cell_type":
        dirs += [Path(lit_dir) / "gene_pdfs" / g for g in question["genes"]]
    return [d for d in dirs if d.exists()]


async def curate_questions(bundle, args, settings):
    """(literature_evidence block, n paper-qa runs) for every curation question of the bundle."""
    questions, n_ran = {}, 0
    for plan_name, plan in bundle.get("literature_plan", {}).items():
        for q in plan.get("curation_questions", []):
            dirs = question_pdf_dirs(q, args.lit_dir, args.literature_dir)
            cache = args.evidence_dir / "questions" / f"{q['id'].replace(':', '__')}.json"
            result = None if args.overwrite or not cache.exists() else json.loads(cache.read_text())
            if result is not None and result.get("n_pdfs", 0) == 0:
                result = None
            ran = result is None
            if not dirs:
                result, ran = {"verdict": None, "n_pdfs": 0, "status": "no_literature"}, False
            elif ran:
                metas = [p for d in dirs for p in ((load_search_log(d) or {}).get("result", {}).get("papers", []))]
                result = await qa.answer_question(q["question"], dirs, settings, metas=metas)
                result["status"] = "answered" if result["n_pdfs"] else "no_parsed_pdf"
                if result["n_pdfs"]:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(result, indent=2))
            n_ran += ran
            questions[q["id"]] = {"plan": plan_name, "question_type": q["question_type"], "question": q["question"],
                                  "genes": q["genes"], "query_ids": q["query_ids"],
                                  "pdf_dirs": [str(d) for d in dirs], **result}
            print(f"    {q['id']}: {result.get('status')} verdict={result.get('verdict')}"
                  f"{' (paper-qa run)' if ran else ''}")
    counts = {}
    for r in questions.values():
        counts[str(r.get("verdict"))] = counts.get(str(r.get("verdict")), 0) + 1
    return {"params": {"qa": args.qa_params}, "n_questions": len(questions),
            "n_by_verdict": counts, "questions": questions}, n_ran


def build_parser():
    p = argparse.ArgumentParser(description="Curate gene-regulator evidence from OmniPath JSON and paper-qa over literature PDFs.")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database.")
    p.add_argument("--lit_dir", default=None, help=f"{LIT_FOLDER} folder (read first, and written). Default: <info_dir>/../{LIT_FOLDER}.")
    p.add_argument("--literature_dir", default=None, help="Literature_search folder from 1.1.Search_literature/run_literature_search.py. Default: <lit_dir>/Literature_search.")
    p.add_argument("--evidence_dir", default=None, help="paper-qa cache folder. Default: <info_dir>/../Evidence_curation.")
    p.add_argument("--config", default=str(HERE / "config.yaml"), help="paper-qa settings (qa: block).")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--cell_type", default=None, help="Cell-type context for paper-qa. Default: the bundle's cell_type.")
    p.add_argument("--targets", nargs="+", default=["pairs", "questions"], choices=["pairs", "questions"], help="pairs: OmniPath pairs -> gene_interactions.Literature; questions: literature_plan curation questions -> literature_evidence.")
    p.add_argument("--overwrite", action="store_true", help="Re-run paper-qa for pairs / questions that already have a cached result.")
    return p


def main():
    args = build_parser().parse_args()
    load_env()
    info_dir = Path(args.info_dir)
    args.lit_dir = Path(args.lit_dir or info_dir.resolve().parent / LIT_FOLDER)
    args.literature_dir = Path(args.literature_dir or args.lit_dir / "Literature_search")
    args.evidence_dir = Path(args.evidence_dir or info_dir.resolve().parent / "Evidence_curation")
    if not args.literature_dir.exists():
        raise FileNotFoundError(f"Literature folder not found: {args.literature_dir} "
                                "(run 1.Search/1.1.Search_literature/run_literature_search.py first)")
    cfg = load_config(args.config)
    args.qa_params = cfg.get("qa", {})
    settings = qa.build_settings(cfg)

    paths = {}
    for p in dict.fromkeys(args.programs):
        extended = args.lit_dir / f"P{p}.json"
        paths[f"P{p}"] = extended if extended.exists() else info_dir / f"P{p}.json"
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No extended bundle for {len(missing)} program(s): {missing}")

    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    summary, total_ran = {}, 0
    for label, path in paths.items():
        bundle = json.loads(path.read_text())
        summary[label] = {}
        if "pairs" in args.targets and "OmniPath" in bundle.get("gene_interactions", {}):
            print(f"[{label}] curating {len(bundle['gene_interactions']['OmniPath']['pairs'])} pair(s)")
            block, n_ran = asyncio.run(curate_program(bundle, args, settings))
            bundle["gene_interactions"][SOURCE] = block
            summary[label]["pairs"] = block["n_by_status"]
            total_ran += n_ran
            print(f"  {label}: pairs {block['n_by_status']} ({n_ran} paper-qa run(s))")
        if "questions" in args.targets and bundle.get("literature_plan"):
            print(f"[{label}] answering curation questions")
            block, n_ran = asyncio.run(curate_questions(bundle, args, settings))
            bundle["literature_evidence"] = block
            summary[label]["questions"] = block["n_by_verdict"]
            total_ran += n_ran
            print(f"  {label}: questions {block['n_by_verdict']} ({n_ran} paper-qa run(s))")
        if not summary[label]:
            print(f"  [warn] {label}: no OmniPath pairs or literature_plan to curate")
        args.lit_dir.mkdir(parents=True, exist_ok=True)
        (args.lit_dir / f"{label}.json").write_text(json.dumps(bundle, indent=2))

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_dir": str(info_dir), "lit_dir": str(args.lit_dir), "literature_dir": str(args.literature_dir),
                   "config": str(args.config)},
        "qa": args.qa_params,
        "per_program": summary,
        "n_paperqa_runs": total_ran,
    }
    (args.evidence_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(paths)} program(s) -> {args.lit_dir} (gene_interactions.{SOURCE}, literature_evidence); "
          f"cache/meta -> {args.evidence_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
