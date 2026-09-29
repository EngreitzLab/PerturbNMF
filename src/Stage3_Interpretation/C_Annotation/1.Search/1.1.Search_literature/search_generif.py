"""GeneRIF provenance: trace the papers behind each gene's NCBI GeneRIFs and summarize their context.

For every gene of a program (program_genes, distinctive_genes, regulators), the top-N PMIDs of
gene_info.NCBI.generif_pmids (from 1.0.Search_database/search_gene/search_NCBI.py) are fetched from
PubMed (abstract + ids). With --download_pdfs, open-access PDFs go to gene_pdfs/<GENE>/<PMID>.pdf and
their text is added to the abstract. One LLM call per gene extracts, per paper, the claim, evidence
type, species, cell line / tissue, condition and partner genes, plus a gene-level context summary:

    <out_dir>/P<k>.json                 1.0 bundle + gene_info.GeneRIF on every gene entry
    <out_dir>/gene_pdfs/<GENE>/*.pdf    open-access PDFs (--download_pdfs)
    <out_dir>/gene_cache/<GENE>.json    per-gene result, reused by other programs and re-runs
    <out_dir>/meta_GeneRIF.json

Env: ANTHROPIC_API_KEY, NCBI_EMAIL / NCBI_API_KEY, UNPAYWALL_EMAIL; read from AGeneTic/.env when present.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIT_DIR = HERE.parent   # 1.1.Search_literature: src / llm_summary_agent / pdf_download
sys.path[:0] = [str(HERE), str(LIT_DIR)]

from llm_summary_agent import Summarize_GeneRIF_Agent  # noqa: E402
from pdf_download import download_paper  # noqa: E402
from search_ncbi import NCBI_BASE_URL, _get, _ncbi_params, _parse_pubmed_xml  # noqa: E402
from src import (attach_gene_block, default_out_dir, describe_gene, gene_entries,  # noqa: E402
                 load_env, load_program_JSON, make_handler, write_bundles, write_meta)

SOURCE = "GeneRIF"
STOP_WORDS = {"cell", "cells", "line", "human", "mouse", "primary"}
_METHODS_RE = re.compile(r"\n\s*(materials?\s+and\s+methods|methods|experimental procedures)\s*\n", re.IGNORECASE)


# query
def fetch_records(pmids):
    """{pmid: PubMed record} via one efetch call."""
    if not pmids:
        return {}
    xml_text = _get(f"{NCBI_BASE_URL}/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **_ncbi_params()},
                    as_json=False)
    time.sleep(0.34)
    return {r["pmid"]: r for r in _parse_pubmed_xml(str(xml_text))}


def context_terms(cell_type, extra):
    words = re.findall(r"[A-Za-z0-9-]+", cell_type or "")
    terms = [w for w in words if len(w) > 3 and w.lower() not in STOP_WORDS] + list(extra or [])
    return [t.lower().rstrip("*") for t in dict.fromkeys(terms)]


def rank_records(records, terms):
    """Cell-type-matching papers first, then primary research over reviews; stable otherwise."""
    def score(r):
        text = f"{r.get('title') or ''} {r.get('abstract') or ''}".lower()
        return (-any(t in text for t in terms), r.get("study_type") == "review")
    return sorted(records, key=score)


def pdf_text(path, max_chars):
    """Methods-first excerpt of a PDF (the part that states cell lines, species and conditions)."""
    try:
        import fitz
        with fitz.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as e:  # a broken PDF must not stop the gene
        print(f"    [warn] could not read {path}: {e}")
        return None
    m = _METHODS_RE.search(text)
    start = m.start() if m else 0
    return text[start:start + max_chars].strip() or None


# per gene
def cache_key(pmids, cell_type, args):
    return {"pmids": pmids, "cell_type": cell_type, "model": args.model,
            "download_pdfs": args.download_pdfs, "pdf_chars": args.pdf_chars}


def summarize_gene(gene, entry, cell_type, terms, handler, out_dir, args):
    """(GeneRIF block, ran LLM?) for one gene."""
    stored = (entry.get("gene_info", {}).get("NCBI") or {}).get("generif_pmids") or []
    if not stored:
        return {"found": False, "reason": "no GeneRIF PMIDs in gene_info.NCBI"}, False

    records = fetch_records([str(p) for p in stored])
    ranked = rank_records([records[str(p)] for p in stored if str(p) in records], terms)[:args.top_n]
    pmids = [r["pmid"] for r in ranked]

    cache_file = out_dir / "gene_cache" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', gene)}.json"
    key = cache_key(pmids, cell_type, args)
    if cache_file.exists() and not args.overwrite:
        cached = json.loads(cache_file.read_text())
        if cached.get("key") == key:
            return cached["block"], False

    papers = []
    for r in ranked:
        paper = {k: r.get(k) for k in ("pmid", "title", "year", "journal", "doi", "pmcid", "study_type",
                                       "is_retracted", "abstract")}
        paper.update(pdf_path=None, pdf_status="not requested", fulltext=None)
        if args.download_pdfs:
            path, msg = download_paper(out_dir / "gene_pdfs" / gene, r["pmid"],
                                       pmid=r["pmid"], pmcid=r.get("pmcid"), doi=r.get("doi"))
            paper.update(pdf_path=path, pdf_status=msg)
            if path:
                paper["fulltext"] = pdf_text(path, args.pdf_chars)
        papers.append(paper)

    summary = Summarize_GeneRIF_Agent(gene, describe_gene(gene, entry), papers, cell_type, handler,
                                      effort=args.effort, max_tokens=args.max_tokens)
    extracted = {p["pmid"]: p for p in (summary or {}).get("papers", [])}
    for paper in papers:
        paper.update({k: v for k, v in extracted.get(paper["pmid"], {}).items() if k != "pmid"})
        paper["text_source"] = "abstract+pdf" if paper.pop("fulltext") else "abstract"
        paper.pop("abstract")
    block = {
        "found": True,
        "pmids_considered": [str(p) for p in stored],
        "papers": papers,
        "context_summary": (summary or {}).get("context_summary"),
        "evidence_in_cell_type": (summary or {}).get("evidence_in_cell_type"),
        "llm_error": summary is None,
        "cache_file": str(cache_file),
    }
    if summary is not None:   # cache only successful extractions
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"key": key, "block": block}, indent=2))
    return block, True


def build_parser():
    p = argparse.ArgumentParser(description="Summarize the GeneRIF papers of every program gene (claim, species, cell line, condition).")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database.")
    p.add_argument("--out_dir", default=None, help=f"Output folder. Default: <info_dir>/../{default_out_dir('x').name}.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--cell_type", default=None, help="Target cell type for context matching. Default: the bundle's cell_type.")
    p.add_argument("--context_terms", nargs="*", default=[], help="Extra words that mark a paper as cell-type matched when ranking (e.g. endothelial HUVEC HAEC).")

    # query
    p.add_argument("--top_n", type=int, default=5, help="GeneRIF papers summarized per gene, after ranking. The pool is gene_info.NCBI.generif_pmids, capped upstream by search_NCBI.py --top_generif.")
    p.add_argument("--download_pdfs", action="store_true", help="Download open-access PDFs into <out_dir>/gene_pdfs/<GENE>/ and give their text to the summary agent.")
    p.add_argument("--pdf_chars", type=int, default=15000, help="Max PDF characters (from the Methods section when found) given to the agent per paper.")

    # agent
    p.add_argument("--model", default="claude-sonnet-5", help="Claude model for the summary agent.")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="Effort level of the summary agent.")
    p.add_argument("--max_tokens", type=int, default=16000, help="Max output tokens per gene call.")
    p.add_argument("--overwrite", action="store_true", help="Ignore gene_cache/ and re-summarize every gene.")
    return p


def main():
    args = build_parser().parse_args()
    load_env()
    out_dir = Path(args.out_dir or default_out_dir(args.info_dir))
    bundles = load_program_JSON(args.info_dir, out_dir, args.programs)
    handler = make_handler(args.model)

    blocks, n_llm = {}, 0
    for label, bundle in bundles.items():
        cell_type = args.cell_type if args.cell_type is not None else bundle.get("cell_type", "")
        terms = context_terms(cell_type, args.context_terms)
        entries = gene_entries(bundle)
        print(f"[{label}] {len(entries)} genes; context terms: {terms}")
        for gene, entry in entries.items():
            if gene in blocks:   # same gene in an earlier program of this run
                continue
            block, ran = summarize_gene(gene, entry, cell_type, terms, handler, out_dir, args)
            blocks[gene] = block
            n_llm += ran
            if block["found"]:
                n_pdf = sum(bool(p["pdf_path"]) for p in block["papers"])
                print(f"  {gene}: {len(block['papers'])} papers, {n_pdf} PDFs, "
                      f"in cell type={block['evidence_in_cell_type']}"
                      f"{' (cached)' if not ran else ''}{' [LLM error]' if block['llm_error'] else ''}")
            else:
                print(f"  {gene}: {block['reason']}")
        bundles[label] = attach_gene_block(bundle, SOURCE, blocks)

    write_bundles(out_dir, bundles)
    found = [b for b in blocks.values() if b["found"]]
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_dir": str(args.info_dir)},
        "params": {k: getattr(args, k) for k in ("cell_type", "context_terms", "top_n", "download_pdfs",
                                                  "pdf_chars", "model", "effort", "max_tokens")},
        "programs": list(bundles),
        "n_genes": len(blocks),
        "n_with_generif": len(found),
        "n_llm_calls": n_llm,
        "n_llm_errors": sum(b["llm_error"] for b in found),
        "n_papers": sum(len(b["papers"]) for b in found),
        "n_pdfs": sum(bool(p["pdf_path"]) for b in found for p in b["papers"]),
        "n_evidence_in_cell_type": sum(bool(b["evidence_in_cell_type"]) for b in found),
    }
    write_meta(out_dir, SOURCE, meta)
    print(f"[done] {len(bundles)} bundle(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
