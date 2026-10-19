"""Gene function categorization: 3 open-access PubMed PDFs per gene, read and categorized by function.

For every gene of a program (--gene_lists: program_genes, distinctive_genes, regulators), PubMed is searched for papers
on the gene's function ('(SYMBOL[tiab] OR alias[tiab]) AND (function[tiab] OR role[tiab] ...)', relevance
order, --pool_size hits). Retracted papers are dropped; papers with the symbol or an alias in the title come first, then
cell-type-matching papers, then primary research over reviews. Open-access PDFs are downloaded down that
ranking until --n_pdfs succeed (gene_function_pdfs/<GENE>/<PMID>.pdf); when none can be downloaded the top
--n_pdfs abstracts are used instead. One LLM call per gene extracts, per paper, the claim, functional role,
evidence type, species, cell line / tissue and condition, and assigns the gene 1-3 function categories
(llm_summary_agent.FunctionCategory) with a short function summary:

    <out_dir>/P<k>.json                          1.0 bundle + gene_info.Gene_function_categorization
    <out_dir>/gene_function_pdfs/<GENE>/*.pdf    open-access PDFs
    <out_dir>/gene_function_cache/<GENE>.json    per-gene result, reused by other programs and re-runs
    <out_dir>/meta_Gene_function_categorization.json

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
sys.path[:0] = [str(HERE), str(LIT_DIR), str(LIT_DIR.parent / "api")]

from budget import add_budget_args, attach  # noqa: E402
from llm_summary_agent import Summarize_GeneFunction_Agent  # noqa: E402
from pdf_download import download_paper  # noqa: E402
from search_generif import context_terms, fetch_records, rank_records  # noqa: E402
from search_ncbi import NCBI_BASE_URL, _get, _ncbi_params, normalize_pmid  # noqa: E402
from src import (attach_gene_block, default_out_dir, describe_gene, gene_aliases,  # noqa: E402
                 gene_entries, load_env, load_program_JSON, make_handler, write_bundles, write_meta)

SOURCE = "Gene_function_categorization"
GENE_LIST_CHOICES = ["program_genes", "distinctive_genes", "regulators"]
FUNCTION_TERMS = ["function", "role", "regulat*", "mechanism"]


# query
def query_names(gene, entry, max_aliases):
    """[gene, up to max_aliases aliases of 3+ characters]."""
    aliases = [a for a in gene_aliases(gene, entry)[1:] if len(a) >= 3 and a.upper() != gene.upper()]
    return list(dict.fromkeys([gene] + aliases[:max_aliases]))


def build_query(names, args):
    """'(SYMBOL[tiab] OR alias[tiab] ...) AND (function[tiab] OR ...)'."""
    name_q = " OR ".join(f'"{n}"[tiab]' for n in names)
    func_q = " OR ".join(f"{t}[tiab]" for t in args.function_terms)
    return f"({name_q}) AND ({func_q})" if func_q else f"({name_q})"


def search_pmids(query, pool_size):
    """PMIDs of a PubMed esearch, in relevance order."""
    payload = _get(f"{NCBI_BASE_URL}/esearch.fcgi",
                   params={"db": "pubmed", "term": query, "retmax": pool_size, "retmode": "json",
                           "sort": "relevance", **_ncbi_params()})
    time.sleep(0.34)
    ids = (payload.get("esearchresult", {}) if isinstance(payload, dict) else {}).get("idlist", [])
    return [p for p in map(normalize_pmid, ids) if p]


def rank_gene_records(records, names, terms):
    """Not retracted; symbol / alias in title first, then search_generif.rank_records order (cell type, non-review)."""
    in_title = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)
    kept = rank_records([r for r in records if not r.get("is_retracted")], terms)
    return sorted(kept, key=lambda r: not in_title.search(r.get("title") or ""))


def pdf_text(path, max_chars):
    """First max_chars of a PDF's text (abstract, introduction, results: where the function is stated)."""
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception as e:  # a broken PDF must not stop the gene
        print(f"    [warn] could not read {path}: {e}")
        return None
    return text[:max_chars].strip() or None


# per gene
def selected_entries(bundle, gene_lists):
    """{gene: entry} restricted to the genes of gene_lists, in src.gene_entries order."""
    keep = {e["gene"] for k in gene_lists if k != "regulators" for e in bundle.get(k, []) if isinstance(e, dict)}
    if "regulators" in gene_lists:
        keep |= {r["gene"] for regs in bundle.get("perturbation_regulators", {}).values() for r in regs}
    return {g: e for g, e in gene_entries(bundle).items() if g in keep}


def cache_key(query, cell_type, args):
    """Not the PMIDs: PubMed's relevance order changes between identical searches."""
    return {"query": query, "pool_size": args.pool_size, "cell_type": cell_type, "model": args.model,
            "n_pdfs": args.n_pdfs, "pdf_chars": args.pdf_chars}


def paper_record(r):
    return {k: r.get(k) for k in ("pmid", "title", "year", "journal", "doi", "pmcid", "study_type", "abstract")}


def collect_papers(gene, ranked, out_dir, args):
    """Download down the ranking until n_pdfs PDFs; no PDF at all -> the top n_pdfs abstracts."""
    papers, tried = [], []
    for r in ranked:
        if len(papers) >= args.n_pdfs:
            break
        path, msg = download_paper(out_dir / "gene_function_pdfs" / gene, r["pmid"],
                                   pmid=r["pmid"], pmcid=r.get("pmcid"), doi=r.get("doi"))
        tried.append({"pmid": r["pmid"], "pdf_status": msg})
        if path:
            papers.append({**paper_record(r), "pdf_path": path, "pdf_status": msg,
                           "fulltext": pdf_text(path, args.pdf_chars)})
    if not papers:
        papers = [{**paper_record(r), "pdf_path": None, "pdf_status": "no open-access PDF in pool",
                   "fulltext": None} for r in ranked[:args.n_pdfs]]
    return papers, tried


def categorize_gene(gene, entry, cell_type, terms, handler, out_dir, args):
    """(Gene_function_categorization block, ran LLM?) for one gene."""
    names = query_names(gene, entry, args.max_aliases)
    query = build_query(names, args)
    cache_file = out_dir / "gene_function_cache" / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', gene)}.json"
    key = cache_key(query, cell_type, args)
    if cache_file.exists() and not args.overwrite:
        cached = json.loads(cache_file.read_text())
        if cached.get("key") == key:
            return cached["block"], False

    try:
        pmids = search_pmids(query, args.pool_size)
    except Exception as e:  # one failing search must not stop the run
        return {"found": False, "query": query, "reason": f"PubMed search failed: {e}"}, False
    if not pmids:
        return {"found": False, "query": query, "reason": "no PubMed hits"}, False

    records = fetch_records(pmids)
    ranked = rank_gene_records([records[p] for p in pmids if p in records], names, terms)
    papers, tried = collect_papers(gene, ranked, out_dir, args)

    summary = Summarize_GeneFunction_Agent(gene, describe_gene(gene, entry), papers, cell_type, handler,
                                           effort=args.effort, max_tokens=args.max_tokens)
    extracted = {p["pmid"]: p for p in (summary or {}).get("papers", [])}
    for paper in papers:
        paper.update({k: v for k, v in extracted.get(paper["pmid"], {}).items() if k != "pmid"})
        paper["text_source"] = "abstract+pdf" if paper.pop("fulltext") else "abstract"
        paper.pop("abstract")
    block = {
        "found": True,
        "query": query,
        "pmids_considered": pmids,
        "downloads_tried": tried,
        "papers": papers,
        "categories": (summary or {}).get("categories", []),
        "primary_category": (summary or {}).get("primary_category"),
        "function_summary": (summary or {}).get("function_summary"),
        "evidence_in_cell_type": (summary or {}).get("evidence_in_cell_type"),
        "llm_error": summary is None,
        "summary_model": getattr(handler, "model_label", args.model),
        "cache_file": str(cache_file),
    }
    if summary is not None:   # cache only successful extractions
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"key": key, "block": block}, indent=2))
    return block, True


def build_parser():
    p = argparse.ArgumentParser(description="Gene function categorization: download open-access PubMed PDFs for every program gene and have an LLM categorize its function.")

    # IO
    p.add_argument("--info_dir", required=True, help="Gene_info_extended_PerturbNMF_Info folder from 1.0.Search_database.")
    p.add_argument("--out_dir", default=None, help=f"Output folder. Default: <info_dir>/../{default_out_dir('x').name}.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3).")
    p.add_argument("--cell_type", default=None, help="Target cell type for context matching. Default: the bundle's cell_type.")
    p.add_argument("--gene_lists", nargs="+", default=GENE_LIST_CHOICES, choices=GENE_LIST_CHOICES, help="Which genes get papers, space separated (e.g. program_genes for the top-loading genes only).")
    p.add_argument("--context_terms", nargs="*", default=[], help="Extra words that mark a paper as cell-type matched when ranking (e.g. endothelial HUVEC HAEC).")

    # query
    p.add_argument("--n_pdfs", type=int, default=3, help="PDFs downloaded and summarized per gene (fewer when the pool has fewer open-access papers).")
    p.add_argument("--pool_size", type=int, default=20, help="PubMed hits per gene that the downloads are tried on, in ranked order.")
    p.add_argument("--max_aliases", type=int, default=3, help="Max gene aliases (3+ characters) OR'd into the PubMed query.")
    p.add_argument("--function_terms", nargs="*", default=FUNCTION_TERMS, help="Title/abstract words, OR'd and ANDed with the gene names; pass with no words to search the gene names alone.")
    p.add_argument("--pdf_chars", type=int, default=15000, help="Max PDF characters (from the start of the paper) given to the agent per paper.")

    # agent
    p.add_argument("--model", default="claude-sonnet-5", help="Claude model for the summary agent.")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"], help="Effort level of the summary agent.")
    p.add_argument("--max_tokens", type=int, default=16000, help="Max output tokens per gene call.")
    p.add_argument("--overwrite", action="store_true", help="Ignore gene_function_cache/ and re-categorize every gene.")
    add_budget_args(p, reserve_default=1.0)
    p.add_argument("--llm_backend", default="api", choices=["api", "session"], help="api = Anthropic API (ANTHROPIC_API_KEY); session = each gene's summary prompt is written to --session_dir and the answer written there by a Claude Code session is read on a second run (api/session_llm.py). The cache key keeps --model, so cached genes are reused either way.")
    p.add_argument("--session_dir", default=None, help="Request / answer folder for --llm_backend session.")
    p.add_argument("--session_model", default="claude-code-session", help="Model name recorded (summary_model) for session answers.")
    return p


def main():
    args = build_parser().parse_args()
    load_env()
    out_dir = Path(args.out_dir or default_out_dir(args.info_dir))
    bundles = load_program_JSON(args.info_dir, out_dir, args.programs)
    if args.llm_backend == "session":
        from session_llm import session_llm
        handler = session_llm(args.session_dir, "gene_function", args.session_model)
    else:
        handler = make_handler(args.model)

    blocks, n_llm = {}, 0
    for label, bundle in bundles.items():
        cell_type = args.cell_type if args.cell_type is not None else bundle.get("cell_type", "")
        terms = context_terms(cell_type, args.context_terms)
        entries = selected_entries(bundle, args.gene_lists)
        print(f"[{label}] {len(entries)} genes; context terms: {terms}")
        attach(handler, args, label, "gene_function")   # this program's spend cap (no-op without --budget_dir)
        for gene, entry in entries.items():
            if gene in blocks and not blocks[gene].get("llm_error"):   # same gene in an earlier program of this run
                continue
            handler.key = re.sub(r"[^A-Za-z0-9._-]+", "_", gene)   # names the request with --llm_backend session
            block, ran = categorize_gene(gene, entry, cell_type, terms, handler, out_dir, args)
            blocks[gene] = block
            n_llm += ran
            if block["found"]:
                n_pdf = sum(bool(p["pdf_path"]) for p in block["papers"])
                print(f"  {gene}: {len(block['papers'])} papers, {n_pdf} PDFs, "
                      f"categories={block['categories']}"
                      f"{' (cached)' if not ran else ''}{' [LLM error]' if block['llm_error'] else ''}")
            else:
                print(f"  {gene}: {block['reason']}")
        bundles[label] = attach_gene_block(bundle, SOURCE, blocks)
        write_bundles(out_dir, {label: bundles[label]})   # after every program, so an interrupted run keeps its work

    found = [b for b in blocks.values() if b["found"]]
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"info_dir": str(args.info_dir)},
        "params": {k: getattr(args, k) for k in ("cell_type", "gene_lists", "context_terms", "n_pdfs", "pool_size", "max_aliases",
                                                  "function_terms", "pdf_chars", "model", "effort", "max_tokens")},
        "programs": list(bundles),
        "n_genes": len(blocks),
        "n_with_papers": len(found),
        "n_llm_calls": n_llm,
        "n_llm_errors": sum(b["llm_error"] for b in found),
        "n_papers": sum(len(b["papers"]) for b in found),
        "n_pdfs": sum(bool(p["pdf_path"]) for b in found for p in b["papers"]),
        "n_genes_with_n_pdfs": sum(sum(bool(p["pdf_path"]) for p in b["papers"]) >= args.n_pdfs for b in found),
        "n_evidence_in_cell_type": sum(bool(b["evidence_in_cell_type"]) for b in found),
    }
    write_meta(out_dir, SOURCE, meta)
    print(f"[done] {len(bundles)} bundle(s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
