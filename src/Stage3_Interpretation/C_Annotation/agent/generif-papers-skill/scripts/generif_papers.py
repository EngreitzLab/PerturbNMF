#!/usr/bin/env python3
"""A gene's GeneRIF papers (NCBI's curated gene-function references): the GeneRIF PMIDs from the request or from
the gene's NCBI block in gene_info.jsonl, ranked (screen's cell type first, primary research over reviews), with
open-access PDFs.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.pdf_download import download_paper  # noqa: E402
from _lib.pubmed_core import context_terms, fetch_records, rank_records, search_pmids  # noqa: E402
from _lib.skill_lib import SkillError, append_jsonl, read_jsonl, require_str, run  # noqa: E402

SKILL = "generif-papers-skill"


def collect(gene, ranked, req, skill):
    """Papers of one gene, with up to n_pdfs open-access PDFs downloaded into pdf_dir; appended to write_to."""
    n_pdfs = int(req.get("n_pdfs", 0))
    pdf_dir = Path(req["pdf_dir"]) / gene if req.get("pdf_dir") else None
    papers, n_got = [], 0
    for r in ranked:
        paper = {
            k: r.get(k)
            for k in (
                "pmid",
                "doi",
                "pmcid",
                "title",
                "year",
                "journal",
                "study_type",
                "abstract",
            )
        }
        paper["pdf_path"], paper["pdf_status"] = None, "not requested"
        if pdf_dir and n_got < n_pdfs:
            path, msg = download_paper(
                pdf_dir,
                r["pmid"],
                pmid=r["pmid"],
                pmcid=r.get("pmcid"),
                doi=r.get("doi"),
            )
            paper["pdf_path"], paper["pdf_status"] = (
                (str(Path(path).resolve()) if path else None),
                msg,
            )
            n_got += bool(path)
        papers.append(paper)
    out = {"gene": gene, "n_papers": len(papers), "n_pdfs": n_got}
    if req.get("write_to"):
        rows = [
            {
                "source": skill,
                "genes": [gene],
                "status": "candidate",
                "finding": None,
                "stance": None,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}",
                **p,
            }
            for p in papers
        ]
        added, dup = append_jsonl(
            req["write_to"],
            rows,
            dedupe=lambda r: ("pmid", r.get("pmid") or r.get("title")),
        )
        out.update(
            {"written_to": req["write_to"], "added": added, "already_present": dup}
        )
    n = int(req.get("abstract_chars", 400))
    out["papers"] = [
        {
            **{k: v for k, v in p.items() if k != "abstract"},
            "abstract": (p["abstract"] or "")[:n] or None,
        }
        for p in papers
    ]
    return out


def handle(req):
    gene = require_str(req, "gene")
    pmids = [str(p) for p in req.get("pmids") or []]
    if not pmids and req.get("gene_info"):
        row = next(
            (r for r in read_jsonl(req["gene_info"]) if r.get("gene") == gene), {}
        )
        pmids = [str(p) for p in (row.get("NCBI") or {}).get("generif_pmids") or []]
    if not pmids:
        raise SkillError(
            "not_found",
            f"no GeneRIF PMIDs for {gene}: pass `pmids`, or run ncbi-gene-skill into gene_info first",
        )
    records = fetch_records(pmids)
    ranked = rank_records(
        [records[p] for p in pmids if p in records],
        context_terms(req.get("cell_type"), req.get("context_terms")),
    )
    return collect(gene, ranked[: int(req.get("n_papers", 5))], req, SKILL)


if __name__ == "__main__":
    sys.exit(run(handle))
