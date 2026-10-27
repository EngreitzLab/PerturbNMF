#!/usr/bin/env python3
"""MyGene.info gene cards (official symbol, name, aliases, summary, GO biological process, GeneRIFs) for gene symbols.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, as_list, merge_jsonl, request_with_retry, require_str_list, run  # noqa: E402,F401

SOURCE = "MyGene"
MYGENE_URL = "https://mygene.info/v3/query"


# query
def query_mygene(genes, species="human", fields="symbol,name,alias,summary,go.BP.term,entrezgene,generif",
                 batch_size=1000):
    """Batched POST to MyGene.info, matching each gene against symbol and alias
    (so renamed genes such as KIAA1429 -> VIRMA still resolve). Returns the raw hits."""
    hits = []
    for i in range(0, len(genes), batch_size):
        data = {"q": ",".join(genes[i:i + batch_size]), "scopes": "symbol,alias",
                "species": species, "fields": fields}
        hits += request_with_retry("POST", MYGENE_URL, data=data).json()
    return hits


def pick_hits(hits):
    """One hit per query: exact symbol match first, then highest _score; notfound dropped."""
    best = {}
    for h in hits:
        if h.get("notfound"):
            continue
        q = h["query"]
        key = (h.get("symbol") == q, h.get("_score", 0))
        if q not in best or key > best[q][0]:
            best[q] = (key, h)
    return {q: h for q, (_, h) in best.items()}


def format_gene_info(hit, top_go_bp=10, top_generif=10):
    """Aliases and function (summary + GO biological process terms) of one hit, plus
    GeneRIF statements with the PMID of the paper supporting each."""
    go_bp = [t["term"] for t in as_list(hit.get("go", {}).get("BP")) if "term" in t]
    generif = [{"pmid": str(r["pubmed"]), "text": r.get("text")}
               for r in as_list(hit.get("generif")) if "pubmed" in r]
    return {
        "found": True,
        "symbol": hit.get("symbol"),
        "entrezgene": str(hit["entrezgene"]) if "entrezgene" in hit else None,
        "name": hit.get("name"),
        "alias": [str(a) for a in as_list(hit.get("alias"))],
        "summary": hit.get("summary"),
        "go_bp": list(dict.fromkeys(go_bp))[:top_go_bp],
        "generif": generif[:top_generif],
    }


def handle(req):
    genes = require_str_list(req, "genes", max_len=1000)
    hits = pick_hits(query_mygene(genes, req.get("species", "human")))
    info = {g: format_gene_info(h, int(req.get("top_go_bp", 10)), int(req.get("top_generif", 10))) for g, h in hits.items()}
    not_found = [g for g in genes if g not in info]
    out = {"source": MYGENE_URL, "n_found": len(info), "not_found": not_found}
    if req.get("write_to"):
        merge_jsonl(req["write_to"], "gene", {g: {SOURCE: v} for g, v in info.items()}
                    | {g: {SOURCE: {"found": False}} for g in not_found})
        out["written_to"] = req["write_to"]
    else:
        out["genes"] = info
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
