#!/usr/bin/env python3
"""UniProt reviewed entries (protein name, function, subcellular location, tissue specificity, pathway, PMIDs) for gene symbols.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, as_list, merge_jsonl, request_with_retry, require_str_list, run  # noqa: E402,F401

SOURCE = "UniProt"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = ("accession,gene_primary,gene_synonym,protein_name,cc_function,"
          "cc_subcellular_location,cc_tissue_specificity,cc_pathway")


# query
def query_uniprot(genes, taxid=9606, batch_size=100):
    """Reviewed (Swiss-Prot) entries whose gene name or synonym (gene_exact) matches a
    query gene, following the Link: rel="next" pagination."""
    entries = []
    for i in range(0, len(genes), batch_size):
        terms = " OR ".join(f"gene_exact:{g}" for g in genes[i:i + batch_size])
        url = UNIPROT_URL
        params = {"query": f"({terms}) AND organism_id:{taxid} AND reviewed:true",
                  "fields": FIELDS, "format": "json", "size": 500}
        while url:
            resp = request_with_retry("GET", url, params=params)
            entries += resp.json().get("results", [])
            url, params = resp.links.get("next", {}).get("url"), None  # next url carries the query
    return entries


def _gene_names(entry):
    """(primary names, synonyms) over all gene blocks of an entry."""
    primary, synonyms = [], []
    for g in entry.get("genes", []):
        if "geneName" in g:
            primary.append(g["geneName"]["value"])
        synonyms += [s["value"] for s in g.get("synonyms", [])]
    return primary, synonyms


def pick_entries(genes, entries):
    """One entry per query gene: primary gene name match first, then synonym match."""
    by_primary, by_synonym = {}, {}
    for e in entries:
        primary, synonyms = _gene_names(e)
        for n in primary:
            by_primary.setdefault(n, e)
        for n in synonyms:
            by_synonym.setdefault(n, e)
    return {g: by_primary.get(g) or by_synonym[g] for g in genes if g in by_primary or g in by_synonym}


def _comments(entry, ctype):
    return [c for c in entry.get("comments", []) if c.get("commentType") == ctype]


def _texts(entry, ctype):
    return " ".join(t["value"] for c in _comments(entry, ctype) for t in c.get("texts", [])) or None


def format_gene_info(entry):
    primary, synonyms = _gene_names(entry)
    function_pmids = [ev["id"] for c in _comments(entry, "FUNCTION") for t in c.get("texts", [])
                      for ev in t.get("evidences", []) if ev.get("source") == "PubMed"]
    locations = [loc["location"]["value"] for c in _comments(entry, "SUBCELLULAR LOCATION")
                 for loc in c.get("subcellularLocations", []) if "location" in loc]
    rec_name = entry.get("proteinDescription", {}).get("recommendedName", {})
    return {
        "found": True,
        "accession": entry.get("primaryAccession"),
        "symbol": primary[0] if primary else None,
        "alias": synonyms,
        "protein_name": rec_name.get("fullName", {}).get("value"),
        "function": _texts(entry, "FUNCTION"),
        "function_pmids": list(dict.fromkeys(function_pmids)),
        "subcellular_location": list(dict.fromkeys(locations)),
        "tissue_specificity": _texts(entry, "TISSUE SPECIFICITY"),
        "pathway": _texts(entry, "PATHWAY"),
    }


def handle(req):
    genes = require_str_list(req, "genes", max_len=1000)
    entries = pick_entries(genes, query_uniprot(genes, int(req.get("taxid", 9606))))
    info = {g: format_gene_info(e) for g, e in entries.items()}
    not_found = [g for g in genes if g not in info]
    out = {"source": UNIPROT_URL, "n_found": len(info), "not_found": not_found}
    if req.get("write_to"):
        merge_jsonl(req["write_to"], "gene", {g: {SOURCE: v} for g, v in info.items()}
                    | {g: {SOURCE: {"found": False}} for g in not_found})
        out["written_to"] = req["write_to"]
    else:
        out["genes"] = info
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
