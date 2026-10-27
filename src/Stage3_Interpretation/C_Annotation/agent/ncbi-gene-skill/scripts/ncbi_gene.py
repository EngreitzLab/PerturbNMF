#!/usr/bin/env python3
"""NCBI Gene records (official symbol, description, aliases, location, summary, GeneRIF PMIDs) for gene symbols.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, as_list, merge_jsonl, request_with_retry, require_str_list, run  # noqa: E402,F401

SOURCE = "NCBI"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_params():
    """tool/email/api_key identification, same env vars as search_claude_mediated/search_ncbi.py."""
    params = {"tool": "PerturbNMF-annotation"}
    email = os.getenv("PUBMED_EMAIL") or os.getenv("NCBI_EMAIL")
    if email:
        params["email"] = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def _eutils(endpoint, data):
    """POST to an E-utility, pausing to stay under NCBI's rate limit (3/s, 10/s with a key)."""
    resp = request_with_retry("POST", f"{NCBI_BASE_URL}/{endpoint}.fcgi",
                              data={**_ncbi_params(), **data, "retmode": "json"})
    time.sleep(0.1 if os.getenv("NCBI_API_KEY") else 0.34)
    return resp.json()


# query
def search_gene_ids(genes, taxid=9606, batch_size=200):
    """NCBI Gene ids whose symbol or alias ([sym]) matches any query gene."""
    ids = []
    for i in range(0, len(genes), batch_size):
        terms = " OR ".join(f"{g}[sym]" for g in genes[i:i + batch_size])
        res = _eutils("esearch", {"db": "gene", "term": f"({terms}) AND {taxid}[taxid] AND alive[prop]",
                                  "retmax": 10000})
        ids += res["esearchresult"]["idlist"]
    return list(dict.fromkeys(ids))


def fetch_summaries(ids, batch_size=200):
    """{gene_id: esummary record}."""
    records = {}
    for i in range(0, len(ids), batch_size):
        res = _eutils("esummary", {"db": "gene", "id": ",".join(ids[i:i + batch_size])})["result"]
        records.update({uid: res[uid] for uid in res["uids"]})
    return records


def fetch_generif_pmids(ids, batch_size=20):
    """{gene_id: [GeneRIF PMIDs]} via elink gene_pubmed_rif (one linkset per id). Kept to small
    batches: well-studied genes link thousands of PMIDs and NCBI cuts off large responses."""
    pmids = {}
    for i in range(0, len(ids), batch_size):
        data = {"dbfrom": "gene", "db": "pubmed", "linkname": "gene_pubmed_rif",
                "id": ids[i:i + batch_size]}  # repeated id= gives one linkset per gene
        for ls in _eutils("elink", data).get("linksets", []):
            links = [l for db in ls.get("linksetdbs", []) for l in db.get("links", [])]
            pmids[str(ls["ids"][0])] = [str(l) for l in links]
    return pmids


def _split(value, sep):
    return [s.strip() for s in str(value or "").split(sep) if s.strip()]


def pick_records(genes, records):
    """One record per query gene: official symbol match first, then alias match. The NCBI
    'name' can differ from the HGNC 'nomenclaturesymbol' (ND6 vs MT-ND6), so both count."""
    by_symbol, by_alias = {}, {}
    for uid, r in records.items():
        for s in (r.get("name"), r.get("nomenclaturesymbol")):
            if s:
                by_symbol.setdefault(s, uid)
        for a in _split(r.get("otheraliases"), ","):
            by_alias.setdefault(a, uid)
    picked = {}
    for g in genes:
        uid = by_symbol.get(g) or by_alias.get(g)
        if uid:
            picked[g] = uid
    return picked


def format_gene_info(uid, record, generif_pmids, top_generif=10):
    return {
        "found": True,
        "gene_id": uid,
        "symbol": record.get("nomenclaturesymbol") or record.get("name"),
        "name": record.get("nomenclaturename") or record.get("description"),
        "alias": _split(record.get("otheraliases"), ","),
        "other_designations": _split(record.get("otherdesignations"), "|"),
        "summary": record.get("summary") or None,
        "map_location": record.get("maplocation") or None,
        "generif_pmids": generif_pmids.get(uid, [])[:top_generif],
    }


def handle(req):
    genes = require_str_list(req, "genes", max_len=1000)
    taxid = int(req.get("taxid", 9606))
    ids = search_gene_ids(genes, taxid)
    records = fetch_summaries(ids)
    picked = pick_records(genes, records)
    generif = fetch_generif_pmids(sorted(set(picked.values()))) if req.get("generif", True) else {}
    info = {g: format_gene_info(uid, records[uid], generif, int(req.get("top_generif", 10)))
            for g, uid in picked.items()}
    not_found = [g for g in genes if g not in info]
    out = {"source": NCBI_BASE_URL, "n_found": len(info), "not_found": not_found}
    if req.get("write_to"):
        merge_jsonl(req["write_to"], "gene", {g: {SOURCE: v} for g, v in info.items()}
                    | {g: {SOURCE: {"found": False}} for g in not_found})
        out["written_to"] = req["write_to"]
    else:
        out["genes"] = info
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
