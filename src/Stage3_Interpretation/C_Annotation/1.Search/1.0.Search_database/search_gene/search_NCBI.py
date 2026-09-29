import argparse
import os
import sys
import time
from datetime import datetime

from src import (default_out_dir, load_program_JSON, report_query,
                         request_with_retry, split_cached, write_outputs)

SOURCE = "NCBI"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_params():
    """tool/email/api_key identification, same env vars as search_claude_mediated/search_ncbi.py."""
    params = {"tool": "PerturbNMF-AGeneTic"}
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


def build_parser():
    p = argparse.ArgumentParser(description="Extend program bundles with NCBI Gene aliases, summary and GeneRIF PMIDs.")

    # IO
    p.add_argument("--bundle_dir", required=True, help="program_bundles folder written by Extract_program_information.py.")
    p.add_argument("--out_dir", default=None, help="Shared output directory (existing P<k>.json there are extended). Default: <bundle_dir>/../Gene_info_extended_PerturbNMF_Info.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3); only these bundles are read.")
    p.add_argument("--taxid", type=int, default=9606, help="NCBI taxonomy id (9606 = human).")

    # query
    p.add_argument("--top_generif", type=int, default=10, help="Max GeneRIF PMIDs kept per gene.")
    p.add_argument("--batch_size", type=int, default=200, help="Genes/ids per esearch/esummary request.")
    p.add_argument("--elink_batch_size", type=int, default=20, help="Gene ids per GeneRIF elink request (large batches get cut off by NCBI).")
    p.add_argument("--overwrite", action="store_true", help="Re-query genes that already have this source's info in --out_dir.")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = args.out_dir or default_out_dir(args.bundle_dir)

    bundles = load_program_JSON(args.bundle_dir, out_dir, args.programs)

    # one set of queries for the union of genes across programs, skipping genes that already have this source
    genes, cached, query = split_cached(bundles, SOURCE, args.overwrite)
    info = {}
    if query:
        print(f"[{SOURCE}] querying {len(query)} genes -> {NCBI_BASE_URL}")
        ids = search_gene_ids(query, args.taxid, args.batch_size)
        records = fetch_summaries(ids, args.batch_size)
        picked = pick_records(query, records)
        generif = fetch_generif_pmids(list(dict.fromkeys(picked.values())), args.elink_batch_size)
        info = {g: format_gene_info(uid, records[uid], generif, args.top_generif)
                for g, uid in picked.items()}
    renamed, not_found = report_query(SOURCE, query, info)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"bundle_dir": str(args.bundle_dir)},
        "source": NCBI_BASE_URL,
        "params": {k: getattr(args, k) for k in ("taxid", "top_generif", "batch_size", "elink_batch_size")},
        "programs": list(bundles),
        "n_genes": len(genes),
        "n_skipped": len(cached),
        "n_queried": len(query),
        "n_found": len(info),
        "renamed": renamed,
        "not_found": not_found,
    }
    write_outputs(out_dir, bundles, SOURCE, {**cached, **info}, meta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
