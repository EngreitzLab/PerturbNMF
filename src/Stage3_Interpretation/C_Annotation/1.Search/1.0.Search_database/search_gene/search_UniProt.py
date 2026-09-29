import argparse
import sys
from datetime import datetime

from src import (default_out_dir, load_program_JSON, report_query,
                         request_with_retry, split_cached, write_outputs)

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


def build_parser():
    p = argparse.ArgumentParser(description="Extend program bundles with UniProtKB protein function, location and tissue specificity.")

    # IO
    p.add_argument("--bundle_dir", required=True, help="program_bundles folder written by Extract_program_information.py.")
    p.add_argument("--out_dir", default=None, help="Shared output directory (existing P<k>.json there are extended). Default: <bundle_dir>/../Gene_info_extended_PerturbNMF_Info.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3); only these bundles are read.")
    p.add_argument("--taxid", type=int, default=9606, help="NCBI taxonomy id (9606 = human).")

    # query
    p.add_argument("--batch_size", type=int, default=100, help="Genes per UniProt query.")
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
        print(f"[{SOURCE}] querying {len(query)} genes -> {UNIPROT_URL}")
        picked = pick_entries(query, query_uniprot(query, args.taxid, args.batch_size))
        info = {g: format_gene_info(e) for g, e in picked.items()}
    renamed, not_found = report_query(SOURCE, query, info)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"bundle_dir": str(args.bundle_dir)},
        "source": UNIPROT_URL,
        "params": {"taxid": args.taxid, "batch_size": args.batch_size, "fields": FIELDS,
                   "reviewed_only": True},
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
