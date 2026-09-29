import argparse
import sys
from datetime import datetime

from src import (as_list, default_out_dir, load_program_JSON, report_query,
                         request_with_retry, split_cached, write_outputs)

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


def build_parser():
    p = argparse.ArgumentParser(description="Extend program bundles with MyGene.info aliases and gene function.")

    # IO
    p.add_argument("--bundle_dir", required=True, help="program_bundles folder written by Extract_program_information.py.")
    p.add_argument("--out_dir", default=None, help="Shared output directory (existing P<k>.json there are extended). Default: <bundle_dir>/../Gene_info_extended_PerturbNMF_Info.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3); only these bundles are read.")
    p.add_argument("--species", default="human", help="MyGene species (name or taxid).")

    # query
    p.add_argument("--fields", default="symbol,name,alias,summary,go.BP.term,entrezgene,generif", help="MyGene fields to request.")
    p.add_argument("--top_go_bp", type=int, default=10, help="Max GO biological process terms kept per gene.")
    p.add_argument("--top_generif", type=int, default=10, help="Max GeneRIF statements (PMID + text) kept per gene; needs 'generif' in --fields.")
    p.add_argument("--batch_size", type=int, default=1000, help="Genes per MyGene request (API max 1000).")
    p.add_argument("--overwrite", action="store_true", help="Re-query genes that already have this source's info in --out_dir.")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = args.out_dir or default_out_dir(args.bundle_dir)

    bundles = load_program_JSON(args.bundle_dir, out_dir, args.programs)

    # one query for the union of genes across programs, skipping genes that already have this source
    genes, cached, query = split_cached(bundles, SOURCE, args.overwrite)
    info = {}
    if query:
        print(f"[{SOURCE}] querying {len(query)} genes -> {MYGENE_URL}")
        hits = pick_hits(query_mygene(query, args.species, args.fields, args.batch_size))
        info = {g: format_gene_info(h, args.top_go_bp, args.top_generif) for g, h in hits.items()}
    renamed, not_found = report_query(SOURCE, query, info)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"bundle_dir": str(args.bundle_dir)},
        "source": MYGENE_URL,
        "params": {k: getattr(args, k) for k in ("species", "fields", "top_go_bp", "top_generif", "batch_size")},
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
