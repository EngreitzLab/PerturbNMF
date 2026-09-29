import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "search_gene"))
from src import as_list, default_out_dir, gene_name, load_program_JSON, request_with_retry  # noqa: E402

SOURCE = "OmniPath"
OMNIPATH_URL = "https://omnipathdb.org/interactions"
ALL_DATASETS = ("omnipath", "pathwayextra", "kinaseextra", "ligrecextra", "collectri", "dorothea",
                "tf_target", "mirnatarget", "lncrna_mrna", "tf_mirna", "small_molecule")
FIELDS = "sources,references,curation_effort,type,datasets,extra_attrs,dorothea_level"
GENE_INFO_SOURCES = ("MyGene", "NCBI", "UniProt")
# stored in params so results written in an older layout are re-queried, not skipped
PAIRS_FORMAT = "regulator_gene_v3"
# gene_category in the extract step's regulator_gene block -> query category used in the pairs
CATEGORY_NAMES = {"program_gene": "program_gene-regulator", "distinctive_gene": "distinctive_gene-regulator"}


# loaders
def select_from_pair_list(bundle):
    """Query sets from the bundle's regulator_gene block (Extract_program_information.py):
    ({category: [gene entries]}, {regulator: {cond: stats}}, {regulator: entry}, {(gene, regulator)})."""
    entries = {gene_name(e): e for key in ("program_genes", "distinctive_genes") for e in bundle[key]}
    entries.update({gene_name(r): r for regs in bundle.get("perturbation_regulators", {}).values() for r in regs})
    gene_sets, regulators, reg_entries, listed = {}, {}, {}, set()
    for p in bundle["regulator_gene"]["pairs"].values():
        gene, reg = p["gene"], p["regulator"]
        cat = CATEGORY_NAMES[p["gene_category"]]
        members = gene_sets.setdefault(cat, [])
        if all(gene_name(e) != gene for e in members):
            members.append(entries.get(gene, gene))
        regulators.setdefault(reg, p["regulator_stats"])
        reg_entries.setdefault(reg, entries.get(reg, {"gene": reg}))
        listed.add((gene, reg))
    return gene_sets, regulators, reg_entries, listed


def select_sets(bundle, top_gene=15, top_regulator=6, include_unique=False, top_unique_gene=8):
    """Query sets of one program: {category: [gene entries]}, {regulator: {cond: stats}},
    {regulator: entry} and the listed (gene, regulator) pairs (None = every combination).

    Bundles with a regulator_gene block use that pair list. Older bundles fall back to the flags:
    the first N entries are the top N (the extract step ranks them); regulators take the first
    top_regulator per condition, merged across conditions."""
    if bundle.get("regulator_gene"):
        return select_from_pair_list(bundle)
    gene_sets = {"program_gene-regulator": bundle["program_genes"][:top_gene]}
    if include_unique:
        gene_sets["distinctive_gene-regulator"] = bundle["distinctive_genes"][:top_unique_gene]
    regulators, reg_entries = {}, {}
    for cond, regs in bundle.get("perturbation_regulators", {}).items():
        for r in regs[:top_regulator]:
            g = gene_name(r)
            regulators.setdefault(g, {})[cond] = {"log2fc": r.get("log2fc"), "adj_pval": r.get("adj_pval")}
            reg_entries.setdefault(g, r)
    return gene_sets, regulators, reg_entries, None


def name_candidates(entry, use_aliases=True):
    """Names to try for one gene: the bundle name, then symbols/aliases/UniProt accession
    from the gene_info blocks added by the search_gene scripts."""
    names = [gene_name(entry)]
    if use_aliases and isinstance(entry, dict):
        for src in GENE_INFO_SOURCES:
            info = entry.get("gene_info", {}).get(src, {})
            if not info.get("found"):
                continue
            names += [info.get("symbol"), info.get("accession")] + as_list(info.get("alias"))
    return list(dict.fromkeys(str(n) for n in names if n))


def canonical_id(entry):
    """Stable id of a gene (UniProt accession, then NCBI/MyGene gene id, then its name), so
    an old and a new name of the same gene (MESDC1 / TLNRD1) are not paired with each other."""
    info = entry.get("gene_info", {}) if isinstance(entry, dict) else {}
    for src, key in (("UniProt", "accession"), ("NCBI", "gene_id"), ("MyGene", "entrezgene")):
        if info.get(src, {}).get(key):
            return f"{src}:{info[src][key]}"
    return gene_name(entry)


def build_candidate_map(entries, use_aliases=True):
    """{candidate name: gene}. A candidate that is the bundle name of a queried gene belongs
    to that gene; any other candidate claimed by two genes is dropped as ambiguous."""
    genes = {gene_name(e) for e in entries.values()}
    claims = {}
    for g, e in entries.items():
        for n in name_candidates(e, use_aliases):
            claims.setdefault(n, set()).add(g)
    cand_map, dropped = {}, {}
    for n, owners in claims.items():
        if n in genes:
            cand_map[n] = n
        elif len(owners) == 1:
            cand_map[n] = next(iter(owners))
        else:
            dropped[n] = sorted(owners)
    return cand_map, dropped


# query
def query_omnipath(sources, targets, datasets, taxid=9606):
    """Interactions with source in `sources` AND target in `targets` (source_target=AND;
    without it OmniPath ORs the two lists). Unknown names are ignored by the server."""
    if not sources or not targets:
        return []
    params = {"sources": ",".join(sources), "targets": ",".join(targets),
              "source_target": "AND", "genesymbols": "yes", "organisms": taxid,
              "datasets": ",".join(datasets), "fields": FIELDS, "format": "json"}
    return request_with_retry("GET", OMNIPATH_URL, params=params).json()


def fetch_records(gene_cands, reg_cands, datasets, taxid=9606):
    """Records in both directions (genes -> regulators, regulators -> genes), deduped."""
    records = {}
    for src, tgt in ((gene_cands, reg_cands), (reg_cands, gene_cands)):
        for r in query_omnipath(src, tgt, datasets, taxid):
            records.setdefault((r["source"], r["target"], r.get("type")), r)
    return list(records.values())


# pairs
def _resolve(record, side, cand_map):
    """(gene, name OmniPath matched) for one side of a record: gene symbol, then UniProt accession."""
    for n in (record.get(f"{side}_genesymbol"), record.get(side)):
        if n in cand_map:
            return cand_map[n], n
    return None, None


def _sign(stim, inhib):
    return {(True, True): "both", (True, False): "activation",
            (False, True): "inhibition"}.get((bool(stim), bool(inhib)), "unknown")


def _pmids(references):
    """Unique PMIDs from OmniPath's 'Resource:PMID;Resource:PMID' string."""
    refs = [r.split(":", 1)[-1] for r in str(references or "").split(";") if r]
    return list(dict.fromkeys(p for p in refs if p.isdigit()))


def format_interaction(record, direction, max_pmids=20):
    pmids = _pmids(record.get("references"))
    mechanisms = [str(m) for k, v in (record.get("extra_attrs") or {}).items()
                  if k.endswith("_mechanism") for m in as_list(v)]
    return {
        "source": record.get("source_genesymbol"),
        "target": record.get("target_genesymbol"),
        "direction": direction,
        "sign": _sign(record.get("is_stimulation"), record.get("is_inhibition")),
        "consensus_sign": _sign(record.get("consensus_stimulation"), record.get("consensus_inhibition")),
        "type": record.get("type"),
        "datasets": [d for d in ALL_DATASETS if record.get(d) is True],
        "resources": as_list(record.get("sources")),
        "curation_effort": record.get("curation_effort"),
        "n_references": len(pmids),
        "pmids": pmids[:max_pmids],
        "mechanisms": list(dict.fromkeys(mechanisms)),
        "dorothea_level": as_list(record.get("dorothea_level")),
    }


def build_pairs(records, gene_category, regulators, cand_map, canon, max_pmids=20, listed=None):
    """{"GENE-REGULATOR": entry} for EVERY tested pair (the listed pairs, or every gene x
    regulator when listed is None): pairs with OmniPath records first (strongest first), then
    pairs with none (found: false), in selection order. A gene is never paired with itself,
    including under another name (same canon id)."""
    tried = {}
    for n, g in cand_map.items():
        tried.setdefault(g, []).append(n)

    pairs = {}
    for r in records:
        (src, src_as), (tgt, tgt_as) = _resolve(r, "source", cand_map), _resolve(r, "target", cand_map)
        if src is None or tgt is None or canon[src] == canon[tgt]:
            continue
        if src in regulators and tgt in gene_category:
            gene, reg, gene_as, reg_as, direction = tgt, src, tgt_as, src_as, "regulator->gene"
        elif tgt in regulators and src in gene_category:
            gene, reg, gene_as, reg_as, direction = src, tgt, src_as, tgt_as, "gene->regulator"
        else:
            continue  # gene-gene or regulator-regulator record
        if not r.get("is_directed"):
            direction = "undirected"
        pair = pairs.setdefault((gene, reg), {"matched_as": {"gene": gene_as, "regulator": reg_as},
                                              "interactions": [], "pmids": []})
        pair["interactions"].append(format_interaction(r, direction, max_pmids))
        pair["pmids"] += _pmids(r.get("references"))

    out = []
    for gene, cat in gene_category.items():
        for reg in regulators:
            if canon[gene] == canon[reg] or (listed is not None and (gene, reg) not in listed):
                continue
            hit = pairs.get((gene, reg), {"matched_as": None, "interactions": [], "pmids": []})
            inter = hit["interactions"]
            pmids = list(dict.fromkeys(hit["pmids"]))
            resources = list(dict.fromkeys(s for i in inter for s in i["resources"]))
            out.append({
                "pair": f"{gene}-{reg}",
                "gene": gene, "regulator": reg,
                "query_category": cat,
                "gene_category": cat.split("-")[0],
                "found": bool(inter),
                "regulator_stats": regulators[reg],
                "tried_names": {"gene": tried.get(gene, []), "regulator": tried.get(reg, [])},
                "matched_as": hit["matched_as"],
                "n_references": len(pmids),
                "pmids": pmids[:max_pmids],
                "n_resources": len(resources),
                "resources": resources,
                "curation_effort": sum(i["curation_effort"] or 0 for i in inter),
                "interaction_categories": list(dict.fromkeys(i["type"] for i in inter)),
                "datasets": list(dict.fromkeys(d for i in inter for d in i["datasets"])),
                "directions": list(dict.fromkeys(i["direction"] for i in inter)),
                "signs": list(dict.fromkeys(i["sign"] for i in inter)),
                "interactions": inter,
            })
    # stable sort: found pairs by evidence, not-found pairs keep selection order
    out.sort(key=lambda p: (p["found"], p["n_references"], p["curation_effort"]), reverse=True)
    return {p.pop("pair"): p for p in out}


def search_program(bundle, params):
    """OmniPath block of one program, plus the alias candidates dropped as ambiguous."""
    gene_sets, regulators, reg_entries, listed = select_sets(
        bundle, params["top_gene"], params["top_regulator"],
        params["include_unique"], params["top_unique_gene"])
    gene_entries, gene_category = {}, {}
    for cat, entries in gene_sets.items():
        for e in entries:
            gene_entries.setdefault(gene_name(e), e)
            gene_category.setdefault(gene_name(e), cat)
    cand_map, dropped = build_candidate_map({**gene_entries, **reg_entries}, params["use_aliases"])

    gene_cands = [n for n, g in cand_map.items() if g in gene_category]
    reg_cands = [n for n, g in cand_map.items() if g in regulators]
    records = fetch_records(gene_cands, reg_cands, params["datasets"], params["taxid"])
    canon = {g: canonical_id(e) for g, e in {**gene_entries, **reg_entries}.items()}
    pairs = build_pairs(records, gene_category, regulators, cand_map, canon, params["max_pmids"], listed)

    tested = {cat: sum(p["query_category"] == cat for p in pairs.values()) for cat in gene_sets}
    found = {cat: sum(p["query_category"] == cat and p["found"] for p in pairs.values()) for cat in gene_sets}
    block = {"params": params, "pair_source": "regulator_gene" if listed is not None else "flags",
             "n_regulators": len(regulators),
             "n_pairs_tested": tested, "n_pairs_found": found, "pairs": pairs}
    return block, dropped


def build_parser():
    p = argparse.ArgumentParser(description="Query OmniPath for each program's regulator-gene pairs (the regulator_gene list from Extract_program_information.py; older bundles: top genes x top regulators).")

    # IO
    p.add_argument("--bundle_dir", required=True, help="PerturbNMF_Info folder written by Extract_program_information.py.")
    p.add_argument("--out_dir", default=None, help="Shared output directory (existing P<k>.json there are extended). Default: <bundle_dir>/../Gene_info_extended_PerturbNMF_Info.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", required=True, help="Program ids, space separated (e.g. 1 2 3); only these bundles are read.")
    p.add_argument("--taxid", type=int, default=9606, help="NCBI taxonomy id (9606 = human).")

    # select top items of the program
    p.add_argument("--top_gene", type=int, default=15, help="Top-N loaded genes (program_genes) paired with regulators. Only for bundles without a regulator_gene block.")
    p.add_argument("--top_regulator", type=int, default=6, help="Top-N regulators per condition. Only for bundles without a regulator_gene block.")
    p.add_argument("--include_unique", action="store_true", help="Also pair the top unique genes (distinctive_genes) with the regulators. Only for bundles without a regulator_gene block.")
    p.add_argument("--top_unique_gene", type=int, default=8, help="Top-M unique genes, used with --include_unique.")

    # query
    p.add_argument("--datasets", nargs="+", default=list(ALL_DATASETS), choices=ALL_DATASETS, help="OmniPath datasets to search. Default: all.")
    p.add_argument("--no_aliases", action="store_true", help="Query bundle gene names only, not the MyGene/NCBI/UniProt aliases in gene_info.")
    p.add_argument("--max_pmids", type=int, default=20, help="Max PMIDs listed per pair / interaction (counts are uncapped).")
    p.add_argument("--overwrite", action="store_true", help="Re-query programs that already have OmniPath results with the same params.")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir or default_out_dir(args.bundle_dir))
    params = {"top_gene": args.top_gene, "top_regulator": args.top_regulator,
              "include_unique": args.include_unique, "top_unique_gene": args.top_unique_gene,
              "datasets": list(args.datasets), "use_aliases": not args.no_aliases,
              "max_pmids": args.max_pmids, "taxid": args.taxid,
              "pairs_format": PAIRS_FORMAT}

    bundles = load_program_JSON(args.bundle_dir, out_dir, args.programs)
    entries = [e for b in bundles.values() for e in b["program_genes"] + b["distinctive_genes"]
               + [r for regs in b.get("perturbation_regulators", {}).values() for r in regs]]
    if params["use_aliases"] and not any(isinstance(e, dict) and "gene_info" in e for e in entries):
        print("  [warn] no gene_info in the bundles; only bundle gene names are queried "
              "(run search_gene/search_MyGene.py, search_NCBI.py, search_UniProt.py first to use aliases)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary, skipped = {}, []
    for label, bundle in bundles.items():
        # skip programs already searched with the same params
        previous = bundle.get("gene_interactions", {}).get(SOURCE, {})
        if previous.get("params") == params and not args.overwrite:
            skipped.append(label)
            print(f"  {label}: OmniPath results with the same params exist; skipped (use --overwrite)")
            continue
        block, dropped = search_program(bundle, params)
        bundle.setdefault("gene_interactions", {})[SOURCE] = block
        (out_dir / f"{label}.json").write_text(json.dumps(bundle, indent=2))
        summary[label] = {"n_regulators": block["n_regulators"], "n_pairs_tested": block["n_pairs_tested"],
                          "n_pairs_found": block["n_pairs_found"], "dropped_aliases": dropped}
        found = ", ".join(f"{c}={block['n_pairs_found'][c]}/{block['n_pairs_tested'][c]}" for c in block["n_pairs_found"])
        print(f"  {label}: regulators={block['n_regulators']} pairs found [{found}]")
        if dropped:
            print(f"    [warn] ambiguous aliases dropped: {dropped}")

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"bundle_dir": str(args.bundle_dir)},
        "source": OMNIPATH_URL,
        "params": params,
        "programs": list(bundles),
        "skipped": skipped,
        "per_program": summary,
    }
    meta_path = out_dir / f"meta_{SOURCE}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(summary)} searched, {len(skipped)} skipped -> {out_dir}; meta -> {meta_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
