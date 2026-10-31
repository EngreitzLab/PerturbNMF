#!/usr/bin/env python3
"""OmniPath interactions between perturbation regulators and genes (both directions, every dataset), one record per
tested regulator-gene pair, found or not.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import (
    SkillError,
    append_jsonl,
    as_list,
    read_jsonl,
    request_with_retry,
    require_str_list,
    run,
)  # noqa: E402,F401

SOURCE = "OmniPath"
OMNIPATH_URL = "https://omnipathdb.org/interactions"
ALL_DATASETS = (
    "omnipath",
    "pathwayextra",
    "kinaseextra",
    "ligrecextra",
    "collectri",
    "dorothea",
    "tf_target",
    "mirnatarget",
    "lncrna_mrna",
    "tf_mirna",
    "small_molecule",
)
FIELDS = "sources,references,curation_effort,type,datasets,extra_attrs,dorothea_level"
GENE_INFO_SOURCES = ("MyGene", "NCBI", "UniProt")
# stored in params so results written in an older layout are re-queried, not skipped
PAIRS_FORMAT = "regulator_gene_v3"


def name_candidates(entry, use_aliases=True):
    """Names to try for one gene: the bundle name, then symbols/aliases/UniProt accession
    from the gene_info blocks added by the search_gene scripts."""
    names = [gene_name(entry)]
    if use_aliases and isinstance(entry, dict):
        for src in GENE_INFO_SOURCES:
            info = entry.get("gene_info", {}).get(src, {})
            if not info.get("found"):
                continue
            names += [info.get("symbol"), info.get("accession")] + as_list(
                info.get("alias")
            )
    return list(dict.fromkeys(str(n) for n in names if n))


def canonical_id(entry):
    """Stable id of a gene (UniProt accession, then NCBI/MyGene gene id, then its name), so
    an old and a new name of the same gene (MESDC1 / TLNRD1) are not paired with each other."""
    info = entry.get("gene_info", {}) if isinstance(entry, dict) else {}
    for src, key in (
        ("UniProt", "accession"),
        ("NCBI", "gene_id"),
        ("MyGene", "entrezgene"),
    ):
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
    params = {
        "sources": ",".join(sources),
        "targets": ",".join(targets),
        "source_target": "AND",
        "genesymbols": "yes",
        "organisms": taxid,
        "datasets": ",".join(datasets),
        "fields": FIELDS,
        "format": "json",
    }
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
    return {
        (True, True): "both",
        (True, False): "activation",
        (False, True): "inhibition",
    }.get((bool(stim), bool(inhib)), "unknown")


def _pmids(references):
    """Unique PMIDs from OmniPath's 'Resource:PMID;Resource:PMID' string."""
    refs = [r.split(":", 1)[-1] for r in str(references or "").split(";") if r]
    return list(dict.fromkeys(p for p in refs if p.isdigit()))


def format_interaction(record, direction, max_pmids=20):
    pmids = _pmids(record.get("references"))
    mechanisms = [
        str(m)
        for k, v in (record.get("extra_attrs") or {}).items()
        if k.endswith("_mechanism")
        for m in as_list(v)
    ]
    return {
        "source": record.get("source_genesymbol"),
        "target": record.get("target_genesymbol"),
        "direction": direction,
        "sign": _sign(record.get("is_stimulation"), record.get("is_inhibition")),
        "consensus_sign": _sign(
            record.get("consensus_stimulation"), record.get("consensus_inhibition")
        ),
        "type": record.get("type"),
        "datasets": [d for d in ALL_DATASETS if record.get(d) is True],
        "resources": as_list(record.get("sources")),
        "curation_effort": record.get("curation_effort"),
        "n_references": len(pmids),
        "pmids": pmids[:max_pmids],
        "mechanisms": list(dict.fromkeys(mechanisms)),
        "dorothea_level": as_list(record.get("dorothea_level")),
    }


def build_pairs(
    records, gene_category, regulators, cand_map, canon, max_pmids=20, listed=None
):
    """{"GENE-REGULATOR": entry} for EVERY tested pair (the listed pairs, or every gene x
    regulator when listed is None): pairs with OmniPath records first (strongest first), then
    pairs with none (found: false), in selection order. A gene is never paired with itself,
    including under another name (same canon id)."""
    tried = {}
    for n, g in cand_map.items():
        tried.setdefault(g, []).append(n)

    pairs = {}
    for r in records:
        (src, src_as), (tgt, tgt_as) = (
            _resolve(r, "source", cand_map),
            _resolve(r, "target", cand_map),
        )
        if src is None or tgt is None or canon[src] == canon[tgt]:
            continue
        if src in regulators and tgt in gene_category:
            gene, reg, gene_as, reg_as, direction = (
                tgt,
                src,
                tgt_as,
                src_as,
                "regulator->gene",
            )
        elif tgt in regulators and src in gene_category:
            gene, reg, gene_as, reg_as, direction = (
                src,
                tgt,
                src_as,
                tgt_as,
                "gene->regulator",
            )
        else:
            continue  # gene-gene or regulator-regulator record
        if not r.get("is_directed"):
            direction = "undirected"
        pair = pairs.setdefault(
            (gene, reg),
            {
                "matched_as": {"gene": gene_as, "regulator": reg_as},
                "interactions": [],
                "pmids": [],
            },
        )
        pair["interactions"].append(format_interaction(r, direction, max_pmids))
        pair["pmids"] += _pmids(r.get("references"))

    out = []
    for gene, cat in gene_category.items():
        for reg in regulators:
            if canon[gene] == canon[reg] or (
                listed is not None and (gene, reg) not in listed
            ):
                continue
            hit = pairs.get(
                (gene, reg), {"matched_as": None, "interactions": [], "pmids": []}
            )
            inter = hit["interactions"]
            pmids = list(dict.fromkeys(hit["pmids"]))
            resources = list(dict.fromkeys(s for i in inter for s in i["resources"]))
            out.append(
                {
                    "pair": f"{gene}-{reg}",
                    "gene": gene,
                    "regulator": reg,
                    "query_category": cat,
                    "gene_category": cat.split("-")[0],
                    "found": bool(inter),
                    "regulator_stats": regulators[reg],
                    "tried_names": {
                        "gene": tried.get(gene, []),
                        "regulator": tried.get(reg, []),
                    },
                    "matched_as": hit["matched_as"],
                    "n_references": len(pmids),
                    "pmids": pmids[:max_pmids],
                    "n_resources": len(resources),
                    "resources": resources,
                    "curation_effort": sum(i["curation_effort"] or 0 for i in inter),
                    "interaction_categories": list(
                        dict.fromkeys(i["type"] for i in inter)
                    ),
                    "datasets": list(
                        dict.fromkeys(d for i in inter for d in i["datasets"])
                    ),
                    "directions": list(dict.fromkeys(i["direction"] for i in inter)),
                    "signs": list(dict.fromkeys(i["sign"] for i in inter)),
                    "interactions": inter,
                }
            )
    # stable sort: found pairs by evidence, not-found pairs keep selection order
    out.sort(
        key=lambda p: (p["found"], p["n_references"], p["curation_effort"]),
        reverse=True,
    )
    return {p.pop("pair"): p for p in out}


def gene_name(entry):
    return entry if isinstance(entry, str) else entry["gene"]


def handle(req):
    genes = require_str_list(req, "genes", max_len=500)
    regs = req.get("regulators")
    if isinstance(regs, list):
        regs = {r: {} for r in regs}
    if not isinstance(regs, dict) or not regs:
        raise SkillError(
            "invalid_input",
            "`regulators` must be a non-empty list of symbols or {symbol: stats}",
        )
    info = (
        {
            r["gene"]: {k: v for k, v in r.items() if k != "gene"}
            for r in read_jsonl(req["gene_info"])
        }
        if req.get("gene_info")
        else {}
    )
    entries = {g: {"gene": g, "gene_info": info.get(g, {})} for g in [*genes, *regs]}
    category = req.get("gene_category") or {}
    gene_category = {
        g: category.get(g, "program_gene") + "-regulator"
        for g in genes
        if g not in regs
    }
    params = {
        "datasets": req.get("datasets") or list(ALL_DATASETS),
        "taxid": int(req.get("taxid", 9606)),
        "max_pmids": int(req.get("max_pmids", 20)),
        "use_aliases": bool(req.get("use_aliases", True)),
    }
    gene_entries = {g: entries[g] for g in gene_category}
    reg_entries = {r: entries[r] for r in regs}
    cand_map, dropped = build_candidate_map(
        {**gene_entries, **reg_entries}, params["use_aliases"]
    )
    gene_cands = [n for n, g in cand_map.items() if g in gene_category]
    reg_cands = [n for n, g in cand_map.items() if g in regs]
    records = fetch_records(gene_cands, reg_cands, params["datasets"], params["taxid"])
    canon = {g: canonical_id(e) for g, e in entries.items()}
    pairs = build_pairs(
        records, gene_category, regs, cand_map, canon, params["max_pmids"]
    )
    rows = [{"pair": k, **v} for k, v in pairs.items()]
    found = [r for r in rows if r["found"]]
    out = {
        "source": OMNIPATH_URL,
        "n_pairs_tested": len(rows),
        "n_pairs_found": len(found),
        "ambiguous_aliases_dropped": dropped,
    }
    if req.get("write_to"):
        added, dup = append_jsonl(req["write_to"], rows, dedupe=lambda r: r["pair"])
        out.update(
            {"written_to": req["write_to"], "added": added, "already_present": dup}
        )
        out["found"] = [
            {
                k: r[k]
                for k in (
                    "gene",
                    "regulator",
                    "directions",
                    "signs",
                    "interaction_categories",
                    "n_references",
                )
            }
            for r in found
        ]
    else:
        out["pairs"] = rows
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
