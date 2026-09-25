"""Decide, for every grouped regulator, whether a neighbouring promoter could explain its membership.

CRISPRi represses a window around where its guides bind, so a guide for gene A can also silence
gene B whose promoter sits next to A's — most often a divergent (bidirectional) pair. If B is the
gene that matters (another member of the group, a subunit of the group's complex, a STRING partner
of its members), A is in the group because of B, and annotating A's own biology would be wrong.

Per member (core, peripheral or rescued), for each gene with a TSS within --window bp of where its
guides act (annotator_core/gene_coordinates.py; guide positions when the library names carry them,
else the target's TSSs), with knockdown evidence from measure_neighbour_knockdown.py when given:

  neighbour knocked down   log2FC <= --knockdown-log2fc and q < 0.05 (measured in the screen's cells)
  target knocked down      the same test on the target itself
  explains the group       the neighbour is another member of the group, or shares a curated
                           complex with a member, or has a STRING edge (combined score >=
                           --string-score) to a member

  EXCLUDE  the neighbour explains the group AND (it is knocked down, or it was not measured and its
           TSS is within --high-risk-window bp — the confound cannot be ruled out); or the
           neighbour is knocked down and the target itself is not
  FLAG     a protein-coding neighbour within the window that is knocked down, or not measured, but
           does not explain the group — kept for annotation, shown to the annotator as a caveat
  CLEAR    no neighbour, or every neighbour was measured and not knocked down

Two members that are each other's neighbour (a bidirectional pair both in the group, C5orf22 and
DROSHA) are one locus: each one's guides may silence the other, so the data cannot say which
gene the group responds to. The member with more support from the REST of the group (other
members it shares a curated complex or a STRING edge with) is kept and the other excluded —
DROSHA shares Microprocessor with DGCR8, C5orf22 shares nothing. On a tie both stay, flagged
`shared_locus`, and the prompt tells the annotator to count them as one gene.

Output: promoter_confounds.json — per group, per member: decision, reason, neighbours with their
evidence.

Usage:
    python build_promoter_confound_screen.py --groups regulator_groups/regulator_groups.json \
        --gene-coordinates gene_coordinates.tsv [--knockdown regulator_groups/knockdown.tsv] \
        [--complexes omnipath_complexes.tsv] [--string-cache cache] \
        --output regulator_groups/promoter_confounds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from complexes import complexes_by_gene, load_curated_complexes  # noqa: E402
from gene_coordinates import (  # noqa: E402
    guide_positions_on_this_assembly, load_gene_tss, load_guide_table_positions, parse_guide_position,
    promoter_neighbours,
)


def load_knockdown(path: Path | None) -> dict:
    if not path:
        return {}
    table = pd.read_csv(path, sep="\t")
    return {(r.target, r.gene): r._asdict() for r in table.itertuples(index=False)}


def knockdown_call(record: dict | None, threshold: float) -> dict:
    if not record or not record.get("measured") or pd.isna(record.get("log2fc")):
        return {"measured": False}
    down = bool(record["log2fc"] <= threshold and record["q_value"] < 0.05)
    return {"measured": True, "knocked_down": down, "log2fc": round(float(record["log2fc"]), 3),
            "q_value": float(f"{record['q_value']:.3g}"), "n_target_cells": int(record["n_target_cells"])}


def string_partners(members: list[str], neighbours: list[str], cache: Path | None, min_score: float) -> dict:
    """gene -> the members it has a STRING edge with, for members and neighbours (empty without STRING)."""
    if not cache:
        return {}
    from http_cache import CachedHttp
    from string_client import StringClient
    http = CachedHttp(cache)
    edges = StringClient(http).network(sorted(set(members) | set(neighbours)), required_score=int(min_score * 1000))
    http.save()
    partners: dict = {}
    for e in edges:
        for a, b in ((e["a"], e["b"]), (e["b"], e["a"])):
            if b in members and a != b:
                partners.setdefault(a, []).append(b)
    return partners


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--gene-coordinates", required=True, type=Path)
    parser.add_argument("--guide-names", type=Path,
                        help="optional TSV with guide_name, target columns (hCRISPRi-v2 names carry positions)")
    parser.add_argument("--guide-table", type=Path,
                        help="IGVF 'guide RNA sequences' table with guide coordinates (preferred over --guide-names)")
    parser.add_argument("--knockdown", type=Path, help="measure_neighbour_knockdown.py output")
    parser.add_argument("--complexes", type=Path, help="OmniPath complexes TSV")
    parser.add_argument("--string-cache", type=Path, help="enable the STRING-partner test (network: string-db.org)")
    parser.add_argument("--string-score", type=float, default=0.7)
    parser.add_argument("--window", type=int, default=3000)
    parser.add_argument("--high-risk-window", type=int, default=1000)
    parser.add_argument("--knockdown-log2fc", type=float, default=-0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    tss = load_gene_tss(args.gene_coordinates)
    guide_positions: dict[str, list[int]] = {}
    if args.guide_names:
        for row in pd.read_csv(args.guide_names, sep="\t").itertuples(index=False):
            parsed = parse_guide_position(row.guide_name)
            if parsed:
                guide_positions.setdefault(row.target, []).append(parsed[1])
        guide_positions, _ = guide_positions_on_this_assembly(guide_positions, tss)
    if args.guide_table:
        guide_positions = load_guide_table_positions(args.guide_table, tss)
    knockdown = load_knockdown(args.knockdown)
    complexes = load_curated_complexes(args.complexes) if args.complexes else []
    complex_index = complexes_by_gene(complexes)

    report = {"parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "groups": {}}
    counts = {"exclude": 0, "flag": 0, "clear": 0}
    for group in groups["groups"]:
        members = [m["gene"] for m in group["members"]]
        neighbours_of = {}
        for gene in members:
            found = promoter_neighbours(gene, tss, guide_positions, args.window)
            # A neighbour the knockdown step measured but the TSS-only search missed (guide sites
            # reach further than the TSS) is added from the table.
            names = {n["gene"] for n in found}
            for (target, other), record in knockdown.items():
                if target == gene and record["relation"] == "neighbour" and other not in names:
                    found.append({"gene": other, "distance": int(record["distance"]), "gene_type": "",
                                  "orientation": "unknown", "site_source": "guide"})
            neighbours_of[gene] = found
        all_neighbours = sorted({n["gene"] for ns in neighbours_of.values() for n in ns})
        string_of = string_partners(members, all_neighbours, args.string_cache, args.string_score)

        def group_support(gene: str, excluding: str) -> set:
            """Other members (not `excluding`) sharing a curated complex or a STRING edge with gene."""
            others = set(members) - {gene, excluding}
            by_complex = {o for i in complex_index.get(gene, []) for o in complexes[i].members if o in others}
            return by_complex | (set(string_of.get(gene, [])) & others)

        decisions = {}
        for gene in members:
            target_kd = knockdown_call(knockdown.get((gene, gene)), args.knockdown_log2fc)
            others = [m for m in members if m != gene]
            entries, exclude_reasons, flag_reasons, shared = [], [], [], []
            for n in neighbours_of[gene]:
                name = n["gene"]
                kd = knockdown_call(knockdown.get((gene, name)), args.knockdown_log2fc)
                in_group = name in others
                shared_complex = sorted({complexes[i].name for i in complex_index.get(name, [])
                                         for o in others if o in complexes[i].members})
                string_with = sorted(set(string_of.get(name, [])) & set(others))
                explains = in_group or bool(shared_complex) or bool(string_with)
                entry = {**n, "knockdown": kd, "in_group": in_group, "shared_complex": shared_complex[:3],
                         "string_partners_in_group": string_with, "explains_group": explains}
                entries.append(entry)
                if in_group:
                    shared.append(name)
                where = f"{name} ({n['orientation']}, {n['distance']} bp)"
                if in_group:
                    # One locus, two members: keep the better-supported one (see module docstring).
                    mine, theirs = group_support(gene, name), group_support(name, gene)
                    entry["support_in_group"] = {gene: sorted(mine), name: sorted(theirs)}
                    if len(theirs) > len(mine) and (not kd.get("measured") or kd["knocked_down"]):
                        exclude_reasons.append(
                            f"shares a promoter with {where}, another member; {name} is the better-supported "
                            f"gene of the locus ({len(theirs)} vs {len(mine)} other members share a complex or "
                            f"STRING edge){'; ' + gene + ' guides knock it down, log2FC ' + str(kd['log2fc']) if kd.get('measured') else ''}")
                    elif len(theirs) == len(mine):
                        flag_reasons.append(f"shares a promoter with {where}, another member — count the two as one locus")
                    continue
                why = ("another member of this group" if in_group else
                       f"shares a complex with members ({shared_complex[0]})" if shared_complex else
                       f"STRING partner of {', '.join(string_with[:3])}" if string_with else "")
                if kd.get("measured") and kd["knocked_down"]:
                    if explains:
                        exclude_reasons.append(f"guides also knock down {where}, {why} (log2FC {kd['log2fc']})")
                    elif target_kd.get("measured") and not target_kd["knocked_down"]:
                        exclude_reasons.append(f"guides knock down {where} (log2FC {kd['log2fc']}) but not "
                                               f"{gene} itself (log2FC {target_kd['log2fc']})")
                    elif n.get("gene_type", "protein_coding") in ("protein_coding", ""):
                        flag_reasons.append(f"guides also knock down {where} (log2FC {kd['log2fc']})")
                elif not kd.get("measured"):
                    if explains and n["distance"] <= args.high_risk_window:
                        exclude_reasons.append(f"promoter within {n['distance']} bp of {where.split(' (')[0]}, "
                                               f"{why}; its knockdown was not measured, so it cannot be ruled out")
                    elif n.get("gene_type") == "protein_coding" or explains:
                        flag_reasons.append(f"promoter near {where}; knockdown not measured")
            decision = "exclude" if exclude_reasons else ("flag" if flag_reasons else "clear")
            counts[decision] += 1
            decisions[gene] = {"decision": decision, "reasons": exclude_reasons or flag_reasons,
                               "target_knockdown": target_kd, "shared_locus_with": shared, "neighbours": entries}
        report["groups"][str(group["group_id"])] = decisions

    args.output.write_text(json.dumps(report, indent=1))
    print(f"promoter screen: {counts['exclude']} excluded, {counts['flag']} flagged, {counts['clear']} clear -> {args.output}")
    for gid, decisions in report["groups"].items():
        for gene, d in decisions.items():
            if d["decision"] != "clear":
                print(f"  G{gid} {gene}: {d['decision'].upper()} — {d['reasons'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
