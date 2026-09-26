"""Collect the deterministic evidence an annotator sees for each regulator group.

Per group, for the members the promoter screen did not exclude:
  effect signature  the programs the group moves: the members' mean log2FC per program x
                    condition and how many members move it the same way significantly (ranked by
                    the product, so a shared effect outranks one member's large one), and the
                    program's label from a ProgramAnnotatorV3 run when one is given — so the
                    annotation can say WHY these genes cluster in THIS system
  members           role (core / peripheral / rescued), bootstrap stability, r to the group
                    centroid, reliability, effect strength, promoter-screen caveats, NCBI gene
                    summary, and how connected each member is to the others (STRING edges, shared
                    complexes, shared enriched terms) — a member with no connection is the
                    deterministic pre-flag for "unexplained member"
  STRING            edges among members (combined and physical score); PPI enrichment and
                    functional enrichment (GO BP/CC/MF, KEGG, Reactome, ...) against the
                    SCREENED TARGETS as background, not the genome
  complexes         curated complexes (CORUM / ComplexPortal / SIGNOR) with >= 2 members: how many
                    of the complex's subunits are in the group, were perturbed, and exist
  reference pool    PubTator papers whose title/abstract sentence names two members together
                    (the literature link between members), plus the papers the curated complexes
                    cite (their title stands in for a sentence); retracted and unresolvable PMIDs
                    are dropped (PubMed esummary). The annotator may cite only these, as in
                    ProgramAnnotatorV3

Also written, so the shared citation pass (annotator_core) runs on groups unchanged:
  string_enrichment_groups.csv  ProgramExplorer's filtered-enrichment format, program_id = group id
  group_context.json            ncbi_context-shaped: gene_summaries + evidence_snippets per group

Network (run outside any sandbox): string-db.org, mygene.info, www.ncbi.nlm.nih.gov (PubTator3),
eutils.ncbi.nlm.nih.gov. Everything is cached under --cache-dir.

Usage:
    python build_group_evidence.py --groups-dir regulator_groups --targets targets.tsv \
        --complexes omnipath_complexes.tsv --cache-dir cache \
        [--program-annotations dispatch --program-arm v3] --output-dir regulator_groups
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from answer_io import load_answer  # noqa: E402
from build_citation_candidates import MYGENE, SupportFinder  # noqa: E402
from complexes import load_curated_complexes  # noqa: E402
from http_cache import CachedHttp  # noqa: E402
from species import MYGENE_SPECIES, TAXON  # noqa: E402
from string_client import StringClient  # noqa: E402
from verify_cited_pmids import fetch_pubmed_summaries, is_retracted  # noqa: E402

ENRICHMENT_CATEGORIES = ("Process", "Function", "Component", "KEGG", "RCTM", "WikiPathways")
ENRICHMENT_FDR = 0.05
ENRICHMENT_TERMS_SHOWN = 15
SIGNATURE_FEATURES_SHOWN = 12
MAX_PAIRS_SEARCHED = 45          # a 10-member group; larger groups search pairs of their most stable members
POOL_MAX = 40
STRING_EDGE_SCORE = 400


def program_labels(dispatch: Path | None, arm: str) -> dict:
    """program id -> V3 label (and family), from a ProgramAnnotatorV3 dispatch directory."""
    if not dispatch:
        return {}
    labels = {}
    for answer in dispatch.glob(f"{arm}_p*/answer.json"):
        try:
            payload = load_answer(answer)
        except json.JSONDecodeError:
            continue
        pid = int(answer.parent.name.split("_p")[-1])
        labels[pid] = {"label": payload.get("label", ""), "family": payload.get("label_family", "")}
    return labels


def effect_signature(members: list[str], effects: pd.DataFrame, adjusted: pd.DataFrame,
                     labels: dict, alpha: float) -> list[dict]:
    block = effects.loc[members]
    significant = adjusted.loc[members] < alpha
    mean = block.mean(axis=0)
    # Rank by effect size x the share of members that move the program the same way significantly,
    # so a large mean driven by one member does not outrank an effect the whole group shares.
    agreement = ((np.sign(block) == np.sign(mean)) & significant).sum(axis=0)
    score = mean.abs() * agreement / len(members)
    rows = []
    for feature in score.sort_values(ascending=False).index[:SIGNATURE_FEATURES_SHOWN]:
        program, _, condition = feature.partition("|")
        pid = int(program.lstrip("P"))
        agree = int(agreement[feature])
        rows.append({"feature": feature, "program_id": pid, "condition": condition,
                     "mean_log2fc": round(float(mean[feature]), 3),
                     "members_significant_same_direction": agree, "members": len(members),
                     "program_label": labels.get(pid, {}).get("label", "")})
    return rows


def member_correlation(genes: list[str], effects: pd.DataFrame, reliability: pd.Series) -> dict:
    """Member x member correlation of effect profiles, raw and noise-corrected (r / sqrt(rel_i rel_j))."""
    genes = [g for g in genes if g in effects.index]
    raw = np.corrcoef(effects.loc[genes].to_numpy()) if len(genes) > 1 else np.ones((len(genes), len(genes)))
    rel = np.maximum(reliability.reindex(genes).fillna(1.0).to_numpy(), 0.2)
    corrected = np.clip(raw / np.sqrt(np.outer(rel, rel)), -1, 1)
    return {"genes": genes, "raw": np.round(raw, 3).tolist(), "corrected": np.round(corrected, 3).tolist()}


def gene_summary(http: CachedHttp, symbol: str) -> str:
    data = http.get_json(f"{MYGENE}?" + urllib.parse.urlencode(
        {"q": f"symbol:{symbol}", "species": MYGENE_SPECIES, "fields": "summary,name"})) or {}
    hit = (data.get("hits") or [{}])[0]
    return str(hit.get("summary") or hit.get("name") or "")


def co_mention_pool(finder: SupportFinder, members: list[str]) -> list[dict]:
    """Papers whose title/abstract sentence names two members: one PubTator search per pair."""
    pool: dict = {}
    for a, b in itertools.combinations(members, 2):
        pattern = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(b)}(?![A-Za-z0-9-])", re.IGNORECASE)
        for hit in finder.literature(a, [b.lower()]):
            if not pattern.search(hit["sentence"]):
                continue
            entry = pool.setdefault(hit["pmid"], {"pmid": hit["pmid"], "genes": set(), "sentence": hit["sentence"],
                                                  "title": hit.get("title", ""), "year": hit.get("year", "")})
            entry["genes"].update({a, b})
            if len(hit["sentence"]) > len(entry["sentence"]):
                entry["sentence"] = hit["sentence"]
    return list(pool.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups-dir", required=True, type=Path,
                        help="directory with regulator_groups.json, promoter_confounds.json, effect_matrix.tsv, significance.tsv")
    parser.add_argument("--targets", required=True, type=Path, help="TSV with a target_name column: every perturbed gene (the enrichment background)")
    parser.add_argument("--complexes", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--program-annotations", type=Path, help="ProgramAnnotatorV3 dispatch directory (labels for the effect signature)")
    parser.add_argument("--program-arm", default="v3")
    parser.add_argument("--significance", type=float, default=0.05)
    parser.add_argument("--groups", help="comma list of group ids (default: all)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    groups = json.loads((args.groups_dir / "regulator_groups.json").read_text())
    confounds = json.loads((args.groups_dir / "promoter_confounds.json").read_text())["groups"]
    effects = pd.read_csv(args.groups_dir / "effect_matrix.tsv", sep="\t", index_col=0)
    reliability = pd.read_csv(args.groups_dir / "regulator_summary.tsv", sep="\t", index_col=0)["reliability"]
    adjusted = pd.read_csv(args.groups_dir / "significance.tsv", sep="\t", index_col=0)
    targets = pd.read_csv(args.targets, sep="\t")["target_name"].astype(str).tolist()
    complexes = load_curated_complexes(args.complexes)
    labels = program_labels(args.program_annotations, args.program_arm)
    http = CachedHttp(args.cache_dir)
    string = StringClient(http)
    finder = SupportFinder(http, excluded_pmids=set())
    background_ids = list(string.string_ids(sorted(set(targets))).values())
    wanted = {int(g) for g in args.groups.split(",")} if args.groups else None

    evidence, enrichment_rows, context = {}, [], {}
    for group in groups["groups"]:
        gid = group["group_id"]
        if wanted and gid not in wanted:
            continue
        decisions = confounds.get(str(gid), {})
        excluded = [m["gene"] for m in group["members"] if decisions.get(m["gene"], {}).get("decision") == "exclude"]
        members = [m for m in group["members"] if m["gene"] not in excluded]
        names = [m["gene"] for m in members]
        if len(names) < 2:
            evidence[str(gid)] = {"group_id": gid, "skipped": "fewer than 2 members left after the promoter screen",
                                  "excluded": excluded}
            continue

        edges = string.network(names, required_score=STRING_EDGE_SCORE)
        ppi = string.ppi_enrichment(names, background_ids)
        terms = [t for t in string.enrichment(names, background_ids)
                 if t["category"] in ENRICHMENT_CATEGORIES and t["fdr"] < ENRICHMENT_FDR]
        terms.sort(key=lambda t: t["fdr"])
        for t in terms:
            enrichment_rows.append({"program_id": gid, "category": t["category"], "term": t["term"], "term_id": t["term"],
                                    "description": t["description"], "fdr": t["fdr"], "p_value": t["p_value"],
                                    "number_of_genes": t["number_of_genes"],
                                    "number_of_genes_in_background": t["number_of_genes_in_background"],
                                    "ncbiTaxonId": TAXON, "inputGenes": "|".join(t["genes"])})
        touched = []
        for entry in complexes:
            inside = sorted(set(entry.members) & set(names))
            if len(inside) >= 2:
                touched.append({"name": entry.name, "id": entry.complex_id, "members_in_group": inside,
                                "members_perturbed": sorted(set(entry.members) & set(targets)),
                                "size": len(entry.members), "sources": entry.sources, "pmids": entry.pmids[:5]})
        touched.sort(key=lambda c: (-len(c["members_in_group"]), c["size"]))

        summaries = {g: gene_summary(http, g) for g in names}
        by_stability = [m["gene"] for m in sorted(members, key=lambda m: -(m["stability"] or 0))]
        searched = [g for g in by_stability if g in names]
        while len(searched) > 2 and len(searched) * (len(searched) - 1) // 2 > MAX_PAIRS_SEARCHED:
            searched.pop()
        pool = co_mention_pool(finder, searched)
        in_pool = {p["pmid"] for p in pool}
        for c in touched[:8]:
            for pmid in c["pmids"]:
                if pmid in in_pool:
                    continue
                in_pool.add(pmid)
                pool.append({"pmid": pmid, "genes": set(c["members_in_group"]), "sentence": "",
                             "title": "", "year": "", "complex": c["name"]})
        records = fetch_pubmed_summaries([p["pmid"] for p in pool]) if pool else {}
        kept = []
        for p in pool:
            record = records.get(p["pmid"])
            if not record or record.get("error") or is_retracted(record):
                continue
            p["genes"] = sorted(p["genes"])
            p["title"] = p["title"] or record.get("title", "")
            p["year"] = p["year"] or str(record.get("pubdate", ""))[:4]
            if not p["sentence"]:  # a complex reference: its title, tagged with the complex
                p["sentence"] = f"[{p['complex']} reference] {p['title']}"
            kept.append(p)
        kept.sort(key=lambda p: (-len(p["genes"]), p["year"]))
        kept = kept[:POOL_MAX]

        connections = {g: set() for g in names}
        for e in edges:
            connections[e["a"]].add(e["b"])
            connections[e["b"]].add(e["a"])
        for c in touched:
            for g in c["members_in_group"]:
                connections[g].update(set(c["members_in_group"]) - {g})
        for t in terms[:ENRICHMENT_TERMS_SHOWN]:
            inside = set(t["genes"]) & set(names)
            for g in inside:
                connections[g].update(inside - {g})
        member_rows = []
        for m in members:
            d = decisions.get(m["gene"], {})
            member_rows.append({**m, "promoter": {"decision": d.get("decision", "clear"), "reasons": d.get("reasons", []),
                                                  "shared_locus_with": d.get("shared_locus_with", [])},
                                "connected_to": sorted(connections[m["gene"]]), "summary": summaries[m["gene"]][:600]})

        evidence[str(gid)] = {
            "group_id": gid, "stability": group["stability"], "mean_raw_r": group["mean_raw_r"],
            "mean_corrected_r": group["mean_corrected_r"], "strength_tiers": group["strength_tiers"],
            "members": member_rows,
            "excluded": [{"gene": g, "reasons": decisions[g]["reasons"]} for g in excluded],
            "signature": effect_signature(names, effects, adjusted, labels, args.significance),
            "string_edges": edges, "ppi_enrichment": ppi, "enrichment": terms[:ENRICHMENT_TERMS_SHOWN],
            "complexes": touched[:8], "reference_pool": kept,
            "rescue_tests": [t for t in group.get("rescue_tests", []) if t["accepted"]],
            "member_correlation": member_correlation(names + excluded, effects, reliability),
        }
        context[str(gid)] = {
            "gene_summaries": summaries,
            "evidence_snippets": {g: [f"{p['sentence']} (PMID:{p['pmid']})" for p in kept if g in p["genes"]] for g in names},
        }
        print(f"G{gid}: {len(names)} members ({len(excluded)} excluded); {len(edges)} STRING edges, "
              f"PPI p={ppi.get('p_value') if ppi else 'n/a'}; {len(terms)} terms; {len(touched)} complexes; "
              f"{len(kept)} papers", flush=True)
        http.save()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "group_evidence.json").write_text(json.dumps(
        {"programs_labelled": bool(labels), "groups": evidence}, indent=1))
    pd.DataFrame(enrichment_rows).to_csv(args.output_dir / "string_enrichment_groups.csv", index=False)
    (args.output_dir / "group_context.json").write_text(json.dumps(context, indent=1))
    http.save()
    print(f"wrote evidence for {len(evidence)} groups -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
