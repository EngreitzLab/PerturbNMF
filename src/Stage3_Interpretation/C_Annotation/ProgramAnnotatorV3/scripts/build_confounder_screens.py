"""Compute deterministic confounder screens for each cNMF program.

These answer "is there a non-biological or non-specific reason these genes co-vary?" from the
data alone, before any LLM sees the program. The v2 annotation prompt injects the numbers and
requires the model to clear each confounder by citing one.

Every screen reports an observed count, the background expectation, and a fold change, because
"12 of the top 50 genes are ribosomal" means nothing without "and ribosomal genes are 4% of the
background".

Usage:
    python build_confounder_screens.py \
        --gene-loading gene_loading_top300_with_uniqueness.csv \
        --gene-coordinates gene_coordinates.tsv \
        --targets targets.tsv \
        --regulators regulators.csv \
        --output confounder_screens.json

A multi-condition screen whose conditions differ in which cells are present (e.g. timepoints of
a differentiation) can add one screen, `condition_composition`, enabled by passing both
--activity-by-condition and --condition-markers (see ../configs/example_condition_markers.json):

    python build_confounder_screens.py ... \
        --activity-by-condition program_activity_by_condition.csv \
        --condition-markers ../configs/example_condition_markers.json

--stage-markers is a deprecated alias of --condition-markers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy.stats import binomtest, hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from conditions import warn_once  # noqa: E402
from gene_coordinates import load_gene_coordinates  # noqa: E402

TOP_N_GENES = 50
POSITIONAL_WINDOW_BP = 10_000_000

# Tirosh/Seurat cell-cycle sets, with both the legacy and current symbols for the four genes
# that were renamed (MLF1IP/CENPU, FAM64A/PIMREG, HN1/JPT1) so a symbol change cannot silently
# shrink the overlap.
S_PHASE_GENES = """MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1
MLF1IP CENPU HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51
RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8""".split()

G2M_PHASE_GENES = """HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO
CENPF TACC3 FAM64A PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B
HJURP CDCA3 HN1 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR
AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA""".split()

HEAT_SHOCK_GENES = """HSPA1A HSPA1B HSPA6 HSPA8 HSPB1 HSPH1 HSP90AA1 HSP90AB1 DNAJA1 DNAJB1
DNAJB6 BAG3 HSPE1 HSPD1 CHORDC1 AHSA1 UBB""".split()

INTEGRATED_STRESS_GENES = """ATF4 DDIT3 TRIB3 ASNS CHAC1 SESN2 PPP1R15A DDIT4 ATF3 CEBPB SLC7A11
SLC3A2 WARS1 GARS1 CTH PSAT1 PHGDH SHMT2 MTHFD2 XPOT""".split()

UNFOLDED_PROTEIN_GENES = """HSPA5 XBP1 EDEM1 DNAJB9 SEC61A1 PDIA4 PDIA6 HERPUD1 MANF SDF2L1
CRELD2 HYOU1 CALR CANX P4HB""".split()

INTERFERON_GENES = """ISG15 IFI6 IFIT1 IFIT2 IFIT3 IFITM1 IFITM3 MX1 MX2 OAS1 OAS2 OAS3 OASL
STAT1 STAT2 IRF7 IRF9 BST2 XAF1 RSAD2 SAMD9 SAMD9L HLA-A HLA-B HLA-C B2M""".split()

# Symbol-prefix families. Anchored so that e.g. "RPS6KA1" (a kinase) is not counted as a
# ribosomal protein.
PREFIX_FAMILIES = {
    "cytoplasmic_ribosome": re.compile(r"^RP[LS]\d"),
    "mitochondrial_ribosome": re.compile(r"^MRP[LS]\d"),
    "mitochondrially_encoded": re.compile(r"^MT-"),
    "histone": re.compile(r"^(H1-\d|H2A[A-Z]|H2B[A-Z]|H3-\d|H3C\d|H4C\d|HIST\d)"),
    "small_nucleolar_or_nuclear_rna": re.compile(r"^(SNOR|SCARNA|RNU\d|RNVU)"),
    "microrna": re.compile(r"^MIR\d"),
    "long_noncoding_or_antisense": re.compile(r"(^LINC\d|-AS\d$|^MIR\d+HG$)"),
}

NAMED_SETS = {
    "cell_cycle_s_phase": set(S_PHASE_GENES),
    "cell_cycle_g2m_phase": set(G2M_PHASE_GENES),
    "heat_shock": set(HEAT_SHOCK_GENES),
    "integrated_stress_response": set(INTEGRATED_STRESS_GENES),
    "unfolded_protein_response": set(UNFOLDED_PROTEIN_GENES),
    "interferon_response": set(INTERFERON_GENES),
}


def score_positional_concentration(
    genes: List[str], coordinates: Dict[str, dict], background: Counter, n_background: int
) -> dict:
    """Do the top genes pile up on one chromosome, or inside one window of one chromosome?

    Two numbers, because they catch different things: a whole-arm excess is aneuploidy or a
    large CNV, while a tight window is a locus cluster (an imprinted region, a histone cluster,
    an amplicon).
    """
    located = [(g, coordinates[g]) for g in genes if g in coordinates]
    if not located:
        return {"assessable": False, "reason": "no top genes found in the annotation"}

    per_chrom = Counter(info["chrom"] for _, info in located)
    top_chrom, top_chrom_count = per_chrom.most_common(1)[0]
    expected_fraction = background[top_chrom] / n_background if n_background else 0.0
    binomial = binomtest(top_chrom_count, len(located), expected_fraction, alternative="greater")

    best_window = {"chrom": None, "count": 0, "start": None, "end": None, "genes": []}
    by_chrom = defaultdict(list)
    for gene, info in located:
        by_chrom[info["chrom"]].append((info["start"], gene))
    for chrom, entries in by_chrom.items():
        entries.sort()
        for i, (start, _) in enumerate(entries):
            window = [g for s, g in entries[i:] if s - start <= POSITIONAL_WINDOW_BP]
            if len(window) > best_window["count"]:
                best_window = {
                    "chrom": chrom,
                    "count": len(window),
                    "start": start,
                    "end": start + POSITIONAL_WINDOW_BP,
                    "genes": window,
                }

    return {
        "assessable": True,
        "n_genes_located": len(located),
        "top_chromosome": top_chrom,
        "genes_on_top_chromosome": top_chrom_count,
        "expected_on_top_chromosome": round(expected_fraction * len(located), 1),
        "fold_enrichment": round(
            (top_chrom_count / len(located)) / expected_fraction, 2
        ) if expected_fraction else None,
        "binomial_p": f"{binomial.pvalue:.2e}",
        "densest_10mb_window": best_window,
    }


def score_gene_set_overlap(genes: List[str], gene_set: set, background: set) -> dict:
    """Hypergeometric overlap of the top genes with a marker set, within the run's own universe.

    The universe is the program-gene universe, not the genome: asking whether ribosomal genes
    are enriched among the top 50 relative to all human genes would call almost every program
    ribosomal, because the background is not what was measured.
    """
    # The marker sets and prefix patterns are human symbols; compare upper-cased so a mouse
    # screen (Mki67, Rpl13, mt-Co1) is screened too instead of silently scoring zero.
    upper_background = {g.upper(): g for g in background}
    set_in_background = {upper_background[g] for g in gene_set if g in upper_background}
    overlap = sorted(set(genes) & set_in_background)
    if not set_in_background:
        return {"n_overlap": 0, "genes": [], "expected": 0.0, "p_value": None}
    expected = len(genes) * len(set_in_background) / len(background)
    p_value = hypergeom.sf(
        len(overlap) - 1, len(background), len(set_in_background), len(genes)
    )
    return {
        "n_overlap": len(overlap),
        "genes": overlap[:15],
        "expected": round(expected, 2),
        "p_value": f"{p_value:.2e}" if len(overlap) else None,
    }


def score_condition_composition(
    program_activity: pd.DataFrame, top_genes: List[str], condition_markers: Dict[str, dict],
    background: set,
) -> dict:
    """Does this program just track which cells are present in a given condition?

    When conditions differ in cell composition (e.g. timepoints of a differentiation), a program
    can look like a coherent pathway when it is really the transcriptome difference between two
    cell populations. Two deterministic signals of that: activity concentrated in one condition
    (or, for ordered conditions, one contiguous block), and top genes that are the canonical
    markers of that condition's cells. Neither alone is decisive; both together point to
    composition rather than regulation.
    """
    order = list(condition_markers)
    by_condition = program_activity.set_index("condition")["mean_score"].reindex(order)
    total = float(by_condition.sum())
    peak = str(by_condition.idxmax())
    ranked = by_condition.sort_values(ascending=False)
    second = float(ranked.iloc[1]) if len(ranked) > 1 else 0.0
    lowest = float(ranked.iloc[-1])
    markers = {
        condition: score_gene_set_overlap(top_genes, set(entry["genes"]), background)
        | {"description": entry["description"]}
        for condition, entry in condition_markers.items()
    }
    return {
        "condition_order": order,
        "mean_score": {c: round(float(v), 5) for c, v in by_condition.items()},
        "share_of_total": {c: round(float(v) / total, 3) if total else None for c, v in by_condition.items()},
        "peak_condition": peak,
        "peak_share": round(float(by_condition.max()) / total, 3) if total else None,
        "peak_over_second": round(float(by_condition.max()) / second, 2) if second else None,
        "peak_over_lowest": round(float(by_condition.max()) / lowest, 2) if lowest else None,
        "marker_overlap": markers,
    }


def build_screens(
    gene_loading: pd.DataFrame,
    coordinates: Dict[str, dict],
    screen_targets: set,
    regulators: pd.DataFrame,
    activity: pd.DataFrame | None = None,
    condition_markers: Dict[str, dict] | None = None,
) -> Dict[int, dict]:
    background_genes = set(gene_loading["Name"])
    background_chroms = Counter(
        coordinates[g]["chrom"] for g in background_genes if g in coordinates
    )
    n_background_located = sum(background_chroms.values())

    screens: Dict[int, dict] = {}
    for program_id, frame in gene_loading.groupby("program_id"):
        ranked = frame.sort_values("Score", ascending=False)
        top_genes = ranked["Name"].head(TOP_N_GENES).tolist()

        family_hits = {}
        for family, pattern in PREFIX_FAMILIES.items():
            matched = [g for g in top_genes if pattern.search(g.upper())]
            in_background = [g for g in background_genes if pattern.search(g.upper())]
            expected = len(top_genes) * len(in_background) / len(background_genes)
            family_hits[family] = {
                "n_overlap": len(matched),
                "genes": matched[:15],
                "expected": round(expected, 2),
            }

        set_hits = {
            name: score_gene_set_overlap(top_genes, gene_set, background_genes)
            for name, gene_set in NAMED_SETS.items()
        }

        biotypes = Counter(
            coordinates[g]["gene_type"] for g in top_genes if g in coordinates
        )

        program_regulators = regulators[regulators["program_id"] == program_id]
        significant = program_regulators[program_regulators["significant"]]

        cis_targets = sorted(set(top_genes) & screen_targets)

        screens[int(program_id)] = {
            "program_id": int(program_id),
            "n_top_genes_screened": len(top_genes),
            "positional": score_positional_concentration(
                top_genes, coordinates, background_chroms, n_background_located
            ),
            "gene_biotypes": dict(biotypes.most_common()),
            "symbol_families": family_hits,
            "marker_sets": set_hits,
            "cis_targets_in_top_genes": {
                "n_overlap": len(cis_targets),
                "genes": cis_targets[:15],
                "note": "top genes that are themselves CRISPRi targets in this screen",
            },
            "regulators": {
                "n_tested": int(len(program_regulators)),
                "n_significant": int(len(significant)),
            },
        }
        if activity is not None and condition_markers:
            screens[int(program_id)]["condition_composition"] = score_condition_composition(
                activity[activity["program_id"] == program_id], top_genes, condition_markers,
                background_genes,
            )
    return screens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-loading", required=True, type=Path)
    parser.add_argument("--gene-coordinates", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--regulators", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--activity-by-condition", type=Path,
                        help="program_id, condition, mean_score (multi-condition screens only)")
    parser.add_argument("--condition-markers", "--stage-markers", dest="condition_markers", type=Path,
                        help="JSON {condition: {description, genes}}: canonical marker genes of the "
                             "cells in each condition (--stage-markers is a deprecated alias)")
    args = parser.parse_args()

    gene_loading = pd.read_csv(args.gene_loading)
    if "program_id" not in gene_loading.columns:
        gene_loading = gene_loading.rename(columns={"RowID": "program_id"})

    regulators = pd.read_csv(args.regulators)
    regulators["significant"] = (
        regulators["significant"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )

    targets = pd.read_csv(args.targets, sep="\t")
    screen_targets = set(targets["target_name"].dropna().astype(str))

    coordinates = load_gene_coordinates(args.gene_coordinates)

    activity = pd.read_csv(args.activity_by_condition) if args.activity_by_condition else None
    if "--stage-markers" in sys.argv:
        warn_once("--stage-markers is deprecated; use --condition-markers")
    condition_markers = None
    if args.condition_markers:
        condition_markers = {}
        for key, entry in json.loads(args.condition_markers.read_text()).items():
            if key.startswith("_"):
                continue
            if "description" not in entry and "stage" in entry:
                warn_once("marker-file key 'stage' is deprecated; rename it to 'description'")
                entry["description"] = entry.pop("stage")
            condition_markers[key] = entry
    screens = build_screens(
        gene_loading, coordinates, screen_targets, regulators, activity, condition_markers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(screens, indent=2), encoding="utf-8")
    print(f"wrote screens for {len(screens)} programs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
