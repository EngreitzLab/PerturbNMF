"""Deterministic confounder screens per program, written to <run_dir>/Programs/P<k>/confounder_screens.json.

Ported from ProgramAnnotatorV3/scripts/build_confounder_screens.py (branch annotation-v3-annotator-viewer).
The screens answer "is there a non-biological or non-specific reason these genes co-vary?" from the data
alone, before any LLM sees the program. Every screen reports an observed count, the background expectation
and a p-value, because "12 of the top 50 genes are ribosomal" means nothing without "and ribosomal genes
are 4% of the background".

    python build_confounder_screens.py --run_dir <out_dir>/<run_name>/Annotation --gtf_path genes.gtf.gz

Reads <run_dir>/Programs/meta.json (h5mu, perturbation files, keys, membership_top and fdr of
prepare_programs.py) and writes, per program folder, confounder_screens.json

    {label: {families, primary_family, positional_flag, ...}, screens: {...}, params}

  screens.symbol_families   7 symbol-prefix families (ribosomal, mitochondrial, histone, snoRNA, ...)
  screens.marker_sets       6 marker sets (cell cycle S / G2M, heat shock, ISR, UPR, interferon)
  screens.positional        top chromosome (binomial test) and the densest --window_bp window
  screens.gene_biotypes, cis_targets_in_top_genes, regulators, condition_composition (--condition_markers)

The top genes are the program's top --top_n_genes by loading; the background is the run's own gene universe
(every program's top membership_top genes), not the genome. A family or marker set is called when at least
--min_genes top genes hit it with hypergeometric p <= --max_p; primary_family is the most significant one.
The confounder_screens step of each program's plan.md is ticked once the file passes its checks.
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import mudata as mu
import pandas as pd
from scipy.stats import binomtest, hypergeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from program_inputs import load_cell_usage, load_condition, load_gene_spectra, load_perturbation  # noqa: E402

from program_dir import list_programs, mark_done  # noqa: E402

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


# coordinates
def load_gene_coordinates(path):
    """Branch gene_coordinates.tsv (no header: name chrom start end strand gene_type) -> {symbol: record},
    keeping the first record per symbol."""
    coordinates = {}
    with Path(path).open() as handle:
        for line in handle:
            name, chrom, start, end, _strand, gene_type = line.rstrip("\n").split("\t")
            coordinates.setdefault(name, {"chrom": chrom, "start": int(start), "end": int(end),
                                          "gene_type": gene_type})
    return coordinates


_ATTR = re.compile(r'(gene_name|gene_type|gene_biotype) "([^"]+)"')


def load_gtf_coordinates(path):
    """{symbol: {chrom, start, end, gene_type}} from the 'gene' rows of a GTF (.gtf or .gtf.gz),
    first record per symbol (rRNA repeats and PAR genes appear on several contigs)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    coordinates = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split("\t", 8)
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = dict(_ATTR.findall(fields[8]))
            name = attrs.get("gene_name")
            if name and name not in coordinates:
                coordinates[name] = {"chrom": fields[0], "start": int(fields[3]), "end": int(fields[4]),
                                     "gene_type": attrs.get("gene_type") or attrs.get("gene_biotype") or "unknown"}
    return coordinates


# screens (branch code)
def score_positional_concentration(genes, coordinates, background, n_background, window_bp):
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
            window = [g for s, g in entries[i:] if s - start <= window_bp]
            if len(window) > best_window["count"]:
                best_window = {"chrom": chrom, "count": len(window), "start": start,
                               "end": start + window_bp, "genes": window}

    return {
        "assessable": True,
        "n_genes_located": len(located),
        "top_chromosome": top_chrom,
        "genes_on_top_chromosome": top_chrom_count,
        "top_chromosome_genes": [g for _, g in sorted(by_chrom[top_chrom])],   # in position order
        "expected_on_top_chromosome": round(expected_fraction * len(located), 1),
        "fold_enrichment": round((top_chrom_count / len(located)) / expected_fraction, 2) if expected_fraction else None,
        "binomial_p": f"{binomial.pvalue:.2e}",
        "densest_window": best_window,
    }


def score_gene_set_overlap(genes, gene_set, background):
    """Hypergeometric overlap of the top genes with a marker set, within the run's own universe.

    The universe is the program-gene universe, not the genome: asking whether ribosomal genes
    are enriched among the top 50 relative to all human genes would call almost every program
    ribosomal, because the background is not what was measured.
    """
    # The marker sets and prefix patterns are human symbols; compare upper-cased so a mouse
    # screen (Mki67, Rpl13, mt-Co1) is screened too instead of silently scoring zero.
    upper_background = {g.upper(): g for g in background}
    set_in_background = {upper_background[g.upper()] for g in gene_set if g.upper() in upper_background}
    overlap = sorted(set(genes) & set_in_background)
    if not set_in_background:
        return {"n_overlap": 0, "genes": [], "expected": 0.0, "p_value": None}
    expected = len(genes) * len(set_in_background) / len(background)
    p_value = hypergeom.sf(len(overlap) - 1, len(background), len(set_in_background), len(genes))
    return {"n_overlap": len(overlap), "genes": overlap, "expected": round(expected, 2),   # every hit (<= top_n_genes)
            "p_value": f"{p_value:.2e}" if len(overlap) else None}


def score_condition_composition(by_condition, top_genes, condition_markers, background):
    """Does this program just track which cells are present in a given condition?

    When conditions differ in cell composition (e.g. timepoints of a differentiation), a program
    can look like a coherent pathway when it is really the transcriptome difference between two
    cell populations. Two deterministic signals of that: activity concentrated in one condition
    (or, for ordered conditions, one contiguous block), and top genes that are the canonical
    markers of that condition's cells. Neither alone is decisive; both together point to
    composition rather than regulation.
    """
    order = list(condition_markers)
    by_condition = by_condition.reindex(order)
    total = float(by_condition.sum())
    ranked = by_condition.sort_values(ascending=False)
    second = float(ranked.iloc[1]) if len(ranked) > 1 else 0.0
    lowest = float(ranked.iloc[-1])
    markers = {c: score_gene_set_overlap(top_genes, set(e["genes"]), background) | {"description": e["description"]}
               for c, e in condition_markers.items()}
    return {
        "condition_order": order,
        "mean_score": {c: round(float(v), 5) for c, v in by_condition.items()},
        "share_of_total": {c: round(float(v) / total, 3) if total else None for c, v in by_condition.items()},
        "peak_condition": str(by_condition.idxmax()),
        "peak_share": round(float(by_condition.max()) / total, 3) if total else None,
        "peak_over_second": round(float(by_condition.max()) / second, 2) if second else None,
        "peak_over_lowest": round(float(by_condition.max()) / lowest, 2) if lowest else None,
        "marker_overlap": markers,
    }


def screen_program(top_genes, background, coordinates, background_chroms, n_located, screen_targets,
                   reg_counts, condition_mean, condition_markers, args):
    """Every screen for one program (branch build_screens body; prefix families also get a p-value)."""
    family_hits = {}
    for family, pattern in PREFIX_FAMILIES.items():
        members = {g for g in background if pattern.search(g.upper())}
        family_hits[family] = score_gene_set_overlap(top_genes, members, background)
    cis = sorted(set(top_genes) & screen_targets)
    screens = {
        "n_top_genes_screened": len(top_genes),
        "top_genes": top_genes,
        "positional": score_positional_concentration(top_genes, coordinates, background_chroms, n_located,
                                                     args.window_bp),
        "gene_biotypes": dict(Counter(coordinates[g]["gene_type"] for g in top_genes if g in coordinates).most_common()),
        "symbol_families": family_hits,
        "marker_sets": {name: score_gene_set_overlap(top_genes, s, background) for name, s in NAMED_SETS.items()},
        "cis_targets_in_top_genes": {
            "n_overlap": len(cis),
            "genes": cis[:15],
            "note": "top genes that are themselves CRISPRi targets in this screen",
        },
        "regulators": reg_counts,
    }
    if condition_mean is not None and condition_markers:
        screens["condition_composition"] = score_condition_composition(condition_mean, top_genes,
                                                                       condition_markers, background)
    return screens


def label_program(screens, args):
    """families: every family / marker set with >= min_genes top genes and p <= max_p (most significant first);
    primary_family: the first of them or 'none'; positional_flag: the top chromosome when it is enriched."""
    hits = []
    for group in ("symbol_families", "marker_sets"):
        for name, s in screens[group].items():
            if s["n_overlap"] >= args.min_genes and s["p_value"] is not None and float(s["p_value"]) <= args.max_p:
                hits.append((float(s["p_value"]), name, s["n_overlap"]))
    hits.sort()
    pos = screens["positional"]
    flagged = (pos.get("assessable") and float(pos["binomial_p"]) <= args.max_p
               and (pos.get("fold_enrichment") or 0) >= args.min_fold)
    return {
        "families": [name for _, name, _ in hits],
        "family_details": [{"family": name, "n_genes": n, "p_value": f"{p:.2e}"} for p, name, n in hits],
        "primary_family": hits[0][1] if hits else "none",
        "positional_flag": pos["top_chromosome"] if flagged else None,
        "positional_detail": ({"chrom": pos["top_chromosome"], "n_genes": pos["genes_on_top_chromosome"],
                               "expected": pos["expected_on_top_chromosome"], "fold": pos["fold_enrichment"],
                               "binomial_p": pos["binomial_p"], "densest_window": pos["densest_window"]}
                              if flagged else None),
    }


# inputs
def load_meta(run_dir):
    path = Path(run_dir) / "Programs" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run prepare_programs.py first")
    return json.loads(path.read_text())


def regulator_counts(assoc_by_cond, pid, fdr):
    """Perturbed genes tested against the program and significant (adj_pval < fdr) in any condition."""
    rows = pd.concat([df[df["program_name"] == pid] for df in assoc_by_cond.values()])
    return {"n_tested": int(rows["target_name"].nunique()),
            "n_significant": int(rows.loc[rows["adj_pval"] < fdr, "target_name"].nunique())}


def load_condition_markers(path):
    if not path:
        return None
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if not k.startswith("_")}


def build_parser():
    p = argparse.ArgumentParser(description="Confounder screens (symbol families, marker sets, chromosome position) per program, written to each program folder.")

    # IO
    p.add_argument("--run_dir", required=True, help="Annotation run folder of prepare_programs.py (Programs/meta.json, Programs/P<k>/).")
    coords = p.add_mutually_exclusive_group(required=True)
    coords.add_argument("--gtf_path", help="GTF (.gtf or .gtf.gz) used for alignment; its 'gene' rows give chromosome, position and gene_type.")
    coords.add_argument("--gene_coordinates", help="Instead of --gtf_path: TSV without header, name chrom start end strand gene_type.")
    p.add_argument("--condition_markers", default=None, help="JSON {condition: {description, genes}} of canonical marker genes per condition; adds the condition_composition screen. Default: off.")

    # context info
    p.add_argument("--programs", type=int, nargs="+", default=None, help="Program ids, space separated (e.g. 1 2 3). Default: every program folder.")

    # screen
    p.add_argument("--top_n_genes", type=int, default=50, help="Top genes per program (by loading) that are screened.")
    p.add_argument("--window_bp", type=int, default=10_000_000, help="Window size of the densest-window positional screen.")
    p.add_argument("--min_genes", type=int, default=3, help="Min top genes in a family / marker set to call it.")
    p.add_argument("--max_p", type=float, default=1e-3, help="Max hypergeometric (families) / binomial (chromosome) p to call a hit.")
    p.add_argument("--min_fold", type=float, default=2.0, help="Min top-chromosome fold enrichment for positional_flag.")
    return p


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    meta = load_meta(run_dir)
    keys, params = meta["keys"], meta["params"]
    fdr, membership_top = params.get("fdr", 0.05), params.get("membership_top", 300)

    prog_adata = mu.read_h5ad(meta["inputs"]["mdata_path"], mod=keys["prog_key"])
    spectra = load_gene_spectra(prog_adata, keys["loadings_key"], keys["gene_name_key"])
    print(f"[load] spectra: {spectra.shape[0]} programs x {spectra.shape[1]} genes")
    ranks = spectra.rank(axis=1, ascending=False, method="first")
    background = set(spectra.columns[(ranks <= membership_top).any(axis=0).values])
    print(f"[load] gene universe: {len(background):,} genes (union of every program's top {membership_top})")

    coordinates = load_gtf_coordinates(args.gtf_path) if args.gtf_path else load_gene_coordinates(args.gene_coordinates)
    background_chroms = Counter(coordinates[g]["chrom"] for g in background if g in coordinates)
    n_located = sum(background_chroms.values())
    print(f"[load] coordinates: {len(coordinates):,} genes; {n_located:,} of the universe located")

    assoc_by_cond = {c: load_perturbation(path) for c, path in meta["inputs"]["perturbation_path"].items()}
    screen_targets = set(pd.concat(assoc_by_cond.values())["target_name"].dropna().astype(str))

    condition_markers = load_condition_markers(args.condition_markers)
    condition_means = None
    if condition_markers:
        labels, _ = load_condition(prog_adata, keys["categorical_key"])
        condition_means = load_cell_usage(prog_adata).groupby(labels.values).mean()   # conditions x programs

    folders = {int(d.name[1:]): d for d in list_programs(run_dir)}
    programs = list(dict.fromkeys(args.programs or folders))
    summary = {}
    for pid in programs:
        d = folders.get(pid)
        if d is None:
            print(f"  P{pid}: no program folder in {run_dir / 'Programs'}; skipped")
            continue
        top_genes = spectra.loc[pid].sort_values(ascending=False).index[:args.top_n_genes].tolist()
        screens = screen_program(top_genes, background, coordinates, background_chroms, n_located, screen_targets,
                                 regulator_counts(assoc_by_cond, pid, fdr),
                                 condition_means[pid] if condition_means is not None else None,
                                 condition_markers, args)
        block = {"label": label_program(screens, args), "screens": screens,
                 "params": {k: getattr(args, k) for k in ("top_n_genes", "window_bp", "min_genes", "max_p", "min_fold")}
                           | {"membership_top": membership_top, "fdr": fdr, "n_background": len(background)}}
        (d / "confounder_screens.json").write_text(json.dumps(block, indent=2))
        errors = mark_done(d, "confounder_screens", by="build_confounder_screens.py")
        lab = block["label"]
        summary[f"P{pid}"] = {"primary_family": lab["primary_family"], "families": lab["families"],
                              "positional_flag": lab["positional_flag"]}
        print(f"  P{pid}: primary={lab['primary_family']} families={lab['families'] or '-'} "
              f"positional={lab['positional_flag'] or '-'}" + (f"  [FAILED: {errors}]" if errors else ""))

    (run_dir / "Programs" / "meta_confounder_screens.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"run_dir": str(run_dir), "gtf_path": args.gtf_path, "gene_coordinates": args.gene_coordinates,
                   "condition_markers": args.condition_markers},
        "params": {k: getattr(args, k) for k in ("top_n_genes", "window_bp", "min_genes", "max_p", "min_fold")}
                  | {"membership_top": membership_top, "fdr": fdr},
        "n_background": len(background),
        "programs": summary,
    }, indent=2))
    n_family = sum(v["primary_family"] != "none" for v in summary.values())
    n_pos = sum(bool(v["positional_flag"]) for v in summary.values())
    print(f"[done] {len(summary)} program(s): {n_family} with a family, {n_pos} positional -> {run_dir / 'Programs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
