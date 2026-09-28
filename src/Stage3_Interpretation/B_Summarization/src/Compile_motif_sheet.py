"""Excel sheets for program TF-motif enrichment (Stage 2 motif_enrichment + candidate_tfs tables).

Kept apart from Compile_excel_sheet.py so it imports with pandas alone (that module pulls in
muon / mygene / sklearn at import time) and can be unit-tested without the full environment.

Inputs (TSV, written by Stage 2):
  motif_enrichment: program, element_type ('promoter'|'enhancer'), tf, [motif_family,] pvalue, fdr,
                    enrichment, n_program_genes_tested, n_background_genes, mean_count_program,
                    mean_count_background, significant (fdr < 0.05 & enrichment > 1)
                    (`tf` = the tested motif: a MotifCompendium cluster such as KLF-SP_0, a HOCOMOCO
                    TF, or a Fi-NeMo pattern's matched cluster; `motif_family` groups them, e.g. KLF-SP)
  candidate_tfs:    program, element_type, tf, [motif_family,] tf_gene_symbol, fdr, enrichment, tf_expressed,
                    tf_program_loading_rank, tf_in_top_program_genes,
                    tf_knockdown_regulates_program, knockdown_log2fc, knockdown_fdr, evidence_tier
                    (Stage2_Evaluation/A_Metrics/src/nominate_candidate_tfs.py)
  Both tables carry a `motif_source` column ('fimo' | 'finemo') when Stage 2 ran with
  --motif_source both. Sources are then kept apart (FIMO first): the summary gets one set of
  columns per element type x source. With one source (or no column) the sheets are unchanged.
"""
from __future__ import annotations

import json
import os

import pandas as pd

ELEMENT_TYPES = ("promoter", "enhancer")
# Strongest evidence first; used to sort the candidate-TF sheet.
# Same order as Stage2_Evaluation/A_Metrics/src/nominate_candidate_tfs.py EVIDENCE_TIER_ORDER.
EVIDENCE_TIER_ORDER = ("motif+regulator", "motif+expressed_in_program", "motif+expressed", "motif_only")
MOTIF_SOURCE_ORDER = ("fimo", "finemo")   # FIMO (MotifCompendium / HOCOMOCO) first, then Fi-NeMo (ChromBPNet)


def list_motif_sources(df: pd.DataFrame) -> list:
    """Motif sources to keep apart, FIMO first; [None] when there is one source or no column."""
    if "motif_source" not in df.columns or df["motif_source"].nunique() < 2:
        return [None]
    present = list(df["motif_source"].astype(str).unique())
    return sorted(present, key=lambda s: (MOTIF_SOURCE_ORDER.index(s) if s in MOTIF_SOURCE_ORDER else len(MOTIF_SOURCE_ORDER), s))


def source_sort_rank(df: pd.DataFrame) -> pd.Series:
    """Sort key putting FIMO rows before Fi-NeMo rows (0 for every row when there is no column)."""
    if "motif_source" not in df.columns:
        return pd.Series(0, index=df.index)
    return df["motif_source"].map({s: i for i, s in enumerate(MOTIF_SOURCE_ORDER)}).fillna(len(MOTIF_SOURCE_ORDER))


def as_bool(values: pd.Series) -> pd.Series:
    """True/False from bools or their string forms ('True', 'true', '1', 'yes')."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_motif_method(Motif_path) -> str:
    """'ttest' or 'correlation': `arguments.motif_method` of the Stage 2 `{K}_motif_enrichment_config.yml`
    next to the table (JSON, written by run_motif_enrichment.py); 'ttest' (the default) if absent."""
    config_path = os.path.splitext(str(Motif_path))[0] + "_config.yml"
    if not os.path.isfile(config_path):
        return "ttest"
    try:
        with open(config_path) as handle:
            return json.load(handle).get("arguments", {}).get("motif_method") or "ttest"
    except ValueError:
        return "ttest"


def read_motif_enrichment(Motif_path, fdr_threshold=0.05, method="ttest"):
    """Read the long motif-enrichment table; derive `significant` if the file lacks it
    (fdr < threshold and enrichment > 1; for the correlation method, r > 0)."""
    df = pd.read_csv(Motif_path, sep="\t")
    df["element_type"] = df["element_type"].astype(str).str.lower()
    if "significant" in df.columns:
        df["significant"] = as_bool(df["significant"])
    else:
        df["significant"] = (df["fdr"] < fdr_threshold) & (df["enrichment"] > (0 if method == "correlation" else 1))
    return df


def format_motif_hits(rows: pd.DataFrame, method: str = "ttest") -> str:
    """'KLF2 (2.10x, FDR 3.0e-05); SOX17 (...)' in the order given; 'KLF2 (r=0.12, FDR ...)' for the
    correlation method (`enrichment` holds the correlation r)."""
    effect = "r={:.2f}" if method == "correlation" else "{:.2f}x"
    return "; ".join(
        f"{tf} ({effect.format(enrichment)}, FDR {fdr:.1e})"
        for tf, enrichment, fdr in zip(rows["tf"], rows["enrichment"], rows["fdr"])
    )


def format_motif_families(rows: pd.DataFrame) -> str:
    """'KLF-SP (4); GATA (1)': motif families of the significant rows in order of their best motif (rows
    sorted by FDR), with the number of significant motifs of each family."""
    counts = rows.groupby("motif_family", sort=False).size()
    return "; ".join(f"{family} ({n})" for family, n in counts.items())


def Compile_Motif_sheet(Motif_path, top_n=5, fdr_threshold=0.05, method=None):
    """Per-program motif summary plus the significant rows.

    `method` ('ttest' | 'correlation') = the Stage 2 test; None reads it from the Stage 2 config next
    to the table (read_motif_method). With 'correlation' the effect is shown as "r=0.12".

    Returns (df_summary, df_significant):
      df_summary     one row per program (index `program_name`): for each element type the number
                     of significant motifs, the number tested, and the top `top_n` significant
                     motifs by FDR (ties: higher enrichment first) as "TF (enrichment x, FDR)"; with a
                     `motif_family` column also `families_{element type}_motifs`: every family of
                     the significant motifs, best first, with its motif count ("KLF-SP (4); GATA (1)").
                     With several motif sources, one such set per element type x source, the
                     column names suffixed with the source (e.g. `n_significant_promoter_finemo_motifs`).
      df_significant significant rows only (fdr < 0.05 and enrichment > 1), sorted by program,
                     (source,) element type, FDR. The full table stays in the Stage 2 TSV.
    """
    print('Load motif enrichment data')

    method = method or read_motif_method(Motif_path)
    df = read_motif_enrichment(Motif_path, fdr_threshold=fdr_threshold, method=method)
    significant = df[df["significant"]].assign(source_rank=source_sort_rank).sort_values(
        ["program", "source_rank", "element_type", "fdr", "enrichment"],
        ascending=[True, True, True, True, False]).drop(columns="source_rank")

    summary = pd.DataFrame(index=pd.Index(sorted(df["program"].unique(), key=str), name="program_name"))
    for source in list_motif_sources(df):
        in_source = (lambda t: t) if source is None else (lambda t: t[t["motif_source"].astype(str) == source])
        suffix = "" if source is None else f"_{source}"
        for element_type in ELEMENT_TYPES:
            typed = in_source(df[df["element_type"] == element_type])
            hits = in_source(significant[significant["element_type"] == element_type])
            summary[f"n_significant_{element_type}{suffix}_motifs"] = hits.groupby("program").size()
            summary[f"n_tested_{element_type}{suffix}_motifs"] = typed.groupby("program")["tf"].nunique()
            top_hits = hits.groupby("program", sort=False).head(top_n)
            summary[f"top{top_n}_{element_type}{suffix}_motifs"] = pd.Series(
                {program: format_motif_hits(rows, method) for program, rows in top_hits.groupby("program")}, dtype=object)
            if "motif_family" in hits.columns:
                summary[f"families_{element_type}{suffix}_motifs"] = pd.Series(
                    {program: format_motif_families(rows) for program, rows in hits.groupby("program")}, dtype=object)
    count_columns = [c for c in summary.columns if c.startswith("n_")]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    summary = summary.fillna("")

    df_significant = significant.set_index("program").rename_axis("program_name")
    return summary, df_significant


def Compile_Candidate_TF_sheet(Candidate_TF_path):
    """Candidate TFs (enriched motif x expression / knockdown evidence), strongest tier first.

    Sorted by program, evidence tier (motif+regulator, motif+expressed_in_program,
    motif+expressed, motif_only), motif source (FIMO first, when present), then motif FDR.
    Index `program_name`.
    """
    print('Load candidate TF data')

    df = pd.read_csv(Candidate_TF_path, sep="\t")
    for column in ("tf_expressed", "tf_in_top_program_genes", "tf_perturbed", "tf_knockdown_regulates_program"):
        if column in df.columns:
            df[column] = as_bool(df[column])
    tier_rank = {tier: i for i, tier in enumerate(EVIDENCE_TIER_ORDER)}
    df["tier_rank"] = df["evidence_tier"].map(tier_rank).fillna(len(tier_rank))
    df["source_rank"] = source_sort_rank(df)
    df = df.sort_values(["program", "tier_rank", "source_rank", "fdr"]).drop(columns=["tier_rank", "source_rank"])
    return df.set_index("program").rename_axis("program_name")
