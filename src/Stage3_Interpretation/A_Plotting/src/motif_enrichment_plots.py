"""Program TF-motif enrichment plots.

Consumes the long results table produced by ``Stage2_Evaluation/A_Metrics/src/motif_enrichment.py``
(columns: program, element_type ['promoter'|'enhancer'], tf, pvalue, fdr, enrichment,
n_program_genes_tested, mean_count_program, mean_count_background, significant) and, optionally,
the ``candidate_tfs`` table (program, tf, element_type, fdr, enrichment, tf_in_top_program_genes,
tf_program_loading_rank, tf_knockdown_regulates_program, knockdown_log2fc, knockdown_fdr).

Style references:
  - Schnitzler et al. Nature 2024, Extended Data Fig. 4: per-program panel with two small
    horizontal bar charts (promoter, enhancer), each showing the top ~5 TF motifs ranked by
    -log10(FDR), TF names in italics on the y-axis. Extended Data Fig. 3 shows a simple
    per-program bar chart of the *count* of enriched motifs (gray=promoter, blue=enhancer)
    ordered across all programs.
  - The legacy ``plot_motif_per_program`` / ``plot_all_days_motif`` functions formerly in
    ``Program_QC_plots.py``: the same ranked horizontal-bar idea, one plot per program. The
    cross-program plot here is a dot heatmap (rows = programs, columns = TF/motif clusters,
    dot size = -log10 FDR, color = enrichment).

With ``--motif_source both`` the tables carry a ``motif_source`` column ('fimo' | 'finemo'); the
sources are never pooled: per-program plots get one row of panels per source (FIMO first) and the
heatmap takes a ``motif_source``. With one source (or no column) the plots are unchanged.

Save PDF + PNG via matplotlib/seaborn only. No plotly (kept cheap enough for 60+ programs / calls
from an HTML-report hook).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

RESULT_COLUMNS = [
    "program", "element_type", "tf", "pvalue", "fdr", "enrichment",
    "n_program_genes_tested", "mean_count_program", "mean_count_background", "significant",
]

CANDIDATE_COLUMNS = [
    "program", "tf", "element_type", "fdr", "enrichment", "tf_in_top_program_genes",
    "tf_program_loading_rank", "tf_knockdown_regulates_program", "knockdown_log2fc", "knockdown_fdr",
]

ELEMENT_TYPE_ORDER = ["promoter", "enhancer"]
ELEMENT_TYPE_COLOR = {"promoter": "#808080", "enhancer": "#1c5f8f"}  # gray, blue (ED Fig. 3 palette)
MOTIF_SOURCE_ORDER = ["fimo", "finemo"]
MOTIF_SOURCE_LABEL = {"fimo": "FIMO", "finemo": "Fi-NeMo"}   # FIMO database: MotifCompendium (default) or HOCOMOCO


def list_motif_sources(table: Optional[pd.DataFrame]) -> list:
    """Motif sources to plot apart, FIMO first; [None] when there is one source or no column."""
    if table is None or "motif_source" not in table.columns or table["motif_source"].nunique() < 2:
        return [None]
    present = list(table["motif_source"].astype(str).unique())
    return sorted(present, key=lambda s: (MOTIF_SOURCE_ORDER.index(s) if s in MOTIF_SOURCE_ORDER
                                          else len(MOTIF_SOURCE_ORDER), s))


def rows_of_source(table: pd.DataFrame, source: Optional[str]) -> pd.DataFrame:
    """Rows of one motif source (all rows for source None)."""
    if source is None or "motif_source" not in table.columns:
        return table
    return table[table["motif_source"].astype(str) == source]


def save_figure(fig, save_path: Optional[str], save_name: Optional[str], formats=("pdf", "png"), dpi=200):
    """Save fig as PDF + PNG under save_path/save_name.{ext}, if both are given."""
    if not (save_path and save_name):
        return
    os.makedirs(save_path, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(save_path, f"{save_name}.{ext}"), format=ext, bbox_inches="tight", dpi=dpi)


def candidate_tf_set(candidate_tfs: Optional[pd.DataFrame], program, element_type, source=None) -> set:
    if candidate_tfs is None or len(candidate_tfs) == 0:
        return set()
    sub = candidate_tfs[(candidate_tfs["program"] == program) & (candidate_tfs["element_type"] == element_type)]
    return set(rows_of_source(sub, source)["tf"])


def plot_program_motif_ranks(
    results: pd.DataFrame,
    program,
    candidate_tfs: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    x_value: str = "neglog10_fdr",
    require_enriched: bool = True,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    show: bool = False,
    ax_size=(3.2, 3.2),
):
    """Per-program ranked motif plot: promoter and enhancer top-N TF motifs side by side.

    Bars are ranked by ``x_value`` (``'neglog10_fdr'`` default, or ``'enrichment'``) among rows with
    ``enrichment > 1`` (``require_enriched=True``, default) — the paper's own significance
    definition excludes depleted motifs (enrichment<1), which otherwise dominate the smallest-FDR
    ranking. Bars are colored gray/steel-blue by element type (ED Fig. 3 palette) and outlined in
    red for TFs that also appear in ``candidate_tfs`` for that program/element_type.
    Cheap enough to call per-program for 60+ programs (small figure, no clustering).

    Returns the ``matplotlib.figure.Figure``.
    """
    sub = results[results["program"] == program].copy()
    sources = list_motif_sources(results)
    fig, axes = plt.subplots(len(sources), 2, figsize=(ax_size[0] * 2, ax_size[1] * len(sources)),
                             sharey=False, squeeze=False)
    panels = [(axes[row, col], element_type, source) for row, source in enumerate(sources)
              for col, element_type in enumerate(ELEMENT_TYPE_ORDER)]

    for ax, element_type, source in panels:
        title = element_type.capitalize() + ("" if source is None else f" — {MOTIF_SOURCE_LABEL.get(source, source)}")
        et_df = rows_of_source(sub[sub["element_type"] == element_type], source).copy()
        if require_enriched:
            et_df = et_df[et_df["enrichment"] > 1]
        if et_df.empty:
            ax.axis("off")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
            continue

        et_df["neglog10_fdr"] = -np.log10(et_df["fdr"].clip(lower=1e-300))
        et_df = et_df.sort_values(x_value, ascending=False).head(top_n)
        et_df = et_df.iloc[::-1]  # largest bar at top

        candidates = candidate_tf_set(candidate_tfs, program, element_type, source)
        colors = [ELEMENT_TYPE_COLOR[element_type]] * len(et_df)
        edgecolors = ["#c0392b" if tf in candidates else "none" for tf in et_df["tf"]]
        linewidths = [1.6 if tf in candidates else 0 for tf in et_df["tf"]]

        ax.barh(range(len(et_df)), et_df[x_value].values, color=colors,
                edgecolor=edgecolors, linewidth=linewidths, alpha=0.85)
        ax.set_yticks(range(len(et_df)))
        ax.set_yticklabels(et_df["tf"], fontsize=9, fontstyle="italic")
        ax.set_xlabel("-log10(FDR)" if x_value == "neglog10_fdr" else "Enrichment", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        if x_value == "neglog10_fdr":
            ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    fig.suptitle(f"Program {program} — top TF motifs", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, save_path, save_name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def select_programs_and_tfs(results: pd.DataFrame, programs, element_type, top_n_per_program, motif_source=None):
    sub = rows_of_source(results[results["element_type"] == element_type], motif_source).copy()
    if programs is not None:
        sub = sub[sub["program"].isin(programs)]
    sub["neglog10_fdr"] = -np.log10(sub["fdr"].clip(lower=1e-300))
    # Rank candidate rows by the paper's own significance definition (fdr<0.05 & enrichment>1)
    # so the heatmap doesn't highlight "top" TFs that are actually depleted (enrichment<1).
    if "significant" in sub.columns:
        ranked = sub[sub["significant"].astype(bool)]
    else:
        ranked = sub[sub["enrichment"] > 1]
    if ranked.empty:
        ranked = sub
    top_tfs = (
        ranked.sort_values("neglog10_fdr", ascending=False)
        .groupby("program", sort=False)
        .head(top_n_per_program)["tf"]
        .unique()
    )
    return sub, list(top_tfs)


def plot_motif_program_heatmap(
    results: pd.DataFrame,
    programs: Optional[Sequence] = None,
    element_type: str = "promoter",
    top_n_per_program: int = 5,
    cluster: bool = True,
    fdr_threshold: float = 0.05,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    show: bool = False,
    figsize: Optional[tuple] = None,
    motif_source: Optional[str] = None,
):
    """Cross-program TF-motif dot heatmap.

    Rows = TF motifs (union of the top ``top_n_per_program`` TFs per selected program, ranked by
    -log10 FDR), columns = programs (``programs`` selects/orders a subset; default = all programs
    with any TF passing the filter). Dot color = enrichment, dot size = -log10(FDR); TF x program
    pairs with fdr >= 1 (no data) are left blank. Rows are hierarchically clustered by default so
    co-regulated TFs group together.
    ``motif_source`` ('fimo' | 'finemo') selects one source of a ``--motif_source both`` table; it
    defaults to the first source (FIMO), so the two are never averaged into one dot.

    Returns the ``matplotlib.figure.Figure``.
    """
    if motif_source is None:
        motif_source = list_motif_sources(results)[0]
    sub, tf_order = select_programs_and_tfs(results, programs, element_type, top_n_per_program, motif_source)
    program_order = list(programs) if programs is not None else sorted(sub["program"].unique(), key=str)

    enrich_mat = sub.pivot_table(index="tf", columns="program", values="enrichment", aggfunc="mean")
    fdr_mat = sub.pivot_table(index="tf", columns="program", values="fdr", aggfunc="min")
    enrich_mat = enrich_mat.reindex(index=tf_order, columns=program_order)
    fdr_mat = fdr_mat.reindex(index=tf_order, columns=program_order)

    if cluster and len(tf_order) > 2:
        filled = enrich_mat.fillna(1.0).values
        try:
            link = linkage(filled, method="average", metric="euclidean")
            order = dendrogram(link, no_plot=True)["leaves"]
            tf_order = [tf_order[i] for i in order]
            enrich_mat = enrich_mat.reindex(index=tf_order)
            fdr_mat = fdr_mat.reindex(index=tf_order)
        except Exception:
            pass

    figsize = figsize or (max(4, 0.45 * len(program_order) + 1.5), max(3, 0.28 * len(tf_order) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    neglog10_fdr = -np.log10(fdr_mat.clip(lower=1e-300))
    max_size = 220
    size_norm = np.clip(neglog10_fdr / max(neglog10_fdr.max().max(), -np.log10(fdr_threshold)), 0, 1)

    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=min(0.5, np.nanmin(enrich_mat.values) if np.isfinite(np.nanmin(enrich_mat.values)) else 0.5),
                                 vmax=max(1.5, np.nanmax(enrich_mat.values) if np.isfinite(np.nanmax(enrich_mat.values)) else 1.5))
    cmap = sns.color_palette("vlag", as_cmap=True)

    for i, tf in enumerate(tf_order):
        for j, prog in enumerate(program_order):
            fdr_val = fdr_mat.loc[tf, prog]
            if pd.isna(fdr_val):
                continue
            enrich_val = enrich_mat.loc[tf, prog]
            size = 12 + size_norm.loc[tf, prog] * max_size
            ax.scatter(j, i, s=size, c=[cmap(norm(enrich_val))], edgecolor="black", linewidth=0.3, zorder=3)
            if fdr_val < fdr_threshold and enrich_val > 1:
                stars = "***" if fdr_val < 0.001 else ("**" if fdr_val < 0.01 else "*")
                ax.text(j, i, stars, ha="center", va="center", fontsize=6, zorder=4,
                         color="white" if norm(enrich_val) > 0.7 or norm(enrich_val) < 0.3 else "black")

    ax.set_xticks(range(len(program_order)))
    ax.set_xticklabels([f"P{p}" for p in program_order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tf_order)))
    ax.set_yticklabels(tf_order, fontsize=8, fontstyle="italic")
    ax.set_xlim(-0.5, len(program_order) - 0.5)
    ax.set_ylim(-0.5, len(tf_order) - 0.5)
    ax.invert_yaxis()
    source_label = "" if motif_source is None else f" ({MOTIF_SOURCE_LABEL.get(motif_source, motif_source)})"
    ax.set_title(f"{element_type.capitalize()} motif enrichment across programs{source_label}", fontsize=12,
                 fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Enrichment", fontsize=9)

    fig.tight_layout()
    save_figure(fig, save_path, save_name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_candidate_tf_summary(
    candidate_tfs: pd.DataFrame,
    program,
    top_n: int = 15,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    show: bool = False,
):
    """Candidate-TF summary: enriched motif x TF expression x knockdown effect, per program.

    Three columns per TF row: (1) -log10(FDR) of the motif enrichment (bar, colored by element
    type), (2) whether the TF is expressed in the program / its loading rank (marker + text),
    (3) knockdown log2FC with a red/blue/gray dot depending on significance and direction,
    analogous to the volcano-style regulator highlighting used elsewhere in this pipeline.
    With several motif sources, each TF label names its source.

    Returns the ``matplotlib.figure.Figure``.
    """
    sub = candidate_tfs[candidate_tfs["program"] == program].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No candidate TFs for program {program}", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        save_figure(fig, save_path, save_name)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    sub["neglog10_fdr"] = -np.log10(sub["fdr"].clip(lower=1e-300))
    sub = sub.sort_values("neglog10_fdr", ascending=False).head(top_n)
    sub = sub.iloc[::-1]
    n = len(sub)

    fig, axes = plt.subplots(1, 3, figsize=(9, max(2.5, 0.35 * n + 1)), sharey=True,
                              gridspec_kw={"width_ratios": [1.1, 0.8, 1.1]})

    ax0 = axes[0]
    colors = [ELEMENT_TYPE_COLOR.get(et, "#808080") for et in sub["element_type"]]
    ax0.barh(range(n), sub["neglog10_fdr"].values, color=colors, alpha=0.85)
    ax0.set_yticks(range(n))
    tf_labels = sub["tf_gene_symbol"] if "tf_gene_symbol" in sub.columns else sub["tf"]
    if list_motif_sources(candidate_tfs) != [None]:   # one row per TF and source: say which source
        tf_labels = tf_labels.astype(str) + " (" + sub["motif_source"].map(
            lambda source: MOTIF_SOURCE_LABEL.get(source, source)) + ")"
    ax0.set_yticklabels(tf_labels, fontsize=9, fontstyle="italic")
    ax0.set_xlabel("-log10(FDR)\nmotif enrichment", fontsize=9)
    ax0.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = axes[1]
    expressed = sub["tf_in_top_program_genes"].astype(bool).values
    ranks = sub.get("tf_program_loading_rank", pd.Series([np.nan] * n))
    ax1.scatter(np.where(expressed, 1, 0), range(n),
                c=["#c0392b" if e else "#cbd5db" for e in expressed], s=60, zorder=3)
    for i, rank in enumerate(ranks):
        if pd.notna(rank):
            ax1.text(1.15, i, f"rank {int(rank)}", va="center", fontsize=7.5)
    ax1.set_xlim(-0.3, 2.0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["no", "yes"], fontsize=8, rotation=20)
    ax1.set_title("TF in top\nprogram genes", fontsize=9)
    for spine in ["top", "right", "left"]:
        ax1.spines[spine].set_visible(False)

    ax2 = axes[2]
    log2fc = sub["knockdown_log2fc"].values
    knockdown_fdr = sub.get("knockdown_fdr", pd.Series([np.nan] * n)).values
    regulates = sub["tf_knockdown_regulates_program"].astype(bool).values
    point_colors = []
    for reg, fc in zip(regulates, log2fc):
        if not reg or pd.isna(fc):
            point_colors.append("#cbd5db")
        elif fc > 0:
            point_colors.append("#c0392b")
        else:
            point_colors.append("#1c5f8f")
    ax2.scatter(np.nan_to_num(log2fc, nan=0.0), range(n), c=point_colors, s=60, zorder=3)
    ax2.axvline(0, color="black", linewidth=0.7, alpha=0.6)
    ax2.set_xlabel("Knockdown log2FC\non program", fontsize=9)
    ax2.set_title("Knockdown effect", fontsize=9)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.suptitle(f"Candidate TFs — program {program}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, save_path, save_name)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def motif_panel_png_path(output_dir: str, program) -> str:
    """Path convention for the per-program motif PNG, for the HTML-report embedding hook.

    Mirrors ``html_Program_QC_plots.py``'s ``program_{N}/images/umap.png`` convention:
    ``program_{N}/images/motif_ranks.png``. Not wired into the HTML report yet — call
    ``plot_program_motif_ranks(..., save_path=os.path.dirname(motif_panel_png_path(...)),
    save_name='motif_ranks')`` when generating a program's image folder, then reference the path
    with an ``<img>`` tag the same way ``umap.png`` is referenced.
    """
    return os.path.join(output_dir, f"program_{program}", "images", "motif_ranks.png")
