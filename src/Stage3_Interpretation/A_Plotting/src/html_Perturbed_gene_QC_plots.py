"""
Interactive HTML export for the per-gene comprehensive plot.

When cNMF_perturbed_gene_analysis.py is invoked with --output_format HTML, this
module renders one self-contained share folder containing:

  * gene_{SYMBOL}/gene_{SYMBOL}.html  : interactive Plotly page
  * gene_{SYMBOL}/metadata.json       : gene-level summary stats
  * gene_{SYMBOL}/data/*.json         : per-panel raw arrays
  * gene_{SYMBOL}/images/*.png        : matplotlib UMAPs + gene dotplot
  * index.html                        : card list linking all genes
  * shared/style.css                  : shared with the program-side share folder

Re-uses the styling, Plotly template, and most figure makers from
html_Program_QC_plots.py so the two share folders look identical.
"""
from __future__ import annotations

import os
import json
import math
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from .Perturbed_gene_QC_plots import (
    plot_umap_per_gene,
    plot_umap_per_gene_guide,
    perturbed_gene_dotplot,
)
from .html_Program_QC_plots import (
    PLOTLY_CDN, GOOGLE_FONTS,
    COLOR_UP, COLOR_DOWN, COLOR_NS, COLOR_NEUTRAL, COLOR_ACCENT,
    _PLOTLY_TEMPLATE, _STYLE_CSS,
    _sanitize, _write_json, _safe_sample_key,
    _fmt_layout, _empty_fig, _fig_to_div,
    _make_correlations_fig, _make_log2fc_fig, _make_volcano_fig, _make_waterfall_fig,
)


# ---------------------------------------------------------------------------
# Gene-specific data builders
# ---------------------------------------------------------------------------

def _build_top_programs(mdata, target_gene, top_n, ensembl_to_symbol_file, gene_name_key):
    """Top N programs by gene-loading score for one gene."""
    X = mdata["cNMF"].varm["loadings"]  # (programs x genes)
    if ensembl_to_symbol_file is None:
        if gene_name_key is not None and gene_name_key in mdata["rna"].var.columns:
            col_names = mdata["rna"].var[gene_name_key].astype(str).tolist()
        else:
            col_names = list(mdata["rna"].var_names)
    else:
        from .utilities import rename_list_gene_dictionary
        col_names = rename_list_gene_dictionary(list(mdata["rna"].var_names), ensembl_to_symbol_file)
    df = pd.DataFrame(data=X, columns=col_names, index=mdata["cNMF"].var_names)
    if target_gene not in df.columns:
        return {"programs": [], "loadings": []}
    matching_cols = (df.columns == target_gene).sum()
    if matching_cols > 1:
        raise ValueError(
            f"Gene symbol {target_gene!r} maps to {matching_cols} columns in the "
            f"loading matrix (duplicate symbols from multiple Ensembl IDs)."
        )
    n_programs = df.shape[0]
    if top_n > n_programs:
        raise ValueError(
            f"top_n={top_n} exceeds the number of programs available "
            f"in the loading matrix ({n_programs})."
        )
    series = df[target_gene].nlargest(top_n).sort_values(ascending=True)
    return {
        "programs": [str(p) for p in series.index],
        "loadings": [float(x) for x in series.values],
    }


def _build_gene_correlations(corr_matrix, target_gene, top_n):
    """Top ± correlated genes from the gene × gene correlation matrix."""
    if target_gene not in corr_matrix.columns:
        return {"programs": [], "r": [], "direction": []}
    s = corr_matrix.loc[target_gene].drop(target_gene, errors="ignore").sort_values(ascending=True)
    top = s[s > 0].head(top_n)
    bottom = s[s < 0].tail(top_n)
    combined = pd.concat([bottom, top])
    return {
        "programs": [str(x) for x in combined.index],
        "r": [float(x) for x in combined.values],
        "direction": ["negative" if x < 0 else "positive" for x in combined.values],
    }


def _build_log2fc_gene(perturb_path, target_gene, target_col, program_col,
                       log2fc_col, num_item, p_value):
    """log2FC bars: programs most affected by perturbation of `target_gene`."""
    df = pd.read_csv(perturb_path, sep="\t")
    df[target_col] = df[target_col].astype(str)
    if target_gene not in df[target_col].values:
        return {"regulators": [], "log2fc": [], "adj_pval": []}
    sub = df.loc[df[target_col] == target_gene]
    sub = sub[sub["adj_pval"] < p_value].sort_values(by=log2fc_col, ascending=False)
    top = sub.head(num_item)
    bottom = sub.tail(num_item)
    plot_data = pd.merge(top, bottom, how="outer").sort_values(by=log2fc_col, ascending=False)
    return {
        "regulators": [str(p) for p in plot_data[program_col].values],
        "log2fc": [float(x) for x in plot_data[log2fc_col].values],
        "adj_pval": [float(x) for x in plot_data["adj_pval"].values],
    }


def _build_volcano_gene(perturb_path, target_gene, target_col, program_col,
                        log2fc_col, down_thred_log, up_thred_log, p_value):
    """Volcano: all programs as effect of perturbing `target_gene`."""
    df = pd.read_csv(perturb_path, sep="\t")
    df[target_col] = df[target_col].astype(str)
    if target_gene not in df[target_col].values:
        return {"regulators": [], "log2fc": [], "neglog10p": [], "category": [],
                "thresholds": {"down": down_thred_log, "up": up_thred_log, "p": p_value}}
    sub = df.loc[df[target_col] == target_gene].copy()
    sub["neglog10p"] = -np.log10(sub["adj_pval"].clip(lower=1e-300))
    cats = []
    for _, r in sub.iterrows():
        sig = r["adj_pval"] <= p_value
        if sig and r[log2fc_col] >= up_thred_log:
            cats.append("up")
        elif sig and r[log2fc_col] <= down_thred_log:
            cats.append("down")
        else:
            cats.append("ns")
    return {
        "regulators": [str(p) for p in sub[program_col].values],
        "log2fc": [float(x) for x in sub[log2fc_col].values],
        "neglog10p": [float(x) for x in sub["neglog10p"].values],
        "adj_pval": [float(x) for x in sub["adj_pval"].values],
        "category": cats,
        "thresholds": {"down": float(down_thred_log), "up": float(up_thred_log), "p": float(p_value)},
    }


def _build_programs_dotplot(mdata, program_list, groupby):
    """Program loading dotplot data: program × condition (mean + frac>0)."""
    if not program_list:
        return {"genes": [], "conditions": [], "mean": [], "frac": []}
    program_names = [str(p) for p in program_list]
    var_names = list(mdata["cNMF"].var_names)
    program_indices, kept = [], []
    for p in program_names:
        if p in var_names:
            program_indices.append(var_names.index(p))
            kept.append(p)
    if not kept:
        return {"genes": [], "conditions": [], "mean": [], "frac": []}

    X = mdata["cNMF"].X
    if hasattr(X, "toarray"):
        X = np.asarray(X.toarray())
    else:
        X = np.asarray(X)

    if groupby in mdata["cNMF"].obs.columns:
        groups = mdata["cNMF"].obs[groupby].values
    else:
        groups = mdata["rna"].obs[groupby].values

    if hasattr(mdata["rna"].obs[groupby], "cat"):
        conds_cat = mdata["rna"].obs[groupby].cat.categories
    else:
        conds_cat = sorted(np.unique(groups))
    conditions = [str(c) for c in conds_cat]

    mean_mat = []
    frac_mat = []
    for c in conditions:
        mask = np.asarray(groups == c).ravel()
        sub = X[mask, :][:, program_indices]
        mean_mat.append([float(v) for v in sub.mean(axis=0)])
        frac_mat.append([float(v) for v in (sub > 0).mean(axis=0)])
    return {
        "genes": kept,  # reuse the same key as the program-side dotplot for symmetry
        "conditions": conditions,
        "mean": mean_mat,
        "frac": frac_mat,
    }


def _build_gene_waterfall(corr_matrix, target_gene, top_num):
    """Waterfall of correlation between target_gene's perturbation effect and every other target."""
    corr_matrix = corr_matrix.copy()
    if target_gene not in corr_matrix.index:
        return {"programs": [], "r": [], "labeled": []}
    s = corr_matrix.loc[target_gene].dropna().sort_values(ascending=False)
    labeled = set(s.head(top_num).index.tolist() + s.tail(top_num).index.tolist())
    return {
        "programs": [str(p) for p in s.index],
        "r": [float(v) for v in s.values],
        "labeled": [str(p) in labeled for p in s.index],
    }


def _build_kd_vs_control(mdata, target_gene, condition_key, control_target_name, gene_name_key):
    """KD vs control bar data per condition. Mirrors plot_perturbation_vs_control_by_condition logic."""
    from scipy import sparse

    if mdata["rna"].n_obs != mdata["cNMF"].n_obs:
        return {"groups": [], "error": "RNA and cNMF have different cell counts"}

    guide_targets = np.array(mdata["cNMF"].uns["guide_targets"])
    ga = mdata["cNMF"].obsm["guide_assignment"]
    X = mdata["rna"].X

    if gene_name_key is not None and gene_name_key in mdata["rna"].var.columns:
        gene_mask = (mdata["rna"].var[gene_name_key].astype(str) == target_gene).values
    else:
        gene_mask = np.array([v == target_gene for v in mdata["rna"].var_names])
    if gene_mask.sum() == 0:
        return {"groups": [], "error": "gene not in expression matrix"}
    gene_idx = int(np.where(gene_mask)[0][0])

    if sparse.issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten()
        gene_expr = np.array(X[:, gene_idx].todense()).flatten()
    else:
        row_sums = np.asarray(X).sum(axis=1)
        gene_expr = np.asarray(X)[:, gene_idx]
    row_sums[row_sums == 0] = 1
    gene_expr_norm = (gene_expr / row_sums) * 1e4

    nt_idx = np.where(guide_targets == control_target_name)[0]
    if len(nt_idx) == 0:
        return {"groups": [], "error": f"control target '{control_target_name}' not in guide_targets"}
    control_mask = np.asarray(ga[:, nt_idx].sum(axis=1)).flatten() > 0

    target_idx = np.where(guide_targets == target_gene)[0]
    if len(target_idx) == 0:
        return {"groups": [], "error": f"gene '{target_gene}' not in guide_targets"}
    perturbed_mask = np.asarray(ga[:, target_idx].sum(axis=1)).flatten() > 0

    conditions_series = mdata["rna"].obs[condition_key]
    if hasattr(conditions_series, "cat"):
        conditions = list(conditions_series.cat.categories)
    else:
        conditions = sorted(conditions_series.unique())
    group_labels = ["All"] + [str(c) for c in conditions]

    pert_means, ctrl_means, pert_sems, ctrl_sems, pert_ns, ctrl_ns = [], [], [], [], [], []
    for label in group_labels:
        if label == "All":
            cond_mask = np.ones(len(gene_expr_norm), dtype=bool)
        else:
            cond_mask = np.array(conditions_series == label)
        ctrl_expr = gene_expr_norm[control_mask & cond_mask]
        pert_expr = gene_expr_norm[perturbed_mask & cond_mask]
        n_c = int(len(ctrl_expr))
        n_p = int(len(pert_expr))
        ctrl_ns.append(n_c)
        pert_ns.append(n_p)
        c_mean = float(np.mean(ctrl_expr)) if n_c > 0 else 0.0
        if n_p == 0 or n_c == 0 or c_mean == 0:
            pert_means.append(None)
            ctrl_means.append(None)
            pert_sems.append(0.0)
            ctrl_sems.append(0.0)
        else:
            pert_means.append(float(np.mean(pert_expr) / c_mean))
            ctrl_means.append(1.0)
            pert_sems.append(float(np.std(pert_expr) / np.sqrt(n_p) / c_mean))
            ctrl_sems.append(float(np.std(ctrl_expr) / np.sqrt(n_c) / c_mean))

    all_effect_pct = None
    if pert_means and pert_means[0] is not None:
        all_effect_pct = float((1 - pert_means[0]) * 100)

    return {
        "groups": [str(x) for x in group_labels],
        "pert_means": pert_means,
        "ctrl_means": ctrl_means,
        "pert_sems": pert_sems,
        "ctrl_sems": ctrl_sems,
        "pert_n": pert_ns,
        "ctrl_n": ctrl_ns,
        "all_effect_pct": all_effect_pct,
        "target_gene": target_gene,
        "control_name": control_target_name,
    }


# ---------------------------------------------------------------------------
# Plotly figure makers for the gene-specific panels
# ---------------------------------------------------------------------------

def _make_top_programs_fig(data):
    if not data["programs"]:
        return _empty_fig("No program loadings")
    fig = go.Figure(go.Bar(
        x=data["loadings"], y=data["programs"], orientation="h",
        marker=dict(color=COLOR_NEUTRAL, line=dict(width=0)),
        hovertemplate="Program %{y}<br>Loading: %{x:.3g}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Gene loading", yaxis_title="Program")
    return _fmt_layout(fig)


def _make_programs_dotplot_fig(data):
    if not data["genes"]:
        return _empty_fig("No programs to plot")
    xs, ys, sizes, colors, hovers = [], [], [], [], []
    max_frac = max((max(row) for row in data["frac"]), default=1.0) or 1.0
    for j, prog in enumerate(data["genes"]):
        for i, cond in enumerate(data["conditions"]):
            xs.append(cond)
            ys.append(f"P{prog}")
            f = data["frac"][i][j]
            m = data["mean"][i][j]
            sizes.append(4 + 20 * (f / max_frac))
            colors.append(m)
            hovers.append(f"Program {prog}<br>{cond}<br>mean={m:.3f}<br>frac={f:.2f}")
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, colorscale="Blues", showscale=True,
                    colorbar=dict(title=dict(text="mean<br>loading", side="right"), thickness=8, len=0.7),
                    line=dict(width=0)),
        text=hovers, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Condition", yaxis_title="Program",
                      xaxis=dict(type="category"), yaxis=dict(type="category"))
    return _fmt_layout(fig)


def _make_kd_vs_control_fig(data):
    if not data.get("groups"):
        msg = data.get("error") or "No data"
        return _empty_fig(msg)
    groups = data["groups"]
    pert = [v if v is not None else 0.0 for v in data["pert_means"]]
    ctrl = [v if v is not None else 0.0 for v in data["ctrl_means"]]
    pert_sem = data["pert_sems"]
    ctrl_sem = data["ctrl_sems"]
    tick_labels = [f"{g}<br>[{p}|{c}]" for g, p, c in zip(groups, data["pert_n"], data["ctrl_n"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=data["target_gene"],
        x=groups, y=pert,
        error_y=dict(type="data", array=pert_sem, color="#222", thickness=1.2, width=4),
        marker=dict(color=COLOR_UP, line=dict(width=0)),
        customdata=list(zip(data["pert_n"], pert_sem)),
        hovertemplate="<b>%{x}</b><br>"
                       f"{data['target_gene']}: " "%{y:.1%} of control<br>"
                       "n=%{customdata[0]}<br>SEM=%{customdata[1]:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"control ({data['control_name']})",
        x=groups, y=ctrl,
        error_y=dict(type="data", array=ctrl_sem, color="#222", thickness=1.2, width=4),
        marker=dict(color=COLOR_NEUTRAL, line=dict(width=0)),
        customdata=list(zip(data["ctrl_n"], ctrl_sem)),
        hovertemplate="<b>%{x}</b><br>control: %{y:.1%}<br>"
                       "n=%{customdata[0]}<br>SEM=%{customdata[1]:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888", line_width=1)
    fig.update_layout(
        barmode="group",
        xaxis=dict(tickmode="array", tickvals=groups, ticktext=tick_labels, type="category"),
        yaxis=dict(tickformat=".0%", range=[0, 1.20]),
        xaxis_title="Condition  [n perturbed | n control]",
        yaxis_title=f"{data['target_gene']} expression (% of control)",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
    )
    return _fmt_layout(fig, height=300)


# ---------------------------------------------------------------------------
# Matplotlib → PNG renderers
# ---------------------------------------------------------------------------

def _render_umap_expression_png(mdata, target_gene, out_path, ensembl_to_symbol_file,
                                 gene_name_key, umap_dot_size, subsample_frac):
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_umap_per_gene(
        mdata=mdata, Target_Gene=target_gene, ax=ax,
        ensembl_to_symbol_file=ensembl_to_symbol_file,
        gene_name_key=gene_name_key, size=umap_dot_size,
        umap_subsample_frac=subsample_frac,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _render_umap_guide_png(mdata, target_gene, out_path, umap_dot_size, subsample_frac):
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_umap_per_gene_guide(
        mdata=mdata, Target_Gene=target_gene, ax=ax,
        size=umap_dot_size, umap_subsample_frac=subsample_frac,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _render_gene_dotplot_png(mdata, target_gene, out_path,
                              ensembl_to_symbol_file, groupby, gene_name_key):
    """Render the scanpy dotplot for the target gene to a PNG (standalone mode).

    Note: perturbed_gene_dotplot calls plt.close() internally when show=False,
    which destroys the figure before we can save it. Pass show=True (a no-op
    in the Agg backend) so the figure stays open, then save via ax.get_figure().
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ax = perturbed_gene_dotplot(
            mdata=mdata, Target_Gene=target_gene,
            ensembl_to_symbol_file=ensembl_to_symbol_file,
            dotplot_groupby=groupby, gene_name_key=gene_name_key,
            figsize=(4, 3),
            save_path=None, save_name=None, show=True, ax=None,
        )
        if ax is None:
            raise RuntimeError(f"gene '{target_gene}' not found by dotplot")
        fig = ax.get_figure()
        fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"dotplot unavailable\n({e})",
                ha="center", va="center", fontsize=10, color="#888")
        ax.set_axis_off()
        fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gene {gene} — cNMF perturbation analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{fonts}" rel="stylesheet">
<link rel="stylesheet" href="../shared/style.css">
<script src="{cdn}"></script>
</head>
<body>
<header class="sticky">
  <div class="header-left">
    <a href="../index.html" class="back">&larr; All genes</a>
    <h1>Gene <span class="pid">{gene}</span> <span class="position">{position}</span></h1>
  </div>
  <div class="header-right">
    <a href="{prev_href}" class="nav-link{prev_disabled}">&larr; Prev</a>
    <a href="{next_href}" class="nav-link{next_disabled}">Next &rarr;</a>
  </div>
</header>

<section class="header-row">
  <div class="panel umap">
    <h3 class="panel-title">UMAP <span class="muted">— expression</span></h3>
    <div class="panel-body"><img src="images/umap_expression.png" alt="UMAP expression of {gene}"></div>
  </div>
  <div class="panel umap">
    <h3 class="panel-title">UMAP <span class="muted">— guide cells</span></h3>
    <div class="panel-body"><img src="images/umap_guide.png" alt="UMAP guide assignment of {gene}"></div>
  </div>
  <div class="panel umap">
    <h3 class="panel-title">Gene expression <span class="muted">by condition</span></h3>
    <div class="panel-body"><img src="images/gene_dotplot.png" alt="Gene dotplot of {gene}"></div>
  </div>
  <div class="panel">
    <h3 class="panel-title">Top loaded programs</h3>
    <div class="panel-body">{top_programs_div}</div>
  </div>
  <div class="panel">
    <h3 class="panel-title">Correlated genes</h3>
    <div class="panel-body">{corr_div}</div>
  </div>
</section>

<section class="heatmap-row">
  <div class="panel wide">
    <h3 class="panel-title">CRISPRi knockdown <span class="muted">vs control, by condition</span></h3>
    <div class="panel-body">{kd_div}</div>
  </div>
</section>

<section class="sample-rows">
{sample_blocks}
</section>

<footer>Generated {timestamp}</footer>
</body>
</html>
"""

_SAMPLE_BLOCK = """<div class="sample-row">
  <h2 class="sample-heading"><span class="sample-badge">{sample}</span></h2>
  <div class="panels-4">
    <div class="panel">
      <h3 class="panel-title">Effect on programs</h3>
      <div class="panel-body">{log2fc_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Volcano</h3>
      <div class="panel-body">{volcano_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Program loadings <span class="muted">(dot plot)</span></h3>
      <div class="panel-body">{dotplot_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Similar genes <span class="muted">(by effect)</span></h3>
      <div class="panel-body">{waterfall_div}</div>
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_gene_html(
    mdata,
    perturb_path_base,
    ensembl_to_symbol_file,
    Target_Gene,
    gene_loading_corr_matrix,
    perturb_corr_by_sample,
    sample,
    html_share_path,
    *,
    top_n_programs=10,
    top_corr_genes=5,
    groupby="sample",
    perturb_target_col="target_name",
    perturb_program_col="program_name",
    perturb_log2fc_col="log2FC",
    volcano_log2fc_min=-0.0,
    volcano_log2fc_max=0.0,
    significance_threshold=0.05,
    gene_name_key="symbol",
    control_target_name="non-targeting",
    umap_dot_size=10,
    subsample_frac=None,
    prev_gene=None,
    next_gene=None,
    position_index=None,
    position_total=None,
):
    """Write gene_{SYMBOL}/ subtree under html_share_path."""
    share_root = Path(html_share_path)
    gene_dir = share_root / f"gene_{Target_Gene}"
    (gene_dir / "data").mkdir(parents=True, exist_ok=True)
    (gene_dir / "images").mkdir(parents=True, exist_ok=True)

    # ---- images (matplotlib → PNG) ----
    _render_umap_expression_png(mdata, Target_Gene, gene_dir / "images" / "umap_expression.png",
                                 ensembl_to_symbol_file, gene_name_key, umap_dot_size, subsample_frac)
    _render_umap_guide_png(mdata, Target_Gene, gene_dir / "images" / "umap_guide.png",
                            umap_dot_size, subsample_frac)
    _render_gene_dotplot_png(mdata, Target_Gene, gene_dir / "images" / "gene_dotplot.png",
                              ensembl_to_symbol_file, groupby, gene_name_key)

    # ---- header-row interactive panels ----
    top_programs_d = _build_top_programs(mdata, Target_Gene, top_n_programs,
                                          ensembl_to_symbol_file, gene_name_key)
    corr_d = _build_gene_correlations(gene_loading_corr_matrix, Target_Gene, top_corr_genes)
    _write_json(gene_dir / "data" / "top_programs.json", top_programs_d)
    _write_json(gene_dir / "data" / "correlations.json", corr_d)

    top_programs_fig = _make_top_programs_fig(top_programs_d)
    corr_fig = _make_correlations_fig(corr_d)

    # ---- wide KD vs control bar ----
    kd_d = _build_kd_vs_control(mdata, Target_Gene, groupby, control_target_name, gene_name_key)
    _write_json(gene_dir / "data" / "kd_vs_control.json", kd_d)
    kd_fig = _make_kd_vs_control_fig(kd_d)

    # ---- per-sample panels ----
    sample_blocks = []
    sig_per_sample = {}
    for samp in sample:
        skey = _safe_sample_key(samp)
        perturb_path = f"{perturb_path_base}_{samp}.txt"

        log2fc_d = _build_log2fc_gene(perturb_path, Target_Gene, perturb_target_col,
                                       perturb_program_col, perturb_log2fc_col,
                                       top_n_programs, significance_threshold)
        volcano_d = _build_volcano_gene(perturb_path, Target_Gene, perturb_target_col,
                                         perturb_program_col, perturb_log2fc_col,
                                         volcano_log2fc_min, volcano_log2fc_max,
                                         significance_threshold)
        programs_dot_d = _build_programs_dotplot(mdata, log2fc_d["regulators"], groupby)
        waterfall_d = _build_gene_waterfall(perturb_corr_by_sample[samp], Target_Gene, top_corr_genes)

        _write_json(gene_dir / "data" / f"log2fc_{skey}.json", log2fc_d)
        _write_json(gene_dir / "data" / f"volcano_{skey}.json", volcano_d)
        _write_json(gene_dir / "data" / f"programs_dotplot_{skey}.json", programs_dot_d)
        _write_json(gene_dir / "data" / f"waterfall_{skey}.json", waterfall_d)

        sig_per_sample[samp] = sum(1 for c in volcano_d["category"] if c != "ns")

        sample_blocks.append(_SAMPLE_BLOCK.format(
            sample=samp,
            log2fc_div=_fig_to_div(_make_log2fc_fig(log2fc_d), f"div-log2fc-{skey}"),
            volcano_div=_fig_to_div(_make_volcano_fig(volcano_d), f"div-volcano-{skey}"),
            dotplot_div=_fig_to_div(_make_programs_dotplot_fig(programs_dot_d), f"div-dotplot-{skey}"),
            waterfall_div=_fig_to_div(_make_waterfall_fig(waterfall_d, Target_Gene), f"div-waterfall-{skey}"),
        ))

    # ---- nav strings ----
    position = ""
    if position_index is not None and position_total is not None:
        position = f"{position_index} / {position_total}"
    prev_href = f"../gene_{prev_gene}/gene_{prev_gene}.html" if prev_gene else "#"
    prev_disabled = "" if prev_gene else " disabled"
    next_href = f"../gene_{next_gene}/gene_{next_gene}.html" if next_gene else "#"
    next_disabled = "" if next_gene else " disabled"

    # ---- assemble page ----
    page = _PAGE_TEMPLATE.format(
        gene=Target_Gene, cdn=PLOTLY_CDN, fonts=GOOGLE_FONTS,
        position=position,
        prev_href=prev_href, prev_disabled=prev_disabled,
        next_href=next_href, next_disabled=next_disabled,
        top_programs_div=_fig_to_div(top_programs_fig, "div-top-programs"),
        corr_div=_fig_to_div(corr_fig, "div-gene-corr"),
        kd_div=_fig_to_div(kd_fig, "div-kd"),
        sample_blocks="".join(sample_blocks),
        timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    with open(gene_dir / f"gene_{Target_Gene}.html", "w") as f:
        f.write(page)

    # ---- metadata.json ----
    metadata = {
        "gene_symbol": Target_Gene,
        "samples": list(map(str, sample)),
        "thresholds": {
            "significance_threshold": float(significance_threshold),
            "volcano_log2fc_min": float(volcano_log2fc_min),
            "volcano_log2fc_max": float(volcano_log2fc_max),
        },
        "top_loaded_programs": top_programs_d["programs"][::-1],
        "top_correlated_genes": {
            "positive": [p for p, d in zip(corr_d["programs"], corr_d["direction"]) if d == "positive"],
            "negative": [p for p, d in zip(corr_d["programs"], corr_d["direction"]) if d == "negative"],
        },
        "n_significant_program_perturbations_total": int(sum(sig_per_sample.values())),
        "n_significant_program_perturbations_per_sample": {str(k): int(v) for k, v in sig_per_sample.items()},
        "kd_effect_pct_of_control": kd_d.get("all_effect_pct"),
        "control_target_name": control_target_name,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    _write_json(gene_dir / "metadata.json", metadata)


def write_gene_share_index(html_share_path, gene_list, config_snapshot):
    """Write index.html + shared/style.css + shared/manifest.json for the gene share folder."""
    share_root = Path(html_share_path)
    (share_root / "shared").mkdir(parents=True, exist_ok=True)
    with open(share_root / "shared" / "style.css", "w") as f:
        f.write(_STYLE_CSS)

    manifest = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "kind": "perturbed_gene",
        "gene_list": list(map(str, gene_list)),
        "config": {k: v for k, v in config_snapshot.items()
                   if not isinstance(v, (np.ndarray,))},
    }
    _write_json(share_root / "shared" / "manifest.json", manifest)

    cards = []
    for gene in gene_list:
        gene = str(gene)
        meta_path = share_root / f"gene_{gene}" / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            m = json.load(f)
        top_progs = m.get("top_loaded_programs", [])[:5]
        n_sig = int(m.get("n_significant_program_perturbations_total", 0))
        kd_pct = m.get("kd_effect_pct_of_control")

        chips = "".join(f'<span class="chip">P{p}</span>' for p in top_progs) \
                or '<span class="chip" style="opacity:0.5">—</span>'
        if kd_pct is None:
            kd_label = "n/a"
            kd_cls = "zero"
        else:
            kd_label = f"{kd_pct:+.0f}% KD" if kd_pct > 0 else f"{kd_pct:+.0f}%"
            kd_cls = "sig" if abs(kd_pct) >= 20 else "zero"

        sig_cls = "sig" if n_sig > 0 else "zero"
        cards.append(
            f'<a class="program-card" href="gene_{gene}/gene_{gene}.html">'
            f'  <div class="pid-block">'
            f'    <span class="pid-label">Gene</span>'
            f'    <span class="pid-value">{gene}</span>'
            f'  </div>'
            f'  <div class="details">'
            f'    <div class="go-term">Top loaded programs</div>'
            f'    <div class="chips">{chips}</div>'
            f'  </div>'
            f'  <div class="badge-block">'
            f'    <span class="badge {kd_cls}">{kd_label}</span>'
            f'    <span class="badge-label">vs control</span>'
            f'  </div>'
            f'  <div class="badge-block">'
            f'    <span class="badge {sig_cls}">{n_sig}</span>'
            f'    <span class="badge-label">sig programs</span>'
            f'  </div>'
            f'</a>'
        )

    n_genes = len(cards)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>cNMF perturbed genes — share index</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{GOOGLE_FONTS}" rel="stylesheet">'
        "<link rel='stylesheet' href='shared/style.css'>"
        "</head><body>"
        f"<header class='index-header'><h1>Perturbed gene analysis</h1>"
        f"<div class='subtitle'>{n_genes} genes · top loaded programs, knockdown effect, and # significant program perturbations</div>"
        "</header>"
        f"<section class='program-list'>{''.join(cards)}</section>"
        f"<footer>Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</footer>"
        "</body></html>"
    )
    with open(share_root / "index.html", "w") as f:
        f.write(html)
