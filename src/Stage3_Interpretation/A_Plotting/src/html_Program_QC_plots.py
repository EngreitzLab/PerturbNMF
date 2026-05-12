"""
Interactive HTML export for the per-program comprehensive plot.

When cNMF_program_analysis.py is invoked with --output_format HTML, this
module renders one self-contained share folder containing:

  * program_{N}/program_{N}.html  : interactive Plotly page
  * program_{N}/metadata.json     : program-level summary stats
  * program_{N}/data/*.json       : per-panel raw arrays
  * program_{N}/images/umap.png   : matplotlib UMAP, embedded as <img>
  * index.html                    : table linking all programs
  * shared/style.css, manifest.json

Plotly is loaded from CDN; the folder is movable.
"""
from __future__ import annotations

import os
import json
import math
import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

from .Program_QC_plots import plot_umap_per_program
from .Program_expression_weighted_plots import compute_program_expression_by_condition
from .utilities import rename_list_gene_dictionary


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap"

# Unified palette — used in every Plotly figure
COLOR_UP = "#c0392b"       # positive effect / upregulated
COLOR_DOWN = "#1c5f8f"     # negative effect / downregulated
COLOR_NS = "#cbd5db"       # not significant
COLOR_NEUTRAL = "#7f8c8d"  # neutral grey bar
COLOR_ACCENT = "#1769aa"   # link / accent

# Plotly template applied to every figure for consistency
_PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=11, color="#222"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#eef0f2", linecolor="#cfd4d8", zerolinecolor="#dfe3e6",
                   ticks="outside", tickcolor="#cfd4d8", tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#eef0f2", linecolor="#cfd4d8", zerolinecolor="#dfe3e6",
                   ticks="outside", tickcolor="#cfd4d8", tickfont=dict(size=10)),
        colorway=[COLOR_ACCENT, COLOR_UP, COLOR_DOWN, COLOR_NEUTRAL],
        margin=dict(l=50, r=20, t=20, b=50),
        hoverlabel=dict(bgcolor="#222", font=dict(color="#fff", size=11)),
    )
)


# ---------------------------------------------------------------------------
# JSON / file helpers
# ---------------------------------------------------------------------------

def _sanitize(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, np.floating):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (pd.Series, pd.Index)):
        return _sanitize(list(o))
    if isinstance(o, pd.DataFrame):
        return _sanitize(o.to_dict(orient="list"))
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize(payload), f, indent=None)


def _safe_sample_key(samp: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(samp))


# ---------------------------------------------------------------------------
# Per-panel data builders. Each returns a JSON-serializable dict.
# Data prep mirrors the matplotlib panel functions; rendering happens in
# _make_*_fig functions below.
# ---------------------------------------------------------------------------

def _build_top_genes(mdata, target_program, num_gene, file_to_dictionary, gene_name_key):
    # Mirror of plot_top_gene_per_program data prep.
    X = mdata["cNMF"].varm["loadings"]
    gene_names = mdata["cNMF"].uns["var_names"]
    if gene_name_key is not None and gene_name_key in mdata["rna"].var.columns:
        ens_to_sym = dict(zip(mdata["rna"].var_names, mdata["rna"].var[gene_name_key].astype(str)))
        gene_names = np.array([ens_to_sym.get(g, g) for g in gene_names])
    if file_to_dictionary is not None:
        gene_names = rename_list_gene_dictionary(list(gene_names), file_to_dictionary)
    df = pd.DataFrame(data=X, columns=gene_names, index=mdata["cNMF"].var_names)
    matching_rows = (df.index == str(target_program)).sum()
    if matching_rows == 0:
        raise ValueError(
            f"Program {str(target_program)!r} not found in the loading matrix. "
            f"Available programs: {list(df.index)}."
        )
    if matching_rows > 1:
        raise ValueError(
            f"Program {str(target_program)!r} matches {matching_rows} rows in the "
            f"loading matrix (duplicate program ids)."
        )
    n_genes = df.shape[1]
    if num_gene > n_genes:
        raise ValueError(
            f"num_gene={num_gene} exceeds the number of genes available "
            f"in the loading matrix ({n_genes})."
        )
    series = df.loc[str(target_program)].nlargest(num_gene).sort_values(ascending=True)
    return {"genes": list(series.index), "loadings": list(series.values)}


def _build_go_terms(GO_path, target_program, num_term, p_value_name, term_col):
    # Mirror of top_GO_per_program data prep.
    df = pd.read_csv(GO_path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    if str(target_program) not in df.index:
        return {"terms": [], "neglog10p": [], "adj_pval": []}
    sub = df.loc[str(target_program)]
    sub = sub.set_index(term_col) if term_col in sub.columns else sub
    ps = sub[p_value_name].nsmallest(num_term)
    return {
        "terms": list(ps.index),
        "neglog10p": [-math.log10(p) if p > 0 else 0.0 for p in ps.values],
        "adj_pval": list(ps.values),
    }


def _build_correlations(program_correlation, target_program, num_program):
    if str(target_program) not in program_correlation.columns:
        return {"programs": [], "r": [], "direction": []}
    s = program_correlation.loc[str(target_program)].drop(str(target_program)).sort_values(ascending=True)
    top = s[s > 0].head(num_program)
    bottom = s[s < 0].tail(num_program)
    combined = pd.concat([bottom, top])
    return {
        "programs": [str(x) for x in combined.index],
        "r": [float(x) for x in combined.values],
        "direction": ["negative" if x < 0 else "positive" for x in combined.values],
    }


def _build_violin(mdata, target_program, groupby):
    X = mdata["cNMF"].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    prog_idx = list(mdata["cNMF"].var_names).index(str(target_program))
    scores = np.asarray(X[:, prog_idx]).ravel()
    if groupby in mdata["cNMF"].obs.columns:
        groups = mdata["cNMF"].obs[groupby].values
    else:
        groups = mdata["rna"].obs[groupby].values
    df = pd.DataFrame({"expression": scores, "group": list(groups)})
    order = list(df["group"].astype("category").cat.categories) if hasattr(df["group"], "cat") else sorted(df["group"].unique())
    summary = df.groupby("group")["expression"].agg(["mean", lambda x: float((x > x.mean()).mean())])
    summary.columns = ["mean", "frac_above"]
    return {
        "groups": list(map(str, order)),
        "per_group_expression": {str(g): list(df.loc[df["group"] == g, "expression"].values) for g in order},
        "summary": {str(g): {"mean": float(summary.loc[g, "mean"]),
                              "frac_above": float(summary.loc[g, "frac_above"])} for g in order},
    }


def _build_log2fc(perturb_path, target_program, tagert_col_name, plot_col_name,
                  log2fc_col, num_item, p_value, gene_list):
    df = pd.read_csv(perturb_path, sep="\t")
    df["program_name"] = df["program_name"].astype(str)
    if str(target_program) not in df[tagert_col_name].astype(str).values:
        return {"regulators": [], "log2fc": [], "adj_pval": []}
    sub = df.loc[df[tagert_col_name].astype(str) == str(target_program)]
    if gene_list:
        sub = sub[sub[plot_col_name].isin(set(gene_list))]
    sub = sub[sub["adj_pval"] < p_value].sort_values(by=log2fc_col, ascending=False)
    top = sub.head(num_item)
    bottom = sub.tail(num_item)
    plot_data = pd.merge(top, bottom, how="outer").sort_values(by=log2fc_col, ascending=False)
    return {
        "regulators": list(plot_data[plot_col_name].astype(str).values),
        "log2fc": [float(x) for x in plot_data[log2fc_col].values],
        "adj_pval": [float(x) for x in plot_data["adj_pval"].values],
    }


def _build_volcano(perturb_path, target_program, tagert_col_name, plot_col_name,
                   log2fc_col, down_thred_log, up_thred_log, p_value, gene_list):
    df = pd.read_csv(perturb_path, sep="\t")
    df["program_name"] = df["program_name"].astype(str)
    if str(target_program) not in df[tagert_col_name].astype(str).values:
        return {"regulators": [], "log2fc": [], "neglog10p": [], "category": [],
                "thresholds": {"down": down_thred_log, "up": up_thred_log, "p": p_value}}
    sub = df.loc[df[tagert_col_name].astype(str) == str(target_program)].copy()
    if gene_list:
        sub = sub[sub[plot_col_name].isin(set(gene_list))]
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
        "regulators": list(sub[plot_col_name].astype(str).values),
        "log2fc": [float(x) for x in sub[log2fc_col].values],
        "neglog10p": [float(x) for x in sub["neglog10p"].values],
        "adj_pval": [float(x) for x in sub["adj_pval"].values],
        "category": cats,
        "thresholds": {"down": float(down_thred_log), "up": float(up_thred_log), "p": float(p_value)},
    }


def _build_dotplot(mdata, gene_list, groupby, gene_name_key):
    if not gene_list:
        return {"genes": [], "conditions": [], "mean": [], "frac": []}
    if gene_name_key is not None and gene_name_key in mdata["rna"].var.columns:
        sym_to_ens = {}
        for ens, sym in zip(mdata["rna"].var_names, mdata["rna"].var[gene_name_key].astype(str)):
            sym_to_ens.setdefault(sym, ens)
        gene_indices = []
        kept_genes = []
        for g in gene_list:
            if g in sym_to_ens:
                gene_indices.append(list(mdata["rna"].var_names).index(sym_to_ens[g]))
                kept_genes.append(g)
        gene_list = kept_genes
    else:
        gene_indices = []
        kept_genes = []
        for g in gene_list:
            if g in mdata["rna"].var_names:
                gene_indices.append(list(mdata["rna"].var_names).index(g))
                kept_genes.append(g)
        gene_list = kept_genes
    if not gene_list:
        return {"genes": [], "conditions": [], "mean": [], "frac": []}

    X = mdata["rna"].X
    groups = mdata["rna"].obs[groupby].values
    conds_cat = mdata["rna"].obs[groupby].astype("category").cat.categories if hasattr(mdata["rna"].obs[groupby], "cat") else sorted(np.unique(groups))
    conditions = [str(c) for c in conds_cat]
    mean_mat = []
    frac_mat = []
    for c in conditions:
        mask = np.asarray(groups == c).ravel()
        sub = X[mask, :][:, gene_indices]
        mean_row = np.asarray(sub.mean(axis=0)).ravel()
        frac_row = np.asarray((sub > 0).mean(axis=0)).ravel()
        mean_mat.append([float(v) for v in mean_row])
        frac_mat.append([float(v) for v in frac_row])
    return {
        "genes": list(gene_list),
        "conditions": conditions,
        "mean": mean_mat,
        "frac": frac_mat,
    }


def _build_waterfall(corr_matrix, target_program, top_num):
    corr_matrix = corr_matrix.copy()
    corr_matrix.index = corr_matrix.index.astype(str)
    if str(target_program) not in corr_matrix.index:
        return {"programs": [], "r": [], "labeled": []}
    s = corr_matrix.loc[str(target_program)].dropna().sort_values(ascending=False)
    labeled = set(s.head(top_num).index.tolist() + s.tail(top_num).index.tolist())
    return {
        "programs": [str(p) for p in s.index],
        "r": [float(v) for v in s.values],
        "labeled": [str(p) in labeled for p in s.index],
    }


def _build_heatmap(perturb_path_base, mdata, target_program, sample,
                   tagert_col_name, plot_col_name, log2fc_col, p_value, groupby):
    dfs = [pd.read_csv(f"{perturb_path_base}_{samp}.txt", sep="\t").assign(sample=samp) for samp in sample]
    combined = pd.concat(dfs, ignore_index=True)
    combined["program_name"] = combined["program_name"].astype(str)
    sub = combined.loc[combined[tagert_col_name].astype(str) == str(target_program)]
    keep = sub[sub["adj_pval"] < p_value][plot_col_name].unique()
    sub = sub[sub[plot_col_name].isin(keep)]
    if sub.empty:
        return {"conditions": [], "regulators": [], "log2fc": [], "sig": [], "expression": []}
    pv = sub.pivot(columns=plot_col_name, index="sample", values=log2fc_col)
    pp = sub.pivot(columns=plot_col_name, index="sample", values="adj_pval")
    pv = pv.reindex(sample)
    pp = pp.reindex(sample)
    expr = compute_program_expression_by_condition(mdata, target_program, groupby=groupby)
    expr_vals = [float(expr.get(c, 0.0)) for c in pv.index]
    return {
        "conditions": list(map(str, pv.index)),
        "regulators": list(map(str, pv.columns)),
        "log2fc": [[None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in row]
                   for row in pv.values],
        "sig": [[(False if (isinstance(v, float) and math.isnan(v)) else bool(v < p_value)) for v in row]
                for row in pp.values],
        "expression": expr_vals,
    }


# ---------------------------------------------------------------------------
# Plotly figure builders. No internal titles — titles are emitted in HTML.
# ---------------------------------------------------------------------------

def _fmt_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(template=_PLOTLY_TEMPLATE, height=height)
    return fig


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=f"<i>{message}</i>", showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(color="#999", size=12),
    )
    fig.update_layout(
        template=_PLOTLY_TEMPLATE, height=160,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def _make_top_genes_fig(data):
    if not data["genes"]:
        return _empty_fig("No gene loadings")
    fig = go.Figure(go.Bar(
        x=data["loadings"], y=data["genes"], orientation="h",
        marker=dict(color=COLOR_NEUTRAL, line=dict(width=0)),
        hovertemplate="%{y}<br>Loading: %{x:.3g}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Gene loading (z-score)", yaxis_title="")
    return _fmt_layout(fig)


def _make_go_fig(data):
    if not data["terms"]:
        return _empty_fig("No GO enrichment")
    fig = go.Figure(go.Bar(
        x=data["neglog10p"], y=data["terms"], orientation="h",
        marker=dict(color=COLOR_NEUTRAL, line=dict(width=0)),
        customdata=data["adj_pval"],
        hovertemplate="%{y}<br>-log10(adj_p): %{x:.2f}<br>adj_p: %{customdata:.2e}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="-log10 adj. p-value", yaxis_title="",
                      yaxis=dict(automargin=True))
    return _fmt_layout(fig)


def _make_correlations_fig(data):
    if not data["programs"]:
        return _empty_fig("No correlations")
    colors = [COLOR_DOWN if d == "negative" else COLOR_UP for d in data["direction"]]
    fig = go.Figure(go.Bar(
        x=data["r"], y=data["programs"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="Program %{y}<br>r = %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=0.5, line_color="#555")
    fig.update_layout(xaxis_title="Pearson r", yaxis_title="Program")
    return _fmt_layout(fig)


def _make_violin_fig(data):
    if not data["groups"]:
        return _empty_fig("No expression data")
    fig = go.Figure()
    for g in data["groups"]:
        s = data["summary"][g]
        fig.add_trace(go.Violin(
            y=data["per_group_expression"][g], name=g,
            box_visible=True, meanline_visible=True, points=False,
            line=dict(color=COLOR_ACCENT, width=1),
            fillcolor="rgba(23,105,170,0.20)",
            hovertemplate=f"{g}<br>mean={s['mean']:.3f}<br>frac>mean={s['frac_above']:.2f}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Condition", yaxis_title="Program expression",
                      showlegend=False)
    return _fmt_layout(fig)


def _make_log2fc_fig(data):
    if not data["regulators"]:
        return _empty_fig("No significant regulators")
    colors = [COLOR_UP if v > 0 else COLOR_DOWN for v in data["log2fc"]]
    fig = go.Figure(go.Bar(
        x=data["log2fc"], y=data["regulators"], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=data["adj_pval"],
        hovertemplate="%{y}<br>log2FC: %{x:.3f}<br>adj_p: %{customdata:.2e}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=0.5, line_color="#555")
    fig.update_layout(xaxis_title="log2FC", yaxis_title="Regulator")
    return _fmt_layout(fig)


def _make_volcano_fig(data):
    if not data["regulators"]:
        return _empty_fig("No data")
    cat_colors = {"up": COLOR_UP, "down": COLOR_DOWN, "ns": COLOR_NS}
    cat_names = {"up": "Up", "down": "Down", "ns": "n.s."}
    fig = go.Figure()
    for cat in ("ns", "down", "up"):
        idx = [i for i, c in enumerate(data["category"]) if c == cat]
        if not idx:
            continue
        fig.add_trace(go.Scattergl(
            x=[data["log2fc"][i] for i in idx],
            y=[data["neglog10p"][i] for i in idx],
            mode="markers", name=cat_names[cat],
            text=[data["regulators"][i] for i in idx],
            customdata=[data["adj_pval"][i] for i in idx],
            marker=dict(color=cat_colors[cat], size=6,
                        opacity=0.85 if cat != "ns" else 0.45,
                        line=dict(width=0)),
            hovertemplate="%{text}<br>log2FC: %{x:.3f}<br>-log10 adj_p: %{y:.2f}<br>adj_p: %{customdata:.2e}<extra></extra>",
        ))
    th = data["thresholds"]
    fig.add_vline(x=th["down"], line_dash="dash", line_color="#999", line_width=1)
    fig.add_vline(x=th["up"], line_dash="dash", line_color="#999", line_width=1)
    if th["p"] > 0:
        fig.add_hline(y=-math.log10(th["p"]), line_dash="dash", line_color="#999", line_width=1)
    fig.update_layout(
        xaxis_title="log2FC", yaxis_title="-log10 adj. p-value",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.08, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)"),
    )
    return _fmt_layout(fig)


def _make_dotplot_fig(data):
    if not data["genes"]:
        return _empty_fig("No regulators expressed")
    xs, ys, sizes, colors, hovers = [], [], [], [], []
    max_frac = max((max(row) for row in data["frac"]), default=1.0) or 1.0
    for j, gene in enumerate(data["genes"]):
        for i, cond in enumerate(data["conditions"]):
            xs.append(cond)
            ys.append(gene)
            f = data["frac"][i][j]
            m = data["mean"][i][j]
            sizes.append(4 + 20 * (f / max_frac))
            colors.append(m)
            hovers.append(f"{gene}<br>{cond}<br>mean={m:.3f}<br>frac={f:.2f}")
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, colorscale="Reds", showscale=True,
                    colorbar=dict(title=dict(text="mean", side="right"), thickness=8, len=0.7),
                    line=dict(width=0)),
        text=hovers, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Condition", yaxis_title="Regulator",
                      xaxis=dict(type="category"), yaxis=dict(type="category"))
    return _fmt_layout(fig)


def _make_waterfall_fig(data, target_program):
    if not data["programs"]:
        return _empty_fig("No correlation data")
    xs = list(range(len(data["programs"])))
    fig = go.Figure(go.Scattergl(
        x=xs, y=data["r"], mode="markers",
        text=data["programs"],
        marker=dict(size=[8 if lab else 5 for lab in data["labeled"]],
                    color=[COLOR_ACCENT if lab else COLOR_NS for lab in data["labeled"]],
                    line=dict(width=0)),
        hovertemplate="Program %{text}<br>r = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)
    fig.update_layout(xaxis_title="Rank", yaxis_title=f"r vs Program {target_program}")
    return _fmt_layout(fig)


def _make_heatmap_fig(data):
    if not data["regulators"]:
        return _empty_fig("No significant regulators across conditions")
    htext = []
    for i, _row in enumerate(data["log2fc"]):
        line = []
        for j, v in enumerate(_row):
            s = " *" if data["sig"][i][j] else ""
            line.append(f"{data['regulators'][j]}<br>{data['conditions'][i]}<br>log2FC={v}{s}")
        htext.append(line)
    fig = go.Figure(go.Heatmap(
        z=data["log2fc"], x=data["regulators"], y=data["conditions"],
        zmin=-1, zmax=1, zmid=0, colorscale="RdBu_r", reversescale=False,
        text=htext, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text="log2FC", side="right"), thickness=8, len=0.7),
    ))
    annotations = []
    for i, row in enumerate(data["sig"]):
        for j, sig in enumerate(row):
            if sig:
                annotations.append(dict(
                    x=data["regulators"][j], y=data["conditions"][i],
                    text="*", showarrow=False,
                    font=dict(color="#111", size=14, family="JetBrains Mono, monospace")))
    height = max(280, 80 + 36 * len(data["conditions"]))
    fig.update_layout(annotations=annotations, xaxis=dict(tickangle=-45), height=height)
    return _fmt_layout(fig, height=height)


# ---------------------------------------------------------------------------
# UMAP PNG rendering — reuses the existing matplotlib function
# ---------------------------------------------------------------------------

def _render_umap_png(mdata, target_program, out_path, subsample_frac):
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_umap_per_program(
        mdata=mdata, Target_Program=target_program, ax=ax,
        subsample_frac=subsample_frac,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Program {pid} — cNMF analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{fonts}" rel="stylesheet">
<link rel="stylesheet" href="../shared/style.css">
<script src="{cdn}"></script>
</head>
<body>
<header class="sticky">
  <div class="header-left">
    <a href="../index.html" class="back">&larr; All programs</a>
    <h1>Program <span class="pid">{pid}</span> <span class="position">{position}</span></h1>
  </div>
  <div class="header-right">
    <a href="{prev_href}" class="nav-link{prev_disabled}">&larr; Prev</a>
    <a href="{next_href}" class="nav-link{next_disabled}">Next &rarr;</a>
  </div>
</header>

<section class="header-row">
  <div class="panel umap">
    <h3 class="panel-title">UMAP <span class="muted">— program score</span></h3>
    <div class="panel-body"><img src="images/umap.png" alt="UMAP of program {pid}"></div>
  </div>
  <div class="panel">
    <h3 class="panel-title">Expression <span class="muted">by condition</span></h3>
    <div class="panel-body">{violin_div}</div>
  </div>
  <div class="panel">
    <h3 class="panel-title">GO enrichment</h3>
    <div class="panel-body">{go_div}</div>
  </div>
  <div class="panel">
    <h3 class="panel-title">Correlated programs</h3>
    <div class="panel-body">{corr_div}</div>
  </div>
  <div class="panel">
    <h3 class="panel-title">Top loaded genes</h3>
    <div class="panel-body">{top_genes_div}</div>
  </div>
</section>

<section class="sample-rows">
{sample_blocks}
</section>

<section class="heatmap-row">
  <div class="panel wide">
    <h3 class="panel-title">Regulator effects <span class="muted">across conditions</span></h3>
    <div class="panel-body">{heatmap_div}</div>
  </div>
</section>

<footer>Generated {timestamp}</footer>
</body>
</html>
"""

_SAMPLE_BLOCK = """<div class="sample-row">
  <h2 class="sample-heading"><span class="sample-badge">{sample}</span></h2>
  <div class="panels-4">
    <div class="panel">
      <h3 class="panel-title">Regulator effects</h3>
      <div class="panel-body">{log2fc_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Volcano</h3>
      <div class="panel-body">{volcano_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Regulator expression</h3>
      <div class="panel-body">{dotplot_div}</div>
    </div>
    <div class="panel">
      <h3 class="panel-title">Similar regulators <span class="muted">(by program)</span></h3>
      <div class="panel-body">{waterfall_div}</div>
    </div>
  </div>
</div>
"""

_STYLE_CSS = """:root {
  --bg: #f5f7fa;
  --panel-bg: #ffffff;
  --border: #eaecef;
  --border-strong: #d8dde2;
  --text: #1f2328;
  --text-muted: #6b7176;
  --accent: #1769aa;
  --shadow-sm: 0 1px 2px rgba(15,23,42,0.04);
  --shadow-md: 0 4px 14px rgba(15,23,42,0.08);
  --radius: 10px;
}

* { box-sizing: border-box; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0;
  padding: 0 28px 56px 28px;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}

/* Sticky header */
header.sticky {
  position: sticky; top: 0; z-index: 50;
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 0; margin-bottom: 20px;
  background: rgba(245,247,250,0.85);
  backdrop-filter: saturate(180%) blur(10px);
  -webkit-backdrop-filter: saturate(180%) blur(10px);
  border-bottom: 1px solid var(--border);
}
.header-left { display: flex; align-items: baseline; gap: 16px; }
.header-right { display: flex; gap: 8px; }
.back { color: var(--text-muted); text-decoration: none; font-size: 13px; font-weight: 500; }
.back:hover { color: var(--accent); }
h1 {
  margin: 0; font-size: 20px; font-weight: 600; color: #0f1419;
  display: inline-flex; align-items: baseline; gap: 10px;
}
h1 .pid { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: var(--accent); }
h1 .position {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
  color: var(--text-muted);
  background: rgba(23,105,170,0.08);
  padding: 2px 8px; border-radius: 999px;
}
.nav-link {
  display: inline-block; padding: 6px 12px;
  font-size: 13px; font-weight: 500; color: var(--text);
  text-decoration: none;
  background: var(--panel-bg);
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  transition: all 0.15s ease;
}
.nav-link:hover { border-color: var(--accent); color: var(--accent); }
.nav-link.disabled { opacity: 0.35; pointer-events: none; }

/* Sample section heading */
section { margin-bottom: 28px; }
.sample-heading { margin: 18px 0 10px 0; }
.sample-badge {
  display: inline-block;
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--accent);
  background: rgba(23,105,170,0.10);
  padding: 4px 10px; border-radius: 6px;
  font-family: 'JetBrains Mono', monospace;
}

/* Layout grids */
.header-row { display: grid; grid-template-columns: 1.25fr 1fr 1fr 1fr 1fr; gap: 14px; }
.panels-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; }

/* Panel cards */
.panel {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px 12px 16px;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.18s ease, border-color 0.18s ease;
  display: flex; flex-direction: column;
  min-width: 0;
}
.panel:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.panel.wide { min-height: 360px; }
.panel.umap .panel-body img { width: 100%; height: auto; display: block; border-radius: 4px; }

.panel-title {
  margin: 0 0 10px 0;
  font-size: 11px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--text);
}
.panel-title .muted {
  text-transform: none; letter-spacing: 0; font-weight: 500; color: var(--text-muted);
}
.panel-body { flex: 1; min-width: 0; }

footer {
  text-align: right; color: var(--text-muted); font-size: 12px;
  padding-top: 10px; border-top: 1px solid var(--border); margin-top: 24px;
}

/* ---- Index page ---- */
.index-header { padding: 18px 0 14px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.index-header h1 { margin: 0; font-size: 22px; font-weight: 600; }
.index-header .subtitle { color: var(--text-muted); font-size: 13px; margin-top: 4px; }

.program-list { display: flex; flex-direction: column; gap: 10px; }
.program-card {
  display: grid;
  grid-template-columns: 130px 1fr auto;
  align-items: center;
  gap: 20px;
  padding: 14px 18px;
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-sm);
  text-decoration: none; color: inherit;
  transition: box-shadow 0.18s ease, border-color 0.18s ease, transform 0.12s ease;
}
.program-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-strong);
  transform: translateY(-1px);
}
.program-card .pid-block {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px; font-weight: 600; color: var(--accent);
}
.program-card .pid-block .pid-label {
  display: block; font-size: 10px; font-weight: 500; color: var(--text-muted);
  letter-spacing: 0.06em; text-transform: uppercase; font-family: 'Inter', sans-serif;
  margin-bottom: 2px;
}
.program-card .pid-block .pid-value { font-size: 18px; }
.program-card .details { min-width: 0; }
.program-card .go-term { font-size: 13px; font-weight: 500; color: var(--text); margin-bottom: 6px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.program-card .chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  display: inline-block;
  font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500;
  color: var(--text);
  background: #f1f3f5;
  border: 1px solid #e6e9ec;
  padding: 2px 7px; border-radius: 4px;
}

.program-card .badge-block { text-align: right; }
.badge {
  display: inline-block;
  font-size: 11px; font-weight: 600;
  padding: 4px 10px; border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
}
.badge.sig { background: rgba(192,57,43,0.10); color: #8a1f12; }
.badge.zero { background: #f0f1f3; color: var(--text-muted); }
.badge-block .badge-label {
  display: block; font-size: 10px; color: var(--text-muted);
  letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px;
  font-family: 'Inter', sans-serif;
}

/* ---- Responsive: stack tightly on narrow screens ---- */
@media (max-width: 1200px) {
  .header-row { grid-template-columns: 1fr 1fr 1fr; }
  .panels-4 { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 720px) {
  .header-row, .panels-4 { grid-template-columns: 1fr; }
  .program-card { grid-template-columns: 1fr; }
}

/* ---- Print ---- */
@media print {
  header.sticky { position: static; background: #fff; backdrop-filter: none; }
  .nav-link { display: none; }
  body { background: #fff; padding: 0 8px; }
  .panel { break-inside: avoid; box-shadow: none; border-color: #ddd; }
  .panel:hover { box-shadow: none; }
}
"""


def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_program_html(
    mdata,
    perturb_path_base,
    GO_path,
    file_to_dictionary,
    Target_Program,
    program_correlation,
    waterfall_correlation,
    sample,
    perturbed_gene_found,
    html_share_path,
    *,
    top_program=5,
    groupby="sample",
    tagert_col_name="program_name",
    plot_col_name="target_name",
    log2fc_col="log2FC",
    top_enrichned_term=10,
    down_thred_log=-0.05,
    up_thred_log=0.05,
    p_value=0.05,
    go_p_value_name="Adjusted P-value",
    go_term_col="Term",
    gene_name_key="symbol",
    subsample_frac=None,
    prev_program_id=None,
    next_program_id=None,
    position_index=None,
    position_total=None,
):
    """Write program_{N}/ subtree under html_share_path."""
    share_root = Path(html_share_path)
    pid = str(Target_Program)
    prog_dir = share_root / f"program_{pid}"
    (prog_dir / "data").mkdir(parents=True, exist_ok=True)
    (prog_dir / "images").mkdir(parents=True, exist_ok=True)

    # UMAP PNG
    _render_umap_png(mdata, Target_Program, prog_dir / "images" / "umap.png", subsample_frac)

    # ---- header panels ----
    top_genes_d = _build_top_genes(mdata, Target_Program, top_enrichned_term, file_to_dictionary, gene_name_key)
    go_d = _build_go_terms(GO_path, Target_Program, top_enrichned_term, go_p_value_name, go_term_col)
    corr_d = _build_correlations(program_correlation, Target_Program, top_program)
    violin_d = _build_violin(mdata, Target_Program, groupby)

    _write_json(prog_dir / "data" / "top_genes.json", top_genes_d)
    _write_json(prog_dir / "data" / "go_terms.json", go_d)
    _write_json(prog_dir / "data" / "correlations.json", corr_d)
    _write_json(prog_dir / "data" / "violin.json", violin_d)

    violin_fig = _make_violin_fig(violin_d)
    go_fig = _make_go_fig(go_d)
    corr_fig = _make_correlations_fig(corr_d)
    top_genes_fig = _make_top_genes_fig(top_genes_d)

    # ---- per-sample panels ----
    sample_blocks = []
    for samp in sample:
        skey = _safe_sample_key(samp)
        perturb_path = f"{perturb_path_base}_{samp}.txt"

        log2fc_d = _build_log2fc(perturb_path, Target_Program, tagert_col_name, plot_col_name,
                                  log2fc_col, top_enrichned_term, p_value, perturbed_gene_found)
        volcano_d = _build_volcano(perturb_path, Target_Program, tagert_col_name, plot_col_name,
                                    log2fc_col, down_thred_log, up_thred_log, p_value, perturbed_gene_found)
        dotplot_d = _build_dotplot(mdata, log2fc_d["regulators"], groupby, gene_name_key)
        waterfall_d = _build_waterfall(waterfall_correlation[samp], Target_Program, top_enrichned_term)

        _write_json(prog_dir / "data" / f"log2fc_{skey}.json", log2fc_d)
        _write_json(prog_dir / "data" / f"volcano_{skey}.json", volcano_d)
        _write_json(prog_dir / "data" / f"dotplot_{skey}.json", dotplot_d)
        _write_json(prog_dir / "data" / f"waterfall_{skey}.json", waterfall_d)

        sample_blocks.append(_SAMPLE_BLOCK.format(
            sample=samp,
            log2fc_div=_fig_to_div(_make_log2fc_fig(log2fc_d), f"div-log2fc-{skey}"),
            volcano_div=_fig_to_div(_make_volcano_fig(volcano_d), f"div-volcano-{skey}"),
            dotplot_div=_fig_to_div(_make_dotplot_fig(dotplot_d), f"div-dotplot-{skey}"),
            waterfall_div=_fig_to_div(_make_waterfall_fig(waterfall_d, Target_Program), f"div-waterfall-{skey}"),
        ))

    # ---- heatmap ----
    heatmap_d = _build_heatmap(perturb_path_base, mdata, Target_Program, sample,
                                tagert_col_name, plot_col_name, log2fc_col, p_value, groupby)
    _write_json(prog_dir / "data" / "heatmap.json", heatmap_d)
    heatmap_fig = _make_heatmap_fig(heatmap_d)

    # ---- nav strings ----
    position = ""
    if position_index is not None and position_total is not None:
        position = f"{position_index} / {position_total}"
    if prev_program_id is None:
        prev_href, prev_disabled = "#", " disabled"
    else:
        prev_href, prev_disabled = f"../program_{prev_program_id}/program_{prev_program_id}.html", ""
    if next_program_id is None:
        next_href, next_disabled = "#", " disabled"
    else:
        next_href, next_disabled = f"../program_{next_program_id}/program_{next_program_id}.html", ""

    # ---- assemble page ----
    page = _PAGE_TEMPLATE.format(
        pid=pid,
        cdn=PLOTLY_CDN,
        fonts=GOOGLE_FONTS,
        position=position,
        prev_href=prev_href, prev_disabled=prev_disabled,
        next_href=next_href, next_disabled=next_disabled,
        violin_div=_fig_to_div(violin_fig, "div-violin"),
        go_div=_fig_to_div(go_fig, "div-go"),
        corr_div=_fig_to_div(corr_fig, "div-corr"),
        top_genes_div=_fig_to_div(top_genes_fig, "div-top-genes"),
        sample_blocks="".join(sample_blocks),
        heatmap_div=_fig_to_div(heatmap_fig, "div-heatmap"),
        timestamp=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    with open(prog_dir / f"program_{pid}.html", "w") as f:
        f.write(page)

    # ---- metadata.json ----
    n_sig_total = 0
    top_go_term = go_d["terms"][0] if go_d["terms"] else None
    for samp in sample:
        skey = _safe_sample_key(samp)
        with open(prog_dir / "data" / f"volcano_{skey}.json") as f:
            v = json.load(f)
        n_sig_total += sum(1 for c in v["category"] if c != "ns")
    metadata = {
        "program_id": pid,
        "samples": list(map(str, sample)),
        "thresholds": {"p_value": float(p_value), "down_thred_log": float(down_thred_log),
                        "up_thred_log": float(up_thred_log)},
        "top_loaded_genes": top_genes_d["genes"][::-1],
        "top_correlated_programs": {
            "positive": [p for p, d in zip(corr_d["programs"], corr_d["direction"]) if d == "positive"],
            "negative": [p for p, d in zip(corr_d["programs"], corr_d["direction"]) if d == "negative"],
        },
        "top_GO_terms": [{"term": t, "adj_pval": p} for t, p in zip(go_d["terms"], go_d["adj_pval"])],
        "top_GO_term": top_go_term,
        "n_significant_regulators_total": n_sig_total,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    _write_json(prog_dir / "metadata.json", metadata)


def write_share_index(html_share_path, program_ids, config_snapshot):
    """Write index.html, shared/style.css, shared/manifest.json."""
    share_root = Path(html_share_path)
    (share_root / "shared").mkdir(parents=True, exist_ok=True)

    with open(share_root / "shared" / "style.css", "w") as f:
        f.write(_STYLE_CSS)

    manifest = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "program_ids": [str(p) for p in program_ids],
        "config": {k: v for k, v in config_snapshot.items()
                   if not isinstance(v, (np.ndarray,))},
    }
    _write_json(share_root / "shared" / "manifest.json", manifest)

    cards = []
    for pid in program_ids:
        pid = str(pid)
        meta_path = share_root / f"program_{pid}" / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            m = json.load(f)
        top_genes = m.get("top_loaded_genes", [])[:5]
        top_go = m.get("top_GO_term") or "—"
        n_sig = int(m.get("n_significant_regulators_total", 0))

        chips = "".join(f'<span class="chip">{g}</span>' for g in top_genes) or '<span class="chip" style="opacity:0.5">—</span>'
        badge_cls = "sig" if n_sig > 0 else "zero"

        cards.append(
            f'<a class="program-card" href="program_{pid}/program_{pid}.html">'
            f'  <div class="pid-block">'
            f'    <span class="pid-label">Program</span>'
            f'    <span class="pid-value">{pid}</span>'
            f'  </div>'
            f'  <div class="details">'
            f'    <div class="go-term" title="{top_go}">{top_go}</div>'
            f'    <div class="chips">{chips}</div>'
            f'  </div>'
            f'  <div class="badge-block">'
            f'    <span class="badge {badge_cls}">{n_sig}</span>'
            f'    <span class="badge-label">sig regulators</span>'
            f'  </div>'
            f'</a>'
        )

    n_progs = len(cards)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>cNMF programs — share index</title>"
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{GOOGLE_FONTS}" rel="stylesheet">'
        "<link rel='stylesheet' href='shared/style.css'>"
        "</head><body>"
        f"<header class='index-header'><h1>cNMF program analysis</h1>"
        f"<div class='subtitle'>{n_progs} programs · top loaded genes, top GO term, and total significant regulators per program</div>"
        "</header>"
        f"<section class='program-list'>{''.join(cards)}</section>"
        f"<footer>Generated {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</footer>"
        "</body></html>"
    )
    with open(share_root / "index.html", "w") as f:
        f.write(html)
