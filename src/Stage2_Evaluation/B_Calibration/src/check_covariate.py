#!/usr/bin/env python3
"""
Covariate independence diagnostic for CRT calibration.

CRT (``CRT.py``) injects the ``--covariates`` / ``--log_covariates`` columns into the
propensity/regression design matrix. If those covariates are mutually redundant
(collinear), the design matrix ``C^T C`` becomes ill-conditioned and CRT p-values are
unreliable. This module lets CRT vet a covariate selection *before* running the
(expensive) test.

It encodes covariates exactly the way CRT does (reusing
``encode_categorical_covariates`` from the vendored ``CRT`` package, with the same
``numeric_as_category_threshold=20`` used by ``get_covar_matrix``), then reports:

  - a Pearson correlation heatmap over the encoded design columns, and
  - a Variance Inflation Factor (VIF) table quantifying multicollinearity per column.

Outputs (under the ``save_path`` passed by the caller, e.g.
``<out_dir>/<run_name>/Evaluation/covariate_check/``):
  - ``covariate_correlation.png``  — heatmap
  - ``covariate_correlation.tsv``  — correlation matrix
  - ``covariate_vif.tsv``          — VIF per encoded column (descending)

This is a library module (no CLI). Call ``run_covariate_check(adata, ...)`` — CRT.py
invokes it when ``--check_covariate`` is set.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless / SLURM-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The vendored CRT package (src/CRT/) is importable because CRT.py (or any caller)
# puts B_Calibration/src on sys.path — same encoding path as CRT, so the covariate
# encoding here matches CRT exactly.
from CRT import encode_categorical_covariates


def build_covar_df(adata, covariates=None, log_covariates=None):
    """Assemble the covariate DataFrame mirroring CRT.reformat_data_for_CRT.

    ``covariates`` columns are taken from ``adata.obs`` as-is; ``log_covariates``
    columns are log1p-transformed and stored as ``log_<key>``.
    """
    covar_dict = {}
    if covariates:
        for key in covariates:
            if key not in adata.obs:
                raise KeyError(f"covariate '{key}' not found in .obs")
            covar_dict[key] = adata.obs[key]
    if log_covariates:
        for key in log_covariates:
            if key not in adata.obs:
                raise KeyError(f"log_covariate '{key}' not found in .obs")
            covar_dict[f"log_{key}"] = np.log1p(adata.obs[key])

    if not covar_dict:
        raise ValueError("No covariates provided; pass covariates and/or log_covariates.")

    return pd.DataFrame(covar_dict, index=adata.obs_names)


def compute_vif(encoded_df):
    """Return a DataFrame of VIF per encoded column.

    A constant column is prepended (VIF is defined relative to an intercept). Columns
    that are perfectly collinear yield inf; a singular design yields NaN — both are
    reported rather than raising, so the offending columns are visible.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = encoded_df.to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(X.shape[0]), X])  # intercept at index 0
    cols = list(encoded_df.columns)

    vifs = []
    for i, col in enumerate(cols, start=1):  # skip the intercept column
        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = np.nan
        vifs.append((col, v))

    vif_df = pd.DataFrame(vifs, columns=["covariate", "VIF"])
    return vif_df.sort_values("VIF", ascending=False, na_position="first").reset_index(drop=True)


def plot_correlation_heatmap(corr, png_path, title):
    """Render a Pearson correlation heatmap (diverging cmap centered at 0)."""
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 3), max(5, 0.7 * n + 2)))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    # annotate cells with the correlation value
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(val) > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def run_check(covar_df, save_path, prefix, threshold, corr_flag=0.8, vif_flag=5.0):
    """One-hot encode, compute correlation + VIF, write outputs, print a summary."""
    encoded = encode_categorical_covariates(
        covar_df,
        drop_first=True,
        numeric_as_category_threshold=threshold,
    )
    # get_dummies emits bool columns; cast so corr/VIF treat them numerically.
    encoded = encoded.astype(np.float64)

    corr = encoded.corr(method="pearson")
    vif_df = compute_vif(encoded)

    os.makedirs(save_path, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""
    corr_tsv = os.path.join(save_path, f"{tag}covariate_correlation.tsv")
    vif_tsv = os.path.join(save_path, f"{tag}covariate_vif.tsv")
    png = os.path.join(save_path, f"{tag}covariate_correlation.png")

    corr.to_csv(corr_tsv, sep="\t")
    vif_df.to_csv(vif_tsv, sep="\t", index=False)
    title = "Covariate correlation" + (f" ({prefix})" if prefix else "")
    plot_correlation_heatmap(corr, png, title)

    # --- stdout summary ---
    label = f"[{prefix}] " if prefix else ""
    print(f"\n{label}Encoded covariate columns ({encoded.shape[1]}): {list(encoded.columns)}")

    # high-correlation off-diagonal pairs
    high_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > corr_flag:
                high_pairs.append((cols[i], cols[j], r))
    if high_pairs:
        print(f"{label}WARNING: |Pearson r| > {corr_flag} pairs (potential redundancy):")
        for a, b, r in sorted(high_pairs, key=lambda t: -abs(t[2])):
            print(f"    {a}  <->  {b} : r = {r:+.2f}")
    else:
        print(f"{label}No covariate pair exceeds |r| > {corr_flag}.")

    high_vif = vif_df[vif_df["VIF"] > vif_flag]
    if len(high_vif):
        print(f"{label}WARNING: VIF > {vif_flag} (multicollinearity):")
        for _, row in high_vif.iterrows():
            print(f"    {row['covariate']} : VIF = {row['VIF']:.2f}")
    else:
        print(f"{label}All VIF <= {vif_flag}.")

    print(f"{label}Wrote:\n    {png}\n    {corr_tsv}\n    {vif_tsv}")


def run_covariate_check(adata, save_path, covariates=None, log_covariates=None,
                        categorical_key=None, per_condition=False,
                        numeric_as_category_threshold=20):
    """High-level entry point: build the covariate design and run correlation + VIF.

    Intended for programmatic use (e.g. CRT.py's ``--check_covariate``). Takes an
    already-loaded AnnData (the cNMF modality of the run's MuData) whose ``.obs``
    holds the covariate columns.

    Parameters
    ----------
    adata : AnnData
        Modality whose ``.obs`` holds the covariate columns.
    save_path : str
        Output directory for the heatmap / TSVs.
    covariates, log_covariates : list[str] or None
        Same semantics as CRT's ``--covariates`` / ``--log_covariates``.
    categorical_key : str or None
        obs key defining conditions; required when ``per_condition=True``.
    per_condition : bool
        If True, run the diagnostic separately per level of ``categorical_key``
        (rows sliced first, so one-hot drops unused levels exactly like CRT, which
        builds a separate design matrix per condition). If False, run once over all
        cells.
    numeric_as_category_threshold : int
        Numeric columns with <= this many unique values are one-hot encoded
        (matches CRT ``get_covar_matrix`` default: 20).
    """
    covar_df = build_covar_df(adata, covariates=covariates, log_covariates=log_covariates)

    if per_condition:
        if not categorical_key:
            raise ValueError("per_condition=True requires categorical_key.")
        if categorical_key not in adata.obs:
            raise KeyError(f"categorical_key '{categorical_key}' not found in .obs")
        conditions = list(pd.unique(adata.obs[categorical_key]))
        print(f"Per-condition covariate check over '{categorical_key}': {conditions}")
        for cond in conditions:
            mask = (adata.obs[categorical_key] == cond).to_numpy()
            run_check(
                covar_df.loc[mask], save_path, prefix=str(cond),
                threshold=numeric_as_category_threshold,
            )
    else:
        run_check(
            covar_df, save_path, prefix="",
            threshold=numeric_as_category_threshold,
        )
