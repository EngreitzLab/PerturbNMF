"""
QQ plot helpers for CRT p-value calibration checks.
"""

from typing import Mapping

import numpy as np
import pandas as pd


def _flatten_pvals_any(obj) -> np.ndarray:
    """
    Flatten p-values from a DataFrame/Series, a mapping of DataFrames (e.g. an
    ensemble dict like ntc_group_pvals_ens), or any array-like into a 1-D finite
    array clipped to (0, 1].
    """
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        arr = obj.to_numpy().ravel()
    elif isinstance(obj, Mapping):
        parts = []
        for df in obj.values():
            a = df.to_numpy() if isinstance(df, (pd.DataFrame, pd.Series)) else np.asarray(df)
            parts.append(np.asarray(a, dtype=np.float64).ravel())
        arr = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    else:
        arr = np.asarray(obj, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite p-values to plot.")
    return np.clip(arr.astype(np.float64), 1e-300, 1.0)


def _qq_xy(pvals: np.ndarray):
    """Return (x, y) = (expected, observed) on the -log10 scale for a QQ plot."""
    m = pvals.size
    expected = (np.arange(1, m + 1) - 0.5) / m  # uniform order-statistic means
    x = -np.log10(expected)
    y = -np.log10(np.sort(pvals))
    return x, y


def qq_plot_real_vs_null(real_pvals, null_pvals, ax=None, title=None):
    """
    QQ plot comparing real perturbation p-values against fake/null (NTC) p-values,
    mirroring the U-test's plot_qq_comparison input style (pass the two p-value
    sets directly). Colors/labels/style are fixed to the CRT convention.

    real_pvals: real perturbation p-values (DataFrame genes x programs, Series,
        array, or a mapping of DataFrames).
    null_pvals: fake/negative-control p-values, e.g. the NTC group ensemble
        (mapping like ntc_group_pvals_ens), a DataFrame, Series, or array.

    Well-calibrated null p-values track the y=x diagonal; real points rising
    above it indicate true perturbation signal.
    """
    real = _flatten_pvals_any(real_pvals)
    null = _flatten_pvals_any(null_pvals)
    x_real, y_real = _qq_xy(real)
    x_null, y_null = _qq_xy(null)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 5))

    # Fixed CRT style/colors: real = purple ("All observed"), null = blue (NTC).
    xmax = max(float(x_real.max()), float(x_null.max()))
    ax.plot([0.0, xmax], [0.0, xmax], color="#333333", linewidth=1.0, label="y = x (null)")
    ax.scatter(x_null, y_null, label="Null (NTC / fake)", color="#1f77b4",
               marker=".", s=14.0, alpha=0.6)
    ax.scatter(x_real, y_real, label="Real (perturbations)", color="#9467bd",
               marker=".", s=14.0, alpha=0.6)

    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    ax.set_title(title if title is not None else "CRT QQ plot: real vs null (NTC) p-values")
    ax.legend()
    return ax
