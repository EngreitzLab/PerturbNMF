"""
QQ plot helpers for CRT p-value calibration checks.
"""

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .diagnostics import (
    crt_null_pvals_from_null_stats_fast,
    qq_expected_grid,
)
from .skew_normal import fit_skew_normal


def qq_plot_ntc_pvals(
    pvals_raw_df: Optional[pd.DataFrame],
    guide2gene: Mapping[str, str],
    ntc_genes: Iterable[str],
    *,
    pvals_skew_df: Optional[pd.DataFrame] = None,
    null_pvals: Optional[Sequence[float]] = None,
    null_stats: Optional[Sequence[float]] = None,
    null_two_sided: bool = True,
    show_null_skew: bool = False,
    null_skew_samples: Optional[int] = None,
    null_skew_seed: Optional[int] = 0,
    expected_grid_raw: Optional[Sequence[float]] = None,
    expected_grid_all: Optional[Sequence[float]] = None,
    ntc_group_pvals_ens: Optional[Mapping[int, pd.DataFrame]] = None,
    ntc_group_pvals_skew_ens: Optional[Mapping[int, pd.DataFrame]] = None,
    show_ntc_ensemble_band: bool = False,
    ntc_band_quantiles: Optional[Sequence[float]] = None,
    ntc_band_alpha: float = 0.15,
    ax=None,
    title: Optional[str] = None,
    label_ntc_raw: str = "NTC (raw)",
    label_skew: str = "NTC (skew)",
    label_null: str = "null",
    label_null_skew: str = "null (skew)",
    label_all: str = "All observed",
    color_ntc_raw: str = "#1f77b4",
    color_skew: str = "#ff7f0e",
    color_null: str = "#7f7f7f",
    color_null_skew: str = "#2ca02c",
    color_all: str = "#9467bd",
    all_marker: str = ".",
    all_marker_size: float = 12.0,
    all_alpha: float = 0.6,
    show_ref_line: bool = True,
    show_conf_band: bool = True,
    conf_alpha: float = 0.05,
    conf_color: str = "#d9d9d9",
    show_all_pvals: bool = True,
):
    """
    QQ plot comparing NTC (negative-control) p-values to a CRT-null reference.
    If pvals_skew_df is provided, plots both raw and skew-calibrated curves.
    Provide null_pvals directly or pass null_stats to compute leave-one-out
    CRT-null p-values. Optionally plot a skew-normal null curve by fitting
    to null_stats and sampling from the fitted distribution. If show_all_pvals
    is True, plot all observed p-values from pvals_raw_df.

    Notes:
        - null_stats: raw CRT null test statistics (e.g., beta_null values)
        - null_pvals: leave-one-out CRT-null p-values computed from null_stats
          (these should be ~Uniform(0,1) under a correct null)
        - If null_stats is provided without null_pvals, this function computes
          leave-one-out null p-values internally.
        - If ntc_group_pvals_ens is provided and you want a skew curve, pass
          ntc_group_pvals_skew_ens to use group-based skew p-values.
        - When ntc_group_pvals_ens is provided, NTC curves concatenate values
          across all groups and ensemble partitions.
    """
    if pvals_raw_df is None:
        raise ValueError("pvals_raw_df is required.")
    if null_pvals is None and null_stats is None:
        raise ValueError("Provide null_pvals or null_stats.")
    if show_null_skew and null_stats is None:
        raise ValueError("null_stats is required when show_null_skew=True.")
    ntc = list(dict.fromkeys(ntc_genes))
    if len(ntc) == 0:
        raise ValueError("ntc_genes must contain at least one gene name.")

    if ntc_group_pvals_ens is None:
        gene_names = set(guide2gene.values())
        missing = [g for g in ntc if g not in gene_names]
        if missing:
            raise ValueError(
                "ntc_genes not found in guide2gene values: "
                + ", ".join(sorted(missing))
            )

    def _extract_pvals(df: pd.DataFrame, label: str) -> np.ndarray:
        missing_idx = [g for g in ntc if g not in df.index]
        if missing_idx:
            raise ValueError(
                f"ntc_genes not found in {label} index: "
                + ", ".join(sorted(missing_idx))
            )
        pvals = df.loc[ntc].to_numpy().ravel()
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size == 0:
            raise ValueError(f"No finite p-values available for {label}.")
        return np.clip(pvals, 1e-300, 1.0)

    def _qq_data(pvals: np.ndarray, expected: Optional[Sequence[float]] = None):
        m = pvals.size
        if expected is None:
            expected = qq_expected_grid(pvals)
        else:
            expected = np.asarray(expected, dtype=np.float64)
            if expected.size != m:
                raise ValueError(
                    "expected grid length does not match p-values length."
                )
        x = -np.log10(expected)
        y = -np.log10(np.sort(pvals))
        return x, y, m

    def _normalize_null_pvals(pvals: Sequence[float]) -> np.ndarray:
        arr = np.asarray(pvals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("null_pvals contains no finite values.")
        return np.clip(arr, 1e-300, 1.0)

    def _extract_all_pvals(df: pd.DataFrame) -> np.ndarray:
        pvals = df.to_numpy().ravel()
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size == 0:
            raise ValueError("No finite p-values available in pvals_raw_df.")
        return np.clip(pvals, 1e-300, 1.0)

    if ntc_group_pvals_ens is not None:
        pvals_list = []
        for df in ntc_group_pvals_ens.values():
            arr = df.to_numpy().ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                pvals_list.append(arr)
        if not pvals_list:
            raise ValueError("ntc_group_pvals_ens contains no finite values.")
        pvals_raw = np.concatenate(pvals_list)
    else:
        pvals_raw = _extract_pvals(pvals_raw_df, "pvals_raw_df")
    x_raw, y_raw, m_raw = _qq_data(pvals_raw, expected=expected_grid_raw)

    if pvals_skew_df is not None:
        if ntc_group_pvals_ens is not None:
            if ntc_group_pvals_skew_ens is None:
                raise ValueError(
                    "Provide ntc_group_pvals_skew_ens when using grouped "
                    "NTC controls with skew p-values."
                )
            pvals_list = []
            for df in ntc_group_pvals_skew_ens.values():
                arr = df.to_numpy().ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    pvals_list.append(arr)
            if not pvals_list:
                raise ValueError(
                    "ntc_group_pvals_skew_ens contains no finite values."
                )
            pvals_skew = np.concatenate(pvals_list)
        else:
            pvals_skew = _extract_pvals(pvals_skew_df, "pvals_skew_df")
        x_skew, y_skew, m_skew = _qq_data(pvals_skew)
    else:
        x_skew = y_skew = None
        m_skew = 0

    if show_all_pvals:
        all_pvals = _extract_all_pvals(pvals_raw_df)
        x_all, y_all, _ = _qq_data(all_pvals, expected=expected_grid_all)
    else:
        x_all = y_all = None

    if null_pvals is None:
        null_arr = crt_null_pvals_from_null_stats_fast(
            np.asarray(null_stats, dtype=np.float64), two_sided=null_two_sided
        )
    else:
        null_arr = _normalize_null_pvals(null_pvals)
    x_null, y_null, m_null = _qq_data(null_arr)

    x_null_skew = y_null_skew = None
    if show_null_skew:
        stats = np.asarray(null_stats, dtype=np.float64)
        stats = stats[np.isfinite(stats)]
        if stats.size == 0:
            raise ValueError("null_stats contains no finite values.")
        mu = stats.mean()
        sd = stats.std()
        if not np.isfinite(sd) or sd <= 0.0:
            raise ValueError("null_stats must have non-zero variance.")

        z_null = (stats - mu) / sd
        params = fit_skew_normal(z_null)
        if not np.all(np.isfinite(params[:3])):
            raise ValueError("Skew-normal fit failed on null_stats.")

        n_draw = stats.size if null_skew_samples is None else int(null_skew_samples)
        if n_draw <= 0:
            raise ValueError("null_skew_samples must be positive.")

        from scipy.stats import skewnorm

        dist = skewnorm(params[2], loc=params[0], scale=params[1])
        rng = np.random.default_rng(null_skew_seed)
        z_samp = dist.rvs(size=n_draw, random_state=rng)
        u = dist.cdf(z_samp)
        if null_two_sided:
            p_skew = 2.0 * np.minimum(u, 1.0 - u)
        else:
            p_skew = 1.0 - u
        p_skew = np.clip(p_skew, 1e-300, 1.0)
        x_null_skew, y_null_skew, _ = _qq_data(p_skew)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(6, 5))

    if show_conf_band:
        from scipy.stats import beta

        i = np.arange(1, m_null + 1)
        lower = beta.ppf(conf_alpha / 2.0, i, m_null - i + 1)
        upper = beta.ppf(1.0 - conf_alpha / 2.0, i, m_null - i + 1)
        lower = -np.log10(np.clip(lower, 1e-300, 1.0))
        upper = -np.log10(np.clip(upper, 1e-300, 1.0))
        expected = (np.arange(1, m_null + 1) - 0.5) / m_null
        x_band = -np.log10(expected)
        ax.fill_between(
            x_band, lower, upper, color=conf_color, alpha=0.5, label="95% CI"
        )

    if show_ref_line:
        x_candidates = [x_raw, x_null]
        if x_skew is not None:
            x_candidates.append(x_skew)
        if x_null_skew is not None:
            x_candidates.append(x_null_skew)
        if x_all is not None:
            x_candidates.append(x_all)
        xmin = min(float(np.min(x)) for x in x_candidates)
        xmax = max(float(np.max(x)) for x in x_candidates)
        ax.plot([xmin, xmax], [xmin, xmax], color="#333333", linewidth=1.0, label="y=x")

    if x_all is not None:
        ax.scatter(
            x_all,
            y_all,
            label=label_all,
            color=color_all,
            marker=all_marker,
            s=all_marker_size,
            alpha=all_alpha,
        )
    ax.plot(x_raw, y_raw, label=label_ntc_raw, color=color_ntc_raw)
    if x_skew is not None:
        ax.plot(x_skew, y_skew, label=label_skew, color=color_skew)
    ax.plot(x_null, y_null, label=label_null, color=color_null, linestyle="--")
    if x_null_skew is not None:
        ax.plot(
            x_null_skew,
            y_null_skew,
            label=label_null_skew,
            color=color_null_skew,
            linestyle=":",
        )

    if ntc_group_pvals_ens is not None and show_ntc_ensemble_band:
        q_grid = (
            np.linspace(0.01, 0.99, 200)
            if ntc_band_quantiles is None
            else np.asarray(ntc_band_quantiles, dtype=np.float64)
        )
        q_grid = q_grid[(q_grid > 0.0) & (q_grid < 1.0)]
        if q_grid.size == 0:
            raise ValueError("ntc_band_quantiles must contain values in (0,1).")

        curves = []
        for df in ntc_group_pvals_ens.values():
            arr = df.to_numpy().ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                curves.append(np.quantile(arr, q_grid))
        if curves:
            curves = np.vstack(curves)
            median = np.median(curves, axis=0)
            lower = np.quantile(curves, 0.25, axis=0)
            upper = np.quantile(curves, 0.75, axis=0)
            x_band = -np.log10(q_grid)
            ax.fill_between(
                x_band,
                -np.log10(lower),
                -np.log10(upper),
                color=color_ntc_raw,
                alpha=ntc_band_alpha,
                label="NTC ensemble IQR",
            )
    ax.set_xlabel("Expected -log10(p)")
    ax.set_ylabel("Observed -log10(p)")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax
