"""
Diagnostic plots for the CRT.

plot_beta_null: one gene's resampled (null) betas, one panel per program. Each
panel shows a histogram of the null betas, a vertical line at beta_obs, and,
where the skew-normal p was used, the fitted skew-normal density. The fit lives
on the standardized scale z = (beta - mean) / sd (skew_normal.py), so on the
beta axis the density is skewnorm.pdf(z) / sd.

plot_qq_real_vs_null: QQ plots (-log10 p vs Uniform) of the real genes and the
non-targeting fake genes (null_guide_assignment_preparation.py) on one axis,
one figure for the empirical p-values and one for the skew-normal p-values;
with per_program=True, also one panel per program.
The skew-normal plot uses the final p (skew-normal where the fit was used,
empirical elsewhere), so every test is on it.
"""

import os
import re
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import skewnorm


def plot_beta_null(
    gene: str,
    beta_obs: np.ndarray,
    beta_null: np.ndarray,
    pvalue_empirical: np.ndarray,
    plot_dir: str,
    program_names: Optional[Sequence[str]] = None,
    pvalue_skewnorm: Optional[np.ndarray] = None,
    skew_params: Optional[np.ndarray] = None,
    flagged: Optional[np.ndarray] = None,
    n_cols: int = 6,
    bins: int = 50,
) -> str:
    """
    Save <plot_dir>/<gene>_beta_null.pdf with one panel per program.

    Args:
        gene: target gene name (used in the title and file name).
        beta_obs: length-K observed betas.
        beta_null: n_rounds x K null betas; NaN rounds are dropped.
        pvalue_empirical: length-K empirical p-values.
        plot_dir: output directory (created if missing).
        program_names: K panel labels; defaults to the 1-based program index.
        pvalue_skewnorm, skew_params: skew-normal output (None if not run); the fitted
            density is drawn where pvalue_skewnorm is finite.
        flagged: programs the skew-normal fit was tried on; a tried but rejected
            fit is labelled "SN rejected".
        n_cols: panels per row.
        bins: histogram bins.

    Returns:
        Path of the saved PDF.
    """
    import matplotlib

    matplotlib.use("Agg")  # also runs inside joblib workers with no display
    import matplotlib.pyplot as plt

    n_programs = beta_obs.shape[0]
    names = (
        list(program_names)
        if program_names is not None
        else [str(k + 1) for k in range(n_programs)]
    )
    flagged_set = set(np.asarray(flagged).tolist()) if flagged is not None else set()
    n_rows = int(np.ceil(n_programs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), squeeze=False
    )

    for k, ax in enumerate(axes.flat):
        if k >= n_programs:
            ax.axis("off")
            continue
        null = beta_null[:, k]
        null = null[np.isfinite(null)]
        if null.size == 0:
            ax.set_title(f"{names[k]}\nno valid null rounds", fontsize=8)
            continue

        ax.hist(null, bins=bins, density=True, color="0.75", edgecolor="none")
        label = f"p_raw={pvalue_empirical[k]:.2g}"

        if pvalue_skewnorm is not None and np.isfinite(pvalue_skewnorm[k]):
            xi, omega, alpha = skew_params[k]
            mean, sd = null.mean(), null.std()
            x = np.linspace(
                min(null.min(), beta_obs[k]), max(null.max(), beta_obs[k]), 400
            )
            ax.plot(
                x,
                skewnorm.pdf((x - mean) / sd, alpha, loc=xi, scale=omega) / sd,
                color="tab:blue",
                lw=1.2,
            )
            label += f"  p_sn={pvalue_skewnorm[k]:.2g}"
        elif pvalue_skewnorm is not None and k in flagged_set:
            label += "  SN rejected"

        if np.isfinite(beta_obs[k]):
            ax.axvline(beta_obs[k], color="tab:red", lw=1.2)
            lo, hi = ax.get_xlim()  # keep beta_obs in view even far outside the null
            pad = 0.05 * (max(hi, beta_obs[k]) - min(lo, beta_obs[k]))
            ax.set_xlim(min(lo, beta_obs[k] - pad), max(hi, beta_obs[k] + pad))

        ax.set_title(f"{names[k]}\n{label}", fontsize=8)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"{gene}: null betas ({beta_null.shape[0]} rounds); red = beta_obs, blue = skew-normal fit",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, re.sub(r"[^\w.-]", "_", gene) + "_beta_null.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def _pval_matrix(obj, kind: str) -> np.ndarray:
    """genes x programs p-values of one CRT_obj; kind = "empirical" or "skew_normal" (final p)."""
    rows = []
    for r in obj.gene_results.values():
        if kind == "empirical" and r.pvalue_empirical is not None:
            rows.append(r.pvalue_empirical)
        else:  # final p; untestable genes have p = 1 and no empirical p
            rows.append(r.pvalue)
    return np.vstack(rows) if rows else np.empty((0, len(obj.selected_program_names)))


def _finite(p: np.ndarray) -> np.ndarray:
    p = np.ravel(p)
    return p[np.isfinite(p)]


def _qq(ax, p: np.ndarray, label: str, color: str) -> float:
    """
    -log10 sorted p against -log10 Uniform quantiles (i - 0.5) / n.
    Returns the largest -log10 value plotted on either axis (0 if p is empty).
    """
    p = np.sort(p)
    expected = -np.log10((np.arange(1, p.size + 1) - 0.5) / p.size)
    observed = -np.log10(p)
    ax.scatter(expected, observed, s=4, color=color, label=label, rasterized=True, lw=0)
    return float(max(expected.max(), observed.max())) if p.size else 0.0


def _plot_qq_per_program(
    P_real, P_null, program_names, path, title, p_line=0.05, floor=None, n_cols=6
) -> None:
    """
    One QQ panel per program (column of P_real / P_null): real (red) vs null (grey),
    with a dotted gray line at p = p_line and a blue line at the empirical p floor.
    """
    import matplotlib.pyplot as plt

    n_programs = len(program_names)
    n_rows = int(np.ceil(n_programs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.5 * n_rows), squeeze=False
    )
    x_max = 1.05 * -np.log10(0.5 / max(P_real.shape[0], P_null.shape[0], 1))
    for k, ax in enumerate(axes.flat):
        if k >= n_programs:
            ax.axis("off")
            continue
        p_real, p_null = _finite(P_real[:, k]), _finite(P_null[:, k])
        if p_real.size:
            _qq(ax, p_real, "real", "tab:red")
        if p_null.size:
            _qq(ax, p_null, "null", "0.4")
        ax.plot([0, x_max], [0, x_max], color="k", lw=0.6, ls="--")
        ax.axhline(-np.log10(p_line), color="0.6", lw=0.5, ls=":")
        y_max = max(ax.get_ylim()[1], x_max)
        if floor is not None:
            ax.axhline(-np.log10(floor), color="tab:blue", lw=0.5, ls="-.")
            y_max = max(y_max, 1.05 * -np.log10(floor))
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        frac = float(np.mean(p_null < 0.05)) if p_null.size else float("nan")
        ax.set_title(f"{program_names[k]}  null p<0.05: {frac:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(
        title
        + "  (red = real genes, grey = NTC null genes; x = expected, y = observed -log10 p;"
        + f" dotted = p {p_line}, blue = empirical p floor)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_qq_real_vs_null(
    real_obj,
    null_obj,
    plot_dir: str,
    prefix: str = "crt",
    per_program: bool = False,
    n_cols: int = 6,
    fmt: str = "pdf",
    font_size: int = 14,
) -> Dict[str, str]:
    """
    Save <plot_dir>/<prefix>_qq_empirical.pdf and <prefix>_qq_skew_normal.pdf:
    real genes (red) and non-targeting fake genes (gray) against Uniform(0, 1),
    all programs pooled. Each plot is square (6 x 6 in, both axes 0..max over
    expected and observed -log10 p, equal aspect), so y = x runs at 45 degrees. With per_program=True, also save
    <prefix>_qq_<kind>_per_program.pdf with one panel per program.

    Args:
        real_obj: CRT_obj after conditional_randomized_test on the real genes.
        null_obj: CRT_obj after conditional_randomized_test on the fake genes
            (prepare_null_guide_assignment).
        plot_dir: output directory (created if missing).
        prefix: file name prefix.
        per_program: also draw the per-program panel grids.
        n_cols: panels per row of the per-program grids.
        fmt: file format / extension of the plots (e.g. "pdf", "png").
        font_size: font size of the pooled plots' axis labels and tick labels;
            the title and legend use 0.75 * font_size.

    Returns:
        {"empirical": path, "skew_normal": path, plus "<kind>_per_program": path
        when per_program=True}; the skew-normal plots are skipped
        (with a warning) unless both objects ran with pvalue_method="skew_normal".
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["empirical"]
    if (
        real_obj.pvalue_method == "skew_normal"
        and null_obj.pvalue_method == "skew_normal"
    ):
        kinds.append("skew_normal")
    else:
        warnings.warn(
            'Skipping the skew-normal QQ plot: both CRT_obj need pvalue_method="skew_normal".'
        )

    os.makedirs(plot_dir, exist_ok=True)
    p_line = 0.05  # dotted gray reference line
    # smallest empirical p: 1 / (1 + all resampling rounds)
    floor = 1.0 / (1.0 + real_obj.resampled_rounds + real_obj.extra_resampled_rounds)
    paths: Dict[str, str] = {}
    for kind in kinds:
        P_real, P_null = _pval_matrix(real_obj, kind), _pval_matrix(null_obj, kind)
        p_real, p_null = _finite(P_real), _finite(P_null)
        fig, ax = plt.subplots(figsize=(6, 6))
        top_real = _qq(
            ax,
            p_real,
            f"real genes ({len(real_obj.gene_results)} genes, {p_real.size:,} tests)",
            "tab:red",
        )
        top_null = _qq(
            ax,
            p_null,
            f"NTC null genes ({len(null_obj.gene_results)} genes, {p_null.size:,} tests)",
            "gray",
        )
        # one shared 0..max range on both axes, so y = x sits at 45 degrees
        lim = 1.05 * max(top_real, top_null, -np.log10(floor), 1.0)
        ax.plot([0, lim], [0, lim], color="k", lw=0.8, ls="--", alpha=0.5)
        ax.axhline(
            -np.log10(p_line), color="0.6", lw=0.8, ls=":", label=f"p = {p_line}"
        )
        ax.axhline(
            -np.log10(floor),
            color="tab:blue",
            lw=0.8,
            ls="-.",
            label=f"empirical p floor ({floor:.1e})",
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        small = 0.75 * font_size  # title and legend
        ax.set_xlabel("expected -log10 p (Uniform)", fontsize=font_size)
        ax.set_ylabel("observed -log10 p", fontsize=font_size)
        ax.tick_params(labelsize=font_size)
        # one short line per piece, so the title fits the 6-inch width
        name = (
            "empirical p"
            if kind == "empirical"
            else "skew-normal p\n(empirical where not fit)"
        )
        frac = float(np.mean(p_null < 0.05)) if p_null.size else float("nan")
        ax.set_title(
            f"CRT QQ ({real_obj.moi} MOI): {name}\nnull frac p<0.05 = {frac:.3f}",
            fontsize=small,
        )
        ax.legend(frameon=False, fontsize=small, loc="lower right")
        fig.tight_layout()
        paths[kind] = os.path.join(plot_dir, f"{prefix}_qq_{kind}.{fmt}")
        fig.savefig(paths[kind], dpi=200)
        plt.close(fig)

        if per_program:
            paths[f"{kind}_per_program"] = os.path.join(
                plot_dir, f"{prefix}_qq_{kind}_per_program.{fmt}"
            )
            _plot_qq_per_program(
                P_real,
                P_null,
                list(real_obj.selected_program_names),
                paths[f"{kind}_per_program"],
                f"CRT QQ ({real_obj.moi} MOI): {name}",
                p_line,
                floor,
                n_cols=n_cols,
            )
    return paths
