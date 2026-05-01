"""
Diagnostic plots for cNMF inference test runs.

Generates visual QC plots after inference completes:
1. Gene loading elbow curves per program (ranked log-loadings with knee detection)
2. K-selection plot (stability vs error across K values)
3. Program usage heatmap (cells x programs, grouped by batch)
4. Loading distribution violin plots (per program)

Usage:
    from plot_diagnostics import generate_all_plots
    generate_all_plots(run_dir, run_name="Inference", K=[5,10], sel_thresh=[2.0])
"""

import os
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mudata as mu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Elbow curves: ranked gene loadings per program
# ---------------------------------------------------------------------------

def plot_elbow_curves(loadings_df, output_path, n_top=2000):
    """Plot sorted log-loadings per program to visualize the signal/noise boundary.

    Args:
        loadings_df: DataFrame with programs as rows and genes as columns
                     (gene_spectra_score format from cNMF).
        output_path: Path to save the figure.
        n_top: Number of top genes to show on x-axis.
    """
    programs = loadings_df.index.tolist()
    n_progs = len(programs)
    ncols = 2
    nrows = (n_progs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx, prog in enumerate(programs):
        ax = axes[idx // ncols, idx % ncols]
        vals = loadings_df.loc[prog].sort_values(ascending=False)
        positive_vals = vals[vals > 0].values

        n_trunc = min(n_top, len(positive_vals))
        if n_trunc < 2:
            ax.set_title(f"{prog} (no positive genes)")
            continue

        y_raw = positive_vals[:n_trunc]
        y = np.log(y_raw)
        y = (y - y.min()) / (y.max() - y.min() + 1e-15)
        x = np.arange(n_trunc)

        ax.plot(x, y, "b-", linewidth=0.8, alpha=0.7)
        ax.set_title(f"{prog} ({len(positive_vals)} pos genes)", fontsize=9)
        ax.set_xlabel("Rank", fontsize=8)
        ax.set_ylabel("log(loading) norm", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n_progs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Gene loading elbow curves", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 2. K-selection plot (stability vs error)
# ---------------------------------------------------------------------------

def plot_k_selection(run_dir, run_name, output_path):
    """Plot the k-selection PNG that cNMF generates during combine/consensus.

    If the cNMF-generated k_selection.png exists, copies it. Otherwise skips.
    """
    cnmf_plot = os.path.join(run_dir, run_name, f"{run_name}.k_selection.png")
    if os.path.exists(cnmf_plot):
        import shutil
        shutil.copy2(cnmf_plot, output_path)
        logger.info(f"Copied k-selection plot: {output_path}")
    else:
        logger.warning(f"k-selection plot not found at {cnmf_plot}, skipping")


# ---------------------------------------------------------------------------
# 3. Program usage heatmap (cells x programs, grouped by batch)
# ---------------------------------------------------------------------------

def plot_usage_heatmap(mdata, output_path, categorical_key="batch"):
    """Heatmap of program usage scores, cells grouped by batch.

    Args:
        mdata: MuData with 'cNMF' modality.
        output_path: Path to save.
        categorical_key: obs column to group cells by.
    """
    usage = pd.DataFrame(
        mdata["cNMF"].X,
        index=mdata["cNMF"].obs_names,
        columns=mdata["cNMF"].var_names,
    )

    # Sort cells by batch then by dominant program
    if categorical_key in mdata["cNMF"].obs.columns:
        batch = mdata["cNMF"].obs[categorical_key]
    elif categorical_key in mdata["rna"].obs.columns:
        batch = mdata["rna"].obs[categorical_key]
    else:
        batch = pd.Series("all", index=usage.index)

    usage["_batch"] = batch.values
    usage["_dominant"] = usage.drop(columns="_batch").idxmax(axis=1)
    usage = usage.sort_values(["_batch", "_dominant"])
    batch_sorted = usage["_batch"]
    usage = usage.drop(columns=["_batch", "_dominant"])

    # Normalize per cell for visualization
    row_sums = usage.sum(axis=1)
    usage_norm = usage.div(row_sums.replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(max(6, len(usage.columns) * 0.5), 8))
    im = ax.imshow(
        usage_norm.values,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel("Program", fontsize=9)
    ax.set_ylabel(f"Cells (grouped by {categorical_key})", fontsize=9)
    ax.set_xticks(range(len(usage.columns)))
    ax.set_xticklabels(usage.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks([])

    # Draw batch boundaries
    batch_vals = batch_sorted.values
    for i in range(1, len(batch_vals)):
        if batch_vals[i] != batch_vals[i - 1]:
            ax.axhline(y=i - 0.5, color="white", linewidth=1)

    plt.colorbar(im, ax=ax, label="Normalized usage", shrink=0.6)
    ax.set_title("Program usage (cells x programs)", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 4. Loading distribution violin plot
# ---------------------------------------------------------------------------

def plot_loading_violins(loadings_df, output_path):
    """Violin plot of gene loading distributions per program.

    Args:
        loadings_df: Programs (rows) x genes (columns).
        output_path: Path to save.
    """
    n_progs = len(loadings_df)
    fig, ax = plt.subplots(figsize=(max(6, n_progs * 0.6), 4))

    data = []
    labels = []
    for prog in loadings_df.index:
        vals = loadings_df.loc[prog].values
        positive = vals[vals > 0]
        if len(positive) > 0:
            data.append(np.log10(positive))
            labels.append(prog)

    if not data:
        plt.close(fig)
        return

    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("log10(loading)", fontsize=9)
    ax.set_title("Gene loading distributions per program", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# 5. Consensus clustering plot check
# ---------------------------------------------------------------------------

def collect_clustering_plots(run_dir, run_name, K_list, sel_thresh_list, output_dir):
    """Copy consensus clustering PNGs generated by cNMF into the plots dir."""
    import shutil

    for k in K_list:
        for thresh in sel_thresh_list:
            thresh_str = str(thresh).replace(".", "_")
            src = os.path.join(
                run_dir, run_name, f"{run_name}.clustering.k_{k}.dt_{thresh_str}.png"
            )
            if os.path.exists(src):
                dst = os.path.join(output_dir, f"clustering_k{k}_dt{thresh_str}.png")
                shutil.copy2(src, dst)
                logger.info(f"Copied: {dst}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(run_dir, run_name="Inference", K_list=None, sel_thresh_list=None,
                       categorical_key="batch"):
    """Generate all diagnostic plots for an inference run.

    Args:
        run_dir: Base output directory (contains run_name/ subdirectory).
        run_name: Name of the cNMF run (default "Inference").
        K_list: List of K values to plot.
        sel_thresh_list: List of density thresholds.
        categorical_key: obs column for batch grouping.

    Returns:
        plots_dir: Path to directory containing all generated plots.
    """
    if K_list is None:
        K_list = [5, 10]
    if sel_thresh_list is None:
        sel_thresh_list = [2.0]

    plots_dir = os.path.join(run_dir, run_name, "diagnosis_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # K-selection plot
    plot_k_selection(run_dir, run_name, os.path.join(plots_dir, "k_selection.png"))

    # Collect consensus clustering plots
    collect_clustering_plots(run_dir, run_name, K_list, sel_thresh_list, plots_dir)

    # Per K x threshold plots
    for k in K_list:
        for thresh in sel_thresh_list:
            thresh_str = str(thresh).replace(".", "_")

            # Load gene spectra scores for elbow + violin plots
            spectra_path = os.path.join(
                run_dir, run_name,
                f"{run_name}.gene_spectra_score.k_{k}.dt_{thresh_str}.txt",
            )
            if os.path.exists(spectra_path):
                loadings_df = pd.read_csv(spectra_path, sep="\t", index_col=0)

                plot_elbow_curves(
                    loadings_df,
                    os.path.join(plots_dir, f"elbow_curves_k{k}_dt{thresh_str}.pdf"),
                )
                plot_loading_violins(
                    loadings_df,
                    os.path.join(plots_dir, f"loading_violins_k{k}_dt{thresh_str}.pdf"),
                )
            else:
                logger.warning(f"Spectra scores not found: {spectra_path}")

            # Load h5mu for usage heatmap
            h5mu_path = os.path.join(
                run_dir, run_name, "adata", f"cNMF_{k}_{thresh_str}.h5mu"
            )
            if os.path.exists(h5mu_path):
                mdata = mu.read(h5mu_path)
                plot_usage_heatmap(
                    mdata,
                    os.path.join(plots_dir, f"usage_heatmap_k{k}_dt{thresh_str}.pdf"),
                    categorical_key=categorical_key,
                )
            else:
                logger.warning(f"h5mu not found: {h5mu_path}")

    logger.info(f"All plots saved to: {plots_dir}")
    return plots_dir
