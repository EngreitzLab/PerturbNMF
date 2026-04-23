import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})


def compute_program_expression_by_condition(mdata, Target_Program, groupby='sample'):
    """Compute mean program expression per condition.

    Parameters
    ----------
    mdata : MuData
        MuData object with a 'cNMF' modality.
    Target_Program : str or int
        Program identifier (column in mdata['cNMF'].var_names).
    groupby : str
        Column in obs to group cells by (e.g. 'sample').

    Returns
    -------
    pd.Series
        Mean program expression indexed by condition name.
    """
    X = mdata['cNMF'].X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    prog_col = list(mdata['cNMF'].var_names).index(str(Target_Program))
    scores = X[:, prog_col]

    # Use rna obs for the groupby column since it may only live there
    if groupby in mdata['cNMF'].obs.columns:
        groups = mdata['cNMF'].obs[groupby].values
    else:
        groups = mdata['rna'].obs[groupby].values

    df = pd.DataFrame({'score': scores, 'group': groups})
    return df.groupby('group')['score'].mean()


def plot_program_heatmap_weighted(
    perturb_path_base, mdata, Target_Program,
    tagert_col_name="target_name", plot_col_name="program_name",
    sample=None, groupby='sample',
    log2fc_col='log2FC', p_value=0.05,
    save_path=None, save_name=None,
    figsize=(12, 5), show=False, ax=None,
    vmin=-1, vmax=1,
):
    """Heatmap of regulator effects with a side bar for program expression.

    * **Heatmap** – log2FC of each regulator's effect on the program per
      condition, with asterisks marking significant hits.
    * **Side bar** – horizontal bar chart showing mean program expression
      per condition, so the viewer can judge how relevant each row is.

    Parameters
    ----------
    perturb_path_base : str
        Base path for perturbation result files.  Files are expected at
        ``{perturb_path_base}_{sample}.txt``.
    mdata : MuData
        MuData object containing a ``'cNMF'`` modality with program
        scores in ``.X`` and cell metadata in ``.obs``.
    Target_Program : str or int
        The program to visualise.
    tagert_col_name : str
        Column in the perturbation file used to filter rows for the
        target program (default ``"target_name"``).
    plot_col_name : str
        Column whose unique values become the x-axis labels (regulators)
        (default ``"program_name"``).
    sample : list of str, optional
        Condition / sample names.  Each must match both a perturbation
        file suffix and a value in ``mdata.obs[groupby]``.
    groupby : str
        Column in ``mdata.obs`` that stores the condition labels.
    log2fc_col : str
        Column name for log2 fold-change values.
    p_value : float
        Significance threshold on ``adj_pval``.
    save_path, save_name : str, optional
        If both provided the figure is saved.
    figsize : tuple
        Figure size in inches.
    show : bool
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw on.  When provided, only the heatmap is
        drawn (the expression bar is omitted).
    vmin, vmax : float
        Colour scale limits for log2FC.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if sample is None:
        sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']

    # --- load perturbation data across conditions ---
    perturbed_df_list = [
        pd.read_csv(f"{perturb_path_base}_{samp}.txt", sep="\t").assign(sample=samp)
        for samp in sample
    ]
    perturbed_df = pd.concat(perturbed_df_list, ignore_index=True)
    perturbed_df['program_name'] = perturbed_df['program_name'].astype(str)

    # filter for target program
    df_sorted = perturbed_df.loc[perturbed_df[tagert_col_name] == str(Target_Program)]

    # keep regulators significant in at least one condition
    genes_to_keep = df_sorted[df_sorted['adj_pval'] < p_value][plot_col_name].unique()
    df_filtered = df_sorted[df_sorted[plot_col_name].isin(genes_to_keep)]

    if df_filtered.empty:
        import warnings
        warnings.warn(
            f"No regulators with adj_pval < {p_value} for program {Target_Program}. "
            "Nothing to plot."
        )
        return None

    # pivot tables
    df_pivot = df_filtered.pivot(columns=plot_col_name, index='sample', values=log2fc_col)
    df_pivot_pval = df_filtered.pivot(columns=plot_col_name, index='sample', values='adj_pval')

    # --- compute mean program expression per condition ---
    expr_series = compute_program_expression_by_condition(mdata, Target_Program, groupby=groupby)

    # --- plot ---
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax_heat = fig.add_subplot(111)
        standalone = True
    else:
        ax_heat = ax
        fig = ax.get_figure()
        standalone = False

    # heatmap (no automatic colorbar — we place it manually)
    sns.heatmap(
        df_pivot, cmap='RdBu_r', center=0,
        cbar=False,
        ax=ax_heat, square=True,
        vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor='white',
    )

    # significance asterisks
    for i in range(len(df_pivot)):
        for j in range(len(df_pivot.columns)):
            if df_pivot_pval.iloc[i, j] < p_value:
                ax_heat.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                             color='black', fontsize=12, weight='bold')

    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right')
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
    ax_heat.set_ylabel('')
    ax_heat.set_xlabel('')
    ax_heat.set_title(
        f"Regulator effects on Program {Target_Program}",
        fontsize=13, fontweight='bold', pad=12,
    )

    # Draw the figure so heatmap position is finalized, then add bar + colorbar
    fig.canvas.draw()
    heat_bbox = ax_heat.get_position()

    # Expression bar: same top/bottom as heatmap, thin strip to its right
    bar_width = 0.03
    bar_gap = 0.005
    ax_bar = fig.add_axes([
        heat_bbox.x1 + bar_gap, heat_bbox.y0,
        bar_width, heat_bbox.height,
    ])

    conditions = list(df_pivot.index)
    expr_vals = [expr_series.get(c, 0.0) for c in conditions]
    max_e = max(expr_vals) if max(expr_vals) > 0 else 1
    norm_vals = [v / max_e for v in expr_vals]
    y_positions = np.arange(len(conditions)) + 0.5

    ax_bar.barh(y_positions, norm_vals, height=0.8,
                 color='#2CA02C', edgecolor='none', alpha=0.8)
    ax_bar.set_ylim(ax_heat.get_ylim())
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(['0', '1'], fontsize=7)
    ax_bar.set_xlabel('Expr.', fontsize=9)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.tick_params(axis='y', length=0)
    ax_bar.tick_params(axis='x', length=2, pad=2)

    # Colorbar: same top/bottom, to the right of expression bar
    import matplotlib.colors as mcolors
    cbar_gap = 0.015
    cbar_width = 0.012
    ax_cbar = fig.add_axes([
        heat_bbox.x1 + bar_gap + bar_width + cbar_gap, heat_bbox.y0,
        cbar_width, heat_bbox.height,
    ])
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar)
    cb.set_label('log2FC', fontsize=9)
    cb.ax.tick_params(labelsize=7)

    if standalone:
        if save_path and save_name:
            fig.savefig(f'{save_path}/{save_name}.pdf', format='pdf',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax_heat


def plot_program_heatmap_expression_scaled(
    perturb_path_base, mdata, Target_Program,
    tagert_col_name="target_name", plot_col_name="program_name",
    sample=None, groupby='sample',
    log2fc_col='log2FC', p_value=0.05,
    save_path=None, save_name=None,
    figsize=(12, 5), show=False, ax=None,
):
    """Heatmap of log2FC scaled by relative program expression per condition.

    For each condition, the log2FC is multiplied by
    ``mean_expr(condition) / max(mean_expr)``, so effects in conditions
    where the program is barely expressed are dampened towards zero.

    Parameters
    ----------
    (same as ``plot_program_heatmap_weighted``, without dot-size params)

    Returns
    -------
    matplotlib.axes.Axes
    """
    if sample is None:
        sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']

    # --- load perturbation data ---
    perturbed_df_list = [
        pd.read_csv(f"{perturb_path_base}_{samp}.txt", sep="\t").assign(sample=samp)
        for samp in sample
    ]
    perturbed_df = pd.concat(perturbed_df_list, ignore_index=True)
    perturbed_df['program_name'] = perturbed_df['program_name'].astype(str)

    df_sorted = perturbed_df.loc[perturbed_df[tagert_col_name] == str(Target_Program)]
    genes_to_keep = df_sorted[df_sorted['adj_pval'] < p_value][plot_col_name].unique()
    df_filtered = df_sorted[df_sorted[plot_col_name].isin(genes_to_keep)]

    df_pivot = df_filtered.pivot(columns=plot_col_name, index='sample', values=log2fc_col)
    df_pivot_pval = df_filtered.pivot(columns=plot_col_name, index='sample', values='adj_pval')

    # --- compute expression weights ---
    expr_series = compute_program_expression_by_condition(mdata, Target_Program, groupby=groupby)
    max_expr = expr_series.max()
    if max_expr > 0:
        weights = expr_series / max_expr
    else:
        weights = expr_series * 0 + 1  # fallback: no scaling

    # scale each row of the pivot by the condition weight
    scaled_pivot = df_pivot.copy()
    for cond in scaled_pivot.index:
        w = weights.get(cond, 0.0)
        scaled_pivot.loc[cond] = scaled_pivot.loc[cond] * w

    # determine symmetric colour limits from the scaled data
    abs_max = max(abs(scaled_pivot.min().min()), abs(scaled_pivot.max().max()), 0.01)

    # --- plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    sns.heatmap(
        scaled_pivot, cmap='RdBu_r', center=0,
        cbar_kws={'label': 'Expr-scaled log2FC', 'shrink': 0.6, 'aspect': 20},
        ax=ax, square=True,
        vmin=-abs_max, vmax=abs_max,
        linewidths=0.5, linecolor='white',
    )

    # significance markers
    for i in range(len(scaled_pivot)):
        for j in range(len(scaled_pivot.columns)):
            if df_pivot_pval.iloc[i, j] < p_value:
                ax.text(j + 0.5, i + 0.5, '*', ha='center', va='center',
                        color='black', fontsize=12, weight='bold')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(
        f"Expression-scaled regulator effects on Program {Target_Program}",
        fontsize=13, fontweight='bold', pad=12,
    )

    # annotate condition weights on the right side
    for i, cond in enumerate(scaled_pivot.index):
        w = weights.get(cond, 0.0)
        ax.text(len(scaled_pivot.columns) + 0.3, i + 0.5,
                f'w={w:.2f}', ha='left', va='center', fontsize=8, color='#666666')

    if standalone:
        fig.subplots_adjust(bottom=0.2)
        if save_path and save_name:
            fig.savefig(f'{save_path}/{save_name}.pdf', format='pdf',
                        bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax
