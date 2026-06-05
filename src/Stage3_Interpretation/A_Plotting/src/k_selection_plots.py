import os
import math
import mudata
import muon as mu
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.regression.mixed_linear_model import MixedLM
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import scipy.sparse as sp
import anndata as ad


# Publication-quality color palette — one distinct color per plot
_COLORS = {
    'primary': '#2c3e50',        # dark slate for line strokes
    'stability': '#3498db',      # blue
    'error': '#e74c3c',          # red
    'go_terms': '#2ecc71',       # green
    'genesets': '#d35400',       # burnt orange
    'traits': '#9b59b6',         # purple
    'perturbation': '#1abc9c',   # teal
    'explained_var': '#f1c40f',  # yellow
}

# Qualitative palette for multi-line plots
_PALETTE = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22', '#34495e']


def _style_ax(ax, xlabel=None, ylabel=None, title=None, hide_top_right=True):
    """Apply publication-quality styling to an axis."""
    if hide_top_right:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', which='major', labelsize=9, width=0.8, length=4,
                   direction='out', color='#333333')
    ax.tick_params(axis='both', which='minor', width=0.5, length=2,
                   direction='out', color='#333333')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='medium', labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='medium', labelpad=6)
    if title:
        ax.set_title(title, fontsize=12, fontweight='semibold', pad=8)


# ---------------------------------------------------------------------------
# Helpers for load_stablity_error_data
# ---------------------------------------------------------------------------

def _load_stats_file(path):
    """Load a stability/error DataFrame from TSV or NPZ."""
    if path.endswith('.npz'):
        npz = np.load(path, allow_pickle=True)
        return pd.DataFrame(data=npz['data'], index=npz['index'], columns=npz['columns'])
    else:
        return pd.read_csv(path, sep='\t')


def _filter_components(stats, components):
    """Keep only the requested K values, if present."""
    stats = stats.copy()
    stats['k'] = stats['k'].astype(int)
    available = set(stats['k'])
    requested = set(components)
    missing = requested - available
    if missing:
        print(f"  Warning: requested K values not in stats file and will be skipped: {sorted(missing)}")
    return stats.loc[stats['k'].isin(requested)].sort_values('k').reset_index(drop=True)


def _cache_has_all_components(stats, components):
    """Return True only if every requested K is present in the cache.

    Tier 1 uses this to decide whether to use the cache or fall through to
    Tier 2 / Tier 3 recomputation. Stability and prediction error are stored
    per-K, so a missing K means that row must be recomputed.
    """
    if stats is None or len(stats) == 0:
        return False
    available_k = set(stats['k'].astype(int))
    missing_k = set(components) - available_k
    if missing_k:
        print(f"  Cache missing K values: {sorted(missing_k)} — falling through to next tier")
        return False
    return True


def _print_stats_summary(stats):
    """Print min/max summary for stability and error."""
    print("min stablity is", stats['silhouette'].min(), flush=True)
    print("max stablity is", stats['silhouette'].max(), flush=True)
    print("min error is", stats['prediction_error'].min(), flush=True)
    print("max error is", stats['prediction_error'].max(), flush=True)


def _compute_stability_error_manual(output_directory, run_name, components,
                                    density_threshold=2.0, local_neighborhood_size=0.30):
    """Tier 5: compute stability/error from merged_spectra + usages without cnmf.consensus()."""
    run_dir = os.path.join(output_directory, run_name)
    cnmf_tmp = os.path.join(run_dir, 'cnmf_tmp')

    # Load normalized counts — fall back to any per-K run dir if shared path missing
    norm_path = os.path.join(run_dir, run_name + '.norm_counts.h5ad')
    if not os.path.exists(norm_path):
        norm_path = os.path.join(cnmf_tmp, run_name + '.norm_counts.h5ad')
    if not os.path.exists(norm_path):
        for k in components:
            fallback = os.path.join(f'{output_directory}_K{k}', run_name, 'cnmf_tmp', run_name + '.norm_counts.h5ad')
            if os.path.exists(fallback):
                norm_path = fallback
                print(f"  Using per-K norm_counts from K={k}: {fallback}")
                break
    norm_counts = sc.read(norm_path)

    rows = []
    for k in components:
        # Load merged spectra — try shared run dir first, then per-K fallback
        spectra_path = os.path.join(cnmf_tmp, f'{run_name}.spectra.k_{k}.merged.df.npz')
        if not os.path.exists(spectra_path):
            per_k_spectra = os.path.join(f'{output_directory}_K{k}', run_name, 'cnmf_tmp', f'{run_name}.spectra.k_{k}.merged.df.npz')
            if os.path.exists(per_k_spectra):
                spectra_path = per_k_spectra
                print(f"  Using per-K spectra fallback for k={k}: {per_k_spectra}")
            else:
                print(f"  Warning: merged spectra not found for k={k}, skipping: {spectra_path}")
                continue
        npz = np.load(spectra_path, allow_pickle=True)
        merged_spectra = pd.DataFrame(data=npz['data'], index=npz['index'], columns=npz['columns'])

        # Stability via KMeans + silhouette
        l2_spectra = (merged_spectra.T / np.sqrt((merged_spectra**2).sum(axis=1))).T
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=1)
        kmeans.fit(l2_spectra)
        labels = pd.Series(kmeans.labels_ + 1, index=l2_spectra.index)
        sil = silhouette_score(l2_spectra.values, labels, metric='euclidean')

        # Median spectra (normalized to probability distributions)
        median_spectra = l2_spectra.groupby(labels).median()
        median_spectra = (median_spectra.T / median_spectra.sum(axis=1)).T

        # Load refit usages — try shared run dir first, then per-K fallback
        dt_str = str(density_threshold).replace('.', '_')
        usage_path = os.path.join(
            run_dir, f'{run_name}.usages.k_{k}.dt_{dt_str}.consensus.txt')
        if not os.path.exists(usage_path):
            per_k_usage = os.path.join(f'{output_directory}_K{k}', run_name, f'{run_name}.usages.k_{k}.dt_{dt_str}.consensus.txt')
            if os.path.exists(per_k_usage):
                usage_path = per_k_usage
            else:
                print(f"  Warning: usage file not found for k={k}, skipping: {usage_path}")
                continue
        rf_usages = pd.read_csv(usage_path, sep='\t', index_col=0)

        # Align dimensions
        median_spectra.columns = range(median_spectra.shape[1])
        median_spectra = median_spectra.reset_index(drop=True)
        rf_usages.columns = range(rf_usages.shape[1])
        rf_usages = rf_usages.reset_index(drop=True)

        # Prediction error (Frobenius norm)
        pred = rf_usages.values.dot(median_spectra.values)
        X = norm_counts.X
        if sp.issparse(X):
            X = X.todense()
        prediction_error = float(((np.asarray(X) - pred) ** 2).sum())

        rows.append({
            'k': k,
            'local_density_threshold': density_threshold,
            'silhouette': sil,
            'prediction_error': prediction_error,
        })

    if not rows:
        raise FileNotFoundError(
            f"Could not compute stability/error for any K in {components}. "
            f"Check that merged spectra and usage files exist under {run_dir}.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main loader — 3-tier fallback
# ---------------------------------------------------------------------------

def load_stablity_error_data(output_directory, run_name, components=[30, 50, 60, 80, 100, 200, 250, 300],
                             stability_file=None):
    """Load stability and prediction-error stats with a 3-tier fallback.

    Tier 1: Load from file — user-supplied path, or auto-detect
            ``{run}.k_selection_stats.df.npz`` / TSV cache
    Tier 2: Compute via ``torch_cnmf.cNMF.consensus()``
    Tier 3: Manual computation from merged_spectra + usages
    """
    run_dir = os.path.join(output_directory, run_name)
    npz_path = os.path.join(run_dir, f'{run_name}.k_selection_stats.df.npz')

    stats = None

    # --- Tier 1: load from file ---
    # Validate completeness against requested K before accepting the cache;
    # if any K is missing, drop stats and fall through to Tier 2 / Tier 3.
    if stability_file is not None:
        if os.path.exists(stability_file):
            print(f"Loading stability/error data from user-supplied file: {stability_file}")
            cached = _load_stats_file(stability_file)
            if _cache_has_all_components(cached, components):
                stats = _filter_components(cached, components)
        else:
            print(f"Warning: stability_file not found: {stability_file}")

    if stats is None and os.path.exists(npz_path):
        print(f"Loading pre-computed stats from {npz_path}")
        cached = _load_stats_file(npz_path)
        if _cache_has_all_components(cached, components):
            stats = _filter_components(cached, components)

    # --- Tier 2: compute via torch_cnmf ---
    if stats is None:
        try:
            from torch_cnmf import cNMF
            print("Running torch_cnmf.consensus() for each K ...")
            cnmf_obj = cNMF(output_dir=output_directory, name=run_name)
            rows = []
            norm_counts = sc.read(cnmf_obj.paths['normalized_counts'])
            for k in components:
                rows.append(
                    cnmf_obj.consensus(
                        k, skip_density_and_return_after_stats=True,
                        show_clustering=False, close_clustergram_fig=True,
                        norm_counts=norm_counts, density_threshold=2.0,
                        local_neighborhood_size=0.3,
                    ).stats
                )
            stats = pd.DataFrame(rows)
        except Exception as e:
            print(f"torch_cnmf.consensus() failed: {e}")

    # --- Tier 3: manual computation ---
    if stats is None:
        print("Computing stability/error manually from merged_spectra + usages ...")
        stats = _compute_stability_error_manual(output_directory, run_name, components)

    _print_stats_summary(stats)
    return stats


# plot NMF stability and error — 2 independent square figures
def plot_stablity_error(stats, folder_name=None, file_name=None, selected_k=None):
    # --- Stability ---
    fig1, ax1 = plt.subplots(figsize=(4, 3.5))
    ax1.plot(stats['k'], stats['silhouette'], color=_COLORS['primary'], linewidth=1.5,
             marker='o', markersize=4, markerfacecolor=_COLORS['stability'], markeredgecolor=_COLORS['primary'],
             markeredgewidth=0.8, zorder=3)
    ax1.fill_between(stats['k'], stats['silhouette'], alpha=0.08, color=_COLORS['stability'])
    if selected_k is not None:
        ax1.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
    _style_ax(ax1, xlabel='Number of components (k)', ylabel='Stability (silhouette)',
              title='Program stability')

    if folder_name and file_name:
        fig1.savefig(f"{folder_name}/{file_name}_stability.svg", bbox_inches="tight")
        fig1.savefig(f"{folder_name}/{file_name}_stability.png", dpi=300, bbox_inches="tight")

    plt.show()

    # --- Error ---
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    ax2.plot(stats['k'], stats['prediction_error'], color=_COLORS['primary'], linewidth=1.5,
             marker='s', markersize=4, markerfacecolor=_COLORS['error'], markeredgecolor=_COLORS['primary'],
             markeredgewidth=0.8, zorder=3)
    ax2.fill_between(stats['k'], stats['prediction_error'], alpha=0.08, color=_COLORS['error'])
    if selected_k is not None:
        ax2.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
    _style_ax(ax2, xlabel='Number of components (k)', ylabel='Prediction error',
              title='Reconstruction error')

    if folder_name and file_name:
        fig2.savefig(f"{folder_name}/{file_name}_error.svg", bbox_inches="tight")
        fig2.savefig(f"{folder_name}/{file_name}_error.png", dpi=300, bbox_inches="tight")

    plt.show()





# Load data for differeent enrichment test
def load_enrichment_data(folder, components = [30, 50, 60, 80, 100, 200, 250, 300], sel_thresh = 2.0, pval = 0.05,
                         go_file=None, geneset_file=None, trait_file=None,
                         term_col='Term', adjpval_col='Adjusted P-value'):

    # loading function
    def load(k, term, folder, term_col, adjpval_col):
        # read evaluation results
        if os.path.getsize(folder) == 0:
            df = pd.DataFrame(columns=[term_col, adjpval_col])
            df['num_programs'] = k
            df['test_term'] = term
            return df
        df = pd.read_csv(folder, sep='\t')
        df = df.loc[df[adjpval_col]<=pval]
        df['num_programs'] = k
        df['test_term'] = term
        # normalize the term column name for downstream counting
        if term_col != 'Term':
            df = df.rename(columns={term_col: 'Term'})

        return df

    # Default file name patterns (original IGVF convention)
    if go_file is None:
        go_file = '{k}_GO_term_enrichment.txt'
    if geneset_file is None:
        geneset_file = '{k}_geneset_enrichment.txt'
    if trait_file is None:
        trait_file = '{k}_trait_enrichment.txt'

    # collect all the results for each k
    term_df = []

    for k in components:
        k_folder = '{}/{}_{}'.format(folder, k, str(sel_thresh).replace('.','_'))

        term_df.append(load(k, 'go_terms', '{}/{}'.format(k_folder, go_file.format(k=k)), term_col, adjpval_col))
        term_df.append(load(k, 'genesets', '{}/{}'.format(k_folder, geneset_file.format(k=k)), term_col, adjpval_col))
        term_df.append(load(k, 'traits', '{}/{}'.format(k_folder, trait_file.format(k=k)), term_col, adjpval_col))

    term_df = pd.concat(term_df, ignore_index=True)

    # Count unique terms per k
    count_df = pd.DataFrame(index=components, columns=term_df['test_term'].unique())

    for k in components:
        for col in count_df.columns:
            count_df.loc[k, col] = term_df.loc[(term_df['num_programs']==k) & (term_df['test_term']==col), 'Term'].unique().shape[0]

    # ensure expected columns exist even when no significant results (e.g. no GWAS hits for mouse data)
    for col in ['go_terms', 'genesets', 'traits']:
        if col not in count_df.columns:
            count_df[col] = 0

    #print out some stats
    print(f"min go_terms for {sel_thresh} is", count_df['go_terms'].min(), flush=True)
    print(f"max go_terms for {sel_thresh} is", count_df['go_terms'].max(), flush=True)

    print(f"min genesets for {sel_thresh} is", count_df['genesets'].min(), flush=True)
    print(f"max genesets for {sel_thresh} is", count_df['genesets'].max(), flush=True)

    print(f"min traits for {sel_thresh}  is", count_df['traits'].min(), flush=True)
    print(f"max traits for {sel_thresh}  is", count_df['traits'].max(), flush=True)

    #print("min motif is", count_df['motifs'].min())
    #print("max motif is", count_df['motifs'].max())


    return count_df


# plot loaded df — 3 independent square figures
def plot_enrichment(count_df, folder_name=None, file_name=None, selected_k=None):
    enrichment_config = [
        ('go_terms',  'Unique GO terms',   'GO term enrichment',   _COLORS['go_terms'],  'o'),
        ('genesets',  'Unique gene sets',   'Gene set enrichment',  _COLORS['genesets'],  's'),
        ('traits',    'Unique traits', 'Trait enrichment', _COLORS['traits'],   'D'),
    ]

    for col, ylabel, title, color, marker in enrichment_config:
        fig, ax = plt.subplots(figsize=(4, 3.5))

        vals = count_df[col].astype(float)
        ax.plot(count_df.index, vals, color=_COLORS['primary'], linewidth=1.5,
                marker=marker, markersize=4, markerfacecolor=color,
                markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
        ax.fill_between(count_df.index, vals, alpha=0.08, color=color)
        if selected_k is not None:
            ax.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
        _style_ax(ax, xlabel='Number of components (k)', ylabel=ylabel, title=title)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

        if folder_name and file_name:
            fig.savefig(f"{folder_name}/{file_name}_{col}.svg", bbox_inches="tight")
            fig.savefig(f"{folder_name}/{file_name}_{col}.png", dpi=300, bbox_inches="tight")

        plt.show()


# load perturbation data
def load_perturbation_data(folder, pval = 0.000335, components = [30, 50, 60, 80, 100, 200, 250, 300], sel_thresh = 2.0,
 conditions = ['D0', 'sample_D1', 'sample_D2', 'sample_D3'], perturbation_file=None,
 perturb_adjpval_col='adj_pval', perturb_target_col='target_name', perturb_log2fc_col='log2FC'):

    # Default file name pattern (original IGVF convention): {k}_perturbation_association_results_{sample}.txt
    # Use {k} and {sample} as placeholders in custom patterns.
    if perturbation_file is None:
        perturbation_file = '{k}_perturbation_association_results_{sample}.txt'

    # Compute no. of unique regulators
    test_stats_df = []

    for k in components:
        # Run perturbation assocation
        for cond in conditions:
            file_path = '{}/{}_{}/{}'.format(
                folder, k, str(sel_thresh).replace('.','_'),
                perturbation_file.format(k=k, sample=cond))
            test_stats_df_ = pd.read_csv(file_path, sep='\t')
            test_stats_df_['condition'] = cond
            test_stats_df_['K'] = k
            test_stats_df.append(test_stats_df_)

    test_stats_df = pd.concat(test_stats_df, ignore_index=True)

    # Normalize column names so downstream functions work with standard names
    rename_map = {}
    if perturb_adjpval_col != 'adj_pval':
        rename_map[perturb_adjpval_col] = 'adj_pval'
    if perturb_target_col != 'target_name':
        rename_map[perturb_target_col] = 'target_name'
    if perturb_log2fc_col != 'log2FC':
        rename_map[perturb_log2fc_col] = 'log2FC'
    if rename_map:
        test_stats_df = test_stats_df.rename(columns=rename_map)

    # pring some stats
    plotting_df = test_stats_df.loc[test_stats_df.adj_pval < pval, ['K','target_name']].drop_duplicates().groupby(['K']).count().reset_index()

    print("min regulators is", plotting_df["target_name"].min(), flush=True)
    print("max regulators is", plotting_df["target_name"].max(), flush=True)

    return test_stats_df


# plot perturbation data — 2 independent square figures
def plot_perturbation(test_stats_df, pval=0.000335, folder_name=None, file_name=None, selected_k=None):
    pval_str = f'{pval:.1e}' if pval < 0.01 else str(pval)

    # --- Per-sample plot ---
    fig1, ax1 = plt.subplots(figsize=(4, 3.5))

    plotting_df_sample = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'condition', 'target_name']]
        .drop_duplicates()
        .groupby(['K', 'condition']).count().reset_index())
    sns.lineplot(x='K', y='target_name', hue='condition', data=plotting_df_sample,
                 palette=_PALETTE[:plotting_df_sample['condition'].nunique()], linewidth=1.5, marker='o', markersize=4,
                 ax=ax1, legend='brief')
    if selected_k is not None:
        ax1.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
    _style_ax(ax1, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title=f'Unique regulators per condition') # (adj. p-value \u2264 {pval_str})')
    ax1.legend(title='Condition', fontsize=8, title_fontsize=9,
               frameon=True, fancybox=False, edgecolor='#cccccc',
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    if folder_name and file_name:
        fig1.savefig(f"{folder_name}/{file_name}_per_condition.svg", bbox_inches="tight")
        fig1.savefig(f"{folder_name}/{file_name}_per_condition.png", dpi=300, bbox_inches="tight")

    plt.show()

    # --- Aggregated plot ---
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))

    plotting_df = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'target_name']]
        .drop_duplicates()
        .groupby(['K']).count().reset_index())
    ax2.plot(plotting_df['K'], plotting_df['target_name'],
             color=_COLORS['primary'], linewidth=1.5,
             marker='o', markersize=4, markerfacecolor=_COLORS['perturbation'],
             markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax2.fill_between(plotting_df['K'], plotting_df['target_name'],
                     alpha=0.08, color=_COLORS['perturbation'])
    if selected_k is not None:
        ax2.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
    _style_ax(ax2, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title=f'Unique regulators for all conditions') #(adj. p-value \u2264 {pval_str})')
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    if folder_name and file_name:
        fig2.savefig(f"{folder_name}/{file_name}_all_conditions.svg", bbox_inches="tight")
        fig2.savefig(f"{folder_name}/{file_name}_all_conditions.png", dpi=300, bbox_inches="tight")

    plt.show()

    return plotting_df



# load total explained variance
def load_explained_variance_data(folder, components = [30, 50, 60, 80, 100, 200, 250, 300], sel_thresh = 2.0,
                                  variance_file=None, variance_col='Total'):

    # Default file name pattern
    if variance_file is None:
        variance_file = '{k}_Explained_Variance_Summary.txt'

    stats = {}
    for k in components:

        input_path = f"{folder}/{k}_{str(sel_thresh).replace('.','_')}/{variance_file.format(k=k)}"
        df = pd.read_csv(input_path, sep = '\t', index_col = 0)

        if variance_col == 'Total':
            stats[k] = df['Total'].values[0]
        else:
            # For per-program variance (e.g. Morphic: variance_explained_ratio), sum all programs
            stats[k] = df[variance_col].sum()

    
    print("min Explained_variance is", min(stats.values()), flush=True)
    print("max Explained_variance is", max(stats.values()), flush=True)

    return stats


# plot NMF explained variance
def plot_explained_variance(stats, folder_name=None, file_name=None, selected_k=None):
    ks = list(stats.keys())
    vals = list(stats.values())

    fig, ax = plt.subplots(figsize=(4, 3.5))

    ax.plot(ks, vals, color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['explained_var'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(ks, vals, alpha=0.08, color=_COLORS['explained_var'])
    if selected_k is not None:
        ax.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Total explained variance',
              title='Explained variance')

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.svg", bbox_inches="tight")
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight")

    plt.show()



# Combined panel figure: 3 rows x 3 columns
# Row 1: Stability, Error, Explained variance
# Row 2: GO terms, Gene sets, Traits
# Row 3: Regulators (all conditions), Regulators (per condition)
def plot_k_selection_panel(stability_stats, count_df, test_stats_df, explained_var_stats,
                           pval=0.05, folder_name=None, file_name=None, selected_k=None):

    # Keep text as editable text in SVG (for Adobe Illustrator)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    pval_str = f'{pval:.1e}' if pval < 0.01 else str(pval)

    def _add_vline(ax):
        if selected_k is not None:
            ax.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)

    # --- Row 1, Col 0: Stability ---
    ax = axes[0, 0]
    ax.plot(stability_stats['k'], stability_stats['silhouette'], color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['stability'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(stability_stats['k'], stability_stats['silhouette'], alpha=0.08, color=_COLORS['stability'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Stability (silhouette)', title='Program stability')

    # --- Row 1, Col 1: Error ---
    ax = axes[0, 1]
    ax.plot(stability_stats['k'], stability_stats['prediction_error'], color=_COLORS['primary'], linewidth=1.5,
            marker='s', markersize=4, markerfacecolor=_COLORS['error'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(stability_stats['k'], stability_stats['prediction_error'], alpha=0.08, color=_COLORS['error'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Prediction error', title='Reconstruction error')

    # --- Row 1, Col 2: Explained variance ---
    ax = axes[0, 2]
    ev_ks = list(explained_var_stats.keys())
    ev_vals = list(explained_var_stats.values())
    ax.plot(ev_ks, ev_vals, color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['explained_var'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(ev_ks, ev_vals, alpha=0.08, color=_COLORS['explained_var'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Total explained variance', title='Explained variance')

    # --- Row 2: Enrichment (GO, genesets, traits) ---
    enrichment_config = [
        (0, 'go_terms',  'Unique GO terms',   'GO term enrichment',   _COLORS['go_terms'],  'o'),
        (1, 'genesets',  'Unique gene sets',   'Gene set enrichment',  _COLORS['genesets'],  's'),
        (2, 'traits',    'Unique traits',      'Trait enrichment',     _COLORS['traits'],    'D'),
    ]
    for col_idx, col, ylabel, title, color, marker in enrichment_config:
        ax = axes[1, col_idx]
        vals = count_df[col].astype(float)
        ax.plot(count_df.index, vals, color=_COLORS['primary'], linewidth=1.5,
                marker=marker, markersize=4, markerfacecolor=color,
                markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
        ax.fill_between(count_df.index, vals, alpha=0.08, color=color)
        _add_vline(ax)
        _style_ax(ax, xlabel='Number of components (k)', ylabel=ylabel, title=title)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 3, Col 0: Regulators (all conditions) ---
    ax = axes[2, 0]
    plotting_df = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'target_name']]
        .drop_duplicates()
        .groupby(['K']).count().reset_index())
    ax.plot(plotting_df['K'], plotting_df['target_name'], color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['perturbation'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(plotting_df['K'], plotting_df['target_name'], alpha=0.08, color=_COLORS['perturbation'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title='Unique regulators for all conditions')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 3, Col 1: Regulators (per sample) ---
    ax = axes[2, 1]
    plotting_df_sample = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'condition', 'target_name']]
        .drop_duplicates()
        .groupby(['K', 'condition']).count().reset_index())
    sns.lineplot(x='K', y='target_name', hue='condition', data=plotting_df_sample,
                 palette=_PALETTE[:plotting_df_sample['condition'].nunique()], linewidth=1.5, marker='o', markersize=4,
                 ax=ax, legend='brief')
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title='Unique regulators per condition')
    ax.legend(title='Condition', fontsize=7, title_fontsize=8,
              frameon=True, fancybox=False, edgecolor='#cccccc',
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 3, Col 2: hide empty panel ---
    axes[2, 2].set_visible(False)

    # Add panel labels (A, B, C, ...)
    panel_labels = 'ABCDEFGH'
    visible_axes = [axes[r, c] for r in range(3) for c in range(3) if axes[r, c].get_visible()]
    for label, ax in zip(panel_labels, visible_axes):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.svg", bbox_inches="tight")
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight")

    plt.show()


# Combined panel figure: 3 rows x 3 columns (no trait enrichment)
# Row 1: Stability, Error, Explained variance
# Row 2: GO terms, Gene sets
# Row 3: Regulators (all conditions), Regulators (per condition)
def plot_k_selection_panel_no_traits(stability_stats, count_df, test_stats_df, explained_var_stats,
                                     pval=0.05, folder_name=None, file_name=None, selected_k=None):

    # Keep text as editable text in SVG (for Adobe Illustrator)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42

    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    pval_str = f'{pval:.1e}' if pval < 0.01 else str(pval)

    def _add_vline(ax):
        if selected_k is not None:
            ax.axvline(x=selected_k, color='red', linestyle='--', linewidth=1, zorder=2)

    # --- Row 1, Col 0: Stability ---
    ax = axes[0, 0]
    ax.plot(stability_stats['k'], stability_stats['silhouette'], color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['stability'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(stability_stats['k'], stability_stats['silhouette'], alpha=0.08, color=_COLORS['stability'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Stability (silhouette)', title='Program stability')

    # --- Row 1, Col 1: Error ---
    ax = axes[0, 1]
    ax.plot(stability_stats['k'], stability_stats['prediction_error'], color=_COLORS['primary'], linewidth=1.5,
            marker='s', markersize=4, markerfacecolor=_COLORS['error'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(stability_stats['k'], stability_stats['prediction_error'], alpha=0.08, color=_COLORS['error'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Prediction error', title='Reconstruction error')

    # --- Row 1, Col 2: Explained variance ---
    ax = axes[0, 2]
    ev_ks = list(explained_var_stats.keys())
    ev_vals = list(explained_var_stats.values())
    ax.plot(ev_ks, ev_vals, color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['explained_var'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(ev_ks, ev_vals, alpha=0.08, color=_COLORS['explained_var'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='Total explained variance', title='Explained variance')

    # --- Row 2: Enrichment (GO, genesets) — no traits ---
    enrichment_config = [
        (0, 'go_terms',  'Unique GO terms',   'GO term enrichment',   _COLORS['go_terms'],  'o'),
        (1, 'genesets',  'Unique gene sets',   'Gene set enrichment',  _COLORS['genesets'],  's'),
    ]
    for col_idx, col, ylabel, title, color, marker in enrichment_config:
        ax = axes[1, col_idx]
        vals = count_df[col].astype(float)
        ax.plot(count_df.index, vals, color=_COLORS['primary'], linewidth=1.5,
                marker=marker, markersize=4, markerfacecolor=color,
                markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
        ax.fill_between(count_df.index, vals, alpha=0.08, color=color)
        _add_vline(ax)
        _style_ax(ax, xlabel='Number of components (k)', ylabel=ylabel, title=title)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 2, Col 2: hide empty panel ---
    axes[1, 2].set_visible(False)

    # --- Row 3, Col 0: Regulators (all conditions) ---
    ax = axes[2, 0]
    plotting_df = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'target_name']]
        .drop_duplicates()
        .groupby(['K']).count().reset_index())
    ax.plot(plotting_df['K'], plotting_df['target_name'], color=_COLORS['primary'], linewidth=1.5,
            marker='o', markersize=4, markerfacecolor=_COLORS['perturbation'],
            markeredgecolor=_COLORS['primary'], markeredgewidth=0.8, zorder=3)
    ax.fill_between(plotting_df['K'], plotting_df['target_name'], alpha=0.08, color=_COLORS['perturbation'])
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title='Unique regulators for all conditions')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 3, Col 1: Regulators (per sample) ---
    ax = axes[2, 1]
    plotting_df_sample = (test_stats_df
        .loc[test_stats_df.adj_pval <= pval, ['K', 'condition', 'target_name']]
        .drop_duplicates()
        .groupby(['K', 'condition']).count().reset_index())
    sns.lineplot(x='K', y='target_name', hue='condition', data=plotting_df_sample,
                 palette=_PALETTE[:plotting_df_sample['condition'].nunique()], linewidth=1.5, marker='o', markersize=4,
                 ax=ax, legend='brief')
    _add_vline(ax)
    _style_ax(ax, xlabel='Number of components (k)', ylabel='No. unique regulators',
              title='Unique regulators per condition')
    ax.legend(title='Condition', fontsize=7, title_fontsize=8,
              frameon=True, fancybox=False, edgecolor='#cccccc',
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))

    # --- Row 3, Col 2: hide empty panel ---
    axes[2, 2].set_visible(False)

    # Add panel labels (A, B, C, ...)
    panel_labels = 'ABCDEFG'
    visible_axes = [axes[r, c] for r in range(3) for c in range(3) if axes[r, c].get_visible()]
    for label, ax in zip(panel_labels, visible_axes):
        ax.text(-0.15, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.svg", bbox_inches="tight")
        fig.savefig(f"{folder_name}/{file_name}.png", dpi=300, bbox_inches="tight")

    plt.show()


# --- per-(K, sel_thresh) program dotplot (moved here from k_quality_plots.py) ---
def programs_dotplots(k, output_dir, run_name, sel_thresh = 2.0, groupby='sample', save_name=None, save_path=None, 
figsize=(4, 30), show=False, ax=None):

    standalone_mode = ax is None


    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently.
        Tries standard PerturbNMF layout, then flat shared-run, then per-K run layout."""
        sel_str = str(sel_thresh).replace(".", "_")
        standard = f'{output_dir}/{run_name}/Inference/adata/cNMF_{k}_{sel_str}.h5mu'
        if os.path.isfile(standard):
            return standard
        flat = f'{output_dir}/{run_name}/adata/cNMF_{k}_{sel_str}.h5mu'
        if os.path.isfile(flat):
            return flat
        per_k = f'{output_dir}/{run_name}_K{k}/Inference/adata/cNMF_{k}_{sel_str}.h5mu'
        if os.path.isfile(per_k):
            return per_k
        return standard


    # read in adata
    path = get_gene_path(output_dir, run_name, k, sel_thresh)
    if not os.path.isfile(path):
        print(f"[programs_dotplots] WARNING: h5mu not found for K={k}, sel_thresh={sel_thresh}; "
              f"tried standard / flat / per-K layouts. Skipping this combination.")
        return None
    mdata = mu.read_h5mu(path)
    adata_new = mdata['cNMF'].copy()
    
    if save_name is None:
        save_name = "Program Loadings by Days"
    
    # Create the dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
                
        grogram_list = adata_new.var_names.tolist()
        dp = sc.pl.dotplot(adata_new, grogram_list, groupby=groupby,
                          figsize=figsize, swap_axes=True, dendrogram=False,
                          show=False, return_fig=True)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        fig = ax.get_figure()
        grogram_list = adata_new.var_names.tolist()
        dp = sc.pl.dotplot(adata_new, grogram_list, groupby=groupby,
                          swap_axes=True, dendrogram=False, show=False,
                          return_fig=True, ax=ax)
        dp.make_figure()
    
    ax.set_title(save_name, fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Program', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel(groupby, fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    label = list(mdata['rna'].obs[groupby].cat.categories)  # Use categories instead
    
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)

    
    # save fig (only in standalone mode)
    if save_name and save_path and standalone_mode:
        fig.savefig(f"{save_path}/{save_name}.png", format='png', bbox_inches='tight', dpi=300)  # Changed to png
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# Plot clustermap given paths to program by gene matrix
