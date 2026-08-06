
import muon as mu 
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import scanpy as sc
import anndata as ad
from pathlib import Path
from PIL import Image
import mygene
import os
from PyPDF2 import PdfMerger
import glob
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from adjustText import adjust_text
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#plt.rcParams["axes.spines.bottom"] = False
#plt.rcParams["axes.spines.left"] = False
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable text)
plt.rcParams['ps.fonttype'] = 42   # For EPS as well
plt.rcParams['svg.fonttype'] = 'none'  # SVG: write real <text> (editable in Illustrator)



plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16,        # Legend text size
    'legend.title_fontsize': 16,  # Legend title size
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 20
})

import sys
# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from .utilities import convert_adata_with_mygene, convert_with_mygene, rename_list_gene_dictionary, rename_adata_gene_dictionary
from ._lazy_corr import LazyGeneCorr, LazyPerturbCorr


def _blank_ax(ax, gene, message=None):
    """Turn *ax* into a clean blank panel with a 'No Guide Found' message.

    Hides spines, ticks, and tick labels, then centres a message.
    """
    if message is None:
        message = f'No Guide Found\nfor {gene}'
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.5, 0.5, message,
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, color='grey')
    return ax


def _safe_name(name):
    """Sanitize a string for use as a filename component.

    Gene/target names may contain path-unsafe characters (e.g. "TET1/2/3" for a
    multi-gene KO target). The slashes would otherwise be interpreted as
    directory separators and raise FileNotFoundError when saving. Mirrors
    _safe_sample_key in the html_* plotting modules.
    """
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(name))



def plot_umap_per_gene(mdata, Target_Gene, ensembl_to_symbol_file = None, ax=None,
 color='purple', save_path=None, save_name=None, figsize=(8,6), show=False, size = None,
 umap_subsample_frac=None, random_state=42, gene_name_key='gene_names', data_key='rna'):
    """Plot gene expression on UMAP embedding.

    Colors cells by expression level of a single gene on a precomputed UMAP.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object with 'rna' modality containing expression and UMAP coordinates.
    Target_Gene : str
        Gene symbol to color by. Must exist in var_names (after optional renaming).
    ensembl_to_symbol_file : str or None
        Path to TSV mapping Ensembl IDs to gene symbols. If None, var_names are
        used as-is. Passed to ``rename_adata_gene_dictionary``.
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None a new figure is created (standalone mode).
    color : str
        Matplotlib color for the high end of the expression gradient.
    save_path : str or None
        Directory to save the figure (standalone mode only).
    save_name : str or None
        File stem for the saved SVG (standalone mode only).
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    size : float or None
        Dot size passed to ``sc.pl.umap``. None uses scanpy default.
    umap_subsample_frac : float or None
        Fraction of cells to randomly subsample before plotting (0–1).
        None plots all cells.
    random_state : int
        Random seed for reproducible subsampling.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes with the plot, or None if the gene was not found (standalone mode).
    """

    # Subsample BEFORE the .copy() so we don't materialize the full mdata[data_key]
    src = mdata[data_key]
    if umap_subsample_frac is not None and 0 < umap_subsample_frac < 1.0:
        np.random.seed(random_state)
        n_cells = src.n_obs
        n_sample = int(n_cells * umap_subsample_frac)
        indices = np.random.choice(n_cells, size=n_sample, replace=False)
        src = src[sorted(indices)]

    if ensembl_to_symbol_file is None:
        renamed = src.copy()
        if gene_name_key is not None and gene_name_key in renamed.var.columns:
            renamed.var_names = renamed.var[gene_name_key].astype(str)
    else:
        renamed = rename_adata_gene_dictionary(src, ensembl_to_symbol_file)

    # Multiple Ensembl IDs can map to the same symbol, making var_names non-unique.
    # scanpy's obs_vector() would then match >1 column for a duplicated symbol and
    # ravel an (n_obs, k) block into a length-k*n_obs color vector, whose argsort
    # indexes out of bounds of obsm['X_umap'] (IndexError). Deduplicate so the query
    # symbol resolves to a single column (first copy keeps the original name).
    renamed.var_names_make_unique()

    # Check if query gene exists
    if Target_Gene not in renamed.var_names.values:
        print(f"Gene {Target_Gene} not found in mdata")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No UMAP to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
    
    
    # Check if gene exists
    gene_name_list = renamed.var_names.tolist()
    if Target_Gene not in gene_name_list:
        print("gene name is not found in adata")
        return None
    
    # Set color
    colors = ['lightgrey', color]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    title = f'{Target_Gene} Expression'
    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Plot on the provided/created axis
    plt.sca(ax)  # Set current axis to ensure scanpy uses correct figure -CC change
    # use_raw=False: var_names were renamed to symbols on .X only, not on .raw
    # (which keeps Ensembl IDs). scanpy defaults use_raw=True when .raw exists, so
    # it would look Target_Gene up in .raw and raise KeyError. Force it to read .X,
    # matching the existence check against renamed.var_names above.
    sc.pl.umap(renamed, color=Target_Gene, title=title, cmap=cmap, ax=ax, show=False, size = size, use_raw=False)
    
    # Rasterize ONLY the scatter points (collections) in this axis
    for collection in ax.collections:
        collection.set_rasterized(True)
    
    ax.set_title(title, fontsize=18, fontweight='bold', loc='center')
    ax.set_xlabel('UMAP 1', fontsize=10, fontweight = 'bold')
    ax.set_ylabel('UMAP 2', fontsize=10, fontweight = 'bold')

    if ax is None:
        plt.tight_layout()

    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{_safe_name(save_name)}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax


def plot_umap_per_gene_guide(mdata, Target_Gene, ax=None, color='red', save_path=None,
save_name=None, figsize=(8,6), show=False, size = None,
umap_subsample_frac=None, random_state=42, data_key='rna', prog_key='cNMF',
guide_targets_key='guide_targets', guide_assignment_key='guide_assignment'):
    """Plot guide perturbation signal on UMAP embedding.

    Colors cells by the sum of guide-assignment counts targeting a specific gene,
    highlighting which cells received guides for that gene.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object. Requires ``mdata['cNMF'].uns['guide_targets']``,
        ``mdata['cNMF'].obsm['guide_assignment']``, and UMAP coordinates in
        ``mdata['rna'].obsm['X_umap']``.
    Target_Gene : str
        Gene name that must appear in ``guide_targets``.
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None a new figure is created (standalone mode).
    color : str
        High-end color for the perturbation gradient.
    save_path : str or None
        Directory to save the figure (standalone mode only).
    save_name : str or None
        File stem for the saved SVG (standalone mode only).
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    size : float or None
        Dot size passed to ``sc.pl.umap``. None uses scanpy default.
    umap_subsample_frac : float or None
        Fraction of cells to randomly subsample before plotting (0–1).
    random_state : int
        Random seed for reproducible subsampling.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes with the plot, or None if the gene was not found (standalone mode).
    """

    # Find which guide columns target this gene and sum only those (avoids densifying full matrix)
    guide_targets = np.array(mdata[prog_key].uns[guide_targets_key])
    target_mask = guide_targets == Target_Gene

    if not target_mask.any():
        print(f"Gene {Target_Gene} not found in guide assignment")
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No UMAP to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax

    guide_mat = mdata[prog_key].obsm[guide_assignment_key]  # (cells x guides), sparse
    target_counts = np.asarray(guide_mat[:, target_mask].sum(axis=1)).flatten() # collapse guide all guides for target gene only

    adata = ad.AnnData(X=target_counts.reshape(-1, 1), obs=mdata[data_key].obs) # make adata with only one column cell x target_gene_guide
    adata.var_names = pd.Index([Target_Gene])
    # X_pca is not used for the UMAP scatter below; copy only if present so h5mu
    # files that ship X_umap without a matching X_pca don't crash here.
    if 'X_pca' in mdata[data_key].obsm:
        adata.obsm['X_pca'] = mdata[data_key].obsm['X_pca']
    adata.obsm['X_umap'] = mdata[data_key].obsm['X_umap']

    # Optionally subsample cells for faster plotting
    if umap_subsample_frac is not None and 0 < umap_subsample_frac < 1.0:
        np.random.seed(random_state)
        n_cells = adata.n_obs
        n_sample = int(n_cells * umap_subsample_frac)
        indices = np.random.choice(n_cells, size=n_sample, replace=False)
        adata = adata[sorted(indices)]

    # Set color
    colors = ['lightgrey', color]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    title = f'{Target_Gene} Perturbation'
    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Plot on the provided/created axis
    plt.sca(ax)  # Set current axis to ensure scanpy uses correct figure -CC change

    sc.pl.umap(adata, color=Target_Gene, title=title, cmap=cmap, ax=ax, show=False, size=size)
    
    # Rasterize ONLY the scatter points (collections) in this axis
    for collection in ax.collections:
        collection.set_rasterized(True)
    
    ax.set_title(title, fontsize=18, fontweight='bold', loc='center')
    ax.set_xlabel('UMAP 1', fontsize=10, fontweight = 'bold')
    ax.set_ylabel('UMAP 2', fontsize=10, fontweight = 'bold')

    if ax is None:
        plt.tight_layout()

    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{_safe_name(save_name)}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax


def plot_top_program_per_gene(mdata,  Target_Gene, ensembl_to_symbol_file = None,top_n_programs=10,
                            ax=None, save_path=None, save_name=None,
                               figsize=(5,8), show=False, gene_name_key='gene_names',
                               data_key='rna', prog_key='cNMF'):
    """Horizontal bar plot of the top cNMF programs by gene-loading score for a gene.

    Reads the cNMF loadings matrix from ``mdata['cNMF'].varm['loadings']`` and
    displays the ``top_n_programs`` programs with the highest loading for
    ``Target_Gene``.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object with 'cNMF' and 'rna' modalities.
    Target_Gene : str
        Gene symbol to query.
    ensembl_to_symbol_file : str or None
        Path to Ensembl-to-symbol TSV. If None, ``mdata['rna'].var_names``
        are used directly. Passed to ``rename_list_gene_dictionary``.
    top_n_programs : int
        Number of top-loading programs to display.
    ax : matplotlib.axes.Axes or None
        Axis to draw on. If None a new figure is created.
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes with the plot, or None if gene not found (standalone mode).
    """

    # read cNMF gene program matrix
    X =  mdata[prog_key].varm["loadings"]

    # rename gene
    if ensembl_to_symbol_file is None:
        if gene_name_key is not None and gene_name_key in mdata[data_key].var.columns:
            col_names = mdata[data_key].var[gene_name_key].astype(str)
        else:
            col_names = mdata[data_key].var_names
        df_renamed = pd.DataFrame(data=X, columns = col_names)
    else:
        renamed_gene_list = rename_list_gene_dictionary( mdata[data_key].var_names, ensembl_to_symbol_file)
        df_renamed = pd.DataFrame(data=X, columns = renamed_gene_list)

    
    # Check if gene exists
    if Target_Gene not in df_renamed.columns:
        print(f"Gene {Target_Gene} not found in mdata")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No program to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
    
    
    
    # Multiple Ensembl IDs collapsing to the same symbol would make
    # df_renamed[Target_Gene] return a DataFrame and break nlargest().
    # Fail explicitly rather than silently averaging or picking one column.
    matching_cols = (df_renamed.columns == Target_Gene).sum()
    if matching_cols > 1:
        raise ValueError(
            f"Gene symbol {Target_Gene!r} maps to {matching_cols} columns in the "
            f"loading matrix (duplicate symbols from multiple Ensembl IDs). "
            f"Deduplicate the gene→symbol mapping upstream before calling "
            f"plot_top_program_per_gene."
        )

    n_programs = df_renamed.shape[0]
    if top_n_programs > n_programs:
        raise ValueError(
            f"top_n_programs={top_n_programs} exceeds the number of programs "
            f"available in the loading matrix ({n_programs}). Lower top_n_programs."
        )

    # sort top x program
    df_sorted = df_renamed[Target_Gene].nlargest(top_n_programs)
    df_sorted = df_sorted.sort_values(ascending=True)

    
    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(df_sorted)), df_sorted.values, color='#808080', alpha=0.8)
    
    # Customize the plot
    ax.set_title(f"Top Loading Program for {Target_Gene}",fontsize=12, fontweight='bold', loc='center')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index, fontsize=10)
    ax.set_xlabel('Gene Loading Score (z-score)',fontsize=10, fontweight='bold') #, loc='center')
    ax.set_ylabel(f'Program Name',fontsize=10, fontweight='bold', loc='center')
    
    # Format x-axis
    ax.set_xlim(0, max(df_sorted.values) * 1.1)
    ax.ticklabel_format(style='scientific', axis='x' )#scilimits=(0,0))
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if standalone:
        plt.tight_layout()

    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{_safe_name(save_name)}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



def perturbed_gene_dotplot(mdata,  Target_Gene, ensembl_to_symbol_file=None, dotplot_groupby='sample', save_name=None,
                          save_path=None, figsize=(3, 2), show=False, ax=None, gene_name_key='gene_names', data_key='rna'):
    """Scanpy dot plot of a single gene's expression grouped by a categorical variable.

    Wraps ``sc.pl.dotplot`` for a single perturbed gene, split by e.g. sample/day.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object with 'rna' modality.
    Target_Gene : str
        Gene symbol to plot.
    ensembl_to_symbol_file : str or None
        Path to Ensembl-to-symbol TSV. Passed to ``rename_adata_gene_dictionary``.
    dotplot_groupby : str
        Column in ``mdata['rna'].obs`` used to group cells (e.g. 'sample').
        Forwarded to ``sc.pl.dotplot(groupby=...)``.
    save_name : str or None
        File stem for the saved PNG (standalone mode only).
    save_path : str or None
        Directory to save the figure.
    figsize : tuple of float
        (width, height) passed to ``sc.pl.dotplot``.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes or None
        The main-plot axes, or None if gene not found (standalone mode).
    """
    # read in adata
    if ensembl_to_symbol_file is None:
        renamed = mdata[data_key].copy()
        if gene_name_key is not None and gene_name_key in renamed.var.columns:
            renamed.var_names = renamed.var[gene_name_key].astype(str)
    else:
        renamed = rename_adata_gene_dictionary(mdata[data_key],ensembl_to_symbol_file)

    renamed.var_names_make_unique()


    # Check if query gene exists
    if Target_Gene not in renamed.var_names.values:
        print(f"Gene {Target_Gene} not found in mdata")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No gene to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
    
    
    if save_name is None:
        save_name = f"{Target_Gene} Expression by days"
    
    # Create the dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
        standalone = True
        dp = sc.pl.dotplot(renamed, Target_Gene, groupby=dotplot_groupby,
                          figsize=figsize, swap_axes=True, dendrogram=False,
                          show=False, return_fig=True, use_raw=False)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        standalone = False
        fig = ax.get_figure()
        dp = sc.pl.dotplot(renamed, Target_Gene, groupby=dotplot_groupby,
                          swap_axes=True, dendrogram=False, show=False,
                          return_fig=True, ax=ax, use_raw=False)
        dp.make_figure()
    
    ax.set_title(f"{Target_Gene} Expression", fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Gene', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Day', fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    label = list(mdata[data_key].obs[dotplot_groupby].cat.categories)  # Use categories instead
    
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)

    if standalone:
        plt.tight_layout()

    # save fig (only in standalone mode)
    if standalone and save_name and save_path:
        fig.savefig(f"{save_path}/{_safe_name(save_name)}.png", format='png', bbox_inches='tight', dpi=300)  # Changed to png

    # Control whether to display the plot (only in standalone mode)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax



def plot_log2FC(perturb_path, Target, perturb_target_col="target_name", perturb_program_col="program_name",
                perturb_log2fc_col='log2FC', num_item=5, significance_threshold=0.05, save_path=None, save_name=None,
                figsize=(5, 4), show=False, ax=None, Day ="", perturb_df=None):
    """Horizontal bar plot of top up- and down-regulated programs by log2 fold-change.

    Reads perturbation-association results (one row per target-gene × program),
    filters to significant hits, and shows the ``num_item`` most positive and
    ``num_item`` most negative log2FC programs.

    Parameters
    ----------
    perturb_path : str
        Path to the perturbation TSV file. Ignored when ``perturb_df`` is provided.
    Target : str
        Gene name to filter on in the ``perturb_target_col`` column.
    perturb_target_col : str
        Column name for target genes in the perturbation results.
    perturb_program_col : str
        Column name for program identifiers in the perturbation results.
    perturb_log2fc_col : str
        Column name for log2 fold-change values.
    num_item : int
        Number of top up- and down-regulated programs to display.
    significance_threshold : float
        Adjusted p-value cutoff; only rows with ``adj_pval < significance_threshold``
        are included.
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.
    Day : str
        Sample/day label appended to the plot title.
    perturb_df : pandas.DataFrame or None
        Pre-loaded perturbation DataFrame. When provided, ``perturb_path``
        is not read, avoiding duplicate I/O.

    Returns
    -------
    (matplotlib.axes.Axes, pandas.DataFrame)
        The axes and a DataFrame of the plotted programs.
    """

    # read df (skip if pre-loaded DataFrame provided)
    if perturb_df is not None:
        df = perturb_df
    else:
        df = pd.read_csv(perturb_path, sep="\t")
    
    # Check if query gene exists
    if Target not in df[perturb_target_col].values:
        print(f"Gene {Target} not found in mdata")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
        return ax, pd.DataFrame()

    # Sort by log2FC
    df_sorted = df.loc[df[perturb_target_col] == Target]
    df_sorted = df_sorted[df_sorted['adj_pval'] < significance_threshold]
    df_sorted = df_sorted.sort_values(by=perturb_log2fc_col, ascending=False)
    # Get top and bottom gene
    top_df = df_sorted.head(num_item)
    bottom_df = df_sorted.tail(num_item)
    # Combine and add category
    top = top_df.copy()
    bottom = bottom_df.copy()
    # Combine data
    plot_data = pd.merge(top, bottom, how='outer')
    plot_data = plot_data.sort_values(by=perturb_log2fc_col, ascending=False)
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Create horizontal bar plot
    colors = ['red' if x > 0 else 'blue' for x in plot_data[perturb_log2fc_col]]
    bars = ax.barh(range(len(plot_data)), plot_data[perturb_log2fc_col], color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data[perturb_program_col],fontsize=10) 


    ax.set_xlabel('Effect on Program Expression',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel( "Name",fontsize=10, fontweight='bold', loc='center')
    ax.set_title(f"Log2 Fold-Change with {Target} {Day}",fontsize=14, fontweight='bold', loc='center')

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)


    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Upregulated'),
                      Patch(facecolor='blue', alpha=0.7, label='Downregulated')]
    ax.legend(handles=legend_elements) #loc='lower right')
    
    # Adjust layout
    ax.grid(axis='x', alpha=0.3)

    if standalone:
        plt.tight_layout()

    # Save if path provided (only when standalone)
    if standalone and save_path:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only when standalone)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax, plot_data
 



def plot_volcano(perturb_path, Target, perturb_target_col="target_name", perturb_program_col="program_name", perturb_log2fc_col = 'log2FC',
                 volcano_log2fc_min=-0.05, volcano_log2fc_max=0.05, significance_threshold=0.05, save_path=None,
                 save_name=None, figsize=(5, 4), show=False, ax=None, Day ="",
                 perturb_df=None, run_adjust_text=True):
    """Volcano plot of perturbation log2FC vs adjusted p-value for one target gene.

    Programs exceeding the fold-change and significance thresholds are
    highlighted and labeled.

    Parameters
    ----------
    perturb_path : str
        Path to the perturbation TSV file. Ignored when ``perturb_df`` is provided.
    Target : str
        Gene name to filter on in the ``perturb_target_col`` column.
    perturb_target_col : str
        Column name for target genes in the perturbation results.
    perturb_program_col : str
        Column name for program identifiers used as text labels.
    perturb_log2fc_col : str
        Column name for log2 fold-change values (x-axis).
    volcano_log2fc_min : float
        Lower log2FC threshold; programs below this are "down-regulated".
    volcano_log2fc_max : float
        Upper log2FC threshold; programs above this are "up-regulated".
    significance_threshold : float
        Adjusted p-value cutoff for significance (horizontal dashed line).
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.
    Day : str
        Sample/day label appended to the plot title.
    perturb_df : pandas.DataFrame or None
        Pre-loaded perturbation DataFrame to avoid duplicate file reads.
    run_adjust_text : bool
        If True, run ``adjustText.adjust_text`` on labels inside this function.
        Set to False when called from ``create_comprehensive_plot`` to avoid
        running adjust_text twice (once here, once in the caller with custom font size).

    Returns
    -------
    (matplotlib.axes.Axes, pandas.DataFrame, list)
        The axes, a DataFrame of significant programs, and the list of
        matplotlib Text objects (for external adjust_text calls).
    """

    # read df (skip if pre-loaded DataFrame provided)
    if perturb_df is not None:
        df = perturb_df
    else:
        df = pd.read_csv(perturb_path, sep="\t")
       
    # Check if query gene exists
    if Target not in df[perturb_target_col].values:
        print(f"Gene {Target} not found in mdata")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
        return ax, pd.DataFrame(), []

    df_program = df.loc[df[perturb_target_col] == Target]
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Plot all points
    ax.scatter(x=df_program[perturb_log2fc_col], 
               y=df_program['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Not significant", color="grey")
    
    # Highlight down- or up-regulated genes
    down = df_program[(df_program[perturb_log2fc_col] <= volcano_log2fc_min) & 
                      (df_program['adj_pval'] <= significance_threshold)]
    up = df_program[(df_program[perturb_log2fc_col] >= volcano_log2fc_max) &
                    (df_program['adj_pval'] <= significance_threshold)]
    
    ax.scatter(x=down[perturb_log2fc_col], 
               y=down['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Down-regulated", color="blue")
    ax.scatter(x=up[perturb_log2fc_col], 
               y=up['adj_pval'].apply(lambda x: -np.log10(x)), 
               s=10, label="Up-regulated", color="red")
    
    # Add text labels
    texts = []
    for i, r in up.iterrows():
        texts.append(ax.text(x=r[perturb_log2fc_col], y=-np.log10(r['adj_pval']), 
                            s=r[perturb_program_col], fontsize=10, zorder=10))
    for i, r in down.iterrows():
        texts.append(ax.text(x=r[perturb_log2fc_col], y=-np.log10(r['adj_pval']), 
                            s=r[perturb_program_col], fontsize=10, zorder=10))
    
    # Adjust text to avoid overlaps
    if run_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=ax)
    
    
    # Set labels and lines
    ax.set_title(f"Volcano Plot for {Target} {Day}",fontsize=14, fontweight='bold', loc='center')
    ax.set_xlabel('Effect on Program Expression',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel("Adjusted p-value, -log10",fontsize=10, fontweight='bold', loc='center')
    ax.axvline(volcano_log2fc_min, color="grey", linestyle="--")
    ax.axvline(volcano_log2fc_max, color="grey", linestyle="--")
    ax.axhline(-np.log10(significance_threshold), color="grey", linestyle="--")
    ax.legend()
    ax.set_title(f"Volcano Plot for {Target} {Day}",fontsize=14, fontweight='bold', loc='center')

    if standalone:
        plt.tight_layout()

    # Save if path provided (only when standalone)
    if standalone and save_path:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only when standalone)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax, pd.merge(up, down, how='outer'), texts   




def programs_dotplot(mdata, Target, dotplot_groupby="sample", program_list=None,
                     save_path=None, save_name=None, figsize=(5, 4), show=False, ax=None, Day="", prog_key='cNMF'):
    """Scanpy dot plot of cNMF program loading scores grouped by a categorical variable.

    Shows selected programs (e.g. significant ones from ``plot_log2FC``) as a
    dot plot split by sample/day using ``sc.pl.dotplot``.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object with 'cNMF' modality containing program loading scores.
    Target : str
        Gene name used only for the plot title.
    dotplot_groupby : str
        Column in ``mdata['cNMF'].obs`` to group cells by (e.g. 'sample').
        Forwarded to ``sc.pl.dotplot(groupby=...)``.
    program_list : list of str or None
        Program names to include. If None or empty, a blank placeholder is shown.
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) passed to ``sc.pl.dotplot``.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.
    Day : str
        Sample/day label appended to the plot title.

    Returns
    -------
    matplotlib.axes.Axes or None
        The main-plot axes, or None if no programs to display (standalone mode).
    """

    # make anndata from program loadings
    adata_new = mdata[prog_key]

    program_name_list = adata_new.var_names.tolist() # get all programs 
    
    program_names = list(map(str, program_list)) if program_list is not None else [] # make program str if given 
    program_names = [program for program in program_names if program in program_name_list]
    
    if not program_names:

        print("No significant Programs")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No programs to display',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{Target} Perturbed Program Loading Scores {Day}", 
                        fontweight='bold', loc='center')
            return ax

    program_names = program_names[::-1]
    
    # Create dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
        standalone = True
        dp = sc.pl.dotplot(adata_new, program_names, groupby=dotplot_groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True, use_raw=False)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        standalone = False
        fig = ax.get_figure()
        dp = sc.pl.dotplot(adata_new, program_names, groupby=dotplot_groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True, ax=ax, use_raw=False)
        dp.make_figure()
    
    ax.set_title(f"{Target} Perturbed Program Loading Scores {Day}", 
                fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Program Name', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel("Day", fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    # Check if it's already categorical
    if hasattr(adata_new.obs[dotplot_groupby], 'cat'):
        label = list(adata_new.obs[dotplot_groupby].cat.categories)
    else:
        label = list(adata_new.obs[dotplot_groupby].unique())
        
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)

    if standalone:
        plt.tight_layout()

    # Save if path provided (only when standalone)
    if standalone and save_path and save_name:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only in standalone mode)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax
 

def compute_gene_correlation_matrix(mdata, ensembl_to_symbol_file=None, gene_name_key='gene_names',
                                    precomputed_path=None, save_path=None, data_key='rna', prog_key='cNMF'):
    """Build a rank-K factored stand-in for the gene-by-gene loading correlation.

    Equivalent to the Pearson correlation of cNMF gene-loading vectors but
    returned as a ``LazyGeneCorr`` shim that exposes the same ``.loc[gene]``
    / ``corr[gene]`` / ``.columns`` API as the legacy DataFrame, without
    materializing the dense G x G matrix.

    Parameters
    ----------
    mdata : muon.MuData
        MuData with ``mdata['cNMF'].varm['loadings']`` (programs x genes)
        and ``mdata['rna'].var_names``.
    ensembl_to_symbol_file : str or None
        Path to Ensembl-to-symbol TSV. If None, ``mdata['rna'].var_names``
        are used.
    gene_name_key : str
        Column of ``mdata['rna'].var`` to use when no mapping file is given.
    precomputed_path : str or None
        If set and the file exists, load the factor from this ``.npz`` and
        skip recomputation.
    save_path : str or None
        If set, write the computed factor to this ``.npz`` for future runs.

    Returns
    -------
    LazyGeneCorr
        Drop-in shim with ``.loc[gene]`` / ``.columns`` / ``corr[gene]``.
    """

    if precomputed_path is not None and Path(precomputed_path).exists():
        return LazyGeneCorr.load_npz(precomputed_path)

    X = mdata[prog_key].varm["loadings"]

    if ensembl_to_symbol_file is None:
        if gene_name_key is not None and gene_name_key in mdata[data_key].var.columns:
            col_names = mdata[data_key].var[gene_name_key].astype(str).tolist()
        else:
            col_names = list(mdata[data_key].var_names)
    else:
        col_names = list(rename_list_gene_dictionary(mdata[data_key].var_names, ensembl_to_symbol_file))

    corr = LazyGeneCorr.from_loadings(X, gene_names=col_names)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        corr.save_npz(save_path)

    return corr

 

def analyze_correlations(gene_loading_corr_matrix, Target, top_corr_genes=5, save_path=None,
                         save_name=None, figsize=(10, 8), show=False, ax=None):
    """Horizontal bar plot of genes most/least correlated with a target gene in cNMF loadings.

    Uses the precomputed gene × gene correlation matrix from
    ``compute_gene_correlation_matrix`` to show the top positively and
    negatively correlated genes.

    Parameters
    ----------
    gene_loading_corr_matrix : pandas.DataFrame
        Symmetric gene × gene correlation matrix (output of
        ``compute_gene_correlation_matrix``).
    Target : str
        Gene symbol to query correlations for.
    top_corr_genes : int
        Number of top positively and top negatively correlated genes to show.
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes with the plot, or None if gene not found (standalone mode).
    """

    # check if gene exists
    if Target not in gene_loading_corr_matrix.columns:
        print(f"Gene {Target} not found in mdata")
        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{_safe_name(save_name)}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No correlation to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
    

    # Get correlations with the target program
    target_correlations = gene_loading_corr_matrix[Target]
    target_correlations = target_correlations.drop(Target)  # Remove self-correlation
    
    # Sort correlations
    sorted_correlations = target_correlations.sort_values(ascending=False)
    
    # Get top and bottom gene
    top = sorted_correlations[sorted_correlations > 0].head(top_corr_genes)
    bottom = sorted_correlations[sorted_correlations < 0].tail(top_corr_genes)
    
    # Combine for plotting
    combined_correlations = pd.concat([bottom, top])
    combined_correlations = combined_correlations.sort_values(ascending=False)
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Create horizontal bar plot
    colors = ['blue' if x < 0 else 'red' for x in combined_correlations]
    bars = ax.barh(range(len(combined_correlations)), combined_correlations.values, 
                   color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_title(f"Gene Loading Correlation for {Target}",fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Gene Name',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Correlation Coefficient',fontsize=10, fontweight='bold', loc='center')
    
    # Set y-axis labels
    ax.set_yticks(range(len(combined_correlations)))
    ax.set_yticklabels(combined_correlations.index, rotation=45, ha='right')
    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive correlation'),
                      Patch(facecolor='blue', alpha=0.7, label='Negative correlation')]
    ax.legend(handles=legend_elements )# loc='lower right')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust layout (only in standalone mode)
    if standalone:
        plt.tight_layout()

    # Save if path provided (only in standalone mode)
    if standalone and save_path:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only in standalone mode)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax



def compute_gene_waterfall_cor(perturb_path, perturb_log2fc_col='log2FC',
                               precomputed_path=None, save_path=None):
    """Build a rank-K factored stand-in for the per-sample target x target correlation.

    Equivalent to ``pivot_df.T.corr()`` over a (target x program) pivot of
    log2FC values, but returned as a ``LazyPerturbCorr`` shim that
    preserves the pairwise-complete NaN semantics and exposes the
    ``.loc[gene]`` / ``.index`` / ``.copy()`` API used by callers without
    materializing the dense T x T matrix.

    Parameters
    ----------
    perturb_path : str
        TSV with columns ``target_name``, ``program_name``, and the log2FC
        column.
    perturb_log2fc_col : str
        Column name for log2 fold-change values used to build the pivot.
    precomputed_path : str or None
        If set and the file exists, load the factor from this ``.npz`` and
        skip recomputation.
    save_path : str or None
        If set, write the computed factor to this ``.npz`` for future runs.

    Returns
    -------
    LazyPerturbCorr
        Drop-in shim with ``.loc[gene]`` returning a correlation row.
    """

    if precomputed_path is not None and Path(precomputed_path).exists():
        return LazyPerturbCorr.load_npz(precomputed_path)

    df = pd.read_csv(perturb_path, sep='\t', index_col=0)
    pivot_df = df.pivot_table(index='target_name', columns='program_name',
                              values=perturb_log2fc_col)
    del df

    corr = LazyPerturbCorr.from_pivot(pivot_df)
    del pivot_df

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        corr.save_npz(save_path)

    return corr



def create_gene_correlation_waterfall(corr_matrix, Target_Gene, top_corr_genes=5, save_path=None,
                         save_name=None, figsize=(3, 5), show=False, ax=None, Day ="",
                         run_adjust_text=True):
    """Waterfall scatter plot of gene-gene perturbation correlation for one target.

    Plots all genes' perturbation-profile correlations with ``Target_Gene``
    (from ``compute_gene_waterfall_cor``), highlighting and labeling the
    most positively and negatively correlated genes.

    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Target-gene × target-gene correlation matrix (output of
        ``compute_gene_waterfall_cor`` for a single sample).
    Target_Gene : str
        Gene to extract correlations for (row in ``corr_matrix``).
    top_corr_genes : int
        Number of top positively and negatively correlated genes to label.
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    figsize : tuple of float
        (width, height) in inches. Only used in standalone mode.
    show : bool
        If True, call ``plt.show()`` in standalone mode.
    ax : matplotlib.axes.Axes or None
        Axis for gridspec mode. If None a new figure is created.
    Day : str
        Sample/day label appended to the plot title.
    run_adjust_text : bool
        If True, run ``adjustText.adjust_text`` inside this function. Set to
        False when called from ``create_comprehensive_plot`` which applies its
        own adjust_text pass with custom font sizes.

    Returns
    -------
    (matplotlib.axes.Axes, list)
        The axes and the list of matplotlib Text objects for the labels.
    """

    # Create figure/axes early so the absent-target branch can draw a placeholder.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # A target with all-NaN log2FC is dropped from the pivot index in
    # compute_gene_waterfall_cor (pivot_table drops all-NaN rows), so it is
    # absent from corr_matrix even though it is in the gene list. Render an
    # empty panel rather than raising KeyError, so the rest of the
    # comprehensive plot for this gene is still produced.
    if Target_Gene not in corr_matrix:
        ax.text(0.5, 0.5, f"No perturbation\ncorrelation data\nfor {Target_Gene}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, style='italic', color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Gene Perturbation Correlation for {Target_Gene} {Day}",
                     fontsize=14, fontweight='bold', loc='center')
        if standalone:
            plt.tight_layout()
            if save_path:
                fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg',
                            bbox_inches='tight', dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)
        return ax, []

    # Convert to DataFrame and sort
    gene_corrs = corr_matrix.loc[Target_Gene].dropna()
    corr_df = (gene_corrs).sort_values(ascending = False)
    
    # Get top N positive and bottom N negative correlations for labeling
    top_positive = corr_df.head(top_corr_genes)
    top_negative = corr_df.tail(top_corr_genes)
    genes_to_label = set(top_positive.index.tolist() + top_negative.index.tolist())
    
    # Use ALL correlations for plotting
    plot_data = corr_df

    # Plot only the markers (no line)
    ax.scatter(range(len(plot_data)), plot_data.values, 
               color='gray', s=16)
    
    # Add horizontal reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Remove x-axis labels
    ax.set_xticks([])

    
    # Highlight the labeled genes with larger markers and add annotations
    texts = []
    for i, (gene, values) in enumerate(plot_data.items()):
        if gene in genes_to_label:
            ax.plot(i, values, 'o', color='darkgray', markersize=5, zorder=5)
            text = ax.text(i, values, gene, fontsize=10, style='italic', ha='center')
            texts.append(text)
            
    if run_adjust_text:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax, fontsize=12)

    ax.set_title(f"Gene Perturbation Correlation for {Target_Gene} {Day}",fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel(f'Correlation of Program\nExpression with\n{Target_Gene} Perturbation',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Perturbed Genes',fontsize=10, fontweight='bold', loc='center')

    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout (only in standalone mode)
    if standalone:
        plt.tight_layout()

    # Save if path provided (only in standalone mode)
    if standalone and save_path:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only in standalone mode)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    
    return ax, texts



def plot_perturbation_vs_control(mdata, target_gene, gene_name_key='gene_names', ax=None, figsize=(3.2, 5.0),
                                 control_target_name='non-targeting',
                                 save_path=None, save_name=None, show=True,
                                 data_key='rna', prog_key='cNMF',
                                 guide_targets_key='guide_targets', guide_assignment_key='guide_assignment'):
    """
    Bar plot comparing expression of a targeted gene in perturbed vs non-targeting control cells,
    normalized to control mean (displayed as %).

    Parameters
    ----------
    mdata : MuData
        Must have mdata['cNMF'].obsm['guide_assignment'], mdata['cNMF'].uns['guide_targets'],
        and gene names accessible via gene_name_key in mdata['rna'].var or mdata['rna'].var_names.
    target_gene : str
        Gene name to plot (must be both a perturbation target and found in gene names).
    gene_name_key : str or None
        Column name in mdata['rna'].var for gene symbol lookup.
        If None or column not found, falls back to mdata['rna'].var_names.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
    figsize : tuple
        Figure size (only used in standalone mode when ax is None).
    control_target_name : str
        Name of the non-targeting control in guide_targets (e.g. 'non-targeting', 'CTRL').
    save_path : str or None
        Directory to save the SVG (standalone mode only).
    save_name : str or None
        File stem for the saved figure.
    show : bool
        If True, call ``plt.show()`` in standalone mode; otherwise close the figure.
    """
    from scipy import sparse

    assert mdata[data_key].n_obs == mdata[prog_key].n_obs, \
        "RNA and cNMF have different number of cells"

    guide_targets = mdata[prog_key].uns[guide_targets_key]
    ga = mdata[prog_key].obsm[guide_assignment_key]
    X = mdata[data_key].X

    # Gene index in expression matrix
    if gene_name_key is not None and gene_name_key in mdata[data_key].var.columns:
        gene_mask = mdata[data_key].var[gene_name_key] == target_gene
    else:
        gene_mask = mdata[data_key].var_names == target_gene

    if gene_mask.sum() == 0:
        print(f"Gene {target_gene} not found in expression matrix")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        return None
    gene_idx = np.where(gene_mask)[0][0]

    # Normalize to counts per 10k
    if sparse.issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten()
        gene_expr = X[:, gene_idx].toarray().ravel()
    else:
        row_sums = X.sum(axis=1)
        gene_expr = X[:, gene_idx]
    row_sums[row_sums == 0] = 1
    gene_expr_norm = (gene_expr / row_sums) * 1e4

    # Non-targeting control cells
    nt_idx = np.where(np.isin(guide_targets, control_target_name))[0]
    control_mask = np.array(ga[:, nt_idx].sum(axis=1)).flatten() > 0
    if control_mask.sum() == 0:
        raise ValueError(
            f"0 control cells for control_target_name='{control_target_name}'. "
            f"Check np.unique(mdata['{prog_key}'].uns['{guide_targets_key}']) for the correct label."
        )

    # Perturbed cells
    target_idx = np.where(guide_targets == target_gene)[0]
    if len(target_idx) == 0:
        print(f"Gene {target_gene} not found in guide_targets")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        return None
    perturbed_mask = np.array(ga[:, target_idx].sum(axis=1)).flatten() > 0

    ctrl_expr = gene_expr_norm[control_mask]
    pert_expr = gene_expr_norm[perturbed_mask]

    # Normalize both to control mean → values are now fractions (1.0 = 100%)
    ctrl_mean = np.mean(ctrl_expr)
    if np.isnan(ctrl_mean):
        print(f"WARNING: ctrl_mean is NaN (n_ctrl={len(ctrl_expr)}); bar plot will be empty.")
    norm_means = np.array([np.mean(pert_expr), ctrl_mean]) / ctrl_mean
    norm_sems  = np.array([
        np.std(pert_expr) / np.sqrt(len(pert_expr)),
        np.std(ctrl_expr) / np.sqrt(len(ctrl_expr)),
    ]) / ctrl_mean

    n_pert = len(pert_expr)
    n_ctrl = len(ctrl_expr)

    # ── Style ────────────────────────────────────────────────────────────────
    COLOR_PERT = '#D62728'
    COLOR_CTRL = '#999999'
    ECAP  = dict(linewidth=1.5, ecolor='black')

    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    bars = ax.bar(
        [0, 1], norm_means, yerr=norm_sems,
        capsize=6, color=[COLOR_PERT, COLOR_CTRL], edgecolor='none',
        width=0.55, error_kw=ECAP,
    )

    # Dashed reference line at 100 %
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=0)

    # Y-axis: percentage ticks
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(round(v * 100))}%'))
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

    # X-axis tick labels  (gene name + n on separate line, matching reference)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f'{target_gene}\n[n = {n_pert}]',
        f'Non\nTargeting\n[n = {n_ctrl}]',
    ], fontsize=10)

    ax.set_ylabel(f'{target_gene} Expression', fontsize=11, labelpad=8)
    effect_pct = (1 - norm_means[0]) * 100
    if np.isnan(effect_pct):
        title_str = 'CRISPRi knockdown:\neffect N/A'
    else:
        title_str = f'CRISPRi knockdown:\n>{int(round(effect_pct))}% effect'
    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)

    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)

    # Adjust layout (only in standalone mode)
    if standalone:
        plt.tight_layout()

    # Save if path provided (only in standalone mode)
    if standalone and save_path:
        fig.savefig(f'{save_path}/{_safe_name(save_name)}.svg', format='svg', bbox_inches='tight', dpi=300)

    # Control whether to display the plot (only in standalone mode)
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax


def plot_perturbation_vs_control_by_condition(mdata, target_gene, condition_key='sample',
                                               gene_name_key='gene_names', figsize=(8, 5),
                                               control_target_name='non-targeting',
                                               ax=None,
                                               save_path=None, save_name=None, show=True, PDF=True,
                                               data_key='rna', prog_key='cNMF',
                                               guide_targets_key='guide_targets', guide_assignment_key='guide_assignment'):
    """
    Grouped bar chart comparing perturbed vs control expression across conditions.

    For each condition (+ "All" pooled), plots a pair of bars (perturbed in red,
    control in grey) normalized to that condition's control mean (displayed as %).
    Scales to any number of conditions on a single axis.

    Parameters
    ----------
    mdata : MuData
        Must have mdata['cNMF'].obsm['guide_assignment'], mdata['cNMF'].uns['guide_targets'],
        and gene names accessible via gene_name_key in mdata['rna'].var or mdata['rna'].var_names.
        Condition info is read from mdata['rna'].obs[condition_key].
    target_gene : str
        Gene name to plot (must be both a perturbation target and found in gene names).
    condition_key : str
        Column in mdata['rna'].obs that defines the conditions
        (e.g. 'sample' with values D0, D1, D2, D3).
    gene_name_key : str or None
        Column name in mdata['rna'].var for gene symbol lookup.
        If None or column not found, falls back to mdata['rna'].var_names.
    figsize : tuple
        (width, height) in inches. Only used in standalone mode when ax is None.
    control_target_name : str
        Name of the non-targeting control in guide_targets (e.g. 'non-targeting', 'CTRL').
    ax : matplotlib.axes.Axes or None
        Pre-created axis to draw into. If None, a new figure is created (standalone mode).
    save_path : str or None
        Directory to save the figure (standalone mode only).
    save_name : str or None
        File stem for the saved figure (standalone mode only).
    show : bool
        If True, display the figure interactively (standalone mode only).
    PDF : bool
        If True, save as PDF; otherwise save as SVG (standalone mode only).

    Returns
    -------
    matplotlib.axes.Axes
        When ax is provided (embedded mode).
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        When no ax provided (standalone mode).
    """
    from scipy import sparse

    assert mdata[data_key].n_obs == mdata[prog_key].n_obs, \
        "RNA and cNMF have different number of cells"

    guide_targets = mdata[prog_key].uns[guide_targets_key]
    ga = mdata[prog_key].obsm[guide_assignment_key]
    X = mdata[data_key].X

    # Gene index in expression matrix
    if gene_name_key is not None and gene_name_key in mdata[data_key].var.columns:
        gene_mask = mdata[data_key].var[gene_name_key] == target_gene
    else:
        gene_mask = mdata[data_key].var_names == target_gene

    if gene_mask.sum() == 0:
        print(f"Gene {target_gene} not found in expression matrix")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', transform=ax.transAxes)
            return ax
        return None

    gene_idx = np.where(gene_mask)[0][0]

    # Normalize to counts per 10k
    if sparse.issparse(X):
        row_sums = np.array(X.sum(axis=1)).flatten()
        gene_expr = X[:, gene_idx].toarray().ravel()
    else:
        row_sums = X.sum(axis=1)
        gene_expr = X[:, gene_idx]
    row_sums[row_sums == 0] = 1
    gene_expr_norm = (gene_expr / row_sums) * 1e4

    # Non-targeting control cells
    nt_idx = np.where(np.isin(guide_targets, control_target_name))[0]
    control_mask = np.array(ga[:, nt_idx].sum(axis=1)).flatten() > 0
    if control_mask.sum() == 0:
        print(f"WARNING: 0 control cells for control_target_name='{control_target_name}'. "
              f"Check np.unique(mdata['{prog_key}'].uns['{guide_targets_key}']) for the correct label.")

    # Perturbed cells
    target_idx = np.where(guide_targets == target_gene)[0]
    if len(target_idx) == 0:
        print(f"Gene {target_gene} not found in guide_targets")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                    ha='center', va='center', transform=ax.transAxes)
            return ax
        return None
    perturbed_mask = np.array(ga[:, target_idx].sum(axis=1)).flatten() > 0

    # Get conditions
    conditions_series = mdata[data_key].obs[condition_key]
    if hasattr(conditions_series, 'cat'):
        conditions = list(conditions_series.cat.categories)
    else:
        conditions = sorted(conditions_series.unique())

    # Group labels: "All" + each condition
    group_labels = ['All'] + [str(c) for c in conditions]
    n_groups = len(group_labels)

    # Compute means and SEMs for each group
    pert_means, ctrl_means = [], []
    pert_sems, ctrl_sems = [], []
    pert_ns, ctrl_ns = [], []

    for label in group_labels:
        if label == 'All':
            cond_mask = np.ones(len(gene_expr_norm), dtype=bool)
        else:
            cond_mask = np.array(conditions_series == label)

        ctrl_expr = gene_expr_norm[control_mask & cond_mask]
        pert_expr = gene_expr_norm[perturbed_mask & cond_mask]

        n_c = len(ctrl_expr)
        n_p = len(pert_expr)
        ctrl_ns.append(n_c)
        pert_ns.append(n_p)

        c_mean = np.mean(ctrl_expr) if n_c > 0 else 0

        if n_p == 0 or n_c == 0 or c_mean == 0:
            pert_means.append(np.nan)
            ctrl_means.append(np.nan)
            pert_sems.append(0)
            ctrl_sems.append(0)
        else:
            pert_means.append(np.mean(pert_expr) / c_mean)
            ctrl_means.append(1.0)  # control is always 100% by definition
            pert_sems.append(np.std(pert_expr) / np.sqrt(n_p) / c_mean)
            ctrl_sems.append(np.std(ctrl_expr) / np.sqrt(n_c) / c_mean)

    pert_means = np.array(pert_means)
    ctrl_means = np.array(ctrl_means)
    pert_sems = np.array(pert_sems)
    ctrl_sems = np.array(ctrl_sems)

    # Create figure or use provided axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Bar positions
    x = np.arange(n_groups)
    bar_width = 0.35

    COLOR_PERT = '#D62728'
    COLOR_CTRL = '#999999'
    ECAP = dict(linewidth=1.5, ecolor='black')

    ax.bar(x - bar_width / 2, pert_means, bar_width, yerr=pert_sems,
           capsize=4, color=COLOR_PERT, edgecolor='none', error_kw=ECAP,
           label=target_gene)
    ax.bar(x + bar_width / 2, ctrl_means, bar_width, yerr=ctrl_sems,
           capsize=4, color=COLOR_CTRL, edgecolor='none', error_kw=ECAP,
           label='Non-Targeting')

    # Reference line at 100%
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=0)

    # X-axis: condition labels with cell counts
    tick_labels = []
    for i, label in enumerate(group_labels):
        tick_labels.append(f'{label}\n[{pert_ns[i]}|{ctrl_ns[i]}]')
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # Y-axis: percentage
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(round(v * 100))}%'))
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

    ax.set_ylabel(f'{target_gene} Expression', fontsize=11, labelpad=8)

    # Title with overall effect
    all_effect = (1 - pert_means[0]) * 100 if not np.isnan(pert_means[0]) else np.nan
    if np.isnan(all_effect):
        title_str = f'CRISPRi knockdown: {target_gene}\noverall effect N/A'
    else:
        title_str = f'CRISPRi knockdown: {target_gene}\n>{int(round(all_effect))}% overall effect'
    ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)

    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)

    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    if standalone:
        plt.tight_layout()

        if save_path and save_name:
            ext = 'pdf' if PDF else 'svg'
            full_path = f"{save_path}/{_safe_name(save_name)}.{ext}"
            fig.savefig(full_path, format=ext, bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    return ax


def create_comprehensive_plot(
    mdata,
    perturb_path_base=None,
    ensembl_to_symbol_file=None,
    Target_Gene=None,
    gene_loading_corr_matrix=None,
    perturb_corr_by_sample=None,
    top_n_programs=10,
    dotplot_groupby='sample',
    perturb_target_col="target_name",
    perturb_program_col="program_name",
    perturb_log2fc_col='log2FC',
    top_corr_genes=5,
    volcano_log2fc_min=-0.05,
    volcano_log2fc_max=0.05,
    significance_threshold=0.05,
    save_path=None,
    save_name=None,
    figsize=None,
    sample=None,
    square_plots=True,
    show=True,
    PDF=True,
    umap_dot_size = None,
    umap_subsample_frac=None,
    gene_name_key='gene_names',
    control_target_name='non-targeting',
    data_key='rna',
    prog_key='cNMF',
    guide_targets_key='guide_targets',
    guide_assignment_key='guide_assignment'
):
    """Create a comprehensive multi-panel figure for one perturbed gene.

    Layout:
        Row 0 (4 square plots): UMAP Expression | UMAP Perturbation | Gene Expression Dotplot | Top Program
        Row 1 (2 plots): KD Grouped Bar Chart | Gene-loading Correlation
        Rows 2..n: one row per sample with Log2FC | Volcano | Program Dotplot | Waterfall

    Rows 2..n are only produced when ``perturb_path_base`` is given. With
    ``perturb_path_base=None`` the figure is just rows 0 and 1, which are built
    entirely from the h5mu (expression, guide assignment, program loadings) and
    need no perturbation-association results. This makes the report usable
    before Stage 2b calibration (CRT / U-test) has been run.

    Parameters
    ----------
    mdata : muon.MuData
        MuData object with 'rna' and 'cNMF' modalities.
    perturb_path_base : str or None
        Base path for perturbation result files. Sample suffix is appended as
        ``f"{perturb_path_base}_{sample}.txt"`` for each sample. If None, the
        per-condition perturbation rows (Log2FC, Volcano, Program Dotplot,
        Waterfall) are skipped entirely and no perturbation file is read.
    ensembl_to_symbol_file : str or None
        Path to Ensembl-to-symbol TSV for gene name conversion. Propagated to
        ``plot_umap_per_gene``, ``plot_top_program_per_gene``, ``perturbed_gene_dotplot``.
    Target_Gene : str
        Gene symbol to analyze. Required.
    gene_loading_corr_matrix : pandas.DataFrame
        Precomputed gene × gene correlation matrix from
        ``compute_gene_correlation_matrix``. Passed to ``analyze_correlations``.
        Required.
    perturb_corr_by_sample : dict[str, pandas.DataFrame] or None
        Dict mapping sample names to precomputed target-gene × target-gene
        correlation matrices from ``compute_gene_waterfall_cor``. Passed to
        ``create_gene_correlation_waterfall``. Required when
        ``perturb_path_base`` is given; ignored when it is None.
    top_n_programs : int
        Number of top programs to show. Passed to ``plot_top_program_per_gene``.
    dotplot_groupby : str
        Column in obs to group cells by (e.g. 'sample'). Passed to
        ``perturbed_gene_dotplot`` and ``programs_dotplot``.
    perturb_target_col : str
        Column for target gene names in perturbation results. Passed to
        ``plot_log2FC`` and ``plot_volcano``.
    perturb_program_col : str
        Column for program names in perturbation results. Passed to
        ``plot_log2FC`` and ``plot_volcano``.
    perturb_log2fc_col : str
        Column for log2FC values. Passed to ``plot_log2FC`` and ``plot_volcano``.
    top_corr_genes : int
        Number of top correlated genes to show. Passed to
        ``analyze_correlations`` and ``create_gene_correlation_waterfall``.
    volcano_log2fc_min : float
        Lower log2FC threshold for volcano plot. Passed to ``plot_volcano``.
    volcano_log2fc_max : float
        Upper log2FC threshold for volcano plot. Passed to ``plot_volcano``.
    significance_threshold : float
        Adjusted p-value cutoff. Passed to ``plot_log2FC`` and ``plot_volcano``.
    save_path : str or None
        Directory to save the output figure.
    save_name : str or None
        File stem for the saved figure (PDF or SVG).
    figsize : tuple of float or None
        (width, height) in inches. When ``square_plots`` is True only the width
        is used and the height is auto-scaled to the number of condition rows
        (``num_rows * 8``). When ``square_plots`` is False the tuple is used
        verbatim; if None, height is auto-scaled with a default width of 35.
    sample : list of str or None
        Sample/day identifiers. Defaults to ``['D0', 'sample_D1', 'sample_D2', 'sample_D3']``.
        Only used when ``perturb_path_base`` is given.
    square_plots : bool
        If True, auto-scale the figure height to the number of condition rows so
        each panel stays roughly square regardless of how many conditions there are.
    show : bool
        If True, display the figure interactively.
    PDF : bool
        If True, save as PDF; otherwise save as SVG.
    umap_dot_size : float or None
        Dot size for UMAP plots. Passed to ``plot_umap_per_gene`` and
        ``plot_umap_per_gene_guide``.
    umap_subsample_frac : float or None
        Fraction of cells to subsample for UMAP plots. Passed to
        ``plot_umap_per_gene`` and ``plot_umap_per_gene_guide``.
    gene_name_key : str or None
        Column in ``mdata['rna'].var`` for gene symbol lookup. Passed to
        ``plot_perturbation_vs_control``.
    control_target_name : str
        Name of the non-targeting control in guide_targets (e.g. 'non-targeting', 'CTRL').
        Passed to ``plot_perturbation_vs_control``.
    """
    
    # These carry defaults only so perturb_path_base can be optional (a
    # non-default parameter cannot follow a defaulted one); they are still required.
    if Target_Gene is None:
        raise ValueError("Target_Gene is required.")
    if gene_loading_corr_matrix is None:
        raise ValueError(
            "gene_loading_corr_matrix is required; build it with compute_gene_correlation_matrix()."
        )

    # Perturbation-derived rows are optional: without the per-sample association
    # files there is nothing to plot for Log2FC / Volcano / Program Dotplot / Waterfall.
    plot_perturbation = perturb_path_base is not None
    if plot_perturbation and perturb_corr_by_sample is None:
        raise ValueError(
            "perturb_corr_by_sample is required when perturb_path_base is given; "
            "build it with compute_gene_waterfall_cor() per sample."
        )

    # Set default samples if not provided
    if sample is None:
        sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']

    # Calculate figure dimensions
    num_samples = len(sample) if plot_perturbation else 0

    # Set matplotlib backend to non-interactive to reduce memory usage
    plt.ioff()

    # Close any existing figures to prevent memory leaks
    plt.close('all')

    # Create figure with custom gridspec
    # 2 header rows + n sample rows (0 sample rows when no perturbation files)
    num_rows = 2 + num_samples

    # Auto-scale the figure height to the number of condition rows so panels stay
    # roughly square no matter how many conditions are plotted. A fixed figsize
    # squishes the per-sample rows once there are more than a few conditions.
    # Width is taken from the provided figsize (default 35); height grows per row.
    PER_ROW_HEIGHT_IN = 7
    if square_plots:
        width = figsize[0] if figsize is not None else 35
        figsize = (width, num_rows * PER_ROW_HEIGHT_IN)
    elif figsize is None:
        figsize = (35, num_rows * PER_ROW_HEIGHT_IN)

    fig = plt.figure(figsize=figsize)

    # Create gridspec with dynamic number of rows using 27 columns for flexibility
    gs = gridspec.GridSpec(num_rows, 27, figure=fig,
                        hspace=0.4,  # Vertical spacing between rows
                        wspace=0.5,  # Horizontal spacing between columns
                        left=0.10,   # Left margin
                        right=0.90)  # Right margin

    # Row 0: 4 square plots — UMAP Expression | UMAP Perturbation | Gene Dotplot | Top Program
    ax_umap_expr = fig.add_subplot(gs[0, 1:6])       # UMAP Expression
    ax_umap_pert = fig.add_subplot(gs[0, 7:12])      # UMAP Perturbation
    ax_dotplot   = fig.add_subplot(gs[0, 13:18])     # Gene Expression Dotplot
    ax_top_prog  = fig.add_subplot(gs[0, 19:24])     # Top Program

    # Row 1: KD grouped bar chart | Correlation analysis
    ax_kd_grouped = fig.add_subplot(gs[1, 1:18])     # KD grouped bar chart
    ax_corr    = fig.add_subplot(gs[1, 19:24])        # Correlation analysis

    # Initialize list with header axes
    axes = [ax_umap_expr, ax_umap_pert, ax_top_prog, ax_dotplot, ax_corr]

    # Dynamically create axes for each sample
    # Each sample row has 4 plots: Log2FC, Volcano, Dotplot, Waterfall
    for sample_idx in range(num_samples):
        row_idx = sample_idx + 2  # Start from row 2 (after 2 header rows)

        ax_log2fc = fig.add_subplot(gs[row_idx, 1:6])    # Log2FC
        ax_volcano = fig.add_subplot(gs[row_idx, 7:12])  # Volcano
        ax_s_dotplot = fig.add_subplot(gs[row_idx, 13:18]) # Dotplot
        ax_waterfall = fig.add_subplot(gs[row_idx, 20:25]) # Waterfall

        axes.extend([ax_log2fc, ax_volcano, ax_s_dotplot, ax_waterfall])

    # Row 0, Plot 1: UMAP Expression
    ax_umap_expr = plot_umap_per_gene(
        mdata=mdata,
        ensembl_to_symbol_file=ensembl_to_symbol_file,
        Target_Gene=Target_Gene,
        figsize=(4, 3),
        size=umap_dot_size,
        ax=ax_umap_expr,
        umap_subsample_frac=umap_subsample_frac,
        gene_name_key=gene_name_key,
        data_key=data_key
    )
    ax_umap_expr.set_title(f"{Target_Gene} Expression", fontsize=18, fontweight='bold', loc='center')
    ax_umap_expr.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax_umap_expr.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')

    # Row 0, Plot 2: UMAP Perturbation
    ax_umap_pert = plot_umap_per_gene_guide(
        mdata=mdata,
        Target_Gene=Target_Gene,
        figsize=(4, 3),
        size=umap_dot_size,
        ax=ax_umap_pert,
        umap_subsample_frac=umap_subsample_frac,
        data_key=data_key,
        prog_key=prog_key,
        guide_targets_key=guide_targets_key,
        guide_assignment_key=guide_assignment_key
    )
    ax_umap_pert.set_title(f"{Target_Gene} Perturbation", fontsize=18, fontweight='bold', loc='center')
    ax_umap_pert.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax_umap_pert.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')

    # Row 0, Plot 3: Gene expression dotplot
    ax_dotplot = perturbed_gene_dotplot(
        mdata=mdata,
        ensembl_to_symbol_file=ensembl_to_symbol_file,
        Target_Gene=Target_Gene,
        dotplot_groupby=dotplot_groupby,
        figsize=(3, 2),
        ax=ax_dotplot,
        gene_name_key=gene_name_key,
        data_key=data_key
    )
    ax_dotplot.set_title(f"{Target_Gene} Expression", fontsize=20, fontweight='bold', loc='center')
    ax_dotplot.set_ylabel('Gene', fontsize=14, fontweight='bold', loc='center')
    ax_dotplot.set_xlabel('Day', fontsize=14, fontweight='bold', loc='center')

    # Row 0, Plot 4: Top program
    ax_top_prog = plot_top_program_per_gene(
        mdata=mdata,
        ensembl_to_symbol_file=ensembl_to_symbol_file,
        Target_Gene=Target_Gene,
        top_n_programs=top_n_programs,
        ax=ax_top_prog,
        figsize=(2, 4),
        gene_name_key=gene_name_key,
        data_key=data_key,
        prog_key=prog_key
    )
    ax_top_prog.set_title(f"Top Loading Program for {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
    ax_top_prog.set_xlabel('Gene Loading Score (z-score)', fontsize=14, fontweight='bold')
    ax_top_prog.set_ylabel(f'Program Name', fontsize=14, fontweight='bold', loc='center')

    # Row 1, Plot 1: KD grouped bar chart (All + per-condition on one axis)
    plot_perturbation_vs_control_by_condition(
        mdata=mdata,
        target_gene=Target_Gene,
        condition_key=dotplot_groupby,
        gene_name_key=gene_name_key,
        control_target_name=control_target_name,
        ax=ax_kd_grouped,
        data_key=data_key,
        prog_key=prog_key,
        guide_targets_key=guide_targets_key,
        guide_assignment_key=guide_assignment_key,
    )

    # Row 1, Plot 2: Correlation analysis
    ax_corr = analyze_correlations(
        gene_loading_corr_matrix=gene_loading_corr_matrix,
        Target=Target_Gene,
        top_corr_genes=top_corr_genes,
        figsize=(4, 3),
        ax=ax_corr
    )
    ax_corr.set_title(f"Gene Loading Correlation for {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
    ax_corr.set_ylabel('Gene Name', fontsize=14, fontweight='bold', loc='center')
    ax_corr.set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold', loc='center')

    # Loop through samples and create 4 plots per sample (Log2FC, Volcano, Dotplot, Waterfall)
    ax_index = 5  # Start after header axes (0: umap_expr, 1: umap_pert, 2: top_prog, 3: dotplot, 4: corr)

    for idx, samp in enumerate(sample if plot_perturbation else []):
        file_name = f"{perturb_path_base}_{samp}.txt"
        perturb_df = pd.read_csv(file_name, sep="\t")  # read once per sample

        # Plot 1: Log2FC plot
        current_ax = axes[ax_index]
        current_ax, df = plot_log2FC(
            perturb_path=file_name,
            Target=Target_Gene,
            perturb_target_col=perturb_target_col,
            perturb_program_col=perturb_program_col,
            perturb_log2fc_col=perturb_log2fc_col,
            significance_threshold=significance_threshold,
            figsize=(4, 3),
            ax=current_ax,
            Day=samp,
            perturb_df=perturb_df
        )
        ax_index += 1
        current_ax.set_xlabel('Effect on Program Expression', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel("Program Name", fontsize=14, fontweight='bold', loc='center')
        current_ax.set_title(f"Program Log2 Fold-Change, {samp} \n {Target_Gene}", fontsize=20, fontweight='bold', loc='center')

        # Plot 2: Volcano plot
        current_ax = axes[ax_index]
        current_ax, program, text = plot_volcano(
            perturb_path=file_name,
            Target=Target_Gene,
            perturb_log2fc_col=perturb_log2fc_col,
            volcano_log2fc_min=volcano_log2fc_min,
            volcano_log2fc_max=volcano_log2fc_max,
            significance_threshold=significance_threshold,
            figsize=(5, 3),
            ax=current_ax,
            Day=samp,
            perturb_df=perturb_df,
            run_adjust_text=False
        )
        ax_index += 1
        current_ax.set_title(f"Volcano Plot, {samp} \n {Target_Gene}", fontsize=20, fontweight='bold', loc='center')
        current_ax.set_xlabel('Effect on Program Expression', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel("Adjusted p-value, -log10", fontsize=14, fontweight='bold', loc='center')
        for t in text:
            t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 

        # Plot 3: Programs dotplot
        current_ax = axes[ax_index]
        current_ax = programs_dotplot(
            mdata=mdata,
            program_list=(df[perturb_program_col].tolist()
                          if perturb_program_col in df.columns else []),
            Target=Target_Gene,
            dotplot_groupby=dotplot_groupby,
            figsize=(5, 3),
            ax=current_ax,
            Day=samp,
            prog_key=prog_key
        )
        ax_index += 1
        current_ax.set_title(f"Perturbed Program Loading Scores, {samp} \n {Target_Gene}", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel('Program Name', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel("Day", fontsize=14, fontweight='bold', loc='center')
        
        # Plot 4: Waterfall plot 
        current_ax = axes[ax_index]
        current_ax, text = create_gene_correlation_waterfall(
            corr_matrix=perturb_corr_by_sample[samp],
            Target_Gene=Target_Gene,
            ax=current_ax,
            Day=samp,
            figsize=(3, 4),
            run_adjust_text=False
        )
        ax_index += 1
        current_ax.set_title(f"Gene Perturbation Correlation, {samp} \n {Target_Gene}", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel(f'Correlation of Program\nExpression with\n{Target_Gene} Perturbation', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel('Perturbed Genes', fontsize=14, fontweight='bold', loc='center')

        for t in text:
            t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 

    # Add main title
    fig.suptitle(f'Comprehensive Analysis: {Target_Gene}', fontweight='bold', y=0.995, fontsize=30)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path and save_name and PDF:
        full_path = f"{save_path}/{_safe_name(save_name)}.pdf"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)

    if save_path and save_name and not PDF:
        full_path = f"{save_path}/{_safe_name(save_name)}.svg"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot
    if show:
        plt.show()
    
    # Always close the figure to prevent memory leak
    plt.close(fig)



def process_single_gene(Target_Gene,
                        mdata,
                        perturb_path_base,
                        ensembl_to_symbol_file,
                        gene_loading_corr_matrix,
                        perturb_corr_by_sample,
                        top_n_programs=10,
                        dotplot_groupby='sample',
                        perturb_target_col="target_name",
                        perturb_program_col="program_name",
                        perturb_log2fc_col='log2FC',
                        top_corr_genes=5,
                        volcano_log2fc_min=-0.05,
                        volcano_log2fc_max=0.05,
                        significance_threshold=0.05,
                        save_path=None,
                        figsize=None,
                        sample=None,
                        square_plots=True,
                        show=True,
                        PDF=True,
                        gene_name_key='gene_names',
                        umap_dot_size=None,
                        umap_subsample_frac=None,
                        control_target_name='non-targeting',
                        data_key='rna',
                        prog_key='cNMF',
                        guide_targets_key='guide_targets',
                        guide_assignment_key='guide_assignment'):
    """Wrapper around ``create_comprehensive_plot`` for a single gene with error handling.

    Catches exceptions so that one failing gene does not abort a batch run.
    All parameters are forwarded to ``create_comprehensive_plot``; see its
    docstring for details.

    Parameters
    ----------
    Target_Gene : str
        Gene symbol to process. Used as ``save_name`` for the output file.

    Returns
    -------
    str
        ``"Success: <gene>"`` on success, or ``"Error: <gene> - <message>"``
        on failure.
    """

    try:
        print(f"Processing gene: {Target_Gene}")

        create_comprehensive_plot(
            mdata=mdata,
            perturb_path_base=perturb_path_base,
            Target_Gene=Target_Gene,
            ensembl_to_symbol_file=ensembl_to_symbol_file,
            gene_loading_corr_matrix=gene_loading_corr_matrix,
            perturb_corr_by_sample=perturb_corr_by_sample,
            top_n_programs=top_n_programs,
            dotplot_groupby=dotplot_groupby,
            perturb_target_col=perturb_target_col,
            perturb_program_col=perturb_program_col,
            perturb_log2fc_col=perturb_log2fc_col,
            top_corr_genes=top_corr_genes,
            volcano_log2fc_min=volcano_log2fc_min,
            volcano_log2fc_max=volcano_log2fc_max,
            significance_threshold=significance_threshold,
            save_path=save_path,
            save_name=Target_Gene,
            figsize=figsize,
            sample=sample,
            square_plots=square_plots,
            show=show,
            PDF=PDF,
            gene_name_key=gene_name_key,
            umap_dot_size=umap_dot_size,
            umap_subsample_frac=umap_subsample_frac,
            control_target_name=control_target_name,
            data_key=data_key,
            prog_key=prog_key,
            guide_targets_key=guide_targets_key,
            guide_assignment_key=guide_assignment_key
        )

        return f"Success: {Target_Gene}"

    except Exception as e:
        print(f"Error processing {Target_Gene}: {str(e)}")
        return f"Error: {Target_Gene} - {str(e)}"


# Module-level dict for sharing large objects with forked worker processes.
# Populated by parallel_gene_processing before Pool creation so that
# child processes inherit via copy-on-write (fork) without pickling.
_shared_pool_data = {}

def _process_gene_worker(Target_Gene):
    """Worker function for ``multiprocessing.Pool.map``.

    Reads shared data from the module-level ``_shared_pool_data`` dict, which
    is populated by ``parallel_gene_processing`` before fork. This avoids
    pickling large objects (mdata, correlation matrices) across processes.

    Parameters
    ----------
    Target_Gene : str
        Gene symbol to process.

    Returns
    -------
    str
        Result string from ``process_single_gene``.
    """
    d = _shared_pool_data
    try:
        return process_single_gene(
            Target_Gene,
            mdata=d['mdata'],
            perturb_path_base=d['perturb_path_base'],
            ensembl_to_symbol_file=d['ensembl_to_symbol_file'],
            gene_loading_corr_matrix=d['gene_loading_corr_matrix'],
            perturb_corr_by_sample=d['perturb_corr_by_sample'],
            top_n_programs=d['top_n_programs'],
            dotplot_groupby=d['dotplot_groupby'],
            perturb_target_col=d['perturb_target_col'],
            perturb_program_col=d['perturb_program_col'],
            perturb_log2fc_col=d['perturb_log2fc_col'],
            top_corr_genes=d['top_corr_genes'],
            volcano_log2fc_min=d['volcano_log2fc_min'],
            volcano_log2fc_max=d['volcano_log2fc_max'],
            significance_threshold=d['significance_threshold'],
            save_path=d['save_path'],
            figsize=d['figsize'],
            sample=d['sample'],
            square_plots=d['square_plots'],
            show=d['show'],
            PDF=d['PDF'],
            gene_name_key=d['gene_name_key'],
            umap_dot_size=d['umap_dot_size'],
            umap_subsample_frac=d['umap_subsample_frac'],
            control_target_name=d['control_target_name'],
            data_key=d['data_key'],
            prog_key=d['prog_key'],
            guide_targets_key=d['guide_targets_key'],
            guide_assignment_key=d['guide_assignment_key'],
        )
    finally:
        # Each worker is reused for many genes (pool.map default chunking),
        # so explicit cleanup prevents matplotlib/pandas state from accumulating per worker.
        import gc
        plt.close('all')
        gc.collect()


def parallel_gene_processing(perturbed_gene_list,
                        mdata,
                        perturb_path_base,
                        ensembl_to_symbol_file,
                        gene_loading_corr_matrix,
                        perturb_corr_by_sample,
                        top_n_programs=10,
                        dotplot_groupby='sample',
                        perturb_target_col="target_name",
                        perturb_program_col="program_name",
                        perturb_log2fc_col='log2FC',
                        top_corr_genes=5,
                        volcano_log2fc_min=-0.05,
                        volcano_log2fc_max=0.05,
                        significance_threshold=0.05,
                        save_path=None,
                        figsize=None,
                        sample=None,
                        square_plots=True,
                        show=True,
                        PDF=True,
                        n_processes=-1,
                        gene_name_key='gene_names',
                        umap_dot_size=None,
                        umap_subsample_frac=None,
                        control_target_name='non-targeting',
                        data_key='rna',
                        prog_key='cNMF',
                        guide_targets_key='guide_targets',
                        guide_assignment_key='guide_assignment'):
    """Process multiple genes in parallel using fork-based multiprocessing.

    Stores all shared data (mdata, correlation matrices, etc.) in the
    module-level ``_shared_pool_data`` dict, then creates a
    ``multiprocessing.Pool``. On Linux (fork), child processes inherit the
    data via copy-on-write without pickling. Each gene is processed by
    ``_process_gene_worker`` → ``process_single_gene`` →
    ``create_comprehensive_plot``.

    Parameters
    ----------
    perturbed_gene_list : list of str
        Gene symbols to process.
    n_processes : int
        Number of worker processes. -1 uses all available CPU cores.

    All other parameters are forwarded to ``create_comprehensive_plot`` via
    ``process_single_gene``. See ``create_comprehensive_plot`` docstring for
    details.

    Returns
    -------
    list of str
        One result string per gene from ``process_single_gene``.
    """

    global _shared_pool_data

    if n_processes == -1:
        n_processes = os.cpu_count()

    # Populate module globals BEFORE fork so workers inherit via copy-on-write
    _shared_pool_data = {
        'mdata': mdata,
        'perturb_path_base': perturb_path_base,
        'ensembl_to_symbol_file': ensembl_to_symbol_file,
        'gene_loading_corr_matrix': gene_loading_corr_matrix,
        'perturb_corr_by_sample': perturb_corr_by_sample,
        'top_n_programs': top_n_programs,
        'dotplot_groupby': dotplot_groupby,
        'perturb_target_col': perturb_target_col,
        'perturb_program_col': perturb_program_col,
        'perturb_log2fc_col': perturb_log2fc_col,
        'top_corr_genes': top_corr_genes,
        'volcano_log2fc_min': volcano_log2fc_min,
        'volcano_log2fc_max': volcano_log2fc_max,
        'significance_threshold': significance_threshold,
        'save_path': save_path,
        'figsize': figsize,
        'sample': sample,
        'square_plots': square_plots,
        'show': show,
        'PDF': PDF,
        'gene_name_key': gene_name_key,
        'umap_dot_size': umap_dot_size,
        'umap_subsample_frac': umap_subsample_frac,
        'control_target_name': control_target_name,
        'data_key': data_key,
        'prog_key': prog_key,
        'guide_targets_key': guide_targets_key,
        'guide_assignment_key': guide_assignment_key,
    }

    print(f"Starting parallel processing of {len(perturbed_gene_list)} genes using {n_processes} processes...")

    # Pool uses fork on Linux — globals inherited without pickling
    with Pool(processes=n_processes) as pool:
        results = pool.map(_process_gene_worker, perturbed_gene_list)

    _shared_pool_data = {}  # clean up

    print("Parallel processing completed!")

    return results