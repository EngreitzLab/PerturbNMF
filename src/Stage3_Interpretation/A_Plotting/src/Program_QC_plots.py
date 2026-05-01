import muon as mu 
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import scanpy as sc
import anndata as ad
from pathlib import Path
import matplotlib.colors as mcolors
from torch_cnmf import cNMF
from adjustText import adjust_text
from PIL import Image
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
#plt.rcParams["axes.spines.bottom"] = False
#plt.rcParams["axes.spines.left"] = False
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable text)
plt.rcParams['ps.fonttype'] = 42   # For EPS as well


plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 16,        # Legend text size
    'legend.title_fontsize': 14,  # Legend title size
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.titlesize': 20
})


import sys
# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from .utilities import convert_adata_with_mygene, convert_with_mygene, rename_list_gene_dictionary, rename_adata_gene_dictionary
from .Program_expression_weighted_plots import plot_program_heatmap_weighted




# make UMAP of program score
def plot_umap_per_program(mdata, Target_Program, ax=None,
 color='purple', save_path=None, save_name=None, figsize=(8,6), show=False,
 subsample_frac=None, random_state=42):

    # read cell and program data
    adata_plot = mdata['cNMF'].copy()

    # Optionally subsample cells for faster plotting
    if subsample_frac is not None and 0 < subsample_frac < 1.0:
        np.random.seed(random_state)
        n_cells = adata_plot.n_obs
        n_sample = int(n_cells * subsample_frac)
        indices = np.random.choice(n_cells, size=n_sample, replace=False)
        adata_plot = adata_plot[sorted(indices)].copy()
    
    colors = ['lightgrey', color]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

    title = f"Program {Target_Program} Expression"

    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False
    
    # Plot on the provided/created axis
    plt.sca(ax)  # Set current axis to ensure scanpy uses correct figure -CC change
    sc.pl.umap(adata_plot, color=str(Target_Program), title=title, cmap=cmap, ax=ax, show=False)
    
    # Rasterize ONLY the scatter points (collections) in this axis
    for collection in ax.collections:
        collection.set_rasterized(True)
    
    ax.set_title(title, fontsize=18, fontweight='bold', loc='center')
    ax.set_xlabel('UMAP 1', fontsize=10, fontweight = 'bold')
    ax.set_ylabel('UMAP 2', fontsize=10, fontweight = 'bold')

    # Adjust layout (only in standalone mode)
    if standalone:
        plt.tight_layout()

        # Save if path provided (only in standalone mode)
        if save_path and save_name:
            fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)

        # Control whether to display the plot (only in standalone mode)
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax




# plot x top loading genes for target program 
def plot_top_gene_per_program(mdata, Target_Program, 
                              num_gene=5, ax=None, 
                              save_path=None, save_name=None, 
                              figsize=(5,8), show=False, file_to_dictionary=None):
    
    # Read cNMF gene program matrix
    X = mdata["cNMF"].varm["loadings"]
    gene_names = mdata["cNMF"].uns['var_names']

    # Rename genes
    if file_to_dictionary is None:
        df_renamed = pd.DataFrame(data=X, columns=gene_names, index=mdata["cNMF"].var_names)
    else:
        renamed_gene_list = rename_list_gene_dictionary(gene_names, file_to_dictionary)
        df_renamed = pd.DataFrame(data=X, columns=renamed_gene_list, index=mdata["cNMF"].var_names)

    # Sort top x genes for the program
    df_sorted = df_renamed.loc[str(Target_Program)].nlargest(num_gene)
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
    ax.set_title(f"Top Loading Genes for Program {Target_Program}", 
                 fontsize=12, fontweight='bold', loc='center')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index, fontsize=10)  # Keep gene names
    ax.set_xlabel('Gene Loading Score (z-score)', fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel('Gene Name', fontsize=10, fontweight='bold', loc='center')
    ax.set_yticklabels(df_sorted.index, fontsize=10, rotation=45, ha='right')

    
    # Format x-axis
    ax.set_xlim(0, max(df_sorted.values) * 1.1)
    ax.ticklabel_format(style='scientific', axis='x')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if standalone:
        plt.tight_layout()
        
        # Save if path and name provided
        if save_path and save_name:
            fig.savefig(f"{save_path}/{save_name}.svg", format='svg', 
                       bbox_inches='tight', dpi=300)
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax
    



# plot x top enriched GO term for target program
def top_GO_per_program(GO_path, Target_Program, num_term = 5, p_value_name = "Adjusted P-value",
 term_col = "Term",
 save_path = None, save_name = None,  ax=None, figsize=(8,6), show=False):

    # read txt file
    df = pd.read_csv(GO_path, sep='\t', index_col=0)
    df.index = df.index.astype(str)

    # local to a program and isolate Term
    if str(Target_Program) not in df.index:
        print(f"Program {Target_Program} not found in GO data")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No GO data to display',
               ha='center', va='center', transform=ax.transAxes)
        return ax, []

    df_program = df.loc[str(Target_Program)]

    # rename index
    df_program.index = df_program[term_col]

    # sort by the smallest p value
    df_sort = df_program[p_value_name].nsmallest(num_term)

    # -log10 tranform
    df_sort_log = -np.log10(df_sort)

    # If no axis provided, create new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Wrap long labels - break at natural points
    import textwrap
    wrapped_labels = []
    for label in df_sort_log.index:
        # Try to wrap at 40 characters, preferring breaks at spaces
        wrapped = textwrap.fill(label, width=30, break_long_words=False, break_on_hyphens=False)
        wrapped_labels.append(wrapped)

    
    # Compute bar spacing based on number of wrapped lines to avoid label overlap
    line_counts = [wrapped.count('\n') + 1 for wrapped in wrapped_labels]
    max_lines = max(line_counts)                                                           
    spacing = max(1.0, max_lines * 0.7)
    positions = [i * spacing for i in range(len(wrapped_labels))]

    # Create horizontal bar plot
    #bars = ax.barh(range(len(df_sort_log)), df_sort_log.values, color='#808080', alpha=0.8)
    bars = ax.barh(positions, df_sort_log.values, color='#808080', alpha=0.8, height=0.8 * spacing)

    # Customize the plot
    ax.set_title(f"GO enrichment for Program {Target_Program}",fontsize=14, fontweight='bold', loc='center')
    #ax.set_yticks(range(len(df_sort_log)))
    ax.set_yticks(positions)
    ax.set_yticklabels(wrapped_labels, fontsize=6, fontweight='bold')
    
    ax.set_xlabel('-log10 Adjusted P-value',fontsize=11, fontweight='bold', loc='center')

    # Format x-axis to match your reference
    ax.set_xlim(0, max(df_sort_log.values)* 1.05) # keep the plotting part smaller
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    plt.subplots_adjust(left=0.3, right=0.7, top=0.93, bottom=0.08)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Adjust layout
    plt.tight_layout()
    
    # Only save if standalone and save_path provided
    if standalone and save_path and save_name:
        fig.savefig(f"{save_path}/{save_name}.svg", format='svg', bbox_inches='tight', dpi=300)
    
    # Only show/close if standalone
    if standalone:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax,wrapped_labels
 



# helper method to compute corr for analyze_correlations
def compute_program_correlation_matrix(mdata):

    X = mdata['cNMF'].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    df =  pd.DataFrame(data=X, index=mdata['cNMF'].obs_names, columns=mdata['cNMF'].var_names)

    program_correlation = df.corr()
    program_correlation = program_correlation.fillna(0)


    return program_correlation

# find most and least simliar programs
def analyze_program_correlations(program_correlation, Target_Program , num_program = 5, 
save_path=None, save_name = None, figsize = (5, 4),show=False, ax=None):

    # check if program exists
    if str(Target_Program) not in program_correlation.columns:
        print(f"Program {Target_Program} not found in mdata")
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{save_name}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No correlation to display',
                   ha='center', va='center', transform=ax.transAxes)
            return ax

    # Get correlations with the target program
    target_correlations = program_correlation.loc[str(Target_Program)]
    target_correlations = target_correlations.drop(str(Target_Program)) # Remove self-correlation
    
    # Sort correlations
    sorted_correlations = target_correlations.sort_values(ascending=True)
    
    # Get top and bottom gene
    top = sorted_correlations[sorted_correlations > 0].head(num_program)
    bottom = sorted_correlations[sorted_correlations < 0].tail(num_program)

    # Combine for plotting
    combined_correlations = pd.concat([bottom, top])

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    
    # Create horizontal bar plot
    colors = ['blue' if x < 0 else 'red' for x in combined_correlations]
    
    bars = ax.barh(range(len(combined_correlations)), combined_correlations.values, 
                   color=colors, alpha=0.7)

    # Customize the plot
    ax.set_title( f"Programs with Correlated Loading Across Cells, \n for Program {Target_Program}", fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Program Name', fontsize=12, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    ax.set_yticks(range(len(combined_correlations)), 
               combined_correlations.index, 
               #rotation=45, 
               ha='right')

    
    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    
    # Add legend
    from matplotlib.patches import Patch
    #legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive correlation'),
    #                  Patch(facecolor='blue', alpha=0.7, label='Negative correlation')]
    #ax.legend(handles=legend_elements)# loc='lower right')
    
     # Adjust layout (only in standalone mode)

    # Add grid
    ax.grid(axis='x', alpha=0.3)

    
    # Save if path provided (only in standalone mode)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax
 



# plot violin plots for program expression on target program 
def plot_violin(mdata, Target_Program, save_path=None, save_name=None, groupby = 'sample', figsize=(3, 5), show=False, ax=None):

    # Build dataframe from cNMF loadings
    X = mdata['cNMF'].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    df = pd.DataFrame(data=X, index=mdata['cNMF'].obs_names, columns=mdata['cNMF'].var_names)

    # Create dataframe with expression and cell type
    df = pd.DataFrame({
        "expression": df[str(Target_Program)],
        "cell_type": mdata['rna'].obs[groupby].values
    })
    
    # Compute summary stats per cell_type
    summary = (
        df.groupby("cell_type")
        .agg(
            mean_expr=("expression", "mean"),
            frac_cells=("expression", lambda x: (x > x.mean()).mean())
        )
        .reset_index()
    )
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.get_figure()
        standalone = False

    # Make violin plot
    ax = sns.violinplot(
        data=df,
        x="cell_type",
        y="expression",
        inner="quartile",
        density_norm='width',
        cut=0,
        ax=ax
    )

    # Annotate mean & fraction above violins
    cell_types_order = df['cell_type'].cat.categories if hasattr(df['cell_type'], 'cat') else sorted(df['cell_type'].unique())

    for i, row in summary.iterrows():
        # Get x position based on cell type order
        x_pos = list(cell_types_order).index(row['cell_type'])
        ax.text(
            x_pos,  # Use correct position
            df["expression"].max()*0.90,  # Just above max y
            f"Mean={row['mean_expr']:.2f}\nFrac={row['frac_cells']:.2f}",
            ha="center", va="bottom", fontsize=9, color="black"
        )

    # Set labels and title
    ax.set_title(f"Program {Target_Program} Expression by Conditions", fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Program Expression', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Conditions', fontsize=10, fontweight='bold', loc='center')

    # Adjust layout (only in standalone mode)
    if standalone:
        plt.tight_layout()

        # Save if path provided (only in standalone mode)
        if save_path and save_name:
            fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)

        # Control whether to display the plot (only in standalone mode)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax
  



# plot barplot for up/down regulated genes log2FC given results from perturbation analysis, return genes in df
def plot_program_log2FC(perturb_path, Target, tagert_col_name="target_name", plot_col_name="program_name",
                log2fc_col='log2FC', num_item=5, p_value=0.05, save_path=None, save_name=None, gene_list = None,
                figsize=(5, 4), show=False, ax=None, Day =""):

    # read df
    df = pd.read_csv(perturb_path, sep="\t")
    df['program_name'] = df['program_name'].astype(str)
    Target = str(Target)


    # Check if query gene exists
    if Target not in df[tagert_col_name].values:
        print(f"{Target} not found in mdata")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
        return ax, pd.DataFrame(columns=df.columns)

    if gene_list is not None:
        expressed_gene =  set(df[plot_col_name].values).intersection(set(gene_list))
        df = df[df[plot_col_name].isin(expressed_gene)]
        

    # Sort by log2FC
    df_sorted = df.loc[df[tagert_col_name] == Target]
    df_sorted = df_sorted[df_sorted['adj_pval'] < p_value]
    df_sorted = df_sorted.sort_values(by=log2fc_col, ascending=False)
    # Get top and bottom gene
    top_df = df_sorted.head(num_item)
    bottom_df = df_sorted.tail(num_item)
    # Combine and add category
    top = top_df.copy()
    bottom = bottom_df.copy()
    # Combine data
    plot_data = pd.merge(top, bottom, how='outer')
    plot_data = plot_data.sort_values(by=log2fc_col, ascending=False)
    
    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create horizontal bar plot
    colors = ['red' if x > 0 else 'blue' for x in plot_data[log2fc_col]]
    bars = ax.barh(range(len(plot_data)), plot_data[log2fc_col], color=colors, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data[plot_col_name],fontsize=10) 


    ax.set_xlabel('Effect on Program Loadings',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel( "Regulator Name",fontsize=10, fontweight='bold', loc='center')
    ax.set_title(f"Regulator Effect on Program Loadings, \n Program {Target}, {Day}",fontsize=14, fontweight='bold', loc='center')

    # Add a vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)


    
    # Add legend
    #from matplotlib.patches import Patch
    #legend_elements = [Patch(facecolor='red', alpha=0.7, label='Upregulated'),
    #                  Patch(facecolor='blue', alpha=0.7, label='Downregulated')]
    #ax.legend(handles=legend_elements) #loc='lower right')
    
    # Adjust layout
    ax.grid(axis='x', alpha=0.3)

    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only when ax is not provided)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only when ax is not provided)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax, plot_data
 






# plot heatmap for program perturbation on different conditions
def plot_program_heatmap(perturb_path_base, Target_Program, tagert_col_name="target_name", plot_col_name="program_name", sample= ['D0', 'sample_D1', 'sample_D2', 'sample_D3'],
                log2fc_col='log2FC', p_value=0.05, save_path=None, save_name=None, groupby = 'sample',
                figsize=(5, 4), show=False, ax=None):


    perturbed_df_list = [pd.read_csv(f"{perturb_path_base}_{samp}.txt", sep="\t").assign(sample=samp) for samp in sample]
    perturbed_df = pd.concat(perturbed_df_list, ignore_index=True)

    perturbed_df['program_name'] = perturbed_df['program_name'].astype(str)

    #sort by target programs
    df_sorted = perturbed_df.loc[perturbed_df[tagert_col_name] == str(Target_Program)]

    # Get unique genes that have adj_pval > 0.05 in any sample
    genes_to_keep = df_sorted[df_sorted['adj_pval'] < p_value][plot_col_name].unique()

    # Filter df to keep only those genes
    df_filtered = df_sorted[df_sorted['target_name'].isin(genes_to_keep)]

    df_pivot = df_filtered.pivot(columns = "target_name", index = groupby, values = log2fc_col)
    df_pivot_pval = df_filtered.pivot(columns = "target_name", index = groupby, values = "adj_pval")


    # Assuming your df has 'sample' column and gene columns with log2FC values
    heatmap_data = df_pivot #.rename(index={'sample_D1': 'D1', 'sample_D2': 'D2', 'sample_D3': 'D3'})


    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()


    # Create heatmap
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'log2FC','shrink': 0.9}, ax=ax, square=True,
                vmin=-1, vmax=1)  

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.set_ylabel('Conditions',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel( "Regulator Name",fontsize=10, fontweight='bold', loc='center')
    ax.set_title(f"Heatmap of Regulator Effects on Program {Target_Program} Across Conditions",fontsize=14, fontweight='bold', loc='center')

    # Add X marks for values with adj_pval < 0.05
    # You'll need to have a separate df with p-values in the same format
    for i in range(len(heatmap_data)):
        for j in range(len(heatmap_data.columns)):
            if df_pivot_pval.iloc[i, j] < 0.05:  # pval_data should have same structure
                ax.text(j+0.5, i+0.5, '*', ha='center', va='center', 
                    color='black', fontsize=14, weight='bold')
    

    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only when ax is not provided)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only when ax is not provided)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax



# plot one volcone plot given perturbation analysis, return genes in df
def plot_program_volcano(perturb_path, Target, tagert_col_name="target_name", plot_col_name="program_name",
                 log2fc_col='log2FC', down_thred_log=-0.05, up_thred_log=0.05, p_value=0.05, save_path=None,
                 save_name=None, figsize=(5, 4), show=False, ax=None, Day ="", gene_list = None):
                 
    df = pd.read_csv(perturb_path, sep="\t")
    df['program_name'] = df['program_name'].astype(str)
    Target = str(Target)


    # Check if query gene exists
    if Target not in df[tagert_col_name].values:
        print(f"Program {Target} not found in data")
        if ax is not None:
            ax.text(0.5, 0.5, 'No data to display',
                   ha='center', va='center', transform=ax.transAxes)
        return ax, pd.DataFrame(), []

    df_program = df.loc[df[tagert_col_name] == Target]

    if gene_list is not None:
        expressed_gene =  set(df_program[plot_col_name].values).intersection(set(gene_list))
        df_program = df_program[df_program[plot_col_name].isin(expressed_gene)]
        
    
    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot all points
    ax.scatter(x=df_program[log2fc_col],
               y=df_program['adj_pval'].apply(lambda x: -np.log10(x)),
               s=10, label="Not significant", color="grey")

    # Highlight down- or up-regulated genes
    down = df_program[(df_program[log2fc_col] <= down_thred_log) &
                      (df_program['adj_pval'] <= p_value)]
    up = df_program[(df_program[log2fc_col] >= up_thred_log) &
                    (df_program['adj_pval'] <= p_value)]

    ax.scatter(x=down[log2fc_col],
               y=down['adj_pval'].apply(lambda x: -np.log10(x)),
               s=10, label="Down-regulated", color="blue")
    ax.scatter(x=up[log2fc_col],
               y=up['adj_pval'].apply(lambda x: -np.log10(x)),
               s=10, label="Up-regulated", color="red")

    # Add text labels
    texts = []
    for i, r in up.iterrows():
        texts.append(ax.text(x=r[log2fc_col], y=-np.log10(r['adj_pval']),
                            s=r[plot_col_name], fontsize=10, zorder=10))
    for i, r in down.iterrows():
        texts.append(ax.text(x=r[log2fc_col], y=-np.log10(r['adj_pval']),
                            s=r[plot_col_name], fontsize=10, zorder=10))
    
    # Adjust text to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), ax=ax)
    
    
    # Set labels and lines
    ax.set_xlabel('Effect on Program Expression',fontsize=10, fontweight='bold', loc='center')
    ax.set_ylabel("-log10 Adjusted p-value ",fontsize=10, fontweight='bold', loc='center')
    ax.axvline(down_thred_log, color="grey", linestyle="--")
    ax.axvline(up_thred_log, color="grey", linestyle="--")
    ax.axhline(-np.log10(p_value), color="grey", linestyle="--")
    ax.legend()
    ax.set_title(f"Volcano Plot for Program Expression, \n Program {Target}, {Day}",fontsize=14, fontweight='bold', loc='center')

    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only when ax is not provided)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only when ax is not provided)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
        
    return ax, pd.merge(up, down, how='outer'), texts  



# given mdata, list of programs to plot, plot dotplot for programs, split by days
def perturbed_program_dotplot(mdata, Target_Program, groupby="sample", gene_list=None,
                     save_path=None, save_name=None, figsize=(5, 4), show=False, ax=None, Day=""): 
    

    adata_gene_list = mdata['rna'].var_names.tolist()
    gene_list = [gene for gene in gene_list if gene in adata_gene_list] if gene_list else []

    # check when there is gene that perturbed this program
    if not gene_list:

        print("No significant Gene or Gene not Found")

        # Handle empty gene list
        if ax is None:
            blank_img = np.ones((300, 200, 3), dtype=np.uint8) * 255
            img = Image.fromarray(blank_img)
            if save_path and save_name:
                full_path = f"{save_path}/{save_name}.png"
                img.save(full_path)
            return None
        else:
            ax.text(0.5, 0.5, 'No Genes to display',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{Target_Program} Perturbed Program Loading Scores {Day}", 
                        fontweight='bold', loc='center')
            return ax

    # check if the regulators are in the adata to plot
    if not set(gene_list).issubset(set(adata_gene_list)):
        missing = set(gene_list) - set(adata_gene_list)
        print(f"The following regulators are not in adata: {missing}")
    
    
    gene_list = gene_list[::-1]
    
    # Create dotplot
    if ax is None:
        # Standalone mode - let scanpy create its own figure
        dp = sc.pl.dotplot(mdata['rna'] , gene_list, groupby=groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True)
        dp.make_figure()
        fig = dp.fig
        ax = dp.ax_dict['mainplot_ax']
    else:
        # Gridspec mode - use provided ax
        fig = ax.get_figure()
        dp = sc.pl.dotplot(mdata['rna'] , gene_list, groupby=groupby, swap_axes=True, figsize=figsize,
                          dendrogram=False, show=False, return_fig=True, ax=ax)
        dp.make_figure()
    
    ax.set_title(f"Regulator Expression \n Program {Target_Program}, {Day}", 
                fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel('Regulator Name', fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel("Conditions", fontsize=10, fontweight='bold', loc='center')
    
    # Get labels and set ticks properly
    label = list(mdata['rna'].obs[groupby].cat.categories)
    
    # Set both ticks and labels together
    tick_positions = range(len(label))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(label, fontsize=8)

    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only when ax is not provided)
    if save_path and save_name and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    return ax





# helper method for computing waterfall corr
def compute_program_waterfall_cor(perturb_path, precomputed_path=None, save_path=None, log2fc_col='log2FC'):

    # Check if a pre-computed correlation matrix exists
    if precomputed_path is not None and Path(precomputed_path).exists():
        corr_matrix = pd.read_csv(precomputed_path, sep='\t', index_col=0)
        return corr_matrix

    df = pd.read_csv(perturb_path, sep='\t', index_col=0)
    df['program_name'] = df['program_name'].astype(str)


    # Pre-process: create matrix with genes as rows, programs as columns
    pivot_df = df.pivot_table(index='program_name', columns='target_name', values=log2fc_col)

    # Compute correlation matrix using numpy - much faster
    corr_matrix = pivot_df.T.corr()
    np.fill_diagonal(corr_matrix.values, np.nan)

    if save_path is not None:
        corr_matrix.index = corr_matrix.index.astype(str)
        corr_matrix.columns = corr_matrix.columns.astype(str)
        corr_matrix.to_csv(save_path, sep='\t')

    return corr_matrix




# plot the waterfall plot for genes that have simliar program loading scores when perturbed 
def create_program_correlation_waterfall(corr_matrix, Target_Program, top_num=5, save_path=None, 
                         save_name=None, figsize=(3, 5), show=False, ax=None, Day =""):

    
    # Convert to DataFrame and sort
    corr_matrix.index = corr_matrix.index.astype(str) # convert type
    gene_correlations = corr_matrix.loc[str(Target_Program)].dropna()
    corr_df = (gene_correlations).sort_values(ascending = False) 
    
    # Get top N positive and bottom N negative correlations for labeling
    top_positive = corr_df.head(top_num)
    top_negative = corr_df.tail(top_num)
    genes_to_label = set(top_positive.index.tolist() + top_negative.index.tolist())
    
    # Use ALL correlations for plotting
    plot_data = corr_df 

    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    
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
            
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax, fontsize=12) 
    
    ax.set_title(f"Programs Regulated by Similiar Regulators \n Program {Target_Program}, {Day}",fontsize=14, fontweight='bold', loc='center')
    ax.set_ylabel(f'Correlation of Perturbed Genes\n Effect on Program {Target_Program}',fontsize=10, fontweight='bold', loc='center')
    ax.set_xlabel('Program Name',fontsize=10, fontweight='bold', loc='center')

    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout (only in standalone mode)
    if ax is None:
        plt.tight_layout()
    
    # Save if path provided (only in standalone mode)
    if save_path and ax is None:
        fig.savefig(f'{save_path}/{save_name}.svg', format='svg', bbox_inches='tight', dpi=300)
    

    # Control whether to display the plot (only in standalone mode)
    if ax is None:
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    
    return ax, texts



# plot program PDF 
def create_comprehensive_program_plot(
    mdata,
    perturb_path_base,
    GO_path,
    file_to_dictionary,
    Target_Program,
    program_correlation,
    waterfall_correlation,
    top_program=5,
    groupby='sample',
    tagert_col_name="program_name",
    plot_col_name="target_name",
    log2fc_col='log2FC',
    top_enrichned_term=10,
    down_thred_log=-0.05,
    up_thred_log=0.05,
    p_value=0.05,
    save_path=None,
    save_name=None,
    figsize=None,
    sample=None,
    square_plots=True,
    show=True,
    PDF=True,
    gene_list = None,
    subsample_frac=None,
    go_p_value_name="Adjusted P-value",
    go_term_col="Term"
):

    
    # Set default samples if not provided
    if sample is None:
        sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']
    
    # Calculate figure dimensions
    num_samples = len(sample)
    
    # Set matplotlib backend to non-interactive to reduce memory usage
    plt.ioff()
    
    # Close any existing figures to prevent memory leaks
    plt.close('all')
    
    # Create figure with custom gridspec
    # 1 header row + n sample rows + heatmap 
    num_rows = 1 + num_samples + 1
    fig = plt.figure(figsize=figsize)
    
    # Create gridspec with dynamic number of rows using 27 columns for flexibility
    gs = gridspec.GridSpec(num_rows, 27, figure=fig, 
                        hspace=0.4,  # Vertical spacing between rows
                        wspace=0.5,  # Horizontal spacing between columns
                        left=0.10,   # Left margin
                        right=0.90)  # Right margin
    
    # First row: 5 header plots
    ax1 = fig.add_subplot(gs[0, 0:5])    # UMAP Expression
    ax21 = fig.add_subplot(gs[0, 6:11])   # Top program
    ax3 = fig.add_subplot(gs[0, 12:16])  # Perturbed gene dotplot
    ax4 = fig.add_subplot(gs[0, 18:21])  # Correlation analysis
    ax2 = fig.add_subplot(gs[0, 22:27])  # UMAP Perturbation
    ax22 = fig.add_subplot(gs[num_rows-1, 2:26]) # heatmap plot
    
    # Initialize list with header axes
    axes = [ax1, ax2, ax3, ax4, ax21]
    
    # Dynamically create axes for each sample
    # Each sample row has 4 plots: Log2FC, Volcano, Dotplot, Waterfall
    for sample_idx in range(num_samples):
        row_idx = sample_idx + 1  # Start from row 1 (after header)
        
        ax_log2fc = fig.add_subplot(gs[row_idx, 1:6])    # Log2FC
        ax_volcano = fig.add_subplot(gs[row_idx, 7:12])  # Volcano
        ax_dotplot = fig.add_subplot(gs[row_idx, 13:18]) # Dotplot
        ax_waterfall = fig.add_subplot(gs[row_idx, 20:25]) # Waterfall
        
        axes.extend([ax_log2fc, ax_volcano, ax_dotplot, ax_waterfall])

    # Plot 1: UMAP
    ax1 = plot_umap_per_program(
        mdata=mdata,
        Target_Program=Target_Program,
        color='purple',
        figsize=(4, 3),
        ax=ax1,
        subsample_frac=subsample_frac
    )
    ax1.set_title(f"Program {Target_Program} Expression", fontsize=18, fontweight='bold', loc='center')
    ax1.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
    ax1.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    
    # Plot 21: violin 
    ax21 = plot_violin(
        mdata = mdata,
        Target_Program = Target_Program, 
        groupby = groupby,
        figsize=(5,3),  
        ax=ax21
        )
    ax21.set_title(f"Program {Target_Program} Expression by Conditions", fontsize=18, fontweight='bold', loc='center')
    ax21.set_ylabel('Program Expression', fontsize=14, fontweight='bold', loc='center')
    ax21.set_xlabel('Conditions', fontsize=14, fontweight='bold', loc='center')
    
    # Plot 2: Correlation analysis
    ax2 =analyze_program_correlations(
        program_correlation = program_correlation, 
        Target_Program = Target_Program, 
        num_program = top_program, 
        figsize = (4, 4),
        ax=ax2
        )
    ax2.set_title(f"Programs with Correlated Loading \nAcross Cells for Program {Target_Program}", fontsize=20, fontweight='bold', loc='center')
    ax2.set_ylabel('Program Name', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'Correlation Coefficient', fontsize=14, fontweight='bold', loc='center')

    
    # Plot 3: Top Gene for program
    ax3 = plot_top_gene_per_program(
        mdata=mdata, 
        Target_Program = Target_Program, 
        num_gene = top_enrichned_term,  
        figsize=(4, 4),
        file_to_dictionary=file_to_dictionary,
        ax=ax3)  
    ax3.set_title(f"Top Loading Genes for Program {Target_Program}", fontsize=20, fontweight='bold', loc='center')
    ax3.set_xlabel('Gene Loading Score (z-score)', fontsize=14, fontweight='bold', loc='center')
    ax3.set_ylabel('Gene Name', fontsize=14, fontweight='bold', loc='center')
    
    
     
    # Plot 4: GO enrichment
    ax4,wrapped_labels=top_GO_per_program(
        GO_path=GO_path,
        Target_Program = Target_Program,
        num_term = top_enrichned_term,
        p_value_name = go_p_value_name,
        term_col = go_term_col,
        figsize=(6,6),
        ax=ax4)
    ax4.set_title(f"GO enrichment for Program {Target_Program}",fontsize=18, fontweight='bold', loc='center')
    ax4.set_yticklabels(wrapped_labels, fontsize=8, fontweight='bold')
    ax4.set_xlabel('-log10 Adjusted P-value',fontsize=11, fontweight='bold', loc='center')


    # Loop through samples and create 4 plots per sample (Log2FC, Volcano, Dotplot, Waterfall)
    ax_index = 5  # Start after header axes (0-5)
    
    for idx, samp in enumerate(sample):
        file_name = f"{perturb_path_base}_{samp}.txt" 

        # Plot 1: Log2FC plot
        current_ax = axes[ax_index]
        current_ax, df = plot_program_log2FC(
            perturb_path=file_name,
            Target=Target_Program,
            tagert_col_name=tagert_col_name,
            plot_col_name=plot_col_name,
            log2fc_col=log2fc_col,
            p_value=p_value,
            figsize=(4, 3),
            gene_list=gene_list,
            ax=current_ax,
            Day=samp
        )
        ax_index += 1
        current_ax.set_xlabel('Effect on Program Expression', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel("Regulator Name", fontsize=14, fontweight='bold', loc='center')
        current_ax.set_title(f"Regulator Effect on Program Expression, \n Program {Target_Program}, {samp}", fontsize=20, fontweight='bold', loc='center')


        # Plot 2: Volcano plot
        current_ax = axes[ax_index]
        current_ax, program, text = plot_program_volcano(
            perturb_path=file_name,
            Target=Target_Program,
            log2fc_col=log2fc_col,
            down_thred_log=down_thred_log,
            up_thred_log=up_thred_log,
            tagert_col_name=tagert_col_name,
            plot_col_name=plot_col_name,
            p_value=p_value,
            figsize=(5, 3),
            gene_list=gene_list,
            ax=current_ax,
            Day=samp
        )
        ax_index += 1
        current_ax.set_title(f"Volcano Plot for Program Expression, \n Program {Target_Program}, {samp}", fontsize=20, fontweight='bold', loc='center')
        current_ax.set_xlabel('Effect on Program Expression', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_ylabel(" -log10 Adjusted p-value", fontsize=14, fontweight='bold', loc='center')
        for t in text:
            t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 
        
       # Plot 3: Programs dotplot
        current_ax = axes[ax_index]
        current_ax = perturbed_program_dotplot(
            mdata=mdata, 
            gene_list=df[plot_col_name].tolist() if not df.empty else [],
            Target_Program=Target_Program,
            groupby=groupby,
            figsize=(5, 3),
            ax=current_ax,
            Day=samp
        )
        current_ax.set_title(f"Regulator Expression \n Program {Target_Program}, {samp} ", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel('Regulator Name', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel("Conditions", fontsize=14, fontweight='bold', loc='center')
        ax_index += 1

        # Plot 4: Waterfall plot 
        current_ax = axes[ax_index]
        current_ax, text = create_program_correlation_waterfall(
            corr_matrix= waterfall_correlation[samp],
            Target_Program=Target_Program,
            top_num=top_enrichned_term,
            ax=current_ax,
            Day=samp,
            figsize=(3, 4),
        )
        ax_index += 1
        current_ax.set_title(f"Programs Regulated by Similiar Regulators \n Program {Target_Program}, {samp}", fontsize=18, fontweight='bold', loc='center')
        current_ax.set_ylabel(f'Correlation of Perturbed Genes\n Effect on Program {Target_Program}', fontsize=14, fontweight='bold', loc='center')
        current_ax.set_xlabel('Program Name', fontsize=14, fontweight='bold', loc='center')
        for t in text:
           t.set_fontsize(14)
        adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=current_ax, fontsize=14) 


    # Plot 5: Heatmap with expression sidebar
    ax22 = plot_program_heatmap_weighted(
        perturb_path_base=perturb_path_base,
        mdata=mdata,
        Target_Program=Target_Program,
        tagert_col_name=tagert_col_name,
        plot_col_name=plot_col_name,
        sample=sample,
        groupby=groupby,
        log2fc_col=log2fc_col,
        p_value=p_value,
        figsize=(15, 5),
        ax=ax22)

    ax22.set_xticklabels(ax22.get_xticklabels(), fontsize=13, rotation=45, ha='right')
    ax22.set_yticklabels(ax22.get_yticklabels(), fontsize=13, rotation=0)
    ax22.set_ylabel('',fontsize=14, fontweight='bold', loc='center')
    ax22.set_xlabel('',fontsize=14, fontweight='bold', loc='center')
    ax22.set_title(f"Regulator effects on Program {Target_Program}",fontsize=18, fontweight='bold', loc='center')



    # Add main title
    fig.suptitle(f'Comprehensive Analysis: Program {Target_Program}', fontweight='bold', y=0.995, fontsize=30)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path and save_name and PDF:
        full_path = f"{save_path}/{save_name}.pdf"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
    
    if save_path and save_name and not PDF:
        full_path = f"{save_path}/{save_name}.svg"
        print(f"Saving figure to {full_path}...")
        fig.savefig(full_path, format='svg', bbox_inches='tight', dpi=300)
    
    # Control whether to display the plot
    if show:
        plt.show()
    
    # Always close the figure to prevent memory leak
    plt.close(fig)


'''working on progress
# plot top enriched enhancer or promoter
def plot_motif_per_program(path, num_term = 10,  program_num = 0, day = None, title = "Enhancer", p_value_name = "Adjusted P-value", folder_name = None, file_name = None):

    # read txt file
    df = pd.read_csv(path, sep='\t', index_col=0)

    # local to a program and isolate Term
    df_program = df.loc[df['program_name'] == program_num]

    if day is not None:
        df_program = df_program.loc[df_program['sample'] == day]

    # rename index
    df_program.index = df_program['motif']

    # sort by the smallest p value
    df_sort = df_program[p_value_name].nsmallest(num_term)

    # -log10 tranform
    df_sort_log = -np.log10(df_sort)

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 8))

    # Create horizontal bar plot
    bars = ax.barh(range(len(df_sort_log)), df_sort_log.values, color='#808080', alpha=0.8)

    # Customize the plot
    ax.set_yticks(range(len(df_sort_log)))
    ax.set_yticklabels(df_sort_log.index, fontsize=10)
    ax.set_xlabel('Adjusted P-value(-10)', fontsize=11)

    # Format x-axis to match your reference
    ax.set_xlim(0, max(df_sort_log.values))
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Add title
    if day is not None:
        ax.set_title(f"{title} at {day} and program {program_num}", fontsize=14, fontweight='bold')
    else:
         ax.set_title(f"{title} at program {program_num}", fontsize=14, fontweight='bold')



    # Adjust layout
    plt.tight_layout()    

    if folder_name and file_name:
        fig.savefig(f"{folder_name}/{file_name}.png")


# plot volcano plots by date
def plot_all_days_motif(path, num_term = 10,  program_num = 0, title = "Enhancer", p_value_name = "Adjusted P-value", folder_name = None, file_name = None):
    for samp in ['D0', 'D1', 'D2', 'D3']:
        plot_motif_per_program(path,num_term=num_term, program_num=program_num,day = samp, p_value_name ="adj_pval",  title = title,folder_name=folder_name,file_name=file_name)
'''