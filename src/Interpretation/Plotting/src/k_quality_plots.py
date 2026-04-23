import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import os
import seaborn as sns
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from torch_cnmf import cNMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import muon as mu
import anndata as ad
import scanpy as sc




# compute correlation coeffcient between two program x gene matrices
def program_corr(matrix1,matrix2):
    
    if matrix2.shape != matrix2.shape:
        print("dim is different")
        return 
    
    num_columns = matrix1.shape[0]
    column_correlations = []
    
    for i in range(num_columns):
        for j in range(num_columns):
            
            # Calculate the correlation coefficient between the i-th row of matrix1 and j-th row of matrix2
            correlation = np.corrcoef(matrix1.iloc[i,:], matrix2.iloc[j,:])[0, 1]
            column_correlations.append(correlation)
            
    df = pd.DataFrame(np.array(column_correlations).reshape(num_columns, num_columns), 
                     columns = matrix2.index,
                     index = matrix1.index)

    return df




# compute euclidea distance btween two program x gene matrices
def program_euclidean(matrix1, matrix2):
    
    if matrix1.shape != matrix2.shape:
        print("dim is different")
        return 
    
    num_rows = matrix1.shape[0]
    euclidean_distances = []
    
    for i in range(num_rows):
        for j in range(num_rows):
            
            # Calculate the Euclidean distance between the i-th row of matrix1 and j-th row of matrix2
            distance = np.sqrt(np.sum((matrix1.iloc[i,:] - matrix2.iloc[j,:])**2))
            euclidean_distances.append(distance)
            
    df = pd.DataFrame(np.array(euclidean_distances).reshape(num_rows, num_rows), 
                     columns=matrix2.index,
                     index=matrix1.index)

    return df




# compute top x overlapped genes in percentages btween two program x gene matrices
def top_genes_overlap(matrix1, matrix2, percentage = False, gene_num = 300):
    
    # check dim 
    if matrix1.shape != matrix2.shape:
        print("Different dim")
        return 
    
    # find out top x genes in each k
    top_genes_1 = matrix1.apply(lambda row: row.nlargest(gene_num).index.tolist(), axis=1)
    top_genes_2 = matrix2.apply(lambda row: row.nlargest(gene_num).index.tolist(), axis=1)

    
    n_k = (top_genes_1.shape[0])
    overlap = np.zeros((n_k, n_k), dtype=float)
    gene_shared_list = pd.Series(dtype=object)

    # generate overlap matrix 
    for i in range(n_k):
        for j in range(n_k):
            if percentage:
                s = len(set(top_genes_1.iloc[i]) & set(top_genes_2.iloc[j]))/gene_num
            else: 
                s = len(set(top_genes_1.iloc[i]) & set(top_genes_2.iloc[j])) 

            # compose a shared gene matrix
            #gene_shared = list(set(top_genes_1.iloc[i]) & set(top_genes_2.iloc[j]))
            #name = f"{top_genes_1.index[i]} VS {top_genes_2.index[j]}"
            #gene_shared_list.at[name] = gene_shared 
            
            overlap[i, j] = int(s)

    
    overlap_df = pd.DataFrame(overlap,
                              index=matrix1.index,
                              columns=matrix2.index)
    
    return overlap_df #, top_genes_1, top_genes_2, gene_shared_list




# for each row, sort the values on the columns, max value is on diagnol
def sort_corr_matrix(Matrix):

    n = Matrix.shape[0]
    Matrix_reordered = Matrix.copy()

    for i in range(n):
        # find index of row maximum
        j = Matrix.iloc[i].values.argmax()

        # move that column to diagonal position
        Matrix_reordered.iloc[i] = np.roll(Matrix.iloc[i].values, i - j)
    
    return Matrix_reordered




# Make a program dotplots by days
def programs_dotplots(k, output_dir, run_name, sel_thresh = 2.0, groupby='sample', save_name=None, save_path=None, 
figsize=(4, 30), show=False, ax=None):

    standalone_mode = ax is None


    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""
        return '{output_dir}/{run_name}/adata/cNMF_{k}_{sel_thresh}.h5mu'.format(
                                                                                output_dir=output_dir,
                                                                                run_name = run_name,
                                                                                k=k,
                                                                                sel_thresh = str(sel_thresh).replace(".", "_"))


    # read in adata
    mdata = mu.read_h5mu(get_gene_path(output_dir, run_name, k, sel_thresh))
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
def consensus_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2, 
                 sel_thresh_1=2.0, sel_thresh_2=2.0,gene_num = 300, 
                 title = "top 300 gene Clustermap", figsize = (10, 10), gene_score=True):

    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""
        if gene_score: 
            return '{output_dir}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
        else: 
            return '{output_dir}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))

    # read in as df 
    df_1 = pd.read_csv(get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1), sep="\t" , index_col = 0)
    df_2 = pd.read_csv(get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2), sep="\t" , index_col = 0)

    # perform overlap gene analysis 
    overlap = top_genes_overlap(df_1, df_2, gene_num = gene_num, percentage  = False)
    sorted_overlap = sort_corr_matrix(overlap)


    # plot sorted program 
    g = sns.clustermap(sorted_overlap, 
                row_cluster=False,   
                col_cluster=False,   
                cmap='coolwarm',      
                figsize= figsize,
                center = 0,                
                xticklabels=False, 
                yticklabels=False)      
                
    import textwrap
    wrapped_title = "\n".join(textwrap.wrap(title, width=50))  
    g.fig.suptitle(wrapped_title, fontsize=15,fontweight='bold')
    print("max value:", sorted_overlap.max().max())
   

#return cNMF run results
def return_cNMF_matrix(k, output_dir, run_name,sel_thresh=2.0):

    def get_gene_score_path(output_dir, run_name, k, sel_thresh):
            return '{output_dir}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
    def get_gene_median_path(output_dir, run_name, k, sel_thresh):
            return '{output_dir}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))

    def get_program_path(output_dir, run_name, k, sel_thresh):
          return '{output_dir}/{run_name}/{run_name}.usages.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))


    # read in as df 
    gene_median = pd.read_csv(get_gene_median_path(output_dir, run_name, k, sel_thresh), sep="\t" , index_col = 0)
    gene_score = pd.read_csv(get_gene_score_path(output_dir, run_name, k, sel_thresh), sep="\t" , index_col = 0)
    program = pd.read_csv(get_program_path(output_dir, run_name, k, sel_thresh), sep="\t" , index_col = 0)

    return gene_median, gene_score,program



# plot boxplot for each K values' max shared genes
def cNMF_boxplot(output_dir_1, output_dir_2, run_name_1, run_name_2, 
                 sel_thresh_1=2.0, sel_thresh_2=2.0, gene_num=300, 
                 components=[30, 50, 60, 80, 100, 200, 300], 
                 title="Max shared genes for each program (gene score)", gene_score = True):
    
    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""
        if gene_score: 
            return '{output_dir}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
        else: 
            return '{output_dir}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
    
    shared_genes = {}
    for k in components:
        df_1 = pd.read_csv(get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1), sep="\t", index_col=0)
        df_2 = pd.read_csv(get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2), sep="\t", index_col=0)
        
        overlap = top_genes_overlap(df_1, df_2, gene_num=gene_num, percentage=False)
        sorted_overlap = sort_corr_matrix(overlap)
        shared_genes[k] = np.diag(sorted_overlap)
    
    fig, ax = plt.subplots()
    box = ax.boxplot(list(shared_genes.values()), labels=list(shared_genes.keys()), patch_artist=True)

    # Change box color to crimson red
    for patch in box['boxes']:
        patch.set_facecolor('crimson')

    # Change median line color to green
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)
        
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Max shared genes", fontsize=12)
    ax.set_xlabel("K", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    return ax, shared_genes




# given a list of shared genes across two programs, plot bar plot
def cNMF_barplot(k, output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1=2.0, sel_thresh_2=2.0, 
gene_num = 300, title="Maxinum shared genes between sk-cd and torch-halsvar", x_label="torch-halsvar Program"
, figsize = (3,5), gene_score=True):


    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""
        if gene_score: 
            return '{output_dir}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
        else: 
            return '{output_dir}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))

    # read in as df 
    df_1 = pd.read_csv(get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1), sep="\t" , index_col = 0)
    df_2 = pd.read_csv(get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2), sep="\t" , index_col = 0)


    # find overlaps 
    overlap = top_genes_overlap(df_1, df_2, gene_num = gene_num, percentage  = False)
    sorted_overlap = sort_corr_matrix(overlap)

    # get overlapped genes 
    sort_list = np.diag(sorted_overlap)

    # Create x-axis positions (indices)
    x_positions = range(len(sort_list))
    
    # Create the bar plot
    plt.figure(figsize=figsize)
    bars = plt.bar(x_positions, sort_list, color='skyblue', alpha=0.7)
    
    # Customize the plot
    plt.title(title, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Shared Genes", fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    
    plt.tight_layout()




# plot stablity of one method vs max shared gene between two methods in scattorplot
def stability_vs_sharedgenes(k,  output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1=2.0, sel_thresh_2=2.0,
 gene_num = 300, local_neighborhood_size=0.30, density_threshold=2.0, 
 title = "Silhouette Score vs max shared genes for torch-hals-online (K=100)", gene_score = True, figsize=(8,6)):

    cnmf_obj= cNMF(output_dir=output_dir_1, name=run_name_1)
    combined = cnmf_obj.combine_nmf(k)

    # use cNMF's Kmean cluster code, get the Kmean label and stablity score for each program 
    n_neighbors = int(local_neighborhood_size * combined.shape[0]/k)
    l2_spectra = (combined.T/np.sqrt((combined**2).sum(axis=1))).T

    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

    score = silhouette_score(l2_spectra, kmeans_cluster_labels)
    sample_scores = silhouette_samples(l2_spectra, kmeans_cluster_labels)

    # get silhouette for each program 
    cluster_scores = {}
    for cluster in np.unique(kmeans_cluster_labels):
        cluster_scores[cluster] = sample_scores[kmeans_cluster_labels == cluster].mean() 
        
    values = list(cluster_scores.values())

    # calcaulte shared list
    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""
        if gene_score: 
            return '{output_dir}/{run_name}/{run_name}.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))
        else: 
            return '{output_dir}/{run_name}/{run_name}.spectra.k_{k}.dt_{sel_thresh}.consensus.txt'.format(
                                                                                        output_dir=output_dir,
                                                                                        run_name = run_name,
                                                                                        k=k,
                                                                                        sel_thresh = str(sel_thresh).replace('.','_'))

    # read in as df 
    df_1 = pd.read_csv(get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1), sep="\t" , index_col = 0)
    df_2 = pd.read_csv(get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2), sep="\t" , index_col = 0)

    # perform overlap gene analysis 
    overlap = top_genes_overlap(df_1, df_2, gene_num = gene_num, percentage  = False)
    sorted_overlap = sort_corr_matrix(overlap)
    sort_list = np.diag(sorted_overlap)

    # Scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(values, sort_list, s=10) # sorted list is from shared genes 
    plt.title(title)
    plt.ylabel("Max shared genes")
    plt.xlabel("Silhouette Score")

    plt.show()
    plt.close()




# For graphing clustermap given corr/eu/overlap matrix
def NMF_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1=2.0, sel_thresh_2=2.0, gene_num = 300,
 method = "corr", title = "NMF clustermap between sk-cd and torch-halsvar-batch", figsize = (4,4), color = True):

    cnmf_obj_1 = cNMF(output_dir=output_dir_1, name=run_name_1)
    combined_1 = cnmf_obj_1.combine_nmf(k)

    cnmf_obj_2 = cNMF(output_dir=output_dir_2, name=run_name_2)
    combined_2 = cnmf_obj_2.combine_nmf(k)


    if method == "corr":
        cluster = program_corr(combined_1,combined_2)
    elif method == "distance":
        cluster = program_euclidean(combined_1,combined_2)
    elif method == "overlap":
        cluster = top_genes_overlap(combined_1, combined_2, percentage = False, gene_num = 300)
    else:
        print(" Method dose not exist")
        return 
 

    # label color for each run 
    if color: 
        palette = sns.color_palette("husl", n_colors=cluster.index.nunique())
        lut = dict(zip(cluster.columns.unique(), palette))
        row_colors = cluster.index.map(lut)
    
        sns.set_theme(style="white")               # optional aesthetics
        g = sns.clustermap(
                cluster,
                cmap="vlag",                       # diverging palette centred at 0
                center=0,                          # keep 0 in the middle of the colour range
                metric="euclidean",                # distance metric for clustering
                figsize=figsize,                  # size in inches
                row_colors=row_colors,            # color axis 
                col_colors = row_colors,          # color axis 
                row_cluster=True,
                col_cluster=True,
                xticklabels=False, 
                yticklabels=False 
        )
    else:
        sns.set_theme(style="white")               # optional aesthetics
        g = sns.clustermap(
                cluster,
                cmap="vlag",                       # diverging palette centred at 0
                center=0,                          # keep 0 in the middle of the colour range
                metric="euclidean",                # distance metric for clustering
                figsize=figsize,                  # size in inches
                row_cluster=True,
                col_cluster=True,
                xticklabels=False, 
                yticklabels=False 
        )


    g.fig.suptitle(title)

    # Remove dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    




def kmean_cluster(k, output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1=2.0, sel_thresh_2=2.0,
 figsize = (5,5),  gene_num = 300, local_neighborhood_size =0.3, 
 title = "Top 300 Genes Clustermap for gene score", gene_score = True):

    # read in data 
    cnmf_obj_1 = cNMF(output_dir=output_dir_1, name=run_name_1)
    combined_1 = cnmf_obj_1.combine_nmf(k)

    cnmf_obj_2 = cNMF(output_dir=output_dir_2, name=run_name_2)
    combined_2 = cnmf_obj_2.combine_nmf(k)

    cluster = top_genes_overlap(combined_1, combined_2, percentage = False, gene_num = 300)


    # use cNMF's Kmean cluster code, get the Kmean label and stablity score for each program 
    n_neighbors = int(local_neighborhood_size * combined_1.shape[0]/k)
    l2_spectra = (combined_1.T/np.sqrt((combined_1**2).sum(axis=1))).T
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels_1 = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

    n_neighbors = int(local_neighborhood_size * combined_2.shape[0]/k)
    l2_spectra = (combined_2.T/np.sqrt((combined_2**2).sum(axis=1))).T
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=1)
    kmeans_model.fit(l2_spectra)
    kmeans_cluster_labels_2 = pd.Series(kmeans_model.labels_+1, index=l2_spectra.index)

    # assign cluster value
    cluster.index = kmeans_cluster_labels_1.values
    cluster.columns = kmeans_cluster_labels_2.values

    palette = sns.color_palette("husl", n_colors=cluster.index.nunique())
    lut = dict(zip(cluster.columns.unique(), palette))
    row_colors = cluster.index.map(lut)

    g = sns.clustermap(
            cluster,
            cmap="vlag",                       # diverging palette centred at 0
            linewidths=.5,                     # grid lines
            center=0,                          # keep 0 in the middle of the colour range
            metric="euclidean",                # distance metric for clustering
            figsize=figsize,                  # size in inches
            row_colors=row_colors,            # color axis 
            col_colors=row_colors,          # color axis 
            row_cluster=True,
            col_cluster=True,
            xticklabels=False,
            yticklabels=False
    )


    # Remove dendrograms
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)




def load_combined_matrix(k, path):

    # Initialize empty list to store dataframes
    dfs = []

    for samp in ["D0", "sample_D1", "sample_D2", "sample_D3"]:
        file = f"{path}/{k}/{k}_perturbation_association_results_{samp}.txt"
        df = pd.read_csv(file, sep="\t", index_col=0)
        dfs.append(df)

    # Concatenate vertically
    combined_df = pd.concat(dfs, axis=0, ignore_index=False)

    return combined_df




def compute_gene_list_perturbation(k, combined_df, pval = 0.05):

    # compute gene list 
    gene_list = {}

    for i in range(0,k):
        sorted_program = combined_df[combined_df["program_name"] == i]
        filter_sorted_program = sorted_program[sorted_program["adj_pval"] < pval]

        genes = filter_sorted_program.index.unique()
        gene_list[i] = genes.tolist()
        
    return gene_list




def compute_gene_list_GO(k, combined_df, pval = 0.05, term_col='Term', adjpval_col='Adjusted P-value'):

    # compute gene list
    GO = {}

    for i in range(0,k):
        sorted_program = combined_df[combined_df.index == i]
        filter_sorted_program = sorted_program[sorted_program[adjpval_col] < pval]
        genes = filter_sorted_program[term_col].unique()
        GO[i] = genes.tolist()

    return GO




def build_overlap_matrix(dict1, dict2):
    # Get all keys and sort them
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())
    
    # Initialize the overlap matrix (float for Jaccard index)
    overlap_matrix = np.zeros((len(keys1), len(keys2)), dtype=float)
    
    # Calculate Jaccard index
    for i, key1 in enumerate(keys1):
        for j, key2 in enumerate(keys2):
            # Convert to sets and calculate Jaccard index
            genes1 = set(dict1[key1])
            genes2 = set(dict2[key2])
            
            intersection = len(genes1 & genes2)
            union = len(genes1 | genes2)
            
            # Calculate Jaccard index (handle empty sets)
            if union == 0:
                overlap_matrix[i, j] = 0.0
            else:
                overlap_matrix[i, j] = intersection / union
    
    df_overlap = pd.DataFrame(overlap_matrix,
                               index=keys1,
                               columns=keys2)
    return df_overlap




def GO_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1 = 2.0, sel_thresh_2 = 2.0,
figsize = (5,5), pval = 0.05, title = "Shared GO torch-halsv-batch vs sk-cd (K=100)",
go_file=None, term_col='Term', adjpval_col='Adjusted P-value'):

    # Default file name pattern
    if go_file is None:
        go_file = '{k}_GO_term_enrichment.txt'

    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""

        return '{output_dir}/{run_name}/Evaluation/{k}_{sel_thresh}/{go_file}'.format(output_dir=output_dir,
                                                            run_name = run_name,
                                                            sel_thresh = str(sel_thresh).replace('.', '_'),
                                                            k=k,
                                                            go_file=go_file.format(k=k))


    # read in as df
    path_1 = get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1)
    path_2 = get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2)

    combined_df_1 = pd.read_csv(path_1, sep="\t" , index_col = 0)
    combined_df_2 = pd.read_csv(path_2, sep="\t" , index_col = 0)

    gene_list_1 = compute_gene_list_GO(k, combined_df_1, pval, term_col=term_col, adjpval_col=adjpval_col)
    gene_list_2 = compute_gene_list_GO(k, combined_df_2, pval, term_col=term_col, adjpval_col=adjpval_col)

    overlap_matrix = build_overlap_matrix(gene_list_1,gene_list_2)

    # graph clustermap on gene scores 
    sorted_overlap = sort_corr_matrix(overlap_matrix)

    # plot sorted program 
    g = sns.clustermap(sorted_overlap, 
                        row_cluster=False,   
                        col_cluster=False,   
                        cmap='coolwarm',      
                        figsize=(5, 5),
                        center = 0,                
                        xticklabels=False, 
                        yticklabels=False)      
                
    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=50))  # adjust width as needed
    g.fig.suptitle(wrapped_title,fontweight='bold')
    print("max value:", sorted_overlap.max().max())




def perturbation_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2, condition, sel_thresh_1 = 2.0, sel_thresh_2 = 2.0, 
figsize = (5,5), pval = 0.05, title = "Shared unique regulators torch-halsv-batch vs sk-cd (K=100)"):

    def get_gene_path(output_dir, run_name, k, sel_thresh, condition):
        """Helper to build path consistently"""

        return '{output_dir}/{run_name}/Evaluation/{k}_{sel_thresh}/{k}_perturbation_association_results_{condition}.txt'.format(output_dir=output_dir,
                                                            run_name = run_name,
                                                            sel_thresh = str(sel_thresh).replace('.', '_'),
                                                            k=k,
                                                            condition = condition )
                                                

    # read in as df 
    path_1 = get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1, condition)
    path_2 = get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2, condition)

    combined_df_1 = load_combined_matrix(k, path_1)
    combined_df_2 = load_combined_matrix(k, path_2)

    gene_list_1 = compute_gene_list_perturbation(k, combined_df_1, pval)
    gene_list_2 = compute_gene_list_perturbation(k, combined_df_2, pval)

    overlap_matrix = build_overlap_matrix(gene_list_1,gene_list_2)

    # graph clustermap on gene scores 
    sorted_overlap = sort_corr_matrix(overlap_matrix)

    # plot sorted program 
    g = sns.clustermap(sorted_overlap, 
                        row_cluster=False,   
                        col_cluster=False,   
                        cmap='vlag',      
                        figsize=(5, 5),
                        center = 0,                
                        xticklabels=False, 
                        yticklabels=False)      
                
    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=50))  # adjust width as needed
    g.fig.suptitle(wrapped_title,fontweight='bold')
    print("max value:", sorted_overlap.max().max())




def geneset_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2, sel_thresh_1 = 2.0, sel_thresh_2 = 2.0,
figsize = (5,5), pval = 0.05, title = "Shared genesets torch-halsv-batch vs sk-cd (K=100)",
geneset_file=None, term_col='Term', adjpval_col='Adjusted P-value'):

    # Default file name pattern
    if geneset_file is None:
        geneset_file = '{k}_geneset_enrichment.txt'

    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""

        return '{output_dir}/{run_name}/Evaluation/{k}_{sel_thresh}/{geneset_file}'.format(output_dir=output_dir,
                                                            sel_thresh = str(sel_thresh).replace('.', '_'),
                                                            run_name = run_name,
                                                            k=k,
                                                            geneset_file=geneset_file.format(k=k))


    # read in as df
    path_1 = get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1)
    path_2 = get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2)

    combined_df_1 = pd.read_csv(path_1, sep="\t" , index_col = 0)
    combined_df_2 = pd.read_csv(path_2, sep="\t" , index_col = 0)

    gene_list_1 = compute_gene_list_GO(k, combined_df_1, pval, term_col=term_col, adjpval_col=adjpval_col)
    gene_list_2 = compute_gene_list_GO(k, combined_df_2, pval, term_col=term_col, adjpval_col=adjpval_col)

    overlap_matrix = build_overlap_matrix(gene_list_1,gene_list_2)

    # graph clustermap on gene scores 
    sorted_overlap = sort_corr_matrix(overlap_matrix)

    # plot sorted program 
    g = sns.clustermap(sorted_overlap, 
                        row_cluster=False,   
                        col_cluster=False,   
                        cmap='coolwarm',      
                        figsize=(5, 5),
                        center = 0,                
                        xticklabels=False, 
                        yticklabels=False)      
                
    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=50))  # adjust width as needed
    g.fig.suptitle(wrapped_title,fontweight='bold')
    print("max value:", sorted_overlap.max().max())



def trait_clustermap(k, output_dir_1, output_dir_2, run_name_1, run_name_2,
figsize = (5,5), pval = 0.05, title = "Shared traits torch-halsv-batch vs sk-cd (K=100)",
sel_thresh_1 = 2.0, sel_thresh_2 = 2.0, trait_file=None, term_col='Term', adjpval_col='Adjusted P-value'):

    # Default file name pattern
    if trait_file is None:
        trait_file = '{k}_trait_enrichment.txt'

    def get_gene_path(output_dir, run_name, k, sel_thresh):
        """Helper to build path consistently"""

        return '{output_dir}/{run_name}/Evaluation/{k}_{sel_thresh}/{trait_file}'.format(output_dir=output_dir,
                                                            run_name = run_name,
                                                            k=k,
                                                            sel_thresh = str(sel_thresh).replace('.', '_'),
                                                            trait_file=trait_file.format(k=k))


    # read in as df
    path_1 = get_gene_path(output_dir_1, run_name_1, k, sel_thresh_1)
    path_2 = get_gene_path(output_dir_2, run_name_2, k, sel_thresh_2)

    combined_df_1 = pd.read_csv(path_1, sep="\t" , index_col = 0)
    combined_df_2 = pd.read_csv(path_2, sep="\t" , index_col = 0)

    gene_list_1 = compute_gene_list_GO(k, combined_df_1, pval, term_col=term_col, adjpval_col=adjpval_col)
    gene_list_2 = compute_gene_list_GO(k, combined_df_2, pval, term_col=term_col, adjpval_col=adjpval_col)

    overlap_matrix = build_overlap_matrix(gene_list_1,gene_list_2)

    # graph clustermap on gene scores 
    sorted_overlap = sort_corr_matrix(overlap_matrix)

    # plot sorted program 
    g = sns.clustermap(sorted_overlap, 
                        row_cluster=False,   
                        col_cluster=False,   
                        cmap='coolwarm',      
                        figsize=(5, 5),
                        center = 0,                
                        xticklabels=False, 
                        yticklabels=False)      
                
    import textwrap

    wrapped_title = "\n".join(textwrap.wrap(title, width=50))  # adjust width as needed
    g.fig.suptitle(wrapped_title,fontweight='bold')
    print("max value:", sorted_overlap.max().max())


def plot_coefficient_variance(shared_genes1, shared_genes2, shared_genes3=None, title1 ="", title2 = "", title3 = ""):
    
    # Calculate CV FIRST (before plotting)
    cv_list1, cv_list2, cv_list3 = [], [], []
    K_values = sorted(shared_genes1.keys())
    
    for k in K_values:
        cv_list1.append(np.std(shared_genes1[k]) / np.mean(shared_genes1[k]))
        cv_list2.append(np.std(shared_genes2[k]) / np.mean(shared_genes2[k]))
    
    if shared_genes3 is not None:
        for k in K_values:
            cv_list3.append(np.std(shared_genes3[k]) / np.mean(shared_genes3[k]))
    
    # Plot ONCE after all calculations
    fig, ax = plt.subplots()
    ax.plot(K_values, cv_list1, 'o-',linewidth=2.5, markersize=8, color='red',label=title1)
    ax.plot(K_values, cv_list2, 'o-', linewidth=2.5, markersize=8, color='blue', label=title2)

    if shared_genes3 is not None:
        ax.plot(K_values, cv_list3, 'o-', linewidth=2.5, markersize=8, color='green', label=title3)

    ax.set_title('CV trend across K values', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax.legend()

'''
# graph pdf for 3 clustermap
def graph_pdf_clustermap(cor, distance, overlap, save_path, filename):

    with PdfPages(f"{save_path}/{filename}") as pdf:
        

        g1 = graph_cluster(cor, save = True, save_folder_name = save_path , save_file_name = "ClusterMap_for_correlation")
        g2 = graph_cluster(distance, save = True, save_folder_name = save_path , save_file_name = "ClusterMap_for_Euclidean_distance")
        g3 = graph_cluster(overlap, save = True, save_folder_name = save_path , save_file_name = "ClusterMap_for_top_300_genes")

        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        fig.suptitle(filename, 
                    fontsize=24, fontweight='bold', y=0.95)

        img1 = plt.imread(f"{save_path}/ClusterMap_for_correlation.png")
        axes[0].imshow(img1)
        axes[0].axis('off')
        
        img2 = plt.imread(f"{save_path}/ClusterMap_for_Euclidean_distance.png") 
        axes[1].imshow(img2)
        axes[1].axis('off')

        img3 = plt.imread(f"{save_path}/ClusterMap_for_top_300_genes.png") 
        axes[2].imshow(img3)
        axes[2].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig,bbox_inches='tight')
        plt.close()
        
        plt.close(g1.fig)
        plt.close(g2.fig)
        plt.close(g3.fig)
        os.remove(f"{save_path}/ClusterMap_for_correlation.png")
        os.remove(f"{save_path}/ClusterMap_for_Euclidean_distance.png")
        os.remove(f"{save_path}/ClusterMap_for_top_300_genes.png")


# graph heatmap base on the clustermap -> TODOs: need more thinking in this
def graph_heatmap(g, r, c, folder_name, file_name, num_gene = 300, sorted = False):
    # g = clustermap
    # r,c dimension of calculating averages

    mat = g.data2d.to_numpy() * num_gene

    assert mat.shape[0] % r == 0 and mat.shape[1] % c == 0, \
           "Rows/cols must divide evenly by block size."

    n_row_blocks = mat.shape[0] // r
    n_col_blocks = mat.shape[1] // c

    blocks = (mat.reshape(n_row_blocks, r, n_col_blocks, c)
            .swapaxes(1, 2)              
            .reshape(-1, r, c))

    block_means = (blocks.mean(axis=(1, 2))).astype(int)
    plt.figure(figsize=(12, 8))
    sns.heatmap(block_means.reshape(10,10),annot=True, cmap='inferno_r',fmt='d')        
    plt.title("Heatmap for matching programs " + file_name)

    if folder_name and file_name:
        g.savefig(f"{folder_name}/{file_name}.png")
 
    plt.show()

    # Sorted heatmap
    if sorted: 
        matrix = block_means.reshape(10,10).tolist()

        for i in range(len(matrix)):
            max_index = matrix[i].index(max(matrix[i]))
            # Swap max element with the diagonal element
            matrix[i][i], matrix[i][max_index] = matrix[i][max_index], matrix[i][i]

        plt.figure(figsize=(12, 8))
        sns.heatmap(np.array(matrix).reshape(r,c),annot=True, cmap='inferno_r',fmt='d')        
        plt.title("Sorted Heatmap for matching programs " + file_name)

        if folder_name and file_name:
            g.savefig(f"{folder_name}/{file_name}_sorted.png")
 
        plt.show()
'''