import sys
import muon as mu 
import numpy as np
import pandas as pd
import argparse
import yaml
import os

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Interpretation.Plotting.src import plot_umap_per_gene, plot_top_program_per_gene, perturbed_gene_dotplot,\
                         plot_log2FC, plot_volcano, programs_dotplot, analyze_correlations, \
                         create_gene_correlation_waterfall, \
                         convert_with_mygene, convert_adata_with_mygene, read_npz, \
                         merge_pdfs_in_folder, merge_svgs_to_pdf, create_comprehensive_plot, rename_adata_gene_dictionary, \
                         rename_list_gene_dictionary, plot_umap_per_gene_guide, process_single_gene, parallel_gene_processing,_process_gene_worker,\
                         compute_gene_correlation_matrix, compute_gene_waterfall_cor,perturbed_program_dotplot, \
                         plot_perturbation_vs_control

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    #io path
    parser.add_argument('--mdata_path', type=str, required=True, help='path to the MuData (.h5mu) file')
    parser.add_argument('--perturb_path_base', type=str, required=True, help='base path for perturbation result files (sample suffix appended automatically)')
    parser.add_argument('--ensembl_to_symbol_file', type=str, default=None, help='path to gene name mapping dictionary file for ID-to-name conversion')
    parser.add_argument('--reference_gtf_path', type=str, default=None, help='path to reference GTF file for checking gene names')

    # plotting variables
    parser.add_argument('--perturb_target_col', type=str, default="target_name", help='column name for target genes in perturbation results')
    parser.add_argument('--perturb_program_col', type=str, default="program_name", help='column name for programs in perturbation results')
    parser.add_argument('--perturb_log2fc_col', type=str, default="log2FC", help='column name for log2 fold change values')
    parser.add_argument('--top_corr_genes', type=int, default=5, help='number of top correlated genes to display per program')
    parser.add_argument('--top_n_programs', type=int, default=10, help='number of top programs to display per gene')
    parser.add_argument('--significance_threshold', type=float, default=0.05, help='p-value threshold for significance')
    parser.add_argument('--volcano_log2fc_min', type=float, default=-0.00, help='lower log2FC threshold for volcano plot')
    parser.add_argument('--volcano_log2fc_max', type=float, default=0.00, help='upper log2FC threshold for volcano plot')
    parser.add_argument('--save_path', type=str, required=True, help='directory path to save output plots')
    parser.add_argument('--square_plots', action="store_true", help='use square aspect ratio for plots')
    parser.add_argument('--figsize', type=float, nargs=2, default=(35, 35), help='figure size as width height')
    parser.add_argument('--show', action="store_true", help='display plots interactively')
    parser.add_argument('--PDF', action="store_true", help='save plots as PDF (default is SVG)')
    parser.add_argument('--n_processes', type=int, default=-1, help='number of parallel processes (-1 for all available cores)')
    parser.add_argument('--sample', nargs='*', type=str, default=None, help='list of sample names (default: D0 sample_D1 sample_D2 sample_D3)')
    parser.add_argument('--umap_dot_size', type=int, default=10, help='dot size for UMAP plots')
    parser.add_argument('--expressed_only', action="store_true", help='only plot perturbed genes found in the gene expression matrix (default: plot all perturbed genes)')
    parser.add_argument('--gene_list_file', type=str, default=None, help='path to a file with one gene name per line to process (overrides automatic perturbed gene detection)')
    parser.add_argument('--subsample_frac', type=float, default=None, help='fraction of cells to subsample for UMAP plots (e.g. 0.1 for 10%%). Default: None (plot all cells)')
    parser.add_argument('--parallel', action="store_true", help='use fork-based multiprocessing to plot genes in parallel (Linux only)')
    parser.add_argument('--corr_matrix_path', type=str, default=None, help='directory for precomputed gene waterfall correlation matrices. Files are expected as <dir>/corr_gene_matrix_<sample>.txt. Falls back to computing if not found.')

    # keys
    parser.add_argument('--data_key', type=str, default="rna", help='key to access gene expression data in MuData')
    parser.add_argument('--prog_key', type=str, default="cNMF", help='key to access cNMF programs in MuData')
    parser.add_argument('--gene_name_key', type=str, default="gene_names", help='key to access gene names in var')
    parser.add_argument('--categorical_key', type=str, default="sample", help='key to access sample/condition labels in obs')
    parser.add_argument('--control_target_name', type=str, default="non-targeting", help='name of non-targeting control in guide_targets (e.g. non-targeting, CTRL)')

    
    args = parser.parse_args()

    if args.sample is None:
        args.sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']



    # save comfigs used         
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')
    os.makedirs(f'{args.save_path}', exist_ok=True)
    with open(f'{args.save_path}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)



    #read mdata
    mdata = mu.read_h5mu(args.mdata_path)

    # check umap exist 
    if 'X_umap' not in mdata[args.prog_key].obsm:
        import scanpy as sc
        adata_tmp = mdata[args.data_key].copy()
        sc.pp.highly_variable_genes(adata_tmp, n_top_genes=2000, subset=True)
        sc.tl.pca(adata_tmp, n_comps=50)
        sc.pp.neighbors(adata_tmp)
        sc.tl.umap(adata_tmp)
        mdata[args.prog_key].obsm['X_pca'] = adata_tmp.obsm['X_pca']
        mdata[args.prog_key].obsm['X_umap'] = adata_tmp.obsm['X_umap']
        del adata_tmp



    # found detected perturbed gene
    perturbed_gene = np.unique(mdata[args.prog_key].uns["guide_targets"])
    gene_list = mdata[args.data_key].var_names.tolist()
    # gene_list = rename_list_gene_dictionary(mdata[args.data_key].var_names.tolist(), args.ensembl_to_symbol_file) # convert gene id to gene name
    perturbed_gene_found = sorted(set(gene_list) & set(perturbed_gene.tolist()))
    perturbed_gene_not_found = sorted(set(perturbed_gene.tolist()) - set(gene_list))
    print(f"there are {len(perturbed_gene_found)} perturbed genes found in expression matrix")
    print(f"there are {len(perturbed_gene_not_found)} perturbed genes NOT found in expression matrix: {perturbed_gene_not_found}")

    if args.gene_list_file is not None:
        with open(args.gene_list_file, 'r') as f:
            genes_requested = sorted([line.strip() for line in f if line.strip()])
        genes_valid = sorted(set(genes_requested) & set(gene_list))
        genes_missing = sorted(set(genes_requested) - set(gene_list))
        if genes_missing:
            print(f"WARNING: {len(genes_missing)} genes from {args.gene_list_file} not found in expression matrix: {genes_missing}")
        genes_to_plot = genes_valid
        print(f"Using {len(genes_to_plot)}/{len(genes_requested)} genes from {args.gene_list_file}")
    elif args.expressed_only:
        genes_to_plot = perturbed_gene_found
    else:
        genes_to_plot = sorted(perturbed_gene.tolist())


    # compute corr once
    correlation_matrix = compute_gene_correlation_matrix(mdata, ensembl_to_symbol_file=args.ensembl_to_symbol_file)

    waterfall_correlation = {}
    for samp in args.sample:
        precomputed = f"{args.corr_matrix_path}/corr_gene_matrix_{samp}.txt" if args.corr_matrix_path else None
        save = f"{args.corr_matrix_path}/corr_gene_matrix_{samp}.txt" if args.corr_matrix_path else None
        df = compute_gene_waterfall_cor(f"{args.perturb_path_base}_{samp}.txt", perturb_log2fc_col=args.perturb_log2fc_col, precomputed_path=precomputed, save_path=save)
        waterfall_correlation[samp] = (df)




    # Graph all pdf
    if args.parallel:
        print("Starting parallel gene processing...")
        try:
            result = parallel_gene_processing(
                perturbed_gene_list=genes_to_plot,
                mdata=mdata,
                perturb_path_base=args.perturb_path_base,
                ensembl_to_symbol_file=args.ensembl_to_symbol_file,
                gene_loading_corr_matrix=correlation_matrix,
                perturb_corr_by_sample=waterfall_correlation,
                top_n_programs=args.top_n_programs,
                dotplot_groupby=args.categorical_key,
                perturb_target_col=args.perturb_target_col,
                perturb_program_col=args.perturb_program_col,
                perturb_log2fc_col=args.perturb_log2fc_col,
                top_corr_genes=args.top_corr_genes,
                volcano_log2fc_min=args.volcano_log2fc_min,
                volcano_log2fc_max=args.volcano_log2fc_max,
                significance_threshold=args.significance_threshold,
                save_path=args.save_path,
                figsize=args.figsize,
                sample=args.sample,
                square_plots=args.square_plots,
                show=args.show,
                PDF=True,
                n_processes=args.n_processes,
                gene_name_key=args.gene_name_key,
                umap_dot_size=args.umap_dot_size,
                umap_subsample_frac=args.subsample_frac,
                control_target_name=args.control_target_name
            )
            print(f"Parallel processing completed. Results: {len(result) if result else 'None'}")
        except Exception as e:
            print(f"ERROR in parallel_gene_processing: {e}")
    else:
        for gene in genes_to_plot:
            create_comprehensive_plot(
                mdata=mdata,
                perturb_path_base=args.perturb_path_base,
                ensembl_to_symbol_file=args.ensembl_to_symbol_file,
                Target_Gene=gene,
                gene_loading_corr_matrix=correlation_matrix,
                perturb_corr_by_sample=waterfall_correlation,
                top_n_programs=args.top_n_programs,
                dotplot_groupby=args.categorical_key,
                perturb_target_col=args.perturb_target_col,
                perturb_program_col=args.perturb_program_col,
                perturb_log2fc_col=args.perturb_log2fc_col,
                top_corr_genes=args.top_corr_genes,
                volcano_log2fc_min=args.volcano_log2fc_min,
                volcano_log2fc_max=args.volcano_log2fc_max,
                significance_threshold=args.significance_threshold,
                save_path=args.save_path,
                save_name=gene,
                figsize=args.figsize,
                sample=args.sample,
                square_plots=args.square_plots,
                show=args.show,
                PDF=True,
                umap_dot_size=args.umap_dot_size,
                umap_subsample_frac=args.subsample_frac,
                gene_name_key=args.gene_name_key,
                control_target_name=args.control_target_name
            )

    # merge pdf 
    if args.PDF:
        merge_pdfs_in_folder(args.save_path, output_filename = "gene.pdf")
    else:
        merge_svgs_to_pdf(args.save_path)

