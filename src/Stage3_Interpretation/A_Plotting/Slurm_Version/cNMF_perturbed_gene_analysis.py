import sys
import gc
import muon as mu
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage3_Interpretation.A_Plotting.src import plot_umap_per_gene, plot_top_program_per_gene, perturbed_gene_dotplot,\
                         plot_log2FC, plot_volcano, programs_dotplot, analyze_correlations, \
                         create_gene_correlation_waterfall, \
                         convert_with_mygene, convert_adata_with_mygene, read_npz, \
                         merge_pdfs_in_folder, merge_svgs_to_pdf, create_comprehensive_plot, rename_adata_gene_dictionary, \
                         rename_list_gene_dictionary, plot_umap_per_gene_guide, process_single_gene, parallel_gene_processing,_process_gene_worker,\
                         compute_gene_correlation_matrix, compute_gene_waterfall_cor,perturbed_program_dotplot, \
                         plot_perturbation_vs_control, \
                         export_gene_html, write_gene_share_index, \
                         check_normalized, ensure_umap

def main():

    parser = argparse.ArgumentParser()

    #io path
    parser.add_argument('--mdata_path', type=str, required=True, help='path to the MuData (.h5mu) file')
    parser.add_argument('--perturb_path_base', type=str, default=None, help='base path for perturbation result files (sample suffix appended automatically). Omit to skip every per-condition perturbation panel (log2FC, volcano, program dotplot, waterfall) and plot only the h5mu-derived rows. Required for --output_format HTML.')
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
    parser.add_argument('--save_path', type=str, required=True, help='directory path to save output (PDF/SVG files or HTML share tree)')
    parser.add_argument('--square_plots', action="store_true", help='use square aspect ratio for plots')
    parser.add_argument('--figsize', type=float, nargs=2, default=(35, 35), help='figure size as width height')
    parser.add_argument('--show', action="store_true", help='display plots interactively')
    parser.add_argument('--output_format', type=str, default='SVG', choices=['PDF', 'SVG', 'HTML'], help='output format: PDF (matplotlib + PyPDF2 merge), SVG (matplotlib + svglib merge), HTML (interactive Plotly share folder)')
    parser.add_argument('--n_processes', type=int, default=4, help='number of parallel processes for --parallel mode. Each worker holds a copy-on-write fork of mdata, so RAM scales with worker count: on large data (e.g. 1M cells x 30K genes), -1 (all cores) on a high-CPU node can hit 30-70 GB RSS. Default 4 keeps it bounded. Set -1 only when you know the per-worker RAM cost.')
    parser.add_argument('--Conditions', nargs='*', type=str, default=['D0', 'sample_D1', 'sample_D2', 'sample_D3'], help='list of condition names')
    parser.add_argument('--umap_dot_size', type=int, default=10, help='dot size for UMAP plots')
    parser.add_argument('--expressed_only', action="store_true", help='only plot perturbed genes found in the gene expression matrix (default: plot all perturbed genes)')
    parser.add_argument('--gene_list_file', type=str, default=None, help='path to a file with one gene name per line to process (overrides automatic perturbed gene detection)')
    parser.add_argument('--subsample_frac', type=float, default=None, help='fraction of cells to subsample for UMAP plots (e.g. 0.1 for 10%%). Default: None (plot all cells)')
    parser.add_argument('--parallel', action="store_true", help='use fork-based multiprocessing to plot genes in parallel (Linux only)')
    parser.add_argument('--corr_matrix_path', type=str, default=None, help='directory for precomputed gene waterfall correlation matrices. Files are expected as <dir>/corr_gene_matrix_<sample>.txt. Falls back to computing if not found.')
    parser.add_argument('--skip_existing', action='store_false', help='[default on] skip genes whose output already exists. Pass --skip_existing to force re-process all.')

    # keys
    parser.add_argument('--data_key', type=str, default="rna", help='key to access gene expression data in MuData')
    parser.add_argument('--prog_key', type=str, default="cNMF", help='key to access cNMF programs in MuData')
    parser.add_argument('--gene_name_key', type=str, default="gene_names", help='key to access gene names in var')
    parser.add_argument('--categorical_key', type=str, default="sample", help='key to access sample/condition labels in obs')
    parser.add_argument('--guide_targets_key', type=str, default="guide_targets", help='key in .uns to access guide target genes (default: guide_targets)')
    parser.add_argument('--guide_assignment_key', type=str, default="guide_assignment", help='key in .obsm to access the guide-assignment matrix (default: guide_assignment)')
    parser.add_argument('--control_target_name', type=str, nargs='+', default=["non-targeting"], help='one or more control labels in guide_targets (e.g. non-targeting, or WT WT111 WT4). A cell is a control if its guide target matches any of these.')

    
    args = parser.parse_args()

    # export_gene_html renders the perturbation sections unconditionally, so the
    # HTML report cannot be produced without the per-sample association files.
    if args.output_format == 'HTML' and args.perturb_path_base is None:
        parser.error("--output_format HTML requires --perturb_path_base "
                     "(the HTML export always renders the perturbation sections). "
                     "Use --output_format PDF or SVG to plot without perturbation results.")

    # save comfigs used
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')
    os.makedirs(f'{args.save_path}', exist_ok=True)
    with open(f'{args.save_path}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)


    #read mdata
    mdata = mu.read_h5mu(args.mdata_path)



    # validate that expression matrix is normalized (TPM/CPM/log), not raw counts.
    # Downstream UMAP/PCA + per-gene expression plots assume normalized values.
    check_normalized(mdata[args.data_key], args.data_key)

    # compute UMAP/PCA from top-variance genes if missing
    ensure_umap(mdata, args.data_key, args.prog_key)


    # found detected perturbed gene (use gene symbols from var column when var_names are Ensembl IDs)
    perturbed_gene = np.unique(mdata[args.prog_key].uns[args.guide_targets_key])
    if args.gene_name_key in mdata[args.data_key].var.columns:
        gene_symbols = mdata[args.data_key].var[args.gene_name_key].astype(str).tolist()
    else:
        gene_symbols = mdata[args.data_key].var_names.tolist()

    # print out how many perturbation is not found expressed 
    perturbed_gene_found = sorted(set(gene_symbols) & set(perturbed_gene.tolist()))
    perturbed_gene_not_found = sorted(set(perturbed_gene.tolist()) - set(gene_symbols))
    print(f"there are {len(perturbed_gene_found)} perturbed genes found in expression matrix")
    print(f"there are {len(perturbed_gene_not_found)} perturbed genes NOT found in expression matrix: {perturbed_gene_not_found}")

    # decide which genes are being ploted 
    if args.gene_list_file is not None: # a given list of genes 
        with open(args.gene_list_file, 'r') as f:
            genes_requested = sorted([line.strip() for line in f if line.strip()])
        genes_valid = sorted(set(genes_requested) & set(gene_symbols))
        genes_missing = sorted(set(genes_requested) - set(gene_symbols))
        if genes_missing:
            print(f"WARNING: {len(genes_missing)} genes from {args.gene_list_file} not found in expression matrix: {genes_missing}")
        genes_to_plot = genes_valid
        print(f"Using {len(genes_to_plot)}/{len(genes_requested)} genes from {args.gene_list_file}")
    elif args.expressed_only: # only expressed
        genes_to_plot = perturbed_gene_found
    else: # all genes
        genes_to_plot = sorted(perturbed_gene.tolist())

    # The control target is never tested in the perturbation-association results,
    # so plotting it produces an empty figure and crashes downstream. Drop it.
    control_set = set(args.control_target_name)
    present = control_set & set(genes_to_plot)
    if present:
        genes_to_plot = [g for g in genes_to_plot if g not in control_set]
        print(f"Excluding control target(s) {sorted(present)} from perturbed-gene plots")

    # Skip-existing support: build the set of genes that actually need processing,
    # but keep `genes_to_plot` as the full ordered list so HTML nav/index stay correct.
    if args.skip_existing:
        if args.output_format == 'HTML':
            done = {p.parent.name[len('gene_'):]
                    for p in Path(args.save_path).glob('gene_*/metadata.json')}
        else:
            ext = '.pdf' if args.output_format == 'PDF' else '.svg'
            done = {p.stem for p in Path(args.save_path).glob(f'*{ext}')}
            
        process_set = {g for g in genes_to_plot if g not in done}
        skipped = len(genes_to_plot) - len(process_set)
        if skipped:
            print(f"Skip-existing: skipping {skipped} already-produced gene(s); {len(process_set)} remaining.")
    else:
        process_set = set(genes_to_plot)


    # compute corr once (cached as .npz factors under corr_dir; reused on subsequent runs)
    corr_dir = args.corr_matrix_path or os.path.join(args.save_path, "_corr_cache")
    os.makedirs(corr_dir, exist_ok=True)

    # The waterfall correlation is built from the perturbation files; without them
    # the per-condition rows are skipped and nothing needs to be computed.
    if args.perturb_path_base is None:
        waterfall_correlation = None
        print("No --perturb_path_base given: skipping per-condition perturbation panels "
              "(log2FC, volcano, program dotplot, waterfall).")
    else:
        waterfall_correlation = {}
        for samp in args.Conditions:
            path = f"{corr_dir}/waterfall_factor_{samp}.npz"
            waterfall_correlation[samp] = compute_gene_waterfall_cor(
                f"{args.perturb_path_base}_{samp}.txt",
                perturb_log2fc_col=args.perturb_log2fc_col,
                precomputed_path=path,
                save_path=path,
            )

    correlation_matrix = compute_gene_correlation_matrix(
        mdata,
        ensembl_to_symbol_file=args.ensembl_to_symbol_file,
        precomputed_path=f"{corr_dir}/gene_loading_factor.npz",
        save_path=f"{corr_dir}/gene_loading_factor.npz",
        data_key=args.data_key,
        prog_key=args.prog_key,
    )
    

    # Graph all genes
    if args.output_format == 'HTML':
        n_genes = len(genes_to_plot)
        for i, gene in enumerate(genes_to_plot):
            if gene not in process_set:
                continue

            # enable pre/after browsing 
            prev_g = genes_to_plot[i - 1] if i > 0 else None 
            next_g = genes_to_plot[i + 1] if i + 1 < n_genes else None

            # make html per gene 
            export_gene_html(
                mdata=mdata,
                perturb_path_base=args.perturb_path_base,
                ensembl_to_symbol_file=args.ensembl_to_symbol_file,
                Target_Gene=gene,
                gene_loading_corr_matrix=correlation_matrix,
                perturb_corr_by_sample=waterfall_correlation,
                sample=args.Conditions,
                html_share_path=args.save_path,
                top_n_programs=args.top_n_programs,
                top_corr_genes=args.top_corr_genes,
                groupby=args.categorical_key,
                perturb_target_col=args.perturb_target_col,
                perturb_program_col=args.perturb_program_col,
                perturb_log2fc_col=args.perturb_log2fc_col,
                volcano_log2fc_min=args.volcano_log2fc_min,
                volcano_log2fc_max=args.volcano_log2fc_max,
                significance_threshold=args.significance_threshold,
                gene_name_key=args.gene_name_key,
                control_target_name=args.control_target_name,
                umap_dot_size=args.umap_dot_size,
                subsample_frac=args.subsample_frac,
                prev_gene=prev_g,
                next_gene=next_g,
                position_index=i + 1,
                position_total=n_genes,
                data_key=args.data_key,
                prog_key=args.prog_key,
                guide_targets_key=args.guide_targets_key,
                guide_assignment_key=args.guide_assignment_key,
            )
            plt.close('all')
            if (i + 1) % 20 == 0:
                gc.collect()

        write_gene_share_index(args.save_path, genes_to_plot, args_dict)

    elif args.parallel:
        print("Starting parallel gene processing...")
        try:
            result = parallel_gene_processing(
                perturbed_gene_list=[g for g in genes_to_plot if g in process_set],
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
                sample=args.Conditions,
                square_plots=args.square_plots,
                show=args.show,
                PDF=(args.output_format == 'PDF'),
                n_processes=args.n_processes,
                gene_name_key=args.gene_name_key,
                umap_dot_size=args.umap_dot_size,
                umap_subsample_frac=args.subsample_frac,
                control_target_name=args.control_target_name,
                data_key=args.data_key,
                prog_key=args.prog_key,
                guide_targets_key=args.guide_targets_key,
                guide_assignment_key=args.guide_assignment_key
            )
            print(f"Parallel processing completed. Results: {len(result) if result else 'None'}")

        except Exception as e:
            print(f"ERROR in parallel_gene_processing: {e}")

    else:
        for i, gene in enumerate(genes_to_plot):

            if gene not in process_set:
                continue

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
                sample=args.Conditions,
                square_plots=args.square_plots,
                show=args.show,
                PDF=(args.output_format == 'PDF'),
                umap_dot_size=args.umap_dot_size,
                umap_subsample_frac=args.subsample_frac,
                gene_name_key=args.gene_name_key,
                control_target_name=args.control_target_name,
                data_key=args.data_key,
                prog_key=args.prog_key,
                guide_targets_key=args.guide_targets_key,
                guide_assignment_key=args.guide_assignment_key
            )
            plt.close('all')
            if (i + 1) % 20 == 0:
                gc.collect()

    # post-loop assembly
    if args.output_format == 'PDF':
        merge_pdfs_in_folder(args.save_path, output_filename="gene.pdf")
    elif args.output_format == 'SVG':
        merge_svgs_to_pdf(args.save_path, output_filename="gene.pdf")
    # no HTML, index already written inside the HTML branch above

    return 0


if __name__ == '__main__':
    sys.exit(main())

