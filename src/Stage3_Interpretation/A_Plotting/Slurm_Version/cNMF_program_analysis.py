import sys
import muon as mu
import numpy as np
import pandas as pd
import argparse
import yaml
import os
from pathlib import Path



# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage3_Interpretation.A_Plotting.src import merge_pdfs_in_folder, merge_svgs_to_pdf

from Stage3_Interpretation.A_Plotting.src import plot_umap_per_program, plot_top_gene_per_program, top_GO_per_program, compute_program_correlation_matrix,\
                              analyze_program_correlations, plot_violin, plot_program_log2FC, plot_program_heatmap, plot_program_volcano, \
                              perturbed_program_dotplot, compute_program_waterfall_cor, create_program_correlation_waterfall, create_comprehensive_program_plot, \
                              export_program_html, write_share_index


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    #io path
    parser.add_argument('--mdata_path', type=str, required=True, help='path to the MuData (.h5mu) file')
    parser.add_argument('--perturb_path_base', type=str, required=True, help='base path for perturbation result files (sample suffix appended automatically)')
    parser.add_argument('--file_to_dictionary', type=str, default=None, help='path to gene name mapping dictionary file for ID-to-name conversion')
    parser.add_argument('--reference_gtf_path', type=str, default=None, help='path to reference GTF file for checking gene names')
    parser.add_argument('--GO_path', type=str, required=True, help='path to Gene Ontology enrichment results directory')

    # plotting variables
    parser.add_argument('--tagert_col_name', type=str, default="program_name", help='column name for target programs in perturbation results')
    parser.add_argument('--plot_col_name', type=str, default="target_name", help='column name for genes in perturbation results')
    parser.add_argument('--log2fc_col', type=str, default="log2FC", help='column name for log2 fold change values')
    parser.add_argument('--top_program', type=int, default=10, help='number of top programs to display')
    parser.add_argument('--top_enrichned_term', type=int, default=10, help='number of top GO enrichment terms to display per program')
    parser.add_argument('--p_value', type=float, default=0.05, help='p-value threshold for significance')
    parser.add_argument('--down_thred_log', type=float, default=-0.00, help='lower log2FC threshold for volcano plot')
    parser.add_argument('--up_thred_log', type=float, default=0.00, help='upper log2FC threshold for volcano plot')
    parser.add_argument('--pdf_save_path', type=str, required=True, help='directory path to save output plots')
    parser.add_argument('--square_plots', action="store_true", help='use square aspect ratio for plots')
    parser.add_argument('--figsize', type=float, nargs=2, default=(35, 35), help='figure size as width height')
    parser.add_argument('--show', action="store_true", help='display plots interactively')
    parser.add_argument('--output_format', type=str, default='SVG', choices=['PDF', 'SVG', 'HTML'],
                        help='output format: PDF (matplotlib + PyPDF2 merge), SVG (matplotlib + svglib merge), HTML (interactive Plotly share folder)')
    parser.add_argument('--html_share_path', type=str, default=None,
                        help='output folder for HTML mode (default: {pdf_save_path}/html_share)')
    parser.add_argument('--PDF', action="store_true",
                        help='[DEPRECATED] alias for --output_format PDF')
    parser.add_argument('--sample', nargs='*', type=str, default=None, help='list of sample names (default: D0 sample_D1 sample_D2 sample_D3)')
    parser.add_argument('--programs', nargs='+', type=int, default=None, help='specific program numbers to plot (e.g. 4 5 6 ... 100). If omitted, all programs are plotted.')
    parser.add_argument('--subsample_frac', type=float, default=None, help='fraction of cells to subsample for UMAP plots (e.g. 0.1 for 10%%). Default: None (plot all cells)')
    parser.add_argument('--corr_matrix_path', type=str, default=None, help='base path for precomputed waterfall correlation matrices (e.g. /path/to/corr_matrix). Files are expected as <base>_<sample>.txt. Falls back to computing if not found.')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='re-process every program; do not skip ones whose output already exists. Default: resume is on.')
    parser.set_defaults(resume=True)

    # keys
    parser.add_argument('--data_key', type=str, default="rna", help='key to access gene expression data in MuData')
    parser.add_argument('--prog_key', type=str, default="cNMF", help='key to access cNMF programs in MuData')
    parser.add_argument('--gene_name_key', type=str, default="gene_names", help='key to access gene names in var')
    parser.add_argument('--categorical_key', type=str, default="sample", help='key to access sample/condition labels in obs')



    args = parser.parse_args()

    if args.PDF:
        print("WARNING: --PDF is deprecated; use --output_format PDF instead.", file=sys.stderr)
        args.output_format = 'PDF'

    if args.sample is None:
        args.sample = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']

    if args.html_share_path is None:
        args.html_share_path = os.path.join(args.pdf_save_path, 'html_share')



    # save comfigs used         
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')
    os.makedirs(f'{args.pdf_save_path}', exist_ok=True)
    with open(f'{args.pdf_save_path}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)



    #read mdata
    mdata = mu.read_h5mu(args.mdata_path)


    # check umap exist
    if 'X_umap' not in mdata['cNMF'].obsm:

        import scanpy as sc
        rna_tmp = mdata['rna'].copy()
        # Select top 2000 genes by variance (robust to any normalization)
        variances = np.array(rna_tmp.X.power(2).mean(axis=0) - np.power(rna_tmp.X.mean(axis=0), 2)).flatten() \
            if hasattr(rna_tmp.X, 'power') else rna_tmp.X.var(axis=0)
        top_idx = np.argsort(variances)[-2000:]
        rna_tmp = rna_tmp[:, top_idx]
        sc.tl.pca(rna_tmp, n_comps=50)
        sc.pp.neighbors(rna_tmp)
        sc.tl.umap(rna_tmp)
        mdata['cNMF'].obsm['X_pca'] = rna_tmp.obsm['X_pca']
        mdata['cNMF'].obsm['X_umap'] = rna_tmp.obsm['X_umap']




    program_len = len(mdata['cNMF'].var) # find out list of programs to process 
    print(f"there are {program_len} Program found")



    # found detected perturbed gene (use gene symbols from var column when var_names are Ensembl IDs)
    perturbed_gene = np.unique(mdata[args.prog_key].uns["guide_targets"])
    if args.gene_name_key in mdata[args.data_key].var.columns:
        gene_symbols = mdata[args.data_key].var[args.gene_name_key].astype(str).tolist()
    else:
        gene_symbols = mdata[args.data_key].var_names.tolist()
    perturbed_gene_found = sorted(set(gene_symbols) & set(perturbed_gene.tolist()))




    # compute correlations
    waterfall_correlation = {}

    for samp in args.sample:
        precomputed = f"{args.corr_matrix_path}/corr_program_matrix_{samp}.txt" if args.corr_matrix_path else None
        save = f"{args.corr_matrix_path}/corr_program_matrix_{samp}.txt" if args.corr_matrix_path else None
        df = compute_program_waterfall_cor(f"{args.perturb_path_base}_{samp}.txt", precomputed_path=precomputed, save_path=save, log2fc_col=args.log2fc_col)
        waterfall_correlation[samp] = (df)

    program_correlation = compute_program_correlation_matrix(mdata)
        
    


    programs_to_plot = args.programs if args.programs is not None else list(mdata[args.prog_key].var_names)

    # Resume support: build the set of programs that actually need processing,
    # but keep `programs_to_plot` as the full ordered list so HTML nav/index stay correct.
    if args.resume:
        if args.output_format == 'HTML':
            done = {p.parent.name[len('program_'):]
                    for p in Path(args.html_share_path).glob('program_*/metadata.json')}
        else:
            ext = '.pdf' if args.output_format == 'PDF' else '.svg'
            done = {p.stem for p in Path(args.pdf_save_path).glob(f'*{ext}')}
        process_set = {str(p) for p in programs_to_plot if str(p) not in done}
        skipped = len(programs_to_plot) - len(process_set)
        if skipped:
            print(f"Resume: skipping {skipped} already-produced program(s); {len(process_set)} remaining.")
    else:
        process_set = {str(p) for p in programs_to_plot}

    programs_str = [str(p) for p in programs_to_plot]
    n_progs = len(programs_str)
    for i, program in enumerate(programs_to_plot):

        if str(program) not in process_set:
            continue

        if args.output_format == 'HTML':
            prev_pid = programs_str[i - 1] if i > 0 else None
            next_pid = programs_str[i + 1] if i + 1 < n_progs else None
            export_program_html(
                mdata=mdata,
                perturb_path_base=args.perturb_path_base,
                GO_path=args.GO_path,
                file_to_dictionary=args.file_to_dictionary,
                Target_Program=str(program),
                program_correlation=program_correlation,
                waterfall_correlation=waterfall_correlation,
                sample=args.sample,
                perturbed_gene_found=perturbed_gene_found,
                html_share_path=args.html_share_path,
                top_program=args.top_program,
                groupby=args.categorical_key,
                tagert_col_name=args.tagert_col_name,
                plot_col_name=args.plot_col_name,
                log2fc_col=args.log2fc_col,
                top_enrichned_term=args.top_enrichned_term,
                down_thred_log=args.down_thred_log,
                up_thred_log=args.up_thred_log,
                p_value=args.p_value,
                gene_name_key=args.gene_name_key,
                subsample_frac=args.subsample_frac,
                prev_program_id=prev_pid,
                next_program_id=next_pid,
                position_index=i + 1,
                position_total=n_progs,
            )
        else:
            create_comprehensive_program_plot(
                mdata=mdata,
                perturb_path_base=args.perturb_path_base,
                GO_path=args.GO_path,
                file_to_dictionary=args.file_to_dictionary,
                Target_Program=str(program),
                program_correlation=program_correlation,
                waterfall_correlation=waterfall_correlation,
                top_program=args.top_program,
                groupby=args.categorical_key,
                tagert_col_name=args.tagert_col_name,
                plot_col_name=args.plot_col_name,
                log2fc_col=args.log2fc_col,
                top_enrichned_term=args.top_enrichned_term,
                down_thred_log=args.down_thred_log,
                up_thred_log=args.up_thred_log,
                p_value=args.p_value,
                save_path=args.pdf_save_path,
                save_name=str(program),
                figsize=args.figsize,
                sample=args.sample,
                square_plots=args.square_plots,
                show=args.show,
                PDF=(args.output_format == 'PDF'),
                gene_list=perturbed_gene_found,
                subsample_frac=args.subsample_frac,
                gene_name_key=args.gene_name_key
            )


    # post-loop assembly
    if args.output_format == 'PDF':
        merge_pdfs_in_folder(args.pdf_save_path, output_filename="program.pdf")
    elif args.output_format == 'SVG':
        merge_svgs_to_pdf(args.pdf_save_path)
    else:  # HTML
        write_share_index(args.html_share_path, programs_to_plot, args_dict)

