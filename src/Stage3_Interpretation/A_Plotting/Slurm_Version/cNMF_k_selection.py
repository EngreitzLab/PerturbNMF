#%%
import sys
from statsmodels.stats.multitest import fdrcorrection
import argparse
import yaml
import os
#%%

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Interpretation.Plotting.src import (load_stablity_error_data, plot_stablity_error,\
                         load_enrichment_data, plot_enrichment,\
                         load_perturbation_data, plot_perturbation,\
                         load_explained_variance_data,plot_explained_variance, programs_dotplots,plot_k_selection_panel
                          )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--groupby', type=str, default="sample")
    parser.add_argument('--K', nargs='*', type=int, default=None) # allow zero input
    parser.add_argument('--save_folder_name',  type=str, required=True)
    parser.add_argument('--pval',  type=float, default=0.05)
    parser.add_argument('--eval_folder_name',  type=str, required=True)
    parser.add_argument('--sel_threshs', nargs='*', type=float, default=None) # allow zero input
    parser.add_argument('--samples', nargs='*', type=str, default = None)
    parser.add_argument('--selected_k', type=int, default=None)

    # Enrichment file name and column name arguments
    parser.add_argument('--go_file', type=str, default=None,
        help='GO enrichment file name pattern. Use {k} as placeholder for K value. Default: {k}_GO_term_enrichment.txt')
    parser.add_argument('--geneset_file', type=str, default=None,
        help='Geneset enrichment file name pattern. Use {k} as placeholder for K value. Default: {k}_geneset_enrichment.txt')
    parser.add_argument('--trait_file', type=str, default=None,
        help='Trait enrichment file name pattern. Use {k} as placeholder for K value. Default: {k}_trait_enrichment.txt')
    parser.add_argument('--term_col', type=str, default='Term',
        help='Column name for the term/pathway name in enrichment files. Default: Term')
    parser.add_argument('--adjpval_col', type=str, default='Adjusted P-value',
        help='Column name for adjusted p-value in enrichment files. Default: Adjusted P-value')
    parser.add_argument('--perturbation_file', type=str, default=None,
        help='Perturbation file name pattern. Use {k} and {sample} as placeholders. Default: {k}_perturbation_association_results_{sample}.txt')
    parser.add_argument('--perturb_adjpval_col', type=str, default='adj_pval',
        help='Column name for adjusted p-value in perturbation files. Default: adj_pval')
    parser.add_argument('--perturb_target_col', type=str, default='target_name',
        help='Column name for target/regulator name in perturbation files. Default: target_name')
    parser.add_argument('--perturb_log2fc_col', type=str, default='log2FC',
        help='Column name for log2 fold change in perturbation files. Default: log2FC')
    parser.add_argument('--variance_file', type=str, default=None,
        help='Explained variance file name pattern. Use {k} as placeholder. Default: {k}_Explained_Variance_Summary.txt')
    parser.add_argument('--variance_col', type=str, default='Total',
        help='Column name for variance values. Use "Total" for summary files or column name for per-program files (will be summed). Default: Total')
    parser.add_argument('--stability_file', type=str, default=None,
        help='Path to a pre-computed stability/error file (TSV or NPZ). Bypasses cnmf.consensus(). '
             'Useful for torch-cNMF runs where the cnmf package is not installed.')


    args = parser.parse_args()

    # either change the array here or run each component in parallel
    if args.K is None:
        k_value = [30, 50, 60, 80, 100, 200, 250, 300]
    else:
        k_value = args.K

    if args.sel_threshs is None:
        sel_thresh_value = [0.4, 0.8, 2.0]
    else:
        sel_thresh_value = args.sel_threshs

    if args.samples is None:
        samples_value = ['D0', 'sample_D1', 'sample_D2', 'sample_D3']
    else:
        samples_value = args.samples


    # save comfigs used
    args.sel_threshs= sel_thresh_value
    args.K=k_value
    args.samples=samples_value

    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')

    os.makedirs(f'{args.save_folder_name}', exist_ok=True)
    with open(f'{args.save_folder_name}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)


    # Stability & Error
    stats_SE = load_stablity_error_data(output_directory = f'{args.output_directory}/{args.run_name}', run_name = 'Inference', components = k_value,
                                        stability_file = args.stability_file)
    plot_stablity_error(stats = stats_SE,folder_name = args.save_folder_name, file_name = "Stability_Error")

    for sel_thresh in sel_thresh_value:

        # Enrichement
        count_df = load_enrichment_data(folder = args.eval_folder_name, components = k_value, sel_thresh = sel_thresh,
            go_file=args.go_file, geneset_file=args.geneset_file, trait_file=args.trait_file,
            term_col=args.term_col, adjpval_col=args.adjpval_col)
        plot_enrichment(count_df,folder_name = args.save_folder_name, file_name = f"Enrichment_{sel_thresh}")

        # Perturbation
        test_stats_df = load_perturbation_data(folder = args.eval_folder_name, components = k_value, sel_thresh = sel_thresh,
        samples = samples_value, pval = args.pval, perturbation_file=args.perturbation_file,
        perturb_adjpval_col=args.perturb_adjpval_col, perturb_target_col=args.perturb_target_col,
        perturb_log2fc_col=args.perturb_log2fc_col)
        plot_perturbation(test_stats_df, folder_name = args.save_folder_name, pval=args.pval,file_name = f"Perturbation_{sel_thresh}")

        # Explained Variance
        stats_EV = load_explained_variance_data(folder = args.eval_folder_name, components=k_value, sel_thresh = sel_thresh,
            variance_file=args.variance_file, variance_col=args.variance_col)
        plot_explained_variance(stats_EV, folder_name = args.save_folder_name, file_name = f"Explained_Variance_{sel_thresh}")

        # Motif (working in progress)

        plot_k_selection_panel(stats_SE, count_df, test_stats_df, stats_EV,
                           pval=args.pval, folder_name= args.save_folder_name, file_name=f'K-selection_panel_{sel_thresh}', selected_k=args.selected_k)

    # program doplots
    for sel_thresh in sel_thresh_value:
        for k in k_value:
            fig = programs_dotplots(k, args.output_directory, args.run_name, sel_thresh = sel_thresh, groupby=args.groupby, figsize=(4, 30),
            show = False, save_name=f"Program_dotplot_{k}_{sel_thresh}", save_path = args.save_folder_name, ax = None)
