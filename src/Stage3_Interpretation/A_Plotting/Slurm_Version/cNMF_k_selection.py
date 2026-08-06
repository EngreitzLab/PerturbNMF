import sys
from statsmodels.stats.multitest import fdrcorrection
import argparse
import yaml
import os

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage3_Interpretation.A_Plotting.src import (load_stablity_error_data, plot_stablity_error,\
                         load_enrichment_data, plot_enrichment,\
                         load_perturbation_data, plot_perturbation,\
                         load_explained_variance_data,plot_explained_variance, programs_dotplots,plot_k_selection_panel
                          )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--groupby', type=str, default="sample")
    parser.add_argument('--K', nargs='*', type=int, default=[30, 50, 70, 80, 100, 200, 300], help='list of K values (number of components)')
    parser.add_argument('--save_folder_name',  type=str, required=True)
    parser.add_argument('--pval',  type=float, default=0.05)
    parser.add_argument('--eval_folder_name',  type=str, required=True)
    parser.add_argument('--sel_threshs', nargs='*', type=float, default=[0.2, 2.0], help='list of density thresholds')
    parser.add_argument('--Conditions', nargs='*', type=str, default=['D0', 'sample_D1', 'sample_D2', 'sample_D3'], help='list of condition labels')
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
    parser.add_argument('--run_program_dotplot', action='store_true',
        help='If set, also generate per-(K, sel_thresh) program dotplots. Requires the inference '
             'h5mu (cNMF_{K}_{thresh}.h5mu) to exist for each (K, sel_thresh). Default off.')


    args = parser.parse_args()

    # save comfigs used
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')

    os.makedirs(f'{args.save_folder_name}', exist_ok=True)
    with open(f'{args.save_folder_name}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)

    # ---- Pre-flight check: verify all required evaluation files exist ----
    _go_pat = args.go_file or '{k}_GO_term_enrichment.txt'
    _gs_pat = args.geneset_file or '{k}_geneset_enrichment.txt'
    _tr_pat = args.trait_file or '{k}_trait_enrichment.txt'
    _pt_pat = args.perturbation_file or '{k}_perturbation_association_results_{sample}.txt'
    _ev_pat = args.variance_file or '{k}_Explained_Variance_Summary.txt'

    missing = []
    for sel_thresh in args.sel_threshs:
        sel_str = str(sel_thresh).replace('.', '_')
        for k in args.K:
            k_folder = os.path.join(args.eval_folder_name, f"{k}_{sel_str}")
            for pat in [_go_pat, _gs_pat, _tr_pat, _ev_pat]:
                fpath = os.path.join(k_folder, pat.format(k=k))
                if not os.path.isfile(fpath):
                    missing.append(fpath)
            for cond in args.Conditions:
                fpath = os.path.join(k_folder, _pt_pat.format(k=k, sample=cond))
                if not os.path.isfile(fpath):
                    missing.append(fpath)
    if missing:
        print(f"ERROR: {len(missing)} required evaluation file(s) not found:")
        for m in missing:
            print(f"  {m}")
        raise FileNotFoundError(f"{len(missing)} required evaluation file(s) missing. See list above.")

    # Stability & Error
    stats_SE = load_stablity_error_data(output_directory = f'{args.output_directory}/{args.run_name}', run_name = 'Inference', components = args.K,
                                        stability_file = args.stability_file)
    plot_stablity_error(stats = stats_SE,folder_name = args.save_folder_name, file_name = "Stability_Error")

    for sel_thresh in args.sel_threshs:

        # Enrichement
        count_df = load_enrichment_data(folder = args.eval_folder_name, components = args.K, sel_thresh = sel_thresh,
            go_file=args.go_file, geneset_file=args.geneset_file, trait_file=args.trait_file,
            term_col=args.term_col, adjpval_col=args.adjpval_col)
        plot_enrichment(count_df,folder_name = args.save_folder_name, file_name = f"Enrichment_{sel_thresh}")

        # Perturbation
        test_stats_df = load_perturbation_data(folder = args.eval_folder_name, components = args.K, sel_thresh = sel_thresh,
        conditions = args.Conditions, pval = args.pval, perturbation_file=args.perturbation_file,
        perturb_adjpval_col=args.perturb_adjpval_col, perturb_target_col=args.perturb_target_col,
        perturb_log2fc_col=args.perturb_log2fc_col)
        plot_perturbation(test_stats_df, folder_name = args.save_folder_name, pval=args.pval,file_name = f"Perturbation_{sel_thresh}")

        # Explained Variance
        stats_EV = load_explained_variance_data(folder = args.eval_folder_name, components=args.K, sel_thresh = sel_thresh,
            variance_file=args.variance_file, variance_col=args.variance_col)
        plot_explained_variance(stats_EV, folder_name = args.save_folder_name, file_name = f"Explained_Variance_{sel_thresh}")

        # Motif (working in progress)

        # K-selection panel 
        plot_k_selection_panel(stats_SE, count_df, test_stats_df, stats_EV,
                           pval=args.pval, folder_name= args.save_folder_name, file_name=f'K-selection_panel_{sel_thresh}', selected_k=args.selected_k)

    # program dotplots (optional — requires inference h5mu per (K, sel_thresh))
    if args.run_program_dotplot:
        for sel_thresh in args.sel_threshs:
            for k in args.K:
                fig = programs_dotplots(k, args.output_directory, args.run_name, sel_thresh = sel_thresh, groupby=args.groupby, figsize=(4, 30),
                show = False, save_name=f"Program_dotplot_{k}_{sel_thresh}", save_path = args.save_folder_name, ax = None)

    return 0


if __name__ == '__main__':
    sys.exit(main())
