
from tests.synthetic_data import make_sceptre_style_synth
from src.sceptre import prepare_crt_inputs
import scanpy as sc
import yaml
import muon as mu
import pandas as pd
import numpy as np
import argparse
import os

from src.sceptre import (
    build_ntc_group_inputs,
    compute_guide_set_null_pvals,
    crt_pvals_for_ntc_groups_ensemble,
    crt_pvals_for_ntc_groups_ensemble_skew,
    make_ntc_groups_ensemble,
    run_all_genes_union_crt,
)
from src.visualization import qq_plot_ntc_pvals



# reformat adata info for CRT
def reformat_data_for_CRT(mdata, mdata_guide, covariates=None, log_covariates=None):

    adata = mdata["cNMF"].copy()
    adata.obsm["cnmf_usage"] = np.asarray(adata.X)  # ensure dense float

    # Align gene modality to adata cells
    g = mdata_guide[adata.obs_names].copy()

    # Guide assignment (cell x guide)
    adata.obsm["guide_assignment"] = g.obsm["guide_assignment"].copy()

    # Program names
    adata.uns["program_names"] = list(adata.var_names)

    # Guide names must match columns of guide_assignment
    guide_names = list(mdata_guide.uns["guide_names"])
    adata.uns["guide_names"] = guide_names

    # guide2gene must map guide_name -> gene name (keys must match guide_names)
    '''
    guide2gene = (
        mdata_guide.uns["guide_targets"]
        .loc[guide_names]
        .to_dict()
    )
    '''

    guide2gene = dict(zip(guide_names,mdata_guide.uns["guide_targets"]))

    adata.uns["guide2gene"] = guide2gene

    # Covariates
    covar_dict = {}
    if covariates:
        for key in covariates:
            covar_dict[key] = adata.obs[key]
    if log_covariates:
        for key in log_covariates:
            covar_dict[f'log_{key}'] = np.log1p(adata.obs[key])

    adata.obsm["covar"] = pd.DataFrame(covar_dict, index=adata.obs_names)

    return adata


# run one CRT
def run_CRT(adata, k, output_folder, args):

    # perform CRT for each condition of cells 
    for condition in adata.obs[args.categorical_key].unique():
        adata_con = adata[adata.obs[args.categorical_key] == condition].copy()
        
        inputs = prepare_crt_inputs(
            adata=adata_con,
            usage_key="cnmf_usage",
            covar_key="covar",
            guide_assignment_key="guide_assignment",
            guide2gene_key="guide2gene"
        )


        out = run_all_genes_union_crt(
            inputs=inputs,
            B=args.number_permutations,
            n_jobs=-1,
            calibrate_skew_normal=True,
            return_raw_pvals=True,
            return_skew_normal=True,
        )

        ntc_labels = args.guide_annotation_key # Identify NTC guides and build guide-frequency bins / real-gene signatures
        ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
            inputs=inputs,
            ntc_label=ntc_labels,
            group_size=args.number_guide,
            n_bins=10,
        )

        # Create multiple random partitions (ensembles) of NTC guides into 6-guide groups
        ntc_groups_ens = make_ntc_groups_ensemble(
            ntc_guides=ntc_guides,
            ntc_freq=guide_freq,
            real_gene_bin_sigs=real_sigs,
            guide_to_bin=guide_to_bin,
            n_ensemble=10,
            seed0=7,
            group_size=args.number_guide,
            max_groups=None,
        )

        # Compute raw CRT p-values for each NTC group in each ensemble
        ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
            inputs=inputs,
            ntc_groups_ens=ntc_groups_ens,
            B=args.number_permutations,
            seed0=23,
        )

        # Compute skew-calibrated CRT p-values for each NTC group in each ensemble
        ntc_group_pvals_skew_ens = crt_pvals_for_ntc_groups_ensemble_skew(
            inputs=inputs,
            ntc_groups_ens=ntc_groups_ens,
            B=args.number_permutations,
            seed0=23,
        )

        # Build CRT-null p-values matched to NTC group units (recommended)
        guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
        null_pvals = np.concatenate(
            [
                # Each entry is a (B, K) matrix of null p-values for one NTC group
                compute_guide_set_null_pvals(
                    guide_idx=[guide_to_col[g] for g in guides],
                    inputs=inputs,
                    B=args.number_permutations,
                ).ravel()
                for groups in ntc_groups_ens
                for guides in groups.values()
            ]
        )

        ax = qq_plot_ntc_pvals(
            pvals_raw_df=out["pvals_raw_df"],
            guide2gene=adata.uns["guide2gene"],
            ntc_genes=ntc_labels,
            pvals_skew_df=out["pvals_df"],
            null_pvals=null_pvals,
            ntc_group_pvals_ens=ntc_group_pvals_ens,
            ntc_group_pvals_skew_ens=ntc_group_pvals_skew_ens,
            show_ntc_ensemble_band=True,
            show_all_pvals=True,
            title=f"QQ plot: grouped NTC controls (raw vs skew) vs CRT null for {condition}",
        )

        import matplotlib.pyplot as plt

        plt.tight_layout()

        if output_folder:
            plt.savefig(f"{output_folder}/CRT_{condition}.png", dpi=100)
            save_result(out, k, output_folder, condition, args)

        plt.close()


# save results
def save_result(out, k, output_folder, condition, args):

    pval_df = out['pvals_skew_df']
    beta_df = out['betas_df']

    pval_long = pval_df.reset_index().melt(
        id_vars='index',
        var_name='program_name',
        value_name='p-value'
    ).rename(columns={'index': 'target_name'})
    
    # Melt beta dataframe
    beta_long = beta_df.reset_index().melt(
        id_vars='index',
        var_name='program_name',
        value_name='log2FC'
    ).rename(columns={'index': 'target_name'})
    
    # Merge on program and target_gene
    result_df = pval_long.merge(
        beta_long,
        on=['program_name', 'target_name'],
        how='inner'
    )
    
    # Reorder columns
    result_df = result_df[[ 'target_name', 'program_name', 'log2FC', 'p-value']]

    # effect size of CRT is not log2fc, its approx_log2FC = [K / (K - 1)] * beta_hat / ln(2)
    #result_df['log2FC'] = result_df['log2FC']/np.log(2)

    # Correct for multiple testing
    if args.FDR_method == 'BH':

        from statsmodels.stats.multitest import multipletests
        result_df['adj_pval'] = multipletests(result_df['p-value'], method='fdr_bh')[1]

    elif args.FDR_method == 'StoreyQ':

        from multipy.fdr import qvalue
        import builtins
        if not hasattr(builtins, 'xrange'):
            builtins.xrange = range
        result_df['adj_pval'] = qvalue(result_df['p-value'].values, threshold=0.05, verbose=False)[1]


    result_df.to_csv(f'{output_folder}/{k}_CRT_{condition}.txt',sep='\t',index=False)
    
    return result_df


# Main Execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--components', nargs='*', type=int, help = "list of K values (number of components) to test (default: [30, 50, 60, 80, 100, 200, 250, 300])", default=None)
    parser.add_argument('--sel_thresh', nargs='*', type=float, help = "list of density threshold values for consensus selection (default: [0.4, 0.8, 2.0])", default=None)
    parser.add_argument('--mdata_guide_path', type=str,  help='Path to MuData object (.h5mu) containing guide assignment information', required=True)

    # keys
    parser.add_argument('--categorical_key', help='Key in .obs to access cell condition/sample labels (default: sample)', type=str, default="sample")

    # Covariates
    parser.add_argument('--covariates', nargs='*', type=str, help='Covariate keys in .obs to include as-is (e.g., biological_sample)', default=None)
    parser.add_argument('--log_covariates', nargs='*', type=str, help='Covariate keys in .obs to log1p-transform before inclusion (e.g., guide_umi_counts total_counts)', default=None)

    # Calibration parameters
    parser.add_argument('--number_guide', help='Number of non-targeting guides to randomly designate as "targeting" in each calibration iteration (default: 6)', type=int, default=6)
    parser.add_argument('--number_permutations', help='Number of calibration iterations to run with (default: 1024)', type=int, default=1024)
    parser.add_argument('--guide_annotation_key', nargs='*', type=str,  help='Name of target for non-targeting/safe-targeting guides,default="non-targeting"', default='non-targeting')
    parser.add_argument('--FDR_method', type=str, choices=['BH', 'StoreyQ'], default='BH', help='FDR correction method: BH (Benjamini-Hochberg) or StoreyQ (Storey Q-value) (default: BH)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results and figures. If not provided, defaults to <out_dir>/<run_name>/Evaluation/<K>_<sel_thresh>/')


    args = parser.parse_args()


    # either change the array here or run each component in parallel
    if args.components is None:
        args.components = [30, 50, 60, 80, 100, 200, 250, 300]

    if args.sel_thresh is None:
        args.sel_thresh = [0.4, 0.8, 2.0]

    # save comfigs used         
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')
    with open(f'{args.save_dir}/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)


    # read guide
    mdata_guide = mu.read(args.mdata_guide_path)


    # run CRT for each K and dt
    for sel_thresh in args.sel_thresh:
        for k in args.components:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            if args.save_dir:
                output_folder = args.save_dir
            else:
                output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"
            os.makedirs(output_folder, exist_ok=True)

            # Load mdata
            mdata = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu')

            # Assign guide
            adata = reformat_data_for_CRT(mdata, mdata_guide,
                                          covariates=args.covariates,
                                          log_covariates=args.log_covariates)

            # add flooring for program matrix with exceesive zeros
            U = adata.obsm["cnmf_usage"].copy()
            U = np.maximum(U, 1e-8)
            U /= U.sum(axis=1, keepdims=True)
            adata.obsm["cnmf_usage"] = U

            # run CRT
            result_df = run_CRT(adata, k,  output_folder, args)


        