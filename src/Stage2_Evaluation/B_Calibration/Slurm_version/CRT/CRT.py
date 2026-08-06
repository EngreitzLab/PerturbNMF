
"""
Conditional Randomization Test (CRT) Calibration Pipeline

Tests differential effect of gene targets on gene programs using CRT
with optional skew-normal calibration.

Usage:
  python CRT.py \
    --out_dir /path/to/output \
    --run_name my_run
"""

import sys
import os
import argparse

import scanpy as sc
import yaml
import muon as mu
import pandas as pd
import numpy as np

# Point at B_Calibration/src so the CRT package (src/CRT/) is importable.
# insert(0, ...) — not append — so the package wins over this same-named script
# (this file is CRT.py; the interpreter puts its own dir on sys.path[0], which would
# otherwise shadow the `CRT` package and cause a circular self-import).
sys.path.insert(0, '/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/B_Calibration/src')

from CRT import (
    prepare_crt_inputs,
    build_ntc_group_inputs,
    crt_pvals_for_ntc_groups_ensemble,
    make_ntc_groups_ensemble,
    run_all_genes_union_crt,
    qq_plot_real_vs_null,
)

# Covariate independence diagnostic (correlation + VIF), run when --check_covariate is
# set. Lives in src/ (added to sys.path above), so it imports as a plain module.
from check_covariate import run_covariate_check



# reformat adata info for CRT
def reformat_data_for_CRT(mdata, covariates=None, log_covariates=None):

    adata = mdata["cNMF"].copy()
    adata.obsm["cnmf_usage"] = np.asarray(adata.X)  # ensure dense float

    # Guide assignment already lives in the cNMF modality of the input mdata
    adata.obsm["guide_assignment"] = adata.obsm["guide_assignment"].copy()

    # Program names
    adata.uns["program_names"] = list(adata.var_names)

    # Guide names must match columns of guide_assignment
    guide_names = list(adata.uns["guide_names"])
    adata.uns["guide_names"] = guide_names

    guide2gene = dict(zip(guide_names, adata.uns["guide_targets"]))
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
def run_CRT(adata, k, sel_thresh, output_folder, args):
    thresh_tag = str(sel_thresh).replace(".", "_")

    # perform CRT for each condition of cells
    for condition in adata.obs[args.categorical_key].unique():
        covar_tag = _covariate_tag(args)
        real_txt = f"{output_folder}/{k}_CRT_{covar_tag}_{condition}.txt"
        fake_txt = f"{output_folder}/{k}_CRT_fake_{covar_tag}_{condition}.txt"
        png = f"{output_folder}/{k}_CRT_{covar_tag}_{condition}.png"

        # Cache path: if both real and fake results already exist, skip the (expensive)
        # CRT recompute and just (re)generate the QQ plot from the cached raw p-values.
        if args.skip_existing and os.path.exists(real_txt) and os.path.exists(fake_txt):
            print(f"  Cached K={k}, sel_thresh={sel_thresh}, condition={condition}: "
                  f"skipping compute, plotting from cache")
            plot_qq_from_cache(real_txt, fake_txt, png, condition)
            summarize_ntc_significance_from_cache(
                real_txt, args, k, output_folder, condition, covar_tag)
            continue

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

        # Compute raw CRT p-values for each NTC group in each ensemble (null distribution)
        ntc_group_pvals_ens = crt_pvals_for_ntc_groups_ensemble(
            inputs=inputs,
            ntc_groups_ens=ntc_groups_ens,
            B=args.number_permutations,
            seed0=23,
        )

        # Save real + fake (NTC null) results (raw p-values for both, so they are comparable)
        save_result(out, k, thresh_tag, output_folder, condition, args, covar_tag)
        save_fake_result(ntc_group_pvals_ens, k, output_folder, condition, args, covar_tag)

        # Empirical-FDR calibration check: treat non-targeting controls as genes and
        # count NTC regulator-program pairs passing FDR < 0.05 (per condition).
        summarize_ntc_significance(out, args, k, output_folder, condition, covar_tag)

        # QQ plot: real raw vs null (NTC) raw p-values
        import matplotlib.pyplot as plt

        qq_plot_real_vs_null(
            real_pvals=out["pvals_raw_df"],
            null_pvals=ntc_group_pvals_ens,
            title=f"CRT QQ: real vs null (NTC) p-values for {condition}",
        )
        plt.tight_layout()
        plt.savefig(png, dpi=100)
        plt.close()


def _covariate_tag(args):
    """Build filename token from covariate args, e.g. percent_mito_log_n_counts."""
    parts = list(args.covariates or []) + [f"log_{c}" for c in (args.log_covariates or [])]
    return "_".join(parts) if parts else "no_covariates"


def _add_adj_pval(df, pcol, args, out_col='adj_pval'):
    """Add column `out_col` = FDR-corrected p-values from column `pcol` (BH or StoreyQ)."""
    if args.FDR_method == 'BH':

        from statsmodels.stats.multitest import multipletests
        df[out_col] = multipletests(df[pcol], method='fdr_bh')[1]

    elif args.FDR_method == 'StoreyQ':

        from multipy.fdr import qvalue
        import builtins
        if not hasattr(builtins, 'xrange'):
            builtins.xrange = range
        df[out_col] = qvalue(df[pcol].values, threshold=0.05, verbose=False)[1]

    return df


def _melt_programs(df, value_name):
    """Melt a genes(rows) x programs(cols) DataFrame to long (target_name, program_name, value)."""
    return df.reset_index().melt(
        id_vars='index',
        var_name='program_name',
        value_name=value_name,
    ).rename(columns={'index': 'target_name'})


# save real perturbation results
def save_result(out, k, thresh_tag, output_folder, condition, args, covar_tag=None):

    pval_long = _melt_programs(out['pvals_skew_df'], 'p-value')
    beta_long = _melt_programs(out['betas_df'], 'beta_clr')

    # Merge on program and target_gene
    result_df = pval_long.merge(
        beta_long,
        on=['program_name', 'target_name'],
        how='inner'
    )

    # Also carry the raw (uncalibrated) CRT p-value so real and NTC-null are on the
    # same scale (the null / QQ plot use raw p-values).
    has_raw = 'pvals_raw_df' in out
    if has_raw:
        raw_long = _melt_programs(out['pvals_raw_df'], 'p-value_raw')
        result_df = result_df.merge(raw_long, on=['program_name', 'target_name'], how='left')

    # The CRT effect size is beta_hat on the CLR (natural-log) scale, NOT a log2 fold
    # change. A shift of beta in CLR coordinate k corresponds to a natural-log fold
    # change of beta * K/(K-1) in program k's usage: the CLR mean absorbs 1/K of the
    # shift across the K programs, so ln-FC = beta * K/(K-1). Converting to base 2:
    #     approx_log2FC = [K / (K - 1)] * beta_hat / ln(2)
    # K = k = number of programs (CLR is centered over all program columns).
    result_df['log2FC'] = result_df['beta_clr'] * (k / (k - 1)) / np.log(2)

    # Multiple-testing correction: FDR for the skew-calibrated AND the raw p-values.
    result_df = _add_adj_pval(result_df, 'p-value', args, out_col='adj_pval')
    if has_raw:
        result_df = _add_adj_pval(result_df, 'p-value_raw', args, out_col='adj_pval_raw')

    # Reorder columns (each p-value followed by its adjusted p-value)
    cols = ['target_name', 'program_name', 'log2FC', 'p-value', 'adj_pval']
    if has_raw:
        cols += ['p-value_raw', 'adj_pval_raw']
    result_df = result_df[cols]

    if covar_tag is None:
        covar_tag = _covariate_tag(args)
    result_df.to_csv(f'{output_folder}/{k}_CRT_{covar_tag}_{condition}.txt', sep='\t', index=False)

    return result_df


# save fake / null (NTC group ensemble) raw p-values
def save_fake_result(ntc_group_pvals_ens, k, output_folder, condition, args, covar_tag=None):
    """
    Write the NTC null distribution (raw p-values only) so it can be re-analyzed /
    re-plotted offline. `ntc_group_pvals_ens` is a dict {ensemble -> DataFrame(rows=NTC
    group id, cols=programs)}. Columns: ensemble, target_name (NTC pseudo-gene id, e.g.
    'ntc_3'), program_name, p-value_raw, adj_pval_raw.
    """
    frames = []
    for e in sorted(ntc_group_pvals_ens.keys()):
        long_e = _melt_programs(ntc_group_pvals_ens[e], 'p-value_raw')
        long_e['ensemble'] = e
        frames.append(long_e)

    fake_df = pd.concat(frames, ignore_index=True)
    fake_df = _add_adj_pval(fake_df, 'p-value_raw', args, out_col='adj_pval_raw')
    fake_df = fake_df[['ensemble', 'target_name', 'program_name', 'p-value_raw', 'adj_pval_raw']]

    if covar_tag is None:
        covar_tag = _covariate_tag(args)
    fake_df.to_csv(f'{output_folder}/{k}_CRT_fake_{covar_tag}_{condition}.txt', sep='\t', index=False)

    return fake_df


# negative-control calibration: empirical FDR from non-targeting "genes"
def _ntc_significance_core(raw_long, args, k, output_folder, condition,
                           covar_tag=None, p_thresh=0.05):
    """
    Shared core for the NTC empirical-FDR check.

    ``raw_long`` is a long DataFrame with (at least) columns ``target_name`` and
    ``adj_pval_raw`` covering EVERY regulator-program pair (all genes x programs),
    where ``adj_pval_raw`` is the raw CRT p-value already FDR-corrected across that
    full family (same --FDR_method as save_result). Non-targeting / safe-targeting
    controls are the rows whose ``target_name`` is in ``args.guide_annotation_key``.

    A pair is "significant" when ``adj_pval_raw < p_thresh``. Treating each control
    label as an ordinary gene, we report the empirical false-discovery rate per
    condition two ways:

      * fraction_of_ntc_pairs = (NTC pairs with FDR < p_thresh) / (all NTC pairs)
        -- the per-condition empirical FPR/FDR among the controls; ~< p_thresh when
        the test is well calibrated.
      * empirical_fdr = (NTC pairs with FDR < p_thresh) / (all pairs with FDR <
        p_thresh, across every gene) -- classical FDR = false / total discoveries.

    Writes a per-gene summary (plus an ALL_NTC row) to
    ``{output_folder}/{k}_CRT_ntc_significance_{covar_tag}_{condition}.txt`` and
    prints both rates. Returns a dict (or None if no control target was tested).
    """
    ntc_labels = args.guide_annotation_key
    if isinstance(ntc_labels, str):
        ntc_labels = [ntc_labels]
    ntc_labels = set(ntc_labels)

    ntc_present = sorted(set(raw_long['target_name']) & ntc_labels)
    if not ntc_present:
        print(f"  [NTC significance] none of {sorted(ntc_labels)} found in tested "
              f"targets (condition={condition}); skipping.")
        return None

    raw_long = raw_long.copy()
    raw_long['significant'] = raw_long['adj_pval_raw'] < p_thresh

    # total discoveries across every regulator-program pair (all genes)
    n_sig_total_all = int(raw_long['significant'].sum())

    ntc_long = raw_long[raw_long['target_name'].isin(ntc_labels)]

    # per-control-gene breakdown
    per_gene = (
        ntc_long.groupby('target_name')
        .agg(n_significant=('significant', 'sum'), n_pairs=('significant', 'size'))
        .reset_index()
    )
    per_gene['n_significant'] = per_gene['n_significant'].astype(int)
    per_gene['fraction_of_ntc_pairs'] = per_gene['n_significant'] / per_gene['n_pairs']

    n_sig_ntc = int(ntc_long['significant'].sum())
    n_ntc_pairs = int(len(ntc_long))
    frac_ntc = n_sig_ntc / n_ntc_pairs if n_ntc_pairs else float('nan')
    empirical_fdr = n_sig_ntc / n_sig_total_all if n_sig_total_all else float('nan')

    print(f"  [NTC significance] condition={condition}: {n_sig_ntc}/{n_ntc_pairs} "
          f"NTC regulator-program pairs pass FDR < {p_thresh} "
          f"(fraction_of_ntc_pairs={frac_ntc:.4f}); "
          f"empirical_fdr={n_sig_ntc}/{n_sig_total_all}={empirical_fdr:.4f} "
          f"across {len(ntc_present)} control target(s): {ntc_present}")

    if covar_tag is None:
        covar_tag = _covariate_tag(args)

    overall = pd.DataFrame([{
        'target_name': 'ALL_NTC',
        'n_significant': n_sig_ntc,
        'n_pairs': n_ntc_pairs,
        'fraction_of_ntc_pairs': frac_ntc,
        'n_significant_all_genes': n_sig_total_all,
        'empirical_fdr': empirical_fdr,
    }])
    summary = pd.concat([per_gene, overall], ignore_index=True)
    summary.to_csv(
        f'{output_folder}/{k}_CRT_ntc_significance_{covar_tag}_{condition}.txt',
        sep='\t', index=False,
    )

    return {'n_significant': n_sig_ntc, 'n_ntc_pairs': n_ntc_pairs,
            'fraction_of_ntc_pairs': frac_ntc,
            'n_significant_all_genes': n_sig_total_all,
            'empirical_fdr': empirical_fdr,
            'per_gene': per_gene, 'p_thresh': p_thresh}


def summarize_ntc_significance(out, args, k, output_folder, condition,
                               covar_tag=None, p_thresh=0.05):
    """
    Empirical-FDR calibration check from the freshly-computed CRT results.

    Takes the RAW CRT p-values in out['pvals_raw_df'] (rows = target gene, cols =
    program), FDR-corrects them across the full family of all regulator-program
    pairs (same --FDR_method as save_result), then delegates to
    ``_ntc_significance_core`` to score the non-targeting controls. See that
    function for the reported metrics and output file.
    """
    pvals_raw = out.get('pvals_raw_df')
    if pvals_raw is None:
        raise KeyError(
            "out['pvals_raw_df'] not found — run_all_genes_union_crt must be called "
            "with return_raw_pvals=True."
        )

    raw_long = _melt_programs(pvals_raw, 'p-value_raw')
    raw_long = _add_adj_pval(raw_long, 'p-value_raw', args, out_col='adj_pval_raw')
    return _ntc_significance_core(raw_long, args, k, output_folder, condition,
                                  covar_tag=covar_tag, p_thresh=p_thresh)


def summarize_ntc_significance_from_cache(real_txt, args, k, output_folder, condition,
                                          covar_tag=None, p_thresh=0.05):
    """
    Same empirical-FDR check as summarize_ntc_significance, but reusing the cached
    real result file (which already carries adj_pval_raw FDR-corrected over the full
    family in save_result). Returns None if the cache predates the raw-p-value
    columns.
    """
    real_df = pd.read_csv(real_txt, sep='\t')
    if 'adj_pval_raw' not in real_df.columns:
        print(f"  [NTC significance] cached {os.path.basename(real_txt)} lacks "
              f"'adj_pval_raw' (condition={condition}); skipping.")
        return None
    return _ntc_significance_core(real_df[['target_name', 'adj_pval_raw']],
                                  args, k, output_folder, condition,
                                  covar_tag=covar_tag, p_thresh=p_thresh)


# regenerate the QQ plot from cached result files (no CRT recompute)
def plot_qq_from_cache(real_txt, fake_txt, png, condition):
    import matplotlib.pyplot as plt

    real_df = pd.read_csv(real_txt, sep='\t')
    fake_df = pd.read_csv(fake_txt, sep='\t')

    qq_plot_real_vs_null(
        real_pvals=real_df['p-value_raw'],
        null_pvals=fake_df['p-value_raw'],
        title=f"CRT QQ: real vs null (NTC) p-values for {condition}",
    )
    plt.tight_layout()
    plt.savefig(png, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Conditional Randomization Test (CRT) calibration pipeline"
    )

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--K', nargs='*', type=int, help="list of K values (number of components) to test", default=[30, 50, 70, 80, 100, 200, 300])
    parser.add_argument('--sel_threshs', nargs='*', type=float, help="list of density threshold values for consensus selection", default=[0.2, 2.0])

    # keys
    parser.add_argument('--categorical_key', help='Key in .obs to access cell condition/sample labels (default: sample)', type=str, default="sample")

    # Covariates
    parser.add_argument('--covariates', nargs='*', type=str, help='Covariate keys in .obs to include as-is (e.g., any continues value, categorical values)', default=None)
    parser.add_argument('--log_covariates', nargs='*', type=str, help='Covariate keys in .obs to log1p-transform before inclusion (e.g., any count based values)', default=None)

    # Covariate independence diagnostic (run before CRT)
    parser.add_argument('--check_covariate', action='store_true', help='Before running CRT, run the covariate independence diagnostic (Pearson correlation heatmap + VIF) on the encoded design matrix and write it to <Evaluation>/covariate_check/. Useful for catching collinear covariates that would make CRT p-values unreliable.')
    parser.add_argument('--numeric_as_category_threshold', type=int, default=20, help='For the covariate check: numeric covariates with <= this many unique values are one-hot encoded (matches CRT get_covar_matrix default: 20).')

    # Calibration parameters
    parser.add_argument('--number_guide', help='Number of non-targeting guides to randomly designate as "targeting" in each calibration iteration (default: 6)', type=int, default=6)
    parser.add_argument('--number_permutations', help='Number of calibration iterations to run with (default: 1024)', type=int, default=1024)
    parser.add_argument('--guide_annotation_key', nargs='*', type=str,  help='Name of target for non-targeting/safe-targeting guides,default="non-targeting"', default='non-targeting')
    parser.add_argument('--FDR_method', type=str, choices=['BH', 'StoreyQ'], default='BH', help='FDR correction method: BH (Benjamini-Hochberg) or StoreyQ (Storey Q-value) (default: BH)')
    parser.add_argument('--save_dir', type=str, default=None, help='Base directory under which {K}_{thresh} subdirs are created. Default: <out_dir>/<run_name>/Evaluation/')
    parser.add_argument('--skip_existing', help='If set, skip per-(K, sel_thresh, condition) computations whose output .txt files already exist. Useful for resuming preempted jobs.', action='store_true')

    args = parser.parse_args()

    # --- Save config (incl. SLURM info) ---
    slurm_info = {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'job_name': os.environ.get('SLURM_JOB_NAME'),
        'partition': os.environ.get('SLURM_JOB_PARTITION'),
        'node_list': os.environ.get('SLURM_JOB_NODELIST'),
        'cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'mem_per_node': os.environ.get('SLURM_MEM_PER_NODE'),
        'time_limit': os.environ.get('SLURM_JOB_TIMELIMIT'),
        'submit_dir': os.environ.get('SLURM_SUBMIT_DIR'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    job_id = slurm_info['job_id'] or 'no_jobid'

    save_config_dir = args.save_dir or f'{args.out_dir}/{args.run_name}/Evaluation'
    os.makedirs(save_config_dir, exist_ok=True)

    config_to_save = {'script_args': vars(args), 'slurm_info': slurm_info}
    with open(f'{save_config_dir}/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)

    # Optional covariate independence diagnostic. Covariates are K-independent, so run
    # it once on the first (K, sel_thresh) MuData. Per-condition, mirroring CRT, which
    # builds a separate design matrix per level of --categorical_key.
    if args.check_covariate:
        first_k = args.K[0]
        first_thresh = args.sel_threshs[0]
        first_tag = str(first_thresh).replace('.', '_')
        check_save = f'{save_config_dir}/covariate_check'
        print(f"Running covariate independence check (K={first_k}, "
              f"sel_thresh={first_thresh}) -> {check_save}")
        check_mdata = mu.read(
            f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{first_k}_{first_tag}.h5mu'
        )
        run_covariate_check(
            check_mdata["cNMF"],
            save_path=check_save,
            covariates=args.covariates,
            log_covariates=args.log_covariates,
            categorical_key=args.categorical_key,
            per_condition=True,
            numeric_as_category_threshold=args.numeric_as_category_threshold,
        )

    # run CRT for each K and dt
    for sel_thresh in args.sel_threshs:
        for k in args.K:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            thresh_tag = str(sel_thresh).replace('.', '_')
            if args.save_dir:
                output_folder = f"{args.save_dir}/{k}_{thresh_tag}"
            else:
                output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{thresh_tag}"
            os.makedirs(output_folder, exist_ok=True)

            # Load mdata
            mdata = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu')

            # Assign guide
            adata = reformat_data_for_CRT(mdata,
                                          covariates=args.covariates,
                                          log_covariates=args.log_covariates)

            # add flooring for program matrix with exceesive zeros
            U = adata.obsm["cnmf_usage"].copy()
            U = np.maximum(U, 1e-8)
            U /= U.sum(axis=1, keepdims=True)
            adata.obsm["cnmf_usage"] = U

            # run CRT
            run_CRT(adata, k, sel_thresh, output_folder, args)

    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())