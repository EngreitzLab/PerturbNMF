"""
Conditional Randomization Test (CRT) Calibration Pipeline

Tests the effect of each target gene on each cNMF program with the CRT package
(B_Calibration/src/CRT/, copied from Tools/CRT/src), separately per condition
(--categorical_key), and checks calibration against a non-targeting (NTC) null.
Results are written in the PerturbNMF format read by Stage 3
({K}_CRT_{covar_tag}_{condition}.txt).

Usage:
  python CRT.py \
    --out_dir /path/to/output \
    --run_name my_run
"""

import argparse
import os
import sys

import yaml

# Point at B_Calibration/src so the CRT package (src/CRT/) is importable.
# insert(0, ...) — not append — so the package wins over this same-named script
# (this file is CRT.py; the interpreter puts its own dir on sys.path[0], which would
# otherwise shadow the `CRT` package and cause a circular self-import).
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'src/Stage2_Evaluation/B_Calibration/src'))

from CRT import (
    conditional_randomized_test,
    format_results_table,
    load_modality,
    load_results_table,
    plot_qq_real_vs_null,
    prepare_crt_inputs,
    prepare_null_guide_assignment,
    split_by_condition,
)

# Covariate independence diagnostic (correlation + VIF), run when --check_covariate is
# set. Lives in src/ (added to sys.path above), so it imports as a plain module.
from check_covariate import run_covariate_check


def _covariate_tag(args):
    """Build filename token from covariate args, e.g. percent_mito_log_n_counts."""
    parts = list(args.covariates or []) + [f"log_{c}" for c in (args.log_covariates or [])]
    return "_".join(parts) if parts else "no_covariates"


def _ntc_labels(args):
    labels = args.guide_annotation_key
    return [labels] if isinstance(labels, str) else list(labels)


def h5mu_path(args, k, sel_thresh):
    thresh_tag = str(sel_thresh).replace(".", "_")
    return f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{thresh_tag}.h5mu'


# build the CRT inputs for one condition
def prepare_condition(mdata_con, args):
    """prepare_crt_inputs on one condition's MuData (tests every target except the controls)."""
    return prepare_crt_inputs(
        mdata_con,
        usage_key=args.prog_key,
        covariates=args.covariates,
        log_covariates=args.log_covariates,
        eps=args.eps,
        numeric_as_category_threshold=args.numeric_as_category_threshold,
        normalization=args.normalization,
        moi=args.moi,
        non_targeting_guide_names=_ntc_labels(args),
        number_guide=args.number_guide,
        resampled_rounds=args.number_permutations,
        extra_resampled_rounds=args.extra_resampled_rounds,
        extra_rounds_p_threshold=args.extra_rounds_p_threshold,
        seed=args.seed,
        pvalue_method=args.pvalue_method,
        propensity_penalty=None if args.propensity_penalty == "none" else args.propensity_penalty,
        control_group=args.control_group,
        matching_ratio=args.matching_ratio,
        matching_caliper=args.matching_caliper,
        matching_distance=args.matching_distance,
    )


# run one CRT (real genes + NTC null) for every condition
def run_CRT(mdata, k, sel_thresh, output_folder, args):
    covar_tag = _covariate_tag(args)

    for condition, mdata_con in split_by_condition(mdata, args.prog_key, args.categorical_key):
        n_cells = mdata_con.mod[args.prog_key].n_obs
        print(f"  K={k}, sel_thresh={sel_thresh}, condition={condition}: {n_cells:,} cells")
        real_txt = f"{output_folder}/{k}_CRT_{covar_tag}_{condition}.txt"
        fake_txt = f"{output_folder}/{k}_CRT_fake_{covar_tag}_{condition}.txt"

        obj = prepare_condition(mdata_con, args)

        # real genes (or cached results with --skip_existing)
        if args.skip_existing and os.path.exists(real_txt):
            print(f"    cached real results: {os.path.basename(real_txt)}")
            load_results_table(obj, real_txt)
        else:
            conditional_randomized_test(obj, n_jobs=args.n_jobs)
        # rewritten even when cached, so adj_pval follows --FDR_method
        format_results_table(obj, args.FDR_method).to_csv(real_txt, sep="\t", index=False)

        if args.skip_null:
            continue

        # NTC null: fake genes built from non-targeting guides, run through the same CRT
        null_obj = prepare_null_guide_assignment(
            obj,
            method=args.null_method,
            n_bins=args.n_bins,
            n_ensemble=args.n_ensemble if args.null_method == "frequency" else None,
        )
        if args.skip_existing and os.path.exists(fake_txt):
            print(f"    cached null results: {os.path.basename(fake_txt)}")
            load_results_table(null_obj, fake_txt)
        else:
            conditional_randomized_test(null_obj, n_jobs=args.n_jobs)
        fake_df = format_results_table(null_obj, args.FDR_method, include_guides=True)
        fake_df.to_csv(fake_txt, sep="\t", index=False)
        print(f"    NTC null: {fake_df['target_name'].nunique()} fake genes; "
              f"frac p<0.05 = {(fake_df['p-value'] < 0.05).mean():.3f}")

        # QQ plots: real vs NTC-null p-values (empirical, and skew-normal when used)
        plot_qq_real_vs_null(obj, null_obj, output_folder, prefix=f"{k}_CRT_{covar_tag}_{condition}",
                             per_program=args.qq_per_program)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Conditional Randomization Test (CRT) calibration pipeline"
    )

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--K', nargs='*', type=int, help="list of K values (number of components) to test", default=[30, 50, 70, 80, 100, 200, 300])
    parser.add_argument('--sel_threshs', nargs='*', type=float, help="list of density threshold values for consensus selection", default=[0.2, 2.0])
    parser.add_argument('--save_dir', type=str, default=None, help='Base directory under which {K}_{thresh} subdirs are created. Default: <out_dir>/<run_name>/Evaluation/')
    parser.add_argument('--skip_existing', help="If set, reuse the per-(K, sel_thresh, condition) {K}_CRT_*.txt / {K}_CRT_fake_*.txt tables already in the output folder instead of recomputing (tables and QQ plots are rewritten). Useful for resuming preempted jobs.", action='store_true')

    # keys
    parser.add_argument('--prog_key', type=str, default='cNMF', help='Modality with program usage (.X), obsm["guide_assignment"], uns["guide_names"/"guide_targets"] and covariates (default: cNMF)')
    parser.add_argument('--categorical_key', help="Key in .obs holding each cell's condition label (e.g. timepoint, stimulus, donor); the CRT runs separately per condition. Default: sample", type=str, default="sample")

    # Covariates
    parser.add_argument('--covariates', nargs='*', type=str, help='Covariate keys in .obs to include as-is (e.g., any continues value, categorical values)', default=None)
    parser.add_argument('--log_covariates', nargs='*', type=str, help='Covariate keys in .obs to log1p-transform before inclusion (e.g., any count based values)', default=None)
    parser.add_argument('--numeric_as_category_threshold', type=int, default=20, help='Numeric covariates with <= this many unique values are one-hot encoded (CRT design and --check_covariate; default: 20).')

    # Covariate independence diagnostic (run before CRT)
    parser.add_argument('--check_covariate', action='store_true', help='Before running CRT, run the covariate independence diagnostic (Pearson correlation heatmap + VIF) on the encoded design matrix and write it to <Evaluation>/covariate_check/. Useful for catching collinear covariates that would make CRT p-values unreliable.')

    # Model
    parser.add_argument('--moi', choices=['high', 'low'], default='high', help='high: CRT (Bernoulli resampling from a per-gene propensity model, controls = all other cells); low: permutation test against control-guide cells (default: high)')
    parser.add_argument('--normalization', choices=['row_normalize', 'clr'], default='row_normalize', help='Outcome: per-cell usage share (row_normalize) or CLR of floored usage (clr). Sets how log2FC is derived from beta (default: row_normalize)')
    parser.add_argument('--eps', type=float, default=1e-3, help='Share floor before CLR (normalization=clr only; default: 1e-3)')
    parser.add_argument('--propensity_penalty', choices=['none', 'l2', 'old_perturbnmf'], default='none', help='moi=high propensity model: none (unpenalized MLE, as sceptre), l2 (ridge), or old_perturbnmf (old fit, comparison only) (default: none)')
    parser.add_argument('--control_group', choices=['all', 'ntc_cells', 'matched'], default='all', help='Control cells per gene: all other cells, ntc_cells (cells with a control guide; moi=high), or matched (propensity-matched) (default: all)')
    parser.add_argument('--matching_ratio', type=int, default=4, help='control_group=matched: controls per treated cell (default: 4)')
    parser.add_argument('--matching_caliper', type=float, default=None, help='control_group=matched: caliper in SDs of the distance (default: none)')
    parser.add_argument('--matching_distance', choices=['propensity', 'logit'], default='propensity', help='control_group=matched: match on the propensity score or its logit')

    # Resampling / p-values
    parser.add_argument('--number_permutations', help='Stage-1 resampling rounds per gene (default: 4999)', type=int, default=4999)
    parser.add_argument('--extra_resampled_rounds', type=int, default=49999, help='Stage-2 rounds added for programs with stage-1 p < --extra_rounds_p_threshold; 0 disables (default: 49999)')
    parser.add_argument('--extra_rounds_p_threshold', type=float, default=0.05, help='Programs with stage-1 p below this get the stage-2 rounds and the skew-normal fit (default: 0.05)')
    parser.add_argument('--pvalue_method', choices=['empirical', 'skew_normal'], default='skew_normal', help='Final p-value: empirical resampling p, or sceptre-style skew-normal tail fit (default: skew_normal)')
    parser.add_argument('--FDR_method', type=str, choices=['BH', 'StoreyQ'], default='BH', help='FDR correction method: BH (Benjamini-Hochberg) or StoreyQ (Storey Q-value) (default: BH)')
    parser.add_argument('--seed', type=int, default=123, help='Base seed; each gene resamples with crc32(gene) ^ seed (default: 123)')
    parser.add_argument('--n_jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)), help='Parallel workers over genes (default: $SLURM_CPUS_PER_TASK or 1)')

    # NTC null / calibration
    parser.add_argument('--guide_annotation_key', nargs='*', type=str, help='Name(s) of the non-targeting/safe-targeting target in uns["guide_targets"], default="non-targeting"', default=['non-targeting'])
    parser.add_argument('--number_guide', help='NTC guides per fake (null) gene (default: 6)', type=int, default=6)
    parser.add_argument('--null_method', choices=['frequency', 'random'], default='frequency', help='frequency: NTC groups matched to real genes\' guide-frequency bins; random: sceptre-style random NTC guide sets (default: frequency)')
    parser.add_argument('--n_bins', type=int, default=10, help='null_method=frequency: guide-frequency bins (default: 10)')
    parser.add_argument('--n_ensemble', type=int, default=10, help='null_method=frequency: number of random NTC partitions (default: 10)')
    parser.add_argument('--skip_null', action='store_true', help='Real genes only; no NTC null, fake file or QQ plots')
    parser.add_argument('--qq_per_program', action='store_true', help='Also save QQ plots with one panel per program')
    return parser


def main():
    args = build_parser().parse_args()
    args.guide_annotation_key = _ntc_labels(args)

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

    covariate_cols = (args.covariates or []) + (args.log_covariates or [])

    # Optional covariate independence diagnostic. Covariates are K-independent, so run
    # it once on the first (K, sel_thresh) MuData. Per-condition, mirroring CRT, which
    # builds a separate design matrix per level of --categorical_key.
    if args.check_covariate:
        first_k, first_thresh = args.K[0], args.sel_threshs[0]
        check_save = f'{save_config_dir}/covariate_check'
        print(f"Running covariate independence check (K={first_k}, "
              f"sel_thresh={first_thresh}) -> {check_save}")
        check_mdata = load_modality(h5mu_path(args, first_k, first_thresh), args.prog_key, covariate_cols)
        run_covariate_check(
            check_mdata[args.prog_key],
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

            # Load the program modality (cells with a missing covariate are dropped)
            mdata = load_modality(h5mu_path(args, k, sel_thresh), args.prog_key, covariate_cols)

            # run CRT per condition
            run_CRT(mdata, k, sel_thresh, output_folder, args)

    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
