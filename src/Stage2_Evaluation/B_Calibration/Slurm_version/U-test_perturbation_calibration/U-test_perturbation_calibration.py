#!/usr/bin/env python3
"""
U-test Perturbation Calibration Analysis

This script computes perturbation association tests on real and fake (calibration) data
to evaluate the statistical properties of perturbation detection methods.
"""

import os
import sys
import yaml
import logging
import argparse

import pandas as pd
import numpy as np
import muon as mu
import cnmf
import scanpy as sc

import seaborn as sns
from matplotlib import pyplot as plt
from qmplot import qqplot

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage2_Evaluation.A_Metrics.src import (
    compute_perturbation_association,
)



# Compute Real Perturbation Tests
def compute_real_perturbation_tests():
    """Compute perturbation association tests on real data"""

    # Load guide annotations and extract non-targeting guides or use the name of non-targeting
    # NOTE: guide_annotation_path extracts guide NAMES (e.g. "non-targeting_00014")
    # from the annotation table index, but compute_perturbation_association expects
    # target GROUP names (e.g. "non-targeting") from mdata.uns["guide_targets"].
    # Prefer using --guide_annotation_key (default: ["non-targeting"]) for key-based access.
    if args.guide_annotation_path is not None:
        df_target = pd.read_csv(args.guide_annotation_path, sep="\t", index_col=0)
        df_target_non = df_target[df_target["targeting"] == False]
        reference_targets = df_target_non.index.values.tolist()
    else:
        reference_targets = args.guide_annotation_key

    test_stats_real_df = []

    # Validate once using the first K and sel_thresh
    first_k = args.components[0]
    first_thresh = args.sel_thresh[0]
    thresh_str = str(first_thresh).replace('.', '_')
    mdata_check = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu')

    mdata_targets = set(mdata_check[args.prog_key].uns[args.guide_targets_key])
    matched_ref = mdata_targets.intersection(reference_targets)
    print(f"reference_targets overlap with mdata guide_targets: {matched_ref}")
    if len(matched_ref) == 0:
        raise ValueError(
            f"No reference_targets found in mdata guide_targets. "
            f"reference_targets contains guide names (e.g. {reference_targets[:3]}), "
            f"but guide_targets contains group names (e.g. {list(mdata_targets)[:3]}). "
            f"Use --guide_annotation_key instead of --guide_annotation_path."
        )

    if args.guide_annotation_path is not None:
        n_file_guides = len(df_target)
        n_mdata_guides = len(mdata_check[args.prog_key].uns[args.guide_names_key])
        if n_file_guides != n_mdata_guides:
            raise ValueError(
                f"Guide count mismatch: annotation file has {n_file_guides} guides, "
                f"but mdata has {n_mdata_guides} guides. "
                f"Ensure the annotation file matches the mdata guide set."
            )

    del mdata_check

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"
            os.makedirs(output_folder, exist_ok=True)

            # Load mdata
            mdata = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu')

            # Run perturbation association for each sample
            for samp in mdata[args.data_key].obs[args.categorical_key].unique():
                mdata_ = mdata[mdata[args.data_key].obs[args.categorical_key] == samp]

                test_stats_df = compute_perturbation_association(
                    mdata_,
                    prog_key=args.prog_key,
                    collapse_targets=True,
                    pseudobulk=False,
                    reference_targets=reference_targets,
                    FDR_method=args.FDR_method,
                    n_jobs=-1,
                    inplace=False
                )

                # Save results
                test_stats_df.to_csv(
                    f'{output_folder}/{k}_perturbation_association_results_{samp}.txt',
                    sep='\t',
                    index=False
                )

                # Add metadata
                test_stats_df['sample'] = samp
                test_stats_df['K'] = k
                test_stats_df['sel_thresh'] = sel_thresh
                test_stats_real_df.append(test_stats_df)

    # Concatenate all results
    test_stats_real_df = pd.concat(test_stats_real_df, ignore_index=True)
    test_stats_real_df['real'] = True

    return test_stats_real_df


# Compute Fake Perturbation Tests (Calibration)
def compute_fake_perturbation_tests():
    """Calibrate tests using fake target guides"""

    # read guide annotation file to find non-targeting guide names and targets
    if args.guide_annotation_path is not None:
        guide_target = pd.read_csv(args.guide_annotation_path, sep='\t')
        non_targeting_idx = guide_target.index[guide_target.targeting == False] # targeting is col with True / False
        guide_target = guide_target.loc[non_targeting_idx] # subset only non-targeting/safe-targeting guides
        guide_target['type'] = "non-targeting" # set both safe-targeting and non-targeting to be non-targeting
    else:
        guide_target = None
        non_targeting_idx = None

    test_stats_fake_dfs = []

    # Extract guide info from mdata if no annotation file provided
    if guide_target is None:
        first_k = args.components[0]
        thresh_str = str(args.sel_thresh[0]).replace('.', '_')
        mdata_check = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu')

        guide_targets_arr = mdata_check[args.prog_key].uns[args.guide_targets_key]
        guide_names_arr = mdata_check[args.prog_key].uns[args.guide_names_key]
        non_targeting_idx = np.where(guide_targets_arr == 'non-targeting')[0]

        print(f'  Non-targeting guides: {len(non_targeting_idx)}')

        # make a new annotation df when we don't have annotation file 
        guide_target = pd.DataFrame({
            args.guide_names_key: guide_names_arr[non_targeting_idx],
            'type': 'non-targeting'
        })
        del mdata_check

    # Validate guide count between annotation file and mdata
    if args.guide_annotation_path is not None:
        first_k = args.components[0]
        thresh_str = str(args.sel_thresh[0]).replace('.', '_')
        mdata_check = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{first_k}_{thresh_str}.h5mu')

        n_file_guides = len(pd.read_csv(args.guide_annotation_path, sep='\t'))
        n_mdata_guides = len(mdata_check[args.prog_key].uns[args.guide_names_key])
        
        if n_file_guides != n_mdata_guides:
            raise ValueError(
                f"Guide count mismatch: annotation file has {n_file_guides} guides, "
                f"but mdata has {n_mdata_guides} guides. "
                f"Ensure the annotation file matches the mdata guide set."
            )
        del mdata_check

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            # Load mdata
            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"
            os.makedirs(output_folder, exist_ok=True)

            mdata = mu.read(
                f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu'
            )

            test_stats_fake_dfs_temp = []
            for i in range(args.number_run):
                print(f"  Running iteration {i+1}/{args.number_run}")

                # Randomly make number_guide non-targeting guides "targeting"
                guide_target_ = guide_target.copy()
                selected_guides = np.random.choice(guide_target_[args.guide_names_key],args.number_guide,replace=False)
                guide_target_.loc[guide_target_.guide_names.isin(selected_guides),'type'] = 'targeting' # type is a col with targeting / non-targeting

                # Filter to only non-targeting guides that exist in both datasets
                valid_guide_mask = np.isin(mdata[args.prog_key].uns[args.guide_names_key],guide_target_[args.guide_names_key].values)
                valid_indices = np.where(valid_guide_mask)[0]

                print(f"  Found {len(valid_indices)} valid non-targeting guides out of " f"{len(mdata[args.prog_key].uns[args.guide_names_key])} total")

                if len(valid_indices) == 0:
                    raise ValueError("No valid guides found")

                _mdata = mdata.copy()
                _mdata[args.prog_key].obsm[args.guide_assignment_key] = mdata[args.prog_key].obsm[args.guide_assignment_key][:, non_targeting_idx]
                _mdata[args.prog_key].uns[args.guide_names_key] = mdata[args.prog_key].uns[args.guide_names_key][non_targeting_idx]
                _mdata[args.prog_key].uns[args.guide_targets_key] = guide_target_.loc[guide_target_.guide_names.isin(mdata[args.prog_key].uns[args.guide_names_key]),'type'].values

                # Run perturbation association for each sample
                for samp in _mdata[args.data_key].obs[args.categorical_key].unique():
                    mdata_samp = _mdata[_mdata[args.data_key].obs[args.categorical_key] == samp]

                    test_stats_df = compute_perturbation_association(
                        mdata_samp,
                        prog_key=args.prog_key,
                        collapse_targets=True,
                        pseudobulk=False,
                        reference_targets=args.guide_annotation_key,
                        FDR_method=args.FDR_method,
                        n_jobs=-1,
                        inplace=False
                    )


                    test_stats_df[args.categorical_key] = samp
                    test_stats_df['K'] = k
                    test_stats_df['run'] = i
                    test_stats_df['sel_thresh'] = sel_thresh
                    test_stats_fake_dfs.append(test_stats_df) # combine all
                    test_stats_fake_dfs_temp.append(test_stats_df) # combine for each k and sel_thresh

            # Save results per condition
            test_stats_fake_dfs_temp = pd.concat(test_stats_fake_dfs_temp, ignore_index=True)
            for samp, samp_df in test_stats_fake_dfs_temp.groupby(args.categorical_key):
                samp_df.to_csv(
                    f'{output_folder}/{k}_fake_perturbation_association_results_{samp}.txt',
                    sep='\t',
                    index=False
                )


    # Concatenate all results
    test_stats_fake_dfs = pd.concat(test_stats_fake_dfs, ignore_index=True)

    test_stats_fake_dfs['real'] = False

    return test_stats_fake_dfs


# load results
def load_real_perturbation_tests():
    """Load pre-computed real perturbation test results"""

    # Discover sample names from the first K directory
    first_k = args.components[0]
    thresh_str = str(args.sel_thresh[0]).replace('.', '_')
    first_dir = f'{args.out_dir}/{args.run_name}/Evaluation/{first_k}_{thresh_str}'
    sample_files = [f for f in os.listdir(first_dir) if f.startswith(f'{first_k}_perturbation_association_results_') and f.endswith('.txt')]
    samples = [f.replace(f'{first_k}_perturbation_association_results_', '').replace('.txt', '') for f in sample_files]
    print(f"Discovered samples: {samples}")

    test_stats_real_df = []

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            for samp in samples:
                thresh_str = str(sel_thresh).replace('.', '_')
                test_stats_df_ = pd.read_csv(
                    f'{args.out_dir}/{args.run_name}/Evaluation/{k}_{thresh_str}/{k}_perturbation_association_results_{samp}.txt',
                    sep='\t'
                )
                test_stats_df_['sample'] = samp
                test_stats_df_['K'] = k
                test_stats_df_['sel_thresh'] = sel_thresh
                test_stats_real_df.append(test_stats_df_)

    test_stats_real_df = pd.concat(test_stats_real_df, ignore_index=True)
    test_stats_real_df['real'] = True

    return test_stats_real_df


def load_fake_perturbation_tests():
    """Load pre-computed fake perturbation test results (per condition)"""

    # Discover sample names from the first K directory
    first_k = args.components[0]
    thresh_str = str(args.sel_thresh[0]).replace('.', '_')
    first_dir = f'{args.out_dir}/{args.run_name}/Evaluation/{first_k}_{thresh_str}'
    sample_files = [f for f in os.listdir(first_dir) if f.startswith(f'{first_k}_fake_perturbation_association_results_') and f.endswith('.txt')]
    samples = [f.replace(f'{first_k}_fake_perturbation_association_results_', '').replace('.txt', '') for f in sample_files]
    print(f"Discovered samples for fake tests: {samples}")

    test_stats_fake_df = []

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            for samp in samples:
                thresh_str = str(sel_thresh).replace('.', '_')
                test_stats_df_ = pd.read_csv(
                    f'{args.out_dir}/{args.run_name}/Evaluation/{k}_{thresh_str}/{k}_fake_perturbation_association_results_{samp}.txt',
                    sep='\t'
                )
                test_stats_fake_df.append(test_stats_df_)

    test_stats_fake_df = pd.concat(test_stats_fake_df, ignore_index=True)
    test_stats_fake_df['real'] = False

    return test_stats_fake_df


# Visualization
def plot_calibration_comparison(test_stats_dfs):
    """Save a violin plot comparing real vs fake perturbation tests for each (K, sel_thresh) into its own folder"""

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            thresh_str = str(sel_thresh).replace('.', '_')
            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{thresh_str}"
            os.makedirs(output_folder, exist_ok=True)

            test_stats_k = test_stats_dfs[
                (test_stats_dfs.K == k) & (test_stats_dfs.sel_thresh == sel_thresh)
            ].copy()

            if len(test_stats_k) == 0:
                print(f"  Skipping K={k}, sel_thresh={sel_thresh}: no data")
                continue

            test_stats_k['neg_log_pval'] = test_stats_k['pval'].apply(lambda x: -np.log(x))

            fig, ax = plt.subplots(figsize=(6, 5))

            sns.violinplot(
                x='sample',
                y='neg_log_pval',
                hue='real',
                data=test_stats_k,
                ax=ax
            )

            ax.set_title(f'K={k}, sel_thresh={sel_thresh}')
            ax.set_xlabel('Sample ID')
            ax.set_ylabel('-ln(p-value)')
            ax.set_ylim(0, 50)
            ax.axhline(8, color='grey', linestyle='dashed')

            plt.tight_layout()
            plt.savefig(f'{output_folder}/U_test_perturbation_association_calibration.png', dpi=100)
            plt.close(fig)


def plot_qq_comparison(test_stats_dfs):
    """Save a QQ plot comparing real and null distributions for each (K, sel_thresh) into its own folder"""

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            thresh_str = str(sel_thresh).replace('.', '_')
            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{thresh_str}"
            os.makedirs(output_folder, exist_ok=True)

            real_pvals = test_stats_dfs.loc[
                (test_stats_dfs.K == k) &
                (test_stats_dfs.sel_thresh == sel_thresh) &
                (test_stats_dfs.real == True),
                'pval'
            ]

            null_pvals = test_stats_dfs.loc[
                (test_stats_dfs.K == k) &
                (test_stats_dfs.sel_thresh == sel_thresh) &
                (test_stats_dfs.real == False) &
                (test_stats_dfs.target_name == 'targeting'),
                'pval'
            ]

            if len(real_pvals) == 0 and len(null_pvals) == 0:
                print(f"  Skipping K={k}, sel_thresh={sel_thresh}: no data")
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            qqplot(data=real_pvals, ax=ax, color='blue', label='Real')
            qqplot(data=null_pvals, ax=ax, color='red', label='Null')

            lines = ax.get_lines()
            all_x = np.concatenate([line.get_xdata() for line in lines])
            all_y = np.concatenate([line.get_ydata() for line in lines])
            padding = 0.05
            x_range = all_x.max() - all_x.min()
            y_range = all_y.max() - all_y.min()
            ax.set_xlim(all_x.min() - padding * x_range, all_x.max() + padding * x_range)
            ax.set_ylim(all_y.min() - padding * y_range, all_y.max() + padding * y_range)

            ax.set_title(f'K={k}, sel_thresh={sel_thresh}')
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{output_folder}/U_test_perturbation_association_qqplot.png', dpi=100)
            plt.close(fig)


def main():
    global args, mdata_guide

    parser = argparse.ArgumentParser(
        description="U-test perturbation calibration analysis"
    )

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--components', nargs='*', type=int, help = "list of K values (number of components) to test (default: [30, 50, 60, 80, 100, 200, 250, 300])", default=None)
    parser.add_argument('--sel_thresh', nargs='*', type=float, help = "list of density threshold values for consensus selection (default: [0.4, 0.8, 2.0])", default=None)

    # resources
    parser.add_argument('--mdata_guide_path', type=str,  help='Path to MuData object (.h5mu) containing guide assignment information (optional if guide info already in h5mu)', default=None)
    parser.add_argument('--guide_annotation_path', type=str,  help='Path to tab-separated file with guide annotations including "targeting" column (True/False) to identify non-targeting guides for calibration')
    parser.add_argument('--guide_annotation_key', nargs='+', type=str, help='Name of target for non-targeting/safe-targeting guides, default="non-targeting"', default=['non-targeting'])
    parser.add_argument('--reference_gtf_path', type=str,  help='Path to reference GTF file for validating gene names during format checking (optional)')

    # keys
    parser.add_argument('--data_key', help='Key to access gene expression data in MuData object (default: rna)', type=str, default="rna")
    parser.add_argument('--prog_key', help='Key to access cNMF programs in MuData object (default: cNMF)', type=str, default="cNMF")
    parser.add_argument('--categorical_key', help='Key in .obs to access cell condition/sample labels (default: sample)', type=str, default="sample")
    parser.add_argument('--guide_names_key', help='Key in .uns to access guide names (default: guide_names)', type=str, default="guide_names")
    parser.add_argument('--guide_targets_key', help='Key in .uns to access guide target genes (default: guide_targets)', type=str, default="guide_targets")
    parser.add_argument('--guide_assignment_key', help='Key in .obsm to access guide assignment matrix (default: guide_assignment)', type=str, default="guide_assignment")
    parser.add_argument('--organism', help='Organism/species for analysis (default: human)', type=str, default="human")
    parser.add_argument('--FDR_method', help='Method for FDR correction in real perturbation tests (default: StoreyQ)', type=str, default="StoreyQ")

    # check format
    parser.add_argument('--check_format', help='If set, validate MuData format and check for all necessary keys before running calibration', action="store_true")

    # Calibration parameters
    parser.add_argument('--number_run', help='Number of calibration iterations to run with randomly selected fake targeting guides (default: 300)', type=int, default=300)
    parser.add_argument('--number_guide', help='Number of non-targeting guides to randomly designate as "targeting" in each calibration iteration (default: 6)', type=int, default=6)
    parser.add_argument('--compute_real_perturbation_tests', help='If set, compute perturbation association tests on real targeting guides', action="store_true")
    parser.add_argument('--compute_fake_perturbation_tests', help='If set, compute perturbation association tests on fake targeting guides (calibration null distribution)', action="store_true")
    parser.add_argument('--visualizations', help='If set, generate and save QQ plots and violin plots comparing real vs null distributions', action="store_true")

    args = parser.parse_args()

    # either change the array here or run each component in parallel
    if args.components is None:
        args.components = [30, 50, 60, 80, 100, 200, 250, 300]

    if args.sel_thresh is None:
        args.sel_thresh = [0.4, 0.8, 2.0]

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

    config_to_save = {'script_args': vars(args), 'slurm_info': slurm_info}
    with open(f'{args.out_dir}/{args.run_name}/Evaluation/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)


    print("=" * 80)
    print("U-test Perturbation Calibration Analysis")
    print("=" * 80)

    # Compute real perturbation tests (per-(K, sel_thresh, sample) files saved inside)
    if args.compute_real_perturbation_tests:
        test_stats_real_df = compute_real_perturbation_tests()

    # Compute fake perturbation tests (per-(K, sel_thresh, sample) files saved inside)
    if args.compute_fake_perturbation_tests:
        test_stats_fake_df = compute_fake_perturbation_tests()

    # Load Merge datasets for visualizations
    if args.visualizations:

        # Load results if not computed in this run
        if not args.compute_real_perturbation_tests:
            print("\nLoading pre-computed real perturbation results...")
            test_stats_real_df = load_real_perturbation_tests()
        if not args.compute_fake_perturbation_tests:
            print("\nLoading pre-computed fake perturbation results...")
            test_stats_fake_df = load_fake_perturbation_tests()

        print("\nMerging datasets...")
        test_stats_dfs = pd.concat([test_stats_real_df, test_stats_fake_df], ignore_index=True)

        # Create visualizations
        print("\nCreating visualizations...")

        plot_calibration_comparison(test_stats_dfs)

        plot_qq_comparison(test_stats_dfs)

    print("\nPipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
