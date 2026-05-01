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

from Evaluation.src import (
    compute_perturbation_association,
)


# Helper method to reformat data 
def _assign_guide(mdata, mdata_guide):
    mdata[args.data_key].obsm[args.guide_assignment_key] = mdata[args.data_key].obsm[args.guide_assignment_key].toarray()
    mdata[args.prog_key].obsm[args.guide_assignment_key] = mdata[args.prog_key].obsm[args.guide_assignment_key].toarray()


# Compute Real Perturbation Tests
def compute_real_perturbation_tests():
    """Compute perturbation association tests on real data"""

    # Load guide annotations and extract non-targeting guides or use the name of non-targeting 
    if args.guide_annotation_path is not None:
        df_target = pd.read_csv(args.guide_annotation_path, sep="\t", index_col=0)
        df_target_non = df_target[df_target["targeting"] == False]
        reference_targets = df_target_non.index.values.tolist()
    else:
        reference_targets = args.guide_annotation_key

    test_stats_real_df = []

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"
            os.makedirs(output_folder, exist_ok=True)

            # Load mdata
            mdata = mu.read(f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu')

            # Assign guide
            _assign_guide(mdata, mdata_guide)

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
    guide_target = pd.read_csv(args.guide_annotation_path, sep='\t')
    non_targeting_idx = guide_target.index[guide_target.targeting == False] # targeting is col with True / False 
    guide_target = guide_target.loc[non_targeting_idx] # subset only non-targeting/safe-targeting guides 
    guide_target.type = "non-targeting" # set both safe-targeting and non-targeting to be non-targeting 

    test_stats_fake_dfs = []

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            print(f"Processing K={k}, sel_thresh={sel_thresh}")

            # Load mdata
            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"
            os.makedirs(output_folder, exist_ok=True)

            mdata = mu.read(
                f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{k}_{str(sel_thresh).replace(".","_")}.h5mu'
            )

            # Assign guide
            _assign_guide(mdata, mdata_guide)
            
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
                        reference_targets=args.reference_targets,
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

            # Save results
            test_stats_fake_dfs_temp = pd.concat(test_stats_fake_dfs_temp, ignore_index=True)
            test_stats_fake_dfs_temp.to_csv(
                f'{output_folder}/{k}_fake_perturbation_association_results.txt',
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

    test_stats_real_df = []

    for sel_thresh in args.sel_thresh:
        for k in args.components:
            for samp in ['D0', 'D4', 'D7']:  # Update with actual sample names
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


# Visualization
def plot_calibration_comparison(test_stats_dfs, output_path=None):
    """Create violin plots comparing real vs fake perturbation tests"""

    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 8))

    for i, k in enumerate(args.components):
        if i >= len(axs.flat) - 1:
            break

        test_stats_k = test_stats_dfs[test_stats_dfs.K == k].copy()
        test_stats_k['neg_log_pval'] = test_stats_k['pval'].apply(lambda x: -np.log(x))

        sns.violinplot(
            x='sample',
            y='neg_log_pval',
            hue='real',
            data=test_stats_k,
            ax=axs.flat[i]
        )

        axs.flat[i].set_title(f'K={k}')
        axs.flat[i].set_xlabel('Sample ID')
        axs.flat[i].set_ylabel('-ln(p-value)')
        axs.flat[i].set_ylim(0, 50)
        axs.flat[i].axhline(8, color='grey', linestyle='dashed')

    axs.flat[-1].axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100)

    return fig


def plot_qq_comparison(test_stats_dfs, output_path=None):
    """Create QQ plots comparing real and null distributions"""

    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 8))

    for i, k in enumerate(args.components):
        if i >= len(axs.flat) - 1:
            break

        ax = axs.flat[i]

        # Real data
        real_pvals = test_stats_dfs.loc[
            (test_stats_dfs.K == k) & (test_stats_dfs.real == True),
            'pval'
        ]

        # Null data
        null_pvals = test_stats_dfs.loc[
            (test_stats_dfs.K == k) &
            (test_stats_dfs.real == False) &
            (test_stats_dfs.target_name == 'targeting'),
            'pval'
        ]

        # Plot both
        qqplot(data=real_pvals, ax=ax, color='blue', label='Real')
        qqplot(data=null_pvals, ax=ax, color='red', label='Null')

        # Adjust axis limits
        lines = ax.get_lines()
        all_x = np.concatenate([line.get_xdata() for line in lines])
        all_y = np.concatenate([line.get_ydata() for line in lines])
        padding = 0.05

        x_range = all_x.max() - all_x.min()
        y_range = all_y.max() - all_y.min()
        ax.set_xlim(all_x.min() - padding * x_range, all_x.max() + padding * x_range)
        ax.set_ylim(all_y.min() - padding * y_range, all_y.max() + padding * y_range)

        ax.set_title(f'K={k}')
        ax.legend()

    axs.flat[-1].axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=100)

    return fig


# Main Execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--components', nargs='*', type=int, help = "list of K values (number of components) to test (default: [30, 50, 60, 80, 100, 200, 250, 300])", default=None)
    parser.add_argument('--sel_thresh', nargs='*', type=float, help = "list of density threshold values for consensus selection (default: [0.4, 0.8, 2.0])", default=None)

    # resources
    parser.add_argument('--mdata_guide_path', type=str,  help='Path to MuData object (.h5mu) containing guide assignment information', required=True)
    parser.add_argument('--guide_annotation_path', type=str,  help='Path to tab-separated file with guide annotations including "targeting" column (True/False) to identify non-targeting guides for calibration')
    parser.add_argument('--guide_annotation_key', type=str,  help='Name of target for non-targeting/safe-targeting guides,default="non-targeting"', default= ['non-targeting'])
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

    # save comfigs used         
    args_dict = vars(args)
    job_id = os.environ.get('SLURM_JOB_ID')
    with open(f'{args.out_dir}/{args.run_name}/Evaluation/config_{job_id}.yml', 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False, width=1000)


    print("=" * 80)
    print("U-test Perturbation Calibration Analysis")
    print("=" * 80)


    # Load guide data
    mdata_guide = mu.read(args.mdata_guide_path)

    # Compute real perturbation tests
    if args.compute_real_perturbation_tests:

        test_stats_real_df = compute_real_perturbation_tests()

    # Compute fake perturbation tests
    if args.compute_fake_perturbation_tests:

        test_stats_fake_df = compute_fake_perturbation_tests()

    # Load Merge datasets for visualizations

    if args.visualizations:

        print("\nMerging datasets...")
        test_stats_dfs = pd.concat([test_stats_real_df, test_stats_fake_df], ignore_index=True) 

        # Create visualizations
        print("\nCreating visualizations...")


        plot_calibration_comparison(
            test_stats_dfs,
            output_path=f'{args.out_dir}/{args.run_name}/perturbation_association_calibration.png'
        )

        plot_qq_comparison(
            test_stats_dfs,
            output_path=f'{args.out_dir}/{args.run_name}/perturbation_association_qqplot_overlap.png'
        )

    print("\nAnalysis complete!")
