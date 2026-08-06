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

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

# Point at B_Calibration/src so the U_test package (src/U_test/) is importable.
sys.path.insert(0, '/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/B_Calibration/src')

from U_test import (
    compute_real_perturbation_tests,
    compute_fake_perturbation_tests,
    load_real_perturbation_tests,
    load_fake_perturbation_tests,
    plot_calibration_comparison,
    plot_qq_comparison,
)


def main():
    global args, mdata_guide

    parser = argparse.ArgumentParser(
        description="U-test perturbation calibration analysis"
    )

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files for calibration analysis', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to perform calibration on (must match name used during inference)', type=str, required=True)
    parser.add_argument('--K', nargs='*', type=int, help="list of K values (number of components) to test", default=[30, 50, 70, 80, 100, 200, 300])
    parser.add_argument('--sel_threshs', nargs='*', type=float, help="list of density threshold values for consensus selection", default=[0.2, 2.0])

    # resources
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
    parser.add_argument('--skip_existing', help='If set, skip per-(K, sel_thresh, sample) computations whose output files already exist. Existing outputs are loaded into the accumulator so visualizations still work. Useful for resuming preempted jobs.', action="store_true")

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

    config_to_save = {'script_args': vars(args), 'slurm_info': slurm_info}
    with open(f'{args.out_dir}/{args.run_name}/Evaluation/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)


    print("=" * 80)
    print("U-test Perturbation Calibration Analysis")
    print("=" * 80)

    # Compute real perturbation tests (per-(K, sel_thresh, sample) files saved inside)
    if args.compute_real_perturbation_tests:
        test_stats_real_df = compute_real_perturbation_tests(
            out_dir=args.out_dir,
            run_name=args.run_name,
            K=args.K,
            sel_threshs=args.sel_threshs,
            prog_key=args.prog_key,
            data_key=args.data_key,
            categorical_key=args.categorical_key,
            guide_targets_key=args.guide_targets_key,
            guide_names_key=args.guide_names_key,
            guide_annotation_path=args.guide_annotation_path,
            guide_annotation_key=args.guide_annotation_key,
            FDR_method=args.FDR_method,
            skip_existing=args.skip_existing,
        )

    # Compute fake perturbation tests (per-(K, sel_thresh, sample) files saved inside)
    if args.compute_fake_perturbation_tests:
        test_stats_fake_df = compute_fake_perturbation_tests(
            out_dir=args.out_dir,
            run_name=args.run_name,
            K=args.K,
            sel_threshs=args.sel_threshs,
            prog_key=args.prog_key,
            data_key=args.data_key,
            categorical_key=args.categorical_key,
            guide_names_key=args.guide_names_key,
            guide_targets_key=args.guide_targets_key,
            guide_assignment_key=args.guide_assignment_key,
            guide_annotation_path=args.guide_annotation_path,
            guide_annotation_key=args.guide_annotation_key,
            number_run=args.number_run,
            number_guide=args.number_guide,
            FDR_method=args.FDR_method,
            skip_existing=args.skip_existing,
        )

    # Load Merge datasets for visualizations
    if args.visualizations:

        # Load results if not computed in this run
        if not args.compute_real_perturbation_tests:
            print("\nLoading pre-computed real perturbation results...")
            test_stats_real_df = load_real_perturbation_tests(
                out_dir=args.out_dir, run_name=args.run_name, K=args.K, sel_threshs=args.sel_threshs,
            )
        if not args.compute_fake_perturbation_tests:
            print("\nLoading pre-computed fake perturbation results...")
            test_stats_fake_df = load_fake_perturbation_tests(
                out_dir=args.out_dir, run_name=args.run_name, K=args.K, sel_threshs=args.sel_threshs,
            )

        print("\nMerging datasets...")
        test_stats_dfs = pd.concat([test_stats_real_df, test_stats_fake_df], ignore_index=True)

        # Create visualizations
        print("\nCreating visualizations...")

        plot_calibration_comparison(
            test_stats_dfs, out_dir=args.out_dir, run_name=args.run_name, K=args.K, sel_threshs=args.sel_threshs,
        )

        plot_qq_comparison(
            test_stats_dfs, out_dir=args.out_dir, run_name=args.run_name, K=args.K, sel_threshs=args.sel_threshs,
        )

    print("\nPipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
