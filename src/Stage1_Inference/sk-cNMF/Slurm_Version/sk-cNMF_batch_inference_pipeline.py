"""
sk-cNMF Inference Pipeline

Runs the full sk-cNMF inference workflow:
  prepare -> factorize -> combine -> k_selection_plot -> consensus -> compile -> annotate

Usage:
  python sk-cNMF_batch_inference_pipeline.py \
    --counts_fn data/pbmc3k_raw.h5ad \
    --output_directory example_output/cNMF \
    --run_name pbmc_test \
    --species human \
    --run_factorize --run_refit --run_complie_annotation
"""

import sys
import argparse
import cnmf
import yaml
import os
import pandas as pd
import numpy as np

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage1_Inference.src import (
    run_cnmf_consensus, get_top_indices_fast, annotate_genes_to_excel, \
    rename_and_move_files_NMF, rename_all_NMF, compile_results
)
from Stage1_Inference.src.plot_diagnostics import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="sk-cNMF inference pipeline"
    )

    #IO
    parser.add_argument('--counts_fn', type=str, required=True, help='Path to input counts file (e.g., .h5mu or .h5ad)')
    parser.add_argument('--output_directory', type=str, required=True, help='Directory where all outputs will be saved')
    parser.add_argument('--run_name', type=str, required=True, help='Name for this cNMF run (used for output file naming)')
    parser.add_argument('--nmf_seeds_path', type=str, help='Path to .npy file containing NMF random seeds', default = None)


    # cNMF parameters
    parser.add_argument('--numiter', type = int, default = 10, help='Number of NMF replicates to run (default: 10)')
    parser.add_argument('--numhvgenes', type = int, default = 5451, help='Number of highly variable genes to use (default: 5451)')
    parser.add_argument('--seed', type = int, default = 14, help='Random seed for reproducibility (default: 14)')
    parser.add_argument('--K', nargs='*', type=int, default=None, help='List of K values (number of components) to test. If not provided, defaults to [30, 50, 60, 80, 100, 200, 250, 300]')
    parser.add_argument('--init', type = str, default = 'random', help='Initialization method for NMF (default: random)')
    parser.add_argument('--loss', default = 'frobenius', help='Loss function for NMF (default: frobenius)')
    parser.add_argument('--algo', type = str, default = 'mu', help='Algorithm for NMF optimization (default: mu - multiplicative update)')
    parser.add_argument('--max_NMF_iter', type = int , default = 500, help='Maximum number of iterations for NMF (default: 500)')
    parser.add_argument('--tol', type = float , default = 1e4, help='Tolerance for NMF convergence (default: 1e4)')
    parser.add_argument('--sel_thresh', nargs='*', type=float, default=[2.0], help='Density threshold(s) for consensus matrix filtering. If not provided, defaults to [0.4, 0.8, 2.0]')

    # annotation parameters
    parser.add_argument('--species', type=str, required=True, help='Species for gene annotation (e.g., human, mouse)')
    parser.add_argument('--parallel_running', action="store_true", help='If set, enables parallel processing mode for combining results from multiple K values')
    parser.add_argument('--num_gene', type = int, default = 300, help='Number of top genes to use for program annotation (default: 300)')
    parser.add_argument('--run_refit', action="store_true", help='If set, run the combine and consensus steps after factorization')
    parser.add_argument('--run_complie_annotation', action="store_true", help='If set, compile results and generate gene annotations for all K values')
    parser.add_argument('--run_factorize', action="store_true", help='If set, run the NMF factorization step')
    parser.add_argument('--run_diagnostic_plots', action="store_true", help='Generate diagnostic plots (elbow curves, usage heatmaps, loading violins)')

    # keys
    parser.add_argument('--data_key', help='Key to access gene expression data in MuData object (default: rna)', type=str, default="rna")
    parser.add_argument('--prog_key', help='Key to store cNMF programs in MuData object (default: cNMF)', type=str, default="cNMF")
    parser.add_argument('--categorical_key', help='Key in .obs to access cell condition/sample labels (default: sample)', type=str, default="sample")
    parser.add_argument('--guide_names_key', help='Key in .uns to access guide names (default: guide_names)', type=str, default="guide_names")
    parser.add_argument('--guide_targets_key', help='Key in .uns to access guide target genes (default: guide_targets)', type=str, default="guide_targets")
    parser.add_argument('--guide_assignment_key', help='Key in .obsm to access guide assignment matrix (default: guide_assignment_key)', type=str, default="guide_assignment_key")
    parser.add_argument('--gene_names_key', type=str, default=None,
                        help="Column in adata.var with gene names to use in compiled results (e.g. 'symbol'). If None, uses var_names.")



    args = parser.parse_args()

    # either change the array here or run each component in parallel
    if args.K is None:
        args.K = [30, 50, 60, 80, 100, 200, 250, 300]

    if args.sel_thresh is None:
        args.sel_thresh = [0.4, 0.8, 2.0]


    # read seeds
    if args.nmf_seeds_path is not None:
        nmf_seeds = np.load(args.nmf_seeds_path)
    else:
        nmf_seeds = None


    # create output directory
    run_dir = f'{args.output_directory}/{args.run_name}'
    inference_dir = f'{run_dir}/Inference'
    os.makedirs(inference_dir, exist_ok=True)

    # --- Save config (incl. SLURM info) ---
    slurm_info = {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'job_name': os.environ.get('SLURM_JOB_NAME'),
        'partition': os.environ.get('SLURM_JOB_PARTITION'),
        'node_list': os.environ.get('SLURM_JOB_NODELIST'),
        'num_nodes': os.environ.get('SLURM_JOB_NUM_NODES'),
        'ntasks': os.environ.get('SLURM_NTASKS'),
        'cpus_per_task': os.environ.get('SLURM_CPUS_PER_TASK'),
        'mem_per_node': os.environ.get('SLURM_MEM_PER_NODE'),
        'mem_per_cpu': os.environ.get('SLURM_MEM_PER_CPU'),
        'gres': os.environ.get('SLURM_JOB_GRES'),
        'gpu_device_ids': os.environ.get('SLURM_GPUS_ON_NODE'),
        'gpu_type': os.environ.get('SLURM_JOB_GPUS'),
        'time_limit': os.environ.get('SLURM_JOB_TIMELIMIT'),
        'time_remaining': os.environ.get('SLURM_TIMELIMIT'),
        'submit_dir': os.environ.get('SLURM_SUBMIT_DIR'),
        'submit_host': os.environ.get('SLURM_SUBMIT_HOST'),
        'constraint': os.environ.get('SLURM_JOB_CONSTRAINT'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    job_id = slurm_info['job_id'] or 'no_jobid'

    config_to_save = {'script_args': vars(args), 'slurm_info': slurm_info}
    with open(f'{inference_dir}/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)

    # running cnmf
    cnmf_obj = cnmf.cNMF(output_dir=run_dir, name='Inference')

    cnmf_obj.prepare(counts_fn= args.counts_fn, components= args.K, n_iter= args.numiter,  densify=False, tpm_fn=None, seed= args.seed,
            beta_loss = args.loss,num_highvar_genes=args.numhvgenes, genes_file=None,
            alpha_usage=0.0, alpha_spectra=0.0, init=args.init, max_NMF_iter=args.max_NMF_iter, algo = args.algo, tol = args.tol, nmf_seeds = nmf_seeds)

    if args.run_factorize:

        cnmf_obj.factorize(total_workers = 1,skip_completed_runs=True)

    if args.run_refit:

        cnmf_obj.combine()

        cnmf_obj.k_selection_plot()

        # Consensus plots with all k to choose thresh
        run_cnmf_consensus(cnmf_obj,
                            components=args.K,
                            density_thresholds=args.sel_thresh)

    if args.run_complie_annotation:

        # Save all cNMF scores in separate mudata objects
        compile_results(run_dir, 'Inference', components= args.K, sel_threshs = args.sel_thresh,
        guide_names_key = args.guide_names_key, guide_targets_key = args.guide_targets_key, categorical_key= args.categorical_key,
        guide_assignment_key = args.guide_assignment_key, gene_names_key = args.gene_names_key )

        # annotation for all K
        os.makedirs(f'{inference_dir}/Annotation', exist_ok=True)

        # annotation for all K
        for i in args.sel_thresh:
            for k in args.K:
                df = pd.read_csv('{inference_dir}/Inference.gene_spectra_scores.k_{k}.dt_{sel_thresh}.txt'.format(
                                                                                        inference_dir=inference_dir,
                                                                                        k=k,
                                                                                        sel_thresh = str(i).replace('.','_')),
                                                                                        sep='\t', index_col=0)
                overlap = get_top_indices_fast(df, gene_num=args.num_gene)
                annotate_genes_to_excel(overlap, species = args.species, output_file= f'{inference_dir}/Annotation/{k}.xlsx')


    # --- Diagnostic plots ---
    if args.run_diagnostic_plots:
        print("Generating diagnostic plots...")
        generate_all_plots(
            run_dir=run_dir,
            run_name='Inference',
            K_list=args.K,
            sel_thresh_list=args.sel_thresh,
            categorical_key=args.categorical_key,
        )

    # combine the parallel ran K value into "run_name_all" file
    if args.parallel_running:

        rename_all_NMF(source_folder = inference_dir,
                                destination_folder = f"{run_dir}/Inference_all/cnmf_tmp",
                                file_name_input = 'Inference',
                                file_name_output = "Inference_all",
                                len = args.numiter,
                                components = args.K)

    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())





