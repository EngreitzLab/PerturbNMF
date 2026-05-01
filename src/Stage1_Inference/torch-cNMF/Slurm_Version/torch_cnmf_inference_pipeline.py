"""
torch-cNMF Inference Pipeline (Updated for torch_cnmf)

Runs the full cNMF inference workflow:
  prepare -> factorize -> combine -> k_selection_plot -> consensus -> compile -> annotate

Usage:
  python torch_cnmf_inference_pipeline.py \
    --counts_fn data/pbmc3k_raw.h5ad \
    --output_directory example_output/cNMF \
    --run_name pbmc_test \
    --species human \
    --run_factorize --run_refit --run_compile_annotation
"""

import sys
import argparse
import yaml
import os
import pandas as pd
import numpy as np
import scanpy as sc

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from torch_cnmf import cNMF

from Stage1_Inference.src import (
    run_cnmf_consensus, get_top_indices_fast, annotate_genes_to_excel,
    rename_and_move_files_NMF, rename_all_NMF, compile_results
)
from Stage1_Inference.src.plot_diagnostics import generate_all_plots



def main():
    parser = argparse.ArgumentParser(
        description="torch-cNMF inference pipeline (torch_cnmf)"
    )

    # --- I/O ---
    parser.add_argument('--counts_fn', type=str, required=True,
                        help="Path to gene expression data (.h5ad)")
    parser.add_argument('--output_directory', type=str, required=True,
                        help="Output directory for results")
    parser.add_argument('--run_name', type=str, required=True,
                        help="Name for this cNMF run")
    parser.add_argument('--nmf_seeds_path', type=str, default=None,
                        help="Path to text file with custom NMF seeds (one integer per line)")

    # --- cNMF parameters ---
    parser.add_argument('--K', nargs='*', type=int, default=None,
                        help="K values to test (default: [5, 7, 10])")
    parser.add_argument('--sel_thresh', nargs='*', type=float, default=None,
                        help="Density thresholds for consensus (default: [2.0])")
    parser.add_argument('--numiter', type=int, default=10,
                        help="NMF iterations per K")
    parser.add_argument('--densify', action='store_true',
                        help="Densify sparse matrix before factorization")
    parser.add_argument('--tpm_fn', type=str, default=None,
                        help="Pre-computed TPM file path")
    parser.add_argument('--numhvgenes', type=int, default=2000,
                        help="Number of highly variable genes")
    parser.add_argument('--genes_file', type=str, default=None,
                        help="File with gene list (overrides HVG selection)")
    parser.add_argument('--loss', default='frobenius',
                        help="NMF loss function")
    parser.add_argument('--algo', type=str, default='halsvar',
                        help="NMF algorithm: mu | hals | halsvar | bpp")
    parser.add_argument('--mode', type=str, default='batch',
                        help="Learning mode: batch | minibatch | dataloader")
    parser.add_argument('--tol', type=float, default=1e-4,
                        help="Convergence tolerance")
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help="Number of parallel jobs")
    parser.add_argument('--init', type=str, default='random',
                        help="Initialisation: random | nndsvd")
    parser.add_argument('--seed', type=int, default=14,
                        help="Random seed")
    parser.add_argument('--use_gpu', action='store_true',
                        help="Use GPU acceleration")

    # Regularisation
    parser.add_argument('--alpha_usage', type=float, default=0.0)
    parser.add_argument('--alpha_spectra', type=float, default=0.0)
    parser.add_argument('--l1_ratio_usage', type=float, default=0.0)
    parser.add_argument('--l1_ratio_spectra', type=float, default=0.0)

    # Batch-mode NMF
    parser.add_argument('--batch_max_epoch', type=int, default=500)
    parser.add_argument('--batch_hals_tol', type=float, default=0.05)
    parser.add_argument('--batch_hals_max_iter', type=int, default=200)

    # Mini-batch / dataloader mode NMF
    parser.add_argument('--minibatch_max_epoch', type=int, default=20)
    parser.add_argument('--minibatch_size', type=int, default=5000)
    parser.add_argument('--minibatch_max_iter', type=int, default=200)
    parser.add_argument('--minibatch_usage_tol', type=float, default=0.05)
    parser.add_argument('--minibatch_spectra_tol', type=float, default=0.05)
    parser.add_argument('--minibatch_shuffle', action='store_true',
                        help="Shuffle cells in minibatch mode (default: off)")

    # Precision
    parser.add_argument('--fp_precision', type=str, default='float',
                        help="Floating-point precision: float | double")

    # Refit
    parser.add_argument('--sk_cd_refit', action='store_true',
                        help="Use sklearn coordinate descent for refitting")

    # --- Annotation ---
    parser.add_argument('--species', type=str, required=True,
                        help="Species for gene annotation: human | mouse")
    parser.add_argument('--num_gene', type=int, default=300,
                        help="Number of top genes for annotation")
    parser.add_argument('--parallel_running', action='store_true',
                        help="Enable parallel processing mode for multiple K values")

    # --- Preprocessing ---
    parser.add_argument('--remove_noncoding', action='store_true',
                        help="Remove non-coding genes by Ensembl prefix")
    parser.add_argument('--ensembl_prefix', type=str, default='ENSG',
                        help="Ensembl ID prefix for non-coding filter")
    parser.add_argument('--gene_names_key', type=str, default='symbol',
                        help="Column in adata.var with gene names (used for non-coding filter and compiled results)")

    # --- Metadata keys ---
    parser.add_argument('--data_key', type=str, default='rna')
    parser.add_argument('--prog_key', type=str, default='cNMF')
    parser.add_argument('--categorical_key', type=str, default='sample')
    parser.add_argument('--guide_names_key', type=str, default='guide_names')
    parser.add_argument('--guide_targets_key', type=str, default='guide_targets')
    parser.add_argument('--guide_assignment_key', type=str, default='guide_assignment')

    # --- Step selection ---
    parser.add_argument('--run_factorize', action='store_true',
                        help="Run the factorization step")
    parser.add_argument('--run_refit', action='store_true',
                        help="Run combine + k_selection + consensus")
    parser.add_argument('--run_compile_annotation', action='store_true',
                        help="Run result compilation and gene annotation")
    parser.add_argument('--run_diagnostic_plots', action='store_true',
                        help="Generate diagnostic plots (elbow curves, usage heatmaps, loading violins)")

    args = parser.parse_args()

    # --- Defaults ---
    if args.K is None:
        args.K = [5, 7, 10]
    if args.sel_thresh is None:
        args.sel_thresh = [2.0]

    # --- Load custom seeds ---
    nmf_seeds = None
    if args.nmf_seeds_path is not None:
        nmf_seeds = np.load(args.nmf_seeds_path)

    # --- Create output directory ---
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

    # --- Optionally filter non-coding genes ---
    if args.remove_noncoding:
        adata = sc.read(args.counts_fn)
        mask = ~adata.var[args.gene_names_key].str.startswith(args.ensembl_prefix)
        adata = adata[:, mask].copy()
        filtered_path = f'{inference_dir}/adata_without_noncoding.h5ad'
        adata.write(filtered_path)
        args.counts_fn = filtered_path

    # ======================================================================
    # Initialise cNMF (torch_cnmf)
    # ======================================================================
    cnmf_obj = cNMF(output_dir=run_dir, name='Inference')

    # --- Prepare ---
    cnmf_obj.prepare(
        counts_fn=args.counts_fn,
        components=args.K,
        n_iter=args.numiter,
        densify=args.densify,
        tpm_fn=args.tpm_fn,
        num_highvar_genes=args.numhvgenes,
        genes_file=args.genes_file,
        beta_loss=args.loss,
        init=args.init,
        algo=args.algo,
        mode=args.mode,
        tol=args.tol,
        n_jobs=args.n_jobs,
        seed=args.seed,
        use_gpu=args.use_gpu,
        alpha_usage=args.alpha_usage,
        alpha_spectra=args.alpha_spectra,
        l1_ratio_usage=args.l1_ratio_usage,
        l1_ratio_spectra=args.l1_ratio_spectra,
        minibatch_usage_tol=args.minibatch_usage_tol,
        minibatch_spectra_tol=args.minibatch_spectra_tol,
        fp_precision=args.fp_precision,
        batch_max_epoch=args.batch_max_epoch,
        batch_hals_tol=args.batch_hals_tol,
        batch_hals_max_iter=args.batch_hals_max_iter,
        minibatch_max_epoch=args.minibatch_max_epoch,
        minibatch_size=args.minibatch_size,
        minibatch_max_iter=args.minibatch_max_iter,
        minibatch_shuffle=args.minibatch_shuffle,
        sk_cd_refit=args.sk_cd_refit,
        nmf_seeds=nmf_seeds,
    )

    # --- Factorize ---
    if args.run_factorize:
        cnmf_obj.factorize(skip_completed_runs=True)

    # --- Combine + K selection + Consensus ---
    if args.run_refit:
        cnmf_obj.combine()
        cnmf_obj.k_selection_plot()
        run_cnmf_consensus(cnmf_obj,
                           components=args.K,
                           density_thresholds=args.sel_thresh)

    # --- Compile results & annotate ---
    if args.run_compile_annotation:
        compile_results(run_dir, 'Inference',
                        components=args.K, sel_threshs=args.sel_thresh,
                        guide_names_key=args.guide_names_key,
                        guide_targets_key=args.guide_targets_key,
                        categorical_key=args.categorical_key,
                        guide_assignment_key=args.guide_assignment_key,
                        gene_names_key=args.gene_names_key)

        os.makedirs(f'{inference_dir}/Annotation', exist_ok=True)
        for i in args.sel_thresh:
            for k in args.K:
                df = pd.read_csv('{inference_dir}/Inference.gene_spectra_score.k_{k}.dt_{sel_thresh}.txt'.format(
                                    inference_dir=inference_dir,
                                    k=k,
                                    sel_thresh=str(i).replace('.', '_')),
                                    sep='\t', index_col=0)
                overlap = get_top_indices_fast(df, gene_num=args.num_gene)
                annotate_genes_to_excel(overlap, species=args.species,
                                        output_file=f'{inference_dir}/Annotation/{k}_{i}.xlsx')

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

    # --- Combine parallel K values ---
    if args.parallel_running:
        rename_all_NMF(source_folder=inference_dir,
                        destination_folder=f"{run_dir}/Inference_all/cnmf_tmp",
                        file_name_input='Inference',
                        file_name_output="Inference_all",
                        len=args.numiter,
                        components=args.K)

    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
