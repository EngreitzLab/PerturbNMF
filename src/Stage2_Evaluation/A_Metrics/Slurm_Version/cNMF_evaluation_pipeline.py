"""
cNMF Evaluation Pipeline

Runs evaluation metrics on cNMF results:
  categorical association, perturbation association, geneset enrichment,
  trait enrichment, motif enrichment, explained variance

Usage:
  python cNMF_evaluation_pipeline.py \
    --out_dir /path/to/output \
    --run_name my_run \
    --Perform_categorical --Perform_geneset
"""

import os
import sys
import yaml

import h5py
import muon as mu
import pandas as pd
import numpy as np
import argparse
import cnmf
import scanpy as sc

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem


def _read_samples_from_h5mu(h5mu_path, data_key, categorical_key):
    """Read unique values of obs[categorical_key] from an h5mu without loading the full MuData.

    Returns a list of sample names, or None if the column cannot be read (e.g. missing path, decoding error).
    """
    try:
        with h5py.File(h5mu_path, 'r') as f:
            obs_path = f'mod/{data_key}/obs/{categorical_key}'
            if obs_path not in f:
                return None
            col = read_elem(f[obs_path])
            return list(pd.Series(col).dropna().unique())
    except Exception as e:
        print(f"  [warn] Could not read categorical column from {h5mu_path}: {e}")
        return None


# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage2_Evaluation.A_Metrics.src import (
    compute_categorical_association,
    compute_geneset_enrichment,
    compute_trait_enrichment,
    compute_perturbation_association,
    compute_explained_variance,
    compute_motif_enrichment
)


def _assign_guide(mdata, _data_guide, gene_names_key='symbol'):
    mdata['rna'].var_names = _data_guide['rna'].var[gene_names_key].astype(str)


def main():
    parser = argparse.ArgumentParser(
        description="cNMF evaluation pipeline"
    )

    #IO info
    parser.add_argument('--out_dir', help='Directory containing cNMF output files to evaluate', type=str, required=True)
    parser.add_argument('--run_name', help='Name of the cNMF run to evaluate (must match name used during inference)', type=str, required=True)
    parser.add_argument('--K', nargs='*', type=int, default=None, help='List of K values (number of components) to evaluate. If not provided, defaults to [30, 50, 60, 80, 100, 200, 250, 300]')
    parser.add_argument('--sel_thresh', nargs='*', type=float, default=None, help='List of density thresholds to evaluate. If not provided, defaults to [0.4, 0.8, 2.0]')

    # running different tests
    parser.add_argument('--Perform_categorical', action="store_true", help='If set, compute categorical association between programs and sample labels (Kruskal-Wallis test)')
    parser.add_argument('--Perform_perturbation', action="store_true", help='If set, compute perturbation association between programs and guide perturbations')
    parser.add_argument('--Perform_geneset', action="store_true", help='If set, perform gene set enrichment analysis (Reactome and GO terms) on program genes')
    parser.add_argument('--Perform_trait', action="store_true", help='If set, perform GWAS trait enrichment analysis on program genes')
    parser.add_argument('--Perform_explained_variance', action="store_true", help='If set, compute explained variance for each K value')
    parser.add_argument('--Perform_motif', action="store_true", help='If set, perform transcription factor motif enrichment analysis')

    # resources
    parser.add_argument('--X_normalized_path', type=str,  help='Path to normalized cell x gene matrix (.h5ad) from cNMF pipeline (required for explained variance)', default=None)
    parser.add_argument('--guide_annotation_path', type=str,  help='Path to tab-separated file with guide annotations including "targeting" column to identify non-targeting controls (optional)')
    parser.add_argument('--gwas_data_path', type=str,  help='Path to GWAS data file for trait enrichment analysis (required for trait enrichment)', default=None)
    parser.add_argument('--data_guide_path', type=str,  help='Path to mdata that contains additional information (optional)', default = None)

    # keys
    parser.add_argument('--data_key', help='Key to access gene expression data in MuData object (default: rna)', type=str, default="rna")
    parser.add_argument('--prog_key', help='Key to access cNMF programs in MuData object (default: cNMF)', type=str, default="cNMF")
    parser.add_argument('--categorical_key', help='Key in .obs to access cell condition/sample labels for categorical association (default: sample)', type=str, default="sample")
    parser.add_argument('--gene_names_key', type=str, help='Column in data_guide["rna"].var containing gene names (default: symbol)', default='symbol')
    parser.add_argument('--guide_names_key', help='Key in .uns to access guide names (default: guide_names)', type=str, default="guide_names")
    parser.add_argument('--guide_targets_key', help='Key in .uns to access guide target genes (default: guide_targets)', type=str, default="guide_targets")
    parser.add_argument('--guide_assignment_key', help='Key in .obsm to access guide assignment matrix (default: guide_assignment)', type=str, default="guide_assignment")
    parser.add_argument('--guide_annotation_key', nargs='*', type=str, help='name of non-targeting guide targets', default=["non-targeting"])


    parser.add_argument('--organism', help='Organism/species for enrichment analysis (default: human)', type=str, default="human")
    parser.add_argument('--FDR_method', help='Method for FDR correction in perturbation association (default: StoreyQ)', type=str, default="StoreyQ")
    parser.add_argument('--n_top', type = int, help='Number of top loaded genes use to perform enrichment test(default: 300)',  default=300)
    parser.add_argument('--use_cache', action="store_true", help='If set, load gene set libraries from cached JSON in Resources/ instead of downloading. Falls back to download if cache not found, and saves a cache copy after download.')
    parser.add_argument('--skip_existing', action="store_true", help='If set, skip metric computations whose output files already exist. Useful for resuming jobs that were preempted mid-batch.')

    args = parser.parse_args()

    if args.Perform_trait and args.gwas_data_path is None:
        parser.error("--gwas_data_path is required when --Perform_trait is set")

    # either change the array here or run each component in parallel
    if args.K is None:
        args.K = [30, 50, 60, 80, 100, 200, 250, 300]

    if args.sel_thresh is None:
        args.sel_thresh = [0.4, 0.8, 2.0]


    # create output directory
    os.makedirs((f'{args.out_dir}/{args.run_name}/Evaluation'), exist_ok=True)

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
        'time_limit': os.environ.get('SLURM_JOB_TIMELIMIT'),
        'submit_dir': os.environ.get('SLURM_SUBMIT_DIR'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }
    job_id = slurm_info['job_id'] or 'no_jobid'

    config_to_save = {'script_args': vars(args), 'slurm_info': slurm_info}
    with open(f'{args.out_dir}/{args.run_name}/Evaluation/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)


    # defind objects used in explained variance
    if args.Perform_explained_variance:
        cnmf_obj = cnmf.cNMF(output_dir=f'{args.out_dir}/{args.run_name}', name='Inference')
        X_norm = sc.read_h5ad(args.X_normalized_path)
        X = X_norm.X

    # list of non-targeting guides
    # NOTE: guide_annotation_path extracts guide NAMES (e.g. "non-targeting_00014")
    # from the annotation table index, but compute_perturbation_association expects
    # target GROUP names (e.g. "non-targeting") from mdata.uns["guide_targets"].
    # Prefer using --guide_annotation_key (default: ["non-targeting"]) for key-based access.
    if args.guide_annotation_path is not None:
        df_target = pd.read_csv(args.guide_annotation_path, sep = "\t", index_col = 0)
        df_target_non = df_target[df_target["targeting"]==False]
        reference_targets = df_target_non.index.values.tolist()
    else:
        reference_targets = args.guide_annotation_key


    # read data guide
    if args.data_guide_path is not None:
        _data_guide = mu.read(args.data_guide_path)


    for sel_thresh in args.sel_thresh:
        for k in args.K:

            output_folder = f"{args.out_dir}/{args.run_name}/Evaluation/{k}_{str(sel_thresh).replace('.','_')}"

            os.makedirs(output_folder, exist_ok=True)

            # Expected output files for skip checks (perturbation is sample-dependent, checked later)
            out_cat = f'{output_folder}/{k}_categorical_association_results.txt'
            out_posthoc = f'{output_folder}/{k}_categorical_association_posthoc.txt'
            out_geneset = f'{output_folder}/{k}_geneset_enrichment.txt'
            out_go = f'{output_folder}/{k}_GO_term_enrichment.txt'
            out_trait = f'{output_folder}/{k}_trait_enrichment.txt'
            out_expvar = f'{output_folder}/{k}_Explained_Variance.txt'
            out_expvar_sum = f'{output_folder}/{k}_Explained_Variance_Summary.txt'

            # Early-skip: if all enabled metrics already have outputs on disk, avoid the mdata load entirely.
            # For perturbation, sample names come from obs[categorical_key]; read just that column from the
            # h5mu (cheap) instead of loading the full MuData.
            if args.skip_existing:
                early_skip_ok = True
                if args.Perform_categorical and not (os.path.exists(out_cat) and os.path.exists(out_posthoc)):
                    early_skip_ok = False
                if args.Perform_geneset and not (os.path.exists(out_geneset) and os.path.exists(out_go)):
                    early_skip_ok = False
                if args.Perform_trait and not os.path.exists(out_trait):
                    early_skip_ok = False
                if args.Perform_explained_variance and not (os.path.exists(out_expvar) and os.path.exists(out_expvar_sum)):
                    early_skip_ok = False
                if args.Perform_perturbation:
                    h5mu_path = '{out_dir}/{run_name}/Inference/adata/cNMF_{k}_{sel_thresh}.h5mu'.format(
                        out_dir=args.out_dir, run_name=args.run_name, k=k,
                        sel_thresh=str(sel_thresh).replace('.', '_'))
                    samples = _read_samples_from_h5mu(h5mu_path, args.data_key, args.categorical_key)
                    if samples is None:
                        early_skip_ok = False
                    else:
                        pert_outs = [f'{output_folder}/{k}_perturbation_association_results_{samp}.txt' for samp in samples]
                        if not all(os.path.exists(p) for p in pert_outs):
                            early_skip_ok = False
                if early_skip_ok:
                    print(f"[K={k}, thresh={sel_thresh}] All requested outputs already exist; skipping iteration.")
                    continue

            # Load mdata
            mdata = mu.read('{out_dir}/{run_name}/Inference/adata/cNMF_{k}_{sel_thresh}.h5mu'.format(out_dir = args.out_dir,
                                                                                    run_name =args.run_name,
                                                                                    k=k,
                                                                                    sel_thresh = str(sel_thresh).replace('.','_')))





             # assign information
            if args.data_guide_path is not None:
                _assign_guide(mdata, _data_guide, gene_names_key=args.gene_names_key)


            # Run categorical assocation
            if args.Perform_categorical:
                if args.skip_existing and os.path.exists(out_cat) and os.path.exists(out_posthoc):
                    print(f"[K={k}, thresh={sel_thresh}] Skipping categorical (existing outputs found).")
                else:
                    results_df, posthoc_df = compute_categorical_association(mdata, prog_key=args.prog_key, categorical_key=args.categorical_key,
                                                                            pseudobulk_key=None, test='dunn', n_jobs=-1, inplace=False)

                    results_df.to_csv(out_cat, sep='\t', index=False) # This was made wide form to insert into .var of the program anndata.
                    posthoc_df.to_csv(out_posthoc, sep='\t', index=False)


            # Run perturbation assocation
            if args.Perform_perturbation:
                # Validate reference_targets against mdata guide_targets
                mdata_targets = set(mdata[args.prog_key].uns[args.guide_targets_key])
                matched_ref = mdata_targets.intersection(reference_targets)
                print(f"[K={k}] reference_targets overlap with mdata guide_targets: {matched_ref}")
                if len(matched_ref) == 0:
                    raise ValueError(
                        f"No reference_targets found in mdata guide_targets. "
                        f"reference_targets contains guide names (e.g. {reference_targets[:3]}), "
                        f"but guide_targets contains group names (e.g. {list(mdata_targets)[:3]}). "
                        f"Use --guide_annotation_key instead of --guide_annotation_path."
                    )
                for samp in mdata[args.data_key].obs[args.categorical_key].unique():
                    out_pert = '{}/{}_perturbation_association_results_{}.txt'.format(output_folder, k, samp)
                    if args.skip_existing and os.path.exists(out_pert):
                        print(f"[K={k}, thresh={sel_thresh}, samp={samp}] Skipping perturbation (existing output found).")
                        continue
                    mdata_ = mdata[mdata['rna'].obs[args.categorical_key]==samp]
                    test_stats_df = compute_perturbation_association(mdata_, prog_key=args.prog_key,
                                                                    collapse_targets=True,
                                                                    pseudobulk=False,
                                                                    reference_targets=reference_targets,
                                                                    n_jobs=-1, inplace=False, FDR_method = args.FDR_method)
                    test_stats_df.to_csv(out_pert, sep='\t', index=False)


            # Gene-set enrichment
            if args.Perform_geneset:
                if args.skip_existing and os.path.exists(out_geneset) and os.path.exists(out_go):
                    print(f"[K={k}, thresh={sel_thresh}] Skipping geneset/GO enrichment (existing outputs found).")
                else:
                    pre_res = compute_geneset_enrichment(mdata, prog_key=args.prog_key, data_key=args.data_key, prog_name=None,
                                                        organism=args.organism, library='Reactome_2022', method="fisher",
                                                        database='enrichr', n_top=args.n_top, n_jobs=-1,
                                                        inplace=False, user_geneset=None, use_loadings_gene=False,
                                                        gene_names_key=args.gene_names_key, use_cache=args.use_cache) # use_loadings_gene:use all background genes
                    pre_res.to_csv(out_geneset, sep='\t', index=False)


                    # GO Term enrichment
                    pre_res = compute_geneset_enrichment(mdata, prog_key=args.prog_key, data_key=args.data_key, prog_name=None,
                                                        organism=args.organism, library='GO_Biological_Process_2023', method="fisher",
                                                        database='enrichr', n_top=args.n_top, n_jobs=-1,
                                                        inplace=False, user_geneset=None, use_loadings_gene=False,
                                                        gene_names_key=args.gene_names_key, use_cache=args.use_cache) # use_loadings_gene: use all background genes
                    pre_res.to_csv(out_go, sep='\t', index=False)


            # Run trait enrichment
            if args.Perform_trait:
                if args.skip_existing and os.path.exists(out_trait):
                    print(f"[K={k}, thresh={sel_thresh}] Skipping trait enrichment (existing output found).")
                else:
                    pre_res_trait = compute_trait_enrichment(mdata, gwas_data=args.gwas_data_path,
                                                            prog_key=args.prog_key, prog_name=None, data_key=args.data_key,
                                                            library='OT_GWAS', n_jobs=-1, inplace=False,
                                                            key_column='trait_efos', gene_column='gene_name',
                                                            method='fisher', n_top=args.n_top, use_loadings_gene=True,
                                                            gene_names_key=args.gene_names_key) # use_loadings_gene:use all background genes inter. expressed gene
                    pre_res_trait.to_csv(out_trait, sep='\t', index=False)


            # Run motif analysis
            if args.Perform_motif:

                ''' working in progress
                fimo_thresh_enhancer = 1e-6
                fimo_thresh_promoter = 1e-4

                for samp in mdata['rna'].obs[args.categorical_key].unique():
                    for class_, thresh in [('enhancer', fimo_thresh_enhancer),
                                        ('promoter', fimo_thresh_promoter)]:

                        loci_file = '/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/scE2G_links/EnhancerPredictionsAllPutative.ForVariantOverlap.shrunk150bp_{}_{}.tsv'.format(samp, class_)
                        motif_match_df, motif_count_df, motif_enrichment_df = compute_motif_enrichment(
                            mdata,
                            prog_key='cNMF',
                            data_key='rna',
                            motif_file='/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/hocomoco_meme.meme',
                            seq_file='/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/hg38.fa',
                            loci_file=loci_file,
                            window=1000,
                            sig=thresh,
                            eps=1e-4,
                            n_top=2000,
                            n_jobs=-1,
                            inplace=False,
                            gene_names_key=args.gene_names_key
                        )

                        motif_match_df.to_csv(os.path.join(args.out_dir, f'cNMF_{class_}_pearson_topn2000_{samp}_motif_match.txt'), sep='\t', index=False)
                        motif_count_df.to_csv(os.path.join(args.out_dir, f'cNMF_{class_}_pearson_topn2000_{samp}_motif_count.txt'), sep='\t', index=False)
                        motif_enrichment_df.to_csv(os.path.join(args.out_dir, f'cNMF_{class_}_pearson_topn2000_{samp}_motif_enrichment.txt'), sep='\t', index=False)
                '''

            # Run explained variance
            if args.Perform_explained_variance:
                if args.skip_existing and os.path.exists(out_expvar) and os.path.exists(out_expvar_sum):
                    print(f"[K={k}, thresh={sel_thresh}] Skipping explained variance (existing outputs found).")
                else:
                    compute_explained_variance(cnmf_obj, X, k, output_folder = output_folder, thre = str(sel_thresh), program_name=mdata[args.prog_key].var_names )

    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

