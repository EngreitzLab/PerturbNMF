"""
cNMF Excel Summarization

Compiles Stage 1 (.h5mu) and Stage 2 (Evaluation) outputs into a single
multi-sheet Excel summary workbook for one (K, sel_thresh):
  program loadings, GO / geneset / trait enrichment, perturbation association
  (merged with specificity scores), categorical association, explained variance,
  per-target and per-program summaries.

Mirrors the reference notebook flow (JupterNote_Version/cNMF_compile_excel_table.ipynb).

Usage:
  python cNMF_excel_summary.py \
    --out_dir /path/to/output \
    --run_name my_run \
    --K 50 --sel_thresh 0.2 \
    --Sample D0 D1 D2 D3
"""

import os
import sys
import yaml

import muon as mu
import pandas as pd
import argparse

# Change path to wherever you have repo locally
sys.path.append('/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src')

from Stage3_Interpretation.B_Summarization.src import (
    compile_Program_loading_score_sheet_long,
    compile_Program_loading_score_sheet_flat,
    Compile_GO_sheet,
    Compile_Geneset_sheet,
    Compile_Trait_sheet,
    Compile_Perturbation_sheet,
    Compile_Association_sheet,
    Compile_Explained_variance,
    Compile_Target_Summary_sheet,
    Compile_Summary_sheet,
    add_specificity_scores_file,
    check_program_name_match,
)

# Excel hard limit is 1,048,576 rows incl. header; keep one below for the header row.
MAX_ROWS = 1048575


def main():
    parser = argparse.ArgumentParser(
        description="Compile Stage 1 (.h5mu) + Stage 2 (Evaluation) outputs into a "
                    "single multi-sheet Excel summary workbook for one (K, sel_thresh).")

    # IO info
    parser.add_argument('--out_dir', help='Output root directory (contains {run_name}/).', type=str, required=True)
    parser.add_argument('--run_name', help='Run name (subdirectory under out_dir).', type=str, required=True)
    parser.add_argument('--K', help='Number of components (K) to summarize.', type=int, required=True)
    parser.add_argument('--sel_thresh', help='Density selection threshold (dot->underscore handled internally). Default: 0.2', type=float, default=0.2)

    # Optional paths (derived from out_dir/run_name/K/thresh if not given)
    parser.add_argument('--save_path', type=str, default=None,
        help='Output directory for the .xlsx + TSV + sidecar files. '
             'Default: {out_dir}/{run_name}/Interpretation/Summary_table/{K}_{thresh}')
    parser.add_argument('--mdata_path', type=str, default=None,
        help='Path to the Stage 1 .h5mu. '
             'Default: {out_dir}/{run_name}/Inference/adata/cNMF_{K}_{thresh}.h5mu')

    # Compile options
    parser.add_argument('--num_gene', type=int, default=300,
        help='Number of top genes per program / per enriched term to keep. Default: 300')
    parser.add_argument('--perturbation_file_name', type=str, default='perturbation_association_results',
        help='Perturbation result file stem (between "{K}_" and "_{Condition}.txt"). '
             'Default: perturbation_association_results')
    parser.add_argument('--Sample', nargs='*', type=str,
        default=['D0', 'sample_D1', 'sample_D2', 'sample_D3'],
        help='List of condition / sample labels. Default: D0 sample_D1 sample_D2 sample_D3')
    parser.add_argument('--effect_size', type=str, default='log2FC',
        help='Effect-size column in perturbation files. Default: log2FC')
    parser.add_argument('--control_target_name', type=str, default='non-targeting',
        help='Control target name for KD efficiency. Default: non-targeting')


    # keys
    parser.add_argument('--categorical_key', type=str, default='sample',
        help='obs column holding condition/sample labels. Default: sample')
    parser.add_argument('--prog_key', type=str, default='cNMF',
        help='Modality key for cNMF programs in the MuData. Default: cNMF')
    parser.add_argument('--data_key', type=str, default='rna',
        help='Modality key for RNA expression in the MuData. Default: rna')
    parser.add_argument('--guide_targets_key', type=str, default='guide_targets',
        help='uns key for guide target names. Default: guide_targets')
    parser.add_argument('--gene_names_key', type=str, default='symbol',
        help='var column with gene symbols (for loadings / KD efficiency). Default: symbol')
    parser.add_argument('--adjusted_pval_key', type=str, default='Adjusted P-value',
        help='Adjusted p-value column in enrichment files. Default: "Adjusted P-value"')
    parser.add_argument('--non_targeting_key', nargs='*', type=str, default=['non-targeting'],
        help='Control target name(s) for the program summary sheet. Default: non-targeting')

    # enrichment / perturbation column-name keys
    parser.add_argument('--GO_Term_key', type=str, default='Term', help='Index column in GO enrichment file. Default: Term')
    parser.add_argument('--GO_Genes_key', type=str, default='Genes', help='Gene-list column in GO enrichment file. Default: Genes')
    parser.add_argument('--Geneset_Term_key', type=str, default='Term', help='Index column in geneset enrichment file. Default: Term')
    parser.add_argument('--Geneset_Genes_key', type=str, default='Genes', help='Gene-list column in geneset enrichment file. Default: Genes')
    parser.add_argument('--Trait_Term_key', type=str, default='Term', help='Index column in trait enrichment file. Default: Term')
    parser.add_argument('--Trait_Genes_key', type=str, default='Genes', help='Gene-list column in trait enrichment file. Default: Genes')
    parser.add_argument('--Perturbation_Sample_key', type=str, default='Sample',
        help='Sample-label column in perturbation results. Default: Sample')

    args = parser.parse_args()

    # Resolve derived defaults
    thresh_str = str(args.sel_thresh).replace('.', '_')
    eval_base = f'{args.out_dir}/{args.run_name}/Evaluation/{args.K}_{thresh_str}'
    if args.save_path is None:
        args.save_path = f'{args.out_dir}/{args.run_name}/Interpretation/Summary_table/{args.K}_{thresh_str}'
    if args.mdata_path is None:
        args.mdata_path = f'{args.out_dir}/{args.run_name}/Inference/adata/cNMF_{args.K}_{thresh_str}.h5mu'

    # create output directory
    os.makedirs(f'{args.save_path}', exist_ok=True)

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
    with open(f'{args.save_path}/config_{job_id}.yml', 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)

    # ---- Pre-flight check: verify the input h5mu exists ----
    if not os.path.isfile(args.mdata_path):
        raise FileNotFoundError(f"Input .h5mu not found: {args.mdata_path}")

    # ── Load MuData ──
    print(f'Loading MuData: {args.mdata_path}')
    mdata = mu.read(args.mdata_path)

    # ── Evaluation file paths ──
    GO_path = f'{eval_base}/{args.K}_GO_term_enrichment.txt'
    Geneset_path = f'{eval_base}/{args.K}_geneset_enrichment.txt'
    Trait_path = f'{eval_base}/{args.K}_trait_enrichment.txt'
    Perturbation_path_base = f'{eval_base}/{args.K}_{args.perturbation_file_name}'
    Association_path = f'{eval_base}/{args.K}_categorical_association_results.txt'
    Explained_Variance_path = f'{eval_base}/{args.K}_Explained_Variance.txt'

    # ── Program loadings ──
    df_Program_loading_long = compile_Program_loading_score_sheet_long(
        mdata, num_gene=args.num_gene, data_key=args.data_key, gene_names_key=args.gene_names_key)
    df_Program_loading_flat = compile_Program_loading_score_sheet_flat(
        mdata, num_gene=args.num_gene, data_key=args.data_key, gene_names_key=args.gene_names_key)

    # ── Enrichment sheets ──
    df_GO = Compile_GO_sheet(GO_path, gene_num=args.num_gene, term_key=args.GO_Term_key, genes_key=args.GO_Genes_key) if os.path.exists(GO_path) else None
    df_Geneset = Compile_Geneset_sheet(Geneset_path, gene_num=args.num_gene, term_key=args.Geneset_Term_key, genes_key=args.Geneset_Genes_key) if os.path.exists(Geneset_path) else None
    df_Trait = Compile_Trait_sheet(Trait_path, gene_num=args.num_gene, term_key=args.Trait_Term_key, genes_key=args.Trait_Genes_key) if os.path.exists(Trait_path) else None

    # ── Perturbation ──
    perturbation_files = [f'{Perturbation_path_base}_{samp}.txt' for samp in args.Sample]
    if any(os.path.exists(f) for f in perturbation_files):
        df_Perturbation = Compile_Perturbation_sheet(Perturbation_path_base, Sample=args.Sample, sample_key=args.Perturbation_Sample_key)
    else:
        print(f'No perturbation files found for: {Perturbation_path_base}')
        df_Perturbation = None

    # ── Categorical association & explained variance ──
    df_Association = Compile_Association_sheet(Association_path, gene_num=args.num_gene) if os.path.exists(Association_path) else None
    df_Explained_Variance = Compile_Explained_variance(Explained_Variance_path) if os.path.exists(Explained_Variance_path) else None

    # ── Validate program names align across loaded DataFrames ──
    check_program_name_match(mdata, prog_key=args.prog_key, dataframes=[
        df_GO, df_Geneset, df_Trait, df_Perturbation,
        df_Association, df_Explained_Variance
    ])

    # ── Target Summary (writes specificity / correlation / KD-efficiency sidecars to save_path) ──
    df_Target_Summary = Compile_Target_Summary_sheet(
        mdata, Perturbation_path_base,
        Sample=args.Sample, categorical_key=args.categorical_key,
        prog_key=args.prog_key, data_key=args.data_key,
        guide_targets_key=args.guide_targets_key,
        save_path=args.save_path, effect_size=args.effect_size,
        control_target_name=args.control_target_name,
        gene_names_key=args.gene_names_key,
    )

    # ── Program Summary ──
    df_Summary = Compile_Summary_sheet(
        mdata, df_GO, df_Geneset, df_Perturbation, df_Program_loading_flat, df_Explained_Variance,
        Sample=args.Sample, specicicity_path=args.save_path,
        categorical_key=args.categorical_key, non_tagerting_key=args.non_targeting_key,
        effect_size=args.effect_size, adjusted_pval_key=args.adjusted_pval_key,
    )

    # ── Save key DataFrames as separate TSV files ──
    df_Summary.to_csv(f'{args.save_path}/Summary_{args.K}_{thresh_str}.tsv', sep='\t')
    df_Program_loading_long.to_csv(f'{args.save_path}/Program_Loadings_{args.K}_{thresh_str}.tsv', sep='\t')
    df_Target_Summary.to_csv(f'{args.save_path}/Targets_Summary_{args.K}_{thresh_str}.tsv', sep='\t')
    print(f'Saved separate TSV files to {args.save_path}')

    # ── Write Excel ──
    excel_path = f'{args.save_path}/cNMF_{args.K}_{thresh_str}.xlsx'
    print(f'Compiling Excel for K={args.K}, threshold={args.sel_thresh}')
    with pd.ExcelWriter(excel_path) as writer:
        df_Summary.to_excel(writer, sheet_name='Summary', index=True)
        df_Program_loading_long.to_excel(writer, sheet_name='Program Loadings', index=True)
        df_Target_Summary.to_excel(writer, sheet_name='Targets Summary', index=True)

        if df_Association is not None:
            df_Association.to_excel(writer, sheet_name='Sample Association', index=True)

        if df_Perturbation is not None:
            # Re-load perturbation per sample with specificity scores merged in.
            # add_specificity_scores_file returns (full_merged, significant_merged);
            # both carry the specificity_scores column.
            combined_full, combined_sig = [], []
            for samp in args.Sample:
                df_full_, df_sig_ = add_specificity_scores_file(args.save_path, Perturbation_path_base, samp)
                df_full_[args.Perturbation_Sample_key] = samp
                df_sig_[args.Perturbation_Sample_key] = samp
                combined_full.append(df_full_)
                combined_sig.append(df_sig_)
            df_Perturbation_with_spec = pd.concat(combined_full)
            df_Perturbation_sig_with_spec = pd.concat(combined_sig)

            for i in range(0, len(df_Perturbation_with_spec), MAX_ROWS):
                sheet_num = i // MAX_ROWS + 1
                df_Perturbation_with_spec.iloc[i:i + MAX_ROWS].to_excel(
                    writer, sheet_name=f'Perturbation Association {sheet_num}', index=True)

            for i in range(0, len(df_Perturbation_sig_with_spec), MAX_ROWS):
                sheet_num = i // MAX_ROWS + 1
                df_Perturbation_sig_with_spec.iloc[i:i + MAX_ROWS].to_excel(
                    writer, sheet_name=f'significant regulators only {sheet_num}', index=True)

        if df_Trait is not None:
            df_Trait.to_excel(writer, sheet_name='Trait Enrichment', index=True)
        if df_GO is not None:
            df_GO.to_excel(writer, sheet_name='GO Term Enrichment', index=True)
        if df_Geneset is not None:
            df_Geneset.to_excel(writer, sheet_name='Geneset Enrichment', index=True)

    print(f'Done. Saved to {excel_path}')
    print("Pipeline finished.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
