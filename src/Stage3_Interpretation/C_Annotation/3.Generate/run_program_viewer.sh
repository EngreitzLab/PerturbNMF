#!/bin/bash
#SBATCH --job-name=program_viewer
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/program_viewer.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_viewer.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Single-file alternative to run_program_site.sh (same content, run after run_program_label.sh and
# 2.Evidence_curation/run_query_driver_gene.sh). Writes one self-contained HTML (no external files) with
# every labelled program: a searchable rail grouped by the pre-annotation family label, and per program
#   label + function sentence, program activity per condition, top genes + unique genes,
#   driving genes (when query_driver_gene.py ran for the current label),
#   significant regulators (inline SVG volcano per condition + log2FC table with --perturbation_path_base;
#   top regulators given to the annotator), TF motif enrichment (with --motif_enrichment_path: Stage 2
#   run_motif_enrichment.py outputs; families, logos, candidate TFs), confounder rule-outs, layered
#   interpretation and modules (when run_program_interpretation.sh ran), top GO terms, pre-annotation screen
#   (when 0.Prepare/build_confounder_screens.py ran).
# Drop the --perturbation_path_base / --motif_* blocks for a viewer without them (standard library only).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/3.Generate
RESULT=/path/to/out_dir/run_name
K_DIR=60_0_2                     # Evaluation/<K>_<density threshold> folder of the run
K=60

python $SCRIPT_DIR/program_viewer.py \
  --info_dir $RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info \
  --label_dir $RESULT/Interpretation/AGeneTic_test/Program_label \
  --interpretation_dir $RESULT/Interpretation/AGeneTic_test/Program_interpretation \
  --out $RESULT/Interpretation/AGeneTic_test/Report/program_viewer.html \
  --dataset_name "<dataset name>" \
  --top_gene 15 \
  --top_regulator 6 \
  --membership_top 300 \
  --perturbation_path_base $RESULT/Evaluation/$K_DIR/${K}_perturbation_association_results \
  --conditions <condition> \
  --target_col target_name \
  --program_col program_name \
  --log2fc_key log2FC \
  --p_value 0.05 \
  --down_thred_log -0.05 \
  --up_thred_log 0.05 \
  --motif_enrichment_path $RESULT/Evaluation/$K_DIR/${K}_motif_enrichment.txt \
  --candidate_tfs_path $RESULT/Evaluation/$K_DIR/${K}_candidate_tfs.txt \
  --motif_logos_path $RESULT/Evaluation/$K_DIR/${K}_motif_logos.json \
  --motif_source_label "<motif database, promoter window, enhancer links>"
