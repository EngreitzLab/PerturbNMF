#!/bin/bash
#SBATCH --job-name=program_viewer
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/program_viewer.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_viewer.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# The default output of the annotation, run after run_program_agent.sh. Writes one
# self-contained HTML (no external files) with every labelled program of <RUN_DIR>/Programs: a searchable rail grouped
# by the pre-annotation family label, and per program label + function sentence, program activity per condition,
# top genes + unique genes, driving genes (when the optional driver_genes step ran), significant regulators (inline
# SVG volcano per condition + log2FC table with --perturbation_path_base), regulator-gene interaction graph
# (interactions.jsonl), TF motif enrichment (with --motif_enrichment_path), confounder rule-outs, layered
# interpretation and modules, top GO terms.
# Drop the --perturbation_path_base / --motif_* blocks for a viewer without them (standard library only).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/2.Synthesis
RESULT=/path/to/out_dir/run_name
K_DIR=60_0_2                     # Evaluation/<K>_<density threshold> folder of the run
K=60

python $SCRIPT_DIR/program_viewer.py \
  --run_dir $RESULT/Interpretation/Annotation \
  --out $RESULT/Interpretation/Annotation/Report/program_viewer.html \
  --dataset_name "<dataset name>" \
  --home_url "" \
  --top_gene 15 \
  --top_regulator 6 \
  --membership_top 300 \
  --GO_path $RESULT/Evaluation/$K_DIR/${K}_GO_term_enrichment.txt \
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

# Publish to a static site (e.g. a GitHub Pages repo clone): <SITE_DIR>/<dataset slug>/viewer.html, its site.json and
# the root index.html linking every dataset's viewer. Commit + push SITE_DIR yourself (the pages are public).
SITE_DIR=""   # e.g. /path/to/program-reports; empty = do not publish
if [ -n "$SITE_DIR" ]; then
  python $SCRIPT_DIR/publish_viewer.py \
    --viewer $RESULT/Interpretation/Annotation/Report/program_viewer.html \
    --site_dir $SITE_DIR \
    --dataset_name "<dataset name>" \
    --home_url ../index.html
fi
