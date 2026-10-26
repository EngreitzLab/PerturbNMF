#!/bin/bash
#SBATCH --job-name=program_site
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/program_site.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_site.sh
# Edit RESULT, SITE_DIR and the args below; every arg is listed, defaults included.
# Step 3 of the program website (after run_program_label.sh and 2.Evidence_curation/run_query_driver_gene.sh).
# Writes one page per labelled program to <SITE_DIR>/<dataset slug>/P<k>.html plus the index pages:
#   label + function sentence, PerturbNMF input (top genes, unique genes, regulators, GO terms),
#   pre-annotation screen (when 0.Prepare/build_confounder_screens.py ran: family label + enriched chromosome),
#   driving genes (when query_driver_gene.py ran for the current label; other genes searched collapsed),
#   program interactive plots: regulator volcano per condition + heatmap (--perturbation_path_base) and
#   program activity violin by --groupby (--mdata_path; only the program modality is read).
# Drop the --perturbation_path_base / --mdata_path blocks for a page without plots (standard library only).
# SITE_DIR is a clone of your GitHub Pages repo; commit + push it to publish (the pages are public).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/3.Generate
RESULT=/path/to/out_dir/run_name
K_DIR=60_0_2                     # Evaluation/<K>_<density threshold> folder of the run
K=60
SITE_DIR=/path/to/program-reports

python $SCRIPT_DIR/program_site.py \
  --info_dir $RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info \
  --label_dir $RESULT/Interpretation/AGeneTic_test/Program_label \
  --site_dir $SITE_DIR \
  --dataset_name "<dataset name>" \
  --programs 1 \
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
  --mdata_path $RESULT/Inference/adata/cNMF_${K_DIR}.h5mu \
  --groupby <obs column, e.g. day> \
  --prog_key cNMF \
  --max_points_per_group 2000
