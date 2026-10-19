#!/bin/bash
#SBATCH --job-name=agenetic_report
#SBATCH -p <partition>
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=logs/agenetic_report.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_html_report.sh
# Edit RESULT / INFO_DIR and the args below; every arg is listed, defaults included.
# Reads <INFO_DIR>/P<k>.json (gene_info from search_gene, gene_interactions.OmniPath, and
# gene_interactions.Literature when present) and writes one HTML file.
# Leave out --programs to include every P<k>.json in INFO_DIR.
# Add --cdn to load Cytoscape.js from the web instead of inlining it (file ~0.4 MB smaller).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/3.Generate
RESULT=/path/to/out_dir/run_name
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info

python $SCRIPT_DIR/html_report.py \
  --info_dir $INFO_DIR \
  --out $RESULT/Interpretation/AGeneTic_test/Report/AGeneTic_report.html \
  --dataset_name "" \
  --top_regulator 5 \
  --top_gene 15
