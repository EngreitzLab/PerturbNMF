#!/bin/bash
#SBATCH --job-name=agenetic_report
#SBATCH -p normal
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=logs/agenetic_report.%j.log

# Usage: sbatch run_html_report.sh
# Edit RESULT / INFO_DIR and the args below; every arg is listed, defaults included.
# Reads <INFO_DIR>/P<k>.json (gene_info from search_gene, gene_interactions.OmniPath, and
# gene_interactions.Literature when 2.Evidence_curation has run) and writes one HTML file.
# Leave out --programs to include every P<k>.json in INFO_DIR.
# Add --cdn to load Cytoscape.js from the web instead of inlining it (file ~0.4 MB smaller).

set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate NMF_Benchmarking

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/3.Generate
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info

python $SCRIPT_DIR/html_report.py \
  --info_dir $INFO_DIR \
  --out $RESULT/Interpretation/AGeneTic_test/Report/AGeneTic_report.html \
  --dataset_name "Helen teloHAEC 2kG" \
  --top_regulator 5 \
  --top_gene 15
