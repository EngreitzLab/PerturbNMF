#!/bin/bash
#SBATCH --job-name=program_label
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=logs/program_label.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_label.sh   (or bash: one API call per program)
# Edit RESULT and the args below; every arg is listed, defaults included.
# Step 1 of the program website: reads <INFO_DIR>/P<k>.json and writes a Claude-generated name +
# one-sentence function to <OUT_DIR>/P<k>.json. Cached labels are reused; add --overwrite to redo them.
# Then: 2.Evidence_curation/run_query_driver_gene.sh (driving genes; may refine the label), then run_program_site.sh.
# ANTHROPIC_API_KEY comes from C_Annotation/.env.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/3.Generate
RESULT=/path/to/out_dir/run_name
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
OVERWRITE=""   # set to --overwrite to re-label programs that already have a label

python $SCRIPT_DIR/program_label.py \
  --info_dir $INFO_DIR \
  --out_dir $RESULT/Interpretation/AGeneTic_test/Program_label \
  --programs 1 \
  --top_regulator 6 \
  --model claude-sonnet-5 \
  --effort medium \
  $OVERWRITE
