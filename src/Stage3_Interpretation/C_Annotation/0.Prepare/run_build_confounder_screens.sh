#!/bin/bash
#SBATCH --job-name=confounder_screens
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/confounder_screens.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_build_confounder_screens.sh
# Edit RESULT / GTF and the args below; every arg is listed, defaults included.
# Run after run_prepare_programs.sh (reads <RUN_DIR>/Programs/meta.json). Writes confounder_screens.json into every
# program folder: symbol families (ribosomal, mitochondrial, histone, ...), marker sets (cell cycle, heat shock, ISR,
# UPR, interferon) and the chromosome-position screen, and ticks the confounder_screens step of each plan.md.
# CONDITION_MARKERS: JSON {condition: {description, genes}} adds the condition_composition screen ("" = off).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/0.Prepare
RESULT=/path/to/out_dir/run_name
RUN_DIR=$RESULT/Interpretation/Annotation
GTF=/path/to/genes.gtf.gz
CONDITION_MARKERS=""   # e.g. "--condition_markers /path/to/condition_markers.json"

python $SCRIPT_DIR/build_confounder_screens.py \
  --run_dir $RUN_DIR \
  --gtf_path $GTF \
  --top_n_genes 50 \
  --window_bp 10000000 \
  --min_genes 3 \
  --max_p 1e-3 \
  --min_fold 2 \
  $CONDITION_MARKERS
