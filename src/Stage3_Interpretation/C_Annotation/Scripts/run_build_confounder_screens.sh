#!/bin/bash
#SBATCH --job-name=confounder_screens
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/confounder_screens.%j.out
#SBATCH --error=logs/confounder_screens.%j.err

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_build_confounder_screens.sh
# Edit RESULT / GTF and the args below; every arg is listed, defaults included.
# Run right after run_extract_program_information.sh (it reads <INFO_DIR>/meta.json for the h5mu, perturbation
# files and membership_top / fdr). Adds confounder_screens to every <INFO_DIR>/P<k>.json: symbol families
# (ribosomal, mitochondrial, histone, ...), marker sets (cell cycle, heat shock, ISR, UPR, interferon) and the
# chromosome-position screen, with label.families / primary_family / positional_flag.
# ALSO_UPDATE patches the same block into P<k>.json already in later folders (only that key changes).
# CONDITION_MARKERS: JSON {condition: {description, genes}} adds the condition_composition screen ("" = off).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/0.Prepare
RESULT=/path/to/out_dir/run_name
OUT=$RESULT/Interpretation/AGeneTic_test
GTF=/path/to/genes.gtf.gz
CONDITION_MARKERS=""   # e.g. "--condition_markers /path/to/condition_markers.json"

mkdir -p $OUT/logs
python $SCRIPT_DIR/build_confounder_screens.py \
  --info_dir $OUT/PerturbNMF_Info \
  --gtf_path $GTF \
  --also_update $OUT/Gene_info_extended_PerturbNMF_Info $OUT/Literature_info_extended_PerturbNMF_Info \
  --top_n_genes 50 \
  --window_bp 10000000 \
  --min_genes 3 \
  --max_p 1e-3 \
  --min_fold 2 \
  $CONDITION_MARKERS
