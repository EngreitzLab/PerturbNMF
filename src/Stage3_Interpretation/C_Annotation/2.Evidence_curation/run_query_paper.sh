#!/bin/bash
#SBATCH --job-name=query_paper
#SBATCH -p <partition>
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/query_paper.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_query_paper.sh
# Edit PROGRAM_JSON / PDF_DIR / the questions below; every arg is listed, defaults included.
# paper-qa answers each --question over the PDFs directly in PDF_DIR and writes the answers into
# PROGRAM_JSON (literature_evidence.questions); answers already there are kept.
# Answers are cached in <PDF_DIR>/paperqa_cache/; add --overwrite to re-run paper-qa.
# OPENAI_API_KEY goes in C_Annotation/.env.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/2.Evidence_curation
RESULT=/path/to/out_dir/run_name
LIT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
PROGRAM_JSON=$LIT_DIR/P1.json
PDF_DIR=$LIT_DIR/Literature_search/brief__Q1

python $SCRIPT_DIR/query_paper.py \
  --program_json $PROGRAM_JSON \
  --pdf_dir $PDF_DIR \
  --out_json $PROGRAM_JSON \
  --cache_dir $PDF_DIR/paperqa_cache \
  --config $SCRIPT_DIR/config.yaml \
  --question \
    "Which regulator drives this program in endothelial cells?" \
    "Does KIAA1429 (VIRMA) regulate ZNF10?"
