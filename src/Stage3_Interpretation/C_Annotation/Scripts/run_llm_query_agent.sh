#!/bin/bash
#SBATCH --job-name=lit_query_agent
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/lit_query_agent.%j.out
#SBATCH --error=logs/lit_query_agent.%j.err

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_llm_query_agent.sh
# Edit INFO_DIR and the args below; every arg is listed, defaults included.
# Turns each program's research_brief (or QUESTION) into literature queries + curation questions,
# stored in <OUT_DIR>/P<k>.json -> literature_plan[<plan_name>]. Run search_generif.py first so the
# agent sees which genes lack cell-type evidence (optional). Programs that already have the plan are
# skipped; add --overwrite to re-plan. Needs ANTHROPIC_API_KEY in AGeneTic/.env.
set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search
INFO_DIR=/path/to/Interpretation/Gene_info_extended_PerturbNMF_Info
OUT_DIR=$(dirname $INFO_DIR)/Literature_info_extended_PerturbNMF_Info
QUESTION=()                       # e.g. QUESTION=(--question "why does VIRMA knockdown reduce translation genes" --plan_name virma)
DISEASE=(--disease_context "coronary artery disease")   # set to () to skip D9 disease questions
OVERWRITE=""                      # set to --overwrite to re-plan

mkdir -p $OUT_DIR/logs
exec > >(tee -a $OUT_DIR/logs/lit_query_agent.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $OUT_DIR/logs/lit_query_agent.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/llm_query_agent.py \
  --info_dir $INFO_DIR \
  --out_dir $OUT_DIR \
  --programs 1 \
  --max_queries 12 \
  --model claude-sonnet-5 \
  --effort high \
  --max_tokens 16000 \
  "${QUESTION[@]}" "${DISEASE[@]}" $OVERWRITE
