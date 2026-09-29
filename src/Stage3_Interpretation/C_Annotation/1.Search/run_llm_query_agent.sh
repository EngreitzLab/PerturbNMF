#!/bin/bash
#SBATCH --job-name=lit_query_agent
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/lit_query_agent.%j.out
#SBATCH --error=logs/lit_query_agent.%j.err

# Usage: sbatch run_llm_query_agent.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Turns each program's research_brief (or QUESTION) into literature queries + curation questions,
# stored in <OUT_DIR>/P<k>.json -> literature_plan[<plan_name>]. Run search_generif.py first so the
# agent sees which genes lack cell-type evidence (optional). Programs that already have the plan are
# skipped; add --overwrite to re-plan. Needs ANTHROPIC_API_KEY in AGeneTic/.env.
set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate geneqa

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/1.Search
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info
OUT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
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
