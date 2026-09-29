#!/bin/bash
#SBATCH --job-name=lit_search
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/lit_search.%j.out
#SBATCH --error=logs/lit_search.%j.err

# Usage: sbatch run_literature_search.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# --mode auto searches literature_plan queries (llm_query_agent.py) when a program has them, else the
# OmniPath not-found pairs. PDFs + search_log.json go to <LIT_DIR>/Literature_search/<task>/.
# Tasks with a search_log.json are skipped; add --overwrite to re-search.
# Uses the Claude Agent SDK (ANTHROPIC_API_KEY or a Claude Code login); spend is capped per task.
set -euo pipefail
source /oak/stanford/groups/engreitz/Users/ymo/miniforge3/etc/profile.d/conda.sh
conda activate geneqa

SCRIPT_DIR=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/C_Annotation/AGeneTic/1.Search
RESULT=/oak/stanford/groups/engreitz/Users/ymo/Project/Helen_telohaec/Result/073126_2kG_torch_e8_64
INFO_DIR=$RESULT/Interpretation/AGeneTic_test/Gene_info_extended_PerturbNMF_Info
LIT_DIR=$RESULT/Interpretation/AGeneTic_test/Literature_info_extended_PerturbNMF_Info
OVERWRITE=""   # set to --overwrite to re-search

mkdir -p $LIT_DIR/logs
exec > >(tee -a $LIT_DIR/logs/lit_search.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $LIT_DIR/logs/lit_search.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/run_literature_search.py \
  --info_dir $INFO_DIR \
  --lit_dir $LIT_DIR \
  --out_dir $LIT_DIR/Literature_search \
  --programs 1 \
  --mode auto \
  --sources ncbi crossref nature \
  --max_papers 5 \
  --model claude-sonnet-5 \
  --max_turns 20 \
  --max_tool_calls 8 \
  --max_budget_usd 1.0 \
  $OVERWRITE
