#!/bin/bash
#SBATCH --job-name=lit_search
#SBATCH -p <partition>
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
eval "$(conda shell.bash hook)"
conda activate geneqa

# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
SCRIPT_DIR="$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search"
RESULT="/path/to/output_dir/<run_name>"   # edit: run directory with Interpretation/
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
