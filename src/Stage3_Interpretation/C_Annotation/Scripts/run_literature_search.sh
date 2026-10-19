#!/bin/bash
#SBATCH --job-name=lit_search
#SBATCH -p <partition>
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/lit_search.%j.out
#SBATCH --error=logs/lit_search.%j.err

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_literature_search.sh
# Edit PROGRAM_JSON and the args below; every arg is listed, defaults included.
# The agent searches papers, downloads open-access PDFs, reads them and summarizes each paper.
# --mode auto searches the literature_plan queries (llm_query_agent.py) when a program has them, else
# one search from its research_brief (--mode omnipath: OmniPath not-found pairs).
#   PDFs + search_log.json      -> <LIT_DIR>/Literature_search/<task>/
#   extended bundle             -> <LIT_DIR>/P<k>.json -> literature_search[<task>] (papers with pdf_path + summary)
# Tasks with a search_log.json are skipped (re-attached to the bundle); add --overwrite to re-search.
# Uses the Claude Agent SDK (ANTHROPIC_API_KEY or a Claude Code login); spend is capped per task.
set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Search
INFO_DIR=/path/to/Interpretation/Gene_info_extended_PerturbNMF_Info
LIT_DIR=$(dirname $INFO_DIR)/Literature_info_extended_PerturbNMF_Info
PROGRAM_JSON=($INFO_DIR/P1.json)   # one or more bundles, e.g. ($INFO_DIR/P1.json $INFO_DIR/P2.json)
OVERWRITE=""   # set to --overwrite to re-search

mkdir -p $LIT_DIR/logs
exec > >(tee -a $LIT_DIR/logs/lit_search.${SLURM_JOB_ID:-local}.out) \
     2> >(tee -a $LIT_DIR/logs/lit_search.${SLURM_JOB_ID:-local}.err >&2)

python $SCRIPT_DIR/run_literature_search.py \
  --program_json "${PROGRAM_JSON[@]}" \
  --lit_dir $LIT_DIR \
  --out_dir $LIT_DIR/Literature_search \
  --mode auto \
  --sources ncbi crossref nature \
  --max_papers 5 \
  --max_pdf_pages 6 \
  --max_pdf_chars 20000 \
  --model claude-sonnet-5 \
  --max_turns 40 \
  --max_tool_calls 8 \
  --max_budget_usd 2.0 \
  $OVERWRITE
