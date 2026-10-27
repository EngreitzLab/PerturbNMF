#!/bin/bash
#SBATCH --job-name=program_agent
#SBATCH -p <partition>
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/program_agent.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_program_agent.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# One Claude agent per program works Programs/P<k>/plan.md with the program-annotator skill (gene information,
# interactions, literature, regulator context, label, interpretation), calling the skills in
# C_Annotation/agent/. Programs whose required steps are done are skipped, so a rerun continues where it stopped.
# Each agent runs in a Bubblewrap sandbox (sandbox.py): it can write only its own program folder and cannot see
# C_Annotation/.env, other runs or other programs' outputs. Needs bwrap on PATH or BWRAP=/path/to/bwrap.
# --auth apikey bills ANTHROPIC_API_KEY (C_Annotation/.env; capped by --max_budget_usd per program);
# --auth login uses this machine's Claude Code login. Needs network access and the geneqa environment
# (claude-agent-sdk, requests, pymupdf).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate geneqa
export BWRAP=${BWRAP:-bwrap}   # e.g. the cluster's bubblewrap module binary

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/1.Annotate
RESULT=/path/to/out_dir/run_name
RUN_DIR=$RESULT/Interpretation/Annotation
PROGRAMS=""   # e.g. "--programs 1 2 3"; empty = every program with a step left
REDO=""       # "--redo_stale" reopens done steps whose outputs no longer pass their checks (e.g. after a check was tightened)

python $SCRIPT_DIR/run_program_agent.py \
  --run_dir $RUN_DIR \
  $PROGRAMS $REDO \
  --workers 2 \
  --model claude-sonnet-5 \
  --max_turns 250 \
  --max_budget_usd 3.0 \
  --auth apikey
python $PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/status.py --run_dir $RUN_DIR --recheck
