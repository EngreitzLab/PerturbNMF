#!/bin/bash
#SBATCH --job-name=synthesis
#SBATCH -p <partition>
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/synthesis.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_synthesis.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Reads the programs together: build_ledgers.py writes <RUN_DIR>/Synthesis/ (findings, evidence, regulator effects,
# regulator reports, program neighborhoods; no model calls), then synthesize_regulators.py asks Claude, per
# regulator cited in 2+ programs, whether its per-program mechanisms are consistent. The session backend (default)
# writes requests to <RUN_DIR>/Session_LLM/regulator_synthesis: answer them in Claude Code and run this again.

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/2.Synthesis
RESULT=/path/to/out_dir/run_name
RUN_DIR=$RESULT/Interpretation/Annotation

python $SCRIPT_DIR/build_ledgers.py \
  --run_dir $RUN_DIR \
  --top_gene 15 \
  --min_shared_regulators 2 \
  --min_gene_jaccard 0.1 \
  --broad_fraction 0.2
python $SCRIPT_DIR/synthesize_regulators.py \
  --run_dir $RUN_DIR \
  --min_programs 2 \
  --model claude-sonnet-5 \
  --effort medium \
  --max_tokens 6000 \
  --llm_backend session
