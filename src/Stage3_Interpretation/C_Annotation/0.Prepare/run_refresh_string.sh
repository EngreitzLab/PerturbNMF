#!/bin/bash
#SBATCH --job-name=refresh_string
#SBATCH -p <partition>
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/refresh_string.%j.log

# Usage: export PIPELINE_ROOT=/path/to/PerturbNMF; sbatch run_refresh_string.sh
# Edit RESULT and the args below; every arg is listed, defaults included.
# Re-runs STRING in existing program folders (string_sig.tsv + string_genes.json, now with the STRING links among
# the regulators), using the settings recorded in <RUN_DIR>/Programs/meta.json. --rebuild_context auto rewrites
# regulator_context.md (section C5) and the evidence only where the interpretation step is not done.
# STRING needs network access (about 4 calls per program, 1 s apart).

set -euo pipefail
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

SCRIPT_DIR=$PIPELINE_ROOT/src/Stage3_Interpretation/C_Annotation/0.Prepare
RESULT=/path/to/out_dir/run_name
RUN_DIR=$RESULT/Interpretation/Annotation

python $SCRIPT_DIR/refresh_string.py \
  --run_dir $RUN_DIR \
  --rebuild_context auto
