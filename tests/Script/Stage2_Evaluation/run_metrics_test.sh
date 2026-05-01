#!/bin/bash
#SBATCH --partition=engreitz,owners
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_evaluation
#SBATCH --output=tests/output/eval_logs/slurm_%j.out
#SBATCH --error=tests/output/eval_logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

PIPELINE_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:${PYTHONPATH:-}"

# Auto-detect inference output (same priority as conftest.py)
INFERENCE_PATH=""
for candidate in \
    "tests/output/torch-cNMF/batch/Inference" \
    "tests/output/torch-cNMF/dataloader/Inference" \
    "tests/output/torch-cNMF/minibatch/Inference" \
    "tests/output/sk-cNMF/Inference"; do
    if [ -d "$candidate" ]; then
        INFERENCE_PATH="$candidate"
        break
    fi
done

if [ -z "$INFERENCE_PATH" ]; then
    echo "ERROR: No inference output found. Run inference tests first."
    exit 1
fi

EVAL_DIR="$(dirname "$INFERENCE_PATH")/Evaluation"
LOG_DIR="$EVAL_DIR/logs"
mkdir -p "$LOG_DIR"
mkdir -p tests/output/eval_logs

eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

echo "=== Evaluation pipeline unit tests ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Inference path: $INFERENCE_PATH"
echo "Date: $(date)"
echo ""

python -m pytest tests/Script/Stage2_Evaluation/test_metrics.py -v --tb=short \
    --inference-path "$INFERENCE_PATH" 2>&1 | tee "${LOG_DIR}/test_evaluation.out"

echo ""
echo "=== Evaluation tests done ==="
echo "Output saved to: $EVAL_DIR"
