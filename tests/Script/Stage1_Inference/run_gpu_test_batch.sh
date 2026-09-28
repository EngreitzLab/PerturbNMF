#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_torch_batch
#SBATCH --output=tests/output/torch-cNMF/batch/Inference/logs/slurm_%j.out
#SBATCH --error=tests/output/torch-cNMF/batch/Inference/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email>

# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"

set -euo pipefail

PIPELINE_DIR="$PIPELINE_ROOT"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:${PYTHONPATH:-}"

LOG_DIR="tests/output/torch-cNMF/batch/Inference/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate torch-nmf-dl

echo "=== torch-cNMF batch mode test ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(which python)"
echo ""

python -m pytest tests/Script/Stage1_Inference/test_inference_torch.py -v --tb=short \
    -k "TestDataValid or TestBatchMode" 2>&1 | tee "${LOG_DIR}/test_batch.out"

echo ""
echo "=== batch mode test done ==="
