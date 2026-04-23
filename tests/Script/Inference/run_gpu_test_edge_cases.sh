#!/bin/bash
#SBATCH --partition=gpu,owners
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_torch_edge
#SBATCH --output=tests/output/torch-cNMF-edge-cases/logs/slurm_%j.out
#SBATCH --error=tests/output/torch-cNMF-edge-cases/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

PIPELINE_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:$PYTHONPATH"

LOG_DIR="tests/output/torch-cNMF-edge-cases/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate torch-nmf-dl

echo "=== torch-cNMF edge-case tests ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(which python)"
echo ""

python -m pytest tests/Script/Inference/test_edge_cases.py -v --tb=short \
    2>&1 | tee "${LOG_DIR}/test_edge_cases.out"

echo ""
echo "=== edge-case tests done ==="
