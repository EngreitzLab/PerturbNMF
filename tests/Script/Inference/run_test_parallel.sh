#!/bin/bash
#SBATCH --partition=engreitz,owners
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_sk_parallel
#SBATCH --output=tests/output/sk-cNMF-parallel/logs/slurm_%j.out
#SBATCH --error=tests/output/sk-cNMF-parallel/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

PIPELINE_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:$PYTHONPATH"

LOG_DIR="tests/output/sk-cNMF-parallel/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate sk-cNMF

echo "=== sk-cNMF parallel mode test ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo ""

python -m pytest tests/Script/Inference/test_inference_parallel.py -v --tb=short \
    2>&1 | tee "${LOG_DIR}/test_parallel.out"

echo ""
echo "=== parallel mode test done ==="
