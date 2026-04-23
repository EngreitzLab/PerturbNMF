#!/bin/bash
#SBATCH --partition=engreitz,owners
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_evaluation
#SBATCH --output=tests/output/torch-cNMF/dataloader/Evaluation/logs/slurm_%j.out
#SBATCH --error=tests/output/torch-cNMF/dataloader/Evaluation/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

PIPELINE_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:$PYTHONPATH"

LOG_DIR="tests/output/torch-cNMF/dataloader/Evaluation/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

echo "=== Evaluation pipeline unit tests ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Date: $(date)"
echo ""

python -m pytest tests/Script/Evaluation/test_evaluation.py -v --tb=short 2>&1 | tee "${LOG_DIR}/test_evaluation.out"

echo ""
echo "=== Evaluation tests done ==="
echo "Output saved to: tests/output/torch-cNMF/dataloader/Evaluation/"
