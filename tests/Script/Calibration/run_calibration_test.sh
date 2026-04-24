#!/bin/bash
#SBATCH --partition=engreitz,owners
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_calibration
#SBATCH --output=tests/output/torch-cNMF/dataloader/Calibration/logs/slurm_%j.out
#SBATCH --error=tests/output/torch-cNMF/dataloader/Calibration/logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

PIPELINE_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:${PYTHONPATH:-}"

LOG_DIR="tests/output/torch-cNMF/dataloader/Calibration/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking

echo "=== Calibration pipeline unit tests ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Date: $(date)"
echo ""

python -m pytest tests/Script/Calibration/test_calibration.py -v --tb=short 2>&1 | tee "${LOG_DIR}/test_calibration.out"

echo ""
echo "=== Calibration tests done ==="
