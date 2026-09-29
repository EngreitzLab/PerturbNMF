#!/bin/bash
#SBATCH --partition=<partition>
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_crt
#SBATCH --output=tests/output/calibration_logs/slurm_crt_%j.out
#SBATCH --error=tests/output/calibration_logs/slurm_crt_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your_email>

# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"

set -euo pipefail

PIPELINE_DIR="$PIPELINE_ROOT"
cd "$PIPELINE_DIR"
export PYTHONPATH="$PIPELINE_DIR/src:${PYTHONPATH:-}"

LOG_DIR="tests/output/calibration_logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate programDE

echo "=== CRT calibration unit tests (programDE env) ==="
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Date: $(date)"
echo ""

python -m pytest tests/Script/Stage2_Evaluation/test_crt.py::TestCRTReformat -v --tb=short 2>&1 | tee "${LOG_DIR}/test_crt.out"

echo ""
echo "=== CRT tests done ==="
