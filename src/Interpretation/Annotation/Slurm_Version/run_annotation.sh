#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=annotation_pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=engreitz
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

set -euo pipefail

# ---- Project directory ----
ANNOTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ANNOTATION_DIR/src"

# Create logs directory
mkdir -p logs

# Store start time
START_TIME=$(date +%s)

# Print job information
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Annotation directory: $ANNOTATION_DIR"

# ---- Activate conda environment ----
eval "$(conda shell.bash hook)"
conda activate progexplorer

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# ---- Load .env ----
if [ -f "$ANNOTATION_DIR/../../.env" ]; then
    export $(grep -v '^#' "$ANNOTATION_DIR/../../.env" | xargs)
fi

# ---- Run pipeline ----
echo "Running Annotation pipeline..."
python "$SRC_DIR/run_pipeline.py" \
    --config "$ANNOTATION_DIR/configs/pipeline_config.yaml" \
    "$@"

# Calculate and print elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
