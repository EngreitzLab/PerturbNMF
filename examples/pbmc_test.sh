#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=pbmc_test
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output/pbmc_test/Inference/logs/%j.out
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output/pbmc_test/Inference/logs/%j.err
#SBATCH --partition=gpu,owners
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C GPU_SKU:RTX_2080Ti

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ymo@stanford.edu

# Configuration
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output"
RUN_NAME="pbmc_test"
LOG_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output/pbmc_test/Inference/logs"

# Store start time
START_TIME=$(date +%s)

# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Log directory: $LOG_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Start resource monitoring
MONITOR_LOG="$LOG_DIR/resource_monitor_${SLURM_JOB_ID}.log"
monitor_resources() {
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MONITOR_LOG"
        echo "=== GPU Usage ===" >> "$MONITOR_LOG"
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu --format=csv >> "$MONITOR_LOG" 2>/dev/null
        echo "=== Memory ===" >> "$MONITOR_LOG"
        free -h >> "$MONITOR_LOG"
        echo "---" >> "$MONITOR_LOG"
        sleep 30
    done
}
monitor_resources &
MONITOR_PID=$!

# Activate conda environment
echo "Activating conda environment: torch-cNMF"
eval "$(conda shell.bash hook)"
conda activate torch-cNMF

echo "Active env: $CONDA_DEFAULT_ENV"
echo "Python: $(python --version)"

# Run pipeline
echo "Running: inference-torch"
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage1_Inference/torch-cNMF/Slurm_Version/torch_cnmf_inference_pipeline.py \
        --counts_fn /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output/pbmc3k.h5ad \
        --output_directory /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output \
        --run_name pbmc_test \
        --species human \
        --K 5 7 10 \
        --numiter 10 \
        --numhvgenes 2000 \
        --sel_thresh 2.0 \
        --seed 14 \
        --algo halsvar \
        --mode batch \
        --tol 1e-4 \
        --run_factorize \
        --run_refit \
        --run_compile_annotation \
        --run_diagnostic_plots

# Capture exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Pipeline exited with code $EXIT_CODE"
fi

# Stop resource monitoring
kill $MONITOR_PID 2>/dev/null
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "GPU not available"

# Generate h5mu structure summary files
if [ $EXIT_CODE -eq 0 ]; then
    ADATA_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/examples/example_output/pbmc_test/Inference/adata"
    if [ -d "$ADATA_DIR" ]; then
        echo "Generating h5mu structure files..."
        python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/perturbNMF-runner/scripts/generate_h5mu_structure.py dummy --adata_dir "$ADATA_DIR"
        echo "h5mu structure generation complete"
    else
        echo "WARNING: adata directory not found at $ADATA_DIR, skipping structure generation"
    fi
else
    echo "Skipping h5mu structure generation due to pipeline error"
fi

# Elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED} seconds)"
exit $EXIT_CODE
