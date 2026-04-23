#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=consolidated_iterations          # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/020426_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/consolidated_iterations/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/020426_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/consolidated_iterations/logs/%j.err       # Error file
#SBATCH --partition=engreitz,owners,bigmem           # partition name
#SBATCH --time=14:00:00                # Time limit (5 minutes)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=10              # CPUs per task
#SBATCH --mem=786G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address

# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/020426_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50"
RUN_NAME="consolidated_iterations"
LOG_DIR="$OUT_DIR/$RUN_NAME/Inference/logs"

# Store start time
START_TIME=$(date +%s)


# Print some job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "Number of CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Log directory: $LOG_DIR"


# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Activate conda base environment
echo "Activating conda base environment..."
eval "$(conda shell.bash hook)"
conda activate sk-cNMF
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:$PYTHONPATH"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py\
        --counts_fn  "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Data/raw.h5ad"\
        --output_directory "$OUT_DIR" \
        --init "random" \
        --run_name "$RUN_NAME" \
        --algo "cd" \
        --max_NMF_iter 1000 \
        --tol 1e-4 \
        --K 50 \
        --sel_thresh 0.2 2.0 \
        --numhvgenes 17538 \
        --numiter 10 \
        --species "human" \
        --run_complie_annotation \
        --run_refit \
        --run_factorize


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
