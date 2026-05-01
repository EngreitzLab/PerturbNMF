#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=100525_100k_10iter_1000batiter_sk_cd_e7_seed42          # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/NMF_re-inplementing/Results/sk-cNMF_evaluation/100525_100k_10iter_1000batiter_sk_cd_e7_seed42/logs/%A_%a.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/NMF_re-inplementing/Results/sk-cNMF_evaluation/100525_100k_10iter_1000batiter_sk_cd_e7_seed42/logs/%A_%a.err       # Error file
#SBATCH --partition=engreitz           # partition name
#SBATCH --array=1                    # Run parallel jobs (array indices 1-#)
#SBATCH --time=100:00:00                # Time limit
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=1              # CPUs per task
#SBATCH --mem=32G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address

# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/NMF_re-inplementing/Results/sk-cNMF_evaluation/100525_100k_10iter_1000batiter_sk_cd_e7_seed42"
RUN_NAME="100525_100k_10iter_1000batiter_sk_cd_e7_seed42"
LOG_DIR="$OUT_DIR/$RUN_NAME/Inference/logs"

# Store start time
START_TIME=$(date +%s)


# Define K values array
K_VALUES=(300)


# Get K value for this array task
K=${K_VALUES[$((SLURM_ARRAY_TASK_ID-1))]}

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
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:${PYTHONPATH:-}"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py \
        --counts_fn "/oak/stanford/groups/engreitz/Users/ymo/NMF_re-inplementing/Cell_data/100k_250genes_withguide.h5ad" \
        --output_directory "$OUT_DIR" \
        --run_name "${RUN_NAME}_${K}" \
        --init "random" \
        --algo "cd" \
        --K 50 \
        --numiter 10 \
        --max_NMF_iter 1000 \
        --numhvgenes 17538 \
        --tol 1e-4 \
        --seed 14 \
        --species "human" \
        --sel_thresh 0.2 2.0 \
        --run_refit \
        --run_complie_annotation \
        --run_factorize \
        --nmf_seeds_path "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Data/10_seeds_14_${K}.npy"


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
