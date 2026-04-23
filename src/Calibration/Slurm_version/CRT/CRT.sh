#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=total_counts_guides_per_cell_pct_counts_mt   # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Evaluation/50_0_2/CRT_other_covariates/total_counts_guides_per_cell_pct_counts_mt/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Evaluation/50_0_2/CRT_other_covariates/total_counts_guides_per_cell_pct_counts_mt/logs/%j.err       # Error file
#SBATCH --partition=owners,engreitz,bigmem            # partition name
#SBATCH --time=02:00:00                 # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=40              # CPUs per task
#SBATCH --mem=256G                       # Memory per node


# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address


# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result"
RUN_NAME="030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50"
LOG_DIR="$OUT_DIR/$RUN_NAME"

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
mkdir -p "$LOG_DIR/Evaluation/logs"

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate programDE
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:${PYTHONPATH:-}"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Calibration/Slurm_version/CRT/CRT.py \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --mdata_guide_path "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Data/raw_updated_withguide_030526.h5ad" \
        --guide_annotation_key "non-targeting" \
        --number_permutations 5000 \
        --number_guide 6 \
        --components 50 \
        --sel_thresh 0.2 \
        --categorical_key "batch" \
        --log_covariates total_counts guides_per_cell \
        --covariates pct_counts_mt \
        --FDR_method 'StoreyQ' \
        --save_dir '/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Evaluation/50_0_2/CRT_other_covariates/total_counts_guides_per_cell_pct_counts_mt' \



# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
