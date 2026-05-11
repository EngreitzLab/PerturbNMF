#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=U-test_calibration_test   # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/tests/output/torch-cNMF/batch/Evaluation/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/tests/output/torch-cNMF/batch/Evaluation/logs/%j.err       # Error file
#SBATCH --partition=owners,engreitz            # partition name
#SBATCH --time=01:00:00                 # Time limit
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=20              # CPUs per task
#SBATCH --mem=256G                       # Memory per node


# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address


# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/tests/output/torch-cNMF"
RUN_NAME="batch"
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
conda activate NMF_Benchmarking
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:${PYTHONPATH:-}"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/B_Calibration/Slurm_version/U-test_perturbation_calibration/U-test_perturbation_calibration.py \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --guide_annotation_key "non-targeting" \
        --data_key 'rna' \
        --prog_key 'cNMF' \
        --categorical_key 'day' \
        --guide_names_key 'guide_names' \
        --guide_targets_key 'guide_targets' \
        --guide_assignment_key 'guide_assignment' \
        --organism 'human' \
        --FDR_method "StoreyQ" \
        --number_run 10 \
        --number_guide 6 \
        --components 5 10 15 \
        --sel_thresh 2.0 \
        --compute_fake_perturbation_tests
        #--compute_real_perturbation_tests
        #--visualizations


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
