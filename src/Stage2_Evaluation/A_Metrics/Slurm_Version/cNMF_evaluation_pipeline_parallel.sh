#!/bin/bash
set -euo pipefail

# SLURM job configuration
#SBATCH --job-name=111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test_par   # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Evaluation/logs/%A_%a.out   # Output file (%A=array job id, %a=array task id)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Evaluation/logs/%A_%a.err    # Error file
#SBATCH --partition=engreitz            # partition name
#SBATCH --array=1-6                     # Run parallel jobs (array indices 1-#K)
#SBATCH --time=05:00:00                 # Time limit per K
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=20              # CPUs per task
#SBATCH --mem=96G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address

START_TIME=$(date +%s)

# Define K values array (one K per array task)
K_VALUES=(30 50 60 80 100 200)

# Get K value for this array task
K=${K_VALUES[$((SLURM_ARRAY_TASK_ID-1))]}

# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results"
RUN_NAME="111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test"
LOG_DIR="$OUT_DIR/$RUN_NAME"

# Print job and system information for debugging
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "K value for this task: $K"
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
conda activate Evaluation_metric
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:${PYTHONPATH:-}"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Start resource monitoring
MONITOR_LOG="$LOG_DIR/Evaluation/logs/resource_monitor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

monitor_resources() {
    while true; do
        echo "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MONITOR_LOG"
        echo "=== Memory Usage ===" >> "$MONITOR_LOG"
        free -h >> "$MONITOR_LOG"
        echo "=== CPU Usage ===" >> "$MONITOR_LOG"
        top -bn1 | grep "Cpu(s)" >> "$MONITOR_LOG"
        echo "=== Process Memory ===" >> "$MONITOR_LOG"
        ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10 >> "$MONITOR_LOG"
        echo "---" >> "$MONITOR_LOG"
        sleep 30
    done
}

monitor_resources &
MONITOR_PID=$!

SCRIPT_START_TIME=$(date)
echo "Script start time: $SCRIPT_START_TIME"
echo "Initial memory usage:"
free -h

# Run the Python script for a single K
echo "Running Python script with (K=$K)..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --X_normalized_path "$LOG_DIR/cnmf_tmp/$RUN_NAME.norm_counts.h5ad" \
        --Perform_explained_variance \
        --Perform_categorical \
        --Perform_perturbation \
        --Perform_geneset \
        --Perform_trait \
        --data_key 'rna' \
        --prog_key 'cNMF' \
        --categorical_key 'batch' \
        --organism 'human' \
        --gene_names_key "symbol" \
        --guide_annotation_path "/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Data/guide/guide_metadata_v43.tsv" \
        --gwas_data_path '/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage2_Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz' \
        --sel_threshs 0.4 0.8 2.0 \
        --K $K \
        --FDR_method "StoreyQ" \
        --use_cache

        # Reference flags (uncomment + add to the python invocation above to enable):
        #--n_top 300
        #--skip_existing
        #--guide_names_key "guide_names"
        #--guide_targets_key "guide_targets"
        #--guide_assignment_key "guide_assignment"
        #--guide_annotation_key "non-targeting"
        #--Perform_motif
        #--reassign_name


# Cleanup function — runs on both success and failure
cleanup() {
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    SCRIPT_END_TIME=$(date)
    DURATION=$((END_TIME - START_TIME))

    # Stop resource monitoring
    kill $MONITOR_PID 2>/dev/null || true

    echo "========================================="
    echo "EXECUTION SUMMARY"
    echo "========================================="
    if [ $EXIT_CODE -ne 0 ]; then
        echo "*** PIPELINE FAILED with exit code $EXIT_CODE ***"
    fi
    echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
    echo "K value: $K"
    echo "Script start time: $SCRIPT_START_TIME"
    echo "Script end time: $SCRIPT_END_TIME"
    echo "Total execution time: ${DURATION} seconds ($(($DURATION / 3600))h $(($DURATION % 3600 / 60))m $(($DURATION % 60))s)"
    echo "Final memory usage:"
    free -h

    echo "SLURM reported peak memory usage: $(sacct -j ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} --format=MaxRSS --noheader | head -1 | tr -d ' ')" 2>/dev/null || echo "SLURM memory stats not available"

    echo "Resource monitoring log saved to: $MONITOR_LOG"
    echo "Job completed at: $(date)"
    exit $EXIT_CODE
}
trap cleanup EXIT
