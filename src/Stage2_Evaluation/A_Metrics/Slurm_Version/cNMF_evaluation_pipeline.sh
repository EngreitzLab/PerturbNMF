#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=cNMF_evaluation_pipeline           # Job name
#SBATCH --output=/path/to/logs/%j.out   # edit: SLURM does not expand variables here
#SBATCH --error=/path/to/logs/%j.err   # edit: SLURM does not expand variables here
#SBATCH --partition=<partition>            # partition name
#SBATCH --time=05:00:00                 # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=20              # CPUs per task
#SBATCH --mem=96G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=<your_email>    # Email address

# Define the cNMF case
# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"

OUT_DIR="/path/to/output_dir"
RUN_NAME="example_run"
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

# Activate conda base environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate Evaluation_metric
export PYTHONPATH="${PIPELINE_ROOT}/src:${PYTHONPATH:-}"


echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 "${PIPELINE_ROOT}/src/Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py" \
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
        --guide_annotation_path "/path/to/guide_annotation.tsv" \
        --gwas_data_path "${PIPELINE_ROOT}/src/Stage2_Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz" \
        --sel_threshs 0.4 0.8 2.0 \
        --K 30 50 60 80 100 200 \
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





# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
