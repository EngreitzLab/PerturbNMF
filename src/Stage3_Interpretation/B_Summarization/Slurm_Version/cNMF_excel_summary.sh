#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=cNMF_excel_summary                                                                                          # Job name
#SBATCH --output=/path/to/logs/%j.out   # edit: SLURM does not expand variables here
#SBATCH --error=/path/to/logs/%j.err   # edit: SLURM does not expand variables here
#SBATCH --partition=<partition>   # partition name(s)
#SBATCH --time=02:00:00                  # Time limit
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPUs per task
#SBATCH --mem=64G                        # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL       # Send email at start, end, and on failure
#SBATCH --mail-user=<your_email>     # Email address

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
mkdir -p "$LOG_DIR/Interpretation/Summary_table/logs"

# Activate conda environment
echo "Activating conda environment..."
source activate NMF_Benchmarking

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 "${PIPELINE_ROOT}/src/Stage3_Interpretation/B_Summarization/Slurm_Version/cNMF_excel_summary.py" \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --K 50 \
        --sel_thresh 0.2 \
        --num_gene 300 \
        --Sample D0 D1 D2 D3 \
        --categorical_key "batch" \
        --perturbation_file_name "CRT" \
        --effect_size "approx_log2FC" \
        --control_target_name "non-targeting" \
        --non_targeting_key "non-targeting" \
        --prog_key "cNMF" \
        --data_key "rna" \
        --guide_targets_key "guide_targets" \
        --gene_names_key "symbol" \
        --adjusted_pval_key "Adjusted P-value"
        # Optional: override the auto-derived output directory:
        # --save_path "$LOG_DIR/Interpretation/Summary_table/50_0_2" \
        # Optional: override the auto-derived input h5mu path:
        # --mdata_path "$LOG_DIR/Inference/adata/cNMF_50_0_2.h5mu" \
        # Optional: non-default enrichment / perturbation column headers:
        # --GO_Term_key "Term" --GO_Genes_key "Genes" \
        # --Geneset_Term_key "Term" --Geneset_Genes_key "Genes" \
        # --Trait_Term_key "Term" --Trait_Genes_key "Genes" \
        # --Perturbation_Sample_key "Sample"


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
