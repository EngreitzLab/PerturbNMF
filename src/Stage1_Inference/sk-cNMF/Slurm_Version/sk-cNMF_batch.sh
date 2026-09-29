#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=consolidated_iterations          # Job name
#SBATCH --output=/path/to/logs/%j.out   # edit: SLURM does not expand variables here
#SBATCH --error=/path/to/logs/%j.err   # edit: SLURM does not expand variables here
#SBATCH --partition=<partition>           # partition name
#SBATCH --time=14:00:00                # Time limit (5 minutes)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=10              # CPUs per task
#SBATCH --mem=786G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=<your_email>    # Email address

# Define the cNMF case
# Path to your PerturbNMF checkout (export PIPELINE_ROOT=/path/to/PerturbNMF before sbatch)
: "${PIPELINE_ROOT:?set PIPELINE_ROOT to the PerturbNMF repo root}"

OUT_DIR="/path/to/output_dir"
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
export PYTHONPATH="${PIPELINE_ROOT}/src:${PYTHONPATH:-}"

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 "${PIPELINE_ROOT}/src/Stage1_Inference/sk-cNMF/Slurm_Version/sk-cNMF_batch_inference_pipeline.py"\
        --counts_fn  "/path/to/counts.h5ad"\
        --output_directory "$OUT_DIR" \
        --init "random" \
        --run_name "$RUN_NAME" \
        --algo "cd" \
        --max_NMF_iter 1000 \
        --tol 1e-4 \
        --K 50 \
        --sel_threshs 0.2 2.0 \
        --numhvgenes 17538 \
        --numiter 10 \
        --species "human" \
        --run_complie_annotation \
        --run_gene_annotation \
        --run_refit \
        --run_factorize

        # Reference flags (uncomment + add to the python invocation above to enable):
        #--nmf_seeds_path "/path/to/seeds.npy"
        #--loss "frobenius"
        #--seed 14
        #--num_gene 300
        #--run_diagnostic_plots
        #--skip_existing
        #--remove_noncoding
        #--ensembl_prefix "ENSG"
        #--gtf_path "/path/to/annotation.gtf.gz"
        #--gene_id_key "gene_id"
        #--add_gene_names_from_gtf
        #--data_key "rna"
        #--prog_key "cNMF"
        #--categorical_key "sample"
        #--guide_names_key "guide_names"
        #--guide_targets_key "guide_targets"
        #--guide_assignment_key "guide_assignment_key"
        #--gene_names_key "symbol"


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
