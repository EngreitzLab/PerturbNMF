#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=total_counts_guides_per_cell_pct_counts_mt   # Job name
#SBATCH --output=/path/to/logs/%j.out   # edit: SLURM does not expand variables here
#SBATCH --error=/path/to/logs/%j.err   # edit: SLURM does not expand variables here
#SBATCH --partition=<partition>            # partition name
#SBATCH --time=02:00:00                 # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=40              # CPUs per task
#SBATCH --mem=256G                       # Memory per node


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

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate NMF_Benchmarking
export PYTHONPATH="${PIPELINE_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # parallelism comes from --n_jobs

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 "${PIPELINE_ROOT}/src/Stage2_Evaluation/B_Calibration/Slurm_version/CRT/CRT.py" \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --guide_annotation_key "non-targeting" \
        --number_permutations 4999 \
        --extra_resampled_rounds 49999 \
        --number_guide 6 \
        --K 50 \
        --sel_threshs 0.2 \
        --categorical_key "batch" \
        --log_covariates total_counts guides_per_cell \
        --covariates pct_counts_mt \
        --moi high \
        --normalization row_normalize \
        --pvalue_method skew_normal \
        --FDR_method 'StoreyQ' \
        --n_jobs "${SLURM_CPUS_PER_TASK:-1}" \
        --save_dir "$OUT_DIR/$RUN_NAME/Evaluation/K_thresh/"

        # Reference: CRT.py defaults if a flag above is removed:
        #   --K                       default [30, 50, 70, 80, 100, 200, 300]
        #   --sel_threshs             default [0.2, 2.0]
        #   --categorical_key         default "sample"   (one CRT per condition)
        #   --guide_annotation_key    default "non-targeting"
        #   --number_permutations     default 4999   (stage-1 resampling rounds per gene)
        #   --extra_resampled_rounds  default 49999  (stage 2, for programs with stage-1 p < 0.05; 0 disables)
        #   --number_guide            default 6      (NTC guides per null fake gene)
        #   --moi                     default high   (high: CRT; low: permutation vs control-guide cells)
        #   --normalization           default row_normalize  (or clr)
        #   --pvalue_method           default skew_normal    (or empirical)
        #   --FDR_method              default "BH"   (choices: "BH", "StoreyQ")
        #   --n_jobs                  default $SLURM_CPUS_PER_TASK or 1
        #   --save_dir                default <out_dir>/<run_name>/Evaluation/{K}_{thresh}/
        #   --covariates              default None  (e.g. --covariates pct_counts_mt doublet_scores)
        #   --log_covariates          default None  (e.g. --log_covariates total_counts guides_per_cell)
        # Other options (see README / python CRT.py --help): --control_group {all,ntc_cells,matched},
        #   --propensity_penalty, --null_method {frequency,random}, --n_bins, --n_ensemble,
        #   --skip_null, --qq_per_program, --check_covariate, --seed
        #   --skip_existing  off by default; append to resume a preempted job (reuses the
        #                    {K}_CRT_*.txt / {K}_CRT_fake_*.txt tables already written)



# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
