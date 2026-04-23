#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test           # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Evaluation/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Evaluation/logs/%j.err       # Error file
#SBATCH --partition=engreitz            # partition name
#SBATCH --time=05:00:00                 # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=20              # CPUs per task
#SBATCH --mem=96G                       # Memory per node

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address

# Define the cNMF case
OUT_DIR="/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results"
RUN_NAME="111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test"
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
conda activate NMF_Benchmarking
export PYTHONPATH="/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src:$PYTHONPATH"


echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Evaluation/Slurm_Version/cNMF_evaluation_pipeline.py \
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
        --data_guide_path "/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Data/IGVF_D0_example.h5mu" \
        --gene_names_key "symbol" \
        --guide_annotation_path "/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Data/guide/guide_metadata_v43.tsv" \
        --gwas_data_path '/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Evaluation/Resources/OpenTargets_L2G_Filtered.csv.gz' \
        --sel_thresh 0.4 0.8 2.0 \
        --K 30 50 60 80 100 200 \
        --FDR_method "StoreyQ"
        #--Perform_motif \




# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
