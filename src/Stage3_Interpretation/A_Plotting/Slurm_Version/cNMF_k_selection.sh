#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test          # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Plot/k_selection/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/Plot/k_selection/logs/%j.err       # Error file
#SBATCH --partition=engreitz            # partition name
#SBATCH --time=1:00:00                  # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=1               # CPUs per task
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
mkdir -p "$LOG_DIR/Plot/k_selection/logs"

# Activate conda base environment (change deponds on the sk or torch)
echo "Activating conda base environment..."
source activate torch-cNMF 

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Interpretation/Plotting/Slurm_Version/cNMF_k_selection.py\
        --output_directory "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --save_folder_name "$LOG_DIR/Plot/k_selection" \
        --eval_folder_name "$LOG_DIR/Evaluation" \
        --groupby "batch" \
        --K 30 50 60 \
        --sel_threshs 2.0 0.4 \
        --samples 1 2 3 \
        --selected_k 50
        # Optional: for torch-cNMF runs where cnmf is not installed, provide a pre-computed stats file:
        # --stability_file "$OUT_DIR/$RUN_NAME/$RUN_NAME.k_selection_stats.df.npz" \
        # Optional: for non-default file names and column names (e.g. Morphic data), add:
        # --go_file "cNMF_GO_Biological_Process_2023_fisher_geneset_enrichment.txt" \
        # --geneset_file "cNMF_Reactome_2022_fisher_geneset_enrichment.txt" \
        # --trait_file "cNMF_OT_GWAS_fisher_trait_enrichment.txt" \
        # --term_col "term" \
        # --adjpval_col "adj_pval" \
        # --perturbation_file "cNMF_gene_time_point_perturbation_association_{sample}.txt" \
        # --perturb_adjpval_col "adj_pval" \
        # --perturb_target_col "target_name" \
        # --perturb_log2fc_col "log2FC"


# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
