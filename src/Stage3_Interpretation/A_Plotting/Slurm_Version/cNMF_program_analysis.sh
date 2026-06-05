#!/bin/bash

# SLURM job configuration
#SBATCH --job-name=Program           # Job name
#SBATCH --output=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Plot/Program/logs/%j.out      # Output file (%j = job ID)
#SBATCH --error=/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Plot/Program/logs/%j.err       # Error file
#SBATCH --partition=engreitz,owners,bigmem            # partition name
#SBATCH --time=05:00:00                  # Time limit 
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=20               # CPUs per task
#SBATCH --mem=256G                       # Memory per node


# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL      # Send email at start, end, and on failure
#SBATCH --mail-user=ymo@stanford.edu    # Email address


# Define the cNMF case
LOG_DIR="/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Plot/Program"

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
mkdir -p "$LOG_DIR/logs"

# Activate conda base environment
echo "Activating conda base environment..."
source activate Interpretation

echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"


# Run the Python script
echo "Running Python script..."
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src/Stage3_Interpretation/A_Plotting/Slurm_Version/cNMF_program_analysis.py \
        --mdata_path "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/adata/cNMF_50_0_2.h5mu" \
        --perturb_path_base "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Evaluation/50_0_2/50_CRT" \
        --GO_path "/oak/stanford/groups/engreitz/Users/ymo/IGVF_ccperturbseq/Result/030526_100k_cells_100iter_allHVG_torch_halsvar_batch_e7_50/Evaluation/50_0_2/50_GO_term_enrichment.txt" \
        --top_program 5 \
        --p_value 0.05 \
        --pdf_save_path "$LOG_DIR" \
        --output_format PDF \
        --Conditions D0 D1 D2 D3 \
        --square_plots \
        --figsize 35 40 \
        --categorical_key "batch" \
        --subsample_frac 0.1

        # Reference flags (uncomment + add to the python invocation above to enable):
        #--data_key "rna"
        #--prog_key "cNMF"
        #--gene_name_key "gene_names"
        #--output_format "SVG"               # one of PDF | SVG | HTML
        #--html_share_path "$LOG_DIR/html_share"
        #--skip_existing                      # turn OFF skipping; re-process every program (default is to skip already-done)
        #--show                               # display plots interactively
        #--programs 4 5 6                     # plot specific program numbers only
        #--top_enrichned_term 10              # top GO terms per program (note typo: enrichned)
        #--up_thred_log 0.00                  # upper volcano log2FC threshold
        #--down_thred_log -0.00               # lower volcano log2FC threshold
        #--tagert_col_name "program_name"     # column in perturbation results (note typo: tagert)
        #--plot_col_name "target_name"
        #--log2fc_col "log2FC"
        #--corr_matrix_path "/path/to/corr_matrices"
        #--file_to_dictionary "/path/to/gene_name_map.tsv"
        #--reference_gtf_path "/path/to/reference.gtf.gz"





# Calculate and print elapsed time at the end
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "Job completed at: $(date)"
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_TIME} seconds)"
