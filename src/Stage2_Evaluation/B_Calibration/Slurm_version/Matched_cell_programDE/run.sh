#!/bin/bash
#SBATCH --job-name=de_batch_198
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=engreitz
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

# =============================================================================
# Self-contained DE Testing Batch Script
# Dataset: combined_final_merged | k=60 | dt=0.5 | Condition: resting | Batch 198
# =============================================================================
#
# REQUIRED R PACKAGES:
#   install.packages(c("optparse", "Matrix", "MatchIt", "cobalt",
#                      "sandwich", "lmtest", "nanoparquet"))
#
# OPTIONAL (only if using specific methods):
#   install.packages("twopartm")   # hurdle_* methods
#   BiocManager::install("limma")  # de_level='genes'
#
# =============================================================================

# =============================================================================
# INPUT FILES (all in same directory as this script)
# =============================================================================
#
# cells_metadata.tsv
#   - TSV with header row
#   - First column "barcode" contains cell IDs (e.g., "pool1:gex_1_AAACATCG...")
#   - Remaining columns are cell-level covariates (total_counts, pct_counts_mt, etc.)
#   - One row per cell
#
# gene_perturbations_sparse.tsv
#   - Sparse COO format: columns are cell_idx, gene_idx, value
#   - cell_idx: 0-based index into cells_metadata rows
#   - gene_idx: 0-based index into perturbation_names.txt
#   - value: 1 if cell has perturbation, 0 otherwise
#   - Generated from scipy sparse matrix via:
#       sparse_coo = sparse_matrix.tocoo()
#       pd.DataFrame({'cell_idx': sparse_coo.row, 'gene_idx': sparse_coo.col, 'value': sparse_coo.data})
#
# perturbation_names.txt
#   - Plain text, one perturbation name per line, no header
#   - Order matches gene_idx in gene_perturbations_sparse.tsv
#   - Includes both gene perturbations and NT groups (nt_group_*)
#   - NT groups are synthetic null controls created as follows:
#       1. Find all guides where target gene is None/empty/'non-targeting'
#       2. For each NT group, randomly sample 6 NT guides (no replacement)
#       3. A cell is assigned to nt_group_X if it has ANY of those 6 guides
#       4. Number of NT groups = number of gene perturbations (for matched calibration)
#
# combined_final_merged.usages.k_60.dt_0_5.consensus.txt
#   - TSV with header row
#   - First column "barcode" contains cell IDs (must match cells_metadata)
#   - Remaining columns are program loadings: "1", "2", ..., "60"
#   - Values are non-negative (cNMF usage weights)
#
# combined_final_merged.gene_spectra_tpm.k_60.dt_0_5.txt  [OPTIONAL]
#   - TSV with header row (gene IDs as column names: ENSG...)
#   - NO row name column - rows are programs in order (1, 2, ..., 60)
#   - This is the W matrix from cNMF: programs × genes
#   - Used for gene propagation: converts program-level β coefficients to gene-level
#     effects via β_gene = W^T × β_program, with bootstrap SE and bias correction
#   - To enable: add --gene_spectra_tpm combined_final_merged.gene_spectra_tpm.k_60.dt_0_5.txt --n_permutations 100
#
# batch_198.txt
#   - Plain text, one perturbation name per line, no header
#   - Subset of gene perturbations to test in this batch
#   - Names must exist in perturbation_names.txt
#
# nt_batch_0.txt
#   - Plain text, one nt_group name per line, no header
#   - Subset of null controls to test in this batch
#   - Used with --test_type nulls for calibration
#
# =============================================================================

# Get directory where this script lives
SCRIPT_DIR="/oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method"
cd "$SCRIPT_DIR"

# =============================================================================
# R Environment Setup
# =============================================================================

# Activate conda environment
source activate gene_propagation

# Prevent R from reading .Renviron/.Rprofile from $HOME (avoids filesystem throttling)
export R_ENVIRON_USER=/dev/null
export R_PROFILE_USER=/dev/null

# Redirect R caching and temp files to fast local storage
export TMPDIR="${SCRATCH}/tmp_r_$$"
export XDG_CACHE_HOME="${SCRATCH}/tmp_r_$$/cache"
export XDG_DATA_HOME="${SCRATCH}/tmp_r_$$/data"
export XDG_CONFIG_HOME="${SCRATCH}/tmp_r_$$/config"
export R_USER_CACHE_DIR="${XDG_CACHE_HOME}/R"
export R_USER_DATA_DIR="${XDG_DATA_HOME}/R"
export R_USER_CONFIG_DIR="${XDG_CONFIG_HOME}/R"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME"

# Control R threading (prevent thread multiplication with parallel::mclapply)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# =============================================================================
# Run DE Testing
# =============================================================================

echo "Running genes batch 198 (self-contained)..."
echo "Working directory: $SCRIPT_DIR"

stdbuf -oL -eL Rscript run_matching_de_batch.R \
    --input /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method \
    --cell_metadata /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method/cells_metadata.tsv \
    --cnmf_usages /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Results/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test/111025_D0_IGVF_10iter_torch_halsvar_batch_e7_v100s_test.usages.k_30.dt_2_0.consensus.txt \
    --gene_perturbations_sparse /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method/gene_perturbations_sparse.tsv \
    --gene_batch_file /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method/perturbed_expressed_genes.txt \
    --perturbation_names /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method/all_perturbed_genes.txt\
    --condition Day0 \
    --output_dir /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method \
    --test_type genes \
    --method matching \
    --min_cells_per_gene 50 \
    --n_jobs 8 \
    --batch_id 1 \
    --covariates "batch n_counts total_gene_umis num_expressed_genes log1p_total_counts log1p_total_counts_mt" \
    --log_transform_covariates "total_gene_umis n_counts  num_expressed_genes" \
    --categorical_covariates "batch" \
    --methods "ols_pseudo" \
    --matching_ratio 4 \
    --matching_method nearest \
    --n_bootstrap 100 \
    --bootstrap_ncpus 8 \
    --script_dir /oak/stanford/groups/engreitz/Users/ymo/cc-perturb-seq/Script/Tony_method \
    --de_level programs \
    --save_matched_cells \
    2>&1 | tee "${SCRIPT_DIR}/run.log"

    #    --matching_calipers "clusters3_res0.05_clust2_pos:0.1" \

# =============================================================================
# Example: Run on null controls (for calibration)
# =============================================================================
# To test NT groups instead of gene perturbations, change:
#   --gene_batch_file nt_batch_0.txt
#   --test_type nulls
#   --batch_id 0
#
# stdbuf -oL -eL Rscript run_matching_de_batch.R \
#     --input cells_metadata.tsv \
#     --cnmf_usages combined_final_merged.usages.k_60.dt_0_5.consensus.txt \
#     --condition resting \
#     --output_dir . \
#     --gene_batch_file nt_batch_0.txt \
#     --test_type nulls \
#     --method matching \
#     --min_cells_per_gene 50 \
#     --n_jobs 8 \
#     --batch_id 0 \
#     --covariates "pct_counts_mt G2M_score S_score high_mito_vs_low_mito_score_neg high_mito_vs_low_mito_score_pos high_MOI_vs_low_MOI_score_pos high_MOI_vs_low_MOI_score_neg high_ribo_vs_low_ribo_score_pos high_ribo_vs_low_ribo_score_neg pct_counts_ribo n_genes_by_counts total_counts plate Rep guide_umi_counts n_guides_gmm n_guides_total clusters3_res0.05_clust2_pos clusters3_res0.05_clust2_neg clusters4_res0.1_clust1_neg oxldl_vs_resting_score_neg il1b_vs_resting_score_neg" \
#     --log_transform_covariates "total_counts n_genes_by_counts guide_umi_counts" \
#     --categorical_covariates "plate Rep" \
#     --methods "ols_pseudo" \
#     --matching_ratio 4 \
#     --matching_method nearest \
#     --matching_calipers "clusters3_res0.05_clust2_pos:0.1" \
#     --n_bootstrap 100 \
#     --bootstrap_ncpus 8 \
#     --script_dir . \
#     --de_level programs \
#     --save_matched_cells \
#     2>&1 | tee run_nulls.log
