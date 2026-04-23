#!/usr/bin/env Rscript

# =============================================================================
# R-only differential expression testing with propensity score matching
# =============================================================================
# This script implements DE testing using:
# - Propensity score matching (MatchIt) for covariate balance
# - OLS regression with HC3 robust standard errors
# - Balance assessment via cobalt
#
# Replaces the Python+rpy2 pipeline to eliminate overhead and simplify codebase.
# =============================================================================

suppressPackageStartupMessages({
    library(optparse)
    library(Matrix)
    library(MatchIt)
    library(cobalt)
    library(sandwich)  # For HC3 robust standard errors
    library(lmtest)    # For coeftest with robust SE
    library(parallel)
    library(nanoparquet)     # For Parquet file I/O in gene propagation results
})
# Conditionally loaded after parsing args:
#   library(twopartm)  - Only if methods contain 'hurdle'

# Parse command line arguments
option_list <- list(
    make_option("--input", type = "character", help = "P(directory will be used for other TSV files)"),
    make_option("--output_dir", type = "character", help = "Output directory for results"),
    make_option("--cell_metadata", type = "character", help = "path to cells_metadata.tsv "),
    make_option("--cnmf_usages", type = "character", help = "cNMF usages file (program loadings)"),
    make_option("--gene_perturbations_sparse", type = "character", help = "Guide assignment file"),
    make_option("--gene_batch_file", type = "character", help = "File with batch gene names to test (one per line)"),
    make_option("--perturbation_names", type = "character", help = "File with all gene names to test (one per line)"),

    
    make_option("--condition", type = "character", help = "Condition to test"),
    make_option("--test_type", type = "character", default = "genes", help = "What to test: genes or nulls"),
    make_option("--method", type = "character", default = "matching", help = "Statistical method (matching)"),
    make_option("--min_cells_per_gene", type = "integer", default = 50, help = "Minimum cells required per gene"),
    make_option("--n_jobs", type = "integer", default = 1, help = "Number of parallel jobs"),
    make_option("--batch_id", type = "character", help = "Batch identifier for output filename"),
    make_option("--matching_ratio", type = "integer", default = 5, help = "Controls per treated cell"),
    make_option("--matching_method", type = "character", default = "nearest", help = "Matching algorithm"),
    make_option("--matching_calipers", type = "character", default = "", help = "Calipers as covariate:width pairs"),
    make_option("--covariates", type = "character", default = "", help = "Covariate column names (space-separated)"),
    make_option("--log_transform_covariates", type = "character", default = "", help = "Covariates to log-transform"),
    make_option("--categorical_covariates", type = "character", default = "", help = "Categorical covariates to convert to factors"),
    make_option("--methods", type = "character", default = "ols_log", help = "Methods for DE testing and gene propagation (space-separated: ols_log, ols_pseudo, hurdle_marginal)"),
    make_option("--save_matched_cells", action = "store_true", default = FALSE, help = "Save matched cell indices"),
    make_option("--max_control_cells", type = "integer", default = 20000, help = "Max control cells for matching"),
    make_option("--n_bootstrap", type = "integer", default = 500, help = "Number of bootstrap iterations for gene propagation SE estimation"),
    make_option("--bootstrap_ncpus", type = "integer", default = 1, help = "Number of cores for bootstrap parallelization (uses fork-safe SNOW backend)"),
    make_option("--seed", type = "integer", default = 42, help = "Random seed"),
    # Gene propagation options
    make_option("--gene_spectra_tpm", type = "character", default = "", help = "Path to gene spectra TPM file (W matrix) for gene propagation"),
    make_option("--n_permutations", type = "integer", default = 100, help = "Number of permutations for empirical bias correction"),
    make_option("--script_dir", type = "character", help = "Directory containing de_testing_utils.R"),
    # Direct gene-level DE options
    make_option("--de_level", type = "character", default = "programs", help = "DE level: 'programs', 'genes', or 'both'"),
    make_option("--gene_expression", type = "character", default = "", help = "Path to gene_expression_sparse.tsv for direct gene DE"),
    make_option("--response_gene_names", type = "character", default = "", help = "Path to response_gene_names.txt")
)

opt_parser <- OptionParser(option_list = option_list)
args <- parse_args(opt_parser)

# String concatenation operator (defined early for use in logging)
`%R%` <- function(x, y) paste0(x, y)

# Set random seed
set.seed(args$seed)

cat("=" %R% strrep("=", 79), "\n")
cat("Running R-only DE testing with matching\n")
cat("Condition:", args$condition, "\n")
cat("Test type:", args$test_type, "\n")
cat("Method:", args$method, "\n")
cat("DE level:", args$de_level, "\n")
cat("Matching ratio:", args$matching_ratio, "\n")
cat("Matching algorithm:", args$matching_method, "\n")
cat("=" %R% strrep("=", 79), "\n\n")

# =============================================================================
# Gene Propagation Setup
# =============================================================================

# Determine if gene propagation is enabled (only for program-level testing)
# Gene propagation propagates program effects to genes, not relevant for direct gene DE
enable_gene_propagation <- (args$gene_spectra_tpm != "") && (args$de_level %in% c("programs", "both"))

# Source DE testing utilities (always needed for helper functions + gene propagation)
source(file.path(args$script_dir, "de_testing_utils.R"))

if (enable_gene_propagation) {

    cat("\n")
    cat("=" %R% strrep("=", 79), "\n")
    cat("Gene Propagation ENABLED\n")
    cat("=" %R% strrep("=", 79), "\n")
    cat("Gene spectra TPM:", args$gene_spectra_tpm, "\n")
    cat("Permutations for bias correction:", args$n_permutations, "\n")

    # Load gene spectra TPM matrix (W)
    W_matrix <- load_gene_spectra_tpm(args$gene_spectra_tpm)

    # Methods to propagate (same as regression methods)
    propagate_method_names <- strsplit(args$methods, " ")[[1]]
    cat("Methods to propagate:", paste(propagate_method_names, collapse = ", "), "\n")
    cat("=" %R% strrep("=", 79), "\n\n")
} else {
    W_matrix <- NULL
    propagate_method_names <- c()
    cat("\nGene propagation DISABLED (no --gene_spectra_tpm provided)\n\n")
}

# =============================================================================
# Direct Gene-Level DE Setup
# =============================================================================

enable_direct_gene_de <- (args$de_level %in% c("genes", "both")) && (args$gene_expression != "")

if (enable_direct_gene_de) {
    cat("\n")
    cat("=" %R% strrep("=", 79), "\n")
    cat("Direct Gene-Level DE ENABLED\n")
    cat("=" %R% strrep("=", 79), "\n")
    cat("Gene expression file:", args$gene_expression, "\n")
    cat("Response gene names:", args$response_gene_names, "\n")
    cat("=" %R% strrep("=", 79), "\n\n")
} else if (args$de_level %in% c("genes", "both")) {
    cat("\nWARNING: de_level is '", args$de_level, "' but --gene_expression not provided. Skipping direct gene DE.\n\n")
    enable_direct_gene_de <- FALSE
} else {
    cat("\nDirect gene-level DE DISABLED (de_level='programs')\n\n")
}

# =============================================================================
# Helper functions (moved to de_testing_utils.R for modularity)
# =============================================================================
# All helper, matching, regression, and gene propagation functions have been
# moved to scripts/de_testing_utils.R and are sourced above.
#
# This includes:
# - Helper functions: winsorize_nonzero, remove_collinear_covariates, build_covariate_matrix, etc.
# - Matching functions: perform_matching, compute_comprehensive_balance
# - Regression functions: fit_linear_model_robust, fit_ols_native_scale, fit_twopartm_hurdle
# - Gene propagation: load_gene_spectra_tpm, compute_gene_counterfactuals, etc.
# =============================================================================

# Test single gene across all programs
test_single_gene <- function(gene, gene_idx, obs_df, gene_perturbations, program_loadings,
                             covariate_matrix, program_names,
                             matching_ratio, matching_method, calipers_dict,
                             max_control_cells, seed, matched_cells_dir,
                             min_cells_per_gene, has_any_guide,
                             # Gene propagation parameters
                             W_matrix = NULL, enable_gene_propagation = FALSE,
                             n_permutations = 100, propagate_method_names = c(),
                             n_bootstrap = 500, bootstrap_ncpus = 1,
                             # Direct gene-level DE parameters
                             gene_expr_matrix = NULL, response_gene_names = NULL,
                             enable_direct_gene_de = FALSE, de_level = "programs") {

    perturbation_col <- paste0("perturb_", make.names(gene))

    # Extract perturbation indicator
    gene_perturb <- as.vector(gene_perturbations[, gene_idx])

    # Filter to cells with any guide (using pre-computed mask)
    keep_cells <- has_any_guide

    # Subset data
    obs_filtered <- obs_df[keep_cells, , drop = FALSE]
    covariate_matrix_filtered <- covariate_matrix[keep_cells, , drop = FALSE]
    gene_perturb_filtered <- gene_perturb[keep_cells]
    program_loadings_filtered <- program_loadings[keep_cells, , drop = FALSE]

    # Also filter gene expression matrix if direct gene DE is enabled
    gene_expr_filtered <- NULL
    if (enable_direct_gene_de && !is.null(gene_expr_matrix)) {
        gene_expr_filtered <- gene_expr_matrix[keep_cells, , drop = FALSE]
    }

    # Track pre-matching cell counts
    n_treated_prefilter <- sum(gene_perturb_filtered == 1)
    n_control_prefilter <- sum(gene_perturb_filtered == 0)

    cat(sprintf("  Matching: %d treated, %d controls\n", n_treated_prefilter, n_control_prefilter))

    # Add perturbation to design matrix
    X <- covariate_matrix_filtered
    X[[perturbation_col]] <- gene_perturb_filtered

    # Perform matching
    match_result <- perform_matching(
        X, perturbation_col, matching_ratio, matching_method,
        calipers_dict, max_control_cells, seed, verbose = FALSE
    )

    if (is.null(match_result)) {
        stop("Matching failed - insufficient data")
    }

    matched_data <- match_result$matched_data
    matched_indices <- match_result$matched_indices
    n_treated_matched <- match_result$n_treated
    n_control_matched <- match_result$n_control
    distance_map <- match_result$distance_map
    covariates_used <- match_result$covariates_used

    cat(sprintf("  Matched: %d treated, %d controls\n", n_treated_matched, n_control_matched))

    # Extract matching weights and pair IDs from matched data
    matching_weights <- matched_data$weights
    pair_ids <- matched_data$subclass

    # Extract original design matrix for matched cells (without MatchIt's extra columns)
    X_matched <- X[matched_indices, , drop = FALSE]

    # Add propensity score (distance) as a covariate
    X_matched$distance <- matched_data$distance

    # Validate sample sizes
    validation <- validate_sample_sizes(matched_data$treat, min_cells_per_gene, "MATCHING")
    if (!validation$is_valid) {
        return(list(results_df = data.frame(), matched_cells_data = list(), balance_metrics = NULL))
    }

    # Compute comprehensive balance
    treatment_values <- matched_data$treat
    balance_metrics <- compute_comprehensive_balance(matched_indices, treatment_values, obs_filtered)

    if (!is.null(balance_metrics)) {
        balance_metrics$perturbation <- gene
        balance_metrics$perturbation_type <- ifelse(
            startsWith(gene, "nt_group_"),
            "null_control",
            "gene_perturbation"
        )
        balance_metrics$used_in_matching <- balance_metrics$variable %in% covariates_used
    }

    # Collect matched cell barcodes
    matched_cells_data <- list()
    if (!is.null(matched_cells_dir)) {
        for (i in seq_along(matched_indices)) {
            barcode <- matched_indices[i]
            matched_cells_data[[length(matched_cells_data) + 1]] <- list(
                perturbation = gene,
                cell_barcode = barcode,
                treatment = as.integer(matched_data$treat[i]),
                distance = distance_map[barcode],
                weight = matching_weights[i],
                subclass = as.character(pair_ids[i])
            )
        }
    }

    # Get matched program loadings
    program_loadings_matched <- program_loadings_filtered[matched_indices, , drop = FALSE]

    # Get matched gene expression (for direct gene DE)
    gene_expr_matched <- NULL
    if (enable_direct_gene_de && !is.null(gene_expr_filtered)) {
        # DEBUG: Check indices
        cat(sprintf("    DEBUG gene_expr: nrow=%d, n_matched=%d, type=%s\n",
            nrow(gene_expr_filtered), length(matched_indices), typeof(matched_indices)))
        cat(sprintf("    DEBUG: head matched_indices: %s\n", paste(head(matched_indices, 3), collapse=",")))
        cat(sprintf("    DEBUG: head rownames(gene_expr): %s\n", paste(head(rownames(gene_expr_filtered), 3), collapse=",")))
        cat(sprintf("    DEBUG: matched in rownames: %d/%d\n",
            sum(matched_indices %in% rownames(gene_expr_filtered)), length(matched_indices)))
        gene_expr_matched <- gene_expr_filtered[matched_indices, , drop = FALSE]
    }

    # Test each program (only if de_level != "genes")
    results <- list()
    run_program_testing <- (de_level %in% c("programs", "both"))

    if (run_program_testing) {
        cat(sprintf("  Testing %d programs for gene %s...\n", length(program_names), gene))
    for (i in seq_along(program_names)) {
        program <- program_names[i]
        program_start_time <- Sys.time()
        cat(sprintf("    [%d/%d] Program %s... ", i, length(program_names), program))

        y_raw <- program_loadings_matched[, i]

        # Compute contingency table once (on raw data)
        y_scaled <- y_raw * 1e6
        contingency <- compute_contingency_table(y_scaled, X_matched, perturbation_col)

        # Initialize result with metadata (shared across methods)
        result <- list(
            perturbation = gene,
            program = program,
            n_treated_prefilter = n_treated_prefilter,
            n_control_prefilter = n_control_prefilter,
            n_treated_matched = n_treated_matched,
            n_control_matched = n_control_matched,
            n_matched = n_treated_matched + n_control_matched,
            match_rate_treated = n_treated_matched / n_treated_prefilter,
            match_rate_control = n_control_matched / n_control_prefilter,
            contingency_treated_zero = contingency$treated_zero,
            contingency_treated_pos = contingency$treated_pos,
            contingency_control_zero = contingency$control_zero,
            contingency_control_pos = contingency$control_pos
        )

        # Loop over regression methods
        for (method_name in regression_method_names) {
            program_seed <- seed + gene_idx * 1000 + i

            # Check if this method uses gene propagation (and thus bootstrap SE)
            uses_gene_propagation <- enable_gene_propagation && (method_name %in% propagate_method_names)

            if (method_name == "hurdle_marginal") {
                # Apply transformation for hurdle model
                transform_fn <- TRANSFORMATION_FUNCTIONS[[method_name]]
                y_transformed <- transform_fn(y_raw)

                # Fit two-part hurdle model with marginal effects
                # Skip SE calculation if using bootstrap (gene propagation)
                fit_result <- fit_twopartm_hurdle(
                    y_transformed, X_matched, perturbation_col,
                    max_control_cells, program_seed,
                    matching_weights, pair_ids,
                    compute_se = !uses_gene_propagation
                )

                # Store binary component results
                result[[paste0("coef_binary_", method_name)]] <- fit_result$coef_binary
                result[[paste0("se_binary_", method_name)]] <- fit_result$se_binary
                result[[paste0("z_stat_binary_", method_name)]] <- fit_result$z_stat_binary
                result[[paste0("p_value_binary_", method_name)]] <- fit_result$p_value_binary

                # Store magnitude component results
                result[[paste0("coef_mag_", method_name)]] <- fit_result$coef_mag
                result[[paste0("se_mag_", method_name)]] <- fit_result$se_mag
                result[[paste0("t_stat_mag_", method_name)]] <- fit_result$t_stat_mag
                result[[paste0("p_value_mag_", method_name)]] <- fit_result$p_value_mag

                # Store marginal effects results (main result)
                result[[paste0("coef_", method_name)]] <- fit_result$coef_marginal
                result[[paste0("se_", method_name)]] <- fit_result$se_marginal
                result[[paste0("t_stat_", method_name)]] <- fit_result$z_stat_marginal
                result[[paste0("p_value_", method_name)]] <- fit_result$p_value_marginal

                # Store n_nonzero for hurdle model
                result[[paste0("n_nonzero_", method_name)]] <- fit_result$n_nonzero

                # Store n_obs and n_treated from first method only
                if (!("n_obs" %in% names(result))) {
                    result$n_obs <- fit_result$n_obs
                    result$n_treated <- fit_result$n_treated
                }
            } else {
                # Apply method-specific transformation
                transform_fn <- TRANSFORMATION_FUNCTIONS[[method_name]]
                y_transformed <- transform_fn(y_raw)

                # Store pseudocount for ols_pseudo (needed for gene propagation)
                if (method_name == "ols_pseudo") {
                    pseudo <- compute_pseudocount_ols_pseudo(y_raw)
                    result[[paste0("pseudocount_", method_name)]] <- pseudo
                }

                # Fit OLS model with robust SE
                # Skip SE calculation if using bootstrap (gene propagation)
                fit_result <- fit_linear_model_robust(
                    y_transformed, X_matched, perturbation_col,
                    max_control_cells, program_seed,
                    matching_weights, pair_ids,
                    compute_se = !uses_gene_propagation
                )

                # Store results with method suffix
                result[[paste0("coef_", method_name)]] <- fit_result$coef
                result[[paste0("se_", method_name)]] <- fit_result$se
                result[[paste0("t_stat_", method_name)]] <- fit_result$t_stat
                result[[paste0("p_value_", method_name)]] <- fit_result$p_value

                # Store n_obs and n_treated from first method only (should be same for all)
                if (!("n_obs" %in% names(result))) {
                    result$n_obs <- fit_result$n_obs
                    result$n_treated <- fit_result$n_treated
                }
            }
        }

        program_elapsed <- as.numeric(difftime(Sys.time(), program_start_time, units = "secs"))
        cat(sprintf("done (%.1fs)\n", program_elapsed))

        results[[i]] <- result
        results[[i]]$program_time <- program_elapsed
    }

    # ==========================================================================
    # Gene Propagation (if enabled)
    # ==========================================================================
    gene_results <- NULL

    if (enable_gene_propagation && !is.null(W_matrix)) {
        cat(sprintf("\n  Gene propagation for %s...\n", gene))

        # Verify W matrix dimensions match number of programs
        if (nrow(W_matrix) != length(program_names)) {
            cat(sprintf("  WARNING: W matrix has %d programs, but testing %d programs. Skipping gene propagation.\n",
                       nrow(W_matrix), length(program_names)))
        } else {
            # Get matched program loadings (same as used for program testing)
            P_matched <- program_loadings_matched
            treatment_matched <- matched_data$treat

            # Build covariate dataframe (without treatment, will add later)
            covariates_for_prop <- X_matched
            covariates_for_prop[[perturbation_col]] <- NULL  # Remove treatment column

            # Process each propagation method
            gene_results <- list()

            for (method_name in propagate_method_names) {
                cat(sprintf("    Method: %s\n", method_name))

                # Extract β coefficients for this method from program results
                beta_obs <- numeric(length(program_names))

                # Extract pseudocounts if method is ols_pseudo
                pseudocounts_obs <- NULL
                if (method_name == "ols_pseudo") {
                    pseudocounts_obs <- numeric(length(program_names))
                }

                for (i in seq_along(program_names)) {
                    coef_col <- paste0("coef_", method_name)
                    if (coef_col %in% names(results[[i]])) {
                        beta_obs[i] <- results[[i]][[coef_col]]
                    } else {
                        stop(sprintf("Coefficient %s not found in program %d results", coef_col, i))
                    }

                    # Extract pseudocount for ols_pseudo
                    if (method_name == "ols_pseudo") {
                        pseudo_col <- paste0("pseudocount_", method_name)
                        if (pseudo_col %in% names(results[[i]])) {
                            pseudocounts_obs[i] <- results[[i]][[pseudo_col]]
                        } else {
                            stop(sprintf("Pseudocount %s not found in program %d results", pseudo_col, i))
                        }
                    }
                }

                # Compute observed gene effects
                gene_effects_obs <- compute_gene_counterfactuals(
                    P_matched, W_matrix, beta_obs,
                    method = method_name,
                    pseudocounts = pseudocounts_obs
                )

                # Empirical bias correction
                bias <- empirical_bias_correction(
                    P_matched, treatment_matched, covariates_for_prop, W_matrix,
                    method = method_name,
                    n_perm = n_permutations,
                    weights = matching_weights, seed = seed,
                    ncpus = bootstrap_ncpus, verbose = FALSE
                )

                # Bias-corrected gene effects
                gene_effects_corrected <- gene_effects_obs - bias

                # Bootstrap for SE
                boot_results <- bootstrap_program_and_gene_effects(
                    P_matched, treatment_matched, covariates_for_prop, W_matrix,
                    method = method_name,
                    n_boot = n_bootstrap,
                    weights = matching_weights, seed = seed,
                    ncpus = bootstrap_ncpus, verbose = FALSE
                )

                # Compute SEs from bootstrap for BOTH gene and program levels
                se_gene <- apply(boot_results$gene_effects, 2, sd)
                se_program <- apply(boot_results$program_effects, 2, sd)

                # Update program results with bootstrap SE
                for (i in seq_along(program_names)) {
                    # Replace HC3 SE with bootstrap SE
                    results[[i]][[paste0("se_", method_name)]] <- se_program[i]

                    # Recompute z-stat and p-value with bootstrap SE
                    coef <- results[[i]][[paste0("coef_", method_name)]]
                    z_stat <- coef / se_program[i]

                    # Two-tailed z-test (normal distribution for bootstrap SE)
                    p_value_program <- 2 * pnorm(-abs(z_stat))

                    results[[i]][[paste0("t_stat_", method_name)]] <- z_stat  # Actually z-stat with bootstrap SE
                    results[[i]][[paste0("p_value_", method_name)]] <- p_value_program
                }

                # Compute p-values for genes
                z_stat <- gene_effects_corrected / se_gene
                p_value <- 2 * pnorm(-abs(z_stat))

                # Store results for this method
                gene_results[[method_name]] <- list(
                    gene_names = colnames(W_matrix),
                    coef = gene_effects_corrected,
                    se = se_gene,
                    z_stat = z_stat,
                    p_value = p_value
                )

                cat(sprintf("      Completed: %d genes\n", length(gene_effects_corrected)))
            }
        }
    }
    }  # End if (run_program_testing)

    # Initialize gene_results (may be populated by gene propagation above, or stay NULL)
    if (!exists("gene_results")) {
        gene_results <- NULL
    }

    # ==========================================================================
    # Direct Gene-Level DE (if enabled)
    # ==========================================================================
    direct_gene_de_results <- NULL

    if (enable_direct_gene_de && !is.null(gene_expr_matched)) {
        cat(sprintf("\n  Direct gene-level DE for %s...\n", gene))
        cat(sprintf("    Testing %d response genes\n", ncol(gene_expr_matched)))

        # Use the same methods as program testing
        direct_gene_de_results <- test_gene_expression_batch(
            gene_expr_matched, response_gene_names, X_matched,
            perturbation_col, matching_weights, pair_ids,
            regression_method_names, max_control_cells, seed,
            n_jobs = 1  # Serial within perturbation to avoid nested parallelism
        )

        # Add perturbation name
        direct_gene_de_results$perturbation <- gene

        # Add matching metadata
        direct_gene_de_results$n_treated_matched <- n_treated_matched
        direct_gene_de_results$n_control_matched <- n_control_matched

        cat(sprintf("    Completed direct gene DE: %d tests\n", nrow(direct_gene_de_results)))
    }

    # Build results_df (may be empty if de_level == "genes")
    if (length(results) > 0) {
        results_df <- do.call(rbind, lapply(results, as.data.frame))
    } else {
        results_df <- data.frame()
    }

    # Extract program times
    program_times <- sapply(results, function(r) if("program_time" %in% names(r)) r$program_time else NA)

    return(list(
        results_df = results_df,
        matched_cells_data = matched_cells_data,
        balance_metrics = balance_metrics,
        program_times = program_times,
        gene_results = gene_results,
        direct_gene_de_results = direct_gene_de_results
    ))
}

# =============================================================================
# Main execution
# =============================================================================

cat("\nLoading data from TSV files (fast native R loading)...\n")

# Get directory containing TSV files
data_dir <- dirname(args$input)

# 1. Load cell metadata
cat("  Loading cell metadata...\n")
obs_df_full <- read.table(
    file.path(args$cell_metadata),
    sep = "\t", header = TRUE, row.names = 1,
    stringsAsFactors = FALSE, check.names = FALSE
)

# 2. Load program loadings from cNMF usages file
cat("  Loading program loadings from cNMF usages...\n")
program_loadings_full <- read.table(
    args$cnmf_usages,
    sep = "\t", header = TRUE, row.names = 1, check.names = FALSE
)

# 3. Merge on shared cells (intersection)
cat("  Merging on shared cells...\n")
cat(sprintf("    Cells in metadata: %d\n", nrow(obs_df_full)))
cat(sprintf("    Cells in cNMF: %d\n", nrow(program_loadings_full)))

common_cells <- intersect(rownames(obs_df_full), rownames(program_loadings_full))
cat(sprintf("    Shared cells: %d\n", length(common_cells)))

if (length(common_cells) == 0) {
    stop("ERROR: No shared cells between metadata and cNMF usages!")
}

# Subset to common cells
obs_df <- obs_df_full[common_cells, , drop = FALSE]
program_loadings <- as.matrix(program_loadings_full[common_cells, , drop = FALSE])
program_names <- colnames(program_loadings)

# 4. Load gene perturbations (sparse COO format) and filter to common cells
cat("  Loading gene perturbations (sparse)...\n")
gene_pert_coo <- read.table(
    file.path(args$gene_perturbations_sparse),
    sep = "\t", header = TRUE
)

# Map old cell indices to new indices (only keeping common cells)
old_cell_barcodes <- rownames(obs_df_full)
new_cell_idx_map <- setNames(seq_along(common_cells), common_cells)

# Filter sparse entries to only common cells and remap indices
gene_pert_filtered <- gene_pert_coo[gene_pert_coo$cell_idx < length(old_cell_barcodes), ]
gene_pert_filtered$cell_barcode <- old_cell_barcodes[gene_pert_filtered$cell_idx + 1]
gene_pert_filtered <- gene_pert_filtered[gene_pert_filtered$cell_barcode %in% common_cells, ]
gene_pert_filtered$new_cell_idx <- new_cell_idx_map[gene_pert_filtered$cell_barcode]

# Rebuild sparse matrix with new dimensions
gene_perturbations <- sparseMatrix(
    i = gene_pert_filtered$new_cell_idx,
    j = gene_pert_filtered$gene_idx + 1,  # R is 1-indexed
    x = gene_pert_filtered$value,
    dims = c(length(common_cells), max(gene_pert_filtered$gene_idx) + 1)
)

# 5. Load gene/NT group names
gene_names <- readLines(args$perturbation_names)

# 6. Load gene expression for direct gene-level DE (if enabled)
gene_expr_matrix <- NULL
response_gene_names <- NULL
if (enable_direct_gene_de) {
    cat("  Loading gene expression for direct gene DE...\n")
    gene_expr_data <- load_gene_expression(
        args$gene_expression,
        args$response_gene_names,
        common_cells
    )
    gene_expr_matrix <- gene_expr_data$expr_matrix
    response_gene_names <- gene_expr_data$gene_names
    cat(sprintf("  Response genes for DE: %d\n", length(response_gene_names)))
}

cat(sprintf("  Total cells: %d\n", nrow(obs_df)))
cat(sprintf("  Genes in dataset: %d\n", length(gene_names)))
cat(sprintf("  Programs to test: %d\n", length(program_names)))

# Parse covariates
covariate_names <- if (args$covariates != "") strsplit(args$covariates, "\\s+")[[1]] else c()
log_transform_names <- if (args$log_transform_covariates != "") strsplit(args$log_transform_covariates, "\\s+")[[1]] else c()
categorical_covariate_names <- if (args$categorical_covariates != "") strsplit(args$categorical_covariates, "\\s+")[[1]] else c()

cat("\nCovariates:", paste(covariate_names, collapse = ", "), "\n")
cat("Log-transform:", paste(log_transform_names, collapse = ", "), "\n")
cat("Categorical:", paste(categorical_covariate_names, collapse = ", "), "\n")

# Parse regression methods
regression_method_names <- if (args$methods != "") strsplit(args$methods, " ")[[1]] else c("ols_pseudo")
cat("Regression methods:", paste(regression_method_names, collapse = ", "), "\n")

# Conditionally load packages based on methods
if (any(grepl("hurdle", regression_method_names))) {
    suppressPackageStartupMessages(library(twopartm))
    cat("Loaded twopartm for hurdle models\n")
}

# Parse matching calipers
calipers_dict <- NULL
if (args$matching_calipers != "") {
    caliper_pairs <- strsplit(args$matching_calipers, " ")[[1]]
    caliper_names <- character(length(caliper_pairs))
    caliper_values <- numeric(length(caliper_pairs))
    for (i in seq_along(caliper_pairs)) {
        parts <- strsplit(caliper_pairs[i], ":")[[1]]
        caliper_names[i] <- parts[1]
        caliper_values[i] <- as.numeric(parts[2])
    }
    # Create named numeric vector (not list!)
    calipers_dict <- caliper_values
    names(calipers_dict) <- caliper_names
    cat("Matching calipers:", paste(names(calipers_dict), calipers_dict, sep = "=", collapse = ", "), "\n")
}

# Load genes to test from batch file
cat("\nLoading genes from batch file:", args$gene_batch_file, "\n")
genes_to_test <- readLines(args$gene_batch_file)
genes_to_test <- genes_to_test[genes_to_test != ""]
cat(sprintf("  Loaded %d genes from batch file\n", length(genes_to_test)))

# Build gene name to index mapping
gene_name_to_idx <- setNames(seq_along(gene_names), gene_names)

# Pre-compute covariate matrix
cat("\n" %R% strrep("=", 80), "\n")
cat("Pre-computing covariate matrix\n")
cat(strrep("=", 80), "\n")
covariate_matrix <- build_covariate_matrix(obs_df, covariate_names, log_transform_names, categorical_covariate_names)
cat(sprintf("  Covariate matrix shape: %d x %d\n", nrow(covariate_matrix), ncol(covariate_matrix)))
cat("  Columns:", paste(colnames(covariate_matrix), collapse = ", "), "\n")

# Add log-transformed columns back to obs_df for balance assessment
# This ensures balance metrics are computed for both raw and log versions
for (col in colnames(covariate_matrix)) {
    if (startsWith(col, "log_") && !(col %in% colnames(obs_df))) {
        obs_df[[col]] <- covariate_matrix[[col]]
    }
}

# Setup matched cells directory if needed
matched_cells_dir <- NULL
if (args$save_matched_cells) {
    matched_cells_dir <- file.path(args$output_dir, "matched_cells")
    dir.create(matched_cells_dir, recursive = TRUE, showWarnings = FALSE)
    cat("  Saving matched cells to:", matched_cells_dir, "\n")
}

# Run DE testing in parallel
cat("\n" %R% strrep("=", 80), "\n")
cat(sprintf("Testing %d genes using %d parallel jobs\n", length(genes_to_test), args$n_jobs))
cat(strrep("=", 80), "\n")

# Pre-compute has_any_guide mask (same for all genes, avoids redundant computation)
cat("Pre-computing has_any_guide mask...\n")
has_any_guide_global <- Matrix::rowSums(gene_perturbations) > 0
cat(sprintf("  %d cells with guides, %d without\n",
            sum(has_any_guide_global), sum(!has_any_guide_global)))

# Run tests in SERIAL mode only (parallelism disabled to avoid hangs)
# n_jobs parameter is kept for backwards compatibility but ignored
all_results <- list()
all_program_times <- c()  # Track all program execution times for slow node detection
for (i in seq_along(genes_to_test)) {
    gene <- genes_to_test[i]
    gene_start_time <- Sys.time()

    cat(sprintf("\n[%d/%d] Testing gene: %s (started at %s)\n",
                i, length(genes_to_test), gene,
                format(gene_start_time, "%H:%M:%S")))

    gene_idx <- gene_name_to_idx[[gene]]
    result <- test_single_gene(
        gene, gene_idx, obs_df, gene_perturbations, program_loadings,
        covariate_matrix, program_names,
        args$matching_ratio, args$matching_method, calipers_dict,
        args$max_control_cells, args$seed, matched_cells_dir,
        args$min_cells_per_gene, has_any_guide_global,
        # Gene propagation parameters
        W_matrix = W_matrix,
        enable_gene_propagation = enable_gene_propagation,
        n_permutations = args$n_permutations,
        propagate_method_names = propagate_method_names,
        n_bootstrap = args$n_bootstrap,
        bootstrap_ncpus = args$bootstrap_ncpus,
        # Direct gene-level DE parameters
        gene_expr_matrix = gene_expr_matrix,
        response_gene_names = response_gene_names,
        enable_direct_gene_de = enable_direct_gene_de,
        de_level = args$de_level
    )

    gene_elapsed <- as.numeric(difftime(Sys.time(), gene_start_time, units = "secs"))

    # Print completion status
    has_program_results <- !is.null(result$results_df) && nrow(result$results_df) > 0
    has_gene_de_results <- !is.null(result$direct_gene_de_results) && nrow(result$direct_gene_de_results) > 0

    if (has_program_results || has_gene_de_results) {
        status_parts <- c()
        if (has_program_results) {
            status_parts <- c(status_parts, sprintf("%d programs", nrow(result$results_df)))
        }
        if (has_gene_de_results) {
            status_parts <- c(status_parts, sprintf("%d gene DE tests", nrow(result$direct_gene_de_results)))
        }
        cat(sprintf("  ✓ Completed: %s (%.1f seconds, %.2f min total)\n",
                    paste(status_parts, collapse = ", "), gene_elapsed, gene_elapsed / 60))

        # Collect program times for slow node detection
        if (!is.null(result$program_times)) {
            all_program_times <- c(all_program_times, result$program_times)
        }
    } else {
        cat(sprintf("  ✗ Skipped: insufficient data (%.1f seconds)\n", gene_elapsed))
    }

    all_results[[i]] <- result

    # Slow node detection: Check performance after first gene
    # Allow warmup for first gene, then check if running too slowly
    if (i == 1 && length(all_program_times) > 0) {
        # Skip first program (warmup), compute median of remaining programs
        if (length(all_program_times) > 1) {
            median_time <- median(all_program_times[2:length(all_program_times)])

            # Expected: ~0.7s after warmup, threshold: >4s indicates slow node
            slow_threshold <- 4.0

            if (median_time > slow_threshold) {
                cat("\n")
                cat(strrep("=", 80), "\n")
                cat("SLOW NODE DETECTED - TERMINATING EARLY\n")
                cat(strrep("=", 80), "\n")
                cat(sprintf("Median program time after warmup: %.2fs (expected: <1s)\n", median_time))
                cat(sprintf("This node is running %.1fx slower than expected.\n", median_time / 0.7))
                cat("This job is likely on a node with:\n")
                cat("  - CPU frequency throttling\n")
                cat("  - Thermal throttling\n")
                cat("  - Core oversubscription\n")
                cat("  - Unoptimized BLAS library\n")
                cat("\nTerminating to avoid wasting 6+ hours on slow node.\n")
                cat("SLURM can requeue this job to run on a different node.\n")
                cat(strrep("=", 80), "\n")
                quit(status = 1)
            } else {
                cat(sprintf("\n  Performance check: Median program time = %.2fs (healthy)\n", median_time))
            }
        }
    }
}

# Collect results
results_dfs <- list()
all_matched_cells <- list()
all_balance_metrics <- list()
all_gene_results <- list()
all_direct_gene_de_results <- list()

for (result_tuple in all_results) {
    if (!is.null(result_tuple$results_df) && nrow(result_tuple$results_df) > 0) {
        results_dfs[[length(results_dfs) + 1]] <- result_tuple$results_df
    }
    if (length(result_tuple$matched_cells_data) > 0) {
        all_matched_cells <- c(all_matched_cells, result_tuple$matched_cells_data)
    }
    if (!is.null(result_tuple$balance_metrics) && nrow(result_tuple$balance_metrics) > 0) {
        all_balance_metrics[[length(all_balance_metrics) + 1]] <- result_tuple$balance_metrics
    }
    # Collect gene results (if gene propagation was enabled)
    if (!is.null(result_tuple$gene_results) && length(result_tuple$gene_results) > 0) {
        # Extract perturbation name from results_df
        if (!is.null(result_tuple$results_df) && nrow(result_tuple$results_df) > 0) {
            perturbation_name <- result_tuple$results_df$perturbation[1]
            all_gene_results[[perturbation_name]] <- result_tuple$gene_results
        }
    }
    # Collect direct gene DE results
    if (!is.null(result_tuple$direct_gene_de_results) && nrow(result_tuple$direct_gene_de_results) > 0) {
        all_direct_gene_de_results[[length(all_direct_gene_de_results) + 1]] <- result_tuple$direct_gene_de_results
    }
}

# Check if we have any results (program or direct gene DE)
if (length(results_dfs) == 0 && length(all_direct_gene_de_results) == 0) {
    cat("\nERROR: No valid results from testing!\n")
    quit(status = 1)
}

# Combine program results (may be empty if de_level == "genes")
if (length(results_dfs) > 0) {
    results_df <- do.call(rbind, results_dfs)
    cat(sprintf("\nCompleted testing %d target-program pairs\n", nrow(results_df)))
} else {
    results_df <- data.frame()
    cat("\nNo program-level results (de_level may be 'genes')\n")
}

# Save results
cat("\n" %R% strrep("=", 80), "\n")
cat("Saving results\n")
cat(strrep("=", 80), "\n")

# Save program-level results (only if not empty)
if (nrow(results_df) > 0) {
    output_file <- file.path(args$output_dir, sprintf("%s_batch_%s_results.tsv", args$test_type, args$batch_id))
    cat("Saving program batch results to", output_file, "...\n")
    write.table(results_df, output_file, sep = "\t", row.names = FALSE, quote = FALSE)
}

# Save direct gene DE results (if any)
if (length(all_direct_gene_de_results) > 0) {
    cat("\nSaving direct gene-level DE results...\n")
    direct_gene_de_combined <- do.call(rbind, all_direct_gene_de_results)
    gene_de_output_file <- file.path(args$output_dir, sprintf("%s_batch_%s_direct_gene_de_limma.parquet", args$test_type, args$batch_id))
    nanoparquet::write_parquet(direct_gene_de_combined, gene_de_output_file, compression = "snappy")
    cat(sprintf("  Saved %d direct gene DE results to %s\n", nrow(direct_gene_de_combined), gene_de_output_file))
}

# Write matched cells
if (length(all_matched_cells) > 0 && !is.null(matched_cells_dir)) {
    matched_cells_file <- file.path(matched_cells_dir, sprintf("%s_batch_%s_matched_cells.txt", args$test_type, args$batch_id))
    cat(sprintf("\nSaving %d matched cell records to %s...\n", length(all_matched_cells), matched_cells_file))
    matched_cells_df <- do.call(rbind, lapply(all_matched_cells, as.data.frame))
    write.table(matched_cells_df, matched_cells_file, sep = "\t", row.names = FALSE, quote = FALSE)
}

# Write balance metrics
if (length(all_balance_metrics) > 0) {
    balance_combined <- do.call(rbind, all_balance_metrics)
    balance_dir <- file.path(args$output_dir, "diagnostics", "covariate_balance")
    dir.create(balance_dir, recursive = TRUE, showWarnings = FALSE)

    balance_file <- file.path(balance_dir, sprintf("%s_batch_%s_balance.tsv", args$test_type, args$batch_id))
    cat(sprintf("\nSaving balance metrics for %d perturbations to %s...\n",
                length(unique(balance_combined$perturbation)), balance_file))
    write.table(balance_combined, balance_file, sep = "\t", row.names = FALSE, quote = FALSE)
}

# Write gene propagation results (NPZ format)
if (length(all_gene_results) > 0) {
    cat("\n" %R% strrep("=", 80), "\n")
    cat("Saving gene propagation results\n")
    cat(strrep("=", 80), "\n")

    # Get list of perturbations and genes
    perturbation_names <- names(all_gene_results)
    n_perturbations <- length(perturbation_names)

    # Get method names from first perturbation
    method_names_gene <- names(all_gene_results[[1]])

    # Get gene names (same for all perturbations and methods)
    gene_names_all <- all_gene_results[[1]][[method_names_gene[1]]]$gene_names
    n_genes <- length(gene_names_all)

    cat(sprintf("  Perturbations: %d\n", n_perturbations))
    cat(sprintf("  Genes: %d\n", n_genes))
    cat(sprintf("  Methods: %s\n", paste(method_names_gene, collapse = ", ")))

    # For each method, create matrices and save NPZ
    for (method_name in method_names_gene) {
        cat(sprintf("  Processing method: %s\n", method_name))

        # Initialize matrices (perturbations × genes)
        coef_matrix <- matrix(NA, nrow = n_perturbations, ncol = n_genes)
        se_matrix <- matrix(NA, nrow = n_perturbations, ncol = n_genes)
        p_value_matrix <- matrix(NA, nrow = n_perturbations, ncol = n_genes)

        # Fill matrices
        for (i in seq_along(perturbation_names)) {
            pert <- perturbation_names[i]
            if (method_name %in% names(all_gene_results[[pert]])) {
                method_res <- all_gene_results[[pert]][[method_name]]

                coef_matrix[i, ] <- method_res$coef
                se_matrix[i, ] <- method_res$se
                p_value_matrix[i, ] <- method_res$p_value
            }
        }

        # Save as Parquet (compressed, R and Python compatible)
        output_base <- file.path(args$output_dir, sprintf("%s_batch_%s_propagated", args$test_type, args$batch_id))
        save_gene_results_parquet(perturbation_names, gene_names_all,
                                   p_value_matrix, coef_matrix, se_matrix,
                                   method_name, output_base)
    }

    cat(sprintf("  Completed: %d methods saved\n", length(method_names_gene)))
}

# Summary
cat("\n" %R% strrep("=", 80), "\n")
cat("Summary\n")
cat(strrep("=", 80), "\n")
cat("Condition:", args$condition, "\n")
cat("DE level:", args$de_level, "\n")

# Program-level summary (if applicable)
if (nrow(results_df) > 0) {
    cat("Targets tested (programs):", length(unique(results_df$perturbation)), "\n")
    cat("Programs tested:", length(unique(results_df$program)), "\n")
    cat("Total program tests:", nrow(results_df), "\n")

    # Print median p-values for each method
    for (method_name in regression_method_names) {
        p_col <- paste0("p_value_", method_name)
        if (p_col %in% colnames(results_df)) {
            median_p <- median(results_df[[p_col]])
            cat(sprintf("Median p-value programs (%s): %.6f\n", method_name, median_p))
        }
    }
}

# Direct gene DE summary (if applicable)
if (length(all_direct_gene_de_results) > 0) {
    direct_gene_de_combined <- do.call(rbind, all_direct_gene_de_results)
    cat("Perturbations tested (gene DE):", length(unique(direct_gene_de_combined$perturbation)), "\n")
    cat("Response genes tested:", length(unique(direct_gene_de_combined$response_gene)), "\n")
    cat("Total direct gene DE tests:", nrow(direct_gene_de_combined), "\n")
}

cat(strrep("=", 80), "\n")

cat("\nDone!\n")
