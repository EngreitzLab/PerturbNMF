# =============================================================================
# Differential Expression Testing Utilities
# =============================================================================
# Reusable functions for DE testing with propensity score matching and gene
# propagation. Separated from main workflow for modularity and testability.
#
# Function categories:
# 1. Helper functions: winsorize, collinearity removal, covariate building
# 2. Matching functions: propensity score matching, balance assessment
# 3. Regression functions: OLS/Huber fitting, transformations, bootstrap
# 4. Gene propagation: counterfactual effects, bias correction, joint bootstrap
#
# Usage: source("de_testing_utils.R") in main DE testing scripts
# =============================================================================

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Winsorize non-zero values to percentile
winsorize_nonzero <- function(x, percentile) {
    nonzero_mask <- x > 0
    if (sum(nonzero_mask) == 0) return(x)

    nonzero_vals <- x[nonzero_mask]
    cutoff <- quantile(nonzero_vals, probs = percentile / 100)
    x[nonzero_mask & x > cutoff] <- cutoff
    return(x)
}

# Check for collinearity and remove problematic covariates
remove_collinear_covariates <- function(data, covariate_cols, threshold, verbose) {
    numeric_cols <- covariate_cols[sapply(data[covariate_cols], is.numeric)]

    if (length(numeric_cols) <= 1) {
        return(covariate_cols)
    }

    # Compute Pearson correlation matrix
    cor_matrix <- cor(data[, numeric_cols, drop = FALSE], method = "pearson")

    # Find highly collinear pairs (|r| > threshold) and drop the second variable
    cols_to_drop <- c()
    for (i in 1:(length(numeric_cols) - 1)) {
        for (j in (i + 1):length(numeric_cols)) {
            col1 <- numeric_cols[i]
            col2 <- numeric_cols[j]

            if (col1 %in% cols_to_drop || col2 %in% cols_to_drop) {
                next
            }

            if (abs(cor_matrix[i, j]) > threshold) {
                cols_to_drop <- c(cols_to_drop, col2)
                if (verbose) {
                    cat(sprintf("    WARNING: Dropping %s due to collinearity with %s (r=%.4f)\n",
                               col2, col1, cor_matrix[i, j]))
                }
            }
        }
    }

    # Remove collinear covariates
    if (length(cols_to_drop) > 0) {
        covariate_cols <- setdiff(covariate_cols, cols_to_drop)
    }

    return(covariate_cols)
}

# Build covariate matrix with transformations
build_covariate_matrix <- function(obs_df, covariate_names, log_transform_names, categorical_covariates) {
    if (length(covariate_names) == 0) {
        return(data.frame(intercept = rep(1, nrow(obs_df))))
    }

    cov_matrix <- obs_df[, covariate_names, drop = FALSE]

    # Log-transform specified covariates (keep both raw and log versions)
    for (cov_name in log_transform_names) {
        if (cov_name %in% colnames(cov_matrix)) {
            log_col_name <- paste0("log_", cov_name)
            cov_matrix[[log_col_name]] <- log1p(cov_matrix[[cov_name]])
        }
    }

    # Check for constant covariates (zero variance) - these cause problems in matching and regression
    for (col in colnames(cov_matrix)) {
        if (is.numeric(cov_matrix[[col]])) {
            col_var <- var(cov_matrix[[col]])
            if (is.na(col_var) || col_var == 0) {
                stop(sprintf("ERROR: Covariate '%s' is constant (zero variance). All values are %.4f. Constant covariates cannot be used in matching or regression.",
                            col, mean(cov_matrix[[col]])))
            }
        } else {
            # For categorical/character columns, check if only 1 unique value
            n_unique <- length(unique(cov_matrix[[col]][!is.na(cov_matrix[[col]])]))
            if (n_unique <= 1) {
                stop(sprintf("ERROR: Categorical covariate '%s' has only %d unique value(s). Constant covariates cannot be used in matching or regression.",
                            col, n_unique))
            }
        }
    }

    # Convert categorical covariates to factors
    for (col in categorical_covariates) {
        if (col %in% colnames(cov_matrix)) {
            cov_matrix[[col]] <- as.factor(cov_matrix[[col]])
        }
    }

    # Z-score standardize continuous covariates (if > 2 unique values)
    for (col in colnames(cov_matrix)) {
        if (is.numeric(cov_matrix[[col]])) {
            n_unique <- length(unique(cov_matrix[[col]]))
            if (n_unique > 2) {
                std_val <- sd(cov_matrix[[col]])
                if (std_val > 0) {
                    cov_matrix[[col]] <- (cov_matrix[[col]] - mean(cov_matrix[[col]])) / std_val
                }
            }
        }
    }

    return(cov_matrix)
}

# Validate sample sizes
validate_sample_sizes <- function(treatment, min_cells, context = "") {
    n_treated <- sum(treatment == 1)
    n_control <- sum(treatment == 0)

    is_valid <- (n_treated >= min_cells) && (n_control >= min_cells)

    if (!is_valid) {
        cat(sprintf("[%s] Insufficient sample size: treated=%d, control=%d, min=%d\n",
                   context, n_treated, n_control, min_cells))
    }

    return(list(is_valid = is_valid, n_treated = n_treated, n_control = n_control))
}

# Compute contingency table for separation detection
compute_contingency_table <- function(y, X_df, perturbation_col) {
    treatment <- X_df[[perturbation_col]]

    treated_zero <- sum(treatment == 1 & y == 0)
    treated_pos <- sum(treatment == 1 & y > 0)
    control_zero <- sum(treatment == 0 & y == 0)
    control_pos <- sum(treatment == 0 & y > 0)

    return(list(
        treated_zero = treated_zero,
        treated_pos = treated_pos,
        control_zero = control_zero,
        control_pos = control_pos
    ))
}


# =============================================================================
# MATCHING FUNCTIONS
# =============================================================================

# Perform propensity score matching
perform_matching <- function(X_df, treatment_col, matching_ratio, matching_method,
                            calipers_dict, max_control_cells, seed, verbose = FALSE) {

    # Extract treatment indicator
    treatment <- X_df[[treatment_col]]
    n_treated <- sum(treatment == 1)
    n_control <- sum(treatment == 0)

    if (verbose) {
        cat(sprintf("    Matching: %d treated, %d controls available\n", n_treated, n_control))
    }

    # Prepare data for matching (rename treatment column to 'treat')
    match_data <- X_df
    match_data$treat <- treatment

    # Build formula (treat ~ all other covariates except treatment column)
    covariate_cols <- setdiff(colnames(X_df), c(treatment_col, "intercept"))

    # Check for high collinearity (Pearson r > 0.95), but allow log/non-log pairs
    covariate_cols <- remove_collinear_covariates(match_data, covariate_cols, threshold = 0.95, verbose = verbose)

    if (length(covariate_cols) == 0) {
        stop("No valid covariates for matching")
    }

    formula_str <- as.formula(paste("treat ~", paste(covariate_cols, collapse = " + ")))

    # Perform matching
    if (!is.null(calipers_dict) && length(calipers_dict) > 0) {
        matchit_result <- matchit(
            formula_str,
            data = match_data,
            method = matching_method,
            replace = FALSE,
            ratio = matching_ratio,
            min.controls = 50,
            caliper = calipers_dict
        )
    } else {
        matchit_result <- matchit(
            formula_str,
            data = match_data,
            method = matching_method,
            replace = FALSE,
            ratio = matching_ratio,
            min.controls = 50
        )
    }

    # Extract matched data
    matched_data <- match.data(matchit_result)
    matched_indices <- rownames(matched_data)

    # Get propensity scores for all cells
    distance_map <- matchit_result$distance
    names(distance_map) <- rownames(X_df)

    if (verbose) {
        n_matched_treated <- sum(matched_data$treat == 1)
        cat(sprintf("    Matched %d cells (%d treated)\n", nrow(matched_data), n_matched_treated))
    }

    return(list(
        matched_data = matched_data,
        matched_indices = matched_indices,
        n_treated = sum(matched_data$treat == 1),
        n_control = sum(matched_data$treat == 0),
        distance_map = distance_map,
        covariates_used = covariate_cols
    ))
}

# Compute comprehensive covariate balance
compute_comprehensive_balance <- function(matched_indices, treatment, obs_df) {
    # Extract matched cells
    matched_obs <- obs_df[matched_indices, , drop = FALSE]

    # Build exclusion set
    exclude_cols <- c('cell_barcode', 'sample_id', 'round1', 'well', 'biological_sample',
                     'guide_threshold', 'Notes', 'mito_status', 'guide_moi', 'total_counts_mt')

    # Get all columns and filter
    all_cols <- colnames(matched_obs)
    covariate_cols <- setdiff(all_cols, exclude_cols)
    covariate_cols <- covariate_cols[!grepl("^perturb_", covariate_cols)]

    # Filter to valid covariates (more than 1 unique value)
    valid_covariates <- c()
    for (col in covariate_cols) {
        n_unique <- length(unique(matched_obs[[col]]))
        if (n_unique <= 1) next

        # Exclude ID-like columns (every row unique)
        if (is.character(matched_obs[[col]]) || is.factor(matched_obs[[col]])) {
            if (n_unique >= nrow(matched_obs)) next
        }

        valid_covariates <- c(valid_covariates, col)
    }

    if (length(valid_covariates) == 0) {
        return(NULL)
    }

    # Build dataframe for cobalt
    balance_data <- matched_obs[, valid_covariates, drop = FALSE]
    balance_data$treat <- treatment

    # Standardize continuous covariates
    for (col in valid_covariates) {
        if (is.numeric(balance_data[[col]])) {
            n_unique <- length(unique(balance_data[[col]]))
            if (n_unique > 2) {
                std_val <- sd(balance_data[[col]])
                if (std_val > 0) {
                    balance_data[[col]] <- (balance_data[[col]] - mean(balance_data[[col]])) / std_val
                }
            }
        }
    }

    # Remove rows with any NaN
    balance_data <- na.omit(balance_data)

    # Re-check variation after dropna
    final_covariates <- c()
    for (col in valid_covariates) {
        if (col %in% colnames(balance_data)) {
            if (length(unique(balance_data[[col]])) > 1) {
                final_covariates <- c(final_covariates, col)
            }
        }
    }

    if (length(final_covariates) == 0) {
        return(NULL)
    }

    # Build formula
    formula_str <- as.formula(paste("treat ~", paste(final_covariates, collapse = " + ")))

    # Compute balance using cobalt
    bal_result <- bal.tab(
        formula_str,
        data = balance_data,
        thresholds = c(m = 0.1),
        stats = c("m", "v"),
        s.d.denom = "pooled"
    )

    balance_df <- bal_result$Balance
    balance_df$variable <- rownames(balance_df)

    return(balance_df)
}


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

# Define transformation functions for each regression method
TRANSFORMATION_FUNCTIONS <- list(
    ols_log = function(y) log1p(y * 1e6),
    ols_pseudo = function(y) {
        y_scaled <- y * 1e6
        nonzero_vals <- y_scaled[y_scaled > 0]
        if (length(nonzero_vals) > 0) {
            pseudo <- quantile(nonzero_vals, probs = 0.05)
        } else {
            pseudo <- 1
        }
        log(y_scaled + pseudo)
    },
    hurdle_marginal = function(y) log1p(y * 1e6)  # Scale then log-transform for two-part model
)

# Define inverse transformation functions for each regression method
# These convert from transformed scale back to native scale
INVERSE_TRANSFORMATION_FUNCTIONS <- list(
    ols_log = function(T_val, scale = 1e6, pseudo = NULL) {
        # T = log1p(y * scale) => y = (exp(T) - 1) / scale
        return((exp(T_val) - 1) / scale)
    },
    ols_pseudo = function(T_val, scale = 1e6, pseudo = NULL) {
        # T = log(y * scale + pseudo) => y = (exp(T) - pseudo) / scale
        if (is.null(pseudo)) {
            stop("Pseudocount required for ols_pseudo inverse transformation")
        }
        return((exp(T_val) - pseudo) / scale)
    },
    hurdle_marginal = function(T_val, scale = 1e6, pseudo = NULL) {
        # Same as ols_log
        return((exp(T_val) - 1) / scale)
    }
)

# Compute pseudocount for ols_pseudo transformation (stable across dataset)
compute_pseudocount_ols_pseudo <- function(y) {
    y_scaled <- y * 1e6
    nonzero_vals <- y_scaled[y_scaled > 0]
    if (length(nonzero_vals) > 0) {
        pseudo <- quantile(nonzero_vals, probs = 0.05)
    } else {
        pseudo <- 1
    }
    return(pseudo)
}

# Fit OLS regression with matching weights and cluster-robust standard errors
fit_linear_model_robust <- function(y, X_df, perturbation_col, max_control_cells, seed, weights = NULL, cluster_ids = NULL, compute_se = TRUE) {
    # Find perturbation column
    perturb_cols <- grep(perturbation_col, colnames(X_df), value = TRUE)
    if (length(perturb_cols) == 0) {
        return(list(
            coef = NA, se = NA, t_stat = NA, p_value = NA,
            n_obs = NA, n_treated = NA, error = "perturbation column not found"
        ))
    }
    perturb_col <- perturb_cols[1]

    # Check for NAs - these should never happen with clean data
    # If found, crash the pipeline to surface data quality issues
    if (any(is.na(y)) || !all(complete.cases(X_df))) {
        stop(sprintf("Found NA values in regression data. This indicates a data quality issue.\n  NA in outcome: %d/%d\n  Incomplete covariate cases: %d/%d",
                     sum(is.na(y)), length(y), sum(!complete.cases(X_df)), nrow(X_df)))
    }

    # Use all data (no NA filtering needed)
    y_clean <- y
    X_clean <- X_df
    weights_clean <- weights
    cluster_ids_clean <- cluster_ids

    # No minimum observation check - already validated at matching stage (min_cells_per_gene)

    # Subsample control cells if needed
    treatment <- X_clean[[perturb_col]]
    n_control <- sum(treatment == 0)

    if (n_control > max_control_cells) {
        set.seed(seed)
        control_indices <- which(treatment == 0)
        sampled_control_indices <- sample(control_indices, size = max_control_cells, replace = FALSE)

        keep_indices <- c(which(treatment == 1), sampled_control_indices)
        X_clean <- X_clean[keep_indices, , drop = FALSE]
        y_clean <- y_clean[keep_indices]
        weights_clean <- if (!is.null(weights_clean)) weights_clean[keep_indices] else NULL
        cluster_ids_clean <- if (!is.null(cluster_ids_clean)) cluster_ids_clean[keep_indices] else NULL
    }

    # Check for collinearity in covariates (Pearson r > 0.95), but allow log/non-log pairs
    covariate_cols <- colnames(X_clean)
    filtered_cols <- remove_collinear_covariates(X_clean, covariate_cols, threshold = 0.95, verbose = FALSE)
    if (length(filtered_cols) < length(covariate_cols)) {
        # Remove collinear covariates
        X_clean <- X_clean[, filtered_cols, drop = FALSE]
    }

    # Fit OLS model (weighted if weights provided)
    if (!is.null(weights_clean)) {
        model <- lm(y_clean ~ ., data = X_clean, weights = weights_clean)
    } else {
        model <- lm(y_clean ~ ., data = X_clean)
    }

    # Extract perturbation coefficient
    model_coef <- coef(model)
    perturb_idx <- which(names(model_coef) == perturb_col)
    if (length(perturb_idx) == 0) {
        stop(sprintf("Coefficient for %s not found in model", perturb_col))
    }

    coef <- model_coef[perturb_idx]

    # Compute robust standard errors if requested (skip for bootstrap)
    if (compute_se) {
        # Compute robust standard errors (cluster-robust if cluster_ids provided, otherwise HC3)
        if (!is.null(cluster_ids_clean)) {
            robust_se <- coeftest(model, vcov. = vcovCL(model, cluster = cluster_ids_clean))
        } else {
            robust_se <- coeftest(model, vcov. = vcovHC(model, type = "HC3"))
        }

        se <- robust_se[perturb_idx, "Std. Error"]
        t_stat <- robust_se[perturb_idx, "t value"]
        p_value <- robust_se[perturb_idx, "Pr(>|t|)"]
    } else {
        # Skip SE calculation (will use bootstrap SE instead)
        se <- NA
        t_stat <- NA
        p_value <- NA
    }

    return(list(
        coef = coef,
        se = se,
        t_stat = t_stat,
        p_value = p_value,
        n_obs = length(y_clean),
        n_treated = sum(X_clean[[perturb_col]]),
        model = model
    ))
}

# Fit two-part hurdle model with marginal effects using twopartm
fit_twopartm_hurdle <- function(y, X_df, perturbation_col, max_control_cells, seed,
                                weights = NULL, cluster_ids = NULL, compute_se = TRUE) {
    # Find perturbation column
    perturb_cols <- grep(perturbation_col, colnames(X_df), value = TRUE)
    if (length(perturb_cols) == 0) {
        return(list(
            coef_binary = NA, se_binary = NA, z_stat_binary = NA, p_value_binary = NA,
            coef_mag = NA, se_mag = NA, t_stat_mag = NA, p_value_mag = NA,
            coef_marginal = NA, se_marginal = NA, z_stat_marginal = NA, p_value_marginal = NA,
            n_obs = NA, n_treated = NA, n_nonzero = NA, error = "perturbation column not found"
        ))
    }
    perturb_col <- perturb_cols[1]

    # Check for NAs - these should never happen with clean data
    # If found, crash the pipeline to surface data quality issues
    if (any(is.na(y)) || !all(complete.cases(X_df))) {
        stop(sprintf("Found NA values in hurdle model data. This indicates a data quality issue.\n  NA in outcome: %d/%d\n  Incomplete covariate cases: %d/%d",
                     sum(is.na(y)), length(y), sum(!complete.cases(X_df)), nrow(X_df)))
    }

    # Use all data (no NA filtering needed)
    y_clean <- y
    X_clean <- X_df
    weights_clean <- weights

    # No minimum observation check - already validated at matching stage (min_cells_per_gene)

    # Subsample control cells if needed
    treatment <- X_clean[[perturb_col]]
    n_control <- sum(treatment == 0)

    if (n_control > max_control_cells) {
        set.seed(seed)
        control_indices <- which(treatment == 0)
        sampled_control_indices <- sample(control_indices, size = max_control_cells, replace = FALSE)

        keep_indices <- c(which(treatment == 1), sampled_control_indices)
        X_clean <- X_clean[keep_indices, , drop = FALSE]
        y_clean <- y_clean[keep_indices]
        weights_clean <- if (!is.null(weights_clean)) weights_clean[keep_indices] else NULL
    }

    # Check for collinearity
    covariate_cols <- colnames(X_clean)
    filtered_cols <- remove_collinear_covariates(X_clean, covariate_cols, threshold = 0.95, verbose = FALSE)
    if (length(filtered_cols) < length(covariate_cols)) {
        X_clean <- X_clean[, filtered_cols, drop = FALSE]
    }

    # Count non-zero observations (for reporting only - no filtering)
    n_nonzero <- sum(y_clean > 0)

    # No minimum non-zero check - rely on contingency table filtering in post-processing

    # Prepare data for twopartm
    # y is already log1p(raw * 1e6) from transformation function
    model_data <- X_clean
    model_data$y <- y_clean

    # Build formula (same for both parts)
    formula_str <- as.formula(paste("y ~", paste(colnames(X_clean), collapse = " + ")))

    # Fit two-part model with tpm
    if (!is.null(weights_clean)) {
        model <- tpm(
            formula_part1 = formula_str,
            formula_part2 = formula_str,
            data = model_data,
            link_part1 = "logit",
            family_part2 = gaussian(link = "identity"),  # Already log-transformed
            weights = weights_clean
        )
    } else {
        model <- tpm(
            formula_part1 = formula_str,
            formula_part2 = formula_str,
            data = model_data,
            link_part1 = "logit",
            family_part2 = gaussian(link = "identity")  # Already log-transformed
        )
    }

    # Extract results from model summary
    model_summary <- summary(model)

    # Binary component (part 1)
    binary_coefs <- model_summary[[1]]$coefficients
    perturb_binary_idx <- which(rownames(binary_coefs) == perturb_col)

    if (length(perturb_binary_idx) == 0) {
        coef_binary <- NA
        se_binary <- NA
        z_stat_binary <- NA
        p_value_binary <- NA
    } else {
        coef_binary <- binary_coefs[perturb_binary_idx, "Estimate"]
        if (compute_se) {
            se_binary <- binary_coefs[perturb_binary_idx, "Std. Error"]
            z_stat_binary <- binary_coefs[perturb_binary_idx, "z value"]
            p_value_binary <- binary_coefs[perturb_binary_idx, "Pr(>|z|)"]
        } else {
            se_binary <- NA
            z_stat_binary <- NA
            p_value_binary <- NA
        }
    }

    # Continuous component (part 2)
    cont_coefs <- model_summary[[2]]$coefficients
    perturb_cont_idx <- which(rownames(cont_coefs) == perturb_col)

    if (length(perturb_cont_idx) == 0) {
        coef_mag <- NA
        se_mag <- NA
        t_stat_mag <- NA
        p_value_mag <- NA
    } else {
        coef_mag <- cont_coefs[perturb_cont_idx, "Estimate"]
        if (compute_se) {
            se_mag <- cont_coefs[perturb_cont_idx, "Std. Error"]
            t_stat_mag <- cont_coefs[perturb_cont_idx, "t value"]
            p_value_mag <- cont_coefs[perturb_cont_idx, "Pr(>|t|)"]
        } else {
            se_mag <- NA
            t_stat_mag <- NA
            p_value_mag <- NA
        }
    }

    # Compute marginal effects using AME (needed for coef_marginal even when compute_se=FALSE)
    ame_result <- AME(model, term = perturb_col)

    # Extract marginal effect for perturbation variable
    treatment_row <- which(ame_result$Variable == perturb_col)

    if (length(treatment_row) == 0) {
        coef_marginal <- NA
        se_marginal <- NA
        z_stat_marginal <- NA
        p_value_marginal <- NA
    } else {
        coef_marginal <- ame_result[treatment_row, "dydx"]
        if (compute_se) {
            se_marginal <- ame_result[treatment_row, "Std.Err"]
            z_stat_marginal <- ame_result[treatment_row, "z"]
            p_value_marginal <- ame_result[treatment_row, "pvalue"]
        } else {
            se_marginal <- NA
            z_stat_marginal <- NA
            p_value_marginal <- NA
        }
    }

    # Return results
    return(list(
        coef_binary = coef_binary,
        se_binary = se_binary,
        z_stat_binary = z_stat_binary,
        p_value_binary = p_value_binary,
        coef_mag = coef_mag,
        se_mag = se_mag,
        t_stat_mag = t_stat_mag,
        p_value_mag = p_value_mag,
        coef_marginal = coef_marginal,
        se_marginal = se_marginal,
        z_stat_marginal = z_stat_marginal,
        p_value_marginal = p_value_marginal,
        n_obs = length(y_clean),
        n_treated = sum(treatment),
        n_nonzero = n_nonzero
    ))
}


# =============================================================================
# GENE PROPAGATION FUNCTIONS
# =============================================================================

load_gene_spectra_tpm <- function(spectra_path) {
    # Load cNMF gene spectra TPM matrix (non-negative NMF loadings).
    #
    # Returns:
    #     W: matrix with programs as rows, genes as columns
    cat("Loading gene spectra TPM from:", spectra_path, "\n")

    W <- read.table(spectra_path, header = TRUE, row.names = 1, sep = "\t",
                    check.names = FALSE, stringsAsFactors = FALSE)
    W <- as.matrix(W)

    cat(sprintf("  Loaded: %d programs × %d genes\n", nrow(W), ncol(W)))

    # Verify non-negative (cNMF NMF constraint)
    if (any(W < 0)) {
        stop("Gene spectra contains negative values! Should use gene_spectra_tpm.txt (not gene_spectra_score.txt)")
    }

    return(W)
}


compute_gene_counterfactuals <- function(P_cells, W, beta_programs,
                                          method = "ols_log",
                                          pseudocounts = NULL,
                                          scale = 1e6) {
    # Compute gene-level effects using counterfactual approach on shifted scale.
    #
    # For each cell c, program k, and gene g:
    #   Q_baseline[c,k] = shifted program loading (1 + P × scale OR P × scale + pseudo)
    #   Q_treatment[c,k] = Q_baseline[c,k] × exp(β[k])  (always positive!)
    #   G_shifted[c,g] = Σ_k W[k,g] × Q[c,k]
    #
    # Gene effect = mean_c[log(G_shifted_treatment) - log(G_shifted_baseline)]
    #
    # This avoids negative values from inverse transformation by working entirely
    # on the shifted scale: G_shifted = Q × W = (offset + P × scale) × W
    #
    # Parameters:
    #     P_cells: matrix (n_cells × n_programs) - program loadings per cell
    #     W: matrix (n_programs × n_genes) - gene spectra TPM
    #     beta_programs: vector (n_programs) - program-level log-fold changes
    #     method: regression method name (default "ols_log")
    #     pseudocounts: vector (n_programs) - pseudocounts per program (for ols_pseudo)
    #     scale: scaling factor (default 1e6)
    #
    # Returns:
    #     gene_effects: vector (n_genes) - log-fold change per gene
    n_cells <- nrow(P_cells)
    n_programs <- ncol(P_cells)
    n_genes <- ncol(W)

    # Validate dimensions
    if (ncol(P_cells) != nrow(W)) {
        stop(sprintf("Dimension mismatch: P_cells has %d programs, W has %d programs",
                    ncol(P_cells), nrow(W)))
    }
    if (length(beta_programs) != n_programs) {
        stop(sprintf("beta_programs length (%d) != n_programs (%d)",
                    length(beta_programs), n_programs))
    }

    # Compute Q (shifted program loadings) for baseline and treatment
    Q_baseline <- matrix(NA, nrow = n_cells, ncol = n_programs)
    Q_treatment <- matrix(NA, nrow = n_cells, ncol = n_programs)

    for (k in 1:n_programs) {
        # Compute shifted program loadings Q_baseline
        if (method == "ols_log" || method == "hurdle_marginal") {
            # Q = 1 + P × scale
            Q_baseline[, k] <- 1 + P_cells[, k] * scale
        } else if (method == "ols_pseudo") {
            # Q = P × scale + pseudo
            if (is.null(pseudocounts)) {
                stop("Pseudocounts required for ols_pseudo method")
            }
            Q_baseline[, k] <- P_cells[, k] * scale + pseudocounts[k]
        } else {
            stop(sprintf("Unknown method: %s", method))
        }

        # Apply treatment effect: Q_treatment = Q_baseline × exp(β)
        # This is always positive since Q_baseline > 0 and exp(β) > 0
        Q_treatment[, k] <- Q_baseline[, k] * exp(beta_programs[k])
    }

    # Compute gene expression on shifted scale: G_shifted = Q × W
    G_shifted_baseline <- Q_baseline %*% W  # (n_cells × n_genes)
    G_shifted_treatment <- Q_treatment %*% W  # (n_cells × n_genes)

    # Log-fold change per cell on shifted scale
    lfc_per_cell <- log(G_shifted_treatment) - log(G_shifted_baseline)

    # Average over cells
    gene_effects <- colMeans(lfc_per_cell)

    return(gene_effects)
}


fit_programs_extract_beta <- function(P_cells, treatment, covariates_df,
                                       method = "ols_log",
                                       weights = NULL) {
    # Fit program-level models and extract beta coefficients.
    #
    # For each program k:
    #     y_k = transform(P_k) based on method
    #     fit: lm(y_k ~ treatment + covariates)
    #     extract: β_k = coef["treatment"]
    #
    # Parameters:
    #     P_cells: matrix (n_cells × n_programs)
    #     treatment: vector (n_cells) - binary treatment indicator (0/1)
    #     covariates_df: data.frame with covariates (n_cells × n_covariates)
    #     method: regression method name (default "ols_log")
    #     weights: optional vector of weights (e.g., matching weights)
    #
    # Returns:
    #     list(
    #         beta: vector (n_programs) - treatment effect for each program
    #         pseudocounts: vector (n_programs) - pseudocounts per program (NULL for non-ols_pseudo)
    #     )
    n_programs <- ncol(P_cells)
    beta <- numeric(n_programs)
    pseudocounts <- NULL

    # For ols_pseudo, compute pseudocount per program
    if (method == "ols_pseudo") {
        pseudocounts <- numeric(n_programs)
    }

    # Build design matrix
    X_df <- covariates_df
    X_df$treatment <- as.numeric(treatment)

    for (k in 1:n_programs) {
        # Transform program k based on method
        if (method == "ols_log" || method == "hurdle_marginal") {
            y_k <- log1p(P_cells[, k] * 1e6)
        } else if (method == "ols_pseudo") {
            # Compute pseudocount for this program
            pseudo <- compute_pseudocount_ols_pseudo(P_cells[, k])
            pseudocounts[k] <- pseudo
            y_k <- log(P_cells[, k] * 1e6 + pseudo)
        } else {
            stop(sprintf("Unknown method: %s", method))
        }

        # Fit model and extract treatment effect
        if (method == "hurdle_marginal") {
            # Fit two-part model and extract marginal effect
            model_data <- X_df
            model_data$y <- y_k

            formula_str <- as.formula(paste("y ~", paste(colnames(X_df), collapse = " + ")))

            if (!is.null(weights)) {
                fit <- tpm(
                    formula_part1 = formula_str,
                    formula_part2 = formula_str,
                    data = model_data,
                    link_part1 = "logit",
                    family_part2 = gaussian(link = "identity"),
                    weights = weights
                )
            } else {
                fit <- tpm(
                    formula_part1 = formula_str,
                    formula_part2 = formula_str,
                    data = model_data,
                    link_part1 = "logit",
                    family_part2 = gaussian(link = "identity")
                )
            }

            # Compute marginal effect for treatment
            ame_result <- AME(fit, term = "treatment")
            treatment_row <- which(ame_result$Variable == "treatment")

            if (length(treatment_row) > 0) {
                beta[k] <- ame_result[treatment_row, "dydx"]
            } else {
                stop(sprintf("Treatment effect not found in AME result for program %d", k))
            }
        } else {
            # Fit linear model (OLS)
            if (!is.null(weights)) {
                fit <- lm(y_k ~ ., data = X_df, weights = weights)
            } else {
                fit <- lm(y_k ~ ., data = X_df)
            }

            # Extract treatment coefficient
            coef_names <- names(coef(fit))
            treatment_idx <- which(coef_names == "treatment")

            if (length(treatment_idx) > 0) {
                beta[k] <- coef(fit)[treatment_idx]
            } else {
                stop(sprintf("Treatment coefficient not found for program %d", k))
            }
        }
    }

    return(list(
        beta = beta,
        pseudocounts = pseudocounts
    ))
}


empirical_bias_correction <- function(P_cells, treatment, covariates_df, W,
                                       method = "ols_log",
                                       n_perm = 100,
                                       weights = NULL, seed = 42,
                                       ncpus = 1, verbose = TRUE) {
    # Compute empirical bias via permutation testing.
    #
    # For i=1..n_perm:
    #     1. Permute treatment labels
    #     2. Refit program models → β_perm
    #     3. Compute gene counterfactuals
    #
    # Bias = mean(permuted gene effects)
    #
    # Parameters:
    #     P_cells: matrix (n_cells × n_programs)
    #     treatment: vector (n_cells) - binary treatment indicator
    #     covariates_df: data.frame with covariates
    #     W: matrix (n_programs × n_genes)
    #     method: regression method name (default "ols_log")
    #     n_perm: number of permutation iterations (default 100)
    #     weights: optional matching weights
    #     seed: random seed for permutation
    #     ncpus: number of cores for parallel processing
    #     verbose: print progress
    #
    # Returns:
    #     bias: vector (n_genes) - empirical bias for each gene
    n_genes <- ncol(W)

    if (verbose) {
        cat(sprintf("  Computing empirical bias (%d permutations, %d cores)...\n", n_perm, ncpus))
    }

    set.seed(seed)

    # Define permutation function for parallel execution
    run_permutation <- function(i) {
        # Permute treatment
        treatment_perm <- sample(treatment)

        # Refit programs
        fit_result <- fit_programs_extract_beta(P_cells, treatment_perm, covariates_df,
                                                 method, weights)

        # Compute gene effects
        gene_effects_perm <- compute_gene_counterfactuals(P_cells, W, fit_result$beta,
                                                           method, fit_result$pseudocounts)

        return(gene_effects_perm)
    }

    # Run permutations (parallel if ncpus > 1)
    if (ncpus > 1) {
        perm_results <- parallel::mclapply(1:n_perm, run_permutation, mc.cores = ncpus)
    } else {
        perm_results <- lapply(1:n_perm, run_permutation)
    }

    # Convert list to matrix
    perm_effects <- do.call(rbind, perm_results)  # (n_perm × n_genes)

    # Empirical bias = mean of permuted effects
    bias <- colMeans(perm_effects)

    if (verbose) {
        cat(sprintf("  Bias range: [%.6f, %.6f]\n", min(bias), max(bias)))
    }

    return(bias)
}


bootstrap_program_and_gene_effects <- function(P_cells, treatment, covariates_df, W,
                                                 method = "ols_log",
                                                 n_boot = 500,
                                                 weights = NULL, seed = 42,
                                                 ncpus = 1, verbose = TRUE) {
    # Bootstrap both program and gene effects using shared resamples.
    #
    # For b=1..n_boot:
    #     1. Resample cells (with replacement)
    #     2. Refit program models → β_boot
    #     3. Store β_boot (for program-level SE)
    #     4. Compute gene counterfactuals (for gene-level SE)
    #
    # Parameters:
    #     P_cells: matrix (n_cells × n_programs)
    #     treatment: vector (n_cells)
    #     covariates_df: data.frame with covariates
    #     W: matrix (n_programs × n_genes)
    #     method: regression method name (default "ols_log")
    #     n_boot: number of bootstrap iterations (default 500)
    #     weights: optional matching weights
    #     seed: random seed for bootstrap
    #     ncpus: number of cores for parallel processing
    #     verbose: print progress
    #
    # Returns:
    #     list(
    #         program_effects: matrix (n_boot × n_programs)
    #         gene_effects: matrix (n_boot × n_genes)
    #     )
    n_cells <- nrow(P_cells)
    n_programs <- ncol(P_cells)
    n_genes <- ncol(W)

    # Function for single bootstrap iteration
    run_bootstrap <- function(b) {
        # Resample cells
        boot_idx <- sample(1:n_cells, replace = TRUE)

        P_boot <- P_cells[boot_idx, , drop = FALSE]
        treatment_boot <- treatment[boot_idx]
        covariates_boot <- covariates_df[boot_idx, , drop = FALSE]
        weights_boot <- if (!is.null(weights)) weights[boot_idx] else NULL

        # Refit programs
        fit_result <- fit_programs_extract_beta(P_boot, treatment_boot, covariates_boot,
                                                 method, weights_boot)

        # Compute gene effects
        gene_effects_boot <- compute_gene_counterfactuals(P_boot, W, fit_result$beta,
                                                           method, fit_result$pseudocounts)

        return(list(
            program_effects = fit_result$beta,
            gene_effects = gene_effects_boot
        ))
    }

    if (verbose) {
        cat(sprintf("  Bootstrap (%d iterations, %d cores)...\n", n_boot, ncpus))
    }

    set.seed(seed)

    # Run bootstrap (parallel if ncpus > 1)
    if (ncpus > 1) {
        boot_results <- parallel::mclapply(1:n_boot, run_bootstrap, mc.cores = ncpus)
    } else {
        boot_results <- lapply(1:n_boot, run_bootstrap)
    }

    # Extract results
    program_effects_list <- lapply(boot_results, function(x) x$program_effects)
    gene_effects_list <- lapply(boot_results, function(x) x$gene_effects)

    program_effects <- do.call(rbind, program_effects_list)  # (n_boot × n_programs)
    gene_effects <- do.call(rbind, gene_effects_list)  # (n_boot × n_genes)

    if (verbose) {
        cat(sprintf("  Completed %d bootstrap iterations\n", n_boot))
    }

    return(list(
        program_effects = program_effects,
        gene_effects = gene_effects
    ))
}


save_gene_results_parquet <- function(perturbations, genes, p_values_matrix,
                                       coef_matrix, se_matrix, method_name,
                                       output_base) {
    # Save gene-level results in Parquet format (compressed, R and Python compatible).
    #
    # Parameters:
    #     perturbations: vector of perturbation names
    #     genes: vector of gene names
    #     p_values_matrix: matrix (n_perturbations × n_genes)
    #     coef_matrix: matrix (n_perturbations × n_genes)
    #     se_matrix: matrix (n_perturbations × n_genes)
    #     method_name: string (e.g., "ols_log")
    #     output_base: base path for output (will add _method.parquet)
    # Convert matrices to long format data frame
    n_perts <- length(perturbations)
    n_genes <- length(genes)

    # Create long format: one row per perturbation-gene pair
    results_df <- data.frame(
        perturbation = rep(perturbations, each = n_genes),
        gene = rep(genes, times = n_perts),
        p_value = as.vector(t(p_values_matrix)),
        coef_gene = as.vector(t(coef_matrix)),
        se_gene = as.vector(t(se_matrix)),
        stringsAsFactors = FALSE
    )

    # Save as compressed Parquet
    output_file <- sprintf("%s_%s.parquet", output_base, method_name)
    nanoparquet::write_parquet(results_df, output_file, compression = "snappy")

    cat(sprintf("  Saved: %s (%d rows)\n", output_file, nrow(results_df)))
}


# =============================================================================
# DIRECT GENE-LEVEL DE FUNCTIONS
# =============================================================================

load_gene_expression <- function(gene_expr_file, gene_names_file, cell_barcodes) {
    # Load gene expression matrix from Matrix Market format.
    #
    # Parameters:
    #     gene_expr_file: Path to gene_expression.mtx (Matrix Market format)
    #     gene_names_file: Path to response_gene_names.txt
    #     cell_barcodes: Vector of cell barcodes (determines row order for output)
    #
    # Returns:
    #     list(
    #         expr_matrix: sparse Matrix (n_cells × n_genes), reordered to match cell_barcodes
    #         gene_names: vector of gene names
    #     )
    cat("Loading gene expression matrix (Matrix Market format)...\n")

    # Load gene names
    gene_names <- readLines(gene_names_file)
    n_genes <- length(gene_names)
    cat(sprintf("  Response genes: %d\n", n_genes))

    # Load cell barcodes from file (row order of the matrix)
    barcodes_file <- sub("gene_expression\\.mtx$", "gene_expression_barcodes.txt", gene_expr_file)
    matrix_barcodes <- readLines(barcodes_file)
    cat(sprintf("  Matrix barcodes: %d\n", length(matrix_barcodes)))

    # Load Matrix Market file
    gene_expr_sparse <- readMM(gene_expr_file)

    # Convert to dgCMatrix for efficient column access
    gene_expr_sparse <- as(gene_expr_sparse, "dgCMatrix")

    cat(sprintf("  Loaded: %d cells × %d genes\n", nrow(gene_expr_sparse), ncol(gene_expr_sparse)))
    cat(sprintf("  Non-zero entries: %d\n", nnzero(gene_expr_sparse)))

    # Assign row names from barcodes file
    rownames(gene_expr_sparse) <- matrix_barcodes

    # Reorder to match requested cell_barcodes
    if (!all(cell_barcodes %in% matrix_barcodes)) {
        missing <- sum(!cell_barcodes %in% matrix_barcodes)
        stop(sprintf("ERROR: %d requested cells not found in gene expression matrix!", missing))
    }
    gene_expr_sparse <- gene_expr_sparse[cell_barcodes, , drop = FALSE]
    cat(sprintf("  Reordered to match %d requested cells\n", length(cell_barcodes)))

    return(list(
        expr_matrix = gene_expr_sparse,
        gene_names = gene_names
    ))
}


test_gene_expression_single <- function(response_gene_idx, gene_expr_matched, X_matched,
                                         perturbation_col, matching_weights, pair_ids,
                                         method_names, max_control_cells, seed) {
    # Test a single response gene's expression against perturbation.
    #
    # Parameters:
    #     response_gene_idx: Index of response gene to test
    #     gene_expr_matched: Gene expression matrix for matched cells (n_matched × n_genes)
    #     X_matched: Covariate matrix including treatment (n_matched × n_covariates)
    #     perturbation_col: Name of treatment column in X_matched
    #     matching_weights: Matching weights for cells
    #     pair_ids: Matching pair IDs for cluster-robust SE
    #     method_names: Vector of method names (e.g., c("ols_log", "ols_pseudo"))
    #     max_control_cells: Max control cells for subsampling
    #     seed: Random seed
    #
    # Returns:
    #     Named list with coefficients, SE, p-values per method

    # Get expression for this gene
    y_raw <- as.vector(gene_expr_matched[, response_gene_idx])

    result <- list()

    for (method_name in method_names) {
        # Apply transformation
        if (method_name == "ols_log") {
            y_transformed <- log1p(y_raw)
        } else if (method_name == "ols_pseudo") {
            # Compute pseudocount (5th percentile of non-zero)
            nonzero_vals <- y_raw[y_raw > 0]
            if (length(nonzero_vals) > 0) {
                pseudo <- quantile(nonzero_vals, probs = 0.05)
            } else {
                pseudo <- 1e-6
            }
            y_transformed <- log(y_raw + pseudo)
        } else {
            # Default: log1p
            y_transformed <- log1p(y_raw)
        }

        # Fit regression
        fit_result <- fit_linear_model_robust(
            y_transformed, X_matched, perturbation_col,
            max_control_cells, seed,
            matching_weights, pair_ids,
            compute_se = TRUE
        )

        # Store results
        result[[paste0("coef_", method_name)]] <- fit_result$coef
        result[[paste0("se_", method_name)]] <- fit_result$se
        result[[paste0("t_stat_", method_name)]] <- fit_result$t_stat
        result[[paste0("p_value_", method_name)]] <- fit_result$p_value
    }

    return(result)
}


test_gene_expression_batch <- function(gene_expr_matched, gene_names, X_matched,
                                        perturbation_col, matching_weights, pair_ids,
                                        method_names, max_control_cells, seed,
                                        n_jobs = 1) {
    # Test all response genes using limma (vectorized, fast).
    #
    # Uses limma::lmFit + eBayes(trend=TRUE) for efficient testing of all genes.
    # Applies per-gene pseudocount transformation matching our ols_pseudo method.
    #
    # Parameters:
    #     gene_expr_matched: Gene expression matrix for matched cells (n_matched × n_genes)
    #     gene_names: Vector of response gene names
    #     X_matched: Covariate matrix including treatment (n_matched × n_covariates)
    #     perturbation_col: Name of treatment column in X_matched
    #     matching_weights: Matching weights for cells
    #     pair_ids: Matching pair IDs for cluster-robust SE (not used with limma)
    #     method_names: Vector of method names (only ols_pseudo supported with limma)
    #     max_control_cells: Max control cells for subsampling (not used with limma)
    #     seed: Random seed (not used with limma)
    #     n_jobs: Number of parallel jobs (not used - limma is already vectorized)
    #
    # Returns:
    #     Data frame with columns: response_gene, coef_*, se_*, p_value_* per method

    # TODO: Refactor to share preprocessing with fit_linear_model_robust
    # Currently preprocessing (NA check, subsampling, collinearity) is duplicated
    # between lm and limma paths. Should be done once in test_single_gene.

    # Check limma is available
    if (!requireNamespace("limma", quietly = TRUE)) {
        stop("limma package is required for gene-level DE testing. Install with BiocManager::install('limma')")
    }

    # Check for NAs in covariates (match fit_linear_model_robust behavior)
    if (!all(complete.cases(X_matched))) {
        stop(sprintf("Found NA values in covariate data. Incomplete cases: %d/%d",
                     sum(!complete.cases(X_matched)), nrow(X_matched)))
    }

    # Subsample control cells if needed (match fit_linear_model_robust behavior)
    treatment <- X_matched[[perturbation_col]]
    n_control <- sum(treatment == 0)
    keep_indices <- seq_len(nrow(X_matched))

    if (n_control > max_control_cells) {
        set.seed(seed)
        control_indices <- which(treatment == 0)
        sampled_control_indices <- sample(control_indices, size = max_control_cells, replace = FALSE)
        keep_indices <- c(which(treatment == 1), sampled_control_indices)

        X_matched <- X_matched[keep_indices, , drop = FALSE]
        gene_expr_matched <- gene_expr_matched[keep_indices, , drop = FALSE]
        if (!is.null(matching_weights)) matching_weights <- matching_weights[keep_indices]
        if (!is.null(pair_ids)) pair_ids <- pair_ids[keep_indices]

        cat(sprintf("    Subsampled controls: %d -> %d\n", n_control, max_control_cells))
    }

    n_genes <- ncol(gene_expr_matched)
    cat(sprintf("  Testing %d response genes with limma...\n", n_genes))

    # Convert sparse matrix to dense for limma
    if (inherits(gene_expr_matched, "sparseMatrix")) {
        gene_expr_dense <- as.matrix(gene_expr_matched)
    } else {
        gene_expr_dense <- gene_expr_matched
    }

    # Filter genes: require minimum 10 UMIs total AND at least 10 cells with non-zero expression
    min_umi_threshold <- 10
    min_cells_expressing <- 10
    gene_umi_totals <- colSums(gene_expr_dense)
    gene_cells_expressing <- colSums(gene_expr_dense > 0)
    keep_genes <- (gene_umi_totals >= min_umi_threshold) & (gene_cells_expressing >= min_cells_expressing)
    n_filtered <- sum(!keep_genes)
    cat(sprintf("    Filtering genes: keeping %d / %d (removed %d with < %d UMIs or < %d cells expressing)\n",
                sum(keep_genes), length(keep_genes), n_filtered, min_umi_threshold, min_cells_expressing))

    gene_expr_dense <- gene_expr_dense[, keep_genes, drop = FALSE]
    gene_names <- gene_names[keep_genes]

    # Transpose for limma/voom (genes in rows, cells in columns)
    counts <- t(gene_expr_dense)
    rownames(counts) <- gene_names

    # Build design matrix from X_matched (already includes treatment column)
    # X_matched is a data.frame with perturbation_col as the treatment indicator

    # Remove collinear covariates (excluding treatment column)
    covariate_cols <- setdiff(colnames(X_matched), perturbation_col)
    numeric_cols <- covariate_cols[sapply(X_matched[, covariate_cols, drop=FALSE], is.numeric)]
    if (length(numeric_cols) > 1) {
        filtered_cols <- remove_collinear_covariates(X_matched, numeric_cols, threshold = 0.95, verbose = TRUE)
        dropped_cols <- setdiff(numeric_cols, filtered_cols)
        if (length(dropped_cols) > 0) {
            cat(sprintf("    Dropped collinear covariates: %s\n", paste(dropped_cols, collapse=", ")))
            X_matched <- X_matched[, c(perturbation_col, filtered_cols, setdiff(covariate_cols, numeric_cols)), drop=FALSE]
        }
    }

    # Check for non-numeric columns (factors)
    col_types <- sapply(X_matched, function(x) paste(class(x), collapse="/"))
    non_numeric <- names(col_types)[!sapply(X_matched, is.numeric)]

    # Convert to numeric design matrix using model.matrix (handles factors properly)
    if (length(non_numeric) > 0) {
        cat(sprintf("    Using model.matrix to expand %d factor(s)...\n", length(non_numeric)))
        design <- model.matrix(~ . - 1, data = X_matched)
    } else {
        design <- as.matrix(X_matched)
    }

    cat(sprintf("      design class: %s\n", paste(class(design), collapse=", ")))
    cat(sprintf("      design typeof: %s\n", typeof(design)))
    cat(sprintf("      design dim: %d x %d\n", nrow(design), ncol(design)))

    # Use voom to transform counts and compute precision weights
    # voom does log-CPM transformation internally and models mean-variance relationship
    cat("    Running voom transformation...\n")
    v <- limma::voom(counts, design, plot = FALSE)

    # Fit with limma using voom weights
    cat("    Fitting limma models with voom weights...\n")
    if (!is.null(matching_weights) && length(matching_weights) == ncol(counts)) {
        # Combine voom weights with matching weights
        combined_weights <- t(t(v$weights) * matching_weights)
        fit <- limma::lmFit(v, design, weights = combined_weights)
    } else {
        fit <- limma::lmFit(v, design)
    }

    # Save ordinary (non-EB) results before eBayes moderation
    # These are equivalent to running lm()/t-test on each gene
    fit_ordinary <- fit  # Copy before eBayes

    # Apply eBayes (no trend needed - voom already handles mean-variance relationship)
    fit <- limma::eBayes(fit)

    # Extract results for the treatment coefficient
    # DEBUG: Check column names
    cat(sprintf("    DEBUG design colnames: %s\n", paste(head(colnames(design), 10), collapse=", ")))
    cat(sprintf("    DEBUG perturbation_col: %s\n", perturbation_col))
    cat(sprintf("    DEBUG perturbation_col in colnames: %s\n", perturbation_col %in% colnames(design)))

    # Find the treatment column (may be renamed by model.matrix)
    if (!(perturbation_col %in% colnames(design))) {
        # Try to find it - model.matrix might have kept the name or prefixed it
        possible_names <- grep(perturbation_col, colnames(design), value = TRUE)
        if (length(possible_names) > 0) {
            cat(sprintf("    WARNING: perturbation_col '%s' not found, using '%s'\n",
                        perturbation_col, possible_names[1]))
            perturbation_col <- possible_names[1]
        } else {
            stop(sprintf("Cannot find treatment column '%s' in design matrix. Columns: %s",
                         perturbation_col, paste(colnames(design), collapse=", ")))
        }
    }

    # Extract eBayes (moderated) results
    coefs <- fit$coefficients[, perturbation_col]
    ses_eb <- sqrt(fit$s2.post) * fit$stdev.unscaled[, perturbation_col]
    t_stats_eb <- fit$t[, perturbation_col]
    pvals_eb <- fit$p.value[, perturbation_col]

    # Extract ordinary (non-EB) results - equivalent to lm()/t-test
    ses_ordinary <- fit_ordinary$sigma * fit_ordinary$stdev.unscaled[, perturbation_col]
    t_stats_ordinary <- coefs / ses_ordinary
    df_residual <- fit_ordinary$df.residual
    pvals_ordinary <- 2 * pt(abs(t_stats_ordinary), df = df_residual, lower.tail = FALSE)

    # Fix numerical issues: detect essentially constant data (like R's t.test)
    # SE must be meaningful relative to coefficient magnitude
    essentially_constant <- ses_ordinary < 10 * .Machine$double.eps * pmax(abs(coefs), 1)
    n_constant <- sum(essentially_constant, na.rm = TRUE)
    if (n_constant > 0) {
        cat(sprintf("    Fixing %d genes with essentially constant data (SE too small)\n", n_constant))
        ses_ordinary[essentially_constant] <- NA
        t_stats_ordinary[essentially_constant] <- NA
        pvals_ordinary[essentially_constant] <- NA
    }

    # DEBUG: Check coefficient distribution
    cat(sprintf("    DEBUG coef range: [%.4f, %.4f], median=%.4f\n", min(coefs, na.rm=TRUE), max(coefs, na.rm=TRUE), median(coefs, na.rm=TRUE)))
    cat(sprintf("    DEBUG pval_eb range: [%.4e, %.4e], median=%.4f\n", min(pvals_eb, na.rm=TRUE), max(pvals_eb, na.rm=TRUE), median(pvals_eb, na.rm=TRUE)))
    cat(sprintf("    DEBUG pval_ordinary range: [%.4e, %.4e], median=%.4f\n", min(pvals_ordinary, na.rm=TRUE), max(pvals_ordinary, na.rm=TRUE), median(pvals_ordinary, na.rm=TRUE)))
    cat(sprintf("    DEBUG NA coefs: %d, NA pvals_eb: %d, NA pvals_ordinary: %d\n", sum(is.na(coefs)), sum(is.na(pvals_eb)), sum(is.na(pvals_ordinary))))

    cat(sprintf("    Completed: %d genes tested\n", n_genes))

    # Build results data frame with both EB and ordinary results
    results_df <- data.frame(
        response_gene = gene_names,
        coef_limma = coefs,
        se_limma = ses_eb,
        t_stat_limma = t_stats_eb,
        p_value_limma = pvals_eb,
        se_ordinary = ses_ordinary,
        t_stat_ordinary = t_stats_ordinary,
        p_value_ordinary = pvals_ordinary,
        stringsAsFactors = FALSE
    )

    return(results_df)
}


save_gene_de_results_parquet <- function(perturbation, results_df, method_names, output_file) {
    # Save direct gene-level DE results to parquet.
    #
    # Parameters:
    #     perturbation: Name of the perturbation tested
    #     results_df: Data frame from test_gene_expression_batch()
    #     method_names: Vector of method names
    #     output_file: Output parquet path

    # Add perturbation column
    results_df$perturbation <- perturbation

    # Reorder columns: perturbation, response_gene, then results
    col_order <- c("perturbation", "response_gene")
    for (method in method_names) {
        col_order <- c(col_order,
                       paste0("coef_", method),
                       paste0("se_", method),
                       paste0("p_value_", method))
    }

    # Keep only columns that exist
    col_order <- col_order[col_order %in% colnames(results_df)]
    results_df <- results_df[, col_order]

    # Save as parquet
    nanoparquet::write_parquet(results_df, output_file, compression = "snappy")

    cat(sprintf("  Saved gene DE results: %s (%d rows)\n", output_file, nrow(results_df)))
}
