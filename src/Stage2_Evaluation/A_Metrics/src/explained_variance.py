import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import pandas as pd
import scanpy as sc
import argparse
import os
import re
from torch_cnmf import cNMF
from sklearn.decomposition import PCA


def compute_Var(X):
    """
    Calculate total variance across all features in a matrix.

    Parameters
    ----------
    X : array-like
        Input matrix where variance is computed across rows for each column

    Returns
    -------
    float
        Sum of variances across all columns
    """
    return np.sum(np.var(X, axis=0, ddof=1))


def computeVarianceExplained(X, H, Var_X, i):
    """
    Calculate variance explained by a specific gene program component.

    Parameters
    ----------
    X : array-like
        Original data matrix
    H : array-like or pd.DataFrame
        Gene program matrix (components x genes)
    Var_X : float
        Total variance of original data
    i : int
        Index of the specific component to analyze

    Returns
    -------
    float
        Fraction of variance explained by component i
    """
    # Project X onto the program's unit direction h/||h||: prediction = (X h/||h||)(h/||h||)^T
    if not isinstance(H, (pd.DataFrame)):
        nrm = np.sqrt((np.asarray(H[i, :]) ** 2).sum())
        B_k = X @ H[i, :].T / nrm
        numerator = compute_Var(X - np.outer(B_k, H[i, :] / nrm))
    else:
        nrm = np.sqrt((H.iloc[i, :] ** 2).sum())
        B_k = X @ H.iloc[i, :] / nrm
        numerator = compute_Var(X - np.outer(B_k, H.iloc[i, :] / nrm))
    return 1 - numerator / Var_X


def refit_usage_nnls(X, H):
    """
    Fit non-negative usages W minimizing ||X - W H^T||_F with spectra H held fixed,
    solving one exact NNLS problem (scipy.optimize.nnls) per cell.

    Each cell's problem min_w ||x - H w||^2 is reduced to a K x K one using the Cholesky
    factor of the Gram matrix (H^T H = L L^T), which has the same solution:
    ||x - H w||^2 = ||L^T w - L^{-1} H^T x||^2 + const.

    Parameters
    ----------
    X : array-like or sparse matrix
        Normalized gene expression matrix (cells x genes)
    H : np.ndarray
        Gene program matrix (genes x programs)

    Returns
    -------
    np.ndarray
        Refit usage matrix (cells x programs)
    """
    try:
        L = np.linalg.cholesky(H.T @ H)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Spectra Gram matrix is singular (duplicated or all-zero programs); "
            "cannot refit usages by NNLS"
        )
    A = L.T
    B = np.asarray(X @ H)  # cells x programs, H^T x per cell
    B = scipy.linalg.solve_triangular(L, B.T, lower=True).T  # rows are L^{-1} H^T x
    W = np.empty_like(B)
    for c in range(B.shape[0]):
        W[c], _ = scipy.optimize.nnls(A, B[c])
    return W


def compute_explained_variance(
    cnmf_obj,
    X,
    k,
    output_folder,
    program_name=range(1, 51),
    thre="2.0",
    usage_source="nnls",
):
    """
    Compute explained variance for all gene programs in a cNMF factorization.

    Parameters
    ----------
    cnmf_obj : cnmf.cNMF
        Fitted cNMF object containing file paths
    X : array-like
        Normalized gene expression matrix (cells x genes)
    k : int
        Number of components/gene programs
    output_folder : str
        Directory to save variance results
    thre : str, default '2.0'
        Threshold parameter for consensus matrices
    program_name: list
        List of program names
    usage_source : {'nnls', 'provided'}, default 'nnls'
        'nnls' refits usages against X with the consensus spectra fixed and saves them as
        {k}_consensus_usages_norm.txt in output_folder; 'provided' reads the
        *.consensus_median.txt usages saved by cNMF consensus

    Returns
    -------
    None
        Saves variance metrics to TSV files in output_folder
    """

    # Load consensus gene program matrix (H: spectra, genes x programs)
    thre_name = str(thre).replace(".", "_")
    H_path = cnmf_obj.paths["consensus_spectra__txt"] % (
        k,
        thre_name,
    )  ## median_spectra_file
    H_df = pd.read_csv(H_path, sep="\t", index_col=0).T
    H = H_df.to_numpy()
    H = H / H.sum(0)

    if X.shape[1] != H.shape[0]:
        raise ValueError(
            f"X has {X.shape[1]} genes but consensus spectra have {H.shape[0]}; "
            "X must be the run's cnmf_tmp norm_counts matrix"
        )

    X = X.astype(np.float64)
    W_path = cnmf_obj.paths["consensus_usages__txt"] % (k, thre_name)

    if usage_source == "nnls":
        # The saved consensus usages are refit against std-scaled HVG TPM (refit_usage=True in
        # cNMF consensus), so they are not on X's scale. Refit W against X with H fixed (NNLS),
        # the same step cNMF uses to fit usages to norm_counts.
        W = refit_usage_nnls(X, H)

        # Save the norm_counts-refit usages to output_folder (the inference consensus usages are
        # left untouched), with cell IDs taken from the saved consensus usages (both are in
        # norm_counts.obs order)
        cell_index = pd.read_csv(W_path, sep="\t", index_col=0, usecols=[0]).index
        if len(cell_index) != W.shape[0]:
            raise ValueError(
                f"X has {W.shape[0]} cells but {W_path} has {len(cell_index)}"
            )
        pd.DataFrame(W, index=cell_index, columns=H_df.columns).to_csv(
            os.path.join(output_folder, f"{k}_consensus_usages_nnls.txt"), sep="\t"
        )

    elif usage_source == "provided":
        # Use the usages cNMF consensus fit to norm_counts with the median spectra
        # (*.consensus_median.txt, written next to *.consensus.txt)
        W_path = W_path.replace(".consensus.txt", ".consensus_median.txt")
        W_df = pd.read_csv(W_path, sep="\t", index_col=0)
        W = W_df.to_numpy()
        if W.shape != (X.shape[0], H.shape[1]):
            raise ValueError(
                f"Usages in {W_path} have shape {W.shape}, expected {(X.shape[0], H.shape[1])}"
            )

    else:
        raise ValueError(
            f"usage_source must be 'nnls' or 'provided', got {usage_source!r}"
        )

    # Convert sparse matrix to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    WH = W @ H.T
    diff = X - WH
    diff_sumOfSquaresError = (np.asarray(diff) ** 2).sum()
    Var_diff = compute_Var(diff)
    Var_X = compute_Var(X)
    TotalVarianceExplained = 1 - Var_diff / Var_X
    V_k = np.empty([k])

    # Calculate variance explained by each individual program
    for i in range(k):
        V_k[i] = computeVarianceExplained(X, H.T, Var_X, i)

    ProgramID = ["K" + str(k) + "_" + str(i + 1) for i in range(k)]

    # Save variance metrics and summary statistics to files
    metrics_df = pd.DataFrame(
        {
            "VarianceExplained": V_k,
            "ProgramID": ProgramID,
            "program_name": list(program_name),
        }
    )

    metrics_summary = pd.DataFrame(
        {
            "Sum": metrics_df["VarianceExplained"].sum(),
            "Median": metrics_df["VarianceExplained"].median(),
            "Max": metrics_df["VarianceExplained"].max(),
            "Total": TotalVarianceExplained,
        },
        index=[0],
    )

    metrics_df.to_csv(
        os.path.join(output_folder, f"{k}_Explained_Variance.txt"), index=None, sep="\t"
    )
    metrics_summary.to_csv(
        os.path.join(output_folder, f"{k}_Explained_Variance_Summary.txt"),
        index=None,
        sep="\t",
    )
