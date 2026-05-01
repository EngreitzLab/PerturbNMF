import numpy as np
import scipy
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
    if not isinstance(H, (pd.DataFrame)):
        B_k = X @ H[i,:].T / np.sqrt((np.asarray(H[i,:])**2).sum())
        numerator = compute_Var(X - np.outer(B_k, H[i,:]))
    else:
        B_k = X @ H.iloc[i,:] / np.sqrt((H.iloc[i,:]**2).sum())
        numerator = compute_Var(X - np.outer(B_k, H.iloc[i,:]))
    return (1 - numerator / Var_X)


def compute_explained_variance(cnmf_obj, X, k, output_folder, program_name = range(1, 51), thre = '2.0'):
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
        
    Returns
    -------
    None
        Saves variance metrics to TSV files in output_folder
    """

    # Convert sparse matrix to dense if needed 
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Load consensus gene program matrices (W: usage, H: spectra)
    thre_name = str(thre).replace('.', '_')
    H_path = cnmf_obj.paths['consensus_spectra__txt'] % (k, thre_name) ## median_spectra_file
    H_df = pd.read_csv(H_path, sep='\t', index_col=0).T
    H = H_df.to_numpy()
    H = (H/H.sum(0))

    W_path = cnmf_obj.paths['consensus_usages__txt'] % (k, thre_name) ## median_spectra_file
    W_df = pd.read_csv(W_path, sep='\t', index_col=0)
    W = W_df.to_numpy()

    WH = W @ H.T
    diff = X - WH
    diff_sumOfSquaresError = (np.asarray(diff)**2).sum()
    Var_diff = compute_Var(diff)
    Var_X = compute_Var(X)
    TotalVarianceExplained = 1 - Var_diff / Var_X
    V_k = np.empty([k])


    # Calculate variance explained by each individual program 
    for i in range(k):
        V_k[i] = computeVarianceExplained(X, H.T, Var_X, i)

    ProgramID = ['K' + str(k) + '_' + str(i+1) for i in range(k)]

    # Save variance metrics and summary statistics to files
    metrics_df = pd.DataFrame({'VarianceExplained': V_k,
                                'ProgramID': ProgramID,
                                'program_name': list(program_name) })

    metrics_summary = pd.DataFrame({'Sum' : metrics_df['VarianceExplained'].sum(),
                                    'Median' : metrics_df['VarianceExplained'].median(),
                                    'Max' : metrics_df['VarianceExplained'].max(),
                                    'Total' : TotalVarianceExplained},
                                    index = [0])


    metrics_df.to_csv(os.path.join(output_folder, f"{k}_Explained_Variance.txt"), index = None, sep="\t")
    metrics_summary.to_csv(os.path.join(output_folder, f"{k}_Explained_Variance_Summary.txt"), index = None, sep="\t")
 