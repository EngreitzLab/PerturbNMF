"""
OLS effect of a treated indicator on every program, from the precomputed
CtC_inv = (C^T C)^-1 and CtU_norm = C^T Usage_norm (Frisch-Waugh-Lovell).
"""

import numba as nb
import numpy as np


@nb.njit(cache=True)
def OLS_solver(Covariates, Usage_norm, CtC_inv, CtU_norm, treated_obs_idx):
    """
    OLS effect of the treated indicator on every program, adjusting for Covariates.

    Fits Usage_norm ~ Covariates + x, where x is 1 on treated_obs_idx and 0
    elsewhere, and returns the coefficient on x for each program. By
    Frisch-Waugh-Lovell this only needs the precomputed pieces:

        beta = (x^T Y - x^T C CtC_inv CtU_norm) / (x^T x - x^T C CtC_inv C^T x)

    x^T Y and x^T C are column sums over the treated rows and x^T x is their
    count, so the cost is O(n_treated * (p + K)).

    Args:
        Covariates: N x p design matrix (float64, C-contiguous).
        Usage_norm: N x K outcome (float64, C-contiguous).
        CtC_inv: p x p, (Covariates^T Covariates)^-1.
        CtU_norm: p x K, Covariates^T Usage_norm.
        treated_obs_idx: int32 row indexes of the treated cells.

    Returns:
        beta: length-K array; NaN if x is explained by the covariates
            (including an empty treated set, or treated cells = one stratum).
    """
    n_treated = treated_obs_idx.shape[0]
    p = Covariates.shape[1]
    K = Usage_norm.shape[1]

    xC = np.zeros(p)
    xY = np.zeros(K)
    for t in range(n_treated):
        i = treated_obs_idx[t]
        for j in range(p):
            xC[j] += Covariates[i, j]
        for k in range(K):
            xY[k] += Usage_norm[i, k]

    xC_CtC_inv = np.zeros(p)
    for a in range(p):
        acc = 0.0
        for j in range(p):
            acc += xC[j] * CtC_inv[j, a]
        xC_CtC_inv[a] = acc

    denominator = float(n_treated)
    for a in range(p):
        denominator -= xC_CtC_inv[a] * xC[a]

    beta = np.empty(K)
    if denominator <= 1e-10 * max(n_treated, 1):
        beta[:] = np.nan
        return beta
    for k in range(K):
        acc = 0.0
        for a in range(p):
            acc += xC_CtC_inv[a] * CtU_norm[a, k]
        beta[k] = (xY[k] - acc) / denominator
    return beta
