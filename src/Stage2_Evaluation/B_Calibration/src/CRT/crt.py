import numba as nb
import numpy as np


def compute_null_pvals_from_null_stats(
    null_stats: np.ndarray,
    side_code: int = 0,
) -> np.ndarray:
    """
    Compute leave-one-out CRT-null p-values for each null statistic.
    null_stats: array (B,) or (B, K) of null test statistics.
    side_code: 0 two-sided, 1 right-tailed, -1 left-tailed.
    Returns:
        pvals_null with the same shape as null_stats.
    """
    stats = np.asarray(null_stats, dtype=np.float64)
    if stats.ndim == 1:
        squeeze = True
    elif stats.ndim == 2:
        squeeze = False
    else:
        raise ValueError("null_stats must be 1D or 2D.")

    if side_code not in (-1, 0, 1):
        raise ValueError("side_code must be -1, 0, or 1.")

    if stats.ndim == 1:
        stats_2d = stats[:, None]
    else:
        stats_2d = stats

    B, K = stats_2d.shape
    pvals = np.empty_like(stats_2d, dtype=np.float64)

    for k in range(K):
        vals = stats_2d[:, k]
        if side_code == 0:
            vals = np.abs(vals)
        elif side_code == -1:
            vals = -vals

        sorted_vals = np.sort(vals)
        count_lt = np.searchsorted(sorted_vals, vals, side="left")
        pvals[:, k] = (B - count_lt) / float(B)

    if squeeze:
        return np.clip(pvals[:, 0], 1.0 / float(B), 1.0)
    return np.clip(pvals, 1.0 / float(B), 1.0)


@nb.njit(inline="always")
def _sample_unique_ints(B: int, m: int, out: np.ndarray, start: int) -> None:
    """
    Fill `out[start:start+m]` with unique integers in [0, B)
    We randomly draw m unique resampleIDs from B
    out: preallocated array to store the sampled integers
    start: starting index in out to fill
    """
    if m > 0:
        out[start : start + m] = np.random.choice(B, m, replace=False)


@nb.njit
def crt_index_sampler_fast_numba(p: np.ndarray, B: int, seed: int):
    """
    For each cell j: M_j ~ Binomial(B, p_j), then choose M_j resample IDs without replacement.
    Returns resample-wise index lists in CSC-like (indptr, indices) format.
    cell_resample_idprt[j] = starting index in the array that stores resample IDs of cell j.
    cell_resample_ids array contains the resample IDs for all cells
    cell_resample_ids[ cell_resample_idprt[j] : cell_resample_idprt[j+1] ] : resample IDs for cell j
    counts: number of cells in each resample ID
    p: array of length N with probabilities for each cell
    B: number of resamples
    seed: random seed for reproducibility
    Returns:
        indptr: array of length B+1, indptr[b] = starting index in indices for resample b
        indices: array of length total number of selected cells across all resamples,
                 contains cell indices for each resample
    """
    np.random.seed(seed)
    N = p.shape[0]
    M = np.empty(N, dtype=np.int32)

    cell_resample_idprt = np.empty(N + 1, dtype=np.int64)
    cell_resample_idprt[0] = 0
    total = 0
    for j in range(N):
        mj = np.random.binomial(B, p[j])
        M[j] = mj
        total += mj
        cell_resample_idprt[j + 1] = total

    cell_resample_ids = np.empty(total, dtype=np.int32)
    counts = np.zeros(B, dtype=np.int32)

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = cell_resample_idprt[j]
        _sample_unique_ints(B, mj, cell_resample_ids, start)
        for t in range(mj):
            b = cell_resample_ids[start + t]
            counts[b] += 1

    indptr = np.empty(B + 1, dtype=np.int64)
    indptr[0] = 0
    ncells = 0
    for b in range(B):
        ncells += counts[b]
        indptr[b + 1] = ncells

    indices = np.empty(total, dtype=np.int32)
    write = np.empty(B, dtype=np.int64)
    for b in range(B):
        write[b] = indptr[b]

    """
    write[b]: next write position in indices array for resample b
    indices array will store, for each resample b, the list of cell indices j that are included in resample b
    every time we find that cell j is included in resample b, we write j into indices[ write[b] ], and increment write[b] by 1
    indicating the next available position in indices for resample b
    """

    for j in range(N):
        mj = M[j]
        if mj == 0:
            continue
        start = cell_resample_idprt[j]
        for t in range(mj):
            b = cell_resample_ids[start + t]
            pos = write[b]
            indices[pos] = j
            write[b] = pos + 1

    return indptr, indices


@nb.njit
def _beta_from_summaries(
    n1: int,
    v: np.ndarray,
    sY: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
) -> np.ndarray:
    """
    Closed-form OLS coefficient for x in Y ~ x + C using summary stats.
    n1: number of cells with x=1
    v: vector of length p, sum of covariates C over cells with x=1
    sY: vector of length K, sum of outcomes Y over cells with x=1
    A: p x p matrix, C^T C
    CTY: p x K matrix, C^T Y
    Returns:
        beta: vector of length K, OLS coefficients for x in Y ~ x + C using summary stats.

    The key aspect of this function is that it doesn't perform regression directly on (Y, x, and C).
    Instead, it computes the result using pre-calculated summary statistics,
    which is a much more memory and computationally efficient approach for large datasets.

    n1: the sum of square of the target variable x (which is binary, so this is just the count of samples where x=1)
    v: the cross-product of the covariates C and the target variable x (C^T x)
    sY: the cross-product of the outcomes Y and the target variable x (Y^T x)
    A: the inverse of the cross-product of the covariates (C^T C)^{-1}
    CTY: the cross-product of the covariates C and the outcomes Y (C^T Y)
    """
    p = v.shape[0]
    K = sY.shape[0]

    tmp = np.zeros(p, dtype=np.float64)
    for i in range(p):
        acc = 0.0
        for j in range(p):
            acc += v[j] * A[j, i]
        tmp[i] = acc

    den = float(n1)
    for i in range(p):
        den -= tmp[i] * v[i]
    if den <= 1e-12:
        return np.zeros(K, dtype=np.float64)

    beta = np.empty(K, dtype=np.float64)
    for k in range(K):
        acc = 0.0
        for i in range(p):
            acc += tmp[i] * CTY[i, k]
        beta[k] = (sY[k] - acc) / den
    return beta


@nb.njit
def crt_pvals_for_gene(
    indptr: np.ndarray,
    indices: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
):
    """
    Compute CRT p-values and observed betas using summary-based OLS updates.
    indptr: array of length B+1, indptr[b] = starting index in indices for resample b
    indices: array of length total number of selected cells across all resamples,
             contains cell indices for each resample
    C: covariate matrix (N x p)
    Y: outcome matrix (N x K)
    A: p x p matrix, (C^T C)^{-1}
    CTY: p x K matrix, C^T Y
    obs_idx: array of cell indices with treatment
    B: number of resamples
    Returns:
        pvals: array of length K, CRT p-values for each program
        beta_obs: array of length K, observed effect sizes for each program
    """
    N, p = C.shape
    K = Y.shape[1]

    n1_obs = obs_idx.shape[0]
    v_obs = np.zeros(p, dtype=np.float64)
    sY_obs = np.zeros(K, dtype=np.float64)
    for t in range(n1_obs):
        i = obs_idx[t]
        for j in range(p):
            v_obs[j] += C[i, j]
        for k in range(K):
            sY_obs[k] += Y[i, k]

    """ 
        Compute observed betas using summary-based OLS updates. 
        K: number of programs
        ge[k]: count of resamples where |beta_resample[k]| >= |beta_obs[k]|
    """
    beta_obs = _beta_from_summaries(n1_obs, v_obs, sY_obs, A, CTY)
    abs_obs = np.abs(beta_obs)
    ge = np.zeros(K, dtype=np.int32)

    v = np.empty(p, dtype=np.float64)
    sY = np.empty(K, dtype=np.float64)

    for b in range(B):
        for j in range(p):
            v[j] = 0.0
        for k in range(K):
            sY[k] = 0.0

        start = indptr[b]
        end = indptr[b + 1]
        n1 = end - start

        for pos in range(start, end):
            i = indices[pos]
            for j in range(p):
                v[j] += C[i, j]
            for k in range(K):
                sY[k] += Y[i, k]

        beta = _beta_from_summaries(n1, v, sY, A, CTY)
        for k in range(K):
            if abs(beta[k]) >= abs_obs[k]:
                ge[k] += 1

    pvals = np.empty(K, dtype=np.float64)
    for k in range(K):
        pvals[k] = (1.0 + ge[k]) / (B + 1.0)

    return pvals, beta_obs


@nb.njit
def crt_betas_for_gene(
    indptr: np.ndarray,
    indices: np.ndarray,
    C: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    CTY: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
):
    """
    Return observed betas and the full null beta matrix (B x K).
    """
    N, p = C.shape
    K = Y.shape[1]

    n1_obs = obs_idx.shape[0]
    v_obs = np.zeros(p, dtype=np.float64)
    sY_obs = np.zeros(K, dtype=np.float64)
    for t in range(n1_obs):
        i = obs_idx[t]
        for j in range(p):
            v_obs[j] += C[i, j]
        for k in range(K):
            sY_obs[k] += Y[i, k]

    beta_obs = _beta_from_summaries(n1_obs, v_obs, sY_obs, A, CTY)
    beta_null = np.empty((B, K), dtype=np.float64)

    v = np.empty(p, dtype=np.float64)
    sY = np.empty(K, dtype=np.float64)

    for b in range(B):
        for j in range(p):
            v[j] = 0.0
        for k in range(K):
            sY[k] = 0.0

        start = indptr[b]
        end = indptr[b + 1]
        n1 = end - start

        for pos in range(start, end):
            i = indices[pos]
            for j in range(p):
                v[j] += C[i, j]
            for k in range(K):
                sY[k] += Y[i, k]

        beta_null[b, :] = _beta_from_summaries(n1, v, sY, A, CTY)

    return beta_obs, beta_null
