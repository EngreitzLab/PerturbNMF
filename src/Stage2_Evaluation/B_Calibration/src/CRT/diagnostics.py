"""
Diagnostics helpers for CRT null calibration.
"""

from typing import Callable, TYPE_CHECKING

import numpy as np

from .crt import crt_betas_for_gene
from .pipeline_helpers import _fit_propensity, _gene_obs_idx, _gene_seed, _sample_crt_indices
from .propensity import fit_propensity_logistic
if TYPE_CHECKING:
    from .pipeline import CRTInputs


def crt_null_pvals_from_null_stats_fast(
    T_null: np.ndarray, two_sided: bool = True
) -> np.ndarray:
    """
    Compute leave-one-out CRT-null p-values from a 1D null statistics array.
    Definition (two-sided): p_b = (1 + #{b'!=b: |T_{b'}| >= |T_b|}) / B
    """
    vals = np.asarray(T_null, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError("T_null must be a 1D array.")
    B = vals.size
    if B == 0:
        raise ValueError("T_null must contain at least one element.")

    if two_sided:
        vals = np.abs(vals)

    sorted_vals = np.sort(vals)
    count_lt = np.searchsorted(sorted_vals, vals, side="left")
    pvals = (B - count_lt) / float(B)
    min_p = 1.0 / float(B)
    return np.clip(pvals, min_p, 1.0)


def crt_null_pvals_from_null_stats_matrix(
    beta_null_bk: np.ndarray, two_sided: bool = True
) -> np.ndarray:
    """
    Compute null p-values for each column of a (B, K) null-statistics matrix.
    """
    arr = np.asarray(beta_null_bk, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("beta_null_bk must be a 2D array.")
    B, K = arr.shape
    if B == 0:
        raise ValueError("beta_null_bk must contain at least one row.")

    out = np.empty_like(arr, dtype=np.float64)
    for k in range(K):
        out[:, k] = crt_null_pvals_from_null_stats_fast(
            arr[:, k], two_sided=two_sided
        )
    return out


def crt_null_stats_for_test(
    gene: str,
    program_index: int,
    inputs: "CRTInputs",
    B: int = 1023,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
) -> np.ndarray:
    """
    Return beta_null shape (B,) for one selected gene x program.
    """
    if B <= 0:
        raise ValueError("B must be positive.")
    if program_index < 0 or program_index >= inputs.Y.shape[1]:
        raise ValueError("program_index is out of range.")

    obs_idx = _gene_obs_idx(inputs, gene)
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0]:
        raise ValueError(
            "Cannot compute CRT-null statistics when gene is all/none treated."
        )

    p = _fit_propensity(inputs, obs_idx, propensity_model)
    seed = _gene_seed(gene, base_seed)
    indptr, idx = _sample_crt_indices(p, B, seed)

    _, beta_null = crt_betas_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )
    return beta_null[:, program_index]


def qq_expected_grid(pvals: np.ndarray) -> np.ndarray:
    """
    Expected QQ quantiles for a p-value array.
    """
    arr = np.asarray(pvals)
    m = arr.size
    if m == 0:
        raise ValueError("pvals must contain at least one value.")
    return (np.arange(1, m + 1) - 0.5) / float(m)


def is_bh_adjusted_like(pvals: np.ndarray) -> bool:
    """
    Heuristic detector for BH/q-value-like distributions.
    """
    arr = np.asarray(pvals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    frac_ones = np.mean(arr >= 1.0 - 1e-12)
    unique_ratio = np.unique(arr).size / float(arr.size)
    mean_val = np.mean(arr)
    return (frac_ones > 0.2) or (mean_val > 0.6 and unique_ratio < 0.2)
