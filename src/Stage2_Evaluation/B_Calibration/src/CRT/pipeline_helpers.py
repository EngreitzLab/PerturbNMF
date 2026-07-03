"""
Internal helpers for CRT pipeline orchestration.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd

from .adata_utils import union_obs_idx_from_cols
from .crt import crt_betas_for_gene, crt_index_sampler_fast_numba, crt_pvals_for_gene
from .skew_normal import compute_empirical_p_value, fit_and_evaluate_skew_normal


def _extract_probabilities(output) -> np.ndarray:
    """
    Normalize propensity model outputs into probability vector.
    """
    if isinstance(output, tuple) or isinstance(output, list):
        return np.asarray(output[0], dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _gene_obs_idx(inputs: Any, gene: str) -> np.ndarray:
    if gene not in inputs.gene_to_cols:
        raise KeyError(f"Gene `{gene}` not present in gene_to_cols mapping.")
    return union_obs_idx_from_cols(inputs.G, inputs.gene_to_cols[gene])


def _gene_seed(gene: str, base_seed: int) -> int:
    return (hash(gene) ^ base_seed) & 0xFFFFFFFF


def _union_indicator(n_cells: int, obs_idx: np.ndarray) -> np.ndarray:
    y01 = np.zeros(n_cells, dtype=np.int8)
    y01[obs_idx] = 1
    return y01


def _fit_propensity(
    inputs: Any,
    obs_idx: np.ndarray,
    propensity_model: Callable,
) -> np.ndarray:
    y01 = _union_indicator(inputs.C.shape[0], obs_idx)
    return _extract_probabilities(propensity_model(inputs.C, y01))


def _sample_crt_indices(p: np.ndarray, B: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    return crt_index_sampler_fast_numba(p, B, seed)


def _empirical_crt(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return crt_pvals_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )


def _compute_skew_normal_pvals(
    beta_obs: np.ndarray,
    beta_null: np.ndarray,
    side_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit skew-normal to null statistics per program and compute calibrated p-values.
    """
    K = beta_obs.shape[0]
    pvals_sn = np.empty(K, dtype=np.float64)
    params = np.empty((K, 3), dtype=np.float64)

    for k in range(K):
        null_k = beta_null[:, k]
        raw_k = (1.0 + np.sum(np.abs(null_k) >= abs(beta_obs[k]))) / (
            null_k.shape[0] + 1.0
        )
        mu = null_k.mean()
        sd = null_k.std()
        if not np.isfinite(sd) or sd <= 0.0:
            pvals_sn[k] = raw_k
            params[k, :] = np.nan
            continue

        z_null = (null_k - mu) / sd
        z_obs = (beta_obs[k] - mu) / sd

        sn_result = fit_and_evaluate_skew_normal(z_obs, z_null, side_code)
        params_k = sn_result[:3]
        params[k, :] = params_k if np.all(np.isfinite(params_k)) else np.nan
        p_val = sn_result[3]
        if p_val < 0.0:
            p_val = compute_empirical_p_value(z_null, z_obs, side_code)
        if not np.isfinite(p_val):
            p_val = raw_k
        pvals_sn[k] = p_val

    return pvals_sn, params


def _raw_pvals_from_betas(beta_obs: np.ndarray, beta_null: np.ndarray) -> np.ndarray:
    abs_obs = np.abs(beta_obs)
    ge = np.sum(np.abs(beta_null) >= abs_obs, axis=0)
    B = beta_null.shape[0]
    return (1.0 + ge) / (B + 1.0)


def _skew_calibrated_crt(
    inputs: Any,
    indptr: np.ndarray,
    idx: np.ndarray,
    obs_idx: np.ndarray,
    B: int,
    side_code: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta_obs, beta_null = crt_betas_for_gene(
        indptr,
        idx,
        inputs.C,
        inputs.Y,
        inputs.A,
        inputs.CTY,
        obs_idx.astype(np.int32),
        B,
    )
    pvals_sn, skew_params = _compute_skew_normal_pvals(beta_obs, beta_null, side_code)
    pvals_raw = _raw_pvals_from_betas(beta_obs, beta_null)
    return pvals_sn, beta_obs, skew_params, pvals_raw


def _stack_gene_results(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pval_mat = np.vstack([r.pvals for r in results])
    beta_mat = np.vstack([r.betas for r in results])
    treated = np.array([r.n_treated for r in results], dtype=int)

    pvals_df = pd.DataFrame(pval_mat, index=gene_list, columns=program_names)
    betas_df = pd.DataFrame(beta_mat, index=gene_list, columns=program_names)
    treated_df = pd.Series(treated, index=gene_list, name="n_union_positive_cells")
    return pvals_df, betas_df, treated_df


def _stack_skew_outputs(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> Tuple[pd.DataFrame, np.ndarray]:
    pvals_sn_mat = np.vstack([r.pvals_sn for r in results])
    skew_params = np.stack([r.skew_params for r in results])
    pvals_skew_df = pd.DataFrame(
        pvals_sn_mat, index=gene_list, columns=program_names
    )
    return pvals_skew_df, skew_params


def _stack_raw_outputs(
    results: List[Any],
    gene_list: List[str],
    program_names: List[str],
) -> pd.DataFrame:
    pvals_raw_mat = np.vstack([r.pvals_raw for r in results])
    return pd.DataFrame(pvals_raw_mat, index=gene_list, columns=program_names)
