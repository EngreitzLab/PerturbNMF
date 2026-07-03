"""
High-level pipeline to run SCEPTRE-like CRT across genes and programs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .adata_utils import (
    build_gene_to_cols,
    clr_from_usage,
    get_covar_matrix,
    get_from_adata_any,
    get_program_names,
    limit_threading,
    to_csc_matrix,
    union_obs_idx_from_cols,
)
from .pipeline_helpers import (
    _empirical_crt,
    _fit_propensity,
    _gene_obs_idx,
    _gene_seed,
    _sample_crt_indices,
    _skew_calibrated_crt,
    _stack_gene_results,
    _stack_raw_outputs,
    _stack_skew_outputs,
)
from .propensity import fit_propensity_logistic
from .crt import crt_betas_for_gene
from .diagnostics import crt_null_pvals_from_null_stats_matrix


@dataclass
class CRTInputs:
    C: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    CTY: np.ndarray
    G: sp.csc_matrix
    guide_names: List[str]
    guide2gene: Dict[str, str]
    gene_to_cols: Dict[str, List[int]]
    program_names: List[str]
    covar_cols: Optional[List[str]] = None


@dataclass
class CRTGeneResult:
    gene: str
    pvals: np.ndarray
    betas: np.ndarray
    n_treated: int
    pvals_sn: Optional[np.ndarray] = None
    skew_params: Optional[np.ndarray] = None
    pvals_raw: Optional[np.ndarray] = None


def prepare_crt_inputs(
    adata: Any,
    usage_key: str = "cnmf_usage",
    covar_key: str = "covar",
    guide_assignment_key: str = "guide_assignment",
    guide_names_key: str = "guide_names",
    guide2gene_key: str = "guide2gene",
    eps_quantile: float = 1e-4,
    add_intercept: bool = True,
    standardize: bool = True,
    numeric_as_category_threshold: Optional[int] = 20,
    clamp_threads: bool = True,
) -> CRTInputs:
    """
    Load matrices from AnnData, build CLR usage, and precompute regression pieces.
    adata: AnnData-like object with required data
    usage_key: key in adata to extract usage matrix
    covar_key: key in adata to extract covariate matrix
    guide_assignment_key: key in adata to extract guide assignment matrix
    guide_names_key: key in adata.uns to extract guide column names if not in DataFrame
    guide2gene_key: key in adata to extract guide-to-gene mapping
    eps_quantile: quantile for flooring small values in usage before CLR
    add_intercept: whether to add intercept column to covariate matrix
    standardize: whether to z-score covariate columns
    numeric_as_category_threshold: treat numeric columns with <= this many unique values as categorical
    clamp_threads: whether to limit threading for numerical libraries
    Returns:
        CRTInputs dataclass with all required inputs for CRT
    """
    if clamp_threads:
        limit_threading()

    C, covar_cols = get_covar_matrix(
        adata,
        covar_key=covar_key,
        add_intercept=add_intercept,
        standardize=standardize,
        numeric_as_category_threshold=numeric_as_category_threshold,
    )

    U = get_from_adata_any(adata, usage_key)
    if isinstance(U, pd.DataFrame):
        U = U.to_numpy()
    Y = clr_from_usage(U, eps_quantile=eps_quantile)

    G_raw = get_from_adata_any(adata, guide_assignment_key)
    G, guide_names = to_csc_matrix(G_raw)
    if guide_names is None:
        if hasattr(adata, "uns") and guide_names_key in getattr(adata, "uns", {}):
            guide_names = list(getattr(adata, "uns")[guide_names_key])
        else:
            raise ValueError(
                "Guide column names are required; provide them via a DataFrame or adata.uns."
            )

    guide2gene_raw = get_from_adata_any(adata, guide2gene_key)
    guide2gene = dict(guide2gene_raw)
    gene_to_cols = build_gene_to_cols(guide_names, guide2gene)
    if not gene_to_cols:
        raise ValueError("No guides mapped to genes; check guide2gene and guide names.")

    if C.shape[0] != Y.shape[0] or C.shape[0] != G.shape[0]:
        raise ValueError(
            f"Cell counts mismatch: C {C.shape[0]}, Y {Y.shape[0]}, G {G.shape[0]}"
        )

    """
        Precompute regression pieces: A = (C^T C)^{-1}, CTY = C^T Y.
        CtC: p x p matrix, C^T C
        A: inverse of CtC
        CTY: p x K matrix, C^T Y
    """
    CtC = C.T @ C
    A = np.linalg.inv(CtC)
    CTY = C.T @ Y

    program_names = get_program_names(adata, Y.shape[1])

    return CRTInputs(
        C=C.astype(np.float64, copy=False),
        Y=Y.astype(np.float64, copy=False),
        A=A.astype(np.float64, copy=False),
        CTY=CTY.astype(np.float64, copy=False),
        G=G,
        guide_names=list(guide_names),
        guide2gene=guide2gene,
        gene_to_cols=gene_to_cols,
        program_names=program_names,
        covar_cols=covar_cols,
    )


def _trivial_gene_result(
    gene: str,
    inputs: CRTInputs,
    n_treated: int,
    calibrate_skew_normal: bool,
) -> CRTGeneResult:
    K = inputs.Y.shape[1]
    return CRTGeneResult(
        gene=gene,
        pvals=np.ones(K, dtype=np.float64),
        betas=np.zeros(K, dtype=np.float64),
        n_treated=int(n_treated),
        pvals_sn=np.ones(K, dtype=np.float64) if calibrate_skew_normal else None,
        skew_params=np.full((K, 3), np.nan) if calibrate_skew_normal else None,
        pvals_raw=np.ones(K, dtype=np.float64) if calibrate_skew_normal else None,
    )


def run_one_gene_union_crt(
    gene: str,
    inputs: CRTInputs,
    B: int = 1023,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
    calibrate_skew_normal: bool = False,
    skew_normal_side_code: int = 0,
) -> CRTGeneResult:
    """
    Run union CRT for a single gene and return p-values and betas across programs.
    gene: target gene name
    inputs: CRTInputs dataclass with all required inputs
    B: number of resamples for CRT
    base_seed: base random seed for reproducibility
    propensity_model: function to fit propensity scores given C and y01
    calibrate_skew_normal: if True, compute skew-normal calibrated p-values
    skew_normal_side_code: 0 two-sided, 1 right-tailed, -1 left-tailed
    y01: binary union indicator for the gene
    Returns:
        CRTGeneResult dataclass with all results for the gene. If calibrate_skew_normal
        is True, pvals contain the skew-normal calibrated values.
    """
    obs_idx = _gene_obs_idx(inputs, gene)

    """
    Handle edge cases with no treated cells or all treated cells.
    In these cases, a meaningful comparison is impossible. 
    The function returns a trivial result (p-values of 1.0, effect sizes of 0) without performing the test.
    """
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0] or B <= 0:
        return _trivial_gene_result(gene, inputs, obs_idx.size, calibrate_skew_normal)

    p = _fit_propensity(inputs, obs_idx, propensity_model)
    seed = _gene_seed(gene, base_seed)

    indptr, idx = _sample_crt_indices(p, B, seed)

    if calibrate_skew_normal:
        pvals_sn, beta_obs, skew_params, pvals_raw = _skew_calibrated_crt(
            inputs, indptr, idx, obs_idx, B, skew_normal_side_code
        )
        pvals = pvals_sn
    else:
        pvals, beta_obs = _empirical_crt(inputs, indptr, idx, obs_idx, B)
        pvals_sn = None
        skew_params = None
        pvals_raw = None
    return CRTGeneResult(
        gene=gene,
        pvals=pvals,
        betas=beta_obs,
        n_treated=int(obs_idx.size),
        pvals_sn=pvals_sn,
        skew_params=skew_params,
        pvals_raw=pvals_raw,
    )


def compute_gene_null_pvals(
    gene: str,
    inputs: CRTInputs,
    B: int = 1023,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
    side_code: int = 0,
) -> np.ndarray:
    """
    Compute CRT-null p-values for a single gene by resampling.
    Returns a (B, K) array of p-values for each null resample and program.
    """
    if B <= 0:
        raise ValueError("B must be positive.")

    obs_idx = _gene_obs_idx(inputs, gene)
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0]:
        raise ValueError(
            "Cannot compute CRT-null p-values when gene is all/none treated."
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
    if side_code not in (-1, 0, 1):
        raise ValueError("side_code must be -1, 0, or 1.")
    if side_code == 0:
        return crt_null_pvals_from_null_stats_matrix(beta_null, two_sided=True)
    if side_code == 1:
        return crt_null_pvals_from_null_stats_matrix(beta_null, two_sided=False)
    return crt_null_pvals_from_null_stats_matrix(-beta_null, two_sided=False)


def compute_guide_set_null_pvals(
    guide_idx: Iterable[int],
    inputs: CRTInputs,
    B: int = 1023,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
    side_code: int = 0,
) -> np.ndarray:
    """
    Compute CRT-null p-values for a guide set by resampling.
    Returns a (B, K) array of p-values for each null resample and program.
    """
    if B <= 0:
        raise ValueError("B must be positive.")

    guide_idx = np.asarray(list(guide_idx), dtype=np.int32)
    if guide_idx.size == 0:
        raise ValueError("guide_idx must contain at least one column index.")
    guide_idx = np.unique(guide_idx)

    obs_idx = union_obs_idx_from_cols(inputs.G, guide_idx)
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0]:
        raise ValueError(
            "Cannot compute CRT-null p-values when guide set is all/none treated."
        )

    p = _fit_propensity(inputs, obs_idx, propensity_model)
    seed = int(base_seed)
    for val in guide_idx:
        seed = (seed * 1000003) ^ int(val)
    seed &= 0xFFFFFFFF
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
    if side_code not in (-1, 0, 1):
        raise ValueError("side_code must be -1, 0, or 1.")
    if side_code == 0:
        return crt_null_pvals_from_null_stats_matrix(beta_null, two_sided=True)
    if side_code == 1:
        return crt_null_pvals_from_null_stats_matrix(beta_null, two_sided=False)
    return crt_null_pvals_from_null_stats_matrix(-beta_null, two_sided=False)


def run_all_genes_union_crt(
    inputs: CRTInputs,
    genes: Optional[Iterable[str]] = None,
    B: int = 1023,
    n_jobs: int = 8,
    base_seed: int = 123,
    propensity_model: Callable = fit_propensity_logistic,
    backend: str = "loky",
    calibrate_skew_normal: bool = False,
    skew_normal_side_code: int = 0,
    return_skew_normal: bool = False,
    return_raw_pvals: bool = False,
    return_format: str = "dict",
) -> Union[Tuple, Dict[str, Any]]:
    """
    Run union CRT across all genes and return DataFrames for p-values and betas.
    inputs: CRTInputs dataclass with all required inputs
    genes: optional list of genes to test; if None, test all genes in inputs.gene_to_cols
    B: number of resamples for CRT
    n_jobs: number of parallel jobs to use
    base_seed: base random seed for reproducibility
    propensity_model: function to fit propensity scores given C and y01
    backend: joblib parallelization backend
    calibrate_skew_normal: if True, compute skew-normal calibrated p-values per gene
    skew_normal_side_code: 0 two-sided, 1 right-tailed, -1 left-tailed
    return_skew_normal: if True, return skew-normal pvals and parameters
    return_raw_pvals: if True, return raw CRT p-values when skew calibration is enabled
    return_format: "dict" (default) or "tuple" for backward-compatible ordering
    Returns:
        dict with keys:
            pvals_df: DataFrame of CRT p-values (genes x programs). If calibrate_skew_normal
                is True, these are skew-normal calibrated values.
            betas_df: DataFrame of CRT effect sizes (genes x programs)
            treated_df: Series of number of treated cells per gene
            results: list of CRTGeneResult dataclasses for all genes
            pvals_raw_df (optional): raw CRT p-values (genes x programs)
            pvals_skew_df (optional): skew-normal p-values (genes x programs)
            skew_params (optional): fitted parameters (genes x programs x 3)
    """
    gene_list = sorted(inputs.gene_to_cols.keys()) if genes is None else list(genes)

    results: List[CRTGeneResult] = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(run_one_gene_union_crt)(
            gene,
            inputs,
            B=B,
            base_seed=base_seed,
            propensity_model=propensity_model,
            calibrate_skew_normal=calibrate_skew_normal,
            skew_normal_side_code=skew_normal_side_code,
        )
        for gene in gene_list
    )

    pvals_df, betas_df, treated_df = _stack_gene_results(
        results, gene_list, inputs.program_names
    )
    if return_raw_pvals and not calibrate_skew_normal:
        raise ValueError("return_raw_pvals=True requires calibrate_skew_normal=True.")

    if return_format not in ("dict", "tuple"):
        raise ValueError('return_format must be "dict" or "tuple".')

    pvals_raw_df = None
    if return_raw_pvals:
        pvals_raw_df = _stack_raw_outputs(results, gene_list, inputs.program_names)

    pvals_skew_df = None
    skew_params = None
    if return_skew_normal:
        if not calibrate_skew_normal:
            raise ValueError("return_skew_normal=True requires calibrate_skew_normal=True.")
        pvals_skew_df, skew_params = _stack_skew_outputs(
            results, gene_list, inputs.program_names
        )

    if return_format == "tuple":
        out = [pvals_df, betas_df, treated_df, results]
        if return_raw_pvals:
            out.append(pvals_raw_df)
        if return_skew_normal:
            out.extend([pvals_skew_df, skew_params])
        return tuple(out)

    out_dict: Dict[str, Any] = {
        "pvals_df": pvals_df,
        "betas_df": betas_df,
        "treated_df": treated_df,
        "results": results,
    }
    if return_raw_pvals:
        out_dict["pvals_raw_df"] = pvals_raw_df
    if return_skew_normal:
        out_dict["pvals_skew_df"] = pvals_skew_df
        out_dict["skew_params"] = skew_params
    return out_dict

    return pvals_df, betas_df, treated_df, results


def store_results_in_adata(
    adata: Any,
    pvals_df: pd.DataFrame,
    betas_df: pd.DataFrame,
    treated_df: pd.Series,
    prefix: str = "crt_union_gene_program",
    pvals_skew_df: Optional[pd.DataFrame] = None,
    skew_params: Optional[np.ndarray] = None,
) -> None:
    """
    Save CRT outputs into adata.uns with a consistent prefix.
    adata: AnnData-like object to store results in
    pvals_df: DataFrame of CRT p-values (genes x programs)
    betas_df: DataFrame of CRT effect sizes (genes x programs)
    treated_df: Series of number of treated cells per gene
    prefix: prefix for keys in adata.uns to store results
    pvals_skew_df: optional DataFrame of skew-normal p-values (genes x programs)
    skew_params: optional array of skew-normal parameters (genes x programs x 3)
    """

    if not hasattr(adata, "uns"):
        raise AttributeError("AnnData-like object missing `.uns` to store results.")
    adata.uns[f"{prefix}_pvals"] = pvals_df
    adata.uns[f"{prefix}_betas"] = betas_df
    adata.uns[f"{prefix}_n_positive"] = treated_df
    if pvals_skew_df is not None:
        adata.uns[f"{prefix}_pvals_skew"] = pvals_skew_df
    if skew_params is not None:
        adata.uns[f"{prefix}_skew_params"] = skew_params
