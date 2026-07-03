"""
Utilities for extracting matrices from AnnData and preparing covariates/outcomes.
"""

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def limit_threading(n_threads: int = 1) -> None:
    """
    Clamp BLAS/OMP thread counts to keep CRT resampling predictable.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n_threads))


def get_from_adata_any(adata: Any, key: str) -> Any:
    """
    Try common AnnData containers, then fall back to item access.
    """
    if hasattr(adata, "obsm") and key in adata.obsm:
        return adata.obsm[key]
    if hasattr(adata, "layers") and key in adata.layers:
        return adata.layers[key]
    if hasattr(adata, "obsp") and key in adata.obsp:
        return adata.obsp[key]
    if hasattr(adata, "uns") and key in adata.uns:
        return adata.uns[key]
    if hasattr(adata, "obs") and key in adata.obs:
        return adata.obs[key]
    try:
        return adata[key]
    except Exception as exc:  # pylint: disable=broad-except
        raise KeyError(f"Could not find `{key}` in adata.* containers.") from exc


def to_csc_matrix(G: Any) -> Tuple[sp.csc_matrix, Optional[List[str]]]:
    """
    Cast guide assignment to CSC. Returns matrix and optional column names.
    The CSC format is particularly efficient for the column-wise operations
    often used in statistical genetics and single-cell analysis.
    """
    if isinstance(G, pd.DataFrame):
        guide_names = list(G.columns)
        return sp.csc_matrix(G.to_numpy()), guide_names
    if sp.issparse(G):
        return G.tocsc(), None
    arr = np.asarray(G)
    return sp.csc_matrix(arr), None


def encode_categorical_covariates(
    df: pd.DataFrame,
    drop_first: bool = True,
    dummy_na: bool = False,
    numeric_as_category_threshold: Optional[int] = None,
) -> pd.DataFrame:
    """
    One-hot encode categorical/object/bool columns in a covariate DataFrame.
    """
    cat_cols = []
    for col in df.columns:
        series = df[col]
        if (
            pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            cat_cols.append(col)
        elif (
            numeric_as_category_threshold is not None
            and pd.api.types.is_numeric_dtype(series)
        ):
            unique_count = series.nunique(dropna=not dummy_na)
            if unique_count <= numeric_as_category_threshold:
                cat_cols.append(col)

    if not cat_cols:
        return df.copy()

    # Drop categories with no rows in this subset so get_dummies does not emit
    # all-zero dummy columns (absent levels leak in when a Categorical is sliced,
    # e.g. per-condition), which would make C^T C singular downstream.
    df = df.copy()
    for col in cat_cols:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.remove_unused_categories()

    return pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=drop_first,
        dummy_na=dummy_na,
        dtype=np.float64,
    )


def get_covar_matrix(
    adata: Any,
    covar_key: str = "covar",
    add_intercept: bool = True,
    standardize: bool = True,
    one_hot_encode: bool = True,
    drop_first: bool = True,
    numeric_as_category_threshold: Optional[int] = 20,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Fetch covariates, optionally z-score columns and prepend intercept.
    covar_key: key in adata to fetch covariate matrix
    add_intercept: whether to prepend intercept column
    standardize: whether to z-score covariate columns
    one_hot_encode: whether to one-hot encode categorical columns in DataFrame input
    drop_first: drop one level per categorical column to avoid collinearity
    Returns:
        C: covariate matrix (N x p) as numpy array
        covar_cols: list of covariate column names if available, else None
    """
    C = get_from_adata_any(adata, covar_key)
    if isinstance(C, pd.DataFrame):
        if one_hot_encode:
            C = encode_categorical_covariates(
                C,
                drop_first=drop_first,
                numeric_as_category_threshold=numeric_as_category_threshold,
            )
        covar_cols = list(C.columns)
        C = C.to_numpy()
    else:
        covar_cols = None
        C = np.asarray(C)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        if C.dtype.kind in ("O", "U", "S"):
            if not one_hot_encode:
                raise ValueError(
                    "Covariate matrix contains non-numeric values. "
                    "Enable one_hot_encode or provide a DataFrame."
                )
            df = pd.DataFrame(C, columns=[f"covar_{i}" for i in range(C.shape[1])])
            df = encode_categorical_covariates(
                df,
                drop_first=drop_first,
                numeric_as_category_threshold=numeric_as_category_threshold,
            )
            covar_cols = list(df.columns)
            C = df.to_numpy()
    C = C.astype(np.float64, copy=False)

    if standardize:
        mu = C.mean(axis=0)
        sd = C.std(axis=0)
        sd[sd == 0.0] = 1.0
        C = (C - mu) / sd

    if add_intercept:
        C = np.column_stack([np.ones((C.shape[0], 1), dtype=np.float64), C])

    return C, covar_cols


def clr_from_usage(U: Any, eps_quantile: float = 1e-4) -> np.ndarray:
    """
    Compute centered log-ratio from usage matrix with flooring and renormalization.
    U: usage matrix (N x K)
    eps_quantile: quantile for flooring small values in U
    Returns:
        logU: centered log-ratio matrix (N x K)
    """
    U = np.asarray(U, dtype=np.float64)
    eps = np.quantile(U, eps_quantile)
    U2 = np.maximum(U, eps)
    U2 /= U2.sum(axis=1, keepdims=True)
    logU = np.log(U2)
    return logU - logU.mean(axis=1, keepdims=True)


def build_gene_to_cols(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
) -> Dict[str, List[int]]:
    """
    Map each gene to the list of guide column indexes targeting it.
    guide_names: list of guide names corresponding to columns in G
    guide2gene: mapping from guide name to target gene name
    Returns:
        gene_to_cols: mapping from gene name to list of guide column indexes
    """
    guide_to_col = {g: i for i, g in enumerate(guide_names)}
    gene_to_cols: Dict[str, List[int]] = {}
    for guide, gene in guide2gene.items():
        j = guide_to_col.get(guide)
        if j is None:
            continue
        gene_to_cols.setdefault(gene, []).append(j)
    return {gene: sorted(cols) for gene, cols in gene_to_cols.items()}


def union_obs_idx_from_cols(G_csc: sp.csc_matrix, cols: Iterable[int]) -> np.ndarray:
    """
    Collect treated cell indices for a set of guide columns (union indicator).
    G_csc: guide assignment matrix in CSC format
    cols: list of guide column indexes
    Returns:
        rows: sorted array of unique cell indices with at least one guide in cols
    """
    indptr = G_csc.indptr
    indices = G_csc.indices
    chunks: List[np.ndarray] = []
    for j in cols:
        chunks.append(indices[indptr[j] : indptr[j + 1]])
    if not chunks:
        return np.empty(0, dtype=np.int32)
    rows = np.concatenate(chunks).astype(np.int32, copy=False)
    return np.unique(rows)


def get_program_names(adata: Any, n_programs: int) -> List[str]:
    """
    Resolve program names if stored in adata.uns, otherwise auto-number.
    """
    if hasattr(adata, "uns") and isinstance(getattr(adata, "uns", None), dict):
        uns = getattr(adata, "uns")
        if "program_names" in uns:
            return list(uns["program_names"])
    return [f"program_{k}" for k in range(n_programs)]
