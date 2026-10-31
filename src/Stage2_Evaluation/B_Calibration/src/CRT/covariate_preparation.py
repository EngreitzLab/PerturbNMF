"""
Build the covariate design matrix from adata.obs: collect columns, one-hot
encode, z-score, add an intercept, and derive resampling strata.
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def encode_categorical_covariates(
    df: pd.DataFrame,
    drop_original: bool = True,
    dummy_na: bool = False,
    numeric_as_category_threshold: Optional[int] = None,
) -> pd.DataFrame:
    """
    One-hot encode categorical/object/bool columns in a covariate DataFrame.
    All categorical values (dispite number) are one-hotted
    For numberical values, if it has > 20 unique numbers, it is continous, else discrete and one-hotted

    Args:
        df: covariate table (cells x covariates).
        drop_original: drop one level per categorical column to avoid collinearity.
        dummy_na: add a dummy column for NaN.
        numeric_as_category_threshold: treat numeric columns with <= this many
            unique values as categorical (None disables).

    Returns:
        DataFrame with categorical columns replaced by float64 dummies.
    """
    cat_cols = []
    for col in df.columns:
        series = df[col]
        if (
            isinstance(series.dtype, pd.CategoricalDtype)
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

    # Drop categories with no rows so get_dummies does not emit all-zero dummy
    # columns, which would make C^T C singular downstream.
    df = df.copy()
    for col in cat_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].cat.remove_unused_categories()

    return pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=drop_original,
        dummy_na=dummy_na,
        dtype=np.float64,
    )


def build_covariate_table(
    adata: Any,
    covariates: Optional[Sequence[str]] = None,
    log_covariates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Collect covariate columns from adata.obs into one table.
    Log trasnform values like UMI or gene counts, that are skewed and wide-range
    these values can be continous or integer

    Args:
        adata: AnnData whose .obs holds the covariates.
        covariates: obs columns used as is.
        log_covariates: obs columns used as log1p(x), named "log_<col>".

    Returns:
        DataFrame (cells x covariates) indexed by adata.obs_names; no columns
        if neither list is given.
    """
    covariate_dict = {}
    for key in covariates or []:
        covariate_dict[key] = adata.obs[key]
    for key in log_covariates or []:
        covariate_dict[f"log_{key}"] = np.log1p(adata.obs[key])

    return pd.DataFrame(covariate_dict, index=adata.obs_names)


def get_covar_matrix(
    covariate_table: pd.DataFrame,
    add_intercept: bool = True,
    standardize: bool = True,
    one_hot_encode: bool = True,
    drop_original: bool = True,
    numeric_as_category_threshold: Optional[int] = 20,
) -> Tuple[np.ndarray, List[str]]:
    """
    Turn a covariate table into a design matrix: one-hot encode, optionally
    z-score columns, and prepend an intercept.

    Args:
        covariate_table: cells x covariates DataFrame (from build_covariate_table).
        add_intercept: whether to prepend an intercept column.
        standardize: whether to z-score covariate columns.
        one_hot_encode: whether to also one-hot encode numeric columns with
            <= numeric_as_category_threshold unique values. Categorical, string
            and bool columns are always encoded.
        drop_original: drop one level per categorical column to avoid collinearity.
        numeric_as_category_threshold: treat numeric columns with <= this many
            unique values as categorical.

    Returns:
        Covariates: design matrix (N x p) as float64 numpy array.
        covariate_names: column names (without intercept).
    """
    if not isinstance(covariate_table, pd.DataFrame):
        raise TypeError(
            f"covariate_table must be a pandas DataFrame; got {type(covariate_table).__name__}."
        )
    # Categorical/string/bool columns are always one-hot encoded (they cannot
    # enter the regression as numbers); one_hot_encode only controls whether
    # low-cardinality numeric columns are encoded too.
    covariate_table = encode_categorical_covariates(
        covariate_table,
        drop_original=drop_original,
        numeric_as_category_threshold=numeric_as_category_threshold
        if one_hot_encode
        else None,
    )
    covariate_names = list(covariate_table.columns)
    Covariates = covariate_table.to_numpy().astype(np.float64, copy=False)

    if standardize and Covariates.shape[1] > 0:
        mu = Covariates.mean(axis=0)
        sd = Covariates.std(axis=0)
        sd[sd == 0.0] = 1.0
        Covariates = (Covariates - mu) / sd

    if add_intercept:
        Covariates = np.column_stack(
            [np.ones((Covariates.shape[0], 1), dtype=np.float64), Covariates]
        )

    return Covariates, covariate_names


def covariate_strata_from_design(Covariates: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Group cells into strata of identical discrete covariate values, for
    fixed-count CRT resampling.

    A design column counts as discrete if it has at most 2 distinct values
    (one-hot dummies, binary flags). Constant columns (the intercept) are
    ignored. Strata are the unique rows of the discrete columns.

    Args:
        Covariates: design matrix (N x p) as built by get_covar_matrix.

    Returns:
        strata: int64 array of length N, stratum id per cell (0..S-1).
        has_continuous: True if any non-constant column has > 2 distinct
            values; fixed-count resampling is then approximate.
    """
    Covariates = np.asarray(Covariates, dtype=np.float64)
    discrete_cols: List[int] = []
    has_continuous = False
    for j in range(Covariates.shape[1]):
        n_unique = np.unique(Covariates[:, j]).size
        if n_unique == 1:
            continue
        if n_unique == 2:
            discrete_cols.append(j)
        else:
            has_continuous = True
    if not discrete_cols:
        return np.zeros(Covariates.shape[0], dtype=np.int64), has_continuous
    _, strata = np.unique(Covariates[:, discrete_cols], axis=0, return_inverse=True)
    return strata.astype(np.int64).ravel(), has_continuous
