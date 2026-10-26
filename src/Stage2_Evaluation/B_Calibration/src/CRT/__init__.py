"""
Conditional randomization test (CRT) of target genes on cNMF program usage.

Copied from Tools/CRT/src (the CRT rewrite); keep the two in sync. Workflow:

    mdata = load_modality(h5mu, "cNMF", covariate_cols)
    for condition, mdata_con in split_by_condition(mdata, "cNMF", "sample"):
        obj = prepare_crt_inputs(mdata_con, ...)
        conditional_randomized_test(obj, n_jobs=...)
        null_obj = prepare_null_guide_assignment(obj, ...)
        conditional_randomized_test(null_obj, n_jobs=...)
        format_results_table(obj), plot_qq_real_vs_null(obj, null_obj, ...)
"""

from .covariate_preparation import encode_categorical_covariates, get_covar_matrix
from .crt import conditional_randomized_test
from .dataclass import CRT_gene_result, CRT_obj
from .null_guide_assignment_preparation import prepare_null_guide_assignment
from .plotting import plot_qq_real_vs_null
from .prepare import prepare_crt_inputs
from .utility import (
    adjust_pvalues,
    format_results_table,
    load_modality,
    load_results_table,
    split_by_condition,
)

__all__ = [
    "encode_categorical_covariates",
    "get_covar_matrix",
    "conditional_randomized_test",
    "CRT_gene_result",
    "CRT_obj",
    "prepare_null_guide_assignment",
    "plot_qq_real_vs_null",
    "prepare_crt_inputs",
    "adjust_pvalues",
    "format_results_table",
    "load_modality",
    "load_results_table",
    "split_by_condition",
]
