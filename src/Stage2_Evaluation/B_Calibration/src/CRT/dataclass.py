# Dataclasses used in CRT

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class CRT_gene_result:
    gene: str  # target gene name
    n_treated: int  # number of cells carrying any of the gene's guides
    treated_obs_idx: np.ndarray  # row indexes of those cells (int32, sorted)
    pvalue: (
        np.ndarray
    )  # K final p-values (skew-normal where fit, else empirical); 1 if not testable
    betas: np.ndarray  # K effect sizes; 0 if not testable
    pvalue_skewnorm: Optional[np.ndarray] = (
        None  # K skew-normal p-values; NaN where not fit or rejected
    )
    skew_params: Optional[np.ndarray] = None  # K x 3 skew-normal fit parameters
    pvalue_empirical: Optional[np.ndarray] = None  # K empirical resampling p-values
    control_means: Optional[np.ndarray] = None  # K mean Usage_norm in control cells
    n_resampled: int = 0  # resampling rounds used (stage 1, or stage 1 + 2)
    n_control: int = 0  # control cells in the regression (see CRT_obj.control_group)
    propensity_n_iter: float = (
        np.nan
    )  # lbfgs iterations of the propensity fit (0 = intercept-only, no fit;
    # NaN = no propensity model, moi="low" or untestable gene)
    propensity_sum_ratio: float = (
        np.nan
    )  # sum(p) / n_treated; 1 at a converged fit (NaN as above)


@dataclass
class CRT_obj:
    # input
    Covariates: np.ndarray  # N x p design matrix (intercept first if added)
    Usage_norm: np.ndarray  # N x K outcome (CLR or usage share)
    CtC_inv: np.ndarray  # p x p, (Covariates^T Covariates)^-1
    CtU_norm: np.ndarray  # p x K, Covariates^T Usage_norm
    Guide_assignment: sp.csc_matrix  # N x G guide assignment
    guide_names: List[str]  # G guide names (columns of Guide_assignment)
    guide_targets: Dict[str, List[int]]  # gene -> guide column indexes
    guide2gene: Dict[str, str]  # guide name -> target gene
    covariate_names: Optional[List[str]] = (
        None  # column names of Covariates (no intercept)
    )
    covariate_strata: Optional[np.ndarray] = (
        None  # N stratum ids for fixed-count resampling
    )
    has_continuous_covariates: bool = (
        False  # True -> fixed-count resampling is approximate
    )
    has_intercept: bool = True  # True -> column 0 of Covariates is the intercept
    normalization: str = "row_normalize"  # "row_normalize" or "clr"
    selected_gene_names: Optional[List[str]] = (
        None  # target genes to test (keys of guide_targets)
    )
    selected_program_names: Optional[List[str]] = (
        None  # K program names (columns of Usage_norm)
    )
    moi: str = (
        "high"  # "high": CRT (Bernoulli from propensity), controls = all other cells
    )
    # "low": permutation within strata, controls = control-guide cells
    non_targeting_guide_names: Optional[List[str]] = (
        None  # control "genes" (e.g. "non-targeting"); required for moi="low"
    )
    number_guide: Optional[int] = (
        None  # guides per NTC fake gene (null); None = median guides per tested gene
    )
    control_obs_idx: Optional[np.ndarray] = (
        None  # cells carrying a control guide (moi="low")
    )
    control_group: str = "all"  # moi="high" controls: "all" other cells (as before),
    # "ntc_cells" (cells carrying a control guide), or "matched"
    # (propensity-matched, matched_cells.select_matched_cells)
    guide_cell_obs_idx: Optional[np.ndarray] = (
        None  # cells carrying >= 1 guide (any gene, controls included)
    )
    matching_ratio: int = 4  # control_group="matched": controls per treated cell
    matching_caliper: Optional[float] = (
        None  # control_group="matched": caliper in SDs of the distance
    )
    matching_distance: str = (
        "propensity"  # control_group="matched": "propensity" or "logit"
    )
    resampled_rounds: int = 4999  # stage-1 resampling rounds (sceptre B1 = 499)
    extra_resampled_rounds: int = 49999  # stage-2 rounds added when a program's stage-1 p <= threshold (sceptre B2 = 4999)
    extra_rounds_p_threshold: float = (
        0.05  # stage-1 p < this triggers stage 2 (and the skew-normal fit)
    )
    stage3_resampled_rounds: int = (
        0  # stage-3 rounds added when a skew-normal fit is rejected (sceptre B3)
    )
    seed: int = 123  # base seed; each gene's seed = crc32(gene) ^ seed
    pvalue_method: str = (
        "empirical"  # "empirical" or "skew_normal" (sceptre-style tail fit)
    )
    propensity_penalty: Optional[str] = (
        None  # moi="high" propensity: None (MLE, sceptre), "l2" or "old_perturbnmf"
    )
    propensity_C: float = 1.0  # inverse L2 strength (propensity_penalty="l2")
    propensity_tol: Optional[float] = (
        None  # lbfgs tol; None = mode default (1e-10; old_perturbnmf 1e-4)
    )
    propensity_max_iter: Optional[int] = (
        None  # lbfgs max_iter; None = mode default (10,000; old_perturbnmf 200)
    )

    # output: one result per tested gene, keyed by gene name
    gene_results: Dict[str, CRT_gene_result] = field(default_factory=dict)
