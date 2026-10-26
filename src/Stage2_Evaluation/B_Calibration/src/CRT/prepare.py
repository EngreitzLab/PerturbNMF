"""
Build CRT inputs (covariates, outcome Usage_norm, guide assignment) from a MuData object.
"""

import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from .covariate_preparation import (
    build_covariate_table,
    covariate_strata_from_design,
    get_covar_matrix,
)
from .dataclass import CRT_obj
from .utility import (
    get_from_mudata,
    to_csc_matrix,
)

# Usage normalization: "clr" = centered log-ratio of floored usage (clr_from_usage);
# "row_normalize" = per-cell usage share, no log and no floor (row_normalize_usage).
NORMALIZATIONS = ("clr", "row_normalize")


def clr_from_usage(Usage: Any, eps: float = 1e-3) -> np.ndarray:
    """
    Compute centered log-ratio from usage matrix with flooring and renormalization.

    For cell i and program k (K programs), with shares S_i = U_i / sum_j U_ij and
    P_i = max(S_i, eps) / sum_j max(S_ij, eps):

        CLR(P_i)_k = log( P_ik / g(P_i) ) = log(P_ik) - (1/K) * sum_j log(P_ij)

    where g(P_i) = (prod_j P_ij)^(1/K) is the geometric mean of row i,
    so each output row sums to 0.

    The floor is applied to per-cell shares, so eps means the same thing for
    raw (depth-scaled) and row-normalized cNMF usage. Shares below eps,
    including zeros, are raised to eps so log() stays finite. The floor sets
    how extreme a zero looks (log(typical / eps)): too small (e.g. 1e-8) and
    zero-usage cells dominate the CRT null spread and cost power; too large
    (e.g. 1e-2) and real low usage is merged with zeros. See
    normalization="row_normalize" for a floor-free alternative.

    Args:
        Usage: usage matrix U (N cells x K programs); any array-like, non-negative.
        eps: floor on per-cell program shares (default 1e-3).

    Returns:
        CLR-transformed usage matrix (N x K), float64.
    """
    if not 0 < eps < 1:
        raise ValueError(f"eps must be in (0, 1); got {eps}.")
    Usage_share = row_normalize_usage(Usage)
    Usage_floored = np.maximum(Usage_share, eps)
    Usage_floored /= Usage_floored.sum(axis=1, keepdims=True)
    log_usage_floored = np.log(Usage_floored)

    return log_usage_floored - log_usage_floored.mean(axis=1, keepdims=True)


def row_normalize_usage(Usage: Any) -> np.ndarray:
    """
    Row-normalize a usage matrix to per-cell program shares (rows sum to 1).

        Y_ik = U_ik / sum_j U_ij


    Args:
        Usage: usage matrix U (N cells x K programs), non-negative.

    Returns:
        Usage share matrix (N x K), float64.
    """
    Usage = np.asarray(Usage, dtype=np.float64)
    if (Usage < 0).any():
        raise ValueError("Usage matrix has negative entries; expected non-negative usage.")
    row_sums = Usage.sum(axis=1, keepdims=True)
    n_empty = int((row_sums <= 0).sum())
    if n_empty:
        raise ValueError(
            f"{n_empty} cells have zero total usage; cannot row-normalize to shares."
        )
    return Usage / row_sums


def build_gene_to_guide(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
) -> Dict[str, List[int]]:
    """
    Map each gene to the list of guide column indexes targeting it.

    Args:
        guide_names: guide names, in the column order of Guide_assignment.
        guide2gene: mapping from guide name to target gene name.

    Returns:
        guide_targets: mapping from gene name to sorted guide column indexes.
    """
    guide_to_col = {g: i for i, g in enumerate(guide_names)}
    guide_targets: Dict[str, List[int]] = {}
    for guide, gene in guide2gene.items():
        j = guide_to_col.get(guide)
        if j is None:
            continue
        guide_targets.setdefault(gene, []).append(j)
    return {gene: sorted(cols) for gene, cols in guide_targets.items()}


def prepare_crt_inputs(
    mdata: Any,
    usage_key: str = "cNMF",
    guide_assignment_key: str = "guide_assignment",
    guide_names_key: str = "guide_names",
    guide_targets_key: str = "guide_targets",
    covariates: Optional[Sequence[str]] = None,
    log_covariates: Optional[Sequence[str]] = None,
    selected_gene_names: Optional[Sequence[str]] = None,
    selected_program_names: Optional[Sequence[str]] = None,
    eps: float = 1e-3,
    add_intercept: bool = True,
    standardize: bool = True,
    numeric_as_category_threshold: Optional[int] = 20,
    normalization: Literal["clr", "row_normalize"] = "row_normalize",
    moi: Literal["high", "low"] = "high",
    non_targeting_guide_names: Optional[Sequence[str]] = None,
    number_guide: Optional[int] = 6,
    resampled_rounds: int = 4999,
    extra_resampled_rounds: int = 49999,
    extra_rounds_p_threshold: float = 0.05,
    stage3_resampled_rounds: int = 0,
    seed: int = 123,
    pvalue_method: Literal["empirical", "skew_normal"] = "empirical",
    propensity_penalty: Optional[Literal["l2", "old_perturbnmf"]] = None,
    propensity_C: float = 1.0,
    propensity_tol: Optional[float] = None,
    propensity_max_iter: Optional[int] = None,
    control_group: Literal["all", "ntc_cells", "matched"] = "all",
    matching_ratio: int = 4,
    matching_caliper: Optional[float] = None,
    matching_distance: Literal["propensity", "logit"] = "propensity",
) -> CRT_obj:
    """
    Load matrices from one MuData modality, build the normalized usage
    Usage_norm, and precompute regression pieces.

    Args:
        mdata: MuData object.
        usage_key: modality holding usage (its .X), guides and covariates
            (e.g. "cNMF").
        guide_assignment_key: key for the cells x guides matrix.
        guide_names_key: uns key with guide names, used if the guide matrix
            has no column names.
        guide_targets_key: uns key with target genes aligned to guide names.
        covariates: obs columns used as covariates.
        log_covariates: obs columns used as log1p covariates.
        selected_gene_names: target genes to test (keys of the guide -> gene
            mapping); None tests every gene with guides.
        selected_program_names: programs to test, named "<k>" (1-based, e.g. "1")
            for column k - 1 of .X; None tests all. Unselected programs are dropped before
            normalization, so shares (or the CLR geometric mean) are computed
            over the selected programs only.
        eps: floor on per-cell program shares before CLR (normalization="clr" only).
        add_intercept: whether to add an intercept column to the outcome
            regression design. Stored as CRT_obj.has_intercept; the propensity
            model always fits an intercept either way.
        standardize: whether to z-score covariate columns.
        numeric_as_category_threshold: treat numeric columns with <= this many
            unique values as categorical.
        normalization: "clr" (Usage_norm = CLR of floored usage; beta is a CLR
            shift) or "row_normalize" (Usage_norm = per-cell usage share, no
            log/floor; beta is the covariate-adjusted difference in mean usage
            share).
        moi: "high" (sceptre's high-MOI default: CRT, null treated sets drawn
            Bernoulli(p) from a per-gene propensity model, controls = all other
            cells) or "low" (sceptre's low-MOI default: permutation test, controls
            = cells with a control guide, treated labels shuffled within
            covariate strata with fixed per-stratum counts).
        non_targeting_guide_names: control target names as they appear in
            uns[guide_targets_key] (e.g. ["non-targeting"]), not individual guide
            names (e.g. "non-targeting_00014"). Required for moi="low"; when given,
            they are left out of the default selected_gene_names.
        number_guide: guides per NTC fake gene in the null
            (prepare_null_guide_assignment); None uses round(median guides per
            tested gene), capped at the number of NTC guides (sceptre default).
        resampled_rounds: stage-1 resampling rounds per gene (default 4999;
            sceptre B1 = 499).
        extra_resampled_rounds: stage-2 rounds added when any program's stage-1
            p < extra_rounds_p_threshold (default 49999; sceptre B2 = 4999);
            0 disables.
        extra_rounds_p_threshold: programs with stage-1 p < this (default 0.05) get the
            stage-2 rounds and, with pvalue_method="skew_normal", the skew-normal fit.
        stage3_resampled_rounds: pvalue_method="skew_normal" only; rounds added
            when a flagged program's skew-normal fit is rejected, before its
            empirical p is used (sceptre B3 = 24999 for permutations, 0 for
            CRT); 0 disables.
        seed: base seed; gene g resamples with seed crc32(g) ^ seed.
        pvalue_method: "empirical" ((1 + #extreme) / (1 + rounds), floor
            1 / (1 + rounds)) or "skew_normal" (sceptre-style: programs with
            stage-1 p < extra_rounds_p_threshold get a two-sided p from a
            skew-normal fitted to all null rounds; rejected fits fall back to
            the empirical p). Both p-values are stored per gene.
        propensity_penalty: moi="high" propensity model: None (unpenalized MLE,
            as sceptre) or "l2" (ridge on the covariate coefficients, intercept
            unpenalized; old PerturbNMF used L2 with C = 1) or "old_perturbnmf"
            (exact old fit: L2, tol 1e-4, max_iter 200, clipped; comparison only,
            its null is anti-conservative).
        propensity_C: inverse L2 strength (sklearn C); smaller = stronger shrinkage.
        propensity_tol: lbfgs tolerance of the propensity fit; None keeps the
            mode's default (1e-10 with a sum(p) = n treated check; old_perturbnmf
            1e-4). An explicit value skips the check (tolerance tests).
        propensity_max_iter: lbfgs max_iter; None keeps the mode's default
            (10,000; old_perturbnmf 200).
        control_group: control cells in each gene's regression and resampling.
            "all" (default, as before): moi="high" every other cell, moi="low"
            the control-guide cells. "ntc_cells" (moi="high" only): the cells
            carrying a control guide (non_targeting_guide_names), as moi="low"
            uses, but tested by CRT (Bernoulli resampling) instead of permutation. "matched": controls chosen by
            propensity score matching (matched_cells.select_matched_cells) from
            the guide cells (moi="high") or the control-guide cells (moi="low");
            the CRT then runs on the treated + matched cells only.
        matching_ratio: control_group="matched": controls per treated cell.
        matching_caliper: control_group="matched": max treated-control distance
            in SDs of the distance (None = no caliper); treated cells without a
            control in range are dropped.
        matching_distance: control_group="matched": match on the "propensity"
            score or its "logit".

    Returns:
        CRT_obj with the input fields filled in.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {NORMALIZATIONS}; got {normalization!r}."
        )
    if pvalue_method not in ("empirical", "skew_normal"):
        raise ValueError(f'pvalue_method must be "empirical" or "skew_normal"; got {pvalue_method!r}.')
    if moi not in ("high", "low"):
        raise ValueError(f'moi must be "high" or "low"; got {moi!r}.')
    if propensity_penalty not in (None, "l2", "old_perturbnmf"):
        raise ValueError(f'propensity_penalty must be None, "l2" or "old_perturbnmf"; got {propensity_penalty!r}.')
    if not propensity_C > 0:
        raise ValueError(f"propensity_C must be > 0; got {propensity_C}.")
    if propensity_tol is not None and not propensity_tol > 0:
        raise ValueError(f"propensity_tol must be > 0; got {propensity_tol}.")
    if propensity_max_iter is not None and propensity_max_iter < 1:
        raise ValueError(f"propensity_max_iter must be >= 1; got {propensity_max_iter}.")
    if moi == "low" and not non_targeting_guide_names:
        raise ValueError('moi="low" needs non_targeting_guide_names (e.g. ["non-targeting"]).')
    if control_group not in ("all", "ntc_cells", "matched"):
        raise ValueError(f'control_group must be "all", "ntc_cells" or "matched"; got {control_group!r}.')
    if control_group == "ntc_cells" and moi == "low":
        raise ValueError('control_group="ntc_cells" is for moi="high"; moi="low" controls already are the control-guide cells.')
    if control_group == "ntc_cells" and not non_targeting_guide_names:
        raise ValueError('control_group="ntc_cells" needs non_targeting_guide_names (e.g. ["non-targeting"]).')
    if matching_ratio < 1:
        raise ValueError(f"matching_ratio must be >= 1; got {matching_ratio}.")
    if matching_caliper is not None and not matching_caliper > 0:
        raise ValueError(f"matching_caliper must be > 0; got {matching_caliper}.")
    if matching_distance not in ("propensity", "logit"):
        raise ValueError(f'matching_distance must be "propensity" or "logit"; got {matching_distance!r}.')
    if number_guide is not None and number_guide < 1:
        raise ValueError(f"number_guide must be >= 1; got {number_guide}.")
    if resampled_rounds < 1 or extra_resampled_rounds < 0 or stage3_resampled_rounds < 0:
        raise ValueError(
            "resampled_rounds must be >= 1; extra_resampled_rounds and stage3_resampled_rounds >= 0."
        )

    # get adata    
    if usage_key not in mdata.mod:
        raise KeyError(f"Modality `{usage_key}` not in mdata.mod ({list(mdata.mod)}).")
    adata = mdata.mod[usage_key]

    # get covariate matrix
    covariate_table = build_covariate_table(adata, covariates, log_covariates)
    Covariates, covariate_names = get_covar_matrix(
        covariate_table,
        add_intercept=add_intercept,
        standardize=standardize,
        numeric_as_category_threshold=numeric_as_category_threshold,
    )

    # get program usage matrix (modality .X)
    Usage = adata.X
    if sp.issparse(Usage):
        Usage = Usage.toarray()

    # keep only the selected programs (default: all), then normalize
    program_names = [str(k + 1) for k in range(Usage.shape[1])]   # "1".."K", as cNMF program labels / Stage 3 program_name
    if selected_program_names is None:
        selected_program_names = program_names
    else:
        selected_program_names = [str(p) for p in selected_program_names]
        missing = [p for p in selected_program_names if p not in program_names]
        if missing:
            raise KeyError(f"Programs not found: {missing}; expected names like '1' (1-based).")
        program_idx = [program_names.index(p) for p in selected_program_names]
        Usage = Usage[:, program_idx]

    if normalization == "clr":
        Usage_norm = clr_from_usage(Usage=Usage, eps=eps)
    else:
        Usage_norm = row_normalize_usage(Usage)

    # get guide assignment matrix
    Guide_assignment_raw = get_from_mudata(mdata, guide_assignment_key, mod=usage_key)
    Guide_assignment, guide_names = to_csc_matrix(Guide_assignment_raw)
    if guide_names is None:
        if guide_names_key in adata.uns:
            guide_names = list(adata.uns[guide_names_key])
        else:
            raise ValueError(
                "Guide column names are required; provide them via a DataFrame "
                f"or adata.uns['{guide_names_key}']."
            )

    # get guide to gene pairs
    if guide_targets_key not in adata.uns:
        raise KeyError(f"uns['{guide_targets_key}'] not found in modality `{usage_key}`.")
    guide2gene = dict(zip(guide_names, adata.uns[guide_targets_key]))
    guide_targets = build_gene_to_guide(guide_names, guide2gene)

    if not guide_targets:
        raise ValueError("No guides mapped to genes; check guide2gene and guide names.")

    if Covariates.shape[0] != Usage_norm.shape[0] or Covariates.shape[0] != Guide_assignment.shape[0]:
        raise ValueError(
            f"Cell counts mismatch: Covariates {Covariates.shape[0]}, Usage_norm {Usage_norm.shape[0]}, "
            f"Guide_assignment {Guide_assignment.shape[0]}"
        )

    # control cells: any cell carrying a control guide (needed for moi="low")
    non_targeting_guide_names = list(non_targeting_guide_names) if non_targeting_guide_names else None
    control_obs_idx = None
    if non_targeting_guide_names:
        missing = [g for g in non_targeting_guide_names if g not in guide_targets]
        if missing:
            raise KeyError(f"Control genes not found in uns['{guide_targets_key}']: {missing}")
        control_cols = [j for g in non_targeting_guide_names for j in guide_targets[g]]
        control_obs_idx = np.flatnonzero(
            np.asarray(Guide_assignment[:, control_cols].sum(axis=1)).ravel() > 0
        ).astype(np.int32)
        if (moi == "low" or control_group == "ntc_cells") and control_obs_idx.size == 0:
            raise ValueError(f"No cells carry a guide for control genes {non_targeting_guide_names}.")

    # cells with at least one guide (control_group="matched" pool, moi="high")
    guide_cell_obs_idx = np.unique(Guide_assignment.indices).astype(np.int32)

    # genes to test (default: every gene with guides, minus control genes)
    if selected_gene_names is None:
        selected_gene_names = sorted(g for g in guide_targets if g not in (non_targeting_guide_names or []))
    else:
        selected_gene_names = list(selected_gene_names)
        missing = [g for g in selected_gene_names if g not in guide_targets]
        if missing:
            raise KeyError(f"Genes not found in uns['{guide_targets_key}']: {missing}")

    # Precompute regression pieces:
    #   CtC: p x p, Covariates^T Covariates
    #   CtC_inv:  p x p, inverse of CtC
    #   CtU_norm: p x K, Covariates^T Usage_norm
    CtC = Covariates.T @ Covariates
    CtC_inv = np.linalg.inv(CtC)
    CtU_norm = Covariates.T @ Usage_norm

    # strata are only used by the low-MOI permutation test; continuous covariates
    # cannot be stratified on, so the shuffle ignores them (still in the regression)
    covariate_strata, has_continuous_covariates = None, False
    if moi == "low":
        covariate_strata, has_continuous_covariates = covariate_strata_from_design(Covariates)
        if has_continuous_covariates:
            warnings.warn(
                'moi="low": continuous covariates are not used to form permutation strata, '
                "so the permutation null is approximate."
            )

    return CRT_obj(
        Covariates=Covariates,
        Usage_norm=Usage_norm.astype(np.float64, copy=False),
        CtC_inv=CtC_inv.astype(np.float64, copy=False),
        CtU_norm=CtU_norm.astype(np.float64, copy=False),
        Guide_assignment=Guide_assignment,
        guide_names=list(guide_names),
        guide2gene=guide2gene,
        guide_targets=guide_targets,
        covariate_names=covariate_names,
        covariate_strata=covariate_strata,
        has_continuous_covariates=has_continuous_covariates,
        has_intercept=add_intercept,
        normalization=normalization,
        selected_gene_names=selected_gene_names,
        selected_program_names=selected_program_names,
        moi=moi,
        non_targeting_guide_names=non_targeting_guide_names,
        number_guide=number_guide,
        control_obs_idx=control_obs_idx,
        control_group=control_group,
        guide_cell_obs_idx=guide_cell_obs_idx,
        matching_ratio=matching_ratio,
        matching_caliper=matching_caliper,
        matching_distance=matching_distance,
        resampled_rounds=resampled_rounds,
        extra_resampled_rounds=extra_resampled_rounds,
        extra_rounds_p_threshold=extra_rounds_p_threshold,
        stage3_resampled_rounds=stage3_resampled_rounds,
        pvalue_method=pvalue_method,
        propensity_penalty=propensity_penalty,
        propensity_C=propensity_C,
        propensity_tol=propensity_tol,
        propensity_max_iter=propensity_max_iter,
        seed=seed,
    )
