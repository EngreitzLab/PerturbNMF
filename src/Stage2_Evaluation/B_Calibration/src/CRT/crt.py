"""
Conditional randomization / permutation test of each target gene against every program.

moi="high" (sceptre's high-MOI default): CRT. A per-gene propensity model gives
p_i = P(cell i treated | covariates); null treated sets are drawn Bernoulli(p_i)
per cell per round. Controls are all other cells.

moi="low" (sceptre's low-MOI default): permutation test. The regression uses
only treated + control-guide cells; null treated sets shuffle the treated label
within covariate strata, keeping each stratum's treated count fixed.

CRT_obj.control_group changes which control cells enter each gene's test:
"all" (the above), "ntc_cells" (moi="high": cells carrying a control guide), or
"matched" (propensity-matched controls; see _analysis_obs_idx).

Both run in stages like sceptre: CRT_obj.resampled_rounds first, plus
CRT_obj.extra_resampled_rounds when any program's stage-1 p is below
CRT_obj.extra_rounds_p_threshold.

p-values: empirical (1 + #{|null| >= |obs|}) / (1 + rounds) by default. With
CRT_obj.pvalue_method = "skew_normal", programs flagged in stage 1 get a
two-sided p from a skew-normal fitted to all null rounds (skew_normal.py);
fits that fail sceptre's checks fall back to the empirical p, after
CRT_obj.stage3_resampled_rounds more rounds if > 0 (sceptre B3).
"""

import zlib
from typing import Callable, Optional

import numba as nb
import numpy as np
from joblib import Parallel, delayed

from .OLS import OLS_solver
from .plotting import plot_beta_null
from .dataclass import CRT_gene_result, CRT_obj
from .matched_cells import select_matched_cells
from .propensity_model import propensity_model
from .resample import _bernoulli_rounds, _permutation_rounds, _strata_layout
from .skew_normal import skew_normal_pvals



@nb.njit(cache=True)
def _null_betas(Covariates, Usage_norm, CtC_inv, CtU_norm, indptr, indices):
    """One beta row per resampling round; round b's treated cells are indices[indptr[b]:indptr[b+1]]."""
    n_rounds = indptr.shape[0] - 1
    out = np.empty((n_rounds, Usage_norm.shape[1]))
    for b in range(n_rounds):
        out[b] = OLS_solver(Covariates, Usage_norm, CtC_inv, CtU_norm, indices[indptr[b]:indptr[b + 1]])
    return out


def _empirical_pvals(beta_obs: np.ndarray, beta_null: np.ndarray) -> np.ndarray:
    """
    Two-sided resampling p per program: (1 + #{|null| >= |obs|}) / (1 + #valid rounds).
    NaN null rounds (empty or collinear draws) are left out; NaN beta_obs gives p = 1.
    """
    valid = np.isfinite(beta_null)
    exceed = valid & (np.abs(beta_null) >= np.abs(beta_obs))
    pvals = (1.0 + exceed.sum(axis=0)) / (1.0 + valid.sum(axis=0))
    pvals[~np.isfinite(beta_obs)] = 1.0
    return pvals


def _analysis_obs_idx(gene: str, CRT_obj: CRT_obj, treated_obs_idx: np.ndarray):
    """
    Cells in one gene's regression and resampling, from CRT_obj.moi and CRT_obj.control_group.

        control_group="all" (default): moi="high" uses every cell (returns None:
            use CRT_obj's precomputed CtC_inv / CtU_norm); moi="low" uses treated
            + control-guide cells.
        control_group="ntc_cells" (moi="high"): treated + cells carrying a
            control guide (CRT_obj.control_obs_idx), as moi="low".
        control_group="matched": treated cells that found a match + their
            propensity-matched controls (matched_cells.select_matched_cells; the
            pool is the guide cells for moi="high", the control-guide cells for
            moi="low"). Treated cells left unmatched by a caliper are dropped.

    Returns:
        analysis_idx: int32 sorted row indexes, or None for all cells.
        treated_obs_idx: treated cells in the analysis (fewer than given only
            for control_group="matched" with a caliper).
    """
    control_group = getattr(CRT_obj, "control_group", "all")   # getattr: CRT_obj pickled before this field existed
    if control_group == "matched":
        if treated_obs_idx.size == 0:
            return None, treated_obs_idx
        matched = select_matched_cells(
            gene, CRT_obj, ratio=CRT_obj.matching_ratio, caliper=CRT_obj.matching_caliper,
            distance=CRT_obj.matching_distance,
        )
        return matched.matched_obs_idx, matched.treated_obs_idx
    if control_group == "ntc_cells" or CRT_obj.moi == "low":
        return np.union1d(treated_obs_idx, CRT_obj.control_obs_idx).astype(np.int32), treated_obs_idx
    return None, treated_obs_idx


def _treated_obs_idx(CRT_obj: CRT_obj, gene: str) -> np.ndarray:
    """Cells carrying any of the gene's guides (union), int32 sorted."""
    G = CRT_obj.Guide_assignment                # CSC with explicit zeros removed (to_csc_matrix), so the
    guide_cols = CRT_obj.guide_targets[gene]    # stored entries of column j are exactly the cells that carry guide j
    chunks = [G.indices[G.indptr[j] : G.indptr[j + 1]] for j in guide_cols]
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def conditional_randomized_test_per_gene(
    gene: str,
    CRT_obj: CRT_obj,
    propensity_model: Callable = propensity_model,
    beta_resample_plot: bool = False,
    plot_dir: Optional[str] = None,
) -> CRT_gene_result:
    """
    Test one target gene against every selected program.

    Args:
        gene: target gene name (one of CRT_obj.selected_gene_names).
        CRT_obj: prepared inputs from prepare_crt_inputs.
        propensity_model: propensity fit used for moi="high".
        beta_resample_plot: if True, save <plot_dir>/<gene>_beta_null.pdf with the
            null beta histogram per program, beta_obs, and the skew-normal fit
            where it was used (plotting.plot_beta_null). Untestable genes get no plot.
        plot_dir: output directory for beta_resample_plot.

    Returns:
        CRT_gene_result with betas, two-sided p-values (pvalue: final; pvalue_empirical:
        empirical; pvalue_skewnorm / skew_params: skew-normal, NaN where not used),
        treated cells and control means. Genes with no treated cells (or with every
        analysed cell treated) get p = 1 and beta = 0.
    """
    # find treated cell index: cells carrying any of this gene's guides (union)
    if gene not in CRT_obj.selected_gene_names:
        raise KeyError(f"'{gene}' is not part of the selected gene list.")
    if beta_resample_plot and plot_dir is None:
        raise ValueError("beta_resample_plot=True needs plot_dir.")

    treated_obs_idx = _treated_obs_idx(CRT_obj, gene)
    n_programs = CRT_obj.Usage_norm.shape[1]

    # cells in this gene's regression (None = all cells, CRT_obj's precomputed pieces)
    analysis_idx, treated_obs_idx = _analysis_obs_idx(gene, CRT_obj, treated_obs_idx)
    n_treated = int(treated_obs_idx.size)
    if analysis_idx is not None:
        Covariates = np.ascontiguousarray(CRT_obj.Covariates[analysis_idx])
        Usage_norm = np.ascontiguousarray(CRT_obj.Usage_norm[analysis_idx])
        CtC_inv = np.linalg.pinv(Covariates.T @ Covariates)   # pinv: a stratum may be absent from the subset
        CtU_norm = Covariates.T @ Usage_norm
        treated_pos = np.searchsorted(analysis_idx, treated_obs_idx).astype(np.int32)
    else:
        Covariates, Usage_norm = CRT_obj.Covariates, CRT_obj.Usage_norm
        CtC_inv, CtU_norm = CRT_obj.CtC_inv, CRT_obj.CtU_norm
        treated_pos = treated_obs_idx
    n_analysis = Covariates.shape[0]

    # no treated cells (or every cell treated): nothing to compare, so record
    # the treated cells and return p = 1, beta = 0 for every program
    if n_treated == 0 or n_treated == n_analysis:
        return CRT_gene_result(
            gene=gene,
            n_treated=n_treated,
            treated_obs_idx=treated_obs_idx,
            pvalue=np.ones(n_programs),
            betas=np.zeros(n_programs),
        )

    treated_mask = np.zeros(n_analysis, dtype=np.int8)     # 0/1 treated indicator on the analysed cells
    treated_mask[treated_pos] = 1
    control_means = Usage_norm[treated_mask == 0].mean(axis=0)

    # compute beta observed
    beta_obs = OLS_solver(Covariates, Usage_norm, CtC_inv, CtU_norm, treated_pos)

    # null treated-set sampler for this gene
    seed = (zlib.crc32(gene.encode()) ^ CRT_obj.seed) & 0xFFFFFFFF   # per-gene seed, stable across runs (unlike hash())
    propensity_n_iter, propensity_sum_ratio = np.nan, np.nan
    if CRT_obj.moi == "low":
        strata = CRT_obj.covariate_strata[analysis_idx]
        stratum_cells, stratum_ptr, stratum_n_treated = _strata_layout(strata, treated_mask)
        def draw(n_rounds, stage_seed):
            return _permutation_rounds(stratum_cells, stratum_ptr, stratum_n_treated, n_rounds, stage_seed)
    else:
        p, model = propensity_model(Covariates, treated_mask, has_intercept=CRT_obj.has_intercept,
                                    penalty=CRT_obj.propensity_penalty, C=CRT_obj.propensity_C,
                                    # getattr: CRT_obj pickled before these fields existed
                                    tol=getattr(CRT_obj, "propensity_tol", None),
                                    max_iter=getattr(CRT_obj, "propensity_max_iter", None))
        # convergence record: n_iter == max_iter means lbfgs hit its cap; sum(p) / n_treated is 1 at the MLE
        propensity_n_iter = 0.0 if model is None else float(np.max(model.n_iter_))   # float: NaN when no fit
        propensity_sum_ratio = float(p.sum()) / n_treated
        def draw(n_rounds, stage_seed):
            return _bernoulli_rounds(p, n_rounds, stage_seed)

    # stage 1
    indptr, indices = draw(CRT_obj.resampled_rounds, seed)
    beta_null = _null_betas(Covariates, Usage_norm, CtC_inv, CtU_norm, indptr, indices)
    pvalue_empirical = _empirical_pvals(beta_obs, beta_null)
    flagged = np.flatnonzero(pvalue_empirical < CRT_obj.extra_rounds_p_threshold)    # programs that look significant

    # stage 2.A: more rounds only if some program looks significant
    if CRT_obj.extra_resampled_rounds > 0 and flagged.size:
        indptr, indices = draw(CRT_obj.extra_resampled_rounds, (seed + 1) & 0xFFFFFFFF)
        beta_null = np.vstack([beta_null, _null_betas(Covariates, Usage_norm, CtC_inv, CtU_norm, indptr, indices)])
        pvalue_empirical = _empirical_pvals(beta_obs, beta_null)

    # stage 2.B: skew-normal tail fit for the flagged programs (sceptre-style); rejected fits keep the empirical p
    pvalue, pvalue_skewnorm, skew_params = pvalue_empirical, None, None
    if CRT_obj.pvalue_method == "skew_normal":
        pvalue_skewnorm, skew_params = skew_normal_pvals(beta_obs, beta_null, flagged)

        # stage 2.B1: if any flagged fit was rejected, draw more rounds before using its empirical p (sceptre B3)
        rejected = flagged[~np.isfinite(pvalue_skewnorm[flagged])]
        if CRT_obj.stage3_resampled_rounds > 0 and rejected.size:
            indptr, indices = draw(CRT_obj.stage3_resampled_rounds, (seed + 2) & 0xFFFFFFFF)
            beta_null = np.vstack([beta_null, _null_betas(Covariates, Usage_norm, CtC_inv, CtU_norm, indptr, indices)])
            pvalue_empirical = _empirical_pvals(beta_obs, beta_null)

        pvalue = np.where(np.isfinite(pvalue_skewnorm), pvalue_skewnorm, pvalue_empirical)

    if beta_resample_plot:
        plot_beta_null(gene, beta_obs, beta_null, pvalue_empirical, plot_dir,
                       program_names=CRT_obj.selected_program_names,
                       pvalue_skewnorm=pvalue_skewnorm, skew_params=skew_params, flagged=flagged)

    return CRT_gene_result(
        gene=gene,
        n_treated=n_treated,
        treated_obs_idx=treated_obs_idx,
        pvalue=pvalue,
        betas=beta_obs,
        pvalue_skewnorm=pvalue_skewnorm,
        skew_params=skew_params,
        pvalue_empirical=pvalue_empirical,
        control_means=control_means,
        n_resampled=beta_null.shape[0],
        n_control=int(n_analysis - n_treated),
        propensity_n_iter=propensity_n_iter,
        propensity_sum_ratio=propensity_sum_ratio,
    )


def conditional_randomized_test(
    CRT_obj: CRT_obj,
    n_jobs: int = 1,
    backend: str = "loky",
    propensity_model: Callable = propensity_model,
    beta_resample_plot: bool = False,
    plot_dir: Optional[str] = None,
) -> CRT_obj:
    """
    Run conditional_randomized_test_per_gene for every gene in
    CRT_obj.selected_gene_names and store the results in CRT_obj.gene_results.

    Args:
        CRT_obj: prepared inputs from prepare_crt_inputs.
        n_jobs: parallel workers (1 runs serially). Results do not depend on
            n_jobs: each gene's seed comes from its name and CRT_obj.seed.
        backend: joblib backend.
        propensity_model: propensity fit used for moi="high".
        beta_resample_plot: if True, save one null-beta PDF per tested gene
            to plot_dir (see conditional_randomized_test_per_gene).
        plot_dir: output directory for beta_resample_plot.

    Returns:
        The same CRT_obj, with gene_results filled (gene name -> CRT_gene_result).
    """
    if beta_resample_plot and plot_dir is None:
        raise ValueError("beta_resample_plot=True needs plot_dir.")
    genes = list(CRT_obj.selected_gene_names)
    plot_kw = dict(beta_resample_plot=beta_resample_plot, plot_dir=plot_dir)
    if n_jobs == 1:
        results = [conditional_randomized_test_per_gene(g, CRT_obj, propensity_model, **plot_kw) for g in genes]
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(conditional_randomized_test_per_gene)(g, CRT_obj, propensity_model, **plot_kw) for g in genes
        )
    for result in results:
        CRT_obj.gene_results[result.gene] = result
    return CRT_obj
