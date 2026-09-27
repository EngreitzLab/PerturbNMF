"""
Unit tests for CRT null resampling (bernoulli vs fixed_count), the skew-normal p-value
clip, and the cell-count-matched NTC null. Synthetic data only — no inference output.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage2_Evaluation/test_crt_resampling.py -v

Test strategy
  Dimensions:
    resampling:        bernoulli, fixed_count
    covariate design:  categorical only (lane), categorical + continuous (lane + depth)
    target size:       few cells (2-10), moderate (9-80)
    skew-normal side:  observed value near the fitted median (two-sided p near 1)
  Behavior verified:
    (a) fixed_count null sets keep the treated count in every stratum, draw without
        replacement, are uniform within strata (categorical) or propensity-weighted
        (continuous); bernoulli null sets do not keep the count
    (b) under a heavy-tailed null with 2-10-cell targets, fixed_count p-values are
        ~uniform and bernoulli p-values are inflated
    (c) beta_obs does not depend on resampling and equals the OLS coefficient
    plus: pipeline wiring and invalid-option error, two-sided p <= 1, matched NTC
    pseudo-targets have exactly the target's cell count
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Load the vendored CRT package under its own name. test_crt.py puts
# Slurm_version/CRT/ (which holds a CRT.py script) first on sys.path, and that script
# would shadow a plain `import CRT`.
CRT_PKG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "src",
    "Stage2_Evaluation", "B_Calibration", "src", "CRT"))
spec = importlib.util.spec_from_file_location(
    "crt_calibration", os.path.join(CRT_PKG_DIR, "__init__.py"),
    submodule_search_locations=[CRT_PKG_DIR])
crt_pkg = importlib.util.module_from_spec(spec)
sys.modules["crt_calibration"] = crt_pkg
spec.loader.exec_module(crt_pkg)

from crt_calibration.adata_utils import clr_from_usage, covariate_strata_from_design  # noqa: E402
from crt_calibration.crt import crt_betas_for_gene  # noqa: E402
from crt_calibration.ntc_groups import (  # noqa: E402
    crt_pvals_for_matched_ntc_pseudotargets,
    make_ntc_pseudotargets_matched_by_cell_count,
)
from crt_calibration.pipeline import CRTInputs, run_all_genes_union_crt  # noqa: E402
from crt_calibration.pipeline_helpers import (  # noqa: E402
    _draw_null_treated_sets,
    _skew_calibrated_crt,
)
from crt_calibration.propensity import fit_propensity_logistic  # noqa: E402
from crt_calibration.skew_normal import (  # noqa: E402
    compute_empirical_p_value,
    fit_and_evaluate_skew_normal,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def lane_design(lane, continuous=None):
    """Intercept + standardized one-hot lane dummies (drop first), as get_covar_matrix builds."""
    n_lanes = int(lane.max()) + 1
    cols = [(lane == level).astype(float) for level in range(1, n_lanes)]
    if continuous is not None:
        cols.append(np.asarray(continuous, dtype=float))
    X = np.column_stack(cols)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return np.column_stack([np.ones(lane.size), X])


def sparse_usage_clr(rng, n_cells, n_programs):
    """CLR of cNMF-like usage: each program is exactly zero in a small share of cells
    (0.1%-30%), and CRT.py floors zeros at 1e-8, so those cells sit at CLR ~ -18 —
    the heavy tail that dominates beta for few-cell targets."""
    zero_rate = np.geomspace(0.001, 0.3, n_programs)
    used = rng.random((n_cells, n_programs)) >= zero_rate
    U = np.where(used, rng.lognormal(0.0, 1.0, (n_cells, n_programs)), 0.0)
    U[U.sum(axis=1) == 0, 0] = 1.0
    U = np.maximum(U, 1e-8)
    U /= U.sum(axis=1, keepdims=True)
    return clr_from_usage(U)


def make_inputs(C, Y, G=None, guide_names=None, guide2gene=None, gene_to_cols=None):
    strata, has_continuous = covariate_strata_from_design(C)
    if G is None:
        G = sp.csc_matrix((C.shape[0], 1))
        guide_names, guide2gene, gene_to_cols = ["g0"], {"g0": "gene0"}, {"gene0": [0]}
    return CRTInputs(
        C=C, Y=Y, A=np.linalg.inv(C.T @ C), CTY=C.T @ Y, G=G,
        guide_names=guide_names, guide2gene=guide2gene, gene_to_cols=gene_to_cols,
        program_names=[f"program_{k}" for k in range(Y.shape[1])],
        covariate_strata=strata, has_continuous_covariates=has_continuous,
    )


def make_guide_inputs(rng, n_cells=4000, n_targets=6, n_ntc_guides=10):
    """Lane design + guide matrix: target t has 2 guides tagging 2+t cells each; 10 NTC
    guides tag 30 cells each."""
    C = lane_design(rng.integers(0, 3, n_cells))
    Y = sparse_usage_clr(rng, n_cells, 5)
    guide_names, guide2gene, cols = [], {}, []
    for t in range(n_targets):
        for g in range(2):
            name = f"t{t}_g{g}"
            guide_names.append(name)
            guide2gene[name] = f"gene{t}"
            cols.append(rng.choice(n_cells, 2 + t, replace=False))
    for g in range(n_ntc_guides):
        name = f"ntc_g{g}"
        guide_names.append(name)
        guide2gene[name] = "non-targeting"
        cols.append(rng.choice(n_cells, 30, replace=False))
    rows = np.concatenate(cols)
    col_idx = np.concatenate([np.full(c.size, j) for j, c in enumerate(cols)])
    G = sp.csc_matrix((np.ones(rows.size), (rows, col_idx)), shape=(n_cells, len(cols)))
    G.data[:] = 1.0
    gene_to_cols = {}
    for j, name in enumerate(guide_names):
        gene_to_cols.setdefault(guide2gene[name], []).append(j)
    return make_inputs(C, Y, G, guide_names, guide2gene, gene_to_cols)


def resample_sets(indptr, indices):
    return [indices[indptr[b]:indptr[b + 1]] for b in range(indptr.size - 1)]


def union_cells(G, cols):
    return np.unique(np.concatenate([G.indices[G.indptr[j]:G.indptr[j + 1]] for j in cols]))


# ===========================================================================
# (a) fixed-count permutations preserve per-stratum treated counts
# ===========================================================================

def test_strata_are_lanes_for_categorical_design():
    lane = np.random.default_rng(0).integers(0, 5, 2000)
    strata, has_continuous = covariate_strata_from_design(lane_design(lane))
    lanes_per_stratum = pd.crosstab(strata, lane).astype(bool).sum(axis=1)
    assert not has_continuous, "lane dummies only: expected has_continuous=False"
    assert np.unique(strata).size == 5 and lanes_per_stratum.eq(1).all(), (
        f"expected strata == lanes (5); got {np.unique(strata).size} strata, "
        f"lanes per stratum {lanes_per_stratum.tolist()}")


def test_strata_ignore_continuous_column_and_flag_it():
    rng = np.random.default_rng(0)
    lane = rng.integers(0, 3, 500)
    strata, has_continuous = covariate_strata_from_design(
        lane_design(lane, continuous=rng.normal(size=500)))
    assert has_continuous, "a normal covariate column should set has_continuous=True"
    assert np.unique(strata).size == 3, (
        f"strata should come from the 3 lanes only, got {np.unique(strata).size}")


@pytest.mark.parametrize("continuous", [False, True], ids=["lane", "lane+depth"])
def test_fixed_count_keeps_stratum_counts_every_resample(continuous):
    rng = np.random.default_rng(1)
    n_cells = 3000
    lane = rng.integers(0, 6, n_cells)
    depth = rng.normal(size=n_cells) if continuous else None
    inputs = make_inputs(lane_design(lane, continuous=depth), rng.normal(size=(n_cells, 3)))
    assert inputs.has_continuous_covariates == continuous, "precondition: design type"

    obs_idx = np.sort(rng.choice(n_cells, 17, replace=False)).astype(np.int32)
    indptr, indices = _draw_null_treated_sets(
        inputs, obs_idx, 200, seed=7, propensity_model=fit_propensity_logistic,
        resampling="fixed_count")

    expected = np.bincount(inputs.covariate_strata[obs_idx], minlength=6)
    for b, cells in enumerate(resample_sets(indptr, indices)):
        got = np.bincount(inputs.covariate_strata[cells], minlength=6)
        assert np.array_equal(got, expected), (
            f"resample {b}: per-stratum counts {got.tolist()} != observed {expected.tolist()}")
        assert np.unique(cells).size == cells.size, f"resample {b} repeats a cell"


def test_bernoulli_null_sizes_vary():
    """Contrast for (a): the bernoulli sampler does not hold the treated count."""
    rng = np.random.default_rng(2)
    n_cells = 3000
    inputs = make_inputs(lane_design(rng.integers(0, 4, n_cells)), rng.normal(size=(n_cells, 3)))
    obs_idx = np.sort(rng.choice(n_cells, 5, replace=False)).astype(np.int32)
    indptr, _ = _draw_null_treated_sets(
        inputs, obs_idx, 500, seed=3, propensity_model=fit_propensity_logistic,
        resampling="bernoulli")
    sizes = np.diff(indptr)
    assert sizes.min() < 5 < sizes.max(), (
        f"expected bernoulli null sizes on both sides of n=5, got {sizes.min()}..{sizes.max()}")


def test_fixed_count_uniform_within_stratum():
    n_cells = 200
    lane = np.repeat([0, 1], 100)
    inputs = make_inputs(lane_design(lane), np.random.default_rng(3).normal(size=(n_cells, 2)))
    obs_idx = np.array([0, 1, 2, 150], dtype=np.int32)  # 3 cells in lane 0, 1 in lane 1
    B = 20000
    _, indices = _draw_null_treated_sets(
        inputs, obs_idx, B, seed=11, propensity_model=fit_propensity_logistic,
        resampling="fixed_count")
    freq = np.bincount(indices, minlength=n_cells) / B
    assert np.abs(freq[:100] - 0.03).max() < 0.006, (
        f"lane 0 inclusion should be 3/100 per cell, got range {freq[:100].min():.4f}-{freq[:100].max():.4f}")
    assert np.abs(freq[100:] - 0.01).max() < 0.004, (
        f"lane 1 inclusion should be 1/100 per cell, got range {freq[100:].min():.4f}-{freq[100:].max():.4f}")


def test_fixed_count_continuous_draw_follows_propensity():
    rng = np.random.default_rng(4)
    n_cells = 4000
    lane = rng.integers(0, 2, n_cells)
    depth = rng.normal(size=n_cells)
    inputs = make_inputs(lane_design(lane, continuous=depth), rng.normal(size=(n_cells, 2)))
    weights = np.exp(1.5 * depth)  # treated cells enriched for high depth
    obs_idx = np.sort(rng.choice(n_cells, 80, replace=False,
                                 p=weights / weights.sum())).astype(np.int32)
    _, indices = _draw_null_treated_sets(
        inputs, obs_idx, 2000, seed=5, propensity_model=fit_propensity_logistic,
        resampling="fixed_count")
    r = np.corrcoef(np.bincount(indices, minlength=n_cells), depth)[0, 1]
    assert r > 0.5, f"null inclusion should track depth-driven propensity; corr = {r:.2f}"
    assert abs(depth[indices].mean() - depth[obs_idx].mean()) < 0.25, (
        f"null mean depth {depth[indices].mean():.2f} should be near observed "
        f"{depth[obs_idx].mean():.2f}")


# ===========================================================================
# (b) calibration under a simulated null with heavy-tailed outcomes, n = 2-10
# ===========================================================================
#
# Null targets (random cells, independent of Y) with 2-10 cells, lane-only design, CLR
# usage with zero-usage cells at ~ -18. At this cell count (N = 100k) the default
# propensity fit stops early (lbfgs tol=1e-4 on a mean-scaled loss) with sum(p) well
# above n, so bernoulli null sets are larger than the observed set, dilute its extreme
# cells, and the p-values come out too small. fixed_count holds n exactly.

SMALL_TARGET_N_CELLS = 100_000
SMALL_TARGET_COUNT = 90
SMALL_TARGET_B = 999


@pytest.fixture(scope="module")
def small_target_null_pvals():
    """resampling -> ((tests x [raw, skew]) p-values, mean null-set size / n per target)."""
    rng = np.random.default_rng(0)
    lane = rng.integers(0, 8, SMALL_TARGET_N_CELLS)
    inputs = make_inputs(lane_design(lane), sparse_usage_clr(rng, SMALL_TARGET_N_CELLS, 12))
    out = {}
    for resampling in ("bernoulli", "fixed_count"):
        pvals, sizes = [], []
        for t in range(SMALL_TARGET_COUNT):
            n = 2 + t % 9
            obs_idx = np.sort(np.random.default_rng(100 + t).choice(
                SMALL_TARGET_N_CELLS, n, replace=False)).astype(np.int32)
            indptr, indices = _draw_null_treated_sets(
                inputs, obs_idx, SMALL_TARGET_B, seed=t,
                propensity_model=fit_propensity_logistic, resampling=resampling)
            pvals_sn, _, _, pvals_raw = _skew_calibrated_crt(
                inputs, indptr, indices, obs_idx, SMALL_TARGET_B, 0)
            pvals.append(np.column_stack([pvals_raw, pvals_sn]))
            sizes.append(np.diff(indptr).mean() / n)
        out[resampling] = (np.vstack(pvals), np.array(sizes))
    return out


@pytest.mark.parametrize("column", [0, 1], ids=["raw", "skew_normal"])
def test_fixed_count_null_pvals_uniform(small_target_null_pvals, column):
    pvals, sizes = small_target_null_pvals["fixed_count"]
    assert np.allclose(sizes, 1.0), f"precondition: null sets of size n, got {sizes.min()}..{sizes.max()}"
    p = pvals[:, column]
    for alpha in (0.05, 0.1, 0.25, 0.5):
        share = np.mean(p <= alpha)
        assert 0.6 * alpha < share < 1.5 * alpha, (
            f"fixed_count P(p <= {alpha}) = {share:.3f} over {p.size} null tests; "
            f"expected within [0.6, 1.5] x {alpha}")


@pytest.mark.parametrize("column", [0, 1], ids=["raw", "skew_normal"])
def test_bernoulli_null_pvals_inflated(small_target_null_pvals, column):
    pvals, sizes = small_target_null_pvals["bernoulli"]
    assert sizes.mean() > 1.5, (
        f"mechanism check: expected bernoulli null sets > 1.5 n at N=100k (under-converged "
        f"propensity), got {sizes.mean():.2f} n. If sklearn's lbfgs now converges, this "
        f"test's premise changed — re-derive it rather than loosening the bound.")
    p = pvals[:, column]
    assert np.mean(p <= 0.05) > 0.10 and np.mean(p <= 0.01) > 0.03, (
        f"expected bernoulli null inflation; got P(p<=0.05)={np.mean(p <= 0.05):.3f}, "
        f"P(p<=0.01)={np.mean(p <= 0.01):.3f}")


# ===========================================================================
# (c) beta_obs is unchanged by the resampling choice (and equals OLS)
# ===========================================================================

def test_beta_obs_independent_of_resampling_and_equals_ols():
    rng = np.random.default_rng(5)
    n_cells = 5000
    C = lane_design(rng.integers(0, 4, n_cells), continuous=rng.normal(size=n_cells))
    Y = sparse_usage_clr(rng, n_cells, 6)
    inputs = make_inputs(C, Y)
    obs_idx = np.sort(rng.choice(n_cells, 9, replace=False)).astype(np.int32)

    betas = {}
    for resampling in ("bernoulli", "fixed_count"):
        indptr, indices = _draw_null_treated_sets(
            inputs, obs_idx, 50, seed=1, propensity_model=fit_propensity_logistic,
            resampling=resampling)
        betas[resampling], _ = crt_betas_for_gene(
            indptr, indices, inputs.C, inputs.Y, inputs.A, inputs.CTY, obs_idx, 50)
    assert np.array_equal(betas["bernoulli"], betas["fixed_count"]), (
        f"beta_obs differs by resampling: {betas}")

    x = np.zeros(n_cells)
    x[obs_idx] = 1.0
    coef, *_ = np.linalg.lstsq(np.column_stack([x, C]), Y, rcond=None)
    assert np.allclose(betas["fixed_count"], coef[0], rtol=1e-8, atol=1e-8), (
        f"summary-stat beta {betas['fixed_count']} != lstsq {coef[0]}")


def test_fixed_count_null_sets_share_observed_design_sums():
    """Categorical-only: every null set has the observed covariate sums, so the
    permuted beta moves only through the summed outcome (the prototype's algebra)."""
    rng = np.random.default_rng(6)
    n_cells = 3000
    C = lane_design(rng.integers(0, 5, n_cells))
    inputs = make_inputs(C, rng.normal(size=(n_cells, 2)))
    obs_idx = np.sort(rng.choice(n_cells, 12, replace=False)).astype(np.int32)
    indptr, indices = _draw_null_treated_sets(
        inputs, obs_idx, 100, seed=2, propensity_model=fit_propensity_logistic,
        resampling="fixed_count")
    v_obs = C[obs_idx].sum(axis=0)
    for b, cells in enumerate(resample_sets(indptr, indices)):
        assert np.allclose(C[cells].sum(axis=0), v_obs, atol=1e-9), (
            f"resample {b}: design sums {C[cells].sum(axis=0)} != observed {v_obs}")


# ===========================================================================
# Pipeline wiring, skew-normal clip, matched NTC null
# ===========================================================================

def test_run_all_genes_fixed_count_matches_bernoulli_shape_and_betas():
    inputs = make_guide_inputs(np.random.default_rng(7))
    outs = {
        resampling: run_all_genes_union_crt(
            inputs, B=99, n_jobs=1, calibrate_skew_normal=True,
            return_raw_pvals=True, return_skew_normal=True, resampling=resampling)
        for resampling in ("bernoulli", "fixed_count")
    }
    for resampling, out in outs.items():
        p = out["pvals_df"].to_numpy()
        assert out["pvals_df"].shape == (7, 5), f"{resampling}: shape {out['pvals_df'].shape}"
        assert p.max() <= 1.0 and out["pvals_raw_df"].to_numpy().min() >= 1.0 / 100, (
            f"{resampling}: p outside [1/(B+1), 1]")
    pd.testing.assert_frame_equal(outs["bernoulli"]["betas_df"], outs["fixed_count"]["betas_df"])


def test_invalid_resampling_rejected():
    inputs = make_guide_inputs(np.random.default_rng(8))
    with pytest.raises(ValueError, match="resampling"):
        run_all_genes_union_crt(inputs, B=9, n_jobs=1, resampling="poisson")


def test_two_sided_skew_normal_p_clipped_at_one():
    rng = np.random.default_rng(9)
    null = rng.gamma(2.0, 1.0, 5000)
    null = (null - null.mean()) / null.std()
    # observed values around the sample median of a skewed null, where doubling the
    # fitted tail exceeds 1 (up to ~1.04 on this input before the clip)
    max_p = max(fit_and_evaluate_skew_normal(z_obs, null, 0)[3]
                for z_obs in np.linspace(-0.4, 0.2, 61))
    assert max_p == 1.0, f"expected two-sided skew-normal p clipped to exactly 1, got max {max_p}"


def test_two_sided_empirical_p_at_most_one():
    p = compute_empirical_p_value(np.arange(-5, 6, dtype=float), 0.0, 0)
    assert p == 1.0, f"two-sided empirical p at the null median should clip to 1, got {p}"


def test_matched_pseudotargets_have_target_cell_counts():
    inputs = make_guide_inputs(np.random.default_rng(10))
    pseudo_cells, pseudo_guides = make_ntc_pseudotargets_matched_by_cell_count(
        inputs, ntc_label="non-targeting", seed=0)
    ntc_cells = set(union_cells(inputs.G, inputs.gene_to_cols["non-targeting"]))
    assert set(pseudo_cells) == {f"gene{t}" for t in range(6)}, (
        f"expected one pseudo-target per real target, got {sorted(pseudo_cells)}")
    for target, cells in pseudo_cells.items():
        n_real = union_cells(inputs.G, inputs.gene_to_cols[target]).size
        assert cells.size == n_real, f"{target}: {cells.size} pseudo cells != {n_real} target cells"
        assert set(cells) <= ntc_cells, f"{target}: pseudo-target uses non-NTC cells"
        assert all(g.startswith("ntc_") for g in pseudo_guides[target]), (
            f"{target}: guides {pseudo_guides[target]}")


def test_matched_null_scores_every_pseudotarget():
    inputs = make_guide_inputs(np.random.default_rng(11))
    pseudo_cells, _ = make_ntc_pseudotargets_matched_by_cell_count(
        inputs, ntc_label="non-targeting", seed=0)
    skew, raw, n_cells = crt_pvals_for_matched_ntc_pseudotargets(
        inputs, pseudo_cells, B=99, seed0=1, resampling="fixed_count")
    assert skew.shape == raw.shape == (6, 5), f"shapes {skew.shape}, {raw.shape}"
    assert (skew.to_numpy() <= 1.0).all() and (raw.to_numpy() >= 1.0 / 100).all(), (
        "p-values outside [1/(B+1), 1]")
    assert n_cells.to_dict() == {t: c.size for t, c in pseudo_cells.items()}, "n_cells mismatch"
