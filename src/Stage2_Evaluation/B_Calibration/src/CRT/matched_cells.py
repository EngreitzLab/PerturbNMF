"""
Propensity score matching: pick control cells that look like a gene's treated
cells, as in PerturbNMF's Matched_cell_programDE (R MatchIt, method="nearest").

Unlike the CRT, which keeps every cell and uses the propensity score to resample
the treatment, matching uses the propensity score once, to choose the cells.
The effect is then fit on the matched cells only.

Matching is greedy nearest neighbour without replacement (MatchIt defaults):
treated cells are visited in order of decreasing propensity, and each takes the
closest control that no other treated cell has taken. With ratio > 1 this runs
in rounds, as MatchIt does: every treated cell gets its 1st control, then its
2nd, and so on, so the early treated cells don't use up all the good controls.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

import numpy as np
import pandas as pd

from .dataclass import CRT_obj
from .propensity_model import propensity_model


@dataclass
class Matched_cells_result:
    gene: str                         # target gene name
    treated_obs_idx: np.ndarray       # matched treated cells, row indexes into CRT_obj (int32, sorted)
    control_obs_idx: np.ndarray       # matched control cells (int32, sorted)
    matched_obs_idx: np.ndarray       # treated + control cells (int32, sorted)
    subclass: np.ndarray              # pair id per matched_obs_idx cell (= rank of its treated cell)
    weights: np.ndarray               # MatchIt ATT weights per matched_obs_idx cell
    propensity: np.ndarray            # propensity score per matched_obs_idx cell
    n_treated: int                    # treated cells before matching
    n_control_pool: int               # candidate control cells before matching
    n_treated_unmatched: int          # treated cells left with no control (caliper)
    balance: Optional[pd.DataFrame] = None   # standardized mean differences before / after


def _treated_obs_idx(CRT_obj: CRT_obj, gene: str) -> np.ndarray:
    """
    Cells carrying any guide that targets gene (the union, as in the CRT).

    Args:
        CRT_obj: prepared inputs from prepare_crt_inputs.
        gene: target gene name (key of CRT_obj.guide_targets).

    Returns:
        int32 sorted row indexes of the treated cells.
    """
    if gene not in CRT_obj.guide_targets:
        raise KeyError(f"'{gene}' has no guides in CRT_obj.guide_targets.")
    G = CRT_obj.Guide_assignment                # CSC with explicit zeros removed
    chunks = [G.indices[G.indptr[j] : G.indptr[j + 1]] for j in CRT_obj.guide_targets[gene]]
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def guide_cell_obs_idx(CRT_obj: CRT_obj) -> np.ndarray:
    """
    Cells carrying at least one guide (of any gene, control guides included).
    Uses CRT_obj.guide_cell_obs_idx when prepare_crt_inputs stored it.

    Returns:
        int32 sorted row indexes.
    """
    idx = getattr(CRT_obj, "guide_cell_obs_idx", None)    # getattr: CRT_obj pickled before this field existed
    if idx is None:
        idx = np.unique(CRT_obj.Guide_assignment.indices)  # CSC, explicit zeros removed
    return np.asarray(idx, dtype=np.int32)


def _control_pool_idx(CRT_obj: CRT_obj, treated_obs_idx: np.ndarray, guide_cells_only: bool) -> np.ndarray:
    """
    Candidate control cells: cells without the gene's guides.

    moi="low": the control-guide cells (CRT_obj.control_obs_idx).
    moi="high": every other cell, or (guide_cells_only) every other cell that
    carries at least one guide, as the R pipeline does (has_any_guide).

    Args:
        CRT_obj: prepared inputs from prepare_crt_inputs.
        treated_obs_idx: treated cells of this gene.
        guide_cells_only: drop cells with no guide (moi="high" only).

    Returns:
        int32 sorted row indexes of the candidate controls.
    """
    if CRT_obj.moi == "low":
        pool = np.asarray(CRT_obj.control_obs_idx)
    elif guide_cells_only:
        pool = guide_cell_obs_idx(CRT_obj)
    else:
        pool = np.arange(CRT_obj.Covariates.shape[0])
    return np.setdiff1d(pool, treated_obs_idx).astype(np.int32)


def _find(parent: np.ndarray, i: int) -> int:
    """
    Root of i in a union-find array with path compression. Used to jump to the
    nearest control that is still available on one side.
    """
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def _greedy_nearest_neighbor(
    treated_distance: np.ndarray,
    control_distance: np.ndarray,
    ratio: int,
    caliper: Optional[float],
    treated_order: np.ndarray,
    treated_X: Optional[np.ndarray] = None,
    control_X: Optional[np.ndarray] = None,
    X_calipers: Optional[np.ndarray] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Greedy 1:ratio nearest neighbour matching on a 1-D distance, without replacement.

    Controls are sorted by distance. Two union-find arrays give, for any sorted
    position, the nearest still-available control to its left and to its right,
    so each match costs about O(log n) instead of a scan over all controls.
    The search steps outward from the treated cell's position and takes the
    first control that passes the covariate calipers.

    Args:
        treated_distance: distance (propensity or logit) per treated cell.
        control_distance: distance per control cell.
        ratio: controls per treated cell.
        caliper: max |treated - control| distance, in distance units (None = no limit).
        treated_order: order in which treated cells pick controls.
        treated_X, control_X: covariate columns with calipers (n x c), or None.
        X_calipers: max |treated - control| per covariate column (length c).
        seed: seed for breaking ties between equally distant controls at random
            (without covariates every cell has the same propensity, and row
            order would always pick the first cells).

    Returns:
        match: n_treated x ratio array of control indexes (into control_distance),
            -1 where no control was found.
    """
    n_treated, n_control = treated_distance.size, control_distance.size
    shuffle = np.random.default_rng(seed).permutation(n_control)       # random order among ties
    sort_idx = shuffle[np.argsort(control_distance[shuffle], kind="stable")]
    sorted_d = control_distance[sort_idx]
    # parent arrays hold positions in sorted_d plus a sentinel at each end
    # (left: position 0 is "none"; right: position n_control + 1 is "none")
    left = np.arange(n_control + 2)
    right = np.arange(n_control + 2)
    max_gap = np.inf if caliper is None else caliper
    match = np.full((n_treated, ratio), -1, dtype=np.int64)

    def ok(t: int, c_sorted: int) -> bool:
        if X_calipers is None:
            return True
        c = sort_idx[c_sorted]
        return bool(np.all(np.abs(treated_X[t] - control_X[c]) <= X_calipers))

    for r in range(ratio):
        for t in treated_order:
            if r > 0 and match[t, r - 1] < 0:
                continue                          # no earlier control: none will be found now
            d = treated_distance[t]
            pos = int(np.searchsorted(sorted_d, d))       # sorted_d[pos-1] < d <= sorted_d[pos]
            lo = _find(left, pos) - 1                     # nearest available at or left of pos-1
            hi = _find(right, pos + 1) - 1                # nearest available at or right of pos
            chosen = -1
            while lo >= 0 or hi < n_control:
                gap_lo = d - sorted_d[lo] if lo >= 0 else np.inf
                gap_hi = sorted_d[hi] - d if hi < n_control else np.inf
                if min(gap_lo, gap_hi) > max_gap:
                    break
                if gap_lo <= gap_hi:
                    if ok(t, lo):
                        chosen = lo
                        break
                    lo = _find(left, lo) - 1               # skip lo for this treated cell only
                else:
                    if ok(t, hi):
                        chosen = hi
                        break
                    hi = _find(right, hi + 2) - 1
            if chosen < 0:
                continue
            match[t, r] = sort_idx[chosen]
            left[chosen + 1] = chosen                     # remove: left jumps past it
            right[chosen + 1] = chosen + 2                # remove: right jumps past it
    return match


def standardized_mean_differences(
    Covariates: np.ndarray,
    treatment: np.ndarray,
    covariate_names: Optional[list] = None,
    weights: Optional[np.ndarray] = None,
    sd_reference: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Standardized mean difference (treated - control) per covariate.

    Args:
        Covariates: n x c covariate columns (no intercept).
        treatment: 0/1 indicator (length n).
        covariate_names: column names.
        weights: per-cell weights (None = equal).
        sd_reference: denominator per column; None = SD of the treated cells
            (MatchIt's default for the ATT), computed from these cells.

    Returns:
        Series of SMDs indexed by covariate name.
    """
    treatment = np.asarray(treatment).astype(bool)
    w = np.ones(treatment.size) if weights is None else np.asarray(weights, dtype=np.float64)
    mean_t = np.average(Covariates[treatment], axis=0, weights=w[treatment])
    mean_c = np.average(Covariates[~treatment], axis=0, weights=w[~treatment])
    sd = Covariates[treatment].std(axis=0) if sd_reference is None else sd_reference
    sd = np.where(sd > 0, sd, 1.0)
    return pd.Series((mean_t - mean_c) / sd, index=covariate_names)


def _att_weights(subclass_of_control: np.ndarray, n_control: int) -> np.ndarray:
    """
    MatchIt ATT weights for matched controls: 1 / (controls in the pair),
    rescaled so they sum to the number of matched controls.
    """
    _, inverse, counts = np.unique(subclass_of_control, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    return w * (n_control / w.sum())


def select_matched_cells(
    gene: str,
    CRT_obj: CRT_obj,
    ratio: int = 4,
    caliper: Optional[float] = None,
    covariate_calipers: Optional[Dict[str, float]] = None,
    distance: Literal["propensity", "logit"] = "propensity",
    order: Literal["largest", "smallest", "random", "data"] = "largest",
    guide_cells_only: bool = True,
    min_cells: int = 0,
    seed: Optional[int] = None,
    propensity_model: Callable = propensity_model,
) -> Matched_cells_result:
    """
    Select control cells matched to a gene's treated cells by propensity score.

    Steps:
        1. Treated cells = cells carrying any of the gene's guides; candidate
           controls = the other cells (see _control_pool_idx).
        2. Fit P(treated | Covariates) on treated + candidate cells with
           propensity_model (CRT_obj's penalty / tol / max_iter settings).
        3. Greedy 1:ratio nearest neighbour matching on that score, without
           replacement (_greedy_nearest_neighbor).
        4. Pair ids, MatchIt ATT weights, and covariate balance before / after.

    The R pipeline (MatchIt defaults) is distance="propensity", order="largest",
    guide_cells_only=True, ratio=4. Austin (2011) recommends distance="logit"
    with caliper=0.2.

    Args:
        gene: target gene name (key of CRT_obj.guide_targets).
        CRT_obj: prepared inputs from prepare_crt_inputs.
        ratio: controls per treated cell.
        caliper: max distance between a treated cell and its control, in SDs
            of the distance over treated + candidate cells (MatchIt std.caliper).
            None = no limit. Treated cells with no control in range are dropped.
        covariate_calipers: {covariate name: max difference in SDs of that
            covariate}, names from CRT_obj.covariate_names (after one-hot
            encoding, e.g. "batch_2"). Covariates are already z-scored when
            CRT_obj was built with standardize=True.
        distance: match on the propensity score or on its logit.
        order: order in which treated cells pick controls: "largest" /
            "smallest" propensity first, "random", or "data" (row order).
        guide_cells_only: moi="high" only; drop candidate controls with no guide.
        min_cells: raise if fewer treated cells than this.
        seed: seed for order="random" and for breaking distance ties; None = CRT_obj.seed.
        propensity_model: propensity fit, called like the CRT's.

    Returns:
        Matched_cells_result. Row indexes refer to CRT_obj.Covariates /
        CRT_obj.Usage_norm, so CRT_obj.Usage_norm[result.matched_obs_idx] are
        the outcomes of the matched cells.

    Raises:
        ValueError: too few treated cells, or more controls needed than exist.
    """
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1; got {ratio}.")
    if distance not in ("propensity", "logit"):
        raise ValueError(f'distance must be "propensity" or "logit"; got {distance!r}.')
    if order not in ("largest", "smallest", "random", "data"):
        raise ValueError(f'order must be "largest", "smallest", "random" or "data"; got {order!r}.')

    # 1. treated cells and candidate controls
    treated_obs_idx = _treated_obs_idx(CRT_obj, gene)
    pool_obs_idx = _control_pool_idx(CRT_obj, treated_obs_idx, guide_cells_only)
    n_treated, n_pool = treated_obs_idx.size, pool_obs_idx.size
    if n_treated == 0 or n_treated < min_cells:
        raise ValueError(f"'{gene}' has {n_treated} treated cells (min_cells = {min_cells}).")
    if n_pool < n_treated * ratio:
        raise ValueError(
            f"'{gene}': {n_pool} candidate controls for {n_treated} treated cells x ratio {ratio}; "
            f"lower ratio or set guide_cells_only=False."
        )

    # 2. propensity score on treated + candidate cells
    analysis_idx = np.concatenate([treated_obs_idx, pool_obs_idx])
    Covariates = np.ascontiguousarray(CRT_obj.Covariates[analysis_idx])
    treatment = np.r_[np.ones(n_treated, dtype=np.int8), np.zeros(n_pool, dtype=np.int8)]
    p, _ = propensity_model(Covariates, treatment, has_intercept=CRT_obj.has_intercept,
                            penalty=CRT_obj.propensity_penalty, C=CRT_obj.propensity_C,
                            tol=getattr(CRT_obj, "propensity_tol", None),
                            max_iter=getattr(CRT_obj, "propensity_max_iter", None))
    if distance == "logit":
        p_clip = np.clip(p, 1e-12, 1 - 1e-12)
        d = np.log(p_clip / (1 - p_clip))
    else:
        d = p
    caliper_abs = None if caliper is None else caliper * d.std()

    # covariate calipers, in absolute units of the design columns
    X_cols = Covariates[:, int(CRT_obj.has_intercept):]
    names = list(CRT_obj.covariate_names or [f"covariate_{j}" for j in range(X_cols.shape[1])])
    treated_X = control_X = X_calipers = None
    if covariate_calipers:
        missing = set(covariate_calipers) - set(names)
        if missing:
            raise KeyError(f"covariate_calipers not in CRT_obj.covariate_names: {sorted(missing)}")
        cols = [names.index(name) for name in covariate_calipers]
        X_sub = X_cols[:, cols]
        X_calipers = np.array(list(covariate_calipers.values())) * X_sub.std(axis=0)
        treated_X, control_X = X_sub[:n_treated], X_sub[n_treated:]

    # 3. matching
    d_treated = d[:n_treated]
    rng = np.random.default_rng(CRT_obj.seed if seed is None else seed)
    if order == "largest":
        treated_order = np.argsort(-d_treated, kind="stable")
    elif order == "smallest":
        treated_order = np.argsort(d_treated, kind="stable")
    elif order == "random":
        treated_order = rng.permutation(n_treated)
    else:
        treated_order = np.arange(n_treated)
    match = _greedy_nearest_neighbor(d_treated, d[n_treated:], ratio, caliper_abs, treated_order,
                                     treated_X, control_X, X_calipers, seed=int(rng.integers(2**32)))

    # 4. pairs, weights, balance
    has_match = match[:, 0] >= 0
    t_local = np.flatnonzero(has_match)                       # treated cells kept
    subclass_t = np.empty(n_treated, dtype=np.int64)
    subclass_t[treated_order] = np.arange(n_treated)          # pair id = matching rank
    c_t, c_r = np.nonzero(match >= 0)
    c_local = match[c_t, c_r]                                 # matched controls (into the pool)
    subclass_c = subclass_t[c_t]

    local = np.concatenate([t_local, n_treated + c_local])
    subclass = np.concatenate([subclass_t[t_local], subclass_c])
    weights = np.concatenate([np.ones(t_local.size), _att_weights(subclass_c, c_local.size)])
    order_out = np.argsort(analysis_idx[local], kind="stable")
    local, subclass, weights = local[order_out], subclass[order_out], weights[order_out]

    sd_treated = X_cols[:n_treated].std(axis=0)               # same denominator before / after
    balance = pd.DataFrame({
        "smd_before": standardized_mean_differences(X_cols, treatment, names, sd_reference=sd_treated),
        "smd_after": standardized_mean_differences(X_cols[local], treatment[local], names,
                                                    weights=weights, sd_reference=sd_treated),
    })
    balance.loc["propensity"] = [
        standardized_mean_differences(d[:, None], treatment, ["d"], sd_reference=d_treated.std()).iloc[0],
        standardized_mean_differences(d[local][:, None], treatment[local], ["d"], weights=weights,
                                      sd_reference=d_treated.std()).iloc[0],
    ]

    return Matched_cells_result(
        gene=gene,
        treated_obs_idx=np.sort(treated_obs_idx[t_local]).astype(np.int32),
        control_obs_idx=np.sort(pool_obs_idx[c_local]).astype(np.int32),
        matched_obs_idx=analysis_idx[local].astype(np.int32),
        subclass=subclass,
        weights=weights,
        propensity=p[local],
        n_treated=n_treated,
        n_control_pool=n_pool,
        n_treated_unmatched=int(n_treated - t_local.size),
        balance=balance,
    )
