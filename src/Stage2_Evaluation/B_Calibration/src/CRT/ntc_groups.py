"""
NTC guide-group construction and evaluation for QQ diagnostics.
"""

import logging
import zlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from .adata_utils import build_gene_to_cols, union_obs_idx_from_cols
from .pipeline_helpers import (
    _check_resampling,
    _draw_null_treated_sets,
    _empirical_crt,
    _skew_calibrated_crt,
)
from .propensity import fit_propensity_logistic

logger = logging.getLogger(__name__)


def guide_frequency(
    G: sp.spmatrix, guide_names: Sequence[str]
) -> Dict[str, float]:
    """
    Returns prevalence per guide: freq[g] = mean(G[:,g] > 0) across cells.
    """
    if sp.issparse(G):
        G = G.tocsr()
        counts = np.asarray(G.sum(axis=0)).ravel()
    else:
        G = np.asarray(G)
        counts = G.sum(axis=0)
    n_cells = float(G.shape[0])
    freqs = counts / n_cells
    return {g: float(f) for g, f in zip(guide_names, freqs)}


def _normalize_ntc_labels(ntc_label: Union[str, Iterable[str]]) -> Set[str]:
    if isinstance(ntc_label, str):
        return {ntc_label}
    return {str(label) for label in ntc_label}


def _split_guides_by_label(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
    ntc_labels: Set[str],
) -> Tuple[List[str], List[str]]:
    ntc_guides: List[str] = []
    real_guides: List[str] = []
    for guide in guide_names:
        gene = guide2gene.get(guide)
        if gene is None:
            continue
        if gene in ntc_labels:
            ntc_guides.append(guide)
        else:
            real_guides.append(guide)
    return ntc_guides, real_guides


def _guide_bins_from_real_freqs(
    guide_freq: Mapping[str, float],
    real_guides: Sequence[str],
    n_bins: int,
) -> Tuple[Dict[str, int], np.ndarray]:
    real_freqs = np.array([guide_freq[g] for g in real_guides], dtype=np.float64)
    if real_freqs.size == 0:
        raise ValueError("No real guides available to build frequency bins.")
    edges = np.quantile(real_freqs, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    guide_to_bin: Dict[str, int] = {}
    for guide, freq in guide_freq.items():
        bin_id = int(np.digitize(freq, edges[1:-1], right=True))
        guide_to_bin[guide] = bin_id
    return guide_to_bin, edges


def _real_gene_bin_signatures(
    guide_names: Sequence[str],
    guide2gene: Mapping[str, str],
    guide_freq: Mapping[str, float],
    guide_to_bin: Mapping[str, int],
    group_size: int,
    ntc_labels: Set[str],
) -> List[List[int]]:
    gene_to_cols = build_gene_to_cols(list(guide_names), guide2gene)
    sigs: List[List[int]] = []
    for gene, cols in gene_to_cols.items():
        if gene in ntc_labels:
            continue
        guides = [guide_names[i] for i in cols]
        if len(guides) < group_size:
            continue
        guides = sorted(guides, key=lambda g: guide_freq[g])
        guides = guides[:group_size]
        sigs.append([guide_to_bin[g] for g in guides])
    if not sigs:
        raise ValueError("No real-gene bin signatures available.")
    return sigs


def make_ntc_groups_matched_by_freq(
    ntc_guides: Sequence[str],
    ntc_freq: Mapping[str, float],
    real_gene_bin_sigs: Sequence[Sequence[int]],
    guide_to_bin: Mapping[str, int],
    group_size: int = 6,
    seed: int = 0,
    max_groups: Optional[int] = None,
    drop_remainder: bool = True,
    max_attempts: int = 100,
    replace: bool = False,
) -> Dict[str, List[str]]:
    """
    Returns dict group_id -> list of NTC guide names.

    replace=False (default): guides are consumed as groups are built, so no guide is
    shared between groups within a replicate.

    replace=True: guides are returned to the pool after each group, so a guide can
    appear in several groups (never twice in one group), and identical groups are
    rejected. This mirrors how SCEPTRE builds its negative-control ("undercover")
    gRNA groups: each group is an independent without-replacement draw of
    calibration_group_size NTC gRNAs, rejection-sampled so that no two groups are
    the same set (sample_combinations_v2 in src/negative_control_functions.cpp,
    https://github.com/Katsevich-Lab/sceptre). Without replacement, the NTC pool is
    used up after about n_ntc / group_size groups, and the per-bin matching shrinks
    that further: with ~200 NTC guides and 15 guides per target, only 3-6 groups per
    replicate could be formed, too few for a stable null. If max_groups is None,
    replace=True builds one group per real-gene signature (i.e. per real target).
    """
    rng = np.random.default_rng(seed)

    bin_to_guides: Dict[int, List[str]] = {}
    for guide in ntc_guides:
        bin_id = guide_to_bin.get(guide)
        if bin_id is None:
            continue
        bin_to_guides.setdefault(bin_id, []).append(guide)

    if not replace:
        for guides in bin_to_guides.values():
            rng.shuffle(guides)

    # With replacement the pool never runs out, so the loop needs an explicit cap
    if replace and max_groups is None:
        max_groups = len(real_gene_bin_sigs)

    groups: Dict[str, List[str]] = {}
    seen_groups: Set[Tuple[str, ...]] = set()
    attempts = 0
    group_idx = 0

    while True:
        if max_groups is not None and group_idx >= max_groups:
            break
        if not real_gene_bin_sigs:
            break

        sig = list(real_gene_bin_sigs[rng.integers(0, len(real_gene_bin_sigs))])
        if len(sig) != group_size:
            attempts += 1
            if attempts >= max_attempts and drop_remainder:
                break
            continue

        counts: Dict[int, int] = {}
        for bin_id in sig:
            counts[bin_id] = counts.get(bin_id, 0) + 1

        feasible = True
        for bin_id, need in counts.items():
            available = len(bin_to_guides.get(bin_id, []))
            if available < need:
                feasible = False
                break

        if not feasible:
            attempts += 1
            if attempts >= max_attempts and drop_remainder:
                break
            continue

        selected: List[str] = []
        if replace:
            for bin_id, need in counts.items():
                pool = bin_to_guides[bin_id]
                selected.extend(rng.choice(pool, size=need, replace=False).tolist())

            # Reject a group identical to one already drawn (as SCEPTRE does)
            key = tuple(sorted(selected))
            if key in seen_groups:
                attempts += 1
                if attempts >= max_attempts and drop_remainder:
                    break
                continue
            seen_groups.add(key)
        else:
            for bin_id, need in counts.items():
                pool = bin_to_guides[bin_id]
                selected.extend(pool[:need])
                del pool[:need]

        groups[f"ntc_{group_idx}"] = selected
        group_idx += 1
        attempts = 0

    return groups


def make_ntc_groups_ensemble(
    ntc_guides: Sequence[str],
    ntc_freq: Mapping[str, float],
    real_gene_bin_sigs: Sequence[Sequence[int]],
    guide_to_bin: Mapping[str, int],
    n_ensemble: int,
    seed0: int,
    group_size: int = 6,
    max_groups: Optional[int] = None,
    drop_remainder: bool = True,
    replace: bool = False,
) -> List[Dict[str, List[str]]]:
    """
    Returns list of group dicts, one per ensemble replicate.
    See make_ntc_groups_matched_by_freq for replace.
    """
    groups_ens: List[Dict[str, List[str]]] = []
    for e in range(n_ensemble):
        groups = make_ntc_groups_matched_by_freq(
            ntc_guides=ntc_guides,
            ntc_freq=ntc_freq,
            real_gene_bin_sigs=real_gene_bin_sigs,
            guide_to_bin=guide_to_bin,
            group_size=group_size,
            seed=seed0 + e,
            max_groups=max_groups,
            drop_remainder=drop_remainder,
            replace=replace,
        )
        groups_ens.append(groups)
    return groups_ens


def _validate_group_sizes(
    groups: Mapping[str, Sequence[str]],  # CHANGED: dropped `expected_size` param — size now follows --number_guide, not a hardcoded value
) -> Dict[str, float]:
    sizes = np.array([len(guides) for guides in groups.values()], dtype=np.int32)
    if sizes.size == 0:
        raise ValueError("No NTC groups provided for size validation.")
    stats = {
        "n_groups": int(sizes.size),
        "min": int(np.min(sizes)),
        "median": float(np.median(sizes)),
        "max": int(np.max(sizes)),
    }
    logger.info(
        "NTC group sizes: n=%d min=%d median=%.1f max=%d",
        stats["n_groups"],
        stats["min"],
        stats["median"],
        stats["max"],
    )
    # CHANGED: was `if np.any(sizes != expected_size)` comparing to a hardcoded
    # 6. NTC groups are all built at the same group_size (= --number_guide), so we
    # don't hardcode the expected value — only require the groups to be consistent.
    if stats["min"] != stats["max"]:
        raise ValueError(
            f"NTC groups have inconsistent sizes: min={stats['min']} "
            f"max={stats['max']}; expected all groups to share one size."
        )
    return stats


def _crt_pvals_for_treated_cells(
    inputs,
    obs_idx: np.ndarray,
    B: int,
    seed: int,
    propensity_model=fit_propensity_logistic,
    resampling: str = "bernoulli",
    calibrate_skew_normal: bool = False,
    side_code: int = 0,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Score one treated cell set with the same CRT used for real targets.
    Returns (skew-normal p-values or None, raw CRT p-values) across programs.
    """
    K = inputs.Y.shape[1]
    if obs_idx.size == 0 or obs_idx.size == inputs.C.shape[0] or B <= 0:
        ones = np.ones(K, dtype=np.float64)
        return (ones if calibrate_skew_normal else None), ones

    indptr, idx = _draw_null_treated_sets(
        inputs, obs_idx, B, seed, propensity_model, resampling
    )
    if calibrate_skew_normal:
        pvals_sn, _, _, pvals_raw = _skew_calibrated_crt(
            inputs, indptr, idx, obs_idx, B, side_code
        )
        return pvals_sn, pvals_raw
    pvals, _ = _empirical_crt(inputs, indptr, idx, obs_idx, B)
    return None, pvals


def crt_pvals_for_guide_set(
    inputs,
    guide_idx: np.ndarray,
    B: int,
    seed: int,
    propensity_model=fit_propensity_logistic,
    resampling: str = "bernoulli",
) -> np.ndarray:
    """
    Returns CRT p-values across programs for one guide set.
    """
    obs_idx = union_obs_idx_from_cols(inputs.G, guide_idx)
    _, pvals = _crt_pvals_for_treated_cells(
        inputs, obs_idx, B, seed, propensity_model, resampling
    )
    return pvals


def crt_pvals_for_guide_set_skew(
    inputs,
    guide_idx: np.ndarray,
    B: int,
    seed: int,
    propensity_model=fit_propensity_logistic,
    side_code: int = 0,
    resampling: str = "bernoulli",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns skew-calibrated and raw CRT p-values across programs for one guide set.
    """
    obs_idx = union_obs_idx_from_cols(inputs.G, guide_idx)
    return _crt_pvals_for_treated_cells(
        inputs,
        obs_idx,
        B,
        seed,
        propensity_model,
        resampling,
        calibrate_skew_normal=True,
        side_code=side_code,
    )


def _crt_pvals_for_ntc_groups_ensemble_both(
    inputs,
    ntc_groups_ens: Sequence[Mapping[str, Sequence[str]]],
    B: int,
    seed0: int,
    propensity_model,
    resampling: str,
    calibrate_skew_normal: bool,
    side_code: int,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Shared loop: returns (e -> skew DataFrame, e -> raw DataFrame); the skew dict is
    empty when calibrate_skew_normal is False. Group size is inferred from the groups.
    """
    _check_resampling(resampling)
    guide_to_col = {g: i for i, g in enumerate(inputs.guide_names)}
    out_skew: Dict[int, pd.DataFrame] = {}
    out_raw: Dict[int, pd.DataFrame] = {}
    K = inputs.Y.shape[1]

    for e, groups in enumerate(ntc_groups_ens):
        _validate_group_sizes(groups)
        rows_skew: List[np.ndarray] = []
        rows_raw: List[np.ndarray] = []
        group_ids: List[str] = []
        for group_id, guides in groups.items():
            cols = [guide_to_col[g] for g in guides if g in guide_to_col]
            if not cols:
                continue
            seed = (hash((seed0, e, group_id)) & 0xFFFFFFFF)
            obs_idx = union_obs_idx_from_cols(inputs.G, np.asarray(cols, dtype=np.int32))
            pvals_skew, pvals_raw = _crt_pvals_for_treated_cells(
                inputs,
                obs_idx,
                B,
                seed,
                propensity_model,
                resampling,
                calibrate_skew_normal=calibrate_skew_normal,
                side_code=side_code,
            )
            rows_skew.append(pvals_skew)
            rows_raw.append(pvals_raw)
            group_ids.append(group_id)
        empty = np.empty((0, K), dtype=np.float64)
        out_raw[e] = pd.DataFrame(
            np.vstack(rows_raw) if rows_raw else empty,
            index=group_ids,
            columns=inputs.program_names,
        )
        if calibrate_skew_normal:
            out_skew[e] = pd.DataFrame(
                np.vstack(rows_skew) if rows_skew else empty,
                index=group_ids,
                columns=inputs.program_names,
            )
    return out_skew, out_raw


def crt_pvals_for_ntc_groups_ensemble(
    inputs,
    ntc_groups_ens: Sequence[Mapping[str, Sequence[str]]],
    B: int,
    seed0: int,
    propensity_model=fit_propensity_logistic,
    resampling: str = "bernoulli",
) -> Dict[int, pd.DataFrame]:
    """
    Returns mapping e -> DataFrame(rows=group_id, cols=programs) of raw p-values.
    """
    _, out_raw = _crt_pvals_for_ntc_groups_ensemble_both(
        inputs, ntc_groups_ens, B, seed0, propensity_model, resampling,
        calibrate_skew_normal=False, side_code=0,
    )
    return out_raw


def crt_pvals_for_ntc_groups_ensemble_skew(
    inputs,
    ntc_groups_ens: Sequence[Mapping[str, Sequence[str]]],
    B: int,
    seed0: int,
    propensity_model=fit_propensity_logistic,
    side_code: int = 0,
    resampling: str = "bernoulli",
) -> Dict[int, pd.DataFrame]:
    """
    Returns mapping e -> DataFrame(rows=group_id, cols=programs) of skew p-values.
    """
    out_skew, _ = _crt_pvals_for_ntc_groups_ensemble_both(
        inputs, ntc_groups_ens, B, seed0, propensity_model, resampling,
        calibrate_skew_normal=True, side_code=side_code,
    )
    return out_skew


def crt_pvals_for_ntc_groups_ensemble_skew_and_raw(
    inputs,
    ntc_groups_ens: Sequence[Mapping[str, Sequence[str]]],
    B: int,
    seed0: int,
    propensity_model=fit_propensity_logistic,
    side_code: int = 0,
    resampling: str = "bernoulli",
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    One pass over the NTC groups returning both (e -> skew p DataFrame, e -> raw p
    DataFrame). The skew-normal p-value is the one real-target calls are made on, so
    this is the null to compare against when judging those calls. The raw p-values
    are identical to crt_pvals_for_ntc_groups_ensemble with the same seeds.
    """
    return _crt_pvals_for_ntc_groups_ensemble_both(
        inputs, ntc_groups_ens, B, seed0, propensity_model, resampling,
        calibrate_skew_normal=True, side_code=side_code,
    )


def make_ntc_pseudotargets_matched_by_cell_count(
    inputs,
    ntc_label: Union[str, Iterable[str]] = "NTC",
    targets: Optional[Iterable[str]] = None,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Build one negative-control pseudo-target per real target with EXACTLY the target's
    treated-cell count (capped at the number of NTC cells), keeping guide structure:
    whole NTC guides are added in random order until the next guide would overshoot,
    and the remainder is a random subset of that next guide's cells.

    The frequency-matched guide groups above use a fixed number of guides, so their
    cell counts sit well above the few-cell regime where calibration is hardest; this
    null samples every target's cell count, including 2-10 cells.

    targets: real targets to match; default = every gene in inputs.gene_to_cols that
        is not an NTC label
    Returns:
        pseudo_cells: target -> sorted int32 array of NTC cell indices
        pseudo_guides: target -> NTC guide names used
    """
    ntc_labels = _normalize_ntc_labels(ntc_label)
    ntc_cols = [
        j for gene, cols in inputs.gene_to_cols.items() if gene in ntc_labels for j in cols
    ]
    if not ntc_cols:
        raise ValueError(f"No guides found for NTC label(s) {sorted(ntc_labels)}.")
    ntc_cols = np.asarray(sorted(ntc_cols), dtype=np.int64)
    n_ntc_cells = union_obs_idx_from_cols(inputs.G, ntc_cols).size
    if targets is None:
        targets = sorted(g for g in inputs.gene_to_cols if g not in ntc_labels)

    G = inputs.G
    rng = np.random.default_rng(seed)
    pseudo_cells: Dict[str, np.ndarray] = {}
    pseudo_guides: Dict[str, List[str]] = {}
    for target in targets:
        n_cells = min(
            union_obs_idx_from_cols(G, inputs.gene_to_cols[target]).size, n_ntc_cells
        )
        cells = np.empty(0, dtype=np.int32)
        chosen: List[str] = []
        for col in rng.permutation(ntc_cols):
            guide_cells = np.setdiff1d(G.indices[G.indptr[col] : G.indptr[col + 1]], cells)
            if guide_cells.size == 0:
                continue
            need = n_cells - cells.size
            if guide_cells.size > need:
                guide_cells = rng.choice(guide_cells, need, replace=False)
            cells = np.union1d(cells, guide_cells)
            chosen.append(inputs.guide_names[col])
            if cells.size == n_cells:
                break
        pseudo_cells[target] = cells.astype(np.int32)
        pseudo_guides[target] = chosen
    return pseudo_cells, pseudo_guides


def crt_pvals_for_matched_ntc_pseudotargets(
    inputs,
    pseudo_cells: Mapping[str, np.ndarray],
    B: int,
    seed0: int,
    propensity_model=fit_propensity_logistic,
    side_code: int = 0,
    resampling: str = "bernoulli",
    n_jobs: int = 1,
    backend: str = "loky",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Score cell-count-matched NTC pseudo-targets with the same CRT as real targets.
    Returns (skew p DataFrame, raw p DataFrame, n_cells Series), rows = matched target.
    """
    _check_resampling(resampling)
    targets = list(pseudo_cells.keys())
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_crt_pvals_for_treated_cells)(
            inputs,
            np.asarray(pseudo_cells[t], dtype=np.int32),
            B,
            (seed0 + zlib.crc32(t.encode())) & 0xFFFFFFFF,
            propensity_model,
            resampling,
            calibrate_skew_normal=True,
            side_code=side_code,
        )
        for t in targets
    )
    K = inputs.Y.shape[1]
    empty = np.empty((0, K), dtype=np.float64)
    skew = np.vstack([r[0] for r in results]) if results else empty
    raw = np.vstack([r[1] for r in results]) if results else empty
    n_cells = pd.Series(
        [len(pseudo_cells[t]) for t in targets], index=targets, name="n_cells"
    )
    return (
        pd.DataFrame(skew, index=targets, columns=inputs.program_names),
        pd.DataFrame(raw, index=targets, columns=inputs.program_names),
        n_cells,
    )


def build_ntc_group_inputs(
    inputs,
    ntc_label: Union[str, Iterable[str]] = "NTC",
    group_size: int = 6,
    n_bins: int = 20,
) -> Tuple[List[str], Dict[str, float], Dict[str, int], List[List[int]]]:
    """
    Compute guide frequency + bin signatures for NTC grouping.
    ntc_label can be a single label or an iterable of labels.
    """
    G = inputs.G
    guide_names = inputs.guide_names
    guide2gene = inputs.guide2gene
    ntc_labels = _normalize_ntc_labels(ntc_label)

    guide_freq = guide_frequency(G, guide_names)
    ntc_guides, real_guides = _split_guides_by_label(
        guide_names, guide2gene, ntc_labels
    )
    guide_to_bin, _ = _guide_bins_from_real_freqs(
        guide_freq, real_guides, n_bins=n_bins
    )
    real_gene_bin_sigs = _real_gene_bin_signatures(
        guide_names,
        guide2gene,
        guide_freq,
        guide_to_bin,
        group_size=group_size,
        ntc_labels=ntc_labels,
    )
    return ntc_guides, guide_freq, guide_to_bin, real_gene_bin_sigs
