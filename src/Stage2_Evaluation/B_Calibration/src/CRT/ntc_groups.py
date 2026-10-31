"""
NTC guide-group construction matched to real genes' guide frequencies.

Copied from PerturbNMF src/Stage2_Evaluation/B_Calibration/src/CRT/ntc_groups.py
(commit e6f1e65). The grouping functions are unchanged; changes:
- build_gene_to_cols comes from .prepare.build_gene_to_guide (same function);
- the PerturbNMF CRT p-value functions (crt_pvals_for_guide_set*,
  crt_pvals_for_ntc_groups_ensemble*) are dropped: Tools/CRT runs
  conditional_randomized_test on the fake genes instead
  (null_guide_assignment_preparation.py, method="frequency");
- build_ntc_group_inputs reads a CRT_obj (Guide_assignment, guide_names,
  guide2gene) instead of PerturbNMF's CRT inputs.

Steps: guide frequency = column sum of the guide assignment / n_cells;
frequency bins = quantiles of the real guides' frequencies; each real gene's
signature = bins of its group_size lowest-frequency guides; each NTC group copies
a random real gene's signature with NTC guides from the same bins, with no NTC
guide reused within one replicate (ensemble member).
"""

import logging
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import scipy.sparse as sp

from .prepare import build_gene_to_guide as build_gene_to_cols

logger = logging.getLogger(__name__)


def guide_frequency(G: sp.spmatrix, guide_names: Sequence[str]) -> Dict[str, float]:
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
) -> Dict[str, List[str]]:
    """
    Returns dict group_id -> list of NTC guide names (no overlap within replicate).
    """
    rng = np.random.default_rng(seed)

    bin_to_guides: Dict[int, List[str]] = {}
    for guide in ntc_guides:
        bin_id = guide_to_bin.get(guide)
        if bin_id is None:
            continue
        bin_to_guides.setdefault(bin_id, []).append(guide)

    for guides in bin_to_guides.values():
        rng.shuffle(guides)

    groups: Dict[str, List[str]] = {}
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
) -> List[Dict[str, List[str]]]:
    """
    Returns list of group dicts, one per ensemble replicate.
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
        )
        groups_ens.append(groups)
    return groups_ens


def _validate_group_sizes(
    groups: Mapping[
        str, Sequence[str]
    ],  # CHANGED: dropped `expected_size` param — size now follows --number_guide, not a hardcoded value
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


def build_ntc_group_inputs(
    inputs,
    ntc_label: Union[str, Iterable[str]] = "NTC",
    group_size: int = 6,
    n_bins: int = 20,
) -> Tuple[List[str], Dict[str, float], Dict[str, int], List[List[int]]]:
    """
    Compute guide frequency + bin signatures for NTC grouping.
    ntc_label can be a single label or an iterable of labels.
    inputs: CRT_obj (Guide_assignment, guide_names, guide2gene).
    """
    G = inputs.Guide_assignment
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
