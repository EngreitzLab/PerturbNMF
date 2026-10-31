"""
Build null ("fake") target genes out of non-targeting guides for CRT calibration.

method="random" follows sceptre's run_calibration_check
(construct_negative_control_pairs_v2 / sample_combinations_v2):
- each fake gene gets k non-targeting guides drawn without replacement, where
  k = round(median guides per tested gene), capped at the number of NTC guides;
- no two fake genes share the same guide set (sorted-set rejection sampling);
  a guide can still appear in several fake genes;
- if C(n_NTC, k) is no more than the number requested, every combination is used.

method="frequency" follows PerturbNMF's CRT calibration (ntc_groups.py, copied
to this folder): each fake gene copies the guide-frequency-bin signature of a
random real gene, filled with NTC guides from the same bins, and no NTC guide is
reused within one replicate (a partition of the NTC guides). Replicates (seed,
seed + 1, ...) are added until n_null_genes distinct guide sets are reached, or a
fixed n_ensemble replicates are used as in PerturbNMF (n_ensemble=10).

Differences from sceptre (both methods), because here every gene is tested against all K
programs (sceptre tests individual gRNA-group x response-gene pairs):
- n_null_genes defaults to the number of tested real genes, so fake and real
  genes give the same number of tests (sceptre: n_calibration_pairs =
  n_discovery_pairs). No x5 oversampling or random response pairing is needed.
- no pairwise QC (sceptre: n_nonzero_trt / n_nonzero_cntrl >= 7), since real
  genes get no treated-cell QC in this CRT either.

A fake gene is just a new CRT_obj.guide_targets entry ("null_17" -> NTC guide
columns); Guide_assignment, guide_names and guide2gene are unchanged. The
returned CRT_obj tests only the fake genes and runs through
conditional_randomized_test as is. With moi="low", a fake gene's treated cells
are compared with the remaining non-targeting cells, as in sceptre's low-MOI
calibration check. Fake-gene p-values are a calibration diagnostic (QQ plot,
false-hit count); they are not used to adjust the real p-values.
"""

import dataclasses
import itertools
import math
import warnings
from typing import List, Literal, Optional, Sequence

import numpy as np

from .dataclass import CRT_obj
from .ntc_groups import build_ntc_group_inputs, make_ntc_groups_matched_by_freq


def null_group_size(CRT_obj: CRT_obj, ntc_cols: Sequence[int]) -> int:
    """
    sceptre's default calibration group size: round(median guides per tested
    gene), capped at the number of non-targeting guides.

    Args:
        CRT_obj: prepared inputs from prepare_crt_inputs.
        ntc_cols: non-targeting guide column indexes.

    Returns:
        group size k (>= 1).
    """
    n_guides = [len(CRT_obj.guide_targets[g]) for g in CRT_obj.selected_gene_names]
    if not n_guides:
        raise ValueError("No tested genes; cannot compute the median guides per gene.")
    median_size = int(
        np.round(np.median(n_guides))
    )  # numpy rounds .5 to even, like R's round()
    return max(1, min(median_size, len(ntc_cols)))


def sample_null_guide_groups(
    ntc_cols: Sequence[int],
    group_size: int,
    n_groups: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Draw n_groups distinct guide sets of size group_size from ntc_cols.

    Each set is drawn without replacement and sorted; repeated sets are rejected
    (sceptre's sample_combinations_v2). If there are no more than n_groups
    possible sets, every combination is returned instead
    (sceptre's iterate_over_combinations), with a warning if that is fewer than
    requested.

    Args:
        ntc_cols: non-targeting guide column indexes.
        group_size: guides per fake gene.
        n_groups: number of fake genes requested.
        rng: numpy random generator.

    Returns:
        list of sorted guide-column lists, one per fake gene.
    """
    ntc_cols = np.asarray(sorted(ntc_cols), dtype=np.int64)
    n_possible = math.comb(ntc_cols.size, group_size)

    if n_possible <= n_groups:
        if n_possible < n_groups:
            warnings.warn(
                f"Only {n_possible} distinct sets of {group_size} non-targeting guides exist "
                f"({ntc_cols.size} guides); using all of them instead of {n_groups}."
            )
        return [list(map(int, c)) for c in itertools.combinations(ntc_cols, group_size)]

    seen = set()
    groups: List[List[int]] = []
    while len(groups) < n_groups:
        draw = tuple(
            sorted(int(j) for j in rng.choice(ntc_cols, size=group_size, replace=False))
        )
        if draw not in seen:
            seen.add(draw)
            groups.append(list(draw))
    return groups


def frequency_matched_null_guide_groups(
    CRT_obj: CRT_obj,
    group_size: int,
    n_null_genes: Optional[int],
    n_bins: int = 10,
    n_ensemble: Optional[int] = None,
    seed: int = 0,
    max_ensemble: int = 1000,
) -> List[List[int]]:
    """
    Guide-frequency-matched NTC guide sets (PerturbNMF ntc_groups.py).

    Each replicate partitions the NTC guides into groups of group_size that copy
    random real genes' frequency-bin signatures (make_ntc_groups_matched_by_freq,
    seed + e for replicate e). Repeated guide sets across replicates are dropped.

    Args:
        CRT_obj: prepared inputs; needs non_targeting_guide_names.
        group_size: guides per fake gene; only real genes with at least this many
            guides contribute signatures.
        n_null_genes: number of fake genes to collect (replicates are added until
            reached, then trimmed); ignored when n_ensemble is given.
        n_bins: number of guide-frequency quantile bins of the real guides
            (PerturbNMF's CRT.py uses 10).
        n_ensemble: fixed number of replicates (PerturbNMF: 10); None collects
            n_null_genes groups.
        seed: seed of replicate 0.
        max_ensemble: stop after this many replicates when collecting n_null_genes.

    Returns:
        list of sorted guide-column lists, one per fake gene.
    """
    ntc_guides, guide_freq, guide_to_bin, real_sigs = build_ntc_group_inputs(
        CRT_obj,
        ntc_label=CRT_obj.non_targeting_guide_names,
        group_size=group_size,
        n_bins=n_bins,
    )
    guide_to_col = {g: j for j, g in enumerate(CRT_obj.guide_names)}

    seen = set()
    groups: List[List[int]] = []
    n_rounds = n_ensemble if n_ensemble is not None else max_ensemble
    for e in range(n_rounds):
        replicate = make_ntc_groups_matched_by_freq(
            ntc_guides=ntc_guides,
            ntc_freq=guide_freq,
            real_gene_bin_sigs=real_sigs,
            guide_to_bin=guide_to_bin,
            group_size=group_size,
            seed=seed + e,
        )
        if not replicate:
            raise ValueError(
                f"No frequency-matched NTC group of {group_size} guides could be built "
                f"({len(ntc_guides)} NTC guides, {n_bins} bins); lower group_size or n_bins."
            )
        for guides in replicate.values():
            cols = tuple(sorted(guide_to_col[g] for g in guides))
            if cols not in seen:
                seen.add(cols)
                groups.append(list(cols))
        if n_ensemble is None and len(groups) >= n_null_genes:
            return groups[:n_null_genes]

    if n_ensemble is None:
        warnings.warn(
            f"Only {len(groups)} distinct frequency-matched NTC groups after {max_ensemble} "
            f"replicates; using them instead of {n_null_genes}."
        )
    return groups


def prepare_null_guide_assignment(
    CRT_obj: CRT_obj,
    group_size: Optional[int] = None,
    n_null_genes: Optional[int] = None,
    method: Literal["random", "frequency"] = "random",
    seed: Optional[int] = None,
    null_prefix: str = "null_",
    n_bins: int = 10,
    n_ensemble: Optional[int] = None,
) -> CRT_obj:
    """
    Make a CRT_obj whose tested genes are fake genes built from non-targeting guides.

    Args:
        CRT_obj: prepared inputs from prepare_crt_inputs; needs
            non_targeting_guide_names.
        group_size: guides per fake gene; None uses CRT_obj.number_guide, and if
            that is None too, round(median guides per tested gene), capped at the
            number of NTC guides (sceptre default).
        n_null_genes: number of fake genes; None uses the number of tested real
            genes (len(selected_gene_names)).
        method: "random" (sceptre-style random guide sets) or "frequency"
            (guide-frequency-matched groups, PerturbNMF ntc_groups.py).
        seed: seed for drawing the guide sets; None uses CRT_obj.seed.
        null_prefix: fake gene names are f"{null_prefix}{i}".
        n_bins: method="frequency" only; guide-frequency quantile bins.
        n_ensemble: method="frequency" only; fixed number of replicates
            (PerturbNMF: 10) instead of collecting n_null_genes groups.

    Returns:
        A new CRT_obj (the input is not changed; large arrays are shared) with
        the fake genes added to guide_targets, selected_gene_names set to the
        fake genes, and empty gene_results.
    """
    if method not in ("random", "frequency"):
        raise ValueError(f'method must be "random" or "frequency"; got {method!r}.')
    if not CRT_obj.non_targeting_guide_names:
        raise ValueError(
            "CRT_obj.non_targeting_guide_names is empty; pass non_targeting_guide_names "
            '(e.g. ["non-targeting"]) to prepare_crt_inputs.'
        )

    # non-targeting guide columns (union over all control labels)
    ntc_cols = sorted(
        {
            j
            for label in CRT_obj.non_targeting_guide_names
            for j in CRT_obj.guide_targets[label]
        }
    )
    if not ntc_cols:
        raise ValueError(
            f"No guides found for control genes {CRT_obj.non_targeting_guide_names}."
        )

    if group_size is None:
        group_size = CRT_obj.number_guide
    if group_size is None:
        group_size = null_group_size(CRT_obj, ntc_cols)
    if not 1 <= group_size <= len(ntc_cols):
        raise ValueError(
            f"group_size must be in [1, {len(ntc_cols)}] (number of NTC guides); got {group_size}."
        )

    if n_null_genes is None:
        n_null_genes = len(CRT_obj.selected_gene_names)
    if n_null_genes < 1:
        raise ValueError(f"n_null_genes must be >= 1; got {n_null_genes}.")

    seed = CRT_obj.seed if seed is None else seed
    if method == "random":
        groups = sample_null_guide_groups(
            ntc_cols, group_size, n_null_genes, np.random.default_rng(seed)
        )
    else:
        groups = frequency_matched_null_guide_groups(
            CRT_obj,
            group_size,
            n_null_genes,
            n_bins=n_bins,
            n_ensemble=n_ensemble,
            seed=seed,
        )

    null_names = [f"{null_prefix}{i}" for i in range(len(groups))]
    clash = [g for g in null_names if g in CRT_obj.guide_targets]
    if clash:
        raise ValueError(
            f"Fake gene names already exist in guide_targets: {clash[:5]}; change null_prefix."
        )

    return dataclasses.replace(
        CRT_obj,
        guide_targets={**CRT_obj.guide_targets, **dict(zip(null_names, groups))},
        selected_gene_names=null_names,
        gene_results={},
    )
