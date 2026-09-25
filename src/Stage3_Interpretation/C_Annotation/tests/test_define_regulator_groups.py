"""The grouping must recover an equally coherent group whether its effects are strong or weak.

Synthetic screen: two blocks of five regulators share a true effect profile each; block S has
strong effects (r ~ 0.9 between members), block W weak ones (r ~ 0.4), plus unrelated
regulators. A single global correlation cut (GeneProgramExplorer's rule) finds S only; the
noise-corrected shared-nearest-neighbour grouping must find both.
"""
import numpy as np
from collections import Counter

from define_regulator_groups import (
    cluster, correlation, permutation_null, program_blocks, significant_edges, snn_clusters,
)

N_PROGRAMS = 120
NOISE_SD = 1.0


def synthetic_screen(seed=0):
    rng = np.random.default_rng(seed)
    strong_profile, weak_profile = rng.normal(size=N_PROGRAMS), rng.normal(size=N_PROGRAMS)
    rows = [3.0 * strong_profile + rng.normal(scale=NOISE_SD, size=N_PROGRAMS) for _ in range(5)]
    rows += [0.8 * weak_profile + rng.normal(scale=NOISE_SD, size=N_PROGRAMS) for _ in range(5)]
    rows += [rng.normal(scale=rng.uniform(0.5, 3.0)) * rng.normal(size=N_PROGRAMS)
             + rng.normal(scale=NOISE_SD, size=N_PROGRAMS) for _ in range(40)]
    x = np.array(rows)
    reliability = np.clip(1 - NOISE_SD ** 2 / x.var(axis=1), 0, 1)
    features = [f"P{i}|all" for i in range(N_PROGRAMS)]
    return x, reliability, features


def largest_shared_label(labels, idx):
    counts = Counter(labels[i] for i in idx if labels[i] >= 0)
    return counts.most_common(1)[0][1] if counts else 0


def test_noise_corrected_snn_recovers_weak_and_strong_blocks():
    x, reliability, features = synthetic_screen()
    null = permutation_null(x, program_blocks(features), n_permutations=40, seed=0)
    edges = significant_edges(correlation(x), null, fdr=0.05)
    labels = snn_clusters(x, reliability, edges, k=5, cut=0.7, floor=0.05, max_size=20)
    assert largest_shared_label(labels, range(0, 5)) >= 4, labels[:5]
    assert largest_shared_label(labels, range(5, 10)) >= 4, labels[5:10]
    # the two blocks stay apart
    assert not set(labels[0:5]) & set(labels[5:10]) - {-1}


def test_global_cut_misses_the_weak_block():
    x, _, _ = synthetic_screen()
    labels = cluster(1.0 - correlation(x), cut=0.5, max_size=20)
    assert largest_shared_label(labels, range(0, 5)) == 5
    assert largest_shared_label(labels, range(5, 10)) <= 2


def test_program_permutation_moves_whole_programs():
    features = ["P0|D0", "P1|D0", "P0|D1", "P1|D1"]
    blocks = program_blocks(features)
    assert blocks == [[0, 2], [1, 3]]
