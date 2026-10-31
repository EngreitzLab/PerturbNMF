"""
Null treated sets for the CRT. Both samplers return (indptr, indices): round b's
treated cells are indices[indptr[b]:indptr[b+1]].

_bernoulli_rounds:   moi="high", Bernoulli(p_i) per cell per round (sceptre's sampler).
_permutation_rounds: moi="low", fixed treated count within each covariate stratum.
"""

from typing import Tuple

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _bernoulli_rounds(p, n_rounds, seed):
    """
    Bernoulli(p_i) treated sets for n_rounds rounds, drawn as in sceptre's
    crt_index_sampler_fast: cell j is treated in M_j ~ Binomial(n_rounds, p_j)
    rounds, chosen without replacement. Same distribution as one coin flip per
    cell per round, with ~N + n_rounds * sum(p) random draws instead of N * n_rounds.
    """
    np.random.seed(seed)
    n_cells = p.shape[0]
    M = np.empty(n_cells, dtype=np.int64)
    total = 0
    for j in range(n_cells):
        M[j] = np.random.binomial(n_rounds, p[j])
        total += M[j]

    # which rounds each cell is treated in: M[j] distinct rounds by a partial
    # Fisher-Yates shuffle of a round array that is kept between cells. Shuffling
    # any permutation of 0..n_rounds-1 gives a uniform subset, so it never needs a
    # reset, and each cell costs O(M[j]) instead of O(n_rounds) (np.random.choice
    # with replace=False shuffles all n_rounds values for every cell).
    rounds = np.arange(n_rounds)
    cell_rounds = np.empty(total, dtype=np.int64)
    counts = np.zeros(n_rounds, dtype=np.int64)
    pos = 0
    for j in range(n_cells):
        for t in range(M[j]):
            r = t + np.random.randint(0, n_rounds - t)
            tmp = rounds[t]
            rounds[t] = rounds[r]
            rounds[r] = tmp
            cell_rounds[pos + t] = rounds[t]
            counts[rounds[t]] += 1
        pos += M[j]

    # regroup by round
    indptr = np.zeros(n_rounds + 1, dtype=np.int64)
    for b in range(n_rounds):
        indptr[b + 1] = indptr[b] + counts[b]
    indices = np.empty(total, dtype=np.int32)
    write = indptr[:-1].copy()
    pos = 0
    for j in range(n_cells):
        for t in range(M[j]):
            b = cell_rounds[pos + t]
            indices[write[b]] = j
            write[b] += 1
        pos += M[j]
    return indptr, indices


@nb.njit(cache=True)
def _permutation_rounds(stratum_cells, stratum_ptr, stratum_n_treated, n_rounds, seed):
    """
    Fixed-count permutation: in every round, each stratum s gets exactly
    stratum_n_treated[s] treated cells, drawn without replacement from its cells
    stratum_cells[stratum_ptr[s]:stratum_ptr[s+1]] (partial Fisher-Yates).
    """
    np.random.seed(seed)
    n_strata = stratum_ptr.shape[0] - 1
    n_treated = 0
    for s in range(n_strata):
        n_treated += stratum_n_treated[s]

    indptr = np.empty(n_rounds + 1, dtype=np.int64)
    indices = np.empty(n_rounds * n_treated, dtype=np.int32)
    work = stratum_cells.copy()
    pos = 0
    for b in range(n_rounds):
        indptr[b] = pos
        for s in range(n_strata):
            lo = stratum_ptr[s]
            n_s = stratum_ptr[s + 1] - lo
            for t in range(stratum_n_treated[s]):
                r = lo + t + np.random.randint(0, n_s - t)
                tmp = work[lo + t]
                work[lo + t] = work[r]
                work[r] = tmp
                indices[pos] = work[lo + t]
                pos += 1
    indptr[n_rounds] = pos
    return indptr, indices


def _strata_layout(
    strata: np.ndarray, treated_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells grouped by stratum (positions), stratum offsets, and treated count per stratum."""
    _, strata = np.unique(strata, return_inverse=True)
    order = np.argsort(strata, kind="stable").astype(np.int32)
    stratum_ptr = np.concatenate([[0], np.cumsum(np.bincount(strata))]).astype(np.int64)
    stratum_n_treated = np.bincount(
        strata, weights=treated_mask, minlength=stratum_ptr.size - 1
    ).astype(np.int64)
    return order, stratum_ptr, stratum_n_treated
