"""
Unit tests for NTC null-group sampling with replacement (replace=True in
make_ntc_groups_matched_by_freq / make_ntc_groups_ensemble). Synthetic inputs only.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd <repo root>
    python -m pytest tests/Script/Stage2_Evaluation/test_crt_ntc_replace.py -v

Behavior verified:
    (a) replace=False (default) is unchanged: no guide shared between groups, and the
        pool runs out after about n_ntc / group_size groups
    (b) replace=True: no guide twice in one group, guides reused across groups,
        no two identical groups, each group matches a real-gene bin signature
    (c) replace=True caps at max_groups, or at one group per real-gene signature when
        max_groups is None
    (d) the ensemble wrapper passes replace through
"""

import importlib.util
import os
import sys
from collections import Counter

import pytest

# Load the vendored CRT package under its own name (see test_crt_resampling.py)
CRT_PKG_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "src",
    "Stage2_Evaluation", "B_Calibration", "src", "CRT"))
if "crt_calibration" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "crt_calibration", os.path.join(CRT_PKG_DIR, "__init__.py"),
        submodule_search_locations=[CRT_PKG_DIR])
    crt_pkg = importlib.util.module_from_spec(spec)
    sys.modules["crt_calibration"] = crt_pkg
    spec.loader.exec_module(crt_pkg)

from crt_calibration.ntc_groups import (  # noqa: E402
    make_ntc_groups_ensemble,
    make_ntc_groups_matched_by_freq,
)


# 200 NTC guides in 4 frequency bins; 15 guides per real target (the regime where
# sampling without replacement runs out of groups)
N_NTC = 200
N_BINS = 4
GROUP_SIZE = 15
N_REAL_TARGETS = 40


@pytest.fixture
def ntc_inputs():
    ntc_guides = [f"ntc_g{i}" for i in range(N_NTC)]
    guide_to_bin = {g: i % N_BINS for i, g in enumerate(ntc_guides)}
    ntc_freq = {g: 0.01 for g in ntc_guides}
    # Real-gene signatures: sorted bin ids, one per real target
    real_sigs = [
        sorted((t + j) % N_BINS for j in range(GROUP_SIZE))
        for t in range(N_REAL_TARGETS)
    ]
    return ntc_guides, ntc_freq, real_sigs, guide_to_bin


def _make(ntc_inputs, **kwargs):
    ntc_guides, ntc_freq, real_sigs, guide_to_bin = ntc_inputs
    return make_ntc_groups_matched_by_freq(
        ntc_guides=ntc_guides, ntc_freq=ntc_freq, real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin, group_size=GROUP_SIZE, seed=7, **kwargs)


def test_default_consumes_guides(ntc_inputs):
    groups = _make(ntc_inputs)
    all_guides = [g for grp in groups.values() for g in grp]
    assert len(all_guides) == len(set(all_guides))
    assert len(groups) <= N_NTC // GROUP_SIZE


def test_replace_no_repeat_within_group(ntc_inputs):
    groups = _make(ntc_inputs, replace=True)
    for grp in groups.values():
        assert len(grp) == GROUP_SIZE
        assert len(set(grp)) == GROUP_SIZE


def test_replace_reuses_guides_across_groups(ntc_inputs):
    groups = _make(ntc_inputs, replace=True)
    counts = Counter(g for grp in groups.values() for g in grp)
    assert max(counts.values()) > 1
    assert len(groups) > N_NTC // GROUP_SIZE


def test_replace_no_identical_groups(ntc_inputs):
    groups = _make(ntc_inputs, replace=True)
    keys = [tuple(sorted(grp)) for grp in groups.values()]
    assert len(keys) == len(set(keys))


def test_replace_groups_match_a_real_signature(ntc_inputs):
    _, _, real_sigs, guide_to_bin = ntc_inputs
    sigs = {tuple(s) for s in real_sigs}
    groups = _make(ntc_inputs, replace=True)
    for grp in groups.values():
        assert tuple(sorted(guide_to_bin[g] for g in grp)) in sigs


def test_replace_group_count_caps(ntc_inputs):
    assert len(_make(ntc_inputs, replace=True)) == N_REAL_TARGETS
    assert len(_make(ntc_inputs, replace=True, max_groups=9)) == 9


def test_ensemble_passes_replace(ntc_inputs):
    ntc_guides, ntc_freq, real_sigs, guide_to_bin = ntc_inputs
    ens = make_ntc_groups_ensemble(
        ntc_guides=ntc_guides, ntc_freq=ntc_freq, real_gene_bin_sigs=real_sigs,
        guide_to_bin=guide_to_bin, n_ensemble=3, seed0=7, group_size=GROUP_SIZE,
        max_groups=9, replace=True)
    assert [len(groups) for groups in ens] == [9, 9, 9]
    assert ens[0] != ens[1]
