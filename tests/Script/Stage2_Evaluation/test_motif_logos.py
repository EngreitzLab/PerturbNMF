"""Unit tests for Stage2_Evaluation/A_Metrics/src/motif_logos.py ({K}_motif_logos.json matrices).

Test strategy
  MEME parsing:        ids, alt ids, matrix width
  information content: uniform column = 0 bits, one-hot column = 2 bits
  HOCOMOCO model pick: quality A before B; exact id / alt id wins
  CWM trimming:        low-contribution flanks dropped
  build_motif_logos:   only significant motifs; per source; Fi-NeMo from the matched PFM (lowest-q pattern)
"""

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "Stage2_Evaluation", "A_Metrics", "src")
sys.path.insert(0, SRC)
spec = importlib.util.spec_from_file_location("motif_logos", os.path.join(SRC, "motif_logos.py"))
motif_logos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motif_logos)


def write_meme(path, motifs):
    lines = ["MEME version 4", "", "ALPHABET= ACGT", ""]
    for (motif_id, alt_id), rows in motifs.items():
        lines += [f"MOTIF {motif_id} {alt_id}".strip(), "",
                  f"letter-probability matrix: alength= 4 w= {len(rows)} nsites= 20 E= 0"]
        lines += [" ".join(f"{v:.3f}" for v in row) for row in rows] + ["", "URL http://example.org", ""]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def meme_path(tmp_path):
    return write_meme(tmp_path / "db.meme", {
        ("KLF4_HUMAN.H11MO.0.A", ""): [[0, 0, 1, 0], [0.25, 0.25, 0.25, 0.25]],
        ("SP1_HUMAN.H11MO.1.B", ""): [[0, 1, 0, 0]],
        ("SP1_HUMAN.H11MO.0.A", ""): [[1, 0, 0, 0]],
        ("MA0139.1", "CTCF"): [[0, 0, 0, 1]],
        ("GATA_0", ""): [[1, 0, 0, 0], [0, 0, 0, 1]],
    })


def test_read_meme_and_information_content(meme_path):
    motifs = motif_logos.read_meme_motifs(meme_path)
    assert list(motifs) == ["KLF4_HUMAN.H11MO.0.A", "SP1_HUMAN.H11MO.1.B", "SP1_HUMAN.H11MO.0.A", "MA0139.1", "GATA_0"]
    assert motifs["MA0139.1"]["alt_id"] == "CTCF" and motifs["KLF4_HUMAN.H11MO.0.A"]["matrix"].shape == (2, 4)
    heights = motif_logos.information_content_matrix(motifs["KLF4_HUMAN.H11MO.0.A"]["matrix"])
    np.testing.assert_allclose(heights, [[0, 0, 2, 0], [0, 0, 0, 0]], atol=1e-9)


def test_select_hocomoco_model_quality_and_exact_names(meme_path):
    motifs = motif_logos.read_meme_motifs(meme_path)
    assert motif_logos.select_hocomoco_model("SP1", motifs) == "SP1_HUMAN.H11MO.0.A", "quality A before B"
    assert motif_logos.select_hocomoco_model("CTCF", motifs) == "MA0139.1", "alt id match"
    assert motif_logos.select_hocomoco_model("GATA_0", motifs) == "GATA_0", "exact id"
    assert motif_logos.select_hocomoco_model("TEAD1", motifs) is None


def test_trim_matrix_drops_weak_flanks():
    cwm = np.array([[0.01, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -0.4, 0], [0.02, 0, 0, 0]])
    np.testing.assert_array_equal(motif_logos.trim_matrix(cwm), cwm[1:3])


def test_build_motif_logos_significant_only_per_source(meme_path, tmp_path):
    results = pd.DataFrame({
        "program": [1, 1, 2, 1, 1], "element_type": "promoter",
        "tf": ["KLF4", "SP1", "CTCF", "GATA_0", "GATA_2"],
        "significant": [True, False, True, True, False],
        "motif_source": ["fimo", "fimo", "fimo", "finemo", "finemo"]})
    pattern_names = pd.DataFrame({
        "pattern_id": ["pos_patterns.x_counts_pattern_0", "pos_patterns.x_counts_pattern_4",
                       "pos_patterns.x_counts_pattern_9"],
        "tf": ["GATA_0", "GATA_0", "GATA_2"], "top_match": ["GATA_0", "GATA_0", "GATA_2"],
        "top_match_qvalue": [0.3, 0.01, 0.2], "is_named": True})
    logos = motif_logos.build_motif_logos(results, fimo_motif_file=meme_path, fimo_collapse_motif_ids=True,
                                          pattern_names=pattern_names, finemo_pfm_file=meme_path)
    assert set(logos["logos"]["fimo"]) == {"KLF4", "CTCF"}, "SP1 not significant -> no logo"
    assert logos["logos"]["fimo"]["KLF4"]["motif_id"] == "KLF4_HUMAN.H11MO.0.A"
    finemo = logos["logos"]["finemo"]
    assert set(finemo) == {"GATA_0"} and finemo["GATA_0"]["pattern_id"].endswith("pattern_4"), "lowest-q pattern"
    assert finemo["GATA_0"]["kind"] == "information_content" and finemo["GATA_0"]["matrix"] == [[2, 0, 0, 0], [0, 0, 0, 2]]
    assert logos["logos"]["fimo"]["KLF4"]["matrix"] == [[0, 0, 2, 0]], "0-bit flank trimmed from the IC logo"
    path = tmp_path / "logos.json"
    motif_logos.write_motif_logos(logos, str(path))
    assert json.loads(path.read_text()) == json.loads(json.dumps(logos))


def test_finemo_logos_local_compendium_cluster_ids(tmp_path):
    """Local compendium motif ids (cluster_<n>) take the CWM from the compiled compendium h5 (keys
    pos_patterns/<n>); of two clusters with the same database match, the higher match score wins."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "modisco_compiled.h5"
    strong = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    weak = np.array([[0.0, 0.5, 0.0, 0.0]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pos_patterns/7/contrib_scores", data=strong)
        handle.create_dataset("pos_patterns/9/contrib_scores", data=weak)
    pattern_names = pd.DataFrame({
        "pattern_id": ["cluster_9", "cluster_7"], "tf": ["GATA_0", "GATA_0"], "top_match": ["GATA_0", "GATA_0"],
        "top_match_qvalue": np.nan, "top_match_score": [0.85, 0.95], "is_named": True})
    logos = motif_logos.finemo_logos(pattern_names, ["GATA_0"], finemo_motifs=str(path))
    assert logos["GATA_0"]["kind"] == "cwm" and logos["GATA_0"]["pattern_id"] == "cluster_7"
    assert motif_logos.modisco_keys("cluster_7") == ["pos_patterns.7", "neg_patterns.7"]
    assert motif_logos.modisco_keys("pos_patterns.x_counts_pattern_3") == ["pos_patterns.pattern_3"]
