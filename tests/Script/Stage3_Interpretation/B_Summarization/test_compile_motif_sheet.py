"""Unit tests for Stage3_Interpretation/B_Summarization/src/Compile_motif_sheet.py.

Test strategy
  Compile_Motif_sheet:        top-N significant motifs per program x element type, ordered by FDR
                              (ties: higher enrichment); depleted / non-significant rows excluded;
                              counts of significant and tested motifs; program with no significant
                              motif -> 0 and ""; `significant` as strings; `significant` missing
                              (derived from fdr < 0.05 & enrichment > 1); large synthetic table
  Compile_Candidate_TF_sheet: strongest evidence tier first within a program, then FDR; boolean
                              columns parsed from strings
  motif_source (--motif_source both): two sources -> separate columns per element type x source,
                              counts not pooled, FIMO rows first; one source in the column -> same
                              sheets as without the column

Imported by file path so it runs without the package's heavy __init__ (muon / mygene / sklearn).
Run: pytest tests/Script/Stage3_Interpretation/B_Summarization/test_compile_motif_sheet.py
(needs pandas, pytest only).
"""

import importlib.util
import os

import pandas as pd
import pytest

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src", "Stage3_Interpretation",
                           "B_Summarization", "src", "Compile_motif_sheet.py")
spec = importlib.util.spec_from_file_location("Compile_motif_sheet", MODULE_PATH)
Compile_motif_sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Compile_motif_sheet)

MOTIF_COLUMNS = ["program", "element_type", "tf", "pvalue", "fdr", "enrichment", "significant"]


def write_tsv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.fixture
def motif_path(tmp_path):
    rows = [
        (1, "promoter", "KLF4", 1e-7, 1e-6, 2.0, True),
        (1, "promoter", "SOX2", 1e-5, 1e-4, 1.5, True),
        (1, "promoter", "TEAD1", 1e-5, 1e-4, 3.0, True),   # same FDR as SOX2, higher enrichment -> first
        (1, "promoter", "ZFX", 1e-9, 1e-8, 0.2, False),    # depleted
        (1, "promoter", "AHR", 0.3, 0.6, 1.2, False),
        (1, "enhancer", "ETS1", 1e-4, 1e-3, 1.3, True),
        (2, "promoter", "KLF4", 0.2, 0.5, 1.1, False),
        (2, "enhancer", "ETS1", 0.2, 0.5, 1.1, False),
    ]
    return write_tsv(tmp_path / "motif_enrichment.tsv", rows, MOTIF_COLUMNS)


def test_summary_top_motifs_counts_and_order(motif_path):
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(motif_path, top_n=2)
    assert list(summary.index) == [1, 2] and summary.index.name == "program_name"
    got = summary.loc[1, "top2_promoter_motifs"]
    assert got == "KLF4 (2.00x, FDR 1.0e-06); TEAD1 (3.00x, FDR 1.0e-04)", \
        f"top 2 by FDR, ties by higher enrichment, depleted ZFX excluded; got {got!r}"
    assert summary.loc[1, "n_significant_promoter_motifs"] == 3
    assert summary.loc[1, "n_tested_promoter_motifs"] == 5
    assert summary.loc[1, "top2_enhancer_motifs"] == "ETS1 (1.30x, FDR 1.0e-03)"
    assert summary.loc[2, "n_significant_promoter_motifs"] == 0 and summary.loc[2, "top2_promoter_motifs"] == ""
    assert set(significant["tf"]) == {"KLF4", "SOX2", "TEAD1", "ETS1"}, f"significant rows: {sorted(significant['tf'])}"
    assert significant.index.name == "program_name"


def test_significant_as_strings_or_missing(tmp_path):
    rows = [(1, "Promoter", "KLF4", 1e-7, 1e-6, 2.0, "True"), (1, "Promoter", "ZFX", 1e-9, 1e-8, 0.2, "False")]
    as_strings = write_tsv(tmp_path / "strings.tsv", rows, MOTIF_COLUMNS)
    summary, _ = Compile_motif_sheet.Compile_Motif_sheet(as_strings)
    assert summary.loc[1, "n_significant_promoter_motifs"] == 1  # element_type lower-cased too

    missing = write_tsv(tmp_path / "missing.tsv", [r[:-1] for r in rows], MOTIF_COLUMNS[:-1])
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(missing)
    assert list(significant["tf"]) == ["KLF4"]  # fdr < 0.05 & enrichment > 1


def test_candidate_tfs_sorted_by_tier_then_fdr(tmp_path):
    columns = ["program", "element_type", "tf", "fdr", "tf_expressed", "tf_knockdown_regulates_program", "evidence_tier"]
    rows = [
        (1, "promoter", "A", 1e-9, "True", "False", "motif_only"),
        (1, "promoter", "B", 1e-3, "True", "True", "motif+regulator"),
        (1, "enhancer", "C", 1e-2, "True", "False", "motif+expressed_in_program"),
        (1, "promoter", "D", 1e-4, "True", "False", "motif+expressed_in_program"),
        (0, "promoter", "E", 1e-4, "False", "False", "motif_only"),
    ]
    candidates = Compile_motif_sheet.Compile_Candidate_TF_sheet(write_tsv(tmp_path / "c.tsv", rows, columns))
    assert list(candidates["tf"]) == ["E", "B", "D", "C", "A"], f"expected program, tier, FDR order; got {list(candidates['tf'])}"
    assert candidates["tf_knockdown_regulates_program"].dtype == bool
    assert candidates.index.name == "program_name"


def test_large_table_without_significant_column(tmp_path):
    """Synthetic full-size table (programs x element types x TFs, no `significant` column):
    significant rows are derived from fdr < 0.05 & enrichment > 1 and every program is summarized."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = [(program, element_type, f"TF{t}", p, min(1.0, p * 20), enrichment)
            for program in range(12) for element_type in ("promoter", "enhancer") for t in range(80)
            for p, enrichment in [(float(rng.uniform(0, 0.05)), float(rng.uniform(0.5, 2.0)))]]
    table = pd.DataFrame(rows, columns=["program", "element_type", "tf", "pvalue", "fdr", "enrichment"])
    path = tmp_path / "large.tsv"
    table.to_csv(path, sep="\t", index=False)
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(str(path))
    assert len(summary) == 12
    expected = int(((table["fdr"] < 0.05) & (table["enrichment"] > 1)).sum())
    assert expected > 0
    assert len(significant) == expected, f"expected {expected} significant rows, got {len(significant)}"
    assert summary[["n_significant_promoter_motifs", "n_significant_enhancer_motifs"]].to_numpy().sum() == expected


def two_source_rows(motif_path):
    fimo = pd.read_csv(motif_path, sep="\t").assign(motif_source="fimo")
    finemo = pd.DataFrame([(1, "promoter", "KLF-SP", 1e-5, 1e-4, 1.8, True),
                           (1, "enhancer", "ETS", 1e-9, 1e-8, 1.5, True),
                           (1, "enhancer", "KLF-SP", 1e-3, 1e-2, 1.2, True),
                           (2, "promoter", "NFY", 0.4, 0.8, 1.0, False)], columns=MOTIF_COLUMNS)
    return pd.concat([finemo.assign(motif_source="finemo"), fimo], ignore_index=True)


def test_two_motif_sources_are_not_pooled(motif_path, tmp_path):
    path = tmp_path / "both.tsv"
    two_source_rows(motif_path).to_csv(path, sep="\t", index=False)
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(str(path), top_n=2)
    assert list(summary.columns[:3]) == ["n_significant_promoter_fimo_motifs", "n_tested_promoter_fimo_motifs",
                                         "top2_promoter_fimo_motifs"], f"FIMO columns first: {list(summary.columns)}"
    assert summary.loc[1, "n_significant_promoter_fimo_motifs"] == 3 and summary.loc[1, "n_tested_promoter_fimo_motifs"] == 5
    assert summary.loc[1, "n_significant_promoter_finemo_motifs"] == 1 and summary.loc[1, "n_tested_promoter_finemo_motifs"] == 1
    assert summary.loc[1, "top2_enhancer_finemo_motifs"] == "ETS (1.50x, FDR 1.0e-08); KLF-SP (1.20x, FDR 1.0e-02)"
    assert summary.loc[1, "top2_enhancer_fimo_motifs"] == "ETS1 (1.30x, FDR 1.0e-03)"
    assert "n_significant_promoter_motifs" not in summary.columns, "no pooled columns with two sources"
    program_1 = significant.loc[1]
    assert list(program_1["motif_source"]) == ["fimo"] * 4 + ["finemo"] * 3, list(program_1["motif_source"])


def test_single_motif_source_column_gives_the_same_sheets(motif_path, tmp_path):
    path = tmp_path / "one_source.tsv"
    pd.read_csv(motif_path, sep="\t").assign(motif_source="finemo").to_csv(path, sep="\t", index=False)
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(str(path))
    plain_summary, plain_significant = Compile_motif_sheet.Compile_Motif_sheet(motif_path)
    pd.testing.assert_frame_equal(summary, plain_summary)
    pd.testing.assert_frame_equal(significant.drop(columns="motif_source"), plain_significant)


def test_candidate_tfs_fimo_before_finemo_within_a_tier(tmp_path):
    columns = ["program", "element_type", "tf", "fdr", "evidence_tier", "motif_source"]
    rows = [(1, "enhancer", "ETS", 1e-9, "motif+expressed", "finemo"),
            (1, "promoter", "KLF4", 1e-3, "motif+expressed", "fimo"),
            (1, "promoter", "SP1", 1e-2, "motif+regulator", "fimo")]
    candidates = Compile_motif_sheet.Compile_Candidate_TF_sheet(write_tsv(tmp_path / "c.tsv", rows, columns))
    assert list(candidates["tf"]) == ["SP1", "KLF4", "ETS"], list(candidates["tf"])


def test_correlation_method_shows_r(motif_path, tmp_path):
    summary, _ = Compile_motif_sheet.Compile_Motif_sheet(motif_path, top_n=1, method="correlation")
    assert summary.loc[1, "top1_promoter_motifs"] == "KLF4 (r=2.00, FDR 1.0e-06)", summary.loc[1, "top1_promoter_motifs"]
    config = os.path.splitext(motif_path)[0] + "_config.yml"
    with open(config, "w") as handle:
        handle.write('{"arguments": {"motif_method": "correlation", "n_top": 300}}')
    summary, _ = Compile_motif_sheet.Compile_Motif_sheet(motif_path, top_n=1)
    assert summary.loc[1, "top1_promoter_motifs"].startswith("KLF4 (r=2.00"), "method read from the Stage 2 config"
    os.remove(config)
    summary, _ = Compile_motif_sheet.Compile_Motif_sheet(motif_path, top_n=1)
    assert summary.loc[1, "top1_promoter_motifs"] == "KLF4 (2.00x, FDR 1.0e-06)", "default t-test wording"


def test_motif_family_column_adds_family_summary(tmp_path):
    rows = [
        (1, "promoter", "KLF-SP_0", "KLF-SP", 1e-7, 1e-6, 2.0, True),
        (1, "promoter", "GATA_0", "GATA", 1e-6, 1e-5, 1.4, True),
        (1, "promoter", "KLF-SP_1", "KLF-SP", 1e-5, 1e-4, 1.5, True),
        (1, "promoter", "ETV_0", "ETV", 0.3, 0.6, 1.1, False),
        (2, "promoter", "GATA_0", "GATA", 0.3, 0.6, 1.1, False),
    ]
    path = write_tsv(tmp_path / "families.tsv", rows, ["program", "element_type", "tf", "motif_family", "pvalue", "fdr",
                                                       "enrichment", "significant"])
    summary, significant = Compile_motif_sheet.Compile_Motif_sheet(path, top_n=5)
    assert summary.loc[1, "families_promoter_motifs"] == "KLF-SP (2); GATA (1)", "best family first, motif counts"
    assert summary.loc[2, "families_promoter_motifs"] == ""
    assert "motif_family" in significant.columns
    summary_without, _ = Compile_motif_sheet.Compile_Motif_sheet(
        write_tsv(tmp_path / "plain.tsv", [r[:3] + r[4:] for r in rows], MOTIF_COLUMNS), top_n=5)
    assert not any(c.startswith("families_") for c in summary_without.columns), "no family column -> no family summary"
