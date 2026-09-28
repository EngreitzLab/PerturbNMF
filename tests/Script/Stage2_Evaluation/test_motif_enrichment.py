"""Unit tests for Stage2_Evaluation/A_Metrics/src/motif_enrichment.py (t-test method of Schnitzler et al. 2024).

The module is imported by file path so the tests run without the A_Metrics package's heavy
``__init__`` imports, and as a module object so pytest does not collect ``test_motif_enrichment_ttest``.
Run: ``pytest --noconftest tests/Script/Stage2_Evaluation/test_motif_enrichment.py`` (needs pandas,
numpy, scipy, pytest only).

Test strategy
  read_fimo_hits:     p-value below / equal to threshold (strict); sequence_name = gene vs ABC name
                      (with and without bedtools '::' suffix); malformed ABC name -> error
  collapse TF:        id with '_' / without '_'
  counting:           hits in several enhancers of one gene (summed); promoter-class element dropped;
                      non-expressed gene dropped; gene/TF absent -> 0
  top genes:          ties (input order kept), NaN (last)
  BH:                 reference values; NaN ignored in n
  t-test:             p/enrichment vs scipy Welch per cell; background excludes program genes;
                      program genes with 0 hits dropped; constant data -> NaN (R t.test errors)
  significance:       fdr / enrichment boundaries
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "Stage2_Evaluation",
                           "A_Metrics", "src", "motif_enrichment.py")
spec = importlib.util.spec_from_file_location("motif_enrichment", MODULE_PATH)
motif_enrichment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motif_enrichment)

FIMO_HEADER = "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n"


def write_fimo(path, rows):
    with open(path, "w") as handle:
        handle.write(FIMO_HEADER)
        for motif_id, sequence_name, pvalue in rows:
            handle.write(f"{motif_id}\t\t{sequence_name}\t1\t10\t+\t10.0\t{pvalue}\t0.1\tACGT\n")
        handle.write("\n# FIMO (Find Individual Motif Occurrences): Version 5.3.3\n")


def test_collapse_motif_to_tf():
    tfs = motif_enrichment.collapse_motif_to_tf(["ZN329_HUMAN.H11MO.0.C", "AP1_HUMAN.H11MO.0.A", "NOUNDERSCORE"])
    assert tfs.tolist() == ["ZN329", "AP1", "NOUNDERSCORE"], f"TF = text before first '_', got {tfs.tolist()}"


def test_read_fimo_hits_promoter_strict_threshold(tmp_path):
    path = tmp_path / "fimo.tsv"
    write_fimo(path, [("SP1_HUMAN.H11MO.0.A", "GENE1", "1e-05"),
                      ("SP1_HUMAN.H11MO.1.A", "GENE1", "0.0001"),   # == threshold -> dropped
                      ("KLF4_HUMAN.H11MO.0.A", "GENE2", "5e-05")])
    hits = motif_enrichment.read_fimo_hits(str(path), 1e-4)
    assert hits["tf"].tolist() == ["SP1", "KLF4"] and hits["gene"].tolist() == ["GENE1", "GENE2"], (
        f"expected p == threshold dropped and comment lines skipped, got\n{hits}")


def test_read_fimo_hits_abc_parser(tmp_path):
    path = tmp_path / "fimo.tsv"
    write_fimo(path, [("SP1_HUMAN.H11MO.0.A", "chr1:1-9|genic|chr1:0-10|GENE1", "1e-07"),
                      ("SP1_HUMAN.H11MO.0.A", "chr1:1-9|promoter|chr1:0-10|GENE2::chr1:0-10", "1e-07")])
    hits = motif_enrichment.read_fimo_hits(str(path), 1e-6, motif_enrichment.parse_abc_sequence_name)
    assert hits["gene"].tolist() == ["GENE1", "GENE2"], f"target gene = 4th field minus '::', got {hits['gene'].tolist()}"
    assert hits["element_class"].tolist() == ["genic", "promoter"], f"class = 2nd field, got {hits['element_class'].tolist()}"


def test_parse_abc_sequence_name_rejects_wrong_field_count():
    with pytest.raises(ValueError, match="4 '\\|'-separated fields"):
        motif_enrichment.parse_abc_sequence_name(pd.Series(["chr1:1-9|genic|GENE1"]))


def test_count_hits_sums_over_enhancers_and_filters():
    hits = pd.DataFrame({
        "gene": ["A", "A", "A", "B", "C", "D"],
        "tf": ["SP1", "SP1", "KLF4", "SP1", "SP1", "SP1"],
        "element_class": ["genic", "intergenic", "genic", "promoter", "genic", "genic"],
    })
    counts = motif_enrichment.count_hits_per_gene_tf(hits, genes=["A", "B", "C"],
                                                     excluded_element_classes=["promoter"])
    expected = pd.DataFrame({"KLF4": [1, 0], "SP1": [2, 1]}, index=pd.Index(["A", "C"], name="gene"))
    pd.testing.assert_frame_equal(counts, expected, check_dtype=False,
                                  obj="counts (B only promoter-class, D not expressed, A summed over 2 enhancers)")


def test_select_top_program_genes_ties_keep_input_order():
    scores = pd.DataFrame([[1.0, 3.0, 3.0, np.nan]], index=["P1"], columns=["g1", "g2", "g3", "g4"])
    top = motif_enrichment.select_top_program_genes(scores, n_top=3)
    assert top["gene"].tolist() == ["g2", "g3", "g1"] and top["rank"].tolist() == [1, 2, 3], (
        f"expected dplyr arrange(desc) order (stable ties, NaN last), got\n{top}")


def test_adjust_pvalues_bh_matches_reference_and_ignores_nan():
    pvalues = np.array([0.01, 0.04, np.nan, 0.03, 0.2])
    # R: p.adjust(c(0.01, 0.04, 0.03, 0.2), "BH") = 0.04 0.0533 0.0533 0.2
    expected = np.array([0.04, 0.04 * 4 / 3, np.nan, 0.04 * 4 / 3, 0.2])
    np.testing.assert_allclose(motif_enrichment.adjust_pvalues_bh(pvalues), expected, rtol=1e-12,
                               err_msg="BH must match R p.adjust with NaN excluded from n")


def make_counts_and_programs(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(200)]
    counts = pd.DataFrame(rng.poisson(1.0, size=(200, 3)), index=genes, columns=["KLF4", "SP1", "TEAD1"])
    counts.iloc[:30, 0] += 3                             # enrich KLF4 in program P1
    program_genes = pd.DataFrame({
        "program": ["P1"] * 30 + ["P2"] * 40,
        "gene": genes[:30] + genes[100:140],
    })
    return counts, program_genes


def test_ttest_matches_scipy_welch_and_background_excludes_program_genes():
    counts, program_genes = make_counts_and_programs()
    result = motif_enrichment.test_motif_enrichment_ttest(counts, program_genes, "promoter")
    assert len(result) == 6, f"expected 2 programs x 3 TFs rows, got {len(result)}"
    for _, row in result.iterrows():
        in_program = counts.index.isin(program_genes.loc[program_genes["program"] == row["program"], "gene"])
        program_counts = counts.loc[in_program, row["tf"]]
        background_counts = counts.loc[~in_program, row["tf"]]
        expected = stats.ttest_ind(program_counts, background_counts, equal_var=False)
        label = f"{row['program']}/{row['tf']}"
        assert row["pvalue"] == pytest.approx(expected.pvalue, rel=1e-10), (
            f"{label}: Welch p {row['pvalue']} != scipy {expected.pvalue}")
        assert row["enrichment"] == pytest.approx(program_counts.mean() / background_counts.mean(), rel=1e-12), (
            f"{label}: enrichment must be mean(program) / mean(universe minus program)")
        assert row["n_background_genes"] == 200 - in_program.sum(), f"{label}: background must exclude program genes"
    np.testing.assert_allclose(result["fdr"], motif_enrichment.adjust_pvalues_bh(result["pvalue"].to_numpy()),
                               err_msg="fdr must be BH over all program x TF rows")
    klf4_p1 = result[(result["program"] == "P1") & (result["tf"] == "KLF4")].iloc[0]
    assert klf4_p1["pvalue"] < 1e-10 and klf4_p1["enrichment"] > 1, f"planted KLF4 enrichment not detected: {klf4_p1}"


def test_ttest_drops_program_genes_without_hits():
    counts, program_genes = make_counts_and_programs()
    extra = pd.DataFrame({"program": ["P1"] * 5, "gene": [f"nohit{i}" for i in range(5)]})
    with_missing = motif_enrichment.test_motif_enrichment_ttest(
        counts, pd.concat([program_genes, extra]), "promoter")
    without = motif_enrichment.test_motif_enrichment_ttest(counts, program_genes, "promoter")
    pd.testing.assert_frame_equal(with_missing, without, obj="results with extra zero-hit program genes")
    n_tested = with_missing.loc[with_missing["program"] == "P1", "n_program_genes_tested"].unique().tolist()
    assert n_tested == [30], f"zero-hit program genes must be dropped, got n_program_genes_tested {n_tested}"


def test_ttest_constant_data_gives_nan_like_r_error():
    counts = pd.DataFrame({"SP1": [1] * 10}, index=[f"g{i}" for i in range(10)])
    program_genes = pd.DataFrame({"program": ["P1"] * 4, "gene": ["g0", "g1", "g2", "g3"]})
    result = motif_enrichment.test_motif_enrichment_ttest(counts, program_genes, "promoter")
    assert np.isnan(result["pvalue"].iloc[0]) and np.isnan(result["fdr"].iloc[0]), (
        f"constant data (R t.test error) must give NaN p/fdr, got {result.iloc[0].to_dict()}")


def test_flag_significant_boundaries():
    results = pd.DataFrame({"fdr": [0.01, 0.01, 0.2, 0.05, 0.01], "enrichment": [2.0, 0.5, 3.0, 2.0, 1.0]})
    flags = motif_enrichment.flag_significant(results)["significant"].tolist()
    assert flags == [True, False, False, False, False], f"significant = fdr < 0.05 & enrichment > 1 (strict), got {flags}"


# ---------------------------------------------------------------------------
# No p-value filter (Fi-NeMo tables) and the correlation method
# ---------------------------------------------------------------------------

def test_read_fimo_hits_without_threshold_keeps_na_pvalues(tmp_path):
    path = tmp_path / "finemo_hits.tsv"
    write_fimo(path, [("KLF-SP_counts_pattern_0", "GENE1", "NA"),
                      ("GATA_counts_pattern_2", "GENE2", "NA"),
                      ("GATA_counts_pattern_2", "GENE2", "0.5")])
    hits = motif_enrichment.read_fimo_hits(str(path), None)
    assert hits["tf"].tolist() == ["KLF-SP", "GATA", "GATA"], f"all rows kept, TF before '_': {hits}"
    assert hits["pvalue"].isna().sum() == 2, f"NA p-values should read as NaN, got {hits['pvalue'].tolist()}"
    filtered = motif_enrichment.read_fimo_hits(str(path), 1e-4)
    assert filtered.empty, f"a threshold drops NA p-values (NaN < t is False), got\n{filtered}"


def make_correlation_inputs():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(40)]
    loadings = pd.DataFrame(rng.gamma(1.0, 1.0, size=(3, 40)), index=["P0", "P1", "P2"], columns=genes)
    # counts only for genes with hits (G30..G39 absent -> 0 in the default universe)
    counts = pd.DataFrame(rng.poisson(2.0, size=(30, 4)), index=genes[:30], columns=["A", "B", "C", "D"])
    counts["D"] = 0
    counts.loc["G0", "D"] = 0
    return loadings, counts


@pytest.mark.parametrize("correlation, scipy_function", [("pearson", stats.pearsonr), ("spearman", stats.spearmanr)])
def test_correlation_matches_scipy(correlation, scipy_function):
    loadings, counts = make_correlation_inputs()
    counts = counts.drop(columns="D")
    result = motif_enrichment.test_motif_enrichment_correlation(counts, loadings, "promoter", correlation)
    assert list(result.columns) == motif_enrichment.RESULT_COLUMNS, f"columns {list(result.columns)}"
    assert len(result) == 3 * 3, f"one row per program x TF, got {len(result)}"
    full_counts = counts.reindex(loadings.columns, fill_value=0)
    for row in result.itertuples():
        expected = scipy_function(loadings.loc[row.program].to_numpy(), full_counts[row.tf].to_numpy())
        assert row.enrichment == pytest.approx(expected[0], rel=1e-10), f"{row.program}/{row.tf} r"
        assert row.pvalue == pytest.approx(expected[1], rel=1e-8), f"{row.program}/{row.tf} p"
        assert row.n_program_genes_tested == 40, "default universe = every loading gene (zeros filled)"
    assert np.allclose(result["fdr"], motif_enrichment.adjust_pvalues_bh(result["pvalue"].to_numpy())), "BH over all rows"


def test_correlation_universe_restriction_and_constant_tf():
    loadings, counts = make_correlation_inputs()
    result = motif_enrichment.test_motif_enrichment_correlation(
        counts, loadings, "enhancer", universe_genes=list(counts.index) + ["NOT_A_GENE"])
    assert (result["n_program_genes_tested"] == 30).all(), "universe restricted to given genes present in loadings"
    constant = result[result["tf"] == "D"]
    assert constant["pvalue"].isna().all() and constant["enrichment"].isna().all(), (
        f"a TF with constant counts has undefined correlation, got\n{constant}")
    expected = stats.pearsonr(loadings.loc["P1", counts.index].to_numpy(), counts["A"].to_numpy())
    got = result[(result["program"] == "P1") & (result["tf"] == "A")].iloc[0]
    assert got["enrichment"] == pytest.approx(expected[0], rel=1e-10), "restricted-universe correlation"


def test_correlation_rejects_unknown_method():
    loadings, counts = make_correlation_inputs()
    with pytest.raises(ValueError, match="pearson or spearman"):
        motif_enrichment.test_motif_enrichment_correlation(counts, loadings, "promoter", "kendall")


def test_flag_significant_correlation_threshold():
    results = pd.DataFrame({"fdr": [0.01, 0.01, 0.2], "enrichment": [0.3, -0.3, 0.5]})
    flagged = motif_enrichment.flag_significant(results, method="correlation")
    assert flagged["significant"].tolist() == [True, False, False], "correlation: fdr < 0.05 and r > 0"
    assert not motif_enrichment.flag_significant(results)["significant"].any(), "t-test needs enrichment > 1"
    with pytest.raises(ValueError):
        motif_enrichment.flag_significant(results, method="nope")


def test_read_fimo_hits_uses_motif_alt_id_for_jaspar_ids(tmp_path):
    path = tmp_path / "fimo.tsv"
    with open(path, "w") as handle:
        handle.write(FIMO_HEADER)
        handle.write("MA0139.1\tCTCF\tGENEA\t1\t10\t+\t10\t1e-5\t\tACGT\n")                 # JASPAR
        handle.write("CTCF_HUMAN.H11MO.0.A\tCTCF_alt\tGENEA\t1\t10\t+\t10\t1e-5\t\tACGT\n")  # HOCOMOCO: id wins
        handle.write("MA0001.1\t\tGENEB\t1\t10\t+\t10\t1e-5\t\tACGT\n")                    # no alt id
    hits = motif_enrichment.read_fimo_hits(str(path), 1e-4)
    assert hits["tf"].tolist() == ["CTCF", "CTCF", "MA0001.1"]
    assert list(hits.columns) == ["motif_id", "tf", "pvalue", "gene"]
    assert motif_enrichment.read_fimo_hits(str(path), 1e-4, use_motif_alt_id=False)["tf"].tolist() == [
        "MA0139.1", "CTCF", "MA0001.1"]
