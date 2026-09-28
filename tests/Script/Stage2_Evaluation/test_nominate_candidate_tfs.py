"""Unit tests for Stage2_Evaluation/A_Metrics/src/nominate_candidate_tfs.py.

Imported by file path (no heavy A_Metrics ``__init__``). Run:
``pytest --noconftest tests/Script/Stage2_Evaluation/test_nominate_candidate_tfs.py``.

Test strategy
  HOCOMOCO table:   bundled file loads; mnemonics -> HGNC (ZN148, P63, ANDR, ERR1, BC11A, ZBT17);
                    identity names (KLF4; note HOCOMOCO v11 has no KLF2 motif); 1 row per TF name
  name mapping:     HOCOMOCO hit; unknown name passthrough; JASPAR dimer split; lookup disabled
  loading rank:     descending score, NaN last, ties by gene order
  nomination:       only enriched rows (fdr / enrichment boundaries); each evidence tier;
                    TF not in universe -> rank NaN; knockdown not significant -> not regulator;
                    TF perturbed but not in this program's rows; no perturbation table;
                    int vs str program ids; perturbation column rename; missing columns -> error;
                    columns compatible with motif_enrichment_plots.CANDIDATE_COLUMNS
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

REPO = os.path.join(os.path.dirname(__file__), "..", "..", "..")
MODULE_PATH = os.path.join(REPO, "src", "Stage2_Evaluation", "A_Metrics", "src", "nominate_candidate_tfs.py")
spec = importlib.util.spec_from_file_location("nominate_candidate_tfs", MODULE_PATH)
nct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nct)

PLOTS_PATH = os.path.join(REPO, "src", "Stage3_Interpretation", "A_Plotting", "src", "motif_enrichment_plots.py")


# ---------------------------------------------------------------------------
# TF name -> gene symbol
# ---------------------------------------------------------------------------

def test_hocomoco_table_maps_uniprot_mnemonics_to_hgnc():
    table = nct.read_hocomoco_tf_gene_symbols()
    lookup = dict(zip(table["tf"], table["tf_gene_symbol"]))
    expected = {"ZN148": "ZNF148", "P63": "TP63", "ANDR": "AR", "ERR1": "ESRRA", "BC11A": "BCL11A",
                "ZBT17": "ZBTB17", "KLF4": "KLF4", "GATA2": "GATA2", "COT2": "NR2F2"}
    observed = {tf: lookup.get(tf) for tf in expected}
    assert observed == expected, f"expected HOCOMOCO mnemonic->HGNC {expected}, got {observed}; check the bundled annotation tsv"
    assert table["tf"].is_unique, "each HOCOMOCO TF name must map to one gene"
    assert len(table) == 678, f"expected 678 TF names from 769 models (ST24 promoter table has 677), got {len(table)}"


def test_map_tf_names_hocomoco_passthrough_and_dimer():
    mapped = nct.map_tf_names_to_gene_symbols(["P63", "MYNEWTF", "FOS::JUN", "P63"])
    records = mapped.to_dict("records")
    assert records == [
        {"tf": "P63", "tf_gene_symbol": "TP63", "tf_gene_symbol_source": "hocomoco_v11"},
        {"tf": "MYNEWTF", "tf_gene_symbol": "MYNEWTF", "tf_gene_symbol_source": "passthrough"},
        {"tf": "FOS::JUN", "tf_gene_symbol": "FOS", "tf_gene_symbol_source": "passthrough"},
        {"tf": "FOS::JUN", "tf_gene_symbol": "JUN", "tf_gene_symbol_source": "passthrough"},
    ], f"HOCOMOCO hit, passthrough, dimer split and de-duplication failed: {records}"


def test_map_tf_names_lookup_disabled_is_pure_passthrough():
    mapped = nct.map_tf_names_to_gene_symbols(["P63"], hocomoco_tf_gene_symbols=pd.DataFrame())
    assert mapped[["tf_gene_symbol", "tf_gene_symbol_source"]].values.tolist() == [["P63", "passthrough"]], \
        f"empty lookup table should pass names through unchanged, got {mapped.to_dict('records')}"


# ---------------------------------------------------------------------------
# Loading rank
# ---------------------------------------------------------------------------

def test_rank_genes_descending_nan_last_ties_in_gene_order():
    scores = pd.DataFrame([[1.0, 3.0, np.nan, 3.0]], index=[7], columns=["a", "b", "c", "d"])
    ranks = nct.rank_genes_in_programs(scores)
    observed = dict(zip(ranks["gene"], ranks["loading_rank"]))
    assert observed == {"b": 1, "d": 2, "a": 3, "c": 4}, f"expected descending, ties in gene order, NaN last; got {observed}"
    assert ranks["program"].tolist() == [7] * 4, f"program id should be kept as-is, got {ranks['program'].tolist()}"


# ---------------------------------------------------------------------------
# Nomination
# ---------------------------------------------------------------------------

@pytest.fixture
def toy():
    # program 1: KLF2 top gene, GATA2 expressed but low, TP63 (motif P63) not expressed, ANDR knocked down
    genes = ["KLF2", "G1", "G2", "GATA2", "AR", "ERG"]
    scores = pd.DataFrame(
        [[9.0, 8.0, 7.0, 1.0, 0.5, 0.1],
         [0.1, 1.0, 2.0, 9.0, 3.0, 4.0]], index=[1, 2], columns=genes)
    motif = pd.DataFrame({
        "program": [1, 1, 1, 1, 1, 1, 2],
        "element_type": ["promoter", "promoter", "enhancer", "promoter", "promoter", "promoter", "promoter"],
        "tf": ["KLF2", "GATA2", "P63", "ANDR", "ERG", "ERG", "GATA2"],
        "pvalue": [1e-6, 1e-4, 1e-5, 1e-5, 0.01, 1e-9, 1e-8],
        "fdr": [1e-4, 0.049, 1e-3, 1e-3, 0.05, 1e-6, 1e-6],
        "enrichment": [2.0, 1.5, 3.0, 1.2, 2.0, 0.8, 2.5],
    })
    perturbation = pd.DataFrame({
        "target_name": ["AR", "AR", "KLF2", "GATA2", "GATA2"],
        "program_name": [1, 2, 1, 2, 2],
        "log2FC": [-0.8, 0.1, 0.3, -1.2, -0.5],
        "pval": [1e-5, 0.5, 0.2, 1e-8, 1e-3],
        "adj_pval": [1e-3, 0.9, 0.5, 1e-6, 0.01],
    })
    return motif, scores, perturbation


def test_nomination_keeps_only_enriched_rows(toy):
    motif, scores, perturbation = toy
    result = nct.nominate_candidate_tfs(motif, scores, perturbation, n_top_genes=3)
    # ERG: fdr == 0.05 (not <) and enrichment 0.8 (depleted) -> both dropped
    observed = set(zip(result["program"], result["tf"]))
    expected = {("1", "KLF2"), ("1", "GATA2"), ("1", "P63"), ("1", "ANDR"), ("2", "GATA2")}
    assert observed == expected, f"enriched = fdr < 0.05 & enrichment > 1; expected {expected}, got {observed}"


EVIDENCE_FIELDS = ["tf_gene_symbol", "tf_gene_symbol_source", "tf_expressed", "tf_program_loading_rank",
                   "tf_in_top_program_genes", "tf_perturbed", "knockdown_log2fc", "knockdown_fdr",
                   "tf_knockdown_regulates_program", "evidence_tier"]
EXPECTED_EVIDENCE = {
    # top gene, perturbed but knockdown not significant (fdr 0.5); no KLF2 model in HOCOMOCO v11
    ("1", "KLF2"): ["KLF2", "passthrough", True, 1, True, True, 0.3, 0.5, False, "motif+expressed_in_program"],
    # expressed, rank 4 > n_top 3; perturbed but no row for program 1
    ("1", "GATA2"): ["GATA2", "hocomoco_v11", True, 4, False, True, np.nan, np.nan, False, "motif+expressed"],
    # HOCOMOCO mnemonic -> TP63, not in gene universe
    ("1", "P63"): ["TP63", "hocomoco_v11", False, np.nan, False, False, np.nan, np.nan, False, "motif_only"],
    # ANDR -> AR, knockdown significant although AR loads low
    ("1", "ANDR"): ["AR", "hocomoco_v11", True, 5, False, True, -0.8, 1e-3, True, "motif+regulator"],
    # duplicated target rows -> smallest adj_pval kept
    ("2", "GATA2"): ["GATA2", "hocomoco_v11", True, 1, True, True, -1.2, 1e-6, True, "motif+regulator"],
}


def test_nomination_evidence_tiers_and_columns(toy):
    motif, scores, perturbation = toy
    result = nct.nominate_candidate_tfs(motif, scores, perturbation, n_top_genes=3)
    assert list(result.columns) == nct.CANDIDATE_TF_COLUMNS, f"column order changed: {list(result.columns)}"
    for (program, tf), expected in EXPECTED_EVIDENCE.items():
        row = result[(result["program"] == program) & (result["tf"] == tf)]
        assert len(row) == 1, f"expected one row for program {program} TF {tf}, got {len(row)}"
        observed = row[EVIDENCE_FIELDS].iloc[0].tolist()
        matches = [(pd.isna(o) and pd.isna(e)) or (o == pytest.approx(e) if isinstance(e, float) else o == e)
                   for o, e in zip(observed, expected)]
        assert all(matches), (f"program {program} TF {tf}: expected {dict(zip(EVIDENCE_FIELDS, expected))}, "
                              f"got {dict(zip(EVIDENCE_FIELDS, observed))}")
    order = result.loc[result["program"] == "1", "tf"].tolist()
    assert order == ["P63", "ANDR", "KLF2", "GATA2"], f"expected sort by element_type then tier then fdr, got {order}"


def test_nomination_without_perturbation_table(toy):
    motif, scores, _ = toy
    result = nct.nominate_candidate_tfs(motif, scores, None, n_top_genes=3)
    knockdown = result[["tf_perturbed", "tf_knockdown_regulates_program", "knockdown_fdr", "evidence_tier"]]
    assert (not result["tf_perturbed"].any() and not result["tf_knockdown_regulates_program"].any()
            and result["knockdown_fdr"].isna().all() and "motif+regulator" not in set(result["evidence_tier"])), \
        f"no perturbation table should give no knockdown evidence, got\n{knockdown}"


def test_nomination_perturbation_column_rename_and_missing(toy):
    motif, scores, perturbation = toy
    renamed = perturbation.rename(columns={"target_name": "target", "adj_pval": "fdr"})
    result = nct.nominate_candidate_tfs(motif, scores, renamed, n_top_genes=3,
                                        perturbation_columns={"target": "target_name", "fdr": "adj_pval"})
    n_regulators = int((result["evidence_tier"] == "motif+regulator").sum())
    assert n_regulators == 2, f"renamed columns should give the same 2 regulator rows, got {n_regulators}"
    with pytest.raises(ValueError, match="missing columns"):
        nct.nominate_candidate_tfs(motif, scores, renamed, n_top_genes=3)


def test_nomination_columns_cover_plot_module_candidate_columns():
    source = open(PLOTS_PATH).read()
    start = source.index("CANDIDATE_COLUMNS = [")
    plot_columns = eval(source[start + len("CANDIDATE_COLUMNS = "):source.index("]", start) + 1])
    missing = set(plot_columns) - set(nct.CANDIDATE_TF_COLUMNS)
    assert not missing, f"plot_candidate_tf_summary needs columns the nomination table lacks: {missing}"


# ---------------------------------------------------------------------------
# MotifCompendium family names (Fi-NeMo) and the significant column
# ---------------------------------------------------------------------------

def test_map_motif_families_uses_database_tf_lists_only():
    """Members = database TF list of the matched motif(s) that are expressed; no gene-name stems."""
    genes = ["KLF2", "KLF4", "SP1", "SPP1", "GATA2", "TAL1", "GATAD1", "NKX6-1", "NFE2L2", "ACTB"]
    families = {"KLF-SP_0": ["KLF2", "KLF4", "KLF6", "SP1", "MAZ"],     # KLF6 / MAZ not expressed
                "GATA_0": ["GATA1", "GATA2", "TAL1", "ZFPM1"],
                "HOX_0": ["NKX61"],                                     # database drops the hyphen
                "NF2L-NFE_0": ["NF2L2", "BACH1"],                       # HOCOMOCO mnemonic -> NFE2L2
                "pos-counts-pattern-3": []}
    table = nct.map_motif_families_to_gene_symbols(families, genes)
    members = table.groupby("tf", sort=False)["tf_gene_symbol"].apply(list).to_dict()
    assert members["KLF-SP_0"] == ["KLF2", "KLF4", "SP1"], "SPP1 is not on the list"
    assert members["GATA_0"] == ["GATA2", "TAL1"], "GATAD1 is not on the list"
    assert members["HOX_0"] == ["NKX6-1"]
    assert members["NF2L-NFE_0"] == ["NFE2L2"]
    assert members["pos-counts-pattern-3"] == ["pos-counts-pattern-3"]
    sources = table.set_index("tf")["tf_gene_symbol_source"]
    assert sources["pos-counts-pattern-3"] == "passthrough"
    assert set(table.loc[table["tf"] != "pos-counts-pattern-3", "tf_gene_symbol_source"]) == {"motifcompendium"}


def test_hocomoco_tf_families_tfclass_without_code():
    families = nct.read_hocomoco_tf_families().set_index("tf")["motif_family"]
    assert families["KLF4"] == families["SP1"] == "Three-zinc finger Krüppel-related factors"
    assert families["GATA2"] == "GATA-type zinc fingers"
    assert families["FLI1"] == "Ets-related factors"
    assert not families.str.contains(r"\{").any()


def test_nominate_uses_significant_column_and_keeps_motif_source():
    motif_results = pd.DataFrame({
        "program": ["P1", "P1"], "element_type": ["promoter", "promoter"], "tf": ["KLF-SP", "GATA"],
        "pvalue": [1e-6, 1e-6], "fdr": [1e-4, 1e-4], "enrichment": [0.4, -0.2],     # correlation values
        "significant": [True, False], "motif_source": ["finemo", "finemo"]})
    scores = pd.DataFrame([[3.0, 2.0, 1.0]], index=["P1"], columns=["KLF2", "KLF4", "ACTB"])
    symbols = nct.map_motif_families_to_gene_symbols({"KLF-SP": ["KLF2", "KLF4", "SP1"]}, scores.columns,
                                                     tf_name_to_symbol={})
    candidates = nct.nominate_candidate_tfs(motif_results, scores, tf_gene_symbols=symbols)
    assert candidates["tf_gene_symbol"].tolist() == ["KLF2", "KLF4"], "correlation r > 0 rows via `significant`"
    assert candidates["evidence_tier"].tolist() == ["motif+expressed_in_program"] * 2
    assert list(candidates.columns) == nct.CANDIDATE_TF_COLUMNS + ["motif_source"]
