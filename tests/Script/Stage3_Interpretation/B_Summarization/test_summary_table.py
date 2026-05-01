"""Unit tests for Interpretation.Summary_table.src.Compile_excel_sheet."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure the src directory is importable
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

from Interpretation.Summary_table.src import (
    compile_Program_loading_score_sheet_flat,
    Compile_GO_sheet,
    Compile_Geneset_sheet,
    Compile_Trait_sheet,
    Compile_Perturbation_sheet,
    Compile_Association_sheet,
    Compile_Explained_variance,
    Compile_Summary_sheet,
    check_program_name_match,
)
from Interpretation.Summary_table.src.Compile_excel_sheet import (
    Compute_regulator_zscore,
    softmax_with_temperature,
    compute_joint_distribution,
    compute_pointwise_mutual_information,
    get_program_info_Summary_cols,
    get_top_terms_Summary_cols,
    simple_Summary_cols,
    get_significant_programs,
    get_significant_programs_df,
    get_specificity_program,
    get_guide_cells_per_days,
    get_guide_mean_expr_per_day,
)


# ==========================================================================
# Test: compile_Program_loading_score_sheet_flat
# ==========================================================================

class TestProgramLoadingFlat:
    def test_returns_dataframe(self, synthetic_mdata):
        df = compile_Program_loading_score_sheet_flat(synthetic_mdata, num_gene=5)
        assert isinstance(df, pd.DataFrame)

    def test_shape(self, synthetic_mdata):
        n_programs = synthetic_mdata["cNMF"].n_vars
        num_gene = 5
        df = compile_Program_loading_score_sheet_flat(synthetic_mdata, num_gene=num_gene)
        assert df.shape == (n_programs, num_gene)

    def test_index_name(self, synthetic_mdata):
        df = compile_Program_loading_score_sheet_flat(synthetic_mdata, num_gene=5)
        assert df.index.name == "Program"

    def test_columns_are_ranks(self, synthetic_mdata):
        num_gene = 10
        df = compile_Program_loading_score_sheet_flat(synthetic_mdata, num_gene=num_gene)
        assert list(df.columns) == list(range(1, num_gene + 1))

    def test_genes_are_from_var_names(self, synthetic_mdata):
        df = compile_Program_loading_score_sheet_flat(synthetic_mdata, num_gene=5)
        gene_set = set(synthetic_mdata["cNMF"].uns["var_names"])
        for _, row in df.iterrows():
            for gene in row.dropna():
                assert gene in gene_set


# ==========================================================================
# Test: Compile_GO_sheet / Compile_Geneset_sheet / Compile_Trait_sheet
# ==========================================================================

class TestEnrichmentSheets:
    @pytest.fixture
    def go_file(self, output_dir):
        path = output_dir / "test_GO.txt"
        df = pd.DataFrame({
            "Term": ["GO:001", "GO:002"],
            "program_name": ["0", "1"],
            "Adjusted P-value": [0.001, 0.05],
            "Genes": ["A;B;C;D;E;F;G", "X;Y;Z"],
        })
        df.to_csv(path, sep="\t")
        return str(path)

    def test_compile_go_sheet(self, go_file):
        df = Compile_GO_sheet(go_file, gene_num=3)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "Term"
        # Genes should be truncated to 3
        for genes_str in df["Genes"]:
            assert len(genes_str.split(";")) <= 3

    def test_compile_geneset_sheet(self, go_file):
        df = Compile_Geneset_sheet(go_file, gene_num=2)
        assert isinstance(df, pd.DataFrame)
        for genes_str in df["Genes"]:
            assert len(genes_str.split(";")) <= 2

    def test_compile_trait_sheet(self, go_file):
        df = Compile_Trait_sheet(go_file, gene_num=5)
        assert isinstance(df, pd.DataFrame)


# ==========================================================================
# Test: Compile_Perturbation_sheet
# ==========================================================================

class TestPerturbationSheet:
    def test_concatenates_samples(self, perturbation_dir, sample_list):
        df = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        assert isinstance(df, pd.DataFrame)
        assert "Sample" in df.columns
        assert set(df["Sample"].unique()) == set(sample_list)

    def test_has_required_columns(self, perturbation_dir, sample_list):
        df = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        for col in ["target_name", "program_name", "log2FC", "adj_pval"]:
            assert col in df.columns

    def test_row_count(self, perturbation_dir, sample_list):
        df = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        # 5 targets x 3 programs x 3 samples = 45
        assert len(df) == 45


# ==========================================================================
# Test: Compile_Association_sheet
# ==========================================================================

class TestAssociationSheet:
    @pytest.fixture
    def association_file(self, output_dir):
        path = output_dir / "test_association.txt"
        df = pd.DataFrame({
            "program_name": [0, 1, 2],
            "chi2": [10.5, 20.3, 5.1],
            "pval": [0.001, 0.0001, 0.05],
        })
        df.to_csv(path, sep="\t")
        return str(path)

    def test_loads_and_reindexes(self, association_file):
        df = Compile_Association_sheet(association_file)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "program_name"


# ==========================================================================
# Test: Compile_Explained_variance
# ==========================================================================

class TestExplainedVariance:
    @pytest.fixture
    def variance_file(self, output_dir):
        path = output_dir / "test_variance.txt"
        df = pd.DataFrame({
            "program_name": [0, 1, 2],
            "variance_explained": [0.15, 0.10, 0.08],
        })
        df.to_csv(path, sep="\t", index=False)
        return str(path)

    def test_loads_correctly(self, variance_file):
        df = Compile_Explained_variance(variance_file)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "program_name"
        assert "variance_explained" in df.columns
        assert len(df) == 3


# ==========================================================================
# Test: Specificity score helpers
# ==========================================================================

class TestSpecificityHelpers:
    @pytest.fixture
    def perturb_df(self):
        """Small perturbation DataFrame for z-score / softmax tests."""
        rng = np.random.default_rng(7)
        rows = []
        for tgt in ["GeneA", "GeneB", "GeneC"]:
            for prog in ["0", "1", "2"]:
                rows.append({
                    "target_name": tgt,
                    "program_name": prog,
                    "log2FC": rng.normal(0, 1),
                    "adj_pval": rng.uniform(0, 0.1),
                })
        return pd.DataFrame(rows)

    def test_compute_regulator_zscore_shape(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        assert z.shape == (3, 3)  # 3 targets x 3 programs

    def test_compute_regulator_zscore_centered(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        # Median of each row should be close to 0
        row_medians = z.median(axis=1)
        assert (row_medians.abs() < 1e-6).all()

    def test_softmax_sums_to_one(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        w = softmax_with_temperature(z, T=1.0)
        total = w.values.sum()
        assert abs(total - 1.0) < 1e-6

    def test_softmax_all_positive(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        w = softmax_with_temperature(z, T=0.5)
        assert (w.values >= 0).all()

    def test_joint_distribution_sums_to_one(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        w = softmax_with_temperature(z, T=1.0)
        P_tp, P_t, P_p = compute_joint_distribution(w)
        assert abs(P_tp.values.sum() - 1.0) < 1e-6
        assert abs(P_t.sum() - 1.0) < 1e-6
        assert abs(P_p.sum() - 1.0) < 1e-6

    def test_pmi_shape(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        w = softmax_with_temperature(z, T=1.0)
        P_tp, P_t, P_p = compute_joint_distribution(w)
        pmi = compute_pointwise_mutual_information(P_tp, P_t, P_p)
        assert pmi.shape == (3, 3)

    def test_pmi_is_finite(self, perturb_df):
        z = Compute_regulator_zscore(perturb_df)
        w = softmax_with_temperature(z, T=1.0)
        P_tp, P_t, P_p = compute_joint_distribution(w)
        pmi = compute_pointwise_mutual_information(P_tp, P_t, P_p)
        assert np.isfinite(pmi.values).all()


# ==========================================================================
# Test: get_specificity_program (end-to-end with files)
# ==========================================================================

class TestGetSpecificityProgram:
    def test_returns_dataframe(self, perturbation_dir, sample_list):
        df = get_specificity_program(perturbation_dir, Sample=sample_list, T=0.5)
        assert isinstance(df, pd.DataFrame)

    def test_columns_per_sample(self, perturbation_dir, sample_list):
        df = get_specificity_program(perturbation_dir, Sample=sample_list, T=0.5)
        for samp in sample_list:
            assert f"top 5 specific programs (FDR < 0.1) {samp}" in df.columns
            assert f"top 5 specificity scores (FDR < 0.1) {samp}" in df.columns


# ==========================================================================
# Test: get_significant_programs / get_significant_programs_df
# ==========================================================================

class TestSignificantPrograms:
    def test_returns_dict(self, perturbation_dir, sample_list):
        result = get_significant_programs(perturbation_dir, Sample=sample_list)
        assert isinstance(result, dict)

    def test_df_has_count_columns(self, perturbation_dir, sample_list):
        df = get_significant_programs_df(perturbation_dir, Sample=sample_list)
        assert isinstance(df, pd.DataFrame)
        for samp in sample_list:
            assert f"# programs {samp}" in df.columns


# ==========================================================================
# Test: MuData helper functions
# ==========================================================================

class TestMuDataHelpers:
    def test_get_guide_cells_per_days(self, synthetic_mdata):
        df = get_guide_cells_per_days(synthetic_mdata, categorical_key="sample")
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "target_name"
        # Should have a column per unique sample
        n_samples = synthetic_mdata["rna"].obs["sample"].nunique()
        assert df.shape[1] == n_samples
        # All counts should be non-negative
        assert (df.values >= 0).all()

    def test_get_guide_mean_expr_per_day(self, synthetic_mdata):
        df = get_guide_mean_expr_per_day(synthetic_mdata, categorical_key="sample")
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "target_name"
        n_samples = synthetic_mdata["rna"].obs["sample"].nunique()
        assert df.shape[1] == n_samples

    def test_get_program_info_summary_cols(self, synthetic_mdata):
        df = get_program_info_Summary_cols(synthetic_mdata, categorical_key="sample")
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "program_name"
        n_programs = synthetic_mdata["cNMF"].n_vars
        assert len(df) == n_programs


# ==========================================================================
# Test: get_top_terms_Summary_cols
# ==========================================================================

class TestTopTermsSummaryCols:
    def test_returns_dataframe(self, synthetic_go_df, synthetic_geneset_df):
        df = get_top_terms_Summary_cols(synthetic_go_df, synthetic_geneset_df)
        assert isinstance(df, pd.DataFrame)
        assert "top10_enriched_go_terms" in df.columns
        assert "top10_enriched_genesets" in df.columns

    def test_handles_none_inputs(self):
        df = get_top_terms_Summary_cols(None, None)
        assert isinstance(df, pd.DataFrame)


# ==========================================================================
# Test: check_program_name_match
# ==========================================================================

class TestCheckProgramNameMatch:
    def test_no_warning_on_match(self, synthetic_mdata, capsys):
        df = pd.DataFrame({"program_name": ["0", "1", "2"], "value": [1, 2, 3]})
        check_program_name_match(synthetic_mdata, [df])
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_warning_on_mismatch(self, synthetic_mdata, capsys):
        df = pd.DataFrame({"program_name": ["99", "100"], "value": [1, 2]})
        check_program_name_match(synthetic_mdata, [df])
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_skips_none(self, synthetic_mdata):
        # Should not raise
        check_program_name_match(synthetic_mdata, [None, None])


# ==========================================================================
# Test: Compile_Summary_sheet (integration)
# ==========================================================================

class TestCompileSummarySheet:
    def test_returns_dataframe(
        self, synthetic_mdata, synthetic_go_df, synthetic_geneset_df,
        synthetic_program_loading_flat, synthetic_explained_variance_df,
        perturbation_dir, sample_list,
    ):
        df_perturbation = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        df = Compile_Summary_sheet(
            mdata=synthetic_mdata,
            df_GO=synthetic_go_df,
            df_Geneset=synthetic_geneset_df,
            df_Perturbation=df_perturbation,
            df_Program_loading=synthetic_program_loading_flat,
            df_Explained_Variance=synthetic_explained_variance_df,
            Sample=sample_list,
            categorical_key="sample",
        )
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "program_name"
        assert len(df) == synthetic_mdata["cNMF"].n_vars

    def test_has_manual_columns(
        self, synthetic_mdata, synthetic_go_df, synthetic_geneset_df,
        synthetic_program_loading_flat, synthetic_explained_variance_df,
        perturbation_dir, sample_list,
    ):
        df_perturbation = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        df = Compile_Summary_sheet(
            mdata=synthetic_mdata,
            df_GO=synthetic_go_df,
            df_Geneset=synthetic_geneset_df,
            df_Perturbation=df_perturbation,
            df_Program_loading=synthetic_program_loading_flat,
            df_Explained_Variance=synthetic_explained_variance_df,
            Sample=sample_list,
            categorical_key="sample",
        )
        for col in ["manual_annotation_label", "manual_timepoint", "Notes", "Automatic Timepoint"]:
            assert col in df.columns

    def test_works_with_none_enrichment(
        self, synthetic_mdata, synthetic_program_loading_flat,
        synthetic_explained_variance_df, perturbation_dir, sample_list,
    ):
        df_perturbation = Compile_Perturbation_sheet(perturbation_dir, Sample=sample_list)
        df = Compile_Summary_sheet(
            mdata=synthetic_mdata,
            df_GO=None,
            df_Geneset=None,
            df_Perturbation=df_perturbation,
            df_Program_loading=synthetic_program_loading_flat,
            df_Explained_Variance=synthetic_explained_variance_df,
            Sample=sample_list,
            categorical_key="sample",
        )
        assert isinstance(df, pd.DataFrame)
