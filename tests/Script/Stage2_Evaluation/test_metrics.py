"""
Unit tests for the cNMF Evaluation pipeline.

Tests run against real inference output (default: tests/output/torch-cNMF)
and use synthetic data where external APIs or heavy I/O would be needed.

Tests are parametrized over K values (5, 10, 15) where applicable,
saving results into Evaluation/{K}_2_0/ subdirectories.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src
    python -m pytest tests/Script/Stage2_Evaluation/test_metrics.py -v

    # Custom inference path:
    python -m pytest tests/Script/Stage2_Evaluation/ -v --inference-path /path/to/Inference/
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import mudata as mu
from scipy import sparse


# ===========================================================================
# Per-K parametrized tests (K=5, 10, 15)
# Results saved to Evaluation/{K}_2_0/
# ===========================================================================

class TestCategoricalPerK:

    def test_compute_categorical_dunn(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        k = mdata_copy_per_k.uns["test_k"]
        results_df, posthoc_df = compute_categorical_association(
            mdata_copy_per_k, prog_key="cNMF", categorical_key="batch",
            test="dunn", n_jobs=1, inplace=False
        )
        assert "batch_kruskall_wallis_stat" in results_df.columns
        assert len(results_df) == mdata_copy_per_k["cNMF"].n_vars
        results_df.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_categorical_association_results.txt"), sep="\t")
        posthoc_df.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_categorical_association_posthoc.txt"), sep="\t", index=False)


class TestPerturbationPerK:

    def test_compute_perturbation(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        k = mdata_copy_per_k.uns["test_k"]
        result = compute_perturbation_association(
            mdata_copy_per_k, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert isinstance(result, pd.DataFrame)
        assert all(result["pval"].between(0, 1))
        result.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_perturbation_association_results.txt"), sep="\t", index=False)


class TestGenesetPerK:

    def test_get_gene_loadings(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import get_program_gene_loadings
        k = mdata_copy_per_k.uns["test_k"]
        loadings = get_program_gene_loadings(
            mdata_copy_per_k, prog_key="cNMF", data_key="rna", gene_names_key="symbol"
        )
        assert loadings.shape[1] == mdata_copy_per_k["cNMF"].n_vars
        loadings.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_gene_loadings.txt"), sep="\t")

    def test_reactome_enrichment(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import compute_geneset_enrichment
        k = mdata_copy_per_k.uns["test_k"]
        result = compute_geneset_enrichment(
            mdata_copy_per_k, prog_key="cNMF", data_key="rna", prog_name=None,
            organism="human", library="Reactome_2022", method="fisher",
            database="enrichr", n_top=300, n_jobs=1,
            inplace=False, user_geneset=None, use_loadings_gene=False,
            gene_names_key="symbol"
        )
        assert isinstance(result, pd.DataFrame)
        result.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_geneset_enrichment.txt"), sep="\t", index=False)


class TestExplainedVariancePerK:

    def test_compute_explained_variance(self, mdata_copy_per_k, eval_output_dir_per_k, cnmf_obj, X_norm):
        from Stage2_Evaluation.A_Metrics.src.explained_variance import compute_explained_variance
        k = mdata_copy_per_k.uns["test_k"]
        program_names = mdata_copy_per_k["cNMF"].var_names
        compute_explained_variance(cnmf_obj, X_norm, k,
                                   output_folder=eval_output_dir_per_k,
                                   thre='2.0',
                                   program_name=program_names)
        ev_path = os.path.join(eval_output_dir_per_k, f"{k}_Explained_Variance.txt")
        summary_path = os.path.join(eval_output_dir_per_k, f"{k}_Explained_Variance_Summary.txt")
        assert os.path.isfile(ev_path)
        assert os.path.isfile(summary_path)
        ev_df = pd.read_csv(ev_path, sep="\t")
        assert len(ev_df) == k
        assert all(np.isfinite(ev_df["VarianceExplained"]))


class TestGOEnrichmentPerK:

    def test_go_enrichment(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import compute_geneset_enrichment
        k = mdata_copy_per_k.uns["test_k"]
        result = compute_geneset_enrichment(
            mdata_copy_per_k, prog_key="cNMF", data_key="rna", prog_name=None,
            organism="human", library="GO_Biological_Process_2023", method="fisher",
            database="enrichr", n_top=300, n_jobs=1,
            inplace=False, user_geneset=None, use_loadings_gene=False,
            gene_names_key="symbol"
        )
        assert isinstance(result, pd.DataFrame)
        result.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_GO_term_enrichment.txt"), sep="\t", index=False)


class TestTraitEnrichmentPerK:

    def test_trait_enrichment(self, mdata_copy_per_k, eval_output_dir_per_k):
        from Stage2_Evaluation.A_Metrics.src.enrichment_trait import compute_trait_enrichment
        k = mdata_copy_per_k.uns["test_k"]
        gwas_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "src", "Stage2_Evaluation", "Resources", "OpenTargets_L2G_Filtered.csv.gz"
        )
        if not os.path.isfile(gwas_path):
            pytest.skip(f"GWAS data not found: {gwas_path}")
        result = compute_trait_enrichment(
            mdata_copy_per_k, gwas_data=gwas_path,
            prog_key="cNMF", prog_name=None, data_key="rna",
            library="OT_GWAS", n_jobs=1, inplace=False,
            key_column="trait_efos", gene_column="gene_name",
            method="fisher", n_top=300, use_loadings_gene=True,
            gene_names_key="symbol"
        )
        assert isinstance(result, pd.DataFrame)
        result.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_trait_enrichment.txt"), sep="\t", index=False)


class TestIntegrationPerK:

    def test_categorical_then_perturbation(self, mdata_copy_per_k, eval_output_dir_per_k):
        """Run categorical + perturbation per K on real MuData."""
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        k = mdata_copy_per_k.uns["test_k"]

        cat_results, cat_posthoc = compute_categorical_association(
            mdata_copy_per_k, prog_key="cNMF", categorical_key="batch",
            test="dunn", n_jobs=1, inplace=False
        )
        assert len(cat_results) > 0

        perturb_results = compute_perturbation_association(
            mdata_copy_per_k, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert len(perturb_results) > 0

        cat_results.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_integration_categorical.txt"), sep="\t")
        cat_posthoc.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_integration_posthoc.txt"), sep="\t", index=False)
        perturb_results.to_csv(os.path.join(eval_output_dir_per_k, f"{k}_integration_perturbation.txt"), sep="\t", index=False)


# ===========================================================================
# Unit / edge-case tests (K=5 only, no file output)
# ===========================================================================

# --- association_categorical.py ---

class TestCategoricalAssociation:

    def test_kw_single_cell(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import perform_kruskall_wallis
        mdata = mdata_copy
        prog_name = mdata["cNMF"].var_names[0]
        perform_kruskall_wallis(mdata, prog_key="cNMF", prog_name=prog_name,
                                categorical_key="batch", pseudobulk_key=None)
        stat = mdata["cNMF"].var.loc[prog_name, "batch_kruskall_wallis_stat"]
        pval = mdata["cNMF"].var.loc[prog_name, "batch_kruskall_wallis_pval"]
        assert np.isfinite(stat)
        assert 0 <= pval <= 1

    def test_kw_pseudobulk(self, mdata_copy):
        """Pseudobulk KW using sample_id (3 replicates per batch)."""
        from Stage2_Evaluation.A_Metrics.src.association_categorical import perform_kruskall_wallis
        mdata = mdata_copy
        prog_name = mdata["cNMF"].var_names[0]
        perform_kruskall_wallis(mdata, prog_key="cNMF", prog_name=prog_name,
                                categorical_key="batch",
                                pseudobulk_key="sample_id")
        store_key = "batch_sample_id"
        stat = mdata["cNMF"].var.loc[prog_name, f"{store_key}_kruskall_wallis_stat"]
        pval = mdata["cNMF"].var.loc[prog_name, f"{store_key}_kruskall_wallis_pval"]
        assert np.isfinite(stat)
        assert 0 <= pval <= 1

    def test_kw_pseudobulk_too_few_replicates(self, mdata_copy):
        """biological_sample has only 2 reps per batch -> should raise."""
        from Stage2_Evaluation.A_Metrics.src.association_categorical import perform_kruskall_wallis
        mdata = mdata_copy
        prog_name = mdata["cNMF"].var_names[0]
        with pytest.raises(ValueError, match="less than 3 replicates"):
            perform_kruskall_wallis(mdata, prog_key="cNMF", prog_name=prog_name,
                                    categorical_key="batch",
                                    pseudobulk_key="biological_sample")

    def test_perform_correlation_one_vs_all(self):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import perform_correlation
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": rng.choice(["A", "B", "C"], 200),
            "score": rng.normal(0, 1, 200)
        })
        result_list = []
        perform_correlation(df, group_col="group", val_col="score",
                            correlation="pearsonr", mode="one_vs_all", df=result_list)
        assert len(result_list) == 1
        res = result_list[0]
        assert any("pearsonr_pval" in c for c in res.columns)

    def test_perform_correlation_one_vs_one(self):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import perform_correlation
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": rng.choice(["A", "B", "C"], 200),
            "score": rng.normal(0, 1, 200)
        })
        pvals = perform_correlation(df, group_col="group", val_col="score",
                                    correlation="pearsonr", mode="one_vs_one")
        assert pvals.shape == (3, 3)

    def test_compute_categorical_pearson_one_vs_all(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        results_df, posthoc_df = compute_categorical_association(
            mdata_copy, prog_key="cNMF", categorical_key="batch",
            test="pearsonr", mode="one_vs_all", n_jobs=1, inplace=False
        )
        assert "batch_kruskall_wallis_stat" in results_df.columns
        assert posthoc_df.shape[0] > 0


# --- association_perturbation.py ---

class TestPerturbationAssociation:

    def test_get_guide_metadata(self, test_mdata):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import get_guide_metadata
        gm = get_guide_metadata(test_mdata, prog_key="cNMF")
        assert "Target" in gm.columns
        assert len(gm) == 1859
        assert "non-targeting" in gm["Target"].values

    def test_compute_perturbation_single_unit(self):
        """Test the inner per-program Mann-Whitney function with synthetic data."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association_
        rng = np.random.default_rng(42)
        test_data = ad.AnnData(
            X=sparse.csr_matrix(rng.random((30, 3))),
            var=pd.DataFrame(index=["p0", "p1", "p2"])
        )
        ref_data = ad.AnnData(
            X=sparse.csr_matrix(rng.random((50, 3))),
            var=pd.DataFrame(index=["p0", "p1", "p2"])
        )
        results = []
        compute_perturbation_association_(test_data, ref_data, "p0", "target_X", results)
        assert len(results) == 1
        row = results[0]
        assert row[0] == "target_X"
        assert row[1] == "p0"
        assert np.isfinite(row[5])  # stat
        assert 0 <= row[6] <= 1     # pval

    def test_pval_sanitization(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert all(np.isfinite(result["pval"]))
        assert all(result["pval"] >= 0)
        assert all(result["pval"] <= 1)


# --- enrichment_geneset.py ---

class TestGenesetEnrichment:

    def test_create_geneset_dict(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import create_geneset_dict
        df = pd.DataFrame({
            "trait": ["T1", "T1", "T2", "T2", "T2"],
            "gene": ["A", "B", "C", "D", "E"]
        })
        gmt = create_geneset_dict(df, key_column="trait", gene_column="gene")
        assert set(gmt.keys()) == {"T1", "T2"}
        assert gmt["T1"] == ["A", "B"]
        assert gmt["T2"] == ["C", "D", "E"]

    def test_create_geneset_dict_empty(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import create_geneset_dict
        df = pd.DataFrame({"trait": [], "gene": []})
        gmt = create_geneset_dict(df, key_column="trait", gene_column="gene")
        assert gmt == {}

    def test_get_program_gene_loadings_single(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import get_program_gene_loadings
        prog_name = mdata_copy["cNMF"].var_names[0]
        loadings = get_program_gene_loadings(
            mdata_copy, prog_key="cNMF", prog_nam=prog_name,
            data_key="rna", gene_names_key="symbol"
        )
        assert loadings.shape[1] == 1
        assert loadings.columns[0] == prog_name

    @patch("Stage2_Evaluation.A_Metrics.src.enrichment_geneset.gp")
    def test_perform_prerank(self, mock_gp):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import perform_prerank
        mock_res = pd.DataFrame({
            "Name": ["prerank"],
            "Term": ["PATHWAY_A"],
            "ES": [0.5],
            "NES": [1.2],
            "NOM p-val": [0.01],
            "FDR q-val": [0.05],
            "FWER p-val": [0.1],
            "Gene %": ["45%"],
            "Tag %": ["10/50"],
            "Lead_genes": ["GENE1;GENE2"]
        })
        mock_gp.prerank.return_value = MagicMock(res2d=mock_res)
        loadings = pd.DataFrame({"prog_0": np.random.rand(50)},
                                 index=[f"GENE{i}" for i in range(50)])
        result = perform_prerank(loadings, geneset={"PATHWAY_A": ["GENE1", "GENE2"]})
        assert "program_name" in result.columns
        assert "tag_before" in result.columns
        assert result["tag_before"].iloc[0] == 10
        assert result["tag_after"].iloc[0] == 50

    @patch("Stage2_Evaluation.A_Metrics.src.enrichment_geneset.gp")
    def test_perform_fisher_enrich(self, mock_gp):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import perform_fisher_enrich
        mock_res = pd.DataFrame({
            "Gene_set": ["lib"],
            "Term": ["PATHWAY_B"],
            "P-value": [0.01],
            "Adjusted P-value": [0.05],
            "Odds Ratio": [3.0],
            "Combined Score": [10.0],
            "Genes": ["GENE1;GENE2"],
            "Overlap": ["2/10"]
        })
        mock_gp.enrich.return_value = MagicMock(res2d=mock_res)
        loadings = pd.DataFrame({"prog_0": np.random.rand(50)},
                                 index=[f"GENE{i}" for i in range(50)])
        geneset = {"PATHWAY_B": [f"GENE{i}" for i in range(10)]}
        result = perform_fisher_enrich(loadings, geneset=geneset, n_top=20)
        assert "program_name" in result.columns
        assert "overlap_numerator" in result.columns
        assert result["overlap_numerator"].iloc[0] == 2

    def test_insert_enrichment(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import insert_enrichment
        prog_names = mdata_copy["cNMF"].var_names.tolist()
        df = pd.DataFrame({
            "program_name": [prog_names[0], prog_names[0], prog_names[1], prog_names[1]],
            "Term": ["GO_A", "GO_B", "GO_A", "GO_B"],
            "pval": [0.01, 0.05, 0.1, 0.2],
            "score": [10, 5, 3, 1]
        })
        insert_enrichment(mdata_copy, df, library="test_lib", prog_key="cNMF",
                          geneset_index="Term", program_index="program_name")
        assert "pval_test_lib" in mdata_copy["cNMF"].varm
        assert "score_test_lib" in mdata_copy["cNMF"].varm
        assert "gsea_varmap_test_lib" in mdata_copy["cNMF"].uns


# --- enrichment_trait.py ---

class TestTraitEnrichment:

    def test_process_json_format_valid(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_trait import process_json_format_l2g_columns
        row = pd.Series({"efos": "[{'element': 'EFO_001'}, {'element': 'EFO_002'}]"})
        result = process_json_format_l2g_columns(row, "efos")
        assert result == "EFO_001, EFO_002"

    def test_process_json_format_invalid(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_trait import process_json_format_l2g_columns
        row = pd.Series({"efos": "no brackets here"})
        result = process_json_format_l2g_columns(row, "efos")
        assert result is None

    def test_process_enrichment_data(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_trait import process_enrichment_data
        enrich_df = pd.DataFrame({
            "Term": ["EFO_001", "EFO_002"],
            "P-value": [0.001, 0.5],
            "program_name": ["prog_0", "prog_1"],
            "Genes": ["A;B", "C;D"]
        })
        meta_df = pd.DataFrame({
            "trait_efos": ["EFO_001", "EFO_002"],
            "trait_category": ["immune", "cardio"],
            "trait_reported": ["asthma", "heart disease"],
            "study_id": ["S1", "S2"],
            "pmid": ["123", "456"]
        })
        result = process_enrichment_data(enrich_df, meta_df,
                                          pval_col="P-value",
                                          enrich_geneset_id_col="Term",
                                          metadata_geneset_id_col="trait_efos")
        assert "-log10(P-value)" in result.columns
        assert len(result) == 2
        assert result["-log10(P-value)"].max() > 2.0

    def test_process_enrichment_data_zero_pval(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_trait import process_enrichment_data
        enrich_df = pd.DataFrame({
            "Term": ["EFO_001", "EFO_002"],
            "P-value": [0.0, 0.01],
            "program_name": ["prog_0", "prog_1"],
            "Genes": ["A", "B"]
        })
        meta_df = pd.DataFrame({
            "trait_efos": ["EFO_001", "EFO_002"],
            "trait_category": ["immune", "cardio"],
            "trait_reported": ["asthma", "heart"],
            "study_id": ["S1", "S2"],
            "pmid": ["123", "456"]
        })
        result = process_enrichment_data(enrich_df, meta_df,
                                          pval_col="P-value",
                                          enrich_geneset_id_col="Term",
                                          metadata_geneset_id_col="trait_efos")
        assert all(np.isfinite(result["-log10(P-value)"]))


# --- explained_variance.py ---

class TestExplainedVariance:

    def test_compute_Var(self):
        from Stage2_Evaluation.A_Metrics.src.explained_variance import compute_Var
        X = np.array([[1, 2], [3, 4], [5, 6]])
        var = compute_Var(X)
        expected = np.var([1, 3, 5], ddof=1) + np.var([2, 4, 6], ddof=1)
        assert np.isclose(var, expected)

    def test_compute_Var_single_column(self):
        from Stage2_Evaluation.A_Metrics.src.explained_variance import compute_Var
        X = np.array([[1], [2], [3]])
        assert np.isclose(compute_Var(X), np.var([1, 2, 3], ddof=1))

    def test_computeVarianceExplained_numpy(self):
        from Stage2_Evaluation.A_Metrics.src.explained_variance import computeVarianceExplained, compute_Var
        rng = np.random.default_rng(42)
        X = rng.random((100, 10))
        H = rng.random((3, 10))
        Var_X = compute_Var(X)
        ve = computeVarianceExplained(X, H, Var_X, 0)
        assert np.isfinite(ve)

    def test_computeVarianceExplained_dataframe(self):
        from Stage2_Evaluation.A_Metrics.src.explained_variance import computeVarianceExplained, compute_Var
        rng = np.random.default_rng(42)
        X = rng.random((100, 10))
        H_df = pd.DataFrame(rng.random((3, 10)))
        Var_X = compute_Var(X)
        ve = computeVarianceExplained(X, H_df, Var_X, 1)
        assert np.isfinite(ve)


# --- Utilities.py ---

class TestUtilities:

    def test_rename_gene_dictionary(self):
        from Stage2_Evaluation.A_Metrics.src.Utilities import rename_gene_dictionary
        rng = np.random.default_rng(42)
        n_cells, n_genes = 10, 5
        ens_ids = [f"ENS{i:05d}" for i in range(n_genes)]
        gene_names = [f"GENE{i}" for i in range(n_genes)]
        rna_ad = ad.AnnData(
            X=sparse.csr_matrix(rng.random((n_cells, n_genes))),
            var=pd.DataFrame(index=ens_ids),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        )
        mdata = mu.MuData({"rna": rna_ad})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("ensembl_id\tgene\n")
            for ens, gene in zip(ens_ids, gene_names):
                f.write(f"{ens}\t{gene}\n")
            fname = f.name
        try:
            rename_gene_dictionary(mdata, fname)
            assert list(mdata["rna"].var_names) == gene_names
        finally:
            os.unlink(fname)

    def test_rename_gene_dictionary_partial(self):
        """Unmapped IDs stay as-is."""
        from Stage2_Evaluation.A_Metrics.src.Utilities import rename_gene_dictionary
        rng = np.random.default_rng(42)
        n_cells, n_genes = 10, 5
        ens_ids = [f"ENS{i:05d}" for i in range(n_genes)]
        rna_ad = ad.AnnData(
            X=sparse.csr_matrix(rng.random((n_cells, n_genes))),
            var=pd.DataFrame(index=ens_ids),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
        )
        mdata = mu.MuData({"rna": rna_ad})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("ensembl_id\tgene\n")
            for i in range(3):
                f.write(f"ENS{i:05d}\tGENE{i}\n")
            fname = f.name
        try:
            rename_gene_dictionary(mdata, fname)
            expected = ["GENE0", "GENE1", "GENE2", "ENS00003", "ENS00004"]
            assert list(mdata["rna"].var_names) == expected
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
