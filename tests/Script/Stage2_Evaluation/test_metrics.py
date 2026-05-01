"""
Unit tests for the cNMF Evaluation pipeline.

Tests run against real inference output (default: tests/output/torch-cNMF)
and use synthetic data where external APIs or heavy I/O would be needed.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/src
    python -m pytest tests/Script/Stage2_Evaluation/test_evaluation.py -v

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
# Tests for association_categorical.py
# ===========================================================================

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

    def test_compute_categorical_not_inplace_dunn(self, mdata_copy, eval_output_dir):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        results_df, posthoc_df = compute_categorical_association(
            mdata_copy, prog_key="cNMF", categorical_key="batch",
            test="dunn", n_jobs=1, inplace=False
        )
        assert "batch_kruskall_wallis_stat" in results_df.columns
        assert "batch_kruskall_wallis_pval" in results_df.columns
        assert "program_name" in posthoc_df.columns
        assert len(results_df) == 5
        results_df.to_csv(os.path.join(eval_output_dir, "5_categorical_association_results.csv"))
        posthoc_df.to_csv(os.path.join(eval_output_dir, "5_categorical_association_posthoc.csv"))

    def test_compute_categorical_pearson_one_vs_all(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        results_df, posthoc_df = compute_categorical_association(
            mdata_copy, prog_key="cNMF", categorical_key="batch",
            test="pearsonr", mode="one_vs_all", n_jobs=1, inplace=False
        )
        assert "batch_kruskall_wallis_stat" in results_df.columns
        assert posthoc_df.shape[0] > 0


# ===========================================================================
# Tests for association_perturbation.py
# ===========================================================================

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

    def test_compute_perturbation_not_inplace(self, mdata_copy, eval_output_dir):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            guide_names_key="guide_names",
            guide_targets_key="guide_targets",
            guide_assignments_key="guide_assignment",
            reference_targets=["non-targeting"],
            collapse_targets=True,
            n_jobs=1,
            inplace=False,
            FDR_method="BH"
        )
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"target_name", "program_name", "pval", "adj_pval", "log2FC",
                         "stat", "ref_mean", "test_mean"}
        assert expected_cols.issubset(set(result.columns))
        assert all(result["pval"].between(0, 1))
        assert all(np.isfinite(result["adj_pval"]))
        result.to_csv(os.path.join(eval_output_dir, "5_perturbation_association.csv"), index=False)

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


# ===========================================================================
# Tests for enrichment_geneset.py
# ===========================================================================

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

    def test_get_program_gene_loadings_all(self, mdata_copy, eval_output_dir):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import get_program_gene_loadings
        loadings = get_program_gene_loadings(
            mdata_copy, prog_key="cNMF", data_key="rna", gene_names_key="symbol"
        )
        assert loadings.shape == (17495, 5)
        assert loadings.index.name == "gene_names"
        loadings.to_csv(os.path.join(eval_output_dir, "5_gene_loadings.csv"))

    def test_get_program_gene_loadings_single(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import get_program_gene_loadings
        prog_name = mdata_copy["cNMF"].var_names[0]
        loadings = get_program_gene_loadings(
            mdata_copy, prog_key="cNMF", prog_nam=prog_name,
            data_key="rna", gene_names_key="symbol"
        )
        assert loadings.shape[1] == 1
        assert loadings.columns[0] == prog_name

    @patch("Evaluation.src.enrichment_geneset.gp")
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

    @patch("Evaluation.src.enrichment_geneset.gp")
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


# ===========================================================================
# Tests for enrichment_trait.py
# ===========================================================================

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


# ===========================================================================
# Tests for enrichment_motif.py
# ===========================================================================

class TestMotifEnrichment:

    def test_read_loci_valid(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_motif import read_loci
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("chromosome\tstart\tend\tseq_name\tseq_class\tseq_score\tgene_name\n")
            f.write("chr1\t100\t200\tseq1\tpromoter\t0.9\tGENE1\n")
            f.write("chr1\t300\t400\tseq2\tpromoter\t0.8\tGENE2\n")
            fname = f.name
        try:
            loci = read_loci(fname)
            assert len(loci) == 2
            assert "gene_name" in loci.columns
        finally:
            os.unlink(fname)

    def test_read_loci_missing_column(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_motif import read_loci
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("chromosome\tstart\tend\n")
            f.write("chr1\t100\t200\n")
            fname = f.name
        try:
            with pytest.raises(ValueError, match="not formatted correctly"):
                read_loci(fname)
        finally:
            os.unlink(fname)

    def test_compute_motif_instances(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_motif import compute_motif_instances
        motif_df = pd.DataFrame({
            "gene_name": ["G1", "G1", "G2", "G2", "G3"],
            "motif_name": ["M1", "M1", "M2", "M1", "M2"],
            "adj_pval": [0.01, 0.02, 0.03, 0.01, 0.04]
        })
        gene_names = np.array(["G1", "G2", "G3", "G4"])
        result = compute_motif_instances(motif_df, gene_names=gene_names, sig=0.05)
        assert result.shape[0] == 4
        assert set(result.columns) == {"M1", "M2"}
        assert pd.isna(result.loc["G4", "M1"])

    def test_perform_correlation_motif(self):
        from Stage2_Evaluation.A_Metrics.src.enrichment_motif import perform_correlation as motif_corr
        rng = np.random.default_rng(42)
        n_genes, n_motifs, n_progs = 50, 3, 2
        motif_count_df = pd.DataFrame(
            rng.integers(0, 5, size=(n_genes, n_motifs)),
            index=[f"G{i}" for i in range(n_genes)],
            columns=[f"M{j}" for j in range(n_motifs)]
        )
        prog_genes = pd.DataFrame(
            rng.random((n_progs, n_genes)),
            index=[f"prog_{i}" for i in range(n_progs)],
            columns=[f"G{i}" for i in range(n_genes)]
        )
        stat_df = pd.DataFrame(np.nan, index=[f"prog_{i}" for i in range(n_progs)],
                                columns=[f"M{j}" for j in range(n_motifs)])
        pval_df = stat_df.copy()
        motif_corr(motif_count_df, prog_genes, stat_df, pval_df,
                   motif_idx=0, prog_idx=0, correlation="pearsonr")
        assert np.isfinite(float(stat_df.iloc[0, 0]))
        assert 0 <= float(pval_df.iloc[0, 0]) <= 1


# ===========================================================================
# Tests for explained_variance.py
# ===========================================================================

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


# ===========================================================================
# Tests for Self_regulating_programs.py
# ===========================================================================

class TestSelfRegulatingPrograms:

    def test_get_top_genes_real_data(self, spectra_score_path):
        from Stage2_Evaluation.A_Metrics.src.Self_regulating_programs import get_top_genes_per_program
        top = get_top_genes_per_program(spectra_score_path, top_n=10)
        assert "0" in top
        assert "4" in top
        for prog_key in top:
            assert len(top[prog_key]) == 10

    def test_get_top_genes_with_rename(self):
        from Stage2_Evaluation.A_Metrics.src.Self_regulating_programs import get_top_genes_per_program
        rng = np.random.default_rng(42)
        ens_ids = [f"ENS{i:05d}" for i in range(50)]
        gene_dict = {ens: f"GENE{i}" for i, ens in enumerate(ens_ids)}
        data = rng.random((2, 50))
        df = pd.DataFrame(data, index=[1, 2], columns=ens_ids)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            df.to_csv(f, sep="\t")
            fname = f.name
        try:
            top = get_top_genes_per_program(fname, top_n=5, gene_name_dict=gene_dict)
            for gene in top["0"]:
                assert gene.startswith("GENE")
        finally:
            os.unlink(fname)

    def test_find_autoregulatory_hits(self):
        from Stage2_Evaluation.A_Metrics.src.Self_regulating_programs import find_autoregulatory_hits
        perturb_df = pd.DataFrame({
            "target_name": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            "program_name": [0, 0, 1, 1, 2],
            "log2FC": [-0.5, 0.3, -0.8, 0.2, -1.0],
            "p-value": [0.001, 0.1, 0.01, 0.5, 0.001],
            "adj_pval": [0.01, 0.2, 0.04, 0.8, 0.02]
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            perturb_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            top_genes = {
                "0": {"GENE1", "GENE2", "GENEX"},
                "1": {"GENE3", "GENEY"},
                "2": {"GENEZ"}
            }
            hits = find_autoregulatory_hits(fname, top_genes, pval_thresh=0.05)
            assert "GENE1" in hits["target_name"].values
            assert "GENE3" in hits["target_name"].values
            assert "GENE5" not in hits["target_name"].values
            assert "GENE2" not in hits["target_name"].values
        finally:
            os.unlink(fname)

    def test_find_autoregulatory_no_hits(self):
        from Stage2_Evaluation.A_Metrics.src.Self_regulating_programs import find_autoregulatory_hits
        perturb_df = pd.DataFrame({
            "target_name": ["GENE1"],
            "program_name": [0],
            "log2FC": [-0.5],
            "p-value": [0.5],
            "adj_pval": [0.9]
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            perturb_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            top_genes = {"0": {"GENE1"}}
            hits = find_autoregulatory_hits(fname, top_genes, pval_thresh=0.05)
            assert len(hits) == 0
        finally:
            os.unlink(fname)


# ===========================================================================
# Tests for Utilities.py
# ===========================================================================

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


# ===========================================================================
# Integration test
# ===========================================================================

class TestIntegration:

    def test_categorical_then_perturbation(self, mdata_copy, eval_output_dir):
        """Run categorical + perturbation on same real MuData."""
        from Stage2_Evaluation.A_Metrics.src.association_categorical import compute_categorical_association
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association

        cat_results, cat_posthoc = compute_categorical_association(
            mdata_copy, prog_key="cNMF", categorical_key="batch",
            test="dunn", n_jobs=1, inplace=False
        )
        assert len(cat_results) == 5

        perturb_results = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert len(perturb_results) > 0
        assert "adj_pval" in perturb_results.columns

        cat_results.to_csv(os.path.join(eval_output_dir, "5_integration_categorical.csv"))
        cat_posthoc.to_csv(os.path.join(eval_output_dir, "5_integration_posthoc.csv"), index=False)
        perturb_results.to_csv(os.path.join(eval_output_dir, "5_integration_perturbation.csv"), index=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
