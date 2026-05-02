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
            test="dunn", n_jobs=1, inplace=False, pseudobulk_key=None
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
            reference_targets=["non-targeting"],pseudobulk=False,
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

# --- association_perturbation.py: guide annotation table vs key ---

class TestPerturbationReferenceTargets:
    """Test that using a guide annotation table produces the same reference_targets
    as passing the key string directly."""

    def test_annotation_table_extracts_nontargeting(self, guide_annotation_path):
        """The annotation table with targeting=FALSE should yield non-targeting guide names."""
        df = pd.read_csv(guide_annotation_path, sep="\t", index_col=0)
        df_non = df[df["targeting"] == False]
        reference_targets = df_non.index.values.tolist()
        assert len(reference_targets) > 0
        # All extracted targets should be non-targeting type
        assert all(df.loc[t, "type"] == "non-targeting" for t in reference_targets)

    def test_perturbation_with_annotation_table(self, mdata_copy, reference_targets_from_annotation):
        """Run perturbation association using reference_targets extracted from annotation table."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=reference_targets_from_annotation,
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert isinstance(result, pd.DataFrame)
        assert "adj_pval" in result.columns
        assert all(result["pval"].between(0, 1))

    def test_perturbation_with_key_string(self, mdata_copy):
        """Run perturbation association using the direct key string."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        assert isinstance(result, pd.DataFrame)
        assert "adj_pval" in result.columns
        assert all(result["pval"].between(0, 1))

    def test_both_methods_same_results(self, mdata_copy, reference_targets_from_annotation):
        """Both methods should produce identical perturbation results."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import compute_perturbation_association
        result_key = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=["non-targeting"],
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        # Need a fresh copy since mdata_copy is function-scoped but same object within one test
        result_table = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            reference_targets=reference_targets_from_annotation,
            collapse_targets=True, n_jobs=1, inplace=False, FDR_method="BH"
        )
        # Same shape and target names
        assert set(result_key["target_name"]) == set(result_table["target_name"])
        # Merge and compare p-values
        merged = result_key.merge(result_table, on=["target_name", "program_name"], suffixes=("_key", "_table"))
        assert np.allclose(merged["pval_key"], merged["pval_table"], atol=1e-10)


# --- association_perturbation.py: unit tests ---

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
