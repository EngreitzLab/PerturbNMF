"""
Unit tests for U-test perturbation calibration.

Tests use real inference output (same as test_metrics.py).

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage2_Evaluation/test_utest.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


# ===========================================================================
# Tests for perturbation association (core of U-test calibration)
# ===========================================================================

class TestPerturbationAssociation:

    def test_basic_perturbation_association(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "target_name" in result.columns
        assert "program_name" in result.columns
        assert "pval" in result.columns

    def test_perturbation_pvalues_valid(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert (result["pval"] >= 0).all()
        assert (result["pval"] <= 1).all()

    def test_perturbation_per_batch(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        batch = mdata_copy["rna"].obs["batch"].unique()[0]
        mask = mdata_copy["rna"].obs["batch"] == batch
        mdata_sub = mdata_copy[mask]
        result = compute_perturbation_association(
            mdata_sub, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_fdr_bh_correction(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert "adj_pval" in result.columns
        assert (result["adj_pval"] >= 0).all()
        assert (result["adj_pval"] <= 1).all()


# ===========================================================================
# Tests for fake guide calibration logic
# ===========================================================================

class TestFakeGuideCalibration:

    def test_random_guide_selection(self, guide_annotation_df):
        rng = np.random.default_rng(123)
        nt_guides = guide_annotation_df[guide_annotation_df["targeting"] == False]
        n_select = 6
        selected = rng.choice(nt_guides.index.values, n_select, replace=False)
        assert len(selected) == n_select
        assert len(set(selected)) == n_select
        for g in selected:
            assert g in nt_guides.index

    def test_fake_targeting_assignment(self, guide_annotation_df):
        rng = np.random.default_rng(456)
        guide_ann = guide_annotation_df.copy()
        nt_guides = guide_ann[guide_ann["targeting"] == False]
        selected = rng.choice(nt_guides.index.values, 3, replace=False)
        guide_ann.loc[selected, "type"] = "targeting"
        n_targeting = (guide_ann["type"] == "targeting").sum()
        n_original_targeting = (guide_annotation_df["targeting"] == True).sum()
        assert n_targeting == n_original_targeting + 3

    def test_subset_to_nontargeting_guides(self, mdata_copy, guide_annotation_df):
        nt_mask = guide_annotation_df["targeting"] == False
        nt_indices = np.where(nt_mask.values)[0]
        prog = mdata_copy["cNMF"]
        assignment = prog.obsm["guide_assignment"]
        if sparse.issparse(assignment):
            assignment = assignment.toarray()
        original_n_guides = assignment.shape[1]
        subset_assignment = assignment[:, nt_indices]
        assert subset_assignment.shape[1] == nt_mask.sum()
        assert subset_assignment.shape[1] < original_n_guides

    def test_calibration_run_single_iteration(self, mdata_copy, guide_annotation_df):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        rng = np.random.default_rng(789)
        nt_mask = guide_annotation_df["targeting"] == False
        nt_indices = np.where(nt_mask.values)[0]

        prog = mdata_copy["cNMF"]
        assignment = prog.obsm["guide_assignment"]
        if sparse.issparse(assignment):
            assignment = assignment.toarray()
        prog.obsm["guide_assignment"] = assignment[:, nt_indices]
        prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]

        new_targets = np.array(["non-targeting"] * len(nt_indices))
        selected_idx = rng.choice(len(nt_indices), 3, replace=False)
        new_targets[selected_idx] = "targeting"
        prog.uns["guide_targets"] = new_targets

        rna = mdata_copy["rna"]
        rna_assignment = rna.obsm["guide_assignment"]
        if sparse.issparse(rna_assignment):
            rna_assignment = rna_assignment.toarray()
        rna.obsm["guide_assignment"] = rna_assignment[:, nt_indices]
        rna.uns["guide_names"] = prog.uns["guide_names"]
        rna.uns["guide_targets"] = new_targets

        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert result["pval"].median() > 0.01


# ===========================================================================
# Tests for guide metadata handling
# ===========================================================================

class TestGuideMetadata:

    def test_get_guide_metadata(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import get_guide_metadata
        guide_meta = get_guide_metadata(mdata_copy, prog_key="cNMF")
        assert isinstance(guide_meta, pd.DataFrame)
        assert "Target" in guide_meta.columns
        assert len(guide_meta) == len(mdata_copy["cNMF"].uns["guide_names"])

    def test_guide_metadata_has_nontargeting(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import get_guide_metadata
        guide_meta = get_guide_metadata(mdata_copy, prog_key="cNMF")
        nt_count = (guide_meta["Target"] == "non-targeting").sum()
        assert nt_count > 0

    def test_guide_assignment_shape(self, test_mdata):
        assignment = test_mdata["cNMF"].obsm["guide_assignment"]
        if sparse.issparse(assignment):
            assignment = assignment.toarray()
        n_cells = test_mdata["cNMF"].n_obs
        n_guides = len(test_mdata["cNMF"].uns["guide_names"])
        assert assignment.shape == (n_cells, n_guides)


# ===========================================================================
# Tests for output file structure
# ===========================================================================

class TestCalibrationOutput:

    def test_save_perturbation_results(self, mdata_copy, calibration_output_dir):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        out_path = os.path.join(calibration_output_dir, "test_perturbation_results.txt")
        result.to_csv(out_path, sep="\t", index=False)
        assert os.path.exists(out_path)
        loaded = pd.read_csv(out_path, sep="\t")
        assert len(loaded) == len(result)
        assert "target_name" in loaded.columns

    def test_multiple_iterations_concatenate(self, mdata_copy, guide_annotation_df):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        rng = np.random.default_rng(0)
        all_results = []
        nt_mask = guide_annotation_df["targeting"] == False
        nt_indices = np.where(nt_mask.values)[0]

        for iteration in range(3):
            _mdata = mdata_copy.copy()
            prog = _mdata["cNMF"]
            assignment = prog.obsm["guide_assignment"]
            if sparse.issparse(assignment):
                assignment = assignment.toarray()
            prog.obsm["guide_assignment"] = assignment[:, nt_indices]
            prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]

            new_targets = np.array(["non-targeting"] * len(nt_indices))
            selected = rng.choice(len(nt_indices), 3, replace=False)
            new_targets[selected] = "targeting"
            prog.uns["guide_targets"] = new_targets
            _mdata["rna"].uns["guide_names"] = prog.uns["guide_names"]
            _mdata["rna"].uns["guide_targets"] = new_targets

            result = compute_perturbation_association(
                _mdata, prog_key="cNMF",
                collapse_targets=True, pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH", n_jobs=1, inplace=False,
            )
            result["run"] = iteration
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        assert "run" in combined.columns
        assert combined["run"].nunique() == 3


# ===========================================================================
# Tests for edge cases
# ===========================================================================

class TestEdgeCases:

    def test_sparse_guide_assignment(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        prog = mdata_copy["cNMF"]
        assignment = prog.obsm["guide_assignment"]
        if not sparse.issparse(assignment):
            prog.obsm["guide_assignment"] = sparse.csr_matrix(assignment)
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result["pval"].between(0, 1))

    def test_all_pvals_finite_after_fdr(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert all(np.isfinite(result["adj_pval"]))
        assert all(result["adj_pval"] >= 0)
        assert all(result["adj_pval"] <= 1)

    def test_inplace_mode_updates_mdata(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        ret = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=True,
        )
        assert ret is None
        assert "perturbation_association_target_stat" in mdata_copy["cNMF"].varm
        assert "perturbation_association_target_pval" in mdata_copy["cNMF"].varm
        assert "perturbation_association_target_names" in mdata_copy["cNMF"].uns


# ===========================================================================
# Tests for calibration reproducibility
# ===========================================================================

class TestCalibrationReproducibility:

    def test_same_seed_same_guide_selection(self, guide_annotation_df):
        nt_guides = guide_annotation_df[guide_annotation_df["targeting"] == False].index.values
        rng1 = np.random.default_rng(999)
        sel1 = rng1.choice(nt_guides, 3, replace=False)
        rng2 = np.random.default_rng(999)
        sel2 = rng2.choice(nt_guides, 3, replace=False)
        np.testing.assert_array_equal(sel1, sel2)

    def test_different_seeds_differ(self, guide_annotation_df):
        nt_guides = guide_annotation_df[guide_annotation_df["targeting"] == False].index.values
        rng1 = np.random.default_rng(111)
        sel1 = set(rng1.choice(nt_guides, 3, replace=False))
        rng2 = np.random.default_rng(222)
        sel2 = set(rng2.choice(nt_guides, 3, replace=False))
        assert sel1 != sel2


# ===========================================================================
# Tests for condition-stratified calibration
# ===========================================================================

class TestConditionStratifiedCalibration:

    def test_per_batch_results_differ(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        batches = mdata_copy["rna"].obs["batch"].unique()
        assert len(batches) >= 2

        results_by_batch = {}
        for batch in batches[:2]:
            mask = mdata_copy["rna"].obs["batch"] == batch
            mdata_sub = mdata_copy[mask]
            result = compute_perturbation_association(
                mdata_sub, prog_key="cNMF",
                collapse_targets=True, pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH", n_jobs=1, inplace=False,
            )
            results_by_batch[batch] = result

        b1, b2 = list(results_by_batch.keys())
        assert not results_by_batch[b1]["pval"].values.tolist() == results_by_batch[b2]["pval"].values.tolist()

    def test_per_batch_combine_with_metadata(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        all_results = []
        for batch in mdata_copy["rna"].obs["batch"].unique():
            mask = mdata_copy["rna"].obs["batch"] == batch
            mdata_sub = mdata_copy[mask]
            result = compute_perturbation_association(
                mdata_sub, prog_key="cNMF",
                collapse_targets=True, pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH", n_jobs=1, inplace=False,
            )
            result["sample"] = batch
            result["K"] = 5
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        assert "sample" in combined.columns
        assert "K" in combined.columns
        assert combined["sample"].nunique() == mdata_copy["rna"].obs["batch"].nunique()


# ===========================================================================
# Tests for sparse-to-dense conversion
# ===========================================================================

class TestSparseConversion:

    def test_sparse_to_dense_roundtrip(self, test_mdata):
        assignment = test_mdata["cNMF"].obsm["guide_assignment"]
        if sparse.issparse(assignment):
            dense = assignment.toarray()
        else:
            dense = assignment.copy()
        sp = sparse.csr_matrix(dense)
        recovered = sp.toarray()
        np.testing.assert_array_equal(dense, recovered)

    def test_dense_assignment_column_subsetting(self, test_mdata, guide_annotation_df):
        assignment = test_mdata["cNMF"].obsm["guide_assignment"]
        if sparse.issparse(assignment):
            assignment = assignment.toarray()
        nt_mask = guide_annotation_df["targeting"] == False
        nt_indices = np.where(nt_mask.values)[0]
        subset = assignment[:, nt_indices]
        assert subset.shape == (assignment.shape[0], nt_mask.sum())


# ===========================================================================
# Tests for guide-level (non-collapsed) perturbation association
# ===========================================================================

class TestGuideLevelAssociation:

    def test_guide_level_returns_guide_name_col(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=False, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert "guide_name" in result.columns
        assert "target_name" not in result.columns

    def test_guide_level_more_rows_than_target_level(self, mdata_copy):
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        target_result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=True, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        guide_result = compute_perturbation_association(
            mdata_copy, prog_key="cNMF",
            collapse_targets=False, pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH", n_jobs=1, inplace=False,
        )
        assert len(guide_result) > len(target_result)


# ===========================================================================
# Tests for visualization helpers (no MuData needed)
# ===========================================================================

class TestCalibrationVisualization:

    def test_violin_plot_structure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "pval": rng.uniform(0, 1, 200),
            "sample": rng.choice(["d0", "d1"], 200),
            "K": [5] * 200,
            "real": [True] * 100 + [False] * 100,
        })
        df["neg_log_pval"] = -np.log(df["pval"].clip(1e-300))
        fig, ax = plt.subplots()
        sns.violinplot(x="sample", y="neg_log_pval", hue="real", data=df, ax=ax)
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_neg_log_pval_no_inf(self):
        rng = np.random.default_rng(0)
        pvals = rng.uniform(0, 1, 100)
        neg_log = -np.log(np.clip(pvals, 1e-300, None))
        assert all(np.isfinite(neg_log))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
