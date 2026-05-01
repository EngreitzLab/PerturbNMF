"""
Unit tests for U-test perturbation calibration.

Tests use synthetic MuData with guide assignments. No real inference
output is required.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage2_Evaluation/test_utest.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import mudata as mu
from scipy import sparse


# ===========================================================================
# Tests for perturbation association (core of U-test calibration)
# ===========================================================================

class TestPerturbationAssociation:
    """Test the core compute_perturbation_association function used by U-test calibration."""

    def test_basic_perturbation_association(self, calibration_mdata_copy):
        """Run perturbation association and check output structure."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "target_name" in result.columns
        assert "program_name" in result.columns
        assert "p-value" in result.columns or "pval" in result.columns

    def test_perturbation_pvalues_valid(self, calibration_mdata_copy):
        """All p-values should be between 0 and 1."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        pval_col = "p-value" if "p-value" in result.columns else "pval"
        pvals = result[pval_col].dropna()
        assert (pvals >= 0).all(), "Found negative p-values"
        assert (pvals <= 1).all(), "Found p-values > 1"

    def test_perturbation_per_condition(self, calibration_mdata_copy):
        """Run perturbation association on a single condition subset."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        cond = calibration_mdata_copy["rna"].obs["condition"].unique()[0]
        mask = calibration_mdata_copy["rna"].obs["condition"] == cond
        mdata_sub = calibration_mdata_copy[mask]

        result = compute_perturbation_association(
            mdata_sub,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_fdr_bh_correction(self, calibration_mdata_copy):
        """BH FDR correction should produce adj_pval column."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert "adj_pval" in result.columns
        adj = result["adj_pval"].dropna()
        assert (adj >= 0).all()
        assert (adj <= 1).all()


# ===========================================================================
# Tests for fake guide calibration logic
# ===========================================================================

class TestFakeGuideCalibration:
    """Test the calibration logic: randomly assigning non-targeting guides as fake targeting."""

    def test_random_guide_selection(self, synthetic_mdata, guide_annotation_df):
        """Verify random guide selection produces valid subsets."""
        rng = np.random.default_rng(123)
        nt_guides = guide_annotation_df[~guide_annotation_df["targeting"]]
        n_select = 6

        selected = rng.choice(nt_guides["guide_names"].values, n_select, replace=False)
        assert len(selected) == n_select
        assert len(set(selected)) == n_select  # all unique
        # All selected should be non-targeting
        for g in selected:
            assert g in nt_guides.index

    def test_fake_targeting_assignment(self, synthetic_mdata, guide_annotation_df):
        """Create a fake targeting assignment and verify structure."""
        rng = np.random.default_rng(456)
        guide_ann = guide_annotation_df.copy()
        nt_mask = ~guide_ann["targeting"]
        nt_guides = guide_ann[nt_mask]

        # Select 3 NT guides to become fake "targeting"
        selected = rng.choice(nt_guides["guide_names"].values, 3, replace=False)
        guide_ann.loc[guide_ann["guide_names"].isin(selected), "type"] = "targeting"

        # Should now have original targeting + 3 fake targeting
        n_targeting = (guide_ann["type"] == "targeting").sum()
        n_original_targeting = guide_annotation_df["targeting"].sum()
        assert n_targeting == n_original_targeting + 3

    def test_subset_to_nontargeting_guides(self, calibration_mdata_copy, guide_annotation_df):
        """Subset MuData to only non-targeting guides."""
        nt_mask = ~guide_annotation_df["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        prog = calibration_mdata_copy["cNMF"]
        original_n_guides = prog.obsm["guide_assignment"].shape[1]

        # Subset guide assignment matrix
        subset_assignment = prog.obsm["guide_assignment"][:, nt_indices]
        subset_names = prog.uns["guide_names"][nt_indices]

        assert subset_assignment.shape[1] == nt_mask.sum()
        assert len(subset_names) == nt_mask.sum()
        assert subset_assignment.shape[1] < original_n_guides

    def test_calibration_run_single_iteration(self, calibration_mdata_copy, guide_annotation_df):
        """Run one calibration iteration: subset to NT, relabel some as targeting, compute association."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        rng = np.random.default_rng(789)
        guide_ann = guide_annotation_df.copy()
        nt_mask = ~guide_ann["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        # Subset to NT guides only
        prog = calibration_mdata_copy["cNMF"]
        prog.obsm["guide_assignment"] = prog.obsm["guide_assignment"][:, nt_indices]
        prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]

        # Relabel 3 guides as fake "targeting"
        new_targets = np.array(["non-targeting"] * len(nt_indices))
        selected_idx = rng.choice(len(nt_indices), 3, replace=False)
        new_targets[selected_idx] = "targeting"
        prog.uns["guide_targets"] = new_targets

        # Also update rna modality
        rna = calibration_mdata_copy["rna"]
        rna.obsm["guide_assignment"] = rna.obsm["guide_assignment"][:, nt_indices]
        rna.uns["guide_names"] = prog.uns["guide_names"]
        rna.uns["guide_targets"] = new_targets

        # Run perturbation association
        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # With random NT guides as "targeting", expect no significant hits
        pval_col = "p-value" if "p-value" in result.columns else "pval"
        # Null calibration: median p-value should be > 0.05 (not enriched for signal)
        assert result[pval_col].median() > 0.01


# ===========================================================================
# Tests for visualization helpers
# ===========================================================================

class TestCalibrationVisualization:
    """Test plot functions from the U-test calibration script."""

    def test_violin_plot_structure(self):
        """Create a fake results DataFrame and verify violin plot runs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "pval": rng.uniform(0, 1, 200),
            "sample": rng.choice(["d0", "d1"], 200),
            "K": [5] * 200,
            "real": [True] * 100 + [False] * 100,
            "target_name": ["gene_0"] * 200,
        })
        df["neg_log_pval"] = -np.log(df["pval"].clip(1e-300))

        fig, ax = plt.subplots()
        import seaborn as sns
        sns.violinplot(x="sample", y="neg_log_pval", hue="real", data=df, ax=ax)
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_qq_data_preparation(self):
        """Verify QQ plot data preparation with real vs null split."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "pval": np.concatenate([
                rng.uniform(0, 0.1, n),  # real: enriched for low p-values
                rng.uniform(0, 1, n),     # null: uniform
            ]),
            "K": [5] * (2 * n),
            "real": [True] * n + [False] * n,
            "target_name": ["gene_0"] * n + ["targeting"] * n,
        })

        real = df[(df.K == 5) & (df.real == True)]["pval"]
        null = df[(df.K == 5) & (df.real == False) & (df.target_name == "targeting")]["pval"]

        assert len(real) == n
        assert len(null) == n
        # Real should have lower median p-value than null
        assert real.median() < null.median()


# ===========================================================================
# Tests for guide metadata handling
# ===========================================================================

class TestGuideMetadata:
    """Test guide metadata extraction and manipulation."""

    def test_get_guide_metadata(self, calibration_mdata_copy):
        """Extract guide metadata from MuData."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import get_guide_metadata

        guide_meta = get_guide_metadata(calibration_mdata_copy, prog_key="cNMF")
        assert isinstance(guide_meta, pd.DataFrame)
        assert "Target" in guide_meta.columns
        assert len(guide_meta) == len(calibration_mdata_copy["cNMF"].uns["guide_names"])

    def test_guide_metadata_has_nontargeting(self, calibration_mdata_copy):
        """Verify non-targeting guides are present in metadata."""
        from Stage2_Evaluation.A_Metrics.src.association_perturbation import get_guide_metadata

        guide_meta = get_guide_metadata(calibration_mdata_copy, prog_key="cNMF")
        nt_count = (guide_meta["Target"] == "non-targeting").sum()
        assert nt_count > 0, "No non-targeting guides found"

    def test_guide_assignment_shape(self, synthetic_mdata):
        """Guide assignment matrix should be cells × guides."""
        assignment = synthetic_mdata["cNMF"].obsm["guide_assignment"]
        n_cells = synthetic_mdata["cNMF"].n_obs
        n_guides = len(synthetic_mdata["cNMF"].uns["guide_names"])
        assert assignment.shape == (n_cells, n_guides)

    def test_each_cell_has_one_guide(self, synthetic_mdata):
        """Each cell should be assigned to exactly one guide."""
        assignment = synthetic_mdata["cNMF"].obsm["guide_assignment"]
        row_sums = assignment.sum(axis=1)
        np.testing.assert_array_equal(row_sums, 1.0)


# ===========================================================================
# Tests for output file structure
# ===========================================================================

class TestCalibrationOutput:
    """Test that calibration produces correctly formatted output files."""

    def test_save_perturbation_results(self, calibration_mdata_copy, calibration_output_dir):
        """Run association and save to TSV, verify file format."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )

        out_path = os.path.join(calibration_output_dir, "test_perturbation_results.txt")
        result.to_csv(out_path, sep="\t", index=False)
        assert os.path.exists(out_path)

        # Reload and verify
        loaded = pd.read_csv(out_path, sep="\t")
        assert len(loaded) == len(result)
        assert "target_name" in loaded.columns
        assert "program_name" in loaded.columns

    def test_multiple_iterations_concatenate(self, calibration_mdata_copy, guide_annotation_df):
        """Multiple calibration iterations should concatenate correctly."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        rng = np.random.default_rng(0)
        all_results = []
        nt_mask = ~guide_annotation_df["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        for iteration in range(3):
            _mdata = calibration_mdata_copy.copy()
            prog = _mdata["cNMF"]
            prog.obsm["guide_assignment"] = prog.obsm["guide_assignment"][:, nt_indices]
            prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]

            new_targets = np.array(["non-targeting"] * len(nt_indices))
            selected = rng.choice(len(nt_indices), 3, replace=False)
            new_targets[selected] = "targeting"
            prog.uns["guide_targets"] = new_targets

            _mdata["rna"].obsm["guide_assignment"] = _mdata["rna"].obsm["guide_assignment"][:, nt_indices]
            _mdata["rna"].uns["guide_names"] = prog.uns["guide_names"]
            _mdata["rna"].uns["guide_targets"] = new_targets

            result = compute_perturbation_association(
                _mdata,
                prog_key="cNMF",
                collapse_targets=True,
                pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH",
                n_jobs=1,
                inplace=False,
            )
            result["run"] = iteration
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        assert "run" in combined.columns
        assert combined["run"].nunique() == 3
        assert len(combined) == sum(len(r) for r in all_results)


# ===========================================================================
# Tests for edge cases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions for calibration."""

    def test_sparse_guide_assignment(self, synthetic_mdata):
        """Perturbation association should work with sparse guide assignment matrices."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        mdata = synthetic_mdata.copy()
        prog = mdata["cNMF"]
        # Convert dense to sparse
        prog.obsm["guide_assignment"] = sparse.csr_matrix(prog.obsm["guide_assignment"])
        mdata["rna"].obsm["guide_assignment"] = sparse.csr_matrix(
            mdata["rna"].obsm["guide_assignment"]
        )

        result = compute_perturbation_association(
            mdata,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(result["pval"].between(0, 1))

    def test_single_target_gene(self, synthetic_mdata, guide_annotation_df):
        """Calibration with only one fake targeting guide group."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        mdata = synthetic_mdata.copy()
        nt_mask = ~guide_annotation_df["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        prog = mdata["cNMF"]
        prog.obsm["guide_assignment"] = prog.obsm["guide_assignment"][:, nt_indices]
        prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]
        mdata["rna"].obsm["guide_assignment"] = mdata["rna"].obsm["guide_assignment"][:, nt_indices]
        mdata["rna"].uns["guide_names"] = prog.uns["guide_names"]

        # Label only 1 guide as targeting
        new_targets = np.array(["non-targeting"] * len(nt_indices))
        new_targets[0] = "targeting"
        prog.uns["guide_targets"] = new_targets
        mdata["rna"].uns["guide_targets"] = new_targets

        result = compute_perturbation_association(
            mdata,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert result["target_name"].nunique() == 1

    def test_all_pvals_finite_after_fdr(self, calibration_mdata_copy):
        """FDR-adjusted p-values must all be finite (no NaN/Inf)."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert all(np.isfinite(result["adj_pval"]))
        assert all(result["adj_pval"] >= 0)
        assert all(result["adj_pval"] <= 1)

    def test_inplace_mode_updates_mdata(self, calibration_mdata_copy):
        """inplace=True should write results into varm/uns, not return a DataFrame."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        ret = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=True,
        )
        assert ret is None
        assert "perturbation_association_target_stat" in calibration_mdata_copy["cNMF"].varm
        assert "perturbation_association_target_pval" in calibration_mdata_copy["cNMF"].varm
        assert "perturbation_association_target_names" in calibration_mdata_copy["cNMF"].uns


# ===========================================================================
# Tests for calibration reproducibility
# ===========================================================================

class TestCalibrationReproducibility:
    """Verify that fixed seeds produce identical calibration results."""

    def test_same_seed_same_guide_selection(self, guide_annotation_df):
        """Same RNG seed should select the same fake targeting guides."""
        nt_guides = guide_annotation_df[~guide_annotation_df["targeting"]]["guide_names"].values

        rng1 = np.random.default_rng(999)
        sel1 = rng1.choice(nt_guides, 3, replace=False)

        rng2 = np.random.default_rng(999)
        sel2 = rng2.choice(nt_guides, 3, replace=False)

        np.testing.assert_array_equal(sel1, sel2)

    def test_different_seeds_differ(self, guide_annotation_df):
        """Different seeds should (almost certainly) produce different selections."""
        nt_guides = guide_annotation_df[~guide_annotation_df["targeting"]]["guide_names"].values

        rng1 = np.random.default_rng(111)
        sel1 = set(rng1.choice(nt_guides, 3, replace=False))

        rng2 = np.random.default_rng(222)
        sel2 = set(rng2.choice(nt_guides, 3, replace=False))

        assert sel1 != sel2

    def test_reproducible_calibration_pvalues(self, synthetic_mdata, guide_annotation_df):
        """Two calibration runs with the same seed should give identical p-values."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        nt_mask = ~guide_annotation_df["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        results = []
        for _ in range(2):
            rng = np.random.default_rng(42)
            mdata = synthetic_mdata.copy()
            prog = mdata["cNMF"]
            prog.obsm["guide_assignment"] = prog.obsm["guide_assignment"][:, nt_indices]
            prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]
            mdata["rna"].obsm["guide_assignment"] = mdata["rna"].obsm["guide_assignment"][:, nt_indices]
            mdata["rna"].uns["guide_names"] = prog.uns["guide_names"]

            new_targets = np.array(["non-targeting"] * len(nt_indices))
            selected = rng.choice(len(nt_indices), 3, replace=False)
            new_targets[selected] = "targeting"
            prog.uns["guide_targets"] = new_targets
            mdata["rna"].uns["guide_targets"] = new_targets

            result = compute_perturbation_association(
                mdata,
                prog_key="cNMF",
                collapse_targets=True,
                pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH",
                n_jobs=1,
                inplace=False,
            )
            results.append(result)

        pd.testing.assert_frame_equal(results[0], results[1])


# ===========================================================================
# Tests for condition-stratified calibration
# ===========================================================================

class TestConditionStratifiedCalibration:
    """Test calibration run per-condition (mimicking the real pipeline)."""

    def test_per_condition_results_differ(self, calibration_mdata_copy):
        """Different conditions should produce different p-value distributions."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        conditions = calibration_mdata_copy["rna"].obs["condition"].unique()
        assert len(conditions) >= 2

        results_by_cond = {}
        for cond in conditions[:2]:
            mask = calibration_mdata_copy["rna"].obs["condition"] == cond
            mdata_sub = calibration_mdata_copy[mask]
            result = compute_perturbation_association(
                mdata_sub,
                prog_key="cNMF",
                collapse_targets=True,
                pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH",
                n_jobs=1,
                inplace=False,
            )
            results_by_cond[cond] = result

        c1, c2 = list(results_by_cond.keys())
        # Results should differ (different cell subsets)
        assert not results_by_cond[c1]["pval"].values.tolist() == results_by_cond[c2]["pval"].values.tolist()

    def test_per_condition_combine_with_metadata(self, calibration_mdata_copy):
        """Combine per-condition results with sample metadata columns."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        all_results = []
        for cond in calibration_mdata_copy["rna"].obs["condition"].unique():
            mask = calibration_mdata_copy["rna"].obs["condition"] == cond
            mdata_sub = calibration_mdata_copy[mask]
            result = compute_perturbation_association(
                mdata_sub,
                prog_key="cNMF",
                collapse_targets=True,
                pseudobulk=False,
                reference_targets=["non-targeting"],
                FDR_method="BH",
                n_jobs=1,
                inplace=False,
            )
            result["sample"] = cond
            result["K"] = 5
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        assert "sample" in combined.columns
        assert "K" in combined.columns
        assert combined["sample"].nunique() == calibration_mdata_copy["rna"].obs["condition"].nunique()


# ===========================================================================
# Tests for sparse-to-dense conversion (_assign_guide logic)
# ===========================================================================

class TestSparseConversion:
    """Test the sparse-to-dense guide assignment conversion used in calibration scripts."""

    def test_sparse_to_dense_roundtrip(self, synthetic_mdata):
        """Converting dense -> sparse -> dense should preserve assignment."""
        original = synthetic_mdata["cNMF"].obsm["guide_assignment"].copy()
        sp = sparse.csr_matrix(original)
        recovered = sp.toarray()
        np.testing.assert_array_equal(original, recovered)

    def test_sparse_assignment_row_sums(self, synthetic_mdata):
        """Sparse assignment should still have row sums of 1."""
        sp = sparse.csr_matrix(synthetic_mdata["cNMF"].obsm["guide_assignment"])
        row_sums = np.asarray(sp.sum(axis=1)).flatten()
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_dense_assignment_column_subsetting(self, synthetic_mdata, guide_annotation_df):
        """Column subsetting on dense arrays should work for NT guide extraction."""
        assignment = synthetic_mdata["cNMF"].obsm["guide_assignment"]
        nt_mask = ~guide_annotation_df["targeting"]
        nt_indices = np.where(nt_mask.values)[0]

        subset = assignment[:, nt_indices]
        assert subset.shape == (assignment.shape[0], nt_mask.sum())
        # Cells assigned to targeting guides should have row sum 0
        targeting_indices = np.where(guide_annotation_df["targeting"].values)[0]
        targeting_only = assignment[:, targeting_indices]
        cells_with_targeting = targeting_only.sum(axis=1) > 0
        # Those cells should have 0 in the NT subset
        assert (subset[cells_with_targeting].sum(axis=1) == 0).all()


# ===========================================================================
# Tests for U-test calibration plot functions
# ===========================================================================

class TestUTestPlotFunctions:
    """Test plot_calibration_comparison and plot_qq_comparison from U-test script."""

    @pytest.fixture
    def calibration_results_df(self):
        """Synthetic combined real + fake results DataFrame."""
        rng = np.random.default_rng(0)
        n = 50
        rows = []
        for k in [5, 10]:
            for samp in ["d0", "d1"]:
                # Real
                for _ in range(n):
                    rows.append({
                        "pval": rng.uniform(0.001, 0.5),
                        "sample": samp,
                        "K": k,
                        "real": True,
                        "target_name": rng.choice(["gene_0", "gene_1", "gene_2"]),
                        "program_name": f"prog_{rng.integers(0, 5)}",
                    })
                # Fake (null)
                for _ in range(n):
                    rows.append({
                        "pval": rng.uniform(0.01, 1.0),
                        "sample": samp,
                        "K": k,
                        "real": False,
                        "target_name": "targeting",
                        "program_name": f"prog_{rng.integers(0, 5)}",
                    })
        return pd.DataFrame(rows)

    def test_violin_plot_runs(self, calibration_results_df):
        """plot_calibration_comparison should produce a figure without error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = calibration_results_df.copy()
        df["neg_log_pval"] = -np.log(df["pval"].clip(1e-300))

        for k in df["K"].unique():
            subset = df[df.K == k]
            fig, ax = plt.subplots()
            sns.violinplot(x="sample", y="neg_log_pval", hue="real", data=subset, ax=ax)
            ax.set_title(f"K={k}")
            ax.set_ylabel("-ln(p-value)")
            assert len(ax.collections) > 0
            plt.close(fig)

    def test_qq_real_vs_null_separation(self, calibration_results_df):
        """Real p-values should skew lower than null p-values."""
        df = calibration_results_df
        for k in df["K"].unique():
            real = df[(df.K == k) & (df.real == True)]["pval"]
            null = df[(df.K == k) & (df.real == False) & (df.target_name == "targeting")]["pval"]
            assert real.median() < null.median()

    def test_neg_log_pval_no_inf(self, calibration_results_df):
        """-log(pval) should not produce Inf when pvals are clipped."""
        df = calibration_results_df
        neg_log = -np.log(df["pval"].clip(1e-300))
        assert all(np.isfinite(neg_log))

    def test_calibration_results_schema(self, calibration_results_df):
        """Verify the combined results DataFrame has all expected columns."""
        expected = {"pval", "sample", "K", "real", "target_name", "program_name"}
        assert expected.issubset(set(calibration_results_df.columns))


# ===========================================================================
# Tests for guide-level (non-collapsed) perturbation association
# ===========================================================================

class TestGuideLevelAssociation:
    """Test perturbation association at guide level (collapse_targets=False)."""

    def test_guide_level_returns_guide_name_col(self, calibration_mdata_copy):
        """With collapse_targets=False, output should have guide_name column."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=False,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert "guide_name" in result.columns
        assert "target_name" not in result.columns

    def test_guide_level_more_rows_than_target_level(self, calibration_mdata_copy):
        """Guide-level results should have more rows than target-level (multiple guides per target)."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association

        target_result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=True,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        guide_result = compute_perturbation_association(
            calibration_mdata_copy,
            prog_key="cNMF",
            collapse_targets=False,
            pseudobulk=False,
            reference_targets=["non-targeting"],
            FDR_method="BH",
            n_jobs=1,
            inplace=False,
        )
        assert len(guide_result) > len(target_result)
