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


def _get_nontargeting_indices(guide_annotation_df, mdata, prog_key="cNMF"):
    """Get mdata column indices for non-targeting guides by intersecting
    annotation table guide names with mdata guide_names."""
    nt_guide_names = guide_annotation_df[guide_annotation_df["targeting"] == False].index.values
    mdata_guide_names = mdata[prog_key].uns["guide_names"]
    nt_in_mdata = np.isin(mdata_guide_names, nt_guide_names)
    return np.where(nt_in_mdata)[0]


# ===========================================================================
# Helpers to call the original U-test pipeline script
# ===========================================================================

def _load_utest_module():
    """Import the U-test calibration module (has hyphens in path)."""
    import importlib.util
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "src", "Stage2_Evaluation", "B_Calibration", "Slurm_version",
        "U-test_perturbation_calibration", "U-test_perturbation_calibration.py",
    )
    spec = importlib.util.spec_from_file_location("utest_calibration", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_utest_args(inference_path, components, categorical_key="batch",
                     sel_thresh=None, number_run=1, number_guide=3):
    """Create a mock args namespace for the U-test module."""
    from types import SimpleNamespace
    out_dir = os.path.dirname(os.path.dirname(inference_path))
    run_name = os.path.basename(os.path.dirname(inference_path))
    return SimpleNamespace(
        out_dir=out_dir,
        run_name=run_name,
        components=components,
        sel_thresh=sel_thresh if sel_thresh is not None else [2.0],
        guide_annotation_path=None,
        guide_annotation_key=["non-targeting"],
        data_key="rna",
        prog_key="cNMF",
        categorical_key=categorical_key,
        guide_names_key="guide_names",
        guide_targets_key="guide_targets",
        guide_assignment_key="guide_assignment",
        FDR_method="BH",
        mdata_guide_path=None,
        number_run=number_run,
        number_guide=number_guide,
    )


# ===========================================================================
# Tests for perturbation association (core of U-test calibration)
# ===========================================================================

class TestPerturbationAssociation:
    """Run perturbation tests per K using the original U-test pipeline functions."""

    def test_real_perturbation_per_k(self, mdata_copy_per_k, eval_output_dir_per_k, inference_path):
        """Call compute_real_perturbation_tests from the original script, per K per batch.

        Verifies:
        - Returns a combined results DataFrame with K / sel_thresh / sample / real columns
        - Per-(K, sample) text files are saved into Evaluation/{K}_{thresh}/
        - plot_calibration_comparison saves a PNG into the per-(K, sel_thresh) folder
        """
        import matplotlib
        matplotlib.use("Agg")

        utest_mod = _load_utest_module()
        k = mdata_copy_per_k.uns["test_k"]

        utest_mod.args = _make_utest_args(inference_path, components=[k])
        utest_mod.mdata_guide = None

        result_df = utest_mod.compute_real_perturbation_tests()

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert all(result_df["pval"].between(0, 1))
        assert "adj_pval" in result_df.columns
        assert result_df["sample"].nunique() > 1
        assert set(result_df["K"].unique()) == {k}
        assert set(result_df["sel_thresh"].unique()) == set(utest_mod.args.sel_thresh)
        assert all(result_df["real"] == True)

        # Per-(K, sample) text files in Evaluation/{K}_{thresh}/
        for samp in result_df["sample"].unique():
            expected_txt = os.path.join(
                eval_output_dir_per_k,
                f"{k}_perturbation_association_results_{samp}.txt",
            )
            assert os.path.exists(expected_txt), f"Missing per-(K, sample) file: {expected_txt}"

        # Violin plot via original function (saves per-(K, sel_thresh) folder)
        utest_mod.plot_calibration_comparison(result_df)
        expected_png = os.path.join(eval_output_dir_per_k, "U_test_perturbation_association_calibration.png")
        assert os.path.exists(expected_png), f"Missing per-folder calibration plot: {expected_png}"

    def test_fake_perturbation_per_k(self, mdata_copy_per_k, eval_output_dir_per_k, inference_path):
        """Call compute_fake_perturbation_tests from the original script, per K per batch.

        Verifies per-(K, sample) fake result files are saved into Evaluation/{K}_{thresh}/.
        """
        np.random.seed(0)

        utest_mod = _load_utest_module()
        k = mdata_copy_per_k.uns["test_k"]

        utest_mod.args = _make_utest_args(
            inference_path, components=[k], number_run=1, number_guide=3,
        )
        utest_mod.mdata_guide = None

        result_df = utest_mod.compute_fake_perturbation_tests()

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert all(result_df["pval"].between(0, 1))
        assert all(result_df["real"] == False)
        assert set(result_df["K"].unique()) == {k}
        assert set(result_df["sel_thresh"].unique()) == set(utest_mod.args.sel_thresh)

        # Per-(K, sample) fake text files in Evaluation/{K}_{thresh}/
        cat_key = utest_mod.args.categorical_key
        for samp in result_df[cat_key].unique():
            expected_txt = os.path.join(
                eval_output_dir_per_k,
                f"{k}_fake_perturbation_association_results_{samp}.txt",
            )
            assert os.path.exists(expected_txt), f"Missing per-(K, sample) fake file: {expected_txt}"

    def test_plots_per_k_real_eval_folder(self, mdata_copy_per_k, eval_output_dir_per_k, inference_path):
        """Run real + fake tests, then call both plot_qq_comparison and plot_calibration_comparison.

        Verifies that BOTH ``U_test_perturbation_association_qqplot.png`` and
        ``U_test_perturbation_association_calibration.png`` land in the real
        Evaluation/{K}_{thresh}/ folder when fed the combined real+fake frame
        (both plot functions require it: QQ filters on real==True/False with
        target_name=='targeting'; the violin plot compares real vs fake side-by-side).
        """
        import matplotlib
        matplotlib.use("Agg")
        np.random.seed(0)

        utest_mod = _load_utest_module()
        k = mdata_copy_per_k.uns["test_k"]

        utest_mod.args = _make_utest_args(
            inference_path, components=[k], number_run=1, number_guide=3,
        )
        utest_mod.mdata_guide = None

        real_df = utest_mod.compute_real_perturbation_tests()
        fake_df = utest_mod.compute_fake_perturbation_tests()

        # Normalize fake_df:
        # - QQ filter expects target_name == 'targeting'
        # - violin plot expects an x='sample' column; fake uses the categorical_key
        cat_key = utest_mod.args.categorical_key
        fake_df = fake_df.copy()
        if "target_name" not in fake_df.columns:
            fake_df["target_name"] = "targeting"
        if "sample" not in fake_df.columns and cat_key in fake_df.columns:
            fake_df = fake_df.rename(columns={cat_key: "sample"})

        combined = pd.concat([real_df, fake_df], ignore_index=True)

        # Sanity: the combined frame must contain both real and fake rows for K.
        sub = combined[combined.K == k]
        assert (sub["real"] == True).any(), "Combined frame missing real==True rows"
        assert (sub["real"] == False).any(), "Combined frame missing real==False rows"

        utest_mod.plot_qq_comparison(combined)
        utest_mod.plot_calibration_comparison(combined)

        qq_png = os.path.join(eval_output_dir_per_k, "U_test_perturbation_association_qqplot.png")
        cal_png = os.path.join(eval_output_dir_per_k, "U_test_perturbation_association_calibration.png")

        assert os.path.exists(qq_png), f"Missing per-folder QQ plot: {qq_png}"
        assert os.path.getsize(qq_png) > 1000, f"QQ plot looks empty: {qq_png}"
        assert os.path.exists(cal_png), f"Missing per-folder calibration plot: {cal_png}"
        assert os.path.getsize(cal_png) > 1000, f"Calibration plot looks empty: {cal_png}"


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

    def _run_multiple_iterations(self, mdata_copy, guide_annotation_df, use_file_access):
        """Helper: run 3 calibration iterations with key or file-based access."""
        from Stage2_Evaluation.A_Metrics.src import compute_perturbation_association
        rng = np.random.default_rng(0)
        all_results = []

        if use_file_access:
            mdata_copy["cNMF"].uns["guide_targets"] = mdata_copy["cNMF"].uns["guide_names"].copy()
        nt_indices = _get_nontargeting_indices(guide_annotation_df, mdata_copy)
        nt_guide_names = guide_annotation_df[guide_annotation_df["targeting"] == False].index.values

        for iteration in range(3):
            _mdata = mdata_copy.copy()
            prog = _mdata["cNMF"]
            assignment = prog.obsm["guide_assignment"]
            if sparse.issparse(assignment):
                assignment = assignment.toarray()
            prog.obsm["guide_assignment"] = assignment[:, nt_indices]
            prog.uns["guide_names"] = prog.uns["guide_names"][nt_indices]

            if use_file_access:
                new_targets = prog.uns["guide_names"].copy()
            else:
                new_targets = np.array(["non-targeting"] * len(nt_indices))
            selected = rng.choice(len(nt_indices), 3, replace=False)
            new_targets[selected] = "targeting"
            prog.uns["guide_targets"] = new_targets
            _mdata["rna"].uns["guide_names"] = prog.uns["guide_names"]
            _mdata["rna"].uns["guide_targets"] = new_targets

            if use_file_access:
                ref_targets = [g for g in nt_guide_names if g in prog.uns["guide_names"]]
            else:
                ref_targets = ["non-targeting"]

            result = compute_perturbation_association(
                _mdata, prog_key="cNMF",
                collapse_targets=True, pseudobulk=False,
                reference_targets=ref_targets,
                FDR_method="BH", n_jobs=1, inplace=False,
            )
            result["run"] = iteration
            all_results.append(result)

        combined = pd.concat(all_results, ignore_index=True)
        assert "run" in combined.columns
        assert combined["run"].nunique() == 3

    def test_multiple_iterations_concatenate_key(self, mdata_copy, guide_annotation_df):
        """Multiple iterations using key-based reference_targets=['non-targeting']."""
        self._run_multiple_iterations(mdata_copy, guide_annotation_df, use_file_access=False)

    def test_multiple_iterations_concatenate_file(self, mdata_copy, guide_annotation_df):
        """Multiple iterations using file-based reference_targets (individual guide names)."""
        self._run_multiple_iterations(mdata_copy, guide_annotation_df, use_file_access=True)


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

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path, min_bytes=1000):
    """Assert that `path` exists, is non-empty, and starts with the PNG magic header."""
    assert path.exists(), f"Missing plot: {path}"
    size = path.stat().st_size
    assert size > min_bytes, f"Plot too small ({size} bytes), likely empty figure: {path}"
    with open(path, "rb") as fh:
        header = fh.read(len(PNG_MAGIC))
    assert header == PNG_MAGIC, f"File at {path} is not a valid PNG (header={header!r})"


class TestCalibrationVisualization:

    def test_qq_plot_saves_per_folder(self, tmp_path):
        """plot_qq_comparison saves one valid PNG per (K, sel_thresh) into Evaluation/{K}_{thresh}/."""
        import matplotlib
        matplotlib.use("Agg")
        from types import SimpleNamespace

        utest_mod = _load_utest_module()

        components = [5, 10, 15]
        sel_thresh_list = [0.2, 2.0]
        run_name = "test_run"

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        rng = np.random.default_rng(0)
        rows = []
        for k in components:
            for st in sel_thresh_list:
                for p in rng.uniform(0, 1, 50):
                    rows.append({"K": k, "sel_thresh": st, "real": True, "target_name": "geneX", "pval": p})
                for p in rng.uniform(0, 1, 50):
                    rows.append({"K": k, "sel_thresh": st, "real": False, "target_name": "targeting", "pval": p})
        test_stats_dfs = pd.DataFrame(rows)

        utest_mod.plot_qq_comparison(test_stats_dfs)

        # Assert every (K, sel_thresh) QQ plot exists, is non-empty, and is a valid PNG.
        produced_paths = []
        for k in components:
            for st in sel_thresh_list:
                thresh_str = str(st).replace(".", "_")
                expected = tmp_path / run_name / "Evaluation" / f"{k}_{thresh_str}" / "U_test_perturbation_association_qqplot.png"
                _assert_valid_png(expected)
                produced_paths.append(expected)

        # Sanity check: number of plots equals K count x sel_thresh count.
        assert len(produced_paths) == len(components) * len(sel_thresh_list)
        # Each K must appear at least once in the produced set (one plot per sel_thresh per K).
        for k in components:
            matched = [p for p in produced_paths if f"/{k}_" in str(p)]
            assert len(matched) == len(sel_thresh_list), (
                f"Expected {len(sel_thresh_list)} QQ plots for K={k}, got {len(matched)}"
            )

    def test_calibration_plot_saves_per_folder(self, tmp_path):
        """plot_calibration_comparison saves one valid PNG per (K, sel_thresh) into Evaluation/{K}_{thresh}/."""
        import matplotlib
        matplotlib.use("Agg")
        from types import SimpleNamespace

        utest_mod = _load_utest_module()

        components = [5, 10, 15]
        sel_thresh_list = [0.2, 2.0]
        run_name = "test_run"

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        rng = np.random.default_rng(0)
        rows = []
        for k in components:
            for st in sel_thresh_list:
                for samp in ["d0", "d1"]:
                    for p in rng.uniform(1e-6, 1, 50):
                        rows.append({"K": k, "sel_thresh": st, "real": True, "sample": samp, "pval": p})
                    for p in rng.uniform(1e-6, 1, 50):
                        rows.append({"K": k, "sel_thresh": st, "real": False, "sample": samp, "pval": p})
        test_stats_dfs = pd.DataFrame(rows)

        utest_mod.plot_calibration_comparison(test_stats_dfs)

        # Assert every (K, sel_thresh) calibration density plot exists, is non-empty, and is a valid PNG.
        produced_paths = []
        for k in components:
            for st in sel_thresh_list:
                thresh_str = str(st).replace(".", "_")
                expected = tmp_path / run_name / "Evaluation" / f"{k}_{thresh_str}" / "U_test_perturbation_association_calibration.png"
                _assert_valid_png(expected)
                produced_paths.append(expected)

        assert len(produced_paths) == len(components) * len(sel_thresh_list)
        for k in components:
            matched = [p for p in produced_paths if f"/{k}_" in str(p)]
            assert len(matched) == len(sel_thresh_list), (
                f"Expected {len(sel_thresh_list)} calibration plots for K={k}, got {len(matched)}"
            )

    def test_qq_plot_skips_missing_k(self, tmp_path):
        """plot_qq_comparison should skip (no file) for K values with no data, but write plots for the rest."""
        import matplotlib
        matplotlib.use("Agg")
        from types import SimpleNamespace

        utest_mod = _load_utest_module()

        components = [5, 10, 15]
        sel_thresh_list = [2.0]
        run_name = "test_run_skip"

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        # Only provide data for K=5 and K=15; K=10 is intentionally missing.
        rng = np.random.default_rng(1)
        rows = []
        for k in [5, 15]:
            for st in sel_thresh_list:
                for p in rng.uniform(0, 1, 50):
                    rows.append({"K": k, "sel_thresh": st, "real": True, "target_name": "geneX", "pval": p})
                for p in rng.uniform(0, 1, 50):
                    rows.append({"K": k, "sel_thresh": st, "real": False, "target_name": "targeting", "pval": p})
        test_stats_dfs = pd.DataFrame(rows)

        utest_mod.plot_qq_comparison(test_stats_dfs)

        for k in [5, 15]:
            for st in sel_thresh_list:
                thresh_str = str(st).replace(".", "_")
                expected = tmp_path / run_name / "Evaluation" / f"{k}_{thresh_str}" / "U_test_perturbation_association_qqplot.png"
                _assert_valid_png(expected)

        # K=10 had no data — no QQ PNG should have been written.
        thresh_str = str(sel_thresh_list[0]).replace(".", "_")
        missing = tmp_path / run_name / "Evaluation" / f"10_{thresh_str}" / "U_test_perturbation_association_qqplot.png"
        assert not missing.exists(), f"QQ plot should not exist for empty K=10: {missing}"

    def test_calibration_plot_skips_missing_k(self, tmp_path):
        """plot_calibration_comparison should skip K values with no data."""
        import matplotlib
        matplotlib.use("Agg")
        from types import SimpleNamespace

        utest_mod = _load_utest_module()

        components = [5, 10, 15]
        sel_thresh_list = [2.0]
        run_name = "test_run_skip_cal"

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        rng = np.random.default_rng(2)
        rows = []
        for k in [5, 15]:
            for st in sel_thresh_list:
                for samp in ["d0", "d1"]:
                    for p in rng.uniform(1e-6, 1, 50):
                        rows.append({"K": k, "sel_thresh": st, "real": True, "sample": samp, "pval": p})
                    for p in rng.uniform(1e-6, 1, 50):
                        rows.append({"K": k, "sel_thresh": st, "real": False, "sample": samp, "pval": p})
        test_stats_dfs = pd.DataFrame(rows)

        utest_mod.plot_calibration_comparison(test_stats_dfs)

        for k in [5, 15]:
            for st in sel_thresh_list:
                thresh_str = str(st).replace(".", "_")
                expected = tmp_path / run_name / "Evaluation" / f"{k}_{thresh_str}" / "U_test_perturbation_association_calibration.png"
                _assert_valid_png(expected)

        thresh_str = str(sel_thresh_list[0]).replace(".", "_")
        missing = tmp_path / run_name / "Evaluation" / f"10_{thresh_str}" / "U_test_perturbation_association_calibration.png"
        assert not missing.exists(), f"Calibration plot should not exist for empty K=10: {missing}"


# ===========================================================================
# Tests for load_*_perturbation_tests (discover per-(K, sel_thresh, sample) files)
# ===========================================================================

class TestLoadFunctions:
    """Tests for load_real_perturbation_tests and load_fake_perturbation_tests."""

    def _write_dummy_results(self, tmp_path, run_name, components, sel_thresh_list,
                             samples, fake=False, n_rows=10):
        """Create dummy per-(K, sample) result files in Evaluation/{K}_{thresh}/."""
        prefix = "fake_" if fake else ""
        rng = np.random.default_rng(0)
        for k in components:
            for st in sel_thresh_list:
                thresh_str = str(st).replace(".", "_")
                folder = tmp_path / run_name / "Evaluation" / f"{k}_{thresh_str}"
                folder.mkdir(parents=True, exist_ok=True)
                for samp in samples:
                    df = pd.DataFrame({
                        "target_name": [f"g{i}" for i in range(n_rows)],
                        "pval": rng.uniform(0, 1, n_rows),
                        "adj_pval": rng.uniform(0, 1, n_rows),
                    })
                    if fake:
                        df["K"] = k
                        df["sel_thresh"] = st
                        df["batch"] = samp
                        df["run"] = 0
                    fname = f"{k}_{prefix}perturbation_association_results_{samp}.txt"
                    df.to_csv(folder / fname, sep="\t", index=False)

    def test_load_real_perturbation_tests(self, tmp_path):
        """load_real_perturbation_tests discovers samples and stacks per-(K, thresh, sample)."""
        from types import SimpleNamespace
        utest_mod = _load_utest_module()

        run_name = "test_run"
        components = [5, 10]
        sel_thresh_list = [0.2, 2.0]
        samples = ["d0", "d1"]
        n_rows = 10

        self._write_dummy_results(
            tmp_path, run_name, components, sel_thresh_list, samples, fake=False, n_rows=n_rows,
        )

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        df = utest_mod.load_real_perturbation_tests()

        assert isinstance(df, pd.DataFrame)
        assert set(df["sample"].unique()) == set(samples)
        assert set(df["K"].unique()) == set(components)
        assert set(df["sel_thresh"].unique()) == set(sel_thresh_list)
        assert all(df["real"] == True)
        expected_rows = len(components) * len(sel_thresh_list) * len(samples) * n_rows
        assert len(df) == expected_rows

    def test_load_fake_perturbation_tests(self, tmp_path):
        """load_fake_perturbation_tests discovers samples and stacks per-(K, thresh, sample)."""
        from types import SimpleNamespace
        utest_mod = _load_utest_module()

        run_name = "test_run"
        components = [5, 10]
        sel_thresh_list = [0.2, 2.0]
        samples = ["d0", "d1"]
        n_rows = 10

        self._write_dummy_results(
            tmp_path, run_name, components, sel_thresh_list, samples, fake=True, n_rows=n_rows,
        )

        utest_mod.args = SimpleNamespace(
            out_dir=str(tmp_path),
            run_name=run_name,
            components=components,
            sel_thresh=sel_thresh_list,
        )

        df = utest_mod.load_fake_perturbation_tests()

        assert isinstance(df, pd.DataFrame)
        assert all(df["real"] == False)
        assert set(df["K"].unique()) == set(components)
        assert set(df["sel_thresh"].unique()) == set(sel_thresh_list)
        assert set(df["batch"].unique()) == set(samples)
        expected_rows = len(components) * len(sel_thresh_list) * len(samples) * n_rows
        assert len(df) == expected_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
