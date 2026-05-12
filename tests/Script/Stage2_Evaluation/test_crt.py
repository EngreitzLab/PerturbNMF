"""
Unit tests for the CRT (Conditional Randomization Test) calibration.

Requires the programDE conda environment (sceptre dependency).

Usage:
    eval "$(conda shell.bash hook)" && conda activate programDE
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage2_Evaluation/test_crt.py -v
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CRT import (skip entire module if sceptre unavailable)
# ---------------------------------------------------------------------------

CRT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src",
                       "Stage2_Evaluation", "B_Calibration", "Slurm_version", "CRT")
CRT_DIR = os.path.abspath(CRT_DIR)

try:
    sys.path.insert(0, CRT_DIR)
    import CRT as crt_mod  # noqa: F401 — needs sceptre (programDE env)
    from CRT import reformat_data_for_CRT, run_CRT, save_result, main as crt_main
    _HAS_CRT = True
except ImportError:
    _HAS_CRT = False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _make_crt_args(
    *,
    categorical_key="batch",
    covariates=None,
    log_covariates=None,
    number_guide=2,
    number_permutations=8,
    guide_annotation_key="non-targeting",
    FDR_method="BH",
    out_dir=None,
    run_name=None,
    save_dir=None,
    components=None,
    sel_thresh=None,
):
    """Build a SimpleNamespace mirroring CRT.main's argparse namespace."""
    return SimpleNamespace(
        out_dir=out_dir,
        run_name=run_name,
        components=components,
        sel_thresh=sel_thresh,
        categorical_key=categorical_key,
        covariates=covariates,
        log_covariates=log_covariates,
        number_guide=number_guide,
        number_permutations=number_permutations,
        guide_annotation_key=guide_annotation_key,
        FDR_method=FDR_method,
        save_dir=save_dir,
    )


def _subset_to_one_batch(mdata, batch_col="batch", batch=None):
    """Return a MuData restricted to a single batch — keeps run_CRT iterations to 1."""
    import mudata
    rna = mdata["rna"].copy()
    cnmf = mdata["cNMF"].copy()
    if batch is None:
        batch = cnmf.obs[batch_col].iloc[0]
    mask = cnmf.obs[batch_col] == batch
    cnmf = cnmf[mask].copy()
    rna = rna[mask].copy()
    return mudata.MuData({"rna": rna, "cNMF": cnmf})


def _build_fake_crt_out(n_targets=4, n_programs=3, seed=0):
    """Build a minimal `out` dict matching `run_all_genes_union_crt` return,
    just enough for `save_result` (pvals_skew_df + betas_df)."""
    rng = np.random.default_rng(seed)
    targets = [f"gene{i}" for i in range(n_targets)]
    programs = [f"K{i}" for i in range(n_programs)]
    pvals = pd.DataFrame(rng.uniform(1e-6, 1, (n_targets, n_programs)),
                         index=targets, columns=programs)
    betas = pd.DataFrame(rng.normal(0, 0.5, (n_targets, n_programs)),
                         index=targets, columns=programs)
    return {"pvals_skew_df": pvals, "betas_df": betas}


# ===========================================================================
# Reformat: structural tests
# ===========================================================================

@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTReformat:
    """Test the reformat_data_for_CRT function from the CRT calibration script."""

    def test_reformat_basic_structure(self, test_mdata):
        """reformat_data_for_CRT should produce expected keys in obsm/uns."""
        adata = reformat_data_for_CRT(test_mdata.copy())
        assert "cnmf_usage" in adata.obsm
        assert "guide_assignment" in adata.obsm
        assert "guide_names" in adata.uns
        assert "guide2gene" in adata.uns
        assert "program_names" in adata.uns
        assert "covar" in adata.obsm

    def test_reformat_guide2gene_mapping(self, test_mdata):
        """guide2gene dict should map every guide name to its target."""
        adata = reformat_data_for_CRT(test_mdata.copy())
        g2g = adata.uns["guide2gene"]
        assert isinstance(g2g, dict)
        assert len(g2g) == len(adata.uns["guide_names"])
        for gname in adata.uns["guide_names"]:
            assert gname in g2g

    def test_reformat_program_names_match_var(self, test_mdata):
        """program_names should equal cNMF var_names (the K columns)."""
        adata = reformat_data_for_CRT(test_mdata.copy())
        assert list(adata.uns["program_names"]) == list(adata.var_names)

    def test_reformat_cnmf_usage_dense_float(self, test_mdata):
        """cnmf_usage should be a dense ndarray, not sparse."""
        adata = reformat_data_for_CRT(test_mdata.copy())
        usage = adata.obsm["cnmf_usage"]
        assert isinstance(usage, np.ndarray)
        assert usage.shape == (adata.n_obs, adata.n_vars)


# ===========================================================================
# Reformat: covariate variants
# ===========================================================================

@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTReformatCovariates:
    """Vary the `covariates` / `log_covariates` arguments and check covar matrix."""

    def test_reformat_no_covariates_empty_covar(self, test_mdata):
        """With no covariates, covar should be a DataFrame with 0 columns."""
        adata = reformat_data_for_CRT(test_mdata.copy(),
                                      covariates=None, log_covariates=None)
        covar = adata.obsm["covar"]
        assert isinstance(covar, pd.DataFrame)
        assert covar.shape == (adata.n_obs, 0)

    def test_reformat_single_raw_covariate(self, test_mdata):
        """One raw covariate should produce one column in covar with matching values."""
        adata = reformat_data_for_CRT(test_mdata.copy(), covariates=["total_counts"])
        covar = adata.obsm["covar"]
        assert list(covar.columns) == ["total_counts"]
        assert covar.shape == (adata.n_obs, 1)
        # Values should match the source obs column
        np.testing.assert_array_equal(
            covar["total_counts"].values,
            adata.obs["total_counts"].values,
        )

    def test_reformat_multiple_raw_covariates(self, test_mdata):
        """Multiple raw covariates should all appear as separate columns."""
        covs = ["total_counts", "pct_counts_mt", "n_genes_by_counts"]
        adata = reformat_data_for_CRT(test_mdata.copy(), covariates=covs)
        covar = adata.obsm["covar"]
        assert list(covar.columns) == covs
        assert covar.shape == (adata.n_obs, len(covs))

    def test_reformat_log_covariates_prefix(self, test_mdata):
        """log_covariates should be prefixed `log_` and equal log1p(obs[key])."""
        adata = reformat_data_for_CRT(test_mdata.copy(),
                                      log_covariates=["guide_umi_counts"])
        covar = adata.obsm["covar"]
        assert list(covar.columns) == ["log_guide_umi_counts"]
        expected = np.log1p(adata.obs["guide_umi_counts"].values)
        np.testing.assert_allclose(covar["log_guide_umi_counts"].values, expected)

    def test_reformat_combined_covariates_and_log(self, test_mdata):
        """Raw + log covariates should both appear with correct prefixes."""
        adata = reformat_data_for_CRT(
            test_mdata.copy(),
            covariates=["pct_counts_mt"],
            log_covariates=["total_counts", "guide_umi_counts"],
        )
        covar = adata.obsm["covar"]
        assert "pct_counts_mt" in covar.columns
        assert "log_total_counts" in covar.columns
        assert "log_guide_umi_counts" in covar.columns
        assert covar.shape == (adata.n_obs, 3)


# ===========================================================================
# Reformat: parametrized across K values
# ===========================================================================

@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTReformatPerK:
    """Verify reformat works across the inference K values (5, 10, 15)."""

    def test_reformat_per_k_structure(self, mdata_copy_per_k):
        adata = reformat_data_for_CRT(mdata_copy_per_k)
        k = mdata_copy_per_k.uns["test_k"]
        assert adata.n_vars == k
        assert len(adata.uns["program_names"]) == k
        assert adata.obsm["cnmf_usage"].shape == (adata.n_obs, k)


# ===========================================================================
# save_result: FDR method variants
# ===========================================================================

@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTSaveResult:
    """save_result should write a TSV with adj_pval correctly computed for each FDR method."""

    def test_save_result_BH_writes_tsv(self, tmp_path):
        out = _build_fake_crt_out(n_targets=5, n_programs=3)
        args = _make_crt_args(FDR_method="BH")
        df = save_result(out, k=5, output_folder=str(tmp_path), condition="D0", args=args)

        # File written to expected path
        expected = tmp_path / "5_CRT_D0.txt"
        assert expected.exists()

        loaded = pd.read_csv(expected, sep="\t")
        for col in ["target_name", "program_name", "log2FC", "p-value", "adj_pval"]:
            assert col in loaded.columns
        # adj_pval is finite and bounded
        assert loaded["adj_pval"].between(0, 1).all()
        # Returned df should match the saved one (modulo dtype quirks)
        assert len(df) == len(loaded) == 5 * 3

    def test_save_result_StoreyQ_writes_tsv(self, tmp_path):
        """StoreyQ branch should also produce an adj_pval column with finite values."""
        try:
            import multipy.fdr  # noqa: F401
        except ImportError:
            pytest.skip("multipy not installed — StoreyQ branch unavailable")

        out = _build_fake_crt_out(n_targets=5, n_programs=3, seed=1)
        args = _make_crt_args(FDR_method="StoreyQ")
        df = save_result(out, k=10, output_folder=str(tmp_path), condition="D1", args=args)

        expected = tmp_path / "10_CRT_D1.txt"
        assert expected.exists()
        loaded = pd.read_csv(expected, sep="\t")
        assert "adj_pval" in loaded.columns
        # Q-values should be finite
        assert np.isfinite(loaded["adj_pval"]).all()
        assert len(df) == 5 * 3

    def test_save_result_column_order(self, tmp_path):
        """save_result column order should be stable: target, program, log2FC, p-value, adj_pval."""
        out = _build_fake_crt_out(n_targets=3, n_programs=2)
        args = _make_crt_args(FDR_method="BH")
        df = save_result(out, k=5, output_folder=str(tmp_path), condition="D2", args=args)
        assert list(df.columns) == [
            "target_name", "program_name", "log2FC", "p-value", "adj_pval",
        ]


# ===========================================================================
# run_CRT: end-to-end smoke tests on real h5mu
# ===========================================================================

# Mark heavy tests so they can be deselected with `-m "not slow"` if desired.
slow = pytest.mark.slow


@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTRunCRT:
    """Exercise the full run_CRT pipeline on a single batch with tiny B."""

    def _prepare_adata(self, mdata, covariates=None, log_covariates=None):
        """Subset to one batch and run reformat + flooring (mirrors main())."""
        mdata_one = _subset_to_one_batch(mdata)
        adata = reformat_data_for_CRT(mdata_one,
                                      covariates=covariates,
                                      log_covariates=log_covariates)
        U = adata.obsm["cnmf_usage"].copy()
        U = np.maximum(U, 1e-8)
        U /= U.sum(axis=1, keepdims=True)
        adata.obsm["cnmf_usage"] = U
        return adata

    @slow
    def test_run_CRT_small_guide_group(self, test_mdata, tmp_path):
        """Tiny number_guide (2) — smoke test that pipeline runs and writes outputs."""
        import matplotlib
        matplotlib.use("Agg")
        adata = self._prepare_adata(test_mdata)
        args = _make_crt_args(number_guide=2, number_permutations=8, FDR_method="BH")
        run_CRT(adata, k=5, output_folder=str(tmp_path), args=args)

        # At least one batch should produce a PNG + TXT
        pngs = list(tmp_path.glob("CRT_*.png"))
        txts = list(tmp_path.glob("5_CRT_*.txt"))
        assert len(pngs) >= 1, f"No PNG outputs in {tmp_path}"
        assert len(txts) >= 1, f"No TXT outputs in {tmp_path}"

        # Sanity-check the TXT structure
        df = pd.read_csv(txts[0], sep="\t")
        for col in ["target_name", "program_name", "log2FC", "p-value", "adj_pval"]:
            assert col in df.columns
        assert df["p-value"].between(0, 1).all()

    @slow
    def test_run_CRT_large_guide_group(self, test_mdata, tmp_path):
        """Larger number_guide (10) — should still run if enough NTC guides are available.

        Test data has 103 non-targeting guides, so a group size of 10 is well within range.
        """
        import matplotlib
        matplotlib.use("Agg")
        adata = self._prepare_adata(test_mdata)
        args = _make_crt_args(number_guide=10, number_permutations=8, FDR_method="BH")
        run_CRT(adata, k=5, output_folder=str(tmp_path), args=args)

        assert list(tmp_path.glob("5_CRT_*.txt")), f"No TXT outputs in {tmp_path}"

    @slow
    def test_run_CRT_with_covariates(self, test_mdata, tmp_path):
        """Run with raw + log covariates included — covar matrix flows into prepare_crt_inputs."""
        import matplotlib
        matplotlib.use("Agg")
        adata = self._prepare_adata(
            test_mdata,
            covariates=["pct_counts_mt"],
            log_covariates=["total_counts", "guide_umi_counts"],
        )
        # Sanity: covar made it onto adata
        assert adata.obsm["covar"].shape[1] == 3

        args = _make_crt_args(number_guide=2, number_permutations=8, FDR_method="BH")
        run_CRT(adata, k=5, output_folder=str(tmp_path), args=args)
        assert list(tmp_path.glob("5_CRT_*.txt"))

    @slow
    def test_run_CRT_more_permutations(self, test_mdata, tmp_path):
        """Higher number_permutations — verifies the B parameter is wired through correctly."""
        import matplotlib
        matplotlib.use("Agg")
        adata = self._prepare_adata(test_mdata)
        args = _make_crt_args(number_guide=2, number_permutations=32, FDR_method="BH")
        run_CRT(adata, k=5, output_folder=str(tmp_path), args=args)
        assert list(tmp_path.glob("5_CRT_*.txt"))

    @slow
    def test_run_CRT_StoreyQ_fdr(self, test_mdata, tmp_path):
        """Run with StoreyQ FDR — verifies the alternative FDR branch is exercised end-to-end."""
        try:
            import multipy.fdr  # noqa: F401
        except ImportError:
            pytest.skip("multipy not installed — StoreyQ branch unavailable")

        import matplotlib
        matplotlib.use("Agg")
        adata = self._prepare_adata(test_mdata)
        args = _make_crt_args(number_guide=2, number_permutations=8, FDR_method="StoreyQ")
        run_CRT(adata, k=5, output_folder=str(tmp_path), args=args)

        txts = list(tmp_path.glob("5_CRT_*.txt"))
        assert txts
        df = pd.read_csv(txts[0], sep="\t")
        assert np.isfinite(df["adj_pval"]).all()


# ===========================================================================
# main(): CLI integration via sys.argv
# ===========================================================================

@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTMain:
    """Exercise `main()` end-to-end with a constructed argv. Heavy — marked slow."""

    @slow
    def test_main_smoke_BH(self, test_mdata, tmp_path, monkeypatch):
        """Run main() with --components 5 --sel_thresh 2.0 --FDR_method BH.

        Builds a one-batch h5mu at the path main() expects to read
        ({out_dir}/{run_name}/Inference/adata/cNMF_5_2_0.h5mu), then invokes main()
        via sys.argv. All outputs routed to tmp_path/results via --save_dir.
        """
        import matplotlib
        matplotlib.use("Agg")

        # Subset to one batch so the run_CRT inner loop iterates exactly once.
        mdata_one = _subset_to_one_batch(test_mdata)

        out_dir = tmp_path / "out"
        run_name = "smoke"
        adata_dir = out_dir / run_name / "Inference" / "adata"
        adata_dir.mkdir(parents=True, exist_ok=True)
        h5mu_path = adata_dir / "cNMF_5_2_0.h5mu"
        mdata_one.write(h5mu_path)

        save_dir = tmp_path / "results"
        save_dir.mkdir()

        argv = [
            "CRT.py",
            "--out_dir", str(out_dir),
            "--run_name", run_name,
            "--components", "5",
            "--sel_thresh", "2.0",
            "--categorical_key", "batch",
            "--number_guide", "2",
            "--number_permutations", "8",
            "--FDR_method", "BH",
            "--save_dir", str(save_dir),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        rc = crt_main()
        assert rc == 0

        # main() writes PNG + TXT into save_dir (because args.save_dir is set,
        # output_folder == save_dir in run_CRT)
        assert list(save_dir.glob("CRT_*.png"))
        assert list(save_dir.glob("5_CRT_*.txt"))
        # And a config_*.yml at save_config_dir (which == save_dir here)
        assert list(save_dir.glob("config_*.yml")), "main() should have written a config_*.yml"
