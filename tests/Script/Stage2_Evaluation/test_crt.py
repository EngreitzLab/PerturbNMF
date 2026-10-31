"""
Unit tests for the CRT (Conditional Randomization Test) calibration.

Runs in the NMF_Benchmarking conda environment (same as the other Stage 2 tests).

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd <repo root>
    python -m pytest tests/Script/Stage2_Evaluation/test_crt.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# CRT import: the CRT package (B_Calibration/src/CRT/) and the Slurm script
# (Slurm_version/CRT/CRT.py), loaded by path since both are named CRT.
# ---------------------------------------------------------------------------

CALIB_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "src",
        "Stage2_Evaluation",
        "B_Calibration",
    )
)
CRT_SCRIPT = os.path.join(CALIB_DIR, "Slurm_version", "CRT", "CRT.py")

try:
    import importlib.util

    sys.path.insert(0, os.path.join(CALIB_DIR, "src"))
    from CRT import format_results_table, load_modality, split_by_condition  # noqa: F401

    _spec = importlib.util.spec_from_file_location("crt_pipeline", CRT_SCRIPT)
    crt_pipeline = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(crt_pipeline)
    _HAS_CRT = True
except ImportError:
    _HAS_CRT = False

needs_crt = pytest.mark.skipif(
    not _HAS_CRT, reason="CRT dependencies (numba, mudata, ...) not available"
)
slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _make_crt_args(*extra):
    """CRT.py's argparse namespace with fast test settings; extra = more CLI flags."""
    return crt_pipeline.build_parser().parse_args(
        [
            "--out_dir",
            "unused",
            "--run_name",
            "unused",
            "--categorical_key",
            "batch",
            "--number_guide",
            "2",
            "--number_permutations",
            "19",
            "--extra_resampled_rounds",
            "0",
            "--n_jobs",
            "1",
            *extra,
        ]
    )


def _one_batch(mdata, batch="D0"):
    """MuData with only the cNMF modality and one batch, so run_CRT runs one condition."""
    import mudata

    cnmf = mdata["cNMF"]
    return mudata.MuData({"cNMF": cnmf[cnmf.obs["batch"] == batch].copy()})


def _fake_crt_obj(n_genes=4, n_programs=3, normalization="row_normalize", seed=0):
    """Minimal CRT_obj-like object with gene_results, enough for format_results_table."""
    from types import SimpleNamespace
    from CRT import CRT_gene_result

    rng = np.random.default_rng(seed)
    genes = [f"gene{g}" for g in range(n_genes)]
    results = {
        g: CRT_gene_result(
            gene=g,
            n_treated=10,
            treated_obs_idx=np.empty(0, dtype=np.int32),
            pvalue=rng.uniform(size=n_programs),
            betas=rng.normal(0, 0.01, n_programs),
            pvalue_empirical=rng.uniform(size=n_programs),
            control_means=np.full(n_programs, 0.2),
            n_resampled=99,
            n_control=90,
        )
        for g in genes
    }
    return SimpleNamespace(
        gene_results=results,
        selected_gene_names=genes,
        selected_program_names=[str(p + 1) for p in range(n_programs)],
        normalization=normalization,
        guide_names=["gA", "gB"],
        guide_targets={g: [0, 1] for g in genes},
    )


# ===========================================================================
# split_by_condition
# ===========================================================================


@needs_crt
class TestSplitByCondition:
    def test_split_covers_every_cell_once(self, test_mdata):
        import mudata

        mdata = mudata.MuData({"cNMF": test_mdata["cNMF"].copy()})
        parts = dict(split_by_condition(mdata, "cNMF", "batch"))
        assert set(parts) == set(test_mdata["cNMF"].obs["batch"].unique())
        assert sum(m["cNMF"].n_obs for m in parts.values()) == test_mdata["cNMF"].n_obs
        for cond, m in parts.items():
            assert (m["cNMF"].obs["batch"] == cond).all()

    def test_split_selected_conditions(self, test_mdata):
        import mudata

        mdata = mudata.MuData({"cNMF": test_mdata["cNMF"].copy()})
        assert [c for c, _ in split_by_condition(mdata, "cNMF", "batch", ["D1"])] == [
            "D1"
        ]

    def test_split_missing_key_raises(self, test_mdata):
        import mudata

        mdata = mudata.MuData({"cNMF": test_mdata["cNMF"].copy()})
        with pytest.raises(KeyError):
            next(split_by_condition(mdata, "cNMF", "not_a_column"))


# ===========================================================================
# format_results_table (PerturbNMF format, read by Stage 3)
# ===========================================================================


@needs_crt
class TestFormatResultsTable:
    def test_columns_and_program_numbers(self):
        from CRT.utility import RESULTS_COLUMNS

        table = format_results_table(_fake_crt_obj())
        assert list(table.columns) == RESULTS_COLUMNS
        assert list(table.columns[:7]) == [
            "target_name",
            "program_name",
            "log2FC",
            "p-value",
            "adj_pval",
            "pval_empirical",
            "adj_pval_empirical",
        ]
        assert sorted(table["program_name"].unique()) == ["1", "2", "3"]
        assert len(table) == 4 * 3
        assert (
            table["pval_skewnorm"].isna().all()
        )  # no skew-normal fit in the fake results
        assert table["adj_pval"].between(0, 1).all()

    def test_guides_column(self):
        table = format_results_table(_fake_crt_obj(), include_guides=True)
        assert (table["guides"] == "gA;gB").all()

    def test_log2fc_row_normalize(self):
        from CRT.utility import log2fc_from_beta

        np.testing.assert_allclose(
            log2fc_from_beta([0.2, -0.05], [0.2, 0.1], "row_normalize", 2), [1.0, -1.0]
        )

    def test_log2fc_clr(self):
        from CRT.utility import log2fc_from_beta

        np.testing.assert_allclose(
            log2fc_from_beta([np.log(2), 0.0], [0.0, 0.0], "clr", 2), [2.0, 0.0]
        )

    def test_round_trip_through_txt(self, tmp_path):
        from CRT import load_results_table

        obj = _fake_crt_obj()
        path = str(tmp_path / "5_CRT_no_covariates_D0.txt")
        format_results_table(obj, include_guides=True).to_csv(
            path, sep="\t", index=False
        )
        reloaded = _fake_crt_obj(seed=1)
        reloaded.pvalue_method = "empirical"
        load_results_table(reloaded, path)
        for g, r in obj.gene_results.items():
            np.testing.assert_allclose(reloaded.gene_results[g].pvalue, r.pvalue)
            np.testing.assert_allclose(reloaded.gene_results[g].betas, r.betas)

    def test_covariate_tag(self):
        args = _make_crt_args(
            "--covariates", "pct_counts_mt", "--log_covariates", "total_counts"
        )
        assert crt_pipeline._covariate_tag(args) == "pct_counts_mt_log_total_counts"
        assert crt_pipeline._covariate_tag(_make_crt_args()) == "no_covariates"


# ===========================================================================
# run_CRT: end-to-end on one batch of the inference output
# ===========================================================================


@needs_crt
class TestCRTRunCRT:
    def _outputs(self, path, k=5):
        return {
            "txt": sorted(path.glob(f"{k}_CRT_*_D0.txt")),
            "pdf": sorted(path.glob(f"{k}_CRT_*_qq_*.pdf")),
        }

    @slow
    def test_run_CRT_writes_outputs(self, test_mdata, tmp_path):
        args = _make_crt_args()
        crt_pipeline.run_CRT(_one_batch(test_mdata), 5, 2.0, str(tmp_path), args)

        real = tmp_path / "5_CRT_no_covariates_D0.txt"
        fake = tmp_path / "5_CRT_fake_no_covariates_D0.txt"
        assert real.exists() and fake.exists()
        assert self._outputs(tmp_path)["pdf"], f"No QQ PDF in {tmp_path}"

        df = pd.read_csv(real, sep="\t")
        assert df["p-value"].between(0, 1).all()
        assert sorted(df["program_name"].unique()) == list(range(1, 6))
        assert "non-targeting" not in set(
            df["target_name"]
        )  # control targets are not tested

    @slow
    def test_run_CRT_with_covariates_and_StoreyQ(self, test_mdata, tmp_path):
        try:
            import multipy.fdr  # noqa: F401
        except ImportError:
            pytest.skip("multipy not installed — StoreyQ branch unavailable")
        args = _make_crt_args(
            "--covariates",
            "pct_counts_mt",
            "--log_covariates",
            "total_counts",
            "--FDR_method",
            "StoreyQ",
            "--skip_null",
        )
        crt_pipeline.run_CRT(_one_batch(test_mdata), 5, 2.0, str(tmp_path), args)
        df = pd.read_csv(
            tmp_path / "5_CRT_pct_counts_mt_log_total_counts_D0.txt", sep="\t"
        )
        assert np.isfinite(df["adj_pval"]).all()
        assert not list(tmp_path.glob("5_CRT_fake_*"))

    @slow
    def test_run_CRT_low_moi(self, test_mdata, tmp_path):
        args = _make_crt_args("--moi", "low", "--normalization", "clr")
        crt_pipeline.run_CRT(_one_batch(test_mdata), 5, 2.0, str(tmp_path), args)
        df = pd.read_csv(tmp_path / "5_CRT_no_covariates_D0.txt", sep="\t")
        assert (tmp_path / "5_CRT_fake_no_covariates_D0.txt").exists()

    @slow
    def test_run_CRT_skip_existing_reuses_results(
        self, test_mdata, tmp_path, monkeypatch
    ):
        mdata = _one_batch(test_mdata)
        crt_pipeline.run_CRT(mdata, 5, 2.0, str(tmp_path), _make_crt_args())
        first = pd.read_csv(tmp_path / "5_CRT_no_covariates_D0.txt", sep="\t")

        def no_recompute(*a, **kw):
            raise AssertionError("--skip_existing should reuse the cached CRT tables")

        monkeypatch.setattr(crt_pipeline, "conditional_randomized_test", no_recompute)
        crt_pipeline.run_CRT(
            mdata, 5, 2.0, str(tmp_path), _make_crt_args("--skip_existing")
        )
        second = pd.read_csv(tmp_path / "5_CRT_no_covariates_D0.txt", sep="\t")
        pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-12)


# ===========================================================================
# main(): CLI integration via sys.argv
# ===========================================================================


@needs_crt
class TestCRTMain:
    @slow
    def test_main_smoke(self, test_mdata, tmp_path, monkeypatch):
        """main() reads {out_dir}/{run_name}/Inference/adata/cNMF_5_2_0.h5mu and runs per batch."""
        out_dir = tmp_path / "out"
        adata_dir = out_dir / "smoke" / "Inference" / "adata"
        adata_dir.mkdir(parents=True)
        _one_batch(test_mdata).write(adata_dir / "cNMF_5_2_0.h5mu")
        save_dir = tmp_path / "results"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "CRT.py",
                "--out_dir",
                str(out_dir),
                "--run_name",
                "smoke",
                "--K",
                "5",
                "--sel_threshs",
                "2.0",
                "--categorical_key",
                "batch",
                "--number_guide",
                "2",
                "--number_permutations",
                "19",
                "--extra_resampled_rounds",
                "0",
                "--n_jobs",
                "1",
                "--save_dir",
                str(save_dir),
            ],
        )
        assert crt_pipeline.main() == 0
        assert list((save_dir / "5_2_0").glob("5_CRT_*_D0.txt"))
        assert list(save_dir.glob("config_*.yml")), (
            "main() should have written a config_*.yml"
        )
