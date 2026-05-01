"""
Edge-case tests for torch-cNMF inference pipeline (GPU).

Tests boundary conditions and parameter variations that could cause crashes.

Run via SLURM:
    sbatch tests/Script/run_gpu_test_edge_cases.sh

Or directly on a GPU node:
    conda activate torch-nmf-dl
    python -m pytest tests/Script/Inference/test_edge_cases.py -v
"""

import os
import sys
import numpy as np
import pytest
import anndata as ad
import mudata as mu
import scipy.sparse as sp

from .conftest import TEST_SEED, PERSISTENT_OUTPUT

from torch_cnmf import cNMF
from Inference.src.run_cNMF import run_cnmf_consensus, compile_results

# Minimal params for fast tests
# Need >= 5 iters for density filtering to work at thresh=2.0
EDGE_K = [5]
EDGE_NUMITER = 5
EDGE_NUMHVGENES = 2000
EDGE_THRESH = [2.0]

OUTPUT_BASE = str(PERSISTENT_OUTPUT / "torch-cNMF-edge-cases")


def _run_minimal(mini_h5ad_path, test_name, **prepare_overrides):
    """Run a minimal inference pipeline and return output_dir.

    Runs: prepare → factorize → combine → consensus.
    Returns output_dir for assertions.
    """
    output_dir = os.path.join(OUTPUT_BASE, test_name)
    os.makedirs(output_dir, exist_ok=True)
    run_name = "Inference"

    cnmf_obj = cNMF(output_dir=output_dir, name=run_name)

    defaults = dict(
        counts_fn=mini_h5ad_path,
        components=EDGE_K,
        n_iter=EDGE_NUMITER,
        densify=False,
        tpm_fn=None,
        num_highvar_genes=EDGE_NUMHVGENES,
        genes_file=None,
        beta_loss="frobenius",
        init="random",
        algo="halsvar",
        mode="batch",
        tol=1e-4,
        seed=TEST_SEED,
        use_gpu=True,
        alpha_usage=0.0,
        alpha_spectra=0.0,
        fp_precision="float",
        batch_max_epoch=50,
        batch_hals_tol=0.05,
        batch_hals_max_iter=200,
        l1_ratio_usage=0.0,
        l1_ratio_spectra=0.0,
    )
    defaults.update(prepare_overrides)

    cnmf_obj.prepare(**defaults)
    cnmf_obj.factorize(skip_completed_runs=True)
    cnmf_obj.combine()
    run_cnmf_consensus(
        cnmf_obj=cnmf_obj,
        components=EDGE_K,
        density_thresholds=EDGE_THRESH,
    )

    return output_dir


def _assert_basic_outputs(output_dir):
    """Check that minimal expected outputs exist."""
    inference_dir = os.path.join(output_dir, "Inference")
    missing = []

    for k in EDGE_K:
        for thresh in EDGE_THRESH:
            ts = str(thresh).replace(".", "_")
            files = [
                f"Inference.usages.k_{k}.dt_{ts}.consensus.txt",
                f"Inference.gene_spectra_score.k_{k}.dt_{ts}.txt",
            ]
            for f in files:
                if not os.path.exists(os.path.join(inference_dir, f)):
                    missing.append(f)

        # At least one spectra iter file
        cnmf_tmp = os.path.join(inference_dir, "cnmf_tmp")
        spectra = os.path.join(cnmf_tmp, f"Inference.spectra.k_{k}.iter_0.df.npz")
        if not os.path.exists(spectra):
            missing.append(f"cnmf_tmp/Inference.spectra.k_{k}.iter_0.df.npz")

    assert not missing, f"Missing outputs:\n" + "\n".join(f"  {f}" for f in missing)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_minibatch_shuffle(self, mini_h5ad_path):
        """Minibatch mode with shuffle=True completes."""
        output_dir = _run_minimal(
            mini_h5ad_path, "minibatch_shuffle",
            mode="minibatch",
            minibatch_shuffle=True,
            minibatch_size=500,
            minibatch_max_epoch=3,
            minibatch_max_iter=50,
            minibatch_usage_tol=0.05,
            minibatch_spectra_tol=0.05,
        )
        _assert_basic_outputs(output_dir)

    def test_minibatch_no_shuffle(self, mini_h5ad_path):
        """Minibatch mode with shuffle=False completes."""
        output_dir = _run_minimal(
            mini_h5ad_path, "minibatch_no_shuffle",
            mode="minibatch",
            minibatch_shuffle=False,
            minibatch_size=500,
            minibatch_max_epoch=3,
            minibatch_max_iter=50,
            minibatch_usage_tol=0.05,
            minibatch_spectra_tol=0.05,
        )
        _assert_basic_outputs(output_dir)

    def test_custom_seeds(self, mini_h5ad_path):
        """Custom NMF seeds produce output."""
        seeds = np.array([42, 43, 44, 45, 46])
        seeds_path = os.path.join(OUTPUT_BASE, "custom_seeds.npy")
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        np.save(seeds_path, seeds)

        output_dir = _run_minimal(
            mini_h5ad_path, "custom_seeds",
            nmf_seeds=seeds,
        )
        _assert_basic_outputs(output_dir)

    def test_cpu_only(self, mini_h5ad_path):
        """use_gpu=False falls back to CPU without error."""
        output_dir = _run_minimal(
            mini_h5ad_path, "cpu_only",
            use_gpu=False,
        )
        _assert_basic_outputs(output_dir)

    def test_remove_noncoding(self, mini_h5ad_path):
        """Pre-filtering genes with ENSG prefix works."""
        adata = ad.read_h5ad(mini_h5ad_path)

        # Filter out any genes starting with "ENSG" (Ensembl IDs)
        mask = ~adata.var_names.str.startswith("ENSG")
        n_before = adata.n_vars
        adata_filtered = adata[:, mask].copy()
        n_after = adata_filtered.n_vars
        print(f"  Filtered genes: {n_before} → {n_after} (removed {n_before - n_after} ENSG)")

        # Save filtered data
        filtered_dir = os.path.join(OUTPUT_BASE, "remove_noncoding")
        os.makedirs(filtered_dir, exist_ok=True)
        filtered_path = os.path.join(filtered_dir, "filtered.h5ad")
        adata_filtered.write(filtered_path)

        output_dir = _run_minimal(
            filtered_path, "remove_noncoding",
        )
        _assert_basic_outputs(output_dir)

    def test_minibatch_size_larger_than_data(self, mini_h5ad_path):
        """minibatch_size >> n_cells doesn't crash."""
        output_dir = _run_minimal(
            mini_h5ad_path, "large_minibatch",
            mode="minibatch",
            minibatch_size=999999,
            minibatch_max_epoch=3,
            minibatch_max_iter=50,
            minibatch_usage_tol=0.05,
            minibatch_spectra_tol=0.05,
        )
        _assert_basic_outputs(output_dir)

    def test_numhvgenes_larger_than_total(self, mini_h5ad_path):
        """num_highvar_genes > total genes clamps gracefully."""
        output_dir = _run_minimal(
            mini_h5ad_path, "large_numhvgenes",
            num_highvar_genes=999999,
        )
        _assert_basic_outputs(output_dir)

    def test_densify(self, mini_h5ad_path):
        """densify=True works with dense matrix input."""
        output_dir = _run_minimal(
            mini_h5ad_path, "densify",
            densify=True,
        )
        _assert_basic_outputs(output_dir)

    def test_dataloader_mode_edge(self, mini_h5ad_path):
        """Dataloader mode with small minibatch_size completes."""
        output_dir = _run_minimal(
            mini_h5ad_path, "dataloader_small_batch",
            mode="dataloader",
            minibatch_size=100,
            minibatch_max_epoch=3,
            minibatch_max_iter=50,
            minibatch_usage_tol=0.05,
            minibatch_spectra_tol=0.05,
        )
        _assert_basic_outputs(output_dir)

    def test_multiple_thresholds(self, mini_h5ad_path):
        """Multiple density thresholds all produce output."""
        thresholds = [0.4, 0.8, 2.0]
        output_dir = os.path.join(OUTPUT_BASE, "multi_thresh")
        os.makedirs(output_dir, exist_ok=True)

        cnmf_obj = cNMF(output_dir=output_dir, name="Inference")
        cnmf_obj.prepare(
            counts_fn=mini_h5ad_path,
            components=EDGE_K,
            n_iter=EDGE_NUMITER,
            densify=False,
            tpm_fn=None,
            num_highvar_genes=EDGE_NUMHVGENES,
            genes_file=None,
            beta_loss="frobenius",
            init="random",
            algo="halsvar",
            mode="batch",
            tol=1e-4,
            seed=TEST_SEED,
            use_gpu=True,
            alpha_usage=0.0,
            alpha_spectra=0.0,
            fp_precision="float",
            batch_max_epoch=50,
            batch_hals_tol=0.05,
            batch_hals_max_iter=200,
        )
        cnmf_obj.factorize(skip_completed_runs=True)
        cnmf_obj.combine()
        run_cnmf_consensus(
            cnmf_obj=cnmf_obj,
            components=EDGE_K,
            density_thresholds=thresholds,
        )

        inference_dir = os.path.join(output_dir, "Inference")
        missing = []
        for k in EDGE_K:
            for thresh in thresholds:
                ts = str(thresh).replace(".", "_")
                usage = os.path.join(inference_dir, f"Inference.usages.k_{k}.dt_{ts}.consensus.txt")
                if not os.path.exists(usage):
                    missing.append(f"usages k={k} thresh={thresh}")
        assert not missing, f"Missing outputs for thresholds:\n" + "\n".join(f"  {f}" for f in missing)
        print(f"  All {len(thresholds)} thresholds produced output")

    def test_regularization(self, mini_h5ad_path):
        """Non-zero regularization completes without error."""
        output_dir = _run_minimal(
            mini_h5ad_path, "regularization",
            alpha_usage=0.1,
            alpha_spectra=0.1,
            l1_ratio_usage=0.5,
            l1_ratio_spectra=0.5,
        )
        _assert_basic_outputs(output_dir)

    def test_double_precision(self, mini_h5ad_path):
        """fp_precision='double' (float64) completes."""
        output_dir = _run_minimal(
            mini_h5ad_path, "double_precision",
            fp_precision="double",
        )
        _assert_basic_outputs(output_dir)

    def test_reproducibility(self, mini_h5ad_path):
        """Two runs with same seed produce identical spectra."""
        dir_a = _run_minimal(mini_h5ad_path, "repro_a", seed=42)
        dir_b = _run_minimal(mini_h5ad_path, "repro_b", seed=42)

        for k in EDGE_K:
            for i in range(EDGE_NUMITER):
                fname = f"Inference.spectra.k_{k}.iter_{i}.df.npz"
                path_a = os.path.join(dir_a, "Inference", "cnmf_tmp", fname)
                path_b = os.path.join(dir_b, "Inference", "cnmf_tmp", fname)

                data_a = np.load(path_a, allow_pickle=True)
                data_b = np.load(path_b, allow_pickle=True)

                np.testing.assert_array_almost_equal(
                    data_a['data'], data_b['data'],
                    decimal=5,
                    err_msg=f"Spectra mismatch for k={k} iter={i}",
                )
