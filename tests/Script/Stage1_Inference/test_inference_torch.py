"""
End-to-end smoke tests for torch-cNMF inference pipeline (GPU).

Tests three learning modes: batch, minibatch, and dataloader.

Run via SLURM:
    sbatch tests/Script/Stage1_Inference/run_gpu_test_batch.sh
    sbatch tests/Script/Stage1_Inference/run_gpu_test_minibatch.sh
    sbatch tests/Script/Stage1_Inference/run_gpu_test_dataloader.sh

Or directly on a GPU node:
    conda activate torch-nmf-dl
    python -m pytest tests/Script/Stage1_Inference/test_inference_torch.py -v
"""

import os
import sys
import logging
import pytest
import anndata as ad
import mudata as mu

from .conftest import (
    TEST_K, TEST_NUMITER, TEST_NUMHVGENES, TEST_SEL_THRESH, TEST_SEED,
    PERSISTENT_OUTPUT,
)

from torch_cnmf import cNMF
from Stage1_Inference.src.format_checking import check_data_format
from Stage1_Inference.src.run_cNMF import run_cnmf_consensus, compile_results
from Stage1_Inference.src.plot_diagnostics import generate_all_plots


# ---------------------------------------------------------------------------
# Helper: run the full inference pipeline for a given mode and return output dir
# ---------------------------------------------------------------------------

def _make_torch_output_dir(mode):
    """Create output dir for a torch-cNMF mode test.

    Always saves to tests/output/torch-cNMF/<mode>/.
    Cleanup is handled externally (e.g., by the run-tests skill).
    """
    outdir = str(PERSISTENT_OUTPUT / "torch-cNMF" / mode)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _run_full_inference(mini_h5ad_path, mode, extra_prepare_kwargs=None):
    """Run prepare → factorize → combine → consensus → compile for a given mode.

    Returns (output_dir, adata_dir) so callers can inspect outputs.
    """
    output_dir = _make_torch_output_dir(mode)
    run_name = "Inference"

    # Set up per-mode logging
    logs_dir = os.path.join(output_dir, run_name, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    _log_handler = logging.FileHandler(
        os.path.join(logs_dir, f"test_{mode}.log"), mode="w"
    )
    _log_handler.setLevel(logging.INFO)
    logging.root.addHandler(_log_handler)

    cnmf_obj = cNMF(output_dir=output_dir, name=run_name)

    prepare_kwargs = dict(
        counts_fn=mini_h5ad_path,
        components=TEST_K,
        n_iter=TEST_NUMITER,
        densify=False,
        tpm_fn=None,
        num_highvar_genes=TEST_NUMHVGENES,
        genes_file=None,
        beta_loss="frobenius",
        init="random",
        algo="halsvar",
        mode=mode,
        tol=1e-4,
        seed=TEST_SEED,
        use_gpu=True,
        alpha_usage=0.0,
        alpha_spectra=0.0,
        fp_precision="float",
    )
    if extra_prepare_kwargs:
        prepare_kwargs.update(extra_prepare_kwargs)

    cnmf_obj.prepare(**prepare_kwargs)
    cnmf_obj.factorize(skip_completed_runs=True)
    cnmf_obj.combine()
    run_cnmf_consensus(
        cnmf_obj=cnmf_obj,
        components=TEST_K,
        density_thresholds=TEST_SEL_THRESH,
    )
    compile_results(
        output_directory=output_dir,
        run_name=run_name,
        components=TEST_K,
        sel_threshs=TEST_SEL_THRESH,
        guide_names_key="guide_names",
        guide_targets_key="guide_targets",
        categorical_key="batch",
        guide_assignment_key="guide_assignment",
    )

    # Generate diagnostic plots
    generate_all_plots(
        run_dir=output_dir,
        run_name=run_name,
        K_list=TEST_K,
        sel_thresh_list=TEST_SEL_THRESH,
        categorical_key="batch",
    )

    logging.root.removeHandler(_log_handler)
    _log_handler.close()

    adata_dir = os.path.join(output_dir, run_name, "adata")
    return output_dir, adata_dir


def _validate_outputs(output_dir, adata_dir):
    """Check all expected output files exist and h5mu structure is correct."""
    inference_dir = os.path.join(output_dir, "Inference")
    missing = []

    for k in TEST_K:
        for thresh in TEST_SEL_THRESH:
            ts = str(thresh).replace(".", "_")

            expected_files = [
                # Core cNMF outputs
                f"Inference.gene_spectra_score.k_{k}.dt_{ts}.txt",
                f"Inference.gene_spectra_tpm.k_{k}.dt_{ts}.txt",
                f"Inference.usages.k_{k}.dt_{ts}.consensus.txt",
                f"Inference.spectra.k_{k}.dt_{ts}.consensus.txt",
                # Compiled outputs
                f"adata/cNMF_{k}_{ts}.h5mu",
                f"prog_data/NMF_{k}_{ts}.h5ad",
                f"loading/cNMF_scores_{k}_{thresh}.txt",
                f"loading/cNMF_loadings_{k}_{thresh}.txt",
                # Diagnostic plots
                f"diagnosis_plots/elbow_curves_k{k}_dt{ts}.pdf",
                f"diagnosis_plots/usage_heatmap_k{k}_dt{ts}.pdf",
                f"diagnosis_plots/loading_violins_k{k}_dt{ts}.pdf",
                f"diagnosis_plots/clustering_k{k}_dt{ts}.png",
            ]

            for f in expected_files:
                path = os.path.join(inference_dir, f)
                if not os.path.exists(path):
                    missing.append(f)

            # Validate h5mu structure
            h5mu_path = os.path.join(adata_dir, f"cNMF_{k}_{ts}.h5mu")
            if os.path.exists(h5mu_path):
                mdata = mu.read(h5mu_path)
                assert "rna" in mdata.mod, f"k={k}: Missing 'rna' modality"
                assert "cNMF" in mdata.mod, f"k={k}: Missing 'cNMF' modality"
                assert mdata["cNMF"].n_obs > 0, f"k={k}: No cells in cNMF modality"
                assert mdata["cNMF"].n_vars <= k, f"k={k}: Expected <= {k} programs, got {mdata['cNMF'].n_vars}"
                assert "loadings" in mdata["cNMF"].varm, f"k={k}: Missing varm['loadings']"
                assert "guide_names" in mdata["cNMF"].uns, f"k={k}: Missing uns['guide_names']"
                assert "guide_targets" in mdata["cNMF"].uns, f"k={k}: Missing uns['guide_targets']"

    assert not missing, f"Missing {len(missing)} output files:\n" + "\n".join(f"  {f}" for f in missing)
    print(f"\nAll output files verified ({len(TEST_K) * len(TEST_SEL_THRESH)} K×thresh combos)")


# ---------------------------------------------------------------------------
# Test: data validation (shared across modes)
# ---------------------------------------------------------------------------

class TestDataValid:
    def test_data_valid(self, mini_h5ad_path):
        adata = ad.read_h5ad(mini_h5ad_path)
        assert check_data_format(
            adata,
            categorical_key="batch",
            guide_assignment_key="guide_assignment",
        )


# ---------------------------------------------------------------------------
# Test: batch mode (full HALS, default)
# ---------------------------------------------------------------------------

class TestBatchMode:
    """End-to-end test for torch-cNMF in batch mode."""

    def test_batch_inference(self, mini_h5ad_path):
        output_dir, adata_dir = _run_full_inference(
            mini_h5ad_path,
            mode="batch",
            extra_prepare_kwargs=dict(
                batch_max_epoch=50,
                batch_hals_tol=0.05,
                batch_hals_max_iter=200,
            ),
        )
        _validate_outputs(output_dir, adata_dir)


# ---------------------------------------------------------------------------
# Test: minibatch mode
# ---------------------------------------------------------------------------

class TestMinibatchMode:
    """End-to-end test for torch-cNMF in minibatch mode."""

    def test_minibatch_inference(self, mini_h5ad_path):
        output_dir, adata_dir = _run_full_inference(
            mini_h5ad_path,
            mode="minibatch",
            extra_prepare_kwargs=dict(
                minibatch_max_epoch=5,
                minibatch_size=500,
                minibatch_max_iter=50,
                minibatch_usage_tol=0.05,
                minibatch_spectra_tol=0.05,
                minibatch_shuffle=True,
            ),
        )
        _validate_outputs(output_dir, adata_dir)


# ---------------------------------------------------------------------------
# Test: dataloader mode
# ---------------------------------------------------------------------------

class TestDataloaderMode:
    """End-to-end test for torch-cNMF in dataloader mode."""

    def test_dataloader_inference(self, mini_h5ad_path):
        output_dir, adata_dir = _run_full_inference(
            mini_h5ad_path,
            mode="dataloader",
            extra_prepare_kwargs=dict(
                minibatch_max_epoch=5,
                minibatch_size=500,
                minibatch_max_iter=50,
                minibatch_usage_tol=0.05,
                minibatch_spectra_tol=0.05,
                minibatch_shuffle=True,
            ),
        )
        _validate_outputs(output_dir, adata_dir)
