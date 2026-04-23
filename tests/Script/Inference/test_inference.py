"""
End-to-end smoke tests for sk-cNMF inference pipeline.

Run with:
    conda activate sk-cNMF
    python -m pytest tests/Script/Inference/test_inference.py -v

Tests are ordered sequentially — each step depends on the previous.
"""

import os
import sys
import pytest
import anndata as ad
import mudata as mu
import cnmf

from .conftest import TEST_K, TEST_NUMITER, TEST_NUMHVGENES, TEST_SEL_THRESH, TEST_SEED

from Inference.src.format_checking import check_data_format, check_mdata_format
from Inference.src.run_cNMF import run_cnmf_consensus, compile_results
from Inference.src.plot_diagnostics import generate_all_plots


# ---------- shared state across ordered tests ----------

class _State:
    """Mutable container shared across tests in this module."""
    cnmf_obj = None
    run_dir = None


state = _State()


# ---------- tests (run in order) ----------

class TestSkCNMFInference:
    """Ordered end-to-end test for sk-cNMF inference."""

    def test_01_data_valid(self, mini_h5ad_path):
        """Input h5ad passes format validation."""
        adata = ad.read_h5ad(mini_h5ad_path)
        assert adata.shape[0] > 0
        is_valid = check_data_format(
            adata,
            categorical_key="batch",
            guide_assignment_key="guide_assignment",
        )
        assert is_valid

    def test_02_prepare(self, mini_h5ad_path, output_dir):
        """cNMF prepare step creates working files."""
        state.run_dir = output_dir
        run_name = "Inference"

        state.cnmf_obj = cnmf.cNMF(output_dir=output_dir, name=run_name)
        state.cnmf_obj.prepare(
            counts_fn=mini_h5ad_path,
            components=TEST_K,
            n_iter=TEST_NUMITER,
            densify=False,
            tpm_fn=None,
            seed=TEST_SEED,
            beta_loss="frobenius",
            num_highvar_genes=TEST_NUMHVGENES,
            genes_file=None,
            alpha_usage=0.0,
            alpha_spectra=0.0,
            init="random",
            max_NMF_iter=500,
            algo="mu",
            tol=1e4,
        )

        # Check that cnmf_tmp was created with expected files
        cnmf_tmp = os.path.join(output_dir, run_name, "cnmf_tmp")
        assert os.path.isdir(cnmf_tmp), f"cnmf_tmp dir not found at {cnmf_tmp}"

    def test_03_factorize(self):
        """Factorize produces spectra iteration files."""
        assert state.cnmf_obj is not None, "prepare must run first"
        state.cnmf_obj.factorize(total_workers=1, skip_completed_runs=True)

        cnmf_tmp = os.path.join(state.run_dir, "Inference", "cnmf_tmp")
        # Check at least one spectra iter file exists for each K
        for k in TEST_K:
            pattern = f"Inference.spectra.k_{k}.iter_0.df.npz"
            path = os.path.join(cnmf_tmp, pattern)
            assert os.path.exists(path), f"Missing spectra file: {path}"

    def test_04_combine(self):
        """Combine merges iteration results and generate k-selection plot."""
        assert state.cnmf_obj is not None
        state.cnmf_obj.combine()
        state.cnmf_obj.k_selection_plot(close_fig=True)

        k_sel = os.path.join(state.run_dir, "Inference", "Inference.k_selection.png")
        assert os.path.exists(k_sel), f"Missing k-selection plot: {k_sel}"

    def test_05_consensus(self):
        """Consensus runs for all K and density thresholds."""
        assert state.cnmf_obj is not None
        run_cnmf_consensus(
            cnmf_obj=state.cnmf_obj,
            components=TEST_K,
            density_thresholds=TEST_SEL_THRESH,
        )

        # Check that usage files were produced
        run_dir = os.path.join(state.run_dir, "Inference")
        for k in TEST_K:
            for thresh in TEST_SEL_THRESH:
                thresh_str = str(thresh).replace(".", "_")
                usage_file = os.path.join(
                    run_dir, f"Inference.usages.k_{k}.dt_{thresh_str}.consensus.txt"
                )
                assert os.path.exists(usage_file), f"Missing usage file: {usage_file}"

    def test_06_compile_results(self):
        """compile_results produces h5mu files."""
        compile_results(
            output_directory=state.run_dir,
            run_name="Inference",
            components=TEST_K,
            sel_threshs=TEST_SEL_THRESH,
            guide_names_key="guide_names",
            guide_targets_key="guide_targets",
            categorical_key="batch",
            guide_assignment_key="guide_assignment",
        )

        adata_dir = os.path.join(state.run_dir, "Inference", "adata")
        assert os.path.isdir(adata_dir), f"adata dir not found: {adata_dir}"

        for k in TEST_K:
            for thresh in TEST_SEL_THRESH:
                thresh_str = str(thresh).replace(".", "_")
                h5mu_path = os.path.join(adata_dir, f"cNMF_{k}_{thresh_str}.h5mu")
                assert os.path.exists(h5mu_path), f"Missing h5mu: {h5mu_path}"

    def test_07_output_structure(self):
        """Output h5mu has correct modalities and shapes."""
        adata_dir = os.path.join(state.run_dir, "Inference", "adata")

        for k in TEST_K:
            for thresh in TEST_SEL_THRESH:
                thresh_str = str(thresh).replace(".", "_")
                h5mu_path = os.path.join(adata_dir, f"cNMF_{k}_{thresh_str}.h5mu")
                mdata = mu.read(h5mu_path)

                # Check modalities exist
                assert "rna" in mdata.mod, "Missing 'rna' modality"
                assert "cNMF" in mdata.mod, "Missing 'cNMF' modality"

                # Check cNMF shape
                n_cells = mdata["cNMF"].n_obs
                n_programs = mdata["cNMF"].n_vars
                assert n_cells > 0, "No cells in cNMF modality"
                assert n_programs <= k, f"Expected <= {k} programs, got {n_programs}"

                # Check loadings exist
                assert "loadings" in mdata["cNMF"].varm, "Missing varm['loadings']"

                # Check guide metadata propagated
                assert "guide_names" in mdata["cNMF"].uns
                assert "guide_targets" in mdata["cNMF"].uns

    def test_08_diagnostic_plots(self):
        """Generate diagnostic plots from inference outputs."""
        assert state.run_dir is not None
        plots_dir = generate_all_plots(
            run_dir=state.run_dir,
            run_name="Inference",
            K_list=TEST_K,
            sel_thresh_list=TEST_SEL_THRESH,
            categorical_key="batch",
        )
        assert os.path.isdir(plots_dir), f"Plots dir not created: {plots_dir}"

        # Check that key plots were generated
        for k in TEST_K:
            for thresh in TEST_SEL_THRESH:
                thresh_str = str(thresh).replace(".", "_")
                elbow = os.path.join(plots_dir, f"elbow_curves_k{k}_dt{thresh_str}.pdf")
                assert os.path.exists(elbow), f"Missing elbow plot: {elbow}"
                usage = os.path.join(plots_dir, f"usage_heatmap_k{k}_dt{thresh_str}.pdf")
                assert os.path.exists(usage), f"Missing usage heatmap: {usage}"
                violin = os.path.join(plots_dir, f"loading_violins_k{k}_dt{thresh_str}.pdf")
                assert os.path.exists(violin), f"Missing violin plot: {violin}"

        print(f"\nDiagnostic plots saved to: {plots_dir}")

    def test_09_assert_all_outputs(self):
        """Verify every expected output file exists."""
        assert state.run_dir is not None
        inference_dir = os.path.join(state.run_dir, "Inference")
        missing = []

        for k in TEST_K:
            for thresh in TEST_SEL_THRESH:
                ts = str(thresh).replace(".", "_")

                expected = [
                    # Core cNMF outputs
                    f"Inference.gene_spectra_score.k_{k}.dt_{ts}.txt",
                    f"Inference.gene_spectra_tpm.k_{k}.dt_{ts}.txt",
                    f"Inference.usages.k_{k}.dt_{ts}.consensus.txt",
                    f"Inference.spectra.k_{k}.dt_{ts}.consensus.txt",
                    f"Inference.clustering.k_{k}.dt_{ts}.png",
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

                for f in expected:
                    path = os.path.join(inference_dir, f)
                    if not os.path.exists(path):
                        missing.append(f)

        # K-selection plot
        if not os.path.exists(os.path.join(inference_dir, "Inference.k_selection.png")):
            missing.append("Inference.k_selection.png")
        if not os.path.exists(os.path.join(inference_dir, "diagnosis_plots", "k_selection.png")):
            missing.append("diagnosis_plots/k_selection.png")

        # Logs
        if not os.path.exists(os.path.join(inference_dir, "logs", "test_run.log")):
            missing.append("logs/test_run.log")

        assert not missing, f"Missing {len(missing)} output files:\n" + "\n".join(f"  {f}" for f in missing)
        print(f"\nAll output files verified ({len(TEST_K) * len(TEST_SEL_THRESH)} K×thresh combos)")
