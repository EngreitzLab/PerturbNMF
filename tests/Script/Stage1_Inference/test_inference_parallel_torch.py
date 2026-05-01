"""
End-to-end smoke test for parallel K inference mode (torch-cNMF).

In parallel mode, each K value runs as a separate cNMF job, producing
its own Inference_{K}/ directory. After all K jobs finish, rename_all_NMF
combines their spectra files into a single Inference_all/cnmf_tmp/ directory.

This test validates:
  1. Per-K factorization produces spectra files (GPU)
  2. rename_all_NMF correctly merges them into Inference_all/

Run via SLURM:
    sbatch tests/Script/Stage1_Inference/run_gpu_test_parallel.sh

Or directly on a GPU node:
    conda activate torch-nmf-dl
    python -m pytest tests/Script/Stage1_Inference/test_inference_parallel_torch.py -v

Output saved to: tests/output/torch-cNMF-parallel/
"""

import os
import sys
import pytest
import numpy as np

from .conftest import TEST_K, TEST_NUMITER, TEST_NUMHVGENES, TEST_SEL_THRESH, TEST_SEED, PERSISTENT_OUTPUT

from torch_cnmf import cNMF
from Stage1_Inference.src.run_cNMF import rename_all_NMF


# ---------- shared state ----------

class _State:
    per_k_dirs = {}


state = _State()

OUTPUT_DIR = str(PERSISTENT_OUTPUT / "torch-cNMF-parallel")


# ---------- tests ----------

class TestTorchCNMFParallel:
    """Test parallel K factorization and merge with torch-cNMF."""

    def test_01_prepare_and_factorize_per_k(self, mini_h5ad_path):
        """Run prepare + factorize separately for each K (simulates parallel SLURM array jobs)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for k in TEST_K:
            run_name = f"Inference_{k}"

            cnmf_obj = cNMF(output_dir=OUTPUT_DIR, name=run_name)
            cnmf_obj.prepare(
                counts_fn=mini_h5ad_path,
                components=[k],
                n_iter=TEST_NUMITER,
                densify=False,
                tpm_fn=None,
                seed=TEST_SEED,
                beta_loss="frobenius",
                num_highvar_genes=TEST_NUMHVGENES,
                genes_file=None,
                init="random",
                algo="halsvar",
                mode="batch",
                tol=1e-4,
                use_gpu=True,
                alpha_usage=0.0,
                alpha_spectra=0.0,
                fp_precision="float",
                batch_max_epoch=50,
                batch_hals_tol=0.05,
                batch_hals_max_iter=200,
            )
            cnmf_obj.factorize(skip_completed_runs=True)

            k_dir = os.path.join(OUTPUT_DIR, run_name)
            cnmf_tmp = os.path.join(k_dir, "cnmf_tmp")
            assert os.path.isdir(cnmf_tmp), f"cnmf_tmp not found for K={k}"

            for i in range(TEST_NUMITER):
                spectra = os.path.join(cnmf_tmp, f"{run_name}.spectra.k_{k}.iter_{i}.df.npz")
                assert os.path.exists(spectra), f"Missing spectra K={k} iter={i}: {spectra}"

            state.per_k_dirs[k] = k_dir
            print(f"  K={k}: factorized ({TEST_NUMITER} iters)")

    def test_02_combine_parallel_results(self):
        """Combine per-K spectra files into Inference_all/cnmf_tmp/."""
        assert len(state.per_k_dirs) == len(TEST_K), "All K values must be factorized first"

        combined_name = "Inference_all"
        combined_cnmf_tmp = os.path.join(OUTPUT_DIR, combined_name, "cnmf_tmp")

        rename_all_NMF(
            source_folder=os.path.join(OUTPUT_DIR, "Inference"),
            destination_folder=combined_cnmf_tmp,
            file_name_input="Inference",
            file_name_output="Inference_all",
            len=TEST_NUMITER,
            components=TEST_K,
        )

        # Verify all combined spectra exist
        missing = []
        for k in TEST_K:
            for i in range(TEST_NUMITER):
                f = os.path.join(combined_cnmf_tmp, f"{combined_name}.spectra.k_{k}.iter_{i}.df.npz")
                if not os.path.exists(f):
                    missing.append(f"Inference_all.spectra.k_{k}.iter_{i}.df.npz")

        assert not missing, f"Missing {len(missing)} combined spectra:\n" + "\n".join(f"  {f}" for f in missing)
        print(f"  Combined {len(TEST_K)} K values × {TEST_NUMITER} iters into {combined_cnmf_tmp}")
