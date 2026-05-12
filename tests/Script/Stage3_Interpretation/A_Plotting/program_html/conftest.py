"""
Pytest configuration and shared fixtures for program HTML export tests.

Uses REAL inference + evaluation output from
``tests/output/torch-cNMF/batch/`` (the default test inference run).
No synthetic perturbation TSVs are written.

Provides:
  - real k=5 MuData
  - perturb_path_base pointing at the real per-sample U-test result files
  - real program correlation matrix (computed from cNMF loadings)
  - real per-sample waterfall correlation matrices (computed from perturb TSVs)
  - real GO enrichment TSV (5_GO_term_enrichment.txt)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

# Default to the batch run (matches the rest of the test suite).
DEFAULT_INFERENCE_PATH = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Inference"
EVAL_5_2_0 = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Evaluation" / "5_2_0"
HTML_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Interpretation" / "Plotting" / "program_html"


# Sample names match the per-sample perturb TSVs produced by the U-test pipeline.
SAMPLES = ["D0", "D1", "D2", "D3"]


@pytest.fixture(scope="session")
def html_share_path():
    """Real share folder where program_{N}/ subtrees will be written.

    Mirrors the layout the Slurm pipeline produces, e.g.:
    tests/output/torch-cNMF/batch/Plots/program_html_test/html_share/
    """
    outdir = HTML_OUTPUT / "html_share"
    outdir.mkdir(parents=True, exist_ok=True)
    return str(outdir)


@pytest.fixture(scope="session")
def test_mdata():
    """Real k=5 MuData from batch inference output."""
    import mudata
    h5mu_path = DEFAULT_INFERENCE_PATH / "adata" / "cNMF_5_2_0.h5mu"
    if not h5mu_path.exists():
        pytest.skip(f"h5mu not found: {h5mu_path}")
    return mudata.read(str(h5mu_path))


@pytest.fixture(scope="session")
def perturb_path_base():
    """Base path for real per-sample U-test perturbation TSVs.

    The export functions read ``<base>_<sample>.txt``; existing files are
    ``5_perturbation_association_results_{D0..D3}.txt``.
    """
    base = EVAL_5_2_0 / "5_perturbation_association_results"
    # sanity: at least one sample file must exist
    if not (Path(f"{base}_{SAMPLES[0]}.txt")).exists():
        pytest.skip(f"Real perturbation TSVs not found at {base}_*.txt")
    return str(base)


@pytest.fixture(scope="session")
def go_path():
    """Real GO term enrichment TSV from Evaluation/5_2_0/."""
    p = EVAL_5_2_0 / "5_GO_term_enrichment.txt"
    if not p.exists():
        pytest.skip(f"Real GO enrichment file not found: {p}")
    return str(p)


@pytest.fixture(scope="session")
def perturbed_gene_found(test_mdata):
    """Perturbed targets that are also present in the expression matrix."""
    perturbed = list(np.unique(test_mdata["cNMF"].uns["guide_targets"]))
    symbols = set(test_mdata["rna"].var["symbol"].astype(str).tolist())
    return sorted(set(perturbed) & symbols - {"non-targeting"})


@pytest.fixture(scope="session")
def program_correlation(test_mdata):
    """Real program x program correlation matrix from cNMF loadings."""
    from Stage3_Interpretation.A_Plotting.src.Program_QC_plots import compute_program_correlation_matrix
    return compute_program_correlation_matrix(test_mdata)


@pytest.fixture(scope="session")
def waterfall_correlation(perturb_path_base):
    """Real per-sample waterfall correlation matrices computed from perturb TSVs."""
    from Stage3_Interpretation.A_Plotting.src.Program_QC_plots import compute_program_waterfall_cor
    out = {}
    for samp in SAMPLES:
        path = f"{perturb_path_base}_{samp}.txt"
        if not os.path.isfile(path):
            continue
        out[samp] = compute_program_waterfall_cor(path)
    if not out:
        pytest.skip("No per-sample waterfall correlations could be built")
    return out


@pytest.fixture(scope="session")
def available_samples(waterfall_correlation):
    """Subset of SAMPLES that actually have real perturb TSVs on disk."""
    return list(waterfall_correlation.keys())
