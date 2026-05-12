"""
Pytest configuration and shared fixtures for gene HTML export tests.

Uses REAL inference + evaluation output from
``tests/output/torch-cNMF/batch/``. No synthetic data.

Provides:
  - real k=5 MuData
  - perturb_path_base pointing at the real per-sample U-test result files
  - real gene-loading correlation matrix (subsetted to perturbed-gene targets
    for tractability)
  - real per-sample waterfall correlation matrices (computed from perturb TSVs)
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

DEFAULT_INFERENCE_PATH = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Inference"
EVAL_5_2_0 = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Evaluation" / "5_2_0"
HTML_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch" / "Interpretation" / "Plotting" / "gene_html"


SAMPLES = ["D0", "D1", "D2", "D3"]


@pytest.fixture(scope="session")
def html_share_path():
    """Real share folder where gene_{SYM}/ subtrees will be written.

    Mirrors the layout the Slurm pipeline produces, e.g.:
    tests/output/torch-cNMF/batch/Plots/gene_html_test/html_share/
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
def target_gene(test_mdata):
    """Pick a perturbed gene present in expression matrix."""
    guide_targets = list(np.unique(test_mdata["cNMF"].uns["guide_targets"]))
    symbols = set(test_mdata["rna"].var["symbol"].astype(str).tolist())
    candidates = [g for g in guide_targets if g in symbols and g != "non-targeting"]
    if not candidates:
        pytest.skip("No perturbed gene present in expression matrix")
    return candidates[0]


@pytest.fixture(scope="session")
def perturb_path_base():
    """Real per-sample U-test perturbation TSV base path.

    Same files as the program HTML — gene HTML interprets them differently
    (target_name == perturbed gene, program_name == program affected).
    """
    base = EVAL_5_2_0 / "5_perturbation_association_results"
    if not Path(f"{base}_{SAMPLES[0]}.txt").exists():
        pytest.skip(f"Real perturbation TSVs not found at {base}_*.txt")
    return str(base)


@pytest.fixture(scope="session")
def available_samples(perturb_path_base):
    return [s for s in SAMPLES if os.path.isfile(f"{perturb_path_base}_{s}.txt")]


@pytest.fixture(scope="session")
def gene_loading_corr_matrix(test_mdata, target_gene):
    """Gene x gene correlation matrix from real cNMF loadings, subsetted to
    perturbed-gene targets (a few hundred), to keep the matrix tractable.

    The full 17k x 17k matrix would be ~2 GB; subsetting keeps loadings real
    while bounding test memory/runtime.
    """
    loadings = test_mdata["cNMF"].varm["loadings"]  # (programs x genes)
    symbols = test_mdata["rna"].var["symbol"].astype(str).values
    guide_targets = set(np.unique(test_mdata["cNMF"].uns["guide_targets"]).tolist())
    gene_mask = np.array([s in guide_targets for s in symbols])
    sub = loadings[:, gene_mask]
    sub_symbols = symbols[gene_mask]
    df = pd.DataFrame(data=sub, columns=sub_symbols)
    # Drop duplicate column names that arise when multiple Ensembl IDs share a symbol
    df = df.loc[:, ~df.columns.duplicated()]
    corr = df.corr().fillna(0)
    assert target_gene in corr.columns, "target_gene missing from subsetted corr matrix"
    return corr


@pytest.fixture(scope="session")
def perturb_corr_by_sample(perturb_path_base, available_samples):
    """Real per-sample gene x gene waterfall correlation matrices."""
    from Stage3_Interpretation.A_Plotting.src.Perturbed_gene_QC_plots import compute_gene_waterfall_cor
    out = {}
    for samp in available_samples:
        path = f"{perturb_path_base}_{samp}.txt"
        out[samp] = compute_gene_waterfall_cor(path)
    if not out:
        pytest.skip("No per-sample waterfall correlations could be built")
    return out
