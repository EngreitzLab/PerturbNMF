"""
Pytest configuration and shared fixtures for K-Selection plotting tests.

Uses real torch-cNMF batch output from tests/output/torch-cNMF/batch/.
Falls back to synthetic data for helper/logic tests that don't need real files.
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

BATCH_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "batch"
INFERENCE_DIR = BATCH_OUTPUT / "Inference"
EVAL_DIR = BATCH_OUTPUT / "Evaluation"
PLOT_OUTPUT = BATCH_OUTPUT / "Interpretation" / "Plotting" / "K-Selection"

COMPONENTS = [5, 10, 15]
SEL_THRESH = 2.0
SAMPLES = ["D0", "D1", "D2", "D3"]


@pytest.fixture(scope="session")
def kselection_output_dir():
    outdir = str(PLOT_OUTPUT)
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def inference_dir():
    if not INFERENCE_DIR.exists():
        pytest.skip(f"Inference output not found: {INFERENCE_DIR}")
    return str(INFERENCE_DIR)


@pytest.fixture(scope="session")
def eval_dir():
    if not EVAL_DIR.exists():
        pytest.skip(f"Evaluation output not found: {EVAL_DIR}")
    return str(EVAL_DIR)


# ---------------------------------------------------------------------------
# Real data fixtures — loaded once per session from batch output
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_stability_stats(inference_dir):
    """Load stability/error stats from real torch-cNMF batch output."""
    from Stage3_Interpretation.A_Plotting.src.k_selection_plots import load_stablity_error_data
    output_directory = str(BATCH_OUTPUT)
    return load_stablity_error_data(
        output_directory=output_directory,
        run_name="Inference",
        components=COMPONENTS,
    )


@pytest.fixture(scope="session")
def real_enrichment_df(eval_dir):
    """Load enrichment data (GO, genesets, traits) from real evaluation output."""
    # Verify all required enrichment files exist before loading
    sel_str = str(SEL_THRESH).replace('.', '_')
    for k in COMPONENTS:
        k_folder = EVAL_DIR / f"{k}_{sel_str}"
        for pattern in ['{k}_GO_term_enrichment.txt', '{k}_geneset_enrichment.txt', '{k}_trait_enrichment.txt']:
            fpath = k_folder / pattern.format(k=k)
            if not fpath.exists():
                pytest.skip(f"Missing enrichment file: {fpath}")
    from Stage3_Interpretation.A_Plotting.src.k_selection_plots import load_enrichment_data
    return load_enrichment_data(
        folder=str(EVAL_DIR),
        components=COMPONENTS,
        sel_thresh=SEL_THRESH,
        pval=0.05,
    )


@pytest.fixture(scope="session")
def real_perturbation_df(eval_dir):
    """Load perturbation association results from real evaluation output."""
    sel_str = str(SEL_THRESH).replace('.', '_')
    for k in COMPONENTS:
        k_folder = EVAL_DIR / f"{k}_{sel_str}"
        for samp in SAMPLES:
            fpath = k_folder / f"{k}_perturbation_association_results_{samp}.txt"
            if not fpath.exists():
                pytest.skip(f"Missing perturbation file: {fpath}")
    from Stage3_Interpretation.A_Plotting.src.k_selection_plots import load_perturbation_data
    return load_perturbation_data(
        folder=str(EVAL_DIR),
        components=COMPONENTS,
        sel_thresh=SEL_THRESH,
        samples=SAMPLES,
    )


@pytest.fixture(scope="session")
def real_explained_var(eval_dir):
    """Load explained variance data from real evaluation output."""
    sel_str = str(SEL_THRESH).replace('.', '_')
    for k in COMPONENTS:
        fpath = EVAL_DIR / f"{k}_{sel_str}" / f"{k}_Explained_Variance_Summary.txt"
        if not fpath.exists():
            pytest.skip(f"Missing variance file: {fpath}")
    from Stage3_Interpretation.A_Plotting.src.k_selection_plots import load_explained_variance_data
    return load_explained_variance_data(
        folder=str(EVAL_DIR),
        components=COMPONENTS,
        sel_thresh=SEL_THRESH,
    )


@pytest.fixture(scope="session")
def real_spectra_matrices(inference_dir):
    """Load two gene spectra score matrices from real inference output."""
    k5 = INFERENCE_DIR / "Inference.gene_spectra_score.k_5.dt_2_0.txt"
    k10 = INFERENCE_DIR / "Inference.gene_spectra_score.k_10.dt_2_0.txt"
    if not k5.exists() or not k10.exists():
        pytest.skip("Spectra score files not found")
    m1 = pd.read_csv(str(k5), sep='\t', index_col=0)
    m2 = pd.read_csv(str(k10), sep='\t', index_col=0)
    return m1, m2


# ---------------------------------------------------------------------------
# Synthetic fixtures — for pure logic tests that don't need real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_stability_stats():
    """Synthetic DataFrame mimicking load_stablity_error_data output."""
    ks = [5, 10, 15, 20, 30]
    return pd.DataFrame({
        'k': ks,
        'local_density_threshold': [2.0] * 5,
        'silhouette': [0.8, 0.75, 0.7, 0.65, 0.6],
        'prediction_error': [1e6, 8e5, 6e5, 5e5, 4e5],
    })


@pytest.fixture(scope="session")
def synthetic_count_df():
    """Synthetic enrichment count DataFrame."""
    ks = [5, 10, 15, 20, 30]
    return pd.DataFrame({
        'go_terms': [3, 5, 8, 12, 10],
        'genesets': [2, 4, 6, 9, 7],
        'traits': [1, 2, 3, 5, 4],
    }, index=ks)


@pytest.fixture(scope="session")
def synthetic_test_stats_df():
    """Synthetic perturbation stats DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    for k in [5, 10, 15, 20, 30]:
        for samp in ['D0', 'sample_D1', 'sample_D2']:
            for target in [f'gene_{i}' for i in range(5)]:
                rows.append({
                    'K': k, 'sample': samp, 'target_name': target,
                    'adj_pval': rng.uniform(0.0001, 0.1),
                    'log2FC': rng.normal(0, 0.5),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_explained_var():
    """Synthetic explained variance dict."""
    return {5: 0.15, 10: 0.25, 15: 0.35, 20: 0.42, 30: 0.50}


@pytest.fixture(scope="session")
def synthetic_spectra_matrices():
    """Two synthetic programs x genes DataFrames for logic tests."""
    rng = np.random.default_rng(42)
    genes = [f'gene_{i}' for i in range(100)]
    programs = [f'prog_{i}' for i in range(5)]
    m1 = pd.DataFrame(rng.random((5, 100)), index=programs, columns=genes)
    m2 = pd.DataFrame(rng.random((5, 100)), index=programs, columns=genes)
    return m1, m2
