"""
Pytest configuration and shared fixtures for K-Selection plotting tests.

Provides synthetic data fixtures that mimic the output of load_stablity_error_data,
load_enrichment_data, load_perturbation_data, and load_explained_variance_data.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

PLOT_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "Interpretation" / "Plotting" / "KSelection"


@pytest.fixture(scope="session")
def kselection_output_dir():
    outdir = str(PLOT_OUTPUT)
    os.makedirs(outdir, exist_ok=True)
    return outdir


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
    """Synthetic enrichment count DataFrame mimicking load_enrichment_data output."""
    ks = [5, 10, 15, 20, 30]
    return pd.DataFrame({
        'go_terms': [3, 5, 8, 12, 10],
        'genesets': [2, 4, 6, 9, 7],
        'traits': [1, 2, 3, 5, 4],
    }, index=ks)


@pytest.fixture(scope="session")
def synthetic_test_stats_df():
    """Synthetic perturbation stats DataFrame mimicking load_perturbation_data output."""
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
    """Synthetic explained variance dict mimicking load_explained_variance_data output."""
    return {5: 0.15, 10: 0.25, 15: 0.35, 20: 0.42, 30: 0.50}


@pytest.fixture(scope="session")
def synthetic_spectra_matrices():
    """Two synthetic programs x genes DataFrames for k_quality_plots tests."""
    rng = np.random.default_rng(42)
    genes = [f'gene_{i}' for i in range(100)]
    programs = [f'prog_{i}' for i in range(5)]
    m1 = pd.DataFrame(rng.random((5, 100)), index=programs, columns=genes)
    m2 = pd.DataFrame(rng.random((5, 100)), index=programs, columns=genes)
    return m1, m2
