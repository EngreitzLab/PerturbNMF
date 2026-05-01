"""
Pytest configuration and shared fixtures for Gene (Perturbed_gene_QC) plotting tests.

Provides a real MuData fixture (from inference output) and synthetic
perturbation data for testing gene-level plotting functions.
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

DEFAULT_INFERENCE_PATH = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "dataloader" / "Inference"
PLOT_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "Interpretation" / "Plotting" / "Gene"


@pytest.fixture(scope="session")
def gene_output_dir():
    outdir = str(PLOT_OUTPUT)
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def test_mdata():
    """Load real h5mu (k=5) from inference output."""
    import mudata
    h5mu_path = DEFAULT_INFERENCE_PATH / "adata" / "cNMF_5_2_0.h5mu"
    if not h5mu_path.exists():
        pytest.skip(f"h5mu not found: {h5mu_path}")
    return mudata.read(str(h5mu_path))


@pytest.fixture(scope="session")
def synthetic_gene_perturbation_tsv(gene_output_dir):
    """Synthetic perturbation TSV for gene-level plots.

    In Gene QC plots, the perspective is:
    - target_name = the perturbed gene
    - program_name = the program affected (as string)
    """
    rng = np.random.default_rng(42)
    rows = []
    for target in [f'gene_{i}' for i in range(10)]:
        for prog in range(5):
            rows.append({
                'target_name': target,
                'program_name': str(prog),
                'log2FC': rng.normal(0, 0.5),
                'adj_pval': rng.uniform(0.001, 0.2),
                'pval': rng.uniform(0.0005, 0.15),
                'stat': rng.normal(0, 2),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(gene_output_dir, 'synthetic_gene_perturbation.txt')
    df.to_csv(path, sep='\t', index=True)
    return path


@pytest.fixture(scope="session")
def synthetic_gene_correlation():
    """Synthetic gene-gene correlation matrix for waterfall tests."""
    rng = np.random.default_rng(42)
    genes = [f'gene_{i}' for i in range(10)]
    mat = rng.uniform(-0.5, 1.0, (10, 10))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, np.nan)
    return pd.DataFrame(mat, index=genes, columns=genes)
