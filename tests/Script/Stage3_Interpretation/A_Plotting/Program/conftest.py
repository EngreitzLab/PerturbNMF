"""
Pytest configuration and shared fixtures for Program plotting tests.

Provides a real MuData fixture (from inference output) and synthetic
perturbation data for testing program-level plotting functions.
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
PLOT_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "Interpretation" / "Plotting" / "Program"


@pytest.fixture(scope="session")
def program_output_dir():
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
def synthetic_perturbation_tsv(program_output_dir):
    """Write a synthetic perturbation TSV and return its path.

    Matches the format expected by plot_program_log2FC and plot_program_volcano:
    target_name = program being queried, program_name = regulator gene.
    Uses program names '0'-'4' to match the real k=5 mdata.
    """
    rng = np.random.default_rng(42)
    rows = []
    for prog in range(5):
        for target in [f'gene_{i}' for i in range(10)]:
            rows.append({
                'target_name': str(prog),
                'program_name': target,
                'log2FC': rng.normal(0, 0.5),
                'adj_pval': rng.uniform(0.001, 0.2),
                'pval': rng.uniform(0.0005, 0.15),
                'stat': rng.normal(0, 2),
            })
    df = pd.DataFrame(rows)
    path = os.path.join(program_output_dir, 'synthetic_perturbation.txt')
    df.to_csv(path, sep='\t', index=False)
    return path


@pytest.fixture(scope="session")
def synthetic_perturbation_base(program_output_dir):
    """Write per-sample perturbation TSV files for heatmap tests.

    Returns the base path (without _SAMPLE.txt suffix).
    """
    rng = np.random.default_rng(42)
    base = os.path.join(program_output_dir, 'synthetic_perturb')
    for samp in ['D0', 'sample_D1']:
        rows = []
        for prog in range(5):
            for target in [f'gene_{i}' for i in range(8)]:
                rows.append({
                    'target_name': str(prog),
                    'program_name': target,
                    'log2FC': rng.normal(0, 0.5),
                    'adj_pval': rng.uniform(0.001, 0.2),
                    'pval': rng.uniform(0.0005, 0.15),
                    'stat': rng.normal(0, 2),
                })
        df = pd.DataFrame(rows)
        df.to_csv(f'{base}_{samp}.txt', sep='\t', index=False)
    return base


@pytest.fixture(scope="session")
def synthetic_program_correlation():
    """Synthetic program correlation matrix (5x5)."""
    rng = np.random.default_rng(42)
    n = 5
    mat = rng.uniform(-0.5, 1.0, (n, n))
    np.fill_diagonal(mat, 1.0)
    mat = (mat + mat.T) / 2
    names = [str(i) for i in range(n)]
    return pd.DataFrame(mat, index=names, columns=names)


@pytest.fixture(scope="session")
def synthetic_go_tsv(program_output_dir):
    """Write a synthetic GO enrichment TSV and return its path."""
    rows = []
    for prog in range(5):
        for i in range(8):
            rows.append({
                'program_name': prog,
                'Term': f'GO:00{prog}{i:02d} regulation of process {i}',
                'Adjusted P-value': np.random.uniform(0.001, 0.1),
            })
    df = pd.DataFrame(rows)
    df.index = df['program_name']
    df = df.drop(columns=['program_name'])
    path = os.path.join(program_output_dir, 'synthetic_GO.txt')
    df.to_csv(path, sep='\t')
    return path
