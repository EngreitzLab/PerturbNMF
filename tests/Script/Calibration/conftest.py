"""
Pytest configuration and shared fixtures for Calibration pipeline tests.

Provides synthetic MuData with guide assignments for testing U-test
and CRT calibration without requiring real inference output.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu
from scipy import sparse
from pathlib import Path

PIPELINE_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

PERSISTENT_OUTPUT = PIPELINE_ROOT / "tests" / "output" / "torch-cNMF" / "dataloader" / "Calibration"


@pytest.fixture(scope="session")
def calibration_output_dir():
    """Output directory for calibration tests."""
    outdir = str(PERSISTENT_OUTPUT)
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def synthetic_mdata():
    """
    Create a synthetic MuData with guide assignments for calibration testing.

    Structure:
    - 200 cells, 50 genes, 5 programs
    - 3 conditions (d0, d1, d2)
    - 20 guides: 14 targeting (7 targets × 2 guides each), 6 non-targeting
    - Guide assignment: each cell assigned to exactly 1 guide
    """
    rng = np.random.default_rng(42)
    n_cells = 200
    n_genes = 50
    n_programs = 5
    n_targets = 7
    guides_per_target = 2
    n_nt_guides = 6
    n_guides = n_targets * guides_per_target + n_nt_guides  # 20

    # Gene expression (sparse)
    X_rna = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
    X_rna.data = np.abs(X_rna.data) * 10

    # Program usage scores (dense)
    X_prog = rng.exponential(1.0, size=(n_cells, n_programs))

    # Cell metadata
    conditions = rng.choice(["d0", "d1", "d2"], n_cells)
    cell_barcodes = [f"cell_{i:04d}" for i in range(n_cells)]

    # Guide names and targets
    target_names = [f"gene_{i}" for i in range(n_targets)]
    guide_names = []
    guide_targets = []
    for t in target_names:
        for g in range(guides_per_target):
            guide_names.append(f"{t}_guide_{g}")
            guide_targets.append(t)
    for i in range(n_nt_guides):
        guide_names.append(f"nt_guide_{i}")
        guide_targets.append("non-targeting")

    guide_names = np.array(guide_names)
    guide_targets = np.array(guide_targets)

    # Guide assignment matrix (each cell gets 1 guide)
    assignment = np.zeros((n_cells, n_guides), dtype=np.float64)
    cell_guides = rng.integers(0, n_guides, size=n_cells)
    for i, g in enumerate(cell_guides):
        assignment[i, g] = 1.0

    # Build RNA AnnData
    obs_df = pd.DataFrame(
        {"condition": conditions},
        index=cell_barcodes,
    )
    var_rna = pd.DataFrame(index=[f"rna_gene_{i}" for i in range(n_genes)])
    adata_rna = ad.AnnData(X=X_rna, obs=obs_df.copy(), var=var_rna)

    # Build cNMF AnnData
    program_names = [f"program_{i}" for i in range(n_programs)]
    var_prog = pd.DataFrame(index=program_names)
    adata_prog = ad.AnnData(
        X=X_prog,
        obs=obs_df.copy(),
        var=var_prog,
    )
    adata_prog.obsm["guide_assignment"] = assignment
    adata_prog.uns["guide_names"] = guide_names
    adata_prog.uns["guide_targets"] = guide_targets

    # Also add to RNA modality (U-test expects both)
    adata_rna.obsm["guide_assignment"] = assignment
    adata_rna.uns["guide_names"] = guide_names
    adata_rna.uns["guide_targets"] = guide_targets

    mdata = mu.MuData({"rna": adata_rna, "cNMF": adata_prog})
    return mdata


@pytest.fixture
def mdata_copy(synthetic_mdata):
    """Deep copy to avoid cross-test contamination."""
    prog = synthetic_mdata["cNMF"].copy()
    rna = synthetic_mdata["rna"].copy()
    return mu.MuData({"cNMF": prog, "rna": rna})


@pytest.fixture(scope="session")
def guide_annotation_df(synthetic_mdata):
    """Guide annotation DataFrame matching the synthetic MuData."""
    names = synthetic_mdata["cNMF"].uns["guide_names"]
    targets = synthetic_mdata["cNMF"].uns["guide_targets"]
    df = pd.DataFrame({
        "guide_names": names,
        "targeting": [t != "non-targeting" for t in targets],
        "type": ["targeting" if t != "non-targeting" else "non-targeting" for t in targets],
    }, index=names)
    return df
