"""
Pytest configuration and shared fixtures for Inference pipeline tests.

Mirrors the top-level tests/conftest.py constants and fixtures so inference
tests can be run directly from tests/Script/Stage1_Inference/.
"""

import os
import sys
import pytest
import logging
from pathlib import Path

# Add pipeline root to path
PIPELINE_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

MINI_H5AD = PIPELINE_ROOT / "tests" / "data" / "mini_ccperturb.h5ad"
PERSISTENT_OUTPUT = PIPELINE_ROOT / "tests" / "output"

# Shared test parameters
TEST_K = [5, 10, 15]
TEST_NUMITER = 5
TEST_NUMHVGENES = 2000
TEST_SEL_THRESH = [2.0]
TEST_SEED = 14


@pytest.fixture(scope="session")
def mini_h5ad_path():
    """Path to the mini subsampled h5ad."""
    if not MINI_H5AD.exists():
        pytest.skip(
            f"Mini dataset not found at {MINI_H5AD}. "
            "Run 'python tests/Script/create_mini_dataset.py' first."
        )
    return str(MINI_H5AD)


@pytest.fixture(scope="session")
def output_dir():
    """Output directory for sk-cNMF tests."""
    outdir = str(PERSISTENT_OUTPUT / "sk-cNMF")
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session", autouse=True)
def setup_logging(output_dir):
    """Set up log file inside Inference/logs/."""
    logs_dir = os.path.join(output_dir, "Inference", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, "test_run.log"), mode="w")
    fh.setLevel(logging.INFO)
    logging.root.addHandler(fh)


def check_data_format(
    adata,
    guide_names_key="guide_names",
    guide_targets_key="guide_targets",
    categorical_key="batch",
    guide_assignment_key="guide_assignment",
):
    """Validate that an AnnData has the keys cNMF inference relies on.

    Checks for `obs[categorical_key]`, `uns[{guide_names,guide_targets}]`,
    `obsm[{X_pca, X_umap, guide_assignment}]`. If `guide_assignment` is sparse,
    converts it to dense in-place. Returns True iff every required key is present.
    """
    is_valid = True

    if categorical_key not in adata.obs:
        print(f"WARNING: Not found in adata.obs['{categorical_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.obs['{categorical_key}']\n")

    if guide_names_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_names_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_names_key}']\n")

    if guide_targets_key not in adata.uns:
        print(f"WARNING: Not found in adata.uns['{guide_targets_key}']\n")
        is_valid = False
    else:
        print(f"Found adata.uns['{guide_targets_key}']\n")

    if "X_pca" not in adata.obsm:
        print("WARNING: Not found adata.obsm['X_pca']\n")
        is_valid = False
    else:
        print("Found adata.obsm['X_pca']\n")

    if "X_umap" not in adata.obsm:
        print("WARNING: Not found adata.obsm['X_umap']\n")
        is_valid = False
    else:
        print("Found adata.obsm['X_umap']\n")

    if guide_assignment_key not in adata.obsm:
        print(f"WARNING: Not found adata.obsm['{guide_assignment_key}']\n")
        is_valid = False
    else:
        guide_assignment = adata.obsm[guide_assignment_key]
        print(f"Found adata.obsm['{guide_assignment_key}']\n")
        try:
            import scipy.sparse as sp
            if sp.issparse(guide_assignment):
                print(f"WARNING: '{guide_assignment_key}' is sparse. Converting to dense array...")
                dense_array = guide_assignment.toarray()
                adata.obsm[guide_assignment_key] = dense_array
                print(f"'{guide_assignment_key}' converted to dense array (shape: {dense_array.shape})\n")
            else:
                print(f"'{guide_assignment_key}' is already dense (shape: {guide_assignment.shape})\n")
        except Exception as e:
            print(f"WARNING: Error checking '{guide_assignment_key}' sparsity: {e}\n")
            is_valid = False

    return is_valid
