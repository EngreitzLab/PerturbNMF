"""
Pytest configuration and shared fixtures for Evaluation pipeline tests.

Provides:
- Metrics fixtures: --inference-path CLI option, real test data from inference output
- Calibration fixtures: synthetic MuData with guide assignments for U-test/CRT testing
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import anndata as ad
import mudata
from scipy import sparse

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
_TEST_OUTPUT = os.path.join(PIPELINE_ROOT, "tests", "output")

# Preferred inference output directories, in order of priority
_INFERENCE_CANDIDATES = [
    os.path.join(_TEST_OUTPUT, "torch-cNMF", "batch", "Inference"),
    os.path.join(_TEST_OUTPUT, "torch-cNMF", "dataloader", "Inference"),
    os.path.join(_TEST_OUTPUT, "torch-cNMF", "minibatch", "Inference"),
    os.path.join(_TEST_OUTPUT, "sk-cNMF", "Inference"),
]

def _find_inference_path():
    for path in _INFERENCE_CANDIDATES:
        if os.path.isdir(path):
            return path
    return _INFERENCE_CANDIDATES[0]  # default even if missing (will skip at fixture)

DEFAULT_INFERENCE_PATH = _find_inference_path()
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(DEFAULT_INFERENCE_PATH), "Evaluation")


def pytest_addoption(parser):
    parser.addoption(
        "--inference-path",
        action="store",
        default=DEFAULT_INFERENCE_PATH,
        help="Path to inference output directory (default: first available from torch-cNMF/batch, dataloader, minibatch, sk-cNMF)",
    )


@pytest.fixture(scope="session")
def inference_path(request):
    path = request.config.getoption("--inference-path")
    if not os.path.isdir(path):
        pytest.skip(f"Inference path not found: {path}")
    return path


TEST_K_VALUES = [5, 10, 15]


@pytest.fixture(scope="session")
def test_mdata(inference_path):
    """Load the smallest h5mu (k=5) once per test session."""
    h5mu_path = os.path.join(inference_path, "adata", "cNMF_5_2_0.h5mu")
    if not os.path.isfile(h5mu_path):
        pytest.skip(f"h5mu not found: {h5mu_path}")
    return mudata.read(h5mu_path)


@pytest.fixture
def mdata_copy(test_mdata):
    """Deep copy so tests don't mutate the session-scoped original."""
    prog = test_mdata["cNMF"].copy()
    rna = test_mdata["rna"].copy()
    return mudata.MuData({"cNMF": prog, "rna": rna})


@pytest.fixture(scope="session", params=TEST_K_VALUES)
def test_mdata_per_k(request, inference_path):
    """Load h5mu for each K value."""
    k = request.param
    h5mu_path = os.path.join(inference_path, "adata", f"cNMF_{k}_2_0.h5mu")
    if not os.path.isfile(h5mu_path):
        pytest.skip(f"h5mu not found: {h5mu_path}")
    mdata = mudata.read(h5mu_path)
    mdata.uns["test_k"] = k
    return mdata


@pytest.fixture
def mdata_copy_per_k(test_mdata_per_k):
    """Deep copy of per-K MuData so tests don't mutate the original."""
    prog = test_mdata_per_k["cNMF"].copy()
    rna = test_mdata_per_k["rna"].copy()
    md = mudata.MuData({"cNMF": prog, "rna": rna})
    md.uns["test_k"] = test_mdata_per_k.uns["test_k"]
    return md


@pytest.fixture(scope="session")
def eval_output_dir():
    """Fixed output directory for saving evaluation results."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    return EVAL_OUTPUT_DIR


@pytest.fixture
def eval_output_dir_per_k(mdata_copy_per_k):
    """Per-K output directory: Evaluation/{K}_2_0/"""
    k = mdata_copy_per_k.uns["test_k"]
    out_dir = os.path.join(EVAL_OUTPUT_DIR, f"{k}_2_0")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def cnmf_obj(inference_path):
    """Create a cNMF object pointing to the test inference output."""
    from torch_cnmf import cNMF
    run_dir = os.path.dirname(inference_path)  # parent of Inference/
    return cNMF(output_dir=run_dir, name='Inference')


@pytest.fixture(scope="session")
def X_norm(inference_path):
    """Load the normalized counts matrix from inference output."""
    import scanpy as sc
    norm_path = os.path.join(inference_path, "cnmf_tmp", "Inference.norm_counts.h5ad")
    if not os.path.isfile(norm_path):
        pytest.skip(f"Normalized counts not found: {norm_path}")
    adata = sc.read_h5ad(norm_path)
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X


# ---------------------------------------------------------------------------
# Calibration fixtures (synthetic data, no inference output required)
# ---------------------------------------------------------------------------

CALIBRATION_OUTPUT_DIR = os.path.join(_TEST_OUTPUT, "calibration")


@pytest.fixture(scope="session")
def calibration_output_dir():
    """Output directory for calibration tests."""
    os.makedirs(CALIBRATION_OUTPUT_DIR, exist_ok=True)
    return CALIBRATION_OUTPUT_DIR


@pytest.fixture(scope="session")
def synthetic_mdata():
    """
    Create a synthetic MuData with guide assignments for calibration testing.

    Structure:
    - 200 cells, 50 genes, 5 programs
    - 3 conditions (d0, d1, d2)
    - 20 guides: 14 targeting (7 targets x 2 guides each), 6 non-targeting
    - Guide assignment: each cell assigned to exactly 1 guide
    """
    rng = np.random.default_rng(42)
    n_cells = 200
    n_genes = 50
    n_programs = 5
    n_targets = 7
    guides_per_target = 2
    n_nt_guides = 6
    n_guides = n_targets * guides_per_target + n_nt_guides

    X_rna = sparse.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)
    X_rna.data = np.abs(X_rna.data) * 10
    X_prog = rng.exponential(1.0, size=(n_cells, n_programs))

    conditions = rng.choice(["d0", "d1", "d2"], n_cells)
    cell_barcodes = [f"cell_{i:04d}" for i in range(n_cells)]

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

    assignment = np.zeros((n_cells, n_guides), dtype=np.float64)
    cell_guides = rng.integers(0, n_guides, size=n_cells)
    for i, g in enumerate(cell_guides):
        assignment[i, g] = 1.0

    obs_df = pd.DataFrame({"condition": conditions}, index=cell_barcodes)
    var_rna = pd.DataFrame(index=[f"rna_gene_{i}" for i in range(n_genes)])
    adata_rna = ad.AnnData(X=X_rna, obs=obs_df.copy(), var=var_rna)

    program_names = [f"program_{i}" for i in range(n_programs)]
    var_prog = pd.DataFrame(index=program_names)
    adata_prog = ad.AnnData(X=X_prog, obs=obs_df.copy(), var=var_prog)
    adata_prog.obsm["guide_assignment"] = assignment
    adata_prog.uns["guide_names"] = guide_names
    adata_prog.uns["guide_targets"] = guide_targets

    adata_rna.obsm["guide_assignment"] = assignment
    adata_rna.uns["guide_names"] = guide_names
    adata_rna.uns["guide_targets"] = guide_targets

    return mudata.MuData({"rna": adata_rna, "cNMF": adata_prog})


@pytest.fixture
def calibration_mdata_copy(synthetic_mdata):
    """Deep copy of synthetic MuData to avoid cross-test contamination."""
    prog = synthetic_mdata["cNMF"].copy()
    rna = synthetic_mdata["rna"].copy()
    return mudata.MuData({"cNMF": prog, "rna": rna})


@pytest.fixture(scope="session")
def guide_annotation_df(synthetic_mdata):
    """Guide annotation DataFrame matching the synthetic MuData."""
    names = synthetic_mdata["cNMF"].uns["guide_names"]
    targets = synthetic_mdata["cNMF"].uns["guide_targets"]
    return pd.DataFrame({
        "guide_names": names,
        "targeting": [t != "non-targeting" for t in targets],
        "type": ["targeting" if t != "non-targeting" else "non-targeting" for t in targets],
    }, index=names)
