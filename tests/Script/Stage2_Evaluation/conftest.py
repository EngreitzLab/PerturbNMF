"""
Pytest configuration and shared fixtures for Evaluation pipeline tests.

All tests use real inference output from tests/output/torch-cNMF/batch/.
"""

import os
import sys
import pytest
import pandas as pd
import mudata

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


@pytest.fixture(scope="session")
def guide_annotation_path():
    """Path to real guide annotation TSV with 'targeting' column (TRUE/FALSE)."""
    path = os.path.join(PIPELINE_ROOT, "tests", "resources", "guide_annotation.tsv")
    if not os.path.isfile(path):
        pytest.skip(f"Guide annotation not found: {path}")
    return path


@pytest.fixture(scope="session")
def reference_targets_from_annotation(guide_annotation_path, test_mdata):
    """Extract reference target group names by matching annotation table guides to mdata."""
    df = pd.read_csv(guide_annotation_path, sep="\t", index_col=0)
    non_targeting_guides = df[df["targeting"] == False].index.values
    # Map guide names → target group names via mdata
    guide_metadata = pd.DataFrame({
        "guide": test_mdata["cNMF"].uns["guide_names"],
        "target": test_mdata["cNMF"].uns["guide_targets"],
    })
    matched = guide_metadata[guide_metadata["guide"].isin(non_targeting_guides)]
    ref_targets = matched["target"].unique().tolist()
    return ref_targets


# ---------------------------------------------------------------------------
# Calibration fixtures (use same real data as metrics tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def calibration_output_dir():
    """Output directory for calibration tests — saved inside Evaluation/."""
    out_dir = os.path.join(EVAL_OUTPUT_DIR, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def guide_annotation_df(guide_annotation_path):
    """Guide annotation DataFrame from the real annotation table."""
    return pd.read_csv(guide_annotation_path, sep="\t", index_col=0)
