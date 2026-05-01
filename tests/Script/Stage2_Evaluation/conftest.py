"""
Pytest configuration and shared fixtures for Evaluation pipeline tests.

Provides --inference-path CLI option and session-scoped fixtures that load
real test data from the inference output directory.
"""

import os
import sys
import pytest
import mudata

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PIPELINE_ROOT, "src"))
DEFAULT_INFERENCE_PATH = os.path.join(
    PIPELINE_ROOT, "tests", "output", "torch-cNMF", "dataloader", "Inference"
)
EVAL_OUTPUT_DIR = os.path.join(
    PIPELINE_ROOT, "tests", "output", "torch-cNMF", "dataloader", "Evaluation"
)


def pytest_addoption(parser):
    parser.addoption(
        "--inference-path",
        action="store",
        default=DEFAULT_INFERENCE_PATH,
        help="Path to inference output directory (default: tests/output/torch-cNMF/dataloader/Inference/)",
    )


@pytest.fixture(scope="session")
def inference_path(request):
    path = request.config.getoption("--inference-path")
    if not os.path.isdir(path):
        pytest.skip(f"Inference path not found: {path}")
    return path


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


@pytest.fixture(scope="session")
def spectra_score_path(inference_path):
    path = os.path.join(inference_path, "Inference.gene_spectra_score.k_5.dt_2_0.txt")
    if not os.path.isfile(path):
        pytest.skip(f"Spectra score file not found: {path}")
    return path


@pytest.fixture(scope="session")
def eval_output_dir():
    """Fixed output directory for saving evaluation results."""
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    return EVAL_OUTPUT_DIR
