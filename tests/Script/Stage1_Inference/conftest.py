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
