"""
Unit tests for the CRT (Conditional Randomization Test) calibration.

Requires the programDE conda environment (sceptre dependency).

Usage:
    eval "$(conda shell.bash hook)" && conda activate programDE
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage2_Evaluation/test_crt.py -v
"""

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# CRT import (skip entire module if sceptre unavailable)
# ---------------------------------------------------------------------------

CRT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src",
                       "Stage2_Evaluation", "B_Calibration", "Slurm_version", "CRT")

try:
    sys.path.insert(0, CRT_DIR)
    from CRT import reformat_data_for_CRT  # noqa: F401 — needs sceptre (programDE env)
    _HAS_CRT = True
except ImportError:
    _HAS_CRT = False


@pytest.mark.skipif(not _HAS_CRT, reason="CRT requires programDE env (sceptre not available)")
class TestCRTReformat:
    """Test the reformat_data_for_CRT function from the CRT calibration script."""

    def test_reformat_basic_structure(self, test_mdata):
        """reformat_data_for_CRT should produce expected keys in obsm/uns."""
        mdata = test_mdata.copy()
        mdata_guide = mdata["rna"].copy()

        adata = reformat_data_for_CRT(mdata, mdata_guide)
        assert "cnmf_usage" in adata.obsm
        assert "guide_assignment" in adata.obsm
        assert "guide_names" in adata.uns
        assert "guide2gene" in adata.uns
        assert "program_names" in adata.uns
        assert "covar" in adata.obsm

    def test_reformat_guide2gene_mapping(self, test_mdata):
        """guide2gene dict should map every guide name to its target."""
        mdata = test_mdata.copy()
        mdata_guide = mdata["rna"].copy()

        adata = reformat_data_for_CRT(mdata, mdata_guide)
        g2g = adata.uns["guide2gene"]
        assert isinstance(g2g, dict)
        assert len(g2g) == len(adata.uns["guide_names"])
        for gname in adata.uns["guide_names"]:
            assert gname in g2g
