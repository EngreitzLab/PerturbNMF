"""
Unit tests for Program_expression_weighted_plots.py functions.

Tests use real MuData and synthetic perturbation files.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/Program/test_program_expression_weighted_plots.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Stage3_Interpretation.A_Plotting.src.Program_expression_weighted_plots import (
    compute_program_expression_by_condition,
    plot_program_heatmap_weighted,
    plot_program_heatmap_expression_scaled,
)


class TestComputeProgramExpressionByCondition:

    def test_returns_series(self, test_mdata):
        """compute_program_expression_by_condition returns a Series indexed by condition."""
        prog_name = test_mdata['cNMF'].var_names[0]
        result = compute_program_expression_by_condition(
            test_mdata, Target_Program=prog_name, groupby='batch',
        )
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        assert all(np.isfinite(result.values))


class TestPlotProgramHeatmapWeighted:

    def test_saves_file(self, test_mdata, synthetic_perturbation_base, program_output_dir):
        """plot_program_heatmap_weighted produces a plot (or returns None if no sig regulators)."""
        prog_name = test_mdata['cNMF'].var_names[0]
        result = plot_program_heatmap_weighted(
            perturb_path_base=synthetic_perturbation_base,
            mdata=test_mdata,
            Target_Program=prog_name,
            sample=['D0', 'sample_D1'],
            groupby='batch',
            p_value=0.5,  # relaxed threshold to ensure something passes
            save_path=program_output_dir,
            save_name="test_heatmap_weighted",
        )
        plt.close('all')
        # Returns Axes if regulators found, None if empty
        if result is not None:
            assert os.path.isfile(os.path.join(program_output_dir, "test_heatmap_weighted.pdf"))


class TestPlotProgramHeatmapExpressionScaled:

    def test_saves_file(self, test_mdata, synthetic_perturbation_base, program_output_dir):
        """plot_program_heatmap_expression_scaled produces a scaled heatmap."""
        prog_name = test_mdata['cNMF'].var_names[0]
        result = plot_program_heatmap_expression_scaled(
            perturb_path_base=synthetic_perturbation_base,
            mdata=test_mdata,
            Target_Program=prog_name,
            sample=['D0', 'sample_D1'],
            groupby='batch',
            p_value=0.5,
            save_path=program_output_dir,
            save_name="test_heatmap_scaled",
        )
        plt.close('all')
        if result is not None:
            assert os.path.isfile(os.path.join(program_output_dir, "test_heatmap_scaled.pdf"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
