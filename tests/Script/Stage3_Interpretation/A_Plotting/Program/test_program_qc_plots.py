"""
Unit tests for Program_QC_plots.py plotting functions.

Tests use a combination of real MuData (from inference output) and synthetic
perturbation data. All plots are saved to tests/output/Interpretation/Plotting/Program/.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Interpretation/Plotting/Program/test_program_qc_plots.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Interpretation.Plotting.src.Program_QC_plots import (
    compute_program_correlation_matrix,
    analyze_program_correlations,
    plot_top_gene_per_program,
    plot_violin,
    plot_program_log2FC,
    plot_program_volcano,
    compute_program_waterfall_cor,
    create_program_correlation_waterfall,
    top_GO_per_program,
)


class TestComputeProgramCorrelation:

    def test_returns_symmetric_dataframe(self, test_mdata):
        """compute_program_correlation_matrix returns a symmetric DataFrame."""
        result = compute_program_correlation_matrix(test_mdata)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        n_programs = test_mdata['cNMF'].n_vars
        assert result.shape[0] == n_programs
        # Check symmetry
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)


class TestAnalyzeProgramCorrelations:

    def test_returns_axes(self, synthetic_program_correlation, program_output_dir):
        """analyze_program_correlations returns an Axes object."""
        ax = analyze_program_correlations(
            synthetic_program_correlation,
            Target_Program=0,
            num_program=3,
            save_path=program_output_dir,
            save_name="test_program_correlations",
        )
        plt.close('all')
        assert ax is not None

    def test_missing_program(self, synthetic_program_correlation, program_output_dir):
        """analyze_program_correlations handles missing program gracefully."""
        result = analyze_program_correlations(
            synthetic_program_correlation,
            Target_Program=999,
            save_path=program_output_dir,
            save_name="test_program_corr_missing",
        )
        plt.close('all')
        # Returns None in standalone mode when program not found
        assert result is None


class TestPlotTopGenePerProgram:

    def test_returns_axes_and_saves(self, test_mdata, program_output_dir):
        """plot_top_gene_per_program returns Axes and saves SVG."""
        prog_name = test_mdata['cNMF'].var_names[0]
        ax = plot_top_gene_per_program(
            test_mdata,
            Target_Program=prog_name,
            num_gene=5,
            save_path=program_output_dir,
            save_name="test_top_gene_per_program",
        )
        plt.close('all')
        assert ax is not None
        assert os.path.isfile(os.path.join(program_output_dir, "test_top_gene_per_program.svg"))


class TestPlotViolin:

    def test_returns_axes_and_saves(self, test_mdata, program_output_dir):
        """plot_violin returns Axes and saves SVG."""
        prog_name = test_mdata['cNMF'].var_names[0]
        ax = plot_violin(
            test_mdata,
            Target_Program=prog_name,
            groupby='batch',
            save_path=program_output_dir,
            save_name="test_violin",
        )
        plt.close('all')
        assert ax is not None
        assert os.path.isfile(os.path.join(program_output_dir, "test_violin.svg"))


class TestPlotProgramLog2FC:

    def test_returns_axes_and_df(self, synthetic_perturbation_tsv, program_output_dir):
        """plot_program_log2FC returns (Axes, DataFrame)."""
        fig, ax = plt.subplots()
        ax, df = plot_program_log2FC(
            synthetic_perturbation_tsv,
            Target='0',
            tagert_col_name='target_name',
            plot_col_name='program_name',
            num_item=3,
            p_value=0.5,
            ax=ax,
        )
        fig.savefig(os.path.join(program_output_dir, "test_program_log2fc.png"), dpi=100)
        plt.close('all')
        assert ax is not None
        assert isinstance(df, pd.DataFrame)
        assert os.path.isfile(os.path.join(program_output_dir, "test_program_log2fc.png"))

    def test_missing_target(self, synthetic_perturbation_tsv, program_output_dir):
        """plot_program_log2FC handles missing target gracefully."""
        fig, ax = plt.subplots()
        result_ax, df = plot_program_log2FC(
            synthetic_perturbation_tsv,
            Target='NONEXISTENT',
            ax=ax,
        )
        fig.savefig(os.path.join(program_output_dir, "test_program_log2fc_missing.png"), dpi=100)
        plt.close('all')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestPlotProgramVolcano:

    def test_returns_axes_df_texts(self, synthetic_perturbation_tsv, program_output_dir):
        """plot_program_volcano returns (Axes, DataFrame, list)."""
        fig, ax = plt.subplots()
        result_ax, df, texts = plot_program_volcano(
            synthetic_perturbation_tsv,
            Target='0',
            tagert_col_name='target_name',
            plot_col_name='program_name',
            p_value=0.5,
            ax=ax,
        )
        fig.savefig(os.path.join(program_output_dir, "test_program_volcano.png"), dpi=100)
        plt.close('all')
        assert result_ax is not None
        assert isinstance(df, pd.DataFrame)
        assert isinstance(texts, list)
        assert os.path.isfile(os.path.join(program_output_dir, "test_program_volcano.png"))


class TestComputeProgramWaterfallCor:

    def test_returns_correlation_matrix(self, synthetic_perturbation_tsv, program_output_dir):
        """compute_program_waterfall_cor returns a DataFrame with NaN diagonal."""
        save_path = os.path.join(program_output_dir, "test_waterfall_corr.tsv")
        result = compute_program_waterfall_cor(
            synthetic_perturbation_tsv,
            save_path=save_path,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        assert os.path.isfile(save_path)


class TestCreateProgramCorrelationWaterfall:

    def test_returns_axes_and_texts(self, synthetic_program_correlation, program_output_dir):
        """create_program_correlation_waterfall returns (Axes, list of texts)."""
        # Need NaN diagonal like compute_program_waterfall_cor produces
        corr = synthetic_program_correlation.copy()
        np.fill_diagonal(corr.values, np.nan)
        fig, ax = plt.subplots()
        result_ax, texts = create_program_correlation_waterfall(
            corr,
            Target_Program=0,
            top_num=2,
            ax=ax,
        )
        fig.savefig(os.path.join(program_output_dir, "test_program_waterfall.png"), dpi=100)
        plt.close('all')
        assert result_ax is not None
        assert isinstance(texts, list)
        assert os.path.isfile(os.path.join(program_output_dir, "test_program_waterfall.png"))


class TestTopGOPerProgram:

    def test_returns_axes_and_labels(self, synthetic_go_tsv, program_output_dir):
        """top_GO_per_program returns (Axes, list of wrapped labels)."""
        ax, labels = top_GO_per_program(
            synthetic_go_tsv,
            Target_Program=0,
            num_term=3,
            save_path=program_output_dir,
            save_name="test_go_per_program",
        )
        plt.close('all')
        assert ax is not None
        assert isinstance(labels, list)
        assert len(labels) == 3
        assert os.path.isfile(os.path.join(program_output_dir, "test_go_per_program.svg"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
