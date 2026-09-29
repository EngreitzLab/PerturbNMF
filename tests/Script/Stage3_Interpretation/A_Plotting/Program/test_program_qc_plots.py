"""
Unit tests for Program_QC_plots.py plotting functions.

Tests use a combination of real MuData (from inference output) and synthetic
perturbation data. All plots are saved to tests/output/Interpretation/Plotting/Program/.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd <repo root>
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/Program/test_program_qc_plots.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Stage3_Interpretation.A_Plotting.src.Program_QC_plots import (
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

    def test_returns_symmetric_dataframe(self, test_mdata, program_output_dir):
        """compute_program_correlation_matrix returns a symmetric DataFrame equal to pandas .corr()."""
        save_path = os.path.join(program_output_dir, "test_program_corr.npz")
        result = compute_program_correlation_matrix(test_mdata, save_path=save_path)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        n_programs = test_mdata['cNMF'].n_vars
        assert result.shape[0] == n_programs
        # Check symmetry
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)
        # Same values as the legacy pandas path
        X = test_mdata['cNMF'].X
        X = X.toarray() if hasattr(X, 'toarray') else X
        expected = pd.DataFrame(X).corr().fillna(0).values
        np.testing.assert_allclose(result.values, expected, atol=1e-8)
        with np.load(save_path) as f:
            assert f["corr"].shape == (n_programs, n_programs)
            assert list(f["program"]) == list(result.columns)


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

    def test_missing_program_raises(self, test_mdata, program_output_dir):
        """plot_top_gene_per_program raises ValueError for a program id that doesn't exist."""
        with pytest.raises(ValueError, match="not found in the loading matrix"):
            plot_top_gene_per_program(
                test_mdata,
                Target_Program="NONEXISTENT_PROGRAM",
                num_gene=5,
                save_path=program_output_dir,
                save_name="test_top_gene_missing",
            )

    def test_num_gene_exceeds_genes_raises(self, test_mdata, program_output_dir):
        """num_gene larger than n_genes should raise ValueError."""
        prog_name = test_mdata['cNMF'].var_names[0]
        n_genes = test_mdata['cNMF'].varm['loadings'].shape[1]
        with pytest.raises(ValueError, match="exceeds the number of genes"):
            plot_top_gene_per_program(
                test_mdata,
                Target_Program=prog_name,
                num_gene=n_genes + 5,
                save_path=program_output_dir,
                save_name="test_top_gene_too_many",
            )


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
        """compute_program_waterfall_cor returns a DataFrame with diagonal 1 and writes the .npz."""
        save_path = os.path.join(program_output_dir, "test_program_waterfall_corr.npz")
        result = compute_program_waterfall_cor(
            synthetic_perturbation_tsv,
            save_path=save_path,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        d = np.diag(result.values)
        assert np.allclose(d[np.isfinite(d)], 1.0)  # self kept; the waterfall plots drop it
        with np.load(save_path) as f:
            full = f["corr"]
            assert list(f["program"]) == list(result.index)
        # Saved matrix: diagonal 1 (NaN only for a constant program), symmetric
        finite = np.isfinite(np.diag(full))
        assert np.allclose(np.diag(full)[finite], 1.0)
        assert np.allclose(full, full.T, atol=1e-6, equal_nan=True)


class TestCreateProgramCorrelationWaterfall:

    def test_returns_axes_and_texts(self, synthetic_program_correlation, program_output_dir):
        """create_program_correlation_waterfall returns (Axes, list of texts) and skips self."""
        # Diagonal 1, as compute_program_waterfall_cor produces; the plot drops self
        corr = synthetic_program_correlation.copy()
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
        assert "0" not in [t.get_text() for t in texts]  # the program itself is not labeled
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
