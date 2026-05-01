"""
Unit tests for Perturbed_gene_QC_plots.py plotting functions.

Tests use a combination of real MuData (from inference output) and synthetic
perturbation data. All plots are saved to tests/output/Interpretation/Plotting/Gene/.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/Gene/test_perturbed_gene_qc_plots.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Stage3_Interpretation.A_Plotting.src.Perturbed_gene_QC_plots import (
    plot_top_program_per_gene,
    plot_log2FC,
    plot_volcano,
    compute_gene_correlation_matrix,
    analyze_correlations,
    compute_gene_waterfall_cor,
    create_gene_correlation_waterfall,
    plot_perturbation_vs_control,
    plot_perturbation_vs_control_by_condition,
)


class TestPlotTopProgramPerGene:

    def test_returns_axes_and_saves(self, test_mdata, gene_output_dir):
        """plot_top_program_per_gene returns Axes and saves SVG for a known gene."""
        # Pick the first gene from the RNA var_names
        gene = test_mdata['rna'].var_names[0]
        ax = plot_top_program_per_gene(
            test_mdata,
            Target_Gene=gene,
            top_n_programs=3,
            save_path=gene_output_dir,
            save_name="test_top_program_per_gene",
        )
        plt.close('all')
        assert ax is not None
        assert os.path.isfile(os.path.join(gene_output_dir, "test_top_program_per_gene.svg"))

    def test_missing_gene(self, test_mdata, gene_output_dir):
        """plot_top_program_per_gene returns None for a missing gene (standalone)."""
        result = plot_top_program_per_gene(
            test_mdata,
            Target_Gene='NONEXISTENT_GENE_XYZ',
            save_path=gene_output_dir,
            save_name="test_top_program_missing",
        )
        plt.close('all')
        assert result is None


class TestPlotLog2FC:

    def test_returns_axes_and_df(self, synthetic_gene_perturbation_tsv, gene_output_dir):
        """plot_log2FC returns (Axes, DataFrame) and saves SVG."""
        ax, df = plot_log2FC(
            synthetic_gene_perturbation_tsv,
            Target='gene_0',
            num_item=3,
            significance_threshold=0.5,
            save_path=gene_output_dir,
            save_name="test_gene_log2fc",
        )
        plt.close('all')
        assert ax is not None
        assert isinstance(df, pd.DataFrame)
        assert os.path.isfile(os.path.join(gene_output_dir, "test_gene_log2fc.svg"))

    def test_missing_target(self, synthetic_gene_perturbation_tsv, gene_output_dir):
        """plot_log2FC handles missing target gene gracefully."""
        fig, ax = plt.subplots()
        result_ax, df = plot_log2FC(
            synthetic_gene_perturbation_tsv,
            Target='NONEXISTENT',
            ax=ax,
        )
        fig.savefig(os.path.join(gene_output_dir, "test_gene_log2fc_missing.png"), dpi=100)
        plt.close('all')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestPlotVolcano:

    def test_returns_axes_df_texts(self, synthetic_gene_perturbation_tsv, gene_output_dir):
        """plot_volcano returns (Axes, DataFrame, list) and saves SVG."""
        ax, df, texts = plot_volcano(
            synthetic_gene_perturbation_tsv,
            Target='gene_0',
            significance_threshold=0.5,
            save_path=gene_output_dir,
            save_name="test_gene_volcano",
        )
        plt.close('all')
        assert ax is not None
        assert isinstance(df, pd.DataFrame)
        assert isinstance(texts, list)
        assert os.path.isfile(os.path.join(gene_output_dir, "test_gene_volcano.svg"))


class TestComputeGeneCorrelationMatrix:

    def test_returns_symmetric_dataframe(self, test_mdata):
        """compute_gene_correlation_matrix returns a symmetric DataFrame."""
        result = compute_gene_correlation_matrix(test_mdata)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        n_genes = test_mdata['rna'].n_vars
        assert result.shape[0] == n_genes


class TestAnalyzeCorrelations:

    def test_returns_axes(self, gene_output_dir):
        """analyze_correlations returns Axes for a known gene."""
        genes = [f'gene_{i}' for i in range(10)]
        rng = np.random.default_rng(42)
        mat = rng.uniform(-0.5, 1.0, (10, 10))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 1.0)
        corr = pd.DataFrame(mat, index=genes, columns=genes)

        ax = analyze_correlations(
            corr,
            Target='gene_0',
            top_corr_genes=3,
            save_path=gene_output_dir,
            save_name="test_gene_correlations",
        )
        plt.close('all')
        assert ax is not None
        assert os.path.isfile(os.path.join(gene_output_dir, "test_gene_correlations.svg"))

    def test_missing_gene(self, gene_output_dir):
        """analyze_correlations returns None for missing gene (standalone)."""
        corr = pd.DataFrame(
            np.eye(3), index=['a', 'b', 'c'], columns=['a', 'b', 'c']
        )
        result = analyze_correlations(
            corr, Target='MISSING',
            save_path=gene_output_dir,
            save_name="test_gene_corr_missing",
        )
        plt.close('all')
        assert result is None


class TestComputeGeneWaterfallCor:

    def test_returns_correlation_matrix(self, synthetic_gene_perturbation_tsv, gene_output_dir):
        """compute_gene_waterfall_cor returns a DataFrame with NaN diagonal."""
        save_path = os.path.join(gene_output_dir, "test_gene_waterfall_corr.tsv")
        result = compute_gene_waterfall_cor(
            synthetic_gene_perturbation_tsv,
            save_path=save_path,
        )
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]
        # Diagonal should be NaN
        assert all(np.isnan(np.diag(result.values)))
        assert os.path.isfile(save_path)


class TestCreateGeneCorrelationWaterfall:

    def test_returns_axes_and_texts(self, synthetic_gene_correlation, gene_output_dir):
        """create_gene_correlation_waterfall returns (Axes, list of texts)."""
        ax, texts = create_gene_correlation_waterfall(
            synthetic_gene_correlation,
            Target_Gene='gene_0',
            top_corr_genes=2,
            save_path=gene_output_dir,
            save_name="test_gene_waterfall",
        )
        plt.close('all')
        assert ax is not None
        assert isinstance(texts, list)
        assert os.path.isfile(os.path.join(gene_output_dir, "test_gene_waterfall.svg"))


class TestPlotPerturbationVsControl:

    def test_with_real_mdata(self, test_mdata, gene_output_dir):
        """plot_perturbation_vs_control works if guide_assignment exists."""
        if 'guide_assignment' not in test_mdata['cNMF'].obsm:
            pytest.skip("guide_assignment not in mdata")
        if 'guide_targets' not in test_mdata['cNMF'].uns:
            pytest.skip("guide_targets not in mdata")

        # Pick a target gene that actually exists in guide_targets
        targets = np.array(test_mdata['cNMF'].uns['guide_targets'])
        valid = [t for t in np.unique(targets) if t != 'non-targeting']
        if not valid:
            pytest.skip("No targeting guides found")

        target_gene = valid[0]
        fig, ax = plt.subplots()
        result = plot_perturbation_vs_control(
            test_mdata,
            target_gene=target_gene,
            ax=ax,
        )
        fig.savefig(os.path.join(gene_output_dir, "test_perturb_vs_control.png"), dpi=100)
        plt.close('all')
        assert os.path.isfile(os.path.join(gene_output_dir, "test_perturb_vs_control.png"))


class TestPlotPerturbationVsControlByCondition:

    def test_with_real_mdata(self, test_mdata, gene_output_dir):
        """plot_perturbation_vs_control_by_condition works if guide data exists."""
        if 'guide_assignment' not in test_mdata['cNMF'].obsm:
            pytest.skip("guide_assignment not in mdata")
        if 'guide_targets' not in test_mdata['cNMF'].uns:
            pytest.skip("guide_targets not in mdata")

        targets = np.array(test_mdata['cNMF'].uns['guide_targets'])
        valid = [t for t in np.unique(targets) if t != 'non-targeting']
        if not valid:
            pytest.skip("No targeting guides found")

        target_gene = valid[0]
        fig, ax = plt.subplots()
        result = plot_perturbation_vs_control_by_condition(
            test_mdata,
            target_gene=target_gene,
            condition_key='batch',
            ax=ax,
        )
        fig.savefig(os.path.join(gene_output_dir, "test_perturb_vs_control_by_condition.png"), dpi=100)
        plt.close('all')
        assert os.path.isfile(os.path.join(gene_output_dir, "test_perturb_vs_control_by_condition.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
