"""
Unit tests for k_quality_plots.py functions.

Tests use synthetic data for matrix computation and overlap functions.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Interpretation/Plotting/KSelection/test_k_quality_plots.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch_cnmf = pytest.importorskip("torch_cnmf", reason="torch_cnmf required for k_quality_plots imports")

from Interpretation.Plotting.src.k_quality_plots import (
    program_corr,
    program_euclidean,
    top_genes_overlap,
    sort_corr_matrix,
    build_overlap_matrix,
    compute_gene_list_GO,
    compute_gene_list_perturbation,
    plot_coefficient_variance,
)


class TestMatrixFunctions:

    def test_program_corr(self, synthetic_spectra_matrices):
        """program_corr returns a square correlation matrix with diagonal ~1.0."""
        m1, m2 = synthetic_spectra_matrices
        result = program_corr(m1, m1)
        assert result.shape == (5, 5)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)

    def test_program_corr_different_matrices(self, synthetic_spectra_matrices):
        """program_corr between different matrices produces values in [-1, 1]."""
        m1, m2 = synthetic_spectra_matrices
        result = program_corr(m1, m2)
        assert result.shape == (5, 5)
        assert result.values.min() >= -1.0 - 1e-10
        assert result.values.max() <= 1.0 + 1e-10

    def test_program_euclidean(self, synthetic_spectra_matrices):
        """program_euclidean returns non-negative distances, zero on diagonal for same matrix."""
        m1, _ = synthetic_spectra_matrices
        result = program_euclidean(m1, m1)
        assert result.shape == (5, 5)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-10)
        assert (result.values >= -1e-10).all()

    def test_top_genes_overlap(self, synthetic_spectra_matrices):
        """top_genes_overlap returns non-negative integer overlap counts."""
        m1, m2 = synthetic_spectra_matrices
        result = top_genes_overlap(m1, m2, gene_num=20)
        assert result.shape == (5, 5)
        assert (result.values >= 0).all()

    def test_top_genes_overlap_percentage(self, synthetic_spectra_matrices):
        """top_genes_overlap with percentage=True returns values in [0, 1]."""
        m1, _ = synthetic_spectra_matrices
        result = top_genes_overlap(m1, m1, percentage=True, gene_num=20)
        assert result.shape == (5, 5)
        np.testing.assert_allclose(np.diag(result), 1.0, atol=1e-10)

    def test_sort_corr_matrix(self, synthetic_spectra_matrices):
        """sort_corr_matrix produces a reordered matrix with same shape."""
        m1, _ = synthetic_spectra_matrices
        corr = program_corr(m1, m1)
        result = sort_corr_matrix(corr)
        assert result.shape == corr.shape


class TestDataHelpers:

    def test_build_overlap_matrix(self):
        """build_overlap_matrix computes Jaccard indices correctly."""
        d1 = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        d2 = {0: ['a', 'b'], 1: ['e', 'f', 'g']}
        result = build_overlap_matrix(d1, d2)
        assert result.shape == (2, 2)
        # d1[0]={'a','b','c'} & d2[0]={'a','b'} => Jaccard = 2/3
        assert np.isclose(result.iloc[0, 0], 2.0 / 3.0)
        assert (result.values >= 0).all()
        assert (result.values <= 1.0 + 1e-10).all()

    def test_build_overlap_matrix_empty_sets(self):
        """build_overlap_matrix handles empty sets."""
        d1 = {0: [], 1: ['a']}
        d2 = {0: ['b'], 1: []}
        result = build_overlap_matrix(d1, d2)
        assert result.iloc[0, 0] == 0.0  # both empty intersection, union={b}

    def test_compute_gene_list_GO(self):
        """compute_gene_list_GO extracts per-program significant GO terms."""
        df = pd.DataFrame({
            'Term': ['GO_A', 'GO_B', 'GO_C', 'GO_D'],
            'Adjusted P-value': [0.01, 0.06, 0.03, 0.01],
        }, index=[0, 0, 1, 1])
        result = compute_gene_list_GO(k=2, combined_df=df, pval=0.05)
        assert 0 in result
        assert 1 in result
        assert 'GO_A' in result[0]
        assert 'GO_B' not in result[0]  # pval > 0.05
        assert 'GO_C' in result[1]

    def test_compute_gene_list_perturbation(self):
        """compute_gene_list_perturbation extracts per-program significant regulators."""
        df = pd.DataFrame({
            'program_name': [0, 0, 1, 1],
            'adj_pval': [0.01, 0.06, 0.03, 0.8],
        }, index=['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D'])
        result = compute_gene_list_perturbation(k=2, combined_df=df, pval=0.05)
        assert 0 in result
        assert 1 in result
        assert 'GENE_A' in result[0]
        assert 'GENE_B' not in result[0]


class TestPlotFunctions:

    def test_plot_coefficient_variance(self, kselection_output_dir):
        """plot_coefficient_variance runs without crashing."""
        sg1 = {5: np.array([10, 20, 30]), 10: np.array([15, 25, 35])}
        sg2 = {5: np.array([12, 22, 32]), 10: np.array([18, 28, 38])}
        plot_coefficient_variance(sg1, sg2, title1="run1", title2="run2")
        fig = plt.gcf()
        fig.savefig(os.path.join(kselection_output_dir, "test_coefficient_variance.png"))
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_coefficient_variance.png"))

    def test_plot_coefficient_variance_three(self, kselection_output_dir):
        """plot_coefficient_variance with 3 input dicts."""
        sg1 = {5: np.array([10, 20]), 10: np.array([15, 25])}
        sg2 = {5: np.array([12, 22]), 10: np.array([18, 28])}
        sg3 = {5: np.array([14, 24]), 10: np.array([20, 30])}
        plot_coefficient_variance(sg1, sg2, sg3, title1="a", title2="b", title3="c")
        fig = plt.gcf()
        fig.savefig(os.path.join(kselection_output_dir, "test_coefficient_variance_3.png"))
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_coefficient_variance_3.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
