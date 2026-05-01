"""
Unit tests for k_selection_plots.py plotting functions.

Tests use synthetic data and verify that plots are saved to disk.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/KSelection/test_k_selection_plots.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Stage3_Interpretation.A_Plotting.src.k_selection_plots import (
    _style_ax,
    _filter_components,
    _load_stats_file,
    plot_stablity_error,
    plot_enrichment,
    plot_perturbation,
    plot_explained_variance,
    plot_k_selection_panel,
    plot_k_selection_panel_no_traits,
)


class TestHelpers:

    def test_style_ax(self):
        """_style_ax applies styling without crashing."""
        fig, ax = plt.subplots()
        _style_ax(ax, xlabel='X', ylabel='Y', title='Test')
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        plt.close('all')

    def test_filter_components(self):
        """_filter_components keeps only requested K values."""
        stats = pd.DataFrame({
            'k': [5, 10, 15, 20],
            'silhouette': [0.8, 0.7, 0.6, 0.5],
            'prediction_error': [1e6, 8e5, 6e5, 4e5],
        })
        result = _filter_components(stats, [5, 15, 99])
        assert list(result['k']) == [5, 15]
        assert len(result) == 2

    def test_load_stats_file_tsv(self, kselection_output_dir):
        """_load_stats_file reads a TSV file correctly."""
        df = pd.DataFrame({
            'k': [5, 10],
            'silhouette': [0.8, 0.7],
            'prediction_error': [1e6, 8e5],
        })
        path = os.path.join(kselection_output_dir, 'test_stats.tsv')
        df.to_csv(path, sep='\t', index=False)
        result = _load_stats_file(path)
        assert list(result.columns) == ['k', 'silhouette', 'prediction_error']
        assert len(result) == 2


class TestPlotStabilityError:

    def test_saves_files(self, synthetic_stability_stats, kselection_output_dir):
        """plot_stablity_error saves stability and error PNG/SVG files."""
        plot_stablity_error(
            synthetic_stability_stats,
            folder_name=kselection_output_dir,
            file_name="test_stability_error",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_error_stability.png"))
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_error_error.png"))

    def test_no_selected_k(self, synthetic_stability_stats, kselection_output_dir):
        """plot_stablity_error works without selected_k."""
        plot_stablity_error(
            synthetic_stability_stats,
            folder_name=kselection_output_dir,
            file_name="test_stability_no_vline",
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_no_vline_stability.png"))


class TestPlotEnrichment:

    def test_saves_files(self, synthetic_count_df, kselection_output_dir):
        """plot_enrichment saves one file per enrichment category."""
        plot_enrichment(
            synthetic_count_df,
            folder_name=kselection_output_dir,
            file_name="test_enrichment",
            selected_k=15,
        )
        plt.close('all')
        for col in ['go_terms', 'genesets', 'traits']:
            assert os.path.isfile(os.path.join(kselection_output_dir, f"test_enrichment_{col}.png"))


class TestPlotPerturbation:

    def test_saves_files_and_returns_df(self, synthetic_test_stats_df, kselection_output_dir):
        """plot_perturbation saves per-sample and aggregated plots, returns DataFrame."""
        result = plot_perturbation(
            synthetic_test_stats_df,
            pval=0.05,
            folder_name=kselection_output_dir,
            file_name="test_perturbation",
            selected_k=10,
        )
        plt.close('all')
        assert isinstance(result, pd.DataFrame)
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_perturbation_per_sample.png"))
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_perturbation_all_samples.png"))


class TestPlotExplainedVariance:

    def test_saves_file(self, synthetic_explained_var, kselection_output_dir):
        """plot_explained_variance saves PNG file."""
        plot_explained_variance(
            synthetic_explained_var,
            folder_name=kselection_output_dir,
            file_name="test_explained_var",
            selected_k=15,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_explained_var.png"))


class TestPlotPanel:

    def test_k_selection_panel(self, synthetic_stability_stats, synthetic_count_df,
                                synthetic_test_stats_df, synthetic_explained_var,
                                kselection_output_dir):
        """plot_k_selection_panel saves the combined 3x3 panel."""
        plot_k_selection_panel(
            synthetic_stability_stats,
            synthetic_count_df,
            synthetic_test_stats_df,
            synthetic_explained_var,
            pval=0.05,
            folder_name=kselection_output_dir,
            file_name="test_panel",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_panel.png"))

    def test_k_selection_panel_no_traits(self, synthetic_stability_stats, synthetic_count_df,
                                          synthetic_test_stats_df, synthetic_explained_var,
                                          kselection_output_dir):
        """plot_k_selection_panel_no_traits saves the combined panel without traits."""
        plot_k_selection_panel_no_traits(
            synthetic_stability_stats,
            synthetic_count_df,
            synthetic_test_stats_df,
            synthetic_explained_var,
            pval=0.05,
            folder_name=kselection_output_dir,
            file_name="test_panel_no_traits",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_panel_no_traits.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
