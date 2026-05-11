"""
Unit tests for k_selection_plots.py — loaders and plotters.

Tests use real torch-cNMF batch output (K=5,10,15) for data loaders and
plot functions. Helper/logic tests use inline synthetic data.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/KSelection/test_k_selection_plots.py -v
"""

import os

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
    load_stablity_error_data,
    load_enrichment_data,
    load_perturbation_data,
    load_explained_variance_data,
    plot_stablity_error,
    plot_enrichment,
    plot_perturbation,
    plot_explained_variance,
    plot_k_selection_panel,
    plot_k_selection_panel_no_traits,
)


# ---------------------------------------------------------------------------
# Helper / pure-logic tests (no real data needed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loader tests (real data)
# ---------------------------------------------------------------------------

class TestLoadStabilityError:

    def test_returns_valid_dataframe(self, real_stability_stats):
        """load_stablity_error_data returns DataFrame with expected columns."""
        df = real_stability_stats
        assert isinstance(df, pd.DataFrame)
        assert 'k' in df.columns
        assert 'silhouette' in df.columns
        assert 'prediction_error' in df.columns

    def test_contains_expected_k_values(self, real_stability_stats):
        """Loaded stats should contain K=5,10,15."""
        ks = set(real_stability_stats['k'].tolist())
        assert {5, 10, 15}.issubset(ks)

    def test_silhouette_in_valid_range(self, real_stability_stats):
        """Silhouette scores should be in [-1, 1]."""
        vals = real_stability_stats['silhouette']
        assert vals.min() >= -1.0
        assert vals.max() <= 1.0

    def test_prediction_error_positive(self, real_stability_stats):
        """Prediction errors should be non-negative."""
        assert (real_stability_stats['prediction_error'] >= 0).all()


class TestLoadEnrichment:

    def test_returns_valid_dataframe(self, real_enrichment_df):
        """load_enrichment_data returns DataFrame indexed by K."""
        df = real_enrichment_df
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_counts_are_non_negative(self, real_enrichment_df):
        """All enrichment counts should be non-negative integers."""
        for col in real_enrichment_df.columns:
            assert (real_enrichment_df[col] >= 0).all()


class TestLoadPerturbation:

    def test_returns_valid_dataframe(self, real_perturbation_df):
        """load_perturbation_data returns DataFrame with sample and K columns."""
        df = real_perturbation_df
        assert isinstance(df, pd.DataFrame)
        assert 'sample' in df.columns
        assert 'K' in df.columns
        assert len(df) > 0

    def test_contains_expected_samples(self, real_perturbation_df):
        """Loaded data should contain at least some of the expected samples."""
        samples = set(real_perturbation_df['sample'].unique())
        assert len(samples) > 0


class TestLoadExplainedVariance:

    def test_returns_valid_dict(self, real_explained_var):
        """load_explained_variance_data returns a dict mapping K to float."""
        ev = real_explained_var
        assert isinstance(ev, dict)
        assert len(ev) > 0
        for k, v in ev.items():
            assert isinstance(k, (int, np.integer))
            assert isinstance(v, (float, np.floating))

    def test_variance_values_reasonable(self, real_explained_var):
        """Explained variance should be positive."""
        for v in real_explained_var.values():
            assert v > 0


# ---------------------------------------------------------------------------
# Plot tests (real data)
# ---------------------------------------------------------------------------

class TestPlotStabilityError:

    def test_saves_files(self, real_stability_stats, kselection_output_dir):
        """plot_stablity_error saves stability and error PNG/SVG files."""
        plot_stablity_error(
            real_stability_stats,
            folder_name=kselection_output_dir,
            file_name="test_stability_error",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_error_stability.png"))
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_error_error.png"))

    def test_no_selected_k(self, real_stability_stats, kselection_output_dir):
        """plot_stablity_error works without selected_k."""
        plot_stablity_error(
            real_stability_stats,
            folder_name=kselection_output_dir,
            file_name="test_stability_no_vline",
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_stability_no_vline_stability.png"))


class TestPlotEnrichment:

    def test_saves_files(self, real_enrichment_df, kselection_output_dir):
        """plot_enrichment saves one file per enrichment category."""
        plot_enrichment(
            real_enrichment_df,
            folder_name=kselection_output_dir,
            file_name="test_enrichment",
            selected_k=10,
        )
        plt.close('all')
        for col in real_enrichment_df.columns:
            assert os.path.isfile(os.path.join(kselection_output_dir, f"test_enrichment_{col}.png"))


class TestPlotPerturbation:

    def test_saves_files_and_returns_df(self, real_perturbation_df, kselection_output_dir):
        """plot_perturbation saves per-sample and aggregated plots, returns DataFrame."""
        result = plot_perturbation(
            real_perturbation_df,
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

    def test_saves_file(self, real_explained_var, kselection_output_dir):
        """plot_explained_variance saves PNG file."""
        plot_explained_variance(
            real_explained_var,
            folder_name=kselection_output_dir,
            file_name="test_explained_var",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_explained_var.png"))


class TestPlotPanel:

    def test_k_selection_panel(self, real_stability_stats, real_enrichment_df,
                                real_perturbation_df, real_explained_var,
                                kselection_output_dir):
        """plot_k_selection_panel saves the combined panel."""
        plot_k_selection_panel(
            real_stability_stats,
            real_enrichment_df,
            real_perturbation_df,
            real_explained_var,
            pval=0.05,
            folder_name=kselection_output_dir,
            file_name="test_panel",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_panel.png"))

    def test_k_selection_panel_no_traits(self, real_stability_stats, real_enrichment_df,
                                          real_perturbation_df, real_explained_var,
                                          kselection_output_dir):
        """plot_k_selection_panel_no_traits saves the combined panel without traits."""
        plot_k_selection_panel_no_traits(
            real_stability_stats,
            real_enrichment_df,
            real_perturbation_df,
            real_explained_var,
            pval=0.05,
            folder_name=kselection_output_dir,
            file_name="test_panel_no_traits",
            selected_k=10,
        )
        plt.close('all')
        assert os.path.isfile(os.path.join(kselection_output_dir, "test_panel_no_traits.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
