"""
Regression tests for the optional-perturbation path in
html_Perturbed_gene_QC_plots.export_gene_html.

Mirrors create_comprehensive_plot (the PDF builder): when perturb_path_base is
None the per-condition panels are skipped instead of crashing, and the required
correlation arguments are validated up front.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/gene_html/test_html_gene_no_perturb.py -v
"""

import json
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")

from Stage3_Interpretation.A_Plotting.src.html_Perturbed_gene_QC_plots import export_gene_html


def _export(tmp_path, test_mdata, target_gene, gene_loading_corr_matrix,
            available_samples, perturb_path_base=None, perturb_corr_by_sample=None):
    export_gene_html(
        mdata=test_mdata,
        perturb_path_base=perturb_path_base,
        ensembl_to_symbol_file=None,
        Target_Gene=target_gene,
        gene_loading_corr_matrix=gene_loading_corr_matrix,
        perturb_corr_by_sample=perturb_corr_by_sample,
        sample=available_samples,
        html_share_path=str(tmp_path),
        top_n_programs=3,
        top_corr_genes=2,
        groupby="batch",
        significance_threshold=0.5,
        gene_name_key="symbol",
        control_target_name="non-targeting",
        umap_dot_size=4,
        subsample_frac=0.1,
    )
    return Path(tmp_path) / f"gene_{target_gene}"


class TestNoPerturbPathBase:

    def test_page_written_without_perturb_files(
        self, tmp_path, test_mdata, target_gene, gene_loading_corr_matrix, available_samples
    ):
        """perturb_path_base=None still produces a page; no per-condition JSON."""
        gene_dir = _export(tmp_path, test_mdata, target_gene,
                           gene_loading_corr_matrix, available_samples)
        assert (gene_dir / f"gene_{target_gene}.html").is_file()
        # header panels still there
        assert (gene_dir / "data" / "top_programs.json").is_file()
        assert (gene_dir / "data" / "correlations.json").is_file()
        # per-condition panels skipped
        assert not list((gene_dir / "data").glob("log2fc_*.json"))
        assert not list((gene_dir / "data").glob("volcano_*.json"))
        assert not list((gene_dir / "data").glob("waterfall_*.json"))

    def test_metadata_reports_zero_significant(
        self, tmp_path, test_mdata, target_gene, gene_loading_corr_matrix, available_samples
    ):
        gene_dir = _export(tmp_path, test_mdata, target_gene,
                           gene_loading_corr_matrix, available_samples)
        with open(gene_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["n_significant_program_perturbations_total"] == 0
        assert meta["n_significant_program_perturbations_per_sample"] == {}


class TestRequiredArgs:

    def test_missing_loading_corr_raises(
        self, tmp_path, test_mdata, target_gene, available_samples
    ):
        with pytest.raises(ValueError, match="gene_loading_corr_matrix is required"):
            _export(tmp_path, test_mdata, target_gene, None, available_samples)

    def test_perturb_base_without_corr_raises(
        self, tmp_path, test_mdata, target_gene, gene_loading_corr_matrix,
        available_samples, perturb_path_base
    ):
        with pytest.raises(ValueError, match="perturb_corr_by_sample is required"):
            _export(tmp_path, test_mdata, target_gene, gene_loading_corr_matrix,
                    available_samples, perturb_path_base=perturb_path_base,
                    perturb_corr_by_sample=None)
