"""
Regression tests for the optional-perturbation path in
html_Program_QC_plots.export_program_html.

Mirrors create_comprehensive_program_plot (the PDF builder): when
perturb_path_base is None the per-sample panels and the regulator heatmap are
skipped instead of crashing, and the required arguments are validated up front.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/program_html/test_html_program_no_perturb.py -v
"""

import json
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")

from Stage3_Interpretation.A_Plotting.src.html_Program_QC_plots import export_program_html


@pytest.fixture(scope="module")
def target_program(test_mdata):
    return str(test_mdata["cNMF"].var_names[0])


def _export(tmp_path, test_mdata, go_path, target, program_correlation,
            perturbed_gene_found, available_samples,
            perturb_path_base=None, waterfall_correlation=None):
    export_program_html(
        mdata=test_mdata,
        perturb_path_base=perturb_path_base,
        GO_path=go_path,
        file_to_dictionary=None,
        Target_Program=target,
        program_correlation=program_correlation,
        waterfall_correlation=waterfall_correlation,
        sample=available_samples,
        perturbed_gene_found=perturbed_gene_found,
        html_share_path=str(tmp_path),
        top_program=2,
        groupby="batch",
        top_enrichned_term=3,
        p_value=0.5,
        gene_name_key="symbol",
        subsample_frac=0.1,
    )
    return Path(tmp_path) / f"program_{target}"


class TestNoPerturbPathBase:

    def test_page_written_without_perturb_files(
        self, tmp_path, test_mdata, go_path, target_program, program_correlation,
        perturbed_gene_found, available_samples
    ):
        """perturb_path_base=None still produces a page; no per-sample JSON, no heatmap."""
        prog_dir = _export(tmp_path, test_mdata, go_path, target_program,
                           program_correlation, perturbed_gene_found, available_samples)
        assert (prog_dir / f"program_{target_program}.html").is_file()
        # header panels still there
        assert (prog_dir / "data" / "top_genes.json").is_file()
        assert (prog_dir / "data" / "correlations.json").is_file()
        assert (prog_dir / "data" / "violin.json").is_file()
        # per-sample panels and heatmap skipped
        assert not list((prog_dir / "data").glob("log2fc_*.json"))
        assert not list((prog_dir / "data").glob("volcano_*.json"))
        assert not list((prog_dir / "data").glob("waterfall_*.json"))
        assert not (prog_dir / "data" / "heatmap.json").exists()

    def test_metadata_reports_zero_significant(
        self, tmp_path, test_mdata, go_path, target_program, program_correlation,
        perturbed_gene_found, available_samples
    ):
        prog_dir = _export(tmp_path, test_mdata, go_path, target_program,
                           program_correlation, perturbed_gene_found, available_samples)
        with open(prog_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["n_significant_regulators_total"] == 0


class TestRequiredArgs:

    def test_missing_program_correlation_raises(
        self, tmp_path, test_mdata, go_path, target_program,
        perturbed_gene_found, available_samples
    ):
        with pytest.raises(ValueError, match="program_correlation is required"):
            _export(tmp_path, test_mdata, go_path, target_program, None,
                    perturbed_gene_found, available_samples)

    def test_perturb_base_without_waterfall_raises(
        self, tmp_path, test_mdata, go_path, target_program, program_correlation,
        perturbed_gene_found, available_samples, perturb_path_base
    ):
        with pytest.raises(ValueError, match="waterfall_correlation is required"):
            _export(tmp_path, test_mdata, go_path, target_program, program_correlation,
                    perturbed_gene_found, available_samples,
                    perturb_path_base=perturb_path_base, waterfall_correlation=None)
