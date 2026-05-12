"""
Unit tests for html_Program_QC_plots.export_program_html and write_share_index.

Uses real inference + evaluation output from tests/output/torch-cNMF/batch/.
Asserts that the per-program share subtree is written on disk with the
expected files (HTML page, metadata.json, per-panel JSON, UMAP PNG),
and that write_share_index produces index.html + shared/style.css + manifest.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/program_html/test_html_program.py -v
"""

import json
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")

from Stage3_Interpretation.A_Plotting.src.html_Program_QC_plots import (
    export_program_html,
    write_share_index,
    _build_top_genes,
    _build_go_terms,
    _build_correlations,
    _build_violin,
)


# ---------------------------------------------------------------------------
# Tests for individual JSON-builder helpers (cheap, no plot rendering)
# ---------------------------------------------------------------------------

class TestBuilders:

    def test_build_top_genes(self, test_mdata):
        target = str(test_mdata["cNMF"].var_names[0])
        d = _build_top_genes(test_mdata, target, num_gene=5,
                             file_to_dictionary=None, gene_name_key="symbol")
        assert set(d.keys()) == {"genes", "loadings"}
        assert len(d["genes"]) == 5
        assert len(d["loadings"]) == 5

    def test_build_go_terms(self, go_path, test_mdata):
        target = str(test_mdata["cNMF"].var_names[0])
        d = _build_go_terms(go_path, target, num_term=3,
                            p_value_name="Adjusted P-value", term_col="Term")
        assert "terms" in d and "adj_pval" in d and "neglog10p" in d

    def test_build_correlations(self, program_correlation, test_mdata):
        target = str(test_mdata["cNMF"].var_names[0])
        d = _build_correlations(program_correlation, target, num_program=2)
        assert set(d.keys()) == {"programs", "r", "direction"}
        assert all(direc in {"positive", "negative"} for direc in d["direction"])

    def test_build_violin(self, test_mdata):
        target = str(test_mdata["cNMF"].var_names[0])
        d = _build_violin(test_mdata, target, groupby="batch")
        assert set(d.keys()) == {"groups", "per_group_expression", "summary"}
        assert len(d["groups"]) > 0
        assert all(g in d["per_group_expression"] for g in d["groups"])


# ---------------------------------------------------------------------------
# End-to-end: export_program_html writes the full subtree
# ---------------------------------------------------------------------------

class TestExportProgramHTML:

    @pytest.fixture(scope="class")
    def exported_program(self, test_mdata, perturb_path_base, go_path,
                         program_correlation, waterfall_correlation,
                         perturbed_gene_found, html_share_path, available_samples):
        target = str(test_mdata["cNMF"].var_names[0])
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
            html_share_path=html_share_path,
            top_program=2,
            groupby="batch",
            top_enrichned_term=3,
            p_value=0.5,
            gene_name_key="symbol",
            subsample_frac=0.1,
            position_index=1,
            position_total=5,
        )
        return Path(html_share_path) / f"program_{target}", target, available_samples

    def test_html_page_written(self, exported_program):
        prog_dir, pid, _ = exported_program
        html = prog_dir / f"program_{pid}.html"
        assert html.is_file(), f"Missing HTML page: {html}"
        assert html.stat().st_size > 5000
        content = html.read_text()
        assert "Plotly" in content or "plotly" in content
        assert pid in content

    def test_umap_png_written(self, exported_program):
        prog_dir, _, _ = exported_program
        umap = prog_dir / "images" / "umap.png"
        assert umap.is_file(), f"Missing UMAP png: {umap}"
        assert umap.stat().st_size > 1000

    def test_per_panel_json_written(self, exported_program):
        prog_dir, _, samples = exported_program
        data_dir = prog_dir / "data"
        # Header panels
        for fname in ["top_genes.json", "go_terms.json", "correlations.json",
                      "violin.json", "heatmap.json"]:
            p = data_dir / fname
            assert p.is_file(), f"Missing per-panel JSON: {p}"
            with open(p) as f:
                json.load(f)
        # Per-sample panels
        for samp in samples:
            for kind in ["log2fc", "volcano", "dotplot", "waterfall"]:
                p = data_dir / f"{kind}_{samp}.json"
                assert p.is_file(), f"Missing per-sample JSON: {p}"
                with open(p) as f:
                    json.load(f)

    def test_metadata_json(self, exported_program):
        prog_dir, pid, samples = exported_program
        meta = prog_dir / "metadata.json"
        assert meta.is_file()
        with open(meta) as f:
            m = json.load(f)
        assert m["program_id"] == pid
        assert m["samples"] == samples
        assert "n_significant_regulators_total" in m
        assert "top_GO_terms" in m


# ---------------------------------------------------------------------------
# write_share_index
# ---------------------------------------------------------------------------

class TestWriteShareIndex:

    @pytest.fixture
    def ensure_one_program_exported(self, test_mdata, perturb_path_base, go_path,
                                    program_correlation, waterfall_correlation,
                                    perturbed_gene_found, html_share_path,
                                    available_samples):
        target = str(test_mdata["cNMF"].var_names[0])
        prog_dir = Path(html_share_path) / f"program_{target}"
        if not (prog_dir / f"program_{target}.html").exists():
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
                html_share_path=html_share_path,
                top_program=2,
                groupby="batch",
                top_enrichned_term=3,
                p_value=0.5,
                gene_name_key="symbol",
                subsample_frac=0.1,
            )
        return target

    def test_share_index_written(self, ensure_one_program_exported, html_share_path, test_mdata):
        program_ids = [str(p) for p in test_mdata["cNMF"].var_names]
        write_share_index(html_share_path, program_ids, {"test": "config", "k": 5})

        share = Path(html_share_path)
        assert (share / "shared" / "style.css").is_file()
        manifest = share / "shared" / "manifest.json"
        assert manifest.is_file()
        with open(manifest) as f:
            m = json.load(f)
        assert m["program_ids"] == program_ids
        idx = share / "index.html"
        assert idx.is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
