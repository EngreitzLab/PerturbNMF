"""
Unit tests for html_Perturbed_gene_QC_plots.export_gene_html and
write_gene_share_index. Uses real inference + evaluation output from
tests/output/torch-cNMF/batch/.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/gene_html/test_html_gene.py -v
"""

import json
from pathlib import Path

import pytest
import matplotlib
matplotlib.use("Agg")

from Stage3_Interpretation.A_Plotting.src.html_Perturbed_gene_QC_plots import (
    export_gene_html,
    write_gene_share_index,
    _build_top_programs,
    _build_gene_correlations,
    _build_log2fc_gene,
    _build_volcano_gene,
    _build_gene_waterfall,
    _build_kd_vs_control,
)


# ---------------------------------------------------------------------------
# JSON-builder helpers
# ---------------------------------------------------------------------------

class TestBuilders:

    def test_build_top_programs(self, test_mdata, target_gene):
        d = _build_top_programs(test_mdata, target_gene, top_n=3,
                                ensembl_to_symbol_file=None, gene_name_key="symbol")
        assert set(d.keys()) == {"programs", "loadings"}
        assert len(d["programs"]) <= 3

    def test_build_gene_correlations(self, gene_loading_corr_matrix, target_gene):
        d = _build_gene_correlations(gene_loading_corr_matrix, target_gene, top_n=2)
        assert set(d.keys()) == {"programs", "r", "direction"}
        assert all(direc in {"positive", "negative"} for direc in d["direction"])

    def test_build_log2fc_gene(self, perturb_path_base, target_gene, available_samples):
        d = _build_log2fc_gene(
            f"{perturb_path_base}_{available_samples[0]}.txt", target_gene,
            target_col="target_name", program_col="program_name",
            log2fc_col="log2FC", num_item=3, p_value=0.5,
        )
        assert "regulators" in d and "log2fc" in d and "adj_pval" in d

    def test_build_volcano_gene(self, perturb_path_base, target_gene, available_samples):
        d = _build_volcano_gene(
            f"{perturb_path_base}_{available_samples[0]}.txt", target_gene,
            target_col="target_name", program_col="program_name",
            log2fc_col="log2FC",
            down_thred_log=-0.05, up_thred_log=0.05, p_value=0.5,
        )
        assert "category" in d
        assert all(c in {"up", "down", "ns"} for c in d["category"])

    def test_build_gene_waterfall(self, perturb_corr_by_sample, target_gene, available_samples):
        d = _build_gene_waterfall(perturb_corr_by_sample[available_samples[0]],
                                  target_gene, top_num=2)
        assert set(d.keys()) == {"programs", "r", "labeled"}

    def test_build_kd_vs_control(self, test_mdata, target_gene):
        d = _build_kd_vs_control(
            test_mdata, target_gene,
            condition_key="batch", control_target_name="non-targeting",
            gene_name_key="symbol",
        )
        assert "groups" in d


# ---------------------------------------------------------------------------
# End-to-end: export_gene_html writes the full subtree
# ---------------------------------------------------------------------------

class TestExportGeneHTML:

    @pytest.fixture(scope="class")
    def exported_gene(self, test_mdata, target_gene, perturb_path_base,
                      gene_loading_corr_matrix, perturb_corr_by_sample,
                      html_share_path, available_samples):
        export_gene_html(
            mdata=test_mdata,
            perturb_path_base=perturb_path_base,
            ensembl_to_symbol_file=None,
            Target_Gene=target_gene,
            gene_loading_corr_matrix=gene_loading_corr_matrix,
            perturb_corr_by_sample=perturb_corr_by_sample,
            sample=available_samples,
            html_share_path=html_share_path,
            top_n_programs=3,
            top_corr_genes=2,
            groupby="batch",
            significance_threshold=0.5,
            gene_name_key="symbol",
            control_target_name="non-targeting",
            umap_dot_size=4,
            subsample_frac=0.1,
            position_index=1,
            position_total=1,
        )
        return Path(html_share_path) / f"gene_{target_gene}", target_gene, available_samples

    def test_html_page_written(self, exported_gene):
        gene_dir, gene, _ = exported_gene
        html = gene_dir / f"gene_{gene}.html"
        assert html.is_file(), f"Missing HTML page: {html}"
        assert html.stat().st_size > 5000
        content = html.read_text()
        assert "Plotly" in content or "plotly" in content
        assert gene in content

    def test_png_images_written(self, exported_gene):
        gene_dir, _, _ = exported_gene
        for fname in ["umap_expression.png", "umap_guide.png", "gene_dotplot.png"]:
            p = gene_dir / "images" / fname
            assert p.is_file(), f"Missing image: {p}"
            assert p.stat().st_size > 500

    def test_per_panel_json_written(self, exported_gene):
        gene_dir, _, samples = exported_gene
        data_dir = gene_dir / "data"
        for fname in ["top_programs.json", "correlations.json", "kd_vs_control.json"]:
            p = data_dir / fname
            assert p.is_file(), f"Missing per-panel JSON: {p}"
            with open(p) as f:
                json.load(f)
        for samp in samples:
            for kind in ["log2fc", "volcano", "programs_dotplot", "waterfall"]:
                p = data_dir / f"{kind}_{samp}.json"
                assert p.is_file(), f"Missing per-sample JSON: {p}"
                with open(p) as f:
                    json.load(f)

    def test_metadata_json(self, exported_gene):
        gene_dir, gene, samples = exported_gene
        meta = gene_dir / "metadata.json"
        assert meta.is_file()
        with open(meta) as f:
            m = json.load(f)
        assert m["gene_symbol"] == gene
        assert m["samples"] == samples
        assert "n_significant_program_perturbations_total" in m


# ---------------------------------------------------------------------------
# write_gene_share_index
# ---------------------------------------------------------------------------

class TestWriteGeneShareIndex:

    @pytest.fixture
    def ensure_one_gene_exported(self, test_mdata, target_gene, perturb_path_base,
                                 gene_loading_corr_matrix, perturb_corr_by_sample,
                                 html_share_path, available_samples):
        gene_dir = Path(html_share_path) / f"gene_{target_gene}"
        if not (gene_dir / f"gene_{target_gene}.html").exists():
            export_gene_html(
                mdata=test_mdata,
                perturb_path_base=perturb_path_base,
                ensembl_to_symbol_file=None,
                Target_Gene=target_gene,
                gene_loading_corr_matrix=gene_loading_corr_matrix,
                perturb_corr_by_sample=perturb_corr_by_sample,
                sample=available_samples,
                html_share_path=html_share_path,
                top_n_programs=3,
                top_corr_genes=2,
                groupby="batch",
                significance_threshold=0.5,
                gene_name_key="symbol",
                control_target_name="non-targeting",
                umap_dot_size=4,
                subsample_frac=0.1,
            )
        return target_gene

    def test_share_index_written(self, ensure_one_gene_exported, html_share_path, target_gene):
        write_gene_share_index(html_share_path, [target_gene], {"test": "config", "k": 5})

        share = Path(html_share_path)
        assert (share / "shared" / "style.css").is_file()
        manifest = share / "shared" / "manifest.json"
        assert manifest.is_file()
        with open(manifest) as f:
            m = json.load(f)
        assert m["gene_list"] == [target_gene]
        idx = share / "index.html"
        assert idx.is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
