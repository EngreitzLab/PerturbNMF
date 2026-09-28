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

import pandas as pd
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
    build_motifs,
    motif_section_html,
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

    def test_no_panel_without_motifs(self, exported_program):
        prog_dir, pid, _ = exported_program
        assert "motif-row" not in (prog_dir / f"program_{pid}.html").read_text()
        assert not (prog_dir / "data" / "motifs.json").exists()

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



# ---------------------------------------------------------------------------
# Optional TF-motif panel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def motif_tables(test_mdata):
    """Synthetic Stage 2 motif / candidate-TF tables for the first program."""
    target = str(test_mdata["cNMF"].var_names[0])
    motif_results = pd.DataFrame({
        "program": target,
        "element_type": ["promoter", "promoter", "promoter", "enhancer"],
        "tf": ["KLF4", "SOX2", "ZFX", "ETS1"],
        "fdr": [1e-6, 1e-3, 1e-8, 1e-2],
        "enrichment": [2.0, 1.5, 0.2, 1.3],
        "significant": [True, True, False, True],
    })
    candidate_tfs = pd.DataFrame({
        "program": [target], "element_type": ["promoter"], "tf": ["KLF4"], "tf_gene_symbol": ["KLF4"],
        "fdr": [1e-6], "knockdown_log2fc": [-0.5], "evidence_tier": ["motif+regulator"],
    })
    return target, motif_results, candidate_tfs


def test_motif_panel_written_when_motifs_given(test_mdata, perturb_path_base, go_path, program_correlation,
                                               waterfall_correlation, perturbed_gene_found, available_samples,
                                               motif_tables, tmp_path_factory):
    target, motif_results, candidate_tfs = motif_tables
    share = tmp_path_factory.mktemp("html_share_motif")
    export_program_html(
        mdata=test_mdata, perturb_path_base=perturb_path_base, GO_path=go_path, file_to_dictionary=None,
        Target_Program=target, program_correlation=program_correlation,
        waterfall_correlation=waterfall_correlation, sample=available_samples,
        perturbed_gene_found=perturbed_gene_found, html_share_path=str(share), groupby="batch",
        gene_name_key="symbol", subsample_frac=0.1,
        motif_results=motif_results, candidate_tfs=candidate_tfs,
    )
    prog_dir = share / f"program_{target}"
    png = prog_dir / "images" / "motif_ranks.png"
    assert png.is_file() and png.stat().st_size > 0, f"Missing or empty motif panel: {png}"
    motifs = json.loads((prog_dir / "data" / "motifs.json").read_text())
    promoter = [m["tf"] for m in motifs["top"]["promoter"]]
    assert promoter == ["KLF4", "SOX2"], f"Expected significant promoter motifs by FDR (ZFX depleted), got {promoter}"
    assert motifs["candidates"][0]["evidence_tier"] == "motif+regulator", motifs["candidates"]
    html = (prog_dir / f"program_{target}.html").read_text()
    assert "TF motif enrichment" in html and "images/motif_ranks.png" in html, "Motif section missing from page"
    meta = json.loads((prog_dir / "metadata.json").read_text())
    assert meta["top_motifs"]["enhancer"][0]["tf"] == "ETS1", meta["top_motifs"]


def test_build_motifs_keeps_motif_sources_apart():
    fimo = pd.DataFrame({"program": "3", "element_type": ["promoter", "promoter", "enhancer"],
                         "tf": ["KLF4", "SOX2", "ETS1"], "fdr": [1e-6, 1e-3, 1e-2],
                         "enrichment": [2.0, 1.5, 1.3], "significant": True})
    finemo = pd.DataFrame({"program": "3", "element_type": ["enhancer", "enhancer"], "tf": ["ETS", "GATA"],
                           "fdr": [1e-9, 1e-4], "enrichment": [1.6, 1.4], "significant": True})
    motif_results = pd.concat([finemo.assign(motif_source="finemo"), fimo.assign(motif_source="fimo")])
    candidates = pd.DataFrame({"program": "3", "element_type": ["promoter", "enhancer", "promoter"],
                               "tf": ["KLF4", "GATA", "KLF-SP"], "tf_gene_symbol": ["KLF4", "GATA2", "KLF4"],
                               "fdr": [1e-6, 1e-4, 1e-5], "knockdown_log2fc": [None, None, None],
                               "evidence_tier": ["motif+expressed", "motif+expressed", "motif+expressed"],
                               "motif_source": ["fimo", "finemo", "finemo"]})
    motifs = build_motifs(motif_results, candidates, 3)
    assert list(motifs["n_significant"]) == ["promoter_fimo", "enhancer_fimo", "promoter_finemo", "enhancer_finemo"]
    assert motifs["n_significant"] == {"promoter_fimo": 2, "enhancer_fimo": 1, "promoter_finemo": 0,
                                       "enhancer_finemo": 2}, motifs["n_significant"]
    assert [m["tf"] for m in motifs["top"]["enhancer_finemo"]] == ["ETS", "GATA"]
    assert [(c["tf"], c["motif_source"]) for c in motifs["candidates"]] == [
        ("KLF4", "fimo"), ("KLF4", "finemo"), ("GATA2", "finemo")], motifs["candidates"]
    html = motif_section_html(motifs)
    assert "2 promoter FIMO, 1 enhancer FIMO, 0 promoter Fi-NeMo, 2 enhancer Fi-NeMo" in html
    assert "enhancer (Fi-NeMo)" in html

    one_source = build_motifs(fimo.assign(motif_source="fimo"), candidates[candidates["motif_source"] == "fimo"], 3)
    assert one_source == build_motifs(fimo, candidates[candidates["motif_source"] == "fimo"].drop(
        columns="motif_source"), 3), "one-valued motif_source column must change nothing"
    assert list(one_source["n_significant"]) == ["promoter", "enhancer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
