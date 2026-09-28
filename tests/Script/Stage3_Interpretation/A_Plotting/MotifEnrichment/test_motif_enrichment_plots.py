"""Smoke tests for Stage3_Interpretation/A_Plotting/src/motif_enrichment_plots.py.

Test strategy:

Dimensions:
  function:      plot_program_motif_ranks, plot_motif_program_heatmap, plot_candidate_tf_summary
  element_type:  promoter, enhancer (ranks plot covers both per call)
  data_state:    normal (has enriched + depleted rows), empty (no rows for element_type),
                 no_candidates (candidate_tfs table has no rows for the program)

Behavior to verify:
  - Each plotting function returns a matplotlib Figure and writes both PDF and PNG to disk
    (non-empty files), without raising.
  - plot_program_motif_ranks only ranks/plots motifs with enrichment > 1 by default (the paper's
    own significance definition excludes depleted motifs) — regression test for a bug where the
    smallest-FDR rows were dominated by depleted (enrichment=0) motifs.
  - An element_type with zero rows renders a "no data" placeholder instead of raising.
  - motif_source (--motif_source both): the ranks plot gets one row of panels per source (FIMO
    first), candidate outlines follow the source, the heatmap shows one source (default FIMO),
    candidate labels name the source; a one-valued motif_source column changes nothing.

Usage:
    eval "$(conda shell.bash hook)" && conda activate NMF_Benchmarking
    cd <repo root>
    python -m pytest tests/Script/Stage3_Interpretation/A_Plotting/MotifEnrichment/test_motif_enrichment_plots.py -v
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Imported by file path (not via the Stage3_Interpretation.A_Plotting.src package) so this test
# does not require the package's heavy __init__.py imports (muon/scanpy/mygene/...) — same
# workaround used by tests/Script/Stage2_Evaluation/test_motif_enrichment.py.
MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "src",
    "Stage3_Interpretation", "A_Plotting", "src", "motif_enrichment_plots.py",
)
spec = importlib.util.spec_from_file_location("motif_enrichment_plots", MODULE_PATH)
motif_enrichment_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motif_enrichment_plots)

plot_program_motif_ranks = motif_enrichment_plots.plot_program_motif_ranks
plot_motif_program_heatmap = motif_enrichment_plots.plot_motif_program_heatmap
plot_candidate_tf_summary = motif_enrichment_plots.plot_candidate_tf_summary
motif_panel_png_path = motif_enrichment_plots.motif_panel_png_path


def build_results_table():
    """Long results table for 3 programs x 2 element types x several TFs.

    Program 'P1' includes both strongly enriched TFs (enrichment > 1, tiny FDR) and strongly
    depleted TFs (enrichment == 0, tiny FDR) so ranking bugs that ignore enrichment direction
    are caught.
    """
    rows = []
    tfs_enriched = ["TF_A", "TF_B", "TF_C", "TF_D"]
    tfs_depleted = ["TF_DEP1", "TF_DEP2"]
    for program in ["P1", "P2", "P3"]:
        for element_type in ["promoter", "enhancer"]:
            for i, tf in enumerate(tfs_enriched):
                rows.append(dict(program=program, element_type=element_type, tf=tf,
                                  pvalue=10 ** (-(6 + i)), fdr=10 ** (-(5 + i)),
                                  enrichment=2.0 + i, n_program_genes_tested=250,
                                  mean_count_program=3.0, mean_count_background=1.0))
            for tf in tfs_depleted:
                # Very small FDR but enrichment < 1: must NOT be selected as "top" by the ranker.
                rows.append(dict(program=program, element_type=element_type, tf=tf,
                                  pvalue=1e-30, fdr=1e-28, enrichment=0.0,
                                  n_program_genes_tested=250,
                                  mean_count_program=0.0, mean_count_background=3.0))
    results = pd.DataFrame(rows)
    results["significant"] = (results["fdr"] < 0.05) & (results["enrichment"] > 1)
    return results


def build_candidate_tfs():
    return pd.DataFrame({
        "program": ["P1", "P1", "P2"],
        "tf": ["TF_A", "TF_B", "TF_A"],
        "element_type": ["promoter", "enhancer", "promoter"],
        "fdr": [1e-5, 1e-4, 1e-3],
        "enrichment": [3.0, 2.5, 2.0],
        "tf_in_top_program_genes": [True, False, True],
        "tf_program_loading_rank": [5, 120, 40],
        "tf_knockdown_regulates_program": [True, False, True],
        "knockdown_log2fc": [0.8, -0.1, 0.5],
        "knockdown_fdr": [0.001, 0.3, 0.02],
    })


@pytest.fixture(scope="module")
def results_table():
    return build_results_table()


@pytest.fixture(scope="module")
def candidate_tfs_table():
    return build_candidate_tfs()


@pytest.fixture(scope="session")
def plot_output_dir():
    outdir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "output",
        "Stage3_Interpretation", "Plotting", "MotifEnrichment",
    )
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def assert_pdf_and_png_written(save_path, save_name):
    pdf_path = os.path.join(save_path, f"{save_name}.pdf")
    png_path = os.path.join(save_path, f"{save_name}.png")
    for path in (pdf_path, png_path):
        assert os.path.exists(path), f"Expected plot output at {path}, but it was not written."
        assert os.path.getsize(path) > 0, f"Plot output {path} exists but is empty (0 bytes)."


def test_program_motif_ranks_only_shows_enriched_motifs(results_table, plot_output_dir):
    """Regression: ranking must exclude enrichment<=1 rows even when their FDR is tiny."""
    fig = plot_program_motif_ranks(
        results_table, "P1", candidate_tfs=None, top_n=10,
        save_path=plot_output_dir, save_name="test_program_P1_motif_ranks",
    )
    assert isinstance(fig, plt.Figure), f"Expected a matplotlib Figure, got {type(fig)}"
    labels = {t.get_text() for ax in fig.axes for t in ax.get_yticklabels()}
    assert "TF_DEP1" not in labels and "TF_DEP2" not in labels, (
        f"Depleted TFs (enrichment=0) appeared in the ranked plot: {labels}. "
        "plot_program_motif_ranks must filter to enrichment > 1 before ranking by FDR."
    )
    assert "TF_A" in labels, f"Expected enriched TF_A among plotted labels, got {labels}"
    assert_pdf_and_png_written(plot_output_dir, "test_program_P1_motif_ranks")
    plt.close(fig)


def test_program_motif_ranks_highlights_candidate_tfs(results_table, candidate_tfs_table, plot_output_dir):
    """Candidate TFs (from the candidate_tfs table) get a highlighted (non-default) bar edge."""
    fig = plot_program_motif_ranks(
        results_table, "P1", candidate_tfs=candidate_tfs_table, top_n=10,
        save_path=plot_output_dir, save_name="test_program_P1_motif_ranks_candidates",
    )
    promoter_ax = fig.axes[0]
    patches_by_label = dict(zip(
        [t.get_text() for t in promoter_ax.get_yticklabels()],
        promoter_ax.patches,
    ))
    assert "TF_A" in patches_by_label, "Expected candidate TF_A to be plotted in the promoter panel."
    edge = patches_by_label["TF_A"].get_edgecolor()
    assert edge[:3] != (1.0, 1.0, 1.0), f"Expected TF_A (a candidate TF) to have a non-default edge color, got {edge}"
    plt.close(fig)


@pytest.mark.parametrize("element_type", ["promoter", "enhancer"])
def test_program_motif_ranks_handles_missing_element_type(results_table, plot_output_dir, element_type):
    """An element_type with zero rows for the program renders a placeholder, not a crash."""
    sub = results_table[~((results_table["program"] == "P2") & (results_table["element_type"] == element_type))]
    fig = plot_program_motif_ranks(sub, "P2", save_path=plot_output_dir,
                                    save_name=f"test_program_P2_missing_{element_type}")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_motif_program_heatmap_writes_output_and_orders_columns(results_table, plot_output_dir):
    programs = ["P2", "P1", "P3"]
    fig = plot_motif_program_heatmap(
        results_table, programs=programs, element_type="promoter", top_n_per_program=3,
        save_path=plot_output_dir, save_name="test_heatmap_promoter",
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels == [f"P{p}" for p in programs], (
        f"Expected column order to follow the requested `programs` argument {programs}, got {xticklabels}"
    )
    assert_pdf_and_png_written(plot_output_dir, "test_heatmap_promoter")
    plt.close(fig)


def test_candidate_tf_summary_writes_output(candidate_tfs_table, plot_output_dir):
    fig = plot_candidate_tf_summary(
        candidate_tfs_table, "P1", save_path=plot_output_dir, save_name="test_candidate_summary_P1",
    )
    assert isinstance(fig, plt.Figure)
    assert_pdf_and_png_written(plot_output_dir, "test_candidate_summary_P1")
    plt.close(fig)


def test_candidate_tf_summary_handles_empty_program(candidate_tfs_table, plot_output_dir):
    """A program with no candidate TFs renders a placeholder, not a crash."""
    fig = plot_candidate_tf_summary(
        candidate_tfs_table, "P_NOT_PRESENT", save_path=plot_output_dir, save_name="test_candidate_summary_empty",
    )
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert not ax.axison, "Expected the placeholder axes to be turned off for a program with no candidate TFs."
    plt.close(fig)


def test_motif_panel_png_path_matches_html_report_image_convention():
    path = motif_panel_png_path("/tmp/report_out", 42)
    assert path == os.path.join("/tmp/report_out", "program_42", "images", "motif_ranks.png"), (
        f"motif_panel_png_path convention drifted from html_Program_QC_plots.py's "
        f"program_{{N}}/images/*.png layout: got {path}"
    )


def build_two_source_tables(results, candidates):
    finemo = results.copy()
    finemo["tf"] = finemo["tf"].str.replace("TF_", "FAM_")
    finemo["enrichment"] = finemo["enrichment"] * 0.5 + 0.6   # different values: pooling would show
    both = pd.concat([finemo.assign(motif_source="finemo"), results.assign(motif_source="fimo")], ignore_index=True)
    finemo_candidates = candidates.assign(tf=candidates["tf"].str.replace("TF_", "FAM_"), motif_source="finemo")
    return both, pd.concat([candidates.assign(motif_source="fimo"), finemo_candidates], ignore_index=True)


def test_program_motif_ranks_two_sources_one_row_each(results_table, candidate_tfs_table, plot_output_dir):
    both, candidates = build_two_source_tables(results_table, candidate_tfs_table)
    fig = plot_program_motif_ranks(both, "P1", candidate_tfs=candidates, save_path=plot_output_dir,
                                   save_name="test_program_P1_motif_ranks_two_sources")
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Promoter — FIMO", "Enhancer — FIMO", "Promoter — Fi-NeMo",
                      "Enhancer — Fi-NeMo"], titles
    fimo_labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    finemo_labels = {t.get_text() for t in fig.axes[2].get_yticklabels()}
    assert fimo_labels and all(tf.startswith("TF_") for tf in fimo_labels), fimo_labels
    assert finemo_labels and all(tf.startswith("FAM_") for tf in finemo_labels), finemo_labels
    edges = dict(zip([t.get_text() for t in fig.axes[2].get_yticklabels()], fig.axes[2].patches))
    assert edges["FAM_A"].get_linewidth() > 0 and edges["FAM_B"].get_linewidth() == 0, \
        "Fi-NeMo promoter candidate FAM_A outlined; FAM_B (enhancer candidate) not"
    assert_pdf_and_png_written(plot_output_dir, "test_program_P1_motif_ranks_two_sources")
    plt.close(fig)


def test_single_source_column_keeps_the_two_panel_layout(results_table):
    fig = plot_program_motif_ranks(results_table.assign(motif_source="fimo"), "P1")
    assert [ax.get_title() for ax in fig.axes] == ["Promoter", "Enhancer"]
    plt.close(fig)


def test_heatmap_shows_one_source(results_table, candidate_tfs_table):
    both, _ = build_two_source_tables(results_table, candidate_tfs_table)
    for source, prefix, label in (("fimo", "TF_", "FIMO"), ("finemo", "FAM_", "Fi-NeMo")):
        fig = plot_motif_program_heatmap(both, element_type="promoter", motif_source=source, cluster=False)
        rows = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert rows and all(tf.startswith(prefix) for tf in rows), f"{source}: {rows}"
        assert label in fig.axes[0].get_title()
        plt.close(fig)
    fig = plot_motif_program_heatmap(both, element_type="promoter", cluster=False)
    assert "FIMO" in fig.axes[0].get_title(), "default = first source (FIMO)"
    plt.close(fig)


def test_candidate_summary_labels_name_the_source(results_table, candidate_tfs_table):
    _, candidates = build_two_source_tables(results_table, candidate_tfs_table)
    fig = plot_candidate_tf_summary(candidates, "P1")
    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert {"TF_A (FIMO)", "FAM_A (Fi-NeMo)"} <= labels, labels
    plt.close(fig)
