"""Tests for running the pipeline on non-perturbed (no guide) and non-human data.

Covers the code paths that previously assumed CRISPR guide information or human data:

- Stage 2 evaluation skips trait (GWAS) enrichment for non-human species and
  validates --species.
- Stage 3 K-selection runs without trait or perturbation files (--no_trait,
  --no_perturbation).
- Stage 3 Excel summary runs on an h5mu without uns['guide_targets'] and, for
  mouse data, annotates genes with their human orthologs' summaries.

All inputs are small synthetic files; no network access or inference output is needed.

Usage (from the repo root, in the pipeline environment):
    python -m pytest tests/Script/test_non_perturbed.py -v
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SRC = PIPELINE_ROOT / "src"
sys.path.insert(0, str(SRC))

EVAL_SCRIPT = (
    SRC / "Stage2_Evaluation/A_Metrics/Slurm_Version/cNMF_evaluation_pipeline.py"
)
K_SELECTION_SCRIPT = (
    SRC / "Stage3_Interpretation/A_Plotting/Slurm_Version/cNMF_k_selection.py"
)
EXCEL_SCRIPT = (
    SRC / "Stage3_Interpretation/B_Summarization/Slurm_Version/cNMF_excel_summary.py"
)

K_VALUES = [3, 5]
SEL_THRESH = 2.0
SEL_STR = "2_0"
RUN_NAME = "non_perturbed_run"
N_CELLS = 40
N_GENES = 25
SEED = 0


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_mdata(k, rng):
    """MuData with 'rna' and 'cNMF' modalities and no guide information."""
    cells = [f"cell{i}" for i in range(N_CELLS)]
    genes = [f"Gene{i}" for i in range(N_GENES)]  # mouse-style capitalization
    rna = ad.AnnData(
        X=rng.poisson(2.0, size=(N_CELLS, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(
            {"sample": pd.Categorical(rng.choice(["ctrl", "treated"], N_CELLS))},
            index=cells,
        ),
        # Symbols as var_names and no 'symbol' column, as in Seurat-converted data
        var=pd.DataFrame(index=genes),
    )
    prog = ad.AnnData(
        X=rng.random((N_CELLS, k)).astype(np.float32),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=[str(p) for p in range(k)]),
    )
    prog.varm["loadings"] = rng.random((k, N_GENES)).astype(np.float32)
    prog.uns["var_names"] = np.array(genes)
    return md.MuData({"rna": rna, "cNMF": prog})


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Synthetic run directory: inference h5mu files plus Stage 2 outputs without
    trait or perturbation results, as produced for non-perturbed mouse data."""
    rng = np.random.default_rng(SEED)
    out_dir = tmp_path_factory.mktemp("out")
    run = out_dir / RUN_NAME
    adata_dir = run / "Inference" / "adata"
    adata_dir.mkdir(parents=True)

    stability = []
    for k in K_VALUES:
        _make_mdata(k, rng).write(adata_dir / f"cNMF_{k}_{SEL_STR}.h5mu")

        eval_dir = run / "Evaluation" / f"{k}_{SEL_STR}"
        eval_dir.mkdir(parents=True)
        programs = [str(p) for p in range(k)]
        for fname in [f"{k}_GO_term_enrichment.txt", f"{k}_geneset_enrichment.txt"]:
            pd.DataFrame(
                {
                    "Term": [f"TERM_{p}" for p in programs],
                    "Adjusted P-value": rng.uniform(0, 0.1, k),
                    "Genes": ["Gene0;Gene1"] * k,
                    "program_name": programs,
                }
            ).to_csv(eval_dir / fname, sep="\t", index=False)
        pd.DataFrame({"Total": [rng.uniform(0.2, 0.6)]}, index=["all"]).to_csv(
            eval_dir / f"{k}_Explained_Variance_Summary.txt", sep="\t"
        )
        pd.DataFrame(
            {
                "VarianceExplained": rng.uniform(0, 0.1, k),
                "ProgramID": programs,
                "program_name": programs,
            }
        ).to_csv(eval_dir / f"{k}_Explained_Variance.txt", sep="\t", index=False)
        stability.append(
            {
                "k": k,
                "silhouette": rng.uniform(0.5, 1.0),
                "prediction_error": rng.uniform(1, 2),
            }
        )

    stability_file = run / "Inference" / "k_selection_stats.tsv"
    pd.DataFrame(stability).to_csv(stability_file, sep="\t", index=False)
    return {"out_dir": out_dir, "run": run, "stability_file": stability_file}


# Stage 2: evaluation


@pytest.fixture(scope="module")
def evaluation_main():
    return _load_module(EVAL_SCRIPT, "cNMF_evaluation_pipeline").main


def _eval_argv(run_dir, *extra):
    return [
        str(EVAL_SCRIPT),
        "--out_dir",
        str(run_dir["out_dir"]),
        "--run_name",
        RUN_NAME,
        "--K",
        str(K_VALUES[0]),
        "--sel_threshs",
        str(SEL_THRESH),
        *extra,
    ]


def test_trait_enrichment_skipped_for_mouse(
    run_dir, evaluation_main, monkeypatch, capsys
):
    """--Perform_trait with a non-human organism is skipped with a message,
    without requiring --gwas_data_path."""
    monkeypatch.setattr(
        sys, "argv", _eval_argv(run_dir, "--Perform_trait", "--species", "mouse")
    )
    assert evaluation_main() == 0

    assert "skipping trait (GWAS) enrichment" in capsys.readouterr().out
    eval_dir = run_dir["run"] / "Evaluation" / f"{K_VALUES[0]}_{SEL_STR}"
    assert not (eval_dir / f"{K_VALUES[0]}_trait_enrichment.txt").exists()


def test_trait_enrichment_still_requires_gwas_for_human(
    run_dir, evaluation_main, monkeypatch
):
    """Human behavior is unchanged: --Perform_trait needs --gwas_data_path."""
    monkeypatch.setattr(
        sys, "argv", _eval_argv(run_dir, "--Perform_trait", "--species", "human")
    )
    with pytest.raises(SystemExit):
        evaluation_main()


def test_organism_is_case_insensitive(run_dir, evaluation_main, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", _eval_argv(run_dir, "--Perform_trait", "--species", "Mouse")
    )
    assert evaluation_main() == 0
    assert "skipping trait (GWAS) enrichment" in capsys.readouterr().out


def test_unsupported_organism_is_rejected(run_dir, evaluation_main, monkeypatch):
    monkeypatch.setattr(sys, "argv", _eval_argv(run_dir, "--species", "rat"))
    with pytest.raises(SystemExit):
        evaluation_main()


# Stage 3: K-selection plotting


def test_k_selection_without_trait_and_perturbation(run_dir, tmp_path):
    save_dir = tmp_path / "K_selection"
    cmd = [
        sys.executable,
        str(K_SELECTION_SCRIPT),
        "--output_directory",
        str(run_dir["out_dir"]),
        "--run_name",
        RUN_NAME,
        "--save_folder_name",
        str(save_dir),
        "--eval_folder_name",
        str(run_dir["run"] / "Evaluation"),
        "--K",
        *[str(k) for k in K_VALUES],
        "--sel_threshs",
        str(SEL_THRESH),
        "--stability_file",
        str(run_dir["stability_file"]),
        "--no_trait",
        "--no_perturbation",
    ]
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    produced = {p.name for p in save_dir.iterdir()}
    assert f"K-selection_panel_{SEL_THRESH}.png" in produced
    assert f"Enrichment_{SEL_THRESH}_go_terms.png" in produced
    assert f"Enrichment_{SEL_THRESH}_traits.png" not in produced
    assert not any(name.startswith("Perturbation_") for name in produced)


def test_k_selection_requires_files_by_default(run_dir, tmp_path):
    """Without the new flags, missing trait/perturbation files are still an error."""
    cmd = [
        sys.executable,
        str(K_SELECTION_SCRIPT),
        "--output_directory",
        str(run_dir["out_dir"]),
        "--run_name",
        RUN_NAME,
        "--save_folder_name",
        str(tmp_path / "K_selection"),
        "--eval_folder_name",
        str(run_dir["run"] / "Evaluation"),
        "--K",
        *[str(k) for k in K_VALUES],
        "--sel_threshs",
        str(SEL_THRESH),
        "--Conditions",
        "ctrl",
        "--stability_file",
        str(run_dir["stability_file"]),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "required evaluation file(s) not found" in result.stdout


def test_k_selection_panel_hides_regulator_row(tmp_path):
    from Stage3_Interpretation.A_Plotting.src import k_selection_plots as ksp

    stats = pd.DataFrame(
        {"k": K_VALUES, "silhouette": [0.8, 0.7], "prediction_error": [1.5, 1.2]}
    )
    counts = pd.DataFrame(
        {"go_terms": [3, 4], "genesets": [2, 3], "traits": [0, 0]}, index=K_VALUES
    )
    variance = {3: 0.3, 5: 0.4}
    figures = []
    original_subplots = ksp.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        figures.append(axes)
        return fig, axes

    ksp.plt.subplots = capture_subplots
    try:
        for panel_fn in [
            ksp.plot_k_selection_panel,
            ksp.plot_k_selection_panel_no_traits,
        ]:
            panel_fn(
                stats, counts, None, variance, folder_name=str(tmp_path), file_name="p"
            )
    finally:
        ksp.plt.subplots = original_subplots

    for axes in figures:
        assert not axes[2, 0].get_visible()
        assert not axes[2, 1].get_visible()


# Stage 3: Excel summary


class _FakeMyGene:
    """Offline stand-in for mygene.MyGeneInfo that records the species queried."""

    species_seen: ClassVar[list] = []

    def querymany(self, genes, species=None, **kwargs):
        _FakeMyGene.species_seen.append(species)
        return [{"query": g, "summary": f"summary of {g}"} for g in genes]


def test_excel_summary_without_guides(run_dir, monkeypatch, capsys, tmp_path):
    """Non-perturbed mouse data: no Targets Summary sheet, and gene summaries come
    from human orthologs (synthetic table: even-numbered genes have an ortholog)."""
    excel = _load_module(EXCEL_SCRIPT, "cNMF_excel_summary")
    from Stage3_Interpretation.B_Summarization.src import Compile_excel_sheet

    monkeypatch.setattr(Compile_excel_sheet, "MyGeneInfo", _FakeMyGene)
    _FakeMyGene.species_seen.clear()
    ortholog_table = tmp_path / "orthologs.tsv"
    pd.DataFrame(
        {
            "mouse_symbol": [f"Gene{i}" for i in range(0, N_GENES, 2)],
            "mouse_ensembl": pd.NA,
            "human_symbol": [f"HSGENE{i}" for i in range(0, N_GENES, 2)],
            "source": "test",
        }
    ).to_csv(ortholog_table, sep="\t", index=False)

    k = K_VALUES[1]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXCEL_SCRIPT),
            "--out_dir",
            str(run_dir["out_dir"]),
            "--run_name",
            RUN_NAME,
            "--K",
            str(k),
            "--sel_thresh",
            str(SEL_THRESH),
            "--species",
            "mouse",
            "--categorical_key",
            "sample",
            "--num_gene",
            "5",
            "--ortholog_table",
            str(ortholog_table),
        ],
    )
    assert excel.main() == 0

    assert "skipping the Targets Summary sheet" in capsys.readouterr().out
    assert set(_FakeMyGene.species_seen) == {"human"}
    save_path = run_dir["run"] / "Interpretation" / "Summary_table" / f"{k}_{SEL_STR}"
    sheets = pd.ExcelFile(save_path / f"cNMF_{k}_{SEL_STR}.xlsx").sheet_names
    assert "Summary" in sheets and "Program Loadings" in sheets
    assert "Targets Summary" not in sheets
    assert not (save_path / f"Targets_Summary_{k}_{SEL_STR}.tsv").exists()

    loadings = pd.read_csv(
        save_path / f"Program_Loadings_{k}_{SEL_STR}.tsv",
        sep="\t",
        keep_default_na=False,  # keep the literal "N/A"
    )
    for _, row in loadings.iterrows():
        n = int(row["Gene"].removeprefix("Gene"))
        if n % 2 == 0:
            assert row["Human ortholog"] == f"HSGENE{n}"
            assert row["Annotation"] == f"HSGENE{n}: summary of HSGENE{n}"
        else:
            assert row["Human ortholog"] == "" and row["Annotation"] == "N/A"
