"""Unit tests for mouse to human ortholog mapping (Stage2 orthologs.py).

Uses a small synthetic ortholog table, so no network access is needed.

Usage (from the repo root):
    python -m pytest tests/Script/Stage2_Evaluation/test_orthologs.py -v
"""

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest

from Stage2_Evaluation.A_Metrics.src.enrichment_geneset import (
    get_program_gene_loadings,
)
from Stage2_Evaluation.A_Metrics.src.orthologs import (
    TABLE_COLUMNS,
    collapse_to_human,
    load_ortholog_table,
    map_mouse_to_human,
)

# mouse_symbol, mouse_ensembl, human_symbol, source
ORTHOLOGS = [
    ("Cdh5", "ENSMUSG00000031871", "CDH5", "Ensembl"),  # same name, one-to-one
    ("Cdh5", np.nan, "CDH5", "MGI"),  # duplicate from the second source
    ("Zfp451", np.nan, "ZNF451", "MGI"),  # different name
    ("H2-Ab1", np.nan, "HLA-DQB1", "Ensembl"),  # one-to-many
    ("H2-Ab1", np.nan, "HLA-DQB2", "Ensembl"),
    ("Cd59a", np.nan, "CD59", "Ensembl"),  # many-to-one
    ("Cd59b", np.nan, "CD59", "Ensembl"),
    ("Gm12345", "ENSMUSG00000099999", np.nan, "Ensembl"),  # no human ortholog
]
# Upper-cases to an existing human symbol but is not its ortholog in the table.
CASE_DECOY = "Ccdc115"


@pytest.fixture
def table():
    return pd.DataFrame(ORTHOLOGS, columns=TABLE_COLUMNS)


@pytest.fixture
def table_path(table, tmp_path):
    path = tmp_path / "orthologs.tsv"
    table.to_csv(path, sep="\t", index=False)
    return str(path)


def test_map_mouse_to_human(table):
    mapping = map_mouse_to_human(
        ["Cdh5", "Zfp451", "H2-Ab1", "Cd59a", "Gm12345", CASE_DECOY], table
    )
    assert mapping == {
        "Cdh5": ["CDH5"],
        "Zfp451": ["ZNF451"],
        "H2-Ab1": ["HLA-DQB1", "HLA-DQB2"],
        "Cd59a": ["CD59"],
        "Gm12345": [],
        CASE_DECOY: [],  # no fallback to upper-casing
    }


def test_map_ensembl_ids(table):
    mapping = map_mouse_to_human(
        ["ENSMUSG00000031871", "ENSMUSG00000031871.7", "ENSMUSG00000099999"], table
    )
    assert mapping["ENSMUSG00000031871"] == ["CDH5"]
    assert mapping["ENSMUSG00000031871.7"] == ["CDH5"]  # version suffix ignored
    assert mapping["ENSMUSG00000099999"] == []


def test_collapse_to_human(table):
    loadings = pd.DataFrame(
        {"p0": [5.0, 1.0, 2.0, 3.0, 4.0, 9.0], "p1": [0.5, 0.1, 0.2, 0.3, 0.6, 0.9]},
        index=["Cdh5", "Zfp451", "H2-Ab1", "Cd59a", "Cd59b", "Gm12345"],
    )
    human = collapse_to_human(loadings, table)

    assert set(human.index) == {"CDH5", "ZNF451", "HLA-DQB1", "HLA-DQB2", "CD59"}
    # one-to-many: both human orthologs take the mouse loading
    assert human.loc["HLA-DQB1", "p0"] == human.loc["HLA-DQB2", "p0"] == 2.0
    # many-to-one: per-program maximum over the mouse orthologs
    assert human.loc["CD59", "p0"] == 4.0 and human.loc["CD59", "p1"] == 0.6
    assert human.loc["ZNF451", "p1"] == 0.1


def test_collapse_without_any_ortholog_fails(table):
    with pytest.raises(ValueError, match="No mouse genes"):
        collapse_to_human(pd.DataFrame({"p0": [1.0]}, index=["Gm12345"]), table)


def test_load_ortholog_table_validates_columns(tmp_path):
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"mouse": ["Cdh5"], "human": ["CDH5"]}).to_csv(
        bad, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        load_ortholog_table(str(bad))


def _mdata(genes):
    rng = np.random.default_rng(0)
    rna = ad.AnnData(
        X=rng.random((6, len(genes))).astype(np.float32),
        var=pd.DataFrame(index=genes),
    )
    prog = ad.AnnData(
        X=rng.random((6, 2)).astype(np.float32), var=pd.DataFrame(index=["0", "1"])
    )
    prog.varm["loadings"] = rng.random((2, len(genes))).astype(np.float32)
    prog.uns["var_names"] = np.array(genes)
    return md.MuData({"rna": rna, "cNMF": prog})


def test_program_loadings_mouse_uses_orthologs(table_path):
    genes = ["Cdh5", "Zfp451", "H2-Ab1", "Gm12345", CASE_DECOY]
    loadings = get_program_gene_loadings(
        _mdata(genes),
        prog_key="cNMF",
        data_key="rna",
        organism="mouse",
        ortholog_table=table_path,
    )
    assert set(loadings.index) == {"CDH5", "ZNF451", "HLA-DQB1", "HLA-DQB2"}
    assert list(loadings.columns) == ["0", "1"]


def test_program_loadings_mouse_native_symbols(table_path):
    """map_to_human=False (MSigDB mouse gene sets) keeps mouse symbols unchanged."""
    genes = ["Cdh5", "Zfp451"]
    loadings = get_program_gene_loadings(
        _mdata(genes),
        prog_key="cNMF",
        data_key="rna",
        organism="mouse",
        ortholog_table=table_path,
        map_to_human=False,
    )
    assert list(loadings.index) == genes


def test_program_loadings_human_unchanged():
    genes = ["CDH5", "znf451"]
    loadings = get_program_gene_loadings(
        _mdata(genes), prog_key="cNMF", data_key="rna", organism="human"
    )
    assert list(loadings.index) == ["CDH5", "ZNF451"]


def test_unsupported_organism():
    with pytest.raises(ValueError, match="Unsupported organism"):
        get_program_gene_loadings(
            _mdata(["Cdh5"]), prog_key="cNMF", data_key="rna", organism="rat"
        )


# Download handling (network calls are replaced by stubs)

ENSEMBL_HEADER = "Gene stable ID\tGene name\tHuman gene name\n"
ENSEMBL_ROW = "ENSMUSG00000031871\tCdh5\tCDH5\n"


def test_ensembl_requires_completion_stamp(monkeypatch):
    """A response cut short by BioMart (no '[success]' line) is rejected."""
    from Stage2_Evaluation.A_Metrics.src import orthologs

    monkeypatch.setattr(
        orthologs, "_download", lambda url: ENSEMBL_HEADER + ENSEMBL_ROW
    )
    with pytest.raises(RuntimeError, match="completion stamp"):
        orthologs._fetch_ensembl()

    monkeypatch.setattr(
        orthologs, "_download", lambda url: ENSEMBL_HEADER + ENSEMBL_ROW + "[success]\n"
    )
    df = orthologs._fetch_ensembl()
    assert df.to_dict("records") == [
        {
            "mouse_symbol": "Cdh5",
            "mouse_ensembl": "ENSMUSG00000031871",
            "human_symbol": "CDH5",
            "source": "Ensembl",
        }
    ]


def test_download_failure_suggests_ortholog_table(monkeypatch, tmp_path):
    from Stage2_Evaluation.A_Metrics.src import orthologs

    def offline():
        raise OSError("network unreachable")

    monkeypatch.setattr(orthologs, "DEFAULT_TABLE_PATH", str(tmp_path / "cache.tsv"))
    monkeypatch.setattr(orthologs, "build_ortholog_table", offline)
    with pytest.raises(RuntimeError, match="--ortholog_table"):
        orthologs.load_ortholog_table()


def test_cache_is_written_atomically(monkeypatch, tmp_path, table):
    """The cache appears in one step (no partial file) and is reused afterwards."""
    from Stage2_Evaluation.A_Metrics.src import orthologs

    cache = tmp_path / "cache.tsv"
    monkeypatch.setattr(orthologs, "RESOURCES_DIR", str(tmp_path))
    monkeypatch.setattr(orthologs, "DEFAULT_TABLE_PATH", str(cache))
    monkeypatch.setattr(orthologs, "build_ortholog_table", lambda: table)
    replaced = []
    real_replace = orthologs.os.replace

    def record_replace(src, dst):
        assert not cache.exists()  # nothing written to the final path before the move
        replaced.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(orthologs.os, "replace", record_replace)
    first = orthologs.load_ortholog_table()
    assert replaced and replaced[0][1] == str(cache)
    assert list(tmp_path.iterdir()) == [cache]  # temporary file is gone

    monkeypatch.setattr(
        orthologs, "build_ortholog_table", lambda: pytest.fail("rebuilt")
    )
    second = orthologs.load_ortholog_table()
    assert len(second) == len(first)


def test_unwritable_cache_keeps_downloaded_table(monkeypatch, tmp_path, table, caplog):
    """A cache that cannot be written (e.g. read-only Resources/) does not fail the run."""
    from Stage2_Evaluation.A_Metrics.src import orthologs

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")  # makedirs under a regular file fails, even as root
    monkeypatch.setattr(orthologs, "RESOURCES_DIR", str(blocker / "Resources"))
    monkeypatch.setattr(
        orthologs, "DEFAULT_TABLE_PATH", str(blocker / "Resources" / "cache.tsv")
    )
    monkeypatch.setattr(orthologs, "build_ortholog_table", lambda: table)

    loaded = orthologs.load_ortholog_table()
    assert len(loaded) == len(table)
    assert "Could not cache the ortholog table" in caplog.text
    assert list(tmp_path.iterdir()) == [blocker]  # no temporary file left behind
