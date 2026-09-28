"""02_fetch_ncbi_data.py runs the STRING regulator check on the screen's species, not mouse."""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "ProgramExplorer" / "src"


def load_fetch_module(monkeypatch):
    try:
        import requests  # noqa: F401
    except ImportError:  # the annotator env has no requests; nothing here touches the network
        stub = types.ModuleType("requests")
        stub.Response = stub.Session = object  # only used in annotations / never instantiated here
        monkeypatch.setitem(sys.modules, "requests", stub)
    monkeypatch.syspath_prepend(str(SRC))
    spec = importlib.util.spec_from_file_location("fetch_ncbi_data", SRC / "02_fetch_ncbi_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regulator_table():  # the shape load_regulator_data() returns
    return {1: pd.DataFrame({
        "grna_target": ["ACT1", "REP1"], "log_2_fold_change": [-3.0, 2.0],
        "p_value": [0.001, 0.001], "significant": [True, True],
    })}


def test_batch_path_passes_species(monkeypatch):
    module = load_fetch_module(monkeypatch)
    seen = {}

    def fake_batch(regulator_genes, program_genes, species, required_score):
        seen["species"] = species
        return {g: [] for g in regulator_genes}

    monkeypatch.setattr(module, "batch_validate_regulators", fake_batch)
    module.validate_program_regulators(1, regulator_table(), ["G1", "G2"], species=9606)
    assert seen["species"] == 9606


def test_single_query_path_passes_species(monkeypatch):
    module = load_fetch_module(monkeypatch)
    seen = []

    def fake_single(regulator, program_genes, species, required_score, top_n):
        seen.append(species)
        return {"n_interactions": 0, "interactions": []}

    monkeypatch.setattr(module, "get_regulator_program_interactions", fake_single)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    module.validate_program_regulators(1, regulator_table(), ["G1"], use_batch=False, species=10090)
    assert seen and set(seen) == {10090}


def test_default_species_is_human(monkeypatch):
    module = load_fetch_module(monkeypatch)
    seen = {}
    monkeypatch.setattr(module, "batch_validate_regulators",
                        lambda regulator_genes, program_genes, species, required_score: seen.setdefault("s", species) and {})
    module.validate_program_regulators(1, regulator_table(), ["G1"])
    assert seen["s"] == 9606
