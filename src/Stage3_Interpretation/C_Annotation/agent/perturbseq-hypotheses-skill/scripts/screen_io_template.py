#!/usr/bin/env python3
"""Shared loaders for one screen — template the Explorer adapts into `src/screen_io.py`.

Why: in past runs every Analyst re-read the explore docs and rewrote its own loaders, which was
the largest share of the run's cost. The Explorer writes these loaders once, after reading the
upstream pipeline (`data-hygiene.md` §1), so the significance column, effect column, sign
convention and technical-program exclusions are decided in one place. Analysts import them:

    import sys; sys.path.insert(0, "src")
    import screen_io as io
    effects = io.load_effects()                 # tidy: target, program, condition, log2fc, qvalue

Contract: every loader returns a tidy pandas DataFrame with exactly the columns in its docstring,
already restricted to the admitted inputs in `inputs/MANIFEST.txt`. Column names match
`plot_hypothesis_evidence.py`, so its plot functions take these tables unchanged.

After adapting, run `python src/screen_io.py --check`. It prints each table's shape and first row;
paste that output into `explore/README.md` under "Loaders", so Analysts need not open this file
to know what it returns. Delete a loader the run has no data for and say so in the README.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1]

# Decided by the Explorer from the run's own README / calibration report. Copy them into
# study.yaml :: significance as well; this module is where the code reads them.
EFFECT_COLUMN = ""          # e.g. "log2FC"
QVALUE_COLUMN = ""          # e.g. "adj_pval"
TECHNICAL_PROGRAMS: list[str] = []   # excluded from breadth, selectivity and neighbour denominators
NULL_TARGET_LABELS: list[str] = []   # e.g. ["non-targeting", "safe-targeting"]


def load_effects() -> pd.DataFrame:
    """target, program, condition, log2fc, qvalue — one row per target x program x condition.
    Real targets only (null-arm labels removed); promoter aliases collapsed per data-hygiene.md."""
    raise NotImplementedError


def load_null_effects() -> pd.DataFrame:
    """target, program, condition, log2fc, qvalue — the matched non-targeting / safe-targeting arm."""
    raise NotImplementedError


def load_guides() -> pd.DataFrame:
    """guide, target, program, condition, log2fc — per-guide effects, if the run has them."""
    raise NotImplementedError


def load_loadings() -> pd.DataFrame:
    """program, gene, loading — cNMF spectra, long format."""
    raise NotImplementedError


def load_correlation() -> pd.DataFrame:
    """Square DataFrame, index = columns = target: effect-vector correlation over non-technical
    programs (the matrix `perturbation_neighbours.py --matrix-out` writes)."""
    raise NotImplementedError


def load_eligibility() -> pd.DataFrame:
    """explore/regulator_eligibility.tsv as written by the Explorer."""
    return pd.read_csv(ANALYSIS_DIR / "explore/regulator_eligibility.tsv", sep="\t", comment="#")


def load_program_glossary() -> pd.DataFrame:
    """explore/program_glossary.tsv: program id -> short title and technical flag."""
    return pd.read_csv(ANALYSIS_DIR / "explore/program_glossary.tsv", sep="\t", comment="#")


LOADERS = [load_effects, load_null_effects, load_guides, load_loadings, load_correlation,
           load_eligibility, load_program_glossary]


def check_loaders() -> None:
    for loader in LOADERS:
        try:
            table = loader()
        except NotImplementedError:
            print(f"{loader.__name__}: not implemented")
            continue
        print(f"{loader.__name__}: {table.shape[0]} rows x {table.shape[1]} cols; columns={list(table.columns)[:8]}")
        if len(table):
            print(f"  first row: {table.iloc[0].to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="print each loader's shape and first row")
    if parser.parse_args().check:
        check_loaders()
