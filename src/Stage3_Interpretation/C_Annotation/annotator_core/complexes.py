"""Curated protein complexes (CORUM, ComplexPortal, SIGNOR) from the OmniPath complexes table.

Same source and the same "curated" rule as GeneProgramExplorer (`omnipath_client.py`,
`is_curated_complex`): a complex is kept when a hand-curated database lists it, so large real
complexes (Mediator, TFIID) survive while bulk high-throughput records (hu.MAP, Compleat) do not.

The table (https://omnipathdb.org/complexes?format=tsv, ~6 MB) is downloaded once to
`--complexes` and read from disk after that.
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

OMNIPATH_COMPLEXES_URL = "https://omnipathdb.org/complexes?format=tsv"
CURATED_SOURCES = ("CORUM", "ComplexPortal", "SIGNOR")


@dataclass
class Complex:
    name: str
    members: List[str]
    sources: List[str]
    pmids: List[str]
    identifiers: List[str] = field(default_factory=list)

    @property
    def complex_id(self) -> str:
        """The first curated identifier (CORUM:123, ComplexPortal:CPX-1, SIGNOR:SIGNOR-C1)."""
        for prefix in CURATED_SOURCES:
            for identifier in self.identifiers:
                if identifier.lower().startswith(prefix.lower()):
                    return identifier
        return self.name


def fetch_complex_table(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OMNIPATH_COMPLEXES_URL, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def load_curated_complexes(path: Path, curated_sources=CURATED_SOURCES) -> List[Complex]:
    table = pd.read_csv(fetch_complex_table(path), sep="\t", dtype=str).fillna("")
    curated = {s.lower() for s in curated_sources}
    complexes, seen = [], set()
    for row in table.itertuples(index=False):
        sources = [s for s in row.sources.split(";") if s]
        if not any(s.lower() in curated for s in sources):
            continue
        members = sorted({m for m in row.components_genesymbols.split("_") if m})
        key = tuple(members)
        if len(members) < 2 or key in seen:  # OmniPath repeats a complex once per source record
            continue
        seen.add(key)
        complexes.append(Complex(
            name=row.name or "+".join(members[:4]),
            members=members,
            sources=sources,
            pmids=[p for p in re.split(r"[;,]", row.references) if p.strip().isdigit()],
            identifiers=[i for i in row.identifiers.split(";") if i],
        ))
    return complexes


def complexes_by_gene(complexes: List[Complex]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, entry in enumerate(complexes):
        for gene in entry.members:
            index.setdefault(gene, []).append(i)
    return index
