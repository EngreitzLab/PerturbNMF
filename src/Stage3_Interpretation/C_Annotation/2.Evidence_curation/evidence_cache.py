"""Per-pair gene-gene evidence cache.

paper-qa is slow, so we run it once per gene pair and store the ``classify_pair`` result as
one JSON file per pair (filename = sorted, upper-cased symbols, e.g. ``ACVR2B__SMAD2.json``).
The GPE augment step later reads these to attach literature evidence to existing edges.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def pair_key(gene_a: str, gene_b: str) -> str:
    """Order-independent key for a gene pair, e.g. ('SMAD2','ACVR2B') -> 'ACVR2B__SMAD2'."""
    a, b = sorted([gene_a.strip().upper(), gene_b.strip().upper()])
    return f"{a}__{b}"


def cache_path(evidence_dir, gene_a: str, gene_b: str) -> Path:
    return Path(evidence_dir) / f"{pair_key(gene_a, gene_b)}.json"


def write_result(evidence_dir, result: dict) -> Path:
    """Write a ``classify_pair`` result to the cache (keyed by its source/target symbols)."""
    path = cache_path(evidence_dir, result["source_symbol"], result["target_symbol"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def read_result(evidence_dir, gene_a: str, gene_b: str) -> Optional[dict]:
    """Return the cached result for a pair, or None if not cached / unreadable."""
    path = cache_path(evidence_dir, gene_a, gene_b)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
