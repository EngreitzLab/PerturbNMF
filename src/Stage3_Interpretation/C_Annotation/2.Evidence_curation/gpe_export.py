"""Attach cached paper-qa evidence onto an existing GPE contract, and validate it.

Imports GPE's own contract classes live from Tools/GeneProgramExplorer (``pipeline/src``) so the
emitted evidence matches the schema exactly. Attach-to-EXISTING-edges only: for each gene-gene
edge whose pair has a cached result, we append a ``literature`` entry to the edge's ``evidence[]``.
We never create new edges here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_cache import read_result  # noqa: E402

GPE_DIR = "/oak/stanford/groups/engreitz/Users/ymo/Tools/GeneProgramExplorer"
DEFAULT_GPE_SRC = f"{GPE_DIR}/pipeline/src"
DEFAULT_SCHEMA = f"{GPE_DIR}/contract/schema.json"


def _load_gpe(src_dir: str = DEFAULT_GPE_SRC):
    """Put the bundled GPE pipeline on sys.path and return the classes we need."""
    src_dir = str(Path(src_dir).resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from gene_program_explorer.literature_edges import CATEGORY_MAP  # noqa: E402
    from gene_program_explorer.model import Citation, Evidence  # noqa: E402
    return Evidence, Citation, CATEGORY_MAP


def _already_attached(edge: dict) -> bool:
    """True if this edge already carries a paper-qa literature entry (idempotent re-runs)."""
    return any(
        ev.get("kind") == "literature" and ev.get("sourceDb") == "paper-qa"
        for ev in (edge.get("evidence") or [])
    )


def attach_cached_evidence(contract: dict, evidence_dir, src_dir: str = DEFAULT_GPE_SRC) -> int:
    """Append cached literature evidence to existing gene-gene edges. Returns #edges updated."""
    Evidence, Citation, CATEGORY_MAP = _load_gpe(src_dir)

    symbol_by_id = {
        n["id"]: n.get("symbol")
        for n in contract.get("nodes", [])
        if n.get("kind") == "gene" and n.get("symbol")
    }

    updated = 0
    for edge in contract.get("edges", []):
        if (edge.get("endpointType") or "node-node") != "node-node":
            continue
        a, b = symbol_by_id.get(edge.get("source")), symbol_by_id.get(edge.get("target"))
        if not a or not b:
            continue
        if _already_attached(edge):
            continue
        result = read_result(evidence_dir, a, b)
        category = (result or {}).get("category")
        if not result or not category or category not in CATEGORY_MAP:
            continue

        _edge_type, label, _visible = CATEGORY_MAP[category]
        pmids = result.get("pmids") or []
        citation = Citation(
            pmid=pmids[0] if pmids else None,
            doi=result.get("doi"), title=result.get("title"),
            year=result.get("year"), url=result.get("url"),
        )
        has_cite = any([citation.pmid, citation.doi, citation.title, citation.url])
        evidence = Evidence(
            kind="literature", sourceDb="paper-qa",
            interactionType=label, excerpt=result.get("short_excerpt"),
            citation=citation if has_cite else None,
        )
        edge.setdefault("evidence", []).append(evidence.to_dict())
        updated += 1
    return updated


def validate(contract: dict, schema_path: str = DEFAULT_SCHEMA) -> None:
    """jsonschema validation + edge-endpoint referential integrity (raises on failure)."""
    import jsonschema

    schema = json.loads(Path(schema_path).read_text())
    jsonschema.validate(contract, schema)

    node_ids = {n["id"] for n in contract.get("nodes", [])}
    group_ids = {g["id"] for g in contract.get("groups", [])}
    valid = node_ids | group_ids
    for edge in contract.get("edges", []):
        for endpoint in (edge["source"], edge["target"]):
            if endpoint not in valid:
                raise ValueError(f"Edge {edge['id']} references unknown endpoint {endpoint}")
