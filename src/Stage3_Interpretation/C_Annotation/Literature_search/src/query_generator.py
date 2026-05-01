"""LLM-powered query generation from program context."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm_backend import LLMBackend

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


def _format_program_context(row: Dict[str, Any]) -> str:
    """Format an Excel row into a readable context block for the LLM."""
    parts = []
    parts.append(f"Program ID: {row.get('program_id', 'N/A')}")

    genes = row.get("top_genes", "")
    if genes:
        parts.append(f"Top co-regulated genes (ranked): {genes}")

    regulators = row.get("regulators", "")
    if regulators:
        parts.append(f"Regulators: {regulators}")

    day = row.get("day_most_active", "")
    if day:
        parts.append(f"Most active at: {day}")

    ct = row.get("cell_types", "")
    if ct:
        parts.append(f"Cell types: {ct}")

    go = row.get("go_enrichment", "")
    if go:
        parts.append(f"GO enrichment: {go}")

    other = row.get("other_enrichment", "")
    if other:
        parts.append(f"Other enrichment: {other}")

    return "\n".join(parts)


def generate_search_queries(
    llm: LLMBackend,
    program_row: Dict[str, Any],
    max_queries: int = 8,
    interaction_verbs: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Generate diverse PubMed queries for a single program.

    Returns a list of dicts with keys: query, strategy, target.
    """
    system_prompt = _load_prompt("query_generation.txt")
    context = _format_program_context(program_row)
    if interaction_verbs:
        context += f"\nInteraction verbs to use in queries: {', '.join(interaction_verbs)}"
    user_prompt = (
        f"Generate {max_queries} targeted PubMed search queries for the "
        f"following gene expression program:\n\n"
        f"{context}"
    )

    try:
        queries = llm.complete_json(system_prompt, user_prompt)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Failed to parse LLM query output: %s", exc)
        return _fallback_queries(program_row, interaction_verbs)

    if not isinstance(queries, list):
        logger.warning("LLM returned non-list; using fallback queries")
        return _fallback_queries(program_row, interaction_verbs)

    # Validate structure
    valid = []
    for q in queries:
        if isinstance(q, dict) and "query" in q:
            valid.append({
                "query": q["query"],
                "strategy": q.get("strategy", "unknown"),
                "target": q.get("target", ""),
            })
    if not valid:
        logger.warning("No valid queries from LLM; using fallback")
        return _fallback_queries(program_row, interaction_verbs)

    logger.info(
        "Generated %d queries for program %s",
        len(valid),
        program_row.get("program_id", "?"),
    )
    return valid[:max_queries]


def _fallback_queries(
    row: Dict[str, Any],
    interaction_verbs: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Deterministic fallback queries when LLM fails."""
    genes_str = row.get("top_genes", "")
    genes = [g.strip() for g in genes_str.split(",") if g.strip()][:10]
    queries = []

    if genes:
        # Broad gene OR query (top 10)
        gene_or = " OR ".join(genes[:10])
        queries.append({
            "query": f"({gene_or})",
            "strategy": "broad_context",
            "target": "papers mentioning any top genes",
        })

    # Gene pair queries for top 3 pairs
    for i in range(0, min(6, len(genes)), 2):
        if i + 1 < len(genes):
            queries.append({
                "query": f"({genes[i]} AND {genes[i+1]})",
                "strategy": "gene_pair",
                "target": f"co-occurrence of {genes[i]} and {genes[i+1]}",
            })

    # Regulator queries (single column, no direction)
    regs = row.get("regulators", "")
    reg_list = [r.strip() for r in regs.split(",") if r.strip()]
    cell_type = row.get("cell_types", "").split(",")[0].strip()

    for reg in reg_list[:3]:
        if genes:
            # Basic regulator + target genes query
            queries.append({
                "query": f"({reg} AND ({' OR '.join(genes[:5])}))",
                "strategy": "regulator_target",
                "target": f"regulator {reg} with target genes",
            })
            # Regulator + verb + context query (if verbs provided)
            if interaction_verbs and cell_type:
                verb = interaction_verbs[0]
                queries.append({
                    "query": f'("{reg} {verb}" AND {genes[0]} AND {cell_type})',
                    "strategy": "regulator_verb_context",
                    "target": f"{reg} {verb} {genes[0]} in {cell_type}",
                })

    return queries[:8]
