#!/usr/bin/env python3
"""CLI entry point for the Literature Search Agent.

Usage:
    python run_literature_search.py \
        --excel programs.xlsx \
        --output-dir ./output \
        --interactions "regulates,induces,promotes,inhibits" \
        --domain-keywords "angiogenesis,proliferation,migration"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add Annotation/ to sys.path for package imports
_ANNOTATION_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _ANNOTATION_DIR not in sys.path:
    sys.path.insert(0, _ANNOTATION_DIR)

from Literature_search.src.llm_backend import LLMBackend
from Literature_search.src.query_generator import generate_search_queries
from Literature_search.src.search_engine import SearchEngine
from Literature_search.src.paper_fetcher import PaperFetcher, PaperData, summarize_paper
from Literature_search.src.verification import Verifier, VerificationResult
from Literature_search.src.output_writer import (
    write_program_json, write_program_markdown, write_program_html,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Excel loading
# ------------------------------------------------------------------

def _match_column(df_columns: List[str], target: str) -> Optional[str]:
    """Case-insensitive column matching."""
    target_lower = target.lower().replace(" ", "_")
    for col in df_columns:
        if col.lower().replace(" ", "_") == target_lower:
            return col
    return None


COLUMN_FIELDS = {
    "program_id": "program_id",
    "top_genes": "top_genes",
    "regulators": "regulators",
    "day_most_active": "day_most_active",
    "cell_types": "cell_types",
    "go_enrichment": "go_enrichment",
    "other_enrichment": "other_enrichment",
}


def load_excel(excel_path: Path, programs: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Load program rows from Excel."""
    df = pd.read_excel(excel_path)
    logger.info("Loaded Excel: %d rows, %d columns", len(df), len(df.columns))

    # Map column names (case-insensitive)
    col_map = {}
    for field_key, col_name in COLUMN_FIELDS.items():
        actual = _match_column(list(df.columns), col_name)
        if actual:
            col_map[field_key] = actual
        else:
            logger.warning("Column '%s' not found in Excel", col_name)

    rows = []
    for _, excel_row in df.iterrows():
        row = {}
        for field_key, actual_col in col_map.items():
            val = excel_row.get(actual_col, "")
            row[field_key] = str(val) if pd.notna(val) else ""
        rows.append(row)

    # Filter to selected programs
    if programs is not None:
        selected = set(str(p) for p in programs)
        rows = [r for r in rows if r.get("program_id", "") in selected]
        logger.info("Filtered to %d selected programs", len(rows))

    return rows


# ------------------------------------------------------------------
# Per-program processing
# ------------------------------------------------------------------

def _parse_gene_list(text: str) -> List[str]:
    """Parse comma/semicolon separated gene list, filtering Ensembl IDs."""
    if not text:
        return []
    genes = [g.strip() for g in text.replace(";", ",").split(",") if g.strip()]
    return [g for g in genes if not g.startswith("ENSG")]


def process_program(
    row: Dict[str, Any],
    args,
    llm: LLMBackend,
    search_engine: SearchEngine,
    paper_fetcher: PaperFetcher,
    verifier: Verifier,
    interaction_verbs: List[str],
    domain_keywords: List[str],
) -> None:
    """Run the full pipeline for one program."""
    program_id = row.get("program_id", "unknown")
    logger.info("=" * 60)
    logger.info("Processing program %s", program_id)
    logger.info("=" * 60)

    output_dir = args.output_dir
    cache_dir = output_dir / "cache"

    # Check for cached final output (resume support)
    json_path = output_dir / f"program_{program_id}.json"
    if args.resume and json_path.exists():
        logger.info("Program %s already processed (resume). Skipping.", program_id)
        return

    genes = _parse_gene_list(row.get("top_genes", ""))
    if not genes:
        logger.warning("Program %s has no top genes. Skipping.", program_id)
        return

    # -- Step 1: Generate LLM queries --
    query_cache = cache_dir / "queries" / f"{program_id}_llm.json"
    query_cache.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and query_cache.exists():
        llm_queries = json.loads(query_cache.read_text(encoding="utf-8"))
        logger.info("Loaded %d cached LLM queries", len(llm_queries))
    else:
        llm_queries = generate_search_queries(
            llm, row, max_queries=args.max_llm_queries,
            interaction_verbs=interaction_verbs,
        )
        query_cache.write_text(json.dumps(llm_queries, indent=2), encoding="utf-8")

    # -- Step 2: Search --
    search_results = search_engine.search_all_for_program(
        genes=genes,
        llm_queries=llm_queries,
        max_pubtator_results=args.max_pubtator_results,
        max_total=args.max_papers,
    )
    pmids = [r["pmid"] for r in search_results]
    if not pmids:
        logger.warning("No papers found for program %s", program_id)
        return

    # -- Step 3: Fetch papers --
    all_genes = genes + _parse_gene_list(row.get("regulators", ""))
    papers = paper_fetcher.fetch_papers(
        pmids=pmids,
        target_genes=all_genes[:30],
        context_genes=genes,
        domain_keywords=domain_keywords,
    )

    # -- Step 4: Summarize papers (LLM) --
    program_context = {k: v for k, v in row.items() if v}
    for pmid, paper in papers.items():
        summarize_paper(paper, program_context, llm)

    # -- Step 5: Verify --
    verifications: Dict[int, List[VerificationResult]] = {}
    for pmid, paper in papers.items():
        claimed_genes = [g for g in genes if g.upper() in paper.gene_mentions]
        if not claimed_genes:
            claimed_genes = genes[:5]

        vr_list = verifier.verify_paper(
            paper=paper,
            claimed_genes=claimed_genes,
            run_semantic=args.semantic_check,
        )
        verifications[pmid] = vr_list

    # -- Step 6: Write output --
    write_program_json(
        output_dir=output_dir,
        program_id=program_id,
        papers=papers,
        verifications=verifications,
        search_queries=llm_queries,
        program_context=program_context,
    )

    md_path = write_program_markdown(
        output_dir=output_dir,
        program_id=program_id,
        papers=papers,
        verifications=verifications,
        program_context=program_context,
        llm=llm,
    )

    # Read synthesis from the markdown
    md_text = md_path.read_text(encoding="utf-8")
    synthesis_text = ""
    if "## Synthesis" in md_text:
        synthesis_text = md_text.split("## Synthesis\n")[-1].strip()

    write_program_html(
        output_dir=output_dir,
        program_id=program_id,
        papers=papers,
        verifications=verifications,
        program_context=program_context,
        synthesis_text=synthesis_text,
    )

    logger.info("Program %s complete: %d papers, %d verified", program_id, len(papers), len(verifications))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(args) -> None:
    """Run the literature search pipeline."""
    # initializing output dir 
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parse list arguments
    interaction_verbs = [v.strip() for v in args.interactions.split(",") if v.strip()]
    domain_keywords = [k.strip() for k in args.domain_keywords.split(",") if k.strip()]

    # Initialize llm model 
    llm = LLMBackend(
        provider=args.llm_provider,
        model=args.llm_model,
        max_tokens=args.llm_max_tokens,
    )

    # initializing useful agent 
    search_engine = SearchEngine()
    paper_fetcher = PaperFetcher(args.output_dir)
    verifier = Verifier(llm=llm if args.semantic_check else None)

    # Load programs from Excel
    programs = None
    if args.programs:
        programs = [int(x.strip()) for x in args.programs.split(",")]
    rows = load_excel(args.excel, programs=programs)
    logger.info("Processing %d programs", len(rows))

    # annotate each program 
    for row in rows:
        try:
            process_program(
                row, args, llm, search_engine, paper_fetcher, verifier,
                interaction_verbs, domain_keywords,
            )
        except Exception as exc:
            pid = row.get("program_id", "?")
            logger.error("Program %s failed: %s", pid, exc, exc_info=True)

    # Summary
    logger.info("=" * 60)
    logger.info("Literature search complete!")
    logger.info("Token usage: %s", llm.token_usage)
    logger.info("Output: %s", args.output_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Literature Search Agent for PerturbNMF program annotation",
    )
    # Required
    parser.add_argument("--excel", type=Path, required=True, help="Input Excel with program rows")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    # Program selection
    parser.add_argument("--programs", type=str, default=None,
                        help="Comma-separated program IDs (default: all)")

    # Search configuration
    parser.add_argument("--interactions", type=str,
                        default="regulates,induces,promotes,inhibits,suppresses,activates,binds,phosphorylates,modulates,mediates,targets,controls,decreases,increases,blocks,triggers,catalyzes",
                        help="Comma-separated interaction verbs for query formulation")
    parser.add_argument("--domain-keywords", type=str,
                        default="angiogenesis,permeability,barrier,inflammation,proliferation,migration,sprouting,hypoxia,metabolism,junction,adhesion,leukocyte,shear,tip cell,stalk cell,arterial,venous,capillary,blood-brain barrier,bbb",
                        help="Comma-separated domain keywords for evidence scoring")
    parser.add_argument("--max-papers", type=int, default=30,
                        help="Max papers per program (default: 30)")
    parser.add_argument("--max-pubtator-results", type=int, default=50,
                        help="Max results per PubTator query (default: 50)")
    parser.add_argument("--max-llm-queries", type=int, default=8,
                        help="Max LLM-generated queries per program (default: 8)")

    # LLM configuration
    parser.add_argument("--llm-provider", default="stanford",
                        choices=["anthropic", "stanford", "openai", "deepseek", "gemini"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)

    # Verification
    parser.add_argument("--semantic-check", action="store_true", default=False,
                        help="Enable LLM semantic verification (costs tokens, off by default)")

    # Resume
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    args = parser.parse_args(argv)

    # Validate
    if not args.excel.exists():
        parser.error(f"Excel file not found: {args.excel}")
    if not args.output_dir.parent.exists():
        parser.error(f"Parent directory for output does not exist: {args.output_dir.parent}")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # intalize folders to be used 
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    slurm_info = {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'job_name': os.environ.get('SLURM_JOB_NAME'),
        'partition': os.environ.get('SLURM_JOB_PARTITION'),
        'node_list': os.environ.get('SLURM_JOB_NODELIST'),
    }
    job_id = slurm_info['job_id'] or 'no_jobid'
    config_to_save = {
        'args': {k: str(v) for k, v in vars(args).items()},
        'slurm_info': slurm_info,
    }
    config_path = args.output_dir / f'config_{job_id}.yml'
    with open(config_path, 'w') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, width=1000)

    run(args)


if __name__ == "__main__":
    sys.exit(main())
