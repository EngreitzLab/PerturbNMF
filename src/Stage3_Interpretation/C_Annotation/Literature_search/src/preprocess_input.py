#!/usr/bin/env python3
"""Convert a PerturbNMF summary table Excel into the input format
expected by the literature search pipeline.

Usage:
    python preprocess_input.py \
        --summary-excel /path/to/summary_table.xlsx \
        --output /path/to/lit_search_input.xlsx \
        --sheet-name "Summary" \
        --conditions "D0,sample_D1,sample_D2,sample_D3"
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _collect_regulators(row: pd.Series, conditions: List[str]) -> str:
    """Merge 'Top 5 specific regulators (FDR<0.1) {cond}' across conditions, deduplicate."""
    all_regs = []
    for cond in conditions:
        col = f"Top 5 specific regulators (FDR<0.1) {cond}"
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            genes = [g.strip() for g in str(val).replace(",", ";").split(";") if g.strip()]
            all_regs.extend(genes)
    # Also check sigfdr columns as fallback
    for cond in conditions:
        col = f"sigfdr0.05_targets_sorted_abslog2fcd_{cond}"
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip() and not all_regs:
            genes = [g.strip() for g in str(val).replace(",", ";").split(";") if g.strip()]
            all_regs.extend(genes)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for g in all_regs:
        if g not in seen:
            seen.add(g)
            deduped.append(g)
    return ", ".join(deduped)


def _pick_cell_types(row: pd.Series, conditions: List[str], top_n: int = 3) -> str:
    """Pick top conditions by mean program score."""
    scores = {}
    for cond in conditions:
        col = f"Mean program score {cond}"
        val = row.get(col)
        if pd.notna(val):
            try:
                scores[cond] = float(val)
            except (ValueError, TypeError):
                pass
    if not scores:
        return ""
    sorted_conds = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return ", ".join(sorted_conds)


def _safe_str(val, sep_in: str = ";", sep_out: str = ", ") -> str:
    """Convert a semicolon-separated cell to comma-separated, handling NaN."""
    if pd.isna(val) or str(val).strip() == "":
        return ""
    return sep_out.join(g.strip() for g in str(val).split(sep_in) if g.strip())


def convert_summary_to_lit_input(
    summary_df: pd.DataFrame,
    conditions: List[str],
    top_genes_col: str = "top30_loaded_genes",
    num_genes: Optional[int] = None,
) -> pd.DataFrame:
    """Convert a summary table DataFrame into literature search input format.

    Parameters
    ----------
    summary_df : DataFrame
        Summary table with index = program_name.
    conditions : list of str
        Sample/condition names (e.g. ["D0", "sample_D1", ...]).
    top_genes_col : str
        Column name for top loaded genes.
    num_genes : int or None
        If set, truncate gene list to this many genes.

    Returns
    -------
    DataFrame with columns: program_id, top_genes, regulators,
        day_most_active, cell_types, go_enrichment, other_enrichment.
    """
    rows = []
    for prog_id, row in summary_df.iterrows():
        # Top genes
        genes_raw = _safe_str(row.get(top_genes_col, ""))
        if num_genes and genes_raw:
            gene_list = [g.strip() for g in genes_raw.split(",") if g.strip()]
            genes_raw = ", ".join(gene_list[:num_genes])

        # Regulators (merged across conditions, no direction)
        regulators = _collect_regulators(row, conditions)

        # Day most active
        day = row.get("Automatic Timepoint", "")
        day = str(day).strip() if pd.notna(day) else ""

        # Cell types (top conditions by mean score)
        cell_types = _pick_cell_types(row, conditions)

        # Enrichment terms
        go_enrichment = _safe_str(row.get("top10_enriched_go_terms", ""))
        other_enrichment = _safe_str(row.get("top10_enriched_genesets", ""))

        rows.append({
            "program_id": prog_id,
            "top_genes": genes_raw,
            "regulators": regulators,
            "day_most_active": day,
            "cell_types": cell_types,
            "go_enrichment": go_enrichment,
            "other_enrichment": other_enrichment,
        })

    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert PerturbNMF summary table to literature search input"
    )
    parser.add_argument(
        "--summary-excel", type=Path, required=True,
        help="Path to the summary table Excel file"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output Excel path for literature search input"
    )
    parser.add_argument(
        "--sheet-name", type=str, default="Summary",
        help="Sheet name in the summary Excel (default: Summary)"
    )
    parser.add_argument(
        "--top-genes-col", type=str, default="top30_loaded_genes",
        help="Column name for top loaded genes (default: top30_loaded_genes)"
    )
    parser.add_argument(
        "--num-genes", type=int, default=None,
        help="Truncate gene list to this many genes (default: all)"
    )
    parser.add_argument(
        "--conditions", type=str, required=True,
        help="Comma-separated condition/sample names (e.g. 'D0,sample_D1,sample_D2,sample_D3')"
    )
    parser.add_argument(
        "--programs", type=str, default=None,
        help="Comma-separated program IDs to include (default: all)"
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.summary_excel.exists():
        parser.error(f"Summary Excel not found: {args.summary_excel}")

    conditions = [c.strip() for c in args.conditions.split(",")]

    # Load summary table
    logger.info("Loading summary table from %s (sheet: %s)", args.summary_excel, args.sheet_name)
    summary_df = pd.read_excel(args.summary_excel, sheet_name=args.sheet_name, index_col=0)
    logger.info("Loaded %d programs, %d columns", len(summary_df), len(summary_df.columns))

    # Filter programs if specified
    if args.programs:
        selected = {p.strip() for p in args.programs.split(",")}
        summary_df = summary_df[summary_df.index.astype(str).isin(selected)]
        logger.info("Filtered to %d programs", len(summary_df))

    # Convert
    lit_df = convert_summary_to_lit_input(
        summary_df,
        conditions=conditions,
        top_genes_col=args.top_genes_col,
        num_genes=args.num_genes,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lit_df.to_excel(args.output, index=False)
    logger.info("Wrote %d programs to %s", len(lit_df), args.output)
    print(f"\nOutput columns: {list(lit_df.columns)}")
    print(f"Programs: {list(lit_df['program_id'])}")


if __name__ == "__main__":
    sys.exit(main())
