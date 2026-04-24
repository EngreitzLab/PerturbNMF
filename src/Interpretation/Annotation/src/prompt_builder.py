"""
Prompt builder for LLM-based program annotation.

Extracts gene lists, cell-type annotations, enrichment context, NCBI literature
evidence, and regulator data to construct structured prompts for batch annotation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from .column_mapper import standardize_regulator_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-5-20250929"  # Anthropic model name
DEFAULT_ANNOTATION_ROLE = "vascular-biology specialist"
DEFAULT_ANNOTATION_CONTEXT = (
    "a gene program extracted from single-cell Perturb-seq of mouse brain "
    "endothelial cells (ECs)"
)
DEFAULT_SEARCH_KEYWORD = '(endothelial OR endothelium OR "vascular endothelial")'

PROMPT_TEMPLATE = """
## Edited prompt (evidence-first, low-speculation)

You are a {annotation_role} interpreting Topic {program_id}, {annotation_context}.

### Project context
- Literature search keyword/cell type: {search_keyword}

### Goal
- Provide a specific, evidence-anchored interpretation of Topic {program_id}.
- Gene lists are primary evidence; enrichment and cell-type context are cross-checks only.

### Gene evidence (primary)
{gene_context}

### Secondary context (cross-check only; not primary evidence)
{regulator_analysis}
{celltype_context}
{enrichment_context}
{ncbi_context}

### Evidence rules (strict)
- Each biological claim must cite genes and evidences in parentheses.
- If you cannot support a claim from the provided genes, write exactly: "Unclear from the provided gene lists."
- Enrichment/cell-type can be mentioned only when supported by supplied evidence.

### Style rules
- Be mechanistic, specific, and conservative.
- Prefer what the program contains over what it causes.
- State uncertainty explicitly when warranted.

### Output requirements (GitHub-flavored Markdown)
Start with: `## Program {program_id} annotation`

CRITICALLY, include the following two lines near the top, exactly with these bold labels:
- **Brief Summary:** <1-2 sentences>
- **Program label:** <=6 words; no "regulation of", no "process", no generic catch-alls>

Then provide the following sections:

1. **High-level overview (<=120 words)**
   - Main theme(s) grounded in the primary gene lists (each claim supported by >=2 genes).
   - Connect to the cell-type enrichment only if supported by annotation; otherwise state exactly: "Cell-type enrichment noted but gene evidence for mapping is limited."

2. **Functional modules and mechanisms**
   Group genes into 3-5 modules. For each module, use this exact format:
   ```
   Module name: 1-sentence summary
   Key genes: list 2-10
   Proposed mechanism: 1-2 sentences, backed by primary program genes or secondary context, with reasoning.
   ```

3. **Distinctive features**
   - Describe what is most distinctive about Program {program_id} in 1-2 sentences. Cite unique genes and provide reasoning.
   - If evidence is limited or mixed, say so explicitly (and still cite genes).

4. **Regulator analysis**
   List 1-3 most prominent regulators from Perturb-seq, for each regulator use this exact format:
   ```
   regulator_name (role, log2FC=X): [Confidence: High/Medium/Low]
   Propose a mechanistic hypothesis: How might this regulator control the program's genes/pathways? Cite program genes and evidence.
   ```
5. **Program stats**
   - **Top-loading genes:** [list from Gene evidence section]
   - **Unique genes:** [list or "None"]
   - **Cell-type enrichment:** [1-sentence summary]
"""

# ---------------------------------------------------------------------------
# Uniqueness helpers (private)
# ---------------------------------------------------------------------------


def _ensure_program_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "program_id" in df.columns:
        return df
    if "RowID" not in df.columns:
        raise ValueError("CSV must have 'program_id' or 'RowID'")
    updated = df.copy()
    updated["program_id"] = updated["RowID"]
    return updated


def _add_global_uniqueness_scores(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Name", "Score", "program_id"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_sorted = sorted(missing)
        raise ValueError(
            f"CSV missing required columns for uniqueness: {missing_sorted}"
        )

    updated = df.copy()
    updated["Score"] = pd.to_numeric(updated["Score"], errors="coerce")
    updated["program_id"] = pd.to_numeric(updated["program_id"], errors="coerce")

    valid = updated.dropna(subset=["Name", "Score", "program_id"]).copy()
    if valid.empty:
        raise ValueError("No valid rows to compute uniqueness scores.")

    valid["program_id"] = valid["program_id"].astype(int)
    total_programs = valid["program_id"].nunique()
    gene_counts = valid.groupby("Name")["program_id"].nunique().astype(float)
    idf = np.log((total_programs + 1.0) / (gene_counts + 1.0))
    valid["UniquenessScore"] = valid["Score"] * valid["Name"].map(idf)

    updated["UniquenessScore"] = np.nan
    updated.loc[valid.index, "UniquenessScore"] = valid["UniquenessScore"]
    return updated


def _ensure_global_uniqueness(
    df: pd.DataFrame, log: logging.Logger
) -> pd.DataFrame:
    if "UniquenessScore" in df.columns and not df["UniquenessScore"].isna().all():
        return df
    log.info("UniquenessScore missing; computing global uniqueness scores.")
    return _add_global_uniqueness_scores(df)


# ---------------------------------------------------------------------------
# Data loading and formatting helpers
# ---------------------------------------------------------------------------


def _split_pipe_list(value: object) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def _parse_program_id(value: object) -> Optional[int]:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    text = str(value).strip()
    if text.lower().startswith("program_"):
        text = text.split("_", 1)[-1]
    try:
        return int(text)
    except ValueError:
        return None


def load_gene_table(gene_file: Path) -> pd.DataFrame:
    if not gene_file.exists():
        raise FileNotFoundError(f"Gene file not found: {gene_file}")
    df = pd.read_csv(gene_file)
    df = _ensure_program_id_column(df)
    required_cols = {"Name", "Score", "program_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Gene file missing required columns: {missing}")
    df = _ensure_global_uniqueness(df, logger)
    return df


def load_celltype_annotations(
    celltype_dir: Path, celltype_file: Optional[Path] = None
) -> Dict[int, Dict[str, List[str]]]:
    # Use explicit file if provided, otherwise look in directory
    if celltype_file and celltype_file.exists():
        summary_path = celltype_file
    else:
        # Try common filenames in the directory
        for filename in ["celltype_summary.csv", "program_celltype_annotations_summary.csv"]:
            candidate = celltype_dir / filename
            if candidate.exists():
                summary_path = candidate
                break
        else:
            logger.warning("Cell-type summary not found in: %s", celltype_dir)
            return {}

    df = pd.read_csv(summary_path)
    required_cols = {
        "program",
        "highly_cell_type_specific",
        "moderately_enriched",
        "weakly_enriched",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Cell-type summary missing columns: %s", missing)
        return {}
    depleted_column = None
    if "depleted" in df.columns:
        depleted_column = "depleted"
    elif "significantly_lower_expression" in df.columns:
        depleted_column = "significantly_lower_expression"
        logger.warning(
            "Cell-type summary uses legacy column 'significantly_lower_expression'; "
            "prefer 'depleted' in regenerated summaries."
        )
    else:
        logger.warning("Cell-type summary missing depleted column ('depleted').")
        return {}

    annotation_map: Dict[int, Dict[str, List[str]]] = {}
    for _, row in df.iterrows():
        program_id = _parse_program_id(row.get("program"))
        if program_id is None:
            continue
        annotation_map[program_id] = {
            "highly_cell_type_specific": _split_pipe_list(
                row.get("highly_cell_type_specific")
            ),
            "moderately_enriched": _split_pipe_list(row.get("moderately_enriched")),
            "weakly_enriched": _split_pipe_list(row.get("weakly_enriched")),
            "depleted": _split_pipe_list(row.get(depleted_column)),
        }
    return annotation_map


def format_celltype_context(
    annotation_map: Dict[int, Dict[str, List[str]]], program_id: int
) -> str:
    program_info = annotation_map.get(program_id)
    if not program_info:
        return "Cell-type enrichment: Not available."

    lines = ["Cell-type enrichment:"]
    label_map = {
        "highly_cell_type_specific": "Highly specific",
        "moderately_enriched": "Moderately enriched",
        "weakly_enriched": "Weakly enriched",
        "depleted": "Depleted in",
    }
    for key, label in label_map.items():
        values = program_info.get(key, [])
        if values:
            lines.append(f"- {label}: {', '.join(values)}")
    if len(lines) == 1:
        return "Cell-type enrichment: Not available."
    return "\n".join(lines)


def prepare_enrichment_mapping(
    enrichment_file: Optional[Path],
) -> Dict[int, Dict[str, List[dict]]]:
    if not enrichment_file:
        return {}
    if not enrichment_file.exists():
        logger.warning("Enrichment file not found: %s", enrichment_file)
        return {}

    df = pd.read_csv(enrichment_file)
    required_cols = {"program_id", "category", "description", "fdr", "inputGenes"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Enrichment file missing required columns: %s", missing)
        return {}

    df["category"] = df["category"].astype(str).str.strip()
    df_sorted = df.sort_values(["program_id", "category", "fdr"], ascending=[True, True, True])

    enrichment_by_program: Dict[int, Dict[str, List[dict]]] = {}
    for (pid, cat), sub in df_sorted.groupby(["program_id", "category"], sort=False):  # type: ignore
        if cat not in ("KEGG", "Process"):
            continue
        program_map = enrichment_by_program.setdefault(int(pid), {})
        program_map[cat] = sub.to_dict(orient="records")
    return enrichment_by_program


def load_ncbi_context(json_path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if not json_path or not json_path.exists():
        if json_path:
            logger.warning("NCBI file not found: %s", json_path)
        return {}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Parse keys as ints (JSON keys are always strings)
        return {int(k): v for k, v in data.items()}
    except Exception as e:
        logger.error("Error parsing NCBI JSON: %s", e)
        return {}


def format_ncbi_context(ncbi_data: Dict[int, Dict[str, Any]], program_id: int, allowed_genes: Optional[Set[str]] = None) -> str:
    ctx = ncbi_data.get(program_id)
    if not ctx:
        return "Literature evidence: None available."

    lines = []

    # 1. Official Gene Summaries (Filtered by allowed_genes)
    summaries = ctx.get("gene_summaries", {})
    if summaries:
        source = str(ctx.get("gene_summaries_source", "ncbi")).lower()
        source_label = "Harmonizome" if source == "harmonizome" else "Entrez (NCBI)"
        lines.append(f"\nGene Summaries ({source_label}):")
        # Filter: Only show summaries for genes in the allowed list (if provided)
        # And sort alphabetically
        sorted_genes = sorted(summaries.keys())
        count = 0
        for gene in sorted_genes:
            if allowed_genes and gene not in allowed_genes:
                continue

            desc = summaries[gene]
            if source == "ncbi":
                # Remove citations like [provided by ...]
                desc = re.sub(
                    r'\s*\[provided by .*?\]\.?',
                    '',
                    desc,
                    flags=re.IGNORECASE,
                )
            lines.append(f"- {gene}: {desc}")
            count += 1

        if count == 0:
             lines.append("None available for selected genes.")

    # 2. Aggregated Evidence (Snippets)
    # gene -> list of strings "Statement (PMID:123)"
    ev_map = ctx.get("evidence_snippets", {})
    if ev_map:
        lines.append("\nAggregated Evidence (Contextual sentences from literature):")

        # Sort genes to be deterministic
        sorted_ev_genes = sorted(ev_map.keys())
        has_snippets = False

        for sym in sorted_ev_genes:
            if allowed_genes and sym not in allowed_genes:
                continue

            s_list = ev_map[sym]
            if s_list:
                # Robust Deduplication:
                # 1. Normalize text (strip punctuation)
                # 2. Check PMID (Max 1 snippet per PMID per Gene)
                seen_normalized = set()
                seen_pmids = set()

                gene_snippets = []

                for s in s_list:
                    # s format: "Sentence text.." (PMID:12345)
                    # Extract PMID using regex
                    pmid_match = re.search(r'\(PMID:(\d+)\)', s)
                    pmid = pmid_match.group(1) if pmid_match else None

                    # Dedup by PMID (One sentence per paper per gene)
                    if pmid:
                        if pmid in seen_pmids:
                            continue
                        seen_pmids.add(pmid)

                    # Dedup by Content (if PMID parsing fails or for safety)
                    norm = s.strip(" .")
                    if norm in seen_normalized:
                        continue
                    seen_normalized.add(norm)

                    # Clean up display string
                    # Remove double periods, ensure single period
                    clean_s = s.strip()
                    while ".." in clean_s:
                        clean_s = clean_s.replace("..", ".")
                    # It might be missing a period now if we stripped it, or original had it
                    # But re.split retains it.
                    # Just ensure it looks nice.
                    if not clean_s.endswith("."):
                        clean_s += "."

                    gene_snippets.append(f"- {sym}: {clean_s}")

                    if len(gene_snippets) >= 5: break # Max 5 per gene

                if gene_snippets:
                    has_snippets = True
                    lines.extend(gene_snippets)

        if not has_snippets:
             lines.append("None found.")

    return "\n".join(lines)


def load_regulator_data(
    csv_path: Optional[Path],
    significance_threshold: float = 0.05,
) -> Dict[int, pd.DataFrame]:
    """Load significant regulators from SCEPTRE results CSV.

    Returns a dict mapping program_id -> DataFrame with columns:
    [grna_target, log_2_fold_change, p_value]
    """
    if not csv_path or not csv_path.exists():
        if csv_path:
            logger.warning("Regulator file not found: %s", csv_path)
        return {}

    try:
        df = pd.read_csv(csv_path)
        df = standardize_regulator_results(
            df, significance_threshold=significance_threshold
        )
        df = df[df["significant"] == True].copy()

        # Group by program
        result = {}
        for pid, group in df.groupby("program_id"):
            keep_cols = ["grna_target", "log_2_fold_change", "p_value", "significant"]
            if "adj_p_value" in group.columns:
                keep_cols.append("adj_p_value")
            result[pid] = group[keep_cols].copy()

        logger.info("Loaded regulators for %d programs", len(result))
        return result
    except Exception as e:
        logger.error("Error loading regulator data: %s", e)
        return {}


def format_regulator_context(
    regulator_data: Dict[int, pd.DataFrame],
    program_id: int,
    top_n: int = 5,
) -> str:
    """Format basic regulator context for a program (Perturb-seq stats only).

    This is a fallback when no literature validation is available.
    Shows top positive and negative regulators with log2FC.
    """
    reg_df = regulator_data.get(program_id)
    if reg_df is None or len(reg_df) == 0:
        return "Regulator perturbations: None significant."

    # Sort by log2FC
    sorted_df = reg_df.sort_values("log_2_fold_change")

    # Negative log2FC = positive regulators (knockdown reduces program, so gene activates it)
    positive_regs = sorted_df[sorted_df["log_2_fold_change"] < 0].head(top_n)
    # Positive log2FC = negative regulators (knockdown increases program, so gene represses it)
    negative_regs = sorted_df[sorted_df["log_2_fold_change"] > 0].tail(top_n).iloc[::-1]

    lines = ["Regulator perturbations (from Perturb-seq; log2FC indicates effect of knockdown on program activity):"]

    if len(positive_regs) > 0:
        lines.append("\nPositive regulators (activators; knockdown reduces program):")
        for _, row in positive_regs.iterrows():
            lines.append(f"- {row['grna_target']}: log2FC = {row['log_2_fold_change']:.3f}")

    if len(negative_regs) > 0:
        lines.append("\nNegative regulators (repressors; knockdown increases program):")
        for _, row in negative_regs.iterrows():
            lines.append(f"- {row['grna_target']}: log2FC = {row['log_2_fold_change']:.3f}")

    return "\n".join(lines)


def format_regulator_analysis_context(
    regulator_data: Dict[int, pd.DataFrame],
    ncbi_data: Dict[int, Dict[str, Any]],
    program_id: int,
    top_positive_regulators: int = 3,
    top_negative_regulators: int = 3,
    min_score: int = 400,
) -> str:
    """Format comprehensive regulator analysis with compact STRING interactions.

    Shows top regulators (by effect size) with STRING interactions in compact format:
    - Gene (log2FC=X): Target1(score), Target2(score), ...

    Args:
        min_score: Minimum STRING score to display (default 400 = medium confidence)
    """
    reg_df = regulator_data.get(program_id)
    ctx = ncbi_data.get(program_id, {})
    validation = ctx.get("regulator_validation")

    if reg_df is None or len(reg_df) == 0:
        return "### Regulator evidence\nNo significant regulators identified from Perturb-seq."

    lines = ["### Regulator evidence"]
    lines.append(
        f"(Top {top_positive_regulators} activators + "
        f"{top_negative_regulators} repressors by effect size; "
        "STRING-DB interactions shown)"
    )
    lines.append("")

    def format_compact_interactions(val: Dict[str, Any]) -> str:
        """Format STRING interactions compactly: Target1(score), Target2(score), ..."""
        string_ints = val.get("string_interactions", [])
        if not string_ints:
            return ""

        # Filter by min_score and format compactly
        parts = []
        for si in string_ints:
            target = si.get("target", si.get("target_gene", "?"))
            score = si.get("score", 0)
            if score >= min_score:
                parts.append(f"{target}({score})")

        if not parts:
            return ""
        return " → " + ", ".join(parts[:8])  # Limit to 8 targets per regulator

    # Get validated regulators from ncbi_data (includes all significant)
    activators = validation.get("positive_regulators", []) if validation else []
    repressors = validation.get("negative_regulators", []) if validation else []
    n_activators = min(len(activators), top_positive_regulators)
    n_repressors = min(len(repressors), top_negative_regulators)

    # Format activators
    if activators:
        lines.append("#### Activators (knockdown reduces program activity)")
        for reg in activators[:n_activators]:
            gene = reg.get("regulator", "")
            log2fc = reg.get("log2fc", 0)
            interactions_str = format_compact_interactions(reg)

            if interactions_str:
                lines.append(f"- **{gene}** (log2FC={log2fc:.3f}){interactions_str}")
            else:
                lines.append(f"- {gene} (log2FC={log2fc:.3f})")
        lines.append("")

    # Format repressors
    if repressors:
        lines.append("#### Repressors (knockdown increases program activity)")
        for reg in repressors[:n_repressors]:
            gene = reg.get("regulator", "")
            log2fc = reg.get("log2fc", 0)
            interactions_str = format_compact_interactions(reg)

            if interactions_str:
                lines.append(f"- **{gene}** (log2FC={log2fc:+.3f}){interactions_str}")
            else:
                lines.append(f"- {gene} (log2FC={log2fc:+.3f})")
        lines.append("")

    return "\n".join(lines)


def build_enrichment_context(
    enrichment_by_program: Dict[int, Dict[str, List[dict]]],
    program_id: int,
    top_enrichment: int,
    genes_per_term: int,
) -> str:
    program_context = enrichment_by_program.get(program_id, {})
    if not program_context:
        return "STRING enrichment: Not available."

    lines = ["STRING enrichment (cross-check only; do not restate as conclusions):"]
    for category in ("KEGG", "Process"):
        rows = program_context.get(category, [])[:top_enrichment]
        if not rows:
            continue
        for row in rows:
            desc = row.get("description") or row.get("term") or "NA"
            fdr = row.get("fdr")
            fdr_str = f"{float(fdr):.2e}" if isinstance(fdr, (float, int)) else str(fdr)
            genes = _split_pipe_list(row.get("inputGenes"))
            genes = genes[:genes_per_term]
            genes_str = ", ".join(genes) if genes else "NA"
            lines.append(f"- {category}: {desc} (FDR={fdr_str}) - {genes_str}")
    return "\n".join(lines)


def select_program_genes(
    gene_df: pd.DataFrame,
    program_id: int,
    top_loading: int,
    top_unique: int,
) -> Tuple[List[str], List[str]]:
    program_df = gene_df[gene_df["program_id"] == program_id].copy()
    if program_df.empty:
        return [], []
    top_loading_genes = (
        program_df.sort_values("Score", ascending=False)["Name"].head(top_loading).tolist()  # type: ignore
    )
    unique_ranked = program_df.sort_values("UniquenessScore", ascending=False)["Name"].tolist()  # type: ignore
    unique_genes = [gene for gene in unique_ranked if gene not in top_loading_genes]
    return top_loading_genes, unique_genes[:top_unique]


def generate_prompt(
    program_id: int,
    gene_df: pd.DataFrame,
    prompt_template: str,
    top_loading: int,
    top_unique: int,
    celltype_map: Dict[int, Dict[str, List[str]]],
    enrichment_by_program: Dict[int, Dict[str, List[dict]]],
    ncbi_data: Dict[int, Dict[str, Any]],
    top_enrichment: int,
    genes_per_term: int,
    search_keyword: str,
    annotation_role: str,
    annotation_context: str,
    regulator_data: Optional[Dict[int, pd.DataFrame]] = None,
    top_positive_regulators: int = 3,
    top_negative_regulators: int = 3,
) -> Optional[str]:
    top_loading_genes, unique_genes = select_program_genes(
        gene_df=gene_df,
        program_id=program_id,
        top_loading=top_loading,
        top_unique=top_unique,
    )
    if not top_loading_genes:
        logger.warning("No genes found for Program %s", program_id)
        return None

    enrichment_context = build_enrichment_context(
        enrichment_by_program=enrichment_by_program,
        program_id=program_id,
        top_enrichment=top_enrichment,
        genes_per_term=genes_per_term,
    )

    celltype_context = format_celltype_context(celltype_map, program_id)

    # Build gene context
    gene_context = (
        f"Top-loading genes (top {len(top_loading_genes)}):\n"
        f"{', '.join(top_loading_genes)}"
    )
    if unique_genes:
        gene_context += (
            f"\n\nUnique genes (top {len(unique_genes)} non-overlapping):\n"
            f"{', '.join(unique_genes)}"
        )

    allowed_genes = set(top_loading_genes) | set(unique_genes)
    ncbi_context = format_ncbi_context(ncbi_data, program_id, allowed_genes=allowed_genes)

    regulator_analysis = format_regulator_analysis_context(
        regulator_data or {},
        ncbi_data,
        program_id,
        top_positive_regulators=top_positive_regulators,
        top_negative_regulators=top_negative_regulators,
    )

    annotation_role = annotation_role or DEFAULT_ANNOTATION_ROLE
    annotation_context = annotation_context or DEFAULT_ANNOTATION_CONTEXT
    search_keyword = search_keyword or DEFAULT_SEARCH_KEYWORD

    return (
        prompt_template.replace("{program_id}", str(program_id))
        .replace("{gene_context}", gene_context)
        .replace("{regulator_analysis}", regulator_analysis)
        .replace("{celltype_context}", celltype_context)
        .replace("{enrichment_context}", enrichment_context)
        .replace("{ncbi_context}", ncbi_context)
        .replace("{annotation_role}", annotation_role)
        .replace("{annotation_context}", annotation_context)
        .replace("{search_keyword}", search_keyword)
    )


# ---------------------------------------------------------------------------
# parse_topics_value — also used by cmd_prepare
# ---------------------------------------------------------------------------


def parse_topics_value(value: Optional[object]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(t.strip()) for t in value.split(",") if t.strip()]
    return None


# ---------------------------------------------------------------------------
# cmd_prepare  (takes argparse.Namespace)
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    """Prepare a batch request JSON file for the given gene CSV."""
    if not args.gene_file:
        logger.error("--gene-file is required (via CLI or config).")
        return 2
    try:
        gene_df = load_gene_table(Path(args.gene_file))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Error reading gene file: %s", exc)
        return 2

    program_ids = sorted(gene_df["program_id"].dropna().astype(int).unique().tolist())

    # Filter by specific topics if requested
    selected_topics = parse_topics_value(args.topics)
    if selected_topics:
        program_ids = [pid for pid in program_ids if pid in selected_topics]
        logger.info(f"Limiting to specific topics: {program_ids}")
    # Fallback to num-topics limit
    elif args.num_topics:
        program_ids = program_ids[: args.num_topics]
        logger.info("Limiting to first %s topics for testing.", args.num_topics)

    celltype_file = Path(args.celltype_file) if args.celltype_file else None
    celltype_map = load_celltype_annotations(Path(args.celltype_dir), celltype_file)
    enrichment_by_program = prepare_enrichment_mapping(Path(args.enrichment_file))
    ncbi_data = load_ncbi_context(Path(args.ncbi_file) if args.ncbi_file else None)
    regulator_data = load_regulator_data(
        Path(args.regulator_file) if args.regulator_file else None,
        significance_threshold=args.regulator_significance_threshold,
    )

    batch_requests: List[dict] = []
    for program_id in program_ids:
        prompt = generate_prompt(
            program_id=program_id,
            gene_df=gene_df,
            prompt_template=PROMPT_TEMPLATE,
            top_loading=args.top_loading,
            top_unique=args.top_unique,
            celltype_map=celltype_map,
            enrichment_by_program=enrichment_by_program,
            ncbi_data=ncbi_data,
            top_enrichment=args.top_enrichment,
            genes_per_term=args.genes_per_term,
            search_keyword=args.search_keyword,
            annotation_role=args.annotation_role,
            annotation_context=args.annotation_context,
            regulator_data=regulator_data,
            top_positive_regulators=args.top_positive_regulators,
            top_negative_regulators=args.top_negative_regulators,
        )
        if prompt:
            request = {
                "custom_id": f"topic_{program_id}_annotation",
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            batch_requests.append(request)

    if not batch_requests:
        logger.error("No requests were generated. Aborting.")
        return 3

    payload = {"requests": batch_requests}
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Successfully created batch request file at %s", output_path)
        return 0
    except OSError as exc:
        logger.error("Error writing to file: %s", exc)
        return 4
