"""
/**
 * @description
 * Prepare Anthropic batch request JSON for program annotation prompts.
 * It enriches each program prompt with cell-type annotations,
 * top-loading genes, unique genes, and STRING enrichment summaries.
 *
 * Key features:
 * - Builds prompts only (no submission) for batch requests
 * - Uses top-loading + uniqueness genes for concise specificity
 * - Adds cell-type enrichment context from precomputed summaries
 * - Adds top KEGG/Process terms with representative overlap genes
 *
 * @dependencies
 * - pandas: CSV loading and grouping
 *
 * @examples
 * - Prepare prompts only:
 *   python pipeline/03_submit_and_monitor_batch.py prepare \
 *     --gene-file input/genes/FB_moi15_seq2_loading_gene_k100_top300.csv \
 *     --celltype-dir input/celltype \
 *     --enrichment-file input/enrichment/string_enrichment_filtered_process_kegg.csv \
 *     --ncbi-file results/output/ncbi_context.json \
 *     --regulator-file input/regulators/sceptre_discovery_analysis_results.csv \
 *     --output-file results/output/llm_batches/batch_request.json
 * 
 *     # (Note: --ncbi-file is optional but recommended if literature data was fetched)
 */
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from column_mapper import standardize_regulator_results

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set

# Vertex AI imports (optional - only needed for submit-vertex/check-vertex commands)
try:
    from google import genai  # type: ignore
    from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions  # type: ignore
    VERTEX_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    CreateBatchJobConfig = None  # type: ignore
    JobState = None  # type: ignore
    HttpOptions = None  # type: ignore
    VERTEX_AVAILABLE = False

# Anthropic imports (for direct Anthropic Batch API - default)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"  # Anthropic model name
DEFAULT_ANNOTATION_ROLE = "vascular-biology specialist"
DEFAULT_ANNOTATION_CONTEXT = (
    "a gene program extracted from single-cell Perturb-seq of mouse brain "
    "endothelial cells (ECs)"
)
DEFAULT_SEARCH_KEYWORD = '(endothelial OR endothelium OR "vascular endothelial")'

"""
@description
Configuration loader for batch preparation and submission.
It is responsible for reading JSON/YAML configs, applying per-command defaults,
and honoring CLI override precedence with optional test mode.

Key features:
- Supports JSON and YAML (if PyYAML is installed).
- Applies test mode topic limits for prepare commands.

@dependencies
- json: Built-in JSON parser
- yaml (optional): YAML parser when available
- sys: CLI inspection for override detection
"""


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("PyYAML is required for YAML configs.") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise SystemExit("Config must be a mapping at the top level.")
    return data


def get_cli_overrides(argv: List[str]) -> Set[str]:
    overrides: Set[str] = set()
    for token in argv:
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overrides.add(name.replace("-", "_"))
    return overrides


def parse_topics_value(value: Optional[object]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(t.strip()) for t in value.split(",") if t.strip()]
    return None


def apply_test_mode(
    args: argparse.Namespace, config: Dict[str, Any], cli_overrides: Set[str]
) -> argparse.Namespace:
    test_cfg = config.get("test", {}) if isinstance(config.get("test", {}), dict) else {}
    enabled = bool(test_cfg.get("enabled") or config.get("test_mode"))
    if not enabled:
        return args

    topics = test_cfg.get("topics") or test_cfg.get("programs") or config.get("test_programs")
    if hasattr(args, "topics") and "topics" not in cli_overrides and not getattr(args, "topics", None):
        if topics is not None:
            args.topics = topics

    num_topics = test_cfg.get("num_topics") or test_cfg.get("num_programs")
    if hasattr(args, "num_topics") and "num_topics" not in cli_overrides and not getattr(args, "num_topics", None):
        if num_topics is not None:
            args.num_topics = num_topics
    return args


def apply_config_overrides(
    args: argparse.Namespace, config: Dict[str, Any], cli_overrides: Set[str]
) -> argparse.Namespace:
    steps_cfg = config.get("steps", {}) if isinstance(config.get("steps", {}), dict) else {}
    section_map = {
        "prepare": "batch_prepare",
        "submit-vertex": "vertex_submit",
        "check-vertex": "vertex_check",
    }
    step_key = section_map.get(args.command)
    step_cfg = steps_cfg.get(step_key, {}) if step_key else {}
    if not isinstance(step_cfg, dict):
        return args

    for key, value in step_cfg.items():
        dest = key.replace("-", "_")
        if dest in cli_overrides:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
    return args

# Global uniqueness score helpers
"""
@description
This component ensures global gene uniqueness scores are available for pipeline
inputs. It normalizes program identifiers and computes TF-IDF-style uniqueness
when the column is missing.

Key features:
- Accepts RowID or program_id inputs.
- Computes UniquenessScore using a global IDF across all programs.

@dependencies
- numpy: IDF calculation
- pandas: DataFrame manipulation

@examples
- df = ensure_program_id_column(df)
- df = ensure_global_uniqueness(df, logger)
"""


def ensure_program_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "program_id" in df.columns:
        return df
    if "RowID" not in df.columns:
        raise ValueError("CSV must have 'program_id' or 'RowID'")
    updated = df.copy()
    updated["program_id"] = updated["RowID"]
    return updated


def add_global_uniqueness_scores(df: pd.DataFrame) -> pd.DataFrame:
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


def ensure_global_uniqueness(
    df: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    if "UniquenessScore" in df.columns and not df["UniquenessScore"].isna().all():
        return df
    logger.info("UniquenessScore missing; computing global uniqueness scores.")
    return add_global_uniqueness_scores(df)

# Vertex AI Configuration
VERTEX_PROJECT_ID = "hs-vascular-development"
VERTEX_LOCATION = "us-east5"
VERTEX_BUCKET = "gs://perturbseq/batch"

VERTEX_MODEL_MAP = {
    "claude-opus-4-5": "publishers/anthropic/models/claude-opus-4-5",
    "claude-sonnet-4-5": "publishers/anthropic/models/claude-sonnet-4-5",
    "claude-sonnet-4": "publishers/anthropic/models/claude-sonnet-4",
    "claude-haiku-4-5": "publishers/anthropic/models/claude-haiku-4-5",
    "claude-3-7-sonnet": "publishers/anthropic/models/claude-3-7-sonnet",
}


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
    df = ensure_program_id_column(df)
    required_cols = {"Name", "Score", "program_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Gene file missing required columns: {missing}")
    df = ensure_global_uniqueness(df, logger)
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


# =============================================================================
# Vertex AI Functions
# =============================================================================

def convert_to_vertex_jsonl(input_path: Path, output_path: Path) -> int:
    """Convert Anthropic batch JSON to JSONL format for Vertex AI.
    
    Anthropic direct API format:
        {"custom_id": "...", "params": {"model": "...", "messages": [...], "max_tokens": ...}}
    
    Vertex AI Claude batch format:
        {"custom_id": "...", "request": {"messages": [...], "anthropic_version": "vertex-2023-10-16", "max_tokens": ...}}
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    requests = data.get("requests", [])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for req in requests:
            # Convert from Anthropic format to Vertex AI format
            params = req.get("params", {})
            vertex_req = {
                "custom_id": req.get("custom_id", ""),
                "request": {
                    "messages": params.get("messages", []),
                    "anthropic_version": "vertex-2023-10-16",
                    "max_tokens": params.get("max_tokens", 4096),
                }
            }
            # Optionally include system prompt if present
            if "system" in params:
                vertex_req["request"]["system"] = params["system"]
            
            line = json.dumps(vertex_req)
            f.write(line + "\n")
    
    logger.info(f"Converted {len(requests)} requests to Vertex AI JSONL format: {output_path}")
    return len(requests)


def upload_to_gcs(local_path: Path, gcs_uri: str) -> bool:
    """Upload a local file to GCS using gcloud CLI."""
    cmd = ["gcloud", "storage", "cp", str(local_path), gcs_uri]
    logger.info(f"Uploading {local_path} to {gcs_uri}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"GCS upload failed: {result.stderr}")
        return False
    
    logger.info(f"Upload successful: {gcs_uri}")
    return True


def cmd_submit_vertex(args: argparse.Namespace) -> int:
    """Submit a batch job to Vertex AI."""
    if not VERTEX_AVAILABLE:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return 1

    if not args.batch_file:
        logger.error("batch_file is required (via CLI or config).")
        return 1
    
    batch_path = Path(args.batch_file)
    if not batch_path.exists():
        logger.error(f"Batch file not found: {batch_path}")
        return 1
    
    # Generate timestamp for unique naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_name = batch_path.stem
    
    # Convert to JSONL format
    jsonl_path = batch_path.with_suffix(".jsonl")
    convert_to_vertex_jsonl(batch_path, jsonl_path)
    
    # Upload to GCS
    bucket = args.bucket or VERTEX_BUCKET
    gcs_input_uri = f"{bucket}/inputs/{base_name}_{timestamp}.jsonl"
    if not upload_to_gcs(jsonl_path, gcs_input_uri):
        return 2
    
    # Resolve model name
    model = args.model or "claude-sonnet-4-5"
    if model in VERTEX_MODEL_MAP:
        model_path = VERTEX_MODEL_MAP[model]
    elif model.startswith("publishers/"):
        model_path = model
    else:
        model_path = f"publishers/anthropic/models/{model}"
    
    gcs_output_prefix = f"{bucket}/outputs/{base_name}_{timestamp}/"
    
    logger.info(f"Submitting batch job to Vertex AI...")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Input: {gcs_input_uri}")
    logger.info(f"  Output: {gcs_output_prefix}")
    
    client = genai.Client(http_options=HttpOptions(api_version="v1"))  # type: ignore
    
    job = client.batches.create(
        model=model_path,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_prefix),  # type: ignore
    )
    
    print(f"\nJob created!")
    print(f"  Job name: {job.name}")
    print(f"  Job state: {job.state}")
    print(f"  Output: {gcs_output_prefix}")
    
    if args.wait:
        completed_states = {  # type: ignore
            JobState.JOB_STATE_SUCCEEDED,  # type: ignore
            JobState.JOB_STATE_FAILED,  # type: ignore
            JobState.JOB_STATE_CANCELLED,  # type: ignore
            JobState.JOB_STATE_PAUSED,  # type: ignore
        }
        
        print("\nWaiting for job completion (checking every 30 seconds)...")
        
        while job.state not in completed_states:
            time.sleep(30)
            job = client.batches.get(name=job.name)
            print(f"  Job state: {job.state}")
        
        print(f"\nFinal state: {job.state}")
        
        if job.state == JobState.JOB_STATE_SUCCEEDED:  # type: ignore
            print(f"SUCCESS! Results available at: {gcs_output_prefix}")
        else:
            print(f"Job did not succeed.")
            return 3
    else:
        print(f"\nCheck status with: python {__file__} check-vertex --job-name {job.name}")
    
    return 0


def cmd_check_vertex(args: argparse.Namespace) -> int:
    """Check the status of a Vertex AI batch job."""
    if not VERTEX_AVAILABLE:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return 1

    if not args.job_name:
        logger.error("--job-name is required (via CLI or config).")
        return 1
    
    client = genai.Client(http_options=HttpOptions(api_version="v1"))  # type: ignore
    
    job = client.batches.get(name=args.job_name)
    
    print(f"Job name: {job.name}")
    print(f"Job state: {job.state}")
    
    if hasattr(job, 'create_time'):
        print(f"Created: {job.create_time}")
    if hasattr(job, 'update_time'):
        print(f"Updated: {job.update_time}")
    
    # Check if completed
    completed_states = {  # type: ignore
        JobState.JOB_STATE_SUCCEEDED,  # type: ignore
        JobState.JOB_STATE_FAILED,  # type: ignore
        JobState.JOB_STATE_CANCELLED,  # type: ignore
        JobState.JOB_STATE_PAUSED,  # type: ignore
    }
    
    if job.state == JobState.JOB_STATE_SUCCEEDED:  # type: ignore
        print(f"\nJob completed successfully!")
        print(f"Download results with:")
        print(f"  gcloud storage cp '<OUTPUT_PREFIX>*' ./")
    elif job.state in completed_states:
        print(f"\nJob ended with state: {job.state}")
    else:
        print(f"\nJob is still processing...")
    
    return 0


# =============================================================================
# Anthropic Direct Batch API Commands (Default)
# =============================================================================

def cmd_submit_anthropic(args: argparse.Namespace) -> int:
    """Submit a batch job to Anthropic Batch API (default).
    
    This is the preferred method - uses Anthropic's native Batch API directly.
    Faster and simpler than Vertex AI for most use cases.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return 1

    if not args.batch_file:
        logger.error("batch_file is required (via CLI or config).")
        return 1
    
    batch_path = Path(args.batch_file)
    if not batch_path.exists():
        logger.error(f"Batch file not found: {batch_path}")
        return 1
    
    # Load batch request
    with open(batch_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    requests_list = data.get("requests", [])
    if not requests_list:
        logger.error("No requests found in batch file.")
        return 1
    
    logger.info(f"Loaded {len(requests_list)} requests from {batch_path}")
    
    # Convert to Anthropic Batch API format
    # Anthropic expects: {"custom_id": "...", "params": {"model": ..., "max_tokens": ..., "messages": [...]}}
    anthropic_requests = []
    model = args.model or MODEL
    max_tokens = args.max_tokens or 8192
    
    for req in requests_list:
        custom_id = req.get("custom_id", f"request_{len(anthropic_requests)}")
        params = req.get("params", {})
        messages = params.get("messages", [])
        
        if not messages:
            logger.warning(f"Skipping request {custom_id}: no messages")
            continue
        
        anthropic_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
        })
    
    logger.info(f"Submitting {len(anthropic_requests)} requests to Anthropic Batch API...")
    logger.info(f"  Model: {model}")
    logger.info(f"  Max tokens: {max_tokens}")
    
    # Create Anthropic client (supports ANTHROPIC_AUTH_TOKEN for Bearer auth
    # or ANTHROPIC_API_KEY for x-api-key auth)
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not auth_token and not api_key:
        logger.error("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not found in environment. Set in .env file.")
        return 1

    client = anthropic.Anthropic(auth_token=auth_token) if auth_token else anthropic.Anthropic(api_key=api_key)
    
    # Submit batch
    batch = client.messages.batches.create(requests=anthropic_requests)
    
    print(f"\nBatch created!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Requests: {batch.request_counts}")
    
    # Save batch ID for later retrieval
    batch_id_file = batch_path.with_suffix(".batch_id")
    batch_id_file.write_text(batch.id, encoding="utf-8")
    logger.info(f"Saved batch ID to {batch_id_file}")
    
    if args.wait:
        print("\nWaiting for batch completion (checking every 30 seconds)...")
        
        while batch.processing_status == "in_progress":
            time.sleep(30)
            batch = client.messages.batches.retrieve(batch.id)
            print(f"  Status: {batch.processing_status} | Completed: {batch.request_counts.succeeded}/{batch.request_counts.processing + batch.request_counts.succeeded}")
        
        print(f"\nFinal status: {batch.processing_status}")
        
        if batch.processing_status == "ended":
            # Download results
            output_file = batch_path.with_name(f"{batch_path.stem}_results.jsonl")
            logger.info(f"Downloading results to {output_file}...")
            
            with open(output_file, "w", encoding="utf-8") as f:
                for result in client.messages.batches.results(batch.id):
                    f.write(json.dumps(result.model_dump()) + "\n")
            
            print(f"SUCCESS! Results saved to: {output_file}")
            print(f"  Succeeded: {batch.request_counts.succeeded}")
            print(f"  Errored: {batch.request_counts.errored}")
        else:
            print(f"Batch did not complete successfully.")
            return 3
    else:
        print(f"\nCheck status with: python {__file__} check-anthropic --batch-id {batch.id}")
        print(f"Or retrieve results later with: python {__file__} results-anthropic --batch-id {batch.id}")
    
    return 0


def cmd_submit_single(args: argparse.Namespace) -> int:
    """Submit requests one-by-one via client.messages.create().

    Produces the same JSONL output format as the Batch API so that the
    downstream parser (04_parse_and_summarize.py) works unchanged.

    Resume-safe: reads any existing output JSONL and skips custom_ids
    that already succeeded.
    """
    if not ANTHROPIC_AVAILABLE:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return 1

    if not args.batch_file:
        logger.error("batch_file is required.")
        return 1

    batch_path = Path(args.batch_file)
    if not batch_path.exists():
        logger.error(f"Batch file not found: {batch_path}")
        return 1

    # Load batch request
    with open(batch_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    requests_list = data.get("requests", [])
    if not requests_list:
        logger.error("No requests found in batch file.")
        return 1

    model = args.model or MODEL
    max_tokens = args.max_tokens or 8192

    # Determine output path
    output_file = Path(
        args.output
        or str(batch_path.with_name(batch_path.stem + "_results.jsonl"))
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ---- Resume support: load already-completed custom_ids ----
    completed_ids: set = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cid = entry.get("custom_id")
                    rtype = (entry.get("result") or {}).get("type")
                    if cid and rtype == "succeeded":
                        completed_ids.add(cid)
                except json.JSONDecodeError:
                    pass
        if completed_ids:
            logger.info(
                "Resuming: %d request(s) already completed, will be skipped.",
                len(completed_ids),
            )

    # ---- Create Anthropic client ----
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not auth_token and not api_key:
        logger.error(
            "ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not found in environment. "
            "Set in .env file."
        )
        return 1

    client = (
        anthropic.Anthropic(auth_token=auth_token)
        if auth_token
        else anthropic.Anthropic(api_key=api_key)
    )

    total = len(requests_list)
    succeeded = 0
    errored = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, req in enumerate(requests_list, start=1):
            custom_id = req.get("custom_id", f"request_{idx}")
            if custom_id in completed_ids:
                logger.info("[%d/%d] %s: skipped (already completed)", idx, total, custom_id)
                succeeded += 1
                continue

            params = req.get("params", {})
            messages = params.get("messages", [])
            if not messages:
                logger.warning("[%d/%d] %s: skipped (no messages)", idx, total, custom_id)
                continue

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )

                # Build the JSONL entry matching Batch API output format
                content_blocks = []
                for block in response.content:
                    if block.type == "text":
                        content_blocks.append({"type": "text", "text": block.text})

                entry = {
                    "custom_id": custom_id,
                    "result": {
                        "type": "succeeded",
                        "message": {
                            "content": content_blocks,
                        },
                    },
                }

                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                succeeded += 1

                in_tok = response.usage.input_tokens if response.usage else "?"
                out_tok = response.usage.output_tokens if response.usage else "?"
                logger.info(
                    "[%d/%d] %s: succeeded (%sin/%sout)",
                    idx, total, custom_id, in_tok, out_tok,
                )

            except Exception as exc:
                logger.error("[%d/%d] %s: error - %s", idx, total, custom_id, exc)
                entry = {
                    "custom_id": custom_id,
                    "result": {
                        "type": "errored",
                        "error": {"message": str(exc)},
                    },
                }
                out_f.write(json.dumps(entry) + "\n")
                out_f.flush()
                errored += 1

    print(f"\nCompleted: {succeeded} succeeded, {errored} errored out of {total}")
    print(f"Results saved to: {output_file}")
    return 0


def cmd_check_anthropic(args: argparse.Namespace) -> int:
    """Check the status of an Anthropic batch job."""
    if not ANTHROPIC_AVAILABLE:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return 1

    batch_id = args.batch_id
    if not batch_id:
        # Try to read from batch_id file
        if args.batch_file:
            batch_id_file = Path(args.batch_file).with_suffix(".batch_id")
            if batch_id_file.exists():
                batch_id = batch_id_file.read_text(encoding="utf-8").strip()
    
    if not batch_id:
        logger.error("--batch-id is required (or provide --batch-file with saved .batch_id)")
        return 1
    
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not auth_token and not api_key:
        logger.error("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not found in environment.")
        return 1

    client = anthropic.Anthropic(auth_token=auth_token) if auth_token else anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Created: {batch.created_at}")
    print(f"Requests:")
    print(f"  Processing: {batch.request_counts.processing}")
    print(f"  Succeeded: {batch.request_counts.succeeded}")
    print(f"  Errored: {batch.request_counts.errored}")
    print(f"  Canceled: {batch.request_counts.canceled}")
    
    if batch.processing_status == "ended":
        print(f"\nBatch completed! Retrieve results with:")
        print(f"  python {__file__} results-anthropic --batch-id {batch.id}")
    
    return 0


def cmd_results_anthropic(args: argparse.Namespace) -> int:
    """Download results from a completed Anthropic batch job."""
    if not ANTHROPIC_AVAILABLE:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        return 1

    batch_id = args.batch_id
    if not batch_id:
        if args.batch_file:
            batch_id_file = Path(args.batch_file).with_suffix(".batch_id")
            if batch_id_file.exists():
                batch_id = batch_id_file.read_text(encoding="utf-8").strip()
    
    if not batch_id:
        logger.error("--batch-id is required")
        return 1
    
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not auth_token and not api_key:
        logger.error("ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY not found in environment.")
        return 1

    client = anthropic.Anthropic(auth_token=auth_token) if auth_token else anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        logger.error(f"Batch is still processing: {batch.processing_status}")
        return 1
    
    output_file = Path(args.output or f"batch_{batch_id}_results.jsonl")
    
    logger.info(f"Downloading results to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            f.write(json.dumps(result.model_dump()) + "\n")
    
    print(f"Results saved to: {output_file}")
    print(f"  Succeeded: {batch.request_counts.succeeded}")
    print(f"  Errored: {batch.request_counts.errored}")
    
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Anthropic batch request JSON for topic annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            help="Path to config file (YAML or JSON)",
        )
        p.add_argument(
            "--gene-file",
            help=(
                "Path to gene CSV with columns: Name, Score, program_id or RowID. "
                "UniquenessScore computed if missing."
            ),
        )
        p.add_argument(
            "--annotation-role",
            default=DEFAULT_ANNOTATION_ROLE,
            help="Specialist role used in the LLM prompt header",
        )
        p.add_argument(
            "--annotation-context",
            default=DEFAULT_ANNOTATION_CONTEXT,
            help=(
                "Dataset/cell-type description inserted into the LLM prompt header "
                "(e.g., 'a gene program extracted from single-cell RNA-seq of microglia')"
            ),
        )
        p.add_argument(
            "--search-keyword",
            default=DEFAULT_SEARCH_KEYWORD,
            help="Literature search keyword/cell type shown in the LLM prompt",
        )
        p.add_argument(
            "--celltype-dir",
            default="input/celltype",
            help="Directory containing program cell-type annotation summary CSV",
        )
        p.add_argument(
            "--celltype-file",
            help="Path to cell-type summary CSV file (overrides --celltype-dir lookup)",
        )
        p.add_argument(
            "--enrichment-file",
            default="input/enrichment/string_enrichment_filtered_process_kegg.csv",
            help="STRING enrichment CSV (Process/KEGG) with inputGenes column",
        )
        p.add_argument(
            "--ncbi-file",
            help="NCBI Context JSON (from fetch_ncbi_data.py)",
        )
        p.add_argument(
            "--top-loading",
            type=int,
            default=20,
            help="Number of top-loading genes per program",
        )
        p.add_argument(
            "--top-unique",
            type=int,
            default=10,
            help="Number of unique genes per program (non-overlapping)",
        )
        p.add_argument(
            "--top-enrichment",
            type=int,
            default=3,
            help="Top-N rows per category (KEGG, Process) to include",
        )
        p.add_argument(
            "--genes-per-term",
            type=int,
            default=10,
            help="Representative overlap genes per enrichment term",
        )
        p.add_argument(
            "--output-file",
            default="anthropic_batch_request.json",
            help="Path to save the generated batch request JSON",
        )
        p.add_argument(
            "--num-topics",
            type=int,
            help="Limit the number of topics to process (testing)",
        )
        p.add_argument(
            "--topics",
            type=str,
            help="Comma-separated list of topic IDs to process (e.g. '5,6')",
        )
        p.add_argument(
            "--regulator-file",
            type=str,
            help="CSV file with regulator results",
        )
        p.add_argument(
            "--top-positive-regulators",
            type=int,
            default=3,
            help="Number of positive regulators to include in annotation prompts",
        )
        p.add_argument(
            "--top-negative-regulators",
            type=int,
            default=3,
            help="Number of negative regulators to include in annotation prompts",
        )
        p.add_argument(
            "--regulator-significance-threshold",
            type=float,
            default=0.05,
            help="Adjusted p-value threshold used when the regulator file has no 'significant' column",
        )

    p_prepare = subparsers.add_parser("prepare", help="Build the batch request JSON only")
    add_shared_options(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare)

    # =========================================================================
    # Anthropic Direct API Commands (Default - Recommended)
    # =========================================================================
    
    # Anthropic submit command (DEFAULT)
    p_submit = subparsers.add_parser(
        "submit",
        help="Submit batch to Anthropic Batch API (default, recommended)"
    )
    p_submit.add_argument(
        "batch_file",
        nargs="?",
        help="Path to the prepared batch JSON file"
    )
    p_submit.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_submit.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to use (default: {MODEL})"
    )
    p_submit.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for response (default: 8192)"
    )
    p_submit.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion and download results"
    )
    p_submit.set_defaults(func=cmd_submit_anthropic)

    # Anthropic single-message submit command (no Batch API)
    p_submit_single = subparsers.add_parser(
        "submit-single",
        help="Submit requests one-by-one via messages.create (no Batch API)",
    )
    p_submit_single.add_argument(
        "batch_file",
        nargs="?",
        help="Path to the prepared batch JSON file",
    )
    p_submit_single.add_argument(
        "--model",
        default=MODEL,
        help=f"Model to use (default: {MODEL})",
    )
    p_submit_single.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for response (default: 8192)",
    )
    p_submit_single.add_argument(
        "--output",
        help="Output JSONL file path (default: <batch_file>_results.jsonl)",
    )
    p_submit_single.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_submit_single.set_defaults(func=cmd_submit_single)

    # Anthropic check command
    p_check = subparsers.add_parser(
        "check",
        help="Check status of Anthropic batch job"
    )
    p_check.add_argument(
        "--batch-id",
        help="Anthropic batch ID"
    )
    p_check.add_argument(
        "--batch-file",
        help="Path to batch JSON (will read .batch_id file)"
    )
    p_check.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_check.set_defaults(func=cmd_check_anthropic)

    # Anthropic results command
    p_results = subparsers.add_parser(
        "results",
        help="Download results from completed Anthropic batch"
    )
    p_results.add_argument(
        "--batch-id",
        help="Anthropic batch ID"
    )
    p_results.add_argument(
        "--batch-file",
        help="Path to batch JSON (will read .batch_id file)"
    )
    p_results.add_argument(
        "--output",
        help="Output JSONL file path"
    )
    p_results.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_results.set_defaults(func=cmd_results_anthropic)

    # =========================================================================
    # Vertex AI Commands (Alternative)
    # =========================================================================

    # Vertex AI submit command
    p_submit_vertex = subparsers.add_parser(
        "submit-vertex",
        help="Submit a prepared batch JSON to Vertex AI"
    )
    p_submit_vertex.add_argument(
        "batch_file",
        nargs="?",
        help="Path to the prepared batch JSON file"
    )
    p_submit_vertex.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_submit_vertex.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="Model to use (default: claude-sonnet-4-5). Options: " + ", ".join(VERTEX_MODEL_MAP.keys())
    )
    p_submit_vertex.add_argument(
        "--bucket",
        default=VERTEX_BUCKET,
        help=f"GCS bucket prefix (default: {VERTEX_BUCKET})"
    )
    p_submit_vertex.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion instead of returning immediately"
    )
    p_submit_vertex.set_defaults(func=cmd_submit_vertex)

    # Vertex AI check command
    p_check_vertex = subparsers.add_parser(
        "check-vertex",
        help="Check status of a Vertex AI batch job"
    )
    p_check_vertex.add_argument(
        "--job-name",
        help="Full job name from Vertex AI (e.g., projects/.../batchPredictionJobs/...)"
    )
    p_check_vertex.add_argument(
        "--config",
        help="Path to config file (YAML or JSON)",
    )
    p_check_vertex.set_defaults(func=cmd_check_vertex)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(getattr(args, "config", None))
    cli_overrides = get_cli_overrides(sys.argv)
    args = apply_config_overrides(args, config, cli_overrides)
    args = apply_test_mode(args, config, cli_overrides)
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
