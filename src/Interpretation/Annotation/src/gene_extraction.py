"""
Gene extraction and config helpers split from 01_genes_to_string_enrichment.py.

Provides:
- Top-N gene extraction per program from loading CSVs
- UniquenessScore (TF-IDF-style) computation
- Cell-type enrichment validation and summary generation
- Shared config loading / CLI override utilities
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .column_mapper import ColumnMapper, standardize_gene_loading, standardize_celltype_enrichment

logger = logging.getLogger(__name__)

# ----------------------------- Cell-type summary -----------------------------
# Thresholds for categorizing cell-type enrichment (FDR < 0.05 required)
CELLTYPE_THRESHOLDS = {
    'highly_cell_type_specific': {'log2fc_min': 3.0, 'log2fc_max': None},
    'moderately_enriched': {'log2fc_min': 1.5, 'log2fc_max': 3.0},
    'weakly_enriched': {'log2fc_min': 0.5, 'log2fc_max': 1.5},
    'depleted': {'log2fc_min': None, 'log2fc_max': -0.5},
}

# Column order for cell-type summary output
CELLTYPE_CATEGORIES = [
    'highly_cell_type_specific',
    'moderately_enriched',
    'weakly_enriched',
    'depleted',
]


def extract_program_id(value: object) -> Optional[int]:
    """Extract numeric program ID from various naming formats.

    Handles formats like:
    - 'Program_1', 'program_1' -> 1
    - 'Topic_1', 'topic_1' -> 1
    - 'P1', 'p1', 'P_1', 'p_1' -> 1
    - 'X1', 'X_1' -> 1 (regulator file format)
    - '1', 1 -> 1
    - 'Program1', 'Topic1' -> 1

    Args:
        value: String or int containing program identifier

    Returns:
        Integer program ID or None if parsing fails
    """
    import re

    if value is None:
        return None

    # If already an integer, return it
    if isinstance(value, (int, np.integer)):
        return int(value)

    # Convert to string and strip whitespace
    val_str = str(value).strip()

    # Try direct integer conversion first
    try:
        return int(val_str)
    except (ValueError, TypeError):
        pass

    # Try to extract digits from common patterns
    # Patterns: Program_X, Topic_X, P_X, X_X (regulator format), program_X, topic_X, p_X
    patterns = [
        r'^(?:program|topic|p|x)_(\d+)$',  # program_1, topic_1, p_1, x_1
        r'^(?:program|topic|p|x)(\d+)$',   # program1, topic1, p1, x1
        r'^(\d+)$',                         # just the number
    ]

    for pattern in patterns:
        match = re.match(pattern, val_str, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError, IndexError):
                continue

    return None


def validate_celltype_enrichment(df: pd.DataFrame, file_path: Path) -> bool:
    """Validate cell-type enrichment DataFrame format and content.

    Non-fatal validation that logs warnings instead of raising errors.
    This allows the pipeline to continue even with imperfect input data.

    Args:
        df: DataFrame to validate
        file_path: Path to the file (for warning messages)

    Returns:
        True if validation passed, False if issues were found
    """
    warnings_list = []
    has_critical_errors = False

    # 1. Check required columns
    required_cols = {'cell_type', 'program', 'log2_fc_in_vs_out', 'fdr'}
    missing = required_cols - set(df.columns)
    if missing:
        warnings_list.append(f"Missing required columns: {sorted(missing)}")
        warnings_list.append(f"  Found columns: {sorted(df.columns)}")
        warnings_list.append(f"  Required: {sorted(required_cols)}")
        has_critical_errors = True

    # 2. Check DataFrame is not empty
    if df.empty:
        warnings_list.append("Cell-type enrichment file is empty")
        has_critical_errors = True

    # If critical errors, log and return early
    if has_critical_errors:
        logger.warning(f"Cell-type enrichment file has critical issues: {file_path}")
        for w in warnings_list:
            logger.warning(f"  {w}")
        return False

    # 3. Validate 'program' column - flexible format checking
    df_temp = df.copy()
    df_temp['program_id_parsed'] = df_temp['program'].apply(extract_program_id)
    unparseable = df_temp[df_temp['program_id_parsed'].isna()]
    if not unparseable.empty:
        sample_invalid = unparseable['program'].head(5).tolist()
        warnings_list.append(f"Could not parse {len(unparseable)} program identifiers: {sample_invalid}")
        warnings_list.append(f"  Supported formats: Program_X, program_X, Topic_X, topic_X, P_X, p_X, X_X (regulator), ProgramX, TopicX, X")

    # 4. Validate 'log2_fc_in_vs_out' is numeric
    try:
        fc_numeric = pd.to_numeric(df['log2_fc_in_vs_out'], errors='coerce')
        non_numeric_fc = df[fc_numeric.isna()]
        if not non_numeric_fc.empty:
            sample_bad = non_numeric_fc['log2_fc_in_vs_out'].head(5).tolist()
            warnings_list.append(f"Non-numeric values in 'log2_fc_in_vs_out': {sample_bad}")
            warnings_list.append(f"  Found {len(non_numeric_fc)} non-numeric log2FC values (will be ignored)")
    except Exception as e:
        warnings_list.append(f"Failed to validate 'log2_fc_in_vs_out' column: {e}")

    # 5. Validate 'fdr' is numeric and in valid range [0, 1]
    try:
        fdr_numeric = pd.to_numeric(df['fdr'], errors='coerce')
        non_numeric_fdr = df[fdr_numeric.isna()]
        if not non_numeric_fdr.empty:
            sample_bad = non_numeric_fdr['fdr'].head(5).tolist()
            warnings_list.append(f"Non-numeric values in 'fdr': {sample_bad}")
            warnings_list.append(f"  Found {len(non_numeric_fdr)} non-numeric FDR values (will be ignored)")

        # Check FDR range (should be 0-1 for proper FDR values)
        valid_fdr = fdr_numeric.dropna()
        if len(valid_fdr) > 0:
            out_of_range_mask = (valid_fdr < 0) | (valid_fdr > 1)
            if out_of_range_mask.any():
                out_of_range_vals = valid_fdr[out_of_range_mask].head(5).tolist()
                warnings_list.append(f"FDR values out of range [0, 1]: {out_of_range_vals}")
                warnings_list.append(f"  Found {out_of_range_mask.sum()} out-of-range FDR values")
    except Exception as e:
        warnings_list.append(f"Failed to validate 'fdr' column: {e}")

    # 6. Check for completely empty cell_type values
    empty_celltypes = df[df['cell_type'].isna() | (df['cell_type'].astype(str).str.strip() == '')]
    if not empty_celltypes.empty:
        warnings_list.append(f"Found {len(empty_celltypes)} rows with empty 'cell_type' values (will be ignored)")

    # 7. Log summary statistics (informational)
    n_programs = df['program'].nunique()
    n_celltypes = df['cell_type'].nunique()
    logger.info(f"Cell-type enrichment file summary: {file_path}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Unique programs: {n_programs}")
    logger.info(f"  Unique cell types: {n_celltypes}")

    # Check for reasonable data coverage (warnings only)
    if n_programs < 10:
        warnings_list.append(f"Very few programs found ({n_programs}). Expected 50-100 for typical cNMF results.")
    if n_celltypes < 3:
        warnings_list.append(f"Very few cell types found ({n_celltypes}). Expected multiple cell types.")

    # Log all warnings
    if warnings_list:
        logger.warning(f"Cell-type enrichment validation found {len(warnings_list)} issue(s):")
        for w in warnings_list:
            logger.warning(f"  {w}")
        return False
    else:
        logger.info(f"Cell-type enrichment validation passed: {file_path}")
        return True


def generate_celltype_summary(
    enrichment_file: Path,
    output_file: Path,
    thresholds: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    fdr_threshold: float = 0.05,
    topics: Optional[Set[int]] = None,
) -> int:
    """Generate cell-type annotations summary from raw enrichment data.

    Reads a cell-type enrichment CSV (e.g., from Seurat/Scanpy marker finding)
    and categorizes each program's cell-type associations by log2 fold-change.

    Args:
        enrichment_file: Path to raw enrichment CSV with columns:
            cell_type, program, log2_fc_in_vs_out, fdr
        output_file: Path to write summary CSV
        thresholds: Dict of category -> {log2fc_min, log2fc_max}.
            Uses CELLTYPE_THRESHOLDS if None.
        fdr_threshold: FDR cutoff for significance (default: 0.05)
        topics: Optional set of program IDs to include (None = all)

    Returns:
        Number of programs written

    Output format:
        program,highly_cell_type_specific,moderately_enriched,weakly_enriched,depleted
        Program_1,,,,
        Program_2,Large-artery,,BBB-high capillary,
        ...

    Each cell contains pipe-separated cell type names for that category.
    """
    if thresholds is None:
        thresholds = CELLTYPE_THRESHOLDS

    # Read enrichment data
    df = pd.read_csv(enrichment_file)
    logger.info(f"Loaded cell-type enrichment: {enrichment_file} ({len(df)} rows)")

    # Validate input (non-fatal, logs warnings)
    validate_celltype_enrichment(df, enrichment_file)

    # Validate required columns (critical check)
    required_cols = {'cell_type', 'program', 'log2_fc_in_vs_out', 'fdr'}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Enrichment file missing required columns: {missing}")
        logger.error(f"  Found columns: {sorted(df.columns)}")
        logger.error(f"  Cannot proceed without required columns. Please check your input file.")
        raise ValueError(f"Enrichment file missing required columns: {missing}")

    # Extract program ID using flexible parser (handles Program_X, program_X, Topic_X, X, etc.)
    df['program_id'] = df['program'].apply(extract_program_id)

    # Check for programs that couldn't be parsed
    unparsed = df[df['program_id'].isna()]
    if not unparsed.empty:
        logger.warning(f"Could not parse program IDs for {len(unparsed)} rows (will be excluded):")
        logger.warning(f"  Sample values: {unparsed['program'].head(5).tolist()}")
        df = df[df['program_id'].notna()].copy()

    # Ensure program_id is integer type
    df['program_id'] = df['program_id'].astype(int)

    # Filter to requested topics if specified
    if topics:
        df = df[df['program_id'].isin(list(topics))].copy()
        logger.info(f"Filtered to {len(df)} rows for topics: {sorted(topics)}")

    # Filter to significant results
    df_sig = df[df['fdr'] < fdr_threshold].copy()
    logger.info(f"Found {len(df_sig)} significant rows (FDR < {fdr_threshold})")

    # Use cell_type values as-is (assume already has correct names)
    df_sig['cell_type_display'] = df_sig['cell_type']

    # Categorize each row by log2FC thresholds
    def categorize_row(row: pd.Series) -> Optional[str]:
        log2fc = row['log2_fc_in_vs_out']
        for cat, bounds in thresholds.items():
            min_val = bounds.get('log2fc_min')
            max_val = bounds.get('log2fc_max')
            # Check if log2fc falls in this category
            if min_val is not None and max_val is not None:
                if min_val <= log2fc < max_val:
                    return cat
            elif min_val is not None:
                if log2fc >= min_val:
                    return cat
            elif max_val is not None:
                if log2fc <= max_val:
                    return cat
        return None

    df_sig['category'] = df_sig.apply(categorize_row, axis=1)
    df_categorized = df_sig.dropna(subset=['category'])
    logger.info(f"Categorized {len(df_categorized)} rows into enrichment categories")

    # Build summary: for each program, collect cell types per category
    all_programs = sorted(df['program_id'].unique())
    records = []

    for pid in all_programs:
        program_data = df_categorized[df_categorized['program_id'] == pid]
        row = {'program': f'Program_{pid}'}

        for cat in CELLTYPE_CATEGORIES:
            cat_data = program_data[program_data['category'] == cat]
            cell_types = sorted(cat_data['cell_type_display'].unique())
            row[cat] = '|'.join(cell_types) if cell_types else ''

        records.append(row)

    # Create output DataFrame
    summary_df = pd.DataFrame(records)
    summary_df = summary_df[['program'] + CELLTYPE_CATEGORIES]

    # Write output
    ensure_parent_dir(str(output_file))
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Wrote cell-type summary: {output_file} ({len(summary_df)} programs)")

    return len(summary_df)


# --------------------------- File / path helpers ------------------------------

def ensure_parent_dir(path_str: str) -> None:
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def resolve_program_id_column(df: pd.DataFrame) -> str:
    """
    Identify the program ID column using flexible matching.
    Supports: program_id, RowID, topic, Topic, etc. (case-insensitive)
    """
    mapper = ColumnMapper(df)
    try:
        actual_col = mapper.get_column('program_id', required=True)
        return actual_col
    except ValueError as e:
        raise ValueError(
            f"Could not find program ID column. {str(e)}\n"
            f"Supported names: program_id, RowID, topic, Topic, etc."
        )


def normalize_program_id(value: object) -> object:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value


# --------------------------- Extract top genes (CSV) --------------------------

def extract_top_genes_by_program(
    df: pd.DataFrame, n_top: int, id_col: str
) -> Dict[str, List[str]]:
    """
    Extract top-N genes per program using flexible column names.
    Standardizes Gene/Name, Score/Loading columns automatically.
    """
    mapper = ColumnMapper(df)

    # Get standardized column names
    try:
        cols = mapper.get_columns(['gene', 'score'], required=True)
        gene_col = cols['gene']
        score_col = cols['score']
    except ValueError as e:
        raise ValueError(f"Missing required columns for gene extraction: {e}")

    # Verify program ID column exists
    if id_col not in df.columns:
        raise ValueError(f"Program ID column '{id_col}' not found in DataFrame")

    top_map: Dict[str, List[str]] = {}
    for program_id, sub in df.groupby(id_col, sort=True):
        program_id_norm = normalize_program_id(program_id)
        program_key = str(program_id_norm)
        sub_sorted = sub.sort_values(score_col, ascending=False).head(n_top)
        genes = [str(g) for g in sub_sorted[gene_col].dropna().tolist()]
        seen = set()
        unique_genes: List[str] = []
        for g in genes:
            if g not in seen:
                seen.add(g)
                unique_genes.append(g)
        top_map[program_key] = unique_genes
    return top_map


# ----------------------------- Uniqueness table -------------------------------

def default_uniqueness_output(input_path: Path) -> Path:
    suffix = input_path.suffix or ".csv"
    stem = input_path.stem
    if stem.endswith("_with_uniqueness"):
        return input_path
    return input_path.with_name(f"{stem}_with_uniqueness{suffix}")


def build_uniqueness_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Build gene loading table with UniquenessScore using flexible column names.
    """
    mapper = ColumnMapper(df)

    # Get standardized column names
    try:
        cols = mapper.get_columns(['gene', 'score'], required=True)
        gene_col = cols['gene']
        score_col = cols['score']
    except ValueError as e:
        raise ValueError(f"Missing required columns for uniqueness computation: {e}")

    if id_col not in df.columns:
        raise ValueError(f"Program ID column '{id_col}' not found")

    work = df.copy()

    # Standardize column names to Name, Score, program_id
    rename_map = {
        gene_col: 'Name',
        score_col: 'Score',
        id_col: 'program_id'
    }
    work = work.rename(columns=rename_map)

    if "UniquenessScore" not in work.columns or work["UniquenessScore"].isna().all():
        work["Score"] = pd.to_numeric(work["Score"], errors="coerce")
        work["program_id"] = pd.to_numeric(work["program_id"], errors="coerce")
        valid = work.dropna(subset=["Name", "Score", "program_id"]).copy()
        if valid.empty:
            raise ValueError("No valid rows to compute UniquenessScore.")

        valid["program_id"] = valid["program_id"].astype(int)
        total_programs = valid["program_id"].nunique()
        gene_counts = valid.groupby("Name")["program_id"].nunique().astype(float)
        idf = np.log((total_programs + 1.0) / (gene_counts + 1.0))
        valid["UniquenessScore"] = valid["Score"] * valid["Name"].map(idf)

        work["UniquenessScore"] = np.nan
        work.loc[valid.index, "UniquenessScore"] = valid["UniquenessScore"]

    columns = ["Name", "Score", "program_id", "UniquenessScore"]
    out_df = work[columns].copy()
    out_df.dropna(subset=["Name", "Score", "program_id", "UniquenessScore"], inplace=True)
    return out_df


def build_overview_long_table(
    df: pd.DataFrame, top_map: Dict[str, List[str]], id_col: str
) -> pd.DataFrame:
    records = []
    sub_indexed_cache: Dict[object, pd.DataFrame] = {}
    for program_id_str, genes in top_map.items():
        program_id = normalize_program_id(program_id_str)
        if program_id not in sub_indexed_cache:
            sub_indexed_cache[program_id] = (
                df[df[id_col] == program_id].set_index("Name")
            )
        sub_idx = sub_indexed_cache[program_id]
        for rank, gene in enumerate(genes, start=1):
            score = float(sub_idx.loc[gene, "Score"]) if gene in sub_idx.index else float("nan")
            records.append(
                {"program_id": program_id, "rank": rank, "gene": gene, "score": score}
            )
    out_df = pd.DataFrame.from_records(records)
    if not out_df.empty:
        out_df.sort_values(["program_id", "rank"], inplace=True)
    return out_df


# ----------------------------- Config helpers ---------------------------------

DEFAULT_ENRICH_DIR = Path("input/enrichment")
DEFAULT_GENES_JSON_TEMPLATE = "genes_top{n_top}.json"
DEFAULT_GENES_OVERVIEW_TEMPLATE = "genes_overview_top{n_top}.csv"
DEFAULT_STRING_FULL = "string_enrichment_full.csv"
DEFAULT_STRING_FILTERED = "string_enrichment_filtered_process_kegg.csv"


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


def parse_topics(value: Optional[object]) -> Optional[Set[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return {int(v) for v in value}
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return {int(v) for v in items}
    return None


def apply_test_mode(
    args: object, config: Dict[str, Any], cli_overrides: Set[str]
) -> object:
    test_cfg = config.get("test", {}) if isinstance(config.get("test", {}), dict) else {}
    enabled = bool(test_cfg.get("enabled") or config.get("test_mode"))
    if not enabled:
        return args

    topics = test_cfg.get("topics") or test_cfg.get("programs") or config.get("test_programs")
    if hasattr(args, "topics") and "topics" not in cli_overrides and not getattr(args, "topics", None):
        if topics is not None:
            args.topics = topics  # type: ignore[attr-defined]
    return args


def apply_config_overrides(
    args: object,
    config: Dict[str, Any],
    cli_overrides: Set[str],
) -> object:
    steps_cfg = config.get("steps", {}) if isinstance(config.get("steps", {}), dict) else {}
    step_cfg = steps_cfg.get("string_enrichment", {})
    if isinstance(step_cfg, dict) and hasattr(args, "command") and getattr(args, "command") in step_cfg:
        step_cfg = step_cfg.get(getattr(args, "command"), {})
    if not isinstance(step_cfg, dict):
        return args

    for key, value in step_cfg.items():
        dest = key.replace("-", "_")
        if dest in cli_overrides:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
    return args


def apply_default_paths(args: object) -> object:
    n_top = getattr(args, "n_top", None) or 100

    if hasattr(args, "json_out") and not getattr(args, "json_out", None):
        args.json_out = str(DEFAULT_ENRICH_DIR / DEFAULT_GENES_JSON_TEMPLATE.format(n_top=n_top))  # type: ignore[attr-defined]
    if hasattr(args, "csv_out") and not getattr(args, "csv_out", None):
        args.csv_out = str(DEFAULT_ENRICH_DIR / DEFAULT_GENES_OVERVIEW_TEMPLATE.format(n_top=n_top))  # type: ignore[attr-defined]
    if hasattr(args, "out_csv_full") and not getattr(args, "out_csv_full", None):
        args.out_csv_full = str(DEFAULT_ENRICH_DIR / DEFAULT_STRING_FULL)  # type: ignore[attr-defined]
    if hasattr(args, "out_csv_filtered") and not getattr(args, "out_csv_filtered", None):
        args.out_csv_filtered = str(DEFAULT_ENRICH_DIR / DEFAULT_STRING_FILTERED)  # type: ignore[attr-defined]

    return args
