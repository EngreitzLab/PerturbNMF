#!/usr/bin/env python3
"""
Unified pipeline orchestrator for topic annotation workflow.
Runs all 5 steps sequentially with a simplified configuration.

Users only need to specify:
    - gene_loading: Path to the gene loading matrix CSV
    - celltype_enrichment: Optional raw cell-type enrichment CSV (summary auto-generated if provided)
    - regulator_file: Optional SCEPTRE regulator results CSV
    - output_dir: Directory for all outputs (intermediates auto-managed)

Key features:
- Single command to run the entire pipeline
- Auto-generates all intermediate file paths (including cell-type summary)
- Supports --start-from and --stop-after for partial runs
- Test mode with --topics to limit to specific programs
- Resume capability with cached API results

Each step is executed via subprocess, calling the numbered scripts
(01_genes_to_string_enrichment.py ... 05_generate_html_report.py) located
in the same directory as this file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    def load_dotenv() -> None:  # type: ignore[misc]
        pass

# Resolve the directory containing this script (and sibling step scripts).
_SCRIPT_DIR = Path(__file__).resolve().parent

# Add script directory to sys.path so that pipeline_state (and other sibling
# modules imported by step scripts) can be found when this file is executed
# as a standalone script.
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pipeline_state import (  # noqa: E402
    STEP_NAMES,
    PipelineState,
    StepState,
    compute_config_hash,
    init_state,
    load_state,
    mark_step,
    save_state,
)

# Load environment variables from .env file (supports NCBI_API_KEY, etc.)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pipeline steps in order
STEPS = [
    "string_enrichment",   # Step 1: Extract genes + STRING enrichment
    "literature_fetch",    # Step 2: Fetch NCBI/Harmonizome data
    "batch_prepare",       # Step 3a: Prepare LLM batch request
    "batch_submit",        # Step 3b: Submit to Vertex AI
    "parse_results",       # Step 4: Parse LLM responses
    "html_report",         # Step 5: Generate HTML report
]


@dataclass
class PipelineConfig:
    """Simplified pipeline configuration.

    User provides:
    - gene_loading: input gene loading CSV
    - celltype_enrichment: optional raw cell-type enrichment CSV (auto-generates summary if provided)
    - regulator_file: optional SCEPTRE regulator results CSV
    - output_dir: base output directory

    All intermediate paths are auto-generated.
    """
    # Required inputs (user must provide these)
    gene_loading: Path
    celltype_enrichment: Optional[Path]  # Optional raw enrichment file (summary auto-generated)
    regulator_file: Optional[Path]
    output_dir: Path

    # Optional settings with defaults
    topics: Optional[List[int]] = None  # None = all topics
    species: int = 10090  # Mouse by default
    keyword: str = '(endothelial OR endothelium OR "vascular endothelial")'
    annotation_role: str = "vascular-biology specialist"
    annotation_context: str = (
        "a gene program extracted from single-cell Perturb-seq of mouse brain "
        "endothelial cells (ECs)"
    )
    n_top_genes: int = 300
    top_loading: int = 20
    top_unique: int = 10
    top_enrichment: int = 3
    genes_per_term: int = 10
    top_positive_regulators: int = 3
    top_negative_regulators: int = 3
    regulator_significance_threshold: float = 0.05

    # LLM settings
    llm_backend: str = "anthropic"  # "anthropic" (default) or "vertex"
    llm_model: str = "claude-sonnet-4-5-20250929"  # Anthropic model name
    llm_max_tokens: int = 8192
    llm_wait: bool = False  # Default: do not wait; resume later

    # Vertex AI settings (only used if llm_backend="vertex")
    vertex_bucket: str = "gs://perturbseq/batch"

    # Gene summary options
    full_summaries: bool = False  # Use full HTML summaries (3x longer)

    # Resume/caching
    resume: bool = True

    # Auto-generated paths (computed from output_dir)
    _paths: Dict[str, Path] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Generate all intermediate paths from output_dir."""
        self.gene_loading = Path(self.gene_loading)
        if self.celltype_enrichment is not None:
            self.celltype_enrichment = Path(self.celltype_enrichment)
        self.output_dir = Path(self.output_dir)
        if self.regulator_file is not None:
            self.regulator_file = Path(self.regulator_file)

        # Create output subdirectories
        out = self.output_dir
        self._paths = {
            # Step 1 outputs
            "genes_json": out / "genes_top.json",
            "genes_overview": out / "genes_overview.csv",
            "enrichment_full": out / "string_enrichment" / "enrichment_full.csv",
            "enrichment_filtered": out / "string_enrichment" / "enrichment_filtered.csv",
            "enrichment_figures": out / "string_enrichment" / "figures",
            "enrichment_cache": out / "string_enrichment" / "cache",
            "gene_loading_with_uniqueness": out / "gene_loading_with_uniqueness.csv",
            "celltype_summary": out / "celltype_summary.csv",

            # Step 2 outputs
            "ncbi_json": out / "literature_context.json",
            "ncbi_csv": out / "literature_context.csv",

            # Step 3 outputs
            "batch_request_json": out / "llm_batches" / "batch_request.json",
            "batch_request_jsonl": out / "llm_batches" / "batch_request.jsonl",

            # Step 4 outputs
            "annotations_dir": out / "annotations",
            "summary_csv": out / "annotations" / "summary.csv",

            # Step 5 outputs
            "report_html": out / "annotations" / "report.html",
        }

    def get_path(self, key: str) -> Path:
        """Get an auto-generated path by key."""
        return self._paths[key]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from a dictionary (e.g., loaded from YAML)."""
        # Handle nested 'input' section
        if "input" in data:
            input_section = data["input"]
            data["gene_loading"] = input_section.get("gene_loading")
            data["celltype_enrichment"] = input_section.get("celltype_enrichment")
            data["regulator_file"] = input_section.get("regulator_file")

        # Filter to only known fields
        known_fields = {
            "gene_loading", "celltype_enrichment", "output_dir",
            "regulator_file", "topics", "species", "keyword",
            "annotation_role", "annotation_context",
            "n_top_genes", "top_loading", "top_unique",
            "top_enrichment", "genes_per_term",
            "top_positive_regulators", "top_negative_regulators",
            "regulator_significance_threshold",
            "llm_backend", "llm_model", "llm_max_tokens", "llm_wait",
            "vertex_bucket", "full_summaries", "resume"
        }
        if "keyword" not in data and "context" in data:
            data["keyword"] = data["context"]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load config from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise SystemExit("PyYAML is required. Install with: pip install pyyaml")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def run_step_1_string_enrichment(config: PipelineConfig) -> bool:
    """Step 1: Extract top genes and run STRING enrichment."""
    logger.info("=" * 60)
    logger.info("STEP 1: STRING Enrichment")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "01_genes_to_string_enrichment.py"

    cmd = [
        sys.executable, str(script), "all",
        "--input", str(config.gene_loading),
        "--n-top", str(config.n_top_genes),
        "--json-out", str(config.get_path("genes_json")),
        "--csv-out", str(config.get_path("genes_overview")),
        "--species", str(config.species),
        "--out-csv-full", str(config.get_path("enrichment_full")),
        "--out-csv-filtered", str(config.get_path("enrichment_filtered")),
        "--figures-dir", str(config.get_path("enrichment_figures")),
        "--cache-dir", str(config.get_path("enrichment_cache")),
        "--gene-loading-out", str(config.get_path("gene_loading_with_uniqueness")),
    ]

    if config.celltype_enrichment:
        cmd.extend([
            "--celltype-enrichment", str(config.celltype_enrichment),
            "--celltype-summary-out", str(config.get_path("celltype_summary")),
        ])
    else:
        logger.info("Cell-type enrichment file not provided; skipping cell-type summary generation.")

    if config.topics:
        cmd.extend(["--topics", ",".join(map(str, config.topics))])
    if config.resume:
        cmd.append("--resume")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Step 1 failed with return code %d", result.returncode)
        return False

    logger.info("Step 1 completed successfully")
    return True


def run_step_2_literature_fetch(config: PipelineConfig) -> bool:
    """Step 2: Fetch literature evidence from NCBI/Harmonizome."""
    logger.info("=" * 60)
    logger.info("STEP 2: Literature Fetch")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "02_fetch_ncbi_data.py"

    # Use the gene loading with uniqueness if it exists, otherwise original
    gene_input = config.get_path("gene_loading_with_uniqueness")
    if not gene_input.exists():
        gene_input = config.gene_loading

    cmd = [
        sys.executable, str(script),
        "--input", str(gene_input),
        "--keyword", config.keyword,
        "--json-out", str(config.get_path("ncbi_json")),
        "--csv-out", str(config.get_path("ncbi_csv")),
        "--gene-summary-source", "harmonizome",
        # Fetch summaries for exactly top_loading + top_unique genes
        "--top-loading", str(config.top_loading),
        "--top-unique", str(config.top_unique),
    ]

    # Optional: Use full HTML summaries (3x longer)
    if config.full_summaries:
        cmd.append("--full-summaries")

    if config.regulator_file and config.regulator_file.exists():
        cmd.extend(["--regulator-file", str(config.regulator_file)])
        cmd.extend([
            "--top-positive-regulators", str(config.top_positive_regulators),
            "--top-negative-regulators", str(config.top_negative_regulators),
            "--regulator-significance-threshold", str(config.regulator_significance_threshold),
        ])
    if config.topics:
        cmd.extend(["--topics", ",".join(map(str, config.topics))])

    # Pass NCBI API key if available in environment
    ncbi_api_key = os.environ.get("NCBI_API_KEY")
    if ncbi_api_key:
        cmd.extend(["--api-key", ncbi_api_key])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Step 2 failed with return code %d", result.returncode)
        return False

    logger.info("Step 2 completed successfully")
    return True


def run_step_3a_batch_prepare(config: PipelineConfig) -> bool:
    """Step 3a: Prepare LLM batch request."""
    logger.info("=" * 60)
    logger.info("STEP 3a: Prepare Batch Request")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "03_submit_and_monitor_batch.py"

    # Use gene loading with uniqueness if available
    gene_input = config.get_path("gene_loading_with_uniqueness")
    if not gene_input.exists():
        gene_input = config.gene_loading

    # Use the generated celltype summary (output of step 1) if it exists
    celltype_summary = config.get_path("celltype_summary")

    search_keyword = config.keyword
    cmd = [
        sys.executable, str(script), "prepare",
        "--gene-file", str(gene_input),
        "--celltype-dir", str(celltype_summary.parent),  # Fallback directory
        "--enrichment-file", str(config.get_path("enrichment_filtered")),
        "--ncbi-file", str(config.get_path("ncbi_json")),
        "--output-file", str(config.get_path("batch_request_json")),
        "--top-loading", str(config.top_loading),
        "--top-unique", str(config.top_unique),
        "--top-enrichment", str(config.top_enrichment),
        "--genes-per-term", str(config.genes_per_term),
        "--annotation-role", str(config.annotation_role),
        "--annotation-context", str(config.annotation_context),
        "--search-keyword", str(search_keyword),
    ]

    if celltype_summary.exists():
        cmd.extend(["--celltype-file", str(celltype_summary)])

    if config.regulator_file:
        cmd.extend([
            "--regulator-file", str(config.regulator_file),
            "--top-positive-regulators", str(config.top_positive_regulators),
            "--top-negative-regulators", str(config.top_negative_regulators),
            "--regulator-significance-threshold", str(config.regulator_significance_threshold),
        ])
    if config.topics:
        cmd.extend(["--topics", ",".join(map(str, config.topics))])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Step 3a failed with return code %d", result.returncode)
        return False

    logger.info("Step 3a completed successfully")
    return True


def run_step_3b_batch_submit(config: PipelineConfig) -> Optional[tuple[str, Optional[str], bool]]:
    """Step 3b: Submit batch to LLM backend.

    Returns (batch_id, results_path, waiting)
    waiting = True when submission succeeded but batch still running.
    """
    backend = config.llm_backend

    logger.info("=" * 60)
    logger.info(f"STEP 3b: Submit to {backend.upper()}")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "03_submit_and_monitor_batch.py"

    def short_model_name(full: str) -> str:
        parts = full.split("-")
        if parts and parts[-1].isdigit() and len(parts[-1]) == 8:
            return "-".join(parts[:-1])
        return full

    if backend == "vertex":
        # Vertex AI submission
        cmd = [
            sys.executable,
            str(script),
            "submit-vertex",
            str(config.get_path("batch_request_json")),
            "--model",
            short_model_name(config.llm_model),  # Vertex uses short names
            "--bucket",
            config.vertex_bucket,
        ]
    else:
        # Anthropic single-message calls (no Batch API)
        results_jsonl = config.get_path("batch_request_json").with_name(
            config.get_path("batch_request_json").stem + "_results.jsonl"
        )
        cmd = [
            sys.executable,
            str(script),
            "submit-single",
            str(config.get_path("batch_request_json")),
            "--model",
            config.llm_model,
            "--max-tokens",
            str(config.llm_max_tokens),
            "--output",
            str(results_jsonl),
        ]

    if backend == "vertex" and config.llm_wait:
        cmd.append("--wait")

    logger.info(f"Running: {' '.join(cmd)}")

    # Capture output to get the GCS output path
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Step 3b failed with return code %d", result.returncode)
        logger.error("stderr: %s", result.stderr)
        return None

    # Print stdout for visibility (since we captured it)
    if result.stdout:
        for line in result.stdout.splitlines():
            print(line)

    results_path: Optional[str] = None
    waiting = False
    batch_id: Optional[str] = None

    if backend == "vertex":
        # Read batch ID (Vertex stores job name)
        batch_id_file = config.get_path("batch_request_json").with_suffix(".batch_id")
        batch_id = batch_id_file.read_text(encoding="utf-8").strip() if batch_id_file.exists() else None
        waiting = not config.llm_wait

        # Vertex AI: look for GCS prefix
        output_base = None
        for line in result.stdout.splitlines():
            if "Results available at:" in line or "Output:" in line:
                match = re.search(r'(gs://[^\s]+)', line)
                if match:
                    output_base = match.group(1).rstrip('/')
                    break

        output_path = None
        if output_base:
            try:
                list_cmd = ["gcloud", "storage", "ls", output_base + "/"]
                list_result = subprocess.run(list_cmd, capture_output=True, text=True)
                if list_result.returncode == 0:
                    for gcs_line in list_result.stdout.splitlines():
                        if "prediction-model-" in gcs_line:
                            output_path = gcs_line.rstrip('/')
                            break
                if not output_path:
                    output_path = output_base
            except Exception as e:
                logger.warning(f"Could not list GCS directory: {e}")
                output_path = output_base

        if output_path:
            results_path = output_path

        if config.llm_wait and not results_path:
            candidate = config.get_path("batch_request_json").with_name(
                config.get_path("batch_request_json").stem + "_results.jsonl"
            )
            if candidate.exists():
                results_path = str(candidate)
    else:
        # Anthropic single-message: results JSONL already written by submit-single
        batch_id = "single_msg"
        waiting = False
        results_jsonl_path = config.get_path("batch_request_json").with_name(
            config.get_path("batch_request_json").stem + "_results.jsonl"
        )
        if results_jsonl_path.exists():
            results_path = str(results_jsonl_path)

    logger.info("Step 3b completed successfully")
    if results_path:
        logger.info(f"Results location: {results_path}")

    return batch_id or "", results_path, waiting


def run_step_4_parse_results(config: PipelineConfig, results_path: Optional[str] = None) -> bool:
    """Step 4: Parse LLM responses and generate summary.

    Args:
        config: Pipeline configuration
        results_path: Path to results - either:
            - GCS prefix (gs://...) for Vertex AI
            - Local JSONL file path for Anthropic
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Parse Results")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "04_parse_and_summarize.py"

    # Use gene loading with uniqueness if available
    gene_input = config.get_path("gene_loading_with_uniqueness")
    if not gene_input.exists():
        gene_input = config.gene_loading

    cmd = [
        sys.executable, str(script),
        "--markdown-dir", str(config.get_path("annotations_dir")),
        "--summary-csv", str(config.get_path("summary_csv")),
        "--gene-loading-file", str(gene_input),
    ]

    if results_path:
        if results_path.startswith("gs://"):
            # Vertex AI: use GCS prefix
            cmd.extend(["--gcs-prefix", results_path])
        else:
            # Anthropic: use local file
            cmd.extend(["--results-jsonl", results_path])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Step 4 failed with return code %d", result.returncode)
        return False

    logger.info("Step 4 completed successfully")
    return True


def run_step_5_html_report(config: PipelineConfig) -> bool:
    """Step 5: Generate interactive HTML report."""
    logger.info("=" * 60)
    logger.info("STEP 5: Generate HTML Report")
    logger.info("=" * 60)

    script = _SCRIPT_DIR / "05_generate_html_report.py"

    # Use gene loading with uniqueness if available
    gene_input = config.get_path("gene_loading_with_uniqueness")
    if not gene_input.exists():
        gene_input = config.gene_loading

    cmd = [
        sys.executable, str(script),
        "--summary-csv", str(config.get_path("summary_csv")),
        "--annotations-dir", str(config.get_path("annotations_dir")),
        "--enrichment-dir", str(config.get_path("enrichment_figures")),
        "--gene-loading-csv", str(gene_input),
        "--output-html", str(config.get_path("report_html")),
    ]

    if config.regulator_file and config.regulator_file.exists():
        cmd.extend([
            "--volcano-csv", str(config.regulator_file),
            "--regulator-significance-threshold", str(config.regulator_significance_threshold),
        ])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Step 5 failed with return code %d", result.returncode)
        return False

    logger.info("Step 5 completed successfully")
    return True


def run_pipeline(
    config: PipelineConfig,
    start_from: Optional[str] = None,
    stop_after: Optional[str] = None,
    gcs_prefix: Optional[str] = None,
    force_restart: bool = False,
    restart_from: Optional[str] = None,
) -> bool:
    """Run the full pipeline or a subset of steps.

    Args:
        config: Pipeline configuration
        start_from: Step to start from (skips earlier steps)
        stop_after: Step to stop after (skips later steps)
        gcs_prefix: GCS prefix for step 4 (if resuming from batch results)
        force_restart: Ignore existing state and restart
        restart_from: Re-run from the specified step onward

    Returns:
        True if all requested steps completed successfully
    """
    # Validate step names
    for name in [start_from, stop_after, restart_from]:
        if name and name not in STEPS:
            logger.error(f"Unknown step: {name}. Valid steps: {STEPS}")
            return False

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load or initialize pipeline state
    # ------------------------------------------------------------------
    state_path = config.output_dir / "pipeline_state.json"
    config_hash = compute_config_hash(asdict(config))
    state = load_state(state_path)

    if state:
        if state.config_hash != config_hash and not force_restart:
            logger.error(
                "Config has changed since last run. Rerun with --force-restart to overwrite existing outputs."
            )
            return False
        if state.config_hash != config_hash and force_restart:
            logger.warning("Config changed; restarting pipeline and overwriting prior outputs after warning.")
            state = init_state(config_hash, config.topics)
            save_state(state_path, state)
    else:
        state = init_state(config_hash, config.topics)
        save_state(state_path, state)

    # If restart_from provided, reset steps from that step onward to pending
    if restart_from:
        logger.warning(
            "--restart-from supplied; steps from %s onward will be re-run and may overwrite previous outputs.",
            restart_from,
        )
        restart_idx = STEPS.index(restart_from)
        for step in STEPS[restart_idx:]:
            mark_step(state, step, "pending")
        save_state(state_path, state)

    # Determine which steps to run based on completed status and start/stop
    start_idx = STEPS.index(start_from) if start_from else 0
    stop_idx = STEPS.index(stop_after) if stop_after else len(STEPS) - 1
    requested_steps = STEPS[start_idx : stop_idx + 1]

    # Skip completed steps unless restart_from explicitly reset them
    steps_to_run: List[str] = []
    skipped_steps: List[str] = []
    for step in requested_steps:
        step_state = state.steps.get(step)
        if step_state and step_state.status == "completed":
            skipped_steps.append(step)
            continue
        steps_to_run.append(step)

    logger.info("=" * 60)
    logger.info("PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)
    logger.info(f"Output directory: {config.output_dir}")
    if skipped_steps:
        logger.info(f"Skipping completed steps: {skipped_steps}")
    logger.info(f"Steps to run: {steps_to_run}")
    if config.topics:
        logger.info(f"Topics: {config.topics}")
    if state.batch.get("batch_id"):
        logger.info(f"Existing batch ID: {state.batch['batch_id']}")
    logger.info("=" * 60)

    batch_results_path: Optional[str] = None

    # If batch already submitted in previous run, try to resume
    batch_info = state.batch or {}
    if batch_info.get("results_path"):
        batch_results_path = batch_info.get("results_path")

    # ------------------------------------------------------------------
    # Execute steps
    # ------------------------------------------------------------------
    for step in steps_to_run:
        mark_step(state, step, "in_progress")
        save_state(state_path, state)

        if step == "string_enrichment":
            if not run_step_1_string_enrichment(config):
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False
            mark_step(state, step, "completed")
            save_state(state_path, state)
            continue

        if step == "literature_fetch":
            if not run_step_2_literature_fetch(config):
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False
            mark_step(state, step, "completed")
            save_state(state_path, state)
            continue

        if step == "batch_prepare":
            if not run_step_3a_batch_prepare(config):
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False
            mark_step(state, step, "completed")
            save_state(state_path, state)
            continue

        if step == "batch_submit":
            # Submit batch and exit (fire-and-forget pattern)
            # Status checking and result download moved to parse_results step
            submission = run_step_3b_batch_submit(config)
            if submission is None:
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False

            batch_id, results_path, waiting = submission
            if not batch_id:
                logger.error("Batch submission did not return a batch ID.")
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False

            state.batch["batch_id"] = batch_id
            if results_path:
                batch_results_path = results_path
                state.batch["results_path"] = results_path

            if waiting:
                # Batch submitted but not complete - exit cleanly
                mark_step(state, step, "completed", {"batch_id": batch_id})
                save_state(state_path, state)
                logger.info(
                    "Batch submitted (%s). Rerun this command later to check status and continue.",
                    batch_id,
                )
                return True

            mark_step(state, step, "completed", {"batch_id": batch_id})
            save_state(state_path, state)
            continue

        if step == "parse_results":
            # First, try to get results path from state or args
            results_path = batch_results_path or gcs_prefix
            if state.batch.get("results_path"):
                results_path = state.batch.get("results_path")

            # If no results path but batch_id exists, check/fetch batch status
            if not results_path and state.batch.get("batch_id"):
                batch_id = state.batch["batch_id"]
                logger.info("No results path found. Checking batch status for %s...", batch_id)

                from batch_utils import check_anthropic_batch, check_vertex_batch

                if config.llm_backend == "vertex":
                    status, fetched_path = check_vertex_batch(
                        batch_id,
                        f"gs://{config.vertex_bucket}/batch_output",
                    )
                else:
                    # Anthropic
                    results_dest = config.get_path("batch_request_json").with_name(
                        config.get_path("batch_request_json").stem + "_results.jsonl"
                    )
                    status, fetched_path = check_anthropic_batch(batch_id, str(results_dest))

                if status == "in_progress":
                    logger.info("Batch %s still processing. Rerun later to continue.", batch_id)
                    save_state(state_path, state)
                    return True  # Exit cleanly, not an error

                if status == "failed":
                    logger.error("Batch %s failed.", batch_id)
                    mark_step(state, step, "failed")
                    save_state(state_path, state)
                    return False

                # status == "completed"
                if fetched_path:
                    results_path = fetched_path
                    state.batch["results_path"] = fetched_path
                    save_state(state_path, state)
                    logger.info("Batch results downloaded to: %s", fetched_path)

            if not results_path:
                logger.error("No batch results available. Provide --gcs-prefix or ensure batch completed.")
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False

            if not run_step_4_parse_results(config, results_path):
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False

            mark_step(state, step, "completed")
            save_state(state_path, state)
            continue

        if step == "html_report":
            if not run_step_5_html_report(config):
                mark_step(state, step, "failed")
                save_state(state_path, state)
                return False
            mark_step(state, step, "completed")
            save_state(state_path, state)
            continue

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Report: {config.get_path('report_html')}")

    return True


def parse_topics_arg(value: str) -> List[int]:
    """Parse comma-separated topic IDs."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run the topic annotation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with config file
  python run_pipeline.py --config configs/pipeline_config.yaml

  # Full run with CLI arguments
  python run_pipeline.py \\
    --gene-loading input/genes/FB_moi15_seq2_loading_gene_k100_top300.csv \\
    --celltype-enrichment input/celltype/fp_seq2_cnmf_celltype_l2_program_enrichment.csv \\
    --regulator-file input/regulators/sceptre_discovery_analysis_results.csv \\
    --output-dir results/output/my_run

  # Test run with specific topics
  python run_pipeline.py --config configs/pipeline_config.yaml --topics 5,6,8,11,18

  # Partial run: prepare batch only (steps 1-3a)
  python run_pipeline.py --config configs/pipeline_config.yaml --stop-after batch_prepare

  # Resume from parsing (provide GCS prefix from batch job)
  python run_pipeline.py --config configs/pipeline_config.yaml \\
    --start-from parse_results \\
    --gcs-prefix gs://perturbseq/batch/prediction-model-2024...

Steps:
  string_enrichment  Step 1: Extract genes + STRING enrichment + cell-type summary
  literature_fetch   Step 2: Fetch NCBI/Harmonizome data
  batch_prepare      Step 3a: Prepare LLM batch request
  batch_submit       Step 3b: Submit to LLM backend (Anthropic/Vertex)
  parse_results      Step 4: Parse LLM responses
  html_report        Step 5: Generate HTML report
"""
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML config file"
    )

    # Required inputs (can be overridden by CLI)
    parser.add_argument(
        "--gene-loading",
        help="Path to gene loading matrix CSV"
    )
    parser.add_argument(
        "--celltype-enrichment",
        help="Path to raw cell-type enrichment CSV (summary auto-generated)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for all results"
    )

    # Optional inputs
    parser.add_argument(
        "--regulator-file",
        help="Path to SCEPTRE regulator results CSV"
    )

    # Step control
    parser.add_argument(
        "--start-from",
        choices=STEPS,
        help="Step to start from (skips earlier steps)"
    )
    parser.add_argument(
        "--stop-after",
        choices=STEPS,
        help="Step to stop after (skips later steps)"
    )
    parser.add_argument(
        "--gcs-prefix",
        help="GCS prefix for batch results (for resuming at parse_results)"
    )

    # Optional settings
    parser.add_argument(
        "--topics",
        help="Comma-separated list of topic IDs to process (default: all)"
    )
    parser.add_argument(
        "--species",
        type=int,
        default=10090,
        help="NCBI taxonomy ID (default: 10090 for mouse)"
    )
    parser.add_argument(
        "--keyword",
        help="PubMed search keyword for your tissue/cell type"
    )
    parser.add_argument(
        "--annotation-role",
        help="Specialist role used in the LLM prompt header"
    )
    parser.add_argument(
        "--annotation-context",
        help=(
            "Dataset/cell-type description inserted into the LLM prompt header "
            "(e.g., 'a gene program extracted from single-cell RNA-seq of microglia')"
        ),
    )
    parser.add_argument(
        "--top-positive-regulators",
        type=int,
        help="Number of positive regulators to include per program",
    )
    parser.add_argument(
        "--top-negative-regulators",
        type=int,
        help="Number of negative regulators to include per program",
    )
    parser.add_argument(
        "--regulator-significance-threshold",
        type=float,
        help="Adjusted p-value threshold used when the regulator file has no 'significant' column",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume/caching (re-query all APIs)"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch completion (default: submit and exit; resume later)")
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Ignore existing state (warn) and restart pipeline, potentially overwriting outputs"
    )
    parser.add_argument(
        "--restart-from",
        choices=STEPS,
        help="Re-run from the specified step (warns about overwriting downstream outputs)"
    )

    args = parser.parse_args()

    # Load config from file or build from CLI args
    if args.config:
        config = PipelineConfig.from_yaml(Path(args.config))
    else:
        # Require CLI arguments if no config file
        if not args.gene_loading or not args.output_dir:
            parser.error(
                "Either --config or both --gene-loading and --output-dir are required"
            )
        config = PipelineConfig(
            gene_loading=Path(args.gene_loading),
            celltype_enrichment=Path(args.celltype_enrichment) if args.celltype_enrichment else None,
            regulator_file=Path(args.regulator_file) if args.regulator_file else None,
            output_dir=Path(args.output_dir),
            top_positive_regulators=args.top_positive_regulators or 3,
            top_negative_regulators=args.top_negative_regulators or 3,
            regulator_significance_threshold=args.regulator_significance_threshold or 0.05,
        )

    # Apply CLI overrides
    if args.gene_loading:
        config.gene_loading = Path(args.gene_loading)
    if args.celltype_enrichment:
        config.celltype_enrichment = Path(args.celltype_enrichment)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.__post_init__()  # Regenerate paths
    if args.regulator_file:
        config.regulator_file = Path(args.regulator_file)
    if args.topics:
        config.topics = parse_topics_arg(args.topics)
    if args.species:
        config.species = args.species
    if args.keyword:
        config.keyword = args.keyword
    if args.annotation_role:
        config.annotation_role = args.annotation_role
    if args.annotation_context:
        config.annotation_context = args.annotation_context
    if args.top_positive_regulators is not None:
        config.top_positive_regulators = args.top_positive_regulators
    if args.top_negative_regulators is not None:
        config.top_negative_regulators = args.top_negative_regulators
    if args.regulator_significance_threshold is not None:
        config.regulator_significance_threshold = args.regulator_significance_threshold
    if args.no_resume:
        config.resume = False
    if args.wait:
        config.llm_wait = True

    # Validate inputs exist
    if not config.gene_loading.exists():
        logger.error(f"Gene loading file not found: {config.gene_loading}")
        return 1
    if config.celltype_enrichment and not config.celltype_enrichment.exists():
        logger.warning(
            "Cell-type enrichment file not found: %s (continuing without cell-type context)",
            config.celltype_enrichment,
        )
        config.celltype_enrichment = None
    if config.regulator_file and not config.regulator_file.exists():
        logger.warning(
            "Regulator file not found: %s (continuing without regulator context/volcano plots)",
            config.regulator_file,
        )
        config.regulator_file = None

    # Run pipeline
    success = run_pipeline(
        config=config,
        start_from=args.start_from,
        stop_after=args.stop_after,
        gcs_prefix=args.gcs_prefix,
        force_restart=args.force_restart,
        restart_from=args.restart_from,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
