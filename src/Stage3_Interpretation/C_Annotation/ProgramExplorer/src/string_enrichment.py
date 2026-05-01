"""
STRING enrichment API helpers split from 01_genes_to_string_enrichment.py.

Provides:
- STRING functional enrichment API calls with retries
- Per-program JSON caching for enrichment results
- Full and filtered (Process/KEGG) CSV construction
- Enrichment figure download from STRING API
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ----------------------------- Constants --------------------------------------

STRING_ENRICH_ENDPOINT = "https://string-db.org/api/json/enrichment"

# ----------------------------- Caching ----------------------------------------


def cache_path(cache_dir: Path, program_id: int) -> Path:
    return cache_dir / f"program_{program_id}_enrichment.json"


def load_cached_results(cache_dir: Path, program_id: int) -> Optional[List[Dict[str, Any]]]:
    cache_file = cache_path(cache_dir, program_id)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Cache file is invalid JSON: %s", cache_file)
        return None
    if not isinstance(data, list):
        logger.warning("Cache file has unexpected format: %s", cache_file)
        return None
    return data


def write_cached_results(cache_dir: Path, program_id: int, results: List[Dict[str, Any]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path(cache_dir, program_id)
    cache_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


# ----------------------------- API calls --------------------------------------


def call_string_enrichment(
    genes: List[str], species: int, retries: int = 3, sleep_between: float = 0.6
) -> List[Dict[str, Any]]:
    identifiers_value = "\r".join(genes)
    params = {
        "identifiers": identifiers_value,
        "species": species,
        "caller_identity": "topic_analysis_string_enrichment",
    }

    attempt = 0
    while attempt <= retries:
        try:
            response = requests.get(STRING_ENRICH_ENDPOINT, params=params, timeout=60)
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as json_err:
                    logger.error(f"Failed to parse JSON (n={len(genes)}): {json_err}")
                    data = []
                return data if isinstance(data, list) else []
            else:
                logger.warning(f"STRING returned status {response.status_code}: {response.text[:200]}")
        except requests.RequestException as e:
            logger.warning(f"HTTP error on STRING request (attempt {attempt+1}/{retries+1}): {e}")

        attempt += 1
        time.sleep(min(2.0, sleep_between * (attempt + 1)))

    return []


# ----------------------------- CSV builders -----------------------------------


def build_full_csv(program_to_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pid, terms in program_to_results.items():
        for t in terms:
            rows.append(
                {
                    "program_id": int(pid),
                    "category": str(t.get("category", "NA")),
                    "term": str(t.get("term", t.get("description", "NA"))),
                    "term_id": str(t.get("term_id", "NA")),
                    "description": str(t.get("description", t.get("term", "NA"))),
                    "fdr": float(t.get("fdr", float("nan"))),
                    "p_value": float(t.get("p_value", float("nan"))),
                    "number_of_genes": int(t.get("number_of_genes", 0)),
                    "number_of_genes_in_background": int(t.get("number_of_genes_in_background", 0)),
                    "ncbiTaxonId": int(t.get("ncbiTaxonId", 0)),
                    "inputGenes": "|".join(t.get("inputGenes", [])) if t.get("inputGenes") else "",
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["program_id", "fdr", "p_value"], inplace=True)
    return df


def filter_process_kegg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    category_mask = df["category"].str.contains("Process|KEGG", case=False, na=False)
    background_mask = df["number_of_genes_in_background"] < 500
    filtered_df = df[category_mask & background_mask].copy()
    if not filtered_df.empty:
        filtered_df.sort_values(["program_id", "fdr", "p_value"], inplace=True)
    return filtered_df


# ----------------------------- Figure download --------------------------------


def download_string_enrichment_figure(
    genes: List[str],
    species: int,
    category: str,
    output_path: Path,
    retries: int = 3,
    resume: bool = False,
) -> bool:
    """Download enrichment figure directly from STRING API.

    Args:
        genes: List of gene identifiers
        species: NCBI taxonomy ID
        category: Enrichment category (e.g., "Process", "KEGG")
        output_path: Path to save the figure
        retries: Number of retry attempts
        resume: If True, skip download when output_path already exists

    Returns:
        True if successful, False otherwise
    """
    if not genes:
        return False

    if resume and output_path.exists():
        return True

    # STRING enrichment figure endpoint
    base_url = "https://string-db.org/api/image/enrichmentfigure"

    # Prepare parameters
    identifiers_value = "\r".join(genes)
    params = {
        "identifiers": identifiers_value,
        "species": species,
        "category": category,
        "caller_identity": "topic_analysis_string_enrichment",
    }

    attempt = 0
    while attempt <= retries:
        try:
            response = requests.get(base_url, params=params, timeout=120)
            if response.status_code == 200:
                # Check if response is actually an image
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    logger.warning(
                        "STRING returned non-image content for category %s: %s",
                        category,
                        content_type,
                    )
                    return False
            else:
                logger.warning(
                    "STRING figure API returned status %s", response.status_code
                )
        except requests.RequestException as e:
            logger.warning(
                "HTTP error downloading figure (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )

        attempt += 1
        time.sleep(min(3.0, 1.0 * (attempt + 1)))

    return False
