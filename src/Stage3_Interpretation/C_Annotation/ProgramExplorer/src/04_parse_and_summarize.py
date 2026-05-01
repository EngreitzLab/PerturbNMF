"""
/**
 * @description 
 * Parse Anthropic batch JSONL results into per-topic markdown files AND generate
 * a summary CSV with unique topic names, keywords, top genes, and brief summaries.
 * 
 * This merges the previous Step 03 (parse results) and Step 04 (generate topic
 * summary) into a single convenient CLI. By default it performs both actions.
 * You can also run only one of the steps using flags.
 * 
 * Key features:
 * - Robust JSONL parsing with per-topic markdown outputs
 * - Unique, descriptive names per topic based on keywords and optional top genes
 * - Optional inclusion of top genes from a gene loading CSV (columns: Name, Score, RowID)
 * 
 * @dependencies
 * - json, os, re, argparse
 * - pandas (for CSV writing and optional gene loading)
 * 
 * @examples
 * - End-to-end (parse → summarize):
 *   python pipeline/04_parse_and_summarize.py \
 *     --results-jsonl results/output/annotations/batch_results.jsonl \
 *     --markdown-dir results/output/annotations \
 *     --summary-csv results/output/annotations/summary.csv \
 *     --gene-loading-file input/genes/FB_moi15_seq2_loading_gene_k100_top300.csv
 * 
 * - Parse only:
 *   python pipeline/04_parse_and_summarize.py \
 *     --results-jsonl <results.jsonl> --markdown-dir <md_dir> --no-summary
 * 
 * - Summarize only (assumes markdown files already exist):
 *   python pipeline/04_parse_and_summarize.py \
 *     --markdown-dir <md_dir> --summary-csv <summary.csv> --no-parse \
 *     --gene-loading-file <genes.csv>
 */
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

"""
@description
Configuration loader for parsing and summarizing batch results.
It is responsible for reading JSON/YAML configs and applying per-step defaults
with CLI override precedence.

Key features:
- Supports JSON and YAML (if PyYAML is installed).
- Applies config values when CLI flags are omitted.

@dependencies
- json: Built-in JSON parser
- yaml (optional): YAML parser when available
- sys: CLI inspection for override detection
"""


def load_config(config_path: str | None) -> Dict[str, object]:
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


def get_cli_overrides(argv: List[str]) -> set[str]:
    overrides: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overrides.add(name.replace("-", "_"))
    return overrides


def apply_config_overrides(
    args: argparse.Namespace, config: Dict[str, object], cli_overrides: set[str]
) -> argparse.Namespace:
    steps_cfg = config.get("steps", {}) if isinstance(config.get("steps", {}), dict) else {}
    step_cfg = steps_cfg.get("parse_results", {})
    if not isinstance(step_cfg, dict):
        return args

    for key, value in step_cfg.items():
        dest = str(key).replace("-", "_")
        if dest in cli_overrides:
            continue
        if hasattr(args, dest):
            setattr(args, dest, value)
    return args


def download_from_gcs(gcs_prefix: str, local_dir: str) -> List[str]:
    """Download all files from a GCS prefix to a local directory.
    
    Returns list of downloaded file paths.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    # Ensure prefix ends with * for glob-like behavior
    if not gcs_prefix.endswith("*"):
        gcs_prefix = gcs_prefix.rstrip("/") + "/*"
    
    print(f"Downloading from {gcs_prefix} to {local_dir}...")
    
    cmd = ["gcloud", "storage", "cp", gcs_prefix, local_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ✗ GCS download failed: {result.stderr}")
        return []
    
    # Find downloaded files
    downloaded = glob.glob(os.path.join(local_dir, "*"))
    print(f"  ✓ Downloaded {len(downloaded)} files")
    return downloaded


def parse_final_results(result_file: str, output_dir: str) -> List[int]:
    """Parse the final batch result JSONL and write per-topic markdown files.

    Supports both Anthropic direct API and Vertex AI response formats:
    - Anthropic: {"custom_id": "...", "result": {"type": "succeeded", "message": {"content": [{"text": "..."}]}}}
    - Vertex AI: {"custom_id": "...", "response": {"content": [{"text": "..."}]}}
    
    Returns a list of topic IDs successfully written.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing {result_file}...")
    saved_ids: List[int] = []

    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                if custom_id and isinstance(custom_id, str) and custom_id.startswith("topic_"):
                    # expected format: topic_<num>_annotation
                    m = re.match(r"topic_(\d+)", custom_id)
                    if not m:
                        continue
                    topic_number = int(m.group(1))

                    text_content = ""
                    
                    # Try Vertex AI format first: {"response": {"content": [{"text": "..."}]}}
                    response = data.get("response", {})
                    if response:
                        content = response.get("content", [])
                        if isinstance(content, list) and content:
                            first = content[0]
                            text_content = first.get("text", "") if isinstance(first, dict) else ""
                    
                    # Fallback to Anthropic format: {"result": {"type": "succeeded", "message": {"content": [...]}}}
                    if not text_content:
                        result = data.get("result", {})
                        if result.get("type") == "succeeded":
                            message = result.get("message", {})
                            content = message.get("content", [])
                            if isinstance(content, list) and content:
                                first = content[0]
                                text_content = first.get("text", "") if isinstance(first, dict) else ""
                        elif result.get("type") == "errored":
                            error = result.get("error", {})
                            print(f"  ✗ Topic {topic_number} failed. Reason: {error.get('message', 'Unknown error')}")
                            continue

                    if text_content:
                        output_filename = os.path.join(output_dir, f"topic_{topic_number}_annotation.md")
                        with open(output_filename, "w", encoding="utf-8") as out_f:
                            out_f.write(text_content)
                        print(f"  ✓ Saved Topic {topic_number}")
                        saved_ids.append(topic_number)
                    else:
                        print(f"  ✗ Topic {topic_number} had empty content")
            except json.JSONDecodeError:
                print(f"  ✗ Could not parse line: {line.strip()}")
            except Exception as e:
                print(f"  ✗ Error processing line: {e}")

    return saved_ids


def load_top_genes_by_topic(gene_loading_file: str, top_n: int = 10) -> Dict[int, List[str]]:
    """Load top genes by RowID from gene loading CSV."""
    top_genes_by_topic: Dict[int, List[str]] = {}
    if not gene_loading_file or not os.path.exists(gene_loading_file):
        return top_genes_by_topic

    print(f"Loading gene data from {gene_loading_file}...")
    try:
        gene_df = pd.read_csv(gene_loading_file)
        if "program_id" in gene_df.columns:
            group_col = "program_id"
        elif "RowID" in gene_df.columns:
            group_col = "RowID"
        else:
             print("  ✗ CSV missing 'program_id' or 'RowID' column")
             return top_genes_by_topic

        for topic_id, group in gene_df.groupby(group_col):
            genes = (
                group.sort_values("Score", ascending=False)["Name"].astype(str).head(top_n).tolist()
            )
            top_genes_by_topic[int(topic_id)] = genes
        print(f"  ✓ Loaded gene data for {len(top_genes_by_topic)} topics")
    except Exception as e:
        print(f"  ✗ Error loading gene data: {e}")

    return top_genes_by_topic


def generate_unique_topic_names(
    input_dir: str, output_csv: str, gene_loading_file: str | None = None
) -> None:
    """Generate a summary CSV from per-topic markdown files with unique names."""
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found - {input_dir}")
        return

    md_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".md")],
        key=lambda x: int(re.search(r"topic_(\d+)_", x).group(1)),
    )

    top_genes_by_topic = load_top_genes_by_topic(gene_loading_file) if gene_loading_file else {}

    topic_rows: List[Dict[str, object]] = []
    used_names = set()

    print("Generating unique names for each topic...")
    for filename in md_files:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            topic_number = int(re.search(r"topic_(\d+)_", filename).group(1))
            # Try to match "Program label" (V9+ prompt) or fallback to "Three Key Words" (legacy)
            summary_match = re.search(
                r"\*\*Brief Summary:\*\*(.*?)\*\*(?:Program label|Three Key Words):\*\*",
                content,
                re.DOTALL,
            )
            keywords_match = re.search(r"\*\*(?:Program label|Three Key Words):\*\*(.*?)\n", content, re.DOTALL)

            summary = summary_match.group(1).strip() if summary_match else ""
            keywords_str = keywords_match.group(1).strip() if keywords_match else ""

            # Base name from first keyword
            base_name = f"Topic {topic_number}"
            keywords_list: List[str] = []
            if keywords_str:
                keywords_list = [kw.strip().title() for kw in keywords_str.split(",")]
                if keywords_list:
                    base_name = keywords_list[0]
                    if len(keywords_list) > 1:
                        base_name += f": {keywords_list[1]}"
                        if len(keywords_list) > 2:
                            base_name += f" & {keywords_list[2]}"

            # Ensure uniqueness using top genes or summary terms if needed
            final_name = base_name
            counter = 2
            if topic_number in top_genes_by_topic and final_name in used_names:
                signature_gene = top_genes_by_topic[topic_number][0]
                final_name = f"{base_name} ({signature_gene})"

            while final_name in used_names:
                if topic_number in top_genes_by_topic and counter - 2 < len(top_genes_by_topic[topic_number]):
                    signature_gene = top_genes_by_topic[topic_number][counter - 2]
                    final_name = f"{base_name} ({signature_gene})"
                else:
                    distinguishing_terms = re.findall(
                        r"\b(EGFR|Wnt|Xenobiotic|Estrogen|Mitochondrial|Senescence|Proliferation|Inflammation|Autophagy|Lipid|Glucose|Amino Acid)\b",
                        summary,
                        re.IGNORECASE,
                    )
                    if distinguishing_terms and counter - 2 < len(distinguishing_terms):
                        final_name = f"{base_name} ({distinguishing_terms[counter - 2]})"
                    else:
                        final_name = f"{base_name} ({counter})"
                counter += 1

            used_names.add(final_name)

            top_genes = top_genes_by_topic.get(topic_number, [])
            top_genes_str = ", ".join(top_genes[:10]) if top_genes else ""

            topic_rows.append(
                {
                    "Topic": topic_number,
                    "Name": final_name,
                    "Keywords": keywords_str,
                    "Top_Genes": top_genes_str,
                    "Summary": summary,
                }
            )
            print(f"  ✓ Topic {topic_number}: {final_name}")
            if top_genes:
                print(f"    Top genes: {', '.join(top_genes[:3])}...")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    if not topic_rows:
        print("No topic data was extracted. CSV file will not be created.")
        return

    print(f"\nWriting unique names to {output_csv}...")
    try:
        df = pd.DataFrame(topic_rows)
        df.sort_values(["Topic"], inplace=True)
        df.to_csv(output_csv, index=False)
        print("  ✓ CSV file created successfully.")
    except Exception as e:
        print(f"  ✗ Error writing CSV file: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse Anthropic JSONL results to markdown and generate a summary CSV."
    )
    parser.add_argument("--config", help="Path to config file (YAML or JSON)")
    parser.add_argument("--results-jsonl", help="Path to the .jsonl results file from Anthropic Batch API")
    parser.add_argument("--markdown-dir", help="Directory to write/read per-topic markdown files")
    parser.add_argument("--summary-csv", help="Path to write the summary CSV")
    parser.add_argument(
        "--gene-loading-file",
        help="Optional gene loading CSV (columns: Name, Score, RowID) to enrich summary naming",
    )
    parser.add_argument("--no-parse", action="store_true", help="Skip parsing JSONL → markdown step")
    parser.add_argument("--no-summary", action="store_true", help="Skip generating summary CSV step")
    parser.add_argument(
        "--gcs-prefix",
        help="GCS prefix to download Vertex AI results from (e.g., gs://bucket/outputs/job_123/)"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    cli_overrides = get_cli_overrides(sys.argv)
    args = apply_config_overrides(args, config, cli_overrides)

    if args.no_parse and args.no_summary:
        raise SystemExit("Both --no-parse and --no-summary are set; nothing to do.")

    # Optional: Download from GCS first
    results_jsonl = args.results_jsonl
    if args.gcs_prefix:
        # Download Vertex AI results from GCS
        temp_dir = tempfile.mkdtemp(prefix="vertex_results_")
        downloaded = download_from_gcs(args.gcs_prefix, temp_dir)
        
        # Find JSONL files in downloaded results
        jsonl_files = [f for f in downloaded if f.endswith(".jsonl")]
        if jsonl_files:
            results_jsonl = jsonl_files[0]
            print(f"Using downloaded results: {results_jsonl}")
        else:
            raise SystemExit(f"No JSONL files found in GCS prefix: {args.gcs_prefix}")

    # Step A: parse JSONL → markdown
    if not args.no_parse:
        if not args.markdown_dir:
            raise SystemExit("--markdown-dir is required unless --no-parse is set")
        if not results_jsonl:
            raise SystemExit("--results-jsonl or --gcs-prefix is required unless --no-parse is set")
        parse_final_results(results_jsonl, args.markdown_dir)

    # Step B: markdown → summary CSV
    if not args.no_summary:
        if not args.markdown_dir:
            raise SystemExit("--markdown-dir is required unless --no-summary is set")
        if not args.summary_csv:
            raise SystemExit("--summary-csv is required unless --no-summary is set")
        generate_unique_topic_names(args.markdown_dir, args.summary_csv, args.gene_loading_file)


if __name__ == "__main__":
    main()
