"""
Library module for parsing batch JSONL results and generating topic summaries.

Ported from 04_parse_and_summarize.py — contains only reusable functions,
no CLI / argparse / config-loading code.

Functions:
    download_from_gcs       — download files from a GCS prefix
    parse_final_results     — parse JSONL into per-topic markdown files
    load_top_genes_by_topic — load top genes per program from a gene-loading CSV
    generate_unique_topic_names — build a summary CSV with unique topic names
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd


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
        print(f"  GCS download failed: {result.stderr}")
        return []

    # Find downloaded files
    downloaded = glob.glob(os.path.join(local_dir, "*"))
    print(f"  Downloaded {len(downloaded)} files")
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
                            print(f"  Topic {topic_number} failed. Reason: {error.get('message', 'Unknown error')}")
                            continue

                    if text_content:
                        output_filename = os.path.join(output_dir, f"topic_{topic_number}_annotation.md")
                        with open(output_filename, "w", encoding="utf-8") as out_f:
                            out_f.write(text_content)
                        print(f"  Saved Topic {topic_number}")
                        saved_ids.append(topic_number)
                    else:
                        print(f"  Topic {topic_number} had empty content")
            except json.JSONDecodeError:
                print(f"  Could not parse line: {line.strip()}")
            except Exception as e:
                print(f"  Error processing line: {e}")

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
            print("  CSV missing 'program_id' or 'RowID' column")
            return top_genes_by_topic

        for topic_id, group in gene_df.groupby(group_col):
            genes = (
                group.sort_values("Score", ascending=False)["Name"].astype(str).head(top_n).tolist()
            )
            top_genes_by_topic[int(topic_id)] = genes
        print(f"  Loaded gene data for {len(top_genes_by_topic)} topics")
    except Exception as e:
        print(f"  Error loading gene data: {e}")

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
    used_names: set[str] = set()

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
            print(f"  Topic {topic_number}: {final_name}")
            if top_genes:
                print(f"    Top genes: {', '.join(top_genes[:3])}...")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    if not topic_rows:
        print("No topic data was extracted. CSV file will not be created.")
        return

    print(f"\nWriting unique names to {output_csv}...")
    try:
        df = pd.DataFrame(topic_rows)
        df.sort_values(["Topic"], inplace=True)
        df.to_csv(output_csv, index=False)
        print("  CSV file created successfully.")
    except Exception as e:
        print(f"  Error writing CSV file: {e}")
