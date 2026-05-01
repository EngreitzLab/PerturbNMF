"""
LLM submission and monitoring for batch annotation.

Provides functions to submit prompts to Anthropic (direct Batch API or
single-message) and Vertex AI, check job status, and retrieve results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are already set

# ---------------------------------------------------------------------------
# Vertex AI imports (optional)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Anthropic imports (optional)
# ---------------------------------------------------------------------------
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Vertex AI configuration constants
# ---------------------------------------------------------------------------

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

# Default model name (Anthropic)
MODEL = "claude-sonnet-4-5-20250929"


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
