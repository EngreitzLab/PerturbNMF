"""
@description
Batch job status checking and result fetching utilities.
It is responsible for checking Anthropic and Vertex AI batch job status
and downloading results when complete.

Key features:
- Check-once-and-exit pattern for idempotent resume
- Supports both Anthropic Batch API and Vertex AI
- Downloads results to local path when complete

@dependencies
- anthropic: Anthropic Batch API client
- google.cloud.aiplatform: Vertex AI client (optional)

@examples
- status, path = check_anthropic_batch("msgbatch_xxx", "output/results.jsonl")
- status, path = check_vertex_batch("job_id", "gs://bucket/prefix")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_anthropic_batch(
    batch_id: str,
    output_path: str,
) -> Tuple[str, Optional[str]]:
    """Check Anthropic batch status and download results if complete.

    Args:
        batch_id: Anthropic batch ID (e.g., msgbatch_xxx)
        output_path: Local path to save results JSONL

    Returns:
        Tuple of (status, results_path):
            status: "in_progress" | "completed" | "failed"
            results_path: Path to downloaded JSONL if completed, else None
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return "failed", None

    try:
        import os
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        client = anthropic.Anthropic(auth_token=auth_token) if auth_token else anthropic.Anthropic()
        batch = client.messages.batches.retrieve(batch_id)
    except Exception as exc:
        logger.error("Failed to retrieve batch %s: %s", batch_id, exc)
        return "failed", None

    status = batch.processing_status
    counts = batch.request_counts

    logger.info(
        "Batch %s: status=%s, succeeded=%d, errored=%d, processing=%d",
        batch_id,
        status,
        counts.succeeded,
        counts.errored,
        counts.processing,
    )

    if status == "in_progress":
        return "in_progress", None

    if status == "ended":
        if counts.errored > 0 and counts.succeeded == 0:
            logger.error("All batch requests failed (%d errors)", counts.errored)
            return "failed", None

        # Download results
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with output_file.open("w", encoding="utf-8") as f:
                for result in client.messages.batches.results(batch_id):
                    f.write(result.model_dump_json() + "\n")

            logger.info("Downloaded results to %s", output_path)
            return "completed", output_path

        except Exception as exc:
            logger.error("Failed to download batch results: %s", exc)
            return "failed", None

    # Other statuses (canceling, expired, etc.)
    logger.error("Batch ended with unexpected status: %s", status)
    return "failed", None


def check_vertex_batch(
    job_name: str,
    output_prefix: str,
) -> Tuple[str, Optional[str]]:
    """Check Vertex AI batch job status and return results path if complete.

    Args:
        job_name: Vertex AI batch job name/ID
        output_prefix: GCS prefix where results are stored

    Returns:
        Tuple of (status, results_path):
            status: "in_progress" | "completed" | "failed"
            results_path: GCS prefix to results if completed, else None
    """
    try:
        from google.cloud import aiplatform
        from google.cloud.aiplatform_v1.types import JobState
    except ImportError:
        logger.error("google-cloud-aiplatform package not installed")
        return "failed", None

    try:
        # Initialize client
        aiplatform.init()

        # Get batch prediction job
        job = aiplatform.BatchPredictionJob(job_name)
        state = job.state

        logger.info("Vertex job %s: state=%s", job_name, state.name)

        if state == JobState.JOB_STATE_SUCCEEDED:
            # Results are at output_prefix
            results_path = output_prefix
            if hasattr(job, "output_info") and job.output_info:
                results_path = job.output_info.gcs_output_directory
            logger.info("Job completed. Results at: %s", results_path)
            return "completed", results_path

        if state in (
            JobState.JOB_STATE_RUNNING,
            JobState.JOB_STATE_PENDING,
            JobState.JOB_STATE_QUEUED,
        ):
            return "in_progress", None

        # Failed or cancelled
        logger.error("Vertex job ended with state: %s", state.name)
        return "failed", None

    except Exception as exc:
        logger.error("Failed to check Vertex job %s: %s", job_name, exc)
        return "failed", None
