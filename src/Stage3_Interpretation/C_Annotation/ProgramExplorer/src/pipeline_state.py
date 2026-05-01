from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


STEP_NAMES = [
    "string_enrichment",
    "literature_fetch",
    "batch_prepare",
    "batch_submit",
    "parse_results",
    "html_report",
]


@dataclass
class StepState:
    status: str = "pending"  # pending | in_progress | submitted | completed | failed
    completed_at: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    config_hash: str
    started_at: str
    topics: Optional[list[int]]
    steps: Dict[str, StepState] = field(default_factory=dict)
    batch: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "started_at": self.started_at,
            "topics": self.topics,
            "steps": {k: asdict(v) for k, v in self.steps.items()},
            "batch": self.batch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        steps_raw = data.get("steps", {})
        steps = {k: StepState(**v) for k, v in steps_raw.items()}
        return cls(
            config_hash=data.get("config_hash", ""),
            started_at=data.get("started_at", datetime.utcnow().isoformat()),
            topics=data.get("topics"),
            steps=steps,
            batch=data.get("batch", {}),
        )


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute a stable hash of the config dict.

    Excludes keys that don't affect pipeline outputs:
    - Keys starting with "_" (internal paths map)
    - llm_wait: Only affects wait behavior, not results
    - resume: Runtime flag, not a config setting
    """
    excluded_keys = {"llm_wait", "resume"}
    filtered = {
        k: v
        for k, v in config_dict.items()
        if not k.startswith("_") and k not in excluded_keys
    }
    payload = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def init_state(config_hash: str, topics: Optional[list[int]]) -> PipelineState:
    steps = {name: StepState() for name in STEP_NAMES}
    return PipelineState(
        config_hash=config_hash,
        started_at=datetime.utcnow().isoformat(),
        topics=topics,
        steps=steps,
        batch={},
    )


def load_state(path: Path) -> Optional[PipelineState]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PipelineState.from_dict(data)


def save_state(path: Path, state: PipelineState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def mark_step(state: PipelineState, step: str, status: str, info: Optional[Dict[str, Any]] = None) -> None:
    s = state.steps.get(step) or StepState()
    s.status = status
    if status == "completed":
        s.completed_at = datetime.utcnow().isoformat()
    if info:
        s.info.update(info)
    state.steps[step] = s
