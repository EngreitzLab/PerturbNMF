"""Per-program run records of the LLM steps: what each step did to each program, and why it stopped.

    <step out_dir>/_runs/P<k>.json
        {"program", "step", "status", "model", "prompts": {id: sha256}, "duration_s", "usd", "detail", "errors",
         "raw_answer", "updated"}

status is one of
    completed           the step's P<k>.json was written by this run
    cached              P<k>.json from an earlier run was kept
    pending_session     session backend: the request is written, the answer not yet imported
    failed              no answer (API error, refusal, cut off, budget skip)
    failed_validation   an answer came back but did not pass the step's checks (kept in raw_answer)

The step's own P<k>.json stays the result; this record is the bookkeeping beside it (status.py reads both).
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

STATUSES = ("completed", "cached", "pending_session", "failed", "failed_validation")


def run_path(out_dir, program):
    return Path(out_dir) / "_runs" / f"{program}.json"


def stale_prompt(out, *prompts):
    """{"stale_prompts": [...]} when the cached output out was made with other prompt files than prompts (api/prompts)."""
    made = json.loads(Path(out).read_text()).get("prompts")
    if made is None:
        return {"stale_prompts": "unrecorded (made before prompts were versioned)"}
    stale = [p.id for p in prompts if made.get(p.id) != p.sha256]
    return {"stale_prompts": stale} if stale else {}


def read_run(out_dir, program):
    path = run_path(out_dir, program)
    return json.loads(path.read_text()) if path.exists() else None


class StepRun:
    """One step on one program: create it before the call (starts the clock), finish(status, ...) after it."""

    def __init__(self, out_dir, program, step, model, prompts=None, budget=None, record_name=None):
        self.out_dir, self.program, self.step, self.model = Path(out_dir), program, step, model
        self.record_name = record_name or program   # _runs/<record_name>.json (program folders: the step id)
        self.prompts = prompts or {}
        self.budget = budget                    # api/budget.ProgramBudget or None
        self.t0 = time.time()
        self.usd0 = self._step_usd()

    def _step_usd(self):
        if self.budget is None:
            return 0.0
        return json.loads(self.budget.path.read_text())["by_step"].get(self.step, 0.0)

    def finish(self, status, detail=None, errors=(), raw_answer=None):
        if status not in STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        record = {
            "program": self.program, "step": self.step, "status": status, "model": self.model,
            "prompts": self.prompts, "duration_s": round(time.time() - self.t0, 2),
            "usd": round(self._step_usd() - self.usd0, 6) if self.budget is not None else None,
            "detail": detail or {}, "errors": list(errors),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        if raw_answer is not None:
            record["raw_answer"] = raw_answer
        path = run_path(self.out_dir, self.record_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        tmp.replace(path)
        return record
