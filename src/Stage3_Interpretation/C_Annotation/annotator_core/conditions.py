"""The conditions of a multi-condition screen, read the same way by every annotator step.

A screen read out separately in several conditions — timepoints, stimuli, doses, donors,
genotypes, cohorts — is a multi-condition screen. The config describes it with:

  conditions                  [{"label": "D0", "description": "..."}, ...]; labels match the
                              `condition` column of the per-condition tables
  settings.condition_variable what the conditions vary, in the reader's words (e.g. "timepoint",
                              "cytokine stimulus", "donor", "age x sex cohort"); default "condition"
  settings.condition_design   "unordered" (default) or "ordered". Only "ordered" (timepoints, a
                              dose series, a differentiation) adds ordering language to prompts:
                              peak-before-effect reasoning, contiguous blocks of conditions.

Deprecated spellings still work, with a warning on stderr: a condition's "stage" (now
"description"); condition_design "time_course" (now "ordered") and "groups" (now "unordered").
"""
from __future__ import annotations

import sys
from typing import List

DESIGNS = ("unordered", "ordered")
DEPRECATED_DESIGNS = {"time_course": "ordered", "groups": "unordered"}
warned: set = set()


def warn_once(message: str) -> None:
    if message not in warned:
        warned.add(message)
        print(f"WARNING: {message}", file=sys.stderr)


def read_setting(settings: dict, key: str, deprecated_key: str, default=None):
    """settings[key], falling back to a deprecated key (with a warning), then to the default."""
    if key in settings:
        return settings[key]
    if deprecated_key in settings:
        warn_once(f"setting '{deprecated_key}' is deprecated; rename it to '{key}'")
        return settings[deprecated_key]
    return default


def normalise_conditions(conditions: List[dict] | None) -> List[dict]:
    """Each condition as {"label", "description"}, accepting the deprecated "stage" key."""
    out = []
    for condition in conditions or []:
        entry = dict(condition)
        if "description" not in entry and "stage" in entry:
            warn_once("condition key 'stage' is deprecated; rename it to 'description'")
            entry["description"] = entry["stage"]
        entry.pop("stage", None)
        entry.setdefault("description", "")
        out.append(entry)
    return out


def read_condition_design(settings: dict) -> str:
    """'ordered' or 'unordered' (the default)."""
    if "condition_design" not in settings:
        warn_once('settings.condition_design not set; treating the conditions as unordered '
                  '(set "ordered" for timepoints or a dose series)')
    design = settings.get("condition_design", "unordered")
    if design in DEPRECATED_DESIGNS:
        warn_once(f"condition_design '{design}' is deprecated; use '{DEPRECATED_DESIGNS[design]}'")
        design = DEPRECATED_DESIGNS[design]
    if design not in DESIGNS:
        raise SystemExit(f"settings.condition_design must be one of {DESIGNS}, got {design!r}")
    return design


def read_condition_variable(settings: dict) -> str:
    return settings.get("condition_variable") or "condition"
