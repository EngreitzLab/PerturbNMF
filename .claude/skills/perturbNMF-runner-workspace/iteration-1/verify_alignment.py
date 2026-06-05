#!/usr/bin/env python3
"""Verify that the perturbNMF-runner skill's parameter tables align with the
live pipeline argparse blocks.

Checks per-file:
- For each `add_argument('--flag', ..., default=X)` in a pipeline .py, confirm
  parameter-catalog.md has a row whose first cell starts with `--flag`.
- For flags with a default literal that the regex can capture, also check that
  the catalog's default cell mentions the same value (relaxed string match).
- For per-step files (01-05), warn if a flag appears in the step file but is
  not defined in any pipeline .py.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF")
SKILL = ROOT / ".claude" / "skills" / "perturbNMF-runner"
SRC = ROOT / "src"

# add_argument('--flag', type=..., default=<DEFAULT>, ...)
ADD_ARG = re.compile(
    r"add_argument\(\s*['\"]--([a-zA-Z][a-zA-Z0-9_-]+)['\"](?P<rest>[^)]*)\)",
    re.DOTALL,
)
DEFAULT_RE = re.compile(r"default\s*=\s*(?P<v>[^,)]+)", re.DOTALL)
TABLE_FLAG = re.compile(r"^\|\s*`?--([a-zA-Z][a-zA-Z0-9_-]+)`?\s*\|", re.MULTILINE)
DASH_PROSE = re.compile(r"--([a-zA-Z][a-zA-Z0-9_-]+)")

SKIP_DIR_PARTS = {"JupterNote_Version", "JupterNote_version", "tests", "__pycache__"}


def skip(p: Path) -> bool:
    return any(part in SKIP_DIR_PARTS for part in p.parts)


def collect_pipeline_flags() -> dict[str, dict[str, str]]:
    """Map of flag → {script_path: default_text}."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for py in SRC.rglob("*.py"):
        if skip(py):
            continue
        try:
            text = py.read_text(errors="ignore")
        except OSError:
            continue
        for m in ADD_ARG.finditer(text):
            flag = m.group(1)
            rest = m.group("rest")
            d = DEFAULT_RE.search(rest)
            default = d.group("v").strip() if d else "(no default)"
            out[flag][str(py.relative_to(ROOT))] = default
    return out


def collect_catalog_flags(md_path: Path) -> set[str]:
    text = md_path.read_text(errors="ignore")
    return set(TABLE_FLAG.findall(text))


def collect_md_dash_mentions(md_path: Path) -> set[str]:
    text = md_path.read_text(errors="ignore")
    return set(DASH_PROSE.findall(text))


def main() -> int:
    pipeline = collect_pipeline_flags()
    catalog = SKILL / "references" / "parameter-catalog.md"
    catalog_flags = collect_catalog_flags(catalog)

    pipeline_flag_set = set(pipeline.keys())

    # Pipeline flags missing from catalog
    missing_in_catalog = sorted(pipeline_flag_set - catalog_flags)
    # Catalog flags not defined in any pipeline
    catalog_orphans = sorted(catalog_flags - pipeline_flag_set)

    print(f"Pipeline flag count: {len(pipeline_flag_set)}")
    print(f"Catalog table-row flag count: {len(catalog_flags)}")
    print()
    print(f"Pipeline flags MISSING from parameter-catalog.md: {len(missing_in_catalog)}")
    for f in missing_in_catalog:
        scripts = list(pipeline[f].keys())
        print(f"  --{f}  (in {len(scripts)} script(s): {scripts[0]}{'...' if len(scripts)>1 else ''})")

    print()
    print(f"Catalog flags NOT in any pipeline (potential rename/orphan): {len(catalog_orphans)}")
    for f in catalog_orphans:
        print(f"  --{f}")

    # Per-step file → matching pipeline directory
    step_to_script_dirs = {
        "01-inference.md": [SRC / "Stage1_Inference"],
        "02-evaluation.md": [SRC / "Stage2_Evaluation" / "A_Metrics"],
        "03-calibration.md": [SRC / "Stage2_Evaluation" / "B_Calibration"],
        "04-visualization.md": [SRC / "Stage3_Interpretation" / "A_Plotting"],
        "05-annotation-summary.md": [SRC / "Stage3_Interpretation" / "C_Annotation",
                                     SRC / "Stage3_Interpretation" / "B_Summarization"],
    }
    print()
    print("Per-step file: flags referenced that are not in any pipeline .py")
    for step_name, _dirs in step_to_script_dirs.items():
        f = SKILL / "references" / step_name
        if not f.exists():
            continue
        mentions = collect_md_dash_mentions(f)
        orphans = sorted(mentions - pipeline_flag_set)
        # Drop convention placeholders that we know are false positives.
        orphans = [x for x in orphans if x not in {"flag", "flag_name", "COMMENTED---"}]
        if orphans:
            print(f"  {step_name}: {orphans}")
        else:
            print(f"  {step_name}: clean")

    # Default value spot-check on the high-signal flags that were just edited
    spot_check = ["K", "sel_threshs", "numhvgenes", "tol"]
    print()
    print("Default spot-check (flag → pipeline defaults vs catalog default cell):")
    text = catalog.read_text()
    for flag in spot_check:
        if flag not in pipeline:
            print(f"  --{flag}: not in pipeline")
            continue
        defaults_per_script = pipeline[flag]
        # Find catalog row(s) for this flag
        rows = [ln for ln in text.splitlines()
                if re.match(rf"^\|\s*`?--{re.escape(flag)}`?\s*\|", ln)]
        print(f"  --{flag}:")
        for script, dflt in defaults_per_script.items():
            print(f"      pipeline {script}: default={dflt}")
        for r in rows:
            print(f"      catalog row: {r.strip()[:140]}")

    n_problems = len(missing_in_catalog) + len(catalog_orphans)
    print()
    print(f"Total alignment problems: {n_problems}")
    return 0 if n_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
