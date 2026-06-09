#!/usr/bin/env python3
"""On-demand drift check: detect param drift in PerturbNMF.

Source of truth: argparse `--arg` names in every pipeline .py under src/.
Consumers checked for drift:
  - Sibling README.md  (per-pipeline-script: args must be in that .py)
  - Sibling .sh runner (per-pipeline-script: args in the `python3 <py>` block)
  - Skill SKILL.md + references/*.md (global: args must be in *some* pipeline .py)

Output to stdout = drift report. Exit 0 always (informational, not a gate).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF")

if not (ROOT / ".git").exists():
    sys.exit(0)

ARG_RE = re.compile(r"add_argument\(\s*['\"]--([a-zA-Z][a-zA-Z0-9_-]+)['\"]")
R_ARG_RE = re.compile(r"make_option\(\s*['\"]--([a-zA-Z][a-zA-Z0-9_-]+)['\"]")
PYTEST_ARG_RE = re.compile(r"addoption\(\s*['\"]--([a-zA-Z][a-zA-Z0-9_-]+)['\"]")
DASH_RE = re.compile(r"--([a-zA-Z][a-zA-Z0-9_-]+)")
TABLE_ROW_RE = re.compile(r"^\|\s*`?([a-z][a-z0-9_-]+)`?\s*\|", re.MULTILINE)

SKIP_DIR_PARTS = {"JupterNote_Version", "JupterNote_version", "tests", ".git", "__pycache__"}
# Wildcard placeholders in narrative prose ("renamed --foo_* to --bar_*").
WILDCARD_RE = re.compile(r"--[a-zA-Z0-9_]+_\*")
# Abstract placeholders used in skill markdown to document conventions, not
# real CLI flags. Examples: `--flag`, `--flag_name` are stand-ins meaning "any
# flag name"; `---COMMENTED---` is a literal sentinel string the convention
# uses to separate active from optional flags in generated scripts. The regex
# can't tell these apart from real flag references, so we exempt them.
SKILL_PLACEHOLDER_ALLOWLIST = {"flag", "flag_name", "COMMENTED---"}
# Lines describing a rename — old name on the line is documentation, not drift.
RENAME_RE = re.compile(r"(?:→|->|\bis now\b|\brenamed\b|\bformerly\b|\bwas\s+renamed\b)", re.IGNORECASE)
# Lines referencing external CLI tools — their --flags belong to those tools.
EXTERNAL_TOOL_RE = re.compile(
    r"\b(sacct|sbatch|squeue|scancel|sinfo|srun|salloc|jupyter|nbconvert|pytest|conftest)\b"
)


def skip(p: Path) -> bool:
    return any(part in SKIP_DIR_PARTS for part in p.parts)


def py_args(py: Path) -> set[str]:
    try:
        return set(ARG_RE.findall(py.read_text(errors="ignore")))
    except OSError:
        return set()


def r_args(r: Path) -> set[str]:
    try:
        return set(R_ARG_RE.findall(r.read_text(errors="ignore")))
    except OSError:
        return set()


def collect_pipeline_args() -> dict[Path, set[str]]:
    out: dict[Path, set[str]] = {}
    for py in (ROOT / "src").rglob("*.py"):
        if skip(py):
            continue
        args = py_args(py)
        if args:
            out[py] = args
    for r in (ROOT / "src").rglob("*.R"):
        if skip(r):
            continue
        args = r_args(r)
        if args:
            out[r] = args
    return out


def collect_skill_helper_args() -> set[str]:
    """Args defined in skill helper scripts (.claude/skills/**/scripts/*.py).
    These are valid CLI flags — the skill's own helpers — even though they're
    not pipeline args. Used only to suppress false positives in skill markdown.
    """
    out: set[str] = set()
    for py in (ROOT / ".claude" / "skills").rglob("scripts/*.py"):
        out.update(py_args(py))
    return out


def collect_conftest_args() -> set[str]:
    """Pytest fixture flags defined via parser.addoption(...) in conftest.py."""
    out: set[str] = set()
    for f in ROOT.rglob("conftest.py"):
        if "tests" not in f.parts:
            continue
        try:
            out.update(PYTEST_ARG_RE.findall(f.read_text(errors="ignore")))
        except OSError:
            continue
    return out


def _skip_line(line: str) -> bool:
    """True if this line's --flags should not be treated as pipeline-arg drift.

    Suppresses rename notes (intentional old-name docs) and lines describing
    external CLI tools (sacct/jupyter/pytest etc. — their flags belong to them).
    """
    return bool(RENAME_RE.search(line) or EXTERNAL_TOOL_RE.search(line))


def md_param_table_names(md: str) -> set[str]:
    """First-column identifiers under any '## Parameter*' or '## Argument*' heading."""
    out: set[str] = set()
    in_section = False
    for line in md.splitlines():
        if line.lstrip().startswith("#"):
            heading = line.lstrip("# ").lower()
            in_section = "parameter" in heading or "argument" in heading
            continue
        if in_section and not _skip_line(line):
            m = TABLE_ROW_RE.match(line)
            if m and m.group(1).lower() not in {
                "parameter", "argument", "name", "type", "default",
                "description", "required", "optional", "key",
            }:
                out.add(m.group(1))
    return out


def md_dash_mentions(md: str) -> set[str]:
    """Scan --flag mentions in markdown, skipping rename/external-tool lines."""
    stripped = WILDCARD_RE.sub("", md)
    out: set[str] = set()
    for line in stripped.splitlines():
        if _skip_line(line):
            continue
        out.update(DASH_RE.findall(line))
    return out


def sh_runner_args(sh: Path, target_basename: str | None = None) -> set[str]:
    """Args in the `python3 ... <pipeline.py>` or `Rscript ... <pipeline.R>` block.

    Skips #SBATCH directives and other comments. If target_basename is given,
    only return args from a block invoking that script.
    """
    try:
        lines = sh.read_text(errors="ignore").splitlines()
    except OSError:
        return set()

    out: set[str] = set()
    in_block = False
    for raw in lines:
        s = raw.strip()
        if s.startswith("#"):
            continue
        if not in_block:
            invokes = (("python" in s and ".py" in s) or ("Rscript" in s and ".R" in s))
            if invokes:
                if target_basename is None or target_basename in s:
                    in_block = True
                continue
        else:
            m = re.match(r"--([a-zA-Z][a-zA-Z0-9_-]+)", s)
            if m:
                out.add(m.group(1))
            if not s.endswith("\\"):
                in_block = False
    return out


def find_sibling(py: Path, suffix: str) -> Path | None:
    """README.md or matching .sh in the same directory as the pipeline .py."""
    if suffix == ".md":
        cand = py.parent / "README.md"
        return cand if cand.exists() else None
    if suffix == ".sh":
        # Prefer same-stem .sh; else any .sh in the dir.
        same = py.with_suffix(".sh")
        if same.exists():
            return same
        shs = list(py.parent.glob("*.sh"))
        return shs[0] if len(shs) == 1 else None
    return None


def check_per_script(pipeline: dict[Path, set[str]]) -> list[tuple[Path, Path, set[str]]]:
    """For each pipeline .py, compare its sibling README and .sh to its argparse.

    Returns list of (consumer_file, pipeline_py, stale_arg_set).
    """
    findings: list[tuple[Path, Path, set[str]]] = []
    for py, current in pipeline.items():
        readme = find_sibling(py, ".md")
        if readme:
            text = readme.read_text(errors="ignore")
            mentioned = md_dash_mentions(text) | md_param_table_names(text)
            stale = mentioned - current
            if stale:
                findings.append((readme, py, stale))
        sh = find_sibling(py, ".sh")
        if sh:
            mentioned = sh_runner_args(sh, target_basename=py.name)
            stale = mentioned - current
            if stale:
                findings.append((sh, py, stale))
    return findings


def check_skill_globally(global_args: set[str], helper_args: set[str]) -> dict[Path, set[str]]:
    """Skill markdown: an arg must be in some pipeline .py OR in a skill helper .py.
    The latter exempts the skill's own CLI flags (--gpu, --stage, etc.)."""
    valid = global_args | helper_args
    out: dict[Path, set[str]] = {}
    for f in (ROOT / ".claude" / "skills").rglob("*.md"):
        if skip(f):
            continue
        text = f.read_text(errors="ignore")
        mentioned = md_dash_mentions(text)
        stale = (mentioned - valid) - SKILL_PLACEHOLDER_ALLOWLIST
        if stale:
            out[f] = stale
    return out


def main() -> int:
    pipeline = collect_pipeline_args()
    if not pipeline:
        print("No pipeline .py files found under src/. Nothing to check.")
        return 0
    global_args = set().union(*pipeline.values())
    helper_args = collect_skill_helper_args() | collect_conftest_args()

    per_script = check_per_script(pipeline)
    skill_drift = check_skill_globally(global_args, helper_args)

    if not per_script and not skill_drift:
        print("No drift detected. README/.sh/skill docs are aligned with src/ argparse.")
        return 0

    lines: list[str] = ["<doc-sync-check>", "Param drift detected in PerturbNMF."]

    if per_script:
        lines.append("")
        lines.append("Per-script drift (sibling README/.sh has args missing from its .py):")
        # Group by consumer file for readability
        by_consumer: dict[Path, list[tuple[Path, set[str]]]] = defaultdict(list)
        for consumer, py, stale in per_script:
            by_consumer[consumer].append((py, stale))
        for consumer, entries in sorted(by_consumer.items()):
            rel = consumer.relative_to(ROOT)
            lines.append(f"  {rel}")
            for py, stale in entries:
                py_rel = py.relative_to(ROOT)
                for a in sorted(stale):
                    lines.append(f"    - {a}    (not in {py_rel})")

    if skill_drift:
        lines.append("")
        lines.append("Skill drift (args mentioned but not defined in ANY pipeline .py):")
        for path, stale in sorted(skill_drift.items()):
            rel = path.relative_to(ROOT)
            lines.append(f"  {rel}")
            for a in sorted(stale):
                lines.append(f"    - {a}")

    lines += [
        "",
        "Action: for each stale arg, ASK the user before editing each consumer file.",
        "Show the exact line(s) and proposed change. After approved edits, re-run:",
        "  python3 .claude/skills/pipeline-drift-check/scripts/check-pipeline-drift.py",
        "Skip Jupyter notebooks and tests/ unless explicitly asked.",
        "</doc-sync-check>",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
