---
name: pipeline-drift-check
description: >
  Detect parameter drift across the PerturbNMF pipeline. Cross-checks every
  `--flag` defined in `src/**/*.py` (argparse) and `src/**/*.R` (optparse) against
  what is mentioned in sibling README.md files, sibling .sh runners, and the
  perturbNMF-runner skill markdown (SKILL.md + references/*.md). Use this skill
  whenever the user says "check drift", "drift check", "check parameter drift",
  "run drift check", "are the docs in sync", "did I update everything", "check
  if README/sh/skill match the .py", "sync check", or "doc sync". Also trigger
  after the user edits any pipeline argparse, README, SLURM .sh runner, or the
  perturbNMF-runner skill references — and any time they ask whether parameter
  names in .py, .sh, README, and the skill files agree.
---

# Pipeline Drift Check Skill

Detect param drift between source-of-truth argparse definitions and documentation/runner consumers in the PerturbNMF pipeline.

## What it checks

Source of truth: `add_argument("--flag", ...)` in every `src/**/*.py` and `make_option("--flag", ...)` in every `src/**/*.R`.

Consumers checked:
- **Sibling README.md** of each pipeline script — `--flag` mentions and `## Parameters` table rows must exist in that script's argparse.
- **Sibling .sh runner** — flags inside the `python3 <pipeline.py>` (or `Rscript`) block must exist in that script.
- **Skill markdown** (`.claude/skills/**/*.md`) — every `--flag` mentioned must be defined in *some* pipeline .py/.R, OR in a skill helper script under `.claude/skills/**/scripts/*.py`, OR in a test `conftest.py` `addoption()`.

## Suppressions (intentional, not bugs)

- Lines containing rename notes (`→`, `->`, "renamed", "formerly", "is now") — old names are documentation.
- Lines mentioning external CLI tools (`sacct`, `sbatch`, `jupyter`, `pytest`, etc.) — their flags belong to those tools.
- Wildcard prose placeholders like `--foo_*`.
- Skill convention placeholders: `--flag`, `--flag_name`, `---COMMENTED---`.
- Skipped directories: `JupterNote_Version`, `tests/`, `.git`, `__pycache__`.

## How to invoke

```bash
python3 /oak/stanford/groups/engreitz/Users/ymo/Tools/PerturbNMF/.claude/skills/pipeline-drift-check/scripts/check-pipeline-drift.py
```

Exit code is always 0. Output is the drift report (or a "no drift detected" line).

## When you find drift

For each stale arg:
1. Show the user the exact file and line(s) where the stale flag appears.
2. Propose the fix (remove the stale flag, rename it, or add it to the .py if it should exist).
3. **Ask before editing** — never silently rewrite README / .sh / skill files.
4. After approved edits, re-run the script to confirm the report is clean.

Skip Jupyter notebooks and `tests/` content unless the user explicitly asks.

## Related

- There is a global SessionStart hook at `/home/users/ymo/.claude/scripts/check-pipeline-drift.py` that runs the same logic automatically at the start of every session. This skill is for **on-demand** invocation mid-session (e.g., right after editing argparse, README, or skill references).
- Last verification snapshot: `.claude/skills/perturbNMF-runner-workspace/iteration-1/summary.md`.
