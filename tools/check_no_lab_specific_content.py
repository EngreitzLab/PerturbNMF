#!/usr/bin/env python3
"""Fail if tracked files (or new changes) contain lab- or project-specific content.

This repository is public. It must not contain study/dataset names, collaborator
names, personal emails, or lab/cluster-specific paths. This script greps for a
list of forbidden patterns and exits non-zero on any hit that is not covered by
tools/lab_specific_allowlist.txt.

Usage (from anywhere inside the repo):
    python3 tools/check_no_lab_specific_content.py              # scan all tracked files
    python3 tools/check_no_lab_specific_content.py --staged     # scan lines added in the index (pre-commit)
    python3 tools/check_no_lab_specific_content.py --diff origin/main   # scan lines added since a ref

Both file contents and file paths are checked. Binary files are checked by path only.
Standard library only (runs in CI without installing anything).
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

# Case-insensitive regexes. Keep this list in sync with CLAUDE.md
# ("Public repository — no project-specific content").
FORBIDDEN_PATTERNS = [
    r"oak/stanford",
    r"/scratch/",
    r"\$\{?SCRATCH\b",
    r"sherlock",
    r"/ymo/",
    r"Users/ymo",
    r"ymo@",
    r"opushkar",
    r"pushkarev",
    r"\bolga\b",
    r"telohaec",
    r"cc-perturb",
    r"ccperturb",
    r"cc_perturb",
    r"groups/engreitz",
    r"Users/engreitz",
    r"engreitz,owners",
    r"@stanford\.edu",
    r"tony_method",
]
FORBIDDEN_RE = re.compile("|".join(f"(?:{p})" for p in FORBIDDEN_PATTERNS), re.IGNORECASE)

REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
)
ALLOWLIST_PATH = REPO_ROOT / "tools" / "lab_specific_allowlist.txt"


def load_allowlist(path: Path) -> tuple[list[str], list[str]]:
    """Return (path globs exempt entirely, literal strings removed before matching)."""
    path_globs: list[str] = []
    literals: list[str] = []
    if not path.exists():
        return path_globs, literals
    for raw in path.read_text().splitlines():
        line = raw.split(" #", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        kind, _, value = line.partition(":")
        value = value.strip()
        if kind == "path" and value:
            path_globs.append(value)
        elif kind == "text" and value:
            literals.append(value)
        else:
            sys.exit(f"{path}: cannot parse allowlist line: {raw!r} (expected 'path: <glob>' or 'text: <literal>')")
    return path_globs, literals


def is_path_allowed(file_path: str, path_globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(file_path, g) for g in path_globs)


def find_hits(text: str, literals: list[str]) -> list[str]:
    for literal in literals:
        text = re.sub(re.escape(literal), "", text, flags=re.IGNORECASE)
    return [m.group(0) for m in FORBIDDEN_RE.finditer(text)]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(8192)
    except OSError:
        return True


def scan_tracked(path_globs: list[str], literals: list[str]) -> list[str]:
    problems = []
    for file_path in git("ls-files", "-z").split("\0"):
        if not file_path or is_path_allowed(file_path, path_globs):
            continue
        for hit in find_hits(file_path, literals):
            problems.append(f"{file_path}: [path] matches '{hit}'")
        full = REPO_ROOT / file_path
        if not full.is_file() or is_binary(full):
            continue
        try:
            lines = full.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            for hit in find_hits(line, literals):
                problems.append(f"{file_path}:{lineno}: matches '{hit}': {line.strip()[:160]}")
    return problems


def scan_diff(diff_args: list[str], path_globs: list[str], literals: list[str]) -> list[str]:
    """Scan only added lines (and added/renamed file paths) in a diff."""
    problems = []
    diff = git("diff", "--no-color", "--unified=0", "--no-ext-diff", *diff_args)
    current_file = None
    lineno = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            current_file = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            if current_file and not is_path_allowed(current_file, path_globs):
                for hit in find_hits(current_file, literals):
                    problems.append(f"{current_file}: [path] matches '{hit}'")
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
            continue
        if line.startswith("+") and current_file:
            if not is_path_allowed(current_file, path_globs):
                for hit in find_hits(line[1:], literals):
                    problems.append(f"{current_file}:{lineno}: matches '{hit}': {line[1:].strip()[:160]}")
            lineno += 1
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="Scan lines added in the index (for pre-commit)")
    mode.add_argument("--diff", metavar="REF", help="Scan lines added since REF (e.g. origin/main)")
    args = parser.parse_args()

    path_globs, literals = load_allowlist(ALLOWLIST_PATH)
    if args.staged:
        problems = scan_diff(["--cached"], path_globs, literals)
    elif args.diff:
        problems = scan_diff([f"{args.diff}...HEAD"], path_globs, literals)
    else:
        problems = scan_tracked(path_globs, literals)

    if problems:
        print("Lab/project-specific content found (this repo is public):\n")
        print("\n".join(problems))
        print(
            f"\n{len(problems)} hit(s). Replace with placeholders (/path/to/..., <partition>, "
            "<your_email>) or env vars. If a hit is legitimate (e.g. authorship metadata), add it to "
            "tools/lab_specific_allowlist.txt with a comment."
        )
        return 1
    print("OK: no lab/project-specific content found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
