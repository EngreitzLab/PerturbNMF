#!/usr/bin/env python3
"""Fail if any held-out term appears in the text a blind run will read.

Terms come from a plain file kept OUTSIDE the run directory (one term per line; `#` comments).
By default every term matches case-insensitively on whole-token boundaries, so a mouse symbol
like "Insr" also catches the human "INSR" written in a doc. Pass --case-sensitive to restore
exact-case matching (useful for a term that is a common English word in lower case). Data inputs
are not scanned: gene names legitimately appear there. Every non-binary file under the given
paths is scanned (binary = a NUL byte in the first 8 KB); `.git` directories are skipped.

    python check_for_leaked_terms.py --terms ~/held_out/my_study_terms.txt \
        --paths RUN_DIR/.claude RUN_DIR/study.yaml RUN_DIR/README.md

Stdlib only. Exit 1 on any hit.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BINARY_SNIFF_BYTES = 8192


def read_terms(path: Path) -> list[str]:
    terms = []
    for line in path.read_text().splitlines():
        term = line.split("#", 1)[0].strip()
        if term:
            terms.append(term)
    if not terms:
        sys.exit(f"{path}: no terms")
    return terms


def compile_term(term: str, case_sensitive: bool) -> re.Pattern:
    flags = 0 if case_sensitive else re.I
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", flags)


def is_binary_file(path: Path) -> bool:
    with open(path, "rb") as handle:
        return b"\0" in handle.read(BINARY_SNIFF_BYTES)


def iter_text_files(paths: list[Path]):
    for path in paths:
        if path.is_file():
            if not is_binary_file(path):
                yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if ".git" in child.parts:
                    continue
                if child.is_file() and not is_binary_file(child):
                    yield child


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--case-sensitive", action="store_true",
                        help="match terms exactly as written instead of case-insensitively")
    args = parser.parse_args()

    patterns = {term: compile_term(term, args.case_sensitive) for term in read_terms(args.terms)}
    hits = 0
    scanned = 0
    for file in iter_text_files(args.paths):
        scanned += 1
        for line_number, line in enumerate(file.read_text(errors="replace").splitlines(), start=1):
            for term, pattern in patterns.items():
                if pattern.search(line):
                    hits += 1
                    print(f"LEAK {file}:{line_number}: '{term}': {line.strip()[:140]}")
    print(f"scanned {scanned} files for {len(patterns)} held-out terms: {hits} hit(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
