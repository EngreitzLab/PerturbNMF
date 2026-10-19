#!/usr/bin/env python3
"""Lint a hypothesis report .qmd against references/report-template.md.

Fails if the report talks about its own process, buries the hypotheses, or ships a card without
its claim, confidence, figure or experiment.

    python lint_hypothesis_report.py report/<study>_top10_hypotheses.qmd --n-hypotheses 10

Stdlib only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Phrase-level process terms. Single words that collide with ordinary biology vocabulary
# ("leak", "recovered", "round", "gate", "blind", "skill", "agent", "verdict") are deliberately
# left out — only ban them once they say something process-specific.
PROCESS_TERMS = [
    "admission gate", "funnel", "uncontaminated", "contaminated", "contaminated run",
    "predecessor", "predecessor study", "predecessor paper",
    "answer key", "held-out key", "held-out term",
    "evidence class", "selection rule",
    "ledger", "reviser", "falsifier", "sub-agent", "subagent", "the skill",
    "pipeline stage", "stage 4", "stage 5",
    "literature verdict", "leaked term",
    "blind run", "blind pass",
    "in round", "round 1", "round 2", "round 3",
    "kill test", "survived",
]
CARD_SLOTS = ["**Claim.**", "**Why investigate.**", "**Confidence:", "**What the screen shows.**",
              "**What is already known.**", "**Experiment.**", "*If true:*", "*If false:*"]
# Single-letter codes (X1, C6, H07, P36): ledger ids, v1 class codes, bare program numbers.
# Real gene symbols that happen to look like this are allowed by default; --allow adds more.
INTERNAL_CODE = re.compile(r"(?<![\w#/.-])[A-Z]\d{1,3}(?![\w/.-])")
DEFAULT_ALLOWED_CODES = {"C3", "C5", "F2", "F3", "F5", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "H19", "P2",
                         # cell lines
                         "K562", "A549", "U937", "T47D"}
IMAGE_MARKUP = re.compile(r"!\[[^\]]*\]\([^)]+\)|<img |include-graphics")
FIGURE_CHUNK = re.compile(r"```\{(?:r|python)[^}]*\}(.*?)```", re.S)


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (preamble, [(heading, body)]) for level-2 headings, skipping YAML front matter
    and code blocks. `preamble` is everything before the first '## ' heading (title/subtitle)."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    preamble_lines: list[str] = []
    sections, heading, lines, in_code = [], None, [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(lines)))
            heading, lines = line[3:].strip(), []
        elif heading is None:
            preamble_lines.append(line)
        else:
            lines.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(lines)))
    return "\n".join(preamble_lines), sections


def report_title(text: str) -> str:
    match = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", text, flags=re.M) or re.search(r"^# (.+)$", text, flags=re.M)
    return match.group(1).strip() if match else ""


def strip_links_and_code(body: str) -> str:
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"`[^`]*`", "", body)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    body = re.sub(r"\{#[^}]*\}", "", body)
    body = re.sub(r"<[^>]+>", "", body)
    return body


def has_figure(body: str) -> bool:
    """A markdown image, <img>, include-graphics, or an {r}/{python} chunk that renders a figure
    (carries a fig- label or fig-cap)."""
    if IMAGE_MARKUP.search(body):
        return True
    for chunk in FIGURE_CHUNK.finditer(body):
        content = chunk.group(1)
        if "label: fig-" in content or "fig-cap" in content:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("--n-hypotheses", type=int, required=True)
    parser.add_argument("--allow", nargs="*", default=[], help="extra gene symbols that look like internal codes, e.g. F30")
    parser.add_argument("--extra-banned", nargs="*", default=[], help="extra process terms/phrases to ban")
    args = parser.parse_args()

    text = args.report.read_text()
    errors: list[str] = []
    banned_terms = PROCESS_TERMS + args.extra_banned
    allowed_codes = DEFAULT_ALLOWED_CODES | set(args.allow)

    title = report_title(text)
    if not re.match(rf"Top {args.n_hypotheses} hypotheses\b", title):
        errors.append(f"title must start with 'Top {args.n_hypotheses} hypotheses', got: {title!r}")

    preamble, sections = split_sections(text)
    headings = [h for h, _ in sections]
    if len(headings) < 2 or not re.search(r"(test first|top picks|three)", headings[0], re.I):
        errors.append(f"first section must be the top picks (e.g. 'Three to test first'), got: {headings[:1]}")
    if len(headings) < 2 or not re.search(r"ranked", headings[1], re.I):
        errors.append(f"second section must be the ranked table (e.g. 'All N, ranked'), got: {headings[1:2]}")

    cards = [(h, b) for h, b in sections if re.search(r"\{#h\d+\}", h)]
    if len(cards) != args.n_hypotheses:
        errors.append(f"{len(cards)} hypothesis cards (headings with {{#hK}}), expected {args.n_hypotheses}")
    for heading, body in cards:
        name = re.sub(r"\s*\{#.*\}", "", heading)
        missing = [slot for slot in CARD_SLOTS if slot not in body]
        if missing:
            errors.append(f"card '{name}': missing {missing}")
        if not has_figure(body):
            errors.append(f"card '{name}': no figure")

    sections_with_preamble = [("preamble", preamble)] + sections
    for heading, body in sections_with_preamble:
        if heading.lower().startswith("methods"):
            continue
        prose = strip_links_and_code(heading + "\n" + body)
        for term in banned_terms:
            for match in re.finditer(rf"\b{re.escape(term)}\b", prose, flags=re.I):
                context = prose[max(0, match.start() - 40): match.end() + 40].replace("\n", " ")
                errors.append(f"process term '{term}' in section '{heading}': …{context}…")
        for match in INTERNAL_CODE.finditer(prose):
            if match.group() in allowed_codes:
                continue
            context = prose[max(0, match.start() - 40): match.end() + 40].replace("\n", " ")
            errors.append(f"internal code '{match.group()}' in section '{heading}' (use titles / gene names): …{context}…")

    for term in banned_terms:
        if re.search(rf"\b{re.escape(term)}\b", title, flags=re.I):
            errors.append(f"process term '{term}' in the title")

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        sys.exit(f"{len(errors)} report lint error(s)")
    print(f"report lint OK: {len(cards)} cards, title {title!r}")


if __name__ == "__main__":
    main()
