#!/usr/bin/env python3
"""Print a compact, role-shaped view of the hypothesis ledger.

The ledger grows to ~1 MB over a run (about 8k tokens per entry), mostly superseded tests and
history. Agents read this view instead of dumping `ledger/hypotheses.jsonl`. The ledger itself is
unchanged and the validator still reads all of it. Agents WRITE to the ledger as before.

    python print_ledger_view.py ledger/hypotheses.jsonl --ids H03,H07          # one batch
    python print_ledger_view.py ledger/hypotheses.jsonl --status candidate     # still open
    python print_ledger_view.py ledger/hypotheses.jsonl --status supported --view writer

Views:
    brief   (default) current version only: claim, predictions, tests registered against the
            current version (spec, result, summary, trace), the latest literature check,
            unresolved critique, experiment, and one line per history event.
    writer  what the Critic, Ranker and Writer need: brief minus history, test specs and
            literature notes paths; keeps test summaries, headline figures and citations.
    full    the raw entry, pretty-printed. For a single entry when a role truly needs everything.

Stdlib only. Run as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CORE_FIELDS = [
    "id",
    "version",
    "status",
    "title",
    "claim",
    "mechanism",
    "why_investigate",
    "members",
    "programs",
    "tags",
    "study_question",
    "definitions",
]
OUTCOME_FIELDS = [
    "confidence",
    "experiment",
    "ratings",
    "rank",
    "top_pick",
    "cartoon",
    "kill_reason",
    "merged_into",
]


def read_ledger(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def current_version_tests(entry: dict) -> list[dict]:
    version = entry.get("version", 1)
    return [
        test
        for test in entry.get("tests", [])
        if test.get("hypothesis_version", 1) == version
    ]


def latest_literature(entry: dict) -> dict | None:
    checks = entry.get("literature", [])
    return max(checks, key=lambda check: check.get("round", 0)) if checks else None


def history_lines(entry: dict) -> list[str]:
    lines = []
    for event in entry.get("history", []):
        line = f"r{event.get('round')} {event.get('action')} ({event.get('actor')})"
        if event.get("repass"):
            line += " [repass]"
        if event.get("reason"):
            line += f": {event['reason']}"
        lines.append(line)
    return lines


def brief_view(entry: dict) -> dict:
    view = {
        field: entry[field]
        for field in CORE_FIELDS
        if entry.get(field) not in (None, "", [], {})
    }
    view["predictions"] = entry.get("predictions", [])
    view["tests_current_version"] = current_version_tests(entry)
    superseded = len(entry.get("tests", [])) - len(view["tests_current_version"])
    if superseded:
        view["tests_superseded"] = f"{superseded} tests on earlier versions omitted"
    literature = latest_literature(entry)
    if literature:
        view["literature_latest"] = literature
    open_critique = [
        item for item in entry.get("critique", []) if not item.get("resolved")
    ]
    if open_critique:
        view["critique_open"] = open_critique
    view.update(
        {
            field: entry[field]
            for field in OUTCOME_FIELDS
            if entry.get(field) not in (None, "", [], {})
        }
    )
    view["history"] = history_lines(entry)
    return view


def writer_view(entry: dict) -> dict:
    view = brief_view(entry)
    view.pop("history", None)
    view.pop("tests_superseded", None)
    view["tests_current_version"] = [
        {
            key: test.get(key)
            for key in (
                "id",
                "kind",
                "counterexample",
                "result",
                "summary",
                "trace",
                "headline_figure",
            )
            if test.get(key) is not None
        }
        for test in view["tests_current_version"]
    ]
    literature = view.get("literature_latest")
    if literature:
        view["literature_latest"] = {
            key: literature.get(key)
            for key in ("round", "verdict", "summary", "citations")
        }
    return view


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--ids", help="comma-separated entry ids, e.g. H03,H07")
    parser.add_argument(
        "--status", help="comma-separated statuses, e.g. candidate or supported"
    )
    parser.add_argument("--view", choices=["brief", "writer", "full"], default="brief")
    args = parser.parse_args()

    entries = read_ledger(args.ledger)
    if args.ids:
        wanted = set(args.ids.split(","))
        missing = wanted - {entry.get("id") for entry in entries}
        if missing:
            sys.exit(f"not in ledger: {', '.join(sorted(missing))}")
        entries = [entry for entry in entries if entry.get("id") in wanted]
    if args.status:
        statuses = set(args.status.split(","))
        entries = [entry for entry in entries if entry.get("status") in statuses]

    render = {"brief": brief_view, "writer": writer_view, "full": lambda entry: entry}[
        args.view
    ]
    for entry in entries:
        print(json.dumps(render(entry), indent=1, ensure_ascii=False))
        print()
    print(f"# {len(entries)} entries, view={args.view}", file=sys.stderr)


if __name__ == "__main__":
    main()
