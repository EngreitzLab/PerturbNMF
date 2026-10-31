#!/usr/bin/env python3
"""Validate ledger/hypotheses.jsonl against references/ledger-schema.md.

Run after every loop step. Exits non-zero on any rule violation so the orchestrator cannot
advance a round that only annotated hypotheses instead of revising or killing them.

    python validate_hypothesis_ledger.py ledger/hypotheses.jsonl --round 2 \
        --snapshot-dir ledger/snapshots --max-rounds 3
    python validate_hypothesis_ledger.py ledger/hypotheses.jsonl --round 3 --final \
        --n-hypotheses 10 --analysis-dir .
    python validate_hypothesis_ledger.py ledger/hypotheses.jsonl --round 0 \
        --eligibility explore/regulator_eligibility.tsv    # after the Generator step

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STATUSES = {"candidate", "supported", "killed", "merged", "shelved"}
REVISER_ACTIONS = {
    "survived",
    "sharpened",
    "reframed",
    "merged",
    "split",
    "killed",
    "shelved",
}
ITERATION_ACTIONS = REVISER_ACTIONS - {"survived"}
USER_ACTIONS = {"user_kill", "user_push"}
VERSION_BUMP_ACTIONS = {"sharpened", "reframed"}
TERMINAL_RESULT_ACTIONS = {"killed", "merged", "shelved", "survived"}
TEST_RESULTS = {"supports", "refutes", "inconclusive", "not_run"}
TEST_KINDS = {"data", "external_data", "literature"}
LITERATURE_VERDICTS = {
    "known_in_this_cell_type",
    "known_elsewhere",
    "novel",
    "contradicted",
}
CONFIDENCE_LEVELS = {"low", "medium", "high"}
LEAD_ELIGIBILITY = {"yes", "flag", "no"}
OPEN_STATUSES = {"candidate", "supported"}
MIN_ITERATION_SHARE = 0.30

# What status a hypothesis must have once its most recent status-determining action is X.
STATUS_FOR_TERMINAL_ACTION = {
    "killed": "killed",
    "user_kill": "killed",
    "merged": "merged",
    "shelved": "shelved",
    "survived": "supported",
    "sharpened": "candidate",
    "reframed": "candidate",
    "split": "candidate",
}


def read_ledger(path: Path) -> list[dict]:
    entries = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                sys.exit(f"{path}:{line_number}: not valid JSON ({error})")
    return entries


def reviser_history_events(entry: dict) -> list[dict]:
    """Actual Reviser decisions — actor == 'reviser', in ledger (chronological) order."""
    return [
        h
        for h in entry.get("history", [])
        if h.get("actor") == "reviser" and h.get("action") in REVISER_ACTIONS
    ]


def round_completing_events(entry: dict) -> list[dict]:
    """Events that satisfy 'every round ends in an action': Reviser actions, or a user kill.

    A user push is advice to the Reviser, not a decision: the Reviser still has to act that round.
    """
    return [
        h
        for h in entry.get("history", [])
        if (h.get("actor") == "reviser" and h.get("action") in REVISER_ACTIONS)
        or (h.get("actor") == "user" and h.get("action") == "user_kill")
    ]


def status_determining_events(entry: dict) -> list[dict]:
    """Events whose action fixes the expected status: Reviser actions or a user kill."""
    return [
        h
        for h in entry.get("history", [])
        if (h.get("actor") == "reviser" and h.get("action") in REVISER_ACTIONS)
        or (h.get("actor") == "user" and h.get("action") == "user_kill")
    ]


def created_round(entry: dict) -> tuple[int, bool]:
    """Round the entry was created in, and whether it came from a split (born at round end)."""
    for event in entry.get("history", []):
        if event.get("action") == "created":
            round_value = event.get("round")
            return int(round_value) if round_value is not None else 1, event.get(
                "actor"
            ) == "reviser"
    return 1, False


def terminal_round(entry: dict) -> int | None:
    """Round the entry first reached a terminal-for-round-accounting state, if any."""
    for event in entry.get("history", []):
        actor, action = event.get("actor"), event.get("action")
        is_terminal = (actor == "reviser" and action in TERMINAL_RESULT_ACTIONS) or (
            actor == "user" and action == "user_kill"
        )
        if is_terminal:
            round_value = event.get("round")
            return int(round_value) if round_value is not None else None
    return None


def check_history_and_test_ids(entry: dict, errors: list[str]) -> None:
    """Rule: missing 'round' in history events or missing test 'id' are errors, not crashes."""
    hid = entry.get("id", "<no id>")
    for event in entry.get("history", []):
        if event.get("round") is None:
            errors.append(
                f"{hid}: history event '{event.get('action', '<no action>')}' missing 'round'"
            )
    for test in entry.get("tests", []):
        if not test.get("id"):
            errors.append(f"{hid}: a test is missing 'id'")


def check_entry_fields(entry: dict, errors: list[str]) -> None:
    hid = entry.get("id", "<no id>")
    for field in ("id", "version", "status", "title", "claim", "tests", "history"):
        if field not in entry:
            errors.append(f"{hid}: missing field '{field}'")
    if entry.get("status") not in STATUSES:
        errors.append(
            f"{hid}: status '{entry.get('status')}' not in {sorted(STATUSES)}"
        )
    for test in entry.get("tests", []):
        tid = f"{hid}/{test.get('id', '<no id>')}"
        if test.get("kind") not in TEST_KINDS:
            errors.append(
                f"{tid}: kind '{test.get('kind')}' not in {sorted(TEST_KINDS)}"
            )
        if test.get("result", "not_run") not in TEST_RESULTS:
            errors.append(
                f"{tid}: result '{test.get('result')}' not in {sorted(TEST_RESULTS)}"
            )
        if not test.get("kill_if"):
            errors.append(
                f"{tid}: no 'kill_if' — a test that cannot kill the hypothesis is not a test"
            )
        for field in ("registered_round", "hypothesis_version"):
            if field not in test:
                errors.append(f"{tid}: missing '{field}'")
    for review in entry.get("literature", []):
        if review.get("verdict") not in LITERATURE_VERDICTS:
            errors.append(
                f"{hid}: literature verdict '{review.get('verdict')}' not in {sorted(LITERATURE_VERDICTS)}"
            )
        if not review.get("citations"):
            errors.append(
                f"{hid}: literature verdict in round {review.get('round')} has no citations"
            )
        for citation in review.get("citations", []):
            if not citation.get("id") or not citation.get("says"):
                errors.append(
                    f"{hid}: every citation needs an 'id' (PMID/DOI) and what it 'says'"
                )


def check_version(entry: dict, errors: list[str]) -> None:
    """Rule 5: version == 1 + number of sharpened/reframed actions; each needs previous_claim."""
    hid = entry.get("id", "<no id>")
    version_bumps = [
        e
        for e in reviser_history_events(entry)
        if e.get("action") in VERSION_BUMP_ACTIONS
    ]
    expected_version = 1 + len(version_bumps)
    if entry.get("version") != expected_version:
        errors.append(
            f"{hid}: version {entry.get('version')} but {len(version_bumps)} sharpened/reframed "
            f"action(s) imply version {expected_version}"
        )
    for event in version_bumps:
        if not event.get("previous_claim"):
            errors.append(
                f"{hid}: {event.get('action')} in round {event.get('round', '?')} missing 'previous_claim'"
            )


def check_status_matches_terminal_action(entry: dict, errors: list[str]) -> None:
    """Status must match the most recent status-determining action, error on mismatch."""
    hid = entry.get("id", "<no id>")
    events = status_determining_events(entry)
    if not events:
        return
    last_action = events[-1].get("action")
    expected_status = STATUS_FOR_TERMINAL_ACTION.get(last_action)
    if expected_status is None:
        return
    status = entry.get("status")
    if status != expected_status:
        errors.append(
            f"{hid}: last action '{last_action}' implies status '{expected_status}', got '{status}'"
        )
        return
    if expected_status == "killed" and not entry.get("kill_reason"):
        errors.append(f"{hid}: killed without a kill_reason")
    if expected_status == "merged" and not entry.get("merged_into"):
        errors.append(f"{hid}: merged without merged_into")


def check_rounds(
    entry: dict, current_round: int, max_rounds: int, errors: list[str]
) -> None:
    """Rule 2 (every round ends in an action) and rule 8 (budget)."""
    hid = entry.get("id", "<no id>")
    born, born_from_split = created_round(entry)
    first_round = born + 1 if born_from_split else born
    ended = terminal_round(entry)
    last_round = min(current_round, ended) if ended else current_round
    actions_by_round: dict[int, list[dict]] = {}
    for event in round_completing_events(entry):
        round_value = event.get("round")
        if round_value is None:
            continue  # already reported by check_history_and_test_ids
        actions_by_round.setdefault(int(round_value), []).append(event)
    for round_number in range(first_round, last_round + 1):
        events = actions_by_round.get(round_number, [])
        actions = [e.get("action") for e in events]
        if not actions:
            errors.append(
                f"{hid}: no Reviser action in round {round_number}. Adding results or a literature "
                "verdict is not an action — survive, sharpen, reframe, merge, split, kill or shelve."
            )
        elif len(actions) > 1:
            all_but_last_are_split = all(a == "split" for a in actions[:-1])
            one_extra_repass = len(actions) == 2 and any(
                e.get("repass") for e in events
            )
            if not (all_but_last_are_split or one_extra_repass):
                errors.append(
                    f"{hid}: {len(actions)} Reviser actions in round {round_number}: {actions}"
                )
    extra_passes = sum(
        1 for c in entry.get("critique", []) if c.get("severity") == "major"
    )
    budget = max_rounds + min(extra_passes, 1)
    reviser_events = reviser_history_events(entry)
    if len(reviser_events) > budget:
        errors.append(
            f"{hid}: {len(reviser_events)} Reviser actions exceeds budget {budget}"
        )


def check_survival(entry: dict, errors: list[str]) -> None:
    """Rules 3 and 5: 'survived' must be earned, keyed on the LAST Reviser action, and a
    version bump needs new tests plus literature for the current version."""
    hid = entry.get("id", "<no id>")
    events = reviser_history_events(entry)
    if not events or events[-1].get("action") != "survived":
        if entry.get("status") == "supported":
            errors.append(
                f"{hid}: status 'supported' without the last Reviser action being 'survived'"
            )
        return
    version = entry.get("version")
    current_tests = [
        t for t in entry.get("tests", []) if t.get("hypothesis_version") == version
    ]
    if not current_tests:
        errors.append(
            f"{hid}: survived at version {version} with no tests registered against that version"
        )
    unrun = [
        t.get("id", "<no id>")
        for t in current_tests
        if t.get("result", "not_run") == "not_run"
    ]
    if unrun:
        errors.append(f"{hid}: survived with tests not run: {unrun}")
    counterexamples = [
        t
        for t in current_tests
        if t.get("counterexample")
        and t.get("kind") in {"data", "external_data"}
        and t.get("result") == "supports"
    ]
    if not counterexamples:
        errors.append(
            f"{hid}: survived without a data counter-example test that ran and came back "
            "'supports' (inconclusive does not count)"
        )
    refuted = [
        t.get("id", "<no id>") for t in current_tests if t.get("result") == "refutes"
    ]
    if refuted:
        errors.append(
            f"{hid}: survived although tests refuted it: {refuted} — kill or revise instead"
        )

    # Round of the last version bump before this survival — literature must be at least that fresh.
    version_bump_round = None
    for event in reversed(events[:-1]):
        if event.get("action") in VERSION_BUMP_ACTIONS:
            version_bump_round = event.get("round")
            break
    reviews = entry.get("literature", [])
    if not reviews:
        errors.append(f"{hid}: survived with no literature review")
        return
    if version_bump_round is not None:
        applicable = [r for r in reviews if (r.get("round") or 0) >= version_bump_round]
        if not applicable:
            errors.append(
                f"{hid}: survived at version {version} but no literature review at or after round "
                f"{version_bump_round} (the round of the last version bump)"
            )
        latest = applicable[-1] if applicable else reviews[-1]
    else:
        latest = reviews[-1]
    if latest.get("verdict") == "known_in_this_cell_type":
        errors.append(
            f"{hid}: survived although the latest literature verdict for this version is known_in_this_cell_type"
        )


def check_preregistration(
    entries: list[dict], snapshot_dir: Path, errors: list[str]
) -> None:
    """Rule 4: every test with a result appears without a result in an earlier snapshot."""
    registered_unrun: set[tuple[str, str, int]] = set()
    for snapshot in sorted(snapshot_dir.glob("r*_*.jsonl")):
        for entry in read_ledger(snapshot):
            for test in entry.get("tests", []):
                if test.get("result", "not_run") == "not_run":
                    registered_unrun.add(
                        (
                            entry.get("id"),
                            test.get("id"),
                            test.get("hypothesis_version"),
                        )
                    )
    for entry in entries:
        for test in entry.get("tests", []):
            if test.get("result", "not_run") != "not_run":
                key = (entry.get("id"), test.get("id"), test.get("hypothesis_version"))
                if key not in registered_unrun:
                    errors.append(
                        f"{entry.get('id')}/{test.get('id')}: has a result but never appears unrun in a snapshot "
                        "— tests must be registered (and snapshotted) before they are run"
                    )


def check_final(
    entries: list[dict],
    n_hypotheses: int,
    analysis_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Rules 6 and 7: writer eligibility and iteration health."""
    supported = [e for e in entries if e.get("status") == "supported"]
    if len(supported) < n_hypotheses:
        warnings.append(
            f"{len(supported)} supported hypotheses, fewer than the {n_hypotheses} requested — report what survived, do not pad"
        )
    for entry in supported:
        hid = entry.get("id", "<no id>")
        if entry.get("confidence") not in CONFIDENCE_LEVELS:
            errors.append(
                f"{hid}: confidence '{entry.get('confidence')}' not in {sorted(CONFIDENCE_LEVELS)}"
            )
        why_investigate = entry.get("why_investigate") or ""
        if len(why_investigate.split()) < 25:
            errors.append(
                f"{hid}: why_investigate missing or under 25 words — say what the finding would mean if true"
            )
        experiment = entry.get("experiment") or {}
        for field in ("design", "if_true", "if_false"):
            if not experiment.get(field):
                errors.append(f"{hid}: experiment.{field} is empty")
        figures = [
            t.get("headline_figure")
            for t in entry.get("tests", [])
            if t.get("kind") in {"data", "external_data"} and t.get("headline_figure")
        ]
        if not figures:
            errors.append(f"{hid}: no data test with a headline_figure")
        for figure in figures:
            if not (analysis_dir / figure).exists():
                errors.append(f"{hid}: headline_figure {figure} does not exist")
        cartoon = entry.get("cartoon")
        if not cartoon:
            errors.append(
                f"{hid}: no 'cartoon' — every supported hypothesis needs a mechanism "
                "schematic (references/cartoons.md)"
            )
        elif not (analysis_dir / cartoon).exists():
            errors.append(f"{hid}: cartoon {cartoon} does not exist")
        open_majors = [
            c
            for c in entry.get("critique", [])
            if c.get("severity") == "major" and not c.get("resolved")
        ]
        if open_majors:
            errors.append(f"{hid}: {len(open_majors)} unresolved Major critique(s)")
    ranked = sorted((e for e in supported if e.get("rank")), key=lambda e: e["rank"])
    if supported and len(ranked) != len(supported):
        errors.append("every supported hypothesis needs a rank")
    iterated = [
        e
        for e in entries
        if any(a.get("action") in ITERATION_ACTIONS for a in reviser_history_events(e))
    ]
    share = len(iterated) / len(entries) if entries else 0.0
    message = f"iteration health: {len(iterated)}/{len(entries)} candidates ({share:.0%}) were revised, merged, split, killed or shelved"
    if share < MIN_ITERATION_SHARE:
        warnings.append(
            message
            + f" — below {MIN_ITERATION_SHARE:.0%}; the loop annotated rather than iterated"
        )
    else:
        print(message)


def read_eligibility(path: Path) -> dict[str, tuple[str, str]]:
    """target -> (lead_eligible, reason) from the Explorer's regulator_eligibility.tsv."""
    lines = [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    header = lines[0].split("\t")
    for column in ("target", "lead_eligible", "reason"):
        if column not in header:
            sys.exit(f"{path}: missing column '{column}'")
    eligibility = {}
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        if row["lead_eligible"] not in LEAD_ELIGIBILITY:
            sys.exit(
                f"{path}: {row['target']}: lead_eligible '{row['lead_eligible']}' "
                f"not in {sorted(LEAD_ELIGIBILITY)}"
            )
        eligibility[row["target"]] = (row["lead_eligible"], row["reason"])
    return eligibility


def check_member_eligibility(
    entry: dict,
    eligibility: dict[str, tuple[str, str]],
    errors: list[str],
    warnings: list[str],
) -> None:
    """An open hypothesis may not rest on a member that failed the Explorer's kill-checks."""
    if entry.get("status") not in OPEN_STATUSES:
        return
    hid = entry.get("id", "<no id>")
    for member in entry.get("members", []):
        verdict, reason = eligibility.get(member, ("missing", ""))
        if verdict == "no":
            errors.append(
                f"{hid}: member {member} is not lead-eligible ({reason}); "
                "cite it as supporting evidence instead, or drop it"
            )
        elif verdict == "missing":
            warnings.append(
                f"{hid}: member {member} is not an admitted regulator, so its "
                "kill-checks were not run; the Falsifier must register them"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--round", type=int, required=True, help="the round that has just finished"
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--n-hypotheses", type=int, default=10)
    parser.add_argument("--analysis-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--eligibility",
        type=Path,
        help="explore/regulator_eligibility.tsv; open hypotheses may not rest on a 'no' member",
    )
    parser.add_argument(
        "--mid-round",
        action="store_true",
        help="skip the end-of-round action check (use after Falsifier/Analyst/Literature steps)",
    )
    args = parser.parse_args()

    entries = read_ledger(args.ledger)
    errors: list[str] = []
    warnings: list[str] = []
    eligibility = read_eligibility(args.eligibility) if args.eligibility else None
    ids = [e.get("id") for e in entries]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        errors.append(f"duplicate ids: {sorted(duplicates)}")
    for entry in entries:
        check_history_and_test_ids(entry, errors)
        check_entry_fields(entry, errors)
        check_version(entry, errors)
        check_status_matches_terminal_action(entry, errors)
        completed_round = args.round - 1 if args.mid_round else args.round
        if completed_round >= 1:
            check_rounds(entry, completed_round, args.max_rounds, errors)
        check_survival(entry, errors)
        if eligibility is not None:
            check_member_eligibility(entry, eligibility, errors, warnings)
    if args.snapshot_dir:
        check_preregistration(entries, args.snapshot_dir, errors)
    if args.final:
        check_final(entries, args.n_hypotheses, args.analysis_dir, errors, warnings)

    counts = {s: sum(e.get("status") == s for e in entries) for s in sorted(STATUSES)}
    print("status counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        sys.exit(f"{len(errors)} ledger error(s)")
    print("ledger OK")


if __name__ == "__main__":
    main()
