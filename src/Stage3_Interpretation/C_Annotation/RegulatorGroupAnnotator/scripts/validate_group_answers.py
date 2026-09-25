"""Gate for regulator-group annotations. An answer that fails is moved aside and re-dispatched.

Per group (`<arm>_p<group id>/answer.json` next to its `prompt.md`):
  1. valid JSON with every required key; group_id matches the directory
  2. every member of the prompt's section B appears in `regulators` exactly once, with a valid
     role and confidence; an `unexplained` member carries a hypothesis and a test
  3. no EXCLUDED member (section C) appears in `regulators`, `label_evidence.regulators` or
     `shared_function.support_members`; no symbol outside the members does either
  4. every PMID cited anywhere is in the reference pool (section H)
  5. every program in `why_here.programs` is one of the section-D signature programs
  6. label rules shared with ProgramAnnotatorV3 (annotator_core/gate_rules.py) plus group words;
     no coherence talk in the brief summary; >= 2 competing readings; every confounder assessed
     with a valid status

Usage:
    python validate_group_answers.py --dispatch dispatch_groups --arm rg
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from answer_io import load_answer  # noqa: E402
from gate_rules import (  # noqa: E402
    VALID_STATUSES, bare_pmid, label_problems, pmids_offered_in_prompt, summary_problems,
)

REQUIRED_KEYS = [
    "group_id", "label", "brief_summary", "confounder_assessment", "shared_function", "why_here",
    "regulators", "label_evidence", "competing_readings", "coherence", "citations",
]
REQUIRED_CONFOUNDERS = {"generic_fitness_or_stress", "differentiation_delay", "promoter_neighbour", "weak_effect_noise"}
BANNED_LABEL_WORDS = {"group", "cluster", "module", "regulators", "program", "regulation"}
ROLES = {"core_explained", "consistent", "unexplained"}
CONFIDENCE = {"high", "medium", "low"}


def section(prompt: str, heading: str) -> str:
    parts = re.split(r"^## ", prompt, flags=re.MULTILINE)
    return next((p for p in parts if p.startswith(heading)), "")


def prompt_members(prompt: str) -> List[str]:
    return re.findall(r"^\| ([A-Za-z0-9._-]+) \| (?:core|peripheral|rescued)", section(prompt, "B. Members"), flags=re.MULTILINE)


def prompt_excluded(prompt: str) -> List[str]:
    return re.findall(r"^- ([A-Za-z0-9._-]+):", section(prompt, "C. EXCLUDED"), flags=re.MULTILINE)


def prompt_programs(prompt: str) -> set:
    return {int(p) for p in re.findall(r"^- program (\d+)", section(prompt, "D. What"), flags=re.MULTILINE)}


def symbol(value) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", str(value)).strip()


def validate(group_id: int, directory: Path) -> Tuple[List[str], List[str]]:
    tag = f"G{group_id}"
    answer_path, prompt_path = directory / "answer.json", directory / "prompt.md"
    if not answer_path.exists():
        return [f"{tag}: no answer.json"], []
    try:
        payload = load_answer(answer_path)
    except ValueError as exc:
        return [f"{tag}: answer is not valid JSON ({exc})"], []
    prompt = prompt_path.read_text()
    problems: List[str] = [f"{tag}: missing required key '{k}'" for k in REQUIRED_KEYS if k not in payload]
    warnings: List[str] = []
    if payload.get("group_id") not in (group_id, str(group_id)):
        problems.append(f"{tag}: group_id is {payload.get('group_id')!r}")

    members, excluded = prompt_members(prompt), set(prompt_excluded(prompt))
    roles = [symbol(r.get("symbol")) for r in payload.get("regulators", [])]
    missing = sorted(set(members) - set(roles))
    doubled = sorted({s for s in roles if roles.count(s) > 1})
    if missing:
        problems.append(f"{tag}: members without a role: {', '.join(missing)}")
    if doubled:
        problems.append(f"{tag}: members listed twice: {', '.join(doubled)}")
    for entry in payload.get("regulators", []):
        name = symbol(entry.get("symbol"))
        if entry.get("role") not in ROLES:
            problems.append(f"{tag}: {name} has invalid role {entry.get('role')!r}")
        if entry.get("confidence") not in CONFIDENCE:
            problems.append(f"{tag}: {name} has invalid confidence {entry.get('confidence')!r}")
        if entry.get("role") == "unexplained" and not (str(entry.get("hypothesis", "")).strip()
                                                       and str(entry.get("what_would_test_it", "")).strip()):
            problems.append(f"{tag}: unexplained member {name} needs a hypothesis and what_would_test_it")

    named = set(roles)
    named |= {symbol(r.get("symbol")) for r in (payload.get("label_evidence") or {}).get("regulators", [])}
    named |= {symbol(g) for g in (payload.get("shared_function") or {}).get("support_members", [])}
    named.discard("")
    if named & excluded:
        problems.append(f"{tag}: excluded member(s) interpreted: {', '.join(sorted(named & excluded))}")
    outsiders = sorted(named - set(members) - excluded)
    if outsiders:
        problems.append(f"{tag}: non-member(s) given a role or used as support: {', '.join(outsiders)}")

    offered = pmids_offered_in_prompt(prompt)
    cited = {bare_pmid(c.get("pmid")) for c in payload.get("citations", []) if c.get("pmid")}
    for block in [payload.get("shared_function") or {}, payload.get("why_here") or {}, *payload.get("regulators", [])]:
        cited |= {bare_pmid(p) for p in block.get("pmids") or []}
    invented = sorted(p for p in cited if p and p not in offered)
    if invented:
        problems.append(f"{tag}: PMID(s) cited that were not in the reference pool: {', '.join(invented)}")

    signature = prompt_programs(prompt)
    for entry in (payload.get("why_here") or {}).get("programs", []):
        try:
            pid = int(entry.get("program_id"))
        except (TypeError, ValueError):
            problems.append(f"{tag}: why_here program_id {entry.get('program_id')!r} is not a number")
            continue
        if pid not in signature:
            warnings.append(f"{tag}: why_here names program {pid}, not in the section-D signature")

    problems += label_problems(tag, str(payload.get("label", "")), BANNED_LABEL_WORDS)
    problems += summary_problems(tag, str(payload.get("brief_summary", "")))
    if len(payload.get("competing_readings", [])) < 2:
        problems.append(f"{tag}: only {len(payload.get('competing_readings', []))} competing reading(s), need >= 2")
    assessed = {c.get("confounder") for c in payload.get("confounder_assessment", [])}
    if REQUIRED_CONFOUNDERS - assessed:
        problems.append(f"{tag}: confounders not assessed: {', '.join(sorted(REQUIRED_CONFOUNDERS - assessed))}")
    bad = [c.get("confounder") for c in payload.get("confounder_assessment", []) if c.get("status") not in VALID_STATUSES]
    if bad:
        problems.append(f"{tag}: invalid status on {bad}")
    return problems, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--arm", default="rg")
    args = parser.parse_args()

    directories = sorted(args.dispatch.glob(f"{args.arm}_p*"), key=lambda d: int(d.name.split("_p")[-1]))
    all_problems, all_warnings = [], []
    for directory in directories:
        group_id = int(directory.name.split("_p")[-1])
        problems, warnings = validate(group_id, directory)
        all_warnings += warnings
        all_problems += problems
        if not problems:
            print(f"G{group_id}: PASS")
    for warning in all_warnings:
        print(f"WARN {warning}")
    for problem in all_problems:
        print(f"FAIL {problem}")
    print(f"\n{len(all_problems)} problem(s) across {len(directories)} groups")
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
