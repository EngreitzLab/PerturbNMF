"""Validate v3 annotation JSON against the evidence it was given.

This is the gate the v2 design calls for: an annotation that cites a gene it was never shown,
or a PMID that was not in its reference pool, is a defect regardless of how good the prose is.
Reporting the defect is the point — do not silently repair it.

Checks, per program:
  1. valid JSON with every required top-level key
  2. every gene named in label_evidence / modules / unplaced_genes appears in the prompt's own
     gene lists
  3. every PMID cited appears in the prompt's reference pool
  4. label is <= 6 words and free of the banned words
  5. >= 2 competing_readings, and every Step-1 confounder carries a status
  6. time courses only (the prompt carries a stage_composition screen): stage_composition is
     assessed, temporal_window is filled, and its peak day matches the prompt's. A regulator
     timed to a day on which it was not significant is reported as a WARN, not a failure.

Usage:
    python validate_annotation_answers.py --dispatch <annotation_dispatch> --arm v3 --programs 0-49
    python validate_annotation_answers.py --dispatch <annotation_dispatch> --arm v3 --programs 2,12,36
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "annotator_core"))
from record_item_problems import record_item_problems  # noqa: E402
from gate_rules import (  # noqa: E402
    VALID_STATUSES, bare_pmid, label_problems, pmids_offered_in_prompt, summary_problems,
)

REQUIRED_KEYS = [
    "program_id", "label", "brief_summary", "confounder_assessment", "interpretation",
    "label_evidence", "competing_readings", "coherence", "modules", "regulators", "citations",
]
REQUIRED_CONFOUNDERS = {
    "positional", "cell_cycle", "technical_qc", "essentiality_or_growth_arrest",
    "rna_processing_or_decay", "cis_target_effects", "ribosome_translation_housekeeping",
}
BANNED_LABEL_WORDS = {"program", "process", "regulation"}


def genes_offered_in_prompt(prompt: str) -> set:
    """Every symbol-shaped token anywhere in the prompt.

    Deliberately the WHOLE prompt, not just the two gene tables. The rule the model is held to
    is "every gene you mention must appear in the supplied evidence" — and the enrichment
    overlap lists, the confounder screens and the gene summaries are all supplied evidence. An
    earlier version of this check only scanned sections A and B and flagged four grounded
    answers as fabrications, which is exactly the kind of false alarm that trains people to
    ignore a gate.
    """
    return set(re.findall(r"\b[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,6})?\b", prompt)) | set(
        re.findall(r"\b[A-Z][A-Za-z0-9.-]{2,20}\b", prompt)
    )


PEAK_DAY = re.compile(r"^Peak day: (\S+)", re.MULTILINE)
CROSS_DAY_ROW = re.compile(r"^- ([A-Za-z0-9.-]+): ((?:\S+ (?:[+-]\d+\.\d+\*?|n/a)\s*)+)", re.MULTILINE)


def significant_days_by_regulator(prompt: str) -> dict:
    """{regulator: {days it was significant}} from the prompt's cross-day profile."""
    section = prompt.split("### Cross-day profile", 1)
    if len(section) < 2:
        return {}
    body = section[1].split("\n## ", 1)[0]
    days = {}
    for gene, cells in CROSS_DAY_ROW.findall(body):
        days[gene] = {day for day, value in re.findall(r"(\S+) ([+-]\d+\.\d+\*?|n/a)", cells) if value.endswith("*")}
    return days


def validate_time_course(program_id: int, payload: dict, prompt: str) -> Tuple[List[str], List[str]]:
    problems, warnings = [], []
    window = (payload.get("interpretation") or {}).get("temporal_window")
    if not isinstance(window, dict) or not str(window.get("claim", "")).strip():
        return [f"P{program_id}: temporal_window not filled"], warnings
    expected_peak = PEAK_DAY.search(prompt)
    claimed_peak = re.match(r"\s*([A-Za-z0-9]+)", str(window.get("peak_condition", "")))
    if expected_peak and (not claimed_peak or claimed_peak.group(1) != expected_peak.group(1)):
        problems.append(
            f"P{program_id}: temporal_window.peak_condition={window.get('peak_condition')!r}, "
            f"prompt says {expected_peak.group(1)}"
        )
    significant_days = significant_days_by_regulator(prompt)
    for entry in window.get("regulator_timing") or []:
        symbol = str(entry.get("symbol", "")).strip()
        if symbol not in significant_days:
            warnings.append(f"P{program_id}: timed regulator {symbol!r} is not significant on any day")
            continue
        unsupported = sorted(set(entry.get("conditions") or []) - significant_days[symbol])
        if unsupported:
            warnings.append(
                f"P{program_id}: {symbol} timed to {unsupported}, significant only on "
                f"{sorted(significant_days[symbol])}"
            )
    return problems, warnings


def validate(program_id: int, directory: Path) -> Tuple[List[str], List[str]]:
    problems: List[str] = []
    answer_path = directory / "answer.json"
    prompt_path = directory / "prompt.md"

    if not answer_path.exists():
        return [f"P{program_id}: no answer.json"], []

    raw = re.sub(r"^```(?:json)?|```$", "", answer_path.read_text().strip(), flags=re.MULTILINE)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"P{program_id}: answer is not valid JSON ({exc})"], []

    for key in REQUIRED_KEYS:
        if key not in payload:
            problems.append(f"P{program_id}: missing required key '{key}'")

    prompt = prompt_path.read_text()
    offered_genes = genes_offered_in_prompt(prompt)
    offered_pmids = pmids_offered_in_prompt(prompt)

    cited_genes = set()
    for entry in (payload.get("label_evidence") or {}).get("genes", []):
        if entry.get("symbol"):
            cited_genes.add(entry["symbol"])
    for module in payload.get("modules", []):
        cited_genes.update(module.get("genes", []))
    cited_genes.update(payload.get("unplaced_genes", []))

    # Models sometimes annotate a symbol in place ("CDH5(regulator)"); check the symbol itself.
    cited_genes = {re.sub(r"\s*\(.*\)\s*$", "", str(g)).strip() for g in cited_genes}
    # Symbols such as 7SK-1, Y_RNA or 5_8S_rRNA escape the symbol regexes; accept any symbol
    # that appears literally in the prompt as a whole token.
    invented_genes = sorted(
        g for g in cited_genes
        if g and g not in offered_genes
        and not re.search(rf"(?<![\w-]){re.escape(g)}(?![\w-])", prompt)
    )
    if invented_genes:
        problems.append(
            f"P{program_id}: {len(invented_genes)} gene(s) named but not in the prompt's gene "
            f"lists: {', '.join(invented_genes[:12])}"
        )

    # Normalise "PMID:123" to "123" before comparing, or a correctly-pooled citation is reported
    # as a fabrication.
    cited_pmids = {bare_pmid(c.get("pmid")) for c in payload.get("citations", []) if c.get("pmid")}
    for module in payload.get("modules", []):
        cited_pmids.update(bare_pmid(p) for p in (module.get("support") or {}).get("pmids", []))
    invented_pmids = sorted(p for p in cited_pmids if p and p not in offered_pmids)
    if invented_pmids:
        problems.append(
            f"P{program_id}: PMID(s) cited that were not in the reference pool: "
            f"{', '.join(invented_pmids)}"
        )

    problems += label_problems(f"P{program_id}", str(payload.get("label", "")), BANNED_LABEL_WORDS)
    problems += summary_problems(f"P{program_id}", str(payload.get("brief_summary", "")))

    readings = payload.get("competing_readings", [])
    if len(readings) < 2:
        problems.append(f"P{program_id}: only {len(readings)} competing reading(s), need >= 2")

    time_course = "stage composition (time course)" in prompt
    required = REQUIRED_CONFOUNDERS | ({"stage_composition"} if time_course else set())
    assessed = {c.get("confounder") for c in payload.get("confounder_assessment", [])}
    missing = required - assessed
    if missing:
        problems.append(f"P{program_id}: confounders not assessed: {', '.join(sorted(missing))}")
    bad_status = [
        c.get("confounder")
        for c in payload.get("confounder_assessment", [])
        if c.get("status") not in VALID_STATUSES
    ]
    if bad_status:
        problems.append(f"P{program_id}: invalid status on {bad_status}")

    warnings: List[str] = []
    if time_course:
        timing_problems, warnings = validate_time_course(program_id, payload, prompt)
        problems += timing_problems

    return problems, warnings


def write_problems_file(directory: Path, problems: List[str]) -> None:
    """Record this gate's problems in <directory>/problems.json for the repair pass
    (annotator_core/repair_rejected_answers.sh); a passing directory clears this gate's entry.
    The "P<id>: " prefix is dropped: the repair call sees one item only."""
    record_item_problems(directory, "validate_annotation_answers", [re.sub(r"^P\d+: ", "", p) for p in problems])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--programs", required=True, help="comma list, or a range like 0-49")
    parser.add_argument("--arm", default="v3")
    parser.add_argument("--write-problems", action="store_true",
                        help="record each failing item's problems in problems.json for repair_rejected_answers.sh")
    args = parser.parse_args()

    if re.fullmatch(r"\d+-\d+", args.programs):
        first, last = map(int, args.programs.split("-"))
        program_ids = list(range(first, last + 1))
    else:
        program_ids = [int(p) for p in args.programs.split(",")]

    all_problems: List[str] = []
    all_warnings: List[str] = []
    for program_id in program_ids:
        problems, warnings = validate(program_id, args.dispatch / f"{args.arm}_p{program_id}")
        if args.write_problems:
            write_problems_file(args.dispatch / f"{args.arm}_p{program_id}", problems)
        all_warnings += warnings
        if problems:
            all_problems += problems
        else:
            print(f"P{program_id}: PASS")

    for warning in all_warnings:
        print(f"WARN {warning}")
    for problem in all_problems:
        print(f"FAIL {problem}")
    print(f"\n{len(all_problems)} problem(s) across the validated programs")
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
