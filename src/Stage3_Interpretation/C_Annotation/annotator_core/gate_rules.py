"""Rules every annotation gate applies, whatever was annotated (a program, a regulator group).

A gate is deterministic: an answer that breaks a rule is moved aside and re-dispatched. The
rules here are the ones that do not depend on the subject: the label is short, names biology
rather than the answer's own quality, carries no gene tag; every PMID was offered in the prompt.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# Words that describe the answer's quality rather than its biology. Coherence has its own field.
QUALITY_LABEL_PHRASES = [
    "grab bag", "grab-bag", "incoherent", "coherence", "heterogeneous", "mixed", "unclear",
    "miscellaneous",
]
QUALITY_SUMMARY_PHRASES = ["coheren", "incoherent", "heterogene", "grab bag", "grab-bag"]
PARENTHESISED_GENE = re.compile(r"\([A-Z][A-Z0-9-]{1,9}\)")
VALID_STATUSES = {"primary_explanation", "contributing", "ruled_out", "cannot_assess"}
MAX_LABEL_WORDS = 6


def pmids_offered_in_prompt(prompt: str) -> set:
    return set(re.findall(r"PMID:(\d+)", prompt))


def bare_pmid(value) -> str:
    """Models write the identifier both ways ("42440233" and "PMID:42440233")."""
    return re.sub(r"^\s*PMID[:\s]*", "", str(value), flags=re.IGNORECASE).strip()


def label_problems(prefix: str, label: str, banned_words: Iterable[str]) -> List[str]:
    problems = []
    # " / " and " - " are label separators, not words.
    words = [w for w in label.split() if w not in {"/", "-"}]
    if len(words) > MAX_LABEL_WORDS:
        problems.append(f"{prefix}: label is {len(words)} words (max {MAX_LABEL_WORDS}): {label!r}")
    lowered = {w.strip(",.").lower() for w in words}
    hit = lowered & set(banned_words)
    if hit:
        problems.append(f"{prefix}: label uses banned word(s) {sorted(hit)}: {label!r}")
    for phrase in QUALITY_LABEL_PHRASES:
        if phrase in label.lower():
            problems.append(f"{prefix}: label uses quality word {phrase!r}: {label!r}")
    if PARENTHESISED_GENE.search(label):
        problems.append(f"{prefix}: label carries a parenthesised gene tag: {label!r}")
    return problems


def summary_problems(prefix: str, summary: str) -> List[str]:
    lowered = summary.lower()
    return [f"{prefix}: brief_summary comments on coherence ({phrase!r})"
            for phrase in QUALITY_SUMMARY_PHRASES if phrase in lowered]
