"""Gate for the citation pass: every choice must be one of the candidates it was offered.

Per program:
  1. valid JSON; every claim in candidates.json answered exactly once
  2. every support's `ref` exists under THAT claim (L<n> literature, D<n> database)
  3. literature: the PMID matches the ref, and `quote` is verbatim from its sentence or the
     paper's title (both are shown in the prompt); an elision mark ("...") is allowed between
     verbatim pieces, paraphrase is not
  4. database: every PMID given (a comma list is allowed) is one the database entry itself cites —
     its GO annotation's reference, or a "[PubMed N]" printed in the entry text (gene summaries)
  5. a claim with no supports carries a none_reason
  6. a support with role "discovery" is a literature candidate that is not a review
An id slip (the ref points at the wrong literature candidate, but the PMID and quote match
exactly one other literature candidate under the same claim) is resolved to that candidate and
reported as a WARN — resolve_literature_ref() is shared with the viewer so both agree.
Existence and retraction of the chosen PMIDs is checked separately by verify_cited_pmids.py
(--answer-key claims). Entailment (does the sentence really support the claim) is NOT checked.

Prints coverage: claims with a direct PMID, any PMID, database-only, none.

With --write-problems, each program's problems go to <dir>/problems.json for the repair pass
(repair_rejected_answers.sh); a passing program clears this gate's entry.

Usage:
    python validate_citation_answers.py --dispatch <citations_dispatch> --arm cite [--write-problems]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from record_item_problems import record_item_problems


def load(path: Path):
    return json.loads(re.sub(r"^```(?:json)?|```$", "", path.read_text().strip(), flags=re.MULTILINE))


def normalise(text: str) -> str:
    # Abstract text from the tokenised sources carries a space before punctuation ("CCM2 , and");
    # a quote that drops it is still verbatim, so both sides lose it.
    text = re.sub(r"\s+", " ", str(text))
    return re.sub(r" (?=[,;:.)\]])", "", text).strip().strip('"').strip()


def is_verbatim(quote: str, text: str) -> bool:
    """A quote is verbatim if it is a substring, or if the pieces around an elision mark
    ("..." or "…") are each substrings of the text, in order. Paraphrase fails."""
    pieces = [p.strip(" ,;") for p in re.split(r"\s*(?:\.\.\.|…)\s*", quote.lower()) if p.strip(" ,;")]
    position = 0
    for piece in pieces:
        found = text.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return bool(pieces)


def resolve_literature_ref(claim: dict, ref: str, pmid: str, quote: str):
    """(index, slipped) of the literature candidate a support means, or (None, False)."""
    match = re.fullmatch(r"L(\d+)", ref)
    index = int(match.group(1)) - 1 if match else None
    def fits(i):
        entry = claim["literature"][i]
        texts = [normalise(entry["sentence"]).lower(), normalise(entry.get("title", "")).lower()]
        return entry["pmid"] == pmid and any(is_verbatim(normalise(quote), t) for t in texts)
    if index is not None and index < len(claim["literature"]) and fits(index):
        return index, False
    others = [i for i in range(len(claim["literature"])) if fits(i)]
    if len({claim["literature"][i]["pmid"] for i in others}) != 1:
        return None, False  # no match, or two different papers: ambiguous, so it fails
    # Several sentences of the same paper can fit (they share a title): prefer the sentence match.
    by_sentence = [i for i in others if is_verbatim(normalise(quote).lower(), normalise(claim["literature"][i]["sentence"]).lower())]
    return (by_sentence or others)[0], True


def validate_program(directory: Path, coverage: Counter, warnings: list) -> list:
    pid = directory.name.split("_p")[-1]
    candidates = json.loads((directory / "candidates.json").read_text())
    answer_path = directory / "answer.json"
    if not answer_path.exists():
        return [f"P{pid}: no answer.json"]
    try:
        answer = load(answer_path)
    except json.JSONDecodeError as exc:
        return [f"P{pid}: not valid JSON ({exc})"]

    problems = []
    offered = {c["claim_id"]: c for c in candidates["claims"]}
    answered = Counter(str(a.get("claim_id")) for a in answer.get("claims", []))
    missing = sorted(set(offered) - set(answered))
    if missing:
        problems.append(f"P{pid}: claims not answered: {', '.join(missing)}")
    doubled = sorted(k for k, n in answered.items() if n > 1)
    if doubled:
        problems.append(f"P{pid}: claims answered twice: {', '.join(doubled)}")

    for entry in answer.get("claims", []):
        claim_id = str(entry.get("claim_id"))
        claim = offered.get(claim_id)
        if not claim:
            problems.append(f"P{pid}: answered unknown claim {claim_id}")
            continue
        supports = entry.get("supports") or []
        if not supports:
            coverage["none"] += 1
            if not str(entry.get("none_reason", "")).strip():
                problems.append(f"P{pid} {claim_id}: no support and no none_reason")
            continue
        has_pmid = has_direct_pmid = False
        for support in supports:
            ref = str(support.get("ref", ""))
            pmids = [re.sub(r"^\s*PMID[:\s]*", "", p, flags=re.I).strip()
                     for p in re.split(r"[,;\s]+(?=(?:PMID[:\s]*)?\d)", str(support.get("pmid") or "")) if p.strip()]
            pmid = pmids[0] if pmids else ""
            match = re.fullmatch(r"([LD])(\d+)", ref)
            if not match:
                problems.append(f"P{pid} {claim_id}: bad ref {ref!r}")
                continue
            index = int(match.group(2)) - 1
            if match.group(1) == "L":
                resolved, slipped = resolve_literature_ref(claim, ref, pmid, support.get("quote", ""))
                if index >= len(claim["literature"]) and resolved is None:
                    problems.append(f"P{pid} {claim_id}: {ref} was not offered")
                    continue
                target = claim["literature"][resolved if resolved is not None else index]
                if support.get("role") == "discovery":
                    coverage["_discovery"] += 1
                    if target.get("is_review"):
                        problems.append(f"P{pid} {claim_id}: discovery support {ref} is a review")
                    # Plausibility: a "discovery" much newer than a heavily cited primary candidate
                    # that was on offer is probably a restatement. Reported, not failed.
                    year = int(target["year"]) if str(target.get("year", "")).isdigit() else None
                    older = [e for e in claim["literature"] if str(e.get("year", "")).isdigit() and year
                             and int(e["year"]) <= year - 10 and not e.get("is_review")
                             and (e.get("cited_by") or 0) >= 3 * max(target.get("cited_by") or 1, 1)]
                    if older:
                        best = max(older, key=lambda e: e.get("cited_by") or 0)
                        warnings.append(f"P{pid} {claim_id}: discovery {target['pmid']} ({year}) may be a restatement — "
                                        f"primary {best['pmid']} ({best['year']}, {best.get('cited_by')} citations) was offered")
                if slipped:
                    warnings.append(f"P{pid} {claim_id}: ref {ref} resolved to L{resolved + 1} (PMID and quote match it exactly)")
                    has_pmid = True
                    has_direct_pmid |= support.get("strength") == "direct"
                    continue
                offered_entry = claim["literature"][index]
                if pmid != offered_entry["pmid"]:
                    problems.append(f"P{pid} {claim_id}: {ref} is PMID {offered_entry['pmid']}, answer says {pmid!r}")
                quote = normalise(support.get("quote", ""))
                offered_text = [normalise(offered_entry["sentence"]).lower(), normalise(offered_entry.get("title", "")).lower()]
                if not quote or not any(is_verbatim(quote, text) for text in offered_text):
                    problems.append(f"P{pid} {claim_id}: quote for {ref} is not a verbatim part of its sentence or title")
                has_pmid = True
                has_direct_pmid |= support.get("strength") == "direct"
            else:
                if support.get("role") == "discovery":
                    problems.append(f"P{pid} {claim_id}: discovery support {ref} is a database entry, not a paper")
                if index >= len(claim["database"]):
                    problems.append(f"P{pid} {claim_id}: {ref} was not offered")
                    continue
                entry = claim["database"][index]
                allowed = {p["pmid"] for p in entry["pmids"]}
                allowed |= set(re.findall(r"(?:PubMed|PMID)[:\s]*(\d{6,9})", str(entry.get("term", ""))))
                for one in pmids:
                    if one not in allowed:
                        problems.append(f"P{pid} {claim_id}: PMID {one} is not cited by {ref} (GO annotation or entry text)")
                if pmid:
                    has_pmid = True
                    has_direct_pmid |= support.get("strength") == "direct"
        if has_direct_pmid:
            coverage["direct PMID"] += 1
        elif has_pmid:
            coverage["indirect PMID only"] += 1
        else:
            coverage["database term, no PMID"] += 1
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--write-problems", action="store_true",
                        help="record each failing item's problems in problems.json for repair_rejected_answers.sh")
    args = parser.parse_args()

    coverage, problems, warnings = Counter(), [], []
    directories = sorted(args.dispatch.glob(f"{args.arm}_p*"), key=lambda d: int(d.name.split("_p")[-1]))
    for directory in directories:
        found = validate_program(directory, coverage, warnings)
        if args.write_problems:
            # The "P<id>" prefix is dropped: the repair call sees one item only.
            record_item_problems(directory, "validate_citation_answers", [re.sub(r"^P\d+:? ", "", p) for p in found])
        problems += found
        if not found:
            print(f"{directory.name}: PASS")
    total = sum(n for k, n in coverage.items() if not k.startswith("_"))
    print(f"\ncoverage over {total} claims:")
    if coverage["_discovery"]:
        print(f"  discovery-role supports  {coverage['_discovery']:>4}  (papers marked as the original study)")
    for key in ("direct PMID", "indirect PMID only", "database term, no PMID", "none"):
        print(f"  {key:<24} {coverage[key]:>4}  ({coverage[key] / total:.0%})" if total else f"  {key}: 0")
    for warning in warnings:
        print(f"WARN {warning}")
    for problem in problems:
        print(f"FAIL {problem}")
    print(f"\n{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
