"""Check every PMID cited in the answers against NCBI: does it exist, and is it retracted?

Pool membership (checked by validate_v2_answers.py) only proves the model did not invent an
identifier. This proves the identifier resolves to a real, non-retracted paper. Neither proves
the paper supports the claim -- that is entailment, and nothing here checks it.

Usage:
    python verify_cited_pmids.py --dispatch <annotation_dispatch> --arm v3                      # annotations
    python verify_cited_pmids.py --dispatch <citations_dispatch> --arm cite --answer-key claims  # citation pass
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
# A retracted paper and a retraction NOTICE are different pubtypes; neither may be cited.
# A check on the first alone lets retraction notices through.
RETRACTED = {"Retracted Publication", "Retraction of Publication"}
NON_PRIMARY = {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Comment"}
# Batched with retries: one request for every PMID in a 60-program run is large enough that the
# chunked response is sometimes cut off mid-stream (http.client.IncompleteRead).
BATCH_SIZE = 20


def fetch_pubmed_summaries(pmids: Iterable[str]) -> Dict[str, dict]:
    """esummary records keyed by PMID; a missing or errored key means the PMID does not resolve."""
    result: Dict[str, dict] = {}
    ids = sorted(set(pmids))
    for start in range(0, len(ids), BATCH_SIZE):
        query = urllib.parse.urlencode(
            {"db": "pubmed", "retmode": "json", "id": ",".join(ids[start : start + BATCH_SIZE])}
        )
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(f"{ESUMMARY}?{query}", timeout=60) as response:
                    result.update(json.load(response).get("result", {}))
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"batch {start // BATCH_SIZE + 1}: attempt {attempt} failed ({exc}); retrying")
                time.sleep(3 * attempt)
        time.sleep(0.4)
    return result


def is_retracted(record: dict) -> bool:
    return bool(set(record.get("pubtype", [])) & RETRACTED) or str(
        record.get("title", "")
    ).upper().startswith("RETRACTION")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--answer-key", default="citations", choices=["citations", "claims"],
                        help="citations: an annotation answer; claims: a citation-pass answer")
    args = parser.parse_args()

    pmids = {}
    for answer in sorted(args.dispatch.glob(f"{args.arm}_p*/answer.json")):
        payload = json.loads(re.sub(r"^```(?:json)?|```$", "", answer.read_text().strip(), flags=re.M))
        if args.answer_key == "claims":
            cited = [s for c in payload.get("claims", []) for s in (c.get("supports") or [])]
        else:
            cited = payload.get("citations", [])
        for citation in cited:
            for key in re.findall(r"\d{6,9}", str(citation.get("pmid") or "")):  # "123, 456" is two PMIDs
                pmids.setdefault(key, []).append(answer.parent.name)

    if not pmids:
        print("no PMIDs cited")
        return 0

    result = fetch_pubmed_summaries(pmids)

    problems = []
    for pmid, programs in sorted(pmids.items()):
        record = result.get(pmid)
        if not record or record.get("error"):
            problems.append(f"NONEXISTENT {pmid} (cited by {', '.join(programs)})")
            continue
        types = set(record.get("pubtype", []))
        flag = ""
        if is_retracted(record):
            problems.append(f"RETRACTED {pmid} (cited by {', '.join(programs)})")
            flag = "  <-- RETRACTED"
        elif types & NON_PRIMARY:
            flag = f"  <-- non-primary ({', '.join(sorted(types & NON_PRIMARY))})"
        print(f"{pmid}  {record.get('pubdate','?')[:4]}  {record.get('title','')[:82]}{flag}")

    print(f"\n{len(pmids)} distinct PMIDs cited; {len(problems)} hard defect(s)")
    for problem in problems:
        print("  " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
