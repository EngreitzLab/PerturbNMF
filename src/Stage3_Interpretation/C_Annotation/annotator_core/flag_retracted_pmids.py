"""List every retracted or non-resolving PMID in the literature context, so the prompt builder
can drop it from the reference pool.

The pool is scraped (PubTator), not curated, and retracted papers do reach it and do get cited.
Filtering at build time means an annotator is never offered one; verify_cited_pmids.py still
re-checks the citations afterwards.

The citation pass retrieves far more papers than the annotation pool, so its candidates are
screened too (--candidates-dir, repeatable): every literature PMID and GO-annotation PMID.

Usage:
    python flag_retracted_pmids.py --ncbi-context ncbi_context.json --output excluded_pool_pmids.json
    python flag_retracted_pmids.py --candidates-dir citation_candidates --output excluded_candidate_pmids.json
"""
import argparse
import json
import re
from pathlib import Path

from verify_cited_pmids import fetch_pubmed_summaries, is_retracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncbi-context", type=Path)
    parser.add_argument("--candidates-dir", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pool = set()
    if args.ncbi_context:
        for program in json.loads(args.ncbi_context.read_text()).values():
            for snippets in (program.get("evidence_snippets") or {}).values():
                for snippet in snippets:
                    pool.update(re.findall(r"\(PMID:(\d+)\)", snippet))
    for directory in args.candidates_dir:
        for path in directory.glob("program_*.json"):
            for claim in json.loads(path.read_text())["claims"]:
                pool.update(entry["pmid"] for entry in claim["literature"])
                pool.update(p["pmid"] for entry in claim["database"] for p in entry["pmids"])

    records = fetch_pubmed_summaries(pool)
    retracted = sorted(p for p in pool if records.get(p) and not records[p].get("error") and is_retracted(records[p]))
    unresolved = sorted(p for p in pool if not records.get(p) or records[p].get("error"))

    args.output.write_text(
        json.dumps({"retracted": retracted, "unresolved": unresolved}, indent=2), encoding="utf-8"
    )
    print(f"{len(pool)} pool PMIDs checked: {len(retracted)} retracted, {len(unresolved)} unresolved")
    for pmid in retracted:
        print(f"  RETRACTED {pmid}  {records[pmid].get('title', '')[:80]}")
    for pmid in unresolved:
        print(f"  UNRESOLVED {pmid}")
    print(f"wrote -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
