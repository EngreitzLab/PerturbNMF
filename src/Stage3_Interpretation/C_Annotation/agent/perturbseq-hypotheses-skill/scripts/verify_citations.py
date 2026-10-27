#!/usr/bin/env python3
"""Verify every DOI, PMID and NCT number in a report against its source of record.

Crossref for DOIs, NCBI E-utilities for PMIDs, ClinicalTrials.gov v2 for NCTs. Prints a
table and exits non-zero if anything fails, so it can gate a render.

Why this exists
---------------
In one audit of a draft, 7 of 16 PMIDs were defective. In the report that ran this check
before shipping, 72 of 72 passed. The proposer should query the literature directly --
verification downstream is a backstop, not the primary mechanism.

What it cannot do
-----------------
It checks that an identifier resolves and that the surrounding text is consistent with the
record's title / first author / year. It CANNOT check that the cited paper supports the
claim. Spot-read the abstracts of the load-bearing citations yourself.

Usage
-----
    python verify_citations.py report.qmd
    python verify_citations.py report.qmd --json outputs/citation_check.json
    python verify_citations.py report.qmd --email you@stanford.edu   # NCBI etiquette
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]*[A-Za-z0-9]")
PMID_RE = re.compile(r"\bPMID:?\s*(\d{6,9})\b|pubmed\.ncbi\.nlm\.nih\.gov/(\d{6,9})")
NCT_RE = re.compile(r"\bNCT\d{8}\b")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

TIMEOUT = 20
PAUSE = 0.34  # NCBI: <= 3 req/s without a key


def context_for(text: str, span: tuple[int, int], width: int = 200) -> str:
    a = max(0, span[0] - width)
    b = min(len(text), span[1] + width)
    return " ".join(text[a:b].split())


def check_doi(doi: str) -> dict:
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         headers={"User-Agent": "EngreitzLab-citation-check/1.0"},
                         timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"ok": False, "error": f"network: {e}"}
    if r.status_code == 404:
        return {"ok": False, "error": "not found in Crossref"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    m = r.json().get("message", {})
    authors = m.get("author") or []
    parts = (m.get("issued", {}).get("date-parts") or [[None]])[0]
    return {"ok": True,
            "title": (m.get("title") or [""])[0],
            "first_author": authors[0].get("family") if authors else None,
            "journal": (m.get("container-title") or [""])[0],
            "year": parts[0] if parts else None}


def check_pmid(pmid: str, email: str | None) -> dict:
    params = {"db": "pubmed", "id": pmid, "retmode": "json"}
    if email:
        params["email"] = email
        params["tool"] = "EngreitzLab-citation-check"
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                         params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"ok": False, "error": f"network: {e}"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    rec = (r.json().get("result") or {}).get(pmid)
    if not rec or "error" in rec or not rec.get("title"):
        return {"ok": False, "error": "no such PMID"}
    authors = rec.get("authors") or []
    pubdate = rec.get("pubdate", "")
    ym = YEAR_RE.search(pubdate)
    return {"ok": True,
            "title": rec.get("title", "").rstrip("."),
            "first_author": (authors[0].get("name", "").split()[0] if authors else None),
            "journal": rec.get("source"),
            "year": int(ym.group(0)) if ym else None}


def check_nct(nct: str) -> dict:
    try:
        r = requests.get(f"https://clinicaltrials.gov/api/v2/studies/{nct}",
                         params={"fields": "protocolSection"}, timeout=TIMEOUT)
    except requests.RequestException as e:
        return {"ok": False, "error": f"network: {e}"}
    if r.status_code == 404:
        return {"ok": False, "error": "no such NCT"}
    if r.status_code != 200:
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    p = r.json().get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    return {"ok": True,
            "title": ident.get("briefTitle"),
            "status": status.get("overallStatus"),
            "phase": ", ".join(design.get("phases") or []) or None,
            "year": (status.get("startDateStruct", {}).get("date") or "")[:4] or None}


CITED_AUTHOR_RE = re.compile(r"\b([A-Z][a-zA-ZÀ-ɏ'-]{2,})\s+(?:et al\.?|and\s+[A-Z])")


def consistent(rec: dict, ctx: str, kind: str = "doi") -> tuple[list[str], list[str]]:
    """Cheap consistency checks between the record and the surrounding prose.

    Returns (warnings, notes). A warning means the prose and the record DISAGREE. A note
    means the prose gives nothing to check against -- a bare `PMID 12345` with no author
    or year. Bare identifiers are a style finding, not a correctness one, so they are kept
    out of the failure count and reported separately.
    """
    warn: list[str] = []
    notes: list[str] = []

    fa = rec.get("first_author")
    cited = [m.group(1) for m in CITED_AUTHOR_RE.finditer(ctx)]
    if fa and len(fa) > 3:
        if not cited:
            notes.append("bare identifier: no author named in the surrounding text")
        elif not any(fa.lower() == c.lower() for c in cited):
            warn.append(f"record first author {fa!r} vs cited {sorted(set(cited))}")

    # Trials are cited by their NCT number, not by year -- a start-date mismatch with the
    # surrounding prose is expected and would be pure noise.
    yr = rec.get("year")
    if yr and kind != "nct":
        years = {int(m.group(0)) for m in YEAR_RE.finditer(ctx)}
        if not years:
            notes.append("bare identifier: no year in the surrounding text")
        elif int(yr) not in years:
            warn.append(f"record year {yr} not among years in context {sorted(years)}")
    return warn, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, nargs="+", help=".qmd / .md / .html to scan")
    ap.add_argument("--email", default=None, help="contact address for NCBI E-utilities")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    text = "\n".join(p.read_text(errors="replace") for p in args.report)

    found: dict[str, dict] = {}
    for m in DOI_RE.finditer(text):
        found.setdefault(f"doi:{m.group(0).rstrip('.),;')}",
                         {"kind": "doi", "id": m.group(0).rstrip(".),;"),
                          "context": context_for(text, m.span())})
    for m in PMID_RE.finditer(text):
        pmid = m.group(1) or m.group(2)
        found.setdefault(f"pmid:{pmid}",
                         {"kind": "pmid", "id": pmid, "context": context_for(text, m.span())})
    for m in NCT_RE.finditer(text):
        found.setdefault(f"nct:{m.group(0)}",
                         {"kind": "nct", "id": m.group(0), "context": context_for(text, m.span())})

    if not found:
        print("No DOIs, PMIDs or NCT numbers found. If the report makes literature claims, "
              "that is itself a finding: citations must be inline and hyperlinked at the claim.")
        sys.exit(0)

    print(f"Checking {len(found)} identifiers from {', '.join(p.name for p in args.report)}\n")

    results, failures, warnings, bare = [], [], [], []
    for key, item in sorted(found.items()):
        kind, ident = item["kind"], item["id"]
        if kind == "doi":
            rec = check_doi(ident)
        elif kind == "pmid":
            rec = check_pmid(ident, args.email)
            time.sleep(PAUSE)
        else:
            rec = check_nct(ident)

        if not rec.get("ok"):
            failures.append((kind, ident, rec.get("error")))
            print(f"  FAIL  {kind.upper():5s} {ident:32s} {rec.get('error')}")
        else:
            warn, notes = consistent(rec, item["context"], kind)
            rec["warnings"], rec["notes"] = warn, notes
            title = (rec.get("title") or "")[:70]
            if warn:
                warnings.append((kind, ident, warn))
                print(f"  WARN  {kind.upper():5s} {ident:32s} {title}")
                for w in warn:
                    print(f"        -> {w}")
            else:
                if notes:
                    bare.append((kind, ident))
                print(f"  ok    {kind.upper():5s} {ident:32s} {title}")
        results.append({**item, **rec})

    print(f"\n{len(results) - len(failures)} of {len(results)} resolved; "
          f"{len(failures)} failed, {len(warnings)} disagree with the surrounding text.")
    if bare:
        print(f"{len(bare)} are bare identifiers with no author or year in the prose "
              f"({', '.join(i for _, i in bare[:8])}"
              f"{'...' if len(bare) > 8 else ''}) — nothing to cross-check them against. "
              f"The house style is an inline hyperlinked citation carrying author, journal "
              f"and year at the claim.")
    print("\nThis check cannot tell you whether a paper supports the claim it is attached to. "
          "Spot-read the abstracts of the load-bearing citations.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"-> {args.json}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
