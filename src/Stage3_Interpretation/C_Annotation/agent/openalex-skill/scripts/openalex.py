#!/usr/bin/env python3
"""OpenAlex full-text relevance search: DOI, PMID, title, year, venue, type, abstract, citation count and open-access PDF links.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, append_jsonl, require_str, require_str_list, run  # noqa: E402,F401

import os
import re
import time

import requests


# OpenAlex API, ported from gene-program-interpreter research/literature.py
OPENALEX_BASE_URL = "https://api.openalex.org"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-annotation/0.1 (literature search)"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{1,9}$")


def normalize_doi(value):
    """Return a bare lowercase DOI, or None if value is not a DOI."""
    if not value:
        return None
    doi = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    doi = doi.rstrip(".,;)")
    return doi if _DOI_RE.fullmatch(doi) else None


def normalize_pmid(value):
    if value is None:
        return None
    pmid = str(value).strip()
    return pmid if _PMID_RE.fullmatch(pmid) else None


def _bounded_query(value):
    q = " ".join(str(value or "").split())
    if not q:
        raise ValueError("query must not be empty")
    return q[:512]


def _bounded_limit(value, maximum):
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = maximum
    return max(1, min(n, maximum))


def _openalex_params():
    key = os.getenv("OPENALEX_API_KEY")
    if not key:
        raise RuntimeError("OPENALEX_API_KEY is not set; use search_NCBI / search_Crossref instead")
    params = {"api_key": key}
    email = os.getenv("OPENALEX_EMAIL") or os.getenv("OPENALEX_MAILTO")
    if email:
        params["mailto"] = email
    return params


def _get(url, params=None, timeout=30, retries=1):
    """GET JSON with one retry on transient HTTP status / network errors."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code in TRANSIENT_STATUS and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            if attempt >= retries:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"OpenAlex request failed: {last}")


def _rebuild_abstract(inverted_index):
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inverted_index:
        return None
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions)) or None


def _openalex_record(item):
    ids = item.get("ids") or {}
    pmid_raw = ids.get("pmid")
    pmid = normalize_pmid(str(pmid_raw).rstrip("/").rsplit("/", 1)[-1]) if pmid_raw else None
    source = (item.get("primary_location") or {}).get("source") or {}
    return {
        "openalex_id": item.get("id"),
        "pmid": pmid,
        "doi": normalize_doi(ids.get("doi") or item.get("doi")),
        "title": item.get("display_name") or item.get("title") or "No Title",
        "year": item.get("publication_year"),
        "journal": source.get("display_name"),
        "study_type": item.get("type"),
        "abstract": _rebuild_abstract(item.get("abstract_inverted_index")),
        "is_preprint": item.get("type") == "preprint",
        "is_retracted": bool(item.get("is_retracted")),
        "cited_by_count": int(item.get("cited_by_count") or 0),
        # open-access PDF locations, tried first by download_paper
        "pdf_urls": list(dict.fromkeys(
            loc.get("pdf_url") for loc in [item.get("best_oa_location") or {}] + (item.get("locations") or [])
            if loc and loc.get("pdf_url"))),
    }

SKILL = "openalex-skill"


def fetch_records(query, max_results):
    payload = _get(f"{OPENALEX_BASE_URL}/works",
                   params={"search": _bounded_query(query), "per-page": max_results, **_openalex_params()})
    return [_openalex_record(item) for item in payload.get("results", [])][:max_results]


# shared by the literature skills (identical in each)
def record_key(r):
    return ("pmid", r["pmid"]) if r.get("pmid") else ("doi", r["doi"]) if r.get("doi") else ("title", r["title"].casefold())


def handle(req):
    query = require_str(req, "query")
    max_results = max(1, min(int(req.get("max_results", 5)), MAX_SEARCH_RESULTS))
    records = fetch_records(query, max_results)
    for r in records:
        r.setdefault("url", f"https://doi.org/{r['doi']}" if r.get("doi") else
                     (f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}" if r.get("pmid") else None))
    out = {"source": SKILL, "query": query, "n_results": len(records)}
    if req.get("write_to"):
        genes = require_str_list(req, "genes")
        rows = [{"source": SKILL, "query": query, "genes": genes, "status": "candidate", "finding": None,
                 "stance": None, "pdf_path": None, **r} for r in records]
        added, dup = append_jsonl(req["write_to"], rows, dedupe=record_key)
        out.update({"written_to": req["write_to"], "added": added, "already_present": dup})
    n = int(req.get("abstract_chars", 600))
    out["records"] = [{**{k: v for k, v in r.items() if k != "abstract"},
                       "abstract": (r.get("abstract") or "")[:n] or None} for r in records]
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
