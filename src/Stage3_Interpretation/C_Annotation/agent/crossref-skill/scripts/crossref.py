#!/usr/bin/env python3
"""Crossref bibliographic search or exact DOI lookup: DOI, title, authors, year, journal, type, abstract (sparse) and publisher PDF links.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, append_jsonl, require_str, require_str_list, run  # noqa: E402,F401

import html
import os
import re
import time
from urllib.parse import quote

import requests


# Crossref API, ported from gene-program-interpreter research/literature.py (resolve_doi)
CROSSREF_BASE_URL = "https://api.crossref.org"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-annotation/0.1 (literature search)"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def normalize_doi(value):
    """Return a bare lowercase DOI, or None if value is not a DOI."""
    if not value:
        return None
    doi = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    doi = doi.rstrip(".,;)")
    return doi if _DOI_RE.fullmatch(doi) else None


def _clean(text):
    """Strip JATS / HTML tags and collapse whitespace."""
    if not text:
        return None
    return " ".join(html.unescape(_TAG_RE.sub(" ", text)).split()) or None


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


def _get(url, params=None, timeout=30, retries=1):
    """GET JSON with one retry on transient HTTP status / network errors."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
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
    raise RuntimeError(f"Crossref request failed: {last}")


def _crossref_record(item):
    if not item:
        return None
    titles = item.get("title") or []
    title = titles[0] if isinstance(titles, list) and titles else str(titles or "")
    date_parts = (
        (item.get("published") or {}).get("date-parts")
        or (item.get("issued") or {}).get("date-parts")
        or []
    )
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    container = item.get("container-title") or []
    relation = item.get("relation") or {}
    authors = [
        a.get("family") or a.get("name") or "" for a in (item.get("author") or [])
    ]
    authors = [a for a in authors if a]
    return {
        "doi": normalize_doi(item.get("DOI")),
        "title": _clean(title) or "No Title",
        "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        if authors
        else None,
        "year": year,
        "journal": container[0] if container else None,
        "study_type": item.get("type"),
        "abstract": _clean(item.get("abstract")),
        "is_preprint": str(item.get("subtype") or "").casefold() == "preprint"
        or item.get("type") == "posted-content",
        "is_retracted": "is-retracted-by" in relation,
        # publisher-deposited PDF links, tried first by download_paper
        "pdf_urls": [
            link.get("URL")
            for link in (item.get("link") or [])
            if link.get("URL") and "pdf" in str(link.get("content-type", "")).lower()
        ],
    }


SKILL = "crossref-skill"


def fetch_records(query, max_results):
    params = {}
    email = os.getenv("CROSSREF_MAILTO") or os.getenv("PUBMED_EMAIL")
    if email:
        params["mailto"] = email
    doi = normalize_doi(query)
    if doi:
        payload = _get(
            f"{CROSSREF_BASE_URL}/works/{quote(doi, safe='')}", params=params
        )
        items = [(payload or {}).get("message", {})]
    else:
        payload = _get(
            f"{CROSSREF_BASE_URL}/works",
            params={
                "query.bibliographic": _bounded_query(query),
                "rows": max_results,
                **params,
            },
        )
        items = (payload or {}).get("message", {}).get("items", [])
    return [r for r in (_crossref_record(item) for item in items) if r][:max_results]


# shared by the literature skills (identical in each)
def record_key(r):
    return (
        ("pmid", r["pmid"])
        if r.get("pmid")
        else ("doi", r["doi"])
        if r.get("doi")
        else ("title", r["title"].casefold())
    )


def handle(req):
    query = require_str(req, "query")
    max_results = max(1, min(int(req.get("max_results", 5)), MAX_SEARCH_RESULTS))
    records = fetch_records(query, max_results)
    for r in records:
        r.setdefault(
            "url",
            f"https://doi.org/{r['doi']}"
            if r.get("doi")
            else (
                f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}"
                if r.get("pmid")
                else None
            ),
        )
    out = {"source": SKILL, "query": query, "n_results": len(records)}
    if req.get("write_to"):
        genes = require_str_list(req, "genes")
        rows = [
            {
                "source": SKILL,
                "query": query,
                "genes": genes,
                "status": "candidate",
                "finding": None,
                "stance": None,
                "pdf_path": None,
                **r,
            }
            for r in records
        ]
        added, dup = append_jsonl(req["write_to"], rows, dedupe=record_key)
        out.update(
            {"written_to": req["write_to"], "added": added, "already_present": dup}
        )
    n = int(req.get("abstract_chars", 600))
    out["records"] = [
        {
            **{k: v for k, v in r.items() if k != "abstract"},
            "abstract": (r.get("abstract") or "")[:n] or None,
        }
        for r in records
    ]
    return out


if __name__ == "__main__":
    sys.exit(run(handle))
