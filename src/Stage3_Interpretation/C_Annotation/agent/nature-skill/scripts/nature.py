#!/usr/bin/env python3
"""Nature-portfolio article search (Crossref, DOI prefix 10.1038): DOI, title, authors, year, journal, abstract (from the article page when Crossref has none) and PDF link.

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

import requests


# Nature portfolio search (Crossref, DOI prefix 10.1038) + PDF download from nature.com.
#
# PDFs come from nature.com under an institutional subscription, which Nature grants by IP
# address: it works from a subscribing institution's network (cluster / campus / VPN) with no
# login. Off that network, only open-access articles download. Institutional licenses cover
# research use, not bulk harvesting, so results are capped and PDF requests are spaced out.
CROSSREF_BASE_URL = "https://api.crossref.org"
NATURE_BASE_URL = "https://www.nature.com"
NATURE_DOI_PREFIX = "10.1038"
MAX_SEARCH_RESULTS = 15
PDF_PAUSE_SECONDS = 2.0
USER_AGENT = "PerturbNMF-annotation/0.1 (literature search)"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

_TAG_RE = re.compile(r"<[^>]+>")
_DESCRIPTION_RE = re.compile(
    r'<meta\s+name="(?:dc\.description|description)"\s+content="([^"]*)"', re.IGNORECASE
)


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


def _landing_page_abstract(article_url, timeout=30):
    """Nature rarely deposits abstracts in Crossref; read the article page's description meta."""
    try:
        resp = requests.get(
            article_url, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        m = _DESCRIPTION_RE.search(resp.text) if resp.status_code == 200 else None
        return _clean(m.group(1)) if m else None
    except requests.RequestException:
        return None


def _nature_record(item):
    doi = str(item.get("DOI") or "").lower()
    if not doi.startswith(NATURE_DOI_PREFIX + "/"):
        return None
    article_id = doi.split("/", 1)[1]  # 10.1038/s41586-020-2012-7 -> s41586-020-2012-7
    titles = item.get("title") or []
    date_parts = (
        (item.get("published") or {}).get("date-parts")
        or (item.get("issued") or {}).get("date-parts")
        or []
    )
    container = item.get("container-title") or []
    authors = [
        a.get("family") or a.get("name") or "" for a in (item.get("author") or [])
    ]
    authors = [a for a in authors if a]
    return {
        "doi": doi,
        "article_id": article_id,
        "url": f"{NATURE_BASE_URL}/articles/{article_id}",
        "pdf_url": f"{NATURE_BASE_URL}/articles/{article_id}.pdf",
        "title": _clean(titles[0] if titles else "") or "No Title",
        "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        if authors
        else None,
        "year": date_parts[0][0] if date_parts and date_parts[0] else None,
        "journal": container[0] if container else None,
        "study_type": item.get("type"),
        "abstract": _clean(item.get("abstract")),
        "is_retracted": "is-retracted-by" in (item.get("relation") or {}),
    }


SKILL = "nature-skill"


def fetch_records(query, max_results):
    params = {
        "query.bibliographic": _bounded_query(query),
        "rows": max_results,
        "filter": f"prefix:{NATURE_DOI_PREFIX},type:journal-article",
    }
    email = os.getenv("CROSSREF_MAILTO") or os.getenv("PUBMED_EMAIL")
    if email:
        params["mailto"] = email
    payload = _get(f"{CROSSREF_BASE_URL}/works", params=params)
    items = (payload or {}).get("message", {}).get("items", [])
    records = [r for r in (_nature_record(item) for item in items) if r][:max_results]
    for r in records:
        r["abstract"] = r["abstract"] or _landing_page_abstract(r["url"])
        r["pdf_urls"] = [r.pop("pdf_url")]
    return records


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
