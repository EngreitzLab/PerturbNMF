import html
import os
import re
import time
from urllib.parse import quote

import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# Crossref API, ported from gene-program-interpreter research/literature.py (resolve_doi)
CROSSREF_BASE_URL = "https://api.crossref.org"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-AGeneTic/0.1 (literature search)"
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
            doi = doi[len(prefix):]
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
    raise RuntimeError(f"Crossref request failed: {last}")


def _crossref_record(item):
    if not item:
        return None
    titles = item.get("title") or []
    title = titles[0] if isinstance(titles, list) and titles else str(titles or "")
    date_parts = ((item.get("published") or {}).get("date-parts")
                  or (item.get("issued") or {}).get("date-parts") or [])
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    container = item.get("container-title") or []
    relation = item.get("relation") or {}
    authors = [a.get("family") or a.get("name") or "" for a in (item.get("author") or [])]
    authors = [a for a in authors if a]
    return {
        "doi": normalize_doi(item.get("DOI")),
        "title": _clean(title) or "No Title",
        "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else None,
        "year": year,
        "journal": container[0] if container else None,
        "study_type": item.get("type"),
        "abstract": _clean(item.get("abstract")),
        "is_preprint": str(item.get("subtype") or "").casefold() == "preprint" or item.get("type") == "posted-content",
        "is_retracted": "is-retracted-by" in relation,
        # publisher-deposited PDF links, tried first by download_paper
        "pdf_urls": [link.get("URL") for link in (item.get("link") or [])
                     if link.get("URL") and "pdf" in str(link.get("content-type", "")).lower()],
    }


def search_Crossref(
    query: str = "",
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> str:
    """Search Crossref for works related to the query, or resolve a single DOI

    When searching, you should consider:

    1. If the query is a DOI (e.g. 10.1038/nature12373 or https://doi.org/...), the DOI is
       resolved directly and exactly one record is returned (useful to verify a citation)
    2. Otherwise the query is matched bibliographically (title, authors, journal, year),
       so a citation string like "Engreitz 2016 Nature enhancer" works well
    3. Crossref abstracts are sparse; many records return metadata only

    Env (optional): CROSSREF_MAILTO / PUBMED_EMAIL (Crossref polite pool). No API key needed.

    Args:
        query (str): The query (or DOI) to search for
        max_results (int): The maximum number of results to return, default is 3 (capped at 15)
        mini_handler: The LLM handler for summarizing abstracts
        out_dir (str): Directory to download open-access PDFs into; None skips downloading
    """
    try:
        query = _bounded_query(query)
        max_results = _bounded_limit(max_results, MAX_SEARCH_RESULTS)
        params = {}
        email = os.getenv("CROSSREF_MAILTO") or os.getenv("PUBMED_EMAIL")
        if email:
            params["mailto"] = email

        doi = normalize_doi(query)
        if doi:
            payload = _get(f"{CROSSREF_BASE_URL}/works/{quote(doi, safe='')}", params=params)
            items = [(payload or {}).get("message", {})]
        else:
            payload = _get(
                f"{CROSSREF_BASE_URL}/works",
                params={"query.bibliographic": query, "rows": max_results, **params},
            )
            items = (payload or {}).get("message", {}).get("items", [])
        records = [r for r in (_crossref_record(item) for item in items) if r][:max_results]
    except Exception as e:
        return f"Error during Crossref search: {e}"

    if not records:
        return "No results found on Crossref."

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for rec in records:
        url = f"https://doi.org/{rec['doi']}" if rec["doi"] else "No URL"
        tags = "".join(t for t, flag in ((" [RETRACTED]", rec["is_retracted"]), (" [PREPRINT]", rec["is_preprint"])) if flag)
        info = f"Authors: {rec['authors']} | Year: {rec['year']} | Journal: {rec['journal']} | Type: {rec['study_type']}"

        content = "No abstract available."
        if rec["abstract"]:
            content = Summarize_Agent(rec["abstract"], mini_handler) if mini_handler else rec["abstract"]

        pdf_line = ''
        if out_dir:
            pdf_path, msg = download_paper(out_dir, rec["doi"] or rec["title"], doi=rec["doi"], pdf_urls=rec["pdf_urls"])
            downloads.append((rec["title"], pdf_path, msg))
            pdf_line = '\n' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + rec["title"] + tags + '\n' + 'Url: ' + url + '\n' + info + pdf_line + '\n' + 'Summary: ' + content)

    if out_dir:
        report_downloads("Crossref", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: Crossref"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_Crossref on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_Crossref(query=q, max_results=args.max_results, out_dir=args.out_dir))
