import os
import re
import time

import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# OpenAlex API, ported from gene-program-interpreter research/literature.py
OPENALEX_BASE_URL = "https://api.openalex.org"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-AGeneTic/0.1 (literature search)"
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


def search_OpenAlex(
    query: str = "",
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> str:
    """Search OpenAlex (cross-publisher, includes preprints) for works related to the query

    When searching, you should consider:

    1. The query is full-text relevance search over title, abstract and fulltext;
       plain keywords work best (e.g. "GATA4 cardiomyocyte differentiation")
    2. Boolean operators AND, OR, NOT (upper case) and "quoted phrases" are supported
    3. Records carry DOI / PMID and citation counts, useful for cross-checking with search_NCBI

    Env: OPENALEX_API_KEY (required), OPENALEX_EMAIL / OPENALEX_MAILTO (optional)

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 3 (capped at 15)
        mini_handler: The LLM handler for summarizing abstracts
        out_dir (str): Directory to download open-access PDFs into; None skips downloading
    """
    try:
        query = _bounded_query(query)
        max_results = _bounded_limit(max_results, MAX_SEARCH_RESULTS)
        payload = _get(
            f"{OPENALEX_BASE_URL}/works",
            params={"search": query, "per-page": max_results, **_openalex_params()},
        )
        records = [_openalex_record(item) for item in payload.get("results", [])][:max_results]
    except Exception as e:
        return f"Error during OpenAlex search: {e}"

    if not records:
        return "No results found on OpenAlex."

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for rec in records:
        url = f"https://doi.org/{rec['doi']}" if rec["doi"] else (rec["openalex_id"] or "No URL")
        tags = "".join(t for t, flag in ((" [RETRACTED]", rec["is_retracted"]), (" [PREPRINT]", rec["is_preprint"])) if flag)
        info = (f"Year: {rec['year']} | Journal: {rec['journal']} | PMID: {rec['pmid']} | "
                f"Type: {rec['study_type']} | Cited by: {rec['cited_by_count']}")

        content = "No abstract available."
        if rec["abstract"]:
            content = Summarize_Agent(rec["abstract"], mini_handler) if mini_handler else rec["abstract"]

        pdf_line = ''
        if out_dir:
            pdf_path, msg = download_paper(out_dir, rec["doi"] or rec["pmid"] or rec["openalex_id"], pmid=rec["pmid"], doi=rec["doi"], pdf_urls=rec["pdf_urls"])
            downloads.append((rec["title"], pdf_path, msg))
            pdf_line = '\n' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + rec["title"] + tags + '\n' + 'Url: ' + url + '\n' + info + pdf_line + '\n' + 'Summary: ' + content)

    if out_dir:
        report_downloads("OpenAlex", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: OpenAlex"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_OpenAlex on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_OpenAlex(query=q, max_results=args.max_results, out_dir=args.out_dir))
