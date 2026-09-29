import html
import os
import re
import time

import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# Nature portfolio search (Crossref, DOI prefix 10.1038) + PDF download from nature.com.
#
# PDFs come from nature.com under Stanford's institutional subscription, which Nature grants
# by IP address: it works from Sherlock / campus / Stanford VPN (Sherlock egresses from a
# Stanford 171.67.x address) with no SUNet login. Off the Stanford network, only open-access
# articles download. Stanford's license covers research use, not bulk harvesting, so results
# are capped and PDF requests are spaced out.
CROSSREF_BASE_URL = "https://api.crossref.org"
NATURE_BASE_URL = "https://www.nature.com"
NATURE_DOI_PREFIX = "10.1038"
MAX_SEARCH_RESULTS = 15
PDF_PAUSE_SECONDS = 2.0
USER_AGENT = "PerturbNMF-AGeneTic/0.1 (literature search)"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

_TAG_RE = re.compile(r"<[^>]+>")
_DESCRIPTION_RE = re.compile(r'<meta\s+name="(?:dc\.description|description)"\s+content="([^"]*)"', re.IGNORECASE)


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


def _landing_page_abstract(article_url, timeout=30):
    """Nature rarely deposits abstracts in Crossref; read the article page's description meta."""
    try:
        resp = requests.get(article_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        m = _DESCRIPTION_RE.search(resp.text) if resp.status_code == 200 else None
        return _clean(m.group(1)) if m else None
    except requests.RequestException:
        return None


def _nature_record(item):
    doi = str(item.get("DOI") or "").lower()
    if not doi.startswith(NATURE_DOI_PREFIX + "/"):
        return None
    article_id = doi.split("/", 1)[1]           # 10.1038/s41586-020-2012-7 -> s41586-020-2012-7
    titles = item.get("title") or []
    date_parts = ((item.get("published") or {}).get("date-parts")
                  or (item.get("issued") or {}).get("date-parts") or [])
    container = item.get("container-title") or []
    authors = [a.get("family") or a.get("name") or "" for a in (item.get("author") or [])]
    authors = [a for a in authors if a]
    return {
        "doi": doi,
        "article_id": article_id,
        "url": f"{NATURE_BASE_URL}/articles/{article_id}",
        "pdf_url": f"{NATURE_BASE_URL}/articles/{article_id}.pdf",
        "title": _clean(titles[0] if titles else "") or "No Title",
        "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else None,
        "year": date_parts[0][0] if date_parts and date_parts[0] else None,
        "journal": container[0] if container else None,
        "study_type": item.get("type"),
        "abstract": _clean(item.get("abstract")),
        "is_retracted": "is-retracted-by" in (item.get("relation") or {}),
    }


def search_Nature(
    query: str = "",
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> str:
    """Search Nature portfolio journals (Nature, Nature Genetics, Nature Communications, ...)
    for articles related to the query and optionally download their PDFs

    When searching, you should consider:

    1. Plain keywords work best (e.g. "GATA4 cardiomyocyte differentiation"); matching is
       bibliographic (title, authors, journal, year) via Crossref, restricted to DOI prefix 10.1038
    2. Adding a journal name or year to the query biases toward it (e.g. "... Nature Genetics 2023")
    3. Only journal articles are returned (no news, editorials split out by Crossref type)
    4. PDF download uses Stanford's subscription by IP: run on Sherlock / campus / Stanford VPN

    Env (optional): CROSSREF_MAILTO / PUBMED_EMAIL (Crossref polite pool). No API key needed.

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 3 (capped at 15)
        mini_handler: The LLM handler for summarizing abstracts
        out_dir (str): Directory to download PDFs into; None skips downloading
    """
    try:
        query = _bounded_query(query)
        max_results = _bounded_limit(max_results, MAX_SEARCH_RESULTS)
        params = {"query.bibliographic": query, "rows": max_results,
                  "filter": f"prefix:{NATURE_DOI_PREFIX},type:journal-article"}
        email = os.getenv("CROSSREF_MAILTO") or os.getenv("PUBMED_EMAIL")
        if email:
            params["mailto"] = email
        payload = _get(f"{CROSSREF_BASE_URL}/works", params=params)
        items = (payload or {}).get("message", {}).get("items", [])
        records = [r for r in (_nature_record(item) for item in items) if r][:max_results]
    except Exception as e:
        return f"Error during Nature search: {e}"

    if not records:
        return "No results found on Nature."

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for i, rec in enumerate(records):
        tags = " [RETRACTED]" if rec["is_retracted"] else ""
        info = f"Authors: {rec['authors']} | Year: {rec['year']} | Journal: {rec['journal']} | DOI: {rec['doi']}"

        abstract = rec["abstract"] or _landing_page_abstract(rec["url"])
        content = "No abstract available."
        if abstract:
            content = Summarize_Agent(abstract, mini_handler) if mini_handler else abstract

        pdf_line = ''
        if out_dir:
            if i:
                time.sleep(PDF_PAUSE_SECONDS)   # polite spacing under the institutional license
            pdf_path, msg = download_paper(out_dir, f"nature_{rec['article_id']}",
                                           doi=rec["doi"], pdf_urls=[rec["pdf_url"]])
            if not pdf_path:
                msg += "; is this running on the Stanford network (Sherlock / campus / VPN)?"
            downloads.append((rec["title"], pdf_path, msg))
            pdf_line = '\n' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + rec["title"] + tags + '\n' + 'Url: ' + rec["url"] + '\n' + info + pdf_line + '\n' + 'Summary: ' + content)

    if out_dir:
        report_downloads("Nature", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: Nature (nature.com, Stanford subscription)"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_Nature on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_Nature(query=q, max_results=args.max_results, out_dir=args.out_dir))
