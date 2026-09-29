import os
import re
import time
import xml.etree.ElementTree as ET

import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # llm_summary_agent / pdf_download
from llm_summary_agent import Summarize_Agent
from pdf_download import download_paper, report_downloads

# NCBI E-utilities (PubMed), ported from gene-program-interpreter research/literature.py
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-AGeneTic/0.1 (literature search)"
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_PMID_RE = re.compile(r"^\d{1,9}$")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


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


def _ncbi_params():
    params = {"tool": "PerturbNMF-AGeneTic"}
    email = os.getenv("PUBMED_EMAIL") or os.getenv("NCBI_EMAIL")
    if email:
        params["email"] = email
    key = os.getenv("NCBI_API_KEY")
    if key:
        params["api_key"] = key
    return params


def _get(url, params=None, as_json=True, timeout=30, retries=1):
    """GET with one retry on transient HTTP status / network errors."""
    last = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if resp.status_code in TRANSIENT_STATUS and attempt < retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json() if as_json else resp.text
        except (requests.Timeout, requests.ConnectionError) as e:
            last = e
            if attempt >= retries:
                break
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"NCBI request failed: {last}")


def _node_text(node):
    if node is None:
        return None
    return " ".join("".join(node.itertext()).split()) or None


def _pubmed_year(article):
    for path in (".//ArticleDate/Year", ".//Journal/JournalIssue/PubDate/Year",
                 ".//Journal/JournalIssue/PubDate/MedlineDate"):
        text = _node_text(article.find(path))
        if text:
            m = _YEAR_RE.search(text)
            if m:
                return int(m.group())
    return None


def _study_type(pub_types):
    lowered = [t.casefold() for t in pub_types]
    for key, label in (("review", "review"), ("clinical trial", "clinical trial"),
                       ("meta-analysis", "meta-analysis"), ("randomized", "randomized trial")):
        if any(key in t for t in lowered):
            return label
    non_generic = [t for t in pub_types if t.casefold() not in {"journal article", "research support"}]
    return non_generic[0] if non_generic else (pub_types[0] if pub_types else None)


def _parse_pubmed_xml(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    records = []
    for citation in root.findall(".//PubmedArticle"):
        article = citation.find(".//MedlineCitation/Article")
        medline = citation.find(".//MedlineCitation")
        if article is None or medline is None:
            continue
        ids = {
            node.attrib.get("IdType", "").lower(): (node.text or "").strip()
            for node in citation.findall(".//PubmedData/ArticleIdList/ArticleId")
        }
        pmid = normalize_pmid(_node_text(medline.find("PMID")) or ids.get("pubmed"))
        if not pmid:
            continue
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = _node_text(node)
            if text:
                label = node.attrib.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)
        pub_types = [t for t in (_node_text(n) for n in article.findall(".//PublicationTypeList/PublicationType")) if t]
        comments = [node.attrib.get("RefType", "")
                    for node in medline.findall(".//CommentsCorrectionsList/CommentsCorrections")]
        journal_node = article.find(".//Journal")
        journal = _node_text(journal_node.find("Title")) if journal_node is not None else None
        retracted = any(t.casefold() in {"retracted publication", "retraction of publication"} for t in pub_types) \
            or any("retraction" in c.casefold() for c in comments)
        records.append({
            "pmid": pmid,
            "doi": normalize_doi(ids.get("doi")),
            "pmcid": ids.get("pmc") or None,
            "title": _node_text(article.find("ArticleTitle")) or "No Title",
            "year": _pubmed_year(article),
            "journal": journal,
            "study_type": _study_type(pub_types),
            "abstract": " ".join(abstract_parts) or None,
            "is_preprint": any("preprint" in t.casefold() for t in pub_types),
            "is_retracted": retracted,
        })
    return records


def search_NCBI(
    query: str = "",
    max_results: int = 3,
    mini_handler = None,
    out_dir: str = None
) -> str:
    """Search PubMed through NCBI E-utilities (esearch -> efetch) for articles related to the query

    Unlike search_PubMed (langchain), each record carries PMID, DOI, year, journal,
    study type and retraction / preprint flags parsed from the PubMed XML.

    When searching, you should consider:

    1. Search Fields: Title [ti], Abstract [ab], Title/Abstract [tiab], Author [au],
       MeSH Terms [mesh], Publication Date [dp]
    2. Boolean Operators: AND, OR, NOT, parentheses () for grouping
    3. Examples:
    - Basic: GATA4 cardiomyocyte
    - Field-specific: GATA4[tiab] AND heart development[mesh]
    - Date range: GATA4 AND (2020[dp]:2024[dp])

    Env (optional): NCBI_API_KEY (lifts rate limit 3 -> 10 rps), PUBMED_EMAIL / NCBI_EMAIL

    Args:
        query (str): The query to search for
        max_results (int): The maximum number of results to return, default is 3 (capped at 15)
        mini_handler: The LLM handler for summarizing abstracts
        out_dir (str): Directory to download open-access PDFs into; None skips downloading
    """
    try:
        query = _bounded_query(query)
        max_results = _bounded_limit(max_results, MAX_SEARCH_RESULTS)

        # Search for PMIDs related to the query
        payload = _get(
            f"{NCBI_BASE_URL}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results,
                    "retmode": "json", "sort": "relevance", **_ncbi_params()},
        )
        result = payload.get("esearchresult", {}) if isinstance(payload, dict) else {}
        pmids = [p for p in (normalize_pmid(x) for x in result.get("idlist", [])) if p][:max_results]
        if not pmids:
            return "No results found on NCBI PubMed."

        # Fetch canonical metadata for the PMIDs
        xml_text = _get(
            f"{NCBI_BASE_URL}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **_ncbi_params()},
            as_json=False,
        )
        records = _parse_pubmed_xml(str(xml_text))
    except Exception as e:
        return f"Error during NCBI search: {e}"

    # Extract the relevant information from the search results
    data_list = []
    downloads = []
    for rec in records:
        url = "https://pubmed.ncbi.nlm.nih.gov/" + rec["pmid"]
        tags = "".join(t for t, flag in ((" [RETRACTED]", rec["is_retracted"]), (" [PREPRINT]", rec["is_preprint"])) if flag)
        info = f"Year: {rec['year']} | Journal: {rec['journal']} | DOI: {rec['doi']} | Type: {rec['study_type']}"

        content = "No abstract available."
        if rec["abstract"]:
            content = Summarize_Agent(rec["abstract"], mini_handler) if mini_handler else rec["abstract"]

        pdf_line = ''
        if out_dir:
            pdf_path, msg = download_paper(out_dir, rec["pmcid"] or rec["pmid"], pmid=rec["pmid"], pmcid=rec["pmcid"], doi=rec["doi"])
            downloads.append((rec["title"], pdf_path, msg))
            pdf_line = '\n' + 'PDF: ' + (pdf_path or f'Not downloaded ({msg})')

        data_list.append('Title: ' + rec["title"] + tags + '\n' + 'Url: ' + url + '\n' + info + pdf_line + '\n' + 'Summary: ' + content)

    if out_dir:
        report_downloads("NCBI", query, downloads, out_dir)

    return "\n".join(data_list) + '\n' + "Data source: NCBI PubMed (E-utilities)"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run search_NCBI on one or more queries.")
    parser.add_argument("--query", nargs="+", required=True, help="One or more queries, each quoted.")
    parser.add_argument("--max_results", type=int, default=3, help="Max papers per query (capped at 15).")
    parser.add_argument("--out_dir", default=None, help="Download open-access PDFs here. Default: no downloads.")
    args = parser.parse_args()
    for q in args.query:
        print(f"===== {q}")
        print(search_NCBI(query=q, max_results=args.max_results, out_dir=args.out_dir))
