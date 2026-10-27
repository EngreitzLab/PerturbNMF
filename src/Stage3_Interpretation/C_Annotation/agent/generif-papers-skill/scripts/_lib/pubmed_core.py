"""PubMed E-utilities helpers (vendored from pubmed-skill): esearch, efetch, XML parsing, ranking."""
import os
import re
import time
import xml.etree.ElementTree as ET

import requests


# NCBI E-utilities (PubMed), ported from gene-program-interpreter research/literature.py
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_SEARCH_RESULTS = 15
USER_AGENT = "PerturbNMF-annotation/0.1 (literature search)"
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
    params = {"tool": "PerturbNMF-annotation"}
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


def search_pmids(query, pool_size):
    """PMIDs of a PubMed esearch, in relevance order."""
    payload = _get(f"{NCBI_BASE_URL}/esearch.fcgi",
                   params={"db": "pubmed", "term": query, "retmax": pool_size, "retmode": "json",
                           "sort": "relevance", **_ncbi_params()})
    time.sleep(0.34)
    ids = (payload.get("esearchresult", {}) if isinstance(payload, dict) else {}).get("idlist", [])
    return [p for p in map(normalize_pmid, ids) if p]


def fetch_records(pmids):
    """{pmid: PubMed record} via one efetch call."""
    if not pmids:
        return {}
    xml_text = _get(f"{NCBI_BASE_URL}/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **_ncbi_params()},
                    as_json=False)
    time.sleep(0.34)
    return {r["pmid"]: r for r in _parse_pubmed_xml(str(xml_text))}


STOP_WORDS = {"cell", "cells", "line", "human", "mouse", "primary"}


def context_terms(cell_type, extra=()):
    words = re.findall(r"[A-Za-z0-9-]+", cell_type or "")
    terms = [w for w in words if len(w) > 3 and w.lower() not in STOP_WORDS] + list(extra or [])
    return [t.lower().rstrip("*") for t in dict.fromkeys(terms)]


def rank_records(records, terms, names=()):
    """Not retracted; gene name in the title first, then cell-type-matching papers, then primary research over
    reviews; stable otherwise."""
    in_title = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE) if names else None

    def score(r):
        text = f"{r.get('title') or ''} {r.get('abstract') or ''}".lower()
        return (bool(in_title) and not in_title.search(r.get("title") or ""),
                -any(t in text for t in terms), r.get("study_type") == "review")
    return sorted([r for r in records if not r.get("is_retracted")], key=score)
