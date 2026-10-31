#!/usr/bin/env python3
"""PubMed search through NCBI E-utilities (esearch + efetch): PMID, DOI, PMCID, title, year, journal, study type, abstract and retraction / preprint flags.

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import SkillError, append_jsonl, require_str, require_str_list, run  # noqa: E402,F401

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
            doi = doi[len(prefix) :]
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
            resp = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
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
    for path in (
        ".//ArticleDate/Year",
        ".//Journal/JournalIssue/PubDate/Year",
        ".//Journal/JournalIssue/PubDate/MedlineDate",
    ):
        text = _node_text(article.find(path))
        if text:
            m = _YEAR_RE.search(text)
            if m:
                return int(m.group())
    return None


def _study_type(pub_types):
    lowered = [t.casefold() for t in pub_types]
    for key, label in (
        ("review", "review"),
        ("clinical trial", "clinical trial"),
        ("meta-analysis", "meta-analysis"),
        ("randomized", "randomized trial"),
    ):
        if any(key in t for t in lowered):
            return label
    non_generic = [
        t
        for t in pub_types
        if t.casefold() not in {"journal article", "research support"}
    ]
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
        pub_types = [
            t
            for t in (
                _node_text(n)
                for n in article.findall(".//PublicationTypeList/PublicationType")
            )
            if t
        ]
        comments = [
            node.attrib.get("RefType", "")
            for node in medline.findall(
                ".//CommentsCorrectionsList/CommentsCorrections"
            )
        ]
        journal_node = article.find(".//Journal")
        journal = (
            _node_text(journal_node.find("Title")) if journal_node is not None else None
        )
        retracted = any(
            t.casefold() in {"retracted publication", "retraction of publication"}
            for t in pub_types
        ) or any("retraction" in c.casefold() for c in comments)
        records.append(
            {
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
            }
        )
    return records


SKILL = "pubmed-skill"


def fetch_records(query, max_results):
    payload = _get(
        f"{NCBI_BASE_URL}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": _bounded_query(query),
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            **_ncbi_params(),
        },
    )
    result = payload.get("esearchresult", {}) if isinstance(payload, dict) else {}
    pmids = [p for p in (normalize_pmid(x) for x in result.get("idlist", [])) if p][
        :max_results
    ]
    if not pmids:
        return []
    xml_text = _get(
        f"{NCBI_BASE_URL}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            **_ncbi_params(),
        },
        as_json=False,
    )
    return _parse_pubmed_xml(str(xml_text))


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
