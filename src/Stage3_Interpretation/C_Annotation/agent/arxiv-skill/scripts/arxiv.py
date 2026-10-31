#!/usr/bin/env python3
"""arXiv search through the arXiv export API (Atom): arXiv id, title, authors, year, primary category, abstract
and PDF link. Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import (
    SkillError,
    append_jsonl,
    request_with_retry,
    require_str,
    require_str_list,
    run,
)  # noqa: E402,F401

SKILL = "arxiv-skill"
ARXIV_API = "https://export.arxiv.org/api/query"
MAX_SEARCH_RESULTS = 15
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _text(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else None


def fetch_records(query, max_results):
    resp = request_with_retry(
        "GET",
        ARXIV_API,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        },
    )
    root = ET.fromstring(resp.text)
    records = []
    for e in root.findall("a:entry", NS):
        arxiv_id = (_text(e.find("a:id", NS)) or "").rsplit("/abs/", 1)[-1]
        published = _text(e.find("a:published", NS)) or ""
        doi = _text(e.find("arxiv:doi", NS))
        pdf = next(
            (
                l.attrib["href"]
                for l in e.findall("a:link", NS)
                if l.attrib.get("title") == "pdf"
            ),
            None,
        )
        authors = [_text(a.find("a:name", NS)) for a in e.findall("a:author", NS)]
        records.append(
            {
                "arxiv_id": arxiv_id,
                "pmid": None,
                "doi": doi.lower() if doi else None,
                "title": _text(e.find("a:title", NS)) or "No Title",
                "authors": ", ".join(authors[:3])
                + (" et al." if len(authors) > 3 else ""),
                "year": int(published[:4]) if re.match(r"\d{4}", published) else None,
                "journal": "arXiv",
                "study_type": "preprint",
                "is_preprint": True,
                "is_retracted": False,
                "category": (
                    e.find("arxiv:primary_category", NS).attrib.get("term")
                    if e.find("arxiv:primary_category", NS) is not None
                    else None
                ),
                "abstract": _text(e.find("a:summary", NS)),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_urls": [pdf] if pdf else [],
            }
        )
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
