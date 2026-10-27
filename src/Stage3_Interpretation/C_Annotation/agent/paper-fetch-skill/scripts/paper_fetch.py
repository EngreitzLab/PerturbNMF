#!/usr/bin/env python3
"""Fetch, read and record a paper for a program's literature step.

    action "fetch"  download the open-access PDF of one paper (direct PDF links, PMC open data, PMID->PMCID,
                    Unpaywall) into out_dir; with `literature`, the paper's row gets its pdf_path
    action "read"   plain text of pages start_page .. start_page + max_pages - 1 of a PDF
    action "note"   record what a paper shows in its literature.jsonl row: status read | irrelevant, finding,
                    stance, context, read_from

Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.pdf_download import download_paper  # noqa: E402
from _lib.skill_lib import SkillError, read_jsonl, require_str, run, write_jsonl  # noqa: E402

STANCES = ("supports", "contradicts", "mixed", "context_only")


def find_row(rows, req):
    """Index of the literature row matching req's pmid, else doi, else title."""
    for key in ("pmid", "doi"):
        if req.get(key):
            hits = [i for i, r in enumerate(rows) if str(r.get(key) or "").lower() == str(req[key]).lower()]
            if hits:
                return hits[0]
    if req.get("title"):
        hits = [i for i, r in enumerate(rows) if str(r.get("title", "")).casefold() == req["title"].casefold()]
        if hits:
            return hits[0]
    raise SkillError("not_found", "no literature row matches the given pmid / doi / title")


def fetch(req):
    out_dir = Path(require_str(req, "out_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(req.get(k) for k in ("pmid", "pmcid", "doi", "pdf_urls")):
        raise SkillError("invalid_input", "give at least one of pmid, pmcid, doi, pdf_urls")
    name = req.get("name") or req.get("pmcid") or req.get("pmid") or req.get("doi")
    path, message = download_paper(out_dir, name, pmid=req.get("pmid"), pmcid=req.get("pmcid"), doi=req.get("doi"),
                                   pdf_urls=req.get("pdf_urls") or ())
    out = {"pdf_path": str(Path(path).resolve()) if path else None, "message": message}
    if path and req.get("literature"):
        rows = read_jsonl(req["literature"])
        i = find_row(rows, req)
        rows[i]["pdf_path"] = out["pdf_path"]
        write_jsonl(req["literature"], rows)
        out["literature_row_updated"] = True
    return out


def read(req):
    import pymupdf
    path = Path(require_str(req, "pdf_path"))
    if not path.is_file():
        raise SkillError("not_found", f"no PDF at {path}")
    start = max(1, int(req.get("start_page", 1)))
    n = max(1, min(int(req.get("max_pages", 3)), 10))
    max_chars = int(req.get("max_chars", 20000))
    with pymupdf.open(str(path)) as doc:
        last = min(doc.page_count, start + n - 1)
        text = "\n".join(doc[i].get_text() for i in range(start - 1, last))
        return {"pdf_path": str(path), "page_count": doc.page_count, "pages": [start, last],
                "text": text[:max_chars], "truncated": len(text) > max_chars}


def note(req):
    lit = require_str(req, "literature")
    status = req.get("status", "read")
    if status not in ("read", "irrelevant"):
        raise SkillError("invalid_input", "`status` is read or irrelevant")
    rows = read_jsonl(lit)
    i = find_row(rows, req)
    update = {"status": status}
    if status == "read":
        update["finding"] = require_str(req, "finding")
        stance = req.get("stance", "context_only")
        if stance not in STANCES:
            raise SkillError("invalid_input", f"`stance` must be one of {STANCES}")
        update["stance"] = stance
        update["read_from"] = req.get("read_from", "fulltext" if rows[i].get("pdf_path") else "abstract")
        if req.get("context"):
            update["context"] = req["context"]
        if req.get("genes"):
            update["genes"] = list(dict.fromkeys([*rows[i].get("genes", []), *req["genes"]]))
    rows[i].update(update)
    write_jsonl(lit, rows)
    return {"literature": lit, "row": {k: rows[i].get(k) for k in ("pmid", "doi", "title", "status", "stance",
                                                                   "finding", "genes")}}


def handle(req):
    action = req.get("action")
    if action not in ("fetch", "read", "note"):
        raise SkillError("invalid_input", "`action` must be fetch, read or note")
    return {"action": action, **{"fetch": fetch, "read": read, "note": note}[action](req)}


if __name__ == "__main__":
    sys.exit(run(handle))
