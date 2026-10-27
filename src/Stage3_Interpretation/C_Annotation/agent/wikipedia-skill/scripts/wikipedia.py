#!/usr/bin/env python3
"""Wikipedia article search (MediaWiki action API): titles, URLs and the plain-text start of each article.
Reads one JSON object on stdin and writes one JSON object on stdout (see ../SKILL.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.skill_lib import request_with_retry, require_str, run  # noqa: E402

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "PerturbNMF-annotation/0.2 (gene program annotation)"


def wiki_get(params):
    return request_with_retry("GET", WIKI_API, params={**params, "format": "json", "formatversion": 2},
                              headers={"User-Agent": USER_AGENT}).json()


def handle(req):
    query = require_str(req, "query")
    n = max(1, min(int(req.get("max_results", 3)), 10))
    chars = int(req.get("max_chars", 3000))
    hits = wiki_get({"action": "query", "list": "search", "srsearch": query, "srlimit": n})["query"]["search"]
    if not hits:
        return {"source": WIKI_API, "query": query, "articles": []}
    pages = wiki_get({"action": "query", "prop": "extracts", "explaintext": 1,
                      "titles": "|".join(h["title"] for h in hits)})["query"]["pages"]
    text = {p["title"]: p.get("extract") or "" for p in pages}
    return {"source": WIKI_API, "query": query,
            "articles": [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                          "text": text.get(h["title"], "")[:chars]} for h in hits]}


if __name__ == "__main__":
    sys.exit(run(handle))
