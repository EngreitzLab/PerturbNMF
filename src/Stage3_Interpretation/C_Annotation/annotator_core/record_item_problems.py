"""Record one gate's problems for one dispatch item in <item_dir>/problems.json.

Several gates judge the same item (validate_citation_answers.py, verify_cited_pmids.py), and a
repair pass re-prompts from their findings. Each gate owns its own entry under "sources", so a
gate that now passes clears only what it wrote; "problems" is the de-duplicated union, in order:

    {"problems": ["G1: L4 was not offered", ...],
     "sources": {"validate_citation_answers": [...], "verify_cited_pmids": [...]}}

When no gate has anything left, the file is removed: no problems.json means no known problem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

PROBLEMS_FILE = "problems.json"


def record_item_problems(item_dir: Path, source: str, problems: List[str]) -> None:
    path = item_dir / PROBLEMS_FILE
    sources = {}
    if path.exists():
        try:
            sources = json.loads(path.read_text()).get("sources") or {}
        except (json.JSONDecodeError, AttributeError):
            sources = {}
    if problems:
        sources[source] = list(dict.fromkeys(problems))
    else:
        sources.pop(source, None)
    if not sources:
        path.unlink(missing_ok=True)
        return
    union = list(dict.fromkeys(p for entries in sources.values() for p in entries))
    path.write_text(json.dumps({"problems": union, "sources": sources}, indent=1, ensure_ascii=False))
