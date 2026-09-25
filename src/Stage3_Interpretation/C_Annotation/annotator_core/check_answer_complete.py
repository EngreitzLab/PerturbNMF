"""Exit 0 if an annotation answer is complete, 1 otherwise.

A non-empty file is not a complete answer: `claude -p` has returned the last 86 bytes of a JSON
object and exited 0. So a JSON answer counts only if it parses as an object that
carries a label (an annotation), a programs list (a collision-pass answer) or a claims list (a
citation-pass answer); a markdown answer
only if it carries a "Program label" line.

Usage: python check_answer_complete.py <answer.json|answer.md>
"""
import json
import re
import sys
from pathlib import Path


def is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if path.suffix == ".md":
        return "Program label" in text
    raw = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return (
        bool(str(payload.get("label", "")).strip())
        or bool(payload.get("programs"))
        or bool(payload.get("claims"))
    )


if __name__ == "__main__":
    sys.exit(0 if is_complete(Path(sys.argv[1])) else 1)
