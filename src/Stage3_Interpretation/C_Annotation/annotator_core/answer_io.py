"""Read an LLM answer file written by run_blinded_annotations.sh.

`claude -p` sometimes wraps the JSON in a markdown fence even when told not to, so the fence is
stripped before parsing. Shared by every annotator, gate and viewer.
"""
import json
import re
from pathlib import Path


def load_answer(path: Path) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", path.read_text().strip(), flags=re.MULTILINE)
    return json.loads(raw)
