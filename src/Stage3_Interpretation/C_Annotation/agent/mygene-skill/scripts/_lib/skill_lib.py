"""Helpers every skill script uses: JSON request/response, HTTP retries, program-file writes.

Canonical copy: C_Annotation/agent/_shared/skill_lib.py. Each skill carries an identical copy in
scripts/_lib/skill_lib.py (agent/_shared/sync_skill_lib.py copies it and checks the copies), so a skill
folder never imports from outside itself.
"""
import json
import sys
import time
from pathlib import Path

TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class SkillError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def run(handler):
    """Read one JSON object from stdin, call handler(request) -> dict, print {"ok": true, ...} or the error."""
    try:
        raw = sys.stdin.read()
        try:
            req = json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            raise SkillError("invalid_json", str(e))
        if not isinstance(req, dict):
            raise SkillError("invalid_input", "the request must be one JSON object")
        out = {"ok": True, **handler(req)}
        out.setdefault("warnings", [])
    except SkillError as e:
        out = {"ok": False, "error": {"code": e.code, "message": e.message}, "warnings": []}
    except Exception as e:   # network or API failures are reported, never raised
        out = {"ok": False, "error": {"code": "runtime_error", "message": f"{type(e).__name__}: {e}"},
               "warnings": []}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if out["ok"] else 1


def require_str_list(req, key, max_len=None):
    value = req.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise SkillError("invalid_input", f"`{key}` must be a non-empty list of strings")
    if max_len and len(value) > max_len:
        raise SkillError("invalid_input", f"`{key}` has {len(value)} items; at most {max_len}")
    return [v.strip() for v in value]


def require_str(req, key):
    value = req.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillError("invalid_input", f"`{key}` must be a non-empty string")
    return value.strip()


def request_with_retry(method, url, retries=3, timeout=60, **kwargs):
    """requests.<method> with exponential backoff on transient status codes and dropped connections."""
    import requests
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
            continue
        if resp.status_code in TRANSIENT_STATUS and attempt < retries:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    tmp.replace(path)


def merge_jsonl(path, key, updates):
    """Rows keyed by row[key]: each update {key value: {field: value}} is merged into its row (created when
    missing); other rows are untouched. Returns the number of rows written."""
    rows = read_jsonl(path)
    index = {r[key]: r for r in rows}
    for k, fields in updates.items():
        if k not in index:
            index[k] = {key: k}
            rows.append(index[k])
        index[k].update(fields)
    write_jsonl(path, rows)
    return len(rows)


def append_jsonl(path, new_rows, dedupe):
    """Append rows whose dedupe(row) is not yet in the file; returns (added, skipped as duplicates)."""
    rows = read_jsonl(path)
    seen = {dedupe(r) for r in rows}
    added = 0
    for r in new_rows:
        k = dedupe(r)
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
        added += 1
    write_jsonl(path, rows)
    return added, len(new_rows) - added
