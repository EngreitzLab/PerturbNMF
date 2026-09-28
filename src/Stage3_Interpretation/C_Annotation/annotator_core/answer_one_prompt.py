"""Answer, or repair, one blinded prompt directory with a headless `claude -p` call.

Called once per directory by run_blinded_annotations.sh (answer) and repair_rejected_answers.sh
(repair). A directory holds `prompt.md` (the per-item data) and, optionally, `system.md` (the
static instructions, byte-identical across a batch).

The call is minimal on purpose:

- `--safe-mode --strict-mcp-config --disable-slash-commands` and a working directory outside the
  user's tree. Minimal call: no CLAUDE.md, skills, plugins, hooks or MCP from the caller's
  directory, so answers do not depend on where the call is launched.
- `--tools ""`: no tools at all. `--allowed-tools ""` (used before) only governs permission
  prompts; the old calls ran Bash and one read another program's prompt.
- `--system-prompt-file system.md`: replaces Claude Code's system prompt with the task's static
  instructions, which the API then caches across the batch.
- `--output-format json`: the answer arrives with its token usage and cost, appended to
  `usage.jsonl` in the directory.

Skipping: a directory is done when its answer is complete (check_answer_complete.py) AND
`answer.prompt_sha256` matches the current system.md + prompt.md. A changed prompt re-dispatches
(the old answer is kept as answer.stale.<n>.json). Answers written before hashes existed are
skipped with a note, as before.

Repair: when a validator has written `problems.json` (`--write-problems`), the prompt is re-sent
with the previous answer and the problems, and the model returns only the fields that change.
They are merged into the answer (the rejected answer is kept as answer.rejected.<n>.json); run
the validator again afterwards. At most MAX_REPAIRS per directory, then a full re-dispatch is
needed (delete answer.json).

Usage:
    python answer_one_prompt.py answer <dir>
    python answer_one_prompt.py repair <dir>
    python answer_one_prompt.py needs-work <dir>     # exit 0 if the directory still needs an answer
Env: ANNOTATOR_MODEL (default sonnet), CLAUDE_BIN (default claude; tests use a fake).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_answer_complete import is_complete  # noqa: E402

DEFAULT_SYSTEM = (
    "You are a careful computational biologist annotating results from a Perturb-seq screen. "
    "Follow the instructions in the user message exactly and answer with the requested JSON only."
)
MAX_REPAIRS = 2
ID_KEYS = ("claim_id", "symbol", "confounder", "name", "reading")


def prompt_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    system = directory / "system.md"
    digest.update(system.read_bytes() if system.exists() else b"")
    digest.update(b"\0")
    digest.update((directory / "prompt.md").read_bytes())
    return digest.hexdigest()


def next_free(directory: Path, stem: str) -> Path:
    n = 1
    while (directory / f"{stem}.{n}.json").exists():
        n += 1
    return directory / f"{stem}.{n}.json"


def clean_answer_text(text: str) -> str:
    """Strip a stray markdown fence and a leading '+' on JSON numbers ("log2fc": +1.0).

    Models add a fence even when told not to, and copy the signed log2FCs the prompts print."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return re.sub(r'(": *)\+([0-9.])', r"\1\2", text)


def needs_work(directory: Path) -> bool:
    answer = directory / "answer.json"
    if not is_complete(answer):
        return True
    recorded = directory / "answer.prompt_sha256"
    if not recorded.exists():
        return False  # answered before hashes existed; keep the old behaviour
    return recorded.read_text().strip() != prompt_sha256(directory)


def call_claude(system_text: str | None, system_file: Path | None, user_text: str) -> dict:
    """One tool-less, context-free `claude -p` call. Returns the parsed --output-format json."""
    command = [
        os.environ.get("CLAUDE_BIN", "claude"), "-p",
        "--model", os.environ.get("ANNOTATOR_MODEL", "sonnet"),
        "--safe-mode", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
        "--no-session-persistence", "--output-format", "json",
    ]
    command += ["--system-prompt-file", str(system_file)] if system_file else ["--system-prompt", system_text or DEFAULT_SYSTEM]
    # A neutral working directory: nothing to discover upward, nothing to read beside it.
    with tempfile.TemporaryDirectory(prefix="annotator_") as neutral_cwd:
        completed = subprocess.run(command, input=user_text, capture_output=True, text=True, cwd=neutral_cwd)
    if completed.returncode != 0 or not completed.stdout.strip():
        # A usage limit exits nonzero with an empty stderr; dispatch_until_complete.sh retries.
        raise RuntimeError(f"claude exited {completed.returncode}: {completed.stderr.strip()[:200]}")
    payload = json.loads(completed.stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"claude reported an error: {str(payload.get('result'))[:200]}")
    return payload


def log_usage(directory: Path, kind: str, payload: dict, prompt_hash: str) -> None:
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "model": os.environ.get("ANNOTATOR_MODEL", "sonnet"),
        "prompt_sha256": prompt_hash,
        "cost_usd": payload.get("total_cost_usd"),
        "duration_ms": payload.get("duration_ms"),
        "usage": {key: payload.get("usage", {}).get(key) for key in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
    }
    with (directory / "usage.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def system_file_for(directory: Path) -> Path | None:
    system = directory / "system.md"
    return system if system.exists() else None


def answer(directory: Path) -> int:
    name = directory.name
    if not needs_work(directory):
        note = "" if (directory / "answer.prompt_sha256").exists() else ", no prompt hash recorded"
        print(f"skip  {name} (already answered{note})")
        return 0
    answer_path = directory / "answer.json"
    if answer_path.exists() and answer_path.stat().st_size:
        stem = "answer.stale" if is_complete(answer_path) else "answer.invalid"
        answer_path.rename(next_free(directory, stem))

    prompt_hash = prompt_sha256(directory)
    try:
        payload = call_claude(None, system_file_for(directory), (directory / "prompt.md").read_text(encoding="utf-8"))
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"FAIL  {name} — {error}")
        return 1
    log_usage(directory, "answer", payload, prompt_hash)
    answer_path.write_text(clean_answer_text(str(payload.get("result", ""))) + "\n", encoding="utf-8")
    if not is_complete(answer_path):
        print(f"FAIL  {name} — incomplete answer ({answer_path.stat().st_size} bytes), will retry")
        return 1
    (directory / "answer.prompt_sha256").write_text(prompt_hash + "\n")
    print(f"ok    {name} ({answer_path.stat().st_size} bytes, ${payload.get('total_cost_usd', 0):.3f})")
    return 0


def item_id(item: dict) -> tuple[str, object] | None:
    for key in ID_KEYS:
        if key in item:
            return key, item[key]
    return None


def merge_patch(original: dict, patch: dict) -> dict:
    """Top-level fields in the patch replace the original's. For a list of objects carrying an id
    field (claim_id, symbol, ...), patched items replace items with the same id; new ids append."""
    merged = dict(original)
    for key, value in patch.items():
        old = original.get(key)
        if (isinstance(value, list) and isinstance(old, list) and value and old
                and all(isinstance(item, dict) and item_id(item) for item in value)
                and all(isinstance(item, dict) and item_id(item) for item in old)):
            replaced = {item_id(item): item for item in value}
            merged_list = [replaced.pop(item_id(item), item) for item in old]
            merged[key] = merged_list + list(replaced.values())
        else:
            merged[key] = value
    return merged


def repair_count(directory: Path) -> int:
    usage = directory / "usage.jsonl"
    if not usage.exists():
        return 0
    return sum(1 for line in usage.read_text().splitlines() if line.strip() and json.loads(line).get("kind") == "repair")


def repair(directory: Path) -> int:
    name = directory.name
    problems_path = directory / "problems.json"
    answer_path = directory / "answer.json"
    if not problems_path.exists():
        return 0
    if not is_complete(answer_path):
        print(f"skip  {name} (no complete answer to repair — run the dispatch)")
        return 0
    if repair_count(directory) >= MAX_REPAIRS:
        print(f"FAIL  {name} — {MAX_REPAIRS} repairs already; delete answer.json for a full re-dispatch")
        return 1
    problems = json.loads(problems_path.read_text()).get("problems", [])
    original = json.loads(clean_answer_text(answer_path.read_text(encoding="utf-8")))
    id_note = ", ".join(ID_KEYS)
    user_text = (
        (directory / "prompt.md").read_text(encoding="utf-8")
        + "\n\n=== YOUR PREVIOUS ANSWER ===\n" + json.dumps(original, indent=1, ensure_ascii=False)
        + "\n\n=== PROBLEMS THE CHECKER FOUND IN IT ===\n" + "\n".join(f"- {problem}" for problem in problems)
        + "\n\n=== WHAT TO RETURN ===\nFix every problem above, following the same instructions as before. "
        "Return ONE JSON object holding ONLY the top-level fields that must change, each with its full "
        f"corrected value. For a list whose items carry an id field ({id_note}), you may return only the "
        "items that change; they replace the items with the same id. JSON only, no commentary."
    )
    prompt_hash = prompt_sha256(directory)
    try:
        payload = call_claude(None, system_file_for(directory), user_text)
        patch = json.loads(clean_answer_text(str(payload.get("result", ""))))
    except (RuntimeError, json.JSONDecodeError) as error:
        print(f"FAIL  {name} — repair: {error}")
        return 1
    log_usage(directory, "repair", payload, prompt_hash)
    if not isinstance(patch, dict):
        print(f"FAIL  {name} — repair returned {type(patch).__name__}, not an object")
        return 1
    answer_path.rename(next_free(directory, "answer.rejected"))
    answer_path.write_text(json.dumps(merge_patch(original, patch), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "answer.prompt_sha256").write_text(prompt_hash + "\n")
    problems_path.rename(next_free(directory, "problems.repaired"))
    print(f"ok    {name} repaired {sorted(patch)} (${payload.get('total_cost_usd', 0):.3f}) — re-run the validator")
    return 0


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("answer", "repair", "needs-work"):
        print(__doc__)
        return 2
    mode, directory = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode == "needs-work":
        return 0 if needs_work(directory) else 1
    return answer(directory) if mode == "answer" else repair(directory)


if __name__ == "__main__":
    sys.exit(main())
