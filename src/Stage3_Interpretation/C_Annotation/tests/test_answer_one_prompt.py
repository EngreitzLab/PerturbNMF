"""answer_one_prompt.py: the minimal claude -p call, prompt-hash skipping, and the repair merge.

A fake `claude` (CLAUDE_BIN) records its argv and stdin and prints a canned --output-format json
payload, so nothing here calls the real CLI.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "annotator_core"
sys.path.insert(0, str(CORE))
from answer_one_prompt import merge_patch  # noqa: E402

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
log = Path(os.environ["FAKE_LOG"])
log.write_text(json.dumps({"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd()}))
print(json.dumps({"result": os.environ["FAKE_RESULT"], "is_error": False, "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 5, "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 100, "output_tokens": 50}}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "call.json"
    monkeypatch.setenv("CLAUDE_BIN", str(script))
    monkeypatch.setenv("FAKE_LOG", str(log))
    return log


def run(mode: str, directory: Path, result: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, FAKE_RESULT=result)
    return subprocess.run([sys.executable, str(CORE / "answer_one_prompt.py"), mode, str(directory)],
                          capture_output=True, text=True, env=env)


def make_item(tmp_path: Path, system: str | None = "STATIC RULES") -> Path:
    directory = tmp_path / "v3_p0"
    directory.mkdir()
    (directory / "prompt.md").write_text("DATA for program 0")
    if system is not None:
        (directory / "system.md").write_text(system)
    return directory


def test_minimal_call_flags_and_neutral_cwd(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    completed = run("answer", directory, '```json\n{"label": "x", "log2fc": +1.5}\n```')
    assert completed.returncode == 0, completed.stdout + completed.stderr
    call = json.loads(fake_claude.read_text())
    argv = call["argv"]
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--system-prompt-file") + 1] == str(directory / "system.md")
    assert call["stdin"] == "DATA for program 0"
    assert not call["cwd"].startswith(str(Path.home() / "Claude"))
    assert json.loads((directory / "answer.json").read_text()) == {"label": "x", "log2fc": 1.5}
    usage = json.loads((directory / "usage.jsonl").read_text().splitlines()[0])
    assert usage["kind"] == "answer" and usage["cost_usd"] == 0.01


def test_default_system_prompt_when_no_system_md(tmp_path, fake_claude):
    directory = make_item(tmp_path, system=None)
    assert run("answer", directory, '{"label": "x"}').returncode == 0
    argv = json.loads(fake_claude.read_text())["argv"]
    assert "--system-prompt" in argv and "--system-prompt-file" not in argv


def test_skip_unchanged_redispatch_changed(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    assert run("answer", directory, '{"label": "first"}').returncode == 0
    assert "skip" in run("answer", directory, '{"label": "second"}').stdout
    (directory / "prompt.md").write_text("DATA for program 0, revised")
    assert run("answer", directory, '{"label": "second"}').returncode == 0
    assert json.loads((directory / "answer.json").read_text())["label"] == "second"
    assert json.loads((directory / "answer.stale.1.json").read_text())["label"] == "first"


def test_legacy_answer_without_hash_is_skipped(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    (directory / "answer.json").write_text('{"label": "old"}')
    assert "no prompt hash recorded" in run("answer", directory, '{"label": "new"}').stdout
    assert not fake_claude.exists()


def test_incomplete_answer_fails_and_is_moved_aside_next_pass(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    assert run("answer", directory, '{"label": "trunc').returncode == 1
    assert run("answer", directory, '{"label": "ok"}').returncode == 0
    assert (directory / "answer.invalid.1.json").exists()


def test_repair_merges_patch_and_keeps_rejected(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    assert run("answer", directory, json.dumps({"label": "bad", "regulators": [
        {"symbol": "A", "role": "x"}, {"symbol": "B", "role": "y"}]})).returncode == 0
    (directory / "problems.json").write_text(json.dumps({"problems": ["label names coherence", "B role wrong"]}))
    patch = {"label": "good", "regulators": [{"symbol": "B", "role": "z"}]}
    assert run("repair", directory, json.dumps(patch)).returncode == 0
    merged = json.loads((directory / "answer.json").read_text())
    assert merged == {"label": "good", "regulators": [{"symbol": "A", "role": "x"}, {"symbol": "B", "role": "z"}]}
    assert json.loads((directory / "answer.rejected.1.json").read_text())["label"] == "bad"
    assert not (directory / "problems.json").exists()
    stdin = json.loads(fake_claude.read_text())["stdin"]
    assert "YOUR PREVIOUS ANSWER" in stdin and "B role wrong" in stdin
    assert "skip" in run("answer", directory, "{}").stdout  # repaired answer counts as current


def test_repair_budget(tmp_path, fake_claude):
    directory = make_item(tmp_path)
    assert run("answer", directory, '{"label": "bad"}').returncode == 0
    for _ in range(2):
        (directory / "problems.json").write_text('{"problems": ["p"]}')
        assert run("repair", directory, '{"label": "still bad"}').returncode == 0
    (directory / "problems.json").write_text('{"problems": ["p"]}')
    assert run("repair", directory, '{"label": "x"}').returncode == 1


def test_merge_patch_by_id_and_top_level():
    original = {"label": "a", "claims": [{"claim_id": "c1", "s": 1}, {"claim_id": "c2", "s": 2}], "notes": ["x"]}
    patch = {"claims": [{"claim_id": "c2", "s": 3}, {"claim_id": "c3", "s": 4}], "notes": ["y"]}
    assert merge_patch(original, patch) == {
        "label": "a", "claims": [{"claim_id": "c1", "s": 1}, {"claim_id": "c2", "s": 3}, {"claim_id": "c3", "s": 4}],
        "notes": ["y"]}
