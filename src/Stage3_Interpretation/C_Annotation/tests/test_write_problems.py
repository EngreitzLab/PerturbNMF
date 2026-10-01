"""--write-problems: failing items get problems.json (prefix stripped), passing items lose it."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script, dispatch, *extra):
    return subprocess.run([sys.executable, str(ROOT / script), "--dispatch", str(dispatch), *extra, "--write-problems"],
                          capture_output=True, text=True)


def test_program_validator_writes_and_clears_problems(tmp_path):
    item = tmp_path / "v3_p0"
    item.mkdir()
    (item / "prompt.md").write_text("no genes here")
    (item / "answer.json").write_text("{not json")
    run("ProgramAnnotatorV3/scripts/validate_annotation_answers.py", tmp_path, "--programs", "0")
    problems = json.loads((item / "problems.json").read_text())["problems"]
    assert problems and not problems[0].startswith("P0:")
    (item / "answer.json").unlink()
    (item / "problems.json").write_text(json.dumps({"problems": ["x"], "sources": {"validate_annotation_answers": ["x"]}}))
    # a missing answer is still a problem; the entry is rewritten, not left stale
    run("ProgramAnnotatorV3/scripts/validate_annotation_answers.py", tmp_path, "--programs", "0")
    assert json.loads((item / "problems.json").read_text())["problems"] == ["no answer.json"]


def test_group_validator_writes_problems(tmp_path):
    item = tmp_path / "rg_p3"
    item.mkdir()
    (item / "prompt.md").write_text("")
    (item / "answer.json").write_text("{not json")
    run("RegulatorGroupAnnotator/scripts/validate_group_answers.py", tmp_path)
    problems = json.loads((item / "problems.json").read_text())["problems"]
    assert problems and not problems[0].startswith("G3:")


def test_citation_validator_writes_and_clears_problems(tmp_path):
    item = tmp_path / "cite_p2"
    item.mkdir()
    claim = {"claim_id": "R1", "symbol": "X", "database": [], "literature": [
        {"pmid": "111", "sentence": "X binds Y.", "title": "A", "year": "2000"}]}
    (item / "candidates.json").write_text(json.dumps({"claims": [claim]}))
    (item / "answer.json").write_text(json.dumps({"claims": [{"claim_id": "R1", "supports": [
        {"ref": "L9", "pmid": "999", "quote": "nothing", "strength": "direct"}]}]}))
    run("annotator_core/validate_citation_answers.py", tmp_path, "--arm", "cite")
    problems = json.loads((item / "problems.json").read_text())["problems"]
    assert problems == ["R1: L9 was not offered"]
    (item / "answer.json").write_text(json.dumps({"claims": [{"claim_id": "R1", "supports": [
        {"ref": "L1", "pmid": "111", "quote": "X binds Y", "strength": "direct"}]}]}))
    run("annotator_core/validate_citation_answers.py", tmp_path, "--arm", "cite")
    assert not (item / "problems.json").exists()
