"""Answer the pipeline's Claude prompts from an interactive Claude Code session instead of the API.

A drop-in for LLM_interface.claude_api.get_json, used with --llm_backend session. Each call is keyed by
the program (set handler.key before the call) and works in two passes over

    <session_dir>/<step>/<key>__<hash>.request.json   {"key", "system", "prompt", "schema", "answer_file"}  (written)
    <session_dir>/<step>/<key>__<hash>.answer.json    the answer: one JSON object matching "schema"         (read)

<hash> is the first 8 hex digits of sha1(system + prompt), so an answer is only ever read back for the exact
prompt it answered: a rerun with --overwrite, a changed label or new evidence writes a new request.

First pass: no answer yet -> the request is written and the call returns None (pending). Someone (e.g.
Claude Code agents) writes each answer file. Second pass: the answer is validated against the same pydantic
schema the API call would use and returned. Nothing is billed to the API. The steps default to this backend
(--llm_backend session) with --session_dir <run folder>/Session_LLM.
"""
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError


class session_llm:
    def __init__(self, session_dir, step, model_label="claude-code-session"):
        self.dir = Path(session_dir) / step
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_label = model_label
        self.key = None
        self.budget, self.budget_step, self.budget_reserve = None, "", 0.0   # same attributes as claude_api
        self.n_pending = 0
        self.last_status, self.last_error = None, None   # of the latest call: answered | pending | invalid

    def get_json(self, system_prompt, prompt, schema_model, effort="medium", max_tokens=4000):
        if not self.key:
            raise ValueError("set handler.key (e.g. 'P12') before each call")
        name = f"{self.key}__{hashlib.sha1((system_prompt + prompt).encode()).hexdigest()[:8]}"
        answer = self.dir / f"{name}.answer.json"
        if answer.exists():
            try:
                parsed = schema_model.model_validate_json(answer.read_text())
            except ValidationError as e:
                print(f"  [session] {answer} does not match the schema: {e}")
                self.last_status, self.last_error = "invalid", f"{answer.name} does not match the schema"
                return None
            self.last_status, self.last_error = "answered", None
            return parsed
        (self.dir / f"{name}.request.json").write_text(json.dumps({
            "key": name, "system": system_prompt, "prompt": prompt,
            "schema": schema_model.model_json_schema(), "answer_file": str(answer)}, indent=2))
        self.n_pending += 1
        self.last_status, self.last_error = "pending", None
        print(f"  [session] {name}: request written; answer pending ({answer})")
        return None

    def report(self):
        """Closing note of a session run: how many answers are still to be written before the second pass."""
        if self.n_pending:
            print(f"[session] {self.n_pending} request(s) pending in {self.dir}: write each *.answer.json (e.g. with "
                  f"Claude Code agents), then rerun the same command to import them")
