"""Answer the pipeline's Claude prompts from an interactive Claude Code session instead of the API.

A drop-in for LLM_interface.claude_api.get_json, used with --llm_backend session. Each call is keyed by
the program (set handler.key before the call) and works in two passes over

    <session_dir>/<step>/<key>.request.json   {"key", "system", "prompt", "schema", "answer_file"}  (written)
    <session_dir>/<step>/<key>.answer.json    the answer: one JSON object matching "schema"         (read)

First pass: no answer yet -> the request is written and the call returns None (pending). Someone (e.g.
Claude Code agents) writes each answer file. Second pass: the answer is validated against the same pydantic
schema the API call would use and returned. Nothing is billed to the API.
"""
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

    def get_json(self, system_prompt, prompt, schema_model, effort="medium", max_tokens=4000):
        if not self.key:
            raise ValueError("set handler.key (e.g. 'P12') before each call")
        answer = self.dir / f"{self.key}.answer.json"
        if answer.exists():
            try:
                return schema_model.model_validate_json(answer.read_text())
            except ValidationError as e:
                print(f"  [session] {answer} does not match the schema: {e}")
                return None
        (self.dir / f"{self.key}.request.json").write_text(json.dumps({
            "key": self.key, "system": system_prompt, "prompt": prompt,
            "schema": schema_model.model_json_schema(), "answer_file": str(answer)}, indent=2))
        self.n_pending += 1
        print(f"  [session] {self.key}: request written; answer pending ({answer})")
        return None
