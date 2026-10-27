"""Prompt files of the LLM steps, loaded with their identity so every output records the exact prompt it used.

    C_Annotation/prompts/<step>/<part>.md     e.g. prompts/interpretation/system.md, prompts/interpretation/task.md

Each file holds one prompt string exactly as the step sends it, plus one trailing newline that load() strips.
A prompt file is edited like code (git keeps its history); the outputs carry its id and sha256, so a changed
prompt is visible in every record made with it (status.py flags records made with an older prompt).
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str        # "<step>/<part>.md", relative to PROMPT_DIR
    text: str
    sha256: str


def load(prompt_id):
    """Prompt prompts/<prompt_id>; the file's one trailing newline is not part of the prompt."""
    path = PROMPT_DIR / prompt_id
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    text = text[:-1] if text.endswith("\n") else text
    return Prompt(prompt_id, text, hashlib.sha256(text.encode()).hexdigest())


def record(*prompts):
    """{prompt id: sha256} of the prompts behind an output, for its "prompts" field."""
    return {p.id: p.sha256 for p in prompts}
