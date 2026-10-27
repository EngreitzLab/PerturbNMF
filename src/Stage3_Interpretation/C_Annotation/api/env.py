"""Keys and the API handler shared by the pipeline scripts (not by the skills, which read their environment)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from C_Annotation/.env into os.environ (variables already set win)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_handler(model):
    """Anthropic API handler (api/LLM_interface.claude_api); needs ANTHROPIC_API_KEY."""
    sys.path.insert(0, str(ROOT / "api"))
    from LLM_interface import claude_api
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (export it or add it to C_Annotation/.env)")
    return claude_api(api_key=key, model=model)
