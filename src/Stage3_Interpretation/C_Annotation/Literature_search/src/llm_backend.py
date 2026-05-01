"""Configurable LLM backend for literature search.

Supports: anthropic, openai, deepseek, gemini.
Provider classes adapted from DeepRare/api/interface.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------------------------------------------
# Provider classes (from DeepRare/api/interface.py)
# ------------------------------------------------------------------

class Openai_api:
    def __init__(self, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                seed=seed,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return str(completion.choices[0].message.content)
        except Exception as e:
            print(e)
            return None


class deepseek_api:
    def __init__(self, api_key, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        if model == 'deepseek-v3-241226':
            self.model = "deepseek-chat"
        elif model == 'deepseek-r1-250120':
            self.model = "deepseek-reasoner"
        else:
            self.model = model

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return str(completion.choices[0].message.content)
        except Exception as e:
            print(e)
            return None


class gemini_api:
    def __init__(self, api_key, model):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            response = self.model.generate_content(full_prompt)
            return str(response.text)
        except Exception as e:
            print(e)
            return None


class claude_api:
    def __init__(self, api_key, model):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return str(message.content[0].text)
        except Exception as e:
            print(e)
            return None


class stanford_api:
    """Stanford AI API Gateway — OpenAI-compatible endpoint serving multiple models.

    Endpoints:
        GET  https://aiapi-prod.stanford.edu/v1/models
        POST https://aiapi-prod.stanford.edu/v1/chat/completions
        POST https://aiapi-prod.stanford.edu/v1/embeddings
        POST https://aiapi-prod.stanford.edu/v1/images/generations
    Auth: Bearer token via STANFORD_API_KEY
    """
    def __init__(self, api_key, base_url, model):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._validate_model(model)
        self.model = model

    def _validate_model(self, model):
        """Check if the requested model is available on the Stanford gateway."""
        try:
            available = [m.id for m in self.client.models.list().data]
            if model not in available:
                raise ValueError(
                    f"Model '{model}' not available on Stanford API. "
                    f"Available models: {available}"
                )
            logger.info("Stanford API model validated: %s", model)
        except Exception as e:
            if "not available" in str(e):
                raise
            logger.warning("Could not validate model (network error): %s", e)

    def get_completion(self, system_prompt, prompt, seed=42):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return str(completion.choices[0].message.content)
        except Exception as e:
            print(e)
            return None


# ------------------------------------------------------------------
# Provider registry
# ------------------------------------------------------------------

PROVIDER_MAP = {
    "anthropic": (claude_api,     "ANTHROPIC_API_KEY"),
    "stanford":  (stanford_api,   "STANFORD_API_KEY"),
    "openai":    (Openai_api,     "OPENAI_API_KEY"),
    "deepseek":  (deepseek_api,   "DEEPSEEK_API_KEY"),
    "gemini":    (gemini_api,     "GOOGLE_API_KEY"),
}

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "stanford":  "claude-4-5-sonnet",  # also: gpt-5, o1, o3-mini, gpt-5-mini, claude-3-haiku, claude-4-sonnet, claude-opus-4-6, Llama-4, gemini-2.5-pro
    "openai":    "gpt-4o",
    "deepseek":  "deepseek-v3-241226",
    "gemini":    "gemini-2.0-flash",
}


# ------------------------------------------------------------------
# LLMBackend wrapper
# ------------------------------------------------------------------

class LLMBackend:
    """Thin wrapper providing complete() and complete_json() over provider classes."""

    def __init__(
        self,
        provider: str = "stanford",
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(self.provider, "")
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Auto-detect Stanford API gateway
        if self.provider == "anthropic":
            stanford_key = os.environ.get("STANFORD_API_KEY")
            stanford_url = os.environ.get("STANFORD_BASE_URL")
            if stanford_key and stanford_url:
                self.provider = "stanford"

        if self.provider not in PROVIDER_MAP:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Choose from: {', '.join(PROVIDER_MAP.keys())}"
            )

        cls, env_key = PROVIDER_MAP[self.provider]
        api_key = os.environ.get(env_key)
        if not api_key:
            raise EnvironmentError(f"Set {env_key} in your environment.")

        if self.provider == "stanford":
            self._client = cls(
                api_key=api_key,
                base_url=os.environ["STANFORD_BASE_URL"],
                model=self.model,
            )
        else:
            self._client = cls(api_key=api_key, model=self.model)

        logger.info("LLMBackend initialised: provider=%s model=%s", self.provider, self.model)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single message and return the text response."""
        result = self._client.get_completion(system_prompt, user_prompt)
        if result is None:
            raise RuntimeError(f"LLM call returned None ({self.provider}/{self.model})")
        return result

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> list | dict:
        """Complete and parse the response as JSON."""
        text = self.complete(system_prompt, user_prompt, **kwargs)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)

    @property
    def token_usage(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

