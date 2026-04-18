from __future__ import annotations

import os
import time
from typing import Any

import litellm

from app.models.ollama_client import OllamaGeneration

# Silence litellm's verbose logging by default.
litellm.suppress_debug_info = True

# Provider → list of commonly available models.
PROVIDER_MODELS: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "openai/o4-mini",
    ],
    "anthropic": [
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3.5-sonnet-20241022",
        "anthropic/claude-3-haiku-20240307",
    ],
    "google": [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
    ],
}

# Maps provider key → environment variable that litellm reads.
_PROVIDER_ENV_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
}


class LiteLLMClient:
    """Thin wrapper around *litellm.completion* that returns the same
    ``OllamaGeneration`` dataclass used by ``OllamaClient`` so that all
    downstream parsing / decision logic stays unchanged."""

    def __init__(self, api_keys: dict[str, str] | None = None) -> None:
        self._api_keys: dict[str, str] = {}
        if api_keys:
            for provider, key in api_keys.items():
                if key and key.strip():
                    self._api_keys[provider.lower()] = key.strip()

    def _inject_env_keys(self) -> None:
        """Push stored keys into environment variables so litellm picks them up."""
        for provider, key in self._api_keys.items():
            env_var = _PROVIDER_ENV_KEYS.get(provider)
            if env_var and key:
                os.environ[env_var] = key

    def list_models(self, provider: str | None = None) -> list[str]:
        """Return known model identifiers, optionally filtered to *provider*."""
        if provider:
            provider = provider.lower()
            if provider not in self._api_keys:
                return []
            return list(PROVIDER_MODELS.get(provider, []))
        models: list[str] = []
        for prov in self._api_keys:
            models.extend(PROVIDER_MODELS.get(prov, []))
        return models

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout_seconds: int = 60,
        retries: int = 2,
        think: bool | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> OllamaGeneration:
        self._inject_env_keys()

        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_seconds,
            "num_retries": retries,
        }

        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                started = time.perf_counter()
                response = litellm.completion(**kwargs)
                latency_ms = int((time.perf_counter() - started) * 1000)

                text = response.choices[0].message.content or ""
                raw_payload = response.model_dump() if hasattr(response, "model_dump") else {}

                return OllamaGeneration(
                    text=text,
                    raw_payload=raw_payload,
                    latency_ms=latency_ms,
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"LiteLLM generation failed after retries: {last_error}")
