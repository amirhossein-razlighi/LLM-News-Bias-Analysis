from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class OllamaGeneration:
    text: str
    raw_payload: dict
    latency_ms: int


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def list_models(self, timeout_seconds: int = 10) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return [m["name"] for m in payload.get("models", []) if "name" in m]

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout_seconds: int = 60,
        retries: int = 2,
        think: bool | None = None,
    ) -> OllamaGeneration:
        total_predict = max_tokens * 4 if think else max_tokens
        generate_payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": total_predict,
            },
        }
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": total_predict,
            },
        }
        compat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": total_predict,
        }

        # Qwen3 and other reasoning models can emit large "thinking" traces that
        # consume the token budget and truncate the actual answer. For controlled
        # evaluation runs, disable thinking explicitly when configured.
        if think is not None:
            generate_payload["think"] = think
            chat_payload["think"] = think

        endpoint_requests = [
            ("/api/generate", generate_payload),
            ("/api/chat", chat_payload),
            ("/v1/chat/completions", compat_payload),
        ]

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            for endpoint, request_payload in endpoint_requests:
                try:
                    started = time.perf_counter()
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json=request_payload,
                        timeout=timeout_seconds,
                    )

                    if response.status_code == 404:
                        endpoint_hint = self._build_404_hint(response, model)
                        if endpoint_hint is not None:
                            raise RuntimeError(endpoint_hint)
                        # Endpoint missing in this Ollama variant; try the next endpoint.
                        continue

                    response.raise_for_status()
                    payload = response.json()
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    text = self._extract_text_from_payload(endpoint, payload)
                    return OllamaGeneration(text=text, raw_payload=payload, latency_ms=latency_ms)
                except Exception as exc:  # pragma: no cover - error path tested via retries
                    last_error = exc

            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")

    def _extract_text_from_payload(self, endpoint: str, payload: dict[str, Any]) -> str:
        if endpoint == "/api/generate":
            return str(payload.get("response", ""))
        if endpoint == "/api/chat":
            message = payload.get("message") or {}
            return str(message.get("content", ""))
        if endpoint == "/v1/chat/completions":
            choices = payload.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                return str(message.get("content", ""))
            return ""
        return ""

    def _build_404_hint(self, response: requests.Response, model: str) -> str | None:
        try:
            payload = response.json()
            raw_error = payload.get("error")
            error_text = str(raw_error).lower() if raw_error is not None else ""
        except Exception:
            payload = None
            error_text = (response.text or "").lower()

        if "model" in error_text and "not found" in error_text:
            installed_hint = ""
            try:
                installed = self.list_models()
                installed_hint = f" Installed models: {installed}."
            except Exception:
                pass
            return (
                f"Model '{model}' was not found in local Ollama. "
                f"Run 'ollama pull {model}' and retry.{installed_hint}"
            )

        # Returning None means this was likely an endpoint mismatch, so caller should try fallback endpoints.
        if "not found" in error_text and "model" not in error_text:
            return None

        if payload is not None and payload.get("error"):
            return f"Ollama returned 404: {payload.get('error')}"

        return None
