"""LLM provider adapter layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests

from core.config import AppConfig, load_config


@dataclass(frozen=True)
class LLMResponse:
    """Standard response returned by any provider implementation."""

    text: str
    raw: Any | None = None


class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a completion for the supplied prompt."""


class OllamaProvider(BaseLLMProvider):
    """Talk to a local Ollama instance over HTTP."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(text=str(data.get("response", "")).strip(), raw=data)


class GeminiProvider(BaseLLMProvider):
    """Use the Google Gemini API for hosted inference."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")

        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key,
            },
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(text=self._extract_text(data).strip(), raw=data)

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    text_parts.append(str(text))
        return "\n".join(text_parts)


def create_llm_provider(config: AppConfig | None = None) -> BaseLLMProvider:
    """Create the active provider from environment-driven config."""

    resolved_config = config or load_config()
    provider_name = resolved_config.llm_provider.lower()

    if provider_name == "ollama":
        return OllamaProvider(
            base_url=resolved_config.ollama_base_url,
            model=resolved_config.ollama_model,
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=resolved_config.gemini_api_key,
            model=resolved_config.gemini_model,
        )

    raise ValueError(f"Unsupported LLM provider: {resolved_config.llm_provider}")