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

        import google.generativeai as genai

        self._genai = genai
        self._genai.configure(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        generation_config: dict[str, Any] = {}
        if system_prompt:
            generation_config["system_instruction"] = system_prompt

        model = self._genai.GenerativeModel(self.model, generation_config=generation_config or None)
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or ""
        return LLMResponse(text=text.strip(), raw=response)


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
