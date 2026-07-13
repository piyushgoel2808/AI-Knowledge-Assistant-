"""Application configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings loaded from environment variables."""

    llm_provider: str
    ollama_base_url: str
    ollama_model: str
    gemini_api_key: str
    gemini_model: str
    # Added fields for Groq
    groq_api_key: str
    groq_model: str
    chroma_persist_dir: Path
    upload_dir: Path


def load_config() -> AppConfig:
    """Load config values with sensible local defaults."""

    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip(),
        # Added mapping for Groq environment variables
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")).expanduser(),
        upload_dir=Path(os.getenv("UPLOAD_DIR", "data/uploads")).expanduser(),
    )