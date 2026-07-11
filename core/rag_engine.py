"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.ai_service import BaseLLMProvider, LLMResponse, create_llm_provider
from core.config import AppConfig, load_config
from core.vector_store import RetrievedChunk, search_documents


DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class ChatMessage:
    """Minimal chat history entry used by the prompt builder."""

    role: str
    content: str


def answer_question(
    question: str,
    chat_history: Sequence[ChatMessage] | None = None,
    top_k: int = DEFAULT_TOP_K,
    config: AppConfig | None = None,
    provider: BaseLLMProvider | None = None,
) -> dict[str, object]:
    """Answer a question using only retrieved document context."""

    resolved_config = config or load_config()
    resolved_provider = provider or create_llm_provider(resolved_config)

    retrieved_chunks = search_documents(
        query=question,
        top_k=top_k,
        config=resolved_config,
    )

    if not retrieved_chunks:
        return {
            "answer": "I could not find the answer in the uploaded documents.",
            "sources": [],
            "confidence": "low",
            "refusal": True,
        }

    system_prompt = _build_system_prompt()
    prompt = _build_prompt(question=question, chat_history=chat_history, chunks=retrieved_chunks)
    llm_response = resolved_provider.generate(prompt=prompt, system_prompt=system_prompt)

    answer_text = llm_response.text.strip()
    if not answer_text:
        answer_text = "I could not find the answer in the uploaded documents."

    sources = [_chunk_to_source(chunk) for chunk in retrieved_chunks]
    answer_with_sources = _append_source_markers(answer_text, sources)

    return {
        "answer": answer_with_sources,
        "sources": sources,
        "confidence": "medium" if sources else "low",
        "refusal": False,
        "raw_response": llm_response.raw,
    }


def _build_system_prompt() -> str:
    return (
        "You are a strict retrieval-augmented assistant. "
        "Answer only from the provided document context. "
        "If the context does not contain the answer, say you could not find it in the uploaded documents. "
        "Do not use outside knowledge. Do not guess. "
        "When answering, be concise and only state claims supported by the context."
    )


def _build_prompt(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    chunks: Sequence[RetrievedChunk],
) -> str:
    history_block = _format_chat_history(chat_history)
    context_block = _format_context(chunks)

    return (
        "Conversation history:\n"
        f"{history_block}\n\n"
        "Retrieved document context:\n"
        f"{context_block}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Instructions:\n"
        "- Use only the retrieved document context.\n"
        "- If the answer is not present, say you could not find it in the uploaded documents.\n"
        "- Do not mention information that is not supported by the context.\n"
        "- Keep the answer short and direct.\n"
    )


def _format_chat_history(chat_history: Sequence[ChatMessage] | None) -> str:
    if not chat_history:
        return "No prior conversation."

    lines: list[str] = []
    for message in chat_history:
        role = message.role.strip().lower() or "user"
        content = message.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "n/a")
        section = chunk.metadata.get("section", "Document")
        lines.append(
            f"[S{index}] source={source} page={page} section={section}\n{chunk.content.strip()}"
        )
    return "\n\n".join(lines)


def _chunk_to_source(chunk: RetrievedChunk) -> dict[str, object]:
    return {
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page"),
        "section": chunk.metadata.get("section", "Document"),
        "snippet": chunk.content.strip(),
        "chunk_id": chunk.id,
        "distance": chunk.distance,
    }


def _append_source_markers(answer: str, sources: Sequence[dict[str, object]]) -> str:
    if not sources:
        return answer

    markers = [
        f"[S{index}] {source['source']} page {source['page']} section {source['section']}"
        for index, source in enumerate(sources, start=1)
    ]
    return f"{answer}\n\nSources:\n" + "\n".join(markers)
