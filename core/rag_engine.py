"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from core.ai_service import BaseLLMProvider, LLMResponse, create_llm_provider
from core.config import AppConfig, load_config
from core.vector_store import RetrievedChunk, search_documents, count_documents


@dataclass(frozen=True)
class ChatMessage:
    """Minimal chat history entry used by the prompt builder."""

    role: str
    content: str


def answer_question(
    question: str,
    chat_history: Sequence[ChatMessage] | None = None,
    top_k: int | None = None,
    config: AppConfig | None = None,
    provider: BaseLLMProvider | None = None,
) -> dict[str, object]:
    """Answer a question using only retrieved document context."""

    resolved_config = config or load_config()
    resolved_provider = provider or create_llm_provider(resolved_config)

    # DYNAMIC RETRIEVEL: Check total database chunk count dynamically
    total_indexed_chunks = count_documents(config=resolved_config)
    
    # Fallback to at least 4 chunks, or use the maximum available chunks in the DB
    search_limit = max(4, total_indexed_chunks) if top_k is None else top_k

    retrieved_chunks = search_documents(
        query=question,
        top_k=search_limit,
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

    # Parse citations like [S1], [S2] even if combined like [S1, S2]
    cited_indices = set()
    for match in re.findall(r"\[(.*?)\]", answer_text):
        numbers = re.findall(r"\d+", match)
        cited_indices.update(int(n) for n in numbers)

    sources = []
    # ONLY build sources if the AI actually cited something
    if cited_indices:
        for index, chunk in enumerate(retrieved_chunks, start=1):
            if index in cited_indices:
                source_dict = _chunk_to_source(chunk)
                source_dict["cite_index"] = index
                
                # HIGHLIGHTING LOGIC: Find exactly matching phrases
                source_dict["snippet"] = _extract_and_highlight_match(
                    chunk_text=chunk.content.strip(), 
                    answer_text=answer_text
                )
                sources.append(source_dict)

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": "medium" if sources else "low",
        "refusal": False,
        "raw_response": llm_response.raw,
    }


def _extract_and_highlight_match(chunk_text: str, answer_text: str) -> str:
    """Finds overlapping sentences between the document chunk and the AI answer, 
    returning only the relevant sentences with the matched text highlighted.
    """
    # Clean up the AI text to look for keywords (removes formatting and citations)
    clean_answer = re.sub(r"\[.*?\]", "", answer_text).lower()
    
    # Split the original chunk text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
    matched_sentences = []

    for sentence in sentences:
        # Split sentence into words to find keywords (filtering short words out)
        words = [w.strip(",.()\"'").lower() for w in sentence.split() if len(w) > 4]
        
        # Check if multiple meaningful words from this sentence appear in the AI's answer
        match_count = sum(1 for word in words if word in clean_answer)
        
        if match_count >= 2:  # Found a solid match!
            # Highlight keywords inside the sentence using markdown bold tags
            highlighted_sentence = sentence
            for word in set(w for w in sentence.split() if len(w) > 4 and w.lower().strip(",.()\"'") in clean_answer):
                # Clean up word from punctuation to protect regex characters
                safe_word = re.escape(word.strip(",.()\"'"))
                if safe_word:
                    highlighted_sentence = re.sub(
                        rf"\b({safe_word})\b", 
                        r"**\1**", 
                        highlighted_sentence, 
                        flags=re.IGNORECASE
                    )
            matched_sentences.append(highlighted_sentence)

    # Fallback: if sentence parsing failed to map, return the original chunk snippet
    return "\n... ".join(matched_sentences) if matched_sentences else chunk_text


def _build_system_prompt() -> str:
    return (
        "You are a strict retrieval-augmented assistant. "
        "Answer only from the provided document context. "
        "If the context does not contain the answer, say you could not find it in the uploaded documents. "
        "Do not use outside knowledge. Do not guess. "
        "When answering, be concise and only state claims supported by the context. "
        "IMPORTANT: You MUST cite the specific source context you used by appending its exact marker (e.g., [S1], [S2]) to the end of the relevant sentence or clause."
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