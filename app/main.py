"""Streamlit UI for the AI Knowledge Assistant."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.ai_service import create_llm_provider
from core.config import load_config
from core.ingestion import ingest_file, is_supported_file
from core.rag_engine import ChatMessage, answer_question
from core.vector_store import add_documents, clear_collection, count_documents


APP_TITLE = "AI Knowledge Assistant"


def main() -> None:
    """Render the Streamlit app."""

    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    config = load_config()
    _initialize_state()

    st.title(APP_TITLE)
    st.caption("Upload PDFs, DOCX files, or TXT notes, then ask questions grounded in those documents.")

    _render_sidebar(config)

    _render_chat_area(config)


def _initialize_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "ingested_file_names" not in st.session_state:
        st.session_state.ingested_file_names = set()


def _render_sidebar(config) -> None:
    with st.sidebar:
        st.header("Documents")
        st.write(f"Stored chunks: {count_documents(config=config)}")
        st.write(f"Active provider: {config.llm_provider}")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        col_ingest, col_clear = st.columns(2)
        with col_ingest:
            ingest_clicked = st.button("Ingest", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("Clear", use_container_width=True)

        if ingest_clicked:
            _ingest_uploaded_files(config)
            st.rerun()

        if clear_clicked:
            clear_collection(config=config)
            st.session_state.chat_history = []
            st.session_state.ingested_file_names = set()
            st.success("Vector store cleared.")

        st.divider()
        st.subheader("Conversation")
        if st.button("Reset chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def _ingest_uploaded_files(config) -> None:
    if not st.session_state.uploaded_files:
        st.warning("Upload at least one supported file before ingesting.")
        return

    saved_documents = []
    upload_dir = config.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in st.session_state.uploaded_files:
        if not is_supported_file(uploaded_file.name):
            continue

        target_path = upload_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_documents.extend(ingest_file(target_path))
        st.session_state.ingested_file_names.add(uploaded_file.name)

    if not saved_documents:
        st.warning("No supported documents were ingested.")
        return

    add_documents(saved_documents, config=config)
    st.success(f"Ingested {len(saved_documents)} chunks from {len(st.session_state.ingested_file_names)} file(s).")


def _render_chat_area(config) -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    if prompt := st.chat_input("Ask a question about the uploaded documents"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        provider = create_llm_provider(config)
        history = [ChatMessage(role=item["role"], content=item["content"]) for item in st.session_state.chat_history[:-1]]
        result = answer_question(
            question=prompt,
            chat_history=history,
            config=config,
            provider=provider,
        )

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
        st.session_state.chat_history.append(assistant_message)

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            _render_sources(result["sources"])


def _render_sources(sources) -> None:
    with st.expander("Sources", expanded=False):
        for index, source in enumerate(sources, start=1):
            source_name = source.get("source", "unknown")
            page = source.get("page", "n/a")
            section = source.get("section", "Document")
            snippet = source.get("snippet", "")

            st.markdown(f"**{index}. {source_name}**")
            st.write(f"Page: {page} | Section: {section}")
            st.code(snippet, language="text")


if __name__ == "__main__":
    main()
