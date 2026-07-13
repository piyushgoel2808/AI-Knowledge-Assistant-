"""FastAPI backend for the React frontend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import load_config
from core.ingestion import ingest_file, is_supported_file
from core.rag_engine import ChatMessage, answer_question
from core.vector_store import add_documents, clear_collection, count_documents


app = FastAPI(title="AI Knowledge Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, Any]:
    config = load_config()
    return {
        "status": "ok",
        "provider": config.llm_provider,
        "stored_chunks": count_documents(config=config),
    }


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    config = load_config()
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    upload_dir = config.upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    accepted_files: list[str] = []
    all_documents = []

    for uploaded_file in files:
        filename = uploaded_file.filename or ""
        if not is_supported_file(filename):
            continue

        target_path = upload_dir / Path(filename).name
        with target_path.open("wb") as target_file:
            shutil.copyfileobj(uploaded_file.file, target_file)

        accepted_files.append(target_path.name)
        all_documents.extend(ingest_file(target_path))

    if not all_documents:
        raise HTTPException(status_code=400, detail="No supported documents were uploaded.")

    add_documents(all_documents, config=config)

    return {
        "message": "Documents uploaded and indexed.",
        "files": accepted_files,
        "chunks_indexed": len(all_documents),
        "stored_chunks": count_documents(config=config),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    config = load_config()
    chat_history = [
        ChatMessage(role=item.get("role", "user"), content=item.get("content", ""))
        for item in request.history
    ]

    return answer_question(
        question=request.question,
        chat_history=chat_history,
        config=config,
    )


@app.post("/documents/clear")
def clear_documents() -> dict[str, Any]:
    config = load_config()
    clear_collection(config=config)
    return {"message": "Vector store cleared."}
