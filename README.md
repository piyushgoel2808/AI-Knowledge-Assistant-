# AI Knowledge Assistant

AI Knowledge Assistant is a simple local RAG app for uploading documents, asking questions, and getting answers grounded in the uploaded content.

The project uses a React frontend for the user interface, a FastAPI backend for upload and chat endpoints, and a Python core for ingestion, vector search, and LLM orchestration.

## Project Overview

The app lets you:

- upload PDF, DOCX, and TXT files
- index document chunks in local ChromaDB storage
- ask follow-up questions in a chat UI
- get answers with source snippets, page numbers, and section labels

The retrieval layer is designed to reduce hallucinations by forcing the model to answer only from the uploaded documents.

## Technology Stack

- Frontend: React, TypeScript, Vite
- Backend: FastAPI, Uvicorn
- Core Python libraries: ChromaDB, LangChain, PyMuPDF, python-docx, python-dotenv
- LLM providers: Gemini, Groq, Ollama
- Embeddings: SentenceTransformer embedding function through ChromaDB

## System Architecture and Data Flow

The application follows a standard modular Retrieval-Augmented Generation (RAG) architecture. The workflow is divided into two primary pipelines: Document Ingestion and Question Answering.

### Document Ingestion Pipeline

When a user uploads documents through the React frontend, the following sequence is executed:

1. **Upload Request:** The React UI sends the selected files (`.pdf`, `.docx`, `.txt`) to the FastAPI backend via the `/documents/upload` endpoint.
2. **File Storage:** The backend temporarily saves the raw files to the local directory at `data/uploads/`.
3. **Extraction and Parsing:** The core ingestion engine processes the files based on their extension. PDFs are parsed using PyMuPDF, Word documents use `python-docx`, and text files are read natively.
4. **Chunking:** The extracted text is passed through LangChain's `RecursiveCharacterTextSplitter`. This breaks the content into manageable, overlapping chunks while preserving metadata like the source name and page number.
5. **Embedding and Indexing:** The chunks are sent to the vector store manager. A SentenceTransformer model converts the text into dense vector embeddings. These embeddings and their metadata are persisted locally using ChromaDB in the `data/chroma/` directory.

### Question Answering Pipeline

When a user submits a question in the chat interface, the system processes the request as follows:

1. **Query Submission:** The React UI sends the user's question and the conversation history to the `/chat` endpoint.
2. **Vector Retrieval:** The RAG orchestrator converts the user's query into a vector using the embedding model. It queries the ChromaDB collection to retrieve the most semantically similar document chunks.
3. **Prompt Construction:** The system dynamically builds a prompt containing system instructions, the conversation history, the retrieved document chunks with source markers, and the user's current question.
4. **LLM Generation:** The prompt is routed through the adapter layer to the configured Large Language Model (Ollama, Gemini, or Groq). The LLM reads the context and generates an answer with embedded citations.
5. **Post-Processing:** The backend parses the LLM's response to extract cited chunk IDs and maps the exact sentences from the source material used to formulate the answer.
6. **Response Delivery:** The final response, including the generated answer and a structured list of source snippets, is sent back to the frontend for display.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AI-Knowledge-Assistant-
```

### 2. Set up the Python environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up the frontend

Use Node.js 18 or newer.

```bash
cd frontend
npm install
```

### 4. Configure environment variables

Copy the example file and fill in your local values.

```bash
copy .env.example .env
copy frontend\.env.example frontend\.env
```

Important values:

- `LLM_PROVIDER=gemini` or `groq` or `ollama`
- `GEMINI_API_KEY` for Gemini
- `GROQ_API_KEY` for Groq
- `GEMINI_MODEL=gemini-flash-latest`
- `GROQ_MODEL=llama-3.3-70b-versatile`
- `CHROMA_PERSIST_DIR=data/chroma`
- `UPLOAD_DIR=data/uploads`
- `VITE_API_URL=http://localhost:8000`

## How to Run the Application

Open two terminals.

### Terminal 1: start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### Terminal 2: start the frontend

```bash
cd frontend
npm run dev
```

Then open the frontend URL shown by Vite, usually `http://localhost:5173`.

## How to Use It

1. Open the frontend in your browser.
2. Upload one or more supported files.
3. Click the upload/index button.
4. Ask a question in the chat box.
5. Expand the source section to review the exact snippets used in the answer.

## Key Design Decisions

- The UI is intentionally simple so the document workflow is easy to understand in an interview setting.
- Business logic stays in the Python core, while the frontend only handles presentation and API calls.
- ChromaDB is used locally with persistent storage so the app works without a cloud vector database.
- The provider layer uses an adapter pattern so the model can be switched without rewriting the rest of the app.
- Answers are grounded in retrieved chunks and the response includes source metadata for traceability.

## Assumptions

- The app is meant to run locally.
- A valid LLM provider key or local Ollama endpoint is available before asking questions.
- Documents are stored on disk in the configured upload folder and indexed into local ChromaDB.
- The frontend and backend run on `localhost` during development.

## Known Limitations

- This is a local single-user app, not a multi-user production system.
- Answer quality depends on the uploaded documents and the selected model.
- The citation extraction is based on retrieved chunks and prompt output, so it is helpful but not perfect.
- Scanned PDFs may require local OCR support, and OCR quality depends on the installed Tesseract setup.
- DOCX and TXT support is basic and does not include advanced document structure parsing.
- There is no authentication, access control, or document management UI beyond upload, clear, and chat.
- No Control on Token Usage Management 

## Future Improvements

- Adding authentication and per-user document isolation.
- Improving citation formatting and make source highlighting more precise.
- Add better PDF table extraction and richer DOCX structure parsing.
- Adding document deletion and re-indexing controls in the UI.
- Adding a complete Docker setup for both frontend and backend.

## Repository Structure

```text
backend/      FastAPI app and API routes
core/         ingestion, vector store, RAG engine, and LLM provider adapters
frontend/     React frontend built with Vite
data/         local uploads and ChromaDB persistence
```
