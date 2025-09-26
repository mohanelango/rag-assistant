# Architecture

**Ingestion Pipeline**
- Loaders: WebBaseLoader, WikipediaLoader, PyPDFLoader
- Cleaning & normalization (strip nav/boilerplate, collapse whitespace)
- Chunking: RecursiveCharacterTextSplitter (1000/150)
- Embeddings: sentence-transformers (configurable)
- Vector DB: Chroma (default) or FAISS (switch in `settings.yaml`)

**RAG Chain**
- Retriever: `vectorstore.as_retriever(search_kwargs={"k": k})`
- Prompt: concise system instruction + context stuffing + question
- LLM: OpenAI / Ollama / HuggingFace (config-driven, hard fallback order)
- Output: answer + source attributions (Source + Type + Name + Title + Page + Score)

**Interfaces**
- CLI (`src/rag/cli.py`)
- HTTP API (`src/rag/api.py`)

**Non-Goals**
- Not a full UI
- Not streaming to clients (kept simple; add later if needed)
