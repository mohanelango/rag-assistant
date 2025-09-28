# RAG Assistant (LangChain + Chroma / FAISS)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)


> **TL;DR Workflow:**  
> ```
>  [User Query] → [Retriever: Chroma] → [Context + Prompt] → [LLM] → [Answer + Sources]
> ```
> Minimal, production-ready RAG pipeline using Chroma + LangChain.

A production-ready **Retrieval-Augmented Generation (RAG)** system that:

- Uses **Chroma** as the default vector store (persistent and easy to inspect)
- Embeds and indexes a corpus (Web urls + Wikipedia topics + PDFs)
- Exposes a functional **LangChain** pipeline: User Question → Retrieval (Chroma) → Prompt Construction (Context + Question) → LLM Response
- Provides both a CLI and REST API interface

---
##  Why This Project?

I built this project to explore how production-grade RAG systems are structured.  
It demonstrates:
- **Modular architecture** (ingestion pipeline, retrieval layer, generation layer)
- **LangChain integration** with pluggable vector stores (Chroma, FAISS)
- **Robustness features** like staleness detection and deterministic ingestion
- **Deployment-readiness** with CLI, API, and Makefile automation

This project serves as a portfolio piece to showcase my ability to design scalable, maintainable AI/ML systems end-to-end.

---
## Tech Stack

- **Language:** Python 3.10+
- **LLM Orchestration:** LangChain
- **Vector Stores:** Chroma (default), FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **API:** FastAPI
- **Testing:** pytest
- **Automation:** Makefile targets for ingestion, queries, cleaning, testing
---
## 📌 Quickstart

### 1️⃣ Prerequisites
- **Python** 3.10+
- (Optional) **Docker**
- One of the supported LLM backends:
  - **OpenAI** (requires `OPENAI_API_KEY` in `.env`)
  - **Ollama** (e.g., `llama3` running locally)
  - **HuggingFace Inference API** (requires `HUGGINGFACEHUB_API_TOKEN`)

---

### 2️⃣ Setup
```bash
git clone <https://github.com/mohanelango/rag-assistant.git> rag-assistant
cd rag-assistant
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to select your preferred LLM provider (OpenAI / Ollama / HF).  
Optionally tune `configs/settings.yaml` for chunk size, retrieval `k`, or model parameters.

---
💻 Global Note on Commands (Cross-Platform)

This project provides a Makefile with shortcuts (make ingest, make ask, etc.).

Linux / macOS – Works out-of-the-box if you have GNU Make.

Windows – If make is not available, install it via:

<details> <summary> Install GNU Make on Windows</summary>

Scoop
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install make

```
Chocolatey:
```
choco install make
```

Or, run the equivalent Python commands directly, as shown below.
</details>

### 3️⃣ Ingest Your Corpus
With Make:
```bash
make ingest
```
Without Make (Windows fallback):
```
python -m src.rag.ingest
```
This will:
- Load all sources from `configs/sources.yaml`
- Chunk them into 1000-character blocks (150 overlap)
- Embed with `sentence-transformers/all-MiniLM-L6-v2`
- Build and persist a **Chroma vector store** at `./vectorstore/chroma_index/`

Smart Staleness Detection:

If you modify configs/sources.yaml or change embedding/chunking settings but forget to re-ingest,
the CLI and API will detect that your sources are newer than the last index build and display a warning:

⚠️ WARNING: sources.yaml was modified after the last ingestion.
Run `make clean && make ingest` to update your index before querying.

---

### 4️⃣ Query the System (CLI):
With Make:
```bash
make ask Q="What is Retrieval-Augmented Generation and why is it useful?"
```
Without Make (Windows fallback):
```
python -m src.rag.cli --question "What is Retrieval-Augmented Generation and why is it useful?"
```

This prints:
- A context-aware generated answer
- A deduplicated list of sources from the Chroma index (with similarity scores)
> **ℹ️ Metadata Rich Output:**  
> Not just the source, but also:
> - `type` (url, wikipedia, pdf)
> - `name` (for PDFs, file name)
> - `page` (for PDFs, page number)
> - `title` (for Wikipedia, page title)
> - `score` (similarity score)
> 
> This provides traceability and helps reviewers verify that answers are well-grounded.

Sample CLI Output:
![CLI Example](docs/screenshots/cli_output.png)
---

### 5️⃣ Serve via API
With Make:
```bash
make serve
```
Without Make (Windows fallback):
```
uvicorn src.rag.api:app --reload --port 8000
```
Query with:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain FAISS vs Chroma differences"}'
```

Sample API Output:
![CLI Example](docs/screenshots/api_output.png)
---
### 6️⃣ Clean Vector Store (Optional)

If you want to reset and rebuild your index (e.g., after changing chunk size, sources, or embeddings):

With Make:
```bash
make clean
```
Clean a custom vectorstore (if you used VECTORSTORE_DIR):
```bash
make clean VECTORSTORE_DIR=./vectorstore/test_index
```
Without Make:
```bash
rm -rf vectorstore/chroma_index vectorstore/faiss_index
```
Tip: Always run make clean && make ingest after you modify configs/sources.yaml or change chunking/embedding settings.
This ensures you don't mix old and new embeddings in the same vector store.

### Why This is Important

- **Avoids stale embeddings** when sources, chunk size, or embedding model are changed  
- **Prevents index corruption** if you switch from FAISS ↔ Chroma  
- **Makes ingestion deterministic** — ensures results match new settings exactly  

---
### Testing
Run the test suite to validate ingestion, retrieval, and API endpoints:
```bash
make test
```
Run tests against a custom vectorstore directory (recommended for CI isolation):
```bash
make test VECTORSTORE_DIR=./vectorstore/test_index
```
---
Overriding Config Files

You can switch to alternate configs (e.g., settings.prod.yaml, sources.alt.yaml) without changing code:
```
make ingest SETTINGS=configs/settings.prod.yaml SOURCES=configs/sources.alt.yaml
make ask Q="What is RAG?" SETTINGS=configs/settings.prod.yaml
```
Or with Python directly:
```
python -m src.rag.ingest --settings configs/settings.prod.yaml --sources configs/sources.alt.yaml
python -m src.rag.cli --question "What is RAG?" --settings configs/settings.prod.yaml

```

---
## TL;DR: How It Works
---
User Question
   │
   ▼
Retriever (Chroma DB)
   │
   ▼
Context + Question → Prompt Template
   │
   ▼
LLM (OpenAI / Ollama / HF)
   │
   ▼
Generated Answer + Source List

This ASCII diagram is a high-level view of the full RAG flow.
For a detailed view, see [docs/architecture.md](docs/architecture.md)
___
## ⚙️ Configs

- `configs/settings.yaml` — retrieval settings, embedding model, Chroma persist directory  
- `configs/sources.yaml` — URLs (Ready Tensor guides), Wikipedia topics, PDFs

---

## Architecture Overview

```mermaid
flowchart TD

subgraph Ingestion["Ingestion Pipeline"]
    A[Sources: Ready Tensor Guides + Wikipedia + PDFs] --> B[Loaders: URL + WikipediaLoader + PDFs]
    B --> C[Clean & Normalize Text]
    C --> D[Chunking: RecursiveCharacterTextSplitter]
    D --> E[Embeddings: sentence-transformers/all-MiniLM-L6-v2]
    E --> F[Vector Store: Chroma Persisted DB]
end

subgraph Retrieval["Query-Time Retrieval"]
    Q[User Query] --> R[Retriever (Chroma)]
    R --> CONTEXT[Retrieved Chunks]
end

subgraph Generation["Answer Generation"]
    CONTEXT --> PROMPT[Prompt Template (System + Context)]
    PROMPT --> LLM[LLM: OpenAI / Ollama / HF]
    LLM --> ANSWER[Generated Response + Sources]
end

F --> R
```

For a deeper explanation, see [`docs/architecture.md`](docs/architecture.md)

---
##  License

Distributed under the [MIT License](LICENSE).

