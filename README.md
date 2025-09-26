# RAG Assistant (LangChain + Chroma / FAISS)

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
git clone <your-repo-url> rag-assistant
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

.Scope:
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
scoop install make
```
.Chocolatey:
```
choco install make
```
Or, run the equivalent Python commands directly, as shown below.
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
Example API response:

```json
{
    "answer": "Here’s what the provided sources establish about the differences:\n\n- What they are\n  - FAISS: A similarity search/nearest-neighbor technology used as the core search engine inside several vector databases (e.g., OpenSearch, Milvus, Vearch) and widely used in benchmarks and frameworks (Haystack, LangChain) [Wikipedia:FAISS].\n  - Chroma: An open-source vector database tailored to applications with large language models (LLMs) [Wikipedia:Chroma (database)].\n\n- Scope and role in the stack\n  - FAISS: Lower-level search/indexing component; provides algorithms and tools that vector databases can build on [Wikipedia:FAISS].\n  - Chroma: A full vector database product used in LLM/RAG tech stacks [Wikipedia:Chroma (database)].\n\n- Features highlighted\n  - FAISS: Offers k-means clustering, random-matrix rotations, PCA, data deduplication, and a standalone vector codec for lossy compression; serves typical applications like recommender systems, data mining, text retrieval, and content moderation; has demonstrated extreme scale (indexing 1.5 trillion 144-d vectors in internal Meta applications) [Wikipedia:FAISS].\n  - Chroma: Positioned for LLM-focused workloads and used in academic RAG studies; organizational details include HQ in San Francisco and a $18M seed round in April 2023 [Wikipedia:Chroma (database)].\n\n- Ecosystem and usage\n  - FAISS: Considered a baseline in similarity search benchmarks; integrated with Haystack and LangChain; used underneath other vector databases [Wikipedia:FAISS].\n  - Chroma: Used directly as a vector database within RAG stacks [Wikipedia:Chroma (database)].\n\nWhat’s missing to make the comparison complete\n- No details here on Chroma’s specific features (index types, compression, deduplication, clustering), performance, scale limits, or query capabilities.\n- No direct head-to-head performance or feature benchmarks between FAISS and Chroma.\n- No deployment, persistence, or API details for either beyond the high-level roles.\n  \nSources:\n- Wikipedia:FAISS\n- Wikipedia:Chroma (database)",
    "sources": [
        {
            "source": "Wikipedia:Chroma (database)",
            "type": "wikipedia",
            "name": null,
            "title": "Chroma (vector database)",
            "page": null,
            "score": 1.3143
        },
        {
            "source": "Wikipedia:FAISS",
            "type": "wikipedia",
            "name": null,
            "title": "FAISS",
            "page": null,
            "score": 1.3655
        },
        {
            "source": "https://app.readytensor.ai/publications/WsaE5uxLBqnH",
            "type": "url",
            "name": null,
            "title": "Technical Excellence in AI/ML Publications: An Evaluation Rubric by Ready Tensor",
            "page": null,
            "score": 1.5636
        }
    ]
}
```
---
### 6️⃣ Clean Vector Store (Optional)

If you want to reset and rebuild your index (e.g., after changing chunk size, sources, or embeddings):

With Make:
```bash
make clean
```
Without Make:
```
rm -rf vectorstore/chroma_index vectorstore/faiss_index
```
Tip: Always run make clean && make ingest after you modify configs/sources.yaml or change chunking/embedding settings.
This ensures you don't mix old and new embeddings in the same vector store.

### ✅ Why This is Important

- **Avoids stale embeddings** when sources, chunk size, or embedding model are changed  
- **Prevents index corruption** if you switch from FAISS ↔ Chroma  
- **Makes ingestion deterministic** — ensures results match new settings exactly  

---
🔄 Overriding Config Files

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

For a deeper explanation, see [`docs/architecture.md`](docs/architecture.md).


