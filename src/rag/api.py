"""
FastAPI REST API exposing the RAG assistant.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from .utils import load_yaml, get_unique_sources, warn_if_stale
from .chain import build_rag_chain

DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"

# Load settings and check freshness once at startup
warn_if_stale(DEFAULT_SETTINGS_PATH, DEFAULT_SOURCES_PATH)

app = FastAPI(title="RAG Assistant API")


class AskRequest(BaseModel):
    question: str


SETTINGS = load_yaml(DEFAULT_SETTINGS_PATH)
EMBED = HuggingFaceEmbeddings(model_name=SETTINGS["embeddings"]["model_name"])
CHAIN, RETRIEVER = build_rag_chain(SETTINGS, EMBED)


@app.post("/ask")
def ask(req: AskRequest):
    """
    Handle a question request and return answer + sources with rich metadata.
    """
    try:
        res = CHAIN.invoke({"question": req.question})
        answer = getattr(res, "content", str(res))

        docs_and_scores = RETRIEVER.vectorstore.similarity_search_with_score(
            req.question, k=SETTINGS["retrieval"]["k"]
        )

        unique_sources = get_unique_sources(docs_and_scores)

        return {"answer": answer, "sources": unique_sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
