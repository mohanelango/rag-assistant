"""
FastAPI REST API exposing the RAG assistant.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from .utils import load_yaml, get_unique_sources, warn_if_stale
from .chain import build_rag_chain
from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"


logger.info("Starting RAG Assistant API service...")
# Load settings and check freshness once at startup
warn_if_stale(DEFAULT_SETTINGS_PATH, DEFAULT_SOURCES_PATH)

app = FastAPI(title="RAG Assistant API")


class AskRequest(BaseModel):
    question: str


try:
    SETTINGS = load_yaml(DEFAULT_SETTINGS_PATH)
    EMBED = HuggingFaceEmbeddings(model_name=SETTINGS["embeddings"]["model_name"])
    CHAIN, RETRIEVER = build_rag_chain(SETTINGS, EMBED)
    logger.info(f"RAG chain initialized with model: {SETTINGS['embeddings']['model_name']}")
except Exception as e:
    logger.exception("Failed to initialize RAG chain during startup")
    raise


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request at {request.url.path}")
    response = await call_next(request)
    logger.info(f"Completed {request.method} {request.url.path} with status {response.status_code}")
    return response


@app.post("/ask")
def ask(req: AskRequest):
    """
    Handle a question request and return answer + sources with rich metadata.
    """
    logger.info(f"Received API question: {req.question}")
    try:
        res = CHAIN.invoke({"question": req.question})
        answer = getattr(res, "content", str(res))
        logger.debug("LLM response generated successfully")

        docs_and_scores = RETRIEVER.vectorstore.similarity_search_with_score(
            req.question, k=SETTINGS["retrieval"]["k"]
        )
        logger.debug(f"Retrieved {len(docs_and_scores)} documents from vectorstore")

        unique_sources = get_unique_sources(docs_and_scores)
        logger.info(f"Returning answer with {len(unique_sources)} unique sources")
        return {"answer": answer, "sources": unique_sources}

    except Exception as e:
        logger.exception("Error while processing API request")
        raise HTTPException(status_code=500, detail="Internal server error")
