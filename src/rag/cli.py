"""
Command-line interface for querying the RAG system.

Provides:
- Clean question → retrieval → generation flow
- Deduplicated source list with similarity scores
- Staleness check: warns if sources.yaml is newer than vectorstore
"""

import argparse
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.query_processing import process_query
from .utils import load_yaml, get_unique_sources
from .chain import build_rag_chain
from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"


def warn_if_stale(settings_path: str, sources_path: str):
    """
    Warn if sources.yaml is newer than vectorstore (Chroma or FAISS).
    """
    settings = load_yaml(settings_path)
    vs_type = settings["vectorstore"]["type"]
    chroma_dir = settings["vectorstore"]["chroma_dir"]
    faiss_dir = settings["vectorstore"]["persist_dir"]

    try:
        sources_mtime = Path(sources_path).stat().st_mtime
    except FileNotFoundError:
        logger.warning(f"Sources file {sources_path} not found, skipping staleness check.")
        return

    index_mtime = None
    if vs_type == "chroma" and Path(chroma_dir).exists():
        index_files = list(Path(chroma_dir).rglob("*"))
        if index_files:
            index_mtime = max(f.stat().st_mtime for f in index_files)
    elif vs_type == "faiss" and Path(faiss_dir).exists():
        index_files = list(Path(faiss_dir).rglob("*"))
        if index_files:
            index_mtime = max(f.stat().st_mtime for f in index_files)

    if index_mtime and sources_mtime > index_mtime:
        logger.warning("⚠️ sources.yaml was modified after the last ingestion.")
        logger.warning("Run `make clean && make ingest` to update your index before querying.")


def main(question: str, settings_path: str = DEFAULT_SETTINGS_PATH, sources_path: str = DEFAULT_SOURCES_PATH):
    """
    Query the RAG pipeline from the CLI.
    """
    logger.info("Starting CLI query...")
    warn_if_stale(settings_path, sources_path)

    # Load settings and embeddings
    logger.info(f"Loading settings from {settings_path}")
    settings = load_yaml(settings_path)
    logger.debug(f"Settings loaded: {settings}")

    logger.info(f"Initializing embeddings model: {settings['embeddings']['model_name']}")
    embeddings = HuggingFaceEmbeddings(model_name=settings["embeddings"]["model_name"])

    logger.info("Building RAG chain and retriever...")
    chain, retriever = build_rag_chain(settings, embeddings)

    # --- Query Preprocessing ---
    logger.info("Preprocessing user query...")
    expand = settings.get("query", {}).get("expand", False)
    qp = process_query(question, settings, expand=expand)

    logger.info(f"Query type: {qp['type']} | Cleaned: {qp['cleaned']} | Keywords: {qp['keywords']}")
    processed_question = qp["expanded"]

    # --- Run RAG Pipeline ---
    logger.info(f"Invoking chain with processed query: {processed_question}")
    result = chain.invoke({"question": processed_question})
    answer_text = getattr(result, "content", str(result))
    logger.debug(f"LLM answer generated (length: {len(answer_text)} characters)")

    # Retrieve docs with similarity scores
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(
        processed_question, k=settings["retrieval"]["k"]
    )
    logger.info(f"Retrieved {len(docs_and_scores)} document chunks for context")

    # Deduplicate sources
    unique_sources = get_unique_sources(docs_and_scores)
    logger.info(f"Deduplicated sources count: {len(unique_sources)}")

    print("\n=== ANSWER ===\n")
    print(answer_text.strip())
    print("\n=== SOURCES (deduplicated, with metadata) ===")
    print(json.dumps(unique_sources, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions to the RAG assistant")
    parser.add_argument("--question", "-q", required=True, help="The question to ask")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH, help="Path to settings.yaml")
    parser.add_argument("--sources", default=DEFAULT_SOURCES_PATH, help="Path to sources.yaml")
    args = parser.parse_args()

    logger.info(f"Received CLI question: {args.question}")
    try:
        main(args.question, args.settings, args.sources)
        logger.info("Query completed successfully")
    except Exception as e:
        logger.exception("An error occurred while processing the CLI query")
        raise
