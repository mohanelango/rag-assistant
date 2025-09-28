"""
Ingestion pipeline entrypoint.

- Loads sources from configs
- Cleans and chunks documents
- Embeds them and builds Chroma index (default)
"""

import argparse
import os

from langchain_community.vectorstores import FAISS
from .utils import load_yaml, ensure_dir
from .loaders import load_from_urls, load_from_wikipedia, load_all_sources
from .vectorstore import build_embeddings, create_or_load_vectorstore
from .utils import chunk_docs
from src.logging_config import get_logger
DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"
logger = get_logger(__name__)


def main(settings_path: str = DEFAULT_SETTINGS_PATH, sources_path: str = DEFAULT_SOURCES_PATH,
         vectorstore_override: str | None = None):
    """
    Main ingestion routine: load → clean → chunk → embed → index.

    Args:
        settings_path (str): Path to settings.yaml. Defaults to configs/settings.yaml.
        sources_path (str): Path to sources.yaml. Defaults to configs/sources.yaml.
        vectorstore_override (str): Optional override for vectorstore/chroma_dir/faiss_dir.
    """
    settings = load_yaml(settings_path)
    sources = load_yaml(sources_path)

    vs_type = settings["vectorstore"]["type"]
    faiss_dir = vectorstore_override or os.getenv(
        "VECTORSTORE_DIR", settings["vectorstore"]["persist_dir"]
    )
    chroma_dir = vectorstore_override or os.getenv(
        "VECTORSTORE_DIR", settings["vectorstore"]["chroma_dir"]
    )
    model_name = settings["embeddings"]["model_name"]
    chunk_size = settings["chunking"]["chunk_size"]
    overlap = settings["chunking"]["chunk_overlap"]
    logger.info(f"Loading sources from {sources_path}")
    docs = load_all_sources(sources)
    logger.info(f"Loaded {len(docs)} documents")
    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=overlap)
    logger.info(f"Chunked into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

    embeddings = build_embeddings(model_name)
    logger.info(f"Built embeddings with model: {model_name}")

    if vs_type == "faiss":
        ensure_dir(faiss_dir)
        vs = create_or_load_vectorstore("faiss", faiss_dir, chroma_dir, embeddings, chunks=chunks)
        FAISS.save_local(vs, folder_path=faiss_dir)
        logger.info(f"FAISS index saved to {faiss_dir}")
    else:
        vs = create_or_load_vectorstore("chroma", faiss_dir, chroma_dir, embeddings, chunks=chunks)
        logger.info(f"Chroma index persisted to {chroma_dir}")
        try:
            vs._client.persist()
            vs._client.close()
        except AttributeError:
            logger.warning("Could not close Chroma client (may not be present)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and index documents for RAG")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH, help="Path to settings.yaml")
    parser.add_argument("--sources", default=DEFAULT_SOURCES_PATH, help="Path to sources.yaml")
    parser.add_argument(
        "--vectorstore", default=None, help="Override vectorstore directory (useful for tests/CI)"
    )
    args = parser.parse_args()
    main(args.settings, args.sources, vectorstore_override=args.vectorstore)
