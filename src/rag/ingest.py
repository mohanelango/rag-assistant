"""
Ingestion pipeline entrypoint.

- Loads sources from configs
- Cleans and chunks documents
- Embeds them and builds Chroma index (default)
"""

import argparse
from langchain_community.vectorstores import FAISS
from .utils import load_yaml, ensure_dir
from .loaders import load_from_urls, load_from_wikipedia, load_all_sources
from .vectorstore import build_embeddings, create_or_load_vectorstore
from .utils import chunk_docs

DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"


def main(settings_path: str = DEFAULT_SETTINGS_PATH, sources_path: str = DEFAULT_SOURCES_PATH):
    """
    Main ingestion routine: load → clean → chunk → embed → index.

    Args:
        settings_path (str): Path to settings.yaml. Defaults to configs/settings.yaml.
        sources_path (str): Path to sources.yaml. Defaults to configs/sources.yaml.
    """
    settings = load_yaml(settings_path)
    sources = load_yaml(sources_path)

    vs_type = settings["vectorstore"]["type"]
    faiss_dir = settings["vectorstore"]["persist_dir"]
    chroma_dir = settings["vectorstore"]["chroma_dir"]

    model_name = settings["embeddings"]["model_name"]
    chunk_size = settings["chunking"]["chunk_size"]
    overlap = settings["chunking"]["chunk_overlap"]

    docs = load_all_sources(sources)
    print(f"[ingest] loaded {len(docs)} raw docs")
    chunks = chunk_docs(docs, chunk_size=chunk_size, chunk_overlap=overlap)
    print(f"[ingest] chunked into {len(chunks)} documents")

    embeddings = build_embeddings(model_name)

    if vs_type == "faiss":
        ensure_dir(faiss_dir)
        vs = create_or_load_vectorstore("faiss", faiss_dir, chroma_dir, embeddings, chunks=chunks)
        FAISS.save_local(vs, folder_path=faiss_dir)
        print(f"[ingest] faiss index saved to: {faiss_dir}")
    else:
        create_or_load_vectorstore("chroma", faiss_dir, chroma_dir, embeddings, chunks=chunks)
        print(f"[ingest] chroma index persisted to: {chroma_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and index documents for RAG")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH, help="Path to settings.yaml")
    parser.add_argument("--sources", default=DEFAULT_SOURCES_PATH, help="Path to sources.yaml")
    args = parser.parse_args()
    main(args.settings, args.sources)
