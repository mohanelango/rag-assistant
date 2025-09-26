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
from .utils import load_yaml, get_unique_sources
from .chain import build_rag_chain

DEFAULT_SETTINGS_PATH = "configs/settings.yaml"
DEFAULT_SOURCES_PATH = "configs/sources.yaml"


def warn_if_stale(settings_path: str, sources_path: str):
    """
    Warn if sources.yaml is newer than vectorstore (Chroma or FAISS).

    Args:
        settings_path (str): Path to settings.yaml
        sources_path (str): Path to sources.yaml
    """
    settings = load_yaml(settings_path)
    vs_type = settings["vectorstore"]["type"]
    chroma_dir = settings["vectorstore"]["chroma_dir"]
    faiss_dir = settings["vectorstore"]["persist_dir"]

    try:
        sources_mtime = Path(sources_path).stat().st_mtime
    except FileNotFoundError:
        return  # no sources file

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
        print("⚠️ WARNING: sources.yaml was modified after the last ingestion.")
        print("Run `make clean && make ingest` to update your index before querying.\n")


def main(question: str, settings_path: str = DEFAULT_SETTINGS_PATH, sources_path: str = DEFAULT_SOURCES_PATH):
    """
    Query the RAG pipeline from the CLI.

    Args:
        question (str): User question.
        settings_path (str): Path to settings.yaml (defaults to configs/settings.yaml)
        sources_path (str): Path to sources.yaml (defaults to configs/sources.yaml)
    """
    warn_if_stale(settings_path, sources_path)

    # Load config and embeddings
    settings = load_yaml(settings_path)
    embeddings = HuggingFaceEmbeddings(model_name=settings["embeddings"]["model_name"])

    # Build chain and retriever
    chain, retriever = build_rag_chain(settings, embeddings)

    # Run the RAG pipeline
    result = chain.invoke({"question": question})
    answer_text = getattr(result, "content", str(result))

    # Retrieve docs with similarity scores
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(
        question, k=settings["retrieval"]["k"]
    )

    # Deduplicate by (source, page) combo so multiple chunks from the same page don't repeat
    unique_sources = get_unique_sources(docs_and_scores)

    print("\n=== ANSWER ===\n")
    print(answer_text.strip())

    print("\n=== SOURCES (deduplicated, with metadata) ===")
    print(json.dumps(unique_sources, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions to the RAG assistant")
    parser.add_argument("--question", "-q", required=True, help="The question to ask")
    parser.add_argument(
        "--settings", default=DEFAULT_SETTINGS_PATH,
        help="Path to settings.yaml (default: configs/settings.yaml)"
    )
    parser.add_argument(
        "--sources", default=DEFAULT_SOURCES_PATH,
        help="Path to sources.yaml (default: configs/sources.yaml)"
    )
    args = parser.parse_args()
    main(args.question, args.settings, args.sources)
