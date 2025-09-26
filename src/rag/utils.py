import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ENV_PATTERN = re.compile(r"\$\{([^:}]+)(?::([^}]+))?\}")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file and expand environment variable references.

    Supports syntax like:
        ${ENV_VAR:default_value}

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        dict: Parsed YAML with expanded environment variables.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Expand ${VAR:default} manually
    def replacer(match: re.Match) -> str:
        var_name, default_val = match.group(1), match.group(2)
        return os.getenv(var_name, default_val if default_val is not None else "")

    expanded = ENV_PATTERN.sub(replacer, raw)
    return yaml.safe_load(expanded)


def clean_text(text: str) -> str:
    """
    Clean raw text by normalizing whitespace and removing noise.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned and normalized text.
    """
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def ensure_dir(path: str | Path) -> None:
    """
    Ensure a directory exists.

    Args:
        path (str | Path): Directory to create if missing.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str | None:
    """
    Retrieve an environment variable with a fallback.

    Args:
        name (str): Name of the environment variable.
        default (str | None): Value if not found.

    Returns:
        str | None: Environment value or default.
    """
    return os.getenv(name, default)


def warn_if_stale(settings_path: str, sources_path: str) -> bool:
    """
    Warn if sources.yaml is newer than the vectorstore (Chroma or FAISS).

    Args:
        settings_path (str): Path to settings.yaml
        sources_path (str): Path to sources.yaml

    Returns:
        bool: True if sources are newer (stale index), False otherwise.
    """
    settings = load_yaml(settings_path)
    vs_type = settings["vectorstore"]["type"]
    chroma_dir = settings["vectorstore"]["chroma_dir"]
    faiss_dir = settings["vectorstore"]["persist_dir"]

    try:
        sources_mtime = Path(sources_path).stat().st_mtime
    except FileNotFoundError:
        return False

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
        return True
    return False


def chunk_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Split documents into smaller chunks while preserving metadata.

    Args:
        docs (list): List of Document objects.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of chunked Document objects with metadata preserved.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []
    for doc in docs:
        # Split and propagate metadata into every chunk
        chunks = splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata = dict(doc.metadata)  # deep copy metadata
            chunk.metadata["chunk_size"] = chunk_size
        all_chunks.extend(chunks)

    return all_chunks


def get_unique_sources(docs_and_scores):
    """
    Deduplicate and extract source metadata for display.

    Args:
        docs_and_scores (list): List of (Document, score) tuples.

    Returns:
        list: List of dicts with source, type, name, title, page, and score.
    """
    seen_keys = set()
    unique_sources = []
    for doc, score in docs_and_scores:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_sources.append({
                "source": doc.metadata.get("source"),
                "type": doc.metadata.get("type"),
                "name": doc.metadata.get("name"),
                "title": doc.metadata.get("title"),
                "page": doc.metadata.get("page"),
                "score": round(float(score), 4)
            })
    return unique_sources
