"""
Document loaders for URLs, Wikipedia, and PDFs with consistent metadata and logging.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader, PyPDFLoader
from langchain_core.documents import Document
import os

from src.logging_config import get_logger

logger = get_logger(__name__)


def load_from_urls(urls: List[str]) -> List[Document]:
    """Load documents from a list of web URLs with metadata."""
    if not urls:
        logger.info("No web URLs specified in sources.")
        return []
    logger.info(f"Loading {len(urls)} URLs...")
    docs = []
    for url in urls:
        try:
            logger.debug(f"Fetching URL: {url}")
            loader = WebBaseLoader([url])
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = url
                doc.metadata["type"] = "url"
            logger.info(f"Loaded {len(loaded_docs)} docs from {url}")
            docs.extend(loaded_docs)
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {e}")
    logger.info(f"Total URL docs loaded: {len(docs)}")
    return docs


def load_from_wikipedia(items: List[Dict[str, Any]]) -> List[Document]:
    """Load documents from Wikipedia with metadata."""
    if not items:
        logger.info("No Wikipedia items specified in sources.")
        return []
    logger.info(f"Loading {len(items)} Wikipedia queries...")
    docs: List[Document] = []
    for item in items:
        query = item.get("query")
        lang = item.get("lang", "en")
        if not query:
            logger.warning("Skipping Wikipedia item with missing query.")
            continue
        try:
            logger.debug(f"Fetching Wikipedia article: query={query}, lang={lang}")
            loader = WikipediaLoader(query=query, lang=lang, load_max_docs=1)
            page_docs = loader.load()
            for d in page_docs:
                d.metadata["source"] = f"Wikipedia:{query}"
                d.metadata["type"] = "wikipedia"
            logger.info(f"Loaded {len(page_docs)} docs from Wikipedia query: {query}")
            docs.extend(page_docs)

        except Exception as e:
            logger.error(f"Failed to load Wikipedia article '{query}': {e}")
    logger.info(f"Total Wikipedia docs loaded: {len(docs)}")
    return docs


def load_from_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load and extract text from a list of local PDF files with metadata."""
    if not pdf_paths:
        logger.info("No PDF files specified in sources.")
        return []
    logger.info(f"Loading {len(pdf_paths)} PDFs...")
    docs = []
    for path in pdf_paths:
        file_path = Path(path)
        if file_path.exists():
            try:
                logger.debug(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(str(file_path))
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = str(file_path.resolve())
                    doc.metadata["type"] = "pdf"
                    doc.metadata["name"] = file_path.name
                    doc.metadata["page"] = doc.metadata.get("page", None)
                logger.info(f"Loaded {len(loaded_docs)} docs from PDF: {file_path.name}")
                docs.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Failed to load PDF {file_path}: {e}")
        else:
            logger.warning(f"PDF not found: {file_path}")
    logger.info(f"Total PDF docs loaded: {len(docs)}")
    return docs


def load_all_sources(sources: Dict[str, Any]) -> List[Document]:
    """Convenience function to load all sources and return a single docs list."""
    logger.info("Starting to load all sources...")
    docs = []
    docs += load_from_urls(sources.get("web_urls", []))
    docs += load_from_wikipedia(sources.get("wikipedia", []))
    docs += load_from_pdfs(sources.get("pdf_files", []))
    logger.info(f"Total combined documents loaded: {len(docs)}")
    return docs
