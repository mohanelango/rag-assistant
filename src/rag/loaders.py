"""
Document loaders for URLs, Wikipedia, and PDFs with consistent metadata.
"""

from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader, PyPDFLoader
from typing import List, Dict, Any
from langchain_core.documents import Document


def load_from_urls(urls):
    """Load documents from a list of web URLs with metadata."""
    docs = []
    for url in urls:
        loader = WebBaseLoader([url])
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = url
            doc.metadata["type"] = "url"
        docs.extend(loaded_docs)
    return docs


def load_from_wikipedia(items: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for item in items:
        query = item.get("query")
        lang = item.get("lang", "en")
        if not query:
            continue
        loader = WikipediaLoader(query=query, lang=lang, load_max_docs=1)
        page_docs = loader.load()
        for d in page_docs:
            d.metadata["source"] = f"Wikipedia:{query}"
            d.metadata["type"] = "wikipedia"
        docs.extend(page_docs)
    return docs


def load_from_pdfs(pdf_paths):
    """Load and extract text from a list of local PDF files with metadata."""
    docs = []
    for path in pdf_paths:
        file_path = Path(path)
        if file_path.exists():
            loader = PyPDFLoader(str(file_path))
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = str(file_path.resolve())
                doc.metadata["type"] = "pdf"
                doc.metadata["name"] = file_path.name
                doc.metadata["page"] = doc.metadata.get("page", None)
            docs.extend(loaded_docs)
        else:
            print(f"[warn] PDF not found: {path}")
    return docs


def load_all_sources(sources):
    """Convenience function to load all sources and return a single docs list."""
    docs = []
    docs += load_from_urls(sources.get("web_urls", []))
    docs += load_from_wikipedia(sources.get("wikipedia", []))
    docs += load_from_pdfs(sources.get("pdf_files", []))
    return docs
