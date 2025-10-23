"""
query_processing.py

Handles all query preprocessing before retrieval:
- Cleans and normalizes user queries
- Classifies question intent (factual / conceptual / procedural)
- Expands or rephrases queries for improved embedding retrieval
- Extracts keywords for better matching
"""

import re
from typing import Dict
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from .chain import build_llm


def clean_query(query: str) -> str:
    """Remove extra spaces, punctuation noise, and normalize case."""
    query = query.strip().lower()
    query = re.sub(r"[\n\r\t]+", " ", query)
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"[^\w\s\?\-\.]", "", query)
    return query


def classify_query(query: str) -> str:
    """Heuristic classification of query type."""
    factual_keywords = ["when", "where", "who", "what", "which"]
    conceptual_keywords = ["why", "how", "explain", "describe"]
    procedural_keywords = ["steps", "process", "how to"]

    q = query.lower()
    if any(k in q for k in procedural_keywords):
        return "procedural"
    elif any(k in q for k in conceptual_keywords):
        return "conceptual"
    elif any(k in q for k in factual_keywords):
        return "factual"
    else:
        return "generic"


def expand_query(query: str, settings: dict) -> str:
    """
    Expands or paraphrases a query for improved retrieval.

    Args:
        query (str): The user's cleaned query.
        settings (dict): Loaded settings.yaml configuration.

    Returns:
        str: Expanded query string (or original if expansion disabled).
    """
    llm = build_llm(settings)
    prompt = PromptTemplate.from_template(
        "Rewrite this query into a clearer and more detailed research-oriented form:\n{query}"
    )
    try:
        # Generate a formatted string prompt
        formatted_prompt = prompt.format(query=query)
        response = llm.invoke(formatted_prompt)
        return getattr(response, "content", str(response)).strip()
    except Exception as e:
        print(f"[warn] Query expansion failed ({e}), using original query.")
        return query


def extract_keywords(query: str) -> list:
    """Simple keyword extraction heuristic (can be replaced by RAKE/spacy)."""
    stopwords = {"what", "is", "the", "a", "an", "of", "and", "in", "to", "why", "how", "on"}
    words = [w for w in re.findall(r"\b\w+\b", query.lower()) if w not in stopwords]
    return list(dict.fromkeys(words))  # unique order-preserving


def process_query(query: str, settings: dict, expand: bool = False) -> Dict[str, any]:
    """
    Full preprocessing pipeline returning structured query info.
    """
    cleaned = clean_query(query)
    qtype = classify_query(cleaned)
    keywords = extract_keywords(cleaned)
    expanded = expand_query(cleaned, settings) if expand else cleaned
    return {"original": query, "cleaned": cleaned, "expanded": expanded, "type": qtype, "keywords": keywords}
