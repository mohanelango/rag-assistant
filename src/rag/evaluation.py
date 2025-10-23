"""
Retrieval Evaluation Utilities

This module provides functions to evaluate retrieval performance
using metrics such as Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
"""

from typing import List, Tuple
from langchain.schema import Document


def precision_at_k(retrieved_docs: List[Document], relevant_docs: List[str], k: int = 5) -> float:
    """Compute Precision@K"""
    retrieved_sources = [d.metadata.get("source") for d in retrieved_docs[:k]]
    relevant_hits = [s for s in retrieved_sources if s in relevant_docs]
    return len(relevant_hits) / k if k > 0 else 0.0


def recall_at_k(retrieved_docs: List[Document], relevant_docs: List[str], k: int = 5) -> float:
    """Compute Recall@K"""
    retrieved_sources = [d.metadata.get("source") for d in retrieved_docs[:k]]
    relevant_hits = [s for s in retrieved_sources if s in relevant_docs]
    return len(relevant_hits) / len(relevant_docs) if relevant_docs else 0.0


def mean_reciprocal_rank(results: List[List[str]], relevant_docs: List[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR)"""
    ranks = []
    for query_results in results:
        for rank, doc in enumerate(query_results, start=1):
            if doc in relevant_docs:
                ranks.append(1.0 / rank)
                break
        else:
            ranks.append(0.0)
    return sum(ranks) / len(ranks) if ranks else 0.0
