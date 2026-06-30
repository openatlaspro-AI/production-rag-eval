"""Offline retrievers for the multi-method benchmark.

Each retriever exposes a `search(query, k) -> list[int]` method returning corpus
indices (0-indexed row in `data/trend_signals.jsonl`) ranked best-first.

Implementations are deliberately small and dependency-light so the benchmark is
reproducible offline with only OpenAI embeddings (for the dense query vector)
plus rank-bm25 / numpy / networkx.

  - DenseRetriever  : cosine over OpenAI corpus embeddings
  - BM25Retriever   : lexical BM25 over title + title_en
  - HybridRetriever : Reciprocal Rank Fusion (RRF, k0=60) of dense + bm25
  - GraphRetriever  : entity co-occurrence graph, 1-hop expansion (no LLM)
  - mmr_rerank      : Maximal Marginal Relevance diversification (no network)
"""

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.graph import GraphRetriever
from src.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from src.retrievers.rerank import mmr_rerank

__all__ = [
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "GraphRetriever",
    "reciprocal_rank_fusion",
    "mmr_rerank",
]
