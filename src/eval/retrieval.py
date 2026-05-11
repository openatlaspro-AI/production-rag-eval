"""Retrieval metrics: precision@k, recall@k, MRR.

Pure functions — no I/O. Each takes a labeled `relevant` set and an ordered
`retrieved` list and returns a float. Use these from `run_baseline.py`.

Conventions:
  - `relevant` is a set of document IDs labeled as relevant for the query.
  - `retrieved` is an ordered list of document IDs returned by the retrieval
    system, sorted by similarity descending (rank 1 = best).
  - `k` is the rank cutoff (1-indexed in practice; we slice [:k] which is
    0-indexed slicing — equivalent for top-k).
  - For empty-relevant queries (negative tests), recall_at_k and
    reciprocal_rank return NaN. The orchestrator filters these out before
    aggregating.
"""

from __future__ import annotations

import math


def precision_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Fraction of top-k retrieved that are relevant.

    precision@k = |relevant ∩ retrieved[:k]| / k

    Always divides by k (the rank cutoff), not by len(retrieved). This penalizes
    under-retrieval — if the system returns only 3 docs but k=10, precision@10
    will be hits/10, not hits/3. Standard IR convention.

    Returns 0.0 for k=0 (no cutoff means no precision).
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Fraction of relevant docs that appear in top-k.

    recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Returns NaN when relevant is empty (negative test): recall is mathematically
    undefined. Caller should filter NaN before averaging.
    """
    if not relevant:
        return math.nan
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(relevant: set[int], retrieved: list[int]) -> float:
    """Reciprocal rank of the first relevant doc in retrieved order.

    RR = 1 / rank(first relevant), or 0 if no relevant doc in `retrieved`.
    Returns NaN when relevant is empty (negative test).

    Mean of RR across queries = MRR.
    """
    if not relevant:
        return math.nan
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hits_in_top_k(relevant: set[int], retrieved: list[int], k: int) -> int:
    """Count of relevant docs in top-k. Convenience helper for the per-query summary."""
    if k <= 0:
        return 0
    return sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
