"""Hybrid retriever: Reciprocal Rank Fusion of dense + BM25 ranked lists.

RRF score for a doc = Σ_lists 1 / (k0 + rank), where rank is 1-indexed position
in that list (k0=60, the standard default from Cormack et al. 2009). Fusing on
rank rather than raw scores avoids having to calibrate incomparable cosine vs.
BM25 magnitudes.
"""

from __future__ import annotations

from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever

DEFAULT_K0 = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k0: int = DEFAULT_K0,
) -> list[int]:
    """Fuse several ranked ID lists into one, best-first, by RRF.

    Each input list is ordered best-first. Returns the union of all IDs sorted
    by descending fused score. Ties are broken by doc id for determinism.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


class HybridRetriever:
    """RRF fusion over a dense and a BM25 retriever.

    `candidate_k` controls how deep each base list goes before fusion; deeper
    lists let a doc that's strong in only one modality still surface.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        k0: int = DEFAULT_K0,
        candidate_k: int = 50,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.k0 = k0
        self.candidate_k = candidate_k

    def search(self, query: str, k: int = 10) -> list[int]:
        dense_list = self.dense.search(query, k=self.candidate_k)
        bm25_list = self.bm25.search(query, k=self.candidate_k)
        fused = reciprocal_rank_fusion([dense_list, bm25_list], k0=self.k0)
        return fused[:k]
