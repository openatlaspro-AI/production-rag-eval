"""Unit tests for the offline retrievers. No network — synthetic fixtures only.

Covers the test list from the design spec:
  - BM25 ranks a doc containing the query term above one that doesn't.
  - RRF: a doc ranked highly by BOTH lists beats one ranked highly by only one.
  - Graph expansion: a doc sharing an entity with a seed is reachable; an
    isolated doc is not.
  - MMR: given two near-duplicate top hits, the second is demoted for diversity.
  - Metric sanity: precision_at_k / reciprocal_rank match hand-built values.
"""

from __future__ import annotations

import numpy as np

from src.eval.retrieval import precision_at_k, reciprocal_rank
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.graph import GraphRetriever, extract_entities
from src.retrievers.hybrid import reciprocal_rank_fusion
from src.retrievers.rerank import mmr_rerank


def _docs(titles_en: list[str]) -> list[dict]:
    return [{"title": "", "title_en": t} for t in titles_en]


# ---------------------------------------------------------------- BM25


def test_bm25_ranks_query_term_doc_first():
    docs = _docs(
        [
            "China and the United States resume trade talks",
            "A recipe for chocolate cake",
            "Iran oil tanker incident in the strait",
        ]
    )
    bm25 = BM25Retriever(docs)
    ranking = bm25.search("china trade", k=3)
    assert ranking[0] == 0  # the doc containing 'china'/'trade' ranks first


def test_bm25_irrelevant_doc_not_top():
    docs = _docs(["hantavirus outbreak spreads", "football world cup schedule"])
    bm25 = BM25Retriever(docs)
    assert bm25.search("hantavirus", k=2)[0] == 0


# ---------------------------------------------------------------- RRF


def test_rrf_doc_in_both_lists_wins():
    # doc 5 is rank-1 in list A and rank-2 in list B; doc 9 is rank-1 in B only.
    list_a = [5, 1, 2, 3]
    list_b = [9, 5, 7, 8]
    fused = reciprocal_rank_fusion([list_a, list_b], k0=60)
    assert fused[0] == 5
    assert fused.index(5) < fused.index(9)


def test_rrf_is_deterministic_tiebreak():
    # Two docs with identical fused score -> tie broken by ascending id.
    fused = reciprocal_rank_fusion([[1], [2]], k0=60)
    assert fused == [1, 2]


# ---------------------------------------------------------------- Graph


def test_extract_entities_finds_proper_nouns_and_keywords():
    ents = extract_entities("Trump says China-US talks proceed; Iran oil falls")
    assert "trump" in ents
    assert "china" in ents
    assert "iran" in ents


def test_graph_expansion_reaches_shared_entity_doc():
    docs = _docs(
        [
            "Iran and Hormuz tensions rise",  # 0 seed (matches 'iran')
            "Hormuz strait oil shipping disrupted",  # 1 shares 'hormuz' with 0
            "A cooking show about cakes",  # 2 isolated
        ]
    )
    graph = GraphRetriever(docs, dense=None)
    ranking = graph.search("iran", k=3)
    assert 0 in ranking  # seed reachable
    assert 1 in ranking  # shared-entity neighbour reachable via 1-hop
    assert 2 not in ranking  # isolated doc not reachable


def test_graph_isolated_doc_unreachable():
    docs = _docs(["Iran oil news", "Completely unrelated dessert recipe"])
    graph = GraphRetriever(docs, dense=None)
    ranking = graph.search("iran", k=2)
    assert ranking == [0]


# ---------------------------------------------------------------- MMR rerank


def test_mmr_demotes_near_duplicate():
    # Candidates 0 and 1 are near-identical; 2 is diverse but slightly less
    # relevant. MMR should pick 0 then 2 (not 1) for the top-2.
    corpus = np.array(
        [
            [1.0, 0.0, 0.0],  # 0: very relevant
            [0.99, 0.01, 0.0],  # 1: near-duplicate of 0
            [0.7, 0.7, 0.0],  # 2: relevant but diverse
        ],
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # lambda < 0.5 weights diversity more than relevance, so the near-duplicate
    # (1) is demoted below the diverse-but-relevant doc (2).
    out = mmr_rerank(query, [0, 1, 2], corpus, k=2, lambda_mult=0.3)
    assert out[0] == 0
    assert out[1] == 2  # diverse doc beats the near-duplicate


def test_mmr_pure_relevance_keeps_order():
    corpus = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    out = mmr_rerank(query, [0, 1, 2], corpus, k=3, lambda_mult=1.0)
    assert out == [0, 1, 2]


# ---------------------------------------------------------------- Dense (injected embed_fn, no network)


def test_dense_search_with_injected_embed_fn():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    dense = DenseRetriever(corpus_emb=corpus, embed_fn=lambda q: [1.0, 0.0], use_cache=False)
    assert dense.search("anything", k=1)[0] == 0


# ---------------------------------------------------------------- Metric sanity


def test_precision_and_rr_handbuilt():
    relevant = {2, 5}
    retrieved = [1, 2, 3, 4, 5]
    assert precision_at_k(relevant, retrieved, 5) == 2 / 5
    assert precision_at_k(relevant, retrieved, 2) == 1 / 2
    assert reciprocal_rank(relevant, retrieved) == 1 / 2  # first relevant at rank 2
