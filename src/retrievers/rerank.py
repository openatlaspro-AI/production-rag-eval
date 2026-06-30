"""Reranking stage. Default = Maximal Marginal Relevance (MMR), deterministic
and network-free. An optional OpenAI LLM reranker is available behind a flag.

MMR re-orders a candidate list to balance relevance to the query against
diversity (novelty vs. already-selected docs):

    MMR = argmax_{d in candidates} [ λ * rel(d, q) - (1-λ) * max_{s in selected} sim(d, s) ]

Relevance and pairwise similarity both use the corpus embeddings (cosine). This
demotes near-duplicate top hits in favor of a doc that covers a different facet.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(
    query_emb: np.ndarray,
    candidate_ids: list[int],
    corpus_emb: np.ndarray,
    k: int = 10,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Reorder `candidate_ids` by MMR. Inputs need not be pre-normalized.

    Parameters
    ----------
    query_emb:
        (dim,) query vector.
    candidate_ids:
        Candidate corpus indices (best-first from a prior stage).
    corpus_emb:
        (N, dim) corpus matrix; rows indexed by candidate id.
    k:
        Number of results to return.
    lambda_mult:
        Trade-off; 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if not candidate_ids:
        return []

    def norm(mat: np.ndarray) -> np.ndarray:
        return mat / np.clip(np.linalg.norm(mat, axis=-1, keepdims=True), 1e-12, None)

    q = query_emb / max(float(np.linalg.norm(query_emb)), 1e-12)
    cand = np.asarray(candidate_ids)
    cand_emb = norm(corpus_emb[cand])
    rel = cand_emb @ q  # cosine relevance to query

    selected: list[int] = []
    remaining = list(range(len(cand)))
    k = min(k, len(cand))

    while len(selected) < k and remaining:
        best_local = None
        best_score = -np.inf
        for r in remaining:
            if not selected:
                score = rel[r]
            else:
                sel_emb = cand_emb[selected]
                redundancy = float(np.max(sel_emb @ cand_emb[r]))
                score = lambda_mult * rel[r] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score = score
                best_local = r
        selected.append(best_local)
        remaining.remove(best_local)

    return [int(cand[i]) for i in selected]


def llm_rerank(
    query: str,
    candidate_ids: list[int],
    docs: list[dict],
    k: int = 10,
    model: str = "gpt-4o-mini",
) -> list[int]:
    """Optional OpenAI LLM reranker. Only call when a key is present.

    Asks the model to return the candidate ids most relevant to the query, in
    order. Falls back to the input order on any parse failure (never fabricates).
    """
    import json

    from openai import OpenAI

    listing = "\n".join(
        f"{cid}: {(docs[cid].get('title_en') or docs[cid].get('title') or '')}"
        for cid in candidate_ids
    )
    prompt = (
        "You are a search reranker. Given a query and candidate documents "
        "(id: title), return the ids most relevant to the query, best first, "
        f"as a JSON list of integers (at most {k}). Only use the given ids.\n\n"
        f"Query: {query}\n\nCandidates:\n{listing}\n\nJSON list:"
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    try:
        start, end = text.index("["), text.rindex("]") + 1
        order = json.loads(text[start:end])
        valid = [int(c) for c in order if int(c) in set(candidate_ids)]
        # Append any candidates the model dropped, preserving prior order.
        for cid in candidate_ids:
            if cid not in valid:
                valid.append(cid)
        return valid[:k]
    except (ValueError, json.JSONDecodeError):
        return candidate_ids[:k]
