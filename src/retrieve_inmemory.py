"""In-memory cosine retrieval over the 400-doc corpus for the Streamlit demo.

Streamlit Cloud has no Postgres, so we load pre-computed embeddings
(data/embeddings.npy) at app boot and do brute-force cosine top-k in numpy.
~1 ms per query at this corpus size — FAISS would be overkill.

Production pgvector path (src/retrieve.py) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieve import RetrievalHit

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
JSONL_PATH = DATA_DIR / "trend_signals.jsonl"


def load_corpus() -> tuple[np.ndarray, list[dict]]:
    """Load + L2-normalize the corpus embeddings, plus per-doc metadata.

    Call once at app startup (cache with @st.cache_resource).
    Returns (normalized_embeddings of shape (N, 1024), docs list of length N).
    """
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EMBEDDINGS_PATH}. Run `python -m scripts.precompute_embeddings`."
        )
    emb = np.load(EMBEDDINGS_PATH)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)

    docs = [json.loads(line) for line in JSONL_PATH.open(encoding="utf-8")]
    if len(docs) != emb.shape[0]:
        raise ValueError(
            f"corpus mismatch: {len(docs)} docs vs {emb.shape[0]} embeddings. "
            "Re-run scripts/precompute_embeddings."
        )
    return emb, docs


def inmemory_search(
    q_emb: np.ndarray,
    corpus_emb: np.ndarray,
    docs: list[dict],
    k: int = 5,
) -> list[RetrievalHit]:
    """Top-k cosine similarity. `corpus_emb` must already be L2-normalized.

    Returns RetrievalHit objects so downstream code matches the pgvector path.
    `id` is the row index in the jsonl (not a DB primary key).
    """
    q = q_emb / max(float(np.linalg.norm(q_emb)), 1e-12)
    sims = corpus_emb @ q  # cosine similarity (both sides normalized)
    k = min(k, sims.shape[0])
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    hits: list[RetrievalHit] = []
    for i in top_idx:
        d = docs[int(i)]
        platforms = d.get("platforms") or []
        hits.append(
            RetrievalHit(
                id=int(i),
                source="trend",
                title=d.get("title"),
                title_en=d.get("title_en"),
                platform=platforms[0] if platforms else None,
                rank=d.get("min_rank"),
                similarity=float(sims[i]),
                metadata={
                    "platforms": platforms,
                    "crawl_count": d.get("crawl_count"),
                    "url": d.get("url"),
                    "first_seen": d.get("first_seen"),
                    "external_id": d.get("external_id"),
                },
            )
        )
    return hits
