"""Dense cosine retriever over OpenAI corpus embeddings.

Loads `data/embeddings_openai.npy` (L2-normalized at load), embeds the query
once with OpenAI `text-embedding-3-small`, and returns top-k by cosine
similarity. Query embeddings are cached to disk keyed by the query string so
re-running the benchmark needs no API calls after the first pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
EMB_PATH = DATA_DIR / "embeddings_openai.npy"
QCACHE_PATH = DATA_DIR / "query_embeddings_openai.json"

MODEL = "text-embedding-3-small"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


class DenseRetriever:
    """Cosine top-k over precomputed OpenAI corpus embeddings.

    Parameters
    ----------
    corpus_emb:
        Optional precomputed (N, dim) matrix. If omitted, loaded from disk.
    embed_fn:
        Optional callable(str) -> list[float] for query embedding. Injected in
        tests to avoid network. Defaults to the OpenAI API (lazy-imported).
    use_cache:
        Persist query embeddings to disk so repeat runs are network-free.
    """

    def __init__(
        self,
        corpus_emb: np.ndarray | None = None,
        embed_fn=None,
        use_cache: bool = True,
    ) -> None:
        if corpus_emb is None:
            if not EMB_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {EMB_PATH}. Run `python -m scripts.embed_openai` first."
                )
            corpus_emb = np.load(EMB_PATH)
        self.corpus_emb = _l2_normalize(np.asarray(corpus_emb, dtype=np.float32))
        self._embed_fn = embed_fn
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] = {}
        if use_cache and QCACHE_PATH.exists():
            self._cache = json.loads(QCACHE_PATH.read_text())

    def _openai_embed(self, query: str) -> list[float]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.embeddings.create(model=MODEL, input=[query])
        return resp.data[0].embedding

    def embed_query(self, query: str) -> np.ndarray:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        if self.use_cache and key in self._cache:
            vec = self._cache[key]
        elif self._embed_fn is not None:
            vec = list(self._embed_fn(query))
        else:
            vec = self._openai_embed(query)
            if self.use_cache:
                self._cache[key] = vec
                QCACHE_PATH.write_text(json.dumps(self._cache))
        arr = np.asarray(vec, dtype=np.float32)
        return arr / max(float(np.linalg.norm(arr)), 1e-12)

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every corpus doc, shape (N,)."""
        q = self.embed_query(query)
        return self.corpus_emb @ q

    def search(self, query: str, k: int = 10) -> list[int]:
        sims = self.similarities(query)
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [int(i) for i in top]
