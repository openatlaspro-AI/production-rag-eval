"""Embed the 400-doc trend corpus with OpenAI `text-embedding-3-small`.

Writes `data/embeddings_openai.npy` (shape (N, 1536), float32) plus a meta
JSON. Idempotent: skips the API entirely if the artifacts already exist, unless
`--force` is passed. Batches the API calls so the whole corpus is a handful of
requests.

This is a NEW, self-contained embedding artifact distinct from the original
Mistral `data/embeddings.npy`. The offline multi-method benchmark
(`src/eval/run_offline.py`) reads these.

Usage:
    python -m scripts.embed_openai            # embed if missing
    python -m scripts.embed_openai --force    # always re-embed
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
JSONL_PATH = DATA_DIR / "trend_signals.jsonl"
EMB_PATH = DATA_DIR / "embeddings_openai.npy"
META_PATH = DATA_DIR / "embeddings_openai_meta.json"

MODEL = "text-embedding-3-small"
BATCH_SIZE = 128


def doc_text(doc: dict) -> str:
    """Text we embed for a corpus doc: the Chinese title + English title.

    Mirrors the in-memory retrieval convention (title + title_en). Both are
    included so the dense retriever sees the bilingual signal.
    """
    title = (doc.get("title") or "").strip()
    title_en = (doc.get("title_en") or "").strip()
    return f"{title}\n{title_en}".strip()


def load_docs() -> list[dict]:
    return [json.loads(line) for line in JSONL_PATH.open(encoding="utf-8")]


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed `texts` in batches, preserving order. Returns (len(texts), dim)."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        resp = client.embeddings.create(model=MODEL, input=batch)
        # API guarantees data is returned in input order, but sort by index to be safe.
        for item in sorted(resp.data, key=lambda d: d.index):
            vectors.append(item.embedding)
    return np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed corpus with OpenAI.")
    parser.add_argument("--force", action="store_true", help="Re-embed even if artifacts exist.")
    args = parser.parse_args()

    if EMB_PATH.exists() and META_PATH.exists() and not args.force:
        emb = np.load(EMB_PATH)
        print(f"[skip] {EMB_PATH.name} already exists ({emb.shape}). Use --force to re-embed.")
        return

    docs = load_docs()
    texts = [doc_text(d) for d in docs]
    print(f"Embedding {len(texts)} docs with {MODEL} (batch={BATCH_SIZE})...")

    client = OpenAI()
    emb = embed_texts(client, texts)
    if emb.shape[0] != len(docs):
        raise RuntimeError(f"embedding count {emb.shape[0]} != doc count {len(docs)}")

    np.save(EMB_PATH, emb)
    meta = {
        "model": MODEL,
        "n_docs": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_jsonl": JSONL_PATH.name,
        "embed_text": "title + title_en (newline-joined)",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {EMB_PATH.relative_to(REPO_ROOT)}  shape={emb.shape}")
    print(f"Saved -> {META_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
