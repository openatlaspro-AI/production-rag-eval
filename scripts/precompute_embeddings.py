"""One-shot: embed the 400 trend titles → write data/embeddings.npy + meta sidecar.

Used by the Streamlit Cloud demo, which has no Postgres. The committed .npy is
loaded at app boot for in-memory cosine retrieval (see src/retrieve_inmemory.py).

Re-run whenever data/trend_signals.jsonl changes:
    python -m scripts.precompute_embeddings
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mistralai import Mistral
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.config import settings

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
JSONL_PATH = DATA_DIR / "trend_signals.jsonl"
NPY_PATH = DATA_DIR / "embeddings.npy"
META_PATH = DATA_DIR / "embeddings_meta.json"

BATCH_SIZE = 50
RATE_LIMIT_SLEEP = 0.5


def main() -> None:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing {JSONL_PATH} — run `make ingest` first.")

    docs = [json.loads(line) for line in JSONL_PATH.open(encoding="utf-8")]
    texts = [d["title_en"] for d in docs]
    console.print(f"[dim]Embedding {len(texts):,} titles via {settings.mistral_embed_model}[/dim]")

    client = Mistral(api_key=settings.mistral_api_key)
    embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("embedding", total=len(texts))
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = client.embeddings.create(model=settings.mistral_embed_model, inputs=batch)
            items = sorted(response.data, key=lambda d: getattr(d, "index", 0))
            embeddings.extend(d.embedding for d in items)
            progress.update(task, advance=len(batch))
            time.sleep(RATE_LIMIT_SLEEP)

    arr = np.asarray(embeddings, dtype=np.float32)
    assert arr.shape == (len(texts), 1024), f"unexpected shape {arr.shape}"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(NPY_PATH, arr)

    meta = {
        "model": settings.mistral_embed_model,
        "n_docs": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": JSONL_PATH.name,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    size_kb = NPY_PATH.stat().st_size / 1024
    console.print(f"[green]✓[/green] wrote {NPY_PATH.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
    console.print(f"[green]✓[/green] wrote {META_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
