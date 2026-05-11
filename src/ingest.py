"""Ingest TrendRadar SQLite DBs → translate → embed → insert into pgvector.

Pipeline:
  1. Walk all daily DBs in TRENDRADAR_DB_DIR
  2. Aggregate news_items by deduplicated title (sum crawl_count, union platforms)
  3. Sort by total crawl_count, take top N
  4. Translate Chinese titles → English in batches via mistral-small-latest
  5. Embed English titles in batches via mistral-embed (1024-dim)
  6. Write data/trend_signals.jsonl (committed)
  7. Insert into documents table (pgvector)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
from mistralai import Mistral
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config import settings

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_trends() -> list[dict]:
    """Aggregate news_items across all daily TrendRadar DBs."""
    db_dir = settings.trendradar_db_dir
    if not db_dir.exists():
        raise FileNotFoundError(f"TrendRadar dir not found: {db_dir}")

    db_paths = sorted(db_dir.glob("*.db"))
    console.print(f"[dim]Found {len(db_paths)} daily DBs in {db_dir}[/dim]")

    agg: dict[str, dict] = defaultdict(
        lambda: {
            "crawl_count": 0,
            "platforms": set(),
            "ranks": [],
            "urls": set(),
            "first_seen": None,
            "external_ids": [],
        }
    )

    for db_path in db_paths:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(
                "SELECT id, title, platform_id, rank, url, first_crawl_time, crawl_count "
                "FROM news_items"
            )
            for row in cur:
                id_, title, platform, rank, url, first_seen, count = row
                if not title or not title.strip():
                    continue
                t = title.strip()
                d = agg[t]
                d["crawl_count"] += count or 1
                d["platforms"].add(platform)
                d["ranks"].append(rank)
                if url:
                    d["urls"].add(url)
                d["external_ids"].append(f"{db_path.stem}_{id_}")
                if d["first_seen"] is None or (first_seen and first_seen < d["first_seen"]):
                    d["first_seen"] = first_seen
        finally:
            conn.close()

    items = []
    for title, d in agg.items():
        items.append(
            {
                "title": title,
                "crawl_count": d["crawl_count"],
                "platforms": sorted(d["platforms"]),
                "min_rank": min(d["ranks"]) if d["ranks"] else 999,
                "url": next(iter(d["urls"])) if d["urls"] else None,
                "external_id": d["external_ids"][0],
                "first_seen": d["first_seen"],
            }
        )
    items.sort(key=lambda x: (-x["crawl_count"], x["min_rank"]))
    return items


def translate_batch(client: Mistral, texts: list[str], model: str) -> list[str]:
    """Translate Chinese to English. Returns list aligned to input."""
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You translate Chinese news headlines to concise English. "
                    "For each numbered input, output the English translation on its own line, "
                    "prefixed with the same number and period (e.g. '1. ...'). "
                    "Keep each translation under 100 characters. "
                    "Output ONLY the numbered translations, no other text."
                ),
            },
            {"role": "user", "content": numbered},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()

    # Parse numbered output back to list
    translations = [""] * len(texts)
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match "N. text" or "N) text" or "N: text"
        for i in range(1, len(texts) + 1):
            for sep in [f"{i}. ", f"{i}) ", f"{i}: ", f"{i} - "]:
                if line.startswith(sep):
                    translations[i - 1] = line[len(sep):].strip()
                    break

    # Fallback for any blanks: use original
    for i, (t, orig) in enumerate(zip(translations, texts)):
        if not t:
            translations[i] = orig
    return translations


def embed_batch(client: Mistral, texts: list[str], model: str) -> list[list[float]]:
    """Embed batch of texts via mistral-embed."""
    response = client.embeddings.create(model=model, inputs=texts)
    # Sort by index to ensure alignment (Mistral returns data with index field)
    items = sorted(response.data, key=lambda d: getattr(d, "index", 0))
    return [d.embedding for d in items]


def main(
    top_n: int = 400,
    translate_batch_size: int = 10,
    embed_batch_size: int = 50,
    rate_limit_sleep: float = 0.5,
) -> None:
    console.rule("[bold cyan]Loading trends from SQLite DBs[/bold cyan]")
    items = load_trends()
    console.print(
        f"  Aggregated [bold]{len(items):,}[/bold] unique titles "
        f"(top crawl_count: {items[0]['crawl_count']})"
    )

    items = items[:top_n]
    console.print(f"  Selected top [bold]{len(items)}[/bold] by aggregated crawl_count")

    client = Mistral(api_key=settings.mistral_api_key)

    # === Translate ===
    console.rule("[bold cyan]Translating titles (Chinese → English)[/bold cyan]")
    titles_zh = [it["title"] for it in items]
    translations: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("translating", total=len(titles_zh))
        for i in range(0, len(titles_zh), translate_batch_size):
            batch = titles_zh[i : i + translate_batch_size]
            try:
                t = translate_batch(client, batch, settings.mistral_gen_model_small)
            except Exception as e:
                console.print(f"[red]  ! batch {i} failed: {e}[/red]")
                t = batch  # fallback: original
            translations.extend(t)
            progress.update(task, advance=len(batch))
            time.sleep(rate_limit_sleep)

    for it, en in zip(items, translations):
        it["title_en"] = en

    # === Embed ===
    console.rule("[bold cyan]Embedding (mistral-embed, 1024-dim)[/bold cyan]")
    embeddings: list[list[float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("embedding", total=len(items))
        for i in range(0, len(items), embed_batch_size):
            batch_texts = [it["title_en"] for it in items[i : i + embed_batch_size]]
            try:
                e = embed_batch(client, batch_texts, settings.mistral_embed_model)
            except Exception as ex:
                console.print(f"[red]  ! embed batch {i} failed: {ex}[/red]")
                raise
            embeddings.extend(e)
            progress.update(task, advance=len(batch_texts))
            time.sleep(rate_limit_sleep)

    assert len(embeddings) == len(items), f"embedding count mismatch: {len(embeddings)} vs {len(items)}"

    # === Write JSONL (committed snapshot) ===
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = DATA_DIR / "trend_signals.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(
                json.dumps(
                    {
                        "external_id": it["external_id"],
                        "title": it["title"],
                        "title_en": it["title_en"],
                        "platforms": it["platforms"],
                        "min_rank": it["min_rank"],
                        "crawl_count": it["crawl_count"],
                        "url": it.get("url"),
                        "first_seen": it.get("first_seen"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    console.print(f"  Wrote [bold]{jsonl_path.relative_to(REPO_ROOT)}[/bold] ({jsonl_path.stat().st_size:,} bytes)")

    # === Insert into Postgres ===
    console.rule("[bold cyan]Inserting into pgvector[/bold cyan]")
    inserted = 0
    skipped = 0
    with psycopg.connect(settings.postgres_dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for it, emb in zip(items, embeddings):
                cur.execute(
                    """
                    INSERT INTO documents
                        (source, external_id, title, title_en, content, content_en,
                         platform, rank, metadata, embedding)
                    VALUES ('trend', %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, external_id, content) DO NOTHING
                    RETURNING id
                    """,
                    (
                        it["external_id"],
                        it["title"],
                        it["title_en"],
                        it["title"],          # content = original (Chinese) for trends
                        it["title_en"],       # content_en = English
                        it["platforms"][0] if it["platforms"] else None,
                        it["min_rank"],
                        json.dumps(
                            {
                                "platforms": it["platforms"],
                                "crawl_count": it["crawl_count"],
                                "url": it.get("url"),
                                "first_seen": it.get("first_seen"),
                            }
                        ),
                        np.array(emb, dtype=np.float32),
                    ),
                )
                if cur.fetchone() is not None:
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  Inserted: [bold green]{inserted}[/bold green]")
    console.print(f"  Skipped (already exists): [yellow]{skipped}[/yellow]")
    console.print(f"  Total in DB → [dim]docker compose exec postgres psql -U rag -d familyhq_rag -c 'SELECT COUNT(*) FROM documents'[/dim]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(1)
