"""Similarity search over pgvector documents. Returns top-k by cosine similarity."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import psycopg
from mistralai import Mistral
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.table import Table

from src.config import settings

console = Console()


@dataclass
class RetrievalHit:
    id: int
    source: str
    title: str | None
    title_en: str | None
    platform: str | None
    rank: int | None
    similarity: float
    metadata: dict


def embed_query(client: Mistral, query: str) -> np.ndarray:
    """Embed a single query via mistral-embed."""
    response = client.embeddings.create(model=settings.mistral_embed_model, inputs=[query])
    return np.array(response.data[0].embedding, dtype=np.float32)


def similarity_search(query: str, k: int = 5, source: str | None = None) -> list[RetrievalHit]:
    """Top-k cosine similarity over documents.embedding.

    Args:
        query: natural-language query (English)
        k: number of results
        source: optional filter — 'trend' or 'product'

    Returns:
        List of RetrievalHit, sorted by similarity desc.
    """
    client = Mistral(api_key=settings.mistral_api_key)
    q_emb = embed_query(client, query)

    where_clause = ""
    params: list = [q_emb]
    if source:
        where_clause = "WHERE source = %s"
        params.append(source)
    params.extend([q_emb, k])

    sql = f"""
        SELECT id, source, title, title_en, platform, rank, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with psycopg.connect(settings.postgres_dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        RetrievalHit(
            id=r[0], source=r[1], title=r[2], title_en=r[3],
            platform=r[4], rank=r[5], metadata=r[6] or {},
            similarity=float(r[7]),
        )
        for r in rows
    ]


def print_hits(query: str, hits: list[RetrievalHit], elapsed_ms: float) -> None:
    """Pretty-print retrieval results in a Rich table."""
    table = Table(title=f'Query: "{query}"  ({len(hits)} hits in {elapsed_ms:.0f}ms)', show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("sim", justify="right", style="cyan")
    table.add_column("source", style="green", width=8)
    table.add_column("platform", style="magenta", width=12)
    table.add_column("English", style="bold white")
    table.add_column("Original (zh)", style="dim")

    for i, h in enumerate(hits, 1):
        table.add_row(
            str(i),
            f"{h.similarity:.3f}",
            h.source,
            h.platform or "—",
            (h.title_en or "")[:80],
            (h.title or "")[:40],
        )
    console.print(table)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what trends relate to family or household"
    console.print(f"[dim]Searching for:[/dim] [bold]{query}[/bold]\n")
    t0 = time.perf_counter()
    hits = similarity_search(query, k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not hits:
        console.print("[red]No results — is the documents table populated? Run `make ingest` first.[/red]")
        sys.exit(1)
    print_hits(query, hits, elapsed_ms)
