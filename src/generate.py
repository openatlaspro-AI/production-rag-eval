"""RAG response generation. Combines retrieved context with Mistral chat completion.

Returns full instrumentation: answer, sources used, model, token counts, latency
per stage, and total USD cost. This makes it easy to plug into the eval harness
on Day 4 without re-instrumenting.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass

import psycopg
from mistralai import Mistral
from pgvector.psycopg import register_vector
from rich.console import Console

from src.config import settings
from src.pricing import calc_cost, format_cost
from src.retrieve import RetrievalHit, embed_query, pgvector_search

console = Console()

ALLOWED_MODELS = {"mistral-small-latest", "mistral-large-latest"}

SYSTEM_PROMPT = """You are a research assistant answering questions about trending topics in Chinese-language media.

Answer the user's question using ONLY the numbered context items below. The items are news headlines that have been translated to English. Cite sources inline using bracket notation: [1], [2], etc., referring to the numbered items.

Rules:
- Use ONLY the provided context. If the answer is not in the context, say so plainly.
- Cite every factual claim with the corresponding [N] reference.
- Keep the answer concise (3-6 sentences).
- Do not invent details not present in the context."""


@dataclass
class TimingBreakdown:
    embed_ms: float
    retrieve_ms: float
    generate_ms: float
    total_ms: float


@dataclass
class CostBreakdown:
    embed_usd: float
    generate_usd: float
    total_usd: float
    embed_tokens_in: int
    generate_tokens_in: int
    generate_tokens_out: int


@dataclass
class RagResponse:
    query: str
    answer: str
    sources: list[RetrievalHit]
    model: str
    timing: TimingBreakdown
    cost: CostBreakdown


def build_user_message(query: str, hits: list[RetrievalHit]) -> str:
    """Format the retrieved hits as numbered context plus the user's question."""
    lines = ["Context:"]
    for i, h in enumerate(hits, 1):
        platform = f" (source: {h.platform})" if h.platform else ""
        title = h.title_en or h.title or "(no title)"
        lines.append(f"[{i}] {title}{platform}")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def generate_rag_response(
    query: str,
    *,
    k: int = 5,
    model: str = "mistral-small-latest",
    source: str | None = None,
    client: Mistral | None = None,
    conn: psycopg.Connection | None = None,
) -> RagResponse:
    """End-to-end RAG: embed → retrieve → generate.

    Pass a shared `client` and `conn` for API/eval use to avoid per-call setup.
    """
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model!r} not in allowlist: {sorted(ALLOWED_MODELS)}")

    own_client = client is None
    own_conn = conn is None

    if own_client:
        client = Mistral(api_key=settings.mistral_api_key)
    if own_conn:
        conn = psycopg.connect(settings.postgres_dsn)
        register_vector(conn)

    try:
        # 1. Embed the query
        t0 = time.perf_counter()
        q_emb, embed_tokens_in = embed_query(client, query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. Retrieve from pgvector
        t1 = time.perf_counter()
        hits = pgvector_search(conn, q_emb, k=k, source=source)
        retrieve_ms = (time.perf_counter() - t1) * 1000

        # 3. Generate answer
        user_msg = build_user_message(query, hits)
        t2 = time.perf_counter()
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        generate_ms = (time.perf_counter() - t2) * 1000

        answer = chat_response.choices[0].message.content
        usage = chat_response.usage
        gen_tokens_in = usage.prompt_tokens if usage else 0
        gen_tokens_out = usage.completion_tokens if usage else 0

    finally:
        if own_conn:
            conn.close()

    embed_usd = calc_cost(settings.mistral_embed_model, embed_tokens_in, 0)
    generate_usd = calc_cost(model, gen_tokens_in, gen_tokens_out)

    return RagResponse(
        query=query,
        answer=answer,
        sources=hits,
        model=model,
        timing=TimingBreakdown(
            embed_ms=embed_ms,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            total_ms=embed_ms + retrieve_ms + generate_ms,
        ),
        cost=CostBreakdown(
            embed_usd=embed_usd,
            generate_usd=generate_usd,
            total_usd=embed_usd + generate_usd,
            embed_tokens_in=embed_tokens_in,
            generate_tokens_in=gen_tokens_in,
            generate_tokens_out=gen_tokens_out,
        ),
    )


def print_response(r: RagResponse) -> None:
    """Pretty-print a RagResponse for the CLI."""
    console.rule(f'[bold cyan]Query: "{r.query}"[/bold cyan]')
    console.print(r.answer)
    console.print()
    console.rule("[dim]Sources[/dim]")
    for i, h in enumerate(r.sources, 1):
        console.print(
            f"  [{i}] [cyan]{h.similarity:.3f}[/cyan]  "
            f"[green]{h.source}[/green]/[magenta]{h.platform or '—'}[/magenta]  "
            f"{h.title_en[:80] if h.title_en else (h.title or '')[:80]}"
        )
    console.print()
    console.rule(f"[dim]Timing & cost ({r.model})[/dim]")
    t = r.timing
    c = r.cost
    console.print(
        f"  embed: {t.embed_ms:6.1f}ms  retrieve: {t.retrieve_ms:5.1f}ms  "
        f"generate: {t.generate_ms:6.1f}ms  [bold]total: {t.total_ms:.0f}ms[/bold]"
    )
    console.print(
        f"  tokens: {c.embed_tokens_in} embed-in / {c.generate_tokens_in} gen-in / {c.generate_tokens_out} gen-out  "
        f"[bold]cost: {format_cost(c.total_usd)}[/bold] "
        f"(embed {format_cost(c.embed_usd)} + gen {format_cost(c.generate_usd)})"
    )


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what are the most discussed family or household trends"
    model = "mistral-small-latest"
    if "--large" in sys.argv:
        model = "mistral-large-latest"
    r = generate_rag_response(query, model=model)
    print_response(r)
