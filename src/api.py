"""FastAPI endpoint for RAG queries.

POST /query   — main RAG endpoint with full instrumentation
GET  /health  — readiness probe (Mistral key present + Postgres reachable)

Uses a connection pool + shared Mistral client across requests for low per-query
overhead. Pool size is configurable; default min=1 max=5.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from mistralai import Mistral
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from src.config import settings
from src.generate import ALLOWED_MODELS, generate_rag_response
from src.retrieve import RetrievalHit

log = logging.getLogger("production-rag-eval.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


# ---- Request / response models ----

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=20)
    model: str = Field("mistral-small-latest")
    source: str | None = Field(None, description="Filter: 'trend' or 'product'")


class SourceItem(BaseModel):
    rank: int
    similarity: float
    title_en: str | None
    title: str | None
    platform: str | None
    source: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_hit(cls, position: int, hit: RetrievalHit) -> "SourceItem":
        return cls(
            rank=position,
            similarity=hit.similarity,
            title_en=hit.title_en,
            title=hit.title,
            platform=hit.platform,
            source=hit.source,
            metadata=hit.metadata,
        )


class TimingBlock(BaseModel):
    embed_ms: float
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class CostBlock(BaseModel):
    embed_usd: float
    generate_usd: float
    total_usd: float
    embed_tokens_in: int
    generate_tokens_in: int
    generate_tokens_out: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    model: str
    sources: list[SourceItem]
    timing: TimingBlock
    cost: CostBlock


# ---- App lifecycle ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open Mistral client + Postgres pool on startup; close on shutdown."""
    log.info("Starting up — opening Mistral client and Postgres pool")

    app.state.mistral = Mistral(api_key=settings.mistral_api_key)

    def configure(conn):
        register_vector(conn)

    app.state.pool = ConnectionPool(
        settings.postgres_dsn,
        min_size=1,
        max_size=5,
        configure=configure,
        open=True,
    )
    app.state.pool.wait()  # wait for at least min_size connections to be ready
    log.info("Pool ready (min=1, max=5)")

    try:
        yield
    finally:
        log.info("Shutting down — closing pool")
        app.state.pool.close()


app = FastAPI(
    title="Production RAG Eval",
    version="0.1.0",
    description="RAG over a Chinese-news trend corpus with reproducible eval suite. Mistral + pgvector + FastAPI.",
    lifespan=lifespan,
)


# ---- Endpoints ----

@app.get("/health")
def health():
    """Readiness probe — confirms Mistral client built and pool is open."""
    pool_open = app.state.pool is not None and not app.state.pool.closed
    return {
        "status": "ok" if pool_open else "degraded",
        "service": "production-rag-eval",
        "version": "0.1.0",
        "mistral_configured": bool(settings.mistral_api_key),
        "pool_open": pool_open,
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """Run a RAG query end-to-end. Returns answer + sources + timing + cost."""
    if req.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model must be one of {sorted(ALLOWED_MODELS)}",
        )
    if req.source not in (None, "trend", "product"):
        raise HTTPException(status_code=400, detail="source must be 'trend', 'product', or null")

    with app.state.pool.connection() as conn:
        r = generate_rag_response(
            req.query,
            k=req.k,
            model=req.model,
            source=req.source,
            client=app.state.mistral,
            conn=conn,
        )

    return QueryResponse(
        query=r.query,
        answer=r.answer,
        model=r.model,
        sources=[SourceItem.from_hit(i + 1, h) for i, h in enumerate(r.sources)],
        timing=TimingBlock(
            embed_ms=r.timing.embed_ms,
            retrieve_ms=r.timing.retrieve_ms,
            generate_ms=r.timing.generate_ms,
            total_ms=r.timing.total_ms,
        ),
        cost=CostBlock(
            embed_usd=r.cost.embed_usd,
            generate_usd=r.cost.generate_usd,
            total_usd=r.cost.total_usd,
            embed_tokens_in=r.cost.embed_tokens_in,
            generate_tokens_in=r.cost.generate_tokens_in,
            generate_tokens_out=r.cost.generate_tokens_out,
        ),
    )
