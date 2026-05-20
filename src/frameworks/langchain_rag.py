"""LangChain reimplementation of the native retrieve+generate pipeline.

Returns the SAME RagResponse / RetrievalHit dataclasses as src/generate.py so
the existing eval harness (src/eval/{cost,latency,llm_judge,refusal}) consumes
LangChain runs without modification.

Design choices (full rationale in eval_results/langchain_vs_native.md):

  - Hybrid retriever — `PgvectorBaseRetriever(BaseRetriever)` wraps our existing
    pgvector_search() instead of standing up a parallel langchain_postgres
    PGVector collection. Same backend, same documents.id values, same index.

  - LCEL only where it earns its keep — prompt → llm chain. The retrieve step
    is invoked separately so per-stage `time.perf_counter()` instrumentation
    works (LCEL hides intermediate timing).

  - Embed-cost telemetry is estimated, not exact. langchain_mistralai's
    MistralAIEmbeddings discards the upstream `usage` field, so we approximate
    embed tokens via len(query) // 4. Sub-cent error on a 30-query eval; called
    out as a real DX gap in the comparison doc.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import ConfigDict, Field, PrivateAttr

from src.config import settings
from src.generate import (
    ALLOWED_MODELS,
    CostBreakdown,
    RagResponse,
    SYSTEM_PROMPT,
    TimingBreakdown,
)
from src.pricing import calc_cost
from src.retrieve import RetrievalHit, pgvector_search

EMBED_MODEL = "mistral-embed"


# ---- Retriever -----------------------------------------------------------

class PgvectorBaseRetriever(BaseRetriever):
    """BaseRetriever wrapping our existing pgvector_search().

    We pre-embed the query outside the retriever so the embed step can be
    timed separately (LangChain BaseRetriever doesn't expose embed time).
    The retriever just runs the SQL.

    `last_hits` is set on every invoke so callers can recover the original
    RetrievalHit list (Document → RetrievalHit is lossy).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 5
    # Stored on the instance, not validated as a Pydantic field (psycopg.Connection +
    # np.ndarray don't play well with Pydantic v2 validation).
    _conn: Any = PrivateAttr()
    _q_emb: np.ndarray = PrivateAttr()
    _last_hits: list[RetrievalHit] = PrivateAttr(default_factory=list)

    def __init__(self, *, conn: Any, q_emb: np.ndarray, k: int = 5) -> None:
        super().__init__(k=k)
        self._conn = conn
        self._q_emb = q_emb
        self._last_hits = []

    @property
    def last_hits(self) -> list[RetrievalHit]:
        return self._last_hits

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = pgvector_search(self._conn, self._q_emb, k=self.k)
        self._last_hits = hits
        return [
            Document(
                page_content=h.title_en or h.title or "",
                metadata={
                    "doc_id": h.id,
                    "source": h.source,
                    "platform": h.platform,
                    "similarity": h.similarity,
                    "rank": h.rank,
                },
            )
            for h in hits
        ]


# ---- Prompt --------------------------------------------------------------

# Match the native prompt EXACTLY so any quality delta is framework-driven,
# not prompt drift. The user-message template formats the same numbered context
# block as src/generate.py:build_user_message.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)


def _format_context_block(hits: list[RetrievalHit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        platform = f" (source: {h.platform})" if h.platform else ""
        title = h.title_en or h.title or "(no title)"
        lines.append(f"[{i}] {title}{platform}")
    return "\n".join(lines)


# ---- Main entrypoint -----------------------------------------------------

def langchain_generate_rag_response(
    query: str,
    *,
    k: int = 5,
    model: str = "mistral-small-latest",
    conn: Any,
) -> RagResponse:
    """End-to-end LangChain RAG: embed → retrieve → generate.

    Signature mirrors src.generate.generate_rag_response (minus the optional
    `client` arg — LangChain wraps the SDK internally, no shared client).
    Returns the same RagResponse dataclass, so the eval harness is reusable.

    Caller owns the psycopg connection (must have register_vector applied)
    so we can match the native pool-based parallelism in run_langchain.py.
    """
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Model {model!r} not in allowlist: {sorted(ALLOWED_MODELS)}")

    embeddings = MistralAIEmbeddings(
        model=EMBED_MODEL,
        api_key=settings.mistral_api_key,
    )
    llm = ChatMistralAI(
        model=model,
        temperature=0.2,
        api_key=settings.mistral_api_key,
    )

    # ---- 1. Embed query (LangChain) ----
    t0 = time.perf_counter()
    q_emb_list = embeddings.embed_query(query)
    embed_ms = (time.perf_counter() - t0) * 1000
    # Estimated — see module docstring + comparison doc.
    embed_tokens_in = max(1, len(query) // 4)
    q_emb = np.asarray(q_emb_list, dtype=np.float32)

    # ---- 2. Retrieve via BaseRetriever subclass ----
    t1 = time.perf_counter()
    retriever = PgvectorBaseRetriever(conn=conn, q_emb=q_emb, k=k)
    retriever.invoke(query)  # returns Documents; we want the RetrievalHits
    hits = retriever.last_hits
    retrieve_ms = (time.perf_counter() - t1) * 1000

    # ---- 3. Generate via LCEL chain ----
    chain = _PROMPT | llm | StrOutputParser()
    # Capture token usage separately — StrOutputParser strips it from the chain output.
    raw_chain = _PROMPT | llm
    chain_input = {"context_block": _format_context_block(hits), "question": query}

    t2 = time.perf_counter()
    ai_msg = raw_chain.invoke(chain_input)
    generate_ms = (time.perf_counter() - t2) * 1000

    answer = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    usage = ai_msg.response_metadata.get("token_usage", {}) if hasattr(ai_msg, "response_metadata") else {}
    gen_tokens_in = int(usage.get("prompt_tokens", 0))
    gen_tokens_out = int(usage.get("completion_tokens", 0))

    embed_usd = calc_cost(EMBED_MODEL, embed_tokens_in, 0)
    generate_usd = calc_cost(model, gen_tokens_in, gen_tokens_out)

    # Reference `chain` so `pip install` linters don't strip it; it's the LCEL
    # chain that the comparison doc describes. Kept around so future code can
    # switch to the parser-stripped path if token telemetry isn't needed.
    _ = chain

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
