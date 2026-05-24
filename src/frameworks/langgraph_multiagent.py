"""LangGraph multi-agent RAG over the same pgvector backend as native + LangChain.

Architecture:
    Query → Planner → Researcher (per sub-question) → Critic → [loop≤1] → Writer → RagResponse

Reuses (does not modify):
  - src.retrieve.pgvector_search — same backend as native + LangChain runs
  - src.generate.SYSTEM_PROMPT — same writer prompt for fair LLM-judge comparison
  - src.generate.{RagResponse, TimingBreakdown, CostBreakdown} — eval-suite contract
  - src.pricing.calc_cost — same cost arithmetic

Per-agent metrics are tracked in state.agent_calls and surfaced via the optional
`return_agent_metrics=True` flag on the entrypoint, so the eval row carries the
standard RagResponse fields AND the per-agent breakdown for the comparison doc.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal, TypedDict

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, StateGraph
from mistralai import Mistral

from src.config import settings
from src.generate import (
    ALLOWED_MODELS,
    CostBreakdown,
    RagResponse,
    SYSTEM_PROMPT,
    TimingBreakdown,
    build_user_message,
)
from src.pricing import calc_cost
from src.retrieve import RetrievalHit, embed_query, pgvector_search

EMBED_MODEL = "mistral-embed"
MAX_SUB_QUESTIONS = 3
MAX_CRITIC_LOOPS = 1
MAX_WRITER_CONTEXT_HITS = 10


class AgentMetrics(TypedDict):
    agent: str
    sub_call_idx: int
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    usd: float


class GraphState(TypedDict):
    # Inputs
    query: str
    k: int
    writer_model: str
    helper_model: str
    conn: Any
    embed_client: Any

    # Planner
    sub_questions: list[str]

    # Researcher
    hits: list[RetrievalHit]
    research_notes: list[str]

    # Critic
    critic_verdict: Literal["pass", "needs_more_research", ""]
    refined_queries: list[str]
    critic_passes: int

    # Writer
    final_answer: str

    # Telemetry
    agent_calls: list[AgentMetrics]
    embed_tokens_in: int
    embed_latency_ms: float
    retrieve_latency_ms: float


def _call_llm(
    *, agent: str, model: str, system: str, user: str, sub_call_idx: int = 0
) -> tuple[str, AgentMetrics]:
    """Single ChatMistralAI call with token + latency capture.

    Constructs a fresh ChatMistralAI per call. The SDK is cheap to instantiate
    and this avoids threading a client through state. Temperature fixed at 0.2
    to match the native pipeline.
    """
    llm = ChatMistralAI(model=model, temperature=0.2, api_key=settings.mistral_api_key)
    t0 = time.perf_counter()
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    latency_ms = (time.perf_counter() - t0) * 1000
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    usage = msg.response_metadata.get("token_usage", {}) if hasattr(msg, "response_metadata") else {}
    tokens_in = int(usage.get("prompt_tokens", 0))
    tokens_out = int(usage.get("completion_tokens", 0))
    metrics: AgentMetrics = {
        "agent": agent,
        "sub_call_idx": sub_call_idx,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_ms": latency_ms,
        "usd": calc_cost(model, tokens_in, tokens_out),
    }
    return content, metrics


_PLANNER_SYSTEM = (
    "You decompose a user question about Chinese-media trends into AT MOST "
    f"{MAX_SUB_QUESTIONS} focused sub-questions that a retrieval system can answer "
    "independently. If the question is already atomic, return a single-item list "
    "containing the original question verbatim. Respond with STRICT JSON: a list of "
    "strings, no prose, no markdown fences."
)


def _planner_node(state: GraphState) -> dict:
    user = f"User question: {state['query']}\n\nReturn the JSON list now."
    content, metrics = _call_llm(
        agent="planner",
        model=state["helper_model"],
        system=_PLANNER_SYSTEM,
        user=user,
    )
    sub_questions: list[str]
    try:
        parsed = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        if isinstance(parsed, list) and all(isinstance(s, str) and s.strip() for s in parsed):
            sub_questions = [s.strip() for s in parsed][:MAX_SUB_QUESTIONS]
        else:
            sub_questions = [state["query"]]
    except (json.JSONDecodeError, ValueError):
        sub_questions = [state["query"]]
    if not sub_questions:
        sub_questions = [state["query"]]
    return {
        "sub_questions": sub_questions,
        "agent_calls": state["agent_calls"] + [metrics],
    }


_RESEARCHER_SYSTEM = (
    "Summarize in 1–2 sentences what the retrieved headlines say about the question. "
    "Cite as [doc:{id}]. If the retrieved items do not address the question, "
    "say exactly: 'no coverage found'. Do not invent details."
)


def _researcher_node(state: GraphState) -> dict:
    # On first pass, work through state['sub_questions']. On second pass (after
    # critic loop), state['refined_queries'] are appended to sub_questions and
    # processed; we only run those that haven't been processed yet.
    already_processed = len(state["research_notes"])
    queries_to_run = state["sub_questions"][already_processed:]

    new_hits: list[RetrievalHit] = []
    new_notes: list[str] = []
    new_calls: list[AgentMetrics] = []
    embed_tokens_added = 0
    embed_ms_added = 0.0
    retrieve_ms_added = 0.0

    seen_ids = {h.id for h in state["hits"]}

    for idx, sq in enumerate(queries_to_run, start=already_processed):
        # Embed sub-question
        t0 = time.perf_counter()
        q_emb, tok_in = embed_query(state["embed_client"], sq)
        embed_ms_added += (time.perf_counter() - t0) * 1000
        embed_tokens_added += tok_in

        # Retrieve
        t1 = time.perf_counter()
        sq_hits = pgvector_search(state["conn"], q_emb, k=state["k"])
        retrieve_ms_added += (time.perf_counter() - t1) * 1000

        # Dedup
        for h in sq_hits:
            if h.id not in seen_ids:
                seen_ids.add(h.id)
                new_hits.append(h)

        # Synthesize
        ctx_lines = [
            f"[doc:{h.id}] {h.title_en or h.title or '(no title)'}"
            for h in sq_hits
        ]
        user = f"Question: {sq}\n\nRetrieved:\n" + "\n".join(ctx_lines)
        note, metrics = _call_llm(
            agent="researcher",
            model=state["helper_model"],
            system=_RESEARCHER_SYSTEM,
            user=user,
            sub_call_idx=idx,
        )
        new_notes.append(note)
        new_calls.append(metrics)

    return {
        "hits": state["hits"] + new_hits,
        "research_notes": state["research_notes"] + new_notes,
        "agent_calls": state["agent_calls"] + new_calls,
        "embed_tokens_in": state["embed_tokens_in"] + embed_tokens_added,
        "embed_latency_ms": state["embed_latency_ms"] + embed_ms_added,
        "retrieve_latency_ms": state["retrieve_latency_ms"] + retrieve_ms_added,
    }


_CRITIC_SYSTEM = (
    "You are a coverage critic. Decide whether the research notes collectively cover "
    "the user's original question. Respond with STRICT JSON: "
    '{"verdict": "pass" | "needs_more_research", "refined_queries": [<=2 strings or []]}. '
    "Only request more research for a CONCRETE gap visible in the notes. If notes "
    "already address the question or report 'no coverage found' for a topic genuinely "
    "absent from the corpus, return verdict='pass' with refined_queries=[]."
)


def _critic_node(state: GraphState) -> dict:
    notes_block = "\n".join(
        f"[Q{i+1}: {sq}]\n{note}"
        for i, (sq, note) in enumerate(zip(state["sub_questions"], state["research_notes"]))
    )
    user = (
        f"Original question: {state['query']}\n\n"
        f"Research notes so far:\n{notes_block}\n\n"
        "Return the JSON verdict now."
    )
    content, metrics = _call_llm(
        agent="critic",
        model=state["helper_model"],
        system=_CRITIC_SYSTEM,
        user=user,
    )

    verdict: Literal["pass", "needs_more_research"] = "pass"
    refined: list[str] = []
    try:
        parsed = json.loads(content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        if isinstance(parsed, dict):
            v = parsed.get("verdict", "pass")
            if v == "needs_more_research":
                verdict = "needs_more_research"
            rq = parsed.get("refined_queries", [])
            if isinstance(rq, list):
                refined = [s.strip() for s in rq if isinstance(s, str) and s.strip()][:2]
    except (json.JSONDecodeError, ValueError):
        pass  # default to pass

    return {
        "critic_verdict": verdict,
        "refined_queries": refined,
        "critic_passes": state["critic_passes"] + 1,
        "agent_calls": state["agent_calls"] + [metrics],
    }


def _critic_route(state: GraphState) -> str:
    """Conditional edge: loop back to researcher at most MAX_CRITIC_LOOPS times."""
    if (
        state["critic_verdict"] == "needs_more_research"
        and state["critic_passes"] <= MAX_CRITIC_LOOPS
        and state["refined_queries"]
    ):
        return "researcher_loop"
    return "writer"


def _merge_refined_into_subquestions(state: GraphState) -> dict:
    """Pre-researcher-loop hop: append refined_queries to sub_questions list so the
    researcher picks up only the new ones (via the already_processed offset)."""
    return {
        "sub_questions": state["sub_questions"] + state["refined_queries"],
        "refined_queries": [],
    }
