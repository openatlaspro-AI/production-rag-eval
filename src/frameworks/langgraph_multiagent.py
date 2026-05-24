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
