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
