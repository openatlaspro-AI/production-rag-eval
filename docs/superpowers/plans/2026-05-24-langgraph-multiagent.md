# LangGraph Multi-Agent RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4-agent LangGraph pipeline (Planner → Researcher → Critic → Writer) that reuses the existing pgvector retriever, returns the same `RagResponse`/`RetrievalHit` dataclasses so the existing eval suite works unchanged, and produces an honest side-by-side comparison vs the single-agent baseline.

**Architecture:** A `StateGraph` with a typed `TypedDict` state carrying `query`, `sub_questions`, accumulated `hits`, `critic_passes`, `draft_answer`, `final_answer`, plus per-agent metric lists. The Researcher node fans out over sub-questions sequentially (one `pgvector_search` call each, k=5). The Critic edge conditionally loops once back to Researcher with refined queries, then proceeds to Writer. All agents call Mistral via `ChatMistralAI` (langchain-mistralai already in deps) for response-metadata token telemetry. Hits accumulate deduped by `doc_id` across the run; cost/timing sums across all agent calls populate a single `RagResponse.cost` / `RagResponse.timing` so the eval modules consume runs unmodified.

**Tech Stack:** Python 3.12, `langgraph>=0.2.0`, existing `langchain-mistralai>=0.2.0`, existing `src.retrieve.pgvector_search`, existing `src.eval.*` modules (no edits), existing `src.pricing.calc_cost`.

---

## File Structure

**Create:**
- `src/frameworks/langgraph_multiagent.py` (~300 LoC) — agent graph + `langgraph_generate_rag_response()` entrypoint matching the LangChain module's signature
- `src/eval/run_langgraph.py` (~250 LoC) — eval runner, near-clone of `run_langchain.py`
- `eval_results/langgraph/` — output dir (created by the runner)
- `eval_results/multiagent_vs_single.md` — comparison doc

**Modify:**
- `pyproject.toml` — add `langgraph>=0.2.0` to `[project.optional-dependencies].langchain` group

**Do NOT modify:**
- `src/retrieve.py`, `src/generate.py`, `src/pricing.py`
- `src/eval/cost.py`, `src/eval/latency.py`, `src/eval/llm_judge.py`, `src/eval/refusal.py`, `src/eval/retrieval.py`, `src/eval/run_baseline.py`, `src/eval/_retry.py`
- `src/frameworks/langchain_rag.py` (reference only)
- `requirements.txt` (Streamlit demo stays single-agent)

---

## State Schema (locks contract for all tasks below)

```python
from typing import TypedDict, Literal
import numpy as np
from src.retrieve import RetrievalHit

class AgentMetrics(TypedDict):
    agent: str                  # "planner" | "researcher" | "critic" | "writer"
    sub_call_idx: int           # 0 for planner/critic/writer; 0..N-1 for researcher per sub-question
    model: str                  # "mistral-small-latest" or "mistral-large-latest"
    tokens_in: int
    tokens_out: int
    latency_ms: float
    usd: float

class GraphState(TypedDict):
    # Inputs (set once)
    query: str
    k: int                      # per-sub-question retrieval k (default 5)
    writer_model: str           # "mistral-small-latest" by default
    helper_model: str           # "mistral-small-latest" — planner/researcher/critic
    conn: object                # psycopg connection (kept out of LangGraph serialization)
    embed_client: object        # Mistral SDK client for embeddings only

    # Planner output
    sub_questions: list[str]    # max 3

    # Researcher output (cumulative across loops)
    hits: list[RetrievalHit]    # deduped by hit.id
    research_notes: list[str]   # one short synthesis per sub-question call

    # Critic output
    critic_verdict: Literal["pass", "needs_more_research", ""]
    refined_queries: list[str]  # populated only on needs_more_research
    critic_passes: int          # 0 → 1 (max), prevents infinite loop

    # Writer output
    final_answer: str

    # Telemetry (appended by every node)
    agent_calls: list[AgentMetrics]
    embed_tokens_in: int        # sum of all mistral-embed input tokens for sub-question retrieval
    embed_latency_ms: float     # sum of all embed-call wall times
    retrieve_latency_ms: float  # sum of all pgvector_search wall times
```

**Why TypedDict over Pydantic:** LangGraph's `StateGraph` accepts any mapping; `TypedDict` avoids validation overhead and lets us stash a `psycopg.Connection` and `Mistral` client on the state without Pydantic config gymnastics (the LangChain module had to fight `ConfigDict(arbitrary_types_allowed=True)` for the same reason).

---

## Agent Prompt Sketches (final wording lives in code)

### Planner
- System: "You decompose a user question about Chinese-media trends into AT MOST 3 focused sub-questions a retrieval system can answer independently. If the question is already atomic, return a single-item list containing the original question verbatim."
- Output: JSON list of strings, parsed with `json.loads` + length-cap 3.
- Fallback: if JSON parse fails, treat the query as atomic (1 sub-question = original query).

### Researcher (called once per sub-question, sequentially)
- For each sub-question: embed → pgvector_search(k=5) → tiny synthesis: "Summarize in 1–2 sentences what the retrieved headlines say about: {sub_question}. Cite as [doc:{id}]. If retrieval found nothing relevant, say 'no coverage found'."
- Synthesis output goes into `research_notes`. Hits go into `state.hits` (dedup by id).

### Critic
- Input: original query + sub_questions + research_notes.
- System: "You are a coverage critic. Decide whether the research notes collectively cover the user's original question. Respond with strict JSON: `{\"verdict\": \"pass\" | \"needs_more_research\", \"refined_queries\": [<=2 strings or []]}`. Only request more research for a CONCRETE gap visible in the notes. If notes already address the question or report 'no coverage found' for a topic genuinely absent from the corpus, return pass."
- Loop guard: if `critic_passes >= 1`, force route to Writer regardless of verdict.

### Writer
- Reuses `src.generate.SYSTEM_PROMPT` verbatim (matches native + LangChain prompts for fair quality comparison).
- User message: same numbered-context format as `src.generate.build_user_message`, fed with the deduped `state.hits` (capped at 10 to control context size — note this in code; the headline comparison runs on a corpus where 10 hits ≈ 2.5K tokens).

---

## Eval Reuse Strategy

- Output JSON schema per run row matches `run_langchain.run_one_query` EXACTLY (so `cost.compute_cost`, `latency.compute_latency`, `llm_judge.run_judge`, `refusal.compute_refusal`, `retrieval.*` all just work).
- `embed_tokens_in` = sum across all sub-question embeddings.
- `generate_tokens_in` / `generate_tokens_out` = sum across ALL agent LLM calls (planner + N researcher synthesizers + critic + writer).
- `embed_usd` + `generate_usd` + `total_usd` = sum across all calls.
- `embed_ms` / `retrieve_ms` / `generate_ms` / `total_ms` = sums across the corresponding stages.
- Per-agent breakdown stored in an **extra** field `agent_calls: list[AgentMetrics]` — ignored by existing eval modules, consumed by the comparison doc generator.
- `retrieved_top_k_ids` = the deduped hit list IDs in their FIRST-SEEN order. This means precision@5/recall@5/MRR semantics shift slightly vs single-agent (the multi-agent path sees MORE documents because it ran multiple queries). The comparison doc flags this explicitly — multi-agent retrieval metrics are NOT strictly comparable to single-agent retrieval metrics on this eval, and we discuss why.

---

## Task 1: Add langgraph dependency

**Files:**
- Modify: `pyproject.toml` (the `[project.optional-dependencies].langchain` block)

- [ ] **Step 1: Edit pyproject.toml**

Edit the existing `langchain = [...]` block to add `"langgraph>=0.2.0"`:

```toml
langchain = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-mistralai>=0.2.0",
    "langgraph>=0.2.0",
]
```

- [ ] **Step 2: Install the new dependency**

Run: `pip install -e ".[langchain]"`
Expected: `Successfully installed langgraph-0.x.x` (or `Requirement already satisfied` if pulled transitively).

- [ ] **Step 3: Verify import works**

Run: `python -c "from langgraph.graph import StateGraph, END; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps(langchain): add langgraph for multi-agent RAG comparison"
```

---

## Task 2: Multi-agent module skeleton + state schema

**Files:**
- Create: `src/frameworks/langgraph_multiagent.py`

- [ ] **Step 1: Write skeleton with state schema, imports, and module docstring**

Create `src/frameworks/langgraph_multiagent.py`:

```python
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
```

- [ ] **Step 2: Verify the file imports cleanly**

Run: `python -c "from src.frameworks.langgraph_multiagent import GraphState, AgentMetrics; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): module skeleton + GraphState typed schema"
```

---

## Task 3: LLM call helper + per-agent metric capture

**Files:**
- Modify: `src/frameworks/langgraph_multiagent.py` (append at end)

- [ ] **Step 1: Add helper that wraps a ChatMistralAI call and returns (content, AgentMetrics)**

Append to `src/frameworks/langgraph_multiagent.py`:

```python
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
```

- [ ] **Step 2: Smoke-test the helper standalone**

Run:
```bash
python -c "
from src.frameworks.langgraph_multiagent import _call_llm
content, m = _call_llm(agent='test', model='mistral-small-latest', system='Reply with one word.', user='Say hi.')
print('content:', content[:50])
print('metrics:', m)
"
```
Expected: a one-word content string + an AgentMetrics dict with non-zero `tokens_in`, `tokens_out`, `latency_ms`, and `usd`. Cost should be in the millicent range (e.g. `~$0.000005`).

- [ ] **Step 3: Commit**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): _call_llm helper with token+latency capture"
```

---

## Task 4: Planner node

**Files:**
- Modify: `src/frameworks/langgraph_multiagent.py` (append)

- [ ] **Step 1: Add planner node and its prompt**

Append:

```python
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
```

- [ ] **Step 2: Smoke-test the planner standalone with a fake state**

Run:
```bash
python -c "
from src.frameworks.langgraph_multiagent import _planner_node
s = {'query': 'what trends connect Chinese economic indicators to US-China trade tensions',
     'helper_model': 'mistral-small-latest', 'agent_calls': []}
out = _planner_node(s)
print('sub_questions:', out['sub_questions'])
print('n calls:', len(out['agent_calls']))
print('cost:', out['agent_calls'][0]['usd'])
"
```
Expected: 1–3 sub-question strings; cost < $0.0001.

- [ ] **Step 3: Commit**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): planner node with JSON-list decomposition + fallback"
```

---

## Task 5: Researcher node (fan-out over sub-questions)

**Files:**
- Modify: `src/frameworks/langgraph_multiagent.py` (append)

- [ ] **Step 1: Add researcher node**

Append:

```python
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
```

- [ ] **Step 2: Smoke-test researcher with a connected DB**

Run:
```bash
python -c "
import psycopg
from pgvector.psycopg import register_vector
from mistralai import Mistral
from src.config import settings
from src.frameworks.langgraph_multiagent import _researcher_node

conn = psycopg.connect(settings.postgres_dsn); register_vector(conn)
client = Mistral(api_key=settings.mistral_api_key)
s = {
    'query': 'unused', 'k': 5,
    'sub_questions': ['what trends about US-China trade tensions'],
    'hits': [], 'research_notes': [], 'agent_calls': [],
    'embed_tokens_in': 0, 'embed_latency_ms': 0.0, 'retrieve_latency_ms': 0.0,
    'helper_model': 'mistral-small-latest',
    'conn': conn, 'embed_client': client,
}
out = _researcher_node(s)
print('n hits:', len(out['hits']))
print('note:', out['research_notes'][0][:120])
print('cost:', out['agent_calls'][0]['usd'])
conn.close()
"
```
Expected: 5 hits, a short note citing `[doc:...]`, cost < $0.0001.

- [ ] **Step 3: Commit**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): researcher node with embed+search+synthesize per sub-question"
```

---

## Task 6: Critic node + conditional edge

**Files:**
- Modify: `src/frameworks/langgraph_multiagent.py` (append)

- [ ] **Step 1: Add critic node**

Append:

```python
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
```

- [ ] **Step 2: Smoke-test critic standalone**

Run:
```bash
python -c "
from src.frameworks.langgraph_multiagent import _critic_node
s = {
    'query': 'what trends about US-China trade tensions',
    'sub_questions': ['what trends about US-China trade tensions'],
    'research_notes': ['Several headlines discuss tariff escalation and rare-earth export controls [doc:361] [doc:371].'],
    'helper_model': 'mistral-small-latest', 'agent_calls': [], 'critic_passes': 0,
}
out = _critic_node(s)
print('verdict:', out['critic_verdict'])
print('refined:', out['refined_queries'])
print('cost:', out['agent_calls'][0]['usd'])
"
```
Expected: verdict is "pass" or "needs_more_research", cost < $0.0001.

- [ ] **Step 3: Commit**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): critic node + conditional loop edge (max 1 loop)"
```

---

## Task 7: Writer node + graph assembly + entrypoint

**Files:**
- Modify: `src/frameworks/langgraph_multiagent.py` (append)

- [ ] **Step 1: Add writer node, graph builder, and entrypoint function**

Append:

```python
def _writer_node(state: GraphState) -> dict:
    # Cap context size; otherwise a 3-sub-question run could feed 15 hits into the writer.
    writer_hits = state["hits"][:MAX_WRITER_CONTEXT_HITS]
    user_msg = build_user_message(state["query"], writer_hits)
    content, metrics = _call_llm(
        agent="writer",
        model=state["writer_model"],
        system=SYSTEM_PROMPT,
        user=user_msg,
    )
    return {
        "final_answer": content,
        "agent_calls": state["agent_calls"] + [metrics],
    }


def _build_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", _planner_node)
    g.add_node("researcher", _researcher_node)
    g.add_node("critic", _critic_node)
    g.add_node("merge_refined", _merge_refined_into_subquestions)
    g.add_node("writer", _writer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "critic")
    g.add_conditional_edges(
        "critic",
        _critic_route,
        {"researcher_loop": "merge_refined", "writer": "writer"},
    )
    g.add_edge("merge_refined", "researcher")
    g.add_edge("writer", END)
    return g.compile()


_GRAPH = _build_graph()


def langgraph_generate_rag_response(
    query: str,
    *,
    k: int = 5,
    writer_model: str = "mistral-small-latest",
    helper_model: str = "mistral-small-latest",
    conn: Any,
    embed_client: Any = None,
    return_agent_metrics: bool = False,
) -> RagResponse | tuple[RagResponse, list[AgentMetrics]]:
    """End-to-end multi-agent RAG. Returns the SAME RagResponse the native + LangChain
    paths return, with cost/timing summed across every agent call.

    If `return_agent_metrics=True`, returns (RagResponse, agent_calls) so the
    eval runner can persist per-agent breakdowns.
    """
    if writer_model not in ALLOWED_MODELS:
        raise ValueError(f"writer_model {writer_model!r} not in allowlist: {sorted(ALLOWED_MODELS)}")
    if helper_model not in ALLOWED_MODELS:
        raise ValueError(f"helper_model {helper_model!r} not in allowlist: {sorted(ALLOWED_MODELS)}")

    own_client = embed_client is None
    if own_client:
        embed_client = Mistral(api_key=settings.mistral_api_key)

    init_state: GraphState = {
        "query": query, "k": k,
        "writer_model": writer_model, "helper_model": helper_model,
        "conn": conn, "embed_client": embed_client,
        "sub_questions": [], "hits": [], "research_notes": [],
        "critic_verdict": "", "refined_queries": [], "critic_passes": 0,
        "final_answer": "",
        "agent_calls": [], "embed_tokens_in": 0,
        "embed_latency_ms": 0.0, "retrieve_latency_ms": 0.0,
    }

    t_wall_start = time.perf_counter()
    final_state = _GRAPH.invoke(init_state)
    total_wall_ms = (time.perf_counter() - t_wall_start) * 1000

    # Aggregate cost/timing across all agent calls + embed/retrieve telemetry
    gen_tokens_in = sum(m["tokens_in"] for m in final_state["agent_calls"])
    gen_tokens_out = sum(m["tokens_out"] for m in final_state["agent_calls"])
    gen_usd = sum(m["usd"] for m in final_state["agent_calls"])
    gen_latency_ms = sum(m["latency_ms"] for m in final_state["agent_calls"])

    embed_tokens_in = final_state["embed_tokens_in"]
    embed_usd = calc_cost(EMBED_MODEL, embed_tokens_in, 0)
    embed_ms = final_state["embed_latency_ms"]
    retrieve_ms = final_state["retrieve_latency_ms"]

    response = RagResponse(
        query=query,
        answer=final_state["final_answer"],
        sources=final_state["hits"],
        model=writer_model,
        timing=TimingBreakdown(
            embed_ms=embed_ms,
            retrieve_ms=retrieve_ms,
            generate_ms=gen_latency_ms,
            # total_ms = wall clock of the whole graph (better than naive sum because
            # researcher embed+search+synthesize stages overlap conceptually but run
            # sequentially in our implementation; wall is the honest number).
            total_ms=total_wall_ms,
        ),
        cost=CostBreakdown(
            embed_usd=embed_usd,
            generate_usd=gen_usd,
            total_usd=embed_usd + gen_usd,
            embed_tokens_in=embed_tokens_in,
            generate_tokens_in=gen_tokens_in,
            generate_tokens_out=gen_tokens_out,
        ),
    )

    if return_agent_metrics:
        return response, final_state["agent_calls"]
    return response
```

- [ ] **Step 2: Smoke-test entrypoint end-to-end on ONE query**

Run:
```bash
python -c "
import psycopg, json
from pgvector.psycopg import register_vector
from src.config import settings
from src.frameworks.langgraph_multiagent import langgraph_generate_rag_response

conn = psycopg.connect(settings.postgres_dsn); register_vector(conn)
r, calls = langgraph_generate_rag_response(
    'what trends connect Chinese economic indicators to US-China trade tensions',
    conn=conn, return_agent_metrics=True,
)
print('=== ANSWER ===')
print(r.answer)
print('=== AGENT TRACE ===')
for c in calls:
    print(f\"  {c['agent']:11s}  idx={c['sub_call_idx']}  in={c['tokens_in']:4d}  out={c['tokens_out']:4d}  {c['latency_ms']:6.0f}ms  \${c['usd']:.6f}\")
print(f'=== TOTALS ===')
print(f\"  n_sources={len(r.sources)}  embed_tokens={r.cost.embed_tokens_in}\")
print(f\"  gen_in={r.cost.generate_tokens_in}  gen_out={r.cost.generate_tokens_out}\")
print(f\"  total cost=\${r.cost.total_usd:.5f}\")
print(f\"  total wall={r.timing.total_ms:.0f}ms\")
conn.close()
" 2>&1 | tee /tmp/langgraph_smoke.log
```
Expected: an answer with [N] citations; agent trace showing planner → 1-3 researchers → critic → optionally researcher-loop → critic-2 → writer; cost in the **$0.001–$0.005 range**.

- [ ] **Step 3: PAUSE — show smoke output to user**

Halt and surface the smoke output. Report: total cost per query, total wall time per query, projected 30-query × 2-model cost (cost × 60), projected 30-query × 2-model wall time at workers=1, and the critic loop trigger flag (did it loop?).

**Decision gate:** if cost > $0.005/query or projected full-eval cost > $0.10, STOP and re-plan (e.g. cap sub-questions to 2, drop critic, or shorten researcher synthesis).

- [ ] **Step 4: Commit (after user confirms smoke is acceptable)**

```bash
git add src/frameworks/langgraph_multiagent.py
git commit -m "feat(langgraph): writer node + graph assembly + entrypoint with per-agent metrics"
```

---

## Task 8: Eval runner (near-clone of run_langchain.py)

**Files:**
- Create: `src/eval/run_langgraph.py`

- [ ] **Step 1: Create the runner**

Create `src/eval/run_langgraph.py`. Start by copying the entire contents of `src/eval/run_langchain.py` as the base, then make exactly these substitutions:

1. Module docstring: replace "LangChain version" with "LangGraph multi-agent version" and update the description to mention the 4-agent graph + per-agent breakdown.
2. Import: replace `from src.frameworks.langchain_rag import langchain_generate_rag_response` with `from src.frameworks.langgraph_multiagent import langgraph_generate_rag_response`.
3. Output dir: `OUTPUT_DIR = REPO_ROOT / "eval_results" / "langgraph"`.
4. Inside `run_one_query`, replace the `with_retry_broad` body with:

   ```python
   with pool.connection() as conn:
       result = with_retry_broad(lambda: langgraph_generate_rag_response(
           q["query"],
           k=k,
           writer_model=model,
           helper_model="mistral-small-latest",
           conn=conn,
           return_agent_metrics=True,
       ))
       r, agent_calls = result
   ```

5. In the returned dict, add: `"agent_calls": agent_calls,` (after `relevant_doc_ids`). This carries the per-agent breakdown through to the consolidated report.
6. In the `compute_retrieval_metrics` `framework` field: change `"langchain"` → `"langgraph"`.
7. In `consolidated["metadata"]`: change `"framework": "langchain"` → `"framework": "langgraph_multiagent"`; update `note_embed_tokens` to: `"embed_tokens_in is summed across all sub-question embeddings (1-3 per query, plus 1-2 more if the critic looped)."`; remove the LangChain-specific MistralAIEmbeddings note (we use the native Mistral SDK for embeddings here).
8. In progress bar text: replace `"LangChain RAG"` with `"LangGraph multi-agent RAG"`.
9. The `GEN_MODELS` list stays as `["mistral-small-latest", "mistral-large-latest"]` — `model` here represents the **writer model**; helper agents stay on mistral-small.
10. Add a `--smoke` CLI flag that, when present, runs only the FIRST query against `mistral-small-latest` and writes nothing to disk — just prints cost + trace.

Smoke-flag implementation (add at top of `main()`):

```python
if "--smoke" in sys.argv:
    queries = load_eval_set()[:1]
    runs = run_all_queries(queries, ["mistral-small-latest"], k=5, workers=1)
    console.print_json(data=runs[0])
    console.print(f"[bold]Smoke cost:[/bold] ${runs[0]['total_usd']:.5f}  "
                  f"wall={runs[0]['total_ms']:.0f}ms")
    return
```

- [ ] **Step 2: Smoke-test the runner**

Run: `python -m src.eval.run_langgraph --smoke`
Expected: prints a single run JSON + cost line; no files written to `eval_results/langgraph/`.

- [ ] **Step 3: Verify the langgraph output dir is fresh (no leftover artifacts from earlier work)**

Run: `ls eval_results/langgraph/ 2>/dev/null || echo "(no dir yet)"`
Expected: `(no dir yet)` or empty — if anything is there, delete it before the full eval.

- [ ] **Step 4: PAUSE — show user the smoke output + go/no-go for full eval**

Report: smoke cost × 60 = projected full eval cost. Estimated wall: smoke wall × 60 (workers=1) seconds. If projected cost > $0.10 OR projected wall > 30 min, STOP and ask user before running full eval.

- [ ] **Step 5: Run the full eval (only after user gives explicit go)**

Run: `python -m src.eval.run_langgraph 2>&1 | tee /tmp/langgraph_full_eval.log`
Expected: completes in under 30 minutes; writes 6 JSONs to `eval_results/langgraph/`; final cost line is < $0.10.

- [ ] **Step 6: Verify output completeness**

Run:
```bash
ls -la eval_results/langgraph/
python -c "
import json
c = json.load(open('eval_results/langgraph/consolidated_report.json'))
print('n_runs:', len(c['raw_runs']))
print('total_eval_usd:', c['cost']['total_eval_usd'])
print('framework:', c['metadata']['framework'])
print('first run has agent_calls:', 'agent_calls' in c['raw_runs'][0])
"
```
Expected: 60 runs (30 queries × 2 models), total cost < $0.10, framework `"langgraph_multiagent"`, agent_calls present.

- [ ] **Step 7: Commit**

```bash
git add src/eval/run_langgraph.py eval_results/langgraph/
git commit -m "eval(langgraph): full multi-agent eval suite + 60-run results"
```

---

## Task 9: Comparison doc

**Files:**
- Create: `eval_results/multiagent_vs_single.md`

- [ ] **Step 1: Compute the side-by-side numbers**

Run a one-off script to extract comparable metrics from both consolidated reports:

```bash
python -c "
import json
single = json.load(open('eval_results/consolidated_report.json'))
multi  = json.load(open('eval_results/langgraph/consolidated_report.json'))
print('=== SINGLE-AGENT ===')
print(f\"  total_eval_usd: \${single['cost']['total_eval_usd']:.5f}\")
print(f\"  latency p50/p95 (small): {single['latency']['per_model']['mistral-small-latest']['total_ms']['p50']:.0f} / {single['latency']['per_model']['mistral-small-latest']['total_ms']['p95']:.0f}\")
print(f\"  llm_judge composite: {single['llm_judge']}\" )
print('=== MULTI-AGENT ===')
print(f\"  total_eval_usd: \${multi['cost']['total_eval_usd']:.5f}\")
print(f\"  latency p50/p95 (small): {multi['latency']['per_model']['mistral-small-latest']['total_ms']['p50']:.0f} / {multi['latency']['per_model']['mistral-small-latest']['total_ms']['p95']:.0f}\")
# critic loop trigger rate
runs = multi['raw_runs']
looped = sum(1 for r in runs if sum(1 for c in r['agent_calls'] if c['agent']=='critic') > 1)
print(f\"  critic loop trigger rate: {looped}/{len(runs)}\")
# avg agents per query
avg_calls = sum(len(r['agent_calls']) for r in runs) / len(runs)
print(f\"  avg agent calls per query: {avg_calls:.1f}\")
# per-agent cost breakdown
from collections import defaultdict
agent_usd = defaultdict(float)
agent_ms = defaultdict(float)
for r in runs:
    for c in r['agent_calls']:
        agent_usd[c['agent']] += c['usd']
        agent_ms[c['agent']]  += c['latency_ms']
for a in ['planner','researcher','critic','writer']:
    print(f\"    {a:11s}  \${agent_usd[a]:.5f}  {agent_ms[a]/len(runs):.0f}ms avg\")
" 2>&1 | tee /tmp/multiagent_compare.log
```

- [ ] **Step 2: Write the comparison doc**

Create `eval_results/multiagent_vs_single.md` following the structure of `eval_results/langchain_vs_native.md`. Required sections:

1. **Headline finding** (one paragraph): cost multiplier vs single-agent, latency multiplier, quality delta on LLM-judge, critic loop trigger rate. Use the actual numbers from Step 1.
2. **Recommendation** (one paragraph): when to use multi-agent vs when single-agent wins. Specific criteria (e.g. "multi-agent earns its cost when queries genuinely decompose AND coverage gaps are detectable from synthesis notes alone — not for atomic factual lookups").
3. **Side-by-side metrics table** with caveat box about retrieval-metric non-comparability (the multi-agent path sees more docs because it issues multiple queries; precision@5 / recall@5 on the deduped top hits are NOT apples-to-apples with single-agent k=5).
4. **Per-agent cost & latency breakdown table** (planner / researcher / critic / writer rows: total USD, mean latency, mean tokens-in, mean tokens-out).
5. **Critic loop analysis**: how often did it loop, on which query categories, what was the LLM-judge delta on looped vs non-looped queries.
6. **Honest engineer's notes** — 4–6 concrete LangGraph DX frustrations from THIS build. Examples to draw from (use real ones encountered during implementation): state-merge semantics (do dict returns merge or replace? — verify behavior), conditional-edge return-string-must-match-dict-key cargo culting, lack of native loop guards (we had to track `critic_passes` ourselves), serialization warnings if you stash non-JSON-serializable objects (psycopg.Connection) on state, debugging without LangSmith means manual print-tracing, etc.
7. **When to use multi-agent** — bullet list with criteria.

- [ ] **Step 3: PAUSE — show doc + full file list + proposed commit message to user**

List the changed files, show the multiagent_vs_single.md doc, and propose the commit message. WAIT for explicit user "go" before committing.

- [ ] **Step 4: Commit (after user approval)**

```bash
git add eval_results/multiagent_vs_single.md
git commit -m "docs(eval): multi-agent vs single-agent comparison with per-agent cost breakdown"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| 4 agents using LangGraph StateGraph | Tasks 2, 4, 5, 6, 7 |
| State schema with typed fields (Pydantic OR TypedDict) | Task 2 (TypedDict) |
| Planner → decompose, max 3 sub-questions, mistral-small | Task 4 (MAX_SUB_QUESTIONS=3, helper_model default) |
| Researcher → per-sub-question retrieve via existing pgvector_search | Task 5 |
| Critic → judge coverage, max 1 loop, mistral-small | Task 6 (MAX_CRITIC_LOOPS=1) |
| Writer → mistral-small default, mistral-large optional | Task 7 (writer_model param) |
| Returns RagResponse with all hits deduped + summed tokens | Task 7 (seen_ids dedup in Task 5, aggregation in entrypoint) |
| Per-agent metrics tracked | Task 3 (_call_llm), surfaced via return_agent_metrics |
| Reuse src.retrieve.pgvector_search | Task 5 |
| Reuse src.generate prompts (SYSTEM_PROMPT, build_user_message) | Task 7 |
| Return same RagResponse + RetrievalHit | Task 7 |
| Output to eval_results/langgraph/* JSONs | Task 8 |
| Comparison doc at eval_results/multiagent_vs_single.md | Task 9 |
| Default helper = mistral-small, writer configurable | Task 7 |
| run_langgraph.py near-clone of run_langchain.py | Task 8 |
| --smoke flag | Task 8 step 1 |
| Add langgraph to pyproject [langchain] | Task 1 |
| Don't add to streamlit requirements.txt | (out of scope — confirmed by not touching it) |
| Don't edit src/retrieve.py, src/generate.py, src/eval/* | (no task does this — verified) |
| No tool calling / function calling | (none in graph design) |
| SMOKE TEST one query before full eval, show trace + cost | Task 7 step 3, Task 8 step 4 |
| PAUSE before full eval | Task 8 step 4 |
| Budget cap $0.10 total | Task 7 step 3 decision gate, Task 8 step 4 |
| Wall-clock estimate before full eval | Task 8 step 4 |
| PAUSE before committing comparison doc | Task 9 step 3 |
| Honest engineer's notes 4-6 LangGraph DX frustrations | Task 9 step 2 section 6 |
| When-to-use multi-agent criteria, not cheerleading | Task 9 step 2 sections 2 + 7 |

All spec requirements covered.

### Placeholder scan

- No "TBD" / "implement later" / "fill in details" in any task.
- Every code step shows complete code.
- Every command step shows the exact command and expected output.
- Numeric thresholds ($0.005/query smoke gate, $0.10 budget cap, 30-min wall cap) explicit.

### Type consistency

- `AgentMetrics` defined in Task 2, used identically in Tasks 3, 4, 5, 6, 7.
- `GraphState` field names match across all node functions (verified: `helper_model`, `writer_model`, `sub_questions`, `research_notes`, `agent_calls`, `embed_tokens_in`, `embed_latency_ms`, `retrieve_latency_ms`, `critic_passes`, `critic_verdict`, `refined_queries`, `final_answer`).
- `_call_llm` signature consistent across all 4 agent nodes.
- `langgraph_generate_rag_response` signature parallels `langchain_generate_rag_response` (writer_model replaces model; adds helper_model, return_agent_metrics).

Plan is internally consistent.
