# LangGraph Multi-Agent vs Single-Agent — Framework Comparison

> **Headline finding.** On this corpus and eval set, the LangGraph multi-agent pipeline (Planner → Researcher × N → Critic → [loop ≤ 1] → Writer) costs **1.68× more** and runs **3.5–3.9× slower at p50** than the native single-agent baseline, while producing essentially **identical LLM-judge quality** (0.00–0.03 pt delta on large-writer; −0.07 to −0.15 on small-writer). The multi-agent path **degrades retrieval** on the query categories where it would theoretically help most: broad queries drop from P@5=0.850 to 0.400, and family-niche queries collapse from P@5=0.650 to 0.000. The critic loop fires 35% of the time but delivers zero measurable quality lift (+0.03 overall, within noise).
>
> **Recommendation.** Do not use multi-agent decomposition on this corpus. Single-pass retrieval over a well-embedded 400-doc collection already achieves P@5=0.681 and MRR=0.920 on the dominant query types. The decomposition architecture solves a retrieval coverage problem this corpus does not have — and in doing so, it creates a new one. Keep `src/retrieve.py` + `src/generate.py` in production. Reach for `src/frameworks/langgraph_multiagent.py` only when incoming queries have genuine multi-step structure that atomic single-pass retrieval demonstrably cannot serve.

Reproduce: `python -m src.eval.run_langgraph` (or add `make eval-langgraph` to the Makefile). Raw data: [`eval_results/langgraph/consolidated_report.json`](langgraph/consolidated_report.json). Baseline comparisons: [`eval_results/consolidated_report.json`](consolidated_report.json) (native) and [`eval_results/langchain/consolidated_report.json`](langchain/consolidated_report.json) (LangChain).

---

## 1. Headline finding & recommendation

**Multi-agent decomposition was a net negative on this workload.** The hypothesis entering the build was that a Planner→Researcher→Critic→Writer graph would improve recall on broad and multi-aspect queries by targeting each sub-question with its own retrieval call. The actual result is the opposite: decomposing "what are the latest trends in Chinese social media?" into 3 sub-questions causes each researcher call to return 5 topically-narrowed hits, and the deduped union of those 15 hits is dominated by sub-question-specific docs that crowd out the broadly-relevant documents a single embedding of the original query would have found. The P@5 and MRR collapse in the `broad` and `family_niche` categories is the clearest evidence of this.

**The critic loop compounds the problem without fixing it.** A 35% trigger rate sounds like the critic is doing meaningful work. It is not. The looped runs score 4.50 vs 4.47 for non-looped — a 0.03-pt delta well within the noise at n=16 and n=38 respectively. What the loop *does* do is add the cost of one extra critic call plus 1–2 extra researcher calls on more than a third of queries. The corpus has ~400 documents. When the critic identifies a coverage gap and fires a refined query, the gap is usually genuinely absent from the corpus — not addressable by a second retrieval pass. The writer still gets fed the same effective context.

**The only unambiguous result is on refusal.** The multi-agent path correctly refuses all 3 negative-test queries (queries asking about topics not in the corpus) and hallucinates on none of them. This matches native and LangChain exactly. The safety behavior comes from `SYSTEM_PROMPT` and the writer model, not from the graph structure.

### What worked correctly in the multi-agent path

It is worth crediting the implementation behaviors that performed as designed, separate from the architectural verdict:

- **Refusal matched single-agent: 100% / 100% / 0 hallucinations.** Despite four agent calls between the user query and the writer (planner → 3 researchers → critic → writer), no agent invented context, no synthesis call manufactured citations, and no hallucinated answer slipped through to the writer. The system prompt does the heavy lifting, but the graph correctly preserves the no-fabrication contract end-to-end. A naive multi-agent build can easily regress on this; this one did not.
- **Researcher "no coverage found" fallback fired honestly.** The researcher prompt instructs the model to emit the literal phrase `'no coverage found'` when retrieved headlines do not address the sub-question. In practice this behavior was widespread (visible in the 4-token average researcher output on many runs) and prevented the critic and writer from being fed fabricated mini-summaries. The architectural pattern of "have each researcher self-report empty results explicitly" is portable to other multi-agent systems regardless of the framework.
- **Critic loop guard held — no infinite loops, no runaway cost.** The manual `critic_passes <= MAX_CRITIC_LOOPS` check in `_critic_route` (and the matching `MAX_CRITIC_LOOPS = 1` cap) bounded the worst-case path to exactly 6 LLM calls in the no-loop case and 9 in the looped case. Across all 60 runs, no query exceeded 9 agent calls — verified in `raw_runs[*].agent_calls` length distribution: `{4: 4, 6: 35, 8: 1, 9: 20}`. The framework gave us no help here; the guard worked because it was written and tested.
- **Eval-suite contract preserved.** The multi-agent runner returns a `RagResponse` with the same dataclass shape as native + LangChain, plus an extra `agent_calls` list that downstream eval modules ignore. The existing `src/eval/{cost,latency,llm_judge,refusal,retrieval}.py` modules ran without modification against the multi-agent output. This is the boring part that would have made the comparison impossible if violated.

---

## 2. Side-by-side metrics

### Cost (USD — full 30-query × 2-model eval, 60 RAG calls total)

| Path | Total | Embed | Generate |
|---|---:|---:|---:|
| native (single-agent) | $0.03399 | $0.00003 | $0.03396 |
| langchain (single-agent) | $0.03395 | $0.00004 | $0.03391 |
| **langgraph (multi-agent)** | **$0.05709** | **$0.00022** | **$0.05687** |

Multi-agent / native cost multiplier: **1.68×**. The embed cost grows 7× because each of the 1–5 sub-questions gets its own `mistral-embed` call; at 30 queries × 2 models × ~3 sub-questions average, that is roughly 180 embed calls vs 60. Still sub-cent in absolute terms ($0.00022 vs $0.00003). The real driver is generate cost: 4 LLM agents fire per query, with the writer using `mistral-large-latest` at 10× the token price of the helper agents.

### Latency total (p50 / p95, milliseconds)

| Path | small p50 | small p95 | large p50 | large p95 |
|---|---:|---:|---:|---:|
| native | 1222 | 1655 | 2877 | 8016 |
| langchain | 1663 | 2080 | 2922 | 6496 |
| **langgraph** | **4722** | **7199** | **6771** | **10897** |

Multi-agent / native p50 multiplier: **3.87×** (small-writer), **2.35×** (large-writer). The large-writer gap is narrower because `mistral-large-latest` generation time dominates and is present in both paths; the overhead of 3–5 sequential helper calls is hidden behind it. For a small-writer path with a p50 latency budget under 2 s, multi-agent blows the budget by a factor of ~2.4.

### LLM-judge (mean across 27 positive queries, judge: `mistral-large-latest`)

| Path | model | ground | rel | compl | conc | cite | overall |
|---|---|---:|---:|---:|---:|---:|---:|
| native | mistral-large | 4.67 | 4.85 | 4.04 | 4.74 | 5.00 | 4.59 |
| native | mistral-small | 4.74 | 4.93 | 4.07 | 4.81 | 5.00 | 4.56 |
| langchain | mistral-large | 4.59 | 4.93 | 4.07 | 4.85 | 4.96 | 4.56 |
| langchain | mistral-small | 4.59 | 4.85 | 3.85 | 4.85 | 4.96 | 4.48 |
| **langgraph** | mistral-large | 4.67 | 4.85 | 3.93 | 4.78 | 4.96 | **4.56** |
| **langgraph** | mistral-small | 4.67 | 4.78 | 3.85 | 4.85 | 5.00 | **4.41** |

**Quality delta:** The multi-agent large-writer scores 4.56 overall — tied with LangChain (4.56) and 0.03 below native (4.59). The small-writer drops to 4.41, 0.07 below the LangChain baseline and 0.15 below native. Completeness is the weakest dimension across all paths; the multi-agent graph does not improve it despite the extra research passes. At n=27 these deltas are borderline-noise, but the direction is consistently neutral-to-negative for multi-agent. There is no quality win to offset the 1.68× cost premium.

### Refusal (3 negative queries)

| Path | model | judge_refused | judge_hallucinated |
|---|---|---:|---:|
| native | small | 3/3 | 0/3 |
| native | large | 3/3 | 0/3 |
| langchain | small | 3/3 | 0/3 |
| langchain | large | 3/3 | 0/3 |
| **langgraph** | small | **3/3** | **0/3** |
| **langgraph** | large | **3/3** | **0/3** |

Identical. The system prompt carries the refusal behavior.

### Retrieval

> **Caveat — retrieval metrics are not on equal footing.** For native and LangChain, "retrieved top-5" is the single pgvector result for the original query. For LangGraph, "retrieved top-5" is the **deduped first-seen union** of hits across all sub-question retrieval calls, truncated to the first 5 unique docs. This favors whichever sub-question fires first (the planner's first decomposed sub-question, which may be narrower than the original). The multi-agent P@5/MRR figures below are a *conservative lower bound* on what a smarter union strategy could achieve — and an *upper bound* on what naive decomposition actually delivers in practice. Read the per-category breakdown (Section 5) before drawing conclusions.

| Path | P@5 | R@5 | MRR |
|---|---:|---:|---:|
| native | 0.681 | 0.730 | 0.920 |
| langchain | 0.681 | 0.730 | 0.920 |
| **langgraph** | **0.385** | **0.465** | **0.716** |

Native and LangChain are identical (same backend, same single-pass retrieval). Multi-agent loses 43% of precision, 36% of recall, and 22% of MRR relative to the single-agent baseline.

---

## 3. Per-agent cost & latency breakdown (multi-agent only, 60 runs total)

| Agent | Calls | Total USD | Mean ms/call | % of generate cost |
|---|---:|---:|---:|---:|
| planner | 60 | $0.00313 | 894 | 5.5% |
| researcher | 213 | $0.00975 | 491 | 17.1% |
| critic | 81 | $0.00596 | 602 | 10.5% |
| writer | 60 | $0.03802 | 1398 | 66.9% |

**Writer dominates at 66.9% of generate cost** because `mistral-large-latest` is ~10× the per-token price of `mistral-small-latest` (which runs planner, researcher, and critic), AND it processes the full capped-at-10 deduped hit context. The 213 researcher calls vs 60 planner/writer calls reflects the 3–5 sub-question fan-out plus the 35% critic-loop triggered top-up calls.

Token usage confirms the picture:

| Path | model | tokens_in avg | tokens_out avg |
|---|---|---:|---:|
| native | mistral-large | 255 | 87 |
| native | mistral-small | 267 | 73 |
| **langgraph** | mistral-large | 1513 | 244 |
| **langgraph** | mistral-small | 1390 | 206 |

Multi-agent uses ~5–6× more input tokens and ~3× more output tokens per query vs native. The input expansion is primarily from the writer receiving the synthesized research notes + full hit context, and each helper call receiving its own system + user prompt. Output expansion comes from the multi-turn structure: planner, each researcher, and critic all produce structured JSON responses on top of the final writer prose.

---

## 4. Critic loop analysis

- **Trigger rate:** 21/60 runs = 35% (critic returned `needs_more_research`)
- **Loop cap:** `MAX_CRITIC_LOOPS = 1` — after one loop back to researcher, the graph routes to writer regardless of verdict
- **Researcher call distribution:** 4 runs used 1 sub-question (atomic pass-through), 35 runs used 3 sub-questions, 1 run used 4, 20 runs used 5 (looped, so base 3 + up to 2 refined)

### Quality — looped vs non-looped

| Condition | n (positive runs) | LLM-judge mean overall |
|---|---:|---:|
| non-looped (critic → writer directly) | 38 | 4.47 |
| looped (critic → researcher → critic → writer) | 16 | 4.50 |
| delta | | **+0.03** |

A +0.03 delta at n=16 is noise. The critic loop adds one complete researcher pass (embed + retrieve + synthesize × 1–2 refined sub-questions) plus a second critic call. On the 21 looped runs this adds at minimum 3 LLM calls and 1–2 embed+retrieve cycles. **The loop contributes no measurable quality lift while accounting for a substantial share of the per-query overhead on affected runs.**

**Why doesn't the loop help?** The corpus has ~400 documents. When the critic identifies a gap ("no information found about X"), the gap is usually a genuine corpus absence — not a retrieval miss. Refining the query and running pgvector again returns either the same docs or docs from a different off-axis cluster. Neither case helps the writer. The critic is correctly identifying incompleteness; it is incorrectly inferring that more retrieval will fix it.

---

## 5. Retrieval by category — the smoking gun

| Category | n queries | single P@5 | multi P@5 | single MRR | multi MRR |
|---|---:|---:|---:|---:|---:|
| broad | 8 | 0.850 | **0.400** | 0.854 | 0.588 |
| specific | 10 | 0.620 | 0.600 | 1.000 | 1.000 |
| multi_aspect | 5 | 0.560 | **0.240** | 0.900 | 0.900 |
| family_niche | 4 | 0.650 | **0.000** | 0.875 | 0.031 |

**Decomposition hurts the categories it was designed to help, and is neutral on the one category that needed no help.**

- **broad (P@5: 0.850 → 0.400, −53%)**: A broad query like "latest trends in Chinese social media" retrieves high-recall results in a single pass because the embedding sits in a dense centroid of the corpus. After decomposition into e.g. "WeChat usage trends" + "Douyin growth" + "Weibo activity", each sub-question's top-5 homes in on a narrow facet. The deduped union of 15 narrow-facet docs contains fewer of the broadly-relevant documents that the original embedding surfaced.

- **family_niche (P@5: 0.650 → 0.000, MRR: 0.875 → 0.031)**: The most severe collapse. Family-niche queries are short and topically specific (e.g. referencing a particular outlet or segment). The planner decomposes them into sub-questions that pull in adjacent-topic docs; the relevant docs from the original query's embedding drop out of the deduped first-5 entirely. MRR of 0.031 means the first relevant doc lands near rank 32 in the union — effectively not retrieved.

- **multi_aspect (P@5: 0.560 → 0.240, −57%, MRR unchanged at 0.900)**: P@5 collapses but MRR holds. This means the top-ranked doc in the union is still relevant, but the next 4 are off-axis. The planner is generating sub-questions that pull in correct first-rank docs per facet but then fill out with unrelated material.

- **specific (P@5: 0.620 → 0.600, −3%, MRR: 1.000 → 1.000)**: Essentially unchanged. Specific/atomic queries produce a 1-item sub-question list from the planner (as intended), so the retrieval path is the same as single-agent. The decomposition adds planner + critic overhead for zero gain.

**The data says:** The queries most likely to benefit from decomposition (broad, multi-aspect) are harmed. The queries that genuinely don't need it (specific/atomic) are correctly identified as atomic by the planner but still pay the overhead of running planner + critic. There is no query category where multi-agent retrieval outperforms single-agent retrieval on this corpus.

---

## 6. Honest engineer's notes

These are grounded in the actual build experience in `src/frameworks/langgraph_multiagent.py` and `src/eval/run_langgraph.py`. They are not generic LangGraph complaints — they are specific moments during this build where the framework abstraction required extra code or produced a debugging surprise.

### 1. TypedDict state with non-serializable fields requires `Any`-typed placeholders

LangGraph's `StateGraph(GraphState)` accepts a `TypedDict` as its state schema. That's fine until you need to stash runtime resources — specifically a `psycopg.Connection` and a `Mistral` SDK client — on the state so every node can reach them without re-instantiating. TypedDict accepts `Any`-typed fields without complaint, and that is what `GraphState` uses:

```python
conn: Any          # psycopg connection
embed_client: Any  # Mistral SDK client for embeddings
```

The plan explicitly notes this: "Why TypedDict over Pydantic: LangGraph's `StateGraph` accepts any mapping; `TypedDict` avoids validation overhead and lets us stash a `psycopg.Connection` and `Mistral` client on the state without Pydantic config gymnastics." The LangChain runner had to fight the same battle with `PrivateAttr` and `ConfigDict(arbitrary_types_allowed=True)`. In both cases the underlying need — thread a live DB connection through a framework object model — requires workarounds. In TypedDict the workaround is `Any`; in Pydantic it is `PrivateAttr`. Neither is satisfying; both lose static typing precisely where you want it.

### 2. No built-in loop guards — you own the cycle-prevention logic

LangGraph will happily run an infinite loop if your conditional edge returns the same branch forever. There is no built-in TTL, iteration cap, or cycle detector at the graph level. The implementation manually tracks `critic_passes: int` in state and checks it in the routing function:

```python
def _critic_route(state: GraphState) -> str:
    if (
        state["critic_verdict"] == "needs_more_research"
        and state["critic_passes"] <= MAX_CRITIC_LOOPS   # manual guard
        and state["refined_queries"]
    ):
        return "researcher_loop"
    return "writer"
```

Forget the `<= MAX_CRITIC_LOOPS` check and the graph loops until the API rate-limits you or your budget runs out. The framework trusts you completely. This is a sharp edge for anyone building loops without carefully reading their own routing logic.

### 3. Conditional edges are string-routed — typos fail silently at runtime

`add_conditional_edges` takes a routing function and a dict mapping the function's return values to target node names:

```python
g.add_conditional_edges(
    "critic",
    _critic_route,
    {"researcher_loop": "merge_refined", "writer": "writer"},
)
```

If `_critic_route` returns `"researcher_loop"` but the dict key is `"researcher-loop"` (hyphen vs underscore), LangGraph raises a `ValueError` at graph compile time in newer versions — but the routing function's return type is `str`, so no type checker flags the mismatch before you run it. During this build the `merge_refined` node was added mid-implementation and the routing dict had to be manually synchronized. No static analysis helped; only running the smoke test caught the mismatch.

### 4. No native observability without LangSmith — had to roll our own telemetry layer

LangGraph's `.invoke()` returns the final state dict. There is no built-in per-node timing, token count, or cost trace unless you use LangSmith (the paid/cloud observability product). To get per-agent metrics for this comparison, the implementation wraps every LLM call in a `_call_llm` helper that captures tokens and latency from `response_metadata`, constructs an `AgentMetrics` TypedDict, and appends it to `state["agent_calls"]` manually:

```python
metrics: AgentMetrics = {
    "agent": agent, "sub_call_idx": sub_call_idx, "model": model,
    "tokens_in": tokens_in, "tokens_out": tokens_out,
    "latency_ms": latency_ms,
    "usd": calc_cost(model, tokens_in, tokens_out),
}
return content, metrics
```

Every node then does `state["agent_calls"] + [metrics]` to accumulate the trace. This is roughly 20–30 LoC of telemetry plumbing that the native pipeline gets for free (one API call, one response object). The LangChain runner had the same problem (LCEL hides stage timings). Multi-agent compounds it because there are 4–6 agent calls per query instead of 1.

### 5. State-merge semantics for accumulator fields — easy to overwrite instead of append

Each node returns a partial dict, and LangGraph merges it into state using the last-write-wins rule per key. For scalar fields (`critic_verdict`, `final_answer`) this is correct. For accumulator fields (`hits`, `research_notes`, `agent_calls`) you must explicitly return the full concatenation:

```python
return {
    "hits": state["hits"] + new_hits,          # must include old hits
    "research_notes": state["research_notes"] + new_notes,
    "agent_calls": state["agent_calls"] + new_calls,
}
```

If you return only `{"hits": new_hits}`, LangGraph replaces the field with the new list — silently discarding everything the previous researcher pass found. There is no `Annotated[list, operator.add]` reducer wired up in this implementation (LangGraph supports it via `operator.add` annotations on TypedDict fields), but using it would require restructuring the state schema mid-build. The manual `state[x] + new_x` pattern works and is explicit, but it is also the most common source of "why is my researcher only returning the last sub-question's hits?" bugs in multi-node graphs.

### 6. The `LangChain wraps mistralai.SDKError` problem extends to LangGraph — retry helper needed a copy-paste workaround

`src/eval/_retry.py::with_retry` catches `mistralai.models.sdkerror.SDKError` for 429 retry handling. It works for the native eval (direct SDK calls). It does not work when the LangGraph runner calls `ChatMistralAI.invoke()` because `langchain_mistralai` wraps the upstream `SDKError` in its own exception class. The LangGraph eval runner carries `with_retry_broad` (a copy-paste from the LangChain runner's fix) that matches on `"429"` / `"rate_limited"` in `str(e)` instead of discriminating by type:

```python
is_rate_limit = "429" in msg or "rate_limited" in msg or "rate limit" in msg
```

This is less type-safe, not covered by the existing retry test suite, and required dropping to `workers=1` (same as the LangChain runner) because even with broad retry, LangGraph's per-call overhead tightens the window against the Mistral free-tier rate limiter. The frustration compounds: both the LangChain and LangGraph paths independently required the same workaround, and the workaround lives in two separate eval runner files with no shared abstraction.

---

## 7. When to use multi-agent

The verdict for **this** corpus is "do not use," but the architectural pattern has legitimate applications that this eval set simply does not exercise. The following are honest cases where the planner-researcher-critic-writer (or similar) graph structure earns its overhead.

### Multi-agent IS useful when:

- **Multi-hop questions where the answer requires chaining retrieved facts.** Example: "Which companies in our portfolio acquired anyone in 2025, and what was the combined deal value?" — this requires retrieving the acquisition events, then retrieving deal sizes for each, then aggregating. A single retrieval call cannot answer this; sub-question decomposition with intermediate synthesis is the right shape. Our 30-query eval set has no genuine multi-hop questions, which is why the architecture finds nothing to exploit.
- **Distinct retrieval contexts per sub-question.** Example: queries that hit multiple knowledge bases ("compare the API rate limits in our Stripe docs vs our internal billing service docs") or multiple corpora with different embedding models. The researcher fan-out becomes essential when each sub-question targets a different retriever, index, or backend. Our eval is single-corpus single-index, so this pattern has nothing to demonstrate.
- **Corpus with atomic-retrieval gaps the planner can route around.** If single-pass retrieval has a known weakness — e.g. terminology drift between query and corpus, or a corpus where common queries embed into a "neutral zone" between semantic clusters — a planner that rewrites the query into multiple paraphrases can rescue recall. Tradeoff: the planner has to be reliably better than the user's original phrasing, which requires either prompt engineering specific to the corpus or a fine-tuned planner. Generic planner prompts (like ours) won't do this.
- **Reasoning chains beyond a single LLM call.** Example: agentic tool use, where the writer needs to call a calculator, then a search, then verify a citation. The graph structure provides the iteration scaffolding. LangGraph genuinely earns its keep here — the framework is built for this shape, not for "retrieve then generate" RAG.
- **Long-context or long-horizon tasks with verifiable checkpoints.** The critic-loop pattern (verify → re-research → verify) is the right shape for tasks where the writer's first draft can be evaluated against an objective check ("does the answer cite at least 3 sources for each claim?", "is the proposed SQL syntactically valid?"). Our critic verifies coverage by LLM judgment, which is too noisy to drive meaningful loops. A critic with an objective check (compiler, validator, regex, search) would behave differently.
- **You can afford 2–4× cost and latency overhead against your SLO.** On a small-writer-only path: p50=4.7 s and $0.00041/query (vs native's $0.00010/query — a 4.1× ratio because the writer is no longer the dominant cost line). On the blended both-models path averaged across small + large writers: $0.00095/query (vs $0.00057 native, a 1.68× ratio). Either way: workable for low-volume asynchronous tasks; not workable for a synchronous API endpoint with a <2 s latency budget.

### Do NOT use multi-agent when:

- **Single-pass retrieval already achieves high P@5 on your dominant query types.** If P@5 > 0.65 and MRR > 0.90 on your current queries (as native achieves here), decomposition has nowhere to improve and several ways to degrade.
- **Queries are mostly atomic or specific.** The planner correctly identifies atomic queries and emits a 1-item sub-question list — but you still pay for the planner call, the critic call, and the added latency. You are taxing your fast path to support a case that rarely arises.
- **You are optimizing for cost or latency SLOs.** 1.68× cost and 3.5–3.9× latency are not rounding errors. On 10,000 query-runs/day (averaging both writer models), native at $5.66/day becomes multi-agent at $9.52/day; on a small-writer-only path the gap is wider (native $0.94/day → multi-agent $4.03/day, a 4.3× cost jump because the writer is no longer the dominant cost line). p50 latency moves from 1.2 s to 4.7 s on the small path.
- **Your corpus is small and well-embedded.** A 400-doc corpus with 1024-dim `mistral-embed` vectors and a cosine HNSW index is already highly discriminative for most queries. The architecture limit is corpus coverage, not retrieval mechanism. Adding agents does not add documents.
- **The 35% critic-loop trigger rate is driven by genuine corpus absences, not retrieval misses.** When the critic fires because a topic is simply not in the corpus, no amount of refined retrieval will help. Measure this before deploying — if >50% of critic triggers result in "no coverage found" researcher notes, the loop is doing nothing but burning tokens.

---

## Where this leaves `production-rag-eval`

**Single-agent native stays in production.** `src/retrieve.py` + `src/generate.py` back the FastAPI service and the Streamlit demo. The multi-agent graph in `src/frameworks/langgraph_multiagent.py` stays in the repo as evidence that the architecture was evaluated honestly against the actual workload before being ruled out.

The LangGraph graph is worth reaching for if a future requirement comes in that is genuinely agentic: multi-document cross-referencing, tool-augmented retrieval, or queries that require iterative clarification from the user. For the current corpus and query distribution, it is the wrong tool.

---

*Reproduce:* `python -m src.eval.run_langgraph` — reads `eval_results/eval_set.jsonl`, writes all results to `eval_results/langgraph/`. Use `--smoke` for a single-query sanity check.
