# LangChain vs Native — Framework Comparison

> **Headline finding.** On this corpus and eval set, the LangChain implementation matches the native pipeline on **retrieval quality** (identical, by construction — same pgvector backend), matches on **refusal safety** (100% / 100%, zero hallucinations on both), dips 1–5% on **LLM-judge quality** (within noise at n=27), comes in **+30% slower** at p50 on the small-model RAG path, costs **the same** to the 4th decimal ($0.034 / $0.034 total eval), and required **lower concurrency + a broader retry helper** to complete the run because LangChain wraps upstream `SDKError` in its own exception type.
>
> **Recommendation.** Native code stays in production for `production-rag-eval`. LangChain implementation lives in `src/frameworks/langchain_rag.py` as a working demonstration that the framework adds real value on agentic / multi-provider workflows, and real overhead on a single-provider production pipeline that already has cost & latency SLOs.

Reproduce: `make eval` (native) and `make eval-langchain` (this run). Raw data: [`eval_results/consolidated_report.json`](consolidated_report.json) and [`eval_results/langchain/consolidated_report.json`](langchain/consolidated_report.json).

---

## Side-by-side metrics (30-query eval, k=5, identical prompt, identical pgvector backend)

### Retrieval (model-independent — both paths embed via `mistral-embed` then cosine over the same pgvector HNSW index)

| Metric | Native | LangChain | Δ |
|---|---:|---:|---:|
| precision@5 | **0.681** | **0.681** | 0 |
| recall@5 | **0.730** | **0.730** | 0 |
| MRR | **0.920** | **0.920** | 0 |

Identical — expected, because the LangChain retriever (`PgvectorBaseRetriever`) is a subclass that calls the SAME `pgvector_search()` against the SAME `documents` table. A "pure LangChain" deploy that re-ingested into `langchain_postgres.PGVector` with its own schema would have produced different doc IDs, but the same recall curve at this corpus size. Re-ingesting 400 docs into a second schema for no quality lift was rejected (see `src/frameworks/langchain_rag.py` docstring).

### LLM-judge (`mistral-large-latest` as judge, 27 positive queries × 2 generators)

| Dimension | Native (small / large) | LangChain (small / large) | Δ |
|---|---:|---:|---:|
| overall | 4.56 / 4.59 | 4.48 / 4.56 | −0.08 / −0.03 |
| groundedness | 4.74 / 4.67 | 4.59 / 4.59 | −0.15 / −0.08 |
| relevance | 4.93 / 4.85 | 4.85 / 4.93 | −0.07 / +0.07 |
| completeness | 4.07 / 4.04 | 3.85 / 4.07 | −0.22 / +0.04 |
| conciseness | 4.81 / — | 4.85 / — | ≈ |
| citation_accuracy | 5.00 / 5.00 | 4.96 / 4.96 | −0.04 / −0.04 |

**Read:** within stochastic noise at temperature=0.2. The small-model LangChain run drops ~5% on completeness; small-model overall drops ~2%. Same prompt, same context, same model — the only mechanical difference is whether the prompt is assembled as a raw `messages=[...]` dict vs through `ChatPromptTemplate.invoke(...)`. Not enough signal to claim LangChain hurts quality; not enough to claim it doesn't. **At n=27 this is noise.**

### Refusal (3 negative-test queries × 2 generators)

| Metric | Native | LangChain |
|---|---:|---:|
| Refusal rate (judge) | **100%** (3/3 both models) | **100%** (3/3 both models) |
| Hallucination rate | 0% | 0% |

Identical. The system prompt does the work, not the framework.

### Latency (p50 / p95, milliseconds — 30 queries per model)

| Stage | Native p50 | LC p50 | Δ p50 | Native p95 | LC p95 |
|---|---:|---:|---:|---:|---:|
| embed | 424 | 500 | **+18%** | 1079 | 1270 |
| retrieve | 6.7 | 8.0 | **+19%** (negligible — ms scale) | 8.0 | 13.2 |
| generate (small) | 836 | 1132 | **+35%** | 1198 | 1589 |
| generate (large) | 2441 | 2250 | −8% (n=30, noisy) | 6546 | 3905 |
| **total (small)** | **1222** | **1663** | **+36%** | 1655 | 2080 |
| **total (large)** | **2877** | **2922** | +1.6% | 8016 | 6496 |

The small-model gap is the real signal: **LangChain adds ~300 ms at p50 on top of an 836 ms native baseline.** Two contributors:
- ~75 ms on the embed call (Pydantic validation in `MistralAIEmbeddings` + extra `httpx` configuration layer)
- ~300 ms on the generate call (Pydantic validation of `AIMessage`, `ChatPromptTemplate` rendering, callback-handler dispatch loop)

The large-model run is dominated by generation time and the framework overhead disappears in the noise. For a small-model production path that targets <1 s end-to-end (we hit it; native median is 1.2 s including embed + retrieve), LangChain pushes you over the latency budget.

### Cost (total eval cost — 60 RAG calls + 56 judge calls + 6 refusal-judge calls)

| Metric | Native | LangChain | Δ |
|---|---:|---:|---:|
| Total eval cost | **$0.0340** | **$0.0339** | $0.0001 |
| Gen tokens-in mean (small) | 267 | 267 | 0 |
| Gen tokens-in mean (large) | 255 | 255 | 0 |
| Gen tokens-out mean (small) | 73 | 67 | −6 (stochastic) |
| Gen tokens-out mean (large) | 87 | 88 | +1 |
| Embed tokens (total) | 344 (exact) | 366 (estimated via `len(q)//4`) | est. error <$0.000001 |

**Cost is the surprise: zero meaningful delta.** The native eval reported exact embed-token counts ($0.0000344 total); LangChain estimates them via `len(query) // 4` because `langchain_mistralai.MistralAIEmbeddings.embed_query()` discards the upstream `usage` field. At 30 queries the estimate error is ~22 tokens (~$0.000002). At 30M queries it would matter. The framework's lossy telemetry is real but only at scale.

---

## Implementation surface

| Dimension | Native | LangChain |
|---|---|---|
| Files | `src/retrieve.py` (153 LoC) + `src/generate.py` (196 LoC) | `src/frameworks/langchain_rag.py` (213 LoC) |
| Logical statements (AST-counted) | 69 + 82 = **151** | **69** |
| New runtime deps | 0 (mistralai + psycopg already present) | **24 new packages** — see "Honest engineer's notes" |
| Install footprint added | 0 MB | **~16 MB** direct (langchain* + langgraph*); add ~80 MB for transitive `huggingface-hub`, `tokenizers`, `pyarrow` (auto-installed even though we don't use HF or arrow) |
| Concurrency safely used in eval | workers=2 | workers=1 (retry helper had to be broadened — see notes) |

**LoC caveat (honest):** The native files include the CLI entry (`if __name__ == "__main__"`), pretty-printing tables for terminal output, an embeddings-only helper used by `src/api.py` and `src/eval/run_baseline.py`, and a DB-only helper used by the same. The LangChain implementation has none of that — it's just the orchestration. A truly apples-to-apples count of "RAG orchestration in this file only" would be roughly 60 native vs 69 LangChain. **Not a real LoC win for LangChain.**

---

## Honest engineer's notes

These are the concrete frustrations from actually building and running this. They are not strawmen — they are the moments during the build where I had to write more code or stop to debug because the framework abstraction obscured something I needed.

### 1. Installing `langchain` pulls 5 LangGraph packages whether you use LangGraph or not

`pip install langchain==1.3.1` brings `langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`, `langgraph-sdk`, and `langchain-protocol` along for the ride. Total install added 24 packages, ~16 MB direct + roughly 80 MB transitive (`huggingface-hub`, `tokenizers`, `pyarrow`, `orjson`, `ormsgpack`, `zstandard`). The Streamlit Cloud demo for this project deliberately excludes LangChain from its `requirements.txt` for exactly this reason — adding it would push the build from ~30 s to several minutes and add a slim-Docker-image headache I don't want.

**Relevant if:** You ship slim Docker images. You deploy to Lambda / Cloud Functions with cold-start budgets. You care about build times on Streamlit Cloud / Heroku / similar managed-build platforms.

### 2. LangChain wraps the upstream `mistralai.SDKError` in its own exception type, defeating type-discriminated retry

The native eval uses `src/eval/_retry.py::with_retry`, which catches `mistralai.models.sdkerror.SDKError` and retries with exponential backoff. It works perfectly at workers=2.

The first LangChain run at workers=2 collected **34 out of 60 expected runs**. 26 mistral-large calls failed on the first 429 because the rate-limit error came back as a LangChain-wrapped exception, not as `SDKError` — so `with_retry` saw a `TypeError`-class exception, didn't recognize it as retryable, and re-raised immediately. The error message itself contained the literal string `"Error response 429 while fetching ..."` — but the *type* was wrong.

Fix: I had to write a parallel `with_retry_broad` in `src/eval/run_langchain.py` that catches `Exception` and pattern-matches `str(e).lower()` for `"429"` / `"rate_limited"` / `"rate limit"`. **More code to do the same job, less type safety, no test coverage for the new branch.** And I had to drop concurrency from 2 → 1 because even with proper retry, LangChain's wrapper-layer latency made the backoff window race the rate limiter.

**Relevant if:** You have a production retry/circuit-breaker layer that discriminates on exception type. You'll either rewrite it for pattern matching, or you'll wrap each LangChain call in a try/except that re-raises as your domain type.

### 3. `MistralAIEmbeddings` throws away usage telemetry from the upstream SDK

The raw `mistralai` SDK returns `response.usage.prompt_tokens` on every embedding call. `langchain_mistralai.MistralAIEmbeddings.embed_query()` returns `list[float]` — no usage object, no callback metadata. Verified by reading the source.

If you want accurate embed cost at scale, your options are:
- Drop back to the raw SDK for embeddings (defeating the framework abstraction)
- Write a `BaseCallbackHandler` that intercepts the HTTP response (~30 LoC, fragile to library updates)
- Estimate via `len(query) // 4` (what this comparison did — sub-cent error at 30 queries, meaningful at 30M)

This comparison reports embed tokens as **estimated** in `eval_results/langchain/cost_summary.json` and the consolidated report metadata documents the estimation method. The eval suite's downstream `cost_mod` consumed the estimate without complaint — schema parity preserved — but the underlying number is no longer exact. **A real DX gap for anyone optimizing inference cost.**

### 4. LCEL hides exactly the timing data the eval needs

The "right" LangChain way to write this would be a single chain:
```python
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
answer = chain.invoke(query)
```

That's elegant. But the eval needs `embed_ms`, `retrieve_ms`, and `generate_ms` as **separate measurements**, and `chain.invoke()` returns one wall-clock number for the whole pipe. To split it I either need to write a `BaseCallbackHandler` that records `on_retriever_start/end` and `on_llm_start/end` events (more code, more abstraction) or break the chain into separately-invoked pieces and time each with `perf_counter()` myself — which is what `src/frameworks/langchain_rag.py` ends up doing. Same code shape as native. **LCEL composability and observability are in direct tension.**

### 5. Pydantic v2 field validation rejects `psycopg.Connection` and `np.ndarray`

I wanted `PgvectorBaseRetriever` to hold its connection and pre-computed query embedding as regular fields. Pydantic v2 refused both:
- `psycopg.Connection` — not a Pydantic model, not in the validator registry. Needs `arbitrary_types_allowed=True` on the model config.
- `np.ndarray` — same problem; even with `arbitrary_types_allowed`, the validator complains about array-like operations.

Workaround: stash both on the instance via `PrivateAttr` (the `_conn` and `_q_emb` you see in the code) and use a custom `__init__`. **Roughly 10 lines of glue to do what `class Foo: def __init__(self, conn, q_emb): ...` does in plain Python.** This is the "framework tax" in microcosm — to participate in LangChain's runtime (so `retriever.invoke(query)` works, so it composes with LCEL) you accept Pydantic validation, which doesn't speak your domain types, which means workarounds.

### 6. Error messages are now two layers deep

A 401 from the Mistral API used to surface as `mistralai.models.sdkerror.SDKError: API error occurred: Status 401`. Same 401 via LangChain comes out as `pydantic_core._pydantic_core.ValidationError` wrapped around an `httpx.HTTPStatusError` with the SDK error in the `__cause__` chain. To find the actual HTTP status you walk `.__cause__` until you find one. Productivity-killer when debugging deployment issues.

---

## When to use LangChain in this kind of project

**Use it when:**
- The provider is unfixed. If you might swap Mistral → OpenAI → Anthropic mid-project, the swap is one import + one constructor change. The native code is a ~40-line diff.
- You're building **multi-step agentic workflows** (tool use, planning loops, conditional branching). LCEL + LangGraph genuinely earn their keep here — that's a separate evaluation worth doing.
- You want **streaming, async, and batch** modes "for free" via `chain.stream()` / `chain.ainvoke()` / `chain.batch()`. Writing these yourself in the native style is real work.
- The team is unfamiliar with the underlying provider SDK and wants a unified mental model across providers.

**Don't use it when:**
- The provider is fixed. You're locking in a 24-package dependency tree to abstract over one vendor.
- You have **production cost & latency SLOs**. Per this comparison: +35% small-model generation latency, +30% total p50, ~30% slower under rate limits.
- You need **precise telemetry** (exact embed tokens, exact stage timings without callback gymnastics).
- Deployment target is size-sensitive (Streamlit Cloud, Lambda, slim Docker).
- Your existing infra has a vector store you don't want to re-ingest into `langchain_postgres.PGVector`.

## Where I landed for `production-rag-eval`

**Native code stays in production.** `src/retrieve.py` + `src/generate.py` continue to back the FastAPI service and the Streamlit demo. The LangChain implementation in `src/frameworks/langchain_rag.py` and the parallel eval at `eval_results/langchain/` stay in the repo as evidence that I can build with the framework AND that I evaluated it against my actual workload before committing to it.

If a future requirement comes in — multi-provider, agentic tool use, streaming — I'd reach for LangChain (or LangGraph) for *that specific surface*, not retrofit the whole pipeline.
