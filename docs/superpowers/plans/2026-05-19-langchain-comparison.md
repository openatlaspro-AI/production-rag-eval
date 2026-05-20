# LangChain Framework Comparison — Implementation Plan

**Goal:** Demonstrate LangChain fluency by re-implementing the native retrieve+generate path in LangChain, running the existing 30-query eval suite against both, and writing an honest engineer's-eye comparison doc. No marketing copy — the comparison must call out where LangChain adds real value, where it just adds layers, and where it actively hurts (cost telemetry, debuggability).

**Architecture:** Mirror `src/generate.py` with `src/frameworks/langchain_rag.py`. Same `pgvector` backend reached through a `BaseRetriever` subclass that calls our existing `pgvector_search()` — not a parallel `langchain_postgres.PGVector` collection. Return the SAME `RagResponse` / `RetrievalHit` dataclasses so the existing eval modules (`src.eval.cost`, `latency`, `llm_judge`, `refusal`) consume LangChain runs unchanged. Output to `eval_results/langchain/` next to native.

**Tech stack:** langchain ≥0.3, langchain-mistralai ≥0.2, langchain-core (transitive). pgvector + Postgres unchanged.

---

## Key architectural decisions

### 1. Hybrid retriever (BaseRetriever wrapping our pgvector_search)

The spec says "**same pgvector backend**", which rules out spinning up `langchain_postgres.PGVector` with its own `langchain_pg_embedding` schema and re-ingesting 400 docs. Two options remain:

| Option | Description | Demo value | Cost |
|---|---|---|---|
| (b) BaseRetriever subclass | Wrap `pgvector_search()` in LangChain's `BaseRetriever` interface so it composes with LCEL | Shows LangChain integration into existing infra — the realistic enterprise scenario | ~15 LoC overhead |
| (c) Plain function calls | Just swap `mistralai.chat.complete` → `ChatMistralAI.invoke`, no retriever wrapper | Barely "LangChain-style" — minimal demo value | 0 LoC overhead |

**Choosing (b).** It's the honest real-world integration story: most teams have an existing vector store and want to layer LangChain on top, not nuke their schema. The comparison doc explicitly calls out the trade-off: "*purely* LangChain would use `langchain_postgres.PGVector` with its own collection — the duplication is unjustified for an existing pgvector setup."

### 2. Cost telemetry honesty — embed tokens are not exposed by LangChain

This is a real DX gap worth surfacing in the comparison:
- Native `embed_query()` returns `(emb, tokens_in)` from `response.usage.prompt_tokens` — exact.
- LangChain `MistralAIEmbeddings.embed_query()` returns just `list[float]` — **no token usage exposed**.

**Approach:** estimate embed tokens via `max(1, len(query) // 4)` (Mistral tokenizer ~4 chars/token for English). For our 30-query eval, embed cost is <$0.0001 — the estimate error is sub-cent. The comparison doc reports this as a methodological caveat AND as a concrete DX criticism of LangChain:

> *"LangChain's MistralAIEmbeddings discards the `usage` field from the upstream SDK response. Accurate embed-cost telemetry requires writing a custom callback handler or dropping to the raw SDK — a real gap if you're optimizing $0.10/M-token embedding cost at scale."*

Generation cost is fine — `ChatMistralAI` surfaces `response_metadata["token_usage"]` with `prompt_tokens` + `completion_tokens`.

### 3. LCEL only where it pays off

Spec says "use LCEL where it makes sense". It makes sense for the prompt → llm → parser segment:

```python
answer_chain = prompt_template | llm | StrOutputParser()
answer = answer_chain.invoke({"context": ..., "question": ...})
```

It does **not** make sense to compose the entire retrieve→generate flow into one LCEL pipeline (e.g. `RunnableParallel + retriever | prompt | llm`) because we need per-stage `time.perf_counter()` measurements for the eval harness. LCEL hides intermediate timing unless you bolt on callbacks. The comparison doc calls this out as a tension: **LCEL is elegant but works against observability**.

### 4. Reuse eval harness, don't modify it

Existing flow (`src/eval/run_all.py`):
1. Calls `generate_rag_response(...)` → returns `RagResponse` dataclass
2. Converts that to a `run` dict with a stable schema
3. Feeds runs into `cost_mod.compute_cost`, `latency_mod.compute_latency`, `judge_mod.run_judge`, `refusal_mod.compute_refusal`, `compute_retrieval_metrics`

We need a parallel `src/eval/run_langchain.py` that:
- Calls our new `langchain_generate_rag_response(...)` returning the **same `RagResponse` shape**
- Builds the same run dicts
- Feeds them into the EXACT SAME downstream eval modules — `from src.eval import cost as cost_mod`, etc.
- Writes outputs to `eval_results/langchain/` instead of `eval_results/`

Critically: we **import from** `src.eval.*` (allowed — reuse), we do not **edit** any file under `src/eval/` (forbidden by spec).

### 5. Dependency management

`pyproject.toml` already has an optional-extras group:
```toml
[project.optional-dependencies]
langchain = ["langchain>=0.3.0", "langchain-mistralai>=0.2.0", "langchain-postgres>=0.0.12"]
```

The spec says "add to `[dev]`". **Proposing instead: keep the existing `[langchain]` group and add `langchain-core` to it explicitly** (it's a transitive dep but pinning it is clearer for reproducibility). Reasons:
- `[dev]` is for tooling (pytest, ruff, ipykernel). LangChain is a runtime feature dep, not a tooling dep — separation is semantically correct.
- The `langchain-postgres` line is already there from prior planning (unused) — we can drop it since we're not using `PGVector` retriever per decision #1.

**Will revert to `[dev]` if you prefer the literal spec.**

---

## File structure

```
production-rag-eval/
├── src/
│   ├── frameworks/
│   │   ├── __init__.py                       (existing, empty — untouched)
│   │   └── langchain_rag.py                  (new — single-file impl, ~120 LoC)
│   ├── eval/
│   │   └── run_langchain.py                  (new — orchestrator, ~150 LoC mirroring run_all.py)
│   ├── retrieve.py                           (untouched)
│   ├── generate.py                           (untouched)
│   └── eval/* (everything else)              (untouched)
├── eval_results/
│   ├── langchain/                            (new dir)
│   │   ├── baseline.json                     (retrieval metrics, k=10)
│   │   ├── consolidated_report.json
│   │   ├── latency.json
│   │   ├── cost_summary.json
│   │   ├── llm_judge.json
│   │   └── refusal.json
│   └── langchain_vs_native.md                (new — comparison doc)
├── Makefile                                  (modify — `make eval-langchain` target)
├── README.md                                 (modify — "Framework comparison" section)
└── pyproject.toml                            (modify — minor)
```

---

## Module: `src/frameworks/langchain_rag.py`

Single file, ~120 LoC. Sections:

1. **Imports** — `langchain_core.documents.Document`, `langchain_core.retrievers.BaseRetriever`, `langchain_core.prompts.ChatPromptTemplate`, `langchain_core.output_parsers.StrOutputParser`, `langchain_mistralai.MistralAIEmbeddings`, `langchain_mistralai.ChatMistralAI`, plus reuse of our `RetrievalHit`, `RagResponse`, `TimingBreakdown`, `CostBreakdown` from `src.retrieve` and `src.generate`.

2. **`PgvectorBaseRetriever(BaseRetriever)`** — subclass that holds a `psycopg.Connection`, a `MistralAIEmbeddings` instance, and a `k`. `_get_relevant_documents(query, *, run_manager)` calls our existing `pgvector_search()`, returns `list[Document]` with `page_content = title_en` and `metadata = {hit-fields}` so we can rehydrate `RetrievalHit` downstream. **Also stores raw hits** on the instance for per-stage timing access (LangChain's BaseRetriever interface only returns `Document`, but we need our `RetrievalHit` shape for the eval).

3. **`PROMPT_TEMPLATE`** — `ChatPromptTemplate` with system + user templates that match the native `SYSTEM_PROMPT` exactly (so any quality difference is from framework, not prompt drift).

4. **`langchain_generate_rag_response(query, *, k, model, ...)`** — mirrors `src.generate.generate_rag_response` signature. Internal flow:

   ```python
   embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=...)

   # ---- embed (instrumented manually — LangChain hides this) ----
   t0 = time.perf_counter()
   q_emb = embeddings.embed_query(query)
   embed_ms = (time.perf_counter() - t0) * 1000
   embed_tokens_in = max(1, len(query) // 4)  # estimated — see decision #2

   # ---- retrieve (use BaseRetriever subclass) ----
   t1 = time.perf_counter()
   retriever = PgvectorBaseRetriever(conn=conn, q_emb=np.array(q_emb, dtype=np.float32), k=k)
   docs = retriever.invoke(query)            # LCEL-compatible call
   hits = retriever.last_hits                # rehydrated RetrievalHit list
   retrieve_ms = (time.perf_counter() - t1) * 1000

   # ---- generate via LCEL chain ----
   llm = ChatMistralAI(model=model, temperature=0.2, mistral_api_key=...)
   chain = PROMPT_TEMPLATE | llm
   user_input = {"context_block": _format_context(hits), "question": query}
   t2 = time.perf_counter()
   ai_msg = chain.invoke(user_input)
   generate_ms = (time.perf_counter() - t2) * 1000

   answer = ai_msg.content
   usage = ai_msg.response_metadata.get("token_usage", {})
   gen_tokens_in = usage.get("prompt_tokens", 0)
   gen_tokens_out = usage.get("completion_tokens", 0)

   # cost via src.pricing.calc_cost (reused)
   return RagResponse(query=..., answer=answer, sources=hits, model=model,
                       timing=TimingBreakdown(...),
                       cost=CostBreakdown(...))
   ```

   Note: `q_emb` is computed once by `MistralAIEmbeddings`, then handed to the retriever to avoid double-embedding (the BaseRetriever interface usually re-embeds — we sidestep that).

   Reuses: `RetrievalHit`, `RagResponse`, `TimingBreakdown`, `CostBreakdown`, `calc_cost`, `pgvector_search`, `SYSTEM_PROMPT` (string).

---

## Module: `src/eval/run_langchain.py`

Near-clone of `src/eval/run_all.py`, with three deltas:

1. `from src.frameworks.langchain_rag import langchain_generate_rag_response as generate_rag_response`
2. Output paths point to `eval_results/langchain/*.json` instead of `eval_results/*.json`
3. Console title says "LangChain RAG eval suite"

Everything else — retrieval-metric computation, `latency_mod`, `cost_mod`, `judge_mod`, `refusal_mod`, the ThreadPoolExecutor harness, the `.raw_runs.json` checkpoint — is **identical to `run_all.py`** and reused via direct import.

`make eval-langchain` target invokes `python -m src.eval.run_langchain`.

**Cost estimate (will confirm with you before running):** Same shape as native eval: 60 RAG calls × ~(255 in + 80 out tokens) split across mistral-small + mistral-large + ~54 mistral-large judge calls. Native eval cost was **$0.034** per `cost_summary.json`. LangChain run will be in the same ballpark — possibly slightly higher if LangChain's prompt templating adds tokens (one thing we'll specifically measure and report).

---

## Comparison doc: `eval_results/langchain_vs_native.md`

Headline-finding-first, like the README. Skeleton:

```markdown
# LangChain vs Native — Framework Comparison

> **One-line finding:** [filled in after the run — e.g. "LangChain matches the native pipeline on quality and refusal, costs N% more in tokens due to prompt-template overhead, adds ~X ms of latency, and obscures embed-cost telemetry. Worth it for teams that need plug-and-play swap-in of new providers; not worth it for a single-provider production pipeline with cost & latency SLOs."]

## Side-by-side metrics

| Metric                       | Native pipeline | LangChain | Δ        |
|---|---|---|---|
| precision@5 (small)          | 0.682           | …         | …        |
| recall@10 (small)            | 1.000*          | …         | …        |
| MRR                          | 0.920           | …         | …        |
| LLM-judge overall (small)    | 4.56            | …         | …        |
| LLM-judge overall (large)    | 4.59            | …         | …        |
| Refusal rate on negatives    | 100%            | …         | …        |
| p50 total latency (small)    | … ms            | … ms      | …        |
| p95 total latency (small)    | … ms            | … ms      | …        |
| Total eval cost              | $0.034          | $…        | …        |
| Generate tokens-in mean      | 267             | …         | …        |

(Numbers filled from `eval_results/consolidated_report.json` and `eval_results/langchain/consolidated_report.json` after the run.)

## Implementation surface

| Dimension           | Native | LangChain |
|---|---|---|
| LoC (retrieve+generate) | `src/retrieve.py` + `src/generate.py` = ~250 | `src/frameworks/langchain_rag.py` = ~120 |
| New runtime deps    | 0 (mistralai + psycopg already there) | +langchain, +langchain-mistralai, +langchain-core (~XX MB installed) |
| Cold import time    | … ms            | … ms (measure: `python -c "import time; t=time.perf_counter(); from src.X import …; print(time.perf_counter()-t)"`)|

## Honest tradeoffs (engineer's view)

### Where LangChain pulls its weight
- **Provider swappability** — swapping `ChatMistralAI` → `ChatOpenAI` is one import + arg change. The native code is hard-coded to Mistral; replacement is a ~40-line PR.
- **LCEL composability** — `prompt | llm | parser` is genuinely cleaner than the native string-formatting + `client.chat.complete()` boilerplate for the chat segment.
- **Streaming / async / batch out of the box** — `chain.stream()`, `chain.ainvoke()`, `chain.batch()` are free.

### Where LangChain hurts
- **Embed cost telemetry is broken** — `MistralAIEmbeddings.embed_query()` discards usage. Accurate cost tracking requires a custom `BaseCallbackHandler` or dropping back to the raw SDK. (Concrete: our cost_summary went from 4-digit-precision to estimate-via-char-count for the embed line.)
- **Per-stage timing requires manual `perf_counter()` anyway** — LCEL chains hide intermediate timing unless you instrument with callbacks. So we end up using LCEL only for the prompt→llm segment and timing each step ourselves — same as the native code.
- **Error messages are layered** — a 401 from Mistral surfaces as `langchain_mistralai...PydanticValidationError` wrapped around an `httpx.HTTPStatusError`. The native code surfaces the SDK error directly.
- **Deployment surface** — adds ~XX MB to the install. Our Streamlit Cloud demo deliberately doesn't pull it in (`requirements.txt` excludes LangChain).
- **Prompt-template tokens** — if observed: LangChain's `ChatPromptTemplate` may add wrapping tokens vs raw message dicts. (Will report actual delta from the run.)

### When to use LangChain in this kind of project
- **Yes:** prototyping when provider is unfixed; multi-step agentic workflows (where LCEL composition genuinely pays); teams who want streaming/async without writing it.
- **No:** single-provider production pipeline with cost & latency SLOs; teams who need precise telemetry; deployment targets where install size matters.

### Where I'd land for `production-rag-eval`
[Filled after run. Tentative: native code stays. LangChain implementation lives in `src/frameworks/` as a demonstration of multi-framework competence — and an honest record of where the abstraction taxes outweigh the benefits at this project's scope.]
```

The comparison doc is intentionally **engineer-written, not marketing.** Negative findings are equally welcome.

---

## Build order

### Phase 1 — implementation (no API cost)
1. Update `pyproject.toml`: drop `langchain-postgres` from `[langchain]` extras, add `langchain-core` explicitly.
2. `pip install -e ".[langchain]"` (will print package list — included in chat for review).
3. Write `src/frameworks/langchain_rag.py`.
4. Smoke-test: one query end-to-end, print `RagResponse`. Verify shape matches native. **(~$0.0001 cost.)**
5. Write `src/eval/run_langchain.py`.
6. Wire `make eval-langchain` Makefile target.

### Phase 2 — PAUSE checkpoint
7. **PAUSE.** Show user the smoke-test output + sample `RagResponse` dict alongside a native one. Get explicit "go" to run the full eval (~$0.03).

### Phase 3 — eval execution
8. Run `make eval-langchain`. Writes `eval_results/langchain/*.json`.
9. Verify all 5 JSON files present + non-empty + same schema as `eval_results/*.json`.

### Phase 4 — comparison doc
10. Author `eval_results/langchain_vs_native.md` with real numbers from both runs.
11. Update README — new section "Framework comparison: native vs LangChain" linking to the doc.

### Phase 5 — PAUSE before commit
12. **PAUSE.** Show user the comparison doc, file list, commit-message draft. Wait for "commit it".

### Phase 6 — commit
13. Single `feat(frameworks): LangChain implementation + side-by-side eval comparison` commit. Or split into (i) `feat(frameworks): LangChain RAG implementation` + (ii) `eval: run LangChain across the 30-query eval set + comparison doc`. **Will propose split at the commit checkpoint.**

---

## Out of scope (per spec)

- LangGraph
- Migrating production pipeline to LangChain
- Streamlit demo (its `requirements.txt` stays lean — no LangChain)
- Editing anything under `src/eval/` or touching `src/retrieve.py` / `src/generate.py`

---

## Open questions for user before "go"

1. **`pyproject.toml` placement** — use the existing `[langchain]` optional-extras group (recommended) or follow the spec literally and add to `[dev]`?
2. **Embed-cost telemetry** — estimate-via-char-count is good enough (sub-cent error) and gives us a real DX criticism for the doc. OK?
3. **Commit split at the end** — one commit or two (impl + eval/doc)? Will surface again at PAUSE #2 once we see the actual diff.
