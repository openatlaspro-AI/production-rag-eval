# Production RAG Eval

> **Production RAG over a 400-document Chinese-news trend corpus, with a 30-query labeled eval suite. Every README number has a JSON receipt in the repo.**

Mistral API · pgvector on Postgres 16 (HNSW) · FastAPI · LLM-as-judge · refusal eval on negative tests · full latency / cost / quality instrumentation.

[![Python](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Mistral](https://img.shields.io/badge/Mistral-API-FF7000.svg)](https://mistral.ai/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791.svg)](https://github.com/pgvector/pgvector)

## 🎯 Headline finding

> **`mistral-small-latest` is the production default.** It matches `mistral-large-latest` on LLM-judge overall (**4.56 vs 4.59**, judged by `mistral-large` itself) at **10× lower cost** and **2.4× lower latency**. Same 100% refusal rate on negative tests, zero hallucinations. Reserve the large model for cases where you can afford the latency budget.

Full numbers below. Raw data: [`eval_results/consolidated_report.json`](eval_results/consolidated_report.json).

## Live demo

Try it without cloning: **https://openatlaspro-rag-eval.streamlit.app**

Same Mistral pipeline as the production FastAPI service, with one substitution: retrieval runs over pre-computed embeddings in numpy (`src/retrieve_inmemory.py`) instead of pgvector — Streamlit Cloud doesn't run Postgres. Same answer quality, same instrumentation (latency, cost, citations). Production pgvector path in `src/retrieve.py` is untouched.

Run it locally:
```bash
make embeddings           # one-time: precompute data/embeddings.npy (~10s, ~$0.0004)
make demo                 # opens http://localhost:8501
```

## Quick start

```bash
git clone https://github.com/openatlaspro-AI/production-rag-eval.git
cd production-rag-eval && cp .env.example .env   # paste your MISTRAL_API_KEY into .env
make up ingest                            # postgres + pgvector + 400 docs (~2 min)
make api                                  # FastAPI on http://localhost:8000
```

Smoke-test it:

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"what trends involve Chinese geopolitics","k":5}' | jq
```

To reproduce the full eval (60 RAG calls + 54 judge calls, ~$0.03):

```bash
make eval                                 # writes eval_results/consolidated_report.json
```

## Framework comparison: native vs LangChain

The same 30-query eval suite ran against a LangChain reimplementation of the retrieve+generate path (`src/frameworks/langchain_rag.py`) using `MistralAIEmbeddings`, `ChatMistralAI`, `BaseRetriever`, and LCEL. Same pgvector backend, same prompt, same models.

| Metric (small-model) | Native | LangChain |
|---|---:|---:|
| Retrieval P@5 / R@5 / MRR | **0.681 / 0.730 / 0.920** | **0.681 / 0.730 / 0.920** (identical) |
| LLM-judge overall | 4.56 | 4.48 (within noise at n=27) |
| Refusal rate (negatives) | 100% | 100% |
| Total p50 latency | **1222 ms** | **1663 ms** (+36%) |
| Total eval cost | $0.0340 | $0.0339 |

**Headline:** LangChain matches on quality and refusal, costs the same, runs ~30% slower on small-model RAG, and required dropping eval concurrency from 2 → 1 because LangChain wraps `mistralai.SDKError` in its own exception type and defeats type-discriminated retry. Full numbers, six concrete engineer's frustrations, and "when to use / when not to" → [`eval_results/langchain_vs_native.md`](eval_results/langchain_vs_native.md).

Reproduce: `make eval` (native) + `make eval-langchain` (LangChain). Each ~$0.03, ~20 min.

All 6 JSON files in `eval_results/langchain/` are committed — anyone can compare the runs without re-spending the $0.03.

## Offline multi-method benchmark (OpenAI)

A second, **self-contained** benchmark (`src/retrievers/`, `src/eval/run_offline.py`) compares five retrieval methods on the same 30-query labeled set — runnable offline with only OpenAI embeddings, no Mistral or Postgres. Methods: **dense** (cosine over `text-embedding-3-small`), **BM25** (`rank-bm25`), **hybrid** (BM25 + dense via Reciprocal Rank Fusion, k0=60), **graph** (GraphRAG-lite: lexical entity co-occurrence graph, 1-hop expansion, *no LLM*), and **hybrid + rerank** (MMR diversification).

These use **OpenAI `text-embedding-3-small`**, so the `dense` row is a **NEW baseline**, distinct from the Mistral `eval_results/baseline.json` above — they are not directly comparable and are not conflated. The 30 hand labels were reused after empirically verifying that the Postgres `documents.id` ↔ `trend_signals.jsonl` line alignment holds (details + the honest result write-up in [`eval_results/offline/METHODOLOGY.md`](eval_results/offline/METHODOLOGY.md)).

| retriever | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.207 | 0.148 | 0.280 | 0.352 | 0.530 |
| dense | **0.548** | **0.389** | **0.607** | **0.762** | **0.830** |
| graph | 0.326 | 0.233 | 0.357 | 0.438 | 0.497 |
| hybrid | 0.407 | 0.293 | 0.489 | 0.612 | 0.772 |
| hybrid_rerank | 0.267 | 0.211 | 0.341 | 0.481 | 0.798 |

**Honest headline:** on this set of English, topical/semantic queries over short bilingual headlines, **plain dense retrieval wins decisively** (P@5 0.548, MRR 0.830). BM25 is weak (little lexical overlap between natural-language queries and headline text), and fusing it actually *drags hybrid below dense*; the lexical graph retriever and MMR rerank don't beat dense either. The advanced methods underperform here, and we report that rather than tuning until one "wins" — every number above is read from a committed JSON receipt in `eval_results/offline/`.

Reproduce: `make eval-offline` (embeds the corpus once via OpenAI, caches query embeddings, writes `eval_results/offline/*.json` + `comparison.md`). Install deps with `pip install -e ".[offline]"`.

## Architecture

![Architecture diagram](docs/architecture.png)

Two parallel pipelines. The query path (left) serves end-users via FastAPI with a shared `psycopg-pool` + Mistral client. The eval path (right) is reproducible — every number in the README below comes from a JSON file checked into the repo.

## Eval results

Hand-labeled 30-query eval set against the 400-document corpus. Three negative-test queries are excluded from retrieval/quality aggregates — they're scored separately via the refusal eval.

Methodology: [`eval_results/labeling_protocol.md`](eval_results/labeling_protocol.md). Per-query results: [`eval_results/consolidated_report.json`](eval_results/consolidated_report.json) and [`eval_results/baseline.json`](eval_results/baseline.json).

### Retrieval — `mistral-embed` (1024-dim) + pgvector HNSW cosine

| Metric | Aggregate | Notes |
|---|---|---|
| precision@5 | **0.682** | 68% of top-5 are relevant |
| precision@10 | 0.533 | Drops with deeper rank — expected |
| recall@5 | **0.730** | 73% of labeled relevant docs in top-5 |
| recall@10 | 1.000* | *Upper bound — labeling was capped at top-10 retrieval, so true global recall could be lower |
| MRR | **0.920** | First relevant hit at rank 1 for most queries |
| median embed_ms | 352 | Query embedding latency |
| median retrieve_ms | 7 | pgvector HNSW lookup |

### Retrieval — by query category

| Category | n | P@5 | P@10 | R@5 | MRR |
|---|---|---|---|---|---|
| broad (e.g. "Chinese economy") | 8 | 0.85 | 0.71 | 0.63 | 0.85 |
| specific entity (e.g. "hantavirus outbreak") | 10 | 0.62 | 0.46 | 0.81 | **1.00** |
| multi-aspect (e.g. "stocks and oil prices") | 5 | 0.56 | 0.44 | 0.75 | 0.90 |
| family / parenting niche | 4 | 0.65 | 0.48 | 0.71 | 0.88 |
| negative test (expect 0 hits) | 3 | — | — | — | — |

Reading the numbers:
- **MRR = 1.00 for specific entities** — when the user asks about a named thing, the top result is always relevant. The system at its best.
- **Specific P@5 (0.62) < broad P@5 (0.85)** — narrow queries often have only 1 relevant doc in the corpus, so the P@5 ceiling is 0.20. Denominator artifact, not a system failure.
- **Family-niche holds up (P@5 = 0.65) despite being only 3% of corpus.**

### End-to-end — generation, judge, refusal

Full eval run: 60 RAG calls + 54 LLM-judge calls + 6 refusal-judge calls. Wall time 22 min (sequential judge calls — Mistral free-tier `mistral-large` is rate-limited to ~1 req/sec). Reproducible via `make eval`.

| Metric | mistral-small-latest | mistral-large-latest |
|---|---|---|
| **Cost per query** | **$0.000097** | $0.001034 (~10× small) |
| **Latency total p50** | **1,222 ms** | 2,877 ms (~2.4× small) |
| Latency total p95 | 1,655 ms | 8,016 ms |
| Latency total p99 | 2,215 ms | 12,700 ms |
| LLM-judge **overall** (1–5) | **4.56** | 4.59 |
| LLM-judge groundedness | 4.74 | 4.67 |
| LLM-judge relevance | 4.93 | 4.85 |
| LLM-judge completeness | 4.07 | 4.04 |
| LLM-judge conciseness | 4.81 | 4.74 |
| LLM-judge **citation accuracy** | **5.00** | **5.00** |
| **Refusal rate** (negative tests) | **3/3 (100%)** | **3/3 (100%)** |
| Hallucination (negative tests) | **0/3** | **0/3** |
| **Total eval cost** | | **$0.0340** |

Judge model: `mistral-large-latest` (temperature=0, JSON-mode output). Rubric and prompts: [`src/eval/llm_judge.py`](src/eval/llm_judge.py).

**Completeness is the weakest LLM-judge dimension (~4.05)**. The system reliably stays grounded and on-topic but doesn't always weave in every relevant retrieved source. Two fixes worth trying: cross-encoder reranking on top-10 before generation, or longer-context prompts with explicit "use all relevant sources" instruction.

## Why these decisions

- **pgvector over Pinecone / Chroma / Qdrant** — production data already lives in Postgres for most enterprise customers. No new infra to operate; single connection serves embeddings, search, and eval metadata.
- **`mistral-embed` (1024-dim) for embeddings** — provider-agnostic pipeline kept, but using Mistral keeps the data path EU-sovereign-friendly and demonstrates a second Mistral API use case beyond chat.
- **Translate at ingest, not query time** — source corpus is Chinese (Baidu, Weibo, Toutiao, …). Translating once via `mistral-small` and storing both languages cuts query-path latency and enables future cross-lingual interfaces.
- **Native Python over LangChain for the spine** — the eval harness needs to instrument every stage (embed, retrieve, generate) with latency + cost telemetry; LangChain's abstractions hide stage boundaries. A LangChain comparison module is intentionally scoped to a later iteration.
- **30-query hand-labeled eval over LLM-generated** — small enough to label by hand (~60 min) but big enough for meaningful precision/recall variance. Hand-labeling forced familiarity with the corpus, which surfaced data-shape issues early.

Full ADRs with alternatives and trade-offs: [`docs/decisions.md`](docs/decisions.md).

## Trade-offs and what I'd do next

- **Completeness (4.05/5) is the weakest LLM-judge dimension.** Generation is grounded and on-topic but doesn't always pull in every relevant source. Fixes: (a) cross-encoder reranking on top-10, (b) prompt-instruction tweak ("use all provided sources unless irrelevant").
- **Single-stage retrieval today.** Cross-encoder reranking would push precision@5 by an estimated ~0.08 at +120 ms latency.
- **Eval set is 30 queries with binary relevance.** Production version needs 300+ queries with graded relevance (0/1/2/3) and inter-annotator agreement.
- **No caching layer.** Median 1.2 s per query for `mistral-small`; a 50% cache hit rate on `(query, model)` would halve perceived latency. Redis with 1 h TTL is the right next step.
- **Translation at ingest means English query → English content match.** Production would translate queries too for cross-lingual retrieval.
- **Mistral free-tier rate limits force sequential judge calls** (`workers=1`). On a paid tier the full eval runs in ~5 min instead of 22. `src/eval/_retry.py` has exponential backoff; the orchestrator checkpoints raw runs so a rate-limit failure at step 5 doesn't waste step-1 work.

## Stack

Python 3.12 · Mistral API (chat, embed, JSON-mode) · pgvector on Postgres 16 (HNSW, m=16, ef_construction=64) · FastAPI · psycopg-pool · Rich · pytest

## Project structure

```
production-rag-eval/
├── data/                     # corpus snapshot (committed JSONL)
│   ├── trend_signals.jsonl   # 400 trends, Chinese + English
│   └── README.md             # sources + anonymization checklist
├── docker/                   # postgres init.sql (HNSW index + schema)
├── docker-compose.yml        # pgvector/pgvector:pg16 with healthcheck
├── docs/
│   ├── architecture.png
│   └── decisions.md          # ADR-style design records
├── eval_results/             # all eval JSON outputs (checked in)
│   ├── eval_set.jsonl        # 30 labeled queries
│   ├── labeling_protocol.md
│   ├── candidate_queries.md
│   ├── baseline.json         # retrieval-only baseline
│   ├── consolidated_report.json   # full eval suite output
│   ├── latency.json · cost_summary.json · llm_judge.json · refusal.json
├── src/
│   ├── config.py · pricing.py
│   ├── ingest.py             # SQLite → translate → embed → pgvector
│   ├── retrieve.py           # embed_query + pgvector_search
│   ├── generate.py           # end-to-end RAG with timing + cost telemetry
│   ├── api.py                # FastAPI (ConnectionPool + shared client)
│   └── eval/                 # latency · cost · llm_judge · refusal · run_all
├── Makefile                  # up · down · ingest · api · eval · test · fmt
├── pyproject.toml            # mistralai pinned 1.x (2.x has broken __init__)
└── README.md
```

## Personal context

This repo was extracted from a larger personal multi-agent AI engineering project I run on a Mac Mini M4. The trend corpus here is the retrieval layer of that system — the part that decides which signals are worth acting on. The full repo is scoped to one thing: showing what a production-grade RAG pipeline plus a real eval harness look like on a small, honestly-labeled corpus.

## License

[MIT](LICENSE).
