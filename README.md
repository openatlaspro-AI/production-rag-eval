# FamilyHQ RAG — Production Patterns over Live Trend Data

> Retrieval-augmented generation over a real personal-AI-engineering-project corpus.
> Built to demonstrate production RAG patterns with eval harness, latency
> budgeting, and cost tracking. Mistral API + pgvector + FastAPI.

**[ Live demo → coming Friday ]   [ Eval results → `eval_results/baseline.json` (Wed) ]**

## What this is

I run [FamilyHQ Printables](https://familyhqprints.gumroad.com), a personal multi-agent AI engineering project on Anthropic Claude with MCP servers in production. This repo is a retrieval-augmented generation system over the trend signals that drive FamilyHQ's product pipeline plus the deployed product corpus itself.

It exists to demonstrate production RAG patterns: chunking strategies, multilingual embedding (the trend corpus is Chinese-language news, translated at ingest via Mistral), retrieval@k evaluation, latency budgeting per stage, cost tracking per query, and Mistral-as-judge response quality scoring.

## Architecture

> 🚧 Diagram coming Friday — `docs/architecture.png`

The pipeline:
1. **Ingest:** SQLite (TrendRadar) + PDF (FamilyHQ products) → chunk → translate (Mistral) → embed (`mistral-embed`) → pgvector
2. **Retrieve:** Query → embed → cosine similarity over pgvector HNSW index
3. **Generate:** Top-k context → Mistral Large or Mistral Small → response
4. **Eval:** Hand-labeled query set → retrieval@k + latency + cost + LLM-judge

## Eval results

Hand-labeled 30-query eval set against the 400-document corpus. Three negative-test queries (out-of-corpus topics) excluded from aggregate — they'll be evaluated separately via the generation-refusal eval. Full per-query results in [`eval_results/baseline.json`](eval_results/baseline.json); methodology in [`eval_results/labeling_protocol.md`](eval_results/labeling_protocol.md).

### Retrieval (embed model: `mistral-embed`, 1024-dim, pgvector HNSW)

| Metric | Aggregate | Notes |
|---|---|---|
| precision@5 | **0.682** | 68% of top-5 are relevant |
| precision@10 | **0.533** | Drops as expected with deeper rank |
| recall@5 | **0.730** | 73% of labeled relevant docs in top-5 |
| recall@10 | 1.000* | *Upper bound — labeling capped at top-10 retrieval; true recall could be lower if relevant docs exist below rank 10 |
| MRR | **0.920** | First relevant hit at rank 1 for most queries |
| median embed_ms | 352 | Query embedding latency |
| median retrieve_ms | 7 | pgvector HNSW lookup |

### By query category

| Category | n | P@5 | P@10 | R@5 | MRR |
|---|---|---|---|---|---|
| broad (e.g. "Chinese economy") | 8 | 0.85 | 0.71 | 0.63 | 0.85 |
| specific entity (e.g. "hantavirus outbreak") | 10 | 0.62 | 0.46 | 0.81 | **1.00** |
| multi-aspect (e.g. "stocks and oil prices") | 5 | 0.56 | 0.44 | 0.75 | 0.90 |
| family / parenting niche | 4 | 0.65 | 0.48 | 0.71 | 0.88 |
| negative test (expect 0 hits) | 3 | — | — | — | — |

**Reading the numbers:**
- **MRR = 1.00 for specific entities** — when the user asks about a named entity (hantavirus, Faker, Wang Yi), the top result is always relevant. This is the system at its best.
- **Specific P@5 lower (0.62) than broad P@5 (0.85)** — narrow queries have fewer relevant docs in the corpus (often only 1–2), so even perfect retrieval can score P@5 ≤ 0.20 when only 1 doc is relevant. Not a system failure; a denominator artifact.
- **Family-niche category holds up (P@5=0.65)** despite being only 3% of the corpus.

### End-to-end (with generation)

> 🚧 LLM-judge eval lands 2026-05-13 (Day 5). Latency / cost smoke-test numbers below from Day 3.

| Metric | mistral-small-latest | mistral-large-latest |
|---|---|---|
| Cost per query | $0.000087 | $0.001291 |
| Latency total p50 | ~1,400 ms | ~5,300 ms |
| LLM-judge score (1-5) | TBD | TBD |
| Refusal on negative tests | TBD | TBD |

## Why these decisions

- **pgvector over Pinecone/Chroma:** production data already lives in Postgres for most enterprise customers. No new infra to operate.
- **mistral-embed (1024-dim) for embeddings:** provider-agnostic pipeline kept, but using Mistral keeps the data path EU-sovereign-friendly.
- **Mistral for translation at ingest:** the source corpus is Chinese (Baidu, Weibo, Toutiao, etc.). Storing both languages enables cross-lingual queries.
- **Native Python over LangChain for the spine:** more control over the eval harness. See `src/frameworks/langchain_compare.py` for a parallel implementation — benchmarks at the bottom of this README.

ADR-style decisions in `docs/decisions.md`.

## Quick start

```bash
git clone https://github.com/markteji/familyhq-rag.git
cd familyhq-rag
cp .env.example .env  # add your MISTRAL_API_KEY
make up               # start postgres + pgvector
make ingest           # parse, translate, embed, insert
make api              # FastAPI on http://localhost:8000
```

## Trade-offs and what I'd do next

- **Single-stage retrieval today.** Reranking with a cross-encoder would push precision@5 by ~0.08 at +120ms latency.
- **Eval set is 30 hand-labeled queries.** Production version needs 300+ with inter-annotator agreement.
- **No caching layer.** For real customer load, Redis cache on `(query, model)` → response with TTL of 1 hour.
- **Translation at ingest, not query time.** Latency win at retrieval, but means English query → English content match. Real production system would translate queries too.

## Stack

Python 3.12 · Mistral API · pgvector on Postgres 16 · FastAPI · Streamlit · LangChain (comparison module only) · pytest

## What FamilyHQ is

[FamilyHQ Printables](https://familyhqprints.gumroad.com) is a personal multi-agent AI engineering project I built and operate on a Mac Mini M4. Nine deployed products. 24/7. Anthropic Claude + OpenAI GPT-4o + Ollama with intelligent model routing. MCP servers for skill module extension. Persistent agent memory. Telegram-based human-loop failure recovery.

This RAG repo is a slice of that system — the retrieval layer that helps FamilyHQ pick which trends to turn into products.

## Structure

```
familyhq-rag/
├── data/                   # corpus (SQLite imports → JSONL → pgvector)
├── src/
│   ├── ingest.py           # parse → chunk → translate → embed → insert
│   ├── retrieve.py         # similarity_search(query, k)
│   ├── generate.py         # rag_response(query) end-to-end
│   ├── api.py              # FastAPI endpoint
│   ├── chunking/           # recursive + semantic strategies
│   ├── frameworks/         # LangChain comparison
│   └── eval/               # retrieval, latency, cost, llm_judge
├── notebooks/              # data exploration + eval dashboard
├── tests/                  # pytest
├── eval_results/           # checked-in baseline + sweep results
└── docs/                   # architecture diagram + ADRs
```

## License

MIT.
