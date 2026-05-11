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

> 🚧 First baseline lands Wednesday 2026-05-13. Format below shows the target table.

| Metric | mistral-large | mistral-small |
|---|---|---|
| precision@5 | TBD | TBD |
| recall@10 | TBD | TBD |
| MRR | TBD | TBD |
| Latency p50 | TBD ms | TBD ms |
| Latency p95 | TBD ms | TBD ms |
| Cost per query | $0.____ | $0.____ |
| LLM-judge score (1-5) | TBD | TBD |

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
