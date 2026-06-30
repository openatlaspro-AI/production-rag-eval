# GraphRAG + Hybrid Retrieval Upgrade — Design & Plan

**Date:** 2026-06-30
**Repo:** production-rag-eval (existing) — branch `feat/graphrag-hybrid`
**Status:** Approved (autonomous), ready to build

## Goal

Extend the existing reproducible RAG eval with **advanced retrieval methods** and benchmark them
honestly against the dense baseline: **hybrid (BM25 + dense, Reciprocal Rank Fusion)**, a
**lightweight knowledge-graph retriever** (entity co-occurrence expansion), and an optional
**reranking** stage. Make the whole benchmark **self-contained and reproducible with OpenAI
embeddings** so it no longer requires Mistral + Postgres.

This preserves the repo's defining property: **every reported number has a JSON receipt**, and
methodology (labels, alignment, model) is disclosed.

## Why (hiring context)

Closes the "advanced RAG" gap for AI Engineer roles: hybrid search, rank fusion, GraphRAG,
reranking, and an honest multi-method benchmark — extending a repo Mark already owns and that
recruiters can clone and reproduce.

## Hard constraints (honesty)

1. **No fabricated metrics.** Every number in the README/results comes from a committed JSON written
   by the eval runner in this build. If a number can't be generated, it is not claimed.
2. **Verify, don't assume, the label↔corpus alignment.** `eval_results/eval_set.jsonl`
   `relevant_doc_ids` are Postgres `documents.id` (SERIAL). The offline corpus is
   `data/trend_signals.jsonl`. The agent MUST empirically verify whether `documents.id == jsonl line
   number` before using those labels — e.g. for 3 sample queries, read the titles of the claimed
   relevant docs and confirm they are topically on-point for the query. Document the verification in
   `eval_results/offline/METHODOLOGY.md`.
   - If alignment verifies → reuse the existing hand labels (strongest).
   - If it does NOT verify → relabel relevance with an OpenAI LLM judge over the corpus, and clearly
     label results as "LLM-judged relevance (not human gold)" in METHODOLOGY.md and the README.
3. **Disclose the embedding model.** Results are computed with OpenAI `text-embedding-3-small`;
   say so. This is a different embedder than the original Mistral baseline, so the offline dense
   numbers are a NEW baseline, not a restatement of `eval_results/baseline.json`. Do not conflate.

## Architecture (new files; existing files untouched except README/Makefile)

```
scripts/embed_openai.py            # embed corpus -> data/embeddings_openai.npy + meta (idempotent)
src/retrievers/__init__.py
src/retrievers/dense.py            # cosine over OpenAI corpus embeddings (offline baseline)
src/retrievers/bm25.py             # BM25 over corpus text (rank-bm25)
src/retrievers/hybrid.py           # RRF fusion of dense + bm25 ranked lists
src/retrievers/graph.py            # entity co-occurrence graph; seed from query entities, 1-hop expand, score
src/retrievers/rerank.py           # rerank top-N: default MMR (no network); optional OpenAI LLM reranker behind a flag
src/eval/run_offline.py            # compute P@{5,10}, recall@{5,10}, MRR per retriever -> eval_results/offline/<name>.json + comparison.md
eval_results/offline/METHODOLOGY.md
tests/test_retrievers.py           # synthetic-fixture unit tests (no network)
docs/superpowers/specs/2026-06-30-graphrag-hybrid-design.md   # this file
```

- **dense**: query embedded with OpenAI (cached to disk so re-runs need no API); cosine top-k.
- **bm25**: tokenized corpus (title + title_en), `rank_bm25.BM25Okapi`.
- **hybrid**: Reciprocal Rank Fusion `score = Σ 1/(k0 + rank_i)` (k0=60) over dense + bm25.
- **graph**: nodes = docs; edges between docs sharing ≥1 extracted entity (capitalized tokens /
  known keyword list — NO LLM). Seed = docs matching query entities lexically; expand 1 hop; score
  by (#shared-entity weight + dense sim of expanded node). This is "GraphRAG-lite" — honestly
  described as such, not full LLM graph construction.
- **rerank**: MMR diversification over the fused top-N by default (deterministic, no network). An
  `--llm-rerank` flag may use OpenAI to reorder; only run it if the key is present, and write its
  own receipt.

## Eval

`src/eval/run_offline.py` reuses `src/eval/retrieval.py` metric functions (`precision_at_k`,
`recall_at_k`, `reciprocal_rank`) — do not reimplement them. For each retriever, over the
non-negative eval_set queries, compute P@{5,10}, recall@{5,10}, MRR; aggregate overall + by the
existing categories. Write one JSON per retriever to `eval_results/offline/<name>.json` and a
`comparison.md` table. Negative queries stay excluded from aggregates (same convention as
`run_baseline.py`).

## Tests (no network)

`tests/test_retrievers.py` with tiny synthetic fixtures:
- BM25 ranks a doc containing the query term above one that doesn't.
- RRF fusion: a doc ranked highly by BOTH lists beats a doc ranked highly by only one.
- Graph expansion: a doc sharing an entity with a seed doc is reachable; an isolated doc is not.
- MMR rerank: given two near-duplicate top hits, the second is demoted in favor of a diverse doc.
- Metric sanity: `precision_at_k`/`reciprocal_rank` on a hand-built ranking match expected values.
Use small in-memory arrays/dicts — never call OpenAI in tests.

## Deliverables

- All retrievers + runner + tests, `make embed-openai` and `make eval-offline` targets.
- `eval_results/offline/*.json` + `comparison.md` + `METHODOLOGY.md` with REAL numbers.
- README: new "## Offline multi-method benchmark (OpenAI)" section with the comparison table and a
  one-paragraph honest summary of which method won and by how much.
- `ruff` clean, `pytest` green.

## Resume bullet earned (filled with real numbers post-build)

> Extended my open-source RAG eval with **hybrid (BM25 + dense RRF)**, a **knowledge-graph
> retriever**, and **reranking**, benchmarked against a dense baseline on a 30-query labeled set —
> [winning method] improved P@5 from [X] to [Y]. Fully reproducible offline with OpenAI embeddings;
> every number has a committed JSON receipt.
