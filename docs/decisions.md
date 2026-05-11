# Architecture Decision Records (ADRs)

Short, dated, opinionated. Each ADR captures *why* a choice was made and what alternatives were considered.

---

## ADR-001 — pgvector over Pinecone, Chroma, Qdrant
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** Need a vector store for ~500-document corpus with HNSW retrieval and cosine similarity.

**Decision:** Postgres 16 + pgvector extension.

**Why:**
- Production data already lives in Postgres for most enterprise customers. No new infra to operate.
- HNSW + IVFFlat indexes are production-grade.
- Single database per environment simplifies the eval harness — metrics and embeddings query from the same connection.
- Local dev via Docker Compose is one command.

**Alternatives considered:**
- *Pinecone:* SaaS lock-in, separate billing, network hop adds 30–80ms latency.
- *Chroma:* easier to start with, but production hardening story is weaker.
- *Qdrant:* strong product, but adds a new operational concern for customers.

**Trade-off:** Postgres scales horizontally less gracefully than purpose-built vector DBs. For 500 docs this is irrelevant; for 10M+, revisit.

---

## ADR-002 — mistral-embed (1024-dim) for embeddings
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** Need an embedding model. Pipeline should be provider-agnostic but the demo audience is Mistral-aware.

**Decision:** `mistral-embed` (1024 dimensions, cosine similarity).

**Why:**
- Direct Mistral API signal in interview demos.
- Multilingual handling — needed for Chinese trend corpus.
- 1024-dim is the sweet spot for 500-doc corpus: enough expressiveness, fast retrieval.

**Alternatives considered:**
- *OpenAI text-embedding-3-large:* higher quality on English-only benchmarks; loses Mistral signal.
- *BGE-M3:* open-source, multilingual, free; loses Mistral signal and adds GPU/CPU dependency.

**Trade-off:** API cost vs local. At 500-doc scale, total embedding cost is < $0.10 — irrelevant.

---

## ADR-003 — Translate trend corpus at ingest, not query time
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** Trend corpus is Chinese; expected query language is English (recruiter audience).

**Decision:** At ingest, translate Chinese titles to English via Mistral. Store both. Embed English version.

**Why:**
- Latency win: query → embed → retrieve → generate, no translation in the request path.
- Both versions in storage means a future cross-lingual interface is possible without re-ingest.
- Demonstrates Mistral API beyond embeddings (translation use case).

**Alternatives considered:**
- *Translate at query time:* adds 200–500ms per query, repeated cost on every request.
- *Embed Chinese directly:* `mistral-embed` is multilingual; but English-query → English-corpus is more reliable than cross-lingual at small corpus size.

**Trade-off:** Translation quality at ingest is fixed. If Mistral mistranslates a title once, that error persists. Mitigation: spot-check 5% of translations during ingest; flag low-confidence cases.

---

## ADR-004 — Native Python pipeline, LangChain only as comparison module
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** LangChain is the dominant agentic framework; many JDs list it as a requirement. The eval harness needs fine-grained control.

**Decision:** Main spine in native Python (`src/ingest.py`, `src/retrieve.py`, `src/generate.py`). LangChain implementation as a parallel comparison module (`src/frameworks/langchain_compare.py`).

**Why:**
- Eval harness needs to instrument every stage (embed, retrieve, generate) with latency + cost telemetry. LangChain's abstractions hide stage boundaries.
- Comparison module demonstrates LangChain familiarity for JD requirements without making it the spine.
- README ships a benchmark of native vs LangChain — gives interviewers a real talking point.

**Alternatives considered:**
- *LangChain everywhere:* easier to start, harder to instrument cleanly.
- *No LangChain at all:* leaves a JD gap unaddressed.

**Trade-off:** Two implementations to maintain. Mitigation: comparison module is intentionally minimal — only enough surface area to benchmark, not a parallel feature set.

---

## ADR-005 — pypdf for product PDF extraction
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** 3 FamilyHQ product PDFs need text extraction.

**Decision:** `pypdf` for now.

**Why:**
- Pure Python, no system dependencies.
- Handles the simple text layout in FamilyHQ PDFs (printable forms with text labels).

**Alternatives considered:**
- *pdfplumber:* better for complex tabular extraction.
- *unstructured:* more robust but heavier.

**Trade-off:** If FamilyHQ PDFs add complex tables in the future, swap to pdfplumber. Today's PDFs are simple enough.

---

## ADR-006 — 30-query hand-labeled eval set
**Date:** 2026-05-09 | **Status:** Accepted

**Context:** Need a labeled eval set to measure retrieval@k.

**Decision:** Hand-label 30 query-relevance pairs over 2 hours on Tuesday. Document the labeling protocol in `eval_results/labeling_protocol.md`.

**Why:**
- 30 queries is the smallest set that gives meaningful precision@5 / recall@10 / MRR variance estimates.
- Hand-labeling forces familiarity with the corpus — finds dataset bugs early.
- Labeling protocol document signals seriousness of the eval methodology.

**Alternatives considered:**
- *LLM-generated eval set:* fast but circular — same model that's being tested can game its own queries.
- *No eval set, vibes-based:* defeats the entire point of the differentiator.

**Trade-off:** 30 queries is small. Production version needs 300+ with inter-annotator agreement. README explicitly calls this out.
