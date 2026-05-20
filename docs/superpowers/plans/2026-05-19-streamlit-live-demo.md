# Streamlit Live Demo — Implementation Plan

**Goal:** Public Streamlit Cloud app that lets recruiters interact with the production-rag-eval RAG pipeline (query → top-5 hits → cited answer → latency + cost) without cloning the repo or running Postgres.

**Architecture:** Streamlit Cloud has no pgvector. We add an alternate in-process retrieval path that loads pre-computed embeddings from a committed `.npy` file at boot and does brute-force cosine similarity over 400 docs in numpy (~1 ms — FAISS is overkill at this corpus size). Production pgvector path in `src/retrieve.py` stays untouched. Eval suite stays untouched.

**Tech stack:** Streamlit 1.40+, numpy, mistralai 1.x. No new heavy deps.

---

## Key architectural decisions

### 1. Pre-compute embeddings, commit as `data/embeddings.npy`

**Recommended over runtime-embed-at-boot.** 400 × 1024 floats = 1.6 MB → committed.

| Approach | Cold start | API cost on boot | Risk on Mistral outage |
|---|---|---|---|
| Pre-compute + commit `.npy` | ~50 ms | $0 | Demo still works for retrieval (generation will fail but UI shows hits) |
| Runtime embed all 400 at boot | ~5 s + rate limits | ~$0.0004 each cold start | Boot fails entirely |

Re-run `scripts/precompute_embeddings.py` only when `data/trend_signals.jsonl` changes. Makefile target `make embeddings`.

### 2. Numpy retrieval, not FAISS (renaming the spec's filename)

400 docs is too small for FAISS to pay off — a numpy `argpartition` on cosine similarities is ~1 ms with zero new dependencies. FAISS would add ~50 MB to the Streamlit Cloud build for no measurable speedup.

**Proposing:** name the module `src/retrieve_inmemory.py` instead of `src/retrieve_faiss.py`. The spec said FAISS; numpy is a strict simplification. **Will revert to FAISS if you prefer the literal spec.**

### 3. RAG orchestration lives inside `streamlit_app.py`

`generate_rag_response()` in `src/generate.py` is tightly coupled to `pgvector_search(conn, ...)`. Three options considered:

- (a) Refactor `generate.py` to accept an injectable retriever — touches eval code path. **Rejected** (out of scope per spec).
- (b) New `src/rag_inmemory.py` mirroring `generate_rag_response` — adds a module for ~30 lines used in exactly one place.
- (c) Inline the embed → retrieve → generate orchestration in `streamlit_app.py`, reusing `embed_query`, `build_user_message`, `SYSTEM_PROMPT`, `RagResponse`, `CostBreakdown`, `TimingBreakdown` from existing modules. **Chosen** — most YAGNI, all reusable pieces stay reused, no new module for a one-caller wrapper.

### 4. Graceful degradation when `MISTRAL_API_KEY` is missing

`src/config.py` raises on import if the key is missing (pydantic-settings requires it). To handle this without crashing the Streamlit app on boot, the app reads the key directly from `st.secrets` / env and only constructs the `Mistral` client lazily. If absent, the app renders a friendly "add your key in Settings → Secrets" message and disables the Search button.

This means `streamlit_app.py` **does not** `from src.config import settings` directly — it reads the key with its own helper. Other settings (model names) come from `src.pricing.PRICING` keys.

---

## File structure

```
production-rag-eval/
├── app/
│   ├── __init__.py                       (new, empty)
│   └── streamlit_app.py                  (new)
├── src/
│   ├── retrieve_inmemory.py              (new)
│   └── ... (existing, untouched)
├── scripts/
│   └── precompute_embeddings.py          (new)
├── data/
│   ├── trend_signals.jsonl               (existing)
│   └── embeddings.npy                    (new, generated, committed)
├── .streamlit/
│   └── secrets.toml.example              (new)
├── requirements.txt                       (new, for Streamlit Cloud)
├── Makefile                              (modify — add `embeddings` + `streamlit` targets)
└── README.md                             (modify — add Live Demo section)
```

---

## Build order

### Step 1: Pre-compute embeddings script

**File:** `scripts/precompute_embeddings.py`

Reads `data/trend_signals.jsonl`, batches the `title_en` field through `mistral-embed`, writes `data/embeddings.npy` as a `float32` array of shape `(400, 1024)`. Also writes `data/embeddings_meta.json` with `{"model": "mistral-embed", "n_docs": 400, "dim": 1024, "generated_at": "..."}` so we can detect drift.

Logic mirrors the embed phase of `src/ingest.py` (batch size 50, 0.5 s sleep, sorted-by-index alignment).

Run once locally: `python -m scripts.precompute_embeddings` → commits `data/embeddings.npy` (1.6 MB).

### Step 2: In-memory retriever

**File:** `src/retrieve_inmemory.py`

```python
"""In-memory cosine retrieval for the Streamlit demo (no Postgres).

Loads pre-computed embeddings (data/embeddings.npy) + metadata (data/trend_signals.jsonl)
once. Brute-force cosine search via numpy — fast enough for 400 docs (~1ms).
Production code path (pgvector in src/retrieve.py) is untouched.
"""

from pathlib import Path
import json
import numpy as np
from src.retrieve import RetrievalHit

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

def load_corpus() -> tuple[np.ndarray, list[dict]]:
    """Return (normalized_embeddings, doc_metadata) — call once at app startup."""
    emb = np.load(DATA_DIR / "embeddings.npy")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    docs = [json.loads(line) for line in (DATA_DIR / "trend_signals.jsonl").open(encoding="utf-8")]
    assert len(docs) == emb.shape[0], f"mismatch: {len(docs)} docs vs {emb.shape[0]} embeddings"
    return emb, docs

def inmemory_search(
    q_emb: np.ndarray, corpus_emb: np.ndarray, docs: list[dict], k: int = 5
) -> list[RetrievalHit]:
    """Top-k by cosine similarity. q_emb is the query embedding (1D, 1024)."""
    q = q_emb / np.linalg.norm(q_emb)
    sims = corpus_emb @ q                          # (400,) cosine since both normalized
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]  # sort the top-k
    hits = []
    for rank, i in enumerate(top_idx):
        d = docs[i]
        hits.append(RetrievalHit(
            id=int(i),                                       # row index, not a PK
            source="trend",
            title=d.get("title"),
            title_en=d.get("title_en"),
            platform=d.get("platforms", [None])[0],
            rank=d.get("min_rank"),
            similarity=float(sims[i]),
            metadata={k: v for k, v in d.items() if k not in {"title", "title_en"}},
        ))
    return hits
```

Reuses `RetrievalHit` so the rest of the codebase format-matches.

### Step 3: Streamlit app

**File:** `app/streamlit_app.py`

Sections:
1. **Page config + title + 1-line explainer**
2. **Sidebar:** model dropdown (`mistral-small-latest` default, `mistral-large-latest`), k slider (default 5), link to GitHub
3. **Main:** query text input + Search button
4. **Result panes** (only after Search):
   - Generated answer (markdown, with `[N]` citations preserved)
   - Top-5 retrieved docs as a table (rank · similarity · platform · English title · zh original)
   - Latency breakdown (embed / retrieve / generate / total) in `st.metric` columns
   - Cost in `st.metric` (USD with `format_cost`)
5. **Footer:** GitHub repo link

Caching:
- `@st.cache_resource` → Mistral client (one per app instance)
- `@st.cache_resource` → corpus load (embeddings + docs metadata)

Missing-key handling:
```python
def get_api_key() -> str | None:
    try:
        return st.secrets["MISTRAL_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.environ.get("MISTRAL_API_KEY")

api_key = get_api_key()
if not api_key:
    st.warning("⚠️ MISTRAL_API_KEY not set. Add it under Settings → Secrets in Streamlit Cloud …")
    st.stop()
```

Orchestration (inline, ~25 lines):
```python
client = get_client(api_key)                       # cached
corpus_emb, docs = get_corpus()                    # cached

t0 = time.perf_counter()
q_emb, embed_tokens = embed_query(client, query)
embed_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
hits = inmemory_search(q_emb, corpus_emb, docs, k=k)
retrieve_ms = (time.perf_counter() - t1) * 1000

user_msg = build_user_message(query, hits)
t2 = time.perf_counter()
resp = client.chat.complete(
    model=model,
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_msg}],
    temperature=0.2,
)
generate_ms = (time.perf_counter() - t2) * 1000
# … assemble cost via calc_cost(), render.
```

### Step 4: `requirements.txt` (Streamlit Cloud build)

```
streamlit>=1.40.0
mistralai>=1.5.0,<2.0.0
numpy>=1.26.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
```

Streamlit Cloud auto-installs from `requirements.txt` at repo root. We do **not** need psycopg/pgvector/fastapi for the demo — leaving them out keeps the build ~30 s instead of ~3 min.

Caveat: `src/retrieve.py` imports `psycopg` and `pgvector` at module top — `streamlit_app.py` must import from `src.retrieve` only the symbols it needs (`embed_query`, `RetrievalHit`). Because `embed_query` is defined before any pgvector use, **but** the module-level imports still execute on `from src.retrieve import …`. **Fix:** lazy-import inside `embed_query` won't work (caller passes a Mistral instance). Real fix: move the `import psycopg` / `from pgvector.psycopg import register_vector` lines inside the functions that use them in `src/retrieve.py`. That's a 4-line tweak that doesn't change behavior. **Will confirm before doing this** — it edits production code (allowed, since module top-level imports aren't "the pgvector code path" per the spec).

Alternative: copy `embed_query` + `RetrievalHit` into `src/retrieve_inmemory.py` to avoid touching `src/retrieve.py` at all. Slightly DRYer to lazy-import; slightly safer (no edits to prod) to duplicate. **Recommending the lazy-import fix** (~4 lines, defensive against any other future user of `src/retrieve` who doesn't have postgres installed).

Same issue in `src/generate.py` — that file imports `psycopg` and `pgvector` at module top. We need `build_user_message`, `SYSTEM_PROMPT` from it. Same fix: move heavy imports into functions that use them, or duplicate the constants. **Recommending duplicate** for `generate.py` since `SYSTEM_PROMPT` is just a string and `build_user_message` is 10 lines — duplication is cheaper than a refactor.

### Step 5: `.streamlit/secrets.toml.example`

```toml
# Copy to .streamlit/secrets.toml (gitignored) for local testing.
# On Streamlit Cloud, paste into Settings → Secrets.
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

Add `.streamlit/secrets.toml` to `.gitignore`.

### Step 6: Makefile targets

```makefile
embeddings:  ## Pre-compute data/embeddings.npy from trend_signals.jsonl
	python -m scripts.precompute_embeddings

streamlit:  ## Run the live demo locally
	streamlit run app/streamlit_app.py
```

### Step 7: README — add "Live Demo" section after "Quick start"

```markdown
## Live demo

Try it without cloning: **https://production-rag-eval.streamlit.app** *(URL pending deploy)*

In-process retrieval over the same 400-doc corpus (numpy cosine, not pgvector — Streamlit Cloud
doesn't run Postgres). Generation still goes through the live Mistral API. Same answer quality,
same instrumentation (latency, cost, citations). Production pgvector path lives in `src/retrieve.py`.

To run locally:
\`\`\`bash
make embeddings           # one-time: precompute data/embeddings.npy (~10s)
make streamlit            # opens http://localhost:8501
\`\`\`
```

### Step 8: Local test

Run `streamlit run app/streamlit_app.py`, open the URL, exercise:
- Default query → see 5 hits, answer with citations, latency + cost
- Switch model dropdown to `mistral-large-latest` → cost increases ~10×
- Empty query → button disabled or graceful no-op
- Bad/garbage query (e.g. "asdfgh") → still returns 5 hits with low similarity

### Step 9: **PAUSE.** Show user the local URL, wait for sign-off before any git commit.

---

## Out of scope (per spec)

- Streamlit Cloud web UI deploy
- Eval suite changes
- Touching pgvector retrieval logic
- Auth / rate limiting (the demo will burn the demoer's MISTRAL_API_KEY if scraped — accept this for v1; can add a `st.secrets["MAX_QUERIES_PER_HOUR"]` later if it becomes a problem)

---

## Open questions for user before "go"

1. **FAISS vs numpy?** Recommending numpy + rename file to `src/retrieve_inmemory.py`. OK to deviate from spec?
2. **Edit `src/retrieve.py` and `src/generate.py` to lazy-import pgvector/psycopg?** Alternative: duplicate `embed_query` + `build_user_message` + `SYSTEM_PROMPT` into the new in-memory module. Recommending lazy-import for `retrieve.py`, duplication for `generate.py` (smaller surface).
3. **Pre-commit `data/embeddings.npy` (1.6 MB binary)?** Alternative: regenerate at boot from API (slower cold start, costs $0.0004 each).
