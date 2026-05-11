# Data

This directory holds the corpus for the RAG demo. Two source types feed into a single `documents` table in pgvector.

## Sources

### 1. Trend signals — TrendRadar SQLite databases

**Local source (gitignored):** `~/TrendRadar/output/news/*.db`

TrendRadar produces one SQLite database per day with a `news_items` table:

```sql
news_items(
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    platform_id TEXT,        -- baidu, weibo, toutiao, douyin, zhihu, ...
    rank INTEGER,
    url TEXT,
    first_crawl_time TEXT,
    last_crawl_time TEXT,
    crawl_count INTEGER
)
```

**Language:** Chinese (zh-CN). Sources: Baidu Hot, Weibo Hot, Toutiao, Douyin Hot, Zhihu, Bilibili, WallStreetCN, ThePaper, IFeng, Tieba.

**Volume:** ~12,800 rows across 13 daily DBs. After dedup by title hash, ~3,000–5,000 unique trends.

**Ingestion plan:** dedupe → top-N by aggregate `crawl_count` → translate via Mistral → store both languages → embed English version.

**Privacy:** all titles + URLs are public news. No PII.

### 2. Product corpus — printable PDF products + descriptions

**Local sources (gitignored):**
- `~/Documents/FamilyHQ/PDFs_Live/*.pdf` — 3 published printable PDFs locally; 9 deployed total to a Gumroad store
- `~/Documents/FamilyHQ/Scripts/upload_3_new.py` — Python literals containing product titles, prices, and descriptions

> The local path keeps the `FamilyHQ` folder name because that's the actual filesystem location on the author's machine; the repo and dataset framing are independent of it.

**Language:** English.

**Volume:** 3 PDFs × ~10–20 chunks each + 3 product descriptions ≈ 30–50 records.

**Ingestion plan:** extract via `pypdf` → chunk by section → embed directly (already English).

**Privacy:** product copy is published commercial content. Safe to commit.

## Pipeline

```
Raw sources (gitignored)               JSONL (committed)            pgvector (local)
────────────────────────               ────────────────             ────────────────
~/TrendRadar/output/news/*.db    ─┐
                                  ├─→  data/trend_signals.jsonl  ─→  documents (source='trend')
                                  │    (deduped, translated)
                                  │
~/Documents/FamilyHQ/PDFs_Live/  ─┤
~/Documents/FamilyHQ/Scripts/    ─┴─→  data/products.jsonl       ─→  documents (source='product')
                                       (chunked + extracted)
```

## Files in this directory

After running `make ingest`:

- `trend_signals.jsonl` — committed. Each line:
  ```json
  {"id":"trend_1234","date":"2026-05-09","source_platform":"baidu","title_zh":"...","title_en":"...","rank":1,"url":"https://...","crawl_count":5}
  ```
- `products.jsonl` — committed. Each line:
  ```json
  {"id":"product_chore_chart_p2","slug":"family-chore-chart","section":"contents","content":"...","price_cents":599}
  ```

## Anonymization checklist

- [x] No personal user data in any source
- [x] All trend URLs are public news links
- [x] Product copy is published commercial content
- [x] **`~/.openclaw/config/integrations/gumroad.json` is NEVER referenced from this repo** — has API tokens
- [x] `upload_3_new.py` content extracted manually as constants in `src/ingest.py` to avoid pulling the secret config

## Reproducing the corpus

If you want to run this demo without the original sources:

1. Generate a synthetic trend corpus by sampling Wikipedia or HuggingFace datasets — see `notebooks/01_explore_data.ipynb` for a sketch
2. Use any published commercial product copy you have access to
3. Adjust `EVAL_SET_PATH` queries in `eval_results/eval_set.jsonl` to match the synthetic corpus
