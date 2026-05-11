# Eval Set Labeling Protocol — Production RAG Eval

**Created:** 2026-05-09
**Eval set target:** 30 hand-labeled query→relevance pairs
**Time budget:** ~60 minutes (2 minutes per query)
**Output:** `eval_results/eval_set.jsonl`

---

## What this eval set measures

Retrieval quality of `pgvector` cosine similarity over the 400-document trend corpus. Specifically:

- **precision@5** — of the top 5 retrieved, how many are relevant?
- **recall@10** — of all relevant docs in the corpus, how many appear in top 10?
- **MRR** — mean reciprocal rank of the first relevant hit

The eval set does NOT measure generation quality (that's the LLM-judge eval) or end-to-end answer correctness. It measures **whether the retrieval layer brings the right documents to the LLM**.

---

## Scoring system: binary (0 / 1)

For Day 4 baseline, use **binary relevance**:
- **1** — the document is on-topic for the query and a reasonable LLM could answer the query (or part of it) using this document
- **0** — the document is off-topic, only loosely associated, or would mislead the LLM

We are NOT using graded relevance (0/1/2/3) for the baseline — binary keeps annotation fast and metric math simple. Future eval sets can upgrade to graded once the workflow is proven.

---

## What counts as "relevant" for trend retrieval

A document is **relevant** to a query if **at least one** of the following holds:

1. **Topical match** — the document is directly about the query subject. Example: query "Iran-US tensions" → document "US strikes Iran convoy in Hormuz" → relevant.
2. **Entity match** — the document explicitly mentions the entity, person, or event named in the query, AND that mention is the document's primary subject. Example: query "World Cup broadcast" → document "FIFA Asian Cup broadcast rights dispute" → relevant.
3. **Reasonable LLM use** — an LLM constructing an answer to the query would justifiably cite this document.

A document is **NOT relevant** if:

1. **Loose keyword match only** — the query and document share words but the document is about something else. Example: query "family budgeting" → document "Wall Street weekly market wrap" → not relevant (both have "money" but doc is about markets).
2. **Tangential mention** — the document briefly mentions the query subject but is about something else. Example: query "Disney parks" → document "Tech CEO compares product to Disney" → not relevant.
3. **Translation noise** — the English translation accidentally suggests a topic that the original Chinese doesn't have. Use your best read of the title; if translation is ambiguous, mark 0.

---

## Edge cases you'll hit (and how to handle them)

| Situation | Decision |
|---|---|
| Query is broad (e.g., "Chinese economy"), 30 docs match | Label all 30 as relevant. Don't artificially limit. |
| Translation looks weird/garbled | Read original Chinese if you can; otherwise mark 0 (LLM can't grasp it either) |
| Query has two themes (e.g., "family + finance") | Doc must match BOTH to be 1. If only matches one, mark 0 and note "partial". |
| Doc is about an event that *happens to* involve the query subject (e.g., a celebrity's sport match for a "celebrity gossip" query) | Mark 1 only if the celebrity angle is the document's subject |
| Doc is in the corpus but you're not sure of relevance | Mark 0 with `notes: "uncertain — borderline"` |
| Query expects 0 relevant docs (negative test) | `relevant_doc_ids: []`, `notes: "negative test — corpus has no docs about X"` |
| Top-10 retrieval contains 5 highly relevant + 5 totally unrelated | Label only the 5 relevant; the 5 unrelated do not appear in `relevant_doc_ids` |

---

## Labeling workflow (2 min per query)

1. **Read the query** (5s)
2. **Run retrieval** to see top-10 hits:
   ```bash
   .venv/bin/python -m src.retrieve "<query>" 10
   ```
   Or via API once you've built `src/eval/retrieve_topk.py`.
3. **Scan top-10 titles** (~30s)
4. **Decide which are relevant** (~45s)
   - Note the document `id` of each relevant hit (the integer PK from `documents` table)
5. **Optional: extend search** if you suspect more relevant docs exist deeper in the corpus
   - Run with k=20 or k=30
   - This catches recall ceiling honestly
6. **Write the entry** (~30s):
   ```json
   {"id": 1, "query": "...", "relevant_doc_ids": [12, 87, 245], "notes": ""}
   ```

**If you're spending more than 3 minutes on a query, mark it `notes: "skipped — ambiguous"` and move on.** A noisy edge case shouldn't bottleneck the whole eval.

---

## Document IDs

Each document has an integer primary key in the `documents` table. To find IDs:

```bash
docker compose exec -T postgres psql -U rag -d familyhq_rag \
  -c "SELECT id, LEFT(title_en, 80) FROM documents WHERE title_en ILIKE '%<keyword>%' ORDER BY id;"
```

Or query by similarity (use the retrieve.py output which shows `id` field — but note: the current retrieve.py doesn't print `id`, so use the SQL query above for now). On Day 4 we'll add `id` to the retrieve table output.

---

## Coverage targets for the 30 queries

To get meaningful precision/recall numbers, the 30-query set should span:

| Query type | Target count | Why |
|---|---|---|
| Broad topic queries (e.g., "Chinese economy") | ~8 | Tests recall — many relevant docs |
| Specific entity queries (e.g., "hantavirus outbreak") | ~10 | Tests precision — few but exact matches |
| Multi-aspect queries (e.g., "family budgeting tips") | ~5 | Tests vector similarity for compositional concepts |
| Negative tests (queries expected to retrieve nothing) | ~3 | Tests that the system can refuse |
| Family/parenting niche (small corpus segment) | ~4 | The original niche-use case — test retrieval for the slim slice |

---

## What "good" looks like (rough thresholds)

The Day 4 baseline numbers are unknown until we run them. As **rough industry expectations** for a small (~400 doc), small-eval-set system:

| Metric | Below average | Acceptable | Strong |
|---|---|---|---|
| precision@5 | < 0.6 | 0.6–0.8 | > 0.8 |
| recall@10 | < 0.7 | 0.7–0.85 | > 0.85 |
| MRR | < 0.5 | 0.5–0.75 | > 0.75 |

For the README baseline table, report whatever you measure honestly. The labeling protocol document is itself a credibility signal — interviewers will read it and judge methodology before judging numbers.

---

## After labeling

1. Verify all 30 entries have non-TODO query strings
2. Run `src/eval/retrieval.py` (lands Day 4) → produces `eval_results/baseline.json`
3. Update README's Eval Results table with the actual numbers
4. Commit `eval_set.jsonl` + `baseline.json`

---

## Schema reminder

```jsonl
{"id": 1, "query": "what trends are about Chinese economic indicators", "relevant_doc_ids": [12, 87, 245, 311], "notes": "broad query — 4 relevant macro-finance docs"}
{"id": 2, "query": "is anyone talking about family budgeting tips", "relevant_doc_ids": [], "notes": "negative test — corpus has personal-finance docs but none specifically on family budget how-to"}
```

Required fields: `id` (int), `query` (str), `relevant_doc_ids` (list[int]), `notes` (str, can be empty).
