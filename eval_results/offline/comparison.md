# Offline multi-method retrieval benchmark

Embedding model: **OpenAI text-embedding-3-small** · corpus: 400 docs · labeled queries: 27 evaluated (3 negative excluded) · metric cutoffs: 5, 10

Labels are the existing human hand labels in `eval_set.jsonl`; alignment to the offline corpus was empirically verified (see METHODOLOGY.md). These OpenAI dense numbers are a NEW baseline, distinct from the Mistral `eval_results/baseline.json` — do not conflate.

| retriever | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.207 | 0.148 | 0.280 | 0.352 | 0.530 |
| dense | 0.548 | 0.389 | 0.607 | 0.762 | 0.830 |
| graph | 0.326 | 0.233 | 0.357 | 0.438 | 0.497 |
| hybrid | 0.407 | 0.293 | 0.489 | 0.612 | 0.772 |
| hybrid_rerank | 0.267 | 0.211 | 0.341 | 0.481 | 0.798 |

## Summary

On P@5 the best method is **dense** (0.548) vs. dense baseline 0.548 (Δ +0.000). On MRR the best is **dense** (0.830). Every number above is read from the committed per-retriever JSON receipts in this directory.
