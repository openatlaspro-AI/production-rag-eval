"""Offline multi-method retrieval benchmark.

Evaluates each offline retriever (dense / bm25 / hybrid / graph, plus optional
+rerank variants) over the labeled `eval_results/eval_set.jsonl` queries using
the EXISTING metric functions in `src/eval/retrieval.py`. Writes one JSON receipt
per retriever to `eval_results/offline/<name>.json` and a `comparison.md` table.

This benchmark is fully self-contained: it needs only the OpenAI corpus
embeddings (`data/embeddings_openai.npy`) plus a one-time OpenAI query-embedding
pass (cached to disk). It does NOT touch Postgres/Mistral. These dense numbers
are a NEW baseline (OpenAI `text-embedding-3-small`), distinct from the Mistral
`eval_results/baseline.json`. See `eval_results/offline/METHODOLOGY.md`.

Convention matches run_baseline.py: negative-test queries (empty
relevant_doc_ids) are excluded from aggregate metrics.

Label alignment: eval_set `relevant_doc_ids` are Postgres `documents.id` (SERIAL,
1-indexed). The offline corpus is `data/trend_signals.jsonl`. Empirically verified
that `documents.id N` == jsonl line N (1-indexed), so corpus index = id - 1.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import numpy as np
from rich.console import Console
from rich.table import Table

from src.eval.retrieval import (
    hits_in_top_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrievers import (
    BM25Retriever,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    mmr_rerank,
)

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
JSONL_PATH = DATA_DIR / "trend_signals.jsonl"
EMB_PATH = DATA_DIR / "embeddings_openai.npy"
EVAL_SET_PATH = REPO_ROOT / "eval_results" / "eval_set.jsonl"
OUT_DIR = REPO_ROOT / "eval_results" / "offline"

EMBED_MODEL = "text-embedding-3-small"
K_MAX = 10

# Same category bucketing as run_baseline.py.
CATEGORIES = {
    "broad": set(range(1, 9)),
    "specific": set(range(9, 19)),
    "multi_aspect": set(range(19, 24)),
    "negative": set(range(24, 27)),
    "family_niche": set(range(27, 31)),
}


def categorize(qid: int) -> str:
    for name, ids in CATEGORIES.items():
        if qid in ids:
            return name
    return "uncategorized"


def safe_mean(values: list[float]) -> float:
    finite = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return mean(finite) if finite else math.nan


def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    return obj


def load_docs() -> list[dict]:
    return [json.loads(line) for line in JSONL_PATH.open(encoding="utf-8")]


def load_queries() -> list[dict]:
    queries = []
    with open(EVAL_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def id_to_index(doc_id: int) -> int:
    """Postgres documents.id (1-indexed) -> corpus row index (0-indexed)."""
    return doc_id - 1


def index_to_id(idx: int) -> int:
    return idx + 1


def build_retrievers(docs: list[dict], corpus_emb: np.ndarray) -> dict:
    """Construct each named retriever. `search(query, k)` -> list of corpus IDs.

    Retrievers operate on corpus indices internally; we wrap them to return
    1-indexed document ids so metrics line up with eval_set labels.
    """
    dense = DenseRetriever(corpus_emb=corpus_emb)
    bm25 = BM25Retriever(docs)
    hybrid = HybridRetriever(dense, bm25)
    graph = GraphRetriever(docs, dense=dense)
    norm_corpus = corpus_emb / np.clip(
        np.linalg.norm(corpus_emb, axis=1, keepdims=True), 1e-12, None
    )

    def wrap(idx_search):
        return lambda q, k=K_MAX: [index_to_id(i) for i in idx_search(q, k)]

    def hybrid_rerank(query: str, k: int = K_MAX) -> list[int]:
        # Fuse a deeper candidate pool, then MMR-diversify the top-k.
        cands = hybrid.search(query, k=30)
        q_emb = dense.embed_query(query)
        reranked = mmr_rerank(q_emb, cands, norm_corpus, k=k, lambda_mult=0.5)
        return [index_to_id(i) for i in reranked]

    return {
        "dense": wrap(dense.search),
        "bm25": wrap(bm25.search),
        "hybrid": wrap(hybrid.search),
        "graph": wrap(graph.search),
        "hybrid_rerank": hybrid_rerank,
    }


def eval_retriever(name: str, search_fn, queries: list[dict]) -> dict:
    per_query: list[dict] = []
    for q in queries:
        qid = q["id"]
        query = q["query"]
        relevant_set = set(q["relevant_doc_ids"])
        category = categorize(qid)
        is_negative = category == "negative"

        retrieved_ids = search_fn(query, K_MAX)

        if is_negative:
            row = {
                "id": qid,
                "query": query,
                "category": category,
                "n_relevant": 0,
                "retrieved_top_10": retrieved_ids,
                "precision_at_5": math.nan,
                "precision_at_10": math.nan,
                "recall_at_5": math.nan,
                "recall_at_10": math.nan,
                "mrr": math.nan,
            }
        else:
            row = {
                "id": qid,
                "query": query,
                "category": category,
                "n_relevant": len(relevant_set),
                "retrieved_top_10": retrieved_ids,
                "hits_in_top_5": hits_in_top_k(relevant_set, retrieved_ids, 5),
                "hits_in_top_10": hits_in_top_k(relevant_set, retrieved_ids, 10),
                "precision_at_5": precision_at_k(relevant_set, retrieved_ids, 5),
                "precision_at_10": precision_at_k(relevant_set, retrieved_ids, 10),
                "recall_at_5": recall_at_k(relevant_set, retrieved_ids, 5),
                "recall_at_10": recall_at_k(relevant_set, retrieved_ids, 10),
                "mrr": reciprocal_rank(relevant_set, retrieved_ids),
            }
        per_query.append(row)

    pos = [r for r in per_query if r["category"] != "negative"]
    aggregate = {
        "n_queries_total": len(per_query),
        "n_queries_evaluated": len(pos),
        "n_negative_tests": len(per_query) - len(pos),
        "precision_at_5": safe_mean([r["precision_at_5"] for r in pos]),
        "precision_at_10": safe_mean([r["precision_at_10"] for r in pos]),
        "recall_at_5": safe_mean([r["recall_at_5"] for r in pos]),
        "recall_at_10": safe_mean([r["recall_at_10"] for r in pos]),
        "mrr": safe_mean([r["mrr"] for r in pos]),
    }

    by_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in per_query if r["category"] == cat]
        if cat == "negative":
            by_category[cat] = {"n_queries": len(cat_rows), "note": "excluded from aggregates"}
            continue
        if not cat_rows:
            by_category[cat] = {"n_queries": 0}
            continue
        by_category[cat] = {
            "n_queries": len(cat_rows),
            "precision_at_5": safe_mean([r["precision_at_5"] for r in cat_rows]),
            "precision_at_10": safe_mean([r["precision_at_10"] for r in cat_rows]),
            "recall_at_5": safe_mean([r["recall_at_5"] for r in cat_rows]),
            "recall_at_10": safe_mean([r["recall_at_10"] for r in cat_rows]),
            "mrr": safe_mean([r["mrr"] for r in cat_rows]),
        }

    return {
        "metadata": {
            "retriever": name,
            "timestamp": datetime.now(UTC).isoformat(),
            "embed_model": EMBED_MODEL,
            "k_max": K_MAX,
            "corpus_size": 400,
            "similarity_metric": "cosine",
            "eval_set_path": str(EVAL_SET_PATH.relative_to(REPO_ROOT)),
            "label_alignment": "documents.id N == trend_signals.jsonl line N (1-indexed); verified",
            "labels": "existing human hand labels (eval_set.jsonl)",
        },
        "aggregate": aggregate,
        "by_category": by_category,
        "per_query": per_query,
    }


def write_comparison(results: dict[str, dict]) -> None:
    order = ["bm25", "dense", "graph", "hybrid", "hybrid_rerank"]
    names = [n for n in order if n in results]

    lines = [
        "# Offline multi-method retrieval benchmark",
        "",
        f"Embedding model: **OpenAI {EMBED_MODEL}** · corpus: 400 docs · "
        "labeled queries: 27 evaluated (3 negative excluded) · metric cutoffs: 5, 10",
        "",
        "Labels are the existing human hand labels in `eval_set.jsonl`; alignment "
        "to the offline corpus was empirically verified (see METHODOLOGY.md). "
        "These OpenAI dense numbers are a NEW baseline, distinct from the Mistral "
        "`eval_results/baseline.json` — do not conflate.",
        "",
        "| retriever | P@5 | P@10 | R@5 | R@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        a = results[n]["aggregate"]
        lines.append(
            f"| {n} | {a['precision_at_5']:.3f} | {a['precision_at_10']:.3f} | "
            f"{a['recall_at_5']:.3f} | {a['recall_at_10']:.3f} | {a['mrr']:.3f} |"
        )
    lines.append("")

    # Honest one-paragraph summary: which method won on P@5 and MRR.
    best_p5 = max(names, key=lambda n: results[n]["aggregate"]["precision_at_5"])
    best_mrr = max(names, key=lambda n: results[n]["aggregate"]["mrr"])
    dense_p5 = results["dense"]["aggregate"]["precision_at_5"]
    best_p5_val = results[best_p5]["aggregate"]["precision_at_5"]
    delta = best_p5_val - dense_p5
    lines += [
        "## Summary",
        "",
        f"On P@5 the best method is **{best_p5}** ({best_p5_val:.3f}) vs. dense "
        f"baseline {dense_p5:.3f} (Δ {delta:+.3f}). On MRR the best is **{best_mrr}** "
        f"({results[best_mrr]['aggregate']['mrr']:.3f}). Every number above is read "
        "from the committed per-retriever JSON receipts in this directory.",
        "",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(lines))


def print_summary(results: dict[str, dict]) -> None:
    t = Table(title="Offline retrievers — aggregate (negatives excluded)")
    t.add_column("retriever", style="bold")
    for col in ("P@5", "P@10", "R@5", "R@10", "MRR"):
        t.add_column(col, justify="right", style="cyan")
    for n in ("bm25", "dense", "graph", "hybrid", "hybrid_rerank"):
        if n not in results:
            continue
        a = results[n]["aggregate"]
        t.add_row(
            n,
            f"{a['precision_at_5']:.3f}",
            f"{a['precision_at_10']:.3f}",
            f"{a['recall_at_5']:.3f}",
            f"{a['recall_at_10']:.3f}",
            f"{a['mrr']:.3f}",
        )
    console.print(t)


def main() -> None:
    console.rule("[bold cyan]Offline multi-method retrieval benchmark (OpenAI)[/bold cyan]")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing {EMB_PATH}. Run `python -m scripts.embed_openai` first.")

    docs = load_docs()
    corpus_emb = np.load(EMB_PATH)
    queries = load_queries()
    console.print(
        f"[dim]{len(docs)} docs · {len(queries)} queries · embeddings {corpus_emb.shape}[/dim]\n"
    )

    retrievers = build_retrievers(docs, corpus_emb)
    results: dict[str, dict] = {}
    for name, search_fn in retrievers.items():
        console.print(f"[bold]Evaluating[/bold] {name} ...")
        res = eval_retriever(name, search_fn, queries)
        results[name] = res
        out_path = OUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(clean_nan(res), indent=2))
        console.print(f"  saved -> {out_path.relative_to(REPO_ROOT)}")

    console.print()
    print_summary(results)
    write_comparison(results)
    console.print(
        f"\n[bold green]Wrote comparison.md + {len(results)} JSON receipts -> "
        f"{OUT_DIR.relative_to(REPO_ROOT)}[/bold green]"
    )


if __name__ == "__main__":
    main()
