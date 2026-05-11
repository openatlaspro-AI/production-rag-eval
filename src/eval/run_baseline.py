"""Run retrieval baseline eval over eval_set.jsonl. Saves eval_results/baseline.json.

Pipeline:
  1. Load 30 labeled query records from eval_results/eval_set.jsonl
  2. For each query: embed (mistral-embed) + pgvector_search(k=10)
  3. Compute precision@{5,10}, recall@{5,10}, MRR
  4. Negative-test queries (empty relevant_doc_ids) are excluded from
     aggregate metrics — they'll be evaluated separately via generation
     refusal in a future eval.
  5. Aggregate overall + by category (broad/specific/multi_aspect/family_niche)
  6. Print summary tables + write baseline.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import psycopg
from mistralai import Mistral
from pgvector.psycopg import register_vector
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.eval.retrieval import (
    hits_in_top_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieve import embed_query, pgvector_search

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval_results" / "eval_set.jsonl"
OUTPUT_PATH = REPO_ROOT / "eval_results" / "baseline.json"

CATEGORIES = {
    "broad":         set(range(1, 9)),    # IDs 1-8
    "specific":      set(range(9, 19)),   # IDs 9-18
    "multi_aspect":  set(range(19, 24)),  # IDs 19-23
    "negative":      set(range(24, 27)),  # IDs 24-26 (excluded from aggregate)
    "family_niche":  set(range(27, 31)),  # IDs 27-30
}


def categorize(qid: int) -> str:
    for name, ids in CATEGORIES.items():
        if qid in ids:
            return name
    return "uncategorized"


def clean_nan(obj):
    """Recursively replace NaN with None for JSON serialization (NaN is not valid JSON)."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    return obj


def safe_mean(values: list[float]) -> float:
    """Mean ignoring NaN. Returns NaN if all values are NaN."""
    finite = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return mean(finite) if finite else math.nan


def run_eval(k_max: int = 10) -> dict:
    # Load eval set
    queries = []
    with open(EVAL_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    console.print(f"[dim]Loaded {len(queries)} queries from {EVAL_SET_PATH.name}[/dim]\n")

    # Shared client + connection across all queries
    client = Mistral(api_key=settings.mistral_api_key)
    conn = psycopg.connect(settings.postgres_dsn)
    register_vector(conn)

    per_query: list[dict] = []
    t_total_start = time.perf_counter()

    try:
        for q in queries:
            qid = q["id"]
            query = q["query"]
            relevant_set = set(q["relevant_doc_ids"])
            category = categorize(qid)
            is_negative = category == "negative"

            # Embed + search
            t0 = time.perf_counter()
            q_emb, _ = embed_query(client, query)
            embed_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            hits = pgvector_search(conn, q_emb, k=k_max)
            retrieve_ms = (time.perf_counter() - t1) * 1000

            retrieved_ids = [h.id for h in hits]

            if is_negative:
                # Skip metrics for negative tests — generation refusal eval is separate
                row = {
                    "id": qid, "query": query, "category": category,
                    "n_relevant": 0, "retrieved_top_10": retrieved_ids,
                    "hits_in_top_5": None, "hits_in_top_10": None,
                    "precision_at_5": math.nan, "precision_at_10": math.nan,
                    "recall_at_5": math.nan, "recall_at_10": math.nan,
                    "mrr": math.nan,
                    "embed_ms": embed_ms, "retrieve_ms": retrieve_ms,
                }
            else:
                row = {
                    "id": qid, "query": query, "category": category,
                    "n_relevant": len(relevant_set), "retrieved_top_10": retrieved_ids,
                    "hits_in_top_5": hits_in_top_k(relevant_set, retrieved_ids, 5),
                    "hits_in_top_10": hits_in_top_k(relevant_set, retrieved_ids, 10),
                    "precision_at_5": precision_at_k(relevant_set, retrieved_ids, 5),
                    "precision_at_10": precision_at_k(relevant_set, retrieved_ids, 10),
                    "recall_at_5": recall_at_k(relevant_set, retrieved_ids, 5),
                    "recall_at_10": recall_at_k(relevant_set, retrieved_ids, 10),
                    "mrr": reciprocal_rank(relevant_set, retrieved_ids),
                    "embed_ms": embed_ms, "retrieve_ms": retrieve_ms,
                }
            per_query.append(row)

            status = "[dim italic](neg)[/dim italic]" if is_negative else (
                f"P@5={row['precision_at_5']:.2f} R@5={row['recall_at_5']:.2f} MRR={row['mrr']:.2f}"
            )
            console.print(
                f"  Q{qid:>2}  [{category:>12}]  {status}  "
                f"[dim]{query[:55]}[/dim]"
            )
    finally:
        conn.close()

    t_total_ms = (time.perf_counter() - t_total_start) * 1000

    # Aggregate (excluding negative tests)
    pos = [q for q in per_query if q["category"] != "negative"]
    aggregate = {
        "n_queries_total":       len(per_query),
        "n_queries_evaluated":   len(pos),
        "n_negative_tests":      len(per_query) - len(pos),
        "precision_at_5":        safe_mean([q["precision_at_5"] for q in pos]),
        "precision_at_10":       safe_mean([q["precision_at_10"] for q in pos]),
        "recall_at_5":           safe_mean([q["recall_at_5"] for q in pos]),
        "recall_at_10":          safe_mean([q["recall_at_10"] for q in pos]),
        "mrr":                   safe_mean([q["mrr"] for q in pos]),
        "median_embed_ms":       median([q["embed_ms"] for q in per_query]),
        "median_retrieve_ms":    median([q["retrieve_ms"] for q in per_query]),
        "total_wall_ms":         t_total_ms,
    }

    by_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_queries = [q for q in per_query if q["category"] == cat]
        n = len(cat_queries)
        if cat == "negative":
            by_category[cat] = {"n_queries": n, "note": "excluded from precision/recall/MRR — separate refusal eval"}
            continue
        if not cat_queries:
            by_category[cat] = {"n_queries": 0}
            continue
        by_category[cat] = {
            "n_queries":         n,
            "precision_at_5":    safe_mean([q["precision_at_5"] for q in cat_queries]),
            "precision_at_10":   safe_mean([q["precision_at_10"] for q in cat_queries]),
            "recall_at_5":       safe_mean([q["recall_at_5"] for q in cat_queries]),
            "recall_at_10":      safe_mean([q["recall_at_10"] for q in cat_queries]),
            "mrr":               safe_mean([q["mrr"] for q in cat_queries]),
        }

    result = {
        "metadata": {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "model_embed":       settings.mistral_embed_model,
            "k_max":             k_max,
            "corpus_size":       400,
            "vector_dim":        1024,
            "similarity_metric": "cosine",
            "index":             "hnsw (m=16, ef_construction=64)",
            "eval_set_path":     str(EVAL_SET_PATH.relative_to(REPO_ROOT)),
        },
        "aggregate":   aggregate,
        "by_category": by_category,
        "per_query":   per_query,
    }
    return result


def print_summary(result: dict) -> None:
    agg = result["aggregate"]

    # ---- Aggregate table ----
    t = Table(title="Aggregate retrieval metrics (negative tests excluded)", show_lines=False)
    t.add_column("metric", style="bold")
    t.add_column("value", justify="right", style="cyan")
    t.add_row("n queries (total)",      str(agg["n_queries_total"]))
    t.add_row("n queries (evaluated)",  str(agg["n_queries_evaluated"]))
    t.add_row("n negative tests",       str(agg["n_negative_tests"]))
    t.add_row("precision@5",            f"{agg['precision_at_5']:.4f}")
    t.add_row("precision@10",           f"{agg['precision_at_10']:.4f}")
    t.add_row("recall@5",               f"{agg['recall_at_5']:.4f}")
    t.add_row("recall@10",              f"{agg['recall_at_10']:.4f}")
    t.add_row("MRR",                    f"{agg['mrr']:.4f}")
    t.add_row("median embed_ms",        f"{agg['median_embed_ms']:.1f}")
    t.add_row("median retrieve_ms",     f"{agg['median_retrieve_ms']:.1f}")
    t.add_row("total wall time (s)",    f"{agg['total_wall_ms']/1000:.1f}")
    console.print(t)

    # ---- Per-category table ----
    by_cat = result["by_category"]
    t2 = Table(title="Per-category metrics", show_lines=False)
    t2.add_column("category", style="bold green")
    t2.add_column("n", justify="right")
    t2.add_column("P@5", justify="right", style="cyan")
    t2.add_column("P@10", justify="right", style="cyan")
    t2.add_column("R@5", justify="right", style="magenta")
    t2.add_column("R@10", justify="right", style="magenta")
    t2.add_column("MRR", justify="right", style="yellow")
    for cat in ("broad", "specific", "multi_aspect", "family_niche", "negative"):
        d = by_cat.get(cat, {})
        n = d.get("n_queries", 0)
        if cat == "negative" or "precision_at_5" not in d:
            t2.add_row(cat, str(n), "—", "—", "—", "—", "—")
            continue
        t2.add_row(
            cat, str(n),
            f"{d['precision_at_5']:.3f}",
            f"{d['precision_at_10']:.3f}",
            f"{d['recall_at_5']:.3f}",
            f"{d['recall_at_10']:.3f}",
            f"{d['mrr']:.3f}",
        )
    console.print(t2)


def main() -> None:
    console.rule("[bold cyan]FamilyHQ RAG — Retrieval Baseline[/bold cyan]")
    result = run_eval(k_max=10)
    console.print()
    print_summary(result)
    OUTPUT_PATH.write_text(json.dumps(clean_nan(result), indent=2))
    console.print(f"\n[bold green]Saved → {OUTPUT_PATH.relative_to(REPO_ROOT)}[/bold green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(1)
