"""Full eval suite orchestrator.

Pipeline:
  1. Run all 30 queries through end-to-end RAG with BOTH generator models
     (mistral-small-latest + mistral-large-latest). Uses thread pool for
     concurrent API calls. Collects raw runs.
  2. Compute retrieval metrics (precision/recall/MRR) — re-runs baseline logic
     on the small-model runs (retrieval is gen-model-independent).
  3. Compute latency percentiles per stage per model.
  4. Compute cost aggregates per model.
  5. Run LLM-judge on positive (non-negative-test) runs.
  6. Run refusal eval on negative-test runs.
  7. Save individual JSON files + consolidated_report.json.
  8. Print summary tables.

Single command: `python -m src.eval.run_all` (also wired to `make eval`).
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from mistralai import Mistral
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from src.config import settings
from src.eval import cost as cost_mod
from src.eval import latency as latency_mod
from src.eval import llm_judge as judge_mod
from src.eval import refusal as refusal_mod
from src.eval._retry import with_retry
from src.eval.retrieval import hits_in_top_k, precision_at_k, recall_at_k, reciprocal_rank
from src.generate import generate_rag_response
from src.eval.run_baseline import CATEGORIES, categorize, safe_mean, clean_nan

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval_results" / "eval_set.jsonl"
CONSOLIDATED_PATH = REPO_ROOT / "eval_results" / "consolidated_report.json"
BASELINE_PATH = REPO_ROOT / "eval_results" / "baseline.json"
RAW_RUNS_PATH = REPO_ROOT / "eval_results" / ".raw_runs.json"  # checkpoint (gitignored)

GEN_MODELS = ["mistral-small-latest", "mistral-large-latest"]


def load_eval_set() -> list[dict]:
    queries = []
    with open(EVAL_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def run_one_query(
    pool: ConnectionPool,
    client: Mistral,
    q: dict,
    model: str,
    k: int = 5,
) -> dict:
    """Run end-to-end RAG for one (query, model) pair using a pool connection."""
    qid = q["id"]
    category = categorize(qid)
    is_negative = category == "negative"

    with pool.connection() as conn:
        r = with_retry(lambda: generate_rag_response(
            q["query"],
            k=k,
            model=model,
            client=client,
            conn=conn,
        ))

    sources_payload = [
        {
            "id": h.id, "title_en": h.title_en, "title": h.title,
            "platform": h.platform, "source": h.source,
            "similarity": h.similarity,
        }
        for h in r.sources
    ]

    return {
        "query_id":             qid,
        "query":                q["query"],
        "category":             category,
        "is_negative":          is_negative,
        "model":                model,
        "answer":               r.answer,
        "sources":              sources_payload,
        "retrieved_top_k_ids":  [h.id for h in r.sources],
        "embed_ms":             r.timing.embed_ms,
        "retrieve_ms":          r.timing.retrieve_ms,
        "generate_ms":          r.timing.generate_ms,
        "total_ms":             r.timing.total_ms,
        "embed_usd":            r.cost.embed_usd,
        "generate_usd":         r.cost.generate_usd,
        "total_usd":            r.cost.total_usd,
        "embed_tokens_in":      r.cost.embed_tokens_in,
        "generate_tokens_in":   r.cost.generate_tokens_in,
        "generate_tokens_out":  r.cost.generate_tokens_out,
        "relevant_doc_ids":     q["relevant_doc_ids"],
    }


def run_all_queries(queries: list[dict], models: list[str], k: int = 5, workers: int = 4) -> list[dict]:
    """Run all (query, model) combinations in parallel. Returns list of run dicts."""
    client = Mistral(api_key=settings.mistral_api_key)

    def _configure(conn):
        register_vector(conn)

    pool = ConnectionPool(
        settings.postgres_dsn,
        min_size=2, max_size=workers + 2,
        configure=_configure,
        open=True,
    )
    pool.wait()

    all_tasks = [(q, m) for m in models for q in queries]
    runs: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]running RAG (parallel × {0})".format(workers)),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("rag", total=len(all_tasks))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_one_query, pool, client, q, m, k) for q, m in all_tasks]
            for f in as_completed(futures):
                try:
                    runs.append(f.result())
                except Exception as e:
                    console.print(f"[red]Task failed: {e}[/red]")
                progress.update(task, advance=1)

    pool.close()
    runs.sort(key=lambda r: (r["query_id"], r["model"]))
    return runs


def compute_retrieval_metrics(runs: list[dict]) -> dict:
    """Compute retrieval metrics using only one set of runs (retrieval is model-independent).
    We use the small-model runs by convention."""
    # Take one run per query_id (retrieval is the same regardless of gen model)
    seen = set()
    selected = []
    for r in sorted(runs, key=lambda x: x["query_id"]):
        if r["query_id"] in seen:
            continue
        seen.add(r["query_id"])
        selected.append(r)

    per_query = []
    for r in selected:
        qid = r["query_id"]
        relevant_set = set(r["relevant_doc_ids"])
        retrieved_ids = r["retrieved_top_k_ids"]
        is_negative = r["is_negative"]
        category = r["category"]

        if is_negative:
            row = {
                "id": qid, "query": r["query"], "category": category,
                "n_relevant": 0, "retrieved_top_k": retrieved_ids,
                "hits_in_top_5": None, "precision_at_5": math.nan,
                "recall_at_5": math.nan, "mrr": math.nan,
                "embed_ms": r["embed_ms"], "retrieve_ms": r["retrieve_ms"],
            }
        else:
            row = {
                "id": qid, "query": r["query"], "category": category,
                "n_relevant": len(relevant_set), "retrieved_top_k": retrieved_ids,
                "hits_in_top_5": hits_in_top_k(relevant_set, retrieved_ids, 5),
                "precision_at_5": precision_at_k(relevant_set, retrieved_ids, 5),
                "recall_at_5": recall_at_k(relevant_set, retrieved_ids, 5),
                "mrr": reciprocal_rank(relevant_set, retrieved_ids),
                "embed_ms": r["embed_ms"], "retrieve_ms": r["retrieve_ms"],
            }
        per_query.append(row)

    pos = [q for q in per_query if q["category"] != "negative"]
    aggregate = {
        "n_queries_total":     len(per_query),
        "n_queries_evaluated": len(pos),
        "n_negative_tests":    len(per_query) - len(pos),
        "precision_at_5":      safe_mean([q["precision_at_5"] for q in pos]),
        "recall_at_5":         safe_mean([q["recall_at_5"] for q in pos]),
        "mrr":                 safe_mean([q["mrr"] for q in pos]),
        # Note: with k=5 in run_all (smaller than k=10 in run_baseline),
        # we don't report precision@10 / recall@10 here.
    }
    by_category = {}
    for cat in CATEGORIES:
        cat_queries = [q for q in per_query if q["category"] == cat]
        n = len(cat_queries)
        if cat == "negative":
            by_category[cat] = {"n_queries": n}
            continue
        if not cat_queries:
            by_category[cat] = {"n_queries": 0}
            continue
        by_category[cat] = {
            "n_queries":      n,
            "precision_at_5": safe_mean([q["precision_at_5"] for q in cat_queries]),
            "recall_at_5":    safe_mean([q["recall_at_5"] for q in cat_queries]),
            "mrr":            safe_mean([q["mrr"] for q in cat_queries]),
        }

    return {
        "aggregate": aggregate, "by_category": by_category, "per_query": per_query,
        "k_used_in_run_all": 5,
        "note": "For deeper recall@10 metrics, see eval_results/baseline.json (k=10).",
    }


def main() -> None:
    console.rule("[bold cyan]FamilyHQ RAG — Full Eval Suite[/bold cyan]")
    t_start = time.perf_counter()

    queries = load_eval_set()
    console.print(f"  Loaded {len(queries)} queries from {EVAL_SET_PATH.name}")
    console.print(f"  Generators: {GEN_MODELS}")
    console.print()

    # 1. Run all RAG generations (workers=2 to stay under Mistral rate limits)
    #    Checkpoint to .raw_runs.json so judge/refusal steps can resume if they fail.
    console.rule("[bold]Step 1: Run RAG (60 calls, parallel ×2 with retry)[/bold]")
    if RAW_RUNS_PATH.exists():
        import os
        age_min = (time.time() - os.path.getmtime(RAW_RUNS_PATH)) / 60
        if age_min < 60:
            runs = json.loads(RAW_RUNS_PATH.read_text())
            console.print(f"  [yellow]Resumed from checkpoint (age {age_min:.1f}min, {len(runs)} runs)[/yellow]")
        else:
            console.print(f"  [dim]Checkpoint stale ({age_min:.0f}min), re-running…[/dim]")
            runs = run_all_queries(queries, GEN_MODELS, k=5, workers=2)
            RAW_RUNS_PATH.write_text(json.dumps(runs, indent=2))
            console.print(f"  Collected [bold]{len(runs)}[/bold] runs (checkpointed)")
    else:
        runs = run_all_queries(queries, GEN_MODELS, k=5, workers=2)
        RAW_RUNS_PATH.write_text(json.dumps(runs, indent=2))
        console.print(f"  Collected [bold]{len(runs)}[/bold] runs (checkpointed)")
    console.print()

    # 2. Retrieval metrics
    console.rule("[bold]Step 2: Retrieval metrics[/bold]")
    retrieval = compute_retrieval_metrics(runs)
    console.print(
        f"  P@5={retrieval['aggregate']['precision_at_5']:.3f}  "
        f"R@5={retrieval['aggregate']['recall_at_5']:.3f}  "
        f"MRR={retrieval['aggregate']['mrr']:.3f}"
    )
    console.print()

    # 3. Latency
    console.rule("[bold]Step 3: Latency percentiles[/bold]")
    latency = latency_mod.compute_latency(runs)
    latency_mod.print_latency_table(latency)
    console.print()

    # 4. Cost
    console.rule("[bold]Step 4: Cost aggregation[/bold]")
    cost = cost_mod.compute_cost(runs)
    cost_mod.print_cost_table(cost)
    console.print()

    # 5. LLM-judge (workers=1, sequential — Mistral free-tier mistral-large
    #    rate limit is the dominant constraint; concurrent calls trip 429s
    #    faster than retry-backoff can recover.)
    console.rule("[bold]Step 5: LLM-judge (sequential, with retry)[/bold]")
    judge = judge_mod.run_judge(runs, workers=1)
    judge_mod.print_judge_table(judge)
    console.print()

    # 6. Refusal eval (small N, sequential)
    console.rule("[bold]Step 6: Refusal eval[/bold]")
    refusal = refusal_mod.compute_refusal(runs, workers=1)
    refusal_mod.print_refusal_table(refusal)
    console.print()

    # 7. Assemble + save
    total_wall_s = time.perf_counter() - t_start
    consolidated = {
        "metadata": {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "total_wall_seconds": total_wall_s,
            "embed_model":        settings.mistral_embed_model,
            "generator_models":   GEN_MODELS,
            "judge_model":        judge["judge_model"],
            "corpus_size":        400,
            "vector_dim":         1024,
            "similarity":         "cosine (pgvector HNSW)",
            "k_retrieval":        5,
            "eval_set":           str(EVAL_SET_PATH.relative_to(REPO_ROOT)),
        },
        "retrieval": retrieval,
        "latency":   latency,
        "cost":      cost,
        "llm_judge": judge,
        "refusal":   refusal,
        "raw_runs":  runs,
    }

    # Save consolidated + individual files
    CONSOLIDATED_PATH.write_text(json.dumps(clean_nan(consolidated), indent=2))
    (REPO_ROOT / "eval_results" / "latency.json").write_text(json.dumps(clean_nan(latency), indent=2))
    (REPO_ROOT / "eval_results" / "cost_summary.json").write_text(json.dumps(clean_nan(cost), indent=2))
    (REPO_ROOT / "eval_results" / "llm_judge.json").write_text(json.dumps(clean_nan(judge), indent=2))
    (REPO_ROOT / "eval_results" / "refusal.json").write_text(json.dumps(clean_nan(refusal), indent=2))

    console.rule(f"[bold green]Done in {total_wall_s:.1f}s[/bold green]")
    console.print(f"  Total eval cost: [bold]${cost['total_eval_usd']:.4f}[/bold]")
    console.print(f"  Consolidated → eval_results/consolidated_report.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(1)
