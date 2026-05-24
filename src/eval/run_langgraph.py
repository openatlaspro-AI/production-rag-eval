"""LangGraph multi-agent version of src.eval.run_all — same eval set, same metrics,
same JSON schema, but uses src.frameworks.langgraph_multiagent.langgraph_generate_rag_response
for end-to-end RAG via a 4-agent graph (Planner→Researcher→Critic→[loop≤1]→Writer).

Per-agent metrics are captured in each run dict under the "agent_calls" field, providing
a per-query breakdown of tokens, latency, and cost across every agent invocation.

Reuses (does not modify) every downstream eval module:
  - src.eval.cost, latency, llm_judge, refusal
  - src.eval.retrieval (precision/recall/MRR functions)
  - src.eval._retry (with_retry helper)

Output goes to eval_results/langgraph/ (parallel to eval_results/ for native).
Same schema → side-by-side comparison with eval_results/langchain/consolidated_report.json.

Single command: `python -m src.eval.run_langgraph` (also supports --smoke for a single-query sanity check).
"""

from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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
from src.eval._retry import with_retry  # noqa: F401  — kept for parity; superseded by with_retry_broad below
from src.eval.retrieval import hits_in_top_k, precision_at_k, recall_at_k, reciprocal_rank
from src.eval.run_baseline import CATEGORIES, categorize, clean_nan, safe_mean
from src.frameworks.langgraph_multiagent import langgraph_generate_rag_response

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_SET_PATH = REPO_ROOT / "eval_results" / "eval_set.jsonl"
OUTPUT_DIR = REPO_ROOT / "eval_results" / "langgraph"
CONSOLIDATED_PATH = OUTPUT_DIR / "consolidated_report.json"
RAW_RUNS_PATH = OUTPUT_DIR / ".raw_runs.json"  # checkpoint (gitignored)

GEN_MODELS = ["mistral-small-latest", "mistral-large-latest"]


def with_retry_broad(
    fn,
    max_retries: int = 6,
    initial_wait: float = 4.0,
    max_wait: float = 60.0,
):
    """Like src.eval._retry.with_retry but matches by *message content*, not exception type.

    Why: LangGraph (like LangChain) wraps the upstream mistralai.SDKError in its own
    exception class, so the type-discriminated `with_retry` (which catches
    `mistralai.SDKError`) never fires on 429s coming out of langchain_mistralai.
    Matching on '429' / 'rate' in `str(e)` rides through both raw-SDK and
    LangGraph-wrapped errors.
    """
    import random
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — intentionally broad
            msg = str(e).lower()
            is_rate_limit = "429" in msg or "rate_limited" in msg or "rate limit" in msg
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait = min(initial_wait * (2 ** attempt), max_wait)
            wait += random.uniform(0, 0.5 * wait)
            _time.sleep(wait)
            last_exc = e
    raise last_exc if last_exc else RuntimeError("with_retry_broad exhausted")


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
    client: Mistral,  # noqa: ARG001 — kept for signature parity with run_all.run_one_query
    q: dict,
    model: str,
    k: int = 5,
) -> dict:
    """Run end-to-end RAG (LangGraph multi-agent path) for one (query, model) pair."""
    qid = q["id"]
    category = categorize(qid)
    is_negative = category == "negative"

    with pool.connection() as conn:
        result = with_retry_broad(lambda: langgraph_generate_rag_response(
            q["query"],
            k=k,
            writer_model=model,
            helper_model="mistral-small-latest",
            conn=conn,
            return_agent_metrics=True,
        ))
        r, agent_calls = result

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
        "agent_calls":          agent_calls,
    }


def run_all_queries(queries: list[dict], models: list[str], k: int = 5, workers: int = 1) -> list[dict]:
    # workers=1 by default: free-tier mistral-large rate limit is tight and LangGraph's
    # wrapped error type defeats sub-second retry timing. See with_retry_broad docstring.
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
        TextColumn("[progress.description]LangGraph multi-agent RAG (parallel × {0})".format(workers)),
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
    """Identical logic to src.eval.run_all.compute_retrieval_metrics —
    retrieval is gen-model-independent so we pick one set of runs per query.
    """
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
        "framework": "langgraph",
    }


def _write_with_path_override(target_path: Path, payload: dict) -> None:
    """Write a single eval JSON to our langgraph/ subdir."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(clean_nan(payload), indent=2))


def main() -> None:
    if "--smoke" in sys.argv:
        console.rule("[bold cyan]LangGraph Multi-Agent — Smoke Test (1 query)[/bold cyan]")
        queries = load_eval_set()[:1]
        runs = run_all_queries(queries, ["mistral-small-latest"], k=5, workers=1)
        console.print_json(data=runs[0])
        console.print(
            f"[bold]Smoke cost:[/bold] ${runs[0]['total_usd']:.5f}  "
            f"wall={runs[0]['total_ms']:.0f}ms  "
            f"agent_calls={len(runs[0].get('agent_calls', []))}"
        )
        return

    console.rule("[bold cyan]LangGraph Multi-Agent RAG — Full Eval Suite[/bold cyan]")
    t_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_eval_set()
    console.print(f"  Loaded {len(queries)} queries from {EVAL_SET_PATH.name}")
    console.print(f"  Generators: {GEN_MODELS}")
    console.print(f"  Framework: [bold magenta]langgraph_multiagent[/bold magenta]  →  output dir: {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    console.print()

    # 1. Run all RAG generations (with top-up branch if checkpoint is incomplete)
    console.rule("[bold]Step 1: Run LangGraph Multi-Agent RAG (60 query-runs target, workers=1, ~6 LLM calls per query)[/bold]")
    expected_combos = {(q["id"], m) for q in queries for m in GEN_MODELS}
    runs: list[dict] = []

    if RAW_RUNS_PATH.exists():
        import os
        age_min = (time.time() - os.path.getmtime(RAW_RUNS_PATH)) / 60
        if age_min < 60:
            runs = json.loads(RAW_RUNS_PATH.read_text())
            console.print(f"  [yellow]Loaded checkpoint (age {age_min:.1f}min, {len(runs)} runs)[/yellow]")
        else:
            console.print(f"  [dim]Checkpoint stale ({age_min:.0f}min), starting fresh.[/dim]")

    have_combos = {(r["query_id"], r["model"]) for r in runs}
    missing = sorted(expected_combos - have_combos)
    if missing:
        console.print(f"  [yellow]Missing {len(missing)} (query, model) combos — topping up at workers=1[/yellow]")
        missing_queries_by_model: dict[str, list[dict]] = {m: [] for m in GEN_MODELS}
        for qid, m in missing:
            q = next(qq for qq in queries if qq["id"] == qid)
            missing_queries_by_model[m].append(q)
        for m, qs in missing_queries_by_model.items():
            if not qs:
                continue
            new_runs = run_all_queries(qs, [m], k=5, workers=1)
            runs.extend(new_runs)
            RAW_RUNS_PATH.write_text(json.dumps(runs, indent=2))  # checkpoint after each model
            console.print(f"  [green]+{len(new_runs)} {m} runs[/green]  ({len(runs)}/{len(expected_combos)} total)")
        runs.sort(key=lambda r: (r["query_id"], r["model"]))
    else:
        console.print(f"  [green]Checkpoint complete — {len(runs)} runs, no top-up needed[/green]")
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

    # 5. LLM-judge (sequential — Mistral free-tier mistral-large rate limit)
    console.rule("[bold]Step 5: LLM-judge (sequential, with retry)[/bold]")
    judge = judge_mod.run_judge(runs, workers=1)
    judge_mod.print_judge_table(judge)
    console.print()

    # 6. Refusal eval
    console.rule("[bold]Step 6: Refusal eval[/bold]")
    refusal = refusal_mod.compute_refusal(runs, workers=1)
    refusal_mod.print_refusal_table(refusal)
    console.print()

    # 7. Assemble + save (to langgraph/ subdir)
    total_wall_s = time.perf_counter() - t_start
    consolidated = {
        "metadata": {
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "framework":          "langgraph_multiagent",
            "total_wall_seconds": total_wall_s,
            "embed_model":        settings.mistral_embed_model,
            "generator_models":   GEN_MODELS,
            "judge_model":        judge["judge_model"],
            "corpus_size":        400,
            "vector_dim":         1024,
            "similarity":         "cosine (pgvector HNSW)",
            "k_retrieval":        5,
            "eval_set":           str(EVAL_SET_PATH.relative_to(REPO_ROOT)),
            "note_embed_tokens":  (
                "embed_tokens_in is summed across all sub-question embeddings "
                "(1-3 per query, plus 1-2 more if the critic looped). "
                "Per-agent breakdown in raw_runs[*].agent_calls."
            ),
        },
        "retrieval": retrieval,
        "latency":   latency,
        "cost":      cost,
        "llm_judge": judge,
        "refusal":   refusal,
        "raw_runs":  runs,
    }

    _write_with_path_override(CONSOLIDATED_PATH, consolidated)
    _write_with_path_override(OUTPUT_DIR / "baseline.json", retrieval)
    _write_with_path_override(OUTPUT_DIR / "latency.json", latency)
    _write_with_path_override(OUTPUT_DIR / "cost_summary.json", cost)
    _write_with_path_override(OUTPUT_DIR / "llm_judge.json", judge)
    _write_with_path_override(OUTPUT_DIR / "refusal.json", refusal)

    console.rule(f"[bold green]Done in {total_wall_s:.1f}s[/bold green]")
    console.print(f"  Total eval cost: [bold]${cost['total_eval_usd']:.4f}[/bold]")
    console.print(f"  Consolidated → {CONSOLIDATED_PATH.relative_to(REPO_ROOT)}")
    console.print(f"  Compare to native: eval_results/consolidated_report.json")
    console.print(f"  Also compare to: eval_results/langchain/consolidated_report.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        sys.exit(1)
