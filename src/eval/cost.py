"""Cost aggregation across the eval run.

Pure analysis: takes a list of run dicts and computes:
- Per-query cost breakdown (embed / generate / total)
- Per-model aggregate (mean, median, total across eval)
- Embed cost is gen-model-independent (same query embedding regardless of gen model)
- Total eval cost = unique embed costs + sum(generate costs across all models)

CLI: `python -m src.eval.cost` reads consolidated_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, median

from rich.console import Console
from rich.table import Table

from src.pricing import PRICING, format_cost

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "eval_results" / "cost_summary.json"
CONSOLIDATED_PATH = REPO_ROOT / "eval_results" / "consolidated_report.json"


def compute_cost(runs: list[dict]) -> dict:
    """Compute cost breakdown across all runs.

    `runs` schema (each item):
      {
        "model": str,
        "query_id": int,
        "embed_usd": float,
        "generate_usd": float,
        "total_usd": float,
        "embed_tokens_in": int,
        "generate_tokens_in": int,
        "generate_tokens_out": int,
      }
    """
    models = sorted({r["model"] for r in runs})
    result = {"per_model": {}, "pricing_used": PRICING}

    # Embed cost: each unique query is embedded once across both models.
    # We sum unique embed costs (deduped by query_id).
    seen_embed = set()
    embed_total_usd = 0.0
    embed_total_tokens = 0
    for r in runs:
        qid = r["query_id"]
        if qid not in seen_embed:
            seen_embed.add(qid)
            embed_total_usd += r["embed_usd"]
            embed_total_tokens += r["embed_tokens_in"]
    result["embed_total_usd"] = embed_total_usd
    result["embed_total_tokens"] = embed_total_tokens
    result["unique_queries_embedded"] = len(seen_embed)

    # Generation cost per model
    gen_total_usd = 0.0
    for model in models:
        m_runs = [r for r in runs if r["model"] == model]
        gen_costs = [r["generate_usd"] for r in m_runs]
        gen_tokens_in = [r["generate_tokens_in"] for r in m_runs]
        gen_tokens_out = [r["generate_tokens_out"] for r in m_runs]
        m_total_gen = sum(gen_costs)
        gen_total_usd += m_total_gen

        result["per_model"][model] = {
            "n_queries":              len(m_runs),
            "generate_total_usd":     m_total_gen,
            "generate_mean_per_query": mean(gen_costs) if gen_costs else 0,
            "generate_median_per_query": median(gen_costs) if gen_costs else 0,
            "tokens_in_total":        sum(gen_tokens_in),
            "tokens_out_total":       sum(gen_tokens_out),
            "tokens_in_mean":         mean(gen_tokens_in) if gen_tokens_in else 0,
            "tokens_out_mean":        mean(gen_tokens_out) if gen_tokens_out else 0,
        }

    result["generate_total_usd"] = gen_total_usd
    result["total_eval_usd"] = embed_total_usd + gen_total_usd
    return result


def print_cost_table(c: dict) -> None:
    t = Table(title="Cost summary (USD)", show_lines=False)
    t.add_column("model", style="bold green")
    t.add_column("n", justify="right")
    t.add_column("gen mean/query", justify="right", style="cyan")
    t.add_column("gen median/query", justify="right", style="cyan")
    t.add_column("gen total", justify="right", style="cyan")
    t.add_column("avg tokens in", justify="right", style="dim")
    t.add_column("avg tokens out", justify="right", style="dim")

    for model, m in c["per_model"].items():
        t.add_row(
            model, str(m["n_queries"]),
            format_cost(m["generate_mean_per_query"]),
            format_cost(m["generate_median_per_query"]),
            format_cost(m["generate_total_usd"]),
            f"{m['tokens_in_mean']:.0f}",
            f"{m['tokens_out_mean']:.0f}",
        )
    console.print(t)

    # Summary box
    summary = Table(title="Eval totals", show_lines=False)
    summary.add_column("item", style="bold")
    summary.add_column("value", justify="right", style="cyan")
    summary.add_row("unique queries embedded", str(c["unique_queries_embedded"]))
    summary.add_row("embed total cost",        format_cost(c["embed_total_usd"]))
    summary.add_row("embed total tokens",      str(c["embed_total_tokens"]))
    summary.add_row("generate total cost",     format_cost(c["generate_total_usd"]))
    summary.add_row("[bold]TOTAL eval cost",   f"[bold]{format_cost(c['total_eval_usd'])}")
    console.print(summary)


def main() -> None:
    if not CONSOLIDATED_PATH.exists():
        console.print(
            f"[red]No consolidated report at {CONSOLIDATED_PATH.name} — run `make eval` first.[/red]"
        )
        sys.exit(1)
    report = json.loads(CONSOLIDATED_PATH.read_text())
    runs = report["raw_runs"]
    c = compute_cost(runs)
    print_cost_table(c)
    OUTPUT_PATH.write_text(json.dumps(c, indent=2))
    console.print(f"\n[bold green]Saved → {OUTPUT_PATH.relative_to(REPO_ROOT)}[/bold green]")


if __name__ == "__main__":
    main()
