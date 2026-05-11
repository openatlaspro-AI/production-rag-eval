"""Latency percentiles per pipeline stage.

Pure analysis: takes a list of run dicts (each with embed_ms / retrieve_ms /
generate_ms / total_ms / model) and computes p50/p95/p99 per stage, broken
down by generator model.

CLI: `python -m src.eval.latency` reads `eval_results/consolidated_report.json`
and recomputes — useful for re-analysis without re-running the API. If the
consolidated report doesn't exist, instruct user to run `make eval` first.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import median

from rich.console import Console
from rich.table import Table

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "eval_results" / "latency.json"
CONSOLIDATED_PATH = REPO_ROOT / "eval_results" / "consolidated_report.json"


def percentile(values: list[float], pct: float) -> float:
    """Compute a percentile (0-100) via linear interpolation. NaN if empty."""
    if not values:
        return math.nan
    vs = sorted(values)
    if len(vs) == 1:
        return vs[0]
    k = (len(vs) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    if f == c:
        return vs[f]
    return vs[f] * (c - k) + vs[c] * (k - f)


def compute_latency(runs: list[dict]) -> dict:
    """Compute p50/p95/p99 per stage and per generator model.

    `runs` schema (each item):
      {
        "model": "mistral-small-latest" | "mistral-large-latest",
        "embed_ms": float, "retrieve_ms": float,
        "generate_ms": float, "total_ms": float,
        ...
      }

    Embed and retrieve are gen-model-independent (same across both models for
    the same query) — we still compute per-model just to confirm, but the
    aggregate is the meaningful number.
    """
    models = sorted({r["model"] for r in runs})
    result = {
        "stages": {},
        "total_by_model": {},
        "by_model_per_stage": {},
    }

    # Aggregate embed + retrieve (model-independent)
    for stage in ("embed_ms", "retrieve_ms"):
        vals = [r[stage] for r in runs]
        result["stages"][stage] = {
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    # generate_ms and total_ms — split per model
    for stage in ("generate_ms", "total_ms"):
        result["by_model_per_stage"][stage] = {}
        for model in models:
            vals = [r[stage] for r in runs if r["model"] == model]
            result["by_model_per_stage"][stage][model] = {
                "p50": percentile(vals, 50),
                "p95": percentile(vals, 95),
                "p99": percentile(vals, 99),
                "min": min(vals) if vals else math.nan,
                "max": max(vals) if vals else math.nan,
                "n": len(vals),
            }

    # Total per model — flatten for the README headline
    for model in models:
        vals = [r["total_ms"] for r in runs if r["model"] == model]
        result["total_by_model"][model] = {
            "p50": percentile(vals, 50),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
        }

    return result


def print_latency_table(lat: dict) -> None:
    t = Table(title="Latency percentiles (ms)", show_lines=False)
    t.add_column("stage", style="bold")
    t.add_column("model", style="green")
    t.add_column("n", justify="right")
    t.add_column("p50", justify="right", style="cyan")
    t.add_column("p95", justify="right", style="cyan")
    t.add_column("p99", justify="right", style="cyan")
    t.add_column("min", justify="right", style="dim")
    t.add_column("max", justify="right", style="dim")

    for stage in ("embed_ms", "retrieve_ms"):
        s = lat["stages"][stage]
        t.add_row(
            stage, "—", str(s["n"]),
            f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}",
            f"{s['min']:.1f}", f"{s['max']:.1f}",
        )
    for stage in ("generate_ms", "total_ms"):
        by_m = lat["by_model_per_stage"][stage]
        for model, s in by_m.items():
            t.add_row(
                stage, model, str(s["n"]),
                f"{s['p50']:.1f}", f"{s['p95']:.1f}", f"{s['p99']:.1f}",
                f"{s['min']:.1f}", f"{s['max']:.1f}",
            )
    console.print(t)


def main() -> None:
    if not CONSOLIDATED_PATH.exists():
        console.print(
            f"[red]No consolidated report at {CONSOLIDATED_PATH.name} — run `make eval` first.[/red]"
        )
        sys.exit(1)
    report = json.loads(CONSOLIDATED_PATH.read_text())
    runs = report["raw_runs"]
    lat = compute_latency(runs)
    print_latency_table(lat)
    OUTPUT_PATH.write_text(json.dumps(lat, indent=2))
    console.print(f"\n[bold green]Saved → {OUTPUT_PATH.relative_to(REPO_ROOT)}[/bold green]")


if __name__ == "__main__":
    main()
