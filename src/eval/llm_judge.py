"""Mistral-as-judge scoring on a 5-dimension rubric.

Each generated RAG response is scored 1-5 on five dimensions by a strong
judge model (mistral-large-latest by default). Outputs JSON-mode response
for reliable parsing.

Rubric:
  groundedness       — are claims supported by retrieved sources?
  relevance          — does answer address the query?
  completeness       — does it use the most relevant available sources?
  conciseness        — appropriately brief?
  citation_accuracy  — are [N] citations used correctly?
  overall            — judge's holistic score
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

from mistralai import Mistral
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from src.config import settings
from src.eval._retry import with_retry

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "eval_results" / "llm_judge.json"
CONSOLIDATED_PATH = REPO_ROOT / "eval_results" / "consolidated_report.json"

JUDGE_MODEL = "mistral-large-latest"
RUBRIC_DIMS = ["groundedness", "relevance", "completeness", "conciseness", "citation_accuracy", "overall"]

JUDGE_SYSTEM = """You are a strict evaluator of RAG (retrieval-augmented generation) system outputs. Output ONLY a valid JSON object, no prose, no markdown fences.

Rubric — score each dimension 1 to 5 (integers only):

- groundedness: Are claims in the answer supported by the provided sources?
  1 = pure hallucination; 3 = mix of grounded and invented; 5 = every claim traceable to a cited source.

- relevance: Does the answer address what the user asked?
  1 = off-topic; 3 = partial; 5 = directly addresses the query.

- completeness: Does the answer use the most relevant available sources?
  1 = ignores key context; 3 = uses some relevant context; 5 = incorporates all genuinely relevant sources.

- conciseness: Is the answer appropriately brief for the query?
  1 = verbose padding; 3 = okay length; 5 = tight and to the point.

- citation_accuracy: Are inline citations like [1], [2] used correctly?
  1 = citations missing or wrong; 3 = mostly right with some errors; 5 = every claim cited correctly.

- overall: Your holistic 1-5 score (not necessarily the mean).

Output schema:
{
  "groundedness": int,
  "relevance": int,
  "completeness": int,
  "conciseness": int,
  "citation_accuracy": int,
  "overall": int,
  "rationale": "one-sentence justification"
}"""


def build_judge_user_message(query: str, sources: list[dict], answer: str) -> str:
    """Format the inputs the judge needs to see."""
    src_lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title_en") or s.get("title") or "(no title)"
        platform = s.get("platform") or ""
        src_lines.append(f"[{i}] {title} (source: {platform})")
    sources_block = "\n".join(src_lines)
    return (
        f"USER QUERY: {query}\n\n"
        f"RETRIEVED SOURCES:\n{sources_block}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        f"Score the answer per the rubric. Output JSON only."
    )


def judge_one(client: Mistral, query: str, sources: list[dict], answer: str, judge_model: str) -> dict:
    """Call the judge model on a single (query, sources, answer) tuple."""
    response = with_retry(lambda: client.chat.complete(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user_message(query, sources, answer)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    ))
    text = response.choices[0].message.content
    try:
        scores = json.loads(text)
    except json.JSONDecodeError as e:
        # Fallback: judge produced malformed JSON. Mark all dims as None.
        return {
            "error": f"JSON parse failed: {e}",
            "raw": text,
            **{d: None for d in RUBRIC_DIMS},
            "rationale": "judge output malformed",
        }
    # Coerce to ints and clamp 1-5
    for d in RUBRIC_DIMS:
        v = scores.get(d)
        if isinstance(v, (int, float)):
            scores[d] = max(1, min(5, int(round(v))))
        else:
            scores[d] = None
    scores.setdefault("rationale", "")
    return scores


def run_judge(
    runs: list[dict],
    judge_model: str = JUDGE_MODEL,
    workers: int = 4,
) -> dict:
    """Judge every run that is not a negative test. Returns per-run scores +
    aggregate per generator model.

    Skips negative tests (no relevant context → refusal eval handles them).
    """
    client = Mistral(api_key=settings.mistral_api_key)
    pos_runs = [r for r in runs if not r.get("is_negative", False)]

    per_run_scores: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]judging"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("judging", total=len(pos_runs))

        def _judge(run: dict) -> dict:
            scores = judge_one(
                client, run["query"], run["sources"], run["answer"], judge_model
            )
            return {
                "query_id": run["query_id"],
                "query": run["query"],
                "model": run["model"],
                "category": run["category"],
                **scores,
            }

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_judge, r) for r in pos_runs]
            for f in as_completed(futures):
                per_run_scores.append(f.result())
                progress.update(task, advance=1)

    per_run_scores.sort(key=lambda r: (r["query_id"], r["model"]))

    # Aggregate per generator model
    gen_models = sorted({s["model"] for s in per_run_scores})
    per_model: dict[str, dict] = {}
    for model in gen_models:
        m_scores = [s for s in per_run_scores if s["model"] == model and s.get("overall") is not None]
        if not m_scores:
            per_model[model] = {"n_scored": 0}
            continue
        per_model[model] = {"n_scored": len(m_scores)}
        for dim in RUBRIC_DIMS:
            vals = [s[dim] for s in m_scores if isinstance(s.get(dim), int)]
            per_model[model][dim] = {
                "mean": mean(vals) if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "n": len(vals),
            }

    return {
        "judge_model": judge_model,
        "per_model": per_model,
        "per_run": per_run_scores,
    }


def print_judge_table(judge: dict) -> None:
    t = Table(title=f"LLM-judge scores (judge: {judge['judge_model']})", show_lines=False)
    t.add_column("generator", style="bold green")
    t.add_column("n", justify="right")
    for d in RUBRIC_DIMS:
        t.add_column(d[:10], justify="right", style="cyan")
    for model, m in judge["per_model"].items():
        if m["n_scored"] == 0:
            t.add_row(model, "0", *(["—"] * len(RUBRIC_DIMS)))
            continue
        row = [model, str(m["n_scored"])]
        for d in RUBRIC_DIMS:
            v = m[d].get("mean")
            row.append(f"{v:.2f}" if v is not None else "—")
        t.add_row(*row)
    console.print(t)


def main() -> None:
    if not CONSOLIDATED_PATH.exists():
        console.print(
            f"[red]No consolidated report at {CONSOLIDATED_PATH.name} — run `make eval` first.[/red]"
        )
        sys.exit(1)
    report = json.loads(CONSOLIDATED_PATH.read_text())
    runs = report["raw_runs"]
    judge = run_judge(runs)
    print_judge_table(judge)
    OUTPUT_PATH.write_text(json.dumps(judge, indent=2))
    console.print(f"\n[bold green]Saved → {OUTPUT_PATH.relative_to(REPO_ROOT)}[/bold green]")


if __name__ == "__main__":
    main()
