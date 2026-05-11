"""Refusal evaluation on negative-test queries.

For queries whose `relevant_doc_ids` is empty (Q24, Q25, Q26 in our eval set),
the system should refuse to answer rather than hallucinate. We evaluate this
two ways:

  1. Heuristic — pattern-match common refusal phrases in the answer.
  2. LLM-judge — ask a strong judge model whether the answer refused properly.

Both are reported per generator model. They should agree most of the time.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mistralai import Mistral
from rich.console import Console
from rich.table import Table

from src.config import settings
from src.eval._retry import with_retry

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "eval_results" / "refusal.json"
CONSOLIDATED_PATH = REPO_ROOT / "eval_results" / "consolidated_report.json"

JUDGE_MODEL = "mistral-large-latest"

REFUSAL_PATTERNS = [
    "couldn't find", "could not find",
    "no information", "no relevant",
    "not in the context", "not in the provided", "not in our context",
    "context does not", "context doesn't", "context did not",
    "i don't have", "i do not have",
    "cannot answer", "can't answer",
    "unable to", "unable to find",
    "not available", "isn't available",
    "no documents", "no records",
    "doesn't mention", "does not mention",
    "doesn't contain", "does not contain",
    "doesn't include", "does not include",
    "doesn't cover", "does not cover",
    "no mention", "no reference",
    "not appear to", "doesn't appear to",
]


def heuristic_refused(answer: str) -> bool:
    """Returns True if any refusal pattern appears in the answer."""
    a = answer.lower()
    return any(p in a for p in REFUSAL_PATTERNS)


REFUSAL_JUDGE_SYSTEM = """You are evaluating whether a RAG system correctly REFUSED to answer a query because no relevant context was provided. Output ONLY a valid JSON object, no prose, no markdown.

The query has NO relevant documents in the corpus. The correct behavior is to refuse — to say something like "I couldn't find relevant information" — and NOT to invent facts.

Score:
- refused: 1 if the system clearly refused or stated the answer was not in the context; 0 if it confidently answered the question.
- hallucinated: 1 if the system stated facts not present in the context (i.e., made things up); 0 if it stuck to "no relevant info".

Both can be 1 (e.g., system refused on the surface but still hallucinated a detail).

Schema:
{
  "refused": 0 or 1,
  "hallucinated": 0 or 1,
  "rationale": "one-sentence justification"
}"""


def judge_refusal(client: Mistral, query: str, answer: str, judge_model: str) -> dict:
    user_msg = f"USER QUERY: {query}\n\nGENERATED ANSWER:\n{answer}\n\nDid this system correctly refuse? Output JSON only."
    response = with_retry(lambda: client.chat.complete(
        model=judge_model,
        messages=[
            {"role": "system", "content": REFUSAL_JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    ))
    text = response.choices[0].message.content
    try:
        scores = json.loads(text)
    except json.JSONDecodeError as e:
        return {"error": str(e), "refused": None, "hallucinated": None, "rationale": "judge malformed"}
    # Coerce
    for k in ("refused", "hallucinated"):
        v = scores.get(k)
        scores[k] = 1 if v in (1, "1", True, "yes", "true") else (0 if v in (0, "0", False, "no", "false") else None)
    scores.setdefault("rationale", "")
    return scores


def compute_refusal(runs: list[dict], judge_model: str = JUDGE_MODEL, workers: int = 4) -> dict:
    """Evaluate refusal on negative-test runs (is_negative=True).

    Returns per-run + per-model aggregates.
    """
    neg_runs = [r for r in runs if r.get("is_negative", False)]
    if not neg_runs:
        return {"note": "no negative-test runs found", "n_negative": 0}

    client = Mistral(api_key=settings.mistral_api_key)
    per_run: list[dict] = []

    def _evaluate(run: dict) -> dict:
        heuristic = heuristic_refused(run["answer"])
        judge_scores = judge_refusal(client, run["query"], run["answer"], judge_model)
        return {
            "query_id": run["query_id"],
            "query": run["query"],
            "model": run["model"],
            "answer_first_100_chars": run["answer"][:100],
            "heuristic_refused": int(heuristic),
            "judge_refused": judge_scores.get("refused"),
            "judge_hallucinated": judge_scores.get("hallucinated"),
            "judge_rationale": judge_scores.get("rationale", ""),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_evaluate, r) for r in neg_runs]
        for f in as_completed(futures):
            per_run.append(f.result())

    per_run.sort(key=lambda r: (r["query_id"], r["model"]))

    # Per-model aggregates
    gen_models = sorted({r["model"] for r in per_run})
    per_model = {}
    for model in gen_models:
        m_runs = [r for r in per_run if r["model"] == model]
        n = len(m_runs)
        per_model[model] = {
            "n":                  n,
            "heuristic_refused":  sum(r["heuristic_refused"] for r in m_runs),
            "judge_refused":      sum(r["judge_refused"] for r in m_runs if r["judge_refused"] is not None),
            "judge_hallucinated": sum(r["judge_hallucinated"] for r in m_runs if r["judge_hallucinated"] is not None),
            "heuristic_refusal_rate": sum(r["heuristic_refused"] for r in m_runs) / n if n else 0,
            "judge_refusal_rate":     sum(r["judge_refused"] for r in m_runs if r["judge_refused"] is not None) / n if n else 0,
            "hallucination_rate":     sum(r["judge_hallucinated"] for r in m_runs if r["judge_hallucinated"] is not None) / n if n else 0,
        }

    return {
        "judge_model": judge_model,
        "n_negative_queries": len(set(r["query_id"] for r in neg_runs)),
        "per_model":  per_model,
        "per_run":    per_run,
    }


def print_refusal_table(ref: dict) -> None:
    if "per_model" not in ref:
        console.print("[yellow]No refusal data to display.[/yellow]")
        return
    t = Table(title="Refusal eval (negative-test queries)", show_lines=False)
    t.add_column("model", style="bold green")
    t.add_column("n", justify="right")
    t.add_column("heuristic refused", justify="right", style="cyan")
    t.add_column("judge refused", justify="right", style="cyan")
    t.add_column("judge hallucinated", justify="right", style="red")
    t.add_column("refusal rate (judge)", justify="right", style="bold")
    for model, m in ref["per_model"].items():
        t.add_row(
            model, str(m["n"]),
            f"{m['heuristic_refused']}/{m['n']}",
            f"{m['judge_refused']}/{m['n']}",
            f"{m['judge_hallucinated']}/{m['n']}",
            f"{m['judge_refusal_rate']:.2f}",
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
    ref = compute_refusal(runs)
    print_refusal_table(ref)
    OUTPUT_PATH.write_text(json.dumps(ref, indent=2))
    console.print(f"\n[bold green]Saved → {OUTPUT_PATH.relative_to(REPO_ROOT)}[/bold green]")


if __name__ == "__main__":
    main()
