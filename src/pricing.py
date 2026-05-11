"""Mistral API pricing — keep in one place for cost calc reuse.

Prices are per 1M tokens in USD. Source: console.mistral.ai pricing as of 2026-Q1.
Update this file when Mistral changes pricing or you pin different model versions.
"""

from __future__ import annotations

PRICING = {
    "mistral-small-latest": {"input_per_1m": 0.20, "output_per_1m": 0.60},
    "mistral-large-latest": {"input_per_1m": 2.00, "output_per_1m": 6.00},
    "mistral-embed":         {"input_per_1m": 0.10, "output_per_1m": 0.00},
}


def calc_cost(model: str, tokens_in: int, tokens_out: int = 0) -> float:
    """Return USD cost for a single API call given token counts.

    Raises KeyError if model not in PRICING — surface the error early so we don't
    silently emit zero-cost lies in the eval harness.
    """
    p = PRICING[model]
    return (
        tokens_in / 1_000_000 * p["input_per_1m"]
        + tokens_out / 1_000_000 * p["output_per_1m"]
    )


def format_cost(usd: float) -> str:
    """Pretty-print sub-cent costs without scientific notation."""
    if usd < 0.001:
        return f"${usd * 1000:.3f}m"  # millicents
    if usd < 0.01:
        return f"${usd:.5f}"
    return f"${usd:.4f}"
