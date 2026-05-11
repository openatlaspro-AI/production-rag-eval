"""Retry helper for Mistral 429 rate-limit errors.

Wraps a callable with exponential backoff. Free-tier and burst-capped Mistral
plans return 429 on concurrent or high-rate calls; this lets the eval suite
ride through rate limits without manual intervention.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from mistralai.models.sdkerror import SDKError

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 6,
    initial_wait: float = 4.0,
    max_wait: float = 60.0,
    jitter: float = 0.5,
) -> T:
    """Retry on Mistral 429 with exponential backoff + jitter.

    Total max wait across 6 retries: 4 + 8 + 16 + 32 + 60 + 60 = 180s.

    Re-raises any non-429 SDKError immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except SDKError as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate_limited" in msg.lower() or "rate limit" in msg.lower()
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait = min(initial_wait * (2 ** attempt), max_wait)
            wait += random.uniform(0, jitter * wait)
            time.sleep(wait)
            last_exc = e
    # Should be unreachable
    raise last_exc if last_exc else RuntimeError("with_retry exhausted")
