"""Rate-limit-aware LLM client — header-driven RPM-cap recovery.

Free-tier OpenAI-compatible endpoints (OpenRouter, Groq, …) often hit
per-minute caps. The OpenAI SDK's default exponential backoff
(`min(0.5 * 2^n, 8s)`) caps at 8s, which never spans the RPM window — every
retry within the same minute also fails, wasting quota (failed requests still
count toward the limit).

This client wraps `OpenAICompatibleClient` and adds RPM-cap-aware retry,
provider-agnostic (works for any OpenAI-compatible endpoint):

- SDK internal retries are disabled (`max_retries=0`) so the wrapper owns all
  retry behavior — quota is spent exactly once per attempt.
- On `429`, the wrapper inspects `Retry-After` (the only header consistently
  documented across providers):
    * `Retry-After` ≤ 60s → sleep that long, retry (up to ``max_retries`` times).
    * No header → sleep 60s fallback (assumed RPM cap), retry.
    * `Retry-After` > 60s → treat as daily-cap / long throttle, raise without
      sleeping so the caller (pairs.py) skips this monologue.
- Retries up to ``max_retries`` times (default 5). Groq-style TPS caps return a
  short Retry-After but can 429 again on the very next call, so a single retry
  is not enough; we keep retrying within the per-attempt 60s ceiling.
- The response headers from every 429 are dumped to stderr for diagnostics.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from openai import RateLimitError

from scripttuner.llm.openai_compatible import OpenAICompatibleClient

# Hard ceiling on a single sleep. Anything larger almost certainly means a daily
# / hourly cap reset that retrying soon cannot recover from.
_MAX_SLEEP_SECONDS = 60

# Used when the 429 response carries no Retry-After header. Free-tier models are
# observed not to send the standard header; a 60s window is the common RPM
# bucket size and is the safest conservative wait.
_FALLBACK_SLEEP_SECONDS = 60

# Default number of retries on recoverable 429s. Groq TPS caps clear quickly but
# can re-trigger on consecutive calls, so one retry is often not enough.
_DEFAULT_MAX_RETRIES = 5


def _parse_retry_after(headers: Any) -> float | None:
    """Parse the standard ``Retry-After`` header as a delta in seconds.

    Returns ``None`` if the header is absent or unparseable. We intentionally do
    not consult ``X-RateLimit-Reset`` because providers do not document its unit
    (sec vs ms vs Unix timestamp) consistently and guessing is unsafe.
    """
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class RateLimitRetryClient:
    """`LLMClient` Protocol impl with RPM-cap-aware retry, provider-agnostic.

    Wraps `OpenAICompatibleClient` via composition; the inner client is created
    with ``max_retries=0`` so all retry policy lives here.
    """

    def __init__(
        self,
        *,
        model: str,
        sleep: Any = time.sleep,
        provider: dict[str, Any] | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        # Inner client owns the actual HTTP call; we disable its retry loop so
        # quota is consumed exactly once per attempt we explicitly make.
        self._inner = OpenAICompatibleClient(
            model=model, max_retries=0, provider=provider
        )
        self._sleep = sleep
        self._max_retries = max_retries

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        # Up to _max_retries recoverable 429s: sleep per Retry-After, then retry.
        # A long Retry-After (>60s, daily-cap signature) raises immediately.
        for attempt in range(self._max_retries):
            try:
                return self._inner.complete(system, user)
            except RateLimitError as e:
                sleep_for = self._sleep_seconds_for(e)
                if sleep_for is None:
                    raise
                print(
                    f"[rate_limit] 429 — sleeping {sleep_for}s then retrying "
                    f"({attempt + 1}/{self._max_retries})",
                    file=sys.stderr,
                )
                self._sleep(sleep_for)

        # Final attempt after the last sleep; a 429 here propagates to the caller.
        return self._inner.complete(system, user)

    @staticmethod
    def _sleep_seconds_for(error: RateLimitError) -> float | None:
        """Decide how long to wait before the single retry.

        Returns ``None`` when the caller should give up immediately (daily-cap
        signature: Retry-After > 60s).
        """
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        retry_after = _parse_retry_after(headers)
        if retry_after is None:
            return float(_FALLBACK_SLEEP_SECONDS)
        if retry_after <= 0:
            return float(_FALLBACK_SLEEP_SECONDS)
        if retry_after > _MAX_SLEEP_SECONDS:
            return None
        return retry_after
