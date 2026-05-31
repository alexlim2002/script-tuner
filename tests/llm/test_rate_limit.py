from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from openai import RateLimitError

from scripttuner.llm import rate_limit as rate_limit_module
from scripttuner.llm.rate_limit import (
    _FALLBACK_SLEEP_SECONDS,
    _MAX_SLEEP_SECONDS,
    RateLimitRetryClient,
)


def _make_429(retry_after: str | None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return RateLimitError(message="rate limited", response=response, body=None)


class _FakeInner:
    """Stand-in for OpenAICompatibleClient. Plays a script of responses on .complete()."""

    def __init__(self, *, model: str, max_retries: int = 0, provider: Any = None) -> None:
        self.model = model
        self.max_retries = max_retries
        self.provider = provider
        self.calls: list[tuple[str, str]] = []
        self._script: list[Any] = []

    def queue(self, *items: Any) -> _FakeInner:
        self._script.extend(items)
        return self

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((system, user))
        if not self._script:
            raise AssertionError(f"unexpected extra call #{len(self.calls)}")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return cast(tuple[str, dict[str, Any]], item)


@pytest.fixture
def fake_inner(monkeypatch: pytest.MonkeyPatch) -> _FakeInner:
    """Replace OpenAICompatibleClient in openrouter module with a fake."""
    holder: dict[str, _FakeInner] = {}

    def factory(*, model: str, max_retries: int = 0, provider: Any = None) -> _FakeInner:
        inner = _FakeInner(model=model, max_retries=max_retries, provider=provider)
        holder["inner"] = inner
        return inner

    monkeypatch.setattr(rate_limit_module, "OpenAICompatibleClient", factory)
    # Trigger creation by instantiating once; tests grab `holder["inner"]`.
    return holder  # type: ignore[return-value]


def _new_client(holder: dict[str, _FakeInner], sleep_log: list[float]) -> tuple[
    RateLimitRetryClient, _FakeInner
]:
    client = RateLimitRetryClient(model="test/model", sleep=lambda s: sleep_log.append(s))
    inner = holder["inner"]
    return client, inner


# ----- happy path -----


def test_success_passes_through_with_no_sleep(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(("the reply", {"prompt_tokens": 1}))

    text, meta = client.complete("sys", "user")

    assert text == "the reply"
    assert meta == {"prompt_tokens": 1}
    assert len(inner.calls) == 1
    assert sleep_log == []


# ----- header-driven retry -----


def test_429_with_short_retry_after_sleeps_then_retries(
    fake_inner: dict[str, Any],
) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429("30"), ("recovered", {}))

    text, _ = client.complete("sys", "user")

    assert text == "recovered"
    assert len(inner.calls) == 2
    assert sleep_log == [30.0]


def test_429_with_retry_after_at_cap_still_retries(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429(str(_MAX_SLEEP_SECONDS)), ("ok", {}))

    client.complete("sys", "user")

    assert sleep_log == [float(_MAX_SLEEP_SECONDS)]


def test_429_with_long_retry_after_raises_without_sleep(
    fake_inner: dict[str, Any],
) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429(str(_MAX_SLEEP_SECONDS + 1)))

    with pytest.raises(RateLimitError):
        client.complete("sys", "user")

    assert len(inner.calls) == 1  # no retry
    assert sleep_log == []


def test_429_without_header_uses_fallback_sleep(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429(None), ("ok", {}))

    client.complete("sys", "user")

    assert sleep_log == [float(_FALLBACK_SLEEP_SECONDS)]
    assert len(inner.calls) == 2


def test_429_with_zero_retry_after_uses_fallback(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429("0"), ("ok", {}))

    client.complete("sys", "user")

    assert sleep_log == [float(_FALLBACK_SLEEP_SECONDS)]


def test_429_with_unparseable_header_uses_fallback(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429("not-a-number"), ("ok", {}))

    client.complete("sys", "user")

    assert sleep_log == [float(_FALLBACK_SLEEP_SECONDS)]


# ----- retry exhaustion -----


def test_retries_multiple_429s_then_succeeds(fake_inner: dict[str, Any]) -> None:
    """Groq-style TPS cap: several short 429s in a row, then recovery."""
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    inner.queue(_make_429("2"), _make_429("2"), _make_429("2"), ("recovered", {}))

    text, _ = client.complete("sys", "user")

    assert text == "recovered"
    assert len(inner.calls) == 4  # 3 failures + 1 success
    assert sleep_log == [2.0, 2.0, 2.0]


def test_exhausts_max_retries_then_raises(fake_inner: dict[str, Any]) -> None:
    sleep_log: list[float] = []
    client = RateLimitRetryClient(
        model="test/model", sleep=lambda s: sleep_log.append(s), max_retries=3
    )
    inner = fake_inner["inner"]
    inner.queue(_make_429("1"), _make_429("1"), _make_429("1"), _make_429("1"))

    with pytest.raises(RateLimitError):
        client.complete("sys", "user")

    # max_retries=3 → 3 sleeps + a final attempt = 4 calls total
    assert len(inner.calls) == 4
    assert sleep_log == [1.0, 1.0, 1.0]


def test_long_retry_after_raises_mid_loop_without_exhausting(
    fake_inner: dict[str, Any],
) -> None:
    sleep_log: list[float] = []
    client, inner = _new_client(fake_inner, sleep_log)
    # first 429 short (retry), second carries a daily-cap-signature long header
    inner.queue(_make_429("5"), _make_429(str(_MAX_SLEEP_SECONDS + 1)))

    with pytest.raises(RateLimitError):
        client.complete("sys", "user")

    assert len(inner.calls) == 2
    assert sleep_log == [5.0]  # slept once, then long header → immediate raise
