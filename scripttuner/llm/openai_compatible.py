"""OpenAI-compatible LLM client.

Provider-agnostic: works with OpenAI, OpenRouter, Together, Groq, local vLLM, etc.
Authentication and endpoint are read from environment variables that the OpenAI SDK
automatically recognizes:

- ``OPENAI_API_KEY``: API key for any provider
- ``OPENAI_BASE_URL``: Provider endpoint (omit for OpenAI default)
"""

from __future__ import annotations

import sys
from typing import Any

from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam


class OpenAICompatibleClient:
    """`LLMClient` Protocol impl using the OpenAI Python SDK.

    The SDK reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the environment.
    Transient errors (429, 5xx, network) are retried by the SDK up to ``max_retries``
    times with exponential backoff. Permanent failures raise.

    On 429 the response headers are dumped to stderr once per occurrence so that the
    `X-RateLimit-Reset` unit can be confirmed during a real measurement run
    (OpenRouter free-tier models do not send the standard `Retry-After` header that
    the SDK auto-honors). The dump is diagnostic; precise header-driven retry is
    implemented separately once the unit is confirmed.
    """

    def __init__(
        self,
        *,
        model: str,
        max_retries: int = 3,
        provider: dict[str, Any] | None = None,
    ) -> None:
        self._client = OpenAI(max_retries=max_retries)
        self._model = model
        # OpenRouter provider-routing prefs (sort/max_price/...); passed as
        # extra_body.provider. None for non-OpenRouter endpoints (ignored anyway).
        self._provider = provider

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        extra_body = {"provider": self._provider} if self._provider else None
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                extra_body=extra_body,
            )
        except RateLimitError as e:
            response = getattr(e, "response", None)
            if response is not None:
                print(
                    f"[llm] 429 RateLimit; response headers: {dict(response.headers)}",
                    file=sys.stderr,
                )
            raise
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            raise ValueError(
                f"LLM returned no content (finish_reason={choice.finish_reason})"
            )
        metadata: dict[str, Any] = {
            "response_model": resp.model,
            "finish_reason": choice.finish_reason,
        }
        usage = resp.usage
        if usage is not None:
            metadata["prompt_tokens"] = usage.prompt_tokens
            metadata["completion_tokens"] = usage.completion_tokens
            metadata["total_tokens"] = usage.total_tokens
        return content, metadata
