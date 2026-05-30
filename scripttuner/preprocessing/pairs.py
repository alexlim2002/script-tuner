"""모듈 ④ — LLM 역변환 (구어체 → 문어체 paraphrase).

`Monologue`를 입력받아 LLM으로 문어체 paraphrase를 생성하고 `Pair`로 반환한다.
LLM 호출은 `LLMClient` Protocol에 위임 → provider 무관 (cf. `scripttuner/llm/`).

PoC 단계 정책:
- zero-shot (시스템 프롬프트만, few-shot 미사용)
- 순차 처리 + tqdm 진행 표시
- 동일 입력은 디스크 캐시 적중 → 재호출 없음
- 일시 오류는 LLMClient 내부 재시도(openai SDK max_retries)에 위임
- 영구 실패는 해당 monologue를 skip하고 stderr에 로그 + 계속 진행
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from scripttuner.persistence.cache import DiskCache, make_cache_key
from scripttuner.preprocessing.ir import Monologue, Pair
from scripttuner.preprocessing.text_normalize import normalize_punctuation

SYSTEM_PROMPT = (
    "Convert the spoken English transcript into clean written English. "
    "Do not rewrite the sentences from scratch — keep the speaker's wording, "
    "structure, and content as much as possible. "
    "Only remove disfluencies: fillers, repetitions, self-corrections, and "
    "false starts. Fix minor grammar where needed for the result to read as "
    "a written script. "
    "Expand colloquial contractions to their standard forms "
    "(e.g. gonna → going to, wanna → want to, gotta → got to). "
    "Output only the converted text, with no preface or explanation."
)
DEFAULT_PROMPT_VERSION = "v2-zero-shot"
DEFAULT_STYLE = "casual"

_PAUSE_TOKEN_RE = re.compile(r"<pause:\w+>")
_MULTISPACE_RE = re.compile(r"\s+")

# LLM이 종종 출력하는 typographic 문자를 ASCII로 정규화한다.
# spoken_text는 ASCII만 쓰므로 formal_text도 ASCII로 맞춰야 typography 자체가
# 학습 신호에 섞이지 않는다. 후처리 단계라 캐시된 raw 응답을 무효화하지 않는다.
_TYPOGRAPHY_MAP = str.maketrans(
    {
        "‘": "'",  # U+2018 left single quote
        "’": "'",  # U+2019 right single quote
        "“": '"',  # U+201C left double quote
        "”": '"',  # U+201D right double quote
        "‐": "-",  # U+2010 hyphen
        "‑": "-",  # U+2011 non-breaking hyphen
        "‒": "-",  # U+2012 figure dash
        "–": "-",  # U+2013 en-dash
        "—": "-",  # U+2014 em-dash
        "―": "-",  # U+2015 horizontal bar
        "…": "...",  # U+2026 horizontal ellipsis
        " ": " ",  # no-break space
        " ": " ",  # thin space
        " ": " ",  # hair space
        " ": " ",  # narrow no-break space
    }
)


class LLMClient(Protocol):
    """LLM completion 호출 추상화. provider-agnostic.

    구현체는 `scripttuner/llm/` 아래에 둔다.
    """

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        """Send a chat completion request.

        Returns (response_text, metadata) where metadata may include token usage,
        model identifier echoed back, etc.
        Raises on permanent failure (after internal retries).
        """
        ...


def _strip_special_tokens(text: str) -> str:
    """Remove `<pause:*>` tokens and collapse whitespace.

    LLM 입력 정제용. 원본 `monologue.text`는 변경하지 않는다.
    """
    stripped = _PAUSE_TOKEN_RE.sub("", text)
    return _MULTISPACE_RE.sub(" ", stripped).strip()


def _normalize_typography(text: str) -> str:
    """Map curly quotes / em-dashes / ellipsis chars to their ASCII equivalents.

    Applied to LLM output before constructing the `Pair`. spoken_text uses ASCII,
    so this prevents typography itself from becoming a learnable signal.
    """
    return text.translate(_TYPOGRAPHY_MAP)


def convert_to_formal(
    monologues: Iterable[Monologue],
    *,
    client: LLMClient,
    model: str,
    model_alias: str | None = None,
    cache_dir: Path | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    style: str = DEFAULT_STYLE,
    progress: bool = True,
) -> list[Pair]:
    """Convert spoken monologues to (spoken, formal) pairs via LLM.

    Args:
        monologues: Input monologues. `monologue.text` may contain `<pause:*>` tokens;
            they are stripped before LLM call but preserved in `Pair.spoken_text`.
        client: LLM client implementing the `LLMClient` Protocol.
        model: Raw model slug sent to the provider, recorded as ``metadata.model``.
        model_alias: Identity used for the cache key in place of ``model``. Set to
            a stable name shared across routing variants of the same weights
            (e.g. ``openai/gpt-oss-120b:free`` vs ``openai/gpt-oss-120b:nitro``)
            so the cache hits across routes. When ``None`` the raw ``model`` is
            used as the cache key (legacy behavior).
        cache_dir: If provided, sha256-keyed disk cache directory. Cache hits skip
            the LLM call. Failures are not cached.
        prompt_version: Identifier embedded in pair_id and cache key. Bump when
            modifying SYSTEM_PROMPT.
        style: Style label for the produced pair (cf. ADR-0005).
        progress: Show tqdm progress bar.

    Returns:
        List of `Pair` for successfully converted monologues. Failed monologues
        (after client-internal retries) are skipped with a stderr log line.
    """
    cache_key_id = model_alias if model_alias is not None else model
    mono_list = list(monologues)
    cache = DiskCache(cache_dir) if cache_dir is not None else None
    iterator: Iterable[Monologue] = (
        tqdm(mono_list, desc="LLM pairs") if progress else mono_list
    )

    pairs: list[Pair] = []
    for mono in iterator:
        user_text = _strip_special_tokens(mono.text)
        cache_key = make_cache_key(prompt_version, cache_key_id, user_text)

        cached = cache.get(cache_key) if cache is not None else None
        if cached is not None:
            formal_text = cached["formal_text"]
            call_meta: dict[str, Any] = cached.get("call_meta", {})
            from_cache = True
        else:
            try:
                formal_text, call_meta = client.complete(SYSTEM_PROMPT, user_text)
            except Exception as e:  # noqa: BLE001 — log + skip; openai SDK handles transient retries
                print(
                    f"[pairs] skip {mono.monologue_id}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            if cache is not None:
                cache.set(
                    cache_key,
                    {"formal_text": formal_text, "call_meta": call_meta},
                )
            from_cache = False

        formal_text = _normalize_typography(formal_text)

        pair_id = f"{mono.monologue_id}#{style}#{prompt_version}"
        metadata = {
            **call_meta,
            "model": model,
            "model_alias": cache_key_id,
            "prompt_version": prompt_version,
            "from_cache": from_cache,
        }
        pairs.append(
            Pair(
                pair_id=pair_id,
                source=mono.source,
                style=style,
                speaker=mono.speaker,
                spoken_text=normalize_punctuation(mono.text),
                formal_text=formal_text,
                monologue_id=mono.monologue_id,
                metadata=metadata,
            )
        )
    return pairs
