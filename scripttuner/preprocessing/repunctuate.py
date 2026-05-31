"""모듈 ④b — Tier B 재구두점 (쉼표·물음표 복원). cf. ADR-0012.

학습 target(`Pair.spoken_text`)은 SBCSAE 전사 특성상 거의 모든 종결이 온점이고
쉼표·물음표가 빈약하다. 제약 LLM으로 **구두점·대소문자만** 삽입하고 단어는
보존한다(구현안 1: 자유생성 + 사후 토큰검증).

안전장치:
- **불변식**: LLM 출력의 단어열이 입력과 정확히 같아야 한다(`[A-Za-z0-9'-]+`,
  대소문자 무시). 깨지면 그 spoken은 **변경하지 않고 그대로 반환**(Tier A 폴백).
- **pause 재삽입**: `<pause:*>`는 LLM에 보내기 전 strip하고, "N번째 단어 앞"
  위치를 기록해 출력의 같은 단어 경계에 결정론적으로 되돌린다(ADR-0003 신호 보존).
- LLM 입력이 아니라 **target에만** 적용한다(④ formal 생성과 무관).

LLM 라우팅(OpenRouter 1차 / Groq 2차)·배치는 호출부(스크립트/CLI) 책임이다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from scripttuner.persistence.cache import DiskCache, make_cache_key
from scripttuner.preprocessing.text_normalize import normalize_punctuation

if TYPE_CHECKING:
    # 런타임 import는 순환을 만든다 (pairs가 repunctuate를 import하므로).
    # LLMClient는 타입 힌트로만 쓰이고 `from __future__ import annotations`로
    # 평가가 지연되므로 TYPE_CHECKING 가드로 충분하다.
    from scripttuner.preprocessing.pairs import LLMClient

DEFAULT_PROMPT_VERSION = "repunct-v1"

# 구두점·대소문자만 삽입. PoC 프롬프트(no_excl) + 본 구현 보강 2줄.
SYSTEM_PROMPT = (
    "You are a punctuation restorer. The user gives a spoken-English transcript "
    "that has lost most of its punctuation. Insert only sentence punctuation: "
    "periods, commas, and question marks. Capitalize the first letter of each "
    "sentence.\n"
    "STRICT RULES:\n"
    "- Do NOT add, delete, change, reorder, or merge any words.\n"
    "- Keep every filler (um, uh, well, you know, I mean, like), contraction "
    "(gonna, wanna, gotta, 'em), repetition, and false start (e.g. 'y-', 'f-') "
    "exactly as written, including any trailing hyphen.\n"
    "- Only insert or adjust punctuation and capitalization.\n"
    "- Output a single line. Do NOT insert line breaks.\n"
    "- Leave any symbol you do not understand exactly as it appears.\n"
    "Output only the punctuated transcript, with no preface or explanation."
)

_PAUSE_RE = re.compile(r"<pause:\w+>")
# 단어 토큰: false-start 하이픈("y-")·축약("don't","'em")·복합어("two-year")를 한 단위로.
_WORD_RE = re.compile(r"[A-Za-z0-9'-]+")
_MULTISPACE_RE = re.compile(r"\s+")
# 재삽입 위치 탐색용: 단어 또는 pause 토큰
_WORD_OR_PAUSE_RE = re.compile(r"<pause:\w+>|[A-Za-z0-9'-]+")


def _words(text: str) -> list[str]:
    """pause 토큰을 제외한 단어 토큰열(소문자). 불변식 비교의 단일 정의."""
    return _WORD_RE.findall(_PAUSE_RE.sub(" ", text).lower())


def strip_pauses(text: str) -> str:
    """pause 토큰 제거 + 공백 정리. LLM 입력 평문 생성."""
    return _MULTISPACE_RE.sub(" ", _PAUSE_RE.sub(" ", text)).strip()


def extract_pause_positions(text: str) -> tuple[list[tuple[int, str]], int]:
    """각 pause 토큰이 "몇 개의 단어 뒤"에 오는지 기록한다.

    Returns:
        (positions, n_words). positions는 (앞선 단어 수, pause 토큰) 리스트이며
        등장 순서를 보존한다. 끝에 오는 pause는 (n_words, token)으로 기록된다.
    """
    positions: list[tuple[int, str]] = []
    n_words = 0
    for m in _WORD_OR_PAUSE_RE.finditer(text):
        tok = m.group(0)
        if tok.startswith("<pause:"):
            positions.append((n_words, tok))
        else:
            n_words += 1
    return positions, n_words


def reinsert_pauses(output: str, positions: list[tuple[int, str]]) -> str:
    """재구두점된 output의 단어 경계에 pause 토큰을 되돌린다.

    output의 단어열이 원본과 1:1 대응(불변식 통과)임을 전제한다. 인덱스 i의 pause는
    output의 i번째 단어 **바로 앞**(선행 공백/구두점 뒤)에 삽입된다.
    """
    if not positions:
        return output
    by_index: dict[int, list[str]] = {}
    for idx, tok in positions:
        by_index.setdefault(idx, []).append(tok)

    word_spans = list(_WORD_RE.finditer(output))
    parts: list[str] = []
    cursor = 0
    for wi, m in enumerate(word_spans):
        if wi in by_index:
            parts.append(output[cursor : m.start()])
            parts.append(" ".join(by_index[wi]) + " ")
            cursor = m.start()
    parts.append(output[cursor:])
    # 마지막 단어 뒤(끝)에 오는 pause
    end_idx = len(word_spans)
    if end_idx in by_index:
        tail = " " + " ".join(by_index[end_idx])
        parts.append(tail)
    return "".join(parts)


def repunctuate_spoken(
    spoken: str,
    *,
    client: LLMClient,
    cache: DiskCache | None = None,
    model: str,
    model_alias: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> tuple[str, dict[str, Any]]:
    """단일 spoken_text에 Tier B 재구두점을 적용한다.

    Returns:
        (new_spoken, meta). 불변식 통과 시 new_spoken은 재구두점+pause 재삽입 결과,
        실패(단어 변경) 시 입력 ``spoken``을 그대로 반환한다(Tier A 폴백).
        meta: {applied: bool, reason: str, from_cache: bool, ...call_meta}.
        LLM 호출 실패는 예외로 전파한다(호출부가 OpenRouter→Groq 라우팅 결정).
    """
    positions, _ = extract_pause_positions(spoken)
    plain = strip_pauses(spoken)
    if not plain:
        return spoken, {"applied": False, "reason": "empty"}

    cache_key_id = model_alias if model_alias is not None else model
    key = make_cache_key(prompt_version, cache_key_id, plain)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        output = cached["text"]
        call_meta: dict[str, Any] = cached.get("call_meta", {})
        from_cache = True
    else:
        output, call_meta = client.complete(SYSTEM_PROMPT, plain)
        output = output.strip()
        if cache is not None:
            cache.set(key, {"text": output, "call_meta": call_meta})
        from_cache = False

    meta: dict[str, Any] = {**call_meta, "from_cache": from_cache}
    if _words(output) != _words(plain):
        meta.update(applied=False, reason="word_mismatch")
        return spoken, meta

    restored = reinsert_pauses(output, positions)
    restored = normalize_punctuation(restored)
    meta.update(applied=True, reason="ok")
    return restored, meta
