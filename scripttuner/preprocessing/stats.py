"""모듈 ⑤ — Pair 집계 통계.

PoC 단계 산출:

- 카운트: n_pairs, n_unique_speakers
- 길이 분포: spoken/formal 단어 토큰 (pause 제외)
- 사전 기반: filler, pause_short, pause_long per pair
- POS 기반: lexical_density, phrasal_verb_ratio (spacy 영어 모델 필요)
- 비율: reduction_ratio = formal_tokens / spoken_tokens

POS 통계는 `include_pos=False`로 비활성화 가능. spacy 모델이 설치되지 않으면
호출 시점에 친절한 에러 메시지로 안내한다.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from scripttuner.preprocessing.ir import Pair

if TYPE_CHECKING:
    from spacy.language import Language

# 본격 단어형 filler. backchannel(monologue.py)과 의도적으로 구분된다 — backchannel은
# 발화 전체이고 filler는 발화 내부 삽입이다.
SINGLE_WORD_FILLERS: frozenset[str] = frozenset({"um", "uh", "well", "like"})
MULTI_WORD_FILLERS: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("kind", "of"),
    ("sort", "of"),
)

_PAUSE_SHORT_RE = re.compile(r"<pause:short>")
_PAUSE_LONG_RE = re.compile(r"<pause:long>")
_PAUSE_TOKEN_RE = re.compile(r"<pause:\w+>")
_WORD_RE = re.compile(r"\b\w+\b")

# 내용어 (content word) POS — lexical density 분자
CONTENT_POS: frozenset[str] = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})


def _word_tokens(text: str) -> list[str]:
    """단어 토큰 추출. pause 토큰 제거 후 lowercase."""
    stripped = _PAUSE_TOKEN_RE.sub("", text)
    return [m.group(0).lower() for m in _WORD_RE.finditer(stripped)]


def _count_fillers(tokens: list[str]) -> int:
    n = sum(1 for t in tokens if t in SINGLE_WORD_FILLERS)
    for i in range(len(tokens) - 1):
        for pattern in MULTI_WORD_FILLERS:
            if tuple(tokens[i : i + len(pattern)]) == pattern:
                n += 1
    return n


def _distribution(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
    }


def _load_spacy() -> Language:
    try:
        import spacy
    except ImportError as e:
        raise RuntimeError(
            "spacy is not installed. Install with: uv add spacy"
        ) from e
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            "spaCy English model 'en_core_web_sm' is not installed.\n"
            "Install with: uv run python -m spacy download en_core_web_sm"
        ) from e


def _pos_stats(text: str, nlp: Language) -> dict[str, float]:
    """POS 분석으로 lexical_density, phrasal_verb_ratio, pronoun_ratio 계산.

    pronoun_ratio는 구어성 분류기(ADR-0014)용 신규 피처 — 대화의 1·2인칭 빈도를
    잡는다. 분모는 lexical_density와 동일하게 alpha_tokens 길이로 통일한다.
    """
    clean = _PAUSE_TOKEN_RE.sub("", text)
    doc = nlp(clean)
    alpha_tokens = [t for t in doc if t.is_alpha]
    if not alpha_tokens:
        return {"lexical_density": 0.0, "phrasal_verb_ratio": 0.0, "pronoun_ratio": 0.0}
    content = sum(1 for t in alpha_tokens if t.pos_ in CONTENT_POS)
    verbs = [t for t in doc if t.pos_ == "VERB"]
    phrasal = sum(
        1 for t in doc if t.dep_ == "prt" and t.head.pos_ == "VERB"
    )
    pronouns = sum(1 for t in alpha_tokens if t.pos_ == "PRON")
    return {
        "lexical_density": content / len(alpha_tokens),
        "phrasal_verb_ratio": phrasal / len(verbs) if verbs else 0.0,
        "pronoun_ratio": pronouns / len(alpha_tokens),
    }


def compute_stats(
    pairs: Iterable[Pair],
    *,
    include_pos: bool = True,
) -> dict[str, Any]:
    """Pair 리스트에 대한 집계 통계 dict를 반환한다.

    include_pos=False면 POS 기반 통계(`lexical_density`, `phrasal_verb_ratio`)는
    산출하지 않는다 — spacy 모델 없는 환경/테스트용.
    """
    pair_list = list(pairs)
    if not pair_list:
        return {"n_pairs": 0}

    speakers = {p.speaker for p in pair_list}
    spoken_tokens_lists = [_word_tokens(p.spoken_text) for p in pair_list]
    formal_tokens_lists = [_word_tokens(p.formal_text) for p in pair_list]
    spoken_lens = [len(ts) for ts in spoken_tokens_lists]
    formal_lens = [len(ts) for ts in formal_tokens_lists]
    reduction_ratios = [
        (f / s) if s > 0 else 0.0
        for s, f in zip(spoken_lens, formal_lens, strict=True)
    ]
    spoken_fillers = [_count_fillers(ts) for ts in spoken_tokens_lists]
    formal_fillers = [_count_fillers(ts) for ts in formal_tokens_lists]
    spoken_pause_short = [
        len(_PAUSE_SHORT_RE.findall(p.spoken_text)) for p in pair_list
    ]
    spoken_pause_long = [
        len(_PAUSE_LONG_RE.findall(p.spoken_text)) for p in pair_list
    ]

    result: dict[str, Any] = {
        "source": pair_list[0].source,
        "n_pairs": len(pair_list),
        "n_unique_speakers": len(speakers),
        "speakers": sorted(speakers),
        "spoken": {
            "tokens": _distribution(spoken_lens),
            "fillers_per_pair": _distribution(spoken_fillers),
            "pause_short_per_pair": _distribution(spoken_pause_short),
            "pause_long_per_pair": _distribution(spoken_pause_long),
        },
        "formal": {
            "tokens": _distribution(formal_lens),
            "fillers_per_pair": _distribution(formal_fillers),
        },
        "reduction_ratio": _distribution(reduction_ratios),
    }

    if include_pos:
        nlp = _load_spacy()
        spoken_pos = [_pos_stats(p.spoken_text, nlp) for p in pair_list]
        formal_pos = [_pos_stats(p.formal_text, nlp) for p in pair_list]
        result["spoken"]["lexical_density"] = _distribution(
            [d["lexical_density"] for d in spoken_pos]
        )
        result["spoken"]["phrasal_verb_ratio"] = _distribution(
            [d["phrasal_verb_ratio"] for d in spoken_pos]
        )
        result["formal"]["lexical_density"] = _distribution(
            [d["lexical_density"] for d in formal_pos]
        )
        result["formal"]["phrasal_verb_ratio"] = _distribution(
            [d["phrasal_verb_ratio"] for d in formal_pos]
        )

    return result
