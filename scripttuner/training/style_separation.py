"""스타일 분리도 메트릭 — casual vs semi_formal 생성 출력이 실제로 분기되는지 측정.

목적(cf. ADR-0013): 제어 토큰이 두 스타일을 구분해 학습했는지, 아니면 collapse
했는지 판정한다. 두 번째 학습 사이클(semi_formal) 착수·평가의 게이트로, 분리가
없으면 학습 비용을 들여도 의미가 없다.

⑤ stats의 per-text 피처 추출기를 재사용하며 의존성 추가 없음(POS는 옵션).

피처(텍스트당, 길이 정규화):
  - filler_rate         = fillers / tokens            # casual↑
  - pause_rate          = (pause_short+long) / tokens  # casual만 발생(AMI엔 pause 없음)
  - tokens_per_sentence = tokens / 문장수              # semi_formal↑ (절을 길게 이어붙임)
  - phrasal_verb_ratio  (POS)                          # casual↑
  - lexical_density     (POS)                          # semi_formal↑
  - tokens              (길이)                          # 방향 미지정

분절(tokens_per_sentence)이 두 스타일의 핵심 분리 축이다 — casual은 짧게 토막,
semi_formal은 절을 길게 이어붙인다. filler/lexical/phrasal만으로는 이 "발화 리듬"
차이를 못 잡으므로 verdict에 포함한다.

분리도: 두 스타일 그룹 간 표준화 평균차(Cohen's d) = (semi_formal_mean -
casual_mean) / pooled_sd. 부호로 관측 방향을 얻어 기대 방향과 대조한다. 모든
|d|이 COLLAPSE_THRESHOLD 미만이면 collapse로 판정.

분리 측정 방식은 효과크기(Cohen's d)를 채택했다. 분류기 기반 separability(AUC)는
보류(제안서 sklearn 진단 모듈과 합류 여지).
"""

from __future__ import annotations

import statistics
from typing import Any

import re

from scripttuner.preprocessing.stats import (
    _PAUSE_LONG_RE,
    _PAUSE_SHORT_RE,
    _count_fillers,
    _word_tokens,
)

# 문장 종결 부호. 분절(문장 수) 추정용.
_SENT_END_RE = re.compile(r"[.!?]+")

# |d| < 이 값이면 사실상 분리 없음(collapse). 0.2 = "small effect" 하한.
COLLAPSE_THRESHOLD = 0.2

# collapse 판정에 쓰는 피처(register/발화스타일 본질). pause_rate는 코퍼스 표기
# 아티팩트라(NXT/AMI는 pause 미표기) 기본 판정에서 제외 — 전체 d는 features에 그대로
# 리포트한다. tokens는 길이 차라 제외. tokens_per_sentence는 표기 아티팩트가 아닌
# 실제 분절 스타일이라 판정에 포함(POS 불필요해 항상 산출).
DEFAULT_VERDICT_FEATURES: frozenset[str] = frozenset(
    {"filler_rate", "tokens_per_sentence", "phrasal_verb_ratio", "lexical_density"}
)

# 각 피처에서 casual이 더 높을 것으로 기대(True) / semi_formal이 더 높을 것(False).
# None은 방향 미지정(분리도에만 사용, 방향 일치 판정 제외).
_CASUAL_HIGHER: dict[str, bool | None] = {
    "filler_rate": True,
    "pause_rate": True,
    "tokens_per_sentence": False,  # semi_formal이 절을 길게 이어붙임
    "phrasal_verb_ratio": True,
    "lexical_density": False,
    "tokens": None,
}


def _surface_features(text: str) -> dict[str, float]:
    """POS 불필요한 표면 피처 (filler_rate, pause_rate, tokens, tokens_per_sentence)."""
    tokens = _word_tokens(text)
    n = len(tokens)
    fillers = _count_fillers(tokens)
    pauses = len(_PAUSE_SHORT_RE.findall(text)) + len(_PAUSE_LONG_RE.findall(text))
    # 문장 수: pause 토큰 제거 후 종결부호 단위 (빈 조각 제외). 최소 1.
    bare = _PAUSE_SHORT_RE.sub(" ", _PAUSE_LONG_RE.sub(" ", text))
    n_sent = sum(1 for s in _SENT_END_RE.split(bare) if s.strip()) or 1
    return {
        "tokens": float(n),
        "filler_rate": (fillers / n) if n else 0.0,
        "pause_rate": (pauses / n) if n else 0.0,
        "tokens_per_sentence": (n / n_sent) if n else 0.0,
    }


def _feature_matrix(texts: list[str], *, include_pos: bool) -> dict[str, list[float]]:
    """텍스트 리스트 → {feature: [per-text value]}."""
    cols: dict[str, list[float]] = {
        "tokens": [],
        "filler_rate": [],
        "pause_rate": [],
        "tokens_per_sentence": [],
    }
    for t in texts:
        f = _surface_features(t)
        for k in cols:
            cols[k].append(f[k])
    if include_pos:
        from scripttuner.preprocessing.stats import _load_spacy, _pos_stats

        nlp = _load_spacy()
        cols["lexical_density"] = []
        cols["phrasal_verb_ratio"] = []
        for t in texts:
            p = _pos_stats(t, nlp)
            cols["lexical_density"].append(p["lexical_density"])
            cols["phrasal_verb_ratio"].append(p["phrasal_verb_ratio"])
    return cols


def _cohens_d(casual: list[float], semi_formal: list[float]) -> float:
    """표준화 평균차 (semi_formal - casual)/pooled_sd. 분산 0/표본 부족 시 0.0."""
    na, nb = len(casual), len(semi_formal)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = statistics.variance(casual), statistics.variance(semi_formal)
    pooled_sq = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled_sq <= 0:
        return 0.0
    return (statistics.mean(semi_formal) - statistics.mean(casual)) / (pooled_sq**0.5)


def style_separation(
    casual_texts: list[str],
    semi_formal_texts: list[str],
    *,
    include_pos: bool = True,
    verdict_features: frozenset[str] = DEFAULT_VERDICT_FEATURES,
) -> dict[str, Any]:
    """두 스타일 텍스트 그룹의 분리도를 계산한다.

    모든 피처의 Cohen's d는 `features`에 리포트하되, collapse 판정(`collapsed`)은
    `verdict_features`에 든 피처들의 max|d|로만 내린다. pause_rate/tokens 같은
    코퍼스 아티팩트·길이 신호는 기본 판정에서 제외된다(cf. DEFAULT_VERDICT_FEATURES).
    `max_abs_d`는 판정 피처 기준, `max_abs_d_all`은 전체 피처 기준이다.
    """
    if not casual_texts or not semi_formal_texts:
        raise ValueError("both casual_texts and semi_formal_texts must be non-empty")

    cas = _feature_matrix(casual_texts, include_pos=include_pos)
    semi = _feature_matrix(semi_formal_texts, include_pos=include_pos)

    features: dict[str, Any] = {}
    abs_ds_all: list[float] = []
    verdict_ds: list[float] = []
    for name in cas:
        d = _cohens_d(cas[name], semi[name])
        casual_higher = _CASUAL_HIGHER.get(name)
        if casual_higher is None:
            matches = None
        else:
            # casual_higher=True 기대 → d(=semi-casual) < 0 이어야 일치
            matches = (d < 0) if casual_higher else (d > 0)
        used = name in verdict_features
        features[name] = {
            "casual_mean": statistics.mean(cas[name]),
            "semi_formal_mean": statistics.mean(semi[name]),
            "cohens_d": d,
            "expected": (
                None if casual_higher is None
                else ("casual_higher" if casual_higher else "semi_formal_higher")
            ),
            "matches_expected": matches,
            "used_for_verdict": used,
        }
        abs_ds_all.append(abs(d))
        if used:
            verdict_ds.append(abs(d))

    max_abs_d = max(verdict_ds) if verdict_ds else 0.0
    return {
        "stage": "style_separation",
        "n_casual": len(casual_texts),
        "n_semi_formal": len(semi_formal_texts),
        "verdict_features": sorted(verdict_features),
        "features": features,
        "max_abs_d": max_abs_d,
        "max_abs_d_all": max(abs_ds_all) if abs_ds_all else 0.0,
        "collapsed": max_abs_d < COLLAPSE_THRESHOLD,
    }


def style_separation_from_rows(
    rows: list[dict[str, Any]],
    *,
    text_field: str = "prediction",
    include_pos: bool = True,
    verdict_features: frozenset[str] = DEFAULT_VERDICT_FEATURES,
) -> dict[str, Any]:
    """style 라벨된 예측 행들(generate.py 출력)에서 분리도를 계산한다.

    각 행은 `style` ∈ {casual, semi_formal} 과 `text_field`(기본 'prediction')를
    가진다. 두 스타일 모두 존재해야 한다.
    """
    casual = [r[text_field] or "" for r in rows if r.get("style") == "casual"]
    semi = [r[text_field] or "" for r in rows if r.get("style") == "semi_formal"]
    return style_separation(
        casual, semi, include_pos=include_pos, verdict_features=verdict_features
    )
