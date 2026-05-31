"""예측 평가 — prediction vs reference의 구어성 피처 비교.

모듈 ⑤ stats의 피처 추출기(filler/pause/lexical density)를 재사용해, 예측이 reference만큼
"구어체다운지"를 같은 척도로 비교한다. 의존성 추가 없음.

표면·의미 유사도(BLEU/ROUGE/embedding)는 의도적으로 보류 — 정량 품질 메트릭은 본격 학습
단계 진입 시 도입 결정(docs/status.md `보류` 참조).

입력: `generate`가 만든 predictions.jsonl (prediction/reference 필드).
출력: metrics.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ⑤ stats의 per-text 피처 추출기 재사용(동일 패키지 내부 유틸).
from scripttuner.preprocessing.stats import (
    _PAUSE_LONG_RE,
    _PAUSE_SHORT_RE,
    _count_fillers,
    _distribution,
    _word_tokens,
)

# 부호 앞 공백(전사 잔여 표기, Tier A가 제거 대상; cf. ADR-0012) — pause 토큰의
# 콜론은 제외하기 위해 토큰을 먼저 strip한 텍스트에 적용한다.
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +[.,?!;:]")
_PAUSE_TOKEN_RE = re.compile(r"<pause:\w+>")


def _punct_features(texts: list[str]) -> dict[str, Any]:
    """아이템당 구두점 통계 (cf. ADR-0012 검증용).

    Tier B 효과(쉼표·물음표 복원)와 Tier A 효과(' .' 잔여 제거)를 prediction과
    reference에서 같은 척도로 비교한다. ' .' 카운트는 pause 토큰 콜론에 오염되지
    않도록 토큰 strip 후 측정한다.
    """
    stripped = [_PAUSE_TOKEN_RE.sub(" ", t) for t in texts]
    return {
        "commas_per_item": _distribution([t.count(",") for t in texts]),
        "questions_per_item": _distribution([t.count("?") for t in texts]),
        "periods_per_item": _distribution([t.count(".") for t in texts]),
        "exclaims_per_item": _distribution([t.count("!") for t in texts]),
        "space_before_punct_per_item": _distribution(
            [len(_SPACE_BEFORE_PUNCT_RE.findall(t)) for t in stripped]
        ),
    }


def _text_features(texts: list[str]) -> dict[str, Any]:
    """텍스트 리스트의 구어성 피처 분포(길이/filler/pause/구두점)."""
    token_lists = [_word_tokens(t) for t in texts]
    return {
        "tokens": _distribution([len(ts) for ts in token_lists]),
        "fillers_per_item": _distribution([_count_fillers(ts) for ts in token_lists]),
        "pause_short_per_item": _distribution([len(_PAUSE_SHORT_RE.findall(t)) for t in texts]),
        "pause_long_per_item": _distribution([len(_PAUSE_LONG_RE.findall(t)) for t in texts]),
        "punct": _punct_features(texts),
    }


def run_evaluate(
    *,
    predictions_path: Path,
    output_path: Path,
    include_pos: bool = True,
) -> dict[str, Any]:
    """predictions.jsonl을 읽어 구어성 메트릭을 계산하고 metrics.json을 쓴다."""

    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no predictions in {predictions_path}")

    preds = [(r.get("prediction") or "") for r in rows]
    refs = [(r.get("reference") or "") for r in rows]
    pred_lens = [len(_word_tokens(t)) for t in preds]
    ref_lens = [len(_word_tokens(t)) for t in refs]
    length_ratios = [
        (p / r) if r > 0 else 0.0 for p, r in zip(pred_lens, ref_lens, strict=True)
    ]

    prediction = _text_features(preds)
    reference = _text_features(refs)

    if include_pos:
        from scripttuner.preprocessing.stats import _load_spacy, _pos_stats

        nlp = _load_spacy()
        prediction["lexical_density"] = _distribution(
            [_pos_stats(t, nlp)["lexical_density"] for t in preds]
        )
        reference["lexical_density"] = _distribution(
            [_pos_stats(t, nlp)["lexical_density"] for t in refs]
        )

    metrics: dict[str, Any] = {
        "stage": "finetune_eval",
        "predictions": str(predictions_path),
        "n": len(rows),
        "n_empty_predictions": sum(1 for p in preds if not p.strip()),
        "length_ratio_pred_over_ref": _distribution(length_ratios),
        "prediction": prediction,
        "reference": reference,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
