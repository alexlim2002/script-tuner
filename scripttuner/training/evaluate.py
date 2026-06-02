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


def _delta_block(p_input: list[float], p_pred: list[float]) -> dict[str, Any]:
    """input→prediction P(spoken) 델타 블록(분포)."""
    delta = [p - i for p, i in zip(p_pred, p_input, strict=True)]
    return {
        "n": len(delta),
        "p_spoken_input": _distribution(p_input),
        "p_spoken_prediction": _distribution(p_pred),
        "p_spoken_delta": _distribution(delta),  # headline: prediction - input
    }


def _spokenness_block(
    rows: list[dict[str, Any]],
    preds: list[str],
    refs: list[str],
    model_path: Path,
) -> dict[str, Any]:
    """분류기 P(spoken)으로 input→prediction 델타 블록을 만든다(cf. ADR-0014).

    델타의 'before'는 instruction/control token을 벗긴 순수 formal_text다 —
    `input`은 래핑된 프롬프트라 그대로 점수 내면 오염된다.

    reference는 있을 때만(없으면 paired-generation 산출물) 천장(ceiling) 분포로 추가.
    행에 `style`이 있으면 style별 델타도 집계한다(style-branching 최종 모델용).
    """
    from scripttuner.training.formatters import extract_formal_text
    from scripttuner.training.spokenness import load, score

    model = load(model_path)
    inputs_wrapped = [(r.get("input") or "") for r in rows]
    inputs_formal = [extract_formal_text(w) or w for w in inputs_wrapped]
    n_unwrapped = sum(1 for w in inputs_wrapped if extract_formal_text(w) is not None)

    p_input = score(inputs_formal, model)
    p_pred = score(preds, model)

    block: dict[str, Any] = {
        "model": str(model_path),
        "n_input_unwrapped": n_unwrapped,  # 마커 복원 성공 수(나머지는 원문 폴백)
        **_delta_block(p_input, p_pred),
    }

    # reference 천장 — 비어있지 않은 reference가 하나라도 있을 때만(paired엔 없음).
    if any(r.strip() for r in refs):
        block["p_spoken_reference"] = _distribution(score(refs, model))

    # style별 델타 — 행에 style이 있으면(최종 style-branching 모델).
    styles = [r.get("style") for r in rows]
    if any(s for s in styles):
        by_style: dict[str, Any] = {}
        for style in sorted({s for s in styles if s}):
            idx = [i for i, s in enumerate(styles) if s == style]
            by_style[style] = _delta_block(
                [p_input[i] for i in idx], [p_pred[i] for i in idx]
            )
        block["by_style"] = by_style

    return block


def run_evaluate(
    *,
    predictions_path: Path,
    output_path: Path,
    include_pos: bool = True,
    spokenness_model_path: Path | None = None,
) -> dict[str, Any]:
    """predictions.jsonl을 읽어 구어성 메트릭을 계산하고 metrics.json을 쓴다.

    `spokenness_model_path`가 주어지면 분류기 P(spoken)으로 input→prediction 델타
    블록을 추가한다(cf. ADR-0014). 미지정 시 기존 분포 메트릭만 산출한다.
    """

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

    if spokenness_model_path is not None:
        metrics["spokenness"] = _spokenness_block(rows, preds, refs, spokenness_model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics
