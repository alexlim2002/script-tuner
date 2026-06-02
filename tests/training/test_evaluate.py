from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripttuner import cli
from scripttuner.preprocessing.ir import Pair
from scripttuner.training.evaluate import run_evaluate


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _spokenness_deps_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        return False
    return True


def test_evaluate_computes_core_metrics(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        pred_path,
        [
            {
                "prediction": "well I mean you know it was fine",
                "reference": "<pause:long> well it was fine you know",
            },
            {"prediction": "", "reference": "okay sure"},
        ],
    )
    out = tmp_path / "metrics.json"

    metrics = run_evaluate(predictions_path=pred_path, output_path=out, include_pos=False)

    assert metrics["n"] == 2
    assert metrics["n_empty_predictions"] == 1
    assert "tokens" in metrics["prediction"]
    assert "fillers_per_item" in metrics["prediction"]
    assert "pause_long_per_item" in metrics["reference"]
    assert "length_ratio_pred_over_ref" in metrics
    # lexical_density is POS-based, so absent when include_pos=False.
    assert "lexical_density" not in metrics["prediction"]
    # punctuation features present on both sides (cf. ADR-0012 before/after)
    assert "punct" in metrics["prediction"]
    assert "commas_per_item" in metrics["prediction"]["punct"]
    assert "space_before_punct_per_item" in metrics["prediction"]["punct"]
    assert out.exists()


def test_evaluate_punct_features_values(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        pred_path,
        [
            # repunct 출력: 쉼표 2, 물음표 1, ' .' 0
            {"prediction": "Well, okay, you know? Yeah.", "reference": "x"},
            # 전사 잔여: ' .' 2개 (pause 토큰 콜론은 제외돼야 함)
            {"prediction": "hello <pause:long> . there .", "reference": "x"},
        ],
    )
    out = tmp_path / "metrics.json"
    m = run_evaluate(predictions_path=pred_path, output_path=out, include_pos=False)
    p = m["prediction"]["punct"]
    # 아이템 [2, 0] → max 쉼표 2
    assert p["commas_per_item"]["max"] == 2
    assert p["questions_per_item"]["max"] == 1
    # ' .' : 첫 아이템 0, 둘째 2 (pause 콜론 오염 없이)
    assert p["space_before_punct_per_item"]["max"] == 2


def test_extract_formal_text_roundtrip() -> None:
    from scripttuner.training.formatters import _build_input, extract_formal_text

    wrapped = _build_input("the cat sat on the mat", "<style:casual>", "Rewrite this.")
    assert extract_formal_text(wrapped) == "the cat sat on the mat"
    # 마커 없는 평문 → None (호출자 폴백)
    assert extract_formal_text("no markers here") is None


@pytest.mark.skipif(
    not _spokenness_deps_available(),
    reason="spacy en_core_web_sm and/or scikit-learn not installed",
)
def test_evaluate_spokenness_delta(tmp_path: Path) -> None:
    from scripttuner.training.formatters import _build_input
    from scripttuner.training.spokenness import save, train

    spoken = "I think you know what I mean. We just did it. He told me so. Right?"
    formal = (
        "The committee subsequently determined that the proposed methodology "
        "required substantial revision prior to formal publication."
    )
    pairs = [
        Pair(
            pair_id=f"p{i}",
            source="TEST",
            style="casual",
            speaker=f"S{i}",
            spoken_text=spoken,
            formal_text=formal,
            monologue_id=f"p{i}",
        )
        for i in range(10)
    ]
    model, _ = train(pairs, seed=0)
    model_path = tmp_path / "spokenness" / "clf.joblib"
    save(model, model_path)

    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(
        pred_path,
        [{"input": _build_input(formal, "<style:casual>", "Rewrite."),
          "prediction": spoken, "reference": spoken}],
    )
    out = tmp_path / "metrics.json"
    m = run_evaluate(
        predictions_path=pred_path,
        output_path=out,
        include_pos=False,
        spokenness_model_path=model_path,
    )

    s = m["spokenness"]
    assert s["n_input_unwrapped"] == 1  # 래핑된 input에서 formal_text 복원 성공
    assert s["model"] == str(model_path)
    # prediction(구어체)이 input(문어체)보다 더 구어스러움 → 델타 > 0
    assert s["p_spoken_delta"]["mean"] > 0.0
    assert s["p_spoken_prediction"]["mean"] > s["p_spoken_input"]["mean"]
    # reference가 있으니 천장 분포 포함, style 없으니 by_style 없음
    assert "p_spoken_reference" in s
    assert "by_style" not in s


@pytest.mark.skipif(
    not _spokenness_deps_available(),
    reason="spacy en_core_web_sm and/or scikit-learn not installed",
)
def test_evaluate_spokenness_paired_no_reference(tmp_path: Path) -> None:
    """paired-generation 산출물: reference 없음 + style별 집계(최종 모델 형태)."""
    from scripttuner.training.formatters import _build_input
    from scripttuner.training.spokenness import save, train

    spoken = "I think you know what I mean. We just did it. He told me so. Right?"
    formal = (
        "The committee subsequently determined that the proposed methodology "
        "required substantial revision prior to formal publication."
    )
    pairs = [
        Pair(
            pair_id=f"p{i}",
            source="TEST",
            style="casual",
            speaker=f"S{i}",
            spoken_text=spoken,
            formal_text=formal,
            monologue_id=f"p{i}",
        )
        for i in range(10)
    ]
    model, _ = train(pairs, seed=0)
    model_path = tmp_path / "clf.joblib"
    save(model, model_path)

    pred_path = tmp_path / "paired_predictions.jsonl"
    _write_predictions(
        pred_path,
        [
            {
                "style": "casual",
                "input": _build_input(formal, "<style:casual>", "Rewrite."),
                "prediction": spoken,
            },
            {
                "style": "semi_formal",
                "input": _build_input(formal, "<style:semi_formal>", "Rewrite."),
                "prediction": spoken,
            },
        ],
    )
    out = tmp_path / "metrics.json"
    m = run_evaluate(
        predictions_path=pred_path,
        output_path=out,
        include_pos=False,
        spokenness_model_path=model_path,
    )

    s = m["spokenness"]
    # reference 전무 → 천장 분포 생략
    assert "p_spoken_reference" not in s
    # style별 델타 집계 존재
    assert set(s["by_style"]) == {"casual", "semi_formal"}
    assert s["by_style"]["casual"]["n"] == 1
    assert s["by_style"]["semi_formal"]["n"] == 1


def test_evaluate_cli_no_pos(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(pred_path, [{"prediction": "hi there", "reference": "hello"}])

    rc = cli.main(["evaluate", "--predictions", str(pred_path), "--no-pos"])

    assert rc == 0
    assert (tmp_path / "metrics.json").exists()
