from __future__ import annotations

from pathlib import Path

import pytest

from scripttuner.preprocessing.ir import Pair


def _deps_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _deps_available(), reason="spacy en_core_web_sm and/or scikit-learn not installed"
)


def _pair(pid: str, speaker: str, spoken: str, formal: str) -> Pair:
    return Pair(
        pair_id=pid,
        source="TEST",
        style="casual",
        speaker=speaker,
        spoken_text=spoken,
        formal_text=formal,
        monologue_id=pid,
    )


# 구어체: 대명사 많고 짧게 토막. 문어체: 길고 명사 밀도 높고 대명사 없음.
_SPOKEN = "I think you know what I mean. We just did it. He told me so."
_FORMAL = (
    "The committee subsequently determined that the proposed methodology required "
    "substantial revision prior to formal publication in the relevant journal."
)


def test_extract_features_keys_and_pronouns() -> None:
    from scripttuner.preprocessing.stats import _load_spacy
    from scripttuner.training.spokenness import FEATURE_NAMES, extract_features

    nlp = _load_spacy()
    f = extract_features(_SPOKEN, nlp)
    assert set(f) == set(FEATURE_NAMES)
    # 대명사 많은 구어체가 명사 위주 문어체보다 pronoun_ratio가 높다.
    assert f["pronoun_ratio"] > extract_features(_FORMAL, nlp)["pronoun_ratio"]


def test_extract_features_filler_and_contraction() -> None:
    from scripttuner.preprocessing.stats import _load_spacy
    from scripttuner.training.spokenness import extract_features

    nlp = _load_spacy()
    # filler(you know/I mean)·축약형(don't/I'm)이 있는 구어체가 높다.
    spoken = extract_features("Well, you know, I mean, I don't think I'm sure.", nlp)
    formal = extract_features("The analysis did not establish a definitive result.", nlp)
    assert spoken["filler_rate"] > formal["filler_rate"]
    assert spoken["contraction_rate"] > formal["contraction_rate"]
    # pause 마커는 아티팩트라 피처에 없다(제외 확인).
    assert "pause_rate" not in spoken


def test_train_scores_spoken_higher_and_roundtrip(tmp_path: Path) -> None:
    from scripttuner.training.spokenness import load, save, score, train

    # 화자별로 약간씩 변주해 충분한 표본 확보(speaker-aware 의미는 여기선 불필요).
    pairs = [
        _pair(f"p{i}", f"S{i}", f"{_SPOKEN} Right? Okay.", f"{_FORMAL} Furthermore, it persisted.")
        for i in range(10)
    ]
    model, metrics = train(pairs, seed=0)

    assert metrics["n_pairs"] == 10
    assert metrics["n_samples"] == 20
    assert 0.0 <= metrics["holdout_accuracy"] <= 1.0
    assert 0.0 <= metrics["holdout_auc"] <= 1.0
    assert set(metrics["coefficients"]) == set(model.feature_names)

    # 구어체 텍스트의 P(spoken)이 문어체보다 높아야 한다.
    p_spoken, p_formal = score([_SPOKEN, _FORMAL], model)
    assert p_spoken > p_formal

    # joblib 라운드트립: 로드한 모델이 동일 점수를 낸다.
    path = tmp_path / "spokenness" / "clf.joblib"
    save(model, path)
    assert path.exists()
    reloaded = load(path)
    assert score([_SPOKEN, _FORMAL], reloaded) == pytest.approx([p_spoken, p_formal])


def test_train_empty_raises() -> None:
    from scripttuner.training.spokenness import train

    with pytest.raises(ValueError):
        train([])
