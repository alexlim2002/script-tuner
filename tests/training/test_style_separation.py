"""Style separation metric tests (POS disabled to avoid spaCy dependency)."""

from __future__ import annotations

import pytest

from scripttuner.training import style_separation as ss

_CASUAL = [
    "um well you know i mean like it was kind of a long day uh",
    "like you know i mean um it was sort of weird uh well",
    "well um you know like i mean uh it kind of happened",
]
_SEMI = [
    "the proposal is clear and well organized for the committee",
    "we should evaluate the results before the final decision",
    "the team presented a structured overview of the plan",
]


def test_separated_groups_not_collapsed() -> None:
    out = ss.style_separation(_CASUAL, _SEMI, include_pos=False)
    assert out["collapsed"] is False
    fr = out["features"]["filler_rate"]
    assert fr["cohens_d"] < 0  # semi_formal has fewer fillers
    assert fr["matches_expected"] is True
    assert fr["used_for_verdict"] is True
    assert out["max_abs_d"] >= 0.2


def test_tokens_per_sentence_separates_segmentation() -> None:
    # casual: 짧게 토막(문장당 단어 적음), semi: 길게 이어붙임(문장당 단어 많음).
    # filler가 동일해도 분절 차이만으로 분리가 잡혀야 한다.
    casual = ["a b c. d e f. g h i.", "j k. l m. n o.", "p q. r s. t u."]
    semi = ["a b c d e f g h i.", "j k l m n o.", "p q r s t u."]
    out = ss.style_separation(casual, semi, include_pos=False)
    tps = out["features"]["tokens_per_sentence"]
    assert tps["used_for_verdict"] is True
    assert tps["cohens_d"] > 0  # semi_formal has more tokens per sentence
    assert tps["matches_expected"] is True
    assert out["collapsed"] is False


def test_pause_excluded_from_verdict_by_default() -> None:
    # pause_rate differs hugely (corpus artifact) but must NOT drive the verdict;
    # with all other features identical, the verdict should be collapse.
    casual = ["<pause:long> the plan is clear", "<pause:short> we review results", "a day"]
    semi = ["the plan is clear", "we review results", "a day"]
    out = ss.style_separation(casual, semi, include_pos=False)
    assert out["features"]["pause_rate"]["used_for_verdict"] is False
    assert out["features"]["pause_rate"]["cohens_d"] != 0.0  # still reported
    assert abs(out["max_abs_d_all"]) > abs(out["max_abs_d"])  # artifact only in _all
    assert out["collapsed"] is True


def test_verdict_features_override() -> None:
    out = ss.style_separation(
        _CASUAL, _SEMI, include_pos=False, verdict_features=frozenset({"pause_rate"})
    )
    assert out["verdict_features"] == ["pause_rate"]
    assert out["features"]["filler_rate"]["used_for_verdict"] is False


def test_identical_groups_collapse() -> None:
    same = ["the plan is clear and organized", "we will review the results", "a fine day"]
    out = ss.style_separation(same, list(same), include_pos=False)
    assert out["collapsed"] is True
    assert out["max_abs_d"] == pytest.approx(0.0)


def test_cohens_d_small_sample_or_zero_variance() -> None:
    assert ss._cohens_d([1.0], [2.0]) == 0.0  # too few
    assert ss._cohens_d([1.0, 1.0], [1.0, 1.0]) == 0.0  # zero variance


def test_empty_group_raises() -> None:
    with pytest.raises(ValueError):
        ss.style_separation([], _SEMI, include_pos=False)


def test_from_rows_groups_by_style() -> None:
    rows = (
        [{"style": "casual", "prediction": t} for t in _CASUAL]
        + [{"style": "semi_formal", "prediction": t} for t in _SEMI]
    )
    out = ss.style_separation_from_rows(rows, include_pos=False)
    assert out["n_casual"] == 3
    assert out["n_semi_formal"] == 3
    assert out["collapsed"] is False
