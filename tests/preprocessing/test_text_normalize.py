"""text_normalize 단위 테스트 (Tier A 구두점 정규화, ADR-0012)."""

from __future__ import annotations

from scripttuner.preprocessing.text_normalize import normalize_punctuation


def _n(text: str) -> str:
    return normalize_punctuation(text)


def test_space_before_period():
    assert _n("I got now .") == "I got now."


def test_space_before_various_puncts():
    assert _n("wait , really ? yes ; ok :") == "wait, really? yes; ok:"


def test_no_exclamation_left_untouched_when_present():
    assert _n("stop !") == "stop!"


def test_doubled_period_spaced():
    assert _n("signal path there . .") == "signal path there."


def test_doubled_period_adjacent():
    assert _n("just that ..") == "just that."


def test_ellipsis_preserved():
    assert _n("I was gonna ...") == "I was gonna..."


def test_ellipsis_three_dots_no_space_preserved():
    assert _n("hmm... yeah") == "hmm... yeah"


def test_four_dots_collapse():
    assert _n("end ....") == "end."


def test_pause_token_adjacent_punct():
    # pause 토큰 뒤 ' .' → 토큰에 부호가 붙음 (추론 시 토큰 strip 후 재정규화로 정리)
    assert _n("got now <pause:long> . And") == "got now <pause:long>. And"


def test_idempotent():
    once = _n("there . . and now .")
    assert _n(once) == once


def test_no_change_when_clean():
    assert _n("Already clean, right? Yes.") == "Already clean, right? Yes."