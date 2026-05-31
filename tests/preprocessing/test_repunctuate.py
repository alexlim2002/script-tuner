"""Tier B 재구두점 단위 테스트 (ADR-0012). LLM은 FakeClient로 대체.

핵심은 pause 재삽입 정확성 + 불변식 폴백이다 — 네트워크 없이 검증한다.
"""

from __future__ import annotations

from typing import Any

from scripttuner.preprocessing import repunctuate as rp


# ----- pure helpers: pause 추출/재삽입 -----


def test_extract_pause_positions_basic():
    pos, n = rp.extract_pause_positions("<pause:long> No I <pause:short> just")
    assert n == 3  # No, I, just
    assert pos == [(0, "<pause:long>"), (2, "<pause:short>")]


def test_extract_trailing_pause():
    pos, n = rp.extract_pause_positions("okay then <pause:long>")
    assert n == 2
    assert pos == [(2, "<pause:long>")]


def test_strip_pauses_collapses_space():
    assert rp.strip_pauses("a <pause:long>  b <pause:short> c") == "a b c"


def test_reinsert_roundtrip_identity_when_output_equals_plain():
    spoken = "<pause:long> No I <pause:short> just go"
    pos, _ = rp.extract_pause_positions(spoken)
    plain = rp.strip_pauses(spoken)  # "No I just go"
    # 출력이 평문과 동일하면 재삽입은 원래 pause 위치를 복원
    assert rp.reinsert_pauses(plain, pos) == "<pause:long> No I <pause:short> just go"


def test_reinsert_before_word_ignores_output_punctuation():
    # pause는 단어 인덱스 기준 — 출력이 구두점을 어디 넣든 단어 앞에 들어감
    spoken = "No I <pause:long> just go"  # pause before word index 2 ("just")
    pos, _ = rp.extract_pause_positions(spoken)
    output = "No, I. Just go."  # 재구두점된 출력 (단어열 동일)
    assert rp.reinsert_pauses(output, pos) == "No, I. <pause:long> Just go."


def test_reinsert_trailing_pause():
    spoken = "all done <pause:short>"
    pos, _ = rp.extract_pause_positions(spoken)
    assert rp.reinsert_pauses("All done.", pos) == "All done. <pause:short>"


def test_reinsert_multiple_pauses_same_index():
    spoken = "go <pause:short> <pause:long> now"
    pos, _ = rp.extract_pause_positions(spoken)
    assert pos == [(1, "<pause:short>"), (1, "<pause:long>")]
    assert rp.reinsert_pauses("Go now.", pos) == "Go <pause:short> <pause:long> now."


def test_words_excludes_pause_and_lowercases():
    assert rp._words("<pause:long> Don't Go-now") == ["don't", "go-now"]


# ----- repunctuate_spoken: invariant + fallback -----


class FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((system, user))
        return self._response, {"prompt_tokens": 1, "completion_tokens": 1}


def test_apply_success_reinserts_pauses():
    spoken = "<pause:long> well I think so yeah"
    client = FakeClient("Well, I think so, yeah?")
    out, meta = rp.repunctuate_spoken(spoken, client=client, model="m")
    assert meta["applied"] is True
    assert out == "<pause:long> Well, I think so, yeah?"


def test_fallback_on_word_mismatch_returns_input_unchanged():
    spoken = "well I think so"
    # LLM이 단어를 바꿈("believe" 삽입) → 불변식 실패 → 입력 그대로
    client = FakeClient("Well, I believe so.")
    out, meta = rp.repunctuate_spoken(spoken, client=client, model="m")
    assert meta["applied"] is False
    assert meta["reason"] == "word_mismatch"
    assert out == spoken


def test_fallback_preserves_false_start_hyphen():
    # LLM이 false-start 하이픈을 떨어뜨리면(do- → do) 단어열 불일치 → 폴백
    spoken = "I wanna do- it"
    client = FakeClient("I wanna do it.")
    out, meta = rp.repunctuate_spoken(spoken, client=client, model="m")
    assert meta["applied"] is False
    assert out == spoken


def test_empty_after_strip_is_noop():
    out, meta = rp.repunctuate_spoken("<pause:long>", client=FakeClient("x"), model="m")
    assert meta["applied"] is False
    assert meta["reason"] == "empty"


def test_cache_hit_skips_client(tmp_path):
    from scripttuner.persistence.cache import DiskCache

    spoken = "well okay"
    cache = DiskCache(tmp_path)
    c1 = FakeClient("Well, okay.")
    out1, meta1 = rp.repunctuate_spoken(spoken, client=c1, cache=cache, model="m")
    assert meta1["from_cache"] is False
    c2 = FakeClient("SHOULD NOT BE USED")
    out2, meta2 = rp.repunctuate_spoken(spoken, client=c2, cache=cache, model="m")
    assert meta2["from_cache"] is True
    assert len(c2.calls) == 0
    assert out1 == out2 == "Well, okay."
