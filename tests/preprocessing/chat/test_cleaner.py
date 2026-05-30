from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripttuner.preprocessing.chat import cleaner, parser
from scripttuner.preprocessing.ir import Utterance


def _utt(text: str) -> Utterance:
    return Utterance(source="TEST", utterance_id="t#0001", speaker="A", text=text)


# ----- 마커별 단위 케이스 -----


def test_pause_short_tokenized() -> None:
    assert cleaner.clean_text("Well (.) I see") == "Well <pause:short> I see"


def test_pause_long_tokenized() -> None:
    assert cleaner.clean_text("Well (..) I see") == "Well <pause:long> I see"


def test_pause_long_takes_precedence_over_short() -> None:
    # 만약 (..) 가 (.) 두 번으로 잘못 매치되면 출력이 깨짐
    result = cleaner.clean_text("(..) really (.) good")
    assert result == "<pause:long> really <pause:short> good"


def test_overlap_markers_removed() -> None:
    assert cleaner.clean_text("⌈ Good ⌉ thing") == "Good thing"
    assert cleaner.clean_text("⌊2 Yeah ⌋2 right") == "Yeah right"


def test_overlap_close_index_before_word_end() -> None:
    # 닫는 마커 직전 단어끝에 인덱스: 'me2⌋' → 'me' (오염 'me2' 방지)
    assert cleaner.clean_text("⌊2tell me2⌋ now") == "tell me now"
    assert cleaner.clean_text("you ⌊know2⌋ how") == "you know how"


def test_overlap_does_not_touch_real_labels() -> None:
    # F1/F2(포먼트), A1, B2 등 마커 없는 라벨은 보존
    assert cleaner.clean_text("the F2 and the F1") == "the F2 and the F1"
    assert cleaner.clean_text("you can do A1 sauce") == "you can do A1 sauce"
    assert cleaner.clean_text("take B2 vitamins") == "take B2 vitamins"


def test_nonverbal_annotations_removed() -> None:
    assert cleaner.clean_text("&=tsk really &=laugh") == "really"
    assert cleaner.clean_text("&=in So &=ex") == "So"


def test_lang_marker_outer_removed_inner_preserved() -> None:
    # &{l=X And I'm &}l=X → "And I'm"
    assert cleaner.clean_text("&{l=X And I'm &}l=X really") == "And I'm really"


def test_glottal_stop_normalized() -> None:
    assert cleaner.clean_text("ʔuh okay") == "uh okay"
    assert cleaner.clean_text("youʔ would") == "you would"


def test_vowel_lengthening_colon_removed_word_end() -> None:
    assert cleaner.clean_text("I: wanted that") == "I wanted that"
    assert cleaner.clean_text("stuff: I got") == "stuff I got"


def test_vowel_lengthening_colon_removed_word_internal() -> None:
    assert cleaner.clean_text("u:m well") == "um well"
    assert cleaner.clean_text("S:o we") == "So we"
    assert cleaner.clean_text("j:ust because") == "just because"


def test_vowel_lengthening_multiple_colons_in_one_word() -> None:
    assert cleaner.clean_text("perc:e:nt") == "percent"


def test_vowel_lengthening_with_overlap_sandwich() -> None:
    # overlap marker가 colon 사이에 끼는 케이스 (e.g. "Yeah:⌈: ⌉")
    # 1단계 overlap 제거 후 "Yeah::"가 생기는데, 4b가 연속 colon을 모두 제거해야 함
    assert cleaner.clean_text("Yeah:⌈: ⌉ .") == "Yeah ."
    assert cleaner.clean_text("Y⌈2eah⌉2:⌈3: ⌉3 .") == "Yeah ."


def test_trailoff_interrupt_to_period() -> None:
    assert cleaner.clean_text("I want a +/.") == "I want a ."


def test_trailoff_to_ellipsis() -> None:
    assert cleaner.clean_text("I mean I h- +...") == "I mean I h- ..."


def test_word_fillers_preserved() -> None:
    text = "um I mean you know well uh"
    assert cleaner.clean_text(text) == "um I mean you know well uh"


def test_multispace_collapsed() -> None:
    assert cleaner.clean_text("Well    (.)   um") == "Well <pause:short> um"


def test_combined_realistic() -> None:
    raw = "(..) ⌈ &=laugh ⌉ I &{l=X don't &}l=X know +/."
    expected = "<pause:long> I don't know ."
    assert cleaner.clean_text(raw) == expected


# ----- 코멘트 티어 / 스코프 / unintelligible / 밑줄 (ADR-0012 Tier A 보강) -----


def test_comment_tier_removed() -> None:
    assert cleaner.clean_text("piece of horse [% laugh] . I mean") == "piece of horse . I mean"


def test_scope_nonverbal_outer_removed_inner_preserved() -> None:
    # &{n=SNAP just &}n=SNAP → "just" 보존
    assert cleaner.clean_text("dollars &{n=SNAP just &}n=SNAP like") == "dollars just like"


def test_scope_nonverbal_self_closed_removed() -> None:
    assert cleaner.clean_text("wall &{n=THUMP} okay") == "wall okay"


def test_scope_inner_unintelligible_then_stripped() -> None:
    # 스코프 외곽 제거 후 내부 XX가 노출되어 3b가 제거
    assert cleaner.clean_text("sorry &{n=MIC XX knocked &}n=MIC me") == "sorry knocked me"


def test_unintelligible_single_and_runs_removed() -> None:
    assert cleaner.clean_text("you know XX ? Oh") == "you know ? Oh"
    assert cleaner.clean_text("remember that . XXX Ken") == "remember that . Ken"
    assert cleaner.clean_text("fast . X . Hundred") == "fast . . Hundred"


def test_unintelligible_preserves_real_x_words() -> None:
    # X-rays, XSes, X-(false start)는 보존
    assert cleaner.clean_text("more X-rays or") == "more X-rays or"
    assert cleaner.clean_text("section four XSes that") == "section four XSes that"
    assert cleaner.clean_text("and I X- was") == "and I X- was"


def test_unintelligible_after_glottal_strip() -> None:
    # "ʔX"는 ʔ 제거 후 X가 노출되어 unintelligible로 제거돼야 함
    assert cleaner.clean_text("(.) ʔX .") == "<pause:short> ."
    assert cleaner.clean_text("⌊3 ʔX ʔuh ʔuh ⌋3 .") == "uh uh ."


def test_underscore_join_to_space() -> None:
    assert cleaner.clean_text("well uh_you know") == "well uh you know"
    assert cleaner.clean_text("the U_S history") == "the U S history"


def test_uppercase_abbreviations_preserved() -> None:
    # 실제 발화 약어·고유명사는 보존 (X 규칙은 X에만 적용)
    assert cleaner.clean_text("a bank CD as") == "a bank CD as"
    assert cleaner.clean_text("called an IRA it's") == "called an IRA it's"
    assert cleaner.clean_text("see Mike on TV ?") == "see Mike on TV ?"


# ----- clean(list[Utterance]) API -----


def test_clean_returns_new_utterances_with_text_updated() -> None:
    inputs = [
        _utt("(..) Well (.) um"),
        _utt("⌈ &=laugh ⌉ Good ."),
    ]
    out = cleaner.clean(inputs)
    assert len(out) == 2
    assert out[0].text == "<pause:long> Well <pause:short> um"
    assert out[1].text == "Good ."


def test_clean_preserves_other_fields() -> None:
    src = Utterance(
        source="SBCSAE",
        utterance_id="SBC016#0001",
        speaker="TAMM",
        text="(.) hi",
        t_start_ms=100,
        t_end_ms=500,
        metadata={"line_no": 16},
    )
    out = cleaner.clean([src])[0]
    assert out.source == src.source
    assert out.utterance_id == src.utterance_id
    assert out.speaker == src.speaker
    assert out.t_start_ms == src.t_start_ms
    assert out.t_end_ms == src.t_end_ms
    assert out.metadata == src.metadata
    assert out.text == "<pause:short> hi"


# ----- 통합 검증: 실제 SBC016.cha -----

_REAL_FILE = Path("datasets/sbcsae/SBC016.cha")


@pytest.mark.skipif(not _REAL_FILE.exists(), reason="SBC016.cha not downloaded")
def test_integration_sbc016_no_residual_chat_markers() -> None:
    _, utts = parser.parse(_REAL_FILE)
    cleaned = cleaner.clean(utts)
    blob = "\n".join(u.text for u in cleaned)
    # 잔존 마커가 없어야 함
    assert "(.)" not in blob
    assert "(..)" not in blob
    assert "⌈" not in blob and "⌉" not in blob
    assert "⌊" not in blob and "⌋" not in blob
    assert "&=" not in blob
    assert "&{l=" not in blob and "&}l=" not in blob
    assert "ʔ" not in blob
    assert "+/." not in blob
    assert "+..." not in blob
    # vowel-lengthening colon (e.g. "u:m", "I:", "perc:e:nt") 잔존 금지
    # 단, 우리가 생성한 <pause:*> 토큰의 콜론은 제외
    blob_no_pause = re.sub(r"<pause:\w+>", "", blob)
    assert not re.search(r"[a-zA-Z]:", blob_no_pause), (
        "vowel-lengthening colon left in cleaned text"
    )
    # 변환된 포즈 토큰은 있어야 함 (SBC016는 포즈가 매우 많음)
    assert "<pause:short>" in blob
    assert "<pause:long>" in blob


@pytest.mark.skipif(not _REAL_FILE.exists(), reason="SBC016.cha not downloaded")
def test_integration_sbc016_word_fillers_preserved() -> None:
    _, utts = parser.parse(_REAL_FILE)
    cleaned = cleaner.clean(utts)
    blob = " ".join(u.text for u in cleaned).lower()
    # 단어형 필러는 보존
    assert " um " in blob or blob.startswith("um ")
    assert " uh " in blob or blob.startswith("uh ")
    assert "you know" in blob
