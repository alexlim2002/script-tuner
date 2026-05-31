"""AMI cleaner tests."""

from __future__ import annotations

from scripttuner.preprocessing.ami import cleaner
from scripttuner.preprocessing.ir import Utterance


def _utt(text: str, uid: str = "M1#s1") -> Utterance:
    return Utterance(source="AMI", utterance_id=uid, speaker="P001", text=text)


def test_clean_text_removes_markers() -> None:
    out = cleaner.clean_text("Hello, everyone [vocalsound:laugh] today [disfmarker] right")
    assert out == "Hello, everyone today right"


def test_clean_text_removes_gap_and_transformerror() -> None:
    assert cleaner.clean_text("so [gap] then [transformerror] yes") == "so then yes"


def test_clean_text_fixes_space_before_punct() -> None:
    # marker removal can leave a dangling space before punctuation
    assert cleaner.clean_text("well [disfmarker] , okay") == "well, okay"


def test_clean_drops_marker_only_utterances() -> None:
    utts = [_utt("[vocalsound:laugh]"), _utt("real words here", "M1#s2")]
    out = cleaner.clean(utts)
    assert [u.text for u in out] == ["real words here"]


def test_spelled_letters_restored_to_acronym() -> None:
    assert cleaner.clean_text("I have a T_V_ and a L_C_D_") == "I have a TV and a LCD"
    assert cleaner.clean_text("the X_M_L_ viewer") == "the XML viewer"


def test_spelled_letters_keep_affixes() -> None:
    assert cleaner.clean_text("two T_V_'s here") == "two TV's here"
    assert cleaner.clean_text("a T_V_, really") == "a TV, really"
    assert cleaner.clean_text("anti-R_S_I_ stuff") == "anti-RSI stuff"
    assert cleaner.clean_text("just a X_ there") == "just a X there"


def test_normal_words_untouched_by_spelled_rule() -> None:
    assert cleaner.clean_text("a normal sentence here") == "a normal sentence here"


def test_clean_preserves_metadata_and_timing() -> None:
    u = Utterance(
        source="AMI",
        utterance_id="M1#s1",
        speaker="P001",
        text="hi [gap] there",
        t_start_ms=1000,
        t_end_ms=2000,
        metadata={"agent": "A"},
    )
    out = cleaner.clean([u])[0]
    assert out.text == "hi there"
    assert out.t_start_ms == 1000
    assert out.metadata == {"agent": "A"}
