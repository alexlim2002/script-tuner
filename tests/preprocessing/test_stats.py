from __future__ import annotations

import pytest

from scripttuner.preprocessing.ir import Pair
from scripttuner.preprocessing.stats import _count_fillers, _word_tokens, compute_stats


def _pair(
    pid: str,
    speaker: str,
    spoken: str,
    formal: str,
) -> Pair:
    return Pair(
        pair_id=pid,
        source="TEST",
        style="casual",
        speaker=speaker,
        spoken_text=spoken,
        formal_text=formal,
        monologue_id=pid.split("#casual")[0],
    )


# ----- helpers -----


def test_word_tokens_strips_pause_tokens() -> None:
    assert _word_tokens("Well <pause:short> um <pause:long> ok") == ["well", "um", "ok"]


def test_count_fillers_single_word() -> None:
    assert _count_fillers(["um", "well", "hello", "uh"]) == 3


def test_count_fillers_multi_word() -> None:
    # "you know" + "i mean" → 2
    assert _count_fillers(["yeah", "you", "know", ",", "i", "mean", "right"]) == 2


def test_count_fillers_mixed() -> None:
    # "um" + "you know" = 2
    assert _count_fillers(["um", "you", "know"]) == 2


# ----- compute_stats: empty -----


def test_empty_pairs_returns_empty_n() -> None:
    assert compute_stats([], include_pos=False) == {"n_pairs": 0}


# ----- compute_stats: basic counts/lengths -----


def test_basic_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    pairs = [
        _pair("p1", "A", "hello there friend", "hello"),
        _pair("p2", "A", "another one", "another"),
        _pair("p3", "B", "third utterance", "third"),
    ]
    s = compute_stats(pairs, include_pos=False)
    assert s["source"] == "TEST"
    assert s["n_pairs"] == 3
    assert s["n_unique_speakers"] == 2
    assert s["speakers"] == ["A", "B"]


def test_token_distribution_excludes_pause() -> None:
    pairs = [
        _pair("p1", "A", "<pause:short> a b c <pause:long>", "x y"),
    ]
    s = compute_stats(pairs, include_pos=False)
    assert s["spoken"]["tokens"]["min"] == 3.0
    assert s["spoken"]["tokens"]["max"] == 3.0
    assert s["formal"]["tokens"]["mean"] == 2.0


def test_pause_counts() -> None:
    pairs = [
        _pair(
            "p1",
            "A",
            "<pause:short> a <pause:long> b <pause:short> c",
            "a b c",
        )
    ]
    s = compute_stats(pairs, include_pos=False)
    assert s["spoken"]["pause_short_per_pair"]["max"] == 2.0
    assert s["spoken"]["pause_long_per_pair"]["max"] == 1.0


def test_filler_counts() -> None:
    pairs = [
        _pair("p1", "A", "um well you know hi", "hi there"),
    ]
    s = compute_stats(pairs, include_pos=False)
    # "um" + "well" + "you know" = 3
    assert s["spoken"]["fillers_per_pair"]["max"] == 3.0
    # formal has none
    assert s["formal"]["fillers_per_pair"]["max"] == 0.0


def test_reduction_ratio() -> None:
    pairs = [
        _pair("p1", "A", "a b c d e f g h i j", "a b c d e"),  # 10 → 5
        _pair("p2", "A", "a b c d", "a b"),  # 4 → 2
    ]
    s = compute_stats(pairs, include_pos=False)
    assert s["reduction_ratio"]["mean"] == 0.5
    assert s["reduction_ratio"]["min"] == 0.5


def test_reduction_ratio_zero_spoken_is_zero() -> None:
    pairs = [_pair("p1", "A", "", "anything")]
    s = compute_stats(pairs, include_pos=False)
    assert s["reduction_ratio"]["max"] == 0.0


# ----- POS-based stats (spacy required) -----


def _spacy_model_available() -> bool:
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        return False
    return True


pytestmark_spacy = pytest.mark.skipif(
    not _spacy_model_available(), reason="spacy en_core_web_sm not installed"
)


@pytestmark_spacy
def test_pos_stats_lexical_density() -> None:
    # 'a cat runs quickly' = DET NOUN VERB ADV → content 3 of 4 = 0.75
    pairs = [_pair("p1", "A", "a cat runs quickly", "a cat runs quickly")]
    s = compute_stats(pairs)
    assert s["spoken"]["lexical_density"]["mean"] == pytest.approx(0.75, abs=0.05)


@pytestmark_spacy
def test_pos_stats_phrasal_verb_detected() -> None:
    # spaCy의 prt 의존관계가 잡히는 명확한 phrasal verb 케이스
    # (참고: "look up the word"는 prep으로 잡혀서 phrasal verb로 카운트되지 않음)
    pairs = [_pair("p1", "A", "I gave up easily.", "I gave up easily.")]
    s = compute_stats(pairs)
    assert s["spoken"]["phrasal_verb_ratio"]["max"] > 0.0


@pytestmark_spacy
def test_pos_stats_pronoun_ratio() -> None:
    # 'I told you' = PRON VERB PRON → 대명사 2 / alpha 3 ≈ 0.667 (ADR-0014 신규 피처)
    from scripttuner.preprocessing.stats import _load_spacy, _pos_stats

    nlp = _load_spacy()
    assert _pos_stats("I told you", nlp)["pronoun_ratio"] == pytest.approx(
        2 / 3, abs=0.05
    )
    # 대명사 없는 문장은 0
    assert _pos_stats("cats run fast", nlp)["pronoun_ratio"] == pytest.approx(0.0)


@pytestmark_spacy
def test_pos_stats_empty_text_pronoun_ratio_zero() -> None:
    from scripttuner.preprocessing.stats import _load_spacy, _pos_stats

    nlp = _load_spacy()
    assert _pos_stats("<pause:long>", nlp)["pronoun_ratio"] == 0.0
