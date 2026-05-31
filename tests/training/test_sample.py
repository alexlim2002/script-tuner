"""Speaker-aware type-balanced sampler tests."""

from __future__ import annotations

from collections import Counter

from scripttuner.preprocessing.ir import Monologue
from scripttuner.training import sample as smp


def _mono(stem: str, speaker: str, idx: int) -> Monologue:
    return Monologue(
        source="AMI",
        monologue_id=f"{stem}#mono_{idx:04d}",
        speaker=speaker,
        text="word " * 30,
        utterance_ids=(f"{stem}#s{idx}",),
        n_tokens=30,
    )


def _pool() -> tuple[list[Monologue], dict[str, str]]:
    """20 scenario speakers + 20 nonscenario speakers, 5 monologues each."""
    monos: list[Monologue] = []
    types: dict[str, str] = {}
    for i in range(20):
        stem = f"SC{i:02d}"
        types[stem] = "scenario"
        for j in range(5):
            monos.append(_mono(stem, f"sc_spk{i:02d}", j))
    for i in range(20):
        stem = f"NS{i:02d}"
        types[stem] = "nonscenario"
        for j in range(5):
            monos.append(_mono(stem, f"ns_spk{i:02d}", j))
    return monos, types


def _type_counts(selected: list[Monologue], types: dict[str, str]) -> Counter:
    return Counter(types[smp.stem_of(m.monologue_id)] for m in selected)


def test_stem_of() -> None:
    assert smp.stem_of("ES2002a#mono_0007") == "ES2002a"


def test_balanced_5050() -> None:
    monos, types = _pool()
    out = smp.sample_balanced(monos, types, target=100, seed=42)
    tc = _type_counts(out, types)
    assert tc["scenario"] == 50
    assert tc["nonscenario"] == 50


def test_speaker_integrity_no_split_speakers() -> None:
    monos, types = _pool()
    out = smp.sample_balanced(monos, types, target=100, seed=1)
    chosen = {m.speaker for m in out}
    # every monologue of a chosen speaker is present (whole-speaker inclusion)
    by_spk = Counter(m.speaker for m in monos)
    out_by_spk = Counter(m.speaker for m in out)
    for spk in chosen:
        assert out_by_spk[spk] == by_spk[spk]


def test_deterministic_same_seed() -> None:
    monos, types = _pool()
    a = smp.sample_balanced(monos, types, target=100, seed=7)
    b = smp.sample_balanced(monos, types, target=100, seed=7)
    assert [m.monologue_id for m in a] == [m.monologue_id for m in b]


def test_preserves_input_order() -> None:
    monos, types = _pool()
    out = smp.sample_balanced(monos, types, target=100, seed=3)
    idx = {id(m): i for i, m in enumerate(monos)}
    positions = [idx[id(m)] for m in out]
    assert positions == sorted(positions)


def test_fallback_when_one_type_short() -> None:
    # only 5 nonscenario speakers (25 monos); ask for 100 -> 50/50 impossible
    monos, types = _pool()
    monos = [m for m in monos if not (types[smp.stem_of(m.monologue_id)] == "nonscenario" and m.speaker >= "ns_spk05")]
    out = smp.sample_balanced(monos, types, target=100, seed=42)
    tc = _type_counts(out, types)
    assert tc["nonscenario"] == 25  # all available
    assert len(out) >= 75  # topped up from scenario
    assert tc["scenario"] >= 50


def test_overlap_speaker_assigned_single_type() -> None:
    # speaker appears in both a scenario and nonscenario meeting
    monos = [
        _mono("SC00", "shared", 0),
        _mono("SC00", "shared", 1),  # 2 in scenario
        _mono("NS00", "shared", 0),  # 1 in nonscenario -> assigned scenario
        _mono("NS00", "other_ns", 0),
    ]
    types = {"SC00": "scenario", "NS00": "nonscenario"}
    assigned = smp._assign_speaker_types(
        {"shared": monos[:3], "other_ns": [monos[3]]}, types
    )
    assert assigned["shared"] == "scenario"  # majority type
