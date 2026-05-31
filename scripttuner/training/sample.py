"""Speaker-aware, type-balanced monologue 샘플러.

semi_formal(AMI) 풀(~8,900)을 casual(SBCSAE, ~1,800)과 스케일을 맞춘 ~2,000으로
줄이되, 두 편향을 통제한다 (cf. ADR-0013):

  - **주제 편향**: AMI 풀은 scenario(리모컨 설계 단일 주제) 75% 치우침. scenario :
    nonscenario = 1:1로 뽑아 다양성 높은 nonscenario를 가중한다.
  - **화자 누수**: 화자를 한 split에만 두는 speaker-aware split(split.py)을 깨지
    않도록, 샘플 선택도 **화자 단위**로 한다 — 한 화자의 monologue는 통째로
    포함되거나 통째로 제외된다.

순수 함수다(I/O 없음). CLI 배선은 별도.
"""

from __future__ import annotations

import random
from collections import defaultdict

from scripttuner.preprocessing.ir import Monologue


def stem_of(monologue_id: str) -> str:
    """'ES2002a#mono_0001' -> 'ES2002a'."""
    return monologue_id.split("#", 1)[0]


def _assign_speaker_types(
    speaker_monos: dict[str, list[Monologue]],
    meeting_type_of: dict[str, str],
) -> dict[str, str]:
    """각 화자를 단일 type으로 배정한다.

    한 화자가 여러 type 회의에 걸쳐 있으면(겹침 화자), 그 화자의 monologue가 가장
    많은 type으로 배정한다(tie는 type명 정렬로 결정적). 화자 단위 무결성을 위해
    화자는 정확히 한 type에만 속한다.
    """
    assigned: dict[str, str] = {}
    for speaker, monos in speaker_monos.items():
        by_type: dict[str, int] = defaultdict(int)
        for m in monos:
            by_type[meeting_type_of.get(stem_of(m.monologue_id), "unknown")] += 1
        # 최다 type, tie는 type명 사전순
        assigned[speaker] = min(by_type, key=lambda t: (-by_type[t], t))
    return assigned


def sample_balanced(
    monologues: list[Monologue],
    meeting_type_of: dict[str, str],
    *,
    target: int = 2000,
    seed: int = 42,
) -> list[Monologue]:
    """화자 단위·type 균형(1:1) 샘플을 반환한다 (원본 순서 보존).

    Args:
        monologues: 후보 monologue 풀.
        meeting_type_of: stem -> meeting type (e.g. 'scenario'/'nonscenario').
        target: 목표 monologue 수. type별로 target/2씩 채운다.
        seed: 화자 셔플 시드(재현성).

    한 type이 target/2에 못 미치면 가용한 만큼만 담고, 부족분은 상대 type에서
    화자 단위로 보충한다. 화자 무결성·결정성은 항상 유지된다.
    """
    speaker_monos: dict[str, list[Monologue]] = defaultdict(list)
    for m in monologues:
        speaker_monos[m.speaker].append(m)

    speaker_type = _assign_speaker_types(speaker_monos, meeting_type_of)

    # type -> 화자 목록(시드 셔플). monologue 많은 화자부터 담으면 적은 화자가
    # 늘 잘리므로, 셔플로 화자 다양성을 확보한다.
    speakers_by_type: dict[str, list[str]] = defaultdict(list)
    for speaker, t in speaker_type.items():
        speakers_by_type[t].append(speaker)
    rng = random.Random(seed)
    for t in speakers_by_type:
        speakers_by_type[t].sort()  # 결정적 기준 순서
        rng.shuffle(speakers_by_type[t])

    per_type = target // 2
    chosen_speakers: set[str] = set()
    counts: dict[str, int] = defaultdict(int)

    def fill(type_name: str, quota: int) -> None:
        for speaker in speakers_by_type.get(type_name, []):
            if counts[type_name] >= quota:
                break
            if speaker in chosen_speakers:
                continue
            chosen_speakers.add(speaker)
            counts[type_name] += len(speaker_monos[speaker])

    # 1차: 각 type을 per_type까지
    types = sorted(speakers_by_type)
    for t in types:
        fill(t, per_type)

    # 2차: 부족분을 상대 type에서 보충 (전체 target 도달까지)
    selected_total = sum(counts.values())
    if selected_total < target:
        remaining = [
            s
            for t in types
            for s in speakers_by_type[t]
            if s not in chosen_speakers
        ]
        for speaker in remaining:
            if selected_total >= target:
                break
            chosen_speakers.add(speaker)
            selected_total += len(speaker_monos[speaker])

    return [m for m in monologues if m.speaker in chosen_speakers]
