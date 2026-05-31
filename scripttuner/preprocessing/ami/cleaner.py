"""AMI (NXT) 발화 정규화.

① 파서가 raw로 보존한 NXT 마커를 제거하고, 정규화 후 빈 발화는 **제거**한다
(빈 발화를 ③ monologue에 넘기면 화자 버퍼를 조기 flush하므로, cf. ADR-0009).

  - ``[vocalsound:TYPE]`` → 제거 (웃음·기침 등 비어휘)
  - ``[disfmarker]``      → 제거 (비유창성 경계 주석)
  - ``[gap]``             → 제거 (미전사 구간)
  - ``[transformerror]``  → 제거

AMI는 필러(uh, um)를 실제 단어(``<w>``)로 전사하므로 별도 처리하지 않는다.
SBCSAE/Switchboard와 달리 pause 토큰은 없다.

NXT는 철자 발음(spelled-out letters)을 각 글자 뒤 밑줄로 표기한다:
``T_V_``=TV, ``L_C_D_``=LCD, ``X_M_L_``=XML, ``T_V_'s``=TV's,
``anti-R_S_I_``=anti-RSI. 이는 잡음이 아니라 실제 발화(약어)이므로 밑줄만 떼어
약어로 **복원**한다. 밑줄은 NXT에서 일관되게 이 용도이므로 글자 뒤 밑줄을 일괄
제거한다(1글자 ``X_``→X 포함; cf. ADR-0013 검토).
"""

from __future__ import annotations

import re
from dataclasses import replace

from scripttuner.preprocessing.ir import Utterance

# 파서가 삽입한 NXT 마커 토큰
_MARKER_RE = re.compile(r"\[vocalsound:[^\]]*\]|\[disfmarker\]|\[gap\]|\[transformerror\]")
# 철자 발음 표기: 글자 뒤 밑줄 (T_V_ -> TV). 글자에 직접 붙은 '_'만 제거하므로
# 일반 토큰은 영향 없다.
_SPELLED_LETTER_RE = re.compile(r"([A-Za-z])_")
# 마커 제거로 생긴 구두점 앞 공백
_SPACE_BEFORE_PUNC_RE = re.compile(r"\s+([,.!?;:])")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """단일 발화 텍스트에서 NXT 마커·철자표기를 정규화하고 공백/구두점을 정리한다."""
    text = _MARKER_RE.sub(" ", text)
    text = _SPELLED_LETTER_RE.sub(r"\1", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNC_RE.sub(r"\1", text)
    return text


def clean(utterances: list[Utterance]) -> list[Utterance]:
    """text를 정규화하고, 정규화 후 빈 발화는 제거한 새 리스트를 반환한다."""
    cleaned: list[Utterance] = []
    for u in utterances:
        text = clean_text(u.text)
        if text:
            cleaned.append(replace(u, text=text))
    return cleaned
