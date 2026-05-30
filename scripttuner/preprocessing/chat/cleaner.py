"""CHAT (CHILDES) 발화 정규화.

파서가 추출한 `Utterance` 리스트를 받아 CHA-specific 마커를 정규화한다.
처리 정책은 `docs/design/preprocessing_pipeline.md`의 "마커 처리 정책" 표 참조.
포즈 토큰화 근거는 ADR-0003.
"""

from __future__ import annotations

import re
from dataclasses import replace

from scripttuner.preprocessing.ir import Utterance

# 1. 오버랩 마커: ⌈...⌉ 또는 ⌊2...⌋2 형태 (숫자 인덱스 옵셔널)
#    인덱스 숫자는 데이터상 비대칭으로 붙는다: 여는쪽은 기호 뒤(`⌊2`), 닫는쪽은
#    기호 뒤(`⌋2`)뿐 아니라 앞 단어 끝(`me2⌋`)에도 온다. 닫는 마커는 앞뒤 숫자를
#    모두 흡수해 'me2' 같은 오염을 막는다. 진짜 라벨(F2, A1, B2 vitamins 등)은
#    인접한 오버랩 마커가 없으므로 영향받지 않는다.
_OVERLAP_OPEN_RE = re.compile(r"⌈\d*")
_OVERLAP_CLOSE_RE = re.compile(r"\d*⌉\d*")
_OVERLAP_OPEN_LOW_RE = re.compile(r"⌊\d*")
_OVERLAP_CLOSE_LOW_RE = re.compile(r"\d*⌋\d*")

# 2. 스코프 마커: &{X=... ... &}X=X — 외곽 마커만 제거, 내부 발화는 보존
#    l=코드스위칭/L2, n=비언어음(SNAP/THUMP/MIC 등). 둘 다 마커 사이의 단어는
#    실제 발화이므로 보존한다. `&{n=THUMP}` 같은 self-closed 형도 open이 흡수한다.
_SCOPE_OPEN_RE = re.compile(r"&\{[a-z]=\S+\s*")
_SCOPE_CLOSE_RE = re.compile(r"\s*&\}[a-z]=\S+")

# 2b. 코멘트 의존 티어: [% laugh] 등 — 통째 제거 (발화 외 주석)
_COMMENT_TIER_RE = re.compile(r"\[%[^\]]*\]")

# 3. 비언어 어노테이션: &=tsk, &=laugh, &=in, &=ex 등 (&=word 패턴)
_NONVERBAL_RE = re.compile(r"&=\S+")

# 3b. Unintelligible 음성: 단독 대문자 X 런 (X, XX, XXX...). 알아들을 수 없는 음.
#     X-rays·XSes·X-(false start)는 글자/하이픈/아포스트로피가 붙어 자동 회피된다.
_UNINTELLIGIBLE_RE = re.compile(r"(?<![\w'-])X+(?![\w'-])")

# 3c. 밑줄 결합 발화: uh_you, U_S, D_I 등 — `_`를 공백으로 (두 단어 복원)
_UNDERSCORE_JOIN_RE = re.compile(r"(?<=\w)_(?=\w)")

# 4. 성문 폐쇄음 표기: 단어 시작의 ʔ (예: ʔuh → uh, youʔ → you)
_GLOTTAL_RE = re.compile(r"ʔ")

# 4b. Vowel lengthening: 알파벳 직후의 ":" (CHAT에서 모음 늘림 표기)
# 예: "I:" → "I", "u:m" → "um", "perc:e:nt" → "percent"
# overlap marker가 vowel lengthening 사이에 끼면 1단계 제거 후 "Yeah::" 같은
# 연속 colon이 생기므로 `:+`로 연속 colon을 한 번에 잡는다.
_VOWEL_LENGTH_RE = re.compile(r"(?<=[a-zA-Z]):+")

# 5. 발화 중단: +/.  (다른 +/ 형식 변형이 있으면 추가)
_TRAILOFF_INTERRUPT_RE = re.compile(r"\+/\.")

# 6. 말끝 흐림: +...
_TRAILOFF_RE = re.compile(r"\+\.\.\.")

# 7. 포즈 마커: (.) 와 (..) — 정확히 매치 (다른 괄호 표기와 혼동 방지)
#    먼저 (..) 매치한 후 (.) 매치 (긴 것 먼저)
_PAUSE_LONG_RE = re.compile(r"\(\.\.\)")
_PAUSE_SHORT_RE = re.compile(r"\(\.\)")

# 8. 다중 공백
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """단일 발화 텍스트를 정규화한다.

    처리 순서는 마커 간 의존성을 고려한 안전 순서이다 (모듈 docstring 참조).
    """
    # 1. 오버랩 마커 제거
    text = _OVERLAP_OPEN_RE.sub("", text)
    text = _OVERLAP_CLOSE_RE.sub("", text)
    text = _OVERLAP_OPEN_LOW_RE.sub("", text)
    text = _OVERLAP_CLOSE_LOW_RE.sub("", text)

    # 2. 스코프 마커 외곽 제거 (내부 발화 보존). XX 제거(3b)보다 먼저 —
    #    &{n=MIC XX ...} 처럼 스코프 안의 XX를 먼저 노출시킨 뒤 3b가 처리한다.
    text = _SCOPE_OPEN_RE.sub("", text)
    text = _SCOPE_CLOSE_RE.sub("", text)

    # 2b. 코멘트 의존 티어 제거
    text = _COMMENT_TIER_RE.sub("", text)

    # 3. 비언어 어노테이션 제거
    text = _NONVERBAL_RE.sub("", text)

    # 4. 성문 폐쇄음 표기 정규화. 3b(X 제거)보다 먼저 — "ʔX"의 ʔ를 떼야 X가
    #    단독 토큰으로 노출되어 3b가 unintelligible로 인식한다.
    text = _GLOTTAL_RE.sub("", text)

    # 3b. Unintelligible 대문자 X 런 제거
    text = _UNINTELLIGIBLE_RE.sub("", text)

    # 3c. 밑줄 결합 발화 → 공백
    text = _UNDERSCORE_JOIN_RE.sub(" ", text)

    # 4b. Vowel lengthening colon 제거
    text = _VOWEL_LENGTH_RE.sub("", text)

    # 5. 발화 중단 → 자연 종결
    text = _TRAILOFF_INTERRUPT_RE.sub(".", text)

    # 6. 말끝 흐림 → ...
    text = _TRAILOFF_RE.sub("...", text)

    # 7. 포즈 마커 토큰화 (긴 것부터)
    text = _PAUSE_LONG_RE.sub("<pause:long>", text)
    text = _PAUSE_SHORT_RE.sub("<pause:short>", text)

    # 8. 다중 공백 정리
    text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


def clean(utterances: list[Utterance]) -> list[Utterance]:
    """Utterance 리스트의 text 필드를 정규화한 새 리스트를 반환한다.

    Utterance는 frozen이므로 dataclasses.replace로 새 인스턴스를 생성한다.
    """
    return [replace(u, text=clean_text(u.text)) for u in utterances]
