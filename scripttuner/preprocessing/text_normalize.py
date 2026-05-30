"""코퍼스 무관 텍스트 정규화.

학습 target(② cleaner)과 추론 출력(generate 후처리)이 공유하는 구두점 정규화.
근거·범위(Tier A/B, 문법 스코프)는 ADR-0012 참조.
"""

from __future__ import annotations

import re

# 부호 앞 공백 제거: "word ." → "word."
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([.,?!;:])")

# 연속 온점 정리: ".." / "...." → "." (정확히 "..." 생략부호는 보존)
_DOT_RUN_RE = re.compile(r"\.{2,}")


def _collapse_dot_run(m: re.Match[str]) -> str:
    run = m.group(0)
    return "..." if run == "..." else "."


def normalize_punctuation(text: str) -> str:
    """전사 잔여 구두점 표기를 정규화한다 (Tier A, 기계적·결정론적).

    - 부호 앞 공백 제거 (`. , ? ! ; :`)
    - 연속 온점 정리 (`..`/`....` → `.`), 생략부호 `...`는 보존

    쉼표·물음표·느낌표의 *생성*(재구두점)은 범위 밖이다 (Tier B, ADR-0012).

    공백 제거를 먼저 한 뒤 온점을 정리하므로 ``"there . ."`` → ``"there."`` 처럼
    공백으로 갈라진 중복 종결도 한 번에 흡수된다.
    """
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _DOT_RUN_RE.sub(_collapse_dot_run, text)
    return text