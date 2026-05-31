"""Corpus adapter registry — the composition root for per-corpus pipelines.

ADR-0006이 예고한 어댑터 구조의 실체. 각 코퍼스는 자기 포맷을 공통 IR(`Utterance`)로
바꾸는 데 필요한 것(다운로드, stem 열거, 파싱, 클리닝, 백채널 사전)을 하나의
`Adapter`로 묶는다. CLI(③ Monologue 이후 단계 포함)는 이 레지스트리로 디스패치하므로,
새 코퍼스 추가 = 어댑터 한 건 등록이면 된다.

stem-centric 계약: 모든 어댑터는 `(corpus_dir, stem)`만으로 입력을 해석한다. CHAT은
`{stem}.cha` 한 파일, Switchboard는 `{stem}{A,B}-...trans.text` 두 파일을 인터리브
(cf. ADR-0009).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripttuner.data_sources import ami as ami_ds
from scripttuner.data_sources import sbcsae
from scripttuner.data_sources import switchboard as switchboard_ds
from scripttuner.preprocessing.ami import cleaner as ami_cleaner
from scripttuner.preprocessing.ami import parser as ami_parser
from scripttuner.preprocessing.chat import cleaner as chat_cleaner
from scripttuner.preprocessing.chat import parser as chat_parser
from scripttuner.preprocessing.ir import Utterance
from scripttuner.preprocessing.monologue import DEFAULT_BACKCHANNEL_WORDS
from scripttuner.preprocessing.switchboard import cleaner as swb_cleaner
from scripttuner.preprocessing.switchboard import parser as swb_parser


@dataclass(frozen=True)
class Adapter:
    """단일 코퍼스의 ①~② 어댑터 + 보조 설정 묶음."""

    source_name: str
    """IR `source` 값 및 산출물 디렉토리 이름 (e.g. 'SBCSAE', 'Switchboard')."""

    download: Callable[..., list[Path]]
    """다운로더. (dest_dir, *, force, fetcher, ...) -> 확보된 파일 경로."""

    enumerate_stems: Callable[[Path], list[str]]
    """corpus_dir 아래의 처리 단위(stem) 목록을 정렬해 반환."""

    parse_stem: Callable[[Path, str], list[Utterance]]
    """(corpus_dir, stem) -> Utterance 리스트 (마커 보존, 타임스탬프 분리)."""

    clean: Callable[[list[Utterance]], list[Utterance]]
    """Utterance 리스트 정규화 (마커 처리; 어댑터에 따라 빈 발화 drop 포함)."""

    backchannel_words: frozenset[str] = field(default=DEFAULT_BACKCHANNEL_WORDS)
    """③ Monologue 재조립에 쓰는 백채널 사전 (코퍼스별 주입, cf. ADR-0009)."""


def _chat_parse_stem(corpus_dir: Path, stem: str) -> list[Utterance]:
    _, utterances = chat_parser.parse(
        corpus_dir / f"{stem}.cha", source=sbcsae.SOURCE_NAME
    )
    return utterances


def _chat_enumerate_stems(corpus_dir: Path) -> list[str]:
    return sorted(p.stem for p in corpus_dir.glob("*.cha"))


# Switchboard backchannels. `monologue._word_tokens`가 하이픈을 분할하므로
# (`um-hum`->['um','hum'], `uh-huh`->['uh','huh']) 통짜가 아니라 분리 컴포넌트를 넣는다.
_SWITCHBOARD_BACKCHANNEL_WORDS: frozenset[str] = DEFAULT_BACKCHANNEL_WORDS | frozenset(
    {"um", "hum", "uh", "huh"}
)


REGISTRY: dict[str, Adapter] = {
    "ami": Adapter(
        source_name=ami_ds.SOURCE_NAME,
        download=ami_ds.download,
        enumerate_stems=ami_parser.enumerate_stems,
        parse_stem=ami_parser.parse_stem,
        clean=ami_cleaner.clean,
    ),
    "sbcsae": Adapter(
        source_name=sbcsae.SOURCE_NAME,
        download=sbcsae.download,
        enumerate_stems=_chat_enumerate_stems,
        parse_stem=_chat_parse_stem,
        clean=chat_cleaner.clean,
    ),
    "switchboard": Adapter(
        source_name=switchboard_ds.SOURCE_NAME,
        download=switchboard_ds.download,
        enumerate_stems=swb_parser.enumerate_stems,
        parse_stem=swb_parser.parse_conversation,
        clean=swb_cleaner.clean,
        backchannel_words=_SWITCHBOARD_BACKCHANNEL_WORDS,
    ),
}
