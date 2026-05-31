"""AMI Meeting Corpus (NXT) parser.

회의 1건의 화자 채널들(``words/{stem}.{X}.words.xml``)을 ``segments`` 레이어
(``segments/{stem}.{X}.segments.xml``)가 정의한 발화 단위로 읽어, transcriber
시간 기준 단일 시간순 `Utterance` 스트림으로 인터리브한다 (cf. ADR-0009 패턴).
이렇게 SBCSAE/Switchboard와 동일한 화자 교대 스트림이 복원되어 ③ monologue.py가
변경 없이 재사용된다.

speaker는 ``corpusResources/meetings.xml``의 ``global_name``(참가자 식별자)으로
둔다 — 같은 화자가 여러 회의에 재등장하므로, speaker-aware split의 누수 방지에
agent 문자(A/B/C/D)보다 정확하다. agent/role/meeting type은 metadata에 보존한다.

마커는 raw로 보존한다 (마커 처리는 cleaner 담당):
  - ``vocalsound`` (laugh/cough 등) → ``[vocalsound:TYPE]``
  - ``disfmarker`` (비유창성 경계)   → ``[disfmarker]``
  - ``gap`` (미전사 구간)            → ``[gap]``
  - ``transformerror``               → ``[transformerror]``
구두점 토큰(``punc="true"``)은 앞 단어 토큰에 공백 없이 붙인다.

NXT segment는 stand-off 참조다: ``<nite:child href="...words.xml#id(X)..id(Y)"/>``
가 단어 id 범위(문서 순서 inclusive)를 가리킨다. 단일 토큰은 ``#id(X)``.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

from scripttuner.preprocessing.ir import Utterance

SOURCE_NAME = "AMI"
_NITE_ID = "{http://nite.sourceforge.net/}id"
_AGENTS = ("A", "B", "C", "D")


def _localtag(el: ET.Element) -> str:
    return el.tag.split("}")[-1]


def _nid(el: ET.Element) -> str:
    return el.attrib.get(_NITE_ID, "")


def enumerate_stems(corpus_dir: Path) -> list[str]:
    """Return sorted meeting stems (e.g. ['ES2002a', 'ES2002b'])."""
    words_dir = corpus_dir / "words"
    stems = {p.name.split(".")[0] for p in words_dir.glob("*.words.xml")}
    return sorted(stems)


@functools.lru_cache(maxsize=8)
def _load_speakers(meetings_xml: str) -> dict[tuple[str, str], dict[str, str]]:
    """(meeting, agent) -> {global_name, role, meeting_type}. Cached by path."""
    root = ET.parse(meetings_xml).getroot()
    out: dict[tuple[str, str], dict[str, str]] = {}
    for meeting in root:
        if _localtag(meeting) != "meeting":
            continue
        obs = meeting.attrib.get("observation", "")
        mtype = meeting.attrib.get("type", "")
        for sp in meeting:
            if _localtag(sp) != "speaker":
                continue
            agent = sp.attrib.get("nxt_agent", "")
            out[(obs, agent)] = {
                "global_name": sp.attrib.get("global_name", ""),
                "role": sp.attrib.get("role", ""),
                "meeting_type": mtype,
            }
    return out


def _channel_order(words_path: Path) -> tuple[list[str], dict[str, ET.Element]]:
    """Read a words file in document order → (id list, id→element)."""
    root = ET.parse(words_path).getroot()
    order: list[str] = []
    by_id: dict[str, ET.Element] = {}
    for el in root:
        wid = _nid(el)
        if not wid:
            continue
        order.append(wid)
        by_id[wid] = el
    return order, by_id


def _id_inside(token: str) -> str:
    """'id(ES2002a.A.words0)' -> 'ES2002a.A.words0'."""
    return token[token.index("(") + 1 : token.rindex(")")]


def _parse_href(href: str) -> tuple[str, str]:
    """'x.words.xml#id(A)..id(B)' -> ('A', 'B'); '...#id(A)' -> ('A', 'A')."""
    frag = href.split("#", 1)[1]
    if ".." in frag:
        a, b = frag.split("..", 1)
        return _id_inside(a), _id_inside(b)
    one = _id_inside(frag)
    return one, one


def _segment_text(
    order: list[str], by_id: dict[str, ET.Element], start_id: str, end_id: str
) -> str:
    """Build raw text for the inclusive word-id range (markers preserved)."""
    try:
        i, j = order.index(start_id), order.index(end_id)
    except ValueError:
        return ""
    if j < i:
        i, j = j, i
    parts: list[str] = []
    for wid in order[i : j + 1]:
        el = by_id[wid]
        tag = _localtag(el)
        if tag == "w":
            txt = (el.text or "").strip()
            if not txt:
                continue
            if el.attrib.get("punc") == "true" and parts and not parts[-1].startswith("["):
                parts[-1] = parts[-1] + txt
            else:
                parts.append(txt)
        elif tag == "vocalsound":
            parts.append(f"[vocalsound:{el.attrib.get('type', '')}]")
        elif tag == "disfmarker":
            parts.append("[disfmarker]")
        elif tag == "gap":
            parts.append("[gap]")
        elif tag == "transformerror":
            parts.append("[transformerror]")
    return " ".join(parts)


def _ms(value: str | None) -> int | None:
    return round(float(value) * 1000) if value else None


def parse_stem(corpus_dir: Path, stem: str) -> list[Utterance]:
    """Parse one meeting's channels into a time-ordered Utterance stream.

    Markers are preserved verbatim in ``text``; timing comes from each segment's
    ``transcriber_start`` / ``transcriber_end``. Channels are merged with a
    stable sort on ``t_start_ms``.

    Raises FileNotFoundError if no channel has both words and segments files.
    """
    words_dir = corpus_dir / "words"
    segs_dir = corpus_dir / "segments"
    meetings_xml = corpus_dir / "corpusResources" / "meetings.xml"
    speakers = _load_speakers(str(meetings_xml)) if meetings_xml.exists() else {}

    utterances: list[Utterance] = []
    found = False
    for agent in _AGENTS:
        words_path = words_dir / f"{stem}.{agent}.words.xml"
        segs_path = segs_dir / f"{stem}.{agent}.segments.xml"
        if not words_path.exists() or not segs_path.exists():
            continue
        found = True
        order, by_id = _channel_order(words_path)
        sp = speakers.get((stem, agent), {})
        speaker = sp.get("global_name") or f"{stem}.{agent}"
        seg_root = ET.parse(segs_path).getroot()
        for seg in seg_root:
            if _localtag(seg) != "segment":
                continue
            seg_id = _nid(seg)
            child = next((c for c in seg if _localtag(c) == "child"), None)
            if child is None:
                continue
            start_id, end_id = _parse_href(child.attrib.get("href", ""))
            text = _segment_text(order, by_id, start_id, end_id)
            if not text:
                continue
            utterances.append(
                Utterance(
                    source=SOURCE_NAME,
                    utterance_id=f"{stem}#{seg_id}",
                    speaker=speaker,
                    text=text,
                    t_start_ms=_ms(seg.attrib.get("transcriber_start")),
                    t_end_ms=_ms(seg.attrib.get("transcriber_end")),
                    metadata={
                        "agent": agent,
                        "role": sp.get("role", ""),
                        "meeting_type": sp.get("meeting_type", ""),
                        "segment_id": seg_id,
                    },
                )
            )
    if not found:
        raise FileNotFoundError(
            f"no AMI words/segments for stem {stem!r} under {corpus_dir}"
        )
    utterances.sort(key=lambda u: (u.t_start_ms if u.t_start_ms is not None else 0))
    return utterances
