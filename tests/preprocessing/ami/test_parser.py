"""AMI NXT parser tests (synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripttuner.preprocessing.ami import parser

_MEETINGS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="meet.00">
  <meeting nite:id="m1" observation="M1" type="scenario">
    <speaker nite:id="M1_1" nxt_agent="A" global_name="P001" role="PM"/>
    <speaker nite:id="M1_2" nxt_agent="B" global_name="P002" role="ME"/>
  </meeting>
</nite:root>
"""

_WORDS_A = """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="M1.A.words">
  <w nite:id="M1.A.words0" starttime="1.0" endtime="1.2">Hello</w>
  <w nite:id="M1.A.words1" starttime="1.2" endtime="1.2" punc="true">,</w>
  <w nite:id="M1.A.words2" starttime="1.2" endtime="1.6">everyone</w>
  <vocalsound nite:id="M1.A.words3" starttime="1.6" endtime="1.7" type="laugh"/>
  <w nite:id="M1.A.words4" starttime="1.7" endtime="2.0">today</w>
  <disfmarker nite:id="M1.A.words5" starttime="2.0" endtime="2.0"/>
  <w nite:id="M1.A.words6" starttime="2.0" endtime="2.3">right</w>
</nite:root>
"""

_SEGS_A = """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="M1.A.segs">
  <segment nite:id="M1.sync.1" channel="0" transcriber_start="1.0" transcriber_end="2.3">
    <nite:child href="M1.A.words.xml#id(M1.A.words0)..id(M1.A.words6)"/>
  </segment>
</nite:root>
"""

_WORDS_B = """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="M1.B.words">
  <w nite:id="M1.B.words0" starttime="3.0" endtime="3.4">Welcome</w>
</nite:root>
"""

_SEGS_B = """<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="M1.B.segs">
  <segment nite:id="M1.sync.9" channel="1" transcriber_start="3.0" transcriber_end="3.4">
    <nite:child href="M1.B.words.xml#id(M1.B.words0)"/>
  </segment>
</nite:root>
"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "words").mkdir()
    (tmp_path / "segments").mkdir()
    (tmp_path / "corpusResources").mkdir()
    (tmp_path / "corpusResources" / "meetings.xml").write_text(_MEETINGS_XML, encoding="utf-8")
    (tmp_path / "words" / "M1.A.words.xml").write_text(_WORDS_A, encoding="utf-8")
    (tmp_path / "words" / "M1.B.words.xml").write_text(_WORDS_B, encoding="utf-8")
    (tmp_path / "segments" / "M1.A.segments.xml").write_text(_SEGS_A, encoding="utf-8")
    (tmp_path / "segments" / "M1.B.segments.xml").write_text(_SEGS_B, encoding="utf-8")
    return tmp_path


def test_enumerate_stems(corpus_dir: Path) -> None:
    assert parser.enumerate_stems(corpus_dir) == ["M1"]


def test_parse_stem_interleaves_by_time(corpus_dir: Path) -> None:
    utts = parser.parse_stem(corpus_dir, "M1")
    assert [u.speaker for u in utts] == ["P001", "P002"]  # sorted by t_start
    assert [u.t_start_ms for u in utts] == [1000, 3000]


def test_parse_stem_preserves_markers_raw(corpus_dir: Path) -> None:
    a = parser.parse_stem(corpus_dir, "M1")[0]
    # punc attached to prior word; vocalsound/disfmarker preserved raw
    assert a.text == "Hello, everyone [vocalsound:laugh] today [disfmarker] right"
    assert a.utterance_id == "M1#M1.sync.1"
    assert a.metadata == {
        "agent": "A",
        "role": "PM",
        "meeting_type": "scenario",
        "segment_id": "M1.sync.1",
    }


def test_parse_stem_single_token_href(corpus_dir: Path) -> None:
    b = parser.parse_stem(corpus_dir, "M1")[1]
    assert b.text == "Welcome"
    assert b.speaker == "P002"


def test_parse_stem_missing_raises(corpus_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse_stem(corpus_dir, "NOPE")


def test_speaker_falls_back_when_no_mapping(tmp_path: Path) -> None:
    (tmp_path / "words").mkdir()
    (tmp_path / "segments").mkdir()
    (tmp_path / "words" / "M2.A.words.xml").write_text(
        _WORDS_A.replace("M1", "M2"), encoding="utf-8"
    )
    (tmp_path / "segments" / "M2.A.segments.xml").write_text(
        _SEGS_A.replace("M1", "M2"), encoding="utf-8"
    )
    utt = parser.parse_stem(tmp_path, "M2")[0]
    assert utt.speaker == "M2.A"  # no meetings.xml → fallback to stem.agent
