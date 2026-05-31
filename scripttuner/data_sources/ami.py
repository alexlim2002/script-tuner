"""AMI Meeting Corpus annotation downloader.

Source: University of Edinburgh — official manual annotations v1.6.2.
URL:    https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
License: CC BY 4.0 (relicensed 2017-04, cf. ADR-0013). The OpenSLR mirror ships
         v1.6.1 which bundles the *old* CC BY-NC-SA 2.5 licence text, so we pull
         the official v1.6.2 zip instead.

Only the annotation subtrees needed for the (spoken, formal) pair pipeline are
extracted: ``words/`` (per-speaker word streams), ``segments/`` (utterance
boundaries), and ``corpusResources/`` (speaker mapping). Gesture / movement /
summary layers are skipped.

Citation:
    Carletta, J. et al. (2006). The AMI Meeting Corpus: A Pre-Announcement.
    Machine Learning for Multimodal Interaction (MLMI).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

ZIP_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)
SOURCE_NAME = "AMI"
"""IR `source` field value 및 디스크 산출물 디렉토리 이름 (e.g. data/parsed/AMI/)."""

_KEEP_DIRS = ("words", "segments", "corpusResources")


def _default_fetcher(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        data: bytes = resp.read()
    return data


def download(
    dest_dir: Path,
    *,
    force: bool = False,
    fetcher: Callable[[str], bytes] = _default_fetcher,
) -> list[Path]:
    """Download AMI manual annotations (v1.6.2) into dest_dir.

    Extracts only the ``words/``, ``segments/``, and ``corpusResources/``
    subtrees. Returns a sorted list of ``words/*.words.xml`` file paths.

    If force is False and word files plus the speaker map already exist, skip
    the download. Raises RuntimeError if nothing usable was extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    words_dir = dest_dir / "words"
    meetings_xml = dest_dir / "corpusResources" / "meetings.xml"
    existing = sorted(words_dir.glob("*.words.xml")) if words_dir.exists() else []
    if not force and existing and meetings_xml.exists():
        return existing

    zip_bytes = fetcher(ZIP_URL)
    extracted = _extract(zip_bytes, dest_dir)
    if not extracted or not meetings_xml.exists():
        raise RuntimeError(
            f"AMI extraction produced no usable annotations under {dest_dir} "
            f"(words={len(extracted)}, meetings.xml={meetings_xml.exists()})"
        )
    return extracted


def _rel_under_keep(name: str) -> str | None:
    """Return the path relative to a kept top-level dir, or None to skip.

    Handles zips whether or not the kept dirs are nested under a root prefix:
    ``ami_public_manual_1.6.2/words/x.xml`` and ``words/x.xml`` both map to
    ``words/x.xml``.
    """
    parts = name.split("/")
    for i, p in enumerate(parts):
        if p in _KEEP_DIRS:
            return "/".join(parts[i:])
    return None


def _extract(zip_bytes: bytes, dest_dir: Path) -> list[Path]:
    word_files: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = _rel_under_keep(info.filename)
            if rel is None:
                continue
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            if rel.startswith("words/") and rel.endswith(".words.xml"):
                word_files.append(target)
    return sorted(word_files)
