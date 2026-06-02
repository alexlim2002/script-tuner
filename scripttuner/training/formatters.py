"""Model-family-specific fine-tuning formatters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripttuner.persistence.jsonl import read_jsonl
from scripttuner.preprocessing.ir import Pair
from scripttuner.training.registry import FormatKind, model_format_kind
from scripttuner.training.style import STYLE_SPECS, get_style_spec


def format_split_folder(
    *,
    splits_dir: Path,
    output_dir: Path,
    model_key: str,
) -> dict[str, Any]:
    """Format train/validation/test split files for one target model family."""

    kind = model_format_kind(model_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    styles_seen: set[str] = set()

    for split_name in ("train", "validation", "test"):
        pairs = read_jsonl(splits_dir / f"{split_name}.jsonl", Pair)
        rows = [_format_pair(pair, kind=kind) for pair in pairs]
        styles_seen.update(pair.style for pair in pairs)
        counts[split_name] = _write_dict_jsonl(output_dir / f"{split_name}.jsonl", rows)

    manifest: dict[str, Any] = {
        "stage": "finetune_format",
        "model_key": model_key,
        "format": kind,
        "source_splits": str(splits_dir),
        "counts": counts,
        "style_tokens": {
            label: spec.control_token for label, spec in sorted(STYLE_SPECS.items())
        },
        "styles_present": sorted(styles_seen),
        "semi_formal_status": (
            "Reserved for future external corpus or teacher-LLM generated data; "
            "current SBCSAE pairs are expected to be casual unless additional "
            "semi_formal rows are added."
        ),
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _format_pair(pair: Pair, *, kind: FormatKind) -> dict[str, Any]:
    spec = get_style_spec(pair.style)
    input_text = _build_input(pair.formal_text, spec.control_token, spec.instruction)
    base_meta = {
        "pair_id": pair.pair_id,
        "source": pair.source,
        "style": pair.style,
        "speaker": pair.speaker,
        "monologue_id": pair.monologue_id,
        "style_token": spec.control_token,
        "source_metadata": pair.metadata,
    }
    if kind == "chat":
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": pair.spoken_text},
            ],
            **base_meta,
        }
    return {
        "input": input_text,
        "target": pair.spoken_text,
        **base_meta,
    }


# 입력 프롬프트 템플릿의 본문 경계 마커 — 빌드/역파싱이 공유한다.
_INPUT_MARKER = "Input:\n"
_OUTPUT_MARKER = "\n\nOutput:"


def _build_input(formal_text: str, control_token: str, instruction: str) -> str:
    return (
        f"{control_token}\n"
        f"{instruction}\n\n"
        f"{_INPUT_MARKER}"
        f"{formal_text}"
        f"{_OUTPUT_MARKER}"
    )


def extract_formal_text(wrapped_input: str) -> str | None:
    """`_build_input`로 감싼 프롬프트에서 순수 formal_text 본문을 복원한다.

    구어성 델타(ADR-0014)의 'before'는 instruction/control token 스캐폴딩이 아니라
    변환 대상 본문이어야 한다. 마커를 못 찾으면 None(호출자가 폴백 판단).
    """
    start = wrapped_input.find(_INPUT_MARKER)
    if start == -1:
        return None
    body_start = start + len(_INPUT_MARKER)
    end = wrapped_input.find(_OUTPUT_MARKER, body_start)
    body = wrapped_input[body_start:end] if end != -1 else wrapped_input[body_start:]
    return body.strip()


def _write_dict_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    return len(rows)
