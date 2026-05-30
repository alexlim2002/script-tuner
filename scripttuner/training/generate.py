"""학습된 LoRA 어댑터로 추론 — formatted split의 `input`에 대해 예측 생성.

현재 seq2seq(T5Gemma2) 경로 구현. chat(Gemma4) 경로는 추후.
출력: predictions.jsonl — 한 줄에 {pair_id, style, speaker, input, reference, prediction}.
무거운 ML 스택은 함수 내부에서 lazy import.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripttuner.preprocessing.text_normalize import normalize_punctuation
from scripttuner.training.registry import get_model_spec, require_hf_id


def run_generate(
    *,
    model_key: str,
    adapter_dir: str | Path,
    formatted_dir: Path,
    output_path: Path,
    split: str = "test",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    max_seq_length: int = 2048,
    limit: int | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, Any]:
    """어댑터를 base 위에 올려 split을 추론하고 predictions.jsonl을 쓴다."""

    spec = get_model_spec(model_key)
    if spec.format_kind != "seq2seq":
        raise NotImplementedError(
            f"generate currently supports seq2seq models only; {model_key!r} is "
            f"{spec.format_kind!r} (chat generation not implemented yet)."
        )
    hf_id = require_hf_id(model_key)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    split_path = formatted_dir / f"{split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"formatted split not found: {split_path}")
    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        rows = rows[:limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForSeq2SeqLM.from_pretrained(hf_id, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    model.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            inputs = tokenizer(
                [r["input"] for r in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            # 추론 출력 구두점 정규화 (학습 target과 동일 규칙, cf. ADR-0012)
            predictions = [normalize_punctuation(p) for p in predictions]
            for row, prediction in zip(batch, predictions, strict=True):
                out_row = {
                    "pair_id": row.get("pair_id"),
                    "style": row.get("style"),
                    "speaker": row.get("speaker"),
                    "input": row["input"],
                    "reference": row.get("target"),
                    "prediction": prediction,
                }
                f.write(json.dumps(out_row, ensure_ascii=False))
                f.write("\n")
                n += 1

    return {
        "model_key": model_key,
        "adapter_dir": str(adapter_dir),
        "split": split,
        "n": n,
        "output_path": str(output_path),
    }
