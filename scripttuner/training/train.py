"""파인튜닝 — format_kind로 백엔드 분기.

- "chat"    → Unsloth FastLanguageModel + TRL SFTTrainer (decoder-only, 4bit QLoRA)
- "seq2seq" → transformers AutoModelForSeq2SeqLM + Seq2SeqTrainer + PEFT LoRA (encoder-decoder)

산출물: `runs/finetune/<run>/adapter` + `MANIFEST.json`. 무거운 ML 스택은 각 경로 내부에서
lazy import 한다 — 다른 CLI 명령이 이 모듈을 거쳐도 로드되지 않도록.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from scripttuner.training.registry import get_model_spec, require_hf_id

# decoder-only LoRA가 붙는 표준 projection 모듈.
_CHAT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def run_finetune(
    *,
    model_key: str,
    formatted_dir: Path,
    output_dir: Path,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    epochs: float = 1.0,
    max_steps: int | None = None,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    seed: int = 42,
    gradient_checkpointing: bool = True,
) -> dict[str, Any]:
    """formatted split을 학습하고 어댑터 + MANIFEST를 저장한다.

    `max_steps`가 주어지면 epochs 대신 step 수로 종료한다(스모크/빠른 실행용).
    `gradient_checkpointing`을 끄면 activation 재계산을 생략해 20~30% 빨라지나 VRAM을
    더 쓴다(결과 수치는 동일). VRAM 여유가 있을 때(예: T4 16GB)만 끈다.
    """

    spec = get_model_spec(model_key)
    hf_id = require_hf_id(model_key)
    train_path = formatted_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"formatted train data not found: {train_path}")
    val_path = formatted_dir / "validation.jsonl"

    kwargs: dict[str, Any] = {
        "hf_id": hf_id,
        "train_path": train_path,
        "val_path": val_path if val_path.exists() else None,
        "output_dir": output_dir,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if spec.format_kind == "chat":
        result = _train_chat(**kwargs)
    elif spec.format_kind == "seq2seq":
        result = _train_seq2seq(**kwargs)
    else:  # pragma: no cover - registry constrains format_kind
        raise ValueError(f"Unknown format_kind {spec.format_kind!r} for {model_key!r}.")

    manifest: dict[str, Any] = {
        "stage": "finetune_train",
        "model_key": model_key,
        "base_model": hf_id,
        "format": spec.format_kind,
        "backend": result["backend"],
        "formatted_dir": str(formatted_dir),
        "adapter_dir": result["adapter_dir"],
        "n_train": result["n_train"],
        "n_validation": result["n_validation"],
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
        "training": {
            "max_seq_length": max_seq_length,
            "epochs": None if max_steps else epochs,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
            "seed": seed,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "train_loss": result["train_loss"],
        "train_runtime_sec": result["train_runtime_sec"],
        "early_stopping": result.get("early_stopping", False),
        "best_eval_loss": result.get("best_metric"),
        "stopped_epoch": result.get("stopped_epoch"),
        "log_history_path": result.get("log_history_path"),
        "date": date.today().isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _train_chat(
    *,
    hf_id: str,
    train_path: Path,
    val_path: Path | None,
    output_dir: Path,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    epochs: float,
    max_steps: int | None,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    seed: int,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    """decoder-only chat 모델 QLoRA (Unsloth)."""

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from datasets import load_dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=_CHAT_LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
        random_state=seed,
    )

    data_files = {"train": str(train_path)}
    if val_path is not None:
        data_files["validation"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)

    def _to_text(example: dict[str, Any]) -> dict[str, str]:
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(_to_text)

    duration: dict[str, Any] = (
        {"max_steps": max_steps} if max_steps else {"num_train_epochs": epochs}
    )
    args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        seed=seed,
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
        **duration,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        processing_class=tokenizer,
    )
    train_result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics: dict[str, Any] = getattr(train_result, "metrics", {}) or {}
    return {
        "backend": "unsloth-qlora-4bit",
        "adapter_dir": str(adapter_dir),
        "n_train": ds["train"].num_rows,
        "n_validation": ds["validation"].num_rows if "validation" in ds else 0,
        "train_loss": metrics.get("train_loss"),
        "train_runtime_sec": metrics.get("train_runtime"),
    }


def _train_seq2seq(
    *,
    hf_id: str,
    train_path: Path,
    val_path: Path | None,
    output_dir: Path,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    epochs: float,
    max_steps: int | None,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    seed: int,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    """encoder-decoder 모델 LoRA (transformers Seq2SeqTrainer).

    formatted seq2seq JSONL의 `input`/`target`을 인코더 입력/디코더 타깃으로 토큰화한다.
    소형(270m/1b)은 bf16 LoRA로 8GB에 들어간다(4bit 양자화는 향후 4b 확장 시 추가).
    """

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, dtype=torch.bfloat16)
    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    # gradient checkpointing이 PEFT(동결 base + LoRA)에서 동작하려면 입력이 grad를 요구해야 한다.
    # checkpointing을 끄면 불필요하므로 켜진 경우에만 호출한다.
    if gradient_checkpointing:
        model.enable_input_require_grads()

    data_files = {"train": str(train_path)}
    if val_path is not None:
        data_files["validation"] = str(val_path)
    ds = load_dataset("json", data_files=data_files)

    eos_id = tokenizer.eos_token_id

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        model_inputs: dict[str, Any] = dict(
            tokenizer(batch["input"], max_length=max_seq_length, truncation=True)
        )
        # T5Gemma2 tokenizer는 text_target에 EOS를 자동 추가하지 않는다 — 라벨 끝에 EOS가
        # 없으면 모델이 종료 시점을 학습하지 못해 생성 시 tail이 degenerate한다.
        # max_length-1로 truncate한 뒤 EOS를 붙여 최종 시퀀스가 한도를 넘지 않게 한다.
        labels = tokenizer(
            text_target=batch["target"], max_length=max_seq_length - 1, truncation=True
        )
        labels["input_ids"] = [ids + [eos_id] for ids in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(_tokenize, batched=True, remove_columns=ds["train"].column_names)

    # model= is intentionally omitted: its decoder_input_ids precompute calls a model method
    # whose signature is incompatible with T5Gemma2. The model.forward shifts labels itself.
    collator = DataCollatorForSeq2Seq(tokenizer)
    duration: dict[str, Any] = (
        {"max_steps": max_steps} if max_steps else {"num_train_epochs": epochs}
    )
    # validation이 있으면 epoch마다 평가·체크포인트 저장하고 EarlyStopping(patience=2)을 건다.
    # load_best_model_at_end=True 로 평가 최저 loss 시점의 어댑터를 최종 저장.
    eval_enabled = val_path is not None
    eval_kwargs: dict[str, Any] = (
        {
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        if eval_enabled
        else {}
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=batch_size,
        # HF 기본 eval batch는 train과 무관하게 8 — 8GB GPU에서 즉시 OOM이라 train과 동일하게 둔다.
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        seed=seed,
        report_to="none",
        **eval_kwargs,
        **duration,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") if eval_enabled else None,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_enabled else [],
    )
    train_result = trainer.train()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # 학습 곡선 시각화를 위해 step별 train/eval loss 로그를 영구화.
    log_history_path = output_dir / "log_history.json"
    log_history_path.write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metrics: dict[str, Any] = getattr(train_result, "metrics", {}) or {}
    return {
        "backend": "transformers-seq2seq-lora-bf16",
        "adapter_dir": str(adapter_dir),
        "n_train": tokenized["train"].num_rows,
        "n_validation": tokenized["validation"].num_rows if "validation" in tokenized else 0,
        "train_loss": metrics.get("train_loss"),
        "train_runtime_sec": metrics.get("train_runtime"),
        "log_history_path": str(log_history_path),
        "early_stopping": eval_enabled,
        "best_metric": trainer.state.best_metric if eval_enabled else None,
        "stopped_epoch": trainer.state.epoch if eval_enabled else None,
    }
