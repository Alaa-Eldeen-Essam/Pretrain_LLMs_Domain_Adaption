#!/usr/bin/env python3
"""Continue-pretrain Qwen2.5-3B-Instruct with Unsloth LoRA.

This phase assumes the base model and dataset already exist locally. It does
not download by design when --offline is used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, require_file, write_json
from llm_pipeline.paths import apply_cache_environment, configured_file, load_training, project_path


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest Hugging Face Trainer checkpoint in an output folder.

    Needed so training can resume automatically after interruption. Used by the
    training phase before calling `trainer.train`.

    Inputs: training output directory.
    Outputs: latest checkpoint path or None when no checkpoint exists.
    """
    if not output_dir.exists():
        return None
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def report_is_complete(report_path: Path) -> bool:
    """Return whether the training report marks a completed run.

    Needed to skip fully completed training by default while still allowing
    explicit retraining with `--restart`. Used at training startup.

    Inputs: training report path.
    Outputs: boolean completion flag.
    """
    if not report_path.exists():
        return False
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle).get("status") == "trained"
    except Exception:  # noqa: BLE001 - malformed report should not block resume.
        return False


def main() -> int:
    """Run the Unsloth continued-pretraining CLI phase.

    Needed as phase 06 to train a LoRA adapter from local dataset and model
    files. Used after model download and dataset creation.

    Inputs: command-line arguments, local base model, and train/validation
    JSONL files.
    Outputs: exit code, saved adapter/tokenizer files, and training report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Force Hugging Face/Transformers offline mode.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short-run override for validation.")
    parser.add_argument("--restart", action="store_true", help="Delete existing training output and start from the base model.")
    args = parser.parse_args()

    apply_cache_environment()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = load_training()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    train_path = require_file(configured_file("train_dataset"), "train dataset")
    validation_path = require_file(configured_file("validation_dataset"), "validation dataset")
    base_dir = project_path(model_cfg["local_base_dir"])
    adapter_dir = project_path(model_cfg["adapter_dir"])
    report_path = configured_file("train_report")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base model is missing. Run 05_download_model.py later: {base_dir}")
    if args.restart:
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        report_path.unlink(missing_ok=True)
    elif report_is_complete(report_path) and (adapter_dir / "adapter_config.json").exists():
        print(f"Training already completed, skipping. Use --restart to train again: {adapter_dir}")
        return 0
    resume_checkpoint = latest_checkpoint(adapter_dir)
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(base_dir),
        max_seq_length=int(train_cfg["max_seq_length"]),
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(train_cfg["lora_rank"]),
        target_modules=list(train_cfg["target_modules"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    data_files = {"train": str(train_path)}
    if validation_path.stat().st_size > 0:
        data_files["validation"] = str(validation_path)
    dataset = load_dataset("json", data_files=data_files)

    training_args = TrainingArguments(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        fp16=True,
        optim="adamw_8bit",
        report_to=[],
        max_steps=args.max_steps if args.max_steps is not None else -1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        dataset_text_field="text",
        max_seq_length=int(train_cfg["max_seq_length"]),
        args=training_args,
        packing=True,
    )
    if resume_checkpoint is not None:
        metrics = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        metrics = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "status": "trained",
            "base_dir": str(base_dir),
            "adapter_dir": str(adapter_dir),
            "metrics": metrics.metrics,
            "offline": args.offline,
        },
    )
    print(f"Saved LoRA adapter to: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
