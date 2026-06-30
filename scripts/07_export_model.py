#!/usr/bin/env python3
"""Export the trained adapter to merged and GGUF artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json
from llm_pipeline.paths import apply_cache_environment, configured_file, load_training, project_path


def main() -> int:
    """Run the model-export CLI phase.

    Needed as phase 07 to convert the trained adapter into merged and optional
    GGUF artifacts. Used after training completes.

    Inputs: command-line arguments and local adapter directory.
    Outputs: exit code, merged model files, optional GGUF files, and export
    report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Force Hugging Face/Transformers offline mode.")
    parser.add_argument("--skip-gguf", action="store_true", help="Only create the merged HF model.")
    parser.add_argument("--restart", action="store_true", help="Delete existing export outputs and export again.")
    args = parser.parse_args()

    apply_cache_environment()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = load_training()["model"]
    adapter_dir = project_path(cfg["adapter_dir"])
    merged_dir = project_path(cfg["merged_dir"])
    gguf_dir = project_path(cfg["gguf_dir"])
    report_path = configured_file("export_report")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory is missing: {adapter_dir}")
    gguf_exists = gguf_dir.exists() and any(gguf_dir.glob("*.gguf"))
    merged_exists = merged_dir.exists() and any(merged_dir.iterdir())
    if merged_exists and (args.skip_gguf or gguf_exists) and report_path.exists() and not args.restart:
        print(f"Export already exists, skipping. Use --restart to export again: {merged_dir}")
        return 0
    if args.restart:
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        if not args.skip_gguf and gguf_dir.exists():
            shutil.rmtree(gguf_dir)
        report_path.unlink(missing_ok=True)

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    gguf_written = False
    if not args.skip_gguf:
        gguf_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=str(cfg["gguf_quantization"]))
        gguf_written = True

    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "status": "exported",
            "adapter_dir": str(adapter_dir),
            "merged_dir": str(merged_dir),
            "gguf_dir": str(gguf_dir),
            "gguf_written": gguf_written,
        },
    )
    print(f"Saved merged model to: {merged_dir}")
    if gguf_written:
        print(f"Saved GGUF model files to: {gguf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
