#!/usr/bin/env python3
"""Download the base Qwen model into the project-local model/cache folders.

This is the only phase intended to use the internet. It does nothing unless
--allow-download is explicitly provided, so it is safe to inspect/run offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json
from llm_pipeline.paths import apply_cache_environment, configured_file, load_training, project_path


def main() -> int:
    """Run the guarded model-download CLI phase.

    Needed as phase 05 to place the base model under the project-local models
    folder while preventing accidental internet use. Used only when downloads
    are explicitly allowed.

    Inputs: command-line arguments and model config.
    Outputs: exit code and download report; model files only with
    `--allow-download`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true", help="Required before contacting Hugging Face.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision pin.")
    args = parser.parse_args()

    apply_cache_environment()
    cfg = load_training()["model"]
    model_id = cfg["hf_model_id"]
    local_dir = project_path(cfg["local_base_dir"])
    report_path = configured_file("download_report")

    if not args.allow_download:
        write_json(
            report_path,
            {
                "timestamp_utc": now_utc(),
                "status": "skipped",
                "reason": "Pass --allow-download to download the model.",
                "model_id": model_id,
                "local_dir": str(local_dir),
            },
        )
        print("Download skipped. Re-run with --allow-download when internet credits are available.")
        return 0

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=model_id,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "status": "downloaded",
            "model_id": model_id,
            "revision": args.revision,
            "local_dir": str(local_dir),
            "snapshot_path": downloaded,
        },
    )
    print(f"Downloaded model to: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
