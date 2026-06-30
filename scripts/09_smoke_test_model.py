#!/usr/bin/env python3
"""Offline-safe smoke checks for exported artifacts, with optional Ollama run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json
from llm_pipeline.paths import configured_file, load_training, project_path


def main() -> int:
    """Run the exported-artifact smoke-test CLI phase.

    Needed as phase 09 to verify that GGUF and Modelfile artifacts exist, and
    optionally run a local Ollama prompt. Used after Modelfile creation.

    Inputs: command-line arguments and export/Ollama config.
    Outputs: exit code and smoke-test report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ollama", action="store_true", help="Actually run a local Ollama prompt.")
    parser.add_argument("--restart", action="store_true", help="Re-run smoke checks even if the report already exists.")
    parser.add_argument("--prompt", default="اشرح بإيجاز معنى نظام التشويش.", help="Prompt for optional Ollama test.")
    args = parser.parse_args()

    cfg = load_training()
    model_cfg = cfg["model"]
    ollama_cfg = cfg["ollama"]
    gguf_dir = project_path(model_cfg["gguf_dir"])
    modelfile = gguf_dir / "Modelfile"
    ggufs = sorted(gguf_dir.glob("*.gguf")) if gguf_dir.exists() else []
    report_path = configured_file("smoke_report")
    if report_path.exists() and not args.restart:
        print(f"Smoke report already exists, skipping. Use --restart to run again: {report_path}")
        return 0
    if args.restart:
        report_path.unlink(missing_ok=True)

    report = {
        "timestamp_utc": now_utc(),
        "gguf_dir": str(gguf_dir),
        "gguf_count": len(ggufs),
        "gguf_files": [str(path) for path in ggufs],
        "modelfile_exists": modelfile.exists(),
        "ollama_run": None,
    }

    if args.run_ollama:
        completed = subprocess.run(
            ["ollama", "run", ollama_cfg["model_name"], args.prompt],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report["ollama_run"] = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    write_json(report_path, report)
    print(f"Wrote smoke report: {report_path}")
    if not ggufs or not modelfile.exists():
        print("Smoke check incomplete: GGUF and Modelfile must exist before runtime testing.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
