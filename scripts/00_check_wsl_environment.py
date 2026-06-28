#!/usr/bin/env python3
"""Check the WSL/conda environment without downloading anything.

This phase is safe to run before internet access is available. It creates the
project folder tree, pins cache environment variables to the project root, and
reports missing local tools such as Tesseract, Poppler, conda, and CUDA.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json
from llm_pipeline.paths import (
    apply_cache_environment,
    cache_environment,
    configured_file,
    ensure_project_tree,
    is_wsl,
    project_root,
    source_root,
)


def command_version(command: str, *args: str) -> dict:
    """Check whether a command exists and capture a short version output.

    Needed to report local prerequisites without installing anything. Used by
    the environment check phase.

    Inputs: command name/path and version arguments.
    Outputs: dictionary with found flag, path, exit code, output, or error.
    """
    executable = shutil.which(command)
    result = {"command": command, "found": executable is not None, "path": executable}
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        result.update({"exit_code": completed.returncode, "output": output[:1000]})
    except Exception as exc:  # noqa: BLE001 - diagnostics must not crash the check.
        result.update({"error": str(exc)})
    return result


def tesseract_languages() -> dict:
    """Inspect installed Tesseract OCR languages.

    Needed to verify Arabic OCR support before processing PDFs. Used by the
    environment check phase.

    Inputs: none; reads the local `tesseract` executable if available.
    Outputs: dictionary with language list and Arabic availability flag.
    """
    executable = shutil.which("tesseract")
    if executable is None:
        return {"available": False, "languages": [], "has_arabic": False}
    try:
        completed = subprocess.run(
            [executable, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        languages = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and not line.lower().startswith("list of")
        ]
        return {
            "available": True,
            "languages": languages,
            "has_arabic": "ara" in languages,
            "exit_code": completed.returncode,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "languages": [], "has_arabic": False, "error": str(exc)}


def main() -> int:
    """Run the environment-check CLI phase.

    Needed as phase 00 to verify WSL, source paths, cache paths, OCR tools, and
    CUDA visibility before expensive work. Used directly from the terminal.

    Inputs: command-line arguments.
    Outputs: process exit code and `environment_report.json`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Return non-zero if WSL or required OCR tools are missing.")
    args = parser.parse_args()

    ensure_project_tree()
    apply_cache_environment()

    checks = {
        "timestamp_utc": now_utc(),
        "project_root": str(project_root()),
        "source_root": str(source_root()),
        "source_root_exists": source_root().exists(),
        "is_wsl": is_wsl(),
        "cache_environment": cache_environment(),
        "commands": {
            "conda": command_version("conda", "--version"),
            "python": command_version(sys.executable, "--version"),
            "tesseract": command_version("tesseract", "--version"),
            "pdftoppm": command_version("pdftoppm", "-v"),
            "pdfinfo": command_version("pdfinfo", "-v"),
            "nvidia_smi": command_version("nvidia-smi"),
        },
        "tesseract_languages": tesseract_languages(),
    }

    required_ok = (
        checks["is_wsl"]
        and checks["source_root_exists"]
        and checks["commands"]["tesseract"]["found"]
        and checks["commands"]["pdftoppm"]["found"]
        and checks["tesseract_languages"]["has_arabic"]
    )
    checks["ready_for_ocr"] = bool(required_ok)

    report_path = configured_file("environment_report")
    write_json(report_path, checks)
    print(f"Wrote environment report: {report_path}")
    if args.strict and not required_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
