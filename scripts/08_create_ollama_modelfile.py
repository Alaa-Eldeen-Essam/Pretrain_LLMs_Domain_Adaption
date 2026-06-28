#!/usr/bin/env python3
"""Create an Ollama Modelfile for the exported GGUF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json
from llm_pipeline.paths import configured_file, load_training, project_path


def find_gguf(gguf_dir: Path, preferred_name: str) -> Path:
    """Select the GGUF file to reference from the Ollama Modelfile.

    Needed so the Modelfile can point to either the configured filename or the
    first available GGUF export. Used by phase 08.

    Inputs: GGUF directory and preferred filename.
    Outputs: selected GGUF path.
    """
    preferred = gguf_dir / preferred_name
    if preferred.exists():
        return preferred
    matches = sorted(gguf_dir.glob("*.gguf"))
    if not matches:
        raise FileNotFoundError(f"No GGUF file found in: {gguf_dir}")
    return matches[0]


def main() -> int:
    """Run the Ollama Modelfile creation CLI phase.

    Needed as phase 08 to prepare local Ollama import metadata for the exported
    GGUF model. Used after export.

    Inputs: command-line arguments and Ollama/model config.
    Outputs: exit code, Modelfile, and Modelfile report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing Modelfile.")
    args = parser.parse_args()

    cfg = load_training()
    model_cfg = cfg["model"]
    ollama_cfg = cfg["ollama"]
    gguf_dir = project_path(model_cfg["gguf_dir"])
    gguf_path = find_gguf(gguf_dir, model_cfg["gguf_filename"])
    modelfile = gguf_dir / "Modelfile"
    if modelfile.exists() and not args.force:
        raise FileExistsError(f"Modelfile exists. Re-run with --force to overwrite: {modelfile}")

    content = f"""FROM {gguf_path.name}
PARAMETER num_ctx {ollama_cfg["num_ctx"]}
PARAMETER temperature {ollama_cfg["temperature"]}
SYSTEM \"\"\"{ollama_cfg["system_prompt"]}\"\"\"
"""
    modelfile.write_text(content, encoding="utf-8")
    write_json(
        configured_file("modelfile_report"),
        {
            "timestamp_utc": now_utc(),
            "status": "modelfile_created",
            "model_name": ollama_cfg["model_name"],
            "gguf_path": str(gguf_path),
            "modelfile": str(modelfile),
        },
    )
    print(f"Wrote Modelfile: {modelfile}")
    print(f"Create Ollama model later with: ollama create {ollama_cfg['model_name']} -f {modelfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
