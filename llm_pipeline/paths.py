from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    """Return the project root derived from this package location.

    Needed so scripts can be launched from any working directory. Used by all
    shared path helpers.

    Inputs: none.
    Outputs: absolute project root path.
    """
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a UTF-8 YAML configuration file.

    Needed to centralize parsing of path and training config files. Used by
    `load_paths` and `load_training`.

    Inputs: YAML file path.
    Outputs: dictionary configuration, or an empty dictionary for empty files.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_paths() -> Dict[str, Any]:
    """Load the path configuration.

    Needed so every phase reads the same project, source, output, and report
    locations. Used across all scripts and utilities.

    Inputs: none.
    Outputs: dictionary from `configs/paths.yaml`.
    """
    return load_yaml(repo_root() / "configs" / "paths.yaml")


def load_training() -> Dict[str, Any]:
    """Load OCR, dataset, model, training, and Ollama settings.

    Needed so phase behavior is configured rather than hardcoded. Used by OCR,
    dataset, download, training, export, and smoke-test scripts.

    Inputs: none.
    Outputs: dictionary from `configs/training.yaml`.
    """
    return load_yaml(repo_root() / "configs" / "training.yaml")


def is_wsl() -> bool:
    """Detect whether the current Python process is running under WSL.

    Needed to choose Linux `/mnt/d/...` paths versus Windows `D:/...` paths.
    Used by source path resolution and environment checks.

    Inputs: none.
    Outputs: `True` when the platform release indicates WSL.
    """
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def project_root() -> Path:
    """Return the active project root.

    Needed to keep all generated data, caches, models, and reports under the
    project folder. Used by all path-construction helpers.

    Inputs: optional `PROJECT_ROOT` environment variable.
    Outputs: resolved project root path.
    """
    override = os.environ.get("PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root()


def source_root() -> Path:
    """Return the configured source corpus folder.

    Needed so inventory and OCR know where the original references live. Used
    by environment checks and inventory creation.

    Inputs: WSL/Windows path settings from `configs/paths.yaml`.
    Outputs: source corpus path, with an Arabic-path fallback if terminal
    encoding corrupted the YAML literal.
    """
    cfg = load_paths()
    key = "source_root_wsl" if is_wsl() else "source_root_windows"
    configured = Path(cfg[key]).expanduser()
    # Some Windows consoles corrupt Arabic path literals while writing YAML.
    # Keep an ASCII-safe fallback for the known corpus folder name.
    if "Ù" in str(configured) or not configured.exists():
        arabic_refs = "\u0645\u0631\u0627\u062c\u0639"
        if is_wsl():
            return Path("/mnt/d/Behoos_AI/AI Projects/mini_rag/test_mini_rag") / arabic_refs
        return Path("D:/Behoos_AI/AI Projects/mini_rag/test_mini_rag") / arabic_refs
    return configured


def project_path(relative: str) -> Path:
    """Resolve a project-relative path.

    Needed to build output locations without leaking files to `C:` or other
    directories. Used by configuration helpers and model scripts.

    Inputs: relative path string.
    Outputs: absolute path under the project root.
    """
    return project_root() / relative


def configured_file(name: str) -> Path:
    """Resolve a named file from `configs/paths.yaml`.

    Needed so scripts share stable report and artifact filenames. Used by all
    phase scripts.

    Inputs: key from the `files` section.
    Outputs: absolute project-local file path.
    """
    cfg = load_paths()
    return project_path(cfg["files"][name])


def configured_folder(name: str) -> Path:
    """Resolve a named folder from `configs/paths.yaml`.

    Needed so scripts can create and reference standard output folders. Used by
    setup and future extension code.

    Inputs: key from the `folders` section.
    Outputs: absolute project-local folder path.
    """
    cfg = load_paths()
    return project_path(cfg["folders"][name])


def ensure_project_tree() -> None:
    """Create configured project output folders if missing.

    Needed so phases can write outputs without manual directory setup. Used by
    environment and inventory scripts.

    Inputs: folder settings from `configs/paths.yaml`.
    Outputs: none; creates directories under the project root.
    """
    cfg = load_paths()
    for relative in cfg.get("folders", {}).values():
        project_path(relative).mkdir(parents=True, exist_ok=True)
    for extra in ("scripts", "configs", "llm_pipeline"):
        project_path(extra).mkdir(parents=True, exist_ok=True)


def assert_inside_project(path: Path) -> Path:
    """Validate that a write target is inside the project root.

    Needed to enforce the requirement that generated data, models, and caches
    do not go to `C:` or unrelated locations. Used by JSON/JSONL writers.

    Inputs: path to validate.
    Outputs: resolved path when valid.
    """
    root = project_root().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside project root: {resolved}") from exc
    return resolved


def cache_environment() -> Dict[str, str]:
    """Build project-local cache environment variables.

    Needed so Hugging Face, Transformers, Datasets, Torch, and temp files stay
    under the project folder. Used by environment checks and model phases.

    Inputs: current project root.
    Outputs: dictionary of environment variable names and paths.
    """
    root = project_root()
    return {
        "PROJECT_ROOT": str(root),
        "HF_HOME": str(root / "cache" / "hf_home"),
        "HF_HUB_CACHE": str(root / "cache" / "hf_home" / "hub"),
        "TRANSFORMERS_CACHE": str(root / "cache" / "transformers"),
        "HF_DATASETS_CACHE": str(root / "cache" / "datasets"),
        "TORCH_HOME": str(root / "cache" / "torch"),
        "XDG_CACHE_HOME": str(root / "cache" / "xdg"),
        "TMPDIR": str(root / "cache" / "temp"),
    }


def apply_cache_environment() -> None:
    """Apply project-local cache variables to the current process.

    Needed before downloads, training, export, or checks that may create cache
    files. Used by environment, download, train, and export scripts.

    Inputs: none directly; uses `cache_environment`.
    Outputs: none; mutates `os.environ` and creates cache directories.
    """
    for key, value in cache_environment().items():
        os.environ.setdefault(key, value)
        Path(value).mkdir(parents=True, exist_ok=True)
