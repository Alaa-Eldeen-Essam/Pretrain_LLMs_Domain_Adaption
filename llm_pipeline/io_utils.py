from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from .paths import assert_inside_project


def now_utc() -> str:
    """Return the current UTC timestamp for report metadata.

    Needed so every phase report has a consistent machine-readable time. Used
    by all phase scripts when writing JSON reports.

    Inputs: none.
    Outputs: ISO-8601 UTC timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Read a UTF-8 JSONL file one object at a time.

    Needed so later phases can consume previous phase outputs without loading
    unnecessary wrapper formats. Used by inventory, OCR, cleaning, and dataset
    scripts.

    Inputs: path to a JSONL file.
    Outputs: iterator of decoded dictionary records.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write records to a UTF-8 JSONL file, replacing existing content.

    Needed for phase outputs that are complete regenerated artifacts. Used by
    inventory, cleaning, and dataset creation.

    Inputs: destination path and iterable of dictionary records.
    Outputs: number of records written.
    """
    assert_inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Append records to a UTF-8 JSONL file.

    Needed for resumable OCR so completed pages are persisted immediately.
    Used by the OCR extraction phase.

    Inputs: destination path and iterable of dictionary records.
    Outputs: number of records appended.
    """
    assert_inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a formatted UTF-8 JSON report.

    Needed for human-inspectable phase reports and progress files. Used by all
    phase scripts that produce status metadata.

    Inputs: destination path and dictionary payload.
    Outputs: none; writes the JSON file.
    """
    assert_inside_project(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def require_file(path: Path, label: str) -> Path:
    """Validate that a required input file exists and is non-empty.

    Needed so each decoupled phase fails early with a clear message when its
    previous phase output is missing. Used before reading required artifacts.

    Inputs: path to validate and a human-readable label.
    Outputs: the validated path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return path
