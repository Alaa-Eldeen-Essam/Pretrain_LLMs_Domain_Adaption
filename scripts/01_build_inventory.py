#!/usr/bin/env python3
"""Create a source-file manifest for the reference corpus.

The manifest is the contract for later phases. It records every source file,
its hash, size, type, and relative path. No OCR, cleaning, or model work happens
in this phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, write_json, write_jsonl
from llm_pipeline.manifest import build_manifest_record, iter_source_files, summarize_manifest
from llm_pipeline.paths import configured_file, ensure_project_tree, source_root


def main() -> int:
    """Run the source inventory CLI phase.

    Needed as phase 01 to create the manifest consumed by OCR. Used directly
    from the terminal after environment checks.

    Inputs: command-line arguments and configured source folder.
    Outputs: exit code, source manifest JSONL, and inventory summary JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional file limit for quick dry runs.")
    parser.add_argument("--restart", action="store_true", help="Rebuild inventory outputs even if they already exist.")
    args = parser.parse_args()

    ensure_project_tree()
    src = source_root()
    if not src.exists():
        raise FileNotFoundError(f"Source corpus folder does not exist: {src}")
    manifest_path = configured_file("manifest")
    summary_path = configured_file("inventory_summary")
    if manifest_path.exists() and summary_path.exists() and not args.restart:
        print(f"Inventory already exists, skipping. Use --restart to rebuild: {manifest_path}")
        return 0
    if args.restart:
        manifest_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    records = []
    for index, path in enumerate(iter_source_files(src), start=1):
        if args.limit is not None and index > args.limit:
            break
        records.append(build_manifest_record(path, src))

    count = write_jsonl(manifest_path, records)
    summary = summarize_manifest(records)
    summary.update({"timestamp_utc": now_utc(), "source_root": str(src), "manifest_path": str(manifest_path)})
    write_json(summary_path, summary)

    print(f"Wrote {count} manifest records: {manifest_path}")
    print(f"Wrote inventory summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
