#!/usr/bin/env python3
"""Clean OCR output and remove repeated page noise per document."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, read_jsonl, require_file, write_json, write_jsonl
from llm_pipeline.paths import configured_file, load_training
from llm_pipeline.text_cleaning import clean_ocr_text, find_repeated_lines, remove_repeated_lines


def main() -> int:
    """Run the text-cleaning CLI phase.

    Needed as phase 03 to convert raw OCR pages into normalized page text for
    dataset creation. Used after OCR extraction.

    Inputs: command-line arguments and `ocr_pages` JSONL.
    Outputs: exit code, cleaned pages JSONL, and cleaning report JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-pages", type=int, default=None, help="Optional page limit for quick checks.")
    parser.add_argument("--restart", action="store_true", help="Rebuild cleaned text outputs even if they already exist.")
    args = parser.parse_args()

    cfg = load_training()["cleaning"]
    input_path = require_file(configured_file("ocr_pages"), "OCR pages")
    output_path = configured_file("clean_pages")
    report_path = configured_file("cleaning_report")
    if output_path.exists() and report_path.exists() and not args.restart:
        print(f"Cleaned text already exists, skipping. Use --restart to rebuild: {output_path}")
        return 0
    if args.restart:
        output_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    pages = list(read_jsonl(input_path))
    if args.limit_pages is not None:
        pages = pages[: args.limit_pages]

    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for page in pages:
        by_doc[page["doc_id"]].append(page)

    cleaned_records: List[Dict] = []
    dropped = 0
    repeated_by_doc = {}
    for doc_id, doc_pages in by_doc.items():
        initially_clean = [
            clean_ocr_text(page.get("text", ""), strip_tashkeel=bool(cfg["remove_tashkeel"]))
            for page in doc_pages
        ]
        repeated = find_repeated_lines(initially_clean, float(cfg["repeated_line_threshold"]))
        repeated_by_doc[doc_id] = len(repeated)
        for page, text in zip(doc_pages, initially_clean):
            text = remove_repeated_lines(text, repeated)
            if len(text) < int(cfg["min_text_chars"]):
                dropped += 1
                continue
            cleaned_records.append(
                {
                    "doc_id": page["doc_id"],
                    "relative_path": page["relative_path"],
                    "source_path": page["source_path"],
                    "page": page["page"],
                    "file_type": page["file_type"],
                    "text": text,
                    "char_count": len(text),
                }
            )

    write_jsonl(output_path, cleaned_records)
    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "input_pages": len(pages),
            "output_pages": len(cleaned_records),
            "dropped_short_pages": dropped,
            "documents": len(by_doc),
            "repeated_lines_per_doc": repeated_by_doc,
            "output_path": str(output_path),
        },
    )
    print(f"Wrote cleaned pages: {output_path}")
    print(f"Wrote cleaning report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
