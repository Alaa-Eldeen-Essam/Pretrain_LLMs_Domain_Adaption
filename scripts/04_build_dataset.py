#!/usr/bin/env python3
"""Build train/validation JSONL samples for continued pretraining."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import now_utc, read_jsonl, require_file, write_json, write_jsonl
from llm_pipeline.paths import configured_file, load_training


def chunk_text(text: str, *, target_chars: int, max_chars: int, min_chars: int) -> Iterable[str]:
    """Split document text into training-sized chunks.

    Needed to produce unlabeled continued-pretraining samples that fit the
    configured sequence budget. Used by the dataset-building phase.

    Inputs: source text and chunk size limits.
    Outputs: iterator of chunk strings.
    """
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    current: List[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) > target_chars:
            chunk = "\n".join(current).strip()
            if len(chunk) >= min_chars:
                yield chunk[:max_chars]
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 1
    if current:
        chunk = "\n".join(current).strip()
        if len(chunk) >= min_chars:
            yield chunk[:max_chars]


def main() -> int:
    """Run the dataset-building CLI phase.

    Needed as phase 04 to produce train/validation JSONL files for Unsloth.
    Used after cleaned page text exists.

    Inputs: command-line arguments and cleaned pages JSONL.
    Outputs: exit code, train dataset, validation dataset, and dataset report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-docs", type=int, default=None, help="Optional document limit for quick checks.")
    parser.add_argument("--restart", action="store_true", help="Rebuild dataset outputs even if they already exist.")
    args = parser.parse_args()

    cfg = load_training()["dataset"]
    input_path = require_file(configured_file("clean_pages"), "cleaned pages")
    train_path = configured_file("train_dataset")
    validation_path = configured_file("validation_dataset")
    report_path = configured_file("dataset_report")
    if train_path.exists() and validation_path.exists() and report_path.exists() and not args.restart:
        print(f"Dataset already exists, skipping. Use --restart to rebuild: {train_path}")
        return 0
    if args.restart:
        train_path.unlink(missing_ok=True)
        validation_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    pages = list(read_jsonl(input_path))
    by_doc: Dict[str, List[Dict]] = defaultdict(list)
    for page in pages:
        by_doc[page["doc_id"]].append(page)

    doc_ids = sorted(by_doc)
    if args.limit_docs is not None:
        doc_ids = doc_ids[: args.limit_docs]
    rng = random.Random(int(cfg["random_seed"]))
    rng.shuffle(doc_ids)

    validation_count = max(1, int(len(doc_ids) * float(cfg["validation_ratio"]))) if len(doc_ids) > 1 else 0
    validation_docs = set(doc_ids[:validation_count])

    train_records: List[Dict] = []
    validation_records: List[Dict] = []
    for doc_id in doc_ids:
        ordered_pages = sorted(by_doc[doc_id], key=lambda record: int(record["page"]))
        combined = "\n\n".join(page["text"] for page in ordered_pages)
        relative_path = ordered_pages[0]["relative_path"]
        for sample_index, sample_text in enumerate(
            chunk_text(
                combined,
                target_chars=int(cfg["target_chars_per_sample"]),
                max_chars=int(cfg["max_chars_per_sample"]),
                min_chars=int(cfg["min_chars_per_sample"]),
            ),
            start=1,
        ):
            record = {
                "id": f"{doc_id}-{sample_index:04d}",
                "doc_id": doc_id,
                "relative_path": relative_path,
                "text": sample_text,
                "char_count": len(sample_text),
            }
            if doc_id in validation_docs:
                validation_records.append(record)
            else:
                train_records.append(record)

    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "documents": len(doc_ids),
            "validation_documents": len(validation_docs),
            "train_samples": len(train_records),
            "validation_samples": len(validation_records),
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "config": cfg,
        },
    )
    print(f"Wrote train dataset: {train_path}")
    print(f"Wrote validation dataset: {validation_path}")
    print(f"Wrote dataset report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
