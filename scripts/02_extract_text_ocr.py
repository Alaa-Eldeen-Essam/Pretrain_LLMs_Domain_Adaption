#!/usr/bin/env python3
"""Extract text from manifest files with page-level resume.

PDFs are always rendered to images and OCRed with Tesseract. This script never
uses PyMuPDF or embedded PDF text extraction, because the source PDFs have known
encoding problems.

Resume behavior:
- Completed pages are detected from data/ocr_raw/pages.jsonl.
- A partially processed PDF resumes from the next page after the last completed
  page for that doc_id.
- Each page is appended immediately, so Ctrl+C or WSL kills lose at most the
  currently running page.
- The terminal shows one overall document bar and one per-PDF page bar.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_pipeline.io_utils import append_jsonl, now_utc, read_jsonl, require_file, write_json
from llm_pipeline.ocr_utils import ocr_image, ocr_pdf_pages, pdf_page_count, read_docx
from llm_pipeline.paths import configured_file, load_training

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements, fallback is for minimal envs.
    tqdm = None


class SimpleProgress:
    """Small terminal progress fallback when tqdm is unavailable.

    Needed so OCR still gives visible feedback in a minimal environment. Used
    by `progress_bar` when the optional `tqdm` import fails.

    Inputs: total count, initial count, description, unit label, and leave flag.
    Outputs: progress text written to the terminal.
    """

    def __init__(self, *, total: int, initial: int = 0, desc: str = "", unit: str = "item", leave: bool = True):
        """Initialize a fallback progress indicator.

        Needed to store progress state for later updates. Used internally when
        creating a `SimpleProgress` instance.

        Inputs: total, initial count, description, unit label, and leave flag.
        Outputs: initialized object state and first terminal render.
        """
        self.total = total
        self.count = initial
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self._print()

    def __enter__(self) -> "SimpleProgress":
        """Enter context-manager use of the progress indicator.

        Needed so fallback progress can be used with the same `with` pattern as
        `tqdm`. Used by the OCR loop.

        Inputs: none.
        Outputs: this progress object.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context-manager use of the progress indicator.

        Needed to finish the terminal line when the progress bar closes. Used
        automatically by `with` blocks.

        Inputs: exception type, exception value, and traceback from context
        manager protocol.
        Outputs: none.
        """
        if self.leave:
            print()

    def update(self, n: int = 1) -> None:
        """Advance the fallback progress count.

        Needed to show page/document progress as OCR proceeds. Used by the OCR
        loop after each completed item.

        Inputs: increment count.
        Outputs: updated terminal render.
        """
        self.count += n
        self._print()

    def set_description(self, desc: str) -> None:
        """Change the fallback progress description.

        Needed to show updated ETA/status text for the document bar. Used by
        the OCR loop.

        Inputs: new description string.
        Outputs: updated terminal render.
        """
        self.desc = desc
        self._print()

    def _print(self) -> None:
        """Render the fallback progress state to the terminal.

        Needed as the single output path for fallback progress display. Used by
        initialization, updates, and description changes.

        Inputs: current object state.
        Outputs: carriage-return terminal text.
        """
        print(f"\r{self.desc}: {self.count}/{self.total} {self.unit}", end="", flush=True)


def progress_bar(*, total: int, initial: int = 0, desc: str = "", unit: str = "item", leave: bool = True):
    """Create a terminal progress bar.

    Needed to prefer `tqdm` while keeping a no-dependency fallback. Used for
    overall document progress and per-PDF page progress.

    Inputs: total count, initial count, description, unit label, and leave flag.
    Outputs: context-manager progress object.
    """
    if tqdm is None:
        return SimpleProgress(total=total, initial=initial, desc=desc, unit=unit, leave=leave)
    return tqdm(total=total, initial=initial, desc=desc, unit=unit, dynamic_ncols=True, leave=leave)


def read_text_file(path: Path) -> str:
    """Read a text file with Arabic-friendly encoding fallbacks.

    Needed because corpus text files may not all be UTF-8. Used by the OCR
    extraction phase for `.txt` inputs.

    Inputs: text file path.
    Outputs: decoded text string.
    """
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "windows-1256"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_doc_with_system_tool(path: Path) -> str:
    """Read legacy DOC files through installed WSL tools.

    Needed because Python DOCX parsing does not handle old binary `.doc`
    documents. Used by the OCR extraction phase for `.doc` inputs.

    Inputs: legacy DOC file path.
    Outputs: extracted text from `antiword` or `catdoc`.
    """
    for command in ("antiword", "catdoc"):
        try:
            completed = subprocess.run(
                [command, str(path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            continue
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout
    raise RuntimeError("Legacy .doc requires antiword or catdoc in WSL.")


def completed_pages_by_doc(output_path: Path) -> Dict[str, Set[int]]:
    """Read completed OCR page numbers from the output JSONL.

    Needed for page-level resume after Ctrl+C, WSL kill, or machine restart.
    Used before OCR starts.

    Inputs: OCR pages JSONL path.
    Outputs: mapping from doc_id to completed page numbers.
    """
    completed: Dict[str, Set[int]] = defaultdict(set)
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return completed
    for record in read_jsonl(output_path):
        doc_id = str(record.get("doc_id") or "")
        page = record.get("page")
        if doc_id and page is not None:
            completed[doc_id].add(int(page))
    return completed


def seconds_to_hms(seconds: float | None) -> str | None:
    """Format seconds as a compact human-readable duration.

    Needed for progress messages and ETA fields. Used by terminal progress and
    JSON progress reporting.

    Inputs: seconds or None.
    Outputs: formatted duration string or None.
    """
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def short_name(record: Dict, max_len: int = 72) -> str:
    """Return a shortened display name for a source record.

    Needed to keep progress bars readable when corpus paths are long. Used by
    per-PDF page progress.

    Inputs: manifest record and maximum display length.
    Outputs: shortened path/name string.
    """
    name = str(record.get("relative_path") or record.get("source_path") or record.get("doc_id"))
    return name if len(name) <= max_len else f"...{name[-(max_len - 3):]}"


def base_record(record: Dict) -> Dict:
    """Build common OCR output metadata for a manifest record.

    Needed so every OCR page row contains consistent document identity fields.
    Used when writing PDF, image, TXT, DOCX, and DOC outputs.

    Inputs: manifest record.
    Outputs: dictionary with doc_id, source path, relative path, and file type.
    """
    return {
        "doc_id": record["doc_id"],
        "source_path": record["source_path"],
        "relative_path": record["relative_path"],
        "file_type": record["file_type"],
    }


def get_total_pages(record: Dict, *, max_pages: int | None) -> int:
    """Return the effective page count for a source record.

    Needed to decide completion state and initialize progress bars. Used before
    OCR processing each record.

    Inputs: manifest record and optional max-pages cap.
    Outputs: PDF page count or 1 for non-PDF sources.
    """
    if record["file_type"] == "pdf":
        return pdf_page_count(Path(record["source_path"]), max_pages=max_pages)
    return 1


def is_document_complete(record: Dict, completed_pages: Set[int], *, total_pages: int) -> bool:
    """Determine whether all expected pages for a record are already written.

    Needed to skip completed documents and support page-level resume. Used
    during status checks and processing.

    Inputs: manifest record, completed page set, and total page count.
    Outputs: boolean completion flag.
    """
    if total_pages <= 0:
        return False
    return all(page in completed_pages for page in range(1, total_pages + 1))


def next_start_page(completed_pages: Set[int]) -> int:
    """Return the next PDF page number to OCR.

    Needed to resume a partially completed PDF from the next missing page.
    Used by the PDF OCR loop.

    Inputs: set of completed page numbers.
    Outputs: one-based next page number.
    """
    if not completed_pages:
        return 1
    return max(completed_pages) + 1


def write_progress(
    *,
    progress_path: Path,
    status: str,
    total_documents: int,
    completed_documents: int,
    remaining_documents: int,
    current_record: Dict | None,
    current_page: int | None,
    current_page_count: int | None,
    last_completed: Dict | None,
    failures: List[Dict],
    elapsed_seconds: float,
    processed_documents_this_run: int,
    pages_written_this_run: int,
    output_path: Path,
) -> None:
    """Write the OCR progress report JSON.

    Needed to expose current file, page, completion counts, failures, and ETA
    outside the terminal. Used before, during, and after OCR processing.

    Inputs: progress path, status, counts, current/last records, failures,
    timing values, and output path.
    Outputs: none; writes `ocr_progress.json`.
    """
    average = elapsed_seconds / processed_documents_this_run if processed_documents_this_run else None
    estimated = average * remaining_documents if average is not None else None
    write_json(
        progress_path,
        {
            "timestamp_utc": now_utc(),
            "status": status,
            "total_documents": total_documents,
            "completed_documents": completed_documents,
            "remaining_documents": remaining_documents,
            "current_doc_id": current_record.get("doc_id") if current_record else None,
            "current_relative_path": current_record.get("relative_path") if current_record else None,
            "current_page": current_page,
            "current_page_count": current_page_count,
            "last_completed_doc_id": last_completed.get("doc_id") if last_completed else None,
            "last_completed_relative_path": last_completed.get("relative_path") if last_completed else None,
            "failures_this_run": failures,
            "elapsed_seconds_this_run": round(elapsed_seconds, 2),
            "processed_documents_this_run": processed_documents_this_run,
            "pages_written_this_run": pages_written_this_run,
            "average_seconds_per_document_this_run": round(average, 2) if average is not None else None,
            "estimated_remaining_seconds": round(estimated, 2) if estimated is not None else None,
            "estimated_remaining_human": seconds_to_hms(estimated),
            "output_path": str(output_path),
        },
    )


def extract_non_pdf_page(record: Dict, *, lang: str) -> Dict:
    """Extract the single logical page for non-PDF inputs.

    Needed so images, TXT, DOCX, and DOC files share the same page-record output
    format as PDFs. Used by the OCR processing loop.

    Inputs: manifest record and Tesseract language string for image OCR.
    Outputs: OCR/extracted page dictionary.
    """
    path = Path(record["source_path"])
    file_type = record["file_type"]
    base = base_record(record)
    if file_type == "image":
        text = ocr_image(path, lang=lang)
        method = "image_ocr"
    elif file_type == "txt":
        text = read_text_file(path)
        method = "text_file"
    elif file_type == "docx":
        text = read_docx(path)
        method = "docx_paragraphs"
    elif file_type == "doc":
        text = read_doc_with_system_tool(path)
        method = "legacy_doc_tool"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return {**base, "page": 1, "page_count": 1, "text": text, "char_count": len(text), "extraction_method": method}


def main() -> int:
    """Run the OCR extraction CLI phase with page-level resume.

    Needed as phase 02 to turn source files into raw page-level text while
    surviving interruptions and showing terminal progress. Used after inventory
    creation.

    Inputs: command-line arguments and source manifest JSONL.
    Outputs: exit code, raw OCR pages JSONL, OCR report JSON, and progress JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional supported-document limit.")
    parser.add_argument("--doc-id", default=None, help="Process one manifest doc_id only.")
    parser.add_argument("--max-pages", type=int, default=None, help="Override maximum PDF pages for test runs.")
    parser.add_argument("--dpi", type=int, default=None, help="Override OCR DPI. Use 200 if WSL kills the process.")
    parser.add_argument("--restart", action="store_true", help="Delete previous OCR output and start this phase again.")
    parser.add_argument("--status-only", action="store_true", help="Write/print current resume status without OCR work.")
    args = parser.parse_args()

    cfg = load_training()["ocr"]
    lang = cfg["languages"]
    dpi = int(args.dpi if args.dpi is not None else cfg["dpi"])
    max_pages = args.max_pages if args.max_pages is not None else cfg.get("max_pages")

    manifest_path = require_file(configured_file("manifest"), "source manifest")
    output_path = configured_file("ocr_pages")
    report_path = configured_file("ocr_report")
    progress_path = configured_file("ocr_progress")

    records = [record for record in read_jsonl(manifest_path) if record.get("supported")]
    if args.doc_id:
        records = [record for record in records if record["doc_id"] == args.doc_id]
    if args.limit is not None:
        records = records[: args.limit]

    if args.restart:
        output_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    completed_pages = completed_pages_by_doc(output_path)
    page_counts: Dict[str, int] = {}
    completed_documents = 0
    failures: List[Dict] = []

    for record in records:
        try:
            total_pages = get_total_pages(record, max_pages=max_pages)
            page_counts[record["doc_id"]] = total_pages
            if is_document_complete(record, completed_pages.get(record["doc_id"], set()), total_pages=total_pages):
                completed_documents += 1
        except Exception as exc:  # noqa: BLE001 - report bad files but keep scanning.
            failures.append({"doc_id": record["doc_id"], "relative_path": record["relative_path"], "error": str(exc)})

    remaining_documents = max(0, len(records) - completed_documents)
    if args.status_only:
        write_progress(
            progress_path=progress_path,
            status="status_only",
            total_documents=len(records),
            completed_documents=completed_documents,
            remaining_documents=remaining_documents,
            current_record=None,
            current_page=None,
            current_page_count=None,
            last_completed=None,
            failures=failures,
            elapsed_seconds=0,
            processed_documents_this_run=0,
            pages_written_this_run=0,
            output_path=output_path,
        )
        print(f"Completed documents: {completed_documents}")
        print(f"Remaining documents: {remaining_documents}")
        print(f"Progress report: {progress_path}")
        return 0 if not failures else 1

    pages_written = 0
    processed_documents_this_run = 0
    last_completed: Dict | None = None
    started = time.monotonic()

    with progress_bar(total=len(records), initial=completed_documents, desc="Documents", unit="doc") as doc_bar:
        for record in records:
            doc_id = record["doc_id"]
            if doc_id not in page_counts:
                continue

            total_pages = page_counts[doc_id]
            existing_pages = completed_pages.get(doc_id, set())
            if is_document_complete(record, existing_pages, total_pages=total_pages):
                continue

            write_progress(
                progress_path=progress_path,
                status="running",
                total_documents=len(records),
                completed_documents=completed_documents,
                remaining_documents=len(records) - completed_documents,
                current_record=record,
                current_page=next_start_page(existing_pages),
                current_page_count=total_pages,
                last_completed=last_completed,
                failures=failures,
                elapsed_seconds=time.monotonic() - started,
                processed_documents_this_run=processed_documents_this_run,
                pages_written_this_run=pages_written,
                output_path=output_path,
            )

            try:
                if record["file_type"] == "pdf":
                    start_page = next_start_page(existing_pages)
                    with progress_bar(
                        total=total_pages,
                        initial=len(existing_pages),
                        desc=short_name(record),
                        unit="page",
                        leave=False,
                    ) as page_bar:
                        for page in ocr_pdf_pages(
                            Path(record["source_path"]),
                            lang=lang,
                            dpi=dpi,
                            max_pages=max_pages,
                            start_page=start_page,
                        ):
                            append_jsonl(output_path, [{**base_record(record), **page, "extraction_method": "pdf_rendered_ocr"}])
                            pages_written += 1
                            existing_pages.add(int(page["page"]))
                            page_bar.update(1)
                            write_progress(
                                progress_path=progress_path,
                                status="running",
                                total_documents=len(records),
                                completed_documents=completed_documents,
                                remaining_documents=len(records) - completed_documents,
                                current_record=record,
                                current_page=int(page["page"]),
                                current_page_count=total_pages,
                                last_completed=last_completed,
                                failures=failures,
                                elapsed_seconds=time.monotonic() - started,
                                processed_documents_this_run=processed_documents_this_run,
                                pages_written_this_run=pages_written,
                                output_path=output_path,
                            )
                else:
                    page = extract_non_pdf_page(record, lang=lang)
                    append_jsonl(output_path, [page])
                    pages_written += 1
                    existing_pages.add(1)

                completed_documents += 1
                processed_documents_this_run += 1
                last_completed = record
                doc_bar.update(1)
                doc_bar.set_description(f"Documents ETA {seconds_to_hms((time.monotonic() - started) / max(1, processed_documents_this_run) * (len(records) - completed_documents))}")
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the corpus.
                failures.append({"doc_id": doc_id, "relative_path": record["relative_path"], "error": str(exc)})
                print(f"\nFAILED: {record['relative_path']} :: {exc}", file=sys.stderr)

    status = "completed_with_failures" if failures else "completed"
    elapsed = time.monotonic() - started
    write_progress(
        progress_path=progress_path,
        status=status,
        total_documents=len(records),
        completed_documents=completed_documents,
        remaining_documents=max(0, len(records) - completed_documents),
        current_record=None,
        current_page=None,
        current_page_count=None,
        last_completed=last_completed,
        failures=failures,
        elapsed_seconds=elapsed,
        processed_documents_this_run=processed_documents_this_run,
        pages_written_this_run=pages_written,
        output_path=output_path,
    )
    write_json(
        report_path,
        {
            "timestamp_utc": now_utc(),
            "documents_requested": len(records),
            "documents_completed_total": completed_documents,
            "documents_processed_this_run": processed_documents_this_run,
            "pages_written_this_run": pages_written,
            "failures": failures,
            "ocr_languages": lang,
            "dpi": dpi,
            "max_pages": max_pages,
            "output_path": str(output_path),
            "progress_path": str(progress_path),
        },
    )
    print(f"Wrote OCR pages: {output_path}")
    print(f"Wrote OCR report: {report_path}")
    print(f"Wrote OCR progress: {progress_path}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
