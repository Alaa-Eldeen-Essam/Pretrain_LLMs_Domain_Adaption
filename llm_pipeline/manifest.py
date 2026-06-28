from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".txt": "txt",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA-256 hash in streaming blocks.

    Needed to create stable document IDs and detect duplicate/changed files.
    Used by the inventory phase.

    Inputs: file path and optional block size.
    Outputs: lowercase SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_source_files(source_root: Path) -> Iterator[Path]:
    """Yield all files under the configured source corpus folder.

    Needed so inventory creation can scan the complete reference tree
    recursively. Used by the inventory phase.

    Inputs: source root directory.
    Outputs: sorted iterator of file paths.
    """
    for path in sorted(source_root.rglob("*")):
        if path.is_file():
            yield path


def build_manifest_record(path: Path, source_root: Path) -> Dict:
    """Build one manifest row for a source corpus file.

    Needed to centralize file metadata used by OCR and later reporting. Used by
    the inventory phase for every discovered file.

    Inputs: source file path and source root path.
    Outputs: dictionary with ID, paths, extension, type, size, mtime, hash, and
    support flag.
    """
    suffix = path.suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(suffix, "unsupported")
    relative = path.relative_to(source_root).as_posix()
    stat = path.stat()
    sha256 = sha256_file(path)
    return {
        "doc_id": sha256[:16],
        "source_path": str(path),
        "relative_path": relative,
        "extension": suffix,
        "file_type": file_type,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": sha256,
        "supported": file_type != "unsupported",
    }


def summarize_manifest(records: Iterable[Dict]) -> Dict:
    """Summarize manifest records by count, size, and file type.

    Needed to give a quick audit of corpus coverage before OCR starts. Used by
    the inventory phase.

    Inputs: iterable of manifest dictionaries.
    Outputs: summary dictionary for the inventory report.
    """
    total = 0
    supported = 0
    by_type: Dict[str, int] = {}
    total_bytes = 0
    for record in records:
        total += 1
        total_bytes += int(record.get("size_bytes") or 0)
        file_type = str(record.get("file_type") or "unknown")
        by_type[file_type] = by_type.get(file_type, 0) + 1
        if record.get("supported"):
            supported += 1
    return {
        "total_files": total,
        "supported_files": supported,
        "unsupported_files": total - supported,
        "total_bytes": total_bytes,
        "by_type": by_type,
    }
