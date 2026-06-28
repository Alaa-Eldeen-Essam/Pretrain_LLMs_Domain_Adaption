from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, List, Set

ZERO_WIDTH = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\ufeff",
}

TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def remove_zero_width(text: str) -> str:
    """Remove invisible Unicode control characters from OCR text.

    Needed because OCR and copied Arabic text can contain zero-width marks that
    hurt tokenization. Used by `clean_ocr_text`.

    Inputs: raw text.
    Outputs: text without configured zero-width characters.
    """
    return "".join(ch for ch in text if ch not in ZERO_WIDTH)


def remove_tashkeel(text: str) -> str:
    """Remove Arabic diacritics from text.

    Needed to reduce OCR variance and normalize Arabic training text. Used by
    `clean_ocr_text` when configured.

    Inputs: raw or normalized text.
    Outputs: text without Arabic tashkeel characters.
    """
    return TASHKEEL_RE.sub("", text)


def normalize_spaces(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks.

    Needed to make OCR output consistent before chunking into training samples.
    Used by cleaning helpers.

    Inputs: text with arbitrary whitespace.
    Outputs: stripped text with normalized spaces and newlines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ocr_text(text: str, *, strip_tashkeel: bool = True) -> str:
    """Apply baseline Arabic-friendly OCR text normalization.

    Needed to convert noisy page text into stable training text before header
    removal and dataset chunking. Used by the cleaning phase.

    Inputs: page text and whether to remove tashkeel.
    Outputs: cleaned page text.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = remove_zero_width(text)
    if strip_tashkeel:
        text = remove_tashkeel(text)
    text = text.replace("ـ", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    return normalize_spaces(text)


def find_repeated_lines(page_texts: Iterable[str], threshold: float) -> Set[str]:
    """Find likely repeated headers or footers across pages.

    Needed to remove recurring page noise without labels. Used by the cleaning
    phase per document.

    Inputs: iterable of page texts and document-level frequency threshold.
    Outputs: set of repeated line strings.
    """
    pages: List[List[str]] = []
    for text in page_texts:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        pages.append(lines)
    page_count = max(len(pages), 1)
    counts: Counter[str] = Counter()
    for lines in pages:
        for line in set(lines):
            if 3 <= len(line) <= 120:
                counts[line] += 1
    return {line for line, count in counts.items() if count / page_count >= threshold}


def remove_repeated_lines(text: str, repeated_lines: Set[str]) -> str:
    """Remove repeated document noise and standalone page numbers.

    Needed so training samples focus on content rather than headers, footers,
    and pagination artifacts. Used by the cleaning phase.

    Inputs: page text and repeated-line set for its document.
    Outputs: cleaned text with repeated lines removed.
    """
    cleaned = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped in repeated_lines:
            continue
        if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", stripped):
            continue
        cleaned.append(line)
    return normalize_spaces("\n".join(cleaned))
