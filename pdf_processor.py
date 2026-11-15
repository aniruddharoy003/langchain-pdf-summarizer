"""PDF ingestion: text extraction, cleaning, token counting and chunking.

The processor is UI-agnostic: it takes anything pypdf can read (a path, an open
file handle or an in-memory upload buffer) and returns plain text or chunks.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

from config import DEFAULT_CHUNK_TOKENS, MAX_PAGES

logger = logging.getLogger(__name__)

# Control characters that only ever come from broken PDF text layers.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WS = re.compile(r"[ \t\u00a0\u2000-\u200a]+")
_SPACE_BEFORE_NEWLINE = re.compile(r" ?\n ?")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
# Split after ., ! or ? when followed by whitespace.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

UNKNOWN = "Unknown"


class PDFError(Exception):
    """Raised when a PDF cannot be read or contains no usable text."""


@dataclass(frozen=True)
class PDFMetadata:
    num_pages: int
    title: str
    author: str
    subject: str

    @property
    def has_document_info(self) -> bool:
        return any(v != UNKNOWN for v in (self.title, self.author, self.subject))


class PDFProcessor:
    """Extract, clean and split PDF text, sizing chunks by tokens."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        # cl100k_base is a close-enough proxy for Gemini's tokenizer and lets us
        # size chunks deterministically without a network round-trip.
        self._encoding = tiktoken.get_encoding(encoding_name)

    # -- extraction ---------------------------------------------------------
    def extract_text(self, pdf_file: str | Path | BinaryIO) -> str:
        """Return the text of every readable page, joined by blank lines."""
        reader = self._read(pdf_file)
        num_pages = len(reader.pages)
        if num_pages > MAX_PAGES:
            raise PDFError(
                f"PDF has {num_pages} pages; the limit is {MAX_PAGES}. "
                "Split the document or raise MAX_PAGES in config.py."
            )

        pages: list[str] = []
        failed = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the run
                failed += 1
                logger.warning("Skipping page %d: %s", number, exc)

        if failed:
            logger.warning("Skipped %d unreadable page(s)", failed)

        text = "\n\n".join(pages).strip()
        if not text:
            raise PDFError(
                "No extractable text found. The PDF is probably scanned images; "
                "run OCR on it first."
            )
        return text

    def metadata(self, pdf_file: str | Path | BinaryIO) -> PDFMetadata:
        """Document metadata; missing or unreadable fields become 'Unknown'."""
        reader = self._read(pdf_file)
        info = reader.metadata or {}

        def field(key: str) -> str:
            value = info.get(key)
            value = str(value).strip() if value is not None else ""
            return value or UNKNOWN

        return PDFMetadata(
            num_pages=len(reader.pages),
            title=field("/Title"),
            author=field("/Author"),
            subject=field("/Subject"),
        )

    def _read(self, pdf_file: str | Path | BinaryIO) -> PdfReader:
        try:
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)  # type: ignore[union-attr]
            return PdfReader(pdf_file)
        except PDFError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface a single friendly error
            raise PDFError(f"Could not read the PDF: {exc}") from exc

    # -- cleaning -----------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """Normalise whitespace while keeping paragraphs and punctuation."""
        text = _CONTROL_CHARS.sub("", text)
        text = _HORIZONTAL_WS.sub(" ", text)
        text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
        text = _EXCESS_NEWLINES.sub("\n\n", text)
        return text.strip()

    # -- tokens -------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, max_tokens: int) -> str:
        """First ``max_tokens`` tokens of ``text`` (used for analysis/quotes)."""
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    # -- chunking -----------------------------------------------------------
    def chunk_text(self, text: str, max_tokens: int = DEFAULT_CHUNK_TOKENS) -> list[str]:
        """Split text into chunks of at most ``max_tokens``, preferring sentence ends."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        text = text.strip()
        if not text:
            raise PDFError("Nothing to chunk: the text is empty.")

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for piece in self._split_pieces(text, max_tokens):
            tokens = self.count_tokens(piece)
            # +1 accounts for the space inserted when pieces are joined.
            needed = tokens + (1 if current else 0)
            if current and current_tokens + needed > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
                needed = tokens
            current.append(piece)
            current_tokens += needed

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_pieces(self, text: str, max_tokens: int) -> Iterator[str]:
        """Yield sentences, hard-splitting any sentence that exceeds the limit."""
        for sentence in _SENTENCE_BOUNDARY.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            tokens = self._encoding.encode(sentence, disallowed_special=())
            if len(tokens) <= max_tokens:
                yield sentence
                continue
            for start in range(0, len(tokens), max_tokens):
                window = self._encoding.decode(tokens[start : start + max_tokens]).strip()
                if window:
                    yield window
