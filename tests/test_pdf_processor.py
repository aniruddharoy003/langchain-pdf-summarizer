"""Unit tests for pdf_processor."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_processor import PDFError, PDFProcessor


@pytest.fixture()
def processor() -> PDFProcessor:
    return PDFProcessor()


# -- cleaning ---------------------------------------------------------------
def test_clean_text_strips_control_chars_and_collapses_space(processor: PDFProcessor) -> None:
    raw = "Hello\x00   world\t\tagain\n\n\n\nnext line  "
    assert processor.clean_text(raw) == "Hello world again\n\nnext line"


def test_clean_text_keeps_paragraphs_and_punctuation(processor: PDFProcessor) -> None:
    raw = "First point.\n\nSecond point, with a comma."
    assert processor.clean_text(raw) == raw


# -- tokens -----------------------------------------------------------------
def test_count_tokens_is_positive(processor: PDFProcessor) -> None:
    assert processor.count_tokens("hello world") > 0


def test_truncate_respects_token_budget(processor: PDFProcessor) -> None:
    text = "word " * 500
    truncated = processor.truncate(text, 50)
    assert processor.count_tokens(truncated) <= 50
    assert processor.truncate("short", 50) == "short"


# -- chunking ---------------------------------------------------------------
@pytest.mark.parametrize("max_tokens", [50, 200, 800])
def test_chunks_never_exceed_budget(processor: PDFProcessor, max_tokens: int) -> None:
    text = processor.clean_text(
        "This is a sentence about summarization. " * 200
    )
    chunks = processor.chunk_text(text, max_tokens)
    assert len(chunks) > 1
    for chunk in chunks:
        assert processor.count_tokens(chunk) <= max_tokens


def test_chunking_loses_no_words(processor: PDFProcessor) -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = processor.chunk_text(text, 500)
    assert " ".join(chunks).split() == text.split()


def test_hard_split_of_oversized_sentence(processor: PDFProcessor) -> None:
    # One "sentence" with no terminal punctuation, longer than the budget.
    text = " ".join(f"token{i}" for i in range(600))
    chunks = processor.chunk_text(text, 100)
    assert len(chunks) > 1
    assert all(processor.count_tokens(c) <= 100 for c in chunks)


def test_empty_text_raises(processor: PDFProcessor) -> None:
    with pytest.raises(PDFError):
        processor.chunk_text("   ")


def test_non_positive_budget_raises(processor: PDFProcessor) -> None:
    with pytest.raises(ValueError):
        processor.chunk_text("some text", 0)


# -- extraction -------------------------------------------------------------
def test_extract_text_from_generated_pdf(processor: PDFProcessor, sample_pdf: Path) -> None:
    text = processor.extract_text(sample_pdf)
    assert "small" in text.lower()
    assert len(text) > 1000


def test_extract_text_rejects_non_pdf(processor: PDFProcessor, tmp_path: Path) -> None:
    bogus = tmp_path / "not_a.pdf"
    bogus.write_text("this is not a pdf")
    with pytest.raises(PDFError):
        processor.extract_text(bogus)


def test_metadata_from_generated_pdf(processor: PDFProcessor, sample_pdf: Path) -> None:
    meta = processor.metadata(sample_pdf)
    assert meta.num_pages >= 2
    assert meta.title == "The Quiet Rise of Small Language Models"
    assert meta.author == "Shiva Roy"
    assert meta.has_document_info


def test_extract_accepts_open_file_handle(processor: PDFProcessor, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as handle:
        assert len(processor.extract_text(handle)) > 1000
