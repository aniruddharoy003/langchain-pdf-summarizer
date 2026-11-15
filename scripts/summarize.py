"""Command-line summarizer — the headless twin of the Streamlit app.

    python scripts/summarize.py tests/sample.pdf --type detailed
    python scripts/summarize.py report.pdf --type brief --max-tokens 300 --analysis --quotes

Useful for quick checks and for testing the pipeline without a browser.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ANALYSIS_CHAR_LIMIT,
    DEFAULT_CHUNK_TOKENS,
    QUOTES_CHAR_LIMIT,
    SUMMARY_TYPES,
)
from pdf_processor import PDFError, PDFProcessor  # noqa: E402
from summarizer import PDFSummarizer, SummarizerError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a PDF from the terminal.")
    parser.add_argument("pdf", type=Path, help="path to a text-based PDF")
    parser.add_argument(
        "--type",
        default="detailed",
        choices=SUMMARY_TYPES,
        help="summary style (default: detailed)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"token budget per chunk (default: {DEFAULT_CHUNK_TOKENS})",
    )
    parser.add_argument("--analysis", action="store_true", help="also analyze document structure")
    parser.add_argument("--quotes", action="store_true", help="also extract key quotes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.pdf.exists():
        print(f"error: {args.pdf} does not exist")
        return 2

    processor = PDFProcessor()
    started = time.time()

    try:
        metadata = processor.metadata(args.pdf)
        raw = processor.extract_text(args.pdf)
        text = processor.clean_text(raw)
        chunks = processor.chunk_text(text, args.max_tokens)
    except PDFError as exc:
        print(f"error: {exc}")
        return 1

    print(f"file      : {args.pdf.name}")
    print(f"pages     : {metadata.num_pages}")
    print(f"title     : {metadata.title}")
    print(f"author    : {metadata.author}")
    print(f"characters: {len(text):,}")
    print(f"tokens    : {processor.count_tokens(text):,}")
    print(f"chunks    : {len(chunks)} (budget {args.max_tokens} tokens)")

    try:
        summarizer = PDFSummarizer()
    except SummarizerError as exc:
        print(f"error: {exc}")
        return 1

    print(f"model     : {summarizer.model}\n")

    try:
        result = summarizer.summarize_chunks(chunks, args.type)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: summarization failed: {exc}")
        return 1

    for chunk in result.chunk_summaries:
        mark = "ok  " if chunk.ok else "FAIL"
        print(
            f"  [{mark}] chunk {chunk.chunk_number}: "
            f"{chunk.original_chars:,} -> {chunk.summary_chars:,} chars"
        )

    print(
        f"\n=== {args.type.upper()} SUMMARY ({result.succeeded}/{result.total_chunks} chunks) ===\n"
    )
    print(result.final_summary)

    if args.analysis:
        analysis = summarizer.analyze_document(processor.truncate(text, ANALYSIS_CHAR_LIMIT))
        print("\n=== DOCUMENT ANALYSIS ===\n")
        print(analysis.text)

    if args.quotes:
        print("\n=== KEY QUOTES ===\n")
        for i, quote in enumerate(
            summarizer.extract_quotes(processor.truncate(text, QUOTES_CHAR_LIMIT)), start=1
        ):
            print(f'{i}. "{quote}"')

    print(f"\nfinished in {time.time() - started:.1f}s")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
