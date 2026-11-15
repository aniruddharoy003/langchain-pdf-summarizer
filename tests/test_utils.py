"""Tests for the presentation helpers (no Streamlit runtime required)."""

from __future__ import annotations

import pandas as pd

from summarizer import ChunkSummary, SummaryResult
from utils import estimate_processing_time, export_summary_text, summary_dataframe


def _result() -> SummaryResult:
    return SummaryResult(
        summary_type="bullet_points",
        total_chunks=3,
        chunk_summaries=[
            ChunkSummary(1, "first", True, 100, 50),
            ChunkSummary(2, "second", True, 200, 40),
            ChunkSummary(3, "boom", False, 300, 0),
        ],
        final_summary="the merged summary",
    )


def test_dataframe_shape_and_columns() -> None:
    frame = summary_dataframe(_result())
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert list(frame.columns) == [
        "Chunk",
        "Original (chars)",
        "Summary (chars)",
        "Compression",
        "Status",
    ]
    assert frame.loc[0, "Compression"] == 0.5
    assert frame.loc[2, "Status"] == "Failed"


def test_export_contains_summary_and_chunks() -> None:
    text = export_summary_text(_result(), "report.pdf")
    assert "report.pdf" in text
    assert "the merged summary" in text
    assert "Bullet Points" in text
    assert "Chunk 3 (FAILED)" in text


def test_estimate_scales_with_chunk_count() -> None:
    assert estimate_processing_time(0) == "—"
    assert estimate_processing_time(1).startswith("~")
    assert estimate_processing_time(40).startswith("~1 min")
