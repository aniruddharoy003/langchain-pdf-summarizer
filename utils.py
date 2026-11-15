"""Presentation helpers: rendering, exports and small UI calculations.

Kept separate from ``app.py`` so the Streamlit layer stays readable and these
functions stay testable without a browser.
"""

from __future__ import annotations

import math
import time

import pandas as pd
import streamlit as st

from config import MAX_CONCURRENCY, settings
from summarizer import SummaryResult


def key_problem() -> str | None:
    """Return a human-readable API-key problem, or ``None`` when the key is usable."""
    return settings.key_problem


def estimate_processing_time(num_chunks: int) -> str:
    """Rough wall-clock estimate: batches of MAX_CONCURRENCY, ~8s each, +5s reduce."""
    if num_chunks <= 0:
        return "—"
    seconds = math.ceil(num_chunks / MAX_CONCURRENCY) * 8 + 5
    if seconds < 60:
        return f"~{seconds} seconds"
    return f"~{seconds // 60} min {seconds % 60}s"


def summary_dataframe(result: SummaryResult) -> pd.DataFrame:
    """One row per chunk with sizes, compression ratio and status."""
    rows = []
    for chunk in result.chunk_summaries:
        ratio = (
            round(chunk.summary_chars / chunk.original_chars, 2) if chunk.original_chars else 0.0
        )
        rows.append(
            {
                "Chunk": chunk.chunk_number,
                "Original (chars)": chunk.original_chars,
                "Summary (chars)": chunk.summary_chars,
                "Compression": ratio,
                "Status": "Success" if chunk.ok else "Failed",
            }
        )
    return pd.DataFrame(rows)


def export_summary_text(result: SummaryResult, file_name: str = "") -> str:
    """Plain-text report suitable for download."""
    lines = ["PDF Summary Report", "=" * 18, ""]
    if file_name:
        lines.append(f"File: {file_name}")
    lines += [
        f"Summary type: {result.summary_type.replace('_', ' ').title()}",
        f"Chunks processed: {result.total_chunks}",
        f"Chunks succeeded: {result.succeeded}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=" * 13,
        "",
        result.final_summary,
        "",
        "CHUNK SUMMARIES",
        "=" * 14,
        "",
    ]
    for chunk in result.chunk_summaries:
        label = "FAILED" if not chunk.ok else f"{chunk.summary_chars:,} chars"
        lines += [f"Chunk {chunk.chunk_number} ({label}):", chunk.summary, ""]
    return "\n".join(lines)


def render_result(result: SummaryResult, analysis: str | None, quotes: list[str] | None) -> None:
    """Render stats, final summary, per-chunk details, analysis and quotes."""
    cols = st.columns(3)
    cols[0].metric("Chunks", result.total_chunks)
    cols[1].metric("Succeeded", f"{result.succeeded}/{result.total_chunks}")
    cols[2].metric("Style", result.summary_type.replace("_", " ").title())

    if result.failed:
        st.warning(
            f"{result.failed} chunk(s) could not be summarized. "
            "The final summary is built from the successful ones."
        )

    st.subheader("📝 Final summary")
    st.markdown(result.final_summary)

    with st.expander(f"Per-chunk summaries ({result.total_chunks})"):
        for chunk in result.chunk_summaries:
            heading = f"**Chunk {chunk.chunk_number}**"
            if not chunk.ok:
                heading += " ⚠️ failed"
            st.markdown(heading)
            st.write(chunk.summary)
            st.caption(f"{chunk.original_chars:,} chars → {chunk.summary_chars:,} chars")
            st.divider()

    if analysis:
        st.subheader("📊 Document analysis")
        st.markdown(analysis)

    if quotes:
        st.subheader("💬 Key quotes")
        for index, quote in enumerate(quotes, start=1):
            st.markdown(f"{index}. *“{quote}”*")
