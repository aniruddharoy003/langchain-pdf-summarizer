"""Headless smoke test: the Streamlit app must render without raising."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_boots_without_upload() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    assert not at.exception
    assert "PDF AI Summarizer" in at.title[0].value
    assert at.file_uploader
    # No file yet, so the prompt should be visible and no results rendered.
    assert any("Upload a text-based PDF" in info.value for info in at.info)
