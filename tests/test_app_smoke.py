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


def test_app_stops_when_key_is_missing(monkeypatch) -> None:
    """A missing key must produce an actionable error, not a traceback."""
    import utils

    monkeypatch.setattr(utils, "key_problem", lambda: "GEMINI_API_KEY is not set.")
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    assert not at.exception
    assert any("GEMINI_API_KEY is not set" in error.value for error in at.error)
    assert not at.file_uploader
