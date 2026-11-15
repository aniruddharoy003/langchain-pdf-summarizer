"""Shared pytest fixtures and import path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The suite never calls Gemini, so a dummy key keeps it deterministic on a fresh
# clone (no .env) without masking anything: config.load_dotenv(override=False)
# leaves these values in place.
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used-by-tests")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-flash-lite")

import make_sample_pdf  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly generated multi-page text PDF."""
    target = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    target.write_bytes(make_sample_pdf.build_pdf())
    return target
