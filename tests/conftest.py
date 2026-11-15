"""Shared pytest fixtures and import path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import make_sample_pdf  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly generated multi-page text PDF."""
    target = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    target.write_bytes(make_sample_pdf.build_pdf())
    return target
