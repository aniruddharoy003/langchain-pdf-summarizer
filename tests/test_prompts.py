"""Prompt template tests — no network calls."""

from __future__ import annotations

import pytest

from config import SUMMARY_TYPES
from prompts import (
    ANALYSIS,
    CHUNK_SUMMARY,
    COMBINE,
    QUOTES,
    STYLES,
    style_instruction,
)


def test_every_configured_style_has_instructions() -> None:
    for summary_type in SUMMARY_TYPES:
        assert summary_type in STYLES
        assert STYLES[summary_type].strip()


def test_unknown_style_falls_back_to_default() -> None:
    assert style_instruction("nonsense") == style_instruction("detailed")


def test_chunk_summary_renders_style_and_text() -> None:
    messages = CHUNK_SUMMARY.format_messages(style="STYLE-X", text="BODY-Y")
    assert len(messages) == 2
    assert messages[0].type == "system"
    human = messages[1].content
    assert "STYLE-X" in human and "BODY-Y" in human


def test_combine_renders_sections() -> None:
    human = COMBINE.format_messages(style="s", summaries="Section 1: a")[1].content
    assert "Section 1: a" in human


@pytest.mark.parametrize("template", [ANALYSIS, QUOTES])
def test_single_variable_templates_render(template) -> None:
    human = template.format_messages(text="DOC")[1].content
    assert "DOC" in human
