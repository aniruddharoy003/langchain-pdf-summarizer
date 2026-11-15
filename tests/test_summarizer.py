"""Summarizer tests using a fake chat model — no API calls."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

import summarizer as summarizer_module
from summarizer import PDFSummarizer, SummarizerError, clean_quote_line, message_text


def _summarizer_with(responses: list[str]) -> PDFSummarizer:
    """A summarizer whose chains run against canned model responses."""
    instance = PDFSummarizer.__new__(PDFSummarizer)
    instance.model = "fake-model"
    llm = FakeListChatModel(responses=responses)
    parser = summarizer_module.RunnableLambda(message_text)
    instance._chunk_chain = summarizer_module.CHUNK_SUMMARY | llm | parser
    instance._combine_chain = summarizer_module.COMBINE | llm | parser
    instance._analysis_chain = summarizer_module.ANALYSIS | llm | parser
    instance._quotes_chain = summarizer_module.QUOTES | llm | parser
    return instance


# -- message parsing --------------------------------------------------------
def test_message_text_plain_string() -> None:
    assert message_text(AIMessage(content="  hello  ")) == "hello"


def test_message_text_block_list() -> None:
    message = AIMessage(
        content=[{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
    )
    assert message_text(message) == "part one\npart two"


def test_message_text_ignores_non_text_blocks() -> None:
    message = AIMessage(
        content=[{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "answer"}]
    )
    assert message_text(message) == "answer"


def test_message_text_human_message() -> None:
    assert message_text(HumanMessage(content="hi")) == "hi"


# -- quote cleaning ---------------------------------------------------------
def test_clean_quote_line_strips_markers_and_quotes() -> None:
    assert clean_quote_line('1. "A quote."') == "A quote."
    assert clean_quote_line("- 'Another one'") == "Another one"
    assert clean_quote_line("• plain") == "plain"
    assert clean_quote_line("   ") == ""


# -- chains -----------------------------------------------------------------
def test_summarize_chunks_returns_one_entry_per_chunk() -> None:
    summarizer = _summarizer_with(["sum one", "sum two", "sum three", "combined"])
    result = summarizer.summarize_chunks(["a" * 10, "b" * 20, "c" * 30], "detailed")

    assert result.total_chunks == 3
    assert [c.chunk_number for c in result.chunk_summaries] == [1, 2, 3]
    assert result.succeeded == 3
    assert result.final_summary == "combined"
    assert result.chunk_summaries[1].original_chars == 20


def test_single_chunk_skips_combine_step() -> None:
    summarizer = _summarizer_with(["only summary"])
    result = summarizer.summarize_chunks(["solo text"], "brief")
    assert result.final_summary == "only summary"
    assert result.success_rate == 1.0


def test_blank_chunks_are_dropped() -> None:
    summarizer = _summarizer_with(["only"])
    result = summarizer.summarize_chunks(["text", "   ", ""], "brief")
    assert result.total_chunks == 1


def test_no_usable_text_raises() -> None:
    summarizer = _summarizer_with([])
    try:
        summarizer.summarize_chunks(["  "], "brief")
    except SummarizerError as exc:
        assert "No text" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SummarizerError")


def test_missing_api_key_raises() -> None:
    try:
        PDFSummarizer(api_key="")
    except SummarizerError as exc:
        assert "GEMINI_API_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SummarizerError")


def test_extract_quotes_cleans_and_caps() -> None:
    lines = "\n".join(f'{i}. "Quote number {i}"' for i in range(1, 15))
    summarizer = _summarizer_with([lines])
    quotes = summarizer.extract_quotes("doc")
    assert len(quotes) == 10
    assert quotes[0] == "Quote number 1"


def test_analysis_failure_is_reported_not_raised() -> None:
    summarizer = _summarizer_with(["unused"])

    class Boom:
        def invoke(self, *_args, **_kwargs):
            raise RuntimeError("model exploded")

    summarizer._analysis_chain = Boom()
    result = summarizer.analyze_document("doc")
    assert result.ok is False
    assert "model exploded" in result.text
