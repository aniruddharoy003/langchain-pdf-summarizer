"""LangChain map-reduce summarization backed by Google Gemini.

Flow:  chunks -> (map) per-chunk summaries -> (reduce) one combined summary.
The chain layer is thin on purpose; all prompt text lives in ``prompts.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    MAX_CONCURRENCY,
    MAX_OUTPUT_TOKENS,
    MAX_QUOTES,
    REQUEST_TIMEOUT_S,
    RETRY_ATTEMPTS,
    TEMPERATURE,
    THINKING_LEVEL,
    settings,
)
from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, style_instruction

logger = logging.getLogger(__name__)

_BULLET_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
_QUOTE_CHARS = " \"'“”‘’"


class SummarizerError(Exception):
    """Raised when the model cannot be configured or reached."""


@dataclass(frozen=True)
class ChunkSummary:
    chunk_number: int
    summary: str
    ok: bool
    original_chars: int
    summary_chars: int


@dataclass
class SummaryResult:
    summary_type: str
    total_chunks: int
    chunk_summaries: list[ChunkSummary]
    final_summary: str

    @property
    def succeeded(self) -> int:
        return sum(1 for c in self.chunk_summaries if c.ok)

    @property
    def failed(self) -> int:
        return self.total_chunks - self.succeeded

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total_chunks if self.total_chunks else 0.0


@dataclass(frozen=True)
class AnalysisResult:
    ok: bool
    text: str


def message_text(message: BaseMessage) -> str:
    """Plain text from a message, tolerating Gemini's list-of-blocks content.

    ``BaseMessage.text`` joins blocks with no separator, which can glue two
    paragraphs together, so list content is handled here with newlines.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            block_type = (
                block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            )
            block_text = (
                block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            )
            if block_type == "text" and isinstance(block_text, str):
                parts.append(block_text)
        return "\n".join(part for part in parts if part).strip()

    text = getattr(message, "text", None)
    return str(text).strip() if text is not None else ""


class PDFSummarizer:
    """Wraps Gemini in the chains needed by the map-reduce pipeline."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.model
        key = api_key if api_key is not None else settings.api_key
        if not key:
            raise SummarizerError(
                "GEMINI_API_KEY is not set. Add it to .env or pass api_key explicitly."
            )

        self._llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=key,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_level=THINKING_LEVEL,
            max_retries=RETRY_ATTEMPTS,
            timeout=REQUEST_TIMEOUT_S,
        )

        parser = RunnableLambda(message_text)
        self._chunk_chain = CHUNK_SUMMARY | self._llm | parser
        self._combine_chain = COMBINE | self._llm | parser
        self._analysis_chain = ANALYSIS | self._llm | parser
        self._quotes_chain = QUOTES | self._llm | parser

    # -- map ---------------------------------------------------------------
    def summarize_text(self, text: str, summary_type: str) -> str:
        """Summarize a single piece of text."""
        return self._chunk_chain.invoke({"style": style_instruction(summary_type), "text": text})

    def summarize_chunks(self, chunks: list[str], summary_type: str) -> SummaryResult:
        """Summarize every chunk concurrently, then reduce to one summary."""
        chunks = [c for c in chunks if c and c.strip()]
        if not chunks:
            raise SummarizerError("No text to summarize.")

        style = style_instruction(summary_type)
        payloads = [{"style": style, "text": chunk} for chunk in chunks]
        try:
            outputs = self._chunk_chain.batch(
                payloads,
                config={"max_concurrency": MAX_CONCURRENCY},
                return_exceptions=True,
            )
        except Exception as exc:  # noqa: BLE001 - configuration/auth problems
            raise SummarizerError(describe_error(exc)) from exc

        summaries: list[ChunkSummary] = []
        for number, (chunk, output) in enumerate(zip(chunks, outputs, strict=True), start=1):
            ok = not isinstance(output, BaseException)
            if ok:
                text = str(output).strip()
            else:
                logger.warning("Chunk %d failed: %s", number, output)
                text = f"Chunk {number} could not be summarized: {_short(output)}"
            summaries.append(
                ChunkSummary(
                    chunk_number=number,
                    summary=text,
                    ok=ok,
                    original_chars=len(chunk),
                    summary_chars=len(text) if ok else 0,
                )
            )

        return SummaryResult(
            summary_type=summary_type,
            total_chunks=len(chunks),
            chunk_summaries=summaries,
            final_summary=self._reduce(summaries, summary_type),
        )

    # -- reduce ------------------------------------------------------------
    def _reduce(self, summaries: list[ChunkSummary], summary_type: str) -> str:
        usable = [s.summary for s in summaries if s.ok]
        if not usable:
            return "No chunk could be summarized, so no summary was produced."
        if len(usable) == 1:
            return usable[0]

        sections = "\n\n".join(f"Section {i}: {text}" for i, text in enumerate(usable, start=1))
        try:
            return self._combine_chain.invoke(
                {"style": style_instruction(summary_type), "summaries": sections}
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Combine step failed: %s", exc)
            return "Combined summary (automatic merge failed, sections listed):\n\n" + sections

    # -- extras ------------------------------------------------------------
    def analyze_document(self, text: str) -> AnalysisResult:
        """Structural analysis; never raises."""
        try:
            return AnalysisResult(ok=True, text=self._analysis_chain.invoke({"text": text}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analysis failed: %s", exc)
            return AnalysisResult(ok=False, text=f"Could not analyze the document: {_short(exc)}")

    def extract_quotes(self, text: str) -> list[str]:
        """Key quotes, cleaned of numbering, bullets and quote marks."""
        response = self._quotes_chain.invoke({"text": text})
        quotes = [clean_quote_line(line) for line in response.splitlines()]
        return [q for q in quotes if q][:MAX_QUOTES]


def clean_quote_line(line: str) -> str:
    """Strip list markers and surrounding quotation marks from one line."""
    return _BULLET_PREFIX.sub("", line).strip().strip(_QUOTE_CHARS).strip()


def _short(exc: object, limit: int = 200) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def describe_error(exc: BaseException) -> str:
    """Turn provider exceptions into one actionable sentence."""
    raw = str(exc)
    lowered = raw.lower()

    if "429" in raw or "resource_exhausted" in lowered or "quota" in lowered:
        return (
            "Gemini rate limit or quota reached. Wait a minute and retry, use a "
            "smaller document, or pick another model in the sidebar."
        )
    if (
        any(token in lowered for token in ("api key", "permission denied", "unauthenticated"))
        or "401" in raw
        or "403" in raw
    ):
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env and restart."
    if "timeout" in lowered or "deadline" in lowered:
        return (
            "Gemini did not respond in time. Retry, or lower 'Tokens per chunk' so "
            "each request is smaller."
        )
    if "not found" in lowered or "404" in raw:
        return f"The model is not available to this key. Choose a different model. ({_short(raw)})"
    return _short(exc)
