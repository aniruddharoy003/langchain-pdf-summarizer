"""Prompt templates for the summarizer.

Everything the model is told lives here: one shared system persona, the four
style instructions, and the four task templates. Keeping prompts out of the
plumbing makes them easy to iterate on and to test.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from config import DEFAULT_SUMMARY_TYPE

SYSTEM = (
    "You are a precise document summarizer. Use only information contained in the "
    "provided text; never invent facts, numbers or names. Write in clear English. "
    "Respond with the requested content only: no preamble, no closing remarks."
)

STYLES: dict[str, str] = {
    "brief": "A 2-3 sentence overview focusing on the main point and key takeaways.",
    "detailed": (
        "A detailed summary covering:\n"
        "- Main topics and themes\n"
        "- Key arguments or findings\n"
        "- Important details and examples\n"
        "- Conclusions or recommendations"
    ),
    "bullet_points": (
        "A scannable bullet list. Group related points together and use "
        "sub-bullets where it aids clarity."
    ),
    "executive": (
        "An executive summary for decision makers, covering:\n"
        "- Key insights and findings\n"
        "- Strategic implications\n"
        "- Actionable recommendations\n"
        "- Risks and considerations"
    ),
}


def style_instruction(summary_type: str) -> str:
    """Instruction text for a summary style, falling back to the default."""
    return STYLES.get(summary_type, STYLES[DEFAULT_SUMMARY_TYPE])


def _template(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


CHUNK_SUMMARY = _template(
    "Summarize the section below in this style:\n{style}\n\n<section>\n{text}\n</section>"
)

COMBINE = _template(
    "Below are summaries of consecutive sections of a single document. Merge them "
    "into one cohesive summary in this style:\n{style}\n\n"
    "Remove repetition, keep the original ordering of ideas, and do not add "
    "information that is not present in the section summaries.\n\n"
    "<section_summaries>\n{summaries}\n</section_summaries>"
)

ANALYSIS = _template(
    "Analyze the structure and content of the document below. Answer in five short "
    "labelled sections:\n\n"
    "1. Document type\n"
    "2. Main topics and themes\n"
    "3. Key sections or chapters\n"
    "4. Intended audience\n"
    "5. Overall purpose\n\n"
    "<document>\n{text}\n</document>"
)

QUOTES = _template(
    "Extract 5-10 key quotes that best represent the main ideas of the document "
    "below. Copy each quote verbatim, joining line-wrapped text into a single line. "
    "Output one quote per line with no numbering, no bullets and no surrounding "
    "quotation marks.\n\n"
    "<document>\n{text}\n</document>"
)
