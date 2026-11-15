"""Generate a multi-page sample PDF without third-party dependencies.

Usage:
    python scripts/make_sample_pdf.py [output.pdf]

Writes a small, text-based (non-scanned) report so the extraction, cleaning and
chunking pipeline can be exercised end to end. The file is intentionally built
with raw PDF syntax to keep the project dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
MARGIN_X, TOP_Y = 72, 720
LEADING = 14
LINES_PER_PAGE = 44

TITLE = "The Quiet Rise of Small Language Models"
AUTHOR = "Shiva Roy"
SUBJECT = "Technical report on efficient on-device language models"

BODY = """\
Executive summary

Large language models dominate the headlines, but a quieter shift is underway.
Smaller models, typically between one and ten billion parameters, now handle a
large share of production workloads. They are cheaper to serve, easier to
fine-tune, and increasingly capable of running on consumer hardware.

This report examines where small models fit, what they give up, and how teams
should decide between a small specialized model and a large general one.

1. Why small models matter

Serving cost is the first argument. Inference cost scales roughly with the
number of active parameters, so a three billion parameter model can be an order
of magnitude cheaper per token than a frontier model. For high-volume tasks such
as classification, extraction and short-form summarization, that difference
dominates the total budget.

Latency is the second argument. A small model answers in a few hundred
milliseconds on a single accelerator, which makes interactive products feel
instant. When a user waits for a reply, perceived speed matters as much as
answer quality.

Privacy is the third argument. A model that runs on a laptop or a phone never
sends the document to a third party. Regulated industries increasingly prefer
this arrangement because it removes a whole class of compliance questions.

2. Where small models lose

Breadth of knowledge is the obvious gap. A large model has seen more text, so
it answers obscure factual questions more reliably. Small models compensate by
specializing, but a specialized model fails gracefully only inside its domain.

Long-horizon reasoning is the second gap. Multi-step planning, complex code
refactoring and careful mathematical derivation still favour larger models.
The gap narrows each year, yet it has not closed.

Instruction following under ambiguity is the third gap. Large models infer
intent from messy prompts more gracefully. Small models need cleaner prompts
and stricter output schemas.

3. A practical decision procedure

Start by writing down the task, the acceptable error rate and the volume. If
the task is narrow and the volume is high, a small fine-tuned model is usually
the right default. If the task requires broad world knowledge or long chains of
reasoning, use a large model and invest in retrieval instead.

Measure before committing. Build a small evaluation set of one hundred real
examples, score both a small and a large model, and compare cost per correct
answer rather than cost per token. The cheapest model that clears the quality
bar wins.

Design for substitution. Keep the model behind a thin interface so it can be
swapped without rewriting the product. Models improve quickly, and today's
compromise is next year's default.

4. Case study: document summarization

Document summarization is a good example of a task that splits cleanly. Chunk
level summarization is mechanical: extract the main claims from a few thousand
tokens. A small model handles this well and the work parallelizes naturally.

Final synthesis is harder. Merging many partial summaries without repetition or
contradiction benefits from a stronger model. A common pattern is therefore a
small model for the map step and a large model for the reduce step, which keeps
most of the cost saving while preserving coherence.

5. Risks and open questions

Evaluation remains the weakest link. Public benchmarks saturate quickly and
rarely reflect a specific product's failure modes. Teams that skip building an
internal evaluation set tend to discover regressions in production.

Tooling fragmentation is a second risk. Every provider exposes a slightly
different API, and abstractions leak when features such as structured output or
reasoning budgets differ. Portability requires discipline and tests.

Finally, the environmental story is mixed. Smaller models use less energy per
request, but higher request volume can erase the gain. Efficiency claims should
be measured per useful outcome, not per token.

6. Recommendations

Adopt small models as the default for high-volume, narrow tasks. Reserve large
models for synthesis, ambiguity and long-horizon reasoning. Instrument cost and
quality together. Re-evaluate the split every quarter, because the boundary
between small and large capability keeps moving.

Conclusion

The interesting question is no longer whether small models are useful, but how
quickly a team can build the evaluation harness that tells them which model to
use. Teams that measure per correct answer, and that keep their model choice
behind a clean interface, will capture most of the available savings without
paying for it in quality.
"""


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _wrap(paragraph: str, width: int = 88) -> list[str]:
    words, lines, line = paragraph.split(), [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def _paginate() -> list[list[str]]:
    """Lay the report out into pages of at most LINES_PER_PAGE lines."""
    lines: list[str] = []
    for paragraph in BODY.strip().split("\n\n"):
        lines.extend(_wrap(paragraph))
        lines.append("")  # paragraph gap

    pages, page = [], []
    for line in lines:
        page.append(line)
        if len(page) == LINES_PER_PAGE:
            pages.append(page)
            page = []
    if page:
        pages.append(page)
    return pages


def _content_stream(lines: list[str]) -> bytes:
    parts = ["BT", "/F1 11 Tf", f"{LEADING} TL", f"1 0 0 1 {MARGIN_X} {TOP_Y} Tm"]
    for line in lines:
        if line:
            parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


class _Builder:
    """Minimal PDF object table with stable numbering."""

    def __init__(self) -> None:
        self.objects: list[bytes] = []

    def reserve(self) -> int:
        self.objects.append(b"")
        return len(self.objects)

    def add(self, body: bytes) -> int:
        self.objects.append(body)
        return len(self.objects)

    def set(self, number: int, body: bytes) -> None:
        self.objects[number - 1] = body

    def render(self, root: int, info: int) -> bytes:
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, body in enumerate(self.objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

        xref_offset = len(out)
        size = len(self.objects) + 1
        out += f"xref\n0 {size}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            out += f"{offset:010d} 00000 n \n".encode("latin-1")
        out += (
            f"trailer\n<< /Size {size} /Root {root} 0 R /Info {info} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
        return bytes(out)


def build_pdf() -> bytes:
    pages_text = _paginate()
    builder = _Builder()

    catalog = builder.reserve()
    pages_node = builder.reserve()
    font = builder.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    info = builder.add(
        f"<< /Title ({_escape(TITLE)}) /Author ({_escape(AUTHOR)}) "
        f"/Subject ({_escape(SUBJECT)}) /Producer (make_sample_pdf.py) >>".encode("latin-1")
    )

    kids: list[int] = []
    for lines in pages_text:
        page_obj = builder.reserve()
        content = builder.add(
            f"<< /Length {len(_content_stream(lines))} >>\nstream\n".encode("latin-1")
            + _content_stream(lines)
            + b"\nendstream"
        )
        builder.set(
            page_obj,
            (
                f"<< /Type /Page /Parent {pages_node} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> "
                f"/Contents {content} 0 R >>"
            ).encode("latin-1"),
        )
        kids.append(page_obj)

    builder.set(
        pages_node,
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
        f"/Count {len(kids)} >>".encode("latin-1"),
    )
    builder.set(catalog, f"<< /Type /Catalog /Pages {pages_node} 0 R >>".encode("latin-1"))
    return builder.render(root=catalog, info=info)


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "tests" / "sample.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_pdf())
    print(f"wrote {target} ({target.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
