"""Streamlit UI for the PDF summarizer.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import (
    ANALYSIS_CHAR_LIMIT,
    APP_ICON,
    APP_TAGLINE,
    APP_TITLE,
    AVAILABLE_MODELS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_SUMMARY_TYPE,
    MAX_CHUNK_TOKENS,
    MAX_UPLOAD_MB,
    MIN_CHUNK_TOKENS,
    QUOTES_CHAR_LIMIT,
    SUMMARY_TYPES,
    settings,
)
from pdf_processor import PDFError, PDFProcessor
from summarizer import PDFSummarizer, SummarizerError
from utils import (
    estimate_processing_time,
    export_summary_text,
    key_problem,
    render_result,
    summary_dataframe,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


@st.cache_resource(show_spinner=False)
def get_processor() -> PDFProcessor:
    """One tokenizer per server process."""
    return PDFProcessor()


@st.cache_resource(show_spinner=False)
def get_summarizer(model: str) -> PDFSummarizer:
    """One chain set per model; Streamlit keeps it alive across reruns."""
    return PDFSummarizer(model=model)


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Settings")

    models = list(AVAILABLE_MODELS)
    if settings.model not in models:
        models.insert(0, settings.model)

    model = st.sidebar.selectbox(
        "Model",
        models,
        index=models.index(settings.model),
        help="Gemini model used for every LLM call.",
    )
    summary_type = st.sidebar.selectbox(
        "Summary type",
        SUMMARY_TYPES,
        index=SUMMARY_TYPES.index(DEFAULT_SUMMARY_TYPE),
        format_func=lambda value: value.replace("_", " ").title(),
    )
    max_tokens = st.sidebar.slider(
        "Tokens per chunk",
        MIN_CHUNK_TOKENS,
        MAX_CHUNK_TOKENS,
        DEFAULT_CHUNK_TOKENS,
        step=500,
        help="Smaller chunks mean more calls but tighter, more focused summaries.",
    )
    show_analysis = st.sidebar.checkbox(
        "Document analysis", value=True, help="Describe type, themes and audience."
    )
    show_quotes = st.sidebar.checkbox(
        "Key quotes", value=False, help="Extract 5-10 verbatim quotes."
    )

    st.sidebar.divider()
    st.sidebar.caption(f"🔑 Key loaded · default model `{settings.model}`")

    return {
        "model": model,
        "summary_type": summary_type,
        "max_tokens": max_tokens,
        "analysis": show_analysis,
        "quotes": show_quotes,
    }


def run_pipeline(uploaded, options: dict) -> dict:
    """Extract → chunk → summarize (→ analyze / quotes). Raises PDFError."""
    processor = get_processor()

    with st.status("Reading the PDF…", expanded=True) as status:
        metadata = processor.metadata(uploaded)
        text = processor.clean_text(processor.extract_text(uploaded))
        token_count = processor.count_tokens(text)
        st.write(
            f"📖 {metadata.num_pages} page(s), {len(text):,} characters "
            f"(~{token_count:,} tokens)"
        )

        chunks = processor.chunk_text(text, options["max_tokens"])
        st.write(f"✂️ {len(chunks)} chunk(s) · {estimate_processing_time(len(chunks))}")

        try:
            summarizer = get_summarizer(options["model"])
        except SummarizerError as exc:
            status.update(label="Model unavailable", state="error")
            raise PDFError(str(exc)) from exc

        status.update(label=f"Summarizing {len(chunks)} chunk(s)…")
        result = summarizer.summarize_chunks(chunks, options["summary_type"])
        st.write(f"🤖 {result.succeeded}/{result.total_chunks} chunk(s) summarized")

        analysis = None
        if options["analysis"]:
            status.update(label="Analyzing document structure…")
            outcome = summarizer.analyze_document(
                processor.truncate(text, ANALYSIS_CHAR_LIMIT)
            )
            if outcome.ok:
                analysis = outcome.text
            else:
                st.warning("Document analysis failed; continuing without it.")

        quotes = None
        if options["quotes"]:
            status.update(label="Extracting key quotes…")
            try:
                quotes = summarizer.extract_quotes(
                    processor.truncate(text, QUOTES_CHAR_LIMIT)
                )
            except Exception:  # noqa: BLE001 - quotes are a nice-to-have
                st.warning("Quote extraction failed; continuing without it.")

        status.update(label="Done", state="complete", expanded=False)

    return {
        "metadata": metadata,
        "result": result,
        "analysis": analysis,
        "quotes": quotes,
        "characters": len(text),
        "tokens": token_count,
    }


def render_metadata(metadata) -> None:
    left, right = st.columns(2)
    left.info(f"📄 Pages: {metadata.num_pages}")
    left.info(f"📝 Title: {metadata.title}")
    right.info(f"👤 Author: {metadata.author}")
    right.info(f"🏷️ Subject: {metadata.subject}")


def render_downloads(state: dict) -> None:
    result, file_name = state["result"], state["file_name"]
    st.subheader("📤 Export")
    left, right = st.columns(2)
    left.download_button(
        "⬇️ Summary (.txt)",
        data=export_summary_text(result, file_name),
        file_name=f"{file_name}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    frame = summary_dataframe(result)
    right.download_button(
        "⬇️ Chunk stats (.csv)",
        data=frame.to_csv(index=False),
        file_name=f"{file_name}_chunks.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("📈 Processing statistics"):
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if not frame.empty:
            st.caption(
                f"Average compression: {frame['Compression'].mean():.2f} · "
                f"success rate: {(frame['Status'] == 'Success').mean() * 100:.0f}%"
            )


def main() -> None:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(APP_TAGLINE)

    problem = key_problem()
    if problem:
        st.error(f"❌ {problem}")
        st.info("Get a free key at https://aistudio.google.com/apikey, then restart the app.")
        st.stop()

    options = render_sidebar()

    uploaded = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded is None:
        st.info("Upload a text-based PDF to get started. Scanned PDFs need OCR first.")
        return

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"❌ {uploaded.name} is larger than the {MAX_UPLOAD_MB} MB limit.")
        return

    # Results are keyed by file so downloads and reruns do not re-bill the API.
    file_id = uploaded.file_id
    if st.session_state.get("file_id") != file_id:
        st.session_state.file_id = file_id
        st.session_state.pop("results", None)

    st.success(f"✅ {uploaded.name} ({uploaded.size / 1024:.0f} KB)")

    try:
        metadata = get_processor().metadata(uploaded)
    except PDFError as exc:
        st.error(f"❌ {exc}")
        return
    render_metadata(metadata)

    if st.button("🚀 Summarize", type="primary"):
        st.session_state.pop("results", None)
        try:
            state = run_pipeline(uploaded, options)
        except PDFError as exc:
            st.error(f"❌ {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - keep the UI alive
            st.error(f"❌ Unexpected error: {exc}")
            return
        state["file_name"] = uploaded.name
        st.session_state.results = state

    state = st.session_state.get("results")
    if state:
        render_result(state["result"], state["analysis"], state["quotes"])
        render_downloads(state)


if __name__ == "__main__":
    main()
