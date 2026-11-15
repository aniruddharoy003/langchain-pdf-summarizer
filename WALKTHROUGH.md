# Walkthrough

A guided tour of how this project is put together, in the order it was built.
Read this alongside the source; every section maps to one or two files.

---

## 0. Environment and credentials

Three things had to be true before writing any feature code:

1. **Dependencies resolve.** `requirements.txt` is fully pinned. Python 3.15
   needed wheels for `pandas` and `tiktoken`; both publish `cp315` builds, so no
   source builds were required.
2. **The Gemini key works.** A key can be well-formed and still be rejected, so
   `scripts/check_env.py` makes a real call through the same LangChain client the
   app uses. A string-length check proves nothing.
3. **Pushing works.** The repo, the git identity and the credential helper were
   all validated by pushing a scaffold commit before any app code existed. When
   something breaks later, you know it is the code, not the plumbing.

### The thinking discovery

Gemini 3 reasons before answering by default. On `gemini-3.1-flash-lite`, asking
for the single word `OK`:

| Configuration | Wall clock |
| --- | --- |
| default thinking | 52.6 s |
| `thinking_level="minimal"` | 4.1 s |
| `thinking_budget=0` (legacy) | 4.3 s |

The first attempt at key validation actually *timed out* at 60 s. The fix lives
in `config.py` as `THINKING_LEVEL = "minimal"` and is passed to
`ChatGoogleGenerativeAI` in `summarizer.py`.

For Gemini 3+, `thinking_budget` is deprecated in favour of `thinking_level`;
setting both logs a warning and the level wins.

---

## 1. Configuration (`config.py`)

One module owns every tunable: model list, chunk sizes, upload guards, concurrency,
retry counts, prompt prefixes. Nothing else calls `os.getenv`.

`Settings` is a frozen dataclass with a `key_problem` property that returns either
`None` or a sentence a human can act on. The UI renders that string directly, so
"what is wrong with my key" has exactly one implementation.

---

## 2. Ingestion (`pdf_processor.py`)

`PDFProcessor` is deliberately UI-agnostic: it accepts a path, an open file or a
Streamlit upload buffer, because all three are `seek`-able.

**Extraction** reads page by page. One unreadable page increments a counter and
logs a warning instead of aborting the run. If *every* page is empty, the error
message says "probably scanned images" — the single most common support question.

**Cleaning** is four regex passes: strip control characters, collapse horizontal
whitespace, remove spaces hugging newlines, collapse 3+ newlines to a blank line.
Punctuation and paragraph structure survive, which matters for bullet summaries.

**Chunking** works in two stages:

1. Split into sentences with `(?<=[.!?])\s+`.
2. Greedily pack sentences into chunks, hard-splitting any single sentence that
   already exceeds the budget.

Token counts come from `tiktoken`'s `cl100k_base` — an approximation of Gemini's
tokenizer, but a *consistent* one, which is what chunk sizing needs.

The `+1` in the packing loop accounts for the space inserted by `" ".join(...)`,
so a chunk never silently overshoots its budget.

`truncate()` exists because analysis and quote extraction only look at a prefix
of the document (`ANALYSIS_CHAR_LIMIT`, `QUOTES_CHAR_LIMIT`), and cutting by
tokens is safer than cutting by characters.

---

## 3. Prompts (`prompts.py`)

Everything the model is told lives here. A shared `SYSTEM` persona forbids
invention and preamble; `STYLES` maps each summary type to its instructions;
`_template()` wraps a human message into a `ChatPromptTemplate` with that system
message.

Four templates:

| Template | Inputs | Purpose |
| --- | --- | --- |
| `CHUNK_SUMMARY` | `style`, `text` | summarize one section |
| `COMBINE` | `style`, `summaries` | merge sections, remove repetition |
| `ANALYSIS` | `text` | five labelled structural questions |
| `QUOTES` | `text` | 5-10 verbatim quotes, one per line |

Because prompts are data, they are unit-testable: `tests/test_prompts.py` renders
each one and asserts the variables landed.

---

## 4. Chains (`summarizer.py`)

Each task is a chain: `prompt | llm | parser`. The parser is a small
`RunnableLambda(message_text)` rather than `StrOutputParser`, because Gemini 3
returns content as a list of blocks. `BaseMessage.text` joins blocks with an
empty string, which can glue two paragraphs into one word — so `message_text`
handles list content itself and joins with newlines.

**Map step.** `summarize_chunks` filters blank chunks, builds one payload per
chunk, and calls `.batch(..., config={"max_concurrency": 4}, return_exceptions=True)`.
`return_exceptions=True` is the important flag: a single 429 must not destroy the
other chunks' work. Each result becomes a `ChunkSummary` with `ok`, character
counts and the text.

**Reduce step.** `_reduce` collects the successful summaries. One summary is
returned as-is (no wasted call). Otherwise `COMBINE` merges them; if that call
fails, the sections are returned verbatim so the user still gets something.

**Error translation.** `describe_error()` turns provider noise into one sentence:
429/quota, rejected key, timeout, unknown model. `summarize_chunks` wraps
configuration-level failures in `SummarizerError`, and the UI shows that message.

---

## 5. UI (`app.py`, `utils.py`)

`app.py` is orchestration only:

- `@st.cache_resource` keeps the tokenizer and one chain set per model alive
  across reruns.
- The sidebar returns a plain `options` dict.
- Results are stored in `st.session_state` under a **signature** of
  `(file_id, model, summary_type, max_tokens, analysis, quotes)`. Clicking a
  download button reruns the script; without this cache, every click would
  re-summarize the document and re-bill the API. Changing a setting changes the
  signature and correctly invalidates the old result.
- `st.status` reports each stage, so a 30-second wait shows progress instead of
  a spinner.

`utils.py` holds everything that renders or exports: metrics, per-chunk expander,
`.txt` report, `.csv` of chunk statistics. These are pure functions over
`SummaryResult`, which is why they are tested without a browser.

---

## 6. Hardening and tests

The suite (37 tests, no network) is layered:

| File | What it protects |
| --- | --- |
| `test_pdf_processor.py` | cleaning, token limits, hard-splitting, bad files, metadata |
| `test_prompts.py` | template rendering, style fallback |
| `test_summarizer.py` | block parsing, quote cleaning, batch results, failure isolation |
| `test_utils.py` | dataframe columns, export content, time estimate |
| `test_app_smoke.py` | `AppTest` runs the real `app.py` headlessly |

`FakeListChatModel` lets the summarizer be tested without a key: canned responses
flow through the real prompts and the real parser.

---

## 7. Release

`README.md` documents setup, architecture, configuration and troubleshooting.
`ruff` enforces lint and formatting (`pyproject.toml`), `pytest` enforces
behaviour, and the sample PDF generator keeps the whole thing reproducible with
zero extra dependencies.

Tagged `v1.0.0`.
