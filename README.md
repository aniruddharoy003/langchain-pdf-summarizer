# 📄 LangChain PDF Summarizer

An end-to-end PDF summarizer built with **Streamlit + LangChain + Google Gemini**.

Upload a PDF, choose a summary style, and get a summary produced by a map-reduce
pipeline: the document is split into token-sized chunks, each chunk is summarized,
and the chunk summaries are merged into one cohesive result. Optional document
analysis and key-quote extraction are available too.

```
PDF → extract → clean → chunk by tokens
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         summary 1       summary 2  …    summary N     ← map (concurrent)
              └───────────────┼───────────────┘
                              ▼
                    combine → final summary           ← reduce
```

## Features

- **Map-reduce summarization** — long documents are handled without truncation
- **Four summary styles** — brief, detailed, bullet points, executive
- **Document analysis** — type, themes, sections, audience, purpose
- **Key quotes** — 5-10 verbatim quotes, cleaned of list markers
- **Token-aware chunking** — sentence boundaries respected, oversized sentences hard-split
- **Parallel chunk calls** — configurable concurrency, per-chunk failure isolation
- **Exports** — final summary as `.txt`, per-chunk statistics as `.csv`
- **No re-billing** — results are cached per file *and* per setting in session state
- **Friendly errors** — rate limits, bad keys, timeouts and missing models become actionable messages
- **37 offline tests** — the whole pipeline is testable without an API key

## Requirements

- Python 3.10+
- A Google AI Studio API key — free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
copy .env.example .env           # then add your key
```

`.env`:

```ini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

Verify everything is wired correctly:

```bash
python scripts/check_env.py
```

```
env file     : .env (found)
model        : gemini-3.1-flash-lite
api key      : set
thinking     : minimal

calling model ...
reply        : 'OK' (1.4s)

PASS: environment is ready
```

## Run

```bash
streamlit run app.py
```

Or use the headless CLI — handy for testing without a browser:

```bash
python scripts/summarize.py report.pdf --type detailed
python scripts/summarize.py report.pdf --type executive --max-tokens 300 --analysis --quotes
```

## Configuration

`.env` holds secrets; everything else is a constant in [`config.py`](config.py).

| Setting | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | required |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | any model supporting `generateContent` |
| `THINKING_LEVEL` | `minimal` | see performance note below |
| `DEFAULT_CHUNK_TOKENS` | `8000` | slider range 2000-10000 |
| `MAX_CONCURRENCY` | `4` | parallel chunk requests |
| `MAX_UPLOAD_MB` | `25` | upload guard |
| `MAX_PAGES` | `300` | page guard |
| `MAX_OUTPUT_TOKENS` | `4096` | per response |

### Performance note: thinking is expensive

Gemini 3 models reason before answering by default. For mechanical summarization
that is pure latency. Measured on `gemini-3.1-flash-lite`, a one-word reply:

| Setting | Latency |
| --- | --- |
| default (thinking on) | **52.6 s** |
| `thinking_level="minimal"` | **4.1 s** |

The app sets `thinking_level="minimal"`, which is roughly a 10× speed-up on
short calls and still produces full-quality summaries.

## Project structure

```
app.py                  Streamlit UI: sidebar, upload, progress, results, exports
config.py               every tunable and the resolved .env settings
pdf_processor.py        extraction, cleaning, token counting, chunking, metadata
prompts.py              system persona, four styles, four ChatPromptTemplates
summarizer.py           LangChain chains: map, reduce, analysis, quotes
utils.py                rendering, text/CSV export, dataframe, time estimate
scripts/check_env.py    API key + model health check
scripts/summarize.py    CLI version of the pipeline
scripts/make_sample_pdf.py  dependency-free sample PDF generator
tests/                  pytest suite (no network required)
```

## How it works

1. **Extract** — `pypdf` reads every page; a broken page is skipped, not fatal.
2. **Clean** — control characters and repeated whitespace are removed, paragraphs kept.
3. **Chunk** — text is split on sentence boundaries into ≤ *n* `cl100k_base` tokens;
   a single sentence longer than the budget is hard-split by tokens.
4. **Map** — `CHUNK_SUMMARY` → Gemini → text, batched with `max_concurrency=4`.
   A failing chunk is recorded as failed; the run continues.
5. **Reduce** — `COMBINE` merges the section summaries; if it fails, the sections
   are returned verbatim rather than losing the work.
6. **Extras** — `ANALYSIS` and `QUOTES` run against a truncated prefix of the document.

Prompts live in [`prompts.py`](prompts.py) as `ChatPromptTemplate`s. To add a style:
add an entry to `STYLES` and to `SUMMARY_TYPES` in `config.py` — the sidebar picks it up.

## Development

```bash
pip install -r requirements-dev.txt
pytest                 # 37 tests, no network
ruff check .           # lint
ruff format --check .  # formatting
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GEMINI_API_KEY is not set` | Copy `.env.example` to `.env`, add the key, restart the app |
| Rate limit / quota (429) | Wait a minute, use a smaller document, or switch model |
| Model not found (404) | Your key cannot access it — pick another model in the sidebar |
| "No extractable text found" | The PDF is scanned images; run OCR first |
| Requests feel slow | Lower *Tokens per chunk*; check `THINKING_LEVEL` is `minimal` |
| `PDF has N pages; the limit is 300` | Split the document or raise `MAX_PAGES` |

## Roadmap

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Environment, Gemini key validation, GitHub setup | ✅ |
| 1 | Project skeleton and config | ✅ |
| 2 | PDF extraction, cleaning, token-aware chunking | ✅ |
| 3 | LangChain map-reduce summarization | ✅ |
| 4 | Streamlit interface | ✅ |
| 5 | Hardening, retries, tests | ✅ |
| 6 | Documentation and v1.0.0 | ✅ |
| 7 | Final verification | ✅ |

## License

MIT — see [LICENSE](LICENSE).
