# 📄 LangChain PDF Summarizer

An end-to-end PDF summarizer built with **Streamlit + LangChain + Google Gemini**.

Upload a PDF, pick a summary style, and get a summary produced by a map-reduce
pipeline: the document is chunked by tokens, each chunk is summarized, then the
chunk summaries are merged into one cohesive result. Optional document-structure
analysis and key-quote extraction are available too.

## Status

🚧 Under construction — Phase 0 (environment and credential validation) complete.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then add your GEMINI_API_KEY
streamlit run app.py
```

Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Roadmap

| Phase | Scope |
| ----- | ----- |
| 0 | Environment, Gemini key validation, GitHub setup |
| 1 | Project skeleton and config |
| 2 | PDF extraction, cleaning, token-aware chunking |
| 3 | LangChain map-reduce summarization |
| 4 | Streamlit interface |
| 5 | Hardening, retries, tests |
| 6 | Documentation and v1.0.0 |
| 7 | Final verification |

## License

MIT
