"""Central configuration for the PDF summarizer.

Every tunable lives here so the modules below stay free of magic numbers and of
``os.getenv`` calls. Import ``settings`` for the resolved values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

# ``override=False`` keeps real environment variables (CI, Streamlit secrets)
# ahead of the local .env file.
load_dotenv(ENV_FILE, override=False)

# --- App identity -----------------------------------------------------------
APP_TITLE = "PDF AI Summarizer"
APP_ICON = "📄"
APP_TAGLINE = "Upload a PDF and get an intelligent summary powered by Google Gemini."

# --- Models -----------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.1-flash-lite"
AVAILABLE_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)

# --- Summarization ----------------------------------------------------------
SUMMARY_TYPES = ("brief", "detailed", "bullet_points", "executive")
DEFAULT_SUMMARY_TYPE = "detailed"

TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 4096
# Gemini 3 models reason by default. Measured on gemini-3.1-flash-lite: a
# one-word reply took 52.6s with default thinking vs 4.1s with "minimal".
# These summarization prompts need no chain-of-thought, so keep it minimal.
THINKING_LEVEL = "minimal"
MAX_CONCURRENCY = 4

# --- Chunking ---------------------------------------------------------------
DEFAULT_CHUNK_TOKENS = 8000
MIN_CHUNK_TOKENS = 2000
MAX_CHUNK_TOKENS = 10000

# --- Input guards -----------------------------------------------------------
MAX_UPLOAD_MB = 25
MAX_PAGES = 300

# --- Analysis / quotes ------------------------------------------------------
ANALYSIS_CHAR_LIMIT = 3000
QUOTES_CHAR_LIMIT = 5000
MAX_QUOTES = 10

# --- Networking / resilience ------------------------------------------------
REQUEST_TIMEOUT_S = 120
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 2.0


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings."""

    api_key: str | None
    model: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def key_problem(self) -> str | None:
        """Human-readable reason the key is unusable, or ``None`` if it is fine."""
        if not self.api_key:
            return "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        if len(self.api_key) < 20:
            return "GEMINI_API_KEY looks truncated. Create a new one at aistudio.google.com/apikey."
        return None


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def get_settings() -> Settings:
    """Read the environment once and return resolved settings."""
    model = _clean(os.getenv("GEMINI_MODEL")) or DEFAULT_MODEL
    return Settings(api_key=_clean(os.getenv("GEMINI_API_KEY")), model=model)


settings = get_settings()
