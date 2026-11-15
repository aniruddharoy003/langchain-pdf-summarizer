"""Environment health check.

Verifies the API key and the configured model by making one real call through
the same LangChain client the app uses. Run it after editing .env:

    python scripts/check_env.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow ``python scripts/check_env.py`` to import the project modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

from config import (  # noqa: E402
    ENV_FILE,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    THINKING_LEVEL,
    settings,
)


def main() -> int:
    print(f"env file     : {ENV_FILE} ({'found' if ENV_FILE.exists() else 'MISSING'})")
    print(f"model        : {settings.model}")
    print(f"api key      : {'set' if settings.has_api_key else 'MISSING'}")

    if settings.key_problem:
        print(f"\nFAIL: {settings.key_problem}")
        return 1

    print(f"thinking     : {THINKING_LEVEL}")
    print("\ncalling model ...")

    started = time.time()
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=settings.api_key,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_level=THINKING_LEVEL,
        )
        reply = llm.invoke("Reply with the single word: OK")
    except Exception as exc:  # noqa: BLE001 - this script reports, it does not raise
        print(f"FAIL: {type(exc).__name__}: {str(exc)[:400]}")
        return 1

    elapsed = time.time() - started
    print(f"reply        : {reply.text!r} ({elapsed:.1f}s)")
    print("\nPASS: environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
