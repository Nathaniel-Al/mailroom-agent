"""All configuration comes from environment variables (set in Render, never in source)."""
import os


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v or default


AIPIPE_TOKEN = _env("AIPIPE_TOKEN")  # secret, no default
AIPIPE_BASE_URL = _env("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
LLM_MODEL = _env("LLM_MODEL", "gpt-4.1-mini")
DB_PATH = _env("DB_PATH", "mailroom.db")

MAX_OUTPUT_TOKENS = int(_env("MAX_OUTPUT_TOKENS", "4000"))  # per model call
CHUNK_SIZE = int(_env("CHUNK_SIZE", "6"))                  # dossiers per model call
MAX_CONCURRENCY = int(_env("MAX_CONCURRENCY", "10"))         # parallel model calls
REQUEST_BUDGET_S = 48                                       # grader allows 55s per request
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_DOSSIERS = 500
