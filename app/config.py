"""All configuration comes from environment variables (set in Render, never in source)."""
import os


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, "").strip()
    return v or default


# Provider-neutral LLM settings (any OpenAI-compatible endpoint). Old AIPIPE_* names still work.
LLM_API_KEY = _env("LLM_API_KEY") or _env("AIPIPE_TOKEN")  # secret, no default
LLM_BASE_URL = (_env("LLM_BASE_URL") or _env("AIPIPE_BASE_URL")
                or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_MODEL = _env("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT")  # e.g. "low" for Gemini 2.5/3 (limits hidden thinking tokens)
DB_PATH = _env("DB_PATH", "mailroom.db")

MAX_OUTPUT_TOKENS = int(_env("MAX_OUTPUT_TOKENS", "6000"))  # per model call
CHUNK_SIZE = int(_env("CHUNK_SIZE", "6"))                  # dossiers per model call
MAX_CONCURRENCY = int(_env("MAX_CONCURRENCY", "10"))         # parallel model calls
REQUEST_BUDGET_S = 48                                       # grader allows 55s per request
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_DOSSIERS = 500
