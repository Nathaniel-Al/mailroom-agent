"""LLM access through any OpenAI-compatible endpoint (Gemini, AI Pipe, OpenAI...). Key comes from LLM_API_KEY."""
import asyncio
import logging
import time

import httpx

from . import config

log = logging.getLogger("llm")
_gate = asyncio.Lock()
_state = {"next": 0.0}


async def _throttle(t_end):
    """Space model calls out globally (all evaluations share the provider's per-minute quota)."""
    async with _gate:
        now = time.monotonic()
        start = max(now, _state["next"])
        _state["next"] = start + config.LLM_MIN_INTERVAL
    wait = start - now
    if wait > 0:
        if wait > t_end - now - 6:
            raise TimeoutError("LLM throttle wait exceeds budget")
        await asyncio.sleep(wait)
RETRY = (429, 500, 502, 503, 504)


async def chat(system: str, user: str, timeout: float) -> str:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set")
    t_end = time.monotonic() + timeout
    body = {"model": config.LLM_MODEL, "temperature": 0, "max_tokens": config.MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if config.LLM_REASONING_EFFORT:
        body["reasoning_effort"] = config.LLM_REASONING_EFFORT
    async with httpx.AsyncClient() as c:
        for attempt in range(5):
            left = t_end - time.monotonic()
            if left < 3:
                raise TimeoutError("LLM time budget exhausted")
            await _throttle(t_end)
            left = t_end - time.monotonic()
            r = await c.post(config.LLM_BASE_URL + "/chat/completions",
                             headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, json=body, timeout=left)
            if r.status_code == 400:
                dropped = False
                for k in ("response_format", "reasoning_effort"):  # provider rejects an optional field: drop it
                    if k in body and k in r.text:
                        body.pop(k)
                        dropped = True
                if dropped:
                    continue
            if r.status_code in RETRY and attempt < 4:
                try:
                    wait = float(r.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0
                wait = wait or min(2.0 * (attempt + 1), 8.0)
                if wait < t_end - time.monotonic() - 3:
                    log.warning("LLM HTTP %s, retrying in %.1fs", r.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"].get("content") or ""
    raise RuntimeError("LLM call failed")
