"""LLM access through AI Pipe (OpenAI-compatible). Token comes from the AIPIPE_TOKEN env var."""
import httpx

from . import config


async def chat(system: str, user: str, timeout: float) -> str:
    if not config.AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN is not set")
    body = {"model": config.LLM_MODEL, "temperature": 0, "max_tokens": config.MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    async with httpx.AsyncClient(timeout=timeout) as c:
        for attempt in range(2):
            r = await c.post(config.AIPIPE_BASE_URL + "/chat/completions",
                             headers={"Authorization": f"Bearer {config.AIPIPE_TOKEN}"}, json=body)
            if r.status_code == 400 and attempt == 0 and "response_format" in r.text:
                body.pop("response_format")
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"]
