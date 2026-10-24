"""Entrypoint: uvicorn app.main:app. Submit https://<host>/mailroom as the endpoint URL."""
import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import config, service
from .util import PROFILE

logging.basicConfig(level=logging.INFO)
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


def _resp(code, body):
    return Response(body, status_code=code, media_type="application/json")


async def handle(request: Request):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > config.MAX_BODY_BYTES:
        return _resp(413, json.dumps({"error": "body too large"}))
    raw = await request.body()
    if len(raw) > config.MAX_BODY_BYTES:
        return _resp(413, json.dumps({"error": "body too large"}))
    try:
        b = json.loads(raw)
        if not isinstance(b, dict):
            raise ValueError
    except Exception:
        return _resp(400, json.dumps({"error": "malformed JSON"}))
    if b.get("profile") != PROFILE:
        return _resp(400, json.dumps({"error": "unsupported profile"}))
    op = b.get("operation")
    if op == "propose":
        return _resp(*await service.propose(b))
    if op == "commit":
        return _resp(*await service.commit(b))
    return _resp(400, json.dumps({"error": "invalid operation"}))


app.add_api_route("/mailroom", handle, methods=["POST"])
app.add_api_route("/", handle, methods=["POST"])


@app.get("/healthz")
async def healthz():
    return {"ok": True}
