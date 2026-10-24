"""propose / commit protocol logic. Returns (http_status, body_str)."""
import asyncio
import json
import logging
import time

from . import config, decider, store
from .util import PROFILE, canon, import_jwk, proposal_digest, sha, verify_receipt

log = logging.getLogger("service")
_locks: dict = {}


def _lock(k):
    return _locks.setdefault(k, asyncio.Lock())


def _err(code, msg):
    return code, json.dumps({"error": msg})


def _ok(obj):
    return 200, json.dumps(obj, ensure_ascii=False)


def _is_str(x):
    return isinstance(x, str) and x != ""


def validate_propose(b):
    if not _is_str(b.get("evaluationId")):
        return "evaluationId required"
    v = b.get("receiptVerifier")
    if not isinstance(v, dict) or v.get("algorithm") != "Ed25519":
        return "receiptVerifier invalid"
    try:
        import_jwk(v.get("publicKeyJwk"))
    except Exception:
        return "publicKeyJwk invalid"
    al = b.get("allowedActions")
    if not isinstance(al, list) or not al or not all(a in decider.ACTIONS for a in al):
        return "allowedActions invalid"
    ds = b.get("dossiers")
    if not isinstance(ds, list) or not ds or len(ds) > config.MAX_DOSSIERS:
        return "dossiers invalid"
    seen = set()
    for d in ds:
        if not isinstance(d, dict) or not _is_str(d.get("dossierId")) or d["dossierId"] in seen:
            return "dossier invalid or duplicate"
        seen.add(d["dossierId"])
        if not isinstance(d.get("sources"), list) or not d["sources"]:
            return "sources invalid"
        for s in d["sources"]:
            if not isinstance(s, dict) or not isinstance(s.get("lines"), list):
                return "source invalid"
            for ln in s["lines"]:
                if not isinstance(ln, dict) or not _is_str(ln.get("lineId")) or not isinstance(ln.get("text"), str):
                    return "line invalid"
    return None


async def propose(b):
    bad = validate_propose(b)
    if bad:
        return _err(400, bad)
    eid, req_hash = b["evaluationId"], sha(canon(b))
    deadline = time.monotonic() + config.REQUEST_BUDGET_S
    async with _lock(eid):
        row = store.get_eval(eid)
        if row:
            if row["req_hash"] != req_hash:
                return _err(409, "evaluationId reused with different content")
            return 200, row["propose_resp"]  # exact replay: no model work
        ds = b["dossiers"]
        kinds = sorted({(s.get("kind"), s.get("provenance")) for d in ds for s in d["sources"]}, key=str)
        log.info("propose %d dossiers; (kind, provenance) seen: %s", len(ds), kinds)
        decisions = await decider.decide(ds, b["allowedActions"], deadline)
        proposals = []
        for d, dec in zip(ds, decisions):
            proposals.append({"dossierId": d["dossierId"],
                              "callId": "call_" + sha(canon(d))[:40],
                              "action": dec["action"], "target": dec["target"],
                              "payload": dec["payload"], "evidence": dec["evidence"]})
        digest = sha(canon(ds))
        resp = {"profile": PROFILE, "evaluationId": eid, "status": "awaiting_receipts",
                "inputDigest": digest, "proposals": proposals}
        body = json.dumps(resp, ensure_ascii=False)
        store.insert_eval(eid, req_hash, digest, b["receiptVerifier"], body, proposals)  # persist before replying
        return 200, body


def _validate_commit(b, row):
    if not isinstance(b.get("inputDigest"), str) or b["inputDigest"] != row["input_digest"]:
        return 400, "inputDigest mismatch"
    rs = b.get("receipts")
    props = {p["dossierId"]: p for p in row["proposals"]}
    if not isinstance(rs, list) or len(rs) != len(props):
        return 400, "receipts must match proposals"
    seen_d, seen_c, seen_r = set(), set(), set()
    for r in rs:
        if not isinstance(r, dict):
            return 400, "receipt invalid"
        p = props.get(r.get("dossierId"))
        if (not p or r["dossierId"] in seen_d or r.get("callId") != p["callId"] or r.get("callId") in seen_c
                or r.get("action") != p["action"] or r.get("proposalDigest") != proposal_digest(p)
                or not isinstance(r.get("accepted"), bool) or not _is_str(r.get("receiptId"))
                or r["receiptId"] in seen_r or not _is_str(r.get("receiptSignature"))):
            return 400, "receipt does not match proposal"
        seen_d.add(r["dossierId"]); seen_c.add(r["callId"]); seen_r.add(r["receiptId"])
    for r in rs:  # verify EVERY signature before recording any effect
        if not verify_receipt(row["verifier"]["publicKeyJwk"], b["evaluationId"], b["inputDigest"], r):
            return 403, "receipt signature invalid"
    return None


async def commit(b):
    eid = b.get("evaluationId")
    if not _is_str(eid):
        return _err(400, "evaluationId required")
    async with _lock(eid):
        row = store.get_eval(eid)
        if not row:
            return _err(404, "unknown evaluation")
        h = sha(canon(b))
        if row["commit_resp"]:
            if row["commit_hash"] == h:
                return 200, row["commit_resp"]  # exact replay: no repeated effect
            return _err(409, "evaluation already committed with different content")
        bad = _validate_commit(b, row)
        if bad:
            return _err(*bad)
        by = {r["dossierId"]: r for r in b["receipts"]}
        outcomes = []
        for p in row["proposals"]:
            r = by[p["dossierId"]]
            outcomes.append({"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"],
                             "proposalDigest": r["proposalDigest"], "receiptId": r["receiptId"],
                             "status": "executed" if r["accepted"] is True else "rejected"})
        body = json.dumps({"profile": PROFILE, "evaluationId": eid, "status": "completed",
                           "inputDigest": row["input_digest"], "outcomes": outcomes}, ensure_ascii=False)
        store.set_commit(eid, h, body)  # persist before replying
        return 200, body
