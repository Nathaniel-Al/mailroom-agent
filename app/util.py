"""Canonical JSON, hashing and Ed25519 receipt verification."""
import base64
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

PROFILE = "ga5-mailroom-action-gate/v2"


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def b64d(s: str) -> bytes:
    s = s.strip()
    pad = "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s + pad, validate=True)
    except Exception:
        return base64.urlsafe_b64decode(s + pad)


def import_jwk(jwk) -> Ed25519PublicKey:
    if not isinstance(jwk, dict) or jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" \
            or not isinstance(jwk.get("x"), str):
        raise ValueError("bad jwk")
    raw = b64d(jwk["x"])
    if len(raw) != 32:
        raise ValueError("bad key length")
    return Ed25519PublicKey.from_public_bytes(raw)


def proposal_digest(p: dict) -> str:
    return sha(canon({"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"],
                      "target": p.get("target"), "payload": p["payload"], "evidence": sorted(p["evidence"])}))


def verify_receipt(jwk, eval_id, input_digest, r: dict) -> bool:
    try:
        key = import_jwk(jwk)
        inner = {k: r[k] for k in ("dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId")}
        msg = canon({"profile": PROFILE, "evaluationId": eval_id, "inputDigest": input_digest, "receipt": inner})
        key.verify(b64d(r["receiptSignature"]), msg.encode("utf-8"))
        return True
    except (InvalidSignature, Exception):
        return False
