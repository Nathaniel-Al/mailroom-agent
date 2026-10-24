"""SQLite persistence: evaluations (request, proposals, verifier, commit) and the decision cache."""
import json
import os
import sqlite3
import threading

from .config import DB_PATH

_d = os.path.dirname(DB_PATH)
if _d:
    os.makedirs(_d, exist_ok=True)

_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.executescript("""
pragma journal_mode=wal;
create table if not exists evals(
  eval_id text primary key, req_hash text not null, input_digest text not null,
  verifier text not null, propose_resp text not null, proposals text not null,
  commit_hash text, commit_resp text);
create table if not exists cache(key text primary key, val text not null);
""")
_lock = threading.Lock()


def _q(sql, args=()):
    with _lock:
        return _db.execute(sql, args).fetchall()


def get_eval(eid):
    r = _q("select eval_id, req_hash, input_digest, verifier, propose_resp, proposals, commit_hash, commit_resp "
           "from evals where eval_id=?", (eid,))
    if not r:
        return None
    k = ("eval_id", "req_hash", "input_digest", "verifier", "propose_resp", "proposals", "commit_hash", "commit_resp")
    d = dict(zip(k, r[0]))
    d["verifier"] = json.loads(d["verifier"])
    d["proposals"] = json.loads(d["proposals"])
    return d


def insert_eval(eid, req_hash, input_digest, verifier, propose_resp, proposals):
    _q("insert or ignore into evals(eval_id,req_hash,input_digest,verifier,propose_resp,proposals) values(?,?,?,?,?,?)",
       (eid, req_hash, input_digest, json.dumps(verifier), propose_resp, json.dumps(proposals, ensure_ascii=False)))


def set_commit(eid, commit_hash, commit_resp):
    _q("update evals set commit_hash=?, commit_resp=? where eval_id=? and commit_resp is null",
       (commit_hash, commit_resp, eid))


def cache_get(key):
    r = _q("select val from cache where key=?", (key,))
    return json.loads(r[0][0]) if r else None


def cache_put(key, val):
    _q("insert or replace into cache values(?,?)", (key, json.dumps(val, ensure_ascii=False)))
