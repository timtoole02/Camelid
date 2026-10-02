#!/usr/bin/env python3
"""Live edge cases for hybrid document retrieval, driven through the HTTP API on a fresh library.

usage: edge_hybrid.py   (env: NEW_BIN, BASE_BIN, OUT)
Starts and stops its own servers on port 8192 via /tmp/f2h/serve.sh and stop.sh.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 8192
BASE = f"http://127.0.0.1:{PORT}"
DATA = "/tmp/f2h/lib-edge"
DB = os.path.join(DATA, "documents_rag.sqlite3")
NEW_BIN = os.environ["NEW_BIN"]
BASE_BIN = os.environ["BASE_BIN"]
OUT = os.environ.get("OUT", "/tmp/f2h/edge-out")
REPO = os.path.expanduser("~/Camelid")
POLICY = open(os.path.join(REPO, "qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt"), encoding="utf-8").read()
README = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
INCIDENT = "reported to the account owner within seventy-two hours"
QUESTION = "If hackers steal my information, when will you tell me?"
ENCODER_SHA = "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7"
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append({"check": name, "pass": bool(cond), **({"detail": detail} if detail and not cond else {})})
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not cond else ""), flush=True)


def http(method, path, payload=None, raw=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            body = res.read()
            status = res.status
    except urllib.error.HTTPError as err:
        body, status = err.read(), err.code
    try:
        return status, json.loads(body) if body else None
    except json.JSONDecodeError:
        return status, {"raw": body[:200].decode(errors="replace")}


SERVERS = []


def start(binary, env=None, models_dir=None):
    full_env = {**os.environ, **(env or {})}
    if models_dir:
        full_env["MODELS_DIR"] = models_dir
    subprocess.run(["bash", "/tmp/f2h/serve.sh", binary, DATA, str(PORT)], env=full_env, check=True, capture_output=True)
    # Prove the process answering is the one just started, from the binary asked for.
    pid = int(open(os.path.join(DATA, "server.pid")).read())
    argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    expected = subprocess.check_output([binary, "--version"], text=True).split()[-1]
    _, health = http("GET", "/v1/health")
    build = str(health.get("build"))
    assert argv[0].decode() == binary and expected in build, (argv, expected, build)
    SERVERS.append({"binary": os.path.basename(binary), "build": build, "pid": pid,
                    "env": {k: v for k, v in (env or {}).items()},
                    "models_dir": "foreign" if models_dir else "default"})


def stop():
    subprocess.run(["bash", "/tmp/f2h/stop.sh", DATA], check=True, capture_output=True)


def db():
    return sqlite3.connect(DB)


def ingest(name, text, doc_id=None):
    payload = {"filename": name, "content": text, **({"doc_id": doc_id} if doc_id else {})}
    status, body = http("POST", "/api/documents/ingest", payload)
    assert status == 200, (status, body)
    return body["doc_id"]


def coverage(doc_id):
    _, status = http("GET", "/api/documents/index-status")
    return status, next((d for d in status["documents"] if d["id"] == doc_id), None)


def wait_indexed(doc_id, timeout=900):
    started = time.time()
    while time.time() - started < timeout:
        _, entry = coverage(doc_id)
        if entry and entry["indexed_chunks"] + entry["skipped_chunks"] >= entry["indexable_chunks"]:
            return entry
        time.sleep(1)
    return coverage(doc_id)[1]


def search(query, doc_ids=None, mode=None, top_k=4):
    payload = {"query": query, "top_k": top_k, **({"doc_ids": doc_ids} if doc_ids else {}), **({"mode": mode} if mode else {})}
    return http("POST", "/api/documents/search", payload)


def first(body):
    results = (body or {}).get("results") or []
    return results[0] if results else {"excerpt": "", "retrieval": None}


def vector_rows(doc_id):
    with db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM document_chunk_vectors v JOIN document_chunks c ON c.id = v.chunk_id WHERE c.doc_id = ?",
            (doc_id,)).fetchone()[0]


def main():
    subprocess.run(["rm", "-rf", DATA, OUT])
    os.makedirs(OUT)

    print("A. encoder present", flush=True)
    start(NEW_BIN)
    policy = ingest("support-policy.txt", POLICY)
    entry = wait_indexed(policy)
    check("a new upload is indexed in the background", entry["indexed_chunks"] == entry["indexable_chunks"] == 5, entry)
    with db() as conn:
        rows = conn.execute(
            "SELECT v.chunk_sha256 = c.chunk_sha256, v.encoder_sha256, v.dims, length(v.vector) FROM document_chunk_vectors v "
            "JOIN document_chunks c ON c.id = v.chunk_id WHERE c.doc_id = ?", (policy,)).fetchall()
    check("every vector is bound to its chunk hash, the pinned encoder and 768 dims",
          len(rows) == 5 and all(r == (1, ENCODER_SHA, 768, 768 * 4) for r in rows), rows)

    status, body = search(QUESTION, [policy])
    top = first(body)
    check("a question sharing no word with the passage finds it first, by meaning",
          status == 200 and INCIDENT in top["excerpt"] and top["retrieval"] == "semantic" and body["retrieval"]["mode"] == "hybrid",
          [(r["chunk_index"], r["retrieval"]) for r in body["results"]])
    status, body = search(QUESTION, [policy], mode="keyword")
    check("keyword mode alone falls back to the opening passages", body["retrieval"]["mode"] == "attached"
          and [r["chunk_index"] for r in body["results"]] == [0, 1, 2, 3], body["retrieval"])
    status, _ = http("POST", "/api/documents/search", raw=json.dumps({"query": "x", "mode": "bogus"}).encode())
    check("an unknown mode is rejected", status == 422, status)
    status, body = search("refund", [policy], mode="semantic")
    check("semantic mode ranks by meaning only", status == 200 and all(r["retrieval"] == "semantic" for r in body["results"]), body)

    with db() as conn:
        (source_text,) = conn.execute("SELECT source_text FROM documents WHERE id = ?", (policy,)).fetchone()
        conn.execute("UPDATE documents SET source_text = replace(source_text, 'seventy-two hours', 'thirty days') WHERE id = ?", (policy,))
    status, body = search(QUESTION, [policy])
    check("a tampered source serves nothing, not even a meaning match",
          status == 200 and body["results"] == [], [(r["chunk_index"], r["retrieval"]) for r in body["results"]])
    with db() as conn:
        # Restore the exact bytes: the policy already says "thirty days" elsewhere.
        conn.execute("UPDATE documents SET source_text = ? WHERE id = ?", (source_text, policy))
    status, body = search(QUESTION, [policy])
    check("restoring the source restores the meaning match", INCIDENT in first(body)["excerpt"], body)

    with db() as conn:
        chunk_id, content = conn.execute(
            "SELECT id, content FROM document_chunks WHERE doc_id = ? AND instr(content, ?) > 0", (policy, INCIDENT)).fetchone()
        conn.execute("UPDATE document_chunks SET content = replace(content, 'seventy-two', 'ninety-six') WHERE id = ?", (chunk_id,))
    status, body = search(QUESTION, [policy])
    check("a drifted chunk excerpt is withheld while the rest still serve",
          all("ninety-six" not in r["excerpt"] for r in body["results"]) and len(body["results"]) == 4, body["results"])
    with db() as conn:
        conn.execute("UPDATE document_chunks SET content = ? WHERE id = ?", (content, chunk_id))

    old_chunk_ids = [r[0] for r in db().execute("SELECT id FROM document_chunks WHERE doc_id = ?", (policy,))]
    ingest("support-policy.txt", POLICY.replace("seventy-two hours", "forty-eight hours"), doc_id=policy)
    with db() as conn:
        stale = conn.execute(f"SELECT COUNT(*) FROM document_chunk_vectors WHERE chunk_id IN ({','.join('?' * len(old_chunk_ids))})", old_chunk_ids).fetchone()[0]
    check("re-ingesting a document drops the old chunks' vectors", stale == 0, stale)
    entry = wait_indexed(policy)
    status, body = search(QUESTION, [policy])
    check("the re-ingested text is indexed and served", entry["indexed_chunks"] == entry["indexable_chunks"]
          and "forty-eight hours" in first(body)["excerpt"], (entry, first(body)["excerpt"][-200:]))

    doomed = ingest("doomed.txt", "Temporary note about quarterly budget planning.")
    wait_indexed(doomed)
    check("a small upload gets its vector", vector_rows(doomed) == 1)
    http("DELETE", f"/api/documents/{doomed}")
    with db() as conn:
        orphans = conn.execute("SELECT COUNT(*) FROM document_chunk_vectors WHERE chunk_id NOT IN (SELECT id FROM document_chunks)").fetchone()[0]
    check("deleting a document leaves no vector behind", orphans == 0, orphans)

    with db() as conn:
        conn.execute("INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at) VALUES ('legacy', 'old.txt', 'txt', 40, 1, 1)")
        cur = conn.execute("INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES ('legacy', 0, 'Legacy notes mention the zeppelin maintenance window.')")
        conn.execute("INSERT INTO document_chunks_fts (rowid, content) VALUES (?, ?)", (cur.lastrowid, 'Legacy notes mention the zeppelin maintenance window.'))
    _, entry = coverage("legacy")
    status, body = search("zeppelin", ["legacy"])
    check("a pre-citation document is not indexable but still found by keyword",
          entry["indexable_chunks"] == 0 and body["results"] and body["results"][0]["retrieval"] == "keyword"
          and body["retrieval"]["mode"] == "keyword", (entry, body))

    # A late upload must not wait behind a large one that is still being embedded.
    large = ingest("large.md", "\n\n".join([README] * 3))
    started = time.time()
    while time.time() - started < 600:
        _, entry = coverage(large)
        if entry and entry["indexed_chunks"] >= 1:
            break
        time.sleep(0.5)
    late = ingest("late.txt", "A late note: the lighthouse ferry leaves at seven.")
    late_entry = wait_indexed(late, timeout=600)
    _, large_entry = coverage(large)
    check("an upload made while a large file is indexing is embedded before the rest of it",
          late_entry["indexed_chunks"] == late_entry["indexable_chunks"] == 1
          and large_entry["indexed_chunks"] < large_entry["indexable_chunks"], (late_entry, large_entry))
    http("DELETE", f"/api/documents/{large}")
    http("DELETE", f"/api/documents/{late}")
    started = time.time()
    while time.time() - started < 120 and http("GET", "/api/documents/index-status")[1]["semantic"]["indexing"]:
        time.sleep(1)  # let the batch that was in flight at the delete try to store
    with db() as conn:
        orphans = conn.execute("SELECT COUNT(*) FROM document_chunk_vectors WHERE chunk_id NOT IN (SELECT id FROM document_chunks)").fetchone()[0]
    check("deleting a document while it is being indexed leaves no vector behind", orphans == 0, orphans)

    print("B. skip and recovery across a restart", flush=True)
    stop()
    with db() as conn:
        text = "Skip test: the courier leaves at dawn from the east gate."
        h = hashlib.sha256(text.encode()).hexdigest()
        conn.execute("INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text) "
                     "VALUES ('skipdoc', 'skip.txt', 'txt', ?, 1, 2, ?, ?, ?)", (len(text), h, h, text))
        cur = conn.execute("INSERT INTO document_chunks (doc_id, chunk_index, content, byte_start, byte_end, chunk_sha256) VALUES ('skipdoc', 0, ?, 0, ?, ?)",
                           ("Skip test: the courier leaves at noon from the east gate.", len(text), h))
        conn.execute("INSERT INTO document_chunks_fts (rowid, content) VALUES (?, ?)", (cur.lastrowid, text))
        skip_chunk = cur.lastrowid
    start(NEW_BIN)
    entry = wait_indexed("skipdoc")
    with db() as conn:
        row = conn.execute("SELECT vector IS NULL FROM document_chunk_vectors WHERE chunk_id = ?", (skip_chunk,)).fetchone()
    check("a chunk whose text does not match its hash is skipped, not embedded", entry["skipped_chunks"] == 1 and row == (1,), (entry, row))
    with db() as conn:
        conn.execute("UPDATE document_chunks SET content = ? WHERE id = ?", (text, skip_chunk))
    entry = coverage("skipdoc")[1]
    entry = wait_indexed("skipdoc")
    check("restoring that text gets it embedded", entry["indexed_chunks"] == 1 and entry["skipped_chunks"] == 0, entry)

    big = ingest("README.md", README)
    started = time.time()
    while time.time() - started < 900:
        _, entry = coverage(big)
        if entry["indexed_chunks"] >= 5:
            break
        time.sleep(1)
    stop()
    persisted = vector_rows(big)
    check("vectors written before a shutdown persist", 5 <= persisted < entry["indexable_chunks"], (persisted, entry))
    start(NEW_BIN)
    entry = wait_indexed(big)
    check("indexing resumes after a restart and completes", entry["indexed_chunks"] == entry["indexable_chunks"] and vector_rows(big) == entry["indexable_chunks"], entry)

    print("C. encoder disabled or foreign", flush=True)
    stop()
    before = db().execute("SELECT COUNT(*) FROM document_chunk_vectors").fetchone()[0]
    start(NEW_BIN, env={"CAMELID_DOCUMENT_SEMANTIC": "0"})
    status, st = http("GET", "/api/documents/index-status")
    check("disabled: status says so", st["semantic"]["available"] is False and st["semantic"].get("reason") == "semantic_disabled", st["semantic"])
    status, body = search(QUESTION, [policy])
    check("disabled: auto search is keyword-only", status == 200 and body["retrieval"]["mode"] in ("keyword", "attached"), body["retrieval"])
    status, body = search(QUESTION, [policy], mode="hybrid")
    check("disabled: an explicit hybrid request is refused, not degraded", status == 409 and body["error"]["code"] == "semantic_disabled", (status, body))
    extra = ingest("extra.txt", "Disabled mode upload about lighthouse keepers.")
    time.sleep(3)
    after = db().execute("SELECT COUNT(*) FROM document_chunk_vectors").fetchone()[0]
    check("disabled: uploads are not embedded", after == before, (before, after))
    stop()

    foreign = "/tmp/f2h/models-foreign"
    os.makedirs(foreign, exist_ok=True)
    with open(os.path.join(foreign, "nomic-embed-text-v1.5.Q8_0.gguf"), "wb") as fh:
        fh.write(b"GGUF not the pinned encoder")
    start(NEW_BIN, models_dir=foreign)
    status, st = http("GET", "/api/documents/index-status")
    check("a foreign file under the encoder's name is refused", st["semantic"].get("reason") == "encoder_mismatch", st["semantic"])
    status, body = search(QUESTION, [policy], mode="semantic")
    check("and semantic search says why", status == 409 and body["error"]["code"] == "encoder_mismatch", (status, body))
    stop()

    print("D. the previous build on this library, then back", flush=True)
    start(BASE_BIN)
    status, docs = http("GET", "/api/documents")
    check("the previous build opens the library", status == 200 and len(docs) >= 5, status)
    status, body = http("POST", "/api/documents/search", {"query": "refund window", "doc_ids": [policy], "top_k": 3})
    check("and searches it", status == 200 and body["results"], status)
    status, body = http("POST", "/api/documents/ingest", {"filename": "old-build.txt", "content": "The previous build wrote this note about river ferries."})
    old_doc = body["doc_id"]
    check("and ingests into it", status == 200)
    status, _ = http("DELETE", f"/api/documents/{big}")
    with db() as conn:
        orphans = conn.execute("SELECT COUNT(*) FROM document_chunk_vectors WHERE chunk_id NOT IN (SELECT id FROM document_chunks)").fetchone()[0]
    check("a delete by the previous build still removes the vectors", status == 204 and orphans == 0, (status, orphans))
    stop()
    start(NEW_BIN)
    entry = wait_indexed(old_doc)
    extra_entry = wait_indexed(extra)
    check("back on this build, documents written meanwhile are indexed on first status",
          entry["indexed_chunks"] == entry["indexable_chunks"] == 1 and extra_entry["indexed_chunks"] == 1, (entry, extra_entry))
    with db() as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    check("the database is intact", integrity == "ok" and fk == [], (integrity, fk))
    stop()

    passed = sum(r["pass"] for r in RESULTS)
    with open(os.path.join(OUT, "edge-results.json"), "w", encoding="utf-8") as fh:
        json.dump({"passed": passed, "total": len(RESULTS), "servers": SERVERS, "checks": RESULTS}, fh, indent=2)
    print(f"EDGE {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        subprocess.run(["bash", "/tmp/f2h/stop.sh", DATA], capture_output=True)
