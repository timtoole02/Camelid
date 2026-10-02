#!/usr/bin/env python3
"""Live edge cases for whole-library search, driven through the HTTP API on fresh libraries.

usage: edge_library.py   (env: NEW_BIN, OUT)
Starts and stops its own servers on port 8196 via /tmp/f2h/serve.sh and stop.sh.
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request

PORT = 8196
BASE = f"http://127.0.0.1:{PORT}"
NEW_BIN = os.environ["NEW_BIN"]
OUT = os.environ.get("OUT", "/tmp/f2l/edge-out")
REPO = os.path.expanduser("~/Camelid")
MODELS = os.path.join(REPO, "models")
LIB = "/tmp/f2l/lib-edge"
LIB_EMPTY = "/tmp/f2l/lib-edge-empty"
NO_ENCODER = "/tmp/f2l/models-no-encoder"
FLOOR_AT_ONE_CHUNK, FLOOR_PER_LN_CHUNK = 0.6408, 0.0058
POLICY = open(os.path.join(REPO, "qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt"), encoding="utf-8").read()
GUIDE = open(os.path.join(REPO, "qa/evidence-bundles/f2a-knowledge-collections-20260929/source/escalation-guide.txt"), encoding="utf-8").read()
CONTEXT_DOC = open(os.path.join(REPO, "docs/PROJECT_CONTEXT.md"), encoding="utf-8").read()
QUESTION = "If hackers steal my information, when will you tell me?"
INCIDENT = "reported to the account owner within seventy-two hours"
PROBES = [QUESTION, "How long do refunds take?", "Who handles a billing dispute?", "hello", "thanks!",
          "Tell me a joke about penguins.", "What is 17 times 23?", "Write a haiku about autumn leaves."]
RESULTS, SERVERS, OBSERVED = [], [], []
DATA = LIB
API_KEY = None


def check(name, cond, detail=""):
    RESULTS.append({"check": name, "pass": bool(cond), **({"detail": repr(detail)[:600]} if not cond else {})})
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"\n        {repr(detail)[:600]}"), flush=True)


def http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", **({"x-api-key": API_KEY} if API_KEY else {})}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            body, status = res.read(), res.status
    except urllib.error.HTTPError as err:
        body, status = err.read(), err.code
    return status, json.loads(body) if body else None


def start(data, env=None, models_dir=None, extra=None):
    global DATA
    DATA = data
    full_env = {**os.environ, **(env or {}), **({"MODELS_DIR": models_dir} if models_dir else {})}
    subprocess.run(["bash", "/tmp/f2h/serve.sh", NEW_BIN, data, str(PORT), *(extra or [])], env=full_env, check=True, capture_output=True)
    pid = int(open(os.path.join(data, "server.pid")).read())
    argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    expected = subprocess.check_output([NEW_BIN, "--version"], text=True).split()[-1]
    _, health = http("GET", "/v1/health")
    assert argv[0].decode() == NEW_BIN and expected in str(health.get("build")), (argv, expected, health)
    SERVERS.append({"build": health.get("build"), "library": os.path.basename(data), "env": env or {},
                    "models_dir": "without encoder" if models_dir else "default", "api_surface": health.get("api_surface"), "args": extra or []})


def stop():
    subprocess.run(["bash", "/tmp/f2h/stop.sh", DATA], check=True, capture_output=True)


def search(query, **extra):
    return http("POST", "/api/documents/search", {"query": query, "top_k": extra.pop("top_k", 10), **extra})


def err(body):
    return ((body or {}).get("error") or {}).get("code"), ((body or {}).get("error") or {}).get("param")


def ingest(name, text):
    status, body = http("POST", "/api/documents/ingest", {"filename": name, "content": text})
    assert status == 200, (status, body)
    return body["doc_id"]


def wait_indexed(timeout=900):
    import time
    started = time.time()
    while time.time() - started < timeout:
        _, status = http("GET", "/api/documents/index-status")
        if all(d["indexed_chunks"] + d["skipped_chunks"] >= d["indexable_chunks"] for d in status["documents"]):
            return status
        time.sleep(1)
    raise AssertionError("indexing did not finish")


def expected_floor(chunks):
    return min(0.9, max(0.3, FLOOR_AT_ONE_CHUNK + FLOOR_PER_LN_CHUNK * math.log(max(chunks, 1))))


def outside_floor(results, floor, pinned=()):
    return [(r["doc_id"], r["chunk_index"], r.get("similarity")) for r in results
            if r["doc_id"] not in pinned and (r.get("similarity") is None or r["similarity"] < floor - 1e-6)]


def main():
    global API_KEY
    subprocess.run(["rm", "-rf", LIB, LIB_EMPTY, OUT, NO_ENCODER])
    os.makedirs(OUT)

    print("A. an empty library", flush=True)
    start(LIB_EMPTY)
    status, body = search(QUESTION, library=True)
    check("an empty library answers a library search with nothing", status == 200 and body["results"] == [], (status, body))
    stop()

    print("B. the floor, with the encoder", flush=True)
    start(LIB)
    policy = ingest("support-policy.txt", POLICY)
    guide = ingest("escalation-guide.txt", GUIDE)
    context_doc = ingest("PROJECT_CONTEXT.md", CONTEXT_DOC)
    status = wait_indexed()
    chunks = {d["id"]: d["indexed_chunks"] for d in status["documents"]}
    check("three documents are indexed", len(chunks) == 3 and all(chunks.values()), chunks)
    library_chunks = sum(chunks.values())
    floor = expected_floor(library_chunks)

    status, body = search(QUESTION, library=True)
    reported = body.get("retrieval", {}).get("relevance_floor")
    check(f"the floor applied is the one for this library's {library_chunks} indexed chunks ({floor:.4f})",
          status == 200 and reported is not None and abs(reported - floor) < 1e-4, (reported, floor))
    OBSERVED.append({"query": QUESTION, "indexed_chunks": library_chunks, "floor": reported,
                     "served": [(r["filename"], r["chunk_index"], round(r["similarity"], 4), INCIDENT in r["excerpt"]) for r in body.get("results", [])]})
    status, body = search("How long do refunds take?", library=True)
    top = body["results"][0] if body.get("results") else {}
    check("a question the policy answers directly gets its refunds passage first",
          top.get("doc_id") == policy and "Refunds" in top.get("excerpt", ""), top.get("excerpt", "")[:120])
    for query in PROBES:
        status, body = search(query, library=True, top_k=20)
        plain_status, plain = search(query, top_k=20)
        OBSERVED.append({"query": query, "library_results": len(body["results"]),
                         "library_similarities": [round(r["similarity"], 4) for r in body["results"]],
                         "plain_results": len(plain["results"])})
        check(f"every library result for {query!r} carries a similarity at or above the floor",
              status == 200 and not outside_floor(body["results"], floor), outside_floor(body.get("results", []), floor))
        check(f"without the flag, {query!r} is searched as before, with no similarity reported",
              plain_status == 200 and all("similarity" not in r for r in plain["results"]) and "relevance_floor" not in plain["retrieval"], plain.get("retrieval"))
    status, body = search(QUESTION, library=True, mode="semantic")
    check("semantic mode holds to the floor as well", status == 200 and body["results"] and not outside_floor(body["results"], floor)
          and all(r["retrieval"] == "semantic" for r in body["results"]), body.get("results"))
    status, body = search(QUESTION, library=True, mode="keyword")
    check("keyword mode is refused: the floor needs meaning (422)", status == 422 and err(body) == ("library_search_needs_meaning", "mode"), (status, body))
    status, body = search(QUESTION, library=True, top_k=2)
    check("top_k still bounds a library search", status == 200 and len(body["results"]) <= 2, len(body.get("results", [])))

    status, body = search(QUESTION, library=True, doc_ids=[context_doc], top_k=20)
    pinned = [r for r in body["results"] if r["doc_id"] == context_doc]
    check("an attached document is searched in full: its passages count below the floor",
          status == 200 and pinned and any(r["similarity"] is not None and r["similarity"] < floor for r in pinned),
          [(r["chunk_index"], r["similarity"]) for r in pinned])
    check("while the rest of the library is still held to the floor", not outside_floor(body["results"], floor, pinned=(context_doc,)), outside_floor(body["results"], floor, (context_doc,)))
    _, collection = http("POST", "/api/collections", {"name": "Handbook"})
    http("POST", f"/api/collections/{collection['id']}/documents", {"doc_ids": [context_doc]})
    status, body = search(QUESTION, library=True, collection_ids=[collection["id"]], top_k=20)
    check("a collection's members skip the floor the same way",
          status == 200 and any(r["doc_id"] == context_doc and r["similarity"] < floor for r in body["results"])
          and not outside_floor(body["results"], floor, pinned=(context_doc,)), [(r["doc_id"][:8], r["similarity"]) for r in body.get("results", [])])
    status, body = search(QUESTION, library=True, collection_ids=["no-such-collection"])
    check("an unknown collection is still a 404", status == 404 and err(body)[0] == "collection_not_found", (status, body))

    with sqlite3.connect(os.path.join(LIB, "documents_rag.sqlite3")) as conn:
        conn.execute("INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at) VALUES ('legacy', 'old.txt', 'txt', 60, 1, 1)")
        text = "Legacy note: breaches are reported within seventy-two hours."
        cur = conn.execute("INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES ('legacy', 0, ?)", (text,))
        conn.execute("INSERT INTO document_chunks_fts (rowid, content) VALUES (?, ?)", (cur.lastrowid, text))
    status, body = search("breaches reported within seventy-two hours", library=True, top_k=20)
    check("a passage the encoder never indexed is not found by the library search", status == 200 and all(r["doc_id"] != "legacy" for r in body["results"]),
          [r["doc_id"] for r in body.get("results", [])])
    status, body = search("breaches reported within seventy-two hours", library=True, doc_ids=["legacy"], top_k=20)
    legacy = [r for r in body.get("results", []) if r["doc_id"] == "legacy"]
    check("but is found when its document is attached, with no similarity to report",
          status == 200 and legacy and all(r.get("similarity") is None for r in legacy), legacy)
    stop()

    print("C. without the encoder", flush=True)
    os.makedirs(NO_ENCODER, exist_ok=True)
    start(LIB, models_dir=NO_ENCODER)
    status, body = search(QUESTION, library=True)
    check("a library search without the encoder is a 409 naming why", status == 409 and err(body) == ("encoder_not_installed", "library"), (status, body))
    status, body = search("seventy-two hours", doc_ids=[policy])
    check("an attached-document search still works by keyword", status == 200 and body["results"] and body["retrieval"]["mode"] == "keyword", (status, body.get("retrieval")))
    stop()
    start(LIB, env={"CAMELID_DOCUMENT_SEMANTIC": "0"})
    status, body = search(QUESTION, library=True)
    check("with search by meaning turned off it is a 409 semantic_disabled", status == 409 and err(body) == ("semantic_disabled", "library"), (status, body))
    stop()

    print("D. the LAN chat surface", flush=True)
    key_file = os.path.join(LIB, "api.key")
    API_KEY = "edge-library-" + os.urandom(8).hex()
    with open(key_file, "w", encoding="utf-8") as fh:
        fh.write(API_KEY)
    os.chmod(key_file, 0o600)
    start(LIB, extra=["--lan-chat-only", "--api-key-file", key_file, "--no-open"])
    status, body = search(QUESTION, library=True)
    check("the LAN chat surface refuses a library search (403)", status == 403 and err(body)[0] == "lan_chat_only", (status, body))
    stop()
    API_KEY = None
    os.remove(key_file)

    passed = sum(r["pass"] for r in RESULTS)
    with open(os.path.join(OUT, "edge-library.json"), "w", encoding="utf-8") as fh:
        json.dump({"passed": passed, "total": len(RESULTS), "floor_rule": [FLOOR_AT_ONE_CHUNK, FLOOR_PER_LN_CHUNK], "servers": SERVERS,
                   "probes": OBSERVED, "checks": RESULTS}, fh, indent=2)
    print(f"EDGE {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        stop()
