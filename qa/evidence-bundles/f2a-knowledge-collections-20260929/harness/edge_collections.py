#!/usr/bin/env python3
"""Live edge cases for Knowledge Library collections, driven through the HTTP API.

usage: edge_collections.py   (env: NEW_BIN, OLD_BIN, OUT)
Starts and stops its own servers on port 8192 via /tmp/f2h/serve.sh and stop.sh,
each on a fresh library under /tmp/f2c. OLD_BIN is the build before this change.
"""
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
NEW_BIN = os.environ["NEW_BIN"]
OLD_BIN = os.environ["OLD_BIN"]
OUT = os.environ.get("OUT", "/tmp/f2c/edge-out")
LIB = "/tmp/f2c/lib-edge"
LIB_UPGRADE = "/tmp/f2c/lib-upgrade"
LIB_LAN = "/tmp/f2c/lib-lan"
RESULTS = []
SERVERS = []
DATA = LIB
API_KEY = None

DOCS = {
    "leave.txt": "Leave policy. Book annual leave two weeks ahead. Manager approval is required for leave longer than five days.",
    "expenses.txt": "Expense policy. Claims need an itemised receipt. Manager approval is required for claims above two hundred euros.",
    "roadmap.txt": "Product roadmap. The mobile release ships in March after design approval.",
    "security.txt": "Security policy. Laptop encryption needs security team approval before travel.",
}


def check(name, cond, detail=""):
    RESULTS.append({"check": name, "pass": bool(cond), **({"detail": repr(detail)[:600]} if detail and not cond else {})})
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"\n        {repr(detail)[:600]}" if detail and not cond else ""), flush=True)


def http(method, path, payload=None, raw=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    headers = {"Content-Type": "application/json", **({"x-api-key": API_KEY} if API_KEY else {})}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            body, status = res.read(), res.status
    except urllib.error.HTTPError as err:
        body, status = err.read(), err.code
    try:
        return status, json.loads(body) if body else None
    except json.JSONDecodeError:
        return status, {"raw": body[:120].decode(errors="replace")}


def start(binary, data, extra=None):
    global DATA
    DATA = data
    subprocess.run(["bash", "/tmp/f2h/serve.sh", binary, data, str(PORT), *(extra or [])], check=True, capture_output=True)
    # Prove the process answering is the one just started, from the binary asked for.
    pid = int(open(os.path.join(data, "server.pid")).read())
    argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    expected = subprocess.check_output([binary, "--version"], text=True).split()[-1]
    _, health = http("GET", "/v1/health")
    build = str(health.get("build"))
    assert argv[0].decode() == binary and expected in build, (argv, expected, build)
    SERVERS.append({"binary": os.path.basename(binary), "build": build, "pid": pid, "library": os.path.basename(data),
                    "api_surface": health.get("api_surface"), "args": extra or []})


def stop():
    subprocess.run(["bash", "/tmp/f2h/stop.sh", DATA], check=True, capture_output=True)


def db(data=None):
    conn = sqlite3.connect(os.path.join(data or DATA, "documents_rag.sqlite3"))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ingest(name, text=None, doc_id=None, collection_ids=None):
    payload = {"filename": name, "content": text if text is not None else DOCS[name],
               **({"doc_id": doc_id} if doc_id else {}), **({"collection_ids": collection_ids} if collection_ids is not None else {})}
    return http("POST", "/api/documents/ingest", payload)


def create(name):
    return http("POST", "/api/collections", {"name": name})


def collection(cid):
    _, listed = http("GET", "/api/collections")
    return next((c for c in listed if c["id"] == cid), None)


def search(query, doc_ids=None, collection_ids=None, mode=None, top_k=10):
    payload = {"query": query, "top_k": top_k, **({"doc_ids": doc_ids} if doc_ids is not None else {}),
               **({"collection_ids": collection_ids} if collection_ids is not None else {}), **({"mode": mode} if mode else {})}
    return http("POST", "/api/documents/search", payload)


def found(body):
    return {r["doc_id"] for r in (body or {}).get("results") or []}


def err(body):
    return ((body or {}).get("error") or {}).get("code")


def wait_indexed(timeout=300):
    started = time.time()
    while time.time() - started < timeout:
        _, status = http("GET", "/api/documents/index-status")
        docs = status["documents"]
        if all(d["indexed_chunks"] + d["skipped_chunks"] >= d["indexable_chunks"] for d in docs):
            return status
        time.sleep(1)
    return status


def orphans(data=None):
    with db(data) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM document_collection_members m WHERE m.doc_id NOT IN (SELECT id FROM documents) "
            "OR m.collection_id NOT IN (SELECT id FROM document_collections)").fetchone()[0]


def names():
    return [c["name"] for c in http("GET", "/api/collections")[1]]


def section_a():
    print("A. managing collections", flush=True)
    start(NEW_BIN, LIB)
    status, body = http("GET", "/api/collections")
    check("a new library has no collections", status == 200 and body == [], (status, body))
    status, hr = create("  HR policies ")
    check("create trims the name and answers 201 with no members",
          status == 201 and hr["name"] == "HR policies" and hr["doc_ids"] == [] and len(hr["id"]) == 36, (status, hr))
    status, body = create("hr POLICIES")
    check("a name differing only in ASCII case is taken (409)", status == 409 and err(body) == "collection_name_taken", (status, body))
    for label, name in (("an empty name", ""), ("a blank name", "   "), ("81 characters", "x" * 81), ("a control character", "HR\npolicies")):
        status, body = create(name)
        check(f"{label} is rejected (422)", status == 422 and err(body) == "invalid_collection_name", (status, body))
    status, body = create("é" * 80)
    check("80 characters (not bytes) are accepted", status == 201 and len(body["name"]) == 80, (status, body))
    http("DELETE", f"/api/collections/{body['id']}")
    status, rh = create("Équipe RH")
    status2, rh2 = create("équipe rh")
    check("non-ASCII letters are compared exactly (SQLite NOCASE folds ASCII only)", status == 201 and status2 == 201, (status, status2))
    http("DELETE", f"/api/collections/{rh2['id']}")
    status, fin = create("finance")
    check("the list is ordered by name ignoring case", names() == ["finance", "HR policies", "Équipe RH"], names())

    ids = {}
    for name in ("leave.txt", "expenses.txt", "roadmap.txt", "security.txt"):
        status, body = ingest(name)
        assert status == 200, (status, body)
        ids[name] = body["doc_id"]
    with db() as conn:
        before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    status, body = ingest("stray.txt", "Stray note about approval.", collection_ids=[hr["id"], "no-such-collection"])
    with db() as conn:
        after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    check("an upload into an unknown collection is refused and stores nothing",
          status == 404 and err(body) == "collection_not_found" and before == after, (status, body, before, after))
    status, body = ingest("policy-index.txt", "Policy index: leave, expenses and security approval rules.", collection_ids=[hr["id"], fin["id"], hr["id"]])
    ids["policy-index.txt"] = body["doc_id"]
    check("an upload can join several collections at once",
          status == 200 and ids["policy-index.txt"] in collection(hr["id"])["doc_ids"] and ids["policy-index.txt"] in collection(fin["id"])["doc_ids"],
          (status, body))

    status, body = http("POST", f"/api/collections/{hr['id']}/documents", {"doc_ids": [ids["leave.txt"], ids["expenses.txt"], ids["leave.txt"]]})
    check("adding documents keeps the order added and ignores repeats",
          status == 200 and body["doc_ids"] == [ids["policy-index.txt"], ids["leave.txt"], ids["expenses.txt"]], (status, body))
    status, body = http("POST", f"/api/collections/{fin['id']}/documents", {"doc_ids": [ids["expenses.txt"], "no-such-document"]})
    check("adding with one unknown document adds none of them (404)",
          status == 404 and err(body) == "document_not_found" and collection(fin["id"])["doc_ids"] == [ids["policy-index.txt"]], (status, body))
    http("POST", f"/api/collections/{fin['id']}/documents", {"doc_ids": [ids["expenses.txt"]]})
    status, body = http("POST", "/api/collections/no-such-collection/documents", {"doc_ids": [ids["leave.txt"]]})
    check("adding to an unknown collection is a 404", status == 404 and err(body) == "collection_not_found", (status, body))

    status, body = http("PATCH", f"/api/collections/{hr['id']}", {"name": "HR"})
    check("rename answers the renamed collection with its members", status == 200 and body["name"] == "HR" and len(body["doc_ids"]) == 3, (status, body))
    status, body = http("PATCH", f"/api/collections/{hr['id']}", {"name": "FINANCE"})
    check("rename to a taken name is a 409 and changes nothing", status == 409 and collection(hr["id"])["name"] == "HR", (status, body))
    status, body = http("PATCH", f"/api/collections/{hr['id']}", {"name": "hr"})
    check("rename to a different case of its own name is allowed", status == 200 and body["name"] == "hr", (status, body))
    http("PATCH", f"/api/collections/{hr['id']}", {"name": "HR"})
    status, body = http("PATCH", "/api/collections/no-such-collection", {"name": "x"})
    check("rename of an unknown collection is a 404", status == 404, status)
    return ids, hr, fin, rh


def section_b(ids, hr, fin, rh):
    print("B. searching collections", flush=True)
    status, whole = search("approval", mode="keyword")
    check("the whole library holds matches outside any collection", ids["security.txt"] in found(whole) and ids["roadmap.txt"] in found(whole), found(whole))
    hr_members = set(collection(hr["id"])["doc_ids"])
    for mode in ("keyword", "hybrid"):
        status, body = search("approval", collection_ids=[hr["id"]], mode=mode)
        check(f"{mode}: a collection search returns only its members",
              status == 200 and found(body) and found(body) <= hr_members and body["retrieval"]["mode"] == mode, (status, found(body), body.get("retrieval")))
        status2, same = search("approval", doc_ids=sorted(hr_members), mode=mode)
        check(f"{mode}: it ranks exactly like naming its members as doc_ids",
              [(r["doc_id"], r["chunk_index"]) for r in body["results"]] == [(r["doc_id"], r["chunk_index"]) for r in same["results"]], (body["results"], same["results"]))
    status, body = search("approval", doc_ids=[ids["roadmap.txt"]], collection_ids=[fin["id"]], mode="keyword")
    check("doc_ids and collection_ids together search both",
          found(body) == {ids["roadmap.txt"], ids["expenses.txt"], ids["policy-index.txt"]}, found(body))
    status, body = search("approval", collection_ids=[rh["id"]])
    check("an empty collection searches nothing", status == 200 and body["results"] == [] and body["retrieval"]["mode"] == "none", (status, body))
    status, body = search("approval", collection_ids=[])
    check("an empty collection list searches nothing", status == 200 and body["results"] == [] and body["retrieval"]["mode"] == "none", (status, body))
    status, body = search("approval", collection_ids=[hr["id"], "no-such-collection"])
    check("an unknown collection in the scope is a 404, not a silent partial search", status == 404 and err(body) == "collection_not_found", (status, body))
    status, body = search("approval", collection_ids=[hr["id"], fin["id"]], mode="keyword")
    per_doc = {}
    for r in body["results"]:
        per_doc[(r["doc_id"], r["chunk_index"])] = per_doc.get((r["doc_id"], r["chunk_index"]), 0) + 1
    check("a document in two searched collections is searched once", status == 200 and max(per_doc.values()) == 1, per_doc)

    print("C. membership lifecycle", flush=True)
    status, _ = http("DELETE", f"/api/collections/{hr['id']}/documents/{ids['expenses.txt']}")
    status_again, _ = http("DELETE", f"/api/collections/{hr['id']}/documents/{ids['expenses.txt']}")
    _, docs = http("GET", "/api/documents")
    check("removing a member is a 204, idempotent, and keeps the document",
          status == 204 and status_again == 204 and ids["expenses.txt"] not in collection(hr["id"])["doc_ids"]
          and any(d["id"] == ids["expenses.txt"] for d in docs), (status, status_again))
    status, body = http("DELETE", f"/api/collections/no-such-collection/documents/{ids['leave.txt']}")
    check("removing from an unknown collection is a 404", status == 404, status)

    status, body = ingest("leave.txt", DOCS["leave.txt"].replace("two weeks", "three weeks"), doc_id=ids["leave.txt"])
    status2, hits = search("three weeks", collection_ids=[hr["id"]], mode="keyword")
    check("re-ingesting a member keeps its memberships and serves the new text",
          status == 200 and ids["leave.txt"] in collection(hr["id"])["doc_ids"] and ids["leave.txt"] in found(hits), (status, body, found(hits)))
    status, body = ingest("leave.txt", DOCS["leave.txt"], doc_id=ids["leave.txt"], collection_ids=[fin["id"]])
    check("re-ingesting into another collection adds, never drops",
          ids["leave.txt"] in collection(hr["id"])["doc_ids"] and ids["leave.txt"] in collection(fin["id"])["doc_ids"], (status, body))

    status, _ = http("DELETE", f"/api/documents/{ids['policy-index.txt']}")
    check("deleting a document removes it from every collection",
          status == 204 and all(ids["policy-index.txt"] not in c["doc_ids"] for c in http("GET", "/api/collections")[1]) and orphans() == 0, status)
    status, _ = http("DELETE", f"/api/collections/{fin['id']}")
    status_again, body = http("DELETE", f"/api/collections/{fin['id']}")
    _, docs = http("GET", "/api/documents")
    check("deleting a collection keeps its documents; a second delete is a 404",
          status == 204 and status_again == 404 and {ids["leave.txt"], ids["expenses.txt"]} <= {d["id"] for d in docs} and orphans() == 0,
          (status, status_again))
    status, body = search("approval", collection_ids=[fin["id"]])
    check("a deleted collection cannot be searched", status == 404 and err(body) == "collection_not_found", (status, body))


def section_d():
    print("D. the search scope limit", flush=True)
    status, big = create("Big")
    now = int(time.time())
    with db() as conn:
        conn.executemany("INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at) VALUES (?, ?, 'txt', 1, 0, ?)",
                         [(f"bulk-{n:05d}", f"bulk-{n:05d}.txt", now) for n in range(30_001)])
        conn.executemany("INSERT INTO document_collection_members (collection_id, doc_id, added_at) VALUES (?, ?, ?)",
                         [(big["id"], f"bulk-{n:05d}", now) for n in range(30_000)])
    for mode in ("keyword", "hybrid"):
        status, body = search("approval", collection_ids=[big["id"]], mode=mode)
        check(f"{mode}: a scope of exactly 30,000 documents is searched", status == 200 and body["results"] == [], (status, body))
    with db() as conn:
        conn.execute("INSERT INTO document_collection_members (collection_id, doc_id, added_at) VALUES (?, 'bulk-30000', ?)", (big["id"], now))
    status, body = search("approval", collection_ids=[big["id"]], mode="keyword")
    check("one more is a 422 search_scope_too_large, not a database error",
          status == 422 and err(body) == "search_scope_too_large" and "30001" in body["error"]["message"], (status, body))
    status, _ = http("DELETE", f"/api/collections/{big['id']}")
    with db() as conn:
        conn.execute("DELETE FROM documents WHERE id LIKE 'bulk-%'")
        left = conn.execute("SELECT COUNT(*) FROM document_collection_members").fetchone()[0]
    check("deleting a 30,001-member collection removes its memberships", status == 204 and orphans() == 0, (status, left))
    with db() as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    check("the database is intact", integrity == "ok" and fk == [], (integrity, fk))
    stop()


def section_e():
    print("E. the previous build, this build, and back", flush=True)
    start(OLD_BIN, LIB_UPGRADE)
    old_ids = [ingest(name)[1]["doc_id"] for name in ("leave.txt", "expenses.txt", "roadmap.txt")]
    stop()
    start(NEW_BIN, LIB_UPGRADE)
    status, body = http("GET", "/api/collections")
    check("a library written by the previous build opens with no collections", status == 200 and body == [], (status, body))
    status, hr = create("HR")
    http("POST", f"/api/collections/{hr['id']}/documents", {"doc_ids": old_ids[:2]})
    status, body = search("approval", collection_ids=[hr["id"]], mode="keyword")
    check("its documents can be collected and searched", found(body) == set(old_ids[:2]), found(body))
    stop()

    start(OLD_BIN, LIB_UPGRADE)
    status, body = http("POST", "/api/documents/search", {"query": "approval", "doc_ids": old_ids, "top_k": 5, "mode": "keyword"})
    check("the previous build still searches the library", status == 200 and found(body) == set(old_ids), (status, found(body)))
    status, body = ingest("security.txt")
    check("and ingests into it", status == 200, (status, body))
    status, _ = http("DELETE", f"/api/documents/{old_ids[1]}")
    with db(LIB_UPGRADE) as conn:
        left = [r[0] for r in conn.execute("SELECT doc_id FROM document_collection_members ORDER BY rowid")]
    check("a delete by the previous build also removes the membership (the cascade is in the schema)",
          status == 204 and left == [old_ids[0]], (status, left))
    status, body = ingest("leave.txt", DOCS["leave.txt"] + " Updated.", doc_id=old_ids[0])
    with db(LIB_UPGRADE) as conn:
        kept = [r[0] for r in conn.execute("SELECT doc_id FROM document_collection_members ORDER BY rowid")]
    RESULTS.append({"observation": "previous build re-ingest (INSERT OR REPLACE) and memberships",
                    "status": status, "memberships_after": kept, "member_before": old_ids[0]})
    print(f"  NOTE  previous-build re-ingest: memberships after = {kept}", flush=True)
    stop()

    start(NEW_BIN, LIB_UPGRADE)
    got = collection(hr["id"])
    check("back on this build the collection is intact, less what the previous build deleted",
          got is not None and got["name"] == "HR" and set(got["doc_ids"]) <= {old_ids[0]} and orphans() == 0, got)
    with db() as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    check("the database is intact after the round trip", integrity == "ok" and fk == [], (integrity, fk))
    stop()


def section_f():
    global API_KEY
    print("F. the LAN chat surface", flush=True)
    os.makedirs(LIB_LAN, exist_ok=True)
    key_file = os.path.join(LIB_LAN, "api.key")
    API_KEY = "edge-collections-" + os.urandom(8).hex()
    with open(key_file, "w", encoding="utf-8") as fh:
        fh.write(API_KEY)
    os.chmod(key_file, 0o600)
    start(NEW_BIN, LIB_LAN, ["--lan-chat-only", "--api-key-file", key_file, "--no-open"])
    for method, path, payload in (("GET", "/api/collections", None), ("POST", "/api/collections", {"name": "x"}),
                                  ("POST", "/api/documents/search", {"query": "x", "collection_ids": []})):
        status, body = http(method, path, payload)
        check(f"LAN surface: {method} {path} is a 403 lan_chat_only", status == 403 and err(body) == "lan_chat_only", (status, body))
    stop()
    API_KEY = None


def main():
    subprocess.run(["rm", "-rf", LIB, LIB_UPGRADE, LIB_LAN, OUT])
    os.makedirs(OUT)
    ids, hr, fin, rh = section_a()
    wait_indexed()
    section_b(ids, hr, fin, rh)
    section_d()
    section_e()
    section_f()
    checks = [r for r in RESULTS if "check" in r]
    passed = sum(r["pass"] for r in checks)
    with open(os.path.join(OUT, "edge-collections.json"), "w", encoding="utf-8") as fh:
        json.dump({"passed": passed, "total": len(checks), "servers": SERVERS, "checks": RESULTS}, fh, indent=2)
    print(f"EDGE {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        stop()
