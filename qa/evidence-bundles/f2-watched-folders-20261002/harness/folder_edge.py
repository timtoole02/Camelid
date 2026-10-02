#!/usr/bin/env python3
"""Live edge cases for watched folders, driven through the HTTP API.

usage: folder_edge.py   (env: NEW_BIN, OUT)
Starts and stops its own servers on port 8196 via /tmp/f2h/serve.sh and stop.sh,
and builds its folders under /tmp/f2r/fw-edge.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

PORT = 8196
BASE = f"http://127.0.0.1:{PORT}"
NEW_BIN = os.environ["NEW_BIN"]
OUT = os.environ.get("OUT", "/tmp/f2r/fw-edge-out")
HARNESS = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.environ.get("SOURCE", os.path.join(os.path.dirname(HARNESS), "source"))
ROOT = "/tmp/f2r/fw-edge"
WATCHED = f"{ROOT}/watched"
OUTSIDE = f"{ROOT}/outside"
LIB = f"{ROOT}/lib"
LOCAL_UI = {"Origin": BASE}
RESULTS, SERVERS, TIMINGS = [], [], {}
DATA = LIB
API_KEY = None


def check(name, cond, detail=""):
    RESULTS.append({"check": name, "pass": bool(cond), **({"detail": repr(detail)[:600]} if not cond else {})})
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"\n        {repr(detail)[:600]}"), flush=True)


def http(method, path, payload=None, headers=LOCAL_UI):
    data = json.dumps(payload).encode() if payload is not None else None
    all_headers = {"Content-Type": "application/json", **headers, **({"x-api-key": API_KEY} if API_KEY else {})}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            body, status = res.read(), res.status
    except urllib.error.HTTPError as error:
        body, status = error.read(), error.code
    return status, json.loads(body) if body else None


def code(body):
    return ((body or {}).get("error") or {}).get("code")


def start(extra=None):
    subprocess.run(["bash", "/tmp/f2h/serve.sh", NEW_BIN, LIB, str(PORT), *(extra or [])], check=True, capture_output=True)
    pid = int(open(os.path.join(LIB, "server.pid")).read())
    argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    expected = subprocess.check_output([NEW_BIN, "--version"], text=True).split()[-1]
    _, health = http("GET", "/v1/health")
    assert argv[0].decode() == NEW_BIN and expected in str(health.get("build")), (argv, expected, health)
    SERVERS.append({"build": health.get("build"), "api_surface": health.get("api_surface"), "args": extra or []})


def stop():
    subprocess.run(["bash", "/tmp/f2h/stop.sh", LIB], check=True, capture_output=True)


def write(rel, text):
    path = os.path.join(WATCHED, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def folder(folder_id):
    status, folders = http("GET", "/api/folders")
    assert status == 200, (status, folders)
    return next((f for f in folders if f["id"] == folder_id), None)


def wait_scanned(folder_id, after, timeout=120):
    """The folder once a scan that finished after `after` (unix seconds) is recorded."""
    started = time.time()
    while time.time() - started < timeout:
        current = folder(folder_id)
        if current and not current["scanning"] and not current["queued"] and (current["last_scan_at"] or 0) >= after:
            return current, time.time() - started
        time.sleep(0.5)
    raise AssertionError(f"folder {folder_id} was not scanned within {timeout}s")


def wait_indexed(timeout=900):
    started = time.time()
    while time.time() - started < timeout:
        _, status = http("GET", "/api/documents/index-status")
        if all(d["indexed_chunks"] + d["skipped_chunks"] >= d["indexable_chunks"] for d in status["documents"]):
            return
        time.sleep(1)
    raise AssertionError("indexing did not finish")


def documents():
    _, docs = http("GET", "/api/documents")
    return {doc["filename"]: doc["id"] for doc in docs}


def search(query, collection_id):
    status, body = http("POST", "/api/documents/search", {"query": query, "collection_ids": [collection_id], "top_k": 3})
    assert status == 200, (status, body)
    return body


def main():
    global API_KEY
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    os.makedirs(OUTSIDE)
    policy = open(os.path.join(SOURCE, "support-policy.txt"), encoding="utf-8").read()
    guide = open(os.path.join(SOURCE, "escalation-guide.txt"), encoding="utf-8").read()
    write("support-policy.txt", policy)
    write("escalation-guide.txt", guide)
    write("team/notes.md", "# Team notes\n\nThe quarterly review meets on the first Monday of each quarter.\n")
    write(".drafts/hidden.md", "A hidden draft that must not be read.")
    write(".hidden.md", "A hidden file that must not be read.")
    write("photo.png", "not a document type")
    write("empty.txt", "   \n")
    write("broken.pdf", "%PDF-1.7 not really a PDF")
    with open(f"{OUTSIDE}/secret.md", "w", encoding="utf-8") as handle:
        handle.write("A file outside the watched folder, reachable only through a link.")
    os.symlink(f"{OUTSIDE}/secret.md", f"{WATCHED}/linked.md")
    os.symlink(OUTSIDE, f"{WATCHED}/linked-folder")

    print("A. who may use the folder routes")
    start()
    status, body = http("GET", "/api/folders", headers={})
    check("a request without the web UI's origin is refused (403)", status == 403 and code(body) == "local_management_forbidden", (status, body))
    status, body = http("GET", "/api/folders", headers={"Origin": "http://evil.example"})
    check("a request from another origin is refused (403)", status == 403 and code(body) == "local_management_forbidden", (status, body))
    status, body = http("POST", "/api/folders", {"path": WATCHED, "collection_id": "x"}, headers={"Origin": "http://evil.example"})
    check("another origin cannot add a folder (403)", status == 403, (status, body))
    status, body = http("GET", "/api/folders")
    check("the web UI lists folders: none yet", status == 200 and body == [], (status, body))

    print("B. what can be watched")
    status, body = http("POST", "/api/collections", {"name": "Policies"})
    assert status == 201, (status, body)
    collection = body["id"]
    for label, path, want in (("a relative path", "watched", 422), ("a file", f"{WATCHED}/support-policy.txt", 422),
                              ("a missing folder", f"{ROOT}/missing", 422), ("the filesystem root", "/", 422)):
        status, body = http("POST", "/api/folders", {"path": path, "collection_id": collection})
        check(f"{label} is refused ({want} invalid_folder)", status == want and code(body) == "invalid_folder", (status, body))
    status, body = http("POST", "/api/folders", {"path": WATCHED, "collection_id": "no-such-collection"})
    check("an unknown collection is a 404", status == 404 and code(body) == "collection_not_found", (status, body))

    print("C. the first scan")
    began = int(time.time())
    status, body = http("POST", "/api/folders", {"path": WATCHED, "collection_id": collection})
    check("watching the folder answers 201", status == 201 and body["path"] == WATCHED and body["collection_id"] == collection, (status, body))
    folder_id = body["id"]
    scanned, seconds = wait_scanned(folder_id, began)
    TIMINGS["first_scan_s"] = round(seconds, 2)
    check("the first scan adds the three documents", scanned["last_changes"] == {"added": 3, "updated": 0, "removed": 0, "unchanged": 0, "skipped": 2}, scanned)
    check("documents and skipped counts", (scanned["documents"], scanned["skipped_count"]) == (3, 2), scanned)
    reasons = {item["path"]: item["reason"] for item in scanned["skipped"]}
    check("an empty file is skipped for having no text", reasons.get("empty.txt") == "no_text", reasons)
    check("a broken PDF is skipped, not fatal", reasons.get("broken.pdf") in ("no_text", "extract_failed"), reasons)
    names = documents()
    check("the library holds exactly the folder's documents", sorted(names) == ["escalation-guide.txt", "support-policy.txt", "team/notes.md"], names)
    check("hidden files, other types and links were not read", not {"linked.md", "linked-folder/secret.md", ".hidden.md", "photo.png"} & set(names), names)
    _, collections = http("GET", "/api/collections")
    members = next(c for c in collections if c["id"] == collection)["doc_ids"]
    check("every document joined the collection", sorted(members) == sorted(names.values()), (members, names))
    status, source = http("GET", f"/api/documents/{names['support-policy.txt']}/source")
    check("a document's source text is the file's text", status == 200 and source.get("text") == policy, (status, str(source)[:200]))
    wait_indexed()
    body = search("When will you tell me about a data breach?", collection)
    hits = [(r["filename"], r["excerpt"]) for r in body["results"]]
    check("searching the collection finds the incident section in the top three",
          any(name == "support-policy.txt" and "seventy-two hours" in excerpt for name, excerpt in hits), hits)

    print("D. changes are picked up")
    write("support-policy.txt", policy + "\n7. Refund appeals\n\nA refused refund can be appealed once; appeals are decided within ten business days.\n")
    os.remove(f"{WATCHED}/escalation-guide.txt")
    write("onboarding.md", "# Onboarding\n\nNew support engineers shadow the queue for two weeks.\n")
    changed = time.time()
    scanned, seconds = wait_scanned(folder_id, int(changed) + 1, timeout=75)
    TIMINGS["poll_pickup_s"] = round(seconds, 2)
    check("the timer picks up the changes without being asked (within 75 s)", scanned["last_changes"] == {"added": 1, "updated": 1, "removed": 1, "unchanged": 1, "skipped": 2}, scanned)
    after = documents()
    check("the changed file keeps its document id", after.get("support-policy.txt") == names["support-policy.txt"], (after, names))
    check("the deleted file's document is gone", "escalation-guide.txt" not in after, after)
    status, _ = http("GET", f"/api/documents/{names['escalation-guide.txt']}/source")
    check("its source is gone too (404)", status == 404, status)
    wait_indexed()
    body = search("How long does a refund appeal take?", collection)
    hits = [(r["filename"], r["excerpt"]) for r in body["results"]]
    check("the new text is searchable (top three)",
          any(name == "support-policy.txt" and "ten business days" in excerpt for name, excerpt in hits), hits)

    edited = "# Team notes\n\nThe quarterly review moved to the second Tuesday of each quarter.\n"
    write("team/notes.md", edited)
    began = int(time.time()) + 1
    time.sleep(1.1)
    status, body = http("POST", f"/api/folders/{folder_id}/scan")
    check("check now answers 202", status == 202, (status, body))
    scanned, seconds = wait_scanned(folder_id, began)
    TIMINGS["check_now_s"] = round(seconds, 2)
    TIMINGS["check_now_changes"] = scanned["last_changes"]
    _, source = http("GET", f"/api/documents/{after['team/notes.md']}/source")
    check("after check now, the edited file's document holds the new text", source.get("text") == edited, source)

    print("E. a restart")
    stop()
    cancelled = "# Team notes\n\nThe quarterly review is cancelled until further notice, by order of the board.\n"
    write("team/notes.md", cancelled)
    began = int(time.time()) + 1
    time.sleep(1.1)
    start()
    scanned, seconds = wait_scanned(folder_id, began, timeout=30)
    TIMINGS["after_restart_s"] = round(seconds, 2)
    check("a change made while the server was stopped is picked up when it starts", scanned["last_changes"]["updated"] == 1, scanned)
    _, source = http("GET", f"/api/documents/{after['team/notes.md']}/source")
    check("and its document holds the new text", source.get("text") == cancelled, source)

    print("F. overlapping folders, and a folder that disappears")
    for label, path in (("the same folder", WATCHED), ("a folder inside it", f"{WATCHED}/team"), ("the folder around it", ROOT)):
        status, body = http("POST", "/api/folders", {"path": path, "collection_id": collection})
        check(f"{label} is refused (409 folder_overlaps)", status == 409 and code(body) == "folder_overlaps", (status, body))
    os.rename(WATCHED, f"{ROOT}/moved-away")
    began = int(time.time()) + 1
    time.sleep(1.1)
    http("POST", f"/api/folders/{folder_id}/scan")
    scanned, _ = wait_scanned(folder_id, began)
    check("a folder that cannot be opened reports why", "cannot be opened" in (scanned["last_error"] or ""), scanned)
    check("and its documents are kept", sorted(documents()) == ["onboarding.md", "support-policy.txt", "team/notes.md"], documents())
    os.rename(f"{ROOT}/moved-away", WATCHED)
    began = int(time.time()) + 1
    time.sleep(1.1)
    http("POST", f"/api/folders/{folder_id}/scan")
    scanned, _ = wait_scanned(folder_id, began)
    check("once it is back, the error clears and nothing changed", scanned["last_error"] is None and scanned["last_changes"]["unchanged"] == 3, scanned)

    print("G. stopping a watch")
    status, _ = http("DELETE", f"/api/folders/{folder_id}", headers={})
    check("stopping a watch needs the web UI too (403)", status == 403, status)
    status, _ = http("DELETE", f"/api/folders/{folder_id}")
    check("stopping a watch answers 204", status == 204, status)
    check("its documents leave the library", documents() == {}, documents())
    check("the files on disk stay", os.path.exists(f"{WATCHED}/support-policy.txt") and os.path.exists(f"{WATCHED}/team/notes.md"))
    _, collections = http("GET", "/api/collections")
    check("the collection stays", any(c["id"] == collection for c in collections), collections)
    status, body = http("DELETE", f"/api/folders/{folder_id}")
    check("stopping it again is a 404", status == 404 and code(body) == "folder_not_found", (status, body))

    print("H. deleting the collection")
    began = int(time.time())
    status, body = http("POST", "/api/folders", {"path": WATCHED, "collection_id": collection})
    folder_id = body["id"]
    wait_scanned(folder_id, began)
    kept = documents()
    status, _ = http("DELETE", f"/api/collections/{collection}")
    check("deleting the collection stops the watch", status == 204 and http("GET", "/api/folders")[1] == [], status)
    check("and leaves its documents in the library", documents() == kept and len(kept) == 3, (documents(), kept))
    stop()

    print("I. the LAN chat surface")
    key_file = os.path.join(OUT, "lan.key")
    with open(key_file, "w") as handle:
        handle.write("edge-suite-lan-key-0123456789abcdef")
    API_KEY = "edge-suite-lan-key-0123456789abcdef"
    start(["--lan-chat-only", "--api-key-file", key_file, "--no-open"])
    for method, path in (("GET", "/api/folders"), ("POST", "/api/folders"), ("POST", "/api/folders/x/scan"), ("DELETE", "/api/folders/x")):
        status, body = http(method, path, {"path": WATCHED, "collection_id": "x"} if method == "POST" else None)
        check(f"{method} {path} is refused on the LAN chat surface (403)", status == 403 and code(body) == "lan_chat_only", (status, body))
    stop()
    os.remove(key_file)
    API_KEY = None

    passed = sum(r["pass"] for r in RESULTS)
    report = {"passed": passed, "total": len(RESULTS), "servers": SERVERS, "timings": TIMINGS, "checks": RESULTS}
    with open(os.path.join(OUT, "folder-edge.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"FOLDER EDGE {passed}/{len(RESULTS)} passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        subprocess.run(["bash", "/tmp/f2h/stop.sh", LIB], capture_output=True)
