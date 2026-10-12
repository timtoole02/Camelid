"""Live check of the upload size limit against a running camelid serve.

Re-uploads the PDFs the 500-PDF gate saw refused for size (413, or the broken
pipe of a body cut off mid-send), resolves every citation they produce, and
sends one file over 64 MB.

Usage: python3 large_uploads.py <port> <corpus dir> <gate ingest.json> <out json>
"""
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

PORT, CORPUS, GATE_INGEST, OUT = sys.argv[1:5]
BASE = f"http://127.0.0.1:{PORT}"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            raw, status = resp.read(), resp.status
    except urllib.error.HTTPError as err:
        raw, status = err.read(), err.code
    except Exception as err:  # noqa: BLE001
        return 0, {"transport_error": str(err)[:200]}, 0
    ms = (time.perf_counter() - started) * 1000
    try:
        return status, json.loads(raw) if raw else None, ms
    except ValueError:
        return status, {"raw": raw[:200].decode(errors="replace")}, ms


manifest = json.load(open(os.path.join(CORPUS, "arxiv-manifest.json")))
refused = [r["arxiv_id"] for r in json.load(open(GATE_INGEST))["rejected"] if r["status"] in (413, 0)]
rows = []
for arxiv_id in refused:
    meta = manifest[arxiv_id]
    data = open(os.path.join(CORPUS, "pdf", meta["file"]), "rb").read()
    status, body, ms = call("POST", "/api/documents/ingest", {
        "filename": meta["file"], "content": base64.b64encode(data).decode(), "is_base64": True})
    row = {"arxiv_id": arxiv_id, "bytes": len(data), "status": status, "ms": round(ms, 1)}
    if status == 200:
        resolved = 0
        for index in range(body["chunk_count"]):
            s, cite, _ = call("POST", "/api/documents/citation/resolve", {"doc_id": body["doc_id"], "chunk_index": index})
            span = cite.get("span", "") if s == 200 else ""
            if s == 200 and hashlib.sha256(span.encode()).hexdigest() == cite["chunk_sha256"]:
                resolved += 1
        row.update(chunk_count=body["chunk_count"], citations_resolved=resolved)
    else:
        row["code"] = ((body or {}).get("error") or {}).get("code") or body
    rows.append(row)
    print(json.dumps(row), flush=True)

big = b"a" * (64 * 1024 * 1024 + 1)
status, body, _ = call("POST", "/api/documents/ingest", {
    "filename": "too-big.txt", "content": base64.b64encode(big).decode(), "is_base64": True})
over = {"bytes": len(big), "status": status, "code": ((body or {}).get("error") or {}).get("code"),
        "message": ((body or {}).get("error") or {}).get("message")}
print(json.dumps(over))

report = {
    "previously_refused_for_size": len(rows),
    "now_ingested": sum(1 for r in rows if r["status"] == 200),
    "now_refused_extract_failed": sum(1 for r in rows if r.get("code") == "extract_failed"),
    "citations_resolved": sum(r.get("citations_resolved", 0) for r in rows),
    "citations": sum(r.get("chunk_count", 0) for r in rows),
    "over_64_mb": over,
    "rows": rows,
}
json.dump(report, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
