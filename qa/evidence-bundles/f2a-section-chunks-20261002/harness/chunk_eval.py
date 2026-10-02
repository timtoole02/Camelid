#!/usr/bin/env python3
"""Evaluate section-aware chunking with a real server build.

1. Probe: the three-document edge library (support policy, escalation guide,
   PROJECT_CONTEXT.md as of d9318fa7), indexed with the pinned encoder. For each
   question: the top passages by meaning with their cosines, and what a
   whole-library search serves at this library's floor.
2. Calibration libraries: every SciFact and FiQA document is re-ingested from its
   stored canonical text (search by meaning off, so nothing is embedded), and its
   chunk spans are compared with the spans the calibration was collected on.

usage: chunk_eval.py   (env: BIN, TAG, OUT)
"""
import json
import os
import sqlite3
import subprocess
import time
import urllib.request

BIN, TAG, OUT = os.environ["BIN"], os.environ["TAG"], os.environ["OUT"]
PORT = 8196
BASE = f"http://127.0.0.1:{PORT}"
REPO = os.path.expanduser("~/Camelid")
EVAL = "/mnt/disks/data/camelid-eval"
QUESTIONS = [
    "If hackers steal my information, when will you tell me?",
    "When will you tell me about a data breach?",
    "Who handles a billing dispute?",
    "How long do refunds take?",
    "hello", "thanks!", "Tell me a joke about penguins.", "What is 17 times 23?", "Write a haiku about autumn leaves.",
]


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return json.loads(res.read())


def serve(data, env=None):
    subprocess.run(["rm", "-rf", data], check=True)
    subprocess.run(["bash", "/tmp/f2h/serve.sh", BIN, data, str(PORT)], check=True, capture_output=True,
                   env={**os.environ, **(env or {})})


def stop(data):
    subprocess.run(["bash", "/tmp/f2h/stop.sh", data], check=True, capture_output=True)


def git_show(rev_path):
    return subprocess.run(["git", "show", rev_path], cwd=REPO, capture_output=True, text=True, check=True).stdout


report = {"tag": TAG, "binary": os.path.basename(BIN)}

# ---- 1. probe ------------------------------------------------------------------------------------
data = f"/tmp/f2r/lib-chunk-probe-{TAG}"
serve(data)
report["build"] = call("GET", "/v1/health")["build"]
docs = {
    "support-policy.txt": open(f"{REPO}/qa/evidence-bundles/f2a-verifiable-citations-20260927/source/support-policy.txt").read(),
    "escalation-guide.txt": git_show("feat/f2-collections:qa/evidence-bundles/f2a-knowledge-collections-20260929/source/escalation-guide.txt"),
    "PROJECT_CONTEXT.md": git_show("d9318fa7:docs/PROJECT_CONTEXT.md"),
}
ingested = {name: call("POST", "/api/documents/ingest", {"filename": name, "content": text}) for name, text in docs.items()}
deadline = time.time() + 900
while True:
    status = call("GET", "/api/documents/index-status")
    if all(d["indexed_chunks"] + d["skipped_chunks"] >= d["indexable_chunks"] for d in status["documents"]) or time.time() > deadline:
        break
    time.sleep(2)
report["chunks"] = {name: body["chunk_count"] for name, body in ingested.items()}
probes = []
for question in QUESTIONS:
    semantic = call("POST", "/api/documents/search", {"query": question, "top_k": 6, "mode": "semantic"})
    library = call("POST", "/api/documents/search", {"query": question, "top_k": 4, "library": True})
    probes.append({
        "question": question,
        "top_by_meaning": [[r["filename"], r["chunk_index"], round(r["score"], 4), r["excerpt"][:60]] for r in semantic["results"]],
        "floor": library["retrieval"].get("relevance_floor"),
        "library_served": [[r["filename"], r["chunk_index"], round(r["similarity"], 4)] for r in library["results"]],
    })
    print(json.dumps(probes[-1], ensure_ascii=False)[:400], flush=True)
report["probes"] = probes
stop(data)

# ---- 2. calibration libraries -------------------------------------------------------------------
report["calibration_libraries"] = {}
for lib in ("scifact", "fiqa"):
    old = sqlite3.connect(f"file:{EVAL}/lib-{lib}-1000/documents_rag.sqlite3?mode=ro", uri=True)
    rows = old.execute("SELECT id, filename, source_text FROM documents ORDER BY id").fetchall()
    old_spans = {}
    for doc_id, start, end in old.execute("SELECT doc_id, byte_start, byte_end FROM document_chunks ORDER BY doc_id, chunk_index"):
        old_spans.setdefault(doc_id, []).append((start, end))
    data = f"/tmp/f2r/lib-chunk-{lib}-{TAG}"
    serve(data, {"CAMELID_DOCUMENT_SEMANTIC": "0"})
    for doc_id, filename, text in rows:
        call("POST", "/api/documents/ingest", {"doc_id": doc_id, "filename": filename, "content": text})
    stop(data)
    new = sqlite3.connect(f"file:{data}/documents_rag.sqlite3?mode=ro", uri=True)
    new_spans = {}
    for doc_id, start, end in new.execute("SELECT doc_id, byte_start, byte_end FROM document_chunks ORDER BY doc_id, chunk_index"):
        new_spans.setdefault(doc_id, []).append((start, end))
    changed = [doc_id for doc_id, _, _ in rows if old_spans.get(doc_id) != new_spans.get(doc_id)]
    report["calibration_libraries"][lib] = {
        "documents": len(rows), "documents_with_changed_chunks": len(changed),
        "chunks_before": sum(len(v) for v in old_spans.values()), "chunks_after": sum(len(v) for v in new_spans.values()),
        "changed_examples": changed[:5],
    }
    print(lib, json.dumps(report["calibration_libraries"][lib]), flush=True)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, f"chunk-eval-{TAG}.json"), "w") as fh:
    json.dump(report, fh, indent=1, ensure_ascii=False)
print("CHUNK_EVAL_DONE")
