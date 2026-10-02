#!/usr/bin/env python3
"""Embed every calibration query with the pinned encoder through /v1/embeddings, and check the
vectors reproduce the server's own search scores.

The server's semantic-mode search reports each passage's cosine similarity. For every query, the
cosines recomputed here from these vectors and the stored chunk vectors must match the scores
collect_similarity.py recorded (its top 20 per query, both libraries) to within 1e-4.

usage: embed_queries.py   (env: BIN)   writes calibration/query-vectors.npz and embed-check.json
"""
import json
import os
import sqlite3
import subprocess
import sys
import urllib.request

import numpy as np

sys.path.insert(0, "/tmp/f2l")
import collect_similarity as queries  # noqa: E402

PORT = 8198
BASE = f"http://127.0.0.1:{PORT}"
EVAL = "/mnt/disks/data/camelid-eval"
CALIB = f"{EVAL}/calibration"
LIB = "/tmp/f2l/lib-embed"
ENCODER = "nomic-embed-text-v1.5.Q8_0.gguf"


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as res:
        return json.loads(res.read())


def chunk_vectors(library):
    conn = sqlite3.connect(f"{EVAL}/lib-{library}-1000/documents_rag.sqlite3")
    rows = conn.execute(
        "SELECT c.doc_id, c.chunk_index, v.vector FROM document_chunk_vectors v JOIN document_chunks c ON c.id = v.chunk_id "
        "WHERE v.vector IS NOT NULL AND v.chunk_sha256 = c.chunk_sha256 ORDER BY c.id").fetchall()
    keys = [(doc, idx) for doc, idx, _ in rows]
    matrix = np.stack([np.frombuffer(blob, dtype="<f4") for _, _, blob in rows]).astype(np.float64)
    return keys, matrix


def main():
    texts = {}
    for q in queries.scifact_queries() + queries.fiqa_queries() + queries.offtopic():
        texts[(q["source"], q["id"])] = q["text"]
    keys = sorted(texts)
    subprocess.run(["rm", "-rf", LIB])
    subprocess.run(["bash", "/tmp/f2h/serve.sh", os.environ["BIN"], LIB, str(PORT)], check=True, capture_output=True)
    try:
        post("/api/models/load", {"path": f"models/{ENCODER}", "id": ENCODER, "replace": False, "set_active": False})
        vectors = []
        for start in range(0, len(keys), 64):
            batch = [f"search_query: {texts[k]}" for k in keys[start:start + 64]]
            body = post("/v1/embeddings", {"model": ENCODER, "input": batch, "encoding_format": "float"})
            vectors += [item["embedding"] for item in sorted(body["data"], key=lambda item: item["index"])]
            print(f"embedded {len(vectors)}/{len(keys)}", flush=True)
    finally:
        subprocess.run(["bash", "/tmp/f2h/stop.sh", LIB], check=True, capture_output=True)
    q = np.array(vectors, dtype=np.float64)
    np.savez(f"{CALIB}/query-vectors.npz", vectors=q, sources=np.array([k[0] for k in keys]), ids=np.array([k[1] for k in keys]))

    index = {k: i for i, k in enumerate(keys)}
    worst, compared = 0.0, 0
    for library in ("scifact", "fiqa"):
        chunk_keys, matrix = chunk_vectors(library)
        where = {k: i for i, k in enumerate(chunk_keys)}
        norms = np.linalg.norm(matrix, axis=1)
        for line in open(f"{CALIB}/{library}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            qv = q[index[(r["source"], r["id"])]]
            for doc, chunk, server_cos in r["results"]:
                row = matrix[where[(doc, chunk)]]
                cos = float(row @ qv / (norms[where[(doc, chunk)]] * np.linalg.norm(qv)))
                worst = max(worst, abs(cos - server_cos))
                compared += 1
    report = {"queries": len(keys), "dims": int(q.shape[1]), "server_scores_compared": compared, "max_abs_difference": worst}
    json.dump(report, open(f"{CALIB}/embed-check.json", "w"), indent=1)
    print(json.dumps(report))
    print("EMBED", "OK" if worst < 1e-4 else "MISMATCH")


if __name__ == "__main__":
    main()
