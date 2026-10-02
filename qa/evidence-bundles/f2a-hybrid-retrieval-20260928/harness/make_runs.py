#!/usr/bin/env python3
"""Write scifact/runs.json: which build ran which phase, on what host state."""
import hashlib
import json

R = "/tmp/f2h/results"


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


runs = {
    "first_run": {
        "phases": ["ingest", "index", "search", "verify"],
        "build": "v0.7.8-5-gc7ea906a-dirty",
        "build_note": "started before the first code commit existed, from a working tree whose Rust changes "
                      "were exactly these three files; their git blob ids equal the blobs at 63746aa8 "
                      "(git ls-tree 63746aa8 src/api)",
        "rust_blobs": dict(zip(
            ["src/api/documents.rs", "src/api/document_vectors.rs", "src/api/mod.rs"],
            read(f"{R}/first/source-blobs.txt").split())),
        "host_shared_with": "release builds, the library test suite and the live edge suite during part of the index phase",
        "index_phase_host_idle_from_s": 5719,
        "index_phase_host_idle_note": "the last other job on the host (the edge suite) had finished when the index phase reported 5719 s",
        "per_query_sha256": sha256(f"{R}/first/per-query.json"),
        "server_rss_kib_at_end": int(read(f"{R}/first/server-rss-kib.txt")),
        "library_db_bytes_at_end": int(read(f"{R}/first/db-bytes.txt").split()[0]),
    },
    "head_run": {
        "phases": ["index (confirms every chunk already indexed)", "search", "verify"],
        "build": read(f"{R}/head/build.txt").split()[-1],
        "library": "the first run's library, served by the head build",
        "uptime_before": read(f"{R}/head/load-before.txt"),
        "uptime_after": read(f"{R}/head/load-after.txt"),
        "per_query_sha256": sha256(f"{R}/head/per-query.json"),
        "server_rss_kib_at_end": int(read(f"{R}/head/server-rss-kib.txt")),
        "library_db_bytes_at_end": int(read(f"{R}/head/db-bytes.txt").split()[0]),
    },
}
with open("/tmp/f2h/bundle-final/scifact/runs.json", "w", encoding="utf-8") as fh:
    json.dump(runs, fh, indent=2)
print(json.dumps(runs, indent=2))
