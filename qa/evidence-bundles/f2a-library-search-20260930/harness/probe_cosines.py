#!/usr/bin/env python3
"""Cosine of every chunk in the small edge library to a few questions (semantic mode scores are cosines)."""
import json
import subprocess
import urllib.request

BASE = "http://127.0.0.1:8198"
LIB = "/tmp/f2l/lib-probe"
subprocess.run(["rm", "-rf", LIB])
subprocess.run(["mkdir", "-p", LIB])
subprocess.run(["cp", "/tmp/f2l/lib-edge/documents_rag.sqlite3", LIB + "/"], check=True)
subprocess.run(["bash", "/tmp/f2h/serve.sh", "/tmp/f2l/bin/camelid-library", LIB, "8198"], check=True, capture_output=True)
try:
    for q in ["If hackers steal my information, when will you tell me?", "When will you tell me about a data breach?",
              "Who handles a billing dispute?", "How long do refunds take?", "Tell me a joke about penguins.", "hello"]:
        req = urllib.request.Request(BASE + "/api/documents/search", data=json.dumps({"query": q, "top_k": 5, "mode": "semantic"}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        body = json.loads(urllib.request.urlopen(req).read())
        print(q)
        for r in body["results"]:
            print(f"   {r['score']:.4f}  {r['filename']}#{r['chunk_index']}  {r['excerpt'][:70]!r}")
finally:
    subprocess.run(["bash", "/tmp/f2h/stop.sh", LIB], check=True, capture_output=True)
