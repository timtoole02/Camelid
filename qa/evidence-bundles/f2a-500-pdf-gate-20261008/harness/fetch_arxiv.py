"""Download real open-access arXiv PDFs for the F2a 500-PDF gate.

Lists recent submissions across several categories through the arXiv API, then
fetches each PDF from export.arxiv.org with a 3 s pause between requests, as
arXiv asks of automated clients. PDFs are kept out of the repository; the
manifest records each id, title, size and sha256 so the corpus can be rebuilt.

Usage: python3 fetch_arxiv.py <out dir> <target count>
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

OUT, TARGET = sys.argv[1], int(sys.argv[2])
CATEGORIES = ["cs.CL", "cs.LG", "cs.CR", "q-bio.GN", "physics.optics", "math.PR",
              "econ.EM", "astro-ph.EP", "stat.ME", "eess.SP", "q-fin.RM", "cond-mat.mtrl-sci"]
PER_CATEGORY = -(-TARGET // len(CATEGORIES))
MAX_BYTES = 15 * 1024 * 1024
UA = "camelid-f2a-gate/1.0 (local evidence run; contact via github.com/karan68)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}

os.makedirs(os.path.join(OUT, "pdf"), exist_ok=True)
manifest_path = os.path.join(OUT, "arxiv-manifest.json")
manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


candidates = []
for category in CATEGORIES:
    url = ("https://export.arxiv.org/api/query?search_query=cat:" + category +
           f"&start=0&max_results={PER_CATEGORY + 10}&sortBy=submittedDate&sortOrder=descending")
    feed = ET.fromstring(get(url))
    for entry in feed.findall("a:entry", ATOM):
        arxiv_id = entry.find("a:id", ATOM).text.rsplit("/abs/", 1)[1]
        title = " ".join(entry.find("a:title", ATOM).text.split())
        candidates.append((category, arxiv_id, title))
    time.sleep(3)
print(f"{len(candidates)} candidates", flush=True)

seen = set()
for category, arxiv_id, title in candidates:
    have = sum(1 for m in manifest.values() if m.get("status") == "ok")
    if have >= TARGET:
        break
    if arxiv_id in seen or arxiv_id in manifest:
        continue
    seen.add(arxiv_id)
    path = os.path.join(OUT, "pdf", arxiv_id.replace("/", "_") + ".pdf")
    try:
        data = get(f"https://export.arxiv.org/pdf/{arxiv_id}")
        if not data.startswith(b"%PDF"):
            manifest[arxiv_id] = {"status": "not_pdf", "category": category}
        elif len(data) > MAX_BYTES:
            manifest[arxiv_id] = {"status": "too_large", "bytes": len(data), "category": category}
        else:
            open(path, "wb").write(data)
            manifest[arxiv_id] = {"status": "ok", "category": category, "title": title,
                                  "file": os.path.basename(path), "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}
    except Exception as err:  # noqa: BLE001 - recorded, not fatal
        manifest[arxiv_id] = {"status": "error", "error": str(err)[:200], "category": category}
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(arxiv_id, manifest[arxiv_id]["status"], have, flush=True)
    time.sleep(3)

ok = sum(1 for m in manifest.values() if m.get("status") == "ok")
print(f"done: {ok} PDFs", flush=True)
