#!/usr/bin/env python3
"""Prove the library search smoke is causal: each mutation must make it fail.

Every file is restored byte-for-byte afterwards, whatever happens.
"""
import os
import shutil
import subprocess

ROOT = os.path.expanduser("~/Camelid/frontend")
SMOKE = "scripts/library-search-browser-smoke.mjs"
CHROME = subprocess.run(
    "ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1",
    shell=True, capture_output=True, text=True,
).stdout.strip()
VIEW = "src/views/ChatWorkspace.jsx"
TURN = "src/components/chat/MessageTurn.jsx"
CONTEXT = "src/lib/projectContext.js"

MUTATIONS = [
    ("the library flag is never sent", VIEW,
     "res = await search({ ...searchBody, library: true })",
     "res = await search(searchBody)"),
    ("the switch is not saved with the chat", CONTEXT,
     "    search_library: raw?.search_library === true,\n",
     ""),
    ("the LAN surface searches the library", VIEW,
     "const searchLibrary = knowledgeEnabled && normalizeChatContext(chatContext).search_library",
     "const searchLibrary = normalizeChatContext(chatContext).search_library"),
    ("the library claims the attached document's passages", VIEW,
     "passages: requestCitations.filter((citation) => !attachedDocuments.some((doc) => doc.doc_id === citation.doc_id)\n          && !searchedCollections.some((collection) => collection.doc_ids.includes(citation.doc_id))).length,",
     "passages: requestCitations.length,"),
    ("a failed library search is claimed on the message", VIEW,
     "librarySearched = res.ok",
     "librarySearched = true"),
    ("without the encoder, attached documents go unsearched", VIEW,
     "            res = null\n          }\n        }\n        if (!res && pinnedSources) {",
     "          }\n        }\n        if (!res && pinnedSources) {"),
    ("the sent message does not say the library was searched", TURN,
     "        {library && (",
     "        {false && ("),
    ("the missing encoder is not explained", VIEW,
     "{searchLibrary && indexStatus?.semantic && !indexStatus.semantic.available && (",
     "{false && ("),
]


def run(cmd, env=None):
    return subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True, env=env)


env = {**os.environ, "CHROME_PATH": CHROME, "PUPPETEER_EXECUTABLE_PATH": CHROME, "CI": "true"}
caught = 0
for label, rel, old, new in MUTATIONS:
    path = os.path.join(ROOT, rel)
    backup = path + ".mutation-backup"
    shutil.copyfile(path, backup)
    try:
        raw = open(path, "rb").read().decode("utf-8")
        text = raw.replace("\r\n", "\n")
        assert text.count(old) == 1, f"{label}: anchor must occur exactly once ({text.count(old)})"
        open(path, "wb").write(text.replace(old, new).encode("utf-8"))
        build = run("npm run build")
        assert build.returncode == 0, f"{label}: build failed\n{build.stdout[-800:]}"
        smoke = run(f"timeout 300 node {SMOKE}", env)
        failure = next((line for line in (smoke.stderr + smoke.stdout).splitlines() if "AssertionError" in line or "Error:" in line), "")
        if smoke.returncode != 0:
            caught += 1
            print(f"CAUGHT   {label}: {failure.strip()[:170]}", flush=True)
        else:
            print(f"MISSED   {label}", flush=True)
    finally:
        shutil.move(backup, path)

run("npm run build")
print(f"{caught}/{len(MUTATIONS)} mutations caught")
print("MUTATIONS_DONE")
