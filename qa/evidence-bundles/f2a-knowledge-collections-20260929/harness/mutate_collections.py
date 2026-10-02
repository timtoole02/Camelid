#!/usr/bin/env python3
"""Prove the knowledge collections smoke is causal: each mutation must make it fail.

Every file is restored byte-for-byte afterwards, whatever happens.
"""
import os
import shutil
import subprocess

ROOT = os.path.expanduser("~/Camelid/frontend")
SMOKE = "scripts/knowledge-collections-browser-smoke.mjs"
CHROME = subprocess.run(
    "ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1",
    shell=True, capture_output=True, text=True,
).stdout.strip()
VIEW = "src/views/ChatWorkspace.jsx"
TURN = "src/components/chat/MessageTurn.jsx"
CONTEXT = "src/lib/projectContext.js"
KNOWLEDGE = "src/lib/knowledgeCollections.js"
LIBRARY = "src/components/knowledge/KnowledgeLibrary.jsx"

MUTATIONS = [
    ("the chat sends members instead of the collection", VIEW,
     "...(searchedCollections.length > 0 ? { collection_ids: searchedCollections.map((c) => c.id) } : {}),",
     "...(searchedCollections.length > 0 ? { doc_ids: searchedCollections.flatMap((c) => c.doc_ids) } : {}),"),
    ("a deleted collection is still searched", VIEW,
     "searchedCollections = collectionRefs.map((ref) => live.get(ref.id)).filter(Boolean)",
     "searchedCollections = collectionRefs.map((ref) => live.get(ref.id) || { id: ref.id, name: ref.id, doc_ids: [] })"),
    ("the LAN surface asks for collections", VIEW,
     "const knowledgeEnabled = !demoMode && Boolean(runtime) && runtime.api_surface !== 'lan_chat_only'",
     "const knowledgeEnabled = !demoMode"),
    ("the LAN surface searches a chat's collections", VIEW,
     "const collectionRefs = knowledgeEnabled ? contextCollectionRefs(chatContext, projects) : []",
     "const collectionRefs = contextCollectionRefs(chatContext, projects)"),
    ("the sent message does not name its collections", TURN,
     "{collections.length > 0 && (",
     "{false && ("),
    ("project collections do not reach the chat", CONTEXT,
     "if (!config.excluded_collection_ids.includes(id)) refs.set(id, { id, from: 'project' })",
     "if (false) refs.set(id, { id, from: 'project' })"),
    ("a per-chat opt-out of a project collection is ignored", CONTEXT,
     "if (!config.excluded_collection_ids.includes(id)) refs.set(id, { id, from: 'project' })",
     "refs.set(id, { id, from: 'project' })"),
    ("an upload ignores its collection", KNOWLEDGE,
     "...(collectionIds.length ? { collection_ids: collectionIds } : {}),",
     ""),
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
        text = open(path, encoding="utf-8", newline="").read()
        assert text.count(old) == 1, f"{label}: anchor must occur exactly once"
        open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))
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
