#!/usr/bin/env python3
"""Re-pin the SmolLM3 runtime seal to the current src/api/mod.rs (run from the checkout root).

The fixture records the git blob sha1 of src/api/mod.rs; two scripts record that sha1 and
the fixture's sha256. Every old value must appear exactly where expected.
"""
import hashlib
import subprocess

FIXTURE = "qa/model-qualification/fixtures/smollm3-default-thinking-runtime-envelope-v1.json"
SCRIPTS = ["scripts/hf-qualification-smollm3-chat-parity.mjs", "scripts/test-hf-qualification-smollm3-chat-parity.mjs"]


def replace(path, old, new, count):
    text = open(path, encoding="utf-8", newline="").read()
    assert text.count(old) == count, f"{path}: {old} appears {text.count(old)} times, expected {count}"
    open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new))


new_blob = subprocess.check_output(["git", "hash-object", "src/api/mod.rs"], text=True).strip()
fixture_text = open(FIXTURE, encoding="utf-8", newline="").read()
old_blob = fixture_text.split('"source_git_blob_sha1": "', 1)[1].split('"', 1)[0]
old_sha = hashlib.sha256(open(FIXTURE, "rb").read()).hexdigest()
if old_blob == new_blob:
    print("seal already pinned to", new_blob)
    raise SystemExit(0)
replace(FIXTURE, old_blob, new_blob, 1)
new_sha = hashlib.sha256(open(FIXTURE, "rb").read()).hexdigest()
for script in SCRIPTS:
    replace(script, old_blob, new_blob, 1)
    replace(script, old_sha, new_sha, 1)
print(f"blob {old_blob} -> {new_blob}")
print(f"fixture sha256 {old_sha} -> {new_sha}")
