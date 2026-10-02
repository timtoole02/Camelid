#!/usr/bin/env python3
"""Re-derive every number in README.md from this bundle's files.

usage: python3 harness/check_claims.py   (from the bundle root)
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []
CHECKS = 0
BUILD = "v0.7.8-27-gecee5ebf"


def check(name, condition, detail=""):
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as handle:
        return handle.read()


def load(rel):
    return json.loads(read(rel))


for line in read("SHA256SUMS").splitlines():
    digest, rel = line.split("  ", 1)
    with open(os.path.join(ROOT, rel), "rb") as handle:
        check(f"sha256 {rel}", hashlib.sha256(handle.read()).hexdigest() == digest)
listed = {line.split("  ", 1)[1] for line in read("SHA256SUMS").splitlines()}
on_disk = {os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")
           for d, _, files in os.walk(ROOT) for f in files if f != "SHA256SUMS"}
check("SHA256SUMS lists every file", listed == on_disk, sorted(listed ^ on_disk))

readme = read("README.md")
check("README names the build", BUILD in readme)

# The live edge suite.
edge = load("data/folder-edge.json")
check("edge: every check passed", edge["passed"] == edge["total"] and all(c["pass"] for c in edge["checks"]), edge["passed"])
check("edge: README count", f"{edge['passed']} of {edge['total']}" in readme, (edge["passed"], edge["total"]))
check("edge: one build", {s["build"] for s in edge["servers"]} == {BUILD}, edge["servers"])
check("edge: the LAN surface ran", any(s["api_surface"] == "lan_chat_only" for s in edge["servers"]))
names = [c["check"] for c in edge["checks"]]
for wanted in ("a request without the web UI's origin is refused (403)",
               "the timer picks up the changes without being asked (within 75 s)",
               "a change made while the server was stopped is picked up when it starts",
               "hidden files, other types and links were not read",
               "a folder that cannot be opened reports why",
               "and its documents are kept",
               "its documents leave the library",
               "the files on disk stay",
               "deleting the collection stops the watch",
               "and leaves its documents in the library",
               "GET /api/folders is refused on the LAN chat surface (403)"):
    check(f"edge has {wanted!r}", wanted in names)
timings = edge["timings"]
check("edge: poll pickup within one interval and a bit", timings["poll_pickup_s"] <= 35, timings)
check("README poll pickup", f"{timings['poll_pickup_s']} s" in readme, timings["poll_pickup_s"])
check("README restart pickup", f"{timings['after_restart_s']} s" in readme, timings["after_restart_s"])

# The UI against the live server.
capture = load("data/capture-folders.json")
check("capture: no page errors", capture["page_errors"] == [])
check("capture: every folder request answered 2xx", capture["folder_responses"] and all(r.split()[-1].startswith("2") for r in capture["folder_responses"]), capture["folder_responses"])
steps = {s["step"]: s for s in capture["steps"]}
check("capture: watched", steps["watched"]["documents"] == ["escalation-guide.txt", "support-policy.txt", "team/notes.md"]
      and steps["watched"]["folder_status"] == ["3 documents. Checked just now. Last check: 3 added."], steps["watched"])
changed = steps["changed on disk"]
check("capture: changed on disk", changed["documents"] == ["onboarding.md", "support-policy.txt", "team/notes.md"]
      and changed["folder_status"][0].startswith("3 documents. ") and changed["folder_status"][0].endswith("Last check: 1 added, 1 updated, 1 removed."), changed)
check("capture: stopped", steps["stopped"]["documents"] == [] and steps["stopped"]["folder_status"] == [], steps["stopped"])
check("README UI pickup", f"{capture['seconds_until_ui_showed_change']} s" in readme, capture["seconds_until_ui_showed_change"])
for shot in ("1-browse.png", "2-watched.png", "3-changed-on-disk.png", "4-stop-watching.png"):
    check(f"screenshot {shot}", os.path.getsize(os.path.join(ROOT, "screenshots", shot)) > 10000)

# Mutations: each one fails at least one unit test, and the tree is restored.
found = re.findall(r"^mutation (\w+): exit (\d+), failing: (.*?)\s*$", read("tests/mutations.txt"), re.M)
mutations = {name: failing for name, _, failing in found}
check("six mutations", len(mutations) == 6, mutations)
check("every mutation is caught", all(failing != "none" for failing in mutations.values()), mutations)
check("tree restored after mutations", "tree restored" in read("tests/mutations.txt"))
check("README mutation count", "6 mutations" in readme)

unit = read("tests/unit.txt")
folder_tests = re.findall(r"^test api::document_folders::tests::(\w+) \.\.\. ok$", unit, re.M)
check("13 folder unit tests pass", len(folder_tests) == 13, folder_tests)
result = re.search(r"test result: ok\. (\d+) passed; 0 failed", unit)
check("api unit tests pass", bool(result))
check("README api count", result and f"{result.group(1)} tests" in readme, result and result.group(1))
smoke = read("tests/smoke.txt")
check("browser smoke", "watched folders browser smoke: 11 checks passed" in smoke)
lint = read("tests/lint.txt")
for key in ("FMT_EXIT=0", "CLIPPY_DEFAULT_EXIT=0", "CLIPPY_ALL_EXIT=0"):
    check(f"lint {key}", key in lint)
ci = read("tests/ci.txt")
check("public-scrub", "JOB_DONE public-scrub steps=13 ok=13 skipped=0 failed=0" in ci)
check("validation-scripts", "JOB_DONE validation-scripts steps=1 ok=1 skipped=0 failed=0" in ci)
check("gates", "validation gates: 47 passed, 0 failed" in ci)
runs = {block.split("\n", 1)[0]: block for block in ci.split("== ")[1:]}
with_path = runs["frontend, folder branch, PUPPETEER_EXECUTABLE_PATH set"]
without_path = runs["frontend, folder branch, PUPPETEER_EXECUTABLE_PATH unset"]
below = runs["frontend, #788 (no folder changes), PUPPETEER_EXECUTABLE_PATH set"]
failed = lambda block: re.findall(r"^FAIL\(\d+\)\s+[\d.]+s\s+(.+)$", block, re.M)
check("with the path set, only the project context smoke fails", failed(with_path) == ["Frontend project and conversation context browser smoke"], failed(with_path))
check("without it, only the divergence view smoke fails", failed(without_path) == ["Frontend divergence view smoke"], failed(without_path))
check("#788 fails the same smoke the same way", failed(below) == ["Frontend project and conversation context browser smoke"], failed(below))
check("the watched folders smoke passed in both runs", all(re.search(r"^OK\s+[\d.]+s\s+Frontend watched folders browser smoke$", block, re.M) for block in (with_path, without_path)))
check("every other step passed", all(re.search(r"JOB_DONE frontend steps=62 ok=60 skipped=1 failed=1", block) for block in (with_path, without_path)))

if FAILURES:
    print("\n".join(FAILURES))
    print(f"{len(FAILURES)} of {CHECKS} claims failed")
    sys.exit(1)
print(f"all {CHECKS} claims hold")
