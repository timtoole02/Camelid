#!/usr/bin/env python3
"""Check that every number and quoted result in this bundle's README comes from
the bundle's own data, and that SHA256SUMS matches the files.

usage: python3 harness/check_claims.py
Prints each failed claim and exits 1 if there is any.
"""
import hashlib
import json
import math
import os
import random
import re
import sys

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []
checked = 0


def path(name):
    return os.path.join(BUNDLE, name)


def load(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def lines(name):
    with open(path(name), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def text(name):
    with open(path(name), encoding="utf-8") as fh:
        return fh.read()


def sha256_file(name):
    with open(path(name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def normalize(value):
    value = re.sub(r"\s+", " ", value)
    return re.sub(r" ([.,;:])", r"\1", value).strip()


README = normalize(text("README.md"))


def claim(label, needle):
    """The README must state `needle` (whitespace-insensitive)."""
    global checked
    checked += 1
    if normalize(needle) not in README:
        failures.append(f"README does not state {label}: {needle!r}")


def fact(label, condition, detail=""):
    """A property of the data that the README relies on."""
    global checked
    checked += 1
    if not condition:
        failures.append(f"{label}: {detail}" if detail else label)


def pct(x):
    return f"{x * 100:.1f}%"


def sim(x):
    return str(round(x, 4))


capture = load("capture-encoder.json")
no_encoder = load("capture-no-encoder.json")
edge = load("edge/edge-library.json")
first = load("first-floor/edge-library-fixed-floor.json")
endpoint = load("endpoint/endpoint-calibration.json")
per_query = lines("endpoint/endpoint-per-query.jsonl")
rule1 = load("calibration/floor.json")
rule2 = load("calibration/size-floor.json")
embed = load("calibration/embed-check.json")
meta = {lib: load(f"calibration/{lib}-meta.json") for lib in ("scifact", "fiqa")}
subset = load("calibration/fiqa-subset.json")
offtopic = lines("calibration/offtopic-provenance.jsonl")

# probe-cosines.txt: a question line, then "   score  file#chunk  'excerpt'" lines.
probes, current = {}, None
for line in text("first-floor/probe-cosines.txt").splitlines():
    m = re.match(r"\s+([0-9.]+)\s+(\S+#\d+)\s+(.*)$", line)
    if m:
        probes[current].append((float(m.group(1)), m.group(2), m.group(3)))
    elif line.strip():
        current = line.strip()
        probes[current] = []


def probe(question, chunk):
    return next(score for score, name, _ in probes[question] if name == chunk)


# ---- builds ------------------------------------------------------------------------------------------
head = capture["build"]
fact("the build under test is d9318fa7", head == "v0.7.8-17-gd9318fa7", head)
claim("the head build", f"`{head}`")
fact("the no-encoder capture used the same build", no_encoder["build"] == head)
fact("every edge server ran the build under test", edge["servers"] and all(s["build"] == head for s in edge["servers"]))
fact("the endpoint replay ran the build under test", endpoint["build"] == f"camelid {head}", endpoint["build"])
first_builds = {s["build"] for s in first["servers"]}
fact("one build ran the first-floor edge run", len(first_builds) == 1, first_builds)
claim("the first-floor build", f"`{first_builds.pop()}`")
calibration_builds = {m["build"] for m in meta.values()}
fact("one build collected the calibration data", len(calibration_builds) == 1, calibration_builds)
claim("the calibration build", f"`{calibration_builds.pop()}`")
smokes = text("ui/smokes.txt")
mutations = text("ui/mutations.txt")
cargo = text("tests/cargo-test.txt")
fact("the UI runs were on the committed tree", smokes.startswith("commit=d9318fa7 tree_clean=yes")
     and mutations.startswith("commit=d9318fa7 tree_clean=yes"))
fact("the Rust suite ran on the committed tree", "commit=d9318fa7 tracked_changes=0" in cargo)
claim("the commit", "commit `d9318fa7`, clean tree")

# ---- calibration data --------------------------------------------------------------------------------
for record, scripts in (("rule-sha256-before-collection.txt", ("analyze_floor.py", "collect_similarity.py")),
                        ("rule-sha256-size-floor.txt", ("analyze_size_floor.py",))):
    recorded = {os.path.basename(p): d for d, p in (l.split("  ", 1) for l in text(f"calibration/{record}").splitlines() if "  " in l)}
    for script in scripts:
        fact(f"harness/{script} matches the hash recorded before it ran", recorded.get(script) == sha256_file(f"harness/{script}"),
             (recorded.get(script), sha256_file(f"harness/{script}")))
        claim(f"the pre-registered {script}", f"`harness/{script}`")
for lib, name in (("scifact", "SciFact"), ("fiqa", "FiQA")):
    m = meta[lib]
    fact(f"{name} is a 1,000-document library", m["documents"] == 1000)
    claim(f"{name}'s chunks", f"{m['indexed_chunks']:,} indexed chunks")
    rows = lines(f"calibration/{lib}.jsonl")
    fact(f"{lib}.jsonl holds every query", len(rows) == m["queries"] == 905, len(rows))
    fact(f"{lib}.jsonl has top-20 results", all(len(r["results"]) <= 20 for r in rows))
claim("the query count", "all 905 queries in semantic mode, top 20")
positives = {lib: sum(r["set"] == "positive" for r in lines(f"calibration/{lib}.jsonl")) for lib in meta}
fact("SciFact has 300 test claims and FiQA 200 questions", positives == {"scifact": 300, "fiqa": 200}, positives)
fact("the FiQA subset", subset["documents"] == 1000 and len(subset["queries"]) == 200 and subset["seed"] == 20260930)
claim("the FiQA subset", f"relevant documents of {len(subset['queries'])} seeded test questions ({subset['relevant_docs']} documents)")
fact("405 unrelated messages", len(offtopic) == 405)
claim("the unrelated count", f"{len(offtopic)} unrelated chat messages")
written = [r for r in offtopic if r["source"] == "author-written"]
claim("the written lines", f"{len(written)} short lines written for this")
fact("the written lines include hello and thanks!", {"hello", "thanks!"} <= {r["text"] for r in written})
fact("every message has a text hash", all(re.fullmatch(r"[0-9a-f]{64}", r["text_sha256"]) for r in offtopic))
fact("the unrelated sources", {r["source"] for r in offtopic} == {"mt_bench", "alpaca", "gsm8k", "author-written"})

# ---- the first rule ----------------------------------------------------------------------------------
fact("the first floor", rule1["floor"] == 0.69)
claim("the first floor", "**0.69**")
h = rule1["held_out"]
claim("the first rule's held-out size", f"({h['positives']} questions, {h['negatives']} unrelated messages)")
claim("the first rule's held-out result", f"served {pct(h['tpr'])} of questions and kept {pct(h['tnr'])} of unrelated messages quiet")
by = rule1["held_out_by_library"]
claim("the first rule by library", f"SciFact {pct(by['scifact']['tpr'])} and {pct(by['scifact']['tnr'])}, "
      f"FiQA {pct(by['fiqa']['tpr'])} and {pct(by['fiqa']['tnr'])}")

# ---- the first live check ----------------------------------------------------------------------------
fact("the first-floor edge counts", first["passed"] == sum(c["pass"] for c in first["checks"]) and first["total"] == len(first["checks"]))
claim("the first-floor edge result", f"{first['passed']} of {first['total']} checks passed")
failed = [c for c in first["checks"] if not c["pass"]]
fact("exactly one first-floor check failed", len(failed) == 1, [c["check"] for c in failed])
claim("the failed check", f"\"{failed[0]['check']}\"")
fact("it failed on the guide's chunk 1", "'filename': 'escalation-guide.txt', 'chunk_index': 1" in failed[0]["detail"])
first_probes = {p["query"]: p for p in first["probes"]}
QUESTION = "If hackers steal my information, when will you tell me?"
fact("the first floor gave the hacker question one passage at 0.69", first_probes[QUESTION]["library_similarities"] == [0.69])
fact("the first floor gave the billing question nothing", first_probes["Who handles a billing dispute?"]["library_results"] == 0)
claim("the three-document library", "ran against a three-document library")
incident = probe(QUESTION, "support-policy.txt#3")
billing = probe("Who handles a billing dispute?", "escalation-guide.txt#1")
fact("the billing question's best passage is the guide's chunk 1", probes["Who handles a billing dispute?"][0][1] == "escalation-guide.txt#1")
claim("the incident and billing scores", f"the incident passage scored {incident:.4f}, the billing passage {billing:.4f}")
unrelated_max = max(max(s for s, _, _ in probes[q]) for q in ("hello", "Tell me a joke about penguins."))
claim("the unrelated probe ceiling", f"scored at most {unrelated_max:.4f}")
direct = probe("When will you tell me about a data breach?", "support-policy.txt#3")
claim("the direct question's score", f"scores it {direct:.4f}")
fact("the incident chunk begins in the refunds section", "not eligible for refunds under any circumstance" in
     next(e for _, n, e in probes[QUESTION] if n == "support-policy.txt#3") and "4. Data export" in
     next(e for _, n, e in probes[QUESTION] if n == "support-policy.txt#3"))

# ---- the second rule ---------------------------------------------------------------------------------
a, b = rule2["a"], rule2["b"]
claim("the fitted rule", f"**floor = {a} + {b} × ln(indexed chunks)**, clamped to {rule2['clamp'][0]}–{rule2['clamp'][1]}")
claim("the sample count", f"{rule2['samples']:,} samples")
fact("the sample count is 1,810 queries x (5 sizes x 5 + 1)", rule2["samples"] == 1810 * (5 * 5 + 1))
label = lambda size: f"{size:,} documents"
for row in rule2["calibration_per_size"]:
    claim(f"the best floor at {row['size']}", f"| {label(row['size'])} | {row['t_star']} | {row['mean_chunks']:,} |")
xs = [r["mean_ln_n"] for r in rule2["calibration_per_size"]]
ys = [r["t_star"] for r in rule2["calibration_per_size"]]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
fact("a and b are the least-squares fit", round(slope, 4) == b and round(my - slope * mx, 4) == a, (slope, my - slope * mx))
ho = rule2["held_out"]
for row in ho["by_size"]:
    s, f = row["size_aware"], row["fixed_0_69"]
    fact(f"same samples at {row['size']}", (s["positives"], s["negatives"]) == (f["positives"], f["negatives"]))
    claim(f"held-out at {row['size']}", f"| {label(row['size'])} | {s['positives']:,} | {s['negatives']:,} | {pct(s['tpr'])} | "
          f"{pct(s['tnr'])} | {pct(f['tpr'])} | {pct(f['tnr'])} |")
    fact(f"the trade at {row['size']}", s["tpr"] > f["tpr"] and s["tnr"] < f["tnr"])
s, f = ho["size_aware"], ho["fixed_0_69"]
claim("held-out overall", f"| All | {s['positives']:,} | {s['negatives']:,} | {pct(s['tpr'])} | {pct(s['tnr'])} | {pct(f['tpr'])} | {pct(f['tnr'])} |")
fact("the first rule's served rate is below 0.69's at 1,000", rule1["held_out"]["tpr"] < ho["by_size"][-1]["fixed_0_69"]["tpr"])
claim("the top-20 explanation", f"The first rule's {pct(rule1['held_out']['tpr'])} is lower than the {pct(ho['by_size'][-1]['fixed_0_69']['tpr'])} here")
claim("the embedding check", f"{embed['server_scores_compared']:,} scores compared, largest difference {embed['max_abs_difference']:.1e}")
fact("the embedding check is within 1e-4", embed["max_abs_difference"] < 1e-4)


def floor_for(chunks):
    return min(rule2["clamp"][1], max(rule2["clamp"][0], a + b * math.log(max(chunks, 1))))


# ---- the endpoint ------------------------------------------------------------------------------------
claim("the endpoint query count", f"all {endpoint['queries']} calibration queries")
for lib, name in (("scifact", "SciFact"), ("fiqa", "FiQA")):
    fact(f"the {name} floor is the rule's", abs(endpoint["floors"][lib] - floor_for(meta[lib]["indexed_chunks"])) < 1e-6)
claim("the endpoint floors", f"applied {endpoint['floors']['scifact']} on SciFact and {endpoint['floors']['fiqa']} on FiQA")
fact("no passage below the floor", endpoint["results_below_the_floor"] == 0 and sum(r["below_floor"] for r in per_query) == 0)
fact("the endpoint agreed on every query", endpoint["endpoint_agrees_with_calibration"] == endpoint["queries"] == len(per_query)
     and all(r["agrees"] for r in per_query))
claim("the agreement", f"for {endpoint['endpoint_agrees_with_calibration']} of {endpoint['queries']} queries")
claim("the chat's passages", f"first {endpoint['chat_top_k']} passages")
groups = {}
for r in per_query:
    groups.setdefault((r["library"], r["set"], r["source"]), []).append(dict(r))
held = []
for members in groups.values():
    members.sort(key=lambda r: r["id"])
    random.Random(rule1["seed"]).shuffle(members)
    held += members[math.ceil(len(members) / 2):]
for lib, name in (("scifact", "SciFact"), ("fiqa", "FiQA"), (None, "Both")):
    rows = [r for r in held if lib is None or r["library"] == lib]
    pos = [r for r in rows if r["set"] == "positive"]
    neg = [r for r in rows if r["set"] == "negative"]
    served, quiet = sum(r["gold_in_chat_passages"] for r in pos), sum(r["returned"] == 0 for r in neg)
    report = endpoint["held_out"] if lib is None else endpoint["held_out_by_library"][lib]
    fact(f"the per-query rows reproduce the {name} held-out rates",
         (len(pos), served, len(neg), quiet) == (report["positives"], report["positives_with_a_relevant_passage_in_the_4"],
                                                 report["negatives"], report["negatives_with_no_passage"]))
    claim(f"the {name} chat rates", f"| {name} | {served} of {len(pos)} ({pct(served / len(pos))}) | {quiet} of {len(neg)} ({pct(quiet / len(neg))}) |")
    if lib:
        fact(f"the {name} quiet rate equals the simulation's at 1,000", round(quiet / len(neg), 4) == ho["by_library_at_1000"][lib]["size_aware"]["tnr"])
fact("the pooled quiet rate equals the simulation's at 1,000", round(endpoint["held_out"]["negatives_with_no_passage"] / endpoint["held_out"]["negatives"], 4)
     == ho["by_size"][-1]["size_aware"]["tnr"])
claim("the mean passages", f"{endpoint['held_out']['negatives_mean_passages']} passages on average")
noisy = endpoint["held_out"]["negatives"] - endpoint["held_out"]["negatives_with_no_passage"]
claim("the noisy count", f"{noisy} of {endpoint['held_out']['negatives']} did at the chat's 4 passages")
for lib, name in (("scifact", "SciFact"), ("fiqa", "FiQA")):
    lat = endpoint["latency"][lib]
    fact(f"{name} latency covers every query", lat["n"] == 905)
    claim(f"{name} latency", f"| {name} | {lat['library']['p50_ms']:,.1f} / {lat['library']['p95_ms']:,.1f} | "
          f"{lat['plain_whole_library']['p50_ms']:,.1f} / {lat['plain_whole_library']['p95_ms']:,.1f} |")

# ---- edge cases --------------------------------------------------------------------------------------
checks = edge["checks"]
fact("the edge counts are consistent", edge["passed"] == sum(c["pass"] for c in checks) and edge["total"] == len(checks))
fact("every edge check passed", edge["passed"] == edge["total"], [c["check"] for c in checks if not c["pass"]])
claim("the edge result", f"{edge['passed']} of {edge['total']} checks pass")
for c in checks:
    m = re.match(r"(?:every library result for|without the flag,) '(.*?)'", c["check"])
    claim(f"edge check {c['check']!r}", f"\"{m.group(1)}\"" if m else c["check"])
    fact(f"edge check {c['check']!r} is one of the two probe forms", not m or c["check"] in (
        f"every library result for {m.group(1)!r} carries a similarity at or above the floor",
        f"without the flag, {m.group(1)!r} is searched as before, with no similarity reported"))
claim("the probe checks", "every library result carries a similarity at or above the floor, and without the flag the same query is searched as before, with no similarity reported")
fact("the first run's failed check is not among them", failed[0]["check"] not in {c["check"] for c in checks})
observed = edge["probes"][0]
fact("the recorded hacker question", observed["query"] == QUESTION and observed["served"] == [["escalation-guide.txt", 1, 0.69, False]], observed)
claim("the reported floor", f"reports a floor of {observed['floor']}")
fact("the reported floor is the rule's for 29 chunks", observed["indexed_chunks"] == 29 and abs(observed["floor"] - floor_for(29)) < 1e-6)
edge_probes = {p["query"]: p for p in edge["probes"][1:]}
fact("the billing question now gets the guide's passage", edge_probes["Who handles a billing dispute?"]["library_similarities"] == [round(billing, 4)])
claim("the billing passage", f"the escalation guide's chunk 1 at {billing:.4f}")
unrelated = ["hello", "thanks!", "Tell me a joke about penguins.", "What is 17 times 23?", "Write a haiku about autumn leaves."]
fact("the five unrelated messages", all(edge_probes[q]["library_results"] == 0 and edge_probes[q]["plain_results"] == 20 for q in unrelated))
fact("every edge probe ran without the flag too", all(p["plain_results"] == 20 for p in edge_probes.values()))
fact("a LAN chat surface served the LAN check", any(s["api_surface"] == "lan_chat_only" and "--lan-chat-only" in s["args"] for s in edge["servers"]))
guide_path = os.path.join(BUNDLE, "..", "f2a-knowledge-collections-20260929", "source", "escalation-guide.txt")
policy_path = os.path.join(BUNDLE, "..", "f2a-verifiable-citations-20260927", "source", "support-policy.txt")
guide = open(guide_path, encoding="utf-8").read()
policy = open(policy_path, encoding="utf-8").read()
fact("the guide's chunk 1 holds the billing rule", capture["library_escalation-guide.txt"]["chunk_count"] == 2
     and guide.index("3. Billing disputes") > guide.index("he outage lasts longer than thirty minutes."))
fact("the guide's chunk 1 says to forward a breach to the security team", "Forward a suspected data breach to the security team" in guide)

# ---- screenshots -------------------------------------------------------------------------------------
for name, source_path, entry in (("support-policy.txt", policy_path, "library_support-policy.txt"),
                                 ("escalation-guide.txt", guide_path, "library_escalation-guide.txt")):
    with open(source_path, "rb") as fh:
        data = fh.read()
    fact(f"the ingested {name} is the repository copy", capture[entry]["sha256"] == hashlib.sha256(data).hexdigest())
    claim(f"{name}'s size", f"{len(data):,}-byte `{name}`")
claim("PROJECT_CONTEXT.md's size", f"{capture['library_PROJECT_CONTEXT.md']['bytes']:,}-byte `docs/PROJECT_CONTEXT.md`")
fact("the capture's library has 29 indexed chunks", sum(d["indexed_chunks"] for d in capture["index_status"]["documents"]) == 29)
fact("the attach menu", capture["attach_menu"] == ["Documents", "Collections", "Whole library"], capture["attach_menu"])
claim("the attach menu", "**Documents**, **Collections** and **Whole library**")
fact("the composer chip", capture["chip"] == ["Whole library3 docs"], capture["chip"])
claim("the composer chip", "**Whole library · 3 docs**")

answered = capture["answered"]
fact("the UI sent only the library flag", answered["searches"] == [{"query": capture["question"], "top_k": 4, "library": True}], answered["searches"])
fact("the replay returned the same passages", answered["replay"]["results"] == answered["citations"])
floor = answered["replay"]["retrieval"]["relevance_floor"]
fact("every passage reached the floor", all(c["similarity"] >= floor for turn in ("answered", "hard") for c in capture[turn]["citations"]))
claim("the passage count", f"It returned {len(answered['citations'])} passages")
for c in answered["citations"]:
    claim(f"passage {c['filename']}#{c['chunk_index']}", f"`{c['filename']}` chunk {c['chunk_index']} ({sim(c['similarity'])})")
fact("the first passage holds the refunds sentence", answered["citations"][0]["holds_expected_passage"])
claim("the refunds sentence", "\"processing takes five business days\"")
fact("the refunds sentence is the policy's", "processing takes five business days" in policy and "within 60 days" in policy)
fact("the message chip", answered["message_chips"] == [f"Whole library{len(answered['citations'])} passages used"], answered["message_chips"])
claim("the message chip", f"**Whole library · {len(answered['citations'])} passages used**")
answer = answered["answer"]
fact("the answer's bullet points", "5 business days" in answer and "14 days" in answer and "Not eligible for refunds" in answer)
fact("the answer leaves out the 60-day window", "60" not in answer)
quote = "the timeframes for refunds are not explicitly stated in the excerpts"
fact("the answer's last sentence", quote in answer)
claim("the answer's last sentence", f"\"{quote}\"")
fact("no citation pills and no citation screenshot", capture["pills"] == [] and capture["citation_modal"] is None
     and capture["expected_citation_number"] == 1 and not os.path.exists(path("screenshots/04-citation-verified.png")))
fact("the answer wrote no citation markers", not re.search(r"\[\d+\]", answer))

hard = capture["hard"]
fact("the hard question's search", hard["searches"] == [{"query": QUESTION, "top_k": 4, "library": True}] and hard["replay"]["results"] == hard["citations"])
fact("the hard question got the guide's chunk 1 only", [(c["filename"], c["chunk_index"]) for c in hard["citations"]] == [("escalation-guide.txt", 1)]
     and not hard["citations"][0]["holds_hard_passage"])
claim("the hard question's passage", f"`escalation-guide.txt` chunk 1 ({sim(hard['citations'][0]['similarity'])})")
fact("the hard chip", hard["message_chips"] == ["Whole library1 passage used"], hard["message_chips"])
claim("the hard chip", "**Whole library · 1 passage used**")
fact("the hard answer opens on refunds and says the breach timing is missing",
     hard["answer"].startswith("Unfortunately, the document excerpts do not provide information on refund processing times")
     and "notified of a data breach" in hard["answer"])
fact("the incident passage was offered for the refunds question", any(c["holds_hard_passage"] for c in answered["citations"]))

other = capture["unrelated"]
fact("the unrelated message's search", other["searches"] == [{"query": "Tell me a joke about penguins.", "top_k": 4, "library": True}], other["searches"])
fact("the unrelated message got nothing", other["citations"] == [] and other["replay"]["results"] == [])
fact("the unrelated chip", other["message_chips"] == ["Whole libraryno passages used"], other["message_chips"])
claim("the unrelated chip", "**Whole library · no passages used**")
fact("the model told a joke", "penguin" in other["answer"].lower())

note = "Whole-library search needs search by meaning. Semantic document search needs nomic-embed-text-v1.5.Q8_0.gguf in the models directory."
fact("the no-encoder note", no_encoder["note"] == note + "Open Models", no_encoder["note"])
claim("the no-encoder note", f"\"{note}\" with an **Open Models** link")
fact("the no-encoder chip", no_encoder["chip"] == ["Whole library1 doc"], no_encoder["chip"])
claim("the no-encoder chip", "**Whole library · 1 doc**")
fact("the no-encoder reason", no_encoder["index_status"]["reason"] == "encoder_not_installed")
claim("the no-encoder reason", "`encoder_not_installed`")
fact("no page errors", capture["page_errors"] == [] and no_encoder["page_errors"] == [])
shots = sorted(os.listdir(path("screenshots")))
fact("the screenshots", shots == ["01-attach-menu.png", "02-composer-whole-library.png", "03-answer-from-the-library.png",
                                  "05-question-below-the-floor.png", "06-unrelated-message.png", "07-without-the-encoder.png"], shots)
for shot in shots:
    claim(f"the section for {shot}", f"### screenshots/{shot}")

# ---- what this does not show -------------------------------------------------------------------------
top = rule1["held_out_negatives_above_floor"][0]
fact("the highest unrelated message over the first floor", top["id"] == "alpaca-23994" and top["library"] == "scifact")
claim("the arguable example", f"`{top['id']}`, about protein structure, scored {top['top']} against SciFact")

# ---- UI tests ----------------------------------------------------------------------------------------
for smoke in ("library-search-browser", "project-context", "knowledge-collections-browser", "project-context-browser",
              "semantic-index-browser", "citations-browser", "document-viewer-browser"):
    fact(f"the {smoke} smoke passed", f"PASS {smoke}\n" in smokes)
    claim(f"the {smoke} smoke", f"`{smoke}`")
fact("no smoke failed", "FAIL" not in smokes and "BUILD_OK" in smokes)
caught = re.search(r"(\d+)/(\d+) mutations caught", mutations)
fact("every mutation was caught", caught and caught.group(1) == caught.group(2) and "MISSED" not in mutations)
claim("the mutation count", f"{caught.group(1)} of {caught.group(2)} deliberate regressions were caught")
fact("the tree was restored", "tree after: 0 tracked changes" in mutations)
for line in mutations.splitlines():
    if line.startswith("CAUGHT"):
        claim(f"mutation {line!r}", "- " + line[len("CAUGHT"):].split(":", 1)[0].strip())

# ---- Rust tests --------------------------------------------------------------------------------------
run = re.search(r"passed (\d+) failed (\d+) ignored (\d+) binaries (\d+)", cargo)
claim("the Rust run", f"{int(run.group(1)):,} passed, {run.group(2)} failed, {run.group(3)} ignored, across {run.group(4)} test binaries")
fact("the one failure is the tokenizer comparison", run.group(2) == "1" and re.findall(r"^test (\S+) \.\.\. FAILED$", cargo, re.M) == ["spm_tokenizer_matches_hf"])
fact("the tinyllama smoke was skipped", "--skip smoke_admits_tinyllama" in cargo)
previous = open(os.path.join(BUNDLE, "..", "f2a-knowledge-collections-20260929", "tests", "cargo-test.txt"), encoding="utf-8").read()
fact("it failed the same way in #785's bundle", "test spm_tokenizer_matches_hf ... FAILED" in previous
     and "panicked at tests/runnable_tokenizer.rs:126:5" in previous and "panicked at tests/runnable_tokenizer.rs:126:5" in cargo)
gates = re.search(r"validation gates: (\d+) passed, (\d+) failed", cargo)
claim("the gates", f"{gates.group(1)} passed, {gates.group(2)} failed")
fact("the release build succeeded", "BUILD_EXIT=0" in cargo and f"camelid {head}" in cargo)

# ---- the checker's own check -------------------------------------------------------------------------
checker = text("tests/check-the-checker.txt")
cc = re.search(r"(\d+)/(\d+) checker mutations caught", checker)
fact("every checker mutation was caught", cc and cc.group(1) == cc.group(2) and "MISSED" not in checker
     and checker.count("\nCAUGHT ") == int(cc.group(2)) and "baseline: all " in checker)
claim("the checker's own check", f"rejecting {cc.group(1)} of {cc.group(2)}")

# ---- checksums ---------------------------------------------------------------------------------------
listed = {}
for line in text("SHA256SUMS").splitlines():
    digest, name = line.split("  ", 1)
    listed[name] = digest
on_disk = sorted(os.path.relpath(os.path.join(root, f), BUNDLE) for root, _, files in os.walk(BUNDLE)
                 for f in files if f != "SHA256SUMS" and "__pycache__" not in root)
fact("SHA256SUMS lists every file", sorted(listed) == on_disk, set(on_disk) ^ set(listed))
for name, digest in listed.items():
    fact(f"checksum of {name}", os.path.exists(path(name)) and sha256_file(name) == digest)

if failures:
    print("\n".join(failures))
    print(f"{len(failures)} of {checked} claims failed")
    sys.exit(1)
print(f"all {checked} claims hold")
