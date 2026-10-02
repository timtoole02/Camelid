#!/usr/bin/env python3
"""Check that every number and quoted result in this bundle's README comes from the bundle's own data.

usage: python3 harness/check_claims.py
       SCIFACT_DIR=path/to/unzipped/scifact python3 harness/check_claims.py
         also recomputes every per-query SciFact metric from the ranked lists and BEIR's test qrels.
Prints each failed claim and exits 1 if there is any.
"""
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys

BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY = os.path.join(BUNDLE, "..", "f2a-verifiable-citations-20260927", "source", "support-policy.txt")
MODES = ("keyword", "semantic", "hybrid")
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_RESAMPLES = 10000
failures = []
checked = 0


def path(name):
    return os.path.join(BUNDLE, name)


def load(name):
    with open(path(name), encoding="utf-8") as fh:
        return json.load(fh)


def text(name):
    with open(path(name), encoding="utf-8") as fh:
        return fh.read()


def sha256_file(name):
    with open(path(name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def normalize(value):
    value = value.replace("\\\n", " ")
    value = re.sub(r"\s+", " ", value)
    return re.sub(r" ([.,;:])", r"\1", value).strip()


README = normalize(re.sub(r"(?m)^> ?", "", text("README.md")))


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


def thousands(value):
    return f"{value:,}"


# ---- builds -------------------------------------------------------------------------------------------
runs = load("scifact/runs.json")
edge = load("edge/edge-results.json")
causal = load("edge/skip-recovery-causal.json")
semantic_capture = load("capture-semantic.json")
keyword_capture = load("capture-keyword-only.json")
head = runs["head_run"]["build"]
previous = "v0.7.8-5-gc7ea906a"
first = runs["first_run"]["build"]
fact("screenshots come from the head build", semantic_capture["build"] == head == keyword_capture["build"],
     (semantic_capture["build"], keyword_capture["build"], head))
fact("the edge suite ran the head build, and the previous build only for the downgrade",
     {s["build"] for s in edge["servers"] if s["binary"] == "camelid-head"} == {head}
     and {s["build"] for s in edge["servers"] if s["binary"] == "camelid-base"} == {previous})
fact("the edge suite really restarted: every server had its own pid", len({s["pid"] for s in edge["servers"]}) == len(edge["servers"]))
fact("skip recovery compared the first build with the head build",
     causal["before_fix"]["build"] == first and causal["with_fix"]["build"] == head)
for label, value in (("head build", head), ("previous build", previous), ("first-run build", first)):
    claim(label, f"`{value}`")

# ---- screenshots --------------------------------------------------------------------------------------
progress = semantic_capture["progress_status"]["document"]
chip = semantic_capture["progress_chip"]
fact("the chip's count matches index-status at that moment",
     chip == f"indexing {progress['indexed_chunks']}/{progress['indexable_chunks']}", (chip, progress))
claim("the progress chip", f"**{chip}**")
claim("the progress count", f"reports {progress['indexed_chunks']} of {progress['indexable_chunks']} indexed")
claim("the README chunk count", f"splits into {progress['indexable_chunks']} chunks")
policy_status = semantic_capture["policy_status"]
fact("the policy was fully indexed", policy_status["indexed_chunks"] == policy_status["indexable_chunks"])
claim("policy coverage", f"({policy_status['indexed_chunks']} of {policy_status['indexable_chunks']} chunks)")
with open(POLICY, "rb") as fh:
    policy_bytes = fh.read()
claim("the policy size", f"{thousands(len(policy_bytes))}-byte")

transcript = {entry["request"]["mode"]: entry for entry in load("api-transcript.json")}
keyword_entry, hybrid_entry = transcript["keyword"], transcript["hybrid"]
fact("keyword mode fell back to the first four chunks in order",
     keyword_entry["retrieval"]["mode"] == "attached"
     and [r["chunk_index"] for r in keyword_entry["results"]] == [0, 1, 2, 3])
top = hybrid_entry["results"][0]
fact("hybrid ranked the incident chunk first, by meaning",
     hybrid_entry["retrieval"]["mode"] == "hybrid" and top["retrieval"] == "semantic" and top["holds_expected_passage"])
claim("the incident chunk", f"(chunk {top['chunk_index']}) first, found by meaning")
passages = [c for c in semantic_capture["message_chips"] if "passages used" in c]
fact("the message chip names the passages", len(passages) == 1)
claim("the message chip", f"**{passages[0].split('.txt')[-1]}**")
claim("the answer's first paragraph", semantic_capture["answer"])
modal = semantic_capture["citation_modal"]
claim("the citation badge", f"**{modal['badge']}**")
claim("the retrieval note", f"**{modal['found']}**")
span = re.search(r"bytes (\d+)–(\d+)", modal["badge"])
start, end = int(span.group(1)), int(span.group(2))
span_sha = hashlib.sha256(policy_bytes[start:end]).hexdigest()
fact("the verified span hashes to the first hybrid result's excerpt", span_sha == top["excerpt_sha256"], span_sha)
fact("the span is what the modal showed", policy_bytes[start:end].decode("utf-8") == modal["span"])
claim("the span hash", span_sha)
claim("the span command", f"head -c {end} ../f2a-verifiable-citations-20260927/source/support-policy.txt | tail -c +{start + 1} | sha256sum")
note = keyword_capture["note"].removesuffix("Open Models")
claim("the keyword-only note", f"**{note}**")
fact("index-status said why", keyword_capture["index_status"]["reason"] == "encoder_not_installed")
claim("the keyword-only reason", "`encoder_not_installed`")
for phase in ("with", "without"):
    rss = int(text(f"server-rss-kib-{phase}-encoder.txt").strip())
    claim(f"memory {phase} the encoder", f"{thousands(rss)} KiB")

# ---- SciFact ------------------------------------------------------------------------------------------
dataset = load("scifact/dataset.json")
ingest = load("scifact/ingest.json")
indexing = load("scifact/indexing.json")
search = load("scifact/search.json")
first_search = load("scifact/first-run-search.json")
per_query = load("scifact/per-query.json")
verify = load("scifact/verify.json")
previous_keyword = load("scifact/keyword-vs-previous-build.json")
ids = text("scifact/subset-ids.txt").split()

claim("the archive hash", dataset["archive_sha256"])
fact("the subset file is the one ingested", dataset["subset"]["file_sha256"] == ingest["corpus_sha256"])
fact("the id list matches the subset", len(ids) == len(set(ids)) == dataset["subset"]["documents"] == ingest["documents"])
fact("every relevant abstract is in the subset", dataset["subset"]["relevant_included"] == dataset["test_relevant_documents"])
claim("the corpus size", f"all {thousands(dataset['corpus_documents'])} abstracts")
claim("the relevant count", f"all {dataset['test_relevant_documents']} abstracts relevant")
claim("the distractor count", f"plus {len(ids) - dataset['test_relevant_documents']} others")
fact("ingest had no failures", ingest["failures"] == [])
claim("the ingest totals", f"{thousands(ingest['documents'])} documents, {thousands(ingest['chunks'])} chunks, 0 failures")
fact("the head run searched every test claim", search["queries"] == len(per_query) == dataset["test_queries"])
fact("every search reported the mode asked for",
     all(search["ranked_by"][m] == {m: search["queries"]} for m in MODES))
claim("the search count", f"All {len(MODES) * search['queries']} searches reported the mode that was asked for")

qids = sorted(per_query, key=int)
for mode in MODES:
    for metric in ("ndcg10", "recall10", "mrr10"):
        mean = round(statistics.fmean(per_query[q][mode][metric] for q in qids), 4)
        fact(f"{mode} {metric} is the mean of the per-query values", mean == search["metrics"][mode][metric],
             (mean, search["metrics"][mode][metric]))
    m, lat = search["metrics"][mode], search["latency_ms"][mode]
    claim(f"the {mode} row", f"| {m['ndcg10']:.4f} | {m['recall10']:.4f} | {m['mrr10']:.4f} | "
                             f"{lat['p50']:.1f} ms | {lat['p95']:.1f} ms |")
fact("both builds produced the same rankings",
     runs["first_run"]["per_query_sha256"] == runs["head_run"]["per_query_sha256"] == sha256_file("scifact/per-query.json"))
claim("the per-query hash", runs["head_run"]["per_query_sha256"])
fact("the first run's summary metrics equal the head run's", first_search["metrics"] == search["metrics"])
extra = {mode: (search["latency_ms"][mode]["p50"] - search["latency_ms"]["keyword"]["p50"]) / 1000 for mode in ("semantic", "hybrid")}
fact("semantic and hybrid both add about 0.4 s at the median", {round(v, 1) for v in extra.values()} == {0.4}, extra)
claim("the added latency", "about 0.4 s more than keyword search at the median")


def paired(a, b):
    diffs = [per_query[q][a]["ndcg10"] - per_query[q][b]["ndcg10"] for q in qids]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP_RESAMPLES))
    low, high = means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return (sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), sum(d == 0 for d in diffs),
            statistics.fmean(diffs), low, high)


better, worse, same, mean, low, high = paired("hybrid", "keyword")
fact("the summary's better/worse/same agrees", search["hybrid_vs_keyword_ndcg10"] == {"better": better, "worse": worse, "same": same})
claim("hybrid vs keyword", f"hybrid beats keyword on {better} claims, loses on {worse} and ties on {same}; "
                           f"the mean gain is {mean:.4f} (paired bootstrap 95% interval {low:.4f} to {high:.4f})")
better, worse, _, mean, low, high = paired("hybrid", "semantic")
claim("hybrid vs semantic", f"{better} better, {worse} worse, mean gain {mean:.4f} ({low:.4f} to {high:.4f})")
_, _, _, mean, low, high = paired("semantic", "keyword")
claim("semantic vs keyword", f"mean {mean:.4f}, interval {low:.4f} to {high:.4f}".replace("-", "−"))
claim("the bootstrap recipe", f"{thousands(BOOTSTRAP_RESAMPLES)} resamples, seed {BOOTSTRAP_SEED}")

fact("verify re-resolved 50 claims x 20 results with no failure", verify["checked"] == 1000 and verify["failures"] == [])
claim("the verify totals", f"{thousands(verify['checked'])} results, {len(verify['failures'])} failures")

fact("the previous build's rankings equal keyword mode for every claim",
     previous_keyword["identical_top10_documents"] == previous_keyword["queries"] == len(qids)
     and previous_keyword["differing"] == [] and previous_keyword["previous_build"] == previous)
claim("the previous-build comparison", f"identical top-10 documents for all {previous_keyword['queries']} claims")
plat = previous_keyword["previous_build_latency_ms"]
claim("the previous build's keyword latency", f"{plat['p50']:.1f} ms at the median ({plat['p95']:.1f} ms p95)")
claim("the first build's keyword latency", f"{first_search['latency_ms']['keyword']['p50']:.1f} ms")

samples = indexing["samples"]
fact("indexing finished", indexing["indexed_chunks"] == indexing["indexable_chunks"] == ingest["chunks"] and indexing["skipped_chunks"] == 0)
total_s = samples[-1][0]
claim("the indexing wall time", f"{thousands(indexing['indexed_chunks'])} chunks took {thousands(round(total_s))} s wall time "
                                f"({indexing['indexed_chunks'] / total_s:.2f} chunks/s)")
idle_from = runs["first_run"]["index_phase_host_idle_from_s"]
window_start = next(s for s in samples if s[0] >= idle_from)
window_s, window_chunks = samples[-1][0] - window_start[0], samples[-1][1] - window_start[1]
claim("the idle-host indexing rate", f"over the last {round(window_s)} s, after the other jobs had finished, it indexed "
                                     f"{window_chunks} chunks ({window_chunks / window_s:.2f} chunks/s)")
db_bytes = runs["head_run"]["library_db_bytes_at_end"]
vector_bytes = ingest["chunks"] * 768 * 4
claim("the database size", f"{thousands(db_bytes)} bytes, of which the {thousands(ingest['chunks'])} vectors take {thousands(vector_bytes)} bytes")

# ---- edge cases ---------------------------------------------------------------------------------------
fact("every edge check passed", edge["passed"] == edge["total"] == len(edge["checks"]) and all(c["pass"] for c in edge["checks"]))
claim("the edge total", f"{edge['passed']} of {edge['total']} passed")
claim("the edge count", f"through {edge['total']} checks")
fact("before the fix the chunk stayed skipped; with it, it was embedded",
     not causal["before_fix"]["recovered"] and causal["with_fix"]["recovered"] and causal["causal"])
claim("the before-fix wait", f"still skipped after {causal['before_fix']['seconds_waited']:.1f} s")
claim("the with-fix time", f"embedded after {causal['with_fix']['seconds_waited']:.1f} s")

queue = load("edge/late-upload-causal.json")
before, after = queue["before_fix"], queue["with_fix"]
total = before["large_when_note_embedded"]["indexable_chunks"]
fact("late upload compared the previous head with the head", before["build"] == "v0.7.8-8-g5fb64e0f" and after["build"] == head)
fact("before the fix the note waited for the whole large file",
     not before["note_went_first"] and before["large_when_note_embedded"]["indexed_chunks"] == total)
fact("with the fix the note went first", after["note_went_first"] and queue["causal"]
     and after["large_when_note_embedded"]["indexable_chunks"] == total)
claim("the large file size", f"three-copy `README.md` ({total} chunks)")
claim("the late upload before the fix", f"the note waited {before['note_embedded_after_s']:.1f} s, until all {total} chunks "
                                         "of the large file were embedded")
claim("the late upload with the fix", f"embedded after {after['note_embedded_after_s']:.1f} s, with "
                                      f"{after['large_when_note_embedded']['indexed_chunks']} of {total} chunks of the large file done")

# ---- checksums ----------------------------------------------------------------------------------------
listed = {}
for line in text("SHA256SUMS").splitlines():
    digest, name = line.split(maxsplit=1)
    listed[name.strip()] = digest
on_disk = sorted(
    os.path.relpath(os.path.join(root, name), BUNDLE)
    for root, _, names in os.walk(BUNDLE) for name in names if name != "SHA256SUMS")
fact("SHA256SUMS lists every file", sorted(listed) == on_disk, set(listed) ^ set(on_disk))
for name, digest in listed.items():
    fact(f"checksum of {name}", os.path.exists(path(name)) and sha256_file(name) == digest)

# ---- optional: recompute the metrics from BEIR's qrels -------------------------------------------------
scifact_dir = os.environ.get("SCIFACT_DIR")
if scifact_dir:
    with open(os.path.join(scifact_dir, "qrels", "test.tsv"), "rb") as fh:
        qrels_bytes = fh.read()
    fact("the qrels are BEIR's", hashlib.sha256(qrels_bytes).hexdigest() == dataset["files_sha256"]["qrels/test.tsv"])
    relevant = {}
    for line in qrels_bytes.decode("utf-8").splitlines()[1:]:
        qid, did, score = line.split("\t")
        if int(score) > 0:
            relevant.setdefault(qid, {})[did] = int(score)
    for qid in qids:
        rel = relevant[qid]
        for mode in MODES:
            ranked = per_query[qid][mode]["ranked"]
            dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel.values(), reverse=True)[:10]))
            first_hit = next((i + 1 for i, d in enumerate(ranked[:10]) if d in rel), None)
            recomputed = {"ndcg10": dcg / idcg, "recall10": len(set(ranked[:10]) & set(rel)) / len(rel),
                          "mrr10": 1 / first_hit if first_hit else 0.0}
            for metric, value in recomputed.items():
                fact(f"query {qid} {mode} {metric} recomputed", abs(value - per_query[qid][mode][metric]) < 1e-12)

for failure in failures:
    print("FAIL", failure)
print(f"{checked - len(failures)}/{checked} claims hold" + ("" if scifact_dir else " (set SCIFACT_DIR to also recompute metrics)"))
sys.exit(1 if failures else 0)
