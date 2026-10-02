# F2a — hybrid document retrieval: live evidence

Everything here was produced by real `camelid serve` processes, driven over
HTTP or through the shipped chat UI in headless Chrome. Nothing is mocked or
replayed. `harness/check_claims.py` re-derives every number and quoted result
below from the files in this bundle and fails if any of them disagrees.

Host: NVIDIA L4 (22 GiB), 4-vCPU Intel Xeon @ 2.20 GHz, Ubuntu 22.04. Chat
model: `Llama-3.2-3B-Instruct-Q4_K_M.gguf` on the GPU. Encoder:
`nomic-embed-text-v1.5.Q8_0.gguf` (the pinned artifact, sha256
`3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`) on the
CPU. Builds use the release profile with link-time optimization turned off so
they compile in minutes; that does not change what they compute.

| Run | Build (`/v1/health`) | Files |
| --- | --- | --- |
| Screenshots | `v0.7.8-9-g0477691e` | `screenshots/`, `capture-*.json`, `api-transcript.json` |
| Edge cases | `v0.7.8-9-g0477691e`, and `v0.7.8-5-gc7ea906a` for the downgrade | `edge/edge-results.json` |
| Skip recovery, before and after | `v0.7.8-5-gc7ea906a-dirty` (sources of `63746aa8`) and `v0.7.8-9-g0477691e` | `edge/skip-recovery-causal.json` |
| Late upload, before and after | `v0.7.8-8-g5fb64e0f` and `v0.7.8-9-g0477691e` | `edge/late-upload-causal.json` |
| SciFact ingest and index | `v0.7.8-5-gc7ea906a-dirty` (sources of `63746aa8`) | `scifact/ingest.json`, `scifact/indexing.json` |
| SciFact search and verify | `v0.7.8-9-g0477691e`, same library | `scifact/search.json`, `scifact/per-query.json`, `scifact/verify.json` |
| Keyword mode against the previous build | `v0.7.8-5-gc7ea906a`, same library | `scifact/keyword-vs-previous-build.json` |

`0477691e` is the head of the pull request's code. The `-dirty` build was
started before the first commit existed; its three changed Rust files have
exactly the git blobs of `63746aa8` (`scifact/runs.json`). Three code changes
separate it from the head: the skip-recovery and late-upload fixes tested
below, both in the indexer, and a change that stops keyword ranking from
doing work it does not need (see the latency note under SciFact). None of
them changes a ranking. The SciFact search phase ran on both builds and
`per-query.json` came out byte-identical (sha256
`9d6499f6f92d8ad619a8541d4def858d88031d379134e41b765ca324cca8a459`); only the
latencies differ (`scifact/first-run-search.json` is the first build's).

## Screenshots

The policy document is the same 1,535-byte `support-policy.txt` as
`../f2a-verifiable-citations-20260927/source/support-policy.txt`.

### screenshots/01-indexing-progress.png

The repository `README.md` has just been attached. It splits into 37 chunks,
and the chip reads **indexing 16/37** while the encoder works through them in
the background; keyword search is already available. The index-status call
made at the same moment reports 16 of 37 indexed (`capture-semantic.json`).

### screenshots/02-answer-found-by-meaning.png

With `support-policy.txt` attached and fully indexed (5 of 5 chunks), the
question is *If hackers steal my information, when will you tell me?* No
word of it matches the policy, so keyword search alone finds nothing and
falls back to the document's first four chunks in order. Hybrid search ranks
the incident-handling chunk (chunk 3) first, found by meaning
(`api-transcript.json` records both searches). The sent message's chip reads
**4 passages used**. The first paragraph of the model's answer, verbatim
(`capture-semantic.json`):

> According to the document excerpts, if security incidents that affect
> customer data occur, the account owner will be notified within seventy-two
> hours of confirmation, with a written summary of scope and remediation [1].
> This implies that the company will inform the customer about the incident,
> but the exact timing of when they will notify the customer is not specified.

The first sentence is right and cited. The second contradicts it, and the
second paragraph (visible in the screenshot) repeats the contradiction; that
is the 3B model's reasoning, left in on purpose. Retrieval put the right
passage in front of the model, and the citation lets a reader check it.

### screenshots/03-citation-found-by-meaning.png

Opening `[1]`: the server re-derives the span from the stored source before
showing it (**Verified · bytes 942–1433**), and the footer says **Found by
meaning**. Check the span without trusting the UI:

```sh
head -c 1433 ../f2a-verifiable-citations-20260927/source/support-policy.txt | tail -c +943 | sha256sum
# 2b7d5e6d3a68f4977ef23560ed61ca0fcaa1feca44bdc01689bc91df31460ac3
# = excerpt_sha256 of the first hybrid result in api-transcript.json
```

### screenshots/04-keyword-only-without-encoder.png

A second server whose models directory holds only the chat model. The same
policy is attached; the composer says **Keyword search only. Semantic document
search needs nomic-embed-text-v1.5.Q8_0.gguf in the models directory.** with a
link to the Models page, and index-status reports `encoder_not_installed`
(`capture-keyword-only.json`).

Process memory after each phase, both with the chat model loaded: 3,097,008
KiB with the encoder, 2,777,200 KiB without it (`server-rss-kib-*.txt`). The
two phases did different work, so the difference is an indication, not a
measurement of the encoder alone.

## Retrieval quality: BEIR SciFact

**Data.** BEIR's SciFact archive (`scifact.zip`, sha256
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`, identical
to a fresh download; `scifact/dataset.json`). Embedding all 5,183 abstracts on
this CPU would take hours, so the library holds a fixed 1,000-abstract subset:
all 283 abstracts relevant to a test query, plus 717 others drawn with a fixed
seed (`harness/subset.py`, ids in `scifact/subset-ids.txt`). Each abstract is
ingested as one document through `POST /api/documents/ingest`: 1,000
documents, 4,406 chunks, 0 failures.

**Method.** Each of the 300 test claims is sent to `POST /api/documents/search`
over the whole library in `keyword`, `semantic` and `hybrid` mode with
`top_k: 20` (the API maximum). Chunks are collapsed to their documents in rank
order, and the first 10 documents are scored against the official test qrels.
All 900 searches reported the mode that was asked for.

| Mode | nDCG@10 | Recall@10 | MRR@10 | p50 latency | p95 latency |
| --- | --- | --- | --- | --- | --- |
| keyword (BM25) | 0.7855 | 0.8641 | 0.7652 | 17.8 ms | 25.3 ms |
| semantic | 0.8154 | 0.9027 | 0.7918 | 432.8 ms | 724.5 ms |
| hybrid (the new default) | 0.8499 | 0.9282 | 0.8306 | 441.0 ms | 735.3 ms |

Per query (nDCG@10), hybrid beats keyword on 63 claims, loses on 15 and ties
on 222; the mean gain is 0.0644 (paired bootstrap 95% interval 0.0406 to
0.0889). Hybrid also beats semantic alone: 50 better, 27 worse, mean gain
0.0345 (0.0114 to 0.0580). Semantic alone against keyword is not a clear win
here: mean 0.0299, interval −0.0029 to 0.0625. Fusing the two is what pays.
The intervals resample the 300 claims with replacement (10,000 resamples,
seed 20260929).

These numbers are for this subset and this pipeline. They are not comparable
with published full-corpus BEIR results: the library holds 1,000 of 5,183
abstracts, and it ranks chunks, not whole abstracts.

**Keyword mode is the old search.** Served by the previous build
(`c7ea906a`, which has no `mode`), the same library returns identical top-10
documents for all 300 claims, at 17.6 ms at the median (23.9 ms p95)
(`scifact/keyword-vs-previous-build.json`). The first build of this change
took 37.0 ms in keyword mode, because every search also counted index
coverage and checked citations over the whole candidate pool; the head counts
coverage only when meaning takes part, and checks candidates in rank order
until `top_k` pass.

**Every hybrid result still verifies.** For the first 50 claims, each of the
20 hybrid results was re-resolved through
`POST /api/documents/citation/resolve`: 1,000 results, 0 failures
(`scifact/verify.json`).

**Cost.** Semantic and hybrid search first embed the query on the CPU; they
take about 0.4 s more than keyword search at the median. Indexing 4,406
chunks took 6,217 s wall time (0.71 chunks/s) while the host was also
building and testing; over the last 483 s, after the other jobs had finished,
it indexed 477 chunks (0.99 chunks/s) (`scifact/indexing.json`). The library
database was 25,690,112 bytes, of which the 4,406 vectors take 13,535,232
bytes (768 float32 values each).

## Edge cases

`harness/edge_hybrid.py` drives a fresh library through 32 checks, restarting
the server between phases and asserting each time that the process answering
is the one just started from the binary asked for (`servers` in
`edge/edge-results.json`). 32 of 32 passed:

- **With the encoder:** background indexing; every vector bound to its chunk
  hash, the pinned encoder and 768 dimensions; the paraphrased question found
  by meaning; keyword mode alone falling back to the opening passages; an
  unknown `mode` rejected; semantic mode ranking by meaning only; a tampered
  source serving nothing, not even a meaning match, and restoring it
  restoring the match; a drifted chunk withheld while the rest still serve;
  re-ingesting dropping the old vectors and indexing the new text; deleting a
  document leaving no vector behind; a pre-citation document still found by
  keyword; an upload made while a large file is indexing embedded before the
  rest of it; deleting a document while it is being indexed leaving no vector
  behind.
- **Across restarts:** a chunk whose text does not match its hash skipped, not
  embedded, and embedded once its text is restored; vectors written before a
  shutdown persisting; indexing resuming after a restart.
- **Encoder off or foreign:** with `CAMELID_DOCUMENT_SEMANTIC=0`, status says
  so, auto search is keyword-only, an explicit `hybrid` request is refused
  with 409 instead of silently degraded, and uploads are not embedded; a
  foreign file under the encoder's name is refused as `encoder_mismatch`.
- **Downgrade and back:** the previous build (`c7ea906a`) opens, searches and
  ingests into the library, and its delete still removes the vectors; back on
  this build, documents written meanwhile are indexed on the first status
  call, and `PRAGMA integrity_check` and `foreign_key_check` are clean.

**Skip recovery, before and after the fix** (`edge/skip-recovery-causal.json`,
`harness/causal_skip.py`). The same scenario on both builds: a chunk is
skipped because its text does not match its hash, then the text is restored.
On `63746aa8` the chunk was still skipped after 181.4 s. On `0477691e` it was
embedded after 3.0 s.

**Late upload, before and after the fix** (`edge/late-upload-causal.json`,
`harness/causal_queue.py`). A three-copy `README.md` (111 chunks) is
uploaded, and once its first batch is stored, a one-chunk note. On `5fb64e0f`
the note waited 141.0 s, until all 111 chunks of the large file were
embedded. On `0477691e` it was embedded after 51.4 s, with 47 of 111 chunks
of the large file done.

## Reproduce and check

```sh
python3 harness/check_claims.py
# optional: recompute every per-query metric from the ranked lists and BEIR's qrels
SCIFACT_DIR=path/to/unzipped/scifact python3 harness/check_claims.py
```

The harness is included exactly as it ran. It uses a scratch directory,
`/tmp/f2h`, for binaries, libraries and helper scripts, and
`/mnt/disks/data/camelid-eval` for the SciFact data and library.

| Script | What it does |
| --- | --- |
| `run_scifact.sh` | first run: ingest, index, search, verify (`eval_scifact.py`) |
| `run_all_evidence.sh` | every later run on the head, one after another |
| `run_scifact_head.sh` | the head's search and verify on the same library, then `run_base_compare.sh` |
| `base_keyword_compare.py` | the previous build's rankings and latency against keyword mode |
| `run_final_edge.sh` | `edge_hybrid.py`, `causal_skip.py` and `causal_queue.py` |
| `capture_hybrid_run.sh` | the screenshots, through `capture_hybrid.mjs` |
| `subset.py`, `make_scifact_meta.py`, `check_scifact_source.py` | the SciFact subset and its provenance |
| `make_runs.py` | `scifact/runs.json` |
| `serve.sh`, `stop.sh`, `bg.sh` | start a server and wait for health; stop it by recorded pid; detach a job |
| `check_claims.py` | the check above |

`serve.sh` gained its busy-port check, and `stop.sh` its match on renamed
binaries, after the first SciFact run had started; every later run used the
versions here.
