# Embeddings, reranking, and Workspace semantic retrieval

Camelid's first bidirectional encoder lane is deliberately exact-row scoped:

- repository: `nomic-ai/nomic-embed-text-v1.5-GGUF`
- file: `nomic-embed-text-v1.5.Q8_0.gguf`
- size: `146146432` bytes
- SHA-256: `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`
- GGUF architecture: `nomic-bert`

The CPU runtime implements BERT WordPiece tokenization, token/type embeddings,
bidirectional multi-head attention with split-half RoPE, the Nomic parallel
gated-SiLU feed-forward block, post-residual LayerNorm, GGUF pooling metadata,
L2 normalization, and Matryoshka truncation. Q8_0 matrices remain quantized and
execute through Camelid's block-backed Q8 linear path.

## Experimental Microsoft BitNet embeddings

Two additional exact official artifacts are admitted as experimental rows with
cleanroom CPU, Metal, and CUDA `I2_S` projection kernels:

- `microsoft/BitNet-embedding-0.6B/bitnet-embeddings-0.6b-bf16-i2_s.gguf`
  (427,935,008 bytes; SHA-256
  `c89c64f05a2d3f83565250a6762640197fc624df866d4a1bd5853f811219af17`)
- `microsoft/BitNet-embedding-270M/bitnet-embeddings-270m-bf16-i2_s.gguf`
  (367,487,040 bytes; SHA-256
  `8ee5ae971b103cd55758934be54e5c9f7cc2b58b15890615acce8e649988c751`)

They reuse the `qwen3` and `gemma3` architecture identifiers but add seven
per-projection RMSNorm inputs to every layer. Camelid requires the complete norm
set, `I2_S` for every dense projection, the pinned official geometry, and no
language-model output tensor. The files run through `/v1/embeddings`,
`/v1/rerank`, and the bounded Workspace semantic retriever; generative endpoints
fail closed.

The projection runtime keeps the official canonical `I2_S` bytes and exposes
direct `i2_s`, 9-entry `tl1`, and 27-entry `tl2` lookup strategies via
`CAMELID_BITNET_KERNEL`. `CAMELID_BITNET_GPU=0` forces the CPU oracle. See
[BITNET.md](BITNET.md) for the wire-format boundary and hardware tests.

The published model cards describe last-token pooling, while both current GGUF
artifacts declare `pooling_type=1` (mean). Camelid follows the executable GGUF
metadata and L2-normalizes the result. This discrepancy and reference-vector
parity are explicit promotion blockers, so neither BitNet embedding row is a
supported row yet.

Batch execution uses at most eight encoder workers by default so large
Workspace indexes cannot fan out without bound. Set
`CAMELID_EMBEDDING_BATCH_WORKERS` to an integer from 1 through 16 to tune that
CPU/RSS tradeoff; values outside the range use the default.

## HTTP API

Register the encoder without replacing or activating the current generation
model:

```json
POST /api/models/load
{
  "path": "models/nomic-embed-text-v1.5.Q8_0.gguf",
  "id": "nomic-embed-text-v1.5.Q8_0.gguf",
  "replace": false,
  "set_active": false
}
```

OpenAI-compatible embeddings:

```json
POST /v1/embeddings
{
  "model": "nomic-embed-text-v1.5.Q8_0.gguf",
  "input": [
    "search_query: Which files implement semantic retrieval?",
    "search_document: src/chat/semantic_search.rs implements the Workspace index."
  ],
  "encoding_format": "float",
  "dimensions": 256
}
```

`/embedding` and `/embeddings` are aliases. Input is one string or up to 256
strings. `dimensions` must be between 1 and 768. Truncated vectors are
re-normalized. Base64 and token-ID input fail closed.

Embedding-similarity reranking:

```json
POST /v1/rerank
{
  "model": "nomic-embed-text-v1.5.Q8_0.gguf",
  "query": "Where is Workspace semantic search implemented?",
  "documents": [
    "src/chat/semantic_search.rs builds and searches source chunks.",
    {"text": "src/grammar.rs adapts LLGuidance constraints."}
  ],
  "top_n": 2,
  "return_documents": true
}
```

The reranker applies Nomic's `search_query:` and `search_document:` prefixes
when callers omit them, computes cosine similarity, and returns a stable
descending ordering. This is a bi-encoder similarity reranker, not a
classifier-head cross-encoder.

## Workspace integration

When a supported Nomic encoder is registered alongside the active
tool-capable generation model, a Workspace session gets an optional semantic
retriever. On first use it:

1. walks only the selected canonical workspace without following symlinks;
2. skips `.git`, `.camelid`, build outputs, dependencies, virtual environments,
   GGUF files, and non-UTF-8 data;
3. discovers a bounded candidate set, prioritizes primary and nested source
   trees, then selects chunks breadth-first across files so large docs/QA trees
   cannot crowd implementation code out of the index;
4. caps indexed file count, per-file and total bytes, chunk count, chunks per
   file, and rendered excerpts;
5. embeds `search_document: <relative path>\n<chunk>` into an in-memory index;
6. injects the top query matches as explicitly untrusted memory.

The index is session-local and never written into the project. Any encoder
failure produces a notice and falls back to the existing lexical and agent-tool
retrieval path.

## Knowledge Library integration

Document search (`POST /api/documents/search`) ranks by meaning as well as by
keyword when the exact encoder above is in the models directory. Nothing needs
to be registered: the library loads its own copy of
`nomic-embed-text-v1.5.Q8_0.gguf` on first use, only after the file matches the
pinned size and SHA-256, and keeps it independent of whichever chat model is
loaded or switched. A missing file is looked for again on every use, so a later
download is picked up; a file that fails verification is not used.

**Indexing.** After each upload the new chunks are embedded in the
background, newest first, 16 per encoder call, with the `search_document:`
prefix; an upload that arrives while a large file is being embedded goes next
rather than waiting for the rest of that file. Keyword search works
immediately and does not wait for any of it. Vectors live
in the library database (`document_chunk_vectors`), one row per chunk, together
with the hash of the chunk text that was embedded, the encoder's SHA-256 and the
dimension count:

- a vector is only scored while its chunk still carries that hash, and is
  dropped with its chunk when the document is deleted or re-ingested;
- a chunk whose stored text no longer matches its recorded hash is skipped
  rather than embedded, and becomes pending again once the text is restored:
  the next search by meaning that covers it, or the next status request,
  finds it;
- chunks ingested before verifiable citations have no hash, are not embedded,
  and stay keyword-only until the document is attached again.

The indexer also starts when a search that ranks by meaning, or
`GET /api/documents/index-status`, finds pending chunks, so an existing
library is indexed on first use. That
endpoint reports whether the encoder is available (with a reason code when it
is not), whether indexing is running, the error that stopped the last run if
one did, and each document's indexable, indexed and skipped chunk counts. A
stopped run is not retried by searches or status requests, so a persistent
failure cannot turn polling into a retry loop; the next upload or a restart
tries again. The chat composer shows an attached document's progress as
indexing, waiting to index, or indexing stopped with that error.

**Ranking.** `mode` selects the rankers:

| `mode` | Behaviour |
| --- | --- |
| `auto` (default) | `hybrid` when the encoder is available, `keyword` otherwise |
| `keyword` | BM25 over the FTS5 index only |
| `semantic` | cosine similarity only |
| `hybrid` | both, fused by reciprocal rank |

`semantic` and `hybrid` return `409` with the reason code instead of silently
degrading. In `hybrid`, BM25 and cosine each contribute their best 50 chunks
in scope; each list adds `1 / (60 + rank)` for every chunk it holds, and ties
break on the better single-list rank, then chunk id. Citation checks run on
the ranked list in order until `top_k` results pass, so a withheld chunk is
replaced by the next verified one. The response's `retrieval` object reports
what actually ranked the results and the encoder's availability, plus the
scope's index coverage when meaning took part, and every result's `retrieval`
field says whether keyword, meaning or both found it. A scope with nothing
indexed yet reports `keyword`.

**Scope.** Without `doc_ids` or `collection_ids` a search covers the whole
library. Either narrows it: the search covers the named documents and every
member of the named collections, each document once. An empty scope returns no
results with `mode` `none`; an unknown collection is a `404`
(`collection_not_found`); a scope of more than 30,000 documents is a `422`
(`search_scope_too_large`), because each scoped document is bound as one SQLite
variable. Collections are managed with the routes below; members are kept in
`document_collection_members`, whose rows go with their document or their
collection when either is deleted and survive re-ingesting a document.

| Route | Effect |
| --- | --- |
| `GET /api/collections` | every collection by name, with member ids in the order added |
| `POST /api/collections` `{name}` | `201`; names are trimmed, 1–80 characters without control characters, and unique ignoring ASCII case (`409 collection_name_taken`) |
| `PATCH /api/collections/:id` `{name}` | rename, with the same rules |
| `DELETE /api/collections/:id` | `204`; the documents stay in the library |
| `POST /api/collections/:id/documents` `{doc_ids}` | adds them all, or none if any id is unknown (`404 document_not_found`) |
| `DELETE /api/collections/:id/documents/:doc_id` | `204`; the document stays in the library |

`POST /api/documents/ingest` also accepts `collection_ids`: every one must
exist before anything is stored, and the new document joins them all.

**Whole library.** `library: true` searches every document and keeps a passage
from outside `doc_ids` and `collection_ids` only when its cosine similarity to
the query reaches a floor that rises with the number of chunks the library has
indexed: 0.6408 + 0.0058 × ln(chunks), clamped to 0.3–0.9. That is 0.659 at 25
chunks, 0.681 at 1,000 and 0.689 at 4,406. The named documents and collections
are still searched in full and skip the floor, so an attached document is never
crowded out by the rest of the library. Keyword matches are held to the floor
too, which means a passage without a current vector from the pinned encoder
counts only when its document is named. The floor is a similarity, so this
needs the encoder: without it the request is a `409` with the encoder's reason
code and `param` `library`, and `mode` `keyword` is a `422`
(`library_search_needs_meaning`). Each result carries its `similarity`, and
`retrieval.relevance_floor` reports the floor applied. When nothing clears it,
the search returns no results. A search with neither list and no `library`
flag still covers the whole library with no floor, as before.

**How the floor was chosen.** By two rules, each written down and its script
hashed before it was run (`qa/evidence-bundles/f2a-library-search-20260930/calibration/`).
The data: two libraries of 1,000 documents, BEIR SciFact (science claims) and
BEIR FiQA-2018 (financial questions). Each library's own test questions (300
and 200) should find a passage; the other library's questions and 405
unrelated chat messages (MT-Bench prompts, seeded samples of Alpaca
instructions and GSM8K problems, and 25 short chit-chat lines) should not. Half
of every group chose, by the highest true-positive plus true-negative rate;
the other half was held out.

The first rule picked a single floor, 0.69. On the held-out half it served
85.2% of questions and kept 88.4% of unrelated messages quiet, but a live check
on a three-document library found it too strict there: the more chunks a
library holds, the closer its best chance match to an unrelated message, so a
floor fitted on 1,000 documents is higher than a small library needs. The
second rule simulated libraries of 3 to 1,000 documents from the same data
(each question's relevant documents kept), chose the best floor at each size,
and fitted it against ln(chunks). On the held-out half, over all sizes, it
serves 91.7% of questions (a relevant passage above the floor) and keeps 95.1%
of unrelated messages quiet, against 87.6% and 97.1% for 0.69 on the same
samples; at three documents 94.0% and 98.9%, at 1,000 documents 89.2% and
87.2%. Some messages that should find nothing still bring passages, which
reach the model as ordinary cited excerpts, and a natural question whose
answer shares a chunk with other topics can still fall below the floor. The
floor belongs to this encoder; another encoder would need its own.

There is no separate rerank stage: `/v1/rerank` is the same bi-encoder cosine
over the same encoder, so it would rescore candidates with the function the
semantic ranker already used.

Set `CAMELID_DOCUMENT_SEMANTIC=0` (or `false`, `off`, `no`) to keep document
search keyword-only; the encoder is then never loaded.

## Evidence gate

The ignored real-artifact test requires the SHA-pinned GGUF at
`target/embedding-fixtures/nomic-embed-text-v1.5.Q8_0.gguf`:

```text
cargo test --test embedding_real_model -- --ignored --nocapture
```

The gate checks exact tokenizer IDs, shapes/types, deterministic finite
unit-normalized output, semantic ordering, 256-dimensional Matryoshka output,
and three full-vector comparisons against llama.cpp b10173. The vector bar is
cosine greater than `0.9997` and maximum absolute element delta below `0.003`.

On the evidence Windows host, a controlled release sweep over four short
vectors measured median batch time of 304 ms at four workers and 265 ms at
eight workers (13.16 versus 15.09 embeddings/s, a 14.7% throughput increase).
Observed peak working set was effectively flat at 162.4 versus 162.8 MiB; the
metadata/tokenizer-only process baseline was 9.2 MiB. These are host-specific
measurements, not a portable SLA.

No support is implied for another filename, hash, Nomic version, encoder
architecture, quantization, classifier head, GPU backend, or an external vector
database; the Knowledge Library's vector table above is the only persistent
store. The experimental BitNet rows above likewise remain outside the
supported envelope until their own reference-vector receipts are committed.
