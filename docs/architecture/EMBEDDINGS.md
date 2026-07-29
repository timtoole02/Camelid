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
architecture, quantization, classifier head, GPU backend, or persistent vector
database.
