# Gemma 4 12B QAT assistant Metal foundation

This branch adds an isolated assistant only. It does not route target inference,
change the Gemma target cache, or claim speculative-decode speed/correctness.

## Exact artifact gate

- Repository directory: `gemma-4-12B-it-qat-q4_0-unquantized-assistant`
- `model.safetensors`: 845,719,296 bytes
- SHA-256: `67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6`
- Safetensors header: 5,360 bytes
- BF16 payload: 845,713,928 bytes, 48 tensors, contiguous
- `config.json` SHA-256:
  `7638c1d42f9fa73fe44b1a10604766b928b8985263f15775cd1a286a5a12799c`

Pinned geometry is target hidden 3,840, assistant hidden 1,024, FFN 8,192,
four assistant layers, eight sliding KV heads, and one full KV head. The target's
48-layer 5:1 schedule derives the shared source pair as layers 46/47. The 26B
assistant's hidden 2,816, two full KV heads, and layers 28/29 are rejected.

## Resident representation

All 23 assistant matrices are packed from BF16 to canonical signed-scale GGML
Q4_0 in one 237,846,528-byte Metal shared buffer. Norms and four layer scalars
remain exact widened BF16/f32. After packing, the 846 MB source mapping is
released. The output head is the tied packed embedding.

The Q4 arithmetic and BF16 boundaries were forked from the production 26B
assistant at/after `95dcadd0`; no 26B pager or MoE code was imported.

## K=1 boundary

`Gemma4Mtp12AssistantMetal::propose_k1` is deliberately a parity/profiling
scaffold. It:

1. accepts CPU slices for the target scaled embedding, pending normalized
   hidden, and logical sliding/full KV;
2. validates exact 12B geometry and finite values;
3. runs the complete four-layer assistant, tied Q4 head, and deterministic GPU
   argmax in one Metal command buffer; and
4. captures full logits plus the 3,840-wide recurrent hidden for stage/oracle
   comparison.

It is not the serving interface. Target integration should retain this call as
an oracle while adding scoped no-copy target embedding/KV views and a token-only
readback. The target-authoritative verifier and rollback/acceptance logic remain
separate work.

## Device-fed K=1 boundary

`Gemma4Mtp12AssistantMetal::propose_k1_device` is the scoped no-copy companion
to the CPU oracle. It still returns full logits and recurrent hidden for parity,
but it does not upload CPU embedding or KV vectors.

- `Gemma4Mtp12Q6KEmbeddingRow` is one selected 3,150-byte Q6_K row for hidden
  width 3,840. The view may alias the existing full 825,753,600-byte no-copy
  target table. Its checked byte offset is added inside the shader, so a row at
  offset `2 mod 4` remains legal; no 3,150-byte staging copy is made. Metal
  dequantizes the row, multiplies by `sqrt(3840)`, and applies the same BF16 RNE
  boundary as `propose_k1` before pre-projection.
- Sliding KV must identify target layer 46 and provide native-f32 key/value
  views in `[8][max_positions][256]` order.
- Full KV must identify target layer 47 and provide native-f32 key/value views
  in `[1][max_positions][512]` order.
- Both KV capacities must match. `logical_position` is the proposed position
  and the materialized target-prefix length, admitted only in
  `1..max_positions`.
- Every buffer's Metal registry ID, view offset/alignment, exact byte length,
  backing-buffer bound, source layer, and geometry are validated before any
  assistant scratch is changed.

The pending normalized target hidden remains the only 3,840-float CPU input in
this K=1 seam. `propose_k1_device_at_position` preserves the same K=1 arithmetic
while separating the advancing query position from the fixed verified target
KV prefix; it is the composable fallback/oracle for a draft chain.

## Device-resident K<=8 chain

`Gemma4Mtp12AssistantMetal::propose_chain_device_resident` is an isolated,
token-only assistant seam. It is not wired to the CLI or target runtime.

- `draft_k` admits every value in `1..=8` and fails closed outside that range.
  In particular, draft counts 1/3/7 compose with target verifier widths 2/4/8
  because the target batch includes its anchor token.
- `Gemma4Mtp12Q6KEmbeddingTable` aliases the caller's one complete
  825,753,600-byte Q6_K table. It checks `[262144, 3840]` geometry, device,
  range/alignment, and the pinned target SHA-256
  `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
- The initial recurrent hidden is a checked 3,840-float Metal view. Every later
  token id and post-projection recurrent row stays on Metal and directly feeds
  the next step.
- `propose_chain_from_cpu_hidden` is the reachable bridge for the current target
  verifier: it validates and stages exactly 15,360 CPU bytes into
  assistant-owned Metal, then calls the same one-command chain. The ledger
  distinguishes this upload from the zero-upload device-hidden entry point.
- `Gemma4Q6KHead::with_full_table_device` holds the target head mutex while a
  callback consumes the exact mmap-backed table range. Its higher-ranked
  callback prevents the `BufferRef` from escaping; it verifies 12B
  hidden/vocab/byte geometry and the runtime-supplied pinned target SHA before
  constructing the assistant table view.
- One command buffer contains every draft step and is waited once. Production
  readback is only `K * sizeof(u32)` token bytes; test-only scratch inspection
  proves every recurrent bit.
- `target_kv_len` is the immutable verified layer-46/47 prefix. In contrast,
  `proposal_position` advances as `P, P+1, ...` for per-step RoPE and sliding
  bounds. Physical KV slots at and beyond the prefix are never read.
- The timing result records CPU preparation, encoding, wait, GPU busy, kernel,
  and wall time. The ledger records the one command/wait, borrowed table/KV
  spans, assistant matrix and logical KV reads, dynamic score scratch, resident
  chain state, and exact token-only readback bytes.

Assistant argmax deliberately preserves the official first-index/PyTorch tie
rule (lowest vocab id on an exact tie), which is tested on Metal. This differs
from Camelid's target-verifier greedy tie rule; assistant tokens affect proposal
efficiency, while the future target verifier remains output-authoritative.

No target runtime, verifier, CLI, acceptance, or rollback path is wired here.

## Tests

```sh
cargo check --lib
cargo test --lib metal::gemma4_mtp12::tests -- --nocapture
```

The focused suite pins the artifact geometry/SHA, derives layers 46/47 from the
official target schedule, validates the complete tensor table and 23-matrix Q4
layout, checks canonical signed-max Q4 packing, checks proportional full RoPE,
and compiles every production MTP12 Metal pipeline. A sparse synthetic full
assistant additionally compares CPU-fed and device-fed K=1 token, every logit
bit, recurrent-hidden bits, and the gathered/scaled embedding bits. Separate
synthetic view tests prove device, bounds, geometry, and logical-position
refusals fail closed. The resident-chain fixture proves every K=1 through K=8 token and
every recurrent bit against repeated advancing-position K=1 calls, then poisons
all unverified physical KV tail slots and proves K=8 tokens/recurrent rows do
not change. It also proves the 15 KB CPU-hidden bridge is bit-identical. A
focused GPU test pins assistant argmax's first-index tie rule, and a sparse-file
head test checks the exact 825,753,600-byte mutex-scoped table alias.
