# Gemma 4 12B ordered-Q4 target verifier

Status: additive experimental integration; not selected by the ordinary Gemma 4
generation path.

This checkpoint connects the exact dense 12B target geometry to a transactional
K=1/2/4/8 verifier API. It deliberately does not admit K=16. The rollback and
same-universe mechanics are exact; whole-token parity with the pinned plain
target remains a mandatory promotion gate, so this checkpoint does not yet
claim end-to-end losslessness.

## Admitted row

- 48 decoder layers, hidden width 3,840, FFN width 15,360, 16 query heads.
- Forty sliding layers: 8 KV heads, head dimension 256, window 1,024, explicit V.
- Eight global layers: 1 KV head, head dimension 512, V-less (`V = raw K` before
  K norm and RoPE).
- Every decoder projection is Q4_0; the tied embedding/output table is Q6_K.
- No PLE and no target cross-layer KV sharing.

Every shape, norm width, Q4 buffer extent, attention schedule, cache ownership,
RoPE table, window start, and width is validated before a command buffer is
committed. A refusal returns to the caller without silently selecting a different
verifier.

## Arithmetic contract

`Gemma4ResidentModel::verify_consecutive_hidden_ordered_q4` evaluates a
layer-major K-wide causal graph:

1. the exact row-wise RMS reduction and Q8_0 quantizer;
2. one strict shared-weight Q4_0 projection for K columns;
3. the existing per-row QK/V norm and split-half RoPE kernels;
4. f32 KV scatter for every candidate, followed by the existing split-three f32
   attention for each row with its own immutable position/window scalars;
5. ordered Q4_0 output/gate/up/down projections, sandwich norms, GeGLU,
   residuals, and layer scale;
6. row-major final hidden readback only.

K=1 and K>1 use the same Q4-column kernel family. The Q4 gate proves bitwise
identity against the ordered K=1 comparator for K=1/2/4/8 at the real 3,840 and
15,360 contraction widths. The Q6_K `forward_argmax_spec50_batch` API likewise
uses the same head family for normal K=1 and verifier K>1.

SPEC50 deliberately defines the verifier head universe; it is not assumed to
be identical to the established full-logit head, whose projection and Rust
softcap ordering protect saturated-token tie breaks. Promotion therefore needs
a whole-token K=1 greedy-id gate against that pinned established target in
addition to the component-level Q4/Q6 exactness receipts. Its GPU argmax matches
the dense runtime's `Iterator::max_by(total_cmp)` convention: the highest vocab
id wins an exact post-softcap tie.

This creates a self-consistent verifier arithmetic universe. Promotion still
requires the campaign's whole-model K=1 and K-wide token-array gates against the
pinned plain target; this checkpoint does not claim those receipts by itself.

## Transaction and rollback

`Gemma4GpuRuntime::verify_consecutive_greedy` returns a ticket, K greedy ids,
and K final hidden rows. Candidate KV slots are physically written but the
runtime logical cursor does not advance. The caller must resolve the ticket with
`commit_verifier_prefix(ticket, consumed_input_rows)`. This argument counts
physically consumed target inputs, not accepted drafts or emitted outputs. For
`[anchor, d1, ...]`, an immediate mismatch commits 1, accepting `m` drafts
commits `1 + m`, and the bonus prediction is not another forwarded row. Zero-row
rollback is reserved for abort/retry of the whole target call. Rejected slots
remain outside every later attention `position_count` and are overwritten by
the next target call. Wrong tickets, double-pending batches, non-consecutive
starts, and unsupported widths fail closed without changing logical length.

`forward_greedy_ordered_q4` is the authoritative K=1 step for building a prefix
in this same universe. The prompt must be replayed from position zero through
this ordered K=1 API; an established-path prompt cache cannot be adopted.
`prefill_ordered_q4` performs that exact `encode(prompt, true, true)` replay,
commits every prompt input, and returns both the first output prediction and the
last prompt hidden row needed to seed MTP. `generate_greedy_ordered_q4` is the
K=1 qualification loop: it uses the complete model stop set, never emits or
forwards a stop prediction, and reports prefill and decode wall separately.
Its `decode_forward_count` excludes the first output already produced by the
last prefill row, so steady-state tok/s uses that count rather than total emitted
ids.
`qualify_ordered_q4_k1` compares its full greedy-id sequence with a fresh
established-lane sequence and fails at the first mismatch.

The ordered verifier and established full-logit path share the same physical KV
buffers, so the runtime assigns those buffers to one arithmetic lane for the
entire sequence. Once either lane claims position zero, calls into the other
lane fail closed, and both lanes require contiguous positions.
`reset_dense_verifier_sequence` releases that ownership only at a genuine
new-sequence boundary; retained cache bytes are then harmless
because the next lane starts at zero and overwrites every row before exposing
it. Rollback is therefore lossless only inside an all-ordered sequence. It is
not permission to fall back to the established projection arithmetic.

## MTP device seams

Two scoped hooks remove avoidable host staging:

- the selected Q6_K embedding row is borrowed from the existing file-backed
  table as `(BufferRef, byte_offset, 3150 bytes)`; offsets need only be f16
  aligned, so the token-row offset is not rounded or copied;
- selected target KV caches are borrowed as f32
  `[kv_head][max_positions][head_dim]` buffers with exact byte extents. Layers
  46 and 47 expose the 12B assistant's required 8x256 sliding and 1x512 global
  source geometries.

Both hooks are closure-scoped. Model-owned Metal references cannot escape their
borrow. The assistant composition remains a separate integration commit because
K-wide recurrence must pass committed target KV length independently from the
future proposal/RoPE position.

## Memory and safety

Verifier scratch is allocated lazily. The largest allocation is one reusable
58,982,400-byte private Q4 term slab shared by every projection. The SPEC50 head
scratch is also lazy (about 8 MiB for K=8 logits); no decoder or tied-table weight
is duplicated. Width admission remains K<=8, and the Mini2 campaign must keep
contexts short until the whole-target safety/thermal matrix is complete.
