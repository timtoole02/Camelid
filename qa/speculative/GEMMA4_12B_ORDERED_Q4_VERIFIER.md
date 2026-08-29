# Gemma 4 12B ordered-Q4 target verifier

Status: additive experimental integration; not selected by the ordinary Gemma 4
generation path.

This checkpoint connects the exact dense 12B target geometry to a lossless
K=1/2/4/8 verifier API. It deliberately does not admit K=16.

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

This creates a self-consistent verifier arithmetic universe. Promotion still
requires the campaign's whole-model K=1 and K-wide token-array gates against the
pinned plain target; this checkpoint does not claim those receipts by itself.

## Transaction and rollback

`Gemma4GpuRuntime::verify_consecutive_greedy` returns a ticket, K greedy ids,
and K final hidden rows. Candidate KV slots are physically written but the
runtime logical cursor does not advance. The caller must resolve the ticket with
`commit_verifier_prefix(ticket, accepted_rows)`, where `accepted_rows` may be
zero through K. Rejected slots remain outside every later attention
`position_count` and are overwritten by the next target call. Wrong tickets,
double-pending batches, non-consecutive starts, and unsupported widths fail
closed without changing logical length.

`forward_greedy_ordered_q4` is the authoritative K=1 step for building a prefix
in this same universe. It is intentionally separate from the established
full-logit `forward` path.

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
borrow, and the assistant receives logical prefix length separately from the
physical cache capacity.

## Memory and safety

Verifier scratch is allocated lazily. The largest allocation is one reusable
58,982,400-byte private Q4 term slab shared by every projection. The SPEC50 head
scratch is also lazy (about 8 MiB for K=8 logits); no decoder or tied-table weight
is duplicated. Width admission remains K<=8, and the Mini2 campaign must keep
contexts short until the whole-target safety/thermal matrix is complete.

