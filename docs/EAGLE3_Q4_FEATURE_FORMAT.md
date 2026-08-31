# Exact-Q4 EAGLE-3 feature format v1

Schema id: `camelid-eagle3-q4-features-v1`

This is the offline handoff between Camelid's target-authoritative Q4 Metal forward and an
EAGLE-3 trainer. The exporter never substitutes a dense teacher: activations, hidden states and
greedy ids all come from the same loaded GGUF and resident inference kernels used by Camelid.

The JSONL input bridge uses canonical token-position masks:

```json
{"id":"example","input_ids":[...],"loss_mask":[...]}
```

Raw `loss_mask[Q]` is `1` iff corpus token `input_ids[Q]` belongs to the trainable assistant
span. The exporter aligns it to the prediction target: output base
`loss_mask[P] = raw_loss_mask[P+2]`, because the finalized EAGLE base row predicts token `P+2`.
The final two output mask rows are zero sentinels; the final raw corpus token may be trainable.

## Dataset layout

```text
dataset/
  manifest.json
  draft_to_target.u32le
  samples/
    00000000/
      meta.json
      input_ids.u32le
      labels.u32le
      target_argmax.u32le
      loss_mask.u8
      aux_layer_inputs.bf16le
      hidden_state.bf16le
      input_embedding.bf16le
      next_token_embedding.bf16le
      teacher_draft_logits.bf16le
      teacher_logsumexp.f32le
```

All arrays are C-contiguous, row-major and little-endian. Each `meta.json` records the dtype,
shape and SHA-256 of every payload. `manifest.json` records the target GGUF SHA-256,
quantization, fixed target geometry and an ordered `{id,path,length}` sample index. The exporter
writes each complete record independently and atomically republishes the manifest afterward, so
a trainer can consume completed records without retaining the corpus in memory. Both the dataset
and sample metadata pin `positional_contract` to
`eagle3-aux-p-next-token-teacher-p1-runtime-mask-p2-v1`. The manifest also pins the SW512 checkpoint SHA-256,
`draft_mapping_sha256`, `draft_vocab_size: 32000`, and the hashed mapping payload.

| File | Dtype | Shape | Meaning |
| --- | --- | --- | --- |
| `input_ids.u32le` | uint32 | `[T]` | Teacher-forced target input token at row `P`. |
| `labels.u32le` | uint32 | `[T]` | Corpus token at `P+2`; final two rows are `UINT32_MAX`. |
| `target_argmax.u32le` | uint32 | `[T]` | Exact Q4 target greedy prediction from capture row `P+1`; final row is `UINT32_MAX`. |
| `loss_mask.u8` | uint8 | `[T]` | `1` means the token predicted by EAGLE base row `P` is trainable. Final two rows are always `0`. |
| `aux_layer_inputs.bf16le` | bfloat16 | `[T,9216]` | Pre-layer residuals concatenated within each row in layer order `[2,14,25]`; each tap is width 3072. |
| `hidden_state.bf16le` | bfloat16 | `[T,3072]` | Target residual after final RMSNorm and before output projection. Masked bootstrap rows may be zero. |
| `input_embedding.bf16le` | bfloat16 | `[T,3072]` | Exact loaded-Q4 target embedding for `input_ids[P]`; lets a streaming trainer run without loading the target model. |
| `next_token_embedding.bf16le` | bfloat16 | `[T,3072]` | Target embedding for `input_ids[P+1]`, matching runtime pairing `(token[P+1], aux[P])`; invalid final row is zero. |
| `teacher_draft_logits.bf16le` | bfloat16 | `[T,32000]` | Exact-Q4 target logits from capture row `P+1`, gathered in the pinned SW512 `draft_to_target` order. Final row is positive zero. |
| `teacher_logsumexp.f32le` | float32 | `[T]` | Stable log-sum-exp over all 128,256 exact-Q4 target logits from capture row `P+1`; final row is a NaN sentinel. |

`draft_to_target.u32le` is a root-level uint32 `[32000]` payload decoded from the checkpoint's
strictly validated `d2t` tensor and cross-checked against `t2d`. It is never inferred from the
training corpus. Each target forward produces the full target logits once; the exporter gathers
these fixed rows on the host and performs no second target pass.

BF16 conversion is IEEE-754 round-to-nearest-even. The Metal target computes in its production
activation precision and the exporter converts only the completed host capture.

## Alignment invariant

At target row `P`, the one-layer draft head consumes:

```text
target feature = concat(layer_input[2,P], layer_input[14,P], layer_input[25,P])
token feature  = target_embedding(input_ids[P+1])
teacher        = target_distribution(capture_row=P+1)  // predicts token P+2
```

The exporter creates an empty resident session and teacher-forces from position zero, so there
are no missing bootstrap activations. The one-row teacher shift is applied only after all capture
rows have been validated; the final base row is the sole invalid teacher sentinel. The gathered
32K logits are sufficient for SpecForge's restricted-and-renormalized KL objective. Full-vocab
log-sum-exp is included for acceptance/LK metrics, but no full-vocab logits are retained.

## Derived-head admission

Camelid continues to admit the original pinned EAGLE artifacts without any opt-in. A trained or
no-update reserialized head has a different `model.safetensors` hash and is admitted only when
`CAMELID_EAGLE3_ALLOW_DERIVED=1` is set exactly and a sibling `training-receipt.json` validates.
The receipt schema is `camelid-eagle3-mlx-training-receipt-v1` and must contain integer
`tensor_count: 15` plus lowercase SHA-256 strings for `output_weights_sha256`,
`output_config_sha256`, `output_mapping_sha256`, `source_weights_sha256`,
`source_config_sha256`, and `source_mapping_sha256`. Additional trainer audit fields are accepted
for evidence and ignored by the runtime parser.

The output hashes must match the files and validated d2t/t2d mapping in the derived artifact. The
mapping digest is the trainer's exact encoded contract:
`SHA256(d2t_dtype_ascii || raw_d2t_payload || raw_t2d_payload)`. The source weights must be one of
Camelid's pinned checkpoints, while config and mapping hashes must remain identical across source
and output. Config parsing and the complete 15-tensor SafeTensors layout are still validated
normally. Benchmark receipts record the explicit opt-in and validated provenance under
`effective_env`.
