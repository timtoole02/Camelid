# Qwen3-4B EAGLE-3 on Windows: incomplete port

This change is a draft follow-up to #767 (`codex/qwen4b-eagle-chat`), which
depends on #766. Its PR base is #767 so the review shows only the Windows work.

**Learned EAGLE-3 generation is not implemented on CUDA.** The normal Windows
server supports ordinary CUDA generation, but `CAMELID_SPEC_DECODE=eagle3` now
exits before binding a listener or loading models, with an explicit explanation
and instructions to unset that option. A successful Windows build must not be
interpreted as support for Metal's learned speculation path.

## Implemented foundation

- The non-macOS Metal constructor calls the production geometry-aware validator,
  fixing the `cfg(test)` mismatch in the ordinary executable build.
- Generic suffix drafting uses the confidence-qualified chain, as required by
  the existing weak/strong continuation regression.
- CUDA linear verification can snapshot selected **pre-layer residuals** without
  changing projection or attention arithmetic. The opt-in snapshots are copied
  on the CUDA stream and returned in caller order as `[row, hidden]`. The ordinary
  verifier requests no captures. Layer IDs and KV capacity are checked before
  GPU work begins.
- `verify_drafts_cuda_with_layer_inputs` exposes this target seam to inference.
  It returns every verification row while committing only the accepted prefix
  plus bonus. Qwen3-4B's head needs inputs to layers 2, 18, and 33, each 2560 wide.
  Capture calls decline pipeline shards because their local layer indices cannot
  represent absolute target taps.

The existing CUDA K-quant stack limits this pinned target to **two verification
rows (one draft plus bonus)** under its shared-memory budget. Four-row verification
is explicitly refused before KV advances. This port does not raise that kernel
limit or claim support for Metal's wider EAGLE rounds.

## Still required before enabling Windows EAGLE

1. Port the checkpoint's learned encoder, one-layer recurrent draft transformer,
   draft-to-target vocabulary map, and output head to CUDA, including its Qwen
   geometry, normalization, RoPE, and cache semantics.
2. Add prompt capture and target tree-verification capture/commit on CUDA, or
   implement and qualify a separate linear serving scheduler that observes the
   target's two-row kernel cap. The current
   scheduler and pooled draft head still call Metal-specific methods.
3. Validate prompt seeding, rejected-branch rollback, authoritative catch-up,
   repeated requests, streaming, and context boundaries with the pinned head.
4. Collect complete learned-head token-parity and throughput evidence. Target-only
   capture parity establishes neither learned-head parity nor an acceleration.

## Reproduce the bounded checks

On a Windows CUDA host with the pinned Qwen3-4B Q4_K_M target:

```powershell
cargo build --bin camelid
cargo test --test eagle3_platform
cargo test --lib eagle3 -- --test-threads=2
cargo test --lib speculative -- --test-threads=2
cargo test --lib cuda_resident::tests::verify_batch_matches_sequential -- --ignored --exact --nocapture
$env:CAMELID_QWEN3_4B_GGUF = '<model-directory>\Qwen3-4B-Q4_K_M.gguf'
cargo test --test eagle3_cuda_target -- --ignored --nocapture
```

The two explicit CUDA tests fail if the required device/model is unavailable;
they do not treat CPU fallback as CUDA evidence. The synthetic test compares
capture-on and capture-off predictions with sequential CUDA decoding, checks
raw embeddings at layer zero, and compares multi-row captures with single-row
captures. The live test checks all three Qwen taps, a four-row refusal without
KV advancement, one-draft acceptance plus bonus, mismatch rollback, and
subsequent greedy continuation. Every reference/continuation forward must use
the resident GPU, with no general CPU diagnostic fallback. It never loads an EAGLE
head and makes no throughput claim.

The executable integration test runs on non-macOS platforms and launches the
production server binary, so library unit-test compilation cannot mask the
original conditional-compilation defect.
