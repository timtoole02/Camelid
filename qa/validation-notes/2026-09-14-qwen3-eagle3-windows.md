# Windows Qwen3-4B CUDA target-capture validation

Tested the Windows follow-up diff on #767 head
`7642db46e81a1465e960a5f65a3af31b0a137822`, September 14, 2026.
The PR targets `codex/qwen4b-eagle-chat`, which depends on #766.

**Result: CUDA target-capture foundation validated; learned EAGLE port incomplete.**
No EAGLE head was loaded or executed, and no acceleration is claimed.

## Host and target

- Windows x86_64 MSVC, Rust 1.95.0.
- NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB VRAM; driver 576.83, CUDA 12.9.
- Qwen3-4B Q4_K_M target SHA-256:
  `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
  This is #767's exact pinned Qwen target.
- Default Windows features (CUDA enabled by the existing build configuration).
- Successful dev/test builds used `CARGO_PROFILE_DEV_DEBUG=0`,
  `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`, and tests used
  `RUST_MIN_STACK=8388608`. A first build exhausted disk while writing its PDB;
  only this worktree's generated Cargo artifacts were cleaned before rebuilding.
  Disabling symbols does not enable `cfg(test)` in the production library.

## Passed checks

- `cargo build --bin camelid`, executable `--version`, and `serve --help`.
- Production executable integration test: EAGLE mode exits with an explicit
  unsupported-backend error before model loading/listener binding.
- `cargo test --lib eagle3`: 86 passed, 1 ignored.
- Library executable `speculative` filter: 23 passed, 1 ignored, including the
  original weak/strong suffix-confidence regression without weakening it.
- Library executable `tree` filter: 58 passed, 4 ignored. This broad substring
  filter includes other tree-related tests; counts across filters overlap.
- Library executable `stream_step_deltas` filter: 3 passed.
- Explicit CUDA `verify_batch_matches_sequential`: 1 passed, no skip. Verifies
  capture-on/off greedy predictions, reversed caller layer order, raw layer-zero
  embeddings, multi-row/single-row capture agreement (relative/absolute tolerance
  1e-5), and invalid layer/capacity refusals.
- Explicit live `qwen3_cuda_target_capture_acceptance_and_rollback`: 1 passed,
  no skip, 24.39 seconds including cold engine upload. Every reference and
  continuation forward required the resident GPU. All 36 layers were resident;
  cache capacity was 512 positions.

The live reference generated token IDs
`[12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576]`.
The capture call returned inputs to layers `[33, 2, 18]`, each `[2, 2560]`,
with finite, nonzero residuals. One correct draft plus bonus matched the reference.
A wrong draft committed only the bonus; all five subsequent tokens matched.
Four-row verification was refused by the existing two-row K-quant shared-memory
cap without changing the logical KV position. The synthetic test separately
executes a four-row Q8 fixture; it does not widen the Qwen K-quant support claim.

## Production HTTP behavior

Started the ordinary `camelid serve` executable on a loopback address with the
pinned model, GPU enabled, speculation unset, and resident context capped at 512.
Health reached `generation_ready=true`; its selected backend was
`cuda_resident_kquant_runtime`. Runtime logs confirmed all 36 layers on CUDA.
An eight-token greedy completion for `The capital of France is` returned
` Paris. The capital of Germany is Berlin`. A streaming request returned the
same text and terminated with `[DONE]`. The test server was stopped afterward.

## Limits and upstream CI

The learned draft transformer, prompt capture, tree capture/commit, and EAGLE
cache orchestration are still Metal-specific. Windows EAGLE serving now refuses
startup explicitly rather than advertising a mode that fails during generation.
No complete EAGLE parity, throughput comparison, full all-target suite, release
build, or packaged desktop run was performed.

At the inspected upstream head, public-scrub passed; Rust, frontend, and
validation-script jobs failed. Besides the fixed production validator error,
local `cargo clippy --lib --bin camelid --test eagle3_platform --test
eagle3_cuda_target -- -D warnings` still fails on 15 inherited findings: literal
casts in `eagle3.rs` and range-loop lints in `metal.rs`. Those unrelated cleanup
changes are outside this draft. Formatting, diff whitespace, and public-scrub
checks passed.
See [port scope and reproduction](../../docs/qwen3-eagle3-windows.md).
