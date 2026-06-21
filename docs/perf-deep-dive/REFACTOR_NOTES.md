# Refactor: separate macOS (Metal + NEON) from the shared inference path

Goal: make `src/inference.rs`'s CPU matmul/decode hot path **free of macOS-only code** so surgical x86/CUDA perf work (e.g. the P1 tiled-GEMM, decode tuning) can't accidentally touch a Metal/NEON branch that only a Mac can compile. This is a **source-structure** change: it has **zero effect on the Windows/Linux binary** (the moved code was already `#[cfg(target_os="macos")]`-excluded there) and **zero effect on any runtime behaviour on any target**.

## What moved

| new module | gate | contents | lines |
|---|---|---|---|
| `src/inference/metal_seam.rs` | per-fn `cfg(target_os="macos")` + non-macOS no-op | per-matmul Metal offload (encoded row/rows, retained-block hybrid transpose, FFN gate/up hybrid) + the 3 inference-session lifecycle calls | 258 |
| `src/inference/cpu_neon.rs` | module `cfg(target_arch="aarch64")` | 14 Apple-Silicon NEON/dotprod leaf kernels (`q8_0_dot_rows_neon_dotprod`, `q8_0_i8_block_dotprod`, `q8_0_packed_4x{4,8}_block_dotprod`, `horizontal_sum_i32x4`, `aarch64_dotprod_enabled`, …) | 511 |

`src/inference.rs`: **19894 → 19351** (−543). The matmul dispatch functions (`q8_0_dot_rows`, `q8_0_two_dot_rows`, `q8_0_packed_rows4_dot`, …) keep their `cfg(all(target_os="macos", target_arch="aarch64"))` branches — they now just call `cpu_neon::…`. The per-matmul Metal branches now call `metal_seam::…`.

## Why `cpu_neon` is gated on `target_arch="aarch64"`, not `macos`

So it **compiles on `aarch64-unknown-linux-gnu`** and can be CI-verified there. Production behaviour is unchanged because the **dispatch sites stay macOS-gated** — the NEON kernels are only ever *called* on macOS. Off macOS the kernels are unused (`#[cfg_attr(not(target_os="macos"), allow(dead_code, unused_variables))]`). Bodies are byte-for-byte the originals (reduction/accumulation order unchanged → bit-identical).

## Verification

| surface | command | result |
|---|---|---|
| Windows x86_64 (the shared/x86/CUDA path) | `cargo check --bin camelid` | clean |
| aarch64 NEON kernels actually compile | `cargo check --target aarch64-unknown-linux-gnu --bin camelid` | clean (compiles `cpu_neon.rs`) |
| no x86/scalar/CUDA logic touched | `git diff` grep for `avx2\|x86_64\|sse` deletions | **empty** ⇒ Windows binary unchanged |
| runtime decode | greedy CPU decode, byte-for-byte vs pre-refactor | **PASS** (`PERF_RECEIPTS/refactor-parity-stage1-*.json`, + capstone) |
| EOL | `git diff --stat` vs `--ignore-cr-at-eol` | identical (no LF/CRLF churn) |

## Residual / for the Mac CI

- The `metal_seam::` and `cpu_neon::` **dispatch call sites** sit inside `cfg(target_os="macos")` blocks, so they compile **only on macOS** — not checkable on this Windows host. Risk is low: the seam/kernel bodies are verbatim, signatures are Windows-verified (metal_seam) / aarch64-linux-verified (cpu_neon), and every helper they call is *defined in* `inference.rs` (resolved via `use super::*`). A macOS CI/build is the final confirmation.
- **Pre-existing CI blocker (not from this refactor):** `clippy -D warnings` fails on `src/cuda_resident.rs:1556 launch_attention` (`too_many_arguments`, 13/7). It predates these changes (that file is untouched here). My changed files have **zero** clippy warnings.

## Still macOS-entangled in inference.rs (next, optional)

The **resident-decode engine** (`metal::ResidentDecodeState`, `ResidentLayerWeights`, `LogitsStage`, …, ~19 refs) — the Metal GPU decode path, the macOS analog of `cuda_resident`. It's a separate *feature* path, not the CPU matmul hot loop, so it was left for a follow-up (bigger structural move: it's a field on the session struct).
