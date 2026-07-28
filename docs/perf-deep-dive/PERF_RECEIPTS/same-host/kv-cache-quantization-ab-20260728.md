# KV cache quantization before/after report

Date: 2026-07-28

Repository: Camelid repository root

Baseline: clean `origin/main` at `f8649c6294aac0ab742890328320e80cc0434085`

After: the KV cache quantization implementation in this pull request

## Verdict

PASS.

- Exact cache allocation tests pass for F32, F16, Q8_0, and Q4_0, including
  dimensions that are not divisible by the 32-value quantization block.
- Measured process RSS follows the predicted cache sizes.
- The current F32 and F16 control paths show no material throughput regression.
- Q8_0 and Q4_0 have no material short-context decode regression, improve
  long-context CPU decode in the balanced final matrix, and are flat on the
  fully resident CUDA path.
- All measured deterministic generations produced the same token IDs in every
  tested cache mode.
- The complete all-target/all-feature test matrix passes.

The quantized formats remain lossy. Token identity on this benchmark is a
regression check, not a general model-quality certification.

## Test host

| Item | Value |
| --- | --- |
| OS | Windows 11 Home Insider Preview, 10.0.26220 |
| CPU | Intel Core i7-11800H, 8 cores / 16 logical processors |
| RAM | 15.74 GiB |
| GPU | NVIDIA RTX 3060 Laptop GPU, compute capability 8.6, 6 GiB VRAM |
| Model | `Llama-3.2-3B-Instruct-Q8_0.gguf`, 3.187 GiB |
| Threads | 8 |
| Sampling | Greedy, temperature 0 |
| Build | Cargo release profile: opt-level 3, fat LTO, one codegen unit, x86-64-v3 |

The clean baseline and current tree were built in separate source worktrees with
the same Cargo target directory and identical release flags. The baseline
working tree was not modified.

## Benchmark method

CPU measurements used `bench-generate --deterministic`, which disables the GPU
stack and isolates host KV-cache behavior.

- Short context: 5 prompt tokens, 65 generated tokens, 64 measured decode
  steps, warmup enabled, two mirrored-order rounds, four samples per format.
- Long context: 1,010 prompt tokens, 33 generated tokens, 32 measured decode
  steps, two mirrored-order samples per format.
- Memory: process peak RSS reported by `bench-generate`.
- The long prompt crossed Camelid's 2,048-position allocation boundary, so the
  expected memory deltas use a 2,048-position cache capacity.
- CUDA control: 5 prompt tokens, 65 generated tokens, warmup plus three measured
  iterations, all 28 layers resident.

The laptop showed meaningful power/thermal variation, so the report uses
balanced medians rather than individual fastest samples.

## Theoretical KV memory

The model has 28 layers, 8 KV heads, and a 128-value head dimension.

| Cache format | Bytes/token | Bytes at 2,048 positions | Reduction vs F32 |
| --- | ---: | ---: | ---: |
| F32 baseline | 229,376 | 448 MiB | — |
| F16 | 114,688 | 224 MiB | 50.0% |
| Q8_0 | 60,928 | 119 MiB | 73.4% |
| Q4_0 | 32,256 | 63 MiB | 85.9% |

Q8_0 uses 34 bytes per 32-value block and Q4_0 uses 18 bytes per block.
Per-row block rounding is included in allocation and budget calculations.

## Measured host memory

| Configuration | Median peak RSS | RSS saved vs F32 | Expected cache-only saving |
| --- | ---: | ---: | ---: |
| Before: F32 | 4.396 GiB | — | — |
| After: Q8_0 | 4.069 GiB | 335 MiB | 329 MiB |
| After: Q4_0 | 4.007 GiB | 399 MiB | 385 MiB |

The 6–14 MiB difference between process-RSS deltas and cache-only predictions
is normal process high-water variability. Dedicated allocator tests verify the
exact block counts and byte totals independently of RSS.

The existing opt-in F16 lane reduced the same 2,048-position cache by
approximately 218–225 MiB in the mirrored control run, matching the theoretical
224 MiB.

## CPU decode throughput

### Short context

| Configuration | Median tok/s | Change vs before |
| --- | ---: | ---: |
| Before: F32 | 9.029 | — |
| After: Q8_0 | 8.929 | -1.1% |
| After: Q4_0 | 8.909 | -1.3% |

The approximately 1% differences are inside the observed run-to-run noise.
Every one of the 12 measured outputs was token-identical.

The F16 before/after control was also stable: 6.521 versus 6.454 tok/s in its
balanced short-context run (-1.0%).

### Long context

| Configuration | Median tok/s | Change vs before | Median prefill | Prefill change |
| --- | ---: | ---: | ---: | ---: |
| Before: F32 | 7.416 | — | 33.185 s | — |
| After: Q8_0 | 8.288 | +11.8% | 34.442 s | +3.8% |
| After: Q4_0 | 8.211 | +10.7% | 33.497 s | +0.9% |

All long-context outputs were token-identical.

The first implementation used scalar quantized dot and value-dequantization
loops. That candidate measured 4.909 tok/s for Q8_0 and 4.269 tok/s for Q4_0,
or -25.1% and -34.9% versus its paired baseline. The final implementation
adds runtime-dispatched AVX2/FMA Q8_0 and Q4_0 attention kernels and fuses value
dequantization with accumulation. The table above is the post-fix result.

## CUDA resident control

| Configuration | Median tok/s | Change vs before |
| --- | ---: | ---: |
| Before: F32 host cache | 54.386 | — |
| After: Q8_0 host cache | 54.203 | -0.34% |
| After: Q4_0 host cache | 54.197 | -0.35% |

All 28 layers were resident in VRAM and every output was token-identical. The
differences are below 0.4% and are not a material regression.

Host KV quantization does not claim to reduce the resident engine's own VRAM
cache; this control verifies that selecting a host-cache format does not slow
the normal CUDA fast path.

## Correctness and regression validation

- `cargo build --release --bin camelid`
- `cargo test --all-targets --all-features -- --test-threads=1`
  - Library: 1,229 passed, 0 failed, 68 ignored.
  - 59 test-result sections across library, integration, binary, and example
    targets; no failure markers.
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo fmt --all -- --check`
- Focused Q8_0/Q4_0 roundtrip, dot, fused AXPY, tail-row, layout, budget, and
  attention-oracle tests.

The final source includes:

- row-padded quantized offsets and exact allocation accounting;
- tail-safe Q8_0/Q4_0 storage and attention;
- current ggml-compatible Q4_0 signed-max scaling;
- IEEE-correct F16 scale conversion;
- dtype-neutral resident-cache seeding;
- AVX2/FMA quantized attention fast paths with portable scalar fallbacks;
- strict CLI parsing and model-config propagation.

## Binary provenance

| Binary | SHA-256 |
| --- | --- |
| Clean baseline | `C0B31C017394B9BE812E08D597469041B8DA5BCF3C0F795F8BFA817444DB2252` |
| Final rebuilt implementation | `094D0E54073EBA6F2DA6413C1202C1E5F62FE2345BB1916154B2AE0E0F451EFE` |

Raw benchmark JSONL and full test logs are in the ignored local directory
`target\kv-bench-results`.
