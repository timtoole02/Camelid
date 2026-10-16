# Bonsai 2 CUDA projection optimization

This bundle records the follow-up to the initial [CUDA qualification](../cuda/README.md).
The optimized kernel keeps the 28-byte PTQ blocks, float activations, reduction
order, model placement, and target verification semantics. It specializes token
tiles and shares input loads across output rows. Dense F32/BF16 gate projections
also use token-specific kernels. The default is optimized;
`CAMELID_BONSAI2_CUDA_PROJECTION=baseline` selects the previous projection kernel
for a controlled comparison in the same executable.

## Profiling

Nsight Systems captured the counting request, its eight-token warmup, and both
prompt prefills for 2B, 4B, and 9B. Each before/after pair uses the final executable
with CUDA event tracing disabled. These traces diagnose costs; the benchmark
harness runs without a profiler. CUDA graph replay was already active.

| Model | Projection share before | Projection share after | Vocabulary projection before | After |
| --- | ---: | ---: | ---: | ---: |
| 2B | 88.6% | 75.2% | 6.45 ms | 2.01 ms |
| 4B | 90.6% | 79.5% | 8.16 ms | 2.63 ms |
| 9B | 92.0% | 83.1% | 12.88 ms | 4.18 ms |

`final-*-kernels.csv` contains the six final kernel summaries. `profile.json`
records their binary identities, trace digests, and per-grid measurements.
The earlier exploratory V1 summaries remain in `profile-before-kernels.csv`
and `profile-after-kernels.csv`; unlike the final pairs, their CUDA event tracing
settings differ. In that initial trace, large synchronous-readback host time
was GPU waiting, while actual device-to-host copying took only microseconds.
Full traces and isolated experiment executables remain in ignored
`output/bonsai2-profile`. Profiler timings are not the headline speedup metric.

The pinned [Prism CUDA contraction](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b10777-e3de715/ggml/src/ggml-cuda/vecdotq.cuh)
also uses Q8 activations, integer dot products, and a transposed activation
layout. Camelid retains its qualified float target arithmetic. Our independent
integer-drafter prototypes did not beat the optimized float kernel on the 2B
proposal-model shapes and are excluded from the runtime. A 36-byte repacked
weight experiment was faster in isolation but increased packed-weight storage
by 29%; the selected implementation preserves the model's small footprint.

The numerical fixture compares every output float bit against the old kernel
across representative projection widths, 1/2/3/4/5/8/9 tokens, partial row tiles, the vocabulary
dispatch threshold, finite half-scale extremes, residual addition, and dense
F32/BF16 projections. Existing independent CPU-reference, real-model state,
rollback, seeded sampling, stop, and cancellation tests also run.

## Reproduction

Use the host, model files, reference package, and timing protocol from the
[initial qualification](../cuda/README.md). This run uses the same RTX 3060
Laptop 6 GiB, i7-11800H, 16 GB RAM, CUDA 12.9, and NVIDIA driver 576.83.
Run one GPU workload at a time and keep compilation separate from timings.

Build the probe, server, and library tests with `--release --features cuda
--jobs 2 --config profile.release.lto=false --config profile.release.codegen-units=16`.
`manifest.json` records the actual source commit, retained binaries, and hashes.
The test-only build exhausted host memory at Rust opt-level=3; its successful
retry used `--jobs 1 --config profile.release.package.camelid.opt-level=1`.
Probe/server binaries retain opt-level=3. CUDA source and NVRTC settings are
identical in the production and test builds.
Run `cuda-cache-qualification.mjs` to collect the 15 host/device checks.
Run the production frontend's text/image `gui-smoke.mjs` for all four models
with `--send --default-image --require-supported --require-cuda --require-request-rate`.
This exercises the native library-server harness, not an installed Tauri package.
Apple Silicon remains a separate qualification.

Run `cuda-reference-benchmark.mjs` and `spec-benchmark.mjs` with three repetitions,
32 output tokens, the three checked-in prompts, a three-token 2B drafter, and
2048 context slots. Collect the native suite once with projection policy
`baseline`, then again with the variable unset. Reference runs use the pinned
port with fresh measurements. Each mode has an eight-token warmup. Load and
warmup are excluded; uncached prefill and generation are included. Native timing
starts after tokenization, while the reference includes localhost HTTP and
tokenization overhead. Callback timestamps are not a GPU decode clock.

The primary rate is output tokens divided by complete request duration, using
the median of three runs. The JSON preserves all repetitions and IDs. The 27B
configuration streams packed weights on Camelid and uses CPU/GPU layer splitting
in the port, so it is not a comparison of entirely GPU-resident 27B models.

Generate `comparison.json` with `compare-cuda-benchmarks.mjs qa/bonsai2/cuda-optimized`.
Then run `compare-cuda-optimization.mjs` to generate `optimization.json` and the
[complete twelve-case results](RESULTS.md). Record the build and artifact hashes
with `record-cuda-optimized-evidence.mjs COMMIT PROBE TEST_BINARY SERVER SERVER_BUILD_ID`.
Check the completed bundle with `node qa/bonsai2/check-cuda-optimized-evidence.mjs`.
Artifact hashes cover exact file bytes; `.gitattributes` preserves LF endings
for the JSON, CSV, and Markdown receipts on every platform.
The original bundle remains unchanged as historical evidence.
