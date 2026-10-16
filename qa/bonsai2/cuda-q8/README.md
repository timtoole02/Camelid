# Bonsai 2 transposed Q8 CUDA qualification

This bundle continues the [float projection optimization](../cuda-optimized/README.md).
PTQ weights retain their original 28-byte blocks. The default CUDA projection
quantizes rotated activations in groups of 32, with a half scale and an exact
signed integer sum. DP4A contracts four packed bytes per instruction. Transposed
activation planes coalesce the loads across lanes that own adjacent K blocks;
the exact sum removes the unsigned ternary digit bias after contraction.
The [PrismML CUDA reference](https://github.com/PrismML-Eng/llama.cpp/tree/prism-b10777-e3de715/ggml/src/ggml-cuda)
informed the layout and exact-sum technique; see the repository's third-party notice.

The Hadamard transform and quantization share one launch and shared-memory
buffer. Four-token projection tiles bound register use during prefill and
speculative verification. Scalar and batched projections use the same K
partition and reduction order. Dense F32/BF16 gates, attention, recurrent
updates, and model weights retain their previous arithmetic/storage.
Quantization scratch is preallocated per transform before CUDA graph capture.
The 27B capacity planner still streams packed weights while computing language
layers on CUDA.

Q8 changes activation arithmetic and may change logits or token choices from
the float path. This is separate from speculative correctness: drafted and
ordinary requests must agree within the selected target arithmetic policy.
The controls are:

- `CAMELID_BONSAI2_CUDA_PROJECTION=q8` (also the default): transposed Q8.
- `CAMELID_BONSAI2_CUDA_PROJECTION=tiled`: the previous optimized float path.
- `CAMELID_BONSAI2_CUDA_PROJECTION=baseline`: the original generic float kernel.
- `CAMELID_BONSAI2_CUDA_Q8_UNFUSED=1`: separate Hadamard and Q8 quantizer launches,
  using identical quantized bytes, for diagnostic comparisons.

## Measurements

The host is Windows x86_64, i7-11800H, 16 GB RAM, RTX 3060 Laptop 6 GiB, driver
576.83, and CUDA 12.9. The comparison uses the official Prism
`prism-b10777-e3de715` CUDA 12.4 package with installed CUDA 12.9 runtime DLLs.
Both engines use the same model hashes, ChatML counting/code/prose prompts,
greedy sampling, 2048 context slots, f16 KV, eight CPU threads, 32 output tokens,
and the official 2B drafter with a three-token proposal window.

Each prompt/mode warms up with eight output tokens and measures three complete
uncached requests. The native repetitions now share a loaded process instead
of recompiling CUDA for every repetition. Request KV/recurrent state resets
between calls. The native float and Q8 controls use the same final executable
and process policy; the port also reuses its loaded server. This differs from
the previous bundle's native process-per-repetition protocol, so optimization
ratios use the fresh float control recorded here.

The primary rate is output tokens divided by complete request duration, using
the median of the three durations. Loading, NVRTC compilation, and warmup are
excluded; prompt prefill and generation are included. Native timing starts
after tokenization; port timing also includes localhost HTTP, tokenization,
and response parsing. Callback `decode_tokens_per_second` fields are retained
only as diagnostics. All repetitions, token IDs, acceptance counts, and actual
memory placement remain in the JSON receipts. See [RESULTS](RESULTS.md) for every
case, including drafting slowdowns and cross-engine output differences.

The 27B comparison uses different strategies on this 6 GiB GPU: Camelid streams
packed weights to CUDA, while the port executes some target layers on CPU.
Actual placement for each mode is recorded in `comparison.json`. These results
do not describe an entirely GPU-resident 27B comparison on a larger GPU.

## Final profiles

The six final traces show how the remaining GPU cost shifts after the Q8
contractions. These totals include warmup and both prefills, so they are
diagnostic totals rather than the request durations used in RESULTS.

| Model | Float kernel time (ms) | Q8 kernel time (ms) | Float projection share | Q8 projection share |
| --- | ---: | ---: | ---: | ---: |
| 2B | 703.55 | 334.90 | 75.26% | 49.51% |
| 4B | 1527.48 | 767.60 | 79.46% | 60.90% |
| 9B | 2896.83 | 1147.54 | 83.00% | 55.96% |

In the Q8 traces, recurrent updates consume 16.0% of 2B kernel time, 17.3% of
4B, and 11.0% of 9B. Normalization uses another 6.2–8.4%; fused rotation and
quantization use 9.7%, 6.2%, and 20.3%, respectively. Projections remain the
largest component, especially the four-token 4B/9B kernels. These profiles
identify remaining work; the current patch does not claim port-level parity.

## Correctness and reproduction

Build the probe/server with release optimization, `--features cuda --jobs 2
--config profile.release.lto=false --config profile.release.codegen-units=16`.
The library test executable uses `--jobs 1
--config profile.release.package.camelid.opt-level=1` to fit host memory.
The CUDA source and NVRTC settings are identical. `manifest.json` binds the
source commit, source files, retained executables, build configuration, and
artifact hashes. Text receipts use LF bytes via `.gitattributes`.

An initial 27B state test overlapped release compilation and failed a 240 MiB
host allocation. The final suite runs after compilation and passes all sixteen
checks. The failed attempt is retained separately in ignored
`output/bonsai2-q8/27b-cache-host-memory-failure.log`; it is not a benchmark result.

Run `cuda-cache-qualification.mjs TEST_BINARY MODEL_DIRECTORY OUTPUT LOG_DIRECTORY`
with the Q8 policy. It checks fourteen CUDA fixtures/real-model scenarios and
two host memory regressions. The new fixture verifies every quantized byte
against CPU calculations, fused/separate rotation and quantization, independent
CPU integer contractions, partial K/row/token tiles, scalar/batched identity,
zero groups, and residual writes. Other checks cover independent float
references, all four models' full logits/state, rejected-suffix rollback,
attention boundaries, greedy/seeded sampling, stop, and cancellation behavior.

Run `cuda-quality.mjs TEST_BINARY MODEL_DIRECTORY OUTPUT_DIRECTORY` to score
three fixed 32-token float-path continuations under each policy. It records
every token's log probability and top choice, mean negative log likelihood,
and agreement. The predeclared gate permits at most 0.1 nats/token mean NLL
increase. These 96 tokens per model are a bounded numerical regression check,
not a broad quality or corpus-perplexity evaluation.

All 384 scored continuation tokens retain the float path's top choice. The
largest mean NLL increase is 0.000734 nats/token (4B). Each policy's per-token
scores are committed alongside `quality.json`; the evidence checker recomputes
the reported means, deltas, and agreement from those scores.

Run `spec-benchmark.mjs` with `CAMELID_BENCH_CUDA=1`,
`CAMELID_BENCH_REPETITIONS=3`, `CAMELID_BENCH_REUSE_PROCESS=1`, and the selected
projection policy. Run `cuda-reference-benchmark.mjs` for fresh port results.
Generate summaries with `compare-cuda-benchmarks.mjs qa/bonsai2/cuda-q8` and
`compare-cuda-q8.mjs`. Keep all builds, other GPU clients, and profiling separate
from the timed benchmark matrix.

For diagnostic profiles, use Nsight Systems 2025.1.3 with
`profile --trace=cuda,nvtx --sample=none --cuda-graph-trace=node
--cuda-event-trace=false --force-overwrite=true -o PREFIX PROBE MODEL
qa/bonsai2/count.chatml 32`. Set CUDA on, `CAMELID_QWEN35_CUDA_MAXPOS=2048`,
`CAMELID_PROBE_WARMUPS=1`, `CAMELID_PROBE_REPETITIONS=1`, and
`RAYON_NUM_THREADS=8`; unset drafting. Repeat for 2B/4B/9B and both `tiled`/`q8`
policies. Export with `nsys stats --report cuda_gpu_kern_sum --format csv
--output PREFIX PREFIX.nsys-rep`. The six CSVs and `profile.json` retain kernel
times and source/binary bindings. Trace timings include the warmup and both
prefills; the headline request rates come from separate unprofiled runs.

The GUI/API receipts use the production frontend and native library-server
harness, default image settings, an 8192-slot allocation, and supported Windows
CUDA routing. Their cold, short responses are functional checks; the request
rates in their screenshots are separate from the warmed 2048-slot benchmark.
`api-qualification.mjs` covers text, tool-call/result round trips,
1,044-token context recall, seeded sampling, and spatial vision.
`gui-smoke.mjs --send --default-image --require-supported --require-cuda
--require-request-rate` checks text and an attached image for each model.
Installed Tauri shell execution and Apple Silicon qualification are separate.

Record the final bundle with `record-cuda-q8-evidence.mjs COMMIT PROBE TEST_BINARY
SERVER SERVER_BUILD_ID`, then run `node qa/bonsai2/check-cuda-q8-evidence.mjs`.
