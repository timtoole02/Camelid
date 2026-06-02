# Camelid vs MLX Benchmark Report

> Evidence-first comparison of Camelid against Apple's MLX-LM (with llama.cpp as a
> reference) on a single Apple Silicon machine. This is an engineering exercise,
> not marketing: it reports losses as plainly as wins and does not claim a win
> where the measurements do not support one.

## Summary

On a single Apple M4 (16 GB) running Llama-3.2-3B-Instruct at 8-bit, temperature 0,
warm, **Camelid does not currently beat MLX-LM on decode throughput, time-to-first-token,
or peak memory.** The decisive reason is architectural, not incidental:

- **MLX-LM and llama.cpp both run the forward pass on the Metal GPU** (~28–29 tok/s decode).
- **Camelid runs entirely on the CPU** here (~12–17 tok/s decode). Its Metal work so
  far is a single Q8_0 GEMV kernel; there is no end-to-end GPU-resident forward pass yet.

So the honest current headline is: *Camelid is the only CPU-only runtime in this
comparison, and it loses single-node decode to the GPU-backed runtimes by ~1.7×.*
Camelid's credible differentiators today are **operational** (direct GGUF loading,
Rust-native, single static binary, no Python/conversion step) and a **distributed
multi-Mac lane** that this single machine cannot measure.

## Hardware

- Apple M4, 16 GB unified memory, macOS (Darwin 25.5).
- Single machine; same thermal state across runs.

## Software Versions

- Camelid: Cameleer-vendored `camelid` build (carries the CPU residency + ffn_down
  decode wins). Measured via the `camelid bench-generate` subcommand.
- MLX-LM `0.31.3`, MLX (Apple), Python 3.
- llama.cpp (Homebrew, Metal/BLAS backend) via `llama-bench`.

## Models Tested

| Runtime | Model | Quant | Format |
| --- | --- | --- | --- |
| Camelid | Llama-3.2-3B-Instruct | Q8_0 | GGUF (direct) |
| MLX-LM | mlx-community/Llama-3.2-3B-Instruct-8bit | 8-bit | MLX |
| llama.cpp | Llama-3.2-3B-Instruct | Q8_0 | GGUF |

Token counts differ slightly per tokenizer; each runtime reports its own
`prompt_tokens`. Exact cross-runtime token parity is **not** claimed. Q8_0 (GGUF)
and MLX 8-bit are comparable but not bit-identical quantizations.

## Methodology

- Same machine, prompts, generated-token count (`--max-tokens`), temperature 0
  (greedy/argmax, deterministic).
- 1 warmup + N measured iterations; model loaded once per process (load time
  reported separately from generation).
- TTFT = wall time to the first generated token with the model already loaded
  (prefill + first decode). Decode tok/s = tokens 2..N over decode wall time.
- Peak memory from `getrusage(RUSAGE_SELF).ru_maxrss`, cross-checked with
  `/usr/bin/time -l`.
- Harness: [`tools/bench/mlx-compare/`](../../tools/bench/mlx-compare/README.md).

## Results

Initial baseline lane: 13-token prompt, 64 generated tokens, temperature 0, warm,
median of 3 measured iterations. (The 128/512/2k/8k prompt matrix is wired in the
harness and is the documented next step; see *Next Optimization Targets*.)

| Runtime | Backend | Decode tok/s | TTFT (ms) | Peak mem (GB) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| MLX-LM 8-bit | **GPU (Metal)** | **28.9** | **144** | **3.26** | reference ecosystem |
| llama.cpp Q8_0 | **GPU (Metal)** | **28.6** | — | — | `llama-bench` tg64; no per-req TTFT |
| Camelid Q8_0 (`MAC_Q8_REPACK=1`) | CPU | 16.8 | ~790 | 3.49 | packed-rows4 decode |
| Camelid Q8_0 (default) | CPU | 12.3 | 454 | 3.59 | retained Q8 blocks |
| Camelid Q8_0 (canonical `main`) | CPU | 7.0 | 2924 | — | missing the CPU decode wins |

### TTFT

MLX wins (144 ms). Camelid's best TTFT is the **non-repack** path (454 ms);
enabling `CAMELID_MAC_Q8_REPACK` improves decode but *raises* TTFT to ~790 ms
(repack/first-pass cost). There is a real decode-vs-TTFT tradeoff inside Camelid.

### Decode Throughput

GPU runtimes (~28–29 tok/s) clearly lead. Camelid CPU tops out at ~16.8 tok/s
with the packed-rows4 path. This is a CPU-vs-GPU gap, not a kernel-tuning gap.

### Prompt Prefill

llama.cpp Metal prefill ≈ 189 tok/s (pp13). MLX prefill ≈ comparable. Camelid CPU
prefill is the dominant component of its higher TTFT.

### Memory

Close: MLX 3.26 GB vs Camelid 3.49–3.59 GB peak. MLX is marginally lower. Both
keep the full model resident in unified memory.

### Cold Start / Load

All three cold-load the ~3.2 GB model in roughly the same ballpark on this machine
(tens of seconds; Camelid streams weights with the OS page cache disabled). Load is
reported separately and is not part of the decode/TTFT figures.

## Where Camelid Wins

On this single-node 3B Q8 lane: **none of decode, TTFT, or peak memory.** Stating
that plainly is the point of the exercise. Camelid's honest differentiators are not
raw single-node speed:

- **Operational**: loads GGUF directly (no MLX/HF conversion step), is a single
  Rust-native static binary with no Python runtime, and exposes an OpenAI-compatible
  local API.
- **Distributed (unmeasured here)**: a Rust pipeline-parallel runtime across multiple
  consumer Macs — a different *shape* than single-node MLX. This needs ≥2 machines and
  is the most promising place to look for a defensible, distinct win.

## Where MLX Wins

- Decode throughput (~1.7× via the Metal GPU).
- TTFT (~144 ms vs Camelid's 454–790 ms).
- Peak memory (marginally).
- A mature, batteries-included ecosystem (sampling, fine-tuning, distributed).

## Known Unfairness / Gaps

- **CPU vs GPU**: the headline gap is that Camelid runs on CPU while MLX/llama.cpp run
  on Metal. A fair "Camelid GPU vs MLX GPU" comparison cannot be made until Camelid has
  a GPU-resident forward pass (only a Q8_0 GEMV kernel exists today).
- **Quantization**: Q8_0 (GGUF) vs MLX 8-bit are comparable, not identical.
- **Single lane**: only the ~13-token prompt / 64-token output lane is reported here;
  the 128/512/2k/8k matrix is not yet run.
- **Single machine**: the distributed lane is untested.
- The public canonical Camelid repo is missing the CPU decode wins (7 tok/s); these
  live in the Cameleer-vendored build (measured here).

## Next Optimization Targets

Ranked by what the measurements implicate:

1. **GPU-resident forward pass** — the only path to GPU-class decode (~28+ tok/s). The
   batched command-buffer + SIMD Q8_0 kernel already landed; the remaining work is
   running the whole forward pass on Metal with no CPU readback between ops.
2. **Resolve the decode-vs-TTFT tradeoff** of `CAMELID_MAC_Q8_REPACK` (repack lazily or
   in the background so TTFT does not regress).
3. **Port the CPU decode wins into canonical Camelid** (7 → ~17 tok/s) so the public
   repo reflects real performance.
4. **Distributed two-Mac lane** — measure a model that does not fit comfortably on one
   16 GB Mac, where Camelid's shape is genuinely different from single-node MLX.
5. Run the full 128/512/2k/8k prompt matrix.

## Reproduction Commands

```bash
# Camelid (fast CPU decode path)
CAMELID_MAC_Q8_REPACK=1 ./camelid bench-generate <model>.gguf \
  --prompt-file tools/bench/mlx-compare/prompts/prompt-128.txt \
  --max-tokens 128 --temperature 0 --warmup --iterations 10 --json

# MLX-LM
python3 tools/bench/mlx-compare/lib/mlx_generate.py \
  --model mlx-community/Llama-3.2-3B-Instruct-8bit \
  --prompt-file tools/bench/mlx-compare/prompts/prompt-128.txt \
  --max-tokens 128 --temperature 0 --warmup --iterations 10

# llama.cpp reference
llama-bench -m <model>.gguf -p 128 -n 128

# Full matrix (all runtimes, all prompt sizes)
MODEL=<model>.gguf MLX_VENV=<venv> MLX_MODEL=mlx-community/Llama-3.2-3B-Instruct-8bit \
HF_HOME=<hf-cache> CAMELID_BIN=<camelid-binary> CAMELID_MAC_Q8_REPACK=1 \
ITERS=10 MAX_TOKENS=128 PROMPTS="128 512 2k 8k" \
bash tools/bench/mlx-compare/bench.sh
```

## Raw Evidence Bundle

Run `tools/bench/mlx-compare/bench.sh` to produce a timestamped bundle under
`qa/evidence-bundles/mlx-compare-<UTC>/` with `hardware.txt`, `versions.txt`,
`commands.txt`, the prompts, per-iteration `raw/*.jsonl`, and aggregated
`summaries/results.{json,md}`.

## Conclusion

Current result: **Camelid does not beat MLX-LM in the tested single-node lane.**

Camelid wins:
- Nothing on single-node decode / TTFT / memory today.
- (Operationally) direct GGUF loading and a Python-free Rust binary — not a speed claim.

MLX wins:
- Decode (~28.9 vs 16.8 tok/s), TTFT (144 vs 454–790 ms), and peak memory.

The most credible public claim is narrow and honest:
> "I am not trying to replace MLX. MLX is excellent. Camelid is a Rust-native,
> GGUF-first local runtime; on single-node decode it is currently CPU-bound and
> behind MLX's Metal path, and the work to contest that lane is a GPU-resident
> forward pass plus the distributed multi-Mac shape."

The next engineering target is the **GPU-resident forward pass** — without it,
Camelid is bringing a CPU to a GPU fight on the decode lane.
