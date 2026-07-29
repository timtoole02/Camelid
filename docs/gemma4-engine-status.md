# Gemma 4 in Camelid — engine status

**Status (correctness milestone): Gemma 4 runs correctly inside Camelid's
from-scratch engine and produces output token-identical to llama.cpp.** It is now
also served through the HTTP API (`/v1/chat/completions`, streaming and
non-streaming; the lane is on by default, opt-out `CAMELID_GEMMA4_SERVE=0`) and surfaced in the UI as a
supported + downloadable model. The one-time load is ~11x faster (mmap wire-backed
weights, no eager decode). Warm decode is now ~6.75 tok/s (up from ~1.84 after the
Q8×Q8 `sdot` matvec) — usable, but still bounded by the per-token weight read and
by cold mmap faults on an unwarmed 16GB box; do not describe it as "fast."

## What works

- **Architecture support.** `general.architecture = "gemma4"` is recognized; the
  config, the `gemma4` SentencePiece tokenizer, and the full weight binding
  (per-layer-type attention, QK-norm, the five Gemma norms, Per-Layer-Embeddings)
  all load and validate against a real GGUF.
- **Forward pass + generation.** A from-scratch forward pass (`src/gemma4_runtime.rs`)
  with embedding scale, per-layer-type attention (head_dim 256 sliding / 512
  global), QK-norm before RoPE, weightless v-norm, dual-θ RoPE, sliding-window
  masking, GeGLU, cross-layer KV sharing, the 7-step PLE injection, per-layer
  output scale, tied logits, and final soft-cap.
- **Validated bit-against-llama.cpp** at three levels: a single forward
  (`tests/gemma4_forward.rs` asserts argmax 9079), full teacher-forced
  continuation (every position matches), and real autoregressive greedy decode
  via the CLI (token IDs identical — see proof below).
- **In-engine runtime + CLI.** `Gemma4Runtime::load()` + `generate_greedy()`,
  driven by `camelid gemma4-generate`. Incremental KV cache (O(n) decode).
- **HTTP serve (on by default).** `camelid serve --model <gemma4.gguf>` loads
  the runtime and serves `/v1/chat/completions` both non-streaming and
  streaming (SSE, OpenAI `chat.completion.chunk` shape). Opting out with
  `CAMELID_GEMMA4_SERVE=0` restores the metadata-only load.
  The Gemma chat template lives in one place (`gemma4_chat_prompt`). `/v1/health`
  reports `backend`, `model_family`, and `gemma4_available`. The existing
  Llama/3B serve path is untouched, and a gemma4 request with no loaded runtime
  fails clearly (503 `model_not_ready`) — never a silent fallthrough. Verified
  live: health correct; non-streaming returns `Paris` (finish_reason `stop`, no
  prompt echo); streaming yields incremental token deltas then `[DONE]`; unknown
  model id returns a clear `model_not_found` error.

## What does NOT work yet

- **Decode is ~6 tok/s warm** — usable but not at the ~120 GB/s M4 wall (decode
  reads the full 8GB of Q8 weights per token at ~54 GB/s). On CPU this is close
  to the practical ceiling: the matvecs are already large enough to saturate all
  10 cores, so the limit is unified-memory throughput, not dispatch — quantizing
  the shared activation once (`matvec_q`) was measurably perf-neutral. Cold runs
  are dominated by faulting the 8GB wire map from disk, which the 16GB box cannot
  durably cache between runs.

### GPU port: scoped, not yet done

The only lever that reaches the bandwidth wall (~13–15 tok/s) is running decode
on the GPU. Two findings from scoping it:

- **The fit problem is solved.** `metal::q8_wire_nocopy_buffer` wraps a
  page-aligned wire allocation with `new_buffer_with_bytes_no_copy`, so the GPU
  reads our 34-byte wire bytes in place — no second resident copy, fits in 16GB.
  (Weights would move from the `GgufWireMmap` to `wire_mmap::WirePages`, which is
  anonymous host memory and so adds memory-pressure the file-backed mmap avoids.)
- **A hybrid (GPU matvecs + CPU glue) is the wrong shape.** The fast GPU path is
  the *fully resident* Llama graph (`prepare_token`/`encode_attention_block`: one
  command buffer per token, KV on-GPU, zero round-trips). Gemma's forward can't
  reuse it — it needs new Metal kernels for QK-norm, the dual-θ / per-layer-type
  RoPE + head_dim, sliding-window masking, cross-layer KV, the 7-step PLE
  injection, GeGLU, and the final soft-cap. Offloading only the Q8 matvecs while
  keeping the glue on CPU pays ~140 commit/wait round-trips per token
  (`start_inference_session`/`synchronize_active_session` batches independent
  matvecs, but the norm→matvec→norm dependency chain forces a sync between most),
  and a non-resident per-dispatch GEMV is unlikely to hit the wall — so it risks
  a net regression. The real win is the resident-graph port, which is a multi-step
  effort (gemma-specific kernels + parity re-validation), not a single change.

## Recently fixed

- **Decode matvec: scalar f32 mul-add → Q8×Q8 `sdot` (~1.84 → ~6.75 tok/s warm,
  ~3.7x).** The wire-mmap matvec quantizes the activation row to Q8 once
  (`quantize_q8_0_blocks`), then dots each weight row read in place from the
  34-byte wire blocks against it via the NEON `sdot` kernel the Llama path uses
  (`q8_0_wire_row_dot` in `src/inference.rs`, runtime-gated on `dotprod` with a
  portable scalar fallback). Quantizing the activation mirrors llama.cpp's own
  Q8_0 matmul inputs, so the parity tests still pass (teacher-forced continuation
  matches every position; greedy decode emits the identical token ids).

## Recently fixed (earlier)

- **Runtime load time: ~238s → ~21s (~11x).** Weights are no longer eagerly
  decoded into 8GB of `Q8_0Block` structs. The runtime memory-maps the GGUF once
  (`wire_mmap::GgufWireMmap`) and reads Q8 wire blocks in place, decoding the f16
  scale + 32 i8 quants inline in the matmul (vectorized 32-lane dot). Load is now
  bounded by streaming the 8GB from disk (`MADV_WILLNEED` prefetch) instead of the
  per-block decode; when the page cache still holds the file a reload skips the
  read. Generation speed and the exact greedy token IDs are unchanged. (Note: the
  `serve` path additionally hashes the full 8GB GGUF once for the receipt lane, so
  serve startup is slower than the bare `gemma4-generate` runtime load, and on a
  16GB box the page cache does not durably hold an 8GB model between runs.)
- **Wired into the UI.** Gemma 4 E4B-It is a supported + downloadable model in
  the frontend catalog and a `/api/capabilities` compatibility row.

## Still limited

- **Only E4B (Q8_0) exercised.** The dense (12B/31B) and MoE (26B-A4B) variants
  share the code but are untested; Camelid is Q8_0-only (no K-quants).
- **RAM:** Gemma 4 (~8GB) and the 3B backend (~3.4GB) together thrash a 17GB box.
  Run one model at a time.

## Proof artifact

| Field | Value |
|---|---|
| Model file | `/Volumes/Untitled/models/gemma-4-E4B-it-Q8_0.gguf` |
| Size | 8,192,951,456 bytes (7.6 GiB) |
| sha256 (first 1 GiB) | `0ddf918f3ec40d2bac8fd2e7e463253195993e6c7c08980b3164866a57908f3a` |
| GGUF arch | `gemma4`, 42 layers, hidden 2560, heads 8, kv_heads 2, head_dim 256/512, ffn 10240, ctx 131072, vocab 262144 |
| GGUF specials | sliding_window 512, final_logit_softcapping 30, rope θ 1e6/1e4, num_kv_shared_layers 18, tokenizer `gemma4` (SPM) |
| Command | `camelid gemma4-generate <gguf> --prompt "The capital of France is" --max-tokens 12` |
| Prompt | `The capital of France is` |
| Prompt token IDs | `[2, 818, 5279, 529, 7001, 563]` (matches llama.cpp `/tokenize`) |
| **llama.cpp greedy IDs** | `[9079, 236761, 108, 1018, 14977, 53121, 2900, 563, 506, 5279, 529, 7001]` |
| **Camelid greedy IDs** | `[9079, 236761, 108, 1018, 14977, 53121, 2900, 563, 506, 5279, 529, 7001]` — **identical** |
| Decoded text | `The capital of France is Paris.\n\n**Question:** What is the capital of France` |
| Load time | ~21 s (mmap wire-backed, ~11x faster than the original ~238 s eager decode; `serve` adds a one-time 8GB receipt hash on top) |
| Generation speed | ~1.16 tok/s @ 12 tokens (~2.26 tok/s @ 20) |

The oracle was captured with `llama-server` (llama.cpp b9430) on the same GGUF,
greedy (`temperature 0, top_k 1`). The two engines never run simultaneously (RAM).

## How to reproduce

```sh
# 1. Camelid greedy (this engine):
CARGO_TARGET_DIR=/Volumes/Untitled/cargo-targets/Camelid-push \
  cargo run --release --bin camelid -- gemma4-generate \
  /Volumes/Untitled/models/gemma-4-E4B-it-Q8_0.gguf \
  --prompt "The capital of France is" --max-tokens 12
# prints: token_ids + decoded text + load/gen timing

# 2. Bit-against-llama.cpp parity tests (forward + multi-token, gated on the gguf):
CARGO_TARGET_DIR=/Volumes/Untitled/cargo-targets/Camelid-push \
  CAMELID_GEMMA4_GGUF=/Volumes/Untitled/models/gemma-4-E4B-it-Q8_0.gguf \
  cargo test --release --test gemma4_forward -- --nocapture

# 3. llama.cpp oracle (stop any Camelid backend first — RAM):
llama-server -m /Volumes/Untitled/models/gemma-4-E4B-it-Q8_0.gguf -ngl 99 -c 512 &
curl -s localhost:8080/completion -d \
  '{"prompt":"The capital of France is","n_predict":12,"temperature":0,"top_k":1}'

# 4. HTTP serve (on by default; opt-out CAMELID_GEMMA4_SERVE=0) — non-streaming and streaming chat:
camelid serve \
  --model /Volumes/Untitled/models/gemma-4-E4B-it-Q8_0.gguf --addr 127.0.0.1:8231
curl -s 127.0.0.1:8231/v1/health            # backend=gemma4-runtime, gemma4_available=true
curl -s 127.0.0.1:8231/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Gemma-4-E4B-It","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":16,"temperature":0}'
# add "stream":true for SSE (chat.completion.chunk deltas + [DONE]).
```

## Where the code lives

- `src/model.rs` — `gemma4` arch recognition, `Gemma4Metadata`, `Gemma4Binding`.
- `src/inference/gemma4.rs` — forward-pass primitives (embed scale, GeGLU, softcap).
- `src/gemma4_runtime.rs` — the runtime (load + incremental decode +
  `generate_greedy` / `generate_greedy_streaming`).
- `src/api/mod.rs` — serve integration: `gemma4_serve_enabled`, `model_family`,
  `gemma4_chat_prompt`, `resolve_gemma4_runtime`, `gemma4_chat_nonstreaming`,
  `gemma4_chat_streaming`, the gated runtime load, and the `/v1/health` fields.
- `src/main.rs` — `gemma4-generate` CLI subcommand.
- `tests/gemma4_forward.rs` — the bit-against-llama.cpp parity tests.
