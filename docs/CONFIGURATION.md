# Configuration Guide

Last updated: 2026-05-06

This guide documents Camelid's current local configuration reality without pretending every workflow is fully automated.

## Toolchain expectations

### Rust / Cargo

Camelid currently requires Rust/Cargo 1.87+.

On hosts where `/usr/bin/cargo` is older than the required toolchain, prefer one of these paths:

```bash
source "$HOME/.cargo/env"
```

or:

```bash
scripts/with-rustup-cargo.sh build --release --bin camelid
scripts/with-rustup-cargo.sh test --all-targets --all-features
```

### Node / npm

The frontend expects a working Node.js + npm install. Use `npm ci` when you want a reproducible install from the committed lockfile.

## Backend runtime defaults

The common local backend bind address in repo docs is:

```text
127.0.0.1:8181
```

Typical start command:

```bash
target/release/camelid serve --addr 127.0.0.1:8181
```

Appliance-style start command:

```bash
target/release/camelid serve --model /path/to/model.gguf
```

That startup path loads the model immediately and applies the default `auto` execution profile for the current host. Use `CAMELID_PROFILE=safe|auto|experimental|debug` when you need to change planner behavior; keep lower-level experiment env vars as developer overrides rather than the primary user workflow.

## Frontend API base override

The frontend defaults to:

```text
http://127.0.0.1:8181
```

Override it for local dev/build with:

```bash
VITE_CAMELID_API_BASE=http://127.0.0.1:8181 npm run dev
```

You can also edit the API base in the UI while testing.

## Model-path guidance

Repo examples often use paths such as:

```text
models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-1B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Llama-3.2-3B-Instruct-Q8_0.gguf
$CAMELID_MODEL_DIR/Meta-Llama-3-8B-Instruct-Q8_0.gguf
```

These are example local paths, not a guarantee that the repo fetches or manages model files for you.

Recommended practice:

- keep local GGUFs outside version control
- use stable local paths during validation so commands and artifacts stay reproducible
- avoid documenting private absolute paths in public artifacts or docs

## Environment and local-shell assumptions

Current public docs assume:

- `cargo` resolves to a Rust 1.87+ toolchain
- `node` and `npm` are available for frontend work
- `llama-server` is in `PATH` only when you are running parity comparisons

Backend runtime knobs used during performance work:

- `CAMELID_PREFILL_CHUNK_TOKENS` controls how many non-final prompt tokens the backend processes per chunk in the chunked prefill path. Default: `256`, matching the current long-prefill performance lane while keeping the global lazy Q8 file cache disabled outside explicit/scoped reuse. Set it to `1` to force the older sequential prefill path while debugging; invalid/zero values fall back to the default. This is a runtime/performance knob only; it is not support evidence for any model row by itself; the separate published source/runtime-head PASS bundle and synchronized docs/API/frontend updates are what close exact Llama 3 8B checked 1024/2048 packs; the knob itself is not evidence for today's checkout.
- `CAMELID_PREFILL_LAYER_MAJOR` controls the long-context prefill schedule that processes all prefill chunks one layer at a time, reusing file-backed Q8_0 weights across chunks before moving to the next layer. By default it is enabled only when lazy Q8_0 file-backed weights are present. Set it to `0`, `false`, `off`, or `disabled` to force the older chunk-major schedule while debugging.
- `CAMELID_PREFILL_LAYER_MAJOR_CHUNK_TOKENS` controls the per-layer prompt chunk size only for the layer-major schedule. Default: `512`, unless `CAMELID_PREFILL_CHUNK_TOKENS` is explicitly set, in which case the shared chunk setting is reused for comparability. It also accepts `all`, `full`, `prompt`, or `unbounded` for one diagnostic full-prompt prefill chunk. This is a runtime/performance knob only and does not promote any 8B 1024/2048 support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES` controls the layer-major-only scoped Q8_0 raw-byte reuse window when lazy file-backed Q8_0 weights are present and `CAMELID_Q8_0_FILE_CACHE_BYTES` is unset. Default: `268435456` (256 MiB) only for multi-chunk layer-major prefill, where file-backed Q8_0 weights can be reused across chunks; single-chunk prefill skips the default scoped cache unless this scoped knob is set explicitly. Set it to `0` to disable the scoped layer-major cache, or set the global cache knob explicitly to take over all Q8 file-reader cache sizing. This is a bounded RSS/read-reuse tuning knob only and does not promote any 8B support bucket by itself.
- `CAMELID_PREFILL_LAYER_MAJOR_ATTRIBUTION` enables optional structured per-layer/per-prefill-chunk attribution for the layer-major schedule inside forward-memory timings. This is diagnostic instrumentation for memory/Q8 read attribution, not support evidence or a promotion signal by itself.
- Q8 byte-count knobs accept plain bytes or binary suffixes (`KiB`/`MiB`/`GiB`, also `K`/`M`/`G`; underscores and spaces are ignored). This covers `CAMELID_Q8_0_FILE_CACHE_BYTES`, `CAMELID_PREFILL_LAYER_MAJOR_Q8_0_FILE_CACHE_BYTES`, `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES`, `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES`, and `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` without changing their numeric defaults.
- `CAMELID_Q8_0_FILE_READER_CHUNK_BYTES` controls the target Q8_0 row-read chunk size for borrowed/file-backed row readers. Default: `33554432` (32 MiB). This is a read-pattern/performance knob only.
- `CAMELID_Q8_0_FILE_READER_OUTPUT_SCRATCH_BYTES` caps reusable f32 output scratch for multi-row lazy-Q8 file-backed matmuls. Default: `67108864` (64 MiB). This is an RSS/read-reuse tuning knob only.
- `CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES` caps how much per-thread Q8 file-reader scratch capacity is retained after oversized row, scale, quantized-input, and output chunks. Default: `67108864` (64 MiB). This is an RSS headroom knob only; it does not promote 8B 1024/2048 support by itself.
- `CAMELID_KV_CACHE_GROW_TOKENS` controls KV-cache allocation growth for model-sized contexts. Default: `256` positions when context length is at least 512; tiny diagnostic/test contexts keep exact one-position growth. This reduces repeated realloc/copy churn during decode and is a runtime performance knob only.
- `CAMELID_METAL_Q8` / `--metal-q8` enables the macOS Metal Q8_0 encoded file-backed row-dot path. It falls back to CPU when unavailable and is not support evidence by itself.
- `CAMELID_PROFILE` selects the execution-planning profile: `safe` keeps only conservative known-good paths, `auto` keeps default-off experiment lanes disabled, `experimental` allows evidence-lane experiments with a warning, and `debug` favors diagnostics over performance claims.
- On the Ubuntu x86_64 dense Llama Q8_0 evidence lane, the appliance planner keeps x86 Q8 experiment flags off by default. Manual developer overrides remain evidence-lane only and must not be treated as support-contract, portability, accelerator-backend, or broader model-family evidence. Current reference truth for this lane is `qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/README.md`.
- `CAMELID_X86_Q8_REPACK=on` is a default-off Ubuntu x86_64 developer experiment that loads selected dense Llama Q8_0 linears into backend-owned packed runtime storage instead of retaining a duplicate row-major packed sidecar. The current x86 slice covers the dense attention projection family (`blk.*.attn_{q,k,v}.weight`, `blk.*.attn_output.weight`), dense FFN gate/up/down rows, and `output.weight`; leave it unset for the safe fallback.
- `CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER=on`, `CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER=on`, and `CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER=on` are default-off Ubuntu x86_64 developer experiments that let one-row dense attention Q/K/V and attention-output projections consume backend-owned packed Q8_0 runtime storage directly. They fall back unless the runtime plan, tensor type, shape, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense attention Q/K/V packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.attn_{q,k,v}.weight`, and falls back unless all three projections, the runtime plan, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense attention-output packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.attn_output.weight`, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_DECODE_OWNER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row decode output projection consume backend-owned packed `output.weight` storage directly; it falls back unless the tensor, shape, and packed-row guards match exactly.
- `CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row `output.weight` packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `output.weight`, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly. Current evidence for this exact flag is local parity/gate coverage only; no Ubuntu timing/profiling validation is recorded for that local slice, and it must not be treated as Ubuntu throughput, support, portability, or default-on evidence.
- `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row dense FFN gate/up activation consume backend-owned packed `blk.*.ffn_{gate,up}.weight` storage directly with one input quantization; it falls back unless both tensors, shapes, and packed-row guards match exactly. `CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION=on` is a narrower default-off follow-on that fuses the gate/up activation write for that same decode route after the same guards pass.
- `CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for multi-row dense FFN gate/up packed-runtime matmul. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `blk.*.ffn_{gate,up}.weight`, quantizes the shared activation rows once for both projections, and falls back unless the runtime plan, tensor type, dimensions, row count, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER=on` is a default-off Ubuntu x86_64 developer experiment that lets one-row dense FFN-down projection consume backend-owned packed `blk.*.ffn_down.weight` storage directly; the execution planner clears stale values unless a future slice explicitly owns and validates that consumer gate for the run.
- `CAMELID_X86_Q8_PACKED_ROWS4_MATMUL=on` is a default-off Ubuntu x86_64 developer experiment for the current dense FFN-down multi-row packed-runtime matmul slice. It only consumes backend-owned `PackedRows4` Q8_0 runtime storage for `ffn_down`, and falls back unless the runtime plan, tensor type, dimensions, row grouping, and packed interleave guards match exactly.
- `CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL=on`, `CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED=on`, and `CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2=on` are default-off Ubuntu x86_64 developer experiments for FFN-down rows4 GEMM4 work over backend-owned `PackedRows4` runtime storage. Treat the AVX2 flag as code/route-gate evidence only until a fresh canonical Ubuntu parity plus same-host timing/profiling bundle retains it; do not treat any of these flags as support, portability, production-throughput, RSS, or default-on evidence by themselves.
- Explicit x86 disables remain available as developer overrides. The execution planner respects `CAMELID_X86_Q8_REPACK=off` and `CAMELID_X86_Q8_KERNEL=off|disabled` by failing closed to the safe CPU path, and it manages the x86 decode-consumer/matmul/GEMM4 flags so stale owner experiments are cleared unless explicitly selected in a developer run.
- `CAMELID_METAL_Q8_RETAINED` enables the retained-Q8 all-Metal kernel path for focused kernel experiments. Current local 3B profiling showed all-Metal retained Q8 is slower than the retained-Q8 CPU path, so normal macOS serving keeps this off unless explicitly enabled.
- `CAMELID_HYBRID_Q8_RETAINED` controls the retained-Q8 CPU+Metal split path for single-row decode projections. It defaults to off because same-host Apple Silicon sweeps showed the Metal suffix scheduler was slower and used more RSS than the paired CPU Q8 path on the measured 3B short-decode gate. Set it to `1`, `true`, `on`, or `enabled` to opt into the experiment; set it to `0`, `false`, `off`, `disabled`, or `cpu` to force CPU-only. When enabled, it launches a Metal command buffer for a suffix of output rows while CPU threads compute the rest, then merges the output. Tune the GPU slice with `CAMELID_HYBRID_Q8_GPU_PERCENT` (default `10`, capped below 100) or `CAMELID_HYBRID_Q8_GPU_ROWS`.

If a command depends on more than that, document the requirement in the same PR.

## Maintainer-only/private workflows

The following are intentionally not public contributor requirements:

- SSH-based validation-lane access
- private host aliases or machine-specific setup
- unpublished remote worktree conventions
- local absolute paths from a maintainer workstation

Public docs may mention that some promotion-grade reruns happen on an approved Ubuntu validation lane, but they should not expose private operator details.

When summarizing Ubuntu validation status, distinguish host access from evidence status. Do not report negative host-access status unless that exact validation attempt was run for the current evidence bundle and the stderr is cited in the bundle. If remote validation was not attempted, say that plainly and keep the claim scoped to the missing evidence, such as "no Ubuntu timing/profiling validation is recorded."

## Documentation rule of thumb

When adding a new variable, path convention, or host assumption:

1. document the public/local requirement here if contributors need it
2. keep private operator details out of public docs
3. avoid claiming a workflow is turnkey unless the repo actually makes it turnkey
