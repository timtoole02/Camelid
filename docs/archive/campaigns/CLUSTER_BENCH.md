# CLUSTER_BENCH.md — distributed parity lane, measured cost

This file records the latency cost of distributed inference **plainly and without spin**.
Distributed pipeline-parallel inference is *slower* than single-node would be when the model
fits — that is expected and is the point: the lane buys *capability* (a model spread across
nodes' RAM), not speed. The defensible result is token-identity to a single-node reference
with a receipt, never a throughput win.

## Phase 3 — two Macs over LAN (2026-06-13)

- Config: `two-mac-tinyllama-q8`, TinyLlama 1.1B Chat Q8_0
  (sha256 `a4c9bb1d…`, byte-identical on both nodes — verified).
- Topology: `mac-m4-coordinator` owns embedding + layers `[0,11)`; `mini2` (Mac mini M4,
  16 GB) owns layers `[11,22)` + output head + greedy sampling. One hidden-state vector
  (`[1,2048]` f32 = 8192 bytes) crosses the wire per token, one hop per token.
- Lane: CPU (resident paths disabled) on both nodes for deterministic, comparable math
  (DECISIONS D4). Build: the **same** `parity_node` binary on both nodes.
- Gate: token-identical to single-node `camelid` at 50 tokens, **two consecutive runs**.
  Result: **PASS** — both runs token-identical; sealed receipt
  `qa/distributed/two-mac-tinyllama-q8.json`, `receipt_id 33b79d8d…`,
  `first_divergent_generated_token_index = -1`. The receipt id is identical across runs
  (deterministic).

### Measured per-token cost (run 2, decode steps)

| Metric | Value |
|---|---|
| activation bytes / token (coord→worker) | 8192 |
| TTFT (prefill hop round-trip) | ~220 ms |
| avg coordinator-local compute / token (layers [0,11)) | ~72 ms |
| avg hop round-trip / token | ~111 ms |
| **distributed decode throughput** | **~5.5 tok/s** |

**Honest caveats (do not read these numbers as a speed claim):**
- The "hop round-trip" bundles the **worker's compute** (its 11 layers + output head, also
  ~70 ms on the CPU lane) **and** the wire. It is not pure network time. To isolate pure
  link latency, use `camelid bench-network` (Thunderbolt RTT measured here was ~0.6 ms; the
  Wi-Fi LAN is higher and variable).
- This is the **CPU lane** by construction (parity requires determinism). It is far slower
  than the production GPU-resident decode path; these numbers say nothing about camelid's
  normal single-node speed.
- Single-node would be faster than this two-node split for a model this size: one node runs
  all 22 layers with **no** per-token network hop. Distributing a 1.1B model is strictly a
  cost here — the demonstration is parity, not performance. The payoff (capability) only
  appears in Phase 4, where the model does not fit on one node.

### Operational notes (environment, not the engine)

- The Thunderbolt bridge (`169.254.0.0/16`) offers ~0.6 ms RTT but its link-local address
  reassigned mid-session; the gated runs used the stable LAN IP (redacted).
- mini2's worker, launched over SSH, was initially throttled by macOS App Nap / Wi-Fi
  power-save between runs (TCP `accept()` stalled → connect timeouts). Running the worker
  under `caffeinate -dimsu` resolved it. The coordinator also uses bounded connect-retry +
  whole-run retry so a transport flake retries the run and never relaxes the parity gate
  (operating rule: fix the transport, never the threshold).

## Phase 4 — heterogeneous (Mac + Pi): cross-ARM parity measured (2026-06-13)

Setup: 3× Pi 5 (16 GB, aarch64 Linux) found; camelid built **natively on camelid1** in
2m10s, zero errors — aarch64-Linux portability proven. The SAME source builds a clean Linux
binary (no Metal/Accelerate). (The earlier "Pi crash" was a self-inflicted
`pkill -f parity_node` matching the SSH command's own cmdline, not a code bug.)

### `hetero-mac-pi-tinyllama-q8` — Apple coordinator + Cortex worker, TinyLlama Q8_0

| Metric | Value |
|---|---|
| topology | mac-m4 (Apple) [0,11) + embedding/ref → camelid1 Pi (Cortex) [11,22) + head |
| GGUF | byte-identical on both nodes |
| activation bytes/token | 8192 |
| avg coord-local compute/token | ~72 ms |
| avg hop round-trip/token (incl. Pi compute + LAN) | ~148 ms |
| decode throughput | ~4.6 tok/s |
| **token-identical to single-node reference** | **NO** |
| **first divergent generated token** | **25** (of 50) |

**Finding (recorded, not smoothed over — operating rule #6 / spec Phase 4):** the
heterogeneous Apple+Cortex pipeline reproduces the all-Apple reference exactly for **24
generated tokens**, then diverges at token 25. Root cause: cross-platform IEEE-754 differences
(FMA contraction / accumulation order) between Apple and ARM Cortex CPUs — the last-bit
differences in the Pi-computed layers [11,22) compound until a greedy argmax flips. Sealed
receipt: `qa/distributed/hetero-mac-pi-tinyllama-q8.json` (`validated:false`,
`first_divergent_generated_token_index=25`). The lane stays **experimental** for heterogeneous
Apple+Cortex configs: token-identity does NOT hold across this CPU-architecture boundary.

### `hetero-mac-pi-llama13b-q8-capped` — too-big capability (reference gap)

Each shard loads only its slice: Pi worker [20,40) reported 26 GB f32-estimate materialization
(~6.5 GB Q8 resident) **within a 40 GB cap**; the full model's ~52 GB f32 estimate would
exceed it (→ fits no single capped node). The **single-node reference OOM'd the 16 GB Mac**
loading the full 13 GB model — direct evidence the whole model exceeds one 16 GB node while the
shards fit. Consequently a same-engine on-device reference is impossible here; the 13b parity
verdict needs the **llama.cpp oracle** reference (spec-sanctioned for models that fit nowhere
whole). Documented next step: add an external/oracle-reference mode to `parity_node`.

**Honest summary:** the heterogeneous Mac+Pi cluster *runs* sharded inference (capability
demonstrated), but Apple↔Cortex output is **not** token-identical (diverges at token 25) — so
heterogeneous configs are experimental-divergent, and the genuinely-too-big 13b parity awaits
an oracle reference. No speed claims; distributed CPU-lane decode is slower than single-node by
design.
