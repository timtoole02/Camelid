# Camelid Distributed (Pipeline-Parallel) — Validation & 2-Mac Runbook

> Camelid's distinct lane vs. single-node MLX is not raw speed — it's running a model
> **across several consumer Macs**, so each node holds only a fraction of the weights.
> This documents the on-machine validation (including a real bug fixed in the process)
> and the runbook for a real two-Mac measurement over Thunderbolt.

## What was validated (single machine, loopback)

Two Camelid processes on `127.0.0.1`, splitting Llama-3.2-3B-Instruct-Q8_0 (28 layers):

- **master** owns layers `0..14` + the token embedding,
- **worker** owns layers `14..28` + the final norm and output projection (it samples).

Result (temperature 0, greedy):

- **Correctness**: the distributed pipeline produced output **identical to single-node**
  generation for the same prompt — e.g. *"The Rust borrow checker is a tool that prevents
  the compiler from accepting code that would otherwise lead to memory safety issues…"*.
  Greedy + same kernels ⇒ exact match is the correctness oracle.
- **Memory split**: peak RSS was **master 1.76 GB / worker 2.19 GB** — neither process
  holds the full ~3.5 GB model. (The worker is larger because Llama-3.2 ties the output
  projection to the embedding, so the last node carries that ~1.5 GB weight.)

### Bug found and fixed in the process

The first loopback run crashed the worker with
`rms_norm weight shape [0] ... input shape [13, 3072]`. Root cause: `LlamaLoadedWeights::load`
and `validate_dense_shapes` loaded/validated `output_norm` and the output projection on the
**first** node, but in this pipeline the **last** node computes the final norm + logits.
So 2-node pipeline parallel could never complete. Fixed by gating those on `is_last_node`
(the node owning the final transformer layer); single-node (no range) is both first and
last, so it is unaffected. See `src/inference.rs`.

## Honest framing

- A single stream through a pipeline has **one request in flight**, so distributing a model
  that already fits on one machine gives **no decode speedup** — the nodes run sequentially.
  The win is **fit**: running a model too big for one Mac, with each node holding a fraction.
- So the demonstrable advantage needs a model that does **not** fit comfortably on one node
  (e.g. an 8B Q8 ≈ 8.5 GB across two 16 GB Macs ≈ ~4.3 GB/node), not the 3B used for the
  correctness check above.
- This is a *different shape* than single-node MLX, not a "faster than MLX" claim.

## Reproduce the loopback validation

```bash
bash tools/bench/distributed/loopback-verify.sh <model>.gguf
```

## Two-Mac runbook (Thunderbolt 4)

Thunderbolt 4 exposes a **Thunderbolt Bridge** network interface (~20–40 Gbps); the
activation packets per token are small, so the link is not the bottleneck.

1. **Connect** both Macs with a TB4 cable. Each gets a Thunderbolt Bridge interface.
2. **Assign IPs** on the Thunderbolt Bridge (System Settings → Network → Thunderbolt Bridge →
   Details → TCP/IP → Configure manually) — pick two addresses on a private `/24` of your
   choice; call them `MAC_A_IP` and `MAC_B_IP`. Verify with `ifconfig bridge0` and
   `ping "$MAC_B_IP"`.
3. **Stage** the `camelid` binary and the GGUF model on both Macs. (Build on each Mac, or
   copy a binary built for the same Apple-Silicon generation — note the i8mm prefill path is
   opt-in and M2+ only; default decode uses dotprod, present on all Apple Silicon.)
4. **Balance the split by memory.** For tied-embedding models the last node carries the big
   output projection, so give it fewer transformer layers. For an 8B-class model across two
   nodes, start near a 60/40 layer split favoring the first node.
5. **Run** (example, 3B, 28 layers — adjust ranges/model for an 8B):

   On **Mac-B** (worker, last node):
   ```bash
   camelid distribute-worker <model>.gguf \
     --addr "$MAC_B_IP:5005" --layers 14..28 --master-addr "$MAC_A_IP:5006"
   ```
   On **Mac-A** (master, first node):
   ```bash
   camelid distribute-master <model>.gguf \
     --worker-addr "$MAC_B_IP:5005" --layers 0..14 --addr "$MAC_A_IP:5006" \
     --prompt "Explain what a Rust borrow checker does." --max-tokens 64
   ```
6. **Network overhead**: `camelid bench-network` (coordinator/worker) measures per-hop RTT and
   transfer size to confirm the TB link is not limiting.

### What to measure for an honest distributed win

- A model that **does not fit** on one 16 GB Mac (e.g. 8B Q8) running comfortably across two,
  with per-node peak RSS well under 16 GB and usable TTFT / decode.
- Compare against single-node MLX on the same model on one Mac (which must swap or cannot load
  it) — that is the defensible, distinct claim: *Camelid runs locally what one machine can't.*

## Two-Mac result (real hardware, 2026-06-03)

Two **Mac mini M4 (16 GB)** connected over a **Thunderbolt bridge** (~1 ms RTT, all traffic
on `169.254.x`, verified to egress the TB interface — no Wi-Fi). Model:
**Llama-2-13B-chat Q8_0** (13 GB, 40 layers, llama/SPM), split `0..20` (master) / `20..40`
(worker). Driven by `tools/bench/distributed/two-mac-run.sh`, decode path
`CAMELID_MAC_Q8_REPACK=1`.

**The win — a single 16 GB mini cannot run this model at all:**

| | Llama-2-13B **Q8** (13 GB) on one 16 GB mini | Two minis (Camelid, Thunderbolt) |
| --- | --- | --- |
| llama.cpp (Metal) | ❌ **fails to load** | — |
| MLX | ❌ only ships this 13B at 4-bit (same ceiling) | — |
| **Camelid** | — | ✅ **runs at full Q8** |
| Peak RAM / node | n/a (fails) | **master 7.23 GB · worker 7.33 GB** |
| Decode | 0 (can't) | **1.48 tok/s** |

llama.cpp on one mini reports `recommendedMaxWorkingSetSize = 12713 MB` and then
`test_prompt: failed to decode prompt batch, res = -3` — Metal's working-set ceiling on a
16 GB mini (~12.7 GB) is below the 13 GB model. MLX hits the same unified-memory ceiling,
which is why mlx-community publishes this 13B only at 4-bit.

Camelid split the model across the two minis (≈7.2–7.3 GB resident per node — **neither node
holds the full 13 GB**) and produced correct, coherent output over Thunderbolt:

> "The sky appears blue because of a phenomenon called Rayleigh scattering, in which shorter
> (blue) wavelengths of light are scattered more than longer (red) wavelengths by the tiny
> molecules of gases in the atmosphere…"

**The honest claim:** *Camelid runs, at full Q8 precision across two commodity 16 GB Macs, a
model that a single mini (and therefore single-node MLX/llama.cpp) physically cannot load.*

Decode (1.48 tok/s) is latency-bound: the pipeline runs one request in flight, so master and
worker execute serially per token over two TB hops. That is a throughput lever (microbatching
to overlap the stages, or the GPU-resident forward pass), not a correctness limit — the
capability (running the otherwise-unrunnable model) is the result.

### Reproduce

```bash
MODEL=<13B-Q8>.gguf REMOTE_MODEL=<path-on-worker>.gguf \
CAMELID_BIN=<local camelid> REMOTE_BIN=/tmp/camelid WORKER_HOST=<ssh host> \
MASTER_TB_IP=<this TB ip> WORKER_TB_IP=<worker TB ip> TOTAL_LAYERS=40 SPLIT=20 \
MAX_TOKENS=64 bash tools/bench/distributed/two-mac-run.sh
```

## Status

- Loopback 2-node pipeline parallel: **correct** (matches single-node) and **memory-split
  confirmed**, after the last-node weight-loading fix.
- Real two-Mac TB4 run: **done** — Llama-2-13B Q8 runs across two 16 GB minis (7.2 GB/node)
  that a single mini cannot load. Decode 1.48 tok/s (serial pipeline; throughput optimization
  is the next lever).
