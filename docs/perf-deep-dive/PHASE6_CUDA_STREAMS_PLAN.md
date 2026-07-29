# STAMPEDE Phase 6 — CUDA multi-stream overlap: implementation plan (turnkey)

Recon + architect design captured 2026-07-09. Anchors are src/cuda_resident.rs at main 41d47433; re-verify line numbers before editing. Flag: CAMELID_CUDA_STREAMS (default off). Live path only (do not capture across streams — cudarc/WDDM). Biggest risk: cudarc multi-stream event-tracking tax + WDDM sub-100us scheduling may make this a documented NULL like graphs — measure first, then disable_event_tracking, then honest NULL receipt.

---

# STAMPEDE Phase 6 — Implementation Plan: Multi-Stream Overlap for CUDA Resident Decode

All anchors: `src/cuda_resident.rs` at main 41d47433 unless noted.

## 1. Scope and non-goals

Overlap the four independent GEMV chains per `LayerKind::Full` layer inside `forward_pass` (5433–6336). Every kernel is launched unchanged (same grid, same reduction order) — only the stream it is enqueued on changes, with event-joins before every dependent read. Bitwise-neutral by construction (SPEED_FIX_PLAN.md:101–107 constraint: no re-association).

**In scope:** the Full-layer attention block (Q/K/V) and FFN block (gate/up), all quant lanes (dispatch_gemv at 3204–3300 already takes an explicit `&Arc<CudaStream>`, as does every `launch_*` helper — the split is mechanical). Applies to `forward_token` (6341), `forward_token_device` (6431), `forward_token_logits`/`_sample`, and `prefill` (6686) since all call `forward_pass`.

**Out of scope (v1):** SSM layers (5890–6070; qwen35's 4 independent gemvs are a documented follow-up), the batched paths (`verify_batch` 6722, `run_batched_layer_stack` 6876, tree 7193/7495, `prefill_batched` 7666 — they don't go through `forward_pass` and 7711 relies on same-stream ordering), and the graph-capture path (see §5).

## 2. Exact launch sites to split

Per layer, Full path:

| Site (line) | Today | Phase 6 stream |
|---|---|---|
| `launch_rmsnorm_quantize` 5563 / `_q8k` 5597 (and unfused 5575/5585) | main `s` | main — then **record `ev_act`** |
| Q gemv 5642 (qwen35 fused qgate 5616 + deinterleave 5631) | main | **main** (largest output; keeps the critical path warm) |
| Q per-head norm 5692 → rope-Q 5715 | main | main |
| K gemv 5658 | main | **side_a**: wait `ev_act` → K gemv → k-norm 5702 → rope-K 5727 → `kv_scatter` K 5740 → **record `ev_k`** |
| V gemv 5673 | main | **side_b**: wait `ev_act` → V gemv → `kv_scatter` V 5751 (V has no norm/rope) → **record `ev_v`** |
| attention 5767 (split-K) / 5788 (single) | main | main: **wait `ev_k`, wait `ev_v`** → attention → O-proj chain 5807–5887 unchanged on main |
| `launch_rmsnorm_quantize` (FFN) 6079 / `_q8k` 6113 | main | main — **record `ev_ffn`** |
| gate gemv 6125 | main | **main** |
| up gemv 6140 | main | **side_a** (reused): wait `ev_ffn` → up gemv → **record `ev_up`** |
| `launch_silu_mul_quantize` 6158 (/6169/6222) | main | main: **wait `ev_up`** → silu → down 6189/6232 unchanged |

Event-join graph (per layer, 4 events reused every layer):

```
main:   norm+quant ──rec ev_act──► Qgemv → qnorm → ropeQ ──wait ev_k, ev_v──► attention → Oproj+res
side_a:            wait ev_act ──► Kgemv → knorm → ropeK → scatterK ──rec ev_k
side_b:            wait ev_act ──► Vgemv → scatterV ──rec ev_v
main:   ffn norm+quant ──rec ev_ffn──► gate gemv ──wait ev_up──► silu → down+res
side_a:            wait ev_ffn ──► up gemv ──rec ev_up
```

**Hazard coverage (why these four events suffice):**
- `d_position` (uploaded on main at 5471 or 6441) and `d_q8k_*` are ordered before `ev_act`, so side streams see them.
- The shared-scratch WAR hazard (`d_in_quants`/`d_in_scales` reused at every quantize point — the hazard the single stream currently hides): the O-proj quantize at 5819 (write) is enqueued on main *after* attention, which waited on `ev_k`/`ev_v` recorded *after* the side gemvs' reads — WAR protected transitively. Same for silu's write of `d_in_*` vs. up-gemv's read: main waits `ev_up` first. No extra buffers needed.
- Offload interaction: an offloaded layer's `wk`/`wv`/`wup` scratch reads on side streams are ordered after `copy_done[cur_buf]` (main waits at 5512 → `ev_act` transitive), and complete before main's `compute_done[cur_buf].record` at 6256 (transitively via `ev_k`/`ev_v`/`ev_up`). Safe, but v1 should still **disable overlap when `self.offload.is_some()`** — a third concurrent stream contends with the copy engine on this box and muddies the measurement; lift later with a receipt.

## 3. Stream/event lifecycle

- New struct near `OffloadState` (4415): `struct StreamOverlap { side_a: Arc<CudaStream>, side_b: Arc<CudaStream>, ev_act: CudaEvent, ev_k: CudaEvent, ev_v: CudaEvent, ev_ffn: CudaEvent, ev_up: CudaEvent }`, field `overlap: Option<StreamOverlap>` on `CudaResidentDecode` (next to `decode_graph` at 4632).
- Create **once**, in `CudaResidentDecode::new`, via `ctx.new_stream()` + `ctx.new_event(None)` — same pattern as `enable_offload_scratch` (4984–5006) — **but only when the flag is on**. This is load-bearing: creating a second stream flips cudarc 0.19.7 into multi-stream mode with per-slice-arg auto event tracking on every launch (gemma4-cuda-port gotcha; gemma4_runtime.rs:2737). Lazy/conditional creation keeps the default (flag-off) path in single-stream mode with literally zero new code executed.
- Events are recorded/re-recorded every layer; `cuEventRecord` overwrites, and host program order (record enqueued before the wait that consumes it, every layer) makes reuse across layers and tokens correct. No per-token create/destroy.
- Cost: ~8 extra tiny enqueues × ~28 layers ≈ 224/token vs. ~600 launches already enqueue-ahead — noise.
- In `forward_pass`, resolve once at the top (after 5455): `let ov = self.overlap.as_ref().filter(|_| !graph_capture && self.offload.is_none());` and thread `ov` through the two blocks. When `None`, every launch stays on `s` exactly as today (byte-identical control flow).

## 4. Flag

`CAMELID_CUDA_STREAMS` — helper `cuda_streams_enabled()` placed next to `cuda_graphs_enabled()` (4707–4712), same `matches!(Some("1"|"true"|"on"|"yes"))` idiom, **default OFF**. On a GO gate, flip by inverting to the `resident_fusion_enabled()` opt-out style (4719–4724, e.g. `CAMELID_CUDA_NO_STREAMS`) in the promotion PR, consistent with the Phase 3 default-flip pattern.

## 5. Graph-capture decision: live path only

**Decision: do not capture across streams.** Rationale from in-tree evidence:
- cudarc 0.19.7's graph API is minimal (begin/end capture + launch/upload; no `cuStreamGetCaptureInfo`, no capture-dependency editing). The multi-stream-join-during-capture idiom is mechanically possible via raw `event.record`/`stream.wait`, but nothing in-tree does it, both existing capture users are single-stream by design ("all engine work runs on THIS one stream", 2604–2609), and it additionally requires `disable_event_tracking` before any slice alloc (gemma4_runtime.rs:2737 gotcha) — all on a driver (WDDM 576.83/CUDA 12.9) where even single-stream llama-lane capture is **currently broken** (CAPTURE_ISOLATION in the layer loop; STATUS comment 6357–6370).
- So: gate overlap with `!graph_capture` (the `ov` filter in §3). The captured graph, when someone opts into `CAMELID_CUDA_GRAPHS=1`, keeps today's serial single-stream recording — precedent is split-K attention, live-only above SPLITK_THRESHOLD at 5766.
- **Fraction of decode running live: 100% by default.** CAMELID_CUDA_GRAPHS is default-off everywhere (4699–4712, measured no-win: 3B 53.2→52.5), and capture is broken on this box regardless. Unlike split-K, this live/graph divergence carries **zero parity risk**: overlap is bitwise-neutral per kernel, so `splitk_verify_active()` (3982) needs no analogue — verify/graph/live all produce identical bits. State this in the PR.

## 6. Validation (token-identical)

1. Existing device tests: `cargo test --release --all-features` cuda_resident suite (13/13 + ignored device tests run locally). Clippy `--all-targets --all-features -D warnings` + fmt before push.
2. New `scripts/validate-cuda-streams.sh`, cloned from `scripts/validate-cuda-prefill-row.sh` (the established env-flag A/B pattern): sequential server restarts — **never two engines resident at once** (bench-memory-safety hard rule), free-RAM check (model+3GB) before each leg, kill orphans by PID after each leg. Byte-identical greedy diff, OFF vs ON, on:
   - Llama-3.2-3B Q8_0 GPU-resident (the gate model), low-ctx prompts + the long multi-chunk depth prompt from `scripts/qwen3-cuda-prefill-ab.mjs` (~1881 tok);
   - Qwen3-4B Q8_0 (second arch, QK-norm path exercises k-norm on side_a);
   - Qwen3-4B Q4_K_M (K-quant gemv lanes + q8k activation scratch);
   - one leg with `CAMELID_RESIDENT_NO_FUSION=1` (unfused chain);
   - one leg via the device-side loop entry (`forward_token_device`) since it also flows through `forward_pass`.
3. Engaged-check: assert via `CAMELID_RESIDENT_TRACE`/a one-line trace that overlap actually constructed side streams in the ON leg (fake-null sweep trap from Phase 3 — receipts must prove the lever was engaged).

## 7. Measurement (RTX 3060 Laptop 6GB, single-engine)

- Harness: `scripts/bench-qwen3-cuda-resident.mjs` (GPU-active-asserted, two-point decode `(N-1)/(t(N)-t(1))`, greedy temp-0, median-of-5, JSON receipt). One engine at a time; VRAM check that the row is fully resident (3B Q8 and 4B Q8 low-ctx both fit; respect the 2090-pos resident cap for 4B Q8 depth legs).
- Legs: OFF vs ON × {low ctx, depth (~1881)} × {Llama-3B Q8, Qwen3-4B Q8}. Baselines on file: 3B ~53 tok/s, 4B 41.6 low / 26.2 depth (PERF_GAP_REPORT.md:73–74).
- **Gate (STAMPEDE_CONDUCTOR.md:144): ≥ +8% decode at low ctx AND no depth regression → GO.** Depth is expected roughly flat (attention, not gemv, dominates there — that residual is the parity-locked uncoalesced K/V).
- Optional mechanism receipt: ncu with `--clock-control none` (this box's requirement) or nsys timeline showing concurrent gemvs / achieved DRAM% rising from 62% toward llama.cpp's 85%.
- Receipts into `qa/perf/` (git add -f if `.log`), scrub check (`check-public-scrub.sh`, no `<home>` paths) before push; never `git add -A`.

## 8. Biggest risk

**cudarc's multi-stream mode tax.** Instantiating the side streams flips the shared context into multi-stream mode, where cudarc 0.19.7 auto-records and drops a `CudaEvent` per slice argument per launch — across ~600 launches/token × ~7 args, that host-side overhead can plausibly eat the entire +8% (decode is only marginally enqueue-ahead already). Mitigation ladder: (a) measure as-is first; (b) if the ON leg regresses at identical tokens, apply `unsafe ctx.disable_event_tracking()` behind the same flag — precedent gemma4_runtime.rs:2737, but note it is a process-wide bookkeeping toggle (docs/recon/ENGINE_INVERSION_R1_LANE_RECON.md:28–30) and with tracking off we own *all* cross-stream ordering: the §2 event graph plus a pre-decode drain covering load-time default-stream memsets (the gemma4 cos-table-zeroed race); (c) if still null, record an honest NULL receipt — the fallback theory is that WDDM's software scheduler may simply not co-schedule sub-100µs kernels on this driver, which would make Phase 6 a Linux/TCC win and a documented no-op here, like graphs.

Secondary risk, stated for completeness: none from parity (kernels unchanged, disjoint outputs, no atomics across the joined buffers) — the risk is purely performance-null, not correctness.

## Files touched

- `src/cuda_resident.rs` — `StreamOverlap` struct + field, `cuda_streams_enabled()`, `forward_pass` stream threading (attn block 5563–5804, FFN block 6079–6240).
- `scripts/validate-cuda-streams.sh` (new, cloned from `validate-cuda-prefill-row.sh`).
- `docs/perf-deep-dive/STAMPEDE_CONDUCTOR.md` — Phase 6 ledger row (line 167) + graph-vs-live decision receipt.
- `qa/perf/` — bench + parity receipts.