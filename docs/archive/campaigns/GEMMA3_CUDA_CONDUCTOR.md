# gemma3 → CUDA: the Windows GPU lane for the Gemma lineup

Companion to `GEMMA3_METAL_CONDUCTOR.md`. That campaign (PR #560) put the
`gemma-3-1b-it-Q8_0` row on the macOS Metal GPU-resident lane. This one puts the
Gemma lineup on the GPU on **Windows/CUDA**, where every gemma row still decodes
on the CPU.

Scope pinned at Phase 0: the `gemma_3_1b_it_q8_0` row on a CUDA-resident lane,
plus the gemma4 E2B/E4B rows off their CUDA env opt-in. Evidence bar for this
campaign is the **in-tree CPU runnable oracle**, not the external llama.cpp
bundle — the external capture is a named later phase and nothing here claims it.

---

## §1 The gap, as the tree actually reports it

Measured by reading the gates, not by inference. Every one of them names the
same three missing pieces.

| Row | Windows GPU today | The gate that stops it |
|---|---|---|
| `gemma-3-1b-it` Q8_0 | ✗ CPU runnable bridge | `windowed_arch_resident_host_available()` is `#[cfg(not(target_os = "macos"))] → false` (`src/inference.rs`), and the engine-level backstop bails on `windowed_arch && resident_decode_cuda_enabled()` |
| `gemma-4-E2B-it` / `E4B-it` Q8_0, Q4_0 | ~ CPU by default | `gemma4_cuda_enabled()` requires `CAMELID_GEMMA4_CUDA=1` (`src/api/mod.rs`); unset ⇒ the serve lane loads `Gemma4Runtime` (CPU) |
| `gemma-4-12b`, `gemma-4-26B-A4B` | ✗ | supported lane is two-Mac distributed layer sharding; 13.4 GB row against a 6 GB card |
| `gemma-2-9b-it` Q8_0 | ✗ | no gemma2 runtime exists; `planned_exact_row_candidate` |

The two rows this campaign can actually move on this host are gemma3 1B and the
gemma4 E2B/E4B pair. The 12B/26B rows and gemma2 are **out of scope and stay
unclaimed** — recorded here so the omission is deliberate rather than silent.

## §2 What CUDA already has (and why this is not the Metal campaign again)

The Metal campaign had to build gemma3's forward from nothing. CUDA does not
start there: the gemma4 CUDA-resident lane (`Gemma4CudaResident`) already forced
every gemma primitive onto the GPU, and several of them landed in the **generic**
resident engine's kernel set rather than gemma4's private one.

Already present, already loaded by `CudaResidentKernels::new`:

| Requirement | Status on CUDA |
|---|---|
| Sliding-window decode attention | `attention_decode_sw` — compiled, and bound to the generic `attention_sw` field. Never launched by the generic forward. |
| GeGLU (`gelu_tanh(gate) * up`) | `geglu_mul` — compiled, same story |
| QK-norm | `q_norm` / `k_norm` already on the generic `ResidentLayer` (qwen3 uses them) |
| Sandwich post-norms | bound on the shared weights by #560 Phase 1 (`post_attention_norm`, `post_ffn_norm`); never uploaded to a CUDA layer |
| Per-layer dual-theta RoPE | proven in `Gemma4CudaResident` (`d_cos_all`/`d_sin_all`, per-layer slot); the generic engine builds **one** table set |
| Token-by-token prefill for windowed archs | already arch-keyed and backend-independent — `session_prefill_chunk_tokens` returns 1 for any windowed arch |

So the CUDA work is wiring and per-layer variation, not new numerics.

## §3 The five things the generic CUDA engine is missing

1. **Sandwich post-norms** on `ResidentLayer` + upload at `set_layer_located`.
2. **Per-layer schedule** — `use_alt_rope: Vec<bool>` + `window: Vec<Option<usize>>`,
   the exact shape of `metal::ResidentLayerSchedule`, plus ALT (local-theta) rope
   tables alongside the primary.
3. **Per-layer dispatch in `forward_pass_inner`** — select primary vs ALT cos/sin,
   and launch `attention_sw` with the layer's window instead of `attention`.
4. **GeGLU selection** in the FFN (currently `silu_mul` unconditionally).
5. **`sqrt(d_model)` embed scale** before layer 0.

### §3.1 Design note — the fused residual is the one real obstacle

The CUDA engine fuses the post-projection residual add into the O and down GEMVs
(`launch_gemv_residual`, `output[row] += acc`). gemma3's sandwich norms sit
*between* the projection and the residual:

```
h = h + post_attention_norm(o_proj(attn))
h = h + post_ffw_norm(down_proj(geglu(...)))
```

so the fused GEMV cannot be used on a gemma3 layer. The unfused path already
exists (plain GEMV then `launch_residual` / `residual_add`), so the fix is to
select it for windowed archs and insert an `rms_norm` between the two — no new
kernel. Metal hit the same shape; this records why the CUDA layer loop has an
arch-keyed branch where Llama has a fusion.

### §3.2 Design note — split-K never applies to a sliding layer

`SPLITK_THRESHOLD` is 512 and the gemma3 1B row's `attention.sliding_window` is
also **512**. A sliding layer therefore never attends more than 512 keys and can
never cross the split-K threshold, so sliding layers take the plain windowed
kernel unconditionally and global layers keep the existing split-K logic
untouched. This is a correctness *and* a performance non-event, not a tradeoff —
worth stating because "we disabled split-K for gemma3" would otherwise read as a
concession.

The split-K and flash-prefill kernels have no window parameter at all. They are
not extended in this campaign; they are declined for sliding layers.

## §4 Invariants inherited from #560 that must NOT regress

These are load-bearing and were earned by two adversarial review passes on the
Metal campaign. The CUDA flip must preserve every one:

- **The CPU-dense fail-closed stays a choke point.** `forward_layer_timed` and
  `forward_prefill_layer_chunk_timed` are the two points every CPU dense layer
  walk passes through, keyed on parsed metadata (`arch_has_windowed_attention`),
  not the arch string. Widening the resident routing must not tempt anyone back
  into enumerating entry points.
- **Routing is capability- AND quant-aware, never a bare arch list.** The Q8_0
  admission pin (`windowed_arch_resident_quant_admissible`) stays: a Q4_K_M
  gemma3 has no windowed parity receipt on any lane and must not ride a K-quant
  admission onto the GPU.
- **A plan OUTPUT can never be a routing INPUT** (DECISIONS D20). The CUDA host
  probe must consult only inputs, and where the resident selection cannot fire
  the plan fails closed to the safe plan with a windowed-arch reason.
- **Serve and the CLI direct lanes consult the same predicate**, so they cannot
  disagree about which lane a file gets.
- **Tools stay a typed 422** and speculative decode stays declined for windowed
  archs.

## §5 Phases

| Phase | Content | State |
|---|---|---|
| 0 | Windows CPU baseline for gemma3 + gemma4; capture the CPU runnable oracle the internal gate compares against | **done — §7** |
| 1 | The five items in §3 on the generic CUDA resident engine | **done** |
| 2 | Routing flip: admit a CUDA-resident host, drop the engine bail, preserve every §4 invariant | **done — §8** |
| 3 | gemma4 CUDA default-on behind a VRAM fit guard | **done — `qa/gemma3-cuda/phase3/RESULT.md`** |
| 4 | Parity gate **including the >512 windowed pack** — the leg that proves the mask is real | **done — `qa/gemma3-cuda/phase4/RESULT.md`** |

## §8 Outcome

**gemma3 1B Q8_0 is now a GPU row on Windows.** GPU-resident (1963 MiB, 26/26
layers, KV cap 32768), and **9/9 token-and-text identical to the pinned oracle
above the sliding window** at 606 / 1205 / 2403 prompt tokens — the pack that
distinguishes a live mask from a dead one. Below the window it is 10/15, with the
five divergent legs enumerated and attributed to lane numerics on the evidence of
a per-layer hidden-state trace (no step change at any layer; worst relative L2
1.89%). That pack does not pass and the row is not promoted on it.

The evidence bar landed ABOVE what Phase 0 scoped. The plan was an internal gate
against the CPU runnable oracle; #560's committed bundle turned out to carry the
**pinned llama.cpp captures themselves**, so the comparison ran against the
external oracle's recorded tokens at internal-gate cost. Two limits recorded
rather than glossed: a replayed capture cannot be re-probed from the oracle's
side, and it was taken on a Mac.

Two defects were found and fixed inside this campaign's own work — a primary
rope table left on the generic frequency form and the wrong pairing, and a
prefill path that would have cached a non-windowed engine for decode to reuse.
Both are described in the commit; both were the kind that yield fluent, wrong
text rather than an error.

**gemma4 E2B Q8_0 now runs GPU-resident on this 6 GB card** — 2645 MiB, `Paris`,
798 ms against 4,076 ms on the CPU runtime. E4B Q8_0 declines on fit (short by
621 MiB including the headroom floor; it would take the GPU on an 8 GB card) and
Q4_0 was initially declined on the quant gate because the lane **mis-decoded** it — a
pre-existing defect that flipping the default surfaced. That defect has since been
root-caused and fixed (Phase 5: the tied head uploaded raw GGUF wire into kernels
that index a repacked layout), and E2B Q4_0 now runs GPU-resident at 1557 MiB with
a 5/5 token-identical greedy-parity capture against the CPU runtime.

An earlier revision of this section said no gemma4 row reached the GPU here. That
was wrong, and wrong because of a defect in this campaign's own fit guard: it
summed the whole file's tensor bytes, projecting 5055 MiB for a row that actually
uses 2635 MiB, because gemma4's PLE per-layer embedding tables are larger than its
projections and never go to VRAM. The guard now projects from the layer
projections plus a calibrated overhead, and is advisory — the load site falls back
to the CPU runtime on error, which it previously did NOT do (a genuine OOM would
have 503'd a row the CPU was serving fine). Over-conservative fit checks are not
safe; they silently keep working hardware on the CPU, on exactly the mid-size
cards nobody tests.

Phase 3 also delivered a single admission predicate shared by the plan and the
load site, closing the Phase 0 disclosure defect.

### Test state

`cargo test --lib`: **1368 passed, 2 failed**. Both failures reproduce on
unmodified `origin/main` on this host (they assume CUDA-resident decode is
inactive) and are unrelated to this work. Two tests that DID break here were
mine and are fixed: they constructed a "fallback host" by disabling Metal alone,
which stopped meaning "no resident gemma3 lane" the moment CUDA gained one.
`cargo clippy --all-targets --all-features -D warnings` and `cargo fmt` are
clean.

## §6 What this campaign will not claim

Written before the work, so the scope pin is not retrofitted:

- **No external-oracle (llama.cpp) parity claim.** The evidence bar here is the
  in-tree CPU runnable oracle. The external bundle is a later phase.
- **No throughput or speed number** until a measurement phase runs. Whatever the
  lane does on this card is an observation, not a shipped figure.
- **No claim for any gemma3 row or quant other than 1B Q8_0.**
- **No claim for gemma-4-12b, gemma-4-26B-A4B, or gemma2.**
- **No context claim beyond what the packs actually cover.**

---

## §7 Phase 0 results — the premise, measured

Receipts: `qa/gemma3-cuda/phase0/`. Both arms ran with **no env overrides**, one
engine resident at a time, on `origin/main` @ `05984826`.

### §7.1 gemma3 1B Q8_0 — refused by routing, not by hardware

The interesting part is that CUDA is not merely available, it is *engaged*: the
process selects the device, prints `VRAM 5122 MiB free / 6143 MiB total`, and
even runs `Warming up the GPU (building the resident engine, one-time)…` — and
then routing hands the row to the CPU anyway.

```
cuda_resident_active : true
selected_backend     : cpu_reference
prefill_path         : safe_cpu_prefill
decode_path          : safe_cpu_decode
support_level        : supported_exact_row_smoke_sub512
served lane          : runnable      ("id":"chatcmpl-runnable", "lane":"runnable")
reason               : windowed-attention row (gemma3): the only validated dense lane is
                       Metal-resident; no CPU dense plan exists for this arch — serve chats
                       via the runnable bridge on this host; failing closed to safe path
```

Cadence observation: **52.1 s** to answer "Name the capital of France in one
word." — 18 prompt tokens, 2 generated tokens, greedy. That is the user-visible
shape of the problem. Recorded as an observation, not a benchmark.

Note also `support_level: supported_exact_row_smoke_sub512`. On Windows the row
is scoped *below* the 512-token window, because #560's >512 claim explicitly does
not travel off Metal. Lifting that on Windows is downstream of Phase 4, not of
Phase 1.

### §7.2 gemma4 E2B Q4_0 — a plan/routing disagreement, pre-existing on main

Expected: gemma4 decodes on the CPU by default because `gemma4_cuda_enabled()`
requires `CAMELID_GEMMA4_CUDA=1`. That held — 1 token in **2.90 s**.

Unexpected, and worth fixing rather than merely flipping past: the execution plan
**advertises a GPU lane that serve is not running**.

```
execution_plan.selected_backend : cuda_resident_kquant_runtime
execution_plan.decode_path      : kquant_cuda_resident_decode
actual backend                  : gemma4-runtime   (Gemma4ServeRuntime::Local, CPU)
VRAM in use while generating    : 107 MiB
```

107 MiB of VRAM while a 2.83 GB model generates. A resident lane cannot decode
from 107 MiB. The cause is structural: the generic execution plan is computed
from platform + quant and names the CUDA K-quant resident lane whenever CUDA is
active, while `resolve_gemma4_runtime` independently picks the CPU runtime unless
the env opt-in is set. The two never consult each other.

This is the same defect class as **DECISIONS D20** from the Metal campaign — a
plan disclosing a lane that serve is not on. Phase 3 touches exactly this gate,
so the fix there must make the plan and the routing *agree*, not just change
which default they each independently assume. Recorded now so the fix is
attributed to a measured finding rather than retrofitted.

Scope of the finding: observed for gemma4 on Windows/CUDA. Not investigated for
other families or hosts in this phase.

## Host

RTX 3060 Laptop GPU, 6144 MiB, driver 576.83. Windows 11. CUDA builds by default
on this target (`build.rs` sets the cfg; no `--features cuda` needed).

Row under test: `models/gemma-3-1b-it-Q8_0.gguf`, sha256
`b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5` — matches the
ledger identity for `gemma_3_1b_it_q8_0` exactly.
