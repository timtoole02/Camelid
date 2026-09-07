# Model Fit Advisor — End-to-End Plan

**Status:** Draft / docs-first. No runtime code written yet.
**Branch:** `feat/model-fit-advisor`
**Axis:** *Capacity/fit* — strictly separate from the *support/correctness* axis
(`COMPATIBILITY.md`) and the *runnable-lane* axis (`src/runnable/`). A "fits your
hardware" verdict is **never** a support or parity claim.

---

## 1. Problem

Camelid honestly reports which model rows are *validated* (`/api/capabilities` →
`model_compatibility`) and which `(architecture, quant)` combos are *runnable*
(`src/runnable/smoke.rs::oracle_qualified`). It does **not** tell a user whether
*their machine* can actually load and run a given model. Today the only fit
feedback is reactive and late:

- **CUDA:** `src/cuda_vram.rs::evaluate` returns a `VramShortfall` — but only
  **mid-load**, after the user has already committed and (usually) downloaded.
- **CPU:** `src/inference/kv_cache.rs::ensure_position_capacity` aborts with
  `BackendError::KvCacheBudgetExceeded` — but only **during generation**.
- **Catalog browser:** `frontend/src/components/models/CatalogLaneBrowse.jsx`
  predicts each row's *support/runnable* lane before download, but explicitly
  **ignores local hardware** (line ~67: *"we don't gate the download on local
  hardware"*).

So a user on an 8 GB laptop can repeatedly pick an 8B Q8_0 model, wait for a
multi-GB download, and only then hit an OOM/abort — with no guidance toward a
model that *would* fit (a smaller size, a K-quant, or GPU-offload).

## 2. Goal

Add a **capacity/fit advisory layer** that, given the already-detected
`HardwareProfile`, annotates each catalog row with a *fit verdict* **before** the
user commits, plus a lightweight "best for" tag (coding / reasoning / general /
tools). Surface it in the model browser and as a fail-fast guard at the load
boundary, so the CLI, WebUI, and desktop app all inherit it from one place.

## 3. Verified inventory — what already exists (reuse, do not rebuild)

Every item below was read in-tree on branch `feat/model-fit-advisor` at
`origin/main` = `2b8b97c4`.

| Concern | Location | What it gives us | Reused for |
| --- | --- | --- | --- |
| Host hardware probe | `src/capability.rs` → `HardwareProfile::detect()` | `cuda_available`, `cuda_vram_total_bytes`, `cuda_vram_free_bytes`, `cuda_compute_capability`, `cuda_tensor_cores`, `cpu_logical_cores`, `host_ram_total_bytes`, `host_ram_free_bytes`, `simd`. RAM probed on Windows (`GlobalMemoryStatusEx`), Linux (`/proc/meminfo`), and macOS (`hw.memsize` + Mach VM statistics); returns `(0, 0)` = "unknown" on platforms without a probe | The complete input to the fit function |
| VRAM headroom policy | `src/cuda_vram.rs` → `evaluate(free_bytes, alloc_bytes, min_headroom_mib) -> Result<VramPlan, VramShortfall>` | Pure, GPU-free, unit-tested arithmetic; default headroom `512 MiB` (env `CAMELID_MIN_VRAM_HEADROOM_MIB`) | Direct call for the VRAM branch of the verdict |
| RAM/KV budget | `src/inference/kv_cache.rs::ensure_position_capacity` + `kv_bytes_per_token()` | KV projection; budget = `CAMELID_MAX_KV_CACHE_BYTES` else `max(80% avail, 25% total)` RAM | Reference for the KV term + the hard safety net that stays in place |
| Curated catalog | `src/api/mod.rs` → `curated_catalog() -> Vec<CatalogItem>` (~16 rows) | `CatalogItem { catalog_id, name, repo_id, filename, size_bytes, downloads, likes, quant, architecture, license }` | The rows to annotate; `size_bytes` **is** the on-disk weight footprint |
| Catalog view | `src/api/mod.rs` → `CatalogItemView::from_curated` | Adds `oracle_qualified`, `group`, `arch_detected` (the pre-download lane) | The struct we extend with a `fit` field |
| Lane predictor | `src/runnable/smoke.rs::oracle_qualified(architecture, quant)` | Bool for anchored combos: `llama/qwen3/gemma3/phi3 × Q8_0` | Pattern to mirror; the fit axis sits **beside** it, not inside it |
| CLI pull | `src/catalog.rs` → `run_pull` / `print_catalog` | Uses `curated_catalog()` | Where the CLI fit hint prints |
| Frontend browser | `frontend/src/components/models/CatalogLaneBrowse.jsx` (~366 lines) | Per-row lane chips; currently hardware-blind | Where the fit badge renders |
| Model metadata | `src/api/mod.rs:410` `n_params: Option<u64>` | Parameter count — **only available post-load** from GGUF `general.parameter_count`, **not** in the pre-download catalog | Informs the post-load exact path, not the pre-download estimate |

## 4. The gap (what is genuinely new)

1. A **fit-verdict type** and a **pure estimation function** combining
   `HardwareProfile` + a model's footprint.
2. **Per-row metadata the catalog lacks:** a canonical **parameter count** and
   **task tags** (`coding` / `reasoning` / `general` / `tools`). Task tags are
   curated, subjective, and must be labeled advisory — they are not measured.
3. A **surface** (API field + UI badge + CLI line + load-time guard).

## 5. Design principles (non-negotiable, from repo culture)

- **Capacity ≠ support.** A `Fits` verdict never implies parity/validation. Copy
  must say *"your hardware can load/run this"*, never *"this is supported"*.
- **Advisory, not authoritative.** The estimate is a heuristic. The existing
  `VramShortfall` (mid-load) and `KvCacheBudgetExceeded` (mid-gen) guards remain
  the hard safety net and are the source of truth.
- **Split refusals by permanence, not by severity.** A host that is *busy right
  now* and a host that is *too small* are different facts with different
  remedies. Only the second may refuse a load; the first warns and proceeds.
  §9 is the binding mapping.
- **Degrade to silence on unknowns.** When `host_ram_total_bytes == 0` or
  VRAM is unknown, the verdict is `Unknown` and the UI shows nothing scary — it
  **never blocks** a download on an unknown.
- **Pure and testable.** All math lives in a GPU-free, unit-tested function, in
  the `src/cuda_vram.rs::evaluate` style.

### 5.1 Platform note — the macOS probe, and how it turned an advisory into a gate

macOS was an *unprobed* host until `587d415e` "Add macOS desktop sidecar support
(#527)" (2026-07-28): `host_ram_bytes()` fell through to the `(0, 0)` arm, every
Mac reported `Unknown`, and the load guard could not fire there by construction.
That commit wired `crate::gait::host_ram_status()` into `HardwareProfile` and
silently converted every macOS host from never-blocked to blockable — without
revisiting whether the verdicts behind the gate were all permanent facts. They
were not, and the busy-host verdict then began refusing ordinary models on
8 GB machines. Recorded here because the mechanism generalizes: **giving a
platform a probe changes what every gate downstream of that probe does.**

The probe is also conservative in a way that matters. `macos_ram_stats`
(`src/gait/mod.rs`) computes `available` as `(free_count + inactive_count) ×
page_size` — free plus cold-but-resident pages only. It counts none of the other
memory the kernel can reclaim under pressure, and reads roughly **half** of what
`memory_pressure -Q` reports as free on the same host at the same moment. That
makes every consumer of the number pessimistic by roughly 2×.

Correcting the probe is **deliberately out of scope here** and belongs in its own
change: the same figure feeds four independent consumers with different failure
modes, and none of them are exercised by this document's tests —

- the KV auto-budget (`src/inference/kv_cache.rs::kv_cache_budget_from`),
- the resident-MoE expert preflight (`src/inference.rs::preflight_resident_moe_experts`),
- this advisor's `usable_host_ram_bytes`,
- and the content-addressed GAIT receipt's `HostSafety::ram_headroom_gib`, whose
  self-digest makes the number part of a receipt's identity.

Widening `available` would relax a safety budget and change a receipt's bytes in
one commit. Splitting the load guard by permanence (§9) fixes the user-visible
dead end without touching any of them.

## 6. Data model

### 6.1 Fit verdict (new)

```
enum FitVerdict {
    FitsResident,      // weights + KV fit in VRAM within headroom (GPU) OR in RAM (CPU host)
    FitsWithOffload,   // weights exceed VRAM but fit VRAM+host-RAM offload split (CUDA lane)
    CpuOnlyOk,         // no usable GPU, but fits host RAM
    InsufficientFreeMemory, // an idle host would hold it; this one is busy — transient
    WontFit,           // exceeds every available budget
    Unknown,           // hardware unknown or unavailable — advisory-blind
}
```

### 6.2 Estimation inputs

- **Weight bytes:** `CatalogItem.size_bytes` (already exact for curated rows — it
  is the GGUF file size). No estimation needed for the pre-download weight term.
- **KV bytes:** needs `n_layers`, `n_kv_heads`, `head_dim`, `context`, `kv_dtype`
  — **not in the catalog**. Options, decided in Slice 1:
  - (a) Add a small `kv_bytes_per_1k_ctx: u64` (or the raw dims) to `CatalogItem`
    for curated rows only, OR
  - (b) Use a conservative per-architecture heuristic bound and clearly mark the
    KV term approximate. The precise KV path already exists post-load
    (`kv_cache.rs`), so pre-download only needs a *safe* bound.
- **Headroom:** reuse `cuda_vram::min_headroom_mib()` for VRAM; reuse the
  `kv_cache` RAM-budget policy shape for the CPU branch.

### 6.3 Catalog additions (curated only)

```
// added to CatalogItem
params: u64,                 // canonical parameter count (e.g. 3_210_000_000)
task_tags: &'static [&'static str],   // advisory: "coding" | "reasoning" | "general" | "tools"
```

Experimental (live HF) rows keep `arch_detected: false` and therefore report
`FitVerdict::Unknown` for the KV-dependent portion — a filename guess can never
anchor a fit verdict, exactly as it can never anchor a lane.

## 7. Estimation function (Slice 1 core)

New module `src/fit.rs` (pure; unit-tested with no GPU/model):

```
pub struct FitInputs {
    pub weight_bytes: u64,
    pub kv_bytes_at_ctx: u64,   // for the assessed context length
}

pub fn assess(hw: &HardwareProfile, m: &FitInputs) -> FitVerdict;
```

Decision order (host-honest):
1. If `hw.host_ram_total_bytes == 0` and no VRAM info → `Unknown`.
2. GPU present: try VRAM-resident via `cuda_vram::evaluate(vram_free,
   weight+kv, headroom)`. Ok → `FitsResident`. Shortfall but
   `weight+kv <= vram_free + usable_host_ram` → `FitsWithOffload` (mirrors the
   documented VRAM+host-RAM offload split). Else fall through.
3. No usable GPU: if `weight+kv <= usable_host_ram` → `CpuOnlyOk`, else a
   negative verdict.

Negative verdicts are split by permanence: a footprint an *idle* host would have
held (`80%` of **total** RAM, plus total VRAM on the offload branch) is
`InsufficientFreeMemory`; anything larger is `WontFit`. The gate itself still
reads only what is free, so the split cannot make the advisor optimistic — it
changes the explanation and, at the load boundary, the remedy (§9).

`usable_host_ram` is `80%` of **available** RAM with **no total-RAM floor** —
deliberately stricter than the `kv_cache` budget, which floors the same figure at
25% of total. The KV guard can afford that floor because the weights are already
resident when it runs; a pre-load advisor cannot, since flooring here would claim
a model "fits" on a genuinely starved host. `src/fit.rs::usable_host_ram_bytes`
is the contract.

## 8. Slices (end-to-end)

### Slice 1 — Docs + pure core (this PR) — ✅ DONE
- This document.
- `src/fit.rs`: `FitVerdict`, `FitInputs`, `assess()` + private pure
  `assess_with_headroom()` — **pure, no I/O**. Registered in `src/lib.rs`.
- 13 unit tests in `src/fit.rs` covering resident / headroom-nudge / offload /
  won't-fit (VRAM+RAM and CPU-only) / cpu-only / RAM-floor-dominates /
  unknown (no-GPU and GPU-overflow) / cuda-flag-without-VRAM / saturation /
  label + serde stability. Footprints use real catalog byte sizes (3B, 8B).
- **KV-term decision (resolves open Q1):** the pure core takes **explicit byte
  inputs** (`FitInputs { weight_bytes, kv_bytes_at_ctx }`). Deriving
  `kv_bytes_at_ctx` pre-download from architecture metadata is deferred to
  Slice 2 (leaning §6.2b: a conservative per-arch bound, no `CatalogItem` schema
  change). This keeps Slice 1 fully deterministic and schema-neutral.
- **No** change to load/generation/catalog-serialization behavior.
- Gates: `cargo fmt --all -- --check` ✅ · `cargo clippy --lib -- -D warnings` ✅ ·
  `cargo test --lib fit::` → **13 passed** ✅. (Full `--all-features` clippy/test
  is the pre-merge gate; the CUDA build is long and run separately.)

### Slice 2 — Catalog metadata + API field — ✅ DONE
- Added `task_tags: &'static [&'static str]` to `CatalogItem` and all **15**
  curated rows (`curated_catalog()`), constrained to `general` / `reasoning` /
  `coding` / `tools`. Advisory positioning, **not** benchmarked (documented as
  such on the field).
- **`params` dropped (honest deviation):** authoritative parameter counts are not
  available pre-download without loading the GGUF, and `size_bytes` is already the
  exact weight footprint the fit math needs — so a hand-entered `params` would add
  hallucinated precision for no gain. Deferred; revisit only if a UI needs a
  human "size" label beyond `size_bytes`.
- Added `fit: FitVerdict` + `task_tags: Vec<String>` to `CatalogItemView`.
  `from_curated(item, hw)` computes `fit` via `fit::assess(hw,
  fit::advisory_footprint(size_bytes))`; `from_hf` → `fit: Unknown`, empty tags.
- **KV term (finalizes open Q1):** `fit::advisory_footprint` pads weight bytes by
  a flat, conservative `ADVISORY_OVERHEAD_PERCENT = 25` to stand in for KV +
  activations at a modest context (over-estimating keeps a "fits" badge safe). A
  per-architecture bound is a documented future refinement; a flat pad avoids
  inventing per-model dims we cannot know pre-download.
- Hardware is probed once via a new `HardwareProfile::cached()` (`OnceLock`) so the
  catalog handler does not re-probe CUDA per request.
- API tests (`catalog_fit_tests`, 5): curated rows carry tags + a verdict; a huge
  model won't fit a tiny host; experimental rows stay `Unknown`+untagged; an
  unprobed host yields `Unknown` for **every** row (never `WontFit`); all tags are
  in the allowed set.
- Gates: `cargo fmt --all -- --check` ✅ · `cargo clippy --lib -- -D warnings` ✅ ·
  `cargo test --lib` → **684 passed, 0 failed** (18 new/relevant among them) ✅.
- JSON is **additive** (`fit`, `task_tags` new keys) — no frontend change yet;
  the WebUI ignores them until Slice 3.

### Slice 3 — UX surfaces — ✅ DONE
- **WebUI** (`CatalogLaneBrowse.jsx`): a dedicated **"This machine:"** advisory
  line per curated row — `Fits your machine` / `Fits (GPU + RAM offload)` /
  `Fits (CPU)` / `Too big for this machine` — plus `· best for <tags>`. Kept on
  its **own line**, visually distinct from the lane/support chip (fit axis ≠
  support axis); `wont_fit` uses the existing `catalog-row-error` style, others
  `catalog-row-faint`. `unknown`/experimental rows show nothing. Download stays
  **un-gated** — the line informs, it does not block. `vite build` ✅.
- **CLI** (`print_catalog`, `src/catalog.rs`): a `FIT (this host)` column via
  `FitVerdict::cli_label()` (`fits` / `fits (offload)` / `fits (CPU)` / `too big`
  / `unknown`), hardware probed once.
- **Load guard** (`load_model`, `POST /api/models/load`): `fit_preload_guard`
  stats the file, and on a `WontFit` verdict returns a typed **422
  `model_too_large_for_host`** naming the largest catalog row that *does* fit —
  **before** the long load / mid-load OOM. Fires only on `WontFit` from a probed
  host; `Unknown`/`Fits*` and a `CAMELID_SKIP_FIT_CHECK=1` override proceed
  unchanged, so it is a fail-fast convenience, not a new hard gate. Pure decision
  split into testable `fit_preload_message` + `best_fitting_catalog_suggestion`.
  *(The refusal set widened past `WontFit` after this slice and was corrected in
  Slice 9; §9 is the current mapping.)*
- Tests: 4 new (`preload_message_*`, `best_fitting_suggestion_*`) → 9 in
  `catalog_fit_tests`. Gates: `cargo fmt` ✅ · `cargo clippy --lib` ✅ ·
  `cargo test --lib` → **688 passed, 0 failed** ✅ · frontend `vite build` ✅.

### Slice 4 — Docs/ledger alignment — ✅ DONE
- `ROADMAP.md`: added a **Model fit advisor (capacity axis)** lane under *Active
  roadmap lanes*, cross-linking this plan and stating the capacity ≠ support
  discipline (advisory verdict, un-gated download, overridable fail-fast, runtime
  guards stay authoritative).
- `COMPATIBILITY.md`: added a **Capacity rule** to *Governing rules* — the fit
  advisor is capacity-only, never implies parity/validation/support, does not
  appear in `model_compatibility`, and cannot promote any row.
- No evidence bundle minted (no screenshot gate requested); the UI line is covered
  by `vite build` and the API/CLI by the Rust suite.

### Slice 5 — Exact KV footprint at the load guard — ✅ DONE

Replaces the flat 25% pad with the **real** KV formula *where it gates* (the load
guard), following the "only exact data may block" principle.

- `src/fit.rs`: `ModelDims { layers, kv_heads, head_dim }`, `KvDtype {F16,F32}`,
  `kv_bytes(dims, ctx, dtype)` = `layers × kv_heads × head_dim × 2 × dtype_bytes ×
  ctx` — the exact mirror of `kv_cache.rs::kv_bytes_per_token`. `exact_footprint`
  = weights + KV(ctx) + a bounded `ACTIVATION_SCRATCH_BYTES` (512 MiB). Default
  assessment context `ADVISORY_CONTEXT_TOKENS = 4096` (KV scales linearly, so the
  trained max is *not* used — the runtime KV guard governs longer chats).
- **Tested against the engine's own vectors:** `kv_bytes` returns exactly 45,056
  (TinyLlama) / 229,376 (Llama 3.2 3B) B/token — the same figures `kv_cache.rs`
  asserts — so it is correct by construction. Plus context-linearity and
  f16 = ½·f32 tests.
- Load guard (`fit_preload_guard`): now probes **live** memory
  (`HardwareProfile::detect()`, not the cached snapshot) and computes an **exact**
  footprint via `read_model_dims` (header-only `gguf::read_metadata` →
  `LlamaModelConfig::from_gguf` → `DenseLlamaDims::from_config`), picking f16 KV on
  a GPU host / f32 on CPU. Falls back to the coarse pad when the header can't be
  parsed (non-GGUF / unknown arch). The catalog **badge stays on the pad** (it's
  advisory, un-gated) — pre-download exact dims (build-time table / header
  range-fetch) are the Phase-2/3 follow-ups.
- Gates: `cargo test --lib` → **694 passed, 0 failed**; fmt + clippy(lib) clean.
  Live-verified end-to-end: `POST /api/models/load` on a 40 GB file →
  `422 model_too_large_for_host` with the actionable "largest that fits" message;
  catalog chips unchanged (no regression).

### Slice 6 — Exact pre-download badge via GGUF header range-fetch (Phase 2 + 3) — ✅ DONE

Makes the catalog badge **exact** (not just the coarse pad) by reading each
model's real dimensions from its GGUF **header only**, over the network, without
downloading the weights.

- **The trick** (`remote_model_dims`): GGUF stores all metadata + tensor-info at
  the file start. Range-fetch a `HEADER_BYTES = 12 MiB` head (covers 128k-vocab
  Llama metadata), write it to a temp file whose *length* is `set_len` to the real
  size (sparse zero tail — `read_metadata` only validates tensor offsets against
  the length, never reads the tail), then reuse the **trusted parser unchanged**.
  A too-small/failed fetch → `None` → the caller keeps the pad. Live-verified:
  Qwen3-0.6B → 28/8/128, Llama-3.2-1B → 16/8/64.
- **Cache + confidence:** a process-wide `RemoteDimsCache` (keyed `repo/filename`,
  stores `None` on failure to avoid re-hammering). `CatalogItemView` gains
  `fit_confidence` (`"exact"` | `"approx"`). `curated_footprint` uses exact dims
  when cached, else the pad. WebUI shows a `~` marker + tooltip on `approx`.
- **Phase 2 (curated):** `get_catalog` background-warms all 15 curated rows
  (`spawn_blocking`, never blocks the response). Rows start `approx` and upgrade to
  `exact` as headers land. Live-verified: exact count climbed 0 → 3 → 12 across
  renders; Llama 3.2 1B rendered exact "Fits your machine".
- **Phase 3 (arbitrary Hugging Face rows):** `from_hf` gives an honest **exact**
  fit once a header is cached (capacity is orthogonal to verification), else
  `Unknown` — never a filename guess. The warm here is **bounded** to the top
  `HF_DIMS_WARM_LIMIT = 5` rows (a query can return 100+ files; the first naive
  version fanned out 119 fetches — fixed). Live-verified: HF rows upgraded to
  `exact` with only ~11 concurrent fetches, not 119.
- **Data cost / control:** header fetches are `12 MiB` each, cached for the process
  lifetime, opt-out via `CAMELID_NO_REMOTE_DIMS=1`. Follow-ups: a disk cache (so
  restarts don't re-fetch) and truly on-demand HF fetch (per row the user opens).
- Gates: `cargo test --lib` → **698 passed, 0 failed**; fmt + clippy(lib) clean;
  `vite build` clean; network path live-verified (gated test
  `CAMELID_TEST_REMOTE_DIMS=1`).

### Slice 7 — `DimsResolver`: consolidate + harden the dims layer — ✅ DONE

Slice 6 shipped the capability but the caching/fetch plumbing was scattered
across `api/mod.rs` as a set of free functions (`store_remote_dims`,
`cached_remote_dims`, `remote_dims_cache`, ...) with three real defects. Slice 7
extracts one owner — `src/fit_dims.rs` (`DimsResolver`, a `OnceLock` singleton) —
and fixes them:

- **No write amplification.** The old path rewrote the *entire* JSON cache on
  every successful fetch (O(n²) over a warm cycle). The resolver marks entries
  dirty and a single debounced writer (`flush_loop`, 2 s coalescing window,
  `spawn_blocking`) persists once per burst.
- **In-flight de-duplication + bounded concurrency.** An `in_flight` set drops
  duplicate requests before they spawn, and a `tokio::Semaphore`
  (`MAX_CONCURRENT_FETCHES = 4`) caps live curls. The old code could spawn a
  fetch per GET per row; the fan-out is now structurally impossible.
- **Reads are side-effect-free for curated rows.** `get_catalog` no longer warms
  curated rows on read — their dims are warmed exactly once at server startup
  (`start_background`). The HF-search branch is *not* side-effect-free: it schedules
  at most `HF_DIMS_WARM_LIMIT = 5` bounded, de-duplicated background fetches.
  `schedule()` is sync and non-blocking (spawns; never awaits a permit on the
  request path).
- **Bounded, versioned, TTL cache.** `DiskCache { version, entries }`
  (`CACHE_SCHEMA_VERSION = 1`) — a version bump or parse error loads as empty
  (self-healing). `ENTRY_TTL_SECS = 30 days` expiry, LRU eviction at
  `MAX_ENTRIES = 512`. The pre-versioned flat-map cache is read as empty and
  rewritten in the new shape on first flush.
- **Honest negatives.** An unparseable header (e.g. a non-dense architecture) is
  cached as `dims: None` with a `FETCH_BACKOFF_SECS = 30 min` backoff, so failing
  rows are attempted once and then left alone instead of re-fetched every run.
- **Honest confidence on HF rows.** `from_hf` now reports `fit_confidence:
  "unknown"` (was `"approx"`) when it has no dims, matching `fit: unknown`; the
  WebUI renders an explicit dashed `~` estimate chip only for the genuine
  `approx` (curated-pad) state.
- **CI covers the parser.** A hermetic fixture test builds a minimal in-memory
  GGUF header and asserts `dims_from_gguf_file` extracts `{layers, kv_heads,
  head_dim}` — the network path is no longer the only coverage.

Live-verified on an RTX 4060 host: fresh start warmed **11 exact / 4 approx** (the
4 approx are all Gemma-4 E2B/E4B/12B/26B-A4B rows — non-dense-Llama metadata,
correctly cached as negatives, not network failures); a restart served **11 exact
within ~1.5 s from disk with zero re-fetch**; disk cache persisted as
`{version:1, entries:{…}}` with 11 positive + 4 negative entries.

- Gates: `cargo test --lib` → **707 passed, 0 failed**;
  `cargo clippy --all-targets --all-features -- -D warnings` clean;
  `cargo fmt --all -- --check` clean; `vite build` clean.

### Slice 8 — review-round hardening + honest invariants — ✅ DONE

Fixes from a second review pass, each verified against the fork (not the stale
tree):

- **Header-fetch robustness (`fit_dims.rs`).** Unique per-`(repo, filename)` temp
  name (was filename-only → concurrent same-name fetches from different repos
  raced one temp file); no `set_len` disk-allocation trick (parse the header
  prefix against the declared full length via `gguf::read_metadata_with_len` —
  `set_len` is *not* sparse on NTFS and would allocate the model's full size);
  `curl --connect-timeout/--max-time` so a hung transfer can't hold a permit
  forever; an RAII `InFlightGuard` clears the in-flight slot even if the fetch
  task panics or early-returns.
- **Guard off the async worker.** `fit_preload_guard` (blocking `metadata` + GGUF
  read + `HardwareProfile::detect`, which inits a CUDA context on GPU hosts) now
  runs under `spawn_blocking`, consistent with the header fetches; a panic in the
  probe is non-fatal (falls through to the load).
- **Honest RAM-budget docs (`fit.rs`).** The advisor's `80% of available` mirrors
  the *value* of the KV-cache budget's available-RAM constant but is an
  independent reimplementation over `HardwareProfile`. (This bullet said `max(80%
  available, 25% of total)` until the total-RAM floor was dropped in `e2f51ffa`
  — flooring a *pre-load* budget at a quarter of total RAM overcommits a starved
  host, which the KV guard can afford and this one cannot.) Documented the two real,
  intentional divergences from the KV runtime guard: (1) the advisor additionally
  probes Linux, where the KV guard is unprobed; (2) on unprobed RAM the KV guard
  fails open (unbounded) while the advisor abstains (`Unknown`). On macOS both
  consumers reuse the same `hw.memsize` + Mach VM-statistics probe.
- **Cached badge vs live guard (documented).** The catalog badge uses the cached
  startup hardware snapshot; the load guard re-probes live. After a model loads
  and consumes VRAM a badge may still read "fits" while the guard returns 422. The
  badge is a static capacity *hint*, not a reservation; the guard is
  authoritative. Re-probing per GET would re-init CUDA on every catalog request.

**Compatibility note (`POST /api/models/load`).** The advisor adds a fail-fast
path: a request that previously always attempted the load can now return a
`422` on a refusing verdict from a *probed* host. This is a behavior change on a
stable endpoint. It never fires on an unprobed host (`Unknown` never refuses),
the authoritative `VramShortfall`/`KvCache` guards remain the hard net after the
load begins, and two overrides restore the unconditional load-attempt behavior:
`"force": true` on the request body and `CAMELID_SKIP_FIT_CHECK=1` in the
environment. §9 is the current verdict → code → status mapping; it narrowed the
refusal set this slice shipped.

- Gates: `cargo test --lib` → **710 passed, 0 failed**;
  `cargo clippy --all-targets --all-features -- -D warnings` clean;
  `cargo fmt --all -- --check` clean.

### Slice 9 — warn on a busy host; refuse only what the machine cannot hold

Slice 3 bucketed **both** negative verdicts into one refusal, and Slice 8's
compatibility note inherited the same conflation. `InsufficientFreeMemory` means
"this machine is big enough, it is just busy right now" — so a 2.0 GB
Llama-3.2-3B Q4_K_M was refused `422` on an 8 GB Mac whenever ~1.3 GB of
post-reserve headroom was short of its ~2.3 GB estimated footprint. Two things
made that a dead end rather than an inconvenience:

- **The remedy was unperformable by its own audience.** The refusal told the user
  to set `CAMELID_SKIP_FIT_CHECK=1`, but the desktop sidecar is spawned
  args-only (`camelid-desktop/src/engine.rs` sets no `.env()`/`.envs()`), so a
  GUI user has nowhere to put that variable.
- **It was a false statement about the machine.** A host with 1.5 GB free
  reported every row down to 1 GB as too big for it, which reads as "this
  product does not work here".

The fix keeps the advisor's arithmetic and changes only what a refusal *means*:

- `LoadModelRequest` gains `force: bool` (`#[serde(default)]`) — the per-request
  equivalent of the env var, checked before any probe or file read so it works
  even on a path that does not exist. It is the override a UI can put on a
  button; the env var stays for operators.
- `fit_preload_message` returns a three-way `FitPreflight`
  (`Proceed`/`Warn`/`Refuse`) instead of an `Option<refusal>`. `Warn` is carried
  to the client as `warnings: [{code, severity, message}]` on the **200** body,
  omitted entirely when empty so an existing client's parse is unchanged.
- `host_memory_exhausted` is the new floor that keeps the downgrade safe:
  `fit::hard_floor_bytes` = total RAM − wired pages, i.e. the largest allocation
  the machine could satisfy *after evicting everything evictable*. Unlike free
  memory this cannot be improved by closing an application, and on Metal hosts
  nothing downstream would catch an oversized weight allocation (`VramShortfall`
  is CUDA-only; the KV budget guards cache growth, not weights). An unprobed
  wired figure (`0`, meaning "no claim") never manufactures a refusal.
- Every remaining `422` message now names `"force": true` as the retry, except
  `model_requires_unload`, whose better remedy is `"replace": true` (one step,
  no override) — see §9.

`FitVerdict::refuses_load()` is deliberately **unchanged**: a catalog row on a
busy host should still render as unavailable. The load boundary asks the separate
question `FitVerdict::is_transient()`, so the two surfaces can disagree honestly.

## 9. Load-time contract — which verdicts warn, which refuse

The binding mapping for `POST /api/models/load`. A verdict alone does not decide
the outcome: `fit_preload_message` also weighs whether releasing a model of ours
would help and whether the allocation is physically satisfiable at all.

| `FitVerdict` | Additional condition | Preflight | HTTP | Code | Remedy offered |
| --- | --- | --- | --- | --- | --- |
| `fits_resident` / `fits_with_offload` / `cpu_only_ok` | — | proceed | 200 | — | — |
| `unknown` | host unprobed, or the advisor abstained | proceed | 200 | — | — |
| `insufficient_free_memory` | another model of ours is resident | **refuse** | 422 | `model_requires_unload` | `"replace": true`, or `POST /api/models/unload` first |
| `insufficient_free_memory` | footprint > total RAM − wired RAM | **refuse** | 422 | `host_memory_exhausted` | `"force": true` (closing apps cannot help) |
| `insufficient_free_memory` | otherwise — the host is merely busy | **warn** | 200 | `host_memory_unavailable` | none needed; it loads |
| `wont_fit` | another model of ours is resident | **refuse** | 422 | `model_requires_unload` | `"replace": true`, or unload first |
| `wont_fit` | otherwise | **refuse** | 422 | `model_too_large_for_host` | a smaller catalog row, named; or `"force": true` |

Reading rules:

- **`model_requires_unload` is checked first**, ahead of both other refusals and
  ahead of the warn path. It is the most specific, and the only one a caller can
  clear without overriding the check at all, so it must not be masked by the
  generic wording.
- **Warnings ride the success body.** A `Warn` is emitted as a top-level
  `warnings` array on the normal `200` `LoadedModel` response — `[{ "code":
  "host_memory_unavailable", "severity": "warning", "message": … }]` — and the
  key is **omitted entirely** when there is nothing to report (never `null`,
  never `[]`).
- **Refusals keep their existing shape:** `422` with the error envelope
  `{ "error": { "code", "message", "field": "path" } }`.
- **Two overrides skip the preflight entirely**, before any hardware probe or
  file read: `"force": true` on the request body (per-request; the only one the
  desktop app can reach, since its sidecar is spawned args-only) and
  `CAMELID_SKIP_FIT_CHECK=1` in the environment (process-wide; operator-only).
  Both leave `VramShortfall`/`KvCacheBudgetExceeded` as the only gates.
- **`refuses_load()` ≠ this table.** `FitVerdict::refuses_load()` answers the
  *catalog* question (should this row render as unavailable?) and returns true
  for both negative verdicts. The load guard additionally asks
  `FitVerdict::is_transient()`. Keeping them separate is intentional: a busy host
  should grey out a row *and* still honor a deliberate load.
- The guard does not run at all for an idempotent re-load of an
  already-resident model — nothing new is allocated.

## 10. UX placement decision (locked)

The verdict is most valuable **at model-selection time**, embedded in the
existing `CatalogLaneBrowse` picker (which already consumes `/api/capabilities`
and the catalog), **plus** a fail-fast guard at the single `load_model` boundary
so every front-end (CLI, WebUI, desktop sidecar) inherits it. **No** separate
"advisor screen." Rationale: one source of truth, mirrors how the support
contract already flows to every surface, lowest friction.

## 11. Testing strategy

- **Unit (Slice 1):** table-driven `assess()` cases; deterministic, no GPU.
- **API (Slice 2):** extend the `capabilities_*` / catalog serialization tests.
- **Frontend (Slice 3):** chip render + advisory copy; download-not-gated
  assertion.
- **Regression:** existing `VramShortfall` and `KvCacheBudgetExceeded` behavior
  unchanged (they remain the hard net).

## 12. Non-goals / guardrails

- No support-claim widening; a fit verdict is never parity/validation.
- No download gating on hardware (informed choice, not enforcement).
- No network or model-download behavior change in Slices 1–2.
- No throughput/tokens-per-sec prediction (out of scope; footprint only).
- Estimation is advisory; runtime guards are authoritative.

## 13. Open questions (resolve before/within Slice 1)

1. KV term: add per-row dims to `CatalogItem` (§6.2a, precise) vs conservative
   per-arch bound (§6.2b, no schema change)? Leaning 6.2b for Slice 1 to stay
   non-invasive, upgrade to 6.2a only if bounds are too loose.
2. Which context length to assess the KV term at for the default badge — a fixed
   modest default (e.g. 4096) vs the row's trained context? Leaning fixed default
   for a stable, comparable badge, with the trained context noted in a tooltip.
3. Source of task tags — pin to each model card's stated strengths, cite in code,
   and keep the set tiny to avoid overclaiming.
