//! Model *fit* advisor — a capacity verdict for "can this machine run this model?".
//!
//! This is a **capacity axis only**. A [`FitVerdict::FitsResident`] means the
//! model's footprint fits the detected memory budget — it says **nothing** about
//! whether the model is *supported* (`COMPATIBILITY.md`) or *runnable-lane
//! anchored* (`crate::runnable`). Those are separate axes and must never be
//! conflated in copy or code.
//!
//! The math is a pure, GPU-free heuristic over byte counts, in the same spirit as
//! [`crate::cuda_vram::evaluate`] (which this module reuses for the VRAM branch).
//! It is **advisory**: the authoritative guards remain the mid-load
//! [`crate::cuda_vram::VramShortfall`] and the mid-generation
//! `BackendError::KvCacheBudgetExceeded` (`src/inference/kv_cache.rs`). This layer
//! only helps a user *choose* before they commit; it never gates a download and
//! never relaxes a runtime guard.
//!
//! On hosts where memory cannot be probed (e.g. macOS, where
//! [`crate::capability::HardwareProfile`] reports `host_ram_total_bytes == 0`) the
//! verdict degrades to [`FitVerdict::Unknown`] rather than guessing — an unknown
//! host must never read as "won't fit".

use crate::capability::HardwareProfile;

/// Share of *available* host RAM the advisor treats as usable. Mirrors the *value*
/// of `KV_CACHE_BUDGET_AVAILABLE_PERCENT` in `src/inference/kv_cache.rs` — an
/// independent constant kept in sync by convention (see [`usable_host_ram_bytes`]
/// for how the two policies relate), not a shared symbol.
const USABLE_RAM_AVAILABLE_PERCENT: u64 = 80;

// NOTE: the KV-cache budget applies a 25%-of-TOTAL floor to survive a transient
// dip in `available` mid-generation (weights already resident, KV growth still
// guarded incrementally by predict-and-abort). A PRE-LOAD fit advisor has the
// opposite risk profile: the weights are NOT yet resident, so flooring usable
// RAM up to 25% of total would let the advisor claim a model "fits" when the
// host is genuinely starved — an overcommit vector that can OOM the load. This
// box has crashed on memory pressure, so the advisor is conservative and uses
// ONLY actually-available RAM (no total-RAM floor).

/// The advisor's verdict for a single (model footprint, host) pair.
///
/// Serialized in `snake_case` for the catalog API (Slice 2); the string form is
/// also exposed via [`FitVerdict::as_str`] for the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    /// Weights + KV fit within the GPU's VRAM (respecting headroom), or — with no
    /// usable GPU — within the host RAM budget. The comfortable case.
    FitsResident,
    /// Weights + KV exceed VRAM alone but fit the combined VRAM + host-RAM budget,
    /// i.e. the documented CUDA VRAM+host-RAM layer-offload split can carry it.
    FitsWithOffload,
    /// No usable GPU, but the footprint fits the host RAM budget (CPU backend).
    CpuOnlyOk,
    /// The footprint fits what this machine could offer **if nothing else were
    /// resident**, but not what is free right now. The remedy is to free memory
    /// (close applications, unload another model), not to pick a smaller model.
    ///
    /// Kept distinct from [`FitVerdict::WontFit`] because collapsing the two is a
    /// factual error with a real cost: on a 16 GB host with 1.5 GB free, every row
    /// down to a 1 GB model reported "too big for this machine", which is untrue and
    /// reads as "this product does not work here". Both verdicts still refuse a load
    /// (see [`FitVerdict::refuses_load`]) — the distinction is in the explanation,
    /// never in the permission.
    InsufficientFreeMemory,
    /// Exceeds every available budget on this host. The pick would fail at load or
    /// generation time; the UI should steer the user to a smaller/quantized row.
    WontFit,
    /// The host's memory could not be probed (e.g. macOS), so no honest capacity
    /// claim can be made. Advisory-blind: never treated as a failure.
    Unknown,
}

impl FitVerdict {
    /// Stable lowercase label, matching the serialized form. For CLI columns/logs.
    pub fn as_str(self) -> &'static str {
        match self {
            FitVerdict::FitsResident => "fits_resident",
            FitVerdict::FitsWithOffload => "fits_with_offload",
            FitVerdict::CpuOnlyOk => "cpu_only_ok",
            FitVerdict::InsufficientFreeMemory => "insufficient_free_memory",
            FitVerdict::WontFit => "wont_fit",
            FitVerdict::Unknown => "unknown",
        }
    }

    /// Whether the verdict says the model can run *somehow* on this host. `Unknown`
    /// is **not** runnable-negative — it is the absence of a claim — so it returns
    /// `false` here only in the sense of "no positive fit was proven". Callers that
    /// must not block on unknowns should test [`FitVerdict::refuses_load`] instead.
    pub fn is_positive_fit(self) -> bool {
        matches!(
            self,
            FitVerdict::FitsResident | FitVerdict::FitsWithOffload | FitVerdict::CpuOnlyOk
        )
    }

    /// Whether a pre-load guard should refuse on this verdict. True for both
    /// negative verdicts and false for every positive one **and** for `Unknown`
    /// (an unprobed host must never be blocked on a claim we cannot make).
    ///
    /// This is the exact complement of "proceed" and exists so that adding a
    /// negative verdict cannot silently widen what the load guard permits: a new
    /// variant must be classified here or the match fails to compile.
    pub fn refuses_load(self) -> bool {
        match self {
            FitVerdict::WontFit | FitVerdict::InsufficientFreeMemory => true,
            FitVerdict::FitsResident
            | FitVerdict::FitsWithOffload
            | FitVerdict::CpuOnlyOk
            | FitVerdict::Unknown => false,
        }
    }

    /// Short human label for a CLI column or terse log. UI surfaces (WebUI) author
    /// their own copy; this is the terminal-facing wording.
    pub fn cli_label(self) -> &'static str {
        match self {
            FitVerdict::FitsResident => "fits",
            FitVerdict::FitsWithOffload => "fits (offload)",
            FitVerdict::CpuOnlyOk => "fits (CPU)",
            FitVerdict::InsufficientFreeMemory => "needs free memory",
            FitVerdict::WontFit => "too big",
            FitVerdict::Unknown => "unknown",
        }
    }
}

/// The footprint of a model to assess, in bytes.
///
/// `weight_bytes` is exact for curated catalog rows (`CatalogItem.size_bytes` is
/// the GGUF file size). `kv_bytes_at_ctx` is the projected key+value cache for the
/// context length being assessed; deriving it pre-download from architecture
/// metadata is the Slice-2 concern — this pure core simply takes both byte counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitInputs {
    pub weight_bytes: u64,
    pub kv_bytes_at_ctx: u64,
}

impl FitInputs {
    /// Total resident footprint (weights + KV), saturating.
    pub fn footprint_bytes(&self) -> u64 {
        self.weight_bytes.saturating_add(self.kv_bytes_at_ctx)
    }
}

/// The usable host-RAM budget in bytes, or `None` when RAM is unknown
/// (`host_ram_total_bytes == 0`). Applies the same `max(80% of available, 25% of
/// total)` formula as the KV-cache budget in `kv_cache.rs`, but is an independent
/// reimplementation, not the same guard. Two intentional differences:
///
/// - Source: the advisor reads [`HardwareProfile`] RAM (probed on Windows, Linux,
///   and macOS), whereas the KV guard reads `gait::host_ram_status` (probed on
///   Windows and macOS; unprobed on Linux). On Linux the advisor enforces a budget
///   while the KV guard is unbounded.
/// - Unprobed RAM: the KV guard fails *open* (unbounded); the advisor abstains here
///   with `None` (surfaced as [`FitVerdict::Unknown`]) rather than assert a capacity
///   it cannot measure.
pub(crate) fn usable_host_ram_bytes(hw: &HardwareProfile) -> Option<u64> {
    if hw.host_ram_total_bytes == 0 {
        return None;
    }
    let by_available = hw
        .host_ram_free_bytes
        .saturating_mul(USABLE_RAM_AVAILABLE_PERCENT)
        / 100;
    // Conservative pre-load capacity: available RAM only, no total-RAM floor
    // (see the constants above — flooring here overcommits a starved host).
    Some(by_available)
}

/// Whether the host has a GPU we can actually place weights on.
fn has_usable_gpu(hw: &HardwareProfile) -> bool {
    hw.cuda_available && hw.cuda_vram_free_bytes > 0
}

/// The host-RAM budget an *idle* machine would offer: the same
/// [`USABLE_RAM_AVAILABLE_PERCENT`] policy applied to **total** rather than
/// available RAM. `None` when RAM is unprobed.
///
/// This is diagnostic only — it explains why a footprint does not fit right now and
/// never authorizes a load. [`usable_host_ram_bytes`] (available RAM, no floor)
/// remains the sole capacity gate, so this cannot become an overcommit vector.
fn idle_host_ram_bytes(hw: &HardwareProfile) -> Option<u64> {
    if hw.host_ram_total_bytes == 0 {
        return None;
    }
    Some(
        hw.host_ram_total_bytes
            .saturating_mul(USABLE_RAM_AVAILABLE_PERCENT)
            / 100,
    )
}

/// Classify a footprint that does not fit the *current* budget: transient pressure
/// ([`FitVerdict::InsufficientFreeMemory`]) when an idle machine would have held it,
/// otherwise a genuine [`FitVerdict::WontFit`]. `capacity_when_idle` is the matching
/// idle budget for whichever branch (GPU-offload or CPU) is refusing.
fn negative_verdict(footprint: u64, capacity_when_idle: u64) -> FitVerdict {
    if footprint <= capacity_when_idle {
        FitVerdict::InsufficientFreeMemory
    } else {
        FitVerdict::WontFit
    }
}

/// Pure fit decision with an explicit VRAM headroom (in MiB), so the whole thing
/// is deterministic and unit-testable without touching process env or a GPU.
///
/// Decision order (host-honest):
/// 1. Usable GPU present → try VRAM-resident via [`crate::cuda_vram::evaluate`].
///    - Ok → [`FitVerdict::FitsResident`].
///    - Shortfall → offload: fits VRAM + usable host RAM → [`FitVerdict::FitsWithOffload`];
///      RAM known but too small → a negative verdict (see below); RAM unknown →
///      [`FitVerdict::Unknown`].
/// 2. No usable GPU → fits host RAM → [`FitVerdict::CpuOnlyOk`]; too small → a
///    negative verdict; RAM unknown → [`FitVerdict::Unknown`].
///
/// Negative verdicts are split by [`negative_verdict`]: a footprint an *idle* host
/// would have held is [`FitVerdict::InsufficientFreeMemory`], anything larger is
/// [`FitVerdict::WontFit`]. Both refuse a load; only the explanation differs. The
/// gate itself still reads **available** memory only, so the split cannot make the
/// advisor optimistic.
fn assess_with_headroom(hw: &HardwareProfile, m: &FitInputs, vram_headroom_mib: u64) -> FitVerdict {
    let footprint = m.footprint_bytes();
    let usable_ram = usable_host_ram_bytes(hw);

    if has_usable_gpu(hw) {
        match crate::cuda_vram::evaluate(hw.cuda_vram_free_bytes, footprint, vram_headroom_mib) {
            Ok(_) => {
                // A VRAM-resident load still stages weights through host RAM:
                // the GGUF tensors are read/repacked host-side before upload
                // (cuda_resident repack + clone_htod). If host RAM is KNOWN to
                // be too starved to stage the footprint, we cannot honestly
                // promise the load succeeds — abstain with `Unknown` (never a
                // false-positive fit) rather than assert `FitsResident`. When
                // host RAM is unprobed (`None`) we don't block a GPU that has
                // room. (Crash-safety: this box has OOM'd on memory pressure.)
                return match usable_ram {
                    Some(ram) if footprint > ram => FitVerdict::Unknown,
                    _ => FitVerdict::FitsResident,
                };
            }
            Err(_) => {
                return match usable_ram {
                    Some(ram) if footprint <= hw.cuda_vram_free_bytes.saturating_add(ram) => {
                        FitVerdict::FitsWithOffload
                    }
                    // Idle capacity for the offload split is total VRAM plus the
                    // idle host-RAM budget: what the machine could offer with
                    // nothing else resident.
                    Some(_) => negative_verdict(
                        footprint,
                        hw.cuda_vram_total_bytes
                            .saturating_add(idle_host_ram_bytes(hw).unwrap_or(0)),
                    ),
                    None => FitVerdict::Unknown,
                };
            }
        }
    }

    match usable_ram {
        Some(ram) if footprint <= ram => FitVerdict::CpuOnlyOk,
        Some(_) => negative_verdict(footprint, idle_host_ram_bytes(hw).unwrap_or(0)),
        None => FitVerdict::Unknown,
    }
}

/// Assess whether `m` fits `hw`, using the configured VRAM headroom
/// ([`crate::cuda_vram::min_headroom_mib`], env `CAMELID_MIN_VRAM_HEADROOM_MIB`).
///
/// This is the public entry point. It is deterministic given the process env and
/// the passed hardware profile; the pure arithmetic lives in
/// [`assess_with_headroom`] for env-free testing.
pub fn assess(hw: &HardwareProfile, m: &FitInputs) -> FitVerdict {
    assess_with_headroom(hw, m, crate::cuda_vram::min_headroom_mib())
}

/// Advisory allowance, as a percent of weight bytes, for everything resident
/// beyond the weights at a modest default context: the KV cache, activations, and
/// scratch. This is a deliberately coarse, deliberately *conservative* (slightly
/// over-estimating) heuristic for the **pre-download** badge — the exact KV cost
/// is architecture- and context-specific and is enforced at runtime by the KV
/// predict-and-abort guard (`src/inference/kv_cache.rs`). Over-estimating keeps a
/// "fits" badge safe rather than optimistic. A per-architecture bound is a future
/// refinement; a flat pad avoids inventing per-model dimensions we cannot know
/// before the GGUF is on disk.
pub const ADVISORY_OVERHEAD_PERCENT: u64 = 25;

/// Build [`FitInputs`] for a catalog row from its known weight footprint
/// (`CatalogItem.size_bytes`), padding by [`ADVISORY_OVERHEAD_PERCENT`] to stand
/// in for KV + activations at a modest context. The pad is carried in
/// `kv_bytes_at_ctx`; it is an estimate, not a measured KV size.
pub fn advisory_footprint(weight_bytes: u64) -> FitInputs {
    let overhead = weight_bytes.saturating_mul(ADVISORY_OVERHEAD_PERCENT) / 100;
    FitInputs {
        weight_bytes,
        kv_bytes_at_ctx: overhead,
    }
}

/// Bytes-per-number of the KV cache, by execution path. The runtime stores KV as
/// f32 on the CPU path and f16 (half) on the GPU-resident path — mirror that so the
/// estimate matches what the engine actually allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    /// CPU path — f32 (4 bytes).
    F32,
    /// GPU-resident path — f16 (2 bytes).
    F16,
}

impl KvDtype {
    /// Bytes per stored KV element.
    pub fn bytes(self) -> u64 {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
        }
    }
}

/// Whether a resident Metal projection type selects the parity-qualified F16
/// primary KV cache. Keep this predicate shared with the runtime's loaded-weight
/// decision (`inference/metal_resident.rs`) so the preload advisor cannot size a
/// Q1/Q2/K-quant model as F32 while the engine actually builds an F16 cache.
pub(crate) fn metal_f16_kv_tensor_type(tensor_type: crate::gguf::GgufTensorType) -> bool {
    matches!(
        tensor_type,
        crate::gguf::GgufTensorType::Q4K
            | crate::gguf::GgufTensorType::Q6K
            | crate::gguf::GgufTensorType::Q1_0
            | crate::gguf::GgufTensorType::Q2_0G64
            | crate::gguf::GgufTensorType::Q2_0G128
            | crate::gguf::GgufTensorType::Pq2_0
    )
}

/// True for the per-layer dense projections inspected by
/// `inference::metal_resident::weights_use_kquant`. Tensor names are available in
/// the GGUF header, so preload can mirror that runtime decision without loading or
/// expanding the weights.
fn is_dense_projection_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((layer, role)) = rest.split_once('.') else {
        return false;
    };
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    matches!(
        role,
        "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "ffn_gate.weight"
            | "ffn_up.weight"
            | "ffn_down.weight"
    )
}

/// Resolve the resident Metal KV width from a parsed local GGUF header and the
/// same operator overrides consumed by the runtime. `q8` is conservatively sized
/// as F16 because [`KvDtype`] has no block-overhead representation and two bytes
/// per element is still an upper bound for the quantized cache.
pub(crate) fn metal_resident_kv_dtype_for_gguf(
    gguf: &crate::gguf::GgufFile,
    explicit_dtype: Option<&str>,
    legacy_kv16: Option<&str>,
) -> KvDtype {
    match explicit_dtype.map(str::trim).map(str::to_ascii_lowercase) {
        Some(value) if matches!(value.as_str(), "f32" | "float32") => return KvDtype::F32,
        Some(value) if matches!(value.as_str(), "f16" | "half" | "q8" | "q8_0") => {
            return KvDtype::F16
        }
        _ => {}
    }
    if legacy_kv16.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true")) {
        return KvDtype::F16;
    }
    if gguf.tensors.iter().any(|tensor| {
        is_dense_projection_name(&tensor.name) && metal_f16_kv_tensor_type(tensor.tensor_type)
    }) {
        KvDtype::F16
    } else {
        KvDtype::F32
    }
}

/// The architecture dimensions the KV-cache size depends on. Read from GGUF
/// metadata (`block_count`, `attention.head_count_kv`, `head_dim`) — never guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ModelDims {
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
}

impl ModelDims {
    /// Sanity bounds so a mis-parsed or adversarial header can't drive absurd KV
    /// math. Comfortably covers every real dense LLM (largest today ~140 layers,
    /// ~128 kv heads, 256 head_dim) with wide headroom.
    pub fn is_plausible(self) -> bool {
        (1..=400).contains(&self.layers)
            && (1..=256).contains(&self.kv_heads)
            && (1..=2048).contains(&self.head_dim)
    }
}

/// Default context the advisor sizes the KV cache at — a "normal use" budget. KV
/// grows linearly with context, and a model's *trained* max (e.g. 131072) would
/// materialize a KV cache larger than the weights, so the advisory sizes at this
/// fixed, realistic length rather than the theoretical max. The runtime's own KV
/// predict-and-abort guard governs longer conversations.
pub const ADVISORY_CONTEXT_TOKENS: u64 = 4096;

/// The resident Metal cache's minimum allocation quantum. A fresh decode grows
/// on demand from 512 positions rather than materializing the advisory 4K context
/// at load time. Preload admission sizes this initial allocation; the runtime KV
/// budget remains authoritative as a conversation grows beyond it.
pub const METAL_INITIAL_KV_CONTEXT_TOKENS: u64 = 512;

/// Measured resident-Metal load/first-token scratch reserve. On the 4B Q1 row,
/// building the engine plus its initial KV cache raises RSS by ~177 MiB; 256 MiB
/// leaves meaningful driver/fragmentation headroom without applying the generic
/// 512 MiB CPU/CUDA allowance twice to unified memory.
pub const METAL_INITIAL_ACTIVATION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

/// Coarse, bounded allowance for activations + framework scratch beyond weights
/// and KV. Small next to weights/KV for single-sequence decode; a fixed margin
/// keeps a "fits" verdict from being optimistic without pretending precision.
pub const ACTIVATION_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;

/// Exact KV-cache bytes for `dims` at `context_tokens` and dtype `kv`. Mirrors the
/// runtime `kv_bytes_per_token` (`src/inference/kv_cache.rs`):
/// `layers × kv_heads × head_dim × 2 (K+V) × dtype_bytes`, times `context_tokens`.
pub fn kv_bytes(dims: ModelDims, context_tokens: u64, kv: KvDtype) -> u64 {
    dims.layers
        .saturating_mul(dims.kv_heads)
        .saturating_mul(dims.head_dim)
        .saturating_mul(2) // K + V
        .saturating_mul(kv.bytes())
        .saturating_mul(context_tokens)
}

/// Build an **exact** footprint from real model dimensions: weights (on-disk size)
/// plus KV at `context_tokens` plus a bounded activation/scratch margin. Use this
/// wherever GGUF metadata is available (on-disk models, the load guard) instead of
/// the coarse [`advisory_footprint`] pad.
pub fn exact_footprint(
    weight_bytes: u64,
    dims: ModelDims,
    context_tokens: u64,
    kv: KvDtype,
) -> FitInputs {
    exact_footprint_with_scratch(
        weight_bytes,
        dims,
        context_tokens,
        kv,
        ACTIVATION_SCRATCH_BYTES,
    )
}

/// Exact footprint with a lane-specific scratch allowance. Preload uses this for
/// resident Metal, whose unified-memory engine has a measured smaller initial
/// allocation; all existing callers retain [`ACTIVATION_SCRATCH_BYTES`] through
/// [`exact_footprint`].
pub(crate) fn exact_footprint_with_scratch(
    weight_bytes: u64,
    dims: ModelDims,
    context_tokens: u64,
    kv: KvDtype,
    scratch_bytes: u64,
) -> FitInputs {
    let kv_and_scratch = kv_bytes(dims, context_tokens, kv).saturating_add(scratch_bytes);
    FitInputs {
        weight_bytes,
        kv_bytes_at_ctx: kv_and_scratch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::SimdCaps;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// A hardware profile with only the memory-relevant fields set; everything
    /// else defaulted. Keeps the fit tests focused on the capacity math.
    fn profile(
        cuda_available: bool,
        vram_free_bytes: u64,
        ram_total_bytes: u64,
        ram_free_bytes: u64,
    ) -> HardwareProfile {
        HardwareProfile {
            metal_available: false,
            metal_device_name: None,
            metal_unified_memory: false,
            cuda_available,
            cuda_device_count: if cuda_available { 1 } else { 0 },
            cuda_device_name: None,
            cuda_compute_capability: None,
            cuda_tensor_cores: false,
            cuda_vram_total_bytes: vram_free_bytes,
            cuda_vram_free_bytes: vram_free_bytes,
            cpu_logical_cores: 8,
            host_ram_total_bytes: ram_total_bytes,
            host_ram_free_bytes: ram_free_bytes,
            simd: SimdCaps::default(),
        }
    }

    fn inputs(weight_bytes: u64, kv_bytes: u64) -> FitInputs {
        FitInputs {
            weight_bytes,
            kv_bytes_at_ctx: kv_bytes,
        }
    }

    // A small headroom so tests reason in round GiB without the default 512 MiB
    // nudging boundary cases.
    const H: u64 = 0;

    #[test]
    fn resident_when_footprint_fits_vram_with_headroom() {
        // 8 GB card, a ~3.4 GB weight + 0.5 GB KV = ~3.9 GB → resident.
        let hw = profile(true, 8 * GIB, 16 * GIB, 12 * GIB);
        let m = inputs(3_421_898_816, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, 512), FitVerdict::FitsResident);
    }

    #[test]
    fn headroom_pushes_a_tight_fit_out_of_resident() {
        // Footprint is just under free VRAM, but the 512 MiB headroom is violated,
        // so it must NOT be resident. With host RAM available it becomes offload.
        let hw = profile(true, 8 * GIB, 32 * GIB, 24 * GIB);
        let m = inputs(8 * GIB - 100 * MIB, 0);
        let verdict = assess_with_headroom(&hw, &m, 512);
        assert_eq!(verdict, FitVerdict::FitsWithOffload);
    }

    #[test]
    fn offload_when_weights_exceed_vram_but_fit_vram_plus_ram() {
        // 8B Q8_0 (~8.5 GB) on a 6 GB card with 32 GB RAM → VRAM+host-RAM offload.
        let hw = profile(true, 6 * GIB, 32 * GIB, 24 * GIB);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(
            assess_with_headroom(&hw, &m, H),
            FitVerdict::FitsWithOffload
        );
    }

    #[test]
    fn wont_fit_when_footprint_exceeds_vram_plus_ram() {
        // Tiny VRAM + tiny RAM cannot carry a 12 GB model even with offload.
        let hw = profile(true, 2 * GIB, 4 * GIB, 3 * GIB);
        let m = inputs(12 * GIB, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn cpu_only_ok_when_no_gpu_and_fits_ram() {
        // No GPU, 16 GB RAM (healthy) → 80%-of-available = ~9.6 GB budget carries a
        // ~3.4 GB model comfortably.
        let hw = profile(false, 0, 16 * GIB, 12 * GIB);
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn wont_fit_cpu_only_when_model_exceeds_ram_budget() {
        // No GPU, 8 GB RAM with 5 GB free → budget = 80% of 5 GB = 4 GB (no
        // total-RAM floor); an 8.5 GB model won't fit.
        let hw = profile(false, 0, 8 * GIB, 5 * GIB);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn starved_host_reports_insufficient_free_memory_not_wont_fit() {
        // 32 GB total but only 2 GB actually free. A PRE-LOAD advisor must NOT
        // floor up to 25% of total (8 GB) and claim a 3.4 GB model fits — the
        // weights are not resident yet, so that would overcommit and OOM the
        // load. Conservative: budget = 80% of 2 GB = 1.6 GB → refused.
        //
        // But the refusal reason matters: this machine plainly *can* hold a 3.4 GB
        // model, it just cannot right now. Reporting "too big for this machine"
        // here is false and tells the user to buy hardware they already own.
        let hw = profile(false, 0, 32 * GIB, 2 * GIB);
        let m = inputs(3_421_898_816, 256 * MIB);
        let verdict = assess_with_headroom(&hw, &m, H);
        assert_eq!(verdict, FitVerdict::InsufficientFreeMemory);
        // The permission is unchanged — only the explanation differs.
        assert!(!verdict.is_positive_fit());
        assert!(verdict.refuses_load());
    }

    #[test]
    fn freeing_memory_on_the_same_host_turns_the_refusal_into_a_fit() {
        // Same host and same model as above, but idle: the verdict flips to a
        // positive fit. This is what makes `InsufficientFreeMemory` the honest
        // label rather than `WontFit` — the shortage is a state, not a property.
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(
            assess_with_headroom(&profile(false, 0, 32 * GIB, 2 * GIB), &m, H),
            FitVerdict::InsufficientFreeMemory
        );
        assert_eq!(
            assess_with_headroom(&profile(false, 0, 32 * GIB, 28 * GIB), &m, H),
            FitVerdict::CpuOnlyOk
        );
    }

    #[test]
    fn a_model_bigger_than_the_whole_machine_stays_wont_fit_even_when_idle() {
        // 8 GB host, fully idle (7.9 GB free), asked for a 40 GB model. Freeing
        // memory cannot help, so the verdict must stay the permanent one.
        let hw = profile(false, 0, 8 * GIB, 7 * GIB + 900 * MIB);
        let m = inputs(40 * GIB, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn offload_shortfall_distinguishes_transient_pressure_from_a_small_host() {
        // GPU too small for the footprint in both cases, so the offload branch
        // decides. 8.5 GB model + 512 MiB KV = ~9.05 GB.
        let m = inputs(8_541_283_552, 512 * MIB);
        // 2 GB card + 32 GB host that is momentarily starved (1 GB free):
        // idle capacity = 2 + 25.6 = 27.6 GB → transient.
        assert_eq!(
            assess_with_headroom(&profile(true, 2 * GIB, 32 * GIB, GIB), &m, H),
            FitVerdict::InsufficientFreeMemory
        );
        // 2 GB card + a genuinely small 4 GB host: idle capacity = 2 + 3.2 =
        // 5.2 GB → permanent.
        assert_eq!(
            assess_with_headroom(&profile(true, 2 * GIB, 4 * GIB, 3 * GIB), &m, H),
            FitVerdict::WontFit
        );
    }

    #[test]
    fn resident_downgrades_to_unknown_when_host_ram_cannot_stage() {
        // 12 GB card easily holds a 4 GB model in VRAM, but the host has only
        // 2 GB free — too little to stage/repack the weights before upload. The
        // advisor must not assert a confident FitsResident; abstain (Unknown),
        // which never blocks the load, rather than promise a fit that may OOM
        // during staging.
        let hw = profile(true, 12 * GIB, 32 * GIB, 2 * GIB);
        let m = inputs(4 * GIB, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::Unknown);
    }

    #[test]
    fn resident_when_gpu_and_host_ram_both_have_room() {
        // Same GPU, but a healthy host (24 GB free) can stage the 4 GB model →
        // a confident FitsResident.
        let hw = profile(true, 12 * GIB, 32 * GIB, 24 * GIB);
        let m = inputs(4 * GIB, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::FitsResident);
    }

    #[test]
    fn unknown_when_ram_unprobed_and_no_gpu() {
        // An unprobed host with no CUDA has no capacity signal. No honest claim possible.
        let hw = profile(false, 0, 0, 0);
        let m = inputs(3_421_898_816, 256 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::Unknown);
    }

    #[test]
    fn unknown_when_gpu_overflows_and_ram_unprobed() {
        // GPU present but too small, and RAM cannot be probed → offload can't be
        // judged → Unknown (never WontFit on an unknown host).
        let hw = profile(true, 2 * GIB, 0, 0);
        let m = inputs(8_541_283_552, 512 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::Unknown);
    }

    #[test]
    fn cuda_flag_without_vram_is_not_a_usable_gpu() {
        // cuda_available=true but 0 free VRAM → treated as CPU host; fits RAM.
        let hw = profile(true, 0, 16 * GIB, 12 * GIB);
        let m = inputs(2 * GIB, 128 * MIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn footprint_saturates_and_wont_fit_on_extreme_values() {
        let m = inputs(u64::MAX, u64::MAX);
        assert_eq!(m.footprint_bytes(), u64::MAX);
        let hw = profile(false, 0, 16 * GIB, 12 * GIB);
        assert_eq!(assess_with_headroom(&hw, &m, H), FitVerdict::WontFit);
    }

    #[test]
    fn verdict_labels_are_stable() {
        assert_eq!(FitVerdict::FitsResident.as_str(), "fits_resident");
        assert_eq!(FitVerdict::FitsWithOffload.as_str(), "fits_with_offload");
        assert_eq!(FitVerdict::CpuOnlyOk.as_str(), "cpu_only_ok");
        assert_eq!(
            FitVerdict::InsufficientFreeMemory.as_str(),
            "insufficient_free_memory"
        );
        assert_eq!(FitVerdict::WontFit.as_str(), "wont_fit");
        assert_eq!(FitVerdict::Unknown.as_str(), "unknown");
        assert!(FitVerdict::FitsResident.is_positive_fit());
        assert!(FitVerdict::FitsWithOffload.is_positive_fit());
        assert!(FitVerdict::CpuOnlyOk.is_positive_fit());
        assert!(!FitVerdict::InsufficientFreeMemory.is_positive_fit());
        assert!(!FitVerdict::WontFit.is_positive_fit());
        assert!(!FitVerdict::Unknown.is_positive_fit());
    }

    #[test]
    fn only_the_negative_verdicts_refuse_a_load() {
        // An unprobed host must never be blocked: `Unknown` is the absence of a
        // claim, not a negative one.
        assert!(FitVerdict::WontFit.refuses_load());
        assert!(FitVerdict::InsufficientFreeMemory.refuses_load());
        assert!(!FitVerdict::Unknown.refuses_load());
        assert!(!FitVerdict::FitsResident.refuses_load());
        assert!(!FitVerdict::FitsWithOffload.refuses_load());
        assert!(!FitVerdict::CpuOnlyOk.refuses_load());
    }

    #[test]
    fn verdict_serializes_to_snake_case() {
        let json = serde_json::to_string(&FitVerdict::FitsWithOffload).unwrap();
        assert_eq!(json, "\"fits_with_offload\"");
    }

    #[test]
    fn kv_bytes_matches_runtime_vectors_at_one_token_f32() {
        // These are the exact per-token figures kv_cache.rs asserts, so our estimate
        // is correct by construction against the engine's own math.
        // TinyLlama: 22 layers * 4 kv * 64 head_dim -> 45,056 B/token.
        let tiny = ModelDims {
            layers: 22,
            kv_heads: 4,
            head_dim: 64,
        };
        assert_eq!(kv_bytes(tiny, 1, KvDtype::F32), 45_056);
        // Llama 3.2 3B: 28 * 8 * 128 -> 229,376 B/token.
        let l3b = ModelDims {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
        };
        assert_eq!(kv_bytes(l3b, 1, KvDtype::F32), 229_376);
    }

    #[test]
    fn kv_bytes_scales_linearly_with_context() {
        let d = ModelDims {
            layers: 22,
            kv_heads: 4,
            head_dim: 64,
        };
        assert_eq!(kv_bytes(d, 4096, KvDtype::F32), 45_056 * 4096);
        assert_eq!(kv_bytes(d, 0, KvDtype::F32), 0);
    }

    #[test]
    fn kv_f16_is_exactly_half_of_f32() {
        let d = ModelDims {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
        };
        assert_eq!(
            kv_bytes(d, 100, KvDtype::F16) * 2,
            kv_bytes(d, 100, KvDtype::F32)
        );
    }

    fn gguf_with_projection(
        name: &str,
        tensor_type: crate::gguf::GgufTensorType,
    ) -> crate::gguf::GgufFile {
        crate::gguf::GgufFile {
            path: std::path::PathBuf::from("synthetic.gguf"),
            version: 3,
            tensor_count: 1,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::new(),
            tensors: vec![crate::gguf::GgufTensorDescriptor {
                name: name.to_string(),
                dimensions: vec![128, 128],
                tensor_type,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 0,
            }],
        }
    }

    #[test]
    fn metal_kv_dtype_tracks_the_projection_quant_not_an_unrelated_tensor() {
        let q1_projection =
            gguf_with_projection("blk.0.attn_q.weight", crate::gguf::GgufTensorType::Q1_0);
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q1_projection, None, None),
            KvDtype::F16
        );

        let q1_embedding =
            gguf_with_projection("token_embd.weight", crate::gguf::GgufTensorType::Q1_0);
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q1_embedding, None, None),
            KvDtype::F32
        );

        let q8_projection =
            gguf_with_projection("blk.0.attn_q.weight", crate::gguf::GgufTensorType::Q8_0);
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q8_projection, None, None),
            KvDtype::F32
        );
    }

    #[test]
    fn metal_kv_dtype_honors_runtime_overrides_conservatively() {
        let q1 = gguf_with_projection("blk.0.ffn_down.weight", crate::gguf::GgufTensorType::Q1_0);
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q1, Some("f32"), None),
            KvDtype::F32
        );

        let q8 = gguf_with_projection("blk.0.ffn_down.weight", crate::gguf::GgufTensorType::Q8_0);
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q8, Some("f16"), None),
            KvDtype::F16
        );
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q8, Some("q8"), None),
            KvDtype::F16,
            "F16 is a safe upper bound for block-quantized KV"
        );
        assert_eq!(
            metal_resident_kv_dtype_for_gguf(&q8, None, Some("true")),
            KvDtype::F16
        );
    }

    #[test]
    fn exact_footprint_is_weights_plus_kv_plus_margin() {
        let d = ModelDims {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
        };
        let f = exact_footprint(3_000_000_000, d, 4096, KvDtype::F16);
        let expected_kv = kv_bytes(d, 4096, KvDtype::F16) + ACTIVATION_SCRATCH_BYTES;
        assert_eq!(f.weight_bytes, 3_000_000_000);
        assert_eq!(f.kv_bytes_at_ctx, expected_kv);
        assert_eq!(f.footprint_bytes(), 3_000_000_000 + expected_kv);
    }

    #[test]
    fn exact_kv_for_a_big_model_at_default_context_is_bounded() {
        // Llama-3 8B (32 layers, 8 kv, 128 head_dim) f16 KV at 4096 tokens is ~512 MiB
        // — the KV term is modest at normal context; the trained-max context is NOT
        // used here (that would be many GiB), which is the whole point of the default.
        let d = ModelDims {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        let kv = kv_bytes(d, ADVISORY_CONTEXT_TOKENS, KvDtype::F16);
        assert_eq!(kv, 32 * 8 * 128 * 2 * 2 * 4096); // exact
        assert!(
            kv < 1024 * 1024 * 1024,
            "KV at default ctx should be < 1 GiB, got {kv}"
        );
    }

    #[test]
    fn model_dims_plausibility_bounds() {
        assert!(ModelDims {
            layers: 32,
            kv_heads: 8,
            head_dim: 128
        }
        .is_plausible());
        assert!(ModelDims {
            layers: 1,
            kv_heads: 1,
            head_dim: 1
        }
        .is_plausible());
        // Zero or absurd values (a mis-parsed header) are rejected.
        assert!(!ModelDims {
            layers: 0,
            kv_heads: 8,
            head_dim: 128
        }
        .is_plausible());
        assert!(!ModelDims {
            layers: 32,
            kv_heads: 0,
            head_dim: 128
        }
        .is_plausible());
        assert!(!ModelDims {
            layers: 100_000,
            kv_heads: 8,
            head_dim: 128
        }
        .is_plausible());
        assert!(!ModelDims {
            layers: 32,
            kv_heads: 8,
            head_dim: 999_999
        }
        .is_plausible());
    }

    #[test]
    fn model_dims_serde_round_trips() {
        let d = ModelDims {
            layers: 28,
            kv_heads: 8,
            head_dim: 128,
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: ModelDims = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }
}
