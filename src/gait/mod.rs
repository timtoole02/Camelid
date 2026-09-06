//! GAIT — per-(model × machine) execution-profile selection.
//!
//! The campaign goal is a flagless, measured, persisted execution configuration
//! discovered once per (model-fingerprint × machine-fingerprint) pair and reused
//! instantly forever after. This module is the **spine skeleton**: the two
//! fingerprints, their combined key, a fail-closed on-disk store of
//! `camelid.gait-receipt/v1` records, and a selector consulted at the execution
//! planner's decision site.
//!
//! This first slice is deliberately **parity-neutral**. The selector only runs
//! when the `CAMELID_GAIT` bring-up gate is set, and with an empty store (the
//! only state this slice can produce) it returns `None`, so the planner falls
//! through to the existing `requested_profile()` / `Auto` path unchanged. The
//! richer per-stage kernel/chunk configuration (the `MANAGED_ENV_KEYS` surface)
//! is **not** wired here — that is a later lane. Nothing in this module mutates
//! the environment or alters any decode/math path.
//!
//! The fingerprints intentionally hash *geometry*, not weights: two checkpoints
//! of the same shape and quantization share a gait, so a fine-tune reuses its
//! base model's profile.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::execution_plan::ExecutionProfile;
use crate::gguf::GgufFile;
use crate::receipt::{canonical_json, sha256_hex};

pub mod calibrate;
pub mod sentinel;
pub mod substrate;

/// Schema identifier stamped into every v1 gait receipt. Mirrors the
/// `camelid.parity-receipt/v1` family so receipts are cited by fingerprint and
/// trivially checked for tampering.
pub const GAIT_RECEIPT_SCHEMA_V1: &str = "camelid.gait-receipt/v1";

/// Environment gate for the GAIT selector. Bring-up scaffold only: the end state
/// of the campaign has no user flag. When unset (the default), the planner is
/// byte-identical to today.
pub const GAIT_GATE_ENV: &str = "CAMELID_GAIT";

/// True when the GAIT selector is enabled for this process. Recognizes the usual
/// truthy spellings; anything else (including unset) is off.
pub fn gait_enabled() -> bool {
    matches!(
        std::env::var(GAIT_GATE_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// The inference-shaping geometry of a model, derived from GGUF metadata. Two
/// models with identical shape + quantization hash identically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSig {
    pub arch: Option<String>,
    pub n_layers: Option<u32>,
    pub n_embd: Option<u32>,
    pub n_heads: Option<u32>,
    pub n_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub n_ff: Option<u32>,
    pub vocab_size: Option<u32>,
    pub rope_dim: Option<u32>,
    pub sliding_window: Option<u32>,
    pub qk_norm: bool,
    /// Per-stage-class set of quantization types present (e.g. `attn_q -> {Q8_0}`,
    /// `ffn_down -> {Q4K}`). Ordered containers keep serialization deterministic.
    pub quant_classes: BTreeMap<String, BTreeSet<String>>,
}

impl ModelSig {
    /// Derive the signature directly from GGUF metadata + tensor descriptors.
    ///
    /// Reads the same architecture-prefixed keys the model loader uses
    /// (`{arch}.block_count`, `{arch}.attention.head_count`, …) but stays
    /// fail-closed: every field is optional and a missing/odd value is simply
    /// `None` rather than an error, so this never fails for an unsupported or
    /// partially-described model.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let arch = gguf.architecture().map(|s| s.to_string());
        let mu = |suffix: &str| -> Option<u32> {
            arch.as_deref()
                .and_then(|a| gguf.metadata_u32(&format!("{a}.{suffix}")))
        };

        let mut quant_classes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut qk_norm = false;
        let mut token_embd_vocab: Option<u32> = None;
        for tensor in &gguf.tensors {
            let class = tensor_class(&tensor.name);
            quant_classes
                .entry(class.to_string())
                .or_default()
                .insert(format!("{:?}", tensor.tensor_type));
            let lower = tensor.name.to_ascii_lowercase();
            if lower.contains("attn_q_norm") || lower.contains("attn_k_norm") {
                qk_norm = true;
            }
            if class == "token_embd" {
                // token_embd.weight is [n_embd, vocab]; the larger dim is vocab.
                if let Some(max) = tensor.dimensions.iter().copied().max() {
                    token_embd_vocab = u32::try_from(max).ok();
                }
            }
        }

        // Resolve every metadata-derived field before moving `arch` into the
        // struct, so the `mu` closure's borrow of `arch` has ended.
        let n_layers = mu("block_count");
        let n_embd = mu("embedding_length");
        let n_heads = mu("attention.head_count");
        let n_kv_heads = mu("attention.head_count_kv");
        let head_dim = mu("attention.key_length");
        let n_ff = mu("feed_forward_length");
        let vocab_size = mu("vocab_size").or(token_embd_vocab);
        let rope_dim = mu("rope.dimension_count");
        let sliding_window = mu("attention.sliding_window");

        ModelSig {
            arch,
            n_layers,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            n_ff,
            vocab_size,
            rope_dim,
            sliding_window,
            qk_norm,
            quant_classes,
        }
    }

    /// Lowercase-hex SHA-256 of the canonical serialization. Stable across runs
    /// for identical inputs and independent of field declaration order.
    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).expect("ModelSig serializes to JSON");
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

/// Classify a tensor by its inference stage from its GGUF name. Coarse but
/// deterministic; norm tensors are matched first so QK-norm weights do not fall
/// into the attention-projection buckets.
pub(crate) fn tensor_class(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("norm") {
        "norm"
    } else if n.contains("attn_qkv") {
        "attn_qkv"
    } else if n.contains("attn_q") {
        "attn_q"
    } else if n.contains("attn_k") {
        "attn_k"
    } else if n.contains("attn_v") {
        "attn_v"
    } else if n.contains("attn_output") || n.contains("attn_out") {
        "attn_output"
    } else if n.contains("ffn_gate") {
        "ffn_gate"
    } else if n.contains("ffn_up") {
        "ffn_up"
    } else if n.contains("ffn_down") {
        "ffn_down"
    } else if n.contains("token_embd") || n.contains("tok_embeddings") {
        "token_embd"
    } else if n.contains("lm_head") || n.contains("output.weight") {
        "output"
    } else {
        "other"
    }
}

/// The inference-relevant fingerprint of the host machine.
///
/// Captures the facts that change which gait is fastest: precise CPU ISA, the
/// physical/SMT topology, and cache sizes. Read-only detection — it shapes the
/// fingerprint, never decode behavior.
///
/// Three campaign inputs are deliberately still absent and tracked for later
/// lanes: per-core **EfficiencyClass** (P/E split — needs the variable-length
/// `GetLogicalProcessorInformationEx` walk, Lane D), **measured STREAM
/// bandwidth / DRAM latency** (Lane B calibration), and the **Windows power-plan
/// GUID** (Lane F drift detection). Their omission keeps this slice read-only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineSig {
    pub os: String,
    pub arch: String,
    pub logical_cores: usize,
    /// Physical core count (SMT siblings collapsed). `None` if the query failed.
    pub physical_cores: Option<usize>,
    /// True when any physical core exposes SMT siblings.
    pub smt: Option<bool>,
    /// Distinct per-core EfficiencyClass values (sorted). On a hybrid CPU the
    /// higher class is the P-core; a single value means a uniform topology.
    /// Empty when the query is unavailable. This is the input to the eventual
    /// P/E role partition.
    pub efficiency_classes: Vec<u8>,
    /// True when more than one EfficiencyClass is present (a hybrid CPU). `None`
    /// when efficiency classes could not be read.
    pub hybrid: Option<bool>,
    /// Sorted list of detected CPU ISA tokens (e.g. `avx2`, `avx512vnni`; `neon`
    /// on aarch64). Precise where the coarse `SimdCaps` was not.
    pub isa: Vec<String>,
    /// Per-core L1 data cache size in bytes (representative max), if known.
    pub l1d_bytes: Option<u32>,
    pub l2_bytes: Option<u32>,
    /// Shared L3 cache size in bytes, if known.
    pub l3_bytes: Option<u32>,
    /// Active Windows power-plan GUID (e.g. Balanced vs High performance). Part
    /// of the fingerprint because a gait calibrated under one power envelope is
    /// wrong under another — changing it changes the key and forces a fresh
    /// calibration. `None` off Windows or if the query fails.
    pub power_plan: Option<String>,
}

impl MachineSig {
    /// Probe the host cheaply (no CUDA context, unlike `HardwareProfile::detect`).
    pub fn detect() -> Self {
        let topo = detect_topology();
        let efficiency_classes = detect_efficiency_classes();
        let hybrid = if efficiency_classes.is_empty() {
            None
        } else {
            Some(efficiency_classes.len() > 1)
        };
        MachineSig {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            physical_cores: topo.physical_cores,
            smt: topo.smt,
            efficiency_classes,
            hybrid,
            isa: detect_isa(),
            l1d_bytes: topo.l1d_bytes,
            l2_bytes: topo.l2_bytes,
            l3_bytes: topo.l3_bytes,
            power_plan: detect_power_plan_guid(),
        }
    }

    /// Build from an already-detected [`crate::capability::HardwareProfile`],
    /// reusing its coarse SIMD probe. Topology/cache fields are left `None` (the
    /// profile does not carry them); call [`MachineSig::detect`] for the full
    /// fingerprint.
    pub fn from_hardware(profile: &crate::capability::HardwareProfile) -> Self {
        let mut isa = Vec::new();
        if profile.simd.avx2 {
            isa.push("avx2".to_string());
        }
        if profile.simd.fma {
            isa.push("fma".to_string());
        }
        if profile.simd.avx512f {
            isa.push("avx512f".to_string());
        }
        if profile.simd.neon {
            isa.push("neon".to_string());
        }
        isa.sort();
        MachineSig {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            logical_cores: profile.cpu_logical_cores,
            physical_cores: None,
            smt: None,
            efficiency_classes: Vec::new(),
            hybrid: None,
            isa,
            l1d_bytes: None,
            l2_bytes: None,
            l3_bytes: None,
            power_plan: None,
        }
    }

    /// Lowercase-hex SHA-256 of the canonical serialization.
    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).expect("MachineSig serializes to JSON");
        sha256_hex(canonical_json(&value).as_bytes())
    }
}

/// Precise CPU ISA tokens, runtime-detected. On x86-64 this resolves the
/// AVX-512 sub-features the coarse `SimdCaps { avx512f }` could not distinguish.
#[cfg(target_arch = "x86_64")]
fn detect_isa() -> Vec<String> {
    // `is_x86_feature_detected!` requires a direct string literal — it cannot be
    // driven by a macro metavariable, so each probe is spelled out.
    let mut isa = Vec::new();
    if std::is_x86_feature_detected!("avx2") {
        isa.push("avx2".to_string());
    }
    if std::is_x86_feature_detected!("fma") {
        isa.push("fma".to_string());
    }
    if std::is_x86_feature_detected!("avx512f") {
        isa.push("avx512f".to_string());
    }
    if std::is_x86_feature_detected!("avx512bw") {
        isa.push("avx512bw".to_string());
    }
    if std::is_x86_feature_detected!("avx512vl") {
        isa.push("avx512vl".to_string());
    }
    if std::is_x86_feature_detected!("avx512dq") {
        isa.push("avx512dq".to_string());
    }
    if std::is_x86_feature_detected!("avx512cd") {
        isa.push("avx512cd".to_string());
    }
    if std::is_x86_feature_detected!("avx512vbmi") {
        isa.push("avx512vbmi".to_string());
    }
    if std::is_x86_feature_detected!("avx512vnni") {
        isa.push("avx512vnni".to_string());
    }
    if std::is_x86_feature_detected!("avx512bf16") {
        isa.push("avx512bf16".to_string());
    }
    isa.sort();
    isa
}

#[cfg(target_arch = "aarch64")]
fn detect_isa() -> Vec<String> {
    let mut isa = Vec::new();
    if std::arch::is_aarch64_feature_detected!("neon") {
        isa.push("neon".to_string());
    }
    isa
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_isa() -> Vec<String> {
    Vec::new()
}

/// Physical topology + cache, collapsed from the OS processor map.
#[derive(Default)]
struct Topology {
    physical_cores: Option<usize>,
    smt: Option<bool>,
    l1d_bytes: Option<u32>,
    l2_bytes: Option<u32>,
    l3_bytes: Option<u32>,
}

/// Windows x86-64: walk the fixed-size `GetLogicalProcessorInformation` records
/// (the same proven API the Rayon sizing uses) for physical cores, SMT, and
/// L1d/L2/L3 cache sizes. Best-effort: any failure leaves the field `None`.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn detect_topology() -> Topology {
    use windows_sys::Win32::System::SystemInformation::{
        CacheData, CacheUnified, GetLogicalProcessorInformation,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION,
    };
    const RELATION_PROCESSOR_CORE: i32 = 0;
    const RELATION_CACHE: i32 = 2;
    const LTP_PC_SMT: u8 = 0x1;

    let mut topo = Topology::default();
    // SAFETY: standard two-call sizing pattern; union fields are read only for
    // the record relationship that defines them.
    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len);
        if len == 0 {
            return topo;
        }
        let count = len as usize / std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION>();
        if count == 0 {
            return topo;
        }
        let mut buf: Vec<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> = Vec::with_capacity(count);
        if GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) == 0 {
            return topo;
        }
        buf.set_len(count);

        let mut physical = 0usize;
        let mut smt = false;
        let (mut l1d, mut l2, mut l3): (u32, u32, u32) = (0, 0, 0);
        for info in &buf {
            if info.Relationship == RELATION_PROCESSOR_CORE {
                physical += 1;
                if info.Anonymous.ProcessorCore.Flags & LTP_PC_SMT != 0 {
                    smt = true;
                }
            } else if info.Relationship == RELATION_CACHE {
                let cache = info.Anonymous.Cache;
                match cache.Level {
                    1 if cache.Type == CacheData || cache.Type == CacheUnified => {
                        l1d = l1d.max(cache.Size)
                    }
                    2 => l2 = l2.max(cache.Size),
                    3 => l3 = l3.max(cache.Size),
                    _ => {}
                }
            }
        }
        if physical > 0 {
            topo.physical_cores = Some(physical);
            topo.smt = Some(smt);
        }
        if l1d > 0 {
            topo.l1d_bytes = Some(l1d);
        }
        if l2 > 0 {
            topo.l2_bytes = Some(l2);
        }
        if l3 > 0 {
            topo.l3_bytes = Some(l3);
        }
    }
    topo
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn detect_topology() -> Topology {
    Topology::default()
}

/// Windows x86-64: distinct per-core EfficiencyClass values (sorted), via the
/// variable-length `GetLogicalProcessorInformationEx` records walked by their
/// `Size` field. A hybrid CPU reports more than one class (the higher is the
/// P-core); a uniform CPU reports one. Best-effort: returns empty on any failure.
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn detect_efficiency_classes() -> Vec<u8> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut classes = BTreeSet::new();
    // SAFETY: standard two-call sizing; records are walked by their self-reported
    // `Size`, and union/field reads use `read_unaligned` since the byte buffer is
    // only 1-aligned. Only RelationProcessorCore records were requested.
    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len);
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        let ok = GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut len,
        );
        if ok == 0 {
            return Vec::new();
        }
        let total = len as usize;
        let mut offset = 0usize;
        // Each record begins with Relationship (u32) + Size (u32).
        while offset + 8 <= total {
            let rec = buf.as_ptr().add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            let size = std::ptr::read_unaligned(std::ptr::addr_of!((*rec).Size)) as usize;
            if size == 0 || offset + size > total {
                break;
            }
            let relationship = std::ptr::read_unaligned(std::ptr::addr_of!((*rec).Relationship));
            if relationship == RelationProcessorCore {
                let eff = std::ptr::read_unaligned(std::ptr::addr_of!(
                    (*rec).Anonymous.Processor.EfficiencyClass
                ));
                classes.insert(eff);
            }
            offset += size;
        }
    }
    classes.into_iter().collect()
}

#[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
fn detect_efficiency_classes() -> Vec<u8> {
    Vec::new()
}

/// The active Windows power-plan GUID in canonical lowercase form. Best-effort:
/// `None` if the query fails.
#[cfg(windows)]
fn detect_power_plan_guid() -> Option<String> {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::System::Power::PowerGetActiveScheme;

    // SAFETY: PowerGetActiveScheme allocates a GUID via LocalAlloc that the caller
    // must LocalFree. We read it once, format it, and free it.
    unsafe {
        let mut guid_ptr: *mut windows_sys::core::GUID = std::ptr::null_mut();
        let err = PowerGetActiveScheme(std::ptr::null_mut(), &mut guid_ptr);
        if err != 0 || guid_ptr.is_null() {
            return None;
        }
        let formatted = format_guid(&*guid_ptr);
        LocalFree(guid_ptr as *mut core::ffi::c_void);
        Some(formatted)
    }
}

#[cfg(not(windows))]
fn detect_power_plan_guid() -> Option<String> {
    None
}

#[cfg(windows)]
fn format_guid(g: &windows_sys::core::GUID) -> String {
    format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        g.data1,
        g.data2,
        g.data3,
        g.data4[0],
        g.data4[1],
        g.data4[2],
        g.data4[3],
        g.data4[4],
        g.data4[5],
        g.data4[6],
        g.data4[7],
    )
}

/// The flagless selection key: `H(model_sig):H(machine_sig)`. No flags
/// participate.
pub fn gait_key(model: &ModelSig, machine: &MachineSig) -> String {
    format!("{}:{}", model.digest(), machine.digest())
}

/// One persisted execution profile per `gait_key`. The skeleton records only the
/// coarse [`ExecutionProfile`]; the measured per-stage kernel struct is added by
/// a later lane. `receipt_id` is the SHA-256 of the canonical body (every field
/// except itself), so the record is self-verifying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaitReceipt {
    pub schema: String,
    pub receipt_id: String,
    pub gait_key: String,
    pub model_sig: ModelSig,
    pub machine_sig: MachineSig,
    pub recorded_profile: ExecutionProfile,
    /// Measured memory characteristics at calibration time — the roofline
    /// denominator and resonance target. Recorded in the body (so it is bound
    /// into `receipt_id`) but **deliberately not part of `gait_key`**: a noisy
    /// measured float would make the cache lookup miss every run. Absent for
    /// receipts written before a measurement was taken.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryMeasurement>,
    /// The calibration evidence that selected `recorded_profile` — the
    /// matched-trial speedup over baseline, roofline %, and any parity-
    /// disqualified candidates. Absent for a hand-set or not-yet-calibrated
    /// receipt. Part of the canonical body, so bound into `receipt_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<calibrate::CalibrationOutcome>,
    /// §6F host-safety scheduling attestation (the §1.2 cap + the §1.1/§1.2
    /// invariants). Absent for receipts written before this lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduling: Option<Scheduling>,
    /// §6F measured host-safety posture (free-RAM headroom at calibration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_safety: Option<HostSafety>,
}

impl GaitReceipt {
    /// Construct and seal a receipt for the given fingerprints + chosen profile.
    pub fn new(
        model_sig: ModelSig,
        machine_sig: MachineSig,
        recorded_profile: ExecutionProfile,
    ) -> Self {
        let gait_key = gait_key(&model_sig, &machine_sig);
        let mut receipt = GaitReceipt {
            schema: GAIT_RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            gait_key,
            model_sig,
            machine_sig,
            recorded_profile,
            memory: None,
            calibration: None,
            scheduling: None,
            host_safety: None,
        };
        receipt.seal();
        receipt
    }

    /// Attach a memory measurement and re-seal (the measurement is part of the
    /// canonical body).
    pub fn with_memory(mut self, memory: MemoryMeasurement) -> Self {
        self.memory = Some(memory);
        self.seal();
        self
    }

    /// Attach the calibration evidence and re-seal.
    pub fn with_calibration(mut self, calibration: calibrate::CalibrationOutcome) -> Self {
        self.calibration = Some(calibration);
        self.seal();
        self
    }

    /// Attach the §6F scheduling attestation and re-seal.
    pub fn with_scheduling(mut self, scheduling: Scheduling) -> Self {
        self.scheduling = Some(scheduling);
        self.seal();
        self
    }

    /// Attach the §6F measured host-safety posture and re-seal.
    pub fn with_host_safety(mut self, host_safety: HostSafety) -> Self {
        self.host_safety = Some(host_safety);
        self.seal();
        self
    }

    /// Recompute the digest the `receipt_id` field should hold.
    pub fn compute_receipt_id(&self) -> String {
        let mut value = serde_json::to_value(self).expect("GaitReceipt serializes to JSON");
        if let serde_json::Value::Object(map) = &mut value {
            map.remove("receipt_id");
        }
        sha256_hex(canonical_json(&value).as_bytes())
    }

    /// Populate `receipt_id` from the canonical body. Call after every other
    /// field is final.
    pub fn seal(&mut self) {
        self.receipt_id = self.compute_receipt_id();
    }

    /// True when the stored `receipt_id` matches the recomputed digest.
    pub fn verify_self_digest(&self) -> bool {
        self.compute_receipt_id() == self.receipt_id
    }
}

/// Resolve the gait store root. Windows: `%LOCALAPPDATA%\Camelid\gait`. Other
/// platforms get an XDG-ish / temp fallback so the module compiles and is inert
/// everywhere. Returns `None` only if no base directory can be determined.
pub fn gait_dir() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_DATA_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(std::env::temp_dir);
    Some(base.join("Camelid").join("gait"))
}

/// File name for a key. The key contains `:` (invalid on Windows), so it is
/// sanitized; the full key is also recorded inside the receipt and re-checked on
/// load. Crate-visible so the safe-boot sentinel addresses the same receipt file
/// when quarantining a suspect gait.
pub(crate) fn key_filename(key: &str) -> String {
    format!("{}.gait.json", key.replace(':', "_"))
}

/// Persist a receipt under `dir`. Fail-closed: returns the error to the caller,
/// who logs and degrades rather than propagating.
pub fn store_in(dir: &Path, receipt: &GaitReceipt) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(key_filename(&receipt.gait_key));
    let json = serde_json::to_string_pretty(receipt)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Load a receipt for `key` from `dir`. Returns `None` on any failure — missing
/// file, parse error, schema mismatch, digest mismatch, or key mismatch — so a
/// corrupt or stale store can never alter behavior beyond "no cached profile".
pub fn load_from(dir: &Path, key: &str) -> Option<GaitReceipt> {
    let path = dir.join(key_filename(key));
    let text = std::fs::read_to_string(&path).ok()?;
    let receipt: GaitReceipt = serde_json::from_str(&text).ok()?;
    if receipt.schema != GAIT_RECEIPT_SCHEMA_V1 {
        return None;
    }
    if receipt.gait_key != key {
        return None;
    }
    if !receipt.verify_self_digest() {
        return None;
    }
    Some(receipt)
}

/// A gait selected for the current (model × machine): the coarse profile the
/// planner should use plus the scheduling substrate to apply.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedGait {
    /// The `gait_key` this profile was selected for — the safe-boot sentinel
    /// records it in the `.applying` marker so a crashing gait can be quarantined
    /// by key on the next launch.
    pub gait_key: String,
    pub profile: ExecutionProfile,
    pub reason: String,
    pub eco_qos_opt_out: bool,
}

/// Whether a receipt's calibration selected the EcoQoS opt-out. A receipt with
/// no calibration block (e.g. hand-written) defaults to false.
pub(crate) fn receipt_eco_opt_out(receipt: &GaitReceipt) -> bool {
    receipt
        .calibration
        .as_ref()
        .map(|c| c.selected_eco_qos_opt_out)
        .unwrap_or(false)
}

/// The selector consulted at the planner's decision site.
///
/// Returns `Some(SelectedGait)` only when the gate is on AND a valid cached
/// receipt exists for this model on this machine. In every other case it returns
/// `None` and the planner falls through to its existing default.
pub fn maybe_select_profile(gguf: &GgufFile) -> Option<SelectedGait> {
    let dir = gait_dir()?;
    maybe_select_profile_in(&dir, gguf)
}

/// `dir`-explicit core of [`maybe_select_profile`], so the gate / `DISABLE` /
/// quarantine behavior is testable against a temporary store. Returns `None`
/// (the baseline path) when the gate is off, the `DISABLE` kill-file is present,
/// or no loadable receipt exists for this `(model × machine)` — a quarantined
/// receipt has been moved out of `dir`, so it simply misses here.
pub(crate) fn maybe_select_profile_in(dir: &Path, gguf: &GgufFile) -> Option<SelectedGait> {
    if !gait_enabled() {
        return None;
    }
    // §1.3 kill-file: while `DISABLE` exists, serve the baseline unconditionally —
    // no cached profile and no substrate.
    if sentinel::disable_present(dir) {
        return None;
    }
    let model_sig = ModelSig::from_gguf(gguf);
    let machine_sig = MachineSig::detect();
    let key = gait_key(&model_sig, &machine_sig);
    let receipt = load_from(dir, &key)?;
    let reason = format!("gait: applied cached profile for {key}");
    Some(SelectedGait {
        gait_key: key,
        eco_qos_opt_out: receipt_eco_opt_out(&receipt),
        profile: receipt.recorded_profile,
        reason,
    })
}

/// Apply the scheduling substrate recorded in a selected gait. The coarse profile
/// is applied by the planner's existing env machinery; this applies the Windows
/// substrate (EcoQoS) and logs the live gait so it is observable. Best-effort and
/// process-level. Only invoked when a gait was selected (gate on + receipt found).
pub fn apply_selected_gait(gait: &SelectedGait) {
    // §4 safe-boot: record the in-progress apply BEFORE touching the host, so a
    // crash/freeze/wedge during apply or early use leaves a marker that the next
    // launch detects and quarantines. The marker is cleared once a real decode
    // completes (`sentinel::mark_healthy`) or on an orderly exit
    // (`sentinel::clean_shutdown`).
    //
    // WAVE 2: the healthy-clear currently fires on the first completed decode,
    // which fully guards the apply window and a parity-gated profile (a profile
    // is only ever persisted after it decoded cleanly on this exact machine
    // during calibration). When non-parity-gated, host-touching levers land
    // (CPU-set pinning, the sustained-throttle valve, live in-process
    // calibration), move the clear-point to *after a sustained-healthy window*
    // so a gait that wedges the host mid-session is still caught.
    if let Some(dir) = gait_dir() {
        let mut layers: Vec<&str> = vec!["gait"];
        if gait.eco_qos_opt_out {
            layers.push("substrate");
        }
        sentinel::begin_apply(&dir, &gait.gait_key, &layers);
    }
    if gait.eco_qos_opt_out {
        // §1.2: scope the EcoQoS opt-out to the compute worker threads only (the
        // Rayon decode pool + this thread), NOT the whole process — UI and
        // background threads keep their eco-friendly OS-managed default.
        let status = substrate::set_compute_pool_eco_qos(true);
        eprintln!(
            "[gait] applied profile={:?} + eco_qos opt-out (compute pool) -> {status:?}",
            gait.profile
        );
    } else {
        eprintln!("[gait] applied profile={:?}", gait.profile);
    }
}

/// §1.2 host-safety compute-thread budget. The compute pool must always leave the
/// OS headroom: reserve at least one core, and at least two on small machines
/// (`<= 8` physical cores), so the UI/OS keep a performance core no matter what.
/// Pure arithmetic on the physical-core count, so it is platform-independent and
/// directly testable; never returns zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeThreadBudget {
    /// Compute worker threads permitted (`>= 1`).
    pub threads: usize,
    /// Cores actually left for the OS (`physical_cores - threads`).
    pub reserved: usize,
}

/// Resolve the [`ComputeThreadBudget`] for a machine with `physical_cores`.
pub fn compute_thread_budget(physical_cores: usize) -> ComputeThreadBudget {
    // Reserve two cores on small boxes, one on larger ones.
    let want_reserve = if physical_cores <= 8 { 2 } else { 1 };
    // Never drop below a single worker, even on a 1-2 core host.
    let threads = physical_cores.saturating_sub(want_reserve).max(1);
    let reserved = physical_cores.saturating_sub(threads);
    ComputeThreadBudget { threads, reserved }
}

/// §1.1 free-RAM floor: GAIT keeps at least `max(20% of total, 4 GiB)` of physical
/// RAM free, so an allocation campaign (e.g. calibration loading candidate
/// weights) can never drive the host into swap / OOM. Pure, so it is testable.
pub fn ram_headroom_floor(total_bytes: u64) -> u64 {
    const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
    (total_bytes / 5).max(FOUR_GIB)
}

/// True when `available_bytes` of free physical RAM respects the §1.1 floor for a
/// host with `total_bytes` of physical RAM.
pub fn ram_headroom_ok(total_bytes: u64, available_bytes: u64) -> bool {
    available_bytes >= ram_headroom_floor(total_bytes)
}

/// Physical RAM `(total, available)` in bytes via `GlobalMemoryStatusEx`. `None`
/// off Windows or on query failure (the caller then proceeds without the gate
/// rather than blocking).
#[cfg(windows)]
pub fn host_ram_status() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: a zeroed MEMORYSTATUSEX with dwLength set to its own size, exactly
    // as GlobalMemoryStatusEx requires; the call only reads/writes that struct.
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) == 0 {
            return None;
        }
        Some((status.ullTotalPhys, status.ullAvailPhys))
    }
}

/// Physical RAM `(total, available)` in bytes on macOS. `total` is `hw.memsize`;
/// `available` is the reclaimable working set — `(free + inactive)` resident pages ×
/// the VM page size, i.e. the pages the kernel can hand back without paging out live
/// data. Counting only `free` would badly understate headroom under the memory
/// compressor (which deliberately keeps the free pool small), so the cold, reclaimable
/// `inactive` pages are included; the KV guard's 80% factor then adds further headroom
/// over this estimate. `None` only on query failure, so the caller proceeds without the
/// gate rather than blocking.
#[cfg(target_os = "macos")]
pub fn host_ram_status() -> Option<(u64, u64)> {
    macos_ram_stats().map(|(total, available, _wired)| (total, available))
}

/// Physical RAM that cannot be evicted, compressed or swapped out under pressure —
/// `wire_count` × page size on macOS. `None` on platforms with no probe wired up.
///
/// This exists for the pre-load fit floor: `total - unevictable` is the largest
/// allocation the machine could satisfy *even after evicting everything evictable*,
/// which is the one memory claim that is a hard fact rather than a snapshot. Callers
/// must treat `None` as "no claim" and fall back to a total-RAM policy, never as zero.
#[cfg(target_os = "macos")]
pub fn host_ram_unevictable_bytes() -> Option<u64> {
    macos_ram_stats().map(|(_total, _available, wired)| wired)
}

/// `None` off macOS: neither `GlobalMemoryStatusEx` nor `/proc/meminfo` exposes a
/// directly comparable non-evictable figure, and guessing one would turn the fit
/// floor into a fabricated gate. Those hosts keep the total-RAM policy alone.
#[cfg(not(target_os = "macos"))]
pub fn host_ram_unevictable_bytes() -> Option<u64> {
    None
}

/// `(total, available, wired)` in bytes from one Mach probe, so the two public
/// accessors above cannot disagree about the same instant.
#[cfg(target_os = "macos")]
fn macos_ram_stats() -> Option<(u64, u64, u64)> {
    // total physical RAM: hw.memsize, the same sysctl Activity Monitor reports.
    let mut memsize: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = std::ffi::CString::new("hw.memsize").expect("static name");
    // SAFETY: the classic sysctlbyname out-param contract — a u64 sink with its byte
    // length and no input value; the call only reads/writes `memsize` and `len`.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut memsize as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || memsize == 0 {
        return None;
    }
    // Acquire the host name port once for the process lifetime: mach_host_self() mints a
    // fresh send right per call that would otherwise need mach_port_deallocate (no libc
    // binding for that), so caching the one send right bounds it to a single
    // process-lifetime retain (no per-call leak — not zero retains).
    static HOST_PORT: std::sync::OnceLock<libc::mach_port_t> = std::sync::OnceLock::new();
    // `libc::mach_host_self` is deprecated in favor of the `mach2` crate, but that is not
    // a dependency here and the symbol still resolves to the same libSystem trap; scope
    // the allow tightly rather than pull in a new crate just for the host port.
    #[allow(deprecated)]
    // SAFETY: mach_host_self() is an argument-less trap returning a stable send right to
    // the unprivileged host name port.
    let host = *HOST_PORT.get_or_init(|| unsafe { libc::mach_host_self() });
    // available: (free + inactive) resident pages × VM page size via the Mach VM stats.
    // SAFETY: a zeroed vm_statistics64 is a valid (all-counts-zero) sink to be overwritten.
    let mut vm: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: HOST_VM_INFO64 paired with a vm_statistics64 sink and its element count,
    // exactly as host_statistics64 requires; the call only reads/writes `vm` and `count`.
    let kr = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut vm as *mut libc::vm_statistics64 as libc::host_info64_t,
            &mut count,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return None;
    }
    // SAFETY: vm_page_size is a libc extern static — the immutable kernel page size that
    // the vm_statistics64 page counts are denominated in.
    let page = unsafe { libc::vm_page_size } as u64;
    let available = (vm.free_count as u64)
        .saturating_add(vm.inactive_count as u64)
        .saturating_mul(page);
    let wired = (vm.wire_count as u64).saturating_mul(page);
    Some((memsize, available, wired))
}

/// `None` on the remaining unixes (no portable cheap probe wired up): the caller then
/// proceeds without the RAM-derived gate, leaving the explicit env override as the gate.
#[cfg(all(not(windows), not(target_os = "macos")))]
pub fn host_ram_status() -> Option<(u64, u64)> {
    None
}

/// The host-safety scheduling posture recorded in a gait receipt (§6F) — an
/// attestation of the guarantees Waves 1–2 enforce, so a reader can audit *how* a
/// gait runs without trusting prose. Every field is honestly derived: the cap
/// from the topology, the constants from architectural invariants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scheduling {
    /// The §1.2 compute-thread cap (`budget.threads`) — the most workers this gait
    /// will use. `None` if the physical-core count was unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_threads: Option<usize>,
    /// Cores reserved for the OS under that cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved_cores: Option<usize>,
    /// Whether the EcoQoS execution-speed opt-out is applied (compute pool only).
    pub eco_qos_opt_out: bool,
    /// Always `"none"` — GAIT never page-locks the weight arena (§1.1).
    pub memory_locking: String,
    /// Always `"untouched"` — GAIT never alters processor-performance / thermal
    /// limits (§1.2).
    pub thermal_limits: String,
    /// Software weight-stream prefetch depth (0 = baseline; `>0` once §6D lands).
    pub stream_prefetch_depth: u32,
    /// CPU-set the compute pool is pinned to (empty until §1.2 CPU-set pinning).
    pub compute_cpuset: Vec<u32>,
}

impl Scheduling {
    /// Build the scheduling attestation from the machine's physical-core count and
    /// the selected EcoQoS choice: records the §1.2 cap and the §1.1/§1.2
    /// architectural constants.
    pub fn attest(physical_cores: Option<usize>, eco_qos_opt_out: bool) -> Self {
        let budget = physical_cores.map(compute_thread_budget);
        Scheduling {
            compute_threads: budget.map(|b| b.threads),
            reserved_cores: budget.map(|b| b.reserved),
            eco_qos_opt_out,
            memory_locking: "none".to_string(),
            thermal_limits: "untouched".to_string(),
            stream_prefetch_depth: 0,
            compute_cpuset: Vec::new(),
        }
    }
}

/// The measured host-safety posture at calibration time (§6F). Only fields GAIT
/// genuinely measures are recorded — throttle-event counting and UI-responsiveness
/// monitoring are deliberately ABSENT until the §1.2 throttle valve exists, so the
/// receipt never attests a safety number it did not actually measure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HostSafety {
    /// Free physical RAM observed at calibration, in GiB (the §1.1 headroom).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_headroom_gib: Option<f64>,
}

impl HostSafety {
    /// Capture the current host-safety posture (free-RAM headroom). The float is
    /// `round_sig6`-normalized so the content-addressed receipt's self-digest
    /// survives a file round-trip (see [`round_sig6`]).
    pub fn capture() -> Self {
        let ram_headroom_gib = host_ram_status()
            .map(|(_total, avail)| round_sig6(avail as f64 / (1024.0 * 1024.0 * 1024.0)));
        HostSafety { ram_headroom_gib }
    }
}

/// Measured memory characteristics — the roofline denominator (sustained DRAM
/// bandwidth) and the latency the resonance tuning must hide. Measured at
/// calibration, not guessed, and never folded into the gait key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryMeasurement {
    /// Sustained STREAM-Triad bandwidth in GB/s (10^9 bytes).
    pub stream_triad_gbs: f64,
    /// Dependent-load (pointer-chase) DRAM latency in nanoseconds.
    pub dram_latency_ns: f64,
}

/// Measure sustained bandwidth and idle latency. Tens of milliseconds of work;
/// run once at calibration. Buffers are sized past any client LLC so the kernels
/// hit DRAM rather than cache.
///
/// Single-threaded probe. The bandwidth figure is only meaningful in an
/// optimized build (an unoptimized build is overhead-bound, not memory-bound) —
/// which is the build that runs calibration, so this matches the roofline the
/// decode path actually sees.
pub fn measure_memory() -> MemoryMeasurement {
    MemoryMeasurement {
        stream_triad_gbs: round_sig6(measure_stream_triad_gbs()),
        dram_latency_ns: round_sig6(measure_dram_latency_ns()),
    }
}

/// Round to 6 significant figures. serde_json's default parser is not bit-exact
/// for full-precision f64 (off by ~1 ULP on the hardest cases), which would
/// break a content-addressed receipt's self-digest after a file round-trip.
/// Measured/derived evidence does not need more than 6 figures, and short
/// decimals deserialize exactly — so this keeps gait receipts verifiable.
pub(crate) fn round_sig6(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let digits = 5 - x.abs().log10().floor() as i32;
    let factor = 10f64.powi(digits);
    (x * factor).round() / factor
}

#[inline(never)]
fn triad(a: &mut [f64], b: &[f64], c: &[f64], scalar: f64) {
    for i in 0..a.len() {
        a[i] = b[i] + scalar * c[i];
    }
}

fn measure_stream_triad_gbs() -> f64 {
    // 32 MiB per array (> a 24-30 MiB client L3); three arrays touched per pass.
    const N: usize = 4 * 1024 * 1024;
    let mut a = vec![0.0f64; N];
    let b = vec![1.0f64; N];
    let c = vec![2.0f64; N];
    let scalar = 3.0;
    triad(&mut a, &b, &c, scalar); // warm caches/TLB
    let iters = 8;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        triad(&mut a, &b, &c, scalar);
    }
    let secs = start.elapsed().as_secs_f64();
    std::hint::black_box(a[N - 1]);
    // Triad moves 24 bytes per element (read b, read c, write a).
    let bytes = 24.0 * N as f64 * iters as f64;
    if secs > 0.0 {
        bytes / secs / 1e9
    } else {
        0.0
    }
}

fn measure_dram_latency_ns() -> f64 {
    // 64 MiB index ring (> LLC). A coprime stride builds one Hamiltonian cycle,
    // so each load depends on the previous and prefetchers cannot hide latency.
    const N: usize = 16 * 1024 * 1024;
    let mut next = vec![0u32; N];
    let stride = 9973usize; // odd prime -> coprime with the power-of-two N
    let mut idx = 0usize;
    for _ in 0..N {
        let nxt = (idx + stride) % N;
        next[idx] = nxt as u32;
        idx = nxt;
    }
    let mut p = 0usize;
    for _ in 0..(N / 8) {
        p = next[p] as usize; // warm
    }
    let start = std::time::Instant::now();
    for _ in 0..N {
        p = next[p] as usize;
    }
    let secs = start.elapsed().as_secs_f64();
    std::hint::black_box(p);
    secs * 1e9 / N as f64
}

/// Row-dot parity oracles for the K-quant formats.
///
/// The calibration tournament (Lane B) disqualifies any candidate kernel whose
/// row·input dot diverges from a reference. Q8_0 already has one
/// (`Q8_0TensorBlocks::dot_row_f32`); this provides the missing Q4_K and Q6_K
/// references. Each reuses the *validated* super-block `dequantize` path and
/// accumulates in f32 — it is a reference, not a fast path, so clarity and
/// faithfulness beat speed.
pub mod oracle {
    use crate::tensor::{Q4KBlock, Q6KBlock, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, QK_K_BLOCK_SIZE};

    /// Reference dot of one quantized matrix row against `input`.
    ///
    /// `row_bytes` is the contiguous quantized bytes of a single row; `input`
    /// has one value per column and its length must be a positive multiple of
    /// the 256-element super-block. Returns `Err` on a shape/length mismatch.
    fn dot_row_kquant(
        row_bytes: &[u8],
        input: &[f32],
        block_bytes: usize,
        dequant: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
    ) -> Result<f32, String> {
        if input.is_empty() || !input.len().is_multiple_of(QK_K_BLOCK_SIZE) {
            return Err(format!(
                "row-dot input width {} is not a positive multiple of {QK_K_BLOCK_SIZE}",
                input.len()
            ));
        }
        let n_blocks = input.len() / QK_K_BLOCK_SIZE;
        let need = n_blocks * block_bytes;
        if row_bytes.len() < need {
            return Err(format!(
                "row-dot needs {need} quantized bytes for {n_blocks} super-blocks, got {}",
                row_bytes.len()
            ));
        }
        let mut decoded = [0.0f32; QK_K_BLOCK_SIZE];
        let mut sum = 0.0f32;
        for block in 0..n_blocks {
            let chunk = &row_bytes[block * block_bytes..(block + 1) * block_bytes];
            dequant(chunk, &mut decoded);
            let base = block * QK_K_BLOCK_SIZE;
            for (j, &weight) in decoded.iter().enumerate() {
                sum += weight * input[base + j];
            }
        }
        Ok(sum)
    }

    /// Q4_K row·input reference dot.
    pub fn dot_row_q4k(row_bytes: &[u8], input: &[f32]) -> Result<f32, String> {
        dot_row_kquant(row_bytes, input, Q4_K_BLOCK_BYTES, |chunk, out| {
            let bytes: &[u8; Q4_K_BLOCK_BYTES] =
                chunk.try_into().expect("chunk sized to Q4_K_BLOCK_BYTES");
            Q4KBlock::from_bytes(bytes).dequantize(out);
        })
    }

    /// Q6_K row·input reference dot.
    pub fn dot_row_q6k(row_bytes: &[u8], input: &[f32]) -> Result<f32, String> {
        dot_row_kquant(row_bytes, input, Q6_K_BLOCK_BYTES, |chunk, out| {
            let bytes: &[u8; Q6_K_BLOCK_BYTES] =
                chunk.try_into().expect("chunk sized to Q6_K_BLOCK_BYTES");
            Q6KBlock::from_bytes(bytes).dequantize(out);
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // A deterministic, finite super-block: scale = 1.0 (f16 0x3C00), min = 0,
        // with varied scale/value bytes. Any byte pattern decodes without panic;
        // finite scales keep the result NaN-free so equality checks are valid.
        fn synth_block(n: usize, seed: u8) -> Vec<u8> {
            let mut b = vec![0u8; n];
            b[0] = 0x00;
            b[1] = 0x3C; // f16 1.0 scale
            b[2] = 0x00;
            b[3] = 0x3C; // f16 1.0 min
            for (i, byte) in b.iter_mut().enumerate().skip(4) {
                *byte = (i as u8).wrapping_mul(31).wrapping_add(seed);
            }
            b
        }

        fn naive_reference(
            row_bytes: &[u8],
            input: &[f32],
            block_bytes: usize,
            dequant: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
        ) -> f32 {
            let n_blocks = input.len() / QK_K_BLOCK_SIZE;
            let mut decoded = [0.0f32; QK_K_BLOCK_SIZE];
            let mut sum = 0.0f32;
            for block in 0..n_blocks {
                let chunk = &row_bytes[block * block_bytes..(block + 1) * block_bytes];
                dequant(chunk, &mut decoded);
                for j in 0..QK_K_BLOCK_SIZE {
                    sum += decoded[j] * input[block * QK_K_BLOCK_SIZE + j];
                }
            }
            sum
        }

        #[test]
        fn q4k_oracle_matches_direct_dequant_two_blocks() {
            let mut bytes = synth_block(Q4_K_BLOCK_BYTES, 7);
            bytes.extend(synth_block(Q4_K_BLOCK_BYTES, 19));
            let input: Vec<f32> = (0..2 * QK_K_BLOCK_SIZE)
                .map(|i| ((i % 13) as f32) * 0.5 - 3.0)
                .collect();
            let got = dot_row_q4k(&bytes, &input).expect("q4k dot");
            let want = naive_reference(&bytes, &input, Q4_K_BLOCK_BYTES, |c, o| {
                let a: &[u8; Q4_K_BLOCK_BYTES] = c.try_into().unwrap();
                Q4KBlock::from_bytes(a).dequantize(o);
            });
            assert_eq!(got.to_bits(), want.to_bits());
            assert!(got.is_finite());
        }

        #[test]
        fn q6k_oracle_matches_direct_dequant() {
            let bytes = synth_block(Q6_K_BLOCK_BYTES, 5);
            let input: Vec<f32> = (0..QK_K_BLOCK_SIZE).map(|i| (i as f32) * 0.01).collect();
            let got = dot_row_q6k(&bytes, &input).expect("q6k dot");
            let want = naive_reference(&bytes, &input, Q6_K_BLOCK_BYTES, |c, o| {
                let a: &[u8; Q6_K_BLOCK_BYTES] = c.try_into().unwrap();
                Q6KBlock::from_bytes(a).dequantize(o);
            });
            assert_eq!(got.to_bits(), want.to_bits());
        }

        #[test]
        fn rejects_misaligned_width() {
            assert!(dot_row_q4k(&[0u8; Q4_K_BLOCK_BYTES], &[1.0; 100]).is_err());
            assert!(dot_row_q4k(&[], &[]).is_err());
        }

        #[test]
        fn rejects_short_byte_buffer() {
            // One super-block of input but zero bytes of weights.
            assert!(dot_row_q6k(&[], &[1.0; QK_K_BLOCK_SIZE]).is_err());
        }
    }
}

/// Serializes the tests that mutate the process-global `CAMELID_GAIT` env var so
/// they cannot race each other (Rust runs tests in parallel by default). Held for
/// the duration of any test that sets or clears the gate.
#[cfg(test)]
pub(crate) static GAIT_TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};

    fn descriptor(name: &str, ty: GgufTensorType, dims: Vec<u64>) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: dims,
            tensor_type: ty,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        }
    }

    fn sample_gguf() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        metadata.insert("llama.block_count".to_string(), GgufMetadataValue::U32(2));
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufMetadataValue::U32(16),
        );
        metadata.insert(
            "llama.attention.head_count".to_string(),
            GgufMetadataValue::U32(4),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_string(),
            GgufMetadataValue::U32(2),
        );
        metadata.insert(
            "llama.feed_forward_length".to_string(),
            GgufMetadataValue::U32(32),
        );
        GgufFile {
            path: PathBuf::from("sample.gguf"),
            version: 3,
            tensor_count: 3,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![
                descriptor("token_embd.weight", GgufTensorType::Q8_0, vec![16, 128]),
                descriptor("blk.0.attn_q.weight", GgufTensorType::Q8_0, vec![16, 16]),
                descriptor("blk.0.ffn_down.weight", GgufTensorType::Q4K, vec![32, 16]),
            ],
        }
    }

    #[test]
    fn model_sig_reads_geometry_and_quant_classes() {
        let sig = ModelSig::from_gguf(&sample_gguf());
        assert_eq!(sig.arch.as_deref(), Some("llama"));
        assert_eq!(sig.n_layers, Some(2));
        assert_eq!(sig.n_embd, Some(16));
        assert_eq!(sig.n_heads, Some(4));
        assert_eq!(sig.n_kv_heads, Some(2));
        assert_eq!(sig.n_ff, Some(32));
        // vocab falls back to the larger token_embd dimension.
        assert_eq!(sig.vocab_size, Some(128));
        assert_eq!(
            sig.quant_classes.get("attn_q"),
            Some(&BTreeSet::from(["Q8_0".to_string()]))
        );
        assert_eq!(
            sig.quant_classes.get("ffn_down"),
            Some(&BTreeSet::from(["Q4K".to_string()]))
        );
    }

    #[test]
    fn digests_are_deterministic_and_sensitive() {
        let a = ModelSig::from_gguf(&sample_gguf());
        let b = ModelSig::from_gguf(&sample_gguf());
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.digest().len(), 64);

        let mut c = a.clone();
        c.n_layers = Some(99);
        assert_ne!(a.digest(), c.digest());
    }

    #[test]
    fn machine_sig_detect_is_stable() {
        let a = MachineSig::detect();
        let b = MachineSig::detect();
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.os, std::env::consts::OS);
    }

    #[test]
    fn memory_measurement_is_sane() {
        let m = measure_memory();
        assert!(
            m.stream_triad_gbs > 0.5 && m.stream_triad_gbs < 5000.0,
            "implausible bandwidth: {} GB/s",
            m.stream_triad_gbs
        );
        assert!(
            m.dram_latency_ns > 0.0 && m.dram_latency_ns < 100_000.0,
            "implausible latency: {} ns",
            m.dram_latency_ns
        );
    }

    #[test]
    fn receipt_with_memory_reseals_and_round_trips() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_mem_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        let with_mem = base.clone().with_memory(MemoryMeasurement {
            stream_triad_gbs: 30.0,
            dram_latency_ns: 80.0,
        });
        // Attaching a measurement changes the sealed digest but not the key.
        assert_ne!(with_mem.receipt_id, base.receipt_id);
        assert_eq!(with_mem.gait_key, base.gait_key);
        assert!(with_mem.verify_self_digest());

        store_in(&dir, &with_mem).expect("store");
        let loaded = load_from(&dir, &with_mem.gait_key).expect("load");
        assert_eq!(loaded, with_mem);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(windows)]
    #[test]
    fn format_guid_matches_canonical() {
        // Balanced power scheme GUID.
        let g = windows_sys::core::GUID {
            data1: 0x381b_4222,
            data2: 0xf694,
            data3: 0x41f0,
            data4: [0x96, 0x85, 0xff, 0x5b, 0xb2, 0x60, 0xdf, 0x2e],
        };
        assert_eq!(format_guid(&g), "381b4222-f694-41f0-9685-ff5bb260df2e");
    }

    #[cfg(windows)]
    #[test]
    fn power_plan_guid_is_well_formed_when_present() {
        if let Some(pp) = MachineSig::detect().power_plan {
            assert_eq!(pp.len(), 36, "guid = {pp}");
            assert_eq!(pp.matches('-').count(), 4);
            assert!(pp.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        }
    }

    #[test]
    fn machine_sig_topology_is_coherent() {
        let m = MachineSig::detect();
        assert!(m.logical_cores >= 1);
        if let Some(p) = m.physical_cores {
            assert!(p >= 1 && p <= m.logical_cores);
        }
        // A per-core L2 never exceeds the shared L3, when both are known.
        if let (Some(l2), Some(l3)) = (m.l2_bytes, m.l3_bytes) {
            assert!(l2 <= l3);
        }
        // `hybrid` must agree with the number of distinct efficiency classes.
        if !m.efficiency_classes.is_empty() {
            assert_eq!(m.hybrid, Some(m.efficiency_classes.len() > 1));
        }
    }

    #[test]
    fn gait_key_combines_both_digests() {
        let model = ModelSig::from_gguf(&sample_gguf());
        let machine = MachineSig::detect();
        let key = gait_key(&model, &machine);
        assert_eq!(key, format!("{}:{}", model.digest(), machine.digest()));
    }

    #[test]
    fn receipt_round_trips_through_the_store() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let receipt = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        assert!(receipt.verify_self_digest());

        let path = store_in(&dir, &receipt).expect("store");
        assert!(path.exists());

        let loaded = load_from(&dir, &receipt.gait_key).expect("load");
        assert_eq!(loaded, receipt);
        assert!(loaded.verify_self_digest());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_misses_on_empty_store() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_miss_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(load_from(&dir, "deadbeef:cafef00d").is_none());
    }

    #[test]
    fn scheduling_attest_records_cap_and_constants() {
        let s = Scheduling::attest(Some(8), true);
        assert_eq!(s.compute_threads, Some(6)); // 8 - 2 reserve on a small box
        assert_eq!(s.reserved_cores, Some(2));
        assert!(s.eco_qos_opt_out);
        assert_eq!(s.memory_locking, "none");
        assert_eq!(s.thermal_limits, "untouched");
        assert_eq!(s.stream_prefetch_depth, 0);
        assert!(s.compute_cpuset.is_empty());
        // Unknown topology -> no thread numbers, but the invariants still hold.
        let u = Scheduling::attest(None, false);
        assert_eq!(u.compute_threads, None);
        assert_eq!(u.reserved_cores, None);
        assert_eq!(u.memory_locking, "none");
    }

    #[test]
    fn receipt_with_scheduling_and_host_safety_round_trips() {
        let dir = std::env::temp_dir().join(format!("camelid_gait_sched_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let base = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        // 9.5 is an exact binary fraction, so it survives the JSON round-trip the
        // content-addressed self-digest depends on.
        let enriched = base
            .clone()
            .with_scheduling(Scheduling::attest(Some(8), false))
            .with_host_safety(HostSafety {
                ram_headroom_gib: Some(9.5),
            });
        // Enrichment changes the sealed digest but not the key, and stays verifiable.
        assert_ne!(enriched.receipt_id, base.receipt_id);
        assert_eq!(enriched.gait_key, base.gait_key);
        assert!(enriched.verify_self_digest());

        store_in(&dir, &enriched).expect("store");
        let loaded = load_from(&dir, &enriched.gait_key).expect("load");
        assert_eq!(loaded, enriched);
        assert_eq!(
            loaded.scheduling.as_ref().unwrap().thermal_limits,
            "untouched"
        );
        assert_eq!(loaded.host_safety.unwrap().ram_headroom_gib, Some(9.5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selector_is_none_when_gate_disabled() {
        let _env = GAIT_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // With the gate unset, the selector must not touch the store at all.
        std::env::remove_var(GAIT_GATE_ENV);
        assert!(maybe_select_profile(&sample_gguf()).is_none());
    }

    #[test]
    fn receipt_eco_opt_out_defaults_false_without_calibration() {
        let receipt = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        assert!(!receipt_eco_opt_out(&receipt));
    }

    #[test]
    fn receipt_eco_opt_out_reads_calibration_choice() {
        let outcome = calibrate::CalibrationOutcome {
            selected_profile: ExecutionProfile::Auto,
            reason: "test".to_string(),
            baseline_tokens_per_s: 1.0,
            selected_tokens_per_s: 1.0,
            speedup: 1.0,
            roofline_pct: 0.0,
            fell_back: false,
            parity_disqualified: Vec::new(),
            selected_eco_qos_opt_out: true,
            selected_groups_per_chunk: None,
            measured_rounds: 1,
            samples: Vec::new(),
        };
        let receipt = GaitReceipt::new(
            ModelSig::from_gguf(&sample_gguf()),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        )
        .with_calibration(outcome);
        assert!(receipt_eco_opt_out(&receipt));
    }
}

/// §9 host-safety gates. These assert the prime-directive guard rails the v2
/// campaign adds, not just the happy path. Run with:
/// `cargo test --lib -- --include-ignored gait_safety`.
#[cfg(test)]
mod gait_safety {
    use super::*;
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        GAIT_TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("camelid_gait_safety_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_gguf() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );
        metadata.insert("llama.block_count".to_string(), GgufMetadataValue::U32(4));
        metadata.insert(
            "llama.embedding_length".to_string(),
            GgufMetadataValue::U32(64),
        );
        GgufFile {
            path: PathBuf::from("safety.gguf"),
            version: 3,
            tensor_count: 1,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![GgufTensorDescriptor {
                name: "blk.0.attn_q.weight".to_string(),
                dimensions: vec![64, 64],
                tensor_type: GgufTensorType::Q8_0,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 0,
            }],
        }
    }

    /// §1.3: a `DISABLE` kill-file forces the baseline path even with the gate on
    /// and a valid cached receipt present.
    #[test]
    fn disable_file_forces_baseline() {
        let _env = lock_env();
        let dir = temp_dir("disable");
        let gguf = sample_gguf();
        let receipt = GaitReceipt::new(
            ModelSig::from_gguf(&gguf),
            MachineSig::detect(),
            ExecutionProfile::Auto,
        );
        store_in(&dir, &receipt).expect("store receipt");

        std::env::set_var(GAIT_GATE_ENV, "1");
        // Sanity: gate on + receipt present => the selector adopts it.
        assert!(
            maybe_select_profile_in(&dir, &gguf).is_some(),
            "precondition: a cached gait should be selected before DISABLE"
        );
        // Drop the kill-file: the selector must now serve baseline (None), no
        // profile and no substrate, despite the live receipt.
        std::fs::write(dir.join("DISABLE"), "").expect("write DISABLE");
        assert!(
            maybe_select_profile_in(&dir, &gguf).is_none(),
            "DISABLE must force the baseline path"
        );

        std::env::remove_var(GAIT_GATE_ENV);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// §4 crash-injection: a stored receipt + a stale `.applying` marker (a prior
    /// unclean exit) must be quarantined on the next launch, after which the
    /// selector serves the proven baseline.
    #[test]
    fn safe_boot_quarantines_bad_gait() {
        let _env = lock_env();
        let dir = temp_dir("safe_boot");
        let gguf = sample_gguf();
        let model_sig = ModelSig::from_gguf(&gguf);
        let machine_sig = MachineSig::detect();
        let key = gait_key(&model_sig, &machine_sig);

        // The persisted (here, suspect) gait that crashed the prior run.
        let receipt = GaitReceipt::new(model_sig, machine_sig, ExecutionProfile::Experimental);
        store_in(&dir, &receipt).expect("store receipt");
        // The marker the crashed run left behind (written directly so the global
        // armed slot is untouched — this stands in for a previous process).
        let marker = sentinel::ApplyingMarker {
            gait_key: key.clone(),
            layers: vec!["gait".to_string(), "substrate".to_string()],
            pid: 999_999,
            utc_epoch_secs: 0,
        };
        std::fs::write(
            dir.join(".applying"),
            serde_json::to_string(&marker).unwrap(),
        )
        .expect("write marker");

        std::env::set_var(GAIT_GATE_ENV, "1");
        // Sanity: before reconciliation the suspect gait is loadable.
        assert!(
            maybe_select_profile_in(&dir, &gguf).is_some(),
            "precondition: the suspect gait should load before reconciliation"
        );

        // Next launch reconciles the stale marker.
        match sentinel::reconcile_on_startup(&dir) {
            sentinel::StartupReconcile::UncleanExit {
                gait_key,
                quarantined,
            } => {
                assert_eq!(gait_key.as_deref(), Some(key.as_str()));
                assert!(
                    quarantined,
                    "first unclean exit must quarantine (threshold 1)"
                );
            }
            other => panic!("expected UncleanExit, got {other:?}"),
        }

        // The receipt is quarantined, the marker cleared, so the selector now
        // boots the baseline.
        assert!(
            maybe_select_profile_in(&dir, &gguf).is_none(),
            "after quarantine the selector must serve baseline"
        );
        assert!(
            dir.join(".quarantine").join(key_filename(&key)).exists(),
            "the suspect receipt must be preserved under .quarantine/ for diagnosis"
        );
        assert!(
            !dir.join(".applying").exists(),
            "the marker must be cleared"
        );

        std::env::remove_var(GAIT_GATE_ENV);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// §1.2: the compute-thread budget always leaves the OS a core reserve, never
    /// returns zero, and never exceeds the physical-core count.
    #[test]
    fn core_headroom() {
        for &phys in &[1usize, 2, 4, 6, 8, 12, 16, 32, 64] {
            let b = compute_thread_budget(phys);
            assert!(b.threads >= 1, "phys={phys}: must keep >=1 worker");
            assert!(
                b.threads <= phys,
                "phys={phys}: cannot exceed physical cores"
            );
            assert_eq!(
                b.reserved,
                phys - b.threads,
                "phys={phys}: reserved bookkeeping"
            );
            if phys >= 2 {
                assert!(b.reserved >= 1, "phys={phys}: OS must keep >=1 core");
            }
            if (3..=8).contains(&phys) {
                assert!(
                    b.reserved >= 2,
                    "phys={phys}: small boxes reserve >=2 cores"
                );
                assert!(b.threads <= phys - 2, "phys={phys}: threads <= phys-2");
            }
            if phys > 8 {
                assert!(b.threads < phys, "phys={phys}: threads <= phys-1");
            }
        }
    }

    /// §1.1: the weight/KV arena is never page-locked. Audit the load/mmap path
    /// source for the forbidden APIs — locking pages makes them non-reclaimable
    /// and can OOM/wedge the host (the v1 crash mechanism, REMOVED in v2).
    #[test]
    fn no_weight_locking() {
        let root = env!("CARGO_MANIFEST_DIR");
        // The weight-arena load + mmap path. (Deliberately not this file — it
        // contains the forbidden token strings below as the search needles.)
        let arena_files = ["src/wire_mmap.rs", "src/platform_fs.rs"];
        let forbidden = [
            "VirtualLock",
            "MEM_LARGE_PAGES",
            "GetLargePageMinimum",
            "SetProcessWorkingSetSize",
            "mlockall",
            "MAP_LOCKED",
        ];
        for rel in arena_files {
            let path = std::path::Path::new(root).join(rel);
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            for token in forbidden {
                assert!(
                    !src.contains(token),
                    "{rel} must not page-lock the weight arena (found `{token}`)"
                );
            }
        }
    }

    /// §6C: off Windows the scheduling substrate is fully inert — every lever
    /// returns Unavailable, so a non-Windows run is byte-identical to baseline.
    #[cfg(not(windows))]
    #[test]
    fn non_windows_byte_identical() {
        assert_eq!(
            substrate::set_eco_qos_opt_out(true),
            substrate::EcoQosStatus::Unavailable
        );
        assert_eq!(
            substrate::set_thread_eco_qos_opt_out(true),
            substrate::EcoQosStatus::Unavailable
        );
        assert_eq!(
            substrate::set_compute_pool_eco_qos(true),
            substrate::EcoQosStatus::Unavailable
        );
    }

    #[cfg(windows)]
    fn hanging_child() -> std::process::Child {
        std::process::Command::new("cmd")
            .args(["/C", "ping -n 30 127.0.0.1 >NUL"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn hanging child")
    }

    #[cfg(not(windows))]
    fn hanging_child() -> std::process::Child {
        std::process::Command::new("sleep")
            .arg("30")
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn hanging child")
    }

    /// §1.4: a candidate that hangs is killed at the deadline and abandoned — the
    /// supervisor returns promptly and the calling (serving) process survives.
    #[test]
    fn watchdog_survives_hung_candidate() {
        use std::time::{Duration, Instant};
        let child = hanging_child();
        let started = Instant::now();
        let outcome =
            calibrate::supervise(child, Duration::from_millis(400), Duration::from_millis(20));
        let elapsed = started.elapsed();
        assert!(
            matches!(outcome, calibrate::WatchdogOutcome::TimedOut),
            "a hung candidate must time out, got {outcome:?}"
        );
        // Killed at the deadline, not waited out (the child would sleep ~30s); and
        // reaching this line at all proves the supervisor did not wedge us.
        assert!(
            elapsed < Duration::from_secs(5),
            "watchdog must kill promptly, took {elapsed:?}"
        );
    }

    /// §1.1: the free-RAM floor is max(20% of total, 4 GiB), and the headroom
    /// check honors it.
    #[test]
    fn ram_headroom() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // 20% dominates on a big box; the 4 GiB floor dominates on a small one.
        assert_eq!(ram_headroom_floor(100 * GIB), 20 * GIB);
        assert_eq!(ram_headroom_floor(8 * GIB), 4 * GIB);
        assert!(ram_headroom_ok(100 * GIB, 30 * GIB));
        assert!(!ram_headroom_ok(100 * GIB, 10 * GIB));
        assert!(ram_headroom_ok(8 * GIB, 5 * GIB));
        assert!(!ram_headroom_ok(8 * GIB, 2 * GIB));
    }

    /// On macOS, `host_ram_status` reports a live physical-RAM reading (total via
    /// `hw.memsize`, available via the Mach VM statistics) instead of the off-platform
    /// `None`, so the KV predict-and-abort auto-budget actually engages on this host.
    #[cfg(target_os = "macos")]
    #[test]
    fn host_ram_status_reports_live_physical_ram() {
        let (total, available) = host_ram_status().expect("macOS must report host RAM");
        assert!(total > 0, "total physical RAM must be positive");
        assert!(available > 0, "available physical RAM must be positive");
        assert!(
            available <= total,
            "available {available} must not exceed total {total}"
        );
    }
}
