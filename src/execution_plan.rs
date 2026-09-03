use std::{collections::BTreeMap, env, path::Path};

use serde::{Deserialize, Serialize};

use crate::gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType};

const MANAGED_ENV_KEYS: &[&str] = &[
    "CAMELID_PARALLEL_LINEAR",
    "CAMELID_MAC_Q8_REPACK",
    "CAMELID_MAC_Q8_PREFILL_I8MM",
    "CAMELID_MAC_Q8_SCHED",
    "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
    "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
    "CAMELID_FORWARD_RSS_TIMINGS",
    "CAMELID_X86_Q8_REPACK",
    "CAMELID_X86_Q8_KERNEL",
    "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
    "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
    "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
    "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
    "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
    "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
    "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
    "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
    "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
    "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
    "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
    "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
    "CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER",
    "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
    "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
];

struct ManagedPassthroughEnvKey {
    key: &'static str,
    owner_gate: &'static str,
}

const MANAGED_PASSTHROUGH_ENV_KEYS: &[ManagedPassthroughEnvKey] = &[
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
        owner_gate: "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
    },
];

pub const MAC_Q8_PREFILL_I8MM_MIN_ROWS: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProfile {
    Safe,
    Auto,
    Experimental,
    Debug,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub profile: ExecutionProfile,
    pub operating_system: String,
    pub architecture: String,
    pub platform_label: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    pub model_family: String,
    /// Human quantization label: the declared `general.file_type` (llama.cpp
    /// ftype naming) when present and credible, else a tensor-scan bucket.
    /// Descriptive only — routing branches on tensor predicates, not this.
    pub quant_type: String,
    /// Row identity as recognized from `general.name` (or the filename when
    /// the name is junk). Name-derived: it does NOT imply the quant on disk
    /// matches the row's evidence — see `support_level`.
    pub exact_model_row: String,
    /// Plan-level support string for the recognized row, quant-gated: only a
    /// Q8_0 file of a recognized row reports that row's level; every other
    /// quant reports `unknown_or_unvalidated`. `/api/capabilities` remains
    /// the support source of truth.
    pub support_level: String,
    pub selected_backend: String,
    pub selected_q8_path: String,
    pub prefill_path: String,
    pub prefill_runtime_policy: String,
    pub decode_path: String,
    pub thread_count: usize,
    pub diagnostics_status: String,
    pub fallback_path: String,
    /// True when the GPU-resident CUDA decode engine drives decode for this process
    /// (surfaced in `/api/capabilities` so a loaded row reports the live GPU path). The
    /// `selected_backend`/`decode_path` above carry the `cuda_resident_q8_*` labels when
    /// this is set; mirrors the Metal lane's `metal_available` capabilities signal.
    pub cuda_resident_active: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlanOutcome {
    pub plan: ExecutionPlan,
    pub env_updates: BTreeMap<&'static str, Option<&'static str>>,
}

/// The operator-supplied environment the planner reads, captured BEFORE any
/// plan has applied its `env_updates`.
///
/// `MANAGED_ENV_KEYS` are both planner INPUTS (operator opt-outs) and planner
/// OUTPUTS (written by [`PlannerEnv::apply`]). Reading them live makes the
/// planner consult its own previous output: the macOS Metal-resident selection
/// writes `CAMELID_MAC_Q8_REPACK=off`, and the NEXT `plan_for_model` in the same
/// process reads that "off" as an operator opt-out and fails closed to the safe
/// plan. That is a pure disclosure bug — the plan cannot disarm the
/// runtime gates, so serve kept running the Metal-resident lane while
/// `/v1/health`, `/api/capabilities` and `/execution-plan` reported
/// `cpu_reference` / `safe_cpu_decode` from the second load onward (a ~235x
/// misstatement for a gemma3 row, whose CPU bridge decodes at ~0.2 tok/s
/// against the resident lane's ~47). Every desktop user hit it, because the
/// desktop always loads its model at runtime on top of the startup auto-select.
///
/// So this snapshot is the planner's view of what the OPERATOR asked for. Keys
/// the planner never writes are not snapshotted and keep reading live — they
/// are genuine runtime inputs (`CAMELID_METAL_RESIDENT_DECODE`,
/// `CAMELID_PROFILE`, `CAMELID_MAC_Q8_METAL_PLAN`, …) and a stale copy of those
/// would be its own disclosure bug.
///
/// This is the fix `macos_q8_metal_plan_selectable`'s doc comment describes as
/// "the plan to stop overloading one variable for both operator opt-out and
/// plan output"; that function's residual — an operator who PRE-sets
/// `CAMELID_MAC_Q8_REPACK=0` getting a safe plan with resident routing — is
/// deliberately preserved, because a pre-set value IS in the baseline.
#[derive(Clone, Debug, Default)]
pub struct PlannerEnv {
    passthrough_env: BTreeMap<&'static str, Option<String>>,
    /// One entry per [`MANAGED_ENV_KEYS`] key; `None` means the operator left it
    /// unset. An EMPTY map (i.e. `PlannerEnv::default()`) means "no baseline
    /// captured" and every key falls back to reading live.
    managed_env: BTreeMap<&'static str, Option<String>>,
}

impl PlannerEnv {
    pub fn capture() -> Self {
        let passthrough_env = MANAGED_PASSTHROUGH_ENV_KEYS
            .iter()
            .map(|entry| {
                (
                    entry.key,
                    env::var(entry.key)
                        .ok()
                        .filter(|value| managed_positive_usize_value(value)),
                )
            })
            .collect();
        let managed_env = MANAGED_ENV_KEYS
            .iter()
            .map(|key| (*key, env::var(key).ok()))
            .collect();
        Self {
            passthrough_env,
            managed_env,
        }
    }

    /// The value the OPERATOR supplied for `key`, ignoring whatever a
    /// previously applied plan wrote over it. Non-managed keys are never
    /// written by a plan, so they read live.
    fn operator_var(&self, key: &str) -> Option<String> {
        match self.managed_env.get(key) {
            Some(baseline) => baseline.clone(),
            None => env::var(key).ok(),
        }
    }

    fn flag_disabled(&self, key: &str) -> bool {
        self.operator_var(key)
            .is_some_and(|value| flag_value_disabled(&value))
    }

    fn flag_enabled(&self, key: &str) -> bool {
        self.operator_var(key)
            .is_some_and(|value| flag_value_enabled(&value))
    }

    /// [`metal_flag_value_enabled`] against the operator baseline.
    fn metal_flag_enabled(&self, key: &str) -> bool {
        self.operator_var(key)
            .is_some_and(|value| metal_flag_value_enabled(&value))
    }

    /// Whether the operator named the AVX2 kernel explicitly. Currently unused,
    /// retained from the free function it replaces. It reads through the
    /// baseline for the reason this whole type exists: the planner WRITES
    /// `CAMELID_X86_Q8_KERNEL=avx2`, so a live read would make every later plan
    /// in the process believe the operator had asked for AVX2 by name.
    #[allow(dead_code)]
    fn x86_kernel_avx2_explicitly_requested(&self) -> bool {
        self.operator_var("CAMELID_X86_Q8_KERNEL")
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("avx2"))
    }

    /// [`invalid_x86_kernel_value`] against the operator baseline.
    fn invalid_x86_kernel_override(&self) -> Option<String> {
        let value = self.operator_var("CAMELID_X86_Q8_KERNEL")?;
        invalid_x86_kernel_value(&value).map(ToOwned::to_owned)
    }

    pub fn apply(&self, updates: &BTreeMap<&'static str, Option<&'static str>>) {
        for key in MANAGED_ENV_KEYS {
            match updates.get(key).copied().flatten() {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
        for entry in MANAGED_PASSTHROUGH_ENV_KEYS {
            if env_updates_enable_gate(updates, entry.owner_gate) {
                match self
                    .passthrough_env
                    .get(entry.key)
                    .and_then(|value| value.as_deref())
                {
                    Some(value) => env::set_var(entry.key, value),
                    None => env::remove_var(entry.key),
                }
            } else {
                env::remove_var(entry.key);
            }
        }
    }
}

fn managed_positive_usize_value(value: &str) -> bool {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .is_some()
}

fn env_updates_enable_gate(
    updates: &BTreeMap<&'static str, Option<&'static str>>,
    key: &'static str,
) -> bool {
    matches!(updates.get(key).copied().flatten(), Some("on"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanPlatform {
    pub operating_system: String,
    /// Host operating-system product version. On macOS this is the exact
    /// `sw_vers -productVersion` value used to scope host-specific receipts.
    pub operating_system_version: String,
    pub architecture: String,
    pub platform_label: String,
    /// Hardware model identifier (for example `Mac16,10`), distinct from the
    /// processor brand. Unknown on platforms where Camelid has no stable probe.
    pub host_model_identifier: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    /// A usable Metal compute device exists on this host (always false off macOS).
    pub metal_available: bool,
    /// A usable CUDA compute device exists, independent of the resident-Q8 gate.
    pub cuda_available: bool,
    /// State of the platform-neutral user GPU switch.
    pub gpu_accel_enabled: bool,
    /// The CUDA resident decode engine will drive decode for this process (a usable
    /// CUDA device is present, GPU acceleration is on, and neither deterministic mode
    /// nor `CAMELID_CUDA_RESIDENT_DECODE=0` forces the CPU reference). When true, the
    /// CPU Q8 rows4 repack is skipped: the GPU resident engine consumes plain RAM-
    /// resident Q8_0 blocks, and the repack replaces them (the two are mutually
    /// exclusive on weight storage, exactly as the Metal-resident plan handles).
    pub cuda_resident_active: bool,
}

impl PlanPlatform {
    pub fn current() -> Self {
        let operating_system = env::consts::OS.to_string();
        let operating_system_version = operating_system_version();
        let architecture = env::consts::ARCH.to_string();
        let cpu_features = cpu_features();
        let host_model_identifier = host_model_identifier();
        let cpu_model = cpu_model();
        let platform_label = platform_label(&operating_system, &architecture, &cpu_model);
        let metal_available = crate::metal::detect_metal_device().available;
        let cuda_available = crate::cuda::is_available();
        let gpu_accel_enabled = crate::cuda::gpu_accel_enabled();
        let cuda_resident_active = cuda_resident_decode_will_run();
        Self {
            operating_system,
            operating_system_version,
            architecture,
            platform_label,
            host_model_identifier,
            cpu_model,
            cpu_features,
            metal_available,
            cuda_available,
            gpu_accel_enabled,
            cuda_resident_active,
        }
    }
}

/// Planning-time mirror of `inference::resident_decode_cuda_enabled`: true when the GPU
/// resident decode engine will run, so the CPU Q8 rows4 repack must be skipped (the GPU
/// needs un-repacked plain Q8_0 blocks). Deterministic mode and
/// `CAMELID_CUDA_RESIDENT_DECODE=0` force it false (CPU reference), matching the runtime
/// gate. On a host without a usable CUDA device (or a build without the `cuda` feature)
/// `cuda::is_available()` is false, so the CPU repack path is unaffected.
fn cuda_resident_decode_will_run() -> bool {
    if env_flag_enabled("CAMELID_DETERMINISTIC") {
        return false;
    }
    if let Ok(value) = env::var("CAMELID_CUDA_RESIDENT_DECODE") {
        if matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ) {
            return false;
        }
    }
    crate::cuda::is_available() && crate::cuda::gpu_accel_enabled()
}

/// Plan for a model, reading operator opt-outs from the CURRENT environment.
///
/// Correct only for a process that plans ONCE (every one-shot CLI subcommand).
/// A long-lived process that plans per model load — `camelid serve` — must
/// capture a [`PlannerEnv`] before its first load and use
/// [`plan_for_model_with_env`], or the second plan reads the first plan's
/// `env_updates` as operator opt-outs and fails closed to the safe plan.
pub fn plan_for_model(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
) -> ExecutionPlanOutcome {
    plan_for_model_with_env(model_path, gguf, threads, &PlannerEnv::capture())
}

/// Plan for a model against an operator-environment baseline captured before
/// any plan applied its `env_updates`. This is the entry point every repeated
/// planner (serve's model-load pipeline) must use — see [`PlannerEnv`].
pub fn plan_for_model_with_env(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
    planner_env: &PlannerEnv,
) -> ExecutionPlanOutcome {
    plan_for_model_with_platform_and_env(
        model_path,
        gguf,
        threads,
        PlanPlatform::current(),
        planner_env,
    )
}

pub fn plan_for_model_with_platform(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
    platform: PlanPlatform,
) -> ExecutionPlanOutcome {
    plan_for_model_with_platform_and_env(
        model_path,
        gguf,
        threads,
        platform,
        &PlannerEnv::capture(),
    )
}

pub fn plan_for_model_with_platform_and_env(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
    platform: PlanPlatform,
    planner_env: &PlannerEnv,
) -> ExecutionPlanOutcome {
    // GAIT selector (bring-up gate `CAMELID_GAIT`, default off): consult the
    // per-(model × machine) gait store for a cached gait. With the gate off, or
    // on any miss/empty store, this returns None and the existing default path
    // runs unchanged — keeping this byte-identical to today. When a gait is
    // found, apply its scheduling substrate (the coarse profile is applied by
    // the env machinery below).
    let (profile, profile_reason) = match crate::gait::maybe_select_profile(gguf) {
        Some(gait) => {
            crate::gait::apply_selected_gait(&gait);
            (gait.profile, gait.reason)
        }
        None => requested_profile(),
    };
    let row = exact_model_row(model_path, gguf);
    let model_family = model_family(&row, gguf);
    let quant_type = quant_type(gguf);
    let thread_count = threads.unwrap_or_else(default_thread_count);
    let debug_diagnostics = matches!(profile, ExecutionProfile::Debug);
    let operator_forward_rss_timings = planner_env.flag_enabled("CAMELID_FORWARD_RSS_TIMINGS");
    let forward_rss_timings_enabled = debug_diagnostics || operator_forward_rss_timings;
    let diagnostics_status = if debug_diagnostics {
        "debug diagnostics enabled; performance claims disabled".to_string()
    } else if operator_forward_rss_timings {
        "operator-requested RSS timings enabled; performance claims disabled".to_string()
    } else {
        "standard diagnostics; RSS timings disabled by default".to_string()
    };

    let mut reasons = vec![profile_reason];
    reasons.push(format!("exact_model_row={row}"));
    reasons.push(format!("quant_type={quant_type}"));

    let mut env_updates: BTreeMap<&'static str, Option<&'static str>> = BTreeMap::new();
    if forward_rss_timings_enabled {
        env_updates.insert("CAMELID_FORWARD_RSS_TIMINGS", Some("on"));
    }

    // Routing branches on the tensor scan, not the human `quant_type` label
    // above: the label may be more specific (Q6_K, Q5_K_M, TQ1_0) than the
    // lanes it admits to, and a file with a wrong declared file_type must
    // still route by what its tensors actually are. These predicates preserve
    // the pre-label-fix branch semantics exactly: "Q8_0" meant any Q8_0
    // tensor present, "Q4_K_M" meant K-quant tensors with no Q8_0.
    let has_tensor = |t: GgufTensorType| gguf.tensors.iter().any(|tensor| tensor.tensor_type == t);
    let has_q8_0_tensors = has_tensor(GgufTensorType::Q8_0);
    let has_i2_s_tensors = has_tensor(GgufTensorType::I2S);
    let has_kquant_tensors = has_tensor(GgufTensorType::Q4K)
        || has_tensor(GgufTensorType::Q5K)
        || has_tensor(GgufTensorType::Q6K);
    let has_prism_low_bit_tensors = has_tensor(GgufTensorType::Q1_0)
        || has_tensor(GgufTensorType::Q2_0G64)
        || has_tensor(GgufTensorType::Q2_0G128)
        || has_tensor(GgufTensorType::Pq2_0);
    let prism_low_bit_tensor_mix_supported = gguf.tensors.iter().all(|tensor| {
        matches!(
            tensor.tensor_type,
            GgufTensorType::Q1_0
                | GgufTensorType::Q2_0G64
                | GgufTensorType::Q2_0G128
                | GgufTensorType::Pq2_0
                | GgufTensorType::F32
                | GgufTensorType::F16
                | GgufTensorType::BF16
        )
    });
    // ALLOW-list, not a deny-list: the Metal resident K-quant lane consumes
    // Q4_K/Q6_K super-blocks and the unquantized norm/embedding tensors that
    // sit alongside them. Anything else — including `Unknown(_)`, which is how
    // types this reader does not model (IQ2_XXS, IQ3_S, ...) parse — must keep
    // the CPU block-dot route rather than be labelled Metal-resident by
    // omission from a four-entry deny-list.
    let metal_kquant_tensor_mix_supported = gguf.tensors.iter().all(|tensor| {
        matches!(
            tensor.tensor_type,
            GgufTensorType::Q4K
                | GgufTensorType::Q5K
                | GgufTensorType::Q6K
                | GgufTensorType::Q8_0
                | GgufTensorType::F32
                | GgufTensorType::F16
                | GgufTensorType::BF16
        )
    });

    // The static Q8 support table below is intentionally platform-blind, but the
    // Bonsai promotion is lane-specific. Mirror both certified routing predicates
    // here so Safe mode, CPU fallback, or a renamed neighboring quant cannot inherit
    // an exact-row claim.
    let prism_exact_row = prism_bonsai_expected_quant(&row).is_some_and(|expected| {
        expected == quant_type.as_str()
            || (expected == "Q2_0_G128" && matches!(quant_type.as_str(), "Q2_0" | "Q2_0_G128"))
    }) && has_prism_low_bit_tensors
        && prism_low_bit_tensor_mix_supported
        && is_prism_low_bit_metal_arch(gguf);
    let prism_supported_macos = prism_exact_row
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && !matches!(profile, ExecutionProfile::Safe)
        && planner_env.flag_enabled("CAMELID_METAL_RESIDENT_DECODE")
        && !planner_env.flag_disabled("CAMELID_MAC_Q8_METAL_PLAN");
    let prism_supported_windows = prism_exact_row
        && platform.operating_system == "windows"
        && platform.architecture == "x86_64"
        && platform.cuda_resident_active
        && !matches!(profile, ExecutionProfile::Safe)
        && (gguf.architecture() != Some("qwen35")
            || !planner_env.flag_disabled("CAMELID_QWEN35_CUDA"));
    let mut support_level = if prism_supported_macos {
        "supported_exact_row_smoke_macos_metal".to_string()
    } else if prism_supported_windows {
        "supported_exact_row_smoke_windows_cuda".to_string()
    } else {
        support_level(&row, &quant_type)
    };

    let bitnet_gpu_allowed =
        platform.gpu_accel_enabled && !planner_env.flag_disabled("CAMELID_BITNET_GPU");
    let bitnet_backend = if bitnet_gpu_allowed && platform.cuda_available {
        "cuda"
    } else if bitnet_gpu_allowed && platform.metal_available {
        "metal"
    } else {
        "cpu"
    };
    let bitnet_kernel_path =
        match crate::bitnet_kernels::BitNetKernelMode::from_env().effective_cpu() {
            crate::bitnet_kernels::BitNetKernelMode::I2S => "i2_s_canonical_direct",
            crate::bitnet_kernels::BitNetKernelMode::Tl1 => "i2_s_canonical_tl1_lookup",
            crate::bitnet_kernels::BitNetKernelMode::Tl2 => "i2_s_canonical_tl2_lookup",
            crate::bitnet_kernels::BitNetKernelMode::Auto => unreachable!("auto resolved"),
        };

    let (
        selected_backend,
        selected_q8_path,
        prefill_path,
        prefill_runtime_policy,
        decode_path,
        fallback_path,
    ) = if crate::model::is_bitnet_embedding_model(gguf) && has_i2_s_tensors {
        reasons.push(format!(
            "exact Microsoft BitNet embedding GGUF: decoder-only qwen3/gemma3 graph with \
             canonical I2_S projections executes through the experimental {bitnet_backend} \
             cleanroom kernel; GPU dispatch falls back to the CPU oracle and generative \
             endpoints fail closed"
        ));
        (
            match bitnet_backend {
                "cuda" => "bitnet_embedding_cuda_runtime",
                "metal" => "bitnet_embedding_metal_runtime",
                _ => "bitnet_embedding_cpu_runtime",
            },
            bitnet_kernel_path,
            match bitnet_backend {
                "cuda" => "bitnet_embedding_cuda_full_sequence",
                "metal" => "bitnet_embedding_metal_full_sequence",
                _ => "bitnet_embedding_cpu_full_sequence",
            },
            "experimental_exact_artifact_geometry",
            "mean_pool_l2_normalize",
            "no_generation_fallback",
        )
    } else if gguf.architecture() == Some("bitnet-b1.58") && has_i2_s_tensors {
        reasons.push(format!(
            "BitNet-b1.58 canonical I2_S row: causal SubLN graph executes through the \
             experimental {bitnet_backend} cleanroom kernel with CPU fallback"
        ));
        (
            match bitnet_backend {
                "cuda" => "bitnet_runnable_cuda_runtime",
                "metal" => "bitnet_runnable_metal_runtime",
                _ => "bitnet_runnable_cpu_runtime",
            },
            bitnet_kernel_path,
            match bitnet_backend {
                "cuda" => "bitnet_runnable_cuda_prefill",
                "metal" => "bitnet_runnable_metal_prefill",
                _ => "bitnet_runnable_cpu_prefill",
            },
            "experimental_cleanroom_graph",
            match bitnet_backend {
                "cuda" => "bitnet_runnable_cuda_decode",
                "metal" => "bitnet_runnable_metal_decode",
                _ => "bitnet_runnable_cpu_decode",
            },
            "bitnet_cleanroom_cpu_fallback",
        )
    } else if gguf.architecture() == Some("gemma4") {
        // gemma4 rows are served by their OWN runtime (`Gemma4ServeRuntime`), not
        // by the generic dense engine, so the generic Q8/K-quant arms below would
        // describe a lane this row never takes. Phase 0 of the gemma3→CUDA
        // campaign measured exactly that: the plan disclosed
        // `cuda_resident_kquant_runtime` / `kquant_cuda_resident_decode` while
        // serve ran the CPU `Gemma4Runtime` — 107 MiB of VRAM in use while a
        // 2.83 GB model generated. Keyed on the same predicate the load site
        // uses, so the disclosure follows the lane.
        //
        // The FULL admission decision — policy, quant AND fit — because the load
        // site calls this same function. Disclosing policy alone here would say
        // "CUDA" for a row that then falls back on quant or fit, which is the
        // Phase 0 defect wearing different clothes.
        let admitted = gemma4_cuda_lane_admitted(gguf);
        match &admitted {
            Ok(()) => reasons.push(
                "gemma4 row on a CUDA-resident host: the gemma4 CUDA-resident engine drives \
                 decode (Q8_0/Q4_0/Q4_1 layer projections, and the row fits VRAM with headroom)"
                    .into(),
            ),
            Err(why) => reasons.push(format!(
                "gemma4 row: the gemma4 CPU runtime drives decode — CUDA-resident lane declined: \
                 {why}"
            )),
        }
        gemma4_plan(admitted.is_ok())
    } else if has_prism_low_bit_tensors
        && prism_low_bit_tensor_mix_supported
        && is_prism_low_bit_metal_arch(gguf)
        && platform.operating_system == "windows"
        && platform.architecture == "x86_64"
        && platform.cuda_resident_active
        && !matches!(profile, ExecutionProfile::Safe)
        && (gguf.architecture() != Some("qwen35")
            || !planner_env.flag_disabled("CAMELID_QWEN35_CUDA"))
    {
        if gguf.architecture() == Some("qwen35") {
            env_updates.insert("CAMELID_QWEN35_CUDA", Some("on"));
        }
        reasons.push(
            "Prism packed low-bit Windows CUDA lane selected; Q1_0/Q2_0 wire blocks remain \
             packed and execute in the Qwen/Qwen3.5 resident CUDA graph. Q1_0 Qwen3.5 prompts \
             use up to 128-token tensor-core prefill with Bonsai-specific D128 recurrent \
             kernels; greedy text and image decode keep token lookup and RoPE selection \
             on-device. CAMELID_PRISM_CUDA_STRICT=1 retains the exact arithmetic lane. \
             Oversized rows stream a capacity-planned suffix from pinned host RAM"
                .into(),
        );
        (
            "cuda_resident_prism_low_bit_runtime",
            "cuda_resident_prism_packed_wire",
            "prism_low_bit_cuda_resident_prefill",
            "prism_q1_tensor_core_cuda_prefill",
            "prism_low_bit_cuda_resident_decode",
            "scalar_prism_block_decode_fallback",
        )
    } else if has_prism_low_bit_tensors
        && prism_low_bit_tensor_mix_supported
        && is_prism_low_bit_metal_arch(gguf)
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && !matches!(profile, ExecutionProfile::Safe)
        && planner_env.flag_enabled("CAMELID_METAL_RESIDENT_DECODE")
        && !planner_env.flag_disabled("CAMELID_MAC_Q8_METAL_PLAN")
    {
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
        env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
        reasons.push(
            "Prism packed low-bit Metal lane selected; Q1_0/Q2_0 wire blocks remain packed \
             in unified memory and execute directly in the resident Metal kernels"
                .into(),
        );
        (
            "metal_resident_prism_low_bit_runtime",
            "metal_resident_prism_packed_wire",
            "prism_low_bit_metal_resident_prefill",
            "resident_single_command_buffer_prefill",
            "prism_low_bit_metal_resident_decode",
            "scalar_prism_block_decode_fallback",
        )
    } else if has_q8_0_tensors
        && is_lfm2_metal_arch(gguf)
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && lfm2_metal_policy_allows(&profile, planner_env.flag_disabled("CAMELID_LFM2_METAL"))
    {
        // The lfm2 resident Metal graph. Without this arm lfm2 fell through every
        // arm below to the safe path and `/v1/health` reported `cpu_reference` /
        // `safe_cpu_decode` WHILE the Metal graph was serving. The gate reads only
        // operator inputs -- never a plan-written variable, which is the latch that
        // previously made a plan opt itself out on the second load. Default-on with
        // a `=0` opt-out, matching `lfm2_metal_enabled` exactly -- if these two ever
        // disagree, health reports a lane other than the one that ran.
        reasons.push(
            "lfm2 resident Metal lane selected; Q8_0 projections stay resident and the              short-conv ring plus the sparse KV cache live on device"
                .into(),
        );
        (
            "metal_resident_lfm2_runtime",
            "metal_resident_q8_wire",
            "lfm2_metal_resident_prefill",
            "resident_tiled_mm_prefill",
            "lfm2_metal_resident_decode",
            "runnable_cpu_decode_fallback",
        )
    } else if has_q8_0_tensors
        && !has_kquant_tensors
        && metal_kquant_tensor_mix_supported
        && gguf.architecture() == Some("qwen35")
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && !planner_env.flag_disabled("CAMELID_QWEN35_METAL")
    {
        // The qwen35 resident Metal graph, reachable for Q8_0 rows since the
        // loader admission. Without this arm the ornith Q8_0 row fell through
        // every arm below to the final else and `/v1/health` reported
        // `cpu_reference`/`safe_cpu_decode` WHILE the Metal graph was serving —
        // the third recurrence of the fell-through-while-resident defect (Q1/Q2
        // Bonsai, lfm2, now qwen35 Q8_0). The gate reads exactly what the
        // routing reads (`qwen35_metal_enabled`: arch, macOS Metal, the env
        // opt-out via the shared `flag_value_disabled` vocabulary) and
        // deliberately NOT the Safe profile or any plan-written variable:
        // `generate_qwen35_streaming` consults neither, so gating on them here
        // would disclose a lane other than the one that runs.
        reasons.push(
            "qwen35 resident Metal lane selected; Q8_0 wire weights stay resident and \
             attention, gated-delta recurrence, FFN, logits and greedy sampling run on \
             device (opt-out CAMELID_QWEN35_METAL=0; CPU hybrid on resident-build error)"
                .into(),
        );
        (
            "metal_resident_qwen35_runtime",
            "metal_resident_q8_wire",
            "qwen35_metal_resident_prefill",
            "resident_batched_prefill",
            "qwen35_metal_resident_decode",
            "runnable_cpu_decode_fallback",
        )
    } else if has_q8_0_tensors && is_supported_exact_q8_row(&row, gguf) {
        if platform.operating_system == "macos" && platform.architecture == "aarch64" {
            select_macos_q8_plan(
                &profile,
                &platform,
                planner_env,
                is_windowed_attention_arch(gguf),
                &mut env_updates,
                &mut reasons,
            )
        } else if is_windowed_attention_arch(gguf)
            && platform.cuda_resident_active
            && windowed_arch_cuda_resident_plan_selectable()
        {
            // gemma3 on a CUDA host: the CUDA resident engine now carries the
            // windowed forward (per-layer dual-theta RoPE, `attention_decode_sw`,
            // sandwich post-norms, GeGLU, embed scale), so this row has a second
            // validated GPU lane alongside Metal. The CPU dense paths still have
            // no window mask and still fail closed at forward dispatch (H4) —
            // that has not changed and must not.
            reasons.push(
                "windowed-attention row (gemma3) on a CUDA-resident host: the GPU-resident \
                 windowed forward drives decode; prefill is token-by-token (the batched and \
                 flash kernels carry no window)"
                    .into(),
            );
            cuda_resident_windowed_plan()
        } else if is_windowed_attention_arch(gguf) {
            // gemma3 (windowed attention) with NO resident GPU lane available on
            // this host: the CPU dense paths (x86 repack included) have no
            // sliding-window mask and fail closed at forward dispatch (hazard
            // H4), so `camelid serve` routes gemma3 chat through the runnable
            // bridge; the plan must not advertise a CPU dense lane it can never
            // run.
            reasons.push(
                "windowed-attention row (gemma3): no resident GPU lane is active on this host \
                 (Metal-resident on macOS, CUDA-resident on a CUDA host); no CPU dense plan \
                 exists for this arch — serve chats via the runnable bridge on this host; \
                 failing closed to safe path"
                    .into(),
            );
            safe_q8_plan()
        } else if platform.architecture == "x86_64"
            && (platform.operating_system == "linux" || platform.operating_system == "windows")
        {
            // The x86_64 Q8 runtime-repack + AVX2 packed-rows4 path is platform-agnostic
            // Rust (no OS-specific kernels) and is parity-validated bit-identical to the
            // scalar reference on Windows as well as Linux, so both share this plan.
            select_x86_q8_plan(
                &profile,
                &platform,
                planner_env,
                &mut env_updates,
                &mut reasons,
            )
        } else {
            reasons.push(
                    "no validated platform-specific Q8_0 plan for this OS/arch; failing closed to safe path"
                        .into(),
                );
            safe_q8_plan()
        }
    } else if has_q8_0_tensors
        && platform.cuda_resident_active
        && is_gpu_runnable_arch(gguf)
        && !planner_env.flag_disabled("CAMELID_GPU_RUNNABLE_TIER")
    {
        // On by DEFAULT: an uncurated but architecturally-compatible Q8_0 model should just run
        // on the GPU without the user having to opt in. This is safe because admission is gated
        // at runtime by the GPU-vs-CPU parity self-check — a model that is not bit-exact falls
        // back to the CPU reference path. Opt out with CAMELID_GPU_RUNNABLE_TIER=0 (forces CPU).
        reasons.push(
            "non-curated Q8_0 on a resident-capable dense arch: GPU-runnable tier (NOT a \
             supported row) — resident path admitted subject to the runtime parity self-check"
                .into(),
        );
        cuda_resident_q8_runnable_plan()
    } else if has_kquant_tensors
        && metal_kquant_tensor_mix_supported
        && gguf.architecture() == Some("qwen35")
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && !planner_env.flag_disabled("CAMELID_QWEN35_METAL")
    {
        // qwen35 K-quant on the resident Metal graph. Sibling of the Q8_0 arm above —
        // that one keys on `has_q8_0_tensors`, which is FALSE for a Q4_K_M file, so
        // without this arm an ornith Q4_K_M fell through to `select_kquant_plan` and
        // `/v1/health` disclosed `cpu_kquant_block_dot` / `kquant_cpu_block_dot_decode`
        // WHILE the resident Metal graph served it. Verified live before this arm
        // existed. That reason string also asserts "no f32 materialization", which is a
        // specific technical claim about a lane that was not running.
        //
        // `metal_kquant_tensor_mix_supported` is doing real work here, not decoration:
        // routing admits a tensor iff `runnable::model::resident_metal_format` maps it,
        // and a q5_K tensor (ornith Q3_K_M) maps to None -> `prism_metal_weight` errors
        // -> the whole graph declines to the CPU hybrid. The mix predicate excludes
        // exactly Q5_K/Q3_K/Q2_K, so the two agree file-for-file. Gating on bare
        // `has_kquant_tensors` would over-disclose Metal for Q3_K_M — the same defect,
        // sign-flipped.
        //
        // Deliberately NOT gated on the Safe profile or any plan-written variable:
        // `generate_qwen35_streaming` consults neither, and gating on them would
        // reintroduce the latch defect.
        reasons.push(
            "qwen35 K-quant resident Metal lane selected; Q4_K/Q5_K/Q6_K super-blocks \
             (plus any Q8_0 or bf16 tensors the quant recipe kept) stay resident at wire \
             size and attention, gated-delta recurrence, FFN and logits run on device \
             (opt-out CAMELID_QWEN35_METAL=0; CPU hybrid on resident-build error)"
                .into(),
        );
        (
            "metal_resident_qwen35_kquant_runtime",
            "metal_resident_kquant_wire",
            "qwen35_metal_resident_prefill",
            "resident_batched_prefill",
            "qwen35_metal_resident_decode",
            "runnable_cpu_decode_fallback",
        )
    } else if !has_q8_0_tensors && has_kquant_tensors {
        select_kquant_plan(
            &profile,
            &platform,
            planner_env,
            // The runtime's `resident_decode_eligible` rejects architectures the
            // resident dense kernels cannot express (gemma2/gemma3 sandwich
            // norms, NoPE, MoE routing). Mirror that here so the plan never
            // advertises a Metal-resident lane for a model that provably runs on
            // the CPU block-dot path.
            metal_kquant_tensor_mix_supported && is_gpu_runnable_arch(gguf),
            &mut reasons,
        )
    } else {
        reasons.push("non-validated row or quant; failing closed to safe path".into());
        (
            "cpu_reference",
            "safe_dense_or_q8_cpu",
            "safe_cpu_prefill",
            "always_retained_reference_path",
            "safe_cpu_decode",
            "safe_cpu_reference_path",
        )
    };

    // LFM2.5-2.6B is supported only on the two lanes that carry receipts. Its
    // row-table entry is recognition-only so a Linux host, a neighboring Mac,
    // Safe mode, or a CPU fallback cannot inherit the platform-blind claim.
    // Evaluate this after route selection: the Mac receipt is specifically for
    // the resident backend/prefill/decode labels below, not merely for a host
    // on which Metal happens to be present.
    if is_lfm2_5_2_6b_exact_row(&row) {
        support_level = if lfm2_selected_lane_supported(
            &profile,
            &platform,
            model_path,
            gguf,
            &quant_type,
            selected_backend,
            prefill_path,
            decode_path,
        ) {
            "supported_exact_row_smoke".into()
        } else {
            "unknown_or_unvalidated".into()
        };
    }
    if is_phi3_mini_4k_exact_row(&row) {
        support_level = if phi3_selected_lane_supported(
            &profile,
            &platform,
            model_path,
            gguf,
            &quant_type,
            selected_backend,
            prefill_path,
            decode_path,
        ) {
            "supported_exact_row_smoke".into()
        } else {
            "unknown_or_unvalidated".into()
        };
    }
    // Preserve the stable reason ordering (profile, row, quant, support, lane)
    // even though platform-scoped support can only be decided after lane selection.
    reasons.insert(3, format!("support_level={support_level}"));

    let plan = ExecutionPlan {
        profile,
        operating_system: platform.operating_system,
        architecture: platform.architecture,
        platform_label: platform.platform_label,
        cpu_model: platform.cpu_model,
        cpu_features: platform.cpu_features,
        model_family,
        quant_type,
        exact_model_row: row,
        support_level,
        selected_backend: selected_backend.to_string(),
        selected_q8_path: selected_q8_path.to_string(),
        prefill_path: prefill_path.to_string(),
        prefill_runtime_policy: prefill_runtime_policy.to_string(),
        decode_path: decode_path.to_string(),
        thread_count,
        diagnostics_status,
        fallback_path: fallback_path.to_string(),
        cuda_resident_active: platform.cuda_resident_active,
        reasons,
    };
    ExecutionPlanOutcome { plan, env_updates }
}

/// Whether the macOS Q8 Metal-resident PLAN selection can fire in this process
/// at all, independent of the model and of Metal device presence: not the Safe
/// profile (including the fail-closed-to-Safe parse of an unrecognized
/// `CAMELID_PROFILE`), and `CAMELID_MAC_Q8_METAL_PLAN` not opted out. These are
/// the OPERATOR opt-outs among [`select_macos_q8_plan`]'s early returns.
///
/// `CAMELID_MAC_Q8_REPACK` is deliberately NOT consulted even though it is a
/// third early return, because it is a MANAGED_ENV_KEY that the plan itself
/// WRITES: a successful Metal-resident selection sets it to "off", which
/// `env_flag_disabled` reads as disabled. Feeding it back into routing would be
/// a self-defeating latch — the moment `PlannerEnv::apply` ran, routing would
/// decide the Metal plan was unselectable and send gemma3 to the bridge,
/// killing the very lane the plan had just selected. Residual, recorded rather
/// than fixed here: an operator who PRE-sets `CAMELID_MAC_Q8_REPACK=0` still
/// gets a safe plan with resident routing. Closing that needs the plan to stop
/// overloading one variable for both "operator opt-out" and "plan output",
/// which is a wider change than this review.
///
/// Phase 3c: `inference::windowed_arch_resident_host_available` consults this
/// so gemma3's ROUTING and the disclosed execution plan agree. Before, an
/// operator running `CAMELID_PROFILE=safe` or `CAMELID_MAC_Q8_METAL_PLAN=0` got
/// a plan advertising `cpu_reference` / `safe_cpu_decode` (with the windowed
/// arm's reason string literally saying "serve chats via the runnable bridge")
/// while serve ran the Metal-resident lane anyway — the plan is pure disclosure
/// and cannot disarm the runtime gate, so the disclosure was simply wrong.
///
/// Non-windowed archs are unaffected: for them the plan opting out of the Metal
/// selection genuinely does leave a correct CPU lane, which is the point of the
/// opt-out.
pub fn macos_q8_metal_plan_selectable() -> bool {
    !matches!(requested_profile().0, ExecutionProfile::Safe)
        && !env_flag_disabled("CAMELID_MAC_Q8_METAL_PLAN")
}

/// The CUDA-resident twin of [`macos_q8_metal_plan_selectable`]: whether a
/// windowed-attention row's CUDA-resident PLAN selection can fire in this
/// process, considering only operator INPUTS.
///
/// `inference::windowed_arch_resident_host_available` consults this so gemma3's
/// routing and its disclosed plan agree on a CUDA host, exactly as Phase 3c of
/// the Metal campaign made them agree on a Metal host. The D20 invariant that
/// motivated it applies verbatim here: an operator who asks for the safe profile
/// must get the runnable bridge, not a plan saying "bridge" while serve silently
/// runs the GPU.
///
/// Deliberately consults NO plan output. `CAMELID_MAC_Q8_REPACK` is the recorded
/// counter-example on the Metal side — a variable that is both an operator
/// opt-out and something the plan writes, and therefore unusable as a routing
/// input. Nothing in this predicate is written by `PlannerEnv::apply`.
pub fn windowed_arch_cuda_resident_plan_selectable() -> bool {
    !matches!(requested_profile().0, ExecutionProfile::Safe)
        && !env_flag_disabled("CAMELID_GEMMA3_CUDA_RESIDENT")
}

/// Whether the gemma4 serve lane should run on the CUDA-resident engine.
///
/// **Default ON** where a CUDA device is actually driving this process; opt out
/// with `CAMELID_GEMMA4_CUDA=0` (0/off/false/no/disabled). It used to be
/// opt-IN (`CAMELID_GEMMA4_CUDA=1`), which meant every gemma4 row decoded on the
/// CPU out of the box on a Windows/Linux CUDA host even though the resident
/// engine was present and working.
///
/// This is the POLICY half only. It says nothing about whether a given file
/// FITS — that is a separate, per-file check at the load site
/// (`gemma4_cuda_fit_check`), because policy is host-wide while fit is not.
///
/// Single source of truth: `api::gemma4_cuda_enabled` delegates here rather than
/// reading the env itself, so the disclosed execution plan and the lane serve
/// actually loads cannot disagree. That split is what produced the Phase 0
/// finding — the plan advertised `cuda_resident_kquant_runtime` while serve ran
/// the CPU runtime, because the two consulted different things.
pub fn gemma4_cuda_lane_selectable() -> bool {
    if matches!(requested_profile().0, ExecutionProfile::Safe) {
        return false;
    }
    if env_flag_disabled("CAMELID_GEMMA4_CUDA") {
        return false;
    }
    cuda_resident_decode_will_run()
}

/// Whether the qwen35 serve lane should run on the CUDA-resident engine.
///
/// **Default ON** where a CUDA device is actually driving this process; opt out
/// with `CAMELID_QWEN35_CUDA=0` (0/off/false/no/disabled). It used to be opt-IN
/// for every row except Prism low-bit on Windows, which is the gemma4 Phase 0
/// finding wearing different clothes: the certified Ornith K-quant rows decoded
/// on the CPU out of the box on a CUDA host — `select_kquant_plan` advertised
/// `cuda_resident_kquant_runtime` off `platform.cuda_resident_active` while
/// `generate_qwen35_streaming` read `CAMELID_QWEN35_CUDA` itself and fell to the
/// CPU runnable lane, because the two consulted different things.
///
/// This is the POLICY half only. It says nothing about whether a given file
/// FITS: capacity is enforced per request by `qwen35_generation_budget` against
/// `CAMELID_QWEN35_CUDA_MAXPOS`, and every CUDA error raised before a token is
/// emitted falls back to the CPU oracle (`qwen35_cuda_with_cpu_fallback`).
///
/// `--gpu off` and the UI's live toggle stay authoritative: they clear
/// `gpu_accel_enabled`, which `cuda_resident_decode_will_run` requires. An
/// explicit `CAMELID_QWEN35_CUDA=1` is an opt-IN to this lane, never an override
/// of that master switch.
///
/// The default flips only on Windows, which is where the qwen35 CUDA lane
/// carries receipts (the Ornith Q4_K_M parity + agent-eval PASS receipts, and
/// the Prism Windows CUDA branch this predicate replaces). Other CUDA hosts keep
/// the historical opt-in until the same evidence exists for them, so this change
/// cannot alter a platform it has not been measured on.
pub fn qwen35_cuda_lane_selectable() -> bool {
    if matches!(requested_profile().0, ExecutionProfile::Safe) {
        return false;
    }
    if env_flag_disabled("CAMELID_QWEN35_CUDA") {
        return false;
    }
    if !cfg!(windows) && !env_flag_enabled("CAMELID_QWEN35_CUDA") {
        return false;
    }
    cuda_resident_decode_will_run()
}

/// Everything the gemma4 CUDA-resident lane puts in VRAM BESIDES the per-layer
/// projections: the small per-layer norms, the f16 KV cache at the load site's
/// 4096-position window, the GPU tied head, the GPU PLE context projection, and
/// the per-token scratch.
///
/// Calibrated against a measurement rather than guessed. On an RTX 3060,
/// `gemma-4-E2B-it-Q8_0` uploads 1879 MiB of layer projections and settles at
/// 2635 MiB of device memory including the ~107 MiB CUDA context — so the
/// non-projection residency is ~649 MiB. 1024 MiB keeps the projection an
/// over-estimate (~15% for E2B) without being so loose that it stops predicting.
const GEMMA4_RESIDENT_OVERHEAD_MIB: u64 = 1024;

/// The tensors the gemma4 CUDA-resident lane actually uploads per block. The
/// per-layer EMBEDDING tables (`per_layer_token_embd`, `per_layer_model_proj`)
/// and the token embedding / tied head stay on the host or are handled
/// separately, which is why a whole-file byte count is the wrong basis for the
/// fit decision — see `gemma4_cuda_resident_bytes`.
fn is_gemma4_layer_projection(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((_, tail)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        tail,
        "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "ffn_gate.weight"
            | "ffn_up.weight"
            | "ffn_down.weight"
            | "proj.weight"
            | "inp_gate.weight"
    )
}

/// The gemma4 per-block tensors the CUDA-resident lane runs a QUANTIZED GEMV over —
/// the set `gemma4_runtime::nvfp4_cuda_lane_check` format-checks at load.
///
/// **Deliberately NOT `is_gemma4_layer_projection`, and the two must not be merged.**
/// That one answers "what occupies VRAM" and so counts `proj.weight` / `inp_gate.weight`
/// — the per-layer PLE matrices, which the CPU runtime DEQUANTIZES to f32 before upload.
/// Their wire format is therefore irrelevant to lane coverage, and folding them in here
/// would decline the E2B Q4_0 row on its F32 PLE tensors while admitting nothing extra.
/// Conversely this set carries the MoE arms (`ffn_gate_up_exps` / `ffn_down_exps`, the
/// A4B/26B rows), which the lane GEMVs through the same `GemmaLayerQuant` dispatch but
/// which the VRAM projection accounts for separately.
///
/// Selected by name rather than by scanning every tensor's type: a whole-file type scan
/// sweeps in the F32 norms, the `token_embd` head (own lane, own CPU fallback) and the
/// PLE tables. Judging admission on those is precisely what let an E4B Q4_K_M row
/// through on the strength of its Q8_0 `token_embd` while every projection it decodes
/// is Q4_K.
fn gemma4_projection_tensors(gguf: &GgufFile) -> impl Iterator<Item = &GgufTensorDescriptor> {
    const PROJECTION_SUFFIXES: [&str; 9] = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "ffn_gate_up_exps.weight",
        "ffn_down_exps.weight",
    ];
    gguf.tensors.iter().filter(|t| {
        t.name.starts_with("blk.")
            && PROJECTION_SUFFIXES
                .iter()
                .any(|suffix| t.name.ends_with(suffix))
    })
}

/// Projected device bytes for a gemma4 row on the CUDA-resident lane.
///
/// Sums only the per-layer projections and adds
/// [`GEMMA4_RESIDENT_OVERHEAD_MIB`]. It deliberately does NOT sum the whole
/// file: gemma4 is a PLE matformer whose per-layer embedding tables dwarf its
/// projections, and counting them made the fit check reject rows that fit
/// comfortably. Measured on an RTX 3060 with `gemma-4-E2B-it-Q8_0`: whole-file
/// accounting projected 5055 MiB and DECLINED the row, while the lane actually
/// used 2635 MiB and served it in 794 ms. Over-conservative fit checks are not
/// "safe" — they silently keep working hardware on the CPU.
fn gemma4_cuda_resident_bytes(gguf: &GgufFile) -> u64 {
    let projections: u64 = gguf
        .tensors
        .iter()
        .filter(|t| is_gemma4_layer_projection(&t.name))
        .map(|t| t.n_bytes)
        .sum();
    projections + GEMMA4_RESIDENT_OVERHEAD_MIB * 1024 * 1024
}

/// FULL admission decision for the gemma4 CUDA-resident lane: host policy, then
/// the per-file quant check, then the per-file VRAM fit. `Ok(())` means the lane
/// will actually load; `Err(reason)` is the operator-facing reason it will not.
///
/// **This is the single predicate.** Both the disclosed execution plan and the
/// serve load site call it, so `/v1/health` cannot advertise a lane that serve
/// then declines. An earlier revision of this campaign split them — policy in the
/// plan, quant+fit at the load site — and immediately reproduced the Phase 0
/// defect it was written to fix: the plan said `gemma4_cuda_resident_runtime`
/// while a declined row served on the CPU. DECISIONS D20, restated: a disclosure
/// that is not derived from the same decision the runtime makes is not a
/// disclosure.
pub fn gemma4_cuda_lane_admitted(gguf: &GgufFile) -> Result<(), String> {
    if !gemma4_cuda_lane_selectable() {
        return Err(
            "no CUDA-resident host for the gemma4 lane (CAMELID_GEMMA4_CUDA=0, safe profile, \
             deterministic mode, or no usable device)"
                .into(),
        );
    }
    gemma4_projection_quant_admitted(gguf)?;
    // FIT. Projected from the per-layer projections plus a calibrated overhead,
    // NOT from the whole file — see `gemma4_cuda_resident_bytes`. A row that
    // genuinely does not fit falls back rather than allocating into a mid-load
    // OOM; a row that does fit must not be talked out of the GPU by a pessimistic
    // estimate, which is the failure this projection was rewritten to avoid.
    //
    // This check is advisory, not the last line of defence: the load site also
    // falls back to the CPU runtime if `Gemma4CudaResident::load` returns an
    // error, so an under-estimate here degrades to a slower lane rather than a
    // failed request.
    let Some(free) = crate::cuda_vram::free_vram_bytes() else {
        return Ok(());
    };
    let projected = gemma4_cuda_resident_bytes(gguf);
    crate::cuda_vram::evaluate(free, projected, crate::cuda_vram::min_headroom_mib())
        .map(|_| ())
        .map_err(|short| short.to_string())
}

/// The QUANT half of gemma4 CUDA admission: every layer projection the lane GEMVs
/// must be in a format that carries an end-to-end greedy-parity receipt against the
/// CPU gemma4 runtime. Split out from [`gemma4_cuda_lane_admitted`] so it is testable
/// without a CUDA host (that function's first act is a host-policy probe).
///
/// This gate has been wrong in BOTH directions, so the reasoning is recorded here.
/// It used to pin to "any Q8_0 tensor", after `gemma-4-E2B-it-Q4_0` was measured
/// decoding "passe dép oficialmenteynam shalthapp lenghtynam" where the CPU runtime
/// said "Paris". The Q4_0 projections were never the defect: the lane's tied HEAD
/// uploaded raw GGUF wire into `q4k_gemv` / `q6k_gemv`, which read a swizzled /
/// 224 B-padded layout (see `gemma4_runtime::gemma4_head_upload`). A Q4_0 export
/// carries a Q4_K `token_embd`, so that row's logits were formed from wrongly-paired
/// nibbles. Fixed and re-measured — 5/5 prompts token-identical, CUDA vs CPU,
/// `qa/gemma3-cuda/phase5/` — which is what admits Q4_0/Q4_1 here.
///
/// Still declined, deliberately:
/// - **NVFP4.** `nvfp4_cuda_lane_check` covers it and `nvfp4_gemv` has a KERNEL
///   parity test, but no gemma4 ROW has an end-to-end receipt on this lane. Kernel
///   parity is not row parity — that gap is exactly what the Q4_0 mis-decode was
///   (`q4_0_gemv` had a passing kernel parity test the whole time it shipped garbage).
/// - **K-quant projections (Q4_K/Q5_K).** The CPU wire lane serves them and
///   `nvfp4_cuda_lane_check` refuses them at load. Declining HERE too is what makes
///   the disclosure honest: an E4B Q4_K_M row has a Q8_0 `token_embd`, so the old
///   any-Q8_0-tensor test admitted it, the plan printed `gemma4_cuda_resident_runtime`,
///   and the load site then refused and served on the CPU — the D20 defect this
///   predicate exists to prevent.
/// - **Q2_K.** Declined on a NEGATIVE receipt, not merely a missing one:
///   `gemma-4-26B-mixq2k-it` requantised the routed experts to Q2_K and decoded
///   degenerate output on all three A/B prompts, at 14.89 tok/s (`qa/perf/HANDOFF.md`
///   §5). Note `gemma4_projection_tensors` includes `ffn_gate_up_exps`/`ffn_down_exps`,
///   so admitting Q2_K here would admit exactly the tensors that were measured bad.
///
/// **Q6_K is the one K-quant admitted**, because the 26B-A4B ghost row forces the
/// question rather than because the format is trusted in general.
/// `google_gemma-4-26B-A4B-it-Q4_0.gguf` is a MIXED export: 14 of its 30
/// `attn_q.weight` tensors are Q6_K and the other 16 are Q4_0, `ffn_down` is 27×Q4_0
/// plus 3×Q4_1, and `ffn_down_exps` is 23×Q4_0 plus 7×Q4_1. Declining Q6_K declines
/// the row this lane exists to serve. The admission is scoped to that evidence: it
/// says Q6_K appears in a receipted row, NOT that a Q6_K-throughout row is receipted.
fn gemma4_projection_quant_admitted(gguf: &GgufFile) -> Result<(), String> {
    const RECEIPTED_PROJECTION_FORMATS: [GgufTensorType; 4] = [
        GgufTensorType::Q8_0,
        GgufTensorType::Q4_0,
        GgufTensorType::Q4_1,
        GgufTensorType::Q6K,
    ];
    match gemma4_projection_tensors(gguf)
        .find(|t| !RECEIPTED_PROJECTION_FORMATS.contains(&t.tensor_type))
    {
        Some(t) => Err(format!(
            "gemma4 CUDA-resident decode has a greedy-parity receipt for Q8_0/Q4_0/Q4_1/Q6_K layer \
             projections; this row is {} and carries {} as {:?}, which has no parity receipt on \
             this lane (the CPU gemma4 runtime serves it correctly)",
            quant_type(gguf),
            t.name,
            t.tensor_type
        )),
        None => Ok(()),
    }
}

/// Plan labels for a gemma4 row and the lane that will actually serve it.
/// `cuda` distinguishes the CUDA-resident gemma4 engine from the CPU runtime;
/// neither is the generic dense CUDA lane, so neither reuses its labels.
fn gemma4_plan(
    cuda: bool,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if cuda {
        (
            "gemma4_cuda_resident_runtime",
            "gemma4_resident_wire",
            "gemma4_cuda_resident_prefill",
            "resident_prefix_cached_prefill",
            "gemma4_cuda_resident_decode",
            "gemma4_cpu_runtime_fallback_path",
        )
    } else {
        (
            "gemma4_cpu_runtime",
            "gemma4_cpu_wire",
            "gemma4_cpu_prefill",
            "always_retained_reference_path",
            "gemma4_cpu_decode",
            "gemma4_cpu_runtime_fallback_path",
        )
    }
}

/// Plan labels for a windowed-attention (gemma3) row served by the CUDA-resident
/// engine. Distinct strings from the Llama-family CUDA labels so a receipt or a
/// `/v1/health` reader can tell the windowed lane apart from the dense one —
/// they are different forwards (per-layer dual-theta RoPE, a sliding-window
/// attention kernel, sandwich post-norms, GeGLU) and conflating them in the
/// disclosure would hide which code actually ran.
fn cuda_resident_windowed_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cuda_resident_windowed_runtime",
        "cuda_resident_q8_0_wire",
        "windowed_token_by_token_resident_prefill",
        "resident_token_by_token_prefill",
        "q8_0_cuda_resident_windowed_decode",
        "runnable_bridge_fallback_path",
    )
}

fn select_macos_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    planner_env: &PlannerEnv,
    windowed_attention_arch: bool,
    env_updates: &mut BTreeMap<&'static str, Option<&'static str>>,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    // The windowed-arch reason is pushed alongside every early return below,
    // not only the one after the Metal arm: `safe_q8_plan()` names CPU dense
    // paths that a windowed arch can never run (H4), so the disclosure must
    // say where those chats actually go. Phase 3c — under `CAMELID_PROFILE=
    // safe` this previously disclosed a bare "safe profile" reason and the
    // reader had no way to tell gemma3 was being served by the bridge.
    let windowed_bridge_reason = || {
        "windowed-attention row (gemma3) without an active Metal-resident selection: no CPU \
         dense plan exists for this arch — serve chats via the runnable bridge; failing \
         closed to safe path"
            .to_string()
    };
    if matches!(profile, ExecutionProfile::Safe) {
        reasons.push("safe profile selected; optimized Mac Q8 paths disabled".into());
        if windowed_attention_arch {
            reasons.push(windowed_bridge_reason());
        }
        return safe_q8_plan();
    }
    // Baseline, not live env: a previous load's Metal-resident selection wrote
    // `CAMELID_MAC_Q8_REPACK=off` as its OUTPUT, and reading that back here is
    // what made every model loaded after the first disclose `cpu_reference`.
    if planner_env.flag_disabled("CAMELID_MAC_Q8_REPACK") {
        reasons
            .push("CAMELID_MAC_Q8_REPACK disables Mac repack; failing closed to safe path".into());
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
        if windowed_attention_arch {
            reasons.push(windowed_bridge_reason());
        }
        return safe_q8_plan();
    }

    // The Metal-resident Q8_0 stack outranks the CPU repack when the host can run it.
    // The GPU path requires plain RAM-resident Q8_0 blocks, which the rows4 repack
    // replaces — the two are mutually exclusive on weight storage — so selecting Metal
    // means loading plain blocks (the CPU plain-block reference path remains the
    // in-process fallback for sessions the resident gates reject). Selection requires
    // the resident-decode gate (on by default in the CLI entry; absent for embedders
    // and test suites, which keep the validated CPU plans) plus an actual Metal
    // device; CAMELID_MAC_Q8_METAL_PLAN=0 opts back into the CPU repack plan.
    // Both of these are runtime inputs the plan never writes, so the baseline
    // passes them straight through to live env — deliberately, so the disclosed
    // plan tracks an operator flipping them, and so this stays in lockstep with
    // `macos_q8_metal_plan_selectable`, which routing consults live.
    if planner_env.flag_enabled("CAMELID_METAL_RESIDENT_DECODE")
        && !planner_env.flag_disabled("CAMELID_MAC_Q8_METAL_PLAN")
        && platform.metal_available
    {
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
        env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
        reasons.push(
            "Metal resident Q8_0 stack selected (Metal device present, resident decode              enabled); weights stay plain RAM-resident Q8_0 blocks — the rows4 CPU repack              is disabled because the GPU-resident path requires the plain blocks"
                .into(),
        );
        reasons.push("parallel linear enabled by execution plan".into());
        return (
            "metal_resident_q8_runtime",
            "metal_resident_q8_0_wire",
            "q8_0_metal_resident_prefill",
            "resident_single_command_buffer_prefill",
            "q8_0_metal_resident_decode",
            "retained_q8_reference_path",
        );
    }

    // gemma3 (windowed attention): every plan below this point is a CPU dense
    // lane, which has no sliding-window mask and fails closed at forward
    // dispatch for this arch (hazard H4). When the Metal-resident selection
    // above did not fire (resident decode not armed, no Metal device, or the
    // CAMELID_MAC_Q8_METAL_PLAN opt-out), serve routes gemma3 chat through the
    // runnable bridge — advertise the safe fail-closed plan, not a CPU repack
    // lane the arch can never run.
    if windowed_attention_arch {
        reasons.push(windowed_bridge_reason());
        return safe_q8_plan();
    }

    let dotprod = has_feature(&platform.cpu_features, "dotprod");
    let i8mm = has_feature(&platform.cpu_features, "i8mm");
    if !dotprod {
        reasons.push("Apple Silicon dotprod not detected; failing closed to safe path".into());
        return safe_q8_plan();
    }

    env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
    env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("on"));
    reasons.push("validated macOS Apple Silicon Q8_0 runtime repack enabled".into());
    reasons.push("parallel linear enabled by execution plan".into());

    if planner_env.flag_disabled("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN-down decode consumer disabled".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN-down decode consumer gate enabled by default".into());
    }

    if planner_env.flag_disabled("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN gate/up decode consumer disabled".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN gate/up decode consumer gate enabled by default".into());
    }

    let prefill_i8mm_requested = !planner_env.flag_disabled("CAMELID_MAC_Q8_PREFILL_I8MM");
    let prefill_path = if i8mm && prefill_i8mm_requested {
        env_updates.insert("CAMELID_MAC_Q8_PREFILL_I8MM", Some("on"));
        reasons.push("direct-pack prefill I8MM gate enabled by default".into());
        reasons.push(format!(
            "direct-pack I8MM dispatch engages only when prefill rows >= {}",
            MAC_Q8_PREFILL_I8MM_MIN_ROWS
        ));
        if matches!(profile, ExecutionProfile::Experimental) {
            env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("packed_prefill"));
            reasons.push(
                "experimental packed prefill scheduling enabled; single-token decode remains GEMV/DOTPROD"
                    .into(),
            );
            "q8_0_experimental_packed_prefill_i8mm_available"
        } else {
            env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("off"));
            reasons.push(
                "packed prefill scheduling remains experimental and is disabled for auto profile"
                    .into(),
            );
            "q8_0_direct_pack_prefill_i8mm_available"
        }
    } else {
        env_updates.insert("CAMELID_MAC_Q8_PREFILL_I8MM", Some("off"));
        env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("off"));
        if planner_env.flag_disabled("CAMELID_MAC_Q8_PREFILL_I8MM") {
            reasons.push("CAMELID_MAC_Q8_PREFILL_I8MM disables I8MM prefill".into());
        } else {
            reasons
                .push("I8MM/MATMUL_INT8 unavailable; using packed Q8 CPU prefill fallback".into());
        }
        "q8_0_cpu_packed_prefill_fallback_available"
    };

    if matches!(profile, ExecutionProfile::Experimental) {
        reasons.push("experimental profile active; support claims remain unchanged".into());
    }
    if matches!(profile, ExecutionProfile::Debug) {
        reasons.push(
            "debug profile active; RSS timings enabled and performance claims disabled".into(),
        );
    }

    (
        "cpu_q8_runtime_repack",
        "mac_validated_q8_0_repack",
        prefill_path,
        "enabled_when_prefill_rows_gte_4",
        "q8_0_decode_gemv_dotprod",
        "retained_q8_reference_path",
    )
}

fn select_x86_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    planner_env: &PlannerEnv,
    env_updates: &mut BTreeMap<&'static str, Option<&'static str>>,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if matches!(profile, ExecutionProfile::Safe) {
        reasons.push("safe profile selected; optimized x86 Q8 paths disabled".into());
        return safe_q8_plan();
    }
    if platform.cuda_resident_active {
        // A CUDA device is driving decode: keep weights as plain RAM-resident Q8_0
        // blocks (the GPU resident engine cannot consume the CPU rows4 repack — the two
        // are mutually exclusive on weight storage). The CPU repack path only wins when
        // the CPU actually runs decode (no GPU, GPU toggled off, or deterministic mode).
        reasons.push(
            "CUDA resident decode active; GPU-resident Q8_0 engine drives decode (weights stay plain RAM-resident Q8_0 blocks — the CPU rows4 repack is disabled while the GPU drives decode)"
                .into(),
        );
        return cuda_resident_q8_plan();
    }
    if planner_env.flag_disabled("CAMELID_X86_Q8_REPACK")
        || planner_env.flag_disabled("CAMELID_X86_Q8_KERNEL")
    {
        reasons.push(
            "x86 Q8 override disables optimized kernel/repack; failing closed to safe path".into(),
        );
        if planner_env.flag_disabled("CAMELID_X86_Q8_REPACK") {
            env_updates.insert("CAMELID_X86_Q8_REPACK", Some("off"));
        }
        if planner_env.flag_disabled("CAMELID_X86_Q8_KERNEL") {
            env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        }
        return safe_q8_plan();
    }
    if let Some(invalid) = planner_env.invalid_x86_kernel_override() {
        reasons.push(format!(
            "invalid CAMELID_X86_Q8_KERNEL={invalid}; failing closed to safe path"
        ));
        env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        return safe_q8_plan();
    }
    if !has_feature(&platform.cpu_features, "avx2") {
        reasons.push(
            "AVX2 feature not detected for x86 Q8 kernel; failing closed to safe path".into(),
        );
        return safe_q8_plan();
    }

    env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
    env_updates.insert("CAMELID_X86_Q8_REPACK", Some("on"));
    env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("avx2"));
    let optional_x86_q8_gate = |name| {
        if planner_env.flag_disabled(name) {
            Some("off")
        } else {
            Some("on")
        }
    };
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL"),
    );
    // Serial packed decode is the validated Linux default, but on Windows the parallel
    // packed decode runs ~2x faster (TinyLlama 11 -> 19 tok/s, ffn_down 20 -> 10 ms) and
    // stays bit-identical to the reference (each output row is an independent dot, so
    // parallelizing across rows does not change any reduction order). Windows therefore
    // defaults serial-decode OFF; an explicit env opt-in still forces it on.
    let serial_packed_decode = if platform.operating_system == "windows" {
        if planner_env.flag_enabled("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE") {
            Some("on")
        } else {
            Some("off")
        }
    } else {
        optional_x86_q8_gate("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE")
    };
    env_updates.insert(
        "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
        serial_packed_decode,
    );
    env_updates.insert(
        "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
        optional_x86_q8_gate("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
    );
    let ffn_decode_chain_enabled = planner_env.flag_enabled("CAMELID_X86_Q8_FFN_DECODE_CHAIN");
    let ffn_gate_up_decode_consumer_enabled = ffn_decode_chain_enabled
        || planner_env.flag_enabled("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
        if ffn_gate_up_decode_consumer_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
        if ffn_decode_chain_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
    );
    let ffn_down_decode_consumer_enabled = ffn_decode_chain_enabled
        || planner_env.flag_enabled("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER");
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
        if ffn_down_decode_consumer_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER"),
    );
    env_updates.insert("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER", Some("off"));
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
    );

    if ffn_decode_chain_enabled
        && !planner_env.flag_enabled("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER")
    {
        reasons.push(
            "FFN decode-chain opt-in also enables the required FFN gate/up decode consumer gate"
                .into(),
        );
    }
    if ffn_decode_chain_enabled
        && !planner_env.flag_enabled("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER")
    {
        reasons.push(
            "FFN decode-chain opt-in also enables the required FFN-down decode consumer gate"
                .into(),
        );
    }
    reasons.push("validated x86_64 (Linux/Windows) Rust Q8 runtime repack enabled".into());
    reasons.push("validated Rust AVX2 Q8 packed rows4 kernel selected".into());
    reasons.push("attention, FFN, and output experiments enabled by default".into());
    if matches!(profile, ExecutionProfile::Experimental) {
        reasons.push("experimental profile active; support claims remain unchanged".into());
    }

    (
        "cpu_q8_runtime_repack",
        "x86_experimental_q8_0_avx2_rust",
        "q8_0_runtime_packed_rows4_prefill_avx2_available",
        "enabled_when_q8_runtime_storage_active",
        "q8_0_decode_packed_rows4_avx2",
        "retained_q8_reference_path",
    )
}

fn safe_q8_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cpu_reference",
        "safe_q8_0_block_dot",
        "safe_cpu_prefill",
        "always_retained_reference_path",
        "safe_cpu_decode",
        "retained_q8_reference_path",
    )
}

/// Plan labels when the GPU-resident CUDA decode engine drives this process (the NVIDIA
/// analog of `metal_resident_q8_runtime`). Weights stay plain RAM-resident Q8_0 blocks —
/// the engine uploads them to VRAM once and decodes on-device; the CPU rows4 repack is
/// disabled because the GPU consumes the plain blocks. The `retained_q8_reference_path`
/// CPU plan remains the in-process fallback for any token/config the resident gates
/// reject. Validated token-AND-text-identical to the CPU reference (transitively
/// llama.cpp) on the dense Qwen3 Q8_0 ChatML rows; see the COMPATIBILITY.md Windows CUDA
/// section and the `qwen3-*-windows-cuda-resident-parity-*` evidence bundles.
fn cuda_resident_q8_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cuda_resident_q8_runtime",
        "cuda_resident_q8_0_wire",
        "q8_0_cuda_resident_prefill",
        "resident_single_shot_prefill",
        "q8_0_cuda_resident_decode",
        "retained_q8_reference_path",
    )
}

/// Plan labels for the GPU-runnable tier: a Q8_0 model on a resident-capable dense
/// architecture that is NOT a curated supported exact-row. Byte-for-byte the same GPU
/// route as [`cuda_resident_q8_plan`], but every label carries a `_runnable_unvalidated`
/// suffix so telemetry, receipts, and the UI can never mistake it for a supported row.
/// The support_level stays `unknown_or_unvalidated`; admission to this tier is gated at
/// runtime by a GPU-vs-CPU parity self-check (see `inference.rs`), which falls the model
/// back to the CPU reference path if the resident output is not token-identical.
fn cuda_resident_q8_runnable_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cuda_resident_q8_runtime_runnable_unvalidated",
        "cuda_resident_q8_0_wire",
        "q8_0_cuda_resident_prefill_runnable_unvalidated",
        "resident_single_shot_prefill",
        "q8_0_cuda_resident_decode_runnable_unvalidated",
        "retained_q8_reference_path",
    )
}

/// Architectures the CUDA resident engine implements token-identically to the CPU
/// reference, and for which the GPU-runnable tier is eligible when a model is Q8_0 but
/// not a curated support row. Deliberately narrow — dense llama/qwen/mistral only; MoE
/// (expert routing) and archs the generic dense resident kernels cannot express are
/// excluded (gemma — see the gemma3 paragraph below; phi; ssm; qwen35 — served by its
/// own dedicated resident Metal/CUDA graphs, never by this tier) so we
/// never route a model the resident dense kernel cannot run under a GPU label. The
/// runtime `resident_decode_eligible` check + the parity self-check are the backstops.
///
/// gemma3 stays EXCLUDED here even though it now has resident lanes on BOTH GPU
/// backends (Metal, Phase 3b of the Metal campaign; CUDA, Phase 2 of the CUDA one).
/// The reason is no longer "no CUDA windowed forward exists" — one does — it is that
/// both consumers of this predicate are non-Q8-exact tiers that hazard H5 forbids for
/// windowed archs:
///
/// - The GPU-runnable tier admits UNCURATED models on architecture alone. A windowed
///   row admitted that way would carry no windowed parity receipt and would bypass the
///   Q8_0 pin, which is the whole point of H5.
/// - The K-quant plan selection would advertise a resident K-quant lane whose gather
///   drops the gemma3 embed scale, again with no windowed receipt behind it.
///
/// gemma3's dense lane is selected via its curated exact row instead
/// (`is_supported_exact_q8_row` → the macOS Metal-resident Q8 plan, or the
/// CUDA-resident windowed plan on a CUDA host).
fn is_gpu_runnable_arch(gguf: &GgufFile) -> bool {
    let arch = gguf.architecture().unwrap_or("");
    // mobilemoe is the one MoE arch the resident lane routes on the GPU (expert-indexed
    // GEMV + on-device top-k), so it is admitted despite its non-zero expert_count.
    if arch == "mobilemoe" {
        return true;
    }
    if !matches!(arch, "llama" | "qwen2" | "qwen3" | "mistral") {
        return false;
    }
    // Exclude MoE: the resident dense kernel does not implement expert routing. A missing
    // key means dense (the common case); a present non-zero expert_count means MoE.
    gguf.metadata_u32(&format!("{arch}.expert_count"))
        .map(|experts| experts == 0)
        .unwrap_or(true)
}

/// Architectures implemented by the packed Prism Q1_0/Q2_0 GPU runtimes.
///
/// This is deliberately separate from [`is_gpu_runnable_arch`]: that predicate
/// describes the generic dense CUDA/Metal engine and correctly excludes
/// `qwen35`, while the dedicated Qwen3.5 runnable graph is exactly what drives
/// Bonsai on Apple Silicon Metal and Windows CUDA. Sharing the generic predicate made `/v1/health`
/// disclose `cpu_reference` even while the resident Qwen3.5 Metal graph was
/// active.
/// `lfm2` on the resident Metal lane.
///
/// Deliberately its own predicate rather than a shared `is_gpu_runnable_arch`:
/// sharing a generic one is what previously let `/v1/health` disclose
/// `cpu_reference` while a Metal graph was live. The gate below keys on exactly
/// the inputs the ROUTING keys on, so the disclosure cannot drift from reality.
fn is_lfm2_metal_arch(gguf: &GgufFile) -> bool {
    gguf.architecture() == Some("lfm2")
}

fn is_prism_low_bit_metal_arch(gguf: &GgufFile) -> bool {
    let arch = gguf.architecture().unwrap_or("");
    if !matches!(arch, "qwen3" | "qwen35") {
        return false;
    }
    gguf.metadata_u32(&format!("{arch}.expert_count"))
        .map(|experts| experts == 0)
        .unwrap_or(true)
}

/// Plan labels for a mixed K-quant (Q4_K_M = Q4_K + Q6_K) model. K-quant 2-D linears
/// load WIRE-ONLY and are decoded either by the GPU-resident engine (`q4k_gemv`/
/// `q6k_gemv`) when CUDA resident decode is driving this process, or by the CPU
/// block-dot (`q4_k_dot_avx2` + `q6_k_wire_row_dot`) otherwise — neither materializes
/// f32. Descriptive only (no env_updates): the actual route is chosen at runtime by
/// `resident_decode_cuda_active()` and, off the GPU, by `kquant_block_dot_selected()` —
/// which for these wire-only linears is unconditionally the block-dot, since they carry
/// no f32 `data` for anything else to consume. (It is NOT
/// `q4_k_cpu_block_dot_enabled()` any more: that flag chooses between CPU kernels and
/// must not be able to delete the last consumer.) This replaces the old
/// `cpu_reference`/`dense_or_other` mislabel that reported a CPU fallback for a lane
/// that actually runs GPU-resident (K-quant conductor disclosure fix). Greedy parity vs
/// llama.cpp is recorded in the `*-q4_k_m-*-parity-*` evidence bundles.
fn select_kquant_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    planner_env: &PlannerEnv,
    metal_tensor_mix_supported: bool,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if platform.cuda_resident_active {
        reasons.push(
            "CUDA resident decode active; GPU-resident K-quant engine (q4k_gemv/q6k_gemv) drives decode from wire-only Q4_K/Q6_K blocks"
                .into(),
        );
        (
            "cuda_resident_kquant_runtime",
            "cuda_resident_kquant_wire",
            "kquant_cuda_resident_prefill",
            "resident_single_shot_prefill",
            "kquant_cuda_resident_decode",
            "kquant_cpu_block_dot_reference_path",
        )
    } else if !matches!(profile, ExecutionProfile::Safe)
        && platform.operating_system == "macos"
        && platform.architecture == "aarch64"
        && platform.metal_available
        && metal_tensor_mix_supported
        && planner_env.metal_flag_enabled("CAMELID_METAL_RESIDENT_DECODE")
        && planner_env.metal_flag_enabled("CAMELID_METAL_KQUANT")
        && planner_env.metal_flag_enabled("CAMELID_METAL_F32Y")
        && planner_env.metal_flag_enabled("CAMELID_METAL_WIRE")
    {
        reasons.push(
            "Metal resident K-quant stack selected automatically; Q4_K/Q6_K weights stay \
             wire-resident, prefill and decode run on Metal, and unsupported tensor mixes \
             retain the CPU block-dot fallback"
                .into(),
        );
        (
            "metal_resident_kquant_runtime",
            "metal_resident_kquant_wire",
            "kquant_metal_resident_prefill",
            "resident_single_command_buffer_prefill",
            "kquant_metal_resident_decode",
            "kquant_cpu_block_dot_reference_path",
        )
    } else {
        // Unconditional, because the ROUTING is now unconditional. Every 2-D K-quant
        // linear is loaded wire-only (`load_kquant_wire_linear` leaves `data` empty),
        // and `kquant_block_dot_selected` therefore picks the block-dot whatever
        // `CAMELID_X86_Q4K_DECODE` says — a flag may choose between kernels, it may not
        // delete the only consumer.
        //
        // This arm used to be gated on `q4_k_cpu_block_dot_enabled()`, with an `else`
        // that disclosed `cpu_reference` / `safe_cpu_decode` and the reason "K-quant
        // linears have no CPU consumer". Once routing stopped honouring the flag, that
        // fallback described a lane no run could take: with the flag off the plan would
        // have claimed a safe CPU path while the block-dot actually decoded. That is the
        // disclosure drift this function exists to prevent (see the doc comment above),
        // so the branch is gone rather than merely reworded.
        reasons.push(
            "CPU K-quant block-dot decode reads Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS wire blocks; no f32 materialization"
                .into(),
        );
        (
            "cpu_kquant_block_dot",
            "kquant_wire_block_dot",
            "cpu_kquant_block_dot_prefill",
            "always_retained_reference_path",
            "kquant_cpu_block_dot_decode",
            "kquant_cpu_block_dot_reference_path",
        )
    }
}

fn requested_profile() -> (ExecutionProfile, String) {
    match env::var("CAMELID_PROFILE").ok() {
        None => (ExecutionProfile::Auto, "profile=auto default".into()),
        Some(value) if value.eq_ignore_ascii_case("safe") => {
            (ExecutionProfile::Safe, "profile=safe requested".into())
        }
        Some(value) if value.eq_ignore_ascii_case("auto") => {
            (ExecutionProfile::Auto, "profile=auto requested".into())
        }
        Some(value) if value.eq_ignore_ascii_case("experimental") => (
            ExecutionProfile::Experimental,
            "profile=experimental requested; warnings enabled".into(),
        ),
        Some(value) if value.eq_ignore_ascii_case("debug") => (
            ExecutionProfile::Debug,
            "profile=debug requested; diagnostics enabled".into(),
        ),
        Some(value) => (
            ExecutionProfile::Safe,
            format!("invalid CAMELID_PROFILE={value}; failing closed to safe"),
        ),
    }
}

fn exact_model_row(model_path: &Path, gguf: &GgufFile) -> String {
    let from_name = gguf.model_name().map(|value| value.to_string());
    let from_file = model_path
        .file_name()
        .map(|v| v.to_string_lossy().to_string());
    // Prefer the GGUF `general.name`, but if it does NOT map to a recognized support row
    // while the FILENAME does, use the filename. Some GGUF conversions ship a junk
    // `general.name` (e.g. "hub") that would otherwise shadow a perfectly recognizable
    // filename and drop a known, validated model onto the slow cpu_reference path
    // instead of its GPU lane. This only ever UPGRADES an unrecognized name to a
    // recognized row — it never overrides a name that already matches a row.
    if let (Some(name), Some(file)) = (&from_name, &from_file) {
        if recognized_row_level(name) == "unknown_or_unvalidated"
            && recognized_row_level(file) != "unknown_or_unvalidated"
        {
            return file.clone();
        }
    }
    from_name.or(from_file).unwrap_or_else(|| "unknown".into())
}

/// The plan's support-level string for a recognized row, gated on the quant
/// the row's evidence actually covers. Every row in the recognized table is
/// Q8_0 evidence, so any other quant of the same model name reports
/// `unknown_or_unvalidated` here — a Q6_K TinyLlama must not echo the Q8_0
/// gate row's level. `/api/capabilities` (filename-anchored, quant-aware)
/// remains the support source of truth; this string only describes the row
/// the load-time plan recognized.
fn support_level(row: &str, quant_type: &str) -> String {
    if quant_type != "Q8_0" {
        return "unknown_or_unvalidated".into();
    }
    let level = recognized_row_level(row);
    if matches!(
        level,
        "recognized_prism_bonsai_exact_row"
            | "recognized_lfm2_5_2_6b_exact_row"
            | "recognized_phi3_mini_4k_exact_row"
    ) {
        "unknown_or_unvalidated".into()
    } else {
        level.into()
    }
}

/// Exact public LFM2.5 2.6B row recognition. Keep this stem match narrow: a
/// future Base, tool, multimodal, or differently versioned file must not
/// inherit the one Q8_0 receipt merely because its name contains `2.6B`.
fn is_lfm2_5_2_6b_exact_row(row: &str) -> bool {
    let normalized = normalize_row(row);
    let stem = normalized.strip_suffix("_gguf").unwrap_or(&normalized);
    matches!(stem, "lfm2_5_2_6b" | "lfm2_5_2_6b_q8_0")
}

/// Exact public Phi-3 Mini 4K Instruct row recognition. `general.name` in the
/// certified GGUF is only `Phi3`, so recognition must be anchored to the narrow
/// catalog filename rather than widening to every model with a phi3 header.
fn is_phi3_mini_4k_exact_row(row: &str) -> bool {
    let normalized = normalize_row(row);
    let stem = normalized.strip_suffix("_gguf").unwrap_or(&normalized);
    stem == "phi_3_mini_4k_instruct_q8_0" || stem == "phi3_mini_4k_instruct_q8_0"
}

/// Whether the selected execution plan is one of the two receipted LFM lanes.
///
/// Windows evidence covers the non-Safe x86_64 runnable CPU plan. The macOS
/// receipt is narrower: one Mac16,10 with the base Apple M4 CPU, macOS 26.5,
/// and the exact resident-Metal backend/prefill/decode labels. Every other
/// platform or fallback remains recognition-only.
#[allow(clippy::too_many_arguments)]
fn lfm2_selected_lane_supported(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    model_path: &Path,
    gguf: &GgufFile,
    quant_type: &str,
    selected_backend: &str,
    prefill_path: &str,
    decode_path: &str,
) -> bool {
    if matches!(profile, ExecutionProfile::Safe)
        || gguf.architecture() != Some("lfm2")
        || quant_type != "Q8_0"
        || model_path.file_name().and_then(|name| name.to_str()) != Some("LFM2.5-2.6B-Q8_0.gguf")
    {
        return false;
    }

    let windows_runnable_cpu = platform.operating_system == "windows"
        && platform.architecture == "x86_64"
        && selected_backend == "cpu_reference"
        && prefill_path == "safe_cpu_prefill"
        && decode_path == "safe_cpu_decode";
    let exact_m4_resident_metal = platform.operating_system == "macos"
        && platform.operating_system_version.trim() == "26.5"
        && platform.architecture == "aarch64"
        && platform.host_model_identifier.trim() == "Mac16,10"
        && platform.cpu_model.trim() == "Apple M4"
        && selected_backend == "metal_resident_lfm2_runtime"
        && prefill_path == "lfm2_metal_resident_prefill"
        && decode_path == "lfm2_metal_resident_decode";

    windows_runnable_cpu || exact_m4_resident_metal
}

/// Windows x86_64 exact-row lane that cleared Phi-3's stale head-dimension hold.
/// The current receipt covers the conservative CPU reference prefill/decode path;
/// other platforms and a future optimized route remain recognition-only until
/// independently re-run.
#[allow(clippy::too_many_arguments)]
fn phi3_selected_lane_supported(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    model_path: &Path,
    gguf: &GgufFile,
    quant_type: &str,
    selected_backend: &str,
    prefill_path: &str,
    decode_path: &str,
) -> bool {
    !matches!(profile, ExecutionProfile::Safe)
        && platform.operating_system == "windows"
        && platform.architecture == "x86_64"
        && gguf.architecture() == Some("phi3")
        && quant_type == "Q8_0"
        && model_path.file_name().and_then(|name| name.to_str())
            == Some("Phi-3-mini-4k-instruct-Q8_0.gguf")
        && selected_backend == "cpu_reference"
        && prefill_path == "safe_cpu_prefill"
        && decode_path == "safe_cpu_decode"
}

/// Quant certified for one exact Prism/Bonsai artifact name. Exact normalized
/// stem only: size/family substring matching would let neighboring files inherit
/// the Mac mini 2 receipt. Q2 returns the geometry-refined label the tensor scan
/// reports, while the catalog continues to use the upstream-friendly `Q2_0`.
fn prism_bonsai_expected_quant(row: &str) -> Option<&'static str> {
    let normalized = normalize_row(row);
    let stem = normalized.strip_suffix("_gguf").unwrap_or(&normalized);
    Some(match stem {
        "bonsai_4b_q1_0" | "bonsai_8b_q1_0" | "bonsai_27b_q1_0" => "Q1_0",
        "ternary_bonsai_4b_q2_0" | "ternary_bonsai_8b_q2_0" | "ternary_bonsai_27b_q2_0" => {
            "Q2_0_G128"
        }
        "ternary_bonsai_4b_pq2_0" => "PQ2_0",
        _ => return None,
    })
}

/// Quant-blind name→level table. Besides `support_level` above, this powers
/// row *recognition* — the junk general.name → filename upgrade in
/// `exact_model_row` and the supported-row planner gate — which must keep
/// working for non-Q8_0 files: recognition and support claims are different
/// questions.
fn recognized_row_level(row: &str) -> &'static str {
    let normalized = normalize_row(row);
    if normalized.contains("tinyllama") {
        "supported_current_gate"
    } else if normalized.contains("llama_3_2_1b_instruct") {
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("llama_3_2_3b_instruct") {
        // Full 512-8192 ladder re-validated on the anchored canonical GGUF
        // (raw-decode greedy parity vs llama.cpp acd79d603; see the
        // llama32-3b-context-512-8192-anchored evidence bundle). Split from the
        // 8B arm below, whose checked packs remain 512/1024/2048.
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("llama_3_8b_instruct")
        || normalized.contains("meta_llama_3_8b_instruct")
    {
        "supported_exact_row_smoke_512_1024_2048"
    } else if normalized.contains("mistral_7b_instruct_v0_3") {
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("qwen3_0_6b_instruct")
        || normalized.contains("qwen3_1_7b_instruct")
        || normalized.contains("qwen3_4b_instruct")
        || normalized.contains("qwen3_8b_instruct")
    {
        // Dense Qwen3 Q8_0 ChatML rows (thinking disabled), validated token+text
        // identical to llama.cpp at 1/5/50 on the cpu_reference path and on the
        // x86_64 runtime-repack/AVX2 Q8 path (parity re-validated on Windows).
        // Scoped to the short-chat smoke envelope; MoE (A3B), base variants, other
        // sizes/quants, longer context, and thinking-mode are NOT covered.
        // (Replaces the broader `contains("qwen3")` branch from PR #283, whose
        // label claimed 512/1024/2048 context packs and matched MoE/base/other
        // sizes — neither validated for qwen3.)
        "supported_exact_row_smoke_chatml"
    } else if normalized.contains("gemma_3_1b_it") {
        // gemma-3-1b-it Q8_0 (gemma3→Metal Phases 3b-5). The ≥512-token
        // windowed receipt LANDED in Phase 4 — 9/9 legs token-and-text
        // identical to the pinned external oracle at 606/1205/2403 prompt
        // tokens — but this string deliberately STAYS `sub512`, because this
        // table is platform-blind: the same level string is reported on hosts
        // where the resident lane cannot run and the runnable CPU bridge (no
        // window mask) serves the row instead. `sub512` is the envelope that
        // holds on EVERY host that recognizes this row; widening it here would
        // over-claim on the fallback host. The lane-aware 2,403-prompt-token
        // claim lives in `/api/capabilities` (row `gemma_3_1b_it_q8_0`), which
        // is the support source of truth and states the lane it applies to.
        // Non-Q8_0 quants of the same name report unknown via `support_level`
        // and are declined by the resident admission (hazard H5).
        "supported_exact_row_smoke_sub512"
    } else if is_lfm2_5_2_6b_exact_row(row) {
        // Recognition only. Some published conversions use an opaque hash for
        // `general.name`, so recognizing the exact filename lets
        // `exact_model_row` expose the capabilities row. The plan promotes
        // this marker only after a receipted platform and actual selected lane
        // both match; platform-blind `support_level` maps it back to unknown.
        //
        // This does not admit LFM2 to the optimized dense Q8 planner:
        // `is_supported_exact_q8_row` rejects runnable-only architectures
        // before consulting this table.
        "recognized_lfm2_5_2_6b_exact_row"
    } else if is_phi3_mini_4k_exact_row(row) {
        // Recognition only. Windows x86_64 promotes this marker after route
        // selection; the marker itself is platform-blind and therefore never a
        // support claim.
        "recognized_phi3_mini_4k_exact_row"
    } else if normalized.contains("ornith_1_0_9b") {
        // Ornith-1.0-9B (qwen35 hybrid gated-delta-net), certified on the
        // runnable serve lane. `/api/capabilities` has carried
        // `supported_exact_row_smoke` for this row since that certification,
        // but this table had no arm — so the load-time plan disclosed
        // `support_level=unknown_or_unvalidated` for the very same file, and
        // two runtime surfaces contradicted each other about a row that is in
        // fact certified.
        //
        // Adding the arm is only safe because `is_supported_exact_q8_row`
        // excludes runnable-only archs BEFORE it consults this table. qwen35
        // has no optimized dense lane on ANY host; without that exclusion this
        // arm would hand the row to `select_macos_q8_plan` /
        // `select_x86_q8_plan`, describing an engine that cannot express its
        // gated-delta-net layers.
        //
        // Quant-blind, like every arm here: the Q4_K_M and Q3_K_M Ornith rows
        // match this substring too, and `support_level` maps them back to
        // unknown because they are not Q8_0. Their own claims live in
        // `/api/capabilities`, which is quant-aware.
        "supported_exact_row_smoke"
    } else if prism_bonsai_expected_quant(row).is_some() {
        // Recognition only. The plan promotes this to the supported level iff
        // the quant and full macOS Metal routing predicate match above.
        "recognized_prism_bonsai_exact_row"
    } else if normalized.contains("mixtral_8x7b_instruct_v0_1") {
        "bounded_runtime_only_unsupported"
    } else {
        "unknown_or_unvalidated"
    }
}

/// Whether the OPTIMIZED-ENGINE Q8 plan arm may claim this row. Not "is this row
/// supported" — `/api/capabilities` answers that, and the two questions diverge
/// for a row that is certified on a lane this engine does not own.
///
/// Runnable-only archs are excluded FIRST, before the table is consulted at all.
/// `crate::model::is_runnable_only_arch` (qwen35, gemma2, lfm2, bitnet-b1.58)
/// marks the archs
/// whose only correct forward pass lives in `crate::runnable` on EVERY host. The
/// arm this predicate gates dispatches to `select_macos_q8_plan` and
/// `select_x86_q8_plan`, which would describe an engine that cannot run them and
/// would also write dense-Q8 tuning env (`CAMELID_MAC_Q8_REPACK`,
/// `CAMELID_X86_Q8_*`) into a load that never touches those kernels.
///
/// The exclusion lives INSIDE this predicate, not at its call site, so that a
/// certified runnable-only row can be named in `recognized_row_level` — the
/// Ornith Q8_0 row is — without silently acquiring an optimized-lane claim.
fn is_supported_exact_q8_row(row: &str, gguf: &GgufFile) -> bool {
    if crate::model::is_runnable_only_arch(gguf.architecture().unwrap_or_default()) {
        return false;
    }
    // `supported_exact_row_smoke` (the Ornith rows) is deliberately absent: it is
    // a runnable-lane level, and no row carrying it is served by this engine. The
    // arch guard above is the load-bearing lock; this omission is the second one.
    matches!(
        recognized_row_level(row),
        "supported_current_gate"
            | "supported_exact_row_smoke_512_1024_2048_4096_8192"
            | "supported_exact_row_smoke_512_1024_2048"
            | "supported_exact_row_smoke_chatml"
            | "supported_exact_row_smoke_sub512"
    )
}

/// Plan-level mirror of `crate::model::arch_has_windowed_attention`, keyed on
/// the GGUF arch string because the planner works pre-config-parse. gemma3 is
/// the only windowed arch today; a future one must be added in lockstep with
/// the parsed-metadata predicate.
fn is_windowed_attention_arch(gguf: &GgufFile) -> bool {
    gguf.architecture() == Some("gemma3")
}

fn model_family(row: &str, gguf: &GgufFile) -> String {
    let normalized = normalize_row(row);
    if normalized.contains("tinyllama") {
        "tinyllama".into()
    } else if normalized.contains("llama") {
        "llama".into()
    } else if normalized.contains("mistral") {
        "mistral".into()
    } else if normalized.contains("mixtral") {
        "mixtral".into()
    } else {
        gguf.architecture().unwrap_or("unknown").to_string()
    }
}

/// Human label for the file's quantization, reported in the plan (and from
/// there `/v1/health` and the System page). The declared `general.file_type`
/// (llama.cpp ftype naming, shared with receipts) is the most specific
/// truthful source — it keeps a pure-Q6_K or TQ1_0 file from being collapsed
/// into the coarse "Q4_K_M"/"Q8_0" buckets of the tensor-scan fallback. A
/// declared "Q8_0" is accepted only when the scan actually finds Q8_0 tensors,
/// because the Q8_0 label is what gates `support_level` onto a promoted row.
/// Routing never reads this label — the planner branches on the tensor
/// predicates directly (see `plan_for_model_with_platform`).
fn quant_type(gguf: &GgufFile) -> String {
    let has = |t: GgufTensorType| gguf.tensors.iter().any(|tensor| tensor.tensor_type == t);
    let has_q8_0 = has(GgufTensorType::Q8_0);
    // BitNet and Prism both use general.file_type=40 in their respective forks.
    // GGML tensor type 36 is definitive for BitNet's I2_S payload.
    if has(GgufTensorType::I2S) {
        return "I2_S".into();
    }
    // File type 41 identifies Prism Q2_0 but cannot identify its deployed
    // block geometry. The directory resolver can, so it outranks metadata.
    if has(GgufTensorType::Q2_0G64) {
        return "Q2_0_G64".into();
    }
    if has(GgufTensorType::Pq2_0) {
        return "PQ2_0".into();
    }
    if has(GgufTensorType::Q2_0G128) {
        return "Q2_0_G128".into();
    }
    if let Some(declared) = crate::receipt::declared_file_type_label(gguf) {
        if declared != "Q8_0" || has_q8_0 {
            return declared.into();
        }
    }
    if has_q8_0 {
        "Q8_0".into()
    } else if has(GgufTensorType::Q4K) {
        // Undeclared K-quant mix (Q4_K_M = Q4_K + Q6_K). Decoded by the GPU-resident
        // engine (q4k_gemv/q6k_gemv) or, on CPU, the K-quant block-dot — both consume
        // the wire-only blocks. Recognized here so the plan stops mislabeling it as
        // the `dense_or_other` cpu_reference fallback (K-quant conductor disclosure fix).
        "Q4_K_M".into()
    } else if has(GgufTensorType::Q6K) {
        "Q6_K".into()
    } else if has(GgufTensorType::Q1_0) {
        "Q1_0".into()
    } else {
        "dense_or_other".into()
    }
}

fn normalize_row(row: &str) -> String {
    row.to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
}

fn cpu_features() -> Vec<String> {
    let mut out = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            out.push("dotprod".into());
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            out.push("i8mm".into());
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push("avx2".into());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push("avx512f".into());
        }
        let cpuinfo_flags = cpuinfo_flags();
        if cpuinfo_has_flag(&cpuinfo_flags, "avx_vnni") {
            out.push("avx_vnni".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "avx512_vnni") {
            out.push("avx512_vnni".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "amx_tile") {
            out.push("amx_tile".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "amx_int8") {
            out.push("amx_int8".into());
        }
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuinfo_flags() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.trim()
                        .eq_ignore_ascii_case("flags")
                        .then(|| value.trim().to_string())
                })
            })
            .unwrap_or_default()
    }
    #[cfg(not(target_os = "linux"))]
    {
        String::new()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuinfo_has_flag(flags: &str, wanted: &str) -> bool {
    flags.split_whitespace().any(|flag| flag == wanted)
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.strip_prefix("model name").and_then(|rest| {
                        rest.split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                    })
                })
            })
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

fn host_model_identifier() -> String {
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "hw.model"]).unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".into()
    }
}

fn operating_system_version() -> String {
    #[cfg(target_os = "macos")]
    {
        command_output("sw_vers", &["-productVersion"]).unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unknown".into()
    }
}

#[cfg(target_os = "macos")]
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn platform_label(os: &str, arch: &str, cpu_model: &str) -> String {
    if os == "macos" && arch == "aarch64" {
        if cpu_model.to_ascii_lowercase().contains("apple") {
            "macOS arm64 Apple Silicon".into()
        } else {
            "macOS arm64".into()
        }
    } else if os == "linux" && arch == "x86_64" {
        "Ubuntu/Linux x86_64".into()
    } else {
        format!("{os} {arch}")
    }
}

fn has_feature(features: &[String], wanted: &str) -> bool {
    features.iter().any(|feature| feature == wanted)
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

pub(crate) fn flag_value_disabled(value: &str) -> bool {
    let value = value.trim();
    value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("disabled")
        || value.eq_ignore_ascii_case("cpu")
}

fn flag_value_enabled(value: &str) -> bool {
    let value = value.trim();
    value.eq_ignore_ascii_case("1")
        || value.eq_ignore_ascii_case("on")
        || value.eq_ignore_ascii_case("true")
        || value.eq_ignore_ascii_case("enabled")
}

/// Exactly the spelling the Metal runtime accepts (`src/metal.rs`:
/// `f32y_gemv_enabled`, `wire_weights_enabled`, `kquant_resident_enabled`, and
/// `q8_0_env_flag_enabled_default_off`). The generic [`flag_value_enabled`] also
/// accepts `on`/`enabled`, so using it here would let `CAMELID_METAL_KQUANT=on`
/// label a run Metal-resident while the engine actually ran on the CPU.
fn metal_flag_value_enabled(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

/// The offending value when `CAMELID_X86_Q8_KERNEL` names a kernel this build
/// does not have, else `None` (unset, a disable spelling, or a valid AVX2 opt-in
/// all count as "no invalid override").
fn invalid_x86_kernel_value(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("disabled")
        || trimmed.eq_ignore_ascii_case("avx2")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
    {
        None
    } else {
        Some(value)
    }
}

fn env_flag_disabled(key: &str) -> bool {
    env::var(key).is_ok_and(|value| flag_value_disabled(&value))
}

/// Whether the LFM2 resident-Metal plan may be selected from operator policy.
/// Runtime routing calls the same predicate so `CAMELID_PROFILE=safe` and the
/// explicit LFM opt-out cannot leave health reporting CPU while Metal runs.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn lfm2_metal_plan_selectable() -> bool {
    lfm2_metal_policy_allows(
        &requested_profile().0,
        env_flag_disabled("CAMELID_LFM2_METAL"),
    )
}

/// The promotion receipts cover the normal runnable profile, not Safe (or an
/// invalid profile that resolves to Safe). API/catalog host classification uses
/// this same parser so it cannot disagree with the stored execution plan.
pub(crate) fn supported_profile_selected() -> bool {
    !matches!(requested_profile().0, ExecutionProfile::Safe)
}

fn lfm2_metal_policy_allows(profile: &ExecutionProfile, explicitly_disabled: bool) -> bool {
    !matches!(profile, ExecutionProfile::Safe) && !explicitly_disabled
}

#[allow(dead_code)]
fn env_flag_enabled(key: &str) -> bool {
    env::var(key).is_ok_and(|value| flag_value_enabled(&value))
}

#[cfg(test)]
mod tests {
    /// The lfm2 Metal lane is default-on, and ROUTING (`lfm2_metal_enabled`) must
    /// agree with DISCLOSURE (the execution-plan arm) for every value of
    /// `CAMELID_LFM2_METAL`. When they disagree `/v1/health` names a lane other
    /// than the one that ran — which is what an independently-written opt-out list
    /// here produced, differing from the planner's on `no` and `cpu`.
    ///
    /// Both now consult `flag_value_disabled`, so this pins that they stay shared
    /// rather than merely happening to match today.
    #[test]
    fn lfm2_metal_routing_and_plan_gates_share_one_predicate() {
        for v in [
            "0", "off", "OFF", "false", "False", "disabled", "cpu", " 0 ",
        ] {
            assert!(
                super::flag_value_disabled(v),
                "{v:?} must read as an opt-out for BOTH gates"
            );
        }
        for v in ["1", "on", "true", "yes", "", "metal", "anything-else"] {
            assert!(
                !super::flag_value_disabled(v),
                "{v:?} must leave the default-on lane enabled for BOTH gates"
            );
        }

        let _guard = crate::test_support::env_lock();
        env::set_var("CAMELID_PROFILE", "safe");
        env::remove_var("CAMELID_LFM2_METAL");
        assert!(
            !lfm2_metal_plan_selectable(),
            "Safe profile must disable both LFM plan selection and runtime routing"
        );
        env::set_var("CAMELID_PROFILE", "auto");
        assert!(lfm2_metal_plan_selectable());
        env::set_var("CAMELID_LFM2_METAL", "0");
        assert!(!lfm2_metal_plan_selectable());
        env::remove_var("CAMELID_PROFILE");
        env::remove_var("CAMELID_LFM2_METAL");
    }

    /// The qwen35 CUDA lane must ROUTE where the plan DISCLOSES it.
    /// `select_kquant_plan` advertises `cuda_resident_kquant_runtime` off
    /// `platform.cuda_resident_active` alone, but routing additionally required the
    /// output tensor to be a Prism low-bit quant — so a certified Ornith Q4_K_M row
    /// (qwen35) reported the CUDA lane in `/v1/health` and decoded on the CPU.
    /// Measured on an RTX 3060 Laptop: 0.42 tok/s served, 6.14 tok/s once the lane
    /// was reachable, same row, same host.
    ///
    /// Pins the two halves to one predicate: with no operator opt-out, Windows
    /// routing tracks `cuda_resident_decode_will_run` exactly — the same signal the
    /// plan reads — for every qwen35 row, Prism or K-quant.
    #[test]
    fn qwen35_cuda_routing_tracks_the_disclosed_resident_lane() {
        let _guard = crate::test_support::env_lock();
        env::remove_var("CAMELID_QWEN35_CUDA");
        env::remove_var("CAMELID_PROFILE");
        env::remove_var("CAMELID_DETERMINISTIC");

        if cfg!(windows) {
            assert_eq!(
                qwen35_cuda_lane_selectable(),
                cuda_resident_decode_will_run(),
                "with no opt-out, routing must follow the same signal the plan discloses \
                 — not a per-row quant test the plan never applied"
            );
        }

        env::set_var("CAMELID_QWEN35_CUDA", "0");
        assert!(
            !qwen35_cuda_lane_selectable(),
            "an explicit opt-out must still pin the CPU oracle"
        );

        env::set_var("CAMELID_QWEN35_CUDA", "1");
        env::set_var("CAMELID_PROFILE", "safe");
        assert!(
            !qwen35_cuda_lane_selectable(),
            "safe profile must beat an explicit opt-in"
        );
        env::remove_var("CAMELID_PROFILE");

        env::set_var("CAMELID_DETERMINISTIC", "1");
        assert!(
            !qwen35_cuda_lane_selectable(),
            "deterministic mode must pin the CPU oracle even with an explicit opt-in"
        );

        env::remove_var("CAMELID_DETERMINISTIC");
        env::remove_var("CAMELID_QWEN35_CUDA");
    }

    use super::*;
    use crate::{
        gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor},
        test_support::env_lock,
    };
    use std::{collections::BTreeMap, path::PathBuf};

    fn platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            operating_system: os.into(),
            operating_system_version: "fixture-os-version".into(),
            architecture: arch.into(),
            platform_label: platform_label(os, arch, "Apple M4"),
            host_model_identifier: "fixture-host-model".into(),
            cpu_model: "fixture cpu".into(),
            cpu_features: features.iter().map(|feature| (*feature).into()).collect(),
            metal_available: false,
            cuda_available: false,
            gpu_accel_enabled: false,
            cuda_resident_active: false,
        }
    }

    fn metal_platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            metal_available: true,
            gpu_accel_enabled: true,
            ..platform(os, arch, features)
        }
    }

    fn cuda_platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            cuda_available: true,
            gpu_accel_enabled: true,
            cuda_resident_active: true,
            ..platform(os, arch, features)
        }
    }

    fn receipted_lfm2_m4_platform() -> PlanPlatform {
        PlanPlatform {
            operating_system_version: "26.5".into(),
            host_model_identifier: "Mac16,10".into(),
            cpu_model: "Apple M4".into(),
            ..metal_platform("macos", "aarch64", &["dotprod", "i8mm"])
        }
    }

    fn fixture(name: &str) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.name".into(),
            GgufMetadataValue::String(name.into()),
        );
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
        );
        GgufFile {
            path: PathBuf::from("/tmp/model.gguf"),
            version: 3,
            tensor_count: 1,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![GgufTensorDescriptor {
                name: "blk.0.attn_q.weight".into(),
                dimensions: vec![32, 32],
                tensor_type: GgufTensorType::Q8_0,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 34,
            }],
        }
    }

    /// A qwen35 row shaped like ornith-1.0-9b-Q4_K_M: Q4_K + Q6_K projections plus F32
    /// norms. `extra` lets a test add a type the Metal lane cannot run (e.g. Q5_K).
    fn qwen35_kquant_fixture(name: &str, extra: &[GgufTensorType]) -> GgufFile {
        let mut types = vec![
            GgufTensorType::Q4K,
            GgufTensorType::Q6K,
            GgufTensorType::F32,
        ];
        types.extend_from_slice(extra);
        let mut gguf = quant_fixture(name, None, &types);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen35".into()),
        );
        gguf
    }

    /// The qwen35 K-quant Metal arm. Before it existed, an ornith Q4_K_M load disclosed
    /// `cpu_kquant_block_dot` while the resident Metal graph served — verified on a live
    /// load, the fourth instance of that defect. The Q8_0 arm could not cover this row
    /// because it keys on `has_q8_0_tensors`.
    ///
    /// Also pins the Q5_K exclusion: there is no `q5k` Metal kernel, so
    /// `resident_metal_format` maps it to None, `prism_metal_weight` errors and routing
    /// falls back to CPU. Disclosure must fall back with it.
    #[test]
    fn qwen35_kquant_macos_metal_plan_follows_the_routing_gate() {
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_QWEN35_METAL");
        let path = PathBuf::from("/models/ornith-1.0-9b-Q4_K_M.gguf");
        let metal = || metal_platform("macos", "aarch64", &["dotprod", "i8mm"]);
        let plan_for = |gguf: &GgufFile, platform: PlanPlatform| {
            plan_for_model_with_platform_and_env(
                &path,
                gguf,
                Some(8),
                platform,
                &PlannerEnv::capture(),
            )
        };

        let runnable = qwen35_kquant_fixture("Ornith 1.0 9B", &[]);
        let on = plan_for(&runnable, metal());
        assert_eq!(
            on.plan.selected_backend, "metal_resident_qwen35_kquant_runtime",
            "a Q4_K/Q6_K qwen35 row on Metal must disclose the lane that serves it"
        );
        assert_eq!(on.plan.decode_path, "qwen35_metal_resident_decode");

        // Every opt-out spelling the routing honours must flip disclosure too.
        for v in ["0", "off", "disabled", "cpu", "False"] {
            env::set_var("CAMELID_QWEN35_METAL", v);
            assert_ne!(
                plan_for(&runnable, metal()).plan.selected_backend,
                "metal_resident_qwen35_kquant_runtime",
                "opt-out {v:?} must deselect the Metal arm"
            );
        }
        env::remove_var("CAMELID_QWEN35_METAL");

        // Q5_K now has `q5k_linear_simd`/`q5k_linear_tiled` and is mapped by
        // `resident_metal_format`, so routing serves it and disclosure must say so.
        // This assertion was `assert_ne!` while the kernel did not exist; it flips
        // WITH the kernel, which is the point of gating disclosure on the same
        // predicate the loader uses rather than on a hand-maintained list.
        let q5k = qwen35_kquant_fixture("Ornith 1.0 9B", &[GgufTensorType::Q5K]);
        assert_eq!(
            plan_for(&q5k, metal()).plan.selected_backend,
            "metal_resident_qwen35_kquant_runtime",
            "a q5_K tensor is served by the resident Metal graph — disclose it"
        );

        // Q3_K still has no Metal kernel, so it remains the fail-closed case:
        // `resident_metal_format` maps it to None, `prism_metal_weight` errors, and
        // the graph declines to the CPU hybrid. Disclosure must decline with it.
        let q3k = qwen35_kquant_fixture("Ornith 1.0 9B", &[GgufTensorType::Q3K]);
        assert_ne!(
            plan_for(&q3k, metal()).plan.selected_backend,
            "metal_resident_qwen35_kquant_runtime",
            "a q3_K tensor makes prism_metal_weight error — disclosure must not claim Metal"
        );

        // Not on hosts where the resident graph cannot run.
        assert_ne!(
            plan_for(&runnable, platform("linux", "x86_64", &["avx2"]))
                .plan
                .selected_backend,
            "metal_resident_qwen35_kquant_runtime"
        );
        assert_ne!(
            plan_for(&runnable, platform("macos", "aarch64", &["dotprod"]))
                .plan
                .selected_backend,
            "metal_resident_qwen35_kquant_runtime",
            "no Metal device means no Metal lane"
        );
        clear_profile_env();
    }

    /// A qwen35 row shaped like ornith-1.0-9b-Q8_0: Q8_0 projections plus F32
    /// norms, no Prism low-bit and no K-quant tensors.
    fn qwen35_q8_fixture(name: &str) -> GgufFile {
        let mut gguf = quant_fixture(name, None, &[GgufTensorType::Q8_0, GgufTensorType::F32]);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen35".into()),
        );
        gguf
    }

    /// The qwen35 resident Metal arm (Q8_0 admission) — disclosure must follow
    /// the routing gate for every opt-out spelling. Without the arm the ornith
    /// Q8_0 row fell through to the final else and `/v1/health` reported
    /// `cpu_reference`/`safe_cpu_decode` while the Metal graph was serving; the
    /// routing side shares `flag_value_disabled` via `qwen35_metal_enabled`, so
    /// this pins the two gates end to end rather than merely today's values.
    #[test]
    fn qwen35_q8_macos_metal_plan_follows_the_routing_gate() {
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_QWEN35_METAL");
        let path = PathBuf::from("/models/ornith-1.0-9b-Q8_0.gguf");
        let gguf = qwen35_q8_fixture("Ornith 1.0 9B");
        let plan_on = |platform: PlanPlatform| {
            plan_for_model_with_platform_and_env(
                &path,
                &gguf,
                Some(8),
                platform,
                &PlannerEnv::capture(),
            )
        };

        let on = plan_on(metal_platform("macos", "aarch64", &["dotprod", "i8mm"]));
        assert_eq!(on.plan.selected_backend, "metal_resident_qwen35_runtime");
        assert_eq!(on.plan.decode_path, "qwen35_metal_resident_decode");
        assert!(
            on.plan
                .reasons
                .iter()
                .any(|reason| reason.contains("qwen35 resident Metal lane")),
            "the selected lane must be named in reasons: {:?}",
            on.plan.reasons
        );

        // Every opt-out spelling the ROUTING honors must flip the disclosure too.
        for v in ["0", "off", "disabled", "cpu", "False", " 0 "] {
            env::set_var("CAMELID_QWEN35_METAL", v);
            let off = plan_on(metal_platform("macos", "aarch64", &["dotprod", "i8mm"]));
            assert_eq!(
                off.plan.selected_backend, "cpu_reference",
                "opt-out {v:?} must deselect the Metal arm"
            );
        }
        env::remove_var("CAMELID_QWEN35_METAL");

        // The arm must not fire where the resident graph cannot serve.
        let linux = plan_on(platform("linux", "x86_64", &["avx2"]));
        assert_ne!(linux.plan.selected_backend, "metal_resident_qwen35_runtime");
        let no_metal_device = plan_on(platform("macos", "aarch64", &["dotprod"]));
        assert_ne!(
            no_metal_device.plan.selected_backend,
            "metal_resident_qwen35_runtime"
        );
        clear_profile_env();
    }

    /// `fixture` with explicit tensor types and an optional declared
    /// `general.file_type`, for the quant-label truth tests.
    fn quant_fixture(
        name: &str,
        file_type: Option<u32>,
        tensor_types: &[GgufTensorType],
    ) -> GgufFile {
        let mut gguf = fixture(name);
        if let Some(ft) = file_type {
            gguf.metadata
                .insert("general.file_type".into(), GgufMetadataValue::U32(ft));
        }
        gguf.tensor_count = tensor_types.len() as i64;
        gguf.tensors = tensor_types
            .iter()
            .enumerate()
            .map(|(i, t)| GgufTensorDescriptor {
                name: format!("blk.{i}.attn_q.weight"),
                dimensions: vec![32, 32],
                tensor_type: *t,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 34,
            })
            .collect();
        gguf
    }

    /// A gemma4 row shaped like the real exports: one block of layer projections at
    /// `proj_type`, the F32 PLE per-layer matrices, and a tied head at `head_type`.
    fn gemma4_row(name: &str, proj_type: GgufTensorType, head_type: GgufTensorType) -> GgufFile {
        let mut gguf = fixture(name);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma4".into()),
        );
        let mut tensors: Vec<GgufTensorDescriptor> = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]
        .iter()
        .map(|suffix| GgufTensorDescriptor {
            name: format!("blk.0.{suffix}"),
            dimensions: vec![32, 32],
            tensor_type: proj_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 34,
        })
        .collect();
        // The PLE per-layer matrices and norms: uploaded as f32 after a CPU dequant,
        // so their wire format says nothing about lane coverage.
        for suffix in ["proj.weight", "inp_gate.weight", "attn_norm.weight"] {
            tensors.push(GgufTensorDescriptor {
                name: format!("blk.0.{suffix}"),
                dimensions: vec![32, 32],
                tensor_type: GgufTensorType::F32,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 4096,
            });
        }
        for name in ["token_embd.weight", "per_layer_token_embd.weight"] {
            tensors.push(GgufTensorDescriptor {
                name: name.into(),
                dimensions: vec![32, 32],
                tensor_type: head_type,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 4096,
            });
        }
        gguf.tensor_count = tensors.len() as i64;
        gguf.tensors = tensors;
        gguf
    }

    #[test]
    fn gemma4_quant_admission_follows_the_projections_not_the_head() {
        // The row that regressed: Q4_0 projections under a Q4_K tied head. It must be
        // ADMITTED — the mis-decode was the head upload skipping its lane's repack
        // (`gemma4_runtime::gemma4_head_upload`), not the Q4_0 GEMV, and the fix has a
        // 5/5 token-identical receipt in qa/gemma3-cuda/phase5/.
        gemma4_projection_quant_admitted(&gemma4_row(
            "e2b-q4_0",
            GgufTensorType::Q4_0,
            GgufTensorType::Q4K,
        ))
        .expect("Q4_0 projections carry a greedy-parity receipt");

        // The Q8_0 bring-up row keeps its admission.
        gemma4_projection_quant_admitted(&gemma4_row(
            "e4b-q8_0",
            GgufTensorType::Q8_0,
            GgufTensorType::Q8_0,
        ))
        .expect("the Q8_0 bring-up row stays admitted");

        // The mirror of the regressed row, and the reason admission reads projections
        // rather than "does any tensor say Q8_0": a Q4_K_M row has a Q8_0 tied head, so
        // the old test admitted it, the plan disclosed the CUDA lane, and the load site
        // then refused it (`nvfp4_cuda_lane_check`) and served on the CPU.
        let err = gemma4_projection_quant_admitted(&gemma4_row(
            "e4b-q4_k_m",
            GgufTensorType::Q4K,
            GgufTensorType::Q8_0,
        ))
        .expect_err("K-quant projections have no receipt on this lane");
        assert!(
            err.contains("attn_q.weight") && err.contains("Q4K"),
            "the decline must name the tensor and its format: {err}"
        );

        // NVFP4 is lane-COVERED but has no gemma4 row receipt. Kernel parity is not
        // row parity — that distinction is exactly what the Q4_0 mis-decode was.
        gemma4_projection_quant_admitted(&gemma4_row(
            "e4b-nvfp4",
            GgufTensorType::NVFP4,
            GgufTensorType::Q6K,
        ))
        .expect_err("NVFP4 has kernel parity but no gemma4 end-to-end receipt");
    }

    #[test]
    fn gemma4_quant_admission_declines_q2_k_and_admits_the_mixed_26b_row() {
        // Q2_K carries a NEGATIVE receipt, not merely a missing one: `mixq2k` requantised
        // the routed experts to Q2_K and decoded degenerate output on all three A/B
        // prompts. `gemma4_projection_tensors` covers `ffn_*_exps`, so admitting Q2_K
        // here would admit precisely the tensors that were measured bad.
        let err = gemma4_projection_quant_admitted(&gemma4_row(
            "26b-q2_k",
            GgufTensorType::Q2K,
            GgufTensorType::Q8_0,
        ))
        .expect_err("Q2_K projections decoded degenerate output; they are not receipted");
        assert!(
            err.contains("Q2K"),
            "the decline must name the format: {err}"
        );

        // The real 26B-A4B ghost row is MIXED and must stay admitted: 14/30 attn_q are
        // Q6_K and 16/30 are Q4_0, ffn_down is 27xQ4_0 + 3xQ4_1, ffn_down_exps is
        // 23xQ4_0 + 7xQ4_1. Pinning the mix is the point — a single-format fixture
        // cannot catch a regression that declines only the Q6_K half of the row.
        let mut mixed = gemma4_row("26b-a4b-q4_0", GgufTensorType::Q4_0, GgufTensorType::Q8_0);
        for (layer, suffix, ty) in [
            (1_usize, "attn_q.weight", GgufTensorType::Q6K),
            (2, "ffn_down.weight", GgufTensorType::Q4_1),
            (3, "ffn_gate_up_exps.weight", GgufTensorType::Q4_0),
            (4, "ffn_down_exps.weight", GgufTensorType::Q4_1),
        ] {
            mixed.tensors.push(GgufTensorDescriptor {
                name: format!("blk.{layer}.{suffix}"),
                dimensions: vec![32, 32],
                tensor_type: ty,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 34,
            });
        }
        mixed.tensor_count = mixed.tensors.len() as i64;
        gemma4_projection_quant_admitted(&mixed)
            .expect("the mixed Q4_0/Q4_1/Q6_K 26B-A4B row is the row this lane exists to serve");

        // And a single Q2_K expert tensor anywhere in that row must still sink it.
        mixed.tensors.push(GgufTensorDescriptor {
            name: "blk.5.ffn_down_exps.weight".into(),
            dimensions: vec![32, 32],
            tensor_type: GgufTensorType::Q2K,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 34,
        });
        mixed.tensor_count = mixed.tensors.len() as i64;
        let err = gemma4_projection_quant_admitted(&mixed)
            .expect_err("one Q2_K expert tensor is still unreceipted");
        assert!(
            err.contains("ffn_down_exps") && err.contains("Q2K"),
            "the decline must name the offending expert tensor: {err}"
        );
    }

    #[test]
    fn gemma4_quant_admission_ignores_f32_ple_and_norm_tensors() {
        // Regression guard for merging this set with `is_gemma4_layer_projection` (the
        // VRAM-accounting set, which counts `proj`/`inp_gate`). Those are F32 on the
        // wire, so folding them in would decline every real gemma4 row — including the
        // Q8_0 one that has always worked.
        let row = gemma4_row("e2b-q4_0", GgufTensorType::Q4_0, GgufTensorType::Q4K);
        assert!(
            row.tensors
                .iter()
                .any(|t| t.name == "blk.0.proj.weight" && t.tensor_type == GgufTensorType::F32),
            "fixture must carry the F32 PLE tensors this test is about"
        );
        let names: Vec<&str> = gemma4_projection_tensors(&row)
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names.len(), 7, "only the 7 GEMV'd projections: {names:?}");
        for excluded in [
            "blk.0.proj.weight",
            "blk.0.inp_gate.weight",
            "blk.0.attn_norm.weight",
            "token_embd.weight",
            "per_layer_token_embd.weight",
        ] {
            assert!(
                !names.contains(&excluded),
                "{excluded} is not a GEMV'd projection: {names:?}"
            );
        }
    }

    fn clear_profile_env() {
        for key in [
            "CAMELID_PROFILE",
            "CAMELID_FORWARD_RSS_TIMINGS",
            "CAMELID_MAC_Q8_REPACK",
            "CAMELID_MAC_Q8_PREFILL_I8MM",
            "CAMELID_MAC_Q8_SCHED",
            "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
            "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            "CAMELID_METAL_RESIDENT_DECODE",
            "CAMELID_METAL_KQUANT",
            "CAMELID_METAL_F32Y",
            "CAMELID_METAL_WIRE",
            "CAMELID_LFM2_METAL",
            "CAMELID_QWEN35_CUDA",
            "CAMELID_X86_Q8_REPACK",
            "CAMELID_X86_Q8_KERNEL",
            "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
            "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
            "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
            "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
            "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
            "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
            "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
            "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
            "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
            "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
            "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
            "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
            "CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER",
            "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
            "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
        ] {
            env::remove_var(key);
        }
    }

    #[test]
    fn junk_general_name_falls_back_to_recognizable_filename() {
        // A junk general.name ("hub", from some conversions) maps to no support row; it
        // must defer to a recognizable filename so the model reaches its validated row
        // instead of failing closed — and must never override a name that already matches.
        let row = exact_model_row(
            &PathBuf::from("/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"),
            &fixture("hub"),
        );
        assert!(
            is_supported_exact_q8_row(&row, &fixture("hub")),
            "junk general.name must fall back to the recognized filename; got {row:?}"
        );
        // Junk name AND unrecognizable filename stays unrecognized.
        assert_eq!(
            recognized_row_level(&exact_model_row(
                &PathBuf::from("/models/mystery.gguf"),
                &fixture("hub")
            )),
            "unknown_or_unvalidated"
        );
        // A recognized general.name is never overridden by an unrelated filename.
        assert_eq!(
            exact_model_row(
                &PathBuf::from("/models/whatever.gguf"),
                &fixture("Llama 3.2 1B Instruct")
            ),
            "Llama 3.2 1B Instruct"
        );
    }

    #[test]
    fn lfm2_filename_is_recognized_without_platform_blind_support_or_dense_q8_admission() {
        let capabilities_status = crate::api::capabilities_response()
            .model_compatibility
            .iter()
            .find(|target| target.id == "lfm2_5_2_6b_q8_0")
            .expect("the LFM2.5-2.6B Q8_0 row must be advertised")
            .status;
        assert_eq!(capabilities_status, "supported_exact_row_smoke");

        let mut gguf = quant_fixture(
            "799e37a4e60bdaae",
            None,
            &[GgufTensorType::Q8_0, GgufTensorType::F32],
        );
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("lfm2".into()),
        );
        let row = exact_model_row(&PathBuf::from("/models/LFM2.5-2.6B-Q8_0.gguf"), &gguf);

        assert_eq!(row, "LFM2.5-2.6B-Q8_0.gguf");
        assert_eq!(
            recognized_row_level(&row),
            "recognized_lfm2_5_2_6b_exact_row"
        );
        assert_eq!(support_level(&row, "Q8_0"), "unknown_or_unvalidated");
        assert_eq!(support_level(&row, "Q4_K_M"), "unknown_or_unvalidated");
        assert!(
            !is_supported_exact_q8_row(&row, &gguf),
            "LFM2 is certified on its runnable lane, not the optimized dense Q8 engine"
        );
    }

    fn lfm2_q8_fixture() -> GgufFile {
        let mut gguf = quant_fixture(
            "799e37a4e60bdaae",
            None,
            &[GgufTensorType::Q8_0, GgufTensorType::F32],
        );
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("lfm2".into()),
        );
        gguf
    }

    fn lfm2_plan(filename: &str, platform: PlanPlatform) -> ExecutionPlanOutcome {
        plan_for_model_with_platform(
            &PathBuf::from(format!("/models/{filename}")),
            &lfm2_q8_fixture(),
            Some(8),
            platform,
        )
    }

    fn phi3_q8_fixture() -> GgufFile {
        let mut gguf = quant_fixture("Phi3", None, &[GgufTensorType::Q8_0, GgufTensorType::F32]);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("phi3".into()),
        );
        gguf
    }

    fn phi3_plan(filename: &str, platform: PlanPlatform) -> ExecutionPlanOutcome {
        plan_for_model_with_platform(
            &PathBuf::from(format!("/models/{filename}")),
            &phi3_q8_fixture(),
            Some(8),
            platform,
        )
    }

    #[test]
    fn lfm2_support_level_is_limited_to_the_two_receipted_execution_lanes() {
        let _guard = env_lock();
        clear_profile_env();

        let windows = lfm2_plan(
            "LFM2.5-2.6B-Q8_0.gguf",
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(windows.plan.profile, ExecutionProfile::Auto);
        assert_eq!(windows.plan.selected_backend, "cpu_reference");
        assert_eq!(windows.plan.prefill_path, "safe_cpu_prefill");
        assert_eq!(windows.plan.decode_path, "safe_cpu_decode");
        assert_eq!(windows.plan.support_level, "supported_exact_row_smoke");

        let mac = lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", receipted_lfm2_m4_platform());
        assert_eq!(mac.plan.profile, ExecutionProfile::Auto);
        assert_eq!(mac.plan.selected_backend, "metal_resident_lfm2_runtime");
        assert_eq!(mac.plan.prefill_path, "lfm2_metal_resident_prefill");
        assert_eq!(mac.plan.decode_path, "lfm2_metal_resident_decode");
        assert_eq!(mac.plan.support_level, "supported_exact_row_smoke");

        let linux = lfm2_plan(
            "LFM2.5-2.6B-Q8_0.gguf",
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(linux.plan.support_level, "unknown_or_unvalidated");

        clear_profile_env();
    }

    #[test]
    fn lfm2_support_level_fails_closed_off_the_exact_m4_receipt() {
        let _guard = env_lock();
        clear_profile_env();

        let mut m3 = receipted_lfm2_m4_platform();
        m3.host_model_identifier = "Mac15,6".into();
        m3.cpu_model = "Apple M3".into();
        assert_eq!(
            lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", m3).plan.support_level,
            "unknown_or_unvalidated"
        );

        let mut other_model_identifier = receipted_lfm2_m4_platform();
        other_model_identifier.host_model_identifier = "Mac16,11".into();
        assert_eq!(
            lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", other_model_identifier)
                .plan
                .support_level,
            "unknown_or_unvalidated"
        );

        let mut other_os_version = receipted_lfm2_m4_platform();
        other_os_version.operating_system_version = "26.6".into();
        assert_eq!(
            lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", other_os_version)
                .plan
                .support_level,
            "unknown_or_unvalidated"
        );

        let mut cpu_fallback = platform("macos", "aarch64", &["dotprod", "i8mm"]);
        cpu_fallback.operating_system_version = "26.5".into();
        cpu_fallback.host_model_identifier = "Mac16,10".into();
        cpu_fallback.cpu_model = "Apple M4".into();
        let cpu_fallback = lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", cpu_fallback);
        assert_eq!(cpu_fallback.plan.selected_backend, "cpu_reference");
        assert_eq!(cpu_fallback.plan.support_level, "unknown_or_unvalidated");

        env::set_var("CAMELID_PROFILE", "safe");
        let safe_mac = lfm2_plan("LFM2.5-2.6B-Q8_0.gguf", receipted_lfm2_m4_platform());
        assert_eq!(safe_mac.plan.profile, ExecutionProfile::Safe);
        assert_eq!(safe_mac.plan.selected_backend, "cpu_reference");
        assert_eq!(safe_mac.plan.support_level, "unknown_or_unvalidated");
        let safe_windows = lfm2_plan(
            "LFM2.5-2.6B-Q8_0.gguf",
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(safe_windows.plan.profile, ExecutionProfile::Safe);
        assert_eq!(safe_windows.plan.support_level, "unknown_or_unvalidated");
        env::remove_var("CAMELID_PROFILE");

        let neighboring_model =
            lfm2_plan("LFM2.5-2.6B-Base-Q8_0.gguf", receipted_lfm2_m4_platform());
        assert_eq!(
            neighboring_model.plan.support_level,
            "unknown_or_unvalidated"
        );

        let mut renamed = lfm2_q8_fixture();
        renamed.metadata.insert(
            "general.name".into(),
            GgufMetadataValue::String("LFM2.5-2.6B".into()),
        );
        let renamed = plan_for_model_with_platform(
            &PathBuf::from("/models/repacked-lfm2.gguf"),
            &renamed,
            Some(8),
            receipted_lfm2_m4_platform(),
        );
        assert_eq!(
            renamed.plan.support_level, "unknown_or_unvalidated",
            "a general.name match cannot promote a renamed artifact"
        );

        clear_profile_env();
    }

    #[test]
    fn phi3_mini_q8_support_is_exact_and_windows_only() {
        let _guard = env_lock();
        clear_profile_env();

        let windows = phi3_plan(
            "Phi-3-mini-4k-instruct-Q8_0.gguf",
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(
            windows.plan.exact_model_row,
            "Phi-3-mini-4k-instruct-Q8_0.gguf"
        );
        assert_eq!(windows.plan.selected_backend, "cpu_reference");
        assert_eq!(windows.plan.prefill_path, "safe_cpu_prefill");
        assert_eq!(windows.plan.decode_path, "safe_cpu_decode");
        assert_eq!(windows.plan.support_level, "supported_exact_row_smoke");

        for other in [
            phi3_plan(
                "Phi-3-mini-4k-instruct-Q8_0.gguf",
                platform("linux", "x86_64", &["avx2"]),
            ),
            phi3_plan(
                "Phi-3-mini-4k-instruct-Q8_0.gguf",
                platform("windows", "aarch64", &[]),
            ),
            phi3_plan(
                "repacked-phi3.gguf",
                platform("windows", "x86_64", &["avx2"]),
            ),
        ] {
            assert_eq!(other.plan.support_level, "unknown_or_unvalidated");
        }

        env::set_var("CAMELID_PROFILE", "safe");
        assert_eq!(
            phi3_plan(
                "Phi-3-mini-4k-instruct-Q8_0.gguf",
                platform("windows", "x86_64", &["avx2"]),
            )
            .plan
            .support_level,
            "unknown_or_unvalidated"
        );
        clear_profile_env();
    }

    #[test]
    fn junk_named_recognizable_8b_takes_gpu_lane_not_cpu() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"),
            &fixture("hub"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        clear_profile_env();
        assert_eq!(
            outcome.plan.exact_model_row,
            "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
        );
        assert_ne!(
            outcome.plan.selected_backend, "cpu_reference",
            "a recognizable 8B with a junk general.name must not fail closed to CPU"
        );
    }

    #[test]
    fn gpu_runnable_tier_admits_uncurated_q8_llama_by_default_optout_forces_cpu() {
        let _guard = env_lock();
        clear_profile_env();
        // Uncurated: neither general.name ("hub") nor the filename maps to a support row.
        let uncurated = fixture("hub");
        let path = PathBuf::from("/models/my-custom-llama-Q8_0.gguf");
        // DEFAULT (unset): admitted to the GPU-runnable tier with an honest, distinct label; the
        // support_level stays unknown_or_unvalidated (never claims a supported row). Admission is
        // gated at runtime by the parity self-check, so default-on is safe.
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        let on = plan_for_model_with_platform(
            &path,
            &uncurated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(
            on.plan.selected_backend,
            "cuda_resident_q8_runtime_runnable_unvalidated"
        );
        assert_eq!(
            on.plan.support_level, "unknown_or_unvalidated",
            "the runnable tier must never claim a supported row"
        );
        // Explicit opt-out (=0): forced back to the safe CPU reference path.
        env::set_var("CAMELID_GPU_RUNNABLE_TIER", "0");
        let off = plan_for_model_with_platform(
            &path,
            &uncurated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        clear_profile_env();
        assert_eq!(off.plan.selected_backend, "cpu_reference");
    }

    #[test]
    fn gpu_runnable_tier_never_changes_a_curated_row() {
        let _guard = env_lock();
        clear_profile_env();
        let curated = fixture("Llama 3.2 1B Instruct");
        let path = PathBuf::from("/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
        // Tier ON (default, unset).
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        let on = plan_for_model_with_platform(
            &path,
            &curated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        // Tier opted out (=0).
        env::set_var("CAMELID_GPU_RUNNABLE_TIER", "0");
        let off = plan_for_model_with_platform(
            &path,
            &curated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        clear_profile_env();
        // A curated row takes the supported plan and is byte-for-byte identical whether the tier
        // is on (default) or opted out — the tier is a pure additive else-branch after the
        // curated arm, so it can never alter a supported row.
        assert_eq!(on.plan.selected_backend, "cuda_resident_q8_runtime");
        assert_eq!(on.plan.selected_backend, off.plan.selected_backend);
        assert_eq!(on.plan.support_level, off.plan.support_level);
        assert_eq!(on.plan.decode_path, off.plan.decode_path);
    }

    #[test]
    fn safe_profile_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "safe");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(8),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Safe);
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "always_retained_reference_path"
        );
        assert!(outcome
            .plan
            .diagnostics_status
            .contains("RSS timings disabled by default"));
        assert!(!outcome
            .env_updates
            .contains_key("CAMELID_FORWARD_RSS_TIMINGS"));
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
        clear_profile_env();
    }

    #[test]
    fn safe_profile_preserves_explicit_forward_rss_timings_through_plan_apply() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "safe");
        env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "on");
        let planner_env = PlannerEnv::capture();
        let outcome = plan_for_model_with_platform_and_env(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
            &planner_env,
        );

        assert_eq!(outcome.plan.profile, ExecutionProfile::Safe);
        assert!(outcome
            .plan
            .diagnostics_status
            .contains("operator-requested RSS timings enabled"));
        assert_eq!(
            outcome.env_updates.get("CAMELID_FORWARD_RSS_TIMINGS"),
            Some(&Some("on"))
        );

        // This is the exact capture -> plan -> apply sequence used by
        // POST /models/load. A stale live value must not override the captured
        // operator request, and applying the plan must not remove that request.
        env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "off");
        planner_env.apply(&outcome.env_updates);
        assert_eq!(env::var("CAMELID_FORWARD_RSS_TIMINGS").as_deref(), Ok("on"));
        clear_profile_env();
    }

    #[test]
    fn auto_and_experimental_preserve_explicit_forward_rss_timings() {
        let _guard = env_lock();
        for (requested, expected) in [
            ("auto", ExecutionProfile::Auto),
            ("experimental", ExecutionProfile::Experimental),
        ] {
            clear_profile_env();
            env::set_var("CAMELID_PROFILE", requested);
            env::set_var("CAMELID_FORWARD_RSS_TIMINGS", "on");
            let planner_env = PlannerEnv::capture();
            let outcome = plan_for_model_with_platform_and_env(
                &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
                &fixture("Llama 3.2 3B Instruct"),
                Some(8),
                platform("windows", "x86_64", &["avx2"]),
                &planner_env,
            );

            assert_eq!(outcome.plan.profile, expected);
            assert!(outcome
                .plan
                .diagnostics_status
                .contains("operator-requested RSS timings enabled"));
            assert_eq!(
                outcome.env_updates.get("CAMELID_FORWARD_RSS_TIMINGS"),
                Some(&Some("on"))
            );
        }
        clear_profile_env();
    }

    /// `fixture` with the arch overridden to gemma3 (windowed attention).
    fn gemma3_fixture(name: &str) -> GgufFile {
        let mut gguf = fixture(name);
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma3".into()),
        );
        gguf
    }

    /// gemma3→CUDA Phase 2: the curated row must reach the CUDA-resident
    /// WINDOWED plan on a CUDA host — distinct labels from the Llama-family
    /// CUDA plan, because it is a different forward (dual-θ RoPE, a
    /// sliding-window kernel, sandwich post-norms, GeGLU) and conflating them in
    /// the disclosure would hide which code ran.
    #[test]
    fn gemma3_q8_row_selects_the_cuda_resident_windowed_plan_on_a_cuda_host() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        clear_profile_env();
        assert_eq!(
            outcome.plan.selected_backend, "cuda_resident_windowed_runtime",
            "a windowed row on a CUDA host must disclose the windowed CUDA lane"
        );
        assert_eq!(
            outcome.plan.decode_path,
            "q8_0_cuda_resident_windowed_decode"
        );
        assert_eq!(
            outcome.plan.prefill_path, "windowed_token_by_token_resident_prefill",
            "the batched/flash prefill kernels carry no window, so the plan must \
             not advertise them for a windowed row"
        );
    }

    /// The D20 invariant on the CUDA side: routing consults
    /// `windowed_arch_cuda_resident_plan_selectable`, so an operator opt-out must
    /// move the PLAN too. Without this the plan would say "CUDA windowed" while
    /// serve ran the runnable bridge — the exact disagreement Phase 3c of the
    /// Metal campaign existed to kill. Deleting either clause fails this.
    #[test]
    fn windowed_cuda_plan_selectability_tracks_its_opt_outs() {
        let _guard = env_lock();
        clear_profile_env();
        assert!(
            windowed_arch_cuda_resident_plan_selectable(),
            "auto profile with no opt-out must allow the windowed CUDA plan"
        );
        env::set_var("CAMELID_GEMMA3_CUDA_RESIDENT", "0");
        assert!(
            !windowed_arch_cuda_resident_plan_selectable(),
            "CAMELID_GEMMA3_CUDA_RESIDENT=0 must disarm the plan, not just routing"
        );
        env::remove_var("CAMELID_GEMMA3_CUDA_RESIDENT");
        env::set_var("CAMELID_PROFILE", "safe");
        assert!(
            !windowed_arch_cuda_resident_plan_selectable(),
            "the safe profile must disarm the windowed CUDA plan"
        );
        clear_profile_env();
    }

    /// With the CUDA windowed plan opted out, a windowed row must fail CLOSED to
    /// the safe plan and say the bridge serves it — never advertise a CPU dense
    /// lane, which for this arch is fail-closed at every per-layer dispatch (H4).
    #[test]
    fn gemma3_falls_closed_to_the_bridge_when_the_cuda_windowed_plan_is_opted_out() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_GEMMA3_CUDA_RESIDENT", "0");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        env::remove_var("CAMELID_GEMMA3_CUDA_RESIDENT");
        clear_profile_env();
        assert_ne!(
            outcome.plan.selected_backend,
            "cuda_resident_windowed_runtime"
        );
        assert!(
            outcome
                .plan
                .reasons
                .iter()
                .any(|r| r.contains("runnable bridge")),
            "the opted-out plan must name the bridge as the serving lane: {:?}",
            outcome.plan.reasons
        );
    }

    /// The fit projection must be based on what the gemma4 CUDA lane actually
    /// uploads, not the whole file. REGRESSION TEST for a real defect: the first
    /// version summed every tensor, projected 5055 MiB for `gemma-4-E2B-it-Q8_0`
    /// and DECLINED it — while the lane actually uses 2635 MiB and serves the row
    /// in 794 ms on a 6 GB card. gemma4 is a PLE matformer whose per-layer
    /// embedding tables are LARGER than its projections and never reach VRAM.
    #[test]
    fn gemma4_fit_projection_counts_only_what_reaches_vram() {
        assert!(is_gemma4_layer_projection("blk.0.attn_q.weight"));
        assert!(is_gemma4_layer_projection("blk.17.ffn_down.weight"));
        assert!(is_gemma4_layer_projection("blk.3.proj.weight"));
        // The PLE tables and the token embedding are the whole point: they are
        // the bulk of the file and they do NOT go to VRAM.
        assert!(!is_gemma4_layer_projection("per_layer_token_embd.weight"));
        assert!(!is_gemma4_layer_projection("per_layer_model_proj.weight"));
        assert!(!is_gemma4_layer_projection("token_embd.weight"));
        assert!(!is_gemma4_layer_projection("output_norm.weight"));
        // Norms are small and counted in the overhead constant, not per-tensor.
        assert!(!is_gemma4_layer_projection("blk.0.attn_norm.weight"));
        assert!(!is_gemma4_layer_projection("blk.0.post_ffw_norm.weight"));
    }

    /// The projection must admit a row that measurably fits. Built from the real
    /// E2B Q8_0 shape: 1879 MiB of layer projections against the 5122 MiB free
    /// that an RTX 3060 reports at load. Whole-file accounting projected 5055 MiB
    /// here and refused; this asserts the corrected basis admits it.
    #[test]
    fn gemma4_e2b_q8_shape_fits_a_six_gigabyte_card() {
        const MIB: u64 = 1024 * 1024;
        let projections_mib: u64 = 1879;
        let projected = projections_mib * MIB + GEMMA4_RESIDENT_OVERHEAD_MIB * MIB;
        let free = 5122 * MIB;
        assert!(
            crate::cuda_vram::evaluate(free, projected, 512).is_ok(),
            "the measured E2B Q8_0 shape must be admitted on a 6 GB card: \
             projected {} MiB against {} MiB free",
            projected / MIB,
            free / MIB
        );
        // And a genuinely oversized row must still be refused: E4B Q8_0 is
        // 3998 MiB of projections, which does not fit with headroom.
        let e4b = 3998 * MIB + GEMMA4_RESIDENT_OVERHEAD_MIB * MIB;
        assert!(
            crate::cuda_vram::evaluate(free, e4b, 512).is_err(),
            "E4B Q8_0 must still be refused on a 6 GB card"
        );
    }

    #[test]
    fn gemma3_q8_row_selects_metal_resident_plan_on_a_resident_mac() {
        // gemma3→Metal Phase 3b: the curated gemma-3-1b-it Q8_0 row must reach
        // the Metal-resident Q8 plan on a resident-capable macOS host — this
        // selection was absent from the historical checklist and is
        // load-bearing (§3 Phase 3).
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        assert_eq!(outcome.plan.selected_backend, "metal_resident_q8_runtime");
        assert_eq!(outcome.plan.decode_path, "q8_0_metal_resident_decode");
        assert_eq!(
            outcome.plan.support_level,
            "supported_exact_row_smoke_sub512"
        );
    }

    /// The planner must not read its own output back as an operator opt-out.
    ///
    /// A successful Metal-resident selection WRITES `CAMELID_MAC_Q8_REPACK=off`
    /// (the GPU consumes plain blocks, so the CPU rows4 repack is disabled).
    /// While the planner read that key live, the SECOND `plan_for_model` in a
    /// process saw "off" as an operator opt-out and fell through to
    /// `safe_q8_plan` — so serve's second model load disclosed `cpu_reference` /
    /// `safe_cpu_decode` on `/v1/health`, `/api/capabilities` and
    /// `/execution-plan` while it went on decoding at ~47 tok/s on the Metal
    /// resident lane. Reproduced 2026-07-31 on main `56ff2eb3`: the same
    /// Llama 3.2 1B file planned `metal_resident_q8_runtime` when loaded at
    /// startup and `cpu_reference` when re-loaded through
    /// `POST /api/models/load`.
    ///
    /// This drives the real sequence — plan, apply, plan again against the SAME
    /// baseline — because that is what serve does per model load.
    #[test]
    fn planning_twice_in_one_process_does_not_read_the_first_plans_output() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        // Captured before any plan applied its env_updates, exactly as
        // `AppState::new` captures it before serve's first load.
        let planner_env = PlannerEnv::capture();
        let platform = || metal_platform("macos", "aarch64", &["dotprod", "i8mm"]);

        let first = plan_for_model_with_platform_and_env(
            &PathBuf::from("/models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 1B Instruct"),
            Some(8),
            platform(),
            &planner_env,
        );
        assert_eq!(first.plan.selected_backend, "metal_resident_q8_runtime");
        // The load pipeline applies the plan's env before the next load plans.
        planner_env.apply(&first.env_updates);
        assert_eq!(
            env::var("CAMELID_MAC_Q8_REPACK").as_deref(),
            Ok("off"),
            "precondition: the Metal selection writes the key this test is about"
        );

        let second = plan_for_model_with_platform_and_env(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            platform(),
            &planner_env,
        );
        // A third, to catch a fix that only survives one round trip.
        planner_env.apply(&second.env_updates);
        let third = plan_for_model_with_platform_and_env(
            &PathBuf::from("/models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 1B Instruct"),
            Some(8),
            platform(),
            &planner_env,
        );
        clear_profile_env();

        assert_eq!(
            second.plan.selected_backend, "metal_resident_q8_runtime",
            "the second load must disclose the lane it actually runs on, not the \
             first load's CAMELID_MAC_Q8_REPACK=off output"
        );
        assert_eq!(second.plan.decode_path, "q8_0_metal_resident_decode");
        assert!(
            !second
                .plan
                .reasons
                .iter()
                .any(|reason| reason.contains("CAMELID_MAC_Q8_REPACK disables Mac repack")),
            "the second plan must not blame an operator opt-out the operator never set: {:?}",
            second.plan.reasons
        );
        assert_eq!(third.plan.selected_backend, "metal_resident_q8_runtime");
        assert_eq!(third.plan.decode_path, "q8_0_metal_resident_decode");
    }

    /// The other direction: pinning the planner to an operator baseline must not
    /// deafen it to a REAL opt-out. `CAMELID_MAC_Q8_REPACK=0` set before the
    /// baseline is captured still fails the plan closed to the safe path.
    #[test]
    fn operator_set_repack_opt_out_survives_the_planner_env_baseline() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        env::set_var("CAMELID_MAC_Q8_REPACK", "0");
        let planner_env = PlannerEnv::capture();
        let outcome = plan_for_model_with_platform_and_env(
            &PathBuf::from("/models/Llama-3.2-1B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 1B Instruct"),
            Some(8),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
            &planner_env,
        );
        clear_profile_env();
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert!(outcome
            .plan
            .reasons
            .iter()
            .any(|reason| reason.contains("CAMELID_MAC_Q8_REPACK disables Mac repack")));
    }

    /// Phase 3c triage: the disclosed execution plan and the live serve
    /// routing must agree. `windowed_arch_resident_host_available` consults
    /// `macos_q8_metal_plan_selectable`, so an operator opting out of the Metal
    /// plan gets the runnable bridge rather than a plan that says
    /// `cpu_reference` while serve quietly runs the Metal-resident lane.
    /// Deleting any clause of the predicate makes this fail.
    #[test]
    fn macos_q8_metal_plan_selectability_tracks_every_early_return() {
        let _guard = env_lock();
        clear_profile_env();
        assert!(
            macos_q8_metal_plan_selectable(),
            "the default (auto profile, no opt-outs) must allow the Metal plan"
        );

        // CAMELID_MAC_Q8_REPACK is NOT an input: the plan WRITES it to "off"
        // on a successful Metal selection, so consulting it would be a
        // self-defeating latch that disarmed resident routing the instant
        // PlannerEnv::apply ran. Pinned explicitly so it is not "helpfully"
        // added back.
        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_REPACK", "off");
        assert!(
            macos_q8_metal_plan_selectable(),
            "the plan's own CAMELID_MAC_Q8_REPACK=off output must not disarm routing"
        );
        env::remove_var("CAMELID_MAC_Q8_REPACK");

        for (key, value) in [
            ("CAMELID_PROFILE", "safe"),
            // An unrecognized profile fails closed to Safe — same outcome.
            ("CAMELID_PROFILE", "nonsense"),
            ("CAMELID_MAC_Q8_METAL_PLAN", "0"),
        ] {
            clear_profile_env();
            env::remove_var("CAMELID_MAC_Q8_METAL_PLAN");
            env::set_var(key, value);
            assert!(
                !macos_q8_metal_plan_selectable(),
                "{key}={value} must make the Metal plan unselectable, so routing sends a \
                 windowed arch to the runnable bridge"
            );
            env::remove_var(key);
        }
        clear_profile_env();
        env::remove_var("CAMELID_MAC_Q8_METAL_PLAN");
    }

    /// Phase 3c triage: under the Safe profile the windowed plan used to
    /// disclose a bare "safe profile" reason, leaving no way to tell that
    /// gemma3 chats were being served by the bridge rather than by the
    /// `cpu_reference` / `safe_cpu_decode` labels the plan named — labels that
    /// for this arch fail closed at every per-layer dispatch (H4).
    #[test]
    fn windowed_safe_profile_plan_discloses_the_runnable_bridge() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "safe");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert!(
            outcome
                .plan
                .reasons
                .iter()
                .any(|reason| reason.contains("runnable bridge")),
            "the safe-profile windowed plan must disclose where chats actually go: {:?}",
            outcome.plan.reasons
        );
    }

    #[test]
    fn gemma3_q8_row_fails_closed_to_safe_plan_where_metal_resident_cannot_run() {
        // The windowed arch has NO CPU dense plan (hazard H4): off-macOS hosts
        // and macOS-without-resident-selection must fail closed to the safe
        // labels — serve chats via the runnable bridge there — and must never
        // advertise the x86 repack / Mac CPU repack lanes.
        let _guard = env_lock();
        clear_profile_env();
        let linux = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(linux.plan.selected_backend, "cpu_reference");
        assert_eq!(linux.plan.decode_path, "safe_cpu_decode");
        // Recognition is not support: the row is still recognized on the
        // fallback host (support_level reflects the curated Q8_0 evidence).
        assert_eq!(linux.plan.support_level, "supported_exact_row_smoke_sub512");

        // macOS with the resident decode gate unset: the Metal selection in
        // select_macos_q8_plan cannot fire, and the CPU repack lanes must not
        // be advertised for a windowed arch.
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        let mac_no_resident = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q8_0.gguf"),
            &gemma3_fixture("gemma-3-1b-it"),
            Some(8),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        assert_eq!(mac_no_resident.plan.selected_backend, "cpu_reference");
        assert_eq!(mac_no_resident.plan.decode_path, "safe_cpu_decode");
    }

    #[test]
    fn gemma3_kquant_never_takes_the_metal_resident_kquant_plan() {
        // Hazard H5: a gemma3 Q4_K_M must not be advertised onto the Metal
        // resident K-quant lane (no windowed K-quant receipt; the gather drops
        // the embed scale). With every Metal K-quant gate armed, the plan must
        // still fall back to the CPU block-dot labels because
        // `is_gpu_runnable_arch` excludes gemma3.
        let _guard = env_lock();
        clear_profile_env();
        for key in [
            "CAMELID_METAL_RESIDENT_DECODE",
            "CAMELID_METAL_KQUANT",
            "CAMELID_METAL_F32Y",
            "CAMELID_METAL_WIRE",
        ] {
            env::set_var(key, "1");
        }
        let mut gguf = gemma3_fixture("gemma-3-1b-it");
        gguf.tensors = vec![GgufTensorDescriptor {
            name: "blk.0.attn_q.weight".into(),
            dimensions: vec![256, 256],
            tensor_type: GgufTensorType::Q4K,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 144,
        }];
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/gemma-3-1b-it-Q4_K_M.gguf"),
            &gguf,
            Some(8),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        assert_ne!(
            outcome.plan.selected_backend, "metal_resident_kquant_runtime",
            "H5: a windowed arch must never be advertised onto the Metal K-quant lane"
        );
        assert_eq!(
            outcome.plan.support_level, "unknown_or_unvalidated",
            "a non-Q8_0 gemma3 must not echo the curated Q8_0 row's level"
        );
    }

    #[test]
    fn kquant_plan_labels_resident_and_cpu_block_dot_not_cpu_reference() {
        // Disclosure fix: a Q4_K_M model must NOT be labeled the dense_or_other /
        // cpu_reference fallback. quant_type is Q4_K_M, and the backend reflects the
        // real lane: GPU-resident when CUDA drives decode, CPU block-dot otherwise,
        // and only cpu_reference when the block-dot is explicitly disabled with no GPU.
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        let mut gguf = fixture("Qwen3 4B Instruct Q4_K_M");
        gguf.tensors[0].tensor_type = GgufTensorType::Q4K;
        let path = PathBuf::from("/tmp/Qwen3-4B-Q4_K_M.gguf");

        let cpu = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(cpu.plan.quant_type, "Q4_K_M");
        assert_eq!(cpu.plan.selected_backend, "cpu_kquant_block_dot");
        assert_eq!(cpu.plan.decode_path, "kquant_cpu_block_dot_decode");

        let gpu = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(gpu.plan.selected_backend, "cuda_resident_kquant_runtime");
        assert_eq!(gpu.plan.decode_path, "kquant_cuda_resident_decode");
        assert!(gpu.plan.cuda_resident_active);

        // CAMELID_X86_Q4K_DECODE=0 no longer changes the LANE, so it must no longer
        // change the DISCLOSURE. K-quant 2-D linears load wire-only, so the block-dot
        // is their only consumer and `kquant_block_dot_selected` keeps it regardless of
        // the flag; this assertion used to expect `cpu_reference`, which would now be a
        // plan describing a run that cannot happen.
        env::set_var("CAMELID_X86_Q4K_DECODE", "0");
        let off = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(off.plan.selected_backend, "cpu_kquant_block_dot");
        assert_eq!(off.plan.decode_path, "kquant_cpu_block_dot_decode");
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        clear_profile_env();
    }

    #[test]
    fn mac_metal_kquant_plan_is_automatic_but_fail_closed_for_unsupported_mix() {
        let _guard = env_lock();
        clear_profile_env();
        for key in [
            "CAMELID_METAL_RESIDENT_DECODE",
            "CAMELID_METAL_KQUANT",
            "CAMELID_METAL_F32Y",
            "CAMELID_METAL_WIRE",
        ] {
            env::set_var(key, "1");
        }
        let path = PathBuf::from("/tmp/Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        let supported = quant_fixture(
            "Llama 3.2 1B Instruct",
            Some(15),
            &[GgufTensorType::Q4K, GgufTensorType::Q6K],
        );
        let metal = plan_for_model_with_platform(
            &path,
            &supported,
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(metal.plan.selected_backend, "metal_resident_kquant_runtime");
        assert_eq!(metal.plan.decode_path, "kquant_metal_resident_decode");
        assert_eq!(metal.plan.prefill_path, "kquant_metal_resident_prefill");
        assert_eq!(
            metal.plan.fallback_path,
            "kquant_cpu_block_dot_reference_path"
        );

        // Q3_K, not Q5_K: Q5_K gained `q5k_linear_simd`/`q5k_linear_tiled` and is
        // now mapped by `resident_metal_format`, so it is no longer an example of an
        // unsupported mix. Q3_K still maps to None and must still fall closed.
        let unsupported = quant_fixture(
            "Llama 3.2 3B Instruct",
            Some(12),
            &[GgufTensorType::Q3K, GgufTensorType::Q6K],
        );
        let fallback = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q3_K_M.gguf"),
            &unsupported,
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(fallback.plan.selected_backend, "cpu_kquant_block_dot");
        assert_eq!(fallback.plan.decode_path, "kquant_cpu_block_dot_decode");

        // Fail closed on a tensor type this reader does not model. IQ2_XXS /
        // IQ3_S parse as `Unknown(_)`, so a deny-list of four named K-quants
        // would have labelled such a file Metal-resident by omission.
        let unmodelled = quant_fixture(
            "Llama 3.3 70B Instruct",
            Some(10),
            &[
                GgufTensorType::Q4K,
                GgufTensorType::Q6K,
                GgufTensorType::Unknown(19),
            ],
        );
        let unmodelled_plan = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.3-70B-Instruct-IQ2_XXS.gguf"),
            &unmodelled,
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            unmodelled_plan.plan.selected_backend,
            "cpu_kquant_block_dot"
        );

        // Fail closed on an architecture the resident dense kernels cannot
        // express, even with a clean Q4_K/Q6_K mix: the runtime's
        // `resident_decode_eligible` rejects gemma2/gemma3 outright, so the plan
        // must not claim the Metal lane for them.
        let mut gemma3 = quant_fixture(
            "Gemma 3 4B Instruct",
            Some(15),
            &[GgufTensorType::Q4K, GgufTensorType::Q6K],
        );
        gemma3.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma3".into()),
        );
        let gemma3_plan = plan_for_model_with_platform(
            &PathBuf::from("/tmp/gemma-3-4b-it-Q4_K_M.gguf"),
            &gemma3,
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(gemma3_plan.plan.selected_backend, "cpu_kquant_block_dot");

        // An unrecognised flag spelling must not label the run Metal either: the
        // Metal runtime accepts only `1`/`true`, so `on` means "off" there and
        // the plan has to agree.
        env::set_var("CAMELID_METAL_KQUANT", "on");
        let loose_flag = plan_for_model_with_platform(
            &path,
            &supported,
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(loose_flag.plan.selected_backend, "cpu_kquant_block_dot");
        clear_profile_env();
    }

    #[test]
    fn pure_q6k_reports_q6k_label_unknown_level_and_keeps_kquant_routing() {
        // Truth-in-labeling: a pure-Q6_K file (declared ftype 18) must not be
        // collapsed into the "Q4_K_M" bucket, and a Q6_K file of the TinyLlama
        // gate row's NAME must not echo the Q8_0 row's support level. Routing
        // is unchanged: it branches on the tensor scan, never on the label.
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        let q6k = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(18), &[GgufTensorType::Q6K]);
        let path = PathBuf::from("/tmp/tinyllama-1.1b-chat-v1.0.Q6_K.gguf");

        let cpu = plan_for_model_with_platform(
            &path,
            &q6k,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(cpu.plan.quant_type, "Q6_K");
        assert_eq!(
            cpu.plan.support_level, "unknown_or_unvalidated",
            "a Q6_K TinyLlama must not inherit the Q8_0 gate row's support level"
        );
        assert_eq!(
            cpu.plan.selected_backend, "cpu_kquant_block_dot",
            "the label fix must not change K-quant routing"
        );

        let gpu = plan_for_model_with_platform(
            &path,
            &q6k,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(gpu.plan.selected_backend, "cuda_resident_kquant_runtime");
        clear_profile_env();
    }

    #[test]
    fn declared_file_type_names_tq_and_kquant_variants() {
        // A TQ1_0 file that carries a few K-quant tensors reports TQ1_0 (not a
        // K-quant mix guess), and a declared Q5_K_M names itself precisely.
        let tq = quant_fixture(
            "Ternary Bonsai 4B",
            Some(36),
            &[GgufTensorType::Tq1_0, GgufTensorType::Q4K],
        );
        assert_eq!(quant_type(&tq), "TQ1_0");
        let q5km = quant_fixture(
            "whatever",
            Some(17),
            &[GgufTensorType::Q5K, GgufTensorType::Q6K],
        );
        assert_eq!(quant_type(&q5km), "Q5_K_M");
        // Undeclared file_type falls back to the tensor-scan buckets.
        let undeclared_q6k = quant_fixture("whatever", None, &[GgufTensorType::Q6K]);
        assert_eq!(quant_type(&undeclared_q6k), "Q6_K");
        let undeclared_mix = quant_fixture(
            "whatever",
            None,
            &[GgufTensorType::Q4K, GgufTensorType::Q6K],
        );
        assert_eq!(quant_type(&undeclared_mix), "Q4_K_M");
    }

    #[test]
    fn prism_q2_geometry_and_metal_lane_are_disclosed_truthfully() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        for architecture in ["qwen3", "qwen35"] {
            for (tensor_type, expected) in [
                (GgufTensorType::Q2_0G64, "Q2_0_G64"),
                (GgufTensorType::Q2_0G128, "Q2_0_G128"),
                (GgufTensorType::Pq2_0, "PQ2_0"),
            ] {
                let mut gguf = quant_fixture("Ternary Bonsai 1.7B", Some(41), &[tensor_type]);
                gguf.metadata.insert(
                    "general.architecture".into(),
                    GgufMetadataValue::String(architecture.into()),
                );
                let outcome = plan_for_model_with_platform(
                    &PathBuf::from("/models/renamed.gguf"),
                    &gguf,
                    Some(8),
                    metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
                );
                assert_eq!(outcome.plan.quant_type, expected);
                assert_eq!(
                    outcome.plan.selected_backend,
                    "metal_resident_prism_low_bit_runtime"
                );
                assert_eq!(
                    outcome.plan.decode_path,
                    "prism_low_bit_metal_resident_decode"
                );
            }
        }
        clear_profile_env();
    }

    #[test]
    fn prism_windows_cuda_lane_is_disclosed_without_macos_certification() {
        let _guard = env_lock();
        clear_profile_env();
        for architecture in ["qwen3", "qwen35"] {
            for (tensor_type, expected) in [
                (GgufTensorType::Q1_0, "Q1_0"),
                (GgufTensorType::Q2_0G64, "Q2_0_G64"),
                (GgufTensorType::Q2_0G128, "Q2_0_G128"),
                (GgufTensorType::Pq2_0, "PQ2_0"),
            ] {
                let mut gguf = quant_fixture("Ternary Bonsai 4B", None, &[tensor_type]);
                gguf.metadata.insert(
                    "general.architecture".into(),
                    GgufMetadataValue::String(architecture.into()),
                );
                let outcome = plan_for_model_with_platform(
                    &PathBuf::from("C:/models/renamed.gguf"),
                    &gguf,
                    Some(8),
                    cuda_platform("windows", "x86_64", &["avx2"]),
                );
                assert_eq!(outcome.plan.quant_type, expected);
                assert_eq!(
                    outcome.plan.selected_backend,
                    "cuda_resident_prism_low_bit_runtime"
                );
                assert_eq!(
                    outcome.plan.decode_path,
                    "prism_low_bit_cuda_resident_decode"
                );
                assert_eq!(outcome.plan.support_level, "unknown_or_unvalidated");
                if architecture == "qwen35" {
                    assert_eq!(
                        outcome.env_updates.get("CAMELID_QWEN35_CUDA"),
                        Some(&Some("on"))
                    );
                } else {
                    assert!(!outcome.env_updates.contains_key("CAMELID_QWEN35_CUDA"));
                }
            }
        }
        env::set_var("CAMELID_QWEN35_CUDA", "0");
        let mut qwen35 = quant_fixture("Ternary Bonsai 27B", Some(40), &[GgufTensorType::Q1_0]);
        qwen35.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen35".into()),
        );
        let disabled = plan_for_model_with_platform(
            &PathBuf::from("C:/models/Bonsai-27B-Q1_0.gguf"),
            &qwen35,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_ne!(
            disabled.plan.selected_backend,
            "cuda_resident_prism_low_bit_runtime"
        );
        clear_profile_env();
    }

    #[test]
    fn exact_bonsai_rows_report_supported_on_certified_metal_and_windows_cuda_lanes() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");

        for (filename, file_type, tensor_type) in [
            ("Bonsai-4B-Q1_0.gguf", 40, GgufTensorType::Q1_0),
            ("Ternary-Bonsai-4B-Q2_0.gguf", 41, GgufTensorType::Q2_0G128),
            ("Ternary-Bonsai-4B-PQ2_0.gguf", 41, GgufTensorType::Pq2_0),
            ("Bonsai-8B-Q1_0.gguf", 40, GgufTensorType::Q1_0),
            ("Ternary-Bonsai-8B-Q2_0.gguf", 41, GgufTensorType::Q2_0G128),
            ("Bonsai-27B-Q1_0.gguf", 40, GgufTensorType::Q1_0),
            ("Ternary-Bonsai-27B-Q2_0.gguf", 41, GgufTensorType::Q2_0G128),
        ] {
            let mut gguf = quant_fixture("hub", Some(file_type), &[tensor_type]);
            gguf.metadata.insert(
                "general.architecture".into(),
                GgufMetadataValue::String("qwen35".into()),
            );
            let outcome = plan_for_model_with_platform(
                &PathBuf::from(format!("/models/{filename}")),
                &gguf,
                Some(8),
                metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
            );
            assert_eq!(
                outcome.plan.support_level, "supported_exact_row_smoke_macos_metal",
                "{filename} support level",
            );
            assert_eq!(
                outcome.plan.selected_backend, "metal_resident_prism_low_bit_runtime",
                "{filename} backend",
            );

            let windows = plan_for_model_with_platform(
                &PathBuf::from(format!("C:/models/{filename}")),
                &gguf,
                Some(8),
                cuda_platform("windows", "x86_64", &["avx2"]),
            );
            assert_eq!(
                windows.plan.support_level, "supported_exact_row_smoke_windows_cuda",
                "{filename} Windows support level",
            );
            assert_eq!(
                windows.plan.selected_backend, "cuda_resident_prism_low_bit_runtime",
                "{filename} Windows backend",
            );
        }

        let mut q1 = quant_fixture("hub", Some(40), &[GgufTensorType::Q1_0]);
        q1.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen35".into()),
        );
        let windows = plan_for_model_with_platform(
            &PathBuf::from("/models/Bonsai-4B-Q1_0.gguf"),
            &q1,
            Some(8),
            platform("windows", "aarch64", &[]),
        );
        assert_eq!(windows.plan.support_level, "unknown_or_unvalidated");
        assert_ne!(
            windows.plan.selected_backend,
            "metal_resident_prism_low_bit_runtime"
        );
        clear_profile_env();
    }

    #[test]
    fn declared_q8_0_without_q8_0_tensors_is_refused() {
        // A wrong general.file_type=7 with zero Q8_0 tensors must not produce
        // the "Q8_0" label — that label is what gates support_level onto a
        // promoted row (and the supported-Q8 planner branch keys on tensors,
        // so the file routes as what it actually is either way).
        let lying = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(7), &[GgufTensorType::Q6K]);
        assert_eq!(quant_type(&lying), "Q6_K");
        assert_eq!(
            support_level("TinyLlama 1.1B Chat v1.0", &quant_type(&lying)),
            "unknown_or_unvalidated"
        );
    }

    #[test]
    fn supported_q8_row_keeps_level_and_kquant_name_variant_reports_unknown() {
        // Anchor: quant-gating must not move any promoted Q8_0 row…
        let _guard = env_lock();
        clear_profile_env();
        let q8 = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(7), &[GgufTensorType::Q8_0]);
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
            &q8,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.quant_type, "Q8_0");
        assert_eq!(outcome.plan.support_level, "supported_current_gate");
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        clear_profile_env();

        // …while a K-quant file sharing a supported row's NAME stays unknown in
        // the plan string (its own K-quant claim lives in /api/capabilities),
        // and row recognition stays quant-blind for the planner gate.
        assert_eq!(
            support_level("Llama 3.2 3B Instruct", "Q4_K_M"),
            "unknown_or_unvalidated"
        );
        assert!(is_supported_exact_q8_row(
            "Llama 3.2 3B Instruct",
            &fixture("Llama 3.2 3B Instruct")
        ));
    }

    /// Two runtime surfaces describe the Ornith Q8_0 row — the load-time plan
    /// (`/v1/health`, the System page) and `/api/capabilities` — and they used to
    /// contradict each other: the plan emitted
    /// `support_level=unknown_or_unvalidated` because `recognized_row_level` had
    /// no ornith arm, while the capabilities row had read
    /// `supported_exact_row_smoke` since the runnable-lane certification.
    ///
    /// The capabilities status is read from the live table rather than spelled
    /// out here, so the two surfaces cannot drift apart again silently: a change
    /// to that row's status fails this test instead of quietly re-opening the
    /// contradiction.
    #[test]
    fn ornith_q8_plan_support_level_agrees_with_the_capabilities_row() {
        let _guard = env_lock();
        clear_profile_env();

        let capabilities_status = crate::api::capabilities_response()
            .model_compatibility
            .iter()
            .find(|target| target.id == "Ornith 1.0 9B")
            .expect("the Ornith Q8_0 row must be advertised in model_compatibility")
            .status;
        assert_eq!(
            capabilities_status, "supported_exact_row_smoke",
            "the plan's ornith arm carries this exact string; update both together"
        );

        // Shaped like ornith-1.0-9b-Q8_0: Q8_0 projections + F32 norms, the bare
        // `general.name` the loader reports, and the qwen35 arch string.
        let mut gguf = quant_fixture(
            "Ornith 1.0 9B",
            None,
            &[GgufTensorType::Q8_0, GgufTensorType::F32],
        );
        gguf.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("qwen35".into()),
        );

        // Platform-blind, like the table itself: the row is certified on the
        // runnable lane, which exists on every host.
        for platform in [
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
            platform("windows", "x86_64", &["avx2"]),
            platform("linux", "x86_64", &["avx2"]),
        ] {
            let label = platform.platform_label.clone();
            let outcome = plan_for_model_with_platform(
                &PathBuf::from("/models/ornith-1.0-9b-Q8_0.gguf"),
                &gguf,
                Some(8),
                platform,
            );
            assert_eq!(outcome.plan.quant_type, "Q8_0", "{label}");
            assert_eq!(outcome.plan.support_level, capabilities_status, "{label}");
            assert!(
                outcome
                    .plan
                    .reasons
                    .contains(&format!("support_level={capabilities_status}")),
                "{label}: reasons must carry the same level the capabilities row states: {:?}",
                outcome.plan.reasons
            );
        }
        clear_profile_env();
    }

    /// The row being named in `recognized_row_level` must NOT hand it to the
    /// optimized dense Q8 engine. qwen35 is runnable-only on every host, so
    /// `select_macos_q8_plan` / `select_x86_q8_plan` would describe kernels that
    /// cannot express its gated-delta-net layers — and would write dense-Q8
    /// tuning env into a load that never runs them.
    #[test]
    fn runnable_only_archs_never_claim_the_optimized_q8_row_plan() {
        let ornith = {
            let mut gguf = quant_fixture(
                "Ornith 1.0 9B",
                None,
                &[GgufTensorType::Q8_0, GgufTensorType::F32],
            );
            gguf.metadata.insert(
                "general.architecture".into(),
                GgufMetadataValue::String("qwen35".into()),
            );
            gguf
        };
        assert!(
            !is_supported_exact_q8_row("Ornith 1.0 9B", &ornith),
            "the ornith row is certified on the runnable lane, not this engine"
        );

        // The guard is keyed on the ARCH, not on the row name being absent from
        // the level list — so it still holds if a runnable-only arch ever ships
        // under a row name the optimized engine does recognize, and it keeps
        // holding if `supported_exact_row_smoke` is later added to that list.
        for arch in ["qwen35", "gemma2", "lfm2", "bitnet-b1.58"] {
            assert!(
                crate::model::is_runnable_only_arch(arch),
                "{arch} must stay in the runnable-only set for this guard to mean anything"
            );
            let mut gguf = fixture("Llama 3.2 3B Instruct");
            gguf.metadata.insert(
                "general.architecture".into(),
                GgufMetadataValue::String(arch.into()),
            );
            assert!(
                !is_supported_exact_q8_row("Llama 3.2 3B Instruct", &gguf),
                "{arch} is runnable-only and must be refused before the row table is read"
            );
        }
    }

    #[test]
    fn bitnet_i2_s_plans_disclose_the_runtime_that_serves_them() {
        let _guard = env_lock();
        env::remove_var("CAMELID_BITNET_GPU");
        env::remove_var("CAMELID_BITNET_KERNEL");
        let mut causal = quant_fixture(
            "bitnet2b",
            Some(40),
            &[GgufTensorType::I2S, GgufTensorType::F16],
        );
        causal.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("bitnet-b1.58".into()),
        );
        let causal_plan = plan_for_model_with_platform(
            &PathBuf::from("/models/ggml-model-i2_s.gguf"),
            &causal,
            Some(8),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(causal_plan.plan.quant_type, "I2_S");
        assert_eq!(
            causal_plan.plan.selected_backend,
            "bitnet_runnable_cpu_runtime"
        );
        let metal_causal_plan = plan_for_model_with_platform(
            &PathBuf::from("/models/ggml-model-i2_s.gguf"),
            &causal,
            Some(8),
            metal_platform("macos", "aarch64", &["neon"]),
        );
        assert_eq!(
            metal_causal_plan.plan.selected_backend,
            "bitnet_runnable_metal_runtime"
        );
        env::set_var("CAMELID_BITNET_GPU", "0");
        env::set_var("CAMELID_BITNET_KERNEL", "tl1");
        let forced_cpu_plan = plan_for_model_with_platform(
            &PathBuf::from("/models/ggml-model-i2_s.gguf"),
            &causal,
            Some(8),
            metal_platform("macos", "aarch64", &["neon"]),
        );
        assert_eq!(
            forced_cpu_plan.plan.selected_backend,
            "bitnet_runnable_cpu_runtime"
        );
        assert_eq!(
            forced_cpu_plan.plan.selected_q8_path,
            "i2_s_canonical_tl1_lookup"
        );
        env::remove_var("CAMELID_BITNET_GPU");
        env::remove_var("CAMELID_BITNET_KERNEL");

        let mut embedding = quant_fixture(
            "bitnet-embeddings-270m",
            Some(40),
            &[GgufTensorType::I2S, GgufTensorType::F16],
        );
        embedding.metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("gemma3".into()),
        );
        embedding.metadata.insert(
            "general.name".into(),
            GgufMetadataValue::String("bitnet-embeddings-270m".into()),
        );
        let embedding_plan = plan_for_model_with_platform(
            &PathBuf::from("/models/bitnet-embeddings-270m-bf16-i2_s.gguf"),
            &embedding,
            Some(8),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(
            embedding_plan.plan.selected_backend,
            "bitnet_embedding_cpu_runtime"
        );
        assert_eq!(embedding_plan.plan.decode_path, "mean_pool_l2_normalize");
        let cuda_embedding_plan = plan_for_model_with_platform(
            &PathBuf::from("C:/models/bitnet-embeddings-270m-bf16-i2_s.gguf"),
            &embedding,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(
            cuda_embedding_plan.plan.selected_backend,
            "bitnet_embedding_cuda_runtime"
        );
        assert_eq!(
            cuda_embedding_plan.plan.prefill_path,
            "bitnet_embedding_cuda_full_sequence"
        );
    }

    #[test]
    fn mac_metal_resident_plan_selected_when_device_and_gate_present() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_backend, "metal_resident_q8_runtime");
        assert_eq!(outcome.plan.decode_path, "q8_0_metal_resident_decode");
        // The rows4 repack must stay OFF: the GPU path needs plain Q8_0 blocks.
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("off"))
        );
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        clear_profile_env();
    }

    #[test]
    fn windows_cuda_resident_plan_selected_when_engine_active() {
        let _guard = env_lock();
        clear_profile_env();
        // A supported Qwen3 Q8_0 row on a Windows x86_64 host where the CUDA resident
        // decode engine is active: the plan surfaces the GPU-resident backend/decode
        // labels and reports cuda_resident_active, while keeping the row's
        // supported_exact_row_smoke_chatml support level (engine-agnostic).
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen3-0.6B-Q8_0.gguf"),
            &fixture("Qwen3 0.6B Instruct"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cuda_resident_q8_runtime");
        assert_eq!(outcome.plan.decode_path, "q8_0_cuda_resident_decode");
        assert_eq!(outcome.plan.prefill_path, "q8_0_cuda_resident_prefill");
        assert!(outcome.plan.cuda_resident_active);
        assert_eq!(
            outcome.plan.support_level, "supported_exact_row_smoke_chatml",
            "GPU lane reuses the row-keyed support level (Phase 1 design)"
        );
        // The GPU consumes plain Q8_0 blocks: the x86 rows4 repack must NOT be enabled.
        assert_ne!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn windows_cuda_resident_inactive_keeps_cpu_repack_plan() {
        let _guard = env_lock();
        clear_profile_env();
        // Same row/host but no active CUDA engine: the validated x86_64 CPU repack plan.
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen3-0.6B-Q8_0.gguf"),
            &fixture("Qwen3 0.6B Instruct"),
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert!(!outcome.plan.cuda_resident_active);
        clear_profile_env();
    }

    #[test]
    fn mac_metal_plan_requires_device_and_gate() {
        let _guard = env_lock();
        clear_profile_env();
        // Gate present but no Metal device: validated CPU repack plan.
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        // Device present but gate absent (embedder/test default): CPU repack plan.
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        // Explicit opt-out returns the CPU repack plan even with device + gate.
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        env::set_var("CAMELID_MAC_Q8_METAL_PLAN", "0");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        env::remove_var("CAMELID_MAC_Q8_METAL_PLAN");
        clear_profile_env();
    }

    #[test]
    fn mac_auto_selects_validated_mac_path() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Auto);
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        assert_eq!(
            outcome.plan.prefill_path,
            "q8_0_direct_pack_prefill_i8mm_available"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_prefill_rows_gte_4"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_PARALLEL_LINEAR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_PREFILL_I8MM"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_SCHED"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert!(!outcome.env_updates.contains_key("CAMELID_X86_Q8_KERNEL"));
        clear_profile_env();
    }

    #[test]
    fn mac_experimental_allows_packed_prefill_scheduler() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_MAC_Q8_PREFILL_I8MM", "on");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Experimental);
        assert_eq!(
            outcome.plan.prefill_path,
            "q8_0_experimental_packed_prefill_i8mm_available"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_prefill_rows_gte_4"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_SCHED"),
            Some(&Some("packed_prefill"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn mac_ffn_decode_consumer_plan_gates_are_default_on_and_opt_out() {
        let _guard = env_lock();
        clear_profile_env();
        let default_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            default_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            default_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );

        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", "off");
        env::set_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", "off");
        let opt_out_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            opt_out_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            opt_out_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn mac_auto_explicit_matches_auto_default_plan() {
        let _guard = env_lock();
        clear_profile_env();
        let default_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "auto");
        let explicit_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(default_outcome.plan.profile, explicit_outcome.plan.profile);
        assert_eq!(
            default_outcome.plan.selected_backend,
            explicit_outcome.plan.selected_backend
        );
        assert_eq!(
            default_outcome.plan.selected_q8_path,
            explicit_outcome.plan.selected_q8_path
        );
        assert_eq!(
            default_outcome.plan.prefill_path,
            explicit_outcome.plan.prefill_path
        );
        assert_eq!(
            default_outcome.plan.prefill_runtime_policy,
            explicit_outcome.plan.prefill_runtime_policy
        );
        assert_eq!(
            default_outcome.plan.decode_path,
            explicit_outcome.plan.decode_path
        );
        assert_eq!(
            default_outcome.plan.fallback_path,
            explicit_outcome.plan.fallback_path
        );
        assert_eq!(default_outcome.env_updates, explicit_outcome.env_updates);
        clear_profile_env();
    }

    #[test]
    fn ubuntu_auto_enables_x86_optimizations_by_default() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform(
                "linux",
                "x86_64",
                &["avx2", "avx512f", "avx512_vnni", "amx_int8"],
            ),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Auto);
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert_eq!(
            outcome.plan.selected_q8_path,
            "x86_experimental_q8_0_avx2_rust"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("avx2"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_validated_gates_select_rust_avx2_q8_path() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Experimental);
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert_eq!(
            outcome.plan.selected_q8_path,
            "x86_experimental_q8_0_avx2_rust"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_q8_runtime_storage_active"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_PARALLEL_LINEAR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("avx2"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "off");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_ffn_decode_chain_enables_required_gate_up_and_down_legs() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert!(outcome.plan.reasons.iter().any(|reason| reason.contains(
            "FFN decode-chain opt-in also enables the required FFN gate/up decode consumer gate"
        )));
        assert!(outcome.plan.reasons.iter().any(|reason| reason.contains(
            "FFN decode-chain opt-in also enables the required FFN-down decode consumer gate"
        )));
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_clears_stale_x86_q8_decode_consumer_flags() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "7");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "5");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", "on");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "3",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "on");

        PlannerEnv::capture().apply(&BTreeMap::new());

        assert!(env::var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DECODE_CHAIN").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE").is_err());
        assert!(env::var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE").is_err());
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_restores_owned_x86_q8_passthrough_knobs() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "7");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "5");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "3",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        let planner_env = PlannerEnv::capture();

        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "99");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "99");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "99",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "99");

        let updates = BTreeMap::from([
            (
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
                Some("on"),
            ),
            (
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
                Some("on"),
            ),
            ("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", Some("on")),
            ("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", Some("on")),
        ]);
        planner_env.apply(&updates);

        assert_eq!(
            env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK").ok(),
            Some("7".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK").ok(),
            Some("5".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS").ok(),
            Some("3".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").ok(),
            Some("9".into())
        );
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_does_not_restore_packed_rows4_matmul_chunk_groups_without_owner_gate() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        let planner_env = PlannerEnv::capture();

        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "99");

        planner_env.apply(&BTreeMap::new());

        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").is_err());
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_disabled_repack_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "off");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_without_avx2_feature_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["sse4_2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        clear_profile_env();
    }

    #[test]
    fn debug_profile_enables_diagnostics_without_changing_claims() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "debug");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(4),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert!(outcome
            .plan
            .diagnostics_status
            .contains("debug diagnostics"));
        assert_eq!(
            outcome.env_updates.get("CAMELID_FORWARD_RSS_TIMINGS"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn explicit_disable_override_falls_back_to_safe() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_REPACK", "off");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            None,
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn invalid_x86_kernel_override_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_KERNEL", "amx_now_please");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            None,
            platform("linux", "x86_64", &["avx2", "amx_int8"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn unsupported_row_stays_safe() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen2.5-7B-Instruct-Q8_0.gguf"),
            &fixture("Qwen2.5-7B-Instruct-Q8_0.gguf"),
            None,
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.support_level, "unknown_or_unvalidated");
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        clear_profile_env();
    }

    #[test]
    fn mistral_row_selects_validated_q8_plan() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            &fixture("Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            None,
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            outcome.plan.support_level,
            "supported_exact_row_smoke_512_1024_2048_4096_8192"
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        clear_profile_env();
    }

    #[test]
    fn qwen3_rows_select_validated_x86_q8_plan() {
        let _guard = env_lock();
        for name in [
            "Qwen3-0.6B-Instruct-Q8_0.gguf",
            "Qwen3-1.7B-Instruct-Q8_0.gguf",
            "Qwen3-4B-Instruct-Q8_0.gguf",
            "Qwen3-8B-Instruct-Q8_0.gguf",
        ] {
            clear_profile_env();
            let outcome = plan_for_model_with_platform(
                &PathBuf::from(format!("/tmp/{name}")),
                &fixture(name),
                None,
                platform("windows", "x86_64", &["avx2"]),
            );
            assert_eq!(
                outcome.plan.support_level, "supported_exact_row_smoke_chatml",
                "row {name} support_level"
            );
            // Supported Qwen3 Q8 rows engage the validated x86_64 runtime-repack/AVX2
            // plan (not the scalar safe path), matching the other supported Q8 rows.
            assert_eq!(
                outcome.plan.selected_backend, "cpu_q8_runtime_repack",
                "row {name} backend"
            );
            assert_eq!(
                outcome.plan.selected_q8_path, "x86_experimental_q8_0_avx2_rust",
                "row {name} q8 path"
            );
            clear_profile_env();
        }
    }
}
