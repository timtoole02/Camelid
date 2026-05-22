use std::{collections::BTreeMap, env, path::Path};

use serde::Serialize;

use crate::gguf::{GgufFile, GgufTensorType};

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
];

pub const MAC_Q8_PREFILL_I8MM_MIN_ROWS: usize = 4;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
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
    pub quant_type: String,
    pub exact_model_row: String,
    pub support_level: String,
    pub selected_backend: String,
    pub selected_q8_path: String,
    pub prefill_path: String,
    pub prefill_runtime_policy: String,
    pub decode_path: String,
    pub thread_count: usize,
    pub diagnostics_status: String,
    pub fallback_path: String,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlanOutcome {
    pub plan: ExecutionPlan,
    pub env_updates: BTreeMap<&'static str, Option<&'static str>>,
}

#[derive(Clone, Debug, Default)]
pub struct PlannerEnv;

impl PlannerEnv {
    pub fn capture() -> Self {
        Self
    }

    pub fn apply(&self, updates: &BTreeMap<&'static str, Option<&'static str>>) {
        for key in MANAGED_ENV_KEYS {
            match updates.get(key).copied().flatten() {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanPlatform {
    pub operating_system: String,
    pub architecture: String,
    pub platform_label: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
}

impl PlanPlatform {
    pub fn current() -> Self {
        let operating_system = env::consts::OS.to_string();
        let architecture = env::consts::ARCH.to_string();
        let cpu_features = cpu_features();
        let cpu_model = cpu_model();
        let platform_label = platform_label(&operating_system, &architecture, &cpu_model);
        Self {
            operating_system,
            architecture,
            platform_label,
            cpu_model,
            cpu_features,
        }
    }
}

pub fn plan_for_model(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
) -> ExecutionPlanOutcome {
    plan_for_model_with_platform(model_path, gguf, threads, PlanPlatform::current())
}

pub fn plan_for_model_with_platform(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
    platform: PlanPlatform,
) -> ExecutionPlanOutcome {
    let (profile, profile_reason) = requested_profile();
    let row = exact_model_row(model_path, gguf);
    let support_level = support_level(&row);
    let model_family = model_family(&row, gguf);
    let quant_type = quant_type(gguf);
    let thread_count = threads.unwrap_or_else(default_thread_count);
    let diagnostics_status = match profile {
        ExecutionProfile::Debug => {
            "debug diagnostics enabled; performance claims disabled".to_string()
        }
        _ => "standard diagnostics; RSS timings disabled by default".to_string(),
    };

    let mut reasons = vec![profile_reason];
    reasons.push(format!("exact_model_row={row}"));
    reasons.push(format!("support_level={support_level}"));
    reasons.push(format!("quant_type={quant_type}"));

    let mut env_updates: BTreeMap<&'static str, Option<&'static str>> = BTreeMap::new();
    if matches!(profile, ExecutionProfile::Debug) {
        env_updates.insert("CAMELID_FORWARD_RSS_TIMINGS", Some("on"));
    }

    let (
        selected_backend,
        selected_q8_path,
        prefill_path,
        prefill_runtime_policy,
        decode_path,
        fallback_path,
    ) = if quant_type == "Q8_0" && is_supported_exact_q8_row(&row) {
        if platform.operating_system == "macos" && platform.architecture == "aarch64" {
            select_macos_q8_plan(&profile, &platform, &mut env_updates, &mut reasons)
        } else if platform.operating_system == "linux" && platform.architecture == "x86_64" {
            select_linux_x86_q8_plan(&profile, &platform, &mut env_updates, &mut reasons)
        } else {
            reasons.push(
                    "no validated platform-specific Q8_0 plan for this OS/arch; failing closed to safe path"
                        .into(),
                );
            safe_q8_plan()
        }
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
        reasons,
    };
    ExecutionPlanOutcome { plan, env_updates }
}

fn select_macos_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
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
        reasons.push("safe profile selected; optimized Mac Q8 paths disabled".into());
        return safe_q8_plan();
    }
    if env_flag_disabled("CAMELID_MAC_Q8_REPACK") {
        reasons
            .push("CAMELID_MAC_Q8_REPACK disables Mac repack; failing closed to safe path".into());
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
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

    if env_flag_enabled("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN-down decode consumer gate enabled by explicit opt-in".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN-down decode consumer remains default-off".into());
    }

    if env_flag_enabled("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN gate/up decode consumer gate enabled by explicit opt-in".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN gate/up decode consumer remains default-off".into());
    }

    let prefill_i8mm_requested = env_flag_enabled("CAMELID_MAC_Q8_PREFILL_I8MM");
    let prefill_path = if i8mm && prefill_i8mm_requested {
        env_updates.insert("CAMELID_MAC_Q8_PREFILL_I8MM", Some("on"));
        reasons.push("explicit direct-pack prefill I8MM gate enabled".into());
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
        if env_flag_disabled("CAMELID_MAC_Q8_PREFILL_I8MM") {
            reasons.push("CAMELID_MAC_Q8_PREFILL_I8MM disables I8MM prefill".into());
        } else if !prefill_i8mm_requested {
            reasons.push(
                "CAMELID_MAC_Q8_PREFILL_I8MM remains default-off pending longer decode parity evidence"
                    .into(),
            );
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

fn select_linux_x86_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
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
    if !matches!(profile, ExecutionProfile::Experimental) {
        reasons.push(
            "Ubuntu/Linux x86_64 optimized Q8 path requires profile=experimental; failing closed to safe path"
                .into(),
        );
        return safe_q8_plan();
    }
    if env_flag_disabled("CAMELID_X86_Q8_REPACK") || env_flag_disabled("CAMELID_X86_Q8_KERNEL") {
        reasons.push(
            "x86 Q8 override disables optimized kernel/repack; failing closed to safe path".into(),
        );
        if env_flag_disabled("CAMELID_X86_Q8_REPACK") {
            env_updates.insert("CAMELID_X86_Q8_REPACK", Some("off"));
        }
        if env_flag_disabled("CAMELID_X86_Q8_KERNEL") {
            env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        }
        return safe_q8_plan();
    }
    if let Some(invalid) = invalid_x86_kernel_override() {
        reasons.push(format!(
            "invalid CAMELID_X86_Q8_KERNEL={invalid}; failing closed to safe path"
        ));
        env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        return safe_q8_plan();
    }
    if !env_flag_enabled("CAMELID_X86_Q8_REPACK") || !x86_kernel_avx2_explicitly_requested() {
        reasons.push(
            "Ubuntu/Linux x86_64 optimized Q8 path requires explicit CAMELID_X86_Q8_REPACK=on and CAMELID_X86_Q8_KERNEL=avx2 gates; failing closed to safe path"
                .into(),
        );
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
        if env_flag_enabled(name) {
            Some("on")
        } else {
            Some("off")
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
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
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
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
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
    reasons.push("validated Ubuntu/Linux x86_64 Rust Q8 runtime repack enabled".into());
    reasons.push("validated Rust AVX2 Q8 packed rows4 kernel selected".into());
    reasons.push(
        "attention, FFN, and output decode-consumer experiments remain default-off unless explicitly opted in"
            .into(),
    );
    reasons.push("experimental profile active; support claims remain unchanged".into());

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
    gguf.model_name()
        .map(|value| value.to_string())
        .or_else(|| {
            model_path
                .file_name()
                .map(|v| v.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

fn support_level(row: &str) -> String {
    let normalized = normalize_row(row);
    if normalized.contains("tinyllama") {
        "supported_current_gate".into()
    } else if normalized.contains("llama_3_2_1b_instruct") {
        "supported_exact_row_smoke_512_1024_2048_4096_8192".into()
    } else if normalized.contains("llama_3_2_3b_instruct")
        || normalized.contains("llama_3_8b_instruct")
        || normalized.contains("meta_llama_3_8b_instruct")
    {
        "supported_exact_row_smoke_512_1024_2048".into()
    } else if normalized.contains("mistral_7b_instruct_v0_3") {
        "active_validation_unsupported".into()
    } else if normalized.contains("mixtral_8x7b_instruct_v0_1") {
        "bounded_runtime_only_unsupported".into()
    } else {
        "unknown_or_unvalidated".into()
    }
}

fn is_supported_exact_q8_row(row: &str) -> bool {
    matches!(
        support_level(row).as_str(),
        "supported_current_gate"
            | "supported_exact_row_smoke_512_1024_2048_4096_8192"
            | "supported_exact_row_smoke_512_1024_2048"
    )
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

fn quant_type(gguf: &GgufFile) -> String {
    if gguf
        .tensors
        .iter()
        .any(|tensor| tensor.tensor_type == GgufTensorType::Q8_0)
    {
        "Q8_0".into()
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

fn env_flag_disabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("cpu")
        })
        .unwrap_or(false)
}

fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("enabled")
        })
        .unwrap_or(false)
}

fn x86_kernel_avx2_explicitly_requested() -> bool {
    env::var("CAMELID_X86_Q8_KERNEL")
        .map(|value| value.trim().eq_ignore_ascii_case("avx2"))
        .unwrap_or(false)
}

fn invalid_x86_kernel_override() -> Option<String> {
    let value = env::var("CAMELID_X86_Q8_KERNEL").ok()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor},
        test_support::env_lock,
    };
    use std::{collections::BTreeMap, path::PathBuf};

    fn platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            operating_system: os.into(),
            architecture: arch.into(),
            platform_label: platform_label(os, arch, "Apple M4"),
            cpu_model: "fixture cpu".into(),
            cpu_features: features.iter().map(|feature| (*feature).into()).collect(),
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

    fn clear_profile_env() {
        for key in [
            "CAMELID_PROFILE",
            "CAMELID_MAC_Q8_REPACK",
            "CAMELID_MAC_Q8_PREFILL_I8MM",
            "CAMELID_MAC_Q8_SCHED",
            "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
            "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            "CAMELID_X86_Q8_REPACK",
            "CAMELID_X86_Q8_KERNEL",
            "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
            "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
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
        ] {
            env::remove_var(key);
        }
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
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
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
            "q8_0_cpu_packed_prefill_fallback_available"
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
            Some(&Some("off"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_SCHED"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
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
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn mac_ffn_decode_consumer_plan_gates_are_default_off_and_opt_in() {
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
            Some(&Some("off"))
        );
        assert_eq!(
            default_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );

        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
        let opt_in_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            opt_in_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            opt_in_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
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
    fn ubuntu_auto_keeps_x86_experiments_default_off() {
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
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        assert!(!outcome.env_updates.contains_key("CAMELID_X86_Q8_KERNEL"));
        assert!(!outcome.env_updates.contains_key("CAMELID_X86_Q8_REPACK"));
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
        assert!(outcome
            .plan
            .reasons
            .iter()
            .any(|reason| reason.contains("requires profile=experimental")));
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
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
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
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER"),
            Some(&Some("off"))
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
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
            Some(&Some("off"))
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
            Some(&Some("off"))
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
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
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
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING", "on");
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
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");

        PlannerEnv::capture().apply(&BTreeMap::new());

        assert!(env::var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING").is_err());
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
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER").is_err());
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_missing_x86_gate_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        assert!(!outcome.env_updates.contains_key("CAMELID_X86_Q8_REPACK"));
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
            &PathBuf::from("/tmp/Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            &fixture("Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            None,
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.support_level, "active_validation_unsupported");
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        clear_profile_env();
    }
}
