//! Isolated Gemma 4 26B-A4B official MTP-assistant experiment harness.
//!
//! This is deliberately separate from CLAIRE's production drafter. The live
//! experiment is both ignored and default-off, and requires:
//!
//! ```text
//! CAMELID_GEMMA4_MTP_EXPERIMENT=1
//! CAMELID_GEMMA4_MTP_ASSISTANT_PATH=/exact/path/to/model.safetensors
//! CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON=/internal/path/to/admission.json
//! CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE=<same-fresh-nonce-used-by-admission-test>
//! CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_TEST_EXE=/internal/path/to/exact/lib-test-binary
//! CAMELID_GEMMA4_MTP_PILOT_ONLY=1
//! CAMELID_GEMMA4_MTP_REPORT_PATH=/internal/path/to/atomic-report.json
//! ```
//!
//! The parent performs the exact official pair gate, then launches one fresh
//! structured-IPC child for every N/I/M lane. The native assistant is admitted
//! only from an internal-volume, fully resident mapping. The target verifier
//! remains authoritative: unverified assistant tokens cannot be committed.
//!
//! The independent load-only probe has no admission or generation surface:
//!
//! ```text
//! CAMELID_GEMMA4_MTP_LOAD_ONLY_PROBE=1
//! CAMELID_GEMMA4_MTP_LOAD_ONLY_REPORT_PATH=/internal/path/to/checkpoint.json
//! CAMELID_GEMMA4_MTP_ASSISTANT_PATH=/exact/path/to/model.safetensors
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(target_os = "macos")]
use std::ffi::{CStr, CString};

const EXPERIMENT_ENABLE_ENV: &str = "CAMELID_GEMMA4_MTP_EXPERIMENT";
const ASSISTANT_PATH_ENV: &str = "CAMELID_GEMMA4_MTP_ASSISTANT_PATH";
const TARGET_RUNTIME_PATH_ENV: &str = "CAMELID_GEMMA4_26B_GGUF";
const TARGET_CGHOST_PATH_ENV: &str = "CAMELID_GEMMA4_26B_CGHOST";
const TARGET_SOURCE_PATH_ENV: &str = "CAMELID_GEMMA4_MTP_SOURCE_GGUF";
const TARGET_CACHE_MIB_ENV: &str = "SPEC50_CACHE_MIB";
const CHILD_REQUEST_ENV: &str = "CAMELID_GEMMA4_MTP_CHILD_REQUEST";
const CHILD_RESULT_ENV: &str = "CAMELID_GEMMA4_MTP_CHILD_RESULT";
const REPORT_PATH_ENV: &str = "CAMELID_GEMMA4_MTP_REPORT_PATH";
const PILOT_ONLY_ENV: &str = "CAMELID_GEMMA4_MTP_PILOT_ONLY";
const MATRIX_TOKENS_ENV: &str = "CAMELID_GEMMA4_MTP_MATRIX_TOKENS";
const MATRIX_REPETITIONS_ENV: &str = "CAMELID_GEMMA4_MTP_MATRIX_REPETITIONS";
const CHILD_TIMEOUT_SECS_ENV: &str = "CAMELID_GEMMA4_MTP_CHILD_TIMEOUT_SECS";
const NATIVE_ADMISSION_EVIDENCE_ENV: &str = "CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON";
const NATIVE_ADMISSION_RUN_NONCE_ENV: &str = "CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE";
const NATIVE_ADMISSION_TEST_EXE_ENV: &str = "CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_TEST_EXE";
const LOAD_ONLY_PROBE_ENABLE_ENV: &str = "CAMELID_GEMMA4_MTP_LOAD_ONLY_PROBE";
const LOAD_ONLY_PROBE_REPORT_PATH_ENV: &str = "CAMELID_GEMMA4_MTP_LOAD_ONLY_REPORT_PATH";
const DEFAULT_TARGET_RUNTIME_PATH: &str =
    "/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.hot.gguf";
const DEFAULT_TARGET_CGHOST_PATH: &str =
    "/Users/timtoole/models/gemma4-mtp-pair/gemma-4-26B_q4_0-it.v3.cghost";
const DEFAULT_TARGET_SOURCE_PATH: &str = "/Volumes/Untitled/models/gemma-4-26B_q4_0-it.gguf";
const DEFAULT_TARGET_CACHE_MIB: usize = 0;
const DEFAULT_SEEDED_NGRAM_TEXT: &str = "<|channel>thought\n<channel|>";
const TARGET_WARMUP_TOKENS: u64 = 24;
const PILOT_REPETITIONS: u32 = 3;
const DEFAULT_MATRIX_TOKENS: u64 = 64;
const DEFAULT_MATRIX_REPETITIONS: u32 = 3;
const DEFAULT_CHILD_TIMEOUT_SECS: u64 = 30 * 60;
const REPORT_SCHEMA_VERSION: u32 = 4;
const NATIVE_ADMISSION_SCHEMA_VERSION: u32 = 1;
const NATIVE_RECURRENCE_TEST_NAME: &str =
    "camelid::metal::gemma4_mtp::tests::official_target_free_bf16_oracle_matches_native_seven_proposal_recurrence";
const NATIVE_RECURRENCE_ORACLE_SHA256: &str =
    "08cf02bedfec09074eeebaa24b1ffaaa5362badd412b523de6a0d52952e94109";
const NATIVE_RECURRENCE_GENERATION_RECEIPT_SHA256: &str =
    "573864f252b29b63932a440d9224733e086ae4de42b10caf457da975c623053d";
const NATIVE_STAGE_ORACLE_SHA256: &str =
    "f18ed0a4ae9538ed5b41e74c39f9242e9b7a842a7c2d3cc5d7abf9765c4b983e";
// These run-log identities are enforced by the admitted native test. Requiring
// their literal pins in the compile-time-bound native source makes that
// transitive provenance check explicit even though the compact PASS receipt
// serializes only the generation-receipt digest.
const NATIVE_RECURRENCE_RUN1_LOG_SHA256: &str =
    "654d0c1fa5221056efb9bc4cc02a7f9c5c1c79044124fd23e1d94b076a05a7a5";
const NATIVE_RECURRENCE_RUN2_LOG_SHA256: &str =
    "afdebdb0642f130583f3b56d9c1050451541c3da3c545415d62a01bd026184d3";
const NATIVE_RECURRENCE_FAILED_ATTEMPTS_LOG_SHA256: &str =
    "a56f1a59faaa14d1760c1f8ea8b06f63a67f9b87de01802707b2c5f5c05fc0b3";
const NATIVE_RECURRENCE_TOP1_TOKENS: [u32; 7] =
    [53_965, 150_062, 103_463, 48_277, 3_947, 237_729, 236_764];
const NATIVE_REQUIRED_MARGIN_BF16_ULP: [u32; 7] = [2, 2, 2, 0, 2, 1, 2];
const NATIVE_RECURRENCE_REPETITIONS: u32 = 2;
const NATIVE_MIN_TOP16_OVERLAP: u32 = 15;
const NATIVE_MIN_RECURRENT_COSINE: f64 = 0.999_95;
const NATIVE_MAX_RECURRENT_RELATIVE_L2: f64 = 0.01;
const NATIVE_ARGMAX_TIE_POLICY: &str = "lowest_token_id";
const NATIVE_MARGIN_CAP_BF16_ULP: u32 = 2;
const NATIVE_MARGIN_FLOOR_RULE: &str =
    "min(reference_top1_margin_bf16_ulp, native_margin_cap_bf16_ulp)";
const MAX_NATIVE_ADMISSION_AGE: Duration = Duration::from_secs(60 * 60);
const MAX_NATIVE_ADMISSION_CLOCK_SKEW: Duration = Duration::from_secs(5);
const MAX_NATIVE_ADMISSION_CREATE_TO_MTIME: Duration = Duration::from_secs(60);
const GEMMA4_MTP_SOURCE: &str = include_str!("../src/metal/gemma4_mtp.rs");
const METAL_SOURCE: &str = include_str!("../src/metal.rs");
const GEMMA4_RUNTIME_SOURCE: &str = include_str!("../src/gemma4_runtime.rs");
const CARGO_LOCK_SOURCE: &str = include_str!("../Cargo.lock");
const WARMUP_PROMPT: &str =
    "<|turn>user\nSay hello and name three colours.\n<turn|>\n<|turn>model\n";

// These are duplicated deliberately from gemma4_mtp_pair_gate.rs so each
// serialized experiment report is independently auditable. The native adapter
// must populate PairingEvidence from a successful gate receipt; paths or sizes
// alone never establish model identity.
const OFFICIAL_TARGET_REPOSITORY: &str = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf";
const OFFICIAL_TARGET_REVISION: &str = "dfc00409adc70be497fee9c90bfe76b3ee130f2e";
const OFFICIAL_TARGET_BYTES: u64 = 14_439_361_440;
const OFFICIAL_TARGET_SHA256: &str =
    "4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5";
const OFFICIAL_ASSISTANT_REPOSITORY: &str =
    "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant";
const OFFICIAL_ASSISTANT_REVISION: &str = "9537141506fe8875b3ed45b264af13580cb29166";
const OFFICIAL_ASSISTANT_MODEL_BYTES: u64 = 839_427_840;
const OFFICIAL_ASSISTANT_MODEL_SHA256: &str =
    "c082cc581c3ec90d70285c1a41c81544ff56cbc96650f16c900a280940655801";
const OFFICIAL_ASSISTANT_CONFIG_SHA256: &str =
    "23d2bc4a8920f24c23653ff6871437bbd95e52527bf50007aaad05b0b6cab510";
const OFFICIAL_ASSISTANT_TOKENIZER_CONFIG_SHA256: &str =
    "01f2ff1c21ef2e722891380323edcaecd9c86a776aeb9b40148e2f35e3cee4d3";
const OFFICIAL_ASSISTANT_TOKENIZER_SHA256: &str =
    "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c";
const SHARED_VOCAB_SIZE: u64 = 262_144;
const TARGET_SHARED_KV_LAYERS: u32 = 0;
const ASSISTANT_SHARED_KV_LAYERS: u32 = 4;
const TARGET_SHARED_KV_SLIDING_SOURCE_LAYER: u32 = 28;
const TARGET_SHARED_KV_FULL_SOURCE_LAYER: u32 = 29;
const STAGED_TARGET_IDENTITY_SCHEME: &str = "gemma4-moe-sampled-sha256-v1";
const PAIR_SPECIAL_SENTINELS: &[(u32, &str)] = &[
    (0, "<pad>"),
    (1, "<eos>"),
    (2, "<bos>"),
    (3, "<unk>"),
    (4, "<mask>"),
    (46, "<|tool>"),
    (47, "<tool|>"),
    (48, "<|tool_call>"),
    (49, "<tool_call|>"),
    (50, "<|tool_response>"),
    (51, "<tool_response|>"),
    (52, "<|\"|>"),
    (98, "<|think|>"),
    (100, "<|channel>"),
    (101, "<channel|>"),
    (105, "<|turn>"),
    (106, "<turn|>"),
    (255_999, "<|image>"),
    (256_000, "<|audio>"),
    (258_880, "<|image|>"),
    (258_881, "<|audio|>"),
    (258_882, "<image|>"),
    (258_883, "<audio|>"),
];
const PAIR_TEXT_PROBES: &[&str] = &[
    "Hello, Camelid!",
    " leading space",
    "café 日本語\nline 2",
    "<|turn>assistant<|channel>analysis<channel|>",
];

const MONITOR_PERIOD: Duration = Duration::from_secs(1);
const SWAP_ACTIVITY_STREAK_LIMIT: u32 = 3;
const ROLLING_SWAP_WINDOW_MS: u64 = 10_000;
const ROLLING_SWAP_TRAFFIC_LIMIT_BYTES: u64 = 64 * 1024 * 1024;
const SWAP_USED_GROWTH_LIMIT_BYTES: u64 = 256 * 1024 * 1024;
const PRESSURE_STREAK_LIMIT: u32 = 2;
const ECONOMICS_MIN_VERIFY_ROUNDS: u64 = 8;
const ECONOMICS_MIN_EMITTED_TOKENS: u64 = 16;
const ECONOMICS_LOSS_STREAK_LIMIT: u32 = 2;
const NO_GAIN_WALL_RATIO_LIMIT: f64 = 1.05;
// MEASURED 2026-08-21 from a full [route-metal] routing trace (30 layers x 39 steps):
// the decode working set SATURATES at 45 distinct experts/layer by step 15 and never grows
// (first-touch records/token: 240,143,64,110,90,62,70,58,52,28,... then exactly 0 from step 30).
// A 56-slot table therefore holds the entire decode working set, and steady-state expert disk
// traffic is ZERO. The old 32-token budget measured only the cold-start window, so the whole
// 93.8 MB/token figure was warmup amortized over too few tokens. Generate past saturation.
const PILOT_TOKENS: u64 = 256;
const LOAD_ONLY_REPORT_SCHEMA_VERSION: u32 = 4;
const LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const LOAD_ONLY_SOAK_SECONDS: u64 = 30;

fn pilot_only_from(value: Option<&str>) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some("1") => Ok(true),
        Some(value) => Err(format!(
            "{PILOT_ONLY_ENV} must be the exact opt-in value 1, got {value:?}"
        )),
    }
}

// Byte-for-byte copies of the four decision workloads in
// tests/gemma4_spec50_bench.rs. Keep these literals synchronized; changing a
// prompt invalidates the N/I/M comparison.
const PARA: &str = "The ghost lane keeps a directory of resident experts per layer and pages the rest in from the \
sparse file on demand. Each decode step routes eight experts per layer, so the union over a short chunk of \
tokens grows roughly linearly with the chunk length until the hot set saturates. Because decode is bandwidth \
bound, reading every weight once per round and amortising it over several accepted tokens is the only way \
past the single-token wall on a sixteen gigabyte machine.";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Workload {
    Copy,
    CodeEdit,
    JsonYaml,
    Prose,
}

impl Workload {
    fn key(self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::CodeEdit => "code-edit",
            Self::JsonYaml => "json-yaml",
            Self::Prose => "prose",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WorkloadSpec {
    key: Workload,
    prompt: String,
}

fn decision_workloads() -> Vec<WorkloadSpec> {
    vec![
        WorkloadSpec {
            key: Workload::Copy,
            prompt: format!(
                "<|turn>user\nRepeat the following paragraph exactly, word for word:\n\n{PARA}\n<turn|>\n<|turn>model\n"
            ),
        },
        WorkloadSpec {
            key: Workload::CodeEdit,
            prompt: "<|turn>user\nAdd a `pub expires_at: u64,` field at the end of this struct and output the COMPLETE struct definition again, unchanged otherwise, with no explanation:\n\npub struct CacheEntry<K, V> {\n    pub key: K,\n    pub value: V,\n    pub access_count: u64,\n    pub created_at: u64,\n    pub last_hit: u64,\n}\n<turn|>\n<|turn>model\n".to_string(),
        },
        WorkloadSpec {
            key: Workload::JsonYaml,
            prompt: "<|turn>user\nConvert this configuration payload to YAML:\n{\"cluster_id\": \"prod-1\", \"min_replicas\": 4, \"max_replicas\": 32, \"enabled\": true}\n<turn|>\n<|turn>model\n".to_string(),
        },
        WorkloadSpec {
            key: Workload::Prose,
            prompt: "<|turn>user\nExplain in three short paragraphs how a hash map works and when it degrades.\n<turn|>\n<|turn>model\n".to_string(),
        },
    ]
}

/// N is the current seeded n-gram lane. I uses identical N proposals with the
/// assistant loaded and warmed but never invoked, isolating memory pressure.
/// M uses only the official MTP assistant. N0 is an optional diagnostic and is
/// never the primary denominator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum Lane {
    NgramBaseline,
    NgramAssistantIdle,
    Mtp,
    NgramSeedOffDiagnostic,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProposalSource {
    NgramSeeded,
    MtpAssistant,
    NgramSeedOff,
}

/// Exact native entry point selected by the isolated child. N and I must stay
/// on today's shipping n-gram implementation; M is the only lane permitted to
/// enter the experimental assistant driver.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum LaneDriver {
    CurrentSpecDecodeGenerate,
    NativeMtpExperiment,
}

/// Exact target-side environment inherited from the current spec50 authority.
/// Child processes apply these values as overrides; the parent process never
/// mutates global environment while another lane is live.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TargetRuntimeConfig {
    runtime_gguf_path: PathBuf,
    cghost_path: PathBuf,
    expert_cache_mib: usize,
    environment: BTreeMap<String, String>,
}

impl TargetRuntimeConfig {
    fn validate(&self) -> Result<(), String> {
        if !self.runtime_gguf_path.is_absolute() || !self.cghost_path.is_absolute() {
            return Err("target runtime and cghost paths must be absolute".into());
        }
        if self.expert_cache_mib != 0 {
            return Err(
                "hybrid mapped-cold target profile requires a zero-byte host expert cache".into(),
            );
        }
        let expected_environment = exact_target_environment();
        for (key, expected) in &expected_environment {
            if self.environment.get(key) != Some(expected) {
                return Err(format!(
                    "target runtime setting {key} is {:?}, expected {expected:?}",
                    self.environment.get(key)
                ));
            }
        }
        let unexpected = self
            .environment
            .keys()
            .filter(|key| !expected_environment.contains_key(*key))
            .cloned()
            .collect::<Vec<_>>();
        if !unexpected.is_empty() {
            return Err(format!(
                "target runtime contains unexpected settings: {}",
                unexpected.join(", ")
            ));
        }
        Ok(())
    }
}

/// Resolve a timed artifact and prove that it resides on the Mac's internal
/// startup volume. The full official source target is intentionally exempt: it
/// is read only by the parent pair gate before any timed child is started.
fn canonical_internal_timed_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("canonicalize timed {label} {}: {error}", path.display()))?;
    if !canonical.is_file() {
        return Err(format!(
            "timed {label} does not name a regular file: {}",
            canonical.display()
        ));
    }

    #[cfg(target_os = "macos")]
    {
        let raw = canonical
            .to_str()
            .ok_or_else(|| format!("timed {label} path is not UTF-8"))?;
        let raw = CString::new(raw).map_err(|_| format!("timed {label} path contains NUL"))?;
        let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(raw.as_ptr(), &mut stat) } != 0 {
            return Err(format!(
                "statfs timed {label} {}: {}",
                canonical.display(),
                std::io::Error::last_os_error()
            ));
        }
        let mount = unsafe { CStr::from_ptr(stat.f_mntonname.as_ptr()) }
            .to_str()
            .map_err(|_| format!("timed {label} mount point is not UTF-8"))?;
        if mount != "/" && mount != "/System/Volumes/Data" {
            return Err(format!(
                "timed {label} is on external/non-startup mount {mount:?}: {}",
                canonical.display()
            ));
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = label;
        return Err("the internal-only MTP experiment is scoped to macOS".into());
    }

    Ok(canonical)
}

fn current_internal_executable_sha256(label: &str) -> Result<String, String> {
    let path =
        std::env::current_exe().map_err(|error| format!("resolve {label} executable: {error}"))?;
    let path = canonical_internal_timed_file(&path, &format!("{label} executable"))?;
    file_sha256(&path, &format!("{label} executable"))
}

/// One lane is always one fresh child process. N/I carry byte-identical n-gram
/// settings; only I's resident-but-uninvoked assistant differs. M has no n-gram
/// fallback, so every proposal receipt must originate from the official MTP
/// driver. N0 is optional and never contributes the primary denominator.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct LaneExecutionPlan {
    lane: Lane,
    driver: LaneDriver,
    proposal_source: ProposalSource,
    fresh_process: bool,
    target_warmup_tokens: u64,
    snapshot_before_assistant_load: bool,
    load_assistant: bool,
    warm_assistant: bool,
    invoke_assistant: bool,
    proposal_environment: BTreeMap<String, String>,
}

impl LaneExecutionPlan {
    fn for_lane(lane: Lane) -> Self {
        let seeded = BTreeMap::from([
            (
                "CAMELID_GEMMA4_SPEC_SEED_TEXT".to_string(),
                DEFAULT_SEEDED_NGRAM_TEXT.to_string(),
            ),
            ("CAMELID_GEMMA4_SPEC_MIN_MATCH".to_string(), "1".to_string()),
            ("CAMELID_GEMMA4_SPEC_MAX_MATCH".to_string(), "5".to_string()),
            ("CAMELID_GEMMA4_SPEC_ADAPTIVE".to_string(), "0".to_string()),
            ("CAMELID_GEMMA4_SPEC_TIERS".to_string(), "1".to_string()),
            ("CAMELID_GEMMA4_SPEC_TIER_WIDE".to_string(), "5".to_string()),
            ("CAMELID_GEMMA4_SPEC_TIER_MID".to_string(), "3".to_string()),
            (
                "CAMELID_GEMMA4_SPEC_TIER_MID_WIDTH".to_string(),
                "3".to_string(),
            ),
            ("CAMELID_GEMMA4_SPEC_ECON".to_string(), "1".to_string()),
            ("CAMELID_GEMMA4_SPEC_ECON_MIN".to_string(), "12".to_string()),
        ]);
        match lane {
            Lane::NgramBaseline => Self {
                lane,
                driver: LaneDriver::CurrentSpecDecodeGenerate,
                proposal_source: ProposalSource::NgramSeeded,
                fresh_process: true,
                target_warmup_tokens: TARGET_WARMUP_TOKENS,
                snapshot_before_assistant_load: true,
                load_assistant: false,
                warm_assistant: false,
                invoke_assistant: false,
                proposal_environment: seeded,
            },
            Lane::NgramAssistantIdle => Self {
                lane,
                driver: LaneDriver::CurrentSpecDecodeGenerate,
                proposal_source: ProposalSource::NgramSeeded,
                fresh_process: true,
                target_warmup_tokens: TARGET_WARMUP_TOKENS,
                snapshot_before_assistant_load: true,
                load_assistant: true,
                warm_assistant: true,
                invoke_assistant: false,
                proposal_environment: seeded,
            },
            Lane::Mtp => Self {
                lane,
                driver: LaneDriver::NativeMtpExperiment,
                proposal_source: ProposalSource::MtpAssistant,
                fresh_process: true,
                target_warmup_tokens: TARGET_WARMUP_TOKENS,
                snapshot_before_assistant_load: true,
                load_assistant: true,
                warm_assistant: true,
                invoke_assistant: true,
                proposal_environment: BTreeMap::new(),
            },
            Lane::NgramSeedOffDiagnostic => {
                let mut seed_off = seeded;
                seed_off.insert("CAMELID_GEMMA4_SPEC_SEED_TEXT".into(), String::new());
                Self {
                    lane,
                    driver: LaneDriver::CurrentSpecDecodeGenerate,
                    proposal_source: ProposalSource::NgramSeedOff,
                    fresh_process: true,
                    target_warmup_tokens: TARGET_WARMUP_TOKENS,
                    snapshot_before_assistant_load: true,
                    load_assistant: false,
                    warm_assistant: false,
                    invoke_assistant: false,
                    proposal_environment: seed_off,
                }
            }
        }
    }
}

fn validate_lane_plans(plans: &[LaneExecutionPlan]) -> Result<(), String> {
    for lane in [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp] {
        if plans.iter().filter(|plan| plan.lane == lane).count() != 1 {
            return Err(format!("lane plan must contain exactly one {lane:?}"));
        }
    }
    for plan in plans {
        if !plan.fresh_process {
            return Err(format!(
                "{:?} is not isolated in a fresh process",
                plan.lane
            ));
        }
        if plan.target_warmup_tokens != TARGET_WARMUP_TOKENS || !plan.snapshot_before_assistant_load
        {
            return Err(format!(
                "{:?} does not use the authoritative target warmup/snapshot boundary",
                plan.lane
            ));
        }
        let expected = LaneExecutionPlan::for_lane(plan.lane);
        if plan != &expected {
            return Err(format!(
                "{:?} lane wiring differs from authority",
                plan.lane
            ));
        }
    }
    let n = plans
        .iter()
        .find(|plan| plan.lane == Lane::NgramBaseline)
        .expect("presence checked");
    let i = plans
        .iter()
        .find(|plan| plan.lane == Lane::NgramAssistantIdle)
        .expect("presence checked");
    if n.proposal_source != i.proposal_source
        || n.proposal_environment != i.proposal_environment
        || i.invoke_assistant
    {
        return Err("N/I proposal wiring is not identical or I invokes the assistant".into());
    }
    Ok(())
}

fn exact_target_environment() -> BTreeMap<String, String> {
    [
        ("CAMELID_GHOST_ALLOW_LEGACY_SPARSE", "0"),
        ("CAMELID_GEMMA4_GHOST_METAL", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_TURBO", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_COMMON", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_CONTEXT", "1024"),
        ("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT", "0"),
        // Hybrid clean-pager profile: 48 pageable anonymous records cache the
        // hot set while all 128 canonical IDs remain addressable through the
        // retained read-only mapping. No legacy physical-prefix knob may
        // silently redefine either namespace.
        ("CAMELID_GEMMA4_GHOST_READ_THREADS", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS", "1"),
        ("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS", "48"),
        // The initial hybrid receipt is deliberately pageable. Pinned hot
        // records are a separate experiment and must not reuse this profile.
        ("CAMELID_GEMMA4_SLOT_PIN", "0"),
        ("CAMELID_GEMMA4_GHOST_METAL_HOT_PIN", "0"),
        ("CAMELID_GEMMA4_VICTIM_CACHE", "0"),
        ("CAMELID_GEMMA4_VICTIM_MB", "0"),
        ("CAMELID_GEMMA4_ALLOW_DROPPED_EXPERTS", "0"),
        ("CAMELID_GEMMA4_CHAINED_PREDICT", "0"),
        ("CAMELID_SPEC_DECODE", "off"),
        ("CAMELID_GEMMA4_SPEC_CHUNK_MAX", "8"),
        // Preserve spec50's K=8 request exactly. The runtime's K=8 verifier
        // ceiling clamps this to seven proposals plus the target anchor.
        ("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "8"),
        ("CAMELID_GEMMA4_SPEC_K1_LANE", "chained"),
        ("CAMELID_GEMMA4_SPEC_TIMING", "1"),
        // Per-round critical-path decomposition ([metal chained ledger] /
        // [metal chained idle]: slot_wait, slot_filler, nvme_ms, encode,
        // gpu_busy, per-stage GPU). The lane child scrubs every inherited
        // CAMELID_GEMMA4_* key so a tuning knob cannot escape the receipt, so an
        // observability flag only reaches the child if it is serialized here.
        ("CAMELID_GEMMA4_GHOST_METAL_TIMING", "1"),
        // Per-layer routed-expert trace ([route-metal] l=<layer> e=[<expert ids>]).
        // Captures the exact expert access SEQUENCE so the slot-cache hit rate can be
        // simulated offline at any slot count, without re-running the model once per
        // candidate bound - the bound sweep is the decisive open question and 64 slots
        // cannot even be run on this host.
        ("CAMELID_GEMMA4_ROUTE_TRACE", "1"),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .collect()
}

fn target_runtime_config_from_environment() -> Result<TargetRuntimeConfig, String> {
    let path = |name: &str, default: &str| -> Result<PathBuf, String> {
        let path = std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(default));
        if !path.is_absolute() {
            return Err(format!("{name} must be an absolute path"));
        }
        if !path.is_file() {
            return Err(format!("{name} does not name a file: {}", path.display()));
        }
        Ok(path)
    };
    let expert_cache_mib = std::env::var(TARGET_CACHE_MIB_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{TARGET_CACHE_MIB_ENV} is not an integer"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_TARGET_CACHE_MIB);
    let config = TargetRuntimeConfig {
        runtime_gguf_path: path(TARGET_RUNTIME_PATH_ENV, DEFAULT_TARGET_RUNTIME_PATH)?,
        cghost_path: path(TARGET_CGHOST_PATH_ENV, DEFAULT_TARGET_CGHOST_PATH)?,
        expert_cache_mib,
        environment: exact_target_environment(),
    };
    config.validate()?;
    Ok(config)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RoundTelemetry {
    workload: Workload,
    lane: Lane,
    repetition: u32,
    round_index: u64,
    prefix_tokens_before: u64,
    proposal_source: ProposalSource,
    /// False for M's target-only bootstrap; true only when this round called
    /// the assistant proposal kernel during the measured generation.
    assistant_invoked: bool,
    requested_k: u32,
    proposed_k: u32,
    /// Actual width submitted to CLAIRE's target verifier, not requested K.
    verifier_k: u32,
    /// True only when the final output budget clipped this round. Keeping tail
    /// rounds separate prevents the requested/actual K distribution from being
    /// biased downward by an accounting artifact.
    budget_truncated: bool,
    /// Raw verifier prefix length. This can extend beyond an accepted EOT and is
    /// retained for numerical diagnostics only.
    accepted_drafts: u32,
    /// Accepted drafts that are useful to generation, stopping before the first
    /// accepted EOT. This is the decision-metric numerator.
    useful_accepted_drafts: u32,
    /// Target-accepted verifier prefix, including a committed stop token (and
    /// any already-verified prefix after it). This is the acceptance numerator.
    emitted_target_tokens: u32,
    /// User-visible output from this round; stop tokens are omitted.
    visible_output_tokens: u32,
    draft_wall_us: u64,
    draft_gpu_us: u64,
    verify_wall_us: u64,
    verify_gpu_us: u64,
    round_wall_us: u64,
}

impl RoundTelemetry {
    fn validate(&self) -> Result<(), String> {
        let expected_source = match self.lane {
            Lane::NgramBaseline | Lane::NgramAssistantIdle => ProposalSource::NgramSeeded,
            Lane::Mtp => ProposalSource::MtpAssistant,
            Lane::NgramSeedOffDiagnostic => ProposalSource::NgramSeedOff,
        };
        if self.proposal_source != expected_source {
            return Err(format!(
                "round {} lane {:?} used {:?} proposals instead of {:?}",
                self.round_index, self.lane, self.proposal_source, expected_source
            ));
        }
        if !(1..=8).contains(&self.requested_k) {
            return Err(format!(
                "round {} requested K={} is outside the exact 1..=8 regime",
                self.round_index, self.requested_k
            ));
        }
        if self.proposed_k > self.requested_k {
            return Err(format!(
                "round {} proposed K={} exceeds requested K={}",
                self.round_index, self.proposed_k, self.requested_k
            ));
        }
        if self.accepted_drafts > self.proposed_k {
            return Err(format!(
                "round {} accepted {} drafts from only {} proposals",
                self.round_index, self.accepted_drafts, self.proposed_k
            ));
        }
        if self.useful_accepted_drafts > self.accepted_drafts {
            return Err(format!(
                "round {} reports {} useful drafts from only {} raw accepted drafts",
                self.round_index, self.useful_accepted_drafts, self.accepted_drafts
            ));
        }
        if !(1..=8).contains(&self.verifier_k) {
            return Err(format!(
                "round {} verifier K={} is outside the exact 1..=8 regime",
                self.round_index, self.verifier_k
            ));
        }
        if self.verifier_k > self.requested_k {
            return Err(format!(
                "round {} actual verifier K={} exceeds requested K={}",
                self.round_index, self.verifier_k, self.requested_k
            ));
        }
        if self.verifier_k != self.proposed_k.saturating_add(1) {
            return Err(format!(
                "round {} verifier K={} must equal proposed K {} + authoritative anchor",
                self.round_index, self.verifier_k, self.proposed_k
            ));
        }
        if self.emitted_target_tokens != self.useful_accepted_drafts.saturating_add(1) {
            return Err(format!(
                "round {} emitted {} useful target tokens, expected {} useful accepted drafts + authoritative anchor",
                self.round_index,
                self.emitted_target_tokens,
                self.useful_accepted_drafts
            ));
        }
        if self.visible_output_tokens > self.emitted_target_tokens {
            return Err(format!(
                "round {} exposes {} output tokens from only {} target-committed tokens",
                self.round_index, self.visible_output_tokens, self.emitted_target_tokens
            ));
        }
        if self.verify_wall_us == 0 || self.verify_gpu_us == 0 {
            return Err(format!(
                "round {} is missing target-verifier wall/GPU timing",
                self.round_index
            ));
        }
        if self.proposal_source == ProposalSource::MtpAssistant {
            let should_invoke = self.proposed_k > 0;
            if self.assistant_invoked != should_invoke {
                return Err(format!(
                    "round {} assistant invocation={} but proposed K={}",
                    self.round_index, self.assistant_invoked, self.proposed_k
                ));
            }
            if self.assistant_invoked && (self.draft_wall_us == 0 || self.draft_gpu_us == 0) {
                return Err(format!(
                    "round {} is missing MTP-assistant wall/GPU timing",
                    self.round_index
                ));
            }
            if !self.assistant_invoked && (self.draft_wall_us != 0 || self.draft_gpu_us != 0) {
                return Err(format!(
                    "round {} reports assistant timing without an invocation",
                    self.round_index
                ));
            }
        }
        if self.draft_gpu_us > self.draft_wall_us {
            return Err(format!(
                "round {} draft GPU {}us exceeds wall {}us",
                self.round_index, self.draft_gpu_us, self.draft_wall_us
            ));
        }
        if self.verify_gpu_us > self.verify_wall_us {
            return Err(format!(
                "round {} verify GPU {}us exceeds wall {}us",
                self.round_index, self.verify_gpu_us, self.verify_wall_us
            ));
        }
        if self.round_wall_us < self.draft_wall_us.max(self.verify_wall_us) {
            return Err(format!(
                "round {} total wall {}us is shorter than an observed phase",
                self.round_index, self.round_wall_us
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
struct RunMetrics {
    /// Histogram values below are round counts. Use this as the denominator for
    /// all-round percentages, and full_budget_rounds/tail_truncated_rounds for
    /// the split verifier-K histograms.
    rounds: u64,
    proposed_tokens: u64,
    /// Raw target-verifier acceptance, including matches after an accepted EOT.
    accepted_drafts: u64,
    /// Useful acceptance through (but not after) the first accepted EOT.
    useful_accepted_drafts: u64,
    /// Target-accepted verifier tokens, including committed stop tokens.
    emitted_target_tokens: u64,
    /// User-visible output tokens, excluding stop tokens.
    visible_output_tokens: u64,
    /// Target-authoritative boundary tokens emitted without a discarded
    /// verifier forward. They count in throughput but never in K histograms.
    terminal_target_tokens: u64,
    draft_wall_us: u64,
    draft_gpu_us: u64,
    verify_wall_us: u64,
    verify_gpu_us: u64,
    total_wall_us: u64,
    acceptance_probability: f64,
    useful_acceptance_probability: f64,
    accepted_drafts_per_round: f64,
    /// Authoritative emitted tokens per round, including the target anchor.
    accepted_target_tokens_per_round: f64,
    draft_wall_us_per_proposal: f64,
    draft_gpu_us_per_proposal: f64,
    requested_k_histogram: BTreeMap<u32, u64>,
    proposed_k_histogram: BTreeMap<u32, u64>,
    /// The required target-verification K distribution.
    verifier_k_histogram: BTreeMap<u32, u64>,
    accepted_drafts_histogram: BTreeMap<u32, u64>,
    full_budget_rounds: u64,
    tail_truncated_rounds: u64,
    full_budget_requested_k_histogram: BTreeMap<u32, u64>,
    full_budget_proposed_k_histogram: BTreeMap<u32, u64>,
    full_budget_verifier_k_histogram: BTreeMap<u32, u64>,
    tail_truncated_requested_k_histogram: BTreeMap<u32, u64>,
    tail_truncated_proposed_k_histogram: BTreeMap<u32, u64>,
    tail_truncated_verifier_k_histogram: BTreeMap<u32, u64>,
}

impl RunMetrics {
    fn from_rounds(rounds: &[RoundTelemetry]) -> Result<Self, String> {
        Self::from_rounds_and_terminal(rounds, 0)
    }

    fn from_rounds_and_terminal(
        rounds: &[RoundTelemetry],
        terminal_target_tokens: u64,
    ) -> Result<Self, String> {
        let mut metrics = Self::default();
        for round in rounds {
            round.validate()?;
            metrics.rounds += 1;
            metrics.proposed_tokens += u64::from(round.proposed_k);
            metrics.accepted_drafts += u64::from(round.accepted_drafts);
            metrics.useful_accepted_drafts += u64::from(round.useful_accepted_drafts);
            metrics.emitted_target_tokens += u64::from(round.emitted_target_tokens);
            metrics.visible_output_tokens += u64::from(round.visible_output_tokens);
            metrics.draft_wall_us += round.draft_wall_us;
            metrics.draft_gpu_us += round.draft_gpu_us;
            metrics.verify_wall_us += round.verify_wall_us;
            metrics.verify_gpu_us += round.verify_gpu_us;
            metrics.total_wall_us += round.round_wall_us;
            bump(&mut metrics.requested_k_histogram, round.requested_k);
            bump(&mut metrics.proposed_k_histogram, round.proposed_k);
            bump(&mut metrics.verifier_k_histogram, round.verifier_k);
            bump(
                &mut metrics.accepted_drafts_histogram,
                round.accepted_drafts,
            );
            if round.budget_truncated {
                metrics.tail_truncated_rounds += 1;
                bump(
                    &mut metrics.tail_truncated_requested_k_histogram,
                    round.requested_k,
                );
                bump(
                    &mut metrics.tail_truncated_proposed_k_histogram,
                    round.proposed_k,
                );
                bump(
                    &mut metrics.tail_truncated_verifier_k_histogram,
                    round.verifier_k,
                );
            } else {
                metrics.full_budget_rounds += 1;
                bump(
                    &mut metrics.full_budget_requested_k_histogram,
                    round.requested_k,
                );
                bump(
                    &mut metrics.full_budget_proposed_k_histogram,
                    round.proposed_k,
                );
                bump(
                    &mut metrics.full_budget_verifier_k_histogram,
                    round.verifier_k,
                );
            }
        }
        metrics.terminal_target_tokens = terminal_target_tokens;
        metrics.visible_output_tokens = metrics
            .visible_output_tokens
            .saturating_add(terminal_target_tokens);
        metrics.acceptance_probability = ratio(metrics.accepted_drafts, metrics.proposed_tokens);
        metrics.useful_acceptance_probability =
            ratio(metrics.useful_accepted_drafts, metrics.proposed_tokens);
        metrics.accepted_drafts_per_round = ratio(metrics.accepted_drafts, metrics.rounds);
        metrics.accepted_target_tokens_per_round =
            ratio(metrics.emitted_target_tokens, metrics.rounds);
        metrics.draft_wall_us_per_proposal = ratio(metrics.draft_wall_us, metrics.proposed_tokens);
        metrics.draft_gpu_us_per_proposal = ratio(metrics.draft_gpu_us, metrics.proposed_tokens);
        Ok(metrics)
    }
}

fn bump(histogram: &mut BTreeMap<u32, u64>, value: u32) {
    *histogram.entry(value).or_default() += 1;
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Self-contained receipt from tests/gemma4_mtp_pair_gate.rs. The full official
/// target is the hash-pinned source of truth; the runtime target may be a sparse
/// common-core shadow only when both it and its `.cghost` pass the sampled
/// source-identity proof. Assistant staging is whole-file hash exact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct PairingEvidence {
    pair_gate_passed: bool,

    target_repository: String,
    target_revision: String,
    target_source_path: PathBuf,
    target_source_bytes: u64,
    target_source_sha256: String,
    target_source_matches_official: bool,
    target_staged_runtime_path: PathBuf,
    target_staged_cghost_path: PathBuf,
    target_staged_identity_scheme: String,
    target_staged_metadata_matches_source: bool,
    target_cghost_matches_official_source: bool,
    target_cghost_matches_staged_runtime: bool,

    assistant_repository: String,
    assistant_revision: String,
    assistant_staged_model_path: PathBuf,
    assistant_staged_model_bytes: u64,
    assistant_staged_model_sha256: String,
    assistant_staged_config_path: PathBuf,
    assistant_staged_config_sha256: String,
    assistant_staged_tokenizer_config_path: PathBuf,
    assistant_staged_tokenizer_config_sha256: String,
    assistant_staged_tokenizer_path: PathBuf,
    assistant_staged_tokenizer_sha256: String,
    assistant_staged_files_match_official: bool,

    shared_vocab_size: u64,
    tokenizer_mismatch_count: u64,
    target_shared_kv_layers: u32,
    assistant_shared_kv_layers: u32,
    target_shared_kv_sliding_source_layer: u32,
    target_shared_kv_full_source_layer: u32,
}

impl PairingEvidence {
    fn validate(&self, selected_assistant_path: &PathBuf) -> Result<(), String> {
        if !self.pair_gate_passed {
            return Err("official MTP pair gate did not pass".to_string());
        }
        for (label, path) in [
            ("target source", &self.target_source_path),
            ("staged target runtime", &self.target_staged_runtime_path),
            ("staged target cghost", &self.target_staged_cghost_path),
            ("staged assistant model", &self.assistant_staged_model_path),
            (
                "staged assistant config",
                &self.assistant_staged_config_path,
            ),
            (
                "staged assistant tokenizer config",
                &self.assistant_staged_tokenizer_config_path,
            ),
            (
                "staged assistant tokenizer",
                &self.assistant_staged_tokenizer_path,
            ),
        ] {
            if !path.is_absolute() {
                return Err(format!("pair receipt {label} path is not absolute"));
            }
        }
        if &self.assistant_staged_model_path != selected_assistant_path {
            return Err("selected assistant path differs from the pair-gated staged model".into());
        }
        if self.target_repository != OFFICIAL_TARGET_REPOSITORY
            || self.target_revision != OFFICIAL_TARGET_REVISION
            || self.target_source_bytes != OFFICIAL_TARGET_BYTES
            || self.target_source_sha256 != OFFICIAL_TARGET_SHA256
            || !self.target_source_matches_official
        {
            return Err("target source identity is not the pinned official QAT target".into());
        }
        if self.target_staged_identity_scheme != STAGED_TARGET_IDENTITY_SCHEME
            || !self.target_staged_metadata_matches_source
            || !self.target_cghost_matches_official_source
            || !self.target_cghost_matches_staged_runtime
        {
            return Err(
                "staged sparse target/cghost identity is not bound to the official source".into(),
            );
        }
        if self.assistant_repository != OFFICIAL_ASSISTANT_REPOSITORY
            || self.assistant_revision != OFFICIAL_ASSISTANT_REVISION
            || self.assistant_staged_model_bytes != OFFICIAL_ASSISTANT_MODEL_BYTES
            || self.assistant_staged_model_sha256 != OFFICIAL_ASSISTANT_MODEL_SHA256
            || self.assistant_staged_config_sha256 != OFFICIAL_ASSISTANT_CONFIG_SHA256
            || self.assistant_staged_tokenizer_config_sha256
                != OFFICIAL_ASSISTANT_TOKENIZER_CONFIG_SHA256
            || self.assistant_staged_tokenizer_sha256 != OFFICIAL_ASSISTANT_TOKENIZER_SHA256
            || !self.assistant_staged_files_match_official
        {
            return Err("staged assistant identity is not the pinned official checkpoint".into());
        }
        if self.shared_vocab_size != SHARED_VOCAB_SIZE || self.tokenizer_mismatch_count != 0 {
            return Err(format!(
                "tokenizer pair is not exact: vocab={} mismatches={}",
                self.shared_vocab_size, self.tokenizer_mismatch_count
            ));
        }
        if self.target_shared_kv_layers != TARGET_SHARED_KV_LAYERS
            || self.assistant_shared_kv_layers != ASSISTANT_SHARED_KV_LAYERS
            || self.target_shared_kv_sliding_source_layer != TARGET_SHARED_KV_SLIDING_SOURCE_LAYER
            || self.target_shared_kv_full_source_layer != TARGET_SHARED_KV_FULL_SOURCE_LAYER
        {
            return Err(format!(
                "shared-KV metadata mismatch: target={} assistant={} sliding_source={} full_source={}",
                self.target_shared_kv_layers,
                self.assistant_shared_kv_layers,
                self.target_shared_kv_sliding_source_layer,
                self.target_shared_kv_full_source_layer
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeAdmissionStepEvidence {
    step: usize,
    token_id: u32,
    top16_overlap: usize,
    recurrent_cosine: f64,
    recurrent_relative_l2: f64,
    native_margin_bf16_ulp: u32,
    required_margin_bf16_ulp: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeAdmissionThresholds {
    exact_top1_steps: usize,
    minimum_top16_overlap_per_step: usize,
    minimum_recurrent_cosine_per_step: f64,
    maximum_recurrent_relative_l2_per_step: f64,
    native_margin_cap_bf16_ulp: u32,
    native_margin_floor_rule: String,
    required_margin_floors_bf16_ulp: Vec<u32>,
    require_bf16_lattice: bool,
    require_repeat_bit_determinism: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeAdmissionPlatform {
    os: String,
    os_version: String,
    machine_arch: String,
    machine_model: String,
    metal_device_name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeAdmissionEvidence {
    schema_version: u32,
    pass: bool,
    assistant_model_sha256: String,
    recurrence_oracle_sha256: String,
    recurrence_generation_receipt_sha256: String,
    stage_oracle_sha256: String,
    structural_stage_pass: bool,
    admission_test_exe_sha256: String,
    native_source_sha256: String,
    metal_rs_source_sha256: String,
    gemma4_runtime_rs_source_sha256: String,
    cargo_lock_sha256: String,
    platform: NativeAdmissionPlatform,
    created_unix_ms: u64,
    run_nonce: String,
    test_name: String,
    top1_token_ids: Vec<u32>,
    native_repetitions: usize,
    repeat_bit_deterministic: bool,
    per_step: Vec<NativeAdmissionStepEvidence>,
    min_top16_overlap: usize,
    min_recurrent_cosine: f64,
    max_recurrent_relative_l2: f64,
    tie_policy: String,
    tie_policy_test_pass: bool,
    teacher_forcing: bool,
    thresholds: NativeAdmissionThresholds,
}

impl NativeAdmissionEvidence {
    fn validate(
        &self,
        integration_executable_sha256: &str,
        expected_run_nonce: &str,
    ) -> Result<(), String> {
        if self.schema_version != NATIVE_ADMISSION_SCHEMA_VERSION || !self.pass {
            return Err("native recurrence admission receipt is not a schema-v1 PASS".into());
        }
        for (label, digest) in [
            ("assistant model", self.assistant_model_sha256.as_str()),
            ("recurrence oracle", self.recurrence_oracle_sha256.as_str()),
            (
                "recurrence generation receipt",
                self.recurrence_generation_receipt_sha256.as_str(),
            ),
            ("stage oracle", self.stage_oracle_sha256.as_str()),
            (
                "admission test executable",
                self.admission_test_exe_sha256.as_str(),
            ),
            ("native source", self.native_source_sha256.as_str()),
            ("metal.rs source", self.metal_rs_source_sha256.as_str()),
            (
                "gemma4_runtime.rs source",
                self.gemma4_runtime_rs_source_sha256.as_str(),
            ),
            ("Cargo.lock", self.cargo_lock_sha256.as_str()),
        ] {
            if !is_sha256_hex(digest) {
                return Err(format!(
                    "native admission {label} digest is not lowercase SHA-256"
                ));
            }
        }
        if !is_sha256_hex(integration_executable_sha256) {
            return Err("integration-test executable digest is not lowercase SHA-256".into());
        }
        if self.assistant_model_sha256 != OFFICIAL_ASSISTANT_MODEL_SHA256
            || self.recurrence_oracle_sha256 != NATIVE_RECURRENCE_ORACLE_SHA256
            || self.recurrence_generation_receipt_sha256
                != NATIVE_RECURRENCE_GENERATION_RECEIPT_SHA256
            || self.stage_oracle_sha256 != NATIVE_STAGE_ORACLE_SHA256
            || self.native_source_sha256 != native_source_sha256()
            || self.metal_rs_source_sha256 != bytes_sha256(METAL_SOURCE.as_bytes())
            || self.gemma4_runtime_rs_source_sha256
                != bytes_sha256(GEMMA4_RUNTIME_SOURCE.as_bytes())
            || self.cargo_lock_sha256 != bytes_sha256(CARGO_LOCK_SOURCE.as_bytes())
            || !self.structural_stage_pass
        {
            return Err(
                "native admission is not bound to this assistant/oracles/source bundle".into(),
            );
        }
        for (label, digest) in [
            ("recurrence run-1 log", NATIVE_RECURRENCE_RUN1_LOG_SHA256),
            ("recurrence run-2 log", NATIVE_RECURRENCE_RUN2_LOG_SHA256),
            (
                "recurrence failed-attempts log",
                NATIVE_RECURRENCE_FAILED_ATTEMPTS_LOG_SHA256,
            ),
        ] {
            if !GEMMA4_MTP_SOURCE.contains(digest) {
                return Err(format!(
                    "compile-time native source omitted pinned {label} provenance"
                ));
            }
        }
        validate_native_admission_run_nonce(expected_run_nonce)?;
        validate_native_admission_run_nonce(&self.run_nonce)?;
        if self.run_nonce != expected_run_nonce {
            return Err("native admission run nonce differs from the parent invocation".into());
        }
        if self.created_unix_ms == 0 {
            return Err("native admission creation timestamp is absent".into());
        }
        let current_platform = current_native_admission_platform()?;
        if self.platform != current_platform {
            return Err(format!(
                "native admission platform {:?} differs from current pilot host {:?}",
                self.platform, current_platform
            ));
        }
        if self.admission_test_exe_sha256 == integration_executable_sha256 {
            return Err(
                "native admission and integration executables were incorrectly conflated".into(),
            );
        }
        if self.test_name != NATIVE_RECURRENCE_TEST_NAME
            || self.top1_token_ids != NATIVE_RECURRENCE_TOP1_TOKENS
            || self.native_repetitions != NATIVE_RECURRENCE_REPETITIONS as usize
            || !self.repeat_bit_deterministic
            || self.tie_policy != NATIVE_ARGMAX_TIE_POLICY
            || !self.tie_policy_test_pass
            || self.teacher_forcing
        {
            return Err("native recurrence admission identity/policy receipt drifted".into());
        }
        if self.thresholds.exact_top1_steps != NATIVE_RECURRENCE_TOP1_TOKENS.len()
            || self.thresholds.minimum_top16_overlap_per_step != NATIVE_MIN_TOP16_OVERLAP as usize
            || self.thresholds.minimum_recurrent_cosine_per_step != NATIVE_MIN_RECURRENT_COSINE
            || self.thresholds.maximum_recurrent_relative_l2_per_step
                != NATIVE_MAX_RECURRENT_RELATIVE_L2
            || self.thresholds.native_margin_cap_bf16_ulp != NATIVE_MARGIN_CAP_BF16_ULP
            || self.thresholds.native_margin_floor_rule != NATIVE_MARGIN_FLOOR_RULE
            || self.thresholds.required_margin_floors_bf16_ulp != NATIVE_REQUIRED_MARGIN_BF16_ULP
            || !self.thresholds.require_bf16_lattice
            || !self.thresholds.require_repeat_bit_determinism
        {
            return Err("native recurrence admission thresholds drifted".into());
        }
        if self.per_step.len() != NATIVE_RECURRENCE_TOP1_TOKENS.len() {
            return Err(format!(
                "native admission has {} steps, expected {}",
                self.per_step.len(),
                NATIVE_RECURRENCE_TOP1_TOKENS.len()
            ));
        }
        for (index, step) in self.per_step.iter().enumerate() {
            if step.step != index
                || step.token_id != NATIVE_RECURRENCE_TOP1_TOKENS[index]
                || step.top16_overlap < NATIVE_MIN_TOP16_OVERLAP as usize
                || step.top16_overlap > 16
                || !step.recurrent_cosine.is_finite()
                || step.recurrent_cosine < NATIVE_MIN_RECURRENT_COSINE
                || !step.recurrent_relative_l2.is_finite()
                || step.recurrent_relative_l2 < 0.0
                || step.recurrent_relative_l2 > NATIVE_MAX_RECURRENT_RELATIVE_L2
                || step.required_margin_bf16_ulp != NATIVE_REQUIRED_MARGIN_BF16_ULP[index]
                || step.native_margin_bf16_ulp < step.required_margin_bf16_ulp
            {
                return Err(format!(
                    "native recurrence admission step {index} violates its pinned gate"
                ));
            }
        }
        let derived_min_overlap = self
            .per_step
            .iter()
            .map(|step| step.top16_overlap)
            .min()
            .ok_or_else(|| "native admission has no overlap samples".to_string())?;
        let derived_min_cosine = self
            .per_step
            .iter()
            .map(|step| step.recurrent_cosine)
            .fold(f64::INFINITY, f64::min);
        let derived_max_relative_l2 = self
            .per_step
            .iter()
            .map(|step| step.recurrent_relative_l2)
            .fold(0.0f64, f64::max);
        if self.min_top16_overlap != derived_min_overlap
            || self.min_recurrent_cosine.to_bits() != derived_min_cosine.to_bits()
            || self.max_recurrent_relative_l2.to_bits() != derived_max_relative_l2.to_bits()
        {
            return Err("native admission aggregates do not match per-step evidence".into());
        }
        Ok(())
    }
}

/// Raw receipt identity plus the parsed, validated evidence. Children receive
/// this value over structured IPC and never reopen the mutable source path.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NativeAdmissionReceipt {
    receipt_path: PathBuf,
    receipt_sha256: String,
    receipt_raw_json: String,
    receipt_modified_unix_ms: u64,
    parent_validated_unix_ms: u64,
    admission_test_executable_path: PathBuf,
    admission_test_executable_sha256: String,
    evidence: NativeAdmissionEvidence,
}

impl NativeAdmissionReceipt {
    fn validate(
        &self,
        integration_executable_sha256: &str,
        expected_run_nonce: &str,
    ) -> Result<(), String> {
        if !self.receipt_path.is_absolute()
            || !is_sha256_hex(&self.receipt_sha256)
            || self.receipt_modified_unix_ms == 0
            || self.parent_validated_unix_ms == 0
            || !self.admission_test_executable_path.is_absolute()
            || !self
                .admission_test_executable_path
                .starts_with("/Users/timtoole/")
            || !is_sha256_hex(&self.admission_test_executable_sha256)
        {
            return Err("native admission receipt identity is incomplete".into());
        }
        if self.admission_test_executable_sha256 != self.evidence.admission_test_exe_sha256 {
            return Err(
                "parent-rehashed admission test executable differs from the PASS evidence".into(),
            );
        }
        if self.receipt_sha256 != bytes_sha256(self.receipt_raw_json.as_bytes()) {
            return Err("native admission raw receipt does not match its SHA-256".into());
        }
        let reparsed: NativeAdmissionEvidence = serde_json::from_str(&self.receipt_raw_json)
            .map_err(|error| format!("reparse embedded native admission receipt: {error}"))?;
        if reparsed != self.evidence {
            return Err("parsed native admission differs from its embedded raw receipt".into());
        }
        let now_unix_ms = current_unix_ms("validate native admission freshness")?;
        let max_age_ms = MAX_NATIVE_ADMISSION_AGE.as_millis() as u64;
        let max_clock_skew_ms = MAX_NATIVE_ADMISSION_CLOCK_SKEW.as_millis() as u64;
        let max_create_to_mtime_ms = MAX_NATIVE_ADMISSION_CREATE_TO_MTIME.as_millis() as u64;
        if self.evidence.created_unix_ms > now_unix_ms.saturating_add(max_clock_skew_ms) {
            return Err("native admission creation timestamp is in the future".into());
        }
        if now_unix_ms.saturating_sub(self.evidence.created_unix_ms) > max_age_ms {
            return Err("native admission evidence is stale at this validation boundary".into());
        }
        if self
            .receipt_modified_unix_ms
            .saturating_add(max_clock_skew_ms)
            < self.evidence.created_unix_ms
            || self.receipt_modified_unix_ms
                > self
                    .evidence
                    .created_unix_ms
                    .saturating_add(max_create_to_mtime_ms)
        {
            return Err(
                "native admission file mtime is not adjacent to its embedded creation time".into(),
            );
        }
        let admitted_age_ms = self
            .parent_validated_unix_ms
            .checked_sub(self.receipt_modified_unix_ms)
            .ok_or_else(|| {
                "native admission receipt mtime follows parent validation".to_string()
            })?;
        if admitted_age_ms > MAX_NATIVE_ADMISSION_AGE.as_millis() as u64 {
            return Err("native admission was stale when the parent validated it".into());
        }
        let created_age_at_parent_ms = self
            .parent_validated_unix_ms
            .checked_sub(self.evidence.created_unix_ms)
            .ok_or_else(|| {
                "native admission creation timestamp follows parent validation".to_string()
            })?;
        if created_age_at_parent_ms > max_age_ms {
            return Err("native admission evidence was stale when the parent validated it".into());
        }
        if self.parent_validated_unix_ms > now_unix_ms.saturating_add(max_clock_skew_ms) {
            return Err("native admission parent-validation timestamp is in the future".into());
        }
        self.evidence
            .validate(integration_executable_sha256, expected_run_nonce)
    }
}

fn exact_cached_sha256(path: &Path, expected: &str, label: &str) -> Result<(), String> {
    let actual = file_sha256(path, label)?;
    if actual != expected {
        return Err(format!(
            "{label} SHA-256 {actual} does not match pinned {expected}"
        ));
    }
    Ok(())
}

fn file_sha256(path: &Path, label: &str) -> Result<String, String> {
    camelid::receipt::sha256_file_hex_cached(path)
        .map_err(|error| format!("hash {label} {}: {error}", path.display()))
}

fn file_sha256_uncached(path: &Path, label: &str) -> Result<String, String> {
    use std::io::Read as _;

    let mut file = fs::File::open(path)
        .map_err(|error| format!("open {label} {} for SHA-256: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read {label} {} for SHA-256: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn current_unix_ms(label: &str) -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| format!("system clock predates Unix epoch while trying to {label}"))?
        .as_millis()
        .try_into()
        .map_err(|_| format!("Unix timestamp does not fit u64 while trying to {label}"))
}

fn validate_native_admission_run_nonce(value: &str) -> Result<(), String> {
    if !(24..=128).contains(&value.len())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err("native admission run nonce must be 24..=128 URL-safe ASCII characters".into());
    }
    Ok(())
}

fn native_source_sha256() -> String {
    bytes_sha256(GEMMA4_MTP_SOURCE.as_bytes())
}

fn native_admission_command_output(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("run {program} for native-admission provenance: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed while collecting native-admission provenance: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|_| format!("{program} emitted non-UTF-8 provenance"))?
        .trim()
        .to_owned();
    if value.is_empty() {
        return Err(format!("{program} emitted empty provenance"));
    }
    Ok(value)
}

fn current_native_admission_platform() -> Result<NativeAdmissionPlatform, String> {
    let device = metal::Device::system_default()
        .ok_or_else(|| "no system-default Metal device is available".to_string())?;
    Ok(NativeAdmissionPlatform {
        os: std::env::consts::OS.to_owned(),
        os_version: native_admission_command_output("/usr/bin/sw_vers", &["-productVersion"])?,
        machine_arch: std::env::consts::ARCH.to_owned(),
        machine_model: native_admission_command_output("/usr/sbin/sysctl", &["-n", "hw.model"])?,
        metal_device_name: device.name().to_owned(),
    })
}

fn json_u32(root: &serde_json::Value, pointer: &str) -> Result<u32, String> {
    let value = root
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("assistant config {pointer} is missing or not u32"))?;
    u32::try_from(value).map_err(|_| format!("assistant config {pointer} exceeds u32"))
}

fn compare_pair_tokenizers(
    target_file: &camelid::gguf::GgufFile,
    assistant_tokenizer_path: &Path,
) -> Result<u64, String> {
    let target_pieces = target_file
        .metadata_array_strings("tokenizer.ggml.tokens")
        .map_err(|error| format!("read target tokenizer table: {error}"))?;
    let target = camelid::tokenizer::Tokenizer::from_gguf(target_file)
        .map_err(|error| format!("construct target tokenizer: {error}"))?;
    let assistant =
        tokenizers::Tokenizer::from_file(assistant_tokenizer_path).map_err(|error| {
            format!(
                "construct assistant tokenizer {}: {error}",
                assistant_tokenizer_path.display()
            )
        })?;
    let vocab = assistant.get_vocab(true);
    let mut assistant_pieces = vec![None::<String>; SHARED_VOCAB_SIZE as usize];
    for (piece, id) in vocab {
        let Some(slot) = assistant_pieces.get_mut(id as usize) else {
            return Err(format!(
                "assistant tokenizer contains out-of-range ID {id} for {piece:?}"
            ));
        };
        if let Some(previous) = slot.as_ref() {
            if previous.as_bytes() != piece.as_bytes() {
                return Err(format!(
                    "assistant tokenizer duplicate ID {id}: {previous:?} versus {piece:?}"
                ));
            }
        } else {
            *slot = Some(piece);
        }
    }
    if target_pieces.len() != SHARED_VOCAB_SIZE as usize {
        return Err(format!(
            "target tokenizer has {} IDs, expected {SHARED_VOCAB_SIZE}",
            target_pieces.len()
        ));
    }
    let mut mismatches = Vec::new();
    for (id, (target_piece, assistant_piece)) in
        target_pieces.iter().zip(&assistant_pieces).enumerate()
    {
        if assistant_piece
            .as_ref()
            .is_none_or(|piece| piece.as_bytes() != target_piece.as_bytes())
        {
            mismatches.push(id as u32);
        }
    }
    if !mismatches.is_empty() {
        return Err(format!(
            "normalized tokenizer mismatch_count={} first_ids={:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(32)]
        ));
    }
    for &(expected_id, piece) in PAIR_SPECIAL_SENTINELS {
        if target
            .tokens
            .get(expected_id as usize)
            .map(|token| token.text.as_str())
            != Some(piece)
            || assistant.id_to_token(expected_id).as_deref() != Some(piece)
        {
            return Err(format!(
                "special sentinel {expected_id} does not map exactly to {piece:?}"
            ));
        }
        let target_ids = target
            .encode(piece, false, true)
            .map_err(|error| format!("target special encode {piece:?}: {error}"))?;
        let assistant_ids = assistant
            .encode(piece, false)
            .map_err(|error| format!("assistant special encode {piece:?}: {error}"))?;
        if target_ids != [expected_id] || assistant_ids.get_ids() != [expected_id] {
            return Err(format!(
                "special encode {piece:?}: target={target_ids:?} assistant={:?}",
                assistant_ids.get_ids()
            ));
        }
        let target_text = target
            .decode(&[expected_id], false)
            .map_err(|error| format!("target special decode {expected_id}: {error}"))?;
        let assistant_text = assistant
            .decode(&[expected_id], false)
            .map_err(|error| format!("assistant special decode {expected_id}: {error}"))?;
        if target_text != piece || assistant_text != piece {
            return Err(format!(
                "special decode {expected_id}: target={target_text:?} assistant={assistant_text:?}"
            ));
        }
    }
    for probe in PAIR_TEXT_PROBES {
        let target_ids = target
            .encode(probe, false, true)
            .map_err(|error| format!("target probe encode {probe:?}: {error}"))?;
        let assistant_ids = assistant
            .encode(*probe, false)
            .map_err(|error| format!("assistant probe encode {probe:?}: {error}"))?;
        if target_ids != assistant_ids.get_ids() {
            return Err(format!(
                "probe tokenization mismatch for {probe:?}: target={target_ids:?} assistant={:?}",
                assistant_ids.get_ids()
            ));
        }
        let target_text = target
            .decode(&target_ids, false)
            .map_err(|error| format!("target probe decode {probe:?}: {error}"))?;
        let assistant_text = assistant
            .decode(assistant_ids.get_ids(), false)
            .map_err(|error| format!("assistant probe decode {probe:?}: {error}"))?;
        if target_text != assistant_text {
            return Err(format!(
                "probe decode mismatch for {probe:?}: target={target_text:?} assistant={assistant_text:?}"
            ));
        }
    }
    Ok(0)
}

fn establish_pairing_evidence(
    target_runtime: &TargetRuntimeConfig,
    assistant_path: &Path,
) -> Result<PairingEvidence, String> {
    use camelid::{
        ghost::GhostFile,
        model::{Gemma4Binding, LlamaModelConfig},
    };

    let source_path = std::env::var_os(TARGET_SOURCE_PATH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_TARGET_SOURCE_PATH));
    if !source_path.is_absolute() || !source_path.is_file() {
        return Err(format!(
            "{TARGET_SOURCE_PATH_ENV} must name the full official target: {}",
            source_path.display()
        ));
    }
    let source_bytes = std::fs::metadata(&source_path)
        .map_err(|error| format!("stat full target {}: {error}", source_path.display()))?
        .len();
    if source_bytes != OFFICIAL_TARGET_BYTES {
        return Err(format!(
            "full target has {source_bytes} bytes, expected {OFFICIAL_TARGET_BYTES}"
        ));
    }
    exact_cached_sha256(&source_path, OFFICIAL_TARGET_SHA256, "official target")?;

    let assistant_dir = assistant_path
        .parent()
        .ok_or_else(|| "assistant path has no artifact directory".to_string())?;
    let config_path = assistant_dir.join("config.json");
    let tokenizer_config_path = assistant_dir.join("tokenizer_config.json");
    let tokenizer_path = assistant_dir.join("tokenizer.json");
    let assistant_bytes = std::fs::metadata(assistant_path)
        .map_err(|error| format!("stat assistant {}: {error}", assistant_path.display()))?
        .len();
    if assistant_bytes != OFFICIAL_ASSISTANT_MODEL_BYTES {
        return Err(format!(
            "assistant has {assistant_bytes} bytes, expected {OFFICIAL_ASSISTANT_MODEL_BYTES}"
        ));
    }
    for (path, expected, label) in [
        (
            assistant_path,
            OFFICIAL_ASSISTANT_MODEL_SHA256,
            "assistant model",
        ),
        (
            &config_path,
            OFFICIAL_ASSISTANT_CONFIG_SHA256,
            "assistant config",
        ),
        (
            &tokenizer_config_path,
            OFFICIAL_ASSISTANT_TOKENIZER_CONFIG_SHA256,
            "assistant tokenizer config",
        ),
        (
            &tokenizer_path,
            OFFICIAL_ASSISTANT_TOKENIZER_SHA256,
            "assistant tokenizer",
        ),
    ] {
        exact_cached_sha256(path, expected, label)?;
    }

    let assistant_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .map_err(|error| format!("read {}: {error}", config_path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", config_path.display()))?;
    for (pointer, expected) in [
        ("/backbone_hidden_size", 2_816),
        ("/text_config/vocab_size", SHARED_VOCAB_SIZE as u32),
        (
            "/text_config/num_kv_shared_layers",
            ASSISTANT_SHARED_KV_LAYERS,
        ),
        ("/text_config/bos_token_id", 2),
        ("/text_config/eos_token_id", 1),
        ("/text_config/pad_token_id", 0),
    ] {
        let actual = json_u32(&assistant_config, pointer)?;
        if actual != expected {
            return Err(format!(
                "assistant config {pointer}={actual}, expected {expected}"
            ));
        }
    }
    if assistant_config
        .pointer("/model_type")
        .and_then(serde_json::Value::as_str)
        != Some("gemma4_assistant")
    {
        return Err("assistant model_type is not gemma4_assistant".into());
    }

    let source_file = camelid::gguf::read_metadata(&source_path)
        .map_err(|error| format!("read official target metadata: {error}"))?;
    let runtime_file = camelid::gguf::read_metadata(&target_runtime.runtime_gguf_path)
        .map_err(|error| format!("read staged target metadata: {error}"))?;
    if source_file.architecture() != Some("gemma4")
        || source_file.metadata_u32("gemma4.attention.shared_kv_layers")
            != Some(TARGET_SHARED_KV_LAYERS)
        || source_file.tensors != runtime_file.tensors
    {
        return Err("staged target metadata/tensor directory differs from official Gemma 4".into());
    }
    let source_config = LlamaModelConfig::from_gguf(&source_file)
        .map_err(|error| format!("parse official target config: {error}"))?;
    let runtime_config = LlamaModelConfig::from_gguf(&runtime_file)
        .map_err(|error| format!("parse staged target config: {error}"))?;
    if source_config != runtime_config {
        return Err("staged target config differs from official source".into());
    }
    let gemma4 = source_config
        .gemma4
        .as_ref()
        .ok_or_else(|| "official target lacks Gemma 4 metadata".to_string())?;
    if !gemma4.is_sliding_layer(TARGET_SHARED_KV_SLIDING_SOURCE_LAYER as usize)
        || gemma4.is_sliding_layer(TARGET_SHARED_KV_FULL_SOURCE_LAYER as usize)
    {
        return Err("target shared-KV source layers 28/29 have wrong attention flavor".into());
    }
    let source_binding = Gemma4Binding::bind(&source_file, &source_config)
        .map_err(|error| format!("bind official target: {error}"))?;
    let runtime_binding = Gemma4Binding::bind(&runtime_file, &runtime_config)
        .map_err(|error| format!("bind staged target: {error}"))?;
    let ghost = GhostFile::open(&target_runtime.cghost_path)
        .map_err(|error| format!("open staged cghost: {error}"))?;
    if !ghost.has_sampled_source_identity() {
        return Err("staged cghost has no sampled source identity".into());
    }
    ghost
        .validate_moe_source_identity(&source_path, &source_binding, 128)
        .map_err(|error| format!("cghost does not match official target: {error}"))?;
    ghost
        .validate_moe_source_identity(&target_runtime.runtime_gguf_path, &runtime_binding, 128)
        .map_err(|error| format!("cghost does not match staged target: {error}"))?;
    let runtime_pieces = runtime_file
        .metadata_array_strings("tokenizer.ggml.tokens")
        .map_err(|error| format!("read staged target tokenizer: {error}"))?;
    let source_pieces = source_file
        .metadata_array_strings("tokenizer.ggml.tokens")
        .map_err(|error| format!("read source target tokenizer: {error}"))?;
    if runtime_pieces != source_pieces {
        return Err("staged target tokenizer differs from official source".into());
    }
    let tokenizer_mismatch_count = compare_pair_tokenizers(&source_file, &tokenizer_path)?;

    let evidence = PairingEvidence {
        pair_gate_passed: true,
        target_repository: OFFICIAL_TARGET_REPOSITORY.into(),
        target_revision: OFFICIAL_TARGET_REVISION.into(),
        target_source_path: source_path,
        target_source_bytes: source_bytes,
        target_source_sha256: OFFICIAL_TARGET_SHA256.into(),
        target_source_matches_official: true,
        target_staged_runtime_path: target_runtime.runtime_gguf_path.clone(),
        target_staged_cghost_path: target_runtime.cghost_path.clone(),
        target_staged_identity_scheme: STAGED_TARGET_IDENTITY_SCHEME.into(),
        target_staged_metadata_matches_source: true,
        target_cghost_matches_official_source: true,
        target_cghost_matches_staged_runtime: true,
        assistant_repository: OFFICIAL_ASSISTANT_REPOSITORY.into(),
        assistant_revision: OFFICIAL_ASSISTANT_REVISION.into(),
        assistant_staged_model_path: assistant_path.to_path_buf(),
        assistant_staged_model_bytes: assistant_bytes,
        assistant_staged_model_sha256: OFFICIAL_ASSISTANT_MODEL_SHA256.into(),
        assistant_staged_config_path: config_path,
        assistant_staged_config_sha256: OFFICIAL_ASSISTANT_CONFIG_SHA256.into(),
        assistant_staged_tokenizer_config_path: tokenizer_config_path,
        assistant_staged_tokenizer_config_sha256: OFFICIAL_ASSISTANT_TOKENIZER_CONFIG_SHA256.into(),
        assistant_staged_tokenizer_path: tokenizer_path,
        assistant_staged_tokenizer_sha256: OFFICIAL_ASSISTANT_TOKENIZER_SHA256.into(),
        assistant_staged_files_match_official: true,
        shared_vocab_size: SHARED_VOCAB_SIZE,
        tokenizer_mismatch_count,
        target_shared_kv_layers: TARGET_SHARED_KV_LAYERS,
        assistant_shared_kv_layers: ASSISTANT_SHARED_KV_LAYERS,
        target_shared_kv_sliding_source_layer: TARGET_SHARED_KV_SLIDING_SOURCE_LAYER,
        target_shared_kv_full_source_layer: TARGET_SHARED_KV_FULL_SOURCE_LAYER,
    };
    evidence.validate(&assistant_path.to_path_buf())?;
    Ok(evidence)
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct AssistantMemory {
    /// Logical tensor bytes represented by the assistant checkpoint.
    model_bytes: u64,
    file_bytes: u64,
    mapped_bytes: u64,
    /// Directly observed from the native resident ledger. No environment
    /// setting is allowed to stand in for the actual mlock receipt.
    locked_bytes: u64,
    resident_bytes: u64,
    private_bytes: u64,
    /// Target-owned KV pages referenced by the assistant; never count these as
    /// assistant-private allocation.
    borrowed_target_kv_bytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct ProcessMemorySnapshot {
    physical_footprint_bytes: u64,
    rss_bytes: u64,
    peak_rss_bytes: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MemoryPressure {
    Normal,
    Warn,
    Critical,
    Unknown(u32),
}

impl MemoryPressure {
    fn is_normal(self) -> bool {
        self == Self::Normal
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SystemVmSnapshot {
    elapsed_ms: u64,
    page_size: u64,
    free_bytes: u64,
    active_bytes: u64,
    inactive_bytes: u64,
    /// Operational reclaimable-headroom proxy; zero swap-out remains the proof.
    reclaimable_headroom_bytes: u64,
    wired_bytes: u64,
    compressor_occupied_bytes: u64,
    compressed_logical_bytes: u64,
    pageins_bytes: u64,
    pageouts_bytes: u64,
    swapins_bytes: u64,
    swapouts_bytes: u64,
    swap_used_bytes: u64,
    swap_total_bytes: u64,
    pressure: MemoryPressure,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct MemorySnapshot {
    phase: String,
    process: ProcessMemorySnapshot,
    system: SystemVmSnapshot,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct LoadOnlyMemoryDelta {
    physical_footprint_bytes: i64,
    rss_bytes: i64,
    free_bytes: i64,
    compressor_occupied_bytes: i64,
    compressed_logical_bytes: i64,
    pageins_bytes: i64,
    pageouts_bytes: i64,
    swapins_bytes: i64,
    swapouts_bytes: i64,
    swap_used_bytes: i64,
}

impl LoadOnlyMemoryDelta {
    fn between(before: &MemorySnapshot, after: &MemorySnapshot) -> Self {
        Self {
            physical_footprint_bytes: signed_delta(
                after.process.physical_footprint_bytes,
                before.process.physical_footprint_bytes,
            ),
            rss_bytes: signed_delta(after.process.rss_bytes, before.process.rss_bytes),
            free_bytes: signed_delta(after.system.free_bytes, before.system.free_bytes),
            compressor_occupied_bytes: signed_delta(
                after.system.compressor_occupied_bytes,
                before.system.compressor_occupied_bytes,
            ),
            compressed_logical_bytes: signed_delta(
                after.system.compressed_logical_bytes,
                before.system.compressed_logical_bytes,
            ),
            pageins_bytes: signed_delta(after.system.pageins_bytes, before.system.pageins_bytes),
            pageouts_bytes: signed_delta(after.system.pageouts_bytes, before.system.pageouts_bytes),
            swapins_bytes: signed_delta(after.system.swapins_bytes, before.system.swapins_bytes),
            swapouts_bytes: signed_delta(after.system.swapouts_bytes, before.system.swapouts_bytes),
            swap_used_bytes: signed_delta(
                after.system.swap_used_bytes,
                before.system.swap_used_bytes,
            ),
        }
    }
}

/// Exact load-only receipt for the official assistant. This deliberately
/// mirrors the public native ledger instead of serializing a Debug string, so
/// a checkpoint remains machine-auditable if the native type later grows.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct LoadOnlyAssistantLedger {
    file_bytes: u64,
    mapped_bytes: u64,
    locked_bytes: u64,
    resident_pages: u64,
    total_pages: u64,
    payload_bytes: u64,
    decoded_norm_bytes: u64,
    fixed_scratch_bytes: u64,
    hash_us: u128,
    lock_and_residency_us: u128,
    pipeline_compile_us: u128,
    load_wall_us: u128,
}

impl From<camelid::metal::Gemma4MtpResidentLedger> for LoadOnlyAssistantLedger {
    fn from(value: camelid::metal::Gemma4MtpResidentLedger) -> Self {
        Self {
            file_bytes: value.file_bytes,
            mapped_bytes: value.mapped_bytes,
            locked_bytes: value.locked_bytes,
            resident_pages: value.resident_pages,
            total_pages: value.total_pages,
            payload_bytes: value.payload_bytes,
            decoded_norm_bytes: value.decoded_norm_bytes,
            fixed_scratch_bytes: value.fixed_scratch_bytes,
            hash_us: value.hash_us,
            lock_and_residency_us: value.lock_and_residency_us,
            pipeline_compile_us: value.pipeline_compile_us,
            load_wall_us: value.load_wall_us,
        }
    }
}

/// Stable serialized mirror of the test-only target allocation ledger. The
/// runtime type is write-only (`Serialize`); copying the public fields keeps
/// this checkpoint independently readable by offline analysis.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct LoadOnlyTargetLedger {
    target_gguf_logical_bytes: u64,
    cghost_logical_bytes: u64,
    mapping_page_size_bytes: u64,
    target_gguf_mapped_bytes: u64,
    target_gguf_total_pages: u64,
    target_gguf_resident_pages: u64,
    target_gguf_resident_bytes: u64,
    cghost_mapped_bytes: u64,
    cghost_total_pages: u64,
    cghost_resident_pages: u64,
    cghost_resident_bytes: u64,
    host_cache_budget_bytes: u64,
    host_cache_resident_bytes: u64,
    host_cache_explicitly_touched_bytes: u64,
    expert_layer_count: u64,
    expert_logical_slot_count: u64,
    expert_slot_count: u64,
    expert_slot_capacity_bytes: u64,
    expert_file_mapped_slot_count: u64,
    expert_file_mapped_address_span_bytes: u64,
    expert_table_directory_slot_count: u64,
    expert_table_directory_capacity_bytes: u64,
    expert_table_bound_active_slot_count: u64,
    expert_tables_compute_bound: bool,
    expert_slot_explicitly_touched_bytes: u64,
    overflow_slot_count: u64,
    overflow_capacity_bytes: u64,
    overflow_explicitly_touched_bytes: u64,
    victim_record_capacity: u64,
    victim_capacity_bytes: u64,
    victim_explicitly_touched_bytes: u64,
    planned_prewarm_records: u64,
    planned_prewarm_bytes: u64,
    touched_prewarm_records: u64,
    touched_prewarm_bytes: u64,
    common_wire_q4_payload_bytes: u64,
    common_wire_q4_allocation_bytes: u64,
    common_wire_q4_explicitly_touched_bytes: u64,
    common_wire_q4_source_page_window_bytes: u64,
    common_wire_q4_source_resident_before_bytes: u64,
    common_wire_q4_source_resident_after_bytes: u64,
    /// Cumulative mincore-confirmed transitions across advisories. Shared
    /// boundary pages may be counted more than once; this is not unique
    /// recovered residency.
    common_wire_q4_source_confirmed_discarded_bytes: u64,
    common_wire_q4_source_discard_advisory_calls: u64,
    common_wire_q4_source_final_cleanup_advisory_calls: u64,
    common_wire_q4_source_uncached_reads: bool,
    common_wire_q4_source_readahead_disabled: bool,
    common_resident_layer_aux_capacity_bytes: u64,
    common_resident_layer_aux_explicitly_touched_bytes: u64,
    common_resident_layer_aux_transient_peak_capacity_bytes: u64,
    common_resident_layer_aux_transient_peak_explicitly_touched_bytes: u64,
    common_kv_capacity_positions: u64,
    common_kv_bytes: u64,
    common_kv_explicitly_touched_bytes: u64,
    verifier_scratch_capacity_bytes: u64,
    verifier_scratch_explicitly_touched_bytes: u64,
    common_non_kv_scratch_capacity_bytes: u64,
    common_non_kv_scratch_explicitly_touched_bytes: u64,
    common_norm_router_aux_capacity_bytes: u64,
    common_norm_router_aux_explicitly_touched_bytes: u64,
    required_head_wire_payload_bytes: u64,
    required_head_page_window_bytes: u64,
    required_head_page_size_bytes: u64,
    required_head_total_pages: u64,
    required_head_resident_pages: u64,
    required_head_resident_bytes: u64,
    required_head_explicitly_touched_bytes: u64,
    required_head_scratch_capacity_bytes: u64,
    required_head_scratch_explicitly_touched_bytes: u64,
    background_readahead_started: bool,
    expert_slots_active: bool,
    common_kv_scratch_active: bool,
    arbitrary_slot_prewarm_skipped: bool,
    tied_head_active: bool,
    required_head_file_backed_no_copy: bool,
}

impl From<camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger> for LoadOnlyTargetLedger {
    fn from(value: camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger) -> Self {
        Self {
            target_gguf_logical_bytes: value.target_gguf_logical_bytes,
            cghost_logical_bytes: value.cghost_logical_bytes,
            mapping_page_size_bytes: value.mapping_page_size_bytes,
            target_gguf_mapped_bytes: value.target_gguf_mapped_bytes,
            target_gguf_total_pages: value.target_gguf_total_pages,
            target_gguf_resident_pages: value.target_gguf_resident_pages,
            target_gguf_resident_bytes: value.target_gguf_resident_bytes,
            cghost_mapped_bytes: value.cghost_mapped_bytes,
            cghost_total_pages: value.cghost_total_pages,
            cghost_resident_pages: value.cghost_resident_pages,
            cghost_resident_bytes: value.cghost_resident_bytes,
            host_cache_budget_bytes: value.host_cache_budget_bytes,
            host_cache_resident_bytes: value.host_cache_resident_bytes,
            host_cache_explicitly_touched_bytes: value.host_cache_explicitly_touched_bytes,
            expert_layer_count: value.expert_layer_count,
            expert_logical_slot_count: value.expert_logical_slot_count,
            expert_slot_count: value.expert_slot_count,
            expert_slot_capacity_bytes: value.expert_slot_capacity_bytes,
            expert_file_mapped_slot_count: value.expert_file_mapped_slot_count,
            expert_file_mapped_address_span_bytes: value
                .expert_file_mapped_address_span_bytes,
            expert_table_directory_slot_count: value.expert_table_directory_slot_count,
            expert_table_directory_capacity_bytes: value.expert_table_directory_capacity_bytes,
            expert_table_bound_active_slot_count: value.expert_table_bound_active_slot_count,
            expert_tables_compute_bound: value.expert_tables_compute_bound,
            expert_slot_explicitly_touched_bytes: value.expert_slot_explicitly_touched_bytes,
            overflow_slot_count: value.overflow_slot_count,
            overflow_capacity_bytes: value.overflow_capacity_bytes,
            overflow_explicitly_touched_bytes: value.overflow_explicitly_touched_bytes,
            victim_record_capacity: value.victim_record_capacity,
            victim_capacity_bytes: value.victim_capacity_bytes,
            victim_explicitly_touched_bytes: value.victim_explicitly_touched_bytes,
            planned_prewarm_records: value.planned_prewarm_records,
            planned_prewarm_bytes: value.planned_prewarm_bytes,
            touched_prewarm_records: value.touched_prewarm_records,
            touched_prewarm_bytes: value.touched_prewarm_bytes,
            common_wire_q4_payload_bytes: value.common_wire_q4_payload_bytes,
            common_wire_q4_allocation_bytes: value.common_wire_q4_allocation_bytes,
            common_wire_q4_explicitly_touched_bytes: value.common_wire_q4_explicitly_touched_bytes,
            common_wire_q4_source_page_window_bytes: value.common_wire_q4_source_page_window_bytes,
            common_wire_q4_source_resident_before_bytes: value
                .common_wire_q4_source_resident_before_bytes,
            common_wire_q4_source_resident_after_bytes: value
                .common_wire_q4_source_resident_after_bytes,
            common_wire_q4_source_confirmed_discarded_bytes: value
                .common_wire_q4_source_confirmed_discarded_bytes,
            common_wire_q4_source_discard_advisory_calls: value
                .common_wire_q4_source_discard_advisory_calls,
            common_wire_q4_source_final_cleanup_advisory_calls: value
                .common_wire_q4_source_final_cleanup_advisory_calls,
            common_wire_q4_source_uncached_reads: value.common_wire_q4_source_uncached_reads,
            common_wire_q4_source_readahead_disabled: value
                .common_wire_q4_source_readahead_disabled,
            common_resident_layer_aux_capacity_bytes: value
                .common_resident_layer_aux_capacity_bytes,
            common_resident_layer_aux_explicitly_touched_bytes: value
                .common_resident_layer_aux_explicitly_touched_bytes,
            common_resident_layer_aux_transient_peak_capacity_bytes: value
                .common_resident_layer_aux_transient_peak_capacity_bytes,
            common_resident_layer_aux_transient_peak_explicitly_touched_bytes: value
                .common_resident_layer_aux_transient_peak_explicitly_touched_bytes,
            common_kv_capacity_positions: value.common_kv_capacity_positions,
            common_kv_bytes: value.common_kv_bytes,
            common_kv_explicitly_touched_bytes: value.common_kv_explicitly_touched_bytes,
            verifier_scratch_capacity_bytes: value.verifier_scratch_capacity_bytes,
            verifier_scratch_explicitly_touched_bytes: value
                .verifier_scratch_explicitly_touched_bytes,
            common_non_kv_scratch_capacity_bytes: value.common_non_kv_scratch_capacity_bytes,
            common_non_kv_scratch_explicitly_touched_bytes: value
                .common_non_kv_scratch_explicitly_touched_bytes,
            common_norm_router_aux_capacity_bytes: value.common_norm_router_aux_capacity_bytes,
            common_norm_router_aux_explicitly_touched_bytes: value
                .common_norm_router_aux_explicitly_touched_bytes,
            required_head_wire_payload_bytes: value.required_head_wire_payload_bytes,
            required_head_page_window_bytes: value.required_head_page_window_bytes,
            required_head_page_size_bytes: value.required_head_page_size_bytes,
            required_head_total_pages: value.required_head_total_pages,
            required_head_resident_pages: value.required_head_resident_pages,
            required_head_resident_bytes: value.required_head_resident_bytes,
            required_head_explicitly_touched_bytes: value.required_head_explicitly_touched_bytes,
            required_head_scratch_capacity_bytes: value.required_head_scratch_capacity_bytes,
            required_head_scratch_explicitly_touched_bytes: value
                .required_head_scratch_explicitly_touched_bytes,
            background_readahead_started: value.background_readahead_started,
            expert_slots_active: value.expert_slots_active,
            common_kv_scratch_active: value.common_kv_scratch_active,
            arbitrary_slot_prewarm_skipped: value.arbitrary_slot_prewarm_skipped,
            tied_head_active: value.tied_head_active,
            required_head_file_backed_no_copy: value.required_head_file_backed_no_copy,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LoadOnlyTargetObservation {
    phase: String,
    ledger: LoadOnlyTargetLedger,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LoadOnlyPhaseCheckpoint {
    snapshot: MemorySnapshot,
    delta_from_clean: LoadOnlyMemoryDelta,
    delta_from_previous: LoadOnlyMemoryDelta,
    target_observation: Option<LoadOnlyTargetObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LoadOnlyProbeReport {
    schema_version: u32,
    process_id: u32,
    started_unix_ms: u64,
    completed_unix_ms: Option<u64>,
    assistant_path: PathBuf,
    target_runtime: TargetRuntimeConfig,
    forced_file_backed_no_copy_head: bool,
    assistant_ledger: Option<LoadOnlyAssistantLedger>,
    target_final_ledger: Option<LoadOnlyTargetLedger>,
    checkpoints: Vec<LoadOnlyPhaseCheckpoint>,
    monitor_samples: Vec<SystemVmSnapshot>,
    process_monitor_samples: Vec<ProcessMemorySnapshot>,
    monitor_kill_reason: Option<KillReason>,
    soak_seconds: u64,
    soak_started_elapsed_ms: Option<u64>,
    soak_monitor_start_sample: Option<usize>,
    soak_completed_seconds: u64,
    minimum_free_bytes: u64,
    minimum_reclaimable_headroom_bytes: u64,
    all_pressure_samples_normal: bool,
    soak_minimum_free_bytes: Option<u64>,
    soak_minimum_reclaimable_headroom_bytes: Option<u64>,
    soak_all_pressure_samples_normal: bool,
    soak_required_head_total_pages: Option<u64>,
    soak_min_required_head_resident_pages: Option<u64>,
    soak_all_required_head_pages_resident: bool,
    completed: bool,
    failure: Option<String>,
    assistant_warmups: u64,
    assistant_proposals: u64,
    tokenizer_calls: u64,
    target_prefills: u64,
    target_steps: u64,
    target_generations: u64,
    target_kv_borrows: u64,
}

fn load_only_operation_violation(report: &LoadOnlyProbeReport) -> Option<String> {
    let nonzero = [
        ("assistant_warmups", report.assistant_warmups),
        ("assistant_proposals", report.assistant_proposals),
        ("tokenizer_calls", report.tokenizer_calls),
        ("target_prefills", report.target_prefills),
        ("target_steps", report.target_steps),
        ("target_generations", report.target_generations),
        ("target_kv_borrows", report.target_kv_borrows),
    ]
    .into_iter()
    .filter(|(_, count)| *count != 0)
    .map(|(name, count)| format!("{name}={count}"))
    .collect::<Vec<_>>();
    (!nonzero.is_empty()).then(|| {
        format!(
            "load-only no-generation contract recorded forbidden operations: {}",
            nonzero.join(", ")
        )
    })
}

#[cfg(target_os = "macos")]
fn load_only_target_phase_name(
    phase: camelid::gemma4_runtime::Gemma4GhostLoadPhase,
) -> &'static str {
    use camelid::gemma4_runtime::Gemma4GhostLoadPhase;
    match phase {
        Gemma4GhostLoadPhase::TargetObjectMetadataReady => "target_object_metadata_ready",
        Gemma4GhostLoadPhase::CommonQ4WeightsAllocated => "common_q4_weights_allocated",
        Gemma4GhostLoadPhase::CommonResidentLayerAuxAllocated => {
            "common_resident_layer_aux_allocated"
        }
        Gemma4GhostLoadPhase::CommonKvScratchAllocated => "common_kv_scratch_allocated",
        Gemma4GhostLoadPhase::EmptyExpertSlotsAllocated => "empty_expert_slots_allocated",
        Gemma4GhostLoadPhase::AssistantResidencyBarrierComplete => {
            "assistant_residency_barrier_complete"
        }
        Gemma4GhostLoadPhase::RequiredTargetPagesHeadReady => "required_target_pages_head_ready",
        Gemma4GhostLoadPhase::ExpertSlotTablesComputeBound => "expert_slot_tables_compute_bound",
        Gemma4GhostLoadPhase::Complete => "target_load_complete",
    }
}

#[cfg(target_os = "macos")]
fn load_only_hybrid_expert_capacity_ready(
    ledger: &camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger,
) -> bool {
    // Serialized Gemma 4 routed-record geometry: 3,345,408-byte payload
    // rounded up to the existing 16 KiB slot boundary. The anonymous hot
    // tier owns exactly 48 records/layer while the clean mapped tier keeps
    // all 128 canonical expert IDs addressable.
    let stride = 3_358_720u64;
    ledger.expert_slots_active
        && ledger.expert_layer_count == 30
        && ledger.expert_logical_slot_count == ledger.expert_layer_count.saturating_mul(128)
        && ledger.expert_slot_count == ledger.expert_layer_count.saturating_mul(48)
        && ledger.expert_slot_capacity_bytes == ledger.expert_slot_count.saturating_mul(stride)
        && ledger.expert_file_mapped_slot_count
            == ledger.expert_layer_count.saturating_mul(128)
        && ledger.expert_file_mapped_address_span_bytes
            == ledger
                .expert_file_mapped_slot_count
                .saturating_mul(stride)
        && ledger.cghost_logical_bytes >= ledger.expert_file_mapped_address_span_bytes
        && ledger.cghost_mapped_bytes >= ledger.expert_file_mapped_address_span_bytes
        && ledger.expert_table_directory_slot_count
            == ledger.expert_layer_count.saturating_mul(48)
        && ledger.expert_table_directory_capacity_bytes
            == ledger
                .expert_table_directory_slot_count
                .saturating_mul(stride)
        && ledger.overflow_slot_count == 0
        && ledger.overflow_capacity_bytes == 0
        && ledger.victim_record_capacity == 0
        && ledger.victim_capacity_bytes == 0
        && ledger.host_cache_budget_bytes == 0
        && ledger.host_cache_resident_bytes == 0
        && ledger.host_cache_explicitly_touched_bytes == 0
        && ledger.expert_slot_explicitly_touched_bytes == 0
        && ledger.overflow_explicitly_touched_bytes == 0
        && ledger.victim_explicitly_touched_bytes == 0
        && ledger.planned_prewarm_records == 0
        && ledger.planned_prewarm_bytes == 0
        && ledger.touched_prewarm_records == 0
        && ledger.touched_prewarm_bytes == 0
        && ledger.arbitrary_slot_prewarm_skipped
}

#[cfg(target_os = "macos")]
fn load_only_target_phase_violation(
    phase: camelid::gemma4_runtime::Gemma4GhostLoadPhase,
    ledger: &camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger,
) -> Option<String> {
    use camelid::gemma4_runtime::Gemma4GhostLoadPhase;
    let phase_name = load_only_target_phase_name(phase);
    if ledger.background_readahead_started
        || ledger.host_cache_resident_bytes != 0
        || ledger.host_cache_explicitly_touched_bytes != 0
        || ledger.touched_prewarm_records != 0
        || ledger.touched_prewarm_bytes != 0
        || ledger.mapping_page_size_bytes == 0
        || ledger.target_gguf_mapped_bytes == 0
        || ledger.target_gguf_total_pages == 0
        || ledger.target_gguf_resident_pages > ledger.target_gguf_total_pages
        || ledger.cghost_mapped_bytes == 0
        || ledger.cghost_total_pages == 0
        || ledger.cghost_resident_pages > ledger.cghost_total_pages
    {
        return Some(format!(
            "{phase_name}: invalid mapping/readahead ledger {ledger:?}"
        ));
    }
    let common_weights_ready = || {
        ledger.common_wire_q4_payload_bytes > 0
            && ledger.common_wire_q4_allocation_bytes >= ledger.common_wire_q4_payload_bytes
            && ledger.common_wire_q4_explicitly_touched_bytes
                == ledger.common_wire_q4_allocation_bytes
            && ledger.common_wire_q4_source_page_window_bytes > 0
            && ledger.common_wire_q4_source_resident_after_bytes
                <= ledger.common_wire_q4_source_resident_before_bytes
            && ledger.common_wire_q4_source_discard_advisory_calls == 205
            && ledger.common_wire_q4_source_final_cleanup_advisory_calls <= 90
            && ledger.common_wire_q4_source_uncached_reads
            && ledger.common_wire_q4_source_readahead_disabled
    };
    let common_weights_absent = || {
        ledger.common_wire_q4_payload_bytes == 0
            && ledger.common_wire_q4_allocation_bytes == 0
            && ledger.common_wire_q4_explicitly_touched_bytes == 0
            && ledger.common_wire_q4_source_page_window_bytes == 0
            && ledger.common_wire_q4_source_resident_before_bytes == 0
            && ledger.common_wire_q4_source_resident_after_bytes == 0
            && ledger.common_wire_q4_source_confirmed_discarded_bytes == 0
            && ledger.common_wire_q4_source_discard_advisory_calls == 0
            && ledger.common_wire_q4_source_final_cleanup_advisory_calls == 0
            && !ledger.common_wire_q4_source_uncached_reads
            && !ledger.common_wire_q4_source_readahead_disabled
    };
    let common_scratch_ready = || {
        ledger.common_kv_scratch_active
            && ledger.common_kv_capacity_positions > 0
            && ledger.common_kv_bytes > 0
            && ledger.common_kv_explicitly_touched_bytes <= ledger.common_kv_bytes
            && ledger.verifier_scratch_capacity_bytes > 0
            && ledger.verifier_scratch_explicitly_touched_bytes
                <= ledger.verifier_scratch_capacity_bytes
            && ledger.common_non_kv_scratch_capacity_bytes > 0
            && ledger.common_non_kv_scratch_explicitly_touched_bytes
                <= ledger.common_non_kv_scratch_capacity_bytes
            && ledger.common_norm_router_aux_capacity_bytes > 0
            && ledger.common_norm_router_aux_explicitly_touched_bytes
                <= ledger.common_norm_router_aux_capacity_bytes
    };
    let common_resident_aux_ready = || {
        ledger.common_resident_layer_aux_capacity_bytes > 0
            && ledger.common_resident_layer_aux_explicitly_touched_bytes
                <= ledger.common_resident_layer_aux_capacity_bytes
            && ledger.common_resident_layer_aux_transient_peak_capacity_bytes > 0
            && ledger.common_resident_layer_aux_transient_peak_explicitly_touched_bytes
                <= ledger.common_resident_layer_aux_transient_peak_capacity_bytes
    };
    let common_resident_aux_absent = || {
        ledger.common_resident_layer_aux_capacity_bytes == 0
            && ledger.common_resident_layer_aux_explicitly_touched_bytes == 0
            && ledger.common_resident_layer_aux_transient_peak_capacity_bytes == 0
            && ledger.common_resident_layer_aux_transient_peak_explicitly_touched_bytes == 0
    };
    let common_scratch_absent = || {
        !ledger.common_kv_scratch_active
            && ledger.common_kv_capacity_positions == 0
            && ledger.common_kv_bytes == 0
            && ledger.common_kv_explicitly_touched_bytes == 0
            && ledger.verifier_scratch_capacity_bytes == 0
            && ledger.verifier_scratch_explicitly_touched_bytes == 0
            && ledger.common_non_kv_scratch_capacity_bytes == 0
            && ledger.common_non_kv_scratch_explicitly_touched_bytes == 0
            && ledger.common_norm_router_aux_capacity_bytes == 0
            && ledger.common_norm_router_aux_explicitly_touched_bytes == 0
    };
    let empty_slot_capacity_ready = || load_only_hybrid_expert_capacity_ready(ledger);
    let empty_slots_unbound = || {
        empty_slot_capacity_ready()
            && !ledger.expert_tables_compute_bound
            && ledger.expert_table_bound_active_slot_count == 0
    };
    let empty_slots_bound = || {
        empty_slot_capacity_ready()
            && ledger.expert_tables_compute_bound
            && ledger.expert_table_bound_active_slot_count
                == ledger.expert_layer_count.saturating_mul(8)
            && ledger.expert_table_bound_active_slot_count
                <= ledger.expert_file_mapped_slot_count
    };
    let required_head_ready = || {
        ledger.tied_head_active
            && ledger.required_head_file_backed_no_copy
            && ledger.required_head_wire_payload_bytes > 0
            && ledger.required_head_page_window_bytes >= ledger.required_head_wire_payload_bytes
            && ledger.required_head_page_size_bytes > 0
            && ledger.required_head_total_pages > 0
            && ledger.required_head_resident_pages == ledger.required_head_total_pages
            && ledger.required_head_resident_bytes > 0
            && ledger.required_head_explicitly_touched_bytes
                == ledger.required_head_page_window_bytes
            && ledger.required_head_scratch_capacity_bytes > 0
            && ledger.required_head_scratch_explicitly_touched_bytes
                <= ledger.required_head_scratch_capacity_bytes
    };
    let empty_slots_absent = || {
        !ledger.expert_slots_active
            && ledger.expert_layer_count == 0
            && ledger.expert_logical_slot_count == 0
            && ledger.expert_slot_count == 0
            && ledger.expert_slot_capacity_bytes == 0
            && ledger.expert_file_mapped_slot_count == 0
            && ledger.expert_file_mapped_address_span_bytes == 0
            && ledger.expert_table_directory_slot_count == 0
            && ledger.expert_table_directory_capacity_bytes == 0
            && ledger.expert_table_bound_active_slot_count == 0
            && !ledger.expert_tables_compute_bound
            && ledger.expert_slot_explicitly_touched_bytes == 0
            && ledger.overflow_slot_count == 0
            && ledger.overflow_capacity_bytes == 0
            && ledger.overflow_explicitly_touched_bytes == 0
            && ledger.victim_record_capacity == 0
            && ledger.victim_capacity_bytes == 0
            && ledger.victim_explicitly_touched_bytes == 0
            && ledger.planned_prewarm_records == 0
            && ledger.planned_prewarm_bytes == 0
            && ledger.touched_prewarm_records == 0
            && ledger.touched_prewarm_bytes == 0
    };
    let required_head_absent = || {
        !ledger.tied_head_active
            && !ledger.required_head_file_backed_no_copy
            && ledger.required_head_wire_payload_bytes == 0
            && ledger.required_head_page_window_bytes == 0
            && ledger.required_head_page_size_bytes == 0
            && ledger.required_head_total_pages == 0
            && ledger.required_head_resident_pages == 0
            && ledger.required_head_resident_bytes == 0
            && ledger.required_head_explicitly_touched_bytes == 0
            && ledger.required_head_scratch_capacity_bytes == 0
            && ledger.required_head_scratch_explicitly_touched_bytes == 0
    };
    let admitted = match phase {
        Gemma4GhostLoadPhase::TargetObjectMetadataReady => {
            common_weights_absent()
                && common_resident_aux_absent()
                && common_scratch_absent()
                && empty_slots_absent()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::CommonQ4WeightsAllocated => {
            common_weights_ready()
                && common_resident_aux_absent()
                && common_scratch_absent()
                && empty_slots_absent()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::CommonResidentLayerAuxAllocated => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_absent()
                && empty_slots_absent()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::CommonKvScratchAllocated => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_ready()
                && empty_slots_absent()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::EmptyExpertSlotsAllocated => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_ready()
                && empty_slots_unbound()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::AssistantResidencyBarrierComplete => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_ready()
                && empty_slots_unbound()
                && required_head_absent()
        }
        Gemma4GhostLoadPhase::RequiredTargetPagesHeadReady => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_ready()
                && empty_slots_unbound()
                && required_head_ready()
        }
        Gemma4GhostLoadPhase::ExpertSlotTablesComputeBound | Gemma4GhostLoadPhase::Complete => {
            common_weights_ready()
                && common_resident_aux_ready()
                && common_scratch_ready()
                && empty_slots_bound()
                && required_head_ready()
        }
    };
    (!admitted).then(|| format!("{phase_name}: incomplete allocation/touch ledger {ledger:?}"))
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
struct IncrementalMemory {
    physical_footprint_bytes: i64,
    rss_bytes: i64,
    compressor_occupied_bytes: i64,
    compressed_logical_bytes: i64,
    swap_used_bytes: i64,
}

impl IncrementalMemory {
    fn between(before: &MemorySnapshot, after: &MemorySnapshot) -> Self {
        Self {
            physical_footprint_bytes: signed_delta(
                after.process.physical_footprint_bytes,
                before.process.physical_footprint_bytes,
            ),
            rss_bytes: signed_delta(after.process.rss_bytes, before.process.rss_bytes),
            compressor_occupied_bytes: signed_delta(
                after.system.compressor_occupied_bytes,
                before.system.compressor_occupied_bytes,
            ),
            compressed_logical_bytes: signed_delta(
                after.system.compressed_logical_bytes,
                before.system.compressed_logical_bytes,
            ),
            swap_used_bytes: signed_delta(
                after.system.swap_used_bytes,
                before.system.swap_used_bytes,
            ),
        }
    }
}

fn signed_delta(after: u64, before: u64) -> i64 {
    let delta = i128::from(after) - i128::from(before);
    delta.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn peak_incremental_memory(
    baseline: &MemorySnapshot,
    snapshots: &[MemorySnapshot],
    monitor_samples: &[SystemVmSnapshot],
    process_monitor_samples: &[ProcessMemorySnapshot],
) -> IncrementalMemory {
    let mut physical = baseline.process.physical_footprint_bytes;
    let mut rss = baseline.process.rss_bytes;
    let mut compressor = baseline.system.compressor_occupied_bytes;
    let mut compressed = baseline.system.compressed_logical_bytes;
    let mut swap = baseline.system.swap_used_bytes;
    for sample in snapshots {
        physical = physical.max(sample.process.physical_footprint_bytes);
        rss = rss.max(sample.process.rss_bytes);
        compressor = compressor.max(sample.system.compressor_occupied_bytes);
        compressed = compressed.max(sample.system.compressed_logical_bytes);
        swap = swap.max(sample.system.swap_used_bytes);
    }
    for sample in monitor_samples {
        compressor = compressor.max(sample.compressor_occupied_bytes);
        compressed = compressed.max(sample.compressed_logical_bytes);
        swap = swap.max(sample.swap_used_bytes);
    }
    for sample in process_monitor_samples {
        physical = physical.max(sample.physical_footprint_bytes);
        rss = rss.max(sample.rss_bytes);
    }
    IncrementalMemory {
        physical_footprint_bytes: signed_delta(physical, baseline.process.physical_footprint_bytes),
        rss_bytes: signed_delta(rss, baseline.process.rss_bytes),
        compressor_occupied_bytes: signed_delta(
            compressor,
            baseline.system.compressor_occupied_bytes,
        ),
        compressed_logical_bytes: signed_delta(
            compressed,
            baseline.system.compressed_logical_bytes,
        ),
        swap_used_bytes: signed_delta(swap, baseline.system.swap_used_bytes),
    }
}

fn componentwise_max_delta(left: &mut IncrementalMemory, right: &IncrementalMemory) {
    left.physical_footprint_bytes = left
        .physical_footprint_bytes
        .max(right.physical_footprint_bytes);
    left.rss_bytes = left.rss_bytes.max(right.rss_bytes);
    left.compressor_occupied_bytes = left
        .compressor_occupied_bytes
        .max(right.compressor_occupied_bytes);
    left.compressed_logical_bytes = left
        .compressed_logical_bytes
        .max(right.compressed_logical_bytes);
    left.swap_used_bytes = left.swap_used_bytes.max(right.swap_used_bytes);
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct WindowKey {
    workload: Workload,
    repetition: u32,
    prefix_start: u64,
    prefix_end: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct LaneWindow {
    key: WindowKey,
    lane: Lane,
    verify_rounds: u64,
    emitted_tokens: u64,
    accepted_drafts: u64,
    draft_wall_us: u64,
    verify_wall_us: u64,
    total_wall_us: u64,
    assistant_invocations: u64,
    /// N and I must be byte-identical here, proving they used the same proposal
    /// stream/config while only assistant residency changed.
    proposal_trace_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct NimEconomics {
    key: WindowKey,
    verify_rounds: u64,
    emitted_tokens: u64,
    delta_a: i64,
    /// MTP drafting cost.
    d_us: u64,
    /// Assistant-residency pressure cost, isolated by I - N.
    p_us: u64,
    /// Additional target verification cost, positive half of M - I.
    v_plus_us: u64,
    /// Target verification time avoided, negative half of M - I.
    v_minus_us: u64,
    net_saved_us: i64,
    /// Additional accepted tokens per second of D + P + V+ cost.
    score: f64,
    n_wall_us: u64,
    i_wall_us: u64,
    m_wall_us: u64,
}

impl NimEconomics {
    fn derive(n: &LaneWindow, i: &LaneWindow, m: &LaneWindow) -> Result<Self, String> {
        if n.lane != Lane::NgramBaseline
            || i.lane != Lane::NgramAssistantIdle
            || m.lane != Lane::Mtp
        {
            return Err("N/I/M economics received lanes in the wrong roles".to_string());
        }
        if n.key != i.key || n.key != m.key {
            return Err("N/I/M economics requires an identical corresponding prefix".to_string());
        }
        if n.emitted_tokens != i.emitted_tokens || n.emitted_tokens != m.emitted_tokens {
            return Err("N/I/M economics requires equal emitted-token prefix budgets".to_string());
        }
        if n.proposal_trace_sha256 != i.proposal_trace_sha256 {
            return Err("N and I proposal traces/configs are not identical".to_string());
        }
        if n.assistant_invocations != 0 || i.assistant_invocations != 0 {
            return Err("N and I must never invoke the assistant".to_string());
        }
        if m.assistant_invocations == 0 {
            return Err("M must contain at least one assistant invocation".to_string());
        }
        if m.draft_wall_us == 0 {
            return Err("M assistant invocations require measured non-zero draft wall time".into());
        }

        let p_us = i.total_wall_us.saturating_sub(n.total_wall_us);
        let v_plus_us = m.verify_wall_us.saturating_sub(i.verify_wall_us);
        let v_minus_us = i.verify_wall_us.saturating_sub(m.verify_wall_us);
        let cost = m
            .draft_wall_us
            .saturating_add(p_us)
            .saturating_add(v_plus_us);
        let delta_a = m.accepted_drafts as i128 - n.accepted_drafts as i128;
        let delta_a = delta_a.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        let net_saved = i128::from(v_minus_us) - i128::from(cost);
        let net_saved_us = net_saved.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        debug_assert!(cost > 0);
        let score = delta_a as f64 * 1_000_000.0 / cost as f64;

        Ok(Self {
            key: n.key.clone(),
            verify_rounds: n.verify_rounds.max(i.verify_rounds).max(m.verify_rounds),
            emitted_tokens: n.emitted_tokens.max(i.emitted_tokens).max(m.emitted_tokens),
            delta_a,
            d_us: m.draft_wall_us,
            p_us,
            v_plus_us,
            v_minus_us,
            net_saved_us,
            score,
            n_wall_us: n.total_wall_us,
            i_wall_us: i.total_wall_us,
            m_wall_us: m.total_wall_us,
        })
    }

    fn evidence_ready(&self) -> bool {
        self.verify_rounds >= ECONOMICS_MIN_VERIFY_ROUNDS
            || self.emitted_tokens >= ECONOMICS_MIN_EMITTED_TOKENS
    }

    fn costs_at_least_savings(&self) -> bool {
        self.d_us
            .saturating_add(self.p_us)
            .saturating_add(self.v_plus_us)
            >= self.v_minus_us
    }

    fn no_gain_and_materially_slower(&self) -> bool {
        let reference = self.n_wall_us.min(self.i_wall_us);
        self.delta_a <= 0
            && reference > 0
            && self.m_wall_us as f64 >= reference as f64 * NO_GAIN_WALL_RATIO_LIMIT
    }
}

fn process_memory_snapshot() -> Result<ProcessMemorySnapshot, String> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut usage: libc::rusage_info_v2 = std::mem::zeroed();
        let usage_result = libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut usage as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
        );

        #[allow(deprecated)]
        let task = libc::mach_task_self();
        let mut basic: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
            / std::mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let task_result = libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            &mut basic as *mut _ as *mut libc::integer_t,
            &mut count,
        );

        let mut rusage: libc::rusage = std::mem::zeroed();
        let rusage_result = libc::getrusage(libc::RUSAGE_SELF, &mut rusage);
        if usage_result != 0
            || task_result != libc::KERN_SUCCESS
            || rusage_result != 0
            || usage.ri_phys_footprint == 0
            || basic.resident_size == 0
        {
            return Err("macOS process-memory telemetry is unavailable".to_string());
        }
        Ok(ProcessMemorySnapshot {
            physical_footprint_bytes: usage.ri_phys_footprint,
            rss_bytes: basic.resident_size,
            peak_rss_bytes: rusage.ru_maxrss.max(0) as u64,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("the MTP memory-pressure experiment is currently scoped to macOS".to_string())
    }
}

/// Parent-side process sample used while a lane child may be blocked in model
/// hashing, mmap/mlock, or assistant initialization. Unlike task_info/getrusage,
/// proc_pid_rusage can inspect the child without cooperation, so a SIGKILL still
/// leaves a last-known RSS/footprint receipt.
fn process_memory_snapshot_for_pid(pid: u32) -> Result<ProcessMemorySnapshot, String> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut usage: libc::rusage_info_v2 = std::mem::zeroed();
        let result = libc::proc_pid_rusage(
            pid as libc::c_int,
            libc::RUSAGE_INFO_V2,
            &mut usage as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
        );
        if result != 0 || usage.ri_phys_footprint == 0 || usage.ri_resident_size == 0 {
            return Err(format!(
                "macOS process-memory telemetry is unavailable for child PID {pid}"
            ));
        }
        Ok(ProcessMemorySnapshot {
            physical_footprint_bytes: usage.ri_phys_footprint,
            rss_bytes: usage.ri_resident_size,
            // macOS does not expose another process's ru_maxrss through this
            // API. The 1 Hz series itself supplies the observed RSS peak.
            peak_rss_bytes: usage.ri_resident_size,
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = pid;
        Err("the MTP memory-pressure experiment is currently scoped to macOS".to_string())
    }
}

fn system_vm_snapshot(elapsed_ms: u64) -> Result<SystemVmSnapshot, String> {
    #[cfg(target_os = "macos")]
    {
        let vm_output = Command::new("vm_stat")
            .output()
            .map_err(|error| format!("run vm_stat: {error}"))?;
        if !vm_output.status.success() {
            return Err(format!("vm_stat exited with {}", vm_output.status));
        }
        let vm_text = String::from_utf8(vm_output.stdout)
            .map_err(|_| "vm_stat output is not UTF-8".to_string())?;
        let swap_output = Command::new("sysctl")
            .args(["-n", "vm.swapusage"])
            .output()
            .map_err(|error| format!("read vm.swapusage: {error}"))?;
        if !swap_output.status.success() {
            return Err(format!("vm.swapusage exited with {}", swap_output.status));
        }
        let swap_text = String::from_utf8(swap_output.stdout)
            .map_err(|_| "vm.swapusage output is not UTF-8".to_string())?;
        let pressure = macos_memory_pressure()?;
        parse_system_vm_snapshot(&vm_text, &swap_text, pressure, elapsed_ms)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = elapsed_ms;
        Err("the MTP memory-pressure experiment is currently scoped to macOS".to_string())
    }
}

fn capture_memory_snapshot(phase: &str, elapsed_ms: u64) -> Result<MemorySnapshot, String> {
    Ok(MemorySnapshot {
        phase: phase.to_string(),
        process: process_memory_snapshot()?,
        system: system_vm_snapshot(elapsed_ms)?,
    })
}

#[cfg(target_os = "macos")]
fn macos_memory_pressure() -> Result<MemoryPressure, String> {
    let name = CString::new("kern.memorystatus_vm_pressure_level")
        .map_err(|_| "invalid memory-pressure sysctl name".to_string())?;
    let mut level = 0u32;
    let mut size = std::mem::size_of::<u32>();
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut level as *mut u32 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 || size != std::mem::size_of::<u32>() {
        return Err("kern.memorystatus_vm_pressure_level is unavailable".to_string());
    }
    Ok(match level {
        1 => MemoryPressure::Normal,
        2 => MemoryPressure::Warn,
        4 => MemoryPressure::Critical,
        other => MemoryPressure::Unknown(other),
    })
}

fn parse_system_vm_snapshot(
    vm_stat: &str,
    swapusage: &str,
    pressure: MemoryPressure,
    elapsed_ms: u64,
) -> Result<SystemVmSnapshot, String> {
    let page_size = parse_vm_page_size(vm_stat)?;
    let pages = parse_vm_stat_counters(vm_stat);
    for required in [
        "Pages free",
        "Pages inactive",
        "Pages occupied by compressor",
        "Pages stored in compressor",
        "Swapins",
        "Swapouts",
    ] {
        if !pages.contains_key(required) {
            return Err(format!("vm_stat omitted required counter {required:?}"));
        }
    }
    let bytes = |key: &str| {
        pages
            .get(key)
            .copied()
            .unwrap_or(0)
            .saturating_mul(page_size)
    };
    let free_bytes = bytes("Pages free");
    let inactive_bytes = bytes("Pages inactive");
    Ok(SystemVmSnapshot {
        elapsed_ms,
        page_size,
        free_bytes,
        active_bytes: bytes("Pages active"),
        inactive_bytes,
        reclaimable_headroom_bytes: free_bytes.saturating_add(inactive_bytes),
        wired_bytes: bytes("Pages wired down"),
        compressor_occupied_bytes: bytes("Pages occupied by compressor"),
        compressed_logical_bytes: bytes("Pages stored in compressor"),
        pageins_bytes: bytes("Pageins"),
        pageouts_bytes: bytes("Pageouts"),
        swapins_bytes: bytes("Swapins"),
        swapouts_bytes: bytes("Swapouts"),
        swap_used_bytes: parse_named_size(swapusage, "used")?,
        swap_total_bytes: parse_named_size(swapusage, "total")?,
        pressure,
    })
}

fn parse_vm_page_size(text: &str) -> Result<u64, String> {
    let marker = "page size of ";
    let start = text
        .find(marker)
        .ok_or_else(|| "vm_stat omitted page size".to_string())?
        + marker.len();
    let digits: String = text[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    let page_size = digits
        .parse::<u64>()
        .map_err(|_| "vm_stat page size is invalid".to_string())?;
    if page_size == 0 {
        return Err("vm_stat page size is zero".to_string());
    }
    Ok(page_size)
}

fn parse_vm_stat_counters(text: &str) -> BTreeMap<String, u64> {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.trim().split_once(':')?;
            let value = value.trim().trim_end_matches('.').parse::<u64>().ok()?;
            Some((key.to_string(), value))
        })
        .collect()
}

fn parse_named_size(text: &str, name: &str) -> Result<u64, String> {
    let words: Vec<_> = text.split_whitespace().collect();
    let position = words
        .iter()
        .position(|word| word.trim_end_matches(':') == name)
        .ok_or_else(|| format!("vm.swapusage omitted {name}"))?;
    let value = words
        .iter()
        .skip(position + 1)
        .find(|word| **word != "=")
        .ok_or_else(|| format!("vm.swapusage omitted {name} value"))?;
    parse_byte_size(value)
}

fn parse_byte_size(raw: &str) -> Result<u64, String> {
    let raw = raw.trim().trim_end_matches(',');
    let split = raw
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(raw.len());
    let number = raw[..split]
        .parse::<f64>()
        .map_err(|_| format!("invalid byte size {raw:?}"))?;
    let unit = raw[split..].trim().to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "" | "B" => 1.0,
        "K" | "KB" | "KIB" => 1024.0,
        "M" | "MB" | "MIB" => 1024.0 * 1024.0,
        "G" | "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" | "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(format!("unsupported byte-size unit in {raw:?}")),
    };
    if !number.is_finite() || number < 0.0 {
        return Err(format!("invalid byte size {raw:?}"));
    }
    Ok((number * multiplier).round().min(u64::MAX as f64) as u64)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct PilotComparison {
    workload: Workload,
    emitted_tokens: u64,
    n_median_wall_us: u64,
    m_median_wall_us: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum KillReason {
    SwapActivityThreeConsecutiveSamples,
    RollingSwapTraffic64MiB,
    SwapUsedGrew256MiBAndStillIncreasing,
    MemoryPressureTwoConsecutiveSamples,
    EconomicsLostTwoConsecutiveWindows,
    NoAcceptanceGainAndMtpAtLeastFivePercentSlower,
    PilotMedianMtpNotFaster,
    ChildTimeout,
    ChildFailure(String),
    TelemetryUnavailable(String),
}

#[derive(Debug)]
struct KillPolicyState {
    baseline: SystemVmSnapshot,
    previous: SystemVmSnapshot,
    rolling: VecDeque<SystemVmSnapshot>,
    swap_activity_streak: u32,
    pressure_streak: u32,
    economics_loss_streak: u32,
    killed: Option<KillReason>,
}

impl KillPolicyState {
    fn new(baseline: SystemVmSnapshot) -> Self {
        let mut rolling = VecDeque::new();
        rolling.push_back(baseline.clone());
        Self {
            previous: baseline.clone(),
            baseline,
            rolling,
            swap_activity_streak: 0,
            pressure_streak: 0,
            economics_loss_streak: 0,
            killed: None,
        }
    }

    fn observe_memory(&mut self, sample: SystemVmSnapshot) -> Option<KillReason> {
        if self.killed.is_some() {
            return self.killed.clone();
        }

        let swap_advanced = sample.swapins_bytes > self.previous.swapins_bytes
            || sample.swapouts_bytes > self.previous.swapouts_bytes;
        self.swap_activity_streak = if swap_advanced {
            self.swap_activity_streak.saturating_add(1)
        } else {
            0
        };
        if self.swap_activity_streak >= SWAP_ACTIVITY_STREAK_LIMIT {
            return self.kill(KillReason::SwapActivityThreeConsecutiveSamples);
        }

        self.pressure_streak = if sample.pressure.is_normal() {
            0
        } else {
            self.pressure_streak.saturating_add(1)
        };
        if self.pressure_streak >= PRESSURE_STREAK_LIMIT {
            return self.kill(KillReason::MemoryPressureTwoConsecutiveSamples);
        }

        let swap_used_growth = sample
            .swap_used_bytes
            .saturating_sub(self.baseline.swap_used_bytes);
        if swap_used_growth >= SWAP_USED_GROWTH_LIMIT_BYTES
            && sample.swap_used_bytes > self.previous.swap_used_bytes
        {
            return self.kill(KillReason::SwapUsedGrew256MiBAndStillIncreasing);
        }

        self.rolling.push_back(sample.clone());
        while self
            .rolling
            .front()
            .is_some_and(|oldest| oldest.elapsed_ms + ROLLING_SWAP_WINDOW_MS < sample.elapsed_ms)
        {
            self.rolling.pop_front();
        }
        if let Some(oldest) = self.rolling.front() {
            let traffic = sample
                .swapins_bytes
                .saturating_sub(oldest.swapins_bytes)
                .saturating_add(sample.swapouts_bytes.saturating_sub(oldest.swapouts_bytes));
            if traffic >= ROLLING_SWAP_TRAFFIC_LIMIT_BYTES {
                return self.kill(KillReason::RollingSwapTraffic64MiB);
            }
        }

        self.previous = sample;
        None
    }

    fn observe_economics(&mut self, economics: &NimEconomics) -> Option<KillReason> {
        if self.killed.is_some() {
            return self.killed.clone();
        }
        if !economics.evidence_ready() {
            return None;
        }
        if economics.no_gain_and_materially_slower() {
            return self.kill(KillReason::NoAcceptanceGainAndMtpAtLeastFivePercentSlower);
        }
        self.economics_loss_streak = if economics.costs_at_least_savings() {
            self.economics_loss_streak.saturating_add(1)
        } else {
            0
        };
        if self.economics_loss_streak >= ECONOMICS_LOSS_STREAK_LIMIT {
            return self.kill(KillReason::EconomicsLostTwoConsecutiveWindows);
        }
        None
    }

    fn observe_pilot(&mut self, pilot: &PilotComparison) -> Option<KillReason> {
        if self.killed.is_some() {
            return self.killed.clone();
        }
        if pilot.emitted_tokens >= PILOT_TOKENS && pilot.m_median_wall_us >= pilot.n_median_wall_us
        {
            return self.kill(KillReason::PilotMedianMtpNotFaster);
        }
        None
    }

    fn kill(&mut self, reason: KillReason) -> Option<KillReason> {
        self.killed = Some(reason.clone());
        Some(reason)
    }
}

enum MonitorEvent {
    Stop,
}

#[derive(Clone)]
struct ExperimentControl {
    abort: Arc<AtomicBool>,
    events: mpsc::Sender<MonitorEvent>,
}

impl ExperimentControl {
    fn should_abort(&self) -> bool {
        self.abort.load(Ordering::Acquire)
    }
}

#[derive(Clone, Default)]
struct MonitorResults {
    samples: Vec<SystemVmSnapshot>,
    process_samples: Vec<ProcessMemorySnapshot>,
    kill_reason: Option<KillReason>,
}

struct MonitorGuard {
    control: ExperimentControl,
    results: Arc<Mutex<MonitorResults>>,
    join: Option<thread::JoinHandle<()>>,
}

impl MonitorGuard {
    fn start(baseline: MemorySnapshot) -> Self {
        let (sender, receiver) = mpsc::channel();
        let abort = Arc::new(AtomicBool::new(false));
        let results = Arc::new(Mutex::new(MonitorResults {
            samples: vec![baseline.system.clone()],
            process_samples: vec![baseline.process.clone()],
            kill_reason: None,
        }));
        let thread_abort = Arc::clone(&abort);
        let thread_results = Arc::clone(&results);
        let join = thread::spawn(move || {
            let started = Instant::now();
            let mut next_sample = started + MONITOR_PERIOD;
            let mut policy = KillPolicyState::new(baseline.system);
            loop {
                let timeout = next_sample.saturating_duration_since(Instant::now());
                match receiver.recv_timeout(timeout) {
                    Ok(MonitorEvent::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        let elapsed_ms =
                            started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                        match system_vm_snapshot(elapsed_ms) {
                            Ok(sample) => {
                                let process_sample = match process_memory_snapshot() {
                                    Ok(sample) => sample,
                                    Err(error) => {
                                        record_kill(
                                            &thread_abort,
                                            &thread_results,
                                            KillReason::TelemetryUnavailable(error),
                                        );
                                        break;
                                    }
                                };
                                if let Ok(mut results) = thread_results.lock() {
                                    results.samples.push(sample.clone());
                                    results.process_samples.push(process_sample);
                                }
                                if let Some(reason) = policy.observe_memory(sample) {
                                    record_kill(&thread_abort, &thread_results, reason);
                                    break;
                                }
                            }
                            Err(error) => {
                                record_kill(
                                    &thread_abort,
                                    &thread_results,
                                    KillReason::TelemetryUnavailable(error),
                                );
                                break;
                            }
                        }
                        next_sample += MONITOR_PERIOD;
                        if next_sample <= Instant::now() {
                            next_sample = Instant::now() + MONITOR_PERIOD;
                        }
                    }
                }
            }
        });
        Self {
            control: ExperimentControl {
                abort,
                events: sender,
            },
            results,
            join: Some(join),
        }
    }

    fn control(&self) -> ExperimentControl {
        self.control.clone()
    }

    fn finish(mut self) -> MonitorResults {
        let _ = self.control.events.send(MonitorEvent::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        let snapshot = self
            .results
            .lock()
            .map(|results| results.clone())
            .unwrap_or_default();
        snapshot
    }
}

impl Drop for MonitorGuard {
    fn drop(&mut self) {
        let _ = self.control.events.send(MonitorEvent::Stop);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn record_kill(abort: &AtomicBool, results: &Mutex<MonitorResults>, reason: KillReason) {
    abort.store(true, Ordering::Release);
    if let Ok(mut results) = results.lock() {
        results.kill_reason = Some(reason);
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX)) as u64
}

fn load_only_monitor_results(monitor: &MonitorGuard) -> MonitorResults {
    monitor
        .results
        .lock()
        .map(|results| results.clone())
        .unwrap_or_default()
}

fn load_only_safety_violation(
    baseline: &SystemVmSnapshot,
    checkpoints: &[LoadOnlyPhaseCheckpoint],
    monitor_samples: &[SystemVmSnapshot],
) -> Option<String> {
    let check = |phase: &str, sample: &SystemVmSnapshot| {
        if sample.swapouts_bytes > baseline.swapouts_bytes {
            return Some(format!(
                "{phase}: swapouts increased by {} bytes",
                sample.swapouts_bytes - baseline.swapouts_bytes
            ));
        }
        if !sample.pressure.is_normal() {
            return Some(format!(
                "{phase}: memory pressure is {:?}, not normal",
                sample.pressure
            ));
        }
        if sample.reclaimable_headroom_bytes < LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES {
            return Some(format!(
                "{phase}: reclaimable headroom {} is below the {} byte floor",
                sample.reclaimable_headroom_bytes, LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES
            ));
        }
        None
    };

    for checkpoint in checkpoints {
        if let Some(reason) = check(&checkpoint.snapshot.phase, &checkpoint.snapshot.system) {
            return Some(reason);
        }
    }
    for (index, sample) in monitor_samples.iter().enumerate() {
        if let Some(reason) = check(&format!("monitor[{index}]"), sample) {
            return Some(reason);
        }
    }
    None
}

fn refresh_load_only_monitor_receipt(report: &mut LoadOnlyProbeReport, monitor: &MonitorGuard) {
    let results = load_only_monitor_results(monitor);
    report.monitor_samples = results.samples;
    report.process_monitor_samples = results.process_samples;
    report.monitor_kill_reason = results.kill_reason;
    for (index, sample) in report.monitor_samples.iter().enumerate() {
        report.minimum_free_bytes = report.minimum_free_bytes.min(sample.free_bytes);
        report.minimum_reclaimable_headroom_bytes = report
            .minimum_reclaimable_headroom_bytes
            .min(sample.reclaimable_headroom_bytes);
        report.all_pressure_samples_normal &= sample.pressure.is_normal();
        if report
            .soak_monitor_start_sample
            .is_some_and(|started| index >= started)
        {
            report.soak_minimum_free_bytes = Some(
                report
                    .soak_minimum_free_bytes
                    .map_or(sample.free_bytes, |minimum| minimum.min(sample.free_bytes)),
            );
            report.soak_minimum_reclaimable_headroom_bytes = Some(
                report
                    .soak_minimum_reclaimable_headroom_bytes
                    .map_or(sample.reclaimable_headroom_bytes, |minimum| {
                        minimum.min(sample.reclaimable_headroom_bytes)
                    }),
            );
            report.soak_all_pressure_samples_normal &= sample.pressure.is_normal();
        }
    }
}

fn checkpoint_load_only_probe(
    report_path: &Path,
    report: &mut LoadOnlyProbeReport,
    baseline: &SystemVmSnapshot,
    monitor: &MonitorGuard,
    snapshot: MemorySnapshot,
    target_observation: Option<LoadOnlyTargetObservation>,
) -> Result<(), String> {
    let delta_from_clean = report
        .checkpoints
        .first()
        .map(|clean| LoadOnlyMemoryDelta::between(&clean.snapshot, &snapshot))
        .unwrap_or_default();
    let delta_from_previous = report
        .checkpoints
        .last()
        .map(|previous| LoadOnlyMemoryDelta::between(&previous.snapshot, &snapshot))
        .unwrap_or_default();
    report.minimum_free_bytes = report.minimum_free_bytes.min(snapshot.system.free_bytes);
    report.minimum_reclaimable_headroom_bytes = report
        .minimum_reclaimable_headroom_bytes
        .min(snapshot.system.reclaimable_headroom_bytes);
    report.all_pressure_samples_normal &= snapshot.system.pressure.is_normal();
    if snapshot.phase == "both_models_retained_soak_start" {
        report.soak_started_elapsed_ms = Some(snapshot.system.elapsed_ms);
        // Monitor samples and synchronous checkpoints use independent monotonic
        // origins. Pin an index before the next refresh rather than comparing
        // their incomparable elapsed_ms values. Including a concurrently
        // arrived boundary sample is intentionally conservative.
        report.soak_monitor_start_sample = Some(report.monitor_samples.len());
    }
    if snapshot.phase.starts_with("both_models_retained_soak") {
        report.soak_minimum_free_bytes = Some(
            report
                .soak_minimum_free_bytes
                .map_or(snapshot.system.free_bytes, |minimum| {
                    minimum.min(snapshot.system.free_bytes)
                }),
        );
        report.soak_minimum_reclaimable_headroom_bytes = Some(
            report
                .soak_minimum_reclaimable_headroom_bytes
                .map_or(snapshot.system.reclaimable_headroom_bytes, |minimum| {
                    minimum.min(snapshot.system.reclaimable_headroom_bytes)
                }),
        );
        report.soak_all_pressure_samples_normal &= snapshot.system.pressure.is_normal();
    }
    report.checkpoints.push(LoadOnlyPhaseCheckpoint {
        snapshot,
        delta_from_clean,
        delta_from_previous,
        target_observation,
    });
    refresh_load_only_monitor_receipt(report, monitor);
    if let Some(reason) = report.monitor_kill_reason.as_ref() {
        let reason = format!("load-only monitor stopped the probe: {reason:?}");
        report.failure = Some(reason.clone());
        atomic_write_json(report_path, report)?;
        return Err(reason);
    }
    if let Some(reason) =
        load_only_safety_violation(baseline, &report.checkpoints, &report.monitor_samples)
    {
        report.failure = Some(reason.clone());
        atomic_write_json(report_path, report)?;
        return Err(reason);
    }
    atomic_write_json(report_path, report)
}

fn capture_load_only_checkpoint(
    report_path: &Path,
    report: &mut LoadOnlyProbeReport,
    baseline: &SystemVmSnapshot,
    monitor: &MonitorGuard,
    started: Instant,
    phase: impl Into<String>,
    target_observation: Option<LoadOnlyTargetObservation>,
) -> Result<(), String> {
    let phase = phase.into();
    let snapshot = capture_memory_snapshot(
        &phase,
        started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
    )?;
    checkpoint_load_only_probe(
        report_path,
        report,
        baseline,
        monitor,
        snapshot,
        target_observation,
    )
}

#[cfg(target_os = "macos")]
fn capture_load_only_residency_checkpoint(
    report_path: &Path,
    report: &mut LoadOnlyProbeReport,
    baseline: &SystemVmSnapshot,
    monitor: &MonitorGuard,
    started: Instant,
    target: &camelid::gemma4_runtime::Gemma4GhostLoadOnlyRuntime,
    phase: impl Into<String>,
) -> Result<(), String> {
    let phase = phase.into();
    let ledger = target
        .refreshed_residency_ledger()
        .map_err(|error| format!("{phase}: refresh nonfaulting target residency: {error}"))?;
    if report
        .soak_required_head_total_pages
        .is_some_and(|total| total != ledger.required_head_total_pages)
    {
        return Err(format!(
            "{phase}: required head page geometry changed from {:?} to {}",
            report.soak_required_head_total_pages, ledger.required_head_total_pages
        ));
    }
    report.soak_required_head_total_pages = Some(ledger.required_head_total_pages);
    report.soak_min_required_head_resident_pages = Some(
        report
            .soak_min_required_head_resident_pages
            .map_or(ledger.required_head_resident_pages, |minimum| {
                minimum.min(ledger.required_head_resident_pages)
            }),
    );
    report.soak_all_required_head_pages_resident &=
        ledger.required_head_resident_pages == ledger.required_head_total_pages;
    report.target_final_ledger = Some(ledger.into());
    capture_load_only_checkpoint(
        report_path,
        report,
        baseline,
        monitor,
        started,
        phase.clone(),
        Some(LoadOnlyTargetObservation {
            phase: phase.clone(),
            ledger: ledger.into(),
        }),
    )?;
    if ledger.required_head_resident_pages < ledger.required_head_total_pages {
        return Err(format!(
            "{phase}: required no-copy head lost residency: {}/{} pages",
            ledger.required_head_resident_pages, ledger.required_head_total_pages
        ));
    }
    if let Some(error) = load_only_target_phase_violation(
        camelid::gemma4_runtime::Gemma4GhostLoadPhase::Complete,
        &ledger,
    ) {
        return Err(format!(
            "{phase}: refreshed residency contract failed: {error}"
        ));
    }
    Ok(())
}

fn persist_load_only_failure(
    report_path: &Path,
    report: &mut LoadOnlyProbeReport,
    monitor: &MonitorGuard,
    error: impl Into<String>,
) -> String {
    let error = error.into();
    let error = report.failure.as_ref().map_or(error.clone(), |primary| {
        format!("{primary}; follow-on failure: {error}")
    });
    report.failure = Some(error.clone());
    refresh_load_only_monitor_receipt(report, monitor);
    if let Err(write_error) = atomic_write_json(report_path, report) {
        return format!("{error}; additionally failed to checkpoint receipt: {write_error}");
    }
    error
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum RunPhase {
    Pilot,
    Matrix,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LaneRun {
    run_id: String,
    phase: RunPhase,
    workload: Workload,
    lane: Lane,
    repetition: u32,
    requested_output_tokens: u64,
    /// PID of the one-run child. A successful report must not reuse a child
    /// across lanes/workloads, keeping process memory attribution unambiguous.
    child_process_id: u32,
    driver: LaneDriver,
    /// Measured-generation calls only; assistant loading/warmup is excluded.
    assistant_invocations: u64,
    /// SHA-256 of the canonical per-round proposal receipt: round index,
    /// requested K, offered token count, and offered token IDs. Timing, lane,
    /// and assistant residency are deliberately excluded so N and I compare
    /// byte-for-byte without retaining a potentially large token ledger.
    proposal_trace_sha256: String,
    /// Tokens committed after full-target verification. All primary lanes must
    /// produce the same stream; no assistant-only token belongs here.
    output_token_ids: Vec<u32>,
    terminal_target_tokens: u64,
    generation_wall_us: u64,
    completed: bool,
    rounds: Vec<RoundTelemetry>,
    metrics: RunMetrics,
    /// Post-generation target residency captured after the final measured
    /// verifier call and before the child releases either model.
    routed_experts_after_generation:
        Option<camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot>,
}

fn validate_routed_expert_snapshot(
    label: &str,
    snapshot: &camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
) -> Result<(), String> {
    use camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot;

    if snapshot.layer_count == 0
        || snapshot.per_layer.len() != snapshot.layer_count as usize
        || snapshot.slot_record_bytes == 0
        || snapshot.slot_stride_bytes < snapshot.slot_record_bytes
    {
        return Err(format!("{label} has invalid routed-expert geometry"));
    }
    let mut capacity = 0u64;
    let mut physical_budget = 0u64;
    let mut file_mapped_slots = 0u64;
    let mut file_mapped_span = 0u64;
    let mut occupied = 0u64;
    let mut aggregate = Gemma4RoutedExpertSlotStatsSnapshot::default();
    for (index, layer) in snapshot.per_layer.iter().enumerate() {
        if layer.layer_index as usize != index
            || layer.base_slot_capacity == 0
            || layer.physical_base_slot_budget > layer.base_slot_capacity
            || layer.file_mapped_addressable_slots > layer.base_slot_capacity
            || (layer.physical_base_slot_budget == 0
                && layer.file_mapped_addressable_slots == 0)
            || layer.physical_base_slot_budget_bytes
                != layer
                    .physical_base_slot_budget
                    .saturating_mul(snapshot.slot_stride_bytes)
            || layer.file_mapped_address_span_bytes
                != layer
                    .file_mapped_addressable_slots
                    .saturating_mul(snapshot.slot_stride_bytes)
            || layer.occupied_base_slots > layer.physical_base_slot_budget
            || layer.occupied_base_payload_bytes
                != layer
                    .occupied_base_slots
                    .saturating_mul(snapshot.slot_record_bytes)
            || layer.occupied_base_touched_bytes
                != layer
                    .occupied_base_slots
                    .saturating_mul(snapshot.slot_stride_bytes)
        {
            return Err(format!(
                "{label} layer {index} has inconsistent routed-expert occupancy"
            ));
        }
        capacity = capacity.saturating_add(layer.base_slot_capacity);
        physical_budget = physical_budget.saturating_add(layer.physical_base_slot_budget);
        file_mapped_slots =
            file_mapped_slots.saturating_add(layer.file_mapped_addressable_slots);
        file_mapped_span =
            file_mapped_span.saturating_add(layer.file_mapped_address_span_bytes);
        occupied = occupied.saturating_add(layer.occupied_base_slots);
        aggregate.route_lookups = aggregate
            .route_lookups
            .saturating_add(layer.slot_stats.route_lookups);
        aggregate.hits = aggregate.hits.saturating_add(layer.slot_stats.hits);
        aggregate.misses = aggregate.misses.saturating_add(layer.slot_stats.misses);
        aggregate.evictions = aggregate
            .evictions
            .saturating_add(layer.slot_stats.evictions);
        aggregate.host_fills = aggregate
            .host_fills
            .saturating_add(layer.slot_stats.host_fills);
        aggregate.prewarm_copies = aggregate
            .prewarm_copies
            .saturating_add(layer.slot_stats.prewarm_copies);
        aggregate.direct_reads = aggregate
            .direct_reads
            .saturating_add(layer.slot_stats.direct_reads);
        aggregate.direct_read_bytes = aggregate
            .direct_read_bytes
            .saturating_add(layer.slot_stats.direct_read_bytes);
        aggregate.direct_read_failures = aggregate
            .direct_read_failures
            .saturating_add(layer.slot_stats.direct_read_failures);
    }
    if capacity != snapshot.base_slot_capacity
        || physical_budget != snapshot.physical_base_slot_budget
        || file_mapped_slots != snapshot.file_mapped_addressable_slots
        || file_mapped_span != snapshot.file_mapped_address_span_bytes
        || occupied != snapshot.occupied_base_slots
        || snapshot.base_slot_capacity_bytes != capacity.saturating_mul(snapshot.slot_stride_bytes)
        || snapshot.physical_base_slot_budget_bytes
            != physical_budget.saturating_mul(snapshot.slot_stride_bytes)
        || snapshot.occupied_base_payload_bytes
            != occupied.saturating_mul(snapshot.slot_record_bytes)
        || snapshot.occupied_base_touched_bytes
            != occupied.saturating_mul(snapshot.slot_stride_bytes)
        || aggregate != snapshot.aggregate_slot_stats
        || aggregate.route_lookups != aggregate.hits.saturating_add(aggregate.misses)
        || aggregate.direct_read_bytes
            != aggregate
                .direct_reads
                .saturating_mul(snapshot.slot_record_bytes)
        || snapshot.cumulative_expert_payload_read_bytes
            != aggregate
                .direct_read_bytes
                .saturating_add(snapshot.cumulative_chained_demand_read_bytes)
        || snapshot.cumulative_chained_demand_read_bytes
            != snapshot
                .cumulative_chained_demand_loads
                .saturating_mul(snapshot.slot_record_bytes)
        || snapshot.host_cache_resident_bytes > snapshot.host_cache_budget_bytes
        || (snapshot.host_cache_budget_bytes == 0
            && (snapshot.host_cache_hits != 0
                || snapshot.host_cache_misses != 0
                || snapshot.host_cache_evictions != 0
                || snapshot.host_cache_bytes_read != 0
                || snapshot.host_cache_resident_experts != 0
                || snapshot.host_cache_resident_bytes != 0))
    {
        return Err(format!(
            "{label} aggregate routed-expert accounting is inconsistent"
        ));
    }
    if snapshot.interval_routed_expert_union_scope
        != "since_latest_explicit_telemetry_interval_begin_or_runtime_load"
        || snapshot.interval_routed_expert_ids_per_layer.len() != snapshot.layer_count as usize
        || snapshot.interval_routed_unique_per_layer.len() != snapshot.layer_count as usize
    {
        return Err(format!("{label} has an invalid routed-identity interval"));
    }
    let mut interval_unique_sum = 0u32;
    let mut interval_unique_max = 0u32;
    for (layer_index, (identities, count)) in snapshot
        .interval_routed_expert_ids_per_layer
        .iter()
        .zip(&snapshot.interval_routed_unique_per_layer)
        .enumerate()
    {
        if identities.len() != usize::from(*count)
            || identities.iter().any(|expert| *expert >= 128)
            || identities.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(format!(
                "{label} layer {layer_index} has an invalid routed-identity union"
            ));
        }
        interval_unique_sum = interval_unique_sum.saturating_add(u32::from(*count));
        interval_unique_max = interval_unique_max.max(u32::from(*count));
    }
    if interval_unique_sum != snapshot.interval_routed_unique_experts_sum
        || interval_unique_max != snapshot.interval_routed_unique_experts_max
    {
        return Err(format!(
            "{label} has inconsistent routed-identity union totals"
        ));
    }
    if snapshot.last_chained_ledger_scope
        != "latest_completed_chained_attempt_only_not_generation_maximum"
        || snapshot.last_chained_unique_per_layer.len() != snapshot.layer_count as usize
        || snapshot.last_chained_hot_bound_per_layer.len() != snapshot.layer_count as usize
        || snapshot.last_chained_mapped_bound_per_layer.len() != snapshot.layer_count as usize
        || snapshot.last_chained_demand_loads > snapshot.cumulative_chained_demand_loads
        || snapshot.last_chained_demand_read_bytes > snapshot.cumulative_chained_demand_read_bytes
        || u64::from(snapshot.last_chained_slot_hits) > aggregate.hits
        || u64::from(snapshot.last_chained_slot_misses) > aggregate.misses
        || u64::from(snapshot.last_chained_slot_evictions) > aggregate.evictions
    {
        return Err(format!("{label} has an invalid last-chained scope"));
    }
    let unique_sum = snapshot
        .last_chained_unique_per_layer
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>();
    let unique_max = snapshot
        .last_chained_unique_per_layer
        .iter()
        .copied()
        .max()
        .map(u32::from)
        .unwrap_or(0);
    let hot_bound_sum = snapshot
        .last_chained_hot_bound_per_layer
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>();
    let mapped_bound_sum = snapshot
        .last_chained_mapped_bound_per_layer
        .iter()
        .map(|value| u32::from(*value))
        .sum::<u32>();
    let hybrid_mapped_backing = file_mapped_slots > 0 && physical_budget < capacity;
    let tier_counts_fit_unique = snapshot
        .last_chained_unique_per_layer
        .iter()
        .zip(&snapshot.last_chained_hot_bound_per_layer)
        .zip(&snapshot.last_chained_mapped_bound_per_layer)
        .all(|((unique, hot), mapped)| {
            u32::from(*hot).saturating_add(u32::from(*mapped)) <= u32::from(*unique)
        });
    let unique_counts_fit_geometry = snapshot
        .last_chained_unique_per_layer
        .iter()
        .zip(&snapshot.per_layer)
        .all(|(unique, layer)| u64::from(*unique) <= layer.base_slot_capacity);
    let tier_counts_fit_geometry = snapshot
        .last_chained_hot_bound_per_layer
        .iter()
        .zip(&snapshot.last_chained_mapped_bound_per_layer)
        .zip(&snapshot.per_layer)
        .all(|((hot, mapped), layer)| {
            u64::from(*hot) <= layer.physical_base_slot_budget
                && u64::from(*mapped) <= layer.file_mapped_addressable_slots
        });
    if hot_bound_sum != snapshot.last_chained_hot_bound_records
        || mapped_bound_sum != snapshot.last_chained_mapped_bound_records
        || !tier_counts_fit_unique
        || !unique_counts_fit_geometry
        || !tier_counts_fit_geometry
        || (file_mapped_slots == 0 && (hot_bound_sum != 0 || mapped_bound_sum != 0))
        || snapshot.last_chained_demand_read_bytes
            != snapshot
                .last_chained_demand_loads
                .saturating_mul(snapshot.slot_record_bytes)
        || (hybrid_mapped_backing
            && (snapshot
                .last_chained_slot_hits
                .saturating_add(snapshot.last_chained_slot_misses)
                > unique_sum
                || snapshot.last_chained_slot_evictions > snapshot.last_chained_slot_misses
                || snapshot.last_chained_demand_loads
                    > u64::from(snapshot.last_chained_slot_misses)
                || snapshot.last_chained_demand_loads > u64::from(mapped_bound_sum)
                || snapshot.last_chained_overflow_slots != 0
                || snapshot.last_chained_overflow_bytes != 0
                || snapshot.last_chained_overflow_layers != 0
                || snapshot.last_chained_overflow_experts != 0
                || snapshot.last_chained_victim_hits != 0
                || snapshot.last_chained_victim_salvage_copies != 0))
    {
        return Err(format!(
            "{label} has inconsistent chained hot/mapped tier totals"
        ));
    }
    if snapshot.last_chained_round_available {
        let tier_partition_is_exact = snapshot
            .last_chained_unique_per_layer
            .iter()
            .zip(&snapshot.last_chained_hot_bound_per_layer)
            .zip(&snapshot.last_chained_mapped_bound_per_layer)
            .all(|((unique, hot), mapped)| {
                u32::from(*hot).saturating_add(u32::from(*mapped)) == u32::from(*unique)
            });
        if snapshot.last_chained_round_sequence == 0
            || !snapshot.last_chained_k.is_some_and(|value| value > 0)
            || unique_sum != snapshot.last_chained_unique_experts_sum
            || unique_max != snapshot.last_chained_unique_experts_max
            || snapshot.last_chained_overflow_experts > unique_sum
            || (snapshot.last_chained_round_succeeded
                && file_mapped_slots > 0
                && (!tier_partition_is_exact
                    || hot_bound_sum.saturating_add(mapped_bound_sum) != unique_sum
                    || (hybrid_mapped_backing
                        && (snapshot.last_chained_slot_hits != hot_bound_sum
                            || snapshot.last_chained_slot_misses != mapped_bound_sum))
                    || snapshot.last_chained_overflow_slots != 0
                    || snapshot.last_chained_overflow_bytes != 0
                    || snapshot.last_chained_overflow_layers != 0
                    || snapshot.last_chained_overflow_experts != 0))
            || (snapshot.last_chained_round_succeeded
                && (snapshot.last_chained_selected_experts_dropped != 0
                    || snapshot.last_chained_missing_expert_failclose != 0
                    || snapshot.last_chained_slot_capacity_overflow != 0))
        {
            return Err(format!(
                "{label} latest chained-attempt ledger failed consistency/correctness checks"
            ));
        }
    } else if snapshot.last_chained_round_succeeded
        || snapshot.last_chained_round_sequence != 0
        || snapshot.last_chained_k.is_some()
        || unique_sum != 0
        || snapshot.last_chained_unique_experts_sum != 0
        || snapshot.last_chained_unique_experts_max != 0
        || hot_bound_sum != 0
        || mapped_bound_sum != 0
        || snapshot.last_chained_demand_loads != 0
        || snapshot.last_chained_demand_read_bytes != 0
        || snapshot.last_chained_slot_hits != 0
        || snapshot.last_chained_slot_misses != 0
        || snapshot.last_chained_slot_evictions != 0
        || snapshot.last_chained_overflow_slots != 0
        || snapshot.last_chained_overflow_bytes != 0
        || snapshot.last_chained_overflow_layers != 0
        || snapshot.last_chained_overflow_experts != 0
        || snapshot.last_chained_victim_hits != 0
        || snapshot.last_chained_victim_salvage_copies != 0
        || snapshot.last_chained_selected_experts_dropped != 0
        || snapshot.last_chained_missing_expert_failclose != 0
        || snapshot.last_chained_slot_capacity_overflow != 0
        || snapshot.cumulative_chained_demand_loads != 0
        || snapshot.cumulative_chained_demand_read_bytes != 0
    {
        return Err(format!(
            "{label} reports chained facts without an available chained round"
        ));
    }
    Ok(())
}

fn validate_exact_target_hybrid_experts(
    snapshot: &camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
) -> Result<(), String> {
    const LAYERS: u64 = 30;
    const CANONICAL_SLOTS_PER_LAYER: u64 = 128;
    const HOT_SLOTS_PER_LAYER: u64 = 48;
    const RECORD_BYTES: u64 = 3_345_408;
    const STRIDE_BYTES: u64 = 3_358_720;
    if snapshot.per_layer.len() != LAYERS as usize
        || snapshot.last_chained_unique_per_layer.len() != LAYERS as usize
        || snapshot.last_chained_hot_bound_per_layer.len() != LAYERS as usize
        || snapshot.last_chained_mapped_bound_per_layer.len() != LAYERS as usize
        || snapshot.per_layer.iter().any(|layer| {
            layer.base_slot_capacity != CANONICAL_SLOTS_PER_LAYER
                || layer.physical_base_slot_budget != HOT_SLOTS_PER_LAYER
                || layer.physical_base_slot_budget_bytes
                    != HOT_SLOTS_PER_LAYER.saturating_mul(STRIDE_BYTES)
                || layer.file_mapped_addressable_slots != CANONICAL_SLOTS_PER_LAYER
                || layer.file_mapped_address_span_bytes
                    != CANONICAL_SLOTS_PER_LAYER.saturating_mul(STRIDE_BYTES)
                || layer.occupied_base_slots > HOT_SLOTS_PER_LAYER
        })
    {
        return Err(
            "target routed-expert receipt did not preserve 128 canonical / 48 anonymous-hot / 128 mapped-cold records per layer".into(),
        );
    }
    if u64::from(snapshot.layer_count) != LAYERS
        || snapshot.slot_record_bytes != RECORD_BYTES
        || snapshot.slot_stride_bytes != STRIDE_BYTES
        || snapshot.base_slot_capacity != LAYERS.saturating_mul(CANONICAL_SLOTS_PER_LAYER)
        || snapshot.physical_base_slot_budget != LAYERS.saturating_mul(HOT_SLOTS_PER_LAYER)
        || snapshot.file_mapped_addressable_slots
            != LAYERS.saturating_mul(CANONICAL_SLOTS_PER_LAYER)
        || snapshot.base_slot_capacity_bytes
            != snapshot
                .base_slot_capacity
                .saturating_mul(STRIDE_BYTES)
        || snapshot.physical_base_slot_budget_bytes
            != snapshot
                .physical_base_slot_budget
                .saturating_mul(STRIDE_BYTES)
        || snapshot.file_mapped_address_span_bytes
            != snapshot
                .file_mapped_addressable_slots
                .saturating_mul(STRIDE_BYTES)
        || snapshot.occupied_base_slots > snapshot.physical_base_slot_budget
        || snapshot.host_cache_budget_bytes != 0
        || snapshot.host_cache_resident_experts != 0
        || snapshot.host_cache_resident_bytes != 0
        || snapshot.host_cache_hits != 0
        || snapshot.host_cache_misses != 0
        || snapshot.host_cache_evictions != 0
        || snapshot.host_cache_bytes_read != 0
        || snapshot.aggregate_slot_stats.host_fills != 0
        || snapshot.aggregate_slot_stats.prewarm_copies != 0
        || snapshot.aggregate_slot_stats.direct_read_failures != 0
        || snapshot.last_chained_overflow_slots != 0
        || snapshot.last_chained_overflow_bytes != 0
        || snapshot.last_chained_overflow_layers != 0
        || snapshot.last_chained_overflow_experts != 0
        || snapshot.last_chained_victim_hits != 0
        || snapshot.last_chained_victim_salvage_copies != 0
        || snapshot.last_chained_selected_experts_dropped != 0
        || snapshot.last_chained_missing_expert_failclose != 0
        || snapshot.last_chained_slot_capacity_overflow != 0
    {
        return Err(
            "target routed-expert aggregate hybrid capacity/accounting receipt is inconsistent"
                .into(),
        );
    }
    if snapshot.last_chained_round_available && snapshot.last_chained_round_succeeded {
        let expected_promotion_loads = snapshot
            .last_chained_hot_bound_per_layer
            .iter()
            .zip(&snapshot.last_chained_mapped_bound_per_layer)
            .zip(&snapshot.per_layer)
            .map(|((hot, mapped), layer)| {
                u64::from(*mapped).min(
                    layer
                        .physical_base_slot_budget
                        .saturating_sub(u64::from(*hot)),
                )
            })
            .sum::<u64>();
        if snapshot
            .last_chained_slot_hits
            .saturating_add(snapshot.last_chained_slot_misses)
            != snapshot.last_chained_unique_experts_sum
            || snapshot.last_chained_slot_hits != snapshot.last_chained_hot_bound_records
            || snapshot.last_chained_slot_misses != snapshot.last_chained_mapped_bound_records
            || snapshot.last_chained_demand_loads != expected_promotion_loads
            || snapshot.last_chained_demand_read_bytes
                != expected_promotion_loads.saturating_mul(RECORD_BYTES)
            || u64::from(snapshot.last_chained_slot_evictions) > expected_promotion_loads
        {
            return Err(
                "target hybrid chained selection/refill ledger is not an exact unique-expert/promotion partition"
                    .into(),
            );
        }
    }
    Ok(())
}

fn routed_slot_stats_are_monotonic(
    before: &camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot,
    after: &camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot,
) -> bool {
    after.route_lookups >= before.route_lookups
        && after.hits >= before.hits
        && after.misses >= before.misses
        && after.evictions >= before.evictions
        && after.host_fills >= before.host_fills
        && after.prewarm_copies >= before.prewarm_copies
        && after.direct_reads >= before.direct_reads
        && after.direct_read_bytes >= before.direct_read_bytes
        && after.direct_read_failures >= before.direct_read_failures
}

fn validate_routed_expert_transition(
    before: &camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
    after: &camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
) -> Result<(), String> {
    validate_routed_expert_snapshot("post-target-warm snapshot", before)?;
    validate_routed_expert_snapshot("post-generation snapshot", after)?;
    let hybrid_generation = after.file_mapped_addressable_slots > 0
        && after.physical_base_slot_budget < after.base_slot_capacity;
    if hybrid_generation {
        validate_exact_target_hybrid_experts(before)?;
        validate_exact_target_hybrid_experts(after)?;
    }
    let hybrid_generation_accounting_ready = !hybrid_generation
        || (after.aggregate_slot_stats.hits > before.aggregate_slot_stats.hits
            && after.aggregate_slot_stats.misses > before.aggregate_slot_stats.misses
            && after.aggregate_slot_stats.host_fills == 0
            && after.aggregate_slot_stats.prewarm_copies == 0
            && after.aggregate_slot_stats.direct_read_failures == 0);
    if !hybrid_generation_accounting_ready {
        return Err(
            "hybrid generation did not prove positive hot/mapped selection deltas or reported a forbidden host/prewarm/read-failure event".into(),
        );
    }
    if before.layer_count != after.layer_count
        || before.slot_record_bytes != after.slot_record_bytes
        || before.slot_stride_bytes != after.slot_stride_bytes
        || before.base_slot_capacity != after.base_slot_capacity
        || before.base_slot_capacity_bytes != after.base_slot_capacity_bytes
        || before.physical_base_slot_budget != after.physical_base_slot_budget
        || before.physical_base_slot_budget_bytes != after.physical_base_slot_budget_bytes
        || before.file_mapped_addressable_slots != after.file_mapped_addressable_slots
        || before.file_mapped_address_span_bytes != after.file_mapped_address_span_bytes
        || after.occupied_base_slots < before.occupied_base_slots
        || after.cumulative_chained_demand_loads < before.cumulative_chained_demand_loads
        || after.cumulative_chained_demand_read_bytes < before.cumulative_chained_demand_read_bytes
        || after.cumulative_expert_payload_read_bytes < before.cumulative_expert_payload_read_bytes
        || after.host_cache_hits < before.host_cache_hits
        || after.host_cache_misses < before.host_cache_misses
        || after.host_cache_evictions < before.host_cache_evictions
        || after.host_cache_bytes_read < before.host_cache_bytes_read
        || after.interval_routed_expert_union_epoch
            != before.interval_routed_expert_union_epoch.saturating_add(1)
        || after.last_chained_round_sequence < before.last_chained_round_sequence
        || !routed_slot_stats_are_monotonic(
            &before.aggregate_slot_stats,
            &after.aggregate_slot_stats,
        )
        || before
            .per_layer
            .iter()
            .zip(&after.per_layer)
            .any(|(before, after)| {
                before.base_slot_capacity != after.base_slot_capacity
                    || before.physical_base_slot_budget != after.physical_base_slot_budget
                    || before.physical_base_slot_budget_bytes
                        != after.physical_base_slot_budget_bytes
                    || before.file_mapped_addressable_slots
                        != after.file_mapped_addressable_slots
                    || before.file_mapped_address_span_bytes
                        != after.file_mapped_address_span_bytes
                    || after.occupied_base_slots < before.occupied_base_slots
                    || !routed_slot_stats_are_monotonic(&before.slot_stats, &after.slot_stats)
            })
    {
        return Err("routed-expert counters regressed between target warm and generation".into());
    }
    Ok(())
}

impl LaneRun {
    fn validate(&self, expected_plan: &LaneExecutionPlan) -> Result<(), String> {
        if self.run_id.is_empty() {
            return Err("lane run is missing its unique run ID".into());
        }
        if self.requested_output_tokens == 0 {
            return Err("lane run requested a zero-token output budget".into());
        }
        if self.child_process_id == 0 {
            return Err("lane run is missing its isolated child PID".into());
        }
        if self.driver != expected_plan.driver {
            return Err(format!(
                "{:?} run used {:?}, expected {:?}",
                self.lane, self.driver, expected_plan.driver
            ));
        }
        if !is_sha256_hex(&self.proposal_trace_sha256) {
            return Err("lane run proposal trace is not a lowercase SHA-256 digest".into());
        }
        if self.output_token_ids.len() as u64 != self.metrics.visible_output_tokens {
            return Err(format!(
                "lane run retained {} output IDs but reports {} visible target tokens",
                self.output_token_ids.len(),
                self.metrics.visible_output_tokens
            ));
        }
        if self.terminal_target_tokens != self.metrics.terminal_target_tokens
            || self.terminal_target_tokens > 1
        {
            return Err(format!(
                "lane run terminal-token receipt is inconsistent: run={} metrics={}",
                self.terminal_target_tokens, self.metrics.terminal_target_tokens
            ));
        }
        if (self.completed && self.generation_wall_us == 0)
            || self.generation_wall_us < self.metrics.total_wall_us
        {
            return Err(format!(
                "lane run generation wall {}us is shorter than round ledger {}us",
                self.generation_wall_us, self.metrics.total_wall_us
            ));
        }
        match self.lane {
            Lane::Mtp if self.completed && self.assistant_invocations == 0 => {
                return Err("M run never invoked the assistant".into());
            }
            Lane::NgramBaseline | Lane::NgramAssistantIdle | Lane::NgramSeedOffDiagnostic
                if self.assistant_invocations != 0 =>
            {
                return Err(format!(
                    "{:?} run invoked the assistant {} times",
                    self.lane, self.assistant_invocations
                ));
            }
            _ => {}
        }
        if self
            .rounds
            .iter()
            .enumerate()
            .any(|(index, round)| round.budget_truncated && index + 1 != self.rounds.len())
        {
            return Err("only the final verifier round may be budget-truncated".into());
        }
        if let Some(snapshot) = self.routed_experts_after_generation.as_ref() {
            validate_routed_expert_snapshot("lane-run post-generation snapshot", snapshot)?;
        } else if self.completed {
            return Err("completed lane run omitted post-generation expert residency".into());
        }
        Ok(())
    }
}

struct ProposalTraceHasher(Sha256);

impl ProposalTraceHasher {
    fn new(
        workload: Workload,
        prompt_tokens: &[u32],
        proposal_environment: &BTreeMap<String, String>,
    ) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"camelid-gemma4-proposal-trace-v1\0");
        digest.update(workload.key().as_bytes());
        digest.update([0]);
        digest.update((prompt_tokens.len() as u64).to_le_bytes());
        for token in prompt_tokens {
            digest.update(token.to_le_bytes());
        }
        for (key, value) in proposal_environment {
            digest.update((key.len() as u64).to_le_bytes());
            digest.update(key.as_bytes());
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value.as_bytes());
        }
        Self(digest)
    }

    fn record(&mut self, round_index: usize, requested_k: usize, proposals: &[u32]) {
        self.0.update((round_index as u64).to_le_bytes());
        self.0.update((requested_k as u64).to_le_bytes());
        self.0.update((proposals.len() as u64).to_le_bytes());
        for token in proposals {
            self.0.update(token.to_le_bytes());
        }
    }

    fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

fn usize_as_u32(label: &str, value: usize) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{label}={value} exceeds u32"))
}

fn visible_committed_tokens(committed: &[u32], eot: &[u32]) -> usize {
    committed
        .iter()
        .take_while(|token| !eot.contains(token))
        .count()
}

fn ngram_lane_run(
    run_id: String,
    phase: RunPhase,
    workload: Workload,
    lane: Lane,
    repetition: u32,
    requested_output_tokens: u64,
    prompt_token_ids: &[u32],
    eot: &[u32],
    result: camelid::gemma4_runtime::Gemma4SpecDecodeExperimentResult,
) -> Result<LaneRun, String> {
    if !matches!(
        lane,
        Lane::NgramBaseline | Lane::NgramAssistantIdle | Lane::NgramSeedOffDiagnostic
    ) {
        return Err(format!("ngram result cannot populate {lane:?}"));
    }
    let plan = LaneExecutionPlan::for_lane(lane);
    let mut trace =
        ProposalTraceHasher::new(workload, prompt_token_ids, &plan.proposal_environment);
    let mut rounds = Vec::with_capacity(result.rounds.len());
    for receipt in result.rounds {
        trace.record(
            receipt.round_index,
            receipt.requested_k,
            &receipt.proposed_drafts,
        );
        let visible_output_tokens = visible_committed_tokens(&receipt.committed_tokens, eot);
        rounds.push(RoundTelemetry {
            workload,
            lane,
            repetition,
            round_index: receipt.round_index as u64,
            prefix_tokens_before: receipt.prefix_tokens_before as u64,
            proposal_source: match lane {
                Lane::NgramSeedOffDiagnostic => ProposalSource::NgramSeedOff,
                _ => ProposalSource::NgramSeeded,
            },
            assistant_invoked: false,
            requested_k: usize_as_u32("requested K", receipt.requested_k)?,
            proposed_k: usize_as_u32("proposed K", receipt.proposed_drafts.len())?,
            verifier_k: usize_as_u32("verifier K", receipt.verifier_k)?,
            budget_truncated: receipt.budget_truncated,
            accepted_drafts: usize_as_u32("accepted drafts", receipt.accepted_drafts)?,
            useful_accepted_drafts: usize_as_u32(
                "useful accepted drafts",
                receipt.useful_accepted_drafts,
            )?,
            emitted_target_tokens: usize_as_u32(
                "useful target-committed tokens",
                receipt.useful_accepted_drafts.saturating_add(1),
            )?,
            visible_output_tokens: usize_as_u32("visible output tokens", visible_output_tokens)?,
            draft_wall_us: receipt.draft_wall_us,
            draft_gpu_us: 0,
            verify_wall_us: receipt.target_verify_wall_us,
            verify_gpu_us: receipt.target_verify_gpu_us,
            round_wall_us: receipt.total_wall_us,
        });
    }
    let terminal_target_tokens = u64::from(result.terminal_unforwarded_target_token.is_some());
    let metrics = RunMetrics::from_rounds_and_terminal(&rounds, terminal_target_tokens)?;
    let run = LaneRun {
        run_id,
        phase,
        workload,
        lane,
        repetition,
        requested_output_tokens,
        child_process_id: std::process::id(),
        driver: LaneDriver::CurrentSpecDecodeGenerate,
        assistant_invocations: 0,
        proposal_trace_sha256: trace.finish(),
        output_token_ids: result.generated_tokens,
        terminal_target_tokens,
        generation_wall_us: result.total_wall_us,
        completed: !result.aborted,
        rounds,
        metrics,
        routed_experts_after_generation: None,
    };
    Ok(run)
}

fn mtp_lane_run(
    run_id: String,
    phase: RunPhase,
    workload: Workload,
    repetition: u32,
    requested_output_tokens: u64,
    prompt_token_ids: &[u32],
    eot: &[u32],
    result: camelid::gemma4_runtime::Gemma4MtpGenerationResult,
) -> Result<LaneRun, String> {
    let plan = LaneExecutionPlan::for_lane(Lane::Mtp);
    let mut trace =
        ProposalTraceHasher::new(workload, prompt_token_ids, &plan.proposal_environment);
    let mut assistant_invocations = 0u64;
    let mut rounds = Vec::with_capacity(result.rounds.len());
    for receipt in result.rounds {
        trace.record(
            receipt.round_index,
            receipt.requested_k,
            &receipt.proposed_drafts,
        );
        assistant_invocations =
            assistant_invocations.saturating_add(receipt.proposed_drafts.len() as u64);
        let visible_output_tokens = visible_committed_tokens(&receipt.committed_tokens, eot);
        let generated_prefix = receipt
            .prefix_tokens_before
            .checked_sub(prompt_token_ids.len())
            .ok_or_else(|| {
                format!(
                    "M round prefix {} precedes prompt length {}",
                    receipt.prefix_tokens_before,
                    prompt_token_ids.len()
                )
            })?;
        rounds.push(RoundTelemetry {
            workload,
            lane: Lane::Mtp,
            repetition,
            round_index: receipt.round_index as u64,
            prefix_tokens_before: generated_prefix as u64,
            proposal_source: ProposalSource::MtpAssistant,
            assistant_invoked: !receipt.proposed_drafts.is_empty(),
            requested_k: usize_as_u32("requested K", receipt.requested_k)?,
            proposed_k: usize_as_u32("proposed K", receipt.proposed_drafts.len())?,
            verifier_k: usize_as_u32("verifier K", receipt.verifier_k)?,
            budget_truncated: receipt.budget_truncated,
            accepted_drafts: usize_as_u32("accepted drafts", receipt.accepted_drafts)?,
            useful_accepted_drafts: usize_as_u32(
                "useful accepted drafts",
                receipt.useful_accepted_drafts,
            )?,
            emitted_target_tokens: usize_as_u32(
                "useful target-committed tokens",
                receipt.useful_accepted_drafts.saturating_add(1),
            )?,
            visible_output_tokens: usize_as_u32("visible output tokens", visible_output_tokens)?,
            draft_wall_us: receipt.assistant_wall_us,
            draft_gpu_us: receipt.assistant_gpu_us,
            verify_wall_us: receipt.target_verify_wall_us,
            verify_gpu_us: receipt.target_verify_gpu_us,
            round_wall_us: receipt.total_wall_us,
        });
    }
    let terminal_target_tokens = u64::from(result.terminal_unforwarded_target_token.is_some());
    let metrics = RunMetrics::from_rounds_and_terminal(&rounds, terminal_target_tokens)?;
    let run = LaneRun {
        run_id,
        phase,
        workload,
        lane: Lane::Mtp,
        repetition,
        requested_output_tokens,
        child_process_id: std::process::id(),
        driver: LaneDriver::NativeMtpExperiment,
        assistant_invocations,
        proposal_trace_sha256: trace.finish(),
        output_token_ids: result.generated_tokens,
        terminal_target_tokens,
        generation_wall_us: result.total_wall_us,
        completed: !result.aborted,
        rounds,
        metrics,
        routed_experts_after_generation: None,
    };
    Ok(run)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LaneMemoryReceipt {
    run_id: String,
    phase: RunPhase,
    workload: Workload,
    lane: Lane,
    repetition: u32,
    child_process_id: u32,
    assistant_warm_target_free: bool,
    snapshots: Vec<MemorySnapshot>,
    assistant_memory: AssistantMemory,
    assistant_load_delta: IncrementalMemory,
    peak_experiment_delta: IncrementalMemory,
    routed_experts_after_target_warm:
        Option<camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot>,
    routed_experts_after_generation:
        Option<camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot>,
    monitor_samples: Vec<SystemVmSnapshot>,
    process_monitor_samples: Vec<ProcessMemorySnapshot>,
    kill_reason: Option<KillReason>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExperimentReport {
    schema_version: u32,
    pilot_only: bool,
    pilot_tokens: u64,
    pilot_repetitions: u32,
    matrix_tokens: u64,
    matrix_repetitions: u32,
    matrix_skipped_by_pilot_only: bool,
    incorporated_child_receipts: u32,
    atomically_checkpointed_child_receipts: u32,
    atomic_child_checkpointing_enabled: bool,
    native_admission_completed_before_timed_lanes: bool,
    integration_executable_sha256: String,
    native_admission_run_nonce: String,
    native_admission: NativeAdmissionReceipt,
    pair_gate_wall_us: u64,
    pair_gate_completed_before_timed_lanes: bool,
    assistant_path: PathBuf,
    pairing: PairingEvidence,
    target_runtime: TargetRuntimeConfig,
    lane_plans: Vec<LaneExecutionPlan>,
    assistant_memory: AssistantMemory,
    memory_snapshots: Vec<MemorySnapshot>,
    assistant_load_delta: IncrementalMemory,
    peak_experiment_delta: IncrementalMemory,
    lane_memory: Vec<LaneMemoryReceipt>,
    runs: Vec<LaneRun>,
    economics: Vec<NimEconomics>,
    monitor_samples: Vec<SystemVmSnapshot>,
    kill_reason: Option<KillReason>,
    /// Must remain zero: only tokens accepted by the full target may commit.
    unverified_assistant_tokens_committed: u64,
    target_authoritative: bool,
    target_shared_kv_used: bool,
}

impl ExperimentReport {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != REPORT_SCHEMA_VERSION
            || !self.native_admission_completed_before_timed_lanes
            || !self.pair_gate_completed_before_timed_lanes
            || self.pair_gate_wall_us == 0
        {
            return Err("report lacks a pre-timing admission/pair-gate boundary receipt".into());
        }
        self.native_admission.validate(
            &self.integration_executable_sha256,
            &self.native_admission_run_nonce,
        )?;
        self.pairing.validate(&self.assistant_path)?;
        self.target_runtime.validate()?;
        validate_lane_plans(&self.lane_plans)?;
        if self.pilot_tokens != PILOT_TOKENS
            || self.pilot_repetitions == 0
            || self.matrix_tokens == 0
            || self.matrix_repetitions == 0
            || self.incorporated_child_receipts != self.lane_memory.len() as u32
            || (self.atomic_child_checkpointing_enabled
                && self.atomically_checkpointed_child_receipts != self.incorporated_child_receipts)
            || (!self.atomic_child_checkpointing_enabled
                && self.atomically_checkpointed_child_receipts != 0)
        {
            return Err("report has an inconsistent pilot/checkpoint receipt".into());
        }
        if self.pilot_only {
            if self.pilot_repetitions != 1
                || !self.atomic_child_checkpointing_enabled
                || self.runs.iter().any(|run| run.phase == RunPhase::Matrix)
                || self
                    .lane_memory
                    .iter()
                    .any(|memory| memory.phase == RunPhase::Matrix)
            {
                return Err("pilot-only report escaped its exact one-group boundary".into());
            }
            if self.kill_reason.is_none() {
                let complete_pilot = self
                    .runs
                    .iter()
                    .filter(|run| {
                        run.phase == RunPhase::Pilot
                            && run.workload == Workload::Copy
                            && run.repetition == 0
                            && run.requested_output_tokens == PILOT_TOKENS
                            && run.completed
                    })
                    .collect::<Vec<_>>();
                if !self.matrix_skipped_by_pilot_only
                    || complete_pilot.len() != 3
                    || [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp]
                        .into_iter()
                        .any(|lane| {
                            complete_pilot.iter().filter(|run| run.lane == lane).count() != 1
                        })
                {
                    return Err(
                        "successful pilot-only report is not exactly one 32-token N/I/M group"
                            .into(),
                    );
                }
            }
        } else if self.matrix_skipped_by_pilot_only {
            return Err("non-pilot report claims a pilot-only matrix skip".into());
        }
        if self.target_runtime.runtime_gguf_path != self.pairing.target_staged_runtime_path
            || self.target_runtime.cghost_path != self.pairing.target_staged_cghost_path
        {
            return Err("executed target paths differ from the pair-gated staged target".into());
        }
        if !self.target_authoritative {
            return Err("full target was not authoritative".to_string());
        }
        if self.kill_reason.is_none() && !self.target_shared_kv_used {
            return Err("official assistant did not use the target shared-KV path".to_string());
        }
        if self.unverified_assistant_tokens_committed != 0 {
            return Err(format!(
                "{} unverified assistant tokens were committed",
                self.unverified_assistant_tokens_committed
            ));
        }
        let assistant_memory_required = self
            .runs
            .iter()
            .any(|run| matches!(run.lane, Lane::NgramAssistantIdle | Lane::Mtp));
        // `mmap` rounds a mapping up to a whole page, so a CORRECT mapping of a
        // file whose length is not page-aligned is legitimately larger than the
        // file. The official assistant is 839,427,840 B = 51,234.61 pages, so its
        // mapping is 839,434,240 B — 6,400 B more. Comparing `mapped_bytes`
        // directly against `file_bytes` therefore failed closed on every run that
        // actually loaded the assistant; it stayed latent only because no lane
        // that loads the assistant had ever completed before 2026-08-21.
        let page_size = {
            let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if raw > 0 { raw as u64 } else { 16_384 }
        };
        let mapped_upper_bound = self
            .assistant_memory
            .file_bytes
            .div_ceil(page_size)
            .saturating_mul(page_size);
        if assistant_memory_required
            && (self.assistant_memory.file_bytes == 0
                || self.assistant_memory.model_bytes == 0
                || self.assistant_memory.mapped_bytes == 0
                || self.assistant_memory.locked_bytes == 0
                || self.assistant_memory.resident_bytes == 0
                || self.assistant_memory.file_bytes != self.pairing.assistant_staged_model_bytes
                || self.assistant_memory.model_bytes > self.assistant_memory.file_bytes
                || self.assistant_memory.mapped_bytes > mapped_upper_bound
                || self.assistant_memory.locked_bytes != self.assistant_memory.mapped_bytes
                || self.assistant_memory.resident_bytes > self.assistant_memory.mapped_bytes)
        {
            return Err("assistant memory accounting is incomplete or inconsistent".to_string());
        }
        if self.target_shared_kv_used && self.assistant_memory.borrowed_target_kv_bytes == 0 {
            return Err("shared target KV was used without a native capacity receipt".to_string());
        }
        let mut run_ids = BTreeMap::<String, (RunPhase, Workload, Lane, u32)>::new();
        let memory_by_run = self
            .lane_memory
            .iter()
            .map(|memory| (memory.run_id.as_str(), memory))
            .collect::<BTreeMap<_, _>>();
        for memory in &self.lane_memory {
            match (
                memory.routed_experts_after_target_warm.as_ref(),
                memory.routed_experts_after_generation.as_ref(),
            ) {
                (Some(warm), Some(after)) => validate_routed_expert_transition(warm, after)?,
                (Some(warm), None) => {
                    validate_routed_expert_snapshot("lane-memory post-target-warm snapshot", warm)?
                }
                (None, Some(after)) => {
                    validate_routed_expert_snapshot("lane-memory post-generation snapshot", after)?
                }
                (None, None) => {}
            }
        }
        let mut comparable_runs =
            BTreeMap::<(RunPhase, Workload, u32, u64), BTreeMap<Lane, &LaneRun>>::new();
        for run in &self.runs {
            let expected_plan = self
                .lane_plans
                .iter()
                .find(|plan| plan.lane == run.lane)
                .ok_or_else(|| format!("report has no execution plan for {:?}", run.lane))?;
            run.validate(expected_plan)?;
            if let Some(previous) = run_ids.insert(
                run.run_id.clone(),
                (run.phase, run.workload, run.lane, run.repetition),
            ) {
                return Err(format!(
                    "run ID {} was reused by {:?} and {:?}",
                    run.run_id,
                    previous,
                    (run.phase, run.workload, run.lane, run.repetition)
                ));
            }
            let memory = memory_by_run
                .get(run.run_id.as_str())
                .ok_or_else(|| format!("run {} has no child memory receipt", run.run_id))?;
            if memory.child_process_id != run.child_process_id
                || memory.phase != run.phase
                || memory.workload != run.workload
                || memory.lane != run.lane
                || memory.repetition != run.repetition
            {
                return Err(format!(
                    "run {} and its memory receipt disagree",
                    run.run_id
                ));
            }
            match (
                memory.routed_experts_after_target_warm.as_ref(),
                memory.routed_experts_after_generation.as_ref(),
                run.routed_experts_after_generation.as_ref(),
            ) {
                (Some(warm), Some(after), Some(run_after)) => {
                    validate_routed_expert_transition(warm, after)?;
                    if after != run_after {
                        return Err(format!(
                            "run {} and its memory receipt disagree on post-generation expert residency",
                            run.run_id
                        ));
                    }
                }
                _ if run.completed => {
                    return Err(format!(
                        "completed run {} omitted its warm/generation expert residency pair",
                        run.run_id
                    ));
                }
                (warm, after, run_after) => {
                    if let Some(snapshot) = warm {
                        validate_routed_expert_snapshot(
                            "incomplete-run post-target-warm snapshot",
                            snapshot,
                        )?;
                    }
                    if let Some(snapshot) = after.or(run_after) {
                        validate_routed_expert_snapshot(
                            "incomplete-run post-generation snapshot",
                            snapshot,
                        )?;
                    }
                }
            }
            if matches!(run.lane, Lane::NgramAssistantIdle | Lane::Mtp)
                && run.completed
                && !memory.assistant_warm_target_free
            {
                return Err(format!(
                    "run {} did not prove a target-free assistant warmup",
                    run.run_id
                ));
            }
            if run.lane != Lane::Mtp && memory.assistant_memory.borrowed_target_kv_bytes != 0 {
                return Err(format!(
                    "non-M run {} reported borrowed target-KV capacity",
                    run.run_id
                ));
            }
            if run.lane == Lane::Mtp
                && run.assistant_invocations > 0
                && memory.assistant_memory.borrowed_target_kv_bytes == 0
            {
                return Err(format!(
                    "M run {} invoked the assistant without borrowed target-KV capacity",
                    run.run_id
                ));
            }
            if self.kill_reason.is_none() && !run.completed {
                return Err(format!(
                    "successful report contains an incomplete {} {:?} run",
                    run.workload.key(),
                    run.lane
                ));
            }
            for (position, round) in run.rounds.iter().enumerate() {
                if round.workload != run.workload
                    || round.lane != run.lane
                    || round.repetition != run.repetition
                {
                    return Err("round ledger escaped its lane/workload/repetition".to_string());
                }
                if position > 0 {
                    let previous = &run.rounds[position - 1];
                    if round.round_index != previous.round_index.saturating_add(1) {
                        return Err(format!(
                            "{} {:?} round indices are not contiguous at position {}",
                            run.workload.key(),
                            run.lane,
                            position
                        ));
                    }
                    let expected_prefix = previous
                        .prefix_tokens_before
                        .saturating_add(u64::from(previous.emitted_target_tokens));
                    if round.prefix_tokens_before != expected_prefix {
                        return Err(format!(
                            "{} {:?} token prefixes are not contiguous at round {}",
                            run.workload.key(),
                            run.lane,
                            round.round_index
                        ));
                    }
                }
            }
            let recomputed =
                RunMetrics::from_rounds_and_terminal(&run.rounds, run.terminal_target_tokens)?;
            if recomputed != run.metrics {
                return Err(format!(
                    "{} {:?} metrics do not match their round ledger",
                    run.workload.key(),
                    run.lane
                ));
            }
            let peers = comparable_runs
                .entry((
                    run.phase,
                    run.workload,
                    run.repetition,
                    run.requested_output_tokens,
                ))
                .or_default();
            if peers.insert(run.lane, run).is_some() {
                return Err(format!(
                    "duplicate {:?} run for {} repetition {}",
                    run.lane,
                    run.workload.key(),
                    run.repetition
                ));
            }
        }
        for ((phase, workload, repetition, _), peers) in &comparable_runs {
            let primary = [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp];
            for left_index in 0..primary.len() {
                for right_index in (left_index + 1)..primary.len() {
                    let (Some(left), Some(right)) = (
                        peers.get(&primary[left_index]),
                        peers.get(&primary[right_index]),
                    ) else {
                        continue;
                    };
                    let common = left
                        .output_token_ids
                        .len()
                        .min(right.output_token_ids.len());
                    if left.output_token_ids[..common] != right.output_token_ids[..common] {
                        return Err(format!(
                            "target-authoritative output diverged in {phase:?} {} repetition {} between {:?} and {:?}",
                            workload.key(),
                            repetition,
                            left.lane,
                            right.lane
                        ));
                    }
                    if left.completed
                        && right.completed
                        && left.output_token_ids != right.output_token_ids
                    {
                        return Err(format!(
                            "completed target outputs differ in {phase:?} {} repetition {} between {:?} and {:?}",
                            workload.key(),
                            repetition,
                            left.lane,
                            right.lane
                        ));
                    }
                }
            }
            if let (Some(n), Some(i)) = (
                peers.get(&Lane::NgramBaseline),
                peers.get(&Lane::NgramAssistantIdle),
            ) {
                if n.proposal_trace_sha256 != i.proposal_trace_sha256 {
                    return Err(format!(
                        "N/I proposal traces differ in {phase:?} {} repetition {}",
                        workload.key(),
                        repetition
                    ));
                }
                let n_shape = n
                    .rounds
                    .iter()
                    .map(|round| {
                        (
                            round.prefix_tokens_before,
                            round.requested_k,
                            round.proposed_k,
                            round.verifier_k,
                            round.accepted_drafts,
                            round.useful_accepted_drafts,
                        )
                    })
                    .collect::<Vec<_>>();
                let i_shape = i
                    .rounds
                    .iter()
                    .map(|round| {
                        (
                            round.prefix_tokens_before,
                            round.requested_k,
                            round.proposed_k,
                            round.verifier_k,
                            round.accepted_drafts,
                            round.useful_accepted_drafts,
                        )
                    })
                    .collect::<Vec<_>>();
                if n_shape != i_shape {
                    return Err(format!(
                        "N/I round structure differs in {phase:?} {} repetition {}",
                        workload.key(),
                        repetition
                    ));
                }
            }
        }
        if self.kill_reason.is_none() && !self.pilot_only {
            for workload in [
                Workload::Copy,
                Workload::CodeEdit,
                Workload::JsonYaml,
                Workload::Prose,
            ] {
                for lane in [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp] {
                    if !self.runs.iter().any(|run| {
                        run.phase == RunPhase::Matrix
                            && run.workload == workload
                            && run.lane == lane
                            && run.completed
                    }) {
                        return Err(format!(
                            "successful report omitted completed matrix run {} {:?}",
                            workload.key(),
                            lane
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ExperimentRequest {
    assistant_path: PathBuf,
    target_runtime: TargetRuntimeConfig,
    native_admission: NativeAdmissionReceipt,
    integration_executable_sha256: String,
    native_admission_run_nonce: String,
    pairing: PairingEvidence,
    pair_gate_wall_us: u64,
    workloads: Vec<WorkloadSpec>,
    lanes: Vec<Lane>,
    lane_plans: Vec<LaneExecutionPlan>,
    pilot_tokens: u64,
    pilot_repetitions: u32,
    pilot_only: bool,
    report_path: Option<PathBuf>,
    matrix_tokens: u64,
    matrix_repetitions: u32,
    child_timeout_secs: u64,
    native_admission_must_pass: bool,
    pair_gate_must_pass: bool,
    use_target_shared_kv: bool,
    target_must_verify_every_proposal: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChildLaneRequest {
    schema_version: u32,
    run_id: String,
    parent_process_id: u32,
    phase: RunPhase,
    workload: WorkloadSpec,
    lane: Lane,
    repetition: u32,
    requested_output_tokens: u64,
    native_admission: NativeAdmissionReceipt,
    integration_executable_sha256: String,
    native_admission_run_nonce: String,
    assistant_path: PathBuf,
    target_runtime: TargetRuntimeConfig,
    pairing: PairingEvidence,
    plan: LaneExecutionPlan,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ChildLaneResult {
    schema_version: u32,
    run_id: String,
    child_process_id: u32,
    run: Option<LaneRun>,
    memory: LaneMemoryReceipt,
    target_shared_kv_used: bool,
    error: Option<String>,
}

#[derive(Debug)]
enum NativeAdapterError {
    Rejected(String),
}

impl std::fmt::Display for NativeAdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(message) => {
                write!(formatter, "native MTP experiment rejected: {message}")
            }
        }
    }
}

/// Narrow boundary for the future native Gemma 4 Assistant implementation.
/// The implementation must emit genuine target-verifier receipts and genuine
/// GPU timings; callers cannot construct synthetic rounds through this trait.
trait NativeMtpExperimentAdapter {
    fn run(&mut self, request: &ExperimentRequest) -> Result<ExperimentReport, NativeAdapterError>;
}

struct NativeChildProcessAdapter;

fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("JSON path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create JSON directory {}: {error}", parent.display()))?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mtp-json"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize {}: {error}", path.display()))?;
    fs::write(&temporary, bytes)
        .map_err(|error| format!("write temporary JSON {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        format!(
            "publish JSON {} -> {}: {error}",
            temporary.display(),
            path.display()
        )
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn load_native_admission_receipt(
    integration_executable_path: &Path,
    integration_executable_sha256: &str,
    expected_run_nonce: &str,
) -> Result<NativeAdmissionReceipt, String> {
    validate_native_admission_run_nonce(expected_run_nonce)?;
    let path = std::env::var_os(NATIVE_ADMISSION_EVIDENCE_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!("set {NATIVE_ADMISSION_EVIDENCE_ENV} to a fresh native v2 PASS receipt")
        })?;
    if !path.is_absolute() {
        return Err(format!("{NATIVE_ADMISSION_EVIDENCE_ENV} must be absolute"));
    }
    let canonical = canonical_internal_timed_file(&path, "native admission receipt")?;
    if !canonical.starts_with("/Users/timtoole/") {
        return Err(format!(
            "native admission receipt escaped its producer's internal path policy: {}",
            canonical.display()
        ));
    }
    use std::io::Read as _;
    let mut file = fs::File::open(&canonical)
        .map_err(|error| format!("open native admission receipt: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("stat open native admission receipt: {error}"))?;
    if metadata.len() == 0 || metadata.len() > 1024 * 1024 {
        return Err(format!(
            "native admission receipt size {} is outside 1..=1048576 bytes",
            metadata.len()
        ));
    }
    let modified = metadata
        .modified()
        .map_err(|error| format!("read native admission receipt mtime: {error}"))?;
    let parent_validated_at = SystemTime::now();
    let age = parent_validated_at
        .duration_since(modified)
        .map_err(|_| "native admission receipt mtime is in the future".to_string())?;
    if age > MAX_NATIVE_ADMISSION_AGE {
        return Err(format!(
            "native admission receipt is stale ({} seconds old; limit {})",
            age.as_secs(),
            MAX_NATIVE_ADMISSION_AGE.as_secs()
        ));
    }
    let modified_unix_ms = modified
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "native admission receipt mtime predates Unix epoch".to_string())?
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let parent_validated_unix_ms = parent_validated_at
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "parent validation time predates Unix epoch".to_string())?
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;
    let mut raw = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut raw)
        .map_err(|error| format!("read open native admission receipt: {error}"))?;
    let raw_json = String::from_utf8(raw)
        .map_err(|_| "native admission receipt is not UTF-8 JSON".to_string())?;
    let evidence: NativeAdmissionEvidence = serde_json::from_str(&raw_json)
        .map_err(|error| format!("parse strict native admission receipt: {error}"))?;
    let admission_test_executable_path = std::env::var_os(NATIVE_ADMISSION_TEST_EXE_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "set {NATIVE_ADMISSION_TEST_EXE_ENV} to the exact internal lib-test executable that emitted the PASS receipt"
            )
        })?;
    if !admission_test_executable_path.is_absolute() {
        return Err(format!("{NATIVE_ADMISSION_TEST_EXE_ENV} must be absolute"));
    }
    let admission_test_executable_path = canonical_internal_timed_file(
        &admission_test_executable_path,
        "native admission test executable",
    )?;
    if !admission_test_executable_path.starts_with("/Users/timtoole/") {
        return Err(format!(
            "native admission test executable escaped its internal path policy: {}",
            admission_test_executable_path.display()
        ));
    }
    if admission_test_executable_path == integration_executable_path {
        return Err(
            "native admission and integration executable paths were incorrectly conflated".into(),
        );
    }
    let admission_test_executable_sha256 = file_sha256_uncached(
        &admission_test_executable_path,
        "native admission test executable",
    )?;
    if admission_test_executable_sha256 == integration_executable_sha256 {
        return Err(
            "native admission and integration executables were incorrectly conflated".into(),
        );
    }
    if admission_test_executable_sha256 != evidence.admission_test_exe_sha256 {
        return Err(format!(
            "native admission test executable SHA-256 {} does not match PASS receipt {}",
            admission_test_executable_sha256, evidence.admission_test_exe_sha256
        ));
    }
    let receipt = NativeAdmissionReceipt {
        receipt_path: canonical,
        receipt_sha256: bytes_sha256(raw_json.as_bytes()),
        receipt_raw_json: raw_json,
        receipt_modified_unix_ms: modified_unix_ms,
        parent_validated_unix_ms,
        admission_test_executable_path,
        admission_test_executable_sha256,
        evidence,
    };
    receipt.validate(integration_executable_sha256, expected_run_nonce)?;
    Ok(receipt)
}

fn unique_run_id(phase: RunPhase, workload: Workload, lane: Lane, repetition: u32) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "p{}-{now}-{phase:?}-{}-{lane:?}-r{repetition}",
        std::process::id(),
        workload.key()
    )
    .to_ascii_lowercase()
}

fn balanced_lane_order(repetition: u32) -> [Lane; 3] {
    const ORDERS: [[Lane; 3]; 3] = [
        [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp],
        [Lane::NgramAssistantIdle, Lane::Mtp, Lane::NgramBaseline],
        [Lane::Mtp, Lane::NgramBaseline, Lane::NgramAssistantIdle],
    ];
    ORDERS[repetition as usize % ORDERS.len()]
}

fn median_u64(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn lane_window_from_run(run: &LaneRun) -> LaneWindow {
    LaneWindow {
        key: WindowKey {
            workload: run.workload,
            repetition: run.repetition,
            prefix_start: 0,
            prefix_end: run.output_token_ids.len() as u64,
        },
        lane: run.lane,
        verify_rounds: run.metrics.rounds,
        emitted_tokens: run.output_token_ids.len() as u64,
        accepted_drafts: run.metrics.useful_accepted_drafts,
        draft_wall_us: run.metrics.draft_wall_us,
        verify_wall_us: run.metrics.verify_wall_us,
        total_wall_us: run.generation_wall_us,
        assistant_invocations: run.assistant_invocations,
        proposal_trace_sha256: run.proposal_trace_sha256.clone(),
    }
}

fn child_timeout(request: &ExperimentRequest) -> Duration {
    Duration::from_secs(request.child_timeout_secs.max(1))
}

fn cleanup_child_ipc(directory: &Path, request_path: &Path, result_path: &Path) {
    let _ = fs::remove_file(request_path);
    let _ = fs::remove_file(result_path);
    let _ = fs::remove_dir(directory);
}

#[derive(Debug)]
struct ChildLaunchFailure {
    reason: KillReason,
    message: String,
    parent_monitor_samples: Vec<SystemVmSnapshot>,
    parent_process_samples: Vec<ProcessMemorySnapshot>,
    child_process_id: u32,
}

impl ChildLaunchFailure {
    fn new(reason: KillReason, message: String) -> Self {
        Self {
            reason,
            message,
            parent_monitor_samples: Vec::new(),
            parent_process_samples: Vec::new(),
            child_process_id: 0,
        }
    }

    fn monitored(
        reason: KillReason,
        message: String,
        parent_monitor_samples: Vec<SystemVmSnapshot>,
        parent_process_samples: Vec<ProcessMemorySnapshot>,
        child_process_id: u32,
    ) -> Self {
        Self {
            reason,
            message,
            parent_monitor_samples,
            parent_process_samples,
            child_process_id,
        }
    }
}

fn launch_lane_child(
    request: &ExperimentRequest,
    child_request: &ChildLaneRequest,
) -> Result<ChildLaneResult, ChildLaunchFailure> {
    // The finalized integration-test binary is staged onto internal APFS once
    // by the operator. Never copy an executable from an external volume in the
    // lane loop: that would add unreceipted T7 I/O immediately before timing.
    let executable = std::env::current_exe()
        .map_err(|error| {
            ChildLaunchFailure::new(
                KillReason::ChildFailure(error.to_string()),
                format!("resolve integration-test executable: {error}"),
            )
        })
        .and_then(|path| {
            canonical_internal_timed_file(&path, "lane child executable").map_err(|error| {
                ChildLaunchFailure::new(KillReason::ChildFailure(error.clone()), error)
            })
        })?;
    let directory = std::env::temp_dir().join(format!(
        "camelid-mtp-child-{}-{}",
        std::process::id(),
        child_request.run_id
    ));
    fs::create_dir(&directory).map_err(|error| {
        ChildLaunchFailure::new(
            KillReason::ChildFailure(error.to_string()),
            format!(
                "create child IPC directory {}: {error}",
                directory.display()
            ),
        )
    })?;
    let request_path = directory.join("request.json");
    let result_path = directory.join("result.json");
    if let Err(error) = atomic_write_json(&request_path, child_request) {
        let _ = fs::remove_dir(&directory);
        return Err(ChildLaunchFailure::new(
            KillReason::ChildFailure(error.clone()),
            error,
        ));
    }

    let watchdog_baseline = system_vm_snapshot(0).map_err(|error| {
        cleanup_child_ipc(&directory, &request_path, &result_path);
        ChildLaunchFailure::new(KillReason::TelemetryUnavailable(error.clone()), error)
    })?;
    let mut child = match Command::new(&executable)
        .arg("gemma4_mtp_assistant_lane_child")
        .arg("--exact")
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(EXPERIMENT_ENABLE_ENV, "1")
        .env(CHILD_REQUEST_ENV, &request_path)
        .env(CHILD_RESULT_ENV, &result_path)
        // Admission identity is carried only in the immutable structured IPC
        // request. A lane child must never reopen or trust ambient receipt
        // paths/nonces that could change after the parent gate.
        .env_remove(NATIVE_ADMISSION_EVIDENCE_ENV)
        .env_remove(NATIVE_ADMISSION_RUN_NONCE_ENV)
        .env_remove(NATIVE_ADMISSION_TEST_EXE_ENV)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            cleanup_child_ipc(&directory, &request_path, &result_path);
            return Err(ChildLaunchFailure::new(
                KillReason::ChildFailure(error.to_string()),
                format!("spawn isolated lane child: {error}"),
            ));
        }
    };
    let expected_pid = child.id();
    let started = Instant::now();
    let mut next_watchdog_sample = MONITOR_PERIOD;
    let mut watchdog_samples = vec![watchdog_baseline.clone()];
    let mut watchdog_process_samples = Vec::new();
    let mut watchdog = KillPolicyState::new(watchdog_baseline);
    let status = 'child_wait: loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < child_timeout(request) => {
                if started.elapsed() >= next_watchdog_sample {
                    let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                    let process_sample = match process_memory_snapshot_for_pid(expected_pid) {
                        Ok(sample) => sample,
                        Err(error) => {
                            // The child may have exited in the small window
                            // after try_wait returned None. Preserve its valid
                            // structured result instead of fabricating a
                            // telemetry failure for a process that is gone.
                            if let Ok(Some(status)) = child.try_wait() {
                                break 'child_wait status;
                            }
                            let _ = child.kill();
                            let _ = child.wait();
                            cleanup_child_ipc(&directory, &request_path, &result_path);
                            return Err(ChildLaunchFailure::monitored(
                                KillReason::TelemetryUnavailable(error.clone()),
                                format!(
                                    "parent watchdog lost child-process telemetry for {expected_pid}: {error}"
                                ),
                                watchdog_samples,
                                watchdog_process_samples,
                                expected_pid,
                            ));
                        }
                    };
                    let sample = match system_vm_snapshot(elapsed_ms) {
                        Ok(sample) => sample,
                        Err(error) => {
                            if let Ok(Some(status)) = child.try_wait() {
                                break 'child_wait status;
                            }
                            let _ = child.kill();
                            let _ = child.wait();
                            cleanup_child_ipc(&directory, &request_path, &result_path);
                            return Err(ChildLaunchFailure::monitored(
                                KillReason::TelemetryUnavailable(error.clone()),
                                format!(
                                    "parent watchdog lost system telemetry for child {expected_pid}: {error}"
                                ),
                                watchdog_samples,
                                watchdog_process_samples,
                                expected_pid,
                            ));
                        }
                    };
                    watchdog_samples.push(sample.clone());
                    watchdog_process_samples.push(process_sample);
                    if let Some(reason) = watchdog.observe_memory(sample) {
                        let _ = child.kill();
                        let _ = child.wait();
                        cleanup_child_ipc(&directory, &request_path, &result_path);
                        return Err(ChildLaunchFailure::monitored(
                            reason.clone(),
                            format!(
                                "parent 1 Hz watchdog killed lane child {expected_pid}: {reason:?}"
                            ),
                            watchdog_samples,
                            watchdog_process_samples,
                            expected_pid,
                        ));
                    }
                    next_watchdog_sample += MONITOR_PERIOD;
                }
                thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                cleanup_child_ipc(&directory, &request_path, &result_path);
                return Err(ChildLaunchFailure::monitored(
                    KillReason::ChildTimeout,
                    format!(
                        "lane child {expected_pid} exceeded {} seconds",
                        request.child_timeout_secs
                    ),
                    watchdog_samples,
                    watchdog_process_samples,
                    expected_pid,
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                cleanup_child_ipc(&directory, &request_path, &result_path);
                return Err(ChildLaunchFailure::monitored(
                    KillReason::ChildFailure(error.to_string()),
                    format!("poll lane child {expected_pid}: {error}"),
                    watchdog_samples,
                    watchdog_process_samples,
                    expected_pid,
                ));
            }
        }
    };

    let parsed = read_json::<ChildLaneResult>(&result_path).map_err(|error| {
        ChildLaunchFailure::monitored(
            KillReason::ChildFailure(error.clone()),
            format!("lane child {expected_pid} ({status}) produced no valid receipt: {error}"),
            watchdog_samples.clone(),
            watchdog_process_samples.clone(),
            expected_pid,
        )
    });
    cleanup_child_ipc(&directory, &request_path, &result_path);
    let result = parsed?;
    if result.schema_version != REPORT_SCHEMA_VERSION
        || result.run_id != child_request.run_id
        || result.child_process_id != expected_pid
        || result.memory.child_process_id != expected_pid
        || result.memory.run_id != child_request.run_id
    {
        return Err(ChildLaunchFailure::monitored(
            KillReason::ChildFailure("child IPC identity mismatch".into()),
            format!("lane child {expected_pid} returned a mismatched structured receipt"),
            watchdog_samples,
            watchdog_process_samples,
            expected_pid,
        ));
    }
    let routed_receipt = match (
        result.memory.routed_experts_after_target_warm.as_ref(),
        result.memory.routed_experts_after_generation.as_ref(),
    ) {
        (Some(warm), Some(after)) => validate_routed_expert_transition(warm, after),
        (Some(warm), None) => {
            validate_routed_expert_snapshot("child post-target-warm snapshot", warm)
        }
        (None, Some(after)) => {
            validate_routed_expert_snapshot("child post-generation snapshot", after)
        }
        (None, None) => Ok(()),
    };
    if let Err(error) = routed_receipt {
        return Err(ChildLaunchFailure::monitored(
            KillReason::ChildFailure(error.clone()),
            format!("lane child {expected_pid} returned invalid expert residency: {error}"),
            watchdog_samples,
            watchdog_process_samples,
            expected_pid,
        ));
    }
    if let Some(run) = result.run.as_ref() {
        if let Err(error) = run.validate(&child_request.plan) {
            return Err(ChildLaunchFailure::monitored(
                KillReason::ChildFailure(error.clone()),
                format!("lane child {expected_pid} returned an invalid run receipt: {error}"),
                watchdog_samples,
                watchdog_process_samples,
                expected_pid,
            ));
        }
        if run.routed_experts_after_generation.as_ref()
            != result.memory.routed_experts_after_generation.as_ref()
        {
            return Err(ChildLaunchFailure::monitored(
                KillReason::ChildFailure("child run/memory expert residency mismatch".into()),
                format!(
                    "lane child {expected_pid} run and memory receipts disagree on expert residency"
                ),
                watchdog_samples,
                watchdog_process_samples,
                expected_pid,
            ));
        }
    }
    if !status.success() && result.error.is_none() {
        return Err(ChildLaunchFailure::monitored(
            KillReason::ChildFailure(status.to_string()),
            format!("lane child {expected_pid} exited {status} without an error receipt"),
            watchdog_samples,
            watchdog_process_samples,
            expected_pid,
        ));
    }
    Ok(result)
}

impl NativeChildProcessAdapter {
    fn incorporate_child(report: &mut ExperimentReport, child: ChildLaneResult) {
        report.target_shared_kv_used |= child.target_shared_kv_used;
        if child.memory.assistant_memory.file_bytes != 0 {
            if report.assistant_memory.file_bytes == 0 {
                report.assistant_memory = child.memory.assistant_memory.clone();
            } else {
                report.assistant_memory.borrowed_target_kv_bytes = report
                    .assistant_memory
                    .borrowed_target_kv_bytes
                    .max(child.memory.assistant_memory.borrowed_target_kv_bytes);
            }
            componentwise_max_delta(
                &mut report.assistant_load_delta,
                &child.memory.assistant_load_delta,
            );
        }
        componentwise_max_delta(
            &mut report.peak_experiment_delta,
            &child.memory.peak_experiment_delta,
        );
        report
            .memory_snapshots
            .extend(child.memory.snapshots.iter().cloned());
        report
            .monitor_samples
            .extend(child.memory.monitor_samples.iter().cloned());
        if report.kill_reason.is_none() {
            report.kill_reason = child.memory.kill_reason.clone();
        }
        if let Some(run) = child.run {
            report.runs.push(run);
        }
        report.lane_memory.push(child.memory);
        if report.kill_reason.is_none() {
            if let Some(error) = child.error {
                report.kill_reason = Some(KillReason::ChildFailure(error));
            }
        }
        report.incorporated_child_receipts = report.incorporated_child_receipts.saturating_add(1);
    }

    fn checkpoint_report(
        request: &ExperimentRequest,
        report: &mut ExperimentReport,
    ) -> Result<(), String> {
        let Some(path) = request.report_path.as_ref() else {
            return Ok(());
        };
        report.atomically_checkpointed_child_receipts = report.incorporated_child_receipts;
        atomic_write_json(path, report)
    }

    fn execute_group(
        &self,
        request: &ExperimentRequest,
        report: &mut ExperimentReport,
        phase: RunPhase,
        workload: &WorkloadSpec,
        repetition: u32,
        output_tokens: u64,
    ) {
        for lane in balanced_lane_order(repetition) {
            if report.kill_reason.is_some() {
                return;
            }
            let plan = LaneExecutionPlan::for_lane(lane);
            let child_request = ChildLaneRequest {
                schema_version: REPORT_SCHEMA_VERSION,
                run_id: unique_run_id(phase, workload.key, lane, repetition),
                parent_process_id: std::process::id(),
                phase,
                workload: workload.clone(),
                lane,
                repetition,
                requested_output_tokens: output_tokens,
                native_admission: request.native_admission.clone(),
                integration_executable_sha256: request.integration_executable_sha256.clone(),
                native_admission_run_nonce: request.native_admission_run_nonce.clone(),
                assistant_path: request.assistant_path.clone(),
                target_runtime: request.target_runtime.clone(),
                pairing: request.pairing.clone(),
                plan,
            };
            match launch_lane_child(request, &child_request) {
                Ok(child) => {
                    Self::incorporate_child(report, child);
                    if let Err(error) = Self::checkpoint_report(request, report) {
                        report.kill_reason = Some(KillReason::ChildFailure(format!(
                            "atomic top-level checkpoint after child failed: {error}"
                        )));
                        return;
                    }
                }
                Err(failure) => {
                    eprintln!("[mtp-experiment] {}", failure.message);
                    let mut memory = empty_lane_memory(&child_request);
                    memory.child_process_id = failure.child_process_id;
                    memory.monitor_samples = failure.parent_monitor_samples;
                    memory.process_monitor_samples = failure.parent_process_samples;
                    memory.kill_reason = Some(failure.reason.clone());
                    report
                        .monitor_samples
                        .extend(memory.monitor_samples.iter().cloned());
                    report.lane_memory.push(memory);
                    report.incorporated_child_receipts =
                        report.incorporated_child_receipts.saturating_add(1);
                    report.kill_reason = Some(failure.reason);
                    if let Err(error) = Self::checkpoint_report(request, report) {
                        report.kill_reason = Some(KillReason::ChildFailure(format!(
                            "{}; atomic top-level failure checkpoint also failed: {error}",
                            failure.message
                        )));
                    }
                    return;
                }
            }
        }
    }

    fn economics_for_group(
        report: &ExperimentReport,
        phase: RunPhase,
        workload: Workload,
        repetition: u32,
    ) -> Result<Option<NimEconomics>, String> {
        let find = |lane| {
            report.runs.iter().find(|run| {
                run.phase == phase
                    && run.workload == workload
                    && run.repetition == repetition
                    && run.lane == lane
                    && run.completed
            })
        };
        let (Some(n), Some(i), Some(m)) = (
            find(Lane::NgramBaseline),
            find(Lane::NgramAssistantIdle),
            find(Lane::Mtp),
        ) else {
            return Ok(None);
        };
        NimEconomics::derive(
            &lane_window_from_run(n),
            &lane_window_from_run(i),
            &lane_window_from_run(m),
        )
        .map(Some)
    }
}

impl NativeMtpExperimentAdapter for NativeChildProcessAdapter {
    fn run(&mut self, request: &ExperimentRequest) -> Result<ExperimentReport, NativeAdapterError> {
        if !request.native_admission_must_pass
            || !request.pair_gate_must_pass
            || !request.pairing.pair_gate_passed
            || !request.use_target_shared_kv
            || !request.target_must_verify_every_proposal
            || request.lanes != [Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp]
        {
            return Err(NativeAdapterError::Rejected(
                "request relaxed an authoritative experiment invariant".into(),
            ));
        }
        if request.pilot_tokens != PILOT_TOKENS
            || request.pilot_repetitions == 0
            || request.matrix_tokens == 0
            || request.matrix_repetitions == 0
            || request
                .report_path
                .as_ref()
                .is_some_and(|path| !path.is_absolute())
        {
            return Err(NativeAdapterError::Rejected(
                "request has an invalid pilot/matrix/checkpoint boundary".into(),
            ));
        }
        if request.pilot_only && (request.pilot_repetitions != 1 || request.report_path.is_none()) {
            return Err(NativeAdapterError::Rejected(
                "pilot-only mode requires exactly one repetition and an absolute checkpoint path"
                    .into(),
            ));
        }
        let current_executable_sha256 = current_internal_executable_sha256("MTP integration")
            .map_err(NativeAdapterError::Rejected)?;
        if current_executable_sha256 != request.integration_executable_sha256 {
            return Err(NativeAdapterError::Rejected(
                "current integration executable differs from the admitted parent binary".into(),
            ));
        }
        request
            .native_admission
            .validate(
                &request.integration_executable_sha256,
                &request.native_admission_run_nonce,
            )
            .map_err(NativeAdapterError::Rejected)?;
        request
            .pairing
            .validate(&request.assistant_path)
            .map_err(NativeAdapterError::Rejected)?;
        validate_lane_plans(&request.lane_plans).map_err(NativeAdapterError::Rejected)?;

        let mut report = ExperimentReport {
            schema_version: REPORT_SCHEMA_VERSION,
            pilot_only: request.pilot_only,
            pilot_tokens: request.pilot_tokens,
            pilot_repetitions: request.pilot_repetitions,
            matrix_tokens: request.matrix_tokens,
            matrix_repetitions: request.matrix_repetitions,
            matrix_skipped_by_pilot_only: false,
            incorporated_child_receipts: 0,
            atomically_checkpointed_child_receipts: 0,
            atomic_child_checkpointing_enabled: request.report_path.is_some(),
            native_admission_completed_before_timed_lanes: true,
            integration_executable_sha256: request.integration_executable_sha256.clone(),
            native_admission_run_nonce: request.native_admission_run_nonce.clone(),
            native_admission: request.native_admission.clone(),
            pair_gate_wall_us: request.pair_gate_wall_us,
            pair_gate_completed_before_timed_lanes: true,
            assistant_path: request.assistant_path.clone(),
            pairing: request.pairing.clone(),
            target_runtime: request.target_runtime.clone(),
            lane_plans: request.lane_plans.clone(),
            assistant_memory: AssistantMemory::default(),
            memory_snapshots: Vec::new(),
            assistant_load_delta: IncrementalMemory::default(),
            peak_experiment_delta: IncrementalMemory::default(),
            lane_memory: Vec::new(),
            runs: Vec::new(),
            economics: Vec::new(),
            monitor_samples: Vec::new(),
            kill_reason: None,
            unverified_assistant_tokens_committed: 0,
            target_authoritative: true,
            target_shared_kv_used: false,
        };
        Self::checkpoint_report(request, &mut report).map_err(|error| {
            NativeAdapterError::Rejected(format!(
                "initial atomic top-level checkpoint failed before lane launch: {error}"
            ))
        })?;

        let copy = request
            .workloads
            .iter()
            .find(|workload| workload.key == Workload::Copy)
            .ok_or_else(|| NativeAdapterError::Rejected("pilot copy workload is absent".into()))?;
        let mut economics_loss_streak = 0u32;
        for repetition in 0..request.pilot_repetitions {
            self.execute_group(
                request,
                &mut report,
                RunPhase::Pilot,
                copy,
                repetition,
                request.pilot_tokens,
            );
            if report.kill_reason.is_some() {
                return Ok(report);
            }
            let economics = match Self::economics_for_group(
                &report,
                RunPhase::Pilot,
                Workload::Copy,
                repetition,
            ) {
                Ok(economics) => economics,
                Err(error) => {
                    report.kill_reason = Some(KillReason::ChildFailure(format!(
                        "pilot economics receipt failed closed: {error}"
                    )));
                    return Ok(report);
                }
            };
            if let Some(economics) = economics {
                if economics.evidence_ready() && economics.no_gain_and_materially_slower() {
                    report.kill_reason =
                        Some(KillReason::NoAcceptanceGainAndMtpAtLeastFivePercentSlower);
                }
                economics_loss_streak =
                    if economics.evidence_ready() && economics.costs_at_least_savings() {
                        economics_loss_streak.saturating_add(1)
                    } else {
                        0
                    };
                report.economics.push(economics);
                if economics_loss_streak >= ECONOMICS_LOSS_STREAK_LIMIT {
                    report.kill_reason = Some(KillReason::EconomicsLostTwoConsecutiveWindows);
                }
            }
            if report.kill_reason.is_some() {
                return Ok(report);
            }
        }

        let pilot_wall = |lane| {
            report
                .runs
                .iter()
                .filter(|run| {
                    run.phase == RunPhase::Pilot
                        && run.workload == Workload::Copy
                        && run.lane == lane
                        && run.completed
                })
                .map(|run| run.generation_wall_us)
                .collect::<Vec<_>>()
        };
        let Some(n_median) = median_u64(pilot_wall(Lane::NgramBaseline)) else {
            report.kill_reason = Some(KillReason::ChildFailure(
                "pilot has no completed N runs".into(),
            ));
            return Ok(report);
        };
        let Some(m_median) = median_u64(pilot_wall(Lane::Mtp)) else {
            report.kill_reason = Some(KillReason::ChildFailure(
                "pilot has no completed M runs".into(),
            ));
            return Ok(report);
        };
        if m_median >= n_median {
            report.kill_reason = Some(KillReason::PilotMedianMtpNotFaster);
            return Ok(report);
        }

        if request.pilot_only {
            report.matrix_skipped_by_pilot_only = true;
            Self::checkpoint_report(request, &mut report).map_err(|error| {
                NativeAdapterError::Rejected(format!(
                    "final pilot-only checkpoint failed before matrix skip: {error}"
                ))
            })?;
            return Ok(report);
        }

        for workload in &request.workloads {
            for repetition in 0..request.matrix_repetitions {
                self.execute_group(
                    request,
                    &mut report,
                    RunPhase::Matrix,
                    workload,
                    repetition,
                    request.matrix_tokens,
                );
                if report.kill_reason.is_some() {
                    return Ok(report);
                }
                let economics = match Self::economics_for_group(
                    &report,
                    RunPhase::Matrix,
                    workload.key,
                    repetition,
                ) {
                    Ok(economics) => economics,
                    Err(error) => {
                        report.kill_reason = Some(KillReason::ChildFailure(format!(
                            "matrix economics receipt failed closed: {error}"
                        )));
                        return Ok(report);
                    }
                };
                if let Some(economics) = economics {
                    economics_loss_streak =
                        if economics.evidence_ready() && economics.costs_at_least_savings() {
                            economics_loss_streak.saturating_add(1)
                        } else {
                            0
                        };
                    if economics.evidence_ready() && economics.no_gain_and_materially_slower() {
                        report.kill_reason =
                            Some(KillReason::NoAcceptanceGainAndMtpAtLeastFivePercentSlower);
                    }
                    report.economics.push(economics);
                    if economics_loss_streak >= ECONOMICS_LOSS_STREAK_LIMIT {
                        report.kill_reason = Some(KillReason::EconomicsLostTwoConsecutiveWindows);
                    }
                }
                if report.kill_reason.is_some() {
                    return Ok(report);
                }
            }
        }
        Ok(report)
    }
}

fn apply_child_environment(request: &ChildLaneRequest) -> Result<(), String> {
    if request.schema_version != REPORT_SCHEMA_VERSION {
        return Err(format!(
            "child request schema {} is not {}",
            request.schema_version, REPORT_SCHEMA_VERSION
        ));
    }
    if request.plan != LaneExecutionPlan::for_lane(request.lane) {
        return Err("child lane plan differs from the authoritative plan".into());
    }
    request.native_admission.validate(
        &request.integration_executable_sha256,
        &request.native_admission_run_nonce,
    )?;
    request.pairing.validate(&request.assistant_path)?;
    request.target_runtime.validate()?;
    let child_executable = std::env::current_exe()
        .map_err(|error| format!("resolve lane child executable: {error}"))?;
    let child_executable =
        canonical_internal_timed_file(&child_executable, "lane child executable")?;
    let child_executable_sha256 = file_sha256(&child_executable, "lane child executable")?;
    if child_executable_sha256 != request.integration_executable_sha256 {
        return Err("lane child executable digest differs from admitted parent binary".into());
    }
    let runtime = canonical_internal_timed_file(
        &request.target_runtime.runtime_gguf_path,
        "target runtime GGUF",
    )?;
    let cghost =
        canonical_internal_timed_file(&request.target_runtime.cghost_path, "target cghost")?;
    let assistant = canonical_internal_timed_file(&request.assistant_path, "MTP assistant")?;
    if runtime != request.target_runtime.runtime_gguf_path
        || cghost != request.target_runtime.cghost_path
        || assistant != request.assistant_path
    {
        return Err("child received non-canonical timed artifact paths".into());
    }
    for (label, path) in [
        (
            "assistant config",
            &request.pairing.assistant_staged_config_path,
        ),
        (
            "assistant tokenizer config",
            &request.pairing.assistant_staged_tokenizer_config_path,
        ),
        (
            "assistant tokenizer",
            &request.pairing.assistant_staged_tokenizer_path,
        ),
    ] {
        let canonical = canonical_internal_timed_file(path, label)?;
        if &canonical != path {
            return Err(format!("child received a non-canonical timed {label} path"));
        }
    }

    // This child is single-threaded until the target runtime is constructed, so
    // scrubbing and applying its process-local experiment environment cannot
    // race a peer lane. Removing the entire tuning namespaces prevents an
    // inherited benchmark knob from silently escaping the serialized receipt.
    let inherited_tuning_keys = std::env::vars_os()
        .filter_map(|(key, _)| {
            let key_text = key.to_string_lossy();
            (key_text.starts_with("CAMELID_GEMMA4_")
                || key_text.starts_with("CAMELID_GHOST_")
                || key_text.starts_with("CAMELID_SPEC_")
                || key_text.starts_with("SPEC50_"))
            .then_some(key)
        })
        .collect::<Vec<_>>();
    for key in inherited_tuning_keys {
        std::env::remove_var(key);
    }
    for (key, value) in &request.target_runtime.environment {
        std::env::set_var(key, value);
    }
    for (key, value) in &request.plan.proposal_environment {
        std::env::set_var(key, value);
    }
    std::env::set_var(
        TARGET_RUNTIME_PATH_ENV,
        &request.target_runtime.runtime_gguf_path,
    );
    std::env::set_var(TARGET_CGHOST_PATH_ENV, &request.target_runtime.cghost_path);
    std::env::set_var(
        TARGET_CACHE_MIB_ENV,
        request.target_runtime.expert_cache_mib.to_string(),
    );
    std::env::set_var(ASSISTANT_PATH_ENV, &request.assistant_path);
    Ok(())
}

/// Apply only the exact target load environment. This is intentionally not
/// shared with the N/I/M child: the load probe has no proposal environment and
/// must not inherit one from an earlier benchmark shell.
fn apply_load_only_environment(
    target_runtime: &TargetRuntimeConfig,
    assistant_path: &Path,
) -> Result<(), String> {
    target_runtime.validate()?;
    let inherited_tuning_keys = std::env::vars_os()
        .filter_map(|(key, _)| {
            let key_text = key.to_string_lossy();
            (key_text.starts_with("CAMELID_GEMMA4_")
                || key_text.starts_with("CAMELID_GHOST_")
                || key_text.starts_with("CAMELID_SPEC_")
                || key_text.starts_with("SPEC50_"))
            .then_some(key)
        })
        .collect::<Vec<_>>();
    for key in inherited_tuning_keys {
        std::env::remove_var(key);
    }
    for (key, value) in &target_runtime.environment {
        std::env::set_var(key, value);
    }
    // The experiment's serialized TargetRuntimeConfig requires the same
    // file-backed no-copy head used by this observed loader. Force the probe
    // process to restate that value so a future env change cannot allocate a
    // copied resident head before the required-page phase.
    std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT", "0");
    std::env::set_var(TARGET_RUNTIME_PATH_ENV, &target_runtime.runtime_gguf_path);
    std::env::set_var(TARGET_CGHOST_PATH_ENV, &target_runtime.cghost_path);
    std::env::set_var(
        TARGET_CACHE_MIB_ENV,
        target_runtime.expert_cache_mib.to_string(),
    );
    std::env::set_var(ASSISTANT_PATH_ENV, assistant_path);
    Ok(())
}

fn shipping_argmax(logits: &[f32]) -> Result<u32, String> {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .partial_cmp(right.1)
                .unwrap_or(std::cmp::Ordering::Less)
        })
        .map(|(index, _)| index as u32)
        .ok_or_else(|| "target returned empty logits".into())
}

fn warm_target_runtime<C: FnMut() -> bool>(
    runtime: &camelid::gemma4_runtime::Gemma4Runtime,
    mut should_abort: C,
) -> Result<bool, String> {
    if should_abort() {
        return Ok(false);
    }
    let tokens = runtime
        .tokenizer()
        .encode(WARMUP_PROMPT, true, true)
        .map_err(|error| format!("tokenize target warmup: {error}"))?;
    let eot = runtime.stop_token_ids();
    let (mut kc, mut vc) = runtime.empty_kv_caches();
    let Some(mut logits) = runtime
        .prefill_tokens_cancellable_experiment(
            &tokens,
            &mut kc,
            &mut vc,
            TARGET_WARMUP_TOKENS as usize,
            || should_abort(),
        )
        .map_err(|error| format!("target warmup prefill: {error}"))?
    else {
        return Ok(false);
    };
    let mut pos = tokens.len();
    for _ in 0..TARGET_WARMUP_TOKENS {
        if should_abort() {
            return Ok(false);
        }
        let token = shipping_argmax(&logits)?;
        if eot.contains(&token) {
            break;
        }
        logits = runtime
            .step(token, pos, &mut kc, &mut vc)
            .map_err(|error| format!("target warmup step: {error}"))?;
        pos += 1;
    }
    Ok(true)
}

fn assistant_memory_from_ledger(
    ledger: camelid::metal::Gemma4MtpResidentLedger,
    page_size: u64,
) -> AssistantMemory {
    AssistantMemory {
        model_bytes: ledger.payload_bytes,
        file_bytes: ledger.file_bytes,
        mapped_bytes: ledger.mapped_bytes,
        locked_bytes: ledger.locked_bytes,
        resident_bytes: ledger.resident_pages.saturating_mul(page_size),
        private_bytes: ledger
            .decoded_norm_bytes
            .saturating_add(ledger.fixed_scratch_bytes),
        // Set only from the native ledger of a measured M proposal. The
        // target-free synthetic warmup restores the previous proposal ledger,
        // so idle-I cannot falsely claim target KV capacity.
        borrowed_target_kv_bytes: 0,
    }
}

fn empty_lane_memory(request: &ChildLaneRequest) -> LaneMemoryReceipt {
    LaneMemoryReceipt {
        run_id: request.run_id.clone(),
        phase: request.phase,
        workload: request.workload.key,
        lane: request.lane,
        repetition: request.repetition,
        child_process_id: std::process::id(),
        assistant_warm_target_free: false,
        snapshots: Vec::new(),
        assistant_memory: AssistantMemory::default(),
        assistant_load_delta: IncrementalMemory::default(),
        peak_experiment_delta: IncrementalMemory::default(),
        routed_experts_after_target_warm: None,
        routed_experts_after_generation: None,
        monitor_samples: Vec::new(),
        process_monitor_samples: Vec::new(),
        kill_reason: None,
    }
}

fn finish_lane_memory(
    request: &ChildLaneRequest,
    baseline: &MemorySnapshot,
    snapshots: Vec<MemorySnapshot>,
    assistant_memory: AssistantMemory,
    assistant_warm_target_free: bool,
    routed_experts_after_target_warm: Option<
        camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
    >,
    routed_experts_after_generation: Option<
        camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot,
    >,
    monitor_results: MonitorResults,
) -> LaneMemoryReceipt {
    let assistant_after = snapshots
        .iter()
        .find(|snapshot| snapshot.phase.ends_with(":assistant_resident_warm"))
        .unwrap_or(baseline);
    LaneMemoryReceipt {
        run_id: request.run_id.clone(),
        phase: request.phase,
        workload: request.workload.key,
        lane: request.lane,
        repetition: request.repetition,
        child_process_id: std::process::id(),
        assistant_warm_target_free,
        assistant_load_delta: if assistant_memory.file_bytes == 0 {
            IncrementalMemory::default()
        } else {
            IncrementalMemory::between(baseline, assistant_after)
        },
        peak_experiment_delta: peak_incremental_memory(
            baseline,
            &snapshots,
            &monitor_results.samples,
            &monitor_results.process_samples,
        ),
        snapshots,
        assistant_memory,
        routed_experts_after_target_warm,
        routed_experts_after_generation,
        monitor_samples: monitor_results.samples,
        process_monitor_samples: monitor_results.process_samples,
        kill_reason: monitor_results.kill_reason,
    }
}

fn warm_assistant_once(
    assistant: &mut camelid::metal::Gemma4MtpAssistantMetal,
    control: &ExperimentControl,
) -> Result<bool, String> {
    if control.should_abort() {
        return Ok(false);
    }
    let timing = assistant
        .warm_target_free()
        .map_err(|error| format!("target-free assistant warmup: {error}"))?;
    if timing.wall_us == 0 || timing.gpu_us == 0 || timing.gpu_us > timing.wall_us {
        return Err(format!(
            "target-free assistant warmup returned invalid timing wall={}us gpu={}us",
            timing.wall_us, timing.gpu_us
        ));
    }
    if control.should_abort() {
        return Ok(false);
    }
    Ok(true)
}

fn execute_lane_child(request: &ChildLaneRequest) -> Result<ChildLaneResult, String> {
    apply_child_environment(request)?;
    #[cfg(unix)]
    if unsafe { libc::getppid() as u32 } != request.parent_process_id {
        return Err(format!(
            "lane child parent PID {} does not match request {}",
            unsafe { libc::getppid() },
            request.parent_process_id
        ));
    }

    // The hybrid mapped-cold lane requires the retained read-only `.cghost`
    // mapping. `evict_page_cache=true` would set F_NOCACHE and deliberately
    // discard that owner, so this exact profile always keeps the file pager.
    let evict_page_cache = false;
    let runtime = camelid::gemma4_runtime::Gemma4Runtime::load_ghost_moe(
        &request.target_runtime.runtime_gguf_path,
        &request.target_runtime.cghost_path,
        request.target_runtime.expert_cache_mib,
        evict_page_cache,
    )
    .map_err(|error| format!("load internal target pair: {error}"))?;
    let baseline = capture_memory_snapshot(&format!("{}:pre_assistant", request.run_id), 0)?;
    let monitor = MonitorGuard::start(baseline.clone());
    let control = monitor.control();
    let started = Instant::now();
    let mut snapshots = vec![baseline.clone()];
    let mut assistant_memory = AssistantMemory::default();
    let mut assistant_warm_completed = false;
    let mut target_shared_kv_used = false;
    let mut routed_experts_after_target_warm = None;
    let mut routed_experts_after_generation = None;

    macro_rules! child_try {
        ($expression:expr, $label:expr) => {{
            match $expression {
                Ok(value) => value,
                Err(error) => {
                    let error = format!("{}: {error}", $label);
                    // Preserve a structured latest-at-failure expert receipt
                    // (including a >32 union fail-close) before the child
                    // report is finalized. This is a read-only snapshot and
                    // does not retry or touch an expert record.
                    if routed_experts_after_generation.is_none() {
                        routed_experts_after_generation =
                            runtime.routed_expert_residency_snapshot();
                    }
                    if let Ok(snapshot) = capture_memory_snapshot(
                        &format!("{}:failed", request.run_id),
                        started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                    ) {
                        snapshots.push(snapshot);
                    }
                    let monitor_results = monitor.finish();
                    let mut memory = finish_lane_memory(
                        request,
                        &baseline,
                        snapshots,
                        assistant_memory,
                        assistant_warm_completed,
                        routed_experts_after_target_warm.clone(),
                        routed_experts_after_generation.clone(),
                        monitor_results,
                    );
                    if memory.kill_reason.is_none() {
                        memory.kill_reason = Some(KillReason::ChildFailure(error.clone()));
                    }
                    return Ok(ChildLaneResult {
                        schema_version: REPORT_SCHEMA_VERSION,
                        run_id: request.run_id.clone(),
                        child_process_id: std::process::id(),
                        run: None,
                        memory,
                        target_shared_kv_used,
                        error: Some(error),
                    });
                }
            }
        }};
    }

    let mut assistant = if request.plan.load_assistant {
        let loaded = child_try!(
            camelid::metal::Gemma4MtpAssistantMetal::load(&request.assistant_path),
            "load resident internal MTP assistant"
        );
        assistant_memory =
            assistant_memory_from_ledger(loaded.resident_ledger(), snapshots[0].system.page_size);
        let snapshot = child_try!(
            capture_memory_snapshot(
                &format!("{}:assistant_resident", request.run_id),
                started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            ),
            "capture resident-assistant memory"
        );
        snapshots.push(snapshot);
        Some(loaded)
    } else {
        None
    };

    if request.plan.warm_assistant && !control.should_abort() {
        let loaded = child_try!(
            assistant
                .as_mut()
                .ok_or_else(|| "no loaded assistant".to_string()),
            "assistant warm requested"
        );
        assistant_warm_completed = child_try!(
            warm_assistant_once(loaded, &control),
            "target-free assistant warmup"
        );
    }
    // Measure the assistant's resident/warm delta before touching the target's
    // persistent LFU/victim/page-cache state. This is the I/M memory cost that
    // the economics comparison must price.
    let assistant_warm_snapshot = child_try!(
        capture_memory_snapshot(
            &format!("{}:assistant_resident_warm", request.run_id),
            started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        ),
        "capture target-free assistant-warm memory"
    );
    snapshots.push(assistant_warm_snapshot);

    // Every lane now performs exactly one identical canonical target warmup.
    // I/M's preceding assistant warmup is target-free and cannot perturb target
    // expert LFU, victim-ring state, sequence KV, or target-backed page history.
    if !control.should_abort() {
        let target_warm_completed = child_try!(
            warm_target_runtime(&runtime, || control.should_abort()),
            "canonical target warmup"
        );
        if !target_warm_completed && !control.should_abort() {
            child_try!(
                Err::<(), _>("target warmup ended without an abort receipt".to_string()),
                "canonical target warmup"
            );
        }
    }
    let target_warm_experts = child_try!(
        runtime.routed_expert_residency_snapshot().ok_or_else(|| {
            "target routed-expert residency is unavailable after warmup".to_string()
        }),
        "capture post-target-warm expert residency"
    );
    child_try!(
        validate_routed_expert_snapshot("post-target-warm snapshot", &target_warm_experts),
        "validate post-target-warm expert residency"
    );
    child_try!(
        validate_exact_target_hybrid_experts(&target_warm_experts),
        "bind exact 128-canonical/48-hot/mapped-cold expert residency"
    );
    routed_experts_after_target_warm = Some(target_warm_experts);
    child_try!(
        runtime
            .begin_routed_expert_telemetry_interval()
            .then_some(())
            .ok_or_else(|| {
                "target routed-identity telemetry interval could not begin after warmup".to_string()
            }),
        "begin measured routed-identity telemetry interval"
    );
    let post_target_warm_snapshot = child_try!(
        capture_memory_snapshot(
            &format!("{}:target_warm", request.run_id),
            started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        ),
        "capture post-target-warm memory"
    );
    snapshots.push(post_target_warm_snapshot);

    if control.should_abort() {
        let monitor_results = monitor.finish();
        let memory = finish_lane_memory(
            request,
            &baseline,
            snapshots,
            assistant_memory,
            assistant_warm_completed,
            routed_experts_after_target_warm,
            routed_experts_after_generation,
            monitor_results,
        );
        return Ok(ChildLaneResult {
            schema_version: REPORT_SCHEMA_VERSION,
            run_id: request.run_id.clone(),
            child_process_id: std::process::id(),
            run: None,
            memory,
            target_shared_kv_used: false,
            error: None,
        });
    }

    let prompt_tokens = child_try!(
        runtime
            .tokenizer()
            .encode(&request.workload.prompt, true, true)
            .map_err(|error| error.to_string()),
        "tokenize measured workload"
    );
    let eot = runtime.stop_token_ids();
    let (mut kc, mut vc) = runtime.empty_kv_caches();
    let max_new = child_try!(
        usize::try_from(request.requested_output_tokens)
            .map_err(|_| "requested output token budget exceeds usize".to_string()),
        "convert measured output budget"
    );
    let logits = child_try!(
        runtime.prefill_tokens_cancellable_experiment(
            &prompt_tokens,
            &mut kc,
            &mut vc,
            max_new.saturating_sub(1),
            || control.should_abort(),
        ),
        "measured target prefill"
    );
    let Some(logits) = logits else {
        let aborted_snapshot = child_try!(
            capture_memory_snapshot(
                &format!("{}:aborted_prefill", request.run_id),
                started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
            ),
            "capture aborted-prefill memory"
        );
        snapshots.push(aborted_snapshot);
        let monitor_results = monitor.finish();
        let memory = finish_lane_memory(
            request,
            &baseline,
            snapshots,
            assistant_memory,
            assistant_warm_completed,
            routed_experts_after_target_warm,
            routed_experts_after_generation,
            monitor_results,
        );
        return Ok(ChildLaneResult {
            schema_version: REPORT_SCHEMA_VERSION,
            run_id: request.run_id.clone(),
            child_process_id: std::process::id(),
            run: None,
            memory,
            target_shared_kv_used: false,
            error: None,
        });
    };

    let mut run = match request.plan.driver {
        LaneDriver::CurrentSpecDecodeGenerate => {
            let result = child_try!(
                runtime.spec_decode_generate_observed_experiment_cancellable(
                    &mut kc,
                    &mut vc,
                    logits,
                    &prompt_tokens,
                    &eot,
                    max_new,
                    || control.should_abort(),
                ),
                "observed shipping n-gram lane"
            );
            child_try!(
                ngram_lane_run(
                    request.run_id.clone(),
                    request.phase,
                    request.workload.key,
                    request.lane,
                    request.repetition,
                    request.requested_output_tokens,
                    &prompt_tokens,
                    &eot,
                    result,
                ),
                "convert shipping n-gram receipt"
            )
        }
        LaneDriver::NativeMtpExperiment => {
            let loaded = child_try!(
                assistant.as_mut().ok_or_else(|| {
                    "M lane entered without the resident official assistant".to_string()
                }),
                "enter native MTP driver"
            );
            let result = child_try!(
                runtime.generate_mtp_assistant_experiment_cancellable(
                    loaded,
                    &mut kc,
                    &mut vc,
                    logits,
                    prompt_tokens.len(),
                    &eot,
                    max_new,
                    || control.should_abort(),
                    None,
                ),
                "native MTP measured lane"
            );
            let measured_round_invoked_assistant =
                result.rounds.iter().any(|round| !round.bootstrap);
            let measured_ledger = loaded.last_proposal_ledger();
            if measured_round_invoked_assistant && measured_ledger.is_none() {
                child_try!(
                    Err::<(), _>(
                        "measured M round invoked the assistant without a proposal ledger"
                            .to_string(),
                    ),
                    "account measured shared target KV"
                );
            }
            if let Some(ledger) = measured_ledger {
                if ledger.borrowed_target_kv_capacity_bytes == 0 {
                    child_try!(
                        Err::<(), _>(
                            "measured M proposal reported zero borrowed target-KV capacity"
                                .to_string(),
                        ),
                        "account measured shared target KV"
                    );
                }
                assistant_memory.borrowed_target_kv_bytes =
                    ledger.borrowed_target_kv_capacity_bytes;
                target_shared_kv_used = true;
            }
            child_try!(
                mtp_lane_run(
                    request.run_id.clone(),
                    request.phase,
                    request.workload.key,
                    request.repetition,
                    request.requested_output_tokens,
                    &prompt_tokens,
                    &eot,
                    result,
                ),
                "convert native MTP receipt"
            )
        }
    };
    let post_generation_experts = child_try!(
        runtime.routed_expert_residency_snapshot().ok_or_else(|| {
            "target routed-expert residency is unavailable after generation".to_string()
        }),
        "capture post-generation expert residency"
    );
    let target_warm_experts = child_try!(
        routed_experts_after_target_warm
            .as_ref()
            .ok_or_else(|| "post-target-warm expert residency was lost".to_string()),
        "bind routed-expert residency transition"
    );
    child_try!(
        validate_routed_expert_transition(target_warm_experts, &post_generation_experts),
        "validate routed-expert residency transition"
    );
    run.routed_experts_after_generation = Some(post_generation_experts.clone());
    routed_experts_after_generation = Some(post_generation_experts);
    let post_generation_snapshot = child_try!(
        capture_memory_snapshot(
            &format!("{}:post_generation", request.run_id),
            started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        ),
        "capture post-generation memory"
    );
    snapshots.push(post_generation_snapshot);
    let monitor_results = monitor.finish();
    let memory = finish_lane_memory(
        request,
        &baseline,
        snapshots,
        assistant_memory,
        assistant_warm_completed,
        routed_experts_after_target_warm,
        routed_experts_after_generation,
        monitor_results,
    );
    Ok(ChildLaneResult {
        schema_version: REPORT_SCHEMA_VERSION,
        run_id: request.run_id.clone(),
        child_process_id: std::process::id(),
        run: Some(run),
        memory,
        target_shared_kv_used,
        error: None,
    })
}

fn explicit_assistant_model_path() -> Result<PathBuf, String> {
    let value = std::env::var_os(ASSISTANT_PATH_ENV)
        .ok_or_else(|| format!("set {ASSISTANT_PATH_ENV} to the exact official assistant file"))?;
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{ASSISTANT_PATH_ENV} must be an absolute path"));
    }
    if !path.is_file() {
        return Err(format!("assistant file does not exist: {}", path.display()));
    }
    let bytes = std::fs::metadata(&path)
        .map_err(|error| format!("stat assistant file: {error}"))?
        .len();
    if bytes == 0 {
        return Err("assistant file is empty".to_string());
    }
    Ok(path)
}

fn explicit_native_admission_run_nonce() -> Result<String, String> {
    let nonce = std::env::var(NATIVE_ADMISSION_RUN_NONCE_ENV).map_err(|error| {
        format!(
            "set {NATIVE_ADMISSION_RUN_NONCE_ENV} to the same fresh nonce used by the separate native admission test: {error}"
        )
    })?;
    validate_native_admission_run_nonce(&nonce)?;
    Ok(nonce)
}

#[test]
#[ignore = "private subprocess entry point for the isolated parent experiment"]
fn gemma4_mtp_assistant_lane_child() {
    let Some(request_path) = std::env::var_os(CHILD_REQUEST_ENV).map(PathBuf::from) else {
        eprintln!("SKIP: private MTP lane child has no structured request");
        return;
    };
    let result_path = std::env::var_os(CHILD_RESULT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("private MTP lane child has no {CHILD_RESULT_ENV}"));
    let request: ChildLaneRequest = read_json(&request_path)
        .unwrap_or_else(|error| panic!("private MTP lane request is invalid: {error}"));
    let result = match execute_lane_child(&request) {
        Ok(result) => result,
        Err(error) => {
            let mut memory = empty_lane_memory(&request);
            memory.kill_reason = Some(KillReason::ChildFailure(error.clone()));
            ChildLaneResult {
                schema_version: REPORT_SCHEMA_VERSION,
                run_id: request.run_id.clone(),
                child_process_id: std::process::id(),
                run: None,
                memory,
                target_shared_kv_used: false,
                error: Some(error),
            }
        }
    };
    atomic_write_json(&result_path, &result)
        .unwrap_or_else(|error| panic!("publish private MTP lane receipt: {error}"));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "default-off load-only residency probe; never tokenizes, prefills, steps, proposes, or generates"]
fn gemma4_mtp_assistant_load_only_probe() {
    if std::env::var(LOAD_ONLY_PROBE_ENABLE_ENV).ok().as_deref() != Some("1") {
        eprintln!("SKIP: set {LOAD_ONLY_PROBE_ENABLE_ENV}=1 for the isolated load-only probe");
        return;
    }
    let report_path = std::env::var_os(LOAD_ONLY_PROBE_REPORT_PATH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("set {LOAD_ONLY_PROBE_REPORT_PATH_ENV} to an absolute checkpoint path")
        });
    if !report_path.is_absolute() {
        panic!("{LOAD_ONLY_PROBE_REPORT_PATH_ENV} must be absolute");
    }

    let assistant_path = explicit_assistant_model_path()
        .and_then(|path| canonical_internal_timed_file(&path, "load-only MTP assistant"))
        .unwrap_or_else(|error| panic!("load-only assistant path refused: {error}"));
    let mut target_runtime = target_runtime_config_from_environment()
        .unwrap_or_else(|error| panic!("load-only target wiring refused: {error}"));
    target_runtime.runtime_gguf_path = canonical_internal_timed_file(
        &target_runtime.runtime_gguf_path,
        "load-only target runtime GGUF",
    )
    .unwrap_or_else(|error| panic!("load-only target path refused: {error}"));
    target_runtime.cghost_path =
        canonical_internal_timed_file(&target_runtime.cghost_path, "load-only target cghost")
            .unwrap_or_else(|error| panic!("load-only cghost path refused: {error}"));
    apply_load_only_environment(&target_runtime, &assistant_path)
        .unwrap_or_else(|error| panic!("load-only environment refused: {error}"));

    let started = Instant::now();
    let clean = capture_memory_snapshot("clean_baseline", 0)
        .unwrap_or_else(|error| panic!("capture load-only clean baseline: {error}"));
    let baseline_system = clean.system.clone();
    let monitor = MonitorGuard::start(clean.clone());
    let mut report = LoadOnlyProbeReport {
        schema_version: LOAD_ONLY_REPORT_SCHEMA_VERSION,
        process_id: std::process::id(),
        started_unix_ms: unix_time_ms(),
        completed_unix_ms: None,
        assistant_path: assistant_path.clone(),
        target_runtime: target_runtime.clone(),
        forced_file_backed_no_copy_head: true,
        assistant_ledger: None,
        target_final_ledger: None,
        checkpoints: Vec::new(),
        monitor_samples: Vec::new(),
        process_monitor_samples: Vec::new(),
        monitor_kill_reason: None,
        soak_seconds: LOAD_ONLY_SOAK_SECONDS,
        soak_started_elapsed_ms: None,
        soak_monitor_start_sample: None,
        soak_completed_seconds: 0,
        minimum_free_bytes: baseline_system.free_bytes,
        minimum_reclaimable_headroom_bytes: baseline_system.reclaimable_headroom_bytes,
        all_pressure_samples_normal: baseline_system.pressure.is_normal(),
        soak_minimum_free_bytes: None,
        soak_minimum_reclaimable_headroom_bytes: None,
        soak_all_pressure_samples_normal: true,
        soak_required_head_total_pages: None,
        soak_min_required_head_resident_pages: None,
        soak_all_required_head_pages_resident: true,
        completed: false,
        failure: None,
        assistant_warmups: 0,
        assistant_proposals: 0,
        tokenizer_calls: 0,
        target_prefills: 0,
        target_steps: 0,
        target_generations: 0,
        target_kv_borrows: 0,
    };
    if let Err(error) = checkpoint_load_only_probe(
        &report_path,
        &mut report,
        &baseline_system,
        &monitor,
        clean,
        None,
    ) {
        let error = persist_load_only_failure(&report_path, &mut report, &monitor, error);
        panic!("load-only clean baseline refused: {error}");
    }

    let mut assistant: Option<camelid::metal::Gemma4MtpAssistantMetal> = None;
    let mut observe_target = |phase: camelid::gemma4_runtime::Gemma4GhostLoadPhase,
                              ledger: camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger|
     -> camelid::Result<()> {
        let phase_name = load_only_target_phase_name(phase).to_string();
        let assistant_barrier = phase
            == camelid::gemma4_runtime::Gemma4GhostLoadPhase::AssistantResidencyBarrierComplete;
        if assistant_barrier && (assistant.is_none() || report.assistant_ledger.is_none()) {
            return Err(camelid::BackendError::InvalidModelMetadata(
                "assistant residency barrier emitted before the assistant was retained".into(),
            ));
        }
        let observation = LoadOnlyTargetObservation {
            phase: phase_name.clone(),
            ledger: ledger.into(),
        };
        capture_load_only_checkpoint(
            &report_path,
            &mut report,
            &baseline_system,
            &monitor,
            started,
            if assistant_barrier {
                "assistant_resident".to_string()
            } else {
                phase_name
            },
            Some(observation),
        )
        .map_err(|error| {
            camelid::BackendError::InvalidModelMetadata(format!(
                "load-only target checkpoint refused: {error}"
            ))
        })?;
        if let Some(error) = load_only_target_phase_violation(phase, &ledger) {
            return Err(camelid::BackendError::InvalidModelMetadata(error));
        }
        if phase == camelid::gemma4_runtime::Gemma4GhostLoadPhase::EmptyExpertSlotsAllocated {
            if assistant.is_some() {
                return Err(camelid::BackendError::InvalidModelMetadata(
                    "load-only assistant barrier was entered more than once".into(),
                ));
            }
            // This callback is the exact attribution barrier: target metadata,
            // common weights, KV/scratch, and empty slots are already retained,
            // while the required no-copy head page window has not been faulted.
            // Loading only maps/mlocks the assistant and allocates its fixed
            // scratch; it never warms, proposes, borrows KV, or calls target math.
            let loaded = camelid::metal::Gemma4MtpAssistantMetal::load(&assistant_path).map_err(
                |error| {
                    camelid::BackendError::InvalidModelMetadata(format!(
                        "load resident assistant at empty-slot barrier: {error}"
                    ))
                },
            )?;
            let resident = loaded.resident_ledger();
            if resident.locked_bytes == 0
                || resident.locked_bytes != resident.mapped_bytes
                || resident.resident_pages == 0
                || resident.resident_pages != resident.total_pages
            {
                return Err(camelid::BackendError::InvalidModelMetadata(format!(
                    "assistant native locked-residency receipt is inconsistent: {resident:?}"
                )));
            }
            report.assistant_ledger = Some(resident.into());
            assistant = Some(loaded);
        }
        Ok(())
    };
    let target_result = camelid::gemma4_runtime::Gemma4Runtime::load_ghost_moe_load_only_observed(
        &target_runtime.runtime_gguf_path,
        &target_runtime.cghost_path,
        target_runtime.expert_cache_mib,
        false,
        &mut observe_target,
    );
    drop(observe_target);
    let target = match target_result {
        Ok(target) => target,
        Err(error) => {
            let error = persist_load_only_failure(
                &report_path,
                &mut report,
                &monitor,
                format!("observed target load: {error}"),
            );
            panic!("load-only target failed: {error}");
        }
    };
    let assistant = match assistant {
        Some(assistant) => assistant,
        None => {
            let error = persist_load_only_failure(
                &report_path,
                &mut report,
                &monitor,
                "observed target load never entered the empty-slot assistant barrier",
            );
            panic!("load-only assistant barrier failed: {error}");
        }
    };
    let observed_target_phases = report
        .checkpoints
        .iter()
        .filter_map(|checkpoint| {
            checkpoint
                .target_observation
                .as_ref()
                .map(|observation| observation.phase.clone())
        })
        .collect::<Vec<_>>();
    let expected_target_phases = [
        "target_object_metadata_ready",
        "common_q4_weights_allocated",
        "common_resident_layer_aux_allocated",
        "common_kv_scratch_allocated",
        "empty_expert_slots_allocated",
        "assistant_residency_barrier_complete",
        "required_target_pages_head_ready",
        "expert_slot_tables_compute_bound",
        "target_load_complete",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if observed_target_phases != expected_target_phases {
        let error = persist_load_only_failure(
            &report_path,
            &mut report,
            &monitor,
            format!(
                "observed target phase sequence {observed_target_phases:?}, expected {expected_target_phases:?}"
            ),
        );
        panic!("load-only target phase receipt refused: {error}");
    }
    let observed_load_phases = report
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.snapshot.phase.clone())
        .collect::<Vec<_>>();
    let expected_load_phases = [
        "clean_baseline",
        "target_object_metadata_ready",
        "common_q4_weights_allocated",
        "common_resident_layer_aux_allocated",
        "common_kv_scratch_allocated",
        "empty_expert_slots_allocated",
        "assistant_resident",
        "required_target_pages_head_ready",
        "expert_slot_tables_compute_bound",
        "target_load_complete",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    if observed_load_phases != expected_load_phases {
        let error = persist_load_only_failure(
            &report_path,
            &mut report,
            &monitor,
            format!(
                "load-only attribution order {observed_load_phases:?}, expected {expected_load_phases:?}"
            ),
        );
        panic!("load-only attribution order refused: {error}");
    }
    let final_ledger = target.allocation_ledger();
    report.target_final_ledger = Some(final_ledger.into());
    let components = target.metal_components();
    let final_phase_error = load_only_target_phase_violation(
        camelid::gemma4_runtime::Gemma4GhostLoadPhase::Complete,
        &final_ledger,
    );
    if final_phase_error.is_some()
        || std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT")
            .ok()
            .as_deref()
            != Some("0")
        || !report.forced_file_backed_no_copy_head
        || !components.experts
        || !components.common
        || !components.head
    {
        let error = persist_load_only_failure(
            &report_path,
            &mut report,
            &monitor,
            format!(
                "load-only target contract drift: phase_error={final_phase_error:?} ledger={final_ledger:?} components={components:?} head_resident_env={:?}",
                std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT")
            ),
        );
        panic!("load-only target contract refused: {error}");
    }
    if let Err(error) = capture_load_only_residency_checkpoint(
        &report_path,
        &mut report,
        &baseline_system,
        &monitor,
        started,
        &target,
        "both_models_retained_soak_start",
    ) {
        let error = persist_load_only_failure(&report_path, &mut report, &monitor, error);
        panic!("load-only soak start refused: {error}");
    }

    for second in 1..=LOAD_ONLY_SOAK_SECONDS {
        // Keep both owners live for the complete soak. black_box prevents a
        // future optimizing test build from shortening either lexical lifetime.
        std::hint::black_box((&assistant, &target));
        thread::sleep(MONITOR_PERIOD);
        report.soak_completed_seconds = second;
        if let Err(error) = capture_load_only_residency_checkpoint(
            &report_path,
            &mut report,
            &baseline_system,
            &monitor,
            started,
            &target,
            format!("both_models_retained_soak_{second:02}"),
        ) {
            let error = persist_load_only_failure(&report_path, &mut report, &monitor, error);
            panic!("load-only soak refused at second {second}: {error}");
        }
    }

    let monitor_results = monitor.finish();
    report.monitor_samples = monitor_results.samples;
    report.process_monitor_samples = monitor_results.process_samples;
    report.monitor_kill_reason = monitor_results.kill_reason;
    for (index, sample) in report.monitor_samples.iter().enumerate() {
        report.minimum_free_bytes = report.minimum_free_bytes.min(sample.free_bytes);
        report.minimum_reclaimable_headroom_bytes = report
            .minimum_reclaimable_headroom_bytes
            .min(sample.reclaimable_headroom_bytes);
        report.all_pressure_samples_normal &= sample.pressure.is_normal();
        if report
            .soak_monitor_start_sample
            .is_some_and(|started| index >= started)
        {
            report.soak_minimum_free_bytes = Some(
                report
                    .soak_minimum_free_bytes
                    .map_or(sample.free_bytes, |minimum| minimum.min(sample.free_bytes)),
            );
            report.soak_minimum_reclaimable_headroom_bytes = Some(
                report
                    .soak_minimum_reclaimable_headroom_bytes
                    .map_or(sample.reclaimable_headroom_bytes, |minimum| {
                        minimum.min(sample.reclaimable_headroom_bytes)
                    }),
            );
            report.soak_all_pressure_samples_normal &= sample.pressure.is_normal();
        }
    }
    if let Some(reason) = report.monitor_kill_reason.as_ref() {
        report.failure = Some(format!("load-only monitor stopped the probe: {reason:?}"));
    } else if let Some(reason) = load_only_safety_violation(
        &baseline_system,
        &report.checkpoints,
        &report.monitor_samples,
    ) {
        report.failure = Some(reason);
    }
    if report.failure.is_none()
        && (report.soak_completed_seconds != LOAD_ONLY_SOAK_SECONDS
            || report.soak_minimum_free_bytes.is_none()
            || report.soak_minimum_reclaimable_headroom_bytes.is_none()
            || !report.soak_all_pressure_samples_normal
            || report.soak_required_head_total_pages.is_none()
            || report.soak_min_required_head_resident_pages
                != report.soak_required_head_total_pages
            || !report.soak_all_required_head_pages_resident)
    {
        report.failure = Some(format!(
            "incomplete/non-normal/nonresident soak: completed={}/{} minimum_free={:?} minimum_reclaimable_headroom={:?} all_normal={} head_min={:?} head_total={:?} head_all_resident={}",
            report.soak_completed_seconds,
            LOAD_ONLY_SOAK_SECONDS,
            report.soak_minimum_free_bytes,
            report.soak_minimum_reclaimable_headroom_bytes,
            report.soak_all_pressure_samples_normal,
            report.soak_min_required_head_resident_pages,
            report.soak_required_head_total_pages,
            report.soak_all_required_head_pages_resident
        ));
    }
    if report.failure.is_none() {
        report.failure = load_only_operation_violation(&report);
    }
    if let Some(error) = report.failure.as_ref() {
        atomic_write_json(&report_path, &report)
            .unwrap_or_else(|write_error| panic!("{error}; checkpoint failed: {write_error}"));
        panic!("load-only probe failed closed: {error}");
    }
    report.completed = true;
    report.completed_unix_ms = Some(unix_time_ms());
    atomic_write_json(&report_path, &report)
        .unwrap_or_else(|error| panic!("publish completed load-only report: {error}"));
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize load-only report")
    );
}

#[test]
#[ignore = "default-off isolated MTP experiment; requires explicit opt-in and exact assistant path"]
fn gemma4_mtp_assistant_experiment() {
    if std::env::var(EXPERIMENT_ENABLE_ENV).ok().as_deref() != Some("1") {
        eprintln!("SKIP: set {EXPERIMENT_ENABLE_ENV}=1 for the isolated MTP experiment");
        return;
    }
    let integration_executable = std::env::current_exe()
        .unwrap_or_else(|error| panic!("resolve MTP integration executable: {error}"));
    let integration_executable =
        canonical_internal_timed_file(&integration_executable, "MTP integration executable")
            .unwrap_or_else(|error| panic!("MTP experiment executable refused: {error}"));
    let integration_executable_sha256 =
        file_sha256(&integration_executable, "MTP integration executable")
            .unwrap_or_else(|error| panic!("MTP integration executable identity refused: {error}"));
    let native_admission_run_nonce = explicit_native_admission_run_nonce()
        .unwrap_or_else(|error| panic!("native MTP admission nonce refused: {error}"));
    // Numerical admission is checked before the slow full-target T7 pair gate
    // and before any timed child can exist. Children receive the parsed receipt
    // over IPC; they never reopen this mutable path.
    let native_admission = load_native_admission_receipt(
        &integration_executable,
        &integration_executable_sha256,
        &native_admission_run_nonce,
    )
    .unwrap_or_else(|error| panic!("native MTP admission refused: {error}"));
    let assistant_path = explicit_assistant_model_path()
        .unwrap_or_else(|error| panic!("MTP experiment refused before model load: {error}"));
    let assistant_path = canonical_internal_timed_file(&assistant_path, "MTP assistant")
        .unwrap_or_else(|error| panic!("MTP experiment timed path refused: {error}"));
    let mut target_runtime = target_runtime_config_from_environment()
        .unwrap_or_else(|error| panic!("MTP experiment target wiring refused: {error}"));
    target_runtime.runtime_gguf_path =
        canonical_internal_timed_file(&target_runtime.runtime_gguf_path, "target runtime GGUF")
            .unwrap_or_else(|error| panic!("MTP experiment timed path refused: {error}"));
    target_runtime.cghost_path =
        canonical_internal_timed_file(&target_runtime.cghost_path, "target cghost")
            .unwrap_or_else(|error| panic!("MTP experiment timed path refused: {error}"));

    let pilot_only = match std::env::var(PILOT_ONLY_ENV) {
        Err(std::env::VarError::NotPresent) => pilot_only_from(None),
        Ok(value) => pilot_only_from(Some(&value)),
        Err(error) => panic!("read {PILOT_ONLY_ENV}: {error}"),
    }
    .unwrap_or_else(|error| panic!("{error}"));
    let report_path = std::env::var_os(REPORT_PATH_ENV).map(PathBuf::from);
    if report_path.as_ref().is_some_and(|path| !path.is_absolute()) {
        panic!("{REPORT_PATH_ENV} must be absolute");
    }
    if pilot_only && report_path.is_none() {
        panic!("{PILOT_ONLY_ENV}=1 requires {REPORT_PATH_ENV} for atomic child checkpoints");
    }

    // This is the only operation allowed to touch the full T7 source target.
    // It completes before any child monitor, memory baseline, target load, or
    // generation timer exists; children receive only the structured receipt.
    let pair_started = Instant::now();
    let pairing = establish_pairing_evidence(&target_runtime, &assistant_path)
        .unwrap_or_else(|error| panic!("official MTP pair gate refused: {error}"));
    let pair_gate_wall_us = pair_started
        .elapsed()
        .as_micros()
        .min(u128::from(u64::MAX))
        .max(1) as u64;

    let lanes = vec![Lane::NgramBaseline, Lane::NgramAssistantIdle, Lane::Mtp];
    let lane_plans = lanes
        .iter()
        .copied()
        .map(LaneExecutionPlan::for_lane)
        .collect::<Vec<_>>();
    validate_lane_plans(&lane_plans)
        .unwrap_or_else(|error| panic!("MTP experiment lane wiring refused: {error}"));
    let request = ExperimentRequest {
        assistant_path,
        target_runtime,
        native_admission,
        integration_executable_sha256,
        native_admission_run_nonce,
        pairing,
        pair_gate_wall_us,
        workloads: decision_workloads(),
        lanes,
        lane_plans,
        pilot_tokens: PILOT_TOKENS,
        pilot_repetitions: if pilot_only { 1 } else { PILOT_REPETITIONS },
        pilot_only,
        report_path: report_path.clone(),
        matrix_tokens: std::env::var(MATRIX_TOKENS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MATRIX_TOKENS),
        matrix_repetitions: std::env::var(MATRIX_REPETITIONS_ENV)
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_MATRIX_REPETITIONS),
        child_timeout_secs: std::env::var(CHILD_TIMEOUT_SECS_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_CHILD_TIMEOUT_SECS),
        native_admission_must_pass: true,
        pair_gate_must_pass: true,
        use_target_shared_kv: true,
        target_must_verify_every_proposal: true,
    };

    let mut adapter = NativeChildProcessAdapter;
    let result = adapter.run(&request);
    match result {
        Ok(report) => {
            report
                .validate()
                .unwrap_or_else(|error| panic!("MTP experiment report failed closed: {error}"));
            if let Some(path) = report_path.as_ref() {
                atomic_write_json(path, &report)
                    .unwrap_or_else(|error| panic!("persist MTP experiment report: {error}"));
                eprintln!("[mtp-experiment] report={}", path.display());
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("serialize MTP experiment report")
            );
            if let Some(reason) = &report.kill_reason {
                eprintln!("[mtp-experiment] killed safely: {reason:?}");
            }
        }
        Err(error) => panic!(
            "MTP experiment stopped at the native adapter boundary; no measurements were fabricated: {error}"
        ),
    }
}

#[cfg(test)]
mod pure_tests {
    use super::*;

    fn system_sample(
        elapsed_ms: u64,
        swapins_bytes: u64,
        swapouts_bytes: u64,
        swap_used_bytes: u64,
        pressure: MemoryPressure,
    ) -> SystemVmSnapshot {
        SystemVmSnapshot {
            elapsed_ms,
            page_size: 16_384,
            free_bytes: 4 << 30,
            active_bytes: 8 << 30,
            inactive_bytes: 1 << 30,
            reclaimable_headroom_bytes: 5 << 30,
            wired_bytes: 2 << 30,
            compressor_occupied_bytes: 0,
            compressed_logical_bytes: 0,
            pageins_bytes: 0,
            pageouts_bytes: 0,
            swapins_bytes,
            swapouts_bytes,
            swap_used_bytes,
            swap_total_bytes: 4 << 30,
            pressure,
        }
    }

    fn round(requested_k: u32, proposed_k: u32, verifier_k: u32, accepted: u32) -> RoundTelemetry {
        RoundTelemetry {
            workload: Workload::Copy,
            lane: Lane::Mtp,
            repetition: 0,
            round_index: 0,
            prefix_tokens_before: 100,
            proposal_source: ProposalSource::MtpAssistant,
            assistant_invoked: proposed_k > 0,
            requested_k,
            proposed_k,
            verifier_k,
            budget_truncated: false,
            accepted_drafts: accepted,
            useful_accepted_drafts: accepted,
            emitted_target_tokens: accepted + 1,
            visible_output_tokens: accepted + 1,
            draft_wall_us: 400,
            draft_gpu_us: 300,
            verify_wall_us: 4_000,
            verify_gpu_us: 3_000,
            round_wall_us: 4_500,
        }
    }

    fn routed_expert_snapshot() -> camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot {
        use camelid::gemma4_runtime::{
            Gemma4RoutedExpertLayerResidencySnapshot, Gemma4RoutedExpertResidencySnapshot,
            Gemma4RoutedExpertSlotStatsSnapshot,
        };
        let stats = Gemma4RoutedExpertSlotStatsSnapshot {
            route_lookups: 2,
            misses: 2,
            direct_reads: 2,
            direct_read_bytes: 200,
            ..Default::default()
        };
        Gemma4RoutedExpertResidencySnapshot {
            layer_count: 1,
            slot_record_bytes: 100,
            slot_stride_bytes: 128,
            base_slot_capacity: 8,
            base_slot_capacity_bytes: 1_024,
            physical_base_slot_budget: 8,
            physical_base_slot_budget_bytes: 1_024,
            occupied_base_slots: 2,
            occupied_base_payload_bytes: 200,
            occupied_base_touched_bytes: 256,
            per_layer: vec![Gemma4RoutedExpertLayerResidencySnapshot {
                layer_index: 0,
                base_slot_capacity: 8,
                physical_base_slot_budget: 8,
                physical_base_slot_budget_bytes: 1_024,
                occupied_base_slots: 2,
                occupied_base_payload_bytes: 200,
                occupied_base_touched_bytes: 256,
                file_mapped_addressable_slots: 0,
                file_mapped_address_span_bytes: 0,
                slot_stats: stats.clone(),
            }],
            aggregate_slot_stats: stats,
            cumulative_expert_payload_read_bytes: 200,
            host_cache_budget_bytes: 64 * 1024 * 1024,
            interval_routed_expert_union_scope:
                "since_latest_explicit_telemetry_interval_begin_or_runtime_load".into(),
            interval_routed_expert_ids_per_layer: vec![vec![3, 9]],
            interval_routed_unique_per_layer: vec![2],
            interval_routed_unique_experts_sum: 2,
            interval_routed_unique_experts_max: 2,
            last_chained_ledger_scope:
                "latest_completed_chained_attempt_only_not_generation_maximum".into(),
            last_chained_unique_per_layer: vec![0],
            last_chained_hot_bound_per_layer: vec![0],
            last_chained_mapped_bound_per_layer: vec![0],
            ..Default::default()
        }
    }

    fn hybrid_chained_snapshot(
        unique: u16,
        hot: u16,
        mapped: u16,
    ) -> camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot {
        use camelid::gemma4_runtime::{
            Gemma4RoutedExpertLayerResidencySnapshot, Gemma4RoutedExpertResidencySnapshot,
            Gemma4RoutedExpertSlotStatsSnapshot,
        };
        assert_eq!(u32::from(hot) + u32::from(mapped), u32::from(unique));
        let stats = Gemma4RoutedExpertSlotStatsSnapshot {
            route_lookups: u64::from(unique),
            hits: u64::from(hot),
            misses: u64::from(mapped),
            ..Default::default()
        };
        Gemma4RoutedExpertResidencySnapshot {
            layer_count: 1,
            slot_record_bytes: 100,
            slot_stride_bytes: 128,
            base_slot_capacity: 128,
            base_slot_capacity_bytes: 128 * 128,
            physical_base_slot_budget: 48,
            physical_base_slot_budget_bytes: 48 * 128,
            file_mapped_addressable_slots: 128,
            file_mapped_address_span_bytes: 128 * 128,
            occupied_base_slots: 48,
            occupied_base_payload_bytes: 48 * 100,
            occupied_base_touched_bytes: 48 * 128,
            per_layer: vec![Gemma4RoutedExpertLayerResidencySnapshot {
                layer_index: 0,
                base_slot_capacity: 128,
                physical_base_slot_budget: 48,
                physical_base_slot_budget_bytes: 48 * 128,
                file_mapped_addressable_slots: 128,
                file_mapped_address_span_bytes: 128 * 128,
                occupied_base_slots: 48,
                occupied_base_payload_bytes: 48 * 100,
                occupied_base_touched_bytes: 48 * 128,
                slot_stats: stats.clone(),
            }],
            aggregate_slot_stats: stats,
            interval_routed_expert_union_scope:
                "since_latest_explicit_telemetry_interval_begin_or_runtime_load".into(),
            interval_routed_expert_ids_per_layer: vec![(0..unique).collect()],
            interval_routed_unique_per_layer: vec![unique],
            interval_routed_unique_experts_sum: u32::from(unique),
            interval_routed_unique_experts_max: u32::from(unique),
            last_chained_ledger_scope:
                "latest_completed_chained_attempt_only_not_generation_maximum".into(),
            last_chained_round_available: true,
            last_chained_round_succeeded: true,
            last_chained_round_sequence: 1,
            last_chained_k: Some(8),
            last_chained_unique_per_layer: vec![unique],
            last_chained_unique_experts_sum: u32::from(unique),
            last_chained_unique_experts_max: u32::from(unique),
            last_chained_hot_bound_per_layer: vec![hot],
            last_chained_mapped_bound_per_layer: vec![mapped],
            last_chained_hot_bound_records: u32::from(hot),
            last_chained_mapped_bound_records: u32::from(mapped),
            last_chained_slot_hits: u32::from(hot),
            last_chained_slot_misses: u32::from(mapped),
            ..Default::default()
        }
    }

    fn exact_hybrid_geometry_snapshot(
    ) -> camelid::gemma4_runtime::Gemma4RoutedExpertResidencySnapshot {
        use camelid::gemma4_runtime::{
            Gemma4RoutedExpertLayerResidencySnapshot, Gemma4RoutedExpertResidencySnapshot,
        };
        const LAYERS: usize = 30;
        const CANONICAL: u64 = 128;
        const HOT: u64 = 48;
        const RECORD: u64 = 3_345_408;
        const STRIDE: u64 = 3_358_720;
        let per_layer = (0..LAYERS)
            .map(|layer_index| Gemma4RoutedExpertLayerResidencySnapshot {
                layer_index: layer_index as u32,
                base_slot_capacity: CANONICAL,
                physical_base_slot_budget: HOT,
                physical_base_slot_budget_bytes: HOT * STRIDE,
                file_mapped_addressable_slots: CANONICAL,
                file_mapped_address_span_bytes: CANONICAL * STRIDE,
                ..Default::default()
            })
            .collect::<Vec<_>>();
        let canonical_total = LAYERS as u64 * CANONICAL;
        let hot_total = LAYERS as u64 * HOT;
        Gemma4RoutedExpertResidencySnapshot {
            layer_count: LAYERS as u32,
            slot_record_bytes: RECORD,
            slot_stride_bytes: STRIDE,
            base_slot_capacity: canonical_total,
            base_slot_capacity_bytes: canonical_total * STRIDE,
            physical_base_slot_budget: hot_total,
            physical_base_slot_budget_bytes: hot_total * STRIDE,
            file_mapped_addressable_slots: canonical_total,
            file_mapped_address_span_bytes: canonical_total * STRIDE,
            per_layer,
            interval_routed_expert_union_scope:
                "since_latest_explicit_telemetry_interval_begin_or_runtime_load".into(),
            interval_routed_expert_ids_per_layer: vec![Vec::new(); LAYERS],
            interval_routed_unique_per_layer: vec![0; LAYERS],
            last_chained_ledger_scope:
                "latest_completed_chained_attempt_only_not_generation_maximum".into(),
            last_chained_unique_per_layer: vec![0; LAYERS],
            last_chained_hot_bound_per_layer: vec![0; LAYERS],
            last_chained_mapped_bound_per_layer: vec![0; LAYERS],
            ..Default::default()
        }
    }

    #[test]
    fn pilot_only_requires_exact_positive_opt_in() {
        assert_eq!(pilot_only_from(None).unwrap(), false);
        assert_eq!(pilot_only_from(Some("1")).unwrap(), true);
        for invalid in ["", "0", "true", " 1 "] {
            assert!(pilot_only_from(Some(invalid)).is_err());
        }
    }

    #[test]
    fn routed_expert_receipt_binds_occupancy_stride_and_transition() {
        let before = routed_expert_snapshot();
        let mut after = before.clone();
        after.interval_routed_expert_union_epoch = 1;
        assert!(validate_routed_expert_snapshot("test", &before).is_ok());
        assert!(validate_routed_expert_transition(&before, &after).is_ok());

        let mut invalid = after.clone();
        invalid.occupied_base_touched_bytes += 1;
        assert!(validate_routed_expert_snapshot("test", &invalid).is_err());

        let mut invalid_union = after;
        invalid_union.interval_routed_expert_ids_per_layer[0] = vec![9, 3];
        assert!(validate_routed_expert_snapshot("test", &invalid_union).is_err());
    }

    #[test]
    fn hybrid_chained_49_to_64_union_is_hot_plus_mapped_not_overflow() {
        for (unique, hot, mapped) in [(49, 48, 1), (64, 48, 16)] {
            let receipt = hybrid_chained_snapshot(unique, hot, mapped);
            assert!(
                validate_routed_expert_snapshot("hybrid spill", &receipt).is_ok(),
                "unique={unique} hot={hot} mapped={mapped}"
            );

            let mut wrong_partition = receipt.clone();
            wrong_partition.last_chained_mapped_bound_per_layer[0] -= 1;
            wrong_partition.last_chained_mapped_bound_records -= 1;
            assert!(validate_routed_expert_snapshot("hybrid spill", &wrong_partition).is_err());

            let mut swapped_tier_labels = receipt.clone();
            swapped_tier_labels.last_chained_hot_bound_per_layer[0] -= 1;
            swapped_tier_labels.last_chained_mapped_bound_per_layer[0] += 1;
            swapped_tier_labels.last_chained_hot_bound_records -= 1;
            swapped_tier_labels.last_chained_mapped_bound_records += 1;
            assert!(
                validate_routed_expert_snapshot("hybrid spill", &swapped_tier_labels).is_err()
            );

            let mut false_overflow = receipt;
            false_overflow.last_chained_slot_capacity_overflow = 1;
            assert!(validate_routed_expert_snapshot("hybrid spill", &false_overflow).is_err());
        }

        let impossible_hot = hybrid_chained_snapshot(64, 64, 0);
        assert!(validate_routed_expert_snapshot("hybrid hot overflow", &impossible_hot).is_err());

        let mut malformed_bytes = hybrid_chained_snapshot(49, 47, 2);
        malformed_bytes.last_chained_demand_loads = 1;
        malformed_bytes.last_chained_demand_read_bytes = 99;
        malformed_bytes.cumulative_chained_demand_loads = 1;
        malformed_bytes.cumulative_chained_demand_read_bytes = 99;
        malformed_bytes.cumulative_expert_payload_read_bytes = 99;
        assert!(validate_routed_expert_snapshot("hybrid read bytes", &malformed_bytes).is_err());

        malformed_bytes.last_chained_demand_read_bytes = 100;
        malformed_bytes.cumulative_chained_demand_read_bytes = 100;
        malformed_bytes.cumulative_expert_payload_read_bytes = 100;
        assert!(validate_routed_expert_snapshot("hybrid read bytes", &malformed_bytes).is_ok());
    }

    #[test]
    fn hybrid_no_round_receipt_rejects_tier_or_overflow_facts() {
        let clean = exact_hybrid_geometry_snapshot();
        assert!(validate_routed_expert_snapshot("hybrid clean", &clean).is_ok());
        assert!(validate_exact_target_hybrid_experts(&clean).is_ok());

        let mut tier_without_round = clean.clone();
        tier_without_round.last_chained_hot_bound_per_layer[0] = 1;
        tier_without_round.last_chained_hot_bound_records = 1;
        assert!(
            validate_routed_expert_snapshot("hybrid clean", &tier_without_round).is_err()
        );

        let mut no_round_sequence = clean.clone();
        no_round_sequence.last_chained_round_sequence = 1;
        assert!(validate_routed_expert_snapshot("hybrid no round", &no_round_sequence).is_err());

        let mut no_round_io = clean.clone();
        no_round_io.cumulative_chained_demand_loads = 1;
        no_round_io.cumulative_chained_demand_read_bytes = 3_345_408;
        no_round_io.cumulative_expert_payload_read_bytes = 3_345_408;
        assert!(validate_routed_expert_snapshot("hybrid no round", &no_round_io).is_err());

        let mut wrong_capacity = clean.clone();
        wrong_capacity.per_layer[0].physical_base_slot_budget = 49;
        wrong_capacity.per_layer[0].physical_base_slot_budget_bytes = 49 * 3_358_720;
        wrong_capacity.physical_base_slot_budget += 1;
        wrong_capacity.physical_base_slot_budget_bytes += 3_358_720;
        assert!(validate_routed_expert_snapshot("hybrid wrong capacity", &wrong_capacity).is_ok());
        assert!(validate_exact_target_hybrid_experts(&wrong_capacity).is_err());

        let mut failed_overflow = clean.clone();
        failed_overflow.last_chained_round_available = true;
        failed_overflow.last_chained_round_sequence = 1;
        failed_overflow.last_chained_k = Some(8);
        failed_overflow.last_chained_overflow_slots = 1;
        assert!(validate_routed_expert_snapshot("hybrid failed", &failed_overflow).is_err());
        assert!(validate_exact_target_hybrid_experts(&failed_overflow).is_err());

        let mut promoted = clean.clone();
        promoted.last_chained_round_available = true;
        promoted.last_chained_round_succeeded = true;
        promoted.last_chained_round_sequence = 1;
        promoted.last_chained_k = Some(8);
        promoted.last_chained_unique_per_layer[0] = 49;
        promoted.last_chained_unique_experts_sum = 49;
        promoted.last_chained_unique_experts_max = 49;
        promoted.last_chained_hot_bound_per_layer[0] = 47;
        promoted.last_chained_mapped_bound_per_layer[0] = 2;
        promoted.last_chained_hot_bound_records = 47;
        promoted.last_chained_mapped_bound_records = 2;
        promoted.last_chained_slot_hits = 47;
        promoted.last_chained_slot_misses = 2;
        promoted.last_chained_demand_loads = 1;
        promoted.last_chained_demand_read_bytes = 3_345_408;
        promoted.cumulative_chained_demand_loads = 1;
        promoted.cumulative_chained_demand_read_bytes = 3_345_408;
        promoted.cumulative_expert_payload_read_bytes = 3_345_408;
        promoted.aggregate_slot_stats.route_lookups = 49;
        promoted.aggregate_slot_stats.hits = 47;
        promoted.aggregate_slot_stats.misses = 2;
        promoted.per_layer[0].slot_stats = promoted.aggregate_slot_stats.clone();
        assert!(validate_routed_expert_snapshot("hybrid promoted", &promoted).is_ok());
        assert!(validate_exact_target_hybrid_experts(&promoted).is_ok());

        let mut missing_promotion_io = promoted;
        missing_promotion_io.last_chained_demand_loads = 0;
        missing_promotion_io.last_chained_demand_read_bytes = 0;
        missing_promotion_io.cumulative_chained_demand_loads = 0;
        missing_promotion_io.cumulative_chained_demand_read_bytes = 0;
        missing_promotion_io.cumulative_expert_payload_read_bytes = 0;
        assert!(validate_routed_expert_snapshot("hybrid promoted", &missing_promotion_io).is_ok());
        assert!(validate_exact_target_hybrid_experts(&missing_promotion_io).is_err());

        for mutate in [
            |stats: &mut camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot| {
                stats.host_fills = 1;
            },
            |stats: &mut camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot| {
                stats.prewarm_copies = 1;
            },
            |stats: &mut camelid::gemma4_runtime::Gemma4RoutedExpertSlotStatsSnapshot| {
                stats.direct_read_failures = 1;
            },
        ] {
            let mut unexpected_fill = exact_hybrid_geometry_snapshot();
            mutate(&mut unexpected_fill.aggregate_slot_stats);
            assert!(validate_exact_target_hybrid_experts(&unexpected_fill).is_err());
        }
    }

    #[test]
    fn hybrid_generation_transition_requires_hot_and_mapped_deltas() {
        let before = exact_hybrid_geometry_snapshot();
        let mut after = before.clone();
        after.interval_routed_expert_union_epoch =
            before.interval_routed_expert_union_epoch.saturating_add(1);
        after.aggregate_slot_stats.route_lookups = 2;
        after.aggregate_slot_stats.hits = 1;
        after.aggregate_slot_stats.misses = 1;
        after.per_layer[0].slot_stats = after.aggregate_slot_stats.clone();
        assert!(validate_routed_expert_transition(&before, &after).is_ok());

        let mut no_mapped_delta = after.clone();
        no_mapped_delta.aggregate_slot_stats.route_lookups = 1;
        no_mapped_delta.aggregate_slot_stats.misses = 0;
        no_mapped_delta.per_layer[0].slot_stats = no_mapped_delta.aggregate_slot_stats.clone();
        assert!(validate_routed_expert_transition(&before, &no_mapped_delta).is_err());

        let mut no_hot_delta = after;
        no_hot_delta.aggregate_slot_stats.route_lookups = 1;
        no_hot_delta.aggregate_slot_stats.hits = 0;
        no_hot_delta.per_layer[0].slot_stats = no_hot_delta.aggregate_slot_stats.clone();
        assert!(validate_routed_expert_transition(&before, &no_hot_delta).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn load_only_ledger_requires_exact_hybrid_capacity_and_zero_side_caches() {
        use camelid::gemma4_runtime::Gemma4GhostLoadAllocationLedger;
        const LAYERS: u64 = 30;
        const CANONICAL: u64 = 128;
        const HOT: u64 = 48;
        const STRIDE: u64 = 3_358_720;
        let mut ledger = Gemma4GhostLoadAllocationLedger {
            cghost_logical_bytes: LAYERS * CANONICAL * STRIDE,
            cghost_mapped_bytes: LAYERS * CANONICAL * STRIDE,
            expert_layer_count: LAYERS,
            expert_logical_slot_count: LAYERS * CANONICAL,
            expert_slot_count: LAYERS * HOT,
            expert_slot_capacity_bytes: LAYERS * HOT * STRIDE,
            expert_file_mapped_slot_count: LAYERS * CANONICAL,
            expert_file_mapped_address_span_bytes: LAYERS * CANONICAL * STRIDE,
            expert_table_directory_slot_count: LAYERS * HOT,
            expert_table_directory_capacity_bytes: LAYERS * HOT * STRIDE,
            expert_slots_active: true,
            arbitrary_slot_prewarm_skipped: true,
            ..Default::default()
        };
        assert!(load_only_hybrid_expert_capacity_ready(&ledger));

        ledger.expert_file_mapped_slot_count -= 1;
        assert!(!load_only_hybrid_expert_capacity_ready(&ledger));
        ledger.expert_file_mapped_slot_count += 1;
        ledger.host_cache_budget_bytes = 1;
        assert!(!load_only_hybrid_expert_capacity_ready(&ledger));
        ledger.host_cache_budget_bytes = 0;
        ledger.overflow_slot_count = 1;
        assert!(!load_only_hybrid_expert_capacity_ready(&ledger));
        ledger.overflow_slot_count = 0;
        ledger.cghost_mapped_bytes = 1;
        assert!(!load_only_hybrid_expert_capacity_ready(&ledger));
    }

    fn key() -> WindowKey {
        WindowKey {
            workload: Workload::Copy,
            repetition: 0,
            prefix_start: 0,
            prefix_end: 32,
        }
    }

    fn pairing() -> PairingEvidence {
        PairingEvidence {
            pair_gate_passed: true,
            target_repository: OFFICIAL_TARGET_REPOSITORY.into(),
            target_revision: OFFICIAL_TARGET_REVISION.into(),
            target_source_path: "/Volumes/source/target.gguf".into(),
            target_source_bytes: OFFICIAL_TARGET_BYTES,
            target_source_sha256: OFFICIAL_TARGET_SHA256.into(),
            target_source_matches_official: true,
            target_staged_runtime_path: "/runtime/target.hot.gguf".into(),
            target_staged_cghost_path: "/runtime/target.cghost".into(),
            target_staged_identity_scheme: STAGED_TARGET_IDENTITY_SCHEME.into(),
            target_staged_metadata_matches_source: true,
            target_cghost_matches_official_source: true,
            target_cghost_matches_staged_runtime: true,
            assistant_repository: OFFICIAL_ASSISTANT_REPOSITORY.into(),
            assistant_revision: OFFICIAL_ASSISTANT_REVISION.into(),
            assistant_staged_model_path: "/stage/assistant/model.safetensors".into(),
            assistant_staged_model_bytes: OFFICIAL_ASSISTANT_MODEL_BYTES,
            assistant_staged_model_sha256: OFFICIAL_ASSISTANT_MODEL_SHA256.into(),
            assistant_staged_config_path: "/stage/assistant/config.json".into(),
            assistant_staged_config_sha256: OFFICIAL_ASSISTANT_CONFIG_SHA256.into(),
            assistant_staged_tokenizer_config_path: "/stage/assistant/tokenizer_config.json".into(),
            assistant_staged_tokenizer_config_sha256: OFFICIAL_ASSISTANT_TOKENIZER_CONFIG_SHA256
                .into(),
            assistant_staged_tokenizer_path: "/stage/assistant/tokenizer.json".into(),
            assistant_staged_tokenizer_sha256: OFFICIAL_ASSISTANT_TOKENIZER_SHA256.into(),
            assistant_staged_files_match_official: true,
            shared_vocab_size: SHARED_VOCAB_SIZE,
            tokenizer_mismatch_count: 0,
            target_shared_kv_layers: TARGET_SHARED_KV_LAYERS,
            assistant_shared_kv_layers: ASSISTANT_SHARED_KV_LAYERS,
            target_shared_kv_sliding_source_layer: TARGET_SHARED_KV_SLIDING_SOURCE_LAYER,
            target_shared_kv_full_source_layer: TARGET_SHARED_KV_FULL_SOURCE_LAYER,
        }
    }

    fn lane_window(
        lane: Lane,
        rounds: u64,
        emitted: u64,
        accepted: u64,
        draft_us: u64,
        verify_us: u64,
        total_us: u64,
    ) -> LaneWindow {
        LaneWindow {
            key: key(),
            lane,
            verify_rounds: rounds,
            emitted_tokens: emitted,
            accepted_drafts: accepted,
            draft_wall_us: draft_us,
            verify_wall_us: verify_us,
            total_wall_us: total_us,
            assistant_invocations: if lane == Lane::Mtp { 1 } else { 0 },
            proposal_trace_sha256: if lane == Lane::Mtp {
                "mtp".to_string()
            } else {
                "same-n-i".to_string()
            },
        }
    }

    fn losing_economics() -> NimEconomics {
        NimEconomics::derive(
            &lane_window(Lane::NgramBaseline, 8, 16, 10, 100, 8_000, 9_000),
            &lane_window(Lane::NgramAssistantIdle, 8, 16, 10, 100, 8_000, 9_100),
            &lane_window(Lane::Mtp, 8, 16, 12, 2_000, 7_500, 10_000),
        )
        .unwrap()
    }

    #[test]
    fn workload_prompts_are_the_spec50_decision_set() {
        let workloads = decision_workloads();
        assert_eq!(workloads.len(), 4);
        assert_eq!(
            workloads
                .iter()
                .map(|workload| workload.key.key())
                .collect::<Vec<_>>(),
            ["copy", "code-edit", "json-yaml", "prose"]
        );
        assert!(workloads[0].prompt.contains(PARA));
        assert!(workloads[1].prompt.contains("pub created_at: u64"));
        assert!(workloads[2].prompt.contains("\"max_replicas\": 32"));
        assert!(workloads[3].prompt.contains("three short paragraphs"));
    }

    #[test]
    fn target_runtime_settings_pin_the_current_exact_lane() {
        let config = TargetRuntimeConfig {
            runtime_gguf_path: "/runtime/target.gguf".into(),
            cghost_path: "/runtime/target.cghost".into(),
            expert_cache_mib: DEFAULT_TARGET_CACHE_MIB,
            environment: exact_target_environment(),
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.expert_cache_mib, 0);
        assert_eq!(config.environment["CAMELID_GEMMA4_VICTIM_CACHE"], "0");
        assert_eq!(config.environment["CAMELID_GEMMA4_VICTIM_MB"], "0");
        assert_eq!(
            config.environment["CAMELID_GEMMA4_GHOST_METAL_DEMAND_LOAD_ONLY"],
            "1"
        );
        assert_eq!(
            config.environment["CAMELID_GEMMA4_GHOST_METAL_FILE_MAPPED_EXPERTS"],
            "1"
        );
        assert_eq!(
            config.environment["CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS"],
            "48"
        );
        assert_eq!(config.environment["CAMELID_GEMMA4_SLOT_PIN"], "0");
        assert!(!config
            .environment
            .contains_key("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER"));
        assert!(!config
            .environment
            .contains_key("CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER"));
        assert!(!config
            .environment
            .contains_key("CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS"));
        assert!(!config.environment.contains_key("CAMELID_GEMMA4_MTP_PIN"));
        assert_eq!(
            config.environment["CAMELID_GEMMA4_GHOST_METAL_HEAD_RESIDENT"],
            "0"
        );
        assert_eq!(config.environment["CAMELID_GEMMA4_SPEC_CHUNK_MAX"], "8");
        assert_eq!(config.environment["CAMELID_GEMMA4_SPEC_DRAFT_TOKENS"], "8");

        let mut drifted = config.clone();
        drifted
            .environment
            .insert("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS".into(), "048".into());
        assert!(drifted
            .validate()
            .unwrap_err()
            .contains("CAMELID_GEMMA4_GHOST_METAL_HYBRID_HOT_SLOTS"));

        for legacy_key in [
            "CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER",
            "CAMELID_GEMMA4_GHOST_METAL_PER_LAYER_SLOTS",
            "CAMELID_GEMMA4_GHOST_METAL_PHYSICAL_SLOTS_PER_LAYER",
        ] {
            let mut legacy_geometry = config.clone();
            legacy_geometry
                .environment
                .insert(legacy_key.into(), "48".into());
            assert!(
                legacy_geometry
                    .validate()
                    .unwrap_err()
                    .contains("unexpected settings"),
                "{legacy_key}"
            );
        }

        let mut host_cached = config;
        host_cached.expert_cache_mib = 1;
        assert!(host_cached
            .validate()
            .unwrap_err()
            .contains("zero-byte host expert cache"));
    }

    #[test]
    fn lane_plans_isolate_n_i_and_m_in_fresh_processes() {
        let plans = [
            LaneExecutionPlan::for_lane(Lane::NgramBaseline),
            LaneExecutionPlan::for_lane(Lane::NgramAssistantIdle),
            LaneExecutionPlan::for_lane(Lane::Mtp),
        ];
        assert!(validate_lane_plans(&plans).is_ok());
        let n = &plans[0];
        let i = &plans[1];
        let m = &plans[2];
        assert_eq!(n.driver, LaneDriver::CurrentSpecDecodeGenerate);
        assert_eq!(i.driver, LaneDriver::CurrentSpecDecodeGenerate);
        assert_eq!(n.proposal_environment, i.proposal_environment);
        assert!(!n.load_assistant);
        assert!(i.load_assistant && i.warm_assistant && !i.invoke_assistant);
        assert_eq!(m.driver, LaneDriver::NativeMtpExperiment);
        assert!(m.load_assistant && m.warm_assistant && m.invoke_assistant);
        assert!(plans.iter().all(|plan| {
            plan.fresh_process
                && plan.target_warmup_tokens == TARGET_WARMUP_TOKENS
                && plan.snapshot_before_assistant_load
        }));

        let diagnostic = LaneExecutionPlan::for_lane(Lane::NgramSeedOffDiagnostic);
        assert_eq!(
            diagnostic.proposal_environment["CAMELID_GEMMA4_SPEC_SEED_TEXT"],
            ""
        );
    }

    #[test]
    fn pairing_receipt_is_exact_and_fails_closed() {
        let exact = pairing();
        assert!(exact.validate(&exact.assistant_staged_model_path).is_ok());

        let mut mismatch = exact.clone();
        mismatch.tokenizer_mismatch_count = 1;
        assert!(mismatch
            .validate(&mismatch.assistant_staged_model_path)
            .unwrap_err()
            .contains("tokenizer pair is not exact"));

        let mut wrong_shared_kv = exact.clone();
        wrong_shared_kv.target_shared_kv_layers = 4;
        assert!(wrong_shared_kv
            .validate(&wrong_shared_kv.assistant_staged_model_path)
            .unwrap_err()
            .contains("shared-KV metadata mismatch"));

        let mut unbound_stage = exact.clone();
        unbound_stage.target_cghost_matches_staged_runtime = false;
        assert!(unbound_stage
            .validate(&unbound_stage.assistant_staged_model_path)
            .unwrap_err()
            .contains("not bound to the official source"));

        let wrong_selected_path = PathBuf::from("/stage/other/model.safetensors");
        assert!(exact
            .validate(&wrong_selected_path)
            .unwrap_err()
            .contains("selected assistant path differs"));
    }

    #[test]
    fn round_metrics_capture_actual_k_and_acceptance_histograms() {
        let first = round(8, 6, 7, 5);
        let mut tail = round(8, 7, 8, 3);
        tail.budget_truncated = true;
        let metrics = RunMetrics::from_rounds(&[first, tail]).unwrap();
        assert_eq!(metrics.rounds, 2);
        assert_eq!(metrics.acceptance_probability, 8.0 / 13.0);
        assert_eq!(metrics.accepted_drafts_per_round, 4.0);
        assert_eq!(metrics.accepted_target_tokens_per_round, 5.0);
        assert_eq!(metrics.requested_k_histogram, BTreeMap::from([(8, 2)]));
        assert_eq!(
            metrics.proposed_k_histogram,
            BTreeMap::from([(6, 1), (7, 1)])
        );
        assert_eq!(
            metrics.verifier_k_histogram,
            BTreeMap::from([(7, 1), (8, 1)])
        );
        assert_eq!(
            metrics.accepted_drafts_histogram,
            BTreeMap::from([(3, 1), (5, 1)])
        );
        assert_eq!(metrics.full_budget_rounds, 1);
        assert_eq!(metrics.tail_truncated_rounds, 1);
        assert_eq!(
            metrics.full_budget_proposed_k_histogram,
            BTreeMap::from([(6, 1)])
        );
        assert_eq!(
            metrics.full_budget_verifier_k_histogram,
            BTreeMap::from([(7, 1)])
        );
        assert_eq!(
            metrics.tail_truncated_proposed_k_histogram,
            BTreeMap::from([(7, 1)])
        );
        assert_eq!(
            metrics.tail_truncated_verifier_k_histogram,
            BTreeMap::from([(8, 1)])
        );

        let terminal = RunMetrics::from_rounds_and_terminal(&[round(8, 7, 8, 6)], 1).unwrap();
        assert_eq!(terminal.terminal_target_tokens, 1);
        assert_eq!(terminal.emitted_target_tokens, 7);
        assert_eq!(terminal.visible_output_tokens, 8);
        assert_eq!(terminal.accepted_target_tokens_per_round, 7.0);
        assert_eq!(terminal.verifier_k_histogram, BTreeMap::from([(8, 1)]));
    }

    #[test]
    fn invalid_rounds_fail_closed() {
        let mut invalid = round(8, 9, 8, 2);
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("exceeds requested"));
        invalid = round(8, 7, 8, 8);
        assert!(invalid.validate().unwrap_err().contains("accepted"));
        invalid = round(8, 7, 8, 6);
        invalid.proposal_source = ProposalSource::NgramSeeded;
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("proposals instead"));
        invalid = round(8, 7, 7, 6);
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("authoritative anchor"));
        invalid = round(9, 7, 8, 6);
        assert!(invalid.validate().unwrap_err().contains("exact 1..=8"));
        invalid = round(8, 7, 8, 6);
        invalid.emitted_target_tokens = 6;
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("useful accepted drafts + authoritative anchor"));
        invalid = round(8, 7, 8, 6);
        invalid.round_wall_us = 3_999;
        assert!(invalid
            .validate()
            .unwrap_err()
            .contains("shorter than an observed phase"));
    }

    #[test]
    fn lane_run_receipt_proves_driver_trace_and_target_output() {
        let rounds = vec![round(8, 7, 8, 6)];
        let metrics = RunMetrics::from_rounds(&rounds).unwrap();
        let mut run = LaneRun {
            run_id: "test-run".into(),
            phase: RunPhase::Pilot,
            workload: Workload::Copy,
            lane: Lane::Mtp,
            repetition: 0,
            requested_output_tokens: PILOT_TOKENS,
            child_process_id: 42,
            driver: LaneDriver::NativeMtpExperiment,
            assistant_invocations: 7,
            proposal_trace_sha256: "0".repeat(64),
            output_token_ids: vec![1, 2, 3, 4, 5, 6, 7],
            terminal_target_tokens: 0,
            generation_wall_us: 4_500,
            completed: true,
            rounds,
            metrics,
            routed_experts_after_generation: Some(routed_expert_snapshot()),
        };
        let plan = LaneExecutionPlan::for_lane(Lane::Mtp);
        assert!(run.validate(&plan).is_ok());

        run.driver = LaneDriver::CurrentSpecDecodeGenerate;
        assert!(run.validate(&plan).unwrap_err().contains("expected"));
        run.driver = LaneDriver::NativeMtpExperiment;
        run.proposal_trace_sha256 = "A".repeat(64);
        assert!(run.validate(&plan).unwrap_err().contains("SHA-256"));
    }

    #[test]
    fn nim_economics_prices_d_p_and_verification_halves() {
        let n = lane_window(Lane::NgramBaseline, 8, 16, 8, 100, 10_000, 11_000);
        let i = lane_window(Lane::NgramAssistantIdle, 8, 16, 8, 100, 10_000, 12_000);
        let m = lane_window(Lane::Mtp, 8, 16, 11, 2_000, 7_000, 9_500);
        let economics = NimEconomics::derive(&n, &i, &m).unwrap();
        assert_eq!(economics.delta_a, 3);
        assert_eq!(economics.d_us, 2_000);
        assert_eq!(economics.p_us, 1_000);
        assert_eq!(economics.v_plus_us, 0);
        assert_eq!(economics.v_minus_us, 3_000);
        assert_eq!(economics.net_saved_us, 0);
        assert_eq!(economics.score, 1_000.0);
    }

    #[test]
    fn n_and_i_must_have_identical_proposal_traces() {
        let n = lane_window(Lane::NgramBaseline, 8, 16, 8, 1, 10, 11);
        let mut i = lane_window(Lane::NgramAssistantIdle, 8, 16, 8, 1, 10, 11);
        i.proposal_trace_sha256 = "different".to_string();
        let m = lane_window(Lane::Mtp, 8, 16, 9, 1, 9, 10);
        assert!(NimEconomics::derive(&n, &i, &m)
            .unwrap_err()
            .contains("proposal traces"));
    }

    #[test]
    fn parses_macos_vm_and_swap_snapshots() {
        let vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages active: 20.\nPages inactive: 30.\nPages wired down: 40.\nPages occupied by compressor: 5.\nPages stored in compressor: 7.\nPageins: 11.\nPageouts: 12.\nSwapins: 13.\nSwapouts: 14.\n";
        let swap = "total = 4096.00M  used = 256.50M  free = 3839.50M  (encrypted)";
        let snapshot = parse_system_vm_snapshot(vm, swap, MemoryPressure::Normal, 1_000).unwrap();
        assert_eq!(snapshot.page_size, 16_384);
        assert_eq!(snapshot.free_bytes, 10 * 16_384);
        assert_eq!(snapshot.inactive_bytes, 30 * 16_384);
        assert_eq!(snapshot.reclaimable_headroom_bytes, 40 * 16_384);
        assert_eq!(snapshot.swapins_bytes, 13 * 16_384);
        assert_eq!(snapshot.swapouts_bytes, 14 * 16_384);
        assert_eq!(snapshot.swap_used_bytes, 256 * 1024 * 1024 + 512 * 1024);
        assert_eq!(snapshot.swap_total_bytes, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn memory_delta_is_signed() {
        let before = MemorySnapshot {
            phase: "before".into(),
            process: ProcessMemorySnapshot {
                physical_footprint_bytes: 100,
                rss_bytes: 200,
                peak_rss_bytes: 250,
            },
            system: system_sample(0, 0, 0, 50, MemoryPressure::Normal),
        };
        let mut after = before.clone();
        after.process.physical_footprint_bytes = 180;
        after.process.rss_bytes = 150;
        after.system.swap_used_bytes = 75;
        after.system.free_bytes = before.system.free_bytes - 10;
        after.system.compressor_occupied_bytes = 30;
        after.system.compressed_logical_bytes = 40;
        after.system.pageins_bytes = 50;
        after.system.pageouts_bytes = 60;
        after.system.swapins_bytes = 70;
        after.system.swapouts_bytes = 80;
        let delta = IncrementalMemory::between(&before, &after);
        assert_eq!(delta.physical_footprint_bytes, 80);
        assert_eq!(delta.rss_bytes, -50);
        assert_eq!(delta.swap_used_bytes, 25);
        let load_delta = LoadOnlyMemoryDelta::between(&before, &after);
        assert_eq!(load_delta.physical_footprint_bytes, 80);
        assert_eq!(load_delta.rss_bytes, -50);
        assert_eq!(load_delta.free_bytes, -10);
        assert_eq!(load_delta.compressor_occupied_bytes, 30);
        assert_eq!(load_delta.compressed_logical_bytes, 40);
        assert_eq!(load_delta.pageins_bytes, 50);
        assert_eq!(load_delta.pageouts_bytes, 60);
        assert_eq!(load_delta.swapins_bytes, 70);
        assert_eq!(load_delta.swapouts_bytes, 80);
        assert_eq!(load_delta.swap_used_bytes, 25);
    }

    #[test]
    fn load_only_safety_is_strict_at_every_observed_sample() {
        let baseline = system_sample(0, 0, 10_000, 0, MemoryPressure::Normal);
        let safe = LoadOnlyPhaseCheckpoint {
            snapshot: MemorySnapshot {
                phase: "safe".into(),
                process: ProcessMemorySnapshot {
                    physical_footprint_bytes: 1,
                    rss_bytes: 1,
                    peak_rss_bytes: 1,
                },
                system: baseline.clone(),
            },
            delta_from_clean: LoadOnlyMemoryDelta::default(),
            delta_from_previous: LoadOnlyMemoryDelta::default(),
            target_observation: None,
        };
        assert!(load_only_safety_violation(&baseline, &[safe.clone()], &[]).is_none());

        let mut swapped = baseline.clone();
        swapped.swapouts_bytes += swapped.page_size;
        assert!(
            load_only_safety_violation(&baseline, &[safe.clone()], &[swapped])
                .unwrap()
                .contains("swapouts increased")
        );

        let mut pressure = baseline.clone();
        pressure.pressure = MemoryPressure::Warn;
        assert!(
            load_only_safety_violation(&baseline, &[safe.clone()], &[pressure])
                .unwrap()
                .contains("not normal")
        );

        let mut low_strict_free = baseline.clone();
        low_strict_free.free_bytes = 1;
        low_strict_free.inactive_bytes = LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES;
        low_strict_free.reclaimable_headroom_bytes = low_strict_free
            .free_bytes
            .saturating_add(low_strict_free.inactive_bytes);
        assert!(
            load_only_safety_violation(&baseline, &[safe.clone()], &[low_strict_free]).is_none()
        );

        let mut at_floor = baseline.clone();
        at_floor.free_bytes = 1;
        at_floor.inactive_bytes = LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES - 1;
        at_floor.reclaimable_headroom_bytes =
            at_floor.free_bytes.saturating_add(at_floor.inactive_bytes);
        assert!(load_only_safety_violation(&baseline, &[safe.clone()], &[at_floor]).is_none());

        let mut low_headroom = baseline.clone();
        low_headroom.free_bytes = 1;
        low_headroom.inactive_bytes = LOAD_ONLY_MIN_RECLAIMABLE_HEADROOM_BYTES - 2;
        low_headroom.reclaimable_headroom_bytes = low_headroom
            .free_bytes
            .saturating_add(low_headroom.inactive_bytes);
        assert!(
            load_only_safety_violation(&baseline, &[safe], &[low_headroom])
                .unwrap()
                .contains("below")
        );

        let mut saturated = baseline;
        saturated.free_bytes = u64::MAX;
        saturated.inactive_bytes = 1;
        saturated.reclaimable_headroom_bytes = saturated
            .free_bytes
            .saturating_add(saturated.inactive_bytes);
        assert_eq!(saturated.reclaimable_headroom_bytes, u64::MAX);
        assert!(load_only_safety_violation(&saturated, &[], &[saturated.clone()]).is_none());
    }

    #[test]
    fn kill_after_three_consecutive_swap_activity_samples() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        assert_eq!(
            policy.observe_memory(system_sample(1_000, 4_096, 0, 0, MemoryPressure::Normal)),
            None
        );
        assert_eq!(
            policy.observe_memory(system_sample(2_000, 8_192, 0, 0, MemoryPressure::Normal)),
            None
        );
        assert_eq!(
            policy.observe_memory(system_sample(3_000, 12_288, 0, 0, MemoryPressure::Normal)),
            Some(KillReason::SwapActivityThreeConsecutiveSamples)
        );
    }

    #[test]
    fn kill_on_rolling_swap_traffic_limit() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        let reason = policy.observe_memory(system_sample(
            1_000,
            33 * 1024 * 1024,
            32 * 1024 * 1024,
            0,
            MemoryPressure::Normal,
        ));
        assert_eq!(reason, Some(KillReason::RollingSwapTraffic64MiB));
    }

    #[test]
    fn kill_on_swap_used_growth_that_is_still_increasing() {
        let baseline_used = 10 * 1024 * 1024;
        let baseline = system_sample(0, 0, 0, baseline_used, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        let reason = policy.observe_memory(system_sample(
            1_000,
            0,
            0,
            baseline_used + SWAP_USED_GROWTH_LIMIT_BYTES,
            MemoryPressure::Normal,
        ));
        assert_eq!(
            reason,
            Some(KillReason::SwapUsedGrew256MiBAndStillIncreasing)
        );
    }

    #[test]
    fn kill_after_two_non_normal_pressure_samples() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        assert_eq!(
            policy.observe_memory(system_sample(1_000, 0, 0, 0, MemoryPressure::Warn)),
            None
        );
        assert_eq!(
            policy.observe_memory(system_sample(2_000, 0, 0, 0, MemoryPressure::Critical)),
            Some(KillReason::MemoryPressureTwoConsecutiveSamples)
        );
    }

    #[test]
    fn economics_waits_for_evidence_and_kills_after_two_losing_windows() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        let mut too_small = losing_economics();
        too_small.verify_rounds = 7;
        too_small.emitted_tokens = 15;
        assert_eq!(policy.observe_economics(&too_small), None);
        let losing = losing_economics();
        assert_eq!(policy.observe_economics(&losing), None);
        assert_eq!(
            policy.observe_economics(&losing),
            Some(KillReason::EconomicsLostTwoConsecutiveWindows)
        );
    }

    #[test]
    fn no_acceptance_gain_and_five_percent_wall_loss_kills_immediately() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        let economics = NimEconomics::derive(
            &lane_window(Lane::NgramBaseline, 8, 16, 10, 100, 8_000, 10_000),
            &lane_window(Lane::NgramAssistantIdle, 8, 16, 10, 100, 8_000, 10_100),
            &lane_window(Lane::Mtp, 8, 16, 10, 2_000, 8_000, 10_500),
        )
        .unwrap();
        assert_eq!(
            policy.observe_economics(&economics),
            Some(KillReason::NoAcceptanceGainAndMtpAtLeastFivePercentSlower)
        );
    }

    #[test]
    fn thirty_two_token_pilot_requires_mtp_median_to_beat_n() {
        let baseline = system_sample(0, 0, 0, 0, MemoryPressure::Normal);
        let mut policy = KillPolicyState::new(baseline);
        assert_eq!(
            policy.observe_pilot(&PilotComparison {
                workload: Workload::Copy,
                emitted_tokens: PILOT_TOKENS.saturating_sub(1),
                n_median_wall_us: 100,
                m_median_wall_us: 200,
            }),
            None
        );
        assert_eq!(
            policy.observe_pilot(&PilotComparison {
                workload: Workload::Copy,
                emitted_tokens: PILOT_TOKENS,
                n_median_wall_us: 100,
                m_median_wall_us: 100,
            }),
            Some(KillReason::PilotMedianMtpNotFaster)
        );
    }
}
