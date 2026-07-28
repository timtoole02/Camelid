use std::{
    collections::HashMap,
    convert::Infallible,
    env, mem,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use axum::{
    extract::{rejection::JsonRejection, Path as AxumPath, Query, State},
    http::{HeaderMap, StatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use minijinja::{
    context, Environment, Error as MiniJinjaError, ErrorKind as MiniJinjaErrorKind,
    UndefinedBehavior,
};
use serde::{Deserialize, Serialize};
#[cfg(windows)]
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

mod engine;
mod workspace;

use crate::{
    execution_plan::{plan_for_model, ExecutionPlan, PlannerEnv},
    gguf::{read_metadata, GgufFile, GgufTensorDescriptor, GgufTensorType},
    inference::{
        diagnostic_attention_score_scale, diagnostic_ffn_gate_up_order,
        diagnostic_gqa_head_mapping, diagnostic_linear_accumulation_precision,
        diagnostic_output_projection_layout, diagnostic_rectangular_linear_layout,
        diagnostic_rms_norm_epsilon, diagnostic_rope_direction, diagnostic_rope_position_mode,
        diagnostic_square_linear_layout, diagnostic_zero_delta_selector,
        output_projection_diagnostics, q8_schedule_telemetry_enabled, reset_q8_schedule_telemetry,
        rope_pairing_for_config, snapshot_q8_schedule_telemetry,
        speculative::{
            accepted_draft_prefix, ModelDrafter, NGramDrafter, SpecLatch, SpeculativeDrafter,
            DEFAULT_MODEL_DRAFT_TOKENS, DEFAULT_NGRAM_DRAFT_TOKENS, DEFAULT_NGRAM_MAX_MATCH,
            DEFAULT_NGRAM_MIN_MATCH,
        },
        DeltaZeroTarget, LlamaForwardDiagnostics, LlamaForwardTimings, LlamaGenerationStep,
        LlamaInferenceSession, LlamaLayerMemoryTimings, LlamaLayerTimings, LlamaLoadedWeights,
        LlamaOutputProjectionDiagnostic, LlamaQ8ScheduleTelemetry, LlamaSampler, SamplingConfig,
    },
    model::{
        DenseLlamaDims, LlamaAttentionTensors, LlamaFfnTensors, LlamaModelConfig,
        LlamaTensorBinding,
    },
    model_source::{inspect_model_source, ModelSourceInspection, ModelSourceKind},
    receipt::{
        self, LaneIdentity, ParityBlock, ParityReceipt, ReceiptResult, ReferenceIdentity,
        RECEIPT_SCHEMA_V1,
    },
    telemetry,
    tensor::{parse_byte_count_env, CpuTensor, Q8_0Block, TensorStore},
    tokenizer::Tokenizer,
    BackendError,
};

const DEFAULT_CPU_WEIGHT_MATERIALIZATION_LIMIT_BYTES: u64 = 6 * 1024 * 1024 * 1024;
const CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV: &str = "CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES";
const RETAIN_Q8_BLOCKS_ENV: &str = "CAMELID_RETAIN_Q8_0_BLOCKS";
const LAZY_Q8_LINEAR_ENV: &str = "CAMELID_LAZY_Q8_0_LINEAR";
const METADATA_CHAT_TEMPLATE_ENV: &str = "CAMELID_METADATA_CHAT_TEMPLATE";
const GENERATION_TIMEOUT_ENV: &str = "CAMELID_GENERATION_TIMEOUT_MS";
const STREAM_TIMING_DIAGNOSTICS_ENV: &str = "CAMELID_STREAM_TIMING_DIAGNOSTICS";
// Speculative decoding is a default-off serving optimization (lossless greedy
// speculation; see src/inference/speculative.rs). It makes no support claim.
const SPEC_DECODE_ENV: &str = "CAMELID_SPEC_DECODE";
const SPEC_DRAFT_MODEL_ENV: &str = "CAMELID_SPEC_DRAFT_MODEL";
const SPEC_DRAFT_TOKENS_ENV: &str = "CAMELID_SPEC_DRAFT_TOKENS";
const SPEC_NGRAM_MIN_ENV: &str = "CAMELID_SPEC_NGRAM_MIN";
const SPEC_NGRAM_MAX_ENV: &str = "CAMELID_SPEC_NGRAM_MAX";
const PROMPT_PREFIX_CACHE_CAPACITY_ENV: &str = "CAMELID_PREFIX_CACHE_CAPACITY";
const PROMPT_PREFIX_CACHE_MIN_TOKENS_ENV: &str = "CAMELID_PREFIX_CACHE_MIN_TOKENS";
const DEFAULT_PROMPT_PREFIX_CACHE_CAPACITY: usize = 1;
const DEFAULT_PROMPT_PREFIX_CACHE_MIN_TOKENS: usize = 16;
/// Reserved model id for the speculative draft model; loaded without becoming
/// the active model.
const SPEC_DRAFT_MODEL_ID: &str = "spec-draft";
const STREAM_POLL_YIELD_ENV: &str = "CAMELID_STREAM_POLL_YIELD";
const DEFAULT_GENERATION_TIMEOUT_MS: u64 = 15 * 60 * 1000;
const DEFAULT_PUBLIC_CHAT_MAX_TOKENS: u32 = 800;
const JINJA_CHAT_TEMPLATE_NAME: &str = "chat";
const JINJA_CHAT_TEMPLATE_CACHE_LIMIT: usize = 16;

static JINJA_CHAT_TEMPLATE_ENV_CACHE: OnceLock<Mutex<HashMap<String, Arc<Environment<'static>>>>> =
    OnceLock::new();

#[derive(Clone)]
pub struct AppState {
    loaded_models: Arc<RwLock<HashMap<String, LoadedModel>>>,
    /// Gemma 4 serve runtimes (local single-node or distributed layer-sharding),
    /// keyed by model id. Populated only when the gemma4 serve path is enabled
    /// (`CAMELID_GEMMA4_SERVE`) and a gemma4 model is loaded. This is an
    /// additive, parallel path: the Llama/3B backend is untouched.
    gemma4_runtimes: Arc<RwLock<HashMap<String, Arc<Gemma4ServeRuntime>>>>,
    /// Runnable-lane serve runtimes (qwen35/Ornith), keyed by model id. Populated
    /// only when `CAMELID_RUNNABLE_SERVE` is set and a runnable-served arch is
    /// loaded. Additive, parallel to the optimized engine â€” see the runnable serve
    /// bridge near `runnable_chat_nonstreaming`.
    runnable_runtimes: Arc<RwLock<HashMap<String, Arc<RunnableServeRuntime>>>>,
    /// DiffusionGemma serve runtimes, keyed by model id. Populated only when
    /// `CAMELID_DG_SERVE` is set and a diffusion-gemma model is loaded. Additive,
    /// parallel to the optimized engine (which keeps failing closed for this
    /// arch by design) — see the bridge near `dg_chat_nonstreaming`.
    dg_runtimes: Arc<RwLock<HashMap<String, Arc<DgServeRuntime>>>>,
    execution_plans: Arc<RwLock<HashMap<String, ExecutionPlan>>>,
    cached_weights: Arc<RwLock<HashMap<String, Arc<LlamaLoadedWeights>>>>,
    active_model_id: Arc<RwLock<Option<String>>>,
    model_last_used: Arc<RwLock<HashMap<String, std::time::Instant>>>,
    cached_prompt_prefix: Arc<Mutex<PromptPrefixCachePool>>,
    /// Shared leases cover every local GGUF reader and download transition;
    /// deletion takes the exclusive lease before re-validating file identity.
    model_file_lifecycle: Arc<RwLock<()>>,
    /// Serializes load publication and unload teardown. File readers use the
    /// lifecycle lock separately so generation can continue concurrently.
    model_transition: Arc<tokio::sync::Mutex<()>>,
    /// Per-process nonce makes scan-issued deletion tokens opaque and prevents
    /// clients from constructing authorization from visible file metadata.
    #[cfg(windows)]
    model_delete_nonce: Arc<str>,
    /// Destructive local-file management is available only when the listener
    /// itself is bound to loopback.
    allow_local_model_delete: bool,
    generation_sessions: Arc<RwLock<HashMap<String, GenerationSessionSummary>>>,
    verification_reports: Arc<RwLock<HashMap<String, crate::verify::VerificationReport>>>,
    workspace_sessions: workspace::WorkspaceSessionManager,
    /// Process-rotated bearer capability for same-user Workspace CLI clients.
    /// Browser requests continue to use the independent same-origin predicate.
    workspace_cli_token: Option<Arc<str>>,
    /// Actual listener address. Workspace is disabled when this is non-loopback
    /// and uses it for the existing local HTTP model driver.
    serve_addr: SocketAddr,
    /// The engine worker: the single thread where decode compute and
    /// resident-GPU-state mutations execute (see api/engine.rs). This
    /// replaces the old `generation_lock` — serialization is by construction
    /// (one worker thread), not by a lock whose guard lifetime could be
    /// decoupled from the compute it guarded (the orphan-decode hazard;
    /// docs/recon/ENGINE_INVERSION_CONDUCTOR.md).
    engine: engine::EngineHandle,
    planner_env: PlannerEnv,
    configured_threads: Option<usize>,
    /// Server-wide default for opt-in thinking mode (`serve --enable-thinking`).
    /// Applied only to chat requests that omit `camelid_enable_thinking`; an
    /// explicit value in the request always wins. Off by default so the
    /// parity-locked thinking-DISABLED rendering stays the default.
    default_enable_thinking: bool,
    /// The models directory (`serve --models-dir` / `CAMELID_MODELS_DIR`),
    /// resolved to an absolute path once at startup: scanned by
    /// `/api/models/local`, the catalog download target, and the fallback base
    /// for RELATIVE model paths sent to the load endpoints. Absolute request
    /// paths never touch it, and a relative path that exists against the
    /// process CWD keeps its historical meaning — this only adds a fallback.
    models_dir: PathBuf,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            gemma4_runtimes: Arc::new(RwLock::new(HashMap::new())),
            runnable_runtimes: Arc::new(RwLock::new(HashMap::new())),
            dg_runtimes: Arc::new(RwLock::new(HashMap::new())),
            execution_plans: Arc::new(RwLock::new(HashMap::new())),
            cached_weights: Arc::new(RwLock::new(HashMap::new())),
            active_model_id: Arc::new(RwLock::new(None)),
            model_last_used: Arc::new(RwLock::new(HashMap::new())),
            cached_prompt_prefix: Arc::new(Mutex::new(PromptPrefixCachePool::from_env())),
            model_file_lifecycle: Arc::new(RwLock::new(())),
            model_transition: Arc::new(tokio::sync::Mutex::new(())),
            #[cfg(windows)]
            model_delete_nonce: Arc::from(uuid::Uuid::new_v4().to_string()),
            allow_local_model_delete: false,
            generation_sessions: Arc::new(RwLock::new(HashMap::new())),
            verification_reports: Arc::new(RwLock::new(HashMap::new())),
            workspace_sessions: workspace::WorkspaceSessionManager::default(),
            workspace_cli_token: None,
            serve_addr: SocketAddr::from(([127, 0, 0, 1], 8181)),
            engine: engine::EngineHandle::spawn(),
            planner_env: PlannerEnv::capture(),
            configured_threads: None,
            default_enable_thinking: false,
            models_dir: resolve_models_dir(None),
        }
    }
}

impl AppState {
    pub fn with_configured_threads(configured_threads: Option<usize>) -> Self {
        Self {
            configured_threads,
            ..Self::default()
        }
    }

    /// Set the server-wide default for opt-in thinking mode
    /// (`serve --enable-thinking`). An explicit `camelid_enable_thinking` in a
    /// request always overrides this; the default only fills in when the request
    /// is silent.
    pub fn with_default_enable_thinking(mut self, default_enable_thinking: bool) -> Self {
        self.default_enable_thinking = default_enable_thinking;
        self
    }

    fn with_local_model_delete(mut self, allow: bool) -> Self {
        self.allow_local_model_delete = allow;
        self
    }

    /// Set the models directory (`serve --models-dir` / `CAMELID_MODELS_DIR`).
    /// `None` keeps the default: the first existing of `<exe dir>/models` or
    /// `./models` (mirroring `auto_select_model`'s shipped-layout precedent),
    /// falling back to `./models`. The value is resolved to an absolute path
    /// here, once, so it stays stable however the process was launched.
    pub fn with_models_dir(mut self, models_dir: Option<PathBuf>) -> Self {
        self.models_dir = resolve_models_dir(models_dir);
        self
    }

    fn with_serve_addr(mut self, serve_addr: SocketAddr) -> Self {
        self.serve_addr = serve_addr;
        self
    }

    fn with_workspace_cli_token(mut self, token: Option<&str>) -> Self {
        self.workspace_cli_token = token.map(Arc::from);
        self
    }

    /// Register a loaded gemma4 runtime under a model id, exactly as the
    /// `CAMELID_GEMMA4_SERVE` load path would. For integration tests that drive
    /// the live chat routes against a real runtime without the full
    /// model-install flow.
    pub async fn insert_gemma4_runtime_for_tests(
        &self,
        id: &str,
        runtime: crate::gemma4_runtime::Gemma4Runtime,
    ) {
        self.gemma4_runtimes
            .write()
            .await
            .insert(id.to_string(), Arc::new(Gemma4ServeRuntime::Local(runtime)));
    }

    /// Register a distributed gemma4 runtime under a model id, exactly as the
    /// `CAMELID_GEMMA4_WORKER`/`CAMELID_GEMMA4_SPLIT` load path would. For
    /// integration tests that drive the live chat routes against a loopback
    /// worker without the full model-install flow.
    pub async fn insert_gemma4_distributed_runtime_for_tests(
        &self,
        id: &str,
        runtime: crate::gemma4_distributed::Gemma4DistributedRuntime,
    ) {
        self.gemma4_runtimes.write().await.insert(
            id.to_string(),
            Arc::new(Gemma4ServeRuntime::Distributed(runtime)),
        );
    }
}

#[derive(Clone)]
struct CachedPromptPrefix {
    model_id: String,
    model_path: PathBuf,
    token_ids: Vec<u32>,
    sampling: SamplingConfig,
    session: LlamaInferenceSession,
    logits: CpuTensor,
    hidden_state: CpuTensor,
    output_norm_state: CpuTensor,
}

#[derive(Clone)]
struct PromptPrefixCacheEntry {
    cached: Arc<CachedPromptPrefix>,
    last_used: std::time::Instant,
}

struct PromptPrefixCachePool {
    entries: Vec<PromptPrefixCacheEntry>,
    capacity: usize,
}

impl PromptPrefixCachePool {
    fn from_env() -> Self {
        let capacity = env::var(PROMPT_PREFIX_CACHE_CAPACITY_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PROMPT_PREFIX_CACHE_CAPACITY);
        Self::with_capacity(capacity)
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn insert(&mut self, cached: CachedPromptPrefix) {
        let cached = Arc::new(cached);
        let now = std::time::Instant::now();

        if let Some(existing) = self.entries.iter_mut().find(|entry| {
            entry.cached.model_id == cached.model_id
                && entry.cached.model_path == cached.model_path
                && entry.cached.token_ids == cached.token_ids
                && entry.cached.sampling == cached.sampling
        }) {
            existing.cached = cached;
            existing.last_used = now;
            return;
        }

        if self.entries.len() >= self.capacity {
            if let Some(lru_idx) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
            {
                self.entries.remove(lru_idx);
            }
        }

        self.entries.push(PromptPrefixCacheEntry {
            cached,
            last_used: now,
        });
    }
}

#[derive(Clone)]
struct PrefixMatchResult {
    cached: Arc<CachedPromptPrefix>,
    prefix_len: usize,
    is_exact_match: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LoadedModel {
    pub id: String,
    pub path: PathBuf,
    pub gguf: GgufFile,
    pub llama_config: Option<LlamaModelConfig>,
    pub llama_tensors: Option<LlamaTensorBinding>,
    pub unsupported_runtime: Option<UnsupportedRuntimeSummary>,
    pub tokenizer: TokenizerLoadState,
    #[serde(skip)]
    pub tokenizer_runtime: Option<Arc<Tokenizer>>,
    /// Receipt lane identity (exact GGUF hash, quantization, provenance),
    /// hashed once at load time. Identifies the lane for parity receipts; it
    /// makes no support claim about the lane.
    pub lane: LaneIdentity,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnsupportedRuntimeSummary {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum TokenizerLoadState {
    Available(TokenizerSummary),
    Unavailable { code: &'static str, message: String },
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenizerSummary {
    pub model: &'static str,
    pub token_count: usize,
    pub byte_token_count: usize,
    pub special: SpecialTokenSummary,
    pub config: TokenizerConfigSummary,
    pub chat_template: Option<ChatTemplateSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatTemplateSummary {
    pub source: &'static str,
    pub detected_format: &'static str,
    pub length: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SpecialTokenSummary {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub eom: Option<u32>,
    pub unk: Option<u32>,
    pub sep: Option<u32>,
    pub pad: Option<u32>,
    pub mask: Option<u32>,
    pub eog: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TokenizerConfigSummary {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub path: PathBuf,
    pub id: Option<String>,
    /// Release every OTHER resident model before loading this one ("switch
    /// model" semantics), then re-probe live memory and re-run the fit
    /// preflight against the freed state.
    ///
    /// Defaults to `false`, so an existing API client's behavior is unchanged:
    /// models still accumulate unless it opts in. The web UI sends `true`,
    /// which is what makes picking a model in the app a swap rather than a
    /// second resident copy that the fit guard then refuses.
    #[serde(default)]
    pub replace: bool,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub engine: &'static str,
    pub loaded_now: bool,
    pub generation_ready: bool,
    pub active_model_id: Option<String>,
    pub q8_runtime: Q8RuntimeHealth,
    pub execution_plan: Option<ExecutionPlan>,
    /// Which backend serves the active model: "gemma4-runtime", "runnable-runtime",
    /// "diffusion-gemma-runtime", "llama", or "none".
    pub backend: &'static str,
    /// Model family of the active model ("gemma4", "llama-family", ...), if loaded.
    pub model_family: Option<&'static str>,
    /// True when the gemma4 serve path is built (CAMELID_GEMMA4_SERVE) and a gemma4
    /// runtime is loaded for the active model.
    pub gemma4_available: bool,
    /// Engine-queue backpressure gauge: generation jobs accepted and not yet
    /// finished (queued + running). Bounded by CAMELID_QUEUE_DEPTH; beyond the
    /// bound requests get a typed 503 (`engine_queue_full`).
    pub engine_queue_depth: usize,
}

#[derive(Debug, Serialize)]
pub struct Q8RuntimeHealth {
    pub policy: &'static str,
    pub lazy_q8_linear: bool,
    pub retain_q8_blocks: bool,
    pub file_cache_bytes: Option<u64>,
    pub note: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CapabilitiesResponse {
    pub engine: &'static str,
    pub gguf_metadata: bool,
    pub tensor_loading: bool,
    pub tokenization: bool,
    pub inference: bool,
    pub streaming: bool,
    pub model_downloads: bool,
    pub hf_catalog_install: bool,
    pub execution_plan: Option<ExecutionPlan>,
    pub support_contract: SupportContract,
    pub supported_quantization: Vec<SupportItem>,
    pub planned_quantization: Vec<SupportItem>,
    pub supported_model_families: Vec<SupportItem>,
    pub planned_model_families: Vec<SupportItem>,
    pub model_compatibility: Vec<ModelCompatibilityTarget>,
    pub api_features: Vec<SupportItem>,
    pub notes: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct SupportContract {
    pub current_gate: &'static str,
    pub support_policy: &'static str,
    pub unsupported_policy: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SupportItem {
    pub id: &'static str,
    pub status: &'static str,
    pub notes: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelCompatibilityTarget {
    pub id: &'static str,
    pub family: &'static str,
    pub quantization: &'static str,
    pub status: &'static str,
    /// Verified (via the agent-eval harness) to drive a clean tool-call
    /// round-trip. Promoted only with a PASS receipt; default false.
    pub tool_capable: bool,
    pub support_scope: &'static str,
    pub full_support_status: &'static str,
    pub full_support_blockers: &'static str,
    pub metadata_parses: &'static str,
    pub tokenizer_works: &'static str,
    pub tensors_load: &'static str,
    pub generation_runs: &'static str,
    pub parity_audited: &'static str,
    pub performance_measured: &'static str,
    pub frontend_load_path_verified: &'static str,
    pub frontend_readiness_gate: &'static str,
    pub tested_context: &'static str,
    pub chat_template_renderer: &'static str,
    pub chat_template_shape_pack: &'static str,
    pub chat_template_shape_pack_id: &'static str,
    pub bounded_context_512_pack: &'static str,
    pub bounded_context_512_pack_id: &'static str,
    pub bounded_context_window: u32,
    pub bounded_context_1024_pack: &'static str,
    pub bounded_context_1024_pack_id: &'static str,
    pub bounded_context_1024_window: u32,
    pub bounded_context_2048_pack: &'static str,
    pub bounded_context_2048_pack_id: &'static str,
    pub bounded_context_2048_window: u32,
    pub bounded_context_4096_pack: &'static str,
    pub bounded_context_4096_pack_id: &'static str,
    pub bounded_context_4096_window: u32,
    pub bounded_context_8192_pack: &'static str,
    pub bounded_context_8192_pack_id: &'static str,
    pub bounded_context_8192_window: u32,
    pub latest_checked_bucket: &'static str,
    pub latest_checked_result: &'static str,
    pub latest_checked_output: &'static str,
    pub evidence: &'static str,
    pub next_step: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelListItem>,
}

#[derive(Debug, Serialize)]
pub struct ModelListItem {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub meta: Option<ModelListMeta>,
}

#[derive(Debug, Serialize)]
pub struct ModelListMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_vocab: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx_train: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_embd: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_params: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_type: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Minimum-probability sampler filter (llama.cpp `min_p`). Keeps only tokens
    /// with probability >= `min_p * max_probability`. `None`/`0.0` disables it.
    pub min_p: Option<f32>,
    /// Multiplicative repetition penalty (llama.cpp `repeat_penalty`). `1.0`/`None`
    /// is a no-op; values `> 1.0` discourage repeating recent tokens.
    pub repeat_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
    pub camelid_dense_diagnostic_generated_index: Option<u32>,
    /// Optional hard ceiling for the exact rendered prompt plus generation.
    /// Workspace sets this private extension; ordinary OpenAI callers omit it.
    pub camelid_context_budget_tokens: Option<u32>,
    /// Opt-in: attach a parity receipt to the (non-streaming) response. The
    /// receipt is a claim of output for the verifier to check â€” no reference
    /// runs here, so its parity block is emitted as not-compared.
    pub camelid_receipt: Option<bool>,
    /// Opt-in gemma4 thinking mode: renders the reference's enable_thinking
    /// template (system turn opens with the `<|think|>` token). Thinking
    /// channels are stripped from chat output either way. Default: false (the
    /// reference's `enable_thinking:false` rendering).
    pub camelid_enable_thinking: Option<bool>,
    /// OpenAI-style tool/function definitions. When present, they are rendered
    /// into the prompt through the loaded model's own chat template (Hybrid agent
    /// mode); the model's tool-call output is parsed back into `tool_calls` (for
    /// templates that render tools â€” Llama 3.x etc.). Models whose template does
    /// not render tools simply ignore them.
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI `tool_choice`: `"auto"` (default), `"none"` (suppress parsing), or
    /// `"required"`/a specific function (treated as `auto`). Parsed permissively
    /// as a raw value. Declaring it here removes it from `unsupported_fields`.
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI `parallel_tool_calls`: accepted and ignored (Camelid surfaces the
    /// tool calls the model actually emits). Declared here so it is not rejected.
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI `response_format`. `{"type":"json_object"}` turns on JSON-grammar-
    /// constrained decoding (output guaranteed valid JSON). `{"type":"json_schema",
    /// "json_schema":{"schema":{...}}}` constrains the output to a supported JSON
    /// Schema subset -- see `docs/architecture/STRUCTURED_OUTPUTS.md` for the exact
    /// contract; schemas outside the subset are a typed 400 naming the keyword.
    /// `{"type":"text"}`/absent is normal decoding; other types are rejected.
    /// Non-streaming only. Declared here so it is not in `unsupported_fields`.
    pub response_format: Option<serde_json::Value>,
    /// OpenAI `stream_options`. The only honored subfield is `include_usage`
    /// (bool); any other shape or subfield is tolerated silently and ignored,
    /// matching the permissive llama-server oracle. Parsed as a raw value so a
    /// malformed `stream_options` never rejects the request (see
    /// `stream_options_include_usage`). Declaring it here also removes it from
    /// `unsupported_fields`, so the chat route no longer returns the old
    /// "stream_options are not supported yet" error.
    pub stream_options: Option<serde_json::Value>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub stream: Option<bool>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Minimum-probability sampler filter (llama.cpp `min_p`).
    pub min_p: Option<f32>,
    /// Multiplicative repetition penalty (llama.cpp `repeat_penalty`).
    pub repeat_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub best_of: Option<u32>,
    pub logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_prompt_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
    pub camelid_dense_diagnostic_generated_index: Option<u32>,
    pub camelid_receipt: Option<bool>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Content-part types from the OpenAI parts form that Camelid cannot honor
    /// (`image_url`, `input_audio`, `video_url`, â€¦). Camelid generates text
    /// tokens only â€” vision/audio towers are never loaded â€” so the chat
    /// handlers fail closed with a typed `unsupported_multimodal_content`
    /// error whenever this is non-empty. Plain-string content and `text` parts
    /// never populate it.
    #[serde(skip)]
    pub unsupported_content_parts: Vec<String>,
}

/// Wire shape for `ChatMessage` deserialization: accepts the OpenAI plain
/// string form and the content-parts array form. `text` parts are concatenated
/// in order; every non-text part records its `type` so the handler can reject
/// the request with a typed error instead of a generic JSON parse failure.
#[derive(Deserialize)]
struct ChatMessageWire {
    role: String,
    content: ChatContentWire,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChatContentWire {
    Text(String),
    Parts(Vec<ChatContentPartWire>),
}

#[derive(Deserialize)]
struct ChatContentPartWire {
    #[serde(rename = "type")]
    part_type: String,
    text: Option<String>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ChatMessageWire::deserialize(deserializer)?;
        let (content, unsupported_content_parts) = match wire.content {
            ChatContentWire::Text(text) => (text, Vec::new()),
            ChatContentWire::Parts(parts) => {
                let mut text = String::new();
                let mut unsupported = Vec::new();
                for part in parts {
                    if part.part_type == "text" {
                        text.push_str(part.text.as_deref().unwrap_or(""));
                    } else {
                        unsupported.push(part.part_type);
                    }
                }
                (text, unsupported)
            }
        };
        Ok(ChatMessage {
            role: wire.role,
            content,
            unsupported_content_parts,
        })
    }
}

/// Fail-closed guard for multimodal chat input. Returns the typed error
/// response when any message carries non-text content parts.
fn reject_unsupported_multimodal_content(messages: &[ChatMessage]) -> Option<Response> {
    let mut part_types: Vec<&str> = messages
        .iter()
        .flat_map(|m| m.unsupported_content_parts.iter().map(String::as_str))
        .collect();
    if part_types.is_empty() {
        return None;
    }
    part_types.sort_unstable();
    part_types.dedup();
    Some(api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_multimodal_content",
        format!(
            "unsupported multimodal content part(s): {}. Camelid is a text-token \
             inference engine; image/audio/video inputs are fail-closed for every \
             model row (Gemma 4 vision/audio towers are never loaded). Send content \
             as a string or as {{\"type\":\"text\"}} parts.",
            part_types.join(", ")
        ),
        Some("messages"),
    ))
}

enum PromptInput {
    Text(String),
    Chat(Vec<ChatMessage>),
    TokenIds(Vec<u32>),
}

#[derive(Debug, Deserialize)]
pub struct TokenizerEncodeRequest {
    pub text: Option<String>,
    pub add_special: Option<bool>,
    pub parse_special: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TokenizerEncodeResponse {
    pub tokens: Vec<u32>,
    pub token_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerDecodeRequest {
    pub tokens: Option<Vec<u32>>,
    pub remove_special: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TokenizerDecodeResponse {
    pub text: String,
    pub token_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerTokenizeRequest {
    pub content: Option<String>,
    pub add_special: Option<bool>,
    pub parse_special: Option<bool>,
    pub with_pieces: Option<bool>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerTokenizeResponse {
    pub tokens: Vec<LlamaServerTokenizeToken>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum LlamaServerTokenizeToken {
    Id(u32),
    Piece(LlamaServerTokenPiece),
}

#[derive(Debug, Serialize)]
pub struct LlamaServerTokenPiece {
    pub id: u32,
    pub piece: LlamaServerTokenPieceValue,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum LlamaServerTokenPieceValue {
    Text(String),
    Bytes(Vec<u8>),
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerDetokenizeRequest {
    pub tokens: Option<Vec<u32>>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerDetokenizeResponse {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerApplyTemplateRequest {
    pub messages: Option<Vec<ChatMessage>>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerApplyTemplateResponse {
    pub prompt: String,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerCompletionRequest {
    pub model: Option<String>,
    pub prompt: Option<LlamaServerCompletionPrompt>,
    pub n_predict: Option<i32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub temp: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Minimum-probability sampler filter (llama.cpp `min_p`).
    pub min_p: Option<f32>,
    /// Multiplicative repetition penalty (llama.cpp `repeat_penalty`).
    pub repeat_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum LlamaServerCompletionPrompt {
    Text(String),
    TokenIds(Vec<u32>),
}

type LlamaServerCompletionPromptParts = (Option<String>, Option<Vec<u32>>);

#[derive(Debug, Serialize)]
pub struct LlamaServerCompletionResponse {
    pub content: String,
    pub model: String,
    pub stop: bool,
    pub stopped_limit: bool,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    pub camelid: LlamaServerCompletionCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerCompletionCamelid {
    pub compatibility: &'static str,
    pub finish_reason: &'static str,
    pub unsupported: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelListResponse {
    pub data: Vec<LlamaServerModelListItem>,
    pub camelid: LlamaServerModelListCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelListItem {
    pub id: String,
    pub path: Option<String>,
    pub status: LlamaServerModelStatus,
    pub architecture: LlamaServerModelArchitecture,
    pub camelid: LlamaServerModelCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelStatus {
    pub value: &'static str,
    pub args: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelArchitecture {
    pub input_modalities: Vec<&'static str>,
    pub output_modalities: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelCamelid {
    pub generation_ready: bool,
    pub model_path_redacted: bool,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelListCamelid {
    pub compatibility: &'static str,
    pub scope: &'static str,
    pub unsupported: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelLoadResponse {
    pub data: LlamaServerModelListItem,
    pub camelid: LlamaServerModelLoadCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModelLoadCamelid {
    pub compatibility: &'static str,
    pub scope: &'static str,
    pub model_path_redacted: bool,
    pub unsupported: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerPropsResponse {
    pub default_generation_settings: LlamaServerDefaultGenerationSettings,
    pub total_slots: u32,
    pub model_path: Option<String>,
    pub model_id: Option<String>,
    pub chat_template: Option<String>,
    pub chat_template_caps: serde_json::Value,
    pub modalities: LlamaServerModalities,
    pub build_info: &'static str,
    pub is_sleeping: bool,
    pub camelid: LlamaServerPropsCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerDefaultGenerationSettings {
    pub id: u32,
    pub id_task: i32,
    pub n_ctx: u32,
    pub speculative: bool,
    pub is_processing: bool,
    pub params: LlamaServerDefaultGenerationParams,
    pub prompt: &'static str,
    pub next_token: LlamaServerNextTokenProps,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerDefaultGenerationParams {
    pub n_predict: i32,
    pub seed: u32,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop: Vec<String>,
    pub max_tokens: i32,
    pub ignore_eos: bool,
    pub stream: bool,
    pub n_probs: u32,
    pub samplers: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerNextTokenProps {
    pub has_next_token: bool,
    pub has_new_line: bool,
    pub n_remain: i32,
    pub n_decoded: u32,
    pub stopping_word: &'static str,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerModalities {
    pub vision: bool,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerPropsCamelid {
    pub compatibility: &'static str,
    pub generation_ready: bool,
    pub model_path_redacted: bool,
    pub unsupported: Vec<&'static str>,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerReadOnlyQuery {
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerModelsLoadRequest {
    pub model: Option<PathBuf>,
    pub path: Option<PathBuf>,
    pub id: Option<String>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerSlotsQuery {
    pub fail_on_no_slot: Option<String>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerSlotResponse {
    pub id: u32,
    pub id_task: i32,
    pub n_ctx: u32,
    pub speculative: bool,
    pub is_processing: bool,
    pub params: LlamaServerDefaultGenerationParams,
    pub prompt: &'static str,
    pub next_token: LlamaServerNextTokenProps,
    pub camelid: LlamaServerSlotCamelid,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerSlotCamelid {
    pub compatibility: &'static str,
    pub generation_ready: bool,
    pub status: &'static str,
    /// Engine-queue backpressure gauge: generation jobs accepted and not yet
    /// finished (queued + running). See `HealthResponse::engine_queue_depth`.
    pub engine_queue_depth: usize,
    pub unsupported: Vec<&'static str>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GenerationSessionRequest {
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// Minimum-probability sampler filter (llama.cpp `min_p`).
    pub min_p: Option<f32>,
    /// Multiplicative repetition penalty (llama.cpp `repeat_penalty`).
    pub repeat_penalty: Option<f32>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub best_of: Option<u32>,
    pub completion_logprobs: Option<u32>,
    pub chat_logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_prompt_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
    pub camelid_dense_diagnostic_generated_index: Option<u32>,
    pub camelid_context_budget_tokens: Option<u32>,
    /// Opt-in Qwen3/gemma4 thinking mode: when true the chat renderer emits the
    /// template's thinking generation prompt (the model produces its own
    /// `<think>â€¦</think>` block) instead of the deterministic thinking-disabled
    /// shape. `None`/false preserves the parity-locked thinking-disabled default.
    pub camelid_enable_thinking: Option<bool>,
    /// Tool/function definitions, rendered into the prompt via the model's chat
    /// template (agent mode). `None` renders identically to before.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
    #[serde(default, skip_deserializing)]
    default_max_tokens_cap: Option<u32>,
    /// Active output constraint (`response_format: json_object` / `json_schema`),
    /// compiled by the chat handler; never deserialized from the wire.
    #[serde(default, skip_deserializing)]
    constraint: Option<crate::grammar::ConstraintSpec>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationSessionSummary {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub prompt_token_count: usize,
    pub max_tokens: u32,
    pub state: &'static str,
    pub dense_session_ready: bool,
    pub next_step: &'static str,
}

#[derive(Debug, Serialize)]
pub struct GenerationSessionListResponse {
    pub object: &'static str,
    pub data: Vec<GenerationSessionSummary>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
    pub camelid: GenerationDiagnostics,
    /// Present only when the request opted in via `camelid_receipt: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camelid_receipt: Option<ParityReceipt>,
    /// Serve-lane disclosure. Present and `"experimental"` only when the active
    /// model is an implemented decoder that is NOT a supported exact row â€” the
    /// output is unverified and carries no parity claim. Omitted for supported rows
    /// (whose support is asserted by `/api/capabilities`, never by this field) and
    /// is NEVER a parity receipt: the live token stream is not evidence of support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lane: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct GenerationDiagnostics {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub dense_metadata: DenseDiagnosticMetadata,
    pub top_logits: Vec<LogitDiagnostic>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub step_top_logits: Vec<Vec<LogitDiagnostic>>,
    pub output_projection: Vec<LlamaOutputProjectionDiagnostic>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense: Option<LlamaForwardDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_diagnostic_generated_index: Option<usize>,
    pub timings_ms: GenerationTimings,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseDiagnosticMetadata {
    pub embedding_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub head_dim: usize,
    pub rope_dimension_count: usize,
    pub rope_freq_base: f32,
    pub rope_scaling_type: String,
    pub rope_scaling_factor: Option<f32>,
    pub rope_scaling_original_context_length: Option<u32>,
    pub rope_scaling_low_freq_factor: Option<f32>,
    pub rope_scaling_high_freq_factor: Option<f32>,
    pub rope_pairing: &'static str,
    pub rope_direction: &'static str,
    pub rope_position_mode: &'static str,
    pub gqa_head_mapping: &'static str,
    pub attention_score_scale: &'static str,
    pub linear_accumulation: &'static str,
    pub ffn_gate_up_order: &'static str,
    pub rms_norm_epsilon: f32,
    pub rms_norm_effective_epsilon: f32,
    pub square_linear_diagnostic_layout: &'static str,
    pub rectangular_linear_diagnostic_layout: &'static str,
    pub token_embedding_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_is_tied_embedding: bool,
    pub output_projection_layout: &'static str,
    pub output_projection_diagnostic_layout: &'static str,
    pub zero_attention_delta: String,
    pub zero_ffn_delta: String,
    pub projection_orientations: DenseProjectionOrientations,
}

#[derive(Debug, Clone, Serialize)]
pub struct DenseProjectionOrientations {
    pub attention_q: LinearProjectionOrientation,
    pub attention_k: LinearProjectionOrientation,
    pub attention_v: LinearProjectionOrientation,
    pub attention_output: LinearProjectionOrientation,
    pub ffn_gate: LinearProjectionOrientation,
    pub ffn_up: LinearProjectionOrientation,
    pub ffn_down: LinearProjectionOrientation,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinearProjectionOrientation {
    pub shape: Vec<usize>,
    pub input_width: usize,
    pub output_width: usize,
    pub descriptor_layout: &'static str,
    pub runtime_interpretation: &'static str,
    pub square_diagnostic_applies: bool,
}

#[derive(Debug, Serialize)]
pub struct LogitDiagnostic {
    pub token_id: u32,
    pub logit: f32,
    pub probability: f32,
    pub rank: usize,
    pub selected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Default, Serialize)]
pub struct GenerationTimings {
    pub tokenize: u128,
    pub weight_load: u128,
    pub weight_cache_hit: bool,
    pub prompt_cache_hit: bool,
    pub session_create: u128,
    pub generate: u128,
    pub generation: GenerationPhaseTimings,
    pub prompt_evaluation: PromptEvaluationTimings,
    pub layers: Vec<GenerationLayerTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<crate::inference::LlamaForwardMemoryTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q8_schedule: Option<LlamaQ8ScheduleTelemetry>,
}

#[derive(Debug, Default, Serialize)]
pub struct PromptEvaluationTimings {
    pub prompt_token_count: usize,
    pub prefill_token_count: usize,
    pub first_token_evaluated: bool,
    pub prefill: GenerationPhaseTimings,
    pub first_token: GenerationPhaseTimings,
    pub prefill_layers: Vec<GenerationLayerTimings>,
    pub first_token_layers: Vec<GenerationLayerTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_memory: Option<crate::inference::LlamaForwardMemoryTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token_memory: Option<crate::inference::LlamaForwardMemoryTimings>,
}

#[derive(Debug, Default, Serialize)]
pub struct GenerationPhaseTimings {
    pub forward_total: f64,
    pub embedding: f64,
    pub layers_total: f64,
    pub final_norm: f64,
    pub logits: f64,
    pub sample: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct GenerationLayerTimings {
    pub layer_index: usize,
    pub total: f64,
    pub attention_norm: f64,
    pub attention_q: f64,
    pub attention_k: f64,
    pub attention_v: f64,
    pub attention_rope: f64,
    pub kv_cache_write: f64,
    pub attention_context: f64,
    pub attention_output: f64,
    pub attention_residual: f64,
    pub ffn_norm: f64,
    pub ffn_gate: f64,
    pub ffn_up: f64,
    pub ffn_activation: f64,
    pub ffn_down: f64,
    pub ffn_residual: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<LlamaLayerMemoryTimings>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: &'static str,
    /// OpenAI per-token logprobs; present only when `logprobs:true` was requested
    /// (non-streaming, single choice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// OpenAI chat `logprobs` object: one `content` entry per generated token.
#[derive(Debug, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatLogprobContent>,
}

#[derive(Debug, Serialize)]
pub struct ChatLogprobContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<ChatTopLogprob>,
}

#[derive(Debug, Serialize)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: String,
    /// Parsed structured tool calls (OpenAI shape); present only when the model
    /// emitted a tool call and the request supplied `tools`. `content` is empty
    /// when this is set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// OpenAI tool-call object surfaced in an assistant message.
#[derive(Debug, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI shape).
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
    pub camelid: GenerationDiagnostics,
    /// Present only when the request opted in via `camelid_receipt: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camelid_receipt: Option<ParityReceipt>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
    /// OpenAI legacy-completions logprobs; present only when `logprobs:N` was
    /// requested (non-streaming, single choice).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogprobs>,
}

/// OpenAI legacy `/v1/completions` `logprobs` object (parallel arrays).
#[derive(Debug, Serialize)]
pub struct CompletionLogprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<std::collections::BTreeMap<String, f32>>,
    pub text_offset: Vec<usize>,
}

#[derive(Debug, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camelid: Option<serde_json::Value>,
    /// Terminal usage frame for OpenAI `stream_options.include_usage`. `None` on
    /// every role/content/finish chunk, so the field is omitted from the wire
    /// (matching the llama-server oracle, which omits `usage` on content chunks
    /// rather than sending `usage: null`, and keeping the usage-off baseline
    /// byte-identical). `Some` only on the single terminal chunk that carries an
    /// empty `choices` array, emitted after the finish_reason chunk and before
    /// `[DONE]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camelid: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct CompletionStreamChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

/// Why a decode loop stopped early, checked cooperatively at the top of every
/// generated-token step (one relaxed atomic load + one clock read per token).
enum GenerationStopCause {
    /// The request's `CancellationToken` fired — the handler frame that owned
    /// this generation is gone (client disconnect, handler drop).
    Cancelled,
    /// The engine-enforced wall-clock deadline passed. Replaces the old
    /// handler-side `tokio::time::timeout`, which detached (but could not
    /// stop) the blocking decode.
    TimedOut,
}

/// Cooperative stop signal + engine-enforced deadline for one generation.
/// The token is fired by `CancelOnDrop` when the owning handler frame drops;
/// the deadline is armed by `generate_decoded_tokens_blocking` from
/// `CAMELID_GENERATION_TIMEOUT_MS`. The decode loop itself observes both, so
/// compute always stops within one token step of either trip — the handler
/// never detaches a running decode.
#[derive(Clone)]
struct GenerationCancel {
    token: tokio_util::sync::CancellationToken,
    deadline: Option<Instant>,
    timeout: Duration,
    started: Instant,
}

impl GenerationCancel {
    fn unbounded() -> Self {
        Self {
            token: tokio_util::sync::CancellationToken::new(),
            deadline: None,
            timeout: Duration::ZERO,
            started: Instant::now(),
        }
    }

    /// Arm the wall-clock deadline at decode entry (same coverage as the old
    /// handler-side timer: prep/tokenization are not counted).
    fn arm(&mut self, timeout: Duration) {
        self.started = Instant::now();
        self.timeout = timeout;
        self.deadline = self.started.checked_add(timeout);
    }

    fn tripped(&self) -> Option<GenerationStopCause> {
        if self.token.is_cancelled() {
            return Some(GenerationStopCause::Cancelled);
        }
        match self.deadline {
            Some(deadline) if Instant::now() >= deadline => Some(GenerationStopCause::TimedOut),
            _ => None,
        }
    }
}

/// RAII: fires the generation's CancellationToken when the owning frame is
/// dropped for ANY reason — client disconnect (hyper drops the handler
/// future), panic, or normal return (harmless then; the token is no longer
/// read). This is the Rust equivalent of llama.cpp's
/// `~server_response_reader → stop() → SERVER_TASK_TYPE_CANCEL`.
struct CancelOnDrop(tokio_util::sync::CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Response for a decode stopped by cancellation. Never delivered to the
/// (already disconnected) client; observable only in logs and tests.
fn generation_cancelled_response(generated_tokens: usize) -> Box<Response> {
    Box::new(api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "generation_cancelled",
        format!(
            "generation stopped after {generated_tokens} tokens: the request was cancelled before the decode finished"
        ),
        None,
    ))
}

struct PreparedGeneration {
    /// Prevent unload or deletion while preparation/decode owns file-backed
    /// model state. Test-only synthetic generations do not need a lease.
    _model_file_lease: Option<tokio::sync::OwnedRwLockReadGuard<()>>,
    model_id: String,
    model_path: PathBuf,
    token_ids: Vec<u32>,
    max_tokens: u32,
    tokenizer: Arc<Tokenizer>,
    session: LlamaInferenceSession,
    sampling: SamplingConfig,
    /// Stop signal + deadline observed at the top of every decode step.
    cancel: GenerationCancel,
    /// When set, collect per-token logprobs each step (chosen + this many top
    /// alternatives). Forces the full-host-logits decode path (no GPU greedy-fast).
    logprobs_top_n: Option<usize>,
    /// Active output constraint (`response_format` json_object / json_schema): each
    /// step is masked to tokens that keep a valid prefix of the constrained value.
    /// Forces the full-logits CPU decode path.
    constraint: Option<crate::grammar::ConstraintSpec>,
    stop_sequences: Vec<String>,
    logit_diagnostic_token_ids: Vec<u32>,
    collect_dense_diagnostics: bool,
    dense_diagnostic_generated_index: Option<usize>,
    dense_metadata: DenseDiagnosticMetadata,
    timings: GenerationTimings,
    cached_prompt_prefix: Arc<Mutex<PromptPrefixCachePool>>,
    speculative: Option<PreparedSpeculative>,
    /// Captured request identity for the live telemetry stream; taken by the
    /// generation path that actually runs (streaming or blocking).
    telemetry: Option<telemetry::RequestStart>,
}

/// Server-level speculative decoding mode (`CAMELID_SPEC_DECODE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecDecodeMode {
    NGram,
    DraftModel,
}

fn spec_decode_mode_from_env() -> Option<SpecDecodeMode> {
    match env::var(SPEC_DECODE_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("ngram") => Some(SpecDecodeMode::NGram),
        Ok(value) if value.eq_ignore_ascii_case("draft") => Some(SpecDecodeMode::DraftModel),
        _ => None,
    }
}

/// Run speculative decode on the GPU (CAMELID_SPEC_GPU=1): keep the target's
/// resident decode engine active during speculation and verify drafts via the
/// batched GPU `verify_batch` instead of the CPU chunk verify. Opt-in while this
/// lands; lossless either way (the target verify is authoritative), so the flag
/// only changes where the work runs.
fn spec_gpu_enabled() -> bool {
    matches!(
        env::var("CAMELID_SPEC_GPU").ok().as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

fn spec_draft_tokens_from_env(default: usize) -> usize {
    env::var(SPEC_DRAFT_TOKENS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn spec_ngram_min_from_env() -> usize {
    env::var(SPEC_NGRAM_MIN_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_NGRAM_MIN_MATCH)
}

fn spec_ngram_max_from_env() -> usize {
    env::var(SPEC_NGRAM_MAX_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_NGRAM_MAX_MATCH)
}

fn prompt_prefix_cache_min_tokens_from_env() -> usize {
    env::var(PROMPT_PREFIX_CACHE_MIN_TOKENS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PROMPT_PREFIX_CACHE_MIN_TOKENS)
}

/// Per-request speculative decoding state: the drafter plus round counters
/// for the end-of-request acceptance summary.
struct PreparedSpeculative {
    drafter: SpeculativeDrafter,
    draft_tokens: usize,
    latch: SpecLatch,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
}

/// One token's logprob plus its decoded piece and raw UTF-8 bytes (OpenAI-shaped).
#[derive(Debug, Clone)]
struct TokenLogprob {
    token: String,
    logprob: f32,
    bytes: Vec<u8>,
}

/// Per generated-token logprob record: the chosen token plus the top-N
/// alternatives by probability. Collected only when logprobs are requested.
#[derive(Debug, Clone)]
struct StepLogprob {
    chosen: TokenLogprob,
    top: Vec<TokenLogprob>,
}

struct GeneratedText {
    text: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    dense_metadata: DenseDiagnosticMetadata,
    top_logits: Vec<LogitDiagnostic>,
    step_top_logits: Vec<Vec<LogitDiagnostic>>,
    /// Per-token logprobs (chosen + top-N); empty unless logprobs were requested.
    step_logprobs: Vec<StepLogprob>,
    output_projection: Vec<LlamaOutputProjectionDiagnostic>,
    dense: Option<LlamaForwardDiagnostics>,
    dense_diagnostic_generated_index: Option<usize>,
    completion_tokens: usize,
    finish_reason: &'static str,
    timings: GenerationTimings,
    /// Execution-trace rollup `(digest, fold_count)` from a deterministic run, else `None`.
    execution_trace: Option<(String, u64)>,
}

struct GeneratedTokens {
    prompt_token_ids: Vec<u32>,
    token_ids: Vec<u32>,
    dense_metadata: DenseDiagnosticMetadata,
    top_logits: Vec<RawLogitDiagnostic>,
    step_top_logits: Vec<Vec<RawLogitDiagnostic>>,
    /// Per-token logprobs (chosen + top-N); empty unless logprobs were requested.
    step_logprobs: Vec<StepLogprob>,
    output_projection: Vec<LlamaOutputProjectionDiagnostic>,
    dense: Option<LlamaForwardDiagnostics>,
    dense_diagnostic_generated_index: Option<usize>,
    finish_reason: &'static str,
    timings: GenerationTimings,
    /// Execution-trace rollup `(digest, fold_count)` captured during a deterministic run, or
    /// `None` when the trace was not armed (any non-deterministic generation).
    execution_trace: Option<(String, u64)>,
}

fn collect_dense_diagnostics_for_generated_index(
    prepared: &PreparedGeneration,
    generated_index: usize,
) -> bool {
    if !prepared.collect_dense_diagnostics {
        return false;
    }
    prepared
        .dense_diagnostic_generated_index
        .map(|target| target == generated_index)
        .unwrap_or(true)
}

#[derive(Clone)]
struct RawLogitDiagnostic {
    token_id: u32,
    logit: f32,
    probability: f32,
    rank: usize,
    selected: bool,
}

#[derive(Debug, Serialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub code: &'static str,
    pub param: Option<&'static str>,
}

/// The models directory used when `serve` gets no `--models-dir`: the first
/// existing of `<exe dir>/models` (the shipped layout — `camelid.exe` beside a
/// `models/` folder, the same precedent `auto_select_model` follows) then
/// `./models`, falling back to `./models` when neither exists yet.
fn default_models_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join("models");
            if beside.is_dir() {
                return beside;
            }
        }
    }
    PathBuf::from("models")
}

/// Resolve the configured (or defaulted) models directory to an absolute path
/// once at startup, so it stays stable regardless of the launch-context CWD.
/// Uses `std::path::absolute`, NOT `fs::canonicalize`, deliberately: it does
/// not require the directory to exist yet (a fresh install has no models/ until
/// the first download) and never produces the Windows `\\?\` verbatim prefix in
/// user-facing strings.
fn resolve_models_dir(configured: Option<PathBuf>) -> PathBuf {
    let dir = configured.unwrap_or_else(default_models_dir);
    if dir.is_absolute() {
        dir
    } else {
        std::path::absolute(&dir).unwrap_or(dir)
    }
}

/// Resolve a model path from an API request against the configured models
/// directory. Order (first hit wins; nothing is ever rewritten once found):
///
/// 1. An ABSOLUTE path is used exactly as given (unchanged behavior).
/// 2. A relative path keeps its exact historical meaning first: as given,
///    against the process CWD.
/// 3. Otherwise `<models_dir>/<path>` — the fallback that makes relative loads
///    work when the server's CWD is arbitrary (the desktop sidecar case).
/// 4. Otherwise, when the request already leads with the models dir's own
///    folder name (e.g. `models/foo.gguf` against a configured `...\models`),
///    the duplicated leading component is stripped: `<models_dir>/foo.gguf`.
///
/// When nothing exists the typed I/O error names every attempted location, so
/// the 400 tells the caller exactly where the server looked.
fn resolve_request_model_path(
    models_dir: &std::path::Path,
    requested: PathBuf,
) -> Result<PathBuf, BackendError> {
    if requested.is_absolute() || requested.exists() {
        return Ok(requested);
    }
    let mut attempted = vec![format!(
        "'{}' (relative to the server working directory)",
        requested.display()
    )];
    let joined = models_dir.join(&requested);
    if joined.exists() {
        return Ok(joined);
    }
    attempted.push(format!(
        "'{}' (under the configured models directory)",
        joined.display()
    ));
    if let (Some(first), Some(dir_name)) = (requested.components().next(), models_dir.file_name()) {
        if first.as_os_str() == dir_name {
            let stripped: PathBuf = requested.components().skip(1).collect();
            if !stripped.as_os_str().is_empty() {
                let candidate = models_dir.join(&stripped);
                if candidate.exists() {
                    return Ok(candidate);
                }
                attempted.push(format!(
                    "'{}' (under the configured models directory)",
                    candidate.display()
                ));
            }
        }
    }
    Err(BackendError::Io {
        path: requested,
        source: std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("model file not found; tried {}", attempted.join(" and ")),
        ),
    })
}

#[cfg(test)]
mod models_dir_resolution_tests {
    use super::{resolve_models_dir, resolve_request_model_path};
    use crate::BackendError;
    use std::path::PathBuf;

    /// A models dir whose final component is literally `models`, holding one
    /// GGUF — the shipped/desktop layout the resolver's fallbacks target.
    fn models_dir_with(filename: &str) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().expect("create tempdir");
        let models_dir = tmp.path().join("models");
        std::fs::create_dir(&models_dir).expect("create models dir");
        std::fs::write(models_dir.join(filename), b"gguf-stub").expect("write stub");
        (tmp, models_dir)
    }

    #[test]
    fn absolute_path_passes_through_even_when_missing() {
        let (_tmp, models_dir) = models_dir_with("present.gguf");
        let requested = if cfg!(windows) {
            PathBuf::from(r"C:\definitely\missing\model.gguf")
        } else {
            PathBuf::from("/definitely/missing/model.gguf")
        };
        let resolved = resolve_request_model_path(&models_dir, requested.clone())
            .expect("absolute paths are never rewritten");
        assert_eq!(resolved, requested);
    }

    #[test]
    fn relative_path_that_exists_in_cwd_stays_as_given() {
        // `cargo test` runs with the crate root as CWD, where Cargo.toml always
        // exists — the exact historical CWD-relative behavior must win even
        // though a models dir is configured.
        let (_tmp, models_dir) = models_dir_with("present.gguf");
        let resolved = resolve_request_model_path(&models_dir, PathBuf::from("Cargo.toml"))
            .expect("CWD-relative hit resolves");
        assert_eq!(resolved, PathBuf::from("Cargo.toml"));
    }

    #[test]
    fn relative_path_falls_back_to_the_configured_models_dir() {
        let name = "camelid-models-dir-resolution-test.gguf";
        let (_tmp, models_dir) = models_dir_with(name);
        let resolved = resolve_request_model_path(&models_dir, PathBuf::from(name))
            .expect("bare filename falls back to the models dir");
        assert_eq!(resolved, models_dir.join(name));
    }

    #[test]
    fn duplicated_models_component_is_stripped_against_the_models_dir() {
        // The live desktop repro: the client says "models/<file>" while the
        // configured dir already IS ".../models" — the duplicated leading
        // component is stripped rather than looking in ".../models/models/".
        let name = "camelid-models-dir-resolution-test.gguf";
        let (_tmp, models_dir) = models_dir_with(name);
        let resolved = resolve_request_model_path(&models_dir, PathBuf::from("models").join(name))
            .expect("models/-prefixed request resolves into the models dir");
        assert_eq!(resolved, models_dir.join(name));
    }

    #[test]
    fn missing_everywhere_error_names_every_attempted_location() {
        let (_tmp, models_dir) = models_dir_with("present.gguf");
        let requested = PathBuf::from("models").join("camelid-models-dir-missing.gguf");
        let err = resolve_request_model_path(&models_dir, requested.clone())
            .expect_err("nothing exists anywhere");
        assert!(matches!(err, BackendError::Io { .. }));
        let message = err.to_string();
        assert!(
            message.contains(&requested.display().to_string()),
            "error must name the CWD-relative attempt: {message}"
        );
        assert!(
            message.contains(&models_dir.join(&requested).display().to_string()),
            "error must name the models-dir attempt: {message}"
        );
        assert!(
            message.contains(
                &models_dir
                    .join("camelid-models-dir-missing.gguf")
                    .display()
                    .to_string()
            ),
            "error must name the stripped-component attempt: {message}"
        );
    }

    #[test]
    fn resolve_models_dir_is_absolute_for_default_and_relative_configs() {
        assert!(resolve_models_dir(None).is_absolute());
        assert!(resolve_models_dir(Some(PathBuf::from("models"))).is_absolute());
        let explicit = if cfg!(windows) {
            PathBuf::from(r"C:\somewhere\models")
        } else {
            PathBuf::from("/somewhere/models")
        };
        assert_eq!(resolve_models_dir(Some(explicit.clone())), explicit);
    }
}

pub fn router() -> Router {
    router_with_state(AppState::default())
}

#[cfg(test)]
pub(crate) fn router_with_workspace_cli_token_for_tests(token: &str) -> Router {
    router_with_state(AppState::default().with_workspace_cli_token(Some(token)))
}

pub fn router_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/api/capabilities", get(capabilities))
        .route("/api/runtime/gpu", get(gpu_runtime).post(set_gpu_runtime))
        .route("/api/telemetry/stream", get(telemetry_stream))
        .route("/execution-plan", get(execution_plan))
        .route("/api/execution-plan", get(execution_plan))
        .route("/api/models/load", post(load_model))
        .route("/api/models/inspect", post(inspect_model))
        .route("/api/models/unload", post(unload_model))
        .route("/api/models/current", get(current_model))
        .route(
            "/api/models/verify",
            get(model_verification_status).post(verify_current_model),
        )
        .route("/api/models/metadata", get(model_metadata))
        .route("/api/models/tokenizer", get(model_tokenizer))
        .route("/api/models/tokenizer/encode", post(tokenizer_encode))
        .route("/api/models/tokenizer/decode", post(tokenizer_decode))
        .route("/tokenize", post(llama_server_tokenize))
        .route("/detokenize", post(llama_server_detokenize))
        .route("/apply-template", post(llama_server_apply_template))
        .route("/models", get(llama_server_models))
        .route("/models/load", post(llama_server_models_load))
        .route(
            "/models/unload",
            post(unsupported_llama_server_models_unload),
        )
        .route(
            "/props",
            get(llama_server_props).post(unsupported_llama_server_props),
        )
        .route(
            "/slots",
            get(llama_server_slots).post(unsupported_llama_server_slots),
        )
        .route("/metrics", get(unsupported_llama_server_metrics))
        .route("/completion", post(llama_server_completion))
        .route("/infill", post(unsupported_llama_server_infill))
        .route("/embedding", post(unsupported_embeddings))
        .route("/embeddings", post(unsupported_embeddings))
        .route("/rerank", post(unsupported_reranking))
        .route("/reranking", post(unsupported_reranking))
        .route(
            "/api/generation/sessions",
            get(generation_sessions).post(create_generation_session),
        )
        .route("/api/generation/preflight", post(preflight_generation))
        .route(
            "/api/agent/workspace/models",
            get(workspace::compatible_models),
        )
        .route("/api/agent/workspace/threads", get(workspace::list_threads))
        .route(
            "/api/agent/workspace/threads/:id",
            get(workspace::get_thread).delete(workspace::delete_thread),
        )
        .route(
            "/api/agent/workspace/threads/:id/compact",
            post(workspace::compact_thread).delete(workspace::undo_thread_compaction),
        )
        .route("/api/agent/workspace/browse", get(workspace::browse))
        .route(
            "/api/agent/workspace/sessions",
            post(workspace::create_session),
        )
        .route(
            "/api/agent/workspace/sessions/:id",
            get(workspace::session_status).delete(workspace::cancel_session),
        )
        .route(
            "/api/agent/workspace/sessions/:id/events",
            get(workspace::session_events),
        )
        .route(
            "/api/agent/workspace/sessions/:id/messages",
            post(workspace::send_message),
        )
        .route(
            "/api/agent/workspace/sessions/:id/decisions",
            post(workspace::decide),
        )
        .route("/api/models/local", get(local_models))
        .route("/api/models/local/delete", post(delete_local_model))
        .route("/api/models/runnable-receipt", get(runnable_receipt))
        .route("/api/models/runnable-smoke", post(run_runnable_smoke))
        .route("/api/models/catalog", get(get_catalog))
        .route("/api/models/catalog/fit", post(check_catalog_fit))
        .route("/api/models/catalog/install", post(install_catalog_model))
        .route("/api/models/catalog/downloads", get(get_catalog_downloads))
        .route("/api/models/catalog/cancel", post(cancel_catalog_download))
        .route(
            "/api/models/catalog/ack",
            post(acknowledge_catalog_download),
        )
        .route("/v1/models", get(v1_models))
        .route("/v1/models/:model", get(v1_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(unsupported_embeddings))
        .route("/v1/responses", post(unsupported_responses))
        .route("/v1/messages", post(unsupported_messages))
        .route("/v1/rerank", post(unsupported_reranking))
        .route("/v1/reranking", post(unsupported_reranking))
        // Anything not matched above is served from the embedded web UI (the
        // chat surface and its static assets), with a client-side-route
        // fallback to the app shell. API routes are matched first.
        .fallback(crate::web_ui::handler)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn serve(
    addr: SocketAddr,
    configured_threads: Option<usize>,
    initial_model: Option<PathBuf>,
    open_ui: bool,
    default_enable_thinking: bool,
    models_dir: Option<PathBuf>,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let mut state = AppState::with_configured_threads(configured_threads)
        .with_default_enable_thinking(default_enable_thinking)
        .with_local_model_delete(addr.ip().is_loopback())
        .with_models_dir(models_dir)
        .with_serve_addr(addr);
    if let Some(model_path) = initial_model {
        if let Err(err) = load_model_from_path(&state, model_path, None).await {
            tracing::error!(error=%err, "failed to load startup model");
            eprintln!("\n  Could not load that model: {err}");
            eprintln!("  Camelid serves specific validated Q8_0 rows. To get one:");
            eprintln!("      camelid pull            # list supported models");
            eprintln!("      camelid pull <id>       # download one into ./models\n");
            return Err(std::io::Error::other(err.to_string()));
        }
    }
    let workspace_cli_credential = if addr.ip().is_loopback() {
        match crate::workspace_auth::WorkspaceCliCredential::issue(addr) {
            Ok(credential) => Some(credential),
            Err(error) => {
                tracing::warn!(%error, "Workspace CLI authentication is unavailable");
                eprintln!(
                    "  Workspace CLI unavailable: could not create its local credential ({error})"
                );
                None
            }
        }
    } else {
        None
    };
    state = state.with_workspace_cli_token(
        workspace_cli_credential
            .as_ref()
            .map(crate::workspace_auth::WorkspaceCliCredential::token),
    );
    tracing::info!(%addr, "camelid server listening");
    let url = format!("http://{addr}");

    // Warm the model-fit dimension cache in the background: header-only reads
    // (never weights), disk-cached across restarts, de-duplicated, and globally
    // rate-limited. This is why catalog fit badges can be *exact* without any
    // per-request fetching. Best-effort; opt out with CAMELID_NO_REMOTE_DIMS=1.
    crate::fit_dims::start_background(
        curated_catalog()
            .iter()
            .map(|c| (c.repo_id.to_string(), c.filename.to_string(), c.size_bytes))
            .collect(),
    );

    // Warm the generation path BEFORE telling the user we're ready. The GPU resident
    // engine (NVRTC kernel compile + multi-GB weight upload + first prefill) is built
    // lazily on the first generation â€” a one-time cost of several seconds. If it lands
    // on the user's first prompt the model looks "dog slow" (it's really the cold
    // build, on the GPU the whole time). So: start serving in the background, fire one
    // tiny self-request through the exact same code path to build the engine, BLOCK on
    // it, and only then print "ready". After this the first real request is warm
    // (~0.5s) instead of ~10s. The warm-up is best-effort â€” any failure just falls back
    // to the old lazy build and is not fatal.
    let warm_model_id = state.active_model_id.read().await.clone();
    let server = tokio::spawn(async move { axum::serve(listener, router_with_state(state)).await });
    if let Some(model_id) = warm_model_id {
        // Word the banner for the device that will actually serve â€” saying "GPU"
        // on a CPU-only run (e.g. CUDA_VISIBLE_DEVICES=-1) was misleading.
        let warming_msg = if crate::cuda::gpu_accel_enabled() {
            "ðŸª Warming up the GPU (building the resident engine, one-time)â€¦"
        } else {
            "ðŸª Warming up the model (building the engine, one-time)â€¦"
        };
        eprintln!("\n  {warming_msg}");
        warmup_generation_blocking(addr, model_id).await;
    }
    print_ready_banner(&url);
    if workspace_cli_credential.is_some() {
        eprintln!("  Workspace CLI: ready");
    }
    if open_ui {
        crate::web_ui::open_in_browser(&url);
    }
    server.await.map_err(std::io::Error::other)?
}

/// One-shot self-request to build/warm the generation engine, awaited so startup
/// blocks until the GPU resident engine is built (kernels compiled, weights uploaded,
/// first prefill run). Runs the blocking `std::net` round-trip on the blocking pool so
/// the spawned server task can answer it concurrently. Best-effort: any failure (or
/// the 180s safety timeout) just returns, leaving the old lazy-build behaviour.
async fn warmup_generation_blocking(addr: SocketAddr, model_id: String) {
    let _ = tokio::task::spawn_blocking(move || warmup_request(addr, &model_id)).await;
}

/// Send the warm-up chat request and read the full response (blocking). Returns once
/// the forward has completed so the resident engine is built before the caller
/// proceeds. Errors are swallowed by the caller â€” this only shaves the cold start.
fn warmup_request(addr: SocketAddr, model_id: &str) {
    use std::io::{Read, Write};
    // Give the just-spawned listener a moment to start accepting; retry briefly.
    let mut stream = None;
    for _ in 0..40 {
        match std::net::TcpStream::connect(addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    let Some(mut stream) = stream else {
        return;
    };
    // Don't let a stuck forward hang startup forever.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(180)));
    let body = serde_json::json!({
        "model": model_id,
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body,
    );
    if stream.write_all(request.as_bytes()).is_ok() {
        let mut sink = Vec::new();
        let _ = stream.read_to_end(&mut sink);
        tracing::info!("generation warm-up complete");
    }
}

/// Print a clear, human-facing pointer to the web UI once the server is up.
/// `tracing` output is for operators; this banner is for the person who just
/// ran `camelid serve` and wants to know where to click.
fn print_ready_banner(url: &str) {
    eprintln!("\n  ðŸª Camelid is ready");
    eprintln!("  Open the chat UI:  {url}");
    eprintln!("  OpenAI-style API:  {url}/v1/chat/completions\n");
}

/// Live inference telemetry stream (SSE). Subscribers receive only events
/// emitted by real inference work (see `crate::telemetry`); an idle server
/// sends nothing beyond the initial hello and keep-alive comments.
async fn telemetry_stream() -> Response {
    let mut rx = telemetry::hub().subscribe();
    let events = async_stream::stream! {
        let hello = serde_json::json!({
            "event": "hello",
            "schema": telemetry::TELEMETRY_SCHEMA,
            "engine": "camelid",
        });
        yield Ok::<Event, Infallible>(Event::default().event("telemetry").data(hello.to_string()));
        loop {
            match rx.recv().await {
                Ok(envelope) => match serde_json::to_string(&envelope) {
                    Ok(json) => yield Ok(Event::default().event("telemetry").data(json)),
                    Err(_) => continue,
                },
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    let notice = serde_json::json!({ "event": "lagged", "skipped": skipped });
                    yield Ok(Event::default().event("telemetry").data(notice.to_string()));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };
    Sse::new(events)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(10))
                .text("ping"),
        )
        .into_response()
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let active_id_lock = state.active_model_id.read().await;
    let loaded_models = state.loaded_models.read().await;
    let model = active_id_lock.as_ref().and_then(|id| loaded_models.get(id));
    let loaded_now = !loaded_models.is_empty();
    // Is the active model served by a gemma4 runtime?
    let gemma4_available = match active_id_lock.as_ref() {
        Some(id) => state.gemma4_runtimes.read().await.contains_key(id),
        None => false,
    };
    let model_family = model.map(|m| model_family(&m.gguf));
    // Specialized serve lanes are ready only when their active runtime instance exists.
    // Model load inserts the generic metadata before initializing these runtimes, and a
    // failed initialization leaves the model visible but intentionally not ready. Health
    // must agree with request routing, which also requires the runtime-map entry.
    let runnable_serve_ready = match active_id_lock.as_ref() {
        Some(id) => state.runnable_runtimes.read().await.contains_key(id),
        None => false,
    };
    let dg_serve_ready = match active_id_lock.as_ref() {
        Some(id) => state.dg_runtimes.read().await.contains_key(id),
        None => false,
    };
    let backend = health_backend(
        gemma4_available,
        runnable_serve_ready,
        dg_serve_ready,
        model.is_some(),
    );
    let generation_ready = gemma4_available
        || runnable_serve_ready
        || dg_serve_ready
        || model.is_some_and(loaded_model_generation_ready);
    let execution_plans = state.execution_plans.read().await;
    let execution_plan = active_id_lock
        .as_ref()
        .and_then(|id| execution_plans.get(id))
        .cloned();
    Json(HealthResponse {
        ok: true,
        engine: "camelid",
        loaded_now,
        generation_ready,
        active_model_id: active_id_lock.clone(),
        q8_runtime: q8_runtime_health(),
        execution_plan,
        backend,
        model_family,
        gemma4_available,
        engine_queue_depth: state.engine.depth(),
    })
}

fn health_backend(
    gemma4_available: bool,
    runnable_serve_ready: bool,
    dg_serve_ready: bool,
    model_loaded: bool,
) -> &'static str {
    if gemma4_available {
        "gemma4-runtime"
    } else if runnable_serve_ready {
        "runnable-runtime"
    } else if dg_serve_ready {
        "diffusion-gemma-runtime"
    } else if model_loaded {
        "llama"
    } else {
        "none"
    }
}

async fn llama_server_models(
    State(state): State<AppState>,
    Query(query): Query<LlamaServerReadOnlyQuery>,
) -> Response {
    if let Some(response) =
        unsupported_llama_server_query_params("/models", &query.unsupported_fields)
    {
        return response;
    }

    let loaded = state.loaded_models.read().await;
    let data = loaded
        .values()
        .map(|model| LlamaServerModelListItem {
            id: model.id.clone(),
            path: None,
            status: LlamaServerModelStatus {
                value: "loaded",
                args: Vec::new(),
            },
            architecture: LlamaServerModelArchitecture {
                input_modalities: vec!["text"],
                output_modalities: vec!["text"],
            },
            camelid: LlamaServerModelCamelid {
                generation_ready: loaded_model_generation_ready(model),
                model_path_redacted: true,
            },
        })
        .collect();

    Json(LlamaServerModelListResponse {
        data,
        camelid: LlamaServerModelListCamelid {
            compatibility: "partial_llama_server_models_read_only",
            scope: "loaded_models_only",
            unsupported: vec![
                "router_model_cache_listing",
                "models_reload",
                "models_autoload",
                "models_unload",
                "multimodal_architecture_metadata",
            ],
        },
    })
    .into_response()
}

async fn llama_server_models_load(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerModelsLoadRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_json",
                format!("invalid /models/load JSON request: {err}"),
                Some("body"),
            )
        }
    };

    if !req.unsupported_fields.is_empty() {
        let mut fields: Vec<_> = req.unsupported_fields.keys().cloned().collect();
        fields.sort();
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "unsupported /models/load field(s): {}. Camelid supports only a narrow local load alias with model/path plus optional id; router-mode cache, reload, autoload, and model-management semantics remain unsupported.",
                fields.join(", ")
            ),
            Some("body"),
        );
    }

    if req.model.is_some() && req.path.is_some() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "ambiguous_model_path",
            "POST /models/load accepts either `model` or `path`, not both; send one local GGUF path for the narrow load alias.".to_string(),
            Some("model"),
        );
    }

    let Some(path) = req.model.or(req.path) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "missing_model_path",
            "POST /models/load requires a local GGUF path in `model` or `path`; router cache names and autoload semantics are not supported.".to_string(),
            Some("model"),
        );
    };

    match load_model_from_path(&state, path, req.id).await {
        Ok(loaded) => {
            let item = LlamaServerModelListItem {
                id: loaded.id.clone(),
                path: None,
                status: LlamaServerModelStatus {
                    value: "loaded",
                    args: Vec::new(),
                },
                architecture: LlamaServerModelArchitecture {
                    input_modalities: vec!["text"],
                    output_modalities: vec!["text"],
                },
                camelid: LlamaServerModelCamelid {
                    generation_ready: loaded_model_generation_ready(&loaded),
                    model_path_redacted: true,
                },
            };
            Json(LlamaServerModelLoadResponse {
                data: item,
                camelid: LlamaServerModelLoadCamelid {
                    compatibility: "partial_llama_server_models_load_local_path",
                    scope: "single_local_model_load_alias",
                    model_path_redacted: true,
                    unsupported: vec![
                        "router_model_cache_listing",
                        "models_reload",
                        "models_autoload",
                        "models_unload",
                        "multimodal_architecture_metadata",
                        "full_llama_server_model_management",
                    ],
                },
            })
            .into_response()
        }
        Err(err) => llama_server_models_load_error(err),
    }
}

fn llama_server_models_load_error(err: BackendError) -> Response {
    let code = backend_error_code(&err);
    let message = match err {
        BackendError::Io { source, .. } => {
            // Redaction contract for this llama-server alias: never echo local
            // path detail. The resolver's io source now enumerates every
            // attempted location (useful on the native endpoint), so only the
            // pathless error kind may pass through here.
            format!(
                "failed to load requested local GGUF path: {}",
                source.kind()
            )
        }
        other => other.to_string(),
    };
    api_error(StatusCode::BAD_REQUEST, code, message, Some("model"))
}

async fn unsupported_llama_server_models_unload() -> Response {
    unsupported_route(
        "unsupported_llama_server_models_unload",
        "POST /models/unload is not supported yet; Camelid keeps native llama-server router-mode model unloading separate from the stable /api/models/unload path until router semantics and support-contract behavior are implemented and tested",
        Some("model"),
    )
}

fn unsupported_llama_server_query_params(
    route: &'static str,
    fields: &HashMap<String, String>,
) -> Option<Response> {
    if fields.is_empty() {
        return None;
    }

    let mut fields = fields.keys().map(String::as_str).collect::<Vec<_>>();
    fields.sort_unstable();
    Some(api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        format!(
            "{route} query parameter(s) are not supported yet: {}; Camelid exposes this route as active-model read-only discovery, not llama-server router-mode autoload/reload/model selection",
            fields.join(", ")
        ),
        Some("query"),
    ))
}

async fn llama_server_props(
    State(state): State<AppState>,
    Query(query): Query<LlamaServerReadOnlyQuery>,
) -> Response {
    if let Some(response) =
        unsupported_llama_server_query_params("/props", &query.unsupported_fields)
    {
        return response;
    }

    let active_id_lock = state.active_model_id.read().await;
    let loaded_models = state.loaded_models.read().await;
    let model = active_id_lock.as_ref().and_then(|id| loaded_models.get(id));
    let generation_ready = model.is_some_and(loaded_model_generation_ready);
    let n_ctx = model
        .and_then(|model| model.llama_config.as_ref())
        .map(|config| config.context_length)
        .unwrap_or(0);
    let chat_template = model
        .and_then(|model| model.tokenizer_runtime.as_ref())
        .and_then(|tokenizer| tokenizer.chat_template.clone());
    let chat_template_caps = llama_server_chat_template_caps(model);
    let model_id = model.map(|model| model.id.clone());

    Json(LlamaServerPropsResponse {
        default_generation_settings: LlamaServerDefaultGenerationSettings {
            id: 0,
            id_task: -1,
            n_ctx,
            speculative: false,
            is_processing: false,
            params: LlamaServerDefaultGenerationParams {
                n_predict: -1,
                seed: u32::MAX,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                stop: Vec::new(),
                max_tokens: -1,
                ignore_eos: false,
                stream: true,
                n_probs: 0,
                samplers: vec!["greedy"],
            },
            prompt: "",
            next_token: LlamaServerNextTokenProps {
                has_next_token: generation_ready,
                has_new_line: false,
                n_remain: -1,
                n_decoded: 0,
                stopping_word: "",
            },
        },
        total_slots: 1,
        model_path: None,
        model_id,
        chat_template,
        chat_template_caps,
        modalities: LlamaServerModalities { vision: false },
        build_info: "camelid",
        is_sleeping: false,
        camelid: LlamaServerPropsCamelid {
            compatibility: "partial_llama_server_props_read_only",
            generation_ready,
            model_path_redacted: true,
            unsupported: vec![
                "post_props",
                "post_slots",
                "slot_cache_actions",
                "native_completion",
                "native_completion_streaming",
                "full_native_completion_parity",
                "embeddings",
                "reranking",
                "multimodal",
            ],
        },
    })
    .into_response()
}

fn llama_server_chat_template_caps(model: Option<&LoadedModel>) -> serde_json::Value {
    let template = model.and_then(|model| match &model.tokenizer {
        TokenizerLoadState::Available(summary) => summary.chat_template.as_ref(),
        TokenizerLoadState::Unavailable { .. } => None,
    });

    match template {
        Some(template) => serde_json::json!({
            "available": true,
            "requires_loaded_model": true,
            "source": template.source,
            "detected_format": template.detected_format,
            "length": template.length,
            "supported_operations": ["render_prompt"],
            "unsupported": [
                "arbitrary_template_kwargs",
                "tool_call_templates",
                "multimodal_templates",
                "full_llama_server_template_parity"
            ],
        }),
        None => serde_json::json!({
            "available": false,
            "requires_loaded_model": true,
            "source": null,
            "detected_format": null,
            "length": null,
            "supported_operations": [],
            "unsupported": [
                "no_loaded_supported_chat_template",
                "arbitrary_template_kwargs",
                "tool_call_templates",
                "multimodal_templates",
                "full_llama_server_template_parity"
            ],
        }),
    }
}

async fn unsupported_llama_server_props() -> Response {
    unsupported_route(
        "unsupported_llama_server_props",
        "POST /props is not supported yet; Camelid exposes /props as a read-only, privacy-safe llama-server compatibility discovery route",
        Some("props"),
    )
}

async fn llama_server_slots(
    State(state): State<AppState>,
    Query(query): Query<LlamaServerSlotsQuery>,
) -> Response {
    if let Some(response) =
        unsupported_llama_server_query_params("/slots", &query.unsupported_fields)
    {
        return response;
    }

    let active_id_lock = state.active_model_id.read().await;
    let loaded_models = state.loaded_models.read().await;
    let model = active_id_lock.as_ref().and_then(|id| loaded_models.get(id));
    let generation_ready = model.is_some_and(loaded_model_generation_ready);

    if query.fail_on_no_slot.as_deref() == Some("1") && !generation_ready {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_available_slot",
            "no generation-ready Camelid slot is available".to_string(),
            Some("fail_on_no_slot"),
        );
    }

    let n_ctx = model
        .and_then(|model| model.llama_config.as_ref())
        .map(|config| config.context_length)
        .unwrap_or(0);
    let status = if generation_ready {
        "idle_generation_ready"
    } else {
        "unavailable"
    };

    (
        StatusCode::OK,
        Json(vec![LlamaServerSlotResponse {
            id: 0,
            id_task: -1,
            n_ctx,
            speculative: false,
            is_processing: false,
            params: LlamaServerDefaultGenerationParams {
                n_predict: -1,
                seed: u32::MAX,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                stop: Vec::new(),
                max_tokens: -1,
                ignore_eos: false,
                stream: true,
                n_probs: 0,
                samplers: vec!["greedy"],
            },
            prompt: "",
            next_token: LlamaServerNextTokenProps {
                has_next_token: generation_ready,
                has_new_line: false,
                n_remain: -1,
                n_decoded: 0,
                stopping_word: "",
            },
            camelid: LlamaServerSlotCamelid {
                compatibility: "partial_llama_server_slots_read_only",
                generation_ready,
                status,
                engine_queue_depth: state.engine.depth(),
                unsupported: vec![
                    "post_slots",
                    "slot_cache_save_restore_erase",
                    "prompt_cache_metadata",
                    "cancellation_metadata",
                    "continuous_batching_metrics",
                ],
            },
        }]),
    )
        .into_response()
}

async fn unsupported_llama_server_slots() -> Response {
    unsupported_route(
        "unsupported_llama_server_slots",
        "POST /slots and llama-server slot cache actions are not supported yet; Camelid exposes GET /slots as a read-only compatibility snapshot",
        Some("slots"),
    )
}

async fn unsupported_llama_server_metrics() -> Response {
    unsupported_route(
        "unsupported_llama_server_metrics",
        "GET /metrics is not supported yet; Camelid has no llama-server metrics compatibility contract, prompt-cache metrics, or continuous batching telemetry surface for this route",
        Some("metrics"),
    )
}

async fn llama_server_completion(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    if req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "/completion stream=true is not supported yet; use /v1/completions with stream=true for Camelid's stable SSE stream path".to_string(),
            Some("stream"),
        );
    }

    let max_tokens = match llama_server_completion_max_tokens(req.n_predict, req.max_tokens) {
        Ok(max_tokens) => max_tokens,
        Err(response) => return *response,
    };
    let (prompt, camelid_prompt_token_ids) = match llama_server_completion_prompt(req.prompt) {
        Ok(prompt) => prompt,
        Err(response) => return *response,
    };
    let req = GenerationSessionRequest {
        model: req.model,
        prompt,
        messages: None,
        max_tokens,
        stream: Some(false),
        temperature: req.temp.or(req.temperature),
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        logit_bias: req.logit_bias,
        stop: req.stop,
        n: None,
        best_of: None,
        completion_logprobs: None,
        chat_logprobs: None,
        top_logprobs: None,
        camelid_logit_token_ids: None,
        camelid_prompt_token_ids,
        camelid_dense_diagnostics: None,
        camelid_dense_diagnostic_generated_index: None,
        camelid_context_budget_tokens: None,
        camelid_enable_thinking: None,
        tools: None,
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: Some(DEFAULT_PUBLIC_CHAT_MAX_TOKENS),
        constraint: None,
    };
    // Prep runs OUTSIDE any serialization (D3); the decode runs on the engine
    // worker, and CancelOnDrop stops it within one step on client disconnect.
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let _cancel_on_drop = CancelOnDrop(prepared.cancel.token.clone());
    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    match generate_decoded_tokens_blocking(&state, prepared).await {
        Ok(generated) => {
            let finish_reason = generated.finish_reason;
            (
                StatusCode::OK,
                Json(LlamaServerCompletionResponse {
                    content: generated.text,
                    model: model_id,
                    stop: finish_reason != "length",
                    stopped_limit: finish_reason == "length",
                    tokens_predicted: generated.completion_tokens,
                    tokens_evaluated: prompt_token_count,
                    camelid: LlamaServerCompletionCamelid {
                        compatibility: "partial_llama_server_completion_non_streaming",
                        finish_reason,
                        unsupported: vec![
                            "streaming_completion_shape",
                            "slot_selection",
                            "cache_prompt_controls",
                            "llama_server_timings_shape",
                            "rich_token_probabilities",
                        ],
                    },
                }),
            )
                .into_response()
        }
        Err(response) => *response,
    }
}

fn llama_server_completion_max_tokens(
    n_predict: Option<i32>,
    max_tokens: Option<u32>,
) -> std::result::Result<Option<u32>, Box<Response>> {
    let n_predict_max_tokens =
        match n_predict {
            Some(-1) | None => None,
            Some(0) => return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_n_predict",
                "n_predict must be greater than zero, or -1 to use Camelid's bounded default cap"
                    .to_string(),
                Some("n_predict"),
            ))),
            Some(value) if value < -1 => {
                return Err(Box::new(api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_n_predict",
                    "n_predict values below -1 are not supported".to_string(),
                    Some("n_predict"),
                )))
            }
            Some(value) => Some(value as u32),
        };

    if let (Some(from_n_predict), Some(from_max_tokens)) = (n_predict_max_tokens, max_tokens) {
        if from_n_predict != from_max_tokens {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "ambiguous_generation_length",
                "send only one generation length field, or make n_predict and max_tokens match"
                    .to_string(),
                Some("n_predict"),
            )));
        }
    }

    Ok(n_predict_max_tokens.or(max_tokens))
}

fn llama_server_completion_prompt(
    prompt: Option<LlamaServerCompletionPrompt>,
) -> std::result::Result<LlamaServerCompletionPromptParts, Box<Response>> {
    match prompt {
        Some(LlamaServerCompletionPrompt::Text(prompt)) => Ok((Some(prompt), None)),
        Some(LlamaServerCompletionPrompt::TokenIds(token_ids)) if token_ids.is_empty() => {
            Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "empty_prompt_tokens",
                "/completion prompt token-id arrays must contain at least one token".to_string(),
                Some("prompt"),
            )))
        }
        Some(LlamaServerCompletionPrompt::TokenIds(token_ids)) => Ok((None, Some(token_ids))),
        None => Ok((None, None)),
    }
}

async fn unsupported_llama_server_infill() -> Response {
    unsupported_route(
        "unsupported_llama_server_infill",
        "/infill is not supported yet; Camelid has no FIM runtime or compatibility contract for this route",
        Some("input"),
    )
}

async fn unsupported_embeddings() -> Response {
    unsupported_route(
        "unsupported_embeddings",
        "embeddings are not supported yet; Camelid has no embeddings runtime or compatibility contract for this route",
        Some("input"),
    )
}

async fn unsupported_reranking() -> Response {
    unsupported_route(
        "unsupported_reranking",
        "reranking is not supported yet; Camelid has no reranking runtime or compatibility contract for this route",
        Some("input"),
    )
}

async fn unsupported_responses() -> Response {
    unsupported_route(
        "unsupported_responses",
        "OpenAI Responses compatibility is not supported yet; Camelid keeps generation on /v1/completions and /v1/chat/completions until request conversion, streaming, tool, and cancellation semantics are implemented and tested",
        Some("input"),
    )
}

async fn unsupported_messages() -> Response {
    unsupported_route(
        "unsupported_messages",
        "Anthropic Messages compatibility is not supported yet; Camelid keeps generation on /v1/completions and /v1/chat/completions until request conversion, streaming, tool, and cancellation semantics are implemented and tested",
        Some("input"),
    )
}

fn loaded_model_generation_ready(model: &LoadedModel) -> bool {
    let Some(binding) = model.llama_tensors.as_ref() else {
        return false;
    };
    model.llama_config.is_some()
        && matches!(model.tokenizer, TokenizerLoadState::Available(_))
        && guard_cpu_weight_materialization_budget(binding).is_ok()
}

/// Runtime GPU (CUDA) state for the UI toggle. `available` is whether a usable
/// CUDA device is present (the UI shows the toggle only when true); `enabled` is
/// the current switch position. On non-CUDA builds/hosts `available` is false.
#[derive(Serialize)]
struct GpuRuntimeState {
    available: bool,
    enabled: bool,
    device: Option<String>,
    backend: &'static str,
    /// Number of Q8_0 matmuls run on the GPU so far this process (0 if the GPU
    /// path has never executed). Lets the UI/tests confirm the toggle is live.
    run_count: u64,
}

#[derive(Deserialize)]
struct GpuRuntimeRequest {
    enabled: bool,
}

fn current_gpu_runtime() -> GpuRuntimeState {
    GpuRuntimeState {
        available: crate::cuda::is_available(),
        // Report the MASTER GPU-acceleration switch (resident decode), which is what
        // actually runs the model on the GPU â€” not the legacy hybrid-matmul flag, which
        // defaults off and made the UI read "GPU off" while the GPU did all the work.
        enabled: crate::cuda::gpu_accel_enabled(),
        device: crate::cuda::device_name(),
        backend: "cuda",
        run_count: crate::cuda::cuda_q8_run_count(),
    }
}

/// GET `/api/runtime/gpu` â€” current GPU-acceleration availability + on/off state.
async fn gpu_runtime() -> Json<GpuRuntimeState> {
    Json(current_gpu_runtime())
}

/// POST `/api/runtime/gpu` `{ "enabled": bool }` â€” flip GPU acceleration at
/// runtime (no restart). A no-op effect when no CUDA device is present, since
/// the inference dispatch falls back to the CPU reference either way.
async fn set_gpu_runtime(Json(req): Json<GpuRuntimeRequest>) -> Json<GpuRuntimeState> {
    // Flip the master GPU-acceleration switch (the resident decode engine) AND the
    // legacy hybrid Q8 matmul switch together, so "GPU acceleration" is one coherent
    // control: on => resident decode runs on the GPU; off => pure CPU reference.
    crate::cuda::set_gpu_accel_enabled(req.enabled);
    crate::cuda::set_runtime_enabled(req.enabled);
    Json(current_gpu_runtime())
}

async fn capabilities(State(state): State<AppState>) -> Json<CapabilitiesResponse> {
    let active_id_lock = state.active_model_id.read().await;
    let execution_plans = state.execution_plans.read().await;
    let execution_plan = active_id_lock
        .as_ref()
        .and_then(|id| execution_plans.get(id))
        .cloned();
    Json(capabilities_response_with_plan(execution_plan))
}

async fn execution_plan(State(state): State<AppState>) -> Json<Option<ExecutionPlan>> {
    let active_id_lock = state.active_model_id.read().await;
    let execution_plans = state.execution_plans.read().await;
    let execution_plan = active_id_lock
        .as_ref()
        .and_then(|id| execution_plans.get(id))
        .cloned();
    Json(execution_plan)
}

#[cfg(test)]
fn capabilities_response() -> CapabilitiesResponse {
    capabilities_response_with_plan(None)
}

fn capabilities_response_with_plan(execution_plan: Option<ExecutionPlan>) -> CapabilitiesResponse {
    CapabilitiesResponse {
        engine: "camelid",
        gguf_metadata: true,
        tensor_loading: true,
        tokenization: true,
        inference: true,
        streaming: true,
        model_downloads: true,
        hf_catalog_install: true,
        execution_plan,
        support_contract: SupportContract {
            current_gate: "Current exact-row support: TinyLlama Q8_0 current gate; Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs; Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence); and Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs where row-specific PASS artifacts exist. Mistral 7B Instruct v0.3 Q8_0 is supported_exact_row_smoke: checked tokenizer/template, parity (including GPU-vs-CPU greedy continuations on the exact row), bounded 512/1024/2048/4096/8192 context artifacts, and a support-promotion API/WebUI smoke bundle. Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token backend MoE runtime evidence only; later 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by Gate 9A 50-token divergence and a longer-continuation hang, so broad/API/WebUI/frontend readiness remains unsupported. The dense Qwen3 Q8_0 ChatML rows (0.6B/1.7B/4B/8B Instruct, thinking disabled) are supported_exact_row_smoke: qwen2 BPE pre-tokenizer + ChatML renderer, per-head QK-norm + NEOX RoPE, and token+text parity vs llama.cpp at 1/5/50 on macOS/Ubuntu and on Windows x86_64 CPU (cpu_reference + the x86_q8 AVX2 runtime-repack path, bit-identical), and additionally on Windows CUDA: the 0.6B/1.7B/4B rows fully VRAM-resident and the 8B row via the VRAM+host-RAM offload split (RTX 3060 Laptop 6 GB, driver 576.83, CUDA 12.9; GPU decode+single-shot prefill token+text identical to cpu_reference/llama.cpp at 1/5/50); 1.7B additionally has GPU-resident decode+prefill and a 15,373-token single-shot prefill lane on macOS, and thinking-mode is opt-in (leading-trace parity only). The 4B row additionally carries checked bounded-context packs 512/1024/2048/4096/8192, the 1.7B row 512/1024/2048/4096, and the 0.6B row 512/2048/4096/8192 (fully-GPU-resident raw-decode greedy parity vs llama.cpp acd79d603 at 50 tokens; the 1.7B 8192 and 0.6B 1024 buckets are held as documented benign near-ties). These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied.",
            support_policy: "A model, tokenizer, quantization, API feature, or context length is supported only after tests, docs, and real-model evidence exist for that lane.",
            unsupported_policy: "Unsupported combinations should return typed errors instead of silently falling back to best-effort behavior.",
        },
        supported_quantization: vec![
            SupportItem {
                id: "F32",
                status: "supported",
                notes: "reference dense tensor path",
            },
            SupportItem {
                id: "F16",
                status: "supported",
                notes: "decoded into the reference CPU tensor path",
            },
            SupportItem {
                id: "BF16",
                status: "supported",
                notes: "decoded into the reference CPU tensor path",
            },
            SupportItem {
                id: "Q8_0",
                status: "supported_current_gate",
                notes: "TinyLlama remains the current support gate; exact Llama 3.2 1B Instruct Q8_0 now has checked bounded 512/1024/2048/4096/8192-context packs; exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence); and exact Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048-context packs where row-specific PASS artifacts exist. These are exact bounded-pack lanes only; no model-native/larger-context beyond the checked packs, arbitrary-template, production-throughput, portability, neighboring-row, or broad-family support is implied.",
            },
            SupportItem {
                id: "Q4_K_M/Q5_K_M/Q3_K_M",
                status: "supported_named_exact_rows_only",
                notes: "K-quants are certified per exact row only: qwen3_4b_q4_k_m, llama_3_2_3b_instruct_q4_k_m, llama_3_2_3b_instruct_q5_k_m, llama_3_2_1b_instruct_q4_k_m, ornith_1_0_9b_q4_k_m, and ornith_1_0_9b_q3_k_m — each with a row-specific parity bundle cited on its model_compatibility row. Neighboring K-quant files remain unsupported until separately certified; no family- or quant-wide claim is implied.",
            },
            SupportItem {
                id: "Q4_0 (QAT rows)",
                status: "supported_named_exact_rows_only",
                notes: "gemma4_26b_a4b_it_q4_0 (Q4_0 experts + Q6_K tied head) is supported_exact_row_smoke scoped to the two-Mac distributed serve lane; the Gemma 4 E4B QAT row runs the Metal GPU-resident path token-identical to CPU. Engine Q4_0 dequant plus parity-gated wire GEMV kernels (CUDA/Metal) exist as engine facts, not support; no LLaMA/SPM Q4_0 row is certified (see planned_quantization).",
            },
            SupportItem {
                id: "TQ2_0",
                status: "supported_named_exact_rows_only",
                notes: "ternary_bonsai_4b_tq2_0 only: single-node CPU completion smoke (receipt qa/ternary/tq2_0-bonsai-parity-receipt.json). Other TQ2_0 files load to the unverified experimental lane only; no ternary-wide claim.",
            },
            SupportItem {
                id: "IQ4_XS",
                status: "supported_named_exact_rows_only",
                notes: "i-quants are certified per exact row only: llama3_2_1b_instruct_iq4_xs (IQ4_XS 4.25bpw linears + K-quant tied embed/lm_head) is supported_exact_row_smoke scoped to GPU-resident/CPU raw-completion parity smoke, with its receipt cited on the model_compatibility row (qa/iquant/iq4xs-llama3.2-1b-parity-receipt.json). Other IQ4_XS files load to the unverified experimental lane only (runnable-lane admission covers the format; no parity claim); other i-quant formats (IQ3_XXS, IQ2_*, IQ1_*) are unimplemented and fail closed. No i-quant-wide claim.",
            },
        ],
        planned_quantization: vec![
            SupportItem {
                id: "Q4_0/Q5_0",
                status: "planned_beyond_named_certified_rows",
                notes: "legacy smaller-quant lane for LLaMA/SPM rows: CPU dequant and parity-gated Q4_0 wire GEMV kernels exist as engine facts, and the Gemma 4 QAT Q4_0 rows are certified separately, but no LLaMA/SPM Q4_0/Q5_0 exact row is certified — that per-row certification remains planned",
            },
            SupportItem {
                id: "Q4_K_M/Q5_K_M",
                status: "planned_beyond_named_certified_rows",
                notes: "additional K-quant rows beyond the named certified exact rows (see supported_quantization and the K-quant model_compatibility rows) remain planned; support never spreads to neighboring files",
            },
            SupportItem {
                id: "NVFP4",
                status: "planned_beyond_named_certified_rows",
                notes: "Engine facts shipped for the gemma-4-E4B-it pilot only, on Windows and macOS (NVFP4 = E2M1 4-bit weights with UE4M3 sub-block scales, type id 40, 64-element/36-byte superblock). Loads and decodes bit-exact on the CPU wire lane, validated on x86 and on Apple Silicon/ARM (GABBRO Gate G-M1), plus a Windows CUDA-resident nvfp4_gemv dp4a kernel (46/46 bit-identical to the CPU oracle); on macOS the Metal GPU resident lane also runs NVFP4 as of GABBRO M3-followup (opt-in via gemma4-generate-gpu; self-parity-proven forward + T5 sentinel guard; end-to-end run PROVEN on the byte-exact real artifact, isolated decode 12.12 tok/s (1.45x the Q8_0 parent, fastest of NVFP4/Q8_0/Q4_0)), alongside the bit-exact CPU wire lane; sidecar-bearing and NaN-sentinel files fail closed. Quality vs the Q8_0 parent at matched 4.5 bpw: the frozen Gate G3 recorded NO-GO (NVFP4-mm 88.5% vs Q4K-mm 92.6% top-1) and STANDS as history; a disclosed current-engine teacher-forced re-eval is a comparator-sensitive NEAR-TIE (90.5% vs 91.9%, gap 1.4pp within the pre-registered 2.0pp GO tolerance) -- still a space/speed quant, NOT a quality win beyond that tolerance. On an RTX 3060 Laptop the CUDA decode is 1.03x faster than Q8_0 (26.51 vs 25.80 tok/s) and 2.08 GB lighter in VRAM -- decode-only, measured on this box, not a general claim. Runnable-lane admission covers the format with no parity claim; as of D-B6 (2026-07-17) BF16 is a covered exact-decode type, so the produced gemma4-E4B pilot file admits fully (its one BF16 tensor per_layer_model_proj was the prior blocker); it executes via gemma4_runtime (CPU wire + CUDA-resident lanes), not the generic runnable serve bridge; the gemma4-E4B NVFP4 pilot is now a supported_exact_row_smoke row (gemma4_e4b_it_nvfp4, scope exact_row_gpu_resident_raw_decode_parity_smoke_only), oracle-parity-anchored (G-M1 bit-exact CPU + Metal self-parity + BASALT Leg B + fresh macOS spot-check); see qa/evidence-bundles/gabbro/support/. Phase 5 Blackwell tensor-core path is BLOCKED-HW (no sm_120 silicon).",
            },
        ],
        supported_model_families: vec![
            SupportItem {
                id: "llama_spm_decoder",
                status: "supported_current_gate",
                notes: "LLaMA-style decoder path validated on TinyLlama Q8_0 gate",
            },
            SupportItem {
                id: "mistral_instruct_exact_7b_v0_3_q8_0",
                status: "supported_exact_row_smoke_lane",
                notes: "exact Mistral-7B-Instruct-v0.3 Q8_0 has row-specific smoke support: tokenizer/template, deterministic and broader 50-token parity, checked bounded 512/1024/2048/4096/8192-context packs, GPU-vs-CPU greedy parity on the exact row, and a support-promotion API/WebUI smoke bundle. Exact row only; broader Mistral-family, other quants, model-native context, and full support are not implied.",
            },
            SupportItem {
                id: "llama_bpe_decoder_exact_1b_3b_8b_q8_0",
                status: "supported_exact_row_smoke_lanes",
                notes: "exact Llama 3.2 1B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048/4096/8192-context packs; exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke standing on the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder for the current canonical GGUF (prior-upload Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 retained as historical); exact Llama 3 8B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048-context packs, including the published source/runtime-head 8B 1024/2048 PASS bundle at 8e26be0a73c0. Broader 50-token, compact chat-template-shapes, and retained-block lazy-Q8 hot-path evidence remain exact-row bounded pack/measurement evidence only, and broad/full support still needs separate proof.",
            },
            SupportItem {
                id: "qwen3_chatml_exact_0_6b_1_7b_4b_8b_q8_0",
                status: "supported_exact_row_smoke_lanes",
                notes: "exact dense Qwen3 Q8_0 ChatML rows (0.6B/1.7B/4B/8B Instruct, thinking DISABLED) have row-specific smoke support: qwen2 BPE pre-tokenizer + hardcoded ChatML renderer, per-head QK-norm + NEOX (split-half) RoPE, and token-AND-text-identical greedy parity vs llama.cpp at 1/5/50 tokens on macOS/Ubuntu and on Windows x86_64 CPU (both the cpu_reference scalar path and the x86_q8 AVX2 runtime-repack path, bit-identical). 1.7B additionally runs the GPU-resident decode+prefill path and a 15,373-token single-shot prefill lane on macOS, with opt-in thinking-mode leading-trace parity. Exact rows only; other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), thinking-mode token-parity, model-native/larger context beyond the validated envelope, and broad Qwen-family support are not implied.",
            },
        ],
        planned_model_families: vec![
            SupportItem {
                id: "larger_llama_instruct",
                status: "planned",
                notes: "broader LLaMA-family instruct support after row-specific parity, API, WebUI, memory/perf, and portability evidence",
            },
            SupportItem {
                id: "mixtral_moe",
                status: "active_validation_partial_runtime",
                notes: "public readiness: Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token exact-row MoE runtime evidence only. Top-k expert routing runs with lazy/file-backed rank-3 Q8 experts, but later Gate 9A 50-token evidence diverged at generated token index 9 and a longer-continuation backend call hung, so API/WebUI/frontend readiness and broad Mixtral support remain unsupported.",
            },
            SupportItem {
                id: "command_r",
                status: "experimental_scaffolding",
                notes: "Command-R tensor, normalization, tokenizer, and logit-scale scaffolding exists, but the runnable parser/admission path still rejects command-r and no exact-row load, generation, parity, or frontend evidence exists.",
            },
            SupportItem {
                id: "qwen25",
                status: "planned_exact_row_candidate",
                notes: "public readiness: planned first Qwen row only for Qwen2.5-7B-Instruct-Q8_0.gguf; not supported yet. Architecture mapping, tokenizer/template references, bounded load, prompt-token parity, API/WebUI, RSS, and bundle evidence are missing",
            },
            SupportItem {
                id: "gemma2",
                status: "planned_exact_row_candidate",
                notes: "public readiness: planned first Gemma row only for gemma-2-9b-it-Q8_0.gguf; not supported yet. Gemma2 architecture, tokenizer/control-token behavior, template formatting, bounded load, parity, API/WebUI, RSS, and bundle evidence are missing",
            },
            SupportItem {
                id: "phi_falcon_mamba_others",
                status: "future",
                notes: "tracked as future lanes only, not implied support",
            },
        ],
        model_compatibility: vec![
            // Ornith-1.0-9B CUDA-resident quant rows (constrained-VRAM lane).
            // ORDER CONTRACT: these quant-suffixed rows MUST stay ahead of the
            // bare-name "Ornith 1.0 9B" Q8_0 row below. The frontend exact-row
            // matcher (findExactCompatibilityRowByIdentity) returns the FIRST
            // ledger row whose normalized id matches any identity of the loaded
            // model, and every Ornith quant emits the bare general.name identity
            // — only the filename-derived name+quant identity distinguishes the
            // files. The bare-name row itself must keep its id verbatim: the
            // chat `--agent` gate (`active_tool_capable`) matches the
            // server-assigned id (general.name) by exact string.
            ModelCompatibilityTarget {
                id: "ornith_1_0_9b_q4_k_m",
                tool_capable: true,
                family: "ornith",
                quantization: "Q4_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context beyond the tested single-session window, broader arbitrary templates beyond the native Ornith ChatML renderer, portability beyond a single 6GB-class GPU host, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_qwen35_bpe_mark_folding_deferred",
                tensors_load: "validated_all_427_qwen35_tensors_fully_gpu_resident",
                generation_runs: "runnable_serve_chat_plus_agent_eval_validated",
                parity_audited: "cuda_5_prompt_pass_cross_backend_tolerance_attributed_near_ties_vs_llamacpp_acd79d6",
                performance_measured: "cuda_device_decode_loop_18_8_toks_median_measured",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact qwen35 Q4_K_M row (ornith-1.0-9b-Q4_K_M.gguf, sha256 2711bf1e...) is loaded_now=true, generation_ready=true, matching active_model_id, served with CAMELID_RUNNABLE_SERVE=1 and CAMELID_QWEN35_CUDA=1",
                tested_context: "short_serve_smoke_plus_agent_eval_read_list_write",
                chat_template_renderer: "ornith-chatml-native",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "agent_eval_read_list_write",
                latest_checked_result: "pass",
                latest_checked_output: "camelid.agent_eval/v1 PASS (full 3-case battery)",
                evidence: "qwen35 (Ornith-1.0-9B) Q4_K_M fully GPU-resident (CAMELID_QWEN35_CUDA=1): 5-prompt greedy parity vs the pinned llama.cpp acd79d6 CUDA oracle PASSES under the cross-backend tolerance policy — 2/5 token-identical at n=64 and every flip probed and attributed to soft positions with <=0.33-nat top-2 gaps where the oracle's own CPU-vs-CUDA backends also flip (qa/ornith/constrained-vram/RECEIPT_ITEM2_qwen35_parity_cuda.json, probes + oracle/camelid internal-variance controls committed alongside). The full read_file/list_dir/write_file agent battery passes on this exact file with a committed camelid.agent_eval/v1 PASS receipt (qa/agent-eval/ornith-1.0-9b-Q4_K_M-1783019779-PASS.json); tool_capable earned ONLY by that receipt. Decode throughput 18.8 tok/s median via the device-side decode loop (qa/ornith/constrained-vram profile CSVs). NOT model-native/larger context, NOT broader templates, NOT multi-session throughput claims.",
                next_step: "preserve the CUDA parity + agent capability for this exact row; context-pack coverage, the frontend picker load path, and a normalized full-support bundle remain before any broader claim",
            },
            ModelCompatibilityTarget {
                id: "ornith_1_0_9b_q3_k_m",
                tool_capable: false,
                family: "ornith",
                quantization: "Q3_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "direct cross-engine parity receipt on this exact quant (current chain is via the parity-certified CPU oracle), an agent-eval battery for tool_capable, broader templates beyond the native Ornith ChatML renderer, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_qwen35_bpe_mark_folding_deferred",
                tensors_load: "validated_all_427_qwen35_tensors_fully_gpu_resident_native_q5k_gemv",
                generation_runs: "runnable_gpu_single_session_16k_maxpos_validated",
                parity_audited: "gpu_greedy_matches_certified_cpu_oracle_solo_at_16k_maxpos",
                performance_measured: "cuda_device_decode_loop_15_4_toks_median_measured",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact qwen35 Q3_K_M row (ornith-1.0-9b-Q3_K_M.gguf, sha256 16f54df5...) is loaded_now=true, generation_ready=true, matching active_model_id, served with CAMELID_RUNNABLE_SERVE=1 and CAMELID_QWEN35_CUDA=1",
                tested_context: "single_session_16k_maxpos_residency_smoke",
                chat_template_renderer: "ornith-chatml-native",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_16k_residency_and_parity_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "16K full residency 4747 MiB peak (1397 MiB headroom) + GPU==CPU-oracle greedy match",
                evidence: "qwen35 (Ornith-1.0-9B) Q3_K_M, imatrix-quantized in-house from the sha256-pinned bf16 source with llama-quantize at the pinned acd79d6 reference (agentic-coding calibration corpus committed): fully GPU-resident at 16K context with 4747 MiB peak VRAM and 1397 MiB headroom on a 6144 MiB card, beating the llama.cpp figure for the same file via sparse KV over the 8 full-attention layers (qa/ornith/constrained-vram/RECEIPT_ITEM4_residency.json). GPU generation is greedy token-identical to the CPU runnable oracle (single-token + multi-token, identical to the certified Q8_0/Q4_K_M reference sequence); the CPU oracle itself is the lane certified greedy token-identical vs llama.cpp acd79d6 — a DOCUMENTED FRONTIER: no direct side-by-side llama.cpp receipt on this exact quant yet. Held-out coding perplexity 2.4693 vs Q6_K 2.3636 (QUANT_QUALITY_TABLE.md). Decode throughput 15.4 tok/s median via the device-side decode loop. The four q5_K tensors run natively on the q5k_gemv resident kernel at wire size (parity re-certified: GPU==CPU-oracle single-token + greedy at 16K maxpos; previously upcast to Q8_0 blocks, ~+40 MiB, so the cited receipt's peak is a conservative upper bound).",
                next_step: "mint a committed camelid.agent_eval/v1 PASS battery on this exact file to earn tool_capable, and a direct 5-prompt llama.cpp side-by-side parity receipt to close the documented frontier",
            },
            // Ornith-1.0-9B (qwen35 hybrid gated-delta-net), runnable serve lane.
            // tool_capable earned by three committed camelid.agent_eval/v1 PASS
            // receipts (read_file / list_dir / write_file). The id matches the
            // side-loaded server id (general.name) so the live `--agent` gate
            // (`active_tool_capable`) resolves; `family: "ornith"` routes tool-call
            // parsing to the custom `<function=â€¦>` XML parser.
            ModelCompatibilityTarget {
                id: "Ornith 1.0 9B",
                tool_capable: true,
                family: "ornith",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context, broader arbitrary templates beyond the native Ornith ChatML renderer, production throughput on the pure-f32 runnable lane, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_qwen35_bpe_mark_folding_deferred",
                tensors_load: "validated_all_427_qwen35_tensors",
                generation_runs: "runnable_serve_chat_plus_agent_eval_validated",
                parity_audited: "greedy_token_identical_4_prompt_vs_llamacpp_acd79d6",
                performance_measured: "avx2_q8_dot_plus_batched_prefill_measured",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact qwen35 Q8_0 row is loaded_now=true, generation_ready=true, matching active_model_id, served with CAMELID_RUNNABLE_SERVE=1",
                tested_context: "short_serve_smoke_plus_agent_eval_read_list_write",
                chat_template_renderer: "ornith-chatml-native",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "agent_eval_read_list_write",
                latest_checked_result: "pass",
                latest_checked_output: "3x camelid.agent_eval/v1 PASS",
                evidence: "qwen35 (Ornith-1.0-9B) Q8_0 on Windows x86_64 CPU: all 427 tensors of the hybrid gated-delta-net arch load; the from-scratch runnable lane is greedy token-identical to the pinned llama.cpp acd79d6 oracle on 4 prompts (G-PARITY, qa/ornith/G-PARITY-qwen35-vs-llamacpp.md); the model's custom <function=...> tool format lifts cleanly into structured tool_calls (G-TOOLCALL, qa/ornith/G-TOOLCALL-qwen35.md); and three distinct tools (read_file/list_dir/write_file) each pass the agent loop with a committed camelid.agent_eval/v1 PASS receipt (qa/agent-eval/ornith-1.0-9b-Q8_0-{1782768506,1782768988,1782770407}-PASS.json, G-AGENT, qa/ornith/G-AGENT-qwen35.md). tool_capable earned ONLY by those receipts. NOT model-native/larger context, NOT broader templates, NOT production throughput (the runnable lane is correct but slow vs an optimized SIMD/CUDA kernel).",
                next_step: "preserve the parity-certified runnable lane + the read/list/write agent capability for this exact row; an optimized-lane qwen35 kernel for production throughput, context-pack coverage, and the frontend load path remain before any broader/full-support claim",
            },
            // MUSTER M-A1 (2026-07-16): gemma3 1B promoted on the runnable serve lane
            // (the only lane with a correct gemma3 forward — the optimized dense binder
            // silently drops gemma3's norms and has no GeGLU, so it stays fail-closed
            // for chat via is_runnable_serve_arch). Filename-anchored id
            // (`gemma_3_1b_it_q8_0` = normalized GGUF filename) so the frontend
            // exact-row identity matcher resolves the local file; the catalog entry
            // `gemma3_1b_it_q8_0` resolves via the filename/name match.
            ModelCompatibilityTarget {
                id: "gemma_3_1b_it_q8_0",
                tool_capable: false,
                family: "gemma3_runnable_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "strict all-leg parity (one documented 50-token near-tie at a 0.3416-nat gap — ABOVE the Ornith 0.33-nat soft-position line, attributed only under the Llama-3.2-3B gaps-stated precedent), context above the 512-token sliding window (the runnable lane implements no window mask, so longer sequences are mathematically wrong by construction), bounded-context packs, perf (observed ~5 s/token on the pure-f32 runnable CPU lane — recorded, not a claim), tool capability (tools fail closed with a typed 422), GPU lane (runnable CUDA is qwen35-gated), portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_cross_engine_prompt_tokenization_identical_5_of_5_spm_llama_262k_vocab",
                tensors_load: "validated_183_q8_0_plus_157_f32_tied_embedding_runnable_resident",
                generation_runs: "runnable_serve_cpu_chat_greedy_with_eog_stop",
                parity_audited: "gemma3_marker_chat_token_and_text_parity_1_5_50_four_of_five_prompts_all_depths_fifth_identical_at_1_5_single_flip_at_50_index_16_gap_stated_vs_llamacpp_acd79d6",
                performance_measured: "observed_about_5_s_per_token_cpu_runnable_recorded_not_a_perf_claim",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact gemma3 Q8_0 row (gemma-3-1b-it-Q8_0.gguf, sha256 b205840c...) is loaded_now=true, generation_ready=true, matching active_model_id, served with CAMELID_RUNNABLE_SERVE=1",
                tested_context: "gemma3_marker_chat_1_5_50_smoke_total_sequence_well_under_the_512_token_sliding_window",
                chat_template_renderer: "gemma3_marker_native_byte_locked_by_shapes_pack",
                chat_template_shape_pack: "validated_in_src_pack_lock_test",
                chat_template_shape_pack_id: "gemma3-chat-template-shapes-v1",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "runnable_serve_gemma3_chat_parity",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb/README.md",
                evidence: "exact row gemma-3-1b-it-Q8_0.gguf (sha256 b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5, 1,069,306,368 bytes, ggml-org/gemma-3-1b-it-GGUF upstream-verified exact, license gemma): 183 Q8_0 + 157 F32 tensors, tied embedding, served via the runnable lane (CAMELID_RUNNABLE_SERVE=1, CPU) with the new gemma3 marker renderer — byte-locked against the pinned oracle's /apply-template output by qa/prompt-packs/gemma3-chat-template-shapes-v1.json and an in-src pack-lock test — and EOG-stopping dense decode. Greedy chat parity vs pinned llama.cpp acd79d603 (build 9632, CPU llama-server /completion -ngl 0), committed 5-prompt gate pack (qa/prompt-packs/gemma3-chat-gate-pack-v1.json) at 1/5/50: 4/5 prompts token-AND-text identical at every depth (incl. a natural early-stop leg proving stop semantics); the fifth is identical at 1/5 and diverges at position 16 of the 50-token leg — probed in solo oracle sessions: camelid's token is the oracle's immediate #2 at a 0.3416-nat top-2 gap, with the oracle BIT-STABLE across thread-count (default/4/2) and runtime-repack kernel controls. DISCLOSED PLAINLY: that gap exceeds the Ornith <=0.33-nat soft-position precedent and no oracle-side backend flip exists, so the flip rests on the Llama-3.2-3B 'documented near-ties with gaps stated' precedent — the weakest single attribution in the MUSTER campaign to date. Cross-engine prompt tokenization identical on 5/5 prompts (llama /tokenize vs camelid encode, incl. BOS placement); determinism sanity byte-identical across two independent serve sessions; the pre-existing HF-reference receipt qa/runnable/gemma3-parity.json (all_greedy_match=true, SHA-anchored to this exact file) anchors the engine math independently. Sequences are scoped well under the 512-token sliding window the runnable lane does not mask. Observed decode ~5 s/token on the pure-f32 CPU lane (recorded, NOT a perf claim). Bundle qa/evidence-bundles/gemma3-1b-q8-runnable-serve-chat-parity-20260716-head-6d0d57eb. NOT claimed: strict all-leg parity, any context at or beyond the 512-token window, bounded-context packs, tools (fail closed, typed 422), throughput, GPU residency, neighboring gemma3 sizes/quants, or the gemma4 family.",
                next_step: "a runnable-lane sliding-window mask (or an optimized-lane gemma3 forward) before any context claim at or beyond 512 tokens; per-row perf work if the ~5 s/token CPU lane matters; agent-eval battery before any tool_capable claim",
            },
            ModelCompatibilityTarget {
                id: "tinyllama_1_1b_chat_q8_0",
                tool_capable: false,
                family: "llama_spm_decoder",
                quantization: "Q8_0",
                status: "supported_current_gate",
                support_scope: "current_full_gate_exact_row",
                full_support_status: "current_gate_refresh_under_stricter_bar",
                full_support_blockers: "do not widen beyond TinyLlama 1.1B Chat Q8_0 without repeated current-head API/WebUI/parity/RSS/context evidence under the stricter bar; arbitrary/Jinja template behavior and production throughput remain outside this exact current gate unless separately validated",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "validated",
                parity_audited: "validated",
                performance_measured: "measured",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact Q8_0 row is loaded_now=true, generation_ready=true, and selected by active_model_id",
                tested_context: "short_50_token_gate_plus_bounded_512_context_pack",
                chat_template_renderer: "tinyllama-marker",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "tinyllama-chat-template-shapes-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "tinyllama-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "direct_chat_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Certainly! Here",
                evidence: "five-prompt TinyLlama Q8_0 parity gate plus current template-shape, bounded 512-context, API/WebUI, and RSS/perf artifacts recorded in STATUS.md",
                next_step: "extend to larger contexts and additional LLaMA-family/quant targets before broadening support claims",
            },
            ModelCompatibilityTarget {
                id: "llama32_1b_instruct_q8_0",
                tool_capable: false,
                family: "llama_bpe_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context beyond checked packs, broader arbitrary templates beyond the supported metadata-Jinja Llama 3.2 1B row template, production throughput, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_for_compact_and_prompt_pack",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke_validated",
                parity_audited: "compact_50_token_plus_prompt_pack_match",
                performance_measured: "bounded_unique_chat_perf_rss_validated",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "short_api_webui_smoke_plus_first_512_second_1024_third_2048_fourth_4096_and_fifth_8192_context_packs",
                chat_template_renderer: "metadata_jinja_supported_for_exact_row",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "llama3-chat-template-shapes-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "llama3-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "llama3-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "llama3-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "llama3-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "llama3-context-8192-smoke-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "llama3-context-8192-smoke-v1",
                latest_checked_result: "pass",
                latest_checked_output: "CMLD-819",
                evidence: "the exact bartowski Llama-3.2-1B-Instruct-Q8_0 GGUF has exact-row load, completion, chat-completion, frontend-smoke evidence, compact/prompt-pack parity, first bounded 512-context parity, second bounded 1024-context parity, third bounded 2048-context parity after the RoPE frequency-factor fix, fourth bounded 4096-context parity, and fifth bounded 8192-context parity on current head aaf9207d166999a21f4fde2a3f2ac5631f2fcecb, bounded compact template-shape coverage, metadata-Jinja renderer parity for the row template shapes, and bounded unique-chat perf/RSS evidence; Camelid supports exact-row smoke and the checked 512/1024/2048/4096/8192 context packs for this row only, not model-native/larger context beyond checked packs or broader/full support",
                next_step: "preserve exact-row smoke plus checked 512/1024/2048/4096/8192 context support while normalizing model-native/larger context beyond checked packs, broader arbitrary-template behavior beyond the supported 1B metadata-Jinja row template, production throughput, portability, and durable full-support bundle evidence before any broader/full-support claim",
            },
            ModelCompatibilityTarget {
                id: "llama32_3b_instruct_q8_0",
                tool_capable: true,
                family: "llama_bpe_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "the May-era typed lanes (template shapes, perf/RSS, API/WebUI smoke, broader prompt-pack parity) were measured on the prior upload (sha256 b5607b50...) and await re-anchoring to the canonical file; plus model-native/larger context beyond the anchored 512-8192 ladder, broader arbitrary/Jinja templates beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_for_compact_and_small_prompt_pack",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke_plus_five_prompt_api_smoke",
                parity_audited: "compact_and_broader_prompt_pack_50_token_match",
                performance_measured: "bounded_unique_chat_perf_rss_validated",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "short_api_webui_smoke_with_broader_prompt_pack_parity_plus_anchored_raw_decode_512_1024_2048_4096_8192_context_ladder",
                chat_template_renderer: "metadata_jinja_supported_for_exact_row",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "llama3-chat-template-shapes-v1",
                bounded_context_512_pack: "validated_anchored_raw_decode_ladder",
                bounded_context_512_pack_id: "llama32-3b-anchored-raw-ladder-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_anchored_raw_decode_ladder",
                bounded_context_1024_pack_id: "llama32-3b-anchored-raw-ladder-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_anchored_raw_decode_ladder",
                bounded_context_2048_pack_id: "llama32-3b-anchored-raw-ladder-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_anchored_raw_decode_ladder",
                bounded_context_4096_pack_id: "llama32-3b-anchored-raw-ladder-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_anchored_raw_decode_ladder",
                bounded_context_8192_pack_id: "llama32-3b-anchored-raw-ladder-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "llama32-3b-anchored-raw-ladder-v1",
                latest_checked_result: "pass",
                latest_checked_output: "50/50 greedy tokens identical on all five buckets",
                evidence: "the exact tracked Llama-3.2-3B-Instruct-Q8_0 GGUF (sha256 f34112a1..., the file pinned by the June capability-matrix receipts) has a fresh anchored raw-decode context ladder — 512/1024/2048/4096/8192 buckets (408/885/1893/3978/8049 llama3 tokens), token-AND-text identical to pinned llama.cpp acd79d603 at 50 greedy tokens, fully GPU-resident, at qa/evidence-bundles/llama32-3b-context-512-8192-anchored-20260710T2119-head-6527a770/manifest.json — plus June 2026 capability-matrix receipts on the same file. Earlier evidence (canonical Ubuntu API/WebUI refresh at qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json, compact/broader prompt-pack parity, the May 512/1024/2048 template-context packs, template-shape coverage, and bounded unique-chat perf/RSS) was produced against a PRIOR upload of this model name (sha256 b5607b50...) and is retained as historical evidence for that file; it is disclosed, not inherited. Camelid supports exact-row smoke for this row only, not broader/full support",
                next_step: "preserve exact-row smoke plus the anchored checked 512/1024/2048/4096/8192 raw-decode context ladder while re-anchoring the remaining May-era evidence families (API/WebUI refresh, template shapes, perf/RSS) to the canonical file and normalizing model-native/larger context, broader arbitrary/Jinja template behavior, production throughput, portability, and durable full-support bundle evidence before any broader/full-support claim",
            },
            // Llama-3.2-3B-Instruct K-quant rows (mixed K-quant + Q6_K, GPU-resident CUDA
            // decode). Filename-anchored ids (`llama_3_2_3b_instruct_q{4,5}_k_m` = normalized
            // GGUF filename) so the frontend exact-row identity matcher resolves them BEFORE
            // the llama-bpe branch (which only knows the Q8_0 quant and would quant-mismatch).
            // Scoped to raw-decode parity (scripts/raw-decode-parity.mjs); per-quant
            // API/WebUI/chat-template-shape/context/perf evidence is a follow-up, so those
            // fields stay not_promoted while `status` unlocks the exact row.
            ModelCompatibilityTarget {
                id: "llama_3_2_3b_instruct_q4_k_m",
                tool_capable: false,
                family: "llama_bpe_decoder",
                quantization: "Q4_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_gpu_resident_raw_decode_parity_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "strict all-prompt parity on open-ended generation (3/8 probes are documented benign f32 near-ties), per-quant API/WebUI/chat-template-shape/bounded-context/perf evidence (this row inherits the Llama 3.2 metadata-Jinja renderer, not re-proven per quant), other Llama sizes/quants, other K-quant files, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated_168_q4_k_plus_29_q6_k_wire_only_fully_gpu_resident_tied_embeddings",
                generation_runs: "gpu_resident_cuda_raw_decode_parity",
                parity_audited: "raw_decode_token_and_text_parity_1_5_50_confident_probes_all_pass_5_of_8_incl_3p5k_long_context_and_code_3_open_ended_near_ties_documented_gpu_resident_cuda_vs_llamacpp_acd79d6",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact Llama-3.2-3B-Instruct-Q4_K_M.gguf row (sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff) plus Q4_K_M quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id on the CUDA-resident decode lane",
                tested_context: "raw_decode_1_5_50_token_smoke_plus_one_3p5k_long_context_probe",
                chat_template_renderer: "metadata_jinja_llama_3_2_inherited_not_reproven_per_quant",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "windows_cuda_resident_raw_decode_parity",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/llama-3.2-3b-q4_k_m-windows-cuda-resident-parity-20260628T004547Z-head-bb3c3528/README.md",
                evidence: "exact row Llama-3.2-3B-Instruct-Q4_K_M.gguf (sha256 6c1a2b41…, 1.88 GiB): mixed K-quant (168 Q4_K + 29 Q6_K + 58 F32 norms; attn_v / ffn_down / tied token_embd(lm_head) are Q6_K, q/k/o/gate/up are Q4_K; TIED embeddings; one run drives BOTH q4k_gemv and q6k_gemv). GPU-resident CUDA decode is token-AND-text-identical to pinned llama.cpp acd79d6 on all confident/structured raw-completion probes at 1/5/50 tokens (5/8, incl. a ~3.5k-token long-context lighthouse-logbook continuation and code completion, both to depth 50); 3 open-ended probes diverge at a benign greedy f32 near-tie (coherent output throughout — the same documented frontier as the Q8_0 near-tie probes). Bundle qa/evidence-bundles/llama-3.2-3b-q4_k_m-windows-cuda-resident-parity-20260628T004547Z-head-bb3c3528 (confident_probes_all_pass=true), harness scripts/raw-decode-parity.mjs. tool_capable=false is EVIDENCED (not assumed): this exact GGUF FAILs the agent-eval battery (qa/agent-eval/Llama-3.2-3B-Instruct-Q4_K_M-1783377491-FAIL.json) — the lower quant degrades tool-call generation, unlike the Q8_0 sibling (which passes) and the Qwen3-4B-Q4_K_M row (whose native tool template survives Q4). NOT claimed: strict all-prompt parity on open-ended generation, per-quant API/WebUI/chat-template-shape/context/perf evidence, other Llama sizes/quants, or throughput.",
                next_step: "add per-quant API/WebUI/chat-template-shape and bounded-context evidence for this exact Q4_K_M row before any broader claim",
            },
            ModelCompatibilityTarget {
                id: "llama_3_2_3b_instruct_q5_k_m",
                tool_capable: false,
                family: "llama_bpe_decoder",
                quantization: "Q5_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_gpu_resident_raw_decode_parity_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "per-quant API/WebUI/chat-template-shape/bounded-context/perf evidence (this row inherits the Llama 3.2 metadata-Jinja renderer, not re-proven per quant), the model GGUF is not currently on the dev disk (evidenced by the committed bundle), other Llama sizes/quants, other K-quant files, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated_q5_k_plus_q6_k_wire_only_fully_gpu_resident",
                generation_runs: "gpu_resident_cuda_raw_decode_parity",
                parity_audited: "raw_decode_token_and_text_parity_1_5_50_all_pass_gpu_resident_cuda_vs_llamacpp_acd79d603",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact Llama-3.2-3B-Instruct-Q5_K_M.gguf row (sha256 0b94ccd04d908304cec5246a3d942b64417a423bc5c6d47c73bc557e590b5194) plus Q5_K_M quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id on the CUDA-resident decode lane",
                tested_context: "raw_decode_1_5_50_token_smoke",
                chat_template_renderer: "metadata_jinja_llama_3_2_inherited_not_reproven_per_quant",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "windows_cuda_resident_raw_decode_parity",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/llama-3.2-3b-q5_k_m-windows-cuda-resident-parity-20260703T201602Z-head-ab88962/README.md",
                evidence: "exact row Llama-3.2-3B-Instruct-Q5_K_M.gguf (sha256 0b94ccd0…, 2.16 GiB): mixed K-quant (Q5_K + Q6_K; one run drives BOTH q5k_gemv and q6k_gemv). GPU-resident CUDA decode is token-AND-text-identical to pinned llama.cpp acd79d603 at 1/5/50 tokens on the raw-prompt harness (all_pass=true). Bundle qa/evidence-bundles/llama-3.2-3b-q5_k_m-windows-cuda-resident-parity-20260703T201602Z-head-ab88962, harness scripts/raw-decode-parity.mjs. The GGUF is not currently on the dev disk; this row rests on the committed bundle (captured on an RTX 4060 Laptop GPU, a different card than the RTX 3060 dev box). NOT claimed: per-quant API/WebUI/chat-template-shape/context/perf evidence, other Llama sizes/quants, or throughput.",
                next_step: "re-acquire the exact GGUF and add per-quant API/WebUI/chat-template-shape and bounded-context evidence before any broader claim",
            },
            // MUSTER M-B1 (2026-07-16): 1B Q4_K_M promoted on the same raw-decode bar as the
            // 3B K-quant rows above. Filename-anchored id; the file is a LOCAL artifact whose
            // upstream provenance is unresolved (see MUSTER_ACQUISITION.md) — the evidence
            // says so explicitly and no repo is claimed.
            ModelCompatibilityTarget {
                id: "llama_3_2_1b_instruct_q4_k_m",
                tool_capable: false,
                family: "llama_bpe_decoder",
                quantization: "Q4_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_gpu_resident_raw_decode_parity_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "strict all-prompt parity on open-ended generation (3/8 probes are attributed benign greedy near-ties), per-quant API/WebUI/chat-template-shape/bounded-context/perf evidence (this row inherits the Llama 3.2 metadata-Jinja renderer, not re-proven per quant), upstream provenance unresolved (local file, SHA-anchored; matches no surveyed publisher upload), other Llama sizes/quants, other K-quant files, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_prompt_tokenization_identical_8_of_8_incl_1867_token_prompt",
                tensors_load: "validated_96_q4_k_plus_17_q6_k_wire_only_fully_gpu_resident_tied_embeddings",
                generation_runs: "gpu_resident_cuda_raw_decode_parity",
                parity_audited: "raw_decode_token_and_text_parity_1_5_50_confident_probes_all_pass_5_of_8_incl_1867_token_long_context_and_code_3_open_ended_near_ties_attributed_gpu_resident_cuda_vs_llamacpp_acd79d6",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact Llama-3.2-1B-Instruct-Q4_K_M.gguf row (sha256 6a74661014a3e2f139871f81e6cec852c489a627d169de503a3c0434a10c503d) plus Q4_K_M quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id on the CUDA-resident decode lane",
                tested_context: "raw_decode_1_5_50_token_smoke_plus_one_1867_token_long_context_probe",
                chat_template_renderer: "metadata_jinja_llama_3_2_inherited_not_reproven_per_quant",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "windows_cuda_resident_raw_decode_parity",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/llama-3.2-1b-q4_k_m-windows-cuda-resident-parity-20260716T151718Z-head-052c4030/README.md",
                evidence: "exact row Llama-3.2-1B-Instruct-Q4_K_M.gguf (sha256 6a74661014a3e2f139871f81e6cec852c489a627d169de503a3c0434a10c503d, 807,693,984 bytes; LOCAL FILE, upstream provenance unresolved — matches no current upload of six surveyed publishers and carries no producer metadata, see MUSTER_ACQUISITION.md): mixed K-quant (96 Q4_K + 17 Q6_K + 34 F32 norms; attn_v/ffn_down are Q6_K in 8 of 16 layers each; TIED Q6_K token_embd(lm_head); one run drives BOTH q4k_gemv and q6k_gemv). GPU-resident CUDA decode (16/16 layers, cuda_resident_kquant_runtime) is token-AND-text-identical to pinned llama.cpp acd79d603 (build 9632, CPU llama-server /completion -ngl 0) on 5/8 prompts of the committed pack qa/prompt-packs/llama32-1b-q4km-raw-decode-pack-v1.json at every 1/5/50 depth — including code completion, structured JSON, repetitive extraction, and a 1,867-token long-context continuation to depth 50; prompt tokenization identical on all 8 prompts. The 3 open-ended flips are probed and attributed benign near-ties (near-tie-analysis.json): each camelid token is the oracle's immediate #2 at a 0.074/0.106/0.0135-nat top-2 gap (oracle n_probs captured in the same solo capture session), and camelid's CPU K-quant lane emits the oracle's token at 2 of the 3 flip positions (cross-backend control). Determinism sanity: two independent CPU-lane serve sessions byte-identical on the full pack. Bundle qa/evidence-bundles/llama-3.2-1b-q4_k_m-windows-cuda-resident-parity-20260716T151718Z-head-052c4030, harness scripts/raw-decode-parity.mjs unmodified. tool_capable=false: NOT evaluated for this file (no agent-eval receipt; the 3B Q4_K_M sibling FAILs the same battery, so nothing is inherited). NOT claimed: strict all-prompt parity on open-ended generation, per-quant API/WebUI/chat-template-shape/bounded-context/perf evidence, upstream repo provenance, other Llama sizes/quants, or throughput.",
                next_step: "resolve or re-source upstream provenance (or accept the local-file anchor durably), then per-quant API/WebUI/chat-template-shape and bounded-context evidence before any broader claim",
            },
            ModelCompatibilityTarget {
                id: "llama3_8b_instruct_q8_0",
                tool_capable: false,
                family: "llama_bpe_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context beyond the checked 512/1024/2048 packs, arbitrary templates, throughput, portability, repeated current-head evidence, and durable normalized full-support bundles remain missing",
                metadata_parses: "real_artifact_inspected_and_config_guarded",
                tokenizer_works: "validated_for_compact_llama3_bpe",
                tensors_load: "validated_for_lazy_file_backed_q8_backend_runs",
                generation_runs: "api_completion_and_chat_smoke_validated",
                parity_audited: "compact_50_token_plus_broader_50_token_prompt_pack_match",
                performance_measured: "bounded_ubuntu_backend_memory_gate_plus_lazy_q8_hotpath_costs",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "short_api_webui_smoke_with_broader_50_token_plus_checked_512_1024_2048_context_packs",
                chat_template_renderer: "compact",
                chat_template_shape_pack: "validated_compact_pack",
                chat_template_shape_pack_id: "llama3-chat-template-shapes-v1",
                bounded_context_512_pack: "validated_first_pack",
                bounded_context_512_pack_id: "llama3-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "llama3-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "llama3-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "llama3-context-2048-smoke-v1",
                latest_checked_result: "pass",
                latest_checked_output: "CMLD-204",
                evidence: "the exact tracked Llama 3 8B Instruct Q8_0 GGUF has compact prompt-token/1-token/5-token/50-token parity, a three-prompt 50-token Ubuntu parity run, API/frontend smoke, bounded-memory evidence, checked 512/1024/2048-context packs, one compact chat-template-shapes pack, and retained-block lazy-Q8 hot-path cost probes. The published source/runtime-head 1024/2048 pass / current-head 1024/2048 pass is recorded at qa/evidence-bundles/llama3-8b-context-1024-2048-current-head-20260509T041451Z-head-8e26be0a73c0/manifest.json with prompt-token, generated-token, and generated-text parity for CMLD-102 and CMLD-204. Camelid supports exact-row smoke plus the checked bounded packs for this row only; no model-native/larger context or broader/full support is implied",
                next_step: "preserve exact-row smoke plus checked 512/1024/2048 context support while collecting model-native/larger-context proof, broader/full-support, production-throughput, portability, arbitrary-template evidence, and repeated current-head evidence before any wider 8B claim",
            },
            ModelCompatibilityTarget {
                id: "gemma4_e4b_it_q8_0",
                tool_capable: false,
                family: "gemma4_ple_matformer_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "performance/RSS gates, portability, arbitrary/Jinja template coverage beyond the gemma4 marker template, and durable current-head QA bundles remain missing; bounded context packs 512-8192 are checked (full-budget, no recorded frontiers)",
                metadata_parses: "validated",
                tokenizer_works: "validated_for_gemma4_spm",
                tensors_load: "validated_mmap_wire_backed_q8_instant_load",
                generation_runs: "api_nonstreaming_and_streaming_chat_smoke_plus_cli_greedy",
                parity_audited: "basic_v1_pack_5of5_full_budget_cpu_and_gpu_vs_pinned_plain_f32_gemv_comparator_llama_cpp_5d56eff",
                performance_measured: "functional_milestone_only_not_perf_validated",
                frontend_load_path_verified: "served_via_gemma4_runtime_flag",
                frontend_readiness_gate: "green only when this exact gemma4 GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id (serve requires CAMELID_GEMMA4_SERVE=1)",
                tested_context: "short_api_webui_chat_smoke_streaming_and_nonstreaming_plus_cli_greedy_parity",
                chat_template_renderer: "gemma4_marker",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "gemma4-template-shapes-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "gemma4-context-512-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "gemma4-context-1024-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "gemma4-context-2048-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "gemma4-context-4096-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "gemma4-context-8192-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gemma4-context-8192-v1",
                latest_checked_result: "pass",
                latest_checked_output: "8192",
                evidence: "the exact tracked gemma-4-E4B-it-Q8_0 GGUF (8,192,951,456 bytes, general.architecture=gemma4, gemma4 SPM tokenizer) loads through the mmap wire-backed Q8 lane (no eager decode; ~instant load) and generates greedily with token IDs identical to the reference llama.cpp b9430 greedy decode ([9079, 236761, 108, 1018, 14977, 53121, 2900, 563, 506, 5279, 529, 7001] for 'The capital of France is'). Served live through /v1/chat/completions both non-streaming and streaming (OpenAI chat.completion.chunk shape) behind CAMELID_GEMMA4_SERVE; non-streaming returns 'Paris' with finish_reason stop and no prompt echo, streaming yields incremental token deltas then [DONE], /v1/health reports backend=gemma4-runtime/model_family=gemma4/gemma4_available=true, and the CLI greedy output matches the API for the same templated prompt. Committed basic_v1 pack parity vs the pinned plain-f32 GEMV comparator (llama.cpp 5d56eff, --no-repack -fa off -ctk f32 -ctv f32 -ub 1): CPU and GPU both match all five prompts full-budget with no frontier annotations (the previously recorded knife-edge was the missing rope_freqs proportional-rope semantics, since implemented from the reference graph). Distributed layer-sharding greedy output is token-identical to the oracle with fail-closed handshake/checksum/shared-KV guards. Raw logs: qa/evidence-bundles/gemma4-e4b-it-q8-0-20260610T103400Z-head-96a75007b156. See docs/gemma4-engine-status.md. Bounded-context ladder: committed packs + pinned-comparator oracles at qa/gemma4/prompt_packs/context_{512,1024,2048,4096,8192}_v1.json with recall asserted at capture; CPU and GPU parity cells pass full-budget at every bucket with no recorded frontiers (10/10 cells this row). The reference-exact chat template (both thinking modes) is locked byte-for-byte and token-for-token by qa/gemma4/template_shapes_v1.json. Distributed layer-sharding SERVE lane (CAMELID_GEMMA4_WORKER/CAMELID_GEMMA4_SPLIT) reuses wire protocol v1 with per-request worker sessions. Camelid supports exact-row text-token generation + serve smoke with checked bounded 512-8192 context packs for this row only; no model-native/larger context, performance, portability, multimodal, or full support is implied",
                next_step: "add performance/RSS gates and durable current-head QA bundles, and broaden template coverage before any wider gemma4 claim; bounded 512-8192 context packs are checked",
            },
            ModelCompatibilityTarget {
                id: "gemma4_e4b_it_nvfp4",
                tool_capable: false,
                family: "gemma4_ple_matformer_decoder",
                quantization: "NVFP4",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_gpu_resident_raw_decode_parity_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "NVFP4 is a space/speed quant, NOT quality-competitive-beyond-tolerance with Q4_K: a current-engine teacher-forced re-eval is a comparator-sensitive NEAR-TIE (90.5% vs Q4K 91.9%, gap 1.4pp within the pre-registered 2.0pp GO tolerance; the frozen Gate G3 NO-GO at 88.5% vs 92.6% stands as history). performance/RSS gates, bounded context packs (not run for NVFP4), portability, and arbitrary/Jinja template coverage beyond the gemma4 marker template remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_for_gemma4_spm",
                tensors_load: "validated_metal_resident_wire_nvfp4_plus_q8_embeddings",
                generation_runs: "metal_gpu_resident_greedy_e2e_real_artifact_plus_cpu_forced_decode_eval",
                parity_audited: "nvfp4_cpu_decode_bit_exact_arm_gate_g_m1_13of13_plus_metal_self_parity_metal_gemma4_resident_nvfp4_forward_matches_cpu_plus_basalt_leg_b_cross_engine_8of9_plus_fresh_macos_llama_cpp_acd79d603_spotcheck",
                performance_measured: "isolated_128tok_greedy_decode_12_12_tok_s_metal_resident_1_45x_q8_0_parent_not_perf_validated",
                frontend_load_path_verified: "served_via_gemma4_runtime_metal_resident_lane",
                frontend_readiness_gate: "green only when this exact gemma4 NVFP4 GGUF row plus NVFP4 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id (Metal resident lane opt-in via the macOS-only gemma4-generate-gpu subcommand)",
                tested_context: "gpu_resident_raw_decode_parity_smoke_plus_teacher_forced_top1_quality_eval",
                chat_template_renderer: "gemma4_marker",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_resident_raw_decode_parity_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Paris",
                evidence: "the exact tracked gemma-4-E4B-it-NVFP4-mm.gguf (6,058,607,776 bytes, sha256 eb293344972e2b292a043b8e7649b9788dca915b034e5c2721cfc531cf9863d9, general.architecture=gemma4, gemma4 SPM tokenizer; 294 attn/ffn matmul tensors NVFP4 4.5bpw, embeddings Q8_0) loads on the macOS Metal GPU-resident lane and generates greedily end-to-end (first real-artifact run: 'The capital of France is' -> 'Paris.'). DECODE PARITY (the support anchor): NVFP4 CPU wire-lane decode is bit-exact on Apple Silicon/ARM (GABBRO Gate G-M1, 13/13 nvfp4_* tests); the Metal resident forward matches the CPU oracle (metal_gemma4_resident_nvfp4_forward_matches_cpu); BASALT Leg B cross-engine token parity vs pinned llama.cpp on the byte-identical file is 8/9 exact + 1 attributed 0.084-logit near-tie; and a fresh macOS spot-check has llama.cpp acd79d603 and Camelid both decoding 'The capital of France is' -> 'Paris'. QUALITY (honest, current engine): teacher-forced top-1 agreement vs the Q8_0 parent over the committed basic_v1+deep_v1 packs (296 positions, harness validated at Q8_0 self-agreement 296/296 and baseline token-identical to BASALT's committed baseline for 8/9 prompts) is 90.5% (268/296) vs the format-isolated Q4K-mm comparator 91.9% (272/296) -> gap 1.4pp within the pre-registered 2.0pp GO tolerance. This is a comparator-sensitive NEAR-TIE and NOT a claim of quality-competitiveness beyond that tolerance; the frozen Gate G3 NO-GO (88.5% vs 92.6%) stands as a recorded historical result and this is a disclosed current-engine re-eval. PERF: isolated 128-token greedy decode 12.12 tok/s on the Metal resident lane (fastest of NVFP4/Q8_0/Q4_0; 1.45x the Q8_0 parent at 26% smaller). imatrix is not a lever for NVFP4 (ggml-quants.c GGML_UNUSED(quant_weights); pure RTN). Receipts: qa/evidence-bundles/gabbro/support/ (GABBRO_SUPPORT_SUMMARY.md + eval_scores.json), qa/evidence-bundles/gabbro/phase1/ (G-M1). Camelid supports exact-row NVFP4 GPU-resident raw-decode parity smoke for this row only (the gemma4-E4B NVFP4 pilot); no bounded-context packs, quality-competitiveness beyond tolerance, other architectures/quants, performance, portability, multimodal, or full support is implied",
                next_step: "add bounded-context packs, performance/RSS gates, and durable current-head QA bundles for the NVFP4 row before any wider claim; the quality standing is a disclosed current-engine near-tie, not a G3 reversal",
            },
            ModelCompatibilityTarget {
                id: "gemma4_e2b_it_q8_0",
                tool_capable: false,
                family: "gemma4_ple_matformer_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "performance/RSS gates, portability, arbitrary/Jinja template coverage beyond the gemma4 marker template, and durable current-head QA bundles remain missing; bounded context packs 512-8192 are checked (full-budget, no recorded frontiers); multimodal input is fail-closed (text-token generation only)",
                metadata_parses: "validated_including_4to1_sliding_pattern_and_per_layer_ffn_widths",
                tokenizer_works: "validated_for_gemma4_spm",
                tensors_load: "validated_mmap_wire_backed_q8_instant_load",
                generation_runs: "api_nonstreaming_and_streaming_chat_smoke_plus_cli_greedy",
                parity_audited: "basic_v1_pack_5of5_full_budget_cpu_and_gpu_vs_pinned_plain_f32_gemv_comparator_llama_cpp_5d56eff",
                performance_measured: "functional_milestone_only_not_perf_validated",
                frontend_load_path_verified: "served_via_gemma4_runtime_flag",
                frontend_readiness_gate: "green only when this exact gemma4 GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id (serve requires CAMELID_GEMMA4_SERVE=1)",
                tested_context: "five_prompt_basic_v1_pack_greedy_parity_plus_api_webui_chat_smoke",
                chat_template_renderer: "gemma4_marker",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "gemma4-template-shapes-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "gemma4-context-512-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "gemma4-context-1024-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "gemma4-context-2048-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "gemma4-context-4096-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "gemma4-context-8192-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gemma4-context-8192-v1",
                latest_checked_result: "pass",
                latest_checked_output: "8192",
                evidence: "the exact tracked gemma-4-E2B-it-Q8_0 GGUF (5,048,350,848 bytes, general.architecture=gemma4, 35 layers with the 4:1 sliding_window_pattern and per-layer feed_forward_length array parsed from the GGUF, gemma4 SPM tokenizer) loads through the mmap wire-backed Q8 lane and generates greedily with prompt token ids, generated token ids, and generated text identical to the pinned reference llama.cpp 5d56eff for every prompt in qa/gemma4/prompt_packs/basic_v1.json (oracle at qa/gemma4/oracle/gemma-4-E2B-it-Q8_0.basic_v1.json). The Metal GPU-resident runtime matches the same five prompts token-for-token, and distributed layer-sharding greedy output (TCP split 13/35) is token-identical to the oracle with fail-closed handshake/checksum/shared-KV guards. Parity verified under both the repacked and plain-Q8 oracle kernel variants. Served through /v1/chat/completions and /v1/completions (streaming + non-streaming) behind CAMELID_GEMMA4_SERVE. Raw logs: qa/evidence-bundles/gemma4-e2b-it-q8-0-20260610T103119Z-head-96a75007b156. Bounded-context ladder: committed packs + pinned-comparator oracles at qa/gemma4/prompt_packs/context_{512,1024,2048,4096,8192}_v1.json with recall asserted at capture; CPU and GPU parity cells pass full-budget at every bucket with no recorded frontiers (10/10 cells this row). The reference-exact chat template (both thinking modes) is locked byte-for-byte and token-for-token by qa/gemma4/template_shapes_v1.json. Distributed layer-sharding SERVE lane (CAMELID_GEMMA4_WORKER/CAMELID_GEMMA4_SPLIT) reuses wire protocol v1 with per-request worker sessions. The Windows CUDA-resident lane (opt-in --features cuda + CAMELID_GEMMA4_CUDA=1) is token-AND-text identical to the same pinned oracle across basic_v1, deep_v1 (3/4 full-budget plus one oracle-recorded applies_to_gpu near-tie frontier), and the full context ladder 512-8192, with GPU-resident execution verified (nvidia-smi 2.7 GB VRAM / 90-100% GPU utilization on an RTX 3060 Laptop 6 GB, not a CPU fallback); receipts at qa/evidence-bundles/gemma4-e2b-q8-cuda-resident-parity-20260711T014910Z-head-15bf42e3, reproducible via the tests/gemma4_generation_parity.rs CUDA branch. This Windows CUDA parity is the E2B lane only; the E4B CUDA lane remains experimental (first-token argmax). Camelid supports exact-row text-token generation + serve smoke with checked bounded 512-8192 context packs for this row only; no model-native/larger context, performance, portability, multimodal, or full support is implied",
                next_step: "add performance/RSS gates and durable current-head QA bundles, and broaden template coverage before any wider gemma4 claim; bounded 512-8192 context packs are checked",
            },
            ModelCompatibilityTarget {
                id: "gemma4_12b_it_q8_0",
                tool_capable: false,
                family: "gemma4_dense_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_distributed_serve_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "single-node 16GB hosts are memory-bound (the supported lane is two-Mac distributed layer sharding); the reference has no sound bit-exact comparator mode for this row (two recorded comparator frontiers); no bounded context bucket, performance/RSS gate, portability, or durable current-head refresh exists yet; full-row GPU residency is untested and memory-infeasible on 16GB hosts",
                metadata_parses: "validated_including_per_layer_kv_head_array_and_vless_layers",
                tokenizer_works: "validated_for_gemma4_spm",
                tensors_load: "validated_mmap_wire_backed_q8_layer_range_shards",
                generation_runs: "distributed_two_node_api_chat_completions_and_sse_smoke",
                parity_audited: "basic_v1_pack_5of5_distributed_token_identical_to_single_node_3of5_full_budget_vs_pinned_comparator_with_two_recorded_frontiers",
                performance_measured: "two_mac_pair_cadence_6_2_to_6_75_tok_s_recorded_in_cli_bundle_only",
                frontend_load_path_verified: "served_via_gemma4_runtime_flag_distributed_lane",
                frontend_readiness_gate: "green only when this exact gemma4 GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id (serve requires CAMELID_GEMMA4_SERVE=1 plus CAMELID_GEMMA4_WORKER and CAMELID_GEMMA4_SPLIT pointing at a live gemma4-worker holding the tail layers)",
                tested_context: "short_api_chat_completions_sse_smoke_through_the_two_mac_distributed_lane",
                chat_template_renderer: "gemma4_marker",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "distributed_serve_api_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Paris",
                evidence: "the exact tracked gemma-4-12b-it-Q8_0 GGUF (12,669,646,240 bytes, general.architecture=gemma4, 48 layers with the per-layer attention.head_count_kv array and V-less full-attention layers parsed from the GGUF) runs as two-Mac distributed layer sharding: the CLI lane is token-identical to single-node camelid on all five basic_v1 prompts (qa/evidence-bundles/gemma4-12b-it-q8-0-two-mac-20260610T103711Z-head-96a75007b156; decode 6.2-6.75 tok/s across the pair, both 16GB nodes within budget; 3/5 prompts match the pinned comparator full-budget with two recorded reference-comparator frontiers), and the distributed SERVE lane answers /v1/chat/completions (non-streaming and SSE) and /v1/completions end-to-end across the pair (qa/evidence-bundles/gemma4-12b-it-q8-0-distributed-serve-20260610T235155Z-head-80e3dddfbdb4; master shard 0..24 + worker 24..48, wire protocol v1 with per-request worker sessions). Camelid supports exact-row text-token generation + serve smoke for this row only, and only through the two-Mac distributed lane; no single-node, bounded-context, performance, portability, multimodal, or full support is implied",
                next_step: "durable current-head refresh, bounded context buckets through the distributed lane, and performance/RSS gates before any wider 12B claim; single-node support is not a goal on 16GB hosts",
            },
            ModelCompatibilityTarget {
                id: "gemma4_26b_a4b_it_q4_0",
                tool_capable: false,
                family: "gemma4_a4b_moe_decoder",
                quantization: "Q4_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_distributed_serve_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "single-node 16GB hosts are memory-bound (13.4GB row; the supported lane is two-Mac distributed layer sharding); the reference flips greedy near-ties at the f32 precision floor (three recorded probe-verified frontiers); no bounded context bucket, performance/RSS gate, portability, or durable current-head refresh exists yet; full-row GPU residency is untested (QAT Q4_0/Q6_K GPU kernels not implemented)",
                metadata_parses: "validated_including_moe_expert_count_router_and_dual_ffn_norms",
                tokenizer_works: "validated_for_gemma4_spm_with_forced_bos_workaround",
                tensors_load: "validated_mmap_wire_backed_q4_0_experts_q6_k_head_layer_range_shards",
                generation_runs: "distributed_two_node_api_chat_completions_and_sse_smoke",
                parity_audited: "basic_v1_pack_2of5_full_budget_token_identical_plus_3of5_probe_verified_knife_edge_frontiers_vs_pinned_comparator_distributed_equals_single_node",
                performance_measured: "two_mac_validation_decode_about_0_17_tok_s_recorded_not_a_perf_claim",
                frontend_load_path_verified: "served_via_gemma4_runtime_flag_distributed_lane",
                frontend_readiness_gate: "green only when this exact gemma4 GGUF row plus Q4_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id (serve requires CAMELID_GEMMA4_SERVE=1 plus CAMELID_GEMMA4_WORKER and CAMELID_GEMMA4_SPLIT pointing at a live gemma4-worker holding the tail layers)",
                tested_context: "short_api_chat_completions_sse_smoke_through_the_two_mac_distributed_lane",
                chat_template_renderer: "gemma4_marker",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "distributed_serve_api_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Paris",
                evidence: "the exact tracked gemma-4-26B_q4_0-it.gguf (14,439,361,440 bytes, general.architecture=gemma4, A4B MoE: 30 layers, 128 experts top-8, dense shared-expert MLP + sparse expert branch, Q4_0 expert/dense weights + Q6_K tied head, parsed from the GGUF) runs as two-Mac distributed layer sharding (master layers 0..15 local + worker 15..30 + head on the second 16GB M4; model staged over Thunderbolt with full-file sha256 verified identical). On the committed basic_v1 pack vs the pinned plain-f32 reference (llama.cpp 5d56eff --no-repack -fa off -ctk f32 -ctv f32 -ub 1): count-primes (24 tok) and translate-de (16 tok) are full-budget token-identical, and capital-france/haiku-sea/rust-fn match a probe-verified prefix then flip at knife-edge near-ties (camelid top-2 gaps 0.138/0.203/0.419, the reference token is camelid's immediate #2 in all three). Distributed output equals single-node camelid (f32 wire). The distributed SERVE lane answers /v1/chat/completions (non-streaming and SSE) and /v1/completions end-to-end across the pair. Evidence: qa/evidence-bundles/gemma4-26b-it-q4-0-qat-distributed-parity-20260611T084039Z-head-b117d40cb7c3 and the distributed-serve smoke bundle qa/evidence-bundles/gemma4-26b-a4b-it-q4-0-distributed-serve-20260611T092520Z-head-6482254fca12. Camelid supports exact-row text-token generation + serve smoke for this row only, and only through the two-Mac distributed lane; no single-node, bounded-context, performance, portability, multimodal, or full support is implied",
                next_step: "durable current-head refresh, bounded context buckets through the distributed lane, performance/RSS gates, and QAT GPU kernels before any wider 26B claim; single-node support is not a goal on 16GB hosts",
            },
            ModelCompatibilityTarget {
                id: "ternary_bonsai_4b_tq2_0",
                tool_capable: false,
                family: "qwen3_ternary_tq2_0_cpu",
                quantization: "TQ2_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_single_node_cpu_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "single-node CPU only; decode is ~0.53x llama.cpp (the general forward-pass gap, not the ternary kernel); greedy parity flips at low-bit near-ties (3/4 probe prompts token-identical, 1 near-tie); prefill recomputes the head per position; no bounded-context bucket, perf/RSS gate, GPU path, chat-template/WebUI closure, or curated-catalog entry yet; the model is community-sourced (superkaiii/Ternary-Bonsai-4B), not a canonical row",
                metadata_parses: "validated_qwen3_arch_with_yarn_rope_scaling",
                tokenizer_works: "validated_qwen3_bpe",
                tensors_load: "validated_streaming_tq2_0_wire_blocks_no_f32_materialisation_plus_q6_k_streamed_tied_head",
                generation_runs: "single_node_cpu_bench_generate_coherent_3_09gb_rss",
                parity_audited: "four_prompt_24_token_greedy_vs_llamacpp_acd79d6_three_token_identical_one_near_tie_divergence_at_index_4",
                performance_measured: "cpu_decode_about_11_3_tok_s_about_0_53x_llamacpp_recorded_not_a_perf_claim",
                frontend_load_path_verified: "not_verified",
                frontend_readiness_gate: "not gated for the frontend; this row is a CPU completion-smoke lane only (no WebUI/serve closure committed)",
                tested_context: "short_single_node_cpu_completion_smoke",
                chat_template_renderer: "qwen3_chatml",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "single_node_cpu_completion_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Paris",
                evidence: "Ternary-Bonsai-4B-TQ2_0.gguf (sha256 b85dcbaa6f57a9c71252371c97f4c68602c2c5fc61a9e1ce74d963d6fee5047c, general.architecture=qwen3, 36 layers, TQ2_0 2.06bpw ternary linears + Q6_K tied embed/lm_head, yarn rope factor 4) runs end-to-end on a single i7-11800H CPU (CUDA hidden), streaming the TQ2_0 wire blocks and the Q6_K head so the 4B model fits in 3.09GB RSS instead of ~16GB f32. Greedy parity vs llama.cpp acd79d6 (llama-server -ngl 0, /completion temp 0 top_k 1): 3/4 probe prompts token-identical for 24 tokens (capital-of-France, once-upon-a-time, quick-brown-fox), 1 diverges at a near-tie (2+2= continuation, camelid '2' vs llama '4' after both emit the correct '= 4,'). Decode 11.34 tok/s = 0.53x llama.cpp (21.25), the general forward gap. Receipt: qa/ternary/tq2_0-bonsai-parity-receipt.json. Camelid supports exact-row CPU completion smoke for this row only; no bounded-context, performance, serve/WebUI, GPU, or full support is implied",
                next_step: "formal bounded-context + serve/WebUI parity closure, a perf/RSS gate, and a curated current-head refresh before any wider ternary claim; an AVX2 Q6_K head and last-position-only prefill would lift throughput further",
            },
            ModelCompatibilityTarget {
                id: "llama3_2_1b_instruct_iq4_xs",
                tool_capable: false,
                family: "llama_bpe_iq4_xs_resident",
                quantization: "IQ4_XS",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_gpu_resident_raw_decode_parity_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "raw-completion greedy parity only (no chat-template/serve/WebUI closure); greedy flips at low-bit near-ties (GPU-resident 2/4 probe prompts 24-token identical, 2/4 match a probe-verified prefix then diverge at a knife-edge near-tie where the reference token is camelid's immediate #2); CPU-deterministic path 1/4 identical; no bounded-context bucket, perf/RSS gate, runnable-smoke oracle qualification, or curated-catalog entry yet; the model is community-sourced (bartowski/Llama-3.2-1B-Instruct-GGUF), not a canonical row",
                metadata_parses: "validated_llama_arch",
                tokenizer_works: "validated_llama3_bpe_128k_bos_128000",
                tensors_load: "validated_streaming_iq4_xs_136b_wire_superblocks_no_f32_materialisation_gpu_resident_and_cpu_plus_kquant_tied_head",
                generation_runs: "gpu_resident_all_16_layers_bench_generate_coherent_2_35gb_vram_and_cpu_streaming_2_05gb",
                parity_audited: "four_prompt_24_token_greedy_vs_llamacpp_acd79d6_gpu_two_token_identical_two_prefix_then_near_tie_at_index_1",
                performance_measured: "gpu_resident_about_47_tok_s_cpu_about_1_tok_s_recorded_not_a_perf_claim",
                frontend_load_path_verified: "not_verified",
                frontend_readiness_gate: "not gated for the frontend; this row is a GPU-resident/CPU completion-smoke lane only (no WebUI/serve closure committed)",
                tested_context: "short_gpu_resident_and_cpu_raw_completion_smoke",
                chat_template_renderer: "not_promoted",
                chat_template_shape_pack: "not_promoted",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_resident_raw_completion_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "Paris",
                evidence: "Llama-3.2-1B-Instruct-IQ4_XS.gguf (sha256 69e85c871a13a99e4008c09916a957d170e4e5623a5b2e06be46408029a7afba, 743,141,504 bytes, general.architecture=llama, IQ4_XS 4.25bpw linears + K-quant tied embed/lm_head) runs end-to-end both as an all-16-layer GPU-resident model on an RTX 4060 (iq4xs_gemv kernel, ~2.35GB VRAM, ~47 tok/s) and as a CPU wire-streamed model (~2.05GB RSS), streaming the 136-byte IQ4_XS super-blocks with no f32 materialisation. Greedy parity vs llama.cpp acd79d6 (llama-server -ngl 0, /completion temp 0 top_k 1 seed 0): GPU-resident 2/4 probe prompts token-identical for 24 tokens (capital-of-France, first-three-primes), 2/4 match a probe-verified prefix then diverge at a knife-edge near-tie at index 1 (e.g. water/hydrogen: reference '.' logprob -1.211 vs camelid ' atoms' -1.449, the reference token is camelid's immediate #2). Prompt tokenization identical on all 4. The GPU kernel matches the CPU oracle within 1e-4 (iq4xs_gemv_matches_oracle). Receipt: qa/iquant/iq4xs-llama3.2-1b-parity-receipt.json. Camelid supports exact-row GPU-resident/CPU raw-completion parity smoke for this row only; no chat-template, bounded-context, performance, serve/WebUI, or full support is implied",
                next_step: "chat-template + serve/WebUI parity closure, bounded-context buckets, a perf/RSS gate, and a runnable-smoke oracle qualification before any wider IQ4_XS claim; extending the resident lane to IQ2/IQ1 i-quants is the next compression bite",
            },
            ModelCompatibilityTarget {
                id: "llama_spm_q4_0_q5_0",
                tool_capable: false,
                family: "llama_spm_decoder",
                quantization: "Q4_0/Q5_0",
                status: "planned_phase_10",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_certified_row",
                full_support_blockers: "row-specific parity, bounded load/readiness, API/WebUI, RSS/timing, and durable bundle evidence are missing; CPU dequant and parity-gated Q4_0 wire GEMV kernels exist as engine facts only and do not promote support",
                metadata_parses: "descriptor_guarded",
                tokenizer_works: "planned_per_model",
                tensors_load: "cpu_f32_dequant_implemented_engine_fact_no_certified_row",
                generation_runs: "unverified_experimental_lane_only",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "not_started",
                frontend_readiness_gate: "fail-closed until an exact supported row plus runtime readiness exist",
                tested_context: "not_started",
                chat_template_renderer: "not_selected",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "not_selected",
                latest_checked_result: "not_started",
                latest_checked_output: "not_applicable",
                evidence: "CPU f32 dequant for Q4_0/Q4_1/Q5_0/Q5_1 is implemented (src/tensor) and parity-gated Q4_0 wire GEMV kernels exist on CUDA/Metal; the Gemma 4 QAT Q4_0 rows are certified separately as their own exact rows. No LLaMA/SPM Q4_0/Q5_0 exact row has parity/bundle evidence, so such files run unverified in the experimental lane only",
                next_step: "certify a first exact LLaMA/SPM Q4_0/Q5_0 row (specific GGUF + SHA + pinned-oracle parity bundle + support-surface update) before any support claim",
            },
            ModelCompatibilityTarget {
                id: "llama_spm_q4_k_q5_k",
                tool_capable: false,
                family: "llama_spm_decoder",
                quantization: "Q4_K_M/Q5_K_M",
                status: "planned_beyond_named_certified_rows",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_certified_row",
                full_support_blockers: "per-row certification is missing for every K-quant file beyond the named certified exact rows (llama_3_2_3b_instruct_q4_k_m, llama_3_2_3b_instruct_q5_k_m, qwen3_4b_q4_k_m, ornith_1_0_9b_q4_k_m, ornith_1_0_9b_q3_k_m — each its own model_compatibility row); loader/dequant and resident K-quant kernels existing are engine facts and do not promote support",
                metadata_parses: "descriptor_guarded",
                tokenizer_works: "planned_per_model",
                tensors_load: "cpu_f32_dequant_and_wire_kernels_implemented_engine_fact_no_family_claim",
                generation_runs: "certified_on_named_exact_rows_only_otherwise_unverified_experimental_lane",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "not_started",
                frontend_readiness_gate: "fail-closed until an exact supported row plus runtime readiness exist",
                tested_context: "not_started",
                chat_template_renderer: "not_selected",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "not_selected",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "not_selected",
                latest_checked_result: "not_started",
                latest_checked_output: "not_applicable",
                evidence: "the named exact K-quant rows are certified with row-specific parity bundles (see their model_compatibility rows); every other K-quant file dequants/loads as an engine fact and runs unverified in the experimental lane only, with no parity or support claim",
                next_step: "certify additional exact K-quant rows one at a time (specific GGUF + SHA + pinned-oracle parity bundle + support-surface update); support never spreads across neighboring files",
            },
            ModelCompatibilityTarget {
                id: "mistral_7b_instruct_v0_3_q8_0",
                tool_capable: false,
                family: "mistral",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context beyond checked packs, broader arbitrary/Jinja templates beyond the row-scoped renderer and template-shape evidence, production throughput beyond bounded perf/RSS evidence, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke_plus_broader_50_token_api_smoke",
                parity_audited: "tokenizer_template_1tok_bounded_broader_50_token_and_gpu_vs_cpu_greedy_parity_pass",
                performance_measured: "bounded_unique_chat_perf_rss_validated",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "tokenizer_template_1tok_bounded_and_checked_512_1024_2048_4096_8192_context_packs",
                chat_template_renderer: "mistral_instruct",
                chat_template_shape_pack: "validated_bounded_pack",
                chat_template_shape_pack_id: "mistral-instruct-v0.3-chat-template-pack-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "mistral-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "mistral-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "mistral-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "mistral-context-4096-max-ladder-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "mistral-context-8192-max-ladder-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "support_promotion_api_webui_smoke",
                latest_checked_result: "pass",
                latest_checked_output: "CMLD-M7B",
                evidence: "exact tokenizer/template, deterministic 1-token/5-token, broader 50-token, and bounded 512/1024/2048/4096/8192 context evidence are green, GPU-vs-CPU greedy continuations match token-for-token on this exact row, and a support-promotion API/WebUI smoke bundle (qa/evidence-bundles/mistral-7b-v0.3-q8-support-promotion-*) records the promoted contract surface",
                next_step: "repeat the current-head promotion smoke on contract-affecting changes; broader/full support still needs separate proof",
            },
            ModelCompatibilityTarget {
                id: "qwen3_0_6b_instruct_q8_0",
                tool_capable: false,
                family: "qwen3",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_chatml_thinking_disabled_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), thinking-mode token-parity, model-native/larger context beyond the short-chat envelope, production throughput, and WebUI smoke on Windows remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke",
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_cpu_reference_and_x86_q8_avx2",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "api_smoke_validated_webui_follow_up",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "chatml_1_5_50_token_short_chat_smoke_plus_bounded_512_2048_4096_8192_context_raw_decode_parity_1024_near_tie_held",
                chat_template_renderer: "qwen3_chatml_thinking_disabled",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen3-chatml-chat-template-pack-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "qwen3-0p6b-context-512-8192-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "qwen3-0p6b-context-512-8192-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "qwen3-0p6b-context-512-8192-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "qwen3-0p6b-context-512-8192-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_resident_bounded_context_512_2048_4096_8192_raw_decode_parity_plus_disclosed_1024_near_tie",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/qwen3-0p6b-q8-context-512-8192-20260708T045658Z-head-8e8c5a1dd1a2/manifest.json",
                evidence: "exact row Qwen3-0.6B-Q8_0.gguf (explicit head_dim path: head_dim 128 != embedding/head_count 64, sourced from attention.key_length): token-AND-text-identical to llama.cpp at 1/5/50 (ChatML, thinking disabled). macOS bundle qa/evidence-bundles/qwen3-0.6b-q8-chatml-parity-20260614T032905Z-head-63bf015; Windows x86_64 CPU bundle qa/evidence-bundles/qwen3-0.6b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 records all_pass on both the cpu_reference scalar path and the x86_q8 AVX2 path vs llama.cpp 9632 (acd79d603). BOUNDED CONTEXT 512/2048/4096/8192 (446/2013/4088/8253 Qwen3 tokens): camelid fully-GPU-resident CUDA decode (resident KV cap 36297 pos) is token-AND-text-identical to pinned llama.cpp acd79d603 at 50 generated tokens on these four buckets — bundle qa/evidence-bundles/qwen3-0p6b-q8-context-512-8192-20260708T045658Z-head-8e8c5a1dd1a2, harness scripts/raw-decode-parity.mjs (single-model runs). This is ENGINE-CORRECTNESS parity (camelid reproduces llama.cpp's greedy output bit-for-bit) — NOT a claim about the 0.6B's text quality: the tiny model degenerates into repetitive/hallucinated output at longer context (e.g. looping 'Answer:/CMLD-<n>'), and camelid faithfully matches that degenerate output. The 1024 bucket is NOT promoted: it diverges at a documented benign ~0.10-nat greedy near-tie (camelid 'The' -1.40 vs oracle 'Answer' -1.50 at gen-index 12) — an isolated flip, since the adjacent longer 2048/4096/8192 buckets ARE token-identical. NOT claimed: the 1024 bucket, 0.6B text quality/coherence, model-native/larger context beyond 8192, other Qwen3 sizes/quants, thinking-mode, or throughput.",
                next_step: "add WebUI smoke and normalized perf evidence on Windows before any broader claim. The 1024 bounded-context near-tie is a benign isolated flip on this degenerate-at-context tiny model, not a blocker",
            },
            ModelCompatibilityTarget {
                id: "qwen3_1_7b_instruct_q8_0",
                tool_capable: false,
                family: "qwen3",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_chatml_thinking_disabled_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), full-trace thinking-mode token-parity, context beyond the 16,384 single-shot / 40,960 KV ceilings, production throughput, and WebUI smoke on Windows remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke_plus_five_prompt_api_smoke",
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_cpu_reference_x86_q8_avx2_and_macos_gpu_resident_15373_token_prefill",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "api_smoke_validated_webui_follow_up",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "chatml_1_5_50_token_short_chat_plus_macos_15373_token_single_shot_prefill_plus_bounded_512_1024_2048_4096_context_raw_decode_parity",
                chat_template_renderer: "qwen3_chatml_thinking_disabled",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen3-chatml-chat-template-pack-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "qwen3-1p7b-context-512-4096-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "qwen3-1p7b-context-512-4096-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "qwen3-1p7b-context-512-4096-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "qwen3-1p7b-context-512-4096-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_resident_bounded_context_512_1024_2048_4096_raw_decode_parity_plus_disclosed_8192_near_tie",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/qwen3-1p7b-q8-context-512-4096-20260708T025128Z-head-bf802df44ecd/manifest.json",
                evidence: "exact row Qwen3-1.7B-Q8_0.gguf: token-AND-text-identical to llama.cpp at 1/5/50 (ChatML, thinking disabled) on macOS/Ubuntu and Windows x86_64 CPU (cpu_reference + x86_q8 AVX2). macOS bundles qa/evidence-bundles/qwen3-1.7b-q8-chatml-parity-20260614T021844Z-head-f41e374 and qwen3-1.7b-q8-gpu-resident-bigctx-parity-20260614T171846Z-head-f97a896 (GPU-resident decode+prefill; 15,373-token single-shot prefill); thinking-mode leading-trace parity qwen3-1.7b-q8-thinking-enabled-parity-*. Windows bundle qa/evidence-bundles/qwen3-1.7b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 (all_pass). BOUNDED CONTEXT 512/1024/2048/4096 (446/968/2013/4088 Qwen3 tokens): camelid fully-GPU-resident CUDA decode (28 layers VRAM-resident; VRAM-sized resident KV cap 25266 pos, so every bucket fits resident) is token-AND-text-identical to pinned llama.cpp acd79d603 at 50 generated tokens on all four buckets (a contiguous strict prefix) — bundle qa/evidence-bundles/qwen3-1p7b-q8-context-512-4096-20260708T025128Z-head-bf802df44ecd, harness scripts/raw-decode-parity.mjs (single-model runs: oracle captured alone then killed). The bundle records the FULL 5-bucket sweep (sweep_all_pass=false): the 8192 bucket (8253 tokens) diverges at a documented benign greedy near-tie where BOTH engines reach the correct hidden code CMLD-8192 but camelid phrases it one token longer (gen-index 16: camelid ' the' vs oracle 'CMLD'); the 4B-Q8_0 sibling is token-identical at 8192, so this is the smaller 1.7B faltering at the longest context, not a camelid decode error — but not token-identical, so 8192 is NOT claimed. NOT claimed: the 8192 bucket, model-native/larger context beyond 4096, other Qwen3 sizes/quants, full-trace thinking-mode token-parity, or throughput.",
                next_step: "resolve or attribute the 8192 near-tie to extend the contiguous ladder past 4096; add WebUI smoke on Windows before widening the platform claim",
            },
            ModelCompatibilityTarget {
                id: "qwen3_4b_instruct_q8_0",
                tool_capable: true,
                family: "qwen3",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_chatml_thinking_disabled_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), thinking-mode token-parity, model-native/larger context beyond the short-chat envelope, production throughput, and WebUI smoke on Windows remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke",
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_confident_probes_cpu_reference_and_x86_q8_avx2",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "api_smoke_validated_webui_follow_up",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "chatml_1_5_50_token_short_chat_smoke_confident_probes_plus_bounded_512_1024_2048_4096_8192_context_raw_decode_parity",
                chat_template_renderer: "qwen3_chatml_thinking_disabled",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen3-chatml-chat-template-pack-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "qwen3-4b-context-512-2048-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "qwen3-4b-context-512-2048-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "validated_third_pack",
                bounded_context_2048_pack_id: "qwen3-4b-context-512-2048-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "validated_fourth_pack",
                bounded_context_4096_pack_id: "qwen3-4b-context-4096-8192-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "validated_fifth_pack",
                bounded_context_8192_pack_id: "qwen3-4b-context-4096-8192-v1",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "bounded_context_4096_8192_raw_decode_parity_gpu_resident_prefix_plus_cpu_fallback_tail",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/qwen3-4b-q8-context-4096-8192-20260707T170525Z-head-ea3a1ce07bd6/manifest.json",
                evidence: "exact row Qwen3-4B-Q8_0.gguf (explicit head_dim path): token-AND-text-identical to llama.cpp at 1/5/50 on confident probes (capital-of-France, say-hello, 2+2). The 'Name a primary color.' probe is a documented macOS first-token near-tie; on the Windows 9632 comparator it also matched (both 'Red'). macOS bundle qa/evidence-bundles/qwen3-4b-q8-chatml-parity-20260614T054617Z-head-368ed9b; Windows bundle qa/evidence-bundles/qwen3-4b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 (all_pass, cpu_reference + x86_q8 AVX2). BOUNDED CONTEXT 512/1024/2048 (446/968/2013 Qwen3 tokens): camelid GPU-resident decode (cuda_resident_q8) is token-AND-text-identical to pinned llama.cpp acd79d603 at 1/5/50 generated tokens on all 3 buckets — bundle qa/evidence-bundles/qwen3-4b-q8-context-512-2048-20260706T233000Z-head-eabe9dac74fa (all_pass=true), harness scripts/raw-decode-parity.mjs. BOUNDED CONTEXT 4096/8192 (4088/8253 Qwen3 tokens): camelid is token-AND-text-identical to pinned llama.cpp acd79d603 at 50 generated tokens on both buckets — bundle qa/evidence-bundles/qwen3-4b-q8-context-4096-8192-20260707T170525Z-head-ea3a1ce07bd6 (all_pass=true), same harness scripts/raw-decode-parity.mjs. On this 6 GiB card (RTX 3060 Laptop) both buckets exceed the 2090-position VRAM-sized resident KV cap (build_resident_cuda_engine: weights 4315 MiB + 512 MiB headroom -> fits 2090 pos), so decode is GPU-resident for positions <2090 and camelid's CPU-fallback path beyond; the token stream is identical either way (a larger card runs these buckets fully resident). The llama.cpp oracle was captured alone and its server killed before camelid started (engines never co-resident, to stay within host RAM). NOT claimed: GPU-resident decode of the 4096/8192 buckets on this 6 GiB card, model-native/larger context beyond 8192, other Qwen3 sizes/quants, or throughput.",
                next_step: "add WebUI smoke on Windows and normalize model-native/larger context beyond the checked 512/1024/2048/4096/8192 packs before any broader claim",
            },
            // Qwen3-4B Q4_K_M (mixed Q4_K + Q6_K) — first Q4_K_M dense row promoted to
            // runtime support-contract recognition. The GPU-resident CUDA decode lane
            // (q4k_gemv + q6k_gemv, wire-only blocks) is token-AND-text-identical to
            // llama.cpp; a default-on CPU K-quant block-dot lane also decodes this file.
            // ID IS FILENAME-ANCHORED (`qwen3_4b_q4_k_m` = normalized Qwen3-4B-Q4_K_M.gguf):
            // this GGUF's general.name carries a cosmetic "Awq" token ("Qwen3 4B Instruct
            // Awq", file_type 15 name inference), so a name+quant id would never match under
            // the frontend exact-row identity matcher; the clean filename identity does and
            // cannot collide with the Q8_0 row (distinct quant token).
            ModelCompatibilityTarget {
                id: "qwen3_4b_q4_k_m",
                tool_capable: true,
                family: "qwen3",
                quantization: "Q4_K_M",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_chatml_thinking_disabled_gpu_resident_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "other Qwen3 sizes/variants/quants, other K-quant files (Q5_K_M / Q4_K_S / etc.), base variants, Qwen3-MoE (A3B), a committed CPU K-quant ChatML full-parity bundle (only the GPU lane is ChatML-proven; the CPU block-dot lane is raw-completion confident-probe parity), thinking-mode token-parity, model-native/larger context beyond the short-chat envelope, production throughput, and WebUI smoke remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated_216_q4_k_plus_37_q6_k_wire_only_fully_gpu_resident_36_of_36_layers",
                generation_runs: "cuda_resident_kquant_decode_plus_default_on_cpu_kquant_block_dot",
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_gpu_resident_cuda_vs_llamacpp_acd79d6_plus_cpu_kquant_block_dot_raw_completion_confident_probe_parity_near_ties_documented",
                performance_measured: "cuda_device_decode_loop_19_44_toks_median_measured_not_a_throughput_claim",
                frontend_load_path_verified: "not_promoted",
                frontend_readiness_gate: "green only when this exact Qwen3-4B-Q4_K_M.gguf row (sha256 7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5) plus Q4_K_M quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id on the CUDA-resident decode lane",
                tested_context: "chatml_1_5_50_token_short_chat_smoke_gpu_resident_plus_bounded_512_1024_context_raw_decode_parity",
                chat_template_renderer: "qwen3_chatml_thinking_disabled",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen3-chatml-chat-template-pack-v1",
                bounded_context_512_pack: "validated_bounded_pack",
                bounded_context_512_pack_id: "qwen3-4b-q4km-context-512-1024-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "validated_second_pack",
                bounded_context_1024_pack_id: "qwen3-4b-q4km-context-512-1024-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "gpu_resident_bounded_context_512_1024_raw_decode_parity_plus_disclosed_2048_near_tie",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/qwen3-4b-q4km-context-512-1024-20260707T234354Z-head-38324e3265e7/manifest.json",
                evidence: "exact row Qwen3-4B-Q4_K_M.gguf (sha256 7485fe6f…, 2.32 GiB): mixed K-quant (216 Q4_K + 37 Q6_K + 145 F32 norms; attn_v / ffn_down / tied token_embd(lm_head) are Q6_K, q/k/o/gate/up are Q4_K, so one run drives BOTH q4k_gemv and q6k_gemv). GPU-resident CUDA decode (36/36 layers VRAM-resident, ~4.92 GB peak) is token-AND-text-identical to pinned llama.cpp acd79d6 at 1/5/50 tokens on all 3 ChatML thinking-disabled prompts (the 'primary color' near-tie PASSES here), with cross-engine prompt-token parity — bundle qa/evidence-bundles/qwen3-4b-q4_k_m-windows-cuda-resident-parity-20260628T003317Z-head-0dccbf74 (all_pass=true), harness scripts/chat-parity-qwen3-kquant.mjs. A default-on CPU K-quant block-dot lane (CAMELID_X86_Q4K_DECODE, Q4_K AVX2 + Q6_K 8-lane scalar, wire-only) also decodes this file: raw-completion confident probes (capital-of-France, fibonacci) token-identical to llama.cpp to depth 50, with a documented benign f32 near-tie on 2+2 — bundle qa/evidence-bundles/qwen3-4b-q4_k_m-windows-cpu-kquant-decode-parity-20260628T015051Z-head-a86fb46b. tool_capable is EARNED via a committed agent-eval PASS receipt (qa/agent-eval/Qwen3-4B-Q4_K_M-1783378260-PASS.json — read_and_count + list_dir_find + write_greeting all pass on this exact GGUF, unlike the Q8_0-sibling Llama-3.2-3B-Q4_K_M which FAILs the same battery). BOUNDED CONTEXT 512/1024 (446/968 Qwen3 tokens): camelid GPU-resident CUDA decode (q4k_gemv/q6k_gemv, all 36 layers VRAM-resident; VRAM-sized resident KV cap 15888 pos, so every bucket fits fully resident) is token-AND-text-identical to pinned llama.cpp acd79d603 at 50 generated tokens on both promoted buckets — bundle qa/evidence-bundles/qwen3-4b-q4km-context-512-1024-20260707T234354Z-head-38324e3265e7 (harness scripts/raw-decode-parity.mjs, single-model runs: oracle captured alone then killed). The bundle records the FULL 5-bucket sweep (sweep_all_pass=false): 4096/8192 (4088/8253 tokens) were ALSO token-identical, but the promoted ladder is held to a contiguous 512/1024 because the intervening 2048 bucket diverges at a documented benign sub-0.1-nat greedy near-tie (camelid ' llama' -1.92 vs oracle ' the' -2.00, gap 0.08 nat; the Q4 oracle degenerates into a repetition loop while camelid recalls correctly — not a camelid decode error, but not token-identical so not claimed). NOT claimed: the 2048 bucket, the 4096/8192 buckets pending a clean 2048, other Qwen3 sizes/quants, other K-quant files, CPU ChatML full parity, model-native/larger context, thinking-mode, or any GPU-vs-GPU / Q8-vs-Q4 throughput claim.",
                next_step: "resolve the 2048 near-tie (or adopt a near-tie-attributed bar) to extend the contiguous ladder past 1024 to the already-token-identical 4096/8192 buckets; add a CPU K-quant ChatML full-parity bundle and WebUI smoke before any broader claim (tool_capable already earned via the committed agent-eval PASS receipt)",
            },
            ModelCompatibilityTarget {
                id: "qwen3_8b_instruct_q8_0",
                tool_capable: false,
                family: "qwen3",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_chatml_thinking_disabled_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "other Qwen3 sizes/variants/quants, base variants, Qwen3-MoE (A3B), thinking-mode token-parity, 8B large-context token-parity, context beyond the 16,384 single-shot / 40,960 KV ceilings, production throughput, and WebUI smoke on Windows remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke",
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_cpu_reference_x86_q8_avx2_and_macos_gpu_resident",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "api_smoke_validated_webui_follow_up",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "chatml_1_5_50_token_short_chat_smoke",
                chat_template_renderer: "qwen3_chatml_thinking_disabled",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen3-chatml-chat-template-pack-v1",
                bounded_context_512_pack: "not_promoted",
                bounded_context_512_pack_id: "not_selected",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_promoted",
                bounded_context_1024_pack_id: "not_selected",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_promoted",
                bounded_context_2048_pack_id: "not_selected",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "windows_x86_64_chatml_parity",
                latest_checked_result: "pass",
                latest_checked_output: "qa/evidence-bundles/qwen3-8b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23/README.md",
                evidence: "exact row Qwen3-8B-Q8_0.gguf (square head_dim, UNTIED embeddings / separate output.weight): token-AND-text-identical to llama.cpp at 1/5/50 (ChatML, thinking disabled). macOS bundles qa/evidence-bundles/qwen3-8b-q8-chatml-parity-20260614T072602Z-head-368ed9b and qwen3-8b-q8-gpu-resident-parity-20260614T213932Z-head-a0ee3d6 (GPU-resident decode+prefill); Windows bundle qa/evidence-bundles/qwen3-8b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 captured via the two-phase oracle flow (all_pass, cpu_reference + x86_q8 AVX2).",
                next_step: "add WebUI smoke and large-context evidence on Windows before any broader claim",
            },
            ModelCompatibilityTarget {
                id: "mixtral_8x7b_instruct_v0_1_q8_0",
                tool_capable: false,
                family: "mixtral_moe",
                quantization: "Q8_0",
                status: "active_validation_partial_runtime",
                support_scope: "exact_row_bounded_moe_runtime_only",
                full_support_status: "blocked_later_generation_divergence",
                full_support_blockers: "later short-prompt generation still diverges from llama.cpp; API/WebUI readiness, long-context evidence, production throughput, portability, and durable broad prompt coverage are missing",
                metadata_parses: "validated_sparse_header",
                tokenizer_works: "validated_against_llama_cpp_reference",
                tensors_load: "validated_lazy_file_backed_rank3_q8_experts",
                generation_runs: "bounded_one_token_runtime_smoke_observed",
                parity_audited: "prompt_tokens_and_bounded_one_token_match_only",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "fail_closed_partial_runtime_only",
                frontend_readiness_gate: "fail-closed for broad readiness: exact row may be described only as bounded one-token backend runtime evidence until later-generation parity and API/WebUI gates close",
                tested_context: "short_prompt_one_token_probe_only",
                chat_template_renderer: "mixtral_instruct_v0_1_metadata_template_validated",
                chat_template_shape_pack: "validated_reference_pack",
                chat_template_shape_pack_id: "mixtral-instruct-v0.1-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "mixtral-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "mixtral-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "mixtral-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "mixtral-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "mixtral_8x7b_q8_gate9a_50tok_divergence_20260511",
                latest_checked_result: "blocked_later_generation_divergence",
                latest_checked_output: "qa/evidence-bundles/mixtral-8x7b-v0.1-q8-blocker-reconciliation-20260512/README.md",
                evidence: "exact row Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf: sparse-header metadata parses with llama.expert_count=8 and expert_used_count=2 plus rank-3 expert tensors; tokenizer/template prompts match llama.cpp reference pack fixtures/tokenizer/mixtral-8x7b-instruct-v0.1-reference-pack.json; MoE top-k expert routing runs with default full-router softmax weights and lazy/file-backed Q8 experts; bounded one-token backend MoE runtime evidence exists, but Gate 9A 50-token evidence diverged at generated token index 9 and qa/evidence-bundles/mixtral-8x7b-v0.1-q8-longgen-continuation-20260511 records partial_failure plus a backend HTTP hang. No broad Mixtral, API/WebUI/frontend readiness, long-context, production, neighboring-row, or full-support claim is made.",
                next_step: "fix later-generation divergence and rerun row-specific parity/API/WebUI/RSS evidence before any Mixtral support/readiness promotion",
            },
            ModelCompatibilityTarget {
                id: "qwen25_7b_instruct_q8_0",
                tool_capable: false,
                family: "qwen_decoder",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "qwen2 runtime, tokenizer/pre-tokenizer fixtures, ChatML parity, bounded load/readiness, API/WebUI, RSS/timing, context, and durable bundle evidence are missing",
                metadata_parses: "acquisition_planned",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until an exact supported row plus runtime readiness exist",
                tested_context: "not_started",
                chat_template_renderer: "qwen25_instruct_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen25-instruct-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "qwen25-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "qwen25-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "qwen25-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "qwen25-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "first Qwen candidate row selected for planning only: Qwen2.5-7B-Instruct-Q8_0.gguf; tokenizer/template semantics, architecture mapping, bounded load, prompt-token parity, API/WebUI, RSS, and bundle evidence are all still required",
                next_step: "capture acquisition path, model SHA and license/access notes, then add tokenizer/chat-template fixtures and independent prompt-token references before any runtime-support wording",
            },
            ModelCompatibilityTarget {
                id: "phi3_mini_4k_instruct_q8_0",
                tool_capable: false,
                family: "phi3",
                quantization: "Q8_0",
                status: "active_validation_blocked_parity",
                support_scope: "exact_row_validation_only",
                full_support_status: "blocked_generation_parity",
                full_support_blockers: "generation parity is not certified: prefill and incremental decode disagree from the 3rd generated token (deterministically, since the Metal PV row-tail fix); API/WebUI readiness, context, performance, and portability evidence are also missing. Prompt-token parity and the temperature-0 non-determinism no longer block",
                metadata_parses: "observed",
                tokenizer_works: "validated_raw_8_of_8_and_chat_3_of_3_prompt_token_parity",
                tensors_load: "observed",
                generation_runs: "observed_but_not_parity_certified",
                parity_audited: "failed",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_blocked_parity",
                frontend_readiness_gate: "fail-closed until generation parity passes for this exact artifact",
                tested_context: "prompt_token_parity_to_2180_tokens_single_token_decode_reference_matched_multi_token_diverges_at_index_2",
                chat_template_renderer: "phi3_metadata_template_prompt_tokens_reference_matched",
                chat_template_shape_pack: "failed_reference_parity",
                chat_template_shape_pack_id: "phi3-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "phi3-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "phi3-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "phi3-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "phi3-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "phi3_hold_evidence",
                latest_checked_result: "blocked_generation_parity",
                latest_checked_output: "qa/muster/phi3-hold-evidence/README.md",
                evidence: "prompt-token parity PASSES on the exact artifact Phi-3-mini-4k-instruct-Q8_0.gguf: all_match=true over the committed 8-prompt pack (to a 2180-token prompt) and over the 3 rendered chat prompts, against pinned llama.cpp acd79d603 — qa/muster/phi3-hold-evidence/{prompt-token-parity-q8-20260727.json,chat-prompt-token-parity-q8-20260727.json}. The temperature-0 NON-DETERMINISM that previously blocked this row is FIXED: half_mm_batched_f16o ignored its `rows` argument, so at head_dim=96 the prefill PV pass's partial 64-row tile read past each head V slab and wrote into the next head's columns, racing it; bounding the three unguarded sites by `rows` makes the default path bit-stable (6/6 identical single-token, 3/3 identical multi-token) and returns the reference token 32001 — qa/muster/phi3-hold-evidence/nondeterminism-fixed-q8-20260728.json, which also records Llama-3.2-3B and Mistral-7B bit-identical pre/post patch. Generation parity STILL blocks the row for a separate and now-DETERMINISTIC reason: on 'The capital of France is' the reference emits [3681,29889,13,32001,3869] while camelid emits [3681,29889,3681,338,2998], diverging at index 2, and a fresh prefill of the 8-token prefix returns the reference 13 where incremental decode returns 3681. This row is intentionally not advertised as supported",
                next_step: "reconcile phi3 incremental decode with prefill — a fresh prefill of the same prefix matches the reference while decode does not, so compare the decode KV-cache read against the prefill path for head_dim=96 (the same non-multiple-of-64 shape that broke the PV row tile). Then re-run generation parity and capture exact-row API/WebUI and bounded-context evidence before promotion",
            },
            ModelCompatibilityTarget {
                id: "phi4_mini_instruct_q4_k_m",
                tool_capable: false,
                family: "phi3",
                quantization: "Q4_K_M",
                status: "active_validation_blocked_parity",
                support_scope: "exact_row_validation_only",
                full_support_status: "blocked_prompt_and_generation_parity",
                full_support_blockers: "the exact artifact has a documented Windows CUDA raw-decode divergence; CPU parity is inconclusive; forward-path isolation, API/WebUI, context, performance, and portability evidence are all missing",
                metadata_parses: "observed",
                tokenizer_works: "observed_exact_artifact_gpt4o_only",
                tensors_load: "observed_cuda_resident_only",
                generation_runs: "observed_but_not_parity_certified",
                parity_audited: "failed",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_blocked_parity",
                frontend_readiness_gate: "fail-closed until prompt-token and generation parity pass for this exact artifact",
                tested_context: "short_raw_prompt_cuda_diverged_cpu_inconclusive",
                chat_template_renderer: "phi4_metadata_template_under_validation",
                chat_template_shape_pack: "ordinary_message_shape_locked_not_generation_certified",
                chat_template_shape_pack_id: "phi4-mini-chat-template-shapes-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "phi4-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "phi4-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "phi4-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "phi4-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "phi4_hold_raw_cuda_divergence",
                latest_checked_result: "blocked_prompt_and_generation_parity",
                latest_checked_output: "qa/muster/HOLD-phi4-mini-instruct-q4_k_m.json",
                evidence: "exact artifact SHA-256 88c00229914083cd112853aab84ed51b87bdf6b9ce42f532d8c85c7c63b1730a is held in qa/muster/HOLD-phi4-mini-instruct-q4_k_m.json after an unattributed Windows CUDA raw-decode divergence and an inconclusive CPU matrix; it is intentionally absent from curated_catalog and not advertised as supported",
                next_step: "isolate the first divergent token across the Phi-4 forward path, then rerun complete tokenizer, generation, API/WebUI, determinism, and bounded-context evidence before any catalog or support promotion",
            },
            ModelCompatibilityTarget {
                id: "deepseek_r1_distill_qwen_7b_q8_0",
                tool_capable: false,
                family: "qwen25",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "the exact artifact has no committed load, tokenizer/template, generation parity, API/WebUI, context, performance, or portability evidence",
                metadata_parses: "not_started",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until exact-artifact runtime and parity evidence exists",
                tested_context: "not_started",
                chat_template_renderer: "qwen25_instruct_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen25-instruct-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "qwen25-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "qwen25-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "qwen25-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "qwen25-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "planning only: a Qwen2-family base architecture does not certify this DeepSeek-distilled exact artifact",
                next_step: "acquire and SHA-anchor the exact artifact, then run prompt-token, generation, API/WebUI, memory, and bounded-context gates",
            },
            ModelCompatibilityTarget {
                id: "c4ai_command_r_v01_q8_0",
                tool_capable: false,
                family: "command_r",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "command-r is still rejected by runnable architecture admission; the new tensor/tokenizer scaffolding is unreachable and has no exact-row load or parity evidence",
                metadata_parses: "header_only_not_runnable",
                tokenizer_works: "scaffolding_unvalidated",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed while command-r runtime admission rejects the architecture",
                tested_context: "not_started",
                chat_template_renderer: "command_r_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "command-r-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "command-r-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "command-r-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "command-r-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "command-r-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "planning only: logit-scale, normalization, and tokenizer scaffolding do not make command-r runnable or supported",
                next_step: "implement and validate command-r admission/binding, then run exact-artifact tokenizer, generation, parity, API/WebUI, memory, and context gates",
            },
            ModelCompatibilityTarget {
                id: "deepseek_r1_distill_llama_8b_q8_0",
                tool_capable: false,
                family: "llama",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "the exact artifact has no committed load, tokenizer/template, generation parity, API/WebUI, context, performance, or portability evidence",
                metadata_parses: "not_started",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until exact-artifact runtime and parity evidence exists",
                tested_context: "not_started",
                chat_template_renderer: "llama3_distill_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "llama3-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "llama3-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "llama3-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "llama3-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "llama3-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "planning only: support for other Llama exact rows does not certify this DeepSeek-distilled artifact",
                next_step: "acquire and SHA-anchor the exact artifact, then run prompt-token, generation, API/WebUI, memory, and bounded-context gates",
            },
            ModelCompatibilityTarget {
                id: "qwen25_coder_7b_q8_0",
                tool_capable: false,
                family: "qwen25",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "the exact artifact has no committed load, tokenizer/template, generation parity, API/WebUI, context, performance, or portability evidence",
                metadata_parses: "not_started",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until exact-artifact runtime and parity evidence exists",
                tested_context: "not_started",
                chat_template_renderer: "qwen25_coder_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "qwen25-instruct-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "qwen25-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "qwen25-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "qwen25-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "qwen25-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "planning only: Qwen2-family runtime work does not certify this coder-tuned exact artifact",
                next_step: "acquire and SHA-anchor the exact artifact, then run prompt-token, generation, API/WebUI, memory, and bounded-context gates",
            },
            ModelCompatibilityTarget {
                id: "llama31_8b_instruct_q8_0",
                tool_capable: false,
                family: "llama",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "the exact artifact has no committed load, tokenizer/template, generation parity, API/WebUI, context, performance, or portability evidence",
                metadata_parses: "not_started",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until exact-artifact runtime and parity evidence exists",
                tested_context: "not_started",
                chat_template_renderer: "llama31_instruct_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "llama3-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "llama3-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "llama3-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "llama3-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "llama3-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "planning only: support for other Llama exact rows does not certify this 3.1 8B artifact",
                next_step: "acquire and SHA-anchor the exact artifact, then run prompt-token, generation, API/WebUI, memory, and bounded-context gates",
            },
            ModelCompatibilityTarget {
                id: "gemma2_9b_it_q8_0",
                tool_capable: false,
                family: "gemma2_decoder",
                quantization: "Q8_0",
                status: "planned_exact_row_candidate",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "gemma2 runtime, control-token/template fixtures, bounded load/readiness, API/WebUI, RSS/timing, context, and durable bundle evidence are missing",
                metadata_parses: "acquisition_planned",
                tokenizer_works: "not_started",
                tensors_load: "not_started",
                generation_runs: "not_started",
                parity_audited: "not_started",
                performance_measured: "not_started",
                frontend_load_path_verified: "fail_closed_planned",
                frontend_readiness_gate: "fail-closed until an exact supported row plus runtime readiness exist",
                tested_context: "not_started",
                chat_template_renderer: "gemma2_it_planned",
                chat_template_shape_pack: "not_started",
                chat_template_shape_pack_id: "gemma2-it-chat-template-pack-v1",
                bounded_context_512_pack: "not_started",
                bounded_context_512_pack_id: "gemma2-context-512-smoke-v1",
                bounded_context_window: 512,
                bounded_context_1024_pack: "not_started",
                bounded_context_1024_pack_id: "gemma2-context-1024-smoke-v1",
                bounded_context_1024_window: 1024,
                bounded_context_2048_pack: "not_started",
                bounded_context_2048_pack_id: "gemma2-context-2048-smoke-v1",
                bounded_context_2048_window: 2048,
                bounded_context_4096_pack: "not_started",
                bounded_context_4096_pack_id: "gemma2-context-4096-smoke-v1",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "candidate_selected",
                latest_checked_result: "planning_only",
                latest_checked_output: "not_applicable",
                evidence: "first Gemma candidate row selected for planning only: gemma-2-9b-it-Q8_0.gguf; Gemma2 architecture details, tokenizer/control-token behavior, template formatting, bounded load, parity, API/WebUI, RSS, and bundle evidence are all still required",
                next_step: "capture acquisition path, model SHA and license/access notes, then add tokenizer/chat-template fixtures and bounded metadata/load checks before any runtime-support wording",
            },
        ],
        api_features: vec![
            SupportItem {
                id: "camelid_verify",
                status: "partial",
                notes: "CLI `camelid verify <gguf>` plus GET/POST /api/models/verify replay one pinned deterministic request for a built-in exact-GGUF-hash profile and emit a digest-sealed report. The initial profile covers only the tracked Llama 3.2 1B Instruct Q8_0 Windows artifact. A pass proves that one request on those exact bytes; it is not a digital signature, broad certification, support promotion, or performance claim.",
            },
            SupportItem {
                id: "openai_chat_completions",
                status: "supported_current_gate",
                notes: "non-streaming and SSE streaming for loaded supported dense GGUF models",
            },
            SupportItem {
                id: "web_workspace",
                status: "supported_current_gate",
                notes: "loopback WebUI only: durable conversations over exactly read_file/list_dir/literal-content search inside one canonical workspace root; allow_writes=true is rejected. Evidence-first extension inventories are derived from successful list_dir observations. Exact prompt-plus-generation budgeting, SQLite/FTS5 retrieval, reversible automatic compaction, turn-scoped cancellation, a 90-second model-step deadline, and model-transition exclusion fail closed. Available only to supported exact rows with tool_capable=true. The historical Qwen3-4B-Q4_K_M bundle validates the underlying tool-call and sandbox path, but its write scenarios no longer describe the current read-only surface. No write, shell, network, GUI, subagent, unattended, neighboring-model, portability, or throughput claim.",
            },
            SupportItem {
                id: "stream_options.include_usage",
                status: "supported_current_gate",
                notes: "chat-completions streaming only: stream_options.include_usage:true appends one terminal chunk with choices:[] and a usage object {prompt_tokens, completion_tokens, total_tokens} identical to the non-streaming endpoint's counts, then [DONE]. Omitting it is byte-identical to the prior baseline. Malformed/other stream_options shapes and subfields are tolerated and ignored (no error), matching the llama-server acd79d6 oracle; no other stream_options subfield is supported. Evidence: qa/evidence-bundles/stream-options-include-usage-20260623/.",
            },
            SupportItem {
                id: "tokenizer_encode_decode",
                status: "supported_current_gate",
                notes: "loaded-model tokenizer APIs for supported tokenizer families",
            },
            SupportItem {
                id: "llama_server_tokenizer_aliases",
                status: "partial",
                notes: "POST /tokenize and POST /detokenize are bounded loaded-model tokenizer aliases that return token ids/text, with /tokenize with_pieces=true exposing id/piece objects for supported tokenizer lanes. Arbitrary tokenizer kwargs and broader tokenizer parity remain unsupported.",
            },
            SupportItem {
                id: "llama_server_models",
                status: "partial",
                notes: "GET /models returns a privacy-safe read-only list of currently loaded Camelid models with redacted paths and text-only architecture metadata. POST /models/load is a narrow local-path alias over Camelid's stable /api/models/load path and returns a redacted compatibility response. Router-mode query params such as reload/autoload/model selection, cache listing, POST /models/unload, multimodal metadata, and full llama-server model-management parity remain unsupported.",
            },
            SupportItem {
                id: "llama_server_props",
                status: "partial",
                notes: "GET /props returns read-only public server properties, default generation settings, explicit fail-closed chat_template_caps, chat-template metadata when a model is loaded, and Camelid readiness notes. Local model paths are intentionally redacted, router-mode model/autoload query params and POST /props are unsupported, and this does not imply slot lifecycle, native /completion streaming, embeddings, or full llama-server WebUI parity.",
            },
            SupportItem {
                id: "llama_server_slots",
                status: "partial",
                notes: "GET /slots returns a single read-only, privacy-safe slot snapshot with generation readiness and fail_on_no_slot=1 handling. Router-mode model/autoload query params, POST /slots, slot save/restore/erase actions, prompt-cache metadata, cancellation metadata, and continuous batching metrics remain unsupported.",
            },
            SupportItem {
                id: "llama_server_apply_template",
                status: "partial",
                notes: "POST /apply-template renders loaded-model chat messages to a prompt string without inference. It is scoped to Camelid's supported tokenizer/template renderers and returns typed unsupported errors for unknown request fields or unsupported templates.",
            },
            SupportItem {
                id: "llama_server_completion",
                status: "partial",
                notes: "POST /completion accepts a narrow non-streaming text-generation subset: text prompts, token-id prompt arrays, n_predict/max_tokens, supported sampler fields, and stop sequences are mapped onto Camelid's existing generation path. Native stream=true chunks, slot selection, cache_prompt controls, llama-server timings shape, rich token probabilities, and full llama-server generation parity remain unsupported.",
            },
            SupportItem {
                id: "fail_closed_native_compatibility_routes",
                status: "unsupported",
                notes: "Native /infill, /metrics, /embedding, /embeddings, /v1/embeddings, /v1/messages, /rerank, /reranking, /v1/rerank, /v1/reranking, /v1/responses, POST /models/unload, POST /slots, and slot cache actions return typed not_implemented errors until real route semantics and backend support exist. Unsupported /models/load router-mode fields and /completion modes remain typed parameter errors.",
            },
            SupportItem {
                id: "multi_choice_generation",
                status: "unsupported",
                notes: "typed unsupported until implemented and tested",
            },
            SupportItem {
                id: "rich_logprobs",
                status: "partial",
                notes: "diagnostic logit surfaces exist; full OpenAI-compatible logprobs remain planned",
            },
        ],
        notes: vec![
            "GGUF metadata, tokenizer metadata, tensor loading, Camelid dense config extraction, and tensor binding are available",
            "public completion endpoints can generate small OpenAI-compatible non-streaming responses and SSE token streams from a loaded Camelid-supported dense GGUF model",
            "capability fields are intentionally explicit so the frontend and providers do not infer unsupported model families or quantization formats",
        ],
    }
}

/// The largest curated row that has a positive fit on `hw`, as a ready-to-run
/// suggestion string. `None` when nothing in the catalog fits (or the host is
/// unprobed). Used to make the pre-load "too big" error actionable.
fn best_fitting_catalog_suggestion(hw: &crate::capability::HardwareProfile) -> Option<String> {
    curated_catalog()
        .into_iter()
        .filter(|item| {
            crate::fit::assess(hw, &crate::fit::advisory_footprint(item.size_bytes))
                .is_positive_fit()
        })
        .max_by_key(|item| item.size_bytes)
        .map(|item| format!("{} (`camelid pull {}`)", item.name, item.catalog_id))
}

/// Pure advisory message for a pre-load fit check: `Some(message)` when `hw` won't
/// fit `footprint`, else `None`. `size_bytes` is the on-disk size, used only for the
/// human-readable number. Split from the IO wrapper so it is unit-testable with a
/// synthetic host and footprint.
/// What releasing the currently-resident models would give back. `bytes` is an
/// ESTIMATE (the sum of their weight files) used only to word the error — never
/// to decide a verdict. The `replace` path does not trust it: it releases the
/// models for real and then re-probes live memory, so a wrong estimate can never
/// turn into a false "fits" and an OOM.
#[derive(Debug, Clone, Default)]
pub struct ResidentReclaim {
    pub ids: Vec<String>,
    pub bytes: u64,
}

impl ResidentReclaim {
    fn is_empty(&self) -> bool {
        self.ids.is_empty() || self.bytes == 0
    }
}

/// The refusal for a load-blocking verdict: a stable error code plus its message.
///
/// A host that is out of room *because something else is already loaded* is a
/// different situation from a host that is too small for the model, and it has a
/// different remedy (release the other model vs. pick a smaller one). Reporting
/// both as `model_too_large_for_host` is what made the advisor dead-end: it told
/// the operator the machine could not hold a model the machine plainly could.
///
/// There is a third case with a third remedy: nothing of ours is resident, the
/// machine is big enough, but *other applications* have the memory right now
/// (`FitVerdict::InsufficientFreeMemory`). Closing something is the fix, and saying
/// "larger than this machine can hold" there is simply false.
fn fit_preload_message(
    hw: &crate::capability::HardwareProfile,
    footprint: &crate::fit::FitInputs,
    size_bytes: u64,
    reclaim: Option<&ResidentReclaim>,
) -> Option<(&'static str, String)> {
    let verdict = crate::fit::assess(hw, footprint);
    if !verdict.refuses_load() {
        return None;
    }
    let size_gb = size_bytes as f64 / 1e9;
    // Something else of ours is resident: name it, and point at the remedy that
    // works. Checked first because it is the most specific and most actionable.
    if let Some(r) = reclaim.filter(|r| !r.is_empty()) {
        let names = r.ids.join(", ");
        let freed = r.bytes as f64 / 1e9;
        return Some((
            "model_requires_unload",
            format!(
                "This model (~{size_gb:.1} GB) does not fit while {names} is still loaded. \
                 Releasing it frees ~{freed:.1} GB, which should be enough. \
                 Retry with \"replace\": true to swap models in one step (the app's \
                 Load button does this), or POST /api/models/unload first."
            ),
        ));
    }
    // The machine is big enough; it is just busy. Do not send the user shopping
    // for smaller models — tell them what is actually in the way.
    if verdict == crate::fit::FitVerdict::InsufficientFreeMemory {
        let free_gb = hw.host_ram_free_bytes as f64 / 1e9;
        let total_gb = hw.host_ram_total_bytes as f64 / 1e9;
        return Some((
            "host_memory_unavailable",
            format!(
                "This model (~{size_gb:.1} GB) fits this machine, but only ~{free_gb:.1} GB \
                 of ~{total_gb:.1} GB memory is free right now. Close some applications \
                 and retry, or set CAMELID_SKIP_FIT_CHECK=1 to attempt the load anyway."
            ),
        ));
    }
    let base =
        format!("This model (~{size_gb:.1} GB) is larger than this machine can hold in memory.");
    Some((
        "model_too_large_for_host",
        match best_fitting_catalog_suggestion(hw) {
            Some(alt) => format!(
                "{base} The largest catalog model that fits here is {alt}. \
                 Set CAMELID_SKIP_FIT_CHECK=1 to attempt the load anyway."
            ),
            None => format!("{base} Set CAMELID_SKIP_FIT_CHECK=1 to attempt the load anyway."),
        },
    ))
}

/// Env + filesystem wrapper around [`fit_preload_message`]. Probes **live** host
/// memory and computes an **exact** footprint from the GGUF's real dimensions
/// (weights + KV at a normal-use context + a bounded scratch margin) whenever the
/// header parses, falling back to the coarse size pad otherwise. Returns a typed
/// 422 only on a load-refusing verdict ([`crate::fit::FitVerdict::refuses_load`]);
/// `None` (proceed unchanged) on the `CAMELID_SKIP_FIT_CHECK=1` override, a
/// missing/zero-size file, or any `Fits*`/`Unknown` verdict — a fail-fast
/// convenience, never a new hard gate.
/// Whether the load-time fit preflight is overridden off. `CAMELID_SKIP_FIT_CHECK=1`
/// restores the pre-advisor behavior: `POST /api/models/load` attempts the load
/// unconditionally, letting the authoritative `VramShortfall`/`KvCache` guards be
/// the only gate. Exactly the trimmed value `"1"` enables the override; anything
/// else (including unset) keeps the preflight on.
fn fit_check_skipped(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

fn fit_preload_guard(
    path: &std::path::Path,
    reclaim: Option<&ResidentReclaim>,
) -> Option<Response> {
    if fit_check_skipped(std::env::var("CAMELID_SKIP_FIT_CHECK").ok().as_deref()) {
        return None;
    }
    let size = std::fs::metadata(path).ok()?.len();
    if size == 0 {
        return None;
    }
    // Live probe (not the cached startup snapshot): free VRAM/RAM shift as other
    // apps run or a model is already loaded, and this decision must reflect *now*.
    let hw = crate::capability::HardwareProfile::detect();
    // Exact footprint from the GGUF's real dims when the header parses; else the pad.
    let footprint = match crate::fit_dims::dims_from_gguf_file(path) {
        Some(dims) => {
            // KV is stored f16 on the GPU-resident path, f32 on the CPU path.
            let kv_dtype = if hw.cuda_available && hw.cuda_vram_free_bytes > 0 {
                crate::fit::KvDtype::F16
            } else {
                crate::fit::KvDtype::F32
            };
            crate::fit::exact_footprint(size, dims, crate::fit::ADVISORY_CONTEXT_TOKENS, kv_dtype)
        }
        None => crate::fit::advisory_footprint(size),
    };
    let (code, message) = fit_preload_message(&hw, &footprint, size, reclaim)?;
    Some(api_error(
        StatusCode::UNPROCESSABLE_ENTITY,
        code,
        message,
        Some("path"),
    ))
}

async fn load_model(State(state): State<AppState>, Json(req): Json<LoadModelRequest>) -> Response {
    // Resolve relative request paths against the configured models dir BEFORE
    // the fit guard, so the guard sizes the file that will actually be loaded.
    let path = match resolve_request_model_path(&state.models_dir, req.path) {
        Ok(path) => path,
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                backend_error_code(&err),
                err.to_string(),
                Some("path"),
            )
        }
    };
    // What is resident BESIDES the model being requested. A second copy of the
    // same file is not reclaimable room (the load would be idempotent), so it is
    // excluded — otherwise a repeat load would offer to release itself.
    let mut reclaim = ResidentReclaim::default();
    for (id, loaded) in state.loaded_models.read().await.iter() {
        if loaded.path == path {
            continue;
        }
        reclaim.bytes += std::fs::metadata(&loaded.path)
            .map(|m| m.len())
            .unwrap_or(0);
        reclaim.ids.push(id.clone());
    }
    reclaim.ids.sort();

    // "Switch model" semantics. Release the other residents FIRST, then let the
    // preflight below re-probe. The verdict is therefore reclaim-aware without
    // ever estimating reclaimable bytes: the memory is genuinely back before the
    // decision is made, so this cannot manufacture a false "fits" (this host has
    // OOM'd under memory pressure, so the guard must never be optimistic).
    if req.replace && !reclaim.ids.is_empty() {
        if state.workspace_sessions.blocks_model_transition().await {
            return api_error(
                StatusCode::CONFLICT,
                "model_operation_in_progress",
                BackendError::ModelOperationInProgress.to_string(),
                None,
            );
        }
        let ids = reclaim.ids.clone();
        {
            let _transition = state.model_transition.lock().await;
            let _exclusive = state.model_file_lifecycle.write().await;
            for id in &ids {
                if let Err(resp) = release_model(&state, Some(id.clone())).await {
                    return *resp;
                }
            }
        }
        tracing::info!(
            released = %ids.join(", "),
            "replace: released resident model(s) to make room for the requested load"
        );
        // The room is real now, so there is nothing left to suggest releasing.
        reclaim = ResidentReclaim::default();
    }

    // Advisory fail-fast (fit axis, never a support claim): steer away from a
    // near-certain OOM before the expensive load. Overridable via
    // CAMELID_SKIP_FIT_CHECK=1; only fires on a WontFit verdict from a probed host.
    //
    // The guard does blocking I/O (metadata + GGUF header read) and a live hardware
    // probe (HardwareProfile::detect initializes a CUDA context on GPU hosts), so run
    // it on a blocking thread rather than stalling the async worker — consistent with
    // how the header fetches use spawn_blocking. A panic in the probe is non-fatal: we
    // fall through to the load, where VramShortfall/KvCache remain the hard net.
    let guard_path = path.clone();
    {
        let _reader = state.model_file_lifecycle.read().await;
        if let Ok(Some(resp)) =
            tokio::task::spawn_blocking(move || fit_preload_guard(&guard_path, Some(&reclaim)))
                .await
        {
            return resp;
        }
    }
    match load_model_from_path(&state, path, req.id).await {
        Ok(loaded) => (StatusCode::OK, Json(loaded)).into_response(),
        // Fail closed with the exact typed reason and a stable, switchable code.
        // The message already carries the offending architecture/quant and any
        // dedicated-lane redirect (e.g. `camelid diffusion-gemma-chat`).
        Err(err) => {
            let status = if matches!(err, BackendError::ModelOperationInProgress) {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_REQUEST
            };
            api_error(
                status,
                backend_error_code(&err),
                err.to_string(),
                Some("path"),
            )
        }
    }
}

async fn load_model_from_path(
    state: &AppState,
    path: PathBuf,
    id: Option<String>,
) -> Result<LoadedModel, BackendError> {
    load_model_from_path_with_activation(state, path, id, true).await
}

#[derive(Debug, Deserialize)]
struct InspectModelRequest {
    path: PathBuf,
}

#[derive(Debug, Serialize)]
struct InspectBlocker {
    /// Stable, frontend-switchable code (same vocabulary as `error.code`).
    code: &'static str,
    /// Exact typed reason, including the offending architecture and any dedicated-
    /// lane redirect (e.g. `camelid diffusion-gemma-chat`).
    message: String,
}

#[derive(Debug, Serialize)]
struct InspectModelResponse {
    architecture: Option<String>,
    quant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<ModelSourceInspection>,
    /// Predicted lane (`supported` / `experimental_implemented` / `unsupported`).
    lane_class: ModelLaneClass,
    /// The exact typed blocker the load would hit â€” predicted WITHOUT binding
    /// tensors or loading weights. `None` when the architecture is implemented
    /// (it would load and run, supported or experimental).
    #[serde(skip_serializing_if = "Option::is_none")]
    blocker: Option<InspectBlocker>,
}

/// `POST /api/models/inspect` â€” source-level readiness inspection. GGUF files keep
/// the existing header-only lane prediction. Hugging Face SafeTensors directories
/// return descriptor/readiness facts only and never become generation-ready here.
async fn inspect_model(
    State(state): State<AppState>,
    Json(req): Json<InspectModelRequest>,
) -> Response {
    let _reader = state.model_file_lifecycle.read().await;
    // Same request-path resolution as the load endpoints, so inspect and load
    // agree about which file a relative path names.
    let req = match resolve_request_model_path(&state.models_dir, req.path) {
        Ok(path) => InspectModelRequest { path },
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                backend_error_code(&err),
                err.to_string(),
                Some("path"),
            )
        }
    };
    match inspect_model_source(&req.path) {
        Ok(source) if source.manifest.kind == ModelSourceKind::HuggingFaceSafeTensors => {
            return (
                StatusCode::OK,
                Json(InspectModelResponse {
                    architecture: None,
                    quant: None,
                    source: Some(source),
                    lane_class: ModelLaneClass::Unsupported,
                    blocker: Some(InspectBlocker {
                        code: "safetensors_generation_disabled",
                        message: "Hugging Face SafeTensors directory inspection is readiness-only; generation remains disabled until tokenizer parity, tensor orientation, dtype decode, and one-token dense execution fixtures pass".to_string(),
                    }),
                }),
            )
                .into_response();
        }
        Ok(_) => {}
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                backend_error_code(&err),
                err.to_string(),
                Some("path"),
            );
        }
    }

    let path = req.path.clone();
    let parsed = tokio::task::spawn_blocking(move || read_metadata(&path)).await;
    let gguf = match parsed {
        Ok(Ok(gguf)) => gguf,
        Ok(Err(err)) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                backend_error_code(&err),
                err.to_string(),
                Some("path"),
            );
        }
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inspect_task_failed",
                "metadata inspection task panicked".to_string(),
                None,
            );
        }
    };

    let architecture = gguf.architecture().map(ToOwned::to_owned);
    let filename = req
        .path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default()
        .to_string();
    let quant = Some(crate::runnable::headline_quant_of(&gguf)).filter(|q| !q.is_empty());

    // Parse the config header (no tensor bind, no weight load). Ok â‡’ it would load;
    // Err â‡’ it would fail closed with this exact typed reason.
    let (lane_class, blocker) = match LlamaModelConfig::from_gguf(&gguf) {
        Ok(_) => (
            classify_model_lane(architecture.as_deref(), &filename),
            None,
        ),
        Err(err) => (
            ModelLaneClass::Unsupported,
            Some(InspectBlocker {
                code: backend_error_code(&err),
                message: err.to_string(),
            }),
        ),
    };

    (
        StatusCode::OK,
        Json(InspectModelResponse {
            architecture,
            quant,
            source: None,
            lane_class,
            blocker,
        }),
    )
        .into_response()
}

/// The Gemma 4 serve path is gated behind `CAMELID_GEMMA4_SERVE` (1/true/yes).
/// When off, the existing Llama/3B backend behaves exactly as before.
fn gemma4_serve_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_GEMMA4_SERVE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

/// Additionally route the gemma4 serve lane through the CUDA decode engine when
/// `CAMELID_GEMMA4_CUDA` is set (and the build has the `cuda` feature). Off by
/// default; with it off the gemma4 serve lane stays the CPU runtime, unchanged.
#[cfg(feature = "cuda")]
fn gemma4_cuda_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_GEMMA4_CUDA").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

/// Model family from the GGUF `general.architecture`.
fn model_family(gguf: &GgufFile) -> &'static str {
    match gguf.architecture() {
        Some("gemma4") => "gemma4",
        Some("llama" | "mistral" | "qwen2" | "qwen3" | "smollm3" | "gemma3" | "phi3" | "lfm2") => {
            "llama-family"
        }
        Some(_) => "other",
        None => "unknown",
    }
}

/// Gemma 4 chat template constants + renderer. Turns are
/// `<|turn>{system|user|model}\nâ€¦<turn|>\n` and generation follows a trailing
/// `<|turn>model\n`; a leading system message gets its own system turn, and
/// thinking mode injects `<|think|>` (see `gemma4_chat_prompt`). The renderer
/// is locked byte-for-byte to the reference rendering by
/// `qa/gemma4/template_shapes_v1.json`.
/// Gemma 4 turn markers. Gemma 4 RENAMED them from Gemma 3's
/// `<start_of_turn>`/`<end_of_turn>` to `<|turn>` (id 105) / `<turn|>` (id 106)
/// â€” verified against the E2B/E4B/12B GGUF vocab and the GGUF-embedded Jinja
/// chat template (`'<|turn>' + role + '\n'` â€¦ `'<turn|>\n'`). Using the old
/// spellings tokenizes as PLAIN TEXT: the model mimics them back and the stop
/// token never matches.
pub(crate) const GEMMA4_TURN_START: &str = "<|turn>";
pub(crate) const GEMMA4_TURN_END: &str = "<turn|>";
/// Thinking-channel markers (ids 100/101): the model may wrap hidden reasoning
/// in `<|channel>â€¦<channel|>`. The GGUF template strips these spans from chat
/// history; Camelid strips them from chat OUTPUT so hidden reasoning never
/// leaks to the client.
pub(crate) const GEMMA4_CHANNEL_START: &str = "<|channel>";
pub(crate) const GEMMA4_CHANNEL_END: &str = "<channel|>";
/// Thinking-mode token (id 98), injected at the top of the system turn when
/// the reference template runs with `enable_thinking` (its `--jinja` default).
pub(crate) const GEMMA4_THINK: &str = "<|think|>";

#[cfg(test)]
mod gemma4_template_tests {
    use super::*;
    use crate::tokenizer::{
        BpePreTokenizer, BpeRegistry, SpecialTokens, TokenizerConfig, TokenizerModel,
    };

    fn phi4_template_test_tokenizer(template: &str) -> Tokenizer {
        Tokenizer {
            model: TokenizerModel::Gpt2Bpe,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens::default(),
            config: TokenizerConfig {
                add_bos: false,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: Some(template.to_string()),
        }
    }

    #[test]
    fn chat_prompt_uses_gemma4_turn_markers() {
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let prompt = gemma4_chat_prompt(&messages, false);
        assert_eq!(prompt, "<|turn>user\nhi<turn|>\n<|turn>model\n");
        // Gemma 3 marker spellings tokenize as plain text on gemma4 vocabs and
        // must never appear.
        assert!(!prompt.contains("<start_of_turn>"));
        assert!(!prompt.contains("<end_of_turn>"));
    }

    #[test]
    fn qwen3_chatml_prompt_renders_thinking_disabled_generation_prompt() {
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "What is the capital of France?".to_string(),
        }];
        let prompt = render_qwen3_chatml_prompt(&messages, false);
        assert_eq!(
            prompt,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn qwen2_chatml_prompt_matches_default_system_template_shape() {
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "Reply with exactly: hello".to_string(),
        }];

        assert_eq!(
            render_qwen2_chatml_prompt(&messages),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\nReply with exactly: hello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let qwen2_template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. \
                              You are a helpful assistant.<|im_end|>";
        assert!(is_qwen2_chatml_template(qwen2_template));
        assert!(!is_qwen2_chatml_template(
            "<|im_start|>user\n{{ content }}<|im_end|>"
        ));
    }

    #[test]
    fn phi3_prompt_renders_end_marked_turns_and_generation_prompt() {
        let messages = [
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "system".to_string(),
                content: "Be concise.".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "Capital of France?".to_string(),
            },
        ];
        // Oracle-locked semantics (MUSTER M-A2): system renders as its own
        // <|user|> turn, each user/system turn eagerly followed by the
        // <|assistant|> header — matching the pinned oracle's /apply-template
        // output (see phi3-chat-template-shapes-v1.json), NOT the old
        // <|system|>-headed form this test previously asserted.
        assert_eq!(
            render_phi3_prompt(&messages),
            "<|user|>\nBe concise.<|end|>\n<|assistant|>\n<|user|>\nCapital of France?<|end|>\n<|assistant|>\n"
        );
        // Phi-3's <|end|>-separated template must be detected before TinyLlama's.
        let phi3_tmpl = "<|user|>\n{{content}}<|end|>\n<|assistant|>\n";
        assert!(is_phi3_template(phi3_tmpl));
        assert!(!is_phi3_template(
            "<|user|>\n{{content}}</s>\n<|assistant|>\n"
        ));
    }

    #[test]
    fn phi4_compact_prompt_preserves_roles_without_marker_adjacent_newlines() {
        #[derive(serde::Deserialize)]
        struct PackMessage {
            role: String,
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct Shape {
            id: String,
            messages: Vec<PackMessage>,
            expected_prompt: String,
        }
        #[derive(serde::Deserialize)]
        struct Oracle {
            revision: String,
            template: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            pack_id: String,
            oracle: Oracle,
            shapes: Vec<Shape>,
        }

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/phi4-mini-chat-template-shapes-v1.json"
        );
        let raw = std::fs::read_to_string(path).expect("read Phi-4 template shapes pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse Phi-4 template shapes pack");
        assert_eq!(pack.pack_id, "phi4-mini-chat-template-shapes-v1");
        assert_eq!(
            pack.oracle.revision,
            "78eb92a46fc37e6b524df991ed9aca9bc6aa7b80"
        );
        assert!(is_phi4_compact_template(&pack.oracle.template));
        assert!(!is_phi3_template(&pack.oracle.template));

        let phi3_template = "<|user|>\n{{content}}<|end|>\n<|assistant|>\n";
        assert!(is_phi3_template(phi3_template));
        assert!(!is_phi4_compact_template(phi3_template));

        for shape in pack.shapes {
            let messages = shape
                .messages
                .into_iter()
                .map(|message| ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: message.role,
                    content: message.content,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                render_phi4_compact_prompt(&messages),
                shape.expected_prompt,
                "shape {} diverges from the real Phi-4 ordinary-message template",
                shape.id
            );

            let tokenizer = phi4_template_test_tokenizer(&pack.oracle.template);
            let routed = render_chat_prompt_for_tokenization(&messages, &tokenizer);
            assert_eq!(routed.text, shape.expected_prompt);
            assert!(routed.add_special);
            assert!(routed.parse_special);
        }
    }

    #[test]
    fn phi4_tool_requests_fail_closed_instead_of_dropping_tools() {
        let template = "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";
        let tokenizer = phi4_template_test_tokenizer(template);
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "Read notes.txt".to_string(),
        }];
        let tools = [serde_json::json!({
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}}
        })];

        let err = render_chat_prompt_for_tokenization_with_tools(&messages, &tokenizer, &tools)
            .expect_err("Phi-4 tools must not silently fall through to the ordinary renderer");
        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(err
            .to_string()
            .contains("Phi-4 tool-call template rendering is not validated"));
    }

    #[test]
    fn qwen3_chatml_prompt_renders_thinking_enabled_generation_prompt() {
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "What is the capital of France?".to_string(),
        }];
        let prompt = render_qwen3_chatml_prompt(&messages, true);
        // Thinking enabled: bare assistant turn, no pre-filled <think></think>
        // block â€” the model emits its own reasoning (the template default branch).
        assert_eq!(
            prompt,
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen3_chatml_template_detected_and_no_generation_prompt_after_assistant_turn() {
        assert!(is_qwen3_chatml_template(
            "{%- if ... %}<|im_start|>...<|im_end|>..."
        ));
        assert!(!is_qwen3_chatml_template(
            "<|start_header_id|>...<|eot_id|>"
        ));
        // A trailing assistant turn must NOT get an extra generation prompt.
        let messages = [
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hi".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "assistant".to_string(),
                content: "hello".to_string(),
            },
        ];
        let prompt = render_qwen3_chatml_prompt(&messages, false);
        assert_eq!(
            prompt,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
        );
    }

    /// Byte-lock the Qwen3 ChatML renderer against
    /// `qa/prompt-packs/qwen3-chatml-thinking-template-pack-v1.json` for every
    /// shape in both modes. The whole point of the pack is the
    /// `enable_thinking=true` rows: a bare `<|im_start|>assistant\n` generation
    /// turn (no pre-filled `<think></think>` block) is the one strong, bit-exact
    /// guarantee the opt-in thinking lane carries. The parity-locked exact-row
    /// mode stays thinking-DISABLED; this test does not touch that claim.
    #[test]
    fn completions_fail_closed_for_runnable_serve_archs() {
        // Runnable-served archs have no raw-completions bridge: the gate must
        // reject them (falling through would reach the optimized engine, which
        // mis-binds gemma3), and must NOT touch the optimized-lane archs.
        assert!(completions_unsupported_for_arch("qwen35"));
        assert!(completions_unsupported_for_arch("gemma2"));
        assert!(completions_unsupported_for_arch("gemma3"));
        assert!(!completions_unsupported_for_arch("llama"));
        assert!(!completions_unsupported_for_arch("qwen3"));
        assert!(!completions_unsupported_for_arch("mistral"));
        assert!(!completions_unsupported_for_arch("gemma4"));
        assert!(!completions_unsupported_for_arch(""));
    }

    #[test]
    fn phi3_chat_template_pack_locks_renderer() {
        #[derive(serde::Deserialize)]
        struct Shape {
            id: String,
            messages: Vec<PackMessage>,
            expected_prompt: String,
        }
        #[derive(serde::Deserialize)]
        struct PackMessage {
            role: String,
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            pack_id: String,
            oracle: String,
            shapes: Vec<Shape>,
        }

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/phi3-chat-template-shapes-v1.json"
        );
        let raw = std::fs::read_to_string(path).expect("read phi3 template shapes pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse phi3 template shapes pack");

        assert_eq!(pack.pack_id, "phi3-chat-template-shapes-v1");
        assert!(pack.oracle.contains("acd79d603"));
        assert!(
            pack.shapes.len() >= 6,
            "pack must carry the captured shapes"
        );

        for shape in &pack.shapes {
            let messages: Vec<ChatMessage> = shape
                .messages
                .iter()
                .map(|m| ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            let rendered = render_phi3_prompt(&messages);
            assert_eq!(
                rendered, shape.expected_prompt,
                "shape {} diverges from the pinned-oracle /apply-template rendering",
                shape.id
            );
        }
    }

    #[test]
    fn gemma3_chat_template_pack_locks_renderer() {
        #[derive(serde::Deserialize)]
        struct Shape {
            id: String,
            messages: Vec<PackMessage>,
            expected_prompt: String,
        }
        #[derive(serde::Deserialize)]
        struct PackMessage {
            role: String,
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            pack_id: String,
            oracle: String,
            shapes: Vec<Shape>,
        }

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/gemma3-chat-template-shapes-v1.json"
        );
        let raw = std::fs::read_to_string(path).expect("read gemma3 template shapes pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse gemma3 template shapes pack");

        assert_eq!(pack.pack_id, "gemma3-chat-template-shapes-v1");
        // The expected strings are pinned-oracle truth (llama-server --jinja
        // /apply-template at the standing pin), captured before any parity run.
        assert!(pack.oracle.contains("acd79d603"));
        assert!(
            pack.shapes.len() >= 6,
            "pack must carry the captured shapes"
        );

        for shape in &pack.shapes {
            let messages: Vec<ChatMessage> = shape
                .messages
                .iter()
                .map(|m| ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            let rendered = render_gemma3_prompt(&messages);
            assert_eq!(
                rendered, shape.expected_prompt,
                "shape {} diverges from the pinned-oracle /apply-template rendering",
                shape.id
            );
        }
    }

    #[test]
    fn qwen3_chatml_thinking_template_pack_locks_renderer() {
        #[derive(serde::Deserialize)]
        struct Shape {
            id: String,
            enable_thinking: bool,
            messages: Vec<PackMessage>,
            rendered_prompt: String,
        }
        #[derive(serde::Deserialize)]
        struct PackMessage {
            role: String,
            content: String,
        }
        #[derive(serde::Deserialize)]
        struct Pack {
            pack_id: String,
            support_scope: String,
            shapes: Vec<Shape>,
        }

        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/qa/prompt-packs/qwen3-chatml-thinking-template-pack-v1.json"
        );
        let raw = std::fs::read_to_string(path).expect("read qwen3 thinking template pack");
        let pack: Pack = serde_json::from_str(&raw).expect("parse qwen3 thinking template pack");

        // The pack is the opt-in thinking lane; its scope must stay the
        // honestly-bounded one (never the exact-row smoke scope).
        assert_eq!(pack.pack_id, "qwen3-chatml-thinking-template-pack-v1");
        assert_eq!(pack.support_scope, "thinking_opt_in_leading_trace_only");
        assert!(!pack.shapes.is_empty(), "pack must carry shapes");
        // At least one enable_thinking=true shape â€” the lane's whole reason to exist.
        assert!(
            pack.shapes.iter().any(|s| s.enable_thinking),
            "pack must lock at least one enable_thinking=true shape"
        );

        for shape in &pack.shapes {
            let messages: Vec<ChatMessage> = shape
                .messages
                .iter()
                .map(|m| ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            let rendered = render_qwen3_chatml_prompt(&messages, shape.enable_thinking);
            assert_eq!(
                rendered, shape.rendered_prompt,
                "shape {} (enable_thinking={}) diverges from the locked reference rendering",
                shape.id, shape.enable_thinking
            );
        }
    }

    #[test]
    fn qwen3_chatml_prompt_with_tools_renders_tool_definitions_and_suppresses_thinking() {
        let messages = [
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "system".to_string(),
                content: "You are an agent.".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "Read notes.txt".to_string(),
            },
        ];
        let tools = vec![serde_json::json!({
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }
        })];
        let prompt = render_qwen3_chatml_prompt_with_tools(&messages, &tools);
        // System turn merges caller system content + tool definitions.
        assert!(prompt.starts_with("<|im_start|>system\nYou are an agent.\n\n"));
        assert!(prompt.contains("You are a helpful assistant with access to the following tools:"));
        assert!(prompt.contains("\"name\":\"read_file\""));
        assert!(prompt.contains("<tool_call>"));
        // User turn present.
        assert!(prompt.contains("<|im_start|>user\nRead notes.txt<|im_end|>"));
        // Thinking suppressed in assistant generation prompt.
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        // System role only appears once at the start.
        assert_eq!(prompt.matches("<|im_start|>system").count(), 1);
    }

    #[test]
    fn qwen3_chatml_prompt_with_tools_no_system_message() {
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let tools = vec![serde_json::json!({
            "name": "search",
            "description": "Search",
            "parameters": { "type": "object", "properties": {} }
        })];
        let prompt = render_qwen3_chatml_prompt_with_tools(&messages, &tools);
        // Tool block IS the system message when no caller system content.
        assert!(prompt.starts_with(
            "<|im_start|>system\nYou are a helpful assistant with access to the following tools:"
        ));
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
    }

    #[test]
    fn strip_channels_removes_thinking_spans() {
        assert_eq!(
            gemma4_strip_channels("<|channel>secret plan<channel|>Paris"),
            "Paris"
        );
        assert_eq!(
            gemma4_strip_channels("a<|channel>x<channel|>b<|channel>y<channel|>c"),
            "abc"
        );
        // Unterminated channel (token budget hit mid-thought): nothing leaks.
        assert_eq!(gemma4_strip_channels("ok<|channel>still thinking"), "ok");
        assert_eq!(gemma4_strip_channels("plain"), "plain");
    }

    #[test]
    fn streaming_channel_filter_suppresses_thinking_deltas() {
        let mut f = Gemma4ChannelFilter::new();
        assert_eq!(f.filter("Hello "), "Hello ");
        assert_eq!(f.filter("<|channel>"), "");
        assert_eq!(f.filter("hidden reasoning"), "");
        assert_eq!(f.filter("<channel|>"), "");
        assert_eq!(f.filter("Paris"), "Paris");
        // Markers and text inside one delta.
        let mut g = Gemma4ChannelFilter::new();
        assert_eq!(g.filter("A<|channel>h<channel|>B"), "AB");
    }
}

/// Test-only re-export of the gemma4 marker renderer (the template-shapes
/// integration test asserts byte parity against the committed reference pack).
pub fn gemma4_chat_prompt_for_tests(messages: &[ChatMessage], thinking: bool) -> String {
    gemma4_chat_prompt(messages, thinking)
}

fn gemma4_chat_prompt(messages: &[ChatMessage], thinking: bool) -> String {
    // Mirrors the reference (GGUF-embedded Jinja, verified against llama.cpp's
    // /apply-template for both modes):
    // - thinking=false: a leading system message gets its OWN `<|turn>system`
    //   turn (never folded into the user turn); no synthetic system turn
    //   otherwise.
    // - thinking=true (the reference's --jinja default): a system turn is
    //   ALWAYS emitted and opens with the `<|think|>` token, then any system
    //   text.
    let mut out = String::new();
    let mut system_text = String::new();
    let mut rest_start = 0;
    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push_str("\n\n");
            }
            system_text.push_str(&m.content);
            rest_start += 1;
        } else {
            break;
        }
    }
    if thinking {
        out.push_str(GEMMA4_TURN_START);
        out.push_str("system\n");
        out.push_str(GEMMA4_THINK);
        out.push('\n');
        out.push_str(&system_text);
        out.push_str(GEMMA4_TURN_END);
        out.push('\n');
    } else if !system_text.is_empty() {
        out.push_str(GEMMA4_TURN_START);
        out.push_str("system\n");
        out.push_str(&system_text);
        out.push_str(GEMMA4_TURN_END);
        out.push('\n');
    }
    for m in &messages[rest_start..] {
        let role = if m.role == "assistant" {
            "model"
        } else {
            "user"
        };
        out.push_str(GEMMA4_TURN_START);
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push_str(GEMMA4_TURN_END);
        out.push('\n');
    }
    out.push_str(GEMMA4_TURN_START);
    out.push_str("model\n");
    out
}

/// Strip `<|channel>â€¦<channel|>` thinking spans from a complete gemma4 chat
/// response. An unterminated span (generation hit the token budget inside the
/// channel) is stripped to its start â€” hidden reasoning must never leak.
fn gemma4_strip_channels(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(GEMMA4_CHANNEL_START) {
        out.push_str(&rest[..start]);
        let after = &rest[start + GEMMA4_CHANNEL_START.len()..];
        match after.find(GEMMA4_CHANNEL_END) {
            Some(end) => rest = &after[end + GEMMA4_CHANNEL_END.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Streaming-side channel suppressor: feed decoded deltas in, get the
/// client-visible portion out. Marker pieces decode atomically (single vocab
/// tokens), so state flips on exact marker occurrences within a delta.
struct Gemma4ChannelFilter {
    in_channel: bool,
}

impl Gemma4ChannelFilter {
    fn new() -> Self {
        Self { in_channel: false }
    }

    fn filter(&mut self, delta: &str) -> String {
        let mut out = String::new();
        let mut rest = delta;
        loop {
            if self.in_channel {
                match rest.find(GEMMA4_CHANNEL_END) {
                    Some(end) => {
                        self.in_channel = false;
                        rest = &rest[end + GEMMA4_CHANNEL_END.len()..];
                    }
                    None => return out,
                }
            } else {
                match rest.find(GEMMA4_CHANNEL_START) {
                    Some(start) => {
                        out.push_str(&rest[..start]);
                        self.in_channel = true;
                        rest = &rest[start + GEMMA4_CHANNEL_START.len()..];
                    }
                    None => {
                        out.push_str(rest);
                        return out;
                    }
                }
            }
        }
    }
}

/// Resolve the Gemma 4 runtime for a chat request, if this request targets one.
/// Returns `Err(response)` to short-circuit with a clear error (a gemma4 model is
/// loaded but its runtime is missing), `Ok(None)` to fall through to the Llama
/// path, or `Ok(Some(runtime))` to serve via Gemma 4.
/// The serve-lane gemma4 runtime: single-node local decode, or the two-node
/// distributed layer-sharding lane (master shard in-process, tail layers on a
/// worker over TCP). Both expose the same greedy contract, so the chat and
/// completion handlers are lane-agnostic. The distributed lane is configured
/// at model-load time via `CAMELID_GEMMA4_WORKER` + `CAMELID_GEMMA4_SPLIT`
/// (alongside `CAMELID_GEMMA4_SERVE=1`).
// Always stored behind an Arc (AppState::gemma4_runtimes), so there is exactly one
// heap-allocated instance per loaded model; the Cuda variant's resident scratch dwarfs
// the others, but the inline size disparity has no practical cost here.
#[allow(clippy::large_enum_variant)]
pub enum Gemma4ServeRuntime {
    Local(crate::gemma4_runtime::Gemma4Runtime),
    Distributed(crate::gemma4_distributed::Gemma4DistributedRuntime),
    /// CUDA decode engine (stateful GPU runtime -> Mutex; one request at a time).
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Mutex<crate::gemma4_runtime::Gemma4CudaResident>),
}

impl Gemma4ServeRuntime {
    fn generate_greedy(&self, prompt: &str, max_new: usize) -> crate::Result<(String, Vec<u32>)> {
        match self {
            Self::Local(r) => r.generate_greedy(prompt, max_new),
            Self::Distributed(r) => r.generate_greedy(prompt, max_new),
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => m
                .lock()
                .expect("gemma4 cuda runtime lock")
                .generate_greedy(prompt, max_new),
        }
    }

    fn generate_greedy_streaming<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        on_delta: F,
    ) -> crate::Result<(String, Vec<u32>)> {
        match self {
            Self::Local(r) => r.generate_greedy_streaming(prompt, max_new, on_delta),
            Self::Distributed(r) => r.generate_greedy_streaming(prompt, max_new, on_delta),
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => m
                .lock()
                .expect("gemma4 cuda runtime lock")
                .generate_greedy_streaming(prompt, max_new, on_delta),
        }
    }
}

/// Distributed gemma4 serve config from the environment: both vars must be
/// present and well-formed, or the pair is rejected loudly â€” a half-configured
/// distributed lane must never silently fall back to a partial local load.
fn gemma4_distributed_serve_config() -> std::result::Result<Option<(String, usize)>, String> {
    let worker = std::env::var("CAMELID_GEMMA4_WORKER").ok();
    let split = std::env::var("CAMELID_GEMMA4_SPLIT").ok();
    match (worker, split) {
        (None, None) => Ok(None),
        (Some(w), Some(s)) => {
            let split: usize = s.parse().map_err(|_| {
                format!("CAMELID_GEMMA4_SPLIT must be a positive layer index, got {s:?}")
            })?;
            if split == 0 {
                return Err("CAMELID_GEMMA4_SPLIT must be >= 1".to_string());
            }
            Ok(Some((w, split)))
        }
        (Some(_), None) => {
            Err("CAMELID_GEMMA4_WORKER is set but CAMELID_GEMMA4_SPLIT is missing".to_string())
        }
        (None, Some(_)) => {
            Err("CAMELID_GEMMA4_SPLIT is set but CAMELID_GEMMA4_WORKER is missing".to_string())
        }
    }
}

async fn resolve_gemma4_runtime(
    state: &AppState,
    req: &ChatCompletionRequest,
) -> std::result::Result<Option<(String, Arc<Gemma4ServeRuntime>)>, Response> {
    resolve_gemma4_runtime_for_model(state, &req.model).await
}

async fn resolve_gemma4_runtime_for_model(
    state: &AppState,
    model: &Option<String>,
) -> std::result::Result<Option<(String, Arc<Gemma4ServeRuntime>)>, Response> {
    let id = match model.clone() {
        Some(m) => m,
        None => match state.active_model_id.read().await.clone() {
            Some(m) => m,
            None => return Ok(None),
        },
    };
    if let Some(runtime) = state.gemma4_runtimes.read().await.get(&id).cloned() {
        return Ok(Some((id, runtime)));
    }
    // No runtime: if the model itself is gemma4, fail clearly rather than letting
    // the Llama path produce garbage.
    let is_gemma4 = state
        .loaded_models
        .read()
        .await
        .get(&id)
        .map(|m| model_family(&m.gguf) == "gemma4")
        .unwrap_or(false);
    if is_gemma4 {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_ready",
            format!(
                "gemma4 model '{id}' is loaded but its serve runtime is unavailable; \
                 set CAMELID_GEMMA4_SERVE=1 and reload the model"
            ),
            None,
        ));
    }
    Ok(None)
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Non-streaming Gemma 4 chat. Builds the gemma prompt, generates greedily on a
/// blocking thread, and returns a minimal OpenAI-compatible response.
/// Gemma 4 raw completion (non-streaming): BOS + plain prompt text through the
/// greedy runtime â€” the same envelope the committed basic_v1 oracle pack checks.
async fn gemma4_completion_nonstreaming(
    id: String,
    runtime: Arc<Gemma4ServeRuntime>,
    req: &CompletionRequest,
) -> Response {
    let Some(prompt) = req.prompt.clone() else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "prompt is required".to_string(),
            Some("prompt"),
        );
    };
    let max_tokens = req.max_tokens.unwrap_or(64).min(4096) as usize;
    let t_generate = std::time::Instant::now();
    let result =
        tokio::task::spawn_blocking(move || runtime.generate_greedy(&prompt, max_tokens)).await;
    let generate_ms = t_generate.elapsed().as_secs_f64() * 1e3;
    let (text, ids) = match result {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                e.to_string(),
                None,
            )
        }
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                format!("gemma4 generation task panicked: {e}"),
                None,
            )
        }
    };
    let body = serde_json::json!({
        "id": "cmpl-gemma4",
        "object": "text_completion",
        "created": unix_secs(),
        "model": id,
        "choices": [{
            "index": 0,
            "text": text,
            "logprobs": null,
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": ids.len(), "total_tokens": ids.len() },
        "camelid": {
            "generated_token_ids": ids,
            // Wall-clock totals only: the gemma4 lane does not (yet) report
            // per-layer timing buckets like the Llama diagnostics do.
            "timings_ms": {
                "generate": generate_ms,
                "generation": { "forward_total": generate_ms },
                "prompt_evaluation": {},
                "lane": "gemma4_wall_clock_total_only",
            },
        },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Gemma 4 raw completion, streaming (SSE `text_completion` chunks + [DONE]).
async fn gemma4_completion_streaming(
    id: String,
    runtime: Arc<Gemma4ServeRuntime>,
    req: &CompletionRequest,
) -> Response {
    let Some(prompt) = req.prompt.clone() else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "prompt is required".to_string(),
            Some("prompt"),
        );
    };
    let max_tokens = req.max_tokens.unwrap_or(64).min(4096) as usize;
    let created = unix_secs();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();
    tokio::task::spawn_blocking(move || {
        let send_tx = tx.clone();
        let result = runtime.generate_greedy_streaming(&prompt, max_tokens, |delta| {
            let _ = send_tx.send(Ok(delta.to_string()));
        });
        if let Err(e) = result {
            let _ = tx.send(Err(e.to_string()));
        }
    });

    let events = async_stream::stream! {
        let mut errored = false;
        while let Some(item) = rx.recv().await {
            match item {
                Ok(delta) => {
                    let chunk = serde_json::json!({
                        "id": "cmpl-gemma4",
                        "object": "text_completion",
                        "created": created,
                        "model": id,
                        "choices": [{ "index": 0, "text": delta, "logprobs": null, "finish_reason": null }],
                    });
                    yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                }
                Err(e) => {
                    let err = serde_json::json!({ "error": { "message": e, "type": "generation_error" } });
                    yield Ok(Event::default().data(err.to_string()));
                    errored = true;
                    break;
                }
            }
        }
        if !errored {
            let done = serde_json::json!({
                "id": "cmpl-gemma4",
                "object": "text_completion",
                "created": created,
                "model": id,
                "choices": [{ "index": 0, "text": "", "logprobs": null, "finish_reason": "stop" }],
            });
            yield Ok(Event::default().data(done.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(events).into_response()
}

async fn gemma4_chat_nonstreaming(
    id: String,
    runtime: Arc<Gemma4ServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let prompt = gemma4_chat_prompt(&messages, req.camelid_enable_thinking.unwrap_or(false));
    let max_tokens = req.max_tokens.unwrap_or(256).min(4096) as usize;
    let t_generate = std::time::Instant::now();
    // Lifecycle telemetry only on this lane: the gemma4 runtime does not
    // (yet) report prompt token counts or per-layer events, so those fields
    // stay at their "not reported" zero values rather than being estimated.
    let telemetry_guard =
        telemetry::RequestGuard::begin(gemma4_telemetry_start(&id, max_tokens as u32, false));
    let result =
        tokio::task::spawn_blocking(move || runtime.generate_greedy(&prompt, max_tokens)).await;
    let generate_ms = t_generate.elapsed().as_secs_f64() * 1e3;
    let (text, ids) = match result {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            telemetry_guard.finish(gemma4_telemetry_error(e.to_string()));
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                e.to_string(),
                None,
            );
        }
        Err(e) => {
            let message = format!("gemma4 generation task panicked: {e}");
            telemetry_guard.finish(gemma4_telemetry_error(message.clone()));
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                message,
                None,
            );
        }
    };
    telemetry_guard.finish(telemetry::RequestFinish {
        status: "ok",
        finish_reason: Some("stop".to_string()),
        completion_tokens: ids.len(),
        ttft_ms: None,
        decode_tps: None,
        prefill_tps: None,
        error: None,
    });
    let body = serde_json::json!({
        "id": "chatcmpl-gemma4",
        "object": "chat.completion",
        "created": unix_secs(),
        "model": id,
        "choices": [{
            "index": 0,
            // Thinking channels are stripped: hidden reasoning never reaches
            // the client or re-enters chat history.
            "message": { "role": "assistant", "content": gemma4_strip_channels(&text) },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 0, "completion_tokens": ids.len(), "total_tokens": ids.len() },
        "camelid": {
            "generated_token_ids": ids,
            // Wall-clock totals only: the gemma4 lane does not (yet) report
            // per-layer timing buckets like the Llama diagnostics do.
            "timings_ms": {
                "generate": generate_ms,
                "generation": { "forward_total": generate_ms },
                "prompt_evaluation": {},
                "lane": "gemma4_wall_clock_total_only",
            },
        },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Telemetry request identity for the gemma4 serve lane. Prompt token count
/// and context length are not reported by this runtime, so they are recorded
/// as 0 ("not reported") instead of being estimated.
fn gemma4_telemetry_start(
    model_id: &str,
    max_tokens: u32,
    stream: bool,
) -> telemetry::RequestStart {
    telemetry::RequestStart {
        request_id: uuid::Uuid::new_v4().to_string(),
        model_id: model_id.to_string(),
        backend: "gemma4-runtime".to_string(),
        quantization: String::new(),
        architecture: "gemma4".to_string(),
        prompt_tokens: 0,
        max_tokens,
        context_length: 0,
        temperature: 0.0,
        stream,
    }
}

fn gemma4_telemetry_error(message: String) -> telemetry::RequestFinish {
    telemetry::RequestFinish {
        status: "error",
        finish_reason: None,
        completion_tokens: 0,
        ttft_ms: None,
        decode_tps: None,
        prefill_tps: None,
        error: Some(message),
    }
}

// ===================================================================================
// Runnable-lane serve bridge (additive, gated by CAMELID_RUNNABLE_SERVE).
//
// Architectures implemented only in the runnable (pure-f32 oracle) lane â€” currently
// `qwen35` (Ornith) â€” are not in the optimized inference engine, so the Llama serve
// path fails closed on them. This bridge mirrors the gemma4 serve pattern: a parallel
// per-model-id runtime map, a short-circuit at the top of `chat_completions`, and a
// dedicated chat handler. The optimized lane is untouched. Generation is greedy
// (matches the brief) on a blocking thread; the runtime is `&self`-immutable so it
// needs no Mutex (unlike the CUDA gemma4 variant).
// ===================================================================================

/// The Ornith / qwen35 tool-call instruction literal, byte-for-byte from the GGUF
/// chat template (the custom `<tool_call><function=â€¦><parameter=â€¦>` format the model
/// was trained on). Rendering anything else makes the model emit the wrong format.
const ORNITH_TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The runnable serve lane is gated behind `CAMELID_RUNNABLE_SERVE` (1/true/yes).
/// When off, a qwen35 model load is metadata-only (no serve runtime) exactly as today.
fn runnable_serve_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_RUNNABLE_SERVE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

/// True for architectures served through the runnable bridge (qwen35, and — since
/// MUSTER M-A1 — gemma3, whose only correct forward lives in the runnable lane; the
/// optimized dense binder silently drops gemma3's QK/post norms and has no GeGLU).
/// Deliberate side effect of listing an arch here: WITHOUT `CAMELID_RUNNABLE_SERVE=1`
/// its chat requests get a typed 503 instead of falling through to the mis-bound
/// optimized engine — fail-closed by design.
fn is_runnable_serve_arch(arch: &str) -> bool {
    matches!(arch, "qwen35" | "gemma2" | "gemma3")
}

/// A runnable-lane model wrapped for the serve path: greedy generation + the GGUF
/// tokenizer (for prompt encode, EOG stop set, and detokenize).
pub struct RunnableServeRuntime {
    model: crate::runnable::RunnableModel,
    tokenizer: std::sync::Arc<Tokenizer>,
    architecture: String,
}

impl RunnableServeRuntime {
    fn load(path: &std::path::Path) -> std::result::Result<Self, BackendError> {
        let path_str = path.to_string_lossy().to_string();
        let gguf = crate::gguf::read_metadata(&path_str)?;
        let architecture = gguf.architecture().unwrap_or_default().to_string();
        let tokenizer = std::sync::Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model = crate::runnable::RunnableModel::load(&path_str)?;
        Ok(Self {
            model,
            tokenizer,
            architecture,
        })
    }

    /// Greedy-generate from already-tokenized `prompt_ids`, stopping at the first EOG
    /// (`<|im_end|>` / eos). Returns the detokenized text + the generated token ids.
    fn generate_greedy(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
    ) -> std::result::Result<(String, Vec<u32>), BackendError> {
        let stop: Vec<u32> = self.tokenizer.special.eog.iter().copied().collect();
        let ids = self.model.generate_stopping(prompt_ids, max_new, &stop)?;
        let text = self.tokenizer.decode(&ids, true).unwrap_or_default();
        Ok((text, ids))
    }

    /// [`generate_greedy`](Self::generate_greedy) with a per-token-id callback —
    /// the runnable lane's SSE source. Returns the same (text, ids) as the
    /// non-streaming path (identical generation by construction).
    fn generate_greedy_streaming<F: FnMut(u32)>(
        &self,
        prompt_ids: &[u32],
        max_new: usize,
        mut on_token: F,
    ) -> std::result::Result<(String, Vec<u32>), BackendError> {
        let stop: Vec<u32> = self.tokenizer.special.eog.iter().copied().collect();
        let ids =
            self.model
                .generate_stopping_streaming(prompt_ids, max_new, &stop, &mut on_token)?;
        let text = self.tokenizer.decode(&ids, true).unwrap_or_default();
        Ok((text, ids))
    }
}

/// Render a gemma3 chat prompt byte-faithful to the GGUF `tokenizer.chat_template`
/// (MUSTER M-A1): `<bos>` + per-turn `<start_of_turn>{role}\n{content|trim}<end_of_turn>\n`
/// with assistant renamed to "model"; a leading system message is folded into the FIRST
/// loop turn as a `{system_content}\n\n` prefix inserted after the role header and
/// before the trimmed content — the prefix itself is NOT trimmed, exactly like the
/// template's `messages[0]['content'] + '\n\n'`; generation prompt `<start_of_turn>model\n`
/// unless the last message is an assistant turn. The template's multimodal branch is
/// unreachable here (string content only; image parts fail closed upstream) and its
/// user/assistant-alternation `raise_exception` is not re-enforced (the parity packs and
/// smoke exercise only valid alternating shapes). The rendered string carries NO BOS —
/// byte-identical to the pinned oracle's `/apply-template` output — and the runnable
/// bridge encodes gemma3 with `add_special=true` so BOS lands at token level exactly
/// like llama-server's own chat path (the GGUF declares `add_bos_token=true`, bos=2).
fn render_gemma3_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let mut first_user_prefix: Option<&str> = None;
    let mut loop_messages: &[ChatMessage] = messages;
    if let Some(first) = messages.first() {
        if first.role.trim() == "system" {
            first_user_prefix = Some(first.content.as_str());
            loop_messages = &messages[1..];
        }
    }
    let mut append_generation_prompt = true;
    for (i, message) in loop_messages.iter().enumerate() {
        let trimmed_role = message.role.trim();
        let role = if trimmed_role == "assistant" {
            "model"
        } else {
            trimmed_role
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        if i == 0 {
            if let Some(prefix) = first_user_prefix {
                prompt.push_str(prefix);
                prompt.push_str("\n\n");
            }
        }
        prompt.push_str(message.content.trim());
        prompt.push_str("<end_of_turn>\n");
        append_generation_prompt = trimmed_role != "assistant";
    }
    if append_generation_prompt {
        prompt.push_str("<start_of_turn>model\n");
    }
    prompt
}

/// Render an Ornith/qwen35 ChatML prompt (no tools). The generation prompt opens the
/// reasoning block (`<think>\n`) when thinking is enabled, else prefills an empty one.
fn render_ornith_chatml_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut prompt = String::new();
    let mut append_generation_prompt = true;
    for message in messages {
        let role = message.role.trim();
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        append_generation_prompt = role != "assistant";
    }
    if append_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
        prompt.push_str(if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    prompt
}

/// Render an Ornith/qwen35 ChatML prompt with tool definitions, faithful to the GGUF
/// template's tools system block + custom `<function=â€¦>` instructions. `tools` are the
/// flat function objects (`{name,description,parameters}`). Tool results (`role:"tool"`)
/// are wrapped in `<tool_response>` user turns as the template expects.
fn render_ornith_chatml_prompt_with_tools(
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
    enable_thinking: bool,
) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        if let Ok(json) = serde_json::to_string(tool) {
            prompt.push('\n');
            prompt.push_str(&json);
        }
    }
    prompt.push_str("\n</tools>");
    prompt.push_str(ORNITH_TOOL_INSTRUCTIONS);
    for message in messages {
        if message.role.trim() == "system" && !message.content.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(&message.content);
        }
    }
    prompt.push_str("<|im_end|>\n");

    let mut append_generation_prompt = true;
    for message in messages {
        let role = message.role.trim();
        if role == "system" {
            continue;
        }
        if role == "tool" {
            prompt.push_str("<|im_start|>user\n<tool_response>\n");
            prompt.push_str(&message.content);
            prompt.push_str("\n</tool_response><|im_end|>\n");
            append_generation_prompt = true;
            continue;
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        append_generation_prompt = role != "assistant";
    }
    if append_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
        prompt.push_str(if enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    prompt
}

/// Split an Ornith generation into `(reasoning, content)` on the first `</think>`.
/// The generation prompt prefills `<think>` (or an empty think block), so the model's
/// output is `REASONING</think>\n\nCONTENT` (thinking on) or just `CONTENT` (off). The
/// reasoning is surfaced separately and never re-enters tool parsing or content.
fn split_ornith_think(text: &str) -> (Option<String>, String) {
    if let Some(end) = text.find("</think>") {
        let reasoning = text[..end].trim_start_matches("<think>").trim().to_string();
        let content = text[end + "</think>".len()..].trim_start().to_string();
        let reasoning = if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        };
        (reasoning, content)
    } else {
        (None, text.to_string())
    }
}

/// Lift Ornith/qwen35 `<tool_call><function=NAME><parameter=ARG>VALUE</parameter>â€¦
/// </function></tool_call>` XML into OpenAI `tool_calls` JSON
/// (`{id,type:"function",function:{name,arguments:<json-string>}}`). Mirrors the
/// chat-lane `parse_ornith`; `arguments` is a JSON object string. Scalars stay
/// strings; values that look like JSON objects/arrays are decoded.
fn parse_ornith_tool_calls_json(text: &str) -> Vec<serde_json::Value> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(fstart) = rest.find("<function=") {
        let after = &rest[fstart + "<function=".len()..];
        let Some(name_end) = after.find('>') else {
            break;
        };
        let name = after[..name_end].trim().to_string();
        let body = &after[name_end + 1..];
        let (params_blob, next) = match body.find("</function>") {
            Some(end) => (&body[..end], &body[end + "</function>".len()..]),
            None => (body, ""),
        };
        let mut args = serde_json::Map::new();
        let mut p = params_blob;
        while let Some(ps) = p.find("<parameter=") {
            let pa = &p[ps + "<parameter=".len()..];
            let Some(pname_end) = pa.find('>') else { break };
            let pname = pa[..pname_end].trim().to_string();
            let pbody = &pa[pname_end + 1..];
            let (pval, pnext) = match pbody.find("</parameter>") {
                Some(end) => (&pbody[..end], &pbody[end + "</parameter>".len()..]),
                None => (pbody, ""),
            };
            let v = pval.strip_prefix('\n').unwrap_or(pval);
            let v = v.strip_suffix('\n').unwrap_or(v);
            let trimmed = v.trim();
            let value = if trimmed.starts_with('{') || trimmed.starts_with('[') {
                serde_json::from_str::<serde_json::Value>(trimmed)
                    .unwrap_or_else(|_| serde_json::Value::String(v.to_string()))
            } else {
                serde_json::Value::String(v.to_string())
            };
            if !pname.is_empty() {
                args.insert(pname, value);
            }
            p = pnext;
        }
        if !name.is_empty() {
            calls.push(serde_json::json!({
                "id": format!("call_{}", calls.len()),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": serde_json::Value::Object(args).to_string(),
                },
            }));
        }
        rest = next;
    }
    calls
}

/// Resolve a runnable serve runtime for the requested (or active) model id.
async fn resolve_runnable_runtime(
    state: &AppState,
    model: &Option<String>,
) -> std::result::Result<Option<(String, Arc<RunnableServeRuntime>)>, Response> {
    let id = match model.clone() {
        Some(m) => m,
        None => match state.active_model_id.read().await.clone() {
            Some(m) => m,
            None => return Ok(None),
        },
    };
    if let Some(runtime) = state.runnable_runtimes.read().await.get(&id).cloned() {
        return Ok(Some((id, runtime)));
    }
    // Loaded as a runnable-served arch but no runtime â†’ fail clearly rather than
    // letting the Llama path produce garbage / an unsupported error.
    let needs_runnable = state
        .loaded_models
        .read()
        .await
        .get(&id)
        .map(|m| is_runnable_serve_arch(m.gguf.architecture().unwrap_or_default()))
        .unwrap_or(false);
    if needs_runnable {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_ready",
            format!(
                "model '{id}' (runnable-lane architecture) is loaded but its serve runtime \
                 is unavailable; set CAMELID_RUNNABLE_SERVE=1 and reload the model"
            ),
            None,
        ));
    }
    Ok(None)
}

/// True when raw `/v1/completions` must fail closed for this architecture: the
/// runnable-served archs have no completions bridge, and falling through would
/// reach the optimized engine — which for gemma3 is numerically mis-bound (its
/// norms are silently dropped and the forward has no GeGLU/dual-RoPE). Their
/// only served surface is `/v1/chat/completions`. Pure decision half of
/// [`reject_completions_for_runnable_arch`] so the gate itself is unit-testable.
fn completions_unsupported_for_arch(arch: &str) -> bool {
    is_runnable_serve_arch(arch)
}

/// Fail-closed guard for `/v1/completions` (MUSTER M-A1 follow-up): reject
/// requests targeting a loaded runnable-served model with a typed error that
/// points callers at the chat endpoint, instead of silently serving them from
/// the wrong engine. Returns `Some(response)` when the request must be rejected.
async fn reject_completions_for_runnable_arch(
    state: &AppState,
    model: &Option<String>,
) -> Option<Response> {
    let id = match model.clone() {
        Some(m) => m,
        None => state.active_model_id.read().await.clone()?,
    };
    let unsupported = state
        .loaded_models
        .read()
        .await
        .get(&id)
        .map(|m| completions_unsupported_for_arch(m.gguf.architecture().unwrap_or_default()))
        .unwrap_or(false);
    if unsupported {
        return Some(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_completions_lane",
            format!(
                "model '{id}' is a runnable-lane architecture served only via \
                 /v1/chat/completions; raw /v1/completions has no runnable bridge and \
                 fails closed rather than falling through to the optimized engine"
            ),
            None,
        ));
    }
    None
}

/// Load the runnable serve runtime for a model id (blocking thread; ~9.5 GB read).
async fn load_runnable_serve_runtime(
    state: &AppState,
    id: &str,
    model_path: &std::path::Path,
) -> std::result::Result<(), BackendError> {
    let load_path = model_path.to_path_buf();
    let runtime = tokio::task::spawn_blocking(move || RunnableServeRuntime::load(&load_path))
        .await
        .map_err(|e| {
            BackendError::InvalidModelMetadata(format!("runnable serve load task panicked: {e}"))
        })??;
    let arch = runtime.architecture.clone();
    state
        .runnable_runtimes
        .write()
        .await
        .insert(id.to_string(), Arc::new(runtime));
    tracing::info!(model = %id, arch = %arch, "runnable serve runtime loaded");
    Ok(())
}

/// Non-streaming chat for a runnable-served model (qwen35/Ornith): render the Ornith
/// ChatML prompt (with tools when present), greedy-generate to EOG, split the
/// `<think>` reasoning, and lift `<function=â€¦>` tool calls into structured `tool_calls`
/// (the content keeps the tool-call text so the agent's client-side parser also lifts).
async fn runnable_chat_nonstreaming(
    id: String,
    runtime: Arc<RunnableServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let enable_thinking = req.camelid_enable_thinking.unwrap_or(false);
    let tools: Vec<serde_json::Value> = req
        .tools
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.get("function").cloned().unwrap_or(t))
        .collect();
    let prompt_text = if runtime.architecture == "gemma2" || runtime.architecture == "gemma3" {
        if !tools.is_empty() {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_tools",
                "the gemma runnable serve lane does not support tools: the model's chat template has no tools branch and no tool-call grammar is certified for this row".to_string(),
                None,
            );
        }
        render_gemma3_prompt(&messages)
    } else if tools.is_empty() {
        render_ornith_chatml_prompt(&messages, enable_thinking)
    } else {
        render_ornith_chatml_prompt_with_tools(&messages, &tools, enable_thinking)
    };
    // gemma3 renders carry no BOS string (oracle /apply-template parity) — BOS is
    // added at token level (add_special=true), matching llama-server's chat path.
    let add_special = runtime.architecture == "gemma2" || runtime.architecture == "gemma3";
    let prompt_ids = match runtime.tokenizer.encode(&prompt_text, add_special, true) {
        Ok(ids) => ids,
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "tokenize_error",
                e.to_string(),
                None,
            )
        }
    };
    let prompt_token_count = prompt_ids.len();
    let max_tokens = req.max_tokens.unwrap_or(256).min(4096) as usize;
    let rt = runtime.clone();
    let result =
        tokio::task::spawn_blocking(move || rt.generate_greedy(&prompt_ids, max_tokens)).await;
    let (text, ids) = match result {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                e.to_string(),
                None,
            )
        }
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                format!("runnable generation task panicked: {e}"),
                None,
            )
        }
    };

    let (reasoning, content) = split_ornith_think(&text);
    // Structured tool_calls (OpenAI shape) lifted from the Ornith `<function=â€¦>` XML.
    // The agent loop ALSO re-parses the content text client-side (chat-lane
    // `parse_ornith`), so the content keeps the tool-call text below.
    let tool_calls = parse_ornith_tool_calls_json(&content);
    let finish_reason = if tool_calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };

    let mut message = serde_json::json!({ "role": "assistant", "content": content });
    if let Some(r) = reasoning {
        message["reasoning_content"] = serde_json::Value::String(r);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = serde_json::Value::Array(tool_calls);
    }
    let body = serde_json::json!({
        "id": "chatcmpl-runnable",
        "object": "chat.completion",
        "created": unix_secs(),
        "model": id,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason }],
        "usage": {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": ids.len(),
            "total_tokens": prompt_token_count + ids.len(),
        },
        "camelid": { "generated_token_ids": ids, "lane": "runnable_qwen35" },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Streaming chat for a runnable-served model (qwen35/Ornith), SSE. Mirrors the
/// OpenAI `chat.completion.chunk` shape and the non-streaming bridge's semantics:
/// think-block tokens stream as `delta.reasoning_content`, post-`</think>` tokens
/// as `delta.content` (tool-call XML included, as in non-streaming), then one
/// aggregate `tool_calls` delta when the finished content lifts into structured
/// calls, the finish_reason chunk, an optional `stream_options.include_usage`
/// terminal usage chunk, and `[DONE]`. The phase switch keys on the `</think>`
/// TOKEN ID (a single user_defined token in the qwen35 vocab), so no text
/// scanning is needed; per-phase text is decoded incrementally with UTF-8
/// hold-back (a multi-token code point emits only once complete).
async fn runnable_chat_streaming(
    id: String,
    runtime: Arc<RunnableServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let enable_thinking = req.camelid_enable_thinking.unwrap_or(false);
    let tools: Vec<serde_json::Value> = req
        .tools
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.get("function").cloned().unwrap_or(t))
        .collect();
    let prompt_text = if runtime.architecture == "gemma2" || runtime.architecture == "gemma3" {
        if !tools.is_empty() {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_tools",
                "the gemma runnable serve lane does not support tools: the model's chat template has no tools branch and no tool-call grammar is certified for this row".to_string(),
                None,
            );
        }
        render_gemma3_prompt(&messages)
    } else if tools.is_empty() {
        render_ornith_chatml_prompt(&messages, enable_thinking)
    } else {
        render_ornith_chatml_prompt_with_tools(&messages, &tools, enable_thinking)
    };
    // gemma3 renders carry no BOS string (oracle /apply-template parity) — BOS is
    // added at token level (add_special=true), matching llama-server's chat path.
    let add_special = runtime.architecture == "gemma2" || runtime.architecture == "gemma3";
    let prompt_ids = match runtime.tokenizer.encode(&prompt_text, add_special, true) {
        Ok(ids) => ids,
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "tokenize_error",
                e.to_string(),
                None,
            )
        }
    };
    let prompt_token_count = prompt_ids.len();
    let max_tokens = req.max_tokens.unwrap_or(256).min(4096) as usize;
    let include_usage = stream_options_include_usage(req.stream_options.as_ref());
    let think_close = runtime.tokenizer.token_to_id.get("</think>").copied();
    let created = unix_secs();

    enum StreamItem {
        Token(u32),
        Done(String, Vec<u32>),
        Fail(String),
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamItem>();
    let rt = runtime.clone();
    tokio::task::spawn_blocking(move || {
        let send_tx = tx.clone();
        let result = rt.generate_greedy_streaming(&prompt_ids, max_tokens, |tok| {
            let _ = send_tx.send(StreamItem::Token(tok));
        });
        match result {
            Ok((text, ids)) => {
                let _ = tx.send(StreamItem::Done(text, ids));
            }
            Err(e) => {
                let _ = tx.send(StreamItem::Fail(e.to_string()));
            }
        }
    });

    let tokenizer = runtime.tokenizer.clone();
    let events = async_stream::stream! {
        let chunk = |delta: serde_json::Value, finish: Option<&str>| {
            serde_json::json!({
                "id": "chatcmpl-runnable",
                "object": "chat.completion.chunk",
                "created": created,
                "model": id,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        };
        yield Ok::<Event, std::convert::Infallible>(
            Event::default().data(chunk(serde_json::json!({ "role": "assistant" }), None).to_string()),
        );

        // Per-phase incremental decode state. `emitted` counts bytes of the phase's
        // decoded string already sent; `seen_visible` gates the leading-whitespace
        // trim (mirroring split_ornith_think's trim of the reasoning/content edges).
        let mut in_think = enable_thinking && think_close.is_some();
        let mut phase_ids: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let mut seen_visible = false;
        let mut final_state: Option<std::result::Result<(String, Vec<u32>), String>> = None;

        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Token(tok) => {
                    if in_think && Some(tok) == think_close {
                        // Phase switch: reasoning is done; content decodes fresh.
                        in_think = false;
                        phase_ids.clear();
                        emitted = 0;
                        seen_visible = false;
                        continue;
                    }
                    phase_ids.push(tok);
                    let decoded = tokenizer.decode(&phase_ids, true).unwrap_or_default();
                    // UTF-8 hold-back: a code point split across tokens decodes to
                    // U+FFFD until its continuation arrives — wait for it.
                    if decoded.ends_with('\u{FFFD}') {
                        continue;
                    }
                    let mut new_start = emitted;
                    if !seen_visible {
                        let vis = decoded[new_start..]
                            .find(|c: char| !c.is_whitespace())
                            .map(|off| new_start + off);
                        match vis {
                            Some(v) => {
                                new_start = v;
                                seen_visible = true;
                            }
                            None => continue, // still leading whitespace — hold
                        }
                    }
                    if new_start >= decoded.len() {
                        continue;
                    }
                    let delta_text = decoded[new_start..].to_string();
                    emitted = decoded.len();
                    let delta = if in_think {
                        serde_json::json!({ "reasoning_content": delta_text })
                    } else {
                        serde_json::json!({ "content": delta_text })
                    };
                    yield Ok(Event::default().data(chunk(delta, None).to_string()));
                }
                StreamItem::Done(text, ids) => {
                    final_state = Some(Ok((text, ids)));
                    break;
                }
                StreamItem::Fail(e) => {
                    final_state = Some(Err(e));
                    break;
                }
            }
        }

        match final_state {
            Some(Ok((text, ids))) => {
                let (_reasoning, content) = split_ornith_think(&text);
                let tool_calls = parse_ornith_tool_calls_json(&content);
                let finish = if tool_calls.is_empty() { "stop" } else { "tool_calls" };
                if !tool_calls.is_empty() {
                    let deltas: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut d = c.clone();
                            d["index"] = serde_json::json!(i);
                            d
                        })
                        .collect();
                    yield Ok(Event::default().data(
                        chunk(serde_json::json!({ "tool_calls": deltas }), None).to_string(),
                    ));
                }
                yield Ok(Event::default().data(chunk(serde_json::json!({}), Some(finish)).to_string()));
                if include_usage {
                    let usage = serde_json::json!({
                        "id": "chatcmpl-runnable",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_token_count,
                            "completion_tokens": ids.len(),
                            "total_tokens": prompt_token_count + ids.len(),
                        },
                    });
                    yield Ok(Event::default().data(usage.to_string()));
                }
            }
            Some(Err(e)) => {
                let err = serde_json::json!({ "error": { "message": e, "type": "generation_error" } });
                yield Ok(Event::default().data(err.to_string()));
            }
            None => {}
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(events).into_response()
}

// ===================================================================================
// DiffusionGemma serve bridge (additive, gated by CAMELID_DG_SERVE).
//
// The block-diffusion lane cannot run on the AR engine — `model.rs` fails closed for
// this arch BY DESIGN and stays that way. This bridge mirrors the runnable-lane
// pattern: a parallel per-model-id runtime map, a short-circuit at the top of
// `chat_completions`, and dedicated handlers over `DgChat` (the Phase 6 bit-exact
// chat wrapper). SINGLE-TURN: the DG chat template renders exactly one user message
// (the parity-proven reference path), so the LAST user message wins and prior turns
// are ignored. A denoise block is minutes of compute: steps per block come from
// `CAMELID_DG_MAX_STEPS` (default: the reference 48 with adaptive early stop),
// blocks per answer from `CAMELID_DG_MAX_BLOCKS` (default 1). The SSE stream emits
// one content delta per COMMITTED BLOCK with keep-alive comments in between.
// ===================================================================================

/// The DiffusionGemma serve lane is gated behind `CAMELID_DG_SERVE` (1/true/yes).
/// When off, a diffusion-gemma load stays metadata-only (fail-closed redirect).
fn dg_serve_enabled() -> bool {
    matches!(
        std::env::var("CAMELID_DG_SERVE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

/// True for the DiffusionGemma architecture spellings the loader recognizes.
fn is_dg_serve_arch(arch: &str) -> bool {
    let a = arch.to_ascii_lowercase().replace(['-', '_'], "");
    a == "diffusiongemma" || a == "gemmadiffusion"
}

/// Per-block denoise step override for serve (`CAMELID_DG_MAX_STEPS`); `None`
/// keeps the reference default (48, adaptive early stop — ~40 min/block on a
/// 6 GB-GPU laptop; 8-16 trades quality for interactive latency).
fn dg_serve_steps() -> Option<i32> {
    std::env::var("CAMELID_DG_MAX_STEPS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .map(|v| v.max(1))
}

/// Blocks per answer for serve (`CAMELID_DG_MAX_BLOCKS`, default 1, clamped 1..=8).
/// Each block is a full 256-token canvas denoise.
fn dg_serve_blocks() -> i32 {
    std::env::var("CAMELID_DG_MAX_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1)
        .clamp(1, 8)
}

/// Map the multi-canvas stop reason onto an OpenAI finish_reason: hitting the
/// block budget or the ubatch guard is a truncation ("length"); a trim (EOG /
/// repetition cut) is a natural stop.
fn dg_finish_reason(stop: &str) -> &'static str {
    match stop {
        "blocks" | "ubatch" => "length",
        _ => "stop",
    }
}

/// A DiffusionGemma model wrapped for the serve path: the Phase 6 `DgChat`
/// (render + tokenize + multi-canvas denoise + detokenize).
pub struct DgServeRuntime {
    chat: crate::diffusion_gemma::chat::DgChat,
}

/// Whole-generation serialization for the DiffusionGemma serve lane. On CUDA
/// builds every dg request funnels through the process-global engine singleton
/// (`diffusion_gemma::cuda::ENGINE`), whose internal `Mutex` is held only per
/// kernel op — state like `Engine::last_logits` is handed BETWEEN ops within a
/// forward step, so two concurrent requests interleaving ops logically corrupt
/// each other's generations. This lane-global lock mirrors the gemma4 Cuda
/// pattern (`Gemma4ServeRuntime::Cuda`'s whole-decode `Mutex`): held for the
/// entire generate call, acquired INSIDE the `spawn_blocking` closure so a
/// dropped SSE stream (client disconnect drops the async frame, not the
/// blocking thread) can never release it mid-decode. Unconditional on CPU
/// builds too — harmless, and it keeps the invariant build-independent.
static DG_GENERATION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the dg whole-generation lock. Poisoning is handled the way the
/// gemma4 Cuda lane handles its runtime lock (`.expect`): a panicked
/// generation poisons the lane and later requests fail loudly (surfaced as a
/// generation_error by the handlers' panic paths) rather than decode against
/// possibly-corrupt engine state.
fn dg_generation_guard() -> std::sync::MutexGuard<'static, ()> {
    DG_GENERATION_LOCK.lock().expect("dg generation lock")
}

impl DgServeRuntime {
    fn load(path: &std::path::Path) -> std::result::Result<Self, BackendError> {
        Ok(Self {
            chat: crate::diffusion_gemma::chat::DgChat::load(path)?,
        })
    }
}

/// Resolve a DiffusionGemma serve runtime for the requested (or active) model id.
async fn resolve_dg_runtime(
    state: &AppState,
    model: &Option<String>,
) -> std::result::Result<Option<(String, Arc<DgServeRuntime>)>, Response> {
    let id = match model.clone() {
        Some(m) => m,
        None => match state.active_model_id.read().await.clone() {
            Some(m) => m,
            None => return Ok(None),
        },
    };
    if let Some(runtime) = state.dg_runtimes.read().await.get(&id).cloned() {
        return Ok(Some((id, runtime)));
    }
    // Loaded as a diffusion-gemma arch but no runtime → fail clearly rather than
    // letting the Llama path return its unsupported-architecture error.
    let needs_dg = state
        .loaded_models
        .read()
        .await
        .get(&id)
        .map(|m| is_dg_serve_arch(m.gguf.architecture().unwrap_or_default()))
        .unwrap_or(false);
    if needs_dg {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_ready",
            format!(
                "model '{id}' (DiffusionGemma) is loaded but its serve runtime is \
                 unavailable; set CAMELID_DG_SERVE=1 and reload the model"
            ),
            None,
        ));
    }
    Ok(None)
}

/// Load the DiffusionGemma serve runtime for a model id (blocking thread; the
/// GGUF is lazy-mmapped so this is seconds, not a full read).
async fn load_dg_serve_runtime(
    state: &AppState,
    id: &str,
    model_path: &std::path::Path,
) -> std::result::Result<(), BackendError> {
    let load_path = model_path.to_path_buf();
    let runtime = tokio::task::spawn_blocking(move || DgServeRuntime::load(&load_path))
        .await
        .map_err(|e| {
            BackendError::InvalidModelMetadata(format!("dg serve load task panicked: {e}"))
        })??;
    state
        .dg_runtimes
        .write()
        .await
        .insert(id.to_string(), Arc::new(runtime));
    tracing::info!(model = %id, "diffusion-gemma serve runtime loaded");
    Ok(())
}

/// The last user message — the DG chat template is single-turn (the
/// parity-proven reference path renders exactly one message).
fn dg_last_user_message(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| m.role.trim().eq_ignore_ascii_case("user"))
        .map(|m| m.content.clone())
}

/// EB sampler params for a serve request: request `seed` (reference default 0)
/// + the `CAMELID_DG_MAX_STEPS` override.
fn dg_serve_params(seed: Option<u64>) -> crate::diffusion_gemma::DgEbParams {
    let defaults = crate::diffusion_gemma::DgEbParams::default();
    crate::diffusion_gemma::DgEbParams {
        seed: seed.unwrap_or(0) as u32,
        max_steps: dg_serve_steps().unwrap_or(defaults.max_steps),
        ..defaults
    }
}

/// Non-streaming chat for a DiffusionGemma model: render the single-turn chat
/// template for the last user message, run the multi-canvas denoise loop,
/// detokenize. Minutes of compute — the client waits on one response.
async fn dg_chat_nonstreaming(
    id: String,
    runtime: Arc<DgServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let Some(user_msg) = dg_last_user_message(&messages) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "the DiffusionGemma serve lane needs at least one user message".to_string(),
            Some("messages"),
        );
    };
    let params = dg_serve_params(req.seed);
    let n_blocks = dg_serve_blocks();
    let rt = runtime.clone();
    let result = tokio::task::spawn_blocking(move || {
        // Serialize the WHOLE generation (see `DG_GENERATION_LOCK`): the CUDA
        // engine is process-global and its internal lock is per-kernel-op only.
        let _generation_guard = dg_generation_guard();
        let prompt_tokens = rt.chat.render_prompt(&user_msg).map(|v| v.len())?;
        rt.chat
            .generate(&user_msg, &params, n_blocks, 1100, |_, _, _, _| {})
            .map(|out| (prompt_tokens, out))
    })
    .await;
    let (prompt_tokens, (text, stop, ids)) = match result {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                e.to_string(),
                None,
            )
        }
        Err(e) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                format!("dg generation task panicked: {e}"),
                None,
            )
        }
    };
    let body = serde_json::json!({
        "id": "chatcmpl-diffusion-gemma",
        "object": "chat.completion",
        "created": unix_secs(),
        "model": id,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": dg_finish_reason(&stop),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": ids.len(),
            "total_tokens": prompt_tokens + ids.len(),
        },
        "camelid": { "lane": "diffusion_gemma", "stop": stop, "note": "single-turn template: last user message only" },
    });
    (StatusCode::OK, Json(body)).into_response()
}

/// Streaming chat for a DiffusionGemma model (SSE, OpenAI chunk shape): a role
/// chunk, then ONE content delta per committed denoise block (a block is
/// minutes of compute — SSE keep-alive comments cover the gaps), the
/// finish_reason chunk, optional usage, `[DONE]`.
async fn dg_chat_streaming(
    id: String,
    runtime: Arc<DgServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let Some(user_msg) = dg_last_user_message(&messages) else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "the DiffusionGemma serve lane needs at least one user message".to_string(),
            Some("messages"),
        );
    };
    let params = dg_serve_params(req.seed);
    let n_blocks = dg_serve_blocks();
    let include_usage = stream_options_include_usage(req.stream_options.as_ref());
    let created = unix_secs();

    /// (prompt_tokens, text, stop, response ids) from a finished generation.
    type DgDone = (usize, String, String, Vec<i32>);
    enum StreamItem {
        /// Live draft: the FULL argmax canvas after a denoise step (a
        /// diffusion answer exists in whole from step 0 and refines in
        /// place). Sent as a non-standard delta field; standard OpenAI
        /// clients ignore it.
        Preview(usize, i32, String),
        Block(Vec<i32>),
        Done(DgDone),
        Fail(String),
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<StreamItem>();
    let rt = runtime.clone();
    tokio::task::spawn_blocking(move || {
        // Serialize the WHOLE generation (see `DG_GENERATION_LOCK`). Acquired
        // here on the blocking thread — NOT in the async frame — so a dropped
        // SSE stream cannot release the guard while the decode is in flight.
        let _generation_guard = dg_generation_guard();
        let send_tx = tx.clone();
        let step_tx = tx.clone();
        let prompt_tokens = match rt.chat.render_prompt(&user_msg) {
            Ok(ids) => ids.len(),
            Err(e) => {
                let _ = tx.send(StreamItem::Fail(e.to_string()));
                return;
            }
        };
        let result = rt.chat.generate_live(
            &user_msg,
            &params,
            n_blocks,
            1100,
            |b, step, draft| {
                let _ = step_tx.send(StreamItem::Preview(b, step, draft));
            },
            |_b, committed| {
                let _ = send_tx.send(StreamItem::Block(committed.to_vec()));
            },
        );
        match result {
            Ok((text, stop, ids)) => {
                let _ = tx.send(StreamItem::Done((prompt_tokens, text, stop, ids)));
            }
            Err(e) => {
                let _ = tx.send(StreamItem::Fail(e.to_string()));
            }
        }
    });

    let rt = runtime.clone();
    let events = async_stream::stream! {
        let chunk = |delta: serde_json::Value, finish: Option<&str>| {
            serde_json::json!({
                "id": "chatcmpl-diffusion-gemma",
                "object": "chat.completion.chunk",
                "created": created,
                "model": id,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        };
        yield Ok::<Event, std::convert::Infallible>(
            Event::default().data(chunk(serde_json::json!({ "role": "assistant" }), None).to_string()),
        );

        // Incremental decode: the response is the concatenation of committed
        // blocks; decode the whole accumulated id list each block (cheap next
        // to a denoise) and emit only the new suffix.
        let mut all_ids: Vec<i32> = Vec::new();
        let mut emitted = 0usize;
        let mut final_state: Option<std::result::Result<DgDone, String>> = None;

        while let Some(item) = rx.recv().await {
            match item {
                StreamItem::Preview(b, step, draft) => {
                    // Live canvas frame: the whole forming answer, refreshed
                    // per denoise step. Non-standard field — OpenAI-shaped
                    // clients ignore it; aware UIs render the draft in place.
                    yield Ok(Event::default().data(
                        chunk(
                            serde_json::json!({
                                "camelid_canvas_preview": {
                                    "block": b,
                                    "step": step,
                                    "text": draft,
                                }
                            }),
                            None,
                        )
                        .to_string(),
                    ));
                }
                StreamItem::Block(committed) => {
                    all_ids.extend_from_slice(&committed);
                    let decoded = rt.chat.decode_response(&all_ids).unwrap_or_default();
                    if decoded.len() > emitted {
                        let delta_text = decoded[emitted..].to_string();
                        emitted = decoded.len();
                        yield Ok(Event::default().data(
                            chunk(serde_json::json!({ "content": delta_text }), None).to_string(),
                        ));
                    }
                }
                StreamItem::Done(done) => {
                    final_state = Some(Ok(done));
                    break;
                }
                StreamItem::Fail(e) => {
                    final_state = Some(Err(e));
                    break;
                }
            }
        }

        match final_state {
            Some(Ok((prompt_tokens, text, stop, ids))) => {
                // Any tail the block deltas missed (e.g. UTF-8 boundary).
                if text.len() > emitted {
                    yield Ok(Event::default().data(
                        chunk(serde_json::json!({ "content": text[emitted..].to_string() }), None).to_string(),
                    ));
                }
                yield Ok(Event::default().data(
                    chunk(serde_json::json!({}), Some(dg_finish_reason(&stop))).to_string(),
                ));
                if include_usage {
                    let usage = serde_json::json!({
                        "id": "chatcmpl-diffusion-gemma",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": id,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": ids.len(),
                            "total_tokens": prompt_tokens + ids.len(),
                        },
                    });
                    yield Ok(Event::default().data(usage.to_string()));
                }
            }
            Some(Err(e)) => {
                let err = serde_json::json!({ "error": { "message": e, "type": "generation_error" } });
                yield Ok(Event::default().data(err.to_string()));
            }
            None => {}
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    // A denoise block is minutes of silence; keep the SSE connection (and any
    // proxies) alive with comment frames.
    Sse::new(events)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(10))
                .text("dg-denoising"),
        )
        .into_response()
}

/// Streaming Gemma 4 chat (SSE). Mirrors the OpenAI `chat.completion.chunk`
/// shape: a role chunk, one content delta per generated token, a final
/// finish_reason chunk, then `[DONE]`. Generation runs on a blocking thread and
/// pushes deltas through an mpsc channel that this stream forwards.
async fn gemma4_chat_streaming(
    id: String,
    runtime: Arc<Gemma4ServeRuntime>,
    req: &ChatCompletionRequest,
) -> Response {
    let messages = req.messages.clone().unwrap_or_default();
    let prompt = gemma4_chat_prompt(&messages, req.camelid_enable_thinking.unwrap_or(false));
    let max_tokens = req.max_tokens.unwrap_or(256).min(4096) as usize;
    let created = unix_secs();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Result<String, String>>();
    tokio::task::spawn_blocking(move || {
        let send_tx = tx.clone();
        let result = runtime.generate_greedy_streaming(&prompt, max_tokens, |delta| {
            let _ = send_tx.send(Ok(delta.to_string()));
        });
        if let Err(e) = result {
            let _ = tx.send(Err(e.to_string()));
        }
    });

    let events = async_stream::stream! {
        // Lifecycle telemetry; a dropped stream (client disconnect) closes
        // the run via the guard's Drop.
        let mut telemetry_guard = Some(telemetry::RequestGuard::begin(gemma4_telemetry_start(
            &id,
            max_tokens as u32,
            true,
        )));
        let mut telemetry_tokens = 0usize;
        // Role chunk.
        let role = serde_json::json!({
            "id": "chatcmpl-gemma4",
            "object": "chat.completion.chunk",
            "created": created,
            "model": id,
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
        });
        yield Ok::<Event, std::convert::Infallible>(Event::default().data(role.to_string()));

        let mut errored = false;
        let mut channel_filter = Gemma4ChannelFilter::new();
        while let Some(item) = rx.recv().await {
            match item {
                Ok(delta) => {
                    // One delta per really-decoded token on this lane (the
                    // runtime invokes the callback per generated token), so
                    // this pulse maps 1:1 to real decode work.
                    telemetry_tokens += 1;
                    telemetry::emit(telemetry::Event::TokenDecoded {
                        token_id: None,
                        context_position: None,
                        layers_total: None,
                    });
                    // Suppress thinking-channel spans; an empty visible delta
                    // emits nothing.
                    let visible = channel_filter.filter(&delta);
                    if visible.is_empty() {
                        continue;
                    }
                    let chunk = serde_json::json!({
                        "id": "chatcmpl-gemma4",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": id,
                        "choices": [{ "index": 0, "delta": { "content": visible }, "finish_reason": null }],
                    });
                    yield Ok(Event::default().data(chunk.to_string()));
                }
                Err(e) => {
                    if let Some(guard) = telemetry_guard.take() {
                        guard.finish(gemma4_telemetry_error(e.clone()));
                    }
                    let err = serde_json::json!({ "error": { "message": e, "type": "generation_error" } });
                    yield Ok(Event::default().data(err.to_string()));
                    errored = true;
                    break;
                }
            }
        }

        if !errored {
            if let Some(guard) = telemetry_guard.take() {
                guard.finish(telemetry::RequestFinish {
                    status: "ok",
                    finish_reason: Some("stop".to_string()),
                    completion_tokens: telemetry_tokens,
                    ttft_ms: None,
                    decode_tps: None,
                    prefill_tps: None,
                    error: None,
                });
            }
            let done = serde_json::json!({
                "id": "chatcmpl-gemma4",
                "object": "chat.completion.chunk",
                "created": created,
                "model": id,
                "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            });
            yield Ok(Event::default().data(done.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(events).into_response()
}

async fn load_model_from_path_with_activation(
    state: &AppState,
    path: PathBuf,
    id: Option<String>,
    set_active: bool,
) -> Result<LoadedModel, BackendError> {
    if state.workspace_sessions.blocks_model_transition().await {
        return Err(BackendError::ModelOperationInProgress);
    }
    let _transition = state.model_transition.lock().await;
    let _reader = state.model_file_lifecycle.read().await;
    // Every load funnels through here, so resolve relative paths against the
    // configured models dir at this one chokepoint (absolute paths and
    // CWD-relative hits pass through untouched — see resolve_request_model_path).
    // Resolution runs BEFORE the idempotent fast path so a repeated relative
    // request compares equal to the resolved path the first load stored.
    let path = resolve_request_model_path(&state.models_dir, path)?;
    // Idempotent fast path: the same id already loaded from the same file
    // returns the existing record instead of re-running the full load pipeline
    // (an 8 GB row re-reads the whole file for its receipt otherwise; repeat
    // loads from smoke tooling were timing out on it).
    if let Some(requested_id) = id.as_deref() {
        let loaded = state.loaded_models.read().await;
        if let Some(existing) = loaded.get(requested_id) {
            if existing.path == path {
                let existing = existing.clone();
                drop(loaded);
                if set_active {
                    *state.active_model_id.write().await = Some(requested_id.to_string());
                }
                // Heal a missing gemma4 serve runtime: a client that
                // disconnects mid-load cancels the handler future AFTER the
                // loaded_models insert but BEFORE the runtime insert, and the
                // fast path would otherwise return a record whose chat routes
                // 503 forever.
                if gemma4_serve_enabled()
                    && model_family(&existing.gguf) == "gemma4"
                    && !state
                        .gemma4_runtimes
                        .read()
                        .await
                        .contains_key(requested_id)
                {
                    load_gemma4_serve_runtime(state, requested_id, &existing.path).await?;
                }
                return Ok(existing);
            }
        }
    }
    let mut gguf = read_metadata(&path)?;
    let outcome = plan_for_model(&path, &gguf, state.configured_threads);
    state.planner_env.apply(&outcome.env_updates);
    log_selected_execution_plan(&outcome.plan);
    let id = id
        .or_else(|| gguf.model_name().map(ToOwned::to_owned))
        .or_else(|| path.file_stem().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_else(|| "loaded-model".to_string());
    let llama_config_result = LlamaModelConfig::from_gguf(&gguf);
    // Some dense decoders (e.g. phi3) ship fused attn_qkv / gate-up tensors. Synthesize
    // the split tensors the binder + forward path expect (no-op for already-split rows),
    // so the fused layout becomes attemptable without touching the parity-gated path.
    if let Ok(config) = &llama_config_result {
        if let Err(err) = crate::model::expand_fused_dense_tensors(&mut gguf, config) {
            eprintln!("[camelid] fused-tensor expansion skipped: {err}");
        }
    }
    // Capture the exact typed blocker so a loaded-but-non-runnable model surfaces
    // WHY it fails closed (architecture not implemented, missing/invalid metadata,
    // DiffusionGemma redirect, â€¦) instead of silently sitting non-generative.
    let unsupported_runtime = match &llama_config_result {
        Err(
            err @ (BackendError::UnsupportedModelArchitecture(_)
            | BackendError::InvalidModelMetadata(_)
            | BackendError::UnsupportedGguf(_)),
        ) => Some(UnsupportedRuntimeSummary {
            code: backend_error_code(err),
            message: err.to_string(),
        }),
        _ => None,
    };
    let llama_config = llama_config_result.ok();
    let llama_tensors = llama_config
        .as_ref()
        .and_then(|config| LlamaTensorBinding::bind(&gguf, config).ok());
    let tokenizer_result = Tokenizer::from_gguf(&gguf);
    let tokenizer = tokenizer_state_from_result(tokenizer_result.as_ref());
    let tokenizer_runtime = tokenizer_result.ok().map(Arc::new);
    // Hash the exact GGUF bytes once at load time so receipts can name the
    // lane without re-hashing per request.
    let gguf_sha256 = receipt::sha256_file_hex(&path).map_err(|err| match err {
        receipt::ReceiptError::Io { path, source } => BackendError::Io { path, source },
        other => BackendError::InvalidModelMetadata(other.to_string()),
    })?;
    let tokenizer_kind = tokenizer_runtime
        .as_ref()
        .map(|tokenizer| tokenizer.model.as_summary_model());
    let lane = LaneIdentity::capture(&id, &path, &gguf, tokenizer_kind, gguf_sha256);
    let loaded = LoadedModel {
        id: id.clone(),
        path,
        gguf,
        llama_config,
        llama_tensors,
        unsupported_runtime,
        tokenizer,
        tokenizer_runtime,
        lane,
    };

    state
        .loaded_models
        .write()
        .await
        .insert(id.clone(), loaded.clone());
    state
        .execution_plans
        .write()
        .await
        .insert(id.clone(), outcome.plan);
    state
        .model_last_used
        .write()
        .await
        .insert(id.clone(), std::time::Instant::now());
    if set_active {
        *state.active_model_id.write().await = Some(id.clone());
        clear_prompt_prefix_cache(state);
    }

    // Gemma 4 serve path (additive, gated by CAMELID_GEMMA4_SERVE): load a
    // serve runtime so /v1/chat can route to it. Fail clearly on error â€” never
    // silently fall back to the Llama path (which would produce garbage here).
    if gemma4_serve_enabled() && model_family(&loaded.gguf) == "gemma4" {
        load_gemma4_serve_runtime(state, &id, &loaded.path).await?;
    }

    // Runnable serve path (additive, gated by CAMELID_RUNNABLE_SERVE): load a
    // runnable-lane runtime (qwen35/Ornith) so /v1/chat can route to it.
    if runnable_serve_enabled()
        && is_runnable_serve_arch(loaded.gguf.architecture().unwrap_or_default())
    {
        load_runnable_serve_runtime(state, &id, &loaded.path).await?;
    }

    // DiffusionGemma serve path (additive, gated by CAMELID_DG_SERVE): the AR
    // engine keeps failing closed for this arch (`unsupported_runtime` carries
    // the dedicated-lane redirect); the bridge loads the Phase 6 DgChat runtime
    // so /v1/chat routes to the diffusion lane instead.
    if dg_serve_enabled() && is_dg_serve_arch(loaded.gguf.architecture().unwrap_or_default()) {
        load_dg_serve_runtime(state, &id, &loaded.path).await?;
    }

    Ok(loaded)
}

/// Load (or reload) the gemma4 serve runtime for a model id. With
/// CAMELID_GEMMA4_WORKER + CAMELID_GEMMA4_SPLIT set, the runtime is the
/// distributed layer-sharding lane (master shard locally, tail layers on the
/// worker); a half-configured or unreachable pair fails the load.
async fn load_gemma4_serve_runtime(
    state: &AppState,
    id: &str,
    model_path: &std::path::Path,
) -> std::result::Result<(), BackendError> {
    let distributed =
        gemma4_distributed_serve_config().map_err(BackendError::InvalidModelMetadata)?;
    let load_path = model_path.to_path_buf();
    let runtime = tokio::task::spawn_blocking(move || match distributed {
        Some((worker_addr, split)) => crate::gemma4_distributed::Gemma4DistributedRuntime::connect(
            &load_path,
            &worker_addr,
            split,
        )
        .map(Gemma4ServeRuntime::Distributed),
        None => {
            #[cfg(feature = "cuda")]
            {
                if gemma4_cuda_enabled() {
                    // KV-cache context window. 4096 fits the 6 GB card (the attention
                    // kernel's shared memory is (2*head_dim + max_positions)*4 bytes, well
                    // under 48 KB, and the f16 KV adds only ~100-200 MB) and gives real
                    // multi-turn headroom; overflow past it is guarded in the runtime.
                    return crate::gemma4_runtime::Gemma4CudaResident::load(&load_path, 4096)
                        .map(|r| Gemma4ServeRuntime::Cuda(std::sync::Mutex::new(r)));
                }
            }
            crate::gemma4_runtime::Gemma4Runtime::load(&load_path).map(Gemma4ServeRuntime::Local)
        }
    })
    .await
    .map_err(|e| {
        BackendError::InvalidModelMetadata(format!("gemma4 runtime load task panicked: {e}"))
    })??;
    let lane = match &runtime {
        Gemma4ServeRuntime::Local(_) => "local",
        Gemma4ServeRuntime::Distributed(_) => "distributed",
        #[cfg(feature = "cuda")]
        Gemma4ServeRuntime::Cuda(_) => "cuda",
    };
    state
        .gemma4_runtimes
        .write()
        .await
        .insert(id.to_string(), Arc::new(runtime));
    tracing::info!(model = %id, lane, "gemma4 runtime loaded for serve path");
    Ok(())
}

fn log_selected_execution_plan(plan: &ExecutionPlan) {
    tracing::info!(
        profile=?plan.profile,
        platform=%plan.platform_label,
        cpu_model=%plan.cpu_model,
        cpu_features=?plan.cpu_features,
        model=%plan.exact_model_row,
        support_level=%plan.support_level,
        backend=%plan.selected_backend,
        q8_path=%plan.selected_q8_path,
        prefill_path=%plan.prefill_path,
        prefill_runtime_policy=%plan.prefill_runtime_policy,
        decode_path=%plan.decode_path,
        threads=plan.thread_count,
        diagnostics=%plan.diagnostics_status,
        fallback_path=%plan.fallback_path,
        reasons=?plan.reasons,
        "Camelid execution plan selected"
    );
}

#[derive(Debug, Deserialize)]
pub struct UnloadModelRequest {
    pub id: Option<String>,
}

async fn unload_model(
    State(state): State<AppState>,
    payload: Option<Json<UnloadModelRequest>>,
) -> Response {
    if state.workspace_sessions.blocks_model_transition().await {
        return api_error(
            StatusCode::CONFLICT,
            "model_operation_in_progress",
            BackendError::ModelOperationInProgress.to_string(),
            None,
        );
    }
    let _transition = state.model_transition.lock().await;
    let _exclusive = state.model_file_lifecycle.write().await;
    let model_id = if let Some(Json(req)) = payload {
        req.id
    } else {
        None
    };

    let target_id = if let Some(id) = model_id {
        Some(id)
    } else {
        state.active_model_id.read().await.clone()
    };

    if let Err(resp) = release_model(&state, target_id).await {
        return *resp;
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Drop a model (`Some(id)`) or every model (`None`) from the registries and
/// release the GPU state the engine holds for it.
///
/// Shared by `/api/models/unload` and the `replace` load path so both free the
/// SAME things. That sharing is load-bearing, not tidiness: the registry clears
/// below are CPU-side only, and the resident decode engine keeps its weights in
/// process-global caches. Skipping `reset_resident_caches` leaves ~4.7 GB parked
/// on the device, which starves the next model into an NVIDIA sysmem spill and
/// makes decode ~20x slower — so a `replace` that hand-rolled the teardown would
/// silently reintroduce exactly the bug this reset exists to prevent.
///
/// The caller must already hold the model-transition lock and an exclusive
/// `model_file_lifecycle` guard. `Err` carries a ready-to-return response.
async fn release_model(state: &AppState, target: Option<String>) -> Result<(), Box<Response>> {
    if let Some(id) = target {
        state.loaded_models.write().await.remove(&id);
        state.gemma4_runtimes.write().await.remove(&id);
        state.runnable_runtimes.write().await.remove(&id);
        state.dg_runtimes.write().await.remove(&id);
        state.execution_plans.write().await.remove(&id);
        state.cached_weights.write().await.remove(&id);
        state.model_last_used.write().await.remove(&id);

        let mut active = state.active_model_id.write().await;
        if active.as_ref() == Some(&id) {
            *active = state.loaded_models.read().await.keys().next().cloned();
        }
    } else {
        state.loaded_models.write().await.clear();
        state.gemma4_runtimes.write().await.clear();
        state.runnable_runtimes.write().await.clear();
        state.dg_runtimes.write().await.clear();
        state.execution_plans.write().await.clear();
        state.cached_weights.write().await.clear();
        state.model_last_used.write().await.clear();
        *state.active_model_id.write().await = None;
    }

    clear_prompt_prefix_cache(state);
    // Free the GPU VRAM held by the resident decode engine. The clears above only drop
    // the CPU-side registries; the Llama resident engine lives in process-global caches
    // (see inference::reset_resident_caches) that unload never touched, so its ~4.7 GB
    // stayed on the device and starved the next model into a host-RAM spill (NVIDIA
    // sysmem fallback), making decode ~20x slower. (A gemma4 CUDA runtime's VRAM is
    // freed by dropping it from gemma4_runtimes above.)
    //
    // The reset mutates engine-owned GPU state, so it runs as an ENGINE JOB —
    // it can never race a decode. A failed post is surfaced, never skipped
    // silently (a skipped reset is the 20x-slowdown VRAM leak all over again).
    if let Err(err) = state
        .engine
        .run_exclusive(crate::inference::reset_resident_caches)
        .await
    {
        return Err(engine_post_error_response(err));
    }
    Ok(())
}

async fn current_model(State(state): State<AppState>) -> Response {
    let active_id = state.active_model_id.read().await;
    if let Some(id) = active_id.as_ref() {
        if let Some(model) = state.loaded_models.read().await.get(id).cloned() {
            return (StatusCode::OK, Json(model)).into_response();
        }
    }
    api_error(
        StatusCode::NOT_FOUND,
        "model_not_loaded",
        BackendError::ModelNotLoaded.to_string(),
        None,
    )
}

#[derive(Debug, Serialize)]
struct ModelVerificationStatus {
    model_id: String,
    gguf_sha256: String,
    eligible: bool,
    profile_id: Option<String>,
    report: Option<crate::verify::VerificationReport>,
}

async fn active_model_for_verification(state: &AppState) -> Result<LoadedModel, Response> {
    let Some(active_id) = state.active_model_id.read().await.clone() else {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "model_not_loaded",
            BackendError::ModelNotLoaded.to_string(),
            None,
        ));
    };
    state
        .loaded_models
        .read()
        .await
        .get(&active_id)
        .cloned()
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                "model_not_loaded",
                BackendError::ModelNotLoaded.to_string(),
                None,
            )
        })
}

async fn verification_status_for_model(
    state: &AppState,
    model: &LoadedModel,
) -> Result<ModelVerificationStatus, String> {
    let profile = crate::verify::profile_for_sha256(&model.lane.gguf_sha256)?;
    let report = state
        .verification_reports
        .read()
        .await
        .get(&model.lane.gguf_sha256)
        .cloned();
    Ok(ModelVerificationStatus {
        model_id: model.id.clone(),
        gguf_sha256: model.lane.gguf_sha256.clone(),
        eligible: profile.is_some(),
        profile_id: profile.map(|profile| profile.profile_id),
        report,
    })
}

async fn model_verification_status(State(state): State<AppState>) -> Response {
    let model = match active_model_for_verification(&state).await {
        Ok(model) => model,
        Err(response) => return response,
    };
    match verification_status_for_model(&state, &model).await {
        Ok(status) => (StatusCode::OK, Json(status)).into_response(),
        Err(err) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "verification_profile_invalid",
            err,
            None,
        ),
    }
}

async fn verify_current_model(State(state): State<AppState>) -> Response {
    let model = match active_model_for_verification(&state).await {
        Ok(model) => model,
        Err(response) => return response,
    };
    let profile = match crate::verify::profile_for_sha256(&model.lane.gguf_sha256) {
        Ok(Some(profile)) => profile,
        Ok(None) => {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "verification_profile_unavailable",
                "no built-in verification profile matches the active model's exact GGUF hash"
                    .to_string(),
                Some("model"),
            )
        }
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "verification_profile_invalid",
                err,
                None,
            )
        }
    };
    let replay = match replay_loaded_receipt_request(&state, &model.id, &profile.request).await {
        Ok(replay) => replay,
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "verification_replay_failed",
                err,
                None,
            )
        }
    };
    let active_id = state.active_model_id.read().await.clone();
    let active_hash = match active_id.as_deref() {
        Some(id) if id == model.id => state
            .loaded_models
            .read()
            .await
            .get(id)
            .map(|loaded| loaded.lane.gguf_sha256.clone()),
        _ => None,
    };
    if replay.lane.gguf_sha256 != profile.model.gguf_sha256
        || active_hash.as_deref() != Some(profile.model.gguf_sha256.as_str())
    {
        return api_error(
            StatusCode::CONFLICT,
            "model_changed_during_verification",
            "the active model changed while verification was running; no result was stored"
                .to_string(),
            Some("model"),
        );
    }
    let report = match crate::verify::evaluate(&profile, &replay.result) {
        Ok(report) => report,
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "verification_report_failed",
                err.to_string(),
                None,
            )
        }
    };
    state
        .verification_reports
        .write()
        .await
        .insert(model.lane.gguf_sha256.clone(), report.clone());
    (StatusCode::OK, Json(report)).into_response()
}

async fn model_metadata(State(state): State<AppState>) -> Response {
    let active_id = state.active_model_id.read().await;
    if let Some(id) = active_id.as_ref() {
        if let Some(model) = state.loaded_models.read().await.get(id) {
            return (StatusCode::OK, Json(&model.gguf)).into_response();
        }
    }
    api_error(
        StatusCode::NOT_FOUND,
        "model_not_loaded",
        BackendError::ModelNotLoaded.to_string(),
        None,
    )
}

async fn model_tokenizer(State(state): State<AppState>) -> Response {
    let active_id = state.active_model_id.read().await;
    if let Some(id) = active_id.as_ref() {
        if let Some(model) = state.loaded_models.read().await.get(id) {
            match &model.tokenizer {
                TokenizerLoadState::Available(summary) => {
                    return (StatusCode::OK, Json(summary)).into_response();
                }
                TokenizerLoadState::Unavailable { code, message } => {
                    return api_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        code,
                        message.clone(),
                        None,
                    );
                }
            }
        }
    }
    api_error(
        StatusCode::NOT_FOUND,
        "model_not_loaded",
        BackendError::ModelNotLoaded.to_string(),
        None,
    )
}

async fn get_or_load_model(
    state: &AppState,
    model_id: Option<&str>,
) -> Result<LoadedModel, Response> {
    let loaded_models = state.loaded_models.read().await;

    let target_id = if let Some(id) = model_id {
        id.to_string()
    } else if let Some(active) = state.active_model_id.read().await.as_ref() {
        active.clone()
    } else if loaded_models.len() == 1 {
        loaded_models.keys().next().unwrap().clone()
    } else {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "model_not_loaded",
            BackendError::ModelNotLoaded.to_string(),
            None,
        ));
    };

    if let Some(loaded) = loaded_models.get(&target_id) {
        state
            .model_last_used
            .write()
            .await
            .insert(target_id.clone(), std::time::Instant::now());
        *state.active_model_id.write().await = Some(target_id.clone());
        return Ok(loaded.clone());
    }

    for (id, loaded) in loaded_models.iter() {
        if id == &target_id
            || loaded.id == target_id
            || loaded
                .path
                .file_name()
                .is_some_and(|f| f.to_string_lossy() == target_id)
        {
            state
                .model_last_used
                .write()
                .await
                .insert(id.clone(), std::time::Instant::now());
            *state.active_model_id.write().await = Some(id.clone());
            return Ok(loaded.clone());
        }
    }

    // Check if the model can be loaded from disk
    let path = resolve_model_path(&state.models_dir, &target_id);
    if let Some(path) = path {
        if path.exists() {
            drop(loaded_models);
            match load_model_from_path(state, path, Some(target_id.clone())).await {
                Ok(loaded) => return Ok(loaded),
                Err(err) => {
                    return Err(api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "model_load_failed",
                        format!("Failed to load model {target_id} on-demand: {err}"),
                        None,
                    ))
                }
            }
        }
    }

    // If it doesn't exist on disk, check if any models are loaded
    if loaded_models.is_empty() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "model_not_loaded",
            BackendError::ModelNotLoaded.to_string(),
            None,
        ));
    }

    Err(api_error(
        StatusCode::NOT_FOUND,
        "model_not_found",
        format!("Requested model '{target_id}' is not loaded or could not be found on disk"),
        Some("model"),
    ))
}

fn resolve_model_path(models_dir: &std::path::Path, model_id: &str) -> Option<PathBuf> {
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Some(path);
    }

    let local_path = models_dir.join(model_id);
    if local_path.exists() {
        return Some(local_path);
    }

    for item in curated_catalog() {
        if item.catalog_id == model_id || item.filename == model_id {
            let cat_path = models_dir.join(item.filename);
            return Some(cat_path);
        }
    }

    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(stem) = p.file_stem() {
                    if stem.to_string_lossy() == model_id {
                        return Some(p);
                    }
                }
                if let Some(name) = p.file_name() {
                    if name.to_string_lossy() == model_id {
                        return Some(p);
                    }
                }
            }
        }
    }

    None
}

async fn load_weights_lru(
    state: &AppState,
    model: &LoadedModel,
    binding: &LlamaTensorBinding,
) -> Result<Arc<LlamaLoadedWeights>, Response> {
    {
        let cached = state.cached_weights.read().await;
        if let Some(weights) = cached.get(&model.id) {
            state
                .model_last_used
                .write()
                .await
                .insert(model.id.clone(), std::time::Instant::now());
            return Ok(weights.clone());
        }
    }

    let estimated_bytes = guard_cpu_weight_materialization_budget(binding).map_err(|err| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "cpu_weight_materialization_exceeds_budget",
            err.to_string(),
            Some("model"),
        )
    })?;

    let limit_bytes = cpu_weight_materialization_limit_bytes().unwrap_or(u64::MAX);

    loop {
        let loaded = state.loaded_models.read().await;
        let cached = state.cached_weights.read().await;

        let mut current_sum = 0u64;
        for (id, _) in cached.iter() {
            if id != &model.id {
                if let Some(m) = loaded.get(id) {
                    if let Some(b) = m.llama_tensors.as_ref() {
                        if let Ok(bytes) = estimate_cpu_weight_materialization_bytes(b) {
                            current_sum += bytes;
                        }
                    }
                }
            }
        }

        if current_sum + estimated_bytes <= limit_bytes {
            break;
        }

        let last_used = state.model_last_used.read().await;
        let mut lru_id: Option<String> = None;
        let mut oldest_time = std::time::Instant::now();

        for (id, _) in cached.iter() {
            if id != &model.id {
                let time = last_used
                    .get(id)
                    .cloned()
                    .unwrap_or_else(std::time::Instant::now);
                if time < oldest_time {
                    oldest_time = time;
                    lru_id = Some(id.clone());
                }
            }
        }

        drop(cached);
        drop(loaded);
        drop(last_used);

        if let Some(evict_id) = lru_id {
            tracing::info!(model=%evict_id, "LRU evicting weights of model to stay under budget");
            let mut cached_write = state.cached_weights.write().await;
            cached_write.remove(&evict_id);
        } else {
            break;
        }
    }

    let store = TensorStore::open(&model.path, &model.gguf);
    let range = if let Some(&(layer_start, layer_end)) = crate::distributed::DISTRIBUTED_RANGE.get()
    {
        tracing::info!(
            "API loader running in distributed coordinator mode; loading layers {}..{}",
            layer_start,
            layer_end
        );
        Some(layer_start..layer_end)
    } else {
        None
    };
    let weights = Arc::new(
        LlamaLoadedWeights::load(&store, binding, range).map_err(|err| {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "loaded_cpu_weights_unavailable",
                err.to_string(),
                Some("model"),
            )
        })?,
    );

    state
        .cached_weights
        .write()
        .await
        .insert(model.id.clone(), weights.clone());
    state
        .model_last_used
        .write()
        .await
        .insert(model.id.clone(), std::time::Instant::now());

    Ok(weights)
}

async fn tokenizer_encode(
    State(state): State<AppState>,
    payload: std::result::Result<Json<TokenizerEncodeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "malformed_json",
                err.to_string(),
                None,
            )
        }
    };
    let text = match req.text {
        Some(text) => text,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "missing_tokenizer_text",
                "tokenizer encode request requires a text field".to_string(),
                Some("text"),
            )
        }
    };
    let tokenizer = match loaded_tokenizer(&state).await {
        Ok(tokenizer) => tokenizer,
        Err(response) => return response,
    };
    match tokenizer.encode(
        &text,
        req.add_special.unwrap_or(true),
        req.parse_special.unwrap_or(false),
    ) {
        Ok(tokens) => (
            StatusCode::OK,
            Json(TokenizerEncodeResponse {
                token_count: tokens.len(),
                tokens,
            }),
        )
            .into_response(),
        Err(err) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "tokenization_failed",
            err.to_string(),
            Some("text"),
        ),
    }
}

async fn tokenizer_decode(
    State(state): State<AppState>,
    payload: std::result::Result<Json<TokenizerDecodeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "malformed_json",
                err.to_string(),
                None,
            )
        }
    };
    let tokens = match req.tokens {
        Some(tokens) => tokens,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "missing_tokenizer_tokens",
                "tokenizer decode request requires a tokens field".to_string(),
                Some("tokens"),
            )
        }
    };
    let token_count = tokens.len();
    let tokenizer = match loaded_tokenizer(&state).await {
        Ok(tokenizer) => tokenizer,
        Err(response) => return response,
    };
    match tokenizer.decode(&tokens, req.remove_special.unwrap_or(true)) {
        Ok(text) => (
            StatusCode::OK,
            Json(TokenizerDecodeResponse { text, token_count }),
        )
            .into_response(),
        Err(err) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "token_decode_failed",
            err.to_string(),
            Some("tokens"),
        ),
    }
}

async fn llama_server_tokenize(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerTokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    if !req.unsupported_fields.is_empty() {
        let mut fields = req
            .unsupported_fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        fields.sort_unstable();
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "/tokenize unsupported request field(s): {}",
                fields.join(", ")
            ),
            Some("request"),
        );
    }
    let with_pieces = req.with_pieces.unwrap_or(false);
    let content = match req.content {
        Some(content) => content,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "missing_tokenizer_content",
                "/tokenize request requires a content field".to_string(),
                Some("content"),
            )
        }
    };
    let tokenizer = match loaded_tokenizer(&state).await {
        Ok(tokenizer) => tokenizer,
        Err(response) => return response,
    };
    match tokenizer.encode(
        &content,
        req.add_special.unwrap_or(false),
        req.parse_special.unwrap_or(true),
    ) {
        Ok(tokens) => match llama_server_tokenize_tokens(&tokenizer, tokens, with_pieces) {
            Ok(tokens) => {
                (StatusCode::OK, Json(LlamaServerTokenizeResponse { tokens })).into_response()
            }
            Err(err) => api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "token_piece_failed",
                err.to_string(),
                Some("tokens"),
            ),
        },
        Err(err) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "tokenization_failed",
            err.to_string(),
            Some("content"),
        ),
    }
}

fn llama_server_tokenize_tokens(
    tokenizer: &Tokenizer,
    token_ids: Vec<u32>,
    with_pieces: bool,
) -> std::result::Result<Vec<LlamaServerTokenizeToken>, BackendError> {
    if !with_pieces {
        return Ok(token_ids
            .into_iter()
            .map(LlamaServerTokenizeToken::Id)
            .collect());
    }

    token_ids
        .into_iter()
        .map(|id| {
            let piece = llama_server_token_piece(tokenizer, id)?;
            Ok(LlamaServerTokenizeToken::Piece(LlamaServerTokenPiece {
                id,
                piece,
            }))
        })
        .collect()
}

fn llama_server_token_piece(
    tokenizer: &Tokenizer,
    token_id: u32,
) -> std::result::Result<LlamaServerTokenPieceValue, BackendError> {
    match tokenizer.decode(&[token_id], false) {
        Ok(text) => Ok(LlamaServerTokenPieceValue::Text(text)),
        Err(err) => {
            let Some(raw_piece) = tokenizer.token_text(Some(token_id)) else {
                return Err(err);
            };
            let Some(byte) = parse_llama_server_byte_token(raw_piece) else {
                return Err(err);
            };
            Ok(LlamaServerTokenPieceValue::Bytes(vec![byte]))
        }
    }
}

fn parse_llama_server_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

async fn llama_server_detokenize(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerDetokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    if !req.unsupported_fields.is_empty() {
        let mut fields = req
            .unsupported_fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        fields.sort_unstable();
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "/detokenize unsupported request field(s): {}",
                fields.join(", ")
            ),
            Some("request"),
        );
    }
    let tokens = match req.tokens {
        Some(tokens) => tokens,
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "missing_tokenizer_tokens",
                "/detokenize request requires a tokens field".to_string(),
                Some("tokens"),
            )
        }
    };
    let tokenizer = match loaded_tokenizer(&state).await {
        Ok(tokenizer) => tokenizer,
        Err(response) => return response,
    };
    match tokenizer.decode(&tokens, false) {
        Ok(content) => (
            StatusCode::OK,
            Json(LlamaServerDetokenizeResponse { content }),
        )
            .into_response(),
        Err(err) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "token_decode_failed",
            err.to_string(),
            Some("tokens"),
        ),
    }
}

async fn llama_server_apply_template(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerApplyTemplateRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    if !req.unsupported_fields.is_empty() {
        let mut fields = req
            .unsupported_fields
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        fields.sort_unstable();
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "/apply-template unsupported request field(s): {}",
                fields.join(", ")
            ),
            Some("request"),
        );
    }
    let messages = match req.messages {
        Some(messages) if !messages.is_empty() => messages,
        Some(_) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "empty_chat_messages",
                "/apply-template messages must contain at least one chat message".to_string(),
                Some("messages"),
            )
        }
        None => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "missing_chat_messages",
                "/apply-template request requires a messages field".to_string(),
                Some("messages"),
            )
        }
    };
    if let Err(response) = validate_chat_messages(&messages) {
        return *response;
    }

    let model = match get_or_load_model(&state, None).await {
        Ok(model) => model,
        Err(response) => return response,
    };
    let tokenizer = match model.tokenizer_runtime.clone() {
        Some(tokenizer) => tokenizer,
        None => match Tokenizer::from_gguf(&model.gguf) {
            Ok(tokenizer) => Arc::new(tokenizer),
            Err(err) => {
                return api_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    tokenizer_error_code(&err),
                    err.to_string(),
                    None,
                )
            }
        },
    };
    let rendered = match render_chat_prompt_for_tokenization_for_model_result(
        &messages,
        &tokenizer,
        Some(&model.id),
        // Template-preview endpoint: render the deterministic thinking-disabled
        // shape (the generation path honors camelid_enable_thinking per request).
        false,
    ) {
        Ok(rendered) => rendered,
        Err(err) => {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "unsupported_chat_template",
                format!(
                    "chat template rendering failed for loaded model {:?}: {err}",
                    model.id
                ),
                Some("messages"),
            )
        }
    };

    (
        StatusCode::OK,
        Json(LlamaServerApplyTemplateResponse {
            prompt: rendered.text,
        }),
    )
        .into_response()
}

async fn v1_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let loaded = state.loaded_models.read().await;
    let data = loaded.values().map(model_list_item).collect();
    Json(ModelListResponse {
        object: "list",
        data,
    })
}

async fn v1_model(AxumPath(model_id): AxumPath<String>, State(state): State<AppState>) -> Response {
    let loaded = state.loaded_models.read().await;
    match loaded.get(&model_id) {
        Some(model) => (StatusCode::OK, Json(model_list_item(model))).into_response(),
        None => api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("model '{model_id}' is not loaded"),
            Some("model"),
        ),
    }
}

fn model_list_item(model: &LoadedModel) -> ModelListItem {
    ModelListItem {
        id: model.id.clone(),
        object: "model",
        created: 0,
        owned_by: "camelid",
        meta: model_list_meta(model),
    }
}

fn model_list_meta(model: &LoadedModel) -> Option<ModelListMeta> {
    let config = model.llama_config.as_ref()?;
    Some(ModelListMeta {
        n_vocab: config.vocab_size,
        n_ctx_train: Some(config.context_length),
        n_embd: Some(config.embedding_length),
        n_params: model_parameter_count(&model.gguf),
        size: std::fs::metadata(&model.path)
            .ok()
            .map(|metadata| metadata.len()),
        file_type: config.file_type,
    })
}

fn model_parameter_count(gguf: &GgufFile) -> Option<u64> {
    gguf.tensors.iter().try_fold(0u64, |total, tensor| {
        let tensor_params = tensor
            .dimensions
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(*dim))?;
        total.checked_add(tensor_params)
    })
}

async fn generation_sessions(State(state): State<AppState>) -> Json<GenerationSessionListResponse> {
    let sessions = state.generation_sessions.read().await;
    Json(GenerationSessionListResponse {
        object: "list",
        data: sessions.values().cloned().collect(),
    })
}

async fn create_generation_session(
    State(state): State<AppState>,
    payload: std::result::Result<Json<GenerationSessionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    match validate_generation_request(&state, req).await {
        Ok(summary) => {
            state
                .generation_sessions
                .write()
                .await
                .insert(summary.id.clone(), summary.clone());
            (StatusCode::CREATED, Json(summary)).into_response()
        }
        Err(response) => response,
    }
}

async fn preflight_generation(
    State(state): State<AppState>,
    payload: std::result::Result<Json<GenerationSessionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    match validate_generation_request(&state, req).await {
        Ok(summary) => Json(summary).into_response(),
        Err(response) => response,
    }
}

/// Every choice of an n>1 request, prepared upfront (KV caches allocate
/// lazily, so n prepared sessions cost weights-Arc clones, not n KV buffers),
/// sharing ONE cancel signal so a dropped handler stops whichever choice is
/// decoding.
struct PreparedMultiChoice {
    prepared: Vec<PreparedGeneration>,
    cancel: GenerationCancel,
    model_id: String,
    prompt_token_count: usize,
}

async fn prepare_multi_choice(
    state: &AppState,
    req: GenerationSessionRequest,
    n_choices: u32,
    base_seed: u64,
) -> std::result::Result<PreparedMultiChoice, Response> {
    let cancel = GenerationCancel::unbounded();
    let mut prepared_choices = Vec::with_capacity(n_choices as usize);
    for index in 0..n_choices {
        let mut req_choice = req.clone();
        // Each choice is its own generation with a distinct, reproducible seed
        // (base seed offset by the choice index), so n>1 yields independent
        // samples that still reproduce exactly for a fixed request seed.
        req_choice.n = None;
        req_choice.seed = Some(base_seed.wrapping_add(u64::from(index)));
        let mut prepared = prepare_generation(state, req_choice).await?;
        prepared.cancel = cancel.clone();
        prepared_choices.push(prepared);
    }
    let first = prepared_choices
        .first()
        .expect("n_choices >= 1 guarantees at least one prepared choice");
    let model_id = first.model_id.clone();
    let prompt_token_count = first.token_ids.len();
    Ok(PreparedMultiChoice {
        prepared: prepared_choices,
        cancel,
        model_id,
        prompt_token_count,
    })
}

/// Run every choice sequentially inside ONE engine job. The transitional
/// streaming bridge lock is held across ALL choices, preserving the
/// pre-inversion "lock spans every choice" coverage.
async fn run_multi_choice_on_engine(
    state: &AppState,
    choices: PreparedMultiChoice,
) -> std::result::Result<Vec<GeneratedText>, Box<Response>> {
    let prepared_choices = choices.prepared;
    match state
        .engine
        .run_exclusive(move || {
            let mut generated = Vec::with_capacity(prepared_choices.len());
            for prepared in prepared_choices {
                generated.push(run_decode_job_serialized(prepared)?);
            }
            Ok(generated)
        })
        .await
    {
        Ok(result) => result,
        Err(err) => Err(engine_post_error_response(err)),
    }
}

/// Non-streaming multi-choice (`n` > 1) text completion. Mirrors
/// `chat_completions_multi_choice`: each choice is an independent, reproducibly
/// seeded generation; `camelid` diagnostics mirror the first choice; usage counts
/// the prompt once and sums completion tokens. Streaming and receipts are rejected
/// upstream. All n decodes run inside ONE engine job (one queue slot), so no
/// other request interleaves between choices.
async fn completions_multi_choice(
    state: &AppState,
    req: GenerationSessionRequest,
    n_choices: u32,
) -> Response {
    let base_seed = req.seed.unwrap_or(0);
    let prepared_choices = match prepare_multi_choice(state, req, n_choices, base_seed).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let _cancel_on_drop = CancelOnDrop(prepared_choices.cancel.token.clone());
    let model_id = prepared_choices.model_id.clone();
    let prompt_token_count = prepared_choices.prompt_token_count;
    let results = match run_multi_choice_on_engine(state, prepared_choices).await {
        Ok(results) => results,
        Err(response) => return *response,
    };
    let mut choices = Vec::with_capacity(results.len());
    let mut total_completion_tokens = 0usize;
    let mut first_diagnostics: Option<GenerationDiagnostics> = None;
    for (index, generated) in results.into_iter().enumerate() {
        let finish_reason = generated.finish_reason;
        total_completion_tokens += generated.completion_tokens;
        let text = generated.text.clone();
        if index == 0 {
            first_diagnostics = Some(GenerationDiagnostics {
                prompt_token_ids: generated.prompt_token_ids,
                generated_token_ids: generated.generated_token_ids,
                dense_metadata: generated.dense_metadata,
                top_logits: generated.top_logits,
                step_top_logits: generated.step_top_logits,
                output_projection: generated.output_projection,
                dense: generated.dense,
                dense_diagnostic_generated_index: generated.dense_diagnostic_generated_index,
                timings_ms: generated.timings,
            });
        }
        choices.push(CompletionChoice {
            index: index as u32,
            text,
            finish_reason,
            // Logprobs are rejected upstream for n>1.
            logprobs: None,
        });
    }
    let camelid =
        first_diagnostics.expect("n_choices >= 1 guarantees the first choice produced diagnostics");
    (
        StatusCode::OK,
        Json(CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: 0,
            model: model_id,
            choices,
            usage: CompletionUsage {
                prompt_tokens: prompt_token_count,
                completion_tokens: total_completion_tokens,
                total_tokens: prompt_token_count + total_completion_tokens,
            },
            camelid,
            camelid_receipt: None,
        }),
    )
        .into_response()
}

async fn completions(
    State(state): State<AppState>,
    payload: std::result::Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    // Gemma 4 serve path (additive, gated by CAMELID_GEMMA4_SERVE): raw greedy
    // completion against the gemma4 runtime, mirroring the chat short-circuit.
    match resolve_gemma4_runtime_for_model(&state, &req.model).await {
        Ok(Some((id, runtime))) => {
            return if req.stream.unwrap_or(false) {
                gemma4_completion_streaming(id, runtime, &req).await
            } else {
                gemma4_completion_nonstreaming(id, runtime, &req).await
            };
        }
        Ok(None) => {}
        Err(resp) => return resp,
    }
    // Runnable-served architectures (qwen35, gemma3) fail closed on raw
    // completions — chat is their only served surface. This also covers the
    // multi-choice fan-out below, which is reached only through this handler.
    if let Some(resp) = reject_completions_for_runnable_arch(&state, &req.model).await {
        return resp;
    }
    // Multi-choice (n > 1) fans out into independent generations. Validate the
    // count and reject the combinations Camelid does not implement before the
    // request is consumed.
    let n_choices = req.n.unwrap_or(1);
    if n_choices == 0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "n must be at least 1".to_string(),
            Some("n"),
        );
    }
    if n_choices > MAX_N_CHOICES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            format!("n must be between 1 and {MAX_N_CHOICES}"),
            Some("n"),
        );
    }
    if n_choices > 1 && req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "n greater than 1 is not supported with stream:true; request multiple choices without streaming".to_string(),
            Some("n"),
        );
    }
    if n_choices > 1 && req.camelid_receipt.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "camelid_receipt is not supported with n greater than 1; a receipt records one complete generation".to_string(),
            Some("camelid_receipt"),
        );
    }
    // Capture the receipt stamp before the request is consumed. Receipts are
    // strictly opt-in and never silently attached.
    let receipt_stamp = if req.camelid_receipt.unwrap_or(false) {
        if req.stream.unwrap_or(false) {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "camelid_receipt is not supported with stream:true; receipts record one complete non-streaming generation".to_string(),
                Some("camelid_receipt"),
            );
        }
        match receipt_completion_request_stamp(&req) {
            Ok(stamp) => Some(stamp),
            Err(response) => return *response,
        }
    } else {
        None
    };
    // Logprobs are non-streaming, single-choice only (a per-chunk / per-choice
    // follow-up). Reject the unsupported combinations before runtime.
    let wants_logprobs = req.logprobs.is_some();
    if wants_logprobs && req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logprobs are not supported with stream:true; request logprobs without streaming"
                .to_string(),
            Some("logprobs"),
        );
    }
    if wants_logprobs && n_choices > 1 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logprobs are not supported with n greater than 1".to_string(),
            Some("logprobs"),
        );
    }
    let req = GenerationSessionRequest {
        model: req.model,
        prompt: req.prompt,
        messages: None,
        max_tokens: req.max_tokens,
        stream: req.stream,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        logit_bias: req.logit_bias,
        stop: req.stop,
        n: req.n,
        best_of: req.best_of,
        completion_logprobs: req.logprobs,
        chat_logprobs: None,
        top_logprobs: None,
        camelid_logit_token_ids: req.camelid_logit_token_ids,
        camelid_prompt_token_ids: req.camelid_prompt_token_ids,
        camelid_dense_diagnostics: req.camelid_dense_diagnostics,
        camelid_dense_diagnostic_generated_index: req.camelid_dense_diagnostic_generated_index,
        camelid_context_budget_tokens: None,
        camelid_enable_thinking: None,
        tools: None,
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: None,
        constraint: None,
    };
    let stream = req.stream.unwrap_or(false);
    if n_choices > 1 {
        // Non-streaming independent multi-choice generation: all n choices
        // run inside ONE engine job, so no other request interleaves.
        return completions_multi_choice(&state, req, n_choices).await;
    }
    // Prep runs OUTSIDE any serialization (D3); see chat_completions.
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if stream {
        // Text-completion streaming does not implement stream_options yet
        // (scope: chat-completions only), so usage is never emitted here.
        return stream_completion(&state, prepared, false, false);
    }

    // The decode runs on the engine worker; CancelOnDrop stops it within one
    // step if this handler frame is dropped (client disconnect).
    let _cancel_on_drop = CancelOnDrop(prepared.cancel.token.clone());
    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    let effective_max_tokens = prepared.max_tokens;
    match generate_decoded_tokens_blocking(&state, prepared).await {
        Ok(generated) => {
            let camelid_receipt = match receipt_stamp {
                Some(stamp) => {
                    build_server_receipt(&state, &model_id, stamp, effective_max_tokens, &generated)
                        .await
                }
                None => None,
            };
            let GeneratedText {
                text,
                prompt_token_ids,
                generated_token_ids,
                dense_metadata,
                top_logits,
                step_top_logits,
                step_logprobs,
                output_projection,
                dense,
                dense_diagnostic_generated_index,
                completion_tokens,
                finish_reason,
                timings,
                execution_trace: _,
            } = generated;
            let completion_logprobs = if step_logprobs.is_empty() {
                None
            } else {
                Some(build_completion_logprobs(&step_logprobs))
            };
            (
                StatusCode::OK,
                Json(CompletionResponse {
                    id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                    object: "text_completion",
                    created: 0,
                    model: model_id,
                    choices: vec![CompletionChoice {
                        index: 0,
                        text,
                        finish_reason,
                        logprobs: completion_logprobs,
                    }],
                    usage: CompletionUsage {
                        prompt_tokens: prompt_token_count,
                        completion_tokens,
                        total_tokens: prompt_token_count + completion_tokens,
                    },
                    camelid: GenerationDiagnostics {
                        prompt_token_ids,
                        generated_token_ids,
                        dense_metadata,
                        top_logits,
                        step_top_logits,
                        output_projection,
                        dense,
                        dense_diagnostic_generated_index,
                        timings_ms: timings,
                    },
                    camelid_receipt,
                }),
            )
                .into_response()
        }
        Err(response) => *response,
    }
}

/// Non-streaming multi-choice (`n` > 1) chat generation: runs `n_choices`
/// independent generations, each with its own derived seed so the choices are
/// distinct yet reproducible, and assembles them into one OpenAI response. Each
/// choice is a full generation (its own prefill + decode) â€” a capability, not a
/// throughput claim. `camelid` diagnostics mirror the first choice; usage counts
/// the prompt once and sums completion tokens across choices. Streaming and
/// receipts are rejected upstream for this path. All n decodes run inside ONE
/// engine job (one queue slot), so no other request interleaves between choices.
async fn chat_completions_multi_choice(
    state: &AppState,
    req: GenerationSessionRequest,
    n_choices: u32,
) -> Response {
    let base_seed = req.seed.unwrap_or(0);
    let prepared_choices = match prepare_multi_choice(state, req, n_choices, base_seed).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    let _cancel_on_drop = CancelOnDrop(prepared_choices.cancel.token.clone());
    let model_id = prepared_choices.model_id.clone();
    let prompt_token_count = prepared_choices.prompt_token_count;
    let results = match run_multi_choice_on_engine(state, prepared_choices).await {
        Ok(results) => results,
        Err(response) => return *response,
    };
    // Disclose the serve lane once — every choice ran the same model.
    let lane = match state.loaded_models.read().await.get(&model_id) {
        Some(model) if classify_loaded_model(model) == ModelLaneClass::ExperimentalImplemented => {
            Some("experimental")
        }
        _ => None,
    };
    let mut choices = Vec::with_capacity(results.len());
    let mut total_completion_tokens = 0usize;
    let mut first_diagnostics: Option<GenerationDiagnostics> = None;
    for (index, generated) in results.into_iter().enumerate() {
        let content = if lane.is_some() {
            generated.text.trim().to_string()
        } else {
            generated.text.clone()
        };
        let finish_reason = generated.finish_reason;
        total_completion_tokens += generated.completion_tokens;
        if index == 0 {
            first_diagnostics = Some(GenerationDiagnostics {
                prompt_token_ids: generated.prompt_token_ids,
                generated_token_ids: generated.generated_token_ids,
                dense_metadata: generated.dense_metadata,
                top_logits: generated.top_logits,
                step_top_logits: generated.step_top_logits,
                output_projection: generated.output_projection,
                dense: generated.dense,
                dense_diagnostic_generated_index: generated.dense_diagnostic_generated_index,
                timings_ms: generated.timings,
            });
        }
        choices.push(ChatCompletionChoice {
            index: index as u32,
            message: ChatCompletionMessage {
                role: "assistant",
                content,
                // Tool-call parsing is single-choice only (the non-streaming path).
                tool_calls: None,
            },
            finish_reason,
            // Logprobs are rejected upstream for n>1.
            logprobs: None,
        });
    }
    let camelid =
        first_diagnostics.expect("n_choices >= 1 guarantees the first choice produced diagnostics");
    (
        StatusCode::OK,
        Json(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion",
            created: 0,
            model: model_id,
            choices,
            usage: CompletionUsage {
                prompt_tokens: prompt_token_count,
                completion_tokens: total_completion_tokens,
                total_tokens: prompt_token_count + total_completion_tokens,
            },
            camelid,
            camelid_receipt: None,
            lane,
        }),
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: std::result::Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
    // Fail closed on multimodal input before any routing: no Camelid row loads
    // a vision/audio tower, so image/audio/video parts must produce a typed
    // error, never a silent text-only generation.
    if let Some(messages) = req.messages.as_deref() {
        if let Some(response) = reject_unsupported_multimodal_content(messages) {
            return response;
        }
    }
    // response_format -> constrained decoding (json_object / json_schema,
    // non-streaming). Parsed BEFORE lane dispatch -- it is pure request-shape
    // validation with no model access -- so a malformed schema 400s uniformly on
    // every lane, and the env-gated serve lanes below (which do not enforce
    // constraints) can reject a constrained request instead of silently
    // returning unconstrained output with a 200.
    let constraint = match constraint_from_response_format(req.response_format.as_ref()) {
        Ok(constraint) => constraint,
        Err(response) => return *response,
    };
    let constraint_unsupported_on_lane = || {
        api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "response_format constrained decoding is not supported on this model's serve lane yet"
                .to_string(),
            Some("response_format"),
        )
    };
    // Gemma 4 serve path (additive, gated by CAMELID_GEMMA4_SERVE). Short-circuits
    // if this request targets a loaded gemma4 runtime; otherwise falls through to
    // the existing Llama/3B path unchanged.
    match resolve_gemma4_runtime(&state, &req).await {
        Ok(Some((id, runtime))) => {
            if constraint.is_some() {
                return constraint_unsupported_on_lane();
            }
            return if req.stream.unwrap_or(false) {
                gemma4_chat_streaming(id, runtime, &req).await
            } else {
                gemma4_chat_nonstreaming(id, runtime, &req).await
            };
        }
        Ok(None) => {}
        Err(resp) => return resp,
    }
    // Runnable serve path (additive, gated by CAMELID_RUNNABLE_SERVE): short-circuits
    // a qwen35/Ornith model to the runnable lane. Streaming mirrors the OpenAI
    // chunk shape with think tokens as `reasoning_content` deltas.
    match resolve_runnable_runtime(&state, &req.model).await {
        Ok(Some((id, runtime))) => {
            if constraint.is_some() {
                return constraint_unsupported_on_lane();
            }
            if req.stream.unwrap_or(false) {
                return runnable_chat_streaming(id, runtime, &req).await;
            }
            return runnable_chat_nonstreaming(id, runtime, &req).await;
        }
        Ok(None) => {}
        Err(resp) => return resp,
    }
    // DiffusionGemma serve path (additive, gated by CAMELID_DG_SERVE):
    // short-circuits a diffusion-gemma model to the dedicated diffusion lane
    // (block-level SSE; a denoise block is minutes of compute).
    match resolve_dg_runtime(&state, &req.model).await {
        Ok(Some((id, runtime))) => {
            if constraint.is_some() {
                return constraint_unsupported_on_lane();
            }
            if req.stream.unwrap_or(false) {
                return dg_chat_streaming(id, runtime, &req).await;
            }
            return dg_chat_nonstreaming(id, runtime, &req).await;
        }
        Ok(None) => {}
        Err(resp) => return resp,
    }
    // Multi-choice (n > 1) fans out into independent generations. Validate the
    // count and reject the combinations Camelid does not implement before the
    // request is consumed.
    let n_choices = req.n.unwrap_or(1);
    if n_choices == 0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "n must be at least 1".to_string(),
            Some("n"),
        );
    }
    if n_choices > MAX_N_CHOICES {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            format!("n must be between 1 and {MAX_N_CHOICES}"),
            Some("n"),
        );
    }
    if n_choices > 1 && req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "n greater than 1 is not supported with stream:true; request multiple choices without streaming".to_string(),
            Some("n"),
        );
    }
    if n_choices > 1 && req.camelid_receipt.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "camelid_receipt is not supported with n greater than 1; a receipt records one complete generation".to_string(),
            Some("camelid_receipt"),
        );
    }
    // Capture the receipt stamp before the request is consumed. Receipts are
    // strictly opt-in and never silently attached.
    let receipt_stamp = if req.camelid_receipt.unwrap_or(false) {
        if req.stream.unwrap_or(false) {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "camelid_receipt is not supported with stream:true; receipts record one complete non-streaming generation".to_string(),
                Some("camelid_receipt"),
            );
        }
        match receipt_request_stamp(&req) {
            Ok(stamp) => Some(stamp),
            Err(response) => return *response,
        }
    } else {
        None
    };
    // Logprobs are non-streaming, single-choice only (per-chunk / per-choice logprobs
    // are a follow-up). Reject the unsupported combinations before runtime.
    let wants_logprobs = matches!(req.logprobs, Some(true)) || req.top_logprobs.is_some();
    // Tool calls are surfaced on the non-streaming single-choice path when the
    // request supplied tools and tool_choice is not "none".
    let tools_active = req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && tool_choice_allows_calls(req.tool_choice.as_ref());
    // The constraint itself was parsed before lane dispatch (see above); this
    // captures whether one is active before `constraint` moves into the
    // generation request, for the tool-call parse gate below.
    let constraint_active = constraint.is_some();
    if constraint_active && req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "response_format constrained decoding is not supported with stream:true; request it without streaming".to_string(),
            Some("response_format"),
        );
    }
    if wants_logprobs && req.stream.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logprobs are not supported with stream:true; request logprobs without streaming"
                .to_string(),
            Some("logprobs"),
        );
    }
    if wants_logprobs && n_choices > 1 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "logprobs are not supported with n greater than 1".to_string(),
            Some("logprobs"),
        );
    }
    // OpenAI stream_options.include_usage: resolved here (permissive â€” see
    // stream_options_include_usage) before `req` is consumed into the generation
    // request. Threaded into stream_completion; ignored on the non-streaming
    // branch, which already returns `usage`.
    let include_usage = stream_options_include_usage(req.stream_options.as_ref());
    let req = GenerationSessionRequest {
        model: req.model,
        prompt: None,
        messages: req.messages,
        max_tokens: req.max_tokens,
        stream: req.stream,
        temperature: req.temperature,
        top_k: req.top_k,
        top_p: req.top_p,
        seed: req.seed,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        min_p: req.min_p,
        repeat_penalty: req.repeat_penalty,
        logit_bias: req.logit_bias,
        stop: req.stop,
        n: req.n,
        best_of: None,
        completion_logprobs: None,
        chat_logprobs: req.logprobs,
        top_logprobs: req.top_logprobs,
        camelid_logit_token_ids: req.camelid_logit_token_ids,
        camelid_prompt_token_ids: None,
        camelid_dense_diagnostics: req.camelid_dense_diagnostics,
        camelid_dense_diagnostic_generated_index: req.camelid_dense_diagnostic_generated_index,
        camelid_context_budget_tokens: req.camelid_context_budget_tokens,
        // An explicit request value always wins; the server-wide default
        // (`serve --enable-thinking`) only fills in when the request is silent.
        camelid_enable_thinking: req
            .camelid_enable_thinking
            .or(state.default_enable_thinking.then_some(true)),
        tools: req.tools,
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: Some(DEFAULT_PUBLIC_CHAT_MAX_TOKENS),
        constraint,
    };
    let stream = req.stream.unwrap_or(false);
    if n_choices > 1 {
        // Non-streaming independent multi-choice generation: all n choices
        // run inside ONE engine job, so no other request interleaves.
        return chat_completions_multi_choice(&state, req, n_choices).await;
    }
    // Prep runs OUTSIDE any serialization (D3): tokenization and template
    // rendering never stall behind another request's decode. The GPU-runnable
    // parity probe inside prepare_generation serializes via its own engine job.
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if stream {
        // Streaming decodes are engine jobs too; the SSE layer just maps the
        // job's events onto chunks.
        return stream_completion(&state, prepared, true, include_usage);
    }

    // The decode runs on the engine worker; CancelOnDrop stops it within one
    // step if this handler frame is dropped (client disconnect).
    let _cancel_on_drop = CancelOnDrop(prepared.cancel.token.clone());
    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    let effective_max_tokens = prepared.max_tokens;
    match generate_decoded_tokens_blocking(&state, prepared).await {
        Ok(generated) => {
            let camelid_receipt = match receipt_stamp {
                Some(stamp) => {
                    build_server_receipt(&state, &model_id, stamp, effective_max_tokens, &generated)
                        .await
                }
                None => None,
            };
            // Disclose the serve lane: "experimental" only when the active model is
            // an implemented decoder that is NOT a supported exact row. Never set
            // for supported rows; never a parity claim.
            let lane = match state.loaded_models.read().await.get(&model_id) {
                Some(model)
                    if classify_loaded_model(model) == ModelLaneClass::ExperimentalImplemented =>
                {
                    Some("experimental")
                }
                _ => None,
            };
            // Experimental models have no parity contract, so trim the leading/trailing
            // whitespace some of them emit around the answer for a clean chat bubble.
            // Supported rows are left byte-identical (their generated text is contractual).
            let content = if lane.is_some() {
                generated.text.trim().to_string()
            } else {
                generated.text
            };
            // Parse the model's tool-call output into structured tool_calls when the
            // request supplied tools and tool_choice permits it. On a tool call,
            // content is emptied and finish_reason flips to "tool_calls".
            let tool_calls = if should_parse_tool_calls(tools_active, constraint_active) {
                parse_tool_calls(&content)
            } else {
                None
            };
            let (content, finish_reason) = if tool_calls.is_some() {
                (String::new(), "tool_calls")
            } else {
                (content, generated.finish_reason)
            };
            let logprobs = if generated.step_logprobs.is_empty() {
                None
            } else {
                Some(build_chat_logprobs(&generated.step_logprobs))
            };
            (
                StatusCode::OK,
                Json(ChatCompletionResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion",
                    created: 0,
                    model: model_id,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatCompletionMessage {
                            role: "assistant",
                            content,
                            tool_calls,
                        },
                        finish_reason,
                        logprobs,
                    }],
                    usage: CompletionUsage {
                        prompt_tokens: prompt_token_count,
                        completion_tokens: generated.completion_tokens,
                        total_tokens: prompt_token_count + generated.completion_tokens,
                    },
                    camelid: GenerationDiagnostics {
                        prompt_token_ids: generated.prompt_token_ids,
                        generated_token_ids: generated.generated_token_ids,
                        dense_metadata: generated.dense_metadata,
                        top_logits: generated.top_logits,
                        step_top_logits: generated.step_top_logits,
                        output_projection: generated.output_projection,
                        dense: generated.dense,
                        dense_diagnostic_generated_index: generated
                            .dense_diagnostic_generated_index,
                        timings_ms: generated.timings,
                    },
                    camelid_receipt,
                    lane,
                }),
            )
                .into_response()
        }
        Err(response) => *response,
    }
}

/// What a server-emitted receipt records about the incoming request, captured
/// before the request is consumed by the generation path. Effective values
/// are recorded (e.g. the greedy default temperature 0.0 when omitted) so the
/// verifier can replay the exact request.
struct ReceiptRequestStamp {
    endpoint: &'static str,
    messages_or_prompt: serde_json::Value,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<u32>,
    seed: Option<u64>,
    stop: Vec<String>,
    /// The raw `response_format` request value (not the compiled spec): the
    /// receipt records the request as made, and replay re-parses it through
    /// the same constraint compiler serving used.
    response_format: Option<serde_json::Value>,
    reproducible: bool,
}

fn receipt_request_stamp(
    req: &ChatCompletionRequest,
) -> std::result::Result<ReceiptRequestStamp, Box<Response>> {
    let messages_or_prompt = match serde_json::to_value(&req.messages) {
        Ok(value) => value,
        Err(err) => {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("camelid_receipt could not record the request messages: {err}"),
                Some("messages"),
            )))
        }
    };
    Ok(receipt_stamp_from_parts(
        "/v1/chat/completions",
        messages_or_prompt,
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        stop_spec_to_vec(req.stop.as_ref()),
        req.response_format.clone().filter(|v| !v.is_null()),
    ))
}

/// Receipt stamp for the raw `/v1/completions` endpoint. The receipt records the
/// prompt string (not chat messages); the verifier replays the same endpoint and
/// re-runs the reference engine on the receipt's exact prompt token ids.
fn receipt_completion_request_stamp(
    req: &CompletionRequest,
) -> std::result::Result<ReceiptRequestStamp, Box<Response>> {
    let prompt = req.prompt.clone().ok_or_else(|| {
        Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "camelid_receipt requires a prompt; receipts record one complete generation"
                .to_string(),
            Some("prompt"),
        ))
    })?;
    Ok(receipt_stamp_from_parts(
        "/v1/completions",
        serde_json::Value::String(prompt),
        req.temperature,
        req.top_p,
        req.top_k,
        req.seed,
        stop_spec_to_vec(req.stop.as_ref()),
        // The raw completions endpoint has no response_format field.
        None,
    ))
}

#[allow(clippy::too_many_arguments)]
fn receipt_stamp_from_parts(
    endpoint: &'static str,
    messages_or_prompt: serde_json::Value,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    stop: Vec<String>,
    response_format: Option<serde_json::Value>,
) -> ReceiptRequestStamp {
    let temperature = f64::from(temperature.unwrap_or(0.0));
    // Reproducible means byte-for-byte replayable: strict greedy decoding
    // with no top-p/top-k sampling in play. Anything else is stamped
    // `reproducible: false` and is never presented as verifiable.
    let reproducible = temperature == 0.0 && top_p.is_none() && top_k.is_none();
    ReceiptRequestStamp {
        endpoint,
        messages_or_prompt,
        temperature,
        top_p: top_p.map(f64::from),
        top_k,
        seed,
        stop,
        response_format,
        reproducible,
    }
}

fn stop_spec_to_vec(stop: Option<&StopSpec>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(StopSpec::One(value)) => vec![value.clone()],
        Some(StopSpec::Many(values)) => values.clone(),
    }
}

/// Build the opt-in server-side receipt for one completed generation. No
/// reference engine runs here, so the parity block is emitted not-compared:
/// the receipt is a claim of output that `camelid verify-receipt` checks
/// independently. It is not a support promotion for the lane.
async fn build_server_receipt(
    state: &AppState,
    model_id: &str,
    stamp: ReceiptRequestStamp,
    effective_max_tokens: u32,
    generated: &GeneratedText,
) -> Option<ParityReceipt> {
    let lane = {
        let models = state.loaded_models.read().await;
        models.get(model_id)?.lane.clone()
    };
    // Attach the execution-trace rollup only for reproducible (deterministic, greedy) runs that
    // actually captured one. Non-reproducible or non-deterministic runs leave it absent, so the
    // receipt serializes and digests exactly as before.
    let execution_trace = generated
        .execution_trace
        .as_ref()
        .filter(|_| stamp.reproducible)
        .map(|(digest, fold_count)| {
            receipt::ExecutionTraceBlock::rollup_v1(
                digest.clone(),
                *fold_count,
                generated.generated_token_ids.len(),
            )
        });
    let mut receipt = ParityReceipt {
        schema: RECEIPT_SCHEMA_V1.to_string(),
        receipt_id: String::new(),
        created_utc: receipt::rfc3339_utc_now(),
        lane,
        reference: ReferenceIdentity {
            tool: "llama.cpp".to_string(),
            binary: "llama-server".to_string(),
            version: None,
            commit: None,
        },
        request: receipt::ReceiptRequest {
            endpoint: stamp.endpoint.to_string(),
            messages_or_prompt: stamp.messages_or_prompt,
            max_tokens: effective_max_tokens,
            temperature: stamp.temperature,
            top_p: stamp.top_p,
            top_k: stamp.top_k,
            seed: stamp.seed,
            stop: stamp.stop,
            response_format: stamp.response_format,
        },
        reproducible: stamp.reproducible,
        result: ReceiptResult {
            prompt_token_ids: generated.prompt_token_ids.clone(),
            generated_token_ids: generated.generated_token_ids.clone(),
            generated_text: generated.text.clone(),
            completion_tokens: generated.completion_tokens as u32,
            finish_reason: generated.finish_reason.to_string(),
        },
        parity: ParityBlock::not_compared(),
        // This is the supported-lane serving path; leave the lane absent (= supported,
        // the legacy default) so existing receipts keep their exact digests.
        execution_lane: None,
        execution_trace,
        quality_tier: None,
        signature: None,
    };
    if let Err(err) = receipt.seal() {
        tracing::warn!(error = %err, "failed to seal camelid_receipt; omitting receipt");
        return None;
    }
    telemetry::emit(telemetry::Event::ReceiptWritten {
        receipt_id: receipt.receipt_id.clone(),
        reproducible: receipt.reproducible,
        gguf_sha256: Some(receipt.lane.gguf_sha256.clone()),
    });
    Some(receipt)
}

/// Outcome of an in-process deterministic replay for `verify-receipt`.
pub struct ReceiptReplay {
    pub lane: LaneIdentity,
    pub result: ReceiptResult,
    /// Execution-trace rollup digest re-derived by this replay, when the replay ran on the
    /// deterministic lane (else `None`). Verification compares it against the receipt's block.
    pub execution_trace_digest: Option<String>,
}

/// Replay a receipt's request through the exact non-streaming generation path
/// (same handlers `serve` uses) against the given GGUF, returning what this
/// build produced. Used by `camelid verify-receipt` to prove the receipt's
/// recorded output is reproducible by Camelid itself.
pub async fn replay_receipt_request(
    gguf_path: &std::path::Path,
    configured_threads: Option<usize>,
    request: &receipt::ReceiptRequest,
) -> std::result::Result<ReceiptReplay, String> {
    let state = AppState::with_configured_threads(configured_threads);
    let loaded = load_model_from_path(&state, gguf_path.to_path_buf(), None)
        .await
        .map_err(|err| format!("model load failed: {err}"))?;
    replay_loaded_receipt_request(&state, &loaded.id, request).await
}

async fn replay_loaded_receipt_request(
    state: &AppState,
    model_id: &str,
    request: &receipt::ReceiptRequest,
) -> std::result::Result<ReceiptReplay, String> {
    let loaded = state
        .loaded_models
        .read()
        .await
        .get(model_id)
        .cloned()
        .ok_or_else(|| BackendError::ModelNotLoaded.to_string())?;
    let is_chat = request.endpoint.contains("chat");
    let (prompt, messages) = if is_chat {
        let messages: Vec<ChatMessage> = serde_json::from_value(request.messages_or_prompt.clone())
            .map_err(|err| {
                format!("receipt messages_or_prompt does not parse as chat messages: {err}")
            })?;
        (None, Some(messages))
    } else {
        let prompt: String =
            serde_json::from_value(request.messages_or_prompt.clone()).map_err(|err| {
                format!("receipt messages_or_prompt does not parse as a prompt string: {err}")
            })?;
        (Some(prompt), None)
    };
    // Re-compile the receipt's recorded response_format through the same
    // constraint compiler serving used, so a constrained receipt replays
    // constrained (an unconstrained replay of a constrained generation would
    // deterministically fail verify-receipt on an honest receipt).
    let constraint = match constraint_from_response_format(request.response_format.as_ref()) {
        Ok(constraint) => constraint,
        Err(response) => return Err(response_error_text(*response).await),
    };
    let session_request = GenerationSessionRequest {
        model: Some(loaded.id.clone()),
        prompt,
        messages,
        max_tokens: Some(request.max_tokens),
        stream: Some(false),
        temperature: Some(request.temperature as f32),
        top_k: request.top_k,
        top_p: request.top_p.map(|value| value as f32),
        seed: request.seed,
        presence_penalty: None,
        frequency_penalty: None,
        min_p: None,
        repeat_penalty: None,
        logit_bias: None,
        stop: if request.stop.is_empty() {
            None
        } else {
            Some(StopSpec::Many(request.stop.clone()))
        },
        n: None,
        best_of: None,
        completion_logprobs: None,
        chat_logprobs: None,
        top_logprobs: None,
        camelid_logit_token_ids: None,
        camelid_prompt_token_ids: None,
        camelid_dense_diagnostics: None,
        camelid_dense_diagnostic_generated_index: None,
        camelid_context_budget_tokens: None,
        camelid_enable_thinking: None,
        tools: None,
        unsupported_fields: HashMap::new(),
        default_max_tokens_cap: None,
        constraint,
    };
    // Prep needs no lock: the GPU-runnable-tier parity probe inside
    // prepare_generation serializes via its own engine job. The replay decode
    // itself now runs as an engine job too — closing the pre-inversion hole
    // where a receipt replay decoded UNLOCKED next to a live request.
    let prepared = match prepare_generation(state, session_request).await {
        Ok(prepared) => prepared,
        Err(response) => return Err(response_error_text(response).await),
    };
    let generated = match generate_decoded_tokens_blocking(state, prepared).await {
        Ok(generated) => generated,
        Err(response) => return Err(response_error_text(*response).await),
    };
    Ok(ReceiptReplay {
        lane: loaded.lane,
        execution_trace_digest: generated.execution_trace.map(|(digest, _)| digest),
        result: ReceiptResult {
            prompt_token_ids: generated.prompt_token_ids,
            generated_token_ids: generated.generated_token_ids,
            generated_text: generated.text,
            completion_tokens: generated.completion_tokens as u32,
            finish_reason: generated.finish_reason.to_string(),
        },
    })
}

async fn response_error_text(response: Response) -> String {
    let status = response.status();
    match axum::body::to_bytes(response.into_body(), 64 * 1024).await {
        Ok(bytes) => format!("{status}: {}", String::from_utf8_lossy(&bytes)),
        Err(_) => status.to_string(),
    }
}

async fn validate_generation_request(
    state: &AppState,
    req: GenerationSessionRequest,
) -> std::result::Result<GenerationSessionSummary, Response> {
    // Prep needs no lock: the GPU-runnable-tier parity probe inside
    // prepare_generation serializes via its own engine job. This path
    // validates/creates a session and does not itself decode.
    let prepared = prepare_generation(state, req).await?;

    Ok(GenerationSessionSummary {
        id: format!(
            "gen-{}-{}",
            prepared.model_id,
            state.generation_sessions.read().await.len() + 1
        ),
        object: "generation.session",
        model: prepared.model_id,
        prompt_token_count: prepared.token_ids.len(),
        max_tokens: prepared.max_tokens,
        state: "validated",
        dense_session_ready: true,
        next_step: "public completion endpoints can generate tokens until EOS, max_tokens, or context limit and return either non-streaming JSON or OpenAI-compatible SSE chunks",
    })
}

fn cpu_weight_materialization_limit_bytes() -> std::result::Result<u64, BackendError> {
    match env::var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV) {
        Ok(value) if value.trim().is_empty() => Ok(DEFAULT_CPU_WEIGHT_MATERIALIZATION_LIMIT_BYTES),
        Ok(value) => value.trim().parse::<u64>().map_err(|err| {
            BackendError::InvalidModelMetadata(format!(
                "invalid {CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV} {value:?}: {err}"
            ))
        }),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_CPU_WEIGHT_MATERIALIZATION_LIMIT_BYTES),
        Err(err) => Err(BackendError::InvalidModelMetadata(format!(
            "invalid {CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV}: {err}"
        ))),
    }
}

fn cpu_weight_materialization_retains_q8_blocks() -> bool {
    matches!(
        env::var(RETAIN_Q8_BLOCKS_ENV).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn lazy_q8_linear_materialization_enabled() -> bool {
    match env::var(LAZY_Q8_LINEAR_ENV) {
        Ok(value)
            if value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled") =>
        {
            false
        }
        Ok(_) | Err(env::VarError::NotPresent) => true,
        Err(_) => true,
    }
}

fn q8_file_cache_bytes_for_health() -> Option<u64> {
    parse_byte_count_env("CAMELID_Q8_0_FILE_CACHE_BYTES").map(|value| value as u64)
}

fn q8_lazy_env_value_disabled(value: &str) -> bool {
    value.eq_ignore_ascii_case("0")
        || value.eq_ignore_ascii_case("false")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("disabled")
}

fn q8_lazy_env_present_and_enabled() -> bool {
    matches!(env::var(LAZY_Q8_LINEAR_ENV), Ok(value) if !q8_lazy_env_value_disabled(&value))
}

fn q8_runtime_health() -> Q8RuntimeHealth {
    let lazy_q8_linear = lazy_q8_linear_materialization_enabled();
    let retain_q8_blocks = cpu_weight_materialization_retains_q8_blocks();
    let forced_lazy = q8_lazy_env_present_and_enabled();
    let policy = if forced_lazy {
        "forced_lazy_file_backed_q8"
    } else if lazy_q8_linear {
        "lazy_q8_linear_default_or_auto_retain"
    } else if retain_q8_blocks {
        "eager_f32_with_retained_q8_blocks"
    } else {
        "eager_cpu_materialization"
    };
    let note = if forced_lazy {
        "Q8_0 linears are explicitly forced to file-backed lazy reads; retained-block settings do not override that loader path."
    } else if lazy_q8_linear {
        "Q8_0 linears are lazy by policy unless the loader auto-retains a fitting compact Q8 model."
    } else if retain_q8_blocks {
        "Lazy Q8_0 linears are disabled and Q8_0 source blocks are retained alongside eager f32 CPU weights."
    } else {
        "Lazy Q8_0 linears are disabled; CPU weights may be eagerly materialized within the configured budget."
    };

    Q8RuntimeHealth {
        policy,
        lazy_q8_linear,
        retain_q8_blocks,
        file_cache_bytes: q8_file_cache_bytes_for_health(),
        note,
    }
}

fn estimate_cpu_weight_materialization_bytes(binding: &LlamaTensorBinding) -> crate::Result<u64> {
    fn tensor_estimate(
        desc: &GgufTensorDescriptor,
        retain_q8_blocks: bool,
        lazy_q8_linear: bool,
    ) -> crate::Result<u64> {
        let element_count = desc.dimensions.iter().try_fold(1u64, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} element count overflow while estimating CPU materialization",
                    desc.name
                ))
            })
        })?;
        let file_backed_q8_linear = lazy_q8_linear
            && desc.tensor_type == GgufTensorType::Q8_0
            && matches!(desc.dimensions.len(), 2 | 3);
        // K-quant 2-D/3-D linears (Q4_K/Q5_K/Q6_K/Q2_K/Q3_K) load via the wire path
        // (`load_kquant_wire_linear`), retaining only the compact super-block wire
        // bytes â€” they never materialize an f32 copy, so they must not be counted
        // against the f32 budget (otherwise a 4B Q2_K/Q4_K model's ~16 GB f32 estimate
        // wrongly trips the safety limit even though the resident GPU path uses wire).
        let wire_resident_kquant = matches!(
            desc.tensor_type,
            GgufTensorType::Q4K
                | GgufTensorType::Q5K
                | GgufTensorType::Q6K
                | GgufTensorType::Q2K
                | GgufTensorType::Q3K
        ) && matches!(desc.dimensions.len(), 2 | 3);
        let f32_bytes = if file_backed_q8_linear || wire_resident_kquant {
            0
        } else {
            element_count.checked_mul(4).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} f32 materialization byte estimate overflow",
                    desc.name
                ))
            })?
        };
        let retained_source_bytes = if retain_q8_blocks
            && !file_backed_q8_linear
            && desc.tensor_type == GgufTensorType::Q8_0
        {
            let q8_block_count = element_count.checked_add(31).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8 block-count estimate overflow",
                    desc.name
                ))
            })? / 32;
            q8_block_count
                .checked_mul(mem::size_of::<Q8_0Block>() as u64)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "tensor {} q8 block materialization byte estimate overflow",
                        desc.name
                    ))
                })?
        } else {
            0
        };
        f32_bytes.checked_add(retained_source_bytes).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} CPU materialization byte estimate overflow",
                desc.name
            ))
        })
    }

    let retain_q8_blocks = cpu_weight_materialization_retains_q8_blocks();
    let lazy_q8_linear = lazy_q8_linear_materialization_enabled();
    let mut total = tensor_estimate(&binding.token_embedding, retain_q8_blocks, lazy_q8_linear)?
        .checked_add(tensor_estimate(
            &binding.output_norm,
            retain_q8_blocks,
            lazy_q8_linear,
        )?)
        .ok_or_else(|| {
            BackendError::InvalidTensorData(
                "CPU materialization byte estimate overflow".to_string(),
            )
        })?;
    if !binding.output_is_tied_embedding {
        total = total
            .checked_add(tensor_estimate(
                &binding.output,
                retain_q8_blocks,
                lazy_q8_linear,
            )?)
            .ok_or_else(|| {
                BackendError::InvalidTensorData(
                    "CPU materialization byte estimate overflow".to_string(),
                )
            })?;
    }
    if let Some(rope_freqs) = &binding.rope_freqs {
        total = total
            .checked_add(tensor_estimate(
                rope_freqs,
                retain_q8_blocks,
                lazy_q8_linear,
            )?)
            .ok_or_else(|| {
                BackendError::InvalidTensorData(
                    "CPU materialization byte estimate overflow".to_string(),
                )
            })?;
    }
    for layer in &binding.layers {
        let mut attention_tensors = vec![&layer.attention_norm, &layer.attention_output];
        match &layer.attention {
            LlamaAttentionTensors::Standard {
                q, k, v, biases, ..
            } => {
                attention_tensors.extend([q, k, v]);
                if let Some(biases) = biases {
                    attention_tensors.extend([&biases.q, &biases.k, &biases.v]);
                }
            }
            LlamaAttentionTensors::Mla {
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                kv_a_proj_with_mqa,
                kv_a_layernorm,
                kv_b_proj,
            } => {
                attention_tensors.extend([
                    q_a_proj,
                    q_a_layernorm,
                    q_b_proj,
                    kv_a_proj_with_mqa,
                    kv_a_layernorm,
                    kv_b_proj,
                ]);
            }
        }
        for desc in attention_tensors {
            total = total
                .checked_add(tensor_estimate(desc, retain_q8_blocks, lazy_q8_linear)?)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "CPU materialization byte estimate overflow".to_string(),
                    )
                })?;
        }
        total = total
            .checked_add(tensor_estimate(
                &layer.ffn_norm,
                retain_q8_blocks,
                lazy_q8_linear,
            )?)
            .ok_or_else(|| {
                BackendError::InvalidTensorData(
                    "CPU materialization byte estimate overflow".to_string(),
                )
            })?;
        match &layer.ffn {
            LlamaFfnTensors::Dense { gate, up, down } => {
                for desc in [gate, up, down] {
                    total = total
                        .checked_add(tensor_estimate(desc, retain_q8_blocks, lazy_q8_linear)?)
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(
                                "CPU materialization byte estimate overflow".to_string(),
                            )
                        })?;
                }
            }
            LlamaFfnTensors::MoE {
                router,
                gate_experts,
                up_experts,
                down_experts,
            } => {
                for desc in std::iter::once(router)
                    .chain(gate_experts.descriptors())
                    .chain(up_experts.descriptors())
                    .chain(down_experts.descriptors())
                {
                    total = total
                        .checked_add(tensor_estimate(desc, retain_q8_blocks, lazy_q8_linear)?)
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(
                                "CPU materialization byte estimate overflow".to_string(),
                            )
                        })?;
                }
            }
            LlamaFfnTensors::DeepSeekMoE {
                shared_gate,
                shared_up,
                shared_down,
                router,
                gate_experts,
                up_experts,
                down_experts,
            } => {
                for desc in [shared_gate, shared_up, shared_down, router]
                    .into_iter()
                    .chain(gate_experts.descriptors())
                    .chain(up_experts.descriptors())
                    .chain(down_experts.descriptors())
                {
                    total = total
                        .checked_add(tensor_estimate(desc, retain_q8_blocks, lazy_q8_linear)?)
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(
                                "CPU materialization byte estimate overflow".to_string(),
                            )
                        })?;
                }
            }
        }
    }
    Ok(total)
}

/// Whether this model's linears are all GPU-resident-eligible quants (Q8_0 / Q4_K /
/// Q6_K) in a dense (non-MoE) layout. Such a model is loaded WIRE-ONLY: K-quant 2-D
/// linears keep only their packed super-block wire bytes (see `load_kquant_wire_linear`
/// in `LlamaLoadedWeights::load_with_ownership`) and Q8_0 linears keep their 36-byte
/// blocks/pages â€” neither materializes the f32 the CPU-budget guard estimates. The CUDA
/// resident decode engine reads those packed bytes in place (q8_gemv / q4k_gemv /
/// q6k_gemv), so the f32 quantity the guard sizes is never produced for this model.
///
/// This is the binding-level mirror of `LlamaInferenceSession::resident_decode_eligible`
/// (which needs a built session); it intentionally checks only what the guard needs â€”
/// the per-tensor quant types and the dense layout. Anything else (an f16/f32 linear, a
/// MoE router/expert stack) keeps the eager-f32 CPU path and stays under the guard.
fn binding_all_resident_quant_linears(binding: &LlamaTensorBinding) -> bool {
    let is_resident_quant = |desc: &GgufTensorDescriptor| {
        matches!(
            desc.tensor_type,
            GgufTensorType::Q8_0 | GgufTensorType::Q4K | GgufTensorType::Q5K | GgufTensorType::Q6K
        )
    };
    if !is_resident_quant(&binding.token_embedding) {
        return false;
    }
    if !binding.output_is_tied_embedding && !is_resident_quant(&binding.output) {
        return false;
    }
    binding.layers.iter().all(|layer| {
        let dense = match &layer.ffn {
            LlamaFfnTensors::Dense { gate, up, down } => {
                is_resident_quant(gate) && is_resident_quant(up) && is_resident_quant(down)
            }
            // MoE is not resident-eligible; its experts take the eager-f32 CPU path,
            // which the guard must keep protecting.
            LlamaFfnTensors::MoE { .. } | LlamaFfnTensors::DeepSeekMoE { .. } => false,
        };
        let attention_resident = match &layer.attention {
            LlamaAttentionTensors::Standard {
                q, k, v, biases, ..
            } => {
                biases.is_none()
                    && is_resident_quant(q)
                    && is_resident_quant(k)
                    && is_resident_quant(v)
            }
            LlamaAttentionTensors::Mla { .. } => false, // MLA not resident eligible
        };
        dense && attention_resident && is_resident_quant(&layer.attention_output)
    })
}

/// Whether the GPU-resident decode engine will run this model â€” and therefore the
/// CPU f32 weight materialization the budget guard estimates is never produced. True
/// only when CUDA resident decode is active for this process AND every linear is a
/// resident-eligible quant (`binding_all_resident_quant_linears`). On non-CUDA builds
/// `resident_decode_cuda_active()` is `false`, so the guard always applies there.
fn binding_runs_on_resident_gpu(binding: &LlamaTensorBinding) -> bool {
    crate::inference::resident_decode_cuda_active() && binding_all_resident_quant_linears(binding)
}

/// Whether the CPU decode path consumes every linear WIRE-ONLY â€” and so, like the
/// resident-GPU case, never materializes the f32 weights the budget guard sizes. True
/// when the K-quant CPU block-dot is enabled (Q4_K/Q6_K linears stream their wire bytes
/// via `q4_k`/`q6_k_block_dot`, Q8_0 linears stream their packed blocks) AND every linear
/// is a resident-eligible quant. Without this, serve CPU mode FALSE-POSITIVES the guard
/// on K-quant models (estimating the ~16 GB f32 the wire-only path never produces),
/// because `binding_runs_on_resident_gpu` is false on CPU. K-quant super-blocks are
/// 256-wide so the block-dot always engages (the 7130 dispatch requires that), matching
/// the wire-only invariant documented on `binding_all_resident_quant_linears`.
fn binding_runs_on_cpu_wire_only(binding: &LlamaTensorBinding) -> bool {
    crate::inference::q4_k_cpu_block_dot_enabled() && binding_all_resident_quant_linears(binding)
}

fn enforce_context_budget(
    prompt_tokens: usize,
    max_tokens: u32,
    budget_tokens: u32,
) -> Result<(), String> {
    let total = (prompt_tokens as u64).saturating_add(u64::from(max_tokens));
    if total > u64::from(budget_tokens) {
        return Err(format!(
            "exact rendered prompt ({prompt_tokens} tokens) plus generation allowance \
             ({max_tokens} tokens) exceeds the Workspace context budget ({budget_tokens} tokens)"
        ));
    }
    Ok(())
}

pub(super) fn model_resident_cache_key(model_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    model_id.hash(&mut hasher);
    hasher.finish()
}

fn guard_cpu_weight_materialization_budget(binding: &LlamaTensorBinding) -> crate::Result<u64> {
    // Resident-GPU models load wire-only (packed Q8_0/Q4_K/Q6_K bytes the CUDA engine
    // reads in place) and never materialize the f32 weights this guard sizes. Bypass the
    // CPU budget for them â€” but ONLY them; genuinely CPU-bound large models (or any
    // build/host without the resident GPU path) still hit the guard below.
    if binding_runs_on_resident_gpu(binding) {
        return Ok(0);
    }
    // CPU K-quant block-dot decode is also wire-only (no f32 materialization), so the
    // same bypass applies on the CPU lane â€” otherwise a K-quant model that runs fine via
    // the block-dot is wrongly rejected here. (K-quant conductor Phase 2 follow-up.)
    if binding_runs_on_cpu_wire_only(binding) {
        return Ok(0);
    }
    let estimated_bytes = estimate_cpu_weight_materialization_bytes(binding)?;
    let limit_bytes = cpu_weight_materialization_limit_bytes()?;
    if estimated_bytes > limit_bytes {
        return Err(BackendError::UnsupportedTensorType(format!(
            "estimated CPU f32 weight materialization is {estimated_bytes} bytes, above safety limit {limit_bytes} bytes; current CPU path eagerly decodes dense weights and may trigger host memory pressure. Lower model size/quant target, add lazy/mmap weight materialization, or raise {CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV} deliberately for a controlled run"
        )));
    }
    Ok(estimated_bytes)
}

async fn prepare_generation(
    state: &AppState,
    req: GenerationSessionRequest,
) -> std::result::Result<PreparedGeneration, Response> {
    let requested_max_tokens = req.max_tokens;
    let context_budget_tokens = req.camelid_context_budget_tokens;
    let request_tools = req.tools.clone();
    validate_unsupported_generation_fields(&req).map_err(|response| *response)?;
    validate_choice_and_logprob_fields(&req).map_err(|response| *response)?;
    let sampling = sampling_config_from_request(&req).map_err(|response| *response)?;
    let stop_sequences =
        stop_sequences_from_request(req.stop.as_ref()).map_err(|response| *response)?;
    let logit_diagnostic_token_ids =
        diagnostic_logit_token_ids(req.camelid_logit_token_ids.as_deref())
            .map_err(|response| *response)?;
    let collect_dense_diagnostics = req.camelid_dense_diagnostics.unwrap_or(false)
        || req.camelid_dense_diagnostic_generated_index.is_some();
    let speculative_mode = spec_decode_mode_from_env();
    if requested_max_tokens == Some(0) {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_max_tokens",
            "max_tokens must be greater than zero".to_string(),
            Some("max_tokens"),
        ));
    }

    let input = match (req.prompt, req.messages, req.camelid_prompt_token_ids) {
        (Some(prompt), None, None) if !prompt.is_empty() => PromptInput::Text(prompt),
        (None, Some(messages), None) if !messages.is_empty() => {
            validate_chat_messages(&messages).map_err(|response| *response)?;
            PromptInput::Chat(messages)
        }
        (None, None, Some(token_ids)) if !token_ids.is_empty() => PromptInput::TokenIds(token_ids),
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) | (_, Some(_), Some(_)) => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "ambiguous_generation_input",
                "send exactly one of prompt, messages, or camelid_prompt_token_ids".to_string(),
                None,
            ))
        }
        _ => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "missing_generation_input",
                "generation requires a non-empty prompt, messages array, or camelid_prompt_token_ids array".to_string(),
                None,
            ))
        }
    };

    let model = match get_or_load_model(state, req.model.as_deref()).await {
        Ok(m) => m,
        Err(res) => return Err(res),
    };
    let draft_may_be_used = speculative_mode == Some(SpecDecodeMode::DraftModel)
        && sampling == SamplingConfig::default()
        && !collect_dense_diagnostics
        && logit_diagnostic_token_ids.is_empty();
    if draft_may_be_used {
        ensure_spec_draft_model_loaded(state).await?;
    }

    let model_file_lease = state.model_file_lifecycle.clone().read_owned().await;
    let model = {
        let loaded = state.loaded_models.read().await;
        loaded
            .get(&model.id)
            .filter(|current| current.path == model.path)
            .cloned()
    }
    .ok_or_else(|| {
        api_error(
            StatusCode::CONFLICT,
            "model_transitioned",
            "the model was unloaded while generation was preparing; retry the request".to_string(),
            Some("model"),
        )
    })?;

    let mut timings = GenerationTimings::default();
    let tokenization_started = Instant::now();
    let tokenizer = match model.tokenizer_runtime.clone() {
        Some(tokenizer) => tokenizer,
        None => Arc::new(Tokenizer::from_gguf(&model.gguf).map_err(|err| {
            api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                tokenizer_error_code(&err),
                err.to_string(),
                None,
            )
        })?),
    };
    let token_ids = match input {
        PromptInput::Text(prompt) => {
            let rendered_prompt = RenderedPrompt {
                text: prompt,
                add_special: true,
                parse_special: false,
            };
            let mut token_ids = tokenizer
                .encode(
                    &rendered_prompt.text,
                    rendered_prompt.add_special,
                    rendered_prompt.parse_special,
                )
                .map_err(|err| {
                    api_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "tokenization_failed",
                        err.to_string(),
                        Some("prompt"),
                    )
                })?;
            normalize_mistral_instruct_bos_prefix_tokens(
                &mut token_ids,
                &rendered_prompt,
                &tokenizer,
            );
            token_ids
        }
        PromptInput::Chat(messages) => {
            let rendered_prompt = match request_tools.as_deref() {
                Some(tools) if !tools.is_empty() => {
                    render_chat_prompt_for_tokenization_with_tools(&messages, &tokenizer, tools)
                }
                _ => render_chat_prompt_for_tokenization_for_model_result(
                    &messages,
                    &tokenizer,
                    Some(&model.id),
                    req.camelid_enable_thinking.unwrap_or(false),
                ),
            }
            .map_err(|err| {
                api_error(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "unsupported_chat_template",
                    format!(
                        "chat template rendering failed for loaded model {:?}: {err}",
                        model.id
                    ),
                    Some("messages"),
                )
            })?;
            let mut token_ids = tokenizer
                .encode(
                    &rendered_prompt.text,
                    rendered_prompt.add_special,
                    rendered_prompt.parse_special,
                )
                .map_err(|err| {
                    api_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "tokenization_failed",
                        err.to_string(),
                        Some("messages"),
                    )
                })?;
            normalize_mistral_instruct_bos_prefix_tokens(
                &mut token_ids,
                &rendered_prompt,
                &tokenizer,
            );
            token_ids
        }
        PromptInput::TokenIds(token_ids) => token_ids,
    };
    timings.tokenize = tokenization_started.elapsed().as_millis();
    if token_ids.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "empty_prompt_tokens",
            "prompt encoded to zero tokens".to_string(),
            Some("prompt"),
        ));
    }

    let config = model.llama_config.as_ref().ok_or_else(|| {
        let message = model
            .unsupported_runtime
            .as_ref()
            .map(|unsupported| unsupported.message.clone())
            .unwrap_or_else(|| {
                "loaded model does not expose a Camelid-supported dense GGUF config".to_string()
            });
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "unsupported_model_architecture",
            message,
            Some("model"),
        )
    })?;
    let binding = model.llama_tensors.as_ref().ok_or_else(|| {
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing_dense_tensors",
            "loaded model metadata does not contain the Camelid dense tensors required for generation"
                .to_string(),
            Some("model"),
        )
    })?;

    let context_length = config.context_length as usize;
    // A response limit (max_tokens) is an UPPER BOUND, not a demand: clamp it to
    // the room left in the context window instead of rejecting. This makes the
    // common "response limit == full context" case (e.g. an 8192 limit on an
    // 8192-context model) generate up to the remaining room automatically rather
    // than erroring. The only genuine failure is a prompt that already fills the
    // whole context, leaving no room to generate even a single token.
    if token_ids.len() >= context_length {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!(
                "prompt token count {} leaves no room for generation in context length {}",
                token_ids.len(),
                config.context_length
            ),
            Some("prompt"),
        ));
    }

    let available_max_tokens = (context_length - token_ids.len()) as u32;
    let max_tokens = match requested_max_tokens {
        // Clamp an explicit request down to whatever actually fits.
        Some(requested) => requested.min(available_max_tokens),
        // No explicit limit: use the caller's default cap (itself bounded by the
        // available room), or fill the remaining context.
        None => req
            .default_max_tokens_cap
            .map(|cap| cap.min(available_max_tokens))
            .unwrap_or(available_max_tokens),
    };
    if let Some(index) = req.camelid_dense_diagnostic_generated_index {
        if index >= max_tokens {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_dense_diagnostic_generated_index",
                format!(
                    "camelid_dense_diagnostic_generated_index {} is outside max_tokens {}",
                    index, max_tokens
                ),
                Some("camelid_dense_diagnostic_generated_index"),
            ));
        }
    }
    if let Some(budget_tokens) = context_budget_tokens {
        if budget_tokens == 0 {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_context_budget",
                "camelid_context_budget_tokens must be greater than zero".to_string(),
                Some("camelid_context_budget_tokens"),
            ));
        }
        if let Err(message) = enforce_context_budget(token_ids.len(), max_tokens, budget_tokens) {
            return Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "context_budget_exceeded",
                message,
                Some("camelid_context_budget_tokens"),
            ));
        }
    }

    let weight_load_started = Instant::now();
    let cache_hit = state.cached_weights.read().await.contains_key(&model.id);
    let weights = match load_weights_lru(state, &model, binding).await {
        Ok(w) => w,
        Err(res) => return Err(res),
    };
    timings.weight_cache_hit = cache_hit;
    timings.weight_load = weight_load_started.elapsed().as_millis();
    let session_create_started = Instant::now();
    let dense_metadata = dense_diagnostic_metadata(config, binding, &weights);
    let mut session = LlamaInferenceSession::new(config.clone(), weights).map_err(|err| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "dense_session_unavailable",
            err.to_string(),
            Some("model"),
        )
    })?;
    // Pin the GPU resident-decode engine cache to the model identity (not the
    // per-load weights Arc pointer), so every request for this model reuses the
    // uploaded weights instead of rebuilding the engine each time.
    let resident_cache_key = model_resident_cache_key(&model.id);
    session.set_resident_cache_key(resident_cache_key);
    timings.session_create = session_create_started.elapsed().as_millis();

    // GPU-runnable tier: an uncurated model whose plan routed it onto the resident path
    // (its `selected_backend` carries the `_runnable_unvalidated` label) is admitted to the
    // GPU only after a one-time parity self-check — greedy decode on the resident GPU path
    // must be token-identical to the CPU reference. The verdict is cached per model and the
    // resident-eligibility choke point consults it, so a FAIL runs the model on CPU. Runs on
    // a blocking thread (it builds engines + decodes) and is awaited so the verdict is set
    // BEFORE this request generates. Curated rows never take this branch.
    if crate::inference::resident_decode_cuda_active() {
        let is_runnable_tier = {
            let plans = state.execution_plans.read().await;
            plans
                .get(&model.id)
                .map(|plan| plan.selected_backend.contains("_runnable_unvalidated"))
                .unwrap_or(false)
        };
        if is_runnable_tier {
            let probe_weights = std::sync::Arc::clone(&session.weights);
            let probe_config = config.clone();
            let probe_label = model.id.clone();
            // The probe drives the process-global single-slot resident CUDA
            // engine, so it runs as an ENGINE JOB — serialized against every
            // decode by construction (prep itself holds no lock).
            //
            // Fail CLOSED on panic: an uncurated, arbitrary model can hit an
            // unwrap/index deep in the resident engine build or a CUDA kernel.
            // The catch_unwind is INSIDE the job so a panic becomes a FAILED
            // verdict; an engine post/await failure (queue full, shutdown) is
            // a retryable 503 instead — it must NOT record a durable FAIL for
            // a probe that never ran.
            let probe_passed = state
                .engine
                .run_exclusive(move || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        crate::inference::ensure_resident_parity_verdict(
                            &probe_config,
                            &probe_weights,
                            resident_cache_key,
                            &probe_label,
                        )
                    }))
                    .unwrap_or(false)
                })
                .await
                .map_err(|err| *engine_post_error_response(err))?;
            if !probe_passed {
                crate::inference::record_resident_parity_fail(resident_cache_key);
            }
        }
    }

    // Lossless greedy speculation is a server-level opt-in and only engages
    // for plain greedy requests with no per-step logit consumers; anything
    // else keeps the unchanged vanilla decode loop.
    let speculative = match speculative_mode {
        None => None,
        Some(_)
            if sampling != SamplingConfig::default()
                || collect_dense_diagnostics
                || !logit_diagnostic_token_ids.is_empty()
                || session.weights.layer_range.is_some() =>
        {
            None
        }
        Some(SpecDecodeMode::NGram) => Some(PreparedSpeculative {
            drafter: SpeculativeDrafter::NGram(NGramDrafter::new(
                spec_ngram_min_from_env(),
                spec_ngram_max_from_env(),
            )),
            draft_tokens: spec_draft_tokens_from_env(DEFAULT_NGRAM_DRAFT_TOKENS),
            latch: SpecLatch::default(),
            rounds: 0,
            drafted: 0,
            accepted_drafts: 0,
        }),
        Some(SpecDecodeMode::DraftModel) => Some(PreparedSpeculative {
            drafter: build_model_drafter(state, &model, &tokenizer).await?,
            draft_tokens: spec_draft_tokens_from_env(DEFAULT_MODEL_DRAFT_TOKENS),
            latch: SpecLatch::default(),
            rounds: 0,
            drafted: 0,
            accepted_drafts: 0,
        }),
    };
    // CPU speculation needs CPU-authoritative KV for the chunk-verify rollback, so
    // the target stays off the resident paths. GPU speculation (CAMELID_SPEC_GPU)
    // keeps the target resident and verifies drafts via the batched GPU verify.
    session.set_resident_paths_disabled(speculative.is_some() && !spec_gpu_enabled());

    let telemetry_backend = {
        let plans = state.execution_plans.read().await;
        plans
            .get(&model.id)
            .map(|plan| plan.selected_backend.clone())
            .unwrap_or_else(|| "llama".to_string())
    };
    let telemetry_start = telemetry::RequestStart {
        request_id: uuid::Uuid::new_v4().to_string(),
        model_id: model.id.clone(),
        backend: telemetry_backend,
        quantization: model.lane.quantization.clone(),
        architecture: model.lane.architecture.clone(),
        prompt_tokens: token_ids.len(),
        max_tokens,
        context_length,
        temperature: f64::from(req.temperature.unwrap_or(0.0)),
        stream: req.stream.unwrap_or(false),
    };

    // Logprobs request â†’ collect chosen + top-N each step. Chat uses logprobs:true
    // plus top_logprobs:N; legacy completions uses logprobs:N directly.
    let logprobs_top_n = if matches!(req.chat_logprobs, Some(true)) {
        Some(req.top_logprobs.unwrap_or(0) as usize)
    } else {
        req.completion_logprobs.map(|n| n as usize)
    };
    Ok(PreparedGeneration {
        _model_file_lease: Some(model_file_lease),
        model_id: model.id,
        model_path: model.path,
        token_ids,
        max_tokens,
        tokenizer,
        session,
        sampling,
        logprobs_top_n,
        constraint: req.constraint.clone(),
        stop_sequences,
        logit_diagnostic_token_ids,
        collect_dense_diagnostics,
        dense_diagnostic_generated_index: req
            .camelid_dense_diagnostic_generated_index
            .map(|index| index as usize),
        dense_metadata,
        timings,
        cached_prompt_prefix: state.cached_prompt_prefix.clone(),
        speculative,
        telemetry: Some(telemetry_start),
        cancel: GenerationCancel::unbounded(),
    })
}

/// Build the draft-model drafter for `CAMELID_SPEC_DECODE=draft`: load the
/// configured draft GGUF under the reserved `spec-draft` id (without making
/// it the active model) and fail closed unless its token mapping is
/// identical to the target's â€” drafted token ids must mean the same text in
/// the target vocabulary.
async fn ensure_spec_draft_model_loaded(state: &AppState) -> std::result::Result<(), Response> {
    let Some(path) = env::var_os(SPEC_DRAFT_MODEL_ENV).map(PathBuf::from) else {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_model_missing",
            format!(
                "{SPEC_DECODE_ENV}=draft requires {SPEC_DRAFT_MODEL_ENV} to point at a draft model GGUF"
            ),
            None,
        ));
    };
    let already_loaded = state
        .loaded_models
        .read()
        .await
        .get(SPEC_DRAFT_MODEL_ID)
        .is_some_and(|model| model.path == path);
    if already_loaded {
        return Ok(());
    }
    load_model_from_path_with_activation(state, path, Some(SPEC_DRAFT_MODEL_ID.to_string()), false)
        .await
        .map(|_| ())
        .map_err(|err| {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "spec_draft_model_load_failed",
                err.to_string(),
                None,
            )
        })
}

async fn build_model_drafter(
    state: &AppState,
    target: &LoadedModel,
    target_tokenizer: &Tokenizer,
) -> std::result::Result<SpeculativeDrafter, Response> {
    let Some(path) = env::var_os(SPEC_DRAFT_MODEL_ENV).map(PathBuf::from) else {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_model_missing",
            format!(
                "{SPEC_DECODE_ENV}=draft requires {SPEC_DRAFT_MODEL_ENV} to point at a draft \
                 model GGUF"
            ),
            None,
        ));
    };
    let existing = {
        let loaded = state.loaded_models.read().await;
        loaded
            .get(SPEC_DRAFT_MODEL_ID)
            .filter(|model| model.path == path)
            .cloned()
    };
    let draft_model = existing.ok_or_else(|| {
        api_error(
            StatusCode::CONFLICT,
            "spec_draft_model_transitioned",
            "the speculative draft model was unloaded while generation was preparing; retry the request"
                .to_string(),
            None,
        )
    })?;
    let Some(draft_tokenizer) = draft_model.tokenizer_runtime.as_deref() else {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_tokenizer_unavailable",
            format!("draft model {SPEC_DRAFT_MODEL_ID} has no usable tokenizer"),
            None,
        ));
    };
    if !tokenizers_share_token_mapping(target_tokenizer, draft_tokenizer) {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_tokenizer_mismatch",
            format!(
                "draft model token mapping differs from target {:?}; speculative drafting \
                 requires an identical vocabulary",
                target.id
            ),
            None,
        ));
    }
    let Some(draft_config) = draft_model.llama_config.clone() else {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_model_unsupported",
            format!("draft model {SPEC_DRAFT_MODEL_ID} has no supported runtime config"),
            None,
        ));
    };
    let Some(draft_binding) = draft_model.llama_tensors.as_ref() else {
        return Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_model_unsupported",
            format!("draft model {SPEC_DRAFT_MODEL_ID} has no bound tensors"),
            None,
        ));
    };
    let draft_weights = load_weights_lru(state, &draft_model, draft_binding).await?;
    let draft_session = LlamaInferenceSession::new(draft_config, draft_weights).map_err(|err| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "spec_draft_session_unavailable",
            err.to_string(),
            None,
        )
    })?;
    Ok(SpeculativeDrafter::Model(Box::new(ModelDrafter::new(
        draft_session,
    ))))
}

/// Identical token mapping: same tokenizer model and the same token text at
/// every id. This is the correctness requirement for cross-model drafting.
fn tokenizers_share_token_mapping(a: &Tokenizer, b: &Tokenizer) -> bool {
    a.model == b.model
        && a.tokens.len() == b.tokens.len()
        && a.tokens
            .iter()
            .zip(b.tokens.iter())
            .all(|(left, right)| left.text == right.text)
}

fn dense_diagnostic_metadata(
    config: &LlamaModelConfig,
    binding: &LlamaTensorBinding,
    weights: &LlamaLoadedWeights,
) -> DenseDiagnosticMetadata {
    let dims = DenseLlamaDims::from_config(config).expect("validated Camelid dense dimensions");
    let head_dim = dims.head_dim;
    let output_shape = weights.output_projection().shape.dims.clone();
    let output_projection_layout =
        if output_shape.first() == Some(&(config.embedding_length as usize)) {
            "input_output"
        } else {
            "output_input"
        };
    let first_layer = weights
        .layers
        .first()
        .expect("validated Camelid dense binding has at least one layer");

    DenseDiagnosticMetadata {
        embedding_length: config.embedding_length,
        attention_head_count: config.attention_head_count,
        attention_head_count_kv: config.attention_head_count_kv,
        head_dim,
        rope_dimension_count: config.rope_dimension_count.unwrap_or(head_dim as u32) as usize,
        rope_freq_base: config.rope_freq_base.unwrap_or(10_000.0),
        rope_scaling_type: config
            .rope_scaling_type
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("none")
            .to_string(),
        rope_scaling_factor: config.rope_scaling_factor,
        rope_scaling_original_context_length: config.rope_scaling_original_context_length,
        rope_scaling_low_freq_factor: config.rope_scaling_low_freq_factor,
        rope_scaling_high_freq_factor: config.rope_scaling_high_freq_factor,
        rope_pairing: rope_pairing_for_config(config)
            .map(|pairing| pairing.label())
            .unwrap_or("invalid_env"),
        rope_direction: diagnostic_rope_direction()
            .map(|direction| direction.label())
            .unwrap_or("invalid_env"),
        rope_position_mode: diagnostic_rope_position_mode()
            .map(|position_mode| position_mode.label())
            .unwrap_or("invalid_env"),
        gqa_head_mapping: diagnostic_gqa_head_mapping()
            .map(|mapping| mapping.label())
            .unwrap_or("invalid_env"),
        attention_score_scale: diagnostic_attention_score_scale()
            .map(|scale| scale.label())
            .unwrap_or("invalid_env"),
        linear_accumulation: diagnostic_linear_accumulation_precision()
            .map(|precision| precision.label())
            .unwrap_or("invalid_env"),
        ffn_gate_up_order: diagnostic_ffn_gate_up_order()
            .map(|order| order.label())
            .unwrap_or("invalid_env"),
        rms_norm_epsilon: config.rms_norm_epsilon,
        rms_norm_effective_epsilon: diagnostic_rms_norm_epsilon(config.rms_norm_epsilon)
            .unwrap_or(config.rms_norm_epsilon),
        square_linear_diagnostic_layout: diagnostic_square_linear_layout()
            .map(|layout| layout.label())
            .unwrap_or("invalid_env"),
        rectangular_linear_diagnostic_layout: diagnostic_rectangular_linear_layout()
            .map(|layout| layout.label())
            .unwrap_or("invalid_env"),
        token_embedding_shape: weights.token_embedding.shape.dims.clone(),
        output_shape,
        output_is_tied_embedding: binding.output_is_tied_embedding,
        output_projection_layout,
        output_projection_diagnostic_layout: diagnostic_output_projection_layout()
            .map(|layout| layout.label())
            .unwrap_or("invalid_env"),
        zero_attention_delta: diagnostic_zero_delta_selector(DeltaZeroTarget::Attention)
            .unwrap_or_else(|_| "invalid_env".to_string()),
        zero_ffn_delta: diagnostic_zero_delta_selector(DeltaZeroTarget::Ffn)
            .unwrap_or_else(|_| "invalid_env".to_string()),
        projection_orientations: DenseProjectionOrientations {
            attention_q: linear_projection_orientation(
                &first_layer.attention_q,
                dims.embedding_length,
                dims.embedding_length,
                "attention_q",
            ),
            attention_k: linear_projection_orientation(
                &first_layer.attention_k,
                dims.embedding_length,
                dims.kv_width,
                "attention_k",
            ),
            attention_v: linear_projection_orientation(
                &first_layer.attention_v,
                dims.embedding_length,
                dims.kv_width,
                "attention_v",
            ),
            attention_output: linear_projection_orientation(
                &first_layer.attention_output,
                dims.embedding_length,
                dims.embedding_length,
                "attention_output",
            ),
            ffn_gate: linear_projection_orientation(
                &first_layer.ffn_gate,
                dims.embedding_length,
                dims.feed_forward_length,
                "ffn_gate",
            ),
            ffn_up: linear_projection_orientation(
                &first_layer.ffn_up,
                dims.embedding_length,
                dims.feed_forward_length,
                "ffn_up",
            ),
            ffn_down: linear_projection_orientation(
                &first_layer.ffn_down,
                dims.feed_forward_length,
                dims.embedding_length,
                "ffn_down",
            ),
        },
    }
}

fn linear_projection_orientation(
    weight: &CpuTensor,
    input_width: usize,
    output_width: usize,
    rectangular_role: &str,
) -> LinearProjectionOrientation {
    let shape = weight.shape.dims.clone();
    let square_diagnostic_applies = shape.as_slice() == [input_width, input_width];
    let descriptor_layout = if shape.as_slice() == [input_width, output_width] {
        "input_output"
    } else if shape.as_slice() == [output_width, input_width] {
        "output_input"
    } else {
        "incompatible"
    };
    let runtime_interpretation = if square_diagnostic_applies {
        diagnostic_square_linear_layout()
            .map(|layout| match layout {
                crate::inference::SquareLinearLayout::Descriptor => "descriptor",
                crate::inference::SquareLinearLayout::Transposed => "rhs_transposed",
            })
            .unwrap_or("invalid_env")
    } else {
        crate::inference::diagnostic_rectangular_linear_layout_for_role(rectangular_role)
            .map(|layout| match layout {
                crate::inference::RectangularLinearLayout::Descriptor => "descriptor_forced",
                crate::inference::RectangularLinearLayout::Transposed => "rhs_transposed_forced",
                crate::inference::RectangularLinearLayout::Auto
                    if shape.first() == Some(&input_width) =>
                {
                    "descriptor"
                }
                crate::inference::RectangularLinearLayout::Auto
                    if shape.get(1) == Some(&input_width) =>
                {
                    "rhs_transposed"
                }
                crate::inference::RectangularLinearLayout::Auto => "incompatible",
            })
            .unwrap_or("invalid_env")
    };

    LinearProjectionOrientation {
        shape,
        input_width,
        output_width,
        descriptor_layout,
        runtime_interpretation,
        square_diagnostic_applies,
    }
}

/// Upper bound on `n` (independent choices). Each choice is a full, independent
/// generation (its own prompt prefill + decode), so the cap keeps a single
/// request from fanning out into unbounded server work.
const MAX_N_CHOICES: u32 = 8;

/// Upper bound on requested logprobs (chosen token + this many top alternatives),
/// for both chat `top_logprobs` and legacy completions `logprobs`.
const MAX_LOGPROBS: u32 = 20;

fn validate_choice_and_logprob_fields(
    req: &GenerationSessionRequest,
) -> std::result::Result<(), Box<Response>> {
    if matches!(req.n, Some(0)) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "n must be at least 1".to_string(),
            Some("n"),
        )));
    }
    if matches!(req.n, Some(value) if value > MAX_N_CHOICES) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            format!("n must be between 1 and {MAX_N_CHOICES}"),
            Some("n"),
        )));
    }
    if matches!(req.best_of, Some(0)) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            "best_of must be at least 1".to_string(),
            Some("best_of"),
        )));
    }
    if matches!(req.best_of, Some(value) if value > 1) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "best_of values greater than 1 are not supported yet; this backend does not sample multiple candidates server-side".to_string(),
            Some("best_of"),
        )));
    }
    if matches!(req.completion_logprobs, Some(value) if value > MAX_LOGPROBS) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_parameter",
            format!("logprobs must be between 0 and {MAX_LOGPROBS}"),
            Some("logprobs"),
        )));
    }
    // Chat `top_logprobs` requires `logprobs:true` and is capped, matching OpenAI.
    if let Some(top_logprobs) = req.top_logprobs {
        if !matches!(req.chat_logprobs, Some(true)) {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_parameter",
                "top_logprobs requires logprobs to be set to true".to_string(),
                Some("top_logprobs"),
            )));
        }
        if top_logprobs > MAX_LOGPROBS {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_parameter",
                format!("top_logprobs must be between 0 and {MAX_LOGPROBS}"),
                Some("top_logprobs"),
            )));
        }
    }
    Ok(())
}

fn validate_unsupported_generation_fields(
    req: &GenerationSessionRequest,
) -> std::result::Result<(), Box<Response>> {
    let Some(param) = req.unsupported_fields.keys().min().map(String::as_str) else {
        return Ok(());
    };
    let message = match param {
        "parse_tool_calls" => {
            "the camelid parse_tool_calls control is not supported on this route yet"
        }
        "response_format" | "json_schema" | "schema" | "grammar" => {
            // Only the raw-completions / llama-server compatibility routes flatten
            // these keys into unsupported_fields; the chat route declares
            // response_format and enforces it.
            "constrained generation is supported on /v1/chat/completions via response_format; this route does not support json_schema/grammar fields"
        }
        "stream_options" => {
            "OpenAI stream_options are not supported yet; Camelid streams plain SSE chunks"
        }
        "echo" | "suffix" => "completion echo/suffix compatibility is not supported yet",
        "mirostat" | "mirostat_tau" | "mirostat_eta" | "typical_p" | "tfs_z" | "ignore_eos"
        | "n_keep" => "this llama-server sampler/control field is not supported yet",
        "cache_prompt" | "id_slot" | "id_task" | "slot_id" => {
            "llama-server slot/cache controls are not supported by this compatibility route yet"
        }
        "input" => {
            "embeddings/Responses-style input payloads are not supported on generation routes"
        }
        _ => "unsupported generation request field",
    };
    Err(Box::new(api_error(
        StatusCode::BAD_REQUEST,
        "unsupported_parameter",
        message.to_string(),
        Some(static_param_name(param)),
    )))
}

fn static_param_name(param: &str) -> &'static str {
    match param {
        "tools" => "tools",
        "tool_choice" => "tool_choice",
        "parallel_tool_calls" => "parallel_tool_calls",
        "parse_tool_calls" => "parse_tool_calls",
        "response_format" => "response_format",
        "json_schema" => "json_schema",
        "schema" => "schema",
        "grammar" => "grammar",
        "stream_options" => "stream_options",
        "echo" => "echo",
        "suffix" => "suffix",
        "mirostat" => "mirostat",
        "mirostat_tau" => "mirostat_tau",
        "mirostat_eta" => "mirostat_eta",
        "typical_p" => "typical_p",
        "tfs_z" => "tfs_z",
        "ignore_eos" => "ignore_eos",
        "n_keep" => "n_keep",
        "cache_prompt" => "cache_prompt",
        "id_slot" => "id_slot",
        "id_task" => "id_task",
        "slot_id" => "slot_id",
        "input" => "input",
        _ => "request",
    }
}

/// Resolve OpenAI `stream_options.include_usage` permissively, matching the
/// pinned llama-server oracle (commit acd79d6), which returns HTTP 200 for every
/// malformed shape rather than a structured error. Returns `false` (usage chunk
/// off) when `stream_options` is absent, `null`, or not an object; when
/// `include_usage` is absent or not a boolean; and ignores any unknown subfields.
/// Only an explicit `include_usage: true` turns the terminal usage chunk on, so
/// it is the sole honored subfield (exact-row support); nothing else is claimed.
/// `stream: false` needs no handling here â€” the non-streaming response already
/// carries `usage`, so a non-streaming request with `stream_options` "just works"
/// untouched.
fn stream_options_include_usage(stream_options: Option<&serde_json::Value>) -> bool {
    stream_options
        .and_then(serde_json::Value::as_object)
        .and_then(|map| map.get("include_usage"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn stop_sequences_from_request(
    stop: Option<&StopSpec>,
) -> std::result::Result<Vec<String>, Box<Response>> {
    let Some(stop) = stop else {
        return Ok(Vec::new());
    };
    let sequences = match stop {
        StopSpec::One(value) => vec![value.clone()],
        StopSpec::Many(values) => values.clone(),
    };
    if sequences.is_empty() {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_stop",
            "stop must be a non-empty string or a non-empty array of strings".to_string(),
            Some("stop"),
        )));
    }
    if sequences.len() > 4 {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_stop",
            "stop may contain at most 4 sequences".to_string(),
            Some("stop"),
        )));
    }
    if sequences.iter().any(|value| value.is_empty()) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_stop",
            "stop sequences must not be empty".to_string(),
            Some("stop"),
        )));
    }
    Ok(sequences)
}

fn sampling_config_from_request(
    req: &GenerationSessionRequest,
) -> std::result::Result<SamplingConfig, Box<Response>> {
    let config = SamplingConfig {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: req.top_k.map(|value| value as usize),
        top_p: req.top_p,
        min_p: req.min_p,
        seed: req.seed,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repeat_penalty: req.repeat_penalty.unwrap_or(1.0),
        logit_bias: parse_logit_bias(req.logit_bias.as_ref()).map_err(|message| {
            Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_sampling_parameter",
                message,
                Some("logit_bias"),
            ))
        })?,
    };
    config.validate().map_err(|err| {
        Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_sampling_parameter",
            err.to_string(),
            None,
        ))
    })?;
    Ok(config)
}

fn diagnostic_logit_token_ids(
    token_ids: Option<&[u32]>,
) -> std::result::Result<Vec<u32>, Box<Response>> {
    let Some(token_ids) = token_ids else {
        return Ok(Vec::new());
    };
    if token_ids.len() > 16 {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "invalid_logit_diagnostic_token_ids",
            "camelid_logit_token_ids accepts at most 16 token ids".to_string(),
            Some("camelid_logit_token_ids"),
        )));
    }
    let mut deduped = Vec::with_capacity(token_ids.len());
    for token_id in token_ids {
        if !deduped.contains(token_id) {
            deduped.push(*token_id);
        }
    }
    Ok(deduped)
}

fn parse_logit_bias(
    logit_bias: Option<&HashMap<String, f32>>,
) -> std::result::Result<Vec<(usize, f32)>, String> {
    let Some(logit_bias) = logit_bias else {
        return Ok(Vec::new());
    };
    let mut parsed = Vec::with_capacity(logit_bias.len());
    for (token_id, bias) in logit_bias {
        let token_id = token_id.parse::<usize>().map_err(|_| {
            format!("logit_bias token id {token_id:?} must be a non-negative integer string")
        })?;
        if !bias.is_finite() || !(-100.0..=100.0).contains(bias) {
            return Err(format!(
                "logit_bias for token {token_id} must be finite and in [-100, 100], got {bias}"
            ));
        }
        parsed.push((token_id, *bias));
    }
    parsed.sort_by_key(|(token_id, _)| *token_id);
    Ok(parsed)
}

/// Map an engine post/await failure onto the typed error envelope. QueueFull
/// is the explicit backpressure signal (D2): the bounded queue rejected the
/// job, the client should retry shortly. Unavailable covers engine shutdown
/// and a job whose result channel was lost (e.g. the job panicked and the
/// engine contained it).
fn engine_post_error_response(err: engine::EnginePostError) -> Box<Response> {
    match err {
        engine::EnginePostError::QueueFull => {
            let mut response = api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine_queue_full",
                format!(
                    "the generation queue is full; retry shortly (depth is bounded by {})",
                    engine::QUEUE_DEPTH_ENV
                ),
                None,
            );
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                "1".parse().expect("static header"),
            );
            Box::new(response)
        }
        engine::EnginePostError::Unavailable => Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_worker_failed",
            "generation worker failed before completing the request".to_string(),
            None,
        )),
    }
}

/// The decode body — caller must be the exclusive compute owner (the engine
/// worker thread). The wall-clock timeout is enforced inside the decode loop
/// (see `GenerationCancel`), so the job always runs to a clean stop — it is
/// never detached.
fn run_decode_job_serialized(
    mut prepared: PreparedGeneration,
) -> std::result::Result<GeneratedText, Box<Response>> {
    #[cfg(test)]
    let _decode_probe = decode_probe::enter();
    // Arm the deadline before the test-sleep hook: the sleep stands in for a
    // slow decode step and must count against the wall-clock budget.
    let timeout = generation_timeout_duration()?;
    prepared.cancel.arm(timeout);
    if let Some(duration) = generation_step_test_sleep_duration() {
        std::thread::sleep(duration);
    }
    let cancel = prepared.cancel.clone();
    let result = generate_decoded_tokens(prepared);
    // Â§4 safe-boot: a decode ran to completion under the applied gait without
    // wedging the host, so clear the in-progress marker. Cheap and idempotent.
    // Not marked when the decode was cancelled or timed out — those did not
    // prove a healthy full decode.
    if cancel.tripped().is_none() {
        crate::gait::sentinel::mark_healthy();
    }
    result
}

/// Post the blocking decode to the engine worker and await its result. If the
/// calling handler frame is dropped (client disconnect), `CancelOnDrop` stops
/// the running decode within one step; the job itself is never detached from
/// the engine's serialization.
async fn generate_decoded_tokens_blocking(
    state: &AppState,
    prepared: PreparedGeneration,
) -> std::result::Result<GeneratedText, Box<Response>> {
    match state
        .engine
        .run_exclusive(move || run_decode_job_serialized(prepared))
        .await
    {
        Ok(result) => result,
        Err(err) => Err(engine_post_error_response(err)),
    }
}

struct StreamStepRequest {
    /// Try the resident GPU-sampling greedy fast lane before the general step.
    greedy_fast: bool,
    input: Vec<u32>,
    sampler: LlamaSampler,
    history: Vec<u32>,
    collect_dense_diagnostics: bool,
}

/// A generation step whose token was sampled ON the GPU (resident greedy fast lane): no
/// logits crossed back to the CPU, so the logits/hidden fields are empty placeholders.
/// Only valid on steps whose consumers do not read them (everything after the first
/// generated token when no per-step logit diagnostics were requested).
fn gpu_sampled_generation_step(
    next_token_id: u32,
    forward_us: u128,
) -> crate::Result<LlamaGenerationStep> {
    // This step's token was really sampled (on the GPU); the engine-level
    // sampler hook never runs for it, so the decode pulse is emitted here.
    telemetry::emit(telemetry::Event::TokenDecoded {
        token_id: Some(next_token_id),
        context_position: None,
        layers_total: None,
    });
    Ok(LlamaGenerationStep {
        prompt_token_count: 1,
        prefill_token_count: 0,
        next_token_id,
        logits: CpuTensor::from_f32("gpu_sampled_no_logits", vec![1, 0], Vec::new())?,
        hidden_state: CpuTensor::from_f32("gpu_sampled_no_hidden", vec![1, 0], Vec::new())?,
        output_norm_state: CpuTensor::from_f32("gpu_sampled_no_norm", vec![1, 0], Vec::new())?,
        timings: LlamaForwardTimings {
            total: forward_us,
            ..LlamaForwardTimings::default()
        },
        prefill_timings: LlamaForwardTimings::default(),
        first_token_timings: LlamaForwardTimings::default(),
        sample: 0,
        diagnostics: None,
    })
}

/// One streaming decode step, run ON THE ENGINE THREAD. No `spawn_blocking`,
/// no session ownership ping-pong (D4), no per-step detach window: the step
/// borrows the job-owned session and runs to completion. Wall-clock budget is
/// enforced BETWEEN steps by the streaming job (a step stuck inside a single
/// forward cannot be interrupted safely — same decision as non-streaming).
fn run_stream_step(
    session: &mut LlamaInferenceSession,
    request: StreamStepRequest,
) -> std::result::Result<LlamaGenerationStep, Box<Response>> {
    let StreamStepRequest {
        greedy_fast,
        input,
        sampler,
        history,
        collect_dense_diagnostics,
    } = request;
    let step_error = |err: String| {
        Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_step_failed",
            err,
            None,
        ))
    };
    if greedy_fast {
        if let Some((id, forward_us)) = session
            .generate_next_token_greedy_resident(input[0])
            .map_err(|err| step_error(err.to_string()))?
        {
            return gpu_sampled_generation_step(id, forward_us)
                .map_err(|err| step_error(err.to_string()));
        }
    }
    // Temperature-only sampling: draw the next token on the GPU (Gumbel-max)
    // instead of copying the full logits row to the host and sorting it on the
    // CPU (~7 ms/token). Other sampler shapes fall through to the CPU path.
    if !collect_dense_diagnostics {
        if let LlamaSampler::Sampling(cfg) = &sampler {
            if let Some((id, forward_us)) = session
                .generate_next_token_sampled_resident(input[0], cfg)
                .map_err(|err| step_error(err.to_string()))?
            {
                return gpu_sampled_generation_step(id, forward_us)
                    .map_err(|err| step_error(err.to_string()));
            }
        }
    }
    session
        .generate_next_token_with_history_diagnostics(
            &input,
            sampler,
            &history,
            collect_dense_diagnostics,
            None,
        )
        .map_err(|err| step_error(err.to_string()))
}

fn generation_timeout_response(
    timeout: Duration,
    elapsed: Duration,
    generated_tokens: Option<usize>,
) -> Box<Response> {
    Box::new(
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(generation_timeout_error_json(
                timeout,
                elapsed,
                generated_tokens,
            )),
        )
            .into_response(),
    )
}

fn generation_timeout_error_json(
    timeout: Duration,
    elapsed: Duration,
    generated_tokens: Option<usize>,
) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": format!(
                "generation exceeded the configured wall-clock timeout of {} ms; reduce max_tokens, use streaming/progress instrumentation, or raise {GENERATION_TIMEOUT_ENV} for a controlled hardening run",
                timeout.as_millis()
            ),
            "type": "runtime_unavailable",
            "code": "generation_timeout",
            "param": "max_tokens",
            "timeout_trace": {
                "timeout_ms": timeout.as_millis(),
                "elapsed_ms": elapsed.as_millis(),
                "generated_tokens": generated_tokens,
                "timeout_env": GENERATION_TIMEOUT_ENV,
            }
        }
    })
}

#[cfg(test)]
fn generation_step_test_sleep_duration() -> Option<Duration> {
    env::var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .map(Duration::from_millis)
}

/// Test-only concurrency probe for the decode jobs. Counts how many decode
/// bodies are live at once, measured on the compute itself rather than on any
/// serialization primitive — which is what let the Gate 0 tests catch compute
/// outliving its (since-deleted) generation-lock guard. Tests reset the probe
/// under `test_support::env_lock` and assert `max_seen() <= 1`.
#[cfg(test)]
pub(crate) mod decode_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static MAX_SEEN: AtomicUsize = AtomicUsize::new(0);

    pub(crate) struct ProbeGuard;

    impl Drop for ProbeGuard {
        fn drop(&mut self) {
            // Saturating: a guard leaked past its test (e.g. a detached
            // stream step finishing after `reset()` zeroed the counter) must
            // never underflow the gauge for later tests.
            let _ = ACTIVE.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                Some(value.saturating_sub(1))
            });
        }
    }

    pub(crate) fn enter() -> ProbeGuard {
        let now = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
        MAX_SEEN.fetch_max(now, Ordering::SeqCst);
        ProbeGuard
    }

    pub(crate) fn reset() {
        ACTIVE.store(0, Ordering::SeqCst);
        MAX_SEEN.store(0, Ordering::SeqCst);
    }

    pub(crate) fn active() -> usize {
        ACTIVE.load(Ordering::SeqCst)
    }

    pub(crate) fn max_seen() -> usize {
        MAX_SEEN.load(Ordering::SeqCst)
    }
}

#[cfg(not(test))]
fn generation_step_test_sleep_duration() -> Option<Duration> {
    None
}

fn generation_timeout_duration() -> std::result::Result<Duration, Box<Response>> {
    match env::var(GENERATION_TIMEOUT_ENV) {
        Ok(value) if value.trim().is_empty() => {
            Ok(Duration::from_millis(DEFAULT_GENERATION_TIMEOUT_MS))
        }
        Ok(value) => {
            let millis = value.trim().parse::<u64>().map_err(|err| {
                Box::new(api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "invalid_generation_timeout",
                    format!("invalid {GENERATION_TIMEOUT_ENV} {value:?}: {err}"),
                    None,
                ))
            })?;
            if millis == 0 {
                Ok(Duration::from_millis(u64::MAX))
            } else {
                Ok(Duration::from_millis(millis))
            }
        }
        Err(env::VarError::NotPresent) => Ok(Duration::from_millis(DEFAULT_GENERATION_TIMEOUT_MS)),
        Err(err) => Err(Box::new(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "invalid_generation_timeout",
            format!("invalid {GENERATION_TIMEOUT_ENV}: {err}"),
            None,
        ))),
    }
}

fn generate_decoded_tokens(
    mut prepared: PreparedGeneration,
) -> std::result::Result<GeneratedText, Box<Response>> {
    // Lifecycle telemetry for the blocking (non-streaming) path. If the
    // generation errors out, the guard's Drop closes the run on the stream.
    let telemetry_guard = prepared
        .telemetry
        .take()
        .map(telemetry::RequestGuard::begin);
    let tokenizer = prepared.tokenizer.clone();
    let stop_sequences = prepared.stop_sequences.clone();
    let generated = generate_token_ids(prepared)?;
    if let Some(guard) = telemetry_guard {
        guard.finish(telemetry::RequestFinish {
            status: "ok",
            finish_reason: Some(generated.finish_reason.to_string()),
            completion_tokens: generated.token_ids.len(),
            ttft_ms: None,
            decode_tps: None,
            prefill_tps: None,
            error: None,
        });
    }
    let mut text = tokenizer
        .decode(&generated.token_ids, true)
        .map_err(|err| {
            Box::new(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "token_decode_failed",
                err.to_string(),
                None,
            ))
        })?;
    if generated.finish_reason == "stop" {
        text = truncate_at_stop_sequence(text, &stop_sequences);
    }
    let top_logits = generated
        .top_logits
        .iter()
        .map(|entry| LogitDiagnostic {
            token_id: entry.token_id,
            logit: entry.logit,
            probability: entry.probability,
            rank: entry.rank,
            selected: entry.selected,
            text: tokenizer.decode(&[entry.token_id], true).ok(),
        })
        .collect();
    Ok(GeneratedText {
        text,
        prompt_token_ids: generated.prompt_token_ids,
        completion_tokens: generated.token_ids.len(),
        generated_token_ids: generated.token_ids,
        dense_metadata: generated.dense_metadata,
        top_logits,
        step_top_logits: generated
            .step_top_logits
            .iter()
            .map(|step| {
                step.iter()
                    .map(|entry| LogitDiagnostic {
                        token_id: entry.token_id,
                        logit: entry.logit,
                        probability: entry.probability,
                        rank: entry.rank,
                        selected: entry.selected,
                        text: tokenizer.decode(&[entry.token_id], true).ok(),
                    })
                    .collect()
            })
            .collect(),
        step_logprobs: generated.step_logprobs,
        output_projection: generated.output_projection,
        dense: generated.dense,
        dense_diagnostic_generated_index: generated.dense_diagnostic_generated_index,
        finish_reason: generated.finish_reason,
        timings: generated.timings,
        execution_trace: generated.execution_trace,
    })
}

fn clear_prompt_prefix_cache(state: &AppState) {
    if let Ok(mut pool) = state.cached_prompt_prefix.lock() {
        pool.clear();
    }
}

fn lookup_prompt_prefix_cache(prepared: &PreparedGeneration) -> Option<PrefixMatchResult> {
    if prepared.constraint.is_some() {
        return None;
    }
    let mut pool = prepared.cached_prompt_prefix.lock().ok()?;
    let min_prefix = prompt_prefix_cache_min_tokens_from_env();
    let mut best: Option<(usize, bool, usize, std::time::Instant)> = None;

    for (index, entry) in pool.entries.iter().enumerate() {
        let cached = &entry.cached;
        let same_cache_key = cached.model_id == prepared.model_id
            && cached.model_path == prepared.model_path
            && cached.sampling == prepared.sampling;
        if !same_cache_key || cached.session.kv_position() != cached.token_ids.len() {
            continue;
        }

        let common_len = common_prefix_len(&cached.token_ids, &prepared.token_ids);
        let is_exact_match = cached.token_ids == prepared.token_ids;
        if !is_exact_match && common_len < min_prefix {
            continue;
        }

        let candidate_rank = (is_exact_match, common_len, entry.last_used);
        let should_replace = best
            .as_ref()
            .map(|(_, exact, prefix_len, last_used)| {
                candidate_rank > (*exact, *prefix_len, *last_used)
            })
            .unwrap_or(true);
        if should_replace {
            best = Some((index, is_exact_match, common_len, entry.last_used));
        }
    }

    if let Some((index, is_exact_match, common_len, _)) = best {
        let entry = &mut pool.entries[index];
        entry.last_used = std::time::Instant::now();
        let cached = Arc::clone(&entry.cached);
        let prefix_len = if is_exact_match {
            common_len
        } else {
            common_len.min(prepared.token_ids.len().saturating_sub(1))
        };
        Some(PrefixMatchResult {
            cached,
            prefix_len,
            is_exact_match,
        })
    } else {
        None
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(&x, &y)| x == y).count()
}

fn store_prompt_prefix_cache(prepared: &PreparedGeneration, step: &LlamaGenerationStep) {
    if prepared.constraint.is_some() {
        return;
    }
    if !prepared.session.cpu_kv_authoritative()
        || prepared.session.kv_position() != prepared.token_ids.len()
    {
        return;
    }

    // Cloning a populated session can copy hundreds of MiB of KV data. Do that
    // outside the global cache lock so concurrent requests can still look up
    // their own prefixes.
    let cached = CachedPromptPrefix {
        model_id: prepared.model_id.clone(),
        model_path: prepared.model_path.clone(),
        token_ids: prepared.token_ids.clone(),
        sampling: prepared.sampling.clone(),
        session: prepared.session.clone(),
        logits: step.logits.clone(),
        hidden_state: step.hidden_state.clone(),
        output_norm_state: step.output_norm_state.clone(),
    };

    if let Ok(mut pool) = prepared.cached_prompt_prefix.lock() {
        pool.insert(cached);
    }
}

fn sample_cached_prompt_prefix(
    cached: &CachedPromptPrefix,
    history: &[u32],
) -> std::result::Result<LlamaGenerationStep, Box<Response>> {
    let sampler = if cached.sampling == SamplingConfig::default() {
        LlamaSampler::Greedy
    } else {
        LlamaSampler::Sampling(cached.sampling.clone())
    };
    let sample_started = Instant::now();
    let next_token_id = sampler
        .sample_with_history(&cached.logits, history)
        .map_err(|err| {
            Box::new(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "generation_step_failed",
                err.to_string(),
                None,
            ))
        })?;
    Ok(LlamaGenerationStep {
        prompt_token_count: cached.token_ids.len(),
        prefill_token_count: cached.token_ids.len().saturating_sub(1),
        next_token_id,
        logits: cached.logits.clone(),
        hidden_state: cached.hidden_state.clone(),
        output_norm_state: cached.output_norm_state.clone(),
        timings: LlamaForwardTimings::default(),
        prefill_timings: LlamaForwardTimings::default(),
        first_token_timings: LlamaForwardTimings::default(),
        sample: sample_started.elapsed().as_micros(),
        diagnostics: None,
    })
}

struct GenerationStepAccumulator<'a> {
    generated: &'a mut Vec<u32>,
    history: &'a mut Vec<u32>,
    top_logits: &'a mut Vec<RawLogitDiagnostic>,
    output_projection: &'a mut Vec<LlamaOutputProjectionDiagnostic>,
    dense: &'a mut Option<LlamaForwardDiagnostics>,
    finish_reason: &'a mut &'static str,
}

fn consume_generation_step(
    prepared: &PreparedGeneration,
    step: LlamaGenerationStep,
    acc: GenerationStepAccumulator<'_>,
) -> std::result::Result<(), Box<Response>> {
    if acc.top_logits.is_empty() {
        *acc.top_logits =
            top_logit_diagnostics(&step.logits, 8, &prepared.logit_diagnostic_token_ids).map_err(
                |err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "logit_diagnostic_failed",
                        err.to_string(),
                        None,
                    ))
                },
            )?;
        let projection_token_ids = acc
            .top_logits
            .iter()
            .map(|entry| entry.token_id)
            .collect::<Vec<_>>();
        if prepared.collect_dense_diagnostics {
            *acc.output_projection = output_projection_diagnostics(
                &step.output_norm_state,
                prepared.session.weights.output_projection(),
                &step.logits,
                &projection_token_ids,
                Some(&step.hidden_state),
                Some(&prepared.session.weights.output_norm),
                step.diagnostics
                    .as_ref()
                    .map(|diagnostics| diagnostics.final_norm.scale),
            )
            .map_err(|err| {
                Box::new(api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "output_projection_diagnostic_failed",
                    err.to_string(),
                    None,
                ))
            })?;
            *acc.dense = step.diagnostics.clone();
        }
    }
    acc.generated.push(step.next_token_id);
    acc.history.push(step.next_token_id);
    if prepared.tokenizer.special.eog.contains(&step.next_token_id) {
        *acc.finish_reason = "stop";
    } else if !prepared.stop_sequences.is_empty() {
        let text = prepared
            .tokenizer
            .decode(acc.generated, true)
            .map_err(|err| {
                Box::new(api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "token_decode_failed",
                    err.to_string(),
                    None,
                ))
            })?;
        if contains_stop_sequence(&text, &prepared.stop_sequences) {
            *acc.finish_reason = "stop";
        }
    }
    Ok(())
}

fn generate_token_ids(
    mut prepared: PreparedGeneration,
) -> std::result::Result<GeneratedTokens, Box<Response>> {
    let generation_started = Instant::now();
    let collect_q8_schedule = q8_schedule_telemetry_enabled();
    if collect_q8_schedule {
        reset_q8_schedule_telemetry();
    }
    let mut input = prepared.token_ids.clone();
    let mut history = prepared.token_ids.clone();
    let mut generated = Vec::new();
    let mut top_logits = Vec::new();
    let mut step_top_logits = Vec::new();
    let mut step_logprobs: Vec<StepLogprob> = Vec::new();
    // Constrained-decoding setup (response_format json_object / json_schema). Cache
    // each token's output bytes once; mask the logits to tokens that keep a valid
    // prefix of the constrained value each step; stop as soon as it completes.
    let constraint_active = prepared.constraint.is_some();
    let grammar_vocab = prepared.tokenizer.tokens.len();
    let grammar_token_bytes: Vec<Vec<u8>> = if constraint_active {
        (0..grammar_vocab as u32)
            .map(|id| {
                prepared
                    .tokenizer
                    .decode(&[id], false)
                    .unwrap_or_default()
                    .into_bytes()
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut grammar: Option<crate::grammar::ConstraintState> = prepared
        .constraint
        .as_ref()
        .map(crate::grammar::ConstraintSpec::build);
    let mut grammar_mask: Vec<bool> =
        vec![false; if constraint_active { grammar_vocab } else { 0 }];
    let collect_step_top_logits = !prepared.logit_diagnostic_token_ids.is_empty();
    let mut output_projection = Vec::new();
    let mut dense = None;
    let mut dense_diagnostic_generated_index = None;
    let mut finish_reason = "length";
    let mut forward_timings = LlamaForwardTimings::default();
    let mut sample = 0;

    // The execution-trace rollup is captured only on the deterministic CPU lane (the only lane
    // where it is reduction-order-stable). When it will be armed, bypass the prompt-prefix cache
    // so the served run and any later verify re-run fold an identical forward (a cache hit skips
    // the prompt forwards on one side only, which would desync the digest).
    let want_execution_trace = crate::inference::deterministic_mode_enabled();
    // The GPU-resident CUDA decode engine reuses a cached prompt-prefix session by
    // reseeding the GPU KV from the f16-rounded host history and resuming, which is not
    // bit-identical to a fresh GPU prefill (different reduction order) and flips
    // near-tie tokens. Bypass the cache whenever that engine drives decode so every
    // request takes the clean prefill path; the CPU lane stays reduction-order-stable
    // and keeps the cache.
    let resident_cuda_active = crate::inference::resident_decode_cuda_active();

    if !prepared.collect_dense_diagnostics && !want_execution_trace && !resident_cuda_active {
        if let Some(match_res) = lookup_prompt_prefix_cache(&prepared) {
            let mut cached_session = match_res.cached.session.clone();
            // The cached session's resident-path pin reflects the request
            // that stored it; re-pin for this request's mode.
            cached_session
                .set_resident_paths_disabled(prepared.speculative.is_some() && !spec_gpu_enabled());

            if match_res.is_exact_match {
                prepared.session = cached_session;
                input.clear();
                let first_step = sample_cached_prompt_prefix(&match_res.cached, &history)?;
                let cached_next_token = first_step.next_token_id;
                sample += first_step.sample;
                prepared.timings.prompt_cache_hit = true;
                consume_generation_step(
                    &prepared,
                    first_step,
                    GenerationStepAccumulator {
                        generated: &mut generated,
                        history: &mut history,
                        top_logits: &mut top_logits,
                        output_projection: &mut output_projection,
                        dense: &mut dense,
                        finish_reason: &mut finish_reason,
                    },
                )?;
                if finish_reason == "length" {
                    input.push(cached_next_token);
                }
            } else {
                let k = match_res.prefix_len;
                // A malformed/stale cache entry must never make generation fail:
                // retain the fresh session and perform a cold prefill if rollback
                // cannot establish the requested prefix position.
                if cached_session.rollback_to_position(k).is_ok() {
                    prepared.session = cached_session;
                    input = prepared.token_ids[k..].to_vec();
                    prepared.timings.prompt_cache_hit = true;
                }
            }
        }
    }

    // Arm the rollup now that the session is settled (past any prompt-cache swap). Fails closed
    // unless deterministic mode is active, so non-deterministic generations never trace.
    let tracing_armed = want_execution_trace && prepared.session.enable_execution_trace();

    for _ in generated.len() as u32..prepared.max_tokens {
        if finish_reason != "length" {
            break;
        }
        // Speculative rounds emit several tokens per loop iteration, so the
        // range alone no longer bounds the budget; enforce max_tokens on the
        // emitted count directly.
        if generated.len() >= prepared.max_tokens as usize {
            break;
        }
        // Cooperative stop: cancellation (the owning handler frame is gone) or
        // the engine-enforced wall-clock deadline. Checked once per step (per
        // speculative ROUND on the spec path), so compute never outlives its
        // request by more than one step. The timeout payload keeps the exact
        // pre-inversion shape (generated_tokens stays null on this path).
        match prepared.cancel.tripped() {
            Some(GenerationStopCause::TimedOut) => {
                return Err(generation_timeout_response(
                    prepared.cancel.timeout,
                    prepared.cancel.started.elapsed(),
                    None,
                ));
            }
            Some(GenerationStopCause::Cancelled) => {
                return Err(generation_cancelled_response(generated.len()));
            }
            None => {}
        }
        let generated_index = generated.len();
        let collect_dense_for_step =
            collect_dense_diagnostics_for_generated_index(&prepared, generated_index);
        let mut sampling = prepared.sampling.clone();
        if let Some(seed) = sampling.seed {
            sampling.seed = Some(seed.wrapping_add(generated.len() as u64));
        }
        let sampler = if sampling == SamplingConfig::default() {
            LlamaSampler::Greedy
        } else {
            LlamaSampler::Sampling(sampling)
        };
        // Lossless greedy speculation: draft tokens, verify them in ONE
        // batched forward (one weight read for the whole batch), accept the
        // longest matching prefix plus the target's own next token, and roll
        // rejected KV entries back. Every emitted token is the target's own
        // greedy argmax given its accepted prefix. Engages only after the
        // first step (prompt evaluated, first-step diagnostics captured) and
        // never alongside per-step logit consumers.
        if let Some(spec) = prepared.speculative.as_mut().filter(|_| {
            input.len() == 1
                && matches!(sampler, LlamaSampler::Greedy)
                && !collect_dense_for_step
                && !collect_step_top_logits
                && prepared.logprobs_top_n.is_none()
                && grammar.is_none()
                && !top_logits.is_empty()
        }) {
            if !spec.latch.should_speculate() {
                spec.latch.note_skip();
            } else {
                let remaining = (prepared.max_tokens as usize).saturating_sub(generated.len());
                let context_room = prepared.session.remaining_context();
                let drafts = if remaining > 0 && context_room > 0 {
                    let draft_budget = spec
                        .draft_tokens
                        .min(remaining.saturating_sub(1))
                        .min(context_room.saturating_sub(1));
                    spec.drafter.draft(&history, draft_budget).map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "speculative_draft_failed",
                            err.to_string(),
                            None,
                        ))
                    })?
                } else {
                    Vec::new()
                };
                // No drafts (e.g. no n-gram match) -> fall through to the plain
                // single-token step below; a one-token verify chunk would only
                // add chunk-path overhead over the tuned decode step. This is an
                // anchor miss, not a latch-directed skip or an acceptance
                // measurement, so it must not advance the re-probe cooldown.
                if !drafts.is_empty() {
                    // GPU speculative decode (CAMELID_SPEC_GPU=1): verify all drafts in one
                    // batched forward on the target's resident engine, which manages the KV
                    // itself (accept the longest matching prefix, advance position). Falls
                    // back to the CPU chunk verify when the engine isn't resident-ready.
                    // Lossless either way — the emitted tokens are the target's own greedy
                    // argmax given the accepted prefix.
                    let gpu_accepted = if spec_gpu_enabled() {
                        prepared
                            .session
                            .verify_drafts_gpu(input[0], &drafts)
                            .map_err(|err| {
                                Box::new(api_error(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "speculative_verify_failed",
                                    err.to_string(),
                                    None,
                                ))
                            })?
                    } else {
                        None
                    };
                    let emitted: Vec<u32> = if let Some(acc) = gpu_accepted {
                        let accepted_count = (acc.len() as u64).saturating_sub(1);
                        spec.rounds += 1;
                        spec.drafted += drafts.len() as u64;
                        spec.accepted_drafts += accepted_count;
                        spec.latch.note_verified(accepted_count as u32);
                        acc
                    } else {
                        let base_position = prepared.session.kv_position();
                        let mut batch = Vec::with_capacity(1 + drafts.len());
                        batch.push(input[0]);
                        batch.extend_from_slice(&drafts);
                        let (predictions, round_timings) = prepared
                            .session
                            .forward_greedy_verify_chunk(&batch)
                            .map_err(|err| {
                                Box::new(api_error(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "speculative_verify_failed",
                                    err.to_string(),
                                    None,
                                ))
                            })?;
                        let accepted = accepted_draft_prefix(&drafts, &predictions);
                        prepared
                            .session
                            .rollback_to_position(base_position + 1 + accepted)
                            .map_err(|err| {
                                Box::new(api_error(
                                    StatusCode::SERVICE_UNAVAILABLE,
                                    "speculative_rollback_failed",
                                    err.to_string(),
                                    None,
                                ))
                            })?;
                        spec.rounds += 1;
                        spec.drafted += drafts.len() as u64;
                        spec.accepted_drafts += accepted as u64;
                        spec.latch.note_verified(accepted as u32);
                        forward_timings.add_assign(&round_timings);
                        predictions[..=accepted].to_vec()
                    };
                    for &token in &emitted {
                        generated.push(token);
                        history.push(token);
                        if prepared.tokenizer.special.eog.contains(&token) {
                            finish_reason = "stop";
                            break;
                        }
                        if !prepared.stop_sequences.is_empty() {
                            let text =
                                prepared.tokenizer.decode(&generated, true).map_err(|err| {
                                    Box::new(api_error(
                                        StatusCode::SERVICE_UNAVAILABLE,
                                        "token_decode_failed",
                                        err.to_string(),
                                        None,
                                    ))
                                })?;
                            if contains_stop_sequence(&text, &prepared.stop_sequences) {
                                finish_reason = "stop";
                                break;
                            }
                        }
                    }
                    if finish_reason != "length" || generated.len() >= prepared.max_tokens as usize
                    {
                        break;
                    }
                    input.clear();
                    input.push(*history.last().expect("history grows every round"));
                    continue;
                }
            }
        }
        // JSON-grammar mask for this step: a token is allowed iff its bytes keep a
        // valid JSON-object prefix; EOG only once the object is complete.
        let grammar_allowed: Option<&[bool]> = match grammar.as_ref() {
            Some(state) => {
                // `done` gates whether EOG is allowed this step. With the current loop
                // structure it is always false here (the loop breaks the moment a token
                // drives the state to done, below); it is kept defensive so the mask
                // stays correct if that stop/advance ordering ever changes.
                let done = state.is_done();
                let mut any_allowed = false;
                for (id, slot) in grammar_mask.iter_mut().enumerate() {
                    let bytes = grammar_token_bytes
                        .get(id)
                        .map(|v| v.as_slice())
                        .unwrap_or_default();
                    *slot = if prepared.tokenizer.special.eog.contains(&(id as u32)) {
                        done
                    } else if bytes.is_empty() {
                        false
                    } else {
                        state.accepts(bytes)
                    };
                    any_allowed |= *slot;
                }
                // Fail closed if no token can extend the constrained value and
                // stopping is not yet allowed. This arises when the next required byte
                // is only reachable through a multibyte-UTF-8 fragment token:
                // `grammar_token_bytes` decodes each token in isolation and drops
                // incomplete UTF-8, so a byte-level-BPE / byte-fallback fragment decodes
                // to "" and is masked out above (having a byte token is not enough — it
                // must survive an in-isolation decode). Non-ASCII `enum`/`const` literals,
                // which would force exactly such a byte, are rejected at schema-compile
                // time (a request-time 400), so in practice this only guards free-form
                // values on pathological tokenizers; sampling a fully-masked (all -inf)
                // distribution would otherwise emit an invalid token or NaN, so we
                // surface a typed error instead.
                if !any_allowed {
                    return Err(Box::new(api_error(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "constraint_unsatisfiable",
                        "the response_format constraint cannot be satisfied by this model's tokenizer (no token can produce the next required byte)".to_string(),
                        Some("response_format"),
                    )));
                }
                Some(grammar_mask.as_slice())
            }
            None => None,
        };
        // Single-token continuations with no per-step logit consumers ride the
        // resident GPU fast lane: greedy via GPU argmax, temperature-only sampling
        // via GPU Gumbel-max. Everything else takes the general step.
        let fast_step = if input.len() == 1
            && !collect_dense_for_step
            && !collect_step_top_logits
            && prepared.logprobs_top_n.is_none()
            && grammar.is_none()
            && !top_logits.is_empty()
        {
            match &sampler {
                LlamaSampler::Greedy => prepared
                    .session
                    .generate_next_token_greedy_resident(input[0])
                    .map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "generation_step_failed",
                            err.to_string(),
                            None,
                        ))
                    })?,
                LlamaSampler::Sampling(cfg) => prepared
                    .session
                    .generate_next_token_sampled_resident(input[0], cfg)
                    .map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "generation_step_failed",
                            err.to_string(),
                            None,
                        ))
                    })?,
            }
        } else {
            None
        };
        let step = match fast_step {
            Some((id, forward_us)) => {
                gpu_sampled_generation_step(id, forward_us).map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "generation_step_failed",
                        err.to_string(),
                        None,
                    ))
                })?
            }
            None => prepared
                .session
                .generate_next_token_with_history_diagnostics(
                    &input,
                    sampler,
                    &history,
                    collect_dense_for_step,
                    grammar_allowed,
                )
                .map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "generation_step_failed",
                        err.to_string(),
                        None,
                    ))
                })?,
        };
        if generated.is_empty()
            && !prepared.collect_dense_diagnostics
            && step.diagnostics.is_none()
            && !resident_cuda_active
        {
            // Don't cache a GPU-prefilled session: it can only be replayed bit-exactly
            // by a fresh GPU prefill, and the lookup above is skipped while the resident
            // CUDA engine is active anyway. Keeping it out also avoids a CPU request
            // (after a GPU-off toggle) reusing a GPU-built (f16-seeded) session.
            store_prompt_prefix_cache(&prepared, &step);
        }
        if generated.is_empty() {
            prepared.timings.prompt_evaluation = prompt_evaluation_timings_from_step(&step);
        }
        forward_timings.add_assign(&step.timings);
        sample += step.sample;
        if top_logits.is_empty() || collect_step_top_logits || collect_dense_for_step {
            let current_top_logits =
                top_logit_diagnostics(&step.logits, 8, &prepared.logit_diagnostic_token_ids)
                    .map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "logit_diagnostic_failed",
                            err.to_string(),
                            None,
                        ))
                    })?;
            if collect_step_top_logits {
                step_top_logits.push(current_top_logits.clone());
            }
            if top_logits.is_empty() {
                top_logits = current_top_logits.clone();
            }
            if collect_dense_for_step && dense.is_none() {
                let projection_token_ids = current_top_logits
                    .iter()
                    .map(|entry| entry.token_id)
                    .collect::<Vec<_>>();
                output_projection = output_projection_diagnostics(
                    &step.output_norm_state,
                    prepared.session.weights.output_projection(),
                    &step.logits,
                    &projection_token_ids,
                    Some(&step.hidden_state),
                    Some(&prepared.session.weights.output_norm),
                    step.diagnostics
                        .as_ref()
                        .map(|diagnostics| diagnostics.final_norm.scale),
                )
                .map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "output_projection_diagnostic_failed",
                        err.to_string(),
                        None,
                    ))
                })?;
                dense = step.diagnostics.clone();
                dense_diagnostic_generated_index = Some(generated_index);
            }
        }
        if let Some(top_n) = prepared.logprobs_top_n {
            step_logprobs.push(
                compute_step_logprobs(
                    &step.logits.data,
                    step.next_token_id,
                    top_n,
                    &prepared.tokenizer,
                )
                .map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "logprob_capture_failed",
                        err.to_string(),
                        None,
                    ))
                })?,
            );
        }
        generated.push(step.next_token_id);
        history.push(step.next_token_id);
        // Advance the JSON grammar by the chosen token's bytes; stop the moment the
        // top-level object closes (the mask guaranteed the bytes are acceptable).
        if let Some(state) = grammar.as_mut() {
            if let Some(bytes) = grammar_token_bytes.get(step.next_token_id as usize) {
                for &b in bytes {
                    // The mask guaranteed every byte of the chosen token is acceptable.
                    // Advance in all builds (the grammar must consume the emitted bytes)
                    // and assert acceptance in debug, so a mask/advance divergence fails
                    // a test instead of silently emitting invalid output.
                    let accepted = state.advance(b).is_ok();
                    debug_assert!(
                        accepted,
                        "constrained-decode mask allowed a token whose bytes the grammar rejects"
                    );
                }
            }
            if state.is_done() {
                finish_reason = "stop";
                break;
            }
        }
        if prepared.tokenizer.special.eog.contains(&step.next_token_id) {
            finish_reason = "stop";
            break;
        }
        if !prepared.stop_sequences.is_empty() {
            let text = prepared.tokenizer.decode(&generated, true).map_err(|err| {
                Box::new(api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "token_decode_failed",
                    err.to_string(),
                    None,
                ))
            })?;
            if contains_stop_sequence(&text, &prepared.stop_sequences) {
                finish_reason = "stop";
                break;
            }
        }
        input.clear();
        input.push(step.next_token_id);
    }

    if let Some(spec) = &prepared.speculative {
        let acceptance_pct = if spec.drafted == 0 {
            0.0
        } else {
            spec.accepted_drafts as f64 * 100.0 / spec.drafted as f64
        };
        tracing::info!(
            rounds = spec.rounds,
            drafted = spec.drafted,
            accepted_drafts = spec.accepted_drafts,
            acceptance_pct,
            generated = generated.len(),
            "speculative decode summary"
        );
    }

    prepared.timings.generate = generation_started.elapsed().as_millis();
    prepared.timings.generation = generation_phase_timings_from_forward(&forward_timings, sample);
    prepared.timings.layers = generation_layer_timings_from_forward(&forward_timings.layers);
    prepared.timings.memory = forward_timings.memory;
    if collect_q8_schedule {
        prepared.timings.q8_schedule = Some(snapshot_q8_schedule_telemetry());
    }

    // Finalize the rollup before any field is moved out of `prepared`.
    let execution_trace = if tracing_armed {
        let fold_count = prepared.session.execution_trace_fold_count();
        prepared
            .session
            .take_execution_trace_digest()
            .zip(fold_count)
    } else {
        None
    };
    Ok(GeneratedTokens {
        prompt_token_ids: prepared.token_ids,
        token_ids: generated,
        dense_metadata: prepared.dense_metadata,
        top_logits,
        step_top_logits,
        step_logprobs,
        output_projection,
        dense,
        dense_diagnostic_generated_index,
        finish_reason,
        timings: prepared.timings,
        execution_trace,
    })
}

/// Numeric core (tokenizer-free, testable): the chosen-token logprob plus the
/// top-N `(token_id, logprob)` by probability, from a stable f64 log-sum-exp over
/// the full vocab. `logprob[t] = logit[t] - logsumexp(logits)`.
fn step_logprob_values(
    logits: &[f32],
    chosen: u32,
    top_n: usize,
) -> crate::Result<(f32, Vec<(u32, f32)>)> {
    if logits.is_empty() {
        return Err(BackendError::RuntimeShapeMismatch(
            "logprob capture received empty logits".to_string(),
        ));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        if !l.is_finite() {
            return Err(BackendError::RuntimeShapeMismatch(
                "logprob capture received a non-finite logit".to_string(),
            ));
        }
        sum += ((l - max) as f64).exp();
    }
    let log_sum_exp = max as f64 + sum.ln();
    let logprob_of = |id: usize| -> f32 { (logits[id] as f64 - log_sum_exp) as f32 };

    let chosen_idx = chosen as usize;
    if chosen_idx >= logits.len() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "chosen token {chosen} is outside vocabulary size {}",
            logits.len()
        )));
    }
    let chosen_lp = logprob_of(chosen_idx);

    let mut top = Vec::new();
    if top_n > 0 {
        let mut ranked: Vec<(u32, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(id, l)| (id as u32, l))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for (id, _) in ranked.iter().take(top_n) {
            top.push((*id, logprob_of(*id as usize)));
        }
    }
    Ok((chosen_lp, top))
}

/// Compute per-token logprobs for one decode step: the chosen token plus the top-N
/// alternatives, each decoded to its OpenAI-shaped piece + bytes.
fn compute_step_logprobs(
    logits: &[f32],
    chosen: u32,
    top_n: usize,
    tokenizer: &Tokenizer,
) -> crate::Result<StepLogprob> {
    let (chosen_lp, top) = step_logprob_values(logits, chosen, top_n)?;
    let chosen_entry = token_logprob(chosen, chosen_lp, tokenizer)?;
    let top = top
        .into_iter()
        .map(|(id, lp)| token_logprob(id, lp, tokenizer))
        .collect::<crate::Result<Vec<_>>>()?;
    Ok(StepLogprob {
        chosen: chosen_entry,
        top,
    })
}

/// Decode a single token to its OpenAI-shaped logprob entry (piece text + raw bytes).
fn token_logprob(
    token_id: u32,
    logprob: f32,
    tokenizer: &Tokenizer,
) -> crate::Result<TokenLogprob> {
    let token = tokenizer.decode(&[token_id], false)?;
    let bytes = token.clone().into_bytes();
    Ok(TokenLogprob {
        token,
        logprob,
        bytes,
    })
}

/// Build the OpenAI chat `logprobs` object (one `content` entry per token).
fn build_chat_logprobs(steps: &[StepLogprob]) -> ChatLogprobs {
    ChatLogprobs {
        content: steps
            .iter()
            .map(|s| ChatLogprobContent {
                token: s.chosen.token.clone(),
                logprob: s.chosen.logprob,
                bytes: s.chosen.bytes.clone(),
                top_logprobs: s
                    .top
                    .iter()
                    .map(|t| ChatTopLogprob {
                        token: t.token.clone(),
                        logprob: t.logprob,
                        bytes: t.bytes.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Build the OpenAI legacy-completions `logprobs` object (parallel arrays).
/// `text_offset` is the running char offset of each token's piece in the output.
fn build_completion_logprobs(steps: &[StepLogprob]) -> CompletionLogprobs {
    let mut tokens = Vec::with_capacity(steps.len());
    let mut token_logprobs = Vec::with_capacity(steps.len());
    let mut top_logprobs = Vec::with_capacity(steps.len());
    let mut text_offset = Vec::with_capacity(steps.len());
    let mut offset = 0usize;
    for s in steps {
        text_offset.push(offset);
        offset += s.chosen.token.chars().count();
        tokens.push(s.chosen.token.clone());
        token_logprobs.push(s.chosen.logprob);
        let mut map = std::collections::BTreeMap::new();
        for t in &s.top {
            map.insert(t.token.clone(), t.logprob);
        }
        top_logprobs.push(map);
    }
    CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }
}

/// Interpret OpenAI `response_format` into an optional output constraint.
/// `Ok(None)` = normal decoding (text / absent); `Ok(Some(_))` = constrained
/// decoding (`json_object`, or `json_schema` whose schema is inside the supported
/// subset); `Err` = a typed 400 for an unsupported shape or a schema Camelid cannot
/// enforce byte-for-byte.
fn constraint_from_response_format(
    response_format: Option<&serde_json::Value>,
) -> std::result::Result<Option<crate::grammar::ConstraintSpec>, Box<Response>> {
    let Some(value) = response_format.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    match value.get("type").and_then(serde_json::Value::as_str) {
        Some("json_object") => Ok(Some(crate::grammar::ConstraintSpec::json_object())),
        Some("json_schema") => {
            let schema = value
                .get("json_schema")
                .and_then(|json_schema| json_schema.get("schema"))
                .ok_or_else(|| {
                    Box::new(api_error(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "response_format json_schema requires a `json_schema.schema` object"
                            .to_string(),
                        Some("response_format"),
                    ))
                })?;
            crate::grammar::ConstraintSpec::from_schema(schema)
                .map(Some)
                .map_err(|err| {
                    Box::new(api_error(
                        StatusCode::BAD_REQUEST,
                        "unsupported_parameter",
                        format!("response_format json_schema is not supported: {err}"),
                        Some("response_format"),
                    ))
                })
        }
        Some("text") | None => Ok(None),
        Some(other) => Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            format!(
                "response_format type {other:?} is not supported; only json_object, json_schema, and text are honored"
            ),
            Some("response_format"),
        ))),
    }
}

/// Whether `tool_choice` permits surfacing tool calls. `"none"` suppresses them;
/// everything else (auto / required / a specific function / absent) allows.
fn tool_choice_allows_calls(tool_choice: Option<&serde_json::Value>) -> bool {
    !matches!(tool_choice.and_then(|value| value.as_str()), Some("none"))
}

/// Whether the model's raw content should be probed for a tool call.
/// Constrained output is content by definition: OpenAI allows tools and
/// response_format together, and a schema that legitimately declares a `name`
/// property (extremely common) must not have its constrained output
/// reclassified into a fabricated tool call with emptied content. The tools are
/// still rendered into the prompt template under a constraint; only this
/// post-hoc reclassification is disabled.
fn should_parse_tool_calls(tools_active: bool, constraint_active: bool) -> bool {
    tools_active && !constraint_active
}

/// Parse a model's tool-call output into OpenAI `tool_calls`. Handles the Llama
/// 3.x form `{"name": <fn>, "parameters": {...}}` (also `"arguments"`), optionally
/// `<|python_tag|>`-prefixed, and tolerates trailing junk small models emit.
/// Returns `None` when the text is prose, not a tool call.
fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let trimmed = text.trim();
    let trimmed = trimmed
        .strip_prefix("<|python_tag|>")
        .unwrap_or(trimmed)
        .trim_start();
    // Read the first complete JSON value; ignore any trailing junk.
    let value = serde_json::Deserializer::from_str(trimmed)
        .into_iter::<serde_json::Value>()
        .next()?
        .ok()?;
    let obj = value.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let args = obj
        .get("parameters")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let arguments = serde_json::to_string(&args).ok()?;
    Some(vec![ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        kind: "function",
        function: ToolCallFunction { name, arguments },
    }])
}

fn top_logit_diagnostics(
    logits: &crate::tensor::CpuTensor,
    count: usize,
    selected_token_ids: &[u32],
) -> crate::Result<Vec<RawLogitDiagnostic>> {
    if logits.shape.dims.len() != 2 || logits.shape.dims[0] != 1 {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "logit diagnostics expected shape [1, vocab], got {:?}",
            logits.shape.dims
        )));
    }
    let mut ranked = logits
        .data
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, logit)| {
            if !logit.is_finite() {
                Err(BackendError::RuntimeShapeMismatch(format!(
                    "logit diagnostics contain non-finite value at token {token_id}"
                )))
            } else {
                Ok((token_id as u32, logit))
            }
        })
        .collect::<crate::Result<Vec<_>>>()?;
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let max_logit = ranked.first().map(|(_, logit)| *logit).unwrap_or(0.0);
    let probability_denominator = ranked
        .iter()
        .map(|(_, logit)| (*logit - max_logit).exp())
        .sum::<f32>();
    let probability_for = |logit: f32| {
        if probability_denominator > 0.0 {
            (logit - max_logit).exp() / probability_denominator
        } else {
            0.0
        }
    };

    let mut entries = Vec::new();
    for (rank, (token_id, logit)) in ranked.iter().take(count).enumerate() {
        entries.push(RawLogitDiagnostic {
            token_id: *token_id,
            logit: *logit,
            probability: probability_for(*logit),
            rank: rank + 1,
            selected: selected_token_ids.contains(token_id),
        });
    }
    for selected_token_id in selected_token_ids {
        if entries
            .iter()
            .any(|entry| entry.token_id == *selected_token_id)
        {
            continue;
        }
        let Some((rank, (_, logit))) = ranked
            .iter()
            .enumerate()
            .find(|(_, (token_id, _))| token_id == selected_token_id)
        else {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "logit diagnostic token id {selected_token_id} is outside vocabulary size {}",
                logits.data.len()
            )));
        };
        entries.push(RawLogitDiagnostic {
            token_id: *selected_token_id,
            logit: *logit,
            probability: probability_for(*logit),
            rank: rank + 1,
            selected: true,
        });
    }
    Ok(entries)
}

fn prompt_evaluation_timings_from_step(step: &LlamaGenerationStep) -> PromptEvaluationTimings {
    PromptEvaluationTimings {
        prompt_token_count: step.prompt_token_count,
        prefill_token_count: step.prefill_token_count,
        first_token_evaluated: step.prompt_token_count > 0,
        prefill: generation_phase_timings_from_forward(&step.prefill_timings, 0),
        first_token: generation_phase_timings_from_forward(&step.first_token_timings, step.sample),
        prefill_layers: generation_layer_timings_from_forward(&step.prefill_timings.layers),
        first_token_layers: generation_layer_timings_from_forward(&step.first_token_timings.layers),
        prefill_memory: step.prefill_timings.memory.clone(),
        first_token_memory: step.first_token_timings.memory.clone(),
    }
}

fn generation_phase_timings_from_forward(
    forward_timings: &LlamaForwardTimings,
    sample: u128,
) -> GenerationPhaseTimings {
    GenerationPhaseTimings {
        forward_total: micros_to_ms(forward_timings.total),
        embedding: micros_to_ms(forward_timings.embedding),
        layers_total: micros_to_ms(forward_timings.layers_total),
        final_norm: micros_to_ms(forward_timings.final_norm),
        logits: micros_to_ms(forward_timings.logits),
        sample: micros_to_ms(sample),
    }
}

fn generation_layer_timings_from_forward(
    layers: &[LlamaLayerTimings],
) -> Vec<GenerationLayerTimings> {
    layers
        .iter()
        .map(|layer| GenerationLayerTimings {
            layer_index: layer.layer_index,
            total: micros_to_ms(layer.total),
            attention_norm: micros_to_ms(layer.attention_norm),
            attention_q: micros_to_ms(layer.attention_q),
            attention_k: micros_to_ms(layer.attention_k),
            attention_v: micros_to_ms(layer.attention_v),
            attention_rope: micros_to_ms(layer.attention_rope),
            kv_cache_write: micros_to_ms(layer.kv_cache_write),
            attention_context: micros_to_ms(layer.attention_context),
            attention_output: micros_to_ms(layer.attention_output),
            attention_residual: micros_to_ms(layer.attention_residual),
            ffn_norm: micros_to_ms(layer.ffn_norm),
            ffn_gate: micros_to_ms(layer.ffn_gate),
            ffn_up: micros_to_ms(layer.ffn_up),
            ffn_activation: micros_to_ms(layer.ffn_activation),
            ffn_down: micros_to_ms(layer.ffn_down),
            ffn_residual: micros_to_ms(layer.ffn_residual),
            memory: layer.memory.clone(),
        })
        .collect()
}

fn micros_to_ms(value: u128) -> f64 {
    value as f64 / 1000.0
}

fn stream_timing_diagnostics_enabled() -> bool {
    matches!(
        env::var(STREAM_TIMING_DIAGNOSTICS_ENV).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON") | Ok("yes") | Ok("YES")
    )
}

fn stream_poll_yield_enabled() -> bool {
    matches!(
        env::var(STREAM_POLL_YIELD_ENV).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON") | Ok("yes") | Ok("YES")
    )
}

fn stream_role_timings_json(
    phase: &GenerationPhaseTimings,
    layers: &[GenerationLayerTimings],
) -> serde_json::Value {
    let sum = |value: fn(&GenerationLayerTimings) -> f64| -> f64 {
        layers.iter().map(value).sum::<f64>()
    };
    serde_json::json!({
        "attention_context": sum(|layer| layer.attention_context),
        "attention_output": sum(|layer| layer.attention_output),
        "ffn_gate": sum(|layer| layer.ffn_gate),
        "ffn_up": sum(|layer| layer.ffn_up),
        "ffn_down": sum(|layer| layer.ffn_down),
        "logits": phase.logits,
    })
}

fn stream_layer_role_hotspots_json(
    layers: &[GenerationLayerTimings],
    limit: usize,
) -> serde_json::Value {
    let mut rows = Vec::new();
    for layer in layers {
        let roles = [
            ("attention_norm", layer.attention_norm),
            ("attention_q", layer.attention_q),
            ("attention_k", layer.attention_k),
            ("attention_v", layer.attention_v),
            ("attention_rope", layer.attention_rope),
            ("kv_cache_write", layer.kv_cache_write),
            ("attention_context", layer.attention_context),
            ("attention_output", layer.attention_output),
            ("attention_residual", layer.attention_residual),
            ("ffn_norm", layer.ffn_norm),
            ("ffn_gate", layer.ffn_gate),
            ("ffn_up", layer.ffn_up),
            ("ffn_activation", layer.ffn_activation),
            ("ffn_down", layer.ffn_down),
            ("ffn_residual", layer.ffn_residual),
        ];
        for (role, elapsed_ms) in roles {
            if elapsed_ms > 0.0 && elapsed_ms.is_finite() {
                rows.push((layer.layer_index, role, elapsed_ms));
            }
        }
    }
    rows.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(right.1))
    });

    serde_json::Value::Array(
        rows.into_iter()
            .take(limit)
            .map(|(layer_index, role, elapsed_ms)| {
                serde_json::json!({
                    "layer_index": layer_index,
                    "role": role,
                    "elapsed_ms": elapsed_ms,
                })
            })
            .collect(),
    )
}

fn stream_timing_diagnostics_json(
    timings: &GenerationTimings,
    first_content_ms: Option<u128>,
    stream_events: StreamEventTimings,
) -> serde_json::Value {
    let first_content_accounting = stream_first_content_accounting_json(timings, first_content_ms);
    serde_json::json!({
        "stream_timing_diagnostics": {
            "timings_ms": {
                "generate": timings.generate,
                "first_content": first_content_ms,
                "first_content_accounting": first_content_accounting,
                "stream_event_accounting": stream_event_accounting_json(stream_events),
                "tokenize": timings.tokenize,
                "weight_load": timings.weight_load,
                "weight_cache_hit": timings.weight_cache_hit,
                "prompt_cache_hit": timings.prompt_cache_hit,
                "session_create": timings.session_create,
                "prefill_forward_total": timings.prompt_evaluation.prefill.forward_total,
                "first_token_forward_total": timings.prompt_evaluation.first_token.forward_total,
                "generation_forward_total": timings.generation.forward_total,
                "prefill_role_timings": stream_role_timings_json(&timings.prompt_evaluation.prefill, &timings.prompt_evaluation.prefill_layers),
                "first_token_role_timings": stream_role_timings_json(&timings.prompt_evaluation.first_token, &timings.prompt_evaluation.first_token_layers),
                "generation_role_timings": stream_role_timings_json(&timings.generation, &timings.layers),
                "layer_role_hotspots": {
                    "prefill": stream_layer_role_hotspots_json(&timings.prompt_evaluation.prefill_layers, 10),
                    "first_token": stream_layer_role_hotspots_json(&timings.prompt_evaluation.first_token_layers, 10),
                    "generation": stream_layer_role_hotspots_json(&timings.layers, 10),
                },
            },
            "q8_schedule": timings.q8_schedule,
        }
    })
}

#[derive(Clone, Copy, Default)]
struct StreamEventTimings {
    poll_yield_enabled: bool,
    role_yield: Option<u128>,
    generate_start: Option<u128>,
    first_content_yield: Option<u128>,
    final_yield: Option<u128>,
}

fn stream_event_accounting_json(events: StreamEventTimings) -> serde_json::Value {
    let delta = |later: Option<u128>, earlier: Option<u128>| {
        later
            .zip(earlier)
            .map(|(later, earlier)| later as i128 - earlier as i128)
    };
    serde_json::json!({
        "poll_yield_enabled": events.poll_yield_enabled,
        "role_yield": events.role_yield,
        "generate_start": events.generate_start,
        "first_content_yield": events.first_content_yield,
        "final_yield": events.final_yield,
        "generate_start_minus_role_yield": delta(events.generate_start, events.role_yield),
        "first_content_yield_minus_role_yield": delta(events.first_content_yield, events.role_yield),
        "final_yield_minus_first_content_yield": delta(events.final_yield, events.first_content_yield),
    })
}

fn stream_first_content_accounting_json(
    timings: &GenerationTimings,
    first_content_ms: Option<u128>,
) -> serde_json::Value {
    let prompt_eval_forward = timings.prompt_evaluation.prefill.forward_total
        + timings.prompt_evaluation.first_token.forward_total;
    let prompt_eval_logits =
        timings.prompt_evaluation.prefill.logits + timings.prompt_evaluation.first_token.logits;
    let prompt_eval_sample =
        timings.prompt_evaluation.prefill.sample + timings.prompt_evaluation.first_token.sample;
    let prompt_eval_forward_plus_sample = prompt_eval_forward + prompt_eval_sample;
    let first_content = first_content_ms.map(|value| value as f64);

    serde_json::json!({
        "prompt_eval_forward_total": prompt_eval_forward,
        "prompt_eval_logits": prompt_eval_logits,
        "prompt_eval_sample": prompt_eval_sample,
        "prompt_eval_forward_plus_sample": prompt_eval_forward_plus_sample,
        "first_content_minus_prompt_eval_forward": first_content.map(|value| value - prompt_eval_forward),
        "first_content_minus_prompt_eval_forward_plus_sample": first_content.map(|value| value - prompt_eval_forward_plus_sample),
    })
}

/// Events emitted by the engine-side streaming decode job and mapped 1:1 onto
/// SSE chunks by `stream_completion`. Each variant carries exactly what the
/// SSE layer needs to reproduce the pre-inversion byte stream (chunk shapes
/// unchanged; only timing VALUES may differ).
enum StreamDecodeEvent {
    /// A non-empty text delta (already stop-sequence-truncated and diffed
    /// against previously streamed text).
    Delta(String),
    /// Clean end of generation; terminal.
    Finished {
        finish_reason: &'static str,
        completion_tokens: usize,
        /// Engine-side timing block for the optional timing-diagnostics JSON
        /// (boxed: it dwarfs the per-token Delta variant).
        timings: Box<GenerationTimings>,
        first_content_ms: Option<u128>,
    },
    /// Decode failed; terminal. Maps to `stream_error_message_event`.
    Failed { code: String, message: String },
    /// Wall-clock budget exhausted; terminal. Maps to
    /// `generation_timeout_stream_event` with the exact pre-inversion payload.
    TimedOut {
        timeout: Duration,
        elapsed: Duration,
        generated_tokens: usize,
    },
}

/// The error-code/message pair `stream_error_event` would have produced for a
/// Response-shaped failure, so the engine job can hand the SSE layer
/// byte-identical error frames without shipping the `Response` across.
fn stream_error_parts(response: &Response) -> (String, String) {
    (
        "stream_error".to_string(),
        format!("stream failed after headers: HTTP {}", response.status()),
    )
}

/// The streaming decode loop, run ON THE ENGINE THREAD as one exclusive job.
/// A dropped SSE stream (client disconnect) fires the request's cancel token
/// via `CancelOnDrop` and closes the events channel; the job observes either
/// between steps and stops within one step. The job is never detached from
/// the engine's serialization — the next request's job starts only after this
/// one returns.
fn run_stream_decode_job(
    mut prepared: PreparedGeneration,
    events: tokio::sync::mpsc::Sender<StreamDecodeEvent>,
) {
    #[cfg(test)]
    let _decode_probe = decode_probe::enter();
    // Terminal-event send helper: a closed channel means the client is gone —
    // nothing to report, the job just stops.
    let send = |event: StreamDecodeEvent| events.blocking_send(event).is_ok();
    // Lifecycle telemetry: if this job stops without a clean finish (client
    // disconnect, error), dropping the guard closes the run, so the
    // observatory never shows a stale "running" state.
    let telemetry_guard = prepared
        .telemetry
        .take()
        .map(telemetry::RequestGuard::begin);
    let generation_started = Instant::now();
    let request_timeout = match generation_timeout_duration() {
        Ok(timeout) => timeout,
        Err(response) => {
            let (code, message) = stream_error_parts(&response);
            send(StreamDecodeEvent::Failed { code, message });
            return;
        }
    };
    let stream_timing_diagnostics = stream_timing_diagnostics_enabled();
    let collect_q8_schedule = stream_timing_diagnostics && q8_schedule_telemetry_enabled();
    if collect_q8_schedule {
        reset_q8_schedule_telemetry();
    }
    if let Some(duration) = generation_step_test_sleep_duration() {
        std::thread::sleep(duration);
    }
    let mut input = prepared.token_ids.clone();
    let mut history = prepared.token_ids.clone();
    let mut generated = Vec::new();
    let mut top_logits = Vec::new();
    let mut output_projection = Vec::new();
    let mut dense = None;
    let mut finish_reason = "length";
    let mut streamed_text = String::new();
    let mut first_content_ms = None;
    let mut forward_timings = LlamaForwardTimings::default();
    let mut sample = 0;

    // Bypass the prompt-prefix cache when the CUDA-resident engine drives
    // decode: reusing a cached session reseeds the GPU KV from f16-rounded
    // host history (a different reduction order than a clean GPU prefill),
    // which corrupts the resumed decode â€” mild for greedy (a few near-tie
    // flips) but catastrophic under temperature sampling, where it produces
    // garbled, off-topic output. The non-streaming path already gates the
    // cache this way (see resident_decode_cuda_active); the streaming path
    // must too. The CPU lane is reduction-order-stable and keeps the cache.
    if !prepared.collect_dense_diagnostics && !crate::inference::resident_decode_cuda_active() {
        if let Some(match_res) = lookup_prompt_prefix_cache(&prepared) {
            let mut cached_session = match_res.cached.session.clone();
            cached_session
                .set_resident_paths_disabled(prepared.speculative.is_some() && !spec_gpu_enabled());
            if match_res.is_exact_match {
                prepared.session = cached_session;
                input.clear();
                match sample_cached_prompt_prefix(&match_res.cached, &history) {
                    Ok(first_step) => {
                        let cached_next_token = first_step.next_token_id;
                        prepared.timings.prompt_cache_hit = true;
                        sample += first_step.sample;
                        if let Err(response) = consume_generation_step(
                            &prepared,
                            first_step,
                            GenerationStepAccumulator {
                                generated: &mut generated,
                                history: &mut history,
                                top_logits: &mut top_logits,
                                output_projection: &mut output_projection,
                                dense: &mut dense,
                                finish_reason: &mut finish_reason,
                            },
                        ) {
                            let (code, message) = stream_error_parts(&response);
                            send(StreamDecodeEvent::Failed { code, message });
                            return;
                        }
                        if finish_reason == "length" {
                            input.push(cached_next_token);
                        }
                    }
                    Err(response) => {
                        let (code, message) = stream_error_parts(&response);
                        send(StreamDecodeEvent::Failed { code, message });
                        return;
                    }
                }
            } else {
                let k = match_res.prefix_len;
                if cached_session.rollback_to_position(k).is_ok() {
                    prepared.session = cached_session;
                    input = prepared.token_ids[k..].to_vec();
                    prepared.timings.prompt_cache_hit = true;
                }
            }
        }
    }

    for _ in generated.len() as u32..prepared.max_tokens {
        if finish_reason != "length" {
            break;
        }
        // Cooperative stop: the SSE layer's CancelOnDrop fired (client
        // disconnected). Nobody is listening — drop the telemetry guard
        // (closing the run) and stop within this step boundary.
        if prepared.cancel.token.is_cancelled() {
            return;
        }
        let generated_index = generated.len();
        let collect_dense_for_step =
            collect_dense_diagnostics_for_generated_index(&prepared, generated_index);
        let mut sampling = prepared.sampling.clone();
        if let Some(seed) = sampling.seed {
            sampling.seed = Some(seed.wrapping_add(generated.len() as u64));
        }
        let sampler = if sampling == SamplingConfig::default() {
            LlamaSampler::Greedy
        } else {
            LlamaSampler::Sampling(sampling)
        };
        if request_timeout
            .checked_sub(generation_started.elapsed())
            .is_none()
        {
            send(StreamDecodeEvent::TimedOut {
                timeout: request_timeout,
                elapsed: generation_started.elapsed(),
                generated_tokens: generated.len(),
            });
            return;
        }
        // Greedy single-token continuations with no per-step logit consumers
        // ride the resident GPU-sampling fast lane inside the step.
        let greedy_fast = input.len() == 1
            && matches!(sampler, LlamaSampler::Greedy)
            && !collect_dense_for_step
            && !top_logits.is_empty();
        let step = match run_stream_step(
            &mut prepared.session,
            StreamStepRequest {
                greedy_fast,
                input: input.clone(),
                sampler,
                history: history.clone(),
                collect_dense_diagnostics: collect_dense_for_step,
            },
        ) {
            Ok(step) => step,
            Err(response) => {
                let (code, message) = stream_error_parts(&response);
                send(StreamDecodeEvent::Failed { code, message });
                return;
            }
        };
        if generated.is_empty() && !prepared.collect_dense_diagnostics && step.diagnostics.is_none()
        {
            store_prompt_prefix_cache(&prepared, &step);
        }
        if generated.is_empty() {
            prepared.timings.prompt_evaluation = prompt_evaluation_timings_from_step(&step);
        }
        forward_timings.add_assign(&step.timings);
        sample += step.sample;
        if let Err(response) = consume_generation_step(
            &prepared,
            step,
            GenerationStepAccumulator {
                generated: &mut generated,
                history: &mut history,
                top_logits: &mut top_logits,
                output_projection: &mut output_projection,
                dense: &mut dense,
                finish_reason: &mut finish_reason,
            },
        ) {
            let (code, message) = stream_error_parts(&response);
            send(StreamDecodeEvent::Failed { code, message });
            return;
        }

        let mut text = match prepared.tokenizer.decode(&generated, true) {
            Ok(text) => text,
            Err(err) => {
                send(StreamDecodeEvent::Failed {
                    code: "token_decode_failed".to_string(),
                    message: err.to_string(),
                });
                return;
            }
        };
        if finish_reason == "stop" {
            text = truncate_at_stop_sequence(text, &prepared.stop_sequences);
        }
        let delta = text
            .strip_prefix(&streamed_text)
            .map(str::to_owned)
            .unwrap_or_else(|| text.clone());
        streamed_text = text;
        if !delta.is_empty() {
            if first_content_ms.is_none() {
                first_content_ms = Some(generation_started.elapsed().as_millis());
            }
            // A closed channel is a hangup: stop decoding.
            if !send(StreamDecodeEvent::Delta(delta)) {
                return;
            }
        }
        if finish_reason != "length" {
            break;
        }
        input.clear();
        if let Some(last_token) = generated.last().copied() {
            input.push(last_token);
        }
    }

    prepared.timings.generate = generation_started.elapsed().as_millis();
    prepared.timings.generation = generation_phase_timings_from_forward(&forward_timings, sample);
    prepared.timings.layers = generation_layer_timings_from_forward(&forward_timings.layers);
    prepared.timings.memory = forward_timings.memory;
    if collect_q8_schedule {
        prepared.timings.q8_schedule = Some(snapshot_q8_schedule_telemetry());
    }
    if let Some(guard) = telemetry_guard {
        let ttft_ms = first_content_ms.map(|ms| ms as u64);
        let decode_tps = match (first_content_ms, generated.len()) {
            (Some(first_ms), count) if count > 1 => {
                let decode_ms = generation_started
                    .elapsed()
                    .as_millis()
                    .saturating_sub(first_ms);
                (decode_ms > 0).then(|| (count - 1) as f64 * 1000.0 / decode_ms as f64)
            }
            _ => None,
        };
        guard.finish(telemetry::RequestFinish {
            status: "ok",
            finish_reason: Some(finish_reason.to_string()),
            completion_tokens: generated.len(),
            ttft_ms,
            decode_tps,
            prefill_tps: None,
            error: None,
        });
    }
    crate::gait::sentinel::mark_healthy();
    send(StreamDecodeEvent::Finished {
        finish_reason,
        completion_tokens: generated.len(),
        timings: Box::new(prepared.timings),
        first_content_ms,
    });
}

fn stream_completion(
    state: &AppState,
    mut prepared: PreparedGeneration,
    chat: bool,
    include_usage: bool,
) -> Response {
    // Speculation only runs in the non-streaming loop; streaming requests on
    // a spec-enabled server keep the unchanged vanilla path (including the
    // GPU-resident lanes the speculative pin would otherwise turn off).
    prepared.speculative = None;
    prepared.session.set_resident_paths_disabled(false);
    let model_id = prepared.model_id.clone();
    // Captured before the job so the streaming usage frame reports the exact
    // same prompt count as the non-streaming path (single source of truth).
    let prompt_token_count = prepared.token_ids.len();
    let stream_timing_diagnostics = stream_timing_diagnostics_enabled();
    let stream_poll_yield = stream_poll_yield_enabled();
    let stream_id = if chat {
        format!("chatcmpl-{}", uuid::Uuid::new_v4())
    } else {
        format!("cmpl-{}", uuid::Uuid::new_v4())
    };
    let cancel_token = prepared.cancel.token.clone();
    // Small buffer: the engine may run ahead of a slow client by at most this
    // many deltas, then parks on the channel send — mirroring pre-inversion
    // behavior, where an unpolled generator simply stopped decoding.
    let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(32);
    // Post the decode job before returning the SSE response. A full queue is
    // the same typed backpressure non-streaming requests get.
    if let Err(err) = state
        .engine
        .post(engine::EngineTask::Exclusive(Box::new(move || {
            run_stream_decode_job(prepared, events_tx);
        })))
    {
        return *engine_post_error_response(err);
    }
    let events = async_stream::stream! {
        // Dropping this generator (client disconnect) fires the request's
        // CancellationToken and closes the events channel; the engine job
        // observes either within one step and stops cleanly. Compute is never
        // detached from the engine's serialization.
        let _cancel_on_drop = CancelOnDrop(cancel_token);
        let stream_started = Instant::now();
        let mut stream_event_timings = StreamEventTimings {
            poll_yield_enabled: stream_poll_yield,
            ..StreamEventTimings::default()
        };
        if chat {
            stream_event_timings.role_yield = Some(stream_started.elapsed().as_millis());
            let role_chunk = ChatCompletionStreamChunk {
                id: stream_id.clone(),
                object: "chat.completion.chunk",
                created: 0,
                model: model_id.clone(),
                choices: vec![ChatCompletionStreamChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
                camelid: None,
                usage: None,
            };
            yield sse_json_event(&role_chunk);
            if stream_poll_yield {
                tokio::task::yield_now().await;
            }
        }
        stream_event_timings.generate_start = Some(stream_started.elapsed().as_millis());

        while let Some(event) = events_rx.recv().await {
            match event {
                StreamDecodeEvent::Delta(delta) => {
                    if stream_event_timings.first_content_yield.is_none() {
                        stream_event_timings.first_content_yield =
                            Some(stream_started.elapsed().as_millis());
                    }
                    if chat {
                        let chunk = ChatCompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "chat.completion.chunk",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![ChatCompletionStreamChoice {
                                index: 0,
                                delta: ChatCompletionDelta {
                                    role: None,
                                    content: Some(delta),
                                },
                                finish_reason: None,
                            }],
                            camelid: None,
                            usage: None,
                        };
                        yield sse_json_event(&chunk);
                    } else {
                        let chunk = CompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "text_completion",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![CompletionStreamChoice {
                                index: 0,
                                text: delta,
                                finish_reason: None,
                            }],
                            camelid: None,
                        };
                        yield sse_json_event(&chunk);
                    }
                    if stream_poll_yield {
                        tokio::task::yield_now().await;
                    }
                }
                StreamDecodeEvent::Failed { code, message } => {
                    yield stream_error_message_event(&code, message);
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                StreamDecodeEvent::TimedOut { timeout, elapsed, generated_tokens } => {
                    yield generation_timeout_stream_event(timeout, elapsed, generated_tokens);
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                StreamDecodeEvent::Finished {
                    finish_reason,
                    completion_tokens,
                    timings,
                    first_content_ms,
                } => {
                    stream_event_timings.final_yield = Some(stream_started.elapsed().as_millis());
                    let camelid_diagnostics = stream_timing_diagnostics.then(|| {
                        stream_timing_diagnostics_json(&timings, first_content_ms, stream_event_timings)
                    });
                    if chat {
                        let final_chunk = ChatCompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "chat.completion.chunk",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![ChatCompletionStreamChoice {
                                index: 0,
                                delta: ChatCompletionDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(finish_reason),
                            }],
                            camelid: camelid_diagnostics.clone(),
                            usage: None,
                        };
                        yield sse_json_event(&final_chunk);
                        // OpenAI stream_options.include_usage: exactly one terminal chunk
                        // with an empty `choices` array, carrying the same usage integers the
                        // non-streaming endpoint returns for this prompt+output (prompt =
                        // prepared.token_ids.len(); completion = sampled-token count). Emitted
                        // after the finish_reason chunk and before [DONE], matching the
                        // llama-server oracle ordering. Omitted entirely when include_usage is
                        // false, so the usage-off stream is byte-identical to the baseline.
                        if include_usage {
                            let usage_chunk = ChatCompletionStreamChunk {
                                id: stream_id.clone(),
                                object: "chat.completion.chunk",
                                created: 0,
                                model: model_id.clone(),
                                choices: Vec::new(),
                                camelid: None,
                                usage: Some(CompletionUsage {
                                    prompt_tokens: prompt_token_count,
                                    completion_tokens,
                                    total_tokens: prompt_token_count + completion_tokens,
                                }),
                            };
                            yield sse_json_event(&usage_chunk);
                        }
                    } else {
                        let final_chunk = CompletionStreamChunk {
                            id: stream_id.clone(),
                            object: "text_completion",
                            created: 0,
                            model: model_id.clone(),
                            choices: vec![CompletionStreamChoice {
                                index: 0,
                                text: String::new(),
                                finish_reason: Some(finish_reason),
                            }],
                            camelid: camelid_diagnostics,
                        };
                        yield sse_json_event(&final_chunk);
                    }
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
        }
        // The engine job ended without a terminal event: it was contained
        // after a panic, or the engine is shutting down. Typed error frame so
        // the client never sees a silently truncated stream.
        yield stream_error_message_event(
            "generation_worker_failed",
            "generation worker failed before completing the stream".to_string(),
        );
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(events).into_response()
}

fn generation_timeout_stream_event(
    timeout: Duration,
    elapsed: Duration,
    generated_tokens: usize,
) -> Result<Event, Infallible> {
    Ok(Event::default()
        .event("error")
        .data(generation_timeout_error_json(timeout, elapsed, Some(generated_tokens)).to_string()))
}

fn stream_error_message_event(code: &str, message: String) -> Result<Event, Infallible> {
    Ok(Event::default().event("error").data(
        serde_json::json!({
            "error": {
                "code": code,
                "message": message,
            }
        })
        .to_string(),
    ))
}

fn sse_json_event<T: Serialize>(value: &T) -> Result<Event, Infallible> {
    Ok(Event::default().data(
        serde_json::to_string(value)
            .expect("OpenAI-compatible SSE chunk serialization cannot fail"),
    ))
}

fn contains_stop_sequence(text: &str, stop_sequences: &[String]) -> bool {
    stop_sequences
        .iter()
        .any(|sequence| text.contains(sequence))
}

fn truncate_at_stop_sequence(mut text: String, stop_sequences: &[String]) -> String {
    if let Some(stop_index) = stop_sequences
        .iter()
        .filter_map(|sequence| text.find(sequence))
        .min()
    {
        text.truncate(stop_index);
    }
    text
}

fn validate_chat_messages(messages: &[ChatMessage]) -> std::result::Result<(), Box<Response>> {
    for (idx, message) in messages.iter().enumerate() {
        if message.role.trim().is_empty() {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_message_role",
                format!("message {idx} role must not be empty"),
                Some("messages"),
            )));
        }
        if message.content.is_empty() {
            return Err(Box::new(api_error(
                StatusCode::BAD_REQUEST,
                "invalid_message_content",
                format!("message {idx} content must not be empty"),
                Some("messages"),
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderedPrompt {
    text: String,
    add_special: bool,
    parse_special: bool,
}

#[derive(Debug, Serialize)]
struct ChatTemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

fn normalize_mistral_instruct_bos_prefix_tokens(
    token_ids: &mut Vec<u32>,
    rendered_prompt: &RenderedPrompt,
    tokenizer: &Tokenizer,
) {
    let Some(bos_id) = tokenizer.special.bos else {
        return;
    };
    let Some(bos_text) = tokenizer.token_text(Some(bos_id)) else {
        return;
    };
    let Some(space_id) = tokenizer.token_id("â–") else {
        return;
    };
    if rendered_prompt.parse_special
        && rendered_prompt
            .text
            .starts_with(&format!("{bos_text}[INST]"))
        && token_ids.first() == Some(&bos_id)
        && token_ids.get(1) == Some(&space_id)
    {
        token_ids.remove(1);
    }
}

#[cfg(test)]
fn render_chat_prompt_for_tokenization(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
) -> RenderedPrompt {
    render_chat_prompt_for_tokenization_for_model(messages, tokenizer, None)
}

fn render_chat_prompt_for_tokenization_for_model_result(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    model_id: Option<&str>,
    enable_thinking: bool,
) -> std::result::Result<RenderedPrompt, MiniJinjaError> {
    let exact_llama32_metadata_jinja_row =
        model_id.and_then(llama32_metadata_jinja_exact_row_label);
    if let Some(template) = tokenizer.chat_template.as_deref() {
        if metadata_chat_template_enabled() {
            return render_metadata_jinja_chat_template_prompt(messages, tokenizer, template, None);
        }
        if let Some(row_label) = exact_llama32_metadata_jinja_row {
            if is_llama3_instruct_template(template) {
                return render_metadata_jinja_chat_template_prompt(
                    messages, tokenizer, template, None,
                );
            }
            return Err(exact_llama32_metadata_jinja_chat_template_error(
                &format!(
                    "{row_label} requires a recognized Llama 3 metadata chat_template containing <|start_header_id|>, <|end_header_id|>, and <|eot_id|>"
                ),
            ));
        }
    } else if let Some(row_label) = exact_llama32_metadata_jinja_row {
        return Err(exact_llama32_metadata_jinja_chat_template_error(&format!(
            "{row_label} requires tokenizer.chat_template metadata for chat prompt rendering"
        )));
    }

    Ok(render_chat_prompt_for_tokenization_fallback(
        messages,
        tokenizer,
        enable_thinking,
    ))
}

/// Agent mode: render the chat prompt with tool definitions threaded into the
/// model's own chat template. Used only when a request carries `tools`; when the
/// model has no chat template, tools cannot be rendered and we fall back.
fn render_chat_prompt_for_tokenization_with_tools(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    tools: &[serde_json::Value],
) -> std::result::Result<RenderedPrompt, MiniJinjaError> {
    // Normalize OpenAI-style `{ "type":"function", "function":{â€¦} }` tools to the
    // flat function objects (`{ "name", "description", "parameters" }`) the chat
    // templates (Llama 3.x etc.) actually render â€” matching llama.cpp / vLLM. This
    // keeps the wire format OpenAI-standard while the prompt the model sees aligns
    // with the `{"name":â€¦, "parameters":â€¦}` response format the template requests
    // (the nested envelope otherwise leaks into the prompt and small models echo
    // the schema). See `TOOLCALL_DIAG.md`.
    let normalized: Vec<serde_json::Value> = tools
        .iter()
        .map(|tool| {
            tool.get("function")
                .cloned()
                .unwrap_or_else(|| tool.clone())
        })
        .collect();
    if let Some(template) = tokenizer.chat_template.as_deref() {
        if is_phi4_compact_template(template) {
            return Err(MiniJinjaError::new(
                MiniJinjaErrorKind::InvalidOperation,
                "Phi-4 tool-call template rendering is not validated; failing closed",
            ));
        }
        // Mistral Instruct templates (v0.3 GGUF) don't reference the `tools`
        // Jinja variable â€” the generic Jinja render silently drops them. Use
        // the dedicated renderer that produces [AVAILABLE_TOOLS] natively.
        if is_mistral_instruct_template(template) {
            return Ok(RenderedPrompt {
                text: render_mistral_instruct_prompt_with_tools(messages, tokenizer, tools),
                add_special: false,
                parse_special: true,
            });
        }
        if is_qwen2_chatml_template(template) {
            return Err(MiniJinjaError::new(
                MiniJinjaErrorKind::InvalidOperation,
                "Qwen2/Qwen2.5 tool-call template rendering is not validated; failing closed",
            ));
        }
        // Qwen3's full Jinja template uses constructs minijinja can't evaluate;
        // render tools via the dedicated ChatML+tools path instead.
        if is_qwen3_chatml_template(template) {
            return Ok(RenderedPrompt {
                text: render_qwen3_chatml_prompt_with_tools(messages, &normalized),
                add_special: false,
                parse_special: true,
            });
        }
        return render_metadata_jinja_chat_template_prompt(
            messages,
            tokenizer,
            template,
            Some(&normalized),
        );
    }
    // Agent/tools rendering is the deterministic thinking-disabled path.
    Ok(render_chat_prompt_for_tokenization_fallback(
        messages, tokenizer, false,
    ))
}

#[cfg(test)]
fn render_chat_prompt_for_tokenization_for_model(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    model_id: Option<&str>,
) -> RenderedPrompt {
    render_chat_prompt_for_tokenization_for_model_result(messages, tokenizer, model_id, false)
        .unwrap_or_else(|_| {
            render_chat_prompt_for_tokenization_fallback(messages, tokenizer, false)
        })
}

fn render_chat_prompt_for_tokenization_fallback(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    enable_thinking: bool,
) -> RenderedPrompt {
    if let Some(template) = tokenizer.chat_template.as_deref() {
        // The marker strings themselves (<|user|>, <|assistant|>, <|system|>)
        // are not vocab entries and stay plain SPM text either way; the
        // template's `</s>` IS a control token, and llama-server encodes it
        // as EOS when tokenizing the rendered template â€” so chat prompts
        // parse specials (chat_prompt_parse_special), with dummy-prefix
        // handling after control tokens preserved by encode_piece.
        // Checked BEFORE tinyllama: Phi-3 reuses the same <|user|>/<|assistant|>
        // marker spellings but separates turns with <|end|> (not </s>) and stops on
        // <|end|>. Routing it through the tinyllama renderer used the wrong separator
        // and stop token, so generation rambled.
        if is_phi4_compact_template(template) {
            return RenderedPrompt {
                text: render_phi4_compact_prompt(messages),
                add_special: true,
                parse_special: true,
            };
        }
        if is_phi3_template(template) {
            return RenderedPrompt {
                text: render_phi3_prompt(messages),
                // Phi-3 sets add_bos_token=true; the tokenizer prepends <s>.
                add_special: true,
                // Parse specials so <|user|>/<|assistant|>/<|end|> become the control
                // token ids â€” in particular <|end|> as the end-of-turn stop token.
                parse_special: true,
            };
        }
        if is_tinyllama_marker_template(template) {
            return RenderedPrompt {
                text: render_tinyllama_marker_prompt(messages, tokenizer),
                add_special: true,
                parse_special: tokenizer.chat_prompt_parse_special(),
            };
        }
        if is_llama3_instruct_template(template) {
            return RenderedPrompt {
                text: render_llama3_instruct_prompt(messages),
                add_special: true,
                parse_special: tokenizer.chat_prompt_parse_special(),
            };
        }
        if is_mistral_instruct_template(template) {
            return RenderedPrompt {
                text: render_mistral_instruct_prompt(messages, tokenizer),
                // The Mistral instruct renderer emits the BOS token text from
                // the metadata template shape (`{{ bos_token }}[INST] ...`).
                // Do not also ask the tokenizer to prepend BOS, or the first
                // exact-row parity check gets a duplicated BOS. Keep special
                // parsing on so SPM normalization does not add a dummy prefix
                // before the rendered `<s>` control token.
                add_special: false,
                parse_special: true,
            };
        }
        if is_qwen2_chatml_template(template) {
            return RenderedPrompt {
                text: render_qwen2_chatml_prompt(messages),
                add_special: false,
                parse_special: true,
            };
        }
        if is_qwen3_chatml_template(template) {
            return RenderedPrompt {
                text: render_qwen3_chatml_prompt(messages, enable_thinking),
                // Qwen3 has add_bos_token=false; the ChatML template fully
                // specifies the prompt. Parse specials so <|im_start|>/<|im_end|>
                // become control token ids (151644/151645), not literal text.
                add_special: false,
                parse_special: true,
            };
        }
    }

    RenderedPrompt {
        text: render_role_colon_prompt(messages),
        add_special: true,
        parse_special: tokenizer.chat_prompt_parse_special(),
    }
}

/// Render the no-tools Qwen2/Qwen2.5 ChatML template shipped in the model GGUF.
/// Unlike Qwen3, this template injects Qwen's default system preamble when the
/// caller did not supply one and opens a plain assistant turn (no thinking tags).
fn render_qwen2_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    let first_is_system = messages
        .first()
        .is_some_and(|message| message.role.trim() == "system");

    prompt.push_str("<|im_start|>system\n");
    if first_is_system {
        prompt.push_str(&messages[0].content);
    } else {
        prompt.push_str("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");
    }
    prompt.push_str("<|im_end|>\n");

    let mut append_generation_prompt = true;
    for (index, message) in messages.iter().enumerate() {
        let role = message.role.trim();
        if index == 0 && first_is_system {
            continue;
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        append_generation_prompt = role != "assistant";
    }
    if append_generation_prompt {
        prompt.push_str("<|im_start|>assistant\n");
    }
    prompt
}

/// Render a Qwen3 ChatML prompt. Mirrors the GGUF jinja template for the standard
/// cases (optional leading system turn, then user/assistant turns), then appends
/// the generation prompt for the requested thinking mode:
/// - `enable_thinking = false` (the deterministic, parity-locked default):
///   `<|im_start|>assistant\n<think>\n\n</think>\n\n` (the template's
///   `enable_thinking is false` branch â€” a direct answer, no reasoning).
/// - `enable_thinking = true`: `<|im_start|>assistant\n` (the template's default
///   branch â€” the model emits its own `<think>â€¦</think>` reasoning block first).
///
/// Verified token-identical to the reference for single-turn user prompts.
fn render_qwen3_chatml_prompt(messages: &[ChatMessage], enable_thinking: bool) -> String {
    let mut prompt = String::new();
    let mut append_generation_prompt = true;
    for message in messages {
        let role = message.role.trim();
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        // If the caller already supplied a trailing assistant turn, don't append
        // another generation prompt.
        append_generation_prompt = role != "assistant";
    }
    if append_generation_prompt {
        if enable_thinking {
            // Thinking enabled: the bare assistant turn matches the GGUF template's
            // default branch; the model generates its own <think>â€¦</think> block.
            prompt.push_str("<|im_start|>assistant\n");
        } else {
            // Thinking disabled: the empty <think></think> block matches the GGUF
            // template's `enable_thinking is false` branch, giving a direct,
            // deterministic answer for parity.
            prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
        }
    }
    prompt
}

/// Renders a Qwen3 ChatML prompt with tool definitions injected into the system
/// message and thinking suppressed. The format matches Qwen3's expected tool-call
/// protocol: tools as flat JSON objects in the system turn, model responds with
/// `<tool_call>{"name":â€¦,"arguments":{â€¦}}</tool_call>`.
fn render_qwen3_chatml_prompt_with_tools(
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> String {
    let mut prompt = String::new();

    // Build tool definitions block for the system message.
    let mut tool_block =
        String::from("You are a helpful assistant with access to the following tools:\n\n");
    for tool in tools {
        if let Ok(json) = serde_json::to_string(tool) {
            tool_block.push_str(&json);
            tool_block.push('\n');
        }
    }
    tool_block.push_str(
        "\nWhen you need to call a tool, put your tool call in the format:\n\
         <tool_call>\n\
         {\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n\
         </tool_call>",
    );

    // Emit system turn: merge any existing system content with tool definitions.
    prompt.push_str("<|im_start|>system\n");
    let mut has_system = false;
    for message in messages {
        if message.role.trim() == "system" {
            if !message.content.is_empty() {
                prompt.push_str(&message.content);
                prompt.push_str("\n\n");
            }
            has_system = true;
        }
    }
    if !has_system {
        // No system message from caller â€” tool block IS the system message.
    }
    prompt.push_str(&tool_block);
    prompt.push_str("<|im_end|>\n");

    // Emit non-system messages.
    let mut append_generation_prompt = true;
    for message in messages {
        let role = message.role.trim();
        if role == "system" {
            continue;
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        append_generation_prompt = role != "assistant";
    }

    if append_generation_prompt {
        // Suppress thinking for tool-calling: pre-fill empty <think></think> so
        // the model proceeds directly to <tool_call> output without burning tokens
        // on chain-of-thought reasoning.
        prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    prompt
}

#[cfg(test)]
fn render_chat_prompt(messages: &[ChatMessage], tokenizer: &Tokenizer) -> String {
    render_chat_prompt_for_tokenization(messages, tokenizer).text
}

fn is_tinyllama_marker_template(template: &str) -> bool {
    template.contains("<|user|>")
        && template.contains("<|assistant|>")
        && template.contains("<|system|>")
}

fn is_llama3_instruct_template(template: &str) -> bool {
    template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
}

fn is_mistral_instruct_template(template: &str) -> bool {
    template.contains("[INST]")
        && template.contains("[/INST]")
        && (template.contains("bos_token") || template.contains("</s>"))
}

fn is_qwen2_chatml_template(template: &str) -> bool {
    template.contains("<|im_start|>")
        && template.contains("<|im_end|>")
        && template.contains("You are Qwen, created by Alibaba Cloud")
}

/// Generic Qwen3 ChatML template detector: `<|im_start|>` / `<|im_end|>`.
/// Qwen2/Qwen2.5's distinct default-system template is detected first by
/// [`is_qwen2_chatml_template`].
/// turn markers. camelid's minijinja cannot render the full Qwen3 jinja template
/// (it uses constructs that evaluate to undefined), so the dense ChatML path is
/// rendered by [`render_qwen3_chatml_prompt`] instead.
fn is_qwen3_chatml_template(template: &str) -> bool {
    template.contains("<|im_start|>") && template.contains("<|im_end|>")
}

/// Phi-4's compact marker template keeps every role marker adjacent to content:
/// `<|role|>content<|end|>`, followed by a bare `<|assistant|>` generation prompt.
/// Check it before Phi-3 because both families use the same marker vocabulary.
fn is_phi4_compact_template(template: &str) -> bool {
    template.contains("message['role'] == 'system'")
        && template.contains("'tools' in message")
        && template.contains("message['content'] + '<|end|>'")
        && template.contains("<|tool|>")
        && template.contains("<|/tool|>")
        && template.contains("add_generation_prompt")
        && template.contains("'<|assistant|>'")
}

/// Phi-3 chat template detector: `<|user|>`/`<|assistant|>` turns separated by the
/// `<|end|>` turn marker. Phi-3 shares the `<|user|>`/`<|assistant|>`/`<|system|>`
/// spellings with TinyLlama's marker template (which separates turns with `</s>`),
/// so the `<|end|>` marker is the distinguishing signal and this MUST be checked
/// before [`is_tinyllama_marker_template`].
fn is_phi3_template(template: &str) -> bool {
    template.contains("<|assistant|>")
        && template.contains("<|end|>")
        && template.contains("<|user|>")
}

fn exact_llama32_metadata_jinja_chat_template_error(message: &str) -> MiniJinjaError {
    MiniJinjaError::new(MiniJinjaErrorKind::InvalidOperation, message.to_string())
}

fn llama32_metadata_jinja_exact_row_label(model_id: &str) -> Option<&'static str> {
    if is_llama32_1b_exact_row_model_id(model_id) {
        return Some("exact Llama 3.2 1B Instruct Q8_0");
    }
    if is_llama32_3b_exact_row_model_id(model_id) {
        return Some("exact Llama 3.2 3B Instruct Q8_0");
    }
    None
}

fn is_llama32_1b_exact_row_model_id(model_id: &str) -> bool {
    let normalized = model_id
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            '-' | '.' | ' ' => '_',
            ch => ch,
        })
        .collect::<String>();
    normalized.contains("llama32_1b_instruct_q8_0")
        || normalized.contains("llama_3_2_1b_instruct_q8_0")
}

fn is_llama32_3b_exact_row_model_id(model_id: &str) -> bool {
    let normalized = model_id
        .chars()
        .flat_map(char::to_lowercase)
        .map(|ch| match ch {
            '-' | '.' | ' ' => '_',
            ch => ch,
        })
        .collect::<String>();
    normalized.contains("llama32_3b_instruct_q8_0")
        || normalized.contains("llama_3_2_3b_instruct_q8_0")
}

fn metadata_chat_template_enabled() -> bool {
    matches!(
        env::var(METADATA_CHAT_TEMPLATE_ENV),
        Ok(value)
            if value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("metadata")
    )
}

fn render_metadata_jinja_chat_template_prompt(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    template: &str,
    tools: Option<&[serde_json::Value]>,
) -> std::result::Result<RenderedPrompt, MiniJinjaError> {
    let rendered = render_jinja_chat_template(messages, tokenizer, template, tools)?;
    Ok(RenderedPrompt {
        add_special: !rendered_prompt_starts_with_token_text(
            &rendered,
            tokenizer.special.bos,
            tokenizer,
        ),
        text: rendered,
        parse_special: tokenizer.chat_prompt_parse_special(),
    })
}

fn render_jinja_chat_template(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    template: &str,
    tools: Option<&[serde_json::Value]>,
) -> std::result::Result<String, MiniJinjaError> {
    let template_messages = messages
        .iter()
        .map(|message| ChatTemplateMessage {
            role: message.role.trim(),
            content: message.content.as_str(),
        })
        .collect::<Vec<_>>();
    let bos_token = tokenizer.token_text(tokenizer.special.bos).unwrap_or("");
    let eos_token = tokenizer.token_text(tokenizer.special.eos).unwrap_or("");
    let eot_token = tokenizer.token_text(tokenizer.special.eot).unwrap_or("");
    let eom_token = tokenizer.token_text(tokenizer.special.eom).unwrap_or("");
    let unk_token = tokenizer.token_text(tokenizer.special.unk).unwrap_or("");

    let env = cached_jinja_chat_template_environment(template)?;
    let compiled = env.get_template(JINJA_CHAT_TEMPLATE_NAME)?;
    compiled.render(context! {
        messages => template_messages,
        bos_token => bos_token,
        eos_token => eos_token,
        eot_token => eot_token,
        eom_token => eom_token,
        unk_token => unk_token,
        add_generation_prompt => true,
        // Agent mode: the model's own template renders these. When `tools` is
        // None both resolve to none, so the template takes its no-tools path and
        // the render is byte-identical to before.
        tools => tools,
        custom_tools => tools,
    })
}

fn cached_jinja_chat_template_environment(
    template: &str,
) -> std::result::Result<Arc<Environment<'static>>, MiniJinjaError> {
    let cache = JINJA_CHAT_TEMPLATE_ENV_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(env) = cache
        .lock()
        .expect("jinja chat-template cache poisoned")
        .get(template)
        .cloned()
    {
        return Ok(env);
    }

    let mut env = Environment::new();
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    env.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, MiniJinjaError> {
            Err(MiniJinjaError::new(
                MiniJinjaErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_template_owned(JINJA_CHAT_TEMPLATE_NAME.to_string(), template.to_string())?;
    let env = Arc::new(env);

    let mut cache = cache.lock().expect("jinja chat-template cache poisoned");
    if let Some(cached) = cache.get(template) {
        return Ok(Arc::clone(cached));
    }
    if cache.len() >= JINJA_CHAT_TEMPLATE_CACHE_LIMIT {
        cache.clear();
    }
    cache.insert(template.to_string(), Arc::clone(&env));
    Ok(env)
}

#[cfg(test)]
fn clear_jinja_chat_template_environment_cache() {
    if let Some(cache) = JINJA_CHAT_TEMPLATE_ENV_CACHE.get() {
        cache
            .lock()
            .expect("jinja chat-template cache poisoned")
            .clear();
    }
}

#[cfg(test)]
fn jinja_chat_template_environment_cache_len() -> usize {
    JINJA_CHAT_TEMPLATE_ENV_CACHE
        .get()
        .map(|cache| {
            cache
                .lock()
                .expect("jinja chat-template cache poisoned")
                .len()
        })
        .unwrap_or(0)
}

fn rendered_prompt_starts_with_token_text(
    rendered: &str,
    token_id: Option<u32>,
    tokenizer: &Tokenizer,
) -> bool {
    tokenizer
        .token_text(token_id)
        .is_some_and(|token_text| !token_text.is_empty() && rendered.starts_with(token_text))
}

fn render_tinyllama_marker_prompt(messages: &[ChatMessage], tokenizer: &Tokenizer) -> String {
    let eos = tokenizer
        .token_text(tokenizer.special.eos)
        .unwrap_or("</s>");
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>\n");
        prompt.push_str(&message.content);
        prompt.push_str(eos);
        prompt.push('\n');
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<|assistant|>\n");
    }
    prompt
}

fn render_llama3_instruct_prompt(messages: &[ChatMessage]) -> String {
    render_llama3_instruct_prompt_with_options(
        messages,
        false,
        messages
            .last()
            .is_none_or(|message| message.role.trim() != "assistant"),
    )
}

fn render_llama3_instruct_prompt_with_options(
    messages: &[ChatMessage],
    trim_content: bool,
    append_generation_prompt: bool,
) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(message.role.trim());
        prompt.push_str("<|end_header_id|>\n\n");
        if trim_content {
            prompt.push_str(message.content.trim());
        } else {
            prompt.push_str(&message.content);
        }
        prompt.push_str("<|eot_id|>");
    }
    if append_generation_prompt {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    prompt
}

fn render_mistral_instruct_prompt(messages: &[ChatMessage], tokenizer: &Tokenizer) -> String {
    let bos = tokenizer.token_text(tokenizer.special.bos).unwrap_or("<s>");
    let eos = tokenizer
        .token_text(tokenizer.special.eos)
        .unwrap_or("</s>");
    let mut prompt = String::new();
    let mut system = None;
    let mut idx = 0;

    if let Some(first) = messages.first() {
        if first.role.trim() == "system" {
            system = Some(first.content.trim());
            idx = 1;
        }
    }

    while idx < messages.len() {
        let message = &messages[idx];
        if message.role.trim() != "user" {
            idx += 1;
            continue;
        }

        prompt.push_str(bos);
        prompt.push_str("[INST] ");
        if let Some(system_content) = system.take() {
            prompt.push_str(system_content);
            prompt.push_str("\n\n");
        }
        prompt.push_str(message.content.trim());
        prompt.push_str(" [/INST]");

        if let Some(assistant) = messages.get(idx + 1) {
            if assistant.role.trim() == "assistant" {
                prompt.push(' ');
                prompt.push_str(assistant.content.trim());
                prompt.push_str(eos);
                idx += 2;
                continue;
            }
        }
        break;
    }

    prompt
}

/// Mistral Instruct v0.3+ with native tool calling: injects `[AVAILABLE_TOOLS]`
/// before the conversation and renders tool results as `[TOOL_RESULTS]` blocks.
/// The GGUF-embedded Jinja template for this model does not reference the `tools`
/// variable, so the generic Jinja path silently drops them. This renderer produces
/// the format documented in Mistral's tokenizer v2 spec.
fn render_mistral_instruct_prompt_with_tools(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    tools: &[serde_json::Value],
) -> String {
    let bos = tokenizer.token_text(tokenizer.special.bos).unwrap_or("<s>");
    let eos = tokenizer
        .token_text(tokenizer.special.eos)
        .unwrap_or("</s>");
    let mut prompt = String::new();

    // [AVAILABLE_TOOLS] block before the conversation.
    prompt.push_str(bos);
    prompt.push_str("[AVAILABLE_TOOLS] ");
    prompt.push_str(&serde_json::to_string(tools).unwrap_or_else(|_| "[]".into()));
    prompt.push_str("[/AVAILABLE_TOOLS]");

    let mut system: Option<&str> = None;
    let mut idx = 0;
    let mut tool_call_counter: u32 = 0;

    if let Some(first) = messages.first() {
        if first.role.trim() == "system" {
            system = Some(first.content.trim());
            idx = 1;
        }
    }

    while idx < messages.len() {
        let message = &messages[idx];
        let role = message.role.trim();
        match role {
            "user" => {
                prompt.push_str("[INST] ");
                // When [AVAILABLE_TOOLS] is active, discard the system message.
                // Mistral v0.3 was fine-tuned on:
                //   [AVAILABLE_TOOLS]...[/AVAILABLE_TOOLS][INST] query [/INST]
                // Adding system text inside [INST] pushes the model off-distribution
                // and causes it to generate prose instead of [TOOL_CALLS].
                system.take();
                prompt.push_str(message.content.trim());
                prompt.push_str(" [/INST]");
            }
            "assistant" => {
                // If the content looks like tool-call output from the agent loop
                // (formatted as "name(args)"), wrap it in [TOOL_CALLS] so the model
                // sees its own output format in multi-turn history. Otherwise emit
                // as plain assistant text.
                let content = message.content.trim();
                if looks_like_agent_tool_call(content) {
                    tool_call_counter += 1;
                    let id = format!("call{:05}", tool_call_counter);
                    let tc_json = agent_call_to_mistral_json(content, &id);
                    prompt.push_str("[TOOL_CALLS] ");
                    prompt.push_str(&tc_json);
                    prompt.push_str(eos);
                } else {
                    prompt.push_str(content);
                    prompt.push_str(eos);
                }
            }
            "tool" => {
                let id = format!("call{:05}", tool_call_counter);
                prompt.push_str("[TOOL_RESULTS] ");
                prompt.push_str(&format!(
                    "{{\"content\": {}, \"call_id\": \"{}\"}}",
                    serde_json::to_string(message.content.trim()).unwrap_or_else(|_| "\"\"".into()),
                    id
                ));
                prompt.push_str("[/TOOL_RESULTS]");
            }
            _ => {
                // Skip unknown roles (system already consumed above).
            }
        }
        idx += 1;
    }

    prompt
}

/// Returns true if the content looks like the agent loop's tool-call rendering
/// (e.g. `read_file({"path":"notes.txt"})`).
fn looks_like_agent_tool_call(content: &str) -> bool {
    let first_line = content.lines().next().unwrap_or("");
    if let Some(paren) = first_line.find('(') {
        let name = &first_line[..paren];
        let rest = &first_line[paren..];
        !name.is_empty()
            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
            && rest.ends_with(')')
            && rest.contains('{')
    } else {
        false
    }
}

/// Convert the agent loop's `name({"key":"val"})` format to Mistral's
/// `[{"name":"...", "arguments":{...}, "id":"..."}]` JSON array.
fn agent_call_to_mistral_json(content: &str, id: &str) -> String {
    let mut calls: Vec<serde_json::Value> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if let Some(paren) = line.find('(') {
            let name = &line[..paren];
            let args_str = &line[paren + 1..line.len().saturating_sub(1)];
            let args: serde_json::Value = serde_json::from_str(args_str)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            calls.push(serde_json::json!({
                "name": name,
                "arguments": args,
                "id": id,
            }));
        }
    }
    serde_json::to_string(&calls).unwrap_or_else(|_| "[]".into())
}

/// Render the Phi-3 chat template: each turn as `<|{role}|>\n{content}<|end|>\n`,
/// then the `<|assistant|>\n` generation prompt. Mirrors the GGUF jinja template;
/// `<|end|>` is the end-of-turn marker (and stop token under `parse_special`).
fn render_phi3_prompt(messages: &[ChatMessage]) -> String {
    // Byte-faithful to the Phi-3-mini GGUF template AS THE PINNED ORACLE RENDERS
    // IT (llama-server --jinja /apply-template, locked by
    // qa/prompt-packs/phi3-chat-template-shapes-v1.json — MUSTER M-A2):
    // user AND system messages each render as a `<|user|>` turn eagerly followed
    // by the `<|assistant|>\n` header (the template emits it per user/system
    // message, not as a separate generation prompt); assistant messages are
    // HEADERLESS (`{content}<|end|>\n` — the header came from the previous
    // turn's eager emission), and a TRAILING assistant message keeps no final
    // `<|end|>` (the oracle strips it: assistant-prefill continuation). Content
    // is verbatim — the template has no trim filters. No BOS string: the route
    // encodes with add_special=true, matching llama-server's chat path.
    let mut prompt = String::new();
    for (i, message) in messages.iter().enumerate() {
        if message.role.trim() == "assistant" {
            prompt.push_str(&message.content);
            if i + 1 < messages.len() {
                prompt.push_str("<|end|>\n");
            }
        } else {
            prompt.push_str("<|user|>\n");
            prompt.push_str(&message.content);
            prompt.push_str("<|end|>\n<|assistant|>\n");
        }
    }
    prompt
}

/// Render the ordinary-message branch of Phi-4's compact metadata template.
/// Tool-bearing requests fail closed until that template path is independently validated.
fn render_phi4_compact_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>");
        prompt.push_str(&message.content);
        prompt.push_str("<|end|>");
    }
    prompt.push_str("<|assistant|>");
    prompt
}

fn render_role_colon_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(message.role.trim());
        prompt.push_str(": ");
        prompt.push_str(&message.content);
        prompt.push('\n');
    }
    prompt
}

async fn loaded_tokenizer(state: &AppState) -> std::result::Result<Tokenizer, Response> {
    let active_id = state.active_model_id.read().await;
    let loaded_models = state.loaded_models.read().await;
    let model = active_id
        .as_ref()
        .and_then(|id| loaded_models.get(id))
        .ok_or_else(|| {
            api_error(
                StatusCode::NOT_FOUND,
                "model_not_loaded",
                BackendError::ModelNotLoaded.to_string(),
                None,
            )
        })?;
    Tokenizer::from_gguf(&model.gguf).map_err(|err| {
        api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            tokenizer_error_code(&err),
            err.to_string(),
            None,
        )
    })
}

fn malformed_json_error(err: JsonRejection) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        "malformed_json",
        err.to_string(),
        None,
    )
}

fn unsupported_route(
    code: &'static str,
    message: &'static str,
    param: Option<&'static str>,
) -> Response {
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        code,
        message.to_string(),
        param,
    )
}

fn api_error(
    status: StatusCode,
    code: &'static str,
    message: String,
    param: Option<&'static str>,
) -> Response {
    // Surface failures that happen while a generation is live on the
    // telemetry stream. Errors raised outside an active generation
    // (request validation, model management) are not inference errors and
    // stay off the stream.
    if telemetry::hub().request_active() {
        telemetry::emit(telemetry::Event::InferenceError {
            code: code.to_string(),
            message: message.clone(),
        });
    }
    let error_type = match status {
        StatusCode::NOT_IMPLEMENTED => "not_implemented",
        StatusCode::SERVICE_UNAVAILABLE => "runtime_unavailable",
        StatusCode::UNPROCESSABLE_ENTITY => "model_unavailable",
        _ => "invalid_request",
    };
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorBody {
                message,
                error_type,
                code,
                param,
            },
        }),
    )
        .into_response()
}

fn tokenizer_state_from_result(
    tokenizer: std::result::Result<&Tokenizer, &BackendError>,
) -> TokenizerLoadState {
    match tokenizer {
        Ok(tokenizer) => TokenizerLoadState::Available(tokenizer_summary(tokenizer)),
        Err(err) => TokenizerLoadState::Unavailable {
            code: tokenizer_error_code(err),
            message: err.to_string(),
        },
    }
}

fn tokenizer_summary(tokenizer: &Tokenizer) -> TokenizerSummary {
    TokenizerSummary {
        model: tokenizer.model.as_summary_model(),
        token_count: tokenizer.tokens.len(),
        byte_token_count: tokenizer.byte_token_to_id.len(),
        special: SpecialTokenSummary {
            bos: tokenizer.special.bos,
            eos: tokenizer.special.eos,
            eot: tokenizer.special.eot,
            eom: tokenizer.special.eom,
            unk: tokenizer.special.unk,
            sep: tokenizer.special.sep,
            pad: tokenizer.special.pad,
            mask: tokenizer.special.mask,
            eog: tokenizer.special.eog.iter().copied().collect(),
        },
        config: TokenizerConfigSummary {
            add_bos: tokenizer.config.add_bos,
            add_eos: tokenizer.config.add_eos,
            add_sep: tokenizer.config.add_sep,
            add_space_prefix: tokenizer.config.add_space_prefix,
            remove_extra_whitespaces: tokenizer.config.remove_extra_whitespaces,
        },
        chat_template: tokenizer
            .chat_template
            .as_deref()
            .map(|template| ChatTemplateSummary {
                source: "tokenizer.chat_template",
                detected_format: detect_chat_template_format(template),
                length: template.len(),
            }),
    }
}

fn detect_chat_template_format(template: &str) -> &'static str {
    if is_llama3_instruct_template(template) {
        "llama3_instruct"
    } else if is_mistral_instruct_template(template) {
        "mistral_instruct"
    } else if is_tinyllama_marker_template(template) {
        "tinyllama_marker"
    } else {
        "metadata_unparsed"
    }
}

fn tokenizer_error_code(err: &BackendError) -> &'static str {
    match err {
        BackendError::TokenizerNotAvailable => "tokenizer_not_available",
        BackendError::UnsupportedTokenizer(_) => "unsupported_tokenizer",
        BackendError::InvalidTokenizerMetadata(_) => "invalid_tokenizer_metadata",
        _ => "tokenizer_unavailable",
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet, HashMap},
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use crate::{
        execution_plan::{ExecutionPlan, ExecutionProfile},
        inference::{
            DecodeBindingCell, DecodeLinearBindings, LlamaInferenceSession, LlamaLayerWeights,
            LlamaLoadedWeights, SamplingConfig,
        },
        model::LlamaModelConfig,
        tensor::CpuTensor,
        tokenizer::{
            BpePreTokenizer, BpeRegistry, SpecialTokens, Token, TokenKind, Tokenizer,
            TokenizerConfig, TokenizerModel,
        },
    };

    use super::*;

    #[test]
    fn health_backend_reports_the_effective_serving_lane() {
        assert_eq!(health_backend(false, false, false, false), "none");
        assert_eq!(health_backend(false, false, false, true), "llama");
        assert_eq!(health_backend(false, true, false, true), "runnable-runtime");
        assert_eq!(
            health_backend(false, false, true, true),
            "diffusion-gemma-runtime"
        );
        assert_eq!(
            health_backend(true, true, true, true),
            "gemma4-runtime",
            "the eagerly loaded Gemma4 runtime has highest precedence"
        );
    }

    // ---------------------------------------------------------------------
    // Engine-inversion orphan-decode harness (P0-T1 / P0-T2 / P0-T3).
    //
    // The deleted `generation_lock_serializes_decoding` test proved that lock
    // GUARDS serialized — it could not see compute outliving its guard (the
    // orphan-decode hazard, Gate 0). Its successor invariant lives in
    // `engine::tests::engine_executes_at_most_one_job_at_a_time`, measured on
    // the compute itself. These tests assert the request-level invariant: no
    // decode is ever live when the next request becomes the exclusive compute
    // owner, and two decodes never overlap, across client disconnect, server
    // timeout, and mid-stream hangup. On the pre-inversion tree they FAILED;
    // that failure is the Gate 0 receipt bundle.
    // ---------------------------------------------------------------------

    struct OrphanObservation {
        /// Decodes still live when the next request became the compute owner.
        orphans_at_next_acquire: usize,
        /// Peak concurrent decodes over the whole scenario.
        max_concurrent: usize,
        /// Error response from the follow-up request B, if it failed.
        request_b_error: Option<String>,
    }

    pub(super) fn orphan_test_prepared(model: &str) -> PreparedGeneration {
        let session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
        let mut prepared = prepared_for_cache("tiny", model, vec![1, 2], session);
        // The shared test tokenizer has an empty vocab, which fails the decode-
        // to-text tail of a SUCCESSFUL generation; give request B a real (if
        // tiny) vocab matching tiny_config's vocab_size of 3.
        let mut tokenizer = test_tokenizer();
        tokenizer.tokens = (0..3u32)
            .map(|id| Token {
                id,
                text: format!("t{id}"),
                score: 0.0,
                kind: TokenKind::Normal,
            })
            .collect();
        tokenizer.token_to_id = tokenizer
            .tokens
            .iter()
            .map(|token| (token.text.clone(), token.id))
            .collect();
        prepared.tokenizer = Arc::new(tokenizer);
        prepared
    }

    /// Runs "request B" the way the non-streaming handlers do — post the
    /// decode to the engine worker with CancelOnDrop in the frame — and
    /// reports what the probe saw at the moment B became the exclusive
    /// compute owner (engine thread + transitional streaming bridge).
    /// Callers must have cleared the test-sleep hook so B is fast.
    async fn run_request_b_and_observe(state: &AppState) -> OrphanObservation {
        let prepared_b = orphan_test_prepared("model-b.gguf");
        let _cancel_on_drop = CancelOnDrop(prepared_b.cancel.token.clone());
        let outcome = state
            .engine
            .run_exclusive(move || {
                // Exactly the handler's decode job, with the probe sampled at
                // the moment this request becomes the exclusive compute owner.
                let orphans_at_next_acquire = decode_probe::active();
                let result = run_decode_job_serialized(prepared_b);
                (orphans_at_next_acquire, result)
            })
            .await;
        let (orphans_at_next_acquire, result_b) = match outcome {
            Ok(observed) => observed,
            Err(err) => (usize::MAX, Err(engine_post_error_response(err))),
        };
        let request_b_error = match result_b {
            Ok(_) => None,
            Err(response) => {
                let status = response.status();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    .unwrap_or_else(|err| format!("<unreadable body: {err}>"));
                Some(format!("{status}: {body}"))
            }
        };
        OrphanObservation {
            orphans_at_next_acquire,
            max_concurrent: decode_probe::max_seen(),
            request_b_error,
        }
    }

    /// Waits for every live decode worker to exit so a failing scenario cannot
    /// leak an orphan (and a probe underflow) into sibling tests.
    async fn drain_decode_probe() {
        let started = Instant::now();
        while decode_probe::active() != 0 {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "decode worker never finished; probe would poison later tests",
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }

    fn assert_no_orphan_overlap(observed: OrphanObservation, scenario: &str) {
        assert!(
            observed.request_b_error.is_none(),
            "{scenario}: request B should decode normally, got {:?}",
            observed.request_b_error,
        );
        assert_eq!(
            observed.orphans_at_next_acquire, 0,
            "{scenario}: a request became the exclusive compute owner while a \
             previous request's decode was still running (orphaned compute)",
        );
        assert!(
            observed.max_concurrent <= 1,
            "{scenario}: two decodes overlapped (max concurrent = {}) — the exact \
             corruption class the engine's serialization exists to prevent",
            observed.max_concurrent,
        );
    }

    /// P0-T2 — deterministic: the server's own generation timeout returns a 503
    /// but `tokio::time::timeout` only drops the JoinHandle; the blocking decode
    /// is detached, not stopped. The handler frame then drops the lock guard and
    /// the next request decodes concurrently with the orphan.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn generation_timeout_must_not_orphan_decode() {
        let _env_guard = crate::test_support::env_lock();
        // Wait out any probe leaked by a prior test (e.g. a detached stream
        // step still finishing) before resetting the gauge.
        drain_decode_probe().await;
        decode_probe::reset();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "100");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "1500");

        let state = AppState::default();

        // Request A, exactly as the non-streaming handlers run it (see
        // chat_completions): the decode is an engine job.
        let prepared_a = orphan_test_prepared("model-a.gguf");
        let _cancel_on_drop_a = CancelOnDrop(prepared_a.cancel.token.clone());
        let result_a = generate_decoded_tokens_blocking(&state, prepared_a).await;
        assert!(
            result_a.is_err(),
            "request A must hit the generation timeout"
        );

        // Request B arrives right after the 503.
        std::env::set_var(GENERATION_TIMEOUT_ENV, "60000");
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        let observed = run_request_b_and_observe(&state).await;

        drain_decode_probe().await;
        std::env::remove_var(GENERATION_TIMEOUT_ENV);
        assert_no_orphan_overlap(observed, "P0-T2 generation timeout");
    }

    /// P0-T1 — client disconnect mid non-streaming request: hyper drops the
    /// handler future at its await point (modeled by aborting a task with the
    /// handler's exact shape), releasing the guard while the decode runs on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn client_disconnect_must_not_orphan_decode() {
        let _env_guard = crate::test_support::env_lock();
        // Wait out any probe leaked by a prior test (e.g. a detached stream
        // step still finishing) before resetting the gauge.
        drain_decode_probe().await;
        decode_probe::reset();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "60000");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "1500");

        let state = AppState::default();

        let state_for_a = state.clone();
        let handler_a = tokio::spawn(async move {
            // Exact handler shape (see chat_completions): the decode is an
            // engine job; CancelOnDrop fires if this frame is dropped.
            let prepared = orphan_test_prepared("model-a.gguf");
            let _cancel_on_drop = CancelOnDrop(prepared.cancel.token.clone());
            let _ = generate_decoded_tokens_blocking(&state_for_a, prepared).await;
        });

        // Wait until A's decode is actually live on the blocking pool, then
        // "disconnect": drop the handler future.
        let started = Instant::now();
        while decode_probe::active() == 0 {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "request A's decode never reached the blocking pool",
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        handler_a.abort();
        let _ = handler_a.await;

        // Request B right after the disconnect.
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        let observed = run_request_b_and_observe(&state).await;

        drain_decode_probe().await;
        std::env::remove_var(GENERATION_TIMEOUT_ENV);
        assert_no_orphan_overlap(observed, "P0-T1 client disconnect");
    }

    /// P0-T3 — client disconnect mid SSE stream: dropping the response body
    /// drops the `async_stream` generator (and the guard it holds) between
    /// polls while the current token step is still on the blocking pool. The
    /// orphan window is one step, and it overlaps the next request's decode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stream_disconnect_must_not_orphan_decode_step() {
        let _env_guard = crate::test_support::env_lock();
        // Wait out any probe leaked by a prior test (e.g. a detached stream
        // step still finishing) before resetting the gauge.
        drain_decode_probe().await;
        decode_probe::reset();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "60000");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "1500");

        let state = AppState::default();
        let response = stream_completion(&state, orphan_test_prepared("model-a.gguf"), true, false);

        // Drive the SSE body the way a client does. The reader task polls the
        // stream into its first token step; aborting it is the hangup — the
        // body (and generator, and guard) drop mid-step.
        let reader = tokio::spawn(async move {
            let _ = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        });
        let started = Instant::now();
        while decode_probe::active() == 0 {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "stream A's token step never reached the blocking pool",
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        reader.abort();
        let _ = reader.await;

        // Request B right after the disconnect.
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        let observed = run_request_b_and_observe(&state).await;

        drain_decode_probe().await;
        std::env::remove_var(GENERATION_TIMEOUT_ENV);
        assert_no_orphan_overlap(observed, "P0-T3 stream disconnect");
    }

    /// Regression test for the dg-vs-dg CUDA corruption hazard: on CUDA builds
    /// every DiffusionGemma serve request runs through the process-global
    /// engine singleton whose internal Mutex is per-kernel-op only, so
    /// `dg_chat_nonstreaming` / `dg_chat_streaming` must hold
    /// `DG_GENERATION_LOCK` for the whole generation inside their
    /// `spawn_blocking` closures. This verifies the lock serializes — with
    /// many blocking tasks acquiring it the way the handlers do, never more
    /// than one is inside the critical section at a time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dg_generation_lock_serializes_generations() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..32 {
            let active = active.clone();
            let max_seen = max_seen.clone();
            handles.push(tokio::task::spawn_blocking(move || {
                // Same acquisition both dg handlers perform: the guard is
                // taken on the blocking thread, around the whole generation.
                let _generation_guard = dg_generation_guard();
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, Ordering::SeqCst);
                // Linger while holding the guard. If the lock failed to
                // serialize, another task would enter here and push
                // `active` > 1.
                for _ in 0..8 {
                    std::thread::yield_now();
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "DG_GENERATION_LOCK must serialize dg generations: never more than one in flight",
        );
    }

    fn completion_request_with(
        prompt: Option<&str>,
        temperature: Option<f32>,
    ) -> CompletionRequest {
        CompletionRequest {
            model: None,
            prompt: prompt.map(str::to_string),
            stream: None,
            max_tokens: Some(16),
            temperature,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            seed: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            stop: None,
            n: None,
            best_of: None,
            logprobs: None,
            camelid_logit_token_ids: None,
            camelid_prompt_token_ids: None,
            camelid_dense_diagnostics: None,
            camelid_dense_diagnostic_generated_index: None,
            camelid_receipt: Some(true),
            unsupported_fields: HashMap::new(),
        }
    }

    #[test]
    fn receipt_stamp_records_response_format() {
        // M4 regression: a constrained request's receipt must carry the raw
        // response_format so replay re-compiles the same constraint -- an
        // unconstrained replay of a constrained generation deterministically
        // fails verify-receipt on an honest receipt.
        let response_format = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"schema": {
                "type": "object", "additionalProperties": false,
                "properties": {"a": {"type": "integer"}}, "required": ["a"]
            }}
        });
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "response_format": response_format,
        }))
        .expect("chat request parses");
        let stamp = receipt_request_stamp(&req).expect("stamp");
        assert_eq!(stamp.response_format, Some(response_format));

        // Unconstrained requests stamp None (the field stays out of the receipt).
        let plain: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }))
        .expect("chat request parses");
        assert_eq!(
            receipt_request_stamp(&plain)
                .expect("stamp")
                .response_format,
            None
        );
    }

    #[test]
    fn completion_receipt_stamp_records_prompt_endpoint_and_reproducibility() {
        // Greedy (no temperature/top-p/top-k): records the prompt verbatim under the
        // raw completions endpoint and is reproducible.
        let stamp = receipt_completion_request_stamp(&completion_request_with(
            Some("The three primary colors are"),
            None,
        ))
        .expect("greedy completion stamp");
        assert_eq!(stamp.endpoint, "/v1/completions");
        assert_eq!(
            stamp.messages_or_prompt,
            serde_json::Value::String("The three primary colors are".to_string())
        );
        assert_eq!(stamp.temperature, 0.0);
        assert!(stamp.reproducible);

        // Non-zero temperature is recorded but never presented as reproducible.
        let sampled =
            receipt_completion_request_stamp(&completion_request_with(Some("hello"), Some(0.8)))
                .expect("sampled completion stamp");
        assert_eq!(sampled.temperature, f64::from(0.8_f32));
        assert!(!sampled.reproducible);

        // A receipt without a prompt is a request error, not a silent empty record.
        assert!(receipt_completion_request_stamp(&completion_request_with(None, None)).is_err());
    }

    #[test]
    fn stream_timing_diagnostics_env_is_default_off_and_opt_in() {
        let _env_guard = crate::test_support::env_lock();
        env::remove_var(STREAM_TIMING_DIAGNOSTICS_ENV);
        assert!(!stream_timing_diagnostics_enabled());

        env::set_var(STREAM_TIMING_DIAGNOSTICS_ENV, "on");
        assert!(stream_timing_diagnostics_enabled());

        env::set_var(STREAM_TIMING_DIAGNOSTICS_ENV, "0");
        assert!(!stream_timing_diagnostics_enabled());
        env::remove_var(STREAM_TIMING_DIAGNOSTICS_ENV);
    }

    #[test]
    fn streaming_chunks_omit_camelid_diagnostics_by_default() {
        let chunk = ChatCompletionStreamChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk",
            created: 1,
            model: "test-model".into(),
            choices: vec![ChatCompletionStreamChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some("assistant"),
                    content: None,
                },
                finish_reason: None,
            }],
            camelid: None,
            usage: None,
        };

        let value = serde_json::to_value(chunk).expect("stream chunk should serialize");
        assert!(value.get("camelid").is_none());
        // The usage frame is omitted from the wire on every non-terminal chunk
        // (stream_options.include_usage off), keeping the baseline byte-identical.
        assert!(value.get("usage").is_none());
    }

    #[test]
    fn stream_options_include_usage_resolves_permissively() {
        use serde_json::json;
        // (a) absent -> off.
        assert!(!stream_options_include_usage(None));
        // null / non-object stream_options -> off (tolerated, never an error).
        assert!(!stream_options_include_usage(Some(&json!(null))));
        assert!(!stream_options_include_usage(Some(&json!("yes"))));
        assert!(!stream_options_include_usage(Some(&json!(true))));
        // Object without include_usage -> off.
        assert!(!stream_options_include_usage(Some(&json!({}))));
        // (b) include_usage: false -> off.
        assert!(!stream_options_include_usage(Some(
            &json!({"include_usage": false})
        )));
        // Wrong-typed include_usage -> off (matches the permissive oracle; the
        // request is never rejected â€” see ref_err_bad_type capture, HTTP 200).
        assert!(!stream_options_include_usage(Some(
            &json!({"include_usage": "yes"})
        )));
        assert!(!stream_options_include_usage(Some(
            &json!({"include_usage": 1})
        )));
        // (c) the one true case.
        assert!(stream_options_include_usage(Some(
            &json!({"include_usage": true})
        )));
        // Unknown subfields are tolerated and ignored (never promoted to a
        // support row); include_usage is still honored alongside them.
        assert!(stream_options_include_usage(Some(&json!({
            "include_usage": true,
            "continuous_usage_stats": true
        }))));
        assert!(!stream_options_include_usage(Some(&json!({
            "some_future_field": 42
        }))));
    }

    #[test]
    fn chat_request_accepts_stream_options_without_marking_it_unsupported() {
        // Declaring stream_options as a typed field removes it from the
        // flatten-captured unsupported_fields, so the chat route no longer
        // returns the old "stream_options are not supported yet" error.
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "qwen3",
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .expect("request with stream_options should deserialize");
        assert!(req.stream_options.is_some());
        assert!(!req.unsupported_fields.contains_key("stream_options"));
        assert!(stream_options_include_usage(req.stream_options.as_ref()));
    }

    #[test]
    fn min_p_and_repeat_penalty_are_typed_fields_not_unsupported() {
        // Declaring min_p/repeat_penalty as typed sampler fields removes them from
        // the flatten-captured unsupported_fields, so the generation routes no
        // longer reject them with the old "sampler field not supported yet" error.
        let chat: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "hi"}],
            "min_p": 0.05,
            "repeat_penalty": 1.2
        }))
        .expect("chat request with min_p/repeat_penalty should deserialize");
        assert_eq!(chat.min_p, Some(0.05));
        assert_eq!(chat.repeat_penalty, Some(1.2));
        assert!(!chat.unsupported_fields.contains_key("min_p"));
        assert!(!chat.unsupported_fields.contains_key("repeat_penalty"));

        // They thread all the way into SamplingConfig...
        let req: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi",
            "temperature": 0.8,
            "min_p": 0.05,
            "repeat_penalty": 1.2
        }))
        .expect("session request should deserialize");
        let config = sampling_config_from_request(&req).expect("valid sampler config");
        assert_eq!(config.min_p, Some(0.05));
        assert_eq!(config.repeat_penalty, 1.2);

        // ...and an out-of-range value is a typed 400, not a panic.
        let bad: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi",
            "min_p": 1.5
        }))
        .expect("session request should deserialize");
        assert!(sampling_config_from_request(&bad).is_err());
    }

    #[test]
    fn n_choices_bounds_are_validated() {
        // n within [1, MAX_N_CHOICES] is accepted; 0 and > MAX are typed errors.
        let at_cap: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "n": MAX_N_CHOICES
        }))
        .expect("session request should deserialize");
        assert!(validate_choice_and_logprob_fields(&at_cap).is_ok());

        let zero: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "n": 0
        }))
        .expect("session request should deserialize");
        assert!(validate_choice_and_logprob_fields(&zero).is_err());

        let too_many: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "n": MAX_N_CHOICES + 1
        }))
        .expect("session request should deserialize");
        assert!(validate_choice_and_logprob_fields(&too_many).is_err());
    }

    #[test]
    fn step_logprob_values_are_log_softmax() {
        // Uniform logits: every logprob = ln(1/n); ties broken by ascending id.
        let (chosen, top) = step_logprob_values(&[0.0, 0.0, 0.0, 0.0], 2, 2).unwrap();
        let uniform = (0.25f32).ln();
        assert!(
            (chosen - uniform).abs() < 1e-5,
            "chosen {chosen} vs {uniform}"
        );
        assert_eq!(
            top.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![0, 1]
        );
        for (_, lp) in &top {
            assert!((*lp - uniform).abs() < 1e-5);
        }
        // Non-uniform: chosen logprob == logit - logsumexp.
        let logits = [2.0f32, 1.0, 0.0];
        let lse = logits.iter().map(|l| l.exp()).sum::<f32>().ln();
        let (c0, _) = step_logprob_values(&logits, 0, 0).unwrap();
        assert!((c0 - (2.0 - lse)).abs() < 1e-4, "c0 {c0} vs {}", 2.0 - lse);
    }

    #[test]
    fn step_logprob_values_top_is_argmax_and_normalized() {
        let logits = [1.0f32, 3.0, 2.0, 0.5];
        let (_, top) = step_logprob_values(&logits, 0, logits.len()).unwrap();
        assert_eq!(top[0].0, 1, "argmax id");
        assert_eq!(top[1].0, 2);
        // exp of all logprobs sums to ~1 (a valid distribution).
        let total: f32 = top.iter().map(|(_, lp)| lp.exp()).sum();
        assert!((total - 1.0).abs() < 1e-4, "sum {total}");
        // top_n=0 yields no alternatives; chosen out of vocab is a typed error.
        assert!(step_logprob_values(&logits, 0, 0).unwrap().1.is_empty());
        assert!(step_logprob_values(&logits, 99, 1).is_err());
    }

    #[test]
    fn build_logprobs_shapes_match_steps() {
        let step = |tok: &str, lp: f32, top: Vec<(&str, f32)>| StepLogprob {
            chosen: TokenLogprob {
                token: tok.to_string(),
                logprob: lp,
                bytes: tok.as_bytes().to_vec(),
            },
            top: top
                .into_iter()
                .map(|(t, l)| TokenLogprob {
                    token: t.to_string(),
                    logprob: l,
                    bytes: t.as_bytes().to_vec(),
                })
                .collect(),
        };
        let steps = vec![
            step("He", -0.1, vec![("He", -0.1), ("Hi", -2.0)]),
            step("llo", -0.5, vec![("llo", -0.5)]),
        ];
        let chat = build_chat_logprobs(&steps);
        assert_eq!(chat.content.len(), 2);
        assert_eq!(chat.content[0].token, "He");
        assert_eq!(chat.content[0].top_logprobs.len(), 2);
        let comp = build_completion_logprobs(&steps);
        assert_eq!(comp.tokens, vec!["He".to_string(), "llo".to_string()]);
        assert_eq!(comp.token_logprobs.len(), 2);
        assert_eq!(comp.text_offset, vec![0, 2]); // "He" is 2 chars
        assert_eq!(comp.top_logprobs[0].get("Hi"), Some(&-2.0));
    }

    #[test]
    fn logprobs_request_validation() {
        // chat logprobs:true is now accepted (was a 400 stub).
        let chat: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "chat_logprobs": true, "top_logprobs": 5
        }))
        .expect("deserialize");
        assert!(validate_choice_and_logprob_fields(&chat).is_ok());
        // top_logprobs without logprobs:true is rejected.
        let bad: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "top_logprobs": 5
        }))
        .expect("deserialize");
        assert!(validate_choice_and_logprob_fields(&bad).is_err());
        // out-of-range is rejected.
        let oob: GenerationSessionRequest = serde_json::from_value(serde_json::json!({
            "prompt": "hi", "completion_logprobs": MAX_LOGPROBS + 1
        }))
        .expect("deserialize");
        assert!(validate_choice_and_logprob_fields(&oob).is_err());
    }

    #[test]
    fn constrained_output_is_never_reclassified_as_tool_call() {
        // M3 regression: a schema that legitimately declares a `name` property
        // produces output parse_tool_calls would happily match -- under an active
        // constraint that output is content by definition and must not be
        // reclassified (content emptied, fabricated ToolCall, finish_reason
        // "tool_calls"). OpenAI allows tools + response_format together.
        let constrained_content = r#"{"name":"x","parameters":{}}"#;
        assert!(
            parse_tool_calls(constrained_content).is_some(),
            "the gate, not the parser, is what protects constrained output"
        );
        assert!(!should_parse_tool_calls(true, true));
        assert!(should_parse_tool_calls(true, false));
        assert!(!should_parse_tool_calls(false, false));
        assert!(!should_parse_tool_calls(false, true));
    }

    #[test]
    fn parse_tool_calls_extracts_llama_format() {
        // Llama 3.x: {"name", "parameters"}; arguments becomes a JSON string.
        let tc = parse_tool_calls(r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#)
            .unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "get_weather");
        assert_eq!(tc[0].kind, "function");
        assert!(tc[0].id.starts_with("call_"));
        let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    #[test]
    fn parse_tool_calls_tolerates_python_tag_and_junk() {
        // python_tag prefix + trailing junk small models emit; "arguments" variant.
        let tc = parse_tool_calls(r#"<|python_tag|>{"name": "f", "arguments": {"x": 1}} trailing"#)
            .unwrap();
        assert_eq!(tc[0].function.name, "f");
        let args: serde_json::Value = serde_json::from_str(&tc[0].function.arguments).unwrap();
        assert_eq!(args["x"], 1);
        // prose is not a tool call; a JSON object without "name" is not either.
        assert!(parse_tool_calls("The weather in Paris is sunny.").is_none());
        assert!(parse_tool_calls(r#"{"parameters": {"x": 1}}"#).is_none());
    }

    #[test]
    fn tool_choice_none_suppresses_calls() {
        assert!(!tool_choice_allows_calls(Some(&serde_json::json!("none"))));
        assert!(tool_choice_allows_calls(Some(&serde_json::json!("auto"))));
        assert!(tool_choice_allows_calls(Some(&serde_json::json!(
            "required"
        ))));
        assert!(tool_choice_allows_calls(None));
    }

    #[test]
    fn response_format_maps_to_constraint() {
        use serde_json::json;
        // json_object and a supported json_schema enable constrained decoding;
        // text / absent leave it off.
        assert!(
            constraint_from_response_format(Some(&json!({"type": "json_object"})))
                .unwrap()
                .is_some()
        );
        assert!(
            constraint_from_response_format(Some(&json!({"type": "text"})))
                .unwrap()
                .is_none()
        );
        assert!(constraint_from_response_format(None).unwrap().is_none());
        assert!(constraint_from_response_format(Some(&json!(null)))
            .unwrap()
            .is_none());
        // A supported json_schema compiles.
        let good = json!({
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {
                "type": "object", "additionalProperties": false,
                "properties": {"a": {"type": "string"}}, "required": ["a"]
            }}
        });
        assert!(constraint_from_response_format(Some(&good))
            .unwrap()
            .is_some());
        // A schema outside the subset is a typed error, not silently ignored.
        let bad = json!({"type": "json_schema", "json_schema": {"schema": {"type": "string"}}});
        assert!(constraint_from_response_format(Some(&bad)).is_err());
        // A json_schema without the schema payload is an error.
        assert!(constraint_from_response_format(Some(&json!({"type": "json_schema"}))).is_err());
        // Unknown response_format types remain errors.
        assert!(constraint_from_response_format(Some(&json!({"type": "yaml"}))).is_err());
    }

    #[test]
    fn terminal_usage_chunk_has_empty_choices_array_and_usage() {
        // The terminal include_usage frame: empty `choices` array (not omitted),
        // the three real usage integers, no camelid diagnostics. Mirrors the
        // llama-server oracle terminal chunk (minus its server-specific extras).
        let chunk = ChatCompletionStreamChunk {
            id: "chatcmpl-test".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "test-model".into(),
            choices: Vec::new(),
            camelid: None,
            usage: Some(CompletionUsage {
                prompt_tokens: 21,
                completion_tokens: 16,
                total_tokens: 37,
            }),
        };
        let value = serde_json::to_value(chunk).expect("usage chunk should serialize");
        assert_eq!(value.get("choices"), Some(&serde_json::json!([])));
        assert_eq!(value["object"], "chat.completion.chunk");
        assert_eq!(value["usage"]["prompt_tokens"], 21);
        assert_eq!(value["usage"]["completion_tokens"], 16);
        assert_eq!(value["usage"]["total_tokens"], 37);
        assert!(value.get("camelid").is_none());
    }

    #[test]
    fn stream_timing_diagnostics_json_sums_roles_and_scopes_q8_schedule() {
        let mut timings = GenerationTimings {
            generate: 1234,
            weight_cache_hit: true,
            prompt_cache_hit: false,
            ..GenerationTimings::default()
        };
        timings.prompt_evaluation.prefill.forward_total = 10.0;
        timings.prompt_evaluation.prefill.logits = 1.5;
        timings.prompt_evaluation.prefill.sample = 0.25;
        timings.prompt_evaluation.first_token.forward_total = 20.0;
        timings.prompt_evaluation.first_token.logits = 2.5;
        timings.prompt_evaluation.first_token.sample = 0.75;
        timings.generation.forward_total = 30.0;
        timings.prompt_evaluation.prefill_layers = vec![GenerationLayerTimings {
            layer_index: 2,
            attention_context: 1.25,
            attention_output: 2.0,
            ffn_gate: 3.0,
            ffn_up: 4.0,
            ffn_down: 5.0,
            ..GenerationLayerTimings::default()
        }];
        timings.prompt_evaluation.first_token_layers = vec![GenerationLayerTimings {
            layer_index: 3,
            attention_context: 0.5,
            ..GenerationLayerTimings::default()
        }];
        timings.layers = vec![
            GenerationLayerTimings {
                layer_index: 4,
                ffn_down: 7.0,
                ..GenerationLayerTimings::default()
            },
            GenerationLayerTimings {
                layer_index: 5,
                ffn_down: 11.0,
                attention_output: 13.0,
                ..GenerationLayerTimings::default()
            },
        ];

        let value = stream_timing_diagnostics_json(
            &timings,
            Some(321),
            StreamEventTimings {
                poll_yield_enabled: true,
                role_yield: Some(3),
                generate_start: Some(5),
                first_content_yield: Some(326),
                final_yield: Some(1239),
            },
        );
        let diagnostics = &value["stream_timing_diagnostics"];
        assert_eq!(diagnostics["timings_ms"]["generate"], 1234);
        assert_eq!(diagnostics["timings_ms"]["first_content"], 321);
        assert_eq!(
            diagnostics["timings_ms"]["stream_event_accounting"]["poll_yield_enabled"],
            true
        );
        assert_eq!(
            diagnostics["timings_ms"]["stream_event_accounting"]["role_yield"],
            3
        );
        assert_eq!(
            diagnostics["timings_ms"]["stream_event_accounting"]["generate_start_minus_role_yield"],
            2
        );
        assert_eq!(
            diagnostics["timings_ms"]["stream_event_accounting"]
                ["first_content_yield_minus_role_yield"],
            323
        );
        assert_eq!(
            diagnostics["timings_ms"]["stream_event_accounting"]
                ["final_yield_minus_first_content_yield"],
            913
        );
        assert_eq!(
            diagnostics["timings_ms"]["first_content_accounting"]["prompt_eval_forward_total"],
            30.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["first_content_accounting"]["prompt_eval_logits"],
            4.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["first_content_accounting"]["prompt_eval_sample"],
            1.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["first_content_accounting"]
                ["first_content_minus_prompt_eval_forward"],
            291.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["first_content_accounting"]
                ["first_content_minus_prompt_eval_forward_plus_sample"],
            290.0
        );
        assert_eq!(diagnostics["timings_ms"]["weight_cache_hit"], true);
        assert_eq!(diagnostics["timings_ms"]["prompt_cache_hit"], false);
        assert_eq!(
            diagnostics["timings_ms"]["prefill_role_timings"]["ffn_down"],
            5.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["generation_role_timings"]["ffn_down"],
            18.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["generation_role_timings"]["attention_output"],
            13.0
        );
        assert_eq!(
            diagnostics["timings_ms"]["layer_role_hotspots"]["prefill"][0]["role"],
            "ffn_down"
        );
        assert_eq!(
            diagnostics["timings_ms"]["layer_role_hotspots"]["prefill"][0]["layer_index"],
            2
        );
        assert_eq!(
            diagnostics["timings_ms"]["layer_role_hotspots"]["generation"][0]["role"],
            "attention_output"
        );
        assert_eq!(
            diagnostics["timings_ms"]["layer_role_hotspots"]["generation"][0]["elapsed_ms"],
            13.0
        );
        assert!(diagnostics["q8_schedule"].is_null());
    }

    #[test]
    fn capabilities_can_include_selected_execution_plan() {
        let plan = ExecutionPlan {
            profile: ExecutionProfile::Experimental,
            operating_system: "linux".into(),
            architecture: "x86_64".into(),
            platform_label: "Ubuntu/Linux x86_64".into(),
            cpu_model: "Intel Xeon Platinum 8488C".into(),
            cpu_features: vec!["avx2".into()],
            model_family: "llama".into(),
            quant_type: "Q8_0".into(),
            exact_model_row: "Llama 3.2 3B Instruct".into(),
            support_level: "supported_exact_row_smoke_512_1024_2048_4096_8192".into(),
            selected_backend: "cpu_q8_runtime_repack".into(),
            selected_q8_path: "x86_experimental_q8_0_avx2".into(),
            prefill_path: "q8_0_x86_avx2_tiled_gemm_experimental".into(),
            prefill_runtime_policy: "manual_override_only".into(),
            decode_path: "q8_0_decode_avx2".into(),
            thread_count: 16,
            diagnostics_status: "standard diagnostics; RSS timings disabled by default".into(),
            fallback_path: "retained_q8_reference_path".into(),
            cuda_resident_active: false,
            reasons: vec!["default-off Ubuntu x86_64 experiment selected".into()],
        };

        let response = capabilities_response_with_plan(Some(plan.clone()));

        assert_eq!(response.execution_plan, Some(plan));
    }

    #[test]
    fn capabilities_report_llama32_context_pack_boundaries() {
        let response = capabilities_response();
        let targets = response
            .model_compatibility
            .iter()
            .filter(|target| {
                matches!(
                    target.id,
                    "llama32_1b_instruct_q8_0" | "llama32_3b_instruct_q8_0"
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(targets.len(), 2);
        for target in targets {
            assert_eq!(target.status, "supported_exact_row_smoke");
            assert_eq!(target.bounded_context_window, 512);
            assert_eq!(target.bounded_context_1024_window, 1024);
            // The rows no longer share pack lineage: the 1B keeps the May
            // template-pack ids; the 3B's five buckets were re-validated on its
            // anchored canonical GGUF via the raw-decode ladder.
            match target.id {
                "llama32_1b_instruct_q8_0" => {
                    assert_eq!(target.bounded_context_512_pack, "validated_bounded_pack");
                    assert_eq!(
                        target.bounded_context_512_pack_id,
                        "llama3-context-512-smoke-v1"
                    );
                    assert_eq!(target.bounded_context_1024_pack, "validated_second_pack");
                    assert_eq!(
                        target.bounded_context_1024_pack_id,
                        "llama3-context-1024-smoke-v1"
                    );
                    assert!(target
                        .evidence
                        .contains("second bounded 1024-context parity"));
                }
                "llama32_3b_instruct_q8_0" => {
                    assert_eq!(
                        target.bounded_context_512_pack,
                        "validated_anchored_raw_decode_ladder"
                    );
                    assert_eq!(
                        target.bounded_context_512_pack_id,
                        "llama32-3b-anchored-raw-ladder-v1"
                    );
                    assert_eq!(
                        target.bounded_context_1024_pack,
                        "validated_anchored_raw_decode_ladder"
                    );
                    assert_eq!(
                        target.bounded_context_1024_pack_id,
                        "llama32-3b-anchored-raw-ladder-v1"
                    );
                }
                other => panic!("unexpected row {other}"),
            }
        }

        let one_b = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "llama32_1b_instruct_q8_0")
            .expect("1B row should stay advertised");
        assert_eq!(one_b.bounded_context_2048_pack, "validated_third_pack");
        assert_eq!(
            one_b.bounded_context_2048_pack_id,
            "llama3-context-2048-smoke-v1"
        );
        assert_eq!(one_b.bounded_context_2048_window, 2048);
        assert_eq!(one_b.bounded_context_4096_pack, "validated_fourth_pack");
        assert_eq!(
            one_b.bounded_context_4096_pack_id,
            "llama3-context-4096-smoke-v1"
        );
        assert_eq!(one_b.bounded_context_4096_window, 4096);
        assert_eq!(one_b.bounded_context_8192_pack, "validated_fifth_pack");
        assert_eq!(
            one_b.bounded_context_8192_pack_id,
            "llama3-context-8192-smoke-v1"
        );
        assert_eq!(one_b.bounded_context_8192_window, 8192);
        assert_eq!(one_b.latest_checked_bucket, "llama3-context-8192-smoke-v1");
        assert_eq!(one_b.latest_checked_output, "CMLD-819");
        assert!(one_b.evidence.contains("fifth bounded 8192-context parity"));

        let three_b = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "llama32_3b_instruct_q8_0")
            .expect("3B row should stay advertised");
        // Anchored raw-decode ladder: all five buckets re-validated on the
        // current canonical GGUF in one bundle (see the
        // llama32-3b-context-512-8192-anchored evidence bundle).
        assert_eq!(
            three_b.bounded_context_2048_pack,
            "validated_anchored_raw_decode_ladder"
        );
        assert_eq!(
            three_b.bounded_context_2048_pack_id,
            "llama32-3b-anchored-raw-ladder-v1"
        );
        assert_eq!(three_b.bounded_context_2048_window, 2048);
        assert_eq!(
            three_b.bounded_context_4096_pack,
            "validated_anchored_raw_decode_ladder"
        );
        assert_eq!(
            three_b.bounded_context_4096_pack_id,
            "llama32-3b-anchored-raw-ladder-v1"
        );
        assert_eq!(three_b.bounded_context_4096_window, 4096);
        assert_eq!(
            three_b.bounded_context_8192_pack,
            "validated_anchored_raw_decode_ladder"
        );
        assert_eq!(
            three_b.latest_checked_bucket,
            "llama32-3b-anchored-raw-ladder-v1"
        );
        assert!(three_b.evidence.contains("sha256 f34112a1"));
        assert!(three_b.evidence.contains("PRIOR upload"));

        let eight_b = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "llama3_8b_instruct_q8_0")
            .expect("8B row should stay advertised");
        assert_eq!(eight_b.bounded_context_512_pack, "validated_first_pack");
        assert_eq!(
            eight_b.bounded_context_512_pack_id,
            "llama3-context-512-smoke-v1"
        );
        assert_eq!(eight_b.bounded_context_1024_pack, "validated_second_pack");
        assert_eq!(
            eight_b.bounded_context_1024_pack_id,
            "llama3-context-1024-smoke-v1"
        );
        assert_eq!(eight_b.bounded_context_1024_window, 1024);
        assert_eq!(eight_b.bounded_context_2048_pack, "validated_third_pack");
        assert_eq!(
            eight_b.bounded_context_2048_pack_id,
            "llama3-context-2048-smoke-v1"
        );
        assert_eq!(eight_b.bounded_context_2048_window, 2048);
        assert_eq!(eight_b.bounded_context_4096_pack, "not_promoted");
        assert_eq!(eight_b.bounded_context_4096_pack_id, "not_selected");
        assert_eq!(eight_b.bounded_context_4096_window, 4096);
        assert_eq!(
            eight_b.latest_checked_bucket,
            "llama3-context-2048-smoke-v1"
        );
        assert_eq!(eight_b.latest_checked_output, "CMLD-204");
        assert!(eight_b
            .evidence
            .contains("checked 512/1024/2048-context packs"));
        assert!(eight_b.evidence.contains("current-head 1024/2048 pass"));
    }

    #[test]
    fn capabilities_report_qwen3_chatml_supported_rows() {
        let response = capabilities_response();
        for id in [
            "qwen3_0_6b_instruct_q8_0",
            "qwen3_1_7b_instruct_q8_0",
            "qwen3_4b_instruct_q8_0",
            "qwen3_8b_instruct_q8_0",
        ] {
            let target = response
                .model_compatibility
                .iter()
                .find(|target| target.id == id)
                .unwrap_or_else(|| panic!("{id} should be advertised in model_compatibility"));
            assert_eq!(target.family, "qwen3", "{id} family");
            assert_eq!(target.quantization, "Q8_0", "{id} quant");
            assert_eq!(target.status, "supported_exact_row_smoke", "{id} status");
            assert_eq!(
                target.chat_template_renderer, "qwen3_chatml_thinking_disabled",
                "{id} renderer"
            );
            // Parity proven on both the scalar and the AVX2 x86_q8 paths on Windows.
            assert!(
                target.parity_audited.contains("x86_q8_avx2"),
                "{id} parity_audited should cite the x86_q8 AVX2 path"
            );
            // Each row cites its Windows x86_64 parity bundle.
            assert!(
                target.evidence.contains("windows-x86-chatml-parity"),
                "{id} evidence should cite the Windows bundle"
            );
            // Qwen3-4B Q8_0 (512-8192), Qwen3-1.7B Q8_0 (512-4096; 8192 near-tie), and
            // Qwen3-0.6B Q8_0 (512 + 2048/4096/8192; 1024 is a documented benign near-tie
            // hole) have promoted bounded-context packs; only the 8B row stays ChatML
            // short-chat smoke only.
            let expected_ctx512 = if id == "qwen3_4b_instruct_q8_0"
                || id == "qwen3_1_7b_instruct_q8_0"
                || id == "qwen3_0_6b_instruct_q8_0"
            {
                "validated_bounded_pack"
            } else {
                "not_promoted"
            };
            assert_eq!(
                target.bounded_context_512_pack, expected_ctx512,
                "{id} ctx512"
            );
        }
    }

    #[test]
    fn classify_model_lane_separates_supported_experimental_and_unsupported() {
        // Exact supported curated artifact â†’ Supported.
        assert_eq!(
            classify_model_lane(Some("llama"), "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
            ModelLaneClass::Supported,
        );
        assert_eq!(
            classify_model_lane(Some("qwen3"), "Qwen3-0.6B-Q8_0.gguf"),
            ModelLaneClass::Supported,
        );
        // Non-catalog allowlisted artifact of a supported row (in-house requant
        // with no HF catalog source) â†’ Supported.
        assert_eq!(
            classify_model_lane(Some("qwen35"), "ornith-1.0-9b-Q4_K_M.gguf"),
            ModelLaneClass::Supported,
        );
        assert_eq!(
            classify_model_lane(Some("qwen35"), "ornith-1.0-9b-Q3_K_M.gguf"),
            ModelLaneClass::Supported,
        );
        assert_eq!(
            classify_model_lane(Some("qwen35"), "ornith-1.0-9b-Q8_0.gguf"),
            ModelLaneClass::Supported,
        );
        // Implemented architecture but NOT a supported exact artifact (different
        // quant/filename) â†’ experimental, never falsely supported.
        assert_eq!(
            classify_model_lane(Some("qwen3"), "Qwen3-0.6B-Q4_K_M.gguf"),
            ModelLaneClass::ExperimentalImplemented,
        );
        assert_eq!(
            classify_model_lane(Some("qwen35"), "ornith-1.0-9b-Q6_K.gguf"),
            ModelLaneClass::ExperimentalImplemented,
        );
        assert_eq!(
            classify_model_lane(Some("mistral"), "some-random-mistral-finetune-Q8_0.gguf"),
            ModelLaneClass::ExperimentalImplemented,
        );
        // Architecture not in the implemented set â†’ Unsupported (fails closed at load).
        assert_eq!(
            classify_model_lane(Some("falcon"), "falcon-7b-Q8_0.gguf"),
            ModelLaneClass::Unsupported,
        );
        assert_eq!(
            classify_model_lane(None, "headerless.gguf"),
            ModelLaneClass::Unsupported,
        );
    }

    #[test]
    fn backend_error_code_is_stable_and_switchable() {
        use crate::BackendError;
        assert_eq!(
            backend_error_code(&BackendError::UnsupportedModelArchitecture("falcon".into())),
            "unsupported_model_architecture",
        );
        assert_eq!(
            backend_error_code(&BackendError::InvalidModelMetadata("x".into())),
            "invalid_model_metadata",
        );
        assert_eq!(
            backend_error_code(&BackendError::UnsupportedTokenizer("x".into())),
            "unsupported_tokenizer",
        );
        assert_eq!(
            backend_error_code(&BackendError::UnsupportedGguf("x".into())),
            "unsupported_gguf",
        );
        // The offending architecture rides in the message, not the code.
        assert!(BackendError::UnsupportedModelArchitecture("falcon".into())
            .to_string()
            .contains("falcon"));
    }

    #[test]
    fn capabilities_report_exact_8b_1024_2048_after_current_head_alignment() {
        let response = capabilities_response();
        assert!(response.support_contract.current_gate.contains(
            "Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs"
        ));
        assert!(response.support_contract.current_gate.contains(
            "Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192 raw-decode context ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence)"
        ));
        assert!(response
            .support_contract
            .current_gate
            .contains("Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs"));
        assert!(response
            .support_contract
            .current_gate
            .contains("where row-specific PASS artifacts exist"));
        assert!(response
            .support_contract
            .current_gate
            .contains("no model-native/larger context beyond the checked packs"));

        let q8 = response
            .supported_quantization
            .iter()
            .find(|item| item.id == "Q8_0")
            .expect("Q8_0 row should stay advertised");
        assert!(q8.notes.contains(
            "exact Llama 3.2 1B Instruct Q8_0 now has checked bounded 512/1024/2048/4096/8192-context packs"
        ));
        assert!(q8.notes.contains(
            "exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder on the current canonical GGUF (prior-upload Ubuntu API/WebUI refresh at source head e9f926ed1a65 retained as historical evidence)"
        ));
        assert!(q8.notes.contains(
            "exact Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048-context packs"
        ));
        assert!(q8.notes.contains("where row-specific PASS artifacts exist"));
        assert!(!q8.notes.contains("8B 1024/2048 remain red"));

        let llama_bpe = response
            .supported_model_families
            .iter()
            .find(|item| item.id == "llama_bpe_decoder_exact_1b_3b_8b_q8_0")
            .expect("Llama BPE exact-row family should stay advertised");
        assert!(llama_bpe.notes.contains(
            "exact Llama 3.2 1B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048/4096/8192-context packs"
        ));
        assert!(llama_bpe.notes.contains(
            "exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke standing on the anchored checked bounded 512/1024/2048/4096/8192-context raw-decode ladder for the current canonical GGUF (prior-upload Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 retained as historical)"
        ));
        assert!(llama_bpe.notes.contains(
            "exact Llama 3 8B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048-context packs"
        ));
        assert!(llama_bpe
            .notes
            .contains("published source/runtime-head 8B 1024/2048 PASS bundle"));
        assert!(!llama_bpe.notes.contains("8B 1024/2048 current-head bundle"));
        assert!(!llama_bpe.notes.contains("8B 1024/2048 remain red"));

        let eight_b = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "llama3_8b_instruct_q8_0")
            .expect("8B row should stay advertised");

        assert_eq!(eight_b.bounded_context_512_pack, "validated_first_pack");
        assert_eq!(eight_b.bounded_context_1024_pack, "validated_second_pack");
        assert_eq!(eight_b.bounded_context_2048_pack, "validated_third_pack");
        assert_eq!(
            eight_b.bounded_context_1024_pack_id,
            "llama3-context-1024-smoke-v1"
        );
        assert_eq!(
            eight_b.bounded_context_2048_pack_id,
            "llama3-context-2048-smoke-v1"
        );
        assert!(eight_b
            .tested_context
            .contains("checked_512_1024_2048_context_packs"));
        assert!(eight_b
            .next_step
            .contains("checked 512/1024/2048 context support"));
    }

    #[test]
    fn capabilities_report_current_rows_with_fail_closed_full_support_bar() {
        let response = capabilities_response();
        let current_row_ids = [
            "tinyllama_1_1b_chat_q8_0",
            "llama32_1b_instruct_q8_0",
            "llama32_3b_instruct_q8_0",
            "llama3_8b_instruct_q8_0",
        ];

        for id in current_row_ids {
            let target = response
                .model_compatibility
                .iter()
                .find(|target| target.id == id)
                .unwrap_or_else(|| panic!("{id} row should stay advertised"));

            assert!(
                target.frontend_readiness_gate.contains("loaded_now=true")
                    && target
                        .frontend_readiness_gate
                        .contains("generation_ready=true"),
                "{id} must keep frontend/API readiness fail-closed"
            );
            assert!(
                !target.full_support_status.is_empty() && !target.full_support_blockers.is_empty(),
                "{id} must carry the stricter full-support bar"
            );
            assert!(
                target.full_support_blockers.contains("template")
                    || target.full_support_blockers.contains("Jinja"),
                "{id} must not silently promote arbitrary/Jinja template coverage"
            );
            assert!(
                target.full_support_blockers.contains("production")
                    || target.full_support_blockers.contains("throughput"),
                "{id} must not silently promote production throughput"
            );
            assert!(
                !target
                    .performance_measured
                    .contains("production_throughput"),
                "{id} must keep bounded perf/RSS evidence distinct from production throughput"
            );
        }

        let tiny = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "tinyllama_1_1b_chat_q8_0")
            .expect("TinyLlama current gate row should stay advertised");
        assert_eq!(tiny.status, "supported_current_gate");
        assert_eq!(tiny.bounded_context_1024_pack, "not_promoted");
        assert_eq!(tiny.bounded_context_2048_pack, "not_promoted");
        assert_eq!(tiny.bounded_context_4096_pack, "not_promoted");

        let mistral = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "mistral_7b_instruct_v0_3_q8_0")
            .expect("Mistral exact-row lane should stay advertised");
        assert_eq!(mistral.status, "supported_exact_row_smoke");
        assert_eq!(mistral.support_scope, "exact_row_smoke_only");
        assert_eq!(
            mistral.full_support_status,
            "blocked_pending_normalized_full_support"
        );
        assert_eq!(mistral.frontend_load_path_verified, "validated");
        assert_eq!(
            mistral.performance_measured,
            "bounded_unique_chat_perf_rss_validated"
        );
        assert_eq!(
            mistral.latest_checked_bucket,
            "support_promotion_api_webui_smoke"
        );
        assert_eq!(mistral.latest_checked_result, "pass");
        assert_eq!(mistral.latest_checked_output, "CMLD-M7B");
        assert!(mistral.frontend_readiness_gate.contains("green only when"));
        assert_eq!(mistral.bounded_context_8192_pack, "validated_fifth_pack");
        assert_eq!(
            mistral.bounded_context_8192_pack_id,
            "mistral-context-8192-max-ladder-v1"
        );
        assert!(mistral
            .evidence
            .contains("support-promotion API/WebUI smoke bundle"));
    }

    #[test]
    fn capabilities_support_statuses_stay_exact_row_allowlisted() {
        let response = capabilities_response();
        let supported_row_ids = response
            .model_compatibility
            .iter()
            .filter(|target| target.status.starts_with("supported"))
            .map(|target| target.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            supported_row_ids,
            BTreeSet::from([
                // The gemma4 rows are `supported_exact_row_smoke`: exact-row
                // generation + serve smoke only (token-identical to the reference),
                // not bounded-context/perf/full support. Deliberately allowlisted.
                // E2B additionally has committed basic_v1 pack parity vs the
                // pinned llama.cpp 5d56eff oracle.
                "gemma4_e2b_it_q8_0",
                "gemma4_e4b_it_q8_0",
                // gemma4-E4B NVFP4 (GABBRO): supported_exact_row_smoke scoped to
                // exact_row_gpu_resident_raw_decode_parity_smoke_only — the Metal
                // resident NVFP4 lane, oracle-parity-anchored (G-M1 + Metal self-parity
                // + BASALT Leg B + macOS spot-check), clears the pre-registered G3 GO
                // rule in the current engine as a disclosed near-tie (90.5% vs Q4K 91.9%,
                // gap 1.4pp <= 2.0). Evidence: qa/evidence-bundles/gabbro/support/.
                "gemma4_e4b_it_nvfp4",
                // 12B is supported_exact_row_smoke SCOPED TO the two-Mac
                // distributed layer-sharding serve lane (single-node 16GB is
                // memory-bound); promotion bundle + WebUI closure committed.
                "gemma4_12b_it_q8_0",
                // 26B A4B QAT (Q4_0 experts + Q6_K head) is supported_exact_row_smoke
                // SCOPED TO the same two-Mac distributed lane: full basic_v1 parity
                // pack (2/5 full + 3/5 probe-verified frontiers) + distributed serve
                // smoke committed; 13.4GB row is memory-infeasible single-node.
                "gemma4_26b_a4b_it_q4_0",
                // Ternary TQ2_0 (qwen3 arch) single-node CPU completion-smoke lane:
                // streams TQ2_0 + Q6_K head (4B in 3GB RAM), 3/4 probe prompts
                // greedy token-identical vs llama.cpp acd79d6 + 1 near-tie. Decode
                // ~0.53x llama (general forward gap). See qa/ternary/ receipt.
                "ternary_bonsai_4b_tq2_0",
                "llama32_1b_instruct_q8_0",
                "llama32_3b_instruct_q8_0",
                // Llama-3.2-3B-Instruct K-quant rows (filename-anchored ids): GPU-resident
                // CUDA raw-decode parity vs llama.cpp acd79d6 — Q4_K_M confident-probe 5/8
                // (+ documented near-ties), Q5_K_M all_pass. Raw-decode smoke only.
                "llama_3_2_3b_instruct_q4_k_m",
                "llama_3_2_3b_instruct_q5_k_m",
                // Llama-3.2-1B-Instruct Q4_K_M (MUSTER M-B1, filename-anchored id):
                // GPU-resident CUDA raw-decode parity vs llama.cpp acd79d603 on the
                // committed 8-prompt pack — 5/8 token+text identical at every 1/5/50
                // depth (incl. a 1,867-token long-context leg), 3 open-ended flips
                // attributed benign near-ties (oracle-#2 at <=0.106 nat, CPU-lane
                // cross-backend control); prompt tokenization 8/8. LOCAL file,
                // upstream provenance unresolved (SHA-anchored). Raw-decode smoke only.
                "llama_3_2_1b_instruct_q4_k_m",
                // gemma3 1B Q8_0 (MUSTER M-A1, filename-anchored id): runnable serve
                // lane chat parity vs llama.cpp acd79d603 — 4/5 gate-pack prompts
                // token+text identical at every 1/5/50 depth, fifth identical at 1/5
                // with a single documented 50-token flip (oracle-#2, 0.3416-nat gap
                // stated; above the Ornith soft line, disclosed in the evidence);
                // prompt tokenization 5/5 cross-engine; renderer byte-locked by the
                // shapes pack. Chat smoke only, <512-token sliding-window scope.
                "gemma_3_1b_it_q8_0",
                // Llama-3.2-1B-Instruct IQ4_XS (i-quant, llama BPE): GPU-resident
                // iq4xs_gemv raw-decode greedy parity vs llama.cpp acd79d6 — 2/4
                // probe prompts 24-token identical, 2/4 prefix-then-near-tie (the
                // reference token is camelid's #2). Kernel matches the CPU oracle
                // within 1e-4. Raw-completion smoke only. See qa/iquant/ receipt.
                "llama3_2_1b_instruct_iq4_xs",
                "llama3_8b_instruct_q8_0",
                "mistral_7b_instruct_v0_3_q8_0",
                // Dense Qwen3 Q8_0 ChatML rows (thinking disabled): exact-row
                // token+text parity vs llama.cpp at 1/5/50 on macOS/Ubuntu and on
                // Windows x86_64 CPU (cpu_reference + x86_q8 AVX2, bit-identical).
                // 4B (512-8192), 1.7B (512-4096; 8192 near-tie), and 0.6B
                // (512+2048/4096/8192; 1024 near-tie hole) additionally carry checked
                // bounded-context packs; only 8B stays short-chat smoke only.
                "qwen3_0_6b_instruct_q8_0",
                "qwen3_1_7b_instruct_q8_0",
                "qwen3_4b_instruct_q8_0",
                "qwen3_8b_instruct_q8_0",
                // Qwen3-4B Q4_K_M (mixed Q4_K + Q6_K): GPU-resident CUDA decode
                // token+text-identical to llama.cpp acd79d6 at 1/5/50 (ChatML, thinking
                // disabled) + a default-on CPU K-quant block-dot confident-probe lane.
                // Filename-anchored id (general.name carries a cosmetic "Awq" token).
                "qwen3_4b_q4_k_m",
                "tinyllama_1_1b_chat_q8_0",
                // Ornith-1.0-9B (qwen35 hybrid gated-delta-net) runnable serve lane:
                // exact-row serve smoke + greedy token-identical parity vs llama.cpp
                // acd79d6 (4 prompts) + tool_capable via 3 agent-eval PASS receipts.
                // Short-chat/agent smoke only; no bounded-context/perf/full support.
                "Ornith 1.0 9B",
                // Ornith-1.0-9B CUDA-resident quant rows: Q4_K_M has a 5-prompt
                // cross-backend-tolerance parity PASS vs llama.cpp acd79d6 CUDA
                // (attributed near-ties) + a full agent-eval battery PASS on the
                // exact file; Q3_K_M has 16K full-residency + GPU==CPU-oracle
                // parity with a documented direct-cross-engine frontier.
                "ornith_1_0_9b_q4_k_m",
                "ornith_1_0_9b_q3_k_m",
            ])
        );

        let supported_family_ids = response
            .supported_model_families
            .iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            supported_family_ids,
            BTreeSet::from([
                "llama_bpe_decoder_exact_1b_3b_8b_q8_0",
                "llama_spm_decoder",
                "mistral_instruct_exact_7b_v0_3_q8_0",
                "qwen3_chatml_exact_0_6b_1_7b_4b_8b_q8_0",
            ])
        );

        for id in [
            "qwen25_7b_instruct_q8_0",
            "gemma2_9b_it_q8_0",
            "deepseek_r1_distill_qwen_7b_q8_0",
            "c4ai_command_r_v01_q8_0",
            "deepseek_r1_distill_llama_8b_q8_0",
            "qwen25_coder_7b_q8_0",
            "llama31_8b_instruct_q8_0",
        ] {
            let target = response
                .model_compatibility
                .iter()
                .find(|target| target.id == id)
                .unwrap_or_else(|| panic!("{id} row should stay advertised"));
            assert!(
                !target.status.starts_with("supported"),
                "{id} must not become supported through family-level inference"
            );
            assert!(
                target.frontend_readiness_gate.contains("fail-closed"),
                "{id} must keep frontend readiness fail-closed"
            );
        }

        assert!(response
            .support_contract
            .current_gate
            .contains("Current exact-row support"));
        assert!(response.support_contract.current_gate.contains(
            "no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied"
        ));
    }

    #[test]
    fn capabilities_report_next_family_rows_stay_planned_and_fail_closed() {
        let response = capabilities_response();

        let mixtral = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "mixtral_8x7b_instruct_v0_1_q8_0")
            .expect("Mixtral exact-row lane should stay visible");
        assert_eq!(mixtral.status, "active_validation_partial_runtime");
        assert_eq!(mixtral.support_scope, "exact_row_bounded_moe_runtime_only");
        assert_eq!(
            mixtral.generation_runs,
            "bounded_one_token_runtime_smoke_observed"
        );
        assert_eq!(
            mixtral.frontend_load_path_verified,
            "fail_closed_partial_runtime_only"
        );
        assert_eq!(
            mixtral.latest_checked_result,
            "blocked_later_generation_divergence"
        );
        assert!(mixtral.frontend_readiness_gate.contains("fail-closed"));
        assert!(mixtral.evidence.contains("llama.expert_count=8"));
        assert!(mixtral
            .evidence
            .contains("Gate 9A 50-token evidence diverged at generated token index 9"));
        assert!(mixtral.evidence.contains("No broad Mixtral"));

        let planned_rows = [
            "qwen25_7b_instruct_q8_0",
            "gemma2_9b_it_q8_0",
            "deepseek_r1_distill_qwen_7b_q8_0",
            "c4ai_command_r_v01_q8_0",
            "deepseek_r1_distill_llama_8b_q8_0",
            "qwen25_coder_7b_q8_0",
            "llama31_8b_instruct_q8_0",
        ];

        for id in planned_rows {
            let target = response
                .model_compatibility
                .iter()
                .find(|target| target.id == id)
                .unwrap_or_else(|| panic!("{id} planned row should stay advertised"));

            assert_eq!(target.status, "planned_exact_row_candidate");
            assert_eq!(target.support_scope, "future_exact_row_planning_only");
            assert_eq!(
                target.full_support_status,
                "not_applicable_until_runtime_support"
            );
            assert_eq!(target.tensors_load, "not_started");
            assert_eq!(target.generation_runs, "not_started");
            assert_eq!(target.parity_audited, "not_started");
            assert_eq!(target.performance_measured, "not_started");
            assert_eq!(target.bounded_context_512_pack, "not_started");
            assert_eq!(target.bounded_context_1024_pack, "not_started");
            assert_eq!(target.bounded_context_2048_pack, "not_started");
            assert_eq!(target.latest_checked_result, "planning_only");
            assert!(target.frontend_readiness_gate.contains("fail-closed"));
            assert!(target.evidence.contains("planning only"));
        }

        let phi3 = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "phi3_mini_4k_instruct_q8_0")
            .expect("Phi-3 exact-row validation lane should stay visible");
        // The row stays blocked and fail-closed: prompt-token parity was cleared
        // 2026-07-27, but the forward pass still diverges from the reference, so
        // nothing about the support claim moves.
        assert_eq!(phi3.status, "active_validation_blocked_parity");
        assert_eq!(phi3.parity_audited, "failed");
        assert_eq!(phi3.latest_checked_result, "blocked_generation_parity");
        assert!(phi3.frontend_readiness_gate.contains("fail-closed"));
        // Prompt-token parity now passes and must be stated as passing...
        assert!(phi3.evidence.contains("prompt-token parity PASSES"));
        assert!(phi3.evidence.contains("all_match=true"));
        assert!(!phi3.tokenizer_works.starts_with("blocked"));
        // ...the temperature-0 non-determinism must be stated as FIXED (the Metal PV
        // row-tail guard landed)...
        assert!(phi3
            .evidence
            .contains("NON-DETERMINISM that previously blocked this row is"));
        assert!(phi3
            .evidence
            .contains("nondeterminism-fixed-q8-20260728.json"));
        // ...and generation parity must still be stated as blocking, for the separate
        // DETERMINISTIC prefill-vs-decode divergence. This is the same claim an earlier
        // receipt made and had to retract -- it was then inferred from two samples of a
        // non-deterministic engine. It is assertable here only because it is now
        // repeatable (3/3 identical runs), which is what makes it evidence.
        assert!(phi3
            .evidence
            .contains("Generation parity STILL blocks the row"));
        assert!(phi3.evidence.contains("diverging at index 2"));
        assert!(phi3
            .full_support_status
            .contains("blocked_generation_parity"));

        // The held Phi-4 artifact carries a tokenizer admission gate but no support
        // claim, so it must never surface as an installable catalog row.
        assert!(
            !curated_catalog()
                .iter()
                .any(|item| item.filename == "Phi-4-mini-instruct-Q4_K_M.gguf"),
            "held Phi-4 artifact must not become a user-facing catalog row"
        );

        let phi4 = response
            .model_compatibility
            .iter()
            .find(|target| target.id == "phi4_mini_instruct_q4_k_m")
            .expect("Phi-4 exact-row HOLD should stay visible");
        assert_eq!(phi4.status, "active_validation_blocked_parity");
        assert_eq!(phi4.parity_audited, "failed");
        assert_eq!(
            phi4.latest_checked_output,
            "qa/muster/HOLD-phi4-mini-instruct-q4_k_m.json"
        );
        assert!(phi4.frontend_readiness_gate.contains("fail-closed"));
        assert!(phi4
            .evidence
            .contains("intentionally absent from curated_catalog"));
    }

    #[test]
    fn llama_server_props_template_caps_fail_closed_without_loaded_template() {
        let caps = llama_server_chat_template_caps(None);

        assert_eq!(caps["available"], false);
        assert_eq!(caps["requires_loaded_model"], true);
        assert!(caps["source"].is_null());
        assert!(caps["supported_operations"].as_array().unwrap().is_empty());
        assert!(caps["unsupported"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("no_loaded_supported_chat_template")));
        assert!(caps["unsupported"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("full_llama_server_template_parity")));
    }

    #[test]
    fn llama_server_props_template_caps_expose_loaded_template_without_path() {
        let model = LoadedModel {
            id: "loaded-test".to_string(),
            path: PathBuf::from("/private/models/Loaded-Test.gguf"),
            gguf: GgufFile {
                path: PathBuf::from("/private/models/Loaded-Test.gguf"),
                version: 3,
                tensor_count: 0,
                metadata_count: 0,
                alignment: 32,
                data_start_offset: 0,
                metadata: BTreeMap::new(),
                tensors: Vec::new(),
            },
            llama_config: None,
            llama_tensors: None,
            unsupported_runtime: None,
            tokenizer: TokenizerLoadState::Available(TokenizerSummary {
                model: "llama-bpe",
                token_count: 128,
                byte_token_count: 0,
                special: SpecialTokenSummary {
                    bos: None,
                    eos: None,
                    eot: None,
                    eom: None,
                    unk: None,
                    sep: None,
                    pad: None,
                    mask: None,
                    eog: Vec::new(),
                },
                config: TokenizerConfigSummary {
                    add_bos: false,
                    add_eos: false,
                    add_sep: false,
                    add_space_prefix: false,
                    remove_extra_whitespaces: false,
                },
                chat_template: Some(ChatTemplateSummary {
                    source: "tokenizer.chat_template",
                    detected_format: "llama3_header",
                    length: 42,
                }),
            }),
            tokenizer_runtime: None,
            lane: LaneIdentity {
                model_id: "loaded-test".to_string(),
                gguf_sha256: "ab".repeat(32),
                gguf_filename: "Loaded-Test.gguf".to_string(),
                quantization: "unknown".to_string(),
                architecture: "unknown".to_string(),
                tokenizer_kind: "unknown".to_string(),
                tokenizer_sha256: None,
                camelid_version: receipt::camelid_version(),
                camelid_commit: receipt::camelid_commit(),
            },
        };

        let caps = llama_server_chat_template_caps(Some(&model));
        let serialized = caps.to_string();

        assert_eq!(caps["available"], true);
        assert_eq!(caps["requires_loaded_model"], true);
        assert_eq!(caps["source"], "tokenizer.chat_template");
        assert_eq!(caps["detected_format"], "llama3_header");
        assert_eq!(caps["length"], 42);
        assert!(caps["supported_operations"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("render_prompt")));
        assert!(caps["unsupported"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("full_llama_server_template_parity")));
        assert!(!serialized.contains("/private"));
        assert!(!serialized.contains("Loaded-Test.gguf"));
    }

    #[test]
    fn capabilities_describe_props_template_caps_without_broad_completion_claim() {
        let response = capabilities_response();
        let props = response
            .api_features
            .iter()
            .find(|feature| feature.id == "llama_server_props")
            .expect("llama-server props feature should stay advertised");

        assert_eq!(props.status, "partial");
        assert!(props
            .notes
            .contains("explicit fail-closed chat_template_caps"));
        assert!(props.notes.contains("native /completion streaming"));
        assert!(!props
            .notes
            .contains("does not imply slot lifecycle, /completion,"));
    }

    #[test]
    fn selected_logit_diagnostics_include_rank_outside_top_count() {
        let logits =
            CpuTensor::from_f32("logits", vec![1, 5], vec![0.1, 0.5, 0.4, -1.0, 0.3]).unwrap();

        let diagnostics = top_logit_diagnostics(&logits, 2, &[3]).unwrap();

        assert_eq!(diagnostics.len(), 3);
        assert_eq!(diagnostics[0].token_id, 1);
        assert_eq!(diagnostics[0].rank, 1);
        assert!(!diagnostics[0].selected);
        assert!(diagnostics[0].probability > diagnostics[1].probability);
        assert_eq!(diagnostics[1].token_id, 2);
        assert_eq!(diagnostics[1].rank, 2);
        assert_eq!(diagnostics[2].token_id, 3);
        assert_eq!(diagnostics[2].rank, 5);
        assert!(diagnostics[2].probability > 0.0);
        assert!(diagnostics[2].probability < diagnostics[1].probability);
        assert!(diagnostics[2].selected);
    }

    #[test]
    fn timeout_payload_is_scrubbed_and_records_trace_fields() {
        let payload = generation_timeout_error_json(
            Duration::from_millis(7),
            Duration::from_millis(11),
            Some(3),
        );

        let error = &payload["error"];
        assert_eq!(error["code"], "generation_timeout");
        assert_eq!(error["param"], "max_tokens");
        assert_eq!(error["timeout_trace"]["timeout_ms"], 7);
        assert_eq!(error["timeout_trace"]["elapsed_ms"], 11);
        assert_eq!(error["timeout_trace"]["generated_tokens"], 3);
        assert_eq!(
            error["timeout_trace"]["timeout_env"],
            GENERATION_TIMEOUT_ENV
        );
        let serialized = payload.to_string();
        assert!(!serialized.contains("://"));
        assert!(!serialized.contains(&["/Users", "/"].concat()));
        assert!(!serialized.contains("models/"));
    }

    /// D3: prep runs OUTSIDE serialization — a request's validation and
    /// tokenization complete while another decode occupies the engine, never
    /// queued behind it. (Here prep returns a validation error since no model
    /// is loaded; the point is that it RETURNS while the engine is busy.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prep_is_not_serialized_behind_a_running_decode() {
        let _env_guard = crate::test_support::env_lock();
        drain_decode_probe().await;
        decode_probe::reset();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "60000");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "1500");

        let state = AppState::default();
        let state_for_a = state.clone();
        let handler_a = tokio::spawn(async move {
            let prepared = orphan_test_prepared("model-a.gguf");
            let _ = generate_decoded_tokens_blocking(&state_for_a, prepared).await;
        });
        let started = Instant::now();
        while decode_probe::active() == 0 {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "request A's decode never reached the engine",
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let request: GenerationSessionRequest =
            serde_json::from_value(serde_json::json!({ "prompt": "hello" }))
                .expect("minimal request deserializes");
        let prep_started = Instant::now();
        let result = prepare_generation(&state, request).await;
        let prep_elapsed = prep_started.elapsed();
        assert!(result.is_err(), "no model is loaded; prep must error");
        assert!(
            prep_elapsed < Duration::from_millis(500),
            "prep stalled {}ms behind a running decode (D3 regression)",
            prep_elapsed.as_millis(),
        );

        let _ = handler_a.await;
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        drain_decode_probe().await;
        std::env::remove_var(GENERATION_TIMEOUT_ENV);
    }

    /// The streaming wall-clock budget is enforced BETWEEN steps by the
    /// engine-side stream job (the old per-step `tokio::time::timeout` was
    /// itself a detach hazard and is gone). The timeout event must carry the
    /// exact pre-inversion payload: request timeout, elapsed, and the count
    /// of tokens generated so far.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stream_job_timeout_reports_generated_count() {
        let _env_guard = crate::test_support::env_lock();
        drain_decode_probe().await;
        std::env::set_var(GENERATION_TIMEOUT_ENV, "1");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "25");

        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(32);
        let prepared = orphan_test_prepared("model-a.gguf");
        let job = tokio::task::spawn_blocking(move || {
            run_stream_decode_job(prepared, events_tx);
        });
        let mut saw_timeout = None;
        while let Some(event) = events_rx.recv().await {
            if let StreamDecodeEvent::TimedOut {
                timeout,
                generated_tokens,
                ..
            } = event
            {
                saw_timeout = Some((timeout, generated_tokens));
            }
        }
        job.await.unwrap();

        std::env::remove_var(GENERATION_TIMEOUT_ENV);
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        let (timeout, generated_tokens) =
            saw_timeout.expect("stream job should emit TimedOut before any step");
        assert_eq!(timeout.as_millis(), 1);
        assert_eq!(generated_tokens, 0, "budget expired before the first step");
    }

    #[tokio::test]
    async fn non_streaming_generation_timeout_returns_scrubbed_trace() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "1");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "25");
        let session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);

        let state = AppState::default();
        let result = generate_decoded_tokens_blocking(&state, prepared).await;

        std::env::remove_var(GENERATION_TIMEOUT_ENV);
        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        let response = match result {
            Err(response) => *response,
            Ok(_) => panic!("expected non-streaming generation timeout"),
        };
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["error"]["code"], "generation_timeout");
        assert_eq!(body["error"]["param"], "max_tokens");
        assert_eq!(body["error"]["timeout_trace"]["timeout_ms"], 1);
        assert_eq!(
            body["error"]["timeout_trace"]["timeout_env"],
            GENERATION_TIMEOUT_ENV
        );
        assert!(body["error"]["timeout_trace"]["generated_tokens"].is_null());
        let serialized = body.to_string();
        assert!(!serialized.contains("://"));
        assert!(!serialized.contains(&["/Users", "/"].concat()));
        assert!(!serialized.contains("models/"));
    }

    #[test]
    fn cpu_weight_materialization_estimate_defaults_q8_linears_to_file_backed() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
        let binding = materialization_binding(false, GgufTensorType::Q8_0, vec![32, 2]);

        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();

        assert_eq!(estimated, 0);
    }

    #[test]
    fn q8_runtime_health_reports_forced_lazy_before_retain() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "1");
        std::env::set_var(RETAIN_Q8_BLOCKS_ENV, "1");

        let health = q8_runtime_health();

        assert_eq!(health.policy, "forced_lazy_file_backed_q8");
        assert!(health.lazy_q8_linear);
        assert!(health.retain_q8_blocks);
        assert!(health.note.contains("do not override"));
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn q8_runtime_health_reports_default_auto_retain_candidate_policy() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "64 MiB");

        let health = q8_runtime_health();

        assert_eq!(health.policy, "lazy_q8_linear_default_or_auto_retain");
        assert!(health.lazy_q8_linear);
        assert!(!health.retain_q8_blocks);
        assert_eq!(health.file_cache_bytes, Some(64 * 1024 * 1024));
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_runtime_health_reports_eager_retained_duplicate_policy() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "0");
        std::env::set_var(RETAIN_Q8_BLOCKS_ENV, "1");

        let health = q8_runtime_health();

        assert_eq!(health.policy, "eager_f32_with_retained_q8_blocks");
        assert!(!health.lazy_q8_linear);
        assert!(health.retain_q8_blocks);
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn cpu_weight_materialization_estimate_can_count_q8_f32_when_lazy_disabled() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "0");
        let binding = materialization_binding(false, GgufTensorType::Q8_0, vec![32, 2]);

        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();
        let expected_per_tensor = 32 * 2 * 4;
        let expected_tensor_count = 3 + 9;

        assert_eq!(estimated, expected_per_tensor * expected_tensor_count);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn cpu_weight_materialization_estimate_counts_opt_in_q8_retained_blocks() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "0");
        std::env::set_var(RETAIN_Q8_BLOCKS_ENV, "1");
        let binding = materialization_binding(false, GgufTensorType::Q8_0, vec![32, 2]);

        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();
        let expected_per_tensor = (32 * 2 * 4) + (2 * mem::size_of::<Q8_0Block>() as u64);
        let expected_tensor_count = 3 + 9;

        assert_eq!(estimated, expected_per_tensor * expected_tensor_count);
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn cpu_weight_materialization_estimate_counts_lazy_q8_linears_as_file_backed() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "1");
        let binding = materialization_binding(false, GgufTensorType::Q8_0, vec![32, 2]);

        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();

        assert_eq!(estimated, 0);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn cpu_weight_materialization_estimate_skips_tied_output_tensor() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(RETAIN_Q8_BLOCKS_ENV);
        std::env::set_var(LAZY_Q8_LINEAR_ENV, "0");
        let binding = materialization_binding(true, GgufTensorType::Q8_0, vec![32, 2]);

        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();
        let expected_per_tensor = 32 * 2 * 4;
        let expected_tensor_count = 2 + 9;

        assert_eq!(estimated, expected_per_tensor * expected_tensor_count);
        std::env::remove_var(LAZY_Q8_LINEAR_ENV);
    }

    #[test]
    fn cpu_weight_materialization_budget_allows_exact_limit() {
        let _env_guard = crate::test_support::env_lock();
        let binding = materialization_binding(false, GgufTensorType::F32, vec![4, 4]);
        let estimated = estimate_cpu_weight_materialization_bytes(&binding).unwrap();
        std::env::set_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV, estimated.to_string());

        assert_eq!(
            guard_cpu_weight_materialization_budget(&binding).unwrap(),
            estimated
        );
        std::env::remove_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV);
    }

    #[test]
    fn context_budget_counts_exact_prompt_and_generation_allowance() {
        assert!(enforce_context_budget(1_536, 512, 2_048).is_ok());
        let error = enforce_context_budget(1_537, 512, 2_048).unwrap_err();
        assert!(error.contains("exact rendered prompt (1537 tokens)"));
        assert!(error.contains("generation allowance (512 tokens)"));
    }

    #[test]
    fn cpu_weight_materialization_budget_rejects_invalid_env_limit() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV, "not-a-byte-count");
        let binding = materialization_binding(false, GgufTensorType::F32, vec![4, 4]);

        let err = guard_cpu_weight_materialization_budget(&binding)
            .expect_err("invalid materialization budget env should fail closed")
            .to_string();

        assert!(err.contains("invalid CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES"));
        std::env::remove_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV);
    }

    #[test]
    fn binding_all_resident_quant_linears_accepts_kquant_and_q8_dense() {
        // Q4_K / Q6_K / Q8_0 dense bindings load wire-only and run on the resident
        // GPU engine, so they must classify as resident-eligible (the f32 guard's
        // estimate never materializes for them).
        for ty in [
            GgufTensorType::Q8_0,
            GgufTensorType::Q4K,
            GgufTensorType::Q6K,
        ] {
            let binding = materialization_binding(false, ty, vec![256, 256]);
            assert!(
                binding_all_resident_quant_linears(&binding),
                "{ty:?} dense binding should classify as resident-eligible"
            );
        }
    }

    #[test]
    fn binding_all_resident_quant_linears_rejects_non_quant_linears() {
        // An f32/f16 linear is NOT resident-eligible â€” it takes the eager-f32 CPU path,
        // which the guard must keep protecting.
        for ty in [GgufTensorType::F32, GgufTensorType::F16] {
            let binding = materialization_binding(false, ty, vec![256, 256]);
            assert!(
                !binding_all_resident_quant_linears(&binding),
                "{ty:?} binding must not bypass the CPU materialization guard"
            );
        }
    }

    #[test]
    fn binding_all_resident_quant_linears_rejects_attention_biases() {
        let mut binding = materialization_binding(false, GgufTensorType::Q8_0, vec![256, 256]);
        let bias =
            |name: &str, width: u64| materialization_desc(name, GgufTensorType::F32, vec![width]);
        let LlamaAttentionTensors::Standard { biases, .. } = &mut binding.layers[0].attention
        else {
            unreachable!("materialization fixture uses standard attention");
        };
        *biases = Some(crate::model::LlamaAttentionBiasTensors {
            q: bias("blk.0.attn_q.bias", 256),
            k: bias("blk.0.attn_k.bias", 256),
            v: bias("blk.0.attn_v.bias", 256),
        });

        assert!(
            !binding_all_resident_quant_linears(&binding),
            "resident engines must not run a layer whose projection biases they cannot apply"
        );
    }

    #[test]
    fn cpu_weight_materialization_budget_blocks_unsafe_large_decode() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV, "1024");
        let binding = materialization_binding(false, GgufTensorType::F32, vec![64, 64]);

        let err = guard_cpu_weight_materialization_budget(&binding)
            .expect_err("oversized materialization should fail before eager decode")
            .to_string();

        assert!(err.contains("estimated CPU f32 weight materialization"));
        assert!(err.contains(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV));
        std::env::remove_var(CPU_WEIGHT_MATERIALIZATION_LIMIT_ENV);
    }

    #[test]
    fn cpu_weight_materialization_budget_bypassed_for_kquant_block_dot() {
        // The budget guard must treat a K-quant block-dot model as wire-only (no f32
        // materialization) on the CPU lane â€” otherwise serve CPU mode false-positives
        // on the f32 size it never produces (Phase 2 follow-up). Tested directly on
        // `binding_runs_on_cpu_wire_only` to avoid the `resident_decode_cuda_active`
        // GPU-bypass confound on CUDA builds.
        let _env_guard = crate::test_support::env_lock();
        let kquant = materialization_binding(false, GgufTensorType::Q4K, vec![256, 256]);
        let dense_f32 = materialization_binding(false, GgufTensorType::F32, vec![256, 256]);

        std::env::remove_var("CAMELID_X86_Q4K_DECODE"); // block-dot default-on
        assert!(
            binding_runs_on_cpu_wire_only(&kquant),
            "K-quant block-dot model is consumed wire-only on CPU"
        );
        assert!(
            !binding_runs_on_cpu_wire_only(&dense_f32),
            "an f32 model still materializes -> the guard must apply"
        );

        std::env::set_var("CAMELID_X86_Q4K_DECODE", "0"); // opt out: no CPU consumer
        assert!(
            !binding_runs_on_cpu_wire_only(&kquant),
            "with the block-dot disabled there is no wire-only CPU consumer"
        );

        std::env::remove_var("CAMELID_X86_Q4K_DECODE");
    }

    #[test]
    fn prompt_prefix_cache_reuses_exact_prompt_and_invalidates_key_changes() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var("CAMELID_ATTENTION_SCORE_SCALE");
        std::env::remove_var("CAMELID_GQA_HEAD_MAPPING");

        let config = tiny_config();
        let weights = tiny_weights();
        let mut session = LlamaInferenceSession::new(config.clone(), weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2],
                false,
                None,
            )
            .unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);

        assert!(lookup_prompt_prefix_cache(&prepared).is_none());
        store_prompt_prefix_cache(&prepared, &step);

        let match_res = lookup_prompt_prefix_cache(&prepared).expect("exact key cache hit");
        assert_eq!(match_res.cached.session.kv_cache.position, 2);
        assert_eq!(match_res.cached.logits, step.logits);
        assert_eq!(
            sample_cached_prompt_prefix(&match_res.cached, &[1, 2])
                .unwrap()
                .next_token_id,
            step.next_token_id
        );

        // An exact entry must beat a newer, longer entry with the same common
        // prefix. Otherwise the fast cached-logits path is needlessly degraded
        // to a rollback plus one-token prefill.
        let mut longer_session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
        let longer_step = longer_session
            .generate_next_token_with_history_diagnostics(
                &[1, 2, 0],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2, 0],
                false,
                None,
            )
            .unwrap();
        let mut longer = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2, 0], longer_session);
        longer.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        store_prompt_prefix_cache(&longer, &longer_step);
        let exact = lookup_prompt_prefix_cache(&prepared).expect("exact entry still wins");
        assert!(exact.is_exact_match);
        assert_eq!(exact.cached.token_ids, vec![1, 2]);

        let mut different_prompt = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![99, 98],
            match_res.cached.session.clone(),
        );
        different_prompt.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        assert!(lookup_prompt_prefix_cache(&different_prompt).is_none());

        let mut different_sampling = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![1, 2],
            match_res.cached.session.clone(),
        );
        different_sampling.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        different_sampling.sampling.temperature = 0.7;
        assert!(lookup_prompt_prefix_cache(&different_sampling).is_none());

        let mut different_model = prepared_for_cache(
            "tiny",
            "model-b.gguf",
            vec![1, 2],
            match_res.cached.session.clone(),
        );
        different_model.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        assert!(lookup_prompt_prefix_cache(&different_model).is_none());
    }

    #[test]
    fn prompt_prefix_cache_partial_match_and_lru_eviction() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(PROMPT_PREFIX_CACHE_MIN_TOKENS_ENV, "2");
        std::env::set_var("CAMELID_CUDA_RESIDENT_DECODE", "0");
        std::env::remove_var("CAMELID_DETERMINISTIC");

        let config = tiny_spec_config();
        let weights = tiny_weights();
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[0, 1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[0, 1, 2],
                false,
                None,
            )
            .unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![0, 1, 2], session);
        store_prompt_prefix_cache(&prepared, &step);

        let mut extended = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![0, 1, 2, 0, 1],
            prepared.session.clone(),
        );
        extended.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();

        let hit = lookup_prompt_prefix_cache(&extended).expect("partial prefix cache hit");
        assert!(!hit.is_exact_match);
        assert_eq!(hit.prefix_len, 3);

        let mut warm = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![0, 1, 0, 1],
            LlamaInferenceSession::new(tiny_spec_config(), tiny_weights()).unwrap(),
        );
        warm.max_tokens = 3;
        warm.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        let mut cold = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![0, 1, 0, 1],
            LlamaInferenceSession::new(tiny_spec_config(), tiny_weights()).unwrap(),
        );
        cold.max_tokens = 3;

        let warm_output = generate_token_ids(warm).expect("warm partial generation");
        let cold_output = generate_token_ids(cold).expect("cold generation");
        assert_eq!(warm_output.token_ids, cold_output.token_ids);
        assert!(warm_output.timings.prompt_cache_hit);
        assert!(
            warm_output.timings.prompt_evaluation.prefill.forward_total
                + warm_output
                    .timings
                    .prompt_evaluation
                    .first_token
                    .forward_total
                > 0.0,
            "partial hits must report the suffix prefill instead of zeroing prompt timing"
        );

        let pool_ref = prepared.cached_prompt_prefix.clone();
        {
            let mut pool = pool_ref.lock().unwrap();
            pool.clear();
            pool.capacity = 2;
        }

        let mut session1 = LlamaInferenceSession::new(tiny_spec_config(), tiny_weights()).unwrap();
        let step1 = session1
            .generate_next_token_with_history_diagnostics(
                &[0, 1],
                crate::inference::LlamaSampler::Greedy,
                &[0, 1],
                false,
                None,
            )
            .unwrap();
        let mut prep1 = prepared_for_cache("tiny", "model-a.gguf", vec![0, 1], session1);
        prep1.cached_prompt_prefix = pool_ref.clone();

        let mut session2 = LlamaInferenceSession::new(tiny_spec_config(), tiny_weights()).unwrap();
        let step2 = session2
            .generate_next_token_with_history_diagnostics(
                &[0, 2],
                crate::inference::LlamaSampler::Greedy,
                &[0, 2],
                false,
                None,
            )
            .unwrap();
        let mut prep2 = prepared_for_cache("tiny", "model-a.gguf", vec![0, 2], session2);
        prep2.cached_prompt_prefix = pool_ref.clone();

        let mut session3 = LlamaInferenceSession::new(tiny_spec_config(), tiny_weights()).unwrap();
        let step3 = session3
            .generate_next_token_with_history_diagnostics(
                &[1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2],
                false,
                None,
            )
            .unwrap();
        let mut prep3 = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session3);
        prep3.cached_prompt_prefix = pool_ref.clone();

        store_prompt_prefix_cache(&prep1, &step1);
        store_prompt_prefix_cache(&prep2, &step2);
        assert!(lookup_prompt_prefix_cache(&prep1).is_some());
        {
            let mut pool = pool_ref.lock().unwrap();
            let newest = std::time::Instant::now();
            for entry in &mut pool.entries {
                entry.last_used = if entry.cached.token_ids == prep1.token_ids {
                    newest
                } else {
                    newest - std::time::Duration::from_secs(1)
                };
            }
        }

        store_prompt_prefix_cache(&prep3, &step3);
        let remaining = pool_ref
            .lock()
            .unwrap()
            .entries
            .iter()
            .map(|entry| entry.cached.token_ids.clone())
            .collect::<Vec<_>>();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.contains(&prep1.token_ids));
        assert!(remaining.contains(&prep3.token_ids));
        assert!(!remaining.contains(&prep2.token_ids));

        std::env::remove_var(PROMPT_PREFIX_CACHE_MIN_TOKENS_ENV);
        std::env::remove_var("CAMELID_CUDA_RESIDENT_DECODE");
    }

    #[test]
    fn prompt_prefix_cache_defaults_to_one_session_and_rejects_invalid_capacity() {
        let _env_guard = crate::test_support::env_lock();
        std::env::remove_var(PROMPT_PREFIX_CACHE_CAPACITY_ENV);
        assert_eq!(
            PromptPrefixCachePool::from_env().capacity,
            DEFAULT_PROMPT_PREFIX_CACHE_CAPACITY
        );

        std::env::set_var(PROMPT_PREFIX_CACHE_CAPACITY_ENV, "3");
        assert_eq!(PromptPrefixCachePool::from_env().capacity, 3);

        for invalid in ["0", "not-a-number"] {
            std::env::set_var(PROMPT_PREFIX_CACHE_CAPACITY_ENV, invalid);
            assert_eq!(
                PromptPrefixCachePool::from_env().capacity,
                DEFAULT_PROMPT_PREFIX_CACHE_CAPACITY
            );
        }
        std::env::remove_var(PROMPT_PREFIX_CACHE_CAPACITY_ENV);
    }

    #[test]
    fn prompt_prefix_cache_rejects_session_position_mismatch() {
        let mut session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[0, 1],
                crate::inference::LlamaSampler::Greedy,
                &[0, 1],
                false,
                None,
            )
            .unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![0, 1, 2], session);
        store_prompt_prefix_cache(&prepared, &step);
        assert!(prepared
            .cached_prompt_prefix
            .lock()
            .unwrap()
            .entries
            .is_empty());
    }

    #[test]
    fn constrained_generation_skips_prompt_prefix_cache_lookup() {
        // B1 regression: the cache key is (model, path, tokens, sampling) -- the
        // constraint is NOT in the key. On a warm hit the first token would be
        // sampled from raw cached logits with no mask and the grammar state never
        // advanced over it, breaking the response_format guarantee on the second
        // identical request. Constrained requests must not read the cache.
        let config = tiny_config();
        let weights = tiny_weights();
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2],
                false,
                None,
            )
            .unwrap();
        let unconstrained = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);
        store_prompt_prefix_cache(&unconstrained, &step);
        assert!(
            lookup_prompt_prefix_cache(&unconstrained).is_some(),
            "control: the unconstrained twin hits the warm cache"
        );

        let mut constrained = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![1, 2],
            lookup_prompt_prefix_cache(&unconstrained)
                .unwrap()
                .cached
                .session
                .clone(),
        );
        constrained.cached_prompt_prefix = unconstrained.cached_prompt_prefix.clone();
        constrained.constraint = Some(crate::grammar::ConstraintSpec::json_object());
        assert!(
            lookup_prompt_prefix_cache(&constrained).is_none(),
            "a constrained request must never read the prompt-prefix cache"
        );
    }

    #[test]
    fn constrained_generation_does_not_store_prompt_prefix_cache() {
        // Mirror of the lookup test: skipping both directions keeps the invariant
        // one sentence -- the prompt-prefix cache never interacts with constrained
        // decoding.
        let config = tiny_config();
        let weights = tiny_weights();
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2],
                false,
                None,
            )
            .unwrap();
        let mut constrained = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);
        constrained.constraint = Some(crate::grammar::ConstraintSpec::json_object());
        store_prompt_prefix_cache(&constrained, &step);

        let mut unconstrained = prepared_for_cache(
            "tiny",
            "model-a.gguf",
            vec![1, 2],
            constrained.session.clone(),
        );
        unconstrained.cached_prompt_prefix = constrained.cached_prompt_prefix.clone();
        assert!(
            lookup_prompt_prefix_cache(&unconstrained).is_none(),
            "a constrained request must never write the prompt-prefix cache"
        );
    }

    #[test]
    fn cached_prompt_prefix_followed_by_longer_completion_keeps_generating() {
        let config = tiny_config();
        let weights = tiny_weights();
        let mut session = LlamaInferenceSession::new(config, weights).unwrap();
        let step = session
            .generate_next_token_with_history_diagnostics(
                &[1, 2],
                crate::inference::LlamaSampler::Greedy,
                &[1, 2],
                false,
                None,
            )
            .unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);
        store_prompt_prefix_cache(&prepared, &step);

        let generated = generate_token_ids(prepared).expect("cached generation should succeed");

        assert!(!generated.token_ids.is_empty());
    }

    #[test]
    fn renders_tinyllama_marker_prompt_with_eos_newline_and_assistant_prefix() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = Tokenizer {
            model: TokenizerModel::LlamaSpm,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: vec![
                Token {
                    id: 0,
                    text: "<unk>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Unknown,
                },
                Token {
                    id: 1,
                    text: "<s>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Control,
                },
                Token {
                    id: 2,
                    text: "</s>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Control,
                },
            ],
            token_to_id: HashMap::new(),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens {
                bos: Some(1),
                eos: Some(2),
                eog: BTreeSet::from([2]),
                ..SpecialTokens::default()
            },
            config: TokenizerConfig {
                add_bos: true,
                add_eos: false,
                add_sep: false,
                add_space_prefix: true,
                remove_extra_whitespaces: false,
            },
            chat_template: Some("<|user|><|assistant|><|system|>".to_string()),
        };

        assert_eq!(
            render_chat_prompt(
                &[ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
                &tokenizer,
            ),
            "<|user|>\nhello</s>\n<|assistant|>\n"
        );
    }
    #[test]
    fn renders_llama3_instruct_prompt_with_header_tokens_and_special_parsing() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = Tokenizer {
            model: TokenizerModel::Gpt2Bpe,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens::default(),
            config: TokenizerConfig {
                add_bos: true,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: Some(
                "<|start_header_id|>{{ role }}<|end_header_id|>{{ content }}<|eot_id|>".to_string(),
            ),
        };

        assert!(tokenizer.chat_prompt_parse_special());
        assert_eq!(
            detect_chat_template_format(tokenizer.chat_template.as_deref().unwrap()),
            "llama3_instruct"
        );
        assert_eq!(
            render_chat_prompt(
                &[ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " hello ".to_string(),
                }],
                &tokenizer,
            ),
            "<|start_header_id|>user<|end_header_id|>\n\n hello <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn renders_llama3_instruct_prompt_with_system_and_multi_turn_messages() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(
            render_chat_prompt(
                &[
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "system".to_string(),
                        content: "Answer briefly.".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: "Say alpha.".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "assistant".to_string(),
                        content: "alpha".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: "Now say beta.".to_string(),
                    },
                ],
                &tokenizer,
            ),
            "<|start_header_id|>system<|end_header_id|>\n\nAnswer briefly.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSay alpha.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nNow say beta.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn renders_llama3_instruct_prompt_without_generation_header_after_assistant_final() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(
            render_chat_prompt(
                &[
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: "Complete cam".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "assistant".to_string(),
                        content: "elid".to_string(),
                    },
                ],
                &tokenizer,
            ),
            "<|start_header_id|>user<|end_header_id|>\n\nComplete cam<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nelid<|eot_id|>"
        );
    }

    #[test]
    fn renders_llama3_instruct_prompt_preserving_multiline_content() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(
            render_chat_prompt(
                &[ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "Line one.\n\n  Indented line two.  ".to_string(),
                }],
                &tokenizer,
            ),
            "<|start_header_id|>user<|end_header_id|>\n\nLine one.\n\n  Indented line two.  <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn renders_mistral_instruct_prompt_with_system_preamble() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        assert_eq!(
            detect_chat_template_format(tokenizer.chat_template.as_deref().unwrap()),
            "mistral_instruct"
        );
        assert_eq!(
            render_chat_prompt(
                &[
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "system".to_string(),
                        content: " Be brief. ".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: " Hello there. ".to_string(),
                    },
                ],
                &tokenizer,
            ),
            "<s>[INST] Be brief.\n\nHello there. [/INST]"
        );
    }

    #[test]
    fn mistral_instruct_renderer_avoids_duplicate_bos_for_tokenization() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "Hello there.".to_string(),
            }],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(rendered.text, "<s>[INST] Hello there. [/INST]");
    }

    #[test]
    fn renders_mistral_instruct_prompt_with_completed_assistant_turn() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        assert_eq!(
            render_chat_prompt(
                &[
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: "Complete cam".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "assistant".to_string(),
                        content: " elid ".to_string(),
                    },
                    ChatMessage {
                        unsupported_content_parts: Vec::new(),
                        role: "user".to_string(),
                        content: "Now say hi".to_string(),
                    },
                ],
                &tokenizer,
            ),
            "<s>[INST] Complete cam [/INST] elid</s><s>[INST] Now say hi [/INST]"
        );
    }

    #[test]
    fn mistral_instruct_with_tools_injects_available_tools() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"]
                }
            }
        })];

        let messages = vec![ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "Read notes.txt".to_string(),
        }];

        let rendered = render_mistral_instruct_prompt_with_tools(&messages, &tokenizer, &tools);

        assert!(rendered.starts_with("<s>[AVAILABLE_TOOLS] "));
        assert!(rendered.contains("[/AVAILABLE_TOOLS]"));
        assert!(rendered.contains("[INST] Read notes.txt [/INST]"));
        assert!(
            rendered.contains("\"name\":\"read_file\"")
                || rendered.contains("\"name\": \"read_file\"")
        );
    }

    #[test]
    fn mistral_instruct_with_tools_omits_system_message() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "test", "description": "Test tool", "parameters": {}}
        })];

        let messages = vec![
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "Do something".to_string(),
            },
        ];

        let rendered = render_mistral_instruct_prompt_with_tools(&messages, &tokenizer, &tools);

        // System content is discarded when [AVAILABLE_TOOLS] is active
        assert!(!rendered.contains("You are helpful."));
        assert!(rendered.contains("[INST] Do something [/INST]"));
        assert!(rendered.starts_with("<s>[AVAILABLE_TOOLS] "));
    }

    #[test]
    fn mistral_instruct_with_tools_renders_tool_results() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = mistral_test_tokenizer();

        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "read_file", "description": "Read", "parameters": {}}
        })];

        let messages = vec![
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "Read it".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "assistant".to_string(),
                content: "read_file({\"path\":\"notes.txt\"})".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "tool".to_string(),
                content: "alpha\nbeta\ngamma".to_string(),
            },
            ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "How many lines?".to_string(),
            },
        ];

        let rendered = render_mistral_instruct_prompt_with_tools(&messages, &tokenizer, &tools);

        assert!(rendered.contains("[TOOL_CALLS] "));
        assert!(
            rendered.contains("\"name\":\"read_file\"")
                || rendered.contains("\"name\": \"read_file\"")
        );
        assert!(rendered.contains("[TOOL_RESULTS] "));
        assert!(rendered.contains("call_id"));
        assert!(rendered.contains("[/TOOL_RESULTS]"));
        assert!(rendered.contains("[INST] How many lines? [/INST]"));
    }

    #[test]
    fn keeps_compact_llama3_renderer_by_default_for_metadata_templates() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_metadata_subset_test_tokenizer();

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
        );

        assert!(rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|start_header_id|>user<|end_header_id|>\n\n  hello  <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn can_opt_into_llama3_metadata_jinja_renderer_without_duplicate_bos() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_metadata_subset_test_tokenizer();

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_matches_llama3_gguf_template_shape() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "  hello  ".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_executes_full_llama32_gguf_template_system_user() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_FULL_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "  hello  ".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn exact_llama32_1b_row_executes_full_metadata_jinja_template_without_env_opt_in() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_FULL_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Beta? ".to_string(),
                },
            ],
            &tokenizer,
            Some("bartowski/Meta-Llama-3.2-1B-Instruct-Q8_0"),
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAlpha?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBeta?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn exact_llama32_3b_row_executes_full_metadata_jinja_template_without_env_opt_in() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_FULL_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Beta? ".to_string(),
                },
            ],
            &tokenizer,
            Some("bartowski/Meta-Llama-3.2-3B-Instruct-Q8_0"),
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAlpha?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBeta?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    /// TOOLCALL_DIAG: prints how the Llama 3 template renders OpenAI-nested vs
    /// flat function tools, so the rendered prompt is inspectable (run with
    /// `--nocapture`). Asserts the flat form puts `"name"`/`"parameters"` at the
    /// tool's top level (matching the response format the template requests).
    #[test]
    fn tool_render_nested_vs_flat_diagnostic() {
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_FULL_TEMPLATE);
        let user = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "read notes.txt".to_string(),
        }];
        let nested = serde_json::json!([{
            "type":"function",
            "function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}
        }]);
        let flat = serde_json::json!([{
            "name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
        }]);
        let n = nested.as_array().unwrap();
        let f = flat.as_array().unwrap();
        let r_nested =
            render_jinja_chat_template(&user, &tokenizer, LLAMA3_METADATA_FULL_TEMPLATE, Some(n))
                .unwrap();
        let r_flat =
            render_jinja_chat_template(&user, &tokenizer, LLAMA3_METADATA_FULL_TEMPLATE, Some(f))
                .unwrap();
        println!("\n===== NESTED (OpenAI) tools =====\n{r_nested}\n===== FLAT (function) tools =====\n{r_flat}\n=====");
        assert!(r_nested.contains("\"type\": \"function\""));
        assert!(!r_flat.contains("\"type\": \"function\""));
        assert!(r_flat.contains("\"name\": \"read_file\""));
        assert!(r_flat.contains("\"parameters\""));
    }

    #[test]
    fn exact_llama32_1b_row_uses_metadata_jinja_renderer_without_env_opt_in_system_user() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "  hello  ".to_string(),
                },
            ],
            &tokenizer,
            Some("llama32_1b_instruct_q8_0"),
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn exact_llama32_1b_row_uses_metadata_jinja_renderer_without_env_opt_in_user_only() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
        );

        assert!(!rendered.add_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn exact_llama32_1b_row_uses_metadata_jinja_renderer_without_env_opt_in_multi_turn() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: " Answer tersely. ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Beta? ".to_string(),
                },
            ],
            &tokenizer,
            Some("bartowski/Meta-Llama-3.2-1B-Instruct-Q8_0"),
        );

        assert!(!rendered.add_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnswer tersely.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAlpha?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBeta?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn exact_llama32_1b_row_uses_metadata_jinja_renderer_without_env_opt_in_assistant_final() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "Complete cam".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " elid ".to_string(),
                },
            ],
            &tokenizer,
            Some("llama_3.2_1b_instruct_q8_0"),
        );

        assert!(!rendered.add_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nComplete cam<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nelid<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama32_non_q8_or_untracked_row_keeps_compact_renderer_without_env_opt_in() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("llama32_3b_instruct"),
        );

        assert!(rendered.add_special);
        assert_eq!(
            rendered.text,
            "<|start_header_id|>user<|end_header_id|>\n\n  hello  <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn llama32_1b_non_q8_model_id_keeps_compact_renderer_without_env_opt_in() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("Meta-Llama-3.2-1B-Instruct"),
        );

        assert!(rendered.add_special);
        assert_eq!(
            rendered.text,
            "<|start_header_id|>user<|end_header_id|>\n\n  hello  <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn metadata_jinja_renderer_honors_add_generation_prompt_after_assistant_turn() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "Complete cam".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " elid ".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nComplete cam<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nelid<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_executes_templates_without_bos_token_expression() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ message['content'] }}<|eot_id|>{% endfor %}",
        );

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
        );

        assert!(rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|start_header_id|>user<|end_header_id|>  hello  <|eot_id|>"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_handles_multi_turn_chat_template_parity_shape() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: " Answer tersely. ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: " Beta? ".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(!rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(
            rendered.text,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nAnswer tersely.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAlpha?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nalpha<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBeta?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_executes_simple_loop_templates() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}{{ message['role'] }}={{ message['content'] }}\n{% endfor %}",
        );

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "system".to_string(),
                    content: "Be brief.".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(rendered.text, "system=Be brief.\nuser=hello\n");
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_supports_dot_access_message_fields() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}{{ loop.index0 }}:{{ message.role }}={{ message.content }}\n{% endfor %}",
        );

        let rendered = render_chat_prompt_for_tokenization(
            &[
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: " system ".to_string(),
                    content: "Be brief.".to_string(),
                },
                ChatMessage {
                    unsupported_content_parts: Vec::new(),
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ],
            &tokenizer,
        );

        assert!(rendered.add_special);
        assert!(rendered.parse_special);
        assert_eq!(rendered.text, "0:system=Be brief.\n1:user=hello\n");
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn metadata_jinja_renderer_reuses_compiled_template_environment() {
        let _guard = crate::test_support::env_lock();
        clear_jinja_chat_template_environment_cache();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}{{ message['role'] }}={{ message['content'] }}\n{% endfor %}",
        );
        let messages = [ChatMessage {
            unsupported_content_parts: Vec::new(),
            role: "user".to_string(),
            content: "hello".to_string(),
        }];

        let first = render_chat_prompt_for_tokenization(&messages, &tokenizer);
        let second = render_chat_prompt_for_tokenization(&messages, &tokenizer);

        assert_eq!(first, second);
        assert_eq!(jinja_chat_template_environment_cache_len(), 1);
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        clear_jinja_chat_template_environment_cache();
    }

    #[test]
    fn metadata_jinja_renderer_reports_raise_exception_as_unsupported() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let template = "{{ raise_exception('unsupported chat-template branch') }}";
        let tokenizer = llama3_tokenizer_with_template(template);

        let err = render_metadata_jinja_chat_template_prompt(&[], &tokenizer, template, None)
            .unwrap_err();
        assert!(err.to_string().contains("unsupported chat-template branch"));

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
        );
        assert_eq!(rendered.text, "user: hello\n");
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn exact_llama32_1b_required_metadata_jinja_renderer_does_not_silently_fallback() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let template = "{{ raise_exception('unsupported exact-row chat-template branch') }}<|start_header_id|><|end_header_id|><|eot_id|>";
        let tokenizer = llama3_tokenizer_with_template(template);

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("unsupported exact-row chat-template branch"));
    }

    #[test]
    fn exact_llama32_3b_required_metadata_jinja_renderer_does_not_silently_fallback() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template("{{ unsupported_call(messages) }}");

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("llama32_3b_instruct_q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(
            err.to_string()
                .contains("exact Llama 3.2 3B Instruct Q8_0 requires a recognized Llama 3 metadata chat_template"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn exact_llama32_1b_required_renderer_rejects_unrecognized_template_shape() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
        );

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(err
            .to_string()
            .contains("requires a recognized Llama 3 metadata chat_template"));
    }

    #[test]
    fn exact_llama32_3b_required_renderer_rejects_unrecognized_template_shape() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
        );

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-3B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(err.to_string().contains(
            "exact Llama 3.2 3B Instruct Q8_0 requires a recognized Llama 3 metadata chat_template"
        ));
    }

    #[test]
    fn exact_llama32_1b_required_renderer_rejects_missing_template_metadata() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = Tokenizer {
            chat_template: None,
            ..llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE)
        };

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("llama32_1b_instruct_q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(err
            .to_string()
            .contains("requires tokenizer.chat_template metadata"));
    }

    #[test]
    fn exact_llama32_3b_required_renderer_rejects_missing_template_metadata() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = Tokenizer {
            chat_template: None,
            ..llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE)
        };

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("Meta-Llama-3.2-3B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::InvalidOperation);
        assert!(
            err.to_string().contains(
                "exact Llama 3.2 3B Instruct Q8_0 requires tokenizer.chat_template metadata"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn metadata_jinja_renderer_reports_undefined_variables_as_unsupported() {
        let _guard = crate::test_support::env_lock();
        std::env::set_var(METADATA_CHAT_TEMPLATE_ENV, "metadata");
        let template = "{{ unsupported_template_variable }}";
        let tokenizer = llama3_tokenizer_with_template(template);

        let err = render_metadata_jinja_chat_template_prompt(&[], &tokenizer, template, None)
            .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::UndefinedError);
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
    }

    #[test]
    fn exact_llama32_1b_required_metadata_jinja_renderer_reports_undefined_variables() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let template =
            "{{ unsupported_template_variable }}<|start_header_id|><|end_header_id|><|eot_id|>";
        let tokenizer = llama3_tokenizer_with_template(template);

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::UndefinedError);
    }

    #[test]
    fn exact_llama32_3b_required_metadata_jinja_renderer_reports_undefined_variables() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let template =
            "{{ unsupported_template_variable }}<|start_header_id|><|end_header_id|><|eot_id|>";
        let tokenizer = llama3_tokenizer_with_template(template);

        let err = render_chat_prompt_for_tokenization_for_model_result(
            &[ChatMessage {
                unsupported_content_parts: Vec::new(),
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Meta-Llama-3.2-3B-Instruct-Q8_0"),
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), MiniJinjaErrorKind::UndefinedError);
    }

    fn llama3_test_tokenizer() -> Tokenizer {
        llama3_tokenizer_with_template(
            "<|start_header_id|>{{ role }}<|end_header_id|>{{ content }}<|eot_id|>",
        )
    }

    fn mistral_test_tokenizer() -> Tokenizer {
        Tokenizer {
            model: TokenizerModel::LlamaSpm,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: vec![
                Token {
                    id: 0,
                    text: "<unk>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Unknown,
                },
                Token {
                    id: 1,
                    text: "<s>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Control,
                },
                Token {
                    id: 2,
                    text: "</s>".to_string(),
                    score: 0.0,
                    kind: TokenKind::Control,
                },
            ],
            token_to_id: HashMap::from([
                ("<unk>".to_string(), 0),
                ("<s>".to_string(), 1),
                ("</s>".to_string(), 2),
            ]),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens {
                bos: Some(1),
                eos: Some(2),
                eog: BTreeSet::from([2]),
                ..SpecialTokens::default()
            },
            config: TokenizerConfig {
                add_bos: true,
                add_eos: false,
                add_sep: false,
                add_space_prefix: true,
                remove_extra_whitespaces: false,
            },
            chat_template: Some(
                "{{ bos_token }}[INST] {{ messages[0]['content'] }} [/INST]".to_string(),
            ),
        }
    }

    fn llama3_metadata_subset_test_tokenizer() -> Tokenizer {
        llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE)
    }

    const LLAMA3_METADATA_SUBSET_TEMPLATE: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

    const LLAMA3_METADATA_FULL_TEMPLATE: &str = r#"{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;

    fn llama3_tokenizer_with_template(template: &str) -> Tokenizer {
        Tokenizer {
            model: TokenizerModel::Gpt2Bpe,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: vec![Token {
                id: 0,
                text: "<|begin_of_text|>".to_string(),
                score: 0.0,
                kind: TokenKind::Control,
            }],
            token_to_id: HashMap::from([("<|begin_of_text|>".to_string(), 0)]),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens {
                bos: Some(0),
                ..SpecialTokens::default()
            },
            config: TokenizerConfig {
                add_bos: true,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: Some(template.to_string()),
        }
    }

    fn materialization_binding(
        tied_output: bool,
        tensor_type: GgufTensorType,
        dimensions: Vec<u64>,
    ) -> LlamaTensorBinding {
        let desc = |name: &str| materialization_desc(name, tensor_type, dimensions.clone());
        LlamaTensorBinding {
            token_embedding: desc("token_embd.weight"),
            output_norm: desc("output_norm.weight"),
            output: desc("output.weight"),
            output_is_tied_embedding: tied_output,
            rope_freqs: None,
            mla_metadata: None,
            attention_head_count: 1,
            hidden_size: dimensions.first().copied().unwrap_or(1) as usize,
            layers: vec![crate::model::LlamaLayerTensors {
                attention_norm: desc("blk.0.attn_norm.weight"),
                attention: LlamaAttentionTensors::Standard {
                    q: desc("blk.0.attn_q.weight"),
                    k: desc("blk.0.attn_k.weight"),
                    v: desc("blk.0.attn_v.weight"),
                    biases: None,
                    q_norm: None,
                    k_norm: None,
                },
                attention_output: desc("blk.0.attn_output.weight"),
                ffn_norm: desc("blk.0.ffn_norm.weight"),
                ffn: LlamaFfnTensors::Dense {
                    gate: desc("blk.0.ffn_gate.weight"),
                    up: desc("blk.0.ffn_up.weight"),
                    down: desc("blk.0.ffn_down.weight"),
                },
            }],
        }
    }

    fn materialization_desc(
        name: &str,
        tensor_type: GgufTensorType,
        dimensions: Vec<u64>,
    ) -> GgufTensorDescriptor {
        let element_count = dimensions.iter().product::<u64>();
        let n_bytes = match tensor_type {
            GgufTensorType::Q8_0 => element_count.div_ceil(32) * 34,
            GgufTensorType::F32 => element_count * 4,
            GgufTensorType::F16 | GgufTensorType::BF16 => element_count * 2,
            _ => element_count,
        };
        GgufTensorDescriptor {
            name: name.to_string(),
            dimensions,
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    fn prepared_for_cache(
        model_id: &str,
        model_path: &str,
        token_ids: Vec<u32>,
        session: LlamaInferenceSession,
    ) -> PreparedGeneration {
        PreparedGeneration {
            _model_file_lease: None,
            model_id: model_id.to_string(),
            model_path: PathBuf::from(model_path),
            token_ids,
            max_tokens: 1,
            tokenizer: Arc::new(test_tokenizer()),
            session,
            sampling: SamplingConfig::default(),
            logprobs_top_n: None,
            constraint: None,
            stop_sequences: Vec::new(),
            logit_diagnostic_token_ids: Vec::new(),
            collect_dense_diagnostics: false,
            dense_diagnostic_generated_index: None,
            dense_metadata: dummy_dense_metadata(),
            timings: GenerationTimings::default(),
            cached_prompt_prefix: Arc::new(Mutex::new(PromptPrefixCachePool::with_capacity(8))),
            speculative: None,
            telemetry: None,
            cancel: GenerationCancel::unbounded(),
        }
    }

    fn dummy_dense_metadata() -> DenseDiagnosticMetadata {
        let orientation = LinearProjectionOrientation {
            shape: vec![],
            input_width: 0,
            output_width: 0,
            descriptor_layout: "test",
            runtime_interpretation: "test",
            square_diagnostic_applies: false,
        };
        DenseDiagnosticMetadata {
            embedding_length: 4,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            head_dim: 2,
            rope_dimension_count: 2,
            rope_freq_base: 10_000.0,
            rope_scaling_type: "none".to_string(),
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rope_pairing: "split_half",
            rope_direction: "forward",
            rope_position_mode: "zero_based",
            gqa_head_mapping: "grouped",
            attention_score_scale: "head_dim",
            linear_accumulation: "f32",
            ffn_gate_up_order: "gate_up",
            rms_norm_epsilon: 1e-6,
            rms_norm_effective_epsilon: 1e-6,
            square_linear_diagnostic_layout: "test",
            rectangular_linear_diagnostic_layout: "test",
            token_embedding_shape: vec![3, 4],
            output_shape: vec![3, 4],
            output_is_tied_embedding: false,
            output_projection_layout: "output_input",
            output_projection_diagnostic_layout: "output_input",
            zero_attention_delta: "none".to_string(),
            zero_ffn_delta: "none".to_string(),
            projection_orientations: DenseProjectionOrientations {
                attention_q: orientation.clone(),
                attention_k: orientation.clone(),
                attention_v: orientation.clone(),
                attention_output: orientation.clone(),
                ffn_gate: orientation.clone(),
                ffn_up: orientation.clone(),
                ffn_down: orientation,
            },
        }
    }

    fn test_tokenizer() -> Tokenizer {
        Tokenizer {
            model: TokenizerModel::LlamaSpm,
            bpe_pre_tokenizer: BpePreTokenizer::default(),
            tokens: Vec::new(),
            token_to_id: HashMap::new(),
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens::default(),
            config: TokenizerConfig {
                add_bos: false,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: None,
        }
    }

    fn tiny_spec_config() -> LlamaModelConfig {
        LlamaModelConfig {
            context_length: 48,
            ..tiny_config()
        }
    }

    fn vanilla_greedy_tokens(
        config: &LlamaModelConfig,
        weights: &Arc<LlamaLoadedWeights>,
        prompt: &[u32],
        count: usize,
    ) -> Vec<u32> {
        let mut session = LlamaInferenceSession::new(config.clone(), Arc::clone(weights)).unwrap();
        let mut generated = Vec::new();
        let mut history = prompt.to_vec();
        let mut input = prompt.to_vec();
        for _ in 0..count {
            let step = session
                .generate_next_token_with_history_diagnostics(
                    &input,
                    LlamaSampler::Greedy,
                    &history,
                    false,
                    None,
                )
                .unwrap();
            generated.push(step.next_token_id);
            history.push(step.next_token_id);
            input = vec![step.next_token_id];
        }
        generated
    }

    /// Lossless speculation invariant: the spec round loop (batched verify +
    /// rollback) emits exactly the token stream vanilla greedy decode emits.
    /// Callers must hold `test_support::env_lock()`: the vanilla and spec
    /// passes must see the same CAMELID_* kernel-route env, and parallel
    /// tests mutate it.
    fn assert_speculative_matches_vanilla(mut drafter: SpeculativeDrafter) {
        let config = tiny_spec_config();
        let weights = Arc::new(tiny_weights());
        let prompt = vec![0u32, 1, 2, 0];
        let count = 12;
        let vanilla = vanilla_greedy_tokens(&config, &weights, &prompt, count);

        let mut session = LlamaInferenceSession::new(config.clone(), Arc::clone(&weights)).unwrap();
        let mut generated = Vec::new();
        let mut history = prompt.clone();
        // First step mirrors the real loop: the whole prompt through the
        // general path, speculation afterwards.
        let first = session
            .generate_next_token_with_history_diagnostics(
                &prompt,
                LlamaSampler::Greedy,
                &history,
                false,
                None,
            )
            .unwrap();
        generated.push(first.next_token_id);
        history.push(first.next_token_id);
        let mut accepted_total = 0usize;
        let mut latch = SpecLatch::default();
        while generated.len() < count {
            let pending = *history.last().unwrap();
            let mut emitted = Vec::new();
            if latch.should_speculate() {
                let drafts = drafter.draft(&history, 3).unwrap();
                if !drafts.is_empty() {
                    let base = session.kv_position();
                    let mut batch = vec![pending];
                    batch.extend_from_slice(&drafts);
                    let (predictions, _timings) =
                        session.forward_greedy_verify_chunk(&batch).unwrap();
                    let accepted = accepted_draft_prefix(&drafts, &predictions);
                    session.rollback_to_position(base + 1 + accepted).unwrap();
                    latch.note_verified(accepted as u32);
                    accepted_total += accepted;
                    emitted.extend_from_slice(&predictions[..=accepted]);
                }
            } else {
                latch.note_skip();
            }

            // An n-gram anchor miss is neither a verified acceptance result nor
            // a latch-directed skip. It takes one plain step without mutating
            // the latch, exactly like the production loop.
            if emitted.is_empty() {
                let step = session
                    .generate_next_token_with_history_diagnostics(
                        &[pending],
                        LlamaSampler::Greedy,
                        &history,
                        false,
                        None,
                    )
                    .unwrap();
                emitted.push(step.next_token_id);
            }

            for token in emitted {
                if generated.len() >= count {
                    break;
                }
                generated.push(token);
                history.push(token);
            }
        }

        assert_eq!(generated, vanilla);
        // The tiny model settles into a cycle, so the drafters must have
        // actually accepted drafted tokens â€” otherwise this test would pass
        // without exercising multi-token acceptance and rollback.
        assert!(
            accepted_total > 0,
            "speculation accepted no drafts; the test did not exercise the accept path"
        );
    }

    #[test]
    fn speculative_ngram_env_bounds_are_positive_and_normalized() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(SPEC_NGRAM_MIN_ENV);
        std::env::remove_var(SPEC_NGRAM_MAX_ENV);
        assert_eq!(spec_ngram_min_from_env(), DEFAULT_NGRAM_MIN_MATCH);
        assert_eq!(spec_ngram_max_from_env(), DEFAULT_NGRAM_MAX_MATCH);

        std::env::set_var(SPEC_NGRAM_MIN_ENV, "0");
        std::env::set_var(SPEC_NGRAM_MAX_ENV, "invalid");
        assert_eq!(spec_ngram_min_from_env(), DEFAULT_NGRAM_MIN_MATCH);
        assert_eq!(spec_ngram_max_from_env(), DEFAULT_NGRAM_MAX_MATCH);

        std::env::set_var(SPEC_NGRAM_MIN_ENV, "5");
        std::env::set_var(SPEC_NGRAM_MAX_ENV, "2");
        let normalized = NGramDrafter::new(spec_ngram_min_from_env(), spec_ngram_max_from_env());
        assert_eq!(normalized.min_ngram, 5);
        assert_eq!(normalized.max_ngram, 5);

        std::env::remove_var(SPEC_NGRAM_MIN_ENV);
        std::env::remove_var(SPEC_NGRAM_MAX_ENV);
    }

    #[test]
    fn ngram_speculation_matches_vanilla_greedy_decode() {
        let _guard = crate::test_support::env_lock();
        assert_speculative_matches_vanilla(SpeculativeDrafter::NGram(NGramDrafter::default()));
    }

    #[test]
    fn model_drafter_self_draft_matches_vanilla_greedy_decode() {
        let _guard = crate::test_support::env_lock();
        let config = tiny_spec_config();
        let weights = Arc::new(tiny_weights());
        // The target drafts for itself: every draft should be accepted, and
        // the output must still be byte-identical to vanilla decode.
        let draft_session = LlamaInferenceSession::new(config, weights).unwrap();
        assert_speculative_matches_vanilla(SpeculativeDrafter::Model(Box::new(ModelDrafter::new(
            draft_session,
        ))));
    }

    #[test]
    fn verify_chunk_rollback_restores_decode_state() {
        let _guard = crate::test_support::env_lock();
        let config = tiny_spec_config();
        let weights = Arc::new(tiny_weights());
        let prompt = vec![0u32, 1, 2];

        let mut clean = LlamaInferenceSession::new(config.clone(), Arc::clone(&weights)).unwrap();
        clean.forward_greedy_verify_chunk(&prompt).unwrap();
        let (expected, _timings) = clean.forward_greedy_verify_chunk(&[0]).unwrap();

        let mut rolled = LlamaInferenceSession::new(config, weights).unwrap();
        rolled.forward_greedy_verify_chunk(&prompt).unwrap();
        let base = rolled.kv_position();
        // Speculate down a wrong path, then roll it back.
        rolled.forward_greedy_verify_chunk(&[0, 2, 2, 1]).unwrap();
        rolled.rollback_to_position(base).unwrap();
        let (after_rollback, _timings) = rolled.forward_greedy_verify_chunk(&[0]).unwrap();

        assert_eq!(after_rollback, expected);
    }

    fn tiny_config() -> LlamaModelConfig {
        LlamaModelConfig {
            context_length: 4,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 6,
            attention_head_count: 2,
            attention_head_count_kv: 1,
            kv_quant: crate::model::KvCacheQuantization::F16,
            rope_dimension_count: Some(2),
            rope_freq_base: Some(10_000.0),
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: 1e-6,
            vocab_size: Some(3),
            file_type: Some(0),
            rope_neox_pairing: false,
            attention_key_length: None,
            logit_scale: None,
            moe: None,
            gemma4: None,
            qwen35: None,
            mla: None,
        }
    }

    fn tiny_weights() -> LlamaLoadedWeights {
        let hidden = 4;
        let ffn = 6;
        LlamaLoadedWeights {
            token_embedding: tensor(
                "token_embd.weight",
                vec![3, hidden],
                vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            output_norm: ones("output_norm.weight", hidden),
            output: Some(tensor(
                "output.weight",
                vec![3, hidden],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )),
            rope_freqs: None,
            layers: vec![LlamaLayerWeights {
                attention_norm: ones("blk.0.attn_norm.weight", hidden),
                attention_q: select_rows("blk.0.attn_q.weight", hidden, hidden, &[0, 1, 2, 3]),
                attention_k: select_rows("blk.0.attn_k.weight", 2, hidden, &[0, 1]),
                attention_v: scaled_select_rows("blk.0.attn_v.weight", 2, hidden, &[0, 1], 0.5),
                attention_output: select_rows(
                    "blk.0.attn_output.weight",
                    hidden,
                    hidden,
                    &[0, 1, 2, 3],
                ),
                attention_biases: None,
                ffn_norm: ones("blk.0.ffn_norm.weight", hidden),
                ffn_gate: select_rows("blk.0.ffn_gate.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
                ffn_up: select_rows("blk.0.ffn_up.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
                ffn_down: select_rows("blk.0.ffn_down.weight", hidden, ffn, &[0, 1, 2, 3]),
                attention_q_norm: None,
                attention_k_norm: None,
                moe_router: None,
                mla_q_a_proj: None,
                mla_q_a_layernorm: None,
                mla_q_b_proj: None,
                mla_kv_a_proj_with_mqa: None,
                mla_kv_a_layernorm: None,
                mla_kv_b_proj: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                decode_bindings: DecodeLinearBindings::default(),
            }],
            layer_range: None,
            output_projection_binding: DecodeBindingCell::default(),
        }
    }

    fn ones(name: &str, width: usize) -> CpuTensor {
        tensor(name, vec![width], vec![1.0; width])
    }

    fn tensor(name: &str, dims: Vec<usize>, data: Vec<f32>) -> CpuTensor {
        CpuTensor::from_f32(name, dims, data).unwrap()
    }

    fn select_rows(name: &str, rows: usize, cols: usize, source_cols: &[usize]) -> CpuTensor {
        scaled_select_rows(name, rows, cols, source_cols, 1.0)
    }

    fn scaled_select_rows(
        name: &str,
        rows: usize,
        cols: usize,
        source_cols: &[usize],
        scale: f32,
    ) -> CpuTensor {
        let mut data = vec![0.0; rows * cols];
        for (row, source_col) in source_cols.iter().copied().enumerate() {
            data[row * cols + source_col] = scale;
        }
        tensor(name, vec![rows, cols], data)
    }
}

// ==========================================================================
// Curated model catalog and background GGUF downloader endpoints
// ==========================================================================

#[derive(Debug, serde::Serialize, Clone)]
pub struct CatalogItem {
    pub catalog_id: &'static str,
    pub name: &'static str,
    pub repo_id: &'static str,
    pub filename: &'static str,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
    pub quant: &'static str,
    /// `general.architecture` the GGUF will report (authoritative â€” curated, not
    /// inferred from the filename). Drives the catalog's predicted lane.
    pub architecture: &'static str,
    pub license: &'static str,
    /// Advisory "best for" positioning for the Models tab (curated, not
    /// benchmarked). Constrained to: `general`, `reasoning`, `coding`, `tools`.
    pub task_tags: &'static [&'static str],
}

/// A catalog item plus its predicted runnable lane, so the Models tab can show which
/// lane each entry would land in without downloading it. `oracle_qualified` means the
/// `(architecture, quant)` combo is anchored â†’ it would be Compatible after download
/// (unless it also matches a supported contract row, which the frontend resolves).
///
/// Owns its strings so it can carry both **curated** rows (authoritative metadata,
/// `arch_detected: true`) and **experimental** live Hugging Face results
/// (`arch_detected: false`, where `architecture`/`quant` are filename guesses). The
/// JSON field names are unchanged from the previous flattened `CatalogItem` shape;
/// `group`/`arch_detected` are additive. Experimental rows always report
/// `oracle_qualified: false` â€” a filename guess can never anchor a lane, so the
/// frontend resolves them to "not yet in a lane" regardless of name coincidence.
#[derive(Debug, serde::Serialize)]
pub struct CatalogItemView {
    pub catalog_id: String,
    pub name: String,
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
    pub quant: String,
    pub architecture: String,
    pub license: String,
    pub oracle_qualified: bool,
    /// `"curated"` (pinned, known-good, authoritative metadata) or `"experimental"`
    /// (live from Hugging Face; metadata is advisory and must never read as support).
    pub group: &'static str,
    /// Whether `architecture` is authoritative (curated) vs a filename guess (live).
    pub arch_detected: bool,
    /// Capacity advisory: whether THIS host can load/run the row (fit axis only —
    /// never a support/parity claim). `Unknown` for experimental rows and for hosts
    /// whose memory cannot be probed.
    pub fit: crate::fit::FitVerdict,
    /// Advisory "best for" positioning (e.g. `general`, `reasoning`, `coding`,
    /// `tools`) — curated, not benchmarked. Empty for experimental rows.
    pub task_tags: Vec<String>,
    /// Confidence of the `fit` estimate: `"exact"` when computed from the model's
    /// real GGUF dimensions (KV cache sized precisely), `"approx"` when from the
    /// coarse size-based heuristic.
    pub fit_confidence: &'static str,
    /// Whether Camelid implements this row's **guessed** architecture:
    /// `"implemented"`, `"not_implemented"`, or `"unknown"`.
    ///
    /// Exists for exactly one job: letting the Models tab fold away live Hugging
    /// Face results Camelid could never load. It is therefore only derived for
    /// experimental rows, where `architecture` comes from
    /// [`crate::hf_browse::guess_architecture`], whose token vocabulary is the same
    /// one [`crate::model::is_implemented_architecture`] tests.
    ///
    /// **Curated rows always report `"unknown"`**, and that is a correctness
    /// requirement, not laziness. `CatalogItem::architecture` is a curated *label*
    /// whose vocabulary is not the loader's: the shipped catalog contains
    /// `qwen25` (the loader knows `qwen2` and `qwen35`, not `qwen25`) and
    /// `command-r`. Deriving from it stamped three working, pinned rows
    /// `not_implemented` — a false negative on the rows we vouch for. Curated rows
    /// are pinned and never filtered by this axis, so the honest answer is to make
    /// no claim rather than a wrong one.
    ///
    /// Tri-state on purpose even for experimental rows: a confident
    /// `"not_implemented"` is only reported when an architecture token was actually
    /// recognized and sits outside the implemented set. When nothing matched we
    /// report `"unknown"`. Collapsing this to a bool would make "we could not tell"
    /// indistinguishable from "we know it cannot run", and a UI filtering on it
    /// would silently hide loadable models.
    ///
    /// Advisory in every case: the authoritative architecture is read from GGUF
    /// metadata at load time and is the only thing that gates a load.
    pub arch_support: &'static str,
}

/// The `general.architecture` a GGUF guessed as `token` will actually declare.
///
/// The browse guesser names model *families*; the loader allowlist names GGUF
/// architectures, and those are not the same vocabulary — the same mismatch that
/// made curated labels unusable for this axis. Mixtral is the case that bites:
/// its GGUFs declare `llama` (the MoE tensors ride the llama architecture, which
/// is why [`crate::model::MixtralMoeMetadata`] reads `llama.expert_count`), and
/// Camelid ships an evidence-backed `mixtral_8x7b_instruct_v0_1_q8_0` row. Testing
/// the family token directly stamped every Mixtral repo `not_implemented`, which
/// folded a family we vouch for into the "cannot run this" drawer.
fn loader_architecture_for_guess(token: &str) -> &str {
    match token {
        "mixtral" => "llama",
        other => other,
    }
}

/// Advisory architecture-support label for a **filename-guessed** architecture.
/// An empty or unrecognized architecture is `"unknown"`, never `"not_implemented"`.
fn guessed_arch_support(architecture: &str) -> &'static str {
    if crate::model::is_implemented_architecture(loader_architecture_for_guess(architecture)) {
        "implemented"
    } else if architecture.is_empty() {
        "unknown"
    } else {
        "not_implemented"
    }
}

/// KV dtype the runtime will actually allocate on this host: f16 on the
/// GPU-resident path, f32 on CPU. Mirrors the engine so the estimate matches what
/// gets allocated rather than a nominal figure.
fn kv_dtype_for(hw: &crate::capability::HardwareProfile) -> crate::fit::KvDtype {
    if hw.cuda_available && hw.cuda_vram_free_bytes > 0 {
        crate::fit::KvDtype::F16
    } else {
        crate::fit::KvDtype::F32
    }
}

/// Exact footprint (weights + KV at the advisory context + bounded scratch) for a
/// model whose real GGUF dimensions are known. Single definition shared by the
/// curated badge, the experimental badge, and the on-demand fit check, so the three
/// surfaces can never drift into disagreeing about the same model.
fn exact_footprint_for(
    size_bytes: u64,
    dims: crate::fit::ModelDims,
    hw: &crate::capability::HardwareProfile,
) -> crate::fit::FitInputs {
    crate::fit::exact_footprint(
        size_bytes,
        dims,
        crate::fit::ADVISORY_CONTEXT_TOKENS,
        kv_dtype_for(hw),
    )
}

/// Whether `(repo_id, filename)` names a pinned curated row rather than an
/// arbitrary Hugging Face file. Curated rows are the ones we vouch for and the
/// only ones the coarse size pad is authorized for, so the on-demand fit check
/// asks this before deciding whether an unmeasurable model still gets a verdict.
fn is_curated_artifact(repo_id: &str, filename: &str) -> bool {
    curated_catalog()
        .iter()
        .any(|item| item.repo_id == repo_id && item.filename == filename)
}

/// Footprint + confidence for a curated row: an **exact** footprint (weights + KV
/// sized from the model's real GGUF dims) when `dims` are known, else the coarse
/// size pad. Pure: the caller supplies the resolved dims (from `fit_dims`), so this
/// is unit-testable without the global resolver.
fn curated_footprint(
    size_bytes: u64,
    dims: Option<crate::fit::ModelDims>,
    hw: &crate::capability::HardwareProfile,
) -> (crate::fit::FitInputs, &'static str) {
    match dims {
        Some(dims) => (exact_footprint_for(size_bytes, dims, hw), "exact"),
        None => (crate::fit::advisory_footprint(size_bytes), "approx"),
    }
}

impl CatalogItemView {
    /// Build a view for a curated row: authoritative architecture, lane predicted
    /// from the real `(architecture, quant)` via `runnable::oracle_qualified`.
    fn from_curated(item: &CatalogItem, hw: &crate::capability::HardwareProfile) -> Self {
        let (footprint, fit_confidence) = curated_footprint(
            item.size_bytes,
            crate::fit_dims::global().lookup(item.repo_id, item.filename),
            hw,
        );
        CatalogItemView {
            catalog_id: item.catalog_id.to_string(),
            name: item.name.to_string(),
            repo_id: item.repo_id.to_string(),
            filename: item.filename.to_string(),
            size_bytes: item.size_bytes,
            downloads: item.downloads,
            likes: item.likes,
            quant: item.quant.to_string(),
            architecture: item.architecture.to_string(),
            license: item.license.to_string(),
            oracle_qualified: crate::runnable::oracle_qualified(item.architecture, item.quant),
            group: "curated",
            arch_detected: true,
            fit: crate::fit::assess(hw, &footprint),
            task_tags: item.task_tags.iter().map(|t| t.to_string()).collect(),
            fit_confidence,
            // Deliberately no claim: see `arch_support`. The curated `architecture`
            // label uses a different vocabulary from the loader's allowlist.
            arch_support: "unknown",
        }
    }

    /// Build a view for a live Hugging Face result. Architecture/quant are filename
    /// guesses (`arch_detected: false`); `oracle_qualified` is forced `false` so the
    /// experimental row is never predicted Compatible/Supported on a guess alone.
    /// Capacity is orthogonal to verification: when this file's real GGUF header has
    /// been read (cached), we give an honest `exact` fit even for an unverified row;
    /// otherwise we make **no** fit claim (`Unknown`) rather than guess on a filename.
    fn from_hf(
        file: crate::hf_browse::HfGgufFile,
        hw: &crate::capability::HardwareProfile,
    ) -> Self {
        let catalog_id = format!("hf::{}::{}", file.repo_id, file.filename);
        let (fit, fit_confidence) =
            match crate::fit_dims::global().lookup(&file.repo_id, &file.filename) {
                Some(dims) => (
                    crate::fit::assess(hw, &exact_footprint_for(file.size_bytes, dims, hw)),
                    "exact",
                ),
                None => (crate::fit::FitVerdict::Unknown, "unknown"),
            };
        CatalogItemView {
            catalog_id,
            name: file.filename.clone(),
            repo_id: file.repo_id,
            filename: file.filename,
            size_bytes: file.size_bytes,
            downloads: file.downloads,
            likes: file.likes,
            quant: file.quant,
            arch_support: guessed_arch_support(&file.architecture),
            architecture: file.architecture,
            license: String::new(),
            oracle_qualified: false,
            group: "experimental",
            arch_detected: false,
            fit,
            task_tags: Vec::new(),
            fit_confidence,
        }
    }
}

pub fn curated_catalog() -> Vec<CatalogItem> {
    vec![
        CatalogItem {
            catalog_id: "llama32_1b_instruct_q8_0",
            name: "Llama 3.2 1B Instruct Q8_0",
            repo_id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
            filename: "Llama-3.2-1B-Instruct-Q8_0.gguf",
            size_bytes: 1321082528,
            downloads: 142000,
            likes: 540,
            quant: "Q8_0",
            architecture: "llama",
            license: "llama3.2",
            task_tags: &["general", "tools"],
        },
        CatalogItem {
            catalog_id: "llama32_3b_instruct_q8_0",
            name: "Llama 3.2 3B Instruct Q8_0",
            repo_id: "unsloth/Llama-3.2-3B-Instruct-GGUF",
            filename: "Llama-3.2-3B-Instruct-Q8_0.gguf",
            size_bytes: 3421898816,
            downloads: 98000,
            likes: 420,
            quant: "Q8_0",
            architecture: "llama",
            license: "llama3.2",
            task_tags: &["general", "tools"],
        },
        CatalogItem {
            catalog_id: "tinyllama_1_1b_chat_q8_0",
            name: "TinyLlama 1.1B Chat Q8_0",
            repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            // Verified against the HuggingFace resolve Content-Length (2026-06-14);
            // must match exactly or `pull`'s skip-if-complete/resume check refires.
            size_bytes: 1170781568,
            downloads: 512000,
            likes: 1240,
            quant: "Q8_0",
            architecture: "llama",
            license: "other",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "llama3_8b_instruct_q8_0",
            name: "Llama 3 8B Instruct Q8_0",
            repo_id: "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF",
            filename: "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
            size_bytes: 8541283552,
            downloads: 320000,
            likes: 920,
            quant: "Q8_0",
            architecture: "llama",
            license: "llama3",
            task_tags: &["general", "reasoning"],
        },
        CatalogItem {
            catalog_id: "mistral_7b_instruct_v0_3_q8_0",
            name: "Mistral 7B Instruct v0.3 Q8_0",
            repo_id: "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            filename: "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
            size_bytes: 7702565088,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "llama",
            license: "apache-2.0",
            task_tags: &["general", "coding"],
        },
        CatalogItem {
            catalog_id: "qwen3_0_6b_instruct_q8_0",
            name: "Qwen3 0.6B Q8_0",
            repo_id: "Qwen/Qwen3-0.6B-GGUF",
            filename: "Qwen3-0.6B-Q8_0.gguf",
            size_bytes: 639446688,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen3",
            license: "apache-2.0",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "qwen3_1_7b_instruct_q8_0",
            name: "Qwen3 1.7B Q8_0",
            repo_id: "Qwen/Qwen3-1.7B-GGUF",
            filename: "Qwen3-1.7B-Q8_0.gguf",
            size_bytes: 1834426016,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen3",
            license: "apache-2.0",
            task_tags: &["general", "reasoning"],
        },
        CatalogItem {
            catalog_id: "qwen3_4b_instruct_q8_0",
            name: "Qwen3 4B Q8_0",
            repo_id: "Qwen/Qwen3-4B-GGUF",
            filename: "Qwen3-4B-Q8_0.gguf",
            size_bytes: 4280404704,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen3",
            license: "apache-2.0",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            // First non-Q8_0 curated row. `qwen3_4b_q4_k_m` is a
            // `supported_exact_row_smoke` contract row (tool_capable), but it was
            // absent from this catalog, so `camelid pull qwen3_4b_q4_k_m`, the TUI
            // picker, and the Web UI browse all saw it as supported yet could not
            // fetch it (upstream #469). The certified GGUF (sha256 7485fe6f...) is
            // the official Qwen upload, so it downloads from the same repo as the
            // Q8_0 row above. catalog_id MUST equal the ledger row id
            // `qwen3_4b_q4_k_m`: the picker joins the two by exact id, and the
            // server's supported-row check anchors on the exact filename.
            catalog_id: "qwen3_4b_q4_k_m",
            name: "Qwen3 4B Q4_K_M",
            repo_id: "Qwen/Qwen3-4B-GGUF",
            filename: "Qwen3-4B-Q4_K_M.gguf",
            // HuggingFace resolve Content-Length (verified 2026-07-23); matches the
            // certified parity bundle's recorded size. Must stay exact or pull's
            // skip-if-complete/resume check misfires.
            size_bytes: 2497280256,
            downloads: 0,
            likes: 0,
            quant: "Q4_K_M",
            architecture: "qwen3",
            license: "apache-2.0",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            catalog_id: "qwen3_8b_instruct_q8_0",
            name: "Qwen3 8B Q8_0",
            repo_id: "Qwen/Qwen3-8B-GGUF",
            filename: "Qwen3-8B-Q8_0.gguf",
            size_bytes: 8709518112,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen3",
            license: "apache-2.0",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            catalog_id: "gemma4_e4b_it_q8_0",
            name: "Gemma 4 E4B-It Q8_0",
            repo_id: "unsloth/gemma-4-E4B-it-GGUF",
            filename: "gemma-4-E4B-it-Q8_0.gguf",
            size_bytes: 8192951456,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "gemma4",
            license: "gemma",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "gemma4_e2b_it_q8_0",
            name: "Gemma 4 E2B-It Q8_0",
            repo_id: "unsloth/gemma-4-E2B-it-GGUF",
            filename: "gemma-4-E2B-it-Q8_0.gguf",
            size_bytes: 5048350848,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "gemma4",
            license: "gemma",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "gemma4_12b_it_q8_0",
            name: "Gemma 4 12B-It Q8_0 (two-Mac distributed)",
            repo_id: "unsloth/gemma-4-12b-it-GGUF",
            filename: "gemma-4-12b-it-Q8_0.gguf",
            size_bytes: 12669646240,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "gemma4",
            license: "gemma",
            task_tags: &["general", "reasoning"],
        },
        CatalogItem {
            catalog_id: "gemma4_26b_a4b_it_q4_0",
            name: "Gemma 4 26B-A4B-It QAT Q4_0 (two-Mac distributed, MoE)",
            repo_id: "google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
            filename: "gemma-4-26B_q4_0-it.gguf",
            size_bytes: 14439361440,
            downloads: 0,
            likes: 0,
            quant: "Q4_0",
            architecture: "gemma4",
            license: "gemma",
            task_tags: &["reasoning"],
        },
        CatalogItem {
            catalog_id: "gemma3_1b_it_q8_0",
            name: "Gemma 3 1B-It Q8_0",
            repo_id: "ggml-org/gemma-3-1b-it-GGUF",
            filename: "gemma-3-1b-it-Q8_0.gguf",
            size_bytes: 1069306368,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "gemma3",
            license: "gemma",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "phi3_mini_4k_instruct_q8_0",
            name: "Phi-3-mini-4k-instruct Q8_0",
            repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
            filename: "Phi-3-mini-4k-instruct-Q8_0.gguf",
            // Corrected 2026-07-16 (MUSTER Phase 2 stop-and-amend): verified against the
            // live HF tree (lfs.size) — bartowski's single 2024-04-29 upload series is
            // 4,061,221,376 B; the previously baked 4,061,222,688 matched no upload.
            size_bytes: 4061221376,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "phi3",
            license: "mit",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            catalog_id: "deepseek_r1_distill_qwen_7b_q8_0",
            name: "DeepSeek R1 Distill Qwen 7B Q8_0",
            repo_id: "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            filename: "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
            size_bytes: 8098524896,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen25",
            license: "apache-2.0",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            catalog_id: "c4ai_command_r_v01_q8_0",
            name: "Cohere Command R v01 Q8_0",
            repo_id: "second-state/C4AI-Command-R-v01-GGUF",
            filename: "c4ai-command-r-v01-Q8_0.gguf",
            size_bytes: 37179013760,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "command-r",
            license: "cc-by-nc-4.0",
            task_tags: &["general"],
        },
        CatalogItem {
            catalog_id: "deepseek_r1_distill_llama_8b_q8_0",
            name: "DeepSeek R1 Distill Llama 8B Q8_0",
            repo_id: "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
            filename: "DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf",
            size_bytes: 8540773088,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "llama",
            license: "llama3.1",
            task_tags: &["reasoning", "coding"],
        },
        CatalogItem {
            catalog_id: "qwen25_coder_7b_q8_0",
            name: "Qwen2.5 Coder 7B Q8_0",
            repo_id: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            filename: "qwen2.5-coder-7b-instruct-q8_0.gguf",
            size_bytes: 8098525184,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "qwen25",
            license: "apache-2.0",
            task_tags: &["coding"],
        },
        CatalogItem {
            catalog_id: "llama31_8b_instruct_q8_0",
            name: "Llama 3.1 8B Instruct Q8_0",
            repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename: "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            size_bytes: 8540775840,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "llama",
            license: "llama3.1",
            task_tags: &["general"],
        },
    ]
}

#[derive(Debug, serde::Serialize)]
pub struct CatalogResponse {
    pub items: Vec<CatalogItemView>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct CatalogQuery {
    pub query: Option<String>,
    /// Opaque Hugging Face pagination cursor for the experimental group (from a
    /// prior response's `next_cursor`). Ignored when `query` is absent/trivial.
    pub cursor: Option<String>,
}

/// Body of an on-demand capacity check for one live Hugging Face GGUF.
#[derive(Debug, serde::Deserialize)]
pub struct CatalogFitRequest {
    pub repo_id: String,
    pub filename: String,
    /// The file's real size in bytes, from the browse result. Required: the header
    /// parser validates tensor offsets against the model's true full length.
    pub size_bytes: u64,
}

/// Result of an on-demand capacity check. Echoes the identity so a client that
/// fired several checks can match responses to rows without tracking order.
#[derive(Debug, serde::Serialize)]
pub struct CatalogFitResponse {
    pub repo_id: String,
    pub filename: String,
    pub fit: crate::fit::FitVerdict,
    /// `"exact"` when the model's real GGUF dimensions were read, `"unknown"` when
    /// they could not be (unreachable, or not a dense LLaMA-family model).
    pub fit_confidence: &'static str,
    /// Whether the check **settled**: `true` when there is nothing further to learn
    /// by asking again (dimensions resolved, or a definitive negative), `false` when
    /// the attempt failed for a reason a retry could fix.
    ///
    /// Without this a client cannot tell "we checked, and the answer is that we
    /// cannot say" from "we have not checked yet" — both arrive as an `unknown`
    /// verdict — so a "check this" affordance can never be retired and pressing it
    /// looks like a no-op.
    pub checked: bool,
}

/// One local on-disk GGUF with the facts the Models tab needs to bucket it by lane.
/// Cheap to compute: header-only metadata + admission, no model load or generation,
/// no whole-file hashing. The frontend derives "supported" by matching against the
/// `/api/capabilities` contract; the runnable-lane facts come from here.
#[derive(Debug, serde::Serialize)]
pub struct LocalModelEntry {
    pub filename: String,
    pub size_bytes: u64,
    /// Opaque, process-bound authorization for deleting this exact unchanged
    /// directory entry. Clients must return it verbatim; metadata alone is not
    /// sufficient to authorize a destructive operation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delete_token: Option<String>,
    pub architecture: Option<String>,
    /// Headline (dominant non-F32) quant, e.g. `Q8_0`.
    pub quantization: Option<String>,
    /// `llama_spm` / `gpt2_bpe`, from the runnable tokenizer family.
    pub tokenizer_kind: Option<String>,
    /// Passes the runnable covered-set admission gate.
    pub admitted: bool,
    /// Machine-readable refusal reason when `admitted` is false.
    pub admission_reason: Option<String>,
    /// The (architecture, quant) combo is oracle-qualified, so smoke-admission is
    /// allowed. A model that is admitted but NOT oracle_qualified is "combo not yet
    /// anchored" â€” never presented as compatible.
    pub oracle_qualified: bool,
    /// A cached runnable smoke receipt already exists for this file (it passed
    /// smoke-admission before) â€” i.e. it belongs in the Compatible section.
    pub runnable_receipt_present: bool,
    /// The GGUF ships a chat template â€” i.e. it is an instruction-tuned chat model
    /// (vs a base text-completion model). A model capability, not a system fact.
    pub chat_capable: bool,
    /// Trained context window (tokens) from the GGUF â€” a model capability.
    pub context_length: Option<u32>,
    /// Server-computed lane class from real header metadata (architecture) + the
    /// exact-artifact supported-row check. `experimental_implemented` means the
    /// architecture is implemented but this is NOT a supported row â€” attemptable,
    /// unverified, no parity claim. Corroborates the frontend contract gate; it
    /// never promotes a row.
    pub lane_class: ModelLaneClass,
}

#[derive(Debug, serde::Serialize)]
pub struct LocalModelsResponse {
    /// The configured models directory the scan covered (`serve --models-dir` /
    /// `CAMELID_MODELS_DIR`, default exe-adjacent or CWD `models`), as the
    /// display form of the absolute path resolved at startup. The frontend
    /// builds load paths from this, so they stay valid whatever CWD the server
    /// process inherited.
    pub models_dir: String,
    pub models: Vec<LocalModelEntry>,
}

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalModelFileIdentity {
    size_bytes: u64,
    modified_nanos: u128,
    delete_token: Option<String>,
}

fn validate_local_model_filename(filename: &str) -> bool {
    let path = std::path::Path::new(filename);
    filename == filename.trim()
        && !filename.is_empty()
        && !filename.contains(['/', '\\', ':'])
        && matches!(
            (path.components().next(), path.components().nth(1)),
            (Some(std::path::Component::Normal(_)), None)
        )
        && path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

#[cfg(windows)]
fn local_model_delete_token(
    nonce: &str,
    filename: &str,
    size_bytes: u64,
    modified_nanos: u128,
    platform_id: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"camelid.local-model-delete/v1\0");
    hash.update(nonce.as_bytes());
    hash.update(b"\0");
    hash.update(filename.as_bytes());
    hash.update(b"\0");
    hash.update(size_bytes.to_le_bytes());
    hash.update(modified_nanos.to_le_bytes());
    hash.update(platform_id.as_bytes());
    format!("{:x}", hash.finalize())
}

#[cfg(windows)]
struct LocalModelWindowsHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl Drop for LocalModelWindowsHandle {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
fn open_local_model_windows(
    path: &std::path::Path,
    exclusive_delete: bool,
) -> std::io::Result<LocalModelWindowsHandle> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::{
        Foundation::INVALID_HANDLE_VALUE,
        Storage::FileSystem::{
            CreateFileW, DELETE, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OPEN_REPARSE_POINT,
            FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
            OPEN_EXISTING,
        },
    };

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let share_mode = if exclusive_delete {
        FILE_SHARE_READ
    } else {
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    };
    let desired_access = if exclusive_delete {
        DELETE | FILE_READ_ATTRIBUTES
    } else {
        FILE_READ_ATTRIBUTES
    };
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            desired_access,
            share_mode,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(LocalModelWindowsHandle(handle))
    }
}

#[cfg(windows)]
fn windows_handle_identity(
    handle: windows_sys::Win32::Foundation::HANDLE,
    filename: &str,
    nonce: &str,
) -> std::io::Result<Option<LocalModelFileIdentity>> {
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_DIRECTORY,
        FILE_ATTRIBUTE_REPARSE_POINT,
    };

    let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
    if unsafe { GetFileInformationByHandle(handle, info.as_mut_ptr()) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let info = unsafe { info.assume_init() };
    if info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0 {
        return Ok(None);
    }
    let size_bytes = ((info.nFileSizeHigh as u64) << 32) | info.nFileSizeLow as u64;
    let modified_nanos = (((info.ftLastWriteTime.dwHighDateTime as u64) << 32)
        | info.ftLastWriteTime.dwLowDateTime as u64) as u128
        * 100;
    let platform_id = format!(
        "windows:{}:{}:{}:{}",
        info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow, info.nNumberOfLinks
    );
    let delete_token = (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT == 0
        && info.nNumberOfLinks == 1)
        .then(|| {
            local_model_delete_token(nonce, filename, size_bytes, modified_nanos, &platform_id)
        });
    Ok(Some(LocalModelFileIdentity {
        size_bytes,
        modified_nanos,
        delete_token,
    }))
}

#[cfg(windows)]
fn local_model_file_identity(
    models_dir: &std::path::Path,
    filename: &str,
    nonce: &str,
) -> std::io::Result<Option<LocalModelFileIdentity>> {
    if !validate_local_model_filename(filename) {
        return Ok(None);
    }
    let handle = open_local_model_windows(&models_dir.join(filename), false)?;
    windows_handle_identity(handle.0, filename, nonce)
}

#[cfg(windows)]
#[derive(Debug, serde::Deserialize)]
struct DeleteLocalModelRequest {
    filename: String,
    delete_token: String,
}

#[cfg(windows)]
#[derive(Debug, serde::Serialize)]
struct DeleteLocalModelResponse {
    deleted: bool,
    filename: String,
    bytes_freed: u64,
}

#[cfg(windows)]
fn local_management_request_allowed(headers: &HeaderMap) -> bool {
    let authority = headers
        .get("host")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::uri::Authority>().ok());
    let Some(authority) = authority else {
        return false;
    };
    let host = authority.host().trim_matches(['[', ']']);
    let host_is_loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if !host_is_loopback {
        return false;
    }

    let origin = headers
        .get("origin")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::Uri>().ok());
    if let Some(ref origin) = origin {
        if !matches!(origin.scheme_str(), Some("http") | Some("https")) {
            return false;
        }
        if origin.authority() != Some(&authority) {
            return false;
        }
    }
    origin.is_some()
        || headers
            .get("sec-fetch-site")
            .and_then(|value| value.to_str().ok())
            == Some("same-origin")
}

#[cfg(windows)]
#[derive(Debug)]
enum DeleteLocalModelError {
    Io(std::io::Error),
    Busy(std::io::Error),
    Changed,
    Unsafe,
}

#[cfg(windows)]
fn classify_delete_io(err: std::io::Error) -> DeleteLocalModelError {
    match err.raw_os_error() {
        Some(32 | 33) => DeleteLocalModelError::Busy(err),
        _ => DeleteLocalModelError::Io(err),
    }
}

#[cfg(windows)]
fn delete_identity_bound_local_model(
    path: &std::path::Path,
    filename: &str,
    nonce: &str,
    expected_token: &str,
) -> Result<u64, DeleteLocalModelError> {
    use windows_sys::Win32::Storage::FileSystem::{
        FileDispositionInfo, SetFileInformationByHandle, FILE_DISPOSITION_INFO,
    };

    let handle = open_local_model_windows(path, true).map_err(classify_delete_io)?;
    let identity = windows_handle_identity(handle.0, filename, nonce)
        .map_err(DeleteLocalModelError::Io)?
        .ok_or(DeleteLocalModelError::Unsafe)?;
    if identity.delete_token.as_deref() != Some(expected_token) {
        return Err(DeleteLocalModelError::Changed);
    }
    let disposition = FILE_DISPOSITION_INFO { DeleteFile: 1 };
    if unsafe {
        SetFileInformationByHandle(
            handle.0,
            FileDispositionInfo,
            (&disposition as *const FILE_DISPOSITION_INFO).cast(),
            std::mem::size_of::<FILE_DISPOSITION_INFO>() as u32,
        )
    } == 0
    {
        return Err(classify_delete_io(std::io::Error::last_os_error()));
    }
    Ok(identity.size_bytes)
}

#[cfg(not(windows))]
async fn delete_local_model(
    State(_state): State<AppState>,
    _headers: HeaderMap,
    Json(_req): Json<serde_json::Value>,
) -> Response {
    api_error(
        StatusCode::NOT_IMPLEMENTED,
        "model_delete_unsupported",
        "identity-bound model deletion is not available on this platform".to_string(),
        None,
    )
}

#[cfg(windows)]
async fn delete_local_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DeleteLocalModelRequest>,
) -> Response {
    if !state.allow_local_model_delete || !local_management_request_allowed(&headers) {
        return api_error(
            StatusCode::FORBIDDEN,
            "local_management_forbidden",
            "model deletion is available only from Camelid's loopback web UI".to_string(),
            None,
        );
    }
    if !validate_local_model_filename(&req.filename) || req.delete_token.len() != 64 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_model_identity",
            "delete requires one scanned local GGUF filename and its opaque identity token"
                .to_string(),
            Some("filename"),
        );
    }

    let _transition = state.model_transition.lock().await;
    let _exclusive = match state.model_file_lifecycle.try_write() {
        Ok(guard) => guard,
        Err(_) => {
            return api_error(
                StatusCode::CONFLICT,
                "model_operation_in_progress",
                "Wait for the current model operation to finish before deleting files.".to_string(),
                Some("filename"),
            )
        }
    };
    if !state.loaded_models.read().await.is_empty() || state.active_model_id.read().await.is_some()
    {
        return api_error(
            StatusCode::CONFLICT,
            "model_in_use",
            "Unload the current model before deleting any model from disk.".to_string(),
            Some("filename"),
        );
    }
    if active_downloads_map()
        .lock()
        .unwrap()
        .values()
        .any(|download| download.status == "downloading")
    {
        return api_error(
            StatusCode::CONFLICT,
            "model_download_in_progress",
            "Cancel or finish active model downloads before deleting files.".to_string(),
            Some("filename"),
        );
    }

    let identity = match local_model_file_identity(
        &state.models_dir,
        &req.filename,
        &state.model_delete_nonce,
    ) {
        Ok(Some(identity)) => identity,
        Ok(None) => {
            return api_error(
                StatusCode::CONFLICT,
                "unsafe_model_file",
                "the target is not the regular local GGUF returned by the latest scan".to_string(),
                Some("filename"),
            )
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return api_error(
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!("{} is no longer present", req.filename),
                Some("filename"),
            )
        }
        Err(err) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_identity_failed",
                format!("could not verify {}: {err}", req.filename),
                Some("filename"),
            )
        }
    };
    if identity.delete_token.as_deref() != Some(req.delete_token.as_str()) {
        return api_error(
            StatusCode::CONFLICT,
            "model_changed",
            "the model changed after confirmation; refresh Models and try again".to_string(),
            Some("delete_token"),
        );
    }

    let path = state.models_dir.join(&req.filename);
    let filename = req.filename.clone();
    let filename_for_delete = filename.clone();
    let nonce = state.model_delete_nonce.clone();
    let expected_token = req.delete_token.clone();
    let deletion = tokio::task::spawn_blocking(move || {
        delete_identity_bound_local_model(&path, &filename_for_delete, &nonce, &expected_token)
            .map(|bytes| (filename_for_delete, bytes))
    })
    .await;
    let (filename, bytes_freed) = match deletion {
        Ok(Ok(result)) => result,
        Ok(Err(DeleteLocalModelError::Changed)) => {
            return api_error(
                StatusCode::CONFLICT,
                "model_changed",
                "the model changed before deletion; refresh Models and try again".to_string(),
                Some("delete_token"),
            )
        }
        Ok(Err(DeleteLocalModelError::Unsafe)) => {
            return api_error(
                StatusCode::CONFLICT,
                "unsafe_model_file",
                "the target is a link, reparse point, directory, or hard-linked file".to_string(),
                Some("filename"),
            )
        }
        Ok(Err(DeleteLocalModelError::Busy(err))) => {
            return api_error(
                StatusCode::CONFLICT,
                "model_in_use",
                format!("could not delete {filename} because the file is in use: {err}"),
                Some("filename"),
            )
        }
        Ok(Err(DeleteLocalModelError::Io(err))) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_delete_failed",
                format!("could not delete {filename}: {err}"),
                Some("filename"),
            )
        }
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model_delete_failed",
                format!("model deletion task failed for {filename}"),
                Some("filename"),
            )
        }
    };

    local_meta_cache().lock().unwrap().remove(&filename);
    let _ = std::fs::remove_file(runnable_smoke_receipt_path(&filename));
    eprintln!(
        "[models] deleted local GGUF {:?} ({} bytes, identity {}…)",
        filename,
        bytes_freed,
        &req.delete_token[..12]
    );
    (
        StatusCode::OK,
        Json(DeleteLocalModelResponse {
            deleted: true,
            filename,
            bytes_freed,
        }),
    )
        .into_response()
}

/// Cached metadata-derived facts for one local model, keyed by (mtime, size) so a
/// re-download invalidates it. Parsing a GGUF's tensor index is slow for big models
/// (the 16 GB MoE alone is seconds), and this endpoint is polled â€” so we re-parse
/// only files that are new or changed.
#[derive(Clone)]
struct CachedLocalMeta {
    mtime_secs: u64,
    size_bytes: u64,
    architecture: Option<String>,
    quantization: Option<String>,
    tokenizer_kind: Option<String>,
    admitted: bool,
    admission_reason: Option<String>,
    oracle_qualified: bool,
    chat_capable: bool,
    context_length: Option<u32>,
}

fn local_meta_cache() -> &'static std::sync::Mutex<HashMap<String, CachedLocalMeta>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, CachedLocalMeta>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

fn runnable_smoke_receipt_path(filename: &str) -> PathBuf {
    let stem = std::path::Path::new(filename)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    PathBuf::from("qa")
        .join("runnable")
        .join("smoke")
        .join(format!("{stem}.json"))
}

/// `GET /api/models/local` â€” enumerate `<models_dir>/*.gguf` with per-model lane
/// facts. Membership is derived downstream from these facts; nothing here is
/// hand-authored.
async fn local_models(
    State(state): State<AppState>,
    _headers: HeaderMap,
) -> Json<LocalModelsResponse> {
    let _reader = state.model_file_lifecycle.read().await;
    #[cfg(windows)]
    let expose_delete_tokens =
        state.allow_local_model_delete && local_management_request_allowed(&_headers);
    let dir = state.models_dir.clone();
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
            })
            .collect();
        paths.sort();
        for path in paths {
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let metadata = match std::fs::metadata(&path) {
                Ok(metadata) if metadata.is_file() => metadata,
                Ok(_) => {
                    tracing::warn!(%filename, "skipping non-file local GGUF entry");
                    continue;
                }
                Err(err) => {
                    tracing::warn!(%filename, error = %err, "could not inspect local GGUF metadata");
                    continue;
                }
            };
            #[cfg(windows)]
            let identity = match local_model_file_identity(
                &dir,
                &filename,
                &state.model_delete_nonce,
            ) {
                Ok(identity) => identity,
                Err(err) => {
                    tracing::warn!(%filename, error = %err, "could not inspect local GGUF identity");
                    None
                }
            };
            #[cfg(windows)]
            let delete_token = identity
                .as_ref()
                .and_then(|identity| expose_delete_tokens.then(|| identity.delete_token.clone()))
                .flatten();
            #[cfg(not(windows))]
            let delete_token = None;
            #[cfg(windows)]
            let deletable_identity = identity
                .as_ref()
                .filter(|identity| identity.delete_token.is_some());
            #[cfg(windows)]
            let size_bytes =
                deletable_identity.map_or_else(|| metadata.len(), |identity| identity.size_bytes);
            #[cfg(not(windows))]
            let size_bytes = metadata.len();
            let metadata_mtime_secs = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs())
                .unwrap_or_default();
            #[cfg(windows)]
            let mtime_secs = deletable_identity.map_or(metadata_mtime_secs, |identity| {
                (identity.modified_nanos / 1_000_000_000) as u64
            });
            #[cfg(not(windows))]
            let mtime_secs = metadata_mtime_secs;

            // Reuse cached metadata facts when the file is unchanged; only the cheap
            // receipt-present check (which a smoke run can flip) is always live.
            let cached = {
                let cache = local_meta_cache().lock().unwrap();
                cache.get(&filename).cloned()
            };
            let meta = match cached {
                Some(c) if c.mtime_secs == mtime_secs && c.size_bytes == size_bytes => c,
                _ => {
                    let mut c = CachedLocalMeta {
                        mtime_secs,
                        size_bytes,
                        architecture: None,
                        quantization: None,
                        tokenizer_kind: None,
                        admitted: false,
                        admission_reason: None,
                        oracle_qualified: false,
                        chat_capable: false,
                        context_length: None,
                    };
                    match read_metadata(&path) {
                        Ok(gguf) => {
                            let quant = crate::runnable::headline_quant_of(&gguf);
                            c.quantization = Some(quant.clone());
                            // Model capabilities (system-independent): a chat template
                            // means instruction-tuned chat; context_length is the
                            // trained window.
                            c.chat_capable =
                                gguf.metadata_string("tokenizer.chat_template").is_some();
                            c.context_length = gguf
                                .architecture()
                                .and_then(|a| gguf.metadata_u32(&format!("{a}.context_length")));
                            match crate::runnable::admit(&gguf) {
                                Ok(ok) => {
                                    c.admitted = true;
                                    c.tokenizer_kind = Some(
                                        match ok.tokenizer {
                                            crate::runnable::TokenizerFamily::Spm => "llama_spm",
                                            crate::runnable::TokenizerFamily::Bpe => "gpt2_bpe",
                                        }
                                        .to_string(),
                                    );
                                    c.oracle_qualified =
                                        crate::runnable::oracle_qualified(&ok.architecture, &quant);
                                    c.architecture = Some(ok.architecture);
                                }
                                Err(reject) => {
                                    c.architecture = gguf.architecture().map(|s| s.to_string());
                                    c.admission_reason = Some(reject.message);
                                }
                            }
                        }
                        Err(err) => c.admission_reason = Some(format!("GGUF parse failed: {err}")),
                    }
                    local_meta_cache()
                        .lock()
                        .unwrap()
                        .insert(filename.clone(), c.clone());
                    c
                }
            };

            let lane_class = classify_model_lane(meta.architecture.as_deref(), &filename);
            models.push(LocalModelEntry {
                runnable_receipt_present: runnable_smoke_receipt_path(&filename).exists(),
                filename,
                size_bytes,
                delete_token,
                architecture: meta.architecture,
                quantization: meta.quantization,
                tokenizer_kind: meta.tokenizer_kind,
                admitted: meta.admitted,
                admission_reason: meta.admission_reason,
                oracle_qualified: meta.oracle_qualified,
                chat_capable: meta.chat_capable,
                context_length: meta.context_length,
                lane_class,
            });
        }
    }
    Json(LocalModelsResponse {
        models_dir: dir.display().to_string(),
        models,
    })
}

#[derive(Debug, serde::Deserialize)]
struct RunnableReceiptQuery {
    filename: String,
}

/// `GET /api/models/runnable-receipt?filename=<gguf>` â€” return the cached runnable
/// smoke receipt for a model (lane=runnable; attests deterministic execution, not
/// parity). 404 when the model has not been smoke-admitted yet.
async fn runnable_receipt(
    axum::extract::Query(q): axum::extract::Query<RunnableReceiptQuery>,
) -> Response {
    let path = runnable_smoke_receipt_path(&q.filename);
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(mut value) => {
                // Normalize to the bare receipt: older cached artifacts wrap it as
                // `{ "receipt": {...} }`; the endpoint always returns the receipt.
                if let Some(inner) = value.get_mut("receipt").map(std::mem::take) {
                    value = inner;
                }
                (StatusCode::OK, Json(value)).into_response()
            }
            Err(err) => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "runnable_receipt_unreadable",
                format!("cached receipt is not valid JSON: {err}"),
                None,
            ),
        },
        Err(_) => api_error(
            StatusCode::NOT_FOUND,
            "runnable_receipt_not_found",
            format!("no runnable smoke receipt for {}", q.filename),
            None,
        ),
    }
}

#[derive(Debug, serde::Deserialize)]
struct RunnableSmokeRequest {
    filename: String,
}

/// `POST /api/models/runnable-smoke {filename}` â€” run smoke-admission on a local
/// model (admit -> load -> forward sanity -> coherence), cache + return the runnable
/// receipt on pass. User-initiated; the model joins the Compatible section after.
/// CPU-heavy (~minute) so it runs on a blocking thread.
async fn run_runnable_smoke(
    State(state): State<AppState>,
    Json(req): Json<RunnableSmokeRequest>,
) -> Response {
    let filename = req.filename;
    if filename.is_empty()
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains("..")
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_filename",
            "filename must be a bare GGUF name resolved under the configured models directory"
                .to_string(),
            None,
        );
    }
    let path = state.models_dir.join(&filename);
    if !path.exists() {
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!(
                "{filename} is not present in {}",
                state.models_dir.display()
            ),
            None,
        );
    }
    let path_str = path.to_string_lossy().to_string();
    let smoke_filename = filename.clone();
    let reader = state.model_file_lifecycle.clone().read_owned().await;
    let result = tokio::task::spawn_blocking(move || {
        let _reader = reader;
        let report = crate::runnable::smoke_admit(&path_str)?;
        let out = runnable_smoke_receipt_path(&smoke_filename);
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(&report.receipt) {
            let _ = std::fs::write(&out, json);
        }
        Ok::<_, BackendError>(report)
    })
    .await;
    match result {
        Ok(Ok(report)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "passed": true,
                "architecture": report.architecture,
                "quant": report.quant,
                "logit_min": report.logit_min,
                "logit_max": report.logit_max,
                "generated_text": report.generated_text,
                "receipt": report.receipt,
            })),
        )
            .into_response(),
        Ok(Err(err)) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "smoke_admission_failed",
            err.to_string(),
            None,
        ),
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "smoke_task_failed",
            "smoke-admission task panicked".to_string(),
            None,
        ),
    }
}

/// Number of Hugging Face repos a single experimental search inspects (each repo's
/// tree is one extra network round-trip, so this also bounds search latency).
const HF_SEARCH_LIMIT: usize = 15;

/// Lane classification for a downloaded/loaded model, driven only by real GGUF
/// metadata (`general.architecture`) and the exact-artifact supported-row check â€”
/// never a filename guess of the architecture. Drives experimental UI copy only; it
/// never promotes a row or widens what `LlamaModelConfig::from_gguf` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLaneClass {
    /// Exact supported row (asserted by `/api/capabilities`) â€” the full supported lane.
    Supported,
    /// Architecture is implemented but this is NOT a supported row: attemptable,
    /// unverified, no parity claim.
    ExperimentalImplemented,
    /// Architecture is not in the implemented set â€” fails closed at load.
    Unsupported,
}

/// The `id`s of `/api/capabilities` compatibility rows whose status is `supported*`.
/// Memoized â€” these are static contract literals. Reads the SAME rows the contract
/// publishes, so this introduces no second ledger.
fn supported_compatibility_row_ids() -> &'static std::collections::HashSet<&'static str> {
    static IDS: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    IDS.get_or_init(|| {
        capabilities_response_with_plan(None)
            .model_compatibility
            .into_iter()
            .filter(|row| row.status == "supported" || row.status.starts_with("supported_"))
            .map(|row| row.id)
            .collect()
    })
}

/// Supported exact rows whose GGUF artifact is NOT a curated-catalog download —
/// in-house imatrix requants (and the side-loaded Q8_0) with no HF catalog
/// source. Exact filename → compatibility row id; the row must still be
/// `supported_*` in the ledger, so this stays fail-closed at the same trust
/// level as the curated-catalog path (exact-artifact filename match).
const NON_CATALOG_SUPPORTED_ARTIFACTS: &[(&str, &str)] = &[
    ("ornith-1.0-9b-Q8_0.gguf", "Ornith 1.0 9B"),
    ("ornith-1.0-9b-Q4_K_M.gguf", "ornith_1_0_9b_q4_k_m"),
    ("ornith-1.0-9b-Q3_K_M.gguf", "ornith_1_0_9b_q3_k_m"),
    // NB: Qwen3-4B-Q4_K_M.gguf is intentionally NOT here -- it is an official
    // Qwen/Qwen3-4B-GGUF upload, so it lives in curated_catalog() and is covered
    // by the curated-catalog branch of filename_is_supported_exact_row.
];

/// True when `filename` is the exact GGUF artifact of a curated row whose
/// `catalog_id` is a `supported_*` compatibility row, or an allowlisted
/// non-catalog artifact of a `supported_*` row. The ledger is exact-artifact
/// gated, so an exact-filename match is the honest server-side "is this a supported
/// row?" test. Deliberately conservative: a supported model loaded under a
/// non-curated filename classifies as experimental, never falsely as supported.
fn filename_is_supported_exact_row(filename: &str) -> bool {
    let supported = supported_compatibility_row_ids();
    curated_catalog()
        .iter()
        .any(|c| c.filename == filename && supported.contains(c.catalog_id))
        || NON_CATALOG_SUPPORTED_ARTIFACTS
            .iter()
            .any(|(artifact, row_id)| *artifact == filename && supported.contains(row_id))
}

/// Classify a model from real header metadata. `architecture` is the parsed
/// `general.architecture` (NOT a filename guess); `filename` identifies the exact
/// artifact for the supported-row check.
fn classify_model_lane(architecture: Option<&str>, filename: &str) -> ModelLaneClass {
    match architecture {
        Some(arch) if crate::model::is_implemented_architecture(arch) => {
            if filename_is_supported_exact_row(filename) {
                ModelLaneClass::Supported
            } else {
                ModelLaneClass::ExperimentalImplemented
            }
        }
        _ => ModelLaneClass::Unsupported,
    }
}

/// Classify a loaded model from its real GGUF metadata + exact artifact path.
fn classify_loaded_model(model: &LoadedModel) -> ModelLaneClass {
    let filename = model
        .path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or_default();
    classify_model_lane(model.gguf.architecture(), filename)
}

/// Stable, frontend-switchable `error.code` for a typed backend failure. The
/// human message (which already carries the offending architecture/quant and any
/// dedicated-lane redirect, e.g. `camelid diffusion-gemma-chat`) travels separately
/// in `error.message`.
fn backend_error_code(err: &BackendError) -> &'static str {
    match err {
        BackendError::UnsupportedModelArchitecture(_) => "unsupported_model_architecture",
        BackendError::InvalidModelMetadata(_) => "invalid_model_metadata",
        BackendError::UnsupportedGguf(_) => "unsupported_gguf",
        BackendError::InvalidGguf(_) => "invalid_gguf",
        BackendError::UnsupportedTokenizer(_) => "unsupported_tokenizer",
        BackendError::InvalidTokenizerMetadata(_) => "invalid_tokenizer_metadata",
        BackendError::UnsupportedTensorType(_) => "unsupported_tensor_type",
        BackendError::InvalidTensorData(_) => "invalid_tensor_data",
        BackendError::Io { .. } => "model_io_error",
        BackendError::ModelOperationInProgress => "model_operation_in_progress",
        _ => "invalid_model",
    }
}

/// The Models-tab catalog: a **curated** group (pinned, known-good rows, always
/// first) followed, when the user is searching, by an **experimental** group of
/// live Hugging Face results. Experimental rows are advisory-only â€” `oracle_qualified`
/// is forced false and the frontend marks them unverified â€” so live browse can never
/// widen what Camelid claims. A non-trivial `query` (â‰¥2 chars) triggers the HF search;
/// a network failure degrades silently to curated-only.
async fn get_catalog(
    axum::extract::Query(q): axum::extract::Query<CatalogQuery>,
) -> Json<CatalogResponse> {
    let query = q.query.unwrap_or_default();
    let trimmed = query.trim();

    // Curated group: filter by query exactly as before, always emitted first.
    // The fit verdict is host-specific, so probe (cached) once and annotate each row.
    //
    // Deliberate: the badge uses the cached startup snapshot, while the load-time
    // guard (`fit_preload_guard`) re-probes live. So after a model loads and consumes
    // VRAM, a badge may still read "fits" while the guard now rejects with 422. The
    // badge is a static capacity hint, not a live reservation; the guard is
    // authoritative. Re-probing on every GET would re-init CUDA per request.
    let hw = crate::capability::HardwareProfile::cached();
    let curated: Vec<CatalogItemView> = curated_catalog()
        .iter()
        .filter(|item| {
            trimmed.is_empty() || {
                let qs = trimmed.to_lowercase();
                item.name.to_lowercase().contains(&qs)
                    || item.repo_id.to_lowercase().contains(&qs)
                    || item.filename.to_lowercase().contains(&qs)
            }
        })
        .map(|item| CatalogItemView::from_curated(item, hw))
        .collect();

    // Serving this page is side-effect-free for the *curated* rows: their dims are
    // warmed once at server startup via `fit_dims::start_background`, never here. The
    // Hugging Face search branch below is not side-effect-free — it schedules a
    // bounded, de-duplicated set of background header fetches (see HF_DIMS_WARM_LIMIT).
    let mut items = curated;
    let mut next_cursor = None;

    // Experimental group: live Hugging Face results, only when actively searching.
    if trimmed.len() >= 2 {
        match crate::hf_browse::search_gguf(trimmed, HF_SEARCH_LIMIT, q.cursor.as_deref()).await {
            Ok(page) => {
                next_cursor = page.next_cursor;
                // A file already pinned in the curated group is shown there (vetted),
                // not duplicated as a raw experimental row.
                let curated_files: std::collections::HashSet<(String, String)> = curated_catalog()
                    .iter()
                    .map(|c| (c.repo_id.to_string(), c.filename.to_string()))
                    .collect();
                // Schedule header-dim fetches for the top experimental results so a
                // random Hugging Face model shows an honest fit on the next render.
                // The resolver de-dupes and globally rate-limits; we still cap the
                // per-query scheduling so one search can't enqueue 100+ models.
                const HF_DIMS_WARM_LIMIT: usize = 5;
                let mut scheduled = 0usize;
                for f in &page.files {
                    if scheduled >= HF_DIMS_WARM_LIMIT {
                        break;
                    }
                    if curated_files.contains(&(f.repo_id.clone(), f.filename.clone())) {
                        continue;
                    }
                    crate::fit_dims::global().schedule(
                        f.repo_id.clone(),
                        f.filename.clone(),
                        f.size_bytes,
                    );
                    scheduled += 1;
                }
                items.extend(
                    page.files
                        .into_iter()
                        .filter(|f| {
                            !curated_files.contains(&(f.repo_id.clone(), f.filename.clone()))
                        })
                        .map(|f| CatalogItemView::from_hf(f, hw)),
                );
            }
            // Offline / Hub error: keep curated-only rather than failing the page.
            Err(err) => eprintln!("hugging face browse unavailable: {err}"),
        }
    }

    Json(CatalogResponse { items, next_cursor })
}

/// On-demand exact capacity check for one live Hugging Face GGUF.
///
/// A catalog page can only warm a handful of rows in the background
/// (`HF_DIMS_WARM_LIMIT`), so most experimental rows render with no fit claim at
/// all. That cap is invisible to the user, who just sees "unknown" and has no way
/// to learn more about the one model they actually care about. This endpoint is
/// that way: it range-fetches the model's GGUF header, derives its exact KV
/// dimensions, and returns a real verdict for that row.
///
/// Never a guess. If the header cannot be fetched, or parses but is not a dense
/// LLaMA-family model, the answer is `Unknown` with `"unknown"` confidence — the
/// same thing the row already showed, not a size-based estimate dressed up as a
/// check. `checked` then says whether that silence is final or worth retrying.
/// Idempotent and cached: asking twice costs one fetch, and the request shares the
/// resolver's global concurrency limit, dedup and backoff so it cannot be used to
/// drive an unbounded fetch storm.
async fn check_catalog_fit(Json(req): Json<CatalogFitRequest>) -> Response {
    // Same bare-`.gguf`-name rule the download path enforces: this filename is
    // interpolated into a Hub resolve URL.
    if !validate_local_model_filename(&req.filename) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_filename",
            "filename must be one bare .gguf name".to_string(),
            Some("filename"),
        );
    }
    if !crate::fit_dims::is_safe_hf_component(&req.repo_id) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_repo_id",
            "repo_id must be a Hugging Face repository id".to_string(),
            Some("repo_id"),
        );
    }
    if req.size_bytes == 0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_size",
            "size_bytes must be the file's real size in bytes".to_string(),
            Some("size_bytes"),
        );
    }

    // LIVE probe, deliberately not `HardwareProfile::cached()`.
    //
    // `cached()` is a `OnceLock` — memory is read once at startup and frozen for the
    // process. The catalog LIST endpoint accepts that (re-probing per GET would
    // re-init CUDA on every request, for every row). This endpoint is the opposite
    // case: one explicit, user-initiated question about one model, which already
    // costs a network header fetch.
    //
    // It is also a correctness requirement. The refusal this powers tells the user
    // to free memory and check again; against a frozen snapshot that instruction
    // would be a lie, because freeing memory could never change the answer. Probing
    // live also makes a "fits" here agree with the authoritative load guard, which
    // probes live too.
    let hw = match tokio::task::spawn_blocking(crate::capability::HardwareProfile::detect).await {
        Ok(profile) => profile,
        // A panicking probe must not fail the request; fall back to the startup
        // snapshot, which is still a real measurement of this machine.
        Err(_) => crate::capability::HardwareProfile::cached().clone(),
    };
    let resolution = crate::fit_dims::global()
        .resolve_now(req.repo_id.clone(), req.filename.clone(), req.size_bytes)
        .await;
    let (fit, fit_confidence) = match resolution.dims() {
        Some(dims) => (
            crate::fit::assess(&hw, &exact_footprint_for(req.size_bytes, dims, &hw)),
            "exact",
        ),
        // A curated row whose dimensions cannot be read is *listed* with the coarse
        // size pad (see `curated_footprint`), so answering `Unknown` here would make
        // "Re-check" erase a verdict the page already shows — on exactly the rows
        // that need it, since "close some applications, then Re-check" is the remedy
        // we print for them. Same rule as the listing, so the two cannot disagree.
        None if is_curated_artifact(&req.repo_id, &req.filename) => (
            crate::fit::assess(&hw, &crate::fit::advisory_footprint(req.size_bytes)),
            "approx",
        ),
        // A live Hugging Face row gets no such pad: it is shown as `unknown` when
        // unmeasured, and a size-based guess dressed as the result of an explicit
        // check would be worse than the silence it replaced.
        None => (crate::fit::FitVerdict::Unknown, "unknown"),
    };
    Json(CatalogFitResponse {
        repo_id: req.repo_id,
        filename: req.filename,
        fit,
        fit_confidence,
        checked: resolution.is_settled(),
    })
    .into_response()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveDownload {
    pub id: String,
    pub repo_id: String,
    pub filename: String,
    pub continuation_mode: String,
    pub total_bytes: u64,
    pub bytes_downloaded: u64,
    pub status: &'static str,
    #[serde(skip)]
    pub child_pid: Option<u32>,
    #[serde(skip)]
    pub finished_at: Option<std::time::Instant>,
}

#[derive(Debug, serde::Deserialize)]
pub struct InstallCatalogRequest {
    pub catalog_id: String,
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub continuation_mode: Option<String>,
}

static ACTIVE_DOWNLOADS: OnceLock<Mutex<HashMap<String, ActiveDownload>>> = OnceLock::new();
const TERMINAL_DOWNLOAD_RETENTION: Duration = Duration::from_secs(5 * 60);

fn active_downloads_map() -> &'static Mutex<HashMap<String, ActiveDownload>> {
    ACTIVE_DOWNLOADS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn download_destination_key(filename: &str) -> String {
    #[cfg(windows)]
    {
        filename.to_lowercase()
    }
    #[cfg(not(windows))]
    {
        filename.to_string()
    }
}

async fn install_catalog_model(
    State(state): State<AppState>,
    Json(req): Json<InstallCatalogRequest>,
) -> Response {
    let _reader = state.model_file_lifecycle.read().await;
    if !validate_local_model_filename(&req.filename) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_filename",
            "catalog filename must be one bare .gguf name".to_string(),
            Some("filename"),
        );
    }
    let continuation_mode = match req.continuation_mode.as_deref().unwrap_or("download") {
        mode @ ("download" | "smoke" | "start") => mode.to_string(),
        _ => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_continuation_mode",
                "continuation_mode must be download, smoke, or start".to_string(),
                Some("continuation_mode"),
            )
        }
    };
    let mut map = active_downloads_map().lock().unwrap();
    if map
        .get(&req.catalog_id)
        .is_some_and(|existing| existing.status != "downloading")
    {
        map.remove(&req.catalog_id);
    }
    if let Some(existing) = map.get(&req.catalog_id) {
        if existing.repo_id != req.repo_id || existing.filename != req.filename {
            return api_error(
                StatusCode::CONFLICT,
                "download_identity_conflict",
                "this catalog id is already downloading a different artifact".to_string(),
                Some("catalog_id"),
            );
        }
        if existing.continuation_mode != continuation_mode {
            return api_error(
                StatusCode::CONFLICT,
                "download_continuation_conflict",
                "this catalog download is already running with a different continuation"
                    .to_string(),
                Some("continuation_mode"),
            );
        }
        return api_error(
            StatusCode::CONFLICT,
            "download_already_running",
            "this catalog download is already running".to_string(),
            Some("catalog_id"),
        );
    }
    let destination_key = download_destination_key(&req.filename);
    if map.values().any(|download| {
        download.status == "downloading"
            && download_destination_key(&download.filename) == destination_key
    }) {
        return api_error(
            StatusCode::CONFLICT,
            "download_destination_busy",
            "another download is already writing this model filename".to_string(),
            Some("filename"),
        );
    }

    // Download into the configured models dir — the same directory the
    // local-models scan reads — so an installed model is visible to the scan
    // whatever CWD the server process inherited.
    std::fs::create_dir_all(&state.models_dir).ok();
    let dest_path = state.models_dir.join(&req.filename).display().to_string();
    // Download into a `.part` file and only promote it to the final path once curl
    // exits successfully. The loadable GGUF therefore never exists until the
    // download is genuinely complete, so a half-downloaded model cannot be loaded.
    let part_path = format!("{dest_path}.part");
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        req.repo_id, req.filename
    );

    // `-f` makes curl FAIL on an HTTP error (404/403/â€¦) instead of writing the
    // error page to the output file and exiting 0 â€” which previously looked like an
    // instant successful download.
    match std::process::Command::new("curl")
        // Resilience flags for large multi-GB pulls over flaky/throttled CDNs:
        //   --speed-limit/--speed-time: abort (exit 28) if throughput stays below
        //     1 KiB/s for 30s. Without this, a silently stalled-but-still-Established
        //     TCP connection makes curl wait on a dead stream forever, so the download
        //     freezes mid-file and never recovers (the bug this fixes).
        //   --retry/--retry-delay/--retry-all-errors: reconnect on transient errors
        //     AND on that speed-abort (--retry-all-errors covers exit 28); `-C -`
        //     resumes from the existing `.part` offset on each retry, so no bytes are
        //     re-downloaded.
        //   --connect-timeout caps a dead connect attempt.
        .args([
            "-f",
            "-L",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "-o",
            &part_path,
            &url,
        ])
        .spawn()
    {
        Ok(child) => {
            let pid = child.id();
            let download = ActiveDownload {
                id: req.catalog_id.clone(),
                repo_id: req.repo_id.clone(),
                filename: req.filename.clone(),
                continuation_mode,
                total_bytes: req.size_bytes,
                bytes_downloaded: 0,
                status: "downloading",
                child_pid: Some(pid),
                finished_at: None,
            };
            map.insert(req.catalog_id.clone(), download);

            let catalog_id_clone = req.catalog_id.clone();
            let part_path_clone = part_path.clone();
            let dest_path_clone = dest_path.clone();
            let lifecycle = state.model_file_lifecycle.clone();
            tokio::spawn(async move {
                let mut child = child;
                let succeeded = matches!(child.wait(), Ok(status) if status.success());
                // Completion is the curl exit code AND a successful promote of the
                // .part file to the final path, never a size heuristic. The map lock
                // is held across the promote decision so a cancel cannot race the
                // rename: cancel removes the entry, and an untracked (canceled)
                // download must never promote, whatever curl's exit code says.
                let _reader = lifecycle.read().await;
                let mut map = active_downloads_map().lock().unwrap();
                let still_tracked = map.contains_key(&catalog_id_clone);
                let status = finalize_download_artifact(
                    succeeded,
                    still_tracked,
                    &part_path_clone,
                    &dest_path_clone,
                );
                if let Some(dl) = map.get_mut(&catalog_id_clone) {
                    dl.status = status;
                    dl.finished_at = Some(std::time::Instant::now());
                }
            });

            (StatusCode::OK, "Download started").into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to start curl: {}", err),
        )
            .into_response(),
    }
}

async fn get_catalog_downloads(State(state): State<AppState>) -> Json<Vec<ActiveDownload>> {
    let mut map = active_downloads_map().lock().unwrap();
    let mut to_remove = Vec::new();
    for (id, dl) in map.iter_mut() {
        if dl.status == "completed" || dl.status == "failed" {
            if dl
                .finished_at
                .is_some_and(|finished| finished.elapsed() >= TERMINAL_DOWNLOAD_RETENTION)
            {
                to_remove.push(id.clone());
            }
            continue;
        }

        // Progress comes from the in-flight `.part` file. Completion is driven
        // ONLY by curl's exit code (set in the spawn task above), never by a size
        // comparison against the catalog's approximate `size_bytes`, which could
        // flip a download to "completed" before it actually finished.
        let part_path = format!("{}.part", state.models_dir.join(&dl.filename).display());
        if let Ok(metadata) = std::fs::metadata(&part_path) {
            dl.bytes_downloaded = metadata.len();
        }
    }

    let result = map.values().cloned().collect::<Vec<_>>();

    for id in to_remove {
        map.remove(&id);
    }

    Json(result)
}

/// Decide a finished download's fate. Promotion requires BOTH a successful curl
/// exit AND the download still being tracked: a canceled download (entry removed
/// from the map) must never promote its partial file to a loadable GGUF. Every
/// non-promoted outcome removes the `.part` so nothing loadable is left behind.
/// Returns the terminal status to record for a still-tracked download.
fn finalize_download_artifact(
    succeeded: bool,
    still_tracked: bool,
    part_path: &str,
    dest_path: &str,
) -> &'static str {
    let promoted = succeeded && still_tracked && std::fs::rename(part_path, dest_path).is_ok();
    if !promoted {
        std::fs::remove_file(part_path).ok();
    }
    if promoted {
        "completed"
    } else {
        "failed"
    }
}

/// Terminate a spawned download process by PID. The `Child` handle is owned by the
/// wait-task (blocked in `wait()`), so cancellation signals the process from the
/// outside; the wait-task then observes the exit and cleans up the partial file.
/// `kill` does not exist on Windows service PATHs — use `taskkill` there (`/T`
/// also ends curl's own children, `/F` forces termination).
fn kill_download_process(pid: u32) {
    #[cfg(windows)]
    let mut kill_cmd = {
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/PID", &pid.to_string(), "/T", "/F"]);
        cmd
    };
    #[cfg(not(windows))]
    let mut kill_cmd = {
        let mut cmd = std::process::Command::new("kill");
        cmd.arg(pid.to_string());
        cmd
    };
    kill_cmd
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok();
}

#[derive(Debug, serde::Deserialize)]
pub struct CancelDownloadRequest {
    pub id: String,
}

async fn cancel_catalog_download(Json(req): Json<CancelDownloadRequest>) -> Response {
    let mut map = active_downloads_map().lock().unwrap();
    if map
        .get(&req.id)
        .is_some_and(|download| download.status != "downloading")
    {
        return api_error(
            StatusCode::CONFLICT,
            "download_already_completed",
            "the download has already finished and cannot be canceled".to_string(),
            Some("id"),
        );
    }
    if let Some(dl) = map.remove(&req.id) {
        if let Some(pid) = dl.child_pid {
            kill_download_process(pid);
        }
        // The `.part` file is cleaned by the wait-task once curl actually exits
        // (curl may still hold it open right now). Removing the entry here is what
        // guarantees the wait-task can never promote it. An already-completed
        // download keeps its file: cancel is not delete.
        (StatusCode::OK, "Download canceled").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Download not found").into_response()
    }
}

async fn acknowledge_catalog_download(
    State(_state): State<AppState>,
    _headers: HeaderMap,
    Json(req): Json<CancelDownloadRequest>,
) -> Response {
    let mut map = active_downloads_map().lock().unwrap();
    if map
        .get(&req.id)
        .is_some_and(|download| download.status == "downloading")
    {
        return api_error(
            StatusCode::CONFLICT,
            "download_still_running",
            "the download cannot be acknowledged before it finishes".to_string(),
            Some("id"),
        );
    }
    map.remove(&req.id);
    StatusCode::NO_CONTENT.into_response()
}

#[cfg(all(test, windows))]
mod local_model_delete_tests {
    use super::{active_downloads_map, router_with_state, ActiveDownload, AppState};
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
        Router,
    };
    use serde_json::Value;
    use tower::ServiceExt;

    static DELETE_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[test]
    fn management_authorization_requires_exact_loopback_origin() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("sec-fetch-site", "same-origin".parse().unwrap());
        assert!(super::local_management_request_allowed(&headers));

        let mut headers = axum::http::HeaderMap::new();
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("origin", "http://127.0.0.1:8181".parse().unwrap());
        assert!(super::local_management_request_allowed(&headers));
        headers.insert("origin", "http://127.0.0.1:4177".parse().unwrap());
        assert!(!super::local_management_request_allowed(&headers));
        headers.insert("origin", "http://[::1]:8181".parse().unwrap());
        headers.insert("host", "[::1]:8181".parse().unwrap());
        assert!(super::local_management_request_allowed(&headers));
        headers.insert("host", "127.0.0.1:8181".parse().unwrap());
        headers.insert("origin", "https://example.com".parse().unwrap());
        assert!(!super::local_management_request_allowed(&headers));
        headers.insert("sec-fetch-site", "same-origin".parse().unwrap());
        assert!(!super::local_management_request_allowed(&headers));
        headers.insert("origin", "http://127.0.0.1:4177".parse().unwrap());
        headers.insert("host", "attacker.example:8181".parse().unwrap());
        assert!(!super::local_management_request_allowed(&headers));
        headers.clear();
        assert!(!super::local_management_request_allowed(&headers));
    }

    async fn request(
        app: Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder()
            .method(method)
            .uri(uri)
            .header("host", "127.0.0.1:8181")
            .header("sec-fetch-site", "same-origin");
        let request_body = match body {
            Some(value) => {
                builder = builder
                    .header("content-type", "application/json")
                    .header("sec-fetch-site", "same-origin");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        let response = app
            .oneshot(builder.body(request_body).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, body)
    }

    fn state_for(dir: &std::path::Path) -> AppState {
        AppState {
            models_dir: dir.to_path_buf(),
            allow_local_model_delete: true,
            ..AppState::default()
        }
    }

    async fn scanned_token(app: Router, filename: &str) -> String {
        let (status, listing) = request(app, "GET", "/api/models/local", None).await;
        assert_eq!(status, StatusCode::OK);
        listing["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["filename"] == filename)
            .and_then(|entry| entry["delete_token"].as_str())
            .expect("scan-issued delete token")
            .to_string()
    }

    #[tokio::test]
    async fn endpoint_deletes_only_the_identity_bound_sibling() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target.gguf");
        let sibling = temp.path().join("sibling.gguf");
        std::fs::write(&target, b"target bytes").unwrap();
        std::fs::write(&sibling, b"sibling bytes").unwrap();
        let app = router_with_state(state_for(temp.path()));
        let token = scanned_token(app.clone(), "target.gguf").await;

        let (status, body) = request(
            app,
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "target.gguf",
                "delete_token": token,
            })),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["bytes_freed"], 12);
        assert!(!target.exists());
        assert_eq!(std::fs::read(sibling).unwrap(), b"sibling bytes");
    }

    #[tokio::test]
    async fn endpoint_rejects_a_stale_token_after_file_replacement() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("replace.gguf");
        std::fs::write(&target, b"first identity").unwrap();
        let app = router_with_state(state_for(temp.path()));
        let token = scanned_token(app.clone(), "replace.gguf").await;
        std::fs::remove_file(&target).unwrap();
        std::fs::write(&target, b"other identity").unwrap();

        let (status, body) = request(
            app,
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "replace.gguf",
                "delete_token": token,
            })),
        )
        .await;

        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_changed");
        assert!(target.exists());
    }

    #[tokio::test]
    async fn endpoint_rejects_traversal_loaded_state_and_active_downloads() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("guarded.gguf");
        std::fs::write(&target, b"guarded bytes").unwrap();
        let state = state_for(temp.path());
        let app = router_with_state(state.clone());
        let token = scanned_token(app.clone(), "guarded.gguf").await;

        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "../guarded.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        *state.active_model_id.write().await = Some("active-model".to_string());
        let token = scanned_token(app.clone(), "guarded.gguf").await;
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "guarded.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_in_use");
        *state.active_model_id.write().await = None;

        let download_id = format!("delete-test-{}", uuid::Uuid::new_v4());
        active_downloads_map().lock().unwrap().insert(
            download_id.clone(),
            ActiveDownload {
                id: download_id.clone(),
                repo_id: "test/repo".to_string(),
                filename: "other.gguf".to_string(),
                continuation_mode: "download".to_string(),
                total_bytes: 1,
                bytes_downloaded: 0,
                status: "downloading",
                child_pid: None,
                finished_at: None,
            },
        );
        let token = scanned_token(app.clone(), "guarded.gguf").await;
        let (status, body) = request(
            app,
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "guarded.gguf",
                "delete_token": token,
            })),
        )
        .await;
        active_downloads_map().lock().unwrap().remove(&download_id);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_download_in_progress");
        assert!(target.exists());
    }

    #[tokio::test]
    async fn endpoint_rejects_while_file_readers_are_active_then_allows_retry() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("leased.gguf");
        std::fs::write(&target, b"leased bytes").unwrap();
        let state = state_for(temp.path());
        let app = router_with_state(state.clone());
        let token = scanned_token(app.clone(), "leased.gguf").await;
        let reader = state.model_file_lifecycle.read().await;
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "leased.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_operation_in_progress");
        assert!(target.exists());
        drop(reader);

        let token = scanned_token(app.clone(), "leased.gguf").await;
        let (status, _) = request(
            app,
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "leased.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn prepared_generation_lease_blocks_deletion_until_dropped() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("generating.gguf");
        std::fs::write(&target, b"generation bytes").unwrap();
        let state = state_for(temp.path());
        let app = router_with_state(state.clone());
        let token = scanned_token(app.clone(), "generating.gguf").await;
        let lease = state.model_file_lifecycle.clone().read_owned().await;
        let mut prepared = super::tests::orphan_test_prepared("generating.gguf");
        prepared._model_file_lease = Some(lease);

        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "generating.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "model_operation_in_progress");
        assert!(target.exists());

        drop(prepared);
        let token = scanned_token(app.clone(), "generating.gguf").await;
        let (status, _) = request(
            app,
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "generating.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(!target.exists());
    }

    #[tokio::test]
    async fn endpoint_rejects_missing_same_origin_or_non_loopback_capability() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("auth.gguf");
        std::fs::write(&target, b"auth bytes").unwrap();
        let allowed_state = state_for(temp.path());
        let app = router_with_state(allowed_state.clone());
        let token = scanned_token(app.clone(), "auth.gguf").await;
        let request_without_origin = Request::builder()
            .method("POST")
            .uri("/api/models/local/delete")
            .header("host", "127.0.0.1:8181")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "filename": "auth.gguf",
                    "delete_token": token,
                })
                .to_string(),
            ))
            .unwrap();
        let response = app.oneshot(request_without_origin).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(target.exists());

        let mut disabled_state = allowed_state;
        disabled_state.allow_local_model_delete = false;
        let token = super::local_model_file_identity(
            temp.path(),
            "auth.gguf",
            &disabled_state.model_delete_nonce,
        )
        .unwrap()
        .unwrap()
        .delete_token
        .unwrap();
        let (status, body) = request(
            router_with_state(disabled_state),
            "POST",
            "/api/models/local/delete",
            Some(serde_json::json!({
                "filename": "auth.gguf",
                "delete_token": token,
            })),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(body["error"]["code"], "local_management_forbidden");
        assert!(target.exists());
    }

    #[tokio::test]
    async fn scan_lists_hardlinked_models_without_deletion_tokens() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("hardlink.gguf");
        let sibling = temp.path().join("alias.gguf");
        std::fs::write(&target, b"shared bytes").unwrap();
        std::fs::hard_link(&target, &sibling).unwrap();
        let (status, listing) = request(
            router_with_state(state_for(temp.path())),
            "GET",
            "/api/models/local",
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let models = listing["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models
            .iter()
            .all(|entry| entry.get("delete_token").is_none()));
        assert!(target.exists());
        assert!(sibling.exists());
    }

    #[tokio::test]
    async fn scan_includes_uppercase_gguf_extensions() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("UPPER.GGUF"), b"uppercase bytes").unwrap();
        let token = scanned_token(router_with_state(state_for(temp.path())), "UPPER.GGUF").await;
        assert_eq!(token.len(), 64);
    }

    #[tokio::test]
    async fn scan_withholds_delete_tokens_from_cross_origin_clients() {
        let _test = DELETE_TEST_LOCK.lock().await;
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("private.gguf"), b"private bytes").unwrap();
        let app = router_with_state(state_for(temp.path()));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/models/local")
                    .header("host", "127.0.0.1:8181")
                    .header("origin", "http://127.0.0.1:4177")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let listing: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert!(listing["models"][0].get("delete_token").is_none());
        assert!(temp.path().join("private.gguf").exists());
    }
}

#[cfg(all(test, not(windows)))]
mod non_windows_model_delete_tests {
    use super::{router_with_state, AppState};
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn scan_omits_tokens_and_delete_reports_unsupported() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("model.gguf"), b"fixture").unwrap();
        let state = AppState {
            models_dir: temp.path().to_path_buf(),
            allow_local_model_delete: true,
            ..AppState::default()
        };
        let app = router_with_state(state);

        let listing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/models/local")
                    .header("host", "127.0.0.1:8181")
                    .header("sec-fetch-site", "same-origin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listing.status(), StatusCode::OK);
        let listing: serde_json::Value =
            serde_json::from_slice(&to_bytes(listing.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert!(listing["models"][0].get("delete_token").is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/models/local/delete")
                    .header("host", "127.0.0.1:8181")
                    .header("sec-fetch-site", "same-origin")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "filename": "model.gguf",
                            "delete_token": "0".repeat(64),
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["code"], "model_delete_unsupported");
        assert!(temp.path().join("model.gguf").exists());
    }
}

#[cfg(test)]
mod catalog_fit_tests {
    use super::{curated_catalog, CatalogItem, CatalogItemView};
    use crate::capability::{HardwareProfile, SimdCaps};
    use crate::fit::FitVerdict;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn host(cuda: bool, vram_free: u64, ram_total: u64, ram_free: u64) -> HardwareProfile {
        HardwareProfile {
            cuda_available: cuda,
            cuda_device_count: if cuda { 1 } else { 0 },
            cuda_device_name: None,
            cuda_compute_capability: None,
            cuda_tensor_cores: false,
            cuda_vram_total_bytes: vram_free,
            cuda_vram_free_bytes: vram_free,
            cpu_logical_cores: 8,
            host_ram_total_bytes: ram_total,
            host_ram_free_bytes: ram_free,
            simd: SimdCaps::default(),
        }
    }

    fn row(id: &str) -> CatalogItem {
        curated_catalog()
            .into_iter()
            .find(|c| c.catalog_id == id)
            .expect("known catalog row")
    }

    #[test]
    fn curated_view_carries_task_tags_and_a_verdict() {
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        let item = row("tinyllama_1_1b_chat_q8_0");
        let view = CatalogItemView::from_curated(&item, &hw);
        assert_eq!(view.task_tags, vec!["general".to_string()]);
        // ~1.1 GB model on a 64 GB CPU host → comfortably CPU-only.
        assert_eq!(view.fit, FitVerdict::CpuOnlyOk);
    }

    #[test]
    fn curated_huge_model_wont_fit_tiny_cpu_host() {
        // 8 GB total RAM, no GPU → budget ~ max(80% of 5=4, 25% of 8=2)=4 GB;
        // the ~8.5 GB Llama-3 8B (+overhead) cannot fit.
        let hw = host(false, 0, 8 * GIB, 5 * GIB);
        let item = row("llama3_8b_instruct_q8_0");
        let view = CatalogItemView::from_curated(&item, &hw);
        assert_eq!(view.fit, FitVerdict::WontFit);
    }

    #[test]
    fn experimental_from_hf_is_unknown_and_untagged() {
        let file = crate::hf_browse::HfGgufFile {
            repo_id: "someone/Some-Model-GGUF".to_string(),
            filename: "Some-Model-Q4_K_M.gguf".to_string(),
            size_bytes: 2 * GIB,
            downloads: 10,
            likes: 1,
            architecture: "llama".to_string(),
            quant: "Q4_K_M".to_string(),
        };
        // No cached header dims for this repo → no fit claim (Unknown), untagged.
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        let view = CatalogItemView::from_hf(file, &hw);
        assert_eq!(view.fit, FitVerdict::Unknown);
        assert!(view.task_tags.is_empty());
        assert_eq!(view.group, "experimental");
        assert_eq!(view.fit_confidence, "unknown");
    }

    #[test]
    fn hf_row_gets_an_exact_fit_once_its_header_dims_are_cached() {
        let hw = host(false, 0, 4 * GIB, 3 * GIB);
        let (repo, filename) = ("someone/Huge-Model-GGUF", "Huge-Model-Q8_0.gguf");
        // Cache real dims for a genuinely too-big model → honest exact WontFit even on
        // an unverified HF row (capacity is orthogonal to verification).
        let dims = crate::fit::ModelDims {
            layers: 80,
            kv_heads: 8,
            head_dim: 128,
        };
        crate::fit_dims::global().insert_for_test(repo, filename, dims);
        let file = crate::hf_browse::HfGgufFile {
            repo_id: repo.to_string(),
            filename: filename.to_string(),
            size_bytes: 20 * GIB,
            downloads: 0,
            likes: 0,
            architecture: "llama".to_string(),
            quant: "Q8_0".to_string(),
        };
        let view = CatalogItemView::from_hf(file, &hw);
        assert_eq!(view.fit_confidence, "exact");
        assert_eq!(view.fit, FitVerdict::WontFit);
    }

    #[test]
    fn unknown_host_never_reports_wont_fit_for_any_curated_row() {
        // macOS-style: RAM unprobed (0) and no CUDA → advisory-blind, never WontFit.
        let hw = host(false, 0, 0, 0);
        for item in curated_catalog() {
            let view = CatalogItemView::from_curated(&item, &hw);
            assert_eq!(
                view.fit,
                FitVerdict::Unknown,
                "row {} must be Unknown on an unprobed host",
                view.catalog_id
            );
        }
    }

    #[test]
    fn every_curated_row_is_tagged_within_the_allowed_set() {
        for item in curated_catalog() {
            assert!(
                !item.task_tags.is_empty(),
                "curated row {} must carry at least one task tag",
                item.catalog_id
            );
            for tag in item.task_tags {
                assert!(
                    matches!(*tag, "general" | "reasoning" | "coding" | "tools"),
                    "row {} has unexpected task tag {tag}",
                    item.catalog_id
                );
            }
        }
    }

    #[test]
    fn preload_message_suggests_a_fitting_alternative_for_an_oversized_model() {
        // 8 GB RAM, no GPU: a 40 GB model won't fit → actionable message.
        let hw = host(false, 0, 8 * GIB, 5 * GIB);
        let fp = crate::fit::advisory_footprint(40 * GIB);
        let (code, msg) =
            super::fit_preload_message(&hw, &fp, 40 * GIB, None).expect("won't fit -> message");
        assert_eq!(code, "model_too_large_for_host");
        assert!(msg.contains("larger than this machine"));
        assert!(msg.contains("camelid pull"));
        assert!(msg.contains("CAMELID_SKIP_FIT_CHECK=1"));
    }

    #[test]
    fn preload_message_blames_the_resident_model_when_releasing_it_would_help() {
        // Same oversized verdict, but something else is loaded. The remedy is to
        // release THAT, not to go find a smaller model — so the code and the
        // wording must both change.
        let hw = host(false, 0, 8 * GIB, 5 * GIB);
        let fp = crate::fit::advisory_footprint(40 * GIB);
        let reclaim = super::ResidentReclaim {
            ids: vec!["Llama-3.2-1B-Instruct-Q8_0.gguf".to_string()],
            bytes: 3 * GIB,
        };
        let (code, msg) = super::fit_preload_message(&hw, &fp, 40 * GIB, Some(&reclaim))
            .expect("won't fit -> message");
        assert_eq!(code, "model_requires_unload");
        assert!(msg.contains("Llama-3.2-1B-Instruct-Q8_0.gguf"), "{msg}");
        assert!(msg.contains("replace"), "{msg}");
        // The dead-end wording must be gone: the machine CAN hold this model.
        assert!(!msg.contains("larger than this machine"), "{msg}");
    }

    #[test]
    fn an_empty_reclaim_falls_back_to_the_plain_too_large_message() {
        // Nothing else resident (or nothing measurable) → no false promise that
        // releasing something would help.
        let hw = host(false, 0, 8 * GIB, 5 * GIB);
        let fp = crate::fit::advisory_footprint(40 * GIB);
        let empty = super::ResidentReclaim::default();
        let (code, _) = super::fit_preload_message(&hw, &fp, 40 * GIB, Some(&empty))
            .expect("won't fit -> message");
        assert_eq!(code, "model_too_large_for_host");
    }

    #[test]
    fn preload_message_is_none_when_the_model_fits() {
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        let fp = crate::fit::advisory_footprint(2 * GIB);
        assert!(super::fit_preload_message(&hw, &fp, 2 * GIB, None).is_none());
    }

    #[test]
    fn a_resident_model_never_turns_a_fitting_verdict_into_a_refusal() {
        // Reclaim info only WORDS a refusal; it must never create one.
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        let fp = crate::fit::advisory_footprint(2 * GIB);
        let reclaim = super::ResidentReclaim {
            ids: vec!["other.gguf".to_string()],
            bytes: 3 * GIB,
        };
        assert!(super::fit_preload_message(&hw, &fp, 2 * GIB, Some(&reclaim)).is_none());
    }

    #[test]
    fn preload_message_is_none_on_unprobed_host() {
        // Unknown verdict must never hard-block a load.
        let hw = host(false, 0, 0, 0);
        let fp = crate::fit::advisory_footprint(40 * GIB);
        assert!(super::fit_preload_message(&hw, &fp, 40 * GIB, None).is_none());
    }

    #[test]
    fn preload_message_fires_from_an_exact_dims_footprint() {
        // 4 GB RAM host, no GPU: a 5 GB-weights model's EXACT footprint (weights + KV
        // at the default context + scratch) exceeds the budget → typed message.
        let hw = host(false, 0, 4 * GIB, 3 * GIB);
        let dims = crate::fit::ModelDims {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        let fp = crate::fit::exact_footprint(
            5 * GIB,
            dims,
            crate::fit::ADVISORY_CONTEXT_TOKENS,
            crate::fit::KvDtype::F32,
        );
        assert!(super::fit_preload_message(&hw, &fp, 5 * GIB, None).is_some());
    }

    #[test]
    fn load_request_replace_defaults_to_false_for_existing_api_clients() {
        // Back-compat: a body without `replace` must not start evicting models.
        let req: super::LoadModelRequest =
            serde_json::from_str(r#"{"path":"models/x.gguf"}"#).expect("legacy body parses");
        assert!(!req.replace);
        let opted: super::LoadModelRequest =
            serde_json::from_str(r#"{"path":"models/x.gguf","replace":true}"#)
                .expect("replace body parses");
        assert!(opted.replace);
    }

    #[test]
    fn best_fitting_suggestion_picks_something_on_a_big_host_and_never_panics_on_tiny() {
        let big = host(false, 0, 64 * GIB, 48 * GIB);
        let s = super::best_fitting_catalog_suggestion(&big).expect("a row fits a 64 GB host");
        assert!(s.contains("camelid pull"));
        // On a tiny host only small rows fit (or none) — must not panic either way.
        let tiny = host(false, 0, 4 * GIB, 3 * GIB);
        let _ = super::best_fitting_catalog_suggestion(&tiny);
    }

    #[test]
    fn skip_fit_check_override_matches_only_the_exact_flag() {
        // The override restores the pre-advisor load-attempt behavior. It must engage
        // for exactly `1` (trimmed) and nothing else, so the preflight can't be turned
        // off by an unrelated or malformed value.
        assert!(super::fit_check_skipped(Some("1")));
        assert!(super::fit_check_skipped(Some("  1  ")));
        assert!(!super::fit_check_skipped(Some("0")));
        assert!(!super::fit_check_skipped(Some("true")));
        assert!(!super::fit_check_skipped(Some("11")));
        assert!(!super::fit_check_skipped(Some("")));
        assert!(!super::fit_check_skipped(None));
    }

    #[test]
    fn curated_footprint_is_exact_when_dims_known_else_approx() {
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        let size = 8_000_000_000u64;
        // approx: no dims → coarse size pad.
        let (fp_approx, conf) = super::curated_footprint(size, None, &hw);
        assert_eq!(conf, "approx");
        assert_eq!(fp_approx, crate::fit::advisory_footprint(size));
        // exact: real dims → precise KV cache sizing.
        let dims = crate::fit::ModelDims {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        let (fp_exact, conf) = super::curated_footprint(size, Some(dims), &hw);
        assert_eq!(conf, "exact");
        let expected = crate::fit::exact_footprint(
            size,
            dims,
            crate::fit::ADVISORY_CONTEXT_TOKENS,
            crate::fit::KvDtype::F32, // CPU host
        );
        assert_eq!(fp_exact, expected);
    }

    #[test]
    fn arch_support_is_only_claimed_where_the_vocabulary_matches() {
        // Experimental rows: the architecture comes from `hf_browse::guess_architecture`,
        // whose tokens are mapped onto the loader's vocabulary before the allowlist
        // is consulted, so both answers are usable.
        assert_eq!(super::guessed_arch_support("qwen3"), "implemented");
        assert_eq!(super::guessed_arch_support("llama"), "implemented");
        // REGRESSION: the guesser names a model FAMILY, the allowlist names a GGUF
        // architecture. Mixtral GGUFs declare `llama`, and Camelid ships an
        // evidence-backed `mixtral_8x7b_instruct_v0_1_q8_0` row — testing the family
        // token directly folded every Mixtral repo into the "cannot run this" drawer.
        assert_eq!(super::guessed_arch_support("mixtral"), "implemented");
        assert_eq!(super::loader_architecture_for_guess("mixtral"), "llama");
        // A family Camelid genuinely does not implement still reads as a negative.
        assert_eq!(super::guessed_arch_support("mamba"), "not_implemented");
        // A guess that matched nothing is an absence of evidence and must NOT read
        // as a negative, or a UI filtering on it would hide models that load fine.
        assert_eq!(super::guessed_arch_support(""), "unknown");
    }

    #[test]
    fn every_guessable_architecture_token_is_reported_loadable() {
        // The helper is only sound while every token the browse guesser can emit maps
        // onto something the loader recognizes. Pin that: a token that drifts out of
        // the loader's vocabulary would silently hide its whole family behind a label
        // claiming Camelid cannot run it.
        for token in [
            "llama", "mistral", "qwen2", "qwen3", "smollm3", "gemma3", "gemma4", "phi3", "lfm2",
            "mixtral",
        ] {
            assert_eq!(
                super::guessed_arch_support(token),
                "implemented",
                "browse can emit {token}, and Camelid loads that family"
            );
        }
    }

    #[test]
    fn only_curated_artifacts_are_recognized_for_the_size_pad() {
        // The on-demand fit check falls back to the coarse pad for curated rows only,
        // so this predicate decides whether an unmeasurable model keeps a verdict.
        let curated = curated_catalog();
        let row = curated.first().expect("catalog is non-empty");
        assert!(super::is_curated_artifact(row.repo_id, row.filename));
        // Right file, wrong repo (and vice versa) is not a curated row.
        assert!(!super::is_curated_artifact(
            "someone/elses-GGUF",
            row.filename
        ));
        assert!(!super::is_curated_artifact(
            row.repo_id,
            "not-a-curated-file.gguf"
        ));
    }

    #[test]
    fn curated_rows_make_no_architecture_support_claim() {
        // REGRESSION: deriving this from `CatalogItem::architecture` stamped three
        // shipped, pinned rows `not_implemented` — the curated field is a display
        // label (`qwen25`, `command-r`) whose vocabulary is not the loader's
        // (`qwen2`, `qwen35`). A wrong negative on a row we vouch for is worse than
        // no claim, and nothing filters curated rows on this axis.
        let hw = host(false, 0, 64 * GIB, 48 * GIB);
        for item in curated_catalog() {
            let view = CatalogItemView::from_curated(&item, &hw);
            assert_eq!(
                view.arch_support, "unknown",
                "curated row {} must not claim architecture support",
                view.catalog_id
            );
        }
        // The label that caused the false negative is still in the catalog, so this
        // test keeps its teeth.
        assert!(
            curated_catalog()
                .iter()
                .any(|item| !crate::model::is_implemented_architecture(item.architecture)),
            "expected at least one curated label outside the loader allowlist"
        );
    }

    #[test]
    fn a_busy_host_is_refused_without_being_called_too_small() {
        // 64 GB machine with 2 GB free, asked for a ~4 GB model. It is refused (the
        // load would be at risk), but telling the operator to buy a smaller model
        // would be false: the machine is one of the biggest in the catalog's range.
        let hw = host(false, 0, 64 * GIB, 2 * GIB);
        let fp = crate::fit::advisory_footprint(4 * GIB);
        let (code, msg) =
            super::fit_preload_message(&hw, &fp, 4 * GIB, None).expect("refused -> message");
        assert_eq!(code, "host_memory_unavailable");
        assert!(msg.contains("free right now"), "{msg}");
        assert!(msg.contains("Close some applications"), "{msg}");
        assert!(!msg.contains("larger than this machine"), "{msg}");
        // The override is still offered, exactly as for the too-large case.
        assert!(msg.contains("CAMELID_SKIP_FIT_CHECK=1"), "{msg}");
    }

    #[test]
    fn a_resident_model_still_takes_precedence_over_the_busy_host_wording() {
        // Same busy host, but something of OURS is loaded. Releasing that is the
        // specific remedy, so it must win over the generic "close applications".
        let hw = host(false, 0, 64 * GIB, 2 * GIB);
        let fp = crate::fit::advisory_footprint(4 * GIB);
        let reclaim = super::ResidentReclaim {
            ids: vec!["Qwen3-4B-Q8_0.gguf".to_string()],
            bytes: 4 * GIB,
        };
        let (code, msg) = super::fit_preload_message(&hw, &fp, 4 * GIB, Some(&reclaim))
            .expect("refused -> message");
        assert_eq!(code, "model_requires_unload");
        assert!(msg.contains("Qwen3-4B-Q8_0.gguf"), "{msg}");
    }
}

/// HTTP-level checks for the on-demand catalog fit endpoint. These cover the
/// input validation only: the resolve path needs the network, and a test that
/// reached the Hub would be neither hermetic nor fast.
#[cfg(test)]
mod catalog_fit_endpoint_tests {
    use super::{router_with_state, AppState};
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    async fn post_fit(body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let app = router_with_state(AppState::default());
        let request = Request::builder()
            .method("POST")
            .uri("/api/models/catalog/fit")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    async fn assert_rejected(body: serde_json::Value, code: &str) {
        let (status, response) = post_fit(body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response["error"]["code"], code);
    }

    #[tokio::test]
    async fn the_filename_must_be_one_bare_gguf_name() {
        // Both fields are interpolated into a Hub resolve URL, so traversal and
        // nesting are rejected before any fetch is attempted.
        assert_rejected(
            serde_json::json!({
                "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
                "filename": "../../etc/passwd",
                "size_bytes": 1024u64,
            }),
            "invalid_filename",
        )
        .await;
        assert_rejected(
            serde_json::json!({
                "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
                "filename": "nested/model.gguf",
                "size_bytes": 1024u64,
            }),
            "invalid_filename",
        )
        .await;
        assert_rejected(
            serde_json::json!({
                "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
                "filename": "model.bin",
                "size_bytes": 1024u64,
            }),
            "invalid_filename",
        )
        .await;
    }

    #[tokio::test]
    async fn the_repo_id_must_be_a_hugging_face_repository_id() {
        assert_rejected(
            serde_json::json!({
                "repo_id": "unsloth/../secrets",
                "filename": "model.gguf",
                "size_bytes": 1024u64,
            }),
            "invalid_repo_id",
        )
        .await;
        assert_rejected(
            serde_json::json!({
                "repo_id": "",
                "filename": "model.gguf",
                "size_bytes": 1024u64,
            }),
            "invalid_repo_id",
        )
        .await;
    }

    #[tokio::test]
    async fn a_zero_size_is_rejected_rather_than_fetched() {
        // The header parser validates tensor offsets against the true full length,
        // so a zero size could never produce an honest verdict.
        assert_rejected(
            serde_json::json!({
                "repo_id": "unsloth/Phi-4-mini-instruct-GGUF",
                "filename": "model.gguf",
                "size_bytes": 0u64,
            }),
            "invalid_size",
        )
        .await;
    }
}

#[cfg(test)]
mod download_cancel_tests {
    use super::{
        active_downloads_map, download_destination_key, finalize_download_artifact,
        kill_download_process, router_with_state, ActiveDownload, AppState,
    };
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    static DOWNLOAD_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn request(
        app: axum::Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(value) => {
                builder = builder.header("content-type", "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        let response = app.oneshot(builder.body(body).unwrap()).await.unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    fn temp_paths(tag: &str) -> (String, String) {
        let dir =
            std::env::temp_dir().join(format!("camelid-dl-test-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        (
            dir.join("model.gguf.part").to_string_lossy().into_owned(),
            dir.join("model.gguf").to_string_lossy().into_owned(),
        )
    }

    #[test]
    fn successful_tracked_download_promotes_part_to_dest() {
        let (part, dest) = temp_paths("promote");
        std::fs::write(&part, b"payload").unwrap();
        let status = finalize_download_artifact(true, true, &part, &dest);
        assert_eq!(status, "completed");
        assert!(
            !std::path::Path::new(&part).exists(),
            ".part must be renamed away"
        );
        assert_eq!(std::fs::read(&dest).unwrap(), b"payload");
        std::fs::remove_file(&dest).ok();
    }

    #[test]
    fn canceled_download_never_promotes_even_when_curl_succeeded() {
        let (part, dest) = temp_paths("cancel");
        std::fs::write(&part, b"payload").unwrap();
        // still_tracked=false models a cancel that removed the map entry while
        // curl went on to finish successfully (the Windows kill-less failure mode).
        let status = finalize_download_artifact(true, false, &part, &dest);
        assert_eq!(status, "failed");
        assert!(
            !std::path::Path::new(&part).exists(),
            "canceled .part must be removed"
        );
        assert!(
            !std::path::Path::new(&dest).exists(),
            "canceled download must never produce a loadable GGUF"
        );
    }

    #[test]
    fn failed_download_removes_partial() {
        let (part, dest) = temp_paths("fail");
        std::fs::write(&part, b"half").unwrap();
        let status = finalize_download_artifact(false, true, &part, &dest);
        assert_eq!(status, "failed");
        assert!(!std::path::Path::new(&part).exists());
        assert!(!std::path::Path::new(&dest).exists());
    }

    #[test]
    fn destination_reservation_uses_platform_filename_semantics() {
        #[cfg(windows)]
        assert_eq!(
            download_destination_key("Model.GGUF"),
            download_destination_key("model.gguf")
        );
        #[cfg(not(windows))]
        assert_ne!(
            download_destination_key("Model.GGUF"),
            download_destination_key("model.gguf")
        );
    }

    #[tokio::test]
    async fn terminal_downloads_remain_visible_until_acknowledged() {
        let _test = DOWNLOAD_TEST_LOCK.lock().await;
        let id = format!("terminal-test-{}", uuid::Uuid::new_v4());
        let mut terminal = ActiveDownload {
            id: id.clone(),
            repo_id: "test/repo".to_string(),
            filename: "terminal.gguf".to_string(),
            continuation_mode: "start".to_string(),
            total_bytes: 10,
            bytes_downloaded: 10,
            status: "downloading",
            child_pid: None,
            finished_at: Some(std::time::Instant::now()),
        };
        terminal.status = "completed";
        active_downloads_map()
            .lock()
            .unwrap()
            .insert(id.clone(), terminal);
        let app = router_with_state(AppState::default());

        for _ in 0..2 {
            let (status, body) =
                request(app.clone(), "GET", "/api/models/catalog/downloads", None).await;
            assert_eq!(status, StatusCode::OK);
            assert!(body
                .as_array()
                .unwrap()
                .iter()
                .any(|download| download["id"] == id));
        }

        let (status, _) = request(
            app.clone(),
            "POST",
            "/api/models/catalog/ack",
            Some(serde_json::json!({ "id": id })),
        )
        .await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let (_, body) = request(app, "GET", "/api/models/catalog/downloads", None).await;
        assert!(!body
            .as_array()
            .unwrap()
            .iter()
            .any(|download| download["id"] == id));
    }

    #[tokio::test]
    async fn acknowledgement_rejects_active_downloads_and_is_idempotent_when_missing() {
        let _test = DOWNLOAD_TEST_LOCK.lock().await;
        let id = format!("active-test-{}", uuid::Uuid::new_v4());
        active_downloads_map().lock().unwrap().insert(
            id.clone(),
            ActiveDownload {
                id: id.clone(),
                repo_id: "test/repo".to_string(),
                filename: "active.gguf".to_string(),
                continuation_mode: "download".to_string(),
                total_bytes: 10,
                bytes_downloaded: 1,
                status: "downloading",
                child_pid: None,
                finished_at: None,
            },
        );
        let app = router_with_state(AppState::default());
        let (status, body) = request(
            app.clone(),
            "POST",
            "/api/models/catalog/ack",
            Some(serde_json::json!({ "id": id })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["error"]["code"], "download_still_running");
        active_downloads_map().lock().unwrap().remove(&id);

        let (status, _) = request(
            app,
            "POST",
            "/api/models/catalog/ack",
            Some(serde_json::json!({ "id": id })),
        )
        .await;
        assert_eq!(status, StatusCode::NO_CONTENT);
    }

    #[test]
    fn kill_download_process_terminates_a_live_child() {
        // A long-lived stand-in for curl on each platform.
        #[cfg(windows)]
        let mut child = std::process::Command::new("ping")
            .args(["-n", "30", "127.0.0.1"])
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn ping");
        #[cfg(not(windows))]
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");

        kill_download_process(child.id());

        // taskkill/kill act asynchronously; the child must die well before its
        // natural 30s runtime.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().expect("try_wait") {
                assert!(!status.success(), "killed child must not report success");
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "child was not terminated within 10s"
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
