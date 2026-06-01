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
    extract::{rejection::JsonRejection, Path as AxumPath, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use minijinja::{
    context, Environment, Error as MiniJinjaError, ErrorKind as MiniJinjaErrorKind,
    UndefinedBehavior,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::{
    execution_plan::{plan_for_model, ExecutionPlan, PlannerEnv},
    gguf::{read_metadata, GgufFile, GgufTensorDescriptor, GgufTensorType},
    inference::{
        diagnostic_attention_score_scale, diagnostic_ffn_gate_up_order,
        diagnostic_gqa_head_mapping, diagnostic_linear_accumulation_precision,
        diagnostic_output_projection_layout, diagnostic_rectangular_linear_layout,
        diagnostic_rms_norm_epsilon, diagnostic_rope_direction, diagnostic_rope_pairing,
        diagnostic_rope_position_mode, diagnostic_square_linear_layout,
        diagnostic_zero_delta_selector, output_projection_diagnostics,
        q8_schedule_telemetry_enabled, reset_q8_schedule_telemetry, snapshot_q8_schedule_telemetry,
        DeltaZeroTarget, LlamaForwardDiagnostics, LlamaForwardTimings, LlamaGenerationStep,
        LlamaInferenceSession, LlamaLayerMemoryTimings, LlamaLayerTimings, LlamaLoadedWeights,
        LlamaOutputProjectionDiagnostic, LlamaQ8ScheduleTelemetry, LlamaSampler, SamplingConfig,
    },
    model::{DenseLlamaDims, LlamaFfnTensors, LlamaModelConfig, LlamaTensorBinding},
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
    execution_plans: Arc<RwLock<HashMap<String, ExecutionPlan>>>,
    cached_weights: Arc<RwLock<HashMap<String, Arc<LlamaLoadedWeights>>>>,
    active_model_id: Arc<RwLock<Option<String>>>,
    model_last_used: Arc<RwLock<HashMap<String, std::time::Instant>>>,
    cached_prompt_prefix: Arc<Mutex<Option<CachedPromptPrefix>>>,
    generation_sessions: Arc<RwLock<HashMap<String, GenerationSessionSummary>>>,
    planner_env: PlannerEnv,
    configured_threads: Option<usize>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            execution_plans: Arc::new(RwLock::new(HashMap::new())),
            cached_weights: Arc::new(RwLock::new(HashMap::new())),
            active_model_id: Arc::new(RwLock::new(None)),
            model_last_used: Arc::new(RwLock::new(HashMap::new())),
            cached_prompt_prefix: Arc::new(Mutex::new(None)),
            generation_sessions: Arc::new(RwLock::new(HashMap::new())),
            planner_env: PlannerEnv::capture(),
            configured_threads: None,
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
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
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
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub best_of: Option<u32>,
    pub logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_prompt_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
}

#[derive(Debug, Serialize)]
pub struct LlamaServerTokenizeResponse {
    pub tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
pub struct LlamaServerDetokenizeRequest {
    pub tokens: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct LlamaServerDetokenizeResponse {
    pub content: String,
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
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, serde_json::Value>,
    #[serde(default, skip_deserializing)]
    default_max_tokens_cap: Option<u32>,
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
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: String,
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
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
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

struct PreparedGeneration {
    model_id: String,
    model_path: PathBuf,
    token_ids: Vec<u32>,
    max_tokens: u32,
    tokenizer: Arc<Tokenizer>,
    session: LlamaInferenceSession,
    sampling: SamplingConfig,
    stop_sequences: Vec<String>,
    logit_diagnostic_token_ids: Vec<u32>,
    collect_dense_diagnostics: bool,
    dense_metadata: DenseDiagnosticMetadata,
    timings: GenerationTimings,
    cached_prompt_prefix: Arc<Mutex<Option<CachedPromptPrefix>>>,
}

struct GeneratedText {
    text: String,
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    dense_metadata: DenseDiagnosticMetadata,
    top_logits: Vec<LogitDiagnostic>,
    step_top_logits: Vec<Vec<LogitDiagnostic>>,
    output_projection: Vec<LlamaOutputProjectionDiagnostic>,
    dense: Option<LlamaForwardDiagnostics>,
    completion_tokens: usize,
    finish_reason: &'static str,
    timings: GenerationTimings,
}

struct GeneratedTokens {
    prompt_token_ids: Vec<u32>,
    token_ids: Vec<u32>,
    dense_metadata: DenseDiagnosticMetadata,
    top_logits: Vec<RawLogitDiagnostic>,
    step_top_logits: Vec<Vec<RawLogitDiagnostic>>,
    output_projection: Vec<LlamaOutputProjectionDiagnostic>,
    dense: Option<LlamaForwardDiagnostics>,
    finish_reason: &'static str,
    timings: GenerationTimings,
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

pub fn router() -> Router {
    router_with_state(AppState::default())
}

pub fn router_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/api/capabilities", get(capabilities))
        .route("/execution-plan", get(execution_plan))
        .route("/api/execution-plan", get(execution_plan))
        .route("/api/models/load", post(load_model))
        .route("/api/models/unload", post(unload_model))
        .route("/api/models/current", get(current_model))
        .route("/api/models/metadata", get(model_metadata))
        .route("/api/models/tokenizer", get(model_tokenizer))
        .route("/api/models/tokenizer/encode", post(tokenizer_encode))
        .route("/api/models/tokenizer/decode", post(tokenizer_decode))
        .route("/tokenize", post(llama_server_tokenize))
        .route("/detokenize", post(llama_server_detokenize))
        .route("/props", get(llama_server_props))
        .route(
            "/api/generation/sessions",
            get(generation_sessions).post(create_generation_session),
        )
        .route("/api/models/catalog", get(get_catalog))
        .route("/api/models/catalog/install", post(install_catalog_model))
        .route("/api/models/catalog/downloads", get(get_catalog_downloads))
        .route("/api/models/catalog/cancel", post(cancel_catalog_download))
        .route("/v1/models", get(v1_models))
        .route("/v1/models/:model", get(v1_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub async fn serve(
    addr: SocketAddr,
    configured_threads: Option<usize>,
    initial_model: Option<PathBuf>,
) -> std::io::Result<()> {
    let state = AppState::with_configured_threads(configured_threads);
    if let Some(model_path) = initial_model {
        if let Err(err) = load_model_from_path(&state, model_path, None).await {
            tracing::error!(error=%err, "failed to load startup model");
            return Err(std::io::Error::other(err.to_string()));
        }
    }
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "camelid server listening");
    axum::serve(listener, router_with_state(state)).await
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let active_id_lock = state.active_model_id.read().await;
    let loaded_models = state.loaded_models.read().await;
    let model = active_id_lock.as_ref().and_then(|id| loaded_models.get(id));
    let loaded_now = !loaded_models.is_empty();
    let generation_ready = model.is_some_and(loaded_model_generation_ready);
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
    })
}

async fn llama_server_props(State(state): State<AppState>) -> Json<LlamaServerPropsResponse> {
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
        chat_template_caps: serde_json::json!({}),
        modalities: LlamaServerModalities { vision: false },
        build_info: "camelid",
        is_sleeping: false,
        camelid: LlamaServerPropsCamelid {
            compatibility: "partial_llama_server_props_read_only",
            generation_ready,
            model_path_redacted: true,
            unsupported: vec![
                "post_props",
                "slots",
                "native_completion",
                "embeddings",
                "reranking",
                "multimodal",
            ],
        },
    })
}

fn loaded_model_generation_ready(model: &LoadedModel) -> bool {
    let Some(binding) = model.llama_tensors.as_ref() else {
        return false;
    };
    model.llama_config.is_some()
        && matches!(model.tokenizer, TokenizerLoadState::Available(_))
        && guard_cpu_weight_materialization_budget(binding).is_ok()
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
            current_gate: "Current exact-row support: TinyLlama Q8_0 current gate; Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs; Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048 packs; and Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs where row-specific PASS artifacts exist. Mistral 7B Instruct v0.3 Q8_0 has evidence-only bring-up with checked tokenizer/template, parity, and bounded 512/1024/2048/4096/8192 context artifacts, but support remains fail-closed until API/WebUI support-surface proof is synchronized. Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token backend MoE runtime evidence only; later 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by Gate 9A 50-token divergence and a longer-continuation hang, so broad/API/WebUI/frontend readiness remains unsupported. These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied.",
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
                notes: "TinyLlama remains the current support gate; exact Llama 3.2 1B Instruct Q8_0 now has checked bounded 512/1024/2048/4096/8192-context packs; exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs; and exact Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048-context packs where row-specific PASS artifacts exist. These are exact bounded-pack lanes only; no model-native/larger-context beyond the checked packs, arbitrary-template, production-throughput, portability, neighboring-row, or broad-family support is implied.",
            },
        ],
        planned_quantization: vec![
            SupportItem {
                id: "Q4_0/Q5_0",
                status: "planned",
                notes: "legacy smaller GGUF quantization lane",
            },
            SupportItem {
                id: "Q4_K_M/Q5_K_M",
                status: "planned",
                notes: "K-quant lane after simpler quant validation",
            },
        ],
        supported_model_families: vec![
            SupportItem {
                id: "llama_spm_decoder",
                status: "supported_current_gate",
                notes: "LLaMA-style decoder path validated on TinyLlama Q8_0 gate",
            },
            SupportItem {
                id: "llama_bpe_decoder_exact_1b_3b_8b_q8_0",
                status: "supported_exact_row_smoke_lanes",
                notes: "exact Llama 3.2 1B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048/4096/8192-context packs; exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke canonical Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs; exact Llama 3 8B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048-context packs, including the published source/runtime-head 8B 1024/2048 PASS bundle at 8e26be0a73c0. Broader 50-token, compact chat-template-shapes, and retained-block lazy-Q8 hot-path evidence remain exact-row bounded pack/measurement evidence only, and broad/full support still needs separate proof.",
            },
        ],
        planned_model_families: vec![
            SupportItem {
                id: "larger_llama_instruct",
                status: "planned",
                notes: "broader LLaMA-family instruct support after row-specific parity, API, WebUI, memory/perf, and portability evidence",
            },
            SupportItem {
                id: "mistral",
                status: "active_validation_unsupported",
                notes: "Mistral-7B-Instruct-v0.3.Q8_0.gguf has evidence-only tokenizer/template, parity, and bounded context artifacts, but v0.1 support remains fail-closed until API/WebUI support-surface proof, RSS/timing posture, current-head sync, and durable bundle evidence are synchronized",
            },
            SupportItem {
                id: "mixtral_moe",
                status: "active_validation_partial_runtime",
                notes: "public readiness: Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token exact-row MoE runtime evidence only. Top-k expert routing runs with lazy/file-backed rank-3 Q8 experts, but later Gate 9A 50-token evidence diverged at generated token index 9 and a longer-continuation backend call hung, so API/WebUI/frontend readiness and broad Mixtral support remain unsupported.",
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
            ModelCompatibilityTarget {
                id: "tinyllama_1_1b_chat_q8_0",
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
                family: "llama_bpe_decoder",
                quantization: "Q8_0",
                status: "supported_exact_row_smoke",
                support_scope: "exact_row_smoke_only",
                full_support_status: "blocked_pending_normalized_full_support",
                full_support_blockers: "model-native/larger context beyond checked packs, broader arbitrary/Jinja templates beyond row-scoped metadata-Jinja renderer and template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable repeated current-head bundles remain missing",
                metadata_parses: "validated",
                tokenizer_works: "validated_for_compact_and_small_prompt_pack",
                tensors_load: "validated",
                generation_runs: "api_completion_and_chat_smoke_plus_five_prompt_api_smoke",
                parity_audited: "compact_and_broader_prompt_pack_50_token_match",
                performance_measured: "bounded_unique_chat_perf_rss_validated",
                frontend_load_path_verified: "validated",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "short_api_webui_smoke_with_broader_prompt_pack_parity_plus_first_512_second_1024_and_third_2048_context_packs",
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
                bounded_context_4096_pack: "not_promoted",
                bounded_context_4096_pack_id: "not_selected",
                bounded_context_4096_window: 4096,
                bounded_context_8192_pack: "not_promoted",
                bounded_context_8192_pack_id: "not_selected",
                bounded_context_8192_window: 8192,
                latest_checked_bucket: "llama3-context-2048-smoke-v1",
                latest_checked_result: "pass",
                latest_checked_output: "CMLD-204",
                evidence: "the exact tracked Llama-3.2-3B-Instruct-Q8_0 GGUF has canonical Ubuntu main-lane API/WebUI support-gate refresh evidence at qa/evidence-bundles/llama32-3b-api-webui-current-head-20260513T2005Z-head-e9f926e/manifest.json for source head e9f926ed1a65, plus exact-row load, completion, chat-completion, frontend-smoke, five-prompt API smoke evidence, compact prompt-token/deterministic 1-token/5-token/bounded 50-token parity, broader three-prompt 50-token parity, first bounded 512-context parity, second bounded 1024-context parity, third bounded 2048-context parity, bounded compact template-shape coverage, exact-row metadata-Jinja renderer coverage for the recognized Llama 3.2 row template, and bounded unique-chat perf/RSS evidence; Camelid supports exact-row smoke for this row only, not broader/full support",
                next_step: "preserve exact-row smoke plus checked 512/1024/2048 context support while normalizing model-native/larger context, broader arbitrary/Jinja template behavior beyond row-scoped metadata-Jinja/template-shape evidence, production throughput beyond bounded perf/RSS and the first-token direction probe, portability, and durable full-support bundle evidence before any broader/full-support claim",
            },
            ModelCompatibilityTarget {
                id: "llama3_8b_instruct_q8_0",
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
                id: "llama_spm_q4_0_q5_0",
                family: "llama_spm_decoder",
                quantization: "Q4_0/Q5_0",
                status: "planned_phase_10",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "real dequant/matmul support, parity, bounded load/readiness, API/WebUI, RSS/timing, and durable bundle evidence are missing",
                metadata_parses: "descriptor_guarded",
                tokenizer_works: "planned_per_model",
                tensors_load: "unsupported_typed_error",
                generation_runs: "blocked_until_dequant",
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
                evidence: "planned quant tensor fixtures parse descriptors but reject CPU f32 loading until dequant/matmul support exists",
                next_step: "implement legacy smaller-quant dequant tests before any real-model support claim",
            },
            ModelCompatibilityTarget {
                id: "llama_spm_q4_k_q5_k",
                family: "llama_spm_decoder",
                quantization: "Q4_K_M/Q5_K_M",
                status: "planned_phase_10",
                support_scope: "future_exact_row_planning_only",
                full_support_status: "not_applicable_until_runtime_support",
                full_support_blockers: "loader/matmul support, parity, bounded load/readiness, API/WebUI, RSS/timing, and durable bundle evidence are missing",
                metadata_parses: "descriptor_guarded",
                tokenizer_works: "planned_per_model",
                tensors_load: "unsupported_typed_error",
                generation_runs: "blocked_until_dequant",
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
                evidence: "planned K-quant tensor fixtures parse descriptors but reject CPU f32 loading until dequant/matmul support exists",
                next_step: "start after simpler Q4_0/Q5_0 support has loader, matmul, and parity evidence",
            },
            ModelCompatibilityTarget {
                id: "mistral_7b_instruct_v0_3_q8_0",
                family: "mistral",
                quantization: "Q8_0",
                status: "active_validation_unsupported",
                support_scope: "bringup_exact_row_unsupported",
                full_support_status: "blocked_unsupported_bringup",
                full_support_blockers: "API/WebUI readiness, RSS/timing, current-head promotion sync, scrubbed manifest posture, support-surface proof, and durable promotion bundle evidence remain incomplete; exact tokenizer/template references plus row-specific 1-token/bounded/broader parity evidence alone do not promote support",
                metadata_parses: "validated",
                tokenizer_works: "validated",
                tensors_load: "validated",
                generation_runs: "evidence_only_completion_and_chat_smoke_plus_broader_50_token_api_smoke",
                parity_audited: "tokenizer_template_1tok_bounded_and_broader_50_token_parity_pass",
                performance_measured: "evidence_only_bounded_unique_chat_perf_rss",
                frontend_load_path_verified: "fail_closed_pending_support_surface_sync",
                frontend_readiness_gate: "fail-closed for v0.1: do not unlock chat for this row until API/WebUI support-surface proof is synchronized with the parity/context evidence",
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
                latest_checked_bucket: "current_head_api_webui_rss_fail_closed",
                latest_checked_result: "pass",
                latest_checked_output: "CMLD-M7B",
                evidence: "Mistral v0.3 active validation only; exact tokenizer/template, deterministic 1-token/5-token, broader 50-token, and bounded 512/1024/2048/4096/8192 context evidence are green, but support-promotion evidence remains fail-closed",
                next_step: "produce a support-promotion manifest proving API/WebUI readiness, support contract fields, RSS/timing posture, current-head sync, public scrub, and durable bundle evidence before any v0.1 support claim",
            },
            ModelCompatibilityTarget {
                id: "mixtral_8x7b_instruct_v0_1_q8_0",
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
                id: "gemma2_9b_it_q8_0",
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
                id: "openai_chat_completions",
                status: "supported_current_gate",
                notes: "non-streaming and SSE streaming for loaded supported dense GGUF models",
            },
            SupportItem {
                id: "tokenizer_encode_decode",
                status: "supported_current_gate",
                notes: "loaded-model tokenizer APIs for supported tokenizer families",
            },
            SupportItem {
                id: "llama_server_tokenizer_aliases",
                status: "partial",
                notes: "POST /tokenize and POST /detokenize are bounded loaded-model tokenizer aliases that return token ids/text only; piece metadata remains unsupported",
            },
            SupportItem {
                id: "llama_server_props",
                status: "partial",
                notes: "GET /props returns read-only public server properties, default generation settings, chat-template metadata when a model is loaded, and fail-closed Camelid readiness notes. Local model paths are intentionally redacted, POST /props is unsupported, and this does not imply /slots, /completion, embeddings, or full llama-server WebUI parity.",
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

async fn load_model(State(state): State<AppState>, Json(req): Json<LoadModelRequest>) -> Response {
    match load_model_from_path(&state, req.path, req.id).await {
        Ok(loaded) => (StatusCode::OK, Json(loaded)).into_response(),
        Err(err) => api_error(
            StatusCode::BAD_REQUEST,
            "invalid_model",
            err.to_string(),
            Some("path"),
        ),
    }
}

async fn load_model_from_path(
    state: &AppState,
    path: PathBuf,
    id: Option<String>,
) -> Result<LoadedModel, BackendError> {
    let gguf = read_metadata(&path)?;
    let outcome = plan_for_model(&path, &gguf, state.configured_threads);
    state.planner_env.apply(&outcome.env_updates);
    log_selected_execution_plan(&outcome.plan);
    let id = id
        .or_else(|| gguf.model_name().map(ToOwned::to_owned))
        .or_else(|| path.file_stem().map(|s| s.to_string_lossy().to_string()))
        .unwrap_or_else(|| "loaded-model".to_string());
    let llama_config_result = LlamaModelConfig::from_gguf(&gguf);
    let unsupported_runtime = match &llama_config_result {
        Err(BackendError::UnsupportedModelArchitecture(message)) => {
            Some(UnsupportedRuntimeSummary {
                code: "unsupported_model_architecture",
                message: message.clone(),
            })
        }
        _ => None,
    };
    let llama_config = llama_config_result.ok();
    let llama_tensors = llama_config
        .as_ref()
        .and_then(|config| LlamaTensorBinding::bind(&gguf, config).ok());
    let tokenizer_result = Tokenizer::from_gguf(&gguf);
    let tokenizer = tokenizer_state_from_result(tokenizer_result.as_ref());
    let tokenizer_runtime = tokenizer_result.ok().map(Arc::new);
    let loaded = LoadedModel {
        id: id.clone(),
        path,
        gguf,
        llama_config,
        llama_tensors,
        unsupported_runtime,
        tokenizer,
        tokenizer_runtime,
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
    *state.active_model_id.write().await = Some(id.clone());

    clear_prompt_prefix_cache(state);
    Ok(loaded)
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

    if let Some(id) = target_id {
        state.loaded_models.write().await.remove(&id);
        state.execution_plans.write().await.remove(&id);
        state.cached_weights.write().await.remove(&id);
        state.model_last_used.write().await.remove(&id);

        let mut active = state.active_model_id.write().await;
        if active.as_ref() == Some(&id) {
            *active = state.loaded_models.read().await.keys().next().cloned();
        }
    } else {
        state.loaded_models.write().await.clear();
        state.execution_plans.write().await.clear();
        state.cached_weights.write().await.clear();
        state.model_last_used.write().await.clear();
        *state.active_model_id.write().await = None;
    }

    clear_prompt_prefix_cache(&state);
    StatusCode::NO_CONTENT.into_response()
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
    } else {
        if let Some(active) = state.active_model_id.read().await.as_ref() {
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
        }
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
    let path = resolve_model_path(&target_id);
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

fn resolve_model_path(model_id: &str) -> Option<PathBuf> {
    let path = PathBuf::from(model_id);
    if path.exists() {
        return Some(path);
    }

    let local_path = PathBuf::from("models").join(model_id);
    if local_path.exists() {
        return Some(local_path);
    }

    for item in curated_catalog() {
        if item.catalog_id == model_id || item.filename == model_id {
            let cat_path = PathBuf::from("models").join(item.filename);
            return Some(cat_path);
        }
    }

    if let Ok(entries) = std::fs::read_dir("models") {
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
    if req.with_pieces.unwrap_or(false) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "/tokenize with_pieces=true is not supported yet; Camelid currently returns token ids only"
                .to_string(),
            Some("with_pieces"),
        );
    }
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
        Ok(tokens) => {
            (StatusCode::OK, Json(LlamaServerTokenizeResponse { tokens })).into_response()
        }
        Err(err) => api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "tokenization_failed",
            err.to_string(),
            Some("content"),
        ),
    }
}

async fn llama_server_detokenize(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LlamaServerDetokenizeRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
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
    }
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

async fn completions(
    State(state): State<AppState>,
    payload: std::result::Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
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
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: None,
    };
    let stream = req.stream.unwrap_or(false);
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if stream {
        return stream_completion(prepared, false);
    }

    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    match generate_decoded_tokens_blocking(prepared).await {
        Ok(generated) => {
            let GeneratedText {
                text,
                prompt_token_ids,
                generated_token_ids,
                dense_metadata,
                top_logits,
                step_top_logits,
                output_projection,
                dense,
                completion_tokens,
                finish_reason,
                timings,
            } = generated;
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
                        timings_ms: timings,
                    },
                }),
            )
                .into_response()
        }
        Err(response) => *response,
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    payload: std::result::Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(err) => return malformed_json_error(err),
    };
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
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: Some(DEFAULT_PUBLIC_CHAT_MAX_TOKENS),
    };
    let stream = req.stream.unwrap_or(false);
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    if stream {
        return stream_completion(prepared, true);
    }

    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    match generate_decoded_tokens_blocking(prepared).await {
        Ok(generated) => (
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
                        content: generated.text,
                    },
                    finish_reason: generated.finish_reason,
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
                    timings_ms: generated.timings,
                },
            }),
        )
            .into_response(),
        Err(response) => *response,
    }
}

async fn validate_generation_request(
    state: &AppState,
    req: GenerationSessionRequest,
) -> std::result::Result<GenerationSessionSummary, Response> {
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
        let f32_bytes = if file_backed_q8_linear {
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
        for desc in [
            &layer.attention_norm,
            &layer.attention_q,
            &layer.attention_k,
            &layer.attention_v,
            &layer.attention_output,
            &layer.ffn_norm,
        ] {
            total = total
                .checked_add(tensor_estimate(desc, retain_q8_blocks, lazy_q8_linear)?)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "CPU materialization byte estimate overflow".to_string(),
                    )
                })?;
        }
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
        }
    }
    Ok(total)
}

fn guard_cpu_weight_materialization_budget(binding: &LlamaTensorBinding) -> crate::Result<u64> {
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
    validate_unsupported_generation_fields(&req).map_err(|response| *response)?;
    validate_choice_and_logprob_fields(&req).map_err(|response| *response)?;
    let sampling = sampling_config_from_request(&req).map_err(|response| *response)?;
    let stop_sequences =
        stop_sequences_from_request(req.stop.as_ref()).map_err(|response| *response)?;
    let logit_diagnostic_token_ids =
        diagnostic_logit_token_ids(req.camelid_logit_token_ids.as_deref())
            .map_err(|response| *response)?;
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
            let rendered_prompt = render_chat_prompt_for_tokenization_for_model_result(
                &messages,
                &tokenizer,
                Some(&model.id),
            )
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
    if let Some(max_tokens) = requested_max_tokens {
        if token_ids.len() + max_tokens as usize > context_length {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "context_length_exceeded",
                format!(
                    "prompt token count {} plus max_tokens {} exceeds context length {}",
                    token_ids.len(),
                    max_tokens,
                    config.context_length
                ),
                Some("max_tokens"),
            ));
        }
    } else if token_ids.len() >= context_length {
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
    let max_tokens = requested_max_tokens.unwrap_or_else(|| {
        req.default_max_tokens_cap
            .map(|cap| cap.min(available_max_tokens))
            .unwrap_or(available_max_tokens)
    });

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
    let session = LlamaInferenceSession::new(config.clone(), weights).map_err(|err| {
        api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "dense_session_unavailable",
            err.to_string(),
            Some("model"),
        )
    })?;
    timings.session_create = session_create_started.elapsed().as_millis();

    Ok(PreparedGeneration {
        model_id: model.id,
        model_path: model.path,
        token_ids,
        max_tokens,
        tokenizer,
        session,
        sampling,
        stop_sequences,
        logit_diagnostic_token_ids,
        collect_dense_diagnostics: req.camelid_dense_diagnostics.unwrap_or(false),
        dense_metadata,
        timings,
        cached_prompt_prefix: state.cached_prompt_prefix.clone(),
    })
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
        rope_pairing: diagnostic_rope_pairing()
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
    if matches!(req.n, Some(value) if value > 1) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "n values greater than 1 are not supported yet; this backend returns one choice"
                .to_string(),
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
    if req.completion_logprobs.is_some() {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "completion logprobs are not supported yet; omit logprobs to receive text choices without token likelihoods".to_string(),
            Some("logprobs"),
        )));
    }
    if matches!(req.chat_logprobs, Some(true)) {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "chat logprobs are not supported yet; set logprobs to false or omit it".to_string(),
            Some("logprobs"),
        )));
    }
    if req.top_logprobs.is_some() {
        return Err(Box::new(api_error(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "top_logprobs are not supported yet because token logprobs are not exposed".to_string(),
            Some("top_logprobs"),
        )));
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
        "tools" | "tool_choice" | "parallel_tool_calls" | "parse_tool_calls" => {
            "tool/function calling is not supported by Camelid generation routes yet"
        }
        "response_format" | "json_schema" | "schema" | "grammar" => {
            "JSON/schema/grammar constrained generation is not supported yet"
        }
        "stream_options" => {
            "OpenAI stream_options are not supported yet; Camelid streams plain SSE chunks"
        }
        "echo" | "suffix" => "completion echo/suffix compatibility is not supported yet",
        "mirostat" | "mirostat_tau" | "mirostat_eta" | "min_p" | "typical_p" | "tfs_z"
        | "repeat_penalty" | "ignore_eos" | "n_keep" => {
            "this llama-server sampler/control field is not supported yet"
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
        "min_p" => "min_p",
        "typical_p" => "typical_p",
        "tfs_z" => "tfs_z",
        "repeat_penalty" => "repeat_penalty",
        "ignore_eos" => "ignore_eos",
        "n_keep" => "n_keep",
        "input" => "input",
        _ => "request",
    }
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
        seed: req.seed,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
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

async fn generate_decoded_tokens_blocking(
    prepared: PreparedGeneration,
) -> std::result::Result<GeneratedText, Box<Response>> {
    let timeout = generation_timeout_duration()?;
    let started = Instant::now();
    let handle = tokio::task::spawn_blocking(move || generate_decoded_tokens(prepared));
    match tokio::time::timeout(timeout, handle).await {
        Ok(Ok(result)) => result,
        Ok(Err(err)) => Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_worker_failed",
            format!("generation worker failed before completing the request: {err}"),
            None,
        ))),
        Err(_) => Err(generation_timeout_response(
            timeout,
            started.elapsed(),
            None,
        )),
    }
}

struct TimedGenerationStep {
    session: LlamaInferenceSession,
    step: LlamaGenerationStep,
}

enum GenerationStepBlockingError {
    Response(Box<Response>),
    Timeout {
        timeout: Duration,
        elapsed: Duration,
        generated_tokens: usize,
    },
}

struct StreamGenerationStepRequest {
    session: LlamaInferenceSession,
    input: Vec<u32>,
    sampler: LlamaSampler,
    history: Vec<u32>,
    collect_dense_diagnostics: bool,
    step_timeout: Duration,
    request_timeout: Duration,
    request_started: Instant,
    generated_tokens: usize,
}

async fn generate_stream_step_blocking(
    request: StreamGenerationStepRequest,
) -> std::result::Result<TimedGenerationStep, GenerationStepBlockingError> {
    let StreamGenerationStepRequest {
        session,
        input,
        sampler,
        history,
        collect_dense_diagnostics,
        step_timeout,
        request_timeout,
        request_started,
        generated_tokens,
    } = request;
    let test_sleep = generation_step_test_sleep_duration();
    let handle = tokio::task::spawn_blocking(move || {
        if let Some(duration) = test_sleep {
            std::thread::sleep(duration);
        }
        let mut session = session;
        let step = session
            .generate_next_token_with_history_diagnostics(
                &input,
                sampler,
                &history,
                collect_dense_diagnostics,
            )
            .map_err(|err| {
                Box::new(api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "generation_step_failed",
                    err.to_string(),
                    None,
                ))
            })?;
        Ok(TimedGenerationStep { session, step })
    });
    match tokio::time::timeout(step_timeout, handle).await {
        Ok(Ok(result)) => result.map_err(GenerationStepBlockingError::Response),
        Ok(Err(err)) => Err(GenerationStepBlockingError::Response(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation_worker_failed",
            format!("generation worker failed before completing the stream step: {err}"),
            None,
        )))),
        Err(_) => Err(GenerationStepBlockingError::Timeout {
            timeout: request_timeout,
            elapsed: request_started.elapsed(),
            generated_tokens,
        }),
    }
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
    prepared: PreparedGeneration,
) -> std::result::Result<GeneratedText, Box<Response>> {
    let tokenizer = prepared.tokenizer.clone();
    let stop_sequences = prepared.stop_sequences.clone();
    let generated = generate_token_ids(prepared)?;
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
        output_projection: generated.output_projection,
        dense: generated.dense,
        finish_reason: generated.finish_reason,
        timings: generated.timings,
    })
}

fn clear_prompt_prefix_cache(state: &AppState) {
    if let Ok(mut cached) = state.cached_prompt_prefix.lock() {
        *cached = None;
    }
}

fn lookup_prompt_prefix_cache(prepared: &PreparedGeneration) -> Option<CachedPromptPrefix> {
    let cached = prepared.cached_prompt_prefix.lock().ok()?.clone()?;
    (cached.model_id == prepared.model_id
        && cached.model_path == prepared.model_path
        && cached.token_ids == prepared.token_ids
        && cached.sampling == prepared.sampling)
        .then_some(cached)
}

fn store_prompt_prefix_cache(prepared: &PreparedGeneration, step: &LlamaGenerationStep) {
    if let Ok(mut cached) = prepared.cached_prompt_prefix.lock() {
        *cached = Some(CachedPromptPrefix {
            model_id: prepared.model_id.clone(),
            model_path: prepared.model_path.clone(),
            token_ids: prepared.token_ids.clone(),
            sampling: prepared.sampling.clone(),
            session: prepared.session.clone(),
            logits: step.logits.clone(),
            hidden_state: step.hidden_state.clone(),
            output_norm_state: step.output_norm_state.clone(),
        });
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
    let collect_step_top_logits = !prepared.logit_diagnostic_token_ids.is_empty();
    let mut output_projection = Vec::new();
    let mut dense = None;
    let mut finish_reason = "length";
    let mut forward_timings = LlamaForwardTimings::default();
    let mut sample = 0;
    let mut reused_prompt_prefix = false;

    if !prepared.collect_dense_diagnostics {
        if let Some(cached) = lookup_prompt_prefix_cache(&prepared) {
            prepared.session = cached.session.clone();
            input.clear();
            let first_step = sample_cached_prompt_prefix(&cached, &history)?;
            let cached_next_token = first_step.next_token_id;
            sample += first_step.sample;
            reused_prompt_prefix = true;
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
        }
    }

    for _ in generated.len() as u32..prepared.max_tokens {
        if finish_reason != "length" {
            break;
        }
        let mut sampling = prepared.sampling.clone();
        if let Some(seed) = sampling.seed {
            sampling.seed = Some(seed.wrapping_add(generated.len() as u64));
        }
        let sampler = if sampling == SamplingConfig::default() {
            LlamaSampler::Greedy
        } else {
            LlamaSampler::Sampling(sampling)
        };
        let step = prepared
            .session
            .generate_next_token_with_history_diagnostics(
                &input,
                sampler,
                &history,
                prepared.collect_dense_diagnostics,
            )
            .map_err(|err| {
                Box::new(api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "generation_step_failed",
                    err.to_string(),
                    None,
                ))
            })?;
        if !reused_prompt_prefix
            && generated.is_empty()
            && !prepared.collect_dense_diagnostics
            && step.diagnostics.is_none()
        {
            store_prompt_prefix_cache(&prepared, &step);
        }
        if generated.is_empty() && !reused_prompt_prefix {
            prepared.timings.prompt_evaluation = prompt_evaluation_timings_from_step(&step);
        }
        forward_timings.add_assign(&step.timings);
        sample += step.sample;
        if top_logits.is_empty() || collect_step_top_logits {
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
                top_logits = current_top_logits;
                let projection_token_ids = top_logits
                    .iter()
                    .map(|entry| entry.token_id)
                    .collect::<Vec<_>>();
                if prepared.collect_dense_diagnostics {
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
                }
            }
        }
        generated.push(step.next_token_id);
        history.push(step.next_token_id);
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

    prepared.timings.generate = generation_started.elapsed().as_millis();
    prepared.timings.generation = generation_phase_timings_from_forward(&forward_timings, sample);
    prepared.timings.layers = generation_layer_timings_from_forward(&forward_timings.layers);
    prepared.timings.memory = forward_timings.memory;
    if collect_q8_schedule {
        prepared.timings.q8_schedule = Some(snapshot_q8_schedule_telemetry());
    }

    Ok(GeneratedTokens {
        prompt_token_ids: prepared.token_ids,
        token_ids: generated,
        dense_metadata: prepared.dense_metadata,
        top_logits,
        step_top_logits,
        output_projection,
        dense,
        finish_reason,
        timings: prepared.timings,
    })
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

fn stream_completion(mut prepared: PreparedGeneration, chat: bool) -> Response {
    let model_id = prepared.model_id.clone();
    let stream_timing_diagnostics = stream_timing_diagnostics_enabled();
    let stream_poll_yield = stream_poll_yield_enabled();
    let stream_id = if chat {
        format!("chatcmpl-{}", uuid::Uuid::new_v4())
    } else {
        format!("cmpl-{}", uuid::Uuid::new_v4())
    };
    let events = async_stream::stream! {
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
            };
            yield sse_json_event(&role_chunk);
            if stream_poll_yield {
                tokio::task::yield_now().await;
            }
        }

        stream_event_timings.generate_start = Some(stream_started.elapsed().as_millis());
        let generation_started = Instant::now();
        let request_timeout = match generation_timeout_duration() {
            Ok(timeout) => timeout,
            Err(response) => {
                yield stream_error_event(*response);
                yield Ok(Event::default().data("[DONE]"));
                return;
            }
        };
        let collect_q8_schedule = stream_timing_diagnostics && q8_schedule_telemetry_enabled();
        if collect_q8_schedule {
            reset_q8_schedule_telemetry();
        }
        let mut input = prepared.token_ids.clone();
        let mut history = prepared.token_ids.clone();
        let mut generated = Vec::new();
        let mut top_logits = Vec::new();
        let mut output_projection = Vec::new();
        let mut dense = None;
        let mut finish_reason = "length";
        let mut streamed_text = String::new();
        let mut reused_prompt_prefix = false;
        let mut first_content_ms = None;
        let mut forward_timings = LlamaForwardTimings::default();
        let mut sample = 0;

        if !prepared.collect_dense_diagnostics {
            if let Some(cached) = lookup_prompt_prefix_cache(&prepared) {
                prepared.session = cached.session.clone();
                input.clear();
                match sample_cached_prompt_prefix(&cached, &history) {
                    Ok(first_step) => {
                        let cached_next_token = first_step.next_token_id;
                        reused_prompt_prefix = true;
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
                            yield stream_error_event(*response);
                            yield Ok(Event::default().data("[DONE]"));
                            return;
                        }
                        if finish_reason == "length" {
                            input.push(cached_next_token);
                        }
                    }
                    Err(response) => {
                        yield stream_error_event(*response);
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                }
            }
        }

        for _ in generated.len() as u32..prepared.max_tokens {
            if finish_reason != "length" {
                break;
            }
            let mut sampling = prepared.sampling.clone();
            if let Some(seed) = sampling.seed {
                sampling.seed = Some(seed.wrapping_add(generated.len() as u64));
            }
            let sampler = if sampling == SamplingConfig::default() {
                LlamaSampler::Greedy
            } else {
                LlamaSampler::Sampling(sampling)
            };
            let Some(remaining_timeout) = request_timeout.checked_sub(stream_started.elapsed()) else {
                yield generation_timeout_stream_event(request_timeout, stream_started.elapsed(), generated.len());
                yield Ok(Event::default().data("[DONE]"));
                return;
            };
            let TimedGenerationStep { session, step } = match generate_stream_step_blocking(
                StreamGenerationStepRequest {
                    session: prepared.session.clone(),
                    input: input.clone(),
                    sampler,
                    history: history.clone(),
                    collect_dense_diagnostics: prepared.collect_dense_diagnostics,
                    step_timeout: remaining_timeout,
                    request_timeout,
                    request_started: stream_started,
                    generated_tokens: generated.len(),
                },
            )
            .await
            {
                Ok(result) => result,
                Err(GenerationStepBlockingError::Response(response)) => {
                    yield stream_error_event(*response);
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                Err(GenerationStepBlockingError::Timeout { timeout, elapsed, generated_tokens }) => {
                    yield generation_timeout_stream_event(timeout, elapsed, generated_tokens);
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            };
            prepared.session = session;
            if !reused_prompt_prefix
                && generated.is_empty()
                && !prepared.collect_dense_diagnostics
                && step.diagnostics.is_none()
            {
                store_prompt_prefix_cache(&prepared, &step);
            }
            if generated.is_empty() && !reused_prompt_prefix {
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
                yield stream_error_event(*response);
                yield Ok(Event::default().data("[DONE]"));
                return;
            }

            let mut text = match prepared.tokenizer.decode(&generated, true) {
                Ok(text) => text,
                Err(err) => {
                    yield stream_error_message_event("token_decode_failed", err.to_string());
                    yield Ok(Event::default().data("[DONE]"));
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
                    stream_event_timings.first_content_yield = Some(stream_started.elapsed().as_millis());
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
                    };
                    yield sse_json_event(&chunk);
                    if stream_poll_yield {
                        tokio::task::yield_now().await;
                    }
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
                    if stream_poll_yield {
                        tokio::task::yield_now().await;
                    }
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
        stream_event_timings.final_yield = Some(stream_started.elapsed().as_millis());
        let camelid_diagnostics = stream_timing_diagnostics
            .then(|| stream_timing_diagnostics_json(&prepared.timings, first_content_ms, stream_event_timings));

        if chat {
            let final_chunk = ChatCompletionStreamChunk {
                id: stream_id,
                object: "chat.completion.chunk",
                created: 0,
                model: model_id,
                choices: vec![ChatCompletionStreamChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some(finish_reason),
                }],
                camelid: camelid_diagnostics.clone(),
            };
            yield sse_json_event(&final_chunk);
        } else {
            let final_chunk = CompletionStreamChunk {
                id: stream_id,
                object: "text_completion",
                created: 0,
                model: model_id,
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
    };

    Sse::new(events).into_response()
}

fn stream_error_event(response: Response) -> Result<Event, Infallible> {
    stream_error_message_event(
        "stream_error",
        format!("stream failed after headers: HTTP {}", response.status()),
    )
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
    let Some(space_id) = tokenizer.token_id("▁") else {
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
) -> std::result::Result<RenderedPrompt, MiniJinjaError> {
    let exact_llama32_metadata_jinja_row =
        model_id.and_then(llama32_metadata_jinja_exact_row_label);
    if let Some(template) = tokenizer.chat_template.as_deref() {
        if metadata_chat_template_enabled() {
            return render_metadata_jinja_chat_template_prompt(messages, tokenizer, template);
        }
        if let Some(row_label) = exact_llama32_metadata_jinja_row {
            if is_llama3_instruct_template(template) {
                return render_metadata_jinja_chat_template_prompt(messages, tokenizer, template);
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
        messages, tokenizer,
    ))
}

#[cfg(test)]
fn render_chat_prompt_for_tokenization_for_model(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
    model_id: Option<&str>,
) -> RenderedPrompt {
    render_chat_prompt_for_tokenization_for_model_result(messages, tokenizer, model_id)
        .unwrap_or_else(|_| render_chat_prompt_for_tokenization_fallback(messages, tokenizer))
}

fn render_chat_prompt_for_tokenization_fallback(
    messages: &[ChatMessage],
    tokenizer: &Tokenizer,
) -> RenderedPrompt {
    if let Some(template) = tokenizer.chat_template.as_deref() {
        // llama-server applies the TinyLlama marker template as regular SPM text,
        // so chat prompts should keep normal dummy-prefix handling for marker tokens.
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
    }

    RenderedPrompt {
        text: render_role_colon_prompt(messages),
        add_special: true,
        parse_special: tokenizer.chat_prompt_parse_special(),
    }
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
) -> std::result::Result<RenderedPrompt, MiniJinjaError> {
    let rendered = render_jinja_chat_template(messages, tokenizer, template)?;
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

fn api_error(
    status: StatusCode,
    code: &'static str,
    message: String,
    param: Option<&'static str>,
) -> Response {
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
        collections::{BTreeSet, HashMap},
        path::PathBuf,
        sync::{Arc, Mutex},
    };

    use crate::{
        execution_plan::{ExecutionPlan, ExecutionProfile},
        inference::{LlamaInferenceSession, LlamaLayerWeights, LlamaLoadedWeights, SamplingConfig},
        model::LlamaModelConfig,
        tensor::CpuTensor,
        tokenizer::{
            BpeRegistry, SpecialTokens, Token, TokenKind, Tokenizer, TokenizerConfig,
            TokenizerModel,
        },
    };

    use super::*;

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
        };

        let value = serde_json::to_value(chunk).expect("stream chunk should serialize");
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
            support_level: "supported_exact_row_smoke_512_1024_2048".into(),
            selected_backend: "cpu_q8_runtime_repack".into(),
            selected_q8_path: "x86_experimental_q8_0_avx2".into(),
            prefill_path: "q8_0_x86_avx2_tiled_gemm_experimental".into(),
            prefill_runtime_policy: "manual_override_only".into(),
            decode_path: "q8_0_decode_avx2".into(),
            thread_count: 16,
            diagnostics_status: "standard diagnostics; RSS timings disabled by default".into(),
            fallback_path: "retained_q8_reference_path".into(),
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
            assert_eq!(target.bounded_context_512_pack, "validated_bounded_pack");
            assert_eq!(
                target.bounded_context_512_pack_id,
                "llama3-context-512-smoke-v1"
            );
            assert_eq!(target.bounded_context_window, 512);
            assert_eq!(target.bounded_context_1024_pack, "validated_second_pack");
            assert_eq!(
                target.bounded_context_1024_pack_id,
                "llama3-context-1024-smoke-v1"
            );
            assert_eq!(target.bounded_context_1024_window, 1024);
            assert!(target
                .evidence
                .contains("second bounded 1024-context parity"));
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
        assert_eq!(three_b.bounded_context_2048_pack, "validated_third_pack");
        assert_eq!(
            three_b.bounded_context_2048_pack_id,
            "llama3-context-2048-smoke-v1"
        );
        assert_eq!(three_b.bounded_context_2048_window, 2048);
        assert_eq!(three_b.bounded_context_4096_pack, "not_promoted");
        assert_eq!(three_b.bounded_context_4096_pack_id, "not_selected");
        assert_eq!(three_b.bounded_context_4096_window, 4096);

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
    fn capabilities_report_exact_8b_1024_2048_after_current_head_alignment() {
        let response = capabilities_response();
        assert!(response.support_contract.current_gate.contains(
            "Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs"
        ));
        assert!(response.support_contract.current_gate.contains(
            "Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048 packs"
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
            "exact Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs"
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
            "exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke canonical Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs"
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
            .expect("Mistral exact-row bring-up lane should stay advertised");
        assert_eq!(mistral.status, "active_validation_unsupported");
        assert_eq!(mistral.support_scope, "bringup_exact_row_unsupported");
        assert_eq!(mistral.full_support_status, "blocked_unsupported_bringup");
        assert_eq!(
            mistral.frontend_load_path_verified,
            "fail_closed_pending_support_surface_sync"
        );
        assert_eq!(
            mistral.performance_measured,
            "evidence_only_bounded_unique_chat_perf_rss"
        );
        assert_eq!(
            mistral.latest_checked_bucket,
            "current_head_api_webui_rss_fail_closed"
        );
        assert_eq!(mistral.latest_checked_result, "pass");
        assert_eq!(mistral.latest_checked_output, "CMLD-M7B");
        assert!(mistral
            .frontend_readiness_gate
            .contains("fail-closed for v0.1"));
        assert_eq!(mistral.bounded_context_8192_pack, "validated_fifth_pack");
        assert_eq!(
            mistral.bounded_context_8192_pack_id,
            "mistral-context-8192-max-ladder-v1"
        );
        assert!(mistral
            .evidence
            .contains("Mistral v0.3 active validation only"));
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
                "llama32_1b_instruct_q8_0",
                "llama32_3b_instruct_q8_0",
                "llama3_8b_instruct_q8_0",
                "tinyllama_1_1b_chat_q8_0",
            ])
        );

        let supported_family_ids = response
            .supported_model_families
            .iter()
            .map(|item| item.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            supported_family_ids,
            BTreeSet::from(["llama_bpe_decoder_exact_1b_3b_8b_q8_0", "llama_spm_decoder",])
        );

        for id in [
            "mistral_7b_instruct_v0_3_q8_0",
            "mixtral_8x7b_instruct_v0_1_q8_0",
            "qwen25_7b_instruct_q8_0",
            "gemma2_9b_it_q8_0",
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

        let planned_rows = ["qwen25_7b_instruct_q8_0", "gemma2_9b_it_q8_0"];

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

    #[tokio::test]
    async fn stream_step_blocking_timeout_reports_generated_count() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "25");
        let session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();

        let result = generate_stream_step_blocking(StreamGenerationStepRequest {
            session,
            input: vec![1, 2],
            sampler: LlamaSampler::Greedy,
            history: vec![1, 2],
            collect_dense_diagnostics: false,
            step_timeout: Duration::from_millis(1),
            request_timeout: Duration::from_millis(1),
            request_started: Instant::now(),
            generated_tokens: 4,
        })
        .await;

        std::env::remove_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS");
        match result {
            Err(GenerationStepBlockingError::Timeout {
                timeout,
                generated_tokens,
                ..
            }) => {
                assert_eq!(timeout.as_millis(), 1);
                assert_eq!(generated_tokens, 4);
            }
            Err(GenerationStepBlockingError::Response(_)) => panic!("expected timeout error"),
            Ok(_) => panic!("expected stream step to time out"),
        }
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
            )
            .unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);

        assert!(lookup_prompt_prefix_cache(&prepared).is_none());
        store_prompt_prefix_cache(&prepared, &step);

        let cached = lookup_prompt_prefix_cache(&prepared).expect("exact key cache hit");
        assert_eq!(cached.session.kv_cache.position, 2);
        assert_eq!(cached.logits, step.logits);
        assert_eq!(
            sample_cached_prompt_prefix(&cached, &[1, 2])
                .unwrap()
                .next_token_id,
            step.next_token_id
        );

        let mut different_prompt =
            prepared_for_cache("tiny", "model-a.gguf", vec![1], cached.session.clone());
        different_prompt.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        assert!(lookup_prompt_prefix_cache(&different_prompt).is_none());

        let mut different_sampling =
            prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], cached.session.clone());
        different_sampling.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        different_sampling.sampling.temperature = 0.7;
        assert!(lookup_prompt_prefix_cache(&different_sampling).is_none());

        let mut different_model =
            prepared_for_cache("tiny", "model-b.gguf", vec![1, 2], cached.session);
        different_model.cached_prompt_prefix = prepared.cached_prompt_prefix.clone();
        assert!(lookup_prompt_prefix_cache(&different_model).is_none());
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
                        role: "system".to_string(),
                        content: "Answer briefly.".to_string(),
                    },
                    ChatMessage {
                        role: "user".to_string(),
                        content: "Say alpha.".to_string(),
                    },
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: "alpha".to_string(),
                    },
                    ChatMessage {
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
                        role: "user".to_string(),
                        content: "Complete cam".to_string(),
                    },
                    ChatMessage {
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
                        role: "system".to_string(),
                        content: " Be brief. ".to_string(),
                    },
                    ChatMessage {
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
                        role: "user".to_string(),
                        content: "Complete cam".to_string(),
                    },
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: " elid ".to_string(),
                    },
                    ChatMessage {
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
    fn keeps_compact_llama3_renderer_by_default_for_metadata_templates() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_metadata_subset_test_tokenizer();

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
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
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
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
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
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
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
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
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
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

    #[test]
    fn exact_llama32_1b_row_uses_metadata_jinja_renderer_without_env_opt_in_system_user() {
        let _guard = crate::test_support::env_lock();
        std::env::remove_var(METADATA_CHAT_TEMPLATE_ENV);
        let tokenizer = llama3_tokenizer_with_template(LLAMA3_METADATA_SUBSET_TEMPLATE);

        let rendered = render_chat_prompt_for_tokenization_for_model(
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: "  Be brief.  ".to_string(),
                },
                ChatMessage {
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
                    role: "system".to_string(),
                    content: " Answer tersely. ".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
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
                    role: "user".to_string(),
                    content: "Complete cam".to_string(),
                },
                ChatMessage {
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
                    role: "user".to_string(),
                    content: "Complete cam".to_string(),
                },
                ChatMessage {
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
                    role: "system".to_string(),
                    content: " Answer tersely. ".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: " Alpha? ".to_string(),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: " alpha ".to_string(),
                },
                ChatMessage {
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
                    role: "system".to_string(),
                    content: "Be brief.".to_string(),
                },
                ChatMessage {
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
                    role: " system ".to_string(),
                    content: "Be brief.".to_string(),
                },
                ChatMessage {
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

        let err =
            render_metadata_jinja_chat_template_prompt(&[], &tokenizer, template).unwrap_err();
        assert!(err.to_string().contains("unsupported chat-template branch"));

        let rendered = render_chat_prompt_for_tokenization(
            &[ChatMessage {
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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
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
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("llama32_3b_instruct_q8_0"),
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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-3B-Instruct-Q8_0"),
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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("llama32_1b_instruct_q8_0"),
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
                role: "user".to_string(),
                content: "  hello  ".to_string(),
            }],
            &tokenizer,
            Some("Meta-Llama-3.2-3B-Instruct-Q8_0"),
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

        let err =
            render_metadata_jinja_chat_template_prompt(&[], &tokenizer, template).unwrap_err();

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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Llama-3.2-1B-Instruct-Q8_0"),
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
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            &tokenizer,
            Some("Meta-Llama-3.2-3B-Instruct-Q8_0"),
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
            layers: vec![crate::model::LlamaLayerTensors {
                attention_norm: desc("blk.0.attn_norm.weight"),
                attention_q: desc("blk.0.attn_q.weight"),
                attention_k: desc("blk.0.attn_k.weight"),
                attention_v: desc("blk.0.attn_v.weight"),
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
            model_id: model_id.to_string(),
            model_path: PathBuf::from(model_path),
            token_ids,
            max_tokens: 1,
            tokenizer: Arc::new(test_tokenizer()),
            session,
            sampling: SamplingConfig::default(),
            stop_sequences: Vec::new(),
            logit_diagnostic_token_ids: Vec::new(),
            collect_dense_diagnostics: false,
            dense_metadata: dummy_dense_metadata(),
            timings: GenerationTimings::default(),
            cached_prompt_prefix: Arc::new(Mutex::new(None)),
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

    fn tiny_config() -> LlamaModelConfig {
        LlamaModelConfig {
            context_length: 4,
            embedding_length: 4,
            block_count: 1,
            feed_forward_length: 6,
            attention_head_count: 2,
            attention_head_count_kv: 1,
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
            moe: None,
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
                ffn_norm: ones("blk.0.ffn_norm.weight", hidden),
                ffn_gate: select_rows("blk.0.ffn_gate.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
                ffn_up: select_rows("blk.0.ffn_up.weight", ffn, hidden, &[0, 1, 2, 3, 0, 1]),
                ffn_down: select_rows("blk.0.ffn_down.weight", hidden, ffn, &[0, 1, 2, 3]),
                moe_router: None,
            }],
            layer_range: None,
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
    pub license: &'static str,
}

fn curated_catalog() -> Vec<CatalogItem> {
    vec![
        CatalogItem {
            catalog_id: "llama32_1b_instruct_q8_0",
            name: "Llama 3.2 1B Instruct Q8_0",
            repo_id: "unsloth/Llama-3.2-1B-Instruct-GGUF",
            filename: "Llama-3.2-1B-Instruct-Q8_0.gguf",
            size_bytes: 1346203104,
            downloads: 142000,
            likes: 540,
            quant: "Q8_0",
            license: "llama3.2",
        },
        CatalogItem {
            catalog_id: "llama32_3b_instruct_q8_0",
            name: "Llama 3.2 3B Instruct Q8_0",
            repo_id: "unsloth/Llama-3.2-3B-Instruct-GGUF",
            filename: "Llama-3.2-3B-Instruct-Q8_0.gguf",
            size_bytes: 3422709216,
            downloads: 98000,
            likes: 420,
            quant: "Q8_0",
            license: "llama3.2",
        },
        CatalogItem {
            catalog_id: "tinyllama_1_1b_chat_q8_0",
            name: "TinyLlama 1.1B Chat Q8_0",
            repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            size_bytes: 1169007424,
            downloads: 512000,
            likes: 1240,
            quant: "Q8_0",
            license: "other",
        },
        CatalogItem {
            catalog_id: "llama3_8b_instruct_q8_0",
            name: "Llama 3 8B Instruct Q8_0",
            repo_id: "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF",
            filename: "Meta-Llama-3-8B-Instruct.Q8_0.gguf",
            size_bytes: 8540846592,
            downloads: 320000,
            likes: 920,
            quant: "Q8_0",
            license: "llama3",
        },
    ]
}

#[derive(Debug, serde::Serialize)]
pub struct CatalogResponse {
    pub items: Vec<CatalogItem>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
pub struct CatalogQuery {
    pub query: Option<String>,
}

async fn get_catalog(
    axum::extract::Query(q): axum::extract::Query<CatalogQuery>,
) -> Json<CatalogResponse> {
    let items = curated_catalog();
    let filtered = if let Some(query_str) = q.query {
        let qs = query_str.to_lowercase();
        items
            .into_iter()
            .filter(|item| {
                item.name.to_lowercase().contains(&qs)
                    || item.repo_id.to_lowercase().contains(&qs)
                    || item.filename.to_lowercase().contains(&qs)
            })
            .collect()
    } else {
        items
    };
    Json(CatalogResponse {
        items: filtered,
        next_cursor: None,
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActiveDownload {
    pub id: String,
    pub repo_id: String,
    pub filename: String,
    pub total_bytes: u64,
    pub bytes_downloaded: u64,
    pub status: &'static str,
    #[serde(skip)]
    pub child_pid: Option<u32>,
}

#[derive(Debug, serde::Deserialize)]
pub struct InstallCatalogRequest {
    pub catalog_id: String,
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
}

static ACTIVE_DOWNLOADS: OnceLock<Mutex<HashMap<String, ActiveDownload>>> = OnceLock::new();

fn active_downloads_map() -> &'static Mutex<HashMap<String, ActiveDownload>> {
    ACTIVE_DOWNLOADS.get_or_init(|| Mutex::new(HashMap::new()))
}

async fn install_catalog_model(Json(req): Json<InstallCatalogRequest>) -> Response {
    let mut map = active_downloads_map().lock().unwrap();
    if map.contains_key(&req.catalog_id) {
        return (StatusCode::BAD_REQUEST, "Download already running").into_response();
    }

    std::fs::create_dir_all("models").ok();
    let dest_path = format!("models/{}", req.filename);
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        req.repo_id, req.filename
    );

    match std::process::Command::new("curl")
        .args(["-L", "-C", "-", "-o", &dest_path, &url])
        .spawn()
    {
        Ok(child) => {
            let pid = child.id();
            let download = ActiveDownload {
                id: req.catalog_id.clone(),
                repo_id: req.repo_id.clone(),
                filename: req.filename.clone(),
                total_bytes: req.size_bytes,
                bytes_downloaded: 0,
                status: "downloading",
                child_pid: Some(pid),
            };
            map.insert(req.catalog_id.clone(), download);

            let catalog_id_clone = req.catalog_id.clone();
            tokio::spawn(async move {
                let mut child = child;
                if let Ok(status) = child.wait() {
                    let mut map = active_downloads_map().lock().unwrap();
                    if let Some(dl) = map.get_mut(&catalog_id_clone) {
                        if status.success() {
                            dl.status = "completed";
                        } else {
                            dl.status = "failed";
                        }
                    }
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

async fn get_catalog_downloads() -> Json<Vec<ActiveDownload>> {
    let mut map = active_downloads_map().lock().unwrap();
    let mut to_remove = Vec::new();
    for (id, dl) in map.iter_mut() {
        if dl.status == "completed" || dl.status == "failed" {
            to_remove.push(id.clone());
            continue;
        }

        let path = format!("models/{}", dl.filename);
        if let Ok(metadata) = std::fs::metadata(&path) {
            dl.bytes_downloaded = metadata.len();
            if dl.bytes_downloaded >= dl.total_bytes {
                dl.status = "completed";
            }
        }
    }

    let result = map.values().cloned().collect::<Vec<_>>();

    for id in to_remove {
        map.remove(&id);
    }

    Json(result)
}

#[derive(Debug, serde::Deserialize)]
pub struct CancelDownloadRequest {
    pub id: String,
}

async fn cancel_catalog_download(Json(req): Json<CancelDownloadRequest>) -> Response {
    let mut map = active_downloads_map().lock().unwrap();
    if let Some(dl) = map.remove(&req.id) {
        if let Some(pid) = dl.child_pid {
            let mut kill_cmd = std::process::Command::new("kill");
            kill_cmd.arg(pid.to_string());
            kill_cmd.spawn().ok();
        }

        let path = format!("models/{}", dl.filename);
        std::fs::remove_file(path).ok();
        (StatusCode::OK, "Download canceled").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Download not found").into_response()
    }
}
