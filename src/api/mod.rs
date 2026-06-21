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
        speculative::{
            accepted_draft_prefix, ModelDrafter, NGramDrafter, SpeculativeDrafter,
            DEFAULT_MODEL_DRAFT_TOKENS, DEFAULT_NGRAM_DRAFT_TOKENS,
        },
        DeltaZeroTarget, LlamaForwardDiagnostics, LlamaForwardTimings, LlamaGenerationStep,
        LlamaInferenceSession, LlamaLayerMemoryTimings, LlamaLayerTimings, LlamaLoadedWeights,
        LlamaOutputProjectionDiagnostic, LlamaQ8ScheduleTelemetry, LlamaSampler, SamplingConfig,
    },
    model::{DenseLlamaDims, LlamaFfnTensors, LlamaModelConfig, LlamaTensorBinding},
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
    execution_plans: Arc<RwLock<HashMap<String, ExecutionPlan>>>,
    cached_weights: Arc<RwLock<HashMap<String, Arc<LlamaLoadedWeights>>>>,
    active_model_id: Arc<RwLock<Option<String>>>,
    model_last_used: Arc<RwLock<HashMap<String, std::time::Instant>>>,
    cached_prompt_prefix: Arc<Mutex<Option<CachedPromptPrefix>>>,
    generation_sessions: Arc<RwLock<HashMap<String, GenerationSessionSummary>>>,
    planner_env: PlannerEnv,
    configured_threads: Option<usize>,
    /// Server-wide default for opt-in thinking mode (`serve --enable-thinking`).
    /// Applied only to chat requests that omit `camelid_enable_thinking`; an
    /// explicit value in the request always wins. Off by default so the
    /// parity-locked thinking-DISABLED rendering stays the default.
    default_enable_thinking: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            gemma4_runtimes: Arc::new(RwLock::new(HashMap::new())),
            execution_plans: Arc::new(RwLock::new(HashMap::new())),
            cached_weights: Arc::new(RwLock::new(HashMap::new())),
            active_model_id: Arc::new(RwLock::new(None)),
            model_last_used: Arc::new(RwLock::new(HashMap::new())),
            cached_prompt_prefix: Arc::new(Mutex::new(None)),
            generation_sessions: Arc::new(RwLock::new(HashMap::new())),
            planner_env: PlannerEnv::capture(),
            configured_threads: None,
            default_enable_thinking: false,
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
    /// Which backend serves the active model: "gemma4-runtime", "llama", or "none".
    pub backend: &'static str,
    /// Model family of the active model ("gemma4", "llama-family", ...), if loaded.
    pub model_family: Option<&'static str>,
    /// True when the gemma4 serve path is built (CAMELID_GEMMA4_SERVE) and a gemma4
    /// runtime is loaded for the active model.
    pub gemma4_available: bool,
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
    pub logit_bias: Option<HashMap<String, f32>>,
    pub stop: Option<StopSpec>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub camelid_logit_token_ids: Option<Vec<u32>>,
    pub camelid_dense_diagnostics: Option<bool>,
    pub camelid_dense_diagnostic_generated_index: Option<u32>,
    /// Opt-in: attach a parity receipt to the (non-streaming) response. The
    /// receipt is a claim of output for the verifier to check — no reference
    /// runs here, so its parity block is emitted as not-compared.
    pub camelid_receipt: Option<bool>,
    /// Opt-in gemma4 thinking mode: renders the reference's enable_thinking
    /// template (system turn opens with the `<|think|>` token). Thinking
    /// channels are stripped from chat output either way. Default: false (the
    /// reference's `enable_thinking:false` rendering).
    pub camelid_enable_thinking: Option<bool>,
    /// OpenAI-style tool/function definitions. When present, they are rendered
    /// into the prompt through the loaded model's own chat template (Hybrid agent
    /// mode); the model's tool-call output is parsed by the client. Models whose
    /// template does not render tools simply ignore them.
    pub tools: Option<Vec<serde_json::Value>>,
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
    /// (`image_url`, `input_audio`, `video_url`, …). Camelid generates text
    /// tokens only — vision/audio towers are never loaded — so the chat
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
    pub camelid_dense_diagnostic_generated_index: Option<u32>,
    /// Opt-in Qwen3/gemma4 thinking mode: when true the chat renderer emits the
    /// template's thinking generation prompt (the model produces its own
    /// `<think>…</think>` block) instead of the deterministic thinking-disabled
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
    /// model is an implemented decoder that is NOT a supported exact row — the
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
    /// Present only when the request opted in via `camelid_receipt: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camelid_receipt: Option<ParityReceipt>,
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
    dense_diagnostic_generated_index: Option<usize>,
    dense_metadata: DenseDiagnosticMetadata,
    timings: GenerationTimings,
    cached_prompt_prefix: Arc<Mutex<Option<CachedPromptPrefix>>>,
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

/// Per-request speculative decoding state: the drafter plus round counters
/// for the end-of-request acceptance summary.
struct PreparedSpeculative {
    drafter: SpeculativeDrafter,
    draft_tokens: usize,
    rounds: u64,
    drafted: u64,
    accepted_drafts: u64,
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

pub fn router() -> Router {
    router_with_state(AppState::default())
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
        .route("/api/models/metadata", get(model_metadata))
        .route("/api/models/tokenizer", get(model_tokenizer))
        .route("/api/models/tokenizer/encode", post(tokenizer_encode))
        .route("/api/models/tokenizer/decode", post(tokenizer_decode))
        .route("/tokenize", post(llama_server_tokenize))
        .route("/detokenize", post(llama_server_detokenize))
        .route("/apply-template", post(llama_server_apply_template))
        .route("/models", get(llama_server_models))
        .route("/models/load", post(unsupported_llama_server_models_load))
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
        .route("/api/models/local", get(local_models))
        .route("/api/models/runnable-receipt", get(runnable_receipt))
        .route("/api/models/runnable-smoke", post(run_runnable_smoke))
        .route("/api/models/catalog", get(get_catalog))
        .route("/api/models/catalog/install", post(install_catalog_model))
        .route("/api/models/catalog/downloads", get(get_catalog_downloads))
        .route("/api/models/catalog/cancel", post(cancel_catalog_download))
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
) -> std::io::Result<()> {
    let state = AppState::with_configured_threads(configured_threads)
        .with_default_enable_thinking(default_enable_thinking);
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
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "camelid server listening");
    let url = format!("http://{addr}");

    // Warm the generation path BEFORE telling the user we're ready. The GPU resident
    // engine (NVRTC kernel compile + multi-GB weight upload + first prefill) is built
    // lazily on the first generation — a one-time cost of several seconds. If it lands
    // on the user's first prompt the model looks "dog slow" (it's really the cold
    // build, on the GPU the whole time). So: start serving in the background, fire one
    // tiny self-request through the exact same code path to build the engine, BLOCK on
    // it, and only then print "ready". After this the first real request is warm
    // (~0.5s) instead of ~10s. The warm-up is best-effort — any failure just falls back
    // to the old lazy build and is not fatal.
    let warm_model_id = state.active_model_id.read().await.clone();
    let server = tokio::spawn(async move { axum::serve(listener, router_with_state(state)).await });
    if let Some(model_id) = warm_model_id {
        eprintln!("\n  🐪 Warming up the GPU (building the resident engine, one-time)…");
        warmup_generation_blocking(addr, model_id).await;
    }
    print_ready_banner(&url);
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
/// proceeds. Errors are swallowed by the caller — this only shaves the cold start.
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
    eprintln!("\n  🐪 Camelid is ready");
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
    let backend = if gemma4_available {
        "gemma4-runtime"
    } else if model.is_some() {
        "llama"
    } else {
        "none"
    };
    // The gemma4 runtime (Q8-resident) is ready as soon as it is loaded; the Llama
    // f32-budget check does not apply to it.
    let generation_ready = gemma4_available || model.is_some_and(loaded_model_generation_ready);
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
    })
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
                "models_load",
                "models_unload",
                "multimodal_architecture_metadata",
            ],
        },
    })
    .into_response()
}

async fn unsupported_llama_server_models_load() -> Response {
    unsupported_route(
        "unsupported_llama_server_models_load",
        "POST /models/load is not supported yet; Camelid keeps native llama-server router-mode model loading separate from the stable /api/models/load path until cache listing, autoload, and support-contract behavior are implemented and tested",
        Some("model"),
    )
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
        camelid_enable_thinking: None,
        tools: None,
        unsupported_fields: req.unsupported_fields,
        default_max_tokens_cap: Some(DEFAULT_PUBLIC_CHAT_MAX_TOKENS),
    };
    let prepared = match prepare_generation(&state, req).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };

    let model_id = prepared.model_id.clone();
    let prompt_token_count = prepared.token_ids.len();
    match generate_decoded_tokens_blocking(prepared).await {
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
        // actually runs the model on the GPU — not the legacy hybrid-matmul flag, which
        // defaults off and made the UI read "GPU off" while the GPU did all the work.
        enabled: crate::cuda::gpu_accel_enabled(),
        device: crate::cuda::device_name(),
        backend: "cuda",
        run_count: crate::cuda::cuda_q8_run_count(),
    }
}

/// GET `/api/runtime/gpu` — current GPU-acceleration availability + on/off state.
async fn gpu_runtime() -> Json<GpuRuntimeState> {
    Json(current_gpu_runtime())
}

/// POST `/api/runtime/gpu` `{ "enabled": bool }` — flip GPU acceleration at
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
            current_gate: "Current exact-row support: TinyLlama Q8_0 current gate; Llama 3.2 1B Instruct Q8_0 has checked bounded 512/1024/2048/4096/8192 packs; Llama 3.2 3B Instruct Q8_0 is supported_exact_row_smoke with canonical Ubuntu main-lane API/WebUI refresh at source head e9f926ed1a65 plus checked bounded 512/1024/2048 packs; and Llama 3 8B Instruct Q8_0 has checked bounded 512/1024/2048 packs where row-specific PASS artifacts exist. Mistral 7B Instruct v0.3 Q8_0 is supported_exact_row_smoke: checked tokenizer/template, parity (including GPU-vs-CPU greedy continuations on the exact row), bounded 512/1024/2048/4096/8192 context artifacts, and a support-promotion API/WebUI smoke bundle. Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf has bounded one-token backend MoE runtime evidence only; later 5-token/API/WebUI/RSS promotion-candidate artifacts are superseded by Gate 9A 50-token divergence and a longer-continuation hang, so broad/API/WebUI/frontend readiness remains unsupported. The dense Qwen3 Q8_0 ChatML rows (0.6B/1.7B/4B/8B Instruct, thinking disabled) are supported_exact_row_smoke: qwen2 BPE pre-tokenizer + ChatML renderer, per-head QK-norm + NEOX RoPE, and token+text parity vs llama.cpp at 1/5/50 on macOS/Ubuntu and on Windows x86_64 CPU (cpu_reference + the x86_q8 AVX2 runtime-repack path, bit-identical), and additionally on Windows CUDA: the 0.6B/1.7B/4B rows fully VRAM-resident and the 8B row via the VRAM+host-RAM offload split (RTX 3060 Laptop 6 GB, driver 576.83, CUDA 12.9; GPU decode+single-shot prefill token+text identical to cpu_reference/llama.cpp at 1/5/50); 1.7B additionally has GPU-resident decode+prefill and a 15,373-token single-shot prefill lane on macOS, and thinking-mode is opt-in (leading-trace parity only). These are exact bounded lanes only; no model-native/larger context beyond the checked packs, arbitrary-template behavior, production throughput, portability, neighboring-row, or broad-family support is implied.",
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
                id: "mistral_instruct_exact_7b_v0_3_q8_0",
                status: "supported_exact_row_smoke_lane",
                notes: "exact Mistral-7B-Instruct-v0.3 Q8_0 has row-specific smoke support: tokenizer/template, deterministic and broader 50-token parity, checked bounded 512/1024/2048/4096/8192-context packs, GPU-vs-CPU greedy parity on the exact row, and a support-promotion API/WebUI smoke bundle. Exact row only; broader Mistral-family, other quants, model-native context, and full support are not implied.",
            },
            SupportItem {
                id: "llama_bpe_decoder_exact_1b_3b_8b_q8_0",
                status: "supported_exact_row_smoke_lanes",
                notes: "exact Llama 3.2 1B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048/4096/8192-context packs; exact Llama 3.2 3B Instruct Q8_0 has supported_exact_row_smoke canonical Ubuntu main-lane API/WebUI evidence at source head e9f926ed1a65 plus checked bounded 512/1024/2048-context packs; exact Llama 3 8B Instruct Q8_0 has row-specific smoke support with checked bounded 512/1024/2048-context packs, including the published source/runtime-head 8B 1024/2048 PASS bundle at 8e26be0a73c0. Broader 50-token, compact chat-template-shapes, and retained-block lazy-Q8 hot-path evidence remain exact-row bounded pack/measurement evidence only, and broad/full support still needs separate proof.",
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
                evidence: "the exact tracked gemma-4-E2B-it-Q8_0 GGUF (5,048,350,848 bytes, general.architecture=gemma4, 35 layers with the 4:1 sliding_window_pattern and per-layer feed_forward_length array parsed from the GGUF, gemma4 SPM tokenizer) loads through the mmap wire-backed Q8 lane and generates greedily with prompt token ids, generated token ids, and generated text identical to the pinned reference llama.cpp 5d56eff for every prompt in qa/gemma4/prompt_packs/basic_v1.json (oracle at qa/gemma4/oracle/gemma-4-E2B-it-Q8_0.basic_v1.json). The Metal GPU-resident runtime matches the same five prompts token-for-token, and distributed layer-sharding greedy output (TCP split 13/35) is token-identical to the oracle with fail-closed handshake/checksum/shared-KV guards. Parity verified under both the repacked and plain-Q8 oracle kernel variants. Served through /v1/chat/completions and /v1/completions (streaming + non-streaming) behind CAMELID_GEMMA4_SERVE. Raw logs: qa/evidence-bundles/gemma4-e2b-it-q8-0-20260610T103119Z-head-96a75007b156. Bounded-context ladder: committed packs + pinned-comparator oracles at qa/gemma4/prompt_packs/context_{512,1024,2048,4096,8192}_v1.json with recall asserted at capture; CPU and GPU parity cells pass full-budget at every bucket with no recorded frontiers (10/10 cells this row). The reference-exact chat template (both thinking modes) is locked byte-for-byte and token-for-token by qa/gemma4/template_shapes_v1.json. Distributed layer-sharding SERVE lane (CAMELID_GEMMA4_WORKER/CAMELID_GEMMA4_SPLIT) reuses wire protocol v1 with per-request worker sessions. Camelid supports exact-row text-token generation + serve smoke with checked bounded 512-8192 context packs for this row only; no model-native/larger context, performance, portability, multimodal, or full support is implied",
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
                id: "llama_spm_q4_0_q5_0",
                tool_capable: false,
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
                tool_capable: false,
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
                latest_checked_output: "qa/evidence-bundles/qwen3-0.6b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23/README.md",
                evidence: "exact row Qwen3-0.6B-Q8_0.gguf (explicit head_dim path: head_dim 128 != embedding/head_count 64, sourced from attention.key_length): token-AND-text-identical to llama.cpp at 1/5/50 (ChatML, thinking disabled). macOS bundle qa/evidence-bundles/qwen3-0.6b-q8-chatml-parity-20260614T032905Z-head-63bf015; Windows x86_64 CPU bundle qa/evidence-bundles/qwen3-0.6b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 records all_pass on both the cpu_reference scalar path and the x86_q8 AVX2 path vs llama.cpp 9632 (acd79d603).",
                next_step: "add WebUI smoke and normalized context/perf evidence on Windows before any broader claim",
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
                tested_context: "chatml_1_5_50_token_short_chat_plus_macos_15373_token_single_shot_prefill",
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
                latest_checked_output: "qa/evidence-bundles/qwen3-1.7b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23/README.md",
                evidence: "exact row Qwen3-1.7B-Q8_0.gguf: token-AND-text-identical to llama.cpp at 1/5/50 (ChatML, thinking disabled) on macOS/Ubuntu and Windows x86_64 CPU (cpu_reference + x86_q8 AVX2). macOS bundles qa/evidence-bundles/qwen3-1.7b-q8-chatml-parity-20260614T021844Z-head-f41e374 and qwen3-1.7b-q8-gpu-resident-bigctx-parity-20260614T171846Z-head-f97a896 (GPU-resident decode+prefill; 15,373-token single-shot prefill); thinking-mode leading-trace parity qwen3-1.7b-q8-thinking-enabled-parity-*. Windows bundle qa/evidence-bundles/qwen3-1.7b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 (all_pass).",
                next_step: "add WebUI smoke on Windows and re-prove the large-context lane on Windows before widening the platform claim",
            },
            ModelCompatibilityTarget {
                id: "qwen3_4b_instruct_q8_0",
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
                parity_audited: "chatml_thinking_disabled_token_and_text_parity_1_5_50_pass_confident_probes_cpu_reference_and_x86_q8_avx2",
                performance_measured: "not_promoted",
                frontend_load_path_verified: "api_smoke_validated_webui_follow_up",
                frontend_readiness_gate: "green only when this exact GGUF row plus Q8_0 quant match /api/capabilities and the runtime reports loaded_now=true, generation_ready=true, and matching active_model_id",
                tested_context: "chatml_1_5_50_token_short_chat_smoke_confident_probes",
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
                latest_checked_output: "qa/evidence-bundles/qwen3-4b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23/README.md",
                evidence: "exact row Qwen3-4B-Q8_0.gguf (explicit head_dim path): token-AND-text-identical to llama.cpp at 1/5/50 on confident probes (capital-of-France, say-hello, 2+2). The 'Name a primary color.' probe is a documented macOS first-token near-tie; on the Windows 9632 comparator it also matched (both 'Red'). macOS bundle qa/evidence-bundles/qwen3-4b-q8-chatml-parity-20260614T054617Z-head-368ed9b; Windows bundle qa/evidence-bundles/qwen3-4b-q8-windows-x86-chatml-parity-20260616T155745Z-head-fdae7a23 (all_pass, cpu_reference + x86_q8 AVX2).",
                next_step: "add WebUI smoke and normalized context/perf evidence on Windows before any broader claim",
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
                notes: "POST /tokenize and POST /detokenize are bounded loaded-model tokenizer aliases that return token ids/text, with /tokenize with_pieces=true exposing id/piece objects for supported tokenizer lanes. Arbitrary tokenizer kwargs and broader tokenizer parity remain unsupported.",
            },
            SupportItem {
                id: "llama_server_models",
                status: "partial",
                notes: "GET /models returns a privacy-safe read-only list of currently loaded Camelid models with redacted paths and text-only architecture metadata. Router-mode query params such as reload/autoload/model selection, cache listing, POST /models/load, POST /models/unload, multimodal metadata, and full llama-server model-management parity remain unsupported.",
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
                notes: "Native /infill, /metrics, /embedding, /embeddings, /v1/embeddings, /v1/messages, /rerank, /reranking, /v1/rerank, /v1/reranking, /v1/responses, POST /models/load, POST /models/unload, POST /slots, and slot cache actions return typed not_implemented errors until real route semantics and backend support exist. Unsupported /completion modes remain typed parameter errors.",
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
        // Fail closed with the exact typed reason and a stable, switchable code.
        // The message already carries the offending architecture/quant and any
        // dedicated-lane redirect (e.g. `camelid diffusion-gemma-chat`).
        Err(err) => api_error(
            StatusCode::BAD_REQUEST,
            backend_error_code(&err),
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
    /// Predicted lane (`supported` / `experimental_implemented` / `unsupported`).
    lane_class: ModelLaneClass,
    /// The exact typed blocker the load would hit — predicted WITHOUT binding
    /// tensors or loading weights. `None` when the architecture is implemented
    /// (it would load and run, supported or experimental).
    #[serde(skip_serializing_if = "Option::is_none")]
    blocker: Option<InspectBlocker>,
}

/// `POST /api/models/inspect` — header-only prediction of a GGUF's lane and the
/// exact reason it would fail closed, WITHOUT loading weights, so the UI can warn
/// before a multi-GB load attempt. Reads only the GGUF metadata header.
async fn inspect_model(Json(req): Json<InspectModelRequest>) -> Response {
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

    // Parse the config header (no tensor bind, no weight load). Ok ⇒ it would load;
    // Err ⇒ it would fail closed with this exact typed reason.
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
/// `<|turn>{system|user|model}\n…<turn|>\n` and generation follows a trailing
/// `<|turn>model\n`; a leading system message gets its own system turn, and
/// thinking mode injects `<|think|>` (see `gemma4_chat_prompt`). The renderer
/// is locked byte-for-byte to the reference rendering by
/// `qa/gemma4/template_shapes_v1.json`.
/// Gemma 4 turn markers. Gemma 4 RENAMED them from Gemma 3's
/// `<start_of_turn>`/`<end_of_turn>` to `<|turn>` (id 105) / `<turn|>` (id 106)
/// — verified against the E2B/E4B/12B GGUF vocab and the GGUF-embedded Jinja
/// chat template (`'<|turn>' + role + '\n'` … `'<turn|>\n'`). Using the old
/// spellings tokenizes as PLAIN TEXT: the model mimics them back and the stop
/// token never matches.
pub(crate) const GEMMA4_TURN_START: &str = "<|turn>";
pub(crate) const GEMMA4_TURN_END: &str = "<turn|>";
/// Thinking-channel markers (ids 100/101): the model may wrap hidden reasoning
/// in `<|channel>…<channel|>`. The GGUF template strips these spans from chat
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
        assert_eq!(
            render_phi3_prompt(&messages),
            "<|system|>\nBe concise.<|end|>\n<|user|>\nCapital of France?<|end|>\n<|assistant|>\n"
        );
        // Phi-3's <|end|>-separated template must be detected before TinyLlama's.
        let phi3_tmpl = "<|user|>\n{{content}}<|end|>\n<|assistant|>\n";
        assert!(is_phi3_template(phi3_tmpl));
        assert!(!is_phi3_template(
            "<|user|>\n{{content}}</s>\n<|assistant|>\n"
        ));
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
        // block — the model emits its own reasoning (the template default branch).
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
        // At least one enable_thinking=true shape — the lane's whole reason to exist.
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

/// Strip `<|channel>…<channel|>` thinking spans from a complete gemma4 chat
/// response. An unterminated span (generation hit the token budget inside the
/// channel) is stripped to its start — hidden reasoning must never leak.
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
pub enum Gemma4ServeRuntime {
    Local(crate::gemma4_runtime::Gemma4Runtime),
    Distributed(crate::gemma4_distributed::Gemma4DistributedRuntime),
}

impl Gemma4ServeRuntime {
    fn generate_greedy(&self, prompt: &str, max_new: usize) -> crate::Result<(String, Vec<u32>)> {
        match self {
            Self::Local(r) => r.generate_greedy(prompt, max_new),
            Self::Distributed(r) => r.generate_greedy(prompt, max_new),
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
        }
    }
}

/// Distributed gemma4 serve config from the environment: both vars must be
/// present and well-formed, or the pair is rejected loudly — a half-configured
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
/// greedy runtime — the same envelope the committed basic_v1 oracle pack checks.
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
    // DiffusionGemma redirect, …) instead of silently sitting non-generative.
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
    // serve runtime so /v1/chat can route to it. Fail clearly on error — never
    // silently fall back to the Llama path (which would produce garbage here).
    if gemma4_serve_enabled() && model_family(&loaded.gguf) == "gemma4" {
        load_gemma4_serve_runtime(state, &id, &loaded.path).await?;
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
        camelid_dense_diagnostic_generated_index: req.camelid_dense_diagnostic_generated_index,
        camelid_enable_thinking: None,
        tools: None,
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
    let effective_max_tokens = prepared.max_tokens;
    match generate_decoded_tokens_blocking(prepared).await {
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
                output_projection,
                dense,
                dense_diagnostic_generated_index,
                completion_tokens,
                finish_reason,
                timings,
                execution_trace: _,
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
    // Gemma 4 serve path (additive, gated by CAMELID_GEMMA4_SERVE). Short-circuits
    // if this request targets a loaded gemma4 runtime; otherwise falls through to
    // the existing Llama/3B path unchanged.
    match resolve_gemma4_runtime(&state, &req).await {
        Ok(Some((id, runtime))) => {
            return if req.stream.unwrap_or(false) {
                gemma4_chat_streaming(id, runtime, &req).await
            } else {
                gemma4_chat_nonstreaming(id, runtime, &req).await
            };
        }
        Ok(None) => {}
        Err(resp) => return resp,
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
        camelid_dense_diagnostic_generated_index: req.camelid_dense_diagnostic_generated_index,
        // An explicit request value always wins; the server-wide default
        // (`serve --enable-thinking`) only fills in when the request is silent.
        camelid_enable_thinking: req
            .camelid_enable_thinking
            .or(state.default_enable_thinking.then_some(true)),
        tools: req.tools,
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
    let effective_max_tokens = prepared.max_tokens;
    match generate_decoded_tokens_blocking(prepared).await {
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
    ))
}

fn receipt_stamp_from_parts(
    endpoint: &'static str,
    messages_or_prompt: serde_json::Value,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    stop: Vec<String>,
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
        camelid_enable_thinking: None,
        tools: None,
        unsupported_fields: HashMap::new(),
        default_max_tokens_cap: None,
    };
    let prepared = match prepare_generation(&state, session_request).await {
        Ok(prepared) => prepared,
        Err(response) => return Err(response_error_text(response).await),
    };
    let generated = match generate_decoded_tokens_blocking(prepared).await {
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
    let request_tools = req.tools.clone();
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
    {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        model.id.hash(&mut hasher);
        session.set_resident_cache_key(hasher.finish());
    }
    timings.session_create = session_create_started.elapsed().as_millis();

    let collect_dense_diagnostics = req.camelid_dense_diagnostics.unwrap_or(false)
        || req.camelid_dense_diagnostic_generated_index.is_some();
    // Lossless greedy speculation is a server-level opt-in and only engages
    // for plain greedy requests with no per-step logit consumers; anything
    // else keeps the unchanged vanilla decode loop.
    let speculative = match spec_decode_mode_from_env() {
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
            drafter: SpeculativeDrafter::NGram(NGramDrafter::default()),
            draft_tokens: spec_draft_tokens_from_env(DEFAULT_NGRAM_DRAFT_TOKENS),
            rounds: 0,
            drafted: 0,
            accepted_drafts: 0,
        }),
        Some(SpecDecodeMode::DraftModel) => Some(PreparedSpeculative {
            drafter: build_model_drafter(state, &model, &tokenizer).await?,
            draft_tokens: spec_draft_tokens_from_env(DEFAULT_MODEL_DRAFT_TOKENS),
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
        collect_dense_diagnostics,
        dense_diagnostic_generated_index: req
            .camelid_dense_diagnostic_generated_index
            .map(|index| index as usize),
        dense_metadata,
        timings,
        cached_prompt_prefix: state.cached_prompt_prefix.clone(),
        speculative,
        telemetry: Some(telemetry_start),
    })
}

/// Build the draft-model drafter for `CAMELID_SPEC_DECODE=draft`: load the
/// configured draft GGUF under the reserved `spec-draft` id (without making
/// it the active model) and fail closed unless its token mapping is
/// identical to the target's — drafted token ids must mean the same text in
/// the target vocabulary.
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
    let draft_model = match existing {
        Some(model) => model,
        None => load_model_from_path_with_activation(
            state,
            path,
            Some(SPEC_DRAFT_MODEL_ID.to_string()),
            false,
        )
        .await
        .map_err(|err| {
            api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "spec_draft_model_load_failed",
                err.to_string(),
                None,
            )
        })?,
    };
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
        "min_p" => "min_p",
        "typical_p" => "typical_p",
        "tfs_z" => "tfs_z",
        "repeat_penalty" => "repeat_penalty",
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
    let test_sleep = generation_step_test_sleep_duration();
    let handle = tokio::task::spawn_blocking(move || {
        if let Some(duration) = test_sleep {
            std::thread::sleep(duration);
        }
        generate_decoded_tokens(prepared)
    });
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
    /// Try the resident GPU-sampling greedy fast lane before the general step.
    greedy_fast: bool,
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

async fn generate_stream_step_blocking(
    request: StreamGenerationStepRequest,
) -> std::result::Result<TimedGenerationStep, GenerationStepBlockingError> {
    let StreamGenerationStepRequest {
        greedy_fast,
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
        if greedy_fast {
            if let Some((id, forward_us)) = session
                .generate_next_token_greedy_resident(input[0])
                .map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "generation_step_failed",
                        err.to_string(),
                        None,
                    ))
                })?
            {
                let step = gpu_sampled_generation_step(id, forward_us).map_err(|err| {
                    Box::new(api_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "generation_step_failed",
                        err.to_string(),
                        None,
                    ))
                })?;
                return Ok(TimedGenerationStep { session, step });
            }
        }
        // Temperature-only sampling: draw the next token on the GPU (Gumbel-max)
        // instead of copying the full logits row to the host and sorting it on the
        // CPU (~7 ms/token). Other sampler shapes fall through to the CPU path.
        if !collect_dense_diagnostics {
            if let LlamaSampler::Sampling(cfg) = &sampler {
                if let Some((id, forward_us)) = session
                    .generate_next_token_sampled_resident(input[0], cfg)
                    .map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "generation_step_failed",
                            err.to_string(),
                            None,
                        ))
                    })?
                {
                    let step = gpu_sampled_generation_step(id, forward_us).map_err(|err| {
                        Box::new(api_error(
                            StatusCode::SERVICE_UNAVAILABLE,
                            "generation_step_failed",
                            err.to_string(),
                            None,
                        ))
                    })?;
                    return Ok(TimedGenerationStep { session, step });
                }
            }
        }
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
        output_projection: generated.output_projection,
        dense: generated.dense,
        dense_diagnostic_generated_index: generated.dense_diagnostic_generated_index,
        finish_reason: generated.finish_reason,
        timings: generated.timings,
        execution_trace: generated.execution_trace,
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
    // A resident-GPU-prefilled session keeps its K/V history on the GPU only; the cached
    // clone would drop it and resume from empty CPU buffers. Skip caching those sessions —
    // a cache miss just re-runs the (fast, resident) prefill.
    if !prepared.session.cpu_kv_authoritative() {
        return;
    }
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
    let mut dense_diagnostic_generated_index = None;
    let mut finish_reason = "length";
    let mut forward_timings = LlamaForwardTimings::default();
    let mut sample = 0;
    let mut reused_prompt_prefix = false;

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
        if let Some(cached) = lookup_prompt_prefix_cache(&prepared) {
            prepared.session = cached.session.clone();
            // The cached session's resident-path pin reflects the request
            // that stored it; re-pin for this request's mode.
            prepared
                .session
                .set_resident_paths_disabled(prepared.speculative.is_some() && !spec_gpu_enabled());
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
                && !top_logits.is_empty()
        }) {
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
            // No drafts (e.g. no n-gram match) → fall through to the plain
            // single-token step below; a one-token verify chunk would only
            // add chunk-path overhead over the tuned decode step.
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
                    spec.rounds += 1;
                    spec.drafted += drafts.len() as u64;
                    spec.accepted_drafts += (acc.len() as u64).saturating_sub(1);
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
                }
                if finish_reason != "length" || generated.len() >= prepared.max_tokens as usize {
                    break;
                }
                input.clear();
                input.push(*history.last().expect("history grows every round"));
                continue;
            }
        }
        // Single-token continuations with no per-step logit consumers ride the
        // resident GPU fast lane: greedy via GPU argmax, temperature-only sampling
        // via GPU Gumbel-max. Everything else takes the general step.
        let fast_step = if input.len() == 1
            && !collect_dense_for_step
            && !collect_step_top_logits
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
        if !reused_prompt_prefix
            && generated.is_empty()
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
        if generated.is_empty() && !reused_prompt_prefix {
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
        output_projection,
        dense,
        dense_diagnostic_generated_index,
        finish_reason,
        timings: prepared.timings,
        execution_trace,
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
    // Speculation only runs in the non-streaming loop; streaming requests on
    // a spec-enabled server keep the unchanged vanilla path (including the
    // GPU-resident lanes the speculative pin would otherwise turn off).
    prepared.speculative = None;
    prepared.session.set_resident_paths_disabled(false);
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
        // Lifecycle telemetry for the streaming path. Dropping the stream
        // (client disconnect, error return) closes the run via the guard's
        // Drop, so the observatory never shows a stale "running" state.
        let telemetry_guard = prepared.telemetry.take().map(telemetry::RequestGuard::begin);
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

        // Bypass the prompt-prefix cache when the CUDA-resident engine drives
        // decode: reusing a cached session reseeds the GPU KV from f16-rounded
        // host history (a different reduction order than a clean GPU prefill),
        // which corrupts the resumed decode — mild for greedy (a few near-tie
        // flips) but catastrophic under temperature sampling, where it produces
        // garbled, off-topic output. The non-streaming path already gates the
        // cache this way (see resident_decode_cuda_active); the streaming path
        // must too. The CPU lane is reduction-order-stable and keeps the cache.
        if !prepared.collect_dense_diagnostics && !crate::inference::resident_decode_cuda_active() {
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
            let Some(remaining_timeout) = request_timeout.checked_sub(stream_started.elapsed()) else {
                yield generation_timeout_stream_event(request_timeout, stream_started.elapsed(), generated.len());
                yield Ok(Event::default().data("[DONE]"));
                return;
            };
            // Greedy single-token continuations with no per-step logit consumers ride the
            // resident GPU-sampling fast lane inside the blocking step.
            let greedy_fast = input.len() == 1
                && matches!(sampler, LlamaSampler::Greedy)
                && !collect_dense_for_step
                && !top_logits.is_empty();
            let TimedGenerationStep { session, step } = match generate_stream_step_blocking(
                StreamGenerationStepRequest {
                    // take_for_step (NOT clone): keeps the resident GPU session and its
                    // on-GPU KV cache alive across the blocking hand-off.
                    session: prepared.session.take_for_step(),
                    input: input.clone(),
                    sampler,
                    history: history.clone(),
                    collect_dense_diagnostics: collect_dense_for_step,
                    greedy_fast,
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
        if let Some(guard) = telemetry_guard {
            let ttft_ms = first_content_ms.map(|ms| ms as u64);
            let decode_tps = match (first_content_ms, generated.len()) {
                (Some(first_ms), count) if count > 1 => {
                    let decode_ms = generation_started.elapsed().as_millis().saturating_sub(first_ms);
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
    // Normalize OpenAI-style `{ "type":"function", "function":{…} }` tools to the
    // flat function objects (`{ "name", "description", "parameters" }`) the chat
    // templates (Llama 3.x etc.) actually render — matching llama.cpp / vLLM. This
    // keeps the wire format OpenAI-standard while the prompt the model sees aligns
    // with the `{"name":…, "parameters":…}` response format the template requests
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
        // Mistral Instruct templates (v0.3 GGUF) don't reference the `tools`
        // Jinja variable — the generic Jinja render silently drops them. Use
        // the dedicated renderer that produces [AVAILABLE_TOOLS] natively.
        if is_mistral_instruct_template(template) {
            return Ok(RenderedPrompt {
                text: render_mistral_instruct_prompt_with_tools(messages, tokenizer, tools),
                add_special: false,
                parse_special: true,
            });
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
        // as EOS when tokenizing the rendered template — so chat prompts
        // parse specials (chat_prompt_parse_special), with dummy-prefix
        // handling after control tokens preserved by encode_piece.
        // Checked BEFORE tinyllama: Phi-3 reuses the same <|user|>/<|assistant|>
        // marker spellings but separates turns with <|end|> (not </s>) and stops on
        // <|end|>. Routing it through the tinyllama renderer used the wrong separator
        // and stop token, so generation rambled.
        if is_phi3_template(template) {
            return RenderedPrompt {
                text: render_phi3_prompt(messages),
                // Phi-3 sets add_bos_token=true; the tokenizer prepends <s>.
                add_special: true,
                // Parse specials so <|user|>/<|assistant|>/<|end|> become the control
                // token ids — in particular <|end|> as the end-of-turn stop token.
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

/// Render a Qwen3 ChatML prompt. Mirrors the GGUF jinja template for the standard
/// cases (optional leading system turn, then user/assistant turns), then appends
/// the generation prompt for the requested thinking mode:
/// - `enable_thinking = false` (the deterministic, parity-locked default):
///   `<|im_start|>assistant\n<think>\n\n</think>\n\n` (the template's
///   `enable_thinking is false` branch — a direct answer, no reasoning).
/// - `enable_thinking = true`: `<|im_start|>assistant\n` (the template's default
///   branch — the model emits its own `<think>…</think>` reasoning block first).
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
            // default branch; the model generates its own <think>…</think> block.
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
/// `<tool_call>{"name":…,"arguments":{…}}</tool_call>`.
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
        // No system message from caller — tool block IS the system message.
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

/// Qwen3 (and Qwen2) ChatML template detector: `<|im_start|>` / `<|im_end|>`
/// turn markers. camelid's minijinja cannot render the full Qwen3 jinja template
/// (it uses constructs that evaluate to undefined), so the dense ChatML path is
/// rendered by [`render_qwen3_chatml_prompt`] instead.
fn is_qwen3_chatml_template(template: &str) -> bool {
    template.contains("<|im_start|>") && template.contains("<|im_end|>")
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
    let mut prompt = String::new();
    for message in messages {
        let role = match message.role.trim() {
            "system" => "system",
            "assistant" => "assistant",
            _ => "user",
        };
        prompt.push_str("<|");
        prompt.push_str(role);
        prompt.push_str("|>\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|end|>\n");
    }
    prompt.push_str("<|assistant|>\n");
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
        inference::{LlamaInferenceSession, LlamaLayerWeights, LlamaLoadedWeights, SamplingConfig},
        model::LlamaModelConfig,
        tensor::CpuTensor,
        tokenizer::{
            BpePreTokenizer, BpeRegistry, SpecialTokens, Token, TokenKind, Tokenizer,
            TokenizerConfig, TokenizerModel,
        },
    };

    use super::*;

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
            // ChatML short-chat smoke only — no bounded context pack is promoted.
            assert_eq!(
                target.bounded_context_512_pack, "not_promoted",
                "{id} ctx512"
            );
        }
    }

    #[test]
    fn classify_model_lane_separates_supported_experimental_and_unsupported() {
        // Exact supported curated artifact → Supported.
        assert_eq!(
            classify_model_lane(Some("llama"), "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
            ModelLaneClass::Supported,
        );
        assert_eq!(
            classify_model_lane(Some("qwen3"), "Qwen3-0.6B-Q8_0.gguf"),
            ModelLaneClass::Supported,
        );
        // Implemented architecture but NOT a supported exact artifact (different
        // quant/filename) → experimental, never falsely supported.
        assert_eq!(
            classify_model_lane(Some("qwen3"), "Qwen3-0.6B-Q4_K_M.gguf"),
            ModelLaneClass::ExperimentalImplemented,
        );
        assert_eq!(
            classify_model_lane(Some("mistral"), "some-random-mistral-finetune-Q8_0.gguf"),
            ModelLaneClass::ExperimentalImplemented,
        );
        // Architecture not in the implemented set → Unsupported (fails closed at load).
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
                // 12B is supported_exact_row_smoke SCOPED TO the two-Mac
                // distributed layer-sharding serve lane (single-node 16GB is
                // memory-bound); promotion bundle + WebUI closure committed.
                "gemma4_12b_it_q8_0",
                // 26B A4B QAT (Q4_0 experts + Q6_K head) is supported_exact_row_smoke
                // SCOPED TO the same two-Mac distributed lane: full basic_v1 parity
                // pack (2/5 full + 3/5 probe-verified frontiers) + distributed serve
                // smoke committed; 13.4GB row is memory-infeasible single-node.
                "gemma4_26b_a4b_it_q4_0",
                "llama32_1b_instruct_q8_0",
                "llama32_3b_instruct_q8_0",
                "llama3_8b_instruct_q8_0",
                "mistral_7b_instruct_v0_3_q8_0",
                // Dense Qwen3 Q8_0 ChatML rows (thinking disabled): exact-row
                // token+text parity vs llama.cpp at 1/5/50 on macOS/Ubuntu and on
                // Windows x86_64 CPU (cpu_reference + x86_q8 AVX2, bit-identical).
                // Short-chat smoke only; no bounded-context/perf/full support.
                "qwen3_0_6b_instruct_q8_0",
                "qwen3_1_7b_instruct_q8_0",
                "qwen3_4b_instruct_q8_0",
                "qwen3_8b_instruct_q8_0",
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
            BTreeSet::from([
                "llama_bpe_decoder_exact_1b_3b_8b_q8_0",
                "llama_spm_decoder",
                "mistral_instruct_exact_7b_v0_3_q8_0",
                "qwen3_chatml_exact_0_6b_1_7b_4b_8b_q8_0",
            ])
        );

        for id in [
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

    #[tokio::test]
    async fn stream_step_blocking_timeout_reports_generated_count() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "25");
        let session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();

        let result = generate_stream_step_blocking(StreamGenerationStepRequest {
            greedy_fast: false,
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

    #[tokio::test]
    async fn non_streaming_generation_timeout_returns_scrubbed_trace() {
        let _env_guard = crate::test_support::env_lock();
        std::env::set_var(GENERATION_TIMEOUT_ENV, "1");
        std::env::set_var("CAMELID_TEST_GENERATION_STEP_SLEEP_MS", "25");
        let session = LlamaInferenceSession::new(tiny_config(), tiny_weights()).unwrap();
        let prepared = prepared_for_cache("tiny", "model-a.gguf", vec![1, 2], session);

        let result = generate_decoded_tokens_blocking(prepared).await;

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
            layers: vec![crate::model::LlamaLayerTensors {
                attention_norm: desc("blk.0.attn_norm.weight"),
                attention_q: desc("blk.0.attn_q.weight"),
                attention_k: desc("blk.0.attn_k.weight"),
                attention_v: desc("blk.0.attn_v.weight"),
                attention_output: desc("blk.0.attn_output.weight"),
                attention_q_norm: None,
                attention_k_norm: None,
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
            dense_diagnostic_generated_index: None,
            dense_metadata: dummy_dense_metadata(),
            timings: GenerationTimings::default(),
            cached_prompt_prefix: Arc::new(Mutex::new(None)),
            speculative: None,
            telemetry: None,
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
            )
            .unwrap();
        generated.push(first.next_token_id);
        history.push(first.next_token_id);
        let mut accepted_total = 0usize;
        while generated.len() < count {
            let pending = *history.last().unwrap();
            let drafts = drafter.draft(&history, 3).unwrap();
            let base = session.kv_position();
            let mut batch = vec![pending];
            batch.extend_from_slice(&drafts);
            let (predictions, _timings) = session.forward_greedy_verify_chunk(&batch).unwrap();
            let accepted = accepted_draft_prefix(&drafts, &predictions);
            session.rollback_to_position(base + 1 + accepted).unwrap();
            accepted_total += accepted;
            for &token in &predictions[..=accepted] {
                if generated.len() >= count {
                    break;
                }
                generated.push(token);
                history.push(token);
            }
        }

        assert_eq!(generated, vanilla);
        // The tiny model settles into a cycle, so the drafters must have
        // actually accepted drafted tokens — otherwise this test would pass
        // without exercising multi-token acceptance and rollback.
        assert!(
            accepted_total > 0,
            "speculation accepted no drafts; the test did not exercise the accept path"
        );
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
            moe: None,
            gemma4: None,
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
                attention_q_norm: None,
                attention_k_norm: None,
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
    /// `general.architecture` the GGUF will report (authoritative — curated, not
    /// inferred from the filename). Drives the catalog's predicted lane.
    pub architecture: &'static str,
    pub license: &'static str,
}

/// A catalog item plus its predicted runnable lane, so the Models tab can show which
/// lane each entry would land in without downloading it. `oracle_qualified` means the
/// `(architecture, quant)` combo is anchored → it would be Compatible after download
/// (unless it also matches a supported contract row, which the frontend resolves).
///
/// Owns its strings so it can carry both **curated** rows (authoritative metadata,
/// `arch_detected: true`) and **experimental** live Hugging Face results
/// (`arch_detected: false`, where `architecture`/`quant` are filename guesses). The
/// JSON field names are unchanged from the previous flattened `CatalogItem` shape;
/// `group`/`arch_detected` are additive. Experimental rows always report
/// `oracle_qualified: false` — a filename guess can never anchor a lane, so the
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
}

impl CatalogItemView {
    /// Build a view for a curated row: authoritative architecture, lane predicted
    /// from the real `(architecture, quant)` via `runnable::oracle_qualified`.
    fn from_curated(item: &CatalogItem) -> Self {
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
        }
    }

    /// Build a view for a live Hugging Face result. Architecture/quant are filename
    /// guesses (`arch_detected: false`); `oracle_qualified` is forced `false` so the
    /// experimental row is never predicted Compatible/Supported on a guess alone.
    fn from_hf(file: crate::hf_browse::HfGgufFile) -> Self {
        let catalog_id = format!("hf::{}::{}", file.repo_id, file.filename);
        CatalogItemView {
            catalog_id,
            name: file.filename.clone(),
            repo_id: file.repo_id,
            filename: file.filename,
            size_bytes: file.size_bytes,
            downloads: file.downloads,
            likes: file.likes,
            quant: file.quant,
            architecture: file.architecture,
            license: String::new(),
            oracle_qualified: false,
            group: "experimental",
            arch_detected: false,
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
        },
        CatalogItem {
            catalog_id: "phi3_mini_4k_instruct_q8_0",
            name: "Phi-3-mini-4k-instruct Q8_0",
            repo_id: "bartowski/Phi-3-mini-4k-instruct-GGUF",
            filename: "Phi-3-mini-4k-instruct-Q8_0.gguf",
            size_bytes: 4061222688,
            downloads: 0,
            likes: 0,
            quant: "Q8_0",
            architecture: "phi3",
            license: "mit",
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

/// One local on-disk GGUF with the facts the Models tab needs to bucket it by lane.
/// Cheap to compute: header-only metadata + admission, no model load or generation,
/// no whole-file hashing. The frontend derives "supported" by matching against the
/// `/api/capabilities` contract; the runnable-lane facts come from here.
#[derive(Debug, serde::Serialize)]
pub struct LocalModelEntry {
    pub filename: String,
    pub size_bytes: u64,
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
    /// anchored" — never presented as compatible.
    pub oracle_qualified: bool,
    /// A cached runnable smoke receipt already exists for this file (it passed
    /// smoke-admission before) — i.e. it belongs in the Compatible section.
    pub runnable_receipt_present: bool,
    /// The GGUF ships a chat template — i.e. it is an instruction-tuned chat model
    /// (vs a base text-completion model). A model capability, not a system fact.
    pub chat_capable: bool,
    /// Trained context window (tokens) from the GGUF — a model capability.
    pub context_length: Option<u32>,
    /// Server-computed lane class from real header metadata (architecture) + the
    /// exact-artifact supported-row check. `experimental_implemented` means the
    /// architecture is implemented but this is NOT a supported row — attemptable,
    /// unverified, no parity claim. Corroborates the frontend contract gate; it
    /// never promotes a row.
    pub lane_class: ModelLaneClass,
}

#[derive(Debug, serde::Serialize)]
pub struct LocalModelsResponse {
    /// Repo-relative models directory the scan covered.
    pub models_dir: String,
    pub models: Vec<LocalModelEntry>,
}

/// Cached metadata-derived facts for one local model, keyed by (mtime, size) so a
/// re-download invalidates it. Parsing a GGUF's tensor index is slow for big models
/// (the 16 GB MoE alone is seconds), and this endpoint is polled — so we re-parse
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

/// `GET /api/models/local` — enumerate `models/*.gguf` with per-model lane facts.
/// Membership is derived downstream from these facts; nothing here is hand-authored.
async fn local_models() -> Json<LocalModelsResponse> {
    let dir = PathBuf::from("models");
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .collect();
        paths.sort();
        for path in paths {
            let filename = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let fs_meta = std::fs::metadata(&path).ok();
            let size_bytes = fs_meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime_secs = fs_meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

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
        models_dir: "models".to_string(),
        models,
    })
}

#[derive(Debug, serde::Deserialize)]
struct RunnableReceiptQuery {
    filename: String,
}

/// `GET /api/models/runnable-receipt?filename=<gguf>` — return the cached runnable
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

/// `POST /api/models/runnable-smoke {filename}` — run smoke-admission on a local
/// model (admit -> load -> forward sanity -> coherence), cache + return the runnable
/// receipt on pass. User-initiated; the model joins the Compatible section after.
/// CPU-heavy (~minute) so it runs on a blocking thread.
async fn run_runnable_smoke(Json(req): Json<RunnableSmokeRequest>) -> Response {
    let filename = req.filename;
    if filename.is_empty()
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains("..")
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_filename",
            "filename must be a bare GGUF name resolved under models/".to_string(),
            None,
        );
    }
    let path = PathBuf::from("models").join(&filename);
    if !path.exists() {
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("{filename} is not present in models/"),
            None,
        );
    }
    let path_str = path.to_string_lossy().to_string();
    let result = tokio::task::spawn_blocking(move || crate::runnable::smoke_admit(&path_str)).await;
    match result {
        Ok(Ok(report)) => {
            let out = runnable_smoke_receipt_path(&filename);
            if let Some(parent) = out.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(&report.receipt) {
                let _ = std::fs::write(&out, json);
            }
            (
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
                .into_response()
        }
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
/// metadata (`general.architecture`) and the exact-artifact supported-row check —
/// never a filename guess of the architecture. Drives experimental UI copy only; it
/// never promotes a row or widens what `LlamaModelConfig::from_gguf` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLaneClass {
    /// Exact supported row (asserted by `/api/capabilities`) — the full supported lane.
    Supported,
    /// Architecture is implemented but this is NOT a supported row: attemptable,
    /// unverified, no parity claim.
    ExperimentalImplemented,
    /// Architecture is not in the implemented set — fails closed at load.
    Unsupported,
}

/// The `id`s of `/api/capabilities` compatibility rows whose status is `supported*`.
/// Memoized — these are static contract literals. Reads the SAME rows the contract
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

/// True when `filename` is the exact GGUF artifact of a curated row whose
/// `catalog_id` is a `supported_*` compatibility row. The ledger is exact-artifact
/// gated, so an exact-filename match is the honest server-side "is this a supported
/// row?" test. Deliberately conservative: a supported model loaded under a
/// non-curated filename classifies as experimental, never falsely as supported.
fn filename_is_supported_exact_row(filename: &str) -> bool {
    let supported = supported_compatibility_row_ids();
    curated_catalog()
        .iter()
        .any(|c| c.filename == filename && supported.contains(c.catalog_id))
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
        _ => "invalid_model",
    }
}

/// The Models-tab catalog: a **curated** group (pinned, known-good rows, always
/// first) followed, when the user is searching, by an **experimental** group of
/// live Hugging Face results. Experimental rows are advisory-only — `oracle_qualified`
/// is forced false and the frontend marks them unverified — so live browse can never
/// widen what Camelid claims. A non-trivial `query` (≥2 chars) triggers the HF search;
/// a network failure degrades silently to curated-only.
async fn get_catalog(
    axum::extract::Query(q): axum::extract::Query<CatalogQuery>,
) -> Json<CatalogResponse> {
    let query = q.query.unwrap_or_default();
    let trimmed = query.trim();

    // Curated group: filter by query exactly as before, always emitted first.
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
        .map(CatalogItemView::from_curated)
        .collect();

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
                items.extend(
                    page.files
                        .into_iter()
                        .filter(|f| {
                            !curated_files.contains(&(f.repo_id.clone(), f.filename.clone()))
                        })
                        .map(CatalogItemView::from_hf),
                );
            }
            // Offline / Hub error: keep curated-only rather than failing the page.
            Err(err) => eprintln!("hugging face browse unavailable: {err}"),
        }
    }

    Json(CatalogResponse { items, next_cursor })
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
    // Download into a `.part` file and only promote it to the final path once curl
    // exits successfully. The loadable GGUF therefore never exists until the
    // download is genuinely complete, so a half-downloaded model cannot be loaded.
    let part_path = format!("{dest_path}.part");
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        req.repo_id, req.filename
    );

    // `-f` makes curl FAIL on an HTTP error (404/403/…) instead of writing the
    // error page to the output file and exiting 0 — which previously looked like an
    // instant successful download.
    match std::process::Command::new("curl")
        .args(["-f", "-L", "-C", "-", "-o", &part_path, &url])
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
            let part_path_clone = part_path.clone();
            let dest_path_clone = dest_path.clone();
            tokio::spawn(async move {
                let mut child = child;
                let succeeded = matches!(child.wait(), Ok(status) if status.success());
                // Completion is the curl exit code AND a successful promote of the
                // .part file to the final path — never a size heuristic.
                let promoted =
                    succeeded && std::fs::rename(&part_path_clone, &dest_path_clone).is_ok();
                if !promoted {
                    // Leave nothing loadable behind on failure/cancel.
                    std::fs::remove_file(&part_path_clone).ok();
                }
                let mut map = active_downloads_map().lock().unwrap();
                if let Some(dl) = map.get_mut(&catalog_id_clone) {
                    dl.status = if promoted { "completed" } else { "failed" };
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

        // Progress comes from the in-flight `.part` file. Completion is driven
        // ONLY by curl's exit code (set in the spawn task above), never by a size
        // comparison against the catalog's approximate `size_bytes`, which could
        // flip a download to "completed" before it actually finished.
        let part_path = format!("models/{}.part", dl.filename);
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
