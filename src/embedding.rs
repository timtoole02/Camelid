//! Bidirectional encoder execution for embedding and semantic-search models.
//!
//! The first admitted runtime is the exact Nomic-BERT graph used by
//! `nomic-embed-text-v1.5`: WordPiece input, token/type embeddings, split-half
//! rotary full attention, gated-SiLU feed-forward blocks, and configurable
//! pooling. Q8_0 matrices stay quantized and use the same block-dot linear path
//! as generative inference.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use rayon::prelude::*;

use crate::gguf::{read_metadata, GgufFile, GgufTensorType};
use crate::inference::linear_for_role_runtime;
use crate::model::{is_bitnet_embedding_model, LlamaModelConfig};
use crate::runnable::RunnableModel;
use crate::tensor::{CpuTensor, TensorStore};
use crate::tokenizer::Tokenizer;
use crate::{BackendError, Result};

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(target_os = "macos")]
mod metal;

const NOMIC_BERT_ARCH: &str = "nomic-bert";
const MAX_BATCH_INPUTS: usize = 256;
// The checkpoints advertise 32K training context, but Camelid's current
// embedding graph uses host-side quadratic causal attention. Keep the public
// runtime inside the real-model-tested interactive envelope until a resident
// flash-attention path exists.
const MAX_BITNET_EMBEDDING_INPUT_TOKENS: usize = 1_024;
const MAX_BITNET_EMBEDDING_BATCH_TOKENS: usize = 4_096;
const DEFAULT_BATCH_WORKERS: usize = 8;
const MAX_BATCH_WORKERS: usize = 16;
const BATCH_WORKERS_ENV: &str = "CAMELID_EMBEDDING_BATCH_WORKERS";

fn embedding_batch_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let configured = std::env::var(BATCH_WORKERS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|workers| (1..=MAX_BATCH_WORKERS).contains(workers))
            .unwrap_or(DEFAULT_BATCH_WORKERS);
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads().clamp(1, configured))
            .thread_name(|index| format!("camelid-embedding-{index}"))
            .build()
            .expect("embedding worker pool")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingType {
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl PoolingType {
    fn from_gguf(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Mean),
            2 => Ok(Self::Cls),
            3 => Ok(Self::Last),
            4 => Ok(Self::Rank),
            other => Err(BackendError::InvalidModelMetadata(format!(
                "unsupported encoder pooling type {other}; expected 0 (none), 1 (mean), 2 (cls), 3 (last), or 4 (rank)"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Mean => "mean",
            Self::Cls => "cls",
            Self::Last => "last",
            Self::Rank => "rank",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub architecture: String,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub block_count: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub layer_norm_epsilon: f32,
    pub rope_frequency_base: f32,
    pub pooling: PoolingType,
}

#[derive(Debug, Clone)]
pub struct EncoderOutput {
    pub token_ids: Vec<u32>,
    pub token_embeddings: Vec<f32>,
    pub token_count: usize,
    pub embedding_length: usize,
}

impl EncoderOutput {
    pub fn pool(&self, pooling: PoolingType) -> Result<Vec<f32>> {
        if self.token_count == 0 || self.embedding_length == 0 {
            return Err(BackendError::RuntimeShapeMismatch(
                "encoder output cannot pool an empty tensor".to_string(),
            ));
        }
        if self.token_embeddings.len() != self.token_count * self.embedding_length {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "encoder output has {} values, expected {} x {}",
                self.token_embeddings.len(),
                self.token_count,
                self.embedding_length
            )));
        }

        let mut pooled = vec![0.0_f32; self.embedding_length];
        match pooling {
            PoolingType::None => {
                return Err(BackendError::InvalidModelMetadata(
                    "pooling=none produces token embeddings and cannot be represented as one embedding vector"
                        .to_string(),
                ));
            }
            PoolingType::Mean => {
                for row in self.token_embeddings.chunks_exact(self.embedding_length) {
                    for (out, value) in pooled.iter_mut().zip(row) {
                        *out += *value;
                    }
                }
                let scale = 1.0 / self.token_count as f32;
                for value in &mut pooled {
                    *value *= scale;
                }
            }
            PoolingType::Cls => {
                pooled.copy_from_slice(&self.token_embeddings[..self.embedding_length]);
            }
            PoolingType::Last => {
                let start = (self.token_count - 1) * self.embedding_length;
                pooled
                    .copy_from_slice(&self.token_embeddings[start..start + self.embedding_length]);
            }
            PoolingType::Rank => {
                return Err(BackendError::InvalidModelMetadata(
                    "rank pooling requires a classifier head; this Nomic embedding row has no classifier head"
                        .to_string(),
                ));
            }
        }
        l2_normalize(&mut pooled)?;
        Ok(pooled)
    }
}

#[derive(Clone)]
struct EncoderLayer {
    qkv: CpuTensor,
    attention_output: CpuTensor,
    attention_norm_weight: Vec<f32>,
    attention_norm_bias: Vec<f32>,
    ffn_gate: CpuTensor,
    ffn_up: CpuTensor,
    ffn_down: CpuTensor,
    output_norm_weight: Vec<f32>,
    output_norm_bias: Vec<f32>,
}

/// A CPU Nomic-BERT runtime whose Q8 matrices remain resident in quantized form.
pub struct NomicBertRuntime {
    path: PathBuf,
    config: EncoderConfig,
    tokenizer: Arc<Tokenizer>,
    token_embedding: CpuTensor,
    token_type_embedding: Vec<f32>,
    embedding_norm_weight: Vec<f32>,
    embedding_norm_bias: Vec<f32>,
    layers: Vec<EncoderLayer>,
    /// The CUDA layer stack used for document indexing, built on first use.
    #[cfg(feature = "cuda")]
    cuda: cuda::Slot,
    #[cfg(target_os = "macos")]
    metal: metal::Slot,
}

/// API-facing embedding runtime. The enum keeps the established Nomic encoder
/// intact while admitting Microsoft's two exact decoder-only BitNet embedding
/// artifacts through the generic runnable graph.
pub enum EmbeddingRuntime {
    Nomic(Box<NomicBertRuntime>),
    BitNet(Box<BitNetEmbeddingRuntime>),
}

/// Where a batch of document embeddings was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderLane {
    Cpu,
    Cuda,
    Metal,
}

impl EncoderLane {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

impl std::fmt::Debug for EmbeddingRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nomic(runtime) => runtime.fmt(formatter),
            Self::BitNet(runtime) => runtime.fmt(formatter),
        }
    }
}

impl EmbeddingRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = read_metadata(path)?;
        if gguf.architecture() == Some(NOMIC_BERT_ARCH) {
            return NomicBertRuntime::load(path).map(Box::new).map(Self::Nomic);
        }
        if is_bitnet_embedding_model(&gguf) {
            return BitNetEmbeddingRuntime::load(path)
                .map(Box::new)
                .map(Self::BitNet);
        }
        Err(BackendError::UnsupportedModelArchitecture(format!(
            "embedding runtime supports {NOMIC_BERT_ARCH:?} and the exact Microsoft BitNet embedding GGUFs; got architecture {:?}, model {:?}",
            gguf.architecture().unwrap_or("unknown"),
            gguf.model_name().unwrap_or("unknown")
        )))
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Nomic(runtime) => runtime.tokenizer(),
            Self::BitNet(runtime) => runtime.tokenizer(),
        }
    }

    /// Validate the optional OpenAI-compatible `dimensions` override before
    /// running an encoder so API errors can identify that field precisely.
    pub fn validate_dimensions(&self, dimensions: Option<usize>) -> Result<()> {
        let Some(dimensions) = dimensions else {
            return Ok(());
        };
        match self {
            Self::Nomic(runtime) => {
                let native = runtime.config().embedding_length;
                if dimensions == 0 || dimensions > native {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "embedding dimensions must be between 1 and {native}, got {dimensions}"
                    )));
                }
                Ok(())
            }
            Self::BitNet(runtime) => runtime.validate_dimensions(dimensions),
        }
    }

    pub fn embed_batch(
        &self,
        inputs: &[String],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Nomic(runtime) => runtime.embed_batch(inputs, dimensions),
            Self::BitNet(runtime) => runtime.embed_batch(inputs, dimensions),
        }
    }

    /// One search query for the document index: on the Nomic encoder's GPU
    /// lane when it is free, so the query is embedded like the indexed chunks
    /// were, otherwise as [`Self::embed`].
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        match self {
            Self::Nomic(runtime) => runtime.embed_query(text),
            Self::BitNet(runtime) => runtime.embed(text, None),
        }
    }

    /// [`Self::embed_batch`] for the document index, which may run the Nomic
    /// encoder on CUDA or Metal. Reports which lane produced the vectors.
    pub fn embed_documents(&self, inputs: &[String]) -> Result<(Vec<Vec<f32>>, EncoderLane)> {
        match self {
            Self::Nomic(runtime) => runtime.embed_documents(inputs),
            Self::BitNet(runtime) => Ok((runtime.embed_batch(inputs, None)?, EncoderLane::Cpu)),
        }
    }

    pub fn embed(&self, text: &str, dimensions: Option<usize>) -> Result<Vec<f32>> {
        match self {
            Self::Nomic(runtime) => runtime.embed(text, dimensions),
            Self::BitNet(runtime) => runtime.embed(text, dimensions),
        }
    }

    /// Prepare the query side of a retrieval or reranking request using the
    /// instruction convention for the loaded model family.
    pub fn prepare_retrieval_query(&self, text: &str) -> String {
        match self {
            Self::Nomic(_) => with_nomic_task_prefix(text, "search_query: "),
            Self::BitNet(_) => with_bitnet_query_instruction(text),
        }
    }

    /// Prepare the document side of a retrieval or reranking request. BitNet
    /// embeddings are trained without a document-side instruction.
    pub fn prepare_retrieval_document(&self, text: &str) -> String {
        match self {
            Self::Nomic(_) => with_nomic_task_prefix(text, "search_document: "),
            Self::BitNet(_) => text.trim().to_string(),
        }
    }
}

fn with_nomic_task_prefix(text: &str, prefix: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("search_query:")
        || trimmed.starts_with("search_document:")
        || trimmed.starts_with("clustering:")
        || trimmed.starts_with("classification:")
    {
        trimmed.to_string()
    } else {
        format!("{prefix}{trimmed}")
    }
}

fn with_bitnet_query_instruction(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("query:") {
        trimmed.to_string()
    } else {
        format!("query: {trimmed}")
    }
}

fn resolve_bitnet_embedding_pooling(declared: PoolingType) -> Result<PoolingType> {
    match declared {
        PoolingType::Mean | PoolingType::Cls | PoolingType::Last => Ok(declared),
        PoolingType::None => Err(BackendError::InvalidModelMetadata(
            "BitNet embedding pooling=none cannot produce one API embedding vector".into(),
        )),
        PoolingType::Rank => Err(BackendError::InvalidModelMetadata(
            "BitNet embedding pooling=rank requires a classifier head".into(),
        )),
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BitNetEmbeddingConfig {
    pub(crate) context_length: usize,
    pub(crate) embedding_length: usize,
    pooling: PoolingType,
}

/// Header-only admission shared by `/api/models/inspect` and the real loader.
/// Keeping these checks in one place prevents the UI from declaring a malformed
/// look-alike ready only for weight loading to reject it later.
pub(crate) fn validate_bitnet_embedding_metadata(gguf: &GgufFile) -> Result<BitNetEmbeddingConfig> {
    if !is_bitnet_embedding_model(gguf) {
        return Err(BackendError::UnsupportedModelArchitecture(format!(
            "not an exact Microsoft BitNet embedding GGUF: architecture {:?}, model {:?}",
            gguf.architecture().unwrap_or("unknown"),
            gguf.model_name().unwrap_or("unknown")
        )));
    }
    let llama = LlamaModelConfig::from_gguf(gguf)?;
    let expected = match gguf.model_name() {
        Some("bitnet-embeddings-0.6b") => ("qwen3", 32_768, 1_024, 3_072, 28, 16, 8, 128),
        Some("bitnet-embeddings-270m") => ("gemma3", 32_768, 640, 2_048, 18, 4, 1, 256),
        _ => unreachable!("exact BitNet embedding model checked above"),
    };
    let actual = (
        llama.architecture.as_str(),
        llama.context_length as usize,
        llama.embedding_length as usize,
        llama.feed_forward_length as usize,
        llama.block_count as usize,
        llama.attention_head_count as usize,
        llama.attention_head_count_kv as usize,
        llama.attention_key_length.unwrap_or(0) as usize,
    );
    if actual != expected {
        return Err(BackendError::InvalidModelMetadata(format!(
            "BitNet embedding geometry {actual:?} does not match the pinned official artifact {expected:?}"
        )));
    }
    if gguf.metadata_u32("general.file_type") != Some(40) {
        return Err(BackendError::InvalidModelMetadata(
            "BitNet embedding GGUF must declare general.file_type=40 (I2_S)".into(),
        ));
    }
    if gguf
        .tensors
        .iter()
        .any(|tensor| tensor.name == "output.weight")
    {
        return Err(BackendError::InvalidModelMetadata(
            "BitNet embedding artifact unexpectedly contains output.weight".into(),
        ));
    }
    let projection_suffixes = [
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    ];
    let projection_count = gguf
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("blk.")
                && projection_suffixes
                    .iter()
                    .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| {
            if tensor.tensor_type != GgufTensorType::I2S {
                Err(BackendError::UnsupportedTensorType(format!(
                    "BitNet embedding projection {} must be I2_S, got {:?}",
                    tensor.name, tensor.tensor_type
                )))
            } else {
                Ok(())
            }
        })
        .collect::<Result<Vec<_>>>()?
        .len();
    let expected_projections = llama.block_count as usize * projection_suffixes.len();
    if projection_count != expected_projections {
        return Err(BackendError::InvalidModelMetadata(format!(
            "BitNet embedding artifact has {projection_count} dense projections, expected {expected_projections}"
        )));
    }

    let architecture = llama.architecture.as_str();
    // Microsoft's prose says EOS/last-token pooling, but the shipped GGUFs
    // declare pooling_type=1 (mean), and Microsoft's published numeric prefix
    // matches the 0.6B artifact under mean pooling (despite the example command
    // selecting 270M). Treat the artifact metadata as the executable contract.
    let declared_pooling = PoolingType::from_gguf(
        gguf.metadata_u32(&format!("{architecture}.pooling_type"))
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!(
                    "required metadata {architecture}.pooling_type is missing"
                ))
            })?,
    )?;
    let pooling = resolve_bitnet_embedding_pooling(declared_pooling)?;
    Ok(BitNetEmbeddingConfig {
        context_length: expected.1,
        embedding_length: expected.2,
        pooling,
    })
}

/// Runtime for Microsoft's exact BitNet embedding GGUFs. Projection weights
/// remain in canonical I2_S wire form for the cleanroom CPU/Metal/CUDA kernels.
pub struct BitNetEmbeddingRuntime {
    path: PathBuf,
    config: BitNetEmbeddingConfig,
    tokenizer: Arc<Tokenizer>,
    model: RunnableModel,
}

impl std::fmt::Debug for BitNetEmbeddingRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BitNetEmbeddingRuntime")
            .field("path", &self.path)
            .field("config", &self.config)
            .field("tokenizer", &self.tokenizer.model.as_summary_model())
            .finish_non_exhaustive()
    }
}

impl BitNetEmbeddingRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let gguf = read_metadata(&path)?;
        let config = validate_bitnet_embedding_metadata(&gguf)?;
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let model_path = path.to_str().ok_or_else(|| {
            BackendError::InvalidModelMetadata("BitNet model path is not valid UTF-8".into())
        })?;
        let model = RunnableModel::load(model_path)?;
        Ok(Self {
            path,
            config,
            tokenizer,
            model,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn validate_dimensions(&self, dimensions: usize) -> Result<()> {
        if dimensions != self.config.embedding_length {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Microsoft BitNet embeddings have a fixed native dimension of {}; dimension truncation is not supported because these checkpoints do not declare Matryoshka training",
                self.config.embedding_length
            )));
        }
        Ok(())
    }

    pub fn embed(&self, text: &str, dimensions: Option<usize>) -> Result<Vec<f32>> {
        if let Some(dimensions) = dimensions {
            self.validate_dimensions(dimensions)?;
        }
        let token_ids = self.tokenizer.encode(text, true, false)?;
        if token_ids.is_empty() {
            return Err(BackendError::InvalidTokenizerMetadata(
                "BitNet embedding tokenizer produced no tokens".into(),
            ));
        }
        if token_ids.len() > MAX_BITNET_EMBEDDING_INPUT_TOKENS {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding input is {} tokens, above Camelid's tested {}-token BitNet embedding runtime limit (the checkpoint's trained context is {})",
                token_ids.len(),
                MAX_BITNET_EMBEDDING_INPUT_TOKENS,
                self.config.context_length
            )));
        }
        let output = EncoderOutput {
            token_count: token_ids.len(),
            token_embeddings: self.model.forward_hidden_states(&token_ids)?,
            token_ids,
            embedding_length: self.config.embedding_length,
        };
        output.pool(self.config.pooling)
    }

    pub fn embed_batch(
        &self,
        inputs: &[String],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if let Some(dimensions) = dimensions {
            self.validate_dimensions(dimensions)?;
        }
        if inputs.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "embedding input must contain at least one string".into(),
            ));
        }
        if inputs.len() > MAX_BATCH_INPUTS {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding input contains {} strings; maximum batch size is {MAX_BATCH_INPUTS}",
                inputs.len()
            )));
        }
        let token_counts = inputs
            .iter()
            .map(|input| {
                self.tokenizer
                    .encode(input, true, false)
                    .map(|ids| ids.len())
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some((index, count)) = token_counts
            .iter()
            .copied()
            .enumerate()
            .find(|(_, count)| *count > MAX_BITNET_EMBEDDING_INPUT_TOKENS)
        {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding input at index {index} is {count} tokens, above Camelid's tested {MAX_BITNET_EMBEDDING_INPUT_TOKENS}-token BitNet embedding runtime limit"
            )));
        }
        let batch_tokens = token_counts.iter().sum::<usize>();
        if batch_tokens > MAX_BITNET_EMBEDDING_BATCH_TOKENS {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding batch contains {batch_tokens} tokens; Camelid's tested BitNet batch limit is {MAX_BITNET_EMBEDDING_BATCH_TOKENS} tokens"
            )));
        }

        // A single GPU queue already parallelizes each projection. Running
        // multiple full causal graphs concurrently only multiplies activation
        // memory and transfer pressure, so serialize GPU-backed batches. CPU
        // execution retains the bounded worker pool.
        if crate::bitnet_kernels::gpu_allowed() && crate::cuda::gpu_accel_enabled() {
            inputs
                .iter()
                .map(|input| self.embed(input, dimensions))
                .collect()
        } else {
            embedding_batch_pool().install(|| {
                inputs
                    .par_iter()
                    .map(|input| self.embed(input, dimensions))
                    .collect()
            })
        }
    }
}

impl std::fmt::Debug for NomicBertRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NomicBertRuntime")
            .field("path", &self.path)
            .field("config", &self.config)
            .field("tokenizer", &self.tokenizer.model.as_summary_model())
            .field("layer_count", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl NomicBertRuntime {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let gguf = read_metadata(&path)?;
        let config = EncoderConfig::from_gguf(&gguf)?;
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf)?);
        let store = TensorStore::open(&path, &gguf);

        require_q8_matrix(&gguf, "token_embd.weight")?;
        require_f32_tensor(&gguf, "token_types.weight")?;
        require_f32_tensor(&gguf, "token_embd_norm.weight")?;
        require_f32_tensor(&gguf, "token_embd_norm.bias")?;

        let mut token_embedding = store.load_q8_0_block_backed_linear("token_embd.weight")?;
        require_shape(
            &token_embedding,
            &[config.embedding_length, tokenizer.tokens.len()],
            "token embedding descriptor",
        )?;
        // GGUF describes matrices as [input, output], while `embedding_lookup`
        // consumes a [row_count, row_width] view. The wire bytes already contain
        // one width-sized Q8 row per vocabulary item, so this is a zero-copy
        // shape reinterpretation, not a transpose.
        token_embedding.shape.dims.swap(0, 1);
        let token_type_tensor = store.load_cpu_f32("token_types.weight")?;
        require_shape(
            &token_type_tensor,
            &[config.embedding_length, 2],
            "token type embedding descriptor",
        )?;
        let token_type_embedding = token_type_tensor.data[..config.embedding_length].to_vec();
        let embedding_norm_weight =
            load_vector(&store, "token_embd_norm.weight", config.embedding_length)?;
        let embedding_norm_bias =
            load_vector(&store, "token_embd_norm.bias", config.embedding_length)?;

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let qkv_name = format!("blk.{layer}.attn_qkv.weight");
            let attention_output_name = format!("blk.{layer}.attn_output.weight");
            let ffn_gate_name = format!("blk.{layer}.ffn_gate.weight");
            let ffn_up_name = format!("blk.{layer}.ffn_up.weight");
            let ffn_down_name = format!("blk.{layer}.ffn_down.weight");
            for name in [
                &qkv_name,
                &attention_output_name,
                &ffn_gate_name,
                &ffn_up_name,
                &ffn_down_name,
            ] {
                require_q8_matrix(&gguf, name)?;
            }

            let qkv = store.load_q8_0_block_backed_linear(&qkv_name)?;
            let attention_output = store.load_q8_0_block_backed_linear(&attention_output_name)?;
            let ffn_gate = store.load_q8_0_block_backed_linear(&ffn_gate_name)?;
            let ffn_up = store.load_q8_0_block_backed_linear(&ffn_up_name)?;
            let ffn_down = store.load_q8_0_block_backed_linear(&ffn_down_name)?;
            require_shape(
                &qkv,
                &[config.embedding_length, config.embedding_length * 3],
                "attention qkv",
            )?;
            require_shape(
                &attention_output,
                &[config.embedding_length, config.embedding_length],
                "attention output",
            )?;
            require_shape(
                &ffn_gate,
                &[config.embedding_length, config.feed_forward_length],
                "ffn gate",
            )?;
            require_shape(
                &ffn_up,
                &[config.embedding_length, config.feed_forward_length],
                "ffn up",
            )?;
            require_shape(
                &ffn_down,
                &[config.feed_forward_length, config.embedding_length],
                "ffn down",
            )?;

            let attention_norm_weight = load_vector(
                &store,
                &format!("blk.{layer}.attn_output_norm.weight"),
                config.embedding_length,
            )?;
            let attention_norm_bias = load_vector(
                &store,
                &format!("blk.{layer}.attn_output_norm.bias"),
                config.embedding_length,
            )?;
            let output_norm_weight = load_vector(
                &store,
                &format!("blk.{layer}.layer_output_norm.weight"),
                config.embedding_length,
            )?;
            let output_norm_bias = load_vector(
                &store,
                &format!("blk.{layer}.layer_output_norm.bias"),
                config.embedding_length,
            )?;
            layers.push(EncoderLayer {
                qkv,
                attention_output,
                attention_norm_weight,
                attention_norm_bias,
                ffn_gate,
                ffn_up,
                ffn_down,
                output_norm_weight,
                output_norm_bias,
            });
        }

        Ok(Self {
            path,
            config,
            tokenizer,
            token_embedding,
            token_type_embedding,
            embedding_norm_weight,
            embedding_norm_bias,
            layers,
            #[cfg(feature = "cuda")]
            cuda: cuda::Slot::default(),
            #[cfg(target_os = "macos")]
            metal: metal::Slot::default(),
        })
    }

    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn embed(&self, text: &str, dimensions: Option<usize>) -> Result<Vec<f32>> {
        let output = self.encode(text)?;
        let mut embedding = output.pool(self.config.pooling)?;
        if let Some(dimensions) = dimensions {
            if dimensions == 0 || dimensions > embedding.len() {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "embedding dimensions must be between 1 and {}, got {dimensions}",
                    embedding.len()
                )));
            }
            embedding.truncate(dimensions);
            l2_normalize(&mut embedding)?;
        }
        Ok(embedding)
    }

    pub fn embed_batch(
        &self,
        inputs: &[String],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "embedding input must contain at least one string".to_string(),
            ));
        }
        if inputs.len() > MAX_BATCH_INPUTS {
            return Err(BackendError::InvalidModelMetadata(format!(
                "embedding input contains {} strings; maximum batch size is {MAX_BATCH_INPUTS}",
                inputs.len()
            )));
        }
        embedding_batch_pool().install(|| {
            inputs
                .par_iter()
                .map(|input| self.embed(input, dimensions))
                .collect()
        })
    }

    /// One document-search query. Uses the GPU lane only if it is not busy with
    /// an indexing batch, so a search never waits behind one.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if let Some(mut vectors) = self.metal.embed(self, &[text.to_string()], false)? {
            if let Some(vector) = vectors.pop() {
                return Ok(vector);
            }
        }
        #[cfg(feature = "cuda")]
        if let Some(mut vectors) =
            self.cuda
                .embed(self, std::slice::from_ref(&text.to_string()), false)?
        {
            if let Some(vector) = vectors.pop() {
                return Ok(vector);
            }
        }
        self.embed(text, None)
    }

    /// Embed documents for the document index: on the GPU layer stack when GPU
    /// acceleration is on and the lane builds, otherwise as [`Self::embed_batch`].
    /// GPU execution failures fall back to the CPU and disable the failed lane.
    pub fn embed_documents(&self, inputs: &[String]) -> Result<(Vec<Vec<f32>>, EncoderLane)> {
        #[cfg(target_os = "macos")]
        if !inputs.is_empty() && inputs.len() <= MAX_BATCH_INPUTS {
            if let Some(vectors) = self.metal.embed(self, inputs, true)? {
                return Ok((vectors, EncoderLane::Metal));
            }
        }
        #[cfg(feature = "cuda")]
        if !inputs.is_empty() && inputs.len() <= MAX_BATCH_INPUTS {
            if let Some(vectors) = self.cuda.embed(self, inputs, true)? {
                return Ok((vectors, EncoderLane::Cuda));
            }
        }
        Ok((self.embed_batch(inputs, None)?, EncoderLane::Cpu))
    }

    pub fn encode(&self, text: &str) -> Result<EncoderOutput> {
        let (token_ids, mut hidden) = self.embed_input(text)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden = self.forward_layer(hidden, token_ids.len(), layer_index, layer)?;
        }
        Ok(self.output(token_ids, hidden))
    }

    fn output(&self, token_ids: Vec<u32>, token_embeddings: Vec<f32>) -> EncoderOutput {
        EncoderOutput {
            token_count: token_ids.len(),
            token_ids,
            token_embeddings,
            embedding_length: self.config.embedding_length,
        }
    }

    /// Tokenize `text` and build the encoder's input rows: token embeddings plus
    /// the token-type row, layer-normalized. Everything after this is the layer
    /// stack, which the CUDA lane can run instead.
    fn embed_input(&self, text: &str) -> Result<(Vec<u32>, Vec<f32>)> {
        let token_ids = self.tokenizer.encode(text, true, false)?;
        if token_ids.is_empty() {
            return Err(BackendError::InvalidTokenizerMetadata(
                "encoder tokenizer produced no tokens".to_string(),
            ));
        }
        if token_ids.len() > self.config.context_length {
            return Err(BackendError::InvalidModelMetadata(format!(
                "encoder input is {} tokens, above this model's {}-token context",
                token_ids.len(),
                self.config.context_length
            )));
        }

        let tokens = self
            .token_embedding
            .embedding_lookup(&token_ids, "encoder_token_embedding")?;
        let mut hidden = tokens.data;
        for row in hidden.chunks_exact_mut(self.config.embedding_length) {
            for (value, type_value) in row.iter_mut().zip(&self.token_type_embedding) {
                *value += *type_value;
            }
        }
        layer_norm_in_place(
            &mut hidden,
            token_ids.len(),
            self.config.embedding_length,
            &self.embedding_norm_weight,
            &self.embedding_norm_bias,
            self.config.layer_norm_epsilon,
        )?;
        debug_assert_eq!(tokens.shape.dims[0], token_ids.len());
        Ok((token_ids, hidden))
    }

    fn forward_layer(
        &self,
        hidden: Vec<f32>,
        token_count: usize,
        layer_index: usize,
        layer: &EncoderLayer,
    ) -> Result<Vec<f32>> {
        let width = self.config.embedding_length;
        let input = CpuTensor::from_f32(
            format!("encoder_layer_{layer_index}_input"),
            vec![token_count, width],
            hidden.clone(),
        )?;
        let qkv = linear_for_role_runtime(
            &input,
            &layer.qkv,
            format!("encoder_layer_{layer_index}_qkv"),
            "attention_q",
            false,
        )?;
        require_shape(&qkv, &[token_count, width * 3], "projected qkv")?;
        let attention = self.full_attention(&qkv.data, token_count)?;
        let attention = CpuTensor::from_f32(
            format!("encoder_layer_{layer_index}_attention"),
            vec![token_count, width],
            attention,
        )?;
        let attention_output = linear_for_role_runtime(
            &attention,
            &layer.attention_output,
            format!("encoder_layer_{layer_index}_attention_output"),
            "attention_output",
            false,
        )?;
        let mut attention_residual = attention_output.data;
        add_in_place(&mut attention_residual, &hidden)?;
        layer_norm_in_place(
            &mut attention_residual,
            token_count,
            width,
            &layer.attention_norm_weight,
            &layer.attention_norm_bias,
            self.config.layer_norm_epsilon,
        )?;

        let ffn_input = CpuTensor::from_f32(
            format!("encoder_layer_{layer_index}_ffn_input"),
            vec![token_count, width],
            attention_residual.clone(),
        )?;
        let gate = linear_for_role_runtime(
            &ffn_input,
            &layer.ffn_gate,
            format!("encoder_layer_{layer_index}_ffn_gate"),
            "ffn_gate",
            false,
        )?;
        let up = linear_for_role_runtime(
            &ffn_input,
            &layer.ffn_up,
            format!("encoder_layer_{layer_index}_ffn_up"),
            "ffn_up",
            false,
        )?;
        if gate.shape != up.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "encoder layer {layer_index} gate/up shapes differ: {:?} vs {:?}",
                gate.shape.dims, up.shape.dims
            )));
        }
        let gated = gate
            .data
            .iter()
            .zip(&up.data)
            .map(|(gate, up)| silu(*gate) * *up)
            .collect();
        let gated = CpuTensor::from_f32(
            format!("encoder_layer_{layer_index}_ffn_gated"),
            vec![token_count, self.config.feed_forward_length],
            gated,
        )?;
        let ffn_output = linear_for_role_runtime(
            &gated,
            &layer.ffn_down,
            format!("encoder_layer_{layer_index}_ffn_output"),
            "ffn_down",
            false,
        )?;
        let mut output = ffn_output.data;
        add_in_place(&mut output, &attention_residual)?;
        layer_norm_in_place(
            &mut output,
            token_count,
            width,
            &layer.output_norm_weight,
            &layer.output_norm_bias,
            self.config.layer_norm_epsilon,
        )?;
        Ok(output)
    }

    fn full_attention(&self, qkv: &[f32], token_count: usize) -> Result<Vec<f32>> {
        let width = self.config.embedding_length;
        if qkv.len() != token_count * width * 3 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "encoder qkv has {} values, expected {}",
                qkv.len(),
                token_count * width * 3
            )));
        }
        let head_dim = self.config.head_dim;
        let head_count = self.config.head_count;
        let mut query = vec![0.0_f32; token_count * width];
        let mut key = vec![0.0_f32; token_count * width];
        let mut value = vec![0.0_f32; token_count * width];
        for token in 0..token_count {
            let source = &qkv[token * width * 3..(token + 1) * width * 3];
            query[token * width..(token + 1) * width].copy_from_slice(&source[..width]);
            key[token * width..(token + 1) * width].copy_from_slice(&source[width..width * 2]);
            value[token * width..(token + 1) * width].copy_from_slice(&source[width * 2..]);
        }
        apply_neox_rope(
            &mut query,
            token_count,
            head_count,
            head_dim,
            self.config.rope_frequency_base,
        );
        apply_neox_rope(
            &mut key,
            token_count,
            head_count,
            head_dim,
            self.config.rope_frequency_base,
        );

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0_f32; token_count * width];
        let mut scores = vec![0.0_f32; token_count];
        for query_token in 0..token_count {
            for head in 0..head_count {
                let q_start = query_token * width + head * head_dim;
                let q = &query[q_start..q_start + head_dim];
                for (key_token, score) in scores.iter_mut().enumerate() {
                    let k_start = key_token * width + head * head_dim;
                    *score = dot(q, &key[k_start..k_start + head_dim]) * scale;
                }
                softmax_in_place(&mut scores)?;
                let out_start = query_token * width + head * head_dim;
                let out = &mut output[out_start..out_start + head_dim];
                for (key_token, probability) in scores.iter().copied().enumerate() {
                    let v_start = key_token * width + head * head_dim;
                    for (out_value, value) in
                        out.iter_mut().zip(&value[v_start..v_start + head_dim])
                    {
                        *out_value += probability * *value;
                    }
                }
            }
        }
        Ok(output)
    }
}

impl EncoderConfig {
    pub(crate) fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let architecture = gguf.architecture().unwrap_or_default();
        if architecture != NOMIC_BERT_ARCH {
            return Err(BackendError::UnsupportedModelArchitecture(format!(
                "embedding runtime currently supports {NOMIC_BERT_ARCH:?}, got {architecture:?}"
            )));
        }
        let required_u32 = |suffix: &str| {
            gguf.metadata_u32(&format!("{architecture}.{suffix}"))
                .ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "required metadata {architecture}.{suffix} is missing"
                    ))
                })
                .and_then(|value| {
                    usize::try_from(value).map_err(|_| {
                        BackendError::InvalidModelMetadata(format!(
                            "{architecture}.{suffix} does not fit usize"
                        ))
                    })
                })
        };
        let context_length = required_u32("context_length")?;
        let embedding_length = required_u32("embedding_length")?;
        let feed_forward_length = required_u32("feed_forward_length")?;
        let block_count = required_u32("block_count")?;
        let head_count = required_u32("attention.head_count")?;
        if head_count == 0 || embedding_length % head_count != 0 {
            return Err(BackendError::InvalidModelMetadata(format!(
                "encoder embedding length {embedding_length} is not divisible by head count {head_count}"
            )));
        }
        if gguf.metadata_bool(&format!("{architecture}.attention.causal")) != Some(false) {
            return Err(BackendError::InvalidModelMetadata(
                "Nomic-BERT embedding runtime requires attention.causal=false".to_string(),
            ));
        }
        let layer_norm_epsilon = gguf
            .metadata_f32(&format!("{architecture}.attention.layer_norm_epsilon"))
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(
                    "required Nomic-BERT layer norm epsilon is missing".to_string(),
                )
            })?;
        let rope_frequency_base = gguf
            .metadata_f32(&format!("{architecture}.rope.freq_base"))
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(
                    "required Nomic-BERT RoPE frequency base is missing".to_string(),
                )
            })?;
        if !layer_norm_epsilon.is_finite()
            || layer_norm_epsilon <= 0.0
            || !rope_frequency_base.is_finite()
            || rope_frequency_base <= 0.0
        {
            return Err(BackendError::InvalidModelMetadata(
                "Nomic-BERT normalization/RoPE metadata must be finite and positive".to_string(),
            ));
        }
        let pooling = PoolingType::from_gguf(
            gguf.metadata_u32(&format!("{architecture}.pooling_type"))
                .ok_or_else(|| {
                    BackendError::InvalidModelMetadata(
                        "required Nomic-BERT pooling type is missing".to_string(),
                    )
                })?,
        )?;
        Ok(Self {
            architecture: architecture.to_string(),
            context_length,
            embedding_length,
            feed_forward_length,
            block_count,
            head_count,
            head_dim: embedding_length / head_count,
            layer_norm_epsilon,
            rope_frequency_base,
            pooling,
        })
    }
}

fn require_q8_matrix(gguf: &GgufFile, name: &str) -> Result<()> {
    let descriptor = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))?;
    if descriptor.tensor_type != GgufTensorType::Q8_0 || descriptor.dimensions.len() != 2 {
        return Err(BackendError::UnsupportedTensorType(format!(
            "encoder matrix {name} must be rank-2 Q8_0, got {:?} with rank {}",
            descriptor.tensor_type,
            descriptor.dimensions.len()
        )));
    }
    Ok(())
}

fn require_f32_tensor(gguf: &GgufFile, name: &str) -> Result<()> {
    let descriptor = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))?;
    if descriptor.tensor_type != GgufTensorType::F32 {
        return Err(BackendError::UnsupportedTensorType(format!(
            "encoder tensor {name} must be F32, got {:?}",
            descriptor.tensor_type
        )));
    }
    Ok(())
}

fn load_vector(store: &TensorStore, name: &str, length: usize) -> Result<Vec<f32>> {
    let tensor = store.load_cpu_f32(name)?;
    require_shape(&tensor, &[length], name)?;
    Ok(tensor.data)
}

fn require_shape(tensor: &CpuTensor, expected: &[usize], role: &str) -> Result<()> {
    if tensor.shape.dims != expected {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{role} tensor {} expected shape {expected:?}, got {:?}",
            tensor.name, tensor.shape.dims
        )));
    }
    Ok(())
}

fn layer_norm_in_place(
    data: &mut [f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<()> {
    if data.len() != rows * cols || weight.len() != cols || bias.len() != cols {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "layer norm shape mismatch: data={}, rows={rows}, cols={cols}, weight={}, bias={}",
            data.len(),
            weight.len(),
            bias.len()
        )));
    }
    for row in data.chunks_exact_mut(cols) {
        let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / cols as f64;
        let variance = row
            .iter()
            .map(|value| {
                let delta = f64::from(*value) - mean;
                delta * delta
            })
            .sum::<f64>()
            / cols as f64;
        let inverse_stddev = 1.0 / (variance + f64::from(epsilon)).sqrt();
        for ((value, weight), bias) in row.iter_mut().zip(weight).zip(bias) {
            *value = ((f64::from(*value) - mean) * inverse_stddev) as f32 * *weight + *bias;
        }
    }
    Ok(())
}

fn apply_neox_rope(
    data: &mut [f32],
    token_count: usize,
    head_count: usize,
    head_dim: usize,
    base: f32,
) {
    debug_assert_eq!(data.len(), token_count * head_count * head_dim);
    debug_assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    for position in 0..token_count {
        for head in 0..head_count {
            let start = (position * head_count + head) * head_dim;
            for pair in 0..half {
                let frequency = 1.0 / base.powf((2 * pair) as f32 / head_dim as f32);
                let angle = position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let left_index = start + pair;
                let right_index = start + pair + half;
                let left = data[left_index];
                let right = data[right_index];
                data[left_index] = left * cos - right * sin;
                data[right_index] = left * sin + right * cos;
            }
        }
    }
}

fn softmax_in_place(values: &mut [f32]) -> Result<()> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(BackendError::RuntimeShapeMismatch(
            "encoder attention softmax produced an invalid normalization sum".to_string(),
        ));
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

fn l2_normalize(values: &mut [f32]) -> Result<()> {
    let norm = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(BackendError::RuntimeShapeMismatch(
            "embedding has zero or non-finite L2 norm".to_string(),
        ));
    }
    for value in values {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn add_in_place(output: &mut [f32], residual: &[f32]) -> Result<()> {
    if output.len() != residual.len() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "residual add length mismatch: {} vs {}",
            output.len(),
            residual.len()
        )));
    }
    for (output, residual) in output.iter_mut().zip(residual) {
        *output += *residual;
    }
    Ok(())
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[inline]
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() || left.is_empty() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "cosine similarity requires equal non-empty vectors, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    let mut product = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        left_norm += f64::from(*left) * f64::from(*left);
        right_norm += f64::from(*right) * f64::from(*right);
        product += f64::from(*left) * f64::from(*right);
    }
    let denominator = (left_norm * right_norm).sqrt();
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return Err(BackendError::RuntimeShapeMismatch(
            "cosine similarity received a zero or non-finite vector".to_string(),
        ));
    }
    Ok((product / denominator) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neox_rope_position_zero_is_identity_and_position_one_rotates_split_halves() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        apply_neox_rope(&mut values, 2, 1, 4, 1000.0);
        assert_eq!(&values[..4], &[1.0, 2.0, 3.0, 4.0]);
        let expected_frequency = 1.0 / 1000.0_f32.powf(0.5);
        let (sin0, cos0) = 1.0_f32.sin_cos();
        let (sin1, cos1) = expected_frequency.sin_cos();
        assert!((values[4] - (1.0 * cos0 - 3.0 * sin0)).abs() < 1e-6);
        assert!((values[6] - (1.0 * sin0 + 3.0 * cos0)).abs() < 1e-6);
        assert!((values[5] - (2.0 * cos1 - 4.0 * sin1)).abs() < 1e-6);
        assert!((values[7] - (2.0 * sin1 + 4.0 * cos1)).abs() < 1e-6);
    }

    #[test]
    fn pooling_mean_cls_last_and_dimension_truncation_normalize() {
        let output = EncoderOutput {
            token_ids: vec![1, 2],
            token_embeddings: vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0],
            token_count: 2,
            embedding_length: 3,
        };
        let cls = output.pool(PoolingType::Cls).unwrap();
        assert!((cls[0] - 0.6).abs() < 1e-6);
        assert!((cls[1] - 0.8).abs() < 1e-6);
        let last = output.pool(PoolingType::Last).unwrap();
        assert_eq!(last, vec![0.0, 0.0, 1.0]);
        let mean = output.pool(PoolingType::Mean).unwrap();
        let norm = mean.iter().map(|value| value * value).sum::<f32>();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(output.pool(PoolingType::None).is_err());
        assert!(output.pool(PoolingType::Rank).is_err());
    }

    #[test]
    fn bitnet_embedding_pooling_honors_supported_artifact_metadata() {
        for pooling in [PoolingType::Mean, PoolingType::Cls, PoolingType::Last] {
            assert_eq!(resolve_bitnet_embedding_pooling(pooling).unwrap(), pooling);
        }
        assert!(resolve_bitnet_embedding_pooling(PoolingType::None).is_err());
        assert!(resolve_bitnet_embedding_pooling(PoolingType::Rank).is_err());
    }

    #[test]
    fn cosine_similarity_has_expected_ordering() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn retrieval_instructions_match_each_embedding_family() {
        assert_eq!(
            with_nomic_task_prefix("  camelids  ", "search_query: "),
            "search_query: camelids"
        );
        assert_eq!(
            with_nomic_task_prefix("search_document: alpaca", "search_query: "),
            "search_document: alpaca"
        );
        assert_eq!(
            with_bitnet_query_instruction("  What is BitNet?  "),
            "query: What is BitNet?"
        );
        assert_eq!(
            with_bitnet_query_instruction("query: What is BitNet?"),
            "query: What is BitNet?"
        );
    }

    /// #801: the CUDA document encoder against the CPU, on real library chunks.
    /// CAMELID_NOMIC_GGUF = the encoder, CAMELID_BENCH_CHUNKS = JSON lines of chunk text.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs CAMELID_NOMIC_GGUF, CAMELID_BENCH_CHUNKS and a CUDA device"]
    fn cuda_document_vectors_match_the_cpu() {
        let (Ok(model), Ok(chunks)) = (
            std::env::var("CAMELID_NOMIC_GGUF"),
            std::env::var("CAMELID_BENCH_CHUNKS"),
        ) else {
            return;
        };
        let runtime = EmbeddingRuntime::load(&model).expect("load encoder");
        let texts: Vec<String> = std::fs::read_to_string(&chunks)
            .expect("chunks")
            .lines()
            .map(|line| serde_json::from_str::<String>(line).expect("json string"))
            .map(|text| runtime.prepare_retrieval_document(&text))
            .collect();
        let EmbeddingRuntime::Nomic(nomic) = &runtime else {
            panic!("not the Nomic encoder");
        };
        let mut gpu = super::cuda::CudaEncoder::build_for_test(nomic).expect("CUDA encoder");
        let width = nomic.config.embedding_length;

        // One Q8 projection returns exactly the CPU's bits.
        let (_, rows_in) = nomic.embed_input(&texts[0]).unwrap();
        let rows = rows_in.len() / width;
        let input = CpuTensor::from_f32("probe", vec![rows, width], rows_in.clone()).unwrap();
        let cpu_qkv = linear_for_role_runtime(
            &input,
            &nomic.layers[0].qkv,
            "probe_qkv",
            "attention_q",
            false,
        )
        .unwrap()
        .data;
        let gpu_qkv = gpu.qkv_for_test(0, &rows_in, rows).unwrap();
        assert_eq!(cpu_qkv.len(), gpu_qkv.len());
        let differing = cpu_qkv
            .iter()
            .zip(&gpu_qkv)
            .filter(|(cpu, gpu)| cpu.to_bits() != gpu.to_bits())
            .count();
        eprintln!(
            "qkv projection: {differing} of {} values differ from the CPU",
            cpu_qkv.len()
        );
        assert_eq!(differing, 0);

        let prepared: Vec<_> = texts
            .iter()
            .map(|text| nomic.embed_input(text).unwrap())
            .collect();

        // Throughput, after a warm-up forward.
        gpu.forward_for_test(&prepared[..8]).unwrap();
        let started = std::time::Instant::now();
        for batch in prepared.chunks(128) {
            gpu.forward_for_test(batch).unwrap();
        }
        let secs = started.elapsed().as_secs_f64();
        eprintln!(
            "CUDA layer stack: {} chunks in {secs:.2} s = {:.0} chunks/s",
            prepared.len(),
            prepared.len() as f64 / secs
        );

        // Whole vectors: the CPU encoder against the CUDA layer stack. The CPU
        // side is slow in a test build, so CAMELID_BENCH_COMPARE can lower the count.
        let compared = std::env::var("CAMELID_BENCH_COMPARE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(48)
            .min(texts.len());
        if compared == 0 {
            return;
        }
        let pool = |prepared: Vec<(Vec<u32>, Vec<f32>)>, hidden: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            prepared
                .into_iter()
                .zip(hidden)
                .map(|((ids, _), hidden)| {
                    nomic
                        .output(ids, hidden)
                        .pool(nomic.config.pooling)
                        .unwrap()
                })
                .collect()
        };
        let hidden = gpu.forward_for_test(&prepared[..compared]).unwrap();
        let gpu_vectors = pool(prepared[..compared].to_vec(), hidden);
        let (mut min_cosine, mut max_abs) = (f32::INFINITY, 0.0_f32);
        for (text, gpu_vector) in texts.iter().zip(&gpu_vectors) {
            let cpu_vector = nomic.embed(text, None).unwrap();
            min_cosine = min_cosine.min(cosine_similarity(&cpu_vector, gpu_vector).unwrap());
            for (cpu, gpu) in cpu_vector.iter().zip(gpu_vector) {
                max_abs = max_abs.max((cpu - gpu).abs());
            }
        }
        eprintln!("{compared} chunks: min cosine {min_cosine:.7}, max |cpu - gpu| {max_abs:.2e}");
        assert!(min_cosine >= 0.9999, "min cosine {min_cosine}");
    }
}
