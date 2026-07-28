use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{BackendError, Result};

/// Source-level model formats that Camelid can reason about before runtime loading.
///
/// This layer is intentionally narrower than generation support: a Hugging Face
/// SafeTensors directory can be detected and reported without becoming runnable.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSourceKind {
    Gguf,
    HuggingFaceSafeTensors,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSourceManifest {
    pub id: String,
    pub kind: ModelSourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_config: Option<HfLlamaConfigSummary>,
    #[serde(skip_serializing)]
    pub root: PathBuf,
    #[serde(skip_serializing)]
    pub weight_files: Vec<PathBuf>,
    pub tensor_descriptors: Vec<SafeTensorsTensorDescriptor>,
    #[serde(skip_serializing)]
    pub config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub tokenizer_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub tokenizer_config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub special_tokens_map_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub generation_config_path: Option<PathBuf>,
    #[serde(skip_serializing)]
    pub shard_index_path: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HfLlamaConfigSummary {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    /// Explicit per-head dimension from HF `config.json` `head_dim`, when set.
    /// `None` falls back to `hidden_size / num_attention_heads`. Newer Llama
    /// configs may set a `head_dim` that differs from that ratio, so it is
    /// captured rather than assumed.
    pub head_dim: Option<u32>,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: u32,
    pub tie_word_embeddings: bool,
}

impl HfLlamaConfigSummary {
    /// Build a dense-LLaMA runtime config from this validated Hugging Face
    /// `config.json` summary — the HF counterpart to
    /// [`crate::model::LlamaModelConfig::from_gguf`].
    ///
    /// The summary is produced by [`inspect_model_source`], which has already
    /// rejected every out-of-scope shape before a summary can exist (non-`llama`
    /// `model_type`, non-`LlamaForCausalLM` architecture, `rope_scaling`,
    /// sliding-window attention, and missing/invalid fields), so this mapping is
    /// total: it renames the HF fields onto the dense config and threads the
    /// per-head dimension. It does NOT make the model runnable — SafeTensors
    /// generation stays gated until dtype decode, tensor orientation, tokenizer
    /// parity, and one-token execution land.
    pub fn to_llama_config(&self) -> crate::model::LlamaModelConfig {
        // Honor an explicit HF `head_dim` when present; otherwise LLaMA uses the
        // full per-head dimension `hidden_size / num_attention_heads` (the summary
        // guarantees exact divisibility).
        let default_head_dim = self.hidden_size / self.num_attention_heads;
        let explicit_head_dim = self.head_dim.filter(|&h| h > 0);
        let head_dim = explicit_head_dim.unwrap_or(default_head_dim);
        crate::model::LlamaModelConfig {
            context_length: self.max_position_embeddings,
            embedding_length: self.hidden_size,
            block_count: self.num_hidden_layers,
            feed_forward_length: self.intermediate_size,
            attention_head_count: self.num_attention_heads,
            attention_head_count_kv: self.num_key_value_heads,
            kv_quant: std::env::var("CAMELID_KV_QUANT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or_default(),
            // LLaMA rotates the full per-head dimension.
            rope_dimension_count: Some(head_dim),
            rope_freq_base: Some(self.rope_theta),
            // rope_scaling is rejected upstream, so the dense config carries none.
            rope_scaling_type: None,
            rope_scaling_factor: None,
            rope_scaling_original_context_length: None,
            rope_scaling_low_freq_factor: None,
            rope_scaling_high_freq_factor: None,
            rms_norm_epsilon: self.rms_norm_eps,
            vocab_size: Some(self.vocab_size),
            // Runtime-only GGUF hint; absent for a SafeTensors source.
            file_type: None,
            // Carry an explicit head_dim through the dense-dims path only when it
            // is NOT the default ratio (matches the GGUF `attention.key_length`
            // convention, where `None` means `embedding / head_count`).
            attention_key_length: explicit_head_dim.filter(|&h| h != default_head_dim),
            // Deferred, and coupled to the (not-yet-implemented) SafeTensors weight
            // loader: HF stores Q/K un-permuted, so the future decode slice must
            // either permute Q/K at load (matching llama.cpp's GGUF conversion,
            // which is why the dense-path default is `false`) or flip this to NEOX
            // split-half. Left `false` to match the canonical permute-at-load path;
            // it MUST be revisited together with tensor orientation before
            // generation is enabled for a SafeTensors source.
            rope_neox_pairing: false,
            logit_scale: None,
            moe: None,
            gemma4: None,
            qwen35: None,
            mla: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SafeTensorsTensorDescriptor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub shard_file: String,
    #[serde(skip_serializing)]
    pub shard: PathBuf,
    pub data_offsets: [u64; 2],
}

/// Dense LLaMA tensor roles resolved from a Hugging Face SafeTensors manifest —
/// the HF-name counterpart to the GGUF `LlamaTensorBinding`. It holds the source
/// descriptors (name/dtype/shape/shard) for each role and deliberately does NOT
/// open the shards, decode, transpose, or reinterpret any tensor. Orientation and
/// dtype decode are proven in a later slice, so resolving roles here keeps
/// SafeTensors generation gated while making the name→role mapping testable.
#[derive(Clone, Debug, PartialEq)]
pub struct HfLlamaTensorRoles {
    pub token_embedding: SafeTensorsTensorDescriptor,
    pub output_norm: SafeTensorsTensorDescriptor,
    pub output: SafeTensorsTensorDescriptor,
    /// True when `lm_head.weight` is absent and the output projection is tied to
    /// `model.embed_tokens.weight` (config `tie_word_embeddings`).
    pub output_is_tied_embedding: bool,
    pub layers: Vec<HfLlamaLayerRoles>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfLlamaLayerRoles {
    pub attention_norm: SafeTensorsTensorDescriptor,
    pub attention_q: SafeTensorsTensorDescriptor,
    pub attention_k: SafeTensorsTensorDescriptor,
    pub attention_v: SafeTensorsTensorDescriptor,
    pub attention_output: SafeTensorsTensorDescriptor,
    pub ffn_norm: SafeTensorsTensorDescriptor,
    pub ffn_gate: SafeTensorsTensorDescriptor,
    pub ffn_up: SafeTensorsTensorDescriptor,
    pub ffn_down: SafeTensorsTensorDescriptor,
}

/// Validate that a Hugging Face weight descriptor is a 2-D tensor whose two
/// dimensions are the expected `{out, in}` multiset. Orientation-tolerant on
/// purpose: HF stores linear weights as `[out, in]`, but this slice does not
/// commit to which axis is which (that is proven in the later decode slice), so
/// it only rejects a wrong-rank or wrong-size tensor.
fn require_hf_matrix(
    descriptor: SafeTensorsTensorDescriptor,
    expected_a: usize,
    expected_b: usize,
) -> Result<SafeTensorsTensorDescriptor> {
    let matches = descriptor.shape.len() == 2 && {
        let mut got = [descriptor.shape[0], descriptor.shape[1]];
        got.sort_unstable();
        let mut want = [expected_a as u64, expected_b as u64];
        want.sort_unstable();
        got == want
    };
    if !matches {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Hugging Face tensor {} has shape {:?}, expected a 2-D {expected_a}x{expected_b} weight",
            descriptor.name, descriptor.shape
        )));
    }
    Ok(descriptor)
}

/// Validate that a Hugging Face descriptor is a 1-D vector of `expected` length
/// (an RMSNorm weight).
fn require_hf_vector(
    descriptor: SafeTensorsTensorDescriptor,
    expected: usize,
) -> Result<SafeTensorsTensorDescriptor> {
    if descriptor.shape != [expected as u64] {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Hugging Face tensor {} has shape {:?}, expected a 1-D vector of length {expected}",
            descriptor.name, descriptor.shape
        )));
    }
    Ok(descriptor)
}

/// Resolve the dense LLaMA tensor roles from a Hugging Face SafeTensors tensor
/// set, using `config` for the geometry and `tie_word_embeddings` for the output
/// projection. Every role is looked up by its HF name AND its descriptor shape is
/// validated (orientation-tolerant) against the dense dims: a missing tensor
/// fails closed with a typed [`BackendError::TensorNotFound`] naming it, and a
/// mis-shaped tensor with [`BackendError::RuntimeShapeMismatch`]. Descriptor-level
/// only: it does not open shards, decode dtypes, or transpose any tensor, so it
/// never makes the model runnable.
pub fn resolve_hf_llama_tensor_roles(
    descriptors: &[SafeTensorsTensorDescriptor],
    config: &crate::model::LlamaModelConfig,
    tie_word_embeddings: bool,
) -> Result<HfLlamaTensorRoles> {
    let dims = crate::model::DenseLlamaDims::from_config(config)?;
    let by_name: BTreeMap<&str, &SafeTensorsTensorDescriptor> =
        descriptors.iter().map(|d| (d.name.as_str(), d)).collect();
    let required = |name: &str| -> Result<SafeTensorsTensorDescriptor> {
        by_name
            .get(name)
            .map(|descriptor| (*descriptor).clone())
            .ok_or_else(|| {
                BackendError::TensorNotFound(format!(
                    "Hugging Face SafeTensors model is missing required tensor {name}"
                ))
            })
    };

    let token_embedding = require_hf_matrix(
        required("model.embed_tokens.weight")?,
        dims.vocab_size,
        dims.embedding_length,
    )?;
    let output_norm = require_hf_vector(required("model.norm.weight")?, dims.embedding_length)?;
    let (output, output_is_tied_embedding) = match by_name.get("lm_head.weight") {
        Some(descriptor) => (
            require_hf_matrix(
                (*descriptor).clone(),
                dims.vocab_size,
                dims.embedding_length,
            )?,
            false,
        ),
        None if tie_word_embeddings => (token_embedding.clone(), true),
        None => {
            return Err(BackendError::TensorNotFound(
                "Hugging Face SafeTensors model has no lm_head.weight and config.json \
                 tie_word_embeddings is false, so the output projection is unresolved"
                    .to_string(),
            ))
        }
    };

    let mut layers = Vec::with_capacity(config.block_count as usize);
    for layer in 0..config.block_count {
        layers.push(HfLlamaLayerRoles {
            attention_norm: require_hf_vector(
                required(&format!("model.layers.{layer}.input_layernorm.weight"))?,
                dims.embedding_length,
            )?,
            attention_q: require_hf_matrix(
                required(&format!("model.layers.{layer}.self_attn.q_proj.weight"))?,
                dims.q_width,
                dims.embedding_length,
            )?,
            attention_k: require_hf_matrix(
                required(&format!("model.layers.{layer}.self_attn.k_proj.weight"))?,
                dims.kv_width,
                dims.embedding_length,
            )?,
            attention_v: require_hf_matrix(
                required(&format!("model.layers.{layer}.self_attn.v_proj.weight"))?,
                dims.kv_width,
                dims.embedding_length,
            )?,
            attention_output: require_hf_matrix(
                required(&format!("model.layers.{layer}.self_attn.o_proj.weight"))?,
                dims.embedding_length,
                dims.q_width,
            )?,
            ffn_norm: require_hf_vector(
                required(&format!(
                    "model.layers.{layer}.post_attention_layernorm.weight"
                ))?,
                dims.embedding_length,
            )?,
            ffn_gate: require_hf_matrix(
                required(&format!("model.layers.{layer}.mlp.gate_proj.weight"))?,
                dims.feed_forward_length,
                dims.embedding_length,
            )?,
            ffn_up: require_hf_matrix(
                required(&format!("model.layers.{layer}.mlp.up_proj.weight"))?,
                dims.feed_forward_length,
                dims.embedding_length,
            )?,
            ffn_down: require_hf_matrix(
                required(&format!("model.layers.{layer}.mlp.down_proj.weight"))?,
                dims.embedding_length,
                dims.feed_forward_length,
            )?,
        });
    }

    Ok(HfLlamaTensorRoles {
        token_embedding,
        output_norm,
        output,
        output_is_tied_embedding,
        layers,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelSourceReadiness {
    pub metadata_ready: bool,
    pub tokenizer_ready: bool,
    pub weights_ready: bool,
    pub generation_ready: bool,
    pub blockers: Vec<ModelSourceBlocker>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ModelSourceBlocker {
    pub code: &'static str,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSourceInspection {
    pub manifest: ModelSourceManifest,
    pub readiness: ModelSourceReadiness,
}

#[derive(Debug, Deserialize)]
struct HfConfigProbe {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    hidden_size: Option<u32>,
    num_hidden_layers: Option<u32>,
    intermediate_size: Option<u32>,
    num_attention_heads: Option<u32>,
    num_key_value_heads: Option<u32>,
    head_dim: Option<u32>,
    max_position_embeddings: Option<u32>,
    rms_norm_eps: Option<f32>,
    rope_theta: Option<f32>,
    vocab_size: Option<u32>,
    tie_word_embeddings: Option<bool>,
    rope_scaling: Option<Value>,
    sliding_window: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct SafeTensorsHeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Inspect a local model source without constructing runtime weights.
///
/// Existing GGUF callers should continue to use the current loader path. This
/// helper gives the SafeTensors lane a descriptor/readiness seam and deliberately
/// leaves Hugging Face SafeTensors generation disabled.
pub fn inspect_model_source(path: impl AsRef<Path>) -> Result<ModelSourceInspection> {
    let path = path.as_ref();
    let metadata = fs::metadata(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    if metadata.is_file() && has_extension(path, "gguf") {
        return Ok(inspect_gguf_file(path));
    }

    if metadata.is_dir() {
        return inspect_hugging_face_safetensors_dir(path);
    }

    Err(BackendError::InvalidModelMetadata(format!(
        "unsupported model source path {}; expected a .gguf file or local Hugging Face directory",
        public_path_label(path)
    )))
}

fn inspect_gguf_file(path: &Path) -> ModelSourceInspection {
    ModelSourceInspection {
        manifest: ModelSourceManifest {
            id: source_id(path),
            kind: ModelSourceKind::Gguf,
            hf_config: None,
            root: path.to_path_buf(),
            weight_files: vec![path.to_path_buf()],
            tensor_descriptors: Vec::new(),
            config_path: None,
            tokenizer_path: None,
            tokenizer_config_path: None,
            special_tokens_map_path: None,
            generation_config_path: None,
            shard_index_path: None,
        },
        readiness: ModelSourceReadiness {
            metadata_ready: true,
            tokenizer_ready: false,
            weights_ready: true,
            generation_ready: false,
            blockers: vec![blocker(
                "gguf_runtime_loader_unchanged",
                "GGUF source detection is present, but generation readiness is still owned by the existing GGUF runtime loader",
            )],
        },
    }
}

fn inspect_hugging_face_safetensors_dir(path: &Path) -> Result<ModelSourceInspection> {
    let config_path = existing_child(path, "config.json");
    let tokenizer_path = existing_child(path, "tokenizer.json");
    let tokenizer_config_path = existing_child(path, "tokenizer_config.json");
    let special_tokens_map_path = existing_child(path, "special_tokens_map.json");
    let generation_config_path = existing_child(path, "generation_config.json");
    let shard_index_path = existing_child(path, "model.safetensors.index.json");
    let weight_files = safetensors_files(path)?;

    let mut blockers = Vec::new();
    let (metadata_ready, hf_config) = config_path
        .as_ref()
        .map(|config_path| hf_config_summary(config_path, &mut blockers))
        .unwrap_or_else(|| {
            blockers.push(blocker(
                "missing_config_json",
                "Hugging Face SafeTensors directories must include config.json",
            ));
            (false, None)
        });

    let tokenizer_ready = tokenizer_path.is_some();
    if !tokenizer_ready {
        blockers.push(blocker(
            "missing_tokenizer_json",
            "Hugging Face SafeTensors directories must include tokenizer.json before tokenizer readiness can be reported",
        ));
    }

    let (weights_ready, tensor_descriptors) =
        hf_weights_ready(&weight_files, shard_index_path.as_deref(), &mut blockers)?;
    blockers.push(blocker(
        "generation_disabled",
        "SafeTensors generation remains disabled until tokenizer parity, tensor orientation, dtype decode, and one-token dense execution fixtures pass",
    ));

    Ok(ModelSourceInspection {
        manifest: ModelSourceManifest {
            id: source_id(path),
            kind: ModelSourceKind::HuggingFaceSafeTensors,
            hf_config,
            root: path.to_path_buf(),
            weight_files,
            tensor_descriptors,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            special_tokens_map_path,
            generation_config_path,
            shard_index_path,
        },
        readiness: ModelSourceReadiness {
            metadata_ready,
            tokenizer_ready,
            weights_ready,
            generation_ready: false,
            blockers,
        },
    })
}

fn hf_config_summary(
    config_path: &Path,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> (bool, Option<HfLlamaConfigSummary>) {
    let Ok(bytes) = fs::read(config_path) else {
        blockers.push(blocker(
            "config_json_unreadable",
            format!(
                "could not read required Hugging Face config file {}",
                public_path_label(config_path)
            ),
        ));
        return (false, None);
    };
    let Ok(config) = serde_json::from_slice::<HfConfigProbe>(&bytes) else {
        blockers.push(blocker(
            "invalid_config_json",
            format!(
                "required Hugging Face config file {} is not valid JSON",
                public_path_label(config_path)
            ),
        ));
        return (false, None);
    };

    if config.model_type.as_deref() != Some("llama") {
        blockers.push(blocker(
            "unsupported_model_type",
            format!(
                "only dense LLaMA-family SafeTensors config metadata is in scope for the first readiness slice; got {:?}",
                config.model_type
            ),
        ));
        return (false, None);
    }

    let Some(architectures) = config.architectures.clone() else {
        blockers.push(blocker(
            "missing_config_field",
            "Hugging Face config.json is missing required field architectures",
        ));
        return (false, None);
    };
    if !architectures
        .iter()
        .any(|architecture| architecture == "LlamaForCausalLM")
    {
        blockers.push(blocker(
            "unsupported_architecture",
            format!(
                "only LlamaForCausalLM SafeTensors configs are in scope for this readiness slice; got {:?}",
                architectures
            ),
        ));
        return (false, None);
    }

    if config.sliding_window.is_some() {
        blockers.push(blocker(
            "unsupported_sliding_window_attention",
            "sliding-window attention needs an explicit Camelid runtime mapping before SafeTensors metadata can be ready",
        ));
        return (false, None);
    }

    if config.rope_scaling.is_some() {
        blockers.push(blocker(
            "unsupported_rope_scaling",
            "rope_scaling needs an explicit Camelid HF config mapping before SafeTensors metadata can be ready",
        ));
        return (false, None);
    }

    let Some(hidden_size) = required_hf_u32(&config, "hidden_size", config.hidden_size, blockers)
    else {
        return (false, None);
    };
    let Some(num_hidden_layers) = required_hf_u32(
        &config,
        "num_hidden_layers",
        config.num_hidden_layers,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(intermediate_size) = required_hf_u32(
        &config,
        "intermediate_size",
        config.intermediate_size,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(num_attention_heads) = required_hf_u32(
        &config,
        "num_attention_heads",
        config.num_attention_heads,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(num_key_value_heads) = required_hf_u32(
        &config,
        "num_key_value_heads",
        config.num_key_value_heads,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(max_position_embeddings) = required_hf_u32(
        &config,
        "max_position_embeddings",
        config.max_position_embeddings,
        blockers,
    ) else {
        return (false, None);
    };
    let Some(vocab_size) = required_hf_u32(&config, "vocab_size", config.vocab_size, blockers)
    else {
        return (false, None);
    };
    let Some(rms_norm_eps) =
        required_hf_f32(&config, "rms_norm_eps", config.rms_norm_eps, blockers)
    else {
        return (false, None);
    };
    let Some(rope_theta) = required_hf_f32(&config, "rope_theta", config.rope_theta, blockers)
    else {
        return (false, None);
    };
    let Some(tie_word_embeddings) = config.tie_word_embeddings else {
        blockers.push(blocker(
            "missing_config_field",
            "Hugging Face config.json is missing required field tie_word_embeddings",
        ));
        return (false, None);
    };

    if hidden_size % num_attention_heads != 0 {
        blockers.push(blocker(
            "invalid_attention_geometry",
            format!(
                "hidden_size {hidden_size} must be divisible by num_attention_heads {num_attention_heads} before SafeTensors metadata can be ready"
            ),
        ));
        return (false, None);
    }
    if num_key_value_heads > num_attention_heads {
        blockers.push(blocker(
            "invalid_attention_geometry",
            format!(
                "num_key_value_heads {num_key_value_heads} must be <= num_attention_heads {num_attention_heads} before SafeTensors metadata can be ready"
            ),
        ));
        return (false, None);
    }

    (
        true,
        Some(HfLlamaConfigSummary {
            model_type: "llama".to_string(),
            architectures,
            hidden_size,
            num_hidden_layers,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim: config.head_dim.filter(|&h| h > 0),
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            vocab_size,
            tie_word_embeddings,
        }),
    )
}

fn required_hf_u32(
    _config: &HfConfigProbe,
    field: &'static str,
    value: Option<u32>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Option<u32> {
    match value {
        Some(value) if value > 0 => Some(value),
        Some(_) => {
            blockers.push(blocker(
                "invalid_config_field",
                format!("Hugging Face config.json field {field} must be greater than zero"),
            ));
            None
        }
        None => {
            blockers.push(blocker(
                "missing_config_field",
                format!("Hugging Face config.json is missing required field {field}"),
            ));
            None
        }
    }
}

fn required_hf_f32(
    _config: &HfConfigProbe,
    field: &'static str,
    value: Option<f32>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Option<f32> {
    match value {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        Some(_) => {
            blockers.push(blocker(
                "invalid_config_field",
                format!(
                    "Hugging Face config.json field {field} must be finite and greater than zero"
                ),
            ));
            None
        }
        None => {
            blockers.push(blocker(
                "missing_config_field",
                format!("Hugging Face config.json is missing required field {field}"),
            ));
            None
        }
    }
}

fn hf_weights_ready(
    weight_files: &[PathBuf],
    shard_index_path: Option<&Path>,
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Result<(bool, Vec<SafeTensorsTensorDescriptor>)> {
    if weight_files.is_empty() {
        blockers.push(blocker(
            "missing_safetensors_weights",
            "Hugging Face SafeTensors directories must include at least one .safetensors weight file",
        ));
        return Ok((false, Vec::new()));
    }

    let mut ready = true;
    let tensor_descriptors = inspect_safetensors_headers(weight_files, blockers)?;
    if tensor_descriptors.is_empty() {
        ready = false;
    }
    if !safetensors_dtypes_ready(&tensor_descriptors, blockers) {
        ready = false;
    }

    if weight_files.len() > 1 && shard_index_path.is_none() {
        blockers.push(blocker(
            "missing_shard_index",
            "sharded SafeTensors directories must include model.safetensors.index.json before weights readiness can be reported",
        ));
        ready = false;
    }

    if let Some(shard_index_path) = shard_index_path {
        let bytes = fs::read(shard_index_path).map_err(|source| BackendError::Io {
            path: shard_index_path.to_path_buf(),
            source,
        })?;
        let Ok(index) = serde_json::from_slice::<Value>(&bytes) else {
            blockers.push(blocker(
                "invalid_shard_index_json",
                format!(
                    "SafeTensors shard index file {} is not valid JSON",
                    public_path_label(shard_index_path)
                ),
            ));
            return Ok((false, tensor_descriptors));
        };
        if !hf_shard_index_weight_map_ready(&index, weight_files, &tensor_descriptors, blockers) {
            ready = false;
        }
    }

    Ok((ready, tensor_descriptors))
}

fn inspect_safetensors_headers(
    weight_files: &[PathBuf],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> Result<Vec<SafeTensorsTensorDescriptor>> {
    let mut tensor_descriptors = Vec::new();
    let mut seen = BTreeMap::<String, String>::new();
    for weight_file in weight_files {
        match parse_safetensors_header(weight_file) {
            Ok(file_descriptors) => {
                if file_descriptors.is_empty() {
                    blockers.push(blocker(
                        "empty_safetensors_header",
                        format!(
                            "SafeTensors shard {} does not list any tensor descriptors",
                            public_path_label(weight_file)
                        ),
                    ));
                }
                for descriptor in file_descriptors {
                    if let Some(first_shard) = seen.insert(
                        descriptor.name.clone(),
                        public_path_label(&descriptor.shard),
                    ) {
                        blockers.push(blocker(
                            "duplicate_safetensors_tensor",
                            format!(
                                "SafeTensors tensor {} appears in more than one shard: {}, {}",
                                descriptor.name,
                                first_shard,
                                public_path_label(&descriptor.shard)
                            ),
                        ));
                    }
                    tensor_descriptors.push(descriptor);
                }
            }
            Err(message) => blockers.push(blocker(
                "invalid_safetensors_header",
                format!(
                    "SafeTensors shard {} has an invalid header: {}",
                    public_path_label(weight_file),
                    message
                ),
            )),
        }
    }
    tensor_descriptors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensor_descriptors)
}

fn parse_safetensors_header(
    path: &Path,
) -> std::result::Result<Vec<SafeTensorsTensorDescriptor>, String> {
    let mut file = fs::File::open(path).map_err(|err| format!("could not open shard: {err}"))?;
    let file_len = file
        .metadata()
        .map_err(|err| format!("could not stat shard: {err}"))?
        .len();
    if file_len < 8 {
        return Err("file is shorter than the 8-byte SafeTensors header length".to_string());
    }

    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|err| format!("could not read header length: {err}"))?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len = usize::try_from(header_len)
        .map_err(|_| "header length does not fit this platform".to_string())?;
    let header_end = 8u64
        .checked_add(header_len as u64)
        .ok_or_else(|| "header length overflows file offset arithmetic".to_string())?;
    if header_end > file_len {
        return Err("header length extends past end of file".to_string());
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| format!("could not read header JSON: {err}"))?;

    let header = serde_json::from_slice::<Value>(&header_bytes)
        .map_err(|err| format!("header JSON is invalid: {err}"))?;
    let object = header
        .as_object()
        .ok_or_else(|| "header JSON must be an object".to_string())?;
    let payload_len = file_len - header_end;
    let mut descriptors = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let tensor = serde_json::from_value::<SafeTensorsHeaderTensor>(value.clone())
            .map_err(|err| format!("tensor descriptor {name} is invalid: {err}"))?;
        if tensor.data_offsets[0] > tensor.data_offsets[1] {
            return Err(format!(
                "tensor descriptor {name} has descending data_offsets"
            ));
        }
        if tensor.data_offsets[1] > payload_len {
            return Err(format!(
                "tensor descriptor {name} data_offsets extend past shard payload"
            ));
        }
        if let Some(dtype_size) = safetensors_dtype_size(&tensor.dtype) {
            let element_count = tensor.shape.iter().try_fold(1u64, |acc, dim| {
                acc.checked_mul(*dim).ok_or_else(|| {
                    format!("tensor descriptor {name} shape element count overflows")
                })
            })?;
            let expected_len = element_count
                .checked_mul(dtype_size)
                .ok_or_else(|| format!("tensor descriptor {name} byte length overflows"))?;
            let actual_len = tensor.data_offsets[1] - tensor.data_offsets[0];
            if actual_len != expected_len {
                return Err(format!(
                    "tensor descriptor {name} data_offsets length {actual_len} does not match shape/dtype byte length {expected_len}"
                ));
            }
        }
        descriptors.push(SafeTensorsTensorDescriptor {
            name: name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape,
            shard_file: public_path_label(path),
            shard: path.to_path_buf(),
            data_offsets: tensor.data_offsets,
        });
    }
    Ok(descriptors)
}

/// Decode one SafeTensors descriptor's bytes into a dense f32
/// [`crate::tensor::CpuTensor`].
///
/// Reuses the GGUF F32/F16/BF16 decoders so a SafeTensors weight and its GGUF
/// twin decode identically. Does not transpose or reinterpret axes, and does not
/// make the source generation-ready (orientation and tokenizer parity are later
/// slices).
pub fn decode_safetensors_tensor(
    descriptor: &SafeTensorsTensorDescriptor,
) -> Result<crate::tensor::CpuTensor> {
    let dtype = descriptor.dtype.as_str();
    // F32/F16/BF16 only; other dtypes are a typed rejection until a source needs them.
    if !matches!(dtype, "F32" | "F16" | "BF16") {
        return Err(BackendError::UnsupportedTensorType(format!(
            "SafeTensors tensor {} has dtype {}; the decode slice supports F32, F16, and BF16",
            descriptor.name, descriptor.dtype
        )));
    }

    let span = descriptor.data_offsets[1]
        .checked_sub(descriptor.data_offsets[0])
        .ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "SafeTensors tensor {} has descending data_offsets",
                descriptor.name
            ))
        })?;
    let span = usize::try_from(span).map_err(|_| {
        BackendError::InvalidTensorData(format!(
            "SafeTensors tensor {} byte length does not fit this platform",
            descriptor.name
        ))
    })?;

    let dims = descriptor
        .shape
        .iter()
        .map(|&dim| {
            usize::try_from(dim).map_err(|_| {
                BackendError::InvalidTensorData(format!(
                    "SafeTensors tensor {} shape dimension {dim} does not fit this platform",
                    descriptor.name
                ))
            })
        })
        .collect::<Result<Vec<usize>>>()?;

    let mut expected_elements: usize = 1;
    for &dim in &dims {
        expected_elements = expected_elements.checked_mul(dim).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "SafeTensors tensor {} element count overflows usize",
                descriptor.name
            ))
        })?;
    }

    let bytes = read_safetensors_tensor_bytes(descriptor, span)?;

    let data = match dtype {
        "F32" => crate::tensor::decode_f32_tensor(&descriptor.name, &bytes, expected_elements)?,
        "F16" => crate::tensor::decode_f16_tensor(&descriptor.name, &bytes, expected_elements)?,
        "BF16" => crate::tensor::decode_bf16_tensor(&descriptor.name, &bytes, expected_elements)?,
        _ => unreachable!("dtype is guarded to F32/F16/BF16 above"),
    };

    crate::tensor::CpuTensor::from_f32(descriptor.name.clone(), dims, data)
}

/// Read one tensor's raw payload bytes from its shard. `data_offsets` are
/// relative to the payload (after the 8-byte length + JSON header), so the
/// header length is re-read here.
fn read_safetensors_tensor_bytes(
    descriptor: &SafeTensorsTensorDescriptor,
    span: usize,
) -> Result<Vec<u8>> {
    let mut file = fs::File::open(&descriptor.shard).map_err(|source| BackendError::Io {
        path: descriptor.shard.clone(),
        source,
    })?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|source| BackendError::Io {
            path: descriptor.shard.clone(),
            source,
        })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let payload_start = 8u64.checked_add(header_len).ok_or_else(|| {
        BackendError::InvalidTensorData(format!(
            "SafeTensors shard {} header length overflows file offset arithmetic",
            public_path_label(&descriptor.shard)
        ))
    })?;
    let absolute = payload_start
        .checked_add(descriptor.data_offsets[0])
        .ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "SafeTensors tensor {} absolute offset overflows file offset arithmetic",
                descriptor.name
            ))
        })?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|source| BackendError::Io {
            path: descriptor.shard.clone(),
            source,
        })?;
    let mut bytes = vec![0u8; span];
    file.read_exact(&mut bytes)
        .map_err(|source| BackendError::Io {
            path: descriptor.shard.clone(),
            source,
        })?;
    Ok(bytes)
}

fn safetensors_dtype_size(dtype: &str) -> Option<u64> {
    match dtype {
        "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        _ => None,
    }
}

fn safetensors_dtypes_ready(
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> bool {
    let unsupported = tensor_descriptors
        .iter()
        .filter(|descriptor| !matches!(descriptor.dtype.as_str(), "F32" | "F16" | "BF16"))
        .map(|descriptor| format!("{}:{}", descriptor.name, descriptor.dtype))
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        return true;
    }
    blockers.push(blocker(
        "unsupported_safetensors_dtype",
        format!(
            "only F32, F16, and BF16 SafeTensors descriptors are in scope for this readiness slice; unsupported tensors: {}",
            unsupported.join(", ")
        ),
    ));
    false
}

fn hf_shard_index_weight_map_ready(
    index: &Value,
    weight_files: &[PathBuf],
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
    blockers: &mut Vec<ModelSourceBlocker>,
) -> bool {
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        blockers.push(blocker(
            "missing_shard_index_weight_map",
            "model.safetensors.index.json must include a weight_map object before sharded weights readiness can be reported",
        ));
        return false;
    };
    if weight_map.is_empty() {
        blockers.push(blocker(
            "empty_shard_index_weight_map",
            "model.safetensors.index.json weight_map must list at least one tensor shard before sharded weights readiness can be reported",
        ));
        return false;
    }

    let available = weight_files
        .iter()
        .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
        .collect::<BTreeSet<_>>();
    let mut missing = BTreeSet::new();
    let mut invalid = BTreeSet::new();
    let mut invalid_filenames = BTreeSet::new();
    let mut missing_tensors = BTreeSet::new();
    let tensors_by_shard = safetensors_tensors_by_shard(tensor_descriptors);
    for (tensor_name, shard_name) in weight_map {
        let Some(shard_name) = shard_name.as_str() else {
            invalid.insert(tensor_name.as_str());
            continue;
        };
        if !is_plain_safetensors_shard_filename(shard_name) {
            invalid_filenames.insert(tensor_name.as_str());
            continue;
        }
        if !available.contains(shard_name) {
            missing.insert(shard_name);
            continue;
        }
        if !tensors_by_shard
            .get(shard_name)
            .is_some_and(|tensors| tensors.contains(tensor_name.as_str()))
        {
            missing_tensors.insert(format!("{tensor_name} in {shard_name}"));
        }
    }

    if !invalid.is_empty() {
        blockers.push(blocker(
            "invalid_shard_index_weight_map",
            format!(
                "model.safetensors.index.json weight_map entries must map tensor names to shard filenames; invalid entries: {}",
                invalid.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !invalid_filenames.is_empty() {
        blockers.push(blocker(
            "invalid_shard_index_shard_filename",
            format!(
                "model.safetensors.index.json weight_map shard values must be local shard filenames, not paths; invalid tensor entries: {}",
                invalid_filenames.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !missing.is_empty() {
        blockers.push(blocker(
            "missing_sharded_weight_file",
            format!(
                "model.safetensors.index.json references shard files that are not present locally: {}",
                missing.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }
    if !missing_tensors.is_empty() {
        blockers.push(blocker(
            "shard_index_tensor_not_found",
            format!(
                "model.safetensors.index.json references tensors not found in the listed shard headers: {}",
                missing_tensors.into_iter().collect::<Vec<_>>().join(", ")
            ),
        ));
        return false;
    }

    true
}

fn safetensors_tensors_by_shard(
    tensor_descriptors: &[SafeTensorsTensorDescriptor],
) -> BTreeMap<&str, BTreeSet<&str>> {
    let mut tensors_by_shard = BTreeMap::<&str, BTreeSet<&str>>::new();
    for descriptor in tensor_descriptors {
        if let Some(shard_name) = descriptor.shard.file_name().and_then(|name| name.to_str()) {
            tensors_by_shard
                .entry(shard_name)
                .or_default()
                .insert(descriptor.name.as_str());
        }
    }
    tensors_by_shard
}

fn is_plain_safetensors_shard_filename(value: &str) -> bool {
    !value.is_empty()
        && !value.contains('/')
        && !value.contains('\\')
        && value != "."
        && value != ".."
        && has_extension(Path::new(value), "safetensors")
}

fn safetensors_files(path: &Path) -> Result<Vec<PathBuf>> {
    let entries = fs::read_dir(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| BackendError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let candidate = entry.path();
        if candidate.is_file() && has_extension(&candidate, "safetensors") {
            files.push(candidate);
        }
    }
    files.sort();
    Ok(files)
}

fn existing_child(root: &Path, name: &str) -> Option<PathBuf> {
    let candidate = root.join(name);
    candidate.is_file().then_some(candidate)
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
}

fn source_id(path: &Path) -> String {
    let name = if has_extension(path, "gguf") {
        path.file_stem().or_else(|| path.file_name())
    } else {
        path.file_name()
    };
    name.and_then(|value| value.to_str())
        .unwrap_or("model")
        .to_string()
}

fn public_path_label(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model source file")
        .to_string()
}

fn blocker(code: &'static str, message: impl Into<String>) -> ModelSourceBlocker {
    ModelSourceBlocker {
        code,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    #[test]
    fn complete_hf_safetensors_directory_reports_readiness_but_not_generation() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();
        fs::write(dir.path().join("special_tokens_map.json"), "{}").unwrap();
        fs::write(dir.path().join("generation_config.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "BF16", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"model-00001-of-00002.safetensors","lm_head.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert_eq!(
            inspection.manifest.kind,
            ModelSourceKind::HuggingFaceSafeTensors
        );
        assert_eq!(inspection.manifest.weight_files.len(), 2);
        assert!(inspection.manifest.config_path.is_some());
        let hf_config = inspection.manifest.hf_config.as_ref().unwrap();
        assert_eq!(hf_config.hidden_size, 16);
        assert_eq!(hf_config.num_hidden_layers, 2);
        assert_eq!(hf_config.intermediate_size, 64);
        assert_eq!(hf_config.num_attention_heads, 4);
        assert_eq!(hf_config.num_key_value_heads, 2);
        assert_eq!(hf_config.max_position_embeddings, 128);
        assert_eq!(hf_config.vocab_size, 32000);
        assert!(!hf_config.tie_word_embeddings);
        assert!(inspection.manifest.tokenizer_path.is_some());
        assert!(inspection.manifest.shard_index_path.is_some());
        assert_eq!(inspection.manifest.tensor_descriptors.len(), 2);
        assert_eq!(
            inspection.manifest.tensor_descriptors[0].name,
            "lm_head.weight"
        );
        assert_eq!(inspection.manifest.tensor_descriptors[0].dtype, "BF16");
        assert_eq!(
            inspection.manifest.tensor_descriptors[0].shard_file,
            "model-00002-of-00002.safetensors"
        );
        assert_eq!(
            inspection.manifest.tensor_descriptors[1].name,
            "model.embed_tokens.weight"
        );
        assert_eq!(inspection.manifest.tensor_descriptors[1].shape, vec![1, 1]);
        assert!(inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert!(!inspection.readiness.generation_ready);
        assert_blocker_codes(&inspection, &["generation_disabled"]);
    }

    #[test]
    fn missing_tokenizer_fixture_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F16", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(inspection.readiness.metadata_ready);
        assert!(!inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_tokenizer_json", "generation_disabled"],
        );
    }

    #[test]
    fn sharded_weights_without_index_have_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(&inspection, &["missing_shard_index", "generation_disabled"]);
    }

    #[test]
    fn invalid_shard_index_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(dir.path().join("model.safetensors.index.json"), "not json").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_json", "generation_disabled"],
        );
        assert_public_blocker_message_without_local_path(
            &inspection.readiness.blockers[0].message,
            dir.path(),
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.safetensors.index.json"));
    }

    #[test]
    fn shard_index_without_weight_map_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_shard_index_weight_map", "generation_disabled"],
        );
    }

    #[test]
    fn shard_index_referencing_missing_weight_file_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"model-00003-of-00003.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["missing_sharded_weight_file", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model-00003-of-00003.safetensors"));
    }

    #[test]
    fn invalid_config_json_has_sanitized_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), "not json").unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(&inspection, &["invalid_config_json", "generation_disabled"]);
        assert_public_blocker_message_without_local_path(
            &inspection.readiness.blockers[0].message,
            dir.path(),
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("config.json"));
    }

    #[test]
    fn unsupported_config_fixture_has_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"mistral","architectures":["MistralForCausalLM"]}"#,
        )
        .unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.readiness.tokenizer_ready);
        assert!(inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["unsupported_model_type", "generation_disabled"],
        );
    }

    #[test]
    fn missing_required_hf_config_field_blocks_metadata_readiness() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"],"hidden_size":16}"#,
        )
        .unwrap();
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert!(inspection.manifest.hf_config.is_none());
        assert_blocker_codes(
            &inspection,
            &["missing_config_field", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("num_hidden_layers"));
    }

    #[test]
    fn invalid_hf_attention_geometry_blocks_metadata_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config_with_overrides(
            dir.path(),
            &[
                ("hidden_size", serde_json::json!(18)),
                ("num_attention_heads", serde_json::json!(4)),
            ],
        );
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_attention_geometry", "generation_disabled"],
        );
    }

    #[test]
    fn rope_scaling_config_blocks_metadata_readiness_until_mapped() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config_with_overrides(
            dir.path(),
            &[(
                "rope_scaling",
                serde_json::json!({"type":"linear","factor":2.0}),
            )],
        );
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.metadata_ready);
        assert_blocker_codes(
            &inspection,
            &["unsupported_rope_scaling", "generation_disabled"],
        );
    }

    #[test]
    fn shard_index_path_values_have_sanitized_precise_blocker() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"model.embed_tokens.weight":"../private/model-00001-of-00002.safetensors","lm_head.weight":"C:\\private\\model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_shard_filename", "generation_disabled"],
        );
        let message = &inspection.readiness.blockers[0].message;
        assert!(message.contains("model.embed_tokens.weight"));
        assert!(message.contains("lm_head.weight"));
        assert!(!message.contains("../private"));
        assert!(!message.contains("C:"));
        assert!(!message.contains("model-00001-of-00002.safetensors"));
        assert!(!message.contains("model-00002-of-00002.safetensors"));
    }

    #[test]
    fn shard_index_invalid_entries_are_reported_in_stable_tensor_order() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        write_safetensors_file(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("lm_head.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"z.weight":42,"a.weight":false}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_shard_index_weight_map", "generation_disabled"],
        );
        let message = &inspection.readiness.blockers[0].message;
        assert!(message.ends_with("invalid entries: a.weight, z.weight"));
    }

    #[test]
    fn hf_directory_source_id_preserves_dotted_model_name() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("Meta-Llama-3.1-8B-Instruct");
        fs::create_dir(&model_dir).unwrap();
        write_llama_config(&model_dir);
        fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            &model_dir,
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(&model_dir).unwrap();

        assert_eq!(inspection.manifest.id, "Meta-Llama-3.1-8B-Instruct");
    }

    #[test]
    fn invalid_safetensors_header_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        fs::write(dir.path().join("model.safetensors"), b"too short").unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert!(inspection.manifest.tensor_descriptors.is_empty());
        assert_blocker_codes(
            &inspection,
            &["invalid_safetensors_header", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.safetensors"));
    }

    #[test]
    fn unsupported_safetensors_dtype_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "I64", &[1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_eq!(inspection.manifest.tensor_descriptors.len(), 1);
        assert_blocker_codes(
            &inspection,
            &["unsupported_safetensors_dtype", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("model.embed_tokens.weight:I64"));
    }

    #[test]
    fn shard_index_tensor_missing_from_header_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model-00001-of-00001.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"lm_head.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["shard_index_tensor_not_found", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("lm_head.weight in model-00001-of-00001.safetensors"));
    }

    #[test]
    fn serialized_hf_inspection_does_not_expose_local_paths() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        write_safetensors_file(
            dir.path(),
            "model.safetensors",
            &[("model.embed_tokens.weight", "F32", &[1, 1])],
        );

        let inspection = inspect_model_source(dir.path()).unwrap();
        let json = serde_json::to_string(&inspection).unwrap();

        assert_public_blocker_message_without_local_path(&json, dir.path());
        assert!(!json.contains("root"));
        assert!(!json.contains("weight_files"));
        assert!(!json.contains("config_path"));
        assert!(!json.contains("tokenizer_path"));
        assert!(!json.contains("\"shard\""));
        assert!(json.contains("shard_file"));
        assert!(json.contains("model.safetensors"));
    }

    #[test]
    fn safetensors_shape_dtype_offset_mismatch_blocks_weight_readiness() {
        let dir = tempfile::tempdir().unwrap();
        write_llama_config(dir.path());
        fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();
        let header = serde_json::json!({
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 4]
            }
        });
        write_safetensors_bytes(dir.path(), "model.safetensors", &header, &[0, 0, 0, 0]);

        let inspection = inspect_model_source(dir.path()).unwrap();

        assert!(!inspection.readiness.weights_ready);
        assert_blocker_codes(
            &inspection,
            &["invalid_safetensors_header", "generation_disabled"],
        );
        assert!(inspection.readiness.blockers[0]
            .message
            .contains("does not match shape/dtype byte length 8"));
    }

    #[test]
    fn gguf_file_source_id_strips_only_gguf_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TinyLlama-1.1B-Chat-v1.0.Q8_0.gguf");
        fs::write(&path, b"").unwrap();

        let inspection = inspect_model_source(&path).unwrap();

        assert_eq!(inspection.manifest.id, "TinyLlama-1.1B-Chat-v1.0.Q8_0");
    }

    #[test]
    fn unsupported_source_file_error_uses_public_label_without_parent_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-model.bin");
        fs::write(&path, b"").unwrap();

        let err = inspect_model_source(&path).unwrap_err().to_string();

        assert!(err.contains("not-a-model.bin"));
        assert_public_blocker_message_without_local_path(&err, dir.path());
    }

    fn write_llama_config(root: &Path) {
        write_llama_config_with_overrides(root, &[]);
    }

    fn write_llama_config_with_overrides(root: &Path, overrides: &[(&str, Value)]) {
        let mut config = serde_json::Map::from_iter([
            ("model_type".to_string(), serde_json::json!("llama")),
            (
                "architectures".to_string(),
                serde_json::json!(["LlamaForCausalLM"]),
            ),
            ("hidden_size".to_string(), serde_json::json!(16)),
            ("num_hidden_layers".to_string(), serde_json::json!(2)),
            ("intermediate_size".to_string(), serde_json::json!(64)),
            ("num_attention_heads".to_string(), serde_json::json!(4)),
            ("num_key_value_heads".to_string(), serde_json::json!(2)),
            (
                "max_position_embeddings".to_string(),
                serde_json::json!(128),
            ),
            ("rms_norm_eps".to_string(), serde_json::json!(0.00001)),
            ("rope_theta".to_string(), serde_json::json!(10000.0)),
            ("vocab_size".to_string(), serde_json::json!(32000)),
            ("tie_word_embeddings".to_string(), serde_json::json!(false)),
        ]);
        for (key, value) in overrides {
            config.insert((*key).to_string(), value.clone());
        }
        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&Value::Object(config)).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors_file(root: &Path, name: &str, tensors: &[(&str, &str, &[u64])]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut payload = Vec::new();
        for (tensor_name, dtype, shape) in tensors {
            let byte_len = tensor_byte_len(dtype, shape);
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len],
                }),
            );
            payload.resize(payload.len() + usize::try_from(byte_len).unwrap(), 0);
            offset += byte_len;
        }
        write_safetensors_bytes(root, name, &Value::Object(header), &payload);
    }

    fn write_safetensors_bytes(root: &Path, name: &str, header: &Value, payload: &[u8]) {
        let header = serde_json::to_vec(header).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(payload);
        fs::write(root.join(name), file).unwrap();
    }

    fn tensor_byte_len(dtype: &str, shape: &[u64]) -> u64 {
        let element_size = match dtype {
            "F16" | "BF16" => 2,
            "F32" | "I32" => 4,
            "I64" => 8,
            other => panic!("test fixture does not know dtype {other}"),
        };
        shape.iter().product::<u64>() * element_size
    }

    fn assert_blocker_codes(inspection: &ModelSourceInspection, expected: &[&str]) {
        let actual = inspection
            .readiness
            .blockers
            .iter()
            .map(|blocker| blocker.code)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn assert_public_blocker_message_without_local_path(message: &str, root: &Path) {
        let root = root.display().to_string();
        assert!(
            !message.contains(&root),
            "blocker message leaked local path {root:?}: {message}"
        );
        assert!(
            !message.contains("/var/") && !message.contains("/private/"),
            "blocker message leaked a private temp path: {message}"
        );
    }

    fn tiny_hf_summary() -> HfLlamaConfigSummary {
        HfLlamaConfigSummary {
            model_type: "llama".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: 16,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: None,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            vocab_size: 32,
            tie_word_embeddings: false,
        }
    }

    fn desc(name: &str, shape: Vec<u64>) -> SafeTensorsTensorDescriptor {
        let elements: u64 = shape.iter().product();
        SafeTensorsTensorDescriptor {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape,
            shard_file: "model.safetensors".to_string(),
            shard: PathBuf::from("model.safetensors"),
            data_offsets: [0, elements * 4],
        }
    }

    /// A complete, correctly-shaped dense-LLaMA descriptor set for the tiny
    /// `tiny_hf_summary` geometry (hidden 16, 4 heads, 2 kv-heads, head_dim 4 =>
    /// q_width 16, kv_width 8; intermediate 64; vocab 32). `lm_head` optional.
    fn hf_descriptors(layers: u32, with_lm_head: bool) -> Vec<SafeTensorsTensorDescriptor> {
        let mut out = vec![
            desc("model.embed_tokens.weight", vec![32, 16]),
            desc("model.norm.weight", vec![16]),
        ];
        if with_lm_head {
            out.push(desc("lm_head.weight", vec![32, 16]));
        }
        for layer in 0..layers {
            out.extend([
                desc(
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                    vec![16],
                ),
                desc(
                    &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                    vec![16, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                    vec![8, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.self_attn.v_proj.weight"),
                    vec![8, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.self_attn.o_proj.weight"),
                    vec![16, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                    vec![16],
                ),
                desc(
                    &format!("model.layers.{layer}.mlp.gate_proj.weight"),
                    vec![64, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.mlp.up_proj.weight"),
                    vec![64, 16],
                ),
                desc(
                    &format!("model.layers.{layer}.mlp.down_proj.weight"),
                    vec![16, 64],
                ),
            ]);
        }
        out
    }

    #[test]
    fn to_llama_config_maps_dense_llama_fields() {
        let config = tiny_hf_summary().to_llama_config();
        assert_eq!(config.context_length, 128);
        assert_eq!(config.embedding_length, 16);
        assert_eq!(config.block_count, 2);
        assert_eq!(config.feed_forward_length, 64);
        assert_eq!(config.attention_head_count, 4);
        assert_eq!(config.attention_head_count_kv, 2);
        // head_dim = hidden_size / num_attention_heads = 16 / 4 = 4.
        assert_eq!(config.rope_dimension_count, Some(4));
        assert_eq!(config.rope_freq_base, Some(10_000.0));
        assert_eq!(config.rms_norm_epsilon, 1e-5);
        assert_eq!(config.vocab_size, Some(32));
        // Out-of-scope shapes stay unset; the model is not made runnable here.
        assert_eq!(config.rope_scaling_type, None);
        assert_eq!(config.attention_key_length, None);
        assert!(!config.rope_neox_pairing);
        assert!(config.file_type.is_none());
        assert!(config.moe.is_none());
        assert!(config.gemma4.is_none());
        assert!(config.qwen35.is_none());
        // The derived config must pass the same dense-dims gate the GGUF path uses.
        crate::model::DenseLlamaDims::from_config(&config).unwrap();
    }

    #[test]
    fn to_llama_config_honors_explicit_head_dim() {
        // An explicit head_dim that differs from hidden/heads (16/4 = 4) must be
        // threaded through, not silently replaced by the ratio.
        let mut summary = tiny_hf_summary();
        summary.head_dim = Some(8);
        let config = summary.to_llama_config();
        assert_eq!(config.rope_dimension_count, Some(8));
        assert_eq!(config.attention_key_length, Some(8));
        let dims = crate::model::DenseLlamaDims::from_config(&config).unwrap();
        assert_eq!(dims.head_dim, 8);
        assert_eq!(dims.q_width, 4 * 8);
        // A head_dim equal to the default ratio carries no explicit override.
        summary.head_dim = Some(4);
        assert_eq!(summary.to_llama_config().attention_key_length, None);
    }

    #[test]
    fn resolve_hf_llama_tensor_roles_resolves_complete_two_layer_model() {
        let config = tiny_hf_summary().to_llama_config();
        let roles =
            resolve_hf_llama_tensor_roles(&hf_descriptors(2, true), &config, false).unwrap();
        assert_eq!(roles.token_embedding.name, "model.embed_tokens.weight");
        assert_eq!(roles.output_norm.name, "model.norm.weight");
        assert_eq!(roles.output.name, "lm_head.weight");
        assert!(!roles.output_is_tied_embedding);
        assert_eq!(roles.layers.len(), 2);
        assert_eq!(
            roles.layers[1].attention_q.name,
            "model.layers.1.self_attn.q_proj.weight"
        );
        assert_eq!(
            roles.layers[0].ffn_down.name,
            "model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn resolve_hf_llama_tensor_roles_ties_output_to_embedding_when_lm_head_absent() {
        let config = tiny_hf_summary().to_llama_config();
        let roles =
            resolve_hf_llama_tensor_roles(&hf_descriptors(2, false), &config, true).unwrap();
        assert!(roles.output_is_tied_embedding);
        assert_eq!(roles.output.name, "model.embed_tokens.weight");
    }

    #[test]
    fn resolve_hf_llama_tensor_roles_rejects_untied_missing_lm_head() {
        let config = tiny_hf_summary().to_llama_config();
        let err =
            resolve_hf_llama_tensor_roles(&hf_descriptors(2, false), &config, false).unwrap_err();
        assert!(matches!(err, BackendError::TensorNotFound(_)));
        assert!(err.to_string().contains("lm_head.weight"));
    }

    #[test]
    fn resolve_hf_llama_tensor_roles_reports_the_missing_layer_tensor() {
        let config = tiny_hf_summary().to_llama_config();
        let mut descriptors = hf_descriptors(2, true);
        descriptors.retain(|d| d.name != "model.layers.1.mlp.up_proj.weight");
        let err = resolve_hf_llama_tensor_roles(&descriptors, &config, false).unwrap_err();
        assert!(matches!(err, BackendError::TensorNotFound(_)));
        assert!(err
            .to_string()
            .contains("model.layers.1.mlp.up_proj.weight"));
    }

    #[test]
    fn resolve_hf_llama_tensor_roles_rejects_a_mis_shaped_tensor() {
        let config = tiny_hf_summary().to_llama_config();
        let mut descriptors = hf_descriptors(2, true);
        // Corrupt the embedding to a 1x1 tensor: a name-only binder would accept
        // it; the shape gate must reject it with a typed shape mismatch.
        let embed = descriptors
            .iter_mut()
            .find(|d| d.name == "model.embed_tokens.weight")
            .unwrap();
        embed.shape = vec![1, 1];
        let err = resolve_hf_llama_tensor_roles(&descriptors, &config, false).unwrap_err();
        assert!(matches!(err, BackendError::RuntimeShapeMismatch(_)));
        assert!(err.to_string().contains("model.embed_tokens.weight"));
    }

    /// Write a SafeTensors shard with real per-tensor payload bytes and return
    /// its path (the readiness fixtures above only need zero-filled payloads).
    fn write_safetensors_with_payload(
        root: &Path,
        name: &str,
        tensors: &[(&str, &str, &[u64], &[u8])],
    ) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        let mut payload = Vec::new();
        for (tensor_name, dtype, shape, bytes) in tensors {
            let byte_len = bytes.len() as u64;
            header.insert(
                (*tensor_name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + byte_len],
                }),
            );
            payload.extend_from_slice(bytes);
            offset += byte_len;
        }
        write_safetensors_bytes(root, name, &Value::Object(header), &payload);
        root.join(name)
    }

    fn f32_payload(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn u16_payload(bits: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(bits.len() * 2);
        for value in bits {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn decodes_f32_tensor_bytes_into_cpu_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let path = write_safetensors_with_payload(
            dir.path(),
            "model.safetensors",
            &[(
                "model.embed_tokens.weight",
                "F32",
                &[2, 2],
                &f32_payload(&values),
            )],
        );
        let descriptors = parse_safetensors_header(&path).unwrap();
        let tensor = decode_safetensors_tensor(&descriptors[0]).unwrap();
        assert_eq!(tensor.name, "model.embed_tokens.weight");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(tensor.data, values);
    }

    #[test]
    fn decodes_f16_tensor_bytes_into_cpu_tensor() {
        let dir = tempfile::tempdir().unwrap();
        // IEEE half bit patterns: 1.0, 2.0, -2.0, 0.5.
        let bits = [0x3C00u16, 0x4000, 0xC000, 0x3800];
        let path = write_safetensors_with_payload(
            dir.path(),
            "model.safetensors",
            &[("lm_head.weight", "F16", &[4], &u16_payload(&bits))],
        );
        let descriptors = parse_safetensors_header(&path).unwrap();
        let tensor = decode_safetensors_tensor(&descriptors[0]).unwrap();
        assert_eq!(tensor.shape.dims, vec![4]);
        assert_eq!(tensor.data, vec![1.0, 2.0, -2.0, 0.5]);
    }

    #[test]
    fn decodes_bf16_tensor_bytes_into_cpu_tensor() {
        let dir = tempfile::tempdir().unwrap();
        // bfloat16 is the high 16 bits of the f32 pattern: 1.0, -2.0, 0.5, 4.0.
        let bits = [0x3F80u16, 0xC000, 0x3F00, 0x4080];
        let path = write_safetensors_with_payload(
            dir.path(),
            "model.safetensors",
            &[("lm_head.weight", "BF16", &[4], &u16_payload(&bits))],
        );
        let descriptors = parse_safetensors_header(&path).unwrap();
        let tensor = decode_safetensors_tensor(&descriptors[0]).unwrap();
        assert_eq!(tensor.shape.dims, vec![4]);
        assert_eq!(tensor.data, vec![1.0, -2.0, 0.5, 4.0]);
    }

    #[test]
    fn decode_reads_the_correct_shard_slice_for_a_later_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let first = [9.0f32, 9.0];
        let second = [1.5f32, -3.0, 7.0];
        let path = write_safetensors_with_payload(
            dir.path(),
            "model.safetensors",
            &[
                (
                    "model.embed_tokens.weight",
                    "F32",
                    &[2],
                    &f32_payload(&first),
                ),
                ("lm_head.weight", "F32", &[3], &f32_payload(&second)),
            ],
        );
        let descriptors = parse_safetensors_header(&path).unwrap();
        // Descriptors are sorted by name, so lm_head precedes model.embed_tokens;
        // decode each by name to prove offsets, not order, drive the byte slice.
        let lm_head = descriptors
            .iter()
            .find(|d| d.name == "lm_head.weight")
            .unwrap();
        let embed = descriptors
            .iter()
            .find(|d| d.name == "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(decode_safetensors_tensor(lm_head).unwrap().data, second);
        assert_eq!(decode_safetensors_tensor(embed).unwrap().data, first);
    }

    #[test]
    fn decode_rejects_an_unsupported_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors_with_payload(
            dir.path(),
            "model.safetensors",
            &[("some.int.tensor", "I32", &[1], &[0u8, 0, 0, 0])],
        );
        let descriptors = parse_safetensors_header(&path).unwrap();
        let err = decode_safetensors_tensor(&descriptors[0]).unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
        assert!(err.to_string().contains("I32"));
    }
}
