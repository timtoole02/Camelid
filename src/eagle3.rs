//! Strict loader for pinned Llama 3.2 3B and Qwen3 4B EAGLE-3 draft heads.
//!
//! This module deliberately admits the known 15-tensor EAGLE-3
//! checkpoint layouts. Artifact identity is pinned by the benchmark and serving
//! entry points; this loader independently pins their geometry, config variants,
//! and tensor encodings. Keeping loading separate makes the runtime fail closed
//! on the mistakes that most severely damage acceptance: silently accepting a
//! head for a different target model, using the wrong per-head RoPE base or
//! attention window, and interpreting the checkpoint's delta-coded `d2t` values
//! as absolute target token ids.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::Deserialize;

use crate::error::{BackendError, Result};

pub const HIDDEN_SIZE: usize = 3_072;
pub const INTERMEDIATE_SIZE: usize = 8_192;
pub const NUM_HIDDEN_LAYERS: usize = 1;
pub const NUM_ATTENTION_HEADS: usize = 24;
pub const NUM_KEY_VALUE_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const TARGET_VOCAB_SIZE: usize = 128_256;
pub const DRAFT_VOCAB_SIZE: usize = 32_000;
pub const ROPE_THETA: f32 = 500_000.0;
pub const SHAREGPT_ROPE_THETA: f32 = 10_000.0;
pub const RMS_NORM_EPS: f32 = 1.0e-5;
const CONFIG_RMS_NORM_EPS: f64 = 1.0e-5;

/// Target layer-input taps used when an EAGLE-3 checkpoint omits explicit tap ids.
/// Upstream derives `[2, n_layers / 2, n_layers - 3]`; Llama 3.2 3B has 28 layers.
pub const TARGET_LAYER_INPUT_IDS: [usize; 3] = [2, 14, 25];

/// AngelSlim Qwen3-4B EAGLE-3, revision fd331e59626c8e95c392381a16ee59d518727fbb.
pub const QWEN_WEIGHTS_SHA256: &str =
    "58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e";
pub const QWEN_CONFIG_SHA256: &str =
    "1fc560b1fe78e79cd31255da282651f7802bb41a8dc42c7c79e7333f036c5195";
pub const QWEN_TARGET_SHA256: &str =
    "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5";
pub const QWEN_TARGET_LAYER_INPUT_IDS: [usize; 3] = [2, 18, 33];

/// Validated draft geometry. Query width is independent of the residual width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eagle3Geometry {
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
    pub target_vocab: usize,
    pub rms_eps: f32,
}
impl Eagle3Geometry {
    pub const LLAMA: Self = Self {
        hidden: 3072,
        ffn: 8192,
        heads: 24,
        target_vocab: 128256,
        rms_eps: 1e-5,
    };
    pub const QWEN: Self = Self {
        hidden: 2560,
        ffn: 9728,
        heads: 32,
        target_vocab: 151936,
        rms_eps: 1e-6,
    };
    pub fn validate(self) -> std::result::Result<(), String> {
        if self == Self::LLAMA || self == Self::QWEN {
            Ok(())
        } else {
            Err(format!("unsupported EAGLE-3 geometry: {self:?}"))
        }
    }
    pub fn aux_width(self) -> usize {
        3 * self.hidden
    }
    pub fn attn_input(self) -> usize {
        2 * self.hidden
    }
    pub fn query_width(self) -> usize {
        self.heads * HEAD_DIM
    }
    pub fn target_layer_input_ids(self) -> [usize; 3] {
        if self == Self::QWEN {
            QWEN_TARGET_LAYER_INPUT_IDS
        } else {
            TARGET_LAYER_INPUT_IDS
        }
    }
}

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "model.safetensors";
const TRAINING_RECEIPT_FILE: &str = "training-receipt.json";
const EXPECTED_TENSOR_COUNT: usize = 15;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

pub const DERIVED_ALLOW_ENV: &str = "CAMELID_EAGLE3_ALLOW_DERIVED";
pub const DERIVED_TRAINING_RECEIPT_SCHEMA: &str = "camelid-eagle3-mlx-training-receipt-v1";
pub const THOUGHTWORKS_WEIGHTS_SHA256: &str =
    "c0713251464a9b6b5fcf9fb229587bbe59b6fd1521027aef32101d11b9ebbdaf";
pub const SHAREGPT_E8_WEIGHTS_SHA256: &str =
    "0694d52a4c7ebf3d4f9bb833cf5f2610f0cc0d30bf62a2376e0b2ee06cbe3662";
pub const SHAREGPT_E8_CONFIG_SHA256: &str =
    "1f6f8e7dcf67648757016925e28b09c40461e22b0ffe522b9abc9802ec14eff8";
pub const SHAREGPT_E9_WEIGHTS_SHA256: &str =
    "0192ee37dff4b7a86d13011d40e9cf622b331fe76f637d7d1ea24c4b81574304";
pub const SHAREGPT_E9_CONFIG_SHA256: &str =
    "a5b3a9b3674e3233cdc4f34d201a7c366a2b089ec00a41a9da6b430ca8fd3136";
pub const SHAREGPT_SW512_E9_WEIGHTS_SHA256: &str =
    "cf879511aa0e931ac2cfdaf0cc3dfa2e1ec9773c41f3c093a967420222fa84d0";
pub const SHAREGPT_SW512_E9_CONFIG_SHA256: &str =
    "c7997a68fd0f2324b41ab779c13909115b67cac9a36f758cc5b542cba12c2568";

pub const PINNED_WEIGHTS_SHA256: [&str; 4] = [
    THOUGHTWORKS_WEIGHTS_SHA256,
    SHAREGPT_E8_WEIGHTS_SHA256,
    SHAREGPT_E9_WEIGHTS_SHA256,
    SHAREGPT_SW512_E9_WEIGHTS_SHA256,
];

const D2T: &str = "d2t";
const FC: &str = "fc.weight";
const LM_HEAD: &str = "lm_head.weight";
const HIDDEN_NORM: &str = "midlayer.hidden_norm.weight";
const INPUT_NORM: &str = "midlayer.input_layernorm.weight";
const MLP_DOWN: &str = "midlayer.mlp.down_proj.weight";
const MLP_GATE: &str = "midlayer.mlp.gate_proj.weight";
const MLP_UP: &str = "midlayer.mlp.up_proj.weight";
const POST_ATTN_NORM: &str = "midlayer.post_attention_layernorm.weight";
const ATTN_K: &str = "midlayer.self_attn.k_proj.weight";
const ATTN_O: &str = "midlayer.self_attn.o_proj.weight";
const ATTN_Q: &str = "midlayer.self_attn.q_proj.weight";
const ATTN_V: &str = "midlayer.self_attn.v_proj.weight";
const OUTPUT_NORM: &str = "norm.weight";
const T2D: &str = "t2d";

#[derive(Clone, Debug, PartialEq)]
pub struct Eagle3Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub torch_dtype: String,
    pub tie_word_embeddings: bool,
    /// `None` means ordinary full causal attention. A finite window may use the
    /// full-causal runtime only while every possible draft-head position remains
    /// within the window, where the two masks are mathematically identical.
    pub sliding_window: Option<usize>,
}

impl Eagle3Config {
    pub fn geometry(&self) -> Eagle3Geometry {
        Eagle3Geometry {
            hidden: self.hidden_size,
            ffn: self.intermediate_size,
            heads: self.num_attention_heads,
            target_vocab: self.vocab_size,
            rms_eps: self.rms_norm_eps,
        }
    }
}

/// Cryptographically validated provenance for an explicitly admitted derived EAGLE head.
/// The full checkpoint loader still validates every config field and all 15 tensor descriptors
/// before the head can execute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eagle3DerivedProvenance {
    pub schema: String,
    pub receipt_sha256: String,
    pub output_weights_sha256: String,
    pub output_config_sha256: String,
    pub output_mapping_sha256: String,
    pub source_weights_sha256: String,
    pub source_config_sha256: String,
    pub source_mapping_sha256: String,
    pub tensor_count: usize,
}

#[derive(Debug, Deserialize)]
struct Eagle3TrainingReceipt {
    schema: String,
    output_weights_sha256: String,
    output_config_sha256: String,
    output_mapping_sha256: String,
    source_weights_sha256: String,
    source_config_sha256: String,
    source_mapping_sha256: String,
    tensor_count: usize,
}

#[derive(Deserialize)]
struct ConfigFile {
    architectures: Vec<String>,
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    draft_vocab_size: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    dtype: Option<String>,
    tie_word_embeddings: bool,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: Option<bool>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// One dense matrix in the checkpoint's original row-major BF16 representation.
/// No decode or transpose occurs before the future Metal upload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eagle3Bf16Matrix {
    pub name: &'static str,
    /// SafeTensors/Hugging Face order: `[output_rows, input_columns]`.
    pub shape: [usize; 2],
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eagle3Matrices {
    pub feature_fusion: Eagle3Bf16Matrix,
    pub lm_head: Eagle3Bf16Matrix,
    pub mlp_down: Eagle3Bf16Matrix,
    pub mlp_gate: Eagle3Bf16Matrix,
    pub mlp_up: Eagle3Bf16Matrix,
    pub attention_k: Eagle3Bf16Matrix,
    pub attention_o: Eagle3Bf16Matrix,
    pub attention_q: Eagle3Bf16Matrix,
    pub attention_v: Eagle3Bf16Matrix,
}

/// The four residual-width RMSNorm vectors, decoded once to f32 for Metal uniforms.
#[derive(Clone, Debug, PartialEq)]
pub struct Eagle3Norms {
    /// Normalizes the fused target feature `g` before concatenation.
    pub hidden: Vec<f32>,
    /// Normalizes the target model token embedding before concatenation.
    pub input: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub output: Vec<f32>,
}

/// Fully validated, host-resident contents of the pinned EAGLE-3 draft head.
#[derive(Clone, Debug, PartialEq)]
pub struct Eagle3DraftModel {
    pub config: Eagle3Config,
    pub matrices: Eagle3Matrices,
    pub norms: Eagle3Norms,
    /// Signed source offsets from the checkpoint's I32 `d2t` tensor. Retained for
    /// the Metal state contract; row `i` resolves to `d2t_offsets[i] + i`.
    pub d2t_offsets: Vec<i32>,
    /// Draft-vocabulary row -> absolute target-model token id.
    pub draft_to_target: Vec<u32>,
    /// The checkpoint's `t2d` membership mask, cross-checked against `draft_to_target`.
    pub target_to_draft_mask: Vec<bool>,
}

impl Eagle3DraftModel {
    /// Load a supported head with an exact config and tensor-layout contract.
    pub fn load(dir: &Path) -> Result<Self> {
        let config_path = dir.join(CONFIG_FILE);
        let weights_path = dir.join(WEIGHTS_FILE);
        let config = load_config(&config_path)?;

        let (mut file, payload_start, descriptors) = open_weights(&weights_path, &config)?;

        let matrices = Eagle3Matrices {
            feature_fusion: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                FC,
                [config.hidden_size, 3 * config.hidden_size],
            )?,
            lm_head: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                LM_HEAD,
                [DRAFT_VOCAB_SIZE, config.hidden_size],
            )?,
            mlp_down: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_DOWN,
                [config.hidden_size, config.intermediate_size],
            )?,
            mlp_gate: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_GATE,
                [config.intermediate_size, config.hidden_size],
            )?,
            mlp_up: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_UP,
                [config.intermediate_size, config.hidden_size],
            )?,
            attention_k: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_K,
                [NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * config.hidden_size],
            )?,
            attention_o: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_O,
                [
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim,
                ],
            )?,
            attention_q: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_Q,
                [
                    config.num_attention_heads * HEAD_DIM,
                    2 * config.hidden_size,
                ],
            )?,
            attention_v: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_V,
                [NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * config.hidden_size],
            )?,
        };

        let norms = Eagle3Norms {
            hidden: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                HIDDEN_NORM,
                config.hidden_size,
            )?,
            input: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                INPUT_NORM,
                config.hidden_size,
            )?,
            post_attention: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                POST_ATTN_NORM,
                config.hidden_size,
            )?,
            output: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                OUTPUT_NORM,
                config.hidden_size,
            )?,
        };

        let raw_d2t = read_tensor(
            &mut file,
            &weights_path,
            payload_start,
            descriptor(&descriptors, D2T)?,
        )?;
        let (d2t_offsets, draft_to_target) =
            decode_d2t(&raw_d2t, DRAFT_VOCAB_SIZE, config.vocab_size)?;

        let raw_t2d = read_tensor(
            &mut file,
            &weights_path,
            payload_start,
            descriptor(&descriptors, T2D)?,
        )?;
        let target_to_draft_mask =
            decode_and_validate_t2d(&raw_t2d, config.vocab_size, &draft_to_target)?;

        Ok(Self {
            config,
            matrices,
            norms,
            d2t_offsets,
            draft_to_target,
            target_to_draft_mask,
        })
    }
}

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

fn io_error(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_derived_opt_in_value(value: Option<&std::ffi::OsStr>) -> Result<()> {
    if value == Some(std::ffi::OsStr::new("1")) {
        Ok(())
    } else {
        Err(invalid(format!(
            "derived EAGLE-3 checkpoints are disabled; set {DERIVED_ALLOW_ENV}=1 explicitly (got {value:?})"
        )))
    }
}

/// Require the one exact process-level opt-in accepted for non-pinned EAGLE checkpoints.
pub fn require_derived_opt_in() -> Result<()> {
    let value = std::env::var_os(DERIVED_ALLOW_ENV);
    validate_derived_opt_in_value(value.as_deref())
}

fn mapping_sha256(d2t_dtype: &str, raw_d2t: &[u8], raw_t2d: &[u8]) -> String {
    // This is the exact mapping contract emitted by tools/eagle3_mlx/contract.py.
    // Include the encoded d2t dtype and both validated payloads so a receipt cannot
    // silently authorize a representation change that merely decodes to the same IDs.
    let mut bytes = Vec::with_capacity(d2t_dtype.len() + raw_d2t.len() + raw_t2d.len());
    bytes.extend_from_slice(d2t_dtype.as_bytes());
    bytes.extend_from_slice(raw_d2t);
    bytes.extend_from_slice(raw_t2d);
    crate::receipt::sha256_hex(&bytes)
}

/// The config contract each pinned source checkpoint must satisfy: `architectures`, `rope_theta`
/// and `sliding_window` are the only fields `parse_and_validate_config` leaves free, and only
/// this table ties the surviving combination to a specific checkpoint. `false` for any unpinned
/// hash, so callers reject rather than fall back to another variant's contract. Callers are the
/// derived-checkpoint validator below, the serving loader
/// (`api::load_eagle3_checkpoint_cached`, which keeps its own copy of this table for the config
/// the full loader parsed — keep the two in step when a pin is added) and `bench-eagle3`.
pub fn config_matches_pinned_source(config: &Eagle3Config, source_weights_sha256: &str) -> bool {
    match source_weights_sha256 {
        THOUGHTWORKS_WEIGHTS_SHA256 => {
            config.architectures == ["LlamaForCausalLM"]
                && config.rope_theta == ROPE_THETA
                && config.sliding_window.is_none()
        }
        SHAREGPT_E8_WEIGHTS_SHA256 => {
            config.architectures == ["LlamaForCausalLMEagle3"]
                && config.rope_theta == SHAREGPT_ROPE_THETA
                && config.sliding_window.is_none()
        }
        SHAREGPT_E9_WEIGHTS_SHA256 => {
            config.architectures == ["LlamaForCausalLMEagle3"]
                && config.rope_theta == SHAREGPT_ROPE_THETA
                && config.sliding_window == Some(256)
        }
        SHAREGPT_SW512_E9_WEIGHTS_SHA256 => {
            config.architectures == ["LlamaForCausalLMEagle3"]
                && config.rope_theta == SHAREGPT_ROPE_THETA
                && config.sliding_window == Some(512)
        }
        _ => false,
    }
}

fn validate_derived_receipt_fields(
    receipt_bytes: &[u8],
    receipt_sha256: String,
    actual_weights_sha256: &str,
    actual_config_sha256: &str,
    actual_mapping_sha256: &str,
    config: &Eagle3Config,
) -> Result<Eagle3DerivedProvenance> {
    let receipt: Eagle3TrainingReceipt =
        serde_json::from_slice(receipt_bytes).map_err(|error| {
            invalid(format!(
                "invalid derived EAGLE-3 {TRAINING_RECEIPT_FILE}: {error}"
            ))
        })?;
    if receipt.schema != DERIVED_TRAINING_RECEIPT_SCHEMA {
        return Err(invalid(format!(
            "derived EAGLE-3 receipt schema is {:?}, expected {DERIVED_TRAINING_RECEIPT_SCHEMA:?}",
            receipt.schema
        )));
    }
    for (field, value) in [
        ("output_weights_sha256", &receipt.output_weights_sha256),
        ("output_config_sha256", &receipt.output_config_sha256),
        ("output_mapping_sha256", &receipt.output_mapping_sha256),
        ("source_weights_sha256", &receipt.source_weights_sha256),
        ("source_config_sha256", &receipt.source_config_sha256),
        ("source_mapping_sha256", &receipt.source_mapping_sha256),
    ] {
        if !is_lowercase_sha256(value) {
            return Err(invalid(format!(
                "derived EAGLE-3 receipt field {field} must be exactly 64 lowercase hexadecimal characters"
            )));
        }
    }
    if receipt.tensor_count != EXPECTED_TENSOR_COUNT {
        return Err(invalid(format!(
            "derived EAGLE-3 receipt tensor_count is {}, expected {EXPECTED_TENSOR_COUNT}",
            receipt.tensor_count
        )));
    }
    for (field, declared, actual) in [
        (
            "output_weights_sha256",
            receipt.output_weights_sha256.as_str(),
            actual_weights_sha256,
        ),
        (
            "output_config_sha256",
            receipt.output_config_sha256.as_str(),
            actual_config_sha256,
        ),
        (
            "output_mapping_sha256",
            receipt.output_mapping_sha256.as_str(),
            actual_mapping_sha256,
        ),
    ] {
        if declared != actual {
            return Err(invalid(format!(
                "derived EAGLE-3 receipt {field} is {declared}, actual artifact is {actual}"
            )));
        }
    }
    if !PINNED_WEIGHTS_SHA256.contains(&receipt.source_weights_sha256.as_str()) {
        return Err(invalid(format!(
            "derived EAGLE-3 source_weights_sha256 {} is not a pinned source checkpoint",
            receipt.source_weights_sha256
        )));
    }
    if receipt.source_config_sha256 != receipt.output_config_sha256 {
        return Err(invalid(format!(
            "derived EAGLE-3 changed the source config contract: source={} output={}",
            receipt.source_config_sha256, receipt.output_config_sha256
        )));
    }
    if receipt.source_mapping_sha256 != receipt.output_mapping_sha256 {
        return Err(invalid(format!(
            "derived EAGLE-3 changed the source d2t/t2d mapping contract: source={} output={}",
            receipt.source_mapping_sha256, receipt.output_mapping_sha256
        )));
    }
    let pinned_source_config = match receipt.source_weights_sha256.as_str() {
        SHAREGPT_E8_WEIGHTS_SHA256 => Some(SHAREGPT_E8_CONFIG_SHA256),
        SHAREGPT_E9_WEIGHTS_SHA256 => Some(SHAREGPT_E9_CONFIG_SHA256),
        SHAREGPT_SW512_E9_WEIGHTS_SHA256 => Some(SHAREGPT_SW512_E9_CONFIG_SHA256),
        _ => None,
    };
    if let Some(expected) = pinned_source_config {
        if receipt.source_config_sha256 != expected {
            return Err(invalid(format!(
                "derived EAGLE-3 source config SHA-256 is {}, expected {expected} for source weights {}",
                receipt.source_config_sha256, receipt.source_weights_sha256
            )));
        }
    }
    if !config_matches_pinned_source(config, &receipt.source_weights_sha256) {
        return Err(invalid(format!(
            "derived EAGLE-3 config variant does not match pinned source weights {}",
            receipt.source_weights_sha256
        )));
    }
    Ok(Eagle3DerivedProvenance {
        schema: receipt.schema,
        receipt_sha256,
        output_weights_sha256: receipt.output_weights_sha256,
        output_config_sha256: receipt.output_config_sha256,
        output_mapping_sha256: receipt.output_mapping_sha256,
        source_weights_sha256: receipt.source_weights_sha256,
        source_config_sha256: receipt.source_config_sha256,
        source_mapping_sha256: receipt.source_mapping_sha256,
        tensor_count: receipt.tensor_count,
    })
}

/// Validate the standard MLX training receipt and the immutable contract surfaces of a derived
/// checkpoint. Callers must separately require `CAMELID_EAGLE3_ALLOW_DERIVED=1` before invoking
/// this function. The ordinary full loader remains authoritative for matrix payload loading.
pub fn validate_derived_checkpoint(
    dir: &Path,
    actual_weights_sha256: &str,
) -> Result<Eagle3DerivedProvenance> {
    if !is_lowercase_sha256(actual_weights_sha256) {
        return Err(invalid(
            "actual derived EAGLE-3 weights SHA-256 is not lowercase hexadecimal",
        ));
    }
    let config_path = dir.join(CONFIG_FILE);
    let weights_path = dir.join(WEIGHTS_FILE);
    let receipt_path = dir.join(TRAINING_RECEIPT_FILE);
    let config_bytes = fs::read(&config_path).map_err(|source| io_error(&config_path, source))?;
    let config = parse_and_validate_config(&config_bytes)?;
    let actual_config_sha256 = crate::receipt::sha256_hex(&config_bytes);

    // Validate the complete 15-tensor header and byte layout, but read only the mapping tensors
    // here. The full loader repeats this gate before loading matrices into the runtime.
    let (mut file, payload_start, descriptors) = open_weights(&weights_path, &config)?;
    let d2t_descriptor = descriptor(&descriptors, D2T)?;
    let raw_d2t = read_tensor(&mut file, &weights_path, payload_start, d2t_descriptor)?;
    let (_, draft_to_target) = decode_d2t(&raw_d2t, DRAFT_VOCAB_SIZE, config.vocab_size)?;
    let raw_t2d = read_tensor(
        &mut file,
        &weights_path,
        payload_start,
        descriptor(&descriptors, T2D)?,
    )?;
    decode_and_validate_t2d(&raw_t2d, config.vocab_size, &draft_to_target)?;
    let actual_mapping_sha256 = mapping_sha256(d2t_descriptor.dtype, &raw_d2t, &raw_t2d);

    let receipt_bytes =
        fs::read(&receipt_path).map_err(|source| io_error(&receipt_path, source))?;
    let receipt_sha256 = crate::receipt::sha256_hex(&receipt_bytes);
    validate_derived_receipt_fields(
        &receipt_bytes,
        receipt_sha256,
        actual_weights_sha256,
        &actual_config_sha256,
        &actual_mapping_sha256,
        &config,
    )
}

fn require_equal<T: Debug + PartialEq>(field: &str, actual: &T, expected: &T) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "EAGLE-3 config field {field} is {actual:?}, expected {expected:?}"
        )))
    }
}

fn parse_and_validate_config(bytes: &[u8]) -> Result<Eagle3Config> {
    let raw: ConfigFile = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("invalid EAGLE-3 config.json: {error}")))?;

    let architecture_ok = matches!(
        raw.architectures.as_slice(),
        [architecture]
            if architecture == "LlamaForCausalLM"
                || architecture == "LlamaForCausalLMEagle3"
                || architecture == "Eagle3LlamaForCausalLM"
    );
    if !architecture_ok {
        return Err(invalid(format!(
            "EAGLE-3 config field architectures is {:?}, expected exactly one of LlamaForCausalLM or LlamaForCausalLMEagle3",
            raw.architectures
        )));
    }
    let sharegpt_extra = BTreeMap::from([
        ("attention_bias".to_string(), serde_json::json!(false)),
        ("attention_dropout".to_string(), serde_json::json!(0.0)),
        ("bos_token_id".to_string(), serde_json::json!(128000)),
        (
            "eos_token_id".to_string(),
            serde_json::json!([128001, 128008, 128009]),
        ),
        ("hidden_act".to_string(), serde_json::json!("silu")),
        ("initializer_range".to_string(), serde_json::json!(0.02)),
        (
            "max_position_embeddings".to_string(),
            serde_json::json!(131072),
        ),
        ("mlp_bias".to_string(), serde_json::json!(false)),
        ("pad_token_id".to_string(), serde_json::json!(0)),
        ("pretraining_tp".to_string(), serde_json::json!(1)),
        ("rope_scaling".to_string(), serde_json::Value::Null),
        (
            "transformers_version".to_string(),
            serde_json::json!("4.57.1"),
        ),
        ("use_cache".to_string(), serde_json::json!(true)),
    ]);
    let is_sharegpt = raw.architectures == ["LlamaForCausalLMEagle3"];
    let is_qwen = raw.architectures == ["Eagle3LlamaForCausalLM"];
    let geometry = if is_qwen {
        Eagle3Geometry::QWEN
    } else {
        Eagle3Geometry::LLAMA
    };
    let expected_extra = if is_qwen {
        let mut extra = sharegpt_extra;
        extra.insert("bos_token_id".to_string(), serde_json::json!(151643));
        extra.insert("eos_token_id".to_string(), serde_json::json!(151645));
        extra.insert(
            "max_position_embeddings".to_string(),
            serde_json::json!(40960),
        );
        extra.insert("max_window_layers".to_string(), serde_json::json!(36));
        extra.remove("pad_token_id");
        extra
    } else if is_sharegpt {
        sharegpt_extra
    } else {
        BTreeMap::new()
    };
    if raw.extra != expected_extra {
        return Err(invalid(format!(
            "EAGLE-3 config extra fields are {:?}, expected {:?} for architecture {}",
            raw.extra, expected_extra, raw.architectures[0]
        )));
    }
    let sliding_window = match (
        is_sharegpt,
        raw.sliding_window,
        raw.use_sliding_window,
    ) {
        (true, None, None) => None,
        (true, Some(window @ (256 | 512)), Some(true)) => Some(window),
        (false, None, None) => None,
        (false, None, Some(false)) if is_qwen => None,
        (_, window, enabled) => {
            return Err(invalid(format!(
                "EAGLE-3 config sliding-window fields are sliding_window={window:?}, use_sliding_window={enabled:?}; expected both absent, or sliding_window in [256, 512] with use_sliding_window=true for LlamaForCausalLMEagle3"
            )))
        }
    };
    require_equal("model_type", &raw.model_type, &"llama".to_string())?;
    require_equal("hidden_size", &raw.hidden_size, &geometry.hidden)?;
    require_equal("intermediate_size", &raw.intermediate_size, &geometry.ffn)?;
    require_equal(
        "num_hidden_layers",
        &raw.num_hidden_layers,
        &NUM_HIDDEN_LAYERS,
    )?;
    require_equal(
        "num_attention_heads",
        &raw.num_attention_heads,
        &geometry.heads,
    )?;
    require_equal(
        "num_key_value_heads",
        &raw.num_key_value_heads,
        &NUM_KEY_VALUE_HEADS,
    )?;
    require_equal("head_dim", &raw.head_dim, &HEAD_DIM)?;
    require_equal("vocab_size", &raw.vocab_size, &geometry.target_vocab)?;
    require_equal("draft_vocab_size", &raw.draft_vocab_size, &DRAFT_VOCAB_SIZE)?;
    if (is_qwen && raw.rope_theta != 1_000_000.0)
        || (!is_qwen
            && raw.rope_theta != ROPE_THETA as f64
            && raw.rope_theta != SHAREGPT_ROPE_THETA as f64)
    {
        return Err(invalid(format!(
            "EAGLE-3 config field rope_theta is {:?}, expected one of {:?}",
            raw.rope_theta,
            [ROPE_THETA, SHAREGPT_ROPE_THETA]
        )));
    }
    require_equal(
        "rms_norm_eps",
        &raw.rms_norm_eps,
        &(if is_qwen { 1e-6 } else { CONFIG_RMS_NORM_EPS }),
    )?;
    let dtype =
        match (raw.torch_dtype.as_deref(), raw.dtype.as_deref()) {
            (Some(torch), None) | (None, Some(torch)) => torch,
            (Some(torch), Some(dtype)) if torch == dtype => torch,
            (Some(torch), Some(dtype)) => {
                return Err(invalid(format!(
                    "EAGLE-3 config dtype aliases disagree: torch_dtype={torch:?}, dtype={dtype:?}"
                )))
            }
            (None, None) => return Err(invalid(
                "EAGLE-3 config must contain exactly one BF16 dtype field (torch_dtype or dtype)",
            )),
        };
    require_equal("dtype", &dtype, &"bfloat16")?;
    require_equal("tie_word_embeddings", &raw.tie_word_embeddings, &false)?;

    Ok(Eagle3Config {
        architectures: raw.architectures,
        model_type: raw.model_type,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_hidden_layers: raw.num_hidden_layers,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
        vocab_size: raw.vocab_size,
        draft_vocab_size: raw.draft_vocab_size,
        rope_theta: raw.rope_theta as f32,
        rms_norm_eps: raw.rms_norm_eps as f32,
        torch_dtype: dtype.to_string(),
        tie_word_embeddings: raw.tie_word_embeddings,
        sliding_window,
    })
}

fn load_config(path: &Path) -> Result<Eagle3Config> {
    let bytes = fs::read(path).map_err(|source| io_error(path, source))?;
    parse_and_validate_config(&bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorSpec {
    name: &'static str,
    dtype: &'static str,
    shape: &'static [u64],
}

const TENSOR_SPECS: &[TensorSpec] = &[
    TensorSpec {
        name: D2T,
        dtype: "I32",
        shape: &[DRAFT_VOCAB_SIZE as u64],
    },
    TensorSpec {
        name: FC,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64, (3 * HIDDEN_SIZE) as u64],
    },
    TensorSpec {
        name: LM_HEAD,
        dtype: "BF16",
        shape: &[DRAFT_VOCAB_SIZE as u64, HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: HIDDEN_NORM,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: INPUT_NORM,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: MLP_DOWN,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
    },
    TensorSpec {
        name: MLP_GATE,
        dtype: "BF16",
        shape: &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: MLP_UP,
        dtype: "BF16",
        shape: &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: POST_ATTN_NORM,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: ATTN_K,
        dtype: "BF16",
        shape: &[
            (NUM_KEY_VALUE_HEADS * HEAD_DIM) as u64,
            (2 * HIDDEN_SIZE) as u64,
        ],
    },
    TensorSpec {
        name: ATTN_O,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: ATTN_Q,
        dtype: "BF16",
        shape: &[
            (NUM_ATTENTION_HEADS * HEAD_DIM) as u64,
            (2 * HIDDEN_SIZE) as u64,
        ],
    },
    TensorSpec {
        name: ATTN_V,
        dtype: "BF16",
        shape: &[
            (NUM_KEY_VALUE_HEADS * HEAD_DIM) as u64,
            (2 * HIDDEN_SIZE) as u64,
        ],
    },
    TensorSpec {
        name: OUTPUT_NORM,
        dtype: "BF16",
        shape: &[HIDDEN_SIZE as u64],
    },
    TensorSpec {
        name: T2D,
        dtype: "BOOL",
        shape: &[TARGET_VOCAB_SIZE as u64],
    },
];

const QWEN_TENSOR_SPECS: &[TensorSpec] = &[
    TensorSpec {
        name: D2T,
        dtype: "I32",
        shape: &[DRAFT_VOCAB_SIZE as u64],
    },
    TensorSpec {
        name: FC,
        dtype: "BF16",
        shape: &[2560, (3 * 2560) as u64],
    },
    TensorSpec {
        name: LM_HEAD,
        dtype: "BF16",
        shape: &[DRAFT_VOCAB_SIZE as u64, 2560],
    },
    TensorSpec {
        name: HIDDEN_NORM,
        dtype: "BF16",
        shape: &[2560],
    },
    TensorSpec {
        name: INPUT_NORM,
        dtype: "BF16",
        shape: &[2560],
    },
    TensorSpec {
        name: MLP_DOWN,
        dtype: "BF16",
        shape: &[2560, 9728],
    },
    TensorSpec {
        name: MLP_GATE,
        dtype: "BF16",
        shape: &[9728, 2560],
    },
    TensorSpec {
        name: MLP_UP,
        dtype: "BF16",
        shape: &[9728, 2560],
    },
    TensorSpec {
        name: POST_ATTN_NORM,
        dtype: "BF16",
        shape: &[2560],
    },
    TensorSpec {
        name: ATTN_K,
        dtype: "BF16",
        shape: &[(NUM_KEY_VALUE_HEADS * HEAD_DIM) as u64, (2 * 2560) as u64],
    },
    TensorSpec {
        name: ATTN_O,
        dtype: "BF16",
        shape: &[2560, 4096],
    },
    TensorSpec {
        name: ATTN_Q,
        dtype: "BF16",
        shape: &[(32 * HEAD_DIM) as u64, (2 * 2560) as u64],
    },
    TensorSpec {
        name: ATTN_V,
        dtype: "BF16",
        shape: &[(NUM_KEY_VALUE_HEADS * HEAD_DIM) as u64, (2 * 2560) as u64],
    },
    TensorSpec {
        name: OUTPUT_NORM,
        dtype: "BF16",
        shape: &[2560],
    },
    TensorSpec {
        name: T2D,
        dtype: "BOOL",
        shape: &[151936],
    },
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorDescriptor {
    start: u64,
    end: u64,
    dtype: &'static str,
}

impl TensorDescriptor {
    fn len(self) -> u64 {
        self.end - self.start
    }
}

fn dtype_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "BOOL" => Some(1),
        "BF16" => Some(2),
        "I32" => Some(4),
        "I64" => Some(8),
        _ => None,
    }
}

fn tensor_elements(spec: &TensorSpec) -> Result<u64> {
    let elements = spec.shape.iter().try_fold(1u64, |acc, &dimension| {
        acc.checked_mul(dimension).ok_or_else(|| {
            invalid(format!(
                "EAGLE-3 tensor {} element count overflows",
                spec.name
            ))
        })
    })?;
    Ok(elements)
}

fn tensor_bytes(spec: &TensorSpec) -> Result<u64> {
    tensor_elements(spec)?
        .checked_mul(dtype_bytes(spec.dtype).expect("all pinned dtypes have a width"))
        .ok_or_else(|| invalid(format!("EAGLE-3 tensor {} byte count overflows", spec.name)))
}

#[cfg(test)]
fn parse_and_validate_header(
    header: &[u8],
    payload_bytes: u64,
) -> Result<BTreeMap<&'static str, TensorDescriptor>> {
    parse_and_validate_header_with_specs(header, payload_bytes, TENSOR_SPECS)
}

fn parse_and_validate_header_with_specs(
    header_bytes: &[u8],
    payload_bytes: u64,
    specs: &[TensorSpec],
) -> Result<BTreeMap<&'static str, TensorDescriptor>> {
    let header: serde_json::Value = serde_json::from_slice(header_bytes)
        .map_err(|error| invalid(format!("invalid EAGLE-3 SafeTensors header JSON: {error}")))?;
    let object = header
        .as_object()
        .ok_or_else(|| invalid("EAGLE-3 SafeTensors header root is not an object"))?;
    if let Some(metadata) = object.get("__metadata__") {
        if !metadata.is_object() {
            return Err(invalid(
                "EAGLE-3 SafeTensors __metadata__ entry is not an object",
            ));
        }
    }

    let actual_names: BTreeSet<&str> = object
        .keys()
        .filter(|name| name.as_str() != "__metadata__")
        .map(String::as_str)
        .collect();
    let expected_names: BTreeSet<&str> = specs.iter().map(|spec| spec.name).collect();
    if actual_names != expected_names {
        let missing: Vec<&str> = expected_names.difference(&actual_names).copied().collect();
        let extra: Vec<&str> = actual_names.difference(&expected_names).copied().collect();
        return Err(invalid(format!(
            "EAGLE-3 SafeTensors tensor set differs from the pinned {EXPECTED_TENSOR_COUNT}-tensor contract; missing={missing:?}, extra={extra:?}"
        )));
    }

    let mut descriptors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(EXPECTED_TENSOR_COUNT);
    for spec in specs {
        let value = object
            .get(spec.name)
            .expect("name-set equality established above")
            .clone();
        let tensor: HeaderTensor = serde_json::from_value(value).map_err(|error| {
            invalid(format!(
                "EAGLE-3 tensor {} descriptor is invalid: {error}",
                spec.name
            ))
        })?;
        let dtype_allowed = tensor.dtype == spec.dtype
            || (spec.name == D2T && tensor.dtype == "I64" && spec.dtype == "I32");
        if !dtype_allowed {
            return Err(invalid(format!(
                "EAGLE-3 tensor {} dtype is {:?}, expected {}{}",
                spec.name,
                tensor.dtype,
                spec.dtype,
                if spec.name == D2T { " or I64" } else { "" }
            )));
        }
        require_equal(
            &format!("tensor {} shape", spec.name),
            &tensor.shape.as_slice(),
            &spec.shape,
        )?;
        let [start, end] = tensor.data_offsets;
        if start > end {
            return Err(invalid(format!(
                "EAGLE-3 tensor {} has descending data_offsets [{start}, {end}]",
                spec.name
            )));
        }
        if end > payload_bytes {
            return Err(invalid(format!(
                "EAGLE-3 tensor {} ends at payload offset {end}, past {payload_bytes}",
                spec.name
            )));
        }
        let expected_bytes = if tensor.dtype == spec.dtype {
            tensor_bytes(spec)?
        } else {
            let element_bytes = dtype_bytes(&tensor.dtype).ok_or_else(|| {
                invalid(format!(
                    "EAGLE-3 tensor {} has unsupported dtype {}",
                    spec.name, tensor.dtype
                ))
            })?;
            tensor_elements(spec)?
                .checked_mul(element_bytes)
                .ok_or_else(|| {
                    invalid(format!("EAGLE-3 tensor {} byte count overflows", spec.name))
                })?
        };
        if end - start != expected_bytes {
            return Err(invalid(format!(
                "EAGLE-3 tensor {} occupies {} bytes, expected {expected_bytes}",
                spec.name,
                end - start
            )));
        }
        let dtype = if tensor.dtype == "I64" {
            "I64"
        } else {
            spec.dtype
        };
        let descriptor = TensorDescriptor { start, end, dtype };
        descriptors.insert(spec.name, descriptor);
        ranges.push((start, end, spec.name));
    }

    // SafeTensors payloads are dense. Requiring complete, non-overlapping coverage
    // rejects aliases, unaccounted trailing data, and offset-table corruption.
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    let mut cursor = 0u64;
    for (start, end, name) in ranges {
        if start != cursor {
            return Err(invalid(format!(
                "EAGLE-3 tensor {name} starts at {start}, expected contiguous payload offset {cursor}"
            )));
        }
        cursor = end;
    }
    if cursor != payload_bytes {
        return Err(invalid(format!(
            "EAGLE-3 tensors cover {cursor} payload bytes, file contains {payload_bytes}"
        )));
    }

    Ok(descriptors)
}

fn open_weights(
    path: &Path,
    config: &Eagle3Config,
) -> Result<(File, u64, BTreeMap<&'static str, TensorDescriptor>)> {
    let mut file = File::open(path).map_err(|source| io_error(path, source))?;
    let file_bytes = file
        .metadata()
        .map_err(|source| io_error(path, source))?
        .len();
    if file_bytes < 8 {
        return Err(invalid(format!(
            "EAGLE-3 weights file {} is shorter than the 8-byte SafeTensors header length",
            path.display()
        )));
    }

    let mut length_bytes = [0u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|source| io_error(path, source))?;
    let header_bytes = u64::from_le_bytes(length_bytes);
    if header_bytes > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "EAGLE-3 SafeTensors header is {header_bytes} bytes, above the {MAX_HEADER_BYTES}-byte safety limit"
        )));
    }
    let payload_start = 8u64
        .checked_add(header_bytes)
        .ok_or_else(|| invalid("EAGLE-3 SafeTensors header offset overflows"))?;
    if payload_start > file_bytes {
        return Err(invalid(format!(
            "EAGLE-3 SafeTensors header ends at {payload_start}, past {file_bytes}-byte file"
        )));
    }
    let header_len = usize::try_from(header_bytes)
        .map_err(|_| invalid("EAGLE-3 SafeTensors header does not fit this platform"))?;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .map_err(|source| io_error(path, source))?;
    let payload_bytes = file_bytes - payload_start;
    let descriptors = parse_and_validate_header_with_specs(
        &header,
        payload_bytes,
        if config.geometry() == Eagle3Geometry::QWEN {
            QWEN_TENSOR_SPECS
        } else {
            TENSOR_SPECS
        },
    )?;
    Ok((file, payload_start, descriptors))
}

fn descriptor(
    descriptors: &BTreeMap<&'static str, TensorDescriptor>,
    name: &'static str,
) -> Result<TensorDescriptor> {
    descriptors
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("EAGLE-3 tensor {name} is missing after validation")))
}

fn read_tensor(
    file: &mut File,
    path: &Path,
    payload_start: u64,
    descriptor: TensorDescriptor,
) -> Result<Vec<u8>> {
    let absolute = payload_start
        .checked_add(descriptor.start)
        .ok_or_else(|| invalid("EAGLE-3 tensor absolute file offset overflows"))?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|source| io_error(path, source))?;
    let byte_len = usize::try_from(descriptor.len())
        .map_err(|_| invalid("EAGLE-3 tensor byte length does not fit this platform"))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|error| {
        invalid(format!(
            "could not allocate {byte_len} bytes for an EAGLE-3 tensor: {error}"
        ))
    })?;
    bytes.resize(byte_len, 0);
    file.read_exact(&mut bytes)
        .map_err(|source| io_error(path, source))?;
    Ok(bytes)
}

fn load_matrix(
    file: &mut File,
    path: &Path,
    payload_start: u64,
    descriptors: &BTreeMap<&'static str, TensorDescriptor>,
    name: &'static str,
    shape: [usize; 2],
) -> Result<Eagle3Bf16Matrix> {
    let bytes = read_tensor(file, path, payload_start, descriptor(descriptors, name)?)?;
    Ok(Eagle3Bf16Matrix { name, shape, bytes })
}

fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(BackendError::InvalidTensorData(format!(
            "BF16 payload contains an odd byte count {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

fn load_norm(
    file: &mut File,
    path: &Path,
    payload_start: u64,
    descriptors: &BTreeMap<&'static str, TensorDescriptor>,
    name: &'static str,
    hidden: usize,
) -> Result<Vec<f32>> {
    let bytes = read_tensor(file, path, payload_start, descriptor(descriptors, name)?)?;
    let values = decode_bf16(&bytes)?;
    if values.len() != hidden {
        return Err(invalid(format!(
            "EAGLE-3 norm {name} decoded to {} values, expected {hidden}",
            values.len()
        )));
    }
    Ok(values)
}

/// Both known source checkpoints store a monotone delta from each draft row's
/// index. The original head encodes it as I32 and the ShareGPT head as I64;
/// runtime offsets stay I32, so the wider representation is narrowed only after
/// a checked conversion. llama.cpp's converter uses the same `raw[i] + i`
/// reconstruction before runtime.
fn decode_d2t(
    bytes: &[u8],
    draft_vocab: usize,
    target_vocab: usize,
) -> Result<(Vec<i32>, Vec<u32>)> {
    let expected_i32_bytes = draft_vocab
        .checked_mul(4)
        .ok_or_else(|| invalid("EAGLE-3 d2t byte count overflows"))?;
    let expected_i64_bytes = draft_vocab
        .checked_mul(8)
        .ok_or_else(|| invalid("EAGLE-3 d2t byte count overflows"))?;
    let element_bytes = if bytes.len() == expected_i32_bytes {
        4
    } else if bytes.len() == expected_i64_bytes {
        8
    } else {
        return Err(invalid(format!(
            "EAGLE-3 d2t contains {} bytes, expected {expected_i32_bytes} (I32) or {expected_i64_bytes} (I64)",
            bytes.len()
        )));
    };

    let mut seen = BTreeSet::new();
    let mut offsets = Vec::with_capacity(draft_vocab);
    let mut absolute = Vec::with_capacity(draft_vocab);
    for (index, encoded) in bytes.chunks_exact(element_bytes).enumerate() {
        let wide_delta = if element_bytes == 4 {
            i64::from(i32::from_le_bytes([
                encoded[0], encoded[1], encoded[2], encoded[3],
            ]))
        } else {
            i64::from_le_bytes([
                encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
                encoded[7],
            ])
        };
        let delta = i32::try_from(wide_delta).map_err(|_| {
            invalid(format!(
                "EAGLE-3 d2t row {index} offset {wide_delta} does not fit the runtime I32 contract"
            ))
        })?;
        let token = wide_delta
            .checked_add(index as i64)
            .ok_or_else(|| invalid(format!("EAGLE-3 d2t row {index} overflows")))?;
        if token < 0 || token >= target_vocab as i64 {
            return Err(invalid(format!(
                "EAGLE-3 d2t row {index} resolves to target token {token}, outside 0..{target_vocab}"
            )));
        }
        let token = token as usize;
        if !seen.insert(token) {
            return Err(invalid(format!(
                "EAGLE-3 d2t resolves more than one draft row to target token {token}"
            )));
        }
        offsets.push(delta);
        absolute.push(token as u32);
    }
    Ok((offsets, absolute))
}

fn decode_and_validate_t2d(
    bytes: &[u8],
    target_vocab: usize,
    draft_to_target: &[u32],
) -> Result<Vec<bool>> {
    if bytes.len() != target_vocab {
        return Err(invalid(format!(
            "EAGLE-3 t2d contains {} bytes, expected {target_vocab}",
            bytes.len()
        )));
    }
    let mut mask = Vec::with_capacity(target_vocab);
    for (target, &value) in bytes.iter().enumerate() {
        match value {
            0 => mask.push(false),
            1 => mask.push(true),
            _ => {
                return Err(invalid(format!(
                    "EAGLE-3 t2d BOOL row {target} contains non-boolean byte {value}"
                )))
            }
        }
    }
    let marked = mask.iter().filter(|&&present| present).count();
    if marked != draft_to_target.len() {
        return Err(invalid(format!(
            "EAGLE-3 t2d marks {marked} target tokens, but d2t contains {} rows",
            draft_to_target.len()
        )));
    }
    for &target in draft_to_target {
        if !mask[target as usize] {
            return Err(invalid(format!(
                "EAGLE-3 t2d does not mark d2t target token {target}"
            )));
        }
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map, Value};

    const PINNED_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "draft_vocab_size": 32000,
        "rope_theta": 500000.0,
        "rms_norm_eps": 0.00001,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": false
    }"#;

    const SHAREGPT_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "draft_vocab_size": 32000,
        "eos_token_id": [128001, 128008, 128009],
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "max_position_embeddings": 131072,
        "mlp_bias": false,
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rope_theta": 10000.0,
        "rope_scaling": null,
        "rms_norm_eps": 0.00001,
        "sliding_window": 256,
        "dtype": "bfloat16",
        "tie_word_embeddings": false,
        "transformers_version": "4.57.1",
        "use_cache": true,
        "use_sliding_window": true
    }"#;

    fn pinned_header() -> (Map<String, Value>, u64) {
        let mut header = Map::new();
        let mut cursor = 0u64;
        for spec in TENSOR_SPECS {
            let bytes = tensor_bytes(spec).unwrap();
            header.insert(
                spec.name.to_string(),
                json!({
                    "dtype": spec.dtype,
                    "shape": spec.shape,
                    "data_offsets": [cursor, cursor + bytes],
                }),
            );
            cursor += bytes;
        }
        (header, cursor)
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn i64_bytes(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn derived_receipt_json(
        output_weights_sha256: &str,
        output_config_sha256: &str,
        output_mapping_sha256: &str,
    ) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": DERIVED_TRAINING_RECEIPT_SCHEMA,
            "output_weights_sha256": output_weights_sha256,
            "output_config_sha256": output_config_sha256,
            "output_mapping_sha256": output_mapping_sha256,
            "source_weights_sha256": THOUGHTWORKS_WEIGHTS_SHA256,
            "source_config_sha256": output_config_sha256,
            "source_mapping_sha256": output_mapping_sha256,
            "tensor_count": EXPECTED_TENSOR_COUNT,
            "objective": "official-specforge-soft-ce",
            "metrics": {"steps": 0},
        }))
        .unwrap()
    }

    #[test]
    fn qwen_config_and_header_preserve_independent_query_width() {
        let raw = serde_json::json!({
            "architectures":["Eagle3LlamaForCausalLM"], "attention_bias":false,
            "attention_dropout":0.0, "bos_token_id":151643, "draft_vocab_size":32000,
            "dtype":"bfloat16", "eos_token_id":151645, "head_dim":128, "vocab_size":151936,
            "hidden_act":"silu", "hidden_size":2560, "initializer_range":0.02,
            "intermediate_size":9728, "max_position_embeddings":40960,
            "max_window_layers":36, "mlp_bias":false, "model_type":"llama",
            "num_attention_heads":32, "num_hidden_layers":1, "num_key_value_heads":8,
            "pretraining_tp":1, "rms_norm_eps":1e-6, "rope_scaling":null,
            "rope_theta":1000000, "sliding_window":null, "tie_word_embeddings":false,
            "transformers_version":"4.57.1", "use_cache":true, "use_sliding_window":false
        });
        let config = parse_and_validate_config(&serde_json::to_vec(&raw).unwrap()).unwrap();
        assert_eq!(config.geometry(), Eagle3Geometry::QWEN);
        assert_eq!(config.geometry().query_width(), 4096);
        assert_eq!(config.geometry().target_layer_input_ids(), [2, 18, 33]);
        let mut header = Map::new();
        let mut cursor = 0;
        for spec in QWEN_TENSOR_SPECS {
            let bytes = if spec.name == D2T {
                32000 * 8
            } else {
                tensor_bytes(spec).unwrap()
            };
            header.insert(
                spec.name.to_string(),
                json!({
                    "dtype": if spec.name == D2T { "I64" } else { spec.dtype },
                    "shape": spec.shape, "data_offsets": [cursor, cursor + bytes]
                }),
            );
            cursor += bytes;
        }
        assert_eq!(cursor, 436898176);
        let encoded = serde_json::to_vec(&header).unwrap();
        assert!(parse_and_validate_header_with_specs(&encoded, cursor, QWEN_TENSOR_SPECS).is_ok());
        assert!(parse_and_validate_header(&encoded, cursor).is_err());
        header.get_mut(ATTN_O).unwrap()["shape"] = json!([2560, 2560]);
        assert!(parse_and_validate_header_with_specs(
            &serde_json::to_vec(&header).unwrap(),
            cursor,
            QWEN_TENSOR_SPECS
        )
        .is_err());
        let mut wrong = raw;
        wrong["num_attention_heads"] = json!(20);
        assert!(parse_and_validate_config(&serde_json::to_vec(&wrong).unwrap()).is_err());
    }

    #[test]
    #[ignore = "requires pinned Qwen checkpoint on mini2"]
    fn qwen_checkpoint_loads_actual_artifact() {
        let path = std::env::var("CAMELID_QWEN_EAGLE_CHECKPOINT").expect("checkpoint path");
        let model = Eagle3DraftModel::load(Path::new(&path)).unwrap();
        assert_eq!(model.config.geometry(), Eagle3Geometry::QWEN);
        assert_eq!(model.matrices.attention_q.shape, [4096, 5120]);
        assert_eq!(model.matrices.attention_o.shape, [2560, 4096]);
        assert_eq!(model.norms.hidden.len(), 2560);
        assert_eq!(model.draft_to_target.len(), 32000);
        assert!(model.draft_to_target.iter().any(|&id| id > 128256));
    }

    #[test]
    fn derived_checkpoint_is_rejected_without_exact_opt_in() {
        assert!(validate_derived_opt_in_value(None).is_err());
        assert!(validate_derived_opt_in_value(Some(std::ffi::OsStr::new("0"))).is_err());
        assert!(validate_derived_opt_in_value(Some(std::ffi::OsStr::new("true"))).is_err());
        validate_derived_opt_in_value(Some(std::ffi::OsStr::new("1"))).unwrap();
    }

    #[test]
    fn derived_receipt_accepts_standard_provenance_and_extra_audit_fields() {
        let weights = "aa".repeat(32);
        let config_hash = "bb".repeat(32);
        let mapping = "cc".repeat(32);
        let config = parse_and_validate_config(PINNED_CONFIG.as_bytes()).unwrap();
        let receipt = derived_receipt_json(&weights, &config_hash, &mapping);
        let provenance = validate_derived_receipt_fields(
            &receipt,
            "dd".repeat(32),
            &weights,
            &config_hash,
            &mapping,
            &config,
        )
        .unwrap();
        assert_eq!(provenance.schema, DERIVED_TRAINING_RECEIPT_SCHEMA);
        assert_eq!(
            provenance.source_weights_sha256,
            THOUGHTWORKS_WEIGHTS_SHA256
        );
        assert_eq!(provenance.tensor_count, EXPECTED_TENSOR_COUNT);
    }

    #[test]
    fn derived_receipt_rejects_hash_and_contract_tampering() {
        let weights = "aa".repeat(32);
        let config_hash = "bb".repeat(32);
        let mapping = "cc".repeat(32);
        let config = parse_and_validate_config(PINNED_CONFIG.as_bytes()).unwrap();

        let wrong_actual = validate_derived_receipt_fields(
            &derived_receipt_json(&weights, &config_hash, &mapping),
            "dd".repeat(32),
            &"ee".repeat(32),
            &config_hash,
            &mapping,
            &config,
        )
        .unwrap_err();
        assert!(wrong_actual.to_string().contains("output_weights_sha256"));

        let mut changed_mapping: Value =
            serde_json::from_slice(&derived_receipt_json(&weights, &config_hash, &mapping))
                .unwrap();
        changed_mapping["source_mapping_sha256"] = Value::String("ee".repeat(32));
        let changed_mapping = serde_json::to_vec(&changed_mapping).unwrap();
        let error = validate_derived_receipt_fields(
            &changed_mapping,
            "dd".repeat(32),
            &weights,
            &config_hash,
            &mapping,
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("mapping contract"));

        let mut wrong_count: Value =
            serde_json::from_slice(&derived_receipt_json(&weights, &config_hash, &mapping))
                .unwrap();
        wrong_count["tensor_count"] = json!(14);
        let error = validate_derived_receipt_fields(
            &serde_json::to_vec(&wrong_count).unwrap(),
            "dd".repeat(32),
            &weights,
            &config_hash,
            &mapping,
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("tensor_count"));
    }

    #[test]
    fn pinned_config_is_exact_and_exposes_target_taps() {
        let config = parse_and_validate_config(PINNED_CONFIG.as_bytes()).unwrap();
        assert_eq!(config.hidden_size, HIDDEN_SIZE);
        assert_eq!(config.draft_vocab_size, DRAFT_VOCAB_SIZE);
        assert_eq!(config.rope_theta, ROPE_THETA);
        assert_eq!(config.sliding_window, None);
        assert_eq!(TARGET_LAYER_INPUT_IDS, [2, 14, 25]);

        let wrong_width = PINNED_CONFIG.replace("\"hidden_size\": 3072", "\"hidden_size\": 4096");
        let error = parse_and_validate_config(wrong_width.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("hidden_size"));

        let with_unknown = PINNED_CONFIG.replace(
            "\"tie_word_embeddings\": false",
            "\"tie_word_embeddings\": false, \"max_position_embeddings\": 2048",
        );
        let error = parse_and_validate_config(with_unknown.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("extra fields"));
    }

    #[test]
    fn sharegpt_e9_and_e8_configs_pin_the_attention_window() {
        let e9 = parse_and_validate_config(SHAREGPT_CONFIG.as_bytes()).unwrap();
        assert_eq!(e9.architectures, ["LlamaForCausalLMEagle3"]);
        assert_eq!(e9.rope_theta, SHAREGPT_ROPE_THETA);
        assert_eq!(e9.torch_dtype, "bfloat16");
        assert_eq!(e9.sliding_window, Some(256));

        let sw512 = SHAREGPT_CONFIG.replace("\"sliding_window\": 256", "\"sliding_window\": 512");
        let sw512 = parse_and_validate_config(sw512.as_bytes()).unwrap();
        assert_eq!(sw512.sliding_window, Some(512));

        let e8 = SHAREGPT_CONFIG
            .replace("        \"sliding_window\": 256,\n", "")
            .replace("        \"use_sliding_window\": true\n", "")
            .replace(
                "        \"use_cache\": true,\n",
                "        \"use_cache\": true\n",
            );
        let config = parse_and_validate_config(e8.as_bytes()).unwrap();
        assert_eq!(config.architectures, ["LlamaForCausalLMEagle3"]);
        assert_eq!(config.rope_theta, SHAREGPT_ROPE_THETA);
        assert_eq!(config.torch_dtype, "bfloat16");
        assert_eq!(config.sliding_window, None);

        let wrong_window =
            SHAREGPT_CONFIG.replace("\"sliding_window\": 256", "\"sliding_window\": 255");
        let error = parse_and_validate_config(wrong_window.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("sliding-window"));

        let disabled_window = SHAREGPT_CONFIG.replace(
            "\"use_sliding_window\": true",
            "\"use_sliding_window\": false",
        );
        let error = parse_and_validate_config(disabled_window.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("sliding-window"));

        let missing_window = SHAREGPT_CONFIG.replace("        \"sliding_window\": 256,\n", "");
        let error = parse_and_validate_config(missing_window.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("sliding-window"));

        let missing_enable = SHAREGPT_CONFIG.replace(
            "        \"use_cache\": true,\n        \"use_sliding_window\": true\n",
            "        \"use_cache\": true\n",
        );
        let error = parse_and_validate_config(missing_enable.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("sliding-window"));
    }

    #[test]
    fn header_pins_all_fifteen_tensors_and_published_payload_size() {
        let (header, payload_bytes) = pinned_header();
        // 486,297,280-byte published file minus its 1,472-byte prefix.
        assert_eq!(payload_bytes, 486_295_808);
        let encoded = serde_json::to_vec(&Value::Object(header.clone())).unwrap();
        let descriptors = parse_and_validate_header(&encoded, payload_bytes).unwrap();
        assert_eq!(descriptors.len(), EXPECTED_TENSOR_COUNT);
        assert_eq!(descriptors[FC].len(), 56_623_104);
        assert_eq!(descriptors[LM_HEAD].len(), 196_608_000);

        let mut wrong_shape = header;
        wrong_shape
            .get_mut(ATTN_Q)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("shape".into(), json!([4096, 8192]));
        let encoded = serde_json::to_vec(&Value::Object(wrong_shape)).unwrap();
        let error = parse_and_validate_header(&encoded, payload_bytes).unwrap_err();
        assert!(error.to_string().contains(ATTN_Q));
        assert!(error.to_string().contains("shape"));
    }

    #[test]
    fn header_admits_checked_i64_d2t_layout() {
        let (mut header, payload_bytes) = pinned_header();
        let extra = DRAFT_VOCAB_SIZE as u64 * 4;
        for (name, value) in &mut header {
            let descriptor = value.as_object_mut().unwrap();
            if name == D2T {
                descriptor.insert("dtype".into(), json!("I64"));
            }
            let offsets = descriptor
                .get_mut("data_offsets")
                .unwrap()
                .as_array_mut()
                .unwrap();
            let start = offsets[0].as_u64().unwrap();
            let end = offsets[1].as_u64().unwrap();
            if name == D2T {
                offsets[1] = json!(end + extra);
            } else {
                offsets[0] = json!(start + extra);
                offsets[1] = json!(end + extra);
            }
        }
        let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        let descriptors = parse_and_validate_header(&encoded, payload_bytes + extra).unwrap();
        assert_eq!(descriptors[D2T].len(), DRAFT_VOCAB_SIZE as u64 * 8);
    }

    #[test]
    fn header_rejects_missing_extra_and_unaccounted_payload_bytes() {
        let (mut header, payload_bytes) = pinned_header();
        header.remove(T2D);
        header.insert(
            "unexpected.weight".into(),
            json!({"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}),
        );
        let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        let error = parse_and_validate_header(&encoded, payload_bytes).unwrap_err();
        let message = error.to_string();
        assert!(message.contains(T2D));
        assert!(message.contains("unexpected.weight"));

        let (header, payload_bytes) = pinned_header();
        let encoded = serde_json::to_vec(&Value::Object(header)).unwrap();
        let error = parse_and_validate_header(&encoded, payload_bytes + 1).unwrap_err();
        assert!(error.to_string().contains("file contains"));
    }

    #[test]
    fn d2t_is_delta_decoded_then_range_and_uniqueness_checked() {
        let (offsets, decoded) = decode_d2t(&i32_bytes(&[0, 0, 1]), 3, 5).unwrap();
        assert_eq!(offsets, [0, 0, 1]);
        assert_eq!(decoded, [0, 1, 3]);

        let duplicate = decode_d2t(&i32_bytes(&[1, 0]), 2, 4).unwrap_err();
        assert!(duplicate.to_string().contains("more than one draft row"));

        let out_of_range = decode_d2t(&i32_bytes(&[0, 4]), 2, 5).unwrap_err();
        assert!(out_of_range.to_string().contains("outside"));

        let negative = decode_d2t(&i32_bytes(&[-1]), 1, 5).unwrap_err();
        assert!(negative.to_string().contains("outside"));

        let (offsets, decoded) = decode_d2t(&i64_bytes(&[0, 0, 1]), 3, 5).unwrap();
        assert_eq!(offsets, [0, 0, 1]);
        assert_eq!(decoded, [0, 1, 3]);

        let too_wide =
            decode_d2t(&i64_bytes(&[i64::from(i32::MAX) + 1]), 1, usize::MAX).unwrap_err();
        assert!(too_wide.to_string().contains("I32"));
    }

    #[test]
    fn t2d_must_be_the_exact_membership_mask_for_absolute_d2t() {
        let mask = decode_and_validate_t2d(&[1, 1, 0, 1, 0], 5, &[0, 1, 3]).unwrap();
        assert_eq!(mask, [true, true, false, true, false]);

        let missing = decode_and_validate_t2d(&[1, 0, 1, 1, 0], 5, &[0, 1, 3]).unwrap_err();
        assert!(missing.to_string().contains("does not mark"));

        let non_bool = decode_and_validate_t2d(&[1, 2, 0], 3, &[0]).unwrap_err();
        assert!(non_bool.to_string().contains("non-boolean"));
    }

    #[test]
    fn bf16_norm_decode_preserves_little_endian_values() {
        let decoded = decode_bf16(&[0x80, 0x3f, 0x00, 0xc0, 0x00, 0x00]).unwrap();
        assert_eq!(decoded, [1.0, -2.0, 0.0]);
        assert!(decode_bf16(&[0]).is_err());
    }
}
