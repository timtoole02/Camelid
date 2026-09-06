//! Strict loader for the Llama 3.2 3B Instruct EAGLE-3 draft head.
//!
//! This module deliberately admits one pinned artifact contract: the 15-tensor
//! `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` SafeTensors checkpoint.  It does
//! not implement drafting.  Keeping loading separate makes the first runtime
//! slice fail closed on the two mistakes that most severely damage acceptance:
//! silently accepting a head for a different target model, and interpreting the
//! checkpoint's delta-coded `d2t` values as absolute target token ids.

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
pub const RMS_NORM_EPS: f32 = 1.0e-5;
const CONFIG_RMS_NORM_EPS: f64 = 1.0e-5;

/// Target layer-input taps used when an EAGLE-3 checkpoint omits explicit tap ids.
/// Upstream derives `[2, n_layers / 2, n_layers - 3]`; Llama 3.2 3B has 28 layers.
pub const TARGET_LAYER_INPUT_IDS: [usize; 3] = [2, 14, 25];

const CONFIG_FILE: &str = "config.json";
const WEIGHTS_FILE: &str = "model.safetensors";
const EXPECTED_TENSOR_COUNT: usize = 15;
const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;

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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
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
    torch_dtype: String,
    tie_word_embeddings: bool,
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

/// The four 3072-wide RMSNorm vectors, decoded once to f32 for Metal uniforms.
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
    /// Load the exact 15-tensor Thoughtworks Llama 3.2 3B Instruct EAGLE-3 head.
    pub fn load(dir: &Path) -> Result<Self> {
        let config_path = dir.join(CONFIG_FILE);
        let weights_path = dir.join(WEIGHTS_FILE);
        let config = load_config(&config_path)?;

        let (mut file, payload_start, descriptors) = open_weights(&weights_path)?;

        let matrices = Eagle3Matrices {
            feature_fusion: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                FC,
                [HIDDEN_SIZE, 3 * HIDDEN_SIZE],
            )?,
            lm_head: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                LM_HEAD,
                [DRAFT_VOCAB_SIZE, HIDDEN_SIZE],
            )?,
            mlp_down: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_DOWN,
                [HIDDEN_SIZE, INTERMEDIATE_SIZE],
            )?,
            mlp_gate: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_GATE,
                [INTERMEDIATE_SIZE, HIDDEN_SIZE],
            )?,
            mlp_up: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                MLP_UP,
                [INTERMEDIATE_SIZE, HIDDEN_SIZE],
            )?,
            attention_k: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_K,
                [NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE],
            )?,
            attention_o: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_O,
                [HIDDEN_SIZE, HIDDEN_SIZE],
            )?,
            attention_q: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_Q,
                [NUM_ATTENTION_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE],
            )?,
            attention_v: load_matrix(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                ATTN_V,
                [NUM_KEY_VALUE_HEADS * HEAD_DIM, 2 * HIDDEN_SIZE],
            )?,
        };

        let norms = Eagle3Norms {
            hidden: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                HIDDEN_NORM,
            )?,
            input: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                INPUT_NORM,
            )?,
            post_attention: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                POST_ATTN_NORM,
            )?,
            output: load_norm(
                &mut file,
                &weights_path,
                payload_start,
                &descriptors,
                OUTPUT_NORM,
            )?,
        };

        let raw_d2t = read_tensor(
            &mut file,
            &weights_path,
            payload_start,
            descriptor(&descriptors, D2T)?,
        )?;
        let (d2t_offsets, draft_to_target) =
            decode_d2t(&raw_d2t, DRAFT_VOCAB_SIZE, TARGET_VOCAB_SIZE)?;

        let raw_t2d = read_tensor(
            &mut file,
            &weights_path,
            payload_start,
            descriptor(&descriptors, T2D)?,
        )?;
        let target_to_draft_mask =
            decode_and_validate_t2d(&raw_t2d, TARGET_VOCAB_SIZE, &draft_to_target)?;

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

    require_equal(
        "architectures",
        &raw.architectures,
        &vec!["LlamaForCausalLM".to_string()],
    )?;
    require_equal("model_type", &raw.model_type, &"llama".to_string())?;
    require_equal("hidden_size", &raw.hidden_size, &HIDDEN_SIZE)?;
    require_equal(
        "intermediate_size",
        &raw.intermediate_size,
        &INTERMEDIATE_SIZE,
    )?;
    require_equal(
        "num_hidden_layers",
        &raw.num_hidden_layers,
        &NUM_HIDDEN_LAYERS,
    )?;
    require_equal(
        "num_attention_heads",
        &raw.num_attention_heads,
        &NUM_ATTENTION_HEADS,
    )?;
    require_equal(
        "num_key_value_heads",
        &raw.num_key_value_heads,
        &NUM_KEY_VALUE_HEADS,
    )?;
    require_equal("head_dim", &raw.head_dim, &HEAD_DIM)?;
    require_equal("vocab_size", &raw.vocab_size, &TARGET_VOCAB_SIZE)?;
    require_equal("draft_vocab_size", &raw.draft_vocab_size, &DRAFT_VOCAB_SIZE)?;
    require_equal("rope_theta", &raw.rope_theta, &(ROPE_THETA as f64))?;
    require_equal("rms_norm_eps", &raw.rms_norm_eps, &CONFIG_RMS_NORM_EPS)?;
    require_equal("torch_dtype", &raw.torch_dtype, &"bfloat16".to_string())?;
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
        torch_dtype: raw.torch_dtype,
        tie_word_embeddings: raw.tie_word_embeddings,
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
        _ => None,
    }
}

fn tensor_bytes(spec: &TensorSpec) -> Result<u64> {
    let elements = spec.shape.iter().try_fold(1u64, |acc, &dimension| {
        acc.checked_mul(dimension).ok_or_else(|| {
            invalid(format!(
                "EAGLE-3 tensor {} element count overflows",
                spec.name
            ))
        })
    })?;
    elements
        .checked_mul(dtype_bytes(spec.dtype).expect("all pinned dtypes have a width"))
        .ok_or_else(|| invalid(format!("EAGLE-3 tensor {} byte count overflows", spec.name)))
}

fn parse_and_validate_header(
    header_bytes: &[u8],
    payload_bytes: u64,
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
    let expected_names: BTreeSet<&str> = TENSOR_SPECS.iter().map(|spec| spec.name).collect();
    if actual_names != expected_names {
        let missing: Vec<&str> = expected_names.difference(&actual_names).copied().collect();
        let extra: Vec<&str> = actual_names.difference(&expected_names).copied().collect();
        return Err(invalid(format!(
            "EAGLE-3 SafeTensors tensor set differs from the pinned {EXPECTED_TENSOR_COUNT}-tensor contract; missing={missing:?}, extra={extra:?}"
        )));
    }

    let mut descriptors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(EXPECTED_TENSOR_COUNT);
    for spec in TENSOR_SPECS {
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
        require_equal(
            &format!("tensor {} dtype", spec.name),
            &tensor.dtype,
            &spec.dtype.to_string(),
        )?;
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
        let expected_bytes = tensor_bytes(spec)?;
        if end - start != expected_bytes {
            return Err(invalid(format!(
                "EAGLE-3 tensor {} occupies {} bytes, expected {expected_bytes}",
                spec.name,
                end - start
            )));
        }
        let descriptor = TensorDescriptor { start, end };
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

fn open_weights(path: &Path) -> Result<(File, u64, BTreeMap<&'static str, TensorDescriptor>)> {
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
    let descriptors = parse_and_validate_header(&header, payload_bytes)?;
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
) -> Result<Vec<f32>> {
    let bytes = read_tensor(file, path, payload_start, descriptor(descriptors, name)?)?;
    let values = decode_bf16(&bytes)?;
    if values.len() != HIDDEN_SIZE {
        return Err(invalid(format!(
            "EAGLE-3 norm {name} decoded to {} values, expected {HIDDEN_SIZE}",
            values.len()
        )));
    }
    Ok(values)
}

/// The source checkpoint stores a monotone delta from each draft row's index.
/// llama.cpp's converter uses the same `raw[i] + i` reconstruction before runtime.
fn decode_d2t(
    bytes: &[u8],
    draft_vocab: usize,
    target_vocab: usize,
) -> Result<(Vec<i32>, Vec<u32>)> {
    let expected_bytes = draft_vocab
        .checked_mul(4)
        .ok_or_else(|| invalid("EAGLE-3 d2t byte count overflows"))?;
    if bytes.len() != expected_bytes {
        return Err(invalid(format!(
            "EAGLE-3 d2t contains {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }

    let mut seen = vec![false; target_vocab];
    let mut offsets = Vec::with_capacity(draft_vocab);
    let mut absolute = Vec::with_capacity(draft_vocab);
    for (index, encoded) in bytes.chunks_exact(4).enumerate() {
        let delta = i32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        let token = i64::from(delta)
            .checked_add(index as i64)
            .ok_or_else(|| invalid(format!("EAGLE-3 d2t row {index} overflows")))?;
        if token < 0 || token >= target_vocab as i64 {
            return Err(invalid(format!(
                "EAGLE-3 d2t row {index} resolves to target token {token}, outside 0..{target_vocab}"
            )));
        }
        let token = token as usize;
        if seen[token] {
            return Err(invalid(format!(
                "EAGLE-3 d2t resolves more than one draft row to target token {token}"
            )));
        }
        seen[token] = true;
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

    #[test]
    fn pinned_config_is_exact_and_exposes_target_taps() {
        let config = parse_and_validate_config(PINNED_CONFIG.as_bytes()).unwrap();
        assert_eq!(config.hidden_size, HIDDEN_SIZE);
        assert_eq!(config.draft_vocab_size, DRAFT_VOCAB_SIZE);
        assert_eq!(config.rope_theta, ROPE_THETA);
        assert_eq!(TARGET_LAYER_INPUT_IDS, [2, 14, 25]);

        let wrong_width = PINNED_CONFIG.replace("\"hidden_size\": 3072", "\"hidden_size\": 4096");
        let error = parse_and_validate_config(wrong_width.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("hidden_size"));

        let with_unknown = PINNED_CONFIG.replace(
            "\"tie_word_embeddings\": false",
            "\"tie_word_embeddings\": false, \"max_position_embeddings\": 2048",
        );
        let error = parse_and_validate_config(with_unknown.as_bytes()).unwrap_err();
        assert!(error.to_string().contains("unknown field"));
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
