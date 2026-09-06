//! Exact, isolated Metal foundation for the official Gemma 4 12B QAT assistant.
//!
//! This deliberately does not enter the Gemma target/verifier runtime.  It
//! admits one byte-identical assistant artifact, packs all 23 BF16 matrices to
//! canonical GGML Q4_0, releases the 846 MB source mapping, and preserves K=1
//! CPU/device parity oracles beside a scoped one-command K<=15 device
//! chain.  Target verification, acceptance, and rollback remain outside this
//! module, so none of these assistant paths changes target-authoritative output.
//! `CAMELID_GEMMA4_MTP12_DENSE_BF16=1` additionally retains the original BF16
//! dense matrices and uses them for assistant projections; its tied head stays Q4.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    path::{Path, PathBuf},
    time::Instant,
};

use metal::{
    Buffer, BufferRef, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{wire_mmap::GgufWireMmap, BackendError, Result};

#[path = "gemma4_mtp12_shortlist.rs"]
mod shortlist;
#[path = "gemma4_mtp12_bf16.rs"]
mod dense_bf16;
#[path = "gemma4_mtp12_tree.rs"]
mod tree;
pub use tree::Gemma4Mtp12TreeProposal;

pub const GEMMA4_12B_MTP_ASSISTANT_SHA256: &str =
    "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6";
const OFFICIAL_CONFIG_SHA256: &str =
    "7638c1d42f9fa73fe44b1a10604766b928b8985263f15775cd1a286a5a12799c";
const OFFICIAL_STAGED_ASSISTANT_PATH: &str =
    "gemma-4-12B-it-qat-q4_0-unquantized-assistant/model.safetensors";

const EXPECTED_FILE_BYTES: u64 = 845_719_296;
const EXPECTED_HEADER_BYTES: usize = 5_360;
const PAYLOAD_FILE_OFFSET: usize = 8 + EXPECTED_HEADER_BYTES;
const EXPECTED_PAYLOAD_BYTES: usize = 845_713_928;

const TARGET_HIDDEN: usize = 3_840;
const ASSISTANT_HIDDEN: usize = 1_024;
const FFN_HIDDEN: usize = 8_192;
const VOCAB: usize = 262_144;
const N_LAYERS: usize = 4;
const N_HEADS: usize = 16;
const LOCAL_HEAD_DIM: usize = 256;
const LOCAL_KV_HEADS: usize = 8;
const LOCAL_WINDOW: usize = 1_024;
const FULL_HEAD_DIM: usize = 512;
const FULL_KV_HEADS: usize = 1;
const RMS_EPS: f32 = 1.0e-6;
const Q4_0_BLOCK_VALUES: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q6_K_BLOCK_VALUES: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const FULL_Q4_MATRIX_BYTES: u64 = 237_846_528;
/// Maximum official-assistant draft chain paired with the W16 target verifier.
pub const GEMMA4_12B_MTP_MAX_DRAFTS: usize = 15;
const MTP12_CHAIN_MAX_DRAFTS: usize = GEMMA4_12B_MTP_MAX_DRAFTS;
const MTP12_CHAIN_TOKEN_SLOTS: usize = MTP12_CHAIN_MAX_DRAFTS + 1;
const MTP12_CHAIN_RECURRENT_ELEMENTS: usize = MTP12_CHAIN_MAX_DRAFTS * TARGET_HIDDEN;
const MTP12_CHAIN_LOCAL_ROPE_ELEMENTS: usize =
    MTP12_CHAIN_MAX_DRAFTS * LOCAL_HEAD_DIM / 2;
const MTP12_CHAIN_FULL_ROPE_ELEMENTS: usize = MTP12_CHAIN_MAX_DRAFTS * FULL_HEAD_DIM / 2;

fn mtp12_chain_draft_k_admitted(draft_k: usize) -> bool {
    (1..=MTP12_CHAIN_MAX_DRAFTS).contains(&draft_k)
}

/// Exact SHA-256 of the only target artifact admitted by the 12B resident
/// draft-chain API.  The assistant loader independently verifies its own
/// artifact SHA before constructing [`Gemma4Mtp12AssistantMetal`].
pub const GEMMA4_12B_QAT_Q4_0_TARGET_SHA256: &str =
    "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b";

/// Exact wire length of one selected row from the target's tied Q6_K token
/// embedding.  Device-fed K=1 accepts this scoped row instead of retaining or
/// copying the full 825,753,600-byte table in the assistant.
pub const GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES: u64 =
    (TARGET_HIDDEN / Q6_K_BLOCK_VALUES * Q6_K_BLOCK_BYTES) as u64;

/// Exact byte span of the target's complete tied `[262144, 3840]` Q6_K
/// embedding/head table.  The resident chain aliases this caller-owned table;
/// it never stages a second 825,753,600-byte copy.
pub const GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES: u64 =
    GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES * VOCAB as u64;

/// The official 12B target has 48 layers in a 5-local/1-global schedule.
/// Its assistant borrows the final local/global pair, not the 26B pair 28/29.
pub const GEMMA4_12B_MTP_SLIDING_HOST_LAYER: usize = 46;
pub const GEMMA4_12B_MTP_FULL_HOST_LAYER: usize = 47;

/// Borrowed byte range inside a caller-owned Metal buffer.  The assistant
/// never retains the reference beyond one synchronous K=1 call.
#[derive(Clone, Copy)]
pub struct Gemma4Mtp12MetalBufferView<'a> {
    pub buffer: &'a BufferRef,
    pub byte_offset: u64,
    pub byte_len: u64,
}

impl std::fmt::Debug for Gemma4Mtp12MetalBufferView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Gemma4Mtp12MetalBufferView")
            .field("device_registry_id", &self.buffer.device().registry_id())
            .field("buffer_len", &self.buffer.length())
            .field("byte_offset", &self.byte_offset)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

/// One already-selected target token-embedding row in canonical GGML Q6_K
/// wire order.  `hidden` is explicit so a 26B row cannot be mislabeled as 12B
/// even if a larger backing buffer happens to cover the requested byte range.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Mtp12Q6KEmbeddingRow<'a> {
    pub wire: Gemma4Mtp12MetalBufferView<'a>,
    pub hidden: usize,
}

/// Complete target tied embedding/head table admitted by the device-resident
/// draft chain.  The explicit identity and geometry fields make accidental
/// use of the 26B table fail closed before command encoding.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Mtp12Q6KEmbeddingTable<'a> {
    pub wire: Gemma4Mtp12MetalBufferView<'a>,
    pub hidden: usize,
    pub vocab: usize,
    pub target_model_sha256: &'a str,
}

/// One normalized target hidden row already resident on the assistant's Metal
/// device.  The chain BF16-rounds it at the same boundary as repeated K=1 and
/// keeps every later recurrent row on device.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Mtp12DeviceRecurrentHidden<'a> {
    pub values: Gemma4Mtp12MetalBufferView<'a>,
    pub hidden: usize,
}

/// Exact target shared-KV source view.  Key and value are native f32 in
/// `[kv_head][max_positions][head_dim]` order.  `source_layer` must be 46 for
/// the sliding source or 47 for the full source.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Mtp12DeviceKv<'a> {
    pub key: Gemma4Mtp12MetalBufferView<'a>,
    pub value: Gemma4Mtp12MetalBufferView<'a>,
    pub source_layer: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_positions: usize,
}

fn invalid(detail: impl Into<String>) -> BackendError {
    BackendError::InvalidTensorData(format!("Gemma 4 12B MTP assistant: {}", detail.into()))
}

fn official_staged_assistant_path() -> PathBuf {
    let model_root = std::env::var_os("CAMELID_MODEL_ROOT")
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .filter(|home| !home.is_empty())
                .or_else(|| std::env::var_os("USERPROFILE"))
                .filter(|home| !home.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("models")
        });
    model_root.join(OFFICIAL_STAGED_ASSISTANT_PATH)
}

/// Validate/derive the only target source-layer pair admitted by this module.
/// This makes 46/47 a checked consequence of the official 48-layer schedule,
/// rather than an unchecked port of the 26B assistant's 28/29 constants.
pub fn validate_gemma4_12b_shared_kv_schedule(layer_types: &[&str]) -> Result<(usize, usize)> {
    if layer_types.len() != 48 {
        return Err(invalid(format!(
            "target has {} layer types, expected 48",
            layer_types.len()
        )));
    }
    for (index, layer_type) in layer_types.iter().enumerate() {
        let expected = if index % 6 == 5 {
            "full_attention"
        } else {
            "sliding_attention"
        };
        if *layer_type != expected {
            return Err(invalid(format!(
                "target layer {index} is {layer_type:?}, expected {expected:?}"
            )));
        }
    }
    let sliding = layer_types
        .iter()
        .rposition(|kind| *kind == "sliding_attention")
        .ok_or_else(|| invalid("target has no sliding-attention layer"))?;
    let full = layer_types
        .iter()
        .rposition(|kind| *kind == "full_attention")
        .ok_or_else(|| invalid("target has no full-attention layer"))?;
    if (sliding, full)
        != (
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            GEMMA4_12B_MTP_FULL_HOST_LAYER,
        )
    {
        return Err(invalid(format!(
            "derived target shared-KV pair is {sliding}/{full}, expected 46/47"
        )));
    }
    Ok((sliding, full))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpectedTensor {
    name: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

fn push_expected(
    output: &mut Vec<ExpectedTensor>,
    cursor: &mut u64,
    name: impl Into<String>,
    shape: &[u64],
) {
    let start = *cursor;
    let bytes = shape
        .iter()
        .fold(2u64, |bytes, dimension| bytes * dimension);
    *cursor += bytes;
    output.push(ExpectedTensor {
        name: name.into(),
        shape: shape.to_vec(),
        start,
        end: *cursor,
    });
}

/// Payload order emitted by the exact Google safetensors artifact.  Offsets
/// are derived from the BF16 shapes and are still compared field-for-field
/// against the header; the final cursor is pinned to 845,713,928 bytes.
fn expected_tensors() -> Vec<ExpectedTensor> {
    let mut output = Vec::with_capacity(48);
    let mut cursor = 0u64;
    push_expected(
        &mut output,
        &mut cursor,
        "model.embed_tokens.weight",
        &[VOCAB as u64, ASSISTANT_HIDDEN as u64],
    );
    for layer in 0..N_LAYERS {
        let prefix = format!("model.layers.{layer}");
        let q_width = if layer < 3 {
            N_HEADS * LOCAL_HEAD_DIM
        } else {
            N_HEADS * FULL_HEAD_DIM
        };
        let q_norm = if layer < 3 {
            LOCAL_HEAD_DIM
        } else {
            FULL_HEAD_DIM
        };
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![ASSISTANT_HIDDEN as u64]),
            ("layer_scalar", vec![1]),
            (
                "mlp.down_proj.weight",
                vec![ASSISTANT_HIDDEN as u64, FFN_HIDDEN as u64],
            ),
            (
                "mlp.gate_proj.weight",
                vec![FFN_HIDDEN as u64, ASSISTANT_HIDDEN as u64],
            ),
            (
                "mlp.up_proj.weight",
                vec![FFN_HIDDEN as u64, ASSISTANT_HIDDEN as u64],
            ),
            (
                "post_attention_layernorm.weight",
                vec![ASSISTANT_HIDDEN as u64],
            ),
            (
                "post_feedforward_layernorm.weight",
                vec![ASSISTANT_HIDDEN as u64],
            ),
            (
                "pre_feedforward_layernorm.weight",
                vec![ASSISTANT_HIDDEN as u64],
            ),
            (
                "self_attn.o_proj.weight",
                vec![ASSISTANT_HIDDEN as u64, q_width as u64],
            ),
            ("self_attn.q_norm.weight", vec![q_norm as u64]),
            (
                "self_attn.q_proj.weight",
                vec![q_width as u64, ASSISTANT_HIDDEN as u64],
            ),
        ] {
            push_expected(
                &mut output,
                &mut cursor,
                format!("{prefix}.{suffix}"),
                &shape,
            );
        }
    }
    push_expected(
        &mut output,
        &mut cursor,
        "model.norm.weight",
        &[ASSISTANT_HIDDEN as u64],
    );
    push_expected(
        &mut output,
        &mut cursor,
        "post_projection.weight",
        &[TARGET_HIDDEN as u64, ASSISTANT_HIDDEN as u64],
    );
    push_expected(
        &mut output,
        &mut cursor,
        "pre_projection.weight",
        &[ASSISTANT_HIDDEN as u64, (2 * TARGET_HIDDEN) as u64],
    );
    debug_assert_eq!(cursor, EXPECTED_PAYLOAD_BYTES as u64);
    output
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorRef {
    absolute_offset: u64,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Q4TensorRef {
    byte_offset: u64,
    byte_len: u64,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Q4LayerWeights {
    q: Q4TensorRef,
    o: Q4TensorRef,
    gate: Q4TensorRef,
    up: Q4TensorRef,
    down: Q4TensorRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Q4Layout {
    embedding: Q4TensorRef,
    layers: [Q4LayerWeights; N_LAYERS],
    pre_projection: Q4TensorRef,
    post_projection: Q4TensorRef,
    matrix_bytes: u64,
}

#[derive(Clone, Debug)]
struct TensorEntry {
    shape: Vec<u64>,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct AssistantManifest {
    tensors: BTreeMap<String, TensorEntry>,
}

#[derive(Deserialize)]
struct SafetensorsEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl AssistantManifest {
    fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors.get(name).ok_or_else(|| {
            BackendError::TensorNotFound(format!("official Gemma 4 12B MTP tensor {name}"))
        })
    }
    fn matrix(&self, name: &str) -> Result<TensorRef> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 2 {
            return Err(invalid(format!(
                "matrix {name} has rank {}, expected 2",
                tensor.shape.len()
            )));
        }
        Ok(TensorRef {
            absolute_offset: (PAYLOAD_FILE_OFFSET + tensor.start) as u64,
            rows: u32::try_from(tensor.shape[0])
                .map_err(|_| invalid(format!("matrix {name} row count exceeds u32")))?,
            cols: u32::try_from(tensor.shape[1])
                .map_err(|_| invalid(format!("matrix {name} column count exceeds u32")))?,
        })
    }
}

fn parse_and_validate_header(header: &[u8], payload_bytes: usize) -> Result<AssistantManifest> {
    let root: serde_json::Value = serde_json::from_slice(header)
        .map_err(|error| invalid(format!("invalid safetensors header JSON: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| invalid("safetensors header root is not an object"))?;
    let metadata = object
        .get("__metadata__")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid("missing safetensors __metadata__"))?;
    if metadata.len() != 1
        || metadata.get("format").and_then(serde_json::Value::as_str) != Some("pt")
    {
        return Err(invalid("safetensors metadata is not exactly format=pt"));
    }

    let expected = expected_tensors();
    if object.len() != expected.len() + 1 {
        return Err(invalid(format!(
            "tensor count {} does not match expected {}",
            object.len().saturating_sub(1),
            expected.len()
        )));
    }

    let mut tensors = BTreeMap::new();
    for expected in &expected {
        let value = object
            .get(&expected.name)
            .ok_or_else(|| invalid(format!("missing tensor {}", expected.name)))?;
        let actual: SafetensorsEntry = serde_json::from_value(value.clone()).map_err(|error| {
            invalid(format!("invalid descriptor for {}: {error}", expected.name))
        })?;
        if actual.dtype != "BF16"
            || actual.shape != expected.shape
            || actual.data_offsets != [expected.start, expected.end]
        {
            return Err(invalid(format!(
                "tensor {} mismatch: dtype={} shape={:?} offsets={:?}",
                expected.name, actual.dtype, actual.shape, actual.data_offsets
            )));
        }
        tensors.insert(
            expected.name.clone(),
            TensorEntry {
                shape: actual.shape,
                start: usize::try_from(expected.start)
                    .map_err(|_| invalid("tensor start exceeds usize"))?,
                end: usize::try_from(expected.end)
                    .map_err(|_| invalid("tensor end exceeds usize"))?,
            },
        );
    }
    if expected.first().map(|entry| entry.start) != Some(0)
        || expected.last().map(|entry| entry.end) != Some(payload_bytes as u64)
        || expected.windows(2).any(|pair| pair[0].end != pair[1].start)
    {
        return Err(invalid(
            "internal tensor table is not a contiguous full payload",
        ));
    }
    Ok(AssistantManifest { tensors })
}

fn parse_official_manifest(mapping: &GgufWireMmap) -> Result<AssistantManifest> {
    if mapping.file_len() != EXPECTED_FILE_BYTES {
        return Err(invalid(format!(
            "file size {} does not match expected {EXPECTED_FILE_BYTES}",
            mapping.file_len()
        )));
    }
    let prefix = mapping.bytes(0, 8)?;
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(prefix);
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes))
        .map_err(|_| invalid("header length exceeds usize"))?;
    if header_len != EXPECTED_HEADER_BYTES {
        return Err(invalid(format!(
            "header length {header_len} does not match expected {EXPECTED_HEADER_BYTES}"
        )));
    }
    parse_and_validate_header(mapping.bytes(8, header_len)?, EXPECTED_PAYLOAD_BYTES)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_official_config(weight_path: &Path) -> Result<()> {
    let config_path = weight_path
        .parent()
        .ok_or_else(|| invalid("weight path has no parent directory"))?
        .join("config.json");
    let bytes = std::fs::read(&config_path).map_err(|source| BackendError::Io {
        path: config_path.clone(),
        source,
    })?;
    let actual_hash = sha256_hex(&bytes);
    if actual_hash != OFFICIAL_CONFIG_SHA256 {
        return Err(invalid(format!(
            "config SHA-256 {actual_hash} does not match expected {OFFICIAL_CONFIG_SHA256}"
        )));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(format!("invalid config.json: {error}")))?;
    let text = value
        .get("text_config")
        .ok_or_else(|| invalid("assistant config lacks text_config"))?;
    for (name, actual, expected) in [
        (
            "backbone_hidden_size",
            value.get("backbone_hidden_size"),
            TARGET_HIDDEN,
        ),
        ("hidden_size", text.get("hidden_size"), ASSISTANT_HIDDEN),
        (
            "intermediate_size",
            text.get("intermediate_size"),
            FFN_HIDDEN,
        ),
        ("num_hidden_layers", text.get("num_hidden_layers"), N_LAYERS),
        (
            "num_attention_heads",
            text.get("num_attention_heads"),
            N_HEADS,
        ),
        (
            "num_key_value_heads",
            text.get("num_key_value_heads"),
            LOCAL_KV_HEADS,
        ),
        (
            "num_global_key_value_heads",
            text.get("num_global_key_value_heads"),
            FULL_KV_HEADS,
        ),
        ("head_dim", text.get("head_dim"), LOCAL_HEAD_DIM),
        (
            "global_head_dim",
            text.get("global_head_dim"),
            FULL_HEAD_DIM,
        ),
        (
            "num_kv_shared_layers",
            text.get("num_kv_shared_layers"),
            N_LAYERS,
        ),
        ("sliding_window", text.get("sliding_window"), LOCAL_WINDOW),
        ("vocab_size", text.get("vocab_size"), VOCAB),
    ] {
        if actual.and_then(serde_json::Value::as_u64) != Some(expected as u64) {
            return Err(invalid(format!(
                "assistant config {name} is {actual:?}, expected {expected}"
            )));
        }
    }
    if value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| values.first())
        .and_then(serde_json::Value::as_str)
        != Some("Gemma4UnifiedAssistantForCausalLM")
        || value.get("model_type").and_then(serde_json::Value::as_str)
            != Some("gemma4_unified_assistant")
        || value.get("dtype").and_then(serde_json::Value::as_str) != Some("bfloat16")
        || value
            .get("use_ordered_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(false)
        || text
            .get("attention_k_eq_v")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || text
            .get("hidden_activation")
            .and_then(serde_json::Value::as_str)
            != Some("gelu_pytorch_tanh")
        || text
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        return Err(invalid(
            "assistant config semantic flags do not match the official 12B QAT assistant",
        ));
    }
    let actual_layers: Vec<&str> = text
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| invalid("assistant config lacks layer_types"))?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    if actual_layers
        != [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
    {
        return Err(invalid(format!(
            "assistant layer_types mismatch: {actual_layers:?}"
        )));
    }
    let rope = text
        .get("rope_parameters")
        .ok_or_else(|| invalid("assistant config lacks rope_parameters"))?;
    if rope
        .pointer("/sliding_attention/rope_theta")
        .and_then(serde_json::Value::as_f64)
        != Some(10_000.0)
        || rope
            .pointer("/sliding_attention/rope_type")
            .and_then(serde_json::Value::as_str)
            != Some("default")
        || rope
            .pointer("/full_attention/rope_theta")
            .and_then(serde_json::Value::as_f64)
            != Some(1_000_000.0)
        || rope
            .pointer("/full_attention/rope_type")
            .and_then(serde_json::Value::as_str)
            != Some("proportional")
        || rope
            .pointer("/full_attention/partial_rotary_factor")
            .and_then(serde_json::Value::as_f64)
            != Some(0.25)
        || text
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .map(f64::to_bits)
            != Some(1.0e-6f64.to_bits())
        || !text
            .get("final_logit_softcapping")
            .is_some_and(serde_json::Value::is_null)
    {
        return Err(invalid(
            "assistant norm/RoPE/logit configuration does not match the official artifact",
        ));
    }
    Ok(())
}

// Arithmetic is forked from the signed-scale production assistant at
// 95dcadd0.  In particular Q4_0 uses the canonical signed-max scale, Gemma 4
// norm is `normalized * weight` (not `1 + weight`), attention scaling is 1,
// and BF16 tensor boundaries are widened back into f32 scratch.
const MTP12_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float mtp12_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        uint upper = (bits >> 16) | 0x0040u;
        return as_type<float>(upper << 16);
    }
    uint bias = 0x00007fffu + ((bits >> 16) & 1u);
    return as_type<float>((bits + bias) & 0xffff0000u);
}

inline int mtp12_q6k_code(device const uchar* block, uint index) {
    const uint half_index = index >> 7;
    const uint position = index & 127u;
    const uint lane = position & 31u;
    const uint ql = half_index * 64u;
    const uint qh = 128u + half_index * 32u;
    if (position < 32u) {
        return int((block[ql + lane] & 0x0fu) | ((block[qh + lane] & 3u) << 4)) - 32;
    }
    if (position < 64u) {
        return int((block[ql + 32u + lane] & 0x0fu)
            | (((block[qh + lane] >> 2) & 3u) << 4)) - 32;
    }
    if (position < 96u) {
        return int((block[ql + lane] >> 4)
            | (((block[qh + lane] >> 4) & 3u) << 4)) - 32;
    }
    return int((block[ql + 32u + lane] >> 4)
        | (((block[qh + lane] >> 6) & 3u) << 4)) - 32;
}

// Dequantize one caller-selected row from the target's tied Q6_K table,
// multiply by sqrt(3840), and land directly in the BF16-rounded first half of
// the assistant pre-projection input. `row_byte_offset` scopes a larger
// no-copy table buffer without staging or retaining the 3,150-byte row.
kernel void mtp12_gather_q6k_embedding(
    device const uchar* table [[buffer(0)]],
    device float* pre_input [[buffer(1)]],
    constant ulong& row_byte_offset [[buffer(2)]],
    constant float& embedding_scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 3840u) return;
    device const uchar* block = table + row_byte_offset
        + ulong(gid / 256u) * 210ul;
    const float d = float(*reinterpret_cast<device const half*>(block + 208));
    const int group_scale = int(
        reinterpret_cast<device const char*>(block + 192)[(gid & 255u) >> 4]);
    // Match Q6KBlock::dequantize and mtp_scaled_embedding left-to-right:
    // ((d * group_scale) * q) * sqrt(hidden), then the CPU scaffold's BF16 RNE.
    const float dequantized = (d * float(group_scale))
        * float(mtp12_q6k_code(block, gid & 255u));
    pre_input[gid] = mtp12_round_bf16(dequantized * embedding_scale);
}

// Device-resident chain gather. `token_ids[token_index]` selects a row from
// the complete target table while the previous recurrent row is consumed from
// Metal.  This intentionally uses the same left-to-right Q6_K arithmetic and
// BF16 boundary as mtp12_gather_q6k_embedding above.
kernel void mtp12_gather_q6k_embedding_and_recurrent(
    device const uint* token_ids [[buffer(0)]],
    device const uchar* table [[buffer(1)]],
    device const float* recurrent_hidden [[buffer(2)]],
    device float* pre_input [[buffer(3)]],
    constant ulong& table_byte_offset [[buffer(4)]],
    constant uint& token_index [[buffer(5)]],
    constant uint& target_vocab [[buffer(6)]],
    constant float& embedding_scale [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 3840u) return;
    const uint token = token_ids[token_index];
    if (token >= target_vocab) {
        const float invalid = as_type<float>(0x7fc00000u);
        pre_input[gid] = invalid;
        pre_input[3840u + gid] = invalid;
        return;
    }
    const ulong row_byte_offset = table_byte_offset
        + ulong(token) * 3150ul;
    device const uchar* block = table + row_byte_offset
        + ulong(gid / 256u) * 210ul;
    const float d = float(*reinterpret_cast<device const half*>(block + 208));
    const int group_scale = int(
        reinterpret_cast<device const char*>(block + 192)[(gid & 255u) >> 4]);
    const float dequantized = (d * float(group_scale))
        * float(mtp12_q6k_code(block, gid & 255u));
    pre_input[gid] = mtp12_round_bf16(dequantized * embedding_scale);
    pre_input[3840u + gid] = mtp12_round_bf16(recurrent_hidden[gid]);
}

kernel void mtp12_copy_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = input[gid];
}

kernel void mtp12_add_bf16(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) output[gid] = mtp12_round_bf16(a[gid] + b[gid]);
}

kernel void mtp12_add_scale_bf16(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        const float sum = mtp12_round_bf16(a[gid] + b[gid]);
        output[gid] = mtp12_round_bf16(sum * scale);
    }
}

kernel void mtp12_q4_0_gemv(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32u;
    device const uchar* row_bytes = q4_weights + weight_byte_offset
        + ulong(row) * ulong(blocks_per_row) * 18ul;
    float partial = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
        device const uchar* block = row_bytes + ulong(block_index) * 18ul;
        const float d = float(*reinterpret_cast<device const half*>(block));
        device const packed_uchar4* q4 =
            reinterpret_cast<device const packed_uchar4*>(block + 2);
        const uint input_base = block_index * 32u;
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {
            const uchar4 packed = uchar4(q4[k]);
            const uint offset = k * 4u;
            partial += d * (
                float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
              + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
              + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
              + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
              + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
              + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
              + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
              + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
        }
    }
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 =
        simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 =
        simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    const float value = pair01 + pair23;
    if (lane == 0u) {
        output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
    }
}

// ---- Draft-side fused/regeometried kernels (opt-in, default off) ---------------
// Every kernel below is an order-preserving re-arrangement of the established
// kernels above: the per-element expressions, the sequential accumulation
// statements and the reduction/fold orders are copied verbatim, so the drafts
// they produce are bit-identical.  Each is selected independently by one
// CAMELID_GEMMA4_MTP12_FUSE_* variable (unset = established path).

// Four rows per 128-thread threadgroup, one simdgroup per row
// (CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4=1).  Each simdgroup runs exactly the
// per-row program of mtp12_q4_0_gemv: the same lane->block map, the same
// sequential `partial +=` statement, the same shuffle fold and lane-0
// rounding.  The only addition is a register prefetch of the lane's next
// block, guarded so it never reads past the row; it changes load scheduling,
// never arithmetic.
kernel void mtp12_q4_0_gemv_x4(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tg * 4u + sg;
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32u;
    device const uchar* row_bytes = q4_weights + weight_byte_offset
        + ulong(row) * ulong(blocks_per_row) * 18ul;
    float partial = 0.0f;
    half next_d = half(0.0f);
    uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
    if (lane < blocks_per_row) {
        device const uchar* first = row_bytes + ulong(lane) * 18ul;
        next_d = *reinterpret_cast<device const half*>(first);
        device const packed_uchar4* first_q4 =
            reinterpret_cast<device const packed_uchar4*>(first + 2);
        next_q0 = uchar4(first_q4[0]);
        next_q1 = uchar4(first_q4[1]);
        next_q2 = uchar4(first_q4[2]);
        next_q3 = uchar4(first_q4[3]);
    }
    for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
        const float d = float(next_d);
        const uchar4 q[4] = {next_q0, next_q1, next_q2, next_q3};
        if (block_index + 32u < blocks_per_row) {
            device const uchar* next = row_bytes + ulong(block_index + 32u) * 18ul;
            next_d = *reinterpret_cast<device const half*>(next);
            device const packed_uchar4* next_q4 =
                reinterpret_cast<device const packed_uchar4*>(next + 2);
            next_q0 = uchar4(next_q4[0]);
            next_q1 = uchar4(next_q4[1]);
            next_q2 = uchar4(next_q4[2]);
            next_q3 = uchar4(next_q4[3]);
        }
        const uint input_base = block_index * 32u;
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {
            const uchar4 packed = q[k];
            const uint offset = k * 4u;
            partial += d * (
                float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
              + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
              + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
              + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
              + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
              + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
              + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
              + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
        }
    }
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 =
        simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 =
        simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    const float value = pair01 + pair23;
    if (lane == 0u) {
        output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
    }
}

// One 32-thread simdgroup per threadgroup walking `rows_per_group`
// consecutive rows (CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4=2 for cols <= 1024,
// =3 for every shape).  The lane->block map of mtp12_q4_0_gemv is
// row-independent, so lane l's 32 input floats for block l are loaded ONCE
// into registers and reused for every row the simdgroup folds; the
// established kernel re-reads the whole input vector from device memory for
// every row (32 input loads per 18-byte weight block).  Blocks beyond the
// first 32 (cols > 1024) read their inputs in place exactly as before, and
// the next row's first block is prefetched into registers while the current
// row is folded.  The per-row program - block order, the accumulation
// statement, the shuffle fold and the lane-0 rounding - is verbatim, so the
// output bits match mtp12_q4_0_gemv.
kernel void mtp12_q4_0_gemv_staged(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    constant uint& rows_per_group [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint first_row = tg * rows_per_group;
    if (first_row >= rows) return;
    const uint last_row = min(first_row + rows_per_group, rows);
    const uint blocks_per_row = cols / 32u;
    const bool owns_block = lane < blocks_per_row;
    float x[32];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) x[i] = 0.0f;
    if (owns_block) {
        const uint input_base = lane * 32u;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) x[i] = input[input_base + i];
    }
    half next_d = half(0.0f);
    uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
    if (owns_block) {
        device const uchar* first = q4_weights + weight_byte_offset
            + ulong(first_row) * ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
        next_d = *reinterpret_cast<device const half*>(first);
        device const packed_uchar4* first_q4 =
            reinterpret_cast<device const packed_uchar4*>(first + 2);
        next_q0 = uchar4(first_q4[0]);
        next_q1 = uchar4(first_q4[1]);
        next_q2 = uchar4(first_q4[2]);
        next_q3 = uchar4(first_q4[3]);
    }
    for (uint row = first_row; row < last_row; ++row) {
        device const uchar* row_bytes = q4_weights + weight_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        const float first_d = float(next_d);
        const uchar4 first_q[4] = {next_q0, next_q1, next_q2, next_q3};
        if (row + 1u < last_row && owns_block) {
            device const uchar* next = row_bytes + ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
            next_d = *reinterpret_cast<device const half*>(next);
            device const packed_uchar4* next_q4 =
                reinterpret_cast<device const packed_uchar4*>(next + 2);
            next_q0 = uchar4(next_q4[0]);
            next_q1 = uchar4(next_q4[1]);
            next_q2 = uchar4(next_q4[2]);
            next_q3 = uchar4(next_q4[3]);
        }
        float partial = 0.0f;
        if (owns_block) {
            const float d = first_d;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = first_q[k];
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * x[offset]
                  + float(int(packed.x >> 4) - 8) * x[16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * x[offset + 1u]
                  + float(int(packed.y >> 4) - 8) * x[17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * x[offset + 2u]
                  + float(int(packed.z >> 4) - 8) * x[18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * x[offset + 3u]
                  + float(int(packed.w >> 4) - 8) * x[19u + offset]);
            }
        }
        for (uint block_index = lane + 32u; block_index < blocks_per_row; block_index += 32u) {
            device const uchar* block = row_bytes + ulong(block_index) * 18ul;
            const float d = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* q4 =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = uchar4(q4[k]);
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        if (lane == 0u) {
            output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
        }
    }
}

// NOT BIT-IDENTICAL.  Each row is folded by `splits` simdgroups that each own
// a DISJOINT contiguous range of the row's blocks; the per-simdgroup partials
// are then summed in simdgroup order.  Every simdgroup runs the per-row
// program of mtp12_q4_0_gemv verbatim over its own range (same lane->block
// stride, same accumulation statement, same shuffle fold), but the row total
// is (r0 + r1 [+ r2 + r3]) instead of one 32-lane fold over all blocks, so the
// float addition order differs and the drafts it produces are NOT the
// established drafts.  It exists only to measure whether the 1024-row shapes
// are simdgroup-parallelism starved (CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4=4 for
// two splits, =5 for four); it is excluded from the bit gates by construction
// and must not ship without an acceptance re-measurement.
kernel void mtp12_q4_0_gemv_split(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    constant uint& splits [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[4];
    // `row` is threadgroup-uniform, so the whole group leaves together and no
    // thread reaches the barrier alone.
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32u;
    device const uchar* row_bytes = q4_weights + weight_byte_offset
        + ulong(row) * ulong(blocks_per_row) * 18ul;
    const uint per_split = (blocks_per_row + splits - 1u) / splits;
    const uint begin = sg * per_split;
    const uint end = min(begin + per_split, blocks_per_row);
    float partial = 0.0f;
    for (uint block_index = begin + lane; block_index < end; block_index += 32u) {
        device const uchar* block = row_bytes + ulong(block_index) * 18ul;
        const float d = float(*reinterpret_cast<device const half*>(block));
        device const packed_uchar4* q4 =
            reinterpret_cast<device const packed_uchar4*>(block + 2);
        const uint input_base = block_index * 32u;
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {
            const uchar4 packed = uchar4(q4[k]);
            const uint offset = k * 4u;
            partial += d * (
                float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
              + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
              + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
              + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
              + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
              + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
              + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
              + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
        }
    }
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 =
        simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 =
        simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    if (lane == 0u) partials[sg] = pair01 + pair23;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && lane == 0u) {
        float value = partials[0];
        for (uint split = 1u; split < splits; ++split) value += partials[split];
        output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
    }
}

// Gate row, up row and the GELU gate in one dispatch
// (CAMELID_GEMMA4_MTP12_FUSE_GATEUP=1), x4 geometry.  The simdgroup computes
// the gate row and then the up row with two independent copies of the
// mtp12_q4_0_gemv_x4 per-row program (no shared temporaries), rounds each to
// BF16 exactly as round_output_bf16=1 does, and lane 0 applies the
// mtp12_gelu_mul expression verbatim (its input rounds are identities on the
// already-rounded values).
kernel void mtp12_q4_0_gemv_gate_up_gelu(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& gate_byte_offset [[buffer(5)]],
    constant ulong& up_byte_offset [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = tg * 4u + sg;
    if (row >= rows) return;
    const uint blocks_per_row = cols / 32u;
    float gate_value;
    float up_value;
    {
        device const uchar* row_bytes = q4_weights + gate_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        float partial = 0.0f;
        half next_d = half(0.0f);
        uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
        if (lane < blocks_per_row) {
            device const uchar* first = row_bytes + ulong(lane) * 18ul;
            next_d = *reinterpret_cast<device const half*>(first);
            device const packed_uchar4* first_q4 =
                reinterpret_cast<device const packed_uchar4*>(first + 2);
            next_q0 = uchar4(first_q4[0]);
            next_q1 = uchar4(first_q4[1]);
            next_q2 = uchar4(first_q4[2]);
            next_q3 = uchar4(first_q4[3]);
        }
        for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
            const float d = float(next_d);
            const uchar4 q[4] = {next_q0, next_q1, next_q2, next_q3};
            if (block_index + 32u < blocks_per_row) {
                device const uchar* next = row_bytes + ulong(block_index + 32u) * 18ul;
                next_d = *reinterpret_cast<device const half*>(next);
                device const packed_uchar4* next_q4 =
                    reinterpret_cast<device const packed_uchar4*>(next + 2);
                next_q0 = uchar4(next_q4[0]);
                next_q1 = uchar4(next_q4[1]);
                next_q2 = uchar4(next_q4[2]);
                next_q3 = uchar4(next_q4[3]);
            }
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = q[k];
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        gate_value = value;
    }
    {
        device const uchar* row_bytes = q4_weights + up_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        float partial = 0.0f;
        half next_d = half(0.0f);
        uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
        if (lane < blocks_per_row) {
            device const uchar* first = row_bytes + ulong(lane) * 18ul;
            next_d = *reinterpret_cast<device const half*>(first);
            device const packed_uchar4* first_q4 =
                reinterpret_cast<device const packed_uchar4*>(first + 2);
            next_q0 = uchar4(first_q4[0]);
            next_q1 = uchar4(first_q4[1]);
            next_q2 = uchar4(first_q4[2]);
            next_q3 = uchar4(first_q4[3]);
        }
        for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
            const float d = float(next_d);
            const uchar4 q[4] = {next_q0, next_q1, next_q2, next_q3};
            if (block_index + 32u < blocks_per_row) {
                device const uchar* next = row_bytes + ulong(block_index + 32u) * 18ul;
                next_d = *reinterpret_cast<device const half*>(next);
                device const packed_uchar4* next_q4 =
                    reinterpret_cast<device const packed_uchar4*>(next + 2);
                next_q0 = uchar4(next_q4[0]);
                next_q1 = uchar4(next_q4[1]);
                next_q2 = uchar4(next_q4[2]);
                next_q3 = uchar4(next_q4[3]);
            }
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = q[k];
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        up_value = value;
    }
    if (lane == 0u) {
        const float gate = mtp12_round_bf16(gate_value);
        const float up = mtp12_round_bf16(up_value);
        const float x = mtp12_round_bf16(gate);
        const float x3 = x * x * x;
        const float activated = mtp12_round_bf16(
            0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x3))));
        output[row] = mtp12_round_bf16(activated * mtp12_round_bf16(up));
    }
}

// Draft-only shortlist: score all 2048 raw-mean centroids and select the
// largest T scores with stable first-index tie breaking. The target verifier
// remains authoritative; these kernels only change proposed draft tokens.
kernel void mtp12_shortlist_scores(
    device const float* centroids [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* scores [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint col = lane; col < 1024u; col += 32u)
        sum += centroids[row * 1024u + col] * input[col];
    const float value = simd_sum(sum);
    if (lane == 0u) scores[row] = value;
}

kernel void mtp12_shortlist_select(
    device const float* scores [[buffer(0)]],
    device uint* selected [[buffer(1)]],
    constant uint& top [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= 2048u) return;
    // A nonfinite score conservatively admits its cluster. Finite queries
    // produce exactly T selected clusters, including exact score ties.
    const float score = scores[row];
    if (!isfinite(score)) { selected[row] = 1u; return; }
    uint rank = 0u;
    for (uint other = 0u; other < 2048u && rank < top; ++other) {
        const float candidate = scores[other];
        rank += uint(candidate > score || (candidate == score && other < row));
    }
    selected[row] = uint(rank < top);
}

kernel void mtp12_q4_0_gemv_shortlist(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    device const ushort4* token_clusters [[buffer(7)]],
    device const uint* selected_clusters [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (row >= rows) return;
    const ushort4 clusters = token_clusters[row];
    if (selected_clusters[clusters.x] == 0u &&
        selected_clusters[clusters.y] == 0u &&
        selected_clusters[clusters.z] == 0u) {
        if (lane == 0u) output[row] = -INFINITY;
        return;
    }
    const uint blocks_per_row = cols / 32u;
    device const uchar* row_bytes = q4_weights + weight_byte_offset
        + ulong(row) * ulong(blocks_per_row) * 18ul;
    float partial = 0.0f;
    for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
        device const uchar* block = row_bytes + ulong(block_index) * 18ul;
        const float d = float(*reinterpret_cast<device const half*>(block));
        device const packed_uchar4* q4 =
            reinterpret_cast<device const packed_uchar4*>(block + 2);
        const uint input_base = block_index * 32u;
        #pragma unroll
        for (uint k = 0u; k < 4u; ++k) {
            const uchar4 packed = uchar4(q4[k]);
            const uint offset = k * 4u;
            partial += d * (
                float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
              + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
              + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
              + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
              + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
              + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
              + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
              + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
        }
    }
    partial += simd_shuffle_down(partial, ushort(16));
    partial += simd_shuffle_down(partial, ushort(8));
    partial += simd_shuffle_down(partial, ushort(4));
    const float pair01 =
        simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
    const float pair23 =
        simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
    const float value = pair01 + pair23;
    if (lane == 0u) {
        output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
    }
}

// Per-256-row local compaction removes empty GEMV work without a global
// scan, indirect dispatch, or extra scratch. The retained row dot below is
// identical to mtp12_q4_0_gemv_shortlist.
kernel void mtp12_q4_0_gemv_compact_local(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    device const ushort4* token_clusters [[buffer(7)]],
    device const uint* selected_clusters [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint ids[256];
    threadgroup uint counts[8];
    threadgroup uint offsets[8];
    threadgroup uint total;
    const uint candidate = group * 256u + tid;
    uint keep = 0u;
    if (candidate < rows) {
        const ushort4 c = token_clusters[candidate];
        keep = uint(selected_clusters[c.x] || selected_clusters[c.y] || selected_clusters[c.z]);
        output[candidate] = -INFINITY;
    }
    const uint within_sg = simd_prefix_exclusive_sum(keep);
    const uint sg_count = simd_sum(keep);
    if (lane == 0u) counts[sg] = sg_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint n = lane < 8u ? counts[lane] : 0u;
        const uint preceding = simd_prefix_exclusive_sum(n);
        const uint sum = simd_sum(n);
        if (lane < 8u) offsets[lane] = preceding;
        if (lane == 0u) total = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (keep != 0u) ids[offsets[sg] + within_sg] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint index = sg; index < total; index += 8u) {
        const uint row = ids[index];
        const uint blocks_per_row = cols / 32u;
        device const uchar* row_bytes = q4_weights + weight_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        float partial = 0.0f;
        for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
            device const uchar* block = row_bytes + ulong(block_index) * 18ul;
            const float d = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* q4 =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = uchar4(q4[k]);
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        if (lane == 0u) {
            output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
        }
    }
}

// mtp12_q4_0_gemv_compact_local with an order-preserving register prefetch
// in its serial retained-row loop (CAMELID_GEMMA4_MTP12_FUSE_HEAD_PREFETCH=1).
// Before folding row ids[index], each lane loads its FIRST block (block
// `lane`) of row ids[index + 8] into registers; blocks beyond the first
// (cols > 1024) are loaded in place exactly as before.  The compaction, the
// lane->block map, the accumulation statement and the fold are verbatim.
kernel void mtp12_q4_0_gemv_compact_local_prefetch(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    device const ushort4* token_clusters [[buffer(7)]],
    device const uint* selected_clusters [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint ids[256];
    threadgroup uint counts[8];
    threadgroup uint offsets[8];
    threadgroup uint total;
    const uint candidate = group * 256u + tid;
    uint keep = 0u;
    if (candidate < rows) {
        const ushort4 c = token_clusters[candidate];
        keep = uint(selected_clusters[c.x] || selected_clusters[c.y] || selected_clusters[c.z]);
        output[candidate] = -INFINITY;
    }
    const uint within_sg = simd_prefix_exclusive_sum(keep);
    const uint sg_count = simd_sum(keep);
    if (lane == 0u) counts[sg] = sg_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint n = lane < 8u ? counts[lane] : 0u;
        const uint preceding = simd_prefix_exclusive_sum(n);
        const uint sum = simd_sum(n);
        if (lane < 8u) offsets[lane] = preceding;
        if (lane == 0u) total = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (keep != 0u) ids[offsets[sg] + within_sg] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const uint blocks_per_row = cols / 32u;
    half next_d = half(0.0f);
    uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
    if (sg < total && lane < blocks_per_row) {
        device const uchar* first = q4_weights + weight_byte_offset
            + ulong(ids[sg]) * ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
        next_d = *reinterpret_cast<device const half*>(first);
        device const packed_uchar4* first_q4 =
            reinterpret_cast<device const packed_uchar4*>(first + 2);
        next_q0 = uchar4(first_q4[0]);
        next_q1 = uchar4(first_q4[1]);
        next_q2 = uchar4(first_q4[2]);
        next_q3 = uchar4(first_q4[3]);
    }
    for (uint index = sg; index < total; index += 8u) {
        const uint row = ids[index];
        device const uchar* row_bytes = q4_weights + weight_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        const float first_d = float(next_d);
        const uchar4 first_q[4] = {next_q0, next_q1, next_q2, next_q3};
        if (index + 8u < total && lane < blocks_per_row) {
            device const uchar* next = q4_weights + weight_byte_offset
                + ulong(ids[index + 8u]) * ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
            next_d = *reinterpret_cast<device const half*>(next);
            device const packed_uchar4* next_q4 =
                reinterpret_cast<device const packed_uchar4*>(next + 2);
            next_q0 = uchar4(next_q4[0]);
            next_q1 = uchar4(next_q4[1]);
            next_q2 = uchar4(next_q4[2]);
            next_q3 = uchar4(next_q4[3]);
        }
        float partial = 0.0f;
        for (uint block_index = lane; block_index < blocks_per_row; block_index += 32u) {
            float d;
            uchar4 q[4];
            if (block_index == lane) {
                d = first_d;
                q[0] = first_q[0];
                q[1] = first_q[1];
                q[2] = first_q[2];
                q[3] = first_q[3];
            } else {
                device const uchar* block = row_bytes + ulong(block_index) * 18ul;
                d = float(*reinterpret_cast<device const half*>(block));
                device const packed_uchar4* q4 =
                    reinterpret_cast<device const packed_uchar4*>(block + 2);
                q[0] = uchar4(q4[0]);
                q[1] = uchar4(q4[1]);
                q[2] = uchar4(q4[2]);
                q[3] = uchar4(q4[3]);
            }
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = q[k];
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        if (lane == 0u) {
            output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
        }
    }
}

// mtp12_q4_0_gemv_compact_local_prefetch with the lane's 32 input floats
// for its first block staged in registers across the serial retained-row
// loop (CAMELID_GEMMA4_MTP12_FUSE_HEAD_PREFETCH=2).  The compaction, the row
// prefetch, the lane->block map, the accumulation statement and the fold are
// verbatim; only the input loads of block `lane` move out of the row loop
// (blocks beyond the first, cols > 1024, still read their inputs in place).
kernel void mtp12_q4_0_gemv_compact_local_staged(
    device const uchar* q4_weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant ulong& weight_byte_offset [[buffer(5)]],
    constant uint& round_output_bf16 [[buffer(6)]],
    device const ushort4* token_clusters [[buffer(7)]],
    device const uint* selected_clusters [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint ids[256];
    threadgroup uint counts[8];
    threadgroup uint offsets[8];
    threadgroup uint total;
    const uint candidate = group * 256u + tid;
    uint keep = 0u;
    if (candidate < rows) {
        const ushort4 c = token_clusters[candidate];
        keep = uint(selected_clusters[c.x] || selected_clusters[c.y] || selected_clusters[c.z]);
        output[candidate] = -INFINITY;
    }
    const uint within_sg = simd_prefix_exclusive_sum(keep);
    const uint sg_count = simd_sum(keep);
    if (lane == 0u) counts[sg] = sg_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        const uint n = lane < 8u ? counts[lane] : 0u;
        const uint preceding = simd_prefix_exclusive_sum(n);
        const uint sum = simd_sum(n);
        if (lane < 8u) offsets[lane] = preceding;
        if (lane == 0u) total = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (keep != 0u) ids[offsets[sg] + within_sg] = candidate;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    const uint blocks_per_row = cols / 32u;
    const bool owns_block = lane < blocks_per_row;
    float x[32];
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) x[i] = 0.0f;
    if (owns_block && sg < total) {
        const uint input_base = lane * 32u;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) x[i] = input[input_base + i];
    }
    half next_d = half(0.0f);
    uchar4 next_q0 = uchar4(0), next_q1 = uchar4(0), next_q2 = uchar4(0), next_q3 = uchar4(0);
    if (sg < total && owns_block) {
        device const uchar* first = q4_weights + weight_byte_offset
            + ulong(ids[sg]) * ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
        next_d = *reinterpret_cast<device const half*>(first);
        device const packed_uchar4* first_q4 =
            reinterpret_cast<device const packed_uchar4*>(first + 2);
        next_q0 = uchar4(first_q4[0]);
        next_q1 = uchar4(first_q4[1]);
        next_q2 = uchar4(first_q4[2]);
        next_q3 = uchar4(first_q4[3]);
    }
    for (uint index = sg; index < total; index += 8u) {
        const uint row = ids[index];
        device const uchar* row_bytes = q4_weights + weight_byte_offset
            + ulong(row) * ulong(blocks_per_row) * 18ul;
        const float first_d = float(next_d);
        const uchar4 first_q[4] = {next_q0, next_q1, next_q2, next_q3};
        if (index + 8u < total && owns_block) {
            device const uchar* next = q4_weights + weight_byte_offset
                + ulong(ids[index + 8u]) * ulong(blocks_per_row) * 18ul + ulong(lane) * 18ul;
            next_d = *reinterpret_cast<device const half*>(next);
            device const packed_uchar4* next_q4 =
                reinterpret_cast<device const packed_uchar4*>(next + 2);
            next_q0 = uchar4(next_q4[0]);
            next_q1 = uchar4(next_q4[1]);
            next_q2 = uchar4(next_q4[2]);
            next_q3 = uchar4(next_q4[3]);
        }
        float partial = 0.0f;
        if (owns_block) {
            const float d = first_d;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = first_q[k];
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * x[offset]
                  + float(int(packed.x >> 4) - 8) * x[16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * x[offset + 1u]
                  + float(int(packed.y >> 4) - 8) * x[17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * x[offset + 2u]
                  + float(int(packed.z >> 4) - 8) * x[18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * x[offset + 3u]
                  + float(int(packed.w >> 4) - 8) * x[19u + offset]);
            }
        }
        for (uint block_index = lane + 32u; block_index < blocks_per_row; block_index += 32u) {
            device const uchar* block = row_bytes + ulong(block_index) * 18ul;
            const float d = float(*reinterpret_cast<device const half*>(block));
            device const packed_uchar4* q4 =
                reinterpret_cast<device const packed_uchar4*>(block + 2);
            const uint input_base = block_index * 32u;
            #pragma unroll
            for (uint k = 0u; k < 4u; ++k) {
                const uchar4 packed = uchar4(q4[k]);
                const uint offset = k * 4u;
                partial += d * (
                    float(int(packed.x & 0x0f) - 8) * input[input_base + offset]
                  + float(int(packed.x >> 4) - 8) * input[input_base + 16u + offset]
                  + float(int(packed.y & 0x0f) - 8) * input[input_base + offset + 1u]
                  + float(int(packed.y >> 4) - 8) * input[input_base + 17u + offset]
                  + float(int(packed.z & 0x0f) - 8) * input[input_base + offset + 2u]
                  + float(int(packed.z >> 4) - 8) * input[input_base + 18u + offset]
                  + float(int(packed.w & 0x0f) - 8) * input[input_base + offset + 3u]
                  + float(int(packed.w >> 4) - 8) * input[input_base + 19u + offset]);
            }
        }
        partial += simd_shuffle_down(partial, ushort(16));
        partial += simd_shuffle_down(partial, ushort(8));
        partial += simd_shuffle_down(partial, ushort(4));
        const float pair01 =
            simd_shuffle(partial, ushort(0)) + simd_shuffle(partial, ushort(1));
        const float pair23 =
            simd_shuffle(partial, ushort(2)) + simd_shuffle(partial, ushort(3));
        const float value = pair01 + pair23;
        if (lane == 0u) {
            output[row] = round_output_bf16 != 0u ? mtp12_round_bf16(value) : value;
        }
    }
}

// Pinned ATen-style contiguous f32 RMS geometry for all production widths
// (256/512/1024). One threadgroup owns a row/head.
kernel void mtp12_rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float residue[16];
    threadgroup float inverse_rms;
    const uint base = row * width;
    if (tid < 16u) {
        float row_residue = 0.0f;
        for (uint slab = 0u; slab < width; slab += 256u) {
            float slab_residue = 0.0f;
            for (uint item = 0u; item < 16u; ++item) {
                const float value = input[base + slab + tid + item * 16u];
                slab_residue += value * value;
            }
            row_residue += slab_residue;
        }
        residue[tid] = row_residue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        residue[tid] = residue[tid] + residue[4u + tid]
            + residue[8u + tid] + residue[12u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float sum = ((residue[0] + residue[1]) + residue[2]) + residue[3];
        inverse_rms = 1.0f / sqrt(sum / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < width; index += 256u) {
        output[base + index] =
            mtp12_round_bf16(input[base + index] * inverse_rms * weight[index]);
    }
}

// The mtp12_rms_norm reduction recipe over `width` contiguous floats, shared
// by the fused norm kernels below: the same 16-thread slab/item order, the
// same four-way fold and the same final sum.  `input` is a device or a
// threadgroup pointer; the loaded values, not their address space, decide
// the bits.  All 256 threads must call it (three barriers).
template <typename InputPointer>
inline float mtp12_rms_norm_inverse(
    InputPointer input,
    uint width,
    float eps,
    uint tid,
    threadgroup float* residue,
    threadgroup float* inverse_rms) {
    if (tid < 16u) {
        float row_residue = 0.0f;
        for (uint slab = 0u; slab < width; slab += 256u) {
            float slab_residue = 0.0f;
            for (uint item = 0u; item < 16u; ++item) {
                const float value = input[slab + tid + item * 16u];
                slab_residue += value * value;
            }
            row_residue += slab_residue;
        }
        residue[tid] = row_residue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4u) {
        residue[tid] = residue[tid] + residue[4u + tid]
            + residue[8u + tid] + residue[12u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float sum = ((residue[0] + residue[1]) + residue[2]) + residue[3];
        *inverse_rms = 1.0f / sqrt(sum / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *inverse_rms;
}

// post-norm -> residual add [-> scale] -> next norm in one 256-thread
// threadgroup (CAMELID_GEMMA4_MTP12_FUSE_NORM=1).  Replaces
// mtp12_rms_norm(input, weight) + mtp12_add_bf16 / mtp12_add_scale_bf16 +
// mtp12_rms_norm(sum, next_weight) with the identical expressions and
// rounding points: normalized = round((x * inv) * w); sum = round(a + b);
// [sum = round(sum * scale)]; next = round((sum * inv2) * w2).  The summed
// row is both written to `out_residual` and staged in threadgroup memory for
// the second reduction (the same bits the established path re-reads from
// device memory).
kernel void mtp12_norm_add_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const float* next_weight [[buffer(3)]],
    device float* out_residual [[buffer(4)]],
    device float* out_normed [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& apply_scale [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float residue[16];
    threadgroup float inverse_rms;
    threadgroup float next_inverse_rms;
    threadgroup float summed[1024];
    if (width > 1024u) return;
    const float inv = mtp12_rms_norm_inverse(input, width, eps, tid, residue, &inverse_rms);
    for (uint index = tid; index < width; index += 256u) {
        const float normalized = mtp12_round_bf16(input[index] * inv * weight[index]);
        float value;
        if (apply_scale != 0u) {
            const float sum = mtp12_round_bf16(residual[index] + normalized);
            value = mtp12_round_bf16(sum * scale);
        } else {
            value = mtp12_round_bf16(residual[index] + normalized);
        }
        out_residual[index] = value;
        summed[index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float next_inv =
        mtp12_rms_norm_inverse(summed, width, eps, tid, residue, &next_inverse_rms);
    for (uint index = tid; index < width; index += 256u) {
        out_normed[index] = mtp12_round_bf16(summed[index] * next_inv * next_weight[index]);
    }
}

kernel void mtp12_rope_split(
    device float* data [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint half_head = head_dim / 2u;
    const uint total = head_count * half_head;
    if (gid >= total) return;
    const uint head = gid / half_head;
    const uint pair = gid - head * half_head;
    const uint dim0 = head * head_dim + pair;
    const uint dim1 = dim0 + half_head;
    const float x0 = mtp12_round_bf16(data[dim0]);
    const float x1 = mtp12_round_bf16(data[dim1]);
    const float c = mtp12_round_bf16(cos_table[pair]);
    const float s = mtp12_round_bf16(sin_table[pair]);
    data[dim0] = mtp12_round_bf16(
        mtp12_round_bf16(x0 * c) + mtp12_round_bf16((-x1) * s));
    data[dim1] = mtp12_round_bf16(
        mtp12_round_bf16(x1 * c) + mtp12_round_bf16(x0 * s));
}

// Per-head query RMS norm + split RoPE in one dispatch
// (CAMELID_GEMMA4_MTP12_FUSE_QROPE=1): grid N_HEADS x 256 threads.  The head
// is normalized with the mtp12_rms_norm recipe into threadgroup memory, then
// the mtp12_rope_split expressions are applied verbatim from that staging
// (its round of the already-BF16 normalized value is an identity and is kept
// textually).  `cos_table` / `sin_table` are bound at the query step's row.
kernel void mtp12_qnorm_rope(
    device const float* query [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float residue[16];
    threadgroup float inverse_rms;
    threadgroup float normed[512];
    if (head_dim > 512u || (head_dim & 255u) != 0u) return;
    const uint base = head * head_dim;
    const float inv =
        mtp12_rms_norm_inverse(query + base, head_dim, eps, tid, residue, &inverse_rms);
    for (uint index = tid; index < head_dim; index += 256u) {
        normed[index] = mtp12_round_bf16(query[base + index] * inv * weight[index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint half_head = head_dim / 2u;
    for (uint pair = tid; pair < half_head; pair += 256u) {
        const uint dim0 = base + pair;
        const uint dim1 = dim0 + half_head;
        const float x0 = mtp12_round_bf16(normed[pair]);
        const float x1 = mtp12_round_bf16(normed[pair + half_head]);
        const float c = mtp12_round_bf16(cos_table[pair]);
        const float s = mtp12_round_bf16(sin_table[pair]);
        output[dim0] = mtp12_round_bf16(
            mtp12_round_bf16(x0 * c) + mtp12_round_bf16((-x1) * s));
        output[dim1] = mtp12_round_bf16(
            mtp12_round_bf16(x1 * c) + mtp12_round_bf16(x0 * s));
    }
}

kernel void mtp12_gelu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const float x = mtp12_round_bf16(gate[gid]);
    const float x3 = x * x * x;
    const float activated = mtp12_round_bf16(
        0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x3))));
    output[gid] = mtp12_round_bf16(activated * mtp12_round_bf16(up[gid]));
}

inline float4 mtp12_load_bf16x4(device const float* values, uint base) {
    return float4(
        mtp12_round_bf16(values[base]),
        mtp12_round_bf16(values[base + 1u]),
        mtp12_round_bf16(values[base + 2u]),
        mtp12_round_bf16(values[base + 3u]));
}

kernel void mtp12_attention_scores(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads || (head_dim & 31u) != 0u) return;
    const uint kv_head = head / group;
    const uint q_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    for (uint p = lane; p < position_count; p += 32u) {
        const uint k_base = kv_base + p * position_stride;
        float4 acc0 = 0.0f;
        float4 acc1 = 0.0f;
        float4 acc2 = 0.0f;
        float4 acc3 = 0.0f;
        float4 acc4 = 0.0f;
        float4 acc5 = 0.0f;
        float4 acc6 = 0.0f;
        float4 acc7 = 0.0f;
        for (uint d = 0u; d < head_dim; d += 32u) {
            acc0 += mtp12_load_bf16x4(query, q_base + d) * mtp12_load_bf16x4(keys, k_base + d);
            acc1 += mtp12_load_bf16x4(query, q_base + d + 4u) * mtp12_load_bf16x4(keys, k_base + d + 4u);
            acc2 += mtp12_load_bf16x4(query, q_base + d + 8u) * mtp12_load_bf16x4(keys, k_base + d + 8u);
            acc3 += mtp12_load_bf16x4(query, q_base + d + 12u) * mtp12_load_bf16x4(keys, k_base + d + 12u);
            acc4 += mtp12_load_bf16x4(query, q_base + d + 16u) * mtp12_load_bf16x4(keys, k_base + d + 16u);
            acc5 += mtp12_load_bf16x4(query, q_base + d + 20u) * mtp12_load_bf16x4(keys, k_base + d + 20u);
            acc6 += mtp12_load_bf16x4(query, q_base + d + 24u) * mtp12_load_bf16x4(keys, k_base + d + 24u);
            acc7 += mtp12_load_bf16x4(query, q_base + d + 28u) * mtp12_load_bf16x4(keys, k_base + d + 28u);
        }
        acc0 += acc4; acc1 += acc5; acc2 += acc6; acc3 += acc7;
        acc0 += acc2; acc1 += acc3; acc0 += acc1;
        scores[score_base + p] =
            mtp12_round_bf16((acc0.x + acc0.y) + (acc0.z + acc0.w));
    }
}

kernel void mtp12_attention_softmax(
    device float* scores [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads) return;
    const uint base = head * position_count;
    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32u) {
        local_max = max(local_max, scores[base + p]);
    }
    const float max_score = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint p = lane; p < position_count; p += 32u) {
        const float value = exp(scores[base + p] - max_score);
        scores[base + p] = value;
        local_sum += value;
    }
    const float denominator = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint p = lane; p < position_count; p += 32u) {
        scores[base + p] = mtp12_round_bf16(scores[base + p] / denominator);
    }
}

kernel void mtp12_attention_context(
    device const float* values [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads || compact_base + position_count != physical_logical_k) return;
    const uint kv_head = head / group;
    const uint output_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride;
    const uint score_base = head * position_count;
    const uint vector_end = physical_logical_k & ~3u;
    for (uint d = lane; d < head_dim; d += 32u) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        for (uint p = 0u; p < position_count; ++p) {
            const uint absolute_position = compact_base + p;
            const float product = mtp12_round_bf16(probabilities[score_base + p])
                * mtp12_round_bf16(values[kv_base + p * position_stride + d]);
            if (absolute_position >= vector_end) p0 += product;
            else if ((absolute_position & 3u) == 0u) p0 += product;
            else if ((absolute_position & 3u) == 1u) p1 += product;
            else if ((absolute_position & 3u) == 2u) p2 += product;
            else p3 += product;
        }
        output[output_base + d] = mtp12_round_bf16(((p0 + p1) + p2) + p3);
    }
}

// ---- Assistant attention V2 (opt-in, default off) --------------------------------
// Same per-element arithmetic as mtp12_attention_scores / mtp12_attention_context.
// This library is compiled without fast-math, so the textual operation order IS the
// bit order; only the work distribution changes:
//   * scores: grid (head, position block).  The BF16-rounded query is staged once in
//     threadgroup memory and each lane owns ONE position, keeping the eight float4
//     accumulators and the final fold verbatim;
//   * context: grid (head, 128-dim block).  Each lane owns four dims, the four
//     position-parity partials are kept per component, and the probability/value loads
//     are hoisted four positions ahead so memory latency overlaps the serial adds.
kernel void mtp12_attention_scores_v2(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float4 q_tg[128];
    const uint head = tg.x;
    if (head >= n_heads || (head_dim & 31u) != 0u || head_dim > 512u) return;
    const uint q_base = head * head_dim;
    for (uint c = lane; c < head_dim / 4u; c += 32u) {
        q_tg[c] = mtp12_load_bf16x4(query, q_base + c * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = tg.y * 32u + lane;
    if (p >= position_count) return;
    const uint kv_head = head / group;
    const uint k_base = kv_base_offset + kv_head * kv_head_stride + p * position_stride;
    float4 acc0 = 0.0f;
    float4 acc1 = 0.0f;
    float4 acc2 = 0.0f;
    float4 acc3 = 0.0f;
    float4 acc4 = 0.0f;
    float4 acc5 = 0.0f;
    float4 acc6 = 0.0f;
    float4 acc7 = 0.0f;
    for (uint d = 0u; d < head_dim; d += 32u) {
        const uint c = d / 4u;
        acc0 += q_tg[c] * mtp12_load_bf16x4(keys, k_base + d);
        acc1 += q_tg[c + 1u] * mtp12_load_bf16x4(keys, k_base + d + 4u);
        acc2 += q_tg[c + 2u] * mtp12_load_bf16x4(keys, k_base + d + 8u);
        acc3 += q_tg[c + 3u] * mtp12_load_bf16x4(keys, k_base + d + 12u);
        acc4 += q_tg[c + 4u] * mtp12_load_bf16x4(keys, k_base + d + 16u);
        acc5 += q_tg[c + 5u] * mtp12_load_bf16x4(keys, k_base + d + 20u);
        acc6 += q_tg[c + 6u] * mtp12_load_bf16x4(keys, k_base + d + 24u);
        acc7 += q_tg[c + 7u] * mtp12_load_bf16x4(keys, k_base + d + 28u);
    }
    acc0 += acc4; acc1 += acc5; acc2 += acc6; acc3 += acc7;
    acc0 += acc2; acc1 += acc3; acc0 += acc1;
    scores[head * position_count + p] =
        mtp12_round_bf16((acc0.x + acc0.y) + (acc0.z + acc0.w));
}

inline void mtp12_context_v2_accumulate(
    thread float4& p0,
    thread float4& p1,
    thread float4& p2,
    thread float4& p3,
    uint absolute_position,
    uint vector_end,
    float4 product) {
    if (absolute_position >= vector_end) p0 += product;
    else if ((absolute_position & 3u) == 0u) p0 += product;
    else if ((absolute_position & 3u) == 1u) p1 += product;
    else if ((absolute_position & 3u) == 2u) p2 += product;
    else p3 += product;
}

kernel void mtp12_attention_context_v2(
    device const float* values [[buffer(0)]],
    device const float* probabilities [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint head = tg.x;
    if (head >= n_heads || compact_base + position_count != physical_logical_k) return;
    const uint d = tg.y * 128u + lane * 4u;
    if (d + 4u > head_dim) return;
    const uint kv_head = head / group;
    const uint output_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride + d;
    const uint score_base = head * position_count;
    const uint vector_end = physical_logical_k & ~3u;
    float4 p0 = 0.0f;
    float4 p1 = 0.0f;
    float4 p2 = 0.0f;
    float4 p3 = 0.0f;
    uint p = 0u;
    for (; p + 4u <= position_count; p += 4u) {
        const float pr0 = mtp12_round_bf16(probabilities[score_base + p]);
        const float pr1 = mtp12_round_bf16(probabilities[score_base + p + 1u]);
        const float pr2 = mtp12_round_bf16(probabilities[score_base + p + 2u]);
        const float pr3 = mtp12_round_bf16(probabilities[score_base + p + 3u]);
        const float4 v0 = mtp12_load_bf16x4(values, kv_base + p * position_stride);
        const float4 v1 = mtp12_load_bf16x4(values, kv_base + (p + 1u) * position_stride);
        const float4 v2 = mtp12_load_bf16x4(values, kv_base + (p + 2u) * position_stride);
        const float4 v3 = mtp12_load_bf16x4(values, kv_base + (p + 3u) * position_stride);
        const float4 product0 = pr0 * v0;
        const float4 product1 = pr1 * v1;
        const float4 product2 = pr2 * v2;
        const float4 product3 = pr3 * v3;
        mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p, vector_end, product0);
        mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + 1u, vector_end, product1);
        mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + 2u, vector_end, product2);
        mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + 3u, vector_end, product3);
    }
    for (; p < position_count; ++p) {
        const float pr = mtp12_round_bf16(probabilities[score_base + p]);
        const float4 v = mtp12_load_bf16x4(values, kv_base + p * position_stride);
        const float4 product = pr * v;
        mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p, vector_end, product);
    }
    const float4 folded = ((p0 + p1) + p2) + p3;
    output[output_base + d] = mtp12_round_bf16(folded.x);
    output[output_base + d + 1u] = mtp12_round_bf16(folded.y);
    output[output_base + d + 2u] = mtp12_round_bf16(folded.z);
    output[output_base + d + 3u] = mtp12_round_bf16(folded.w);
}

// Softmax folded into the V2 context kernel
// (CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX=1).  Same grid and bindings as
// mtp12_attention_context_v2 except buffer(1) holds the RAW scores, which are
// never written.  Prologue = mtp12_attention_softmax verbatim (lane-strided
// max, lane-strided exp sum, simd_max / simd_sum) and runs before any
// per-lane exit so all 32 lanes take part.  Probabilities are then produced
// once per position per 32-position chunk (lane l computes position p+l with
// the softmax kernel's expression round(exp(s - max) / denominator)) and
// broadcast with simd_shuffle; the context accumulation is the V2 form with
// the load phase unrolled sixteen positions ahead and the per-parity partials
// accumulated in position order, so the output bits match softmax +
// context_v2.
kernel void mtp12_attention_softmax_context_v2(
    device const float* values [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint head = tg.x;
    if (head >= n_heads || compact_base + position_count != physical_logical_k) return;
    const uint score_base = head * position_count;
    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32u) {
        local_max = max(local_max, scores[score_base + p]);
    }
    const float max_score = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint p = lane; p < position_count; p += 32u) {
        const float value = exp(scores[score_base + p] - max_score);
        local_sum += value;
    }
    const float denominator = simd_sum(local_sum);
    // Lanes past the head stay resident (they take part in every shuffle)
    // but read dim 0 and never write.
    const uint lane_dim = tg.y * 128u + lane * 4u;
    const bool active = lane_dim + 4u <= head_dim;
    const uint d = active ? lane_dim : 0u;
    const uint kv_head = head / group;
    const uint output_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride + d;
    const uint vector_end = physical_logical_k & ~3u;
    float4 p0 = 0.0f;
    float4 p1 = 0.0f;
    float4 p2 = 0.0f;
    float4 p3 = 0.0f;
    uint p = 0u;
    for (; p + 32u <= position_count; p += 32u) {
        const float value = exp(scores[score_base + p + lane] - max_score);
        const float e = mtp12_round_bf16(value / denominator);
        {
            float pr[16];
            float4 v[16];
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                pr[j] = mtp12_round_bf16(simd_shuffle(e, ushort(j)));
                v[j] = mtp12_load_bf16x4(values, kv_base + (p + j) * position_stride);
            }
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                const float4 product = pr[j] * v[j];
                mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + j, vector_end, product);
            }
        }
        {
            float pr[16];
            float4 v[16];
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                pr[j] = mtp12_round_bf16(simd_shuffle(e, ushort(16u + j)));
                v[j] = mtp12_load_bf16x4(values, kv_base + (p + 16u + j) * position_stride);
            }
            #pragma unroll
            for (uint j = 0u; j < 16u; ++j) {
                const float4 product = pr[j] * v[j];
                mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + 16u + j, vector_end, product);
            }
        }
    }
    if (p < position_count) {
        const uint remaining = position_count - p;
        float e = 0.0f;
        if (lane < remaining) {
            const float value = exp(scores[score_base + p + lane] - max_score);
            e = mtp12_round_bf16(value / denominator);
        }
        for (uint j = 0u; j < remaining; ++j) {
            const float pr = mtp12_round_bf16(simd_shuffle(e, ushort(j)));
            const float4 v = mtp12_load_bf16x4(values, kv_base + (p + j) * position_stride);
            const float4 product = pr * v;
            mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + j, vector_end, product);
        }
    }
    if (!active) return;
    const float4 folded = ((p0 + p1) + p2) + p3;
    output[output_base + d] = mtp12_round_bf16(folded.x);
    output[output_base + d + 1u] = mtp12_round_bf16(folded.y);
    output[output_base + d + 2u] = mtp12_round_bf16(folded.z);
    output[output_base + d + 3u] = mtp12_round_bf16(folded.w);
}

// Per-head softmax statistics (CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX=2/3):
// the mtp12_attention_softmax max and exp-sum passes verbatim (lane-strided
// order, simd_max / simd_sum) computed ONCE per head by a 32-lane group and
// written as stats[head * 2] = max, stats[head * 2 + 1] = denominator.  The
// scores stay raw; the stats-fed context kernels below turn them into the
// softmax kernel's probabilities with round(exp(s - max) / denominator), so
// the (heads x dim-blocks) context threadgroups no longer redo the two passes
// that mtp12_attention_softmax_context_v2 repeats in every threadgroup.
kernel void mtp12_attention_softmax_stats(
    device const float* scores [[buffer(0)]],
    constant uint& n_heads [[buffer(1)]],
    constant uint& position_count [[buffer(2)]],
    device float* stats [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (head >= n_heads) return;
    const uint base = head * position_count;
    float local_max = -INFINITY;
    for (uint p = lane; p < position_count; p += 32u) {
        local_max = max(local_max, scores[base + p]);
    }
    const float max_score = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint p = lane; p < position_count; p += 32u) {
        const float value = exp(scores[base + p] - max_score);
        local_sum += value;
    }
    const float denominator = simd_sum(local_sum);
    if (lane == 0u) {
        stats[head * 2u] = max_score;
        stats[head * 2u + 1u] = denominator;
    }
}

// The context phase of mtp12_attention_softmax_context_v2 with (max,
// denominator) read from the stats buffer instead of recomputed per
// threadgroup.  Probabilities are produced once per 32-position chunk (lane l
// computes position p+l with the softmax kernel's expression) and broadcast
// with simd_shuffle; DEPTH positions of loads are issued ahead of their
// accumulation (16 as in the fused kernel, 4 as in mtp12_attention_context_v2)
// and the per-parity partials accumulate in position order, so the output
// bits match softmax + context_v2 at either depth.
template <uint DEPTH>
inline void mtp12_context_v2_from_stats(
    device const float* values,
    device const float* scores,
    device float* output,
    uint n_heads,
    uint head_dim,
    uint position_count,
    uint group,
    uint position_stride,
    uint kv_head_stride,
    uint kv_base_offset,
    uint compact_base,
    uint physical_logical_k,
    device const float* stats,
    uint2 tg,
    uint lane) {
    const uint head = tg.x;
    if (head >= n_heads || compact_base + position_count != physical_logical_k) return;
    const uint score_base = head * position_count;
    const float max_score = stats[head * 2u];
    const float denominator = stats[head * 2u + 1u];
    // Lanes past the head stay resident (they take part in every shuffle)
    // but read dim 0 and never write.
    const uint lane_dim = tg.y * 128u + lane * 4u;
    const bool active = lane_dim + 4u <= head_dim;
    const uint d = active ? lane_dim : 0u;
    const uint kv_head = head / group;
    const uint output_base = head * head_dim;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride + d;
    const uint vector_end = physical_logical_k & ~3u;
    float4 p0 = 0.0f;
    float4 p1 = 0.0f;
    float4 p2 = 0.0f;
    float4 p3 = 0.0f;
    uint p = 0u;
    for (; p + 32u <= position_count; p += 32u) {
        const float value = exp(scores[score_base + p + lane] - max_score);
        const float e = mtp12_round_bf16(value / denominator);
        #pragma unroll
        for (uint c = 0u; c < 32u; c += DEPTH) {
            float pr[DEPTH];
            float4 v[DEPTH];
            #pragma unroll
            for (uint j = 0u; j < DEPTH; ++j) {
                pr[j] = mtp12_round_bf16(simd_shuffle(e, ushort(c + j)));
                v[j] = mtp12_load_bf16x4(values, kv_base + (p + c + j) * position_stride);
            }
            #pragma unroll
            for (uint j = 0u; j < DEPTH; ++j) {
                const float4 product = pr[j] * v[j];
                mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + c + j, vector_end, product);
            }
        }
    }
    if (p < position_count) {
        const uint remaining = position_count - p;
        float e = 0.0f;
        if (lane < remaining) {
            const float value = exp(scores[score_base + p + lane] - max_score);
            e = mtp12_round_bf16(value / denominator);
        }
        for (uint j = 0u; j < remaining; ++j) {
            const float pr = mtp12_round_bf16(simd_shuffle(e, ushort(j)));
            const float4 v = mtp12_load_bf16x4(values, kv_base + (p + j) * position_stride);
            const float4 product = pr * v;
            mtp12_context_v2_accumulate(p0, p1, p2, p3, compact_base + p + j, vector_end, product);
        }
    }
    if (!active) return;
    const float4 folded = ((p0 + p1) + p2) + p3;
    output[output_base + d] = mtp12_round_bf16(folded.x);
    output[output_base + d + 1u] = mtp12_round_bf16(folded.y);
    output[output_base + d + 2u] = mtp12_round_bf16(folded.z);
    output[output_base + d + 3u] = mtp12_round_bf16(folded.w);
}

// CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX=2: stats pre-pass + 16-deep context.
kernel void mtp12_attention_context_v2_stats16(
    device const float* values [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    device const float* stats [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    mtp12_context_v2_from_stats<16u>(
        values, scores, output, n_heads, head_dim, position_count, group,
        position_stride, kv_head_stride, kv_base_offset, compact_base,
        physical_logical_k, stats, tg, lane);
}

// CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX=3: stats pre-pass + 4-deep context
// (the mtp12_attention_context_v2 load depth).
kernel void mtp12_attention_context_v2_stats4(
    device const float* values [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& position_count [[buffer(5)]],
    constant uint& group [[buffer(6)]],
    constant uint& position_stride [[buffer(7)]],
    constant uint& kv_head_stride [[buffer(8)]],
    constant uint& kv_base_offset [[buffer(9)]],
    constant uint& compact_base [[buffer(10)]],
    constant uint& physical_logical_k [[buffer(11)]],
    device const float* stats [[buffer(12)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    mtp12_context_v2_from_stats<4u>(
        values, scores, output, n_heads, head_dim, position_count, group,
        position_stride, kv_head_stride, kv_base_offset, compact_base,
        physical_logical_k, stats, tg, lane);
}

// ---- Assistant attention FORM lane (CAMELID_GEMMA4_MTP12_ATTN_FORM) --------------
// Pipelined context modelled on the verifier's gemma4_tree_context_pipelined:
//   * PAIRS query heads of ONE kv head per threadgroup, so each V element is loaded
//     once and reused by all of them (the V address depends only on kv_head, position
//     and dim, never on the query head);
//   * a 32-position block of probabilities is loaded coalesced once per head (lane l
//     holds position p + l) and distributed with simd_broadcast, a bit-preserving move,
//     instead of every lane re-loading every position;
//   * the V loads of the next 8-position sub-block are issued before the current
//     sub-block's products (software double buffering);
//   * LANE_DIMS output dims per lane instead of the fixed four, so the head-dimension
//     axis splits into 32*LANE_DIMS-wide blocks and far more simdgroups stay resident
//     (HD512 at LANE_DIMS 1 is 16 dim blocks per head chunk, not 4).
//
// BIT-IDENTITY with mtp12_attention_context_v2, per output element (head, dim).
// The v2 chain is four accumulators p0..p3 starting at 0.0f, walking p = 0 ..
// position_count-1 in ascending order and adding
//     product_p = mtp12_round_bf16(probabilities[head * position_count + p])
//               * mtp12_round_bf16(values[kv_base + p * position_stride + dim])
// to bucket  k = (compact_base + p >= vector_end) ? 0 : ((compact_base + p) & 3),
// then emitting mtp12_round_bf16(((p0 + p1) + p2) + p3).
//   INTERIOR here is p in [0, interior_end) with interior_end a multiple of 32 that is
//   also <= vector_end - compact_base.  Every interior position therefore takes the
//   PARITY branch (never the >= vector_end branch), and because interior_end and the
//   block stride are multiples of 4 its bucket is ((compact_base & 3) + (p & 3)) & 3.
//   Accumulator c<i> collects exactly the interior positions with (p & 3) == i, in
//   ascending order, from 0.0f - i.e. exactly the head of real bucket
//   ((compact_base & 3) + i) & 3's v2 chain.  The rotation below MOVES c<i> into that
//   real bucket (a copy, not an add: no earlier term exists), and the TAIL loop then
//   continues the very same chain with the v2 branch and the v2 product expression
//   verbatim.  No term is added, dropped or reassociated.
//   Operands are identical too: simd_broadcast(praw[j], t) returns lane t's bits of
//   probabilities[head * position_count + p + t] (all 32 lanes are live throughout the
//   interior because head_dim % (32 * LANE_DIMS) == 0 and p + 32 <= interior_end <=
//   position_count), and the V element is the same mtp12_round_bf16(values[...]) for
//   every head of the group.  Scalar versus float4 component arithmetic is already
//   pinned bit-identical by mtp12_attention_v2_matches_established_kernels_bit_for_bit
//   (the established kernel is scalar per dim, V2 is float4 per lane).
// This library is compiled with fast-math OFF, so the textual order IS the bit order.
template <uint LANE_DIMS, uint PAIRS>
inline void mtp12_context_form(
    device const float* values,
    device const float* probabilities,
    device float* output,
    uint n_heads,
    uint head_dim,
    uint position_count,
    uint group,
    uint position_stride,
    uint kv_head_stride,
    uint kv_base_offset,
    uint compact_base,
    uint physical_logical_k,
    uint2 tg,
    uint lane) {
    const uint head0 = tg.x * PAIRS;
    if (head0 + PAIRS > n_heads || compact_base + position_count != physical_logical_k) return;
    // PAIRS heads of one chunk share a kv head only when PAIRS divides the group.
    if (group < PAIRS || (group % PAIRS) != 0u) return;
    const uint kv_head = head0 / group;
    const uint lane_dim = tg.y * (32u * LANE_DIMS) + lane * LANE_DIMS;
    // Lanes past the head stay resident (they take part in every broadcast) but read
    // dim 0 and never write.
    const bool active = lane_dim + LANE_DIMS <= head_dim;
    const uint d = active ? lane_dim : 0u;
    const uint kv_base = kv_base_offset + kv_head * kv_head_stride + d;
    const uint vector_end = physical_logical_k & ~3u;

    uint sbase[PAIRS];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        sbase[j] = (head0 + j) * position_count;
    }

    // Whole 32-position blocks whose absolute positions are all below vector_end.
    const uint safe = vector_end > compact_base ? vector_end - compact_base : 0u;
    const uint interior_end = min(position_count, safe) & ~31u;

    float c0[PAIRS][LANE_DIMS];
    float c1[PAIRS][LANE_DIMS];
    float c2[PAIRS][LANE_DIMS];
    float c3[PAIRS][LANE_DIMS];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < LANE_DIMS; ++e) {
            c0[j][e] = 0.0f;
            c1[j][e] = 0.0f;
            c2[j][e] = 0.0f;
            c3[j][e] = 0.0f;
        }
    }

    float vbuf[8][LANE_DIMS];
    if (interior_end >= 32u) {
#pragma clang loop unroll(full)
        for (uint t = 0u; t < 8u; ++t) {
#pragma clang loop unroll(full)
            for (uint e = 0u; e < LANE_DIMS; ++e) {
                vbuf[t][e] = values[kv_base + t * position_stride + e];
            }
        }
    }
    for (uint p = 0u; p < interior_end; p += 32u) {
        float praw[PAIRS];
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            praw[j] = probabilities[sbase[j] + p + lane];
        }
#pragma clang loop unroll(full)
        for (uint sub = 0u; sub < 4u; ++sub) {
            float v[8][LANE_DIMS];
#pragma clang loop unroll(full)
            for (uint t = 0u; t < 8u; ++t) {
#pragma clang loop unroll(full)
                for (uint e = 0u; e < LANE_DIMS; ++e) {
                    v[t][e] = vbuf[t][e];
                }
            }
            // Issue the next sub-block's V loads (after sub 3: the next block's first
            // sub-block, if any) before this sub-block's products.
            const uint next = p + (sub + 1u) * 8u;
            if (sub + 1u < 4u || next < interior_end) {
#pragma clang loop unroll(full)
                for (uint t = 0u; t < 8u; ++t) {
#pragma clang loop unroll(full)
                    for (uint e = 0u; e < LANE_DIMS; ++e) {
                        vbuf[t][e] = values[kv_base + (next + t) * position_stride + e];
                    }
                }
            }
#pragma clang loop unroll(full)
            for (uint t = 0u; t < 8u; ++t) {
                const uint slot = sub * 8u + t;
#pragma clang loop unroll(full)
                for (uint j = 0u; j < PAIRS; ++j) {
                    const float pr = mtp12_round_bf16(simd_broadcast(praw[j], ushort(slot)));
#pragma clang loop unroll(full)
                    for (uint e = 0u; e < LANE_DIMS; ++e) {
                        const float product = pr * mtp12_round_bf16(v[t][e]);
                        // slot is a full-unroll constant, so this picks one bucket at
                        // compile time; p is a multiple of 32 so (p + slot) & 3 == slot & 3.
                        if ((slot & 3u) == 0u) c0[j][e] += product;
                        else if ((slot & 3u) == 1u) c1[j][e] += product;
                        else if ((slot & 3u) == 2u) c2[j][e] += product;
                        else c3[j][e] += product;
                    }
                }
            }
        }
    }

    // Real bucket k continues from the interior positions whose offset parity is
    // (k - (compact_base & 3)) & 3; c<i> becomes bucket ((compact_base & 3) + i) & 3.
    float b0[PAIRS][LANE_DIMS];
    float b1[PAIRS][LANE_DIMS];
    float b2[PAIRS][LANE_DIMS];
    float b3[PAIRS][LANE_DIMS];
    const uint bp = compact_base & 3u;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < LANE_DIMS; ++e) {
            const float x0 = c0[j][e];
            const float x1 = c1[j][e];
            const float x2 = c2[j][e];
            const float x3 = c3[j][e];
            if (bp == 0u) {
                b0[j][e] = x0; b1[j][e] = x1; b2[j][e] = x2; b3[j][e] = x3;
            } else if (bp == 1u) {
                b0[j][e] = x3; b1[j][e] = x0; b2[j][e] = x1; b3[j][e] = x2;
            } else if (bp == 2u) {
                b0[j][e] = x2; b1[j][e] = x3; b2[j][e] = x0; b3[j][e] = x1;
            } else {
                b0[j][e] = x1; b1[j][e] = x2; b2[j][e] = x3; b3[j][e] = x0;
            }
        }
    }

    // TAIL: the v2 branch and product verbatim, ascending, onto the same chain.
    for (uint p = interior_end; p < position_count; ++p) {
        const uint absolute_position = compact_base + p;
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
            const float pr = mtp12_round_bf16(probabilities[sbase[j] + p]);
#pragma clang loop unroll(full)
            for (uint e = 0u; e < LANE_DIMS; ++e) {
                const float product =
                    pr * mtp12_round_bf16(values[kv_base + p * position_stride + e]);
                if (absolute_position >= vector_end) b0[j][e] += product;
                else if ((absolute_position & 3u) == 0u) b0[j][e] += product;
                else if ((absolute_position & 3u) == 1u) b1[j][e] += product;
                else if ((absolute_position & 3u) == 2u) b2[j][e] += product;
                else b3[j][e] += product;
            }
        }
    }

    if (!active) return;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        const uint output_base = (head0 + j) * head_dim;
#pragma clang loop unroll(full)
        for (uint e = 0u; e < LANE_DIMS; ++e) {
            const float folded = ((b0[j][e] + b1[j][e]) + b2[j][e]) + b3[j][e];
            output[output_base + d + e] = mtp12_round_bf16(folded);
        }
    }
}

#define MTP12_CONTEXT_FORM_KERNEL(NAME, LANE_DIMS, PAIRS) \
kernel void NAME( \
    device const float* values [[buffer(0)]], \
    device const float* probabilities [[buffer(1)]], \
    device float* output [[buffer(2)]], \
    constant uint& n_heads [[buffer(3)]], \
    constant uint& head_dim [[buffer(4)]], \
    constant uint& position_count [[buffer(5)]], \
    constant uint& group [[buffer(6)]], \
    constant uint& position_stride [[buffer(7)]], \
    constant uint& kv_head_stride [[buffer(8)]], \
    constant uint& kv_base_offset [[buffer(9)]], \
    constant uint& compact_base [[buffer(10)]], \
    constant uint& physical_logical_k [[buffer(11)]], \
    uint2 tg [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    mtp12_context_form<LANE_DIMS, PAIRS>( \
        values, probabilities, output, n_heads, head_dim, position_count, group, \
        position_stride, kv_head_stride, kv_base_offset, compact_base, \
        physical_logical_k, tg, lane); \
}

// Grid (N_HEADS / PAIRS, head_dim / (32 * LANE_DIMS)) x 32 lanes.  At the 12B shapes:
// HD256 (8 KV heads, group 2) d1p1 = 16 x 8 = 128 threadgroups, d1p2 = 8 x 8 = 64 with
// half the V traffic, d4p2 = 8 x 2 = 16; HD512 (1 KV head, group 16) d1p1 = 16 x 16 =
// 256, d1p2 = 128, d1p4 = 64, d1p8 = 32, and today's context_v2 grid is 32 (HD256) /
// 64 (HD512).  Fewer dims per lane means more resident simdgroups; more pairs per
// threadgroup means less V traffic and fewer probability loads.  Selection is by
// receipt, never by default.
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d1p1, 1u, 1u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d2p1, 2u, 1u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d4p1, 4u, 1u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d1p2, 1u, 2u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d2p2, 2u, 2u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d4p2, 4u, 2u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d1p4, 1u, 4u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d2p4, 2u, 4u)
MTP12_CONTEXT_FORM_KERNEL(mtp12_attention_context_d1p8, 1u, 8u)

// Head-paired scores: PAIRS query heads of ONE kv head per threadgroup, so the K row
// of this lane's position is loaded once and dotted against PAIRS staged queries.
// The eight float4 accumulators, their update order and the final fold are
// mtp12_attention_scores_v2 verbatim, the staged query is the same
// mtp12_load_bf16x4, and the K operand bits do not depend on the query head, so the
// scores are bit-identical.  Only the number of threadgroups and the number of times
// each K row is read change.
template <uint PAIRS>
inline void mtp12_scores_form(
    device const float* query,
    device const float* keys,
    device float* scores,
    uint n_heads,
    uint head_dim,
    uint position_count,
    uint group,
    uint position_stride,
    uint kv_head_stride,
    uint kv_base_offset,
    threadgroup float4* q_tg,
    uint2 tg,
    uint lane) {
    const uint head0 = tg.x * PAIRS;
    if (head0 + PAIRS > n_heads || (head_dim & 31u) != 0u || head_dim > 512u) return;
    if (group < PAIRS || (group % PAIRS) != 0u) return;
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        const uint q_base = (head0 + j) * head_dim;
        for (uint c = lane; c < head_dim / 4u; c += 32u) {
            q_tg[j * 128u + c] = mtp12_load_bf16x4(query, q_base + c * 4u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = tg.y * 32u + lane;
    if (p >= position_count) return;
    const uint kv_head = head0 / group;
    const uint k_base = kv_base_offset + kv_head * kv_head_stride + p * position_stride;
    float4 acc[PAIRS][8];
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
#pragma clang loop unroll(full)
        for (uint a = 0u; a < 8u; ++a) {
            acc[j][a] = 0.0f;
        }
    }
    for (uint d = 0u; d < head_dim; d += 32u) {
        const uint c = d / 4u;
        float4 k[8];
#pragma clang loop unroll(full)
        for (uint a = 0u; a < 8u; ++a) {
            k[a] = mtp12_load_bf16x4(keys, k_base + d + a * 4u);
        }
#pragma clang loop unroll(full)
        for (uint j = 0u; j < PAIRS; ++j) {
#pragma clang loop unroll(full)
            for (uint a = 0u; a < 8u; ++a) {
                acc[j][a] += q_tg[j * 128u + c + a] * k[a];
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint j = 0u; j < PAIRS; ++j) {
        float4 acc0 = acc[j][0];
        float4 acc1 = acc[j][1];
        float4 acc2 = acc[j][2];
        float4 acc3 = acc[j][3];
        acc0 += acc[j][4]; acc1 += acc[j][5]; acc2 += acc[j][6]; acc3 += acc[j][7];
        acc0 += acc2; acc1 += acc3; acc0 += acc1;
        scores[(head0 + j) * position_count + p] =
            mtp12_round_bf16((acc0.x + acc0.y) + (acc0.z + acc0.w));
    }
}

#define MTP12_SCORES_FORM_KERNEL(NAME, PAIRS) \
kernel void NAME( \
    device const float* query [[buffer(0)]], \
    device const float* keys [[buffer(1)]], \
    device float* scores [[buffer(2)]], \
    constant uint& n_heads [[buffer(3)]], \
    constant uint& head_dim [[buffer(4)]], \
    constant uint& position_count [[buffer(5)]], \
    constant uint& group [[buffer(6)]], \
    constant uint& position_stride [[buffer(7)]], \
    constant uint& kv_head_stride [[buffer(8)]], \
    constant uint& kv_base_offset [[buffer(9)]], \
    uint2 tg [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    threadgroup float4 q_tg[(PAIRS) * 128u]; \
    mtp12_scores_form<PAIRS>( \
        query, keys, scores, n_heads, head_dim, position_count, group, \
        position_stride, kv_head_stride, kv_base_offset, q_tg, tg, lane); \
}

// Grid (N_HEADS / PAIRS, ceil(position_count / 32)) x 32 lanes.
MTP12_SCORES_FORM_KERNEL(mtp12_attention_scores_q2, 2u)
MTP12_SCORES_FORM_KERNEL(mtp12_attention_scores_q4, 4u)

kernel void mtp12_argmax(
    device const float* logits [[buffer(0)]],
    device uint* output_id [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint best_id = 0u;
    for (uint i = tid; i < count; i += tgsize) {
        const float candidate = logits[i];
        // Assistant recurrence follows PyTorch argmax/first-index ties.  This
        // is intentionally distinct from Camelid's target-verifier greedy
        // fold, whose lossless contract selects the highest id on exact ties.
        if (!isnan(candidate) &&
            (candidate > best || (candidate == best && i < best_id))) {
            best = candidate;
            best_id = i;
        }
    }
    values[tid] = best;
    indices[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride];
            const uint other_id = indices[tid + stride];
            if (other > values[tid] ||
                (other == values[tid] && other_id < indices[tid])) {
                values[tid] = other;
                indices[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) output_id[0] = indices[0];
}

// Two-stage form of `mtp12_argmax` for the 1 MiB assistant vocabulary, with
// the identical first-index rule.  Stage one gives every threadgroup one
// contiguous `chunk` of logits and writes its (max, first index) partial;
// stage two merges the partials in a single threadgroup.  A chunk whose
// logits are all NaN or -inf reports (-inf, 0), which is exactly the state the
// single-threadgroup scan is left in for such a span, so the merge reproduces
// its answer in every case (`assistant_argmax_chunked_matches_single_
// threadgroup_on_adversarial_logits` gates this).
kernel void mtp12_argmax_partial(
    device const float* logits [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device uint* partial_ids [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& chunk [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    const uint begin = group * chunk;
    const uint end = min(begin + chunk, count);
    float best = -INFINITY;
    uint best_id = 0u;
    for (uint i = begin + tid; i < end; i += tgsize) {
        const float candidate = logits[i];
        if (!isnan(candidate) &&
            (candidate > best || (candidate == best && i < best_id))) {
            best = candidate;
            best_id = i;
        }
    }
    values[tid] = best;
    indices[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride];
            const uint other_id = indices[tid + stride];
            if (other > values[tid] ||
                (other == values[tid] && other_id < indices[tid])) {
                values[tid] = other;
                indices[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partial_values[group] = values[0];
        partial_ids[group] = indices[0];
    }
}

kernel void mtp12_argmax_merge(
    device const float* partial_values [[buffer(0)]],
    device const uint* partial_ids [[buffer(1)]],
    device uint* output_id [[buffer(2)]],
    constant uint& partials [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];
    float best = -INFINITY;
    uint best_id = 0u;
    for (uint p = tid; p < partials; p += tgsize) {
        const float candidate = partial_values[p];
        const uint candidate_id = partial_ids[p];
        if (candidate > best || (candidate == best && candidate_id < best_id)) {
            best = candidate;
            best_id = candidate_id;
        }
    }
    values[tid] = best;
    indices[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsize >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float other = values[tid + stride];
            const uint other_id = indices[tid + stride];
            if (other > values[tid] ||
                (other == values[tid] && other_id < indices[tid])) {
                values[tid] = other;
                indices[tid] = other_id;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) output_id[0] = indices[0];
}
"#;

/// Logits per stage-one threadgroup of the chunked assistant argmax: the
/// 262,144-entry vocabulary becomes 256 partials, one per merge lane.
const MTP12_ARGMAX_CHUNK: usize = 1024;

/// Partial slots the assistant scratch reserves for the chunked argmax.
const MTP12_ARGMAX_MAX_PARTIALS: usize = (VOCAB + MTP12_ARGMAX_CHUNK - 1) / MTP12_ARGMAX_CHUNK;

/// `CAMELID_GEMMA4_MTP12_ARGMAX_LEGACY=1` restores the single-threadgroup
/// `mtp12_argmax` scan (an A/B control).  The chunked form is the default
/// because it is bit-identical by construction and by test; it never changes
/// which token the assistant proposes.
fn mtp12_argmax_legacy_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_MTP12_ARGMAX_LEGACY")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// `CAMELID_MTP12_DUMP_DRAFT_QUERIES=<path>`: append every resident-chain
/// draft's head query (final normalized assistant hidden) and its full-head
/// argmax token to `<path>`, for the offline head-shortlist recall experiment.
/// Collect reference labels with the shortlist disabled: with it enabled the
/// token is the shortlisted proposal, not necessarily the full-head argmax.
/// Diagnostic only; unset in production.
fn mtp12_dump_draft_queries_path() -> Option<&'static std::path::Path> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var_os("CAMELID_MTP12_DUMP_DRAFT_QUERIES")
            .filter(|value| !value.is_empty())
            .map(std::path::PathBuf::from)
    })
    .as_deref()
}

/// Independent draft-side fusion selectors for the W8 tree lane's assistant
/// steps.  Each is read once per process and parsed strictly like
/// `CAMELID_GEMMA4_MTP12_TREE_W8`: unset or `0` keeps the established kernel
/// sequence byte-for-byte, a listed nonzero level selects a fused /
/// regeometried kernel, and anything else fails the assistant load.  Every
/// fused kernel is order-preserving (see the shader notes), so the drafts do
/// not change; the selectors exist so a bit mismatch can be bisected without
/// a rebuild and so the tree-step bench can attribute time per kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mtp12FuseFlags {
    /// `CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4` for every dense GEMV routed through
    /// [`Gemma4Mtp12AssistantMetal::encode_dense_gemv`] (tree, resident chain
    /// and K=1 paths alike; never the BF16 dense route): `1` =
    /// `mtp12_q4_0_gemv_x4` (four rows per 128-thread group), `2` =
    /// `mtp12_q4_0_gemv_staged` (register-staged inputs, `gemv_staged_rows`
    /// rows per simdgroup) for cols <= 1024 with the established kernel for
    /// wider rows, `3` = staged for every shape.  Levels `4` and `5` select
    /// `mtp12_q4_0_gemv_split` with two and four simdgroups per row: those two
    /// fold each row in a different order and are therefore NOT bit-identical
    /// - they are a measurement lane only, and enabling them changes the
    /// drafts and invalidates the acceptance rate.
    gemv_x4: u8,
    /// `CAMELID_GEMMA4_MTP12_GEMV_STAGED_ROWS` (1..=64, default 4): rows each
    /// simdgroup of `mtp12_q4_0_gemv_staged` walks.
    gemv_staged_rows: u32,
    /// `CAMELID_GEMMA4_MTP12_FUSE_GATEUP`: gate + up + GELU in one dispatch on
    /// the tree path (skipped while the BF16 dense route is active).
    gate_up: bool,
    /// `CAMELID_GEMMA4_MTP12_FUSE_NORM`: post-norm -> add [-> scale] -> next
    /// norm in one dispatch on the tree path.
    norm: bool,
    /// `CAMELID_GEMMA4_MTP12_FUSE_QROPE`: per-head query norm + RoPE in one
    /// dispatch on the tree path.
    qrope: bool,
    /// `CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX` on the tree path: `1` = softmax
    /// folded into the V2 context kernel (`mtp12_attention_softmax_context_v2`,
    /// every context threadgroup recomputes max/denominator), `2` =
    /// `mtp12_attention_softmax_stats` pre-pass + 16-deep stats-fed context,
    /// `3` = the same pre-pass + 4-deep stats-fed context.
    softmax_ctx: u8,
    /// `CAMELID_GEMMA4_MTP12_FUSE_HEAD_PREFETCH` for the compact shortlist head
    /// GEMV (tree and resident-chain draft heads): `1` = row prefetch, `2` =
    /// row prefetch + register-staged inputs.
    head_prefetch: u8,
}

impl Default for Mtp12FuseFlags {
    fn default() -> Self {
        Self {
            gemv_x4: 0,
            gemv_staged_rows: MTP12_GEMV_STAGED_ROWS_DEFAULT,
            gate_up: false,
            norm: false,
            qrope: false,
            softmax_ctx: 0,
            head_prefetch: 0,
        }
    }
}

const MTP12_FUSE_GEMV_X4_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4";
const MTP12_GEMV_STAGED_ROWS_ENV: &str = "CAMELID_GEMMA4_MTP12_GEMV_STAGED_ROWS";
const MTP12_FUSE_GATEUP_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_GATEUP";
const MTP12_FUSE_NORM_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_NORM";
const MTP12_FUSE_QROPE_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_QROPE";
const MTP12_FUSE_SOFTMAX_CTX_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX";
const MTP12_FUSE_HEAD_PREFETCH_ENV: &str = "CAMELID_GEMMA4_MTP12_FUSE_HEAD_PREFETCH";
/// Highest listed `CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4` level.  Levels 4 and 5
/// select the deliberately NON-bit-identical split-block experiment.
const MTP12_FUSE_GEMV_X4_MAX: u8 = 5;
/// Lowest `CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4` level that changes the drafts.
const MTP12_FUSE_GEMV_X4_FIRST_INEXACT: u8 = 4;
const MTP12_GEMV_STAGED_ROWS_DEFAULT: u32 = 4;
const MTP12_GEMV_STAGED_ROWS_MAX: u32 = 64;

fn mtp12_fuse_flag(name: &str, value: Option<&str>) -> std::result::Result<bool, String> {
    Ok(mtp12_fuse_level(name, value, 1)? != 0)
}

/// Strict level selector: unset or `0` off, `1..=max` a listed variant (no
/// whitespace, no leading zeros, no aliases).
fn mtp12_fuse_level(name: &str, value: Option<&str>, max: u8) -> std::result::Result<u8, String> {
    match value {
        None => Ok(0),
        Some(text) => {
            let level = match text.as_bytes() {
                [digit @ b'0'..=b'9'] => digit - b'0',
                _ => 255,
            };
            if level <= max {
                Ok(level)
            } else {
                Err(format!("{name} must be exactly 0..={max}; got {text:?}"))
            }
        }
    }
}

fn mtp12_gemv_staged_rows(value: Option<&str>) -> std::result::Result<u32, String> {
    match value {
        None => Ok(MTP12_GEMV_STAGED_ROWS_DEFAULT),
        Some(text) => text
            .parse::<u32>()
            .ok()
            .filter(|rows| (1..=MTP12_GEMV_STAGED_ROWS_MAX).contains(rows) && rows.to_string() == text)
            .ok_or_else(|| {
                format!("{MTP12_GEMV_STAGED_ROWS_ENV} must be an integer in 1..={MTP12_GEMV_STAGED_ROWS_MAX}; got {text:?}")
            }),
    }
}

fn mtp12_fuse_flags_from_env() -> std::result::Result<Mtp12FuseFlags, String> {
    let read = |name: &str| -> std::result::Result<Option<String>, String> {
        match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(format!("{name}: {error}")),
        }
    };
    let level = |name: &str, max: u8| -> std::result::Result<u8, String> {
        mtp12_fuse_level(name, read(name)?.as_deref(), max)
    };
    let flag = |name: &str| -> std::result::Result<bool, String> {
        mtp12_fuse_flag(name, read(name)?.as_deref())
    };
    let gemv_x4 = level(MTP12_FUSE_GEMV_X4_ENV, MTP12_FUSE_GEMV_X4_MAX)?;
    if gemv_x4 >= MTP12_FUSE_GEMV_X4_FIRST_INEXACT {
        eprintln!(
            "[gemma4-mtp12] WARNING: {MTP12_FUSE_GEMV_X4_ENV}={gemv_x4} selects the split-block \
             GEMV experiment, which folds each row in a different order and does NOT produce the \
             established drafts; acceptance must be re-measured."
        );
    }
    Ok(Mtp12FuseFlags {
        gemv_x4,
        gemv_staged_rows: mtp12_gemv_staged_rows(read(MTP12_GEMV_STAGED_ROWS_ENV)?.as_deref())?,
        gate_up: flag(MTP12_FUSE_GATEUP_ENV)?,
        norm: flag(MTP12_FUSE_NORM_ENV)?,
        qrope: flag(MTP12_FUSE_QROPE_ENV)?,
        softmax_ctx: level(MTP12_FUSE_SOFTMAX_CTX_ENV, 3)?,
        head_prefetch: level(MTP12_FUSE_HEAD_PREFETCH_ENV, 2)?,
    })
}

/// The process-wide fusion selectors, read once; a malformed value is an
/// assistant-load error rather than a silent default.
fn mtp12_fuse_flags() -> Result<Mtp12FuseFlags> {
    static FLAGS: std::sync::OnceLock<std::result::Result<Mtp12FuseFlags, String>> =
        std::sync::OnceLock::new();
    FLAGS
        .get_or_init(mtp12_fuse_flags_from_env)
        .clone()
        .map_err(invalid)
}

/// The `CAMELID_GEMMA4_MTP12_ATTN_FORM` pipelines, parallel to
/// [`Mtp12ContextForm::PIPELINED`] and [`Mtp12ScoresForm::PAIRED`].
struct Mtp12AttnFormPipelines {
    context: Vec<ComputePipelineState>,
    scores: Vec<ComputePipelineState>,
}

impl Mtp12AttnFormPipelines {
    fn context(&self, form: Mtp12ContextForm) -> Option<&ComputePipelineState> {
        let index = Mtp12ContextForm::PIPELINED
            .iter()
            .position(|candidate| *candidate == form)?;
        self.context.get(index)
    }

    fn scores(&self, form: Mtp12ScoresForm) -> Option<&ComputePipelineState> {
        let index = Mtp12ScoresForm::PAIRED
            .iter()
            .position(|candidate| *candidate == form)?;
        self.scores.get(index)
    }
}

struct Mtp12Pipelines {
    gather_q6k_embedding: ComputePipelineState,
    gather_q6k_embedding_and_recurrent: ComputePipelineState,
    copy_f32: ComputePipelineState,
    q4_gemv: ComputePipelineState,
    q4_gemv_shortlist: ComputePipelineState,
    q4_gemv_shortlist_compact: ComputePipelineState,
    shortlist_scores: ComputePipelineState,
    shortlist_select: ComputePipelineState,
    rms_norm: ComputePipelineState,
    rope: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_softmax: ComputePipelineState,
    attention_context: ComputePipelineState,
    attention_scores_v2: ComputePipelineState,
    attention_context_v2: ComputePipelineState,
    /// Draft-side fused / regeometried kernels (`CAMELID_GEMMA4_MTP12_FUSE_*`).
    q4_gemv_x4: ComputePipelineState,
    q4_gemv_staged: ComputePipelineState,
    /// NOT bit-identical; bench/experiment only (`FUSE_GEMV_X4=4/5`).
    q4_gemv_split: ComputePipelineState,
    q4_gemv_gate_up_gelu: ComputePipelineState,
    q4_gemv_shortlist_compact_prefetch: ComputePipelineState,
    q4_gemv_shortlist_compact_staged: ComputePipelineState,
    norm_add_norm: ComputePipelineState,
    qnorm_rope: ComputePipelineState,
    attention_softmax_context_v2: ComputePipelineState,
    attention_softmax_stats: ComputePipelineState,
    attention_context_v2_stats16: ComputePipelineState,
    attention_context_v2_stats4: ComputePipelineState,
    /// `CAMELID_GEMMA4_MTP12_ATTN_FORM` pipelined context / head-paired scores.
    attention_forms: Mtp12AttnFormPipelines,
    add: ComputePipelineState,
    add_scale: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    argmax: ComputePipelineState,
    argmax_partial: ComputePipelineState,
    argmax_merge: ComputePipelineState,
}

impl Mtp12Pipelines {
    fn new(device: &Device) -> Result<Self> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(MTP12_SHADER, &options)
            .map_err(|error| invalid(format!("Metal shader compilation failed: {error}")))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|error| invalid(format!("Metal function {name} missing: {error}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| invalid(format!("Metal pipeline {name} failed: {error}")))
        };
        Ok(Self {
            gather_q6k_embedding: pipeline("mtp12_gather_q6k_embedding")?,
            gather_q6k_embedding_and_recurrent: pipeline(
                "mtp12_gather_q6k_embedding_and_recurrent",
            )?,
            copy_f32: pipeline("mtp12_copy_f32")?,
            q4_gemv: pipeline("mtp12_q4_0_gemv")?,
            q4_gemv_shortlist: pipeline("mtp12_q4_0_gemv_shortlist")?,
            q4_gemv_shortlist_compact: pipeline("mtp12_q4_0_gemv_compact_local")?,
            shortlist_scores: pipeline("mtp12_shortlist_scores")?,
            shortlist_select: pipeline("mtp12_shortlist_select")?,
            rms_norm: pipeline("mtp12_rms_norm")?,
            rope: pipeline("mtp12_rope_split")?,
            attention_scores: pipeline("mtp12_attention_scores")?,
            attention_softmax: pipeline("mtp12_attention_softmax")?,
            attention_context: pipeline("mtp12_attention_context")?,
            attention_scores_v2: pipeline("mtp12_attention_scores_v2")?,
            attention_context_v2: pipeline("mtp12_attention_context_v2")?,
            q4_gemv_x4: pipeline("mtp12_q4_0_gemv_x4")?,
            q4_gemv_staged: pipeline("mtp12_q4_0_gemv_staged")?,
            q4_gemv_split: pipeline("mtp12_q4_0_gemv_split")?,
            q4_gemv_gate_up_gelu: pipeline("mtp12_q4_0_gemv_gate_up_gelu")?,
            q4_gemv_shortlist_compact_prefetch: pipeline(
                "mtp12_q4_0_gemv_compact_local_prefetch",
            )?,
            q4_gemv_shortlist_compact_staged: pipeline(
                "mtp12_q4_0_gemv_compact_local_staged",
            )?,
            norm_add_norm: pipeline("mtp12_norm_add_norm")?,
            qnorm_rope: pipeline("mtp12_qnorm_rope")?,
            attention_softmax_context_v2: pipeline("mtp12_attention_softmax_context_v2")?,
            attention_softmax_stats: pipeline("mtp12_attention_softmax_stats")?,
            attention_context_v2_stats16: pipeline("mtp12_attention_context_v2_stats16")?,
            attention_context_v2_stats4: pipeline("mtp12_attention_context_v2_stats4")?,
            attention_forms: Mtp12AttnFormPipelines {
                context: Mtp12ContextForm::PIPELINED
                    .iter()
                    .map(|form| pipeline(form.kernel().expect("pipelined context form")))
                    .collect::<Result<Vec<_>>>()?,
                scores: Mtp12ScoresForm::PAIRED
                    .iter()
                    .map(|form| pipeline(form.kernel().expect("paired scores form")))
                    .collect::<Result<Vec<_>>>()?,
            },
            add: pipeline("mtp12_add_bf16")?,
            add_scale: pipeline("mtp12_add_scale_bf16")?,
            gelu_mul: pipeline("mtp12_gelu_mul")?,
            argmax: pipeline("mtp12_argmax")?,
            argmax_partial: pipeline("mtp12_argmax_partial")?,
            argmax_merge: pipeline("mtp12_argmax_merge")?,
        })
    }
}

#[derive(Clone, Copy)]
struct SourceLayerWeights {
    q: TensorRef,
    o: TensorRef,
    gate: TensorRef,
    up: TensorRef,
    down: TensorRef,
}

struct LayerWeights {
    input_norm: Buffer,
    post_attention_norm: Buffer,
    pre_feedforward_norm: Buffer,
    post_feedforward_norm: Buffer,
    q_norm: Buffer,
    matrices: Q4LayerWeights,
    scale: f32,
}

struct Mtp12Scratch {
    pre_input: Buffer,
    hidden: Buffer,
    normed: Buffer,
    query: Buffer,
    query_normed: Buffer,
    context: Buffer,
    attention_projection: Buffer,
    attention_normalized: Buffer,
    attention_residual: Buffer,
    gate: Buffer,
    up: Buffer,
    gated: Buffer,
    down: Buffer,
    down_normalized: Buffer,
    final_normalized: Buffer,
    recurrent_hidden: Buffer,
    chain_initial_recurrent_hidden: Buffer,
    chain_recurrent_hidden: Buffer,
    /// Per-draft copies of `final_normalized`, written only while
    /// `CAMELID_MTP12_DUMP_DRAFT_QUERIES` is set.
    chain_final_normalized: Buffer,
    logits: Buffer,
    output_token: Buffer,
    /// Stage-one (max, first index) partials of the chunked vocabulary argmax.
    argmax_partial_values: Buffer,
    argmax_partial_ids: Buffer,
    local_cos: Buffer,
    local_sin: Buffer,
    full_cos: Buffer,
    full_sin: Buffer,
    /// Per-head (max, denominator) of `mtp12_attention_softmax_stats`
    /// (`CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX=2/3`).
    attention_stats: Buffer,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Mtp12ResidentLedger {
    pub source_file_bytes: u64,
    pub source_sha256_verified: bool,
    pub source_mapping_retained: bool,
    pub packed_q4_matrix_bytes: u64,
    pub dense_bf16_enabled: bool,
    pub dense_bf16_matrix_bytes: u64,
    pub dense_bf16_upload_us: u128,
    pub decoded_norm_bytes: u64,
    pub fixed_scratch_bytes: u64,
    pub quantize_us: u128,
    pub hash_us: u128,
    pub pipeline_compile_us: u128,
    pub load_wall_us: u128,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Mtp12ProposalTiming {
    pub upload_us: u128,
    pub encode_us: u128,
    pub wait_us: u128,
    pub wall_us: u128,
}

#[derive(Clone, Debug)]
pub struct Gemma4Mtp12Proposal {
    pub token: u32,
    /// Captured intentionally for the K=1 BF16/Q4 stage-oracle harness.  The
    /// production speculative loop should omit this 1 MiB readback.
    pub logits: Vec<f32>,
    pub recurrent_hidden: Vec<f32>,
    pub timing: Gemma4Mtp12ProposalTiming,
}

/// Aggregate timing for one K<=15 resident chain. Exactly one command
/// buffer is committed and waited; GPU timestamps cover that complete buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Mtp12ChainTiming {
    pub cpu_prepare_us: u128,
    pub encode_us: u128,
    pub wait_us: u128,
    pub gpu_us: u128,
    pub kernel_us: u128,
    pub wall_us: u128,
}

/// Byte/dispatch accounting for one resident chain.  Alias fields are caller
/// storage, not assistant allocations.  `readback_bytes` is token ids only;
/// recurrent rows remain in Metal scratch for the whole chain.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Mtp12ChainLedger {
    pub draft_k: u32,
    pub command_buffers: u32,
    pub command_buffer_waits: u32,
    pub target_q6k_table_alias_bytes: u64,
    pub target_kv_alias_bytes: u64,
    pub assistant_matrix_read_bytes: u64,
    pub target_kv_read_bytes: u64,
    pub dynamic_attention_scratch_bytes: u64,
    pub resident_chain_state_bytes: u64,
    pub initial_hidden_upload_bytes: u64,
    pub readback_bytes: u64,
}

/// Token-only CPU result from the one-wait resident chain.  The post-projection
/// recurrent rows are deliberately not copied to the CPU production result.
#[derive(Clone, Debug)]
pub struct Gemma4Mtp12ChainProposal {
    pub tokens: Vec<u32>,
    pub timing: Gemma4Mtp12ChainTiming,
    pub ledger: Gemma4Mtp12ChainLedger,
}

/// CPU-visible target KV accepted only by the isolated K=1 scaffold.
/// Layout is `[kv_head][position][head_dim]`, matching Camelid's logical view.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4Mtp12CpuKv<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_len: usize,
}

impl Gemma4Mtp12CpuKv<'_> {
    fn validate(&self, expected_heads: usize, expected_dim: usize, label: &str) -> Result<()> {
        if self.kv_heads != expected_heads || self.head_dim != expected_dim {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP {label} KV is {}x{}, expected {expected_heads}x{expected_dim}",
                self.kv_heads, self.head_dim
            )));
        }
        let elements = self
            .kv_heads
            .checked_mul(self.kv_len)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or_else(|| invalid(format!("{label} KV element count overflow")))?;
        if self.key.len() != elements || self.value.len() != elements {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP {label} KV has key/value lengths {}/{}, expected {elements}",
                self.key.len(),
                self.value.len()
            )));
        }
        if self
            .key
            .iter()
            .chain(self.value)
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP {label} KV contains non-finite values"
            )));
        }
        Ok(())
    }
}

fn validate_metal_buffer_view(
    view: Gemma4Mtp12MetalBufferView<'_>,
    expected_device_registry_id: u64,
    expected_bytes: u64,
    alignment: u64,
    label: &str,
) -> Result<()> {
    if view.buffer.device().registry_id() != expected_device_registry_id {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP {label} buffer belongs to Metal device {}, expected {}",
            view.buffer.device().registry_id(),
            expected_device_registry_id
        )));
    }
    if view.byte_len != expected_bytes {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP {label} view is {} bytes, expected {expected_bytes}",
            view.byte_len
        )));
    }
    if alignment == 0 || !view.byte_offset.is_multiple_of(alignment) {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP {label} byte offset {} is not {alignment}-byte aligned",
            view.byte_offset
        )));
    }
    let end = view
        .byte_offset
        .checked_add(view.byte_len)
        .ok_or_else(|| invalid(format!("{label} view byte range overflows")))?;
    if end > view.buffer.length() {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP {label} view {}..{end} exceeds backing buffer length {}",
            view.byte_offset,
            view.buffer.length()
        )));
    }
    Ok(())
}

impl Gemma4Mtp12Q6KEmbeddingRow<'_> {
    fn validate(&self, expected_device_registry_id: u64) -> Result<()> {
        if self.hidden != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP target Q6_K embedding width is {}, expected {TARGET_HIDDEN}",
                self.hidden
            )));
        }
        // Q6_K's trailing fp16 super-scale requires two-byte alignment.  The
        // shader adds this scoped offset itself, so odd-token rows at a
        // 32-byte-aligned table base (offset mod 4 == 2) remain legal.
        validate_metal_buffer_view(
            self.wire,
            expected_device_registry_id,
            GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES,
            2,
            "target Q6_K embedding row",
        )
    }
}

impl Gemma4Mtp12Q6KEmbeddingTable<'_> {
    fn validate(&self, expected_device_registry_id: u64) -> Result<()> {
        if self.hidden != TARGET_HIDDEN || self.vocab != VOCAB {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP target Q6_K table is [{}x{}], expected [{VOCAB}x{TARGET_HIDDEN}]",
                self.vocab, self.hidden
            )));
        }
        if self.target_model_sha256 != GEMMA4_12B_QAT_Q4_0_TARGET_SHA256 {
            return Err(invalid(format!(
                "target model SHA-256 {} does not match expected {}",
                self.target_model_sha256, GEMMA4_12B_QAT_Q4_0_TARGET_SHA256
            )));
        }
        validate_metal_buffer_view(
            self.wire,
            expected_device_registry_id,
            GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES,
            2,
            "target Q6_K embedding table",
        )
    }
}

impl Gemma4Mtp12DeviceRecurrentHidden<'_> {
    fn validate(&self, expected_device_registry_id: u64) -> Result<()> {
        if self.hidden != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP recurrent hidden width is {}, expected {TARGET_HIDDEN}",
                self.hidden
            )));
        }
        validate_metal_buffer_view(
            self.values,
            expected_device_registry_id,
            (TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
            std::mem::align_of::<f32>() as u64,
            "target recurrent hidden",
        )
    }
}

impl Gemma4Mtp12DeviceKv<'_> {
    fn validate(
        &self,
        expected_device_registry_id: u64,
        expected_layer: usize,
        expected_heads: usize,
        expected_dim: usize,
        label: &str,
    ) -> Result<()> {
        if self.source_layer != expected_layer
            || self.kv_heads != expected_heads
            || self.head_dim != expected_dim
            || self.max_positions == 0
        {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP {label} device KV is layer {} [{}x{}x{}], expected layer {expected_layer} [{expected_heads}xmax_posx{expected_dim}] with non-zero max_pos",
                self.source_layer, self.kv_heads, self.max_positions, self.head_dim
            )));
        }
        let elements = self
            .kv_heads
            .checked_mul(self.max_positions)
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or_else(|| invalid(format!("{label} device KV element count overflow")))?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid(format!("{label} device KV byte count overflow")))?
            as u64;
        let head_stride = self
            .max_positions
            .checked_mul(self.head_dim)
            .ok_or_else(|| invalid(format!("{label} device KV head stride overflow")))?;
        if head_stride > u32::MAX as usize {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP {label} device KV head stride {head_stride} exceeds u32"
            )));
        }
        validate_metal_buffer_view(
            self.key,
            expected_device_registry_id,
            bytes,
            std::mem::align_of::<f32>() as u64,
            &format!("{label} key"),
        )?;
        validate_metal_buffer_view(
            self.value,
            expected_device_registry_id,
            bytes,
            std::mem::align_of::<f32>() as u64,
            &format!("{label} value"),
        )
    }
}

fn validate_device_chain_positions(
    sliding_max_positions: usize,
    full_max_positions: usize,
    target_kv_len: usize,
    proposal_position: usize,
    draft_k: usize,
) -> Result<usize> {
    if sliding_max_positions != full_max_positions {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP device KV capacities differ: sliding={sliding_max_positions} full={full_max_positions}"
        )));
    }
    if target_kv_len == 0 || target_kv_len > sliding_max_positions {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP verified target KV prefix {target_kv_len} is outside 1..={sliding_max_positions}"
        )));
    }
    if proposal_position < target_kv_len {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP proposal position {proposal_position} precedes verified target KV prefix {target_kv_len}"
        )));
    }
    let last_proposal_position = proposal_position
        .checked_add(draft_k - 1)
        .ok_or_else(|| invalid("resident-chain proposal position overflow"))?;
    if last_proposal_position >= sliding_max_positions {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP resident chain ends at proposal position {last_proposal_position}, outside cache capacity {sliding_max_positions}"
        )));
    }
    let last_local_base = last_proposal_position.saturating_sub(LOCAL_WINDOW + 1);
    if last_local_base >= target_kv_len {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "Gemma 4 12B MTP proposal position {last_proposal_position} has no verified sliding-KV row in prefix {target_kv_len}"
        )));
    }
    Ok(last_proposal_position)
}

/// Fully packed 12B assistant.  No target model, pager, MoE state, or verifier
/// is retained here; all 23 matrices live in one bounded Q4_0 Metal buffer.
/// The opt-in BF16 dense experiment owns a second buffer for the 22 dense
/// matrices, retaining the Q4 tied head and the established activation rounding.
pub struct Gemma4Mtp12AssistantMetal {
    single_position: bool,
    fuse: Mtp12FuseFlags,
    shortlist: Option<Mtp12Shortlist>,
    dense_bf16: Option<dense_bf16::Bf16Dense>,
    tree_state: Option<tree::TreeState>,
    packed_q4: Buffer,
    layout: Q4Layout,
    pipelines: Mtp12Pipelines,
    layers: [LayerWeights; N_LAYERS],
    final_norm: Buffer,
    scratch: Mtp12Scratch,
    queue: metal::CommandQueue,
    source_path: PathBuf,
    resident_ledger: Gemma4Mtp12ResidentLedger,
}

fn shared_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(4) as u64, MTLResourceOptions::StorageModeShared)
}

fn write_buffer_f32(buffer: &Buffer, values: &[f32]) -> Result<()> {
    write_buffer_f32_at(buffer, 0, values)
}

fn write_buffer_f32_at(buffer: &BufferRef, byte_offset: u64, values: &[f32]) -> Result<()> {
    let byte_len = std::mem::size_of_val(values);
    let end = (byte_offset as usize)
        .checked_add(byte_len)
        .ok_or_else(|| invalid("buffer write byte range overflow"))?;
    if end > buffer.length() as usize {
        return Err(invalid(format!(
            "buffer write of {byte_len} bytes at {byte_offset} exceeds {}",
            buffer.length()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>().add(byte_offset as usize),
            byte_len,
        );
    }
    Ok(())
}

fn read_buffer_f32(buffer: &Buffer, values: &mut [f32]) -> Result<()> {
    let byte_len = std::mem::size_of_val(values);
    if byte_len > buffer.length() as usize {
        return Err(invalid(format!(
            "buffer read of {byte_len} bytes exceeds {}",
            buffer.length()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().cast::<u8>(),
            values.as_mut_ptr().cast::<u8>(),
            byte_len,
        );
    }
    Ok(())
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16_rne_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(bias) >> 16) as u16
}

fn round_to_bf16_f32(value: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_rne_bits(value))
}

fn f32_buffer(device: &Device, values: &[f32]) -> Result<Buffer> {
    let buffer = shared_buffer(device, std::mem::size_of_val(values));
    write_buffer_f32(&buffer, values)?;
    Ok(buffer)
}

fn decode_bf16(mapping: &GgufWireMmap, tensor: &TensorEntry) -> Result<Vec<f32>> {
    let bytes = mapping.bytes(
        (PAYLOAD_FILE_OFFSET + tensor.start) as u64,
        tensor.end - tensor.start,
    )?;
    if !bytes.len().is_multiple_of(2) {
        return Err(invalid("BF16 tensor has an odd byte count"));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| bf16_bits_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
        .collect())
}

fn q4_0_matrix_bytes(tensor: TensorRef) -> Result<usize> {
    let rows = tensor.rows as usize;
    let cols = tensor.cols as usize;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(Q4_0_BLOCK_VALUES) {
        return Err(invalid(format!(
            "Q4_0 matrix {}x{} is empty or not divisible by {Q4_0_BLOCK_VALUES}",
            tensor.rows, tensor.cols
        )));
    }
    rows.checked_mul(cols / Q4_0_BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(Q4_0_BLOCK_BYTES))
        .ok_or_else(|| invalid("Q4_0 matrix byte size overflow"))
}

fn append_q4_0_layout(tensor: TensorRef, cursor: &mut u64) -> Result<Q4TensorRef> {
    let byte_len = q4_0_matrix_bytes(tensor)? as u64;
    let packed = Q4TensorRef {
        byte_offset: *cursor,
        byte_len,
        rows: tensor.rows,
        cols: tensor.cols,
    };
    *cursor = cursor
        .checked_add(byte_len)
        .ok_or_else(|| invalid("Q4_0 layout size overflow"))?;
    Ok(packed)
}

impl Q4Layout {
    fn build(
        embedding: TensorRef,
        layers: &[SourceLayerWeights; N_LAYERS],
        pre_projection: TensorRef,
        post_projection: TensorRef,
    ) -> Result<Self> {
        let mut cursor = 0u64;
        let embedding = append_q4_0_layout(embedding, &mut cursor)?;
        let mut packed_layers = [Q4LayerWeights::default(); N_LAYERS];
        for (destination, source) in packed_layers.iter_mut().zip(layers) {
            *destination = Q4LayerWeights {
                q: append_q4_0_layout(source.q, &mut cursor)?,
                o: append_q4_0_layout(source.o, &mut cursor)?,
                gate: append_q4_0_layout(source.gate, &mut cursor)?,
                up: append_q4_0_layout(source.up, &mut cursor)?,
                down: append_q4_0_layout(source.down, &mut cursor)?,
            };
        }
        let pre_projection = append_q4_0_layout(pre_projection, &mut cursor)?;
        let post_projection = append_q4_0_layout(post_projection, &mut cursor)?;
        if cursor != FULL_Q4_MATRIX_BYTES {
            return Err(invalid(format!(
                "official Q4_0 pack is {cursor} bytes, expected {FULL_Q4_MATRIX_BYTES}"
            )));
        }
        Ok(Self {
            embedding,
            layers: packed_layers,
            pre_projection,
            post_projection,
            matrix_bytes: cursor,
        })
    }

    fn pairs(
        &self,
        embedding: TensorRef,
        layers: &[SourceLayerWeights; N_LAYERS],
        pre_projection: TensorRef,
        post_projection: TensorRef,
    ) -> Vec<(TensorRef, Q4TensorRef)> {
        let mut pairs = Vec::with_capacity(23);
        pairs.push((embedding, self.embedding));
        for (source, packed) in layers.iter().zip(&self.layers) {
            pairs.extend([
                (source.q, packed.q),
                (source.o, packed.o),
                (source.gate, packed.gate),
                (source.up, packed.up),
                (source.down, packed.down),
            ]);
        }
        pairs.extend([
            (pre_projection, self.pre_projection),
            (post_projection, self.post_projection),
        ]);
        pairs
    }

    fn validate(
        &self,
        embedding: TensorRef,
        layers: &[SourceLayerWeights; N_LAYERS],
        pre_projection: TensorRef,
        post_projection: TensorRef,
        buffer_len: u64,
    ) -> Result<()> {
        let pairs = self.pairs(embedding, layers, pre_projection, post_projection);
        if pairs.len() != 23 {
            return Err(invalid(format!(
                "Q4_0 layout has {} matrices, expected 23",
                pairs.len()
            )));
        }
        let mut cursor = 0u64;
        for (index, (source, packed)) in pairs.iter().enumerate() {
            let expected_len = q4_0_matrix_bytes(*source)? as u64;
            if packed.byte_offset != cursor
                || packed.byte_len != expected_len
                || (packed.rows, packed.cols) != (source.rows, source.cols)
            {
                return Err(invalid(format!(
                    "Q4_0 matrix {index} does not exactly cover its source"
                )));
            }
            cursor += packed.byte_len;
        }
        if cursor != self.matrix_bytes || cursor != buffer_len {
            return Err(invalid(format!(
                "Q4_0 layout covers {cursor} bytes, records {}, buffer has {buffer_len}",
                self.matrix_bytes
            )));
        }
        Ok(())
    }
}

fn quantize_q4_0_row(input: &[u16], output: &mut [u8]) {
    debug_assert_eq!(input.len() % Q4_0_BLOCK_VALUES, 0);
    debug_assert_eq!(
        output.len(),
        input.len() / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES
    );
    for (block_in, block_out) in input
        .chunks_exact(Q4_0_BLOCK_VALUES)
        .zip(output.chunks_exact_mut(Q4_0_BLOCK_BYTES))
    {
        let mut values = [0.0f32; Q4_0_BLOCK_VALUES];
        let mut max_abs = 0.0f32;
        let mut signed_max = 0.0f32;
        for (destination, bits) in values.iter_mut().zip(block_in) {
            let value = bf16_bits_to_f32(u16::from_le(*bits));
            *destination = value;
            if value.abs() > max_abs {
                max_abs = value.abs();
                signed_max = value;
            }
        }
        // Canonical GGML Q4_0: the first max-magnitude value determines the
        // sign of d, and therefore which side receives code -8.
        let scale = signed_max / -8.0;
        let inverse = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        block_out[..2].copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
        for index in 0..16 {
            let low = (values[index] * inverse + 8.5).floor().clamp(0.0, 15.0) as u8;
            let high = (values[index + 16] * inverse + 8.5)
                .floor()
                .clamp(0.0, 15.0) as u8;
            block_out[2 + index] = low | (high << 4);
        }
    }
}

fn quantize_matrix_into_q4_0(
    mapping: &GgufWireMmap,
    source: TensorRef,
    destination: &Buffer,
    packed: Q4TensorRef,
) -> Result<()> {
    let rows = source.rows as usize;
    let cols = source.cols as usize;
    let row_bytes = cols / Q4_0_BLOCK_VALUES * Q4_0_BLOCK_BYTES;
    let expected_bytes = q4_0_matrix_bytes(source)? as u64;
    if (source.rows, source.cols) != (packed.rows, packed.cols)
        || packed.byte_len != expected_bytes
        || packed
            .byte_offset
            .checked_add(packed.byte_len)
            .is_none_or(|end| end > destination.length())
    {
        return Err(invalid("Q4_0 matrix destination is not exact and bounded"));
    }
    let input_bytes = mapping.bytes(source.absolute_offset, rows * cols * 2)?;
    let input =
        unsafe { std::slice::from_raw_parts(input_bytes.as_ptr().cast::<u16>(), rows * cols) };
    let output_address = destination.contents() as usize + packed.byte_offset as usize;
    use rayon::prelude::*;
    (0..rows).into_par_iter().for_each(|row| {
        let row_input = &input[row * cols..(row + 1) * cols];
        let row_output = unsafe {
            std::slice::from_raw_parts_mut((output_address + row * row_bytes) as *mut u8, row_bytes)
        };
        quantize_q4_0_row(row_input, row_output);
    });
    Ok(())
}

fn pack_all_q4_0(
    device: &Device,
    mapping: &GgufWireMmap,
    layout: &Q4Layout,
    embedding: TensorRef,
    layers: &[SourceLayerWeights; N_LAYERS],
    pre_projection: TensorRef,
    post_projection: TensorRef,
) -> Result<(Buffer, u128)> {
    layout.validate(
        embedding,
        layers,
        pre_projection,
        post_projection,
        layout.matrix_bytes,
    )?;
    let buffer = shared_buffer(device, layout.matrix_bytes as usize);
    let started = Instant::now();
    for (source, packed) in layout.pairs(embedding, layers, pre_projection, post_projection) {
        quantize_matrix_into_q4_0(mapping, source, &buffer, packed)?;
    }
    layout.validate(
        embedding,
        layers,
        pre_projection,
        post_projection,
        buffer.length(),
    )?;
    Ok((buffer, started.elapsed().as_micros()))
}

const MTP12_SHORTLIST_CLUSTERS: usize = 2048;

struct Mtp12Shortlist {
    centroids: Buffer,
    token_clusters: Buffer,
    scores: Buffer,
    selected: Buffer,
    top: u32,
}

/// A/B selector for the bit-exact retained-row local compaction kernel.
fn mtp12_shortlist_compact_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_MTP12_SHORTLIST_COMPACT")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

fn mtp12_shortlist_top(value: Option<&str>) -> Result<u32> {
    match value {
        None => Ok(192),
        Some(text) => text.parse::<u32>()
            .ok().filter(|top| (1..=MTP12_SHORTLIST_CLUSTERS as u32).contains(top))
            .ok_or_else(|| invalid("CAMELID_GEMMA4_MTP12_SHORTLIST_TOP must be in 1..=2048")),
    }
}

impl Mtp12Shortlist {
    fn from_env(device: &Device) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("CAMELID_GEMMA4_MTP12_SHORTLIST") else {
            return Ok(None);
        };
        let data = shortlist::read_sidecar(Path::new(&path))?;
        let top_text = std::env::var("CAMELID_GEMMA4_MTP12_SHORTLIST_TOP")
            .map(Some).or_else(|error| match error {
                std::env::VarError::NotPresent => Ok(None),
                _ => Err(invalid("CAMELID_GEMMA4_MTP12_SHORTLIST_TOP is not Unicode")),
            })?;
        let top = mtp12_shortlist_top(top_text.as_deref())?;
        let token_clusters = shared_buffer(device, data.token_clusters.len() * 2);
        unsafe {
            std::ptr::copy_nonoverlapping(data.token_clusters.as_ptr(),
                token_clusters.contents().cast::<u16>(), data.token_clusters.len());
        }
        eprintln!("[gemma4-mtp12] draft head shortlist top={top}/2048 path={}", Path::new(&path).display());
        Ok(Some(Self {
            centroids: f32_buffer(device, &data.centroids)?,
            token_clusters,
            scores: shared_buffer(device, MTP12_SHORTLIST_CLUSTERS * 4),
            selected: shared_buffer(device, MTP12_SHORTLIST_CLUSTERS * 4),
            top,
        }))
    }

    fn byte_len(&self) -> u64 {
        self.centroids.length() + self.token_clusters.length()
            + self.scores.length() + self.selected.length()
    }
}

impl Mtp12Scratch {
    fn new(device: &Device) -> Self {
        let f32s = |count: usize| shared_buffer(device, count * std::mem::size_of::<f32>());
        Self {
            pre_input: f32s(TARGET_HIDDEN * 2),
            hidden: f32s(ASSISTANT_HIDDEN),
            normed: f32s(ASSISTANT_HIDDEN),
            query: f32s(N_HEADS * FULL_HEAD_DIM),
            query_normed: f32s(N_HEADS * FULL_HEAD_DIM),
            context: f32s(N_HEADS * FULL_HEAD_DIM),
            attention_projection: f32s(ASSISTANT_HIDDEN),
            attention_normalized: f32s(ASSISTANT_HIDDEN),
            attention_residual: f32s(ASSISTANT_HIDDEN),
            gate: f32s(FFN_HIDDEN),
            up: f32s(FFN_HIDDEN),
            gated: f32s(FFN_HIDDEN),
            down: f32s(ASSISTANT_HIDDEN),
            down_normalized: f32s(ASSISTANT_HIDDEN),
            final_normalized: f32s(ASSISTANT_HIDDEN),
            recurrent_hidden: f32s(TARGET_HIDDEN),
            chain_initial_recurrent_hidden: f32s(TARGET_HIDDEN),
            chain_recurrent_hidden: f32s(MTP12_CHAIN_RECURRENT_ELEMENTS),
            chain_final_normalized: f32s(MTP12_CHAIN_MAX_DRAFTS * ASSISTANT_HIDDEN),
            logits: f32s(VOCAB),
            // Slot zero holds either the K=1 result or a chain anchor.  Chain
            // draft tokens occupy slots 1..=15 so each can feed the next step.
            output_token: shared_buffer(
                device,
                MTP12_CHAIN_TOKEN_SLOTS * std::mem::size_of::<u32>(),
            ),
            argmax_partial_values: f32s(MTP12_ARGMAX_MAX_PARTIALS),
            argmax_partial_ids: shared_buffer(
                device,
                MTP12_ARGMAX_MAX_PARTIALS * std::mem::size_of::<u32>(),
            ),
            // Row zero remains the K=1 scratch.  The resident chain binds one
            // precomputed RoPE row per advancing proposal position.
            local_cos: f32s(MTP12_CHAIN_LOCAL_ROPE_ELEMENTS),
            local_sin: f32s(MTP12_CHAIN_LOCAL_ROPE_ELEMENTS),
            full_cos: f32s(MTP12_CHAIN_FULL_ROPE_ELEMENTS),
            full_sin: f32s(MTP12_CHAIN_FULL_ROPE_ELEMENTS),
            attention_stats: f32s(N_HEADS * 2),
        }
    }

    fn byte_len(&self) -> u64 {
        [
            &self.pre_input,
            &self.hidden,
            &self.normed,
            &self.query,
            &self.query_normed,
            &self.context,
            &self.attention_projection,
            &self.attention_normalized,
            &self.attention_residual,
            &self.gate,
            &self.up,
            &self.gated,
            &self.down,
            &self.down_normalized,
            &self.final_normalized,
            &self.recurrent_hidden,
            &self.chain_initial_recurrent_hidden,
            &self.chain_recurrent_hidden,
            &self.chain_final_normalized,
            &self.logits,
            &self.output_token,
            &self.argmax_partial_values,
            &self.argmax_partial_ids,
            &self.local_cos,
            &self.local_sin,
            &self.full_cos,
            &self.full_sin,
            &self.attention_stats,
        ]
        .iter()
        .map(|buffer| buffer.length())
        .sum()
    }
}

impl Gemma4Mtp12AssistantMetal {
    /// Load from `$CAMELID_MODEL_ROOT` (or `~/models`) under the exact official
    /// repository directory name.
    pub fn load_staged_official() -> Result<Self> {
        Self::load(&official_staged_assistant_path())
    }

    /// Admit only the exact official 12B QAT assistant.  File length, complete
    /// tensor table, config semantics/config SHA, and model SHA are all checked
    /// before any matrix reaches the resident Q4_0 pack.
    pub fn load(path: &Path) -> Result<Self> {
        let load_started = Instant::now();
        let dense_bf16_enabled = dense_bf16::enabled()?;
        validate_official_config(path)?;
        let mapping = GgufWireMmap::map(path)?;
        mapping.advise_sequential();
        mapping.advise_willneed();
        let manifest = parse_official_manifest(&mapping)?;

        let hash_started = Instant::now();
        let actual_hash = sha256_hex(mapping.bytes(0, EXPECTED_FILE_BYTES as usize)?);
        let hash_us = hash_started.elapsed().as_micros();
        if actual_hash != GEMMA4_12B_MTP_ASSISTANT_SHA256 {
            return Err(invalid(format!(
                "model SHA-256 {actual_hash} does not match expected {GEMMA4_12B_MTP_ASSISTANT_SHA256}"
            )));
        }

        let kernel = super::metal_linear_kernel()
            .ok_or_else(|| invalid("Metal common core is unavailable"))?;
        let device = &kernel.device;
        let embedding = manifest.matrix("model.embed_tokens.weight")?;
        let pre_projection = manifest.matrix("pre_projection.weight")?;
        let post_projection = manifest.matrix("post_projection.weight")?;
        let mut source_layers = Vec::with_capacity(N_LAYERS);
        for layer in 0..N_LAYERS {
            let prefix = format!("model.layers.{layer}");
            source_layers.push(SourceLayerWeights {
                q: manifest.matrix(&format!("{prefix}.self_attn.q_proj.weight"))?,
                o: manifest.matrix(&format!("{prefix}.self_attn.o_proj.weight"))?,
                gate: manifest.matrix(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up: manifest.matrix(&format!("{prefix}.mlp.up_proj.weight"))?,
                down: manifest.matrix(&format!("{prefix}.mlp.down_proj.weight"))?,
            });
        }
        let source_layers: [SourceLayerWeights; N_LAYERS] = source_layers
            .try_into()
            .map_err(|_| invalid("internal assistant layer table is not four layers"))?;
        let layout = Q4Layout::build(embedding, &source_layers, pre_projection, post_projection)?;

        let norm = |name: &str| -> Result<Buffer> {
            f32_buffer(device, &decode_bf16(&mapping, manifest.tensor(name)?)?)
        };
        let mut layers = Vec::with_capacity(N_LAYERS);
        for layer in 0..N_LAYERS {
            let prefix = format!("model.layers.{layer}");
            let scale_values = decode_bf16(
                &mapping,
                manifest.tensor(&format!("{prefix}.layer_scalar"))?,
            )?;
            if scale_values.len() != 1 || !scale_values[0].is_finite() {
                return Err(invalid(format!("layer {layer} scalar is invalid")));
            }
            layers.push(LayerWeights {
                input_norm: norm(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: norm(&format!("{prefix}.post_attention_layernorm.weight"))?,
                pre_feedforward_norm: norm(&format!("{prefix}.pre_feedforward_layernorm.weight"))?,
                post_feedforward_norm: norm(&format!(
                    "{prefix}.post_feedforward_layernorm.weight"
                ))?,
                q_norm: norm(&format!("{prefix}.self_attn.q_norm.weight"))?,
                matrices: layout.layers[layer],
                scale: scale_values[0],
            });
        }
        let final_norm = norm("model.norm.weight")?;
        let decoded_norm_bytes = layers
            .iter()
            .map(|layer| {
                layer.input_norm.length()
                    + layer.post_attention_norm.length()
                    + layer.pre_feedforward_norm.length()
                    + layer.post_feedforward_norm.length()
                    + layer.q_norm.length()
            })
            .sum::<u64>()
            + final_norm.length();

        let (packed_q4, quantize_us) = pack_all_q4_0(
            device,
            &mapping,
            &layout,
            embedding,
            &source_layers,
            pre_projection,
            post_projection,
        )?;
        let dense_bf16 = dense_bf16_enabled.then(|| dense_bf16::Bf16Dense::load(
            device, &mapping,
            layout.pairs(embedding, &source_layers, pre_projection, post_projection),
            layout.embedding,
        )).transpose()?;
        // Every runtime matrix and norm now owns independent Metal storage.
        // Releasing the source is part of the 16 GB admission contract.
        drop(mapping);

        let pipeline_started = Instant::now();
        let pipelines = Mtp12Pipelines::new(device)?;
        let pipeline_compile_us = pipeline_started.elapsed().as_micros()
            + dense_bf16.as_ref().map_or(0, |dense| dense.pipeline_compile_us);
        let scratch = Mtp12Scratch::new(device);
        let shortlist = Mtp12Shortlist::from_env(device)?;
        let shortlist_bytes = shortlist.as_ref().map_or(0, Mtp12Shortlist::byte_len);
        let resident_ledger = Gemma4Mtp12ResidentLedger {
            source_file_bytes: EXPECTED_FILE_BYTES,
            source_sha256_verified: true,
            source_mapping_retained: false,
            packed_q4_matrix_bytes: packed_q4.length(),
            dense_bf16_enabled,
            dense_bf16_matrix_bytes: dense_bf16.as_ref().map_or(0, |dense| dense.byte_len()),
            dense_bf16_upload_us: dense_bf16.as_ref().map_or(0, |dense| dense.upload_us),
            decoded_norm_bytes,
            fixed_scratch_bytes: scratch.byte_len() + shortlist_bytes,
            quantize_us,
            hash_us,
            pipeline_compile_us,
            load_wall_us: load_started.elapsed().as_micros(),
        };
        let layers: [LayerWeights; N_LAYERS] = layers
            .try_into()
            .map_err(|_| invalid("internal resident layer table is not four layers"))?;
        Ok(Self {
            single_position: std::env::var("CAMELID_GEMMA4_MTP12_SINGLE_POSITION")
                .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),
            fuse: mtp12_fuse_flags()?,
            shortlist,
            dense_bf16,
            tree_state: None,
            packed_q4,
            layout,
            pipelines,
            layers,
            final_norm,
            scratch,
            queue: device.new_command_queue(),
            source_path: path.to_path_buf(),
            resident_ledger,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn resident_ledger(&self) -> Gemma4Mtp12ResidentLedger {
        self.resident_ledger
    }

    /// Isolated K=1 assistant forward.  This is deliberately a parity scaffold:
    /// it uploads scoped CPU inputs/KV, captures full logits, and reads recurrent
    /// hidden.  The target-runtime port should preserve this as an oracle while
    /// replacing the copies/readback with borrowed device buffers and token-only
    /// output.
    pub fn propose_k1(
        &mut self,
        target_scaled_embedding: &[f32],
        pending_target_hidden: &[f32],
        sliding: Gemma4Mtp12CpuKv<'_>,
        full: Gemma4Mtp12CpuKv<'_>,
    ) -> Result<Gemma4Mtp12Proposal> {
        let wall_started = Instant::now();
        if target_scaled_embedding.len() != TARGET_HIDDEN
            || pending_target_hidden.len() != TARGET_HIDDEN
        {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP input widths are embedding={} hidden={}, expected {TARGET_HIDDEN}",
                target_scaled_embedding.len(),
                pending_target_hidden.len()
            )));
        }
        if target_scaled_embedding
            .iter()
            .chain(pending_target_hidden)
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 12B MTP input contains non-finite values".into(),
            ));
        }
        sliding.validate(LOCAL_KV_HEADS, LOCAL_HEAD_DIM, "sliding")?;
        full.validate(FULL_KV_HEADS, FULL_HEAD_DIM, "full")?;
        if sliding.kv_len == 0 || sliding.kv_len != full.kv_len {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP shared KV lengths are sliding={} full={}; expected equal non-zero prefixes",
                sliding.kv_len, full.kv_len
            )));
        }
        let logical_len = sliding.kv_len;

        let upload_started = Instant::now();
        let mut pre_input = Vec::with_capacity(TARGET_HIDDEN * 2);
        pre_input.extend(
            target_scaled_embedding
                .iter()
                .copied()
                .map(round_to_bf16_f32),
        );
        pre_input.extend(pending_target_hidden.iter().copied().map(round_to_bf16_f32));
        write_buffer_f32(&self.scratch.pre_input, &pre_input)?;
        write_rope_tables(logical_len, &self.scratch)?;

        let kernel =
            super::metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        let device = &kernel.device;
        let sliding_key = f32_buffer(device, sliding.key)?;
        let sliding_value = f32_buffer(device, sliding.value)?;
        let full_key = f32_buffer(device, full.key)?;
        let full_value = f32_buffer(device, full.value)?;
        let score_elements = N_HEADS
            .checked_mul(logical_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let attention_scores = shared_buffer(
            device,
            score_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| invalid("attention score byte size overflow"))?,
        );
        let upload_us = upload_started.elapsed().as_micros();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        self.encode_dense_gemv(
            encoder,
            &self.scratch.pre_input,
            &self.scratch.hidden,
            self.layout.pre_projection,
            true,
        );
        for layer_index in 0..N_LAYERS {
            let is_sliding = layer_index < 3;
            let (key, value, kv_heads, head_dim, cos, sin) = if is_sliding {
                (
                    &sliding_key,
                    &sliding_value,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    &self.scratch.local_cos,
                    &self.scratch.local_sin,
                )
            } else {
                (
                    &full_key,
                    &full_value,
                    FULL_KV_HEADS,
                    FULL_HEAD_DIM,
                    &self.scratch.full_cos,
                    &self.scratch.full_sin,
                )
            };
            let compact_base = if is_sliding {
                logical_len.saturating_sub(LOCAL_WINDOW + 1)
            } else {
                0
            };
            let position_count = logical_len - compact_base;
            self.encode_layer_k1(
                encoder,
                layer_index,
                key,
                value,
                kv_heads,
                head_dim,
                logical_len,
                compact_base,
                position_count,
                cos,
                sin,
                &attention_scores,
            );
        }
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.hidden,
            &self.final_norm,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
            1,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.final_normalized,
            &self.scratch.recurrent_hidden,
            self.layout.post_projection,
            true,
        );
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
            &self.scratch.final_normalized,
            &self.scratch.logits,
            self.layout.embedding,
            false,
        );
        self.encode_vocab_argmax(encoder, 0);
        encoder.end_encoding();
        let encode_us = encode_started.elapsed().as_micros();
        command_buffer.commit();
        let wait_started = Instant::now();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "Metal command buffer ended with status {:?}",
                command_buffer.status()
            )));
        }
        let token = unsafe { *self.scratch.output_token.contents().cast::<u32>() };
        if token as usize >= VOCAB {
            return Err(invalid(format!(
                "Metal argmax returned invalid token {token}"
            )));
        }
        let mut logits = vec![0.0f32; VOCAB];
        let mut recurrent_hidden = vec![0.0f32; TARGET_HIDDEN];
        read_buffer_f32(&self.scratch.logits, &mut logits)?;
        read_buffer_f32(&self.scratch.recurrent_hidden, &mut recurrent_hidden)?;
        Ok(Gemma4Mtp12Proposal {
            token,
            logits,
            recurrent_hidden,
            timing: Gemma4Mtp12ProposalTiming {
                upload_us,
                encode_us,
                wait_us,
                wall_us: wall_started.elapsed().as_micros(),
            },
        })
    }

    /// Device-fed K=1 companion to [`Self::propose_k1`].  The selected target
    /// Q6_K embedding row and both shared target KV sources remain in their
    /// caller-owned Metal buffers for the complete synchronous call.  Only the
    /// pending normalized target hidden (3,840 f32 values) and small RoPE
    /// tables cross the CPU boundary; no embedding or KV vector is uploaded.
    ///
    /// `logical_position` is both the 0-based position being proposed and the
    /// number of already materialized target KV rows (`0..logical_position`).
    /// K=1 therefore admits `1..max_positions`, leaving the proposal position
    /// itself inside the target cache capacity.  The selected Q6_K row view may
    /// alias a larger full-table no-copy buffer: its offset and exact 3,150-byte
    /// window are checked, but the row is never staged into another wire buffer.
    pub fn propose_k1_device(
        &mut self,
        target_q6k_embedding: Gemma4Mtp12Q6KEmbeddingRow<'_>,
        pending_target_hidden: &[f32],
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        logical_position: usize,
    ) -> Result<Gemma4Mtp12Proposal> {
        self.propose_k1_device_at_position(
            target_q6k_embedding,
            pending_target_hidden,
            sliding,
            full,
            logical_position,
            logical_position,
        )
    }

    /// Composable K=1 oracle for a draft after the verified target prefix.
    /// Unlike [`Self::propose_k1_device`], query RoPE position and readable
    /// target-KV prefix length are explicit and independently checked.  This is
    /// the exact fallback/oracle used to admit the one-command resident chain.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_k1_device_at_position(
        &mut self,
        target_q6k_embedding: Gemma4Mtp12Q6KEmbeddingRow<'_>,
        pending_target_hidden: &[f32],
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        target_kv_len: usize,
        proposal_position: usize,
    ) -> Result<Gemma4Mtp12Proposal> {
        let wall_started = Instant::now();
        if pending_target_hidden.len() != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP pending hidden width is {}, expected {TARGET_HIDDEN}",
                pending_target_hidden.len()
            )));
        }
        if pending_target_hidden.iter().any(|value| !value.is_finite()) {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 12B MTP pending hidden contains non-finite values".into(),
            ));
        }

        let expected_device_registry_id = self.packed_q4.device().registry_id();
        target_q6k_embedding.validate(expected_device_registry_id)?;
        sliding.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )?;
        full.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_FULL_HOST_LAYER,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            "full",
        )?;
        validate_device_chain_positions(
            sliding.max_positions,
            full.max_positions,
            target_kv_len,
            proposal_position,
            1,
        )?;
        let logical_len = target_kv_len;

        let upload_started = Instant::now();
        let pending_bf16: Vec<f32> = pending_target_hidden
            .iter()
            .copied()
            .map(round_to_bf16_f32)
            .collect();
        write_buffer_f32_at(
            &self.scratch.pre_input,
            (TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
            &pending_bf16,
        )?;
        write_rope_tables(proposal_position, &self.scratch)?;

        let kernel =
            super::metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        if kernel.device.registry_id() != expected_device_registry_id {
            return Err(invalid(
                "Metal common core changed devices during K=1 proposal",
            ));
        }
        let score_elements = N_HEADS
            .checked_mul(logical_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let attention_scores = shared_buffer(
            &kernel.device,
            score_elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| invalid("attention score byte size overflow"))?,
        );
        let upload_us = upload_started.elapsed().as_micros();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        encode_q6k_embedding_gather(
            encoder,
            &self.pipelines.gather_q6k_embedding,
            target_q6k_embedding,
            &self.scratch.pre_input,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.pre_input,
            &self.scratch.hidden,
            self.layout.pre_projection,
            true,
        );
        for layer_index in 0..N_LAYERS {
            let is_sliding = layer_index < 3;
            let (source, kv_heads, head_dim, cos, sin) = if is_sliding {
                (
                    sliding,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    &self.scratch.local_cos,
                    &self.scratch.local_sin,
                )
            } else {
                (
                    full,
                    FULL_KV_HEADS,
                    FULL_HEAD_DIM,
                    &self.scratch.full_cos,
                    &self.scratch.full_sin,
                )
            };
            let compact_base = if is_sliding {
                proposal_position
                    .saturating_sub(LOCAL_WINDOW + 1)
                    .min(target_kv_len)
            } else {
                0
            };
            let position_count = logical_len - compact_base;
            self.encode_layer_k1_from_views(
                encoder,
                layer_index,
                source.key.buffer,
                source.value.buffer,
                source.key.byte_offset,
                source.value.byte_offset,
                source.max_positions,
                kv_heads,
                head_dim,
                logical_len,
                compact_base,
                position_count,
                cos,
                sin,
                0,
                &attention_scores,
            );
        }
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.hidden,
            &self.final_norm,
            &self.scratch.final_normalized,
            ASSISTANT_HIDDEN,
            1,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.final_normalized,
            &self.scratch.recurrent_hidden,
            self.layout.post_projection,
            true,
        );
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
            &self.scratch.final_normalized,
            &self.scratch.logits,
            self.layout.embedding,
            false,
        );
        self.encode_vocab_argmax(encoder, 0);
        encoder.end_encoding();
        let encode_us = encode_started.elapsed().as_micros();
        command_buffer.commit();
        let wait_started = Instant::now();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "Metal command buffer ended with status {:?}",
                command_buffer.status()
            )));
        }
        let token = unsafe { *self.scratch.output_token.contents().cast::<u32>() };
        if token as usize >= VOCAB {
            return Err(invalid(format!(
                "Metal argmax returned invalid token {token}"
            )));
        }
        let mut logits = vec![0.0f32; VOCAB];
        let mut recurrent_hidden = vec![0.0f32; TARGET_HIDDEN];
        read_buffer_f32(&self.scratch.logits, &mut logits)?;
        read_buffer_f32(&self.scratch.recurrent_hidden, &mut recurrent_hidden)?;
        Ok(Gemma4Mtp12Proposal {
            token,
            logits,
            recurrent_hidden,
            timing: Gemma4Mtp12ProposalTiming {
                upload_us,
                encode_us,
                wait_us,
                wall_us: wall_started.elapsed().as_micros(),
            },
        })
    }

    /// Lossless one-command device-fed draft chain for the exact 12B target.
    ///
    /// `draft_k` is admitted only in 1..=15. Slot zero of resident token
    /// scratch is initialized with `anchor_token`; each step gathers its target
    /// Q6_K row from the single borrowed full-table alias, consumes the prior
    /// recurrent hidden directly from Metal, and writes the next token/recurrent
    /// row for the following step.  The fixed target layer-46/47 KV prefix is
    /// shared across all steps, matching repeated [`Self::propose_k1_device`].
    /// By default `proposal_position` advances for RoPE. The opt-in
    /// `CAMELID_GEMMA4_MTP12_SINGLE_POSITION=1` keeps every query at the round
    /// anchor, as required by HF SinglePositionMultiTokenCandidateGenerator.
    /// `target_kv_len` stays fixed in either mode;
    /// unverified physical cache rows at and beyond that prefix are never read.
    ///
    /// The production return crosses only `draft_k * sizeof(u32)` bytes back to
    /// the CPU after one final wait.  Per-step recurrent rows are retained in
    /// assistant Metal scratch for exact tests and are never part of this public
    /// CPU result.  No borrowed buffer reference escapes the synchronous call.
    #[allow(clippy::too_many_arguments)]
    /// Vocabulary argmax of `scratch.logits` into `scratch.output_token` at
    /// `output_byte_offset`.  Chunked two-stage kernel by default;
    /// `CAMELID_GEMMA4_MTP12_ARGMAX_LEGACY=1` keeps the single-threadgroup scan.
    fn encode_vocab_argmax(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        output_byte_offset: u64,
    ) {
        if mtp12_argmax_legacy_enabled() {
            encode_argmax_at_offset(
                encoder,
                &self.pipelines.argmax,
                &self.scratch.logits,
                &self.scratch.output_token,
                output_byte_offset,
                VOCAB,
            );
        } else {
            encode_argmax_chunked_at_offset(
                encoder,
                &self.pipelines,
                &self.scratch.logits,
                &self.scratch.argmax_partial_values,
                &self.scratch.argmax_partial_ids,
                &self.scratch.output_token,
                output_byte_offset,
                VOCAB,
            );
        }
    }

    /// Only the resident draft chain uses the optional shortlist. The K=1
    /// full-logit APIs remain full-head parity oracles.
    fn encode_draft_head(&self, encoder: &metal::ComputeCommandEncoderRef) {
        let pipeline = if let Some(shortlist) = &self.shortlist {
            encode_shortlist(encoder, &self.pipelines, shortlist, &self.scratch.final_normalized);
            encoder.set_buffer(7, Some(&shortlist.token_clusters), 0);
            encoder.set_buffer(8, Some(&shortlist.selected), 0);
            if mtp12_shortlist_compact_enabled() {
                let compact = match self.fuse.head_prefetch {
                    0 => &self.pipelines.q4_gemv_shortlist_compact,
                    1 => &self.pipelines.q4_gemv_shortlist_compact_prefetch,
                    _ => &self.pipelines.q4_gemv_shortlist_compact_staged,
                };
                encode_q4_gemv_shortlist_compact(
                    encoder, compact,
                    &self.packed_q4, &self.scratch.final_normalized,
                    &self.scratch.logits, self.layout.embedding,
                );
                return;
            }
            &self.pipelines.q4_gemv_shortlist
        } else {
            &self.pipelines.q4_gemv
        };
        encode_q4_gemv(encoder, pipeline, &self.packed_q4,
            &self.scratch.final_normalized, &self.scratch.logits,
            self.layout.embedding, false);
    }

    pub fn propose_chain_device_resident(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: Gemma4Mtp12DeviceRecurrentHidden<'_>,
        target_q6k_embedding: Gemma4Mtp12Q6KEmbeddingTable<'_>,
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        target_kv_len: usize,
        proposal_position: usize,
        draft_k: usize,
    ) -> Result<Gemma4Mtp12ChainProposal> {
        let wall_started = Instant::now();
        if !mtp12_chain_draft_k_admitted(draft_k) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP resident draft K is {draft_k}, expected 1..={MTP12_CHAIN_MAX_DRAFTS}"
            )));
        }
        if anchor_token as usize >= VOCAB {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP anchor token {anchor_token} exceeds vocab {VOCAB}"
            )));
        }

        let expected_device_registry_id = self.packed_q4.device().registry_id();
        initial_recurrent_hidden.validate(expected_device_registry_id)?;
        target_q6k_embedding.validate(expected_device_registry_id)?;
        sliding.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )?;
        full.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_FULL_HOST_LAYER,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            "full",
        )?;
        validate_device_chain_positions(
            sliding.max_positions,
            full.max_positions,
            target_kv_len,
            proposal_position,
            draft_k,
        )?;

        let prepare_started = Instant::now();
        write_chain_rope_tables(proposal_position, draft_k, self.single_position, &self.scratch)?;
        unsafe {
            *self.scratch.output_token.contents().cast::<u32>() = anchor_token;
        }
        let kernel =
            super::metal_linear_kernel().ok_or_else(|| invalid("Metal common core disappeared"))?;
        if kernel.device.registry_id() != expected_device_registry_id {
            return Err(invalid(
                "assistant queue and Metal common core use different devices",
            ));
        }
        let score_elements = N_HEADS
            .checked_mul(target_kv_len)
            .ok_or_else(|| invalid("attention score size overflow"))?;
        let score_bytes = score_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| invalid("attention score byte size overflow"))?;
        let attention_scores = shared_buffer(&kernel.device, score_bytes);
        let cpu_prepare_us = prepare_started.elapsed().as_micros();
        let dump_queries = mtp12_dump_draft_queries_path();
        if let Some(path) = dump_queries {
            if crate::gemma4_runtime::mtp12_snapshot::enabled_path().is_some() {
                let seed = unsafe {
                    std::slice::from_raw_parts(
                        initial_recurrent_hidden.values.buffer.contents().cast::<u8>()
                            .add(initial_recurrent_hidden.values.byte_offset as usize).cast::<f32>(),
                        TARGET_HIDDEN,
                    )
                };
                crate::gemma4_runtime::mtp12_snapshot::record_initial_seed(
                    path, anchor_token, proposal_position, target_kv_len, draft_k, seed,
                )?;
            }
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encode_started = Instant::now();
        for step in 0..draft_k {
            let (recurrent_buffer, recurrent_byte_offset): (&BufferRef, u64) = if step == 0 {
                (
                    initial_recurrent_hidden.values.buffer,
                    initial_recurrent_hidden.values.byte_offset,
                )
            } else {
                (&self.scratch.recurrent_hidden, 0)
            };
            encode_q6k_embedding_and_recurrent_gather(
                encoder,
                &self.pipelines.gather_q6k_embedding_and_recurrent,
                &self.scratch.output_token,
                step,
                target_q6k_embedding,
                recurrent_buffer,
                recurrent_byte_offset,
                &self.scratch.pre_input,
            );
            self.encode_dense_gemv(
                encoder,
                &self.scratch.pre_input,
                &self.scratch.hidden,
                self.layout.pre_projection,
                true,
            );
            for layer_index in 0..N_LAYERS {
                let is_sliding = layer_index < 3;
                let (source, kv_heads, head_dim, cos, sin, rope_byte_offset) = if is_sliding {
                    (
                        sliding,
                        LOCAL_KV_HEADS,
                        LOCAL_HEAD_DIM,
                        &self.scratch.local_cos,
                        &self.scratch.local_sin,
                        (step * LOCAL_HEAD_DIM / 2 * std::mem::size_of::<f32>()) as u64,
                    )
                } else {
                    (
                        full,
                        FULL_KV_HEADS,
                        FULL_HEAD_DIM,
                        &self.scratch.full_cos,
                        &self.scratch.full_sin,
                        (step * FULL_HEAD_DIM / 2 * std::mem::size_of::<f32>()) as u64,
                    )
                };
                let step_proposal_position =
                    chain_query_position(proposal_position, step, self.single_position);
                let compact_base = if is_sliding {
                    step_proposal_position
                        .saturating_sub(LOCAL_WINDOW + 1)
                        .min(target_kv_len)
                } else {
                    0
                };
                let position_count = target_kv_len - compact_base;
                self.encode_layer_k1_from_views(
                    encoder,
                    layer_index,
                    source.key.buffer,
                    source.value.buffer,
                    source.key.byte_offset,
                    source.value.byte_offset,
                    source.max_positions,
                    kv_heads,
                    head_dim,
                    target_kv_len,
                    compact_base,
                    position_count,
                    cos,
                    sin,
                    rope_byte_offset,
                    &attention_scores,
                );
            }
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.hidden,
                &self.final_norm,
                &self.scratch.final_normalized,
                ASSISTANT_HIDDEN,
                1,
            );
            if dump_queries.is_some() {
                // Keep this step's head query (the vector the tied head is
                // dotted with) for the offline shortlist-recall experiment.
                encode_copy_f32_to_offset(
                    encoder,
                    &self.pipelines.copy_f32,
                    &self.scratch.final_normalized,
                    &self.scratch.chain_final_normalized,
                    (step * ASSISTANT_HIDDEN * std::mem::size_of::<f32>()) as u64,
                    ASSISTANT_HIDDEN,
                );
            }
            self.encode_dense_gemv(
                encoder,
                &self.scratch.final_normalized,
                &self.scratch.recurrent_hidden,
                self.layout.post_projection,
                true,
            );
            encode_copy_f32_to_offset(
                encoder,
                &self.pipelines.copy_f32,
                &self.scratch.recurrent_hidden,
                &self.scratch.chain_recurrent_hidden,
                (step * TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64,
                TARGET_HIDDEN,
            );
            self.encode_draft_head(encoder);
            self.encode_vocab_argmax(
                encoder,
                ((step + 1) * std::mem::size_of::<u32>()) as u64,
            );
        }
        encoder.end_encoding();
        let encode_us = encode_started.elapsed().as_micros();

        command_buffer.commit();
        let wait_started = Instant::now();
        command_buffer.wait_until_completed();
        let wait_us = wait_started.elapsed().as_micros();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(invalid(format!(
                "resident-chain Metal command buffer ended with status {:?}",
                command_buffer.status()
            )));
        }
        let (gpu_us, kernel_us) = super::command_buffer_gpu_times_us(&command_buffer.to_owned());
        let token_ptr = self.scratch.output_token.contents().cast::<u32>();
        let tokens = unsafe { std::slice::from_raw_parts(token_ptr.add(1), draft_k) }.to_vec();
        for (step, token) in tokens.iter().copied().enumerate() {
            if token as usize >= VOCAB {
                return Err(invalid(format!(
                    "resident-chain argmax step {step} returned invalid token {token}"
                )));
            }
        }
        if let Some(path) = dump_queries {
            // One 4,112-byte record per draft: u32 token, u32 step, u32
            // absolute draft sequence position (not the frozen RoPE anchor),
            // u32 draft_k, then the 1,024 f32 head query. Append-only; a write
            // failure only loses the diagnostic.
            use std::io::Write;
            let queries = unsafe {
                std::slice::from_raw_parts(
                    self.scratch.chain_final_normalized.contents().cast::<f32>(),
                    draft_k * ASSISTANT_HIDDEN,
                )
            };
            let mut record = Vec::with_capacity(draft_k * (16 + ASSISTANT_HIDDEN * 4));
            for (step, token) in tokens.iter().copied().enumerate() {
                record.extend_from_slice(&token.to_le_bytes());
                record.extend_from_slice(&(step as u32).to_le_bytes());
                record.extend_from_slice(&((proposal_position + step) as u32).to_le_bytes());
                record.extend_from_slice(&(draft_k as u32).to_le_bytes());
                for value in &queries[step * ASSISTANT_HIDDEN..(step + 1) * ASSISTANT_HIDDEN] {
                    record.extend_from_slice(&value.to_le_bytes());
                }
            }
            if let Err(error) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut file| file.write_all(&record))
            {
                eprintln!("[gemma4-mtp12] draft query dump to {} failed: {error}", path.display());
            }
        }

        let mut total_target_kv_read_bytes = 0u64;
        for step in 0..draft_k {
            let compact_base = chain_query_position(proposal_position, step, self.single_position)
                .saturating_sub(LOCAL_WINDOW + 1)
                .min(target_kv_len);
            total_target_kv_read_bytes = total_target_kv_read_bytes
                .checked_add(target_kv_read_bytes(
                    target_kv_len - compact_base,
                    target_kv_len,
                )?)
                .ok_or_else(|| invalid("resident-chain target KV read ledger overflow"))?;
        }
        let target_kv_alias_bytes = sliding
            .key
            .byte_len
            .checked_add(sliding.value.byte_len)
            .and_then(|bytes| bytes.checked_add(full.key.byte_len))
            .and_then(|bytes| bytes.checked_add(full.value.byte_len))
            .ok_or_else(|| invalid("resident-chain target KV alias ledger overflow"))?;
        let draft_k_u64 = draft_k as u64;
        let ledger = Gemma4Mtp12ChainLedger {
            draft_k: draft_k as u32,
            command_buffers: 1,
            command_buffer_waits: 1,
            target_q6k_table_alias_bytes: target_q6k_embedding.wire.byte_len,
            target_kv_alias_bytes,
            assistant_matrix_read_bytes: self.dense_bf16.as_ref()
                .map_or(FULL_Q4_MATRIX_BYTES, |dense| self.layout.embedding.byte_len + dense.byte_len())
                .checked_mul(draft_k_u64)
                .ok_or_else(|| invalid("resident-chain assistant matrix ledger overflow"))?,
            target_kv_read_bytes: total_target_kv_read_bytes,
            dynamic_attention_scratch_bytes: attention_scores.length(),
            resident_chain_state_bytes: self
                .scratch
                .chain_initial_recurrent_hidden
                .length()
                .checked_add(self.scratch.chain_recurrent_hidden.length())
                .and_then(|bytes| bytes.checked_add(self.scratch.output_token.length()))
                .ok_or_else(|| invalid("resident-chain state ledger overflow"))?,
            initial_hidden_upload_bytes: 0,
            readback_bytes: draft_k_u64 * std::mem::size_of::<u32>() as u64,
        };
        Ok(Gemma4Mtp12ChainProposal {
            tokens,
            timing: Gemma4Mtp12ChainTiming {
                cpu_prepare_us,
                encode_us,
                wait_us,
                gpu_us,
                kernel_us,
                wall_us: wall_started.elapsed().as_micros(),
            },
            ledger,
        })
    }

    /// Runtime-composable companion for target verifiers that currently expose
    /// their normalized recurrent row as a CPU slice.  Exactly 15,360 bytes are
    /// staged into assistant-owned shared Metal storage, then the same
    /// one-command/one-wait resident chain is used.  Tokens remain the only
    /// device-to-CPU readback.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_chain_from_cpu_hidden(
        &mut self,
        anchor_token: u32,
        initial_recurrent_hidden: &[f32],
        target_q6k_embedding: Gemma4Mtp12Q6KEmbeddingTable<'_>,
        sliding: Gemma4Mtp12DeviceKv<'_>,
        full: Gemma4Mtp12DeviceKv<'_>,
        target_kv_len: usize,
        proposal_position: usize,
        draft_k: usize,
    ) -> Result<Gemma4Mtp12ChainProposal> {
        let wall_started = Instant::now();
        if initial_recurrent_hidden.len() != TARGET_HIDDEN {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP initial recurrent hidden width is {}, expected {TARGET_HIDDEN}",
                initial_recurrent_hidden.len()
            )));
        }
        if initial_recurrent_hidden
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(BackendError::RuntimeShapeMismatch(
                "Gemma 4 12B MTP initial recurrent hidden contains non-finite values".into(),
            ));
        }
        if !mtp12_chain_draft_k_admitted(draft_k) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP resident draft K is {draft_k}, expected 1..={MTP12_CHAIN_MAX_DRAFTS}"
            )));
        }
        if anchor_token as usize >= VOCAB {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "Gemma 4 12B MTP anchor token {anchor_token} exceeds vocab {VOCAB}"
            )));
        }
        let expected_device_registry_id = self.packed_q4.device().registry_id();
        target_q6k_embedding.validate(expected_device_registry_id)?;
        sliding.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )?;
        full.validate(
            expected_device_registry_id,
            GEMMA4_12B_MTP_FULL_HOST_LAYER,
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            "full",
        )?;
        validate_device_chain_positions(
            sliding.max_positions,
            full.max_positions,
            target_kv_len,
            proposal_position,
            draft_k,
        )?;

        let stage_started = Instant::now();
        let staged_hidden = self.scratch.chain_initial_recurrent_hidden.to_owned();
        write_buffer_f32(&staged_hidden, initial_recurrent_hidden)?;
        let stage_us = stage_started.elapsed().as_micros();
        let mut proposal = self.propose_chain_device_resident(
            anchor_token,
            Gemma4Mtp12DeviceRecurrentHidden {
                values: Gemma4Mtp12MetalBufferView {
                    buffer: &staged_hidden,
                    byte_offset: 0,
                    byte_len: staged_hidden.length(),
                },
                hidden: TARGET_HIDDEN,
            },
            target_q6k_embedding,
            sliding,
            full,
            target_kv_len,
            proposal_position,
            draft_k,
        )?;
        proposal.timing.cpu_prepare_us = proposal.timing.cpu_prepare_us.saturating_add(stage_us);
        proposal.timing.wall_us = wall_started.elapsed().as_micros();
        proposal.ledger.initial_hidden_upload_bytes =
            (TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64;
        Ok(proposal)
    }

    fn encode_dense_gemv(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        matrix: Q4TensorRef,
        round_output_bf16: bool,
    ) {
        self.encode_dense_gemv_at_offset(encoder, input, output, 0, matrix, round_output_bf16);
    }

    /// One dense GEMV whose row vector lands at `output_byte_offset` inside
    /// `output`.  Routes to the BF16 dense experiment when it is active,
    /// otherwise per `CAMELID_GEMMA4_MTP12_FUSE_GEMV_X4`: `1` =
    /// `mtp12_q4_0_gemv_x4`, `2` = `mtp12_q4_0_gemv_staged` for rows of at
    /// most 1024 columns (one block per lane) and the established kernel for
    /// wider rows, `3` = staged for every shape, unset = the established
    /// `mtp12_q4_0_gemv`; every one of those routes produces the same bits.
    /// Levels `4`/`5` route to the NON-bit-identical `mtp12_q4_0_gemv_split`
    /// (two/four simdgroups per row) and exist only for measurement.
    fn encode_dense_gemv_at_offset(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        output_byte_offset: u64,
        matrix: Q4TensorRef,
        round_output_bf16: bool,
    ) {
        if let Some(dense) = &self.dense_bf16 {
            dense.encode(encoder, input, output, output_byte_offset, matrix, round_output_bf16);
            return;
        }
        let staged = match self.fuse.gemv_x4 {
            2 => matrix.cols <= 1024,
            3 => true,
            _ => false,
        };
        if let Some(splits) = mtp12_gemv_split_count(self.fuse.gemv_x4) {
            encode_q4_gemv_split(encoder, &self.pipelines.q4_gemv_split, &self.packed_q4,
                input, output, output_byte_offset, matrix, round_output_bf16, splits);
        } else if staged {
            encode_q4_gemv_staged(encoder, &self.pipelines.q4_gemv_staged, &self.packed_q4,
                input, output, output_byte_offset, matrix, round_output_bf16,
                self.fuse.gemv_staged_rows);
        } else if self.fuse.gemv_x4 == 1 {
            encode_q4_gemv_at_offset(encoder, &self.pipelines.q4_gemv_x4, &self.packed_q4,
                input, output, output_byte_offset, matrix, round_output_bf16, true);
        } else {
            encode_q4_gemv_at_offset(encoder, &self.pipelines.q4_gemv, &self.packed_q4,
                input, output, output_byte_offset, matrix, round_output_bf16, false);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_k1(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer_index: usize,
        key: &Buffer,
        value: &Buffer,
        kv_heads: usize,
        head_dim: usize,
        logical_len: usize,
        compact_base: usize,
        position_count: usize,
        cos: &Buffer,
        sin: &Buffer,
        attention_scores: &Buffer,
    ) {
        self.encode_layer_k1_from_views(
            encoder,
            layer_index,
            key,
            value,
            0,
            0,
            logical_len,
            kv_heads,
            head_dim,
            logical_len,
            compact_base,
            position_count,
            cos,
            sin,
            0,
            attention_scores,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_layer_k1_from_views(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer_index: usize,
        key: &BufferRef,
        value: &BufferRef,
        key_byte_offset: u64,
        value_byte_offset: u64,
        kv_capacity: usize,
        kv_heads: usize,
        head_dim: usize,
        logical_len: usize,
        compact_base: usize,
        position_count: usize,
        cos: &Buffer,
        sin: &Buffer,
        rope_byte_offset: u64,
        attention_scores: &Buffer,
    ) {
        let layer = &self.layers[layer_index];
        let matrices = layer.matrices;
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.hidden,
            &layer.input_norm,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            1,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.normed,
            &self.scratch.query,
            matrices.q,
            true,
        );
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.query,
            &layer.q_norm,
            &self.scratch.query_normed,
            head_dim,
            N_HEADS,
        );
        encode_rope_at_offset(
            encoder,
            &self.pipelines.rope,
            &self.scratch.query_normed,
            cos,
            sin,
            rope_byte_offset,
            head_dim,
        );
        encode_attention(
            encoder,
            &self.pipelines,
            &self.scratch.query_normed,
            key,
            value,
            key_byte_offset,
            value_byte_offset,
            kv_capacity,
            attention_scores,
            &self.scratch.context,
            kv_heads,
            head_dim,
            logical_len,
            compact_base,
            position_count,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.context,
            &self.scratch.attention_projection,
            matrices.o,
            true,
        );
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.attention_projection,
            &layer.post_attention_norm,
            &self.scratch.attention_normalized,
            ASSISTANT_HIDDEN,
            1,
        );
        encode_add(
            encoder,
            &self.pipelines.add,
            &self.scratch.hidden,
            &self.scratch.attention_normalized,
            &self.scratch.attention_residual,
            ASSISTANT_HIDDEN,
        );
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.attention_residual,
            &layer.pre_feedforward_norm,
            &self.scratch.normed,
            ASSISTANT_HIDDEN,
            1,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.normed,
            &self.scratch.gate,
            matrices.gate,
            true,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.normed,
            &self.scratch.up,
            matrices.up,
            true,
        );
        encode_gelu_mul(
            encoder,
            &self.pipelines.gelu_mul,
            &self.scratch.gate,
            &self.scratch.up,
            &self.scratch.gated,
            FFN_HIDDEN,
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.gated,
            &self.scratch.down,
            matrices.down,
            true,
        );
        encode_rms_norm(
            encoder,
            &self.pipelines.rms_norm,
            &self.scratch.down,
            &layer.post_feedforward_norm,
            &self.scratch.down_normalized,
            ASSISTANT_HIDDEN,
            1,
        );
        encode_add_scale(
            encoder,
            &self.pipelines.add_scale,
            &self.scratch.attention_residual,
            &self.scratch.down_normalized,
            &self.scratch.hidden,
            ASSISTANT_HIDDEN,
            layer.scale,
        );
    }

    /// The W8 tree step's layer: [`Self::encode_layer_k1_from_views`]
    /// re-sequenced around the `CAMELID_GEMMA4_MTP12_FUSE_*` kernels.  Every
    /// selector unset reproduces that function's dispatch list exactly (the
    /// four established callers keep using it directly).
    ///
    /// Contract under `fuse.norm`: the caller has already normalized
    /// `scratch.hidden` into `scratch.normed` with this layer's `input_norm`
    /// (layer 0: a standalone rms_norm; layer l>0: the previous layer's fused
    /// tail), and this layer's tail writes the next input norm
    /// (`next_norm`, `next_normed` = `layers[l+1].input_norm` -> `normed`, or
    /// `final_norm` -> `final_normalized` for the last layer) so the caller
    /// must NOT encode a final rms_norm.  With `fuse.norm` unset the layer
    /// encodes its own input norm and the caller encodes the final norm.
    #[allow(clippy::too_many_arguments)]
    fn encode_layer_k1_fused(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        layer_index: usize,
        key: &BufferRef,
        value: &BufferRef,
        key_byte_offset: u64,
        value_byte_offset: u64,
        kv_capacity: usize,
        kv_heads: usize,
        head_dim: usize,
        logical_len: usize,
        compact_base: usize,
        position_count: usize,
        cos: &Buffer,
        sin: &Buffer,
        rope_byte_offset: u64,
        attention_scores: &Buffer,
        next_norm: &Buffer,
        next_normed: &Buffer,
    ) {
        let layer = &self.layers[layer_index];
        let matrices = layer.matrices;
        let fuse = self.fuse;
        if !fuse.norm {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.hidden,
                &layer.input_norm,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
                1,
            );
        }
        self.encode_dense_gemv(
            encoder,
            &self.scratch.normed,
            &self.scratch.query,
            matrices.q,
            true,
        );
        if fuse.qrope {
            encode_qnorm_rope(
                encoder,
                &self.pipelines.qnorm_rope,
                &self.scratch.query,
                &layer.q_norm,
                cos,
                sin,
                rope_byte_offset,
                &self.scratch.query_normed,
                head_dim,
            );
        } else {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.query,
                &layer.q_norm,
                &self.scratch.query_normed,
                head_dim,
                N_HEADS,
            );
            encode_rope_at_offset(
                encoder,
                &self.pipelines.rope,
                &self.scratch.query_normed,
                cos,
                sin,
                rope_byte_offset,
                head_dim,
            );
        }
        let covers = mtp12_attention_v2_covers(head_dim);
        encode_attention_impl_ex(
            encoder,
            &self.pipelines,
            &self.scratch.query_normed,
            key,
            value,
            key_byte_offset,
            value_byte_offset,
            kv_capacity,
            attention_scores,
            &self.scratch.context,
            kv_heads,
            head_dim,
            logical_len,
            compact_base,
            position_count,
            mtp12_attention_v2_enabled() && covers,
            if covers { fuse.softmax_ctx } else { 0 },
            Some(&self.scratch.attention_stats),
            mtp12_attn_geom(head_dim, kv_heads),
        );
        self.encode_dense_gemv(
            encoder,
            &self.scratch.context,
            &self.scratch.attention_projection,
            matrices.o,
            true,
        );
        if fuse.norm {
            encode_norm_add_norm(
                encoder,
                &self.pipelines.norm_add_norm,
                &self.scratch.attention_projection,
                &layer.post_attention_norm,
                &self.scratch.hidden,
                &layer.pre_feedforward_norm,
                &self.scratch.attention_residual,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
                None,
            );
        } else {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.attention_projection,
                &layer.post_attention_norm,
                &self.scratch.attention_normalized,
                ASSISTANT_HIDDEN,
                1,
            );
            encode_add(
                encoder,
                &self.pipelines.add,
                &self.scratch.hidden,
                &self.scratch.attention_normalized,
                &self.scratch.attention_residual,
                ASSISTANT_HIDDEN,
            );
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.attention_residual,
                &layer.pre_feedforward_norm,
                &self.scratch.normed,
                ASSISTANT_HIDDEN,
                1,
            );
        }
        if fuse.gate_up && self.dense_bf16.is_none() {
            encode_q4_gemv_gate_up_gelu(
                encoder,
                &self.pipelines.q4_gemv_gate_up_gelu,
                &self.packed_q4,
                &self.scratch.normed,
                &self.scratch.gated,
                matrices.gate,
                matrices.up,
            );
        } else {
            self.encode_dense_gemv(
                encoder,
                &self.scratch.normed,
                &self.scratch.gate,
                matrices.gate,
                true,
            );
            self.encode_dense_gemv(
                encoder,
                &self.scratch.normed,
                &self.scratch.up,
                matrices.up,
                true,
            );
            encode_gelu_mul(
                encoder,
                &self.pipelines.gelu_mul,
                &self.scratch.gate,
                &self.scratch.up,
                &self.scratch.gated,
                FFN_HIDDEN,
            );
        }
        self.encode_dense_gemv(
            encoder,
            &self.scratch.gated,
            &self.scratch.down,
            matrices.down,
            true,
        );
        if fuse.norm {
            encode_norm_add_norm(
                encoder,
                &self.pipelines.norm_add_norm,
                &self.scratch.down,
                &layer.post_feedforward_norm,
                &self.scratch.attention_residual,
                next_norm,
                &self.scratch.hidden,
                next_normed,
                ASSISTANT_HIDDEN,
                Some(layer.scale),
            );
        } else {
            encode_rms_norm(
                encoder,
                &self.pipelines.rms_norm,
                &self.scratch.down,
                &layer.post_feedforward_norm,
                &self.scratch.down_normalized,
                ASSISTANT_HIDDEN,
                1,
            );
            encode_add_scale(
                encoder,
                &self.pipelines.add_scale,
                &self.scratch.attention_residual,
                &self.scratch.down_normalized,
                &self.scratch.hidden,
                ASSISTANT_HIDDEN,
                layer.scale,
            );
        }
    }
}

fn dispatch_1d(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    count: usize,
) {
    let width = pipeline.thread_execution_width().max(1);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (count as u64).div_ceil(width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_q6k_embedding_gather(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    row: Gemma4Mtp12Q6KEmbeddingRow<'_>,
    pre_input: &Buffer,
) {
    let embedding_scale = (TARGET_HIDDEN as f32).sqrt();
    encoder.set_compute_pipeline_state(pipeline);
    // Bind the complete caller buffer at offset zero. The explicit u64 row
    // offset remains valid for Q6_K rows whose 3,150-byte stride is only
    // two-byte aligned inside a larger no-copy table.
    encoder.set_buffer(0, Some(row.wire.buffer), 0);
    encoder.set_buffer(1, Some(pre_input), 0);
    encoder.set_bytes(2, 8, &row.wire.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(3, 4, &embedding_scale as *const f32 as *const c_void);
    dispatch_1d(encoder, pipeline, TARGET_HIDDEN);
}

#[allow(clippy::too_many_arguments)]
fn encode_q6k_embedding_and_recurrent_gather(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    token_ids: &Buffer,
    token_index: usize,
    table: Gemma4Mtp12Q6KEmbeddingTable<'_>,
    recurrent_hidden: &BufferRef,
    recurrent_byte_offset: u64,
    pre_input: &Buffer,
) {
    let token_index = token_index as u32;
    let target_vocab = VOCAB as u32;
    let embedding_scale = (TARGET_HIDDEN as f32).sqrt();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(token_ids), 0);
    // Bind the complete caller buffer at zero because a legal Q6_K table view
    // can begin at an offset that is two-byte rather than four-byte aligned.
    encoder.set_buffer(1, Some(table.wire.buffer), 0);
    encoder.set_buffer(2, Some(recurrent_hidden), recurrent_byte_offset);
    encoder.set_buffer(3, Some(pre_input), 0);
    encoder.set_bytes(4, 8, &table.wire.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(5, 4, &token_index as *const u32 as *const c_void);
    encoder.set_bytes(6, 4, &target_vocab as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &embedding_scale as *const f32 as *const c_void);
    dispatch_1d(encoder, pipeline, TARGET_HIDDEN);
}

fn encode_copy_f32_to_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    source: &Buffer,
    destination: &Buffer,
    destination_byte_offset: u64,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(source), 0);
    encoder.set_buffer(1, Some(destination), destination_byte_offset);
    encoder.set_bytes(2, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

fn encode_shortlist(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    shortlist: &Mtp12Shortlist,
    query: &Buffer,
) {
    encoder.set_compute_pipeline_state(&pipelines.shortlist_scores);
    encoder.set_buffer(0, Some(&shortlist.centroids), 0);
    encoder.set_buffer(1, Some(query), 0);
    encoder.set_buffer(2, Some(&shortlist.scores), 0);
    encoder.dispatch_thread_groups(
        MTLSize { width: MTP12_SHORTLIST_CLUSTERS as u64, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    encoder.set_compute_pipeline_state(&pipelines.shortlist_select);
    encoder.set_buffer(0, Some(&shortlist.scores), 0);
    encoder.set_buffer(1, Some(&shortlist.selected), 0);
    encoder.set_bytes(2, 4, &shortlist.top as *const u32 as *const c_void);
    dispatch_1d(encoder, &pipelines.shortlist_select, MTP12_SHORTLIST_CLUSTERS);
}

fn encode_q4_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
) {
    encode_q4_gemv_at_offset(
        encoder, pipeline, weights, input, output, 0, matrix, round_output_bf16, false,
    );
}

/// `output_byte_offset` lands the row vector inside a larger buffer.  `x4`
/// selects the four-rows-per-128-thread geometry (`pipeline` must then be
/// `mtp12_q4_0_gemv_x4`); otherwise one 32-thread group per row as the
/// established `mtp12_q4_0_gemv` / `_shortlist` kernels expect.
#[allow(clippy::too_many_arguments)]
fn encode_q4_gemv_at_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    output_byte_offset: u64,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
    x4: bool,
) {
    let round = u32::from(round_output_bf16);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), output_byte_offset);
    encoder.set_bytes(3, 4, &matrix.cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &matrix.rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &matrix.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round as *const u32 as *const c_void);
    let (groups, threads) = if x4 {
        ((matrix.rows as u64).div_ceil(4), 128)
    } else {
        (matrix.rows as u64, 32)
    };
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

/// `mtp12_q4_0_gemv_staged`: one 32-thread simdgroup per threadgroup folding
/// `rows_per_group` consecutive rows with its input slice register-staged
/// (`pipeline` must be that kernel); bit-identical to `mtp12_q4_0_gemv`.
#[allow(clippy::too_many_arguments)]
fn encode_q4_gemv_staged(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    output_byte_offset: u64,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
    rows_per_group: u32,
) {
    debug_assert!((1..=MTP12_GEMV_STAGED_ROWS_MAX).contains(&rows_per_group));
    let round = u32::from(round_output_bf16);
    let rows_per_group = rows_per_group.clamp(1, MTP12_GEMV_STAGED_ROWS_MAX);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), output_byte_offset);
    encoder.set_bytes(3, 4, &matrix.cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &matrix.rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &matrix.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &rows_per_group as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (matrix.rows as u64).div_ceil(rows_per_group as u64),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
}

/// Simdgroups per row for the split-block experiment, or `None` when the
/// selector level is one of the bit-identical routes.
fn mtp12_gemv_split_count(level: u8) -> Option<u32> {
    match level {
        4 => Some(2),
        5 => Some(4),
        _ => None,
    }
}

/// `mtp12_q4_0_gemv_split`: `splits` simdgroups per row, each folding a
/// disjoint contiguous block range, combined in simdgroup order.  NOT
/// bit-identical to `mtp12_q4_0_gemv` - measurement lane only.
#[allow(clippy::too_many_arguments)]
fn encode_q4_gemv_split(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    output_byte_offset: u64,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
    splits: u32,
) {
    debug_assert!((2..=4).contains(&splits));
    let round = u32::from(round_output_bf16);
    let splits = splits.clamp(1, 4);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), output_byte_offset);
    encoder.set_bytes(3, 4, &matrix.cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &matrix.rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &matrix.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &splits as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: matrix.rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: (splits * 32) as u64,
            height: 1,
            depth: 1,
        },
    );
}

/// Fused gate/up GEMV + GELU gate (`mtp12_q4_0_gemv_gate_up_gelu`): `gate` and
/// `up` share `input`, and `output[row]` receives what
/// `mtp12_gelu_mul(gate_row, up_row)` would after both rows were BF16-rounded.
fn encode_q4_gemv_gate_up_gelu(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    gate: Q4TensorRef,
    up: Q4TensorRef,
) {
    debug_assert_eq!(gate.rows, up.rows);
    debug_assert_eq!(gate.cols, up.cols);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &gate.cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &gate.rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &gate.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 8, &up.byte_offset as *const u64 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (gate.rows as u64).div_ceil(4),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_q4_gemv_shortlist_compact(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: Q4TensorRef,
) {
    let round = 0u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(weights), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &matrix.cols as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &matrix.rows as *const u32 as *const c_void);
    encoder.set_bytes(5, 8, &matrix.byte_offset as *const u64 as *const c_void);
    encoder.set_bytes(6, 4, &round as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (matrix.rows as u64).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_rms_norm(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    width: usize,
    rows: usize,
) {
    let width = width as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &width as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &RMS_EPS as *const f32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_rope_at_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    data: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    table_byte_offset: u64,
    head_dim: usize,
) {
    let heads = N_HEADS as u32;
    let head_dim_u32 = head_dim as u32;
    let count = N_HEADS * head_dim / 2;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(data), 0);
    encoder.set_buffer(1, Some(cos), table_byte_offset);
    encoder.set_buffer(2, Some(sin), table_byte_offset);
    encoder.set_bytes(3, 4, &heads as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

/// Opt-in latency-hiding V2 of the assistant's attention over the target KV
/// (`CAMELID_GEMMA4_MTP12_ATTN_V2=1`).  Same per-element arithmetic as the established
/// kernels (see the shader note on `mtp12_attention_scores_v2`); only the dispatch grids
/// change, so the proposals are bit-identical and acceptance is unchanged.
fn mtp12_attention_v2_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CAMELID_GEMMA4_MTP12_ATTN_V2")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// Geometry the V2 kernels cover: the staged query fits threadgroup memory and the
/// context grid's 128-dim blocks tile the head exactly (both 12B head shapes qualify).
fn mtp12_attention_v2_covers(head_dim: usize) -> bool {
    head_dim % 128 == 0 && head_dim <= 512
}

const MTP12_ATTN_FORM_ENV: &str = "CAMELID_GEMMA4_MTP12_ATTN_FORM";

/// One assistant-attention CONTEXT kernel form.  `V2` is today's
/// `mtp12_attention_context_v2` dispatch byte-for-byte; every
/// `d<LANE_DIMS>p<PAIRS>` is the pipelined `mtp12_context_form<LANE_DIMS, PAIRS>`
/// instantiation - `LANE_DIMS` output dims per lane (so the head-dimension axis
/// splits into more threadgroups) and `PAIRS` query heads of one KV head per
/// threadgroup (so each V element is loaded once for all of them), with the
/// 32-position coalesced probability load distributed by `simd_broadcast` and
/// double-buffered V loads.  All forms are bit-identical to `V2` (the shader
/// note carries the per-element argument; pinned by
/// `mtp12_attention_forms_match_v2_bit_for_bit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mtp12ContextForm {
    V2,
    D1P1,
    D2P1,
    D4P1,
    D1P2,
    D2P2,
    D4P2,
    D1P4,
    D2P4,
    D1P8,
}

impl Mtp12ContextForm {
    /// The pipelined instantiations, in pipeline-table order.
    const PIPELINED: [Self; 9] = [
        Self::D1P1,
        Self::D2P1,
        Self::D4P1,
        Self::D1P2,
        Self::D2P2,
        Self::D4P2,
        Self::D1P4,
        Self::D2P4,
        Self::D1P8,
    ];

    fn parse(value: &str) -> Option<Self> {
        Some(match value.trim().to_ascii_lowercase().as_str() {
            "v2" => Self::V2,
            "d1p1" => Self::D1P1,
            "d2p1" => Self::D2P1,
            "d4p1" => Self::D4P1,
            "d1p2" => Self::D1P2,
            "d2p2" => Self::D2P2,
            "d4p2" => Self::D4P2,
            "d1p4" => Self::D1P4,
            "d2p4" => Self::D2P4,
            "d1p8" => Self::D1P8,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::V2 => "v2",
            Self::D1P1 => "d1p1",
            Self::D2P1 => "d2p1",
            Self::D4P1 => "d4p1",
            Self::D1P2 => "d1p2",
            Self::D2P2 => "d2p2",
            Self::D4P2 => "d4p2",
            Self::D1P4 => "d1p4",
            Self::D2P4 => "d2p4",
            Self::D1P8 => "d1p8",
        }
    }

    /// `(output dims per lane, query heads per threadgroup)`; `V2` has neither.
    fn shape(self) -> Option<(usize, usize)> {
        Some(match self {
            Self::V2 => return None,
            Self::D1P1 => (1, 1),
            Self::D2P1 => (2, 1),
            Self::D4P1 => (4, 1),
            Self::D1P2 => (1, 2),
            Self::D2P2 => (2, 2),
            Self::D4P2 => (4, 2),
            Self::D1P4 => (1, 4),
            Self::D2P4 => (2, 4),
            Self::D1P8 => (1, 8),
        })
    }

    fn kernel(self) -> Option<&'static str> {
        Some(match self {
            Self::V2 => return None,
            Self::D1P1 => "mtp12_attention_context_d1p1",
            Self::D2P1 => "mtp12_attention_context_d2p1",
            Self::D4P1 => "mtp12_attention_context_d4p1",
            Self::D1P2 => "mtp12_attention_context_d1p2",
            Self::D2P2 => "mtp12_attention_context_d2p2",
            Self::D4P2 => "mtp12_attention_context_d4p2",
            Self::D1P4 => "mtp12_attention_context_d1p4",
            Self::D2P4 => "mtp12_attention_context_d2p4",
            Self::D1P8 => "mtp12_attention_context_d1p8",
        })
    }

    /// The grid this form dispatches: `(head chunks, dim blocks)` of 32 lanes.
    fn grid(self, head_dim: usize) -> Option<(usize, usize)> {
        let (lane_dims, pairs) = self.shape()?;
        Some((N_HEADS / pairs, head_dim / (32 * lane_dims)))
    }

    /// A form is available where the head chunks share one KV head, the head
    /// dimension tiles exactly, and the V2 geometry contract holds.
    fn available(self, head_dim: usize, kv_heads: usize) -> bool {
        let Some((lane_dims, pairs)) = self.shape() else {
            return true;
        };
        let group = N_HEADS / kv_heads;
        mtp12_attention_v2_covers(head_dim)
            && kv_heads > 0
            && N_HEADS % kv_heads == 0
            && group % pairs == 0
            && N_HEADS % pairs == 0
            && head_dim % (32 * lane_dims) == 0
    }
}

/// One assistant-attention SCORES kernel form.  `Q1` is today's
/// `mtp12_attention_scores_v2` dispatch byte-for-byte; `q<PAIRS>` is
/// `mtp12_scores_form<PAIRS>`, which stages `PAIRS` queries of one KV head and
/// reads each K row once for all of them.  Bit-identical to `Q1`: the eight
/// float4 accumulators, their order and the fold are unchanged and the K bits
/// do not depend on the query head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mtp12ScoresForm {
    Q1,
    Q2,
    Q4,
}

impl Mtp12ScoresForm {
    /// The paired instantiations, in pipeline-table order.
    const PAIRED: [Self; 2] = [Self::Q2, Self::Q4];

    fn parse(value: &str) -> Option<Self> {
        Some(match value.trim().to_ascii_lowercase().as_str() {
            "q1" => Self::Q1,
            "q2" => Self::Q2,
            "q4" => Self::Q4,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::Q1 => "q1",
            Self::Q2 => "q2",
            Self::Q4 => "q4",
        }
    }

    /// Query heads per threadgroup; `Q1` keeps the one-head kernel.
    fn pairs(self) -> Option<usize> {
        match self {
            Self::Q1 => None,
            Self::Q2 => Some(2),
            Self::Q4 => Some(4),
        }
    }

    fn kernel(self) -> Option<&'static str> {
        Some(match self {
            Self::Q1 => return None,
            Self::Q2 => "mtp12_attention_scores_q2",
            Self::Q4 => "mtp12_attention_scores_q4",
        })
    }

    fn available(self, head_dim: usize, kv_heads: usize) -> bool {
        let Some(pairs) = self.pairs() else {
            return true;
        };
        let group = N_HEADS / kv_heads;
        mtp12_attention_v2_covers(head_dim)
            && kv_heads > 0
            && N_HEADS % kv_heads == 0
            && group % pairs == 0
            && N_HEADS % pairs == 0
    }
}

/// The attention kernels one 12B head geometry binds: `<context>[:<scores>]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mtp12AttnGeom {
    context: Mtp12ContextForm,
    scores: Mtp12ScoresForm,
}

impl Mtp12AttnGeom {
    /// Today's kernels byte-for-byte.
    const TODAY: Self = Self {
        context: Mtp12ContextForm::V2,
        scores: Mtp12ScoresForm::Q1,
    };

    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        let (context, scores) = match value.split_once(':') {
            Some((context, scores)) => (context, Some(scores)),
            None => (value, None),
        };
        Some(Self {
            context: Mtp12ContextForm::parse(context)?,
            scores: match scores {
                Some(scores) => Mtp12ScoresForm::parse(scores)?,
                None => Mtp12ScoresForm::Q1,
            },
        })
    }

    fn label(self) -> String {
        format!("{}:{}", self.context.label(), self.scores.label())
    }

    fn available(self, head_dim: usize, kv_heads: usize) -> bool {
        self.context.available(head_dim, kv_heads) && self.scores.available(head_dim, kv_heads)
    }

    /// Each half that does not fit this geometry falls back to today's kernel.
    fn restrict(self, head_dim: usize, kv_heads: usize) -> Self {
        Self {
            context: if self.context.available(head_dim, kv_heads) {
                self.context
            } else {
                Mtp12ContextForm::V2
            },
            scores: if self.scores.available(head_dim, kv_heads) {
                self.scores
            } else {
                Mtp12ScoresForm::Q1
            },
        }
    }
}

/// The assistant-attention kernel selection: one geometry for the three sliding
/// HD256 layers and one for the single global HD512 layer.  `TODAY` is
/// byte-for-byte today's dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Mtp12AttnSelection {
    sliding: Mtp12AttnGeom,
    global: Mtp12AttnGeom,
}

impl Mtp12AttnSelection {
    const TODAY: Self = Self {
        sliding: Mtp12AttnGeom::TODAY,
        global: Mtp12AttnGeom::TODAY,
    };

    /// `<geom>` applies one spec to both head shapes (each half that a shape
    /// cannot host stays today's kernel); `<sliding>,<global>` selects each
    /// explicitly and REFUSES a spec the shape cannot host.
    fn parse(value: &str) -> Option<Self> {
        let value = value.trim();
        if let Some((sliding, global)) = value.split_once(',') {
            let sliding = Mtp12AttnGeom::parse(sliding)?;
            let global = Mtp12AttnGeom::parse(global)?;
            return (sliding.available(LOCAL_HEAD_DIM, LOCAL_KV_HEADS)
                && global.available(FULL_HEAD_DIM, FULL_KV_HEADS))
            .then_some(Self { sliding, global });
        }
        let geom = Mtp12AttnGeom::parse(value)?;
        Some(Self {
            sliding: geom.restrict(LOCAL_HEAD_DIM, LOCAL_KV_HEADS),
            global: geom.restrict(FULL_HEAD_DIM, FULL_KV_HEADS),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn label(self) -> String {
        format!("{},{}", self.sliding.label(), self.global.label())
    }

    fn for_head_dim(self, head_dim: usize) -> Mtp12AttnGeom {
        if head_dim == LOCAL_HEAD_DIM {
            self.sliding
        } else {
            self.global
        }
    }
}

/// `CAMELID_GEMMA4_MTP12_ATTN_FORM`, read once per process: the assistant's
/// attention kernel forms (`<context>[:<scores>]`, optionally
/// `<sliding>,<global>`).  Context forms are `v2` (today) and
/// `d<dims-per-lane>p<heads-per-threadgroup>`; scores forms are `q1` (today)
/// and `q<heads-per-threadgroup>`.  Unset = today's kernels byte-for-byte; an
/// unparsable value keeps today's kernels and says so once.  The forms are
/// bit-identical, so the drafts and the acceptance rate never change.
fn mtp12_attn_selection() -> Mtp12AttnSelection {
    static SELECTION: std::sync::OnceLock<Mtp12AttnSelection> = std::sync::OnceLock::new();
    *SELECTION.get_or_init(|| {
        let Ok(value) = std::env::var(MTP12_ATTN_FORM_ENV) else {
            return Mtp12AttnSelection::TODAY;
        };
        match Mtp12AttnSelection::parse(&value) {
            Some(selection) => {
                eprintln!(
                    "[gemma4-mtp12] {MTP12_ATTN_FORM_ENV}={value:?} -> sliding HD256 {} / \
                     global HD512 {}",
                    selection.sliding.label(),
                    selection.global.label()
                );
                selection
            }
            None => {
                eprintln!(
                    "[gemma4-mtp12] {MTP12_ATTN_FORM_ENV}={value:?} is not \
                     <context>[:<scores>] or <sliding>,<global> with context in \
                     v2|d1p1|d2p1|d4p1|d1p2|d2p2|d4p2|d1p4|d2p4|d1p8 and scores in \
                     q1|q2|q4 (and hostable by that head shape); keeping today's kernels"
                );
                Mtp12AttnSelection::TODAY
            }
        }
    })
}

/// The forms this layer's geometry actually binds.
fn mtp12_attn_geom(head_dim: usize, kv_heads: usize) -> Mtp12AttnGeom {
    mtp12_attn_selection()
        .for_head_dim(head_dim)
        .restrict(head_dim, kv_heads)
}

#[allow(clippy::too_many_arguments)]
fn encode_attention(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    query: &Buffer,
    key: &BufferRef,
    value: &BufferRef,
    key_byte_offset: u64,
    value_byte_offset: u64,
    kv_capacity: usize,
    scores: &Buffer,
    output: &Buffer,
    kv_heads: usize,
    head_dim: usize,
    logical_len: usize,
    compact_base: usize,
    position_count: usize,
) {
    encode_attention_impl(
        encoder,
        pipelines,
        query,
        key,
        value,
        key_byte_offset,
        value_byte_offset,
        kv_capacity,
        scores,
        output,
        kv_heads,
        head_dim,
        logical_len,
        compact_base,
        position_count,
        mtp12_attention_v2_enabled() && mtp12_attention_v2_covers(head_dim),
    );
}

/// `v2` selects the latency-hiding grids (`mtp12_attention_scores_v2` /
/// `mtp12_attention_context_v2`); the softmax dispatch is shared.  Both routes bind the
/// same arguments and produce bit-identical output (pinned by
/// `mtp12_attention_v2_matches_established_kernels_bit_for_bit`).
#[allow(clippy::too_many_arguments)]
fn encode_attention_impl(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    query: &Buffer,
    key: &BufferRef,
    value: &BufferRef,
    key_byte_offset: u64,
    value_byte_offset: u64,
    kv_capacity: usize,
    scores: &Buffer,
    output: &Buffer,
    kv_heads: usize,
    head_dim: usize,
    logical_len: usize,
    compact_base: usize,
    position_count: usize,
    v2: bool,
) {
    encode_attention_impl_ex(
        encoder,
        pipelines,
        query,
        key,
        value,
        key_byte_offset,
        value_byte_offset,
        kv_capacity,
        scores,
        output,
        kv_heads,
        head_dim,
        logical_len,
        compact_base,
        position_count,
        v2,
        0,
        None,
        mtp12_attn_geom(head_dim, kv_heads),
    );
}

/// `softmax_ctx` replaces the softmax + context pair (all levels require
/// [`mtp12_attention_v2_covers`], all leave the scores buffer holding raw
/// scores, and all are bit-identical to the pair - pinned by
/// `mtp12_attention_softmax_context_fused_matches_pair_bit_for_bit`):
/// `0` = the established pair, `1` = `mtp12_attention_softmax_context_v2`
/// (every context threadgroup recomputes max/denominator), `2` =
/// `mtp12_attention_softmax_stats` + `mtp12_attention_context_v2_stats16`,
/// `3` = `mtp12_attention_softmax_stats` + `mtp12_attention_context_v2_stats4`.
/// `stats` is the `N_HEADS * 2` scratch the pre-pass writes; levels 2 and 3
/// require it.
#[allow(clippy::too_many_arguments)]
fn encode_attention_impl_ex(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    query: &Buffer,
    key: &BufferRef,
    value: &BufferRef,
    key_byte_offset: u64,
    value_byte_offset: u64,
    kv_capacity: usize,
    scores: &Buffer,
    output: &Buffer,
    kv_heads: usize,
    head_dim: usize,
    logical_len: usize,
    compact_base: usize,
    position_count: usize,
    v2: bool,
    softmax_ctx: u8,
    stats: Option<&Buffer>,
    attn: Mtp12AttnGeom,
) {
    debug_assert!(position_count > 0);
    debug_assert!(softmax_ctx <= 3);
    debug_assert_eq!(attn, attn.restrict(head_dim, kv_heads));
    debug_assert!(softmax_ctx == 0 || mtp12_attention_v2_covers(head_dim));
    debug_assert!(softmax_ctx < 2 || stats.is_some());
    debug_assert_eq!(compact_base + position_count, logical_len);
    let n_heads = N_HEADS as u32;
    let head_dim = head_dim as u32;
    let position_count_u32 = position_count as u32;
    let group = (N_HEADS / kv_heads) as u32;
    let position_stride = head_dim;
    let kv_head_stride = (kv_capacity * head_dim as usize) as u32;
    let kv_base_offset = (compact_base * head_dim as usize) as u32;
    let compact_base_u32 = compact_base as u32;
    let logical_len_u32 = logical_len as u32;
    let tg32 = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };

    // The `q<PAIRS>` scores form stages PAIRS queries of one KV head per threadgroup
    // and reads each K row once for all of them; same arguments, same bits.
    let scores_pairs = if v2 { attn.scores.pairs() } else { None };
    encoder.set_compute_pipeline_state(match scores_pairs {
        Some(_) => pipelines
            .attention_forms
            .scores(attn.scores)
            .expect("paired scores pipeline"),
        None if v2 => &pipelines.attention_scores_v2,
        None => &pipelines.attention_scores,
    });
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(key), key_byte_offset);
    encoder.set_buffer(2, Some(scores), 0);
    encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const c_void);
    encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
    encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
    encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
    encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (N_HEADS / scores_pairs.unwrap_or(1)) as u64,
            height: if v2 {
                position_count.div_ceil(32) as u64
            } else {
                1
            },
            depth: 1,
        },
        tg32,
    );

    match softmax_ctx {
        0 => {
            encoder.set_compute_pipeline_state(&pipelines.attention_softmax);
            encoder.set_buffer(0, Some(scores), 0);
            encoder.set_bytes(1, 4, &n_heads as *const u32 as *const c_void);
            encoder.set_bytes(2, 4, &position_count_u32 as *const u32 as *const c_void);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: N_HEADS as u64,
                    height: 1,
                    depth: 1,
                },
                tg32,
            );
        }
        // Levels 2 and 3 hoist the two softmax passes out of the
        // (heads x dim-blocks) context grid into one per-head pre-pass.
        2 | 3 => {
            encoder.set_compute_pipeline_state(&pipelines.attention_softmax_stats);
            encoder.set_buffer(0, Some(scores), 0);
            encoder.set_bytes(1, 4, &n_heads as *const u32 as *const c_void);
            encoder.set_bytes(2, 4, &position_count_u32 as *const u32 as *const c_void);
            encoder.set_buffer(3, stats.map(|v| &**v), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: N_HEADS as u64,
                    height: 1,
                    depth: 1,
                },
                tg32,
            );
        }
        _ => {}
    }

    // A pipelined context form replaces `mtp12_attention_context_v2` on the unfused
    // route only: the softmax-fused levels consume RAW scores and own their prologue.
    let context_grid = if v2 && softmax_ctx == 0 {
        attn.context.grid(head_dim as usize)
    } else {
        None
    };
    encoder.set_compute_pipeline_state(match softmax_ctx {
        _ if context_grid.is_some() => pipelines
            .attention_forms
            .context(attn.context)
            .expect("pipelined context pipeline"),
        1 => &pipelines.attention_softmax_context_v2,
        2 => &pipelines.attention_context_v2_stats16,
        3 => &pipelines.attention_context_v2_stats4,
        _ if v2 => &pipelines.attention_context_v2,
        _ => &pipelines.attention_context,
    });
    encoder.set_buffer(0, Some(value), value_byte_offset);
    encoder.set_buffer(1, Some(scores), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const c_void);
    encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
    encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
    encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
    encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
    encoder.set_bytes(10, 4, &compact_base_u32 as *const u32 as *const c_void);
    encoder.set_bytes(11, 4, &logical_len_u32 as *const u32 as *const c_void);
    if softmax_ctx >= 2 {
        encoder.set_buffer(12, stats.map(|v| &**v), 0);
    }
    let (context_width, context_height) = match context_grid {
        Some((chunks, blocks)) => (chunks as u64, blocks as u64),
        None => (
            N_HEADS as u64,
            if v2 || softmax_ctx != 0 {
                (head_dim as usize / 128) as u64
            } else {
                1
            },
        ),
    };
    encoder.dispatch_thread_groups(
        MTLSize {
            width: context_width,
            height: context_height,
            depth: 1,
        },
        tg32,
    );
}

fn encode_add(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    output: &Buffer,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

fn encode_add_scale(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    a: &Buffer,
    b: &Buffer,
    output: &Buffer,
    count: usize,
    scale: f32,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(a), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &scale as *const f32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

fn encode_gelu_mul(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    gate: &Buffer,
    up: &Buffer,
    output: &Buffer,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(gate), 0);
    encoder.set_buffer(1, Some(up), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

/// `mtp12_norm_add_norm`: `out_residual = round(residual + rms(input, weight))`
/// (then `round(. * scale)` when `scale` is `Some`) and
/// `out_normed = rms(out_residual, next_weight)`, replacing the
/// rms_norm + add[_scale] + rms_norm dispatch chain bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn encode_norm_add_norm(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    input: &Buffer,
    weight: &Buffer,
    residual: &Buffer,
    next_weight: &Buffer,
    out_residual: &Buffer,
    out_normed: &Buffer,
    width: usize,
    scale: Option<f32>,
) {
    debug_assert!(width <= 1024 && width.is_multiple_of(256));
    let width_u32 = width as u32;
    let (scale_value, apply_scale) = match scale {
        Some(scale) => (scale, 1u32),
        None => (0.0f32, 0u32),
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(residual), 0);
    encoder.set_buffer(3, Some(next_weight), 0);
    encoder.set_buffer(4, Some(out_residual), 0);
    encoder.set_buffer(5, Some(out_normed), 0);
    encoder.set_bytes(6, 4, &width_u32 as *const u32 as *const c_void);
    encoder.set_bytes(7, 4, &RMS_EPS as *const f32 as *const c_void);
    encoder.set_bytes(8, 4, &scale_value as *const f32 as *const c_void);
    encoder.set_bytes(9, 4, &apply_scale as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// `mtp12_qnorm_rope`: per-head RMS norm of `query` with `weight` followed by
/// the split RoPE at `table_byte_offset` into `output`, replacing
/// rms_norm(head_dim x N_HEADS) + rope bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn encode_qnorm_rope(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    query: &Buffer,
    weight: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    table_byte_offset: u64,
    output: &Buffer,
    head_dim: usize,
) {
    debug_assert!(head_dim <= 512 && head_dim.is_multiple_of(256));
    let head_dim_u32 = head_dim as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(cos), table_byte_offset);
    encoder.set_buffer(3, Some(sin), table_byte_offset);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_bytes(5, 4, &head_dim_u32 as *const u32 as *const c_void);
    encoder.set_bytes(6, 4, &RMS_EPS as *const f32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: N_HEADS as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Chunked two-stage vocabulary argmax (`mtp12_argmax_partial` +
/// `mtp12_argmax_merge`), bit-identical to [`encode_argmax_at_offset`].
/// `partial_values` / `partial_ids` need `count.div_ceil(MTP12_ARGMAX_CHUNK)`
/// slots each.
#[allow(clippy::too_many_arguments)]
fn encode_argmax_chunked_at_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    logits: &Buffer,
    partial_values: &Buffer,
    partial_ids: &Buffer,
    output: &Buffer,
    output_byte_offset: u64,
    count: usize,
) {
    let partials = count.div_ceil(MTP12_ARGMAX_CHUNK).max(1);
    debug_assert!(
        partials as u64 * std::mem::size_of::<u32>() as u64 <= partial_ids.length()
            && partials as u64 * std::mem::size_of::<f32>() as u64 <= partial_values.length(),
        "chunked argmax partial scratch is too small for {count} logits"
    );
    let count_u32 = count as u32;
    let chunk_u32 = MTP12_ARGMAX_CHUNK as u32;
    let partials_u32 = partials as u32;
    encoder.set_compute_pipeline_state(&pipelines.argmax_partial);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(partial_values), 0);
    encoder.set_buffer(2, Some(partial_ids), 0);
    encoder.set_bytes(3, 4, &count_u32 as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &chunk_u32 as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: partials as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    encoder.set_compute_pipeline_state(&pipelines.argmax_merge);
    encoder.set_buffer(0, Some(partial_values), 0);
    encoder.set_buffer(1, Some(partial_ids), 0);
    encoder.set_buffer(2, Some(output), output_byte_offset);
    encoder.set_bytes(3, 4, &partials_u32 as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_argmax_at_offset(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    logits: &Buffer,
    output: &Buffer,
    output_byte_offset: u64,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(output), output_byte_offset);
    encoder.set_bytes(2, 4, &count_u32 as *const u32 as *const c_void);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

fn target_kv_read_bytes(local_count: usize, full_count: usize) -> Result<u64> {
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    let local = (3u64)
        .checked_mul(LOCAL_KV_HEADS as u64)
        .and_then(|value| value.checked_mul(local_count as u64))
        .and_then(|value| value.checked_mul(LOCAL_HEAD_DIM as u64))
        .ok_or_else(|| invalid("resident-chain local KV read ledger overflow"))?;
    let full = (FULL_KV_HEADS as u64)
        .checked_mul(full_count as u64)
        .and_then(|value| value.checked_mul(FULL_HEAD_DIM as u64))
        .ok_or_else(|| invalid("resident-chain full KV read ledger overflow"))?;
    local
        .checked_add(full)
        .and_then(|value| value.checked_mul(2)) // K plus V.
        .and_then(|value| value.checked_mul(f32_bytes))
        .ok_or_else(|| invalid("resident-chain total KV read ledger overflow"))
}

fn rope_table_values(
    position: usize,
    head_dim: usize,
    theta: f64,
    active_pairs: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(half);
    let mut sin = Vec::with_capacity(half);
    for pair in 0..half {
        let frequency = if pair < active_pairs {
            1.0 / theta.powf((2 * pair) as f64 / head_dim as f64)
        } else {
            0.0
        };
        let (sine, cosine) = (position as f64 * frequency).sin_cos();
        cos.push(round_to_bf16_f32(cosine as f32));
        sin.push(round_to_bf16_f32(sine as f32));
    }
    (cos, sin)
}

fn write_rope_tables(position: usize, scratch: &Mtp12Scratch) -> Result<()> {
    let (local_cos, local_sin) =
        rope_table_values(position, LOCAL_HEAD_DIM, 10_000.0, LOCAL_HEAD_DIM / 2);
    // Proportional full RoPE rotates 512 * 0.25 / 2 = 64 split-half pairs.
    let (full_cos, full_sin) =
        rope_table_values(position, FULL_HEAD_DIM, 1_000_000.0, FULL_HEAD_DIM / 8);
    write_buffer_f32(&scratch.local_cos, &local_cos)?;
    write_buffer_f32(&scratch.local_sin, &local_sin)?;
    write_buffer_f32(&scratch.full_cos, &full_cos)?;
    write_buffer_f32(&scratch.full_sin, &full_sin)?;
    Ok(())
}

fn write_chain_rope_tables(
    proposal_position: usize,
    draft_k: usize,
    single_position: bool,
    scratch: &Mtp12Scratch,
) -> Result<()> {
    for step in 0..draft_k {
        // The public chain validates anchor + K - 1 before entering here.
        let position = chain_query_position(proposal_position, step, single_position);
        let (local_cos, local_sin) =
            rope_table_values(position, LOCAL_HEAD_DIM, 10_000.0, LOCAL_HEAD_DIM / 2);
        let (full_cos, full_sin) =
            rope_table_values(position, FULL_HEAD_DIM, 1_000_000.0, FULL_HEAD_DIM / 8);
        let local_offset = (step * LOCAL_HEAD_DIM / 2 * std::mem::size_of::<f32>()) as u64;
        let full_offset = (step * FULL_HEAD_DIM / 2 * std::mem::size_of::<f32>()) as u64;
        write_buffer_f32_at(&scratch.local_cos, local_offset, &local_cos)?;
        write_buffer_f32_at(&scratch.local_sin, local_offset, &local_sin)?;
        write_buffer_f32_at(&scratch.full_cos, full_offset, &full_cos)?;
        write_buffer_f32_at(&scratch.full_sin, full_offset, &full_sin)?;
    }
    Ok(())
}

/// HF's shared-KV assistant advances the token and recurrent hidden, but its
/// RoPE anchor remains fixed throughout the round. Keep the original advancing
/// mode as an A/B control until the production acceptance gate is measured.
fn chain_query_position(anchor: usize, step: usize, single_position: bool) -> usize {
    if single_position {
        anchor
    } else {
        anchor + step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w16_assistant_bounds_and_resident_extents_are_exact() {
        assert_eq!(GEMMA4_12B_MTP_MAX_DRAFTS, 15);
        assert!(!mtp12_chain_draft_k_admitted(0));
        assert!(mtp12_chain_draft_k_admitted(1));
        assert!(mtp12_chain_draft_k_admitted(15));
        assert!(!mtp12_chain_draft_k_admitted(16));

        assert_eq!(MTP12_CHAIN_TOKEN_SLOTS, 16);
        assert_eq!(MTP12_CHAIN_RECURRENT_ELEMENTS, 57_600);
        assert_eq!(MTP12_CHAIN_LOCAL_ROPE_ELEMENTS, 1_920);
        assert_eq!(MTP12_CHAIN_FULL_ROPE_ELEMENTS, 3_840);
    }

    fn official_target_schedule() -> Vec<&'static str> {
        (0..48)
            .map(|index| {
                if index % 6 == 5 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect()
    }

    fn source_matrix(rows: usize, cols: usize) -> TensorRef {
        TensorRef {
            absolute_offset: 0,
            rows: rows as u32,
            cols: cols as u32,
        }
    }

    fn source_geometry() -> (
        TensorRef,
        [SourceLayerWeights; N_LAYERS],
        TensorRef,
        TensorRef,
    ) {
        let embedding = source_matrix(VOCAB, ASSISTANT_HIDDEN);
        let layers = std::array::from_fn(|layer| {
            let q_width = if layer < 3 {
                N_HEADS * LOCAL_HEAD_DIM
            } else {
                N_HEADS * FULL_HEAD_DIM
            };
            SourceLayerWeights {
                q: source_matrix(q_width, ASSISTANT_HIDDEN),
                o: source_matrix(ASSISTANT_HIDDEN, q_width),
                gate: source_matrix(FFN_HIDDEN, ASSISTANT_HIDDEN),
                up: source_matrix(FFN_HIDDEN, ASSISTANT_HIDDEN),
                down: source_matrix(ASSISTANT_HIDDEN, FFN_HIDDEN),
            }
        });
        let pre = source_matrix(ASSISTANT_HIDDEN, TARGET_HIDDEN * 2);
        let post = source_matrix(TARGET_HIDDEN, ASSISTANT_HIDDEN);
        (embedding, layers, pre, post)
    }

    fn synthetic_official_header() -> Vec<u8> {
        let mut object = serde_json::Map::new();
        object.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        for tensor in expected_tensors() {
            object.insert(
                tensor.name,
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": tensor.shape,
                    "data_offsets": [tensor.start, tensor.end],
                }),
            );
        }
        serde_json::to_vec(&serde_json::Value::Object(object)).expect("synthetic header")
    }

    pub(super) fn synthetic_q6k_embedding_row() -> Vec<u8> {
        let mut row = vec![0u8; GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES as usize];
        let half_scales = [0.5f32, -0.125, 0.03125, -0.0078125, 0.001953125];
        for (block_index, block) in row.chunks_exact_mut(Q6_K_BLOCK_BYTES).enumerate() {
            for (index, byte) in block[..128].iter_mut().enumerate() {
                *byte = (block_index as u8)
                    .wrapping_mul(29)
                    .wrapping_add((index as u8).wrapping_mul(37))
                    .wrapping_add(0x53);
            }
            for (index, byte) in block[128..192].iter_mut().enumerate() {
                *byte = (block_index as u8)
                    .wrapping_mul(43)
                    .wrapping_add((index as u8).wrapping_mul(71))
                    .wrapping_add(0xa6);
            }
            for (index, byte) in block[192..208].iter_mut().enumerate() {
                let mut scale = ((block_index * 11 + index * 7) % 31) as i8 - 15;
                if scale == 0 {
                    scale = 1;
                }
                *byte = scale as u8;
            }
            let d = half_scales[block_index % half_scales.len()];
            block[208..210].copy_from_slice(&crate::tensor::f32_to_f16_bits(d).to_le_bytes());
        }
        // Keep the sparse assistant's observed input deterministically positive
        // while the other 3,839 values exercise varied codes, signed group
        // scales, fp16 super-scales, and BF16 rounding boundaries.
        row[0] = (row[0] & 0xf0) | 1;
        row[128] = (row[128] & 0xfc) | 2;
        row[192] = 1;
        row[208..210].copy_from_slice(&crate::tensor::f32_to_f16_bits(0.5).to_le_bytes());
        row
    }

    fn decode_scaled_q6k_embedding_row(row: &[u8]) -> Vec<f32> {
        let scale = (TARGET_HIDDEN as f32).sqrt();
        let mut output = Vec::with_capacity(TARGET_HIDDEN);
        for bytes in row.chunks_exact(Q6_K_BLOCK_BYTES) {
            let raw: &[u8; Q6_K_BLOCK_BYTES] = bytes.try_into().expect("exact Q6_K block");
            let block = crate::tensor::Q6KBlock::from_bytes(raw);
            let mut decoded = [0.0f32; Q6_K_BLOCK_VALUES];
            block.dequantize(&mut decoded);
            output.extend(decoded.into_iter().map(|value| value * scale));
        }
        assert_eq!(output.len(), TARGET_HIDDEN);
        output
    }

    fn set_q4_unit_weight(buffer: &Buffer, matrix: Q4TensorRef, row: usize, col: usize) {
        set_q4_signed_unit_weight(buffer, matrix, row, col, 1.0);
    }

    fn set_q4_signed_unit_weight(
        buffer: &Buffer,
        matrix: Q4TensorRef,
        row: usize,
        col: usize,
        value: f32,
    ) {
        assert!(value == 1.0 || value == -1.0);
        assert!(row < matrix.rows as usize);
        assert!(col < matrix.cols as usize);
        let blocks_per_row = matrix.cols as usize / Q4_0_BLOCK_VALUES;
        let block_index = row * blocks_per_row + col / Q4_0_BLOCK_VALUES;
        let byte_offset = matrix.byte_offset as usize + block_index * Q4_0_BLOCK_BYTES;
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                buffer.contents().cast::<u8>().add(byte_offset),
                Q4_0_BLOCK_BYTES,
            )
        };
        let scale = if value > 0.0 { -0.125 } else { 0.125 };
        bytes[..2].copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
        bytes[2..].fill(0x88);
        let within = col % Q4_0_BLOCK_VALUES;
        if within < 16 {
            bytes[2 + within] &= 0xf0;
        } else {
            bytes[2 + within - 16] &= 0x0f;
        }
    }

    pub(super) fn synthetic_assistant(device: &Device) -> Gemma4Mtp12AssistantMetal {
        let (embedding, source_layers, pre, post) = source_geometry();
        let layout =
            Q4Layout::build(embedding, &source_layers, pre, post).expect("synthetic exact layout");
        let packed_q4 = shared_buffer(device, layout.matrix_bytes as usize);
        unsafe {
            std::ptr::write_bytes(
                packed_q4.contents().cast::<u8>(),
                0,
                layout.matrix_bytes as usize,
            );
        }

        // A sparse, non-trivial graph: pre-projection consumes both target
        // halves; local layer 0 and full layer 3 consume their distinct KV
        // layouts; residuals carry the result; tied row 1 wins argmax.
        set_q4_unit_weight(&packed_q4, layout.pre_projection, 0, 0);
        set_q4_unit_weight(&packed_q4, layout.pre_projection, 0, TARGET_HIDDEN);
        for layer in [0usize, 3] {
            set_q4_unit_weight(&packed_q4, layout.layers[layer].q, 0, 0);
            set_q4_unit_weight(&packed_q4, layout.layers[layer].o, 0, 0);
        }
        set_q4_unit_weight(&packed_q4, layout.post_projection, 0, 0);
        set_q4_unit_weight(&packed_q4, layout.embedding, 1, 0);
        set_q4_signed_unit_weight(&packed_q4, layout.embedding, 2, 0, -1.0);

        let ones =
            |count: usize| f32_buffer(device, &vec![1.0f32; count]).expect("synthetic norm buffer");
        let layers = std::array::from_fn(|layer| LayerWeights {
            input_norm: ones(ASSISTANT_HIDDEN),
            post_attention_norm: ones(ASSISTANT_HIDDEN),
            pre_feedforward_norm: ones(ASSISTANT_HIDDEN),
            post_feedforward_norm: ones(ASSISTANT_HIDDEN),
            q_norm: ones(if layer < 3 {
                LOCAL_HEAD_DIM
            } else {
                FULL_HEAD_DIM
            }),
            matrices: layout.layers[layer],
            scale: 1.0,
        });
        let final_norm = ones(ASSISTANT_HIDDEN);
        let pipelines = Mtp12Pipelines::new(device).expect("synthetic MTP12 pipelines");
        let scratch = Mtp12Scratch::new(device);
        let resident_ledger = Gemma4Mtp12ResidentLedger {
            packed_q4_matrix_bytes: packed_q4.length(),
            fixed_scratch_bytes: scratch.byte_len(),
            ..Gemma4Mtp12ResidentLedger::default()
        };
        Gemma4Mtp12AssistantMetal {
            single_position: false,
            fuse: Mtp12FuseFlags::default(),
            shortlist: None,
            dense_bf16: None,
            tree_state: None,
            packed_q4,
            layout,
            pipelines,
            layers,
            final_norm,
            scratch,
            queue: device.new_command_queue(),
            source_path: PathBuf::from("<synthetic-gemma4-mtp12>"),
            resident_ledger,
        }
    }

    pub(super) fn synthetic_kv(
        kv_heads: usize,
        head_dim: usize,
        logical_len: usize,
        max_positions: usize,
        phase: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut compact_key = vec![0.0f32; kv_heads * logical_len * head_dim];
        let mut compact_value = vec![0.0f32; kv_heads * logical_len * head_dim];
        let mut physical_key = vec![-123.0f32; kv_heads * max_positions * head_dim];
        let mut physical_value = vec![123.0f32; kv_heads * max_positions * head_dim];
        for head in 0..kv_heads {
            for position in 0..logical_len {
                for dimension in 0..head_dim {
                    let compact = (head * logical_len + position) * head_dim + dimension;
                    let physical = (head * max_positions + position) * head_dim + dimension;
                    let index = phase + head * 7 + position * 5 + dimension;
                    let key = (index as i32 % 17 - 8) as f32 * 0.03125;
                    let value = (index as i32 % 23 - 11) as f32 * 0.015625;
                    compact_key[compact] = key;
                    compact_value[compact] = value;
                    physical_key[physical] = key;
                    physical_value[physical] = value;
                }
            }
        }
        (compact_key, compact_value, physical_key, physical_value)
    }

    pub(super) fn f32_view<'a>(
        buffer: &'a Buffer,
        byte_offset: u64,
        elements: usize,
    ) -> Gemma4Mtp12MetalBufferView<'a> {
        Gemma4Mtp12MetalBufferView {
            buffer,
            byte_offset,
            byte_len: (elements * std::mem::size_of::<f32>()) as u64,
        }
    }

    #[test]
    fn official_12b_geometry_is_not_the_26b_geometry() {
        assert_eq!(TARGET_HIDDEN, 3_840);
        assert_eq!(FULL_KV_HEADS, 1);
        assert_eq!(GEMMA4_12B_MTP_SLIDING_HOST_LAYER, 46);
        assert_eq!(GEMMA4_12B_MTP_FULL_HOST_LAYER, 47);
        assert_eq!(EXPECTED_FILE_BYTES, 845_719_296);
        assert_eq!(GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES, 3_150);
        assert_eq!(GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES, 825_753_600);
        assert_eq!(
            GEMMA4_12B_MTP_ASSISTANT_SHA256,
            "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"
        );
        assert_eq!(
            GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
            "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
        );
    }

    #[test]
    fn shared_kv_pair_is_derived_from_the_exact_target_schedule() {
        let schedule = official_target_schedule();
        assert_eq!(
            validate_gemma4_12b_shared_kv_schedule(&schedule).expect("official schedule"),
            (46, 47)
        );
        let mut wrong = schedule.clone();
        wrong[41] = "sliding_attention";
        assert!(validate_gemma4_12b_shared_kv_schedule(&wrong).is_err());
        assert!(validate_gemma4_12b_shared_kv_schedule(&schedule[..47]).is_err());
    }

    #[test]
    fn tensor_table_is_contiguous_and_pins_the_12b_projection_tail() {
        let tensors = expected_tensors();
        assert_eq!(tensors.len(), 48);
        assert_eq!(tensors.first().map(|tensor| tensor.start), Some(0));
        assert_eq!(
            tensors.last().map(|tensor| tensor.end),
            Some(EXPECTED_PAYLOAD_BYTES as u64)
        );
        assert!(tensors.windows(2).all(|pair| pair[0].end == pair[1].start));
        let norm = tensors
            .iter()
            .find(|tensor| tensor.name == "model.norm.weight")
            .expect("final norm");
        let post = tensors
            .iter()
            .find(|tensor| tensor.name == "post_projection.weight")
            .expect("post projection");
        let pre = tensors
            .iter()
            .find(|tensor| tensor.name == "pre_projection.weight")
            .expect("pre projection");
        assert_eq!((norm.start, norm.end), (822_118_920, 822_120_968));
        assert_eq!(
            (post.shape.as_slice(), post.start, post.end),
            (&[3_840, 1_024][..], 822_120_968, 829_985_288)
        );
        assert_eq!(
            (pre.shape.as_slice(), pre.start, pre.end),
            (&[1_024, 7_680][..], 829_985_288, 845_713_928)
        );
    }

    #[test]
    fn strict_parser_accepts_only_the_pinned_full_manifest() {
        let header = synthetic_official_header();
        let manifest = parse_and_validate_header(&header, EXPECTED_PAYLOAD_BYTES)
            .expect("pinned synthetic manifest");
        assert_eq!(
            manifest
                .matrix("pre_projection.weight")
                .expect("pre projection")
                .cols,
            7_680
        );

        let mut value: serde_json::Value = serde_json::from_slice(&header).expect("header JSON");
        value["post_projection.weight"]["shape"] = serde_json::json!([2_816, 1_024]);
        let mutated = serde_json::to_vec(&value).expect("mutated header");
        assert!(parse_and_validate_header(&mutated, EXPECTED_PAYLOAD_BYTES).is_err());
    }

    #[test]
    fn full_q4_pack_has_23_contiguous_matrices_and_exact_size() {
        let (embedding, layers, pre, post) = source_geometry();
        let layout = Q4Layout::build(embedding, &layers, pre, post).expect("Q4 layout");
        assert_eq!(layout.pairs(embedding, &layers, pre, post).len(), 23);
        assert_eq!(layout.matrix_bytes, 237_846_528);
        layout
            .validate(embedding, &layers, pre, post, FULL_Q4_MATRIX_BYTES)
            .expect("exact Q4 coverage");
        assert!(layout
            .validate(embedding, &layers, pre, post, FULL_Q4_MATRIX_BYTES + 1)
            .is_err());
    }

    #[test]
    fn q4_encoder_uses_canonical_signed_max_scale() {
        let encode = |first: f32| {
            let mut values = vec![f32_to_bf16_rne_bits(0.0); 32];
            values[0] = f32_to_bf16_rne_bits(first);
            values[1] = f32_to_bf16_rne_bits(-first);
            let mut packed = vec![0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_row(&values, &mut packed);
            crate::tensor::f16_bits_to_f32(u16::from_le_bytes([packed[0], packed[1]]))
        };
        // Equal magnitudes deliberately select the first element's sign.
        assert_eq!(encode(8.0), -1.0);
        assert_eq!(encode(-8.0), 1.0);

        let zeros = vec![f32_to_bf16_rne_bits(0.0); 32];
        let mut packed = vec![0xffu8; Q4_0_BLOCK_BYTES];
        quantize_q4_0_row(&zeros, &mut packed);
        // Canonical Q4_0 preserves the signed zero produced by 0 / -8.
        assert_eq!(&packed[..2], &[0, 0x80]);
        assert!(packed[2..].iter().all(|byte| *byte == 0x88));
    }

    #[test]
    fn proportional_full_rope_has_only_64_active_pairs() {
        let (cos, sin) = rope_table_values(17, FULL_HEAD_DIM, 1_000_000.0, FULL_HEAD_DIM / 8);
        assert_eq!(cos.len(), 256);
        assert_eq!(sin.len(), 256);
        assert!(cos[64..].iter().all(|value| *value == 1.0));
        assert!(sin[64..].iter().all(|value| *value == 0.0));
        assert!(sin[..64].iter().any(|value| *value != 0.0));
    }

    #[test]
    fn single_position_queries_keep_anchor_and_sliding_crop_fixed() {
        for anchor in [2usize, 1_024, 1_025, 1_031] {
            for step in 0..MTP12_CHAIN_MAX_DRAFTS {
                assert_eq!(chain_query_position(anchor, step, true), anchor);
                assert_eq!(chain_query_position(anchor, step, false), anchor + step);
                let fixed_base = chain_query_position(anchor, step, true)
                    .saturating_sub(LOCAL_WINDOW + 1);
                assert_eq!(fixed_base, anchor.saturating_sub(LOCAL_WINDOW + 1));
            }
        }
        let (_, fixed_sin) = rope_table_values(2, LOCAL_HEAD_DIM, 10_000.0, LOCAL_HEAD_DIM / 2);
        let (_, advanced_sin) = rope_table_values(3, LOCAL_HEAD_DIM, 10_000.0, LOCAL_HEAD_DIM / 2);
        assert_ne!(
            fixed_sin, advanced_sin,
            "the frozen and advancing RoPE controls must differ"
        );
    }

    #[test]
    fn device_views_fail_closed_on_device_bounds_geometry_and_position() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B device-view validation");
            return;
        };
        let registry_id = device.registry_id();
        let row_buffer = shared_buffer(
            &device,
            GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES as usize + 32,
        );
        let good_row = Gemma4Mtp12Q6KEmbeddingRow {
            wire: Gemma4Mtp12MetalBufferView {
                buffer: &row_buffer,
                byte_offset: 14,
                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES,
            },
            hidden: TARGET_HIDDEN,
        };
        good_row.validate(registry_id).expect("exact Q6_K row view");
        assert!(Gemma4Mtp12Q6KEmbeddingRow {
            hidden: 2_816,
            ..good_row
        }
        .validate(registry_id)
        .is_err());
        assert!(Gemma4Mtp12Q6KEmbeddingRow {
            wire: Gemma4Mtp12MetalBufferView {
                byte_offset: 15,
                ..good_row.wire
            },
            ..good_row
        }
        .validate(registry_id)
        .is_err());
        assert!(Gemma4Mtp12Q6KEmbeddingRow {
            wire: Gemma4Mtp12MetalBufferView {
                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES - 2,
                ..good_row.wire
            },
            ..good_row
        }
        .validate(registry_id)
        .is_err());
        assert!(good_row.validate(registry_id.wrapping_add(1)).is_err());

        let undersized_table_buffer = shared_buffer(
            &device,
            GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES as usize + 64,
        );
        let table = Gemma4Mtp12Q6KEmbeddingTable {
            wire: Gemma4Mtp12MetalBufferView {
                buffer: &undersized_table_buffer,
                byte_offset: 32,
                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES,
            },
            hidden: TARGET_HIDDEN,
            vocab: VOCAB,
            target_model_sha256: GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
        };
        assert!(
            table.validate(registry_id).is_err(),
            "table bounds must fail"
        );
        assert!(Gemma4Mtp12Q6KEmbeddingTable {
            hidden: 2_816,
            ..table
        }
        .validate(registry_id)
        .is_err());
        assert!(Gemma4Mtp12Q6KEmbeddingTable {
            target_model_sha256: "not-the-pinned-target",
            ..table
        }
        .validate(registry_id)
        .is_err());

        let recurrent_buffer = shared_buffer(&device, TARGET_HIDDEN * 4 + 32);
        let recurrent = Gemma4Mtp12DeviceRecurrentHidden {
            values: f32_view(&recurrent_buffer, 16, TARGET_HIDDEN),
            hidden: TARGET_HIDDEN,
        };
        recurrent
            .validate(registry_id)
            .expect("exact resident recurrent view");
        assert!(Gemma4Mtp12DeviceRecurrentHidden {
            hidden: 2_816,
            ..recurrent
        }
        .validate(registry_id)
        .is_err());
        assert!(recurrent.validate(registry_id.wrapping_add(1)).is_err());

        let max_positions = 4usize;
        let kv_bytes = LOCAL_KV_HEADS * max_positions * LOCAL_HEAD_DIM * 4;
        let kv_buffer = shared_buffer(&device, kv_bytes + 32);
        let good_sliding = Gemma4Mtp12DeviceKv {
            key: Gemma4Mtp12MetalBufferView {
                buffer: &kv_buffer,
                byte_offset: 16,
                byte_len: kv_bytes as u64,
            },
            value: Gemma4Mtp12MetalBufferView {
                buffer: &kv_buffer,
                byte_offset: 16,
                byte_len: kv_bytes as u64,
            },
            source_layer: GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            kv_heads: LOCAL_KV_HEADS,
            head_dim: LOCAL_HEAD_DIM,
            max_positions,
        };
        good_sliding
            .validate(
                registry_id,
                GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
                LOCAL_KV_HEADS,
                LOCAL_HEAD_DIM,
                "sliding",
            )
            .expect("exact sliding device KV");
        assert!(Gemma4Mtp12DeviceKv {
            source_layer: 28,
            ..good_sliding
        }
        .validate(
            registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )
        .is_err());
        assert!(Gemma4Mtp12DeviceKv {
            key: Gemma4Mtp12MetalBufferView {
                byte_offset: 17,
                ..good_sliding.key
            },
            ..good_sliding
        }
        .validate(
            registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )
        .is_err());
        assert!(Gemma4Mtp12DeviceKv {
            value: Gemma4Mtp12MetalBufferView {
                byte_offset: 36,
                ..good_sliding.value
            },
            ..good_sliding
        }
        .validate(
            registry_id,
            GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            "sliding",
        )
        .is_err());
        assert!(validate_device_chain_positions(16, 15, 2, 2, 8).is_err());
        assert!(validate_device_chain_positions(16, 16, 0, 2, 8).is_err());
        assert!(validate_device_chain_positions(16, 16, 3, 2, 8).is_err());
        assert!(validate_device_chain_positions(8, 8, 2, 2, 8).is_err());
        assert_eq!(
            validate_device_chain_positions(16, 16, 2, 2, 8)
                .expect("fixed prefix with advancing proposal positions"),
            9
        );
        assert!(validate_device_chain_positions(16, 16, 2, 2, 15).is_err());
        assert_eq!(
            validate_device_chain_positions(17, 17, 2, 2, 15)
                .expect("W16 chain fits through final draft position"),
            16
        );
    }

    #[test]
    fn device_fed_k1_matches_cpu_scaffold_bit_for_bit() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B device-fed K=1 parity");
            return;
        };
        let mut assistant = synthetic_assistant(&device);
        let logical_position = 2usize;
        let max_positions = 4usize;

        let q6_row = synthetic_q6k_embedding_row();
        let target_scaled_embedding = decode_scaled_q6k_embedding_row(&q6_row);
        let pending_target_hidden: Vec<f32> = (0..TARGET_HIDDEN)
            .map(|index| (index as i32 % 13 - 6) as f32 * 0.0078125)
            .collect();
        let (sliding_key, sliding_value, sliding_key_physical, sliding_value_physical) =
            synthetic_kv(
                LOCAL_KV_HEADS,
                LOCAL_HEAD_DIM,
                logical_position,
                max_positions,
                3,
            );
        let (full_key, full_value, full_key_physical, full_value_physical) = synthetic_kv(
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            logical_position,
            max_positions,
            19,
        );

        let cpu = assistant
            .propose_k1(
                &target_scaled_embedding,
                &pending_target_hidden,
                Gemma4Mtp12CpuKv {
                    key: &sliding_key,
                    value: &sliding_value,
                    kv_heads: LOCAL_KV_HEADS,
                    head_dim: LOCAL_HEAD_DIM,
                    kv_len: logical_position,
                },
                Gemma4Mtp12CpuKv {
                    key: &full_key,
                    value: &full_value,
                    kv_heads: FULL_KV_HEADS,
                    head_dim: FULL_HEAD_DIM,
                    kv_len: logical_position,
                },
            )
            .expect("CPU-fed K=1 oracle");
        assert!(
            matches!(cpu.token, 1 | 2),
            "synthetic graph must exercise a non-zero logit; token={} logits[0..3]={:?} recurrent0={}",
            cpu.token,
            &cpu.logits[..3],
            cpu.recurrent_hidden[0]
        );

        let q6_offset = 14u64;
        let q6_buffer = shared_buffer(&device, q6_row.len() + q6_offset as usize + 16);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q6_row.as_ptr(),
                q6_buffer.contents().cast::<u8>().add(q6_offset as usize),
                q6_row.len(),
            );
        }
        let kv_offset = 32u64;
        let sliding_key_buffer = shared_buffer(
            &device,
            std::mem::size_of_val(sliding_key_physical.as_slice()) + kv_offset as usize + 16,
        );
        let sliding_value_buffer = shared_buffer(
            &device,
            std::mem::size_of_val(sliding_value_physical.as_slice()) + kv_offset as usize + 16,
        );
        let full_key_buffer = shared_buffer(
            &device,
            std::mem::size_of_val(full_key_physical.as_slice()) + kv_offset as usize + 16,
        );
        let full_value_buffer = shared_buffer(
            &device,
            std::mem::size_of_val(full_value_physical.as_slice()) + kv_offset as usize + 16,
        );
        write_buffer_f32_at(&sliding_key_buffer, kv_offset, &sliding_key_physical)
            .expect("sliding key upload for fixture");
        write_buffer_f32_at(&sliding_value_buffer, kv_offset, &sliding_value_physical)
            .expect("sliding value upload for fixture");
        write_buffer_f32_at(&full_key_buffer, kv_offset, &full_key_physical)
            .expect("full key upload for fixture");
        write_buffer_f32_at(&full_value_buffer, kv_offset, &full_value_physical)
            .expect("full value upload for fixture");
        let sliding_device = Gemma4Mtp12DeviceKv {
            key: f32_view(&sliding_key_buffer, kv_offset, sliding_key_physical.len()),
            value: f32_view(
                &sliding_value_buffer,
                kv_offset,
                sliding_value_physical.len(),
            ),
            source_layer: GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            kv_heads: LOCAL_KV_HEADS,
            head_dim: LOCAL_HEAD_DIM,
            max_positions,
        };
        let full_device = Gemma4Mtp12DeviceKv {
            key: f32_view(&full_key_buffer, kv_offset, full_key_physical.len()),
            value: f32_view(&full_value_buffer, kv_offset, full_value_physical.len()),
            source_layer: GEMMA4_12B_MTP_FULL_HOST_LAYER,
            kv_heads: FULL_KV_HEADS,
            head_dim: FULL_HEAD_DIM,
            max_positions,
        };
        let device_fed = assistant
            .propose_k1_device(
                Gemma4Mtp12Q6KEmbeddingRow {
                    wire: Gemma4Mtp12MetalBufferView {
                        buffer: &q6_buffer,
                        byte_offset: q6_offset,
                        byte_len: q6_row.len() as u64,
                    },
                    hidden: TARGET_HIDDEN,
                },
                &pending_target_hidden,
                sliding_device,
                full_device,
                logical_position,
            )
            .expect("device-fed K=1");

        assert_eq!(device_fed.token, cpu.token);
        assert_eq!(
            device_fed
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            cpu.logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "device-fed logits must be bit-identical to CPU-fed logits"
        );
        assert_eq!(
            device_fed
                .recurrent_hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            cpu.recurrent_hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "device-fed recurrent hidden must be bit-identical"
        );
        let mut gathered = vec![0.0f32; TARGET_HIDDEN];
        read_buffer_f32(&assistant.scratch.pre_input, &mut gathered)
            .expect("read gathered target embedding");
        assert_eq!(
            gathered
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            target_scaled_embedding
                .iter()
                .copied()
                .map(round_to_bf16_f32)
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            "GPU Q6_K gather + sqrt(3840) scale must match the CPU scaffold"
        );
    }

    #[test]
    fn resident_chain_k1_through_k15_matches_repeated_device_k1_bit_for_bit() {
        resident_chain_matches_repeated_device_k1(false, 2);
    }

    #[test]
    fn resident_chain_single_position_k1_through_k15_matches_fixed_k1_bit_for_bit() {
        resident_chain_matches_repeated_device_k1(true, 2);
    }

    #[test]
    fn resident_chain_single_position_beyond_sliding_window_matches_fixed_k1_bit_for_bit() {
        resident_chain_matches_repeated_device_k1(true, 1_031);
    }

    fn resident_chain_matches_repeated_device_k1(single_position: bool, logical_position: usize) {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B resident-chain parity");
            return;
        };
        let mut assistant = synthetic_assistant(&device);
        assistant.single_position = single_position;
        let anchor_token = 7u32;
        let max_positions = logical_position + 30;
        let initial_hidden: Vec<f32> = (0..TARGET_HIDDEN)
            .map(|index| (index as i32 % 13 - 6) as f32 * 0.0078125)
            .collect();

        let table_offset = 32u64;
        let table_buffer = shared_buffer(
            &device,
            (table_offset + GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES + 32) as usize,
        );
        let q6_row = synthetic_q6k_embedding_row();
        // The sparse assistant deterministically emits token 1 or 2.  Seed
        // those rows plus the anchor; any zero row remains equally visible to
        // the chain and the repeated-K1 oracle if the graph changes.
        for token in [anchor_token, 1, 2] {
            let row_offset = table_offset as usize
                + token as usize * GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q6_row.as_ptr(),
                    table_buffer.contents().cast::<u8>().add(row_offset),
                    q6_row.len(),
                );
            }
        }
        let target_table = Gemma4Mtp12Q6KEmbeddingTable {
            wire: Gemma4Mtp12MetalBufferView {
                buffer: &table_buffer,
                byte_offset: table_offset,
                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES,
            },
            hidden: TARGET_HIDDEN,
            vocab: VOCAB,
            target_model_sha256: GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
        };

        let hidden_offset = 16u64;
        let initial_hidden_buffer = shared_buffer(
            &device,
            hidden_offset as usize + std::mem::size_of_val(initial_hidden.as_slice()) + 16,
        );
        write_buffer_f32_at(&initial_hidden_buffer, hidden_offset, &initial_hidden)
            .expect("resident initial-hidden fixture upload");
        let initial_hidden_device = Gemma4Mtp12DeviceRecurrentHidden {
            values: f32_view(&initial_hidden_buffer, hidden_offset, TARGET_HIDDEN),
            hidden: TARGET_HIDDEN,
        };

        let (_, _, sliding_key_physical, sliding_value_physical) = synthetic_kv(
            LOCAL_KV_HEADS,
            LOCAL_HEAD_DIM,
            logical_position,
            max_positions,
            3,
        );
        let (_, _, full_key_physical, full_value_physical) = synthetic_kv(
            FULL_KV_HEADS,
            FULL_HEAD_DIM,
            logical_position,
            max_positions,
            19,
        );
        let kv_offset = 32u64;
        let make_kv_buffer = |values: &[f32]| {
            let buffer = shared_buffer(
                &device,
                kv_offset as usize + std::mem::size_of_val(values) + 16,
            );
            write_buffer_f32_at(&buffer, kv_offset, values).expect("resident KV fixture upload");
            buffer
        };
        let sliding_key_buffer = make_kv_buffer(&sliding_key_physical);
        let sliding_value_buffer = make_kv_buffer(&sliding_value_physical);
        let full_key_buffer = make_kv_buffer(&full_key_physical);
        let full_value_buffer = make_kv_buffer(&full_value_physical);
        let sliding_device = Gemma4Mtp12DeviceKv {
            key: f32_view(&sliding_key_buffer, kv_offset, sliding_key_physical.len()),
            value: f32_view(
                &sliding_value_buffer,
                kv_offset,
                sliding_value_physical.len(),
            ),
            source_layer: GEMMA4_12B_MTP_SLIDING_HOST_LAYER,
            kv_heads: LOCAL_KV_HEADS,
            head_dim: LOCAL_HEAD_DIM,
            max_positions,
        };
        let full_device = Gemma4Mtp12DeviceKv {
            key: f32_view(&full_key_buffer, kv_offset, full_key_physical.len()),
            value: f32_view(&full_value_buffer, kv_offset, full_value_physical.len()),
            source_layer: GEMMA4_12B_MTP_FULL_HOST_LAYER,
            kv_heads: FULL_KV_HEADS,
            head_dim: FULL_HEAD_DIM,
            max_positions,
        };

        for draft_k in 1usize..=MTP12_CHAIN_MAX_DRAFTS {
            let mut oracle_token = anchor_token;
            let mut oracle_hidden = initial_hidden.clone();
            let mut oracle_tokens = Vec::with_capacity(draft_k);
            let mut oracle_recurrent = Vec::with_capacity(draft_k * TARGET_HIDDEN);
            for _ in 0..draft_k {
                let row_offset =
                    table_offset + oracle_token as u64 * GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES;
                let proposal = assistant
                    .propose_k1_device_at_position(
                        Gemma4Mtp12Q6KEmbeddingRow {
                            wire: Gemma4Mtp12MetalBufferView {
                                buffer: &table_buffer,
                                byte_offset: row_offset,
                                byte_len: GEMMA4_12B_MTP_Q6K_EMBEDDING_ROW_BYTES,
                            },
                            hidden: TARGET_HIDDEN,
                        },
                        &oracle_hidden,
                        sliding_device,
                        full_device,
                        logical_position,
                        chain_query_position(
                            logical_position, oracle_tokens.len(), single_position,
                        ),
                    )
                    .expect("repeated device-fed K=1 oracle");
                oracle_token = proposal.token;
                oracle_hidden = proposal.recurrent_hidden;
                oracle_tokens.push(oracle_token);
                oracle_recurrent.extend_from_slice(&oracle_hidden);
            }

            let chain = assistant
                .propose_chain_device_resident(
                    anchor_token,
                    initial_hidden_device,
                    target_table,
                    sliding_device,
                    full_device,
                    logical_position,
                    logical_position,
                    draft_k,
                )
                .expect("one-wait resident chain");
            assert_eq!(chain.tokens, oracle_tokens, "K={draft_k} token parity");
            assert_eq!(chain.ledger.draft_k, draft_k as u32);
            assert_eq!(chain.ledger.command_buffers, 1);
            assert_eq!(chain.ledger.command_buffer_waits, 1);
            assert_eq!(
                chain.ledger.target_q6k_table_alias_bytes,
                GEMMA4_12B_MTP_Q6K_EMBEDDING_TABLE_BYTES
            );
            assert_eq!(
                chain.ledger.assistant_matrix_read_bytes,
                FULL_Q4_MATRIX_BYTES * draft_k as u64
            );
            assert_eq!(
                chain.ledger.target_kv_read_bytes,
                (0..draft_k)
                    .map(|step| {
                        let compact_base =
                            chain_query_position(logical_position, step, single_position)
                                .saturating_sub(LOCAL_WINDOW + 1)
                                .min(logical_position);
                        target_kv_read_bytes(logical_position - compact_base, logical_position)
                            .expect("fixture KV ledger")
                    })
                    .sum::<u64>()
            );
            assert_eq!(
                chain.ledger.dynamic_attention_scratch_bytes,
                (N_HEADS * logical_position * std::mem::size_of::<f32>()) as u64
            );
            assert_eq!(
                chain.ledger.resident_chain_state_bytes,
                ((MTP12_CHAIN_MAX_DRAFTS + 1) * TARGET_HIDDEN * std::mem::size_of::<f32>()
                    + (MTP12_CHAIN_MAX_DRAFTS + 1) * std::mem::size_of::<u32>())
                    as u64
            );
            assert_eq!(chain.ledger.initial_hidden_upload_bytes, 0);
            assert_eq!(
                chain.ledger.readback_bytes,
                (draft_k * std::mem::size_of::<u32>()) as u64,
                "production chain must read only token ids"
            );
            let mut resident_recurrent = vec![0.0f32; draft_k * TARGET_HIDDEN];
            read_buffer_f32(
                &assistant.scratch.chain_recurrent_hidden,
                &mut resident_recurrent,
            )
            .expect("test-only resident recurrent readback");
            assert_eq!(
                resident_recurrent
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                oracle_recurrent
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "K={draft_k} recurrent rows must be bit-identical to repeated K=1"
            );

            if draft_k == 4 {
                let staged = assistant
                    .propose_chain_from_cpu_hidden(
                        anchor_token,
                        &initial_hidden,
                        target_table,
                        sliding_device,
                        full_device,
                        logical_position,
                        logical_position,
                        draft_k,
                    )
                    .expect("15 KB CPU-hidden staging wrapper");
                assert_eq!(staged.tokens, oracle_tokens);
                assert_eq!(
                    staged.ledger.initial_hidden_upload_bytes,
                    (TARGET_HIDDEN * std::mem::size_of::<f32>()) as u64
                );
                assert_eq!(staged.ledger.command_buffers, 1);
                assert_eq!(staged.ledger.command_buffer_waits, 1);
                assert_eq!(staged.ledger.readback_bytes, 4 * 4);
                let mut staged_recurrent = vec![0.0f32; draft_k * TARGET_HIDDEN];
                read_buffer_f32(
                    &assistant.scratch.chain_recurrent_hidden,
                    &mut staged_recurrent,
                )
                .expect("test-only staged-chain recurrent readback");
                assert_eq!(
                    staged_recurrent
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    oracle_recurrent
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }

            if draft_k == 8 {
                let baseline_tokens = chain.tokens.clone();
                let baseline_recurrent = resident_recurrent;
                let poison_tail =
                    |buffer: &Buffer, kv_heads: usize, head_dim: usize, phase: f32| {
                        let count = kv_heads * max_positions * head_dim;
                        let values = unsafe {
                            std::slice::from_raw_parts_mut(
                                buffer
                                    .contents()
                                    .cast::<u8>()
                                    .add(kv_offset as usize)
                                    .cast::<f32>(),
                                count,
                            )
                        };
                        for head in 0..kv_heads {
                            for position in logical_position..max_positions {
                                for dimension in 0..head_dim {
                                    let index =
                                        (head * max_positions + position) * head_dim + dimension;
                                    values[index] =
                                        phase + (head * 31 + position * 7 + dimension % 19) as f32;
                                }
                            }
                        }
                    };
                poison_tail(
                    &sliding_key_buffer,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    10_000.0,
                );
                poison_tail(
                    &sliding_value_buffer,
                    LOCAL_KV_HEADS,
                    LOCAL_HEAD_DIM,
                    -20_000.0,
                );
                poison_tail(&full_key_buffer, FULL_KV_HEADS, FULL_HEAD_DIM, 30_000.0);
                poison_tail(&full_value_buffer, FULL_KV_HEADS, FULL_HEAD_DIM, -40_000.0);
                let poisoned = assistant
                    .propose_chain_device_resident(
                        anchor_token,
                        initial_hidden_device,
                        target_table,
                        sliding_device,
                        full_device,
                        logical_position,
                        logical_position,
                        8,
                    )
                    .expect("K=8 chain with poisoned unverified KV tail");
                assert_eq!(
                    poisoned.tokens, baseline_tokens,
                    "slots at and beyond target_kv_len must not affect draft tokens"
                );
                let mut poisoned_recurrent = vec![0.0f32; 8 * TARGET_HIDDEN];
                read_buffer_f32(
                    &assistant.scratch.chain_recurrent_hidden,
                    &mut poisoned_recurrent,
                )
                .expect("test-only poisoned-tail recurrent readback");
                assert_eq!(
                    poisoned_recurrent
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    baseline_recurrent
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "slots at and beyond target_kv_len must not affect recurrent rows"
                );
            }
        }

        assert!(assistant
            .propose_chain_device_resident(
                anchor_token,
                initial_hidden_device,
                target_table,
                sliding_device,
                full_device,
                logical_position,
                logical_position,
                0,
            )
            .is_err());
        assert!(assistant
            .propose_chain_device_resident(
                anchor_token,
                initial_hidden_device,
                target_table,
                sliding_device,
                full_device,
                logical_position,
                logical_position,
                MTP12_CHAIN_MAX_DRAFTS + 1,
            )
            .is_err());
        assert!(assistant
            .propose_chain_device_resident(
                VOCAB as u32,
                initial_hidden_device,
                target_table,
                sliding_device,
                full_device,
                logical_position,
                logical_position,
                1,
            )
            .is_err());
    }

    #[test]
    fn assistant_argmax_ties_choose_first_vocab_index() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B assistant argmax tie test");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("MTP12 argmax pipeline");
        let mut logits = vec![-8.0f32; 16];
        logits[3] = 4.0;
        logits[9] = 4.0;
        let logits = f32_buffer(&device, &logits).expect("argmax tie fixture");
        let output = shared_buffer(&device, std::mem::size_of::<u32>());
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_argmax_at_offset(encoder, &pipelines.argmax, &logits, &output, 0, 16);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(unsafe { *output.contents().cast::<u32>() }, 3);
    }


    /// Adversarial assistant-attention fixture over a target-shaped KV view: queries in
    /// three magnitude bands, LCG-noised K/V with a per-position marker so any dropped or
    /// duplicated position changes the output.
    fn mtp12_attention_fixture(
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ seed;
        let mut noise = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (((state >> 40) & 0xFFFF) as f32 / 65_535.0 - 0.5) * 2.0
        };
        let query: Vec<f32> = (0..N_HEADS * head_dim)
            .map(|index| {
                let band = match (index / head_dim) % 3 {
                    0 => 0.25,
                    1 => 1.0,
                    _ => 3.0,
                };
                (((index * 17) % 61) as f32 - 30.0) * 0.03125 * band + noise() * 0.125
            })
            .collect();
        let mut keys = vec![0.0f32; kv_heads * capacity * head_dim];
        let mut values = vec![0.0f32; keys.len()];
        for kv_head in 0..kv_heads {
            for position in 0..capacity {
                let marker = (position % 97) as f32 * 0.01;
                for dim in 0..head_dim {
                    let index = (kv_head * capacity + position) * head_dim + dim;
                    keys[index] =
                        (((index * 13) % 47) as f32 - 23.0) * 0.015625 + noise() * 0.0625 + marker;
                    values[index] =
                        (((index * 19) % 53) as f32 - 26.0) * 0.015625 + noise() * 0.0625 - marker;
                }
            }
        }
        (query, keys, values)
    }

    /// Encodes one assistant attention (established or V2 grids) over the fixture and
    /// returns the context output.
    #[allow(clippy::too_many_arguments)]
    fn mtp12_attention_run(
        device: &Device,
        pipelines: &Mtp12Pipelines,
        v2: bool,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        compact_base: usize,
        position_count: usize,
    ) -> Vec<f32> {
        mtp12_attention_run_ex(
            device,
            pipelines,
            v2,
            0,
            Mtp12AttnGeom::TODAY,
            query,
            keys,
            values,
            kv_heads,
            head_dim,
            capacity,
            compact_base,
            position_count,
        )
    }

    /// As [`mtp12_attention_run`], with `softmax_ctx` selecting one of the
    /// `CAMELID_GEMMA4_MTP12_FUSE_SOFTMAX_CTX` levels in place of the softmax +
    /// context pair (1 = fused, 2 = stats + 16-deep, 3 = stats + 4-deep).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp12_attention_run_ex(
        device: &Device,
        pipelines: &Mtp12Pipelines,
        v2: bool,
        softmax_ctx: u8,
        attn: Mtp12AttnGeom,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        compact_base: usize,
        position_count: usize,
    ) -> Vec<f32> {
        let logical_len = compact_base + position_count;
        assert!(logical_len <= capacity);
        let query_buf = f32_buffer(device, query).expect("query buffer");
        let keys_buf = f32_buffer(device, keys).expect("keys buffer");
        let values_buf = f32_buffer(device, values).expect("values buffer");
        let scores = shared_buffer(device, N_HEADS * logical_len * 4);
        let output = shared_buffer(device, N_HEADS * head_dim * 4);
        let stats = shared_buffer(device, N_HEADS * 2 * 4);
        write_buffer_f32(&output, &vec![f32::NAN; N_HEADS * head_dim]).expect("poison");
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode_attention_impl_ex(
            encoder,
            pipelines,
            &query_buf,
            &keys_buf,
            &values_buf,
            0,
            0,
            capacity,
            &scores,
            &output,
            kv_heads,
            head_dim,
            logical_len,
            compact_base,
            position_count,
            v2,
            softmax_ctx,
            Some(&stats),
            attn,
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let mut out = vec![0.0f32; N_HEADS * head_dim];
        read_buffer_f32(&output, &mut out).expect("read output");
        out
    }

    /// The V2 grids must reproduce the established assistant attention BIT-FOR-BIT on
    /// both 12B head shapes, at depths past the 32-lane stride, at a compacted sliding
    /// base, and with a ragged (non-multiple-of-four) position count.
    #[test]
    fn mtp12_attention_v2_matches_established_kernels_bit_for_bit() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B MTP attention V2 parity");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("Gemma 4 12B MTP pipelines");
        // (kv_heads, head_dim, capacity, compact_base, position_count)
        let cases = [
            (LOCAL_KV_HEADS, LOCAL_HEAD_DIM, 1_100, 0, 64),
            (LOCAL_KV_HEADS, LOCAL_HEAD_DIM, 1_100, 70, 1_025),
            (LOCAL_KV_HEADS, LOCAL_HEAD_DIM, 1_100, 3, 517),
            (LOCAL_KV_HEADS, LOCAL_HEAD_DIM, 1_100, 0, 1),
            (LOCAL_KV_HEADS, LOCAL_HEAD_DIM, 1_100, 1, 2),
            (FULL_KV_HEADS, FULL_HEAD_DIM, 1_100, 0, 1_031),
            (FULL_KV_HEADS, FULL_HEAD_DIM, 1_100, 0, 130),
            (FULL_KV_HEADS, FULL_HEAD_DIM, 1_100, 0, 5),
        ];
        for (index, &(kv_heads, head_dim, capacity, compact_base, position_count)) in
            cases.iter().enumerate()
        {
            let (query, keys, values) =
                mtp12_attention_fixture(kv_heads, head_dim, capacity, index as u64);
            let established = mtp12_attention_run(
                &device,
                &pipelines,
                false,
                &query,
                &keys,
                &values,
                kv_heads,
                head_dim,
                capacity,
                compact_base,
                position_count,
            );
            let candidate = mtp12_attention_run(
                &device,
                &pipelines,
                true,
                &query,
                &keys,
                &values,
                kv_heads,
                head_dim,
                capacity,
                compact_base,
                position_count,
            );
            assert!(
                established.iter().all(|value| value.is_finite()),
                "case {index}: established output is not finite"
            );
            for (element, (expected, actual)) in established.iter().zip(&candidate).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "case {index} (kv {kv_heads} x {head_dim}, base {compact_base}, count {position_count}) element {element}: v2={actual} ({:#010x}) != established={expected} ({:#010x})",
                    actual.to_bits(),
                    expected.to_bits(),
                );
            }
        }
    }

    /// Production-shape timing receipt for the assistant chain's attention: seven
    /// drafts x (three sliding-layer + one full-layer attentions) over the target KV at
    /// several depths, established grids vs V2.  Prints median GPU microseconds per
    /// round and the per-position slope.
    #[test]
    #[ignore]
    fn mtp12_attention_v2_production_receipt() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B MTP attention V2 receipt");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("Gemma 4 12B MTP pipelines");
        const CAPACITY: usize = 2_048;
        const DRAFTS: usize = 7;
        let (query_local, keys_local, values_local) =
            mtp12_attention_fixture(LOCAL_KV_HEADS, LOCAL_HEAD_DIM, CAPACITY, 11);
        let (query_full, keys_full, values_full) =
            mtp12_attention_fixture(FULL_KV_HEADS, FULL_HEAD_DIM, CAPACITY, 12);
        let query_local = f32_buffer(&device, &query_local).expect("query");
        let keys_local = f32_buffer(&device, &keys_local).expect("keys");
        let values_local = f32_buffer(&device, &values_local).expect("values");
        let query_full = f32_buffer(&device, &query_full).expect("query");
        let keys_full = f32_buffer(&device, &keys_full).expect("keys");
        let values_full = f32_buffer(&device, &values_full).expect("values");
        let scores = shared_buffer(&device, N_HEADS * CAPACITY * 4);
        let output = shared_buffer(&device, N_HEADS * FULL_HEAD_DIM * 4);
        let queue = device.new_command_queue();
        let depths = [128usize, 512, 1_024, 2_000];
        for v2 in [false, true] {
            let label = if v2 { "v2" } else { "established" };
            let mut per_depth = Vec::new();
            for &depth in &depths {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    for _draft in 0..DRAFTS {
                        for layer in 0..N_LAYERS {
                            let sliding = layer < 3;
                            let compact_base = if sliding {
                                depth.saturating_sub(LOCAL_WINDOW + 1)
                            } else {
                                0
                            };
                            let position_count = depth - compact_base;
                            let (query, keys, values, kv_heads, head_dim) = if sliding {
                                (
                                    &query_local,
                                    &keys_local,
                                    &values_local,
                                    LOCAL_KV_HEADS,
                                    LOCAL_HEAD_DIM,
                                )
                            } else {
                                (
                                    &query_full,
                                    &keys_full,
                                    &values_full,
                                    FULL_KV_HEADS,
                                    FULL_HEAD_DIM,
                                )
                            };
                            encode_attention_impl(
                                encoder,
                                &pipelines,
                                query,
                                keys,
                                values,
                                0,
                                0,
                                CAPACITY,
                                &scores,
                                &output,
                                kv_heads,
                                head_dim,
                                depth,
                                compact_base,
                                position_count,
                                v2,
                            );
                        }
                    }
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                    let (gpu_us, _) = super::super::command_buffer_gpu_times_us(&command.to_owned());
                    samples.push(gpu_us);
                }
                samples.sort_unstable();
                let median = samples[samples.len() / 2];
                println!(
                    "[mtp12-attn-v2-receipt] {label} depth={depth} drafts={DRAFTS} gpu_us median={median} min={} max={}",
                    samples[0],
                    samples[samples.len() - 1]
                );
                per_depth.push((depth, median));
            }
            let (d0, t0) = per_depth[0];
            let (d1, t1) = per_depth[2];
            println!(
                "[mtp12-attn-v2-receipt] {label} slope({d0}->{d1}) = {:.4} ms/position per {DRAFTS}-draft round",
                (t1 as f64 - t0 as f64) / 1_000.0 / (d1 - d0) as f64
            );
        }
    }

    /// CPU model of the assistant argmax rule: NaN skipped, first index on an
    /// exact tie (+0 and -0 compare equal), 0 when nothing exceeds -inf.
    fn reference_first_index_argmax(logits: &[f32]) -> u32 {
        let mut best = f32::NEG_INFINITY;
        let mut best_id = 0u32;
        for (index, &value) in logits.iter().enumerate() {
            let index = index as u32;
            if !value.is_nan() && (value > best || (value == best && index < best_id)) {
                best = value;
                best_id = index;
            }
        }
        best_id
    }

    /// Run the single-threadgroup and the chunked argmax on `logits` in one
    /// command buffer; returns (single, chunked).
    fn run_argmax_pair(
        device: &Device,
        queue: &metal::CommandQueue,
        pipelines: &Mtp12Pipelines,
        logits: &[f32],
    ) -> (u32, u32) {
        let count = logits.len();
        let logits_buffer = f32_buffer(device, logits).expect("argmax fixture");
        let output = shared_buffer(device, 2 * std::mem::size_of::<u32>());
        let slots = count.div_ceil(MTP12_ARGMAX_CHUNK).max(1);
        let partial_values = shared_buffer(device, slots * std::mem::size_of::<f32>());
        let partial_ids = shared_buffer(device, slots * std::mem::size_of::<u32>());
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_argmax_at_offset(encoder, &pipelines.argmax, &logits_buffer, &output, 0, count);
        encode_argmax_chunked_at_offset(
            encoder,
            pipelines,
            &logits_buffer,
            &partial_values,
            &partial_ids,
            &output,
            std::mem::size_of::<u32>() as u64,
            count,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let out = unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), 2) };
        (out[0], out[1])
    }

    /// The chunked argmax must return the identical token to the
    /// single-threadgroup scan (and to the CPU rule) on every adversarial
    /// layout: ties across chunk boundaries, all-equal, all -inf, all NaN,
    /// signed zero, f32 extremes, +inf, and pseudo-random logits with the
    /// maximum duplicated.  Odd lengths exercise the partial last chunk.
    #[test]
    fn assistant_argmax_chunked_matches_single_threadgroup_on_adversarial_logits() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B assistant chunked argmax test");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("MTP12 argmax pipelines");
        let queue = device.new_command_queue();
        let mut cases: Vec<(String, Vec<f32>)> = Vec::new();
        let with = |base: f32, count: usize, sets: &[(usize, f32)]| {
            let mut logits = vec![base; count];
            for &(index, value) in sets {
                logits[index] = value;
            }
            logits
        };
        cases.push(("existing tie 3/9".into(), with(-8.0, 16, &[(3, 4.0), (9, 4.0)])));
        cases.push((
            "tie across chunk boundary".into(),
            with(-1.0, VOCAB, &[(1023, 7.5), (1024, 7.5)]),
        ));
        cases.push(("tie first/last".into(), with(-1.0, VOCAB, &[(0, 7.5), (VOCAB - 1, 7.5)])));
        cases.push((
            "tie mid/last".into(),
            with(-1.0, VOCAB, &[(1024, 7.5), (VOCAB - 1, 7.5)]),
        ));
        cases.push(("all equal".into(), vec![0.0; VOCAB]));
        cases.push(("all -inf".into(), vec![f32::NEG_INFINITY; VOCAB]));
        cases.push(("all nan".into(), vec![f32::NAN; VOCAB]));
        cases.push(("max at last".into(), with(-1.0, VOCAB, &[(VOCAB - 1, 2.0)])));
        {
            let mut logits = vec![-3.0f32; VOCAB];
            for value in logits.iter_mut().take(10) {
                *value = f32::NAN;
            }
            for value in logits.iter_mut().take(2048).skip(10) {
                *value = f32::NEG_INFINITY;
            }
            logits[200_000] = 1.0e30;
            logits[200_001] = 1.0e30;
            cases.push(("nan/-inf prefix, 1e30 pair".into(), logits));
        }
        cases.push((
            "signed zero tie".into(),
            with(-1.0, VOCAB, &[(5000, -0.0), (70, 0.0)]),
        ));
        cases.push((
            "f32::MAX pair".into(),
            with(f32::MIN, VOCAB, &[(131_072, f32::MAX), (131_071, f32::MAX)]),
        ));
        cases.push((
            "+inf pair".into(),
            with(-1.0e38, VOCAB, &[(9_000, f32::INFINITY), (8_999, f32::INFINITY)]),
        ));
        cases.push(("-f32::MAX floor, one -1".into(), with(-f32::MAX, VOCAB, &[(77_777, -1.0)])));
        for (seed, count) in [
            (1u64, VOCAB),
            (2, VOCAB),
            (3, 1),
            (4, 1023),
            (5, 1025),
            (6, 4097),
            (7, 100_001),
        ] {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678);
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / 16_777_216.0 * 40.0 - 20.0
            };
            let mut logits: Vec<f32> = (0..count).map(|_| next()).collect();
            let peak = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) + 1.0;
            let dupes = 37.min(count);
            for d in 0..dupes {
                let index = ((d * 7919 + 13) * 31) % count;
                logits[index] = peak;
            }
            cases.push((format!("xorshift seed {seed} count {count} x{dupes} peaks"), logits));
        }
        for (label, logits) in &cases {
            let expected = reference_first_index_argmax(logits);
            let (single, chunked) = run_argmax_pair(&device, &queue, &pipelines, logits);
            assert_eq!(single, expected, "{label}: single-threadgroup argmax vs CPU rule");
            assert_eq!(chunked, expected, "{label}: chunked argmax vs CPU rule");
        }
        eprintln!(
            "[mtp12-argmax] chunked == single-threadgroup == CPU rule on {} adversarial cases",
            cases.len()
        );
    }

    /// GPU time per vocabulary argmax, single-threadgroup vs chunked (30
    /// encodes per command buffer, hardware timestamps).
    #[test]
    #[ignore]
    fn assistant_argmax_bench_single_vs_chunked() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping assistant argmax bench");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("MTP12 argmax pipelines");
        let queue = device.new_command_queue();
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let logits: Vec<f32> = (0..VOCAB)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / 16_777_216.0 * 40.0 - 20.0
            })
            .collect();
        let logits_buffer = f32_buffer(&device, &logits).expect("bench logits");
        let output = shared_buffer(&device, std::mem::size_of::<u32>());
        let partial_values = f32_buffer(&device, &vec![0.0; MTP12_ARGMAX_MAX_PARTIALS]).unwrap();
        let partial_ids = shared_buffer(&device, MTP12_ARGMAX_MAX_PARTIALS * 4);
        const REPS: usize = 30;
        for (label, chunked) in [("single-threadgroup mtp12_argmax", false), ("chunked partial+merge", true)] {
            for pass in 0..2 {
                let reps = if pass == 0 { 1 } else { REPS };
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                for _ in 0..reps {
                    if chunked {
                        encode_argmax_chunked_at_offset(
                            encoder,
                            &pipelines,
                            &logits_buffer,
                            &partial_values,
                            &partial_ids,
                            &output,
                            0,
                            VOCAB,
                        );
                    } else {
                        encode_argmax_at_offset(
                            encoder,
                            &pipelines.argmax,
                            &logits_buffer,
                            &output,
                            0,
                            VOCAB,
                        );
                    }
                }
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
                if pass == 1 {
                    let (gpu_us, _) =
                        super::super::command_buffer_gpu_times_us(&command_buffer.to_owned());
                    eprintln!(
                        "[mtp12-argmax] {label:<34} {:8.1} us per argmax (token {})",
                        gpu_us as f64 / REPS as f64,
                        unsafe { *output.contents().cast::<u32>() }
                    );
                }
            }
        }
    }


    #[test]
    fn assistant_shortlist_top_rejects_invalid_settings() {
        assert_eq!(mtp12_shortlist_top(None).unwrap(), 192);
        for top in [1, 128, 192, 256, 2048] {
            assert_eq!(mtp12_shortlist_top(Some(&top.to_string())).unwrap(), top);
        }
        for bad in ["", "0", "2049", "-1", "true", "128 "] {
            assert!(mtp12_shortlist_top(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn assistant_shortlist_matches_masked_full_head_and_stable_top_clusters() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("shortlist pipelines");
        let queue = device.new_command_queue();
        const ROWS: usize = 8193;
        const OFFSET: usize = 18;
        let matrix = Q4TensorRef {
            byte_offset: OFFSET as u64,
            byte_len: (ROWS * ASSISTANT_HIDDEN / 32 * 18) as u64,
            rows: ROWS as u32, cols: ASSISTANT_HIDDEN as u32,
        };
        let weights = shared_buffer(&device, OFFSET + matrix.byte_len as usize);
        let bytes = unsafe { std::slice::from_raw_parts_mut(
            weights.contents().cast::<u8>(), weights.length() as usize) };
        let mut random = 0x9876_4321_abcdu64;
        for block in bytes[OFFSET..].chunks_exact_mut(18) {
            random ^= random << 13; random ^= random >> 7; random ^= random << 17;
            let scale = (random as i16) as f32 / 32768.0;
            block[..2].copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
            for byte in &mut block[2..] {
                random ^= random << 13; random ^= random >> 7; random ^= random << 17;
                *byte = random as u8;
            }
        }
        let query: Vec<f32> = (0..ASSISTANT_HIDDEN)
            .map(|i| if i == 0 { 1.0 } else { ((i * 13 % 71) as f32 - 35.0) / 16.0 }).collect();
        let query_buffer = f32_buffer(&device, &query).unwrap();
        // Only dimension zero contributes; repeated scores exercise stable
        // top-T ties. Remaining query dimensions exercise the real Q4 dot.
        let expected_scores: Vec<f32> = (0..MTP12_SHORTLIST_CLUSTERS)
            .map(|i| (i * 37 % 53) as f32 - 26.0).collect();
        let mut centroids = vec![0.0; MTP12_SHORTLIST_CLUSTERS * ASSISTANT_HIDDEN];
        for (i, score) in expected_scores.iter().enumerate() {
            centroids[i * ASSISTANT_HIDDEN] = *score;
        }
        let clusters: Vec<u16> = (0..ROWS).flat_map(|row| {
            [(row % 2048) as u16, ((row + 71) % 2048) as u16,
             ((row + 193) % 2048) as u16, 0]
        }).collect();
        let cluster_buffer = shared_buffer(&device, clusters.len() * 2);
        unsafe { std::ptr::copy_nonoverlapping(clusters.as_ptr(),
            cluster_buffer.contents().cast::<u16>(), clusters.len()); }
        let mut shortlist = Mtp12Shortlist {
            centroids: f32_buffer(&device, &centroids).unwrap(),
            token_clusters: cluster_buffer,
            scores: shared_buffer(&device, MTP12_SHORTLIST_CLUSTERS * 4),
            selected: shared_buffer(&device, MTP12_SHORTLIST_CLUSTERS * 4), top: 192,
        };
        let full_logits = shared_buffer(&device, ROWS * 4);
        let masked_logits = shared_buffer(&device, ROWS * 4);
        let compact_logits = shared_buffer(&device, ROWS * 4);
        let mut order: Vec<usize> = (0..MTP12_SHORTLIST_CLUSTERS).collect();
        order.sort_by(|&a, &b| expected_scores[b].total_cmp(&expected_scores[a]).then(a.cmp(&b)));
        for top in [1, 17, 128, 192, 256, 2048] {
            shortlist.top = top;
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &query_buffer,
                &full_logits, matrix, false);
            encode_shortlist(encoder, &pipelines, &shortlist, &query_buffer);
            encoder.set_buffer(7, Some(&shortlist.token_clusters), 0);
            encoder.set_buffer(8, Some(&shortlist.selected), 0);
            encode_q4_gemv(encoder, &pipelines.q4_gemv_shortlist, &weights, &query_buffer,
                &masked_logits, matrix, false);
            encode_q4_gemv_shortlist_compact(encoder, &pipelines.q4_gemv_shortlist_compact,
                &weights, &query_buffer, &compact_logits, matrix);
            encoder.end_encoding(); cb.commit(); cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            let selected = unsafe { std::slice::from_raw_parts(
                shortlist.selected.contents().cast::<u32>(), MTP12_SHORTLIST_CLUSTERS) };
            let scores = unsafe { std::slice::from_raw_parts(
                shortlist.scores.contents().cast::<f32>(), MTP12_SHORTLIST_CLUSTERS) };
            assert_eq!(scores, expected_scores, "centroid scores at top={top}");
            let mut expected_selected = vec![0u32; MTP12_SHORTLIST_CLUSTERS];
            for &index in &order[..top as usize] { expected_selected[index] = 1; }
            assert_eq!(selected, expected_selected, "top={top}");
            let full = unsafe { std::slice::from_raw_parts(full_logits.contents().cast::<f32>(), ROWS) };
            let actual = unsafe { std::slice::from_raw_parts(masked_logits.contents().cast::<f32>(), ROWS) };
            let compacted = unsafe { std::slice::from_raw_parts(compact_logits.contents().cast::<f32>(), ROWS) };
            let mut empty_blocks = 0;
            for block in clusters.chunks(256 * 4) {
                empty_blocks += usize::from(block.chunks_exact(4)
                    .all(|ids| ids[..3].iter().all(|&c| expected_selected[c as usize] == 0)));
            }
            if top == 1 { assert!(empty_blocks > 0, "fixture must include all-empty blocks"); }
            if top == 2048 { assert_eq!(empty_blocks, 0); }
            for row in 0..ROWS {
                let admitted = clusters[row * 4..row * 4 + 3].iter()
                    .any(|&c| expected_selected[c as usize] != 0);
                let expected = if admitted { full[row] } else { f32::NEG_INFINITY };
                assert_eq!(actual[row].to_bits(), expected.to_bits(), "top={top} row={row}");
                assert_eq!(compacted[row].to_bits(), expected.to_bits(), "compact top={top} row={row}");
            }
            let expected_token = reference_first_index_argmax(actual);
            let (legacy, chunked) = run_argmax_pair(&device, &queue, &pipelines, compacted);
            assert_eq!(legacy, expected_token);
            assert_eq!(chunked, expected_token);
        }
        unsafe { std::ptr::write_bytes(shortlist.selected.contents().cast::<u8>(), 0,
            MTP12_SHORTLIST_CLUSTERS * 4); }
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_buffer(7, Some(&shortlist.token_clusters), 0);
        encoder.set_buffer(8, Some(&shortlist.selected), 0);
        encode_q4_gemv_shortlist_compact(encoder, &pipelines.q4_gemv_shortlist_compact,
            &weights, &query_buffer, &compact_logits, matrix);
        encoder.end_encoding(); cb.commit(); cb.wait_until_completed();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        let empty = unsafe { std::slice::from_raw_parts(compact_logits.contents().cast::<f32>(), ROWS) };
        assert!(empty.iter().all(|&value| value == f32::NEG_INFINITY));
        assert_eq!(run_argmax_pair(&device, &queue, &pipelines, empty), (0, 0));
    }

    // ---- CAMELID_GEMMA4_MTP12_FUSE_* kernels: bit-for-bit against the established
    // dispatch chains, plus an ignored bandwidth bench --------------------------------

    #[test]
    fn fuse_selectors_parse_strictly() {
        assert!(!mtp12_fuse_flag("X", None).unwrap());
        assert!(!mtp12_fuse_flag("X", Some("0")).unwrap());
        assert!(mtp12_fuse_flag("X", Some("1")).unwrap());
        for bad in ["", " 1", "1 ", "true", "false", "on", "2", "01"] {
            assert!(mtp12_fuse_flag("X", Some(bad)).is_err(), "{bad:?}");
        }
        // Leveled selectors admit exactly 0..=max, one ASCII digit.
        for max in [1u8, 2, 3, 5] {
            for level in 0..=max {
                assert_eq!(
                    mtp12_fuse_level("X", Some(&level.to_string()), max).unwrap(),
                    level
                );
            }
            assert_eq!(mtp12_fuse_level("X", None, max).unwrap(), 0);
            for bad in ["", " 1", "1 ", "true", "01", "-1", "10"] {
                assert!(mtp12_fuse_level("X", Some(bad), max).is_err(), "{bad:?} max={max}");
            }
            if max < 9 {
                assert!(mtp12_fuse_level("X", Some(&(max + 1).to_string()), max).is_err());
            }
        }
        // Levels 0..=3 are the bit-identical dense-GEMV routes gated by
        // gemv_x4_matches_gemv_bit_for_bit; 4 and 5 are the split-block
        // experiment, which is deliberately excluded from that gate.
        assert_eq!(MTP12_FUSE_GEMV_X4_MAX, 5);
        assert_eq!(MTP12_FUSE_GEMV_X4_FIRST_INEXACT, 4);
        for level in 0..MTP12_FUSE_GEMV_X4_FIRST_INEXACT {
            assert!(mtp12_gemv_split_count(level).is_none(), "level {level} must be exact");
        }
        assert_eq!(mtp12_gemv_split_count(4), Some(2));
        assert_eq!(mtp12_gemv_split_count(5), Some(4));
        assert_eq!(mtp12_gemv_staged_rows(None).unwrap(), MTP12_GEMV_STAGED_ROWS_DEFAULT);
        for rows in [1u32, 2, 4, 8, 16, MTP12_GEMV_STAGED_ROWS_MAX] {
            assert_eq!(mtp12_gemv_staged_rows(Some(&rows.to_string())).unwrap(), rows);
        }
        for bad in ["", "0", "65", "-1", " 4", "4 ", "04", "true", "4.0"] {
            assert!(mtp12_gemv_staged_rows(Some(bad)).is_err(), "{bad:?}");
        }
        assert_eq!(Mtp12FuseFlags::default(), Mtp12FuseFlags {
            gemv_x4: 0,
            gemv_staged_rows: MTP12_GEMV_STAGED_ROWS_DEFAULT,
            gate_up: false,
            norm: false,
            qrope: false,
            softmax_ctx: 0,
            head_prefetch: 0,
        });
    }

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn round_bf16_cpu(value: f32) -> f32 {
        let bits = value.to_bits();
        if bits & 0x7fff_ffff > 0x7f80_0000 {
            return f32::from_bits(((bits >> 16) | 0x40) << 16);
        }
        let bias = 0x7fff + ((bits >> 16) & 1);
        f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
    }

    /// BF16-valued activations in roughly [-4, 4] with zeros, negative zeros,
    /// BF16 denormals and tiny normals sprinkled in when `specials` is set.
    pub(super) fn random_bf16_values(count: usize, seed: u64, specials: bool) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..count)
            .map(|i| {
                let r = xorshift(&mut state);
                if specials {
                    match i % 53 {
                        0 => return 0.0,
                        17 => return -0.0,
                        // BF16 denormals: exponent 0, upper mantissa bits only.
                        29 => return f32::from_bits((((r >> 8) as u32) & 0x007f) << 16 | 0x0001_0000),
                        41 => return -f32::from_bits((((r >> 8) as u32) & 0x007f) << 16 | 0x0001_0000),
                        47 => return f32::from_bits(0x0080_0000 + ((((r >> 8) as u32) & 0x3f) << 16)),
                        _ => {}
                    }
                }
                round_bf16_cpu((r >> 40) as f32 / 16_777_216.0 * 8.0 - 4.0)
            })
            .collect()
    }

    /// Random Q4_0 blocks (f16 scale in about [-1, 1), random nibbles) at
    /// `byte_offset` inside a fresh buffer.
    pub(super) fn random_q4_weights(device: &Device, byte_offset: usize, byte_len: usize, seed: u64) -> Buffer {
        let weights = shared_buffer(device, byte_offset + byte_len);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(weights.contents().cast::<u8>(), weights.length() as usize)
        };
        let mut state = 0x9876_4321_abcdu64 ^ seed.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        for block in bytes[byte_offset..byte_offset + byte_len].chunks_exact_mut(18) {
            let scale = (xorshift(&mut state) as i16) as f32 / 32768.0;
            block[..2].copy_from_slice(&crate::tensor::f32_to_f16_bits(scale).to_le_bytes());
            for byte in &mut block[2..] {
                *byte = xorshift(&mut state) as u8;
            }
        }
        weights
    }

    fn read_f32s(buffer: &Buffer, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count];
        read_buffer_f32(buffer, &mut out).expect("read");
        out
    }

    fn assert_bits_equal(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len(), "{label}: length");
        for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                a.to_bits(), e.to_bits(),
                "{label} element {i}: fused={a} ({:#010x}) != established={e} ({:#010x})",
                a.to_bits(), e.to_bits(),
            );
        }
    }

    const FUSE_GEMV_SHAPES: [(usize, usize); 6] =
        [(1_024, 3_840), (4_096, 1_024), (8_192, 1_024), (1_024, 4_096), (1_024, 8_192), (3_840, 1_024)];

    /// Rows-per-simdgroup values covered by the staged-GEMV bit gate: divisors
    /// and non-divisors of every production row count (3 and 6 leave a partial
    /// last group), the default, and the clamp maximum.
    const FUSE_GEMV_STAGED_ROWS: [u32; 8] = [1, 2, 3, 4, 6, 8, 16, MTP12_GEMV_STAGED_ROWS_MAX];

    /// Every order-preserving dense-GEMV variant must reproduce
    /// `mtp12_q4_0_gemv` bit-for-bit at the six production shapes, both
    /// rounding modes, a nonzero weight offset, a nonzero output offset, and
    /// BF16 inputs including zeros and denormals:
    /// `mtp12_q4_0_gemv_x4` (`FUSE_GEMV_X4=1`, four rows per 128-thread group)
    /// and `mtp12_q4_0_gemv_staged` (`FUSE_GEMV_X4=2/3`, register-staged
    /// inputs) at every `CAMELID_GEMMA4_MTP12_GEMV_STAGED_ROWS` value the gate
    /// lists.  Levels 2 and 3 differ only in which shapes they route to the
    /// staged kernel, so covering the kernel at all six shapes covers both.
    /// Levels 4 and 5 (`mtp12_q4_0_gemv_split`) fold each row in a different
    /// order and are deliberately NOT covered here; they are a bench-only lane.
    #[test]
    fn gemv_x4_matches_gemv_bit_for_bit() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        const WEIGHT_OFFSET: usize = 18 * 7;
        const OUTPUT_OFFSET: usize = 4 * 9;
        for (index, &(rows, cols)) in FUSE_GEMV_SHAPES.iter().enumerate() {
            let matrix = Q4TensorRef {
                byte_offset: WEIGHT_OFFSET as u64,
                byte_len: (rows * cols / 32 * 18) as u64,
                rows: rows as u32,
                cols: cols as u32,
            };
            let weights = random_q4_weights(&device, WEIGHT_OFFSET, matrix.byte_len as usize, index as u64);
            let input = f32_buffer(&device, &random_bf16_values(cols, 100 + index as u64, true)).unwrap();
            for round in [false, true] {
                let established = f32_buffer(&device, &vec![f32::NAN; rows]).unwrap();
                let mut variants: Vec<(String, Buffer)> = Vec::new();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &input, &established, matrix, round);
                let x4 = f32_buffer(&device, &vec![f32::NAN; rows + OUTPUT_OFFSET / 4]).unwrap();
                encode_q4_gemv_at_offset(
                    encoder, &pipelines.q4_gemv_x4, &weights, &input, &x4,
                    OUTPUT_OFFSET as u64, matrix, round, true,
                );
                variants.push(("gemv_x4".to_string(), x4));
                for rows_per_group in FUSE_GEMV_STAGED_ROWS {
                    let staged = f32_buffer(&device, &vec![f32::NAN; rows + OUTPUT_OFFSET / 4]).unwrap();
                    encode_q4_gemv_staged(
                        encoder, &pipelines.q4_gemv_staged, &weights, &input, &staged,
                        OUTPUT_OFFSET as u64, matrix, round, rows_per_group,
                    );
                    variants.push((format!("gemv_staged rows_per_group={rows_per_group}"), staged));
                }
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                let expected = read_f32s(&established, rows);
                assert!(expected.iter().all(|v| v.is_finite()), "{rows}x{cols}: fixture not finite");
                for (label, buffer) in &variants {
                    let actual = read_f32s(buffer, rows + OUTPUT_OFFSET / 4);
                    assert!(
                        actual[..OUTPUT_OFFSET / 4].iter().all(|v| v.is_nan()),
                        "{label} {rows}x{cols}: offset prefix clobbered"
                    );
                    assert_bits_equal(
                        &format!("{label} {rows}x{cols} round={round}"),
                        &expected,
                        &actual[OUTPUT_OFFSET / 4..],
                    );
                }
            }
        }
    }

    /// `mtp12_q4_0_gemv_gate_up_gelu` == gemv(gate) + gemv(up) + gelu_mul.
    #[test]
    fn gate_up_gelu_fused_matches_three_dispatch() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        let byte_len = (FFN_HIDDEN * ASSISTANT_HIDDEN / 32 * 18) as u64;
        let gate = Q4TensorRef { byte_offset: 18, byte_len, rows: FFN_HIDDEN as u32, cols: ASSISTANT_HIDDEN as u32 };
        let up = Q4TensorRef { byte_offset: 18 + byte_len, byte_len, rows: FFN_HIDDEN as u32, cols: ASSISTANT_HIDDEN as u32 };
        let weights = random_q4_weights(&device, 18, 2 * byte_len as usize, 0x6a7e);
        for (seed, specials) in [(1u64, true), (2, false), (3, true)] {
            let input = f32_buffer(&device, &random_bf16_values(ASSISTANT_HIDDEN, 900 + seed, specials)).unwrap();
            let gate_out = f32_buffer(&device, &vec![f32::NAN; FFN_HIDDEN]).unwrap();
            let up_out = f32_buffer(&device, &vec![f32::NAN; FFN_HIDDEN]).unwrap();
            let established = f32_buffer(&device, &vec![f32::NAN; FFN_HIDDEN]).unwrap();
            let candidate = f32_buffer(&device, &vec![f32::NAN; FFN_HIDDEN]).unwrap();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &input, &gate_out, gate, true);
            encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &input, &up_out, up, true);
            encode_gelu_mul(encoder, &pipelines.gelu_mul, &gate_out, &up_out, &established, FFN_HIDDEN);
            encode_q4_gemv_gate_up_gelu(
                encoder, &pipelines.q4_gemv_gate_up_gelu, &weights, &input, &candidate, gate, up,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            let expected = read_f32s(&established, FFN_HIDDEN);
            assert!(expected.iter().all(|v| v.is_finite()));
            assert_bits_equal(&format!("gate_up_gelu seed={seed}"), &expected, &read_f32s(&candidate, FFN_HIDDEN));
        }
    }

    /// `mtp12_norm_add_norm` == rms_norm + add (or add_scale) + rms_norm on both
    /// outputs, with and without the layer scale.
    #[test]
    fn norm_add_norm_matches_chain() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        let near_one = |seed: u64| -> Vec<f32> {
            random_bf16_values(ASSISTANT_HIDDEN, seed, false)
                .into_iter().map(|v| round_bf16_cpu(1.0 + v * 0.25)).collect()
        };
        for (case, scale) in [None, Some(0.707_106_8f32), Some(1.0), Some(-1.5), Some(0.0)].into_iter().enumerate() {
            let input = f32_buffer(&device, &random_bf16_values(ASSISTANT_HIDDEN, 10 + case as u64, true)).unwrap();
            let residual = f32_buffer(&device, &random_bf16_values(ASSISTANT_HIDDEN, 20 + case as u64, true)).unwrap();
            let weight = f32_buffer(&device, &near_one(30 + case as u64)).unwrap();
            let next_weight = f32_buffer(&device, &near_one(40 + case as u64)).unwrap();
            let normed = f32_buffer(&device, &vec![f32::NAN; ASSISTANT_HIDDEN]).unwrap();
            let summed = f32_buffer(&device, &vec![f32::NAN; ASSISTANT_HIDDEN]).unwrap();
            let established = f32_buffer(&device, &vec![f32::NAN; ASSISTANT_HIDDEN]).unwrap();
            let out_residual = f32_buffer(&device, &vec![f32::NAN; ASSISTANT_HIDDEN]).unwrap();
            let out_normed = f32_buffer(&device, &vec![f32::NAN; ASSISTANT_HIDDEN]).unwrap();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_rms_norm(encoder, &pipelines.rms_norm, &input, &weight, &normed, ASSISTANT_HIDDEN, 1);
            match scale {
                Some(scale) => encode_add_scale(
                    encoder, &pipelines.add_scale, &residual, &normed, &summed, ASSISTANT_HIDDEN, scale,
                ),
                None => encode_add(encoder, &pipelines.add, &residual, &normed, &summed, ASSISTANT_HIDDEN),
            }
            encode_rms_norm(encoder, &pipelines.rms_norm, &summed, &next_weight, &established, ASSISTANT_HIDDEN, 1);
            encode_norm_add_norm(
                encoder, &pipelines.norm_add_norm, &input, &weight, &residual, &next_weight,
                &out_residual, &out_normed, ASSISTANT_HIDDEN, scale,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            let expected_normed = read_f32s(&established, ASSISTANT_HIDDEN);
            assert!(expected_normed.iter().all(|v| v.is_finite()));
            assert_bits_equal(&format!("norm_add_norm residual scale={scale:?}"), &read_f32s(&summed, ASSISTANT_HIDDEN), &read_f32s(&out_residual, ASSISTANT_HIDDEN));
            assert_bits_equal(&format!("norm_add_norm normed scale={scale:?}"), &expected_normed, &read_f32s(&out_normed, ASSISTANT_HIDDEN));
        }
    }

    /// `mtp12_qnorm_rope` == per-head rms_norm + rope_split at both head dims
    /// and a nonzero rope-table row offset.
    #[test]
    fn qnorm_rope_matches_pair() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        for (case, head_dim) in [LOCAL_HEAD_DIM, FULL_HEAD_DIM].into_iter().enumerate() {
            let half = head_dim / 2;
            let query = f32_buffer(&device, &random_bf16_values(N_HEADS * head_dim, 50 + case as u64, true)).unwrap();
            let weight: Vec<f32> = random_bf16_values(head_dim, 60 + case as u64, false)
                .into_iter().map(|v| round_bf16_cpu(1.0 + v * 0.25)).collect();
            let weight = f32_buffer(&device, &weight).unwrap();
            // Three rope rows of raw (un-rounded) f32 angles; bind the third.
            let mut state = 0xabcd_ef01u64 ^ case as u64;
            let tables: Vec<f32> = (0..3 * half)
                .map(|_| (xorshift(&mut state) >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0)
                .collect();
            let cos = f32_buffer(&device, &tables).unwrap();
            let sin = f32_buffer(&device, &tables.iter().map(|v| -v * 0.5).collect::<Vec<_>>()).unwrap();
            let table_byte_offset = (2 * half * 4) as u64;
            let established = f32_buffer(&device, &vec![f32::NAN; N_HEADS * head_dim]).unwrap();
            let candidate = f32_buffer(&device, &vec![f32::NAN; N_HEADS * head_dim]).unwrap();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            encode_rms_norm(encoder, &pipelines.rms_norm, &query, &weight, &established, head_dim, N_HEADS);
            encode_rope_at_offset(encoder, &pipelines.rope, &established, &cos, &sin, table_byte_offset, head_dim);
            encode_qnorm_rope(
                encoder, &pipelines.qnorm_rope, &query, &weight, &cos, &sin, table_byte_offset,
                &candidate, head_dim,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            let expected = read_f32s(&established, N_HEADS * head_dim);
            assert!(expected.iter().all(|v| v.is_finite()));
            assert_bits_equal(&format!("qnorm_rope head_dim={head_dim}"), &expected, &read_f32s(&candidate, N_HEADS * head_dim));
        }
    }

    /// Every `FUSE_SOFTMAX_CTX` level (1 = fused, 2 = stats + 16-deep context,
    /// 3 = stats + 4-deep context) must reproduce softmax + context_v2 (and the
    /// established pair) bit-for-bit on both head shapes at position counts
    /// around the 32-lane chunk boundaries and at production depth.
    #[test]
    fn mtp12_attention_softmax_context_fused_matches_pair_bit_for_bit() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        const CAPACITY: usize = 1_100;
        for (shape_index, &(kv_heads, head_dim)) in
            [(LOCAL_KV_HEADS, LOCAL_HEAD_DIM), (FULL_KV_HEADS, FULL_HEAD_DIM)].iter().enumerate()
        {
            let (query, keys, values) =
                mtp12_attention_fixture(kv_heads, head_dim, CAPACITY, 40 + shape_index as u64);
            for &position_count in &[1usize, 31, 32, 33, 127, 128, 129, 620, 1_024] {
                for compact_base in [0usize, 3] {
                    let run = |v2: bool, softmax_ctx: u8| {
                        mtp12_attention_run_ex(
                            &device, &pipelines, v2, softmax_ctx, Mtp12AttnGeom::TODAY,
                            &query, &keys, &values, kv_heads, head_dim, CAPACITY,
                            compact_base, position_count,
                        )
                    };
                    let established = run(false, 0);
                    let pair_v2 = run(true, 0);
                    assert!(established.iter().all(|v| v.is_finite()));
                    for level in 1u8..=3 {
                        let fused = run(true, level);
                        let label = format!(
                            "softmax_ctx={level} kv {kv_heads}x{head_dim} base {compact_base} count {position_count}"
                        );
                        assert_bits_equal(&format!("{label} (v2 pair)"), &pair_v2, &fused);
                        assert_bits_equal(&format!("{label} (established)"), &established, &fused);
                    }
                }
            }
        }
    }


    /// Every form the bit gate proves for one head shape: each pipelined context
    /// form against today's scores, each head-paired scores form against today's
    /// context, and the combinations the receipt is most likely to select.
    fn mtp12_attn_gate_geoms(head_dim: usize, kv_heads: usize) -> Vec<Mtp12AttnGeom> {
        let mut geoms = Vec::new();
        let mut push = |context, scores| {
            let geom = Mtp12AttnGeom { context, scores };
            if geom.available(head_dim, kv_heads) && geom != Mtp12AttnGeom::TODAY {
                geoms.push(geom);
            }
        };
        for context in Mtp12ContextForm::PIPELINED {
            push(context, Mtp12ScoresForm::Q1);
        }
        for scores in Mtp12ScoresForm::PAIRED {
            push(Mtp12ContextForm::V2, scores);
        }
        for (context, scores) in [
            (Mtp12ContextForm::D1P1, Mtp12ScoresForm::Q2),
            (Mtp12ContextForm::D1P2, Mtp12ScoresForm::Q2),
            (Mtp12ContextForm::D1P4, Mtp12ScoresForm::Q4),
        ] {
            push(context, scores);
        }
        geoms
    }

    /// Every `CAMELID_GEMMA4_MTP12_ATTN_FORM` variant must reproduce today's
    /// `mtp12_attention_scores_v2` + `mtp12_attention_softmax` +
    /// `mtp12_attention_context_v2` output BIT-FOR-BIT on both 12B head shapes.
    /// `compact_base` 3 rotates every parity bucket (and pushes the
    /// `>= vector_end` tail inside the pipelined interior's exclusion zone);
    /// the position counts straddle the 32-position block boundary, the
    /// four-position unroll and the tree lane's production prefixes.
    #[test]
    fn mtp12_attention_forms_match_v2_bit_for_bit() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B MTP attention form parity");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("Gemma 4 12B MTP pipelines");
        const COUNTS: [usize; 10] = [1, 31, 32, 33, 127, 128, 129, 620, 1_024, 1_500];
        const CAPACITY: usize = 1_503;
        for (shape_index, &(kv_heads, head_dim)) in
            [(LOCAL_KV_HEADS, LOCAL_HEAD_DIM), (FULL_KV_HEADS, FULL_HEAD_DIM)].iter().enumerate()
        {
            let (query, keys, values) =
                mtp12_attention_fixture(kv_heads, head_dim, CAPACITY, 70 + shape_index as u64);
            let geoms = mtp12_attn_gate_geoms(head_dim, kv_heads);
            assert!(
                geoms.len() >= 7,
                "kv {kv_heads} x {head_dim} covers only {} forms",
                geoms.len()
            );
            // The fixture buffers are built once per shape: this gate runs ~500
            // encodings and re-uploading a 12 MiB KV per encoding would dominate it.
            let query = f32_buffer(&device, &query).expect("query buffer");
            let keys = f32_buffer(&device, &keys).expect("keys buffer");
            let values = f32_buffer(&device, &values).expect("values buffer");
            let scores = shared_buffer(&device, N_HEADS * CAPACITY * 4);
            let output = shared_buffer(&device, N_HEADS * head_dim * 4);
            let stats = shared_buffer(&device, N_HEADS * 2 * 4);
            let queue = device.new_command_queue();
            let run = |attn: Mtp12AttnGeom, compact_base: usize, position_count: usize| {
                write_buffer_f32(&output, &vec![f32::NAN; N_HEADS * head_dim]).expect("poison");
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encode_attention_impl_ex(
                    encoder,
                    &pipelines,
                    &query,
                    &keys,
                    &values,
                    0,
                    0,
                    CAPACITY,
                    &scores,
                    &output,
                    kv_heads,
                    head_dim,
                    compact_base + position_count,
                    compact_base,
                    position_count,
                    true,
                    0,
                    Some(&stats),
                    attn,
                );
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let mut out = vec![0.0f32; N_HEADS * head_dim];
                read_buffer_f32(&output, &mut out).expect("read output");
                out
            };
            for &position_count in &COUNTS {
                for compact_base in [0usize, 3] {
                    let today = run(Mtp12AttnGeom::TODAY, compact_base, position_count);
                    assert!(
                        today.iter().all(|value| value.is_finite()),
                        "kv {kv_heads}x{head_dim} base {compact_base} count {position_count}: \
                         today's output is not finite"
                    );
                    for &geom in &geoms {
                        let label = format!(
                            "form {} kv {kv_heads}x{head_dim} base {compact_base} \
                             count {position_count}",
                            geom.label()
                        );
                        assert_bits_equal(
                            &label,
                            &today,
                            &run(geom, compact_base, position_count),
                        );
                    }
                }
            }
        }
    }

    /// `CAMELID_GEMMA4_MTP12_ATTN_FORM` parses strictly: an unlisted form, a
    /// form the head shape cannot host in an explicit `<sliding>,<global>`
    /// spec, and any stray text are refused; a bare form that one shape cannot
    /// host keeps today's kernels for that shape only.
    #[test]
    fn mtp12_attn_form_selector_parses_strictly() {
        let parse = Mtp12AttnSelection::parse;
        assert_eq!(parse("v2").unwrap(), Mtp12AttnSelection::TODAY);
        assert_eq!(parse("v2:q1").unwrap(), Mtp12AttnSelection::TODAY);
        assert_eq!(parse("d1p2").unwrap().label(), "d1p2:q1,d1p2:q1");
        assert_eq!(parse("d1p2:q2").unwrap().label(), "d1p2:q2,d1p2:q2");
        // group 2 at HD256 cannot host four heads per threadgroup; that half
        // keeps today's kernels while HD512 (group 16) takes the form.
        assert_eq!(parse("d1p4:q4").unwrap().label(), "v2:q1,d1p4:q4");
        assert_eq!(parse("d1p2,d1p8").unwrap().label(), "d1p2:q1,d1p8:q1");
        assert_eq!(parse("d1p1:q2,d1p4:q4").unwrap().label(), "d1p1:q2,d1p4:q4");
        // Explicit per-shape specs REFUSE an unhostable form rather than
        // silently dropping it.
        assert!(parse("d1p4,d1p4").is_none());
        assert!(parse("d1p2:q4,d1p2:q4").is_none());
        assert!(parse("d1p16").is_none());
        assert!(parse("d1p2:q3").is_none());
        assert!(parse("").is_none());
        assert!(parse("d1p2,").is_none());
    }

    /// Synthetic compact-head fixture: `rows` tokens with three random clusters
    /// each and a random `top`-of-2048 selection (about the production retained
    /// fraction).  Returns (token_clusters, selected).
    pub(super) fn synthetic_head_selection(device: &Device, rows: usize, top: usize, seed: u64) -> (Buffer, Buffer) {
        let mut state = 0x5eed_5eedu64 ^ seed;
        let clusters: Vec<u16> = (0..rows)
            .flat_map(|_| {
                let a = (xorshift(&mut state) % MTP12_SHORTLIST_CLUSTERS as u64) as u16;
                let b = (xorshift(&mut state) % MTP12_SHORTLIST_CLUSTERS as u64) as u16;
                let c = (xorshift(&mut state) % MTP12_SHORTLIST_CLUSTERS as u64) as u16;
                [a, b, c, 0]
            })
            .collect();
        let cluster_buffer = shared_buffer(device, clusters.len() * 2);
        unsafe {
            std::ptr::copy_nonoverlapping(clusters.as_ptr(), cluster_buffer.contents().cast::<u16>(), clusters.len());
        }
        let mut order: Vec<usize> = (0..MTP12_SHORTLIST_CLUSTERS).collect();
        for i in (1..order.len()).rev() {
            let j = (xorshift(&mut state) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let mut selected = vec![0u32; MTP12_SHORTLIST_CLUSTERS];
        for &index in &order[..top] {
            selected[index] = 1;
        }
        let selected_buffer = shared_buffer(device, selected.len() * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(selected.as_ptr(), selected_buffer.contents().cast::<u32>(), selected.len());
        }
        (cluster_buffer, selected_buffer)
    }

    /// Both compact-head variants must equal `mtp12_q4_0_gemv_compact_local`
    /// bit-for-bit at cols 1024 (one block per lane) and 2048 (a second
    /// in-place block): `_prefetch` (`FUSE_HEAD_PREFETCH=1`, row prefetch) and
    /// `_staged` (`FUSE_HEAD_PREFETCH=2`, row prefetch + register-staged
    /// inputs).  `top` spans an empty, a production-sized and a full shortlist.
    #[test]
    fn head_prefetch_matches_compact_local_bit_for_bit() {
        let Some(device) = Device::system_default() else { return; };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        const ROWS: usize = 8_193;
        for (case, cols) in [1_024usize, 2_048].into_iter().enumerate() {
            let matrix = Q4TensorRef {
                byte_offset: 18,
                byte_len: (ROWS * cols / 32 * 18) as u64,
                rows: ROWS as u32,
                cols: cols as u32,
            };
            let weights = random_q4_weights(&device, 18, matrix.byte_len as usize, 0x4ead + case as u64);
            let input = f32_buffer(&device, &random_bf16_values(cols, 700 + case as u64, true)).unwrap();
            for top in [1usize, 384, 2_048] {
                let (clusters, selected) = synthetic_head_selection(&device, ROWS, top, top as u64);
                let established = f32_buffer(&device, &vec![f32::NAN; ROWS]).unwrap();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                encoder.set_buffer(7, Some(&clusters), 0);
                encoder.set_buffer(8, Some(&selected), 0);
                encode_q4_gemv_shortlist_compact(
                    encoder, &pipelines.q4_gemv_shortlist_compact, &weights, &input, &established, matrix,
                );
                let candidates: Vec<(&str, Buffer)> = [
                    ("head_prefetch=1", &pipelines.q4_gemv_shortlist_compact_prefetch),
                    ("head_prefetch=2", &pipelines.q4_gemv_shortlist_compact_staged),
                ]
                .into_iter()
                .map(|(label, pipeline)| {
                    let output = f32_buffer(&device, &vec![f32::NAN; ROWS]).unwrap();
                    encoder.set_buffer(7, Some(&clusters), 0);
                    encoder.set_buffer(8, Some(&selected), 0);
                    encode_q4_gemv_shortlist_compact(encoder, pipeline, &weights, &input, &output, matrix);
                    (label, output)
                })
                .collect();
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                let expected = read_f32s(&established, ROWS);
                assert!(expected.iter().any(|v| v.is_finite()) || top == 1, "fixture retained nothing");
                assert!(expected.iter().all(|v| !v.is_nan()));
                for (label, output) in &candidates {
                    assert_bits_equal(
                        &format!("{label} cols={cols} top={top}"),
                        &expected,
                        &read_f32s(output, ROWS),
                    );
                }
            }
        }
    }

    /// GB/s per production GEMV shape (30 encodes per command buffer, hardware
    /// timestamps): the established one-row-per-group kernel, x4, the
    /// register-staged kernel at five rows-per-simdgroup values and the
    /// NON-bit-identical split-block experiment at two and four simdgroups per
    /// row; plus the fused gate/up/GELU vs three dispatches and a synthetic
    /// 262,144-row compact head (random 384-of-2048 selection) with the row
    /// prefetch and the register-staged variants.
    #[test]
    #[ignore]
    fn assistant_gemv_bench_shapes() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping assistant GEMV bench");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        const REPS: usize = 30;
        let time = |encode: &dyn Fn(&metal::ComputeCommandEncoderRef)| -> f64 {
            let mut gpu_us = 0.0;
            for pass in 0..2 {
                let reps = if pass == 0 { 1 } else { REPS };
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                for _ in 0..reps {
                    encode(encoder);
                }
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                if pass == 1 {
                    let (us, _) = super::super::command_buffer_gpu_times_us(&cb.to_owned());
                    gpu_us = us as f64 / REPS as f64;
                }
            }
            gpu_us
        };
        for (index, &(rows, cols)) in FUSE_GEMV_SHAPES.iter().enumerate() {
            let matrix = Q4TensorRef {
                byte_offset: 0,
                byte_len: (rows * cols / 32 * 18) as u64,
                rows: rows as u32,
                cols: cols as u32,
            };
            let weights = random_q4_weights(&device, 0, matrix.byte_len as usize, 500 + index as u64);
            let input = f32_buffer(&device, &random_bf16_values(cols, 600 + index as u64, false)).unwrap();
            let output = shared_buffer(&device, rows * 4);
            let report = |label: String, us: f64| {
                eprintln!(
                    "[mtp12-gemv-bench] {rows:>5}x{cols:<5} {label:<26} {us:8.1} us  {:6.1} GB/s",
                    matrix.byte_len as f64 / us / 1.0e3
                );
            };
            for (label, x4) in [("gemv", false), ("gemv_x4", true)] {
                let pipeline = if x4 { &pipelines.q4_gemv_x4 } else { &pipelines.q4_gemv };
                let us = time(&|encoder| {
                    encode_q4_gemv_at_offset(encoder, pipeline, &weights, &input, &output, 0, matrix, true, x4);
                });
                report(label.to_string(), us);
            }
            for rows_per_group in [1u32, 2, 4, 8, 16] {
                let us = time(&|encoder| {
                    encode_q4_gemv_staged(
                        encoder, &pipelines.q4_gemv_staged, &weights, &input, &output, 0, matrix,
                        true, rows_per_group,
                    );
                });
                report(format!("gemv_staged rows={rows_per_group}"), us);
            }
            // NOT bit-identical (different fold order); reported so the
            // simdgroup-parallelism ceiling of the 1024-row shapes is visible.
            for splits in [2u32, 4] {
                let us = time(&|encoder| {
                    encode_q4_gemv_split(
                        encoder, &pipelines.q4_gemv_split, &weights, &input, &output, 0, matrix,
                        true, splits,
                    );
                });
                report(format!("gemv_split x{splits} (INEXACT)"), us);
            }
        }
        {
            let byte_len = (FFN_HIDDEN * ASSISTANT_HIDDEN / 32 * 18) as u64;
            let gate = Q4TensorRef { byte_offset: 0, byte_len, rows: FFN_HIDDEN as u32, cols: ASSISTANT_HIDDEN as u32 };
            let up = Q4TensorRef { byte_offset: byte_len, byte_len, rows: FFN_HIDDEN as u32, cols: ASSISTANT_HIDDEN as u32 };
            let weights = random_q4_weights(&device, 0, 2 * byte_len as usize, 0x9a7e);
            let input = f32_buffer(&device, &random_bf16_values(ASSISTANT_HIDDEN, 0x9a7e, false)).unwrap();
            let gate_out = shared_buffer(&device, FFN_HIDDEN * 4);
            let up_out = shared_buffer(&device, FFN_HIDDEN * 4);
            let gated = shared_buffer(&device, FFN_HIDDEN * 4);
            let three = time(&|encoder| {
                encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &input, &gate_out, gate, true);
                encode_q4_gemv(encoder, &pipelines.q4_gemv, &weights, &input, &up_out, up, true);
                encode_gelu_mul(encoder, &pipelines.gelu_mul, &gate_out, &up_out, &gated, FFN_HIDDEN);
            });
            let three_x4 = time(&|encoder| {
                encode_q4_gemv_at_offset(encoder, &pipelines.q4_gemv_x4, &weights, &input, &gate_out, 0, gate, true, true);
                encode_q4_gemv_at_offset(encoder, &pipelines.q4_gemv_x4, &weights, &input, &up_out, 0, up, true, true);
                encode_gelu_mul(encoder, &pipelines.gelu_mul, &gate_out, &up_out, &gated, FFN_HIDDEN);
            });
            let fused = time(&|encoder| {
                encode_q4_gemv_gate_up_gelu(encoder, &pipelines.q4_gemv_gate_up_gelu, &weights, &input, &gated, gate, up);
            });
            let bytes = 2.0 * byte_len as f64;
            eprintln!("[mtp12-gemv-bench] gate+up+gelu three-dispatch    {three:8.1} us  {:6.1} GB/s", bytes / three / 1.0e3);
            eprintln!("[mtp12-gemv-bench] gate+up+gelu three-dispatch x4 {three_x4:8.1} us  {:6.1} GB/s", bytes / three_x4 / 1.0e3);
            eprintln!("[mtp12-gemv-bench] gate+up+gelu fused             {fused:8.1} us  {:6.1} GB/s", bytes / fused / 1.0e3);
        }
        {
            let matrix = Q4TensorRef {
                byte_offset: 0,
                byte_len: (VOCAB * ASSISTANT_HIDDEN / 32 * 18) as u64,
                rows: VOCAB as u32,
                cols: ASSISTANT_HIDDEN as u32,
            };
            let weights = random_q4_weights(&device, 0, matrix.byte_len as usize, 0xead);
            let input = f32_buffer(&device, &random_bf16_values(ASSISTANT_HIDDEN, 0xead, false)).unwrap();
            let logits = shared_buffer(&device, VOCAB * 4);
            let (clusters, selected) = synthetic_head_selection(&device, VOCAB, 384, 0xead);
            let cluster_ids = unsafe { std::slice::from_raw_parts(clusters.contents().cast::<u16>(), VOCAB * 4) };
            let selected_ids = unsafe { std::slice::from_raw_parts(selected.contents().cast::<u32>(), MTP12_SHORTLIST_CLUSTERS) };
            let retained = cluster_ids.chunks_exact(4)
                .filter(|c| c[..3].iter().any(|&id| selected_ids[id as usize] != 0))
                .count();
            for (label, pipeline) in [
                ("compact_local", &pipelines.q4_gemv_shortlist_compact),
                ("compact_local_prefetch", &pipelines.q4_gemv_shortlist_compact_prefetch),
                ("compact_local_staged", &pipelines.q4_gemv_shortlist_compact_staged),
            ] {
                let us = time(&|encoder| {
                    encoder.set_buffer(7, Some(&clusters), 0);
                    encoder.set_buffer(8, Some(&selected), 0);
                    encode_q4_gemv_shortlist_compact(encoder, pipeline, &weights, &input, &logits, matrix);
                });
                eprintln!(
                    "[mtp12-gemv-bench] head {label:<22} {us:8.1} us  retained {retained} rows ({:.1}%)  {:6.1} GB/s over retained bytes",
                    retained as f64 * 100.0 / VOCAB as f64,
                    (retained * ASSISTANT_HIDDEN / 32 * 18) as f64 / us / 1.0e3
                );
            }
        }
    }


    /// One attention dispatch, bound exactly as `encode_attention_impl_ex`
    /// binds it.  The bench needs the phases separately, so the bindings are
    /// mirrored here; any binding change must be mirrored back.
    #[derive(Clone, Copy, Debug)]
    enum AttentionPhase {
        Scores { v2: bool },
        Softmax,
        SoftmaxStats,
        /// `mtp12_attention_context_v2`, `_softmax_context_v2`,
        /// `_context_v2_stats16` or `_context_v2_stats4`.
        Context { softmax_ctx: u8 },
        /// A `CAMELID_GEMMA4_MTP12_ATTN_FORM` pipelined context kernel.
        ContextForm(Mtp12ContextForm),
        /// A `CAMELID_GEMMA4_MTP12_ATTN_FORM` head-paired scores kernel.
        ScoresForm(Mtp12ScoresForm),
    }

    #[allow(clippy::too_many_arguments)]
    fn bench_encode_attention_phase(
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &Mtp12Pipelines,
        phase: AttentionPhase,
        query: &Buffer,
        key: &Buffer,
        value: &Buffer,
        scores: &Buffer,
        output: &Buffer,
        stats: &Buffer,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        position_count: usize,
    ) {
        let n_heads = N_HEADS as u32;
        let head_dim_u32 = head_dim as u32;
        let position_count_u32 = position_count as u32;
        let group = (N_HEADS / kv_heads) as u32;
        let position_stride = head_dim_u32;
        let kv_head_stride = (capacity * head_dim) as u32;
        let kv_base_offset = 0u32;
        let compact_base = 0u32;
        let logical_len = position_count_u32;
        let tg32 = MTLSize { width: 32, height: 1, depth: 1 };
        match phase {
            AttentionPhase::Scores { v2 } => {
                encoder.set_compute_pipeline_state(if v2 {
                    &pipelines.attention_scores_v2
                } else {
                    &pipelines.attention_scores
                });
                encoder.set_buffer(0, Some(query), 0);
                encoder.set_buffer(1, Some(key), 0);
                encoder.set_buffer(2, Some(scores), 0);
                encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
                encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
                encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
                encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: N_HEADS as u64,
                        height: if v2 { position_count.div_ceil(32) as u64 } else { 1 },
                        depth: 1,
                    },
                    tg32,
                );
            }
            AttentionPhase::ScoresForm(form) => {
                let pairs = form.pairs().expect("paired scores form");
                encoder.set_compute_pipeline_state(
                    pipelines.attention_forms.scores(form).expect("scores pipeline"),
                );
                encoder.set_buffer(0, Some(query), 0);
                encoder.set_buffer(1, Some(key), 0);
                encoder.set_buffer(2, Some(scores), 0);
                encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
                encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
                encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
                encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: (N_HEADS / pairs) as u64,
                        height: position_count.div_ceil(32) as u64,
                        depth: 1,
                    },
                    tg32,
                );
            }
            AttentionPhase::ContextForm(form) => {
                let (chunks, blocks) = form.grid(head_dim).expect("pipelined context form");
                encoder.set_compute_pipeline_state(
                    pipelines.attention_forms.context(form).expect("context pipeline"),
                );
                encoder.set_buffer(0, Some(value), 0);
                encoder.set_buffer(1, Some(scores), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
                encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
                encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
                encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
                encoder.set_bytes(10, 4, &compact_base as *const u32 as *const c_void);
                encoder.set_bytes(11, 4, &logical_len as *const u32 as *const c_void);
                encoder.dispatch_thread_groups(
                    MTLSize { width: chunks as u64, height: blocks as u64, depth: 1 },
                    tg32,
                );
            }
            AttentionPhase::Softmax => {
                encoder.set_compute_pipeline_state(&pipelines.attention_softmax);
                encoder.set_buffer(0, Some(scores), 0);
                encoder.set_bytes(1, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(2, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.dispatch_thread_groups(
                    MTLSize { width: N_HEADS as u64, height: 1, depth: 1 },
                    tg32,
                );
            }
            AttentionPhase::SoftmaxStats => {
                encoder.set_compute_pipeline_state(&pipelines.attention_softmax_stats);
                encoder.set_buffer(0, Some(scores), 0);
                encoder.set_bytes(1, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(2, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.set_buffer(3, Some(stats), 0);
                encoder.dispatch_thread_groups(
                    MTLSize { width: N_HEADS as u64, height: 1, depth: 1 },
                    tg32,
                );
            }
            AttentionPhase::Context { softmax_ctx } => {
                encoder.set_compute_pipeline_state(match softmax_ctx {
                    1 => &pipelines.attention_softmax_context_v2,
                    2 => &pipelines.attention_context_v2_stats16,
                    3 => &pipelines.attention_context_v2_stats4,
                    _ => &pipelines.attention_context_v2,
                });
                encoder.set_buffer(0, Some(value), 0);
                encoder.set_buffer(1, Some(scores), 0);
                encoder.set_buffer(2, Some(output), 0);
                encoder.set_bytes(3, 4, &n_heads as *const u32 as *const c_void);
                encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
                encoder.set_bytes(5, 4, &position_count_u32 as *const u32 as *const c_void);
                encoder.set_bytes(6, 4, &group as *const u32 as *const c_void);
                encoder.set_bytes(7, 4, &position_stride as *const u32 as *const c_void);
                encoder.set_bytes(8, 4, &kv_head_stride as *const u32 as *const c_void);
                encoder.set_bytes(9, 4, &kv_base_offset as *const u32 as *const c_void);
                encoder.set_bytes(10, 4, &compact_base as *const u32 as *const c_void);
                encoder.set_bytes(11, 4, &logical_len as *const u32 as *const c_void);
                if softmax_ctx >= 2 {
                    encoder.set_buffer(12, Some(stats), 0);
                }
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: N_HEADS as u64,
                        height: (head_dim / 128) as u64,
                        depth: 1,
                    },
                    tg32,
                );
            }
        }
    }

    /// Per-kernel GPU cost of the assistant's attention phases on both 12B head
    /// shapes at the tree lane's production prefixes: the established
    /// softmax + `context_v2` pair against the fused `softmax_context_v2` and
    /// against the stats pre-pass + stats-fed context at both load depths, then
    /// every `CAMELID_GEMMA4_MTP12_ATTN_FORM` candidate the shape can host -
    /// each pipelined `d<dims>p<pairs>` context kernel against `context_v2`
    /// (with its threadgroup grid, so occupancy can be read off) and each
    /// head-paired `q<pairs>` scores kernel against `scores_v2`.
    /// Prefix lengths come from `CAMELID_MTP12_BENCH_PREFIX` (default
    /// "620,1500"); every prefix is measured with `compact_base` 0, so
    /// `position_count` is the prefix itself.
    ///
    /// Invoke (release, on the rig):
    ///   cargo test --release --lib assistant_attention_kernel_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn assistant_attention_kernel_bench() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping assistant attention bench");
            return;
        };
        let pipelines = Mtp12Pipelines::new(&device).expect("pipelines");
        let queue = device.new_command_queue();
        const REPS: usize = 20;
        const BUFFERS: usize = 5;
        let median = |encode: &dyn Fn(&metal::ComputeCommandEncoderRef)| -> f64 {
            let mut samples = Vec::with_capacity(BUFFERS);
            for pass in 0..=BUFFERS {
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();
                for _ in 0..REPS {
                    encode(encoder);
                }
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                if pass > 0 {
                    let (us, _) = super::super::command_buffer_gpu_times_us(&cb.to_owned());
                    samples.push(us as f64 / REPS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let prefixes: Vec<usize> = std::env::var("CAMELID_MTP12_BENCH_PREFIX")
            .unwrap_or_else(|_| "620,1500".to_string())
            .split(',')
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| text.parse::<usize>().expect("prefix must be an integer"))
            .collect();
        eprintln!(
            "[mtp12-attn-bench] {REPS} encodes/buffer, median of {BUFFERS} buffers, compact_base 0"
        );
        for (index, &(kv_heads, head_dim)) in
            [(LOCAL_KV_HEADS, LOCAL_HEAD_DIM), (FULL_KV_HEADS, FULL_HEAD_DIM)].iter().enumerate()
        {
            for &position_count in &prefixes {
                let capacity = position_count;
                let (query, keys, values) =
                    mtp12_attention_fixture(kv_heads, head_dim, capacity, 900 + index as u64);
                let query = f32_buffer(&device, &query).unwrap();
                let keys = f32_buffer(&device, &keys).unwrap();
                let values = f32_buffer(&device, &values).unwrap();
                let scores = shared_buffer(&device, N_HEADS * position_count * 4);
                let output = shared_buffer(&device, N_HEADS * head_dim * 4);
                let stats = shared_buffer(&device, N_HEADS * 2 * 4);
                let run = |phases: &[AttentionPhase]| -> f64 {
                    median(&|encoder| {
                        for &phase in phases {
                            bench_encode_attention_phase(
                                encoder, &pipelines, phase, &query, &keys, &values, &scores,
                                &output, &stats, kv_heads, head_dim, capacity, position_count,
                            );
                        }
                    })
                };
                let group = N_HEADS / kv_heads;
                eprintln!(
                    "[mtp12-attn-bench] ---- HD{head_dim} ({kv_heads} KV heads, group {group}), \
                     prefix {position_count} ----"
                );
                let scores_v2 = run(&[AttentionPhase::Scores { v2: true }]);
                let softmax = run(&[AttentionPhase::Softmax]);
                let stats_pass = run(&[AttentionPhase::SoftmaxStats]);
                let context_v2 = run(&[AttentionPhase::Context { softmax_ctx: 0 }]);
                let fused = run(&[AttentionPhase::Context { softmax_ctx: 1 }]);
                let ctx16 = run(&[AttentionPhase::Context { softmax_ctx: 2 }]);
                let ctx4 = run(&[AttentionPhase::Context { softmax_ctx: 3 }]);
                for (label, us) in [
                    ("scores_v2", scores_v2),
                    ("softmax", softmax),
                    ("softmax_stats", stats_pass),
                    ("context_v2", context_v2),
                    ("softmax_context_v2 (fused)", fused),
                    ("context_v2_stats16", ctx16),
                    ("context_v2_stats4", ctx4),
                ] {
                    eprintln!("[mtp12-attn-bench]   kernel {label:<28} {us:8.1} us");
                }
                let pair = softmax + context_v2;
                for (label, us) in [
                    ("softmax + context_v2 (level 0)", pair),
                    ("softmax_context_v2 (level 1)", fused),
                    ("stats + context_v2_stats16 (level 2)", stats_pass + ctx16),
                    ("stats + context_v2_stats4 (level 3)", stats_pass + ctx4),
                ] {
                    eprintln!(
                        "[mtp12-attn-bench]   phase  {label:<38} {us:8.1} us  {:+6.1}% vs pair",
                        (us - pair) * 100.0 / pair
                    );
                }
                eprintln!(
                    "[mtp12-attn-bench]   whole  {:<38} {:8.1} us",
                    "scores_v2 + best phase",
                    scores_v2 + [pair, fused, stats_pass + ctx16, stats_pass + ctx4]
                        .into_iter()
                        .fold(f64::INFINITY, f64::min)
                );

                // CAMELID_GEMMA4_MTP12_ATTN_FORM candidates (bit-identical to the
                // scores_v2 / context_v2 kernels above; see
                // mtp12_attention_forms_match_v2_bit_for_bit).
                let mut best_scores = (Mtp12ScoresForm::Q1.label(), scores_v2);
                for form in Mtp12ScoresForm::PAIRED {
                    if !form.available(head_dim, kv_heads) {
                        continue;
                    }
                    let us = run(&[AttentionPhase::ScoresForm(form)]);
                    eprintln!(
                        "[mtp12-attn-bench]   form   scores {:<31} {us:8.1} us                           {:+6.1}% vs scores_v2",
                        form.label(),
                        (us - scores_v2) * 100.0 / scores_v2
                    );
                    if us < best_scores.1 {
                        best_scores = (form.label(), us);
                    }
                }
                let mut best_context = (Mtp12ContextForm::V2.label(), context_v2);
                for form in Mtp12ContextForm::PIPELINED {
                    if !form.available(head_dim, kv_heads) {
                        continue;
                    }
                    let (chunks, blocks) = form.grid(head_dim).expect("grid");
                    let us = run(&[AttentionPhase::ContextForm(form)]);
                    eprintln!(
                        "[mtp12-attn-bench]   form   context {:<6} ({chunks:>2}x{blocks:>2} tg)                          {:<14} {us:8.1} us  {:+6.1}% vs context_v2",
                        form.label(),
                        "",
                        (us - context_v2) * 100.0 / context_v2
                    );
                    if us < best_context.1 {
                        best_context = (form.label(), us);
                    }
                }
                eprintln!(
                    "[mtp12-attn-bench]   whole  {:<38} {:8.1} us  {:+6.1}% vs today",
                    format!(
                        "{} + softmax + {}",
                        best_scores.0, best_context.0
                    ),
                    best_scores.1 + softmax + best_context.1,
                    (best_scores.1 + softmax + best_context.1 - (scores_v2 + pair)) * 100.0
                        / (scores_v2 + pair)
                );
            }
        }
    }

    #[test]
    fn production_mtp12_shader_compiles() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B MTP shader compile");
            return;
        };
        Mtp12Pipelines::new(&device).expect("Gemma 4 12B MTP pipelines");
    }
}
