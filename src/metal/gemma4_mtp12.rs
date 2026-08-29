//! Exact, isolated Metal foundation for the official Gemma 4 12B QAT assistant.
//!
//! This deliberately does not enter the Gemma target/verifier runtime.  It
//! admits one byte-identical assistant artifact, packs all 23 BF16 matrices to
//! canonical GGML Q4_0, releases the 846 MB source mapping, and exposes a K=1
//! CPU-input parity/profiling call.  The explicit copy boundary is intentional:
//! the future verifier integration can replace it with scoped target-buffer
//! views without weakening artifact or assistant-graph admission.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    path::{Path, PathBuf},
    time::Instant,
};

use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{wire_mmap::GgufWireMmap, BackendError, Result};

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
const FULL_Q4_MATRIX_BYTES: u64 = 237_846_528;

/// The official 12B target has 48 layers in a 5-local/1-global schedule.
/// Its assistant borrows the final local/global pair, not the 26B pair 28/29.
pub const GEMMA4_12B_MTP_SLIDING_HOST_LAYER: usize = 46;
pub const GEMMA4_12B_MTP_FULL_HOST_LAYER: usize = 47;

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
"#;

struct Mtp12Pipelines {
    q4_gemv: ComputePipelineState,
    rms_norm: ComputePipelineState,
    rope: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_softmax: ComputePipelineState,
    attention_context: ComputePipelineState,
    add: ComputePipelineState,
    add_scale: ComputePipelineState,
    gelu_mul: ComputePipelineState,
    argmax: ComputePipelineState,
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
            q4_gemv: pipeline("mtp12_q4_0_gemv")?,
            rms_norm: pipeline("mtp12_rms_norm")?,
            rope: pipeline("mtp12_rope_split")?,
            attention_scores: pipeline("mtp12_attention_scores")?,
            attention_softmax: pipeline("mtp12_attention_softmax")?,
            attention_context: pipeline("mtp12_attention_context")?,
            add: pipeline("mtp12_add_bf16")?,
            add_scale: pipeline("mtp12_add_scale_bf16")?,
            gelu_mul: pipeline("mtp12_gelu_mul")?,
            argmax: pipeline("mtp12_argmax")?,
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
    logits: Buffer,
    output_token: Buffer,
    local_cos: Buffer,
    local_sin: Buffer,
    full_cos: Buffer,
    full_sin: Buffer,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gemma4Mtp12ResidentLedger {
    pub source_file_bytes: u64,
    pub source_sha256_verified: bool,
    pub source_mapping_retained: bool,
    pub packed_q4_matrix_bytes: u64,
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

/// Fully packed 12B assistant.  No target model, pager, MoE state, or verifier
/// is retained here; all 23 matrices live in one bounded Q4_0 Metal buffer.
pub struct Gemma4Mtp12AssistantMetal {
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
    let byte_len = std::mem::size_of_val(values);
    if byte_len > buffer.length() as usize {
        return Err(invalid(format!(
            "buffer write of {byte_len} bytes exceeds {}",
            buffer.length()
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
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
            logits: f32s(VOCAB),
            output_token: shared_buffer(device, std::mem::size_of::<u32>()),
            local_cos: f32s(LOCAL_HEAD_DIM / 2),
            local_sin: f32s(LOCAL_HEAD_DIM / 2),
            full_cos: f32s(FULL_HEAD_DIM / 2),
            full_sin: f32s(FULL_HEAD_DIM / 2),
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
            &self.logits,
            &self.output_token,
            &self.local_cos,
            &self.local_sin,
            &self.full_cos,
            &self.full_sin,
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
        // Every runtime matrix and norm now owns independent Metal storage.
        // Releasing the source is part of the 16 GB admission contract.
        drop(mapping);

        let pipeline_started = Instant::now();
        let pipelines = Mtp12Pipelines::new(device)?;
        let pipeline_compile_us = pipeline_started.elapsed().as_micros();
        let scratch = Mtp12Scratch::new(device);
        let resident_ledger = Gemma4Mtp12ResidentLedger {
            source_file_bytes: EXPECTED_FILE_BYTES,
            source_sha256_verified: true,
            source_mapping_retained: false,
            packed_q4_matrix_bytes: packed_q4.length(),
            decoded_norm_bytes,
            fixed_scratch_bytes: scratch.byte_len(),
            quantize_us,
            hash_us,
            pipeline_compile_us,
            load_wall_us: load_started.elapsed().as_micros(),
        };
        let layers: [LayerWeights; N_LAYERS] = layers
            .try_into()
            .map_err(|_| invalid("internal resident layer table is not four layers"))?;
        Ok(Self {
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
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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
        encode_argmax(
            encoder,
            &self.pipelines.argmax,
            &self.scratch.logits,
            &self.scratch.output_token,
            VOCAB,
        );
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
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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
        encode_rope(
            encoder,
            &self.pipelines.rope,
            &self.scratch.query_normed,
            cos,
            sin,
            head_dim,
        );
        encode_attention(
            encoder,
            &self.pipelines,
            &self.scratch.query_normed,
            key,
            value,
            attention_scores,
            &self.scratch.context,
            kv_heads,
            head_dim,
            logical_len,
            compact_base,
            position_count,
        );
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
            &self.scratch.normed,
            &self.scratch.gate,
            matrices.gate,
            true,
        );
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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
        encode_q4_gemv(
            encoder,
            &self.pipelines.q4_gemv,
            &self.packed_q4,
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

fn encode_q4_gemv(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    weights: &Buffer,
    input: &Buffer,
    output: &Buffer,
    matrix: Q4TensorRef,
    round_output_bf16: bool,
) {
    let round = u32::from(round_output_bf16);
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
            width: matrix.rows as u64,
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

fn encode_rope(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    data: &Buffer,
    cos: &Buffer,
    sin: &Buffer,
    head_dim: usize,
) {
    let heads = N_HEADS as u32;
    let head_dim_u32 = head_dim as u32;
    let count = N_HEADS * head_dim / 2;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(data), 0);
    encoder.set_buffer(1, Some(cos), 0);
    encoder.set_buffer(2, Some(sin), 0);
    encoder.set_bytes(3, 4, &heads as *const u32 as *const c_void);
    encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const c_void);
    dispatch_1d(encoder, pipeline, count);
}

#[allow(clippy::too_many_arguments)]
fn encode_attention(
    encoder: &metal::ComputeCommandEncoderRef,
    pipelines: &Mtp12Pipelines,
    query: &Buffer,
    key: &Buffer,
    value: &Buffer,
    scores: &Buffer,
    output: &Buffer,
    kv_heads: usize,
    head_dim: usize,
    logical_len: usize,
    compact_base: usize,
    position_count: usize,
) {
    debug_assert!(position_count > 0);
    debug_assert_eq!(compact_base + position_count, logical_len);
    let n_heads = N_HEADS as u32;
    let head_dim = head_dim as u32;
    let position_count_u32 = position_count as u32;
    let group = (N_HEADS / kv_heads) as u32;
    let position_stride = head_dim;
    let kv_head_stride = (logical_len * head_dim as usize) as u32;
    let kv_base_offset = (compact_base * head_dim as usize) as u32;
    let compact_base_u32 = compact_base as u32;
    let logical_len_u32 = logical_len as u32;

    encoder.set_compute_pipeline_state(&pipelines.attention_scores);
    encoder.set_buffer(0, Some(query), 0);
    encoder.set_buffer(1, Some(key), 0);
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
            width: N_HEADS as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

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
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&pipelines.attention_context);
    encoder.set_buffer(0, Some(value), 0);
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
    encoder.dispatch_thread_groups(
        MTLSize {
            width: N_HEADS as u64,
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

fn encode_argmax(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    logits: &Buffer,
    output: &Buffer,
    count: usize,
) {
    let count_u32 = count as u32;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(logits), 0);
    encoder.set_buffer(1, Some(output), 0);
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn official_12b_geometry_is_not_the_26b_geometry() {
        assert_eq!(TARGET_HIDDEN, 3_840);
        assert_eq!(FULL_KV_HEADS, 1);
        assert_eq!(GEMMA4_12B_MTP_SLIDING_HOST_LAYER, 46);
        assert_eq!(GEMMA4_12B_MTP_FULL_HOST_LAYER, 47);
        assert_eq!(EXPECTED_FILE_BYTES, 845_719_296);
        assert_eq!(
            GEMMA4_12B_MTP_ASSISTANT_SHA256,
            "67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6"
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
    fn production_mtp12_shader_compiles() {
        let Some(device) = Device::system_default() else {
            eprintln!("Metal is unavailable; skipping Gemma 4 12B MTP shader compile");
            return;
        };
        Mtp12Pipelines::new(&device).expect("Gemma 4 12B MTP pipelines");
    }
}
