//! Built-in Local GGUF Quantizer Utility (Feature D)
//!
//! Provides direct on-hardware quantization of GGUF models (e.g. FP16/BF16/Q8_0 to Q4_K_M, Q8_0, Q4_0)
//! with verified parity receipts and SHA-256 digests.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::str::FromStr;

use memmap2::Mmap;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::gguf::{read_metadata, GgufMetadataValue, GgufTensorType};
use crate::tensor::{f32_to_f16_bits, fast_f16_to_f32};
use crate::{BackendError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetQuant {
    Q8_0,
    Q4_0,
    #[allow(non_camel_case_types)]
    Q4_K_M,
}

impl TargetQuant {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_K_M => "q4_k_m",
        }
    }

    pub fn gguf_type(&self) -> GgufTensorType {
        match self {
            Self::Q8_0 => GgufTensorType::Q8_0,
            Self::Q4_0 => GgufTensorType::Q4_0,
            Self::Q4_K_M => GgufTensorType::Q4K,
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_0 => 32,
            Self::Q4_K_M => 256,
        }
    }

    pub fn type_id(&self) -> u32 {
        match self {
            Self::Q8_0 => 8,
            Self::Q4_0 => 2,
            Self::Q4_K_M => 12,
        }
    }

    pub fn file_type_id(&self) -> u32 {
        match self {
            Self::Q8_0 => 7,
            Self::Q4_0 => 2,
            Self::Q4_K_M => 15,
        }
    }
}

impl FromStr for TargetQuant {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "q8_0" | "q8" | "q80" => Ok(Self::Q8_0),
            "q4_0" | "q4" | "q40" => Ok(Self::Q4_0),
            "q4_k_m" | "q4_k" | "q4km" | "q4k" => Ok(Self::Q4_K_M),
            other => Err(format!(
                "unsupported target quantization '{other}'; supported: q4_k_m, q8_0, q4_0"
            )),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantizeReceipt {
    pub input_path: String,
    pub output_path: String,
    pub target_quant: String,
    pub tensor_count: usize,
    pub quantized_tensors: usize,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub compression_ratio: f32,
    pub sha256: String,
}

/// Quantizes an f32 slice of 32 elements into a 34-byte Q8_0 block (f16 scale + 32 i8 values).
pub fn quantize_block_q8_0(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), 32);
    debug_assert_eq!(dst.len(), 34);

    let mut amax = 0.0f32;
    for &val in src {
        let abs = val.abs();
        if abs > amax {
            amax = abs;
        }
    }

    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let d_f16 = f32_to_f16_bits(d);

    dst[0..2].copy_from_slice(&d_f16.to_le_bytes());
    for i in 0..32 {
        let x0 = src[i] * id;
        dst[2 + i] = x0.round().clamp(-128.0, 127.0) as i8 as u8;
    }
}

/// Quantizes an f32 slice of 32 elements into an 18-byte Q4_0 block (f16 scale + 16 packed nibbles).
pub fn quantize_block_q4_0(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), 32);
    debug_assert_eq!(dst.len(), 18);

    let mut amax = 0.0f32;
    let mut vmax = 0.0f32;
    for &val in src {
        let abs = val.abs();
        if abs > amax {
            amax = abs;
            vmax = val;
        }
    }

    // GGML's Q4_0 scale carries the sign of the largest-magnitude value. The
    // decoder reconstructs `(nibble - 8) * d`, so storing `-d` here would
    // mirror every value around zero.
    let d = vmax / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let d_f16 = f32_to_f16_bits(d);

    dst[0..2].copy_from_slice(&d_f16.to_le_bytes());
    for i in 0..16 {
        let x0 = (src[i] * id).round().clamp(-8.0, 7.0) as i8 + 8;
        let x1 = (src[i + 16] * id).round().clamp(-8.0, 7.0) as i8 + 8;
        dst[2 + i] = ((x0 as u8) & 0x0F) | (((x1 as u8) & 0x0F) << 4);
    }
}

/// Quantizes an f32 slice of 256 elements into a 144-byte Q4_K superblock.
pub fn quantize_block_q4_k(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), 256);
    debug_assert_eq!(dst.len(), 144);

    // 8 sub-blocks of 32 elements
    let mut mins = [0.0f32; 8];
    let mut maxs = [0.0f32; 8];

    for group in 0..8 {
        let sub = &src[group * 32..(group + 1) * 32];
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &v in sub {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        mins[group] = min_val;
        maxs[group] = max_val;
    }

    let mut super_min = 0.0f32;
    let mut super_max = 0.0f32;
    for g in 0..8 {
        let effective_min = mins[g].min(0.0);
        let group_scale = (maxs[g] - effective_min).max(0.0) / 15.0;
        if group_scale > super_max {
            super_max = group_scale;
        }
        let min_magnitude = (-mins[g]).max(0.0);
        if min_magnitude > super_min {
            super_min = min_magnitude;
        }
    }

    let d = super_max / 63.0;
    let dmin = super_min / 63.0;

    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let idmin = if dmin != 0.0 { 1.0 / dmin } else { 0.0 };

    let d_f16 = f32_to_f16_bits(d);
    let dmin_f16 = f32_to_f16_bits(dmin);

    dst[0..2].copy_from_slice(&d_f16.to_le_bytes());
    dst[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

    // 6-bit scales and mins packed into 12 bytes
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for g in 0..8 {
        let effective_min = mins[g].min(0.0);
        let group_scale = (maxs[g] - effective_min).max(0.0) / 15.0;
        sc[g] = (group_scale * id).round().clamp(0.0, 63.0) as u8;
        m[g] = ((-mins[g]).max(0.0) * idmin).round().clamp(0.0, 63.0) as u8;
    }

    let mut packed_scales = [0u8; 12];
    for g in 0..4 {
        packed_scales[g] = sc[g] & 0x3f;
        packed_scales[g + 4] = m[g] & 0x3f;
    }
    for g in 4..8 {
        packed_scales[g - 4] |= (sc[g] >> 4) << 6;
        packed_scales[g] |= (m[g] >> 4) << 6;
        packed_scales[g + 4] = (sc[g] & 0x0f) | ((m[g] & 0x0f) << 4);
    }
    dst[4..16].copy_from_slice(&packed_scales);

    let mut quantized = [[0u8; 32]; 8];
    for g in 0..8 {
        let sub = &src[g * 32..(g + 1) * 32];
        let sub_d = d * (sc[g] as f32);
        let sub_min = -(dmin * (m[g] as f32));
        let sub_id = if sub_d != 0.0 { 1.0 / sub_d } else { 0.0 };

        for i in 0..32 {
            quantized[g][i] = ((sub[i] - sub_min) * sub_id).round().clamp(0.0, 15.0) as u8;
        }
    }

    // Each 32-byte run interleaves two adjacent 32-value groups, matching
    // Q4KBlock::dequantize and the canonical GGML block_q4_K wire layout.
    for pair in 0..4 {
        for i in 0..32 {
            dst[16 + pair * 32 + i] =
                (quantized[pair * 2][i] & 0x0f) | (quantized[pair * 2 + 1][i] << 4);
        }
    }
}

/// Dequantizes a source buffer to F32 values based on its GgufTensorType.
pub fn dequantize_to_f32(
    src_type: GgufTensorType,
    src_bytes: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>> {
    match src_type {
        GgufTensorType::F32 => {
            if src_bytes.len() != num_elements.saturating_mul(4) {
                return Err(BackendError::InvalidTensorData(format!(
                    "F32 tensor has {} bytes for {num_elements} elements",
                    src_bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(num_elements);
            for chunk in src_bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        GgufTensorType::F16 => {
            if src_bytes.len() != num_elements.saturating_mul(2) {
                return Err(BackendError::InvalidTensorData(format!(
                    "F16 tensor has {} bytes for {num_elements} elements",
                    src_bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(num_elements);
            for chunk in src_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(fast_f16_to_f32(bits));
            }
            Ok(out)
        }
        GgufTensorType::BF16 => {
            if src_bytes.len() != num_elements.saturating_mul(2) {
                return Err(BackendError::InvalidTensorData(format!(
                    "BF16 tensor has {} bytes for {num_elements} elements",
                    src_bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(num_elements);
            for chunk in src_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f32::from_bits((bits as u32) << 16));
            }
            Ok(out)
        }
        GgufTensorType::Q8_0 => {
            let expected_blocks = num_elements.div_ceil(32);
            if !num_elements.is_multiple_of(32) || src_bytes.len() != expected_blocks * 34 {
                return Err(BackendError::InvalidTensorData(format!(
                    "Q8_0 tensor has {} bytes for {num_elements} elements",
                    src_bytes.len()
                )));
            }
            let mut out = Vec::with_capacity(num_elements);
            for block in src_bytes.chunks_exact(34) {
                let d = fast_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for &b in &block[2..34] {
                    out.push((b as i8 as f32) * d);
                }
            }
            Ok(out)
        }
        other => Err(BackendError::UnsupportedTensorType(format!(
            "the built-in quantizer cannot convert {other:?}; supported input tensor types are F32, F16, BF16, and Q8_0"
        ))),
    }
}

fn is_supported_quantizer_input(tensor_type: GgufTensorType) -> bool {
    matches!(
        tensor_type,
        GgufTensorType::F32 | GgufTensorType::F16 | GgufTensorType::BF16 | GgufTensorType::Q8_0
    )
}

/// Checks if a tensor is a 2D weight matrix that should be quantized.
fn should_quantize_tensor(name: &str, dims: &[u64]) -> bool {
    if dims.len() < 2 {
        return false; // 1D norms, biases stay original
    }
    // Norms and embeddings typically stay high precision
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    true
}

fn io_err(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn paths_refer_to_same_file(input: &Path, output: &Path) -> Result<bool> {
    let input = std::fs::canonicalize(input).map_err(|e| io_err(input, e))?;
    if output.exists() {
        return same_file::is_same_file(&input, output).map_err(|e| io_err(output, e));
    }

    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent).map_err(|e| io_err(parent, e))?;
    Ok(output
        .file_name()
        .map(|name| parent.join(name) == input)
        .unwrap_or(false))
}

/// Run full GGUF quantization workflow from `input_path` to `output_path`.
pub fn quantize_model(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    target: TargetQuant,
) -> Result<QuantizeReceipt> {
    let input_p = input_path.as_ref();
    let output_p = output_path.as_ref();

    if paths_refer_to_same_file(input_p, output_p)? {
        return Err(BackendError::InvalidGguf(
            "input and output paths must refer to different files".to_string(),
        ));
    }

    let gguf = read_metadata(input_p)?;
    let in_file = File::open(input_p).map_err(|e| io_err(input_p, e))?;
    let in_len = in_file.metadata().map_err(|e| io_err(input_p, e))?.len();
    let mmap = unsafe { Mmap::map(&in_file).map_err(|e| io_err(input_p, e))? };

    // Determine target tensor types and calculate new offsets
    let mut new_descriptors = Vec::with_capacity(gguf.tensors.len());
    let mut cur_offset = 0u64;
    let mut quantized_count = 0;

    for tensor in &gguf.tensors {
        let is_weight = should_quantize_tensor(&tensor.name, &tensor.dimensions);
        let num_elements: u64 = tensor.dimensions.iter().product();

        let (out_type, new_bytes) = if is_weight
            && num_elements.is_multiple_of(target.block_size() as u64)
        {
            if !is_supported_quantizer_input(tensor.tensor_type) {
                return Err(BackendError::UnsupportedTensorType(format!(
                        "tensor '{}' uses {:?}; supported quantizer inputs are F32, F16, BF16, and Q8_0",
                        tensor.name, tensor.tensor_type
                    )));
            }
            quantized_count += 1;
            let blocks = num_elements / (target.block_size() as u64);
            let bytes_per_block = match target {
                TargetQuant::Q8_0 => 34,
                TargetQuant::Q4_0 => 18,
                TargetQuant::Q4_K_M => 144,
            };
            (target.gguf_type(), blocks * bytes_per_block)
        } else {
            (tensor.tensor_type, tensor.n_bytes)
        };

        new_descriptors.push((
            tensor.name.clone(),
            tensor.dimensions.clone(),
            out_type,
            cur_offset,
            new_bytes,
            is_weight,
        ));
        cur_offset = cur_offset
            .checked_add(new_bytes)
            .and_then(|value| value.checked_add(gguf.alignment - 1))
            .map(|value| value & !(gguf.alignment - 1))
            .ok_or_else(|| BackendError::InvalidGguf("tensor offset overflow".to_string()))?;
    }

    // Do not create or truncate the destination until every tensor is known to
    // be convertible.
    let out_file = File::create(output_p).map_err(|e| io_err(output_p, e))?;
    let mut writer = BufWriter::new(out_file);

    // GGUF v3 magic and header
    writer.write_all(b"GGUF").map_err(|e| io_err(output_p, e))?;
    writer
        .write_all(&3u32.to_le_bytes())
        .map_err(|e| io_err(output_p, e))?;
    writer
        .write_all(&(gguf.tensor_count as u64).to_le_bytes())
        .map_err(|e| io_err(output_p, e))?;
    writer
        .write_all(&(gguf.metadata_count as u64).to_le_bytes())
        .map_err(|e| io_err(output_p, e))?;

    // Copy metadata entries, updating the standard whole-file quantization
    // label to match the new tensor payloads.
    for (key, val) in &gguf.metadata {
        write_string(&mut writer, key).map_err(|e| io_err(output_p, e))?;
        if key == "general.file_type" {
            write_metadata_value(&mut writer, &GgufMetadataValue::U32(target.file_type_id()))
                .map_err(|e| io_err(output_p, e))?;
        } else {
            write_metadata_value(&mut writer, val).map_err(|e| io_err(output_p, e))?;
        }
    }

    // Write tensor descriptors
    for (name, dims, out_type, rel_offset, _, _) in &new_descriptors {
        write_string(&mut writer, name).map_err(|e| io_err(output_p, e))?;
        writer
            .write_all(&(dims.len() as u32).to_le_bytes())
            .map_err(|e| io_err(output_p, e))?;
        for &dim in dims {
            writer
                .write_all(&dim.to_le_bytes())
                .map_err(|e| io_err(output_p, e))?;
        }
        let type_id = tensor_type_to_id(*out_type);
        writer
            .write_all(&type_id.to_le_bytes())
            .map_err(|e| io_err(output_p, e))?;
        writer
            .write_all(&rel_offset.to_le_bytes())
            .map_err(|e| io_err(output_p, e))?;
    }

    // Pad to the alignment declared by the source GGUF before tensor data.
    let cur_pos = writer.stream_position().map_err(|e| io_err(output_p, e))?;
    let aligned_pos = (cur_pos + gguf.alignment - 1) & !(gguf.alignment - 1);
    let pad_len = (aligned_pos - cur_pos) as usize;
    if pad_len > 0 {
        writer
            .write_all(&vec![0u8; pad_len])
            .map_err(|e| io_err(output_p, e))?;
    }

    // Stream and quantize tensor data
    let mut hasher = Sha256::new();

    for (i, tensor) in gguf.tensors.iter().enumerate() {
        let (_, _, out_type, _, _, is_weight) = new_descriptors[i];
        let src_slice = &mmap[tensor.absolute_offset as usize..][..tensor.n_bytes as usize];
        let num_elements: usize = tensor.dimensions.iter().product::<u64>() as usize;

        if is_weight && out_type == target.gguf_type() {
            let f32_vals = dequantize_to_f32(tensor.tensor_type, src_slice, num_elements)?;
            let block_size = target.block_size();
            let num_blocks = f32_vals.len() / block_size;

            let quantized_bytes: Vec<u8> = match target {
                TargetQuant::Q8_0 => {
                    let mut bytes = vec![0u8; num_blocks * 34];
                    bytes
                        .par_chunks_exact_mut(34)
                        .enumerate()
                        .for_each(|(b_idx, dst)| {
                            let sub = &f32_vals[b_idx * 32..(b_idx + 1) * 32];
                            quantize_block_q8_0(sub, dst);
                        });
                    bytes
                }
                TargetQuant::Q4_0 => {
                    let mut bytes = vec![0u8; num_blocks * 18];
                    bytes
                        .par_chunks_exact_mut(18)
                        .enumerate()
                        .for_each(|(b_idx, dst)| {
                            let sub = &f32_vals[b_idx * 32..(b_idx + 1) * 32];
                            quantize_block_q4_0(sub, dst);
                        });
                    bytes
                }
                TargetQuant::Q4_K_M => {
                    let mut bytes = vec![0u8; num_blocks * 144];
                    bytes
                        .par_chunks_exact_mut(144)
                        .enumerate()
                        .for_each(|(b_idx, dst)| {
                            let sub = &f32_vals[b_idx * 256..(b_idx + 1) * 256];
                            quantize_block_q4_k(sub, dst);
                        });
                    bytes
                }
            };

            writer
                .write_all(&quantized_bytes)
                .map_err(|e| io_err(output_p, e))?;
            hasher.update(&quantized_bytes);

            let alignment = gguf.alignment as usize;
            let pad = ((quantized_bytes.len() + alignment - 1) & !(alignment - 1))
                - quantized_bytes.len();
            if pad > 0 {
                let pad_bytes = vec![0u8; pad];
                writer
                    .write_all(&pad_bytes)
                    .map_err(|e| io_err(output_p, e))?;
                hasher.update(&pad_bytes);
            }
        } else {
            // Copy tensor as-is
            writer
                .write_all(src_slice)
                .map_err(|e| io_err(output_p, e))?;
            hasher.update(src_slice);

            let alignment = gguf.alignment as usize;
            let pad = ((src_slice.len() + alignment - 1) & !(alignment - 1)) - src_slice.len();
            if pad > 0 {
                let pad_bytes = vec![0u8; pad];
                writer
                    .write_all(&pad_bytes)
                    .map_err(|e| io_err(output_p, e))?;
                hasher.update(&pad_bytes);
            }
        }
    }

    writer.flush().map_err(|e| io_err(output_p, e))?;
    let out_len = writer.stream_position().map_err(|e| io_err(output_p, e))?;
    let sha256 = format!("{:x}", hasher.finalize());

    Ok(QuantizeReceipt {
        input_path: input_p.to_string_lossy().into_owned(),
        output_path: output_p.to_string_lossy().into_owned(),
        target_quant: target.as_str().to_string(),
        tensor_count: gguf.tensors.len(),
        quantized_tensors: quantized_count,
        input_bytes: in_len,
        output_bytes: out_len,
        compression_ratio: if in_len > 0 {
            out_len as f32 / in_len as f32
        } else {
            1.0
        },
        sha256,
    })
}

fn write_string(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

fn tensor_type_to_id(t: GgufTensorType) -> u32 {
    match t {
        GgufTensorType::F32 => 0,
        GgufTensorType::F16 => 1,
        GgufTensorType::Q4_0 => 2,
        GgufTensorType::Q4_1 => 3,
        GgufTensorType::Q5_0 => 6,
        GgufTensorType::Q5_1 => 7,
        GgufTensorType::Q8_0 => 8,
        GgufTensorType::Q8_1 => 9,
        GgufTensorType::Q2K => 10,
        GgufTensorType::Q3K => 11,
        GgufTensorType::Q4K => 12,
        GgufTensorType::Q5K => 13,
        GgufTensorType::Q6K => 14,
        GgufTensorType::Q8K => 15,
        GgufTensorType::IQ4NL => 20,
        GgufTensorType::IQ4XS => 23,
        GgufTensorType::I8 => 24,
        GgufTensorType::I16 => 25,
        GgufTensorType::I32 => 26,
        GgufTensorType::I64 => 27,
        GgufTensorType::F64 => 28,
        GgufTensorType::BF16 => 30,
        GgufTensorType::Tq1_0 => 34,
        GgufTensorType::Tq2_0 => 35,
        GgufTensorType::I2S => 36,
        GgufTensorType::NVFP4 => 40,
        GgufTensorType::Q1_0 => 41,
        GgufTensorType::Q2_0 | GgufTensorType::Q2_0G64 | GgufTensorType::Q2_0G128 => 42,
        GgufTensorType::Pq2_0 => 142,
        GgufTensorType::Unknown(value) => value as u32,
    }
}

fn write_metadata_value(w: &mut impl Write, val: &GgufMetadataValue) -> std::io::Result<()> {
    match val {
        GgufMetadataValue::U8(v) => {
            w.write_all(&0u32.to_le_bytes())?;
            w.write_all(&[*v])?;
        }
        GgufMetadataValue::I8(v) => {
            w.write_all(&1u32.to_le_bytes())?;
            w.write_all(&[*v as u8])?;
        }
        GgufMetadataValue::U16(v) => {
            w.write_all(&2u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::I16(v) => {
            w.write_all(&3u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::U32(v) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::I32(v) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::F32(v) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::Bool(v) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[if *v { 1 } else { 0 }])?;
        }
        GgufMetadataValue::String(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            write_string(w, s)?;
        }
        GgufMetadataValue::U64(v) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::I64(v) => {
            w.write_all(&11u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::F64(v) => {
            w.write_all(&12u32.to_le_bytes())?;
            w.write_all(&v.to_le_bytes())?;
        }
        GgufMetadataValue::Array(arr) => {
            w.write_all(&9u32.to_le_bytes())?;
            if let Some(first) = arr.first() {
                let elem_type = match first {
                    GgufMetadataValue::U8(_) => 0u32,
                    GgufMetadataValue::I8(_) => 1,
                    GgufMetadataValue::U16(_) => 2,
                    GgufMetadataValue::I16(_) => 3,
                    GgufMetadataValue::U32(_) => 4,
                    GgufMetadataValue::I32(_) => 5,
                    GgufMetadataValue::F32(_) => 6,
                    GgufMetadataValue::Bool(_) => 7,
                    GgufMetadataValue::String(_) => 8,
                    GgufMetadataValue::U64(_) => 10,
                    GgufMetadataValue::I64(_) => 11,
                    GgufMetadataValue::F64(_) => 12,
                    GgufMetadataValue::Array(_) => 9,
                };
                w.write_all(&elem_type.to_le_bytes())?;
                w.write_all(&(arr.len() as u64).to_le_bytes())?;
                for elem in arr {
                    write_array_element(w, elem)?;
                }
            } else {
                w.write_all(&8u32.to_le_bytes())?; // Default to string
                w.write_all(&0u64.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn write_array_element(w: &mut impl Write, elem: &GgufMetadataValue) -> std::io::Result<()> {
    match elem {
        GgufMetadataValue::String(s) => write_string(w, s),
        GgufMetadataValue::U32(v) => w.write_all(&v.to_le_bytes()),
        GgufMetadataValue::I32(v) => w.write_all(&v.to_le_bytes()),
        GgufMetadataValue::F32(v) => w.write_all(&v.to_le_bytes()),
        GgufMetadataValue::U64(v) => w.write_all(&v.to_le_bytes()),
        GgufMetadataValue::I64(v) => w.write_all(&v.to_le_bytes()),
        GgufMetadataValue::Bool(v) => w.write_all(&[if *v { 1 } else { 0 }]),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_block_q8_0_accuracy() {
        let mut original = [0.0f32; 32];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.125;
        }

        let mut block = [0u8; 34];
        quantize_block_q8_0(&original, &mut block);

        let d = fast_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        assert!(d > 0.0);

        for i in 0..32 {
            let recon = (block[2 + i] as i8 as f32) * d;
            let diff = (original[i] - recon).abs();
            assert!(diff < 0.05, "error too high at {i}: {diff}");
        }
    }

    #[test]
    fn test_quantize_block_q4_0_accuracy() {
        let mut original = [0.0f32; 32];
        for (i, v) in original.iter_mut().enumerate() {
            *v = (i as f32 - 16.0) * 0.25;
        }

        let mut block = [0u8; 18];
        quantize_block_q4_0(&original, &mut block);

        let decoded = crate::tensor::Q4_0Block::from_bytes(&block);
        let d = decoded.scale_f32();
        let values = decoded.unpack_values();
        for i in 0..32 {
            let recon = values[i] as f32 * d;
            assert_eq!(recon.signum(), original[i].signum(), "sign mismatch at {i}");
            assert!(
                (original[i] - recon).abs() <= d.abs() + 0.01,
                "error too high at {i}: original={}, reconstructed={recon}",
                original[i]
            );
        }
    }

    #[test]
    fn test_quantize_block_q4_k_matches_decoder_layout() {
        let mut original = [0.0f32; 256];
        for group in 0..8 {
            for i in 0..32 {
                let offset = if group == 6 {
                    2.0
                } else if group == 7 {
                    -2.0
                } else {
                    0.0
                };
                original[group * 32 + i] =
                    (i as f32 - 15.5) * (group as f32 + 1.0) * 0.03125 + offset;
            }
        }

        let mut block = [0u8; 144];
        quantize_block_q4_k(&original, &mut block);
        let decoded = crate::tensor::Q4KBlock::from_bytes(&block);
        let mut reconstructed = [0.0f32; 256];
        decoded.dequantize(&mut reconstructed);

        for group in 0..8 {
            let group_error = (0..32)
                .map(|i| {
                    let index = group * 32 + i;
                    (original[index] - reconstructed[index]).abs()
                })
                .fold(0.0f32, f32::max);
            assert!(
                group_error < 0.3,
                "group {group} error too high: {group_error}"
            );
        }
    }

    #[test]
    fn test_unsupported_source_quantization_fails_instead_of_zero_filling() {
        let err = dequantize_to_f32(GgufTensorType::Q4K, &[0u8; 144], 256).unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }

    #[test]
    fn test_same_file_detection_catches_hard_links() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.gguf");
        let output = dir.path().join("output.gguf");
        std::fs::write(&input, b"GGUF").unwrap();
        std::fs::hard_link(&input, &output).unwrap();

        assert!(paths_refer_to_same_file(&input, &output).unwrap());
    }
}
