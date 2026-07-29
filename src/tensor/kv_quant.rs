//! Quantized KV-cache block structures and attention helpers.

use serde::{Deserialize, Serialize};
use std::fmt;
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

pub const KV_QUANT_BLOCK_VALUES: usize = 32;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheQuantization {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl fmt::Display for KvCacheQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
        }
    }
}

impl std::str::FromStr for KvCacheQuantization {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "f16" | "fp16" | "none" | "off" => Ok(Self::F16),
            "q8_0" | "q80" | "q8" => Ok(Self::Q8_0),
            "q4_0" | "q40" | "q4" => Ok(Self::Q4_0),
            _ => Err(format!(
                "unknown kv cache quantization format '{s}'; supported: f16, q8_0, q4_0"
            )),
        }
    }
}

/// Block of 32 FP16/FP32 elements quantized to 8-bit Q8_0.
/// Storage per 32 elements: 1 f16 scale (2 bytes) + 32 i8 values (32 bytes) = 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ8_0 {
    pub scale: u16, // f16 bits
    pub qs: [i8; KV_QUANT_BLOCK_VALUES],
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self {
            scale: 0,
            qs: [0; KV_QUANT_BLOCK_VALUES],
        }
    }
}

/// Block of 32 FP16/FP32 elements quantized to 4-bit Q4_0.
/// Storage per 32 elements: 1 f16 scale (2 bytes) + 16 u8 packed nibbles (16 bytes) = 18 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4_0 {
    pub scale: u16, // f16 bits
    pub qs: [u8; KV_QUANT_BLOCK_VALUES / 2],
}

impl Default for BlockQ4_0 {
    fn default() -> Self {
        Self {
            scale: 0,
            qs: [0; KV_QUANT_BLOCK_VALUES / 2],
        }
    }
}

/// Quantize a slice of 32 f32 values into a `BlockQ8_0`.
pub fn quantize_block_q8_0(src: &[f32; KV_QUANT_BLOCK_VALUES]) -> BlockQ8_0 {
    let mut amax = 0.0f32;
    for &v in src {
        let abs_v = v.abs();
        if abs_v > amax {
            amax = abs_v;
        }
    }
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0i8; KV_QUANT_BLOCK_VALUES];
    for i in 0..KV_QUANT_BLOCK_VALUES {
        let val = (src[i] * id).round();
        qs[i] = val.clamp(-127.0, 127.0) as i8;
    }
    BlockQ8_0 {
        scale: super::f32_to_f16_bits(d),
        qs,
    }
}

/// Dequantize a `BlockQ8_0` into a full block or a final partial row.
pub fn dequantize_block_q8_0(block: &BlockQ8_0, dst: &mut [f32]) {
    debug_assert!(dst.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(block.scale);
    for (i, value) in dst.iter_mut().enumerate() {
        *value = block.qs[i] as f32 * d;
    }
}

/// Compute a dot product against a full block or a final partial row.
pub fn vec_dot_q8_0(query: &[f32], key_block: &BlockQ8_0) -> f32 {
    debug_assert!(query.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(key_block.scale);
    let mut sum = 0.0f32;
    for (i, &query_value) in query.iter().enumerate() {
        sum += query_value * (key_block.qs[i] as f32);
    }
    sum * d
}

pub fn quantize_row_q8_0(src: &[f32], dst: &mut [BlockQ8_0]) {
    debug_assert_eq!(dst.len(), src.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in dst.iter_mut().zip(src.chunks(KV_QUANT_BLOCK_VALUES)) {
        let mut values = [0.0f32; KV_QUANT_BLOCK_VALUES];
        values[..chunk.len()].copy_from_slice(chunk);
        *block = quantize_block_q8_0(&values);
    }
}

pub fn dequantize_row_q8_0(src: &[BlockQ8_0], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in src.iter().zip(dst.chunks_mut(KV_QUANT_BLOCK_VALUES)) {
        dequantize_block_q8_0(block, chunk);
    }
}

pub fn vec_dot_row_q8_0(query: &[f32], key_blocks: &[BlockQ8_0]) -> f32 {
    debug_assert_eq!(
        key_blocks.len(),
        query.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    #[cfg(target_arch = "x86_64")]
    if kv_quant_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA check. The implementation
        // handles a final partial block with the scalar helper.
        return unsafe { vec_dot_row_q8_0_avx2(query, key_blocks) };
    }
    query
        .chunks(KV_QUANT_BLOCK_VALUES)
        .zip(key_blocks)
        .map(|(chunk, block)| vec_dot_q8_0(chunk, block))
        .sum()
}

/// Fused `out += probability * dequant(blocks)` for one Q8_0 row.
///
/// Keeping dequantization inside the accumulation avoids a temporary f32
/// block and lets the x86 path expand eight i8 values directly into an AVX2
/// register. The scalar fallback preserves support on every other target.
pub fn axpy_row_q8_0(out: &mut [f32], probability: f32, value_blocks: &[BlockQ8_0]) {
    debug_assert_eq!(
        value_blocks.len(),
        out.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    #[cfg(target_arch = "x86_64")]
    if kv_quant_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA check.
        unsafe { axpy_row_q8_0_avx2(out, probability, value_blocks) };
        return;
    }
    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        let d = super::f16_bits_to_f32(block.scale);
        for (i, out_value) in out_chunk.iter_mut().enumerate() {
            *out_value = (probability * d).mul_add(block.qs[i] as f32, *out_value);
        }
    }
}

/// Quantize a slice of 32 f32 values into a `BlockQ4_0`.
pub fn quantize_block_q4_0(src: &[f32; KV_QUANT_BLOCK_VALUES]) -> BlockQ4_0 {
    let mut amax = 0.0f32;
    let mut signed_max = 0.0f32;
    for &v in src {
        let abs_v = v.abs();
        if abs_v > amax {
            amax = abs_v;
            signed_max = v;
        }
    }
    // Match ggml's Q4_0 reference quantizer: the sign of the first
    // max-magnitude value chooses which side receives the -8 code.
    let d = signed_max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0u8; KV_QUANT_BLOCK_VALUES / 2];
    for i in 0..KV_QUANT_BLOCK_VALUES / 2 {
        let x0 = (src[i] * id + 8.5).floor().clamp(0.0, 15.0) as u8;
        let x1 = (src[i + KV_QUANT_BLOCK_VALUES / 2] * id + 8.5)
            .floor()
            .clamp(0.0, 15.0) as u8;
        qs[i] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
    }
    BlockQ4_0 {
        scale: super::f32_to_f16_bits(d),
        qs,
    }
}

/// Dequantize a `BlockQ4_0` into a full block or a final partial row.
pub fn dequantize_block_q4_0(block: &BlockQ4_0, dst: &mut [f32]) {
    debug_assert!(dst.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(block.scale);
    for (i, value) in dst.iter_mut().enumerate() {
        let packed = block.qs[i % (KV_QUANT_BLOCK_VALUES / 2)];
        let quant = if i < KV_QUANT_BLOCK_VALUES / 2 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        *value = (quant as f32 - 8.0) * d;
    }
}

/// Compute a dot product against a full block or a final partial row.
pub fn vec_dot_q4_0(query: &[f32], key_block: &BlockQ4_0) -> f32 {
    debug_assert!(query.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(key_block.scale);
    let mut sum = 0.0f32;
    for (i, &query_value) in query.iter().enumerate() {
        let packed = key_block.qs[i % (KV_QUANT_BLOCK_VALUES / 2)];
        let quant = if i < KV_QUANT_BLOCK_VALUES / 2 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        sum += query_value * (quant as f32 - 8.0);
    }
    sum * d
}

pub fn quantize_row_q4_0(src: &[f32], dst: &mut [BlockQ4_0]) {
    debug_assert_eq!(dst.len(), src.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in dst.iter_mut().zip(src.chunks(KV_QUANT_BLOCK_VALUES)) {
        let mut values = [0.0f32; KV_QUANT_BLOCK_VALUES];
        values[..chunk.len()].copy_from_slice(chunk);
        *block = quantize_block_q4_0(&values);
    }
}

pub fn dequantize_row_q4_0(src: &[BlockQ4_0], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in src.iter().zip(dst.chunks_mut(KV_QUANT_BLOCK_VALUES)) {
        dequantize_block_q4_0(block, chunk);
    }
}

pub fn vec_dot_row_q4_0(query: &[f32], key_blocks: &[BlockQ4_0]) -> f32 {
    debug_assert_eq!(
        key_blocks.len(),
        query.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    #[cfg(target_arch = "x86_64")]
    if kv_quant_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA check. The implementation
        // handles a final partial block with the scalar helper.
        return unsafe { vec_dot_row_q4_0_avx2(query, key_blocks) };
    }
    query
        .chunks(KV_QUANT_BLOCK_VALUES)
        .zip(key_blocks)
        .map(|(chunk, block)| vec_dot_q4_0(chunk, block))
        .sum()
}

/// Fused `out += probability * dequant(blocks)` for one Q4_0 row.
pub fn axpy_row_q4_0(out: &mut [f32], probability: f32, value_blocks: &[BlockQ4_0]) {
    debug_assert_eq!(
        value_blocks.len(),
        out.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    #[cfg(target_arch = "x86_64")]
    if kv_quant_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA check.
        unsafe { axpy_row_q4_0_avx2(out, probability, value_blocks) };
        return;
    }
    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        let d = super::f16_bits_to_f32(block.scale);
        for (i, out_value) in out_chunk.iter_mut().enumerate() {
            let packed = block.qs[i % (KV_QUANT_BLOCK_VALUES / 2)];
            let quant = if i < KV_QUANT_BLOCK_VALUES / 2 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            *out_value = (probability * d).mul_add(quant as f32 - 8.0, *out_value);
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn kv_quant_avx2_fma_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32x8(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::_mm256_storeu_ps;
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` holds exactly one 256-bit vector.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), value) };
    let t0 = lanes[0] + lanes[4];
    let t1 = lanes[1] + lanes[5];
    let t2 = lanes[2] + lanes[6];
    let t3 = lanes[3] + lanes[7];
    (t0 + t2) + (t1 + t3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_row_q8_0_avx2(query: &[f32], key_blocks: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_loadl_epi64,
    };

    let mut total = 0.0f32;
    for (query_chunk, block) in query.chunks(KV_QUANT_BLOCK_VALUES).zip(key_blocks) {
        if query_chunk.len() != KV_QUANT_BLOCK_VALUES {
            total += vec_dot_q8_0(query_chunk, block);
            continue;
        }
        let mut acc = _mm256_setzero_ps();
        for offset in [0usize, 8, 16, 24] {
            // SAFETY: a full block has 32 query/i8 values and every load is
            // within one of its four eight-element groups.
            let packed =
                unsafe { _mm_loadl_epi64(block.qs.as_ptr().add(offset) as *const __m128i) };
            let quant = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
            let input = unsafe { _mm256_loadu_ps(query_chunk.as_ptr().add(offset)) };
            acc = _mm256_fmadd_ps(input, quant, acc);
        }
        total += unsafe { horizontal_sum_f32x8(acc) } * super::f16_bits_to_f32(block.scale);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_row_q4_0_avx2(query: &[f32], key_blocks: &[BlockQ4_0]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_set1_epi32, _mm256_setzero_ps, _mm256_sub_epi32, _mm_and_si128, _mm_loadl_epi64,
        _mm_set1_epi8, _mm_srli_epi16,
    };

    let mask = _mm_set1_epi8(0x0f);
    let bias = _mm256_set1_epi32(8);
    let mut total = 0.0f32;
    for (query_chunk, block) in query.chunks(KV_QUANT_BLOCK_VALUES).zip(key_blocks) {
        if query_chunk.len() != KV_QUANT_BLOCK_VALUES {
            total += vec_dot_q4_0(query_chunk, block);
            continue;
        }
        let mut acc = _mm256_setzero_ps();
        for byte_offset in [0usize, 8] {
            // SAFETY: each iteration loads eight of the block's 16 packed bytes.
            let packed =
                unsafe { _mm_loadl_epi64(block.qs.as_ptr().add(byte_offset) as *const __m128i) };
            let low = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_and_si128(packed, mask)), bias);
            let high = _mm256_sub_epi32(
                _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(packed, 4), mask)),
                bias,
            );
            let query_low = unsafe { _mm256_loadu_ps(query_chunk.as_ptr().add(byte_offset)) };
            let query_high = unsafe { _mm256_loadu_ps(query_chunk.as_ptr().add(16 + byte_offset)) };
            acc = _mm256_fmadd_ps(query_low, _mm256_cvtepi32_ps(low), acc);
            acc = _mm256_fmadd_ps(query_high, _mm256_cvtepi32_ps(high), acc);
        }
        total += unsafe { horizontal_sum_f32x8(acc) } * super::f16_bits_to_f32(block.scale);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_row_q8_0_avx2(out: &mut [f32], probability: f32, value_blocks: &[BlockQ8_0]) {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _mm_loadl_epi64,
    };

    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        if out_chunk.len() != KV_QUANT_BLOCK_VALUES {
            let d = super::f16_bits_to_f32(block.scale);
            for (i, out_value) in out_chunk.iter_mut().enumerate() {
                *out_value = (probability * d).mul_add(block.qs[i] as f32, *out_value);
            }
            continue;
        }
        let factor = _mm256_set1_ps(probability * super::f16_bits_to_f32(block.scale));
        for offset in [0usize, 8, 16, 24] {
            // SAFETY: full chunks and blocks hold all four eight-element groups.
            unsafe {
                let packed = _mm_loadl_epi64(block.qs.as_ptr().add(offset) as *const __m128i);
                let quant = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(packed));
                let current = _mm256_loadu_ps(out_chunk.as_ptr().add(offset));
                _mm256_storeu_ps(
                    out_chunk.as_mut_ptr().add(offset),
                    _mm256_fmadd_ps(factor, quant, current),
                );
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_row_q4_0_avx2(out: &mut [f32], probability: f32, value_blocks: &[BlockQ4_0]) {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_epi32, _mm_and_si128,
        _mm_loadl_epi64, _mm_set1_epi8, _mm_srli_epi16,
    };

    let mask = _mm_set1_epi8(0x0f);
    let bias = _mm256_set1_epi32(8);
    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        if out_chunk.len() != KV_QUANT_BLOCK_VALUES {
            let d = super::f16_bits_to_f32(block.scale);
            for (i, out_value) in out_chunk.iter_mut().enumerate() {
                let packed = block.qs[i % (KV_QUANT_BLOCK_VALUES / 2)];
                let quant = if i < KV_QUANT_BLOCK_VALUES / 2 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                *out_value = (probability * d).mul_add(quant as f32 - 8.0, *out_value);
            }
            continue;
        }
        let factor = _mm256_set1_ps(probability * super::f16_bits_to_f32(block.scale));
        for byte_offset in [0usize, 8] {
            // SAFETY: full chunks and blocks hold the referenced eight-byte
            // packed group and both corresponding eight-f32 output groups.
            unsafe {
                let packed = _mm_loadl_epi64(block.qs.as_ptr().add(byte_offset) as *const __m128i);
                let low = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_cvtepu8_epi32(_mm_and_si128(packed, mask)),
                    bias,
                ));
                let high = _mm256_cvtepi32_ps(_mm256_sub_epi32(
                    _mm256_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(packed, 4), mask)),
                    bias,
                ));
                let low_offset = byte_offset;
                let high_offset = 16 + byte_offset;
                let current_low = _mm256_loadu_ps(out_chunk.as_ptr().add(low_offset));
                let current_high = _mm256_loadu_ps(out_chunk.as_ptr().add(high_offset));
                _mm256_storeu_ps(
                    out_chunk.as_mut_ptr().add(low_offset),
                    _mm256_fmadd_ps(factor, low, current_low),
                );
                _mm256_storeu_ps(
                    out_chunk.as_mut_ptr().add(high_offset),
                    _mm256_fmadd_ps(factor, high, current_high),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn q8_0_roundtrip_and_dot_match_dequantized_values() {
        let mut original = [0.0f32; KV_QUANT_BLOCK_VALUES];
        for (i, value) in original.iter_mut().enumerate() {
            *value = (i as f32 - 16.0) * 0.25;
        }
        let block = quantize_block_q8_0(&original);
        let mut reconstructed = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_q8_0(&block, &mut reconstructed);

        for i in 0..KV_QUANT_BLOCK_VALUES {
            let diff = (original[i] - reconstructed[i]).abs();
            assert!(
                diff < 0.05,
                "Q8_0 roundtrip diff too high at index {i}: {diff}"
            );
        }
        let query: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 7.0) * 0.125);
        let expected: f32 = query.iter().zip(reconstructed).map(|(q, k)| q * k).sum();
        assert!((vec_dot_q8_0(&query, &block) - expected).abs() < 1e-5);

        let mut accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| i as f32 * 0.01);
        let expected_accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| 0.375f32.mul_add(reconstructed[i], i as f32 * 0.01));
        axpy_row_q8_0(&mut accumulated, 0.375, &[block]);
        for (actual, expected) in accumulated.iter().zip(expected_accumulated) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn q4_0_roundtrip_and_dot_match_dequantized_values() {
        let mut original = [0.0f32; KV_QUANT_BLOCK_VALUES];
        for (i, value) in original.iter_mut().enumerate() {
            *value = (i as f32 - 16.0) * 0.25;
        }
        let block = quantize_block_q4_0(&original);
        let mut reconstructed = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_q4_0(&block, &mut reconstructed);

        for i in 0..KV_QUANT_BLOCK_VALUES {
            let diff = (original[i] - reconstructed[i]).abs();
            assert!(
                diff < 0.30,
                "Q4_0 roundtrip diff too high at index {i}: {diff}"
            );
        }
        let query: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 7.0) * 0.125);
        let expected: f32 = query.iter().zip(reconstructed).map(|(q, k)| q * k).sum();
        assert!((vec_dot_q4_0(&query, &block) - expected).abs() < 1e-5);

        let mut accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| i as f32 * 0.01);
        let expected_accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| 0.375f32.mul_add(reconstructed[i], i as f32 * 0.01));
        axpy_row_q4_0(&mut accumulated, 0.375, &[block]);
        for (actual, expected) in accumulated.iter().zip(expected_accumulated) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn quantized_blocks_have_wire_compatible_sizes() {
        assert_eq!(size_of::<BlockQ8_0>(), 34);
        assert_eq!(size_of::<BlockQ4_0>(), 18);
    }

    #[test]
    fn q4_0_uses_reference_signed_max_scale() {
        let mut negative_max = [0.0; KV_QUANT_BLOCK_VALUES];
        negative_max[0] = -8.0;
        negative_max[KV_QUANT_BLOCK_VALUES / 2] = 7.0;
        let negative_block = quantize_block_q4_0(&negative_max);
        assert_eq!(negative_block.scale, super::super::f32_to_f16_bits(1.0));
        assert_eq!(negative_block.qs[0], 0xf0);

        let mut positive_max = [0.0; KV_QUANT_BLOCK_VALUES];
        positive_max[0] = 8.0;
        positive_max[KV_QUANT_BLOCK_VALUES / 2] = -7.0;
        let positive_block = quantize_block_q4_0(&positive_max);
        assert_eq!(positive_block.scale, super::super::f32_to_f16_bits(-1.0));
        assert_eq!(positive_block.qs[0], 0xf0);
    }

    #[test]
    fn scale_conversion_preserves_small_nonzero_values() {
        let value = 5.960_464_5e-8;
        assert_eq!(
            super::super::f16_bits_to_f32(super::super::f32_to_f16_bits(value)),
            value
        );
    }
}
