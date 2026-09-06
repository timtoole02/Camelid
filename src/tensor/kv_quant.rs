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
    Fp8E4m3,
    Fp8E5m2,
}

impl fmt::Display for KvCacheQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Fp8E4m3 => write!(f, "fp8_e4m3"),
            Self::Fp8E5m2 => write!(f, "fp8_e5m2"),
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
            "fp8_e4m3" | "fp8_e4" | "fp8" | "e4m3" => Ok(Self::Fp8E4m3),
            "fp8_e5m2" | "fp8_e5" | "e5m2" => Ok(Self::Fp8E5m2),
            _ => Err(format!(
                "unknown kv cache quantization format '{s}'; supported: f16, q8_0, q4_0, fp8_e4m3, fp8_e5m2"
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

/// Block of 32 FP16/FP32 elements quantized to 8-bit FP8 E4M3FN (with f16 block scale).
/// Storage per 32 elements: 1 f16 scale (2 bytes) + 32 u8 values (32 bytes) = 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFp8E4m3 {
    pub scale: u16, // f16 bits
    pub qs: [u8; KV_QUANT_BLOCK_VALUES],
}

impl Default for BlockFp8E4m3 {
    fn default() -> Self {
        Self {
            scale: 0,
            qs: [0; KV_QUANT_BLOCK_VALUES],
        }
    }
}

/// Block of 32 FP16/FP32 elements quantized to 8-bit FP8 E5M2 (with f16 block scale).
/// Storage per 32 elements: 1 f16 scale (2 bytes) + 32 u8 values (32 bytes) = 34 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFp8E5m2 {
    pub scale: u16, // f16 bits
    pub qs: [u8; KV_QUANT_BLOCK_VALUES],
}

impl Default for BlockFp8E5m2 {
    fn default() -> Self {
        Self {
            scale: 0,
            qs: [0; KV_QUANT_BLOCK_VALUES],
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

// =========================================================================
// FP8 E4M3 (E4M3FN) and E5M2 Quantization & Attention Primitives
// =========================================================================

/// Convert a single f32 value to an 8-bit FP8 E4M3FN byte.
/// Exponent bias: 7. Range: ~1.95e-3 to 448.0. NaN: 0x7F / 0xFF.
#[inline]
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let abs_v = val.abs();

    if abs_v == 0.0 || !abs_v.is_finite() {
        if val.is_nan() {
            return (sign << 7) | 0x7F;
        }
        if val.is_infinite() {
            return (sign << 7) | 0x7E; // Clamped to max finite
        }
        return sign << 7;
    }

    // Subnormal threshold: 2^-6 * (1/8) = 2^-9 = 1.0 / 512.0
    if abs_v < 0.015625 {
        let m = (abs_v * 512.0).round() as u32;
        if m == 0 {
            return sign << 7;
        }
        if m >= 8 {
            return (sign << 7) | (1 << 3);
        }
        return (sign << 7) | (m as u8);
    }

    if abs_v >= 448.0 {
        return (sign << 7) | 0x7E;
    }

    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7FFFFF;
    let mut new_exp = f32_exp - 127 + 7;

    let mut mant_3bit = (f32_mant + 0x80000) >> 20;
    if mant_3bit >= 8 {
        new_exp += 1;
        mant_3bit = 0;
    }

    if new_exp >= 15 {
        let m = (((abs_v / 256.0) - 1.0) * 8.0).round().clamp(0.0, 6.0) as u8;
        return (sign << 7) | (0xF << 3) | m;
    }
    if new_exp < 1 {
        let m = (abs_v * 512.0).round().clamp(0.0, 7.0) as u8;
        return (sign << 7) | m;
    }

    (sign << 7) | ((new_exp as u8) << 3) | (mant_3bit as u8)
}

/// Convert an 8-bit FP8 E4M3FN byte to f32.
///
/// Pure bit assembly: an FP8 exponent/mantissa field maps onto the f32 fields by a
/// constant exponent rebias (`-7 + 127 = +120`) and a mantissa shift, so no
/// `powi`/`exp2` call is needed.
///
/// This is the *definition* of the decoding, and it is `const` so that
/// [`FP8_E4M3_LUT`] can be built from it at compile time. The hot row kernels index
/// that table rather than calling this, because an FP8 byte has only 256 possible
/// values and a 1 KiB L1-resident table beats re-deriving the bits per element:
/// measured on this row shape, `powi` 950 ns, this function 140 ns, the table
/// 59 ns — the last being parity with the scalar Q8_0 kernel.
#[inline]
pub const fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = ((byte as u32) & 0x80) << 24;
    let exp = ((byte >> 3) & 0x0F) as u32;
    let mant = (byte & 0x07) as u32;

    if exp == 0 {
        // Subnormal: mant * 2^-9. `mant == 0` yields a correctly signed zero.
        // 0x3B00_0000 is the f32 bit pattern for 2^-9.
        return f32::from_bits(sign | 0x3B00_0000) * (mant as f32);
    }
    if exp == 0x0F && mant == 0x07 {
        // E4M3FN reserves S.1111.111 for NaN and has no infinity encoding.
        return f32::NAN;
    }
    // Normal: rebias the exponent (-7 + 127) and left-align the 3-bit mantissa.
    f32::from_bits(sign | ((exp + 120) << 23) | (mant << 20))
}

/// Convert a single f32 value to an 8-bit FP8 E5M2 byte.
/// Exponent bias: 15. Range: ~1.52e-5 to 57344.0.
#[inline]
pub fn f32_to_fp8_e5m2(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let abs_v = val.abs();

    if abs_v == 0.0 || !abs_v.is_finite() {
        if val.is_nan() {
            return (sign << 7) | (0x1F << 2) | 0x1;
        }
        if val.is_infinite() {
            return (sign << 7) | (0x1F << 2);
        }
        return sign << 7;
    }

    // Below E5M2's smallest normal (2^-14) the encoding is subnormal: m * 2^-16,
    // m in 1..=3. Written as a division so the constant stays exactly 2^-14
    // (clippy::excessive_precision rejects the 0.00006103515625 spelling, and its
    // suggested truncation would move the threshold).
    if abs_v < 1.0 / 16384.0 {
        let m = (abs_v * 65536.0).round() as u32;
        if m == 0 {
            return sign << 7;
        }
        if m >= 4 {
            return (sign << 7) | (1 << 2);
        }
        return (sign << 7) | (m as u8);
    }

    if abs_v >= 57344.0 {
        return (sign << 7) | (0x1E << 2) | 0x3;
    }

    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7FFFFF;
    let mut new_exp = f32_exp - 127 + 15;

    let mut mant_2bit = (f32_mant + 0x100000) >> 21;
    if mant_2bit >= 4 {
        new_exp += 1;
        mant_2bit = 0;
    }

    if new_exp >= 31 {
        return (sign << 7) | (0x1E << 2) | 0x3;
    }
    if new_exp < 1 {
        let m = (abs_v * 65536.0).round().clamp(0.0, 3.0) as u8;
        return (sign << 7) | m;
    }

    (sign << 7) | ((new_exp as u8) << 2) | (mant_2bit as u8)
}

/// Convert an 8-bit FP8 E5M2 byte to f32.
#[inline]
pub const fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = ((byte as u32) & 0x80) << 24;
    let exp = ((byte >> 2) & 0x1F) as u32;
    let mant = (byte & 0x03) as u32;

    if exp == 0 {
        // Subnormal: mant * 2^-16. 0x3780_0000 is the f32 bit pattern for 2^-16.
        return f32::from_bits(sign | 0x3780_0000) * (mant as f32);
    }
    if exp == 0x1F {
        // E5M2 keeps IEEE semantics up here: mant == 0 is infinity, otherwise NaN.
        return f32::from_bits(sign | 0x7F80_0000 | (mant << 21));
    }
    // Normal: rebias the exponent (-15 + 127) and left-align the 2-bit mantissa.
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 21))
}

/// Every E4M3 code decoded once, at compile time, from [`fp8_e4m3_to_f32`].
///
/// Built from the decoder rather than written out, so the two cannot diverge;
/// `fp8_lookup_tables_match_their_decoders` pins that as well. 1 KiB, so it stays
/// L1-resident through an attention row.
pub const FP8_E4M3_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = fp8_e4m3_to_f32(i as u8);
        i += 1;
    }
    table
};

/// Every E5M2 code decoded once, at compile time, from [`fp8_e5m2_to_f32`].
pub const FP8_E5M2_LUT: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = fp8_e5m2_to_f32(i as u8);
        i += 1;
    }
    table
};

/// Smallest positive value f16 can hold (`2^-24`, the min subnormal).
///
/// FP8 block scales are stored as f16 bits, exactly like Q8_0/Q4_0. But those divide
/// `amax` by 127 / 8 whereas FP8 divides by 448 / 57344, so an FP8 scale sits 3.5x
/// (E4M3) to 451x (E5M2) closer to the bottom of the f16 range. Without a floor, a
/// block whose `amax` fell under `448 * 2^-24` (E4M3) or `57344 * 2^-24` (E5M2 —
/// i.e. 1.7e-3, an entirely ordinary magnitude) produced a scale of exactly zero and
/// silently dequantized all 32 elements to 0.0, deleting those positions from
/// attention with no diagnostic.
const F16_MIN_POSITIVE: f32 = 5.960_464_5e-8;

/// Clamp a block scale away from f16 underflow, preserving the exact `amax / range`
/// value everywhere it is representable.
///
/// Scaling `amax` onto the format maximum is the standard FP8 convention and is what
/// gives FP8 its outlier headroom, so the divisor itself is kept. A power-of-two scale
/// was measured as an alternative and is worse in every regime (it throws away up to a
/// full binade of usable range), so this floors rather than rounds.
#[inline]
fn fp8_block_scale(amax: f32, range: f32) -> f32 {
    let d = amax / range;
    if amax > 0.0 && d < F16_MIN_POSITIVE {
        F16_MIN_POSITIVE
    } else {
        d
    }
}

pub fn quantize_block_fp8_e4m3(src: &[f32; KV_QUANT_BLOCK_VALUES]) -> BlockFp8E4m3 {
    let mut amax = 0.0f32;
    for &v in src {
        let abs_v = v.abs();
        if abs_v > amax {
            amax = abs_v;
        }
    }
    let d = fp8_block_scale(amax, 448.0);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0u8; KV_QUANT_BLOCK_VALUES];
    for i in 0..KV_QUANT_BLOCK_VALUES {
        qs[i] = f32_to_fp8_e4m3(src[i] * id);
    }
    BlockFp8E4m3 {
        scale: super::f32_to_f16_bits(d),
        qs,
    }
}

pub fn dequantize_block_fp8_e4m3(block: &BlockFp8E4m3, dst: &mut [f32]) {
    debug_assert!(dst.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(block.scale);
    for (i, value) in dst.iter_mut().enumerate() {
        *value = FP8_E4M3_LUT[block.qs[i] as usize] * d;
    }
}

pub fn vec_dot_fp8_e4m3(query: &[f32], key_block: &BlockFp8E4m3) -> f32 {
    debug_assert!(query.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(key_block.scale);
    let mut sum = 0.0f32;
    for (i, &query_value) in query.iter().enumerate() {
        sum += query_value * FP8_E4M3_LUT[key_block.qs[i] as usize];
    }
    sum * d
}

pub fn quantize_row_fp8_e4m3(src: &[f32], dst: &mut [BlockFp8E4m3]) {
    debug_assert_eq!(dst.len(), src.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in dst.iter_mut().zip(src.chunks(KV_QUANT_BLOCK_VALUES)) {
        let mut values = [0.0f32; KV_QUANT_BLOCK_VALUES];
        values[..chunk.len()].copy_from_slice(chunk);
        *block = quantize_block_fp8_e4m3(&values);
    }
}

pub fn dequantize_row_fp8_e4m3(src: &[BlockFp8E4m3], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in src.iter().zip(dst.chunks_mut(KV_QUANT_BLOCK_VALUES)) {
        dequantize_block_fp8_e4m3(block, chunk);
    }
}

pub fn vec_dot_row_fp8_e4m3(query: &[f32], key_blocks: &[BlockFp8E4m3]) -> f32 {
    debug_assert_eq!(
        key_blocks.len(),
        query.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    query
        .chunks(KV_QUANT_BLOCK_VALUES)
        .zip(key_blocks)
        .map(|(chunk, block)| vec_dot_fp8_e4m3(chunk, block))
        .sum()
}

pub fn axpy_row_fp8_e4m3(out: &mut [f32], probability: f32, value_blocks: &[BlockFp8E4m3]) {
    debug_assert_eq!(
        value_blocks.len(),
        out.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        let d = super::f16_bits_to_f32(block.scale);
        for (i, out_value) in out_chunk.iter_mut().enumerate() {
            *out_value = (probability * d).mul_add(FP8_E4M3_LUT[block.qs[i] as usize], *out_value);
        }
    }
}

pub fn quantize_block_fp8_e5m2(src: &[f32; KV_QUANT_BLOCK_VALUES]) -> BlockFp8E5m2 {
    let mut amax = 0.0f32;
    for &v in src {
        let abs_v = v.abs();
        if abs_v > amax {
            amax = abs_v;
        }
    }
    let d = fp8_block_scale(amax, 57344.0);
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0u8; KV_QUANT_BLOCK_VALUES];
    for i in 0..KV_QUANT_BLOCK_VALUES {
        qs[i] = f32_to_fp8_e5m2(src[i] * id);
    }
    BlockFp8E5m2 {
        scale: super::f32_to_f16_bits(d),
        qs,
    }
}

pub fn dequantize_block_fp8_e5m2(block: &BlockFp8E5m2, dst: &mut [f32]) {
    debug_assert!(dst.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(block.scale);
    for (i, value) in dst.iter_mut().enumerate() {
        *value = FP8_E5M2_LUT[block.qs[i] as usize] * d;
    }
}

pub fn vec_dot_fp8_e5m2(query: &[f32], key_block: &BlockFp8E5m2) -> f32 {
    debug_assert!(query.len() <= KV_QUANT_BLOCK_VALUES);
    let d = super::f16_bits_to_f32(key_block.scale);
    let mut sum = 0.0f32;
    for (i, &query_value) in query.iter().enumerate() {
        sum += query_value * FP8_E5M2_LUT[key_block.qs[i] as usize];
    }
    sum * d
}

pub fn quantize_row_fp8_e5m2(src: &[f32], dst: &mut [BlockFp8E5m2]) {
    debug_assert_eq!(dst.len(), src.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in dst.iter_mut().zip(src.chunks(KV_QUANT_BLOCK_VALUES)) {
        let mut values = [0.0f32; KV_QUANT_BLOCK_VALUES];
        values[..chunk.len()].copy_from_slice(chunk);
        *block = quantize_block_fp8_e5m2(&values);
    }
}

pub fn dequantize_row_fp8_e5m2(src: &[BlockFp8E5m2], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len().div_ceil(KV_QUANT_BLOCK_VALUES));
    for (block, chunk) in src.iter().zip(dst.chunks_mut(KV_QUANT_BLOCK_VALUES)) {
        dequantize_block_fp8_e5m2(block, chunk);
    }
}

pub fn vec_dot_row_fp8_e5m2(query: &[f32], key_blocks: &[BlockFp8E5m2]) -> f32 {
    debug_assert_eq!(
        key_blocks.len(),
        query.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    query
        .chunks(KV_QUANT_BLOCK_VALUES)
        .zip(key_blocks)
        .map(|(chunk, block)| vec_dot_fp8_e5m2(chunk, block))
        .sum()
}

pub fn axpy_row_fp8_e5m2(out: &mut [f32], probability: f32, value_blocks: &[BlockFp8E5m2]) {
    debug_assert_eq!(
        value_blocks.len(),
        out.len().div_ceil(KV_QUANT_BLOCK_VALUES)
    );
    for (out_chunk, block) in out.chunks_mut(KV_QUANT_BLOCK_VALUES).zip(value_blocks) {
        let d = super::f16_bits_to_f32(block.scale);
        for (i, out_value) in out_chunk.iter_mut().enumerate() {
            *out_value = (probability * d).mul_add(FP8_E5M2_LUT[block.qs[i] as usize], *out_value);
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

    #[test]
    fn fp8_blocks_have_wire_compatible_sizes() {
        assert_eq!(size_of::<BlockFp8E4m3>(), 34);
        assert_eq!(size_of::<BlockFp8E5m2>(), 34);
    }

    #[test]
    fn fp8_e4m3_roundtrip_and_dot_product() {
        let mut original = [0.0f32; KV_QUANT_BLOCK_VALUES];
        for (i, value) in original.iter_mut().enumerate() {
            *value = (i as f32 - 16.0) * 0.5;
        }
        let block = quantize_block_fp8_e4m3(&original);
        let mut reconstructed = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_fp8_e4m3(&block, &mut reconstructed);

        for i in 0..KV_QUANT_BLOCK_VALUES {
            let diff = (original[i] - reconstructed[i]).abs();
            assert!(
                diff < 0.65,
                "FP8_E4M3 roundtrip diff too high at index {i}: {diff}"
            );
        }
        let query: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 7.0) * 0.125);
        let expected: f32 = query.iter().zip(reconstructed).map(|(q, k)| q * k).sum();
        assert!((vec_dot_fp8_e4m3(&query, &block) - expected).abs() < 1e-4);

        let mut accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| i as f32 * 0.01);
        let expected_accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| 0.375f32.mul_add(reconstructed[i], i as f32 * 0.01));
        axpy_row_fp8_e4m3(&mut accumulated, 0.375, &[block]);
        for (actual, expected) in accumulated.iter().zip(expected_accumulated) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn fp8_e5m2_roundtrip_and_dot_product() {
        let mut original = [0.0f32; KV_QUANT_BLOCK_VALUES];
        for (i, value) in original.iter_mut().enumerate() {
            *value = (i as f32 - 16.0) * 0.5;
        }
        let block = quantize_block_fp8_e5m2(&original);
        let mut reconstructed = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_fp8_e5m2(&block, &mut reconstructed);

        for i in 0..KV_QUANT_BLOCK_VALUES {
            let diff = (original[i] - reconstructed[i]).abs();
            assert!(
                diff < 1.10,
                "FP8_E5M2 roundtrip diff too high at index {i}: {diff}"
            );
        }
        let query: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 7.0) * 0.125);
        let expected: f32 = query.iter().zip(reconstructed).map(|(q, k)| q * k).sum();
        assert!((vec_dot_fp8_e5m2(&query, &block) - expected).abs() < 1e-4);

        let mut accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| i as f32 * 0.01);
        let expected_accumulated: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| 0.375f32.mul_add(reconstructed[i], i as f32 * 0.01));
        axpy_row_fp8_e5m2(&mut accumulated, 0.375, &[block]);
        for (actual, expected) in accumulated.iter().zip(expected_accumulated) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    // ---------------------------------------------------------------------
    // FP8 hardening tests.
    //
    // The pre-existing FP8 dot-product checks above compare `vec_dot_*` against a
    // sum over `dequantize_*` of the SAME block. That is a useful internal-consistency
    // check, but it cannot detect an encoder defect: both sides move together if the
    // encoder is wrong. Everything below compares against either the arithmetic
    // definition of the format or the ORIGINAL pre-quantization values.
    // ---------------------------------------------------------------------

    /// Arithmetic definition of E4M3FN, straight from the format description.
    fn reference_e4m3_to_f32(byte: u8) -> f32 {
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 3) & 0x0F;
        let mant = byte & 0x07;
        if byte & 0x7F == 0x7F {
            return f32::NAN;
        }
        if exp == 0 {
            return sign * (mant as f32) * (1.0 / 512.0);
        }
        sign * (2.0f32).powi(exp as i32 - 7) * (1.0 + (mant as f32) * 0.125)
    }

    /// Arithmetic definition of E5M2.
    fn reference_e5m2_to_f32(byte: u8) -> f32 {
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 2) & 0x1F;
        let mant = byte & 0x03;
        if exp == 31 {
            return if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            };
        }
        if exp == 0 {
            return sign * (mant as f32) * (1.0 / 65536.0);
        }
        sign * (2.0f32).powi(exp as i32 - 15) * (1.0 + (mant as f32) * 0.25)
    }

    /// The decoders are bit-assembly rewrites of a `powi`-based formulation that cost
    /// ~16x the entire Q8_0 dot product in the attention inner loop. Pin every one of
    /// the 256 codes against the arithmetic definition so the rewrite can never drift.
    #[test]
    fn fp8_decoders_match_the_arithmetic_definition_for_all_256_codes() {
        for code in 0..=u8::MAX {
            let (fast, reference) = (fp8_e4m3_to_f32(code), reference_e4m3_to_f32(code));
            if reference.is_nan() {
                assert!(fast.is_nan(), "E4M3 code {code:#04x} should decode to NaN");
            } else {
                assert_eq!(
                    fast.to_bits(),
                    reference.to_bits(),
                    "E4M3 code {code:#04x}: {fast} != {reference}"
                );
            }

            let (fast, reference) = (fp8_e5m2_to_f32(code), reference_e5m2_to_f32(code));
            if reference.is_nan() {
                assert!(fast.is_nan(), "E5M2 code {code:#04x} should decode to NaN");
            } else {
                assert_eq!(
                    fast.to_bits(),
                    reference.to_bits(),
                    "E5M2 code {code:#04x}: {fast} != {reference}"
                );
            }
        }
    }

    /// The hot row kernels index a compile-time table instead of calling the decoder,
    /// so the table has to agree with it on every code. It is generated from the
    /// decoder, which makes this cheap to guarantee and cheap to check.
    #[test]
    fn fp8_lookup_tables_match_their_decoders() {
        for code in 0..=u8::MAX {
            let (table, decoded) = (FP8_E4M3_LUT[code as usize], fp8_e4m3_to_f32(code));
            if decoded.is_nan() {
                assert!(table.is_nan(), "E4M3 LUT[{code:#04x}] should be NaN");
            } else {
                assert_eq!(table.to_bits(), decoded.to_bits(), "E4M3 LUT[{code:#04x}]");
            }

            let (table, decoded) = (FP8_E5M2_LUT[code as usize], fp8_e5m2_to_f32(code));
            if decoded.is_nan() {
                assert!(table.is_nan(), "E5M2 LUT[{code:#04x}] should be NaN");
            } else {
                assert_eq!(table.to_bits(), decoded.to_bits(), "E5M2 LUT[{code:#04x}]");
            }
        }
    }

    /// Spec anchors for the encoders. E4M3FN has NO infinity encoding (unlike E5M2),
    /// so overflow saturates at the largest finite value, 448.
    #[test]
    fn fp8_encoders_hit_their_spec_anchors() {
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1.0)), 1.0);
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(448.0)), 448.0);
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1e9)), 448.0, "saturates");
        assert_eq!(
            fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::INFINITY)),
            448.0,
            "E4M3FN has no infinity encoding, so overflow clamps to max finite"
        );
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(-f32::INFINITY)), -448.0);
        assert!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::NAN)).is_nan());
        // Smallest subnormal step is 2^-9.
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1.0 / 512.0)), 1.0 / 512.0);
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80);

        // E5M2 keeps IEEE semantics at the top of its range.
        assert_eq!(fp8_e5m2_to_f32(f32_to_fp8_e5m2(1.0)), 1.0);
        assert_eq!(fp8_e5m2_to_f32(f32_to_fp8_e5m2(57344.0)), 57344.0);
        assert!(fp8_e5m2_to_f32(f32_to_fp8_e5m2(f32::INFINITY)).is_infinite());
        assert!(fp8_e5m2_to_f32(f32_to_fp8_e5m2(f32::NAN)).is_nan());
        assert_eq!(
            fp8_e5m2_to_f32(f32_to_fp8_e5m2(1.0 / 65536.0)),
            1.0 / 65536.0
        );
    }

    /// Regression: FP8 block scales are stored as f16 but divided by 448 / 57344,
    /// which sits far closer to f16's floor than Q8_0's /127. Without a clamp, an
    /// E5M2 block whose `amax` fell below 1.71e-3 — an entirely ordinary KV magnitude —
    /// produced a scale of exactly 0 and read back as 32 zeros, deleting those
    /// positions from attention with no diagnostic.
    ///
    /// The clamp moves the floor down by roughly five orders of magnitude but cannot
    /// remove it: once the scale itself is pinned at f16's smallest positive value
    /// (2^-24), a block only survives while `amax / 2^-24` still reaches the format's
    /// own smallest subnormal — about 1.2e-10 for E4M3 (2^-33) and 9e-13 for E5M2
    /// (2^-40). This sweep stays inside that supported range; the companion test below
    /// pins the residual floor so it stays documented rather than surprising.
    #[test]
    fn fp8_block_scales_never_underflow_to_zero() {
        let mut magnitude = 1.0f32;
        for _ in 0..6 {
            let src: [f32; KV_QUANT_BLOCK_VALUES] =
                std::array::from_fn(|i| magnitude * (((i % 7) as f32 / 7.0) - 0.5) * 2.0);
            let amax = src.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            assert!(amax > 0.0);

            let mut e4m3 = [0.0f32; KV_QUANT_BLOCK_VALUES];
            dequantize_block_fp8_e4m3(&quantize_block_fp8_e4m3(&src), &mut e4m3);
            assert!(
                e4m3.iter().any(|v| *v != 0.0),
                "E4M3 zeroed an entire block at amax={amax:e}"
            );

            let mut e5m2 = [0.0f32; KV_QUANT_BLOCK_VALUES];
            dequantize_block_fp8_e5m2(&quantize_block_fp8_e5m2(&src), &mut e5m2);
            assert!(
                e5m2.iter().any(|v| *v != 0.0),
                "E5M2 zeroed an entire block at amax={amax:e}"
            );

            magnitude *= 0.02;
        }
    }

    /// Pins where each format's usable range actually ends, and how much the scale
    /// clamp bought. Before the clamp, E4M3 died at `amax < 1.34e-5` and E5M2 at
    /// `amax < 1.71e-3` — both well inside the range of ordinary KV magnitudes.
    #[test]
    fn fp8_dynamic_range_floors_are_where_the_formats_run_out() {
        let probe = |magnitude: f32| {
            let src: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|_| magnitude);
            let mut e4m3 = [0.0f32; KV_QUANT_BLOCK_VALUES];
            dequantize_block_fp8_e4m3(&quantize_block_fp8_e4m3(&src), &mut e4m3);
            let mut e5m2 = [0.0f32; KV_QUANT_BLOCK_VALUES];
            dequantize_block_fp8_e5m2(&quantize_block_fp8_e5m2(&src), &mut e5m2);
            (e4m3[0] != 0.0, e5m2[0] != 0.0)
        };

        // Comfortably inside the supported range for both formats.
        assert_eq!(probe(1.0e-8), (true, true));
        // Past E4M3's floor (~2^-33) but still inside E5M2's much wider exponent range.
        assert_eq!(probe(1.0e-11), (false, true));
        // Below both.
        assert_eq!(probe(1.0e-14), (false, false));

        // Magnitudes that used to be silently destroyed now survive.
        assert_eq!(probe(1.0e-3), (true, true), "E5M2's old cliff was 1.71e-3");
        assert_eq!(probe(1.0e-5), (true, true), "E4M3's old cliff was 1.34e-5");
    }

    /// The reason these formats exist. A block holding one large outlier alongside
    /// ordinary values destroys Q8_0's uniform grid — `amax` is set by the outlier, so
    /// every other element collapses onto a handful of levels — while FP8's exponent
    /// field keeps its relative precision across the whole range.
    #[test]
    fn fp8_e4m3_beats_q8_0_on_outlier_heavy_blocks() {
        let src: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| {
            let base = ((i * 37 % 23) as f32 / 23.0 - 0.5) * 2.0;
            if i == 7 {
                base * 100.0
            } else {
                base
            }
        });

        let rel_err = |out: &[f32; KV_QUANT_BLOCK_VALUES]| -> f64 {
            let (mut se, mut ss) = (0.0f64, 0.0f64);
            for i in 0..KV_QUANT_BLOCK_VALUES {
                let e = (out[i] - src[i]) as f64;
                se += e * e;
                ss += (src[i] as f64) * (src[i] as f64);
            }
            (se / ss).sqrt()
        };

        let mut q8 = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_q8_0(&quantize_block_q8_0(&src), &mut q8);
        let mut fp8 = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_fp8_e4m3(&quantize_block_fp8_e4m3(&src), &mut fp8);

        let (q8_err, fp8_err) = (rel_err(&q8), rel_err(&fp8));
        assert!(
            fp8_err < q8_err,
            "E4M3 should beat Q8_0 on an outlier block: fp8={fp8_err:.5} q8_0={q8_err:.5}"
        );
    }

    /// The honest converse, pinned so nobody mistakes FP8 for a free upgrade: on a
    /// smoothly distributed block at the SAME 34 bytes, Q8_0 spends all 8 bits on the
    /// mantissa and wins comfortably. FP8 is a different trade, not a better one.
    #[test]
    fn q8_0_beats_fp8_on_uniformly_distributed_blocks() {
        let src: [f32; KV_QUANT_BLOCK_VALUES] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.125);

        let worst = |out: &[f32; KV_QUANT_BLOCK_VALUES]| -> f32 {
            (0..KV_QUANT_BLOCK_VALUES).fold(0.0f32, |a, i| a.max((out[i] - src[i]).abs()))
        };

        let mut q8 = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_q8_0(&quantize_block_q8_0(&src), &mut q8);
        let mut fp8 = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_fp8_e4m3(&quantize_block_fp8_e4m3(&src), &mut fp8);

        assert!(
            worst(&q8) < worst(&fp8),
            "Q8_0 should win on uniform data: q8_0={} fp8={}",
            worst(&q8),
            worst(&fp8)
        );
        assert_eq!(
            size_of::<BlockQ8_0>(),
            size_of::<BlockFp8E4m3>(),
            "the comparison is only meaningful at equal size"
        );
    }

    /// Non-circular accuracy check: compare the round trip against the ORIGINAL
    /// values, at a tolerance derived from the format's mantissa width rather than a
    /// hand-picked constant. E4M3 keeps 3 mantissa bits, so a value scaled onto the
    /// top binade carries at most a 1/16 relative step.
    #[test]
    fn fp8_e4m3_roundtrip_error_is_bounded_by_its_mantissa_width() {
        let src: [f32; KV_QUANT_BLOCK_VALUES] =
            std::array::from_fn(|i| ((i * 31 % 17) as f32 / 17.0 - 0.5) * 3.0 + 0.01);
        let mut out = [0.0f32; KV_QUANT_BLOCK_VALUES];
        dequantize_block_fp8_e4m3(&quantize_block_fp8_e4m3(&src), &mut out);

        let amax = src.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for i in 0..KV_QUANT_BLOCK_VALUES {
            let rel = (out[i] - src[i]).abs() / src[i].abs().max(amax / 448.0);
            assert!(
                rel <= 1.0 / 16.0 + 1e-3,
                "index {i}: {} -> {} is a {rel:.4} relative error, past the 3-bit step",
                src[i],
                out[i]
            );
        }
    }

    /// A partial trailing row (head_dim not a multiple of 32) must zero-pad rather
    /// than read past the block or skew the scale.
    #[test]
    fn fp8_handles_rows_that_are_not_a_multiple_of_the_block_size() {
        let row: Vec<f32> = (0..80).map(|i| (i as f32 - 40.0) * 0.05).collect();
        let block_count = row.len().div_ceil(KV_QUANT_BLOCK_VALUES);
        assert_eq!(block_count, 3, "80 values spill into a partial third block");

        let mut blocks = vec![BlockFp8E4m3::default(); block_count];
        quantize_row_fp8_e4m3(&row, &mut blocks);
        let mut out = vec![0.0f32; row.len()];
        dequantize_row_fp8_e4m3(&blocks, &mut out);

        let amax = row.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for (i, (actual, expected)) in out.iter().zip(&row).enumerate() {
            assert!(
                (actual - expected).abs() <= amax / 8.0,
                "index {i}: {expected} -> {actual}"
            );
        }

        // The dot kernel must agree with the dequantized row on the same ragged length.
        let query: Vec<f32> = (0..80).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let expected: f32 = query.iter().zip(&out).map(|(q, k)| q * k).sum();
        let actual = vec_dot_row_fp8_e4m3(&query, &blocks);
        assert!(
            (actual - expected).abs() < 1e-3,
            "ragged vec_dot {actual} != {expected}"
        );
    }
}
