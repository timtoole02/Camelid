//! Dequant-to-f32 for the runnable lane's v1 quant set.
//!
//! Breadth comes from one small dispatch over per-format routines, not a per-format
//! kernel matrix (`RUNNABLE_LANE_SPEC.md`, principle #3). The runnable lane is f32
//! only — no Metal/CUDA fast paths. Each covered quant routes to the crate's already
//! existing, validated block decoder; F16/F32 are handled inline. Correctness is
//! anchored externally against ggml reference fixtures (Gate 2), not trusted from the
//! internal paths it reuses.
//!
//! Covered v1 set: `F32, F16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
//! IQ4_XS, BF16`, plus
//! `NVFP4` (admission-scoped to the gemma4 pilot until Gate G3 — BASALT D-B3).
//! BF16 joined the covered set at BASALT D-B6 (2026-07-17) as an exact-decode type:
//! bf16 is the high 16 bits of f32, so decode is the lossless bit-widening
//! [`crate::tensor::decode_bf16_tensor`] (no new numeric code). Anything else is
//! refused — admission (`super::admit`) should already have rejected it, but dequant
//! fails closed regardless.
//!
//! NVFP4 seam note: admission is metadata-only, so the D17/T5 NaN-sentinel refusal
//! (UE4M3 scale bytes `0x7F`/`0xFF`) cannot happen there — it fires HERE, inside
//! [`decode_nvfp4_tensor`], the fail-closed Phase 1 load path.

use crate::error::{BackendError, Result};
use crate::gguf::GgufTensorType;
use crate::tensor::{
    decode_bf16_tensor, decode_i2_s_tensor, decode_iq4_xs_tensor, decode_nvfp4_tensor,
    decode_q1_0_tensor, decode_q2_0_tensor, decode_q2_k_tensor, decode_q3_k_tensor,
    decode_q4_0_tensor, decode_q4_k_tensor, decode_q5_k_tensor, decode_q6_k_tensor,
    decode_q8_0_tensor, f16_bits_to_f32,
};

/// Dequantize one tensor's wire bytes to a flat row-major `Vec<f32>` of
/// `n_elements` values. `tensor_name` is threaded through only for error messages.
pub fn dequantize(
    tensor_type: GgufTensorType,
    bytes: &[u8],
    n_elements: usize,
    tensor_name: &str,
) -> Result<Vec<f32>> {
    match tensor_type {
        GgufTensorType::F32 => dequantize_f32(bytes, n_elements, tensor_name),
        GgufTensorType::F16 => dequantize_f16(bytes, n_elements, tensor_name),
        GgufTensorType::Q8_0 => decode_q8_0_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q4_0 => decode_q4_0_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q2K => decode_q2_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q3K => decode_q3_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q4K => decode_q4_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q5K => decode_q5_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Q6K => decode_q6_k_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::IQ4XS => decode_iq4_xs_tensor(tensor_name, bytes, n_elements),
        // BASALT D-B6: BF16 is a covered exact-decode quant. bf16 stores the top 16
        // bits of the f32 encoding, so widening (u32::from(u16) << 16) is lossless
        // and bit-deterministic — definitionally identical to the pin's
        // ggml_bf16_to_fp32. Reuses the crate's existing decoder; no new numeric code.
        GgufTensorType::BF16 => decode_bf16_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::I2S => decode_i2_s_tensor(tensor_name, bytes, n_elements),
        // Pin-bitwise NVFP4 decode; refuses NaN-sentinel UE4M3 scale bytes
        // (0x7F/0xFF) per DECISIONS.md D17/T5 — the byte-level half of the
        // admission seam split documented in `super::admit::check_quants`.
        GgufTensorType::NVFP4 => decode_nvfp4_tensor(tensor_name, bytes, n_elements),
        // Sign-only 1-bit decode: element `j` is bit
        // `j % 8` of byte `j / 8` against a shared f16 scale. Order is pinned by
        // `tensor::tests::q1_0_dequant_matches_the_reference_layout`.
        GgufTensorType::Q1_0 => decode_q1_0_tensor(tensor_name, bytes, n_elements),
        GgufTensorType::Ptq1_0 => dequantize_ptq1_0(bytes, n_elements, tensor_name),
        GgufTensorType::Q2_0G64
        | GgufTensorType::Q2_0G128
        | GgufTensorType::Pq2_0 => {
            decode_q2_0_tensor(tensor_name, bytes, n_elements, tensor_type)
        }
        other => Err(BackendError::UnsupportedTensorType(format!(
            "tensor {tensor_name} is {other:?}; runnable dequant covers \
             F32, F16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, BF16, I2_S, NVFP4, Q1_0, Q2_0, PQ2_0"
        ))),
    }
}

/// Prism type 143 uses the TQ1 base-3 code, but with 128-element groups:
/// 16 + 8 interleaved five-trit bytes, then two interleaved four-trit bytes.
/// The scale is at the END of each 28-byte block (unlike PQ2_0).
fn dequantize_ptq1_0(bytes: &[u8], n: usize, name: &str) -> Result<Vec<f32>> {
    if n % 128 != 0 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} PTQ1_0 element count must be divisible by 128"
        )));
    }
    expect_len(bytes.len(), n / 128 * 28, name, "PTQ1_0")?;
    let mut out = Vec::with_capacity(n);
    for block in bytes.chunks_exact(28) {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[26], block[27]]));
        for (start, width, trits) in [(0, 16, 5), (16, 8, 5), (24, 2, 4)] {
            for power in [1u8, 3, 9, 27, 81].iter().take(trits) {
                for &code in &block[start..start + width] {
                    let digit = (u16::from(code.wrapping_mul(*power)) * 3) >> 8;
                    out.push((digit as i16 - 1) as f32 * d);
                }
            }
        }
    }
    Ok(out)
}

fn dequantize_f32(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 4, name, "F32")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn dequantize_f16(bytes: &[u8], n_elements: usize, name: &str) -> Result<Vec<f32>> {
    expect_len(bytes.len(), n_elements * 2, name, "F16")?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn expect_len(actual: usize, expected: usize, name: &str, ty: &str) -> Result<()> {
    if actual != expected {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} {ty} byte length {actual} does not match expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptq1_roundtrips_every_five_trit_code_and_interleaved_tail() {
        for code in 0u16..243 {
            let mut block = [0u8; 28];
            block[26..28].copy_from_slice(&0x4000u16.to_le_bytes()); // scale 2
            let mut expected = Vec::new();
            for (start, width, count) in [(0, 16, 5), (16, 8, 5), (24, 2, 4)] {
                let digits: Vec<Vec<u16>> = (0..width)
                    .map(|i| {
                        let c = (code + i as u16) % 243;
                        (0..count).map(|n| c / 3u16.pow(4 - n as u32) % 3).collect()
                    })
                    .collect();
                for i in 0..width {
                    let c = digits[i].iter().fold(0u16, |a, d| a * 3 + d)
                        * if count == 4 { 3 } else { 1 };
                    block[start + i] = (c * 256).div_ceil(243) as u8;
                }
                for n in 0..count {
                    for digit in &digits {
                        expected.push((digit[n] as f32 - 1.0) * 2.0);
                    }
                }
            }
            assert_eq!(
                dequantize(GgufTensorType::Ptq1_0, &block, 128, "test").unwrap(),
                expected
            );
        }
        assert!(dequantize(GgufTensorType::Ptq1_0, &[0; 27], 128, "bad").is_err());
        assert!(dequantize(GgufTensorType::Ptq1_0, &[0; 28], 127, "bad").is_err());
    }

    #[test]
    fn f32_roundtrips_bit_exact() {
        let vals = [-1.5f32, 0.0, 3.25, f32::MIN_POSITIVE, -0.0];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = dequantize(GgufTensorType::F32, &bytes, vals.len(), "t").unwrap();
        for (a, b) in out.iter().zip(vals.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn f16_known_values() {
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x0000 = 0.0 in IEEE half.
        let bytes = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x00];
        let out = dequantize(GgufTensorType::F16, &bytes, 3, "t").unwrap();
        assert_eq!(out, vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn wrong_length_fails_closed() {
        let err = dequantize(GgufTensorType::F32, &[0u8; 6], 2, "t").unwrap_err();
        assert!(matches!(err, BackendError::InvalidTensorData(_)));
    }

    #[test]
    fn i2_s_dispatch_matches_reference_interleave_and_scale() {
        // 0x1b = codes 0,1,2,3, so each 128-value tile decodes into
        // 32 negative, 32 zero, 32 positive, and 32 zero values.
        let mut bytes = vec![0x1b; 32];
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        bytes.extend_from_slice(&[0; 28]);
        let out = dequantize(GgufTensorType::I2S, &bytes, 128, "blk.0").unwrap();
        assert!(out[0..32].iter().all(|value| *value == -2.0));
        assert!(out[32..64].iter().all(|value| *value == 0.0));
        assert!(out[64..96].iter().all(|value| *value == 2.0));
        assert!(out[96..128].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q2_k_dispatches_to_decoder() {
        let out = dequantize(GgufTensorType::Q2K, &[0u8; 84], 256, "blk.0").unwrap();
        assert_eq!(out, vec![0.0; 256]);
    }

    #[test]
    fn uncovered_quant_refused() {
        let err = dequantize(GgufTensorType::IQ4NL, &[0u8; 18], 32, "blk.0").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }

    #[test]
    fn bf16_dispatches_to_decoder() {
        // BASALT D-B6: the runnable dispatch routes BF16 to the lossless exact-
        // widening decoder. Wire bytes are LE u16: 0x3F80 -> 1.0, 0xC000 -> -2.0,
        // 0x0000 -> +0.0, 0x8000 -> -0.0. Compared on to_bits so -0.0 is distinct.
        let bytes = [0x80, 0x3F, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x80];
        let out = dequantize(GgufTensorType::BF16, &bytes, 4, "blk.0").unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(out[1].to_bits(), (-2.0f32).to_bits());
        assert_eq!(out[2].to_bits(), 0.0f32.to_bits());
        assert_eq!(out[3].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn nvfp4_dispatches_to_decoder() {
        // One all-zero 36-byte block: d[0..4] = 0x00 (a legitimate all-zero scale,
        // NOT a sentinel) → 64 exact zeros. Bit-level parity vs the pin lives in the
        // Phase 1 golden-vector suites; this asserts the dispatch arm routes.
        let out = dequantize(GgufTensorType::NVFP4, &[0u8; 36], 64, "blk.0").unwrap();
        assert_eq!(out.len(), 64);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn nvfp4_nan_sentinel_refused_at_decode() {
        // D17/T5: the NaN-sentinel refusal is a DECODE-time check (admission is
        // metadata-only and never sees wire bytes). 0x7F is the pin CPU's sentinel;
        // 0xFF is the byte the pin's own backends disagree on — both refuse.
        for sentinel in [0x7Fu8, 0xFF] {
            let mut block = [0u8; 36];
            block[0] = sentinel;
            let err = dequantize(GgufTensorType::NVFP4, &block, 64, "blk.0").unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("NaN-sentinel"),
                "sentinel {sentinel:#04x} must refuse with the NaN-sentinel message, got: {msg}"
            );
        }
    }
}
