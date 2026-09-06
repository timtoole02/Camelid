//! Parametric pre-norm decoder, f32 only — the runnable lane's generic graph.
//!
//! One configurable transformer (parameterized from GGUF KV via [`LlamaModelConfig`]):
//! embeddings → N pre-norm blocks (RMSNorm → GQA attention with RoPE → RMSNorm →
//! SwiGLU FFN) → final RMSNorm → logits. Most weights are dequantized to f32
//! ([`super::dequant`]) and run through naive f32 math. BitNet I2_S projections are
//! the deliberate exception: cleanroom CPU and opportunistic Metal/CUDA kernels
//! consume the canonical packed bytes directly. This module ALSO hosts the
//! arch-specific routing that
//! sends qwen35 to its resident Metal (macOS default) or CUDA graph and lfm2 to its
//! Metal engine — those lanes consume packed quantized weights directly, and the f32
//! reference below is their fallback and oracle.
//!
//! Memory: weights stay resident in their compact **quantized** form. The generic
//! reference graph dequantizes one projection at a time, embedding/output projections
//! are handled row-by-row, and BitNet projections remain packed throughout. This
//! avoids expanding a full model to f32.
//!
//! Phase 4 brings this up on **llama** (adjacent-pair RoPE, RMSNorm, SwiGLU, GQA).
//! Architecture-specific switches (qwen3 QK-norm / split-half RoPE, gemma norms +
//! soft-capping, phi3 fused QKV) land in Phase 6.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{BackendError, Result};
use crate::gguf::{read_metadata, GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::inference::{LlamaSampler, SamplingConfig};
use crate::model::LlamaModelConfig;
use crate::tensor::CpuTensor;

use super::admit;

/// A 2-D weight kept in its quantized wire form. ggml layout: `ne = [in, out]`,
/// row-major with out feature `r` occupying one contiguous row of `in` values.
#[derive(Clone)]
enum RawMatBytes {
    /// Ordinary portable backing. Arc makes tied token/output embeddings share
    /// their bytes instead of duplicating the largest matrix in the model.
    Owned(Arc<Vec<u8>>),
    /// Page-aligned packed backing. Metal wraps this allocation with
    /// `newBufferWithBytesNoCopy`, so a Q1/Q2 or Q8_0 projection has one resident copy.
    WirePages(Arc<crate::wire_mmap::WirePages>),
}

impl RawMatBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes.as_slice(),
            Self::WirePages(pages) => pages.bytes(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn owned(bytes: Vec<u8>) -> Self {
        Self::Owned(Arc::new(bytes))
    }

    fn wire_pages(&self) -> Option<&Arc<crate::wire_mmap::WirePages>> {
        match self {
            Self::WirePages(pages) => Some(pages),
            Self::Owned(_) => None,
        }
    }
}

/// The resident Metal format a GGUF tensor type maps to, or `None` if the resident lane
/// cannot consume it. Single source of truth *for the loader*: admitting a type here is
/// what makes it page-backed at load (see [`wants_page_backing`]), so those two cannot
/// drift apart.
///
/// Not macOS-gated, because `wants_page_backing` is not: the loader makes the same
/// backing choice on every target.
///
/// **It is not the only gate on the way to the GPU, and the others do NOT fail loudly.**
/// Admitting a type here has two silent knock-on effects, both of which must be decided
/// deliberately:
///
/// 1. It routes the type into the *hybrid* Prism wire path from
///    [`RawMat::par_matvec`]/[`RawMat::par_matmul`], which keeps its own separate list —
///    `metal::ResidentWeightFormat::hybrid_prism_wire_supported`. `Q8_0` and the K-quants
///    are admitted here and declined there on purpose, so they fall through to the CPU
///    kernel instead of taking a second GPU path with none of the resident lane's parity
///    evidence. That list is exhaustive over `ResidentWeightFormat`, so a brand-new
///    *format variant* is a compile error there — but admitting an existing variant here
///    is **not**, and `prism_wire_hybrid_admission_is_pinned_per_format` stays green.
/// 2. It does NOT update `execution_plan.rs`. The plan picks its arm from tensor types
///    and arch, so a newly-admitted quant will keep disclosing whatever arm it matched
///    before — `/v1/health` naming a lane other than the one serving. That has now
///    happened four times (Prism Q1/Q2, lfm2, qwen35 Q8_0, qwen35 K-quant); the last one
///    was caught only because a live load was checked.
fn resident_metal_format(tt: GgufTensorType) -> Option<crate::metal::ResidentWeightFormat> {
    match tt {
        GgufTensorType::Q1_0 => Some(crate::metal::ResidentWeightFormat::Q1_0),
        GgufTensorType::Q2_0G64 => Some(crate::metal::ResidentWeightFormat::Q2_0G64),
        GgufTensorType::Q2_0G128 | GgufTensorType::Pq2_0 => {
            Some(crate::metal::ResidentWeightFormat::Q2_0G128)
        }
        // Q8_0 wire blocks (34B: f16 scale + 32 i8) are already the layout the resident
        // Metal Q8 GEMV consumes, so they need no repack.
        GgufTensorType::Q8_0 => Some(crate::metal::ResidentWeightFormat::Q8_0),
        // K-quant super-blocks (Q4_K 144B / Q6_K 210B per 256 values) are likewise the
        // exact layout `encode_resident_kquant_matmul_f32` consumes. Admitted as a PAIR:
        // an ornith Q4_K_M file carries Q6_K on `output.weight`, 12 `attn_qkv`, 4
        // `attn_v` and 16 `ffn_down`, and `prism_metal_weight` hard-errors on an
        // unmapped type, so admitting only Q4K yields no resident graph at all.
        GgufTensorType::Q4K => Some(crate::metal::ResidentWeightFormat::Q4K),
        // Q5_K (176B per 256 values) is Q4_K plus a `qh` high-bit plane. Admitted
        // alongside Q6_K for the same PAIR reason: a Q5_K_M/Q5_K_L mix carries Q6_K
        // on some `attn_qkv`/`attn_v`/`ffn_down` and Q8_0 on the embed/output heads,
        // so admitting Q5_K alone would still yield no resident graph.
        GgufTensorType::Q5K => Some(crate::metal::ResidentWeightFormat::Q5K),
        GgufTensorType::Q6K => Some(crate::metal::ResidentWeightFormat::Q6K),
        // bf16 survives some quant recipes on the tiny `ssm_alpha`/`ssm_beta`
        // projections (48 tensors, ~6M params). Widening to f32 is a bit shift, so
        // the dense bf16 kernel reads the wire bytes in place — no conversion pass
        // and no second copy. NOT mapped to DenseF16: different exponent width.
        GgufTensorType::BF16 => Some(crate::metal::ResidentWeightFormat::DenseBF16),
        _ => None,
    }
}

/// Formats accepted by the small hybrid projection bridge. Dense F16 is not a
/// general resident-model admission: it is used only for BitNet's explicitly
/// page-backed tied output head (see `load_raw`).
#[cfg(target_os = "macos")]
fn hybrid_metal_format(tt: GgufTensorType) -> Option<crate::metal::ResidentWeightFormat> {
    match tt {
        GgufTensorType::F16 => Some(crate::metal::ResidentWeightFormat::DenseF16),
        _ => resident_metal_format(tt),
    }
}

/// Tensor types read into `RawMatBytes::WirePages` rather than an owned `Vec`, so the
/// allocation can be wrapped in place with `newBufferWithBytesNoCopy`.
///
/// Derived from [`resident_metal_format`] rather than listed again: `prism_metal_weight`
/// needs a tensor to be *both* page-backed and format-mapped, and when those were two
/// hand-maintained lists, adding a type to only one left it silently failing the
/// page-backed check and falling back to CPU decode.
///
/// Page-backing is otherwise unobservable — `as_slice` is identical either way and
/// `wire_pages` is macOS-only — at the cost of rounding each tensor up to a page.
fn wants_page_backing(tt: GgufTensorType) -> bool {
    tt == GgufTensorType::I2S || resident_metal_format(tt).is_some()
}

/// The ReLU²/SubLN graph below is certified for Microsoft's exact 2B-4T
/// checkpoint, not for every historical model that reused the
/// `bitnet-b1.58` architecture label. Fail closed before loading weights so an
/// older SiLU checkpoint cannot be executed with a plausible-but-wrong graph.
fn validate_bitnet_b158_2b_4t(gguf: &GgufFile, config: &LlamaModelConfig) -> Result<()> {
    if config.architecture != "bitnet-b1.58" {
        return Ok(());
    }

    let head_dim = config
        .attention_key_length
        .or_else(|| {
            config
                .embedding_length
                .checked_div(config.attention_head_count)
        })
        .unwrap_or(0);
    let geometry_matches = gguf.model_name() == Some("bitnet2b")
        && config.context_length == 4_096
        && config.embedding_length == 2_560
        && config.block_count == 30
        && config.feed_forward_length == 6_912
        && config.attention_head_count == 20
        && config.attention_head_count_kv == 5
        && head_dim == 128
        && config.rope_dimension_count == Some(128)
        && config.rope_freq_base == Some(500_000.0)
        && config.vocab_size == Some(128_256)
        && config.file_type == Some(40);

    let projection_suffixes = [
        ".attn_q.weight",
        ".attn_k.weight",
        ".attn_v.weight",
        ".attn_output.weight",
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".ffn_down.weight",
    ];
    let projections: Vec<_> = gguf
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("blk.")
                && projection_suffixes
                    .iter()
                    .any(|suffix| tensor.name.ends_with(suffix))
        })
        .collect();
    let tensors_match = projections.len() == 30 * projection_suffixes.len()
        && projections
            .iter()
            .all(|tensor| tensor.tensor_type == GgufTensorType::I2S)
        && gguf.tensors.iter().any(|tensor| {
            tensor.name == "token_embd.weight" && tensor.tensor_type == GgufTensorType::F16
        })
        && !gguf
            .tensors
            .iter()
            .any(|tensor| tensor.name == "output.weight");

    if !geometry_matches || !tensors_match {
        return Err(BackendError::InvalidModelMetadata(format!(
            "bitnet-b1.58 runnable support is pinned to Microsoft's BitNet-b1.58-2B-4T I2_S artifact; got model {:?}, geometry ({}, {}, {}, {}, {}, {}, head_dim {}, rope {:?}/{:?}, vocab {:?}, file_type {:?}) and {} canonical projections",
            gguf.model_name().unwrap_or("unknown"),
            config.context_length,
            config.embedding_length,
            config.block_count,
            config.feed_forward_length,
            config.attention_head_count,
            config.attention_head_count_kv,
            head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            config.vocab_size,
            config.file_type,
            projections.len()
        )));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn log_prism_metal_hybrid_once() {
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!(
            "[runnable] Metal hybrid bring-up: packed/ternary projections are NoCopy GPU \
             kernels; recurrent/attention state is still CPU-resident"
        );
    });
}

struct RawMat {
    bytes: RawMatBytes,
    tt: GgufTensorType,
    in_features: usize,
    out_features: usize,
}

/// A dequantized 2-D weight (f32), produced transiently from a [`RawMat`].
struct Mat {
    data: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Mat {
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_features);
        let mut output = vec![0.0_f32; self.out_features];
        for (row, value) in output.iter_mut().enumerate() {
            *value = dot(
                &self.data[row * self.in_features..(row + 1) * self.in_features],
                x,
            );
        }
        output
    }
}

impl RawMat {
    fn row_bytes(&self) -> usize {
        self.bytes.len() / self.out_features
    }

    #[cfg(target_os = "macos")]
    fn prism_metal_weight(&self) -> Result<crate::metal::ResidentWeightBytes<'_>> {
        let pages = self.bytes.wire_pages().ok_or_else(|| {
            BackendError::InvalidTensorData("Prism Metal weight is not page-backed".into())
        })?;
        let format = resident_metal_format(self.tt).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "unsupported Qwen3.5 Metal projection type {:?}",
                self.tt
            ))
        })?;
        Ok(crate::metal::ResidentWeightBytes::WirePages { format, pages })
    }

    /// Q8_0 weight bytes for the LFM2 Metal engine.
    ///
    /// Since Q8_0 joined `resident_metal_format`, `prism_metal_weight` returns the
    /// identical value for a page-backed Q8_0 tensor. This helper stays separate for
    /// the two ways it behaves differently: it hard-requires Q8_0 (LFM2 ships every
    /// projection as Q8_0, so any other type is a load error, not a fallback), and it
    /// tolerates non-page-backed bytes — page-backed allocations are handed over in
    /// place (one resident copy, no upload), while `Owned` bytes go through the
    /// resident cache as ordinary wire bytes where `prism_metal_weight` fails closed.
    /// `resolve_resident_weight` already accepts `Q8_0` on both arms.
    #[cfg(target_os = "macos")]
    fn q8_metal_weight(&self) -> Result<crate::metal::ResidentWeightBytes<'_>> {
        if self.tt != GgufTensorType::Q8_0 {
            return Err(BackendError::UnsupportedGguf(format!(
                "lfm2 Metal lane requires Q8_0 projections, got {:?}",
                self.tt
            )));
        }
        if let Some(pages) = self.bytes.wire_pages() {
            return Ok(crate::metal::ResidentWeightBytes::WirePages {
                format: crate::metal::ResidentWeightFormat::Q8_0,
                pages,
            });
        }
        Ok(crate::metal::ResidentWeightBytes::KQuantBytes {
            format: crate::metal::ResidentWeightFormat::Q8_0,
            bytes: self.bytes.as_slice(),
        })
    }

    fn dequant_all(&self, name: &str) -> Result<Mat> {
        Ok(Mat {
            data: super::dequant::dequantize(
                self.tt,
                self.bytes.as_slice(),
                self.in_features * self.out_features,
                name,
            )?,
            in_features: self.in_features,
            out_features: self.out_features,
        })
    }

    /// Preserve the generic graph's f32 reference arithmetic while allowing
    /// canonical I2_S projections to stay packed and use their cleanroom lane.
    fn projection_matvec(&self, input: &[f32], name: &str) -> Result<Vec<f32>> {
        if self.tt == GgufTensorType::I2S {
            self.par_matvec(input, name)
        } else {
            Ok(self.dequant_all(name)?.matvec(input))
        }
    }

    fn projection_matmul(&self, inputs: &[Vec<f32>], name: &str) -> Result<Vec<Vec<f32>>> {
        if self.tt == GgufTensorType::I2S {
            self.par_matmul(inputs)
        } else {
            let matrix = self.dequant_all(name)?;
            Ok(inputs.iter().map(|input| matrix.matvec(input)).collect())
        }
    }

    /// Dequantize a single row `r` (length `in_features`) — for embedding lookup and
    /// the output projection, which touch the huge vocab matrix one row at a time.
    fn dequant_row(&self, r: usize, name: &str) -> Result<Vec<f32>> {
        if self.tt == GgufTensorType::I2S {
            return Err(BackendError::InvalidTensorData(format!(
                "I2_S tensor {name} has a tensor-wide scale trailer and cannot be decoded row-wise"
            )));
        }
        let rb = self.row_bytes();
        let slice = &self.bytes.as_slice()[r * rb..(r + 1) * rb];
        super::dequant::dequantize(self.tt, slice, self.in_features, name)
    }

    /// Carve out a contiguous block of `len` out-features starting at `start` into a
    /// new RawMat. Used to split phi3's fused `attn_qkv` and fused `gate_up` into the
    /// separate projections the generic block expects. Valid because rows are
    /// out-feature-major and each row is a whole number of quant blocks.
    fn split_rows(&self, start: usize, len: usize) -> RawMat {
        let rb = self.row_bytes();
        RawMat {
            bytes: RawMatBytes::owned(
                self.bytes.as_slice()[start * rb..(start + len) * rb].to_vec(),
            ),
            tt: self.tt,
            in_features: self.in_features,
            out_features: len,
        }
    }

    /// Row-parallel matvec: `y[r] = dot(dequant_row(r), x)`, computed across rows
    /// with rayon. **Bit-identical** to `dequant_all(name)?.matvec(x)` — each row's
    /// dot product is sequential (sum order unchanged) and only the independent rows
    /// run in parallel — but ~Nx faster and lower peak memory (no whole-matrix f32
    /// allocation; each row is dequantized, dotted, and dropped). Used by the qwen35
    /// path so the agent loop runs at usable speed without perturbing parity. Q8_0
    /// rows are a whole number of quant blocks, so a per-row dequant equals the
    /// corresponding slice of a whole-matrix dequant.
    fn par_matvec(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        debug_assert_eq!(x.len(), self.in_features);
        if self.tt == GgufTensorType::I2S {
            let mode = crate::bitnet_kernels::BitNetKernelMode::from_env();
            let mut output = vec![0.0_f32; self.out_features];
            if crate::bitnet_kernels::gpu_allowed()
                && crate::cuda::gpu_accel_enabled()
                && crate::cuda::try_bitnet_i2_s_linear_rows(
                    x,
                    self.bytes.as_slice(),
                    1,
                    self.out_features,
                    self.in_features,
                    mode.gpu_code(),
                    &mut output,
                )
            {
                return Ok(output);
            }
            #[cfg(target_os = "macos")]
            if crate::bitnet_kernels::gpu_allowed() && crate::cuda::gpu_accel_enabled() {
                if let Some(pages) = self.bytes.wire_pages() {
                    if let Some(output) = crate::metal::try_bitnet_i2_s_matvec_f32(
                        x,
                        pages,
                        self.out_features,
                        mode.gpu_code(),
                    ) {
                        return Ok(output);
                    }
                }
            }
            return crate::bitnet_kernels::i2_s_matvec(
                self.bytes.as_slice(),
                x,
                self.out_features,
                mode,
            );
        }
        #[cfg(target_os = "macos")]
        if let Some(pages) = self.bytes.wire_pages() {
            if let Some(output) = hybrid_metal_format(self.tt).and_then(|format| {
                crate::metal::try_prism_wire_matvec_f32(x, pages, format, self.out_features)
            }) {
                log_prism_metal_hybrid_once();
                return Ok(output);
            }
        }
        let rb = self.row_bytes();
        match self.tt {
            // Fused, allocation-free dot for the two formats this model uses (Q8_0
            // weights + F32 norms-as-matrices never reach here, but F32 rows can).
            // Bit-identical to `dequant_row(r)` + `dot`: each element is the same
            // `scale*(q as f32)` (Q8_0) / `from_le_bytes` (F32) and the f32
            // accumulation order is unchanged — only the per-row Vec alloc is gone.
            GgufTensorType::Q8_0 => {
                // Quantize the f32 activation to Q8 ONCE and reuse it across every
                // weight row, so each row is an integer maddubs reduction (int8×int8)
                // rather than i8→f32 + f32-FMA. The quantize is O(in); the matmul it
                // feeds is O(out·in), so the cost is negligible.
                let xq = crate::inference::quantize_q8_0_blocks(x);
                Ok((0..self.out_features)
                    .into_par_iter()
                    .map(|r| q8_0_wire_dot(&self.bytes.as_slice()[r * rb..(r + 1) * rb], &xq))
                    .collect())
            }
            GgufTensorType::F32 => Ok((0..self.out_features)
                .into_par_iter()
                .map(|r| f32_row_dot(&self.bytes.as_slice()[r * rb..(r + 1) * rb], x))
                .collect()),
            GgufTensorType::F16 => Ok((0..self.out_features)
                .into_par_iter()
                .map(|r| f16_row_dot(&self.bytes.as_slice()[r * rb..(r + 1) * rb], x))
                .collect()),
            GgufTensorType::BF16 => Ok((0..self.out_features)
                .into_par_iter()
                .map(|r| bf16_row_dot(&self.bytes.as_slice()[r * rb..(r + 1) * rb], x))
                .collect()),
            _ => (0..self.out_features)
                .into_par_iter()
                .map(|r| Ok(dot(&self.dequant_row(r, name)?, x)))
                .collect(),
        }
    }
}

/// Fused Q8_0-row · Q8-quantized-activation dot — the int8×int8 kernel the optimized
/// inference lane uses. The caller quantizes the f32 activation **once** per matvec
/// (`quantize_q8_0_blocks`) and reuses the blocks across every weight row; each row is
/// then an integer maddubs reduction (i8×i8 → i16 → i32) instead of the prior
/// i8→f32-convert + f32-FMA. On x86_64+AVX2 this dispatches to a vectorized maddubs
/// kernel byte-for-byte equal to the optimized lane's `q8_0_dot_rows_avx2`; otherwise
/// the shared scalar/NEON reference (`crate::inference::q8_0_wire_row_dot`). Quantizing
/// the activation makes this numerically *closer* to llama.cpp's own q8×q8 path (which
/// also quantizes activations) than the prior f32-activation dot was — parity stays
/// greedy-token (argmax) and is re-certified by `ornith_qwen35_parity_gen`. `row` is a
/// whole number of 34-byte Q8_0 blocks (f16 scale + 32 i8); `xq` holds the matching
/// block count (`x.len() / 32`).
fn q8_0_wire_dot(row: &[u8], xq: &[crate::tensor::Q8_0Block]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime AVX2 feature check above.
            return unsafe { q8_0_wire_dot_avx2(row, xq) };
        }
    }
    crate::inference::q8_0_wire_row_dot(row, xq)
}

/// AVX2 int8×int8 maddubs dot of a wire-format Q8_0 weight row against quantized
/// activation blocks. Mirrors the optimized lane's `q8_0_dot_rows_avx2` exactly — same
/// `i8::MIN` overflow guard (maddubs' first operand is unsigned, so `i8::MIN` would
/// wrap), same sign trick, same in-register horizontal sum — but loads the 32 weight
/// i8 straight from the wire bytes (contiguous at `base + 2`, after the f16 scale)
/// rather than a decoded `Q8_0Block`, so no resident weight decode is needed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_wire_dot_avx2(row: &[u8], input: &[crate::tensor::Q8_0Block]) -> f32 {
    use std::arch::x86_64::*;
    const WIRE: usize = 34;
    let ones = _mm256_set1_epi16(1);
    let min_i8 = _mm256_set1_epi8(i8::MIN);
    let rptr = row.as_ptr();
    let mut total_sum = 0.0_f32;
    for (b, i_block) in input.iter().enumerate() {
        let base = b * WIRE;
        let scale = crate::tensor::f16_bits_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
        let wptr = rptr.add(base + 2);
        let weight_i8 = _mm256_loadu_si256(wptr.cast());
        let input_i8 = _mm256_loadu_si256(i_block.quants.as_ptr().cast());

        let has_min_i8 = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(weight_i8, min_i8))
            | _mm256_movemask_epi8(_mm256_cmpeq_epi8(input_i8, min_i8)))
            != 0;

        let acc = if has_min_i8 {
            // i8::MIN can't be the |weight| operand of maddubs (it's unsigned); widen.
            let mut acc = _mm256_setzero_si256();
            for offset in [0usize, 16] {
                let weight_half = _mm_loadu_si128(wptr.add(offset).cast());
                let input_half = _mm_loadu_si128(i_block.quants.as_ptr().add(offset).cast());
                let products = _mm256_mullo_epi16(
                    _mm256_cvtepi8_epi16(weight_half),
                    _mm256_cvtepi8_epi16(input_half),
                );
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(products, ones));
            }
            acc
        } else {
            let abs_weight = _mm256_sign_epi8(weight_i8, weight_i8);
            let signed_input = _mm256_sign_epi8(input_i8, weight_i8);
            _mm256_madd_epi16(_mm256_maddubs_epi16(abs_weight, signed_input), ones)
        };

        let sum128 = _mm_add_epi32(
            _mm256_castsi256_si128(acc),
            _mm256_extracti128_si256(acc, 1),
        );
        let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
        let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0xB1));
        let block_sum = _mm_cvtsi128_si32(sum32);
        total_sum += block_sum as f32 * scale * i_block.scale;
    }
    total_sum
}

/// Fused F32-row · f32 dot (no intermediate Vec). Bit-identical to
/// `dequantize_f32` + `dot`.
fn f32_row_dot(row: &[u8], x: &[f32]) -> f32 {
    row.chunks_exact(4)
        .zip(x.iter())
        .map(|(c, &xi)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xi)
        .sum()
}

/// Fused F16-row · f32 dot (no 2,560-element allocation per vocabulary row).
/// Element conversion and accumulation order match `dequantize_f16` + `dot`.
fn f16_row_dot(row: &[u8], x: &[f32]) -> f32 {
    row.chunks_exact(2)
        .zip(x.iter())
        .map(|(c, &xi)| crate::tensor::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) * xi)
        .sum()
}

/// Fused BF16-row · f32 dot. BF16 is exactly the high 16 bits of an f32.
fn bf16_row_dot(row: &[u8], x: &[f32]) -> f32 {
    row.chunks_exact(2)
        .zip(x.iter())
        .map(|(c, &xi)| {
            let bits = u16::from_le_bytes([c[0], c[1]]) as u32;
            f32::from_bits(bits << 16) * xi
        })
        .sum()
}

#[cfg(test)]
mod dense_row_dot_tests {
    use super::*;

    #[test]
    fn fused_f16_and_bf16_rows_match_dequantized_dot() {
        let x = [0.25_f32, -2.0, 3.5, 0.0];
        let f16_bits = [0x3c00_u16, 0xc000, 0x3800, 0x7bff];
        let f16_bytes = f16_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let f16_values = f16_bits
            .iter()
            .map(|&bits| crate::tensor::f16_bits_to_f32(bits))
            .collect::<Vec<_>>();
        assert_eq!(f16_row_dot(&f16_bytes, &x), dot(&f16_values, &x));

        let bf16_bits = [0x3f80_u16, 0xc000, 0x3f00, 0x7f7f];
        let bf16_bytes = bf16_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let bf16_values = bf16_bits
            .iter()
            .map(|&bits| f32::from_bits((bits as u32) << 16))
            .collect::<Vec<_>>();
        assert_eq!(bf16_row_dot(&bf16_bytes, &x), dot(&bf16_values, &x));
    }
}

impl RawMat {
    /// Batched [`par_matvec`]: one output vector per input in `xs`, reading each
    /// weight row ONCE and dotting it against every input — so the resident weights
    /// are streamed once for the whole batch instead of once per input. A 9B forward
    /// reads ~9 GB of weights, which dominates per token; amortizing that read across
    /// all prompt positions is what makes prompt prefill fast. Bit-identical to
    /// calling `par_matvec` on each input separately (same per-element arithmetic and
    /// accumulation order — the row dot is unchanged; only the batching differs).
    fn par_matmul(&self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let m = xs.len();
        let out_f = self.out_features;
        if self.tt == GgufTensorType::I2S {
            let mode = crate::bitnet_kernels::BitNetKernelMode::from_env();
            let input_width = xs.first().map(Vec::len).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "I2_S matmul requires at least one input row".into(),
                )
            })?;
            if input_width != self.in_features || xs.iter().any(|input| input.len() != input_width)
            {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "I2_S matmul expected input rows of width {}, got {:?}",
                    self.in_features,
                    xs.iter().map(Vec::len).collect::<Vec<_>>()
                )));
            }
            let flat_input = xs.iter().flatten().copied().collect::<Vec<_>>();
            let mut flat_output = vec![0.0_f32; m * out_f];
            if crate::bitnet_kernels::gpu_allowed()
                && crate::cuda::gpu_accel_enabled()
                && crate::cuda::try_bitnet_i2_s_linear_rows(
                    &flat_input,
                    self.bytes.as_slice(),
                    m,
                    out_f,
                    input_width,
                    mode.gpu_code(),
                    &mut flat_output,
                )
            {
                return Ok(flat_output
                    .chunks_exact(out_f)
                    .map(<[f32]>::to_vec)
                    .collect());
            }
            #[cfg(target_os = "macos")]
            if crate::bitnet_kernels::gpu_allowed() && crate::cuda::gpu_accel_enabled() {
                if let Some(pages) = self.bytes.wire_pages() {
                    if let Some(output) =
                        crate::metal::try_bitnet_i2_s_matmul_f32(xs, pages, out_f, mode.gpu_code())
                    {
                        return Ok(output);
                    }
                }
            }
            return crate::bitnet_kernels::i2_s_matmul(self.bytes.as_slice(), xs, out_f, mode);
        }
        #[cfg(target_os = "macos")]
        if let Some(pages) = self.bytes.wire_pages() {
            if let Some(output) = hybrid_metal_format(self.tt).and_then(|format| {
                crate::metal::try_prism_wire_matmul_f32(xs, pages, format, self.out_features)
            }) {
                log_prism_metal_hybrid_once();
                return Ok(output);
            }
        }
        let rb = self.row_bytes();
        // flat[r*m + p] = dot(row_r, xs[p]); par over rows so each row is read once.
        let flat: Vec<f32> = match self.tt {
            GgufTensorType::Q8_0 => {
                // Quantize each batched activation to Q8 ONCE (outside the per-row
                // parallel loop), then every row reads the resident weight once and
                // int8×int8-dots it against all positions — weights streamed once for
                // the whole batch, activation quantization amortized across all rows.
                let xqs: Vec<Vec<crate::tensor::Q8_0Block>> = xs
                    .iter()
                    .map(|x| crate::inference::quantize_q8_0_blocks(x))
                    .collect();
                (0..out_f)
                    .into_par_iter()
                    .flat_map_iter(|r| {
                        let row = &self.bytes.as_slice()[r * rb..(r + 1) * rb];
                        xqs.iter().map(move |xq| q8_0_wire_dot(row, xq))
                    })
                    .collect()
            }
            GgufTensorType::F32 => (0..out_f)
                .into_par_iter()
                .flat_map_iter(|r| {
                    let row = &self.bytes.as_slice()[r * rb..(r + 1) * rb];
                    xs.iter().map(move |x| f32_row_dot(row, x))
                })
                .collect(),
            _ => {
                // Fallback (never hit for the Q8_0+F32 qwen35 model): per-input matvec.
                let mut out = Vec::with_capacity(m);
                for x in xs {
                    out.push(self.par_matvec(x, "matmul")?);
                }
                return Ok(out);
            }
        };
        // Transpose flat[r*m + p] -> out[p][r].
        let mut out = vec![vec![0.0f32; out_f]; m];
        for r in 0..out_f {
            let base = r * m;
            for (p, op) in out.iter_mut().enumerate() {
                op[r] = flat[base + p];
            }
        }
        Ok(out)
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    wq: RawMat,
    wk: RawMat,
    wv: RawMat,
    wo: RawMat,
    gate: RawMat,
    up: RawMat,
    down: RawMat,
    /// Per-head QK-norm weights (qwen3, gemma3): RMSNorm over each head's `head_dim`
    /// vector, applied to Q/K after projection and before RoPE. `None` for llama-family.
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    /// BitNet embedding checkpoints normalize each projection input separately.
    /// These seven tensors are an all-or-nothing graph contract for the official
    /// qwen3/gemma3 embedding GGUFs.
    q_norm_in: Option<Vec<f32>>,
    k_norm_in: Option<Vec<f32>>,
    v_norm_in: Option<Vec<f32>>,
    output_norm_in: Option<Vec<f32>>,
    gate_norm_in: Option<Vec<f32>>,
    up_norm_in: Option<Vec<f32>>,
    down_norm_in: Option<Vec<f32>>,
    /// BitNet-b1.58 applies SubLN to the attention value mix and gated FFN
    /// activation before their output projections.
    attn_sub_norm: Option<Vec<f32>>,
    ffn_sub_norm: Option<Vec<f32>>,
    /// gemma 4-norm structure: an extra RMSNorm applied to the attention output and to
    /// the FFN output BEFORE each residual add. `None` for the llama 2-norm structure.
    post_attn_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
}

/// Per-layer K/V cache for incremental decode. Each `k`/`v` grows by `kv_dim` per
/// position. Lets `generate` compute only the new position each step instead of
/// recomputing the whole sequence — O(seq) matmuls total instead of O(seq²).
struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// LFM2 short-conv rolling state, per layer: `(l_cache-1) * d_model`,
    /// layout `[c*(l_cache-1) + t]` with `t=0` oldest. Empty for every layer
    /// of every other architecture, and for LFM2's own attention layers —
    /// sized by [`RunnableModel::new_cache`].
    conv: Vec<Vec<f32>>,
}

impl KvCache {
    fn new(n_layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
            conv: vec![Vec::new(); n_layers],
        }
    }
}

/// A loaded runnable model: parametric config + quantized weights, ready for greedy
/// decode. Weights are dequantized to f32 on demand during the forward pass.
pub struct RunnableModel {
    pub architecture: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// Position-frequency multiplier used by linear RoPE scaling. GGUF stores
    /// the expansion factor (for example 8); llama.cpp applies its reciprocal
    /// to the angle, so an unscaled model carries 1 here.
    rope_freq_scale: f32,
    pub eps: f32,
    pub vocab: usize,
    rope_neox: bool,
    /// Per-layer RoPE base. Uniform for most models; gemma3 alternates a local base
    /// (10000) on sliding-window layers and a global base (1e6) every Nth layer.
    layer_rope_base: Vec<f32>,
    /// gemma scales token embeddings by `sqrt(d_model)`. `None` for non-gemma.
    embed_scale: Option<f32>,
    /// gemma FFN uses GeGLU (gelu-tanh) instead of llama's SwiGLU (silu).
    ffn_gelu: bool,
    /// gemma2 logit soft-caps; gemma3 has neither. `cap * tanh(x / cap)`.
    final_logit_softcap: Option<f32>,
    attn_logit_softcap: Option<f32>,
    /// Logit scale — applied before softmax, commonly in Command R models.
    logit_scale: Option<f32>,
    token_embd: RawMat, // [in=d_model, out=vocab]; row = token embedding
    output: RawMat,     // logits projection; tied models reuse token_embd
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
    /// Qwen3.5 (Ornith) hybrid gated-delta-net runtime. `Some` only for the
    /// `qwen35` architecture, whose layers do not fit the generic `Layer` (SSM
    /// layers have no K/V attention). When set, the forward path is routed to the
    /// dedicated `*_qwen35` methods and `layers` is empty. See [`Qwen35Runtime`].
    qwen35: Option<Qwen35Runtime>,
    /// LFM2 / LFM2.5 hybrid short-convolution runtime. `Some` only for the
    /// `lfm2` architecture, whose conv layers do not fit the generic `Layer`
    /// (they carry `shortconv.*` and no K/V attention). When set, the forward
    /// path is routed to the dedicated `*_lfm2` methods and `layers` is empty.
    /// See [`Lfm2Runtime`].
    lfm2: Option<Lfm2Runtime>,
    /// Exact artifact identity used to admit hash-pinned CUDA specializations.
    /// Geometry alone is not sufficient because unrelated qwen35 files can share it.
    #[cfg(feature = "cuda")]
    resident_cuda_artifact: crate::cuda_resident::ResidentCudaArtifact,
    /// Lazily-built GPU resident decode engine for the qwen35 lane, gated by
    /// `CAMELID_QWEN35_CUDA=1`. `Mutex` gives the `&mut` the per-token forward needs
    /// while `generate_*` take `&self`; built on first use and reused, with the SSM/conv
    /// recurrent state reset at the start of every generate. `None` until first use.
    #[cfg(feature = "cuda")]
    cuda: std::sync::Mutex<Option<crate::cuda_resident::CudaResidentDecode>>,
    /// Fully resident Apple Metal Qwen3.5 hybrid graph. Built lazily so merely
    /// inspecting/loading a model does not allocate recurrent/KV state.
    #[cfg(target_os = "macos")]
    metal_qwen35: std::sync::Mutex<Option<crate::metal::Qwen35MetalDecode>>,
    /// Resident LFM2 Metal graph, built on first use and reused. `Mutex` supplies the
    /// `&mut` a per-token forward needs while `generate_*` take `&self`.
    #[cfg(target_os = "macos")]
    metal_lfm2: std::sync::Mutex<Option<crate::metal::Lfm2MetalDecode>>,
}

/// Keep the first Command R bring-up deliberately narrower than architecture
/// admission. The immutable Aya Expanse 8B Q4_K_M row is small enough to qualify
/// on a normal workstation and has a graph fully described by the pinned
/// llama.cpp source. Neighboring Command R variants differ structurally (notably
/// the >=64-layer Q/K-norm layout), so they fail closed until separately proven.
/// Validate the complete immutable header contract for the only Command-R row
/// Camelid currently attempts. This is header-only: it deliberately does not
/// read tensor payload bytes, but it does require every canonical tensor name,
/// type, and dimension so a lookalike header cannot reach release-mode matvecs
/// with incompatible widths.
pub(crate) fn validate_command_r_attemptability_slice(
    gguf: &GgufFile,
    cfg: &LlamaModelConfig,
) -> Result<()> {
    let exact_metadata = gguf.model_name() == Some("Aya Expanse 8b")
        && gguf.metadata_string("general.license") == Some("cc-by-nc-4.0")
        && gguf.metadata_u32("general.file_type") == Some(15)
        && gguf.metadata_string("tokenizer.ggml.model") == Some("gpt2")
        && gguf.metadata_string("tokenizer.ggml.pre") == Some("command-r")
        && cfg.context_length == 8_192
        && cfg.embedding_length == 4_096
        && cfg.block_count == 32
        && cfg.feed_forward_length == 14_336
        && cfg.attention_head_count == 32
        && cfg.attention_head_count_kv == 8
        && cfg.vocab_size == Some(256_000)
        && cfg.rope_freq_base == Some(10_000.0)
        && cfg.rope_scaling_type.as_deref() == Some("none")
        && cfg.rms_norm_epsilon == 1e-5
        && cfg.logit_scale == Some(0.125);
    if !exact_metadata {
        return Err(BackendError::UnsupportedGguf(format!(
            "command-r runnable attemptability is currently pinned to Aya Expanse 8B \
             Q4_K_M (name='Aya Expanse 8b', file_type=15, 32x4096, FFN 14336, \
             GQA 32/8, vocab 256000, context 8192, command-r GPT-2 tokenizer); \
             observed name={:?}, file_type={:?}, layers={}, embedding={}, ffn={}, \
             heads={}/{}, vocab={:?}, context={}",
            gguf.model_name(),
            gguf.metadata_u32("general.file_type"),
            cfg.block_count,
            cfg.embedding_length,
            cfg.feed_forward_length,
            cfg.attention_head_count,
            cfg.attention_head_count_kv,
            cfg.vocab_size,
            cfg.context_length
        )));
    }

    let mut names = std::collections::HashSet::with_capacity(gguf.tensors.len());
    for tensor in &gguf.tensors {
        let Some((expected_type, expected_dimensions)) =
            command_r_aya_expected_tensor(&tensor.name)
        else {
            return Err(BackendError::UnsupportedGguf(format!(
                "command-r Aya Q4_K_M header contains unexpected tensor {}; the exact \
                 258-descriptor contract has only token_embd/output_norm and eight tensors \
                 per layer (shared attn_norm, Q/K/V/output, and SwiGLU gate/up/down)",
                tensor.name
            )));
        };
        if tensor.tensor_type != expected_type || tensor.dimensions != expected_dimensions {
            return Err(BackendError::UnsupportedGguf(format!(
                "command-r Aya Q4_K_M descriptor mismatch for {}: expected {:?} {:?}, \
                 observed {:?} {:?}",
                tensor.name,
                expected_type,
                expected_dimensions,
                tensor.tensor_type,
                tensor.dimensions
            )));
        }
        if !names.insert(tensor.name.as_str()) {
            return Err(BackendError::UnsupportedGguf(format!(
                "command-r Aya Q4_K_M header contains duplicate tensor {}; refusing a \
                 descriptor set whose first-match binding would be ambiguous",
                tensor.name
            )));
        }
    }
    // The accepted name universe is exactly 2 globals + 32 * 8 layer tensors.
    // Requiring 258 unique recognized names therefore also proves none are absent.
    if names.len() != 258 {
        return Err(BackendError::UnsupportedGguf(format!(
            "command-r Aya Q4_K_M canonical descriptor count mismatch: expected 258 \
             unique tensors, observed {}",
            names.len()
        )));
    }

    Ok(())
}

fn command_r_aya_expected_tensor(name: &str) -> Option<(GgufTensorType, Vec<u64>)> {
    match name {
        "token_embd.weight" => return Some((GgufTensorType::Q6K, vec![4_096, 256_000])),
        "output_norm.weight" => return Some((GgufTensorType::F32, vec![4_096])),
        _ => {}
    }

    let layer_and_tensor = name.strip_prefix("blk.")?;
    let (layer, tensor) = layer_and_tensor.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    if layer >= 32 {
        return None;
    }

    const Q6_DOWN_LAYERS: &[usize] = &[0, 1, 2, 3, 8, 10, 13, 16, 18, 21, 24, 27, 28, 29, 30, 31];
    const Q6_VALUE_LAYERS: &[usize] = &[0, 1, 2, 3, 6, 7, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31];

    Some(match tensor {
        "attn_norm.weight" => (GgufTensorType::F32, vec![4_096]),
        "attn_q.weight" | "attn_output.weight" => (GgufTensorType::Q4K, vec![4_096, 4_096]),
        "attn_k.weight" => (GgufTensorType::Q4K, vec![4_096, 1_024]),
        "attn_v.weight" => (
            if Q6_VALUE_LAYERS.contains(&layer) {
                GgufTensorType::Q6K
            } else {
                GgufTensorType::Q4K
            },
            vec![4_096, 1_024],
        ),
        "ffn_gate.weight" | "ffn_up.weight" => (GgufTensorType::Q4K, vec![4_096, 14_336]),
        "ffn_down.weight" => (
            if Q6_DOWN_LAYERS.contains(&layer) {
                GgufTensorType::Q6K
            } else {
                GgufTensorType::Q4K
            },
            vec![14_336, 4_096],
        ),
        _ => return None,
    })
}

/// Reference addition order from llama.cpp `models/command-r.cpp`:
/// `(ffn_out + residual) + attn_out`. The explicit order matters at f32
/// cancellation frontiers and is therefore kept in one unit-tested helper.
fn command_r_parallel_residual(
    output: &mut [f32],
    residual: &[f32],
    ffn_out: &[f32],
    attn_out: &[f32],
) {
    for (((dst, ffn), residual), attn) in output.iter_mut().zip(ffn_out).zip(residual).zip(attn_out)
    {
        *dst = (*ffn + *residual) + *attn;
    }
}

impl RunnableModel {
    /// Admit, parse config, and read every weight into resident quantized form.
    pub fn load(path: &str) -> Result<Self> {
        let gguf = read_metadata(path)?;
        admit::admit(&gguf).map_err(BackendError::from)?;
        let cfg = LlamaModelConfig::from_gguf(&gguf)?;
        validate_bitnet_b158_2b_4t(&gguf, &cfg)?;
        let arch = gguf
            .architecture()
            .ok_or_else(|| BackendError::InvalidModelMetadata("missing architecture".into()))?
            .to_string();
        let bitnet_embedding = crate::model::is_bitnet_embedding_model(&gguf);

        let d_model = cfg.embedding_length as usize;
        let n_heads = cfg.attention_head_count as usize;
        let n_kv_heads = cfg.attention_head_count_kv as usize;
        let head_dim = cfg
            .attention_key_length
            .map(|v| v as usize)
            .unwrap_or(d_model / n_heads);
        let rope_dim = cfg
            .rope_dimension_count
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let rope_base = cfg.rope_freq_base.unwrap_or(10_000.0);
        let n_layers = cfg.block_count as usize;

        if arch == "command-r" {
            validate_command_r_attemptability_slice(&gguf, &cfg)?;
        }

        let rope_freq_scale = match cfg.rope_scaling_type.as_deref().map(str::trim) {
            None | Some("") | Some("none") => 1.0,
            Some("linear") => {
                let factor = cfg.rope_scaling_factor.unwrap_or(1.0);
                if !factor.is_finite() || factor <= 0.0 {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "runnable lane: linear rope scaling factor must be finite and positive, got {factor}"
                    )));
                }
                factor.recip()
            }
            Some(kind) => {
                return Err(BackendError::UnsupportedGguf(format!(
                    "runnable lane: rope scaling {kind:?} not yet implemented"
                )))
            }
        };

        let mut f = File::open(path).map_err(|e| BackendError::Io {
            path: path.into(),
            source: e,
        })?;

        let load_raw = |f: &mut File, name: &str| -> Result<RawMat> {
            let d = find_tensor(&gguf, name)?;
            let (inf, outf) = mat_dims(d, name)?;
            let bitnet_tied_head = arch == "bitnet-b1.58"
                && d.tensor_type == GgufTensorType::F16
                && matches!(name, "token_embd.weight" | "output.weight");
            let bytes = if wants_page_backing(d.tensor_type) || bitnet_tied_head {
                let byte_len = usize::try_from(d.n_bytes).map_err(|_| {
                    BackendError::InvalidTensorData(format!(
                        "tensor {name} packed byte length {} does not fit usize",
                        d.n_bytes
                    ))
                })?;
                RawMatBytes::WirePages(crate::wire_mmap::WirePages::read_from_file(
                    f,
                    d.absolute_offset,
                    byte_len,
                )?)
            } else {
                RawMatBytes::owned(read_tensor_bytes(f, d, name)?)
            };
            Ok(RawMat {
                bytes,
                tt: d.tensor_type,
                in_features: inf,
                out_features: outf,
            })
        };
        let load_vec = |f: &mut File, name: &str| -> Result<Vec<f32>> {
            let d = find_tensor(&gguf, name)?;
            let n: usize = d.dimensions.iter().product::<u64>() as usize;
            super::dequant::dequantize(d.tensor_type, &read_tensor_bytes(f, d, name)?, n, name)
        };
        let load_vec_opt = |f: &mut File, name: &str| -> Result<Option<Vec<f32>>> {
            if find_tensor(&gguf, name).is_ok() {
                Ok(Some(load_vec(f, name)?))
            } else {
                Ok(None)
            }
        };

        let token_embd = load_raw(&mut f, "token_embd.weight")?;
        let vocab = token_embd.out_features;
        let output = if find_tensor(&gguf, "output.weight").is_ok() {
            load_raw(&mut f, "output.weight")?
        } else {
            // Tied embeddings (e.g. Llama-3.2): reuse token_embd as the logits matrix.
            RawMat {
                bytes: token_embd.bytes.clone(),
                tt: token_embd.tt,
                in_features: token_embd.in_features,
                out_features: token_embd.out_features,
            }
        };
        // LFM2 spells its final RMSNorm `token_embd_norm` rather than
        // `output_norm` — llama.cpp maps LLM_TENSOR_OUTPUT_NORM_LFM2 to that
        // name explicitly (`llama-arch.cpp:362`, commented there as a fix for
        // the wrong tensor name). Every other covered arch keeps `output_norm`.
        let output_norm = if arch == "lfm2" {
            load_vec(&mut f, "token_embd_norm.weight")?
        } else {
            load_vec(&mut f, "output_norm.weight")?
        };

        // Qwen3.5 (Ornith): hybrid gated-delta-net. Layers do not fit the generic
        // dense `Layer` (recurrent/SSM layers carry no K/V projections), so build a
        // dedicated runtime here and route the forward pass to the `*_qwen35` path.
        if arch == "qwen35" {
            let meta = cfg.qwen35.as_ref().ok_or_else(|| {
                BackendError::InvalidModelMetadata("qwen35 metadata missing from config".into())
            })?;
            let d_state = meta.ssm_d_state as usize;
            let num_k_heads = meta.ssm_n_group as usize;
            let num_v_heads = meta.ssm_dt_rank as usize;
            let d_inner = meta.ssm_d_inner as usize;
            let d_conv = meta.ssm_d_conv as usize;
            if num_v_heads == 0 || num_k_heads == 0 || d_state == 0 || d_conv == 0 {
                return Err(BackendError::InvalidModelMetadata(
                    "qwen35: degenerate ssm dims (state/group/rank/conv must be non-zero)".into(),
                ));
            }
            let head_v_dim = d_inner / num_v_heads;
            let key_dim = d_state * num_k_heads;
            let value_dim = head_v_dim * num_v_heads;
            let conv_dim = 2 * key_dim + value_dim;
            let rope_sections =
                match gguf.metadata_array_u32_optional("qwen35.rope.dimension_sections")? {
                    Some(values) if values.len() == 4 => [
                        values[0] as usize,
                        values[1] as usize,
                        values[2] as usize,
                        values[3] as usize,
                    ],
                    Some(values) => {
                        return Err(BackendError::InvalidModelMetadata(format!(
                            "qwen35 rope.dimension_sections must contain four values, got {}",
                            values.len()
                        )))
                    }
                    // Text-only Qwen3.5 rows without explicit multimodal metadata
                    // remain valid and collapse every pair into the time section.
                    None => [rope_dim / 2, 0, 0, 0],
                };
            if rope_sections.iter().sum::<usize>() != rope_dim / 2 {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "qwen35 rope sections {:?} do not cover rope_dim {rope_dim}",
                    rope_sections
                )));
            }

            let mut q35_layers = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let p = |t: &str| format!("blk.{l}.{t}.weight");
                let attn_norm = load_vec(&mut f, &p("attn_norm"))?;
                let post_attn_norm = load_vec(&mut f, &p("post_attention_norm"))?;
                let ffn_gate = load_raw(&mut f, &p("ffn_gate"))?;
                let ffn_up = load_raw(&mut f, &p("ffn_up"))?;
                let ffn_down = load_raw(&mut f, &p("ffn_down"))?;
                let kind = if meta.is_recurrent_layer(l) {
                    Qwen35Kind::Ssm {
                        wqkv: load_raw(&mut f, &p("attn_qkv"))?,
                        wqkv_gate: load_raw(&mut f, &p("attn_gate"))?,
                        // ssm_conv1d.weight is F32 [d_conv, conv_dim]; load flat:
                        // flat[c*d_conv + i] = kernel[tap=i, channel=c].
                        conv1d: load_vec(&mut f, &p("ssm_conv1d"))?,
                        // ssm_dt carries a `.bias` suffix; ssm_a carries NO suffix.
                        dt_bias: load_vec(&mut f, &format!("blk.{l}.ssm_dt.bias"))?,
                        a: load_vec(&mut f, &format!("blk.{l}.ssm_a"))?,
                        beta: load_raw(&mut f, &p("ssm_beta"))?,
                        alpha: load_raw(&mut f, &p("ssm_alpha"))?,
                        ssm_norm: load_vec(&mut f, &p("ssm_norm"))?,
                        ssm_out: load_raw(&mut f, &p("ssm_out"))?,
                    }
                } else {
                    Qwen35Kind::Full {
                        wq: load_raw(&mut f, &p("attn_q"))?, // fused query + output gate
                        wk: load_raw(&mut f, &p("attn_k"))?,
                        wv: load_raw(&mut f, &p("attn_v"))?,
                        wo: load_raw(&mut f, &p("attn_output"))?,
                        q_norm: load_vec(&mut f, &p("attn_q_norm"))?,
                        k_norm: load_vec(&mut f, &p("attn_k_norm"))?,
                    }
                };
                q35_layers.push(Qwen35Layer {
                    attn_norm,
                    post_attn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    kind,
                });
            }

            #[cfg(feature = "cuda")]
            let resident_cuda_artifact = {
                use crate::cuda_resident::{
                    resident_cuda_artifact_from_sha256, ResidentCudaArtifact,
                };

                // Hash only the plausible exact Q1 row. This exact digest admits
                // kernels/layouts validated against one concrete artifact, so a
                // user-writable persistent cache cannot be its authority.
                let ffn_dim = q35_layers
                    .first()
                    .map(|layer| layer.ffn_gate.out_features)
                    .unwrap_or(0);
                let has_q1_projection = q35_layers.iter().any(|layer| {
                    [layer.ffn_gate.tt, layer.ffn_up.tt, layer.ffn_down.tt]
                        .contains(&GgufTensorType::Q1_0)
                });
                let candidate = n_layers == 64
                    && d_model == 5_120
                    && ffn_dim == 17_408
                    && n_heads * head_dim == 6_144
                    && n_kv_heads * head_dim == 1_024
                    && has_q1_projection;

                if candidate {
                    match crate::receipt::sha256_file_hex(std::path::Path::new(path)) {
                        Ok(sha256) => resident_cuda_artifact_from_sha256(&sha256),
                        Err(err) => {
                            eprintln!(
                                "[qwen35] CUDA artifact identity unavailable; exact Bonsai-27B Q1 specializations disabled: {err}"
                            );
                            ResidentCudaArtifact::Generic
                        }
                    }
                } else {
                    ResidentCudaArtifact::Generic
                }
            };

            return Ok(Self {
                architecture: arch,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_dim,
                rope_base,
                rope_freq_scale,
                eps: cfg.rms_norm_epsilon,
                vocab,
                rope_neox: true, // NEOX split-half, partial over rope_dim (64) of head_dim (256)
                n_layers,
                layer_rope_base: vec![rope_base; n_layers],
                embed_scale: None,
                ffn_gelu: false,
                final_logit_softcap: None,
                attn_logit_softcap: None,
                logit_scale: None,
                token_embd,
                output,
                output_norm,
                layers: Vec::new(),
                qwen35: Some(Qwen35Runtime {
                    layers: q35_layers,
                    rope_sections,
                    d_conv,
                    d_state,
                    num_k_heads,
                    num_v_heads,
                    head_v_dim,
                    key_dim,
                    value_dim,
                    conv_dim,
                }),
                lfm2: None,
                #[cfg(feature = "cuda")]
                resident_cuda_artifact,
                #[cfg(feature = "cuda")]
                cuda: std::sync::Mutex::new(None),
                #[cfg(target_os = "macos")]
                metal_qwen35: std::sync::Mutex::new(None),
                #[cfg(target_os = "macos")]
                metal_lfm2: std::sync::Mutex::new(None),
            });
        }

        // LFM2 / LFM2.5: hybrid double-gated short convolution + GQA attention.
        // Conv layers carry `shortconv.{conv,in_proj,out_proj}` and NO
        // `attn_q/k/v`, so they cannot be expressed as the generic dense
        // `Layer`; build a dedicated runtime and route the forward pass to the
        // `*_lfm2` path. Graph ported from llama.cpp `src/models/lfm2.cpp`.
        if arch == "lfm2" {
            let meta = cfg.lfm2.as_ref().ok_or_else(|| {
                BackendError::InvalidModelMetadata("lfm2 metadata missing from config".into())
            })?;
            let l_cache = meta.shortconv_l_cache as usize;
            // llama.cpp asserts `n_shortconv_l_cache > 1` (`lfm2.cpp:167`): the
            // rolling state is `l_cache - 1` wide, so a 0/1 cache has no state
            // to carry and the file is malformed for this graph.
            if l_cache < 2 {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "lfm2: shortconv.l_cache must be >= 2, got {l_cache}"
                )));
            }
            if meta.layer_is_conv.len() != n_layers {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "lfm2: per-layer schedule covers {} layers, model has {n_layers}",
                    meta.layer_is_conv.len()
                )));
            }
            // Every attention layer must agree with the config scalar; a row
            // with mixed non-zero KV widths would need per-layer KV sizing
            // that this runtime does not carry, so refuse rather than
            // mis-shape the cache.
            for (l, &kv) in meta.kv_heads_per_layer.iter().enumerate() {
                if kv != 0 && kv as usize != n_kv_heads {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "lfm2: layer {l} declares {kv} kv heads, expected {n_kv_heads} \
                         (per-layer KV widths are not supported)"
                    )));
                }
            }
            // SLIDING WINDOW — fail closed. llama.cpp's LFM2 loader reads
            // `attention.sliding_window` and, when present and non-zero, marks
            // every ATTENTION layer SWA (`lfm2.cpp:23-28`). This forward is
            // full-causal and carries no window mask, so a windowed row would
            // decode fluent-looking garbage rather than erroring — the same
            // hazard class as gemma3 on the CPU dense path (DECISIONS D20.2).
            // LFM2.5-2.6B declares no window (`n_swa = 0`), so this refuses
            // only rows this lane genuinely cannot run.
            if let Some(window) = gguf.metadata_u32("lfm2.attention.sliding_window") {
                if window > 0 {
                    return Err(BackendError::UnsupportedGguf(format!(
                        "lfm2: attention.sliding_window = {window}, but the runnable \
                         lfm2 forward is full-causal and implements no window mask; \
                         refusing rather than decoding with the wrong attention span"
                    )));
                }
            }
            // An all-conv schedule would leave zero attention layers, and the
            // forward divides by `n_kv_heads` to derive the GQA group. Refuse
            // instead of panicking on a divide-by-zero deep in decode.
            if n_kv_heads == 0 || meta.layer_is_conv.iter().all(|c| *c) {
                return Err(BackendError::UnsupportedGguf(
                    "lfm2: no attention layers in the per-layer schedule (all conv); \
                     this runtime requires at least one GQA layer"
                        .into(),
                ));
            }

            let mut lfm2_layers = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let p = |t: &str| format!("blk.{l}.{t}.weight");
                // `attn_norm` is LFM2's `operator_norm`: the pre-block RMSNorm
                // on BOTH conv and attention layers (`lfm2.cpp:239`).
                let attn_norm = load_vec(&mut f, &p("attn_norm"))?;
                let ffn_norm = load_vec(&mut f, &p("ffn_norm"))?;
                let ffn_gate = load_raw(&mut f, &p("ffn_gate"))?;
                let ffn_up = load_raw(&mut f, &p("ffn_up"))?;
                let ffn_down = load_raw(&mut f, &p("ffn_down"))?;
                let kind = if meta.is_conv_layer(l) {
                    // shortconv.conv is [l_cache, n_embd]; GGUF is row-major
                    // over ne[0], so flat[c*l_cache + t] is channel `c`, tap
                    // `t` — the same channel-major layout the qwen35 conv loop
                    // consumes. Validate the element count HERE: the forward
                    // indexes `conv[ch*l_cache + tap]` for every channel, so a
                    // kernel that disagrees with (l_cache, d_model) would either
                    // panic mid-decode or silently read a neighbouring
                    // channel's taps.
                    let conv = load_vec(&mut f, &p("shortconv.conv"))?;
                    if conv.len() != l_cache * d_model {
                        return Err(BackendError::InvalidTensorData(format!(
                            "lfm2 layer {l}: shortconv.conv has {} elements, expected \
                             l_cache*embedding = {}*{} = {}",
                            conv.len(),
                            l_cache,
                            d_model,
                            l_cache * d_model
                        )));
                    }
                    Lfm2Kind::Conv {
                        conv,
                        in_proj: load_raw(&mut f, &p("shortconv.in_proj"))?,
                        out_proj: load_raw(&mut f, &p("shortconv.out_proj"))?,
                    }
                } else {
                    Lfm2Kind::Attn {
                        wq: load_raw(&mut f, &p("attn_q"))?,
                        wk: load_raw(&mut f, &p("attn_k"))?,
                        wv: load_raw(&mut f, &p("attn_v"))?,
                        wo: load_raw(&mut f, &p("attn_output"))?,
                        q_norm: load_vec(&mut f, &p("attn_q_norm"))?,
                        k_norm: load_vec(&mut f, &p("attn_k_norm"))?,
                    }
                };
                lfm2_layers.push(Lfm2Layer {
                    attn_norm,
                    ffn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    kind,
                });
            }

            return Ok(RunnableModel {
                architecture: arch,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_dim,
                rope_base,
                rope_freq_scale,
                eps: cfg.rms_norm_epsilon,
                vocab,
                // llama.cpp classifies LLM_ARCH_LFM2 as LLAMA_ROPE_TYPE_NEOX
                // (`llama-model.cpp:2477` → `:2492`); the converter leaves Q/K
                // unpermuted, so split-half is what the weights expect.
                rope_neox: true,
                n_layers,
                layer_rope_base: vec![rope_base; n_layers],
                embed_scale: None,
                ffn_gelu: false,
                final_logit_softcap: None,
                attn_logit_softcap: None,
                logit_scale: None,
                token_embd,
                output,
                output_norm,
                layers: Vec::new(),
                qwen35: None,
                lfm2: Some(Lfm2Runtime {
                    layers: lfm2_layers,
                    l_cache,
                }),
                // No hash-pinned CUDA specialization exists for lfm2; the
                // short-conv lane is CPU-only today.
                #[cfg(feature = "cuda")]
                resident_cuda_artifact: crate::cuda_resident::ResidentCudaArtifact::Generic,
                #[cfg(feature = "cuda")]
                cuda: std::sync::Mutex::new(None),
                #[cfg(target_os = "macos")]
                metal_qwen35: std::sync::Mutex::new(None),
                #[cfg(target_os = "macos")]
                metal_lfm2: std::sync::Mutex::new(None),
            });
        }

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let ffn = cfg.feed_forward_length as usize;
        let is_command_r = arch == "command-r";

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = |t: &str| format!("blk.{l}.{t}.weight");

            // phi3 fuses Q/K/V into a single attn_qkv; split it by out-feature rows.
            let (wq, wk, wv) = if find_tensor(&gguf, &p("attn_q")).is_ok() {
                (
                    load_raw(&mut f, &p("attn_q"))?,
                    load_raw(&mut f, &p("attn_k"))?,
                    load_raw(&mut f, &p("attn_v"))?,
                )
            } else {
                let qkv = load_raw(&mut f, &p("attn_qkv"))?;
                (
                    qkv.split_rows(0, q_dim),
                    qkv.split_rows(q_dim, kv_dim),
                    qkv.split_rows(q_dim + kv_dim, kv_dim),
                )
            };
            // phi3 fuses gate+up into ffn_up [2*ffn] (gate first); split it.
            let (gate, up) = if find_tensor(&gguf, &p("ffn_gate")).is_ok() {
                (
                    load_raw(&mut f, &p("ffn_gate"))?,
                    load_raw(&mut f, &p("ffn_up"))?,
                )
            } else {
                let gu = load_raw(&mut f, &p("ffn_up"))?;
                (gu.split_rows(0, ffn), gu.split_rows(ffn, ffn))
            };

            let attn_norm = load_vec(&mut f, &p("attn_norm"))?;
            // Command R's parallel residual feeds both branches from this one
            // LayerNorm and has no separate `ffn_norm` tensor.
            let ffn_norm = if is_command_r {
                attn_norm.clone()
            } else {
                load_vec(&mut f, &p("ffn_norm"))?
            };
            let layer = Layer {
                attn_norm,
                ffn_norm,
                wq,
                wk,
                wv,
                wo: load_raw(&mut f, &p("attn_output"))?,
                gate,
                up,
                down: load_raw(&mut f, &p("ffn_down"))?,
                q_norm: load_vec_opt(&mut f, &p("attn_q_norm"))?,
                k_norm: load_vec_opt(&mut f, &p("attn_k_norm"))?,
                q_norm_in: load_vec_opt(&mut f, &p("attn_q_norm_in"))?,
                k_norm_in: load_vec_opt(&mut f, &p("attn_k_norm_in"))?,
                v_norm_in: load_vec_opt(&mut f, &p("attn_v_norm_in"))?,
                output_norm_in: load_vec_opt(&mut f, &p("attn_output_norm_in"))?,
                gate_norm_in: load_vec_opt(&mut f, &p("ffn_gate_norm_in"))?,
                up_norm_in: load_vec_opt(&mut f, &p("ffn_up_norm_in"))?,
                down_norm_in: load_vec_opt(&mut f, &p("ffn_down_norm_in"))?,
                attn_sub_norm: load_vec_opt(&mut f, &p("attn_sub_norm"))?,
                ffn_sub_norm: load_vec_opt(&mut f, &p("ffn_sub_norm"))?,
                post_attn_norm: load_vec_opt(&mut f, &p("post_attention_norm"))?,
                post_ffn_norm: load_vec_opt(&mut f, &p("post_ffw_norm"))?,
            };
            validate_extra_norms(&layer, l, &arch, bitnet_embedding, d_model, q_dim, ffn)?;
            layers.push(layer);
        }

        let is_gemma = arch.starts_with("gemma");
        // RoPE pairing: NEOX (split-half) for qwen3/gemma/phi3; adjacent/interleaved
        // for standard LLaMA conversions AND Command R. The latter is explicit in
        // pinned llama.cpp's `LLM_ARCH_COMMAND_R` rope-type classification.
        let rope_neox = cfg.rope_neox_pairing || is_gemma || arch == "phi3";
        // gemma3 dual RoPE: the per-layer global/local schedule and both RoPE bases
        // come from the SAME parsed `Gemma3Metadata` the resident lane consumes
        // (`LlamaModelConfig::from_gguf` -> `cfg.gemma3`) — single source of truth,
        // so a row carrying the explicit override keys
        // (`gemma3.attention.sliding_window_pattern` / `gemma3.rope.freq_base_swa`)
        // can never make the two lanes derive different schedules for one file.
        // For the real 1B row (no override keys) the metadata resolves to the
        // reference-pinned pattern 6 / local base 10000 and the required GGUF
        // `rope.freq_base` (1e6) for globals — bit-identical to the constants this
        // lane hardcoded before Phase 1b. The sliding window itself is a no-op for
        // prompts shorter than the window (this lane implements no window mask —
        // a documented full-support blocker).
        let layer_rope_base = if let Some(gemma3) = cfg.gemma3.as_ref() {
            (0..n_layers).map(|i| gemma3.rope_freq_base_at(i)).collect()
        } else {
            vec![rope_base; n_layers]
        };

        let final_logit_softcap = gguf.metadata_f32(&format!("{arch}.final_logit_softcapping"));
        let attn_logit_softcap = gguf.metadata_f32(&format!("{arch}.attn_logit_softcapping"));
        let logit_scale = gguf.metadata_f32(&format!("{arch}.logit_scale"));

        Ok(Self {
            architecture: arch,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rope_base,
            rope_freq_scale,
            eps: cfg.rms_norm_epsilon,
            vocab,
            rope_neox,
            n_layers,
            layer_rope_base,
            embed_scale: if is_gemma {
                Some((d_model as f32).sqrt())
            } else {
                None
            },
            ffn_gelu: is_gemma,
            final_logit_softcap,
            attn_logit_softcap,
            logit_scale,
            token_embd,
            output,
            output_norm,
            layers,
            qwen35: None,
            lfm2: None,
            #[cfg(feature = "cuda")]
            resident_cuda_artifact: crate::cuda_resident::ResidentCudaArtifact::Generic,
            #[cfg(feature = "cuda")]
            cuda: std::sync::Mutex::new(None),
            #[cfg(target_os = "macos")]
            metal_qwen35: std::sync::Mutex::new(None),
            #[cfg(target_os = "macos")]
            metal_lfm2: std::sync::Mutex::new(None),
        })
    }

    fn apply_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        if self.architecture == "command-r" {
            layer_norm(x, weight, self.eps)
        } else {
            rms_norm(x, weight, self.eps)
        }
    }

    fn apply_optional_norm(&self, x: &[f32], weight: Option<&Vec<f32>>) -> Vec<f32> {
        match weight {
            Some(weight) => self.apply_norm(x, weight),
            None => x.to_vec(),
        }
    }

    fn apply_norm_heads(&self, vec: &mut [f32], n_heads: usize, head_dim: usize, weight: &[f32]) {
        if self.architecture == "command-r" {
            layer_norm_heads(vec, n_heads, head_dim, weight, self.eps)
        } else {
            norm_heads(vec, n_heads, head_dim, weight, self.eps)
        }
    }

    /// Forward the whole token sequence; return logits for the **last** position.
    /// Pure f32, deterministic, no KV cache — recomputed each call.
    pub fn forward_logits(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(BackendError::InvalidTensorData(
                "empty token sequence".into(),
            ));
        }
        if self.qwen35.is_some() {
            return self.forward_logits_qwen35(tokens);
        }
        // LFM2's conv state is inherently sequential (a rolling ring per
        // channel), so the whole-sequence forward is the incremental step run
        // over every position — the same lane `generate` uses, which keeps the
        // smoke gate and decode on one code path.
        if let Some(rt) = &self.lfm2 {
            let mut cache = self.new_cache();
            let mut logits = Vec::new();
            // This entry point returns the LAST position's logits, so only that
            // position needs the tied 128k-row head.
            let last = tokens.len().saturating_sub(1);
            for (pos, &tok) in tokens.iter().enumerate() {
                logits = self.forward_step_lfm2(rt, tok, pos, &mut cache, pos == last)?;
            }
            return Ok(logits);
        }
        // Command R uses a parallel attention/FFN residual, while the generic
        // batched helpers below implement sequential pre-norm blocks. Run the
        // same corrected KV-cached step that generation uses so smoke/diagnostic
        // logits cannot accidentally exercise the wrong graph.
        if self.architecture == "command-r" {
            let mut cache = self.new_cache();
            let mut logits = Vec::new();
            let last = tokens.len().saturating_sub(1);
            for (pos, &tok) in tokens.iter().enumerate() {
                logits = self.forward_step_maybe_logits(tok, pos, &mut cache, pos == last)?;
            }
            return Ok(logits);
        }
        let hidden = self.forward_hidden_states(tokens)?;
        let normed = &hidden[hidden.len() - self.d_model..];
        self.project_output_logits(normed)
    }

    /// Project one normalized hidden row through the language-model head and
    /// apply the architecture's final logit transforms.
    fn project_output_logits(&self, normed: &[f32]) -> Result<Vec<f32>> {
        let mut logits = if self.architecture == "bitnet-b1.58" {
            // The official 2B model ties a 128,256 x 2,560 F16 embedding to its
            // LM head. `par_matvec` keeps that matrix page-backed on Metal and
            // uses allocation-free row dots on CPU, instead of making 128k
            // temporary Vecs for every generated token. Windows additionally
            // keeps the immutable F16 wire bytes resident in CUDA and runs one
            // 256-thread reduction block per vocabulary row. Every gate below
            // is intentionally exact: this is not a generic dense-F16 CUDA path.
            let tied_pages = match (
                self.token_embd.bytes.wire_pages(),
                self.output.bytes.wire_pages(),
            ) {
                (Some(token), Some(output)) if Arc::ptr_eq(token, output) => Some(output),
                _ => None,
            };
            let exact_cuda_head = self.output.tt == GgufTensorType::F16
                && self.output.in_features == 2_560
                && self.d_model == 2_560
                && self.output.out_features == 128_256
                && self.vocab == 128_256
                && crate::bitnet_kernels::gpu_allowed()
                && crate::cuda::gpu_accel_enabled();
            if exact_cuda_head {
                if let Some(pages) = tied_pages {
                    let mut output = vec![0.0_f32; self.vocab];
                    if crate::cuda::try_bitnet_f16_head_matvec(
                        normed,
                        pages,
                        self.vocab,
                        self.d_model,
                        &mut output,
                    ) {
                        output
                    } else {
                        self.output.par_matvec(normed, "output")?
                    }
                } else {
                    self.output.par_matvec(normed, "output")?
                }
            } else {
                self.output.par_matvec(normed, "output")?
            }
        } else {
            let mut logits = vec![0.0f32; self.vocab];
            for (token, logit) in logits.iter_mut().enumerate() {
                let row = self.output.dequant_row(token, "output")?;
                *logit = dot(&row, normed);
            }
            logits
        };
        // gemma2 final logit soft-cap (gemma3: None).
        if let Some(cap) = self.final_logit_softcap {
            for l in logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
        if let Some(scale) = self.logit_scale {
            for l in logits.iter_mut() {
                *l *= scale;
            }
        }
        Ok(logits)
    }

    /// Run the generic causal stack and return final-normalized hidden states for
    /// every input position. This is also the encoder surface used by the official
    /// decoder-only BitNet embedding checkpoints before their declared pooling.
    pub(crate) fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.forward_hidden_states_with_cache(tokens, None, None)
    }

    /// Batched generic forward with optional KV capture. BitNet chat uses the
    /// captured cache for prompt prefill; embedding models use the same graph
    /// without a cache. `cancelled` is checked between transformer blocks so a
    /// disconnected streaming request does not continue monopolizing the GPU.
    fn forward_hidden_states_with_cache(
        &self,
        tokens: &[u32],
        mut cache: Option<&mut KvCache>,
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(BackendError::InvalidTensorData(
                "empty token sequence".into(),
            ));
        }
        if self.qwen35.is_some() || self.lfm2.is_some() || self.architecture == "command-r" {
            return Err(BackendError::UnsupportedGguf(format!(
                "{} does not expose full-sequence hidden states",
                self.architecture
            )));
        }
        let seq = tokens.len();
        let dm = self.d_model;
        let mut hidden = vec![0.0f32; seq * dm];
        for (pos, &tok) in tokens.iter().enumerate() {
            let t = tok as usize;
            if t >= self.vocab {
                return Err(BackendError::InvalidTensorData(format!(
                    "token id {t} >= vocab {}",
                    self.vocab
                )));
            }
            let mut row = self.token_embd.dequant_row(t, "token_embd")?;
            if let Some(scale) = self.embed_scale {
                for value in &mut row {
                    *value *= scale;
                }
            }
            hidden[pos * dm..(pos + 1) * dm].copy_from_slice(&row);
        }

        let dump_path = std::env::var("CAMELID_LAYER_DUMP").ok();
        let mut dump = String::new();
        for (li, layer) in self.layers.iter().enumerate() {
            if cancelled.is_some_and(|is_cancelled| is_cancelled()) {
                return Err(BackendError::InvalidTensorData(
                    "generation cancelled".into(),
                ));
            }
            self.attention_block(layer, li, &mut hidden, seq, cache.as_deref_mut())?;
            self.ffn_block(layer, li, &mut hidden, seq)?;
            if dump_path.is_some() {
                let last = &hidden[(seq - 1) * dm..seq * dm];
                let l2 = last.iter().map(|value| value * value).sum::<f32>().sqrt();
                dump.push_str(&format!(
                    "{li}\t{l2:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                    last[0], last[1], last[2], last[3]
                ));
            }
        }
        if let Some(path) = dump_path {
            let _ = std::fs::write(path, &dump);
        }
        for row in hidden.chunks_exact_mut(dm) {
            let normed = self.apply_norm(row, &self.output_norm);
            row.copy_from_slice(&normed);
        }
        Ok(hidden)
    }

    /// Greedy-decode up to `max_new` tokens. Uses an incremental KV cache: the prompt
    /// is prefilled position-by-position, then each new token computes only its own
    /// position and attends over the cache. Produces results bit-identical to the
    /// stateless [`forward_logits`] path (the attention sum order is unchanged), but
    /// O(seq) matmuls instead of O(seq²).
    ///
    /// [`forward_logits`]: RunnableModel::forward_logits
    pub fn generate(&self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(BackendError::InvalidTensorData("empty prompt".into()));
        }
        if self.qwen35.is_some() {
            return self.generate_qwen35(prompt, max_new, &[]);
        }
        // LFM2 resident Metal graph (opt-in via CAMELID_LFM2_METAL while it is
        // being proven). No silent CPU fallback: the conv ring is order-dependent
        // and a mid-stream replay would restart from the prompt, so a Metal failure
        // surfaces as an error rather than a quietly different lane.
        #[cfg(target_os = "macos")]
        if self.lfm2.is_some() && lfm2_metal_enabled() {
            return self.generate_lfm2_metal(prompt, max_new, &[], None, &mut |_| {});
        }
        let mut cache = self.new_cache();
        let last = self.prefill_generic(prompt, &mut cache, None)?;
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        let mut next = argmax(&last);
        for i in 0..max_new {
            out.push(next);
            if i + 1 < max_new {
                let logits = self.forward_step(next, pos, &mut cache)?;
                pos += 1;
                next = argmax(&logits);
            }
        }
        Ok(out)
    }

    /// Decode a Qwen3.5 prompt containing one Prism image. `prefix` is the text
    /// through `<|vision_start|>` and `suffix` begins with
    /// `<|vision_end|>`. Image embeddings occupy physical KV slots but advance
    /// the decoder's multimodal RoPE clock by the larger grid dimension, which
    /// matches llama.cpp's mtmd position contract.
    pub fn generate_vision_stopping_streaming(
        &self,
        prefix: &[u32],
        image: &super::vision::PrismVisionEmbedding,
        suffix: &[u32],
        max_new: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        #[cfg(target_os = "macos")]
        {
            self.generate_qwen35_vision_metal(prefix, image, suffix, max_new, stop, None, on_token)
        }
        #[cfg(not(target_os = "macos"))]
        {
            #[cfg(feature = "cuda")]
            {
                self.generate_qwen35_vision_cuda(
                    prefix, image, suffix, max_new, stop, None, on_token,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (prefix, image, suffix, max_new, stop, on_token);
                Err(BackendError::UnsupportedGguf(
                    "Prism image generation requires Metal or CUDA".into(),
                ))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_vision_stopping_streaming_with_sampling(
        &self,
        prefix: &[u32],
        image: &super::vision::PrismVisionEmbedding,
        suffix: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        #[cfg(target_os = "macos")]
        {
            let sampling = qwen35_sampling_requires_logits(sampling).then_some(sampling);
            self.generate_qwen35_vision_metal(
                prefix, image, suffix, max_new, stop, sampling, on_token,
            )
        }
        #[cfg(not(target_os = "macos"))]
        {
            #[cfg(feature = "cuda")]
            {
                let sampling = qwen35_sampling_requires_logits(sampling).then_some(sampling);
                self.generate_qwen35_vision_cuda(
                    prefix, image, suffix, max_new, stop, sampling, on_token,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (prefix, image, suffix, max_new, stop, sampling, on_token);
                Err(BackendError::UnsupportedGguf(
                    "Prism image generation requires Metal or CUDA".into(),
                ))
            }
        }
    }

    pub fn generate_vision(
        &self,
        prefix: &[u32],
        image: &super::vision::PrismVisionEmbedding,
        suffix: &[u32],
        max_new: usize,
        stop: &[u32],
    ) -> Result<Vec<u32>> {
        self.generate_vision_stopping_streaming(prefix, image, suffix, max_new, stop, &mut |_| {})
    }

    /// A `KvCache` sized for this model, including LFM2's per-conv-layer
    /// rolling short-conv state. Every incremental lane must allocate through
    /// this rather than `KvCache::new`, or LFM2's conv layers would index an
    /// empty ring.
    fn new_cache(&self) -> KvCache {
        let mut cache = KvCache::new(self.n_layers);
        if let Some(rt) = &self.lfm2 {
            for (li, layer) in rt.layers.iter().enumerate() {
                if matches!(layer.kind, Lfm2Kind::Conv { .. }) {
                    cache.conv[li] = vec![0.0f32; (rt.l_cache - 1) * self.d_model];
                }
            }
        }
        cache
    }

    /// Fill the generic decoder cache and return the final prompt position's
    /// logits. BitNet uses the existing full-sequence projection path so every
    /// layer submits one batched I2_S matmul per projection instead of one Metal
    /// command buffer per prompt token. Other generic architectures retain their
    /// established incremental prefill order.
    fn prefill_generic(
        &self,
        prompt: &[u32],
        cache: &mut KvCache,
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<Vec<f32>> {
        if self.architecture == "bitnet-b1.58" {
            let hidden = self.forward_hidden_states_with_cache(prompt, Some(cache), cancelled)?;
            if cancelled.is_some_and(|is_cancelled| is_cancelled()) {
                return Err(BackendError::InvalidTensorData(
                    "generation cancelled".into(),
                ));
            }
            let normed = &hidden[hidden.len() - self.d_model..];
            return self.project_output_logits(normed);
        }

        let mut last = Vec::new();
        let last_prompt = prompt.len().saturating_sub(1);
        for (pos, &token) in prompt.iter().enumerate() {
            if cancelled.is_some_and(|is_cancelled| is_cancelled()) {
                return Err(BackendError::InvalidTensorData(
                    "generation cancelled".into(),
                ));
            }
            last = self.forward_step_maybe_logits(token, pos, cache, pos == last_prompt)?;
        }
        Ok(last)
    }

    /// Incremental forward of one LFM2 token at absolute `pos`. Conv layers
    /// advance their rolling state; attention layers append K/V and attend over
    /// all cached positions. Returns next-token logits.
    ///
    /// Ported from llama.cpp `src/models/lfm2.cpp` (graph at `:235-274`,
    /// short-conv block at `:156-217`). Per layer:
    ///
    /// ```text
    /// prev = h
    /// h    = RMSNorm(h, attn_norm)          // LFM2 "operator_norm"
    /// h    = conv_block(h) | attn_block(h)  // per-layer schedule
    /// h    = prev + h
    /// h    = h + SwiGLU_FFN(RMSNorm(h, ffn_norm))
    /// ```
    /// `need_logits = false` advances the cache and the conv ring but skips the
    /// 128,000-row tied LM head. Every prompt-prefill position except the last
    /// discards its logits, and that projection is 9.7% of the bytes a step
    /// touches — the same economy `decode_token_qwen35` already takes.
    fn forward_step_lfm2(
        &self,
        rt: &Lfm2Runtime,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        need_logits: bool,
    ) -> Result<Vec<f32>> {
        let dm = self.d_model;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = self.n_heads / self.n_kv_heads;
        let q_dim = self.n_heads * hd;
        let kv_dim = self.n_kv_heads * hd;
        let cm1 = rt.l_cache - 1;

        let t = token as usize;
        if t >= self.vocab {
            return Err(BackendError::InvalidTensorData(format!(
                "token id {t} >= vocab {}",
                self.vocab
            )));
        }
        let mut hidden = self.token_embd.dequant_row(t, "token_embd")?;

        for (li, layer) in rt.layers.iter().enumerate() {
            let tn = |t: &str| format!("blk.{li}.{t}");
            // LFM2 applies `operator_norm` before BOTH block kinds. `hidden` is
            // the residual branch and is not touched again until the add below,
            // so no separate `prev` copy is needed.
            let xn = self.apply_norm(&hidden, &layer.attn_norm);

            let mix = match &layer.kind {
                Lfm2Kind::Conv {
                    conv,
                    in_proj,
                    out_proj,
                } => {
                    // in_proj emits 3*d_model, chunked B | C | x (`lfm2.cpp:179-184`).
                    //
                    // Every projection here goes through `par_matvec`, the same
                    // kernel the qwen35 runnable path uses. NOTE it is NOT an
                    // f32-activation dot: for Q8_0 weights it quantizes the
                    // activation once and reduces in int8×int8, which is what
                    // llama.cpp's own q8×q8 path does — so it is numerically
                    // CLOSER to the reference, not bit-identical to
                    // `dequant_all().matvec()`. The parity claim it carries is
                    // greedy-token (argmax) identity, which is exactly what
                    // `tests/lfm2_parity.rs` certifies. It also avoids
                    // materializing each weight matrix as f32, which at 2.6B
                    // would allocate GBs per token.
                    let bcx = in_proj.par_matvec(&xn, &tn("shortconv.in_proj"))?;
                    if bcx.len() != 3 * dm {
                        return Err(BackendError::InvalidTensorData(format!(
                            "lfm2 layer {li}: shortconv.in_proj produced {} values, expected {}",
                            bcx.len(),
                            3 * dm
                        )));
                    }
                    let (b, rest) = bcx.split_at(dm);
                    let (c, x) = rest.split_at(dm);

                    // Causal depthwise conv of width `l_cache` over `bx = b*x`.
                    // The window for channel `ch` is
                    // [state_0 (oldest) .. state_{l_cache-2}, bx_now]; the kernel
                    // is channel-major (`conv[ch*l_cache + tap]`) because the GGUF
                    // tensor is [l_cache, n_embd] and rows run along ne[0].
                    // NOTE: `ggml_ssm_conv` applies NO activation — unlike the
                    // qwen35 SSM conv, there is no SiLU here (`lfm2.cpp:207`).
                    let state = &mut cache.conv[li];
                    let mut y = vec![0.0f32; dm];
                    for ch in 0..dm {
                        let bx = b[ch] * x[ch];
                        let mut acc = 0.0f32;
                        for tap in 0..cm1 {
                            acc += conv[ch * rt.l_cache + tap] * state[ch * cm1 + tap];
                        }
                        acc += conv[ch * rt.l_cache + cm1] * bx;
                        // Second gate: elementwise multiply by C (`lfm2.cpp:210`).
                        y[ch] = c[ch] * acc;
                        // Roll the ring left and append this position's input.
                        for tap in 0..cm1.saturating_sub(1) {
                            state[ch * cm1 + tap] = state[ch * cm1 + tap + 1];
                        }
                        state[ch * cm1 + (cm1 - 1)] = bx;
                    }
                    out_proj.par_matvec(&y, &tn("shortconv.out_proj"))?
                }
                Lfm2Kind::Attn {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let mut qp = wq.par_matvec(&xn, &tn("attn_q"))?;
                    let mut kp = wk.par_matvec(&xn, &tn("attn_k"))?;
                    let vp = wv.par_matvec(&xn, &tn("attn_v"))?;
                    // QK RMSNorm over head_dim, BEFORE RoPE (`lfm2.cpp:137-146`).
                    self.apply_norm_heads(&mut qp, self.n_heads, hd, q_norm);
                    self.apply_norm_heads(&mut kp, self.n_kv_heads, hd, k_norm);
                    let rb = self.layer_rope_base[li];
                    self.apply_rope(&mut qp, self.n_heads, pos, rb);
                    self.apply_rope(&mut kp, self.n_kv_heads, pos, rb);

                    cache.k[li].extend_from_slice(&kp);
                    cache.v[li].extend_from_slice(&vp);
                    let ck = &cache.k[li];
                    let cv = &cache.v[li];
                    // Attention layers are full-causal: LFM2.5-2.6B declares no
                    // sliding window (`n_swa = 0`), so every cached position is
                    // visible. Derived from the cache rather than `pos` because
                    // conv layers do not advance the K/V cache, so an
                    // attention layer's cache depth is the only truth here.
                    let n_pos = ck.len() / kv_dim;

                    let mut attn_out = vec![0.0f32; q_dim];
                    for h in 0..self.n_heads {
                        let kvh = h / group;
                        let qh = &qp[h * hd..(h + 1) * hd];
                        let mut scores = vec![0.0f32; n_pos];
                        let mut mx = f32::NEG_INFINITY;
                        for (j, sj) in scores.iter_mut().enumerate() {
                            let kh = &ck[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                            let s = dot(qh, kh) * scale;
                            *sj = s;
                            if s > mx {
                                mx = s;
                            }
                        }
                        let mut sum = 0.0f32;
                        for s in scores.iter_mut() {
                            *s = (*s - mx).exp();
                            sum += *s;
                        }
                        let oh = &mut attn_out[h * hd..(h + 1) * hd];
                        for (j, s) in scores.iter().enumerate() {
                            let w = *s / sum;
                            let vh = &cv[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                            for d in 0..hd {
                                oh[d] += w * vh[d];
                            }
                        }
                    }
                    wo.par_matvec(&attn_out, &tn("attn_output"))?
                }
            };

            for (h, m) in hidden.iter_mut().zip(mix.iter()) {
                *h += *m;
            }

            // SwiGLU FFN, identical on conv and attention layers.
            let xn2 = self.apply_norm(&hidden, &layer.ffn_norm);
            let g = layer.ffn_gate.par_matvec(&xn2, &tn("ffn_gate"))?;
            let u = layer.ffn_up.par_matvec(&xn2, &tn("ffn_up"))?;
            let act: Vec<f32> = g
                .iter()
                .zip(u.iter())
                .map(|(&gv, &uv)| silu(gv) * uv)
                .collect();
            let d = layer.ffn_down.par_matvec(&act, &tn("ffn_down"))?;
            for (h, dv) in hidden.iter_mut().zip(d.iter()) {
                *h += *dv;
            }
        }

        // Final norm is `token_embd_norm`; the logits matrix is tied to
        // `token_embd` (LFM2.5 ships no `output.weight`). The 128k-row vocab
        // projection dominates a decode step, so take the row-parallel form
        // (same int8-activation caveat as the per-layer projections above).
        if !need_logits {
            return Ok(Vec::new());
        }
        let normed = self.apply_norm(&hidden, &self.output_norm);
        self.output.par_matvec(&normed, "output")
    }

    /// Incremental forward of a single token at absolute `pos`, appending its K/V to
    /// `cache` and attending over all cached positions. Returns next-token logits.
    fn forward_step(&self, token: u32, pos: usize, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.forward_step_maybe_logits(token, pos, cache, true)
    }

    /// [`forward_step`] with the LM head made optional. Prompt positions that only
    /// populate KV state skip the vocabulary projection; the final prompt position
    /// and every decode position still return logits.
    ///
    /// [`forward_step`]: RunnableModel::forward_step
    fn forward_step_maybe_logits(
        &self,
        token: u32,
        pos: usize,
        cache: &mut KvCache,
        need_logits: bool,
    ) -> Result<Vec<f32>> {
        if let Some(rt) = &self.lfm2 {
            return self.forward_step_lfm2(rt, token, pos, cache, need_logits);
        }
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = self.n_heads / self.n_kv_heads;
        let q_dim = self.n_heads * hd;
        let kv_dim = self.n_kv_heads * hd;

        let t = token as usize;
        if t >= self.vocab {
            return Err(BackendError::InvalidTensorData(format!(
                "token id {t} >= vocab {}",
                self.vocab
            )));
        }
        let mut hidden = self.token_embd.dequant_row(t, "token_embd")?;
        if let Some(s) = self.embed_scale {
            for v in hidden.iter_mut() {
                *v *= s;
            }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention (single query position over cached K/V) ---
            let is_command_r = self.architecture == "command-r";
            let command_r_residual = is_command_r.then(|| hidden.clone());
            let xn = self.apply_norm(&hidden, &layer.attn_norm);
            let q_input = self.apply_optional_norm(&xn, layer.q_norm_in.as_ref());
            let k_input = self.apply_optional_norm(&xn, layer.k_norm_in.as_ref());
            let v_input = self.apply_optional_norm(&xn, layer.v_norm_in.as_ref());
            let mut qp = layer.wq.projection_matvec(&q_input, &name(li, "attn_q"))?;
            let mut kp = layer.wk.projection_matvec(&k_input, &name(li, "attn_k"))?;
            let vp = layer.wv.projection_matvec(&v_input, &name(li, "attn_v"))?;
            if let Some(qn) = &layer.q_norm {
                self.apply_norm_heads(&mut qp, self.n_heads, hd, qn);
            }
            if let Some(kn) = &layer.k_norm {
                self.apply_norm_heads(&mut kp, self.n_kv_heads, hd, kn);
            }
            let rb = self.layer_rope_base[li];
            self.apply_rope(&mut qp, self.n_heads, pos, rb);
            self.apply_rope(&mut kp, self.n_kv_heads, pos, rb);
            cache.k[li].extend_from_slice(&kp);
            cache.v[li].extend_from_slice(&vp);
            let ck = &cache.k[li];
            let cv = &cache.v[li];
            let n_pos = pos + 1;

            let mut attn_out = vec![0.0f32; q_dim];
            for h in 0..self.n_heads {
                let kvh = h / group;
                let qh = &qp[h * hd..(h + 1) * hd];
                let mut scores = vec![0.0f32; n_pos];
                let mut mx = f32::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate() {
                    let kh = &ck[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                    let mut s = dot(qh, kh) * scale;
                    if let Some(cap) = self.attn_logit_softcap {
                        s = cap * (s / cap).tanh();
                    }
                    *sj = s;
                    if s > mx {
                        mx = s;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let oh = &mut attn_out[h * hd..(h + 1) * hd];
                for (j, s) in scores.iter().enumerate() {
                    let w = *s / sum;
                    let vh = &cv[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd {
                        oh[d] += w * vh[d];
                    }
                }
            }
            if let Some(norm) = &layer.attn_sub_norm {
                attn_out = self.apply_norm(&attn_out, norm);
            }
            let output_input = self.apply_optional_norm(&attn_out, layer.output_norm_in.as_ref());
            let mut proj = layer
                .wo
                .projection_matvec(&output_input, &name(li, "attn_output"))?;
            if let Some(pn) = &layer.post_attn_norm {
                proj = self.apply_norm(&proj, pn);
            }
            if !is_command_r {
                for (h, p) in hidden.iter_mut().zip(proj.iter()) {
                    *h += *p;
                }
            }

            // --- FFN ---
            // Command R's FFN is parallel with attention: both consume the SAME
            // LayerNorm result. Every other generic architecture keeps the
            // established sequential second-norm path.
            let xn2 = if is_command_r {
                xn
            } else {
                self.apply_norm(&hidden, &layer.ffn_norm)
            };
            let gate_input = self.apply_optional_norm(&xn2, layer.gate_norm_in.as_ref());
            let up_input = self.apply_optional_norm(&xn2, layer.up_norm_in.as_ref());
            let g = layer
                .gate
                .projection_matvec(&gate_input, &name(li, "ffn_gate"))?;
            let u = layer.up.projection_matvec(&up_input, &name(li, "ffn_up"))?;
            let mut act = vec![0.0f32; g.len()];
            for i in 0..g.len() {
                let gated = if self.architecture == "bitnet-b1.58" {
                    // BitNet-b1.58-2B-4T is trained with `hidden_act = relu2`:
                    // ReLU(gate)^2, multiplied by the parallel up projection.
                    // The older BitNet family used SiLU, but applying that graph
                    // to Microsoft's 2B-4T row produces repetitive garbage.
                    g[i].max(0.0).powi(2)
                } else if self.ffn_gelu {
                    gelu_tanh(g[i])
                } else {
                    g[i] / (1.0 + (-g[i]).exp())
                };
                act[i] = gated * u[i];
            }
            if let Some(norm) = &layer.ffn_sub_norm {
                act = self.apply_norm(&act, norm);
            }
            let down_input = self.apply_optional_norm(&act, layer.down_norm_in.as_ref());
            let mut d = layer
                .down
                .projection_matvec(&down_input, &name(li, "ffn_down"))?;
            if let Some(pn) = &layer.post_ffn_norm {
                d = self.apply_norm(&d, pn);
            }
            if let Some(residual) = command_r_residual.as_deref() {
                command_r_parallel_residual(&mut hidden, residual, &d, &proj);
            } else {
                for (h, dv) in hidden.iter_mut().zip(d.iter()) {
                    *h += *dv;
                }
            }
            // gemma3→CUDA campaign localization instrument (see `forward_logits`).
            // This is the KV-cached step the runnable SERVE lane actually runs, so
            // it is the trace that lines up with the resident lanes' per-token
            // forward. Off unless `CAMELID_LAYER_DUMP` is set.
            if let Some(path) = std::env::var("CAMELID_LAYER_DUMP").ok().as_deref() {
                let l2 = hidden.iter().map(|v| v * v).sum::<f32>().sqrt();
                // Keyed on POSITION so this lines up with the resident lanes'
                // traces regardless of load-time warmup forwards.
                let line = format!(
                    "{pos}\t{li}\t{l2:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
                    hidden[0], hidden[1], hidden[2], hidden[3]
                );
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }

        if !need_logits {
            return Ok(Vec::new());
        }
        let normed = self.apply_norm(&hidden, &self.output_norm);
        self.project_output_logits(&normed)
    }

    fn attention_block(
        &self,
        layer: &Layer,
        li: usize,
        hidden: &mut [f32],
        seq: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<()> {
        let dm = self.d_model;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = self.n_heads / self.n_kv_heads;
        let q_dim = self.n_heads * hd;
        let kv_dim = self.n_kv_heads * hd;

        let mut q = vec![0.0f32; seq * q_dim];
        let mut k = vec![0.0f32; seq * kv_dim];
        let mut v = vec![0.0f32; seq * kv_dim];
        let mut q_inputs = Vec::with_capacity(seq);
        let mut k_inputs = Vec::with_capacity(seq);
        let mut v_inputs = Vec::with_capacity(seq);
        for pos in 0..seq {
            let x = &hidden[pos * dm..(pos + 1) * dm];
            let xn = self.apply_norm(x, &layer.attn_norm);
            q_inputs.push(self.apply_optional_norm(&xn, layer.q_norm_in.as_ref()));
            k_inputs.push(self.apply_optional_norm(&xn, layer.k_norm_in.as_ref()));
            v_inputs.push(self.apply_optional_norm(&xn, layer.v_norm_in.as_ref()));
        }
        let q_rows = layer.wq.projection_matmul(&q_inputs, &name(li, "attn_q"))?;
        let k_rows = layer.wk.projection_matmul(&k_inputs, &name(li, "attn_k"))?;
        let v_rows = layer.wv.projection_matmul(&v_inputs, &name(li, "attn_v"))?;
        for pos in 0..seq {
            let mut qp = q_rows[pos].clone();
            let mut kp = k_rows[pos].clone();
            let vp = &v_rows[pos];
            // QK-norm (qwen3, gemma3): per-head RMSNorm before RoPE.
            if let Some(qn) = &layer.q_norm {
                self.apply_norm_heads(&mut qp, self.n_heads, hd, qn);
            }
            if let Some(kn) = &layer.k_norm {
                self.apply_norm_heads(&mut kp, self.n_kv_heads, hd, kn);
            }
            let rope_base = self.layer_rope_base[li];
            self.apply_rope(&mut qp, self.n_heads, pos, rope_base);
            self.apply_rope(&mut kp, self.n_kv_heads, pos, rope_base);
            q[pos * q_dim..(pos + 1) * q_dim].copy_from_slice(&qp);
            k[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&kp);
            v[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(vp);
        }

        let mut output_inputs = Vec::with_capacity(seq);
        for pos in 0..seq {
            let mut attn_out = vec![0.0f32; q_dim];
            for h in 0..self.n_heads {
                let kvh = h / group;
                let qh = &q[pos * q_dim + h * hd..pos * q_dim + (h + 1) * hd];
                let mut scores = vec![0.0f32; pos + 1];
                let mut max = f32::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate() {
                    let kh = &k[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                    let mut s = dot(qh, kh) * scale;
                    // gemma2 attention logit soft-cap (gemma3: None).
                    if let Some(cap) = self.attn_logit_softcap {
                        s = cap * (s / cap).tanh();
                    }
                    *sj = s;
                    if *sj > max {
                        max = *sj;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let oh = &mut attn_out[h * hd..(h + 1) * hd];
                for (j, s) in scores.iter().enumerate() {
                    let w = *s / sum;
                    let vh = &v[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                    for d in 0..hd {
                        oh[d] += w * vh[d];
                    }
                }
            }
            if let Some(norm) = &layer.attn_sub_norm {
                attn_out = self.apply_norm(&attn_out, norm);
            }
            output_inputs.push(self.apply_optional_norm(&attn_out, layer.output_norm_in.as_ref()));
        }
        let projections = layer
            .wo
            .projection_matmul(&output_inputs, &name(li, "attn_output"))?;
        for (pos, projection) in projections.iter().enumerate() {
            let mut proj = projection.clone();
            // gemma: post-attention RMSNorm before the residual add.
            if let Some(pn) = &layer.post_attn_norm {
                proj = self.apply_norm(&proj, pn);
            }
            let dst = &mut hidden[pos * dm..(pos + 1) * dm];
            for (h, p) in dst.iter_mut().zip(proj.iter()) {
                *h += *p;
            }
        }
        if let Some(cache) = cache {
            cache.k[li] = k;
            cache.v[li] = v;
        }
        Ok(())
    }

    fn ffn_block(&self, layer: &Layer, li: usize, hidden: &mut [f32], seq: usize) -> Result<()> {
        let dm = self.d_model;
        let mut gate_inputs = Vec::with_capacity(seq);
        let mut up_inputs = Vec::with_capacity(seq);
        for pos in 0..seq {
            let x = &hidden[pos * dm..(pos + 1) * dm];
            let xn = self.apply_norm(x, &layer.ffn_norm);
            gate_inputs.push(self.apply_optional_norm(&xn, layer.gate_norm_in.as_ref()));
            up_inputs.push(self.apply_optional_norm(&xn, layer.up_norm_in.as_ref()));
        }
        let gate_rows = layer
            .gate
            .projection_matmul(&gate_inputs, &name(li, "ffn_gate"))?;
        let up_rows = layer
            .up
            .projection_matmul(&up_inputs, &name(li, "ffn_up"))?;
        let mut down_inputs = Vec::with_capacity(seq);
        for pos in 0..seq {
            let g = &gate_rows[pos];
            let u = &up_rows[pos];
            // Gated FFN: gemma uses GeGLU (gelu-tanh), llama uses SwiGLU (silu).
            let mut act = vec![0.0f32; g.len()];
            for i in 0..g.len() {
                let gated = if self.architecture == "bitnet-b1.58" {
                    // Exact Microsoft BitNet 2B-4T FFN activation. Keep this
                    // identical to the incremental path above.
                    g[i].max(0.0).powi(2)
                } else if self.ffn_gelu {
                    gelu_tanh(g[i])
                } else {
                    g[i] / (1.0 + (-g[i]).exp())
                };
                act[i] = gated * u[i];
            }
            if let Some(norm) = &layer.ffn_sub_norm {
                act = self.apply_norm(&act, norm);
            }
            down_inputs.push(self.apply_optional_norm(&act, layer.down_norm_in.as_ref()));
        }
        let down_rows = layer
            .down
            .projection_matmul(&down_inputs, &name(li, "ffn_down"))?;
        for (pos, down_row) in down_rows.iter().enumerate() {
            let mut d = down_row.clone();
            // gemma: post-FFN RMSNorm before the residual add.
            if let Some(pn) = &layer.post_ffn_norm {
                d = self.apply_norm(&d, pn);
            }
            let dst = &mut hidden[pos * dm..(pos + 1) * dm];
            for (hv, dv) in dst.iter_mut().zip(d.iter()) {
                *hv += *dv;
            }
        }
        Ok(())
    }

    /// RoPE in place over `n_heads` heads of `head_dim`, rotating the first
    /// `rope_dim` dims at absolute position `pos`. Adjacent even/odd pairing for
    /// llama (`rope_neox=false`); split-half (NEOX) for `rope_neox=true`.
    fn apply_rope(&self, vec: &mut [f32], n_heads: usize, pos: usize, rope_base: f32) {
        let hd = self.head_dim;
        let half = self.rope_dim / 2;
        for h in 0..n_heads {
            let base = h * hd;
            for i in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * i as f32 / self.rope_dim as f32);
                let angle = pos as f32 * self.rope_freq_scale * freq;
                let (sin, cos) = angle.sin_cos();
                let (a, b) = if self.rope_neox {
                    (base + i, base + i + half)
                } else {
                    (base + 2 * i, base + 2 * i + 1)
                };
                let x0 = vec[a];
                let x1 = vec[b];
                vec[a] = x0 * cos - x1 * sin;
                vec[b] = x0 * sin + x1 * cos;
            }
        }
    }
}

// ===================================================================================
// Qwen3.5 (Ornith) — hybrid gated-delta-net (linear attention) + full attention lane.
//
// Faithful re-implementation of llama.cpp's `qwen35` graph (arch string "qwen35",
// `src/models/qwen35.cpp` + `delta-net-base.cpp`) in pure f32 as the runnable lane's
// CPU reference and fallback. On macOS, decode defaults to the full Metal resident
// graph built from these same structs (`generate_qwen35_metal`; opt-out
// `CAMELID_QWEN35_METAL=0`); the pure-f32 forward below remains the oracle.
// The runnable lane decodes one token at a time, so the gated-delta-net AUTOREGRESSIVE
// recurrence covers both prefill and decode (the batched "chunking" path is never
// needed). Each layer is either:
//   * a recurrent (SSM) layer  — conv1d + SiLU → L2-normed q/k, raw v → per-head gated
//     delta-rule state recurrence → gated RMSNorm → out-projection; OR
//   * a full-attention layer   — fused query+gate projection, q/k RMSNorm, partial NEOX
//     RoPE (64 of 256 dims), GQA causal attention, sigmoid output gate, out-projection.
// Both share a standard pre-norm 2-norm block (attn_norm pre-mix, post_attention_norm
// pre-FFN, SwiGLU FFN), each with its own residual.
// ===================================================================================

/// One Qwen3.5 layer's mixing sub-block: either full attention or a gated-delta-net
/// (SSM) recurrence. The surrounding norms + FFN live on [`Qwen35Layer`].
enum Qwen35Kind {
    Full {
        /// Fused query + output gate: out-features = `head_dim * n_head * 2`,
        /// interleaved per head ([query(head_dim) | gate(head_dim)] × n_head).
        wq: RawMat,
        wk: RawMat,
        wv: RawMat,
        wo: RawMat,
        q_norm: Vec<f32>, // per-head RMSNorm weight [head_dim]
        k_norm: Vec<f32>,
    },
    Ssm {
        wqkv: RawMat,      // out = conv_dim = 2*key_dim + value_dim (mixed q|k|v)
        wqkv_gate: RawMat, // out = value_dim (the output gate `z`)
        /// ggml `ssm_conv1d.weight` [d_conv, conv_dim], flat: `[c*d_conv + tap]`.
        conv1d: Vec<f32>,
        dt_bias: Vec<f32>,  // [num_v_heads] (ssm_dt.bias)
        a: Vec<f32>,        // [num_v_heads] = -exp(A_log) (ssm_a, no .weight suffix)
        beta: RawMat,       // out = num_v_heads
        alpha: RawMat,      // out = num_v_heads
        ssm_norm: Vec<f32>, // gated RMSNorm weight [head_v_dim]
        ssm_out: RawMat,    // in = value_dim, out = n_embd
    },
}

struct Qwen35Layer {
    attn_norm: Vec<f32>,      // pre-mix RMSNorm
    post_attn_norm: Vec<f32>, // pre-FFN RMSNorm (GGUF `post_attention_norm`)
    ffn_gate: RawMat,
    ffn_up: RawMat,
    ffn_down: RawMat,
    kind: Qwen35Kind,
}

/// Parsed Qwen3.5 runtime: per-layer weights + the gated-delta-net dims.
struct Qwen35Runtime {
    layers: Vec<Qwen35Layer>,
    /// Interleaved multimodal RoPE pair counts `[time, height, width, extra]`.
    /// Text uses one position in every section; image embeddings supply a 2-D
    /// grid through the same full-attention layers.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    rope_sections: [usize; 4],
    d_conv: usize,      // causal conv kernel width (4)
    d_state: usize,     // per-head state dim = head_k_dim = head_v_dim (128)
    num_k_heads: usize, // key/query heads / groups (16)
    num_v_heads: usize, // value/delta heads (32)
    head_v_dim: usize,  // d_inner / num_v_heads (= d_state, 128)
    key_dim: usize,     // d_state * num_k_heads (2048)
    value_dim: usize,   // head_v_dim * num_v_heads (= d_inner, 4096)
    conv_dim: usize,    // 2*key_dim + value_dim (8192)
}

/// One LFM2 block: either a double-gated short convolution or a GQA attention
/// mix, followed in both cases by the same SwiGLU FFN.
enum Lfm2Kind {
    /// Short-conv layer. `conv` is the depthwise kernel flattened
    /// `[c*l_cache + t]` (channel `c`, tap `t`); `in_proj` emits `3*n_embd`
    /// split as `B | C | x`; `out_proj` maps back to `n_embd`.
    Conv {
        conv: Vec<f32>,
        in_proj: RawMat,
        out_proj: RawMat,
    },
    /// GQA attention layer with per-head-dim QK RMSNorm applied BEFORE RoPE
    /// (`lfm2.cpp:137-146`).
    Attn {
        wq: RawMat,
        wk: RawMat,
        wv: RawMat,
        wo: RawMat,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

struct Lfm2Layer {
    /// LFM2's `operator_norm`: pre-block RMSNorm on conv AND attention layers.
    attn_norm: Vec<f32>,
    /// Pre-FFN RMSNorm.
    ffn_norm: Vec<f32>,
    ffn_gate: RawMat,
    ffn_up: RawMat,
    ffn_down: RawMat,
    kind: Lfm2Kind,
}

/// Parsed LFM2 runtime: per-layer weights + the short-conv kernel width.
struct Lfm2Runtime {
    layers: Vec<Lfm2Layer>,
    /// Short-conv kernel width (`lfm2.shortconv.l_cache`, 3). The rolling conv
    /// state is `l_cache - 1` wide.
    l_cache: usize,
}

/// Per-layer incremental state for qwen35 decode. Full-attention layers grow a
/// standard K/V cache; SSM layers keep a causal-conv ring buffer and the recurrent
/// per-head state matrix (`num_v_heads` × `d_state` × `d_state`).
struct Qwen35Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Conv ring buffer per SSM layer: `(d_conv-1) * conv_dim`, layout
    /// `[c*(d_conv-1) + t]`, `t=0` oldest. Empty for full-attention layers.
    conv: Vec<Vec<f32>>,
    /// Recurrent state per SSM layer: `num_v_heads * d_state * d_state`, per head a
    /// `d_state×d_state` matrix `S[i*d_state + j]` with `i`=key, `j`=value. Empty
    /// for full-attention layers.
    state: Vec<Vec<f32>>,
}

impl Qwen35Cache {
    fn new(rt: &Qwen35Runtime, n_layers: usize) -> Self {
        let mut conv = vec![Vec::new(); n_layers];
        let mut state = vec![Vec::new(); n_layers];
        for (li, layer) in rt.layers.iter().enumerate() {
            if matches!(layer.kind, Qwen35Kind::Ssm { .. }) {
                conv[li] = vec![0.0f32; (rt.d_conv - 1) * rt.conv_dim];
                state[li] = vec![0.0f32; rt.num_v_heads * rt.d_state * rt.d_state];
            }
        }
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
            conv,
            state,
        }
    }
}

impl RunnableModel {
    /// Stateless whole-sequence forward for the smoke gate: scan all positions and
    /// return the last position's logits. Mirrors [`generate_qwen35`] step-for-step.
    ///
    /// [`generate_qwen35`]: RunnableModel::generate_qwen35
    fn forward_logits_qwen35(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let rt = self.qwen35.as_ref().expect("qwen35 runtime present");
        let _ = rt;
        let (_cache, logits) = self.prefill_qwen35(tokens)?;
        Ok(logits)
    }

    /// Batched prompt prefill: process ALL prompt positions through the stack, reading
    /// each weight once per layer (`par_matmul`) instead of once per token — the
    /// memory-bandwidth amortization that makes the prompt fast. Builds `cache` (KV for
    /// full-attn layers; conv + recurrent state for SSM layers) identically to running
    /// `decode_token_qwen35` over the prompt — causal attention means each position
    /// only depends on earlier ones, so batching by layer is bit-identical to the
    /// per-token order — and returns the LAST position's logits.
    /// Item 5 (acceptance economics) harness: teacher-forced greedy argmax at
    /// EVERY position of `tokens` — out[i] = argmax of the logits after
    /// consuming tokens[0..=i] (the model's greedy prediction for position
    /// i+1). CPU path runs the per-token decode with the LM head at each step;
    /// with `CAMELID_QWEN35_CUDA=1` the resident engine computes the same
    /// stream on the GPU. Fresh SSM/KV state per call.
    // Only the env-driven #[ignore] harness tests call this — dead code on the
    // plain lib target.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn qwen35_argmax_stream(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "cuda")]
        {
            if std::env::var("CAMELID_QWEN35_CUDA")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                return self.qwen35_argmax_stream_cuda(tokens);
            }
        }
        let mut cache = Qwen35Cache::new(
            self.qwen35.as_ref().expect("qwen35 runtime present"),
            self.n_layers,
        );
        let mut out = Vec::with_capacity(tokens.len());
        for (pos, &tok) in tokens.iter().enumerate() {
            let logits = self.decode_token_qwen35(tok, pos, &mut cache, true)?;
            out.push(argmax(&logits));
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    #[cfg_attr(not(test), allow(dead_code))]
    fn qwen35_argmax_stream_cuda(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        let max_pos: usize = std::env::var("CAMELID_QWEN35_CUDA_MAXPOS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let mut guard = self
            .cuda
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("qwen35 cuda mutex poisoned".into()))?;
        if guard.is_none() {
            let e = self
                .build_qwen35_resident(max_pos)
                .map_err(BackendError::InvalidTensorData)?;
            *guard = Some(e);
        }
        let engine = guard.as_mut().unwrap();
        engine
            .reset_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let mut out = Vec::with_capacity(tokens.len());
        for (pos, &tok) in tokens.iter().enumerate() {
            let emb = self.token_embd.dequant_row(tok as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
            let next = engine
                .forward_token(&emb, &cos, &sin, pos, scale, true)
                .map_err(BackendError::InvalidTensorData)?
                .ok_or_else(|| {
                    BackendError::InvalidTensorData("no logits on argmax-stream step".into())
                })?;
            out.push(next);
        }
        Ok(out)
    }

    /// Item 5 harness: run the batched CPU prefill over `tokens` and return the
    /// wall-clock seconds (the marginal cost between two prefix lengths is the
    /// batched k-token verify cost).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn qwen35_prefill_timed(&self, tokens: &[u32]) -> Result<f64> {
        let started = std::time::Instant::now();
        let (_cache, _logits) = self.prefill_qwen35(tokens)?;
        Ok(started.elapsed().as_secs_f64())
    }

    fn prefill_qwen35(&self, prompt: &[u32]) -> Result<(Qwen35Cache, Vec<f32>)> {
        let rt = self.qwen35.as_ref().expect("qwen35 runtime present");
        let m = prompt.len();
        let mut cache = Qwen35Cache::new(rt, self.n_layers);
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(m);
        for &tok in prompt {
            let t = tok as usize;
            if t >= self.vocab {
                return Err(BackendError::InvalidTensorData(format!(
                    "token id {t} >= vocab {}",
                    self.vocab
                )));
            }
            hidden.push(self.token_embd.dequant_row(t, "token_embd")?);
        }

        for (li, layer) in rt.layers.iter().enumerate() {
            let xn: Vec<Vec<f32>> = hidden
                .iter()
                .map(|h| self.apply_norm(h, &layer.attn_norm))
                .collect();
            let mix: Vec<Vec<f32>> = match &layer.kind {
                Qwen35Kind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let qg = wq.par_matmul(&xn)?;
                    let k = wk.par_matmul(&xn)?;
                    let v = wv.par_matmul(&xn)?;
                    let mut attn_outs = Vec::with_capacity(m);
                    for p in 0..m {
                        attn_outs.push(self.qwen35_attn_compute(
                            q_norm, k_norm, &qg[p], &k[p], &v[p], p, li, &mut cache,
                        ));
                    }
                    wo.par_matmul(&attn_outs)?
                }
                Qwen35Kind::Ssm {
                    wqkv,
                    wqkv_gate,
                    conv1d,
                    dt_bias,
                    a,
                    beta,
                    alpha,
                    ssm_norm,
                    ssm_out,
                } => {
                    let qkv = wqkv.par_matmul(&xn)?;
                    let z = wqkv_gate.par_matmul(&xn)?;
                    let beta_raw = beta.par_matmul(&xn)?;
                    let alpha_raw = alpha.par_matmul(&xn)?;
                    let mut finals = Vec::with_capacity(m);
                    for p in 0..m {
                        finals.push(self.qwen35_ssm_compute(
                            rt,
                            conv1d,
                            dt_bias,
                            a,
                            ssm_norm,
                            li,
                            &qkv[p],
                            &z[p],
                            &beta_raw[p],
                            &alpha_raw[p],
                            &mut cache,
                        ));
                    }
                    ssm_out.par_matmul(&finals)?
                }
            };
            for (h, mp) in hidden.iter_mut().zip(mix.iter()) {
                for (hv, mv) in h.iter_mut().zip(mp.iter()) {
                    *hv += *mv;
                }
            }

            // FFN (SwiGLU), batched, pre-normed by post_attention_norm.
            let xn2: Vec<Vec<f32>> = hidden
                .iter()
                .map(|h| self.apply_norm(h, &layer.post_attn_norm))
                .collect();
            let g = layer.ffn_gate.par_matmul(&xn2)?;
            let u = layer.ffn_up.par_matmul(&xn2)?;
            let act: Vec<Vec<f32>> = g
                .iter()
                .zip(u.iter())
                .map(|(gp, up)| {
                    gp.iter()
                        .zip(up.iter())
                        .map(|(&gv, &uv)| silu(gv) * uv)
                        .collect()
                })
                .collect();
            let d = layer.ffn_down.par_matmul(&act)?;
            for (h, dp) in hidden.iter_mut().zip(d.iter()) {
                for (hv, dv) in h.iter_mut().zip(dp.iter()) {
                    *hv += *dv;
                }
            }
        }

        let normed = self.apply_norm(&hidden[m - 1], &self.output_norm);
        let logits = self.output.par_matvec(&normed, "output")?;
        Ok((cache, logits))
    }

    /// Greedy decode for qwen35: prefill the prompt position-by-position into the
    /// hybrid cache, then argmax-extend. Bit-identical to [`forward_logits_qwen35`]
    /// for the shared prefix (same per-token math, same accumulation order).
    ///
    /// [`forward_logits_qwen35`]: RunnableModel::forward_logits_qwen35
    /// qwen35 greedy decode. On macOS, routes to the full Metal resident graph by
    /// default (opt-out `CAMELID_QWEN35_METAL=0` via [`qwen35_metal_enabled`]), falling
    /// back to the CPU hybrid lane on any Metal error. With the `cuda` feature, routes
    /// to the CUDA resident lane when `CAMELID_QWEN35_CUDA=1` (default-on for Prism
    /// low-bit rows on Windows; lazy-built, reused, recurrent state reset per call),
    /// falling back to the CPU runnable lane on any CUDA error. The CPU lane is the
    /// certified oracle, and the default only where neither GPU lane applies.
    fn generate_qwen35(&self, prompt: &[u32], max_new: usize, stop: &[u32]) -> Result<Vec<u32>> {
        self.generate_qwen35_streaming(prompt, max_new, stop, None, false, &mut |_| {})
    }

    /// Like [`generate_qwen35`](Self::generate_qwen35) but invokes `on_token` for
    /// every emitted token as soon as it is decided — the serve lane's SSE source.
    /// Token order/content identical to the non-streaming path by construction.
    fn generate_qwen35_streaming(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        stream_tokens_observable: bool,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        #[cfg(not(feature = "cuda"))]
        let _ = stream_tokens_observable;

        #[cfg(target_os = "macos")]
        if qwen35_metal_enabled() {
            match self.generate_qwen35_metal(prompt, max_new, stop, sampling, on_token) {
                Ok(tokens) => return Ok(tokens),
                Err(err) => {
                    eprintln!("[qwen35] resident Metal lane failed ({err}); using hybrid fallback");
                }
            }
        }
        #[cfg(feature = "cuda")]
        {
            // SINGLE SOURCE OF TRUTH with the disclosed execution plan. This lane used
            // to read `CAMELID_QWEN35_CUDA` itself and default ON only for Prism low-bit
            // rows on Windows, so a certified Ornith K-quant row (qwen35 Q4_K_M) served
            // out of the box reported `cuda_resident_kquant_runtime` from
            // `select_kquant_plan` while decoding here on the CPU — measured 0.42 tok/s
            // against 6.1 tok/s on the very same row and host. That is the gemma4 Phase 0
            // defect: the plan and the lane consulted different things. `--gpu off` and
            // the UI toggle stay authoritative because the policy requires
            // `gpu_accel_enabled()`; `CAMELID_QWEN35_CUDA=0` still forces the CPU oracle.
            let cuda_enabled = crate::execution_plan::qwen35_cuda_lane_selectable();
            if cuda_enabled {
                return qwen35_cuda_with_cpu_fallback(
                    on_token,
                    stream_tokens_observable,
                    |tracked_on_token| {
                        self.generate_qwen35_cuda(prompt, max_new, stop, sampling, tracked_on_token)
                    },
                    |fallback_on_token| {
                        self.generate_qwen35_cpu(prompt, max_new, stop, sampling, fallback_on_token)
                    },
                );
            }
        }
        self.generate_qwen35_cpu(prompt, max_new, stop, sampling, on_token)
    }

    #[cfg(target_os = "macos")]
    fn generate_qwen35_metal(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let max_positions = qwen35_metal_context_capacity();
        let max_new = qwen35_generation_budget(prompt.len(), max_new, max_positions, "Metal text")?;
        let mut guard = self
            .metal_qwen35
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("qwen35 Metal mutex poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(self.build_qwen35_metal(max_positions)?);
            eprintln!(
                "[qwen35] full Metal resident graph active (packed weights, attention, \
                 gated-delta recurrence, FFN, logits, GPU greedy, and request sampling)"
            );
        }
        let engine = guard.as_mut().expect("Qwen3.5 Metal engine initialized");
        engine.reset();
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = prompt.to_vec();
        let (&last_prompt_token, prior_prompt) = prompt
            .split_last()
            .ok_or_else(|| BackendError::InvalidTensorData("empty prompt".into()))?;
        let mut prefill = Vec::with_capacity(prior_prompt.len());
        for (position, &token) in prior_prompt.iter().enumerate() {
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(position, self.rope_base, self.rope_dim);
            prefill.push((embedding, cos, sin));
        }
        if !engine.forward_prefill_batch(&prefill) {
            return Err(BackendError::InvalidTensorData(format!(
                "Qwen3.5 Metal batched prefill refused {} prompt slots",
                prefill.len()
            )));
        }
        let last_position = prior_prompt.len();
        let embedding = self
            .token_embd
            .dequant_row(last_prompt_token as usize, "token_embd")?;
        let (cos, sin) = qwen35_rope_tables(last_position, self.rope_base, self.rope_dim);
        let mut next = match &sampler {
            Some(sampler) => qwen35_sample_logits(
                engine
                    .forward_logits(&embedding, &cos, &sin, last_position)
                    .ok_or_else(|| {
                        BackendError::InvalidTensorData(format!(
                            "Qwen3.5 Metal forward refused prompt position {last_position}"
                        ))
                    })?,
                sampler,
                &token_history,
            )?,
            None => engine
                .forward_greedy(&embedding, &cos, &sin, last_position)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "Qwen3.5 Metal forward refused prompt position {last_position}"
                    ))
                })?,
        };
        let mut generated = Vec::with_capacity(max_new);
        let mut position = prompt.len();
        for index in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            token_history.push(next);
            on_token(next);
            if qwen35_repetition_loop(&generated) {
                break;
            }
            if index + 1 < max_new {
                let embedding = self.token_embd.dequant_row(next as usize, "token_embd")?;
                let (cos, sin) = qwen35_rope_tables(position, self.rope_base, self.rope_dim);
                next = match &sampler {
                    Some(sampler) => qwen35_sample_logits(
                        engine
                            .forward_logits(&embedding, &cos, &sin, position)
                            .ok_or_else(|| {
                                BackendError::InvalidTensorData(format!(
                                    "Qwen3.5 Metal forward refused decode position {position}"
                                ))
                            })?,
                        sampler,
                        &token_history,
                    )?,
                    None => engine
                        .forward_greedy(&embedding, &cos, &sin, position)
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(format!(
                                "Qwen3.5 Metal forward refused decode position {position}"
                            ))
                        })?,
                };
                position += 1;
            }
        }
        Ok(generated)
    }

    /// Build the resident LFM2 Metal graph. Every guard here is deliberately
    /// independent of `LlamaModelConfig::from_gguf`'s upstream checks: this engine
    /// bakes in LFM2.5's exact shape, so anything it cannot express must refuse
    /// rather than decode with the wrong graph.
    #[cfg(target_os = "macos")]
    fn build_lfm2_metal(&self, max_positions: usize) -> Result<crate::metal::Lfm2MetalDecode> {
        use crate::metal::{
            Lfm2MetalConfig, Lfm2MetalDecode, Lfm2MetalLayerInput, Lfm2MetalLayerKindInput,
        };
        let runtime = self
            .lfm2
            .as_ref()
            .ok_or_else(|| BackendError::InvalidTensorData("not an lfm2 model".to_string()))?;
        let refuse = |why: &str| -> BackendError {
            BackendError::UnsupportedGguf(format!("lfm2 Metal lane: {why}"))
        };
        // Structural facts the encoder assumes. LFM2.5 satisfies all of them; a row
        // that does not gets a typed refusal and stays on the CPU lane.
        if self.embed_scale.is_some() {
            return Err(refuse("embedding scale is not modelled"));
        }
        if self.ffn_gelu {
            return Err(refuse("GeGLU FFN is not modelled (SwiGLU only)"));
        }
        if self.final_logit_softcap.is_some() || self.attn_logit_softcap.is_some() {
            return Err(refuse("logit soft-caps are not modelled"));
        }
        if self.logit_scale.is_some() {
            return Err(refuse("logit scale is not modelled"));
        }
        if !self.rope_neox {
            return Err(refuse("only NEOX rope pairing is encoded"));
        }
        if self.layer_rope_base.iter().any(|b| *b != self.rope_base) {
            return Err(refuse("per-layer rope bases are not modelled"));
        }
        let ffn_dim = runtime
            .layers
            .first()
            .map(|layer| layer.ffn_gate.out_features)
            .ok_or_else(|| refuse("model has no layers"))?;

        let mut layers = Vec::with_capacity(runtime.layers.len());
        for layer in &runtime.layers {
            if layer.ffn_gate.out_features != ffn_dim || layer.ffn_up.out_features != ffn_dim {
                return Err(refuse("non-uniform FFN width"));
            }
            let kind = match &layer.kind {
                Lfm2Kind::Attn {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => Lfm2MetalLayerKindInput::Attn {
                    q: wq.q8_metal_weight()?,
                    k: wk.q8_metal_weight()?,
                    v: wv.q8_metal_weight()?,
                    output: wo.q8_metal_weight()?,
                    q_norm,
                    k_norm,
                },
                Lfm2Kind::Conv {
                    conv,
                    in_proj,
                    out_proj,
                } => Lfm2MetalLayerKindInput::Conv {
                    in_proj: in_proj.q8_metal_weight()?,
                    out_proj: out_proj.q8_metal_weight()?,
                    conv,
                },
            };
            layers.push(Lfm2MetalLayerInput {
                attn_norm: &layer.attn_norm,
                ffn_norm: &layer.ffn_norm,
                ffn_gate: layer.ffn_gate.q8_metal_weight()?,
                ffn_up: layer.ffn_up.q8_metal_weight()?,
                ffn_down: layer.ffn_down.q8_metal_weight()?,
                kind,
            });
        }
        let config = Lfm2MetalConfig {
            hidden: self.d_model,
            ffn_dim,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            l_cache: runtime.l_cache,
            vocab: self.vocab,
            eps: self.eps,
        };
        Lfm2MetalDecode::new(
            config,
            &layers,
            &self.output_norm,
            self.output.q8_metal_weight()?,
            max_positions,
        )
        .ok_or_else(|| refuse("resident graph construction refused these dimensions"))
    }

    /// Greedy/sampled decode on the resident LFM2 Metal graph.
    ///
    /// Prefill is one `step_*` per prompt token here — the same shape the qwen35
    /// Metal lane uses. Batched prefill is a separate, larger change.
    #[cfg(target_os = "macos")]
    fn generate_lfm2_metal(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let max_positions = lfm2_metal_context_capacity();
        if prompt.len() >= max_positions {
            return Err(BackendError::UnsupportedGguf(format!(
                "lfm2 Metal lane: prompt of {} tokens exceeds the resident capacity of \
                 {max_positions} (raise CAMELID_LFM2_METAL_MAXPOS)",
                prompt.len()
            )));
        }
        let mut guard = self
            .metal_lfm2
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("lfm2 Metal mutex poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(self.build_lfm2_metal(max_positions)?);
            eprintln!(
                "[lfm2] resident Metal graph active (Q8_0 weights, short-conv + GQA \
                 attention, FFN, logits, GPU greedy)"
            );
        }
        let engine = guard
            .as_mut()
            .ok_or_else(|| BackendError::InvalidTensorData("lfm2 Metal engine absent".into()))?;
        // The conv ring is order-dependent, so every request starts from a clean state.
        engine.reset();

        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new);

        let embed =
            |t: u32| -> Result<Vec<f32>> { self.token_embd.dequant_row(t as usize, "token_embd") };
        let step = |engine: &mut crate::metal::Lfm2MetalDecode,
                    tok: u32,
                    pos: usize,
                    want_logits: bool|
         -> Result<(Option<u32>, Option<Vec<f32>>)> {
            let h = self.token_embd.dequant_row(tok as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
            if want_logits {
                let logits = engine.step_logits(&h, &cos, &sin, pos).ok_or_else(|| {
                    BackendError::InvalidTensorData("lfm2 Metal step (logits) failed".into())
                })?;
                Ok((None, Some(logits)))
            } else {
                let t = engine.step_greedy(&h, &cos, &sin, pos).ok_or_else(|| {
                    BackendError::InvalidTensorData("lfm2 Metal step (greedy) failed".into())
                })?;
                Ok((Some(t), None))
            }
        };
        let _ = &embed;

        let want_logits = sampler.is_some();
        let mut next: u32;
        {
            // Chunked prefill for everything but the LAST prompt token: each weight
            // streams once per chunk instead of once per token, and the discarded
            // logits never touch the 128k-row head. The final token goes through a
            // normal step so it produces a selection.
            let head = prompt.len() - 1;
            if head > 0 {
                let mut embeds = Vec::with_capacity(head * self.d_model);
                let mut coss = Vec::with_capacity(head * (self.rope_dim / 2));
                let mut sins = Vec::with_capacity(head * (self.rope_dim / 2));
                for (pos, &tok) in prompt.iter().take(head).enumerate() {
                    embeds.extend_from_slice(
                        &self.token_embd.dequant_row(tok as usize, "token_embd")?,
                    );
                    let (cos, sin) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
                    coss.extend_from_slice(&cos);
                    sins.extend_from_slice(&sin);
                }
                engine
                    .prefill_prompt(&embeds, &coss, &sins, head, lfm2_metal_prefill_chunk())
                    .ok_or_else(|| {
                        BackendError::InvalidTensorData("lfm2 Metal prefill failed".into())
                    })?;
            }
            let last = Some(step(engine, prompt[head], head, want_logits)?);
            let (tok, logits) = last.ok_or_else(|| {
                BackendError::InvalidTensorData("lfm2 Metal lane: empty prompt".into())
            })?;
            next = match (&sampler, tok, logits) {
                (Some(s), _, Some(l)) => qwen35_sample_logits(l, s, &token_history)?,
                (None, Some(t), _) => t,
                _ => {
                    return Err(BackendError::InvalidTensorData(
                        "lfm2 Metal lane: selection/logits mismatch".into(),
                    ))
                }
            };
        }

        let mut pos = prompt.len();
        for i in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            token_history.push(next);
            on_token(next);
            if i + 1 < max_new {
                let (tok, logits) = step(engine, next, pos, want_logits)?;
                pos += 1;
                next = match (&sampler, tok, logits) {
                    (Some(s), _, Some(l)) => qwen35_sample_logits(l, s, &token_history)?,
                    (None, Some(t), _) => t,
                    _ => {
                        return Err(BackendError::InvalidTensorData(
                            "lfm2 Metal lane: selection/logits mismatch".into(),
                        ))
                    }
                };
            }
        }
        Ok(out)
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn generate_qwen35_vision_metal(
        &self,
        prefix: &[u32],
        image: &super::vision::PrismVisionEmbedding,
        suffix: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let runtime = self
            .qwen35
            .as_ref()
            .ok_or_else(|| BackendError::UnsupportedGguf("vision requires qwen35".into()))?;
        if image.grid_width == 0
            || image.grid_height == 0
            || image.embeddings.len() != image.grid_width * image.grid_height
            || image
                .embeddings
                .iter()
                .any(|embedding| embedding.len() != self.d_model)
        {
            return Err(BackendError::InvalidTensorData(format!(
                "vision embedding/grid shape is incompatible with Qwen3.5 width {}",
                self.d_model
            )));
        }
        if runtime.rope_sections[1] == 0 || runtime.rope_sections[2] == 0 {
            return Err(BackendError::InvalidModelMetadata(
                "qwen35 row has no height/width multimodal RoPE sections".into(),
            ));
        }
        let decode_cursor = qwen35_vision_decode_cursor(
            prefix.len(),
            image.grid_width,
            image.grid_height,
            suffix.len(),
        );
        let prompt_slots = decode_cursor.kv_position;
        if prompt_slots == 0 {
            return Err(BackendError::InvalidTensorData(
                "empty multimodal prompt".into(),
            ));
        }
        let max_positions = qwen35_metal_context_capacity();
        let max_new = qwen35_generation_budget(prompt_slots, max_new, max_positions, "multimodal")?;

        let mut guard = self
            .metal_qwen35
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("qwen35 Metal mutex poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(self.build_qwen35_metal(max_positions)?);
            eprintln!("[qwen35] multimodal Metal graph active (Prism image embeddings + IMRoPE)");
        }
        let engine = guard.as_mut().expect("Qwen3.5 Metal engine initialized");
        engine.reset();
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = Vec::with_capacity(prefix.len() + suffix.len() + max_new);
        token_history.extend_from_slice(prefix);
        token_history.extend_from_slice(suffix);
        let mut prompt_inputs = Vec::with_capacity(prompt_slots);

        for (logical_position, &token) in prefix.iter().enumerate() {
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(logical_position, self.rope_base, self.rope_dim);
            prompt_inputs.push((embedding, cos, sin));
        }

        let image_position = prefix.len();
        for (index, embedding) in image.embeddings.iter().enumerate() {
            let row = index / image.grid_width;
            let column = index % image.grid_width;
            let positions = [
                image_position,
                image_position + row,
                image_position + column,
                0,
            ];
            let (cos, sin) = qwen35_imrope_tables(
                positions,
                runtime.rope_sections,
                self.rope_base,
                self.rope_dim,
            );
            prompt_inputs.push((embedding.clone(), cos, sin));
        }

        let suffix_logical_start = decode_cursor.rope_position - suffix.len();
        for (suffix_offset, &token) in suffix.iter().enumerate() {
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(
                suffix_logical_start + suffix_offset,
                self.rope_base,
                self.rope_dim,
            );
            prompt_inputs.push((embedding, cos, sin));
        }
        let mut logical_position = decode_cursor.rope_position;

        let (last_input, prior_inputs) = prompt_inputs.split_last().ok_or_else(|| {
            BackendError::InvalidTensorData("multimodal prompt produced no inputs".into())
        })?;
        if !engine.forward_prefill_batch(prior_inputs) {
            return Err(BackendError::InvalidTensorData(format!(
                "Qwen3.5 Metal batched prefill refused {} multimodal slots",
                prior_inputs.len()
            )));
        }
        let final_slot = prior_inputs.len();
        let mut next = match &sampler {
            Some(sampler) => qwen35_sample_logits(
                engine
                    .forward_logits(&last_input.0, &last_input.1, &last_input.2, final_slot)
                    .ok_or_else(|| {
                        BackendError::InvalidTensorData(format!(
                            "Qwen3.5 Metal refused final multimodal slot {final_slot}"
                        ))
                    })?,
                sampler,
                &token_history,
            )?,
            None => engine
                .forward_greedy(&last_input.0, &last_input.1, &last_input.2, final_slot)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "Qwen3.5 Metal refused final multimodal slot {final_slot}"
                    ))
                })?,
        };
        let mut slot = prompt_slots;
        let mut generated = Vec::with_capacity(max_new);
        for index in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            token_history.push(next);
            on_token(next);
            if qwen35_repetition_loop(&generated) {
                break;
            }
            if index + 1 < max_new {
                let embedding = self.token_embd.dequant_row(next as usize, "token_embd")?;
                let (cos, sin) =
                    qwen35_rope_tables(logical_position, self.rope_base, self.rope_dim);
                next = match &sampler {
                    Some(sampler) => qwen35_sample_logits(
                        engine
                            .forward_logits(&embedding, &cos, &sin, slot)
                            .ok_or_else(|| {
                                BackendError::InvalidTensorData(format!(
                                    "Qwen3.5 Metal refused multimodal decode slot {slot}"
                                ))
                            })?,
                        sampler,
                        &token_history,
                    )?,
                    None => engine
                        .forward_greedy(&embedding, &cos, &sin, slot)
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(format!(
                                "Qwen3.5 Metal refused multimodal decode slot {slot}"
                            ))
                        })?,
                };
                logical_position += 1;
                slot += 1;
            }
        }
        Ok(generated)
    }

    /// Windows/Linux multimodal counterpart to the Metal path. The image tower
    /// supplies language-width embeddings and this method feeds them through the
    /// same resident CUDA Qwen3.5 graph as text tokens, with Qwen3-VL's 3-axis
    /// IMRoPE tables and physical KV slots preserved exactly.
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[allow(clippy::too_many_arguments)]
    fn generate_qwen35_vision_cuda(
        &self,
        prefix: &[u32],
        image: &super::vision::PrismVisionEmbedding,
        suffix: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let runtime = self
            .qwen35
            .as_ref()
            .ok_or_else(|| BackendError::UnsupportedGguf("vision requires qwen35".into()))?;
        if image.grid_width == 0
            || image.grid_height == 0
            || image.embeddings.len() != image.grid_width * image.grid_height
            || image
                .embeddings
                .iter()
                .any(|embedding| embedding.len() != self.d_model)
        {
            return Err(BackendError::InvalidTensorData(format!(
                "vision embedding/grid shape is incompatible with Qwen3.5 width {}",
                self.d_model
            )));
        }
        if runtime.rope_sections[1] == 0 || runtime.rope_sections[2] == 0 {
            return Err(BackendError::InvalidModelMetadata(
                "qwen35 row has no height/width multimodal RoPE sections".into(),
            ));
        }
        let decode_cursor = qwen35_vision_decode_cursor(
            prefix.len(),
            image.grid_width,
            image.grid_height,
            suffix.len(),
        );
        let prompt_slots = decode_cursor.kv_position;
        if prompt_slots == 0 {
            return Err(BackendError::InvalidTensorData(
                "empty multimodal prompt".into(),
            ));
        }
        let max_positions: usize = std::env::var("CAMELID_QWEN35_CUDA_MAXPOS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8192);
        let max_new = qwen35_generation_budget(prompt_slots, max_new, max_positions, "multimodal")?;

        let mut guard = self
            .cuda
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("qwen35 cuda mutex poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(
                self.build_qwen35_resident(max_positions)
                    .map_err(BackendError::InvalidTensorData)?,
            );
            eprintln!("[qwen35] multimodal CUDA graph active (Prism image embeddings + IMRoPE)");
        }
        let engine = guard.as_mut().expect("Qwen3.5 CUDA engine initialized");
        engine
            .reset_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = Vec::with_capacity(prefix.len() + suffix.len() + max_new);
        token_history.extend_from_slice(prefix);
        token_history.extend_from_slice(suffix);
        let mut prompt_inputs = Vec::with_capacity(prompt_slots);

        for (logical_position, &token) in prefix.iter().enumerate() {
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(logical_position, self.rope_base, self.rope_dim);
            prompt_inputs.push((embedding, cos, sin));
        }
        let image_position = prefix.len();
        for (index, embedding) in image.embeddings.iter().enumerate() {
            let row = index / image.grid_width;
            let column = index % image.grid_width;
            let positions = [
                image_position,
                image_position + row,
                image_position + column,
                0,
            ];
            let (cos, sin) = qwen35_imrope_tables(
                positions,
                runtime.rope_sections,
                self.rope_base,
                self.rope_dim,
            );
            prompt_inputs.push((embedding.clone(), cos, sin));
        }
        let suffix_logical_start = decode_cursor.rope_position - suffix.len();
        for (suffix_offset, &token) in suffix.iter().enumerate() {
            let embedding = self.token_embd.dequant_row(token as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(
                suffix_logical_start + suffix_offset,
                self.rope_base,
                self.rope_dim,
            );
            prompt_inputs.push((embedding, cos, sin));
        }
        let mut logical_position = decode_cursor.rope_position;

        let (last_input, prior_inputs) = prompt_inputs.split_last().ok_or_else(|| {
            BackendError::InvalidTensorData("multimodal prompt produced no inputs".into())
        })?;
        if !prior_inputs.is_empty() {
            // Match the Metal optimization pass: enqueue every non-final slot in
            // physical-order and synchronize once after the burst. `prefill` runs
            // the exact same token graph and preserves recurrent/KV dependencies on
            // the CUDA stream; it only removes the per-slot WDDM round trip that made
            // a full-resolution image prompt dramatically slower than Metal.
            let half = self.rope_dim / 2;
            let mut embeddings = Vec::with_capacity(prior_inputs.len() * self.d_model);
            let mut cos_all = Vec::with_capacity(prior_inputs.len() * half);
            let mut sin_all = Vec::with_capacity(prior_inputs.len() * half);
            for (embedding, cos, sin) in prior_inputs {
                embeddings.extend_from_slice(embedding);
                cos_all.extend_from_slice(cos);
                sin_all.extend_from_slice(sin);
            }
            if engine.prefers_batched_prefill() {
                engine
                    .prefill_batched(&embeddings, &cos_all, &sin_all, prior_inputs.len(), scale)
                    .map_err(BackendError::InvalidTensorData)?;
            } else {
                engine
                    .prefill(&embeddings, &cos_all, &sin_all, prior_inputs.len(), scale)
                    .map_err(BackendError::InvalidTensorData)?;
            }
        }
        let final_slot = prior_inputs.len();
        let mut next = match &sampler {
            Some(sampler) => qwen35_sample_logits(
                engine
                    .forward_token_logits(
                        &last_input.0,
                        &last_input.1,
                        &last_input.2,
                        final_slot,
                        scale,
                    )
                    .map_err(BackendError::InvalidTensorData)?,
                sampler,
                &token_history,
            )?,
            None => engine
                .forward_token(
                    &last_input.0,
                    &last_input.1,
                    &last_input.2,
                    final_slot,
                    scale,
                    true,
                )
                .map_err(BackendError::InvalidTensorData)?
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "no logits on final multimodal prompt slot".into(),
                    )
                })?,
        };

        if sampler.is_none() && engine.device_decode_ready() {
            // The multimodal prompt has already been evaluated with its exact
            // per-slot IMRoPE tables. From this point on every input is a text
            // token, so use the same resident embedding table + generated-token
            // ring as text greedy decode. Physical KV slots and logical RoPE
            // positions advance together but start at different offsets after
            // the image span; `forward_token_device` keeps those coordinates
            // separate. As in the text loop, a chunk may speculatively advance
            // reset-per-request state past a stop token without changing output.
            let mut generated = Vec::with_capacity(max_new);
            if max_new == 0 || stop.contains(&next) {
                return Ok(generated);
            }
            generated.push(next);
            token_history.push(next);
            on_token(next);
            if generated.len() >= max_new || qwen35_repetition_loop(&generated) {
                return Ok(generated);
            }

            let chunk_len = qwen35_device_decode_chunk_len();
            let mut produced = 0usize;
            'device: while generated.len() < max_new {
                let want = chunk_len.min(max_new - generated.len());
                for step in
                    qwen35_device_decode_steps(produced, want, prompt_slots, logical_position)
                {
                    engine
                        .forward_token_device(
                            step.previous_output_step,
                            step.uses_host_seed.then_some(next),
                            step.kv_position,
                            step.rope_position,
                            scale,
                            Some(step.output_step),
                        )
                        .map_err(BackendError::InvalidTensorData)?;
                }
                let ids = engine
                    .read_out_tokens(produced, want)
                    .map_err(BackendError::InvalidTensorData)?;
                for id in ids {
                    if stop.contains(&id) {
                        break 'device;
                    }
                    generated.push(id);
                    token_history.push(id);
                    on_token(id);
                    if generated.len() >= max_new || qwen35_repetition_loop(&generated) {
                        break 'device;
                    }
                }
                produced += want;
            }
            return Ok(generated);
        }

        let mut slot = prompt_slots;
        let mut generated = Vec::with_capacity(max_new);
        for index in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            generated.push(next);
            token_history.push(next);
            on_token(next);
            if qwen35_repetition_loop(&generated) {
                break;
            }
            if index + 1 < max_new {
                let embedding = self.token_embd.dequant_row(next as usize, "token_embd")?;
                let (cos, sin) =
                    qwen35_rope_tables(logical_position, self.rope_base, self.rope_dim);
                next = match &sampler {
                    Some(sampler) => qwen35_sample_logits(
                        engine
                            .forward_token_logits(&embedding, &cos, &sin, slot, scale)
                            .map_err(BackendError::InvalidTensorData)?,
                        sampler,
                        &token_history,
                    )?,
                    None => engine
                        .forward_token(&embedding, &cos, &sin, slot, scale, true)
                        .map_err(BackendError::InvalidTensorData)?
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData(
                                "no logits on multimodal decode step".into(),
                            )
                        })?,
                };
                logical_position += 1;
                slot += 1;
            }
        }
        Ok(generated)
    }

    fn generate_qwen35_cpu(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        // Batched prefill of the whole prompt (weights read once per layer), then
        // per-token greedy decode from the resulting cache.
        let (mut cache, last) = self.prefill_qwen35(prompt)?;
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        let mut next = match &sampler {
            Some(sampler) => qwen35_sample_logits(last, sampler, &token_history)?,
            None => argmax(&last),
        };
        for i in 0..max_new {
            // A stop token (EOS / `<|im_end|>` / EOG) ends the turn — and is NOT
            // appended, matching llama.cpp's served output (the stop is consumed).
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            token_history.push(next);
            on_token(next);
            if qwen35_repetition_loop(&out) {
                break;
            }
            if i + 1 < max_new {
                let logits = self.decode_token_qwen35(next, pos, &mut cache, true)?;
                pos += 1;
                next = match &sampler {
                    Some(sampler) => qwen35_sample_logits(logits, sampler, &token_history)?,
                    None => argmax(&logits),
                };
            }
        }
        Ok(out)
    }

    /// GPU resident decode for qwen35. Greedy requests use the device-side token
    /// loop when available; sampled requests keep the layer graph on CUDA and
    /// copy one logits row per token to the CPU sampler.
    #[cfg(feature = "cuda")]
    fn generate_qwen35_cuda(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: Option<&SamplingConfig>,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        // Sparse KV: only the 8 full-attention layers keep a real KV buffer (the 24 SSM
        // layers don't attend — see build_qwen35_resident::sparsify_kv), so KV is ~4x
        // smaller than dense and 8192 positions fit alongside the 5.24 GB Q4_K_M on a 6 GB
        // card (~5.8 GB resident). Default 8192; override (e.g. higher on bigger cards).
        let max_pos: usize = std::env::var("CAMELID_QWEN35_CUDA_MAXPOS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let mut guard = self
            .cuda
            .lock()
            .map_err(|_| BackendError::InvalidTensorData("qwen35 cuda mutex poisoned".into()))?;
        if guard.is_none() {
            let e = self
                .build_qwen35_resident(max_pos)
                .map_err(BackendError::InvalidTensorData)?;
            *guard = Some(e);
        }
        let engine = guard.as_mut().unwrap();
        // CRITICAL: the SSM/conv recurrent state persists across generate calls; reset it
        // at the start of every call or the 2nd+ prompt decodes on stale state.
        engine
            .reset_qwen35_state()
            .map_err(BackendError::InvalidTensorData)?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let sampler = sampling.map(|config| LlamaSampler::Sampling(config.clone()));
        let mut token_history = prompt.to_vec();
        if sampler.is_none() && engine.device_decode_ready() {
            // ---- Device-side decode loop ----------------------------------
            // Every per-token input is produced ON the GPU: the embedding row
            // is gathered from the resident quantized table (fed by the
            // previous argmax slot directly), the rope row comes from the
            // resident all-positions tables, and position arrives as a 4-byte
            // async upload. The host synchronizes once per CHUNK to read the
            // generated ids and check stop tokens — removing the per-token
            // argmax D2H -> CPU dequant_row -> 16 KB H2D round-trip. Kernels
            // and math are identical to the host-fed path (the gather is an
            // elementwise-exact dequant mirror), so greedy output matches
            // token-for-token; the only behavior difference is that forwards
            // scheduled after a mid-chunk stop token advance the (reset-per-
            // call) SSM/KV state without affecting the returned sequence.
            for (i, &tok) in prompt.iter().enumerate() {
                let last = i == prompt.len() - 1;
                engine
                    .forward_token_device(
                        None,
                        Some(tok),
                        i,
                        i,
                        scale,
                        if last { Some(0) } else { None },
                    )
                    .map_err(BackendError::InvalidTensorData)?;
            }
            let chunk_len = qwen35_device_decode_chunk_len();
            let mut out = Vec::with_capacity(max_new);
            let mut t = 0usize; // generated candidates confirmed so far
            'outer: while t < max_new {
                // Candidate t is already in the ring (prefill or prior chunk);
                // forward j (input = candidate t+j at position prompt+t+j)
                // produces candidate t+j+1. Scheduling is bounded by the
                // KV/ring capacity; only candidates covered by fresh forwards
                // (plus the pre-existing candidate t) are read back.
                let want = chunk_len.min(max_new - t);
                let mut scheduled = 0usize;
                for j in 0..want {
                    let pos = prompt.len() + t + j;
                    if pos + 1 >= max_pos {
                        break;
                    }
                    engine
                        .forward_token_device(Some(t + j), None, pos, pos, scale, Some(t + j + 1))
                        .map_err(BackendError::InvalidTensorData)?;
                    scheduled += 1;
                }
                let readable = (scheduled + 1).min(want);
                let ids = engine
                    .read_out_tokens(t, readable)
                    .map_err(BackendError::InvalidTensorData)?;
                for id in ids {
                    if stop.contains(&id) {
                        break 'outer;
                    }
                    out.push(id);
                    on_token(id);
                    if out.len() >= max_new {
                        break 'outer;
                    }
                }
                t += readable;
                if scheduled < want {
                    break; // context capacity exhausted
                }
            }
            return Ok(out);
        }
        // ---- Host-fed loop ------------------------------------------------
        // Used when device tables are unavailable and for sampled generation.
        // Sampling still runs every transformer/SSM layer on CUDA; only the
        // final logits row crosses to the CPU sampler.
        let (&last_prompt_token, prior_prompt) = prompt
            .split_last()
            .ok_or_else(|| BackendError::InvalidTensorData("empty prompt".into()))?;
        if !prior_prompt.is_empty() {
            let half = self.rope_dim / 2;
            let mut embeddings = Vec::with_capacity(prior_prompt.len() * self.d_model);
            let mut cos_all = Vec::with_capacity(prior_prompt.len() * half);
            let mut sin_all = Vec::with_capacity(prior_prompt.len() * half);
            for (position, &token) in prior_prompt.iter().enumerate() {
                embeddings
                    .extend_from_slice(&self.token_embd.dequant_row(token as usize, "token_embd")?);
                let (cos, sin) = qwen35_rope_tables(position, self.rope_base, self.rope_dim);
                cos_all.extend_from_slice(&cos);
                sin_all.extend_from_slice(&sin);
            }
            if engine.prefers_batched_prefill() {
                engine
                    .prefill_batched(&embeddings, &cos_all, &sin_all, prior_prompt.len(), scale)
                    .map_err(BackendError::InvalidTensorData)?;
            } else {
                engine
                    .prefill(&embeddings, &cos_all, &sin_all, prior_prompt.len(), scale)
                    .map_err(BackendError::InvalidTensorData)?;
            }
        }
        let last_position = prior_prompt.len();
        let emb = self
            .token_embd
            .dequant_row(last_prompt_token as usize, "token_embd")?;
        let (cos, sin) = qwen35_rope_tables(last_position, self.rope_base, self.rope_dim);
        let mut next = match &sampler {
            Some(sampler) => qwen35_sample_logits(
                engine
                    .forward_token_logits(&emb, &cos, &sin, last_position, scale)
                    .map_err(BackendError::InvalidTensorData)?,
                sampler,
                &token_history,
            )?,
            None => engine
                .forward_token(&emb, &cos, &sin, last_position, scale, true)
                .map_err(BackendError::InvalidTensorData)?
                .ok_or_else(|| {
                    BackendError::InvalidTensorData("no logits on final prompt token".into())
                })?,
        };
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        for i in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            token_history.push(next);
            on_token(next);
            if qwen35_repetition_loop(&out) {
                break;
            }
            if i + 1 < max_new {
                let emb = self.token_embd.dequant_row(next as usize, "token_embd")?;
                let (cos, sin) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
                next = match &sampler {
                    Some(sampler) => qwen35_sample_logits(
                        engine
                            .forward_token_logits(&emb, &cos, &sin, pos, scale)
                            .map_err(BackendError::InvalidTensorData)?,
                        sampler,
                        &token_history,
                    )?,
                    None => engine
                        .forward_token(&emb, &cos, &sin, pos, scale, true)
                        .map_err(BackendError::InvalidTensorData)?
                        .ok_or_else(|| {
                            BackendError::InvalidTensorData("no logits on decode step".into())
                        })?,
                };
                pos += 1;
            }
        }
        Ok(out)
    }

    /// Greedy decode that stops at the first token in `stop` (EOS / `<|im_end|>` /
    /// EOG) — for the serve path, so a turn ends instead of always emitting `max_new`
    /// tokens. The stop token is consumed, not returned. With an empty `stop` this is
    /// identical to [`generate`]. qwen35 only; other arches fall back to [`generate`].
    ///
    /// [`generate`]: RunnableModel::generate
    pub fn generate_stopping(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(BackendError::InvalidTensorData("empty prompt".into()));
        }
        if self.qwen35.is_some() {
            return self.generate_qwen35_streaming(prompt, max_new, stop, None, false, &mut |_| {});
        }
        self.generate_stopping_streaming(prompt, max_new, stop, &mut |_| {})
    }

    /// Non-streaming counterpart to
    /// [`generate_stopping_streaming_with_sampling`](Self::generate_stopping_streaming_with_sampling).
    /// Its callback is intentionally unobservable, so a late CUDA failure may
    /// safely restart on CPU without duplicating output to a client.
    pub fn generate_stopping_with_sampling(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: &SamplingConfig,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(BackendError::InvalidTensorData("empty prompt".into()));
        }
        if self.qwen35.is_some() {
            let sampling = qwen35_sampling_requires_logits(sampling).then_some(sampling);
            return self.generate_qwen35_streaming(
                prompt,
                max_new,
                stop,
                sampling,
                false,
                &mut |_| {},
            );
        }
        self.generate_stopping_streaming_with_sampling_cancelled(
            prompt,
            max_new,
            stop,
            sampling,
            &|| false,
            &mut |_| {},
        )
    }

    /// [`generate_stopping`](Self::generate_stopping) with a per-token callback
    /// (fires as each token is decided) — the serve lane's streaming source. For
    /// non-qwen35 runnable arches (no incremental hook yet) the tokens are replayed
    /// through the callback after generation completes.
    pub fn generate_stopping_streaming(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_stopping_streaming_cancelled(prompt, max_new, stop, &|| false, on_token)
    }

    /// Greedy streaming with cooperative cancellation. The predicate is checked
    /// between prompt blocks and decode tokens, allowing an SSE disconnect to
    /// release the runnable lane instead of finishing an abandoned long reply.
    pub fn generate_stopping_streaming_cancelled(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        is_cancelled: &dyn Fn() -> bool,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        let greedy = SamplingConfig::default();
        self.generate_stopping_streaming_with_sampling_cancelled(
            prompt,
            max_new,
            stop,
            &greedy,
            is_cancelled,
            on_token,
        )
    }

    /// Qwen3.5 served sampling path. Greedy/no-op configurations retain the
    /// resident GPU argmax; temperature or logit-adjusting configurations expose
    /// only the final logits row to the shared sampler.
    pub fn generate_stopping_streaming_with_sampling(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: &SamplingConfig,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        self.generate_stopping_streaming_with_sampling_cancelled(
            prompt,
            max_new,
            stop,
            sampling,
            &|| false,
            on_token,
        )
    }

    /// Sampling-capable runnable generation with cooperative cancellation. Unlike
    /// the previous generic bridge, BitNet now honors non-greedy sampling instead
    /// of silently ignoring the OpenAI request parameters.
    pub fn generate_stopping_streaming_with_sampling_cancelled(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        sampling: &SamplingConfig,
        is_cancelled: &dyn Fn() -> bool,
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(BackendError::InvalidTensorData("empty prompt".into()));
        }
        if is_cancelled() {
            return Err(BackendError::InvalidTensorData(
                "generation cancelled".into(),
            ));
        }
        if self.qwen35.is_some() {
            let sampling = qwen35_sampling_requires_logits(sampling).then_some(sampling);
            return self.generate_qwen35_streaming(prompt, max_new, stop, sampling, true, on_token);
        }
        #[cfg(target_os = "macos")]
        if self.lfm2.is_some() && lfm2_metal_enabled() {
            let sampling = qwen35_sampling_requires_logits(sampling).then_some(sampling);
            return self.generate_lfm2_metal(prompt, max_new, stop, sampling, on_token);
        }

        let mut cache = self.new_cache();
        let logits = self.prefill_generic(prompt, &mut cache, Some(is_cancelled))?;
        let sampler = qwen35_sampling_requires_logits(sampling)
            .then(|| LlamaSampler::Sampling(sampling.clone()));
        let mut token_history = prompt.to_vec();
        let choose = |logits: Vec<f32>, history: &[u32]| -> Result<u32> {
            match &sampler {
                Some(sampler) => qwen35_sample_logits(logits, sampler, history),
                None => Ok(argmax(&logits)),
            }
        };
        let mut next = choose(logits, &token_history)?;
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        while out.len() < max_new {
            if is_cancelled() {
                return Err(BackendError::InvalidTensorData(
                    "generation cancelled".into(),
                ));
            }
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            token_history.push(next);
            on_token(next);
            // The generic path serves architectures beyond Qwen 3.5 (including
            // BitNet).  Qwen's defensive repetition heuristic is model-specific
            // and can terminate perfectly valid repeated text from those models.
            if out.len() >= max_new {
                break;
            }
            if is_cancelled() {
                return Err(BackendError::InvalidTensorData(
                    "generation cancelled".into(),
                ));
            }
            let logits = self.forward_step(next, pos, &mut cache)?;
            pos += 1;
            next = choose(logits, &token_history)?;
        }
        Ok(out)
    }

    /// One token through the full qwen35 stack at absolute `pos`, mutating `cache`.
    /// Returns next-token logits when `need_logits`, else an empty Vec (the cache is
    /// still advanced). Skipping the 248k-row LM head for the non-final prompt-prefill
    /// positions — whose logits are discarded — is a large prefill speedup and changes
    /// nothing about the kept logits.
    fn decode_token_qwen35(
        &self,
        token: u32,
        pos: usize,
        cache: &mut Qwen35Cache,
        need_logits: bool,
    ) -> Result<Vec<f32>> {
        let rt = self.qwen35.as_ref().expect("qwen35 runtime present");
        let t = token as usize;
        if t >= self.vocab {
            return Err(BackendError::InvalidTensorData(format!(
                "token id {t} >= vocab {}",
                self.vocab
            )));
        }
        let mut hidden = self.token_embd.dequant_row(t, "token_embd")?;

        for (li, layer) in rt.layers.iter().enumerate() {
            let xn = self.apply_norm(&hidden, &layer.attn_norm);
            let mix = match &layer.kind {
                Qwen35Kind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => self.qwen35_full_attn(li, wq, wk, wv, wo, q_norm, k_norm, &xn, pos, cache)?,
                Qwen35Kind::Ssm { .. } => self.qwen35_ssm(rt, layer, li, &xn, cache)?,
            };
            for (h, m) in hidden.iter_mut().zip(mix.iter()) {
                *h += *m;
            }

            // FFN (SwiGLU), pre-normed by post_attention_norm; residual base is the
            // post-attention hidden state (matches qwen35.cpp ffn_residual).
            let xn2 = self.apply_norm(&hidden, &layer.post_attn_norm);
            let g = layer.ffn_gate.par_matvec(&xn2, &name(li, "ffn_gate"))?;
            let u = layer.ffn_up.par_matvec(&xn2, &name(li, "ffn_up"))?;
            let mut act = vec![0.0f32; g.len()];
            for i in 0..g.len() {
                act[i] = silu(g[i]) * u[i];
            }
            let d = layer.ffn_down.par_matvec(&act, &name(li, "ffn_down"))?;
            for (h, dv) in hidden.iter_mut().zip(d.iter()) {
                *h += *dv;
            }
        }

        // Non-final prefill positions don't need logits — skip the LM head entirely.
        if !need_logits {
            return Ok(Vec::new());
        }
        // Final norm + LM head (fused row-parallel; bit-identical to the sequential
        // loop). The 248k-row output projection is the single biggest decode cost.
        let normed = self.apply_norm(&hidden, &self.output_norm);
        self.output.par_matvec(&normed, "output")
    }

    /// Qwen3.5 full-attention layer (per-token): project Q+gate / K / V, then the
    /// shared [`qwen35_attn_compute`], then the output projection.
    ///
    /// [`qwen35_attn_compute`]: RunnableModel::qwen35_attn_compute
    #[allow(clippy::too_many_arguments)]
    fn qwen35_full_attn(
        &self,
        li: usize,
        wq: &RawMat,
        wk: &RawMat,
        wv: &RawMat,
        wo: &RawMat,
        q_norm: &[f32],
        k_norm: &[f32],
        xn: &[f32],
        pos: usize,
        cache: &mut Qwen35Cache,
    ) -> Result<Vec<f32>> {
        let qg = wq.par_matvec(xn, &name(li, "attn_q"))?;
        let k = wk.par_matvec(xn, &name(li, "attn_k"))?;
        let v = wv.par_matvec(xn, &name(li, "attn_v"))?;
        let attn_out = self.qwen35_attn_compute(q_norm, k_norm, &qg, &k, &v, pos, li, cache);
        wo.par_matvec(&attn_out, &name(li, "attn_output"))
    }

    /// The per-position full-attention compute (shared by the per-token and batched
    /// prefill paths): split fused Q+gate, q/k RMSNorm, partial NEOX RoPE, append K/V
    /// to the cache, GQA causal attention over positions `0..=pos`, sigmoid output
    /// gate. `qg`/`k_in`/`v_in` are the already-computed projections for this
    /// position; returns the gated attention context (before the output projection).
    #[allow(clippy::too_many_arguments)]
    fn qwen35_attn_compute(
        &self,
        q_norm: &[f32],
        k_norm: &[f32],
        qg: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        pos: usize,
        li: usize,
        cache: &mut Qwen35Cache,
    ) -> Vec<f32> {
        let hd = self.head_dim;
        let n_head = self.n_heads;
        let n_kv = self.n_kv_heads;
        let group = n_head / n_kv;

        // Fused Q+gate: [query(hd) | gate(hd)] interleaved per head.
        let mut q = vec![0.0f32; n_head * hd];
        let mut gate = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let b = h * hd * 2;
            q[h * hd..(h + 1) * hd].copy_from_slice(&qg[b..b + hd]);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[b + hd..b + 2 * hd]);
        }
        self.apply_norm_heads(&mut q, n_head, hd, q_norm);

        let mut k = k_in.to_vec();
        self.apply_norm_heads(&mut k, n_kv, hd, k_norm);

        // Partial NEOX RoPE: rotates the first rope_dim (64) of each 256-wide head.
        self.apply_rope(&mut q, n_head, pos, self.rope_base);
        self.apply_rope(&mut k, n_kv, pos, self.rope_base);

        cache.k[li].extend_from_slice(&k);
        cache.v[li].extend_from_slice(v_in);
        let ck = &cache.k[li];
        let cv = &cache.v[li];
        let kv_dim = n_kv * hd;
        let n_pos = pos + 1;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out = vec![0.0f32; n_head * hd];
        for h in 0..n_head {
            let kvh = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            let mut scores = vec![0.0f32; n_pos];
            let mut mx = f32::NEG_INFINITY;
            for (j, sj) in scores.iter_mut().enumerate() {
                let kh = &ck[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                let s = dot(qh, kh) * scale;
                *sj = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let oh = &mut attn_out[h * hd..(h + 1) * hd];
            for (j, s) in scores.iter().enumerate() {
                let w = *s / sum;
                let vh = &cv[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                for d in 0..hd {
                    oh[d] += w * vh[d];
                }
            }
        }

        // Sigmoid output gate (the second half of the fused Q projection).
        for (a, gt) in attn_out.iter_mut().zip(gate.iter()) {
            *a *= sigmoid(*gt);
        }
        attn_out
    }

    /// Qwen3.5 gated-delta-net (SSM) layer — the autoregressive recurrence.
    fn qwen35_ssm(
        &self,
        rt: &Qwen35Runtime,
        layer: &Qwen35Layer,
        li: usize,
        xn: &[f32],
        cache: &mut Qwen35Cache,
    ) -> Result<Vec<f32>> {
        let (wqkv, wqkv_gate, conv1d, dt_bias, a, beta_m, alpha_m, ssm_norm, ssm_out) = match &layer
            .kind
        {
            Qwen35Kind::Ssm {
                wqkv,
                wqkv_gate,
                conv1d,
                dt_bias,
                a,
                beta,
                alpha,
                ssm_norm,
                ssm_out,
            } => (
                wqkv, wqkv_gate, conv1d, dt_bias, a, beta, alpha, ssm_norm, ssm_out,
            ),
            Qwen35Kind::Full { .. } => unreachable!("qwen35_ssm called on a full-attention layer"),
        };
        let qkv = wqkv.par_matvec(xn, &name(li, "attn_qkv"))?;
        let z = wqkv_gate.par_matvec(xn, &name(li, "attn_gate"))?;
        let beta_raw = beta_m.par_matvec(xn, &name(li, "ssm_beta"))?;
        let alpha_raw = alpha_m.par_matvec(xn, &name(li, "ssm_alpha"))?;
        let final_out = self.qwen35_ssm_compute(
            rt, conv1d, dt_bias, a, ssm_norm, li, &qkv, &z, &beta_raw, &alpha_raw, cache,
        );
        ssm_out.par_matvec(&final_out, &name(li, "ssm_out"))
    }

    /// The per-position gated-delta-net (SSM) compute, shared by the per-token and
    /// batched prefill paths: β/decay gates, causal conv1d+SiLU, L2-normed q/k, the
    /// gated delta-rule recurrence (mutating the per-head state in `cache`), and the
    /// gated RMSNorm. Inputs are this position's already-computed projections; returns
    /// the value-dim vector before the `ssm_out` projection.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_ssm_compute(
        &self,
        rt: &Qwen35Runtime,
        conv1d: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        ssm_norm: &[f32],
        li: usize,
        qkv: &[f32],
        z: &[f32],
        beta_raw: &[f32],
        alpha_raw: &[f32],
        cache: &mut Qwen35Cache,
    ) -> Vec<f32> {
        let d_state = rt.d_state;
        let nk = rt.num_k_heads;
        let nv = rt.num_v_heads;
        let hv = rt.head_v_dim;
        let key_dim = rt.key_dim;
        let conv_dim = rt.conv_dim;
        let d_conv = rt.d_conv;
        let cm1 = d_conv - 1;

        let mut beta = vec![0.0f32; nv];
        let mut glog = vec![0.0f32; nv];
        for h in 0..nv {
            beta[h] = sigmoid(beta_raw[h]);
            // gate = softplus(alpha + dt_bias) * a, where a = -exp(A_log) (so glog <= 0).
            glog[h] = softplus(alpha_raw[h] + dt_bias[h]) * a[h];
        }

        // Causal depthwise conv1d (kernel d_conv) over conv_dim channels, then SiLU.
        // Window per channel = [state_0(oldest) .. state_{d_conv-2}, current].
        let conv_state = &mut cache.conv[li];
        let mut conv_out = vec![0.0f32; conv_dim];
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for t in 0..cm1 {
                acc += conv1d[c * d_conv + t] * conv_state[c * cm1 + t];
            }
            acc += conv1d[c * d_conv + cm1] * qkv[c];
            conv_out[c] = silu(acc);
            // shift ring buffer left, append current input
            for t in 0..cm1.saturating_sub(1) {
                conv_state[c * cm1 + t] = conv_state[c * cm1 + t + 1];
            }
            conv_state[c * cm1 + (cm1 - 1)] = qkv[c];
        }

        // Split conv output: q(key_dim) | k(key_dim) | v(value_dim).
        let mut q_conv = conv_out[0..key_dim].to_vec();
        let mut k_conv = conv_out[key_dim..2 * key_dim].to_vec();
        let v_conv = &conv_out[2 * key_dim..];
        // L2-normalize each k-head for q and k (per 128-vector); v is not normalized.
        for hk in 0..nk {
            l2_norm_inplace(&mut q_conv[hk * d_state..(hk + 1) * d_state], self.eps);
            l2_norm_inplace(&mut k_conv[hk * d_state..(hk + 1) * d_state], self.eps);
        }
        let qscale = 1.0 / (d_state as f32).sqrt();

        let mut final_out = vec![0.0f32; rt.value_dim];
        let mut sk = vec![0.0f32; d_state];
        let mut dvec = vec![0.0f32; d_state];
        let mut o = vec![0.0f32; d_state];
        for h in 0..nv {
            // GQA: value head h reads key/query head (h % num_k_heads) (ggml tile-repeat).
            let hk = h % nk;
            let qh = &q_conv[hk * d_state..(hk + 1) * d_state];
            let kh = &k_conv[hk * d_state..(hk + 1) * d_state];
            let vh = &v_conv[h * hv..(h + 1) * hv];
            let st = &mut cache.state[li][h * d_state * d_state..(h + 1) * d_state * d_state];

            // decay: S *= exp(g_log)
            let g = glog[h].exp();
            for s in st.iter_mut() {
                *s *= g;
            }
            // sk[j] = Σ_i S[i,j]·k[i]   (contract key index i)
            sk.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..d_state {
                let ki = kh[i];
                let row = &st[i * d_state..(i + 1) * d_state];
                for j in 0..d_state {
                    sk[j] += row[j] * ki;
                }
            }
            // d[j] = (v[j] − sk[j])·β
            let bh = beta[h];
            for j in 0..d_state {
                dvec[j] = (vh[j] - sk[j]) * bh;
            }
            // rank-1 update: S[i,j] += k[i]·d[j]
            for i in 0..d_state {
                let ki = kh[i];
                let row = &mut st[i * d_state..(i + 1) * d_state];
                for j in 0..d_state {
                    row[j] += ki * dvec[j];
                }
            }
            // o[j] = Σ_i S[i,j]·(q[i]·qscale)   (reads the updated state)
            o.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..d_state {
                let qi = qh[i] * qscale;
                let row = &st[i * d_state..(i + 1) * d_state];
                for j in 0..d_state {
                    o[j] += row[j] * qi;
                }
            }
            // gated RMSNorm: RMSNorm(o, ssm_norm) · SiLU(z_head)
            let normed = rms_norm(&o, ssm_norm, self.eps);
            let zh = &z[h * hv..(h + 1) * hv];
            for j in 0..hv {
                final_out[h * hv + j] = normed[j] * silu(zh[j]);
            }
        }
        final_out
    }
}

/// L2 normalize `x` in place: `x / max(sqrt(Σx²), eps)` — matches ggml `ggml_l2_norm`
/// (double-precision sum, `fmax` with eps, no weight).
fn l2_norm_inplace(x: &mut [f32], eps: f32) {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let scale = 1.0f32 / (ss as f32).sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// SiLU / swish: `x · sigmoid(x)`.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable softplus, matching ggml `ggml_compute_softplus_f32`.
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Per-head RMSNorm in place: normalize each of `n_heads` contiguous `head_dim`
/// slices with the shared `weight` (length `head_dim`). Used for QK-norm (qwen3,
/// gemma3).
fn norm_heads(vec: &mut [f32], n_heads: usize, head_dim: usize, weight: &[f32], eps: f32) {
    for h in 0..n_heads {
        let slice = &mut vec[h * head_dim..(h + 1) * head_dim];
        let ss: f32 = slice.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for (x, w) in slice.iter_mut().zip(weight.iter()) {
            *x = *x * inv * *w;
        }
    }
}

/// Per-head LayerNorm in place: normalize each of `n_heads` contiguous `head_dim`
/// slices with the shared `weight` (length `head_dim`). Centers the mean before variance.
fn layer_norm_heads(vec: &mut [f32], n_heads: usize, head_dim: usize, weight: &[f32], eps: f32) {
    for h in 0..n_heads {
        let slice = &mut vec[h * head_dim..(h + 1) * head_dim];
        let mean = slice.iter().sum::<f32>() / head_dim as f32;
        let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (x, w) in slice.iter_mut().zip(weight.iter()) {
            *x = (*x - mean) * inv * *w;
        }
    }
}

/// RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`.
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / n + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * inv * w)
        .collect()
}

fn layer_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| (v - mean) * inv * w)
        .collect()
}

/// gelu with the tanh approximation (`gelu_pytorch_tanh`), gemma's FFN activation.
fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(all(test, not(target_os = "macos"), feature = "cuda"))]
mod prism_vision_cuda_tests {
    use super::*;

    /// Real-model routing probe for the Windows/Linux multimodal decoder. A
    /// zero image embedding isolates the CUDA IMRoPE/KV path from the separately
    /// tested projector while still exercising arbitrary (non-token) inputs.
    /// Two generated tokens force the second token through the device ring with
    /// distinct physical-KV and logical-RoPE positions.
    #[test]
    fn real_prism_row_accepts_an_image_embedding_on_cuda() {
        let Ok(path) = std::env::var("CAMELID_PRISM_27B_GGUF") else {
            eprintln!("SKIP: set CAMELID_PRISM_27B_GGUF");
            return;
        };
        let model = RunnableModel::load(&path).expect("load Prism language row");
        let image = super::super::vision::PrismVisionEmbedding {
            embeddings: vec![vec![0.0; model.d_model]],
            grid_width: 1,
            grid_height: 1,
        };
        let generated = model
            .generate_vision(&[0], &image, &[0], 2, &[])
            .expect("generate multimodal CUDA continuation");
        assert_eq!(generated.len(), 2);
        assert!(generated
            .iter()
            .all(|&token| (token as usize) < model.vocab));
    }
}

fn validate_extra_norms(
    layer: &Layer,
    layer_index: usize,
    architecture: &str,
    require_projection_norms: bool,
    d_model: usize,
    q_dim: usize,
    ffn_dim: usize,
) -> Result<()> {
    let projection_norms = [
        ("attn_q_norm_in", layer.q_norm_in.as_ref(), d_model),
        ("attn_k_norm_in", layer.k_norm_in.as_ref(), d_model),
        ("attn_v_norm_in", layer.v_norm_in.as_ref(), d_model),
        ("attn_output_norm_in", layer.output_norm_in.as_ref(), q_dim),
        ("ffn_gate_norm_in", layer.gate_norm_in.as_ref(), d_model),
        ("ffn_up_norm_in", layer.up_norm_in.as_ref(), d_model),
        ("ffn_down_norm_in", layer.down_norm_in.as_ref(), ffn_dim),
    ];
    let projection_norm_count = projection_norms
        .iter()
        .filter(|(_, value, _)| value.is_some())
        .count();
    if (require_projection_norms && projection_norm_count != projection_norms.len())
        || (!require_projection_norms
            && projection_norm_count != 0
            && projection_norm_count != projection_norms.len())
    {
        return Err(BackendError::InvalidModelMetadata(format!(
            "layer {layer_index}: BitNet projection-input norm set is incomplete ({projection_norm_count}/{} tensors)",
            projection_norms.len()
        )));
    }
    for (name, value, expected) in projection_norms {
        if let Some(value) = value {
            if value.len() != expected {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {layer_index} {name} has {} elements, expected {expected}",
                    value.len()
                )));
            }
        }
    }

    match (&layer.attn_sub_norm, &layer.ffn_sub_norm) {
        (Some(attn), Some(ffn)) => {
            if attn.len() != q_dim || ffn.len() != ffn_dim {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {layer_index}: BitNet SubLN widths are attention={} and ffn={}, expected {q_dim} and {ffn_dim}",
                    attn.len(),
                    ffn.len()
                )));
            }
        }
        (None, None) if architecture != "bitnet-b1.58" => {}
        (None, None) => {
            return Err(BackendError::InvalidModelMetadata(format!(
                "layer {layer_index}: bitnet-b1.58 requires attn_sub_norm and ffn_sub_norm"
            )));
        }
        _ => {
            return Err(BackendError::InvalidModelMetadata(format!(
                "layer {layer_index}: BitNet SubLN requires both attn_sub_norm and ffn_sub_norm"
            )));
        }
    }
    Ok(())
}

fn name(layer: usize, tensor: &str) -> String {
    format!("blk.{layer}.{tensor}")
}

fn find_tensor<'a>(gguf: &'a GgufFile, name: &str) -> Result<&'a GgufTensorDescriptor> {
    gguf.tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))
}

/// ggml `ne = [in_features, out_features]` for a 2-D weight.
fn mat_dims(d: &GgufTensorDescriptor, name: &str) -> Result<(usize, usize)> {
    if d.dimensions.len() != 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} expected 2 dims, got {:?}",
            d.dimensions
        )));
    }
    Ok((d.dimensions[0] as usize, d.dimensions[1] as usize))
}

fn read_tensor_bytes(f: &mut File, d: &GgufTensorDescriptor, name: &str) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; d.n_bytes as usize];
    f.seek(SeekFrom::Start(d.absolute_offset))
        .map_err(|e| BackendError::Io {
            path: name.into(),
            source: e,
        })?;
    f.read_exact(&mut bytes).map_err(|e| BackendError::Io {
        path: name.into(),
        source: e,
    })?;
    Ok(bytes)
}

/// Physical token slots reserved by the resident Qwen3.5 Metal graph. The
/// request's `max_tokens` is an upper bound, not a demand, so callers clamp the
/// generation budget to the unused portion of this capacity.
#[cfg(target_os = "macos")]
/// Resident position capacity for the LFM2 Metal graph. KV is allocated for the 8
/// attention layers only (the 22 conv layers carry a fixed 2-column ring), so 4096
/// positions cost ~8 MiB of KV, not the dense figure.
#[cfg(target_os = "macos")]
fn lfm2_metal_context_capacity() -> usize {
    std::env::var("CAMELID_LFM2_METAL_MAXPOS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096)
}

/// Positions per prefill command buffer. The batched Q8 GEMV tiles 8 columns per
/// weight pass, so anything >= 8 amortises weight streaming; 64 keeps scratch small
/// (~2.75 MB at this model's FFN width).
/// Tiled simdgroup-matrix GEMM for LFM2 prefill projections. Default ON; opt out
/// with `CAMELID_LFM2_PREFILL_MM=0` to force the bit-exact batched GEMV.
#[cfg(target_os = "macos")]
pub(crate) fn lfm2_prefill_mm_enabled() -> bool {
    !std::env::var("CAMELID_LFM2_PREFILL_MM").is_ok_and(|v| {
        let v = v.trim();
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

#[cfg(target_os = "macos")]
fn lfm2_metal_prefill_chunk() -> usize {
    std::env::var("CAMELID_LFM2_METAL_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64)
}

/// Whether the LFM2 resident Metal lane may be used. DEFAULT-ON; the Safe
/// profile or `CAMELID_LFM2_METAL=0` (also `off`/`false`/`no`/`disabled`)
/// selects the CPU path.
///
/// This predicate MUST stay in lockstep with the execution-plan gate in
/// `execution_plan.rs`. They answer the same question, and when they disagree
/// `/v1/health` describes a lane other than the one that ran.
#[cfg(target_os = "macos")]
pub(crate) fn lfm2_metal_enabled() -> bool {
    crate::execution_plan::lfm2_metal_plan_selectable()
}

/// Whether qwen35 decode routes to the resident Metal graph (default on; `=0`
/// opt-out convention).
///
/// This predicate MUST stay in lockstep with the execution-plan gate in
/// `execution_plan.rs` — same rule as [`lfm2_metal_enabled`], and the same
/// history: the inline `== "0" || == "false"` check this replaces already
/// differed from the planner's vocabulary on `off`/`disabled`/`cpu`, which
/// would have let those opt-out spellings flip the disclosed lane while
/// routing kept serving Metal.
#[cfg(target_os = "macos")]
pub(crate) fn qwen35_metal_enabled() -> bool {
    !std::env::var("CAMELID_QWEN35_METAL")
        .is_ok_and(|v| crate::execution_plan::flag_value_disabled(&v))
}

#[cfg(target_os = "macos")]
fn qwen35_metal_context_capacity() -> usize {
    std::env::var("CAMELID_QWEN35_METAL_MAXPOS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096)
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35VisionDecodeCursor {
    kv_position: usize,
    rope_position: usize,
}

/// First text-decode coordinates after one Qwen3-VL image span.
///
/// The KV cache advances by every merged image token, while multimodal RoPE
/// advances only by the larger grid axis. Suffix text advances both clocks.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn qwen35_vision_decode_cursor(
    prefix_len: usize,
    grid_width: usize,
    grid_height: usize,
    suffix_len: usize,
) -> Qwen35VisionDecodeCursor {
    Qwen35VisionDecodeCursor {
        kv_position: prefix_len + grid_width * grid_height + suffix_len,
        rope_position: prefix_len + grid_width.max(grid_height) + suffix_len,
    }
}

#[cfg(any(target_os = "macos", feature = "cuda"))]
fn qwen35_generation_budget(
    prompt_slots: usize,
    requested_max_new: usize,
    resident_capacity: usize,
    lane: &str,
) -> Result<usize> {
    if prompt_slots >= resident_capacity {
        return Err(BackendError::InvalidTensorData(format!(
            "Qwen3.5 {lane} prompt uses {prompt_slots} slots and leaves no room for generation in resident capacity {resident_capacity}"
        )));
    }
    Ok(requested_max_new.min(resident_capacity - prompt_slots))
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen35DeviceDecodeStep {
    previous_output_step: Option<usize>,
    uses_host_seed: bool,
    kv_position: usize,
    rope_position: usize,
    output_step: usize,
}

/// Plan one device-decode chunk after a host-produced first candidate.
///
/// Output step zero consumes that host candidate. Every later step consumes the
/// preceding device output directly. `first_kv_position` and
/// `first_rope_position` deliberately remain independent: after a vision span,
/// the physical cache is farther ahead than Qwen3-VL's logical text clock.
#[cfg(feature = "cuda")]
fn qwen35_device_decode_steps(
    produced: usize,
    count: usize,
    first_kv_position: usize,
    first_rope_position: usize,
) -> impl Iterator<Item = Qwen35DeviceDecodeStep> {
    let end = produced
        .checked_add(count)
        .expect("device decode step range overflow");
    (produced..end).map(move |output_step| Qwen35DeviceDecodeStep {
        previous_output_step: output_step.checked_sub(1),
        uses_host_seed: output_step == 0,
        kv_position: first_kv_position + output_step,
        rope_position: first_rope_position + output_step,
        output_step,
    })
}

#[cfg(feature = "cuda")]
fn qwen35_device_decode_chunk_len() -> usize {
    std::env::var("CAMELID_DEVICE_DECODE_CHUNK")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value >= 1)
        .unwrap_or(8)
}

/// Run the CUDA text lane with a CPU fallback that cannot replay tokens already
/// delivered to a streaming client. Non-streaming callers pass `false` because
/// their callback is deliberately unobservable, preserving the existing
/// CUDA-to-CPU recovery behavior for [`RunnableModel::generate_qwen35`].
#[cfg(feature = "cuda")]
fn qwen35_cuda_with_cpu_fallback<T>(
    on_token: &mut dyn FnMut(u32),
    stream_tokens_observable: bool,
    cuda: impl FnOnce(&mut dyn FnMut(u32)) -> Result<T>,
    cpu: impl FnOnce(&mut dyn FnMut(u32)) -> Result<T>,
) -> Result<T> {
    let mut emitted = false;
    let cuda_result = {
        let mut tracked_on_token = |token| {
            emitted = true;
            on_token(token);
        };
        cuda(&mut tracked_on_token)
    };

    match cuda_result {
        Ok(value) => Ok(value),
        Err(error) if stream_tokens_observable && emitted => {
            eprintln!(
                "[qwen35] CUDA lane failed after streaming output ({error}); refusing CPU replay"
            );
            Err(error)
        }
        Err(error) => {
            eprintln!("[qwen35] CUDA lane failed ({error}); falling back to CPU");
            cpu(on_token)
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod qwen35_cuda_fallback_tests {
    use std::cell::Cell;

    use super::{qwen35_cuda_with_cpu_fallback, BackendError, Result};

    fn cuda_failure(message: &str) -> BackendError {
        BackendError::InvalidTensorData(message.into())
    }

    #[test]
    fn cuda_error_after_a_streamed_token_is_not_replayed_by_cpu() {
        let fallback_called = Cell::new(false);
        let mut delivered = Vec::new();

        let result: Result<Vec<u32>> = qwen35_cuda_with_cpu_fallback(
            &mut |token| delivered.push(token),
            true,
            |on_token| {
                on_token(7);
                Err(cuda_failure("late CUDA failure"))
            },
            |on_token| {
                fallback_called.set(true);
                on_token(7);
                Ok(vec![7])
            },
        );

        assert!(result.is_err());
        assert_eq!(delivered, vec![7]);
        assert!(!fallback_called.get());
    }

    #[test]
    fn cuda_error_before_streaming_still_uses_cpu_fallback() {
        let fallback_called = Cell::new(false);
        let mut delivered = Vec::new();

        let result = qwen35_cuda_with_cpu_fallback(
            &mut |token| delivered.push(token),
            true,
            |_| Err(cuda_failure("early CUDA failure")),
            |on_token| {
                fallback_called.set(true);
                on_token(11);
                Ok(vec![11])
            },
        )
        .expect("pre-stream CUDA failure should fall back");

        assert_eq!(result, vec![11]);
        assert_eq!(delivered, vec![11]);
        assert!(fallback_called.get());
    }

    #[test]
    fn non_streaming_generation_preserves_cuda_to_cpu_recovery() {
        let fallback_called = Cell::new(false);

        let result = qwen35_cuda_with_cpu_fallback(
            &mut |_| {},
            false,
            |on_token| {
                on_token(7);
                Err(cuda_failure("late CUDA failure"))
            },
            |_| {
                fallback_called.set(true);
                Ok(vec![11])
            },
        )
        .expect("non-streaming generation should still fall back");

        assert_eq!(result, vec![11]);
        assert!(fallback_called.get());
    }
}

fn qwen35_sampling_requires_logits(config: &SamplingConfig) -> bool {
    config.temperature > 0.0
        || config.presence_penalty != 0.0
        || config.frequency_penalty != 0.0
        || config.repeat_penalty != 1.0
        || !config.logit_bias.is_empty()
}

fn qwen35_sample_logits(
    logits: Vec<f32>,
    sampler: &LlamaSampler,
    token_history: &[u32],
) -> Result<u32> {
    let vocab = logits.len();
    let logits = CpuTensor::from_f32("qwen35_sample_logits", vec![1, vocab], logits)?;
    sampler.sample_with_history(&logits, token_history)
}

/// Last-resort guard against a model falling into a short exact token cycle.
/// Official Bonsai sampling normally avoids this; the guard prevents an
/// unattended request from emitting thousands of repeated list items if a
/// low-bit near-tie still collapses into a loop.
fn qwen35_repetition_loop(tokens: &[u32]) -> bool {
    const REPEATS: usize = 4;
    const MAX_PERIOD: usize = 16;
    (1..=MAX_PERIOD).any(|period| {
        let needed = period * REPEATS;
        tokens.len() >= needed
            && (1..REPEATS).all(|repeat| {
                let tail = tokens.len() - period;
                let prior = tail - repeat * period;
                tokens[tail..] == tokens[prior..prior + period]
            })
    })
}

/// qwen35 (Ornith) partial-NEOX RoPE cos/sin tables for absolute `pos`, length
/// `rope_dim/2`. VERBATIM `apply_rope`: `1.0/base.powf(2.0*i/rope_dim)` then
/// `(pos*freq).sin_cos()` (sin first) — do NOT use the negated-exponent form the
/// Llama lane uses (last-ULP drift can flip a near-tie greedy token).
#[cfg(any(feature = "cuda", target_os = "macos"))]
#[allow(dead_code)] // used by the GPU test + the M4 generate_qwen35_cuda driver (next).
fn qwen35_rope_tables(pos: usize, rope_base: f32, rope_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let mut cos_t = vec![0.0f32; half];
    let mut sin_t = vec![0.0f32; half];
    for i in 0..half {
        let freq = 1.0f32 / rope_base.powf(2.0 * i as f32 / rope_dim as f32);
        let (s, c) = (pos as f32 * freq).sin_cos();
        cos_t[i] = c;
        sin_t[i] = s;
    }
    (cos_t, sin_t)
}

/// Qwen3.5 IMRoPE table for one image embedding. Pair ownership is interleaved
/// time/height/width (`t,h,w,t,h,w,...`) subject to the section caps, exactly as
/// llama.cpp's `GGML_ROPE_TYPE_IMROPE`; the frequency index continues across
/// sections and therefore must not reset for height or width.
#[cfg(any(target_os = "macos", feature = "cuda"))]
fn qwen35_imrope_tables(
    positions: [usize; 4],
    sections: [usize; 4],
    rope_base: f32,
    rope_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let section_total = sections.iter().sum::<usize>();
    debug_assert_eq!(section_total, half);
    let mut cos_t = vec![0.0f32; half];
    let mut sin_t = vec![0.0f32; half];
    for pair in 0..half {
        let sector = pair % section_total;
        let position = if sector % 3 == 1 && sector < 3 * sections[1] {
            positions[1]
        } else if sector % 3 == 2 && sector < 3 * sections[2] {
            positions[2]
        } else if sector % 3 == 0 && sector < 3 * sections[0] {
            positions[0]
        } else {
            positions[3]
        };
        let freq = 1.0f32 / rope_base.powf(2.0 * pair as f32 / rope_dim as f32);
        let (sin, cos) = (position as f32 * freq).sin_cos();
        cos_t[pair] = cos;
        sin_t[pair] = sin;
    }
    (cos_t, sin_t)
}

#[cfg(all(test, any(target_os = "macos", feature = "cuda")))]
mod qwen35_imrope_tests {
    #[cfg(feature = "cuda")]
    use super::{qwen35_device_decode_steps, Qwen35DeviceDecodeStep};
    use super::{
        qwen35_generation_budget, qwen35_imrope_tables, qwen35_repetition_loop, qwen35_rope_tables,
        qwen35_vision_decode_cursor,
    };

    #[test]
    fn equal_multimodal_positions_collapse_bit_exactly_to_text_rope() {
        let normal = qwen35_rope_tables(37, 10_000_000.0, 64);
        let multimodal = qwen35_imrope_tables([37, 37, 37, 37], [11, 11, 10, 0], 10_000_000.0, 64);
        assert_eq!(normal, multimodal);
    }

    #[test]
    fn vision_decode_cursor_tracks_physical_slots_and_logical_rope() {
        let rectangular = qwen35_vision_decode_cursor(2, 3, 2, 2);
        assert_eq!(rectangular.kv_position, 10);
        assert_eq!(rectangular.rope_position, 7);

        let single = qwen35_vision_decode_cursor(2, 1, 1, 2);
        assert_eq!(single.kv_position, single.rope_position);
        assert_eq!(single.kv_position, 5);
    }

    #[test]
    fn response_upper_bound_clamps_to_remaining_resident_capacity() {
        assert_eq!(
            qwen35_generation_budget(143, 4096, 4096, "multimodal").unwrap(),
            3953
        );
        assert_eq!(
            qwen35_generation_budget(143, 8, 4096, "multimodal").unwrap(),
            8
        );
    }

    #[test]
    fn full_resident_prompt_is_rejected_without_decode_room() {
        let error = qwen35_generation_budget(4096, 4096, 4096, "multimodal")
            .unwrap_err()
            .to_string();
        assert!(error.contains("leaves no room for generation"));
    }

    #[test]
    fn short_exact_repetition_cycles_stop_but_normal_lists_do_not() {
        assert!(qwen35_repetition_loop(&[7, 8, 7, 8, 7, 8, 7, 8]));
        assert!(qwen35_repetition_loop(&[
            1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3
        ]));
        assert!(!qwen35_repetition_loop(&[7, 8, 7, 9, 7, 8, 7, 10]));
        assert!(!qwen35_repetition_loop(&[1, 2, 3, 1, 2, 3]));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn multimodal_device_chunks_keep_kv_and_rope_clocks_separate() {
        let first: Vec<_> = qwen35_device_decode_steps(0, 3, 143, 39).collect();
        assert_eq!(
            first,
            vec![
                Qwen35DeviceDecodeStep {
                    previous_output_step: None,
                    uses_host_seed: true,
                    kv_position: 143,
                    rope_position: 39,
                    output_step: 0,
                },
                Qwen35DeviceDecodeStep {
                    previous_output_step: Some(0),
                    uses_host_seed: false,
                    kv_position: 144,
                    rope_position: 40,
                    output_step: 1,
                },
                Qwen35DeviceDecodeStep {
                    previous_output_step: Some(1),
                    uses_host_seed: false,
                    kv_position: 145,
                    rope_position: 41,
                    output_step: 2,
                },
            ]
        );

        let resumed: Vec<_> = qwen35_device_decode_steps(3, 2, 143, 39).collect();
        assert_eq!(
            resumed,
            vec![
                Qwen35DeviceDecodeStep {
                    previous_output_step: Some(2),
                    uses_host_seed: false,
                    kv_position: 146,
                    rope_position: 42,
                    output_step: 3,
                },
                Qwen35DeviceDecodeStep {
                    previous_output_step: Some(3),
                    uses_host_seed: false,
                    kv_position: 147,
                    rope_position: 43,
                    output_step: 4,
                },
            ]
        );
    }
}

#[cfg(target_os = "macos")]
impl RunnableModel {
    fn build_qwen35_metal(&self, max_positions: usize) -> Result<crate::metal::Qwen35MetalDecode> {
        use crate::metal::{
            Qwen35MetalConfig, Qwen35MetalDecode, Qwen35MetalLayerInput, Qwen35MetalLayerKindInput,
        };
        let runtime = self
            .qwen35
            .as_ref()
            .ok_or_else(|| BackendError::InvalidTensorData("not a Qwen3.5 model".to_string()))?;
        let ffn_dim = runtime
            .layers
            .first()
            .map(|layer| layer.ffn_gate.out_features)
            .ok_or_else(|| BackendError::InvalidTensorData("Qwen3.5 has no layers".into()))?;
        let mut layers = Vec::with_capacity(runtime.layers.len());
        for layer in &runtime.layers {
            let kind = match &layer.kind {
                Qwen35Kind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => Qwen35MetalLayerKindInput::Full {
                    q: wq.prism_metal_weight()?,
                    k: wk.prism_metal_weight()?,
                    v: wv.prism_metal_weight()?,
                    output: wo.prism_metal_weight()?,
                    q_norm,
                    k_norm,
                },
                Qwen35Kind::Ssm {
                    wqkv,
                    wqkv_gate,
                    conv1d,
                    dt_bias,
                    a,
                    beta,
                    alpha,
                    ssm_norm,
                    ssm_out,
                } => Qwen35MetalLayerKindInput::Ssm {
                    qkv: wqkv.prism_metal_weight()?,
                    gate: wqkv_gate.prism_metal_weight()?,
                    beta: beta.prism_metal_weight()?,
                    alpha: alpha.prism_metal_weight()?,
                    output: ssm_out.prism_metal_weight()?,
                    conv1d,
                    dt_bias,
                    a,
                    norm: ssm_norm,
                },
            };
            layers.push(Qwen35MetalLayerInput {
                attn_norm: &layer.attn_norm,
                post_attn_norm: &layer.post_attn_norm,
                ffn_gate: layer.ffn_gate.prism_metal_weight()?,
                ffn_up: layer.ffn_up.prism_metal_weight()?,
                ffn_down: layer.ffn_down.prism_metal_weight()?,
                kind,
            });
        }
        let config = Qwen35MetalConfig {
            hidden: self.d_model,
            ffn_dim,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            rope_dim: self.rope_dim,
            d_conv: runtime.d_conv,
            d_state: runtime.d_state,
            n_key_heads: runtime.num_k_heads,
            n_value_heads: runtime.num_v_heads,
            key_dim: runtime.key_dim,
            value_dim: runtime.value_dim,
            conv_dim: runtime.conv_dim,
            vocab: self.vocab,
            eps: self.eps,
        };
        Qwen35MetalDecode::new(
            config,
            &layers,
            &self.output_norm,
            self.output.prism_metal_weight()?,
            max_positions,
        )
        .ok_or_else(|| {
            BackendError::InvalidTensorData(
                "Qwen3.5 Metal engine rejected the model geometry or Metal is unavailable".into(),
            )
        })
    }
}

#[cfg(feature = "cuda")]
impl RunnableModel {
    /// Build a GPU resident decode engine for this qwen35 (Ornith) model: upload every
    /// layer (SSM or full-attn) + the LM head, mirroring the proven per-layer GPU
    /// sequences. Maps each tensor's quant per-tensor (Q8_0 widened for the CPU seam,
    /// then compacted to f16-scale SoA on upload; K-quants including q5_K raw
    /// passthrough), so the Q8_0, Q4_K_M, and Q3_K_M rows all
    /// build.
    pub(crate) fn build_qwen35_resident(
        &self,
        max_pos: usize,
    ) -> std::result::Result<crate::cuda_resident::CudaResidentDecode, String> {
        use crate::cuda_resident::{widen_q8, CudaResidentDecode, ProjQuant};
        let rt = self.qwen35.as_ref().ok_or("not a qwen35 model")?;
        let ffn_dim = rt.layers[0].ffn_gate.out_features;
        // Per-tensor quant: Q8_0 weights are widened for the CPU seam (set_*'s
        // repack_for_lane then compacts them to f16-scale SoA); K-quant
        // (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K) bytes pass
        // through raw (the q2k/q3k/q4k/q5k/q6k and Prism Q1/Q2 GEMV kernels
        // expand them on the fly).
        // Q5_K previously had no resident GEMV and was upcast to Q8_0 blocks at load
        // (~+40 MiB on stock Q3_K_M's four q5_K tensors); with `q5k_gemv` in-tree it
        // rides its own lane at wire size. Returns (repack-ready bytes, lane).
        let prep = |m: &RawMat| -> std::result::Result<(Vec<u8>, ProjQuant), String> {
            match m.tt {
                GgufTensorType::Q8_0 => Ok((widen_q8(m.bytes.as_slice()), ProjQuant::Q8_0)),
                GgufTensorType::Q2K => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q2K)),
                GgufTensorType::Q3K => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q3K)),
                GgufTensorType::Q4K => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q4K)),
                GgufTensorType::Q5K => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q5K)),
                GgufTensorType::Q6K => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q6K)),
                GgufTensorType::Q1_0 => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q1_0)),
                GgufTensorType::Q2_0G64 => Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q2_0G64)),
                GgufTensorType::Q2_0G128 | GgufTensorType::Pq2_0 => {
                    Ok((m.bytes.as_slice().to_vec(), ProjQuant::Q2_0G128))
                }
                other => Err(format!(
                    "qwen35 CUDA lane: unsupported projection quant {other:?}"
                )),
            }
        };
        let mat_bytes = |m: &RawMat| m.bytes.as_slice().len() as u64;
        // Fit the graph to detected VRAM before the first upload. Full-attention
        // layers can stream all seven projections; recurrent layers keep their
        // mixer resident and stream only the three large FFN projections. Walking
        // from the tail preserves a contiguous resident prefix and reuses the
        // established pinned-host, multi-buffered offload path.
        let layer_bytes: Vec<(u64, u64)> = rt
            .layers
            .iter()
            .map(|layer| {
                let ffn = mat_bytes(&layer.ffn_gate)
                    + mat_bytes(&layer.ffn_up)
                    + mat_bytes(&layer.ffn_down);
                match &layer.kind {
                    Qwen35Kind::Full { wq, wk, wv, wo, .. } => {
                        let total =
                            ffn + mat_bytes(wq) + mat_bytes(wk) + mat_bytes(wv) + mat_bytes(wo);
                        (total, total)
                    }
                    Qwen35Kind::Ssm {
                        wqkv,
                        wqkv_gate,
                        beta,
                        alpha,
                        ssm_out,
                        ..
                    } => (
                        ffn + mat_bytes(wqkv)
                            + mat_bytes(wqkv_gate)
                            + mat_bytes(beta)
                            + mat_bytes(alpha)
                            + mat_bytes(ssm_out),
                        ffn,
                    ),
                }
            })
            .collect();
        let full_layers = rt
            .layers
            .iter()
            .filter(|layer| matches!(layer.kind, Qwen35Kind::Full { .. }))
            .count() as u64;
        let sparse_kv_bytes =
            full_layers * max_pos as u64 * self.n_kv_heads as u64 * self.head_dim as u64 * 2 * 2;
        let free_vram = crate::cuda::probe_capability()
            .map(|capability| capability.vram_free_bytes)
            .unwrap_or(0);
        let headroom_mb = std::env::var("CAMELID_QWEN35_CUDA_HEADROOM_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(768);
        let resident_budget = free_vram
            .saturating_sub(sparse_kv_bytes)
            .saturating_sub(headroom_mb * 1024 * 1024);
        let mut resident_bytes =
            layer_bytes.iter().map(|(total, _)| total).sum::<u64>() + mat_bytes(&self.output);
        let mut offloaded = vec![false; rt.layers.len()];
        if free_vram > 0 && resident_bytes > resident_budget {
            for (index, (_, reclaimable)) in layer_bytes.iter().enumerate().rev() {
                if resident_bytes <= resident_budget {
                    break;
                }
                offloaded[index] = true;
                resident_bytes = resident_bytes.saturating_sub(*reclaimable);
            }
        }
        let offloaded_layers = offloaded.iter().filter(|value| **value).count();
        if offloaded_layers > 0 {
            eprintln!(
                "[qwen35] CUDA capacity plan: {}/{} trailing layers stream from pinned host RAM; required resident weights {} MiB, budget {} MiB (free {} MiB, sparse KV {} MiB, headroom {} MiB)",
                offloaded_layers,
                rt.layers.len(),
                resident_bytes / (1024 * 1024),
                resident_budget / (1024 * 1024),
                free_vram / (1024 * 1024),
                sparse_kv_bytes / (1024 * 1024),
                headroom_mb,
            );
        }
        let mut e = CudaResidentDecode::new_for_artifact(
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.d_model,
            ffn_dim,
            self.rope_dim,
            max_pos,
            self.vocab,
            self.eps,
            self.rope_neox,
            self.resident_cuda_artifact,
        )?;
        e.set_qwen35(
            rt.d_state,
            rt.d_conv,
            rt.num_k_heads,
            rt.num_v_heads,
            rt.head_v_dim,
            rt.key_dim,
            rt.value_dim,
            rt.conv_dim,
        )?;
        // Sparse KV: only the 8 full-attention layers attend; the 24 SSM layers never
        // touch KV (the SSM forward arm skips kv_scatter/attention). new() over-allocated
        // dense KV for all 32 layers, so free the SSM layers' buffers NOW — before any
        // weights are resident — and trim the async pool so the freed VRAM is reused by
        // the weights. With ~4x less KV, max_pos fits ~4x higher in the 6 GB card.
        let keep_full: Vec<bool> = rt
            .layers
            .iter()
            .map(|l| matches!(l.kind, Qwen35Kind::Full { .. }))
            .collect();
        e.sparsify_kv(&keep_full)?;
        crate::cuda::release_async_pool();
        for (layer_index, layer) in rt.layers.iter().enumerate() {
            match &layer.kind {
                Qwen35Kind::Full {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                } => {
                    let (bq, qq) = prep(wq)?;
                    let (bk, qk) = prep(wk)?;
                    let (bv, qv) = prep(wv)?;
                    let (bo, qo) = prep(wo)?;
                    let (bg, qg) = prep(&layer.ffn_gate)?;
                    let (bu, qu) = prep(&layer.ffn_up)?;
                    let (bd, qd) = prep(&layer.ffn_down)?;
                    e.set_layer_located(
                        &bq,
                        &bk,
                        &bv,
                        &bo,
                        &bg,
                        &bu,
                        &bd,
                        &layer.attn_norm,
                        &layer.post_attn_norm,
                        Some(q_norm.as_slice()),
                        Some(k_norm.as_slice()),
                        !offloaded[layer_index],
                        [qq, qk, qv, qo, qg, qu, qd],
                    )?;
                    e.push_ssm_placeholders()?;
                }
                Qwen35Kind::Ssm {
                    wqkv,
                    wqkv_gate,
                    conv1d,
                    dt_bias,
                    a,
                    beta,
                    alpha,
                    ssm_norm,
                    ssm_out,
                } => {
                    let (bg, qg) = prep(&layer.ffn_gate)?;
                    let (bu, qu) = prep(&layer.ffn_up)?;
                    let (bd, qd) = prep(&layer.ffn_down)?;
                    let (bqkv, qqkv) = prep(wqkv)?;
                    let (bgate, qgate) = prep(wqkv_gate)?;
                    let (bbeta, qbeta) = prep(beta)?;
                    let (balpha, qalpha) = prep(alpha)?;
                    let (bout, qout) = prep(ssm_out)?;
                    e.set_layer_ssm_qwen35(
                        &bg,
                        &bu,
                        &bd,
                        &layer.attn_norm,
                        &layer.post_attn_norm,
                        &bqkv,
                        &bgate,
                        &bbeta,
                        &balpha,
                        &bout,
                        conv1d,
                        dt_bias,
                        a,
                        ssm_norm,
                        [qqkv, qgate, qbeta, qalpha, qout],
                        [qg, qu, qd],
                        !offloaded[layer_index],
                        rt.conv_dim,
                        rt.d_conv,
                        rt.num_v_heads,
                        rt.d_state,
                    )?;
                }
            }
        }
        if offloaded_layers > 0 {
            e.enable_offload_scratch()?;
        }
        let (bout, qout) = prep(&self.output)?;
        e.set_output(&self.output_norm, &bout, qout)?;
        // Device-side decode loop: resident quantized embedding table + the
        // all-positions rope tables (built with the VERBATIM qwen35_rope_tables
        // math, so the rope inputs are bit-identical to the host-fed path).
        // Failure (e.g. VRAM headroom) is non-fatal — the driver falls back to
        // the host-fed per-token loop.
        let embd_upload: Option<(Vec<u8>, ProjQuant)> = match self.token_embd.tt {
            GgufTensorType::Q8_0 => {
                Some((self.token_embd.bytes.as_slice().to_vec(), ProjQuant::Q8_0))
            }
            GgufTensorType::Q3K => {
                Some((self.token_embd.bytes.as_slice().to_vec(), ProjQuant::Q3K))
            }
            GgufTensorType::Q4K => {
                Some((self.token_embd.bytes.as_slice().to_vec(), ProjQuant::Q4K))
            }
            GgufTensorType::Q6K => Some((
                crate::cuda_resident::pad_q6k_blocks(self.token_embd.bytes.as_slice()),
                ProjQuant::Q6K,
            )),
            GgufTensorType::Q1_0 => {
                Some((self.token_embd.bytes.as_slice().to_vec(), ProjQuant::Q1_0))
            }
            GgufTensorType::Q2_0G64 => Some((
                self.token_embd.bytes.as_slice().to_vec(),
                ProjQuant::Q2_0G64,
            )),
            GgufTensorType::Q2_0G128 | GgufTensorType::Pq2_0 => Some((
                self.token_embd.bytes.as_slice().to_vec(),
                ProjQuant::Q2_0G128,
            )),
            _ => None,
        };
        if let Some((wire, family)) = embd_upload {
            let half = self.rope_dim / 2;
            let mut cos_all = Vec::with_capacity(max_pos * half);
            let mut sin_all = Vec::with_capacity(max_pos * half);
            for pos in 0..max_pos {
                let (c, sn) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
                cos_all.extend_from_slice(&c);
                sin_all.extend_from_slice(&sn);
            }
            if let Err(err) = e.set_device_decode_tables(&wire, family, &cos_all, &sin_all) {
                eprintln!(
                    "[qwen35] device-side decode tables unavailable ({err}); using the \
                     host-fed decode loop"
                );
            }
        }
        Ok(e)
    }
}

/// The resident-format list and the page-backing it drives. Both are platform-
/// independent — `load_raw` makes the same backing choice on every target — so these
/// run everywhere, not only where the resident lane exists.
#[cfg(test)]
mod resident_format_admission_tests {
    use super::*;
    use crate::metal::ResidentWeightFormat;

    #[test]
    fn admitted_types_map_to_a_resident_format_and_page_back() {
        // One table pins both halves, because page-backing is derived from the mapping.
        // Dropping an entry returns that type to `Owned`, where the resident lane
        // refuses it and the model falls back to CPU decode with no error.
        for (tt, want) in [
            (GgufTensorType::Q1_0, ResidentWeightFormat::Q1_0),
            (GgufTensorType::Q2_0G64, ResidentWeightFormat::Q2_0G64),
            (GgufTensorType::Q2_0G128, ResidentWeightFormat::Q2_0G128),
            (GgufTensorType::Pq2_0, ResidentWeightFormat::Q2_0G128),
            (GgufTensorType::Q8_0, ResidentWeightFormat::Q8_0),
            // Admitted as a PAIR. An ornith Q4_K_M file is Q4_K 217 / Q6_K 33 / F32 177,
            // so dropping either arm leaves `prism_metal_weight` erroring on the other
            // and `build_qwen35_metal` yields no resident graph at all — not a partial
            // one. Measured on that file: 11.3 tok/s decode, phys_footprint 6261 MB
            // (vs 9917 MB for the Q8_0 row).
            (GgufTensorType::Q4K, ResidentWeightFormat::Q4K),
            (GgufTensorType::Q6K, ResidentWeightFormat::Q6K),
            // Q5_K joined the pair once `q5k_linear_simd`/`q5k_linear_tiled` landed
            // (bit-exact vs `q5_k_wire_row_dot` in the Metal K-quant parity gate). A
            // Q5_K_L mix is Q5_K 144 / Q6_K 32 / Q8_0 26 / BF16 48 / F32 177, so it
            // needs the whole set admitted or it gets no resident graph at all.
            (GgufTensorType::Q5K, ResidentWeightFormat::Q5K),
            // bf16 `ssm_alpha`/`ssm_beta` (48 tensors) ride the dense bf16 kernel.
            // Widening bf16 to f32 is a bit shift, so the wire bytes are read in
            // place — this is NOT DenseF16, whose exponent width differs.
            (GgufTensorType::BF16, ResidentWeightFormat::DenseBF16),
        ] {
            assert_eq!(resident_metal_format(tt), Some(want), "{tt:?} must map");
            assert!(wants_page_backing(tt), "{tt:?} must load into WirePages");
        }

        // I2_S has its own cleanroom Metal kernel and canonical tensor-wide
        // trailer, so it is page-backed without pretending to be a Prism format.
        assert_eq!(resident_metal_format(GgufTensorType::I2S), None);
        assert!(wants_page_backing(GgufTensorType::I2S));
    }

    #[test]
    fn unadmitted_types_neither_map_nor_page_back() {
        // Page-backing costs a page per tensor, so it stays opt-in. Q3_K is the
        // fail-closed case now that Q5_K has a kernel: an ornith Q3_K_M carries BOTH
        // q3_K and q5_K, and q3_K still maps to None, so that file must keep failing
        // closed to the CPU hybrid. Shipping only a q5k kernel would not have moved it.
        for tt in [
            GgufTensorType::F32,
            GgufTensorType::F16,
            GgufTensorType::Q4_0,
            GgufTensorType::Q3K,
            GgufTensorType::Q2K,
        ] {
            assert_eq!(resident_metal_format(tt), None, "{tt:?} must not admit");
            assert!(!wants_page_backing(tt), "{tt:?} must stay Owned");
        }
    }
}

/// The macOS-only half of Q8_0 admission: `prism_metal_weight`'s page-backing
/// requirement and the wire block geometry that makes admitting Q8_0 a no-repack
/// change. The format mapping itself is platform-independent and covered above. Runs
/// on any macOS host with no GGUF and no GPU.
#[cfg(all(test, target_os = "macos"))]
mod resident_metal_q8_admission_tests {
    use super::*;
    use crate::metal::{ResidentWeightBytes, ResidentWeightFormat};
    use std::io::Write;

    /// One GGUF `Q8_0` block is an f16 scale followed by 32 int8 quants. The resident
    /// Metal Q8 GEMV consumes exactly this layout, which is why admitting Q8_0 needs no
    /// repack. If either side ever diverges from 34/32, these tests are the tripwire.
    const Q8_0_WIRE_BLOCK_BYTES: usize = 34;
    const Q8_0_BLOCK_VALUES: usize = 32;

    fn raw(
        bytes: RawMatBytes,
        tt: GgufTensorType,
        in_features: usize,
        out_features: usize,
    ) -> RawMat {
        RawMat {
            bytes,
            tt,
            in_features,
            out_features,
        }
    }

    /// `out_features` rows of `in_features` values each, in Q8_0 wire form.
    fn q8_0_wire_bytes(in_features: usize, out_features: usize) -> Vec<u8> {
        let blocks_per_row = in_features / Q8_0_BLOCK_VALUES;
        (0..out_features * blocks_per_row * Q8_0_WIRE_BLOCK_BYTES)
            .map(|i| i as u8)
            .collect()
    }

    #[test]
    fn page_backed_q8_0_admits_as_a_resident_wire_weight() {
        let (in_features, out_features) = (Q8_0_BLOCK_VALUES * 2, 3);
        let wire = q8_0_wire_bytes(in_features, out_features);
        assert_eq!(
            wire.len(),
            out_features * 2 * Q8_0_WIRE_BLOCK_BYTES,
            "fixture must be an exact number of 34-byte Q8_0 wire blocks"
        );

        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&wire).expect("write wire bytes");
        file.flush().expect("flush");
        let pages = crate::wire_mmap::WirePages::read_from_file(file.as_file(), 0, wire.len())
            .expect("Q8_0 tensors must be page-backable");
        assert_eq!(
            pages.bytes(),
            &wire[..],
            "page-backing must not alter bytes"
        );

        let m = raw(
            RawMatBytes::WirePages(pages),
            GgufTensorType::Q8_0,
            in_features,
            out_features,
        );
        match m.prism_metal_weight().expect("page-backed Q8_0 must admit") {
            ResidentWeightBytes::WirePages { format, pages } => {
                assert_eq!(format, ResidentWeightFormat::Q8_0);
                assert_eq!(pages.bytes(), &wire[..]);
            }
            _ => panic!("expected page-backed WirePages backing for a Q8_0 projection"),
        }
    }

    #[test]
    fn owned_q8_0_is_refused_instead_of_silently_copied() {
        // Variant check only: `Owned` is refused outright, so an ordinary heap tensor
        // never reaches the resident lane. The page-alignment invariant itself is
        // `WirePages`' contract, covered by
        // `metal::tests::wire_mmap_nocopy_buffer_gpu_reads_file_bytes`.
        let m = raw(
            RawMatBytes::owned(q8_0_wire_bytes(Q8_0_BLOCK_VALUES, 1)),
            GgufTensorType::Q8_0,
            Q8_0_BLOCK_VALUES,
            1,
        );
        let err = match m.prism_metal_weight() {
            Ok(_) => panic!("owned Q8_0 must not reach the resident lane"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("not page-backed"),
            "unexpected error: {err}"
        );
    }
}

/// GPU single-SSM-layer parity: upload layer 0's REAL Ornith SSM weights and run the
/// whole GPU forward (rmsnorm+quantize -> q8 gemv x4 -> SSM kernel chain -> ssm_out
/// gemv), comparing the layer `mix` to the CPU `qwen35_ssm`. Proves the gemv-from-real-
/// weights path composes with the proven SSM chain — the mechanism the resident
/// forward_pass SSM branch will use. Q8_0 weights (34-byte GGUF blocks widened to the
/// 36-byte f32-scale layout repack_q8_soa expects).
#[cfg(all(test, feature = "cuda"))]
mod gpu_ssm_layer_tests {
    use super::*;
    use crate::cuda_resident::{
        launch_gemv, launch_quantize, launch_rmsnorm_quantize, repack_q8_soa, CudaResidentKernels,
    };
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    fn widen_q8(bytes: &[u8]) -> Vec<u8> {
        let nb = bytes.len() / 34;
        let mut out = Vec::with_capacity(nb * 36);
        for b in 0..nb {
            let base = b * 34;
            let scale =
                crate::tensor::f16_bits_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&bytes[base + 2..base + 34]);
        }
        out
    }

    fn rel_close(a: &[f32], b: &[f32], tol: f32) -> (bool, f32) {
        let mut worst = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs() / y.abs().max(1.0);
            if d > worst {
                worst = d;
            }
        }
        (worst < tol, worst)
    }

    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (Q8) + a CUDA device"]
    fn qwen35_ssm_layer_gpu_matches_cpu() {
        let path = match std::env::var("CAMELID_ORNITH_GGUF") {
            Ok(p) => p,
            Err(_) => return,
        };
        let Ok(k) = CudaResidentKernels::new() else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        let rt = model.qwen35.as_ref().expect("qwen35 runtime");
        let li = 0usize; // layer 0 is SSM ((0+1)%4 != 0)
        let layer = &rt.layers[li];
        let (wqkv, wqkv_gate, conv1d, dt_bias, a_vec, beta_m, alpha_m, ssm_norm, ssm_out) =
            match &layer.kind {
                Qwen35Kind::Ssm {
                    wqkv,
                    wqkv_gate,
                    conv1d,
                    dt_bias,
                    a,
                    beta,
                    alpha,
                    ssm_norm,
                    ssm_out,
                } => (
                    wqkv, wqkv_gate, conv1d, dt_bias, a, beta, alpha, ssm_norm, ssm_out,
                ),
                _ => panic!("layer 0 is not SSM"),
            };
        for m in [wqkv, wqkv_gate, beta_m, alpha_m, ssm_out] {
            assert_eq!(
                m.tt,
                GgufTensorType::Q8_0,
                "test assumes a Q8_0 Ornith GGUF"
            );
        }
        let hidden_dim = model.d_model;
        let eps = model.eps;
        let (ds, nk, nv) = (rt.d_state, rt.num_k_heads, rt.num_v_heads);
        let (key_dim, value_dim, conv_dim, d_conv) =
            (rt.key_dim, rt.value_dim, rt.conv_dim, rt.d_conv);

        // Deterministic pseudo-random hidden activation.
        let mut seed = 0x1234_5678u64;
        let mut nextf = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let hidden: Vec<f32> = (0..hidden_dim).map(|_| nextf()).collect();

        // ---- CPU reference: the layer mix from qwen35_ssm ----
        let mut cache = Qwen35Cache::new(rt, rt.layers.len());
        let xn = rms_norm(&hidden, &layer.attn_norm, eps);
        let cpu_mix = model
            .qwen35_ssm(rt, layer, li, &xn, &mut cache)
            .expect("cpu ssm");

        // ---- GPU forward ----
        let s = &k.stream;
        let up = |m: &RawMat| {
            s.clone_htod(&repack_q8_soa(&widen_q8(m.bytes.as_slice())))
                .unwrap()
        };
        let d_wqkv = up(wqkv);
        let d_wqkv_gate = up(wqkv_gate);
        let d_beta_w = up(beta_m);
        let d_alpha_w = up(alpha_m);
        let d_ssm_out = up(ssm_out);
        let d_conv1d = s.clone_htod(conv1d).unwrap();
        let d_dt = s.clone_htod(dt_bias).unwrap();
        let d_a = s.clone_htod(a_vec).unwrap();
        let d_norm = s.clone_htod(ssm_norm).unwrap();
        let d_attn_norm = s.clone_htod(&layer.attn_norm).unwrap();
        let d_hidden = s.clone_htod(&hidden).unwrap();

        let hb = hidden_dim / 32;
        let vb = value_dim / 32;
        let mut in_q = s.alloc_zeros::<i8>(hidden_dim).unwrap();
        let mut in_s = s.alloc_zeros::<f32>(hb).unwrap();
        let mut d_qkv = s.alloc_zeros::<f32>(conv_dim).unwrap();
        let mut d_z = s.alloc_zeros::<f32>(value_dim).unwrap();
        let mut d_br = s.alloc_zeros::<f32>(nv).unwrap();
        let mut d_ar = s.alloc_zeros::<f32>(nv).unwrap();
        let mut d_beta = s.alloc_zeros::<f32>(nv).unwrap();
        let mut d_glog = s.alloc_zeros::<f32>(nv).unwrap();
        let mut d_conv_out = s.alloc_zeros::<f32>(conv_dim).unwrap();
        let mut d_ssm_mix = s.alloc_zeros::<f32>(value_dim).unwrap();
        let mut d_conv_state = s.alloc_zeros::<f32>(conv_dim * (d_conv - 1)).unwrap();
        let mut d_state = s.alloc_zeros::<f32>(nv * ds * ds).unwrap();
        let mut mix_q = s.alloc_zeros::<i8>(value_dim).unwrap();
        let mut mix_s = s.alloc_zeros::<f32>(vb).unwrap();
        let mut d_mix = s.alloc_zeros::<f32>(hidden_dim).unwrap();

        // attn rmsnorm + quantize the hidden
        launch_rmsnorm_quantize(
            s,
            &k.rms_norm_quantize,
            &d_hidden,
            &d_attn_norm,
            &mut in_q,
            &mut in_s,
            hidden_dim,
            eps,
        )
        .unwrap();
        // projections (Q8_0 q8_gemv): wqkv -> qkv, wqkv_gate -> z, beta -> br, alpha -> ar
        launch_gemv(
            s,
            &k.gemv,
            &in_s,
            &in_q,
            &d_wqkv.slice(0..d_wqkv.len()),
            conv_dim,
            hb,
            &mut d_qkv,
        )
        .unwrap();
        launch_gemv(
            s,
            &k.gemv,
            &in_s,
            &in_q,
            &d_wqkv_gate.slice(0..d_wqkv_gate.len()),
            value_dim,
            hb,
            &mut d_z,
        )
        .unwrap();
        launch_gemv(
            s,
            &k.gemv,
            &in_s,
            &in_q,
            &d_beta_w.slice(0..d_beta_w.len()),
            nv,
            hb,
            &mut d_br,
        )
        .unwrap();
        launch_gemv(
            s,
            &k.gemv,
            &in_s,
            &in_q,
            &d_alpha_w.slice(0..d_alpha_w.len()),
            nv,
            hb,
            &mut d_ar,
        )
        .unwrap();

        let nvi = nv as i32;
        let dsi = ds as i32;
        let nki = nk as i32;
        // gates
        {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (nv as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = s.launch_builder(&k.ssm_gates);
            b.arg(&d_br)
                .arg(&d_ar)
                .arg(&d_dt)
                .arg(&d_a)
                .arg(&mut d_beta)
                .arg(&mut d_glog)
                .arg(&nvi);
            unsafe { b.launch(cfg).unwrap() };
        }
        // conv1d
        {
            let cfg = LaunchConfig {
                grid_dim: (conv_dim.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let cdi = conv_dim as i32;
            let dci = d_conv as i32;
            let mut b = s.launch_builder(&k.ssm_conv1d);
            b.arg(&d_conv1d)
                .arg(&d_qkv)
                .arg(&mut d_conv_state)
                .arg(&mut d_conv_out)
                .arg(&cdi)
                .arg(&dci);
            unsafe { b.launch(cfg).unwrap() };
        }
        // l2norm q (0..key_dim) and k (key_dim..2*key_dim)
        for lo in [0usize, key_dim] {
            let cfg = LaunchConfig {
                grid_dim: (nk as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: (ds as u32) * 4,
            };
            let mut view = d_conv_out.slice_mut(lo..lo + key_dim);
            let mut b = s.launch_builder(&k.ssm_l2_norm_per_head);
            b.arg(&mut view).arg(&dsi).arg(&eps);
            unsafe { b.launch(cfg).unwrap() };
        }
        // delta rule
        {
            let cfg = LaunchConfig {
                grid_dim: (nv as u32, 1, 1),
                block_dim: (ds as u32, 1, 1),
                shared_mem_bytes: (3 * ds as u32) * 4,
            };
            let qv = d_conv_out.slice(0..key_dim);
            let kv = d_conv_out.slice(key_dim..2 * key_dim);
            let vv = d_conv_out.slice(2 * key_dim..2 * key_dim + value_dim);
            let mut b = s.launch_builder(&k.ssm_delta_rule);
            b.arg(&mut d_state)
                .arg(&kv)
                .arg(&qv)
                .arg(&vv)
                .arg(&d_z)
                .arg(&d_beta)
                .arg(&d_glog)
                .arg(&d_norm)
                .arg(&mut d_ssm_mix)
                .arg(&dsi)
                .arg(&nki)
                .arg(&eps);
            unsafe { b.launch(cfg).unwrap() };
        }
        // quantize the SSM mix, then ssm_out projection (Q8_0 gemv) -> d_mix
        launch_quantize(s, &k.quantize, &d_ssm_mix, &mut mix_q, &mut mix_s, vb).unwrap();
        launch_gemv(
            s,
            &k.gemv,
            &mix_s,
            &mix_q,
            &d_ssm_out.slice(0..d_ssm_out.len()),
            hidden_dim,
            vb,
            &mut d_mix,
        )
        .unwrap();

        let mut got = vec![0f32; hidden_dim];
        s.memcpy_dtoh(&d_mix, &mut got).unwrap();
        k.ctx.synchronize().unwrap();
        let (ok, worst) = rel_close(&got, &cpu_mix, 1e-2);
        assert!(
            ok,
            "qwen35 SSM layer GPU vs CPU diverged (worst rel {worst:.3e})"
        );
        eprintln!("qwen35_ssm_layer_gpu: PASS (worst rel {worst:.3e})");
    }
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (Q8) + a CUDA device"]
    fn qwen35_full_attn_layer_gpu_matches_cpu() {
        use crate::cuda_resident::{
            launch_attention, launch_kv_scatter, launch_rms_norm_per_head, launch_rope,
        };
        let path = match std::env::var("CAMELID_ORNITH_GGUF") {
            Ok(p) => p,
            Err(_) => return,
        };
        let Ok(k) = CudaResidentKernels::new() else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        let rt = model.qwen35.as_ref().expect("qwen35 runtime");
        let li = 3usize; // layer 3 is full-attention ((3+1) % 4 == 0)
        let layer = &rt.layers[li];
        let (wq, wk, wv, wo, q_norm, k_norm) = match &layer.kind {
            Qwen35Kind::Full {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
            } => (wq, wk, wv, wo, q_norm, k_norm),
            _ => panic!("layer {li} is not full-attention"),
        };
        for m in [wq, wk, wv, wo] {
            assert_eq!(
                m.tt,
                GgufTensorType::Q8_0,
                "test assumes a Q8_0 Ornith GGUF"
            );
        }

        let hidden_dim = model.d_model; // 4096
        let eps = model.eps; // 1e-6
        let n_head = model.n_heads; // 16
        let n_kv = model.n_kv_heads; // 4
        let hd = model.head_dim; // 256
        let rope_dim = model.rope_dim; // 64
        let rope_base = model.rope_base; // 1e7
        let pairing = if model.rope_neox { 1i32 } else { 0i32 };
        let q_width = n_head * hd; // 4096
        let kv_width = n_kv * hd; // 1024
        let half = rope_dim / 2; // 32 rope pairs
        let hb = hidden_dim / 32; // 128 input blocks per projection row
        let qb = q_width / 32; // 128 blocks for the wo input (attn-out)
        let scale = 1.0f32 / (hd as f32).sqrt(); // 1/sqrt(256) = 0.0625
        let n_steps = 6usize;
        let max_pos = 8usize;
        assert!(n_steps <= max_pos);

        // Deterministic pseudo-random hidden activations, one DISTINCT vector per
        // position (distinct K/V per step => a genuinely non-uniform softmax at pos>0,
        // which is what exercises GQA / q-k-norm / RoPE / scale; pos=0 alone is
        // degenerate: a single key makes softmax==1 and the output == V regardless).
        let mut seed = 0x1234_5678u64;
        let mut nextf = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let hiddens: Vec<Vec<f32>> = (0..n_steps)
            .map(|_| (0..hidden_dim).map(|_| nextf()).collect())
            .collect();

        // ---- GPU: upload Q8_0 weights once (compact f16-scale SoA) ----
        let s = &k.stream;
        let up = |m: &RawMat| {
            s.clone_htod(&repack_q8_soa(&widen_q8(m.bytes.as_slice())))
                .unwrap()
        };
        let d_wq = up(wq);
        let d_wk = up(wk);
        let d_wv = up(wv);
        let d_wo = up(wo);
        let d_qn = s.clone_htod(q_norm).unwrap();
        let d_kn = s.clone_htod(k_norm).unwrap();
        let d_attn_norm = s.clone_htod(&layer.attn_norm).unwrap();

        // device scratch (reused across positions)
        let mut d_hidden = s.alloc_zeros::<f32>(hidden_dim).unwrap();
        let mut in_q = s.alloc_zeros::<i8>(hidden_dim).unwrap();
        let mut in_s = s.alloc_zeros::<f32>(hb).unwrap();
        let mut d_qgate = s.alloc_zeros::<f32>(2 * q_width).unwrap(); // fused [q|gate]
        let mut d_q = s.alloc_zeros::<f32>(q_width).unwrap();
        let mut d_gate = s.alloc_zeros::<f32>(q_width).unwrap();
        let mut d_k = s.alloc_zeros::<f32>(kv_width).unwrap();
        let mut d_v = s.alloc_zeros::<f32>(kv_width).unwrap();
        let mut d_attn = s.alloc_zeros::<f32>(q_width).unwrap();
        let mut d_scores = s.alloc_zeros::<f32>(n_head * max_pos).unwrap();
        let mut mix_q = s.alloc_zeros::<i8>(q_width).unwrap();
        let mut mix_s = s.alloc_zeros::<f32>(qb).unwrap();
        let mut d_mix = s.alloc_zeros::<f32>(hidden_dim).unwrap();

        // persistent KV cache (f16 bits, layout [kv_head][position][head_dim]) +
        // device position + per-pair RoPE tables.
        let mut cache_k = s.alloc_zeros::<u8>(kv_width * max_pos * 2).unwrap();
        let mut cache_v = s.alloc_zeros::<u8>(kv_width * max_pos * 2).unwrap();
        let mut d_pos = s.alloc_zeros::<i32>(1).unwrap();
        let mut d_cos = s.alloc_zeros::<f32>(half).unwrap();
        let mut d_sin = s.alloc_zeros::<f32>(half).unwrap();

        // CPU reference cache (grows per position; only the full-attn layer li used).
        let mut cache = Qwen35Cache::new(rt, rt.layers.len());

        let mut worst_all = 0.0f32;
        // `p` is the absolute position (used in RoPE tables, kv_scatter, d_pos) as well
        // as the index into `hiddens`, so a plain range loop is clearest here.
        #[allow(clippy::needless_range_loop)]
        for p in 0..n_steps {
            let hidden = &hiddens[p];

            // ---- CPU reference: the FULL layer attention mix (incl. wo); also grows
            // cache.k/v[li] exactly as the GPU scatter does. qwen35_full_attn internally
            // calls qwen35_attn_compute then wo.par_matvec — so we compare the post-wo
            // mix on both sides (the GPU pipeline below also ends in wo). ----
            let xn = rms_norm(hidden, &layer.attn_norm, eps);
            let cpu_mix = model
                .qwen35_full_attn(li, wq, wk, wv, wo, q_norm, k_norm, &xn, p, &mut cache)
                .expect("cpu full attn");

            // ---- GPU forward for this position ----
            // RoPE cos/sin for absolute position p (computed in f32 on the host, exactly
            // like apply_rope's per-pair freqs, then uploaded -> GPU RoPE is bit-identical).
            let mut cosv = vec![0f32; half];
            let mut sinv = vec![0f32; half];
            for i in 0..half {
                let freq = 1.0f32 / rope_base.powf(2.0 * i as f32 / rope_dim as f32);
                let (si, ci) = (p as f32 * freq).sin_cos();
                cosv[i] = ci;
                sinv[i] = si;
            }
            s.memcpy_htod(&cosv, &mut d_cos).unwrap();
            s.memcpy_htod(&sinv, &mut d_sin).unwrap();
            s.memcpy_htod(&[p as i32], &mut d_pos).unwrap();
            s.memcpy_htod(hidden.as_slice(), &mut d_hidden).unwrap();

            // attn-norm + quantize the hidden -> in_q / in_s
            launch_rmsnorm_quantize(
                s,
                &k.rms_norm_quantize,
                &d_hidden,
                &d_attn_norm,
                &mut in_q,
                &mut in_s,
                hidden_dim,
                eps,
            )
            .unwrap();

            // fused query+gate projection: wq rows = 2*q_width = 8192
            launch_gemv(
                s,
                &k.gemv,
                &in_s,
                &in_q,
                &d_wq.slice(0..d_wq.len()),
                2 * q_width,
                hb,
                &mut d_qgate,
            )
            .unwrap();
            // K / V projections (kv_width = 1024 rows each)
            launch_gemv(
                s,
                &k.gemv,
                &in_s,
                &in_q,
                &d_wk.slice(0..d_wk.len()),
                kv_width,
                hb,
                &mut d_k,
            )
            .unwrap();
            launch_gemv(
                s,
                &k.gemv,
                &in_s,
                &in_q,
                &d_wv.slice(0..d_wv.len()),
                kv_width,
                hb,
                &mut d_v,
            )
            .unwrap();

            // deinterleave fused [query(hd)|gate(hd)] x n_head -> contiguous d_q / d_gate
            {
                let cfg = LaunchConfig {
                    grid_dim: ((q_width as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (nh, hdi) = (n_head as i32, hd as i32);
                let mut b = s.launch_builder(&k.deinterleave_qgate);
                b.arg(&d_qgate)
                    .arg(&mut d_q)
                    .arg(&mut d_gate)
                    .arg(&nh)
                    .arg(&hdi);
                unsafe { b.launch(cfg).unwrap() };
            }

            // QK per-head RMSNorm (BEFORE RoPE), shared weight across heads
            launch_rms_norm_per_head(s, &k.rms_norm_per_head, &mut d_q, &d_qn, n_head, hd, eps)
                .unwrap();
            launch_rms_norm_per_head(s, &k.rms_norm_per_head, &mut d_k, &d_kn, n_kv, hd, eps)
                .unwrap();

            // partial NEOX RoPE on Q (n_head heads) and K (n_kv heads)
            launch_rope(
                s, &k.rope, &mut d_q, &d_cos, &d_sin, n_head, hd, rope_dim, pairing,
            )
            .unwrap();
            launch_rope(
                s, &k.rope, &mut d_k, &d_cos, &d_sin, n_kv, hd, rope_dim, pairing,
            )
            .unwrap();

            // scatter K (post-norm, post-rope) and V (RAW) into the cache at position p
            launch_kv_scatter(
                s,
                &k.kv_scatter,
                &d_k,
                &mut cache_k,
                &d_pos,
                n_kv,
                hd,
                max_pos,
            )
            .unwrap();
            launch_kv_scatter(
                s,
                &k.kv_scatter,
                &d_v,
                &mut cache_v,
                &d_pos,
                n_kv,
                hd,
                max_pos,
            )
            .unwrap();

            // GQA causal attention over positions 0..=p
            let n_pos = p + 1;
            launch_attention(
                s,
                &k.attention,
                &d_q,
                &cache_k,
                &cache_v,
                &mut d_attn,
                n_head,
                n_kv,
                hd,
                &d_pos,
                n_pos,
                max_pos,
                scale,
                &mut d_scores,
            )
            .unwrap();

            // sigmoid output gate: attn[i] *= sigmoid(gate[i])
            {
                let cfg = LaunchConfig {
                    grid_dim: ((q_width as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i = q_width as i32;
                let mut b = s.launch_builder(&k.sigmoid_mul);
                b.arg(&mut d_attn).arg(&d_gate).arg(&n_i);
                unsafe { b.launch(cfg).unwrap() };
            }

            // quantize the gated attn-out, then the O projection -> d_mix
            launch_quantize(s, &k.quantize, &d_attn, &mut mix_q, &mut mix_s, qb).unwrap();
            launch_gemv(
                s,
                &k.gemv,
                &mix_s,
                &mix_q,
                &d_wo.slice(0..d_wo.len()),
                hidden_dim,
                qb,
                &mut d_mix,
            )
            .unwrap();

            let mut got = vec![0f32; hidden_dim];
            s.memcpy_dtoh(&d_mix, &mut got).unwrap();
            k.ctx.synchronize().unwrap();
            let (_ok, worst) = rel_close(&got, &cpu_mix, 1e-2);
            if worst > worst_all {
                worst_all = worst;
            }
            eprintln!("qwen35_full_attn pos {p}: worst rel {worst:.3e}");
        }
        assert!(
            worst_all < 1e-2,
            "qwen35 full-attn layer GPU vs CPU diverged (worst rel {worst_all:.3e})"
        );
        eprintln!("qwen35_full_attn_layer_gpu: PASS (worst rel {worst_all:.3e})");
    }

    fn argmax_u32(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv {
                    (i, x)
                } else {
                    (bi, bv)
                }
            })
            .0 as u32
    }

    /// Full 32-layer GPU stack: build the resident engine from real weights, run a
    /// prompt token-by-token through forward_pass (SSM + full-attn branches), and assert
    /// the GPU next-token argmax equals the CPU runnable lane's (greedy-token parity).
    #[test]
    #[ignore = "needs CAMELID_PRISM_27B_GGUF (Q1_0) + a CUDA device"]
    fn qwen35_prism_q1_batched_prefill_matches_serial_logits_bitwise() {
        let Ok(path) = std::env::var("CAMELID_PRISM_27B_GGUF") else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load Prism qwen35");
        let prompt: Vec<u32> = vec![
            3710, 369, 279, 6511, 314, 9338, 30, 220, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        ];
        let (&last, prior) = prompt.split_last().unwrap();
        let half = model.rope_dim / 2;
        let mut embeddings = Vec::with_capacity(prior.len() * model.d_model);
        let mut cos_all = Vec::with_capacity(prior.len() * half);
        let mut sin_all = Vec::with_capacity(prior.len() * half);
        for (position, &token) in prior.iter().enumerate() {
            embeddings.extend_from_slice(
                &model
                    .token_embd
                    .dequant_row(token as usize, "token_embd")
                    .expect("embedding"),
            );
            let (cos, sin) = qwen35_rope_tables(position, model.rope_base, model.rope_dim);
            cos_all.extend_from_slice(&cos);
            sin_all.extend_from_slice(&sin);
        }
        let last_embedding = model
            .token_embd
            .dequant_row(last as usize, "token_embd")
            .expect("last embedding");
        let (last_cos, last_sin) = qwen35_rope_tables(prior.len(), model.rope_base, model.rope_dim);
        let scale = 1.0f32 / (model.head_dim as f32).sqrt();
        let mut engine = model
            .build_qwen35_resident(prompt.len() + 8)
            .expect("build Prism resident engine");
        assert!(engine.prefers_batched_prefill());

        engine.reset_qwen35_state().unwrap();
        engine
            .prefill(&embeddings, &cos_all, &sin_all, prior.len(), scale)
            .expect("serial prefill");
        let serial = engine
            .forward_token_logits(&last_embedding, &last_cos, &last_sin, prior.len(), scale)
            .expect("serial logits");

        engine.reset_qwen35_state().unwrap();
        engine
            .prefill_batched(&embeddings, &cos_all, &sin_all, prior.len(), scale)
            .expect("batched prefill");
        let batched = engine
            .forward_token_logits(&last_embedding, &last_cos, &last_sin, prior.len(), scale)
            .expect("batched logits");

        let first_diff = serial
            .iter()
            .zip(&batched)
            .position(|(left, right)| left.to_bits() != right.to_bits());
        assert_eq!(
            first_diff, None,
            "hybrid Q1 batched prefill changed final logits"
        );
    }

    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (Q8) + a CUDA device"]
    fn qwen35_gpu_single_token_matches_cpu() {
        let Ok(path) = std::env::var("CAMELID_ORNITH_GGUF") else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        if model.qwen35.is_none() {
            return;
        }
        let prompt: Vec<u32> = vec![3710, 369, 279, 6511, 314, 9338, 30];
        let cpu_logits = model.forward_logits_qwen35(&prompt).expect("cpu forward");
        let cpu_tok = argmax_u32(&cpu_logits);
        let mut e = match model.build_qwen35_resident(prompt.len() + 4) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("build_qwen35_resident failed: {err}");
                return;
            }
        };
        e.reset_qwen35_state().unwrap();
        let scale = 1.0f32 / (model.head_dim as f32).sqrt();
        let mut gpu_tok = 0u32;
        for (i, &tok) in prompt.iter().enumerate() {
            let emb = model
                .token_embd
                .dequant_row(tok as usize, "token_embd")
                .expect("embd");
            let (cos, sin) = super::qwen35_rope_tables(i, model.rope_base, model.rope_dim);
            let last = i == prompt.len() - 1;
            let out = e
                .forward_token(&emb, &cos, &sin, i, scale, last)
                .expect("gpu forward_token");
            if last {
                gpu_tok = out.expect("logits on final token");
            }
        }
        eprintln!("qwen35_gpu_single_token: cpu={cpu_tok} gpu={gpu_tok}");
        assert_eq!(gpu_tok, cpu_tok, "GPU next-token argmax != CPU runnable");
    }

    /// End-to-end qwen35 GPU greedy decode vs the CPU runnable lane (point
    /// CAMELID_ORNITH_GGUF at the 5.24 GB Q4_K_M so it fits 6 GB VRAM). Also a run-twice
    /// check: the second GPU call reuses the cached engine, so identical streams prove
    /// reset_qwen35_state() actually clears the recurrent SSM/conv state.
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (point at Q4_K_M) + a CUDA device"]
    fn qwen35_gpu_greedy_matches_cpu() {
        let Ok(path) = std::env::var("CAMELID_ORNITH_GGUF") else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        if model.qwen35.is_none() {
            return;
        }
        // "What is the capital of France?" prompt tokens.
        let prompt: Vec<u32> = vec![3710, 369, 279, 6511, 314, 9338, 30];
        let n = 8usize;
        let cpu = model
            .generate_qwen35_cpu(&prompt, n, &[], None, &mut |_| {})
            .expect("cpu gen");
        let gpu = model
            .generate_qwen35_cuda(&prompt, n, &[], None, &mut |_| {})
            .expect("gpu gen");
        // run-twice reuses the cached engine -> validates reset_qwen35_state.
        let gpu2 = model
            .generate_qwen35_cuda(&prompt, n, &[], None, &mut |_| {})
            .expect("gpu gen 2");
        eprintln!("cpu ={cpu:?}");
        eprintln!("gpu ={gpu:?}");
        eprintln!("gpu2={gpu2:?}");
        assert_eq!(
            gpu, gpu2,
            "qwen35 CUDA run-twice differs — SSM state not reset"
        );
        assert_eq!(gpu, cpu, "qwen35 CUDA greedy != CPU runnable");
        // Coherence sanity: decode a 16-token GPU continuation.
        if let Ok(gguf) = read_metadata(&path) {
            if let Ok(tok) = crate::tokenizer::Tokenizer::from_gguf(&gguf) {
                let ids16 = model
                    .generate_qwen35_cuda(&prompt, 16, &[], None, &mut |_| {})
                    .expect("gpu 16-tok");
                let text = tok.decode(&ids16, true).unwrap_or_default();
                eprintln!("qwen35 GPU 16-tok decode: {text}");
            }
        }
        eprintln!("qwen35_gpu_greedy: PASS (8-tok GPU==CPU, run-twice stable)");
    }

    /// Decode tok/s benchmark, GPU resident lane vs CPU runnable lane, same Q4_K_M
    /// model. Two-point timing ((t_hi - t_lo) over (n_hi - n_lo) generated tokens)
    /// cancels the per-call prompt prefill + model-load + GPU build, leaving the
    /// steady-state per-token decode rate. Needs CAMELID_ORNITH_GGUF = the Q4_K_M.
    #[test]
    #[ignore = "tok/s benchmark — needs CAMELID_ORNITH_GGUF (Q4_K_M) + a CUDA device"]
    fn qwen35_gpu_vs_cpu_tokps() {
        let Ok(path) = std::env::var("CAMELID_ORNITH_GGUF") else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        if model.qwen35.is_none() {
            return;
        }
        let prompt: Vec<u32> = vec![3710, 369, 279, 6511, 314, 9338, 30];
        let (n_lo, n_hi) = (16usize, 64usize);
        let secs = |gpu: bool, n: usize| -> f64 {
            let t = std::time::Instant::now();
            let r = if gpu {
                model.generate_qwen35_cuda(&prompt, n, &[], None, &mut |_| {})
            } else {
                model.generate_qwen35_cpu(&prompt, n, &[], None, &mut |_| {})
            };
            r.expect("generate");
            t.elapsed().as_secs_f64()
        };
        // GPU: warm (lazy-build + 5.24 GB upload), then two timed runs.
        let _ = model.generate_qwen35_cuda(&prompt, 4, &[], None, &mut |_| {});
        let g_lo = secs(true, n_lo);
        let g_hi = secs(true, n_hi);
        let gpu_tokps = (n_hi - n_lo) as f64 / (g_hi - g_lo);
        // CPU: two timed runs.
        let c_lo = secs(false, n_lo);
        let c_hi = secs(false, n_hi);
        let cpu_tokps = (n_hi - n_lo) as f64 / (c_hi - c_lo);
        eprintln!(
            "qwen35 DECODE tok/s (Q4_K_M): GPU={gpu_tokps:.2}  CPU={cpu_tokps:.2}  \
             speedup={:.2}x  [GPU {g_lo:.1}s@{n_lo} -> {g_hi:.1}s@{n_hi}; \
             CPU {c_lo:.1}s@{n_lo} -> {c_hi:.1}s@{n_hi}]",
            gpu_tokps / cpu_tokps
        );
    }

    /// Sparse-KV long-context proof: build the resident engine at max_pos=8192 (which
    /// ONLY fits the 6 GB card because sparsify_kv frees the 24 SSM layers' KV — dense KV
    /// at 8192 is ~1 GB on top of the 5.24 GB Q4_K_M = OOM), prefill a >2048-token prompt
    /// (beyond the old dense cap, so the full-attn layers' KV is actually addressed past
    /// 2048), and decode a few tokens. Succeeding (no OOM, in-range tokens) proves sparse
    /// KV both fits and works. Point CAMELID_ORNITH_GGUF at the Q4_K_M.
    #[test]
    #[ignore = "needs CAMELID_ORNITH_GGUF (point at Q4_K_M) + a CUDA device"]
    fn qwen35_gpu_long_context_fits() {
        let Ok(path) = std::env::var("CAMELID_ORNITH_GGUF") else {
            return;
        };
        let model = RunnableModel::load(&path).expect("load qwen35");
        if model.qwen35.is_none() {
            return;
        }
        let max_pos = 8192usize; // only fits with sparse KV
        let mut e = match model.build_qwen35_resident(max_pos) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("build_qwen35_resident({max_pos}) failed (OOM => sparse KV not applied?): {err}");
                panic!("long-context build failed: {err}");
            }
        };
        e.reset_qwen35_state().unwrap();
        let scale = 1.0f32 / (model.head_dim as f32).sqrt();
        // A >2048-token prompt: repeat a short base until past the old dense cap.
        let base: Vec<u32> = vec![3710, 369, 279, 6511, 314, 9338, 30];
        let mut prompt: Vec<u32> = Vec::new();
        while prompt.len() < 2100 {
            prompt.extend_from_slice(&base);
        }
        let plen = prompt.len();
        assert!(
            plen > 2048,
            "prompt must exceed the old 2048 cap (got {plen})"
        );
        let mut next = 0u32;
        for (i, &tok) in prompt.iter().enumerate() {
            let emb = model
                .token_embd
                .dequant_row(tok as usize, "token_embd")
                .expect("embd");
            let (cos, sin) = super::qwen35_rope_tables(i, model.rope_base, model.rope_dim);
            let last = i == plen - 1;
            let out = e
                .forward_token(&emb, &cos, &sin, i, scale, last)
                .expect("gpu forward_token (long-ctx prefill)");
            if last {
                next = out.expect("logits on final prompt token");
            }
        }
        let mut decoded = vec![next];
        for step in 0..4 {
            let pos = plen + step;
            let emb = model
                .token_embd
                .dequant_row(next as usize, "token_embd")
                .expect("embd");
            let (cos, sin) = super::qwen35_rope_tables(pos, model.rope_base, model.rope_dim);
            next = e
                .forward_token(&emb, &cos, &sin, pos, scale, true)
                .expect("gpu forward_token (long-ctx decode)")
                .expect("logits");
            decoded.push(next);
        }
        eprintln!(
            "qwen35_gpu_long_context: prefilled {plen} tokens at max_pos={max_pos}, decoded {decoded:?}"
        );
        assert!(
            decoded.iter().all(|&t| (t as usize) < model.vocab),
            "decoded token out of range"
        );
    }
}

/// Env-gated real-row check that the runnable lane's per-layer RoPE schedule is
/// EXACTLY the reference 1B schedule (globals at 5/11/17/23 on base 1e6, every
/// other layer local on base 10000) — expectations are literal lists, not the
/// derivation formula. Run before AND after the Phase 1b metadata unification
/// (runnable `layer_rope_base` now derives from `cfg.gemma3` instead of local
/// constants) to prove the rewiring is bit-identical for the real row; the
/// printed forward fingerprint makes the before/after comparison exact.
#[cfg(test)]
mod gemma3_schedule_tests {
    use super::RunnableModel;

    #[test]
    fn gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule() {
        let Ok(path) = std::env::var("CAMELID_GEMMA3_GGUF") else {
            eprintln!(
                "SKIP gemma3_real_row_runnable_rope_schedule_is_the_reference_schedule: \
                 set CAMELID_GEMMA3_GGUF to the gemma-3-1b-it-Q8_0 GGUF"
            );
            return;
        };
        let model = RunnableModel::load(&path).expect("load gemma3 runnable model");
        assert_eq!(model.architecture, "gemma3");
        assert_eq!(model.n_layers, 26);

        // Literal expected bases: 1e6 on the four global layers, 10000 elsewhere.
        const G: f32 = 1_000_000.0;
        const L: f32 = 10_000.0;
        let expected: Vec<f32> = vec![
            L, L, L, L, L, G, // layers 0-5
            L, L, L, L, L, G, // layers 6-11
            L, L, L, L, L, G, // layers 12-17
            L, L, L, L, L, G, // layers 18-23
            L, L, // layers 24-25 (no forced-global final layer)
        ];
        assert_eq!(model.layer_rope_base, expected);
        assert!(model.rope_neox, "gemma3 runnable lane must pair NEOX");
        assert_eq!(model.embed_scale, Some((1152.0f32).sqrt()));

        // Bit-exact forward fingerprint over a short prompt: identical output
        // proves the schedule rewiring changed nothing for the real row.
        let logits = model.forward_logits(&[2, 651, 6037]).expect("forward");
        let sum_bits: u64 = logits
            .iter()
            .fold(0u64, |acc, v| acc.wrapping_add(v.to_bits() as u64));
        let head: Vec<String> = logits[..8]
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect();
        eprintln!(
            "gemma3 runnable forward fingerprint: len={} sum_bits={sum_bits:#018x} head={head:?}",
            logits.len()
        );
    }
}

#[cfg(test)]
mod command_r_attemptability_tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::{command_r_parallel_residual, validate_command_r_attemptability_slice};
    use crate::gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor, GgufTensorType};
    use crate::model::LlamaModelConfig;

    fn descriptor(
        name: impl Into<String>,
        tensor_type: GgufTensorType,
        dimensions: Vec<u64>,
    ) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions,
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 0,
        }
    }

    fn exact_aya_header_shape() -> (GgufFile, LlamaModelConfig) {
        let mut metadata = BTreeMap::new();
        let mut put = |key: &str, value| {
            metadata.insert(key.to_string(), value);
        };
        put(
            "general.architecture",
            GgufMetadataValue::String("command-r".into()),
        );
        put(
            "general.name",
            GgufMetadataValue::String("Aya Expanse 8b".into()),
        );
        put(
            "general.license",
            GgufMetadataValue::String("cc-by-nc-4.0".into()),
        );
        put("general.file_type", GgufMetadataValue::U32(15));
        put(
            "tokenizer.ggml.model",
            GgufMetadataValue::String("gpt2".into()),
        );
        put(
            "tokenizer.ggml.pre",
            GgufMetadataValue::String("command-r".into()),
        );
        put("command-r.context_length", GgufMetadataValue::U32(8_192));
        put("command-r.embedding_length", GgufMetadataValue::U32(4_096));
        put("command-r.block_count", GgufMetadataValue::U32(32));
        put(
            "command-r.feed_forward_length",
            GgufMetadataValue::U32(14_336),
        );
        put("command-r.attention.head_count", GgufMetadataValue::U32(32));
        put(
            "command-r.attention.head_count_kv",
            GgufMetadataValue::U32(8),
        );
        put(
            "command-r.attention.layer_norm_epsilon",
            GgufMetadataValue::F32(1e-5),
        );
        put("command-r.rope.freq_base", GgufMetadataValue::F32(10_000.0));
        put(
            "command-r.rope.scaling.type",
            GgufMetadataValue::String("none".into()),
        );
        put("command-r.logit_scale", GgufMetadataValue::F32(0.125));

        const Q6_DOWN_LAYERS: &[usize] =
            &[0, 1, 2, 3, 8, 10, 13, 16, 18, 21, 24, 27, 28, 29, 30, 31];
        const Q6_VALUE_LAYERS: &[usize] =
            &[0, 1, 2, 3, 6, 7, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31];
        let mut tensors = vec![
            descriptor(
                "token_embd.weight",
                GgufTensorType::Q6K,
                vec![4_096, 256_000],
            ),
            descriptor("output_norm.weight", GgufTensorType::F32, vec![4_096]),
        ];
        for layer in 0..32 {
            tensors.extend([
                descriptor(
                    format!("blk.{layer}.attn_norm.weight"),
                    GgufTensorType::F32,
                    vec![4_096],
                ),
                descriptor(
                    format!("blk.{layer}.attn_q.weight"),
                    GgufTensorType::Q4K,
                    vec![4_096, 4_096],
                ),
                descriptor(
                    format!("blk.{layer}.attn_k.weight"),
                    GgufTensorType::Q4K,
                    vec![4_096, 1_024],
                ),
                descriptor(
                    format!("blk.{layer}.attn_v.weight"),
                    if Q6_VALUE_LAYERS.contains(&layer) {
                        GgufTensorType::Q6K
                    } else {
                        GgufTensorType::Q4K
                    },
                    vec![4_096, 1_024],
                ),
                descriptor(
                    format!("blk.{layer}.attn_output.weight"),
                    GgufTensorType::Q4K,
                    vec![4_096, 4_096],
                ),
                descriptor(
                    format!("blk.{layer}.ffn_gate.weight"),
                    GgufTensorType::Q4K,
                    vec![4_096, 14_336],
                ),
                descriptor(
                    format!("blk.{layer}.ffn_up.weight"),
                    GgufTensorType::Q4K,
                    vec![4_096, 14_336],
                ),
                descriptor(
                    format!("blk.{layer}.ffn_down.weight"),
                    if Q6_DOWN_LAYERS.contains(&layer) {
                        GgufTensorType::Q6K
                    } else {
                        GgufTensorType::Q4K
                    },
                    vec![14_336, 4_096],
                ),
            ]);
        }
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.tensor_type == GgufTensorType::F32)
                .count(),
            33
        );
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.tensor_type == GgufTensorType::Q4K)
                .count(),
            192
        );
        assert_eq!(
            tensors
                .iter()
                .filter(|tensor| tensor.tensor_type == GgufTensorType::Q6K)
                .count(),
            33
        );
        let gguf = GgufFile {
            path: PathBuf::new(),
            version: 3,
            tensor_count: tensors.len() as i64,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors,
        };
        let config = LlamaModelConfig::from_gguf(&gguf).expect("exact Aya config");
        (gguf, config)
    }

    #[test]
    fn exact_aya_q4_k_m_header_shape_is_the_only_command_r_load_slice() {
        let (mut gguf, config) = exact_aya_header_shape();
        validate_command_r_attemptability_slice(&gguf, &config)
            .expect("pinned Aya header shape must be attemptable");

        gguf.tensors[2].name = "blk.0.attn_q_norm.weight".into();
        let err = validate_command_r_attemptability_slice(&gguf, &config)
            .expect_err("a QK-normalized neighboring Command R graph must fail closed");
        assert!(err.to_string().contains("unexpected tensor"));
    }

    #[test]
    fn aya_attemptability_rejects_wrong_canonical_dimensions_and_types() {
        let (mut wrong_dimensions, config) = exact_aya_header_shape();
        let q = wrong_dimensions
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.0.attn_q.weight")
            .expect("q descriptor");
        q.dimensions = vec![2_048, 4_096];
        let err = validate_command_r_attemptability_slice(&wrong_dimensions, &config)
            .expect_err("a truncated projection width must fail at header admission");
        assert!(err.to_string().contains("descriptor mismatch"));

        let (mut wrong_type, config) = exact_aya_header_shape();
        let v = wrong_type
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "blk.0.attn_v.weight")
            .expect("value descriptor");
        v.tensor_type = GgufTensorType::Q4K;
        let err = validate_command_r_attemptability_slice(&wrong_type, &config)
            .expect_err("the immutable row's per-tensor quant assignment must be exact");
        assert!(err.to_string().contains("descriptor mismatch"));
    }

    #[test]
    fn command_r_parallel_residual_preserves_reference_addition_order() {
        let mut output = [0.0f32];
        command_r_parallel_residual(&mut output, &[1e20], &[-1e20], &[3.0]);
        assert_eq!(output, [3.0]);

        let residual = 1e20f32;
        let attention = 3.0f32;
        let ffn = -1e20f32;
        assert_eq!((residual + attention) + ffn, 0.0);
    }
}
