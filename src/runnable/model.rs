//! Parametric pre-norm decoder, f32 only — the runnable lane's generic graph.
//!
//! One configurable transformer (parameterized from GGUF KV via [`LlamaModelConfig`]):
//! embeddings → N pre-norm blocks (RMSNorm → GQA attention with RoPE → RMSNorm →
//! SwiGLU FFN) → final RMSNorm → logits. No Metal/CUDA, no fused quantized kernels —
//! weights are dequantized to f32 ([`super::dequant`]) and run through naive f32 math.
//! Speed is the supported lane's job; this path's job is to be obviously correct and
//! deterministic so it can serve as the promotion oracle.
//!
//! Memory: weights stay resident in their compact **quantized** form; each layer's
//! matrices are dequantized to f32 once per forward pass and dropped, and the
//! embedding/output projections are done row-by-row. Peak ≈ raw weights + one layer
//! of f32, rather than the whole model as f32 — deliberate, so a small model fits a
//! tight RAM budget without thrashing (`RUNNABLE_LANE_SPEC.md` working-env guard).
//!
//! Phase 4 brings this up on **llama** (adjacent-pair RoPE, RMSNorm, SwiGLU, GQA).
//! Architecture-specific switches (qwen3 QK-norm / split-half RoPE, gemma norms +
//! soft-capping, phi3 fused QKV) land in Phase 6.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use rayon::prelude::*;

use crate::error::{BackendError, Result};
use crate::gguf::{read_metadata, GgufFile, GgufTensorDescriptor, GgufTensorType};
use crate::model::LlamaModelConfig;

use super::admit;

/// A 2-D weight kept in its quantized wire form. ggml layout: `ne = [in, out]`,
/// row-major with out feature `r` occupying one contiguous row of `in` values.
struct RawMat {
    bytes: Vec<u8>,
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
    /// y[r] = Σ_i data[r*in + i] * x[i].
    fn matvec(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_features);
        let mut y = vec![0.0f32; self.out_features];
        for (r, yr) in y.iter_mut().enumerate() {
            let row = &self.data[r * self.in_features..(r + 1) * self.in_features];
            *yr = dot(row, x);
        }
        y
    }
}

impl RawMat {
    fn row_bytes(&self) -> usize {
        self.bytes.len() / self.out_features
    }

    /// Dequantize the entire matrix to f32 (used per layer, dropped after the layer).
    fn dequant_all(&self, name: &str) -> Result<Mat> {
        let data = super::dequant::dequantize(
            self.tt,
            &self.bytes,
            self.in_features * self.out_features,
            name,
        )?;
        Ok(Mat {
            data,
            in_features: self.in_features,
            out_features: self.out_features,
        })
    }

    /// Dequantize a single row `r` (length `in_features`) — for embedding lookup and
    /// the output projection, which touch the huge vocab matrix one row at a time.
    fn dequant_row(&self, r: usize, name: &str) -> Result<Vec<f32>> {
        let rb = self.row_bytes();
        let slice = &self.bytes[r * rb..(r + 1) * rb];
        super::dequant::dequantize(self.tt, slice, self.in_features, name)
    }

    /// Carve out a contiguous block of `len` out-features starting at `start` into a
    /// new RawMat. Used to split phi3's fused `attn_qkv` and fused `gate_up` into the
    /// separate projections the generic block expects. Valid because rows are
    /// out-feature-major and each row is a whole number of quant blocks.
    fn split_rows(&self, start: usize, len: usize) -> RawMat {
        let rb = self.row_bytes();
        RawMat {
            bytes: self.bytes[start * rb..(start + len) * rb].to_vec(),
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
                    .map(|r| q8_0_wire_dot(&self.bytes[r * rb..(r + 1) * rb], &xq))
                    .collect())
            }
            GgufTensorType::F32 => Ok((0..self.out_features)
                .into_par_iter()
                .map(|r| f32_row_dot(&self.bytes[r * rb..(r + 1) * rb], x))
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
                        let row = &self.bytes[r * rb..(r + 1) * rb];
                        xqs.iter().map(move |xq| q8_0_wire_dot(row, xq))
                    })
                    .collect()
            }
            GgufTensorType::F32 => (0..out_f)
                .into_par_iter()
                .flat_map_iter(|r| {
                    let row = &self.bytes[r * rb..(r + 1) * rb];
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
}

impl KvCache {
    fn new(n_layers: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
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
    /// Lazily-built GPU resident decode engine for the qwen35 lane, gated by
    /// `CAMELID_QWEN35_CUDA=1`. `Mutex` gives the `&mut` the per-token forward needs
    /// while `generate_*` take `&self`; built on first use and reused, with the SSM/conv
    /// recurrent state reset at the start of every generate. `None` until first use.
    #[cfg(feature = "cuda")]
    cuda: std::sync::Mutex<Option<crate::cuda_resident::CudaResidentDecode>>,
}

impl RunnableModel {
    /// Admit, parse config, and read every weight into resident quantized form.
    pub fn load(path: &str) -> Result<Self> {
        let gguf = read_metadata(path)?;
        admit::admit(&gguf).map_err(BackendError::from)?;
        let cfg = LlamaModelConfig::from_gguf(&gguf)?;
        let arch = gguf
            .architecture()
            .ok_or_else(|| BackendError::InvalidModelMetadata("missing architecture".into()))?
            .to_string();

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

        if let Some(kind) = cfg.rope_scaling_type.as_deref() {
            if !kind.is_empty() && kind != "none" {
                // Phase 4 brings up plain llama (TinyLlama: no scaling). linear/yarn/
                // llama3 scaling is a named Phase 6 follow-up, not silently ignored.
                return Err(BackendError::UnsupportedGguf(format!(
                    "runnable lane: rope scaling {kind:?} not yet implemented (Phase 6)"
                )));
            }
        }

        let mut f = File::open(path).map_err(|e| BackendError::Io {
            path: path.into(),
            source: e,
        })?;

        let load_raw = |f: &mut File, name: &str| -> Result<RawMat> {
            let d = find_tensor(&gguf, name)?;
            let (inf, outf) = mat_dims(d, name)?;
            Ok(RawMat {
                bytes: read_tensor_bytes(f, d, name)?,
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
        let output_norm = load_vec(&mut f, "output_norm.weight")?;

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

            return Ok(Self {
                architecture: arch,
                d_model,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_dim,
                rope_base,
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
                    d_conv,
                    d_state,
                    num_k_heads,
                    num_v_heads,
                    head_v_dim,
                    key_dim,
                    value_dim,
                    conv_dim,
                }),
                #[cfg(feature = "cuda")]
                cuda: std::sync::Mutex::new(None),
            });
        }

        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let ffn = cfg.feed_forward_length as usize;

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

            layers.push(Layer {
                attn_norm: load_vec(&mut f, &p("attn_norm"))?,
                ffn_norm: load_vec(&mut f, &p("ffn_norm"))?,
                wq,
                wk,
                wv,
                wo: load_raw(&mut f, &p("attn_output"))?,
                gate,
                up,
                down: load_raw(&mut f, &p("ffn_down"))?,
                q_norm: load_vec_opt(&mut f, &p("attn_q_norm"))?,
                k_norm: load_vec_opt(&mut f, &p("attn_k_norm"))?,
                post_attn_norm: load_vec_opt(&mut f, &p("post_attention_norm"))?,
                post_ffn_norm: load_vec_opt(&mut f, &p("post_ffw_norm"))?,
            });
        }

        let is_gemma = arch.starts_with("gemma");
        let is_command_r = arch == "command-r";
        // RoPE pairing: NEOX (split-half) for qwen3/gemma/phi3/command-r (unpermuted weights);
        // LLAMA (interleaved) for standard variants.
        let rope_neox = cfg.rope_neox_pairing || is_gemma || arch == "phi3" || is_command_r;
        // gemma3 dual RoPE: every Nth layer (sliding_window_pattern, default 6) is a
        // GLOBAL-attention layer using the GGUF freq_base (1e6); the rest are local
        // sliding-window layers using the gemma3 default local base (10000). The
        // sliding window itself is a no-op for prompts shorter than the window.
        let layer_rope_base = if arch == "gemma3" {
            let global = rope_base;
            let local = 10_000.0_f32;
            let pattern = 6usize;
            (0..n_layers)
                .map(|i| {
                    if (i + 1) % pattern == 0 {
                        global
                    } else {
                        local
                    }
                })
                .collect()
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
            #[cfg(feature = "cuda")]
            cuda: std::sync::Mutex::new(None),
        })
    }

    fn apply_norm(&self, x: &[f32], weight: &[f32]) -> Vec<f32> {
        if self.architecture == "command-r" {
            layer_norm(x, weight, self.eps)
        } else {
            rms_norm(x, weight, self.eps)
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
        let seq = tokens.len();
        let dm = self.d_model;

        // Embedding lookup (one dequantized row per token).
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
            // gemma scales embeddings by sqrt(d_model).
            if let Some(scale) = self.embed_scale {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
            hidden[pos * dm..(pos + 1) * dm].copy_from_slice(&row);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            self.attention_block(layer, li, &mut hidden, seq)?;
            self.ffn_block(layer, li, &mut hidden, seq)?;
        }

        // Final norm on the last position, then logits (one dequantized row per vocab).
        let last = &hidden[(seq - 1) * dm..seq * dm];
        let normed = self.apply_norm(last, &self.output_norm);
        let mut logits = vec![0.0f32; self.vocab];
        for (t, lt) in logits.iter_mut().enumerate() {
            let row = self.output.dequant_row(t, "output")?;
            *lt = dot(&row, &normed);
        }
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
        let mut cache = KvCache::new(self.n_layers);
        let mut last = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last = self.forward_step(tok, pos, &mut cache)?;
        }
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

    /// Incremental forward of a single token at absolute `pos`, appending its K/V to
    /// `cache` and attending over all cached positions. Returns next-token logits.
    fn forward_step(&self, token: u32, pos: usize, cache: &mut KvCache) -> Result<Vec<f32>> {
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
            let xn = self.apply_norm(&hidden, &layer.attn_norm);
            let wq = layer.wq.dequant_all(&name(li, "attn_q"))?;
            let wk = layer.wk.dequant_all(&name(li, "attn_k"))?;
            let wv = layer.wv.dequant_all(&name(li, "attn_v"))?;
            let wo = layer.wo.dequant_all(&name(li, "attn_output"))?;
            let mut qp = wq.matvec(&xn);
            let mut kp = wk.matvec(&xn);
            let vp = wv.matvec(&xn);
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
            let mut proj = wo.matvec(&attn_out);
            if let Some(pn) = &layer.post_attn_norm {
                proj = self.apply_norm(&proj, pn);
            }
            for (h, p) in hidden.iter_mut().zip(proj.iter()) {
                *h += *p;
            }

            // --- FFN ---
            let xn2 = self.apply_norm(&hidden, &layer.ffn_norm);
            let gate = layer.gate.dequant_all(&name(li, "ffn_gate"))?;
            let up = layer.up.dequant_all(&name(li, "ffn_up"))?;
            let down = layer.down.dequant_all(&name(li, "ffn_down"))?;
            let g = gate.matvec(&xn2);
            let u = up.matvec(&xn2);
            let mut act = vec![0.0f32; g.len()];
            for i in 0..g.len() {
                let gated = if self.ffn_gelu {
                    gelu_tanh(g[i])
                } else {
                    g[i] / (1.0 + (-g[i]).exp())
                };
                act[i] = gated * u[i];
            }
            let mut d = down.matvec(&act);
            if let Some(pn) = &layer.post_ffn_norm {
                d = self.apply_norm(&d, pn);
            }
            for (h, dv) in hidden.iter_mut().zip(d.iter()) {
                *h += *dv;
            }
        }

        let normed = self.apply_norm(&hidden, &self.output_norm);
        let mut logits = vec![0.0f32; self.vocab];
        for (tk, lt) in logits.iter_mut().enumerate() {
            let row = self.output.dequant_row(tk, "output")?;
            *lt = dot(&row, &normed);
        }
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

    fn attention_block(
        &self,
        layer: &Layer,
        li: usize,
        hidden: &mut [f32],
        seq: usize,
    ) -> Result<()> {
        let dm = self.d_model;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = self.n_heads / self.n_kv_heads;
        let q_dim = self.n_heads * hd;
        let kv_dim = self.n_kv_heads * hd;

        // Dequantize this layer's projection weights once (dropped at block end).
        let wq = layer.wq.dequant_all(&name(li, "attn_q"))?;
        let wk = layer.wk.dequant_all(&name(li, "attn_k"))?;
        let wv = layer.wv.dequant_all(&name(li, "attn_v"))?;
        let wo = layer.wo.dequant_all(&name(li, "attn_output"))?;

        let mut q = vec![0.0f32; seq * q_dim];
        let mut k = vec![0.0f32; seq * kv_dim];
        let mut v = vec![0.0f32; seq * kv_dim];
        for pos in 0..seq {
            let x = &hidden[pos * dm..(pos + 1) * dm];
            let xn = self.apply_norm(x, &layer.attn_norm);
            let mut qp = wq.matvec(&xn);
            let mut kp = wk.matvec(&xn);
            let vp = wv.matvec(&xn);
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
            v[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&vp);
        }

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
            let mut proj = wo.matvec(&attn_out);
            // gemma: post-attention RMSNorm before the residual add.
            if let Some(pn) = &layer.post_attn_norm {
                proj = self.apply_norm(&proj, pn);
            }
            let dst = &mut hidden[pos * dm..(pos + 1) * dm];
            for (h, p) in dst.iter_mut().zip(proj.iter()) {
                *h += *p;
            }
        }
        Ok(())
    }

    fn ffn_block(&self, layer: &Layer, li: usize, hidden: &mut [f32], seq: usize) -> Result<()> {
        let dm = self.d_model;
        let gate = layer.gate.dequant_all(&name(li, "ffn_gate"))?;
        let up = layer.up.dequant_all(&name(li, "ffn_up"))?;
        let down = layer.down.dequant_all(&name(li, "ffn_down"))?;
        for pos in 0..seq {
            let x = &hidden[pos * dm..(pos + 1) * dm];
            let xn = self.apply_norm(x, &layer.ffn_norm);
            let g = gate.matvec(&xn);
            let u = up.matvec(&xn);
            // Gated FFN: gemma uses GeGLU (gelu-tanh), llama uses SwiGLU (silu).
            let mut act = vec![0.0f32; g.len()];
            for i in 0..g.len() {
                let gated = if self.ffn_gelu {
                    gelu_tanh(g[i])
                } else {
                    g[i] / (1.0 + (-g[i]).exp())
                };
                act[i] = gated * u[i];
            }
            let mut d = down.matvec(&act);
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
                let angle = pos as f32 * freq;
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
// `src/models/qwen35.cpp` + `delta-net-base.cpp`) in pure f32 for the runnable lane.
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
    d_conv: usize,      // causal conv kernel width (4)
    d_state: usize,     // per-head state dim = head_k_dim = head_v_dim (128)
    num_k_heads: usize, // key/query heads / groups (16)
    num_v_heads: usize, // value/delta heads (32)
    head_v_dim: usize,  // d_inner / num_v_heads (= d_state, 128)
    key_dim: usize,     // d_state * num_k_heads (2048)
    value_dim: usize,   // head_v_dim * num_v_heads (= d_inner, 4096)
    conv_dim: usize,    // 2*key_dim + value_dim (8192)
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
    /// qwen35 greedy decode. Routes to the GPU resident lane when `CAMELID_QWEN35_CUDA=1`
    /// (lazy-built, reused, recurrent state reset per call), and falls back to the CPU
    /// runnable lane on any CUDA error. The CPU lane is the certified oracle and default.
    fn generate_qwen35(&self, prompt: &[u32], max_new: usize, stop: &[u32]) -> Result<Vec<u32>> {
        self.generate_qwen35_streaming(prompt, max_new, stop, &mut |_| {})
    }

    /// Like [`generate_qwen35`](Self::generate_qwen35) but invokes `on_token` for
    /// every emitted token as soon as it is decided — the serve lane's SSE source.
    /// Token order/content identical to the non-streaming path by construction.
    fn generate_qwen35_streaming(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        #[cfg(feature = "cuda")]
        {
            if std::env::var("CAMELID_QWEN35_CUDA")
                .map(|v| v == "1")
                .unwrap_or(false)
            {
                match self.generate_qwen35_cuda(prompt, max_new, stop, on_token) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        eprintln!("[qwen35] CUDA lane failed ({e}); falling back to CPU");
                    }
                }
            }
        }
        self.generate_qwen35_cpu(prompt, max_new, stop, on_token)
    }

    fn generate_qwen35_cpu(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) -> Result<Vec<u32>> {
        // Batched prefill of the whole prompt (weights read once per layer), then
        // per-token greedy decode from the resulting cache.
        let (mut cache, last) = self.prefill_qwen35(prompt)?;
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        let mut next = argmax(&last);
        for i in 0..max_new {
            // A stop token (EOS / `<|im_end|>` / EOG) ends the turn — and is NOT
            // appended, matching llama.cpp's served output (the stop is consumed).
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            if i + 1 < max_new {
                let logits = self.decode_token_qwen35(next, pos, &mut cache, true)?;
                pos += 1;
                next = argmax(&logits);
            }
        }
        Ok(out)
    }

    /// GPU resident greedy decode for qwen35. Builds the engine on first use (cached in
    /// `self.cuda`), resets the recurrent SSM/conv state, then prefills the prompt and
    /// greedily decodes via `forward_token`. Token-parity with the CPU lane (not
    /// bit-identical: int8 activation quant + FP-reassociated attention).
    #[cfg(feature = "cuda")]
    fn generate_qwen35_cuda(
        &self,
        prompt: &[u32],
        max_new: usize,
        stop: &[u32],
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
        if engine.device_decode_ready() {
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
                        scale,
                        if last { Some(0) } else { None },
                    )
                    .map_err(BackendError::InvalidTensorData)?;
            }
            let chunk_len: usize = std::env::var("CAMELID_DEVICE_DECODE_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&c| c >= 1)
                .unwrap_or(8);
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
                        .forward_token_device(Some(t + j), None, pos, scale, Some(t + j + 1))
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
        // ---- Host-fed fallback loop (device tables unavailable) -----------
        // Prefill: only the final prompt token needs logits.
        let mut next = 0u32;
        for (i, &tok) in prompt.iter().enumerate() {
            let emb = self.token_embd.dequant_row(tok as usize, "token_embd")?;
            let (cos, sin) = qwen35_rope_tables(i, self.rope_base, self.rope_dim);
            let last = i == prompt.len() - 1;
            let out = engine
                .forward_token(&emb, &cos, &sin, i, scale, last)
                .map_err(BackendError::InvalidTensorData)?;
            if last {
                next = out.ok_or_else(|| {
                    BackendError::InvalidTensorData("no logits on final prompt token".into())
                })?;
            }
        }
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        for i in 0..max_new {
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            if i + 1 < max_new {
                let emb = self.token_embd.dequant_row(next as usize, "token_embd")?;
                let (cos, sin) = qwen35_rope_tables(pos, self.rope_base, self.rope_dim);
                next = engine
                    .forward_token(&emb, &cos, &sin, pos, scale, true)
                    .map_err(BackendError::InvalidTensorData)?
                    .ok_or_else(|| {
                        BackendError::InvalidTensorData("no logits on decode step".into())
                    })?;
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
        self.generate_stopping_streaming(prompt, max_new, stop, &mut |_| {})
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
        if prompt.is_empty() {
            return Err(BackendError::InvalidTensorData("empty prompt".into()));
        }
        if self.qwen35.is_some() {
            return self.generate_qwen35_streaming(prompt, max_new, stop, on_token);
        }
        // Dense/generic path (MUSTER M-A1): the same incremental loop as `generate`
        // (byte-identical forward sequence when no stop token fires) plus stop-token
        // handling and true per-token streaming. A stop token ends the turn and is
        // NOT appended — matching the qwen35 lane and llama.cpp's served output.
        let mut cache = KvCache::new(self.n_layers);
        let mut last = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            last = self.forward_step(tok, pos, &mut cache)?;
        }
        let mut out = Vec::with_capacity(max_new);
        let mut pos = prompt.len();
        let mut next = argmax(&last);
        while out.len() < max_new {
            if stop.contains(&next) {
                break;
            }
            out.push(next);
            on_token(next);
            if out.len() < max_new {
                let logits = self.forward_step(next, pos, &mut cache)?;
                pos += 1;
                next = argmax(&logits);
            }
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

/// qwen35 (Ornith) partial-NEOX RoPE cos/sin tables for absolute `pos`, length
/// `rope_dim/2`. VERBATIM `apply_rope`: `1.0/base.powf(2.0*i/rope_dim)` then
/// `(pos*freq).sin_cos()` (sin first) — do NOT use the negated-exponent form the
/// Llama lane uses (last-ULP drift can flip a near-tie greedy token).
#[cfg(feature = "cuda")]
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
        // through raw (the q2k/q3k/q4k/q5k/q6k GEMV kernels expand them on the fly).
        // Q5_K previously had no resident GEMV and was upcast to Q8_0 blocks at load
        // (~+40 MiB on stock Q3_K_M's four q5_K tensors); with `q5k_gemv` in-tree it
        // rides its own lane at wire size. Returns (repack-ready bytes, lane).
        let prep = |m: &RawMat| -> std::result::Result<(Vec<u8>, ProjQuant), String> {
            match m.tt {
                GgufTensorType::Q8_0 => Ok((widen_q8(&m.bytes), ProjQuant::Q8_0)),
                GgufTensorType::Q2K => Ok((m.bytes.clone(), ProjQuant::Q2K)),
                GgufTensorType::Q3K => Ok((m.bytes.clone(), ProjQuant::Q3K)),
                GgufTensorType::Q4K => Ok((m.bytes.clone(), ProjQuant::Q4K)),
                GgufTensorType::Q5K => Ok((m.bytes.clone(), ProjQuant::Q5K)),
                GgufTensorType::Q6K => Ok((m.bytes.clone(), ProjQuant::Q6K)),
                other => Err(format!(
                    "qwen35 CUDA lane: unsupported projection quant {other:?}"
                )),
            }
        };
        let mut e = CudaResidentDecode::new(
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
        for layer in &rt.layers {
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
                        true,
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
                        rt.conv_dim,
                        rt.d_conv,
                        rt.num_v_heads,
                        rt.d_state,
                    )?;
                }
            }
        }
        let (bout, qout) = prep(&self.output)?;
        e.set_output(&self.output_norm, &bout, qout)?;
        // Device-side decode loop: resident quantized embedding table + the
        // all-positions rope tables (built with the VERBATIM qwen35_rope_tables
        // math, so the rope inputs are bit-identical to the host-fed path).
        // Failure (e.g. VRAM headroom) is non-fatal — the driver falls back to
        // the host-fed per-token loop.
        let embd_upload: Option<(Vec<u8>, ProjQuant)> = match self.token_embd.tt {
            GgufTensorType::Q8_0 => Some((self.token_embd.bytes.clone(), ProjQuant::Q8_0)),
            GgufTensorType::Q3K => Some((self.token_embd.bytes.clone(), ProjQuant::Q3K)),
            GgufTensorType::Q4K => Some((self.token_embd.bytes.clone(), ProjQuant::Q4K)),
            GgufTensorType::Q6K => Some((
                crate::cuda_resident::pad_q6k_blocks(&self.token_embd.bytes),
                ProjQuant::Q6K,
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
        let up = |m: &RawMat| s.clone_htod(&repack_q8_soa(&widen_q8(&m.bytes))).unwrap();
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
        let up = |m: &RawMat| s.clone_htod(&repack_q8_soa(&widen_q8(&m.bytes))).unwrap();
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
        let mut mix_q = s.alloc_zeros::<i8>(q_width).unwrap();
        let mut mix_s = s.alloc_zeros::<f32>(qb).unwrap();
        let mut d_mix = s.alloc_zeros::<f32>(hidden_dim).unwrap();

        // persistent KV cache (f16 bits, layout [kv_head][position][head_dim]) +
        // device position + per-pair RoPE tables.
        let mut cache_k = s.alloc_zeros::<u16>(kv_width * max_pos).unwrap();
        let mut cache_v = s.alloc_zeros::<u16>(kv_width * max_pos).unwrap();
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
            .generate_qwen35_cpu(&prompt, n, &[], &mut |_| {})
            .expect("cpu gen");
        let gpu = model
            .generate_qwen35_cuda(&prompt, n, &[], &mut |_| {})
            .expect("gpu gen");
        // run-twice reuses the cached engine -> validates reset_qwen35_state.
        let gpu2 = model
            .generate_qwen35_cuda(&prompt, n, &[], &mut |_| {})
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
                    .generate_qwen35_cuda(&prompt, 16, &[], &mut |_| {})
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
                model.generate_qwen35_cuda(&prompt, n, &[], &mut |_| {})
            } else {
                model.generate_qwen35_cpu(&prompt, n, &[], &mut |_| {})
            };
            r.expect("generate");
            t.elapsed().as_secs_f64()
        };
        // GPU: warm (lazy-build + 5.24 GB upload), then two timed runs.
        let _ = model.generate_qwen35_cuda(&prompt, 4, &[], &mut |_| {});
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
