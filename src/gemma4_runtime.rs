//! Gemma 4 inference runtime — loads a gemma4 GGUF and generates text.
//!
//! The forward math is the one validated bit-for-bit against llama.cpp in
//! `tests/gemma4_forward.rs` (prompt "The capital of France is" → " Paris..."),
//! here driven by an **incremental KV cache**: each [`Gemma4Runtime::step`]
//! processes one token at one position, so the 8GB of Q8 weights are read once
//! per generated token (O(n)) rather than re-prefilled (O(n²)).
//!
//! Weights stay Q8_0 in memory (the model fits in ~8GB; full f32 would not fit a
//! 16GB box); matmuls dequantize on the fly via [`q8_matvec`]. Cross-layer KV
//! sharing: layers >= `first_kv_shared` reuse the last same-type layer's cache.

use crate::gguf::{read_metadata, GgufTensorType};
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::inference::{
    nvfp4_wire_block_dequant, nvfp4_wire_row_dot, q4_0_wire_block_dequant, q4_0_wire_row_dot,
    q4_1_wire_row_dot, q4_k_wire_row_dot, q6_k_wire_block_dequant, q6_k_wire_row_dot,
    q8_0_wire_row_dot, quantize_q8_0_blocks, quantize_q8_k_blocks,
};
use crate::model::{Gemma4Binding, Gemma4Metadata, LlamaModelConfig};
use crate::tensor::{f16_bits_to_f32, Q8_0Block, TensorStore};
use crate::tokenizer::Tokenizer;
use crate::wire_mmap::GgufWireMmap;
use crate::{BackendError, Result};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Q8_0 wire-block geometry (GGUF on-disk format): 32 quantized values per block,
/// stored as a 2-byte little-endian f16 scale followed by 32 i8 quants = 34 bytes.
const Q8_VALUES_PER_BLOCK: usize = 32;
const Q8_WIRE_BYTES_PER_BLOCK: usize = 34;

/// The wire quant formats the gemma4 CPU runtime reads in place. Q8_0 is the
/// proven baseline lane; Q4_0 and Q6_K are the QAT-row formats (all the QAT
/// linear weights are Q4_0; the tied token/per-layer embeddings are Q6_K).
/// NVFP4 is the BASALT pilot matmul-weight format (D17/D-B1: pin `block_nvfp4`
/// byte-for-byte, 64-element/36-byte superblocks; the pilot's embeddings/norms
/// stay in the Q8_0 baseline formats per `basalt_eval_protocol.md` §1).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WireFormat {
    Q8_0,
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
    Nvfp4,
}

impl WireFormat {
    #[inline]
    fn values_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 => 32,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => 256,
            WireFormat::Nvfp4 => crate::tensor::NVFP4_VALUES_PER_BLOCK, // 64
        }
    }

    #[inline]
    fn bytes_per_block(self) -> usize {
        match self {
            WireFormat::Q8_0 => Q8_WIRE_BYTES_PER_BLOCK,
            WireFormat::Q4_0 => crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK,
            // block_q4_1 = f16 d + f16 m + 16 nibbles; Q4_K/Q5_K K-quant superblocks.
            WireFormat::Q4_1 => 20,
            WireFormat::Q4K => 144,
            WireFormat::Q5K => 176,
            WireFormat::Q6K => crate::inference::Q6_K_WIRE_BYTES_PER_BLOCK,
            WireFormat::Nvfp4 => crate::tensor::NVFP4_WIRE_BYTES_PER_BLOCK, // 36
        }
    }

    /// Wire bytes of one weight row spanning `q8_blocks` Q8_0 activation blocks
    /// (32 values each) — for the Q8_0-activation matvec family only. For the
    /// 32-value formats this is `q8_blocks * bytes_per_block`; one 64-value
    /// NVFP4 superblock spans TWO activation blocks, so its row is
    /// `q8_blocks / 2` superblocks wide.
    #[inline]
    fn row_bytes_for_q8_blocks(self, q8_blocks: usize) -> usize {
        debug_assert!(
            (q8_blocks * Q8_VALUES_PER_BLOCK).is_multiple_of(self.values_per_block()),
            "row of {q8_blocks} Q8_0 blocks is not whole {self:?} blocks"
        );
        q8_blocks * Q8_VALUES_PER_BLOCK / self.values_per_block() * self.bytes_per_block()
    }
}

/// A quantized weight read straight from the memory-mapped GGUF — no eager
/// decode and no second resident copy. The mmap pages fault in on first touch
/// (during the first generation) and stay in the OS page cache after, so
/// `load()` is ~instant instead of spending ~240s materializing 8GB of decoded
/// blocks up front. Dequant happens inline in the matmul — only the block
/// scale is decoded per block per pass (negligible next to the mul-adds it
/// scales). Any tensor type outside [`WireFormat`] fails closed at load.
struct WireQuant {
    mmap: Arc<GgufWireMmap>,
    offset: u64,
    element_count: usize,
    format: WireFormat,
}

impl WireQuant {
    fn new(store: &TensorStore, mmap: &Arc<GgufWireMmap>, name: &str) -> Result<Self> {
        let desc = store.descriptor(name)?;
        let format = match desc.tensor_type {
            GgufTensorType::Q8_0 => WireFormat::Q8_0,
            GgufTensorType::Q4_0 => WireFormat::Q4_0,
            GgufTensorType::Q4_1 => WireFormat::Q4_1,
            GgufTensorType::Q4K => WireFormat::Q4K,
            GgufTensorType::Q5K => WireFormat::Q5K,
            GgufTensorType::Q6K => WireFormat::Q6K,
            GgufTensorType::NVFP4 => WireFormat::Nvfp4,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and NVFP4"
                )))
            }
        };
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        if !element_count.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} element count {element_count} is not block-aligned"
            )));
        }
        let byte_len = element_count / format.values_per_block() * format.bytes_per_block();
        if desc.n_bytes as usize != byte_len {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} {format:?} byte size {} != expected {byte_len}",
                desc.n_bytes
            )));
        }
        // Validate the whole tensor range lies inside the mapping once, so the
        // hot-path `bytes()` can index without re-checking.
        mmap.bytes(desc.absolute_offset, byte_len)?;
        // BASALT D17/T5 fail-closed: NVFP4 files carrying NaN-sentinel UE4M3
        // scale bytes (0x7F/0xFF) are refused at load — the pin's own CPU and
        // CUDA backends disagree on 0xFF, so such a file has no well-defined
        // cross-backend oracle. The runnable lane refuses inside
        // `decode_nvfp4_tensor`; this wire lane never runs that decoder (the
        // matvec reads wire bytes in place), so the scan lives here. One
        // sequential pass over the tensor's mapped bytes (which the first
        // generation would fault in anyway); zero scales admit.
        if format == WireFormat::Nvfp4 {
            let wire = mmap.bytes(desc.absolute_offset, byte_len)?;
            if let Some(block_idx) = crate::tensor::nvfp4_find_nan_scale(wire) {
                return Err(BackendError::InvalidTensorData(format!(
                    "tensor {name}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 \
                     scale byte (0x7F/0xFF) — refusing per D17/T5 (fail closed at load)"
                )));
            }
        }
        Ok(Self {
            mmap: mmap.clone(),
            offset: desc.absolute_offset,
            element_count,
            format,
        })
    }

    /// Typed load-time guard for weights bound to a matvec/matmul role
    /// (projection, expert band, or tied head). Q5_K is GATHER-ONLY in this
    /// lane (`per_layer_token_embd`; no Q5_K row-dot kernel is wired here), so
    /// admitting it into a matvec role would surface as a forward-time panic —
    /// refuse it at load instead (invariant I-unknown-type: typed refusal,
    /// never a reachable panic). Every other [`WireFormat`] has a matvec route.
    fn require_matvec_capable(self, name: &str) -> Result<Self> {
        if self.format == WireFormat::Q5K {
            return Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is Q5_K; the gemma4 wire lane serves Q5_K gather-only \
                 (per_layer_token_embd) — it cannot be a projection/head weight"
            )));
        }
        Ok(self)
    }

    /// The tensor's full wire-byte slice. Bounds were validated in `new`.
    #[inline]
    fn bytes(&self) -> &[u8] {
        let byte_len =
            self.element_count / self.format.values_per_block() * self.format.bytes_per_block();
        self.mmap
            .bytes(self.offset, byte_len)
            .expect("wire quant range validated at load")
    }

    #[inline]
    fn block_scale(bytes: &[u8], block: usize) -> f32 {
        let b = block * Q8_WIRE_BYTES_PER_BLOCK;
        f16_bits_to_f32(u16::from_le_bytes([bytes[b], bytes[b + 1]]))
    }

    /// y[o] = sum_i dequant(W[o*in + i]) * x[i]. Rows are block-aligned
    /// (in % 32 == 0). The activation `x` is quantized to Q8 once, then each
    /// output row is a Q8×Q8 NEON `sdot` against the weight row read in place
    /// from the wire bytes ([`q8_0_wire_row_dot`]) — the same fast i8 dot the
    /// Llama path uses, ~Nx the prior scalar f32 mul-add per block. Quantizing
    /// the activation mirrors what llama.cpp does for Q8_0 matmuls, so the
    /// bit-against-llama.cpp parity in `tests/gemma4_forward.rs` is preserved.
    fn matvec(&self, in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(
            in_dim % self.format.values_per_block(),
            0,
            "matvec assumes block-aligned rows"
        );
        match self.format {
            // NVFP4 rides the Q8_0-activation family: the pin's
            // `ggml_vec_dot_nvfp4_q8_0_generic` dots NVFP4 superblocks against
            // Q8_0 activation blocks, exactly like Q8_0/Q4_0/Q4_1.
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q(out_dim, &quantize_q8_0_blocks(x))
            }
            // K-quant rows dot against Q8_K activations (the reference's K-quant
            // activation format) — Q6_K/Q4_K used by the QAT tied embedding head.
            WireFormat::Q4K | WireFormat::Q6K => self.matvec_q8k(out_dim, &quantize_q8_k_blocks(x)),
            // Q5_K is gather-only here (per_layer_token_embd); never a matvec
            // weight — `require_matvec_capable` refuses it typed at load.
            WireFormat::Q5K => unreachable!("Q5_K is gather-only (per_layer_token_embd)"),
        }
    }

    /// One projection off a [`SharedActivation`], routed by the SAME family
    /// split as the top-level [`Self::matvec`]: K-quant weights (Q4_K/Q6_K)
    /// dot Q8_K activations via [`Self::matvec_q8k`], everything else keeps
    /// the Q8_0-activation fast path via [`Self::matvec_q`] byte-for-byte.
    ///
    /// This is the SHA_E3 crash fix: the per-layer projection call sites used
    /// to pre-quantize the shared activation to Q8_0 once and call `matvec_q`
    /// directly, which has no K-quant arms — a latent pre-BASALT gap (no
    /// gemma4 K-quant matmul row existed) that panicked `unreachable!` at
    /// forward time on the campaign's Q4K-mm/Q4_K_M rows. The shared
    /// activation is still quantized at most once PER FAMILY per call site
    /// (lazily), so single-family files pay exactly the old quantize count.
    fn matvec_proj(&self, out_dim: usize, x: &SharedActivation) -> Vec<f32> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q(out_dim, x.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => self.matvec_q8k(out_dim, x.q8_k()),
            // Structurally unreachable: `require_matvec_capable` refuses Q5_K
            // in every matvec-role binding at load (typed, I-unknown-type).
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
        }
    }

    /// Batched sibling of [`Self::matvec_proj`] for the spec-verify chunk
    /// path: routes to [`Self::matmul_q`] / [`Self::matmul_q8k`] by the same
    /// family split, off a [`SharedActivationBatch`].
    fn matmul_proj(&self, out_dim: usize, xs: &SharedActivationBatch) -> Vec<Vec<f32>> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matmul_q(out_dim, xs.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => self.matmul_q8k(out_dim, xs.q8_k()),
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
        }
    }

    /// Row-band sibling of [`Self::matvec_proj`] for the MoE expert matrices:
    /// routes to [`Self::matvec_q_rows`] / [`Self::matvec_q8k_rows`] by the
    /// same family split.
    fn matvec_rows_proj(
        &self,
        row_start: usize,
        out_count: usize,
        x: &SharedActivation,
    ) -> Vec<f32> {
        match self.format {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {
                self.matvec_q_rows(row_start, out_count, x.q8_0())
            }
            WireFormat::Q4K | WireFormat::Q6K => {
                self.matvec_q8k_rows(row_start, out_count, x.q8_k())
            }
            WireFormat::Q5K => unreachable!("Q5_K matvec roles are refused at load"),
        }
    }

    /// [`matvec`] against an activation already quantized to Q8 blocks. Lets a
    /// caller that runs several projections off one activation (q/k/v share the
    /// pre-attention norm; gate/up share the pre-FFN norm) quantize it a single
    /// time instead of once per projection.
    ///
    /// Rows are processed in fixed chunks rather than one rayon task per row:
    /// the 262K-vocab output projection would otherwise spawn 262K tiny tasks
    /// per token and pay closure/steal overhead comparable to the ~48-block dot
    /// itself. Each row's dot is unchanged and rows land at fixed indices, so
    /// the result is bit-identical to the per-row version (greedy parity safe).
    fn matvec_q(&self, out_dim: usize, xq: &[Q8_0Block]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = self.format.row_bytes_for_q8_blocks(xq.len());
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant matvec routes through matvec_q8k")
            }
        };
        let mut out = vec![0f32; out_dim];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Dot a contiguous range of `out_count` output rows starting at
    /// `row_start`, against a pre-quantized activation — used to project a
    /// single MoE expert's matrix out of a 3D `[in_dim, rows, n_expert]` tensor
    /// (expert e occupies rows `e*rows_per_expert ..`). `in_dim` is implied by
    /// `xq.len() * values_per_block`; each row is `xq.len()` blocks wide.
    fn matvec_q_rows(&self, row_start: usize, out_count: usize, xq: &[Q8_0Block]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = self.format.row_bytes_for_q8_blocks(xq.len());
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant rows route through matvec_q8k")
            }
        };
        let mut out = vec![0f32; out_count];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = row_start + chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Repack a contiguous band of `row_count` Q4_0 output rows starting at
    /// `row_start` into the interleaved 8-row [`Q4_0PackedRows8`] layout the AVX2
    /// GEMV consumes. `blocks_per_row` = in_dim/32. `row_start`, `row_count`, and
    /// `blocks_per_row` are all block/8-aligned for the expert bands. Called once
    /// per expert per session (cached), not per token.
    fn pack_rows(
        &self,
        row_start: usize,
        row_count: usize,
        blocks_per_row: usize,
    ) -> crate::tensor::Q4_0PackedRows8 {
        debug_assert_eq!(self.format, WireFormat::Q4_0);
        let row_bytes = blocks_per_row * self.format.bytes_per_block();
        let bytes = self.bytes();
        let start = row_start * row_bytes;
        let band = &bytes[start..start + row_count * row_bytes];
        crate::tensor::Q4_0PackedRows8::from_q4_0_bytes(row_count, blocks_per_row, band)
            .expect("Q4_0 expert band repack (rows multiple of 8, block-aligned)")
    }

    /// Batched [`matvec_q`]: dot each output row against EACH of the `xqs`
    /// activations, reading the weight row from the wire bytes ONCE per row and
    /// reusing it across all `xqs`. For K activations this reads the whole weight
    /// matrix once instead of K times — the speculative-decode bandwidth win, since
    /// verifying K draft tokens then costs a single weight pass. The returned
    /// `out[k]` is bit-identical to `matvec_q(out_dim, xqs[k])` (same row_dot, same
    /// order), so greedy parity is preserved.
    fn matmul_q(&self, out_dim: usize, xqs: &[Vec<Q8_0Block>]) -> Vec<Vec<f32>> {
        const ROW_CHUNK: usize = 64;
        let k = xqs.len();
        if k == 0 {
            return Vec::new();
        }
        let row_bytes = self.format.row_bytes_for_q8_blocks(xqs[0].len());
        let bytes = self.bytes();
        // The batched NVFP4 variant is the same shared-read pattern as its
        // siblings: one weight-row read, `row_dot` looped over the K
        // activations below (correctness-first; no perf claim).
        let row_dot: fn(&[u8], &[Q8_0Block]) -> f32 = match self.format {
            WireFormat::Q8_0 => q8_0_wire_row_dot,
            WireFormat::Q4_0 => q4_0_wire_row_dot,
            WireFormat::Q4_1 => q4_1_wire_row_dot,
            WireFormat::Nvfp4 => nvfp4_wire_row_dot,
            WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
                unreachable!("K-quant matmul routes through matmul_q8k")
            }
        };
        // out[ki][o]; one Vec per activation. Chunk over output rows (the same fixed
        // chunking matvec_q uses) so each weight row is read once and dotted against
        // all k activations. We fill a flat [out_dim * k] buffer in row-chunk order,
        // then transpose into per-activation rows.
        let mut flat = vec![0f32; out_dim * k];
        flat.par_chunks_mut(ROW_CHUNK * k)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                let rows = dst.len() / k;
                for r in 0..rows {
                    let o = base + r;
                    let w = &bytes[o * row_bytes..(o + 1) * row_bytes];
                    for (ki, xq) in xqs.iter().enumerate() {
                        dst[r * k + ki] = row_dot(w, xq);
                    }
                }
            });
        let mut out: Vec<Vec<f32>> = (0..k).map(|_| vec![0f32; out_dim]).collect();
        for o in 0..out_dim {
            for (ki, row) in out.iter_mut().enumerate() {
                row[o] = flat[o * k + ki];
            }
        }
        out
    }

    /// [`matvec`] for Q6_K rows against a Q8_K-quantized activation. Same fixed
    /// row chunking as [`Self::matvec_q`] (greedy-parity-safe ordering).
    fn matvec_q8k(&self, out_dim: usize, xq: &[crate::inference::Q8KBlock]) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matvec_q8k is only for Q6_K/Q4_K weights"),
        };
        let mut out = vec![0f32; out_dim];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// [`Self::matvec_q_rows`] for K-quant (Q4_K/Q6_K) weights against a Q8_K
    /// activation: dot a contiguous band of `out_count` output rows starting at
    /// `row_start` — the MoE expert-band path when the expert matrices are
    /// K-quants. Same fixed row chunking as [`Self::matvec_q8k`], and rows land
    /// at fixed indices, so `out[i]` is bit-identical to row `row_start + i` of
    /// the full [`Self::matvec_q8k`] (greedy parity safe).
    fn matvec_q8k_rows(
        &self,
        row_start: usize,
        out_count: usize,
        xq: &[crate::inference::Q8KBlock],
    ) -> Vec<f32> {
        const ROW_CHUNK: usize = 64;
        let row_bytes = xq.len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matvec_q8k_rows is only for Q6_K/Q4_K weights"),
        };
        let mut out = vec![0f32; out_count];
        out.par_chunks_mut(ROW_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = row_start + chunk_idx * ROW_CHUNK;
                for (i, d) in dst.iter_mut().enumerate() {
                    let o = base + i;
                    *d = row_dot(&bytes[o * row_bytes..(o + 1) * row_bytes], xq);
                }
            });
        out
    }

    /// Batched [`matvec_q8k`]: each Q6_K output row is read once and dotted against
    /// every Q8_K activation in `xqs`. The QAT tied head over K verify positions in a
    /// single weight pass; `out[k]` is bit-identical to `matvec_q8k(out_dim, xqs[k])`.
    fn matmul_q8k(&self, out_dim: usize, xqs: &[Vec<crate::inference::Q8KBlock>]) -> Vec<Vec<f32>> {
        const ROW_CHUNK: usize = 64;
        let k = xqs.len();
        if k == 0 {
            return Vec::new();
        }
        let row_bytes = xqs[0].len() * self.format.bytes_per_block();
        let bytes = self.bytes();
        let row_dot: fn(&[u8], &[crate::inference::Q8KBlock]) -> f32 = match self.format {
            WireFormat::Q6K => q6_k_wire_row_dot,
            WireFormat::Q4K => q4_k_wire_row_dot,
            _ => unreachable!("matmul_q8k is only for Q6_K/Q4_K weights"),
        };
        let mut flat = vec![0f32; out_dim * k];
        flat.par_chunks_mut(ROW_CHUNK * k)
            .enumerate()
            .for_each(|(chunk_idx, dst)| {
                let base = chunk_idx * ROW_CHUNK;
                let rows = dst.len() / k;
                for r in 0..rows {
                    let o = base + r;
                    let w = &bytes[o * row_bytes..(o + 1) * row_bytes];
                    for (ki, xq) in xqs.iter().enumerate() {
                        dst[r * k + ki] = row_dot(w, xq);
                    }
                }
            });
        let mut out: Vec<Vec<f32>> = (0..k).map(|_| vec![0f32; out_dim]).collect();
        for o in 0..out_dim {
            for (ki, row) in out.iter_mut().enumerate() {
                row[o] = flat[o * k + ki];
            }
        }
        out
    }

    /// Dequantize a contiguous element range [start, start+len) — used for
    /// row-major embedding lookups into vocab-major Q8 tables.
    fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        let end = start.checked_add(len).ok_or_else(|| {
            BackendError::InvalidTensorData("wire dequant range overflows usize".into())
        })?;
        if end > self.element_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "wire dequant range {start}..{end} exceeds element count {}",
                self.element_count
            )));
        }
        let bytes = self.bytes();
        let mut out = Vec::with_capacity(len);
        match self.format {
            WireFormat::Q8_0 => {
                const BV: usize = Q8_VALUES_PER_BLOCK;
                const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
                for e in start..end {
                    let block = e / BV;
                    let within = e % BV;
                    let scale = Self::block_scale(bytes, block);
                    let q = bytes[block * BB + 2 + within] as i8;
                    out.push(scale * q as f32);
                }
            }
            WireFormat::Q4_0 => {
                const BB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; 32];
                for e in start..end {
                    if e / 32 != block {
                        block = e / 32;
                        decoded = q4_0_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % 32]);
                }
            }
            WireFormat::Q6K => {
                const BV: usize = crate::inference::Q6_K_VALUES_PER_BLOCK;
                const BB: usize = crate::inference::Q6_K_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        decoded = q6_k_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % BV]);
                }
            }
            // Q4_K tied head + Q5_K per_layer_token_embd are gathered for the input
            // embedding / PLE; decode one 256-value superblock at a time via the shared
            // K-quant decoders (reused, not reimplemented).
            WireFormat::Q4K | WireFormat::Q5K => {
                const BV: usize = 256;
                let bb = self.format.bytes_per_block();
                let mut block = usize::MAX;
                let mut decoded: Vec<f32> = Vec::new();
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        let sb = &bytes[block * bb..(block + 1) * bb];
                        decoded = match self.format {
                            WireFormat::Q4K => {
                                crate::tensor::decode_q4_k_tensor("gemma4 wire gather", sb, BV)?
                            }
                            _ => crate::tensor::decode_q5_k_tensor("gemma4 wire gather", sb, BV)?,
                        };
                    }
                    out.push(decoded[e % BV]);
                }
            }
            // Q4_1 is a matvec-only weight here (ffn_down); no gather decoder is
            // wired. A Q4_1 embedding table would land here, so refuse typed
            // (I-unknown-type: never a reachable panic) — this arm was an
            // `unreachable!` until the SHA_E3 K-quant routing fix swept the
            // lane's reachable-panic arms.
            WireFormat::Q4_1 => {
                return Err(BackendError::UnsupportedTensorType(
                    "gemma4 wire lane cannot gather Q4_1 elements (Q4_1 is a \
                     matvec-only weight format here)"
                        .into(),
                ))
            }
            // NVFP4 gather: decode one 64-value superblock at a time via the
            // pin-bitwise hot-path twin (same pattern as the Q4_0 arm). The
            // BASALT pilot rows keep embeddings Q8_0 (matmul weights only are
            // NVFP4), so this arm only runs on non-pilot shapes; sentinel scale
            // bytes were already refused at load (D17/T5).
            WireFormat::Nvfp4 => {
                const BV: usize = crate::tensor::NVFP4_VALUES_PER_BLOCK;
                const BB: usize = crate::tensor::NVFP4_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for e in start..end {
                    if e / BV != block {
                        block = e / BV;
                        decoded = nvfp4_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out.push(decoded[e % BV]);
                }
            }
        }
        Ok(out)
    }
}

/// A shared per-layer activation with each matvec activation family quantized
/// LAZILY, at most once, however many projections consume it (q/k/v share the
/// pre-attention norm; gate/up share the pre-FFN norm). The Q8_0-family
/// projections (Q8_0/Q4_0/Q4_1/NVFP4) dot Q8_0 blocks; K-quant projections
/// (Q4_K/Q6_K) dot Q8_K blocks — a mixed-format layer quantizes once per
/// family, a single-family layer pays exactly the old single quantize.
/// Single-threaded by construction (a per-step local; rayon parallelism lives
/// INSIDE the matvecs, over output rows), hence the plain `OnceCell`.
struct SharedActivation<'a> {
    x: &'a [f32],
    q8_0: std::cell::OnceCell<Vec<Q8_0Block>>,
    q8_k: std::cell::OnceCell<Vec<crate::inference::Q8KBlock>>,
}

impl<'a> SharedActivation<'a> {
    fn new(x: &'a [f32]) -> Self {
        Self {
            x,
            q8_0: std::cell::OnceCell::new(),
            q8_k: std::cell::OnceCell::new(),
        }
    }

    fn q8_0(&self) -> &[Q8_0Block] {
        self.q8_0.get_or_init(|| quantize_q8_0_blocks(self.x))
    }

    fn q8_k(&self) -> &[crate::inference::Q8KBlock] {
        self.q8_k.get_or_init(|| quantize_q8_k_blocks(self.x))
    }
}

/// The batched (spec-verify [`Gemma4Runtime::step_chunk`]) sibling of
/// [`SharedActivation`]: K activation rows, each quantized family computed
/// lazily once for the whole chunk. Quantization is a pure per-row function,
/// so laziness cannot change any value.
struct SharedActivationBatch<'a> {
    xs: &'a [Vec<f32>],
    q8_0: std::cell::OnceCell<Vec<Vec<Q8_0Block>>>,
    q8_k: std::cell::OnceCell<Vec<Vec<crate::inference::Q8KBlock>>>,
}

impl<'a> SharedActivationBatch<'a> {
    fn new(xs: &'a [Vec<f32>]) -> Self {
        Self {
            xs,
            q8_0: std::cell::OnceCell::new(),
            q8_k: std::cell::OnceCell::new(),
        }
    }

    fn q8_0(&self) -> &[Vec<Q8_0Block>] {
        self.q8_0
            .get_or_init(|| self.xs.iter().map(|x| quantize_q8_0_blocks(x)).collect())
    }

    fn q8_k(&self) -> &[Vec<crate::inference::Q8KBlock>] {
        self.q8_k
            .get_or_init(|| self.xs.iter().map(|x| quantize_q8_k_blocks(x)).collect())
    }
}

/// Greedy-decode stop set: the tokenizer's metadata-declared end ids (EOS/EOT/
/// EOM) plus any end-of-turn marker piece present in the vocab. Gemma 4 renamed
/// the marker from Gemma 3's `<end_of_turn>` to `<turn|>` (id 106; all of
/// E2B/E4B/12B), so a single hardcoded spelling misses the stop and the model
/// emits EOG ids forever. The metadata ids are the authoritative contract;
/// llama.cpp stops on the same set.
fn gemma4_stop_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let sp = &tokenizer.special;
    let mut ids: Vec<u32> = [sp.eos, sp.eot, sp.eom].iter().flatten().copied().collect();
    for marker in ["<turn|>", "<end_of_turn>"] {
        if let Ok(tokens) = tokenizer.encode(marker, false, true) {
            if tokens.len() == 1 {
                ids.push(tokens[0]);
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(crate) fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    (0..out_dim)
        .into_par_iter()
        .map(|o| {
            w[o * in_dim..(o + 1) * in_dim]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

pub(crate) fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = (mss + eps).powf(-0.5);
    match weight {
        Some(w) => x.iter().zip(w).map(|(v, w)| v * inv * w).collect(),
        None => x.iter().map(|v| v * inv).collect(),
    }
}

/// Camelid's gemma4 KV cache is f32. The reference's DEFAULT cache is f16
/// (+ flash attention with an f16-rounded Q path), which flips near-tie argmax
/// positions relative to plain-f32 math — llama.cpp's own `-ctk/-ctv/-fa`
/// settings flip the same positions. Parity oracles are therefore captured with
/// the pinned comparator configuration `-ctk f32 -ctv f32 -fa off --no-repack`
/// (the plain-f32 numeric path this runtime implements); the oracle artifacts
/// record that configuration. `f32_to_f16_bits` (tensor module) remains
/// available for cache-precision experiments.
/// RoPE with optional per-frequency factors (GGUF `rope_freqs.weight`).
///
/// Gemma 4 applies the factor table on FULL-attention layers only ("proportional
/// rope", mirroring llama.cpp's `gemma4-iswa`: `freq_factors` is the layer's
/// `rope_freqs` when `!is_swa`, null otherwise). The shipped table is 1.0 for
/// pair indices 0..64 and 1e30 beyond — dividing the frequency by 1e30 zeroes
/// the rotation, so only the first 64 frequency pairs of a global head carry
/// position. Skipping the factors is numerically close on short prompts but is
/// NOT the reference math (it measurably shifts near-tie logits).
pub(crate) fn apply_rope(
    vec: &mut [f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
    factors: Option<&[f32]>,
) {
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let mut freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            if let Some(factors) = factors {
                freq /= factors[i];
            }
            let (s, c) = (position as f32 * freq).sin_cos();
            let (a, b) = (vec[base + i], vec[base + half + i]);
            vec[base + i] = a * c - b * s;
            vec[base + half + i] = b * c + a * s;
        }
    }
}

struct LayerWeights {
    attn_norm: Vec<f32>,
    attn_q: WireQuant,
    /// `None` on shared-KV layers in trimmed (QAT) exports — never read there.
    attn_k: Option<WireQuant>,
    attn_v: Option<WireQuant>, // None on V-less layers (V = K projection)
    attn_output: WireQuant,
    q_norm: Vec<f32>,
    k_norm: Option<Vec<f32>>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: WireQuant,
    ffn_up: WireQuant,
    ffn_down: WireQuant,
    post_ffw_norm: Vec<f32>,
    // PLE (E-series); inp_gate/proj are small F32 matrices in the GGUF.
    post_norm: Option<Vec<f32>>,
    ple_inp_gate: Option<Vec<f32>>,
    ple_proj: Option<Vec<f32>>,
    ple_output_scale: f32,
    /// Gemma 4 A4B (26B) sparse-expert branch; `None` on dense rows. When
    /// present, the FFN runs the two-branch MoE block (see `MoeWeights`).
    moe: Option<MoeWeights>,
}

/// One expert's two projection matrices, pre-repacked into the interleaved
/// 8-row layout [`crate::tensor::Q4_0PackedRows8`] the AVX2 GEMV consumes. Built
/// lazily on first use and cached (see [`ExpertPackCache`]) so the repack is paid
/// once per expert per session instead of once per token — the packed GEMV then
/// runs with no per-call repack/alloc, which is what makes it beat the (already
/// autovectorized) scalar wire dot.
struct PackedExpert {
    /// Fused gate‖up, `2*n_ff_exp` rows × (n_embd/32) blocks/row.
    gate_up: crate::tensor::Q4_0PackedRows8,
    /// Down, `n_embd` rows × (n_ff_exp/32) blocks/row.
    down: crate::tensor::Q4_0PackedRows8,
}

impl PackedExpert {
    fn byte_len(&self) -> usize {
        self.gate_up.byte_len() + self.down.byte_len()
    }
}

/// Bounded host-RAM cache of [`PackedExpert`]s for ONE MoE layer, keyed by expert
/// index. A greedy decode fires a small, stable subset of the 128 experts, so a
/// modest cap keeps the hot experts pre-packed (steady-state SIMD GEMV with no
/// repack) while bounding the extra RAM — the packed form is a second copy of the
/// expert weights (~11% larger than the mmap wire bytes), so caching ALL experts
/// of ALL layers would blow this box's RAM. Eviction is FIFO on the insertion
/// order (the working set is stable, so FIFO ≈ LRU here). Budget in MiB via
/// `CAMELID_GEMMA4_EXPERT_PACK_MIB` (default 1024; 0 disables the SIMD pack path,
/// falling back to the scalar wire dot). Correctness is independent of the cache:
/// a miss that cannot be cached just repacks on the fly, still bit-exact.
struct ExpertPackCache {
    entries: std::collections::HashMap<u16, Arc<PackedExpert>>,
    order: std::collections::VecDeque<u16>,
    bytes: usize,
    budget_bytes: usize,
}

impl ExpertPackCache {
    fn new(budget_bytes: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            bytes: 0,
            budget_bytes,
        }
    }

    fn get(&self, e: u16) -> Option<Arc<PackedExpert>> {
        self.entries.get(&e).cloned()
    }

    /// Insert `packed` for expert `e`, evicting FIFO until it fits the budget.
    /// If a single expert exceeds the budget it is not cached (returned Arc is
    /// still usable by the caller for this one token).
    fn insert(&mut self, e: u16, packed: Arc<PackedExpert>) {
        let sz = packed.byte_len();
        if sz > self.budget_bytes {
            return;
        }
        while self.bytes + sz > self.budget_bytes {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(p) = self.entries.remove(&old) {
                self.bytes -= p.byte_len();
            }
        }
        if self.entries.insert(e, packed).is_none() {
            self.order.push_back(e);
            self.bytes += sz;
        }
    }
}

/// Per-layer expert-pack budget (bytes) from `CAMELID_GEMMA4_EXPERT_PACK_MIB`
/// (default 1024 MiB). `0` disables pre-packing (scalar wire-dot fallback).
fn expert_pack_budget_bytes() -> usize {
    std::env::var("CAMELID_GEMMA4_EXPERT_PACK_MIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1024)
        .saturating_mul(1024 * 1024)
}

/// GEMV a whole pre-packed [`crate::tensor::Q4_0PackedRows8`] band against a Q8
/// activation, returning one f32 per row. One rayon task per group of 8 rows runs
/// the AVX2 [`crate::inference::q4_0_packed_gemv8`] into eight fixed output slots
/// — no repack, no per-call allocation. Bit-identical to `matvec_q_rows` on the
/// same rows (the kernel is proven bit-exact vs the scalar wire dot and the row
/// order is preserved).
fn packed_band_matvec(packed: &crate::tensor::Q4_0PackedRows8, xq: &[Q8_0Block]) -> Vec<f32> {
    debug_assert_eq!(packed.blocks_per_row, xq.len());
    let mut out = vec![0f32; packed.rows];
    out.par_chunks_mut(8).enumerate().for_each(|(g, dst)| {
        let group_block_start = g * packed.blocks_per_row;
        let mut acc = [0f32; 8];
        crate::inference::q4_0_packed_gemv8(packed, group_block_start, xq, &mut acc);
        dst.copy_from_slice(&acc);
    });
    out
}

/// Sparse 128-expert branch weights for one Gemma 4 A4B MoE layer. The dense
/// `ffn_gate/up/down` on [`LayerWeights`] are the parallel shared-expert MLP.
struct MoeWeights {
    /// Router matrix [n_embd, n_expert], F32, row-major (out=expert).
    gate_inp: Vec<f32>,
    /// Router input scale [n_embd], F32, elementwise.
    gate_inp_scale: Vec<f32>,
    /// Fused per-expert gate‖up, Q4_0 wire; row `e*2*n_ff_exp + o` is expert e
    /// output o (gate for o<n_ff_exp, up for o>=n_ff_exp), in_dim = n_embd.
    gate_up_exps: WireQuant,
    /// Per-expert down, Q4_0 wire; row `e*n_embd + o` is expert e output o,
    /// in_dim = n_ff_exp.
    down_exps: WireQuant,
    /// Per-expert down scale [n_expert], F32, scalar per expert.
    down_exps_scale: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_1: Vec<f32>,
    post_norm_2: Vec<f32>,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    /// Lazy per-expert pre-packed (interleaved 8-row) form of the two Q4_0 expert
    /// matrices, for the AVX2 GEMV expert path. Populated on first use, bounded by
    /// [`expert_pack_budget_bytes`]. `None` when the experts are not Q4_0 (the
    /// pack path only supports Q4_0) or the budget is 0.
    pack_cache: Option<std::sync::Mutex<ExpertPackCache>>,
}

impl MoeWeights {
    /// Return expert `e`'s pre-packed (interleaved 8-row) projections, packing
    /// and caching them on first use. `None` when the pack path is disabled
    /// (non-Q4_0 experts or a 0 budget) — the caller then uses the scalar wire
    /// dot. `hidden` = n_embd (gate_up in_dim), `two_nff` = 2*n_ff_exp (gate_up
    /// row count / down in_dim). Packing happens under the cache lock but the
    /// returned `Arc` is cloned out, so the GEMV runs lock-free.
    fn packed_expert(&self, e: usize, hidden: usize, two_nff: usize) -> Option<Arc<PackedExpert>> {
        let cache = self.pack_cache.as_ref()?;
        let key = e as u16;
        {
            let guard = cache.lock().expect("expert pack cache poisoned");
            if let Some(p) = guard.get(key) {
                return Some(p);
            }
        }
        // Miss: pack this expert's two bands (outside the lock is not required —
        // the pack is the same work regardless — but we build then insert under
        // the lock so concurrent callers converge; decode is single-threaded here
        // so there is no real contention).
        let gu_blocks = hidden / 32; // gate_up in_dim = n_embd
        let down_blocks = two_nff / 2 / 32; // down in_dim = n_ff_exp
        let gate_up = self.gate_up_exps.pack_rows(e * two_nff, two_nff, gu_blocks);
        let down = self.down_exps.pack_rows(e * hidden, hidden, down_blocks);
        let packed = Arc::new(PackedExpert { gate_up, down });
        let mut guard = cache.lock().expect("expert pack cache poisoned");
        guard.insert(key, packed.clone());
        Some(packed)
    }
}

/// Per-phase CPU decode counters (µs), populated only when
/// `CAMELID_GEMMA4_CPU_TIMING=1`. Printed by `generate_greedy` as an average per
/// step: embedding+PLE prep, attention (proj/rope/scores/output), FFN(+PLE
/// injection), and the 262K-vocab output projection. Diagnostics only — no
/// effect on generated tokens.
static CPU_EMBED_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_ATTN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_FFN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_OUTPROJ_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CPU_STEP_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn cpu_timing_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_CPU_TIMING").is_ok_and(|v| v == "1")
}

fn report_cpu_timing() {
    use std::sync::atomic::Ordering::Relaxed;
    let n = CPU_STEP_N.load(Relaxed).max(1);
    eprintln!(
        "[gemma4-cpu-timing] {n} steps: embed+pli {}us, attention {}us, ffn+ple {}us, output-proj {}us (avg/step)",
        CPU_EMBED_US.load(Relaxed) / n,
        CPU_ATTN_US.load(Relaxed) / n,
        CPU_FFN_US.load(Relaxed) / n,
        CPU_OUTPROJ_US.load(Relaxed) / n,
    );
}

/// A loaded Gemma 4 model ready to generate.
///
/// Supports loading a contiguous **layer range** for distributed layer sharding:
/// a shard holds weights only for `[first_layer, first_layer + layers.len())`,
/// computes its own PLE inputs from the token id (PLE depends only on the token,
/// never on upstream activations), and exchanges the hidden state at the cut
/// point. The full single-node runtime is the `0..block_count` special case.
pub struct Gemma4Runtime {
    config: LlamaModelConfig,
    g: Gemma4Metadata,
    tokenizer: Tokenizer,
    /// Global index of the first locally-loaded layer (0 on a full runtime).
    first_layer: usize,
    layers: Vec<LayerWeights>,
    token_embd: WireQuant,
    per_layer_token_embd: Option<WireQuant>,
    per_layer_model_proj: Option<Vec<f32>>, // BF16 -> f32
    per_layer_proj_norm: Option<Vec<f32>>,
    output_norm: Vec<f32>,
    /// GGUF `rope_freqs.weight` — per-frequency factors applied on FULL
    /// attention layers only (None when absent).
    rope_factors: Option<Vec<f32>>,
    first_kv_shared: usize,
    last_sliding_layer: usize,
    last_full_layer: usize,
}

/// One shard step's result: interior shards hand the hidden state to the next
/// shard; the tail shard (owning the final layer) produces logits.
pub enum Gemma4StepOutput {
    Hidden(Vec<f32>),
    Logits(Vec<f32>),
}

/// Per-layer incremental KV cache: `cache[local_layer][position]` is one
/// position's packed `[kv_heads * head_dim]` K (or V) row.
pub type Gemma4KvCache = Vec<Vec<Vec<f32>>>;

impl Gemma4Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_layer_range(path, None)
    }
}

/// BASALT D-B2 fail-closed (DECISIONS.md D17): a ModelOpt-converted NVFP4 GGUF
/// carries per-tensor sidecar scales as separate `.scale` / `.input_scale`
/// tensors that MUST be multiplied post-matmul. The gemma4 wire lane does not
/// implement that multiply, and silently ignoring the sidecars would compute
/// wrong logits — so an NVFP4 file that carries any refuses at load, mirroring
/// the runnable-lane admission check. Pin-quantized rows (the BASALT pilot
/// artifacts, receipted at G2) carry none; the pilot's real
/// `blk.N.layer_output_scale.weight` tensors do NOT match these suffixes.
pub(crate) fn nvfp4_sidecar_check(tensors: &[crate::gguf::GgufTensorDescriptor]) -> Result<()> {
    if tensors
        .iter()
        .any(|t| t.tensor_type == GgufTensorType::NVFP4)
    {
        if let Some(sidecar) = tensors
            .iter()
            .find(|t| t.name.ends_with(".scale") || t.name.ends_with(".input_scale"))
        {
            return Err(BackendError::UnsupportedGguf(format!(
                "NVFP4 GGUF carries per-tensor sidecar scale tensor {}; the gemma4 \
                 wire lane does not apply sidecar scales and refuses rather than \
                 compute wrong logits (BASALT D-B2)",
                sidecar.name
            )));
        }
    }
    Ok(())
}

/// BASALT Amendment 3 §9 platform gate (DECISIONS.md D17 micro-decisions),
/// GABBRO M2 narrowing: NVFP4 admits on Windows AND macOS in this release, and
/// refuses on every other target (Linux et al.). macOS joined the admit set once
/// its CPU wire-lane decode was proven bit-exact on Apple Silicon (GABBRO Gate
/// G-M1, `qa/evidence-bundles/gabbro/phase1/`). This is a RUNTIME check (`cfg!`
/// inside ordinary code), deliberately NOT a `#[cfg]` wall: the decode code
/// compiles on every target, and refused callers get this named refusal instead
/// of a missing symbol. Enforced in BOTH lanes — runnable admission
/// (`runnable::admit`) and this gemma4 wire-lane load path — because either lane
/// alone could otherwise reach NVFP4 weights on an unvalidated platform. Fires
/// AFTER [`nvfp4_sidecar_check`] so the D-B2 posture stays platform-independent.
///
/// NOTE (GABBRO M2): the refusal message reads "Windows/macOS-only" and the
/// support matrices are truthed-up to Windows+macOS in this same ratchet PR
/// (Tim folded the surface truth-up into M2). macOS runs NVFP4 on both the CPU wire
/// lane and the Metal resident GPU lane (GABBRO M3 + followup), the Metal lane
/// guarded by `gemma4_metal_layer_fmt` (covered set) + `nvfp4_metal_sentinel_check`
/// (D17/T5). The fn name `nvfp4_windows_only_check` is retained as an optional
/// internal rename follow-up (pub(crate); not a user surface).
pub(crate) fn nvfp4_windows_only_check(
    tensors: &[crate::gguf::GgufTensorDescriptor],
) -> Result<()> {
    if !cfg!(target_os = "windows")
        && !cfg!(target_os = "macos")
        && tensors
            .iter()
            .any(|t| t.tensor_type == GgufTensorType::NVFP4)
    {
        return Err(BackendError::UnsupportedGguf(
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX".into(),
        ));
    }
    Ok(())
}

/// GABBRO M3-followup (D17/T5 fail-closed): the macOS GPU lane
/// ([`Gemma4GpuRuntime::load`]) now RUNS NVFP4 layer projections (kernel
/// `nvfp4_block_linear_row_ksplit_f32y_wire`), reading their wire bytes RAW via
/// WirePages — which bypasses `WireQuant::new`'s NaN-sentinel scan. So the T5 guard
/// lives here: scan every NVFP4 tensor's UE4M3 scale bytes and refuse `0x7F`/`0xFF`
/// (the pin's CPU and CUDA backends disagree on `0xFF`, so such a file has no
/// well-defined cross-backend oracle), matching the CPU wire lane. Clean NVFP4 — and
/// files without NVFP4 — admit. The shared [`crate::tensor::nvfp4_find_nan_scale`]
/// does the byte scan; called once the mmap is available.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn nvfp4_metal_sentinel_check(
    tensors: &[crate::gguf::GgufTensorDescriptor],
    mmap: &GgufWireMmap,
) -> Result<()> {
    for t in tensors {
        if t.tensor_type == GgufTensorType::NVFP4 {
            let wire = mmap.bytes(t.absolute_offset, t.n_bytes as usize)?;
            if let Some(block_idx) = crate::tensor::nvfp4_find_nan_scale(wire) {
                return Err(BackendError::InvalidTensorData(format!(
                    "tensor {}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 \
                     scale byte (0x7F/0xFF) — refusing on the Metal resident lane per D17/T5",
                    t.name
                )));
            }
        }
    }
    Ok(())
}

/// GABBRO M3-followup: the Metal resident lane's covered layer-projection formats —
/// Q8_0 / Q4_0 / NVFP4, each a parity-gated GPU GEMV. Any other format refuses TYPED
/// and NAMED (invariant I-unknown-type, L4 cell) rather than mis-binding. Extracted so
/// it unit-tests without a real model; the load site probes layer-0 `attn_q` (the
/// export quantizes every layer's projections alike).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn gemma4_metal_layer_fmt(tensor_type: GgufTensorType) -> Result<crate::metal::GemmaWireFmt> {
    // The only non-test caller is `Gemma4GpuRuntime::load`, which is `cfg(macos)`-gated, so
    // the non-macOS lib build sees this as dead; allow it there (the `#[cfg(test)]` covered-set
    // test still exercises it on every platform). Mirrors `nvfp4_metal_sentinel_check`.
    match tensor_type {
        GgufTensorType::Q8_0 => Ok(crate::metal::GemmaWireFmt::Q8_0),
        GgufTensorType::Q4_0 => Ok(crate::metal::GemmaWireFmt::Q4_0),
        GgufTensorType::NVFP4 => Ok(crate::metal::GemmaWireFmt::Nvfp4),
        other => Err(BackendError::UnsupportedTensorType(format!(
            "gemma4 GPU runtime supports Q8_0/Q4_0/NVFP4 layer projections; \
             layer 0 attn_q is {other:?}"
        ))),
    }
}

/// BASALT Amendment 3 review fix (CUDA lane typed refusal), extended at SHA_E
/// review and lifted at Phase 4: the CUDA-resident gemma4 lane repacks layer
/// projections via `GemmaLayerQuant::from_wire`, whose catch-all PANICS on any
/// format outside its covered set. Phase 4 (BASALT G4) added the NVFP4 raw-wire
/// GEMV (`nvfp4_gemv`), so the covered set is now Q8_0/Q4_0/Q4_1/NVFP4. Every
/// remaining lane-uncovered format — the K-quants the CPU wire lane serves (the
/// campaign's own Q4K-mm / Q4_K_M rows) — must still refuse with a typed, named
/// error before that panic seam is reachable (invariant I-unknown-type, L3 cell).
/// cfg-independent over [`WireFormat`]s so it unit-tests without CUDA hardware;
/// the `cfg(feature = "cuda")` load site ([`Gemma4CudaResident::load`]) wires it.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn nvfp4_cuda_lane_check<I: IntoIterator<Item = WireFormat>>(formats: I) -> Result<()> {
    for f in formats {
        match f {
            WireFormat::Q8_0 | WireFormat::Q4_0 | WireFormat::Q4_1 | WireFormat::Nvfp4 => {}
            other => {
                return Err(BackendError::UnsupportedGguf(format!(
                    "gemma4 CUDA-resident lane covers Q8_0/Q4_0/Q4_1/NVFP4 layer projections; \
                     {other:?} is not wired (the CPU lane serves it) — refusing instead \
                     of reaching the repack panic (BASALT I-unknown-type, L3)"
                )));
            }
        }
    }
    Ok(())
}

/// BASALT Amendment 3 review fix (step-boundary proof): the forced-decode loop's
/// boundary bookkeeping, extracted so a unit test can prove the off-by-one
/// contract with a scripted step closure and no model. Drives `step` (feed one
/// token, return the next prediction state) over the forced list starting from
/// the prompt-end prediction, guaranteeing:
///
/// 1. `observe(i, state)` sees the prediction computed BEFORE `forced[i]` is fed
///    as the next input (the teacher-forcing boundary);
/// 2. exactly `forced.len()` observations fire;
/// 3. the FINAL forced token is never fed (its prediction is already observed;
///    feeding it would only compute an unrecorded extra step).
///
/// [`Gemma4Runtime::forced_decode`] rewires through this; the real forward step
/// is untouched.
pub(crate) fn drive_forced_steps<P, E>(
    forced: &[u32],
    prompt_end_prediction: P,
    mut step: impl FnMut(u32) -> std::result::Result<P, E>,
    mut observe: impl FnMut(usize, &P),
) -> std::result::Result<(), E> {
    let mut prediction = prompt_end_prediction;
    for (i, &tok) in forced.iter().enumerate() {
        observe(i, &prediction);
        if i + 1 < forced.len() {
            prediction = step(tok)?;
        }
    }
    Ok(())
}

impl Gemma4Runtime {
    /// Load only the given contiguous global layer range (None = all layers).
    /// Fails closed if the range would separate a KV-sharing layer from the
    /// cache it reads (the split must keep every shared layer on the same shard
    /// as its source layer).
    pub fn load_layer_range(path: &Path, range: Option<std::ops::Range<usize>>) -> Result<Self> {
        let gguf = read_metadata(path)?;
        // BASALT D-B2 fail-closed (DECISIONS.md D17): a ModelOpt-converted NVFP4
        // GGUF carries per-tensor sidecar scales as separate `.scale` /
        // `.input_scale` tensors that MUST be multiplied post-matmul. This wire
        // lane does not implement that multiply, and silently ignoring the
        // sidecars would compute wrong logits — so an NVFP4 file that carries
        // any refuses here, mirroring the runnable-lane admission check.
        // Pin-quantized rows (the BASALT pilot artifacts) carry none.
        nvfp4_sidecar_check(&gguf.tensors)?;
        // BASALT Amendment 3 §9 + GABBRO M2: NVFP4 admits on Windows and macOS
        // in this release (other targets refuse) — a runtime platform gate
        // (after the sidecar check so D-B2 stays platform-independent), mirrored
        // in runnable admission.
        nvfp4_windows_only_check(&gguf.tensors)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let block_count = config.block_count as usize;
        let range = range.unwrap_or(0..block_count);
        if range.start >= range.end || range.end > block_count {
            return Err(BackendError::InvalidModelMetadata(format!(
                "gemma4 layer range {range:?} is invalid for {block_count} layers"
            )));
        }
        // Cross-layer KV sharing constraint: every local layer must read a cache
        // owned by a layer in the same range.
        let plan = g.layer_plan(block_count, config.attention_head_count as usize);
        for l in range.clone() {
            let src = plan[l].kv_source_layer;
            if !range.contains(&src) {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 layer range {range:?} separates layer {l} from its shared \
                     KV source layer {src}; choose a split that keeps the trailing \
                     shared-KV block together (first shared source is layer {})",
                    block_count - g.num_kv_shared_layers as usize
                )));
            }
        }

        // Memory-map the GGUF once. Q8 weights are referenced in place (no eager
        // decode); kick off background readahead so the first generation does not
        // pay the whole cold-fault cost serially. The advisory MUST run off the
        // loading thread: on macOS madvise(MADV_WILLNEED) over a USB-backed
        // volume blocks until the kernel has paged in the advised range —
        // observed live as a 12.7 GB 12B mapping stalling a serve-lane model
        // load for 10+ minutes while loading a half-model shard.
        let mmap = GgufWireMmap::map(path)?;
        {
            let mmap = mmap.clone();
            std::thread::spawn(move || mmap.advise_willneed());
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        // Matvec-role loads (projections, expert bands, the tied head) refuse
        // Q5_K typed at load — it is gather-only in this lane and would
        // otherwise panic at forward time (I-unknown-type, SHA_E3).
        let q8m = |name: &str| -> Result<WireQuant> { q8(name)?.require_matvec_capable(name) };
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(range.len());
        for l in &binding.layers[range.clone()] {
            layers.push(LayerWeights {
                attn_norm: f32t(&l.attn_norm.name)?,
                attn_q: q8m(&l.attn_q.name)?,
                attn_k: l.attn_k.as_ref().map(|d| q8m(&d.name)).transpose()?,
                attn_v: l.attn_v.as_ref().map(|d| q8m(&d.name)).transpose()?,
                attn_output: q8m(&l.attn_output.name)?,
                q_norm: f32t(&l.attn_q_norm.name)?,
                k_norm: l.attn_k_norm.as_ref().map(|d| f32t(&d.name)).transpose()?,
                post_attn_norm: f32t(&l.post_attention_norm.name)?,
                ffn_norm: f32t(&l.ffn_norm.name)?,
                ffn_gate: q8m(&l.ffn_gate.name)?,
                ffn_up: q8m(&l.ffn_up.name)?,
                ffn_down: q8m(&l.ffn_down.name)?,
                post_ffw_norm: f32t(&l.post_ffw_norm.name)?,
                post_norm: l.post_norm.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_inp_gate: l.ple_inp_gate.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_proj: l.ple_proj.as_ref().map(|d| f32t(&d.name)).transpose()?,
                ple_output_scale: l
                    .ple_output_scale
                    .as_ref()
                    .map(|d| f32t(&d.name))
                    .transpose()?
                    .and_then(|v| v.first().copied())
                    .unwrap_or(1.0),
                moe: l
                    .moe
                    .as_ref()
                    .map(|m| -> Result<MoeWeights> {
                        let moe_meta = config.moe.as_ref().ok_or_else(|| {
                            BackendError::InvalidModelMetadata(
                                "gemma4 MoE layer present but no expert metadata".into(),
                            )
                        })?;
                        let n_expert = moe_meta.expert_count as usize;
                        // 2*n_ff_exp = gate_up rows / n_expert; n_ff_exp halves it.
                        let gate_up = q8m(&m.gate_up_exps.name)?;
                        let down = q8m(&m.down_exps.name)?;
                        let two_nff =
                            gate_up.element_count / (n_expert * config.embedding_length as usize);
                        // Enable the AVX2 pre-pack expert path only when BOTH expert
                        // matrices are Q4_0 (the pack format) and a budget is set.
                        let budget = expert_pack_budget_bytes();
                        let pack_cache = if budget > 0
                            && gate_up.format == WireFormat::Q4_0
                            && down.format == WireFormat::Q4_0
                        {
                            Some(std::sync::Mutex::new(ExpertPackCache::new(budget)))
                        } else {
                            None
                        };
                        Ok(MoeWeights {
                            gate_inp: f32t(&m.gate_inp.name)?,
                            gate_inp_scale: f32t(&m.gate_inp_scale.name)?,
                            gate_up_exps: gate_up,
                            down_exps: down,
                            down_exps_scale: f32t(&m.down_exps_scale.name)?,
                            pre_norm_2: f32t(&m.pre_norm_2.name)?,
                            post_norm_1: f32t(&m.post_norm_1.name)?,
                            post_norm_2: f32t(&m.post_norm_2.name)?,
                            n_expert,
                            n_expert_used: moe_meta.expert_used_count as usize,
                            n_ff_exp: two_nff / 2,
                            pack_cache,
                        })
                    })
                    .transpose()?,
            });
        }

        let first_kv_shared = config.block_count as usize - g.num_kv_shared_layers as usize;
        Ok(Self {
            tokenizer,
            first_layer: range.start,
            // The tied head matvecs token_embd on the tail shard, so it takes
            // the matvec-role guard; per_layer_token_embd stays gather-only
            // (plain q8) — Q5_K is legitimate there.
            token_embd: q8m(&binding.token_embedding.name)?,
            per_layer_token_embd: binding
                .per_layer_token_embd
                .as_ref()
                .map(|d| q8(&d.name))
                .transpose()?,
            per_layer_model_proj: binding
                .per_layer_model_proj
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            per_layer_proj_norm: binding
                .per_layer_proj_norm
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            output_norm: f32t(&binding.output_norm.name)?,
            rope_factors: binding
                .rope_freqs
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            first_kv_shared,
            last_sliding_layer: (0..first_kv_shared)
                .rev()
                .find(|&l| g.is_sliding_layer(l))
                .unwrap_or(0),
            last_full_layer: (0..first_kv_shared)
                .rev()
                .find(|&l| !g.is_sliding_layer(l))
                .unwrap_or(0),
            layers,
            config,
            g,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Global layer range loaded on this shard.
    pub fn local_layer_range(&self) -> std::ops::Range<usize> {
        self.first_layer..self.first_layer + self.layers.len()
    }

    pub fn block_count(&self) -> usize {
        self.config.block_count as usize
    }

    pub fn hidden_size(&self) -> usize {
        self.config.embedding_length as usize
    }

    /// Logit-vector length of this model's tied head (`token_embd` rows) — the
    /// exact length `step` returns, and therefore the bound the BASALT harness
    /// uses to validate teacher-forced token ids before decoding.
    pub fn vocab_size(&self) -> usize {
        self.token_embd.element_count / self.hidden_size()
    }

    /// Greedy stop set for this model (metadata EOS/EOT/EOM + literal
    /// `<end_of_turn>` when present).
    pub fn stop_token_ids(&self) -> Vec<u32> {
        gemma4_stop_token_ids(&self.tokenizer)
    }

    /// Fresh per-LOCAL-layer KV caches for one sequence.
    pub fn empty_kv_caches(&self) -> (Gemma4KvCache, Gemma4KvCache) {
        (
            vec![Vec::new(); self.layers.len()],
            vec![Vec::new(); self.layers.len()],
        )
    }

    /// Process one token at absolute `pos`, appending its K/V to the per-layer
    /// caches (`kc`/`vc`; only non-shared layers store entries — shared layers read
    /// the last same-type layer's cache, already updated this step). Returns the
    /// next-token logits.
    fn step(
        &self,
        token: u32,
        pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<f32>> {
        match self.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Logits(logits) => Ok(logits),
            Gemma4StepOutput::Hidden(_) => Err(BackendError::InvalidModelMetadata(
                "step() requires a runtime that owns the final layer; use step_range \
                 on interior shards"
                    .into(),
            )),
        }
    }

    /// True when the batched [`Self::step_chunk`] forward is usable: single-node
    /// (this runtime owns every layer including the head) and no MoE layer. The
    /// speculative-decode lane needs the head shard; MoE rows are distributed-only.
    fn supports_chunk_forward(&self) -> bool {
        self.first_layer == 0
            && self.first_layer + self.layers.len() == self.config.block_count as usize
            && self.layers.iter().all(|lw| lw.moe.is_none())
    }

    /// Batched forward over `tokens` at consecutive positions `start_pos +
    /// 0..tokens.len()`, appending all K K/V rows to the caches and returning the
    /// next-token logits at EACH position. Numerically identical to calling
    /// [`Self::step`] once per token (same dots, same order) — the only difference is
    /// that each weight matrix is read ONCE for the whole chunk via [`matmul_q`]
    /// instead of once per token, which is the speculative-decode verify win.
    /// Requires [`Self::supports_chunk_forward`]; caller guarantees it.
    #[allow(clippy::needless_range_loop)]
    fn step_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<f32>>> {
        let kk = tokens.len();
        debug_assert!(kk > 0);
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_local = self.layers.len();
        let block_count = self.config.block_count as usize;
        let ple_total = block_count * ple_dim;
        let win = self.g.sliding_window as usize;

        // Per-token scaled embedding (== step_range's h0) and the PLE per-layer input.
        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(kk);
        // pli_tok[i][li] is layer li's per-layer input for token i.
        let mut pli_tok: Vec<Vec<Vec<f32>>> = Vec::with_capacity(kk);
        for &token in tokens {
            let h0: Vec<f32> = self
                .token_embd
                .dequantize_elements(token as usize * hidden, hidden)?
                .iter()
                .map(|v| v * (hidden as f32).sqrt())
                .collect();
            let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
                self.per_layer_token_embd.as_ref(),
                self.per_layer_model_proj.as_ref(),
                self.per_layer_proj_norm.as_ref(),
            ) {
                let local_span = n_local * ple_dim;
                let ti = te.dequantize_elements(token as usize * ple_total, local_span)?;
                let proj_local = &proj[0..local_span * hidden];
                let ctx = f32_matvec(proj_local, hidden, local_span, &h0);
                let proj_scale = (hidden as f32).powf(-0.5);
                let ple_embed_scale = (ple_dim as f32).sqrt();
                (0..n_local)
                    .map(|li| {
                        let ctx_l: Vec<f32> = (0..ple_dim)
                            .map(|d| ctx[li * ple_dim + d] * proj_scale)
                            .collect();
                        let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                        (0..ple_dim)
                            .map(|d| {
                                (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                    * std::f32::consts::FRAC_1_SQRT_2
                            })
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            hs.push(h0);
            pli_tok.push(pli);
        }

        for li in 0..n_local {
            let l = li; // single-node: global == local
            let lw = &self.layers[li];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let ffn_dim = self.g.ffn_length_at(l) as usize;
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            // --- attention projections, batched (one weight pass each) ---
            let xn_rows: Vec<Vec<f32>> = hs
                .iter()
                .map(|h| rms_norm(h, Some(&lw.attn_norm), eps))
                .collect();
            let xnq = SharedActivationBatch::new(&xn_rows);
            let mut q_rows = lw.attn_q.matmul_proj(q_dim, &xnq);
            for q in q_rows.iter_mut() {
                for hh in 0..heads {
                    let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
                }
            }
            for (i, q) in q_rows.iter_mut().enumerate() {
                apply_rope(q, heads, head_dim, start_pos + i, theta, rope_factors);
            }

            if l < self.first_kv_shared {
                let mut k_rows = lw
                    .attn_k
                    .as_ref()
                    .expect("validate() guarantees owning layers bind attn_k")
                    .matmul_proj(kv_dim, &xnq);
                let mut v_rows = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matmul_proj(kv_dim, &xnq),
                    None => k_rows.clone(),
                };
                for i in 0..kk {
                    for hh in 0..kv_heads {
                        let s = &mut k_rows[i][hh * head_dim..(hh + 1) * head_dim];
                        s.copy_from_slice(&rms_norm(
                            s,
                            Some(
                                lw.k_norm
                                    .as_deref()
                                    .expect("validate() guarantees owning layers bind attn_k_norm"),
                            ),
                            eps,
                        ));
                        let sv = &mut v_rows[i][hh * head_dim..(hh + 1) * head_dim];
                        sv.copy_from_slice(&rms_norm(sv, None, eps));
                    }
                    apply_rope(
                        &mut k_rows[i],
                        kv_heads,
                        head_dim,
                        start_pos + i,
                        theta,
                        rope_factors,
                    );
                }
                // Append all K rows in position order; query i (below) then reads the
                // cache only up to its own position, so causality holds.
                for i in 0..kk {
                    kc[li].push(std::mem::take(&mut k_rows[i]));
                    vc[li].push(std::mem::take(&mut v_rows[i]));
                }
            }

            let src_global = if l < self.first_kv_shared {
                l
            } else if sliding {
                self.last_sliding_layer
            } else {
                self.last_full_layer
            };
            let src = src_global - self.first_layer;
            let group = heads / self.g.kv_heads_at(src_global) as usize;

            // --- per-position attention (cheap; no big weight read) ---
            let mut attn_rows: Vec<Vec<f32>> = Vec::with_capacity(kk);
            for i in 0..kk {
                let pos = start_pos + i;
                let lo = if sliding {
                    (pos + 1).saturating_sub(win)
                } else {
                    0
                };
                let q = &q_rows[i];
                let mut attn = vec![0f32; q_dim];
                for hh in 0..heads {
                    let kvh = hh / group;
                    let qh = &q[hh * head_dim..(hh + 1) * head_dim];
                    let mut scores: Vec<f32> = (lo..=pos)
                        .map(|p| {
                            let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                            qh.iter().zip(kp).map(|(a, b)| a * b).sum()
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f32;
                    for s in &mut scores {
                        *s = (*s - m).exp();
                        den += *s;
                    }
                    let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                    for (idx, p) in (lo..=pos).enumerate() {
                        let w = scores[idx] / den;
                        let vp = &vc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                        for d in 0..head_dim {
                            out[d] += w * vp[d];
                        }
                    }
                }
                attn_rows.push(attn);
            }
            // o-projection batched, then residual + post-attn norm per token.
            let attn_b = SharedActivationBatch::new(&attn_rows);
            let o_rows = lw.attn_output.matmul_proj(hidden, &attn_b);
            for i in 0..kk {
                let on = rms_norm(&o_rows[i], Some(&lw.post_attn_norm), eps);
                for (a, b) in hs[i].iter_mut().zip(&on) {
                    *a += b;
                }
            }

            // --- FFN (dense), batched ---
            let ffn_rows: Vec<Vec<f32>> = hs
                .iter()
                .map(|h| rms_norm(h, Some(&lw.ffn_norm), eps))
                .collect();
            let ffnq = SharedActivationBatch::new(&ffn_rows);
            let gate_rows = lw.ffn_gate.matmul_proj(ffn_dim, &ffnq);
            let up_rows = lw.ffn_up.matmul_proj(ffn_dim, &ffnq);
            let act_rows: Vec<Vec<f32>> = (0..kk)
                .map(|i| {
                    gate_rows[i]
                        .iter()
                        .zip(&up_rows[i])
                        .map(|(g, u)| gelu_tanh(*g) * u)
                        .collect()
                })
                .collect();
            let actq = SharedActivationBatch::new(&act_rows);
            let mlp_rows = lw.ffn_down.matmul_proj(hidden, &actq);
            for i in 0..kk {
                let ffn_out = rms_norm(&mlp_rows[i], Some(&lw.post_ffw_norm), eps);
                for (a, b) in hs[i].iter_mut().zip(&ffn_out) {
                    *a += b;
                }
                // PLE residual (per token, cheap f32 matvecs).
                if let (Some(ig), Some(pj), Some(pnn)) = (
                    lw.ple_inp_gate.as_ref(),
                    lw.ple_proj.as_ref(),
                    lw.post_norm.as_ref(),
                ) {
                    let mut gated = f32_matvec(ig, hidden, ple_dim, &hs[i]);
                    for (gv, pv) in gated.iter_mut().zip(&pli_tok[i][li]) {
                        *gv = gelu_tanh(*gv) * pv;
                    }
                    let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                    let pnv = rms_norm(&proj, Some(pnn), eps);
                    for (a, b) in hs[i].iter_mut().zip(&pnv) {
                        *a += b;
                    }
                }
                if lw.ple_output_scale != 1.0 {
                    for v in hs[i].iter_mut() {
                        *v *= lw.ple_output_scale;
                    }
                }
            }
        }

        // --- head, batched over the K positions ---
        let vocab = self.config.vocab_size.unwrap() as usize;
        let lastq: Vec<Vec<f32>> = hs
            .iter()
            .map(|h| rms_norm(h, Some(&self.output_norm), eps))
            .collect();
        // Family-routed like every projection (SHA_E3): the old open-coded
        // match sent only Q6_K through the Q8_K family, so a Q4_K tied head
        // hit `matmul_q`'s K-quant unreachable! on this batched path.
        let lastb = SharedActivationBatch::new(&lastq);
        let mut logits_rows: Vec<Vec<f32>> = self.token_embd.matmul_proj(vocab, &lastb);
        if let Some(cap) = self.g.final_logit_softcapping {
            for logits in logits_rows.iter_mut() {
                soft_cap_in_place(logits, cap);
            }
        }
        Ok(logits_rows)
    }

    /// Compute the full two-branch FFN output for a MoE (A4B/26B) layer.
    ///
    /// `li` is the LOCAL layer index (must have `self.layers[li].moe.is_some()`);
    /// `attn_out` is the post-attention residual (the current hidden state before
    /// the FFN). Returns `ffn_out`, the composed dense+expert result that the
    /// caller ADDS to the residual (`h += ffn_out`).
    ///
    /// This is the single source of truth for the MoE FFN math: the CPU forward
    /// loop calls it for MoE layers, and the CUDA-resident lane reuses it to run
    /// the (bit-exact) FFN on the CPU while attention stays on the GPU. Keeping the
    /// math in one place means the two runtimes cannot diverge on the FFN.
    pub(crate) fn moe_layer_ffn(&self, li: usize, attn_out: &[f32]) -> Vec<f32> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn called on a non-MoE layer");

        // Dense "shared expert" MLP branch: ffn_norm -> parallel GeGLU -> down.
        let xn = rms_norm(attn_out, Some(&lw.ffn_norm), eps);
        let xnq = SharedActivation::new(&xn);
        let gate = lw.ffn_gate.matvec_proj(ffn_dim, &xnq);
        let up = lw.ffn_up.matvec_proj(ffn_dim, &xnq);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * u)
            .collect();
        let mlp = lw.ffn_down.matvec(ffn_dim, hidden, &act);
        // Dense branch keeps its own post-norm (post_norm_1).
        let mlp = rms_norm(&mlp, Some(&moe.post_norm_1), eps);

        // Router runs on attn_out with its OWN weightless norm, scaled by
        // 1/sqrt(n_embd), then the elementwise gate_inp_scale.
        let mut r = rms_norm(attn_out, None, eps);
        let inv = 1.0f32 / (hidden as f32).sqrt();
        for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
            *rv = *rv * inv * sv;
        }
        let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
        // softmax over all experts, then top-k by probability.
        let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut idx: Vec<usize> = (0..moe.n_expert).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(moe.n_expert_used);
        if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
            eprintln!("[route] l={l} e={idx:?}");
        }
        // sum-normalize the selected weights (clamped), w_scale=1.
        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);

        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = SharedActivation::new(&cur_moe);
        let two_nff = 2 * moe.n_ff_exp;
        let mut moe_acc = vec![0f32; hidden];
        // Pre-packed (interleaved 8-row) expert matrices for the AVX2 GEMV, packed
        // once per expert per session and cached; `None` disables the fast path.
        // (The packed path exists only for Q4_0 experts — `pack_cache` is `None`
        // otherwise — so its activations are always the Q8_0 family.)
        for &e in &idx {
            let w = probs[e] / wsum;
            let packed = moe.packed_expert(e, hidden, two_nff);
            // fused gate‖up for expert e: rows e*2nff .. +2nff, in_dim=n_embd.
            // Interleaved 8-row AVX2 GEMV (bit-exact vs the scalar row path) when
            // the expert is pre-packed, else the scalar wire dot (routed by the
            // expert matrices' activation family).
            let gate_up = match &packed {
                Some(p) => packed_band_matvec(&p.gate_up, cur_moe_q.q8_0()),
                None => moe
                    .gate_up_exps
                    .matvec_rows_proj(e * two_nff, two_nff, &cur_moe_q),
            };
            let hexp: Vec<f32> = (0..moe.n_ff_exp)
                .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                .collect();
            let hexp_q = SharedActivation::new(&hexp);
            // down for expert e: rows e*n_embd .. +n_embd, in_dim=n_ff_exp.
            let y = match &packed {
                Some(p) => packed_band_matvec(&p.down, hexp_q.q8_0()),
                None => moe.down_exps.matvec_rows_proj(e * hidden, hidden, &hexp_q),
            };
            let scale = moe.down_exps_scale[e] * w;
            for (a, yv) in moe_acc.iter_mut().zip(&y) {
                *a += yv * scale;
            }
        }
        let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);

        // combine the two branches, then the shared post_ffw_norm.
        let mut combined = mlp;
        for (c, m) in combined.iter_mut().zip(&cur_moe) {
            *c += m;
        }
        rms_norm(&combined, Some(&lw.post_ffw_norm), eps)
    }

    /// One token's forward over the locally-loaded layer range.
    ///
    /// `h_in` is the hidden state arriving from the upstream shard (`None` on
    /// the shard owning layer 0, which embeds the token itself). KV caches are
    /// indexed by LOCAL layer (length `self.layers.len()`). PLE inputs are
    /// recomputed locally from the token id — they depend only on the token's
    /// embedding row, never on upstream activations, so no extra wire traffic.
    /// Returns logits on the shard owning the final layer, otherwise the hidden
    /// state to forward.
    pub fn step_range(
        &self,
        token: u32,
        pos: usize,
        h_in: Option<Vec<f32>>,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Gemma4StepOutput> {
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_local = self.layers.len();
        let block_count = self.config.block_count as usize;
        // PLE tables are sized by the GLOBAL layer count.
        let ple_total = block_count * ple_dim;
        let win = self.g.sliding_window as usize;
        let is_tail = self.first_layer + n_local == block_count;

        let timing = cpu_timing_enabled();
        let t_start = std::time::Instant::now();

        // The scaled token embedding: the layer-0 input on the head shard, and
        // the PLE context source on every shard (PLE depends only on the token).
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let mut h = match h_in {
            Some(h_in) => {
                if h_in.len() != hidden {
                    return Err(BackendError::RuntimeShapeMismatch(format!(
                        "shard received hidden state of {} values, expected {hidden}",
                        h_in.len()
                    )));
                }
                h_in
            }
            None => {
                if self.first_layer != 0 {
                    return Err(BackendError::InvalidModelMetadata(
                        "interior shard requires the upstream hidden state".into(),
                    ));
                }
                h0.clone()
            }
        };

        // Per-layer input (token-identity + context) for the LOCAL layers only:
        // pli[li] belongs to global layer first_layer + li.
        let pli: Vec<Vec<f32>> = if let (Some(te), Some(proj), Some(pn)) = (
            self.per_layer_token_embd.as_ref(),
            self.per_layer_model_proj.as_ref(),
            self.per_layer_proj_norm.as_ref(),
        ) {
            let local_span = n_local * ple_dim;
            let ti = te.dequantize_elements(
                token as usize * ple_total + self.first_layer * ple_dim,
                local_span,
            )?;
            // proj is [ple_total rows x hidden] row-major: take the local rows.
            let proj_local = &proj[self.first_layer * ple_dim * hidden
                ..(self.first_layer * ple_dim + local_span) * hidden];
            let ctx = f32_matvec(proj_local, hidden, local_span, &h0);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            (0..n_local)
                .map(|li| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[li * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect()
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut embed_us = t_start.elapsed().as_micros() as u64;
        let (mut attn_us, mut ffn_us) = (0u64, 0u64);

        for li in 0..n_local {
            let t_layer = std::time::Instant::now();
            let l = self.first_layer + li; // global layer index
            let lw = &self.layers[li];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            // Per-layer geometry: 12B varies kv heads across layers, E2B varies
            // the FFN width. Never use the config scalars here.
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let ffn_dim = self.g.ffn_length_at(l) as usize;
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;

            // RoPE frequency factors apply on FULL attention layers only
            // (reference: gemma4-iswa attaches rope_freqs when !is_swa).
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            let xn = rms_norm(&h, Some(&lw.attn_norm), eps);
            // q/k/v all project the same normed input — quantize it once per
            // activation family (lazily; K-quant projections dot Q8_K).
            let xnq = SharedActivation::new(&xn);
            let mut q = lw.attn_q.matvec_proj(q_dim, &xnq);
            for hh in 0..heads {
                let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
            }
            apply_rope(&mut q, heads, head_dim, pos, theta, rope_factors);
            // Diagnostics: dump head-0 Q (post-norm/post-rope) for one layer for
            // cross-runtime attention bisection (CAMELID_GEMMA4_DUMP_ATTN=<layer>).
            if std::env::var("CAMELID_GEMMA4_DUMP_ATTN").ok().as_deref() == Some(&l.to_string()) {
                eprintln!(
                    "[attn] pos {pos} layer {l} q0..2 [{:.6}, {:.6}, {:.6}] q64..65 [{:.6}, {:.6}] q128..129 [{:.6}, {:.6}]",
                    q[0], q[1], q[2], q[64], q[65], q[128], q[129]
                );
            }

            if l < self.first_kv_shared {
                let mut k = lw
                    .attn_k
                    .as_ref()
                    .expect("validate() guarantees owning layers bind attn_k")
                    .matvec_proj(kv_dim, &xnq);
                // V-less layers (12B full attention) reuse the raw K projection
                // as V — reference: `if v_proj is not present, use Kcur as Vcur`.
                // V then takes the usual weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_proj(kv_dim, &xnq),
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(
                        s,
                        Some(
                            lw.k_norm
                                .as_deref()
                                .expect("validate() guarantees owning layers bind attn_k_norm"),
                        ),
                        eps,
                    ));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
                apply_rope(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                kc[li].push(k);
                vc[li].push(v);
            }
            // Global source layer, then LOCAL cache index (the load-time range
            // check guarantees the source lives on this shard).
            let src_global = if l < self.first_kv_shared {
                l
            } else if sliding {
                self.last_sliding_layer
            } else {
                self.last_full_layer
            };
            let src = src_global - self.first_layer;
            // GQA group against the cache actually read — the SOURCE layer's
            // geometry when KV is shared.
            let group = heads / self.g.kv_heads_at(src_global) as usize;
            let lo = if sliding {
                (pos + 1).saturating_sub(win)
            } else {
                0
            };
            let mut attn = vec![0f32; q_dim];
            for hh in 0..heads {
                let kvh = hh / group;
                let qh = &q[hh * head_dim..(hh + 1) * head_dim];
                let mut scores: Vec<f32> = (lo..=pos)
                    .map(|p| {
                        let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                        qh.iter().zip(kp).map(|(a, b)| a * b).sum()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    den += *s;
                }
                let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                for (idx, p) in (lo..=pos).enumerate() {
                    let w = scores[idx] / den;
                    let vp = &vc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                    for d in 0..head_dim {
                        out[d] += w * vp[d];
                    }
                }
            }
            let o = lw.attn_output.matvec(q_dim, hidden, &attn);
            let on = rms_norm(&o, Some(&lw.post_attn_norm), eps);
            for (a, b) in h.iter_mut().zip(&on) {
                *a += b;
            }
            attn_us += t_layer.elapsed().as_micros() as u64;
            let t_ffn = std::time::Instant::now();
            // FFN. MoE (A4B/26B) rows run the two-branch dense+expert block via
            // the shared `moe_layer_ffn` helper (single source of truth, also used
            // by the CUDA-resident lane); dense rows run just the shared-expert MLP:
            // ffn_norm -> parallel GeGLU -> down -> post_ffw_norm.
            let ffn_out = if lw.moe.is_some() {
                self.moe_layer_ffn(li, &h)
            } else {
                let xn = rms_norm(&h, Some(&lw.ffn_norm), eps);
                let xnq = SharedActivation::new(&xn);
                let gate = lw.ffn_gate.matvec_proj(ffn_dim, &xnq);
                let up = lw.ffn_up.matvec_proj(ffn_dim, &xnq);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| gelu_tanh(*g) * u)
                    .collect();
                let mlp = lw.ffn_down.matvec(ffn_dim, hidden, &act);
                rms_norm(&mlp, Some(&lw.post_ffw_norm), eps)
            };
            for (a, b) in h.iter_mut().zip(&ffn_out) {
                *a += b;
            }
            if let (Some(ig), Some(pj), Some(pnn)) = (
                lw.ple_inp_gate.as_ref(),
                lw.ple_proj.as_ref(),
                lw.post_norm.as_ref(),
            ) {
                let mut gated = f32_matvec(ig, hidden, ple_dim, &h);
                for (gv, pv) in gated.iter_mut().zip(&pli[li]) {
                    *gv = gelu_tanh(*gv) * pv;
                }
                let proj = f32_matvec(pj, ple_dim, hidden, &gated);
                let pnv = rms_norm(&proj, Some(pnn), eps);
                for (a, b) in h.iter_mut().zip(&pnv) {
                    *a += b;
                }
            }
            // `layer_output_scale` multiplies the layer output UNCONDITIONALLY
            // when present (reference applies it outside the PLE block; the
            // dense 12B carries it on every layer with no PLE at all). 1.0 when
            // the tensor is absent.
            if lw.ple_output_scale != 1.0 {
                for v in h.iter_mut() {
                    *v *= lw.ple_output_scale;
                }
            }
            ffn_us += t_ffn.elapsed().as_micros() as u64;
            // Diagnostics only: per-layer hidden-state fingerprint for
            // cross-runtime layer bisection (CAMELID_GEMMA4_DUMP_LAYERS=1).
            if std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!(
                    "[h] pos {pos} layer {l} l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                    h[0], h[1], h[2], h[3]
                );
            }
        }

        if !is_tail {
            return Ok(Gemma4StepOutput::Hidden(h));
        }

        let t_out = std::time::Instant::now();
        let last = rms_norm(&h, Some(&self.output_norm), eps);
        let vocab = self.config.vocab_size.unwrap() as usize;
        // token_embd is vocab-major (row v = the v-th embedding), so the tied
        // logits are a single block-wise Q8 matvec — far faster than per-row
        // dequantize_elements over the whole 262k vocab.
        let mut logits = self.token_embd.matvec(hidden, vocab, &last);
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        if timing {
            use std::sync::atomic::Ordering::Relaxed;
            // The PLE prep ran inside the embed window; attention/ffn windows
            // bracket the per-layer work; everything after the last layer is
            // the output projection (norm + 262K-vocab GEMV + soft-cap).
            embed_us = embed_us.min(t_start.elapsed().as_micros() as u64);
            CPU_EMBED_US.fetch_add(embed_us, Relaxed);
            CPU_ATTN_US.fetch_add(attn_us, Relaxed);
            CPU_FFN_US.fetch_add(ffn_us, Relaxed);
            CPU_OUTPROJ_US.fetch_add(
                t_out.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            CPU_STEP_N.fetch_add(1, Relaxed);
        }
        Ok(Gemma4StepOutput::Logits(logits))
    }

    /// Greedily generate up to `max_new` tokens from `prompt`, with an incremental
    /// KV cache (one forward step per token). Returns (decoded continuation, the
    /// generated token ids).
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index, not a count
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        // Lossless n-gram speculative decode (opt-in, single-node non-MoE rows): verify
        // a batch of drafted tokens in ONE weight pass via `step_chunk`. Output is
        // token-for-token identical to the greedy loop below — every committed token is
        // the target's own argmax — so it makes no support/parity claim, only speed.
        if std::env::var("CAMELID_GEMMA4_SPEC_DECODE").is_ok() && self.supports_chunk_forward() {
            let generated =
                self.spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)?;
            if cpu_timing_enabled() {
                report_cpu_timing();
            }
            let text = self.tokenizer.decode(&generated, true)?;
            return Ok((text, generated));
        }
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            logits = self.step(next, pos, &mut kc, &mut vc)?;
            pos += 1;
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// BASALT Phase 3 forced-decode harness surface (`basalt_eval_protocol.md`
    /// §5.1): teacher-force `forced` through the model. Prefills `prompt` exactly
    /// like [`Self::generate_greedy`], then at each continuation step `i` observes
    /// the FULL next-token logit vector (the distribution predicting continuation
    /// position `i`) via `on_step(i, &logits)` BEFORE feeding `forced[i]` as the
    /// next input — regardless of the model's argmax, ignoring stop tokens (the
    /// forced list defines the step count) and never taking the speculative path.
    /// NO engine math changes: the forward pass is the same [`Self::step`] loop
    /// the greedy decoder drives; only the next-token choice differs. The final
    /// forced token is not fed (its prediction is already observed; feeding it
    /// would only compute an unrecorded extra step). Returns the prompt token ids.
    pub fn forced_decode<F: FnMut(usize, &[f32])>(
        &self,
        prompt: &str,
        forced: &[u32],
        mut on_step: F,
    ) -> Result<Vec<u32>> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        // Boundary bookkeeping lives in `drive_forced_steps` (unit-tested with a
        // scripted step fn): observe step i's logits BEFORE feeding forced[i];
        // exactly forced.len() observations; the final forced token never fed.
        let mut pos = prompt_tokens.len();
        drive_forced_steps(
            forced,
            logits,
            |tok| -> Result<Vec<f32>> {
                let next = self.step(tok, pos, &mut kc, &mut vc)?;
                pos += 1;
                Ok(next)
            },
            |i, logits: &Vec<f32>| on_step(i, logits),
        )?;
        Ok(prompt_tokens)
    }

    /// [`Self::generate_greedy`] with a per-step FULL-logit observer, for the
    /// BASALT Phase 3 harness (`--dump-step-logits` without `--force-tokens`):
    /// `on_step(i, &logits)` fires for every continuation logit vector BEFORE its
    /// argmax is taken — including the final vector whose argmax is a stop token
    /// (that step is observed, then the loop breaks without emitting the token).
    /// Always drives the plain one-token [`Self::step`] loop (the speculative
    /// path does not surface per-step logits); the token-choice math is identical
    /// to [`Self::generate_greedy`], so the emitted ids match the unobserved
    /// greedy decode of the same prompt.
    pub fn generate_greedy_observed<F: FnMut(usize, &[f32])>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_step: F,
    ) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        let mut generated = Vec::new();
        for i in 0..max_new {
            on_step(i, &logits);
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            // `next` is generated token #(generated.len()-1), sitting at absolute
            // position prompt_len + that index — identical to generate_greedy's
            // running `pos` counter.
            let pos = prompt_tokens.len() + generated.len() - 1;
            logits = self.step(next, pos, &mut kc, &mut vc)?;
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Lossless n-gram speculative decode, forced on (no env var). Returns the SAME
    /// `(text, ids)` as [`Self::generate_greedy`] token-for-token — speculation only
    /// changes how many tokens fall out of one weight read. Requires a single-node
    /// non-MoE row ([`Self::supports_chunk_forward`]); falls back to the plain greedy
    /// loop otherwise. Exposed for the spec-vs-greedy parity test and the CLI flag.
    pub fn generate_greedy_speculative(
        &self,
        prompt: &str,
        max_new: usize,
    ) -> Result<(String, Vec<u32>)> {
        if !self.supports_chunk_forward() {
            return self.generate_greedy(prompt, max_new);
        }
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        let generated =
            self.spec_decode_generate(&mut kc, &mut vc, logits, &prompt_tokens, &eot, max_new)?;
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Lossless greedy n-gram speculative decode for single-node non-MoE gemma4 rows.
    /// Given the prefilled caches and the prefill `logits` (predicting the first new
    /// position), repeatedly: commit `t0 = argmax(logits)`, draft its continuation from
    /// history (prompt-lookup), verify `[t0, drafts..]` in ONE batched `step_chunk`,
    /// accept the longest prefix of drafts that equals the target's own argmax, roll the
    /// KV cache back to the accepted length, and carry the divergence position's logits
    /// into the next round. Emits exactly the greedy token stream; drafts only change how
    /// many tokens fall out of a single weight read.
    #[allow(clippy::needless_range_loop)]
    fn spec_decode_generate(
        &self,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        mut logits: Vec<f32>,
        prompt_tokens: &[u32],
        eot: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>> {
        use crate::inference::speculative::{
            accepted_draft_prefix, NGramDrafter, DEFAULT_NGRAM_DRAFT_TOKENS,
        };
        let max_draft = std::env::var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NGRAM_DRAFT_TOKENS)
            .max(1);
        let drafter = NGramDrafter::default();
        let argmax = |l: &[f32]| -> u32 {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap()
        };
        let (mut accepted_rounds, mut accepted_drafts) = (0u64, 0u64);
        let spec_timing = std::env::var("CAMELID_GEMMA4_SPEC_TIMING").is_ok();
        let mut history = prompt_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut pos = prompt_tokens.len();
        while generated.len() < max_new {
            // t0 is the target's own next-token argmax — always greedy-correct.
            let t0 = argmax(&logits);
            if eot.contains(&t0) {
                break;
            }
            generated.push(t0);
            history.push(t0);
            if generated.len() >= max_new {
                break;
            }
            let budget = max_new - generated.len();
            let drafts = drafter.draft(&history, max_draft.min(budget));
            // Verify [t0, d1..dm] at positions pos..pos+m in one weight pass: rows[i]
            // predicts position pos+i+1.
            let mut chunk = Vec::with_capacity(1 + drafts.len());
            chunk.push(t0);
            chunk.extend_from_slice(&drafts);
            let rows = self.step_chunk(&chunk, pos, kc, vc)?;
            let preds: Vec<u32> = (0..drafts.len()).map(|i| argmax(&rows[i])).collect();
            let j = accepted_draft_prefix(&drafts, &preds);
            accepted_rounds += 1;
            accepted_drafts += j as u64;
            let mut stopped = false;
            for &d in &drafts[..j] {
                if generated.len() >= max_new {
                    break;
                }
                if eot.contains(&d) {
                    stopped = true;
                    break;
                }
                generated.push(d);
                history.push(d);
            }
            if stopped {
                break;
            }
            // Keep KV through the last accepted position (pos+j); discard the rejected
            // draft tail. rows[j] predicts pos+j+1 → it's next round's t0 source.
            let keep = pos + j + 1;
            for li in 0..kc.len() {
                kc[li].truncate(keep);
                vc[li].truncate(keep);
            }
            pos = keep;
            logits = rows.into_iter().nth(j).expect("rows[j] exists");
        }
        if spec_timing {
            let toks = generated.len().max(1) as f64;
            eprintln!(
                "[spec] {} tokens in {accepted_rounds} verify passes ({:.2} tokens/pass; {accepted_drafts} drafts accepted)",
                generated.len(),
                toks / accepted_rounds.max(1) as f64,
            );
        }
        Ok(generated)
    }

    /// Greedy decode that emits the incremental decoded-text delta after each new
    /// token via `on_delta`. The delta is computed by decoding the cumulative
    /// generated sequence and yielding the newly-appended suffix, which keeps
    /// SentencePiece spacing/multi-byte pieces correct (token-at-a-time decode
    /// would mangle them). Returns the same `(text, ids)` as `generate_greedy`.
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index
    pub fn generate_greedy_streaming<F: FnMut(&str)>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
    ) -> Result<(String, Vec<u32>)> {
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.step(tok, pos, &mut kc, &mut vc)?;
        }
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            // Decode cumulatively and emit only the newly-appended suffix.
            let full = self.tokenizer.decode(&generated, true)?;
            if let Some(delta) = full.strip_prefix(&emitted) {
                if !delta.is_empty() {
                    on_delta(delta);
                }
            }
            emitted = full;
            logits = self.step(next, pos, &mut kc, &mut vc)?;
            pos += 1;
        }
        Ok((emitted, generated))
    }
}

/// GPU-resident gemma4 decode runtime: the Q8 layer weights live on the GPU (nocopy
/// `WirePages`), the per-layer KV caches persist on the GPU, and each token's forward
/// runs in one Metal command buffer ([`crate::metal::Gemma4ResidentModel`]). The
/// per-token embedding, PLE `pli`, and dual-θ RoPE tables are computed on the CPU and
/// uploaded. Gated by `crate::metal::gemma4_gpu_enabled()` at the call site. Numerics
/// follow the CPU [`Gemma4Runtime`] (attention score scale = 1.0 — gemma folds it in).
#[cfg(target_os = "macos")]
pub struct Gemma4GpuRuntime {
    model: crate::metal::Gemma4ResidentModel,
    tokenizer: Tokenizer,
    g: Gemma4Metadata,
    /// token_embd + per_layer_token_embd stay in the FILE-BACKED mmap (not owned RAM).
    /// The 8GB layer weights are anonymous GPU WirePages; if these embeddings were also
    /// owned/anonymous, the OS would swap the WirePages under 16GB pressure (no file
    /// cache to evict) and the GPU forward would thrash. File-backed pages are evicted
    /// (and cheaply re-read) instead — robust, at the cost of a cold-fault on the
    /// per-token row gather.
    token_embd: WireQuant,
    per_layer_token_embd: Option<WireQuant>,
    /// GGUF `rope_freqs.weight` factors — applied on FULL attention layers'
    /// cos/sin tables only (the reference's proportional rope).
    rope_factors: Option<Vec<f32>>,
    _mmap: Arc<GgufWireMmap>,
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
    /// QAT hybrid lane: the tied head is Q6_K (no GPU kernel), so the GPU runs the
    /// decoder layers (Q4_0) and the CPU runs the head. False for the all-Q8 path,
    /// where the head is encoded on the GPU inside `forward_token`.
    head_on_cpu: bool,
    /// Held for the CPU head (`head_on_cpu`): output RMS-norm weights + vocab.
    output_norm: Vec<f32>,
    vocab: usize,
    eps: f32,
}

#[cfg(target_os = "macos")]
impl Gemma4GpuRuntime {
    /// Load the model with the Q8 layer weights resident on the GPU. `max_positions`
    /// is the KV-cache capacity (must cover prompt + generated tokens).
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let gguf = read_metadata(path)?;
        // BASALT Amendment 3 (D-B2 sidecar guard): this lane never ran the sidecar
        // check, so a sidecar-bearing NVFP4 file could compute wrong logits — refuse
        // it here, before any binding (cfg-independent, unit-tested on every host).
        // GABBRO M3-followup lifted the blanket NVFP4 refusal (the Metal resident lane
        // now runs NVFP4 layer projections via nvfp4_block_linear_row_ksplit_f32y_wire);
        // the D17/T5 NaN-sentinel guard moved to nvfp4_metal_sentinel_check below,
        // where the mmap is available to scan the wire bytes the raw upload reads.
        nvfp4_sidecar_check(&gguf.tensors)?;
        let config = LlamaModelConfig::from_gguf(&gguf)?;
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        // The GPU-resident decode kernels run the layer projections as Q8_0 (34-byte
        // wire blocks), Q4_0 (18-byte QAT wire blocks), or NVFP4 (36-byte 64-value
        // superblocks; GABBRO M3) — all parity-gated GPU GEMVs. The tied head is read
        // separately: Q8_0 runs on the GPU (inside forward_token); Q6_K (the QAT tied
        // head, no GPU kernel) runs on the CPU via the held WireQuant. Layer 0's attn_q
        // is representative of the projection format (the export quantizes every
        // layer's projections alike).
        let layer_fmt = gemma4_metal_layer_fmt(
            store
                .descriptor(&binding.layers[0].attn_q.name)?
                .tensor_type,
        )?;
        let head_on_cpu = match store.descriptor(&binding.token_embedding.name)?.tensor_type {
            GgufTensorType::Q8_0 => false, // GPU Q8 head
            GgufTensorType::Q6K => true,   // CPU Q6_K head (QAT tied head)
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "gemma4 GPU runtime supports a Q8_0 or Q6_K tied head; \
                     token embedding is {other:?}"
                )));
            }
        };
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        // The mmap backs token_embd + per_layer_token_embd (file-backed = evictable, so
        // it never forces the anonymous GPU WirePages to swap). GPU layer weights load
        // separately as page-aligned WirePages.
        let mmap = GgufWireMmap::map(path)?;
        // GABBRO M3-followup (D17/T5 fail-closed): the resident lane reads NVFP4 layer
        // wire RAW via WirePages (bypassing WireQuant::new's sentinel scan), so the
        // NaN-sentinel guard fires here — one pass over each NVFP4 tensor's UE4M3 scale
        // bytes before any GPU upload; 0x7F/0xFF refuses fail-closed, matching the CPU
        // wire lane. (nvfp4_sidecar_check for D-B2 already ran up top.)
        nvfp4_metal_sentinel_check(&gguf.tensors, &mmap)?;
        // Warm the embedding mmap off the loading thread (matching the CPU lane): the
        // QAT hybrid head reads the whole Q6_K tied table every token on the CPU, and
        // every row gather hits this mapping, so the first token would otherwise pay the
        // cold page-fault cost serially. madvise(WILLNEED) on a USB-backed volume blocks
        // until the range is paged in, so it MUST NOT run on the loading thread.
        {
            let mmap = mmap.clone();
            std::thread::spawn(move || mmap.advise_willneed());
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let hidden = config.embedding_length as usize;
        let heads = config.attention_head_count as usize;
        let n_layers = config.block_count as usize;
        let vocab = config.vocab_size.unwrap() as usize;
        let eps = config.rms_norm_epsilon;
        let ple_dim = g.per_layer_input_dim as usize;
        let softcap = g.final_logit_softcapping.unwrap_or(0.0);

        let file = std::fs::File::open(path).map_err(|e| BackendError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let pages = |name: &str| -> Result<Arc<crate::wire_mmap::WirePages>> {
            let desc = store.descriptor(name)?;
            crate::wire_mmap::WirePages::read_from_file(
                &file,
                desc.absolute_offset,
                desc.n_bytes as usize,
            )
        };

        let plan = g.layer_plan(n_layers, heads);
        let mut layers = Vec::with_capacity(n_layers);
        let mut ple = Vec::with_capacity(n_layers);
        let mut layer_scales = Vec::with_capacity(n_layers);
        let mut owns_kv = Vec::with_capacity(n_layers);
        let mut kv_source = Vec::with_capacity(n_layers);
        for (l, lb) in binding.layers.iter().enumerate() {
            let hd = g.head_dim_at(l) as usize;
            // Per-layer geometry (12B varies kv heads, E2B varies FFN width).
            let kv_heads = plan[l].kv_heads;
            let ffn_dim = g.ffn_length_at(l) as usize;
            let owns = plan[l].owns_kv;
            // Trimmed shared-KV exports (e.g. E4B QAT) omit attn_k / attn_k_norm /
            // attn_v on non-owning layers: those layers project no K/V and run
            // attention against the source layer's cache, so the resident attention
            // never reads these tensors. Pass never-read placeholders to keep the
            // layer shape uniform. A KV-owning layer that omits them is a real error.
            let q_pages_arc = pages(&lb.attn_q.name)?;
            let k_pages_arc = match &lb.attn_k {
                Some(d) => pages(&d.name)?,
                None if !owns => Arc::clone(&q_pages_arc),
                None => {
                    return Err(BackendError::UnsupportedTensorType(format!(
                        "gemma4 GPU runtime requires attn_k on KV-owning layers; \
                         layer {l} omits it"
                    )));
                }
            };
            let k_norm_v = match &lb.attn_k_norm {
                Some(d) => f32t(&d.name)?,
                None if !owns => vec![0.0f32; hd],
                None => {
                    return Err(BackendError::UnsupportedTensorType(format!(
                        "gemma4 GPU runtime requires attn_k_norm on KV-owning layers; \
                         layer {l} omits it"
                    )));
                }
            };
            let layer = crate::metal::Gemma4ResidentLayer::from_wire_pages(
                layer_fmt,
                f32t(&lb.attn_norm.name)?,
                f32t(&lb.attn_q_norm.name)?,
                k_norm_v,
                f32t(&lb.post_attention_norm.name)?,
                f32t(&lb.ffn_norm.name)?,
                f32t(&lb.post_ffw_norm.name)?,
                &q_pages_arc,
                &k_pages_arc,
                lb.attn_v
                    .as_ref()
                    .map(|d| pages(&d.name))
                    .transpose()?
                    .as_ref(),
                &pages(&lb.attn_output.name)?,
                &pages(&lb.ffn_gate.name)?,
                &pages(&lb.ffn_up.name)?,
                &pages(&lb.ffn_down.name)?,
                heads,
                kv_heads,
                hd,
                ffn_dim,
                eps,
            )
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture("Metal unavailable".into())
            })?;
            layers.push(layer);
            // layer_output_scale is unconditional in the reference. E-series
            // layers apply it inside the PLE encode; dense layers (no PLE) get
            // it standalone via `layer_scales`.
            let output_scale = lb
                .ple_output_scale
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0);
            layer_scales.push(output_scale);
            ple.push(match (&lb.ple_inp_gate, &lb.ple_proj, &lb.post_norm) {
                (Some(ig), Some(pj), Some(pn)) => Some(crate::metal::Gemma4ResidentPle {
                    inp_gate: f32t(&ig.name)?,
                    proj: f32t(&pj.name)?,
                    post_norm: f32t(&pn.name)?,
                    output_scale,
                }),
                _ => None,
            });
            owns_kv.push(plan[l].owns_kv);
            kv_source.push(plan[l].kv_source_layer);
        }

        let token_embd = q8(&binding.token_embedding.name)?;
        let output_norm = f32t(&binding.output_norm.name)?;
        // QAT hybrid (Q6_K head on CPU): don't hand the tied table to the GPU head — pass
        // an empty slice so no ~0.5 GB head buffer is uploaded. The all-Q8 lane passes the
        // wire bytes for the GPU head as before.
        let head_wire: &[u8] = if head_on_cpu { &[] } else { token_embd.bytes() };
        let model = crate::metal::Gemma4ResidentModel::new(
            layers,
            ple,
            layer_scales,
            owns_kv,
            kv_source,
            head_wire,
            output_norm.clone(),
            hidden,
            vocab,
            softcap,
            eps,
            max_positions,
            1.0, // gemma folds the attention scale into the (QK-normed) query
        )
        .ok_or_else(|| BackendError::UnsupportedModelArchitecture("Metal unavailable".into()))?;

        let mut model = model;
        let per_layer_model_proj = binding
            .per_layer_model_proj
            .as_ref()
            .map(|d| f32t(&d.name))
            .transpose()?;
        let per_layer_proj_norm = binding
            .per_layer_proj_norm
            .as_ref()
            .map(|d| f32t(&d.name))
            .transpose()?;
        // Move the per-token pli computation onto the GPU (folded-constant matvec +
        // per-head norm + residual-add), eliminating the ~12ms/token CPU prep.
        if let (Some(proj), Some(pn)) = (&per_layer_model_proj, &per_layer_proj_norm) {
            model.set_pli(proj, pn, ple_dim);
        }

        Ok(Self {
            model,
            tokenizer,
            per_layer_token_embd: binding
                .per_layer_token_embd
                .as_ref()
                .map(|d| q8(&d.name))
                .transpose()?,
            rope_factors: binding
                .rope_freqs
                .as_ref()
                .map(|d| f32t(&d.name))
                .transpose()?,
            token_embd,
            g,
            _mmap: mmap,
            hidden,
            ple_dim,
            n_layers,
            head_on_cpu,
            output_norm,
            vocab,
            eps,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Run one token's forward on the GPU and return the next-token logits.
    fn forward(&self, token: u32, position: usize) -> Result<Vec<f32>> {
        let t_prep = std::time::Instant::now();
        let hidden = self.hidden;
        let ple_dim = self.ple_dim;
        let ple_total = self.n_layers * ple_dim;
        let filled = position + 1;
        // Scaled input embedding (CPU gather).
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        // PLE `pli` is computed ON the GPU (Gemma4ResidentModel::set_pli) — the CPU
        // only gathers this token's per_layer_token_embd row, with the gemma constants
        // (ple_dim^0.5 * FRAC_1_SQRT_2) folded in so the GPU just residual-adds it.
        let ti: Vec<f32> = if let Some(te) = self.per_layer_token_embd.as_ref() {
            let scale = (ple_dim as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            te.dequantize_elements(token as usize * ple_total, ple_total)?
                .iter()
                .map(|v| v * scale)
                .collect()
        } else {
            Vec::new()
        };
        // Per-layer RoPE tables (dual θ, per-type head_dim) + sliding window start.
        let win = self.g.sliding_window as usize;
        let inputs: Vec<crate::metal::Gemma4TokenLayerInput> = (0..self.n_layers)
            .map(|l| {
                let hd = self.g.head_dim_at(l) as usize;
                let theta = self.g.rope_freq_base_at(l);
                let half = hd / 2;
                // Frequency factors (proportional rope) on FULL layers only.
                let factors = if self.g.is_sliding_layer(l) {
                    None
                } else {
                    self.rope_factors.as_deref()
                };
                let (mut cos_t, mut sin_t) = (vec![0f32; half], vec![0f32; half]);
                for i in 0..half {
                    let mut freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                    if let Some(factors) = factors {
                        freq /= factors[i];
                    }
                    let (s, c) = (position as f32 * freq).sin_cos();
                    cos_t[i] = c;
                    sin_t[i] = s;
                }
                let window_start = if self.g.is_sliding_layer(l) {
                    filled.saturating_sub(win)
                } else {
                    0
                };
                crate::metal::Gemma4TokenLayerInput {
                    cos_t,
                    sin_t,
                    pli: Vec::new(), // pli now computed on the GPU; not passed per-layer
                    window_start,
                }
            })
            .collect();
        let prep_us = t_prep.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        // All-Q8 path: the GPU encodes the head and returns logits directly. QAT hybrid
        // path: the GPU returns the final hidden state and the CPU runs the Q6_K tied
        // head (rms_norm -> Q6_K logits matvec -> final_logit_softcap), matching the CPU
        // runtime's head exactly.
        let logits = if self.head_on_cpu {
            let last_hidden = self
                .model
                .forward_token_hidden(&h0, &inputs, &ti, position)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
                })?;
            let last = rms_norm(&last_hidden, Some(&self.output_norm), self.eps);
            let mut logits = self.token_embd.matvec(self.hidden, self.vocab, &last);
            if let Some(cap) = self.g.final_logit_softcapping {
                soft_cap_in_place(&mut logits, cap);
            }
            logits
        } else {
            self.model
                .forward_token(&h0, &inputs, &ti, position)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
                })?
        };
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok() {
            PREP_US.fetch_add(prep_us as u64, std::sync::atomic::Ordering::Relaxed);
            GPU_US.fetch_add(
                t_gpu.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            FWD_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(logits)
    }

    /// Greedy generate up to `max_new` tokens from `prompt` on the GPU.
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward(tok, pos)?;
        }
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        for _ in 0..max_new {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            logits = self.forward(next, pos)?;
            pos += 1;
        }
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok() {
            use std::sync::atomic::Ordering::Relaxed;
            let (n, prep, gpu) = (
                FWD_N.load(Relaxed).max(1),
                PREP_US.load(Relaxed),
                GPU_US.load(Relaxed),
            );
            eprintln!(
                "[gpu-timing] {n} forwards: cpu prep avg {}us, gpu avg {}us (total {}us/fwd)",
                prep / n,
                gpu / n,
                (prep + gpu) / n
            );
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }
}

#[cfg(target_os = "macos")]
static PREP_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static GPU_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static FWD_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-layer RMSNorm weights kept resident on the GPU (small; ~tens of KB/layer).
#[cfg(feature = "cuda")]
struct Gemma4LayerNormsDev {
    attn_norm: cudarc::driver::CudaSlice<f32>,
    q_norm: cudarc::driver::CudaSlice<f32>,
    k_norm: Option<cudarc::driver::CudaSlice<f32>>,
    post_attn_norm: cudarc::driver::CudaSlice<f32>,
    ffn_norm: cudarc::driver::CudaSlice<f32>,
    post_ffw_norm: cudarc::driver::CudaSlice<f32>,
    // MoE-only: the dense shared-expert branch's own post-norm (`post_norm_1`) and
    // the sparse expert-sum post-norm (`post_norm_2`). Resident so the whole MoE
    // dense + compose runs on the GPU (M4). `None` on dense rows.
    moe_post_norm_1: Option<cudarc::driver::CudaSlice<f32>>,
    moe_post_norm_2: Option<cudarc::driver::CudaSlice<f32>>,
}

/// Per-layer projection weights kept resident on the GPU in the SoA layout
/// `q8_gemv` reads (uploaded once at load). For E4B Q8 this is ~4–4.5 GB and fits
/// a 6 GB card because the big embeddings (`token_embd`, `per_layer_token_embd`)
/// stay on the CPU for the head + PLE gather. `k`/`v` exist only on owning layers;
/// `v` is `None` on V-less layers (V reuses the K projection).
#[cfg(feature = "cuda")]
struct Gemma4LayerWeightsDev {
    q: cudarc::driver::CudaSlice<u8>,
    k: Option<cudarc::driver::CudaSlice<u8>>,
    v: Option<cudarc::driver::CudaSlice<u8>>,
    o: cudarc::driver::CudaSlice<u8>,
    gate: cudarc::driver::CudaSlice<u8>,
    up: cudarc::driver::CudaSlice<u8>,
    down: cudarc::driver::CudaSlice<u8>,
    // Per-projection quant lane (mixed Q4_0 file: Q4_0 projections + Q4_1 ffn_down).
    q_q: GemmaLayerQuant,
    k_q: GemmaLayerQuant,
    v_q: GemmaLayerQuant,
    o_q: GemmaLayerQuant,
    gate_q: GemmaLayerQuant,
    up_q: GemmaLayerQuant,
    down_q: GemmaLayerQuant,
}

/// Quant lane of a resident gemma4 layer projection. All consume Q8_0
/// activations; Q8_0 weights are SoA-repacked, Q4_0/Q4_1/NVFP4 are raw wire.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GemmaLayerQuant {
    Q8_0,
    Q4_0,
    Q4_1,
    Nvfp4,
}

#[cfg(feature = "cuda")]
impl GemmaLayerQuant {
    fn from_wire(f: WireFormat) -> Self {
        match f {
            WireFormat::Q8_0 => Self::Q8_0,
            WireFormat::Q4_0 => Self::Q4_0,
            WireFormat::Q4_1 => Self::Q4_1,
            // BASALT Phase 4: NVFP4 layer projections now reside on the CUDA lane
            // (nvfp4_gemv, raw 36-byte wire). `nvfp4_cuda_lane_check` still refuses
            // every other uncovered format before this catch-all can panic.
            WireFormat::Nvfp4 => Self::Nvfp4,
            other => {
                panic!("gemma4 layer projection quant {other:?} unsupported (Q8_0/Q4_0/Q4_1/NVFP4)")
            }
        }
    }
}

/// Per-projection GEMV dispatch for the gemma4 resident layer loop. All lanes take the
/// shared Q8_0 activation buffers (`d_ins`/`d_inq`) and `blocks_per_row = cols/32`; the
/// weight is SoA Q8_0 or raw Q4_0/Q4_1 wire. Mirrors `cuda_resident::dispatch_gemv` but
/// for the gemma4 Q8_0-activation lanes only.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemma_proj_gemv(
    s: &std::sync::Arc<cudarc::driver::CudaStream>,
    kernels: &crate::cuda_resident::CudaResidentKernels,
    quant: GemmaLayerQuant,
    in_scales: &cudarc::driver::CudaSlice<f32>,
    in_quants: &cudarc::driver::CudaSlice<i8>,
    weight: &cudarc::driver::CudaView<'_, u8>,
    rows: usize,
    blocks_per_row: usize,
    out: &mut cudarc::driver::CudaSlice<f32>,
) -> std::result::Result<(), cudarc::driver::DriverError> {
    match quant {
        GemmaLayerQuant::Q8_0 => crate::cuda_resident::launch_gemv(
            s,
            &kernels.gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
        ),
        GemmaLayerQuant::Q4_0 => crate::cuda_resident::launch_q4_0_gemv(
            s,
            &kernels.q4_0_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ),
        GemmaLayerQuant::Q4_1 => crate::cuda_resident::launch_q4_1_gemv(
            s,
            &kernels.q4_1_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ),
        // BASALT Phase 4: NVFP4 raw-wire GEMV. `launch_nvfp4_gemv` returns a typed
        // Nvfp4LaunchError; the odd-block variant is the I-k-div lane guard and is
        // structurally unreachable here (the parse boundary refuses non-%64 NVFP4
        // first-dims at load — k_div_fixture_trips_parse_refusal — so every gemma4
        // projection reaching the CUDA GEMV has an even Q8_0-block count), matching
        // the codebase's guard-then-unreachable idiom (matvec's Q5_K arm).
        GemmaLayerQuant::Nvfp4 => match crate::cuda_resident::launch_nvfp4_gemv(
            s,
            &kernels.nvfp4_gemv,
            in_scales,
            in_quants,
            weight,
            rows,
            blocks_per_row,
            out,
            0,
        ) {
            Ok(()) => Ok(()),
            Err(crate::cuda_resident::Nvfp4LaunchError::Driver(e)) => Err(e),
            Err(crate::cuda_resident::Nvfp4LaunchError::OddBlocksPerRow(bpr)) => unreachable!(
                "gemma4 NVFP4 projection reached the CUDA GEMV with an odd Q8_0-block count \
                 {bpr} (in_dim % 64 != 0); the parse boundary refuses non-%64 NVFP4 tensors \
                 before load"
            ),
        },
    }
}

/// Per-layer PLE weights resident on the GPU (small f32 matrices), so the
/// per-layer PLE injection runs entirely on the device — no host round-trip.
#[cfg(feature = "cuda")]
struct Gemma4LayerPleDev {
    inp_gate: cudarc::driver::CudaSlice<f32>,
    proj: cudarc::driver::CudaSlice<f32>,
    post_norm: cudarc::driver::CudaSlice<f32>,
    output_scale: f32,
}

/// A captured decode CUDA graph, wrapped Send: cudarc's `CudaGraph` is not `Send`,
/// but the engine lives behind a `Mutex` in `Arc<Gemma4ServeRuntime>` (one request
/// at a time), so the raw graph handle is only ever touched under the lock.
#[cfg(feature = "cuda")]
struct SendGraph(cudarc::driver::CudaGraph);
#[cfg(feature = "cuda")]
unsafe impl Send for SendGraph {}

#[cfg(feature = "cuda")]
fn cu(e: cudarc::driver::DriverError) -> BackendError {
    BackendError::InvalidModelMetadata(format!("gemma4 cuda: {e}"))
}

/// Repack a GGUF Q8_0 weight tensor (34-byte blocks: f16 scale + 32 i8) into the
/// compact SoA layout `q8_gemv` reads: all 32-i8 quant groups first, then the
/// original f16 scale bits. Mirrors `cuda_resident::repack_q8_soa` but consumes
/// the raw GGUF wire directly (that helper expects an already-f32-scale 36B block).
#[cfg(feature = "cuda")]
fn q8_wire_to_soa(wire: &[u8]) -> Vec<u8> {
    const W: usize = 34;
    let n = wire.len() / W;
    let mut out = vec![0u8; n * 32 + n * 2];
    let (quants, scales) = out.split_at_mut(n * 32);
    for b in 0..n {
        let blk = &wire[b * W..b * W + W];
        quants[b * 32..b * 32 + 32].copy_from_slice(&blk[2..34]);
        scales[b * 2..b * 2 + 2].copy_from_slice(&blk[0..2]);
    }
    out
}

/// Quant lane of the GPU tied head: Q8_0 (`q8_gemv` over SoA-repacked weight, Q8_0
/// input) or Q6_K (`q6k_gemv` over raw wire, Q8_K input).
#[cfg(feature = "cuda")]
enum HeadLane {
    Q8_0,
    Q4K,
    Q6K,
}

/// Resident GPU tied head. `weight` is the vocab-major projection (SoA for Q8_0, raw
/// Q6_K wire otherwise); input is quantized by the fused rms_norm+quantize into
/// `inq`/`ins`; `logits` is dtoh'd once per token. `blocks` is blocks-per-row passed
/// to the GEMV (`hidden/32` for Q8_0, `hidden/256` for Q6_K).
#[cfg(feature = "cuda")]
struct Gemma4HeadDev {
    lane: HeadLane,
    weight: cudarc::driver::CudaSlice<u8>,
    output_norm: cudarc::driver::CudaSlice<f32>,
    logits: cudarc::driver::CudaSlice<f32>,
    inq: cudarc::driver::CudaSlice<i8>,
    ins: cudarc::driver::CudaSlice<f32>,
    blocks: usize,
    softcap: f32,
}

/// Resident PLE context-projection (the `proj·h` matvec that dominated CPU prep).
/// `proj` (per_layer_model_proj, [block_count*ple_dim x hidden] f32, ~110 MB) and
/// `proj_norm` stay resident; `ti` holds this token's per_layer_token_embd row
/// (gathered+dequantized on the CPU each token — that table is too big to reside).
#[cfg(feature = "cuda")]
struct Gemma4PleCtxDev {
    proj: cudarc::driver::CudaSlice<f32>,
    proj_norm: cudarc::driver::CudaSlice<f32>,
    ti: cudarc::driver::CudaSlice<f32>,
    ple_total: usize,
    proj_scale: f32,
    embed_scale: f32,
}

/// One cached MoE expert's two Q4_0 weight slices, resident on the GPU. `gate_up`
/// is the fused gate‖up rows (`2*n_ff_exp × hidden`) and `down` is the down rows
/// (`hidden × n_ff_exp`) — the exact byte ranges `moe_layer_ffn`'s CPU path reads
/// from the mmap for this expert. `last_used` is the LRU recency stamp.
#[cfg(feature = "cuda")]
struct SserExpertDev {
    gate_up: cudarc::driver::CudaSlice<u8>,
    down: cudarc::driver::CudaSlice<u8>,
    last_used: u64,
}

/// SSER (self-specializing expert residency) VRAM cache: a per-(layer,expert) LRU
/// of Q4_0 expert weight slices. A single user's session fires a skewed, stable
/// subset of the experts; keeping the hot ones resident lets their two GEMVs run on
/// the GPU (336 GB/s) instead of the CPU (the ~187 ms/token MoE wall). Gated behind
/// `CAMELID_SSER_CACHE` (off = M1 all-CPU MoE); capacity `CAMELID_SSER_CACHE_EXPERTS`
/// (#experts). Eviction is LRU on miss-when-full. Bit-exact: the GPU `q4_0_gemv` is
/// proven bit-identical to the CPU `q4_0_wire_row_dot` the cache-miss path uses.
// Throwaway M3 profiling counters (env CAMELID_SSER_PROFILE): total ns spent in
// the dense-MLP branch, router, and expert loop across all MoE layers/tokens.
#[cfg(feature = "cuda")]
static SSER_PROF_DENSE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_ROUTER_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_EXPERT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_HIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "cuda")]
static SSER_PROF_MISS_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "cuda")]
struct SserCache {
    entries: std::collections::HashMap<(u16, u16), SserExpertDev>,
    capacity: usize,
    clock: u64,
    // Diagnostics (per-generate; reset by the harness before each run).
    hits: u64,
    misses: u64,
}

#[cfg(feature = "cuda")]
impl SserCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            capacity: capacity.max(1),
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Mark `key` most-recently-used and return true if resident.
    fn touch(&mut self, key: (u16, u16)) -> bool {
        self.clock += 1;
        let clock = self.clock;
        if let Some(e) = self.entries.get_mut(&key) {
            e.last_used = clock;
            true
        } else {
            false
        }
    }

    /// Evict the least-recently-used entry if at capacity (called before an insert).
    fn evict_if_full(&mut self) {
        if self.entries.len() < self.capacity {
            return;
        }
        if let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_used) {
            self.entries.remove(&victim);
        }
    }
}

/// CUDA gemma4 decode engine (Windows/NVIDIA). Wraps a CPU-loaded [`Gemma4Runtime`]
/// for weights/config/tokenizer and runs the per-token forward through the shared
/// `crate::cuda_resident` kernels. Layer projection weights are streamed from the
/// host mmap per layer (so E4B Q8 fits a 6 GB card); small ops with no large weight
/// read — the scaled embedding and the PLE injection — run on the CPU/GPU as noted.
/// The tied Q6_K head runs on the GPU (`gpu_head`) when resident, else on the CPU.
/// Per-layer geometry (head_dim 256/512, dual-θ RoPE, sliding window, cross-layer KV
/// source) comes from `plan`.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub struct Gemma4CudaResident {
    cpu: Gemma4Runtime,
    kernels: crate::cuda_resident::CudaResidentKernels,
    /// A dedicated non-default stream for the decode forward. The legacy default
    /// stream (`kernels.stream`) cannot be put into capture mode, so all per-token
    /// work runs here to allow recording the layer stack into a CUDA graph.
    cap_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    plan: Vec<crate::model::Gemma4LayerPlan>,
    norms: Vec<Gemma4LayerNormsDev>,
    lweights: Vec<Gemma4LayerWeightsDev>,
    ple: Vec<Option<Gemma4LayerPleDev>>,
    block_count: usize,
    heads: usize,
    hidden: usize,
    ple_dim: usize,
    eps: f32,
    vocab: usize,
    max_positions: usize,
    first_kv_shared: usize,
    half_max: usize,
    /// Captured per-token layer-stack graph (lazily recorded after a warmup pass);
    /// replaying it replaces ~900 per-token kernel launches with one launch.
    decode_graph: Option<SendGraph>,
    /// True once the layer kernels have run once directly (cold first-launch lazy
    /// init isn't capturable, so we warm up before recording the graph).
    warmed: bool,
    /// GPU tied head (Q6_K only). `Some` runs the final projection on the GPU
    /// (fused rms_norm+Q8K-quant -> q6k_gemv over the vocab -> soft-cap), replacing
    /// the ~1.2 s/token CPU Q6_K matvec that otherwise dominates decode. `None` keeps
    /// the head on the CPU (non-Q6_K head, or `hidden` not a multiple of 256).
    gpu_head: Option<Gemma4HeadDev>,
    /// GPU PLE context projection. `Some` runs `proj·h` + per-layer rms-norm + combine
    /// on the GPU (writing `d_pli` directly), replacing the ~27.5M-mult CPU matvec that
    /// was the remaining prep bottleneck. `None` falls back to the CPU pli compute.
    gpu_ple_ctx: Option<Gemma4PleCtxDev>,
    // Per-owning-layer f16 KV caches ([kv_head][pos][head_dim]); None on shared layers.
    cache_k: Vec<Option<cudarc::driver::CudaSlice<u16>>>,
    cache_v: Vec<Option<cudarc::driver::CudaSlice<u16>>>,
    /// Token sequence currently represented in the persistent KV cache (the last request's
    /// prompt + its generated tokens). On the next request the longest matching prefix is
    /// reused, so only the genuinely new tokens are prefilled — this keeps multi-turn TTFT
    /// roughly constant instead of growing with conversation length.
    cached_tokens: Vec<u32>,
    // Reused per-token/per-layer device scratch (sized to per-layer maxima).
    d_hidden: cudarc::driver::CudaSlice<f32>,
    d_normed: cudarc::driver::CudaSlice<f32>,
    d_inq: cudarc::driver::CudaSlice<i8>,
    d_ins: cudarc::driver::CudaSlice<f32>,
    d_q: cudarc::driver::CudaSlice<f32>,
    d_k: cudarc::driver::CudaSlice<f32>,
    d_v: cudarc::driver::CudaSlice<f32>,
    d_attn: cudarc::driver::CudaSlice<f32>,
    d_attnq: cudarc::driver::CudaSlice<i8>,
    d_attns: cudarc::driver::CudaSlice<f32>,
    d_o: cudarc::driver::CudaSlice<f32>,
    d_gate: cudarc::driver::CudaSlice<f32>,
    d_up: cudarc::driver::CudaSlice<f32>,
    d_geglu: cudarc::driver::CudaSlice<f32>,
    d_geglu_q: cudarc::driver::CudaSlice<i8>,
    d_geglu_s: cudarc::driver::CudaSlice<f32>,
    d_ffn_out: cudarc::driver::CudaSlice<f32>,
    // M4: holds the MoE dense shared-expert branch (branch A) result on-device
    // (`rms_norm(down_out, post_norm_1)`) while the sparse expert branch runs, so the
    // two branches can be composed on the GPU without a host round-trip.
    d_mlp: cudarc::driver::CudaSlice<f32>,
    // All layers' RoPE tables for this token (slot li at li*half_max), uploaded once
    // so the per-layer loop has no in-loop memcpy (required for graph capture).
    d_cos_all: cudarc::driver::CudaSlice<f32>,
    d_sin_all: cudarc::driver::CudaSlice<f32>,
    d_position: cudarc::driver::CudaSlice<i32>,
    // PLE scratch (GPU injection): d_pli holds this token's per-layer inputs.
    d_pli: cudarc::driver::CudaSlice<f32>,
    d_ple_gated: cudarc::driver::CudaSlice<f32>,
    d_ple_gated2: cudarc::driver::CudaSlice<f32>,
    d_ple_proj: cudarc::driver::CudaSlice<f32>,
    d_ple_normed: cudarc::driver::CudaSlice<f32>,
    // SSER (M2): per-(layer,expert) VRAM LRU of Q4_0 expert slices. `None` when
    // `CAMELID_SSER_CACHE` is unset (M1 all-CPU MoE). Wrapped in a `RefCell` so the
    // cached FFN can mutate the LRU/counters through a shared `&self` — the per-token
    // forward loop holds long-lived immutable borrows of `self.kernels`/`self.cpu`
    // that a `&mut self` MoE call would conflict with. Device scratch for the expert
    // GEMVs is allocated locally per call (batch-1 tiny GEMVs are launch-bound, so the
    // alloc is negligible and it keeps the hot path `&self`).
    sser: Option<std::cell::RefCell<SserCache>>,
}

#[cfg(feature = "cuda")]
impl Gemma4CudaResident {
    /// Load the model (CPU runtime, weights mmap'd), bring up the CUDA kernels,
    /// upload per-layer norms, and allocate the KV caches + scratch. `max_positions`
    /// bounds the resident KV cache.
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let cpu = Gemma4Runtime::load(path)?;
        // BASALT Amendment 3 review fix: refuse NVFP4 layer projections with a
        // typed error BEFORE the `GemmaLayerQuant::from_wire` catch-all (`upw`
        // below) can panic. The CPU wire lane serves NVFP4 in this release;
        // CUDA-resident NVFP4 is Phase 4 (BASALT).
        nvfp4_cuda_lane_check(cpu.layers.iter().flat_map(|lw| {
            [
                Some(lw.attn_q.format),
                lw.attn_k.as_ref().map(|w| w.format),
                lw.attn_v.as_ref().map(|w| w.format),
                Some(lw.attn_output.format),
                Some(lw.ffn_gate.format),
                Some(lw.ffn_up.format),
                Some(lw.ffn_down.format),
                lw.moe.as_ref().map(|m| m.gate_up_exps.format),
                lw.moe.as_ref().map(|m| m.down_exps.format),
            ]
            .into_iter()
            .flatten()
        }))?;
        let kernels = crate::cuda_resident::CudaResidentKernels::new()
            .map_err(BackendError::InvalidModelMetadata)?;
        // Disable cudarc's automatic cross-stream event tracking. Allocating a second
        // (capture) stream below puts the context in multi-stream mode, which otherwise
        // makes every launch record/drop CudaEvents on its slice args — and event
        // create/destroy is not permitted while a stream is capturing, breaking the
        // decode graph. The whole forward runs on a single stream (`cap_stream`), so
        // ordering is implicit and manual; no auto-sync is needed. All gemma4 device
        // slices are created below while this is off, so they never track events.
        unsafe { kernels.ctx.disable_event_tracking() };
        // Capture-capable stream for the decode graph (the default stream is not).
        let cap_stream = kernels.ctx.new_stream().map_err(cu)?;
        let s = kernels.stream.clone();
        let block_count = cpu.config.block_count as usize;
        let heads = cpu.config.attention_head_count as usize;
        let hidden = cpu.config.embedding_length as usize;
        let vocab = cpu.token_embd.element_count / hidden;
        let eps = cpu.config.rms_norm_epsilon;
        let first_kv_shared = cpu.first_kv_shared;
        let plan = cpu.g.layer_plan(block_count, heads);
        let ple_dim = cpu
            .per_layer_proj_norm
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0);
        // GPU tied head: make the vocab-major head weight resident and run the final
        // projection on the GPU. The CPU matvec over the 262K vocab is ~1.2 s/token —
        // the decode bottleneck — versus a few ms for the GEMV. ~0.55-0.7 GB on E4B.
        let softcap = cpu.g.final_logit_softcapping.unwrap_or(0.0);
        let gpu_head = match cpu.token_embd.format {
            WireFormat::Q8_0 if hidden.is_multiple_of(32) => {
                let blocks = hidden / 32;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q8_0,
                    weight: s
                        .clone_htod(&q8_wire_to_soa(cpu.token_embd.bytes()))
                        .map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(hidden).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            WireFormat::Q6K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q6K,
                    weight: s.clone_htod(cpu.token_embd.bytes()).map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            // Q4_K tied head (mixed Q4_0 file): q4k_gemv over raw 144-byte wire, Q8_K input.
            WireFormat::Q4K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q4K,
                    weight: s.clone_htod(cpu.token_embd.bytes()).map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            _ => None,
        };

        // GPU PLE context projection: make per_layer_model_proj (~110 MB f32) + proj_norm
        // resident so `proj·h` (the ~27.5M-mult per-token matvec that dominated CPU prep)
        // runs on the GPU. The per_layer_token_embd table stays CPU (too big to reside);
        // only this token's row is gathered/dequantized + uploaded each step.
        let gpu_ple_ctx = match (
            cpu.per_layer_model_proj.as_ref(),
            cpu.per_layer_proj_norm.as_ref(),
            cpu.per_layer_token_embd.as_ref(),
        ) {
            (Some(proj), Some(pn), Some(_)) if ple_dim > 0 => {
                let ple_total = block_count * ple_dim;
                Some(Gemma4PleCtxDev {
                    proj: s.clone_htod(&proj[0..ple_total * hidden]).map_err(cu)?,
                    proj_norm: s.clone_htod(pn).map_err(cu)?,
                    ti: s.alloc_zeros::<f32>(ple_total).map_err(cu)?,
                    ple_total,
                    proj_scale: (hidden as f32).powf(-0.5),
                    embed_scale: (ple_dim as f32).sqrt(),
                })
            }
            _ => None,
        };

        // Per-layer maxima for scratch sizing.
        let q_dim_max = plan.iter().map(|p| p.q_dim).max().unwrap_or(0);
        let kv_dim_max = plan.iter().map(|p| p.kv_dim).max().unwrap_or(0);
        let head_dim_max = plan.iter().map(|p| p.head_dim).max().unwrap_or(0);
        let ffn_max = (0..block_count)
            .map(|l| cpu.g.ffn_length_at(l) as usize)
            .max()
            .unwrap_or(0);

        // Upload per-layer norm weights (resident; small).
        let mut norms = Vec::with_capacity(block_count);
        for lw in &cpu.layers {
            norms.push(Gemma4LayerNormsDev {
                attn_norm: s.clone_htod(&lw.attn_norm).map_err(cu)?,
                q_norm: s.clone_htod(&lw.q_norm).map_err(cu)?,
                k_norm: match lw.k_norm.as_ref() {
                    Some(w) => Some(s.clone_htod(w).map_err(cu)?),
                    None => None,
                },
                post_attn_norm: s.clone_htod(&lw.post_attn_norm).map_err(cu)?,
                ffn_norm: s.clone_htod(&lw.ffn_norm).map_err(cu)?,
                post_ffw_norm: s.clone_htod(&lw.post_ffw_norm).map_err(cu)?,
                moe_post_norm_1: match lw.moe.as_ref() {
                    Some(m) => Some(s.clone_htod(&m.post_norm_1).map_err(cu)?),
                    None => None,
                },
                moe_post_norm_2: match lw.moe.as_ref() {
                    Some(m) => Some(s.clone_htod(&m.post_norm_2).map_err(cu)?),
                    None => None,
                },
            });
        }

        // Per-layer projection weights, resident in the SoA layout q8_gemv reads
        // (uploaded once; the big embeddings stay on the CPU). k/v only on owning layers.
        // Repack + upload one projection, tagging its quant lane: Q8_0 -> SoA (q8_gemv),
        // Q4_0/Q4_1 -> raw wire (q4_0_gemv/q4_1_gemv read the wire directly).
        let upw = |wq: &WireQuant| -> Result<(cudarc::driver::CudaSlice<u8>, GemmaLayerQuant)> {
            let quant = GemmaLayerQuant::from_wire(wq.format);
            let bytes = match quant {
                GemmaLayerQuant::Q8_0 => q8_wire_to_soa(wq.bytes()),
                // Q4_0/Q4_1/NVFP4 residency is raw wire passthrough: the GEMV reads
                // the packed nibbles + in-block scales directly, so the 4.x-bpw
                // footprint is preserved in VRAM (no host-side dequant/expansion).
                GemmaLayerQuant::Q4_0 | GemmaLayerQuant::Q4_1 | GemmaLayerQuant::Nvfp4 => {
                    wq.bytes().to_vec()
                }
            };
            Ok((s.clone_htod(&bytes).map_err(cu)?, quant))
        };
        let mut lweights = Vec::with_capacity(block_count);
        for (li, lw) in cpu.layers.iter().enumerate() {
            let owns = plan[li].owns_kv;
            let (q, q_q) = upw(&lw.attn_q)?;
            let (k, k_q) = if owns {
                let (kk, kq) = upw(lw.attn_k.as_ref().expect("owning layer binds attn_k"))?;
                (Some(kk), kq)
            } else {
                (None, GemmaLayerQuant::Q8_0)
            };
            let (v, v_q) = if owns {
                match lw.attn_v.as_ref() {
                    Some(wv) => {
                        let (vv, vq) = upw(wv)?;
                        (Some(vv), vq)
                    }
                    // V-less layers reuse the K weight, so V's quant == K's.
                    None => (None, k_q),
                }
            } else {
                (None, GemmaLayerQuant::Q8_0)
            };
            let (o, o_q) = upw(&lw.attn_output)?;
            let (gate, gate_q) = upw(&lw.ffn_gate)?;
            let (up, up_q) = upw(&lw.ffn_up)?;
            let (down, down_q) = upw(&lw.ffn_down)?;
            lweights.push(Gemma4LayerWeightsDev {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
                q_q,
                k_q,
                v_q,
                o_q,
                gate_q,
                up_q,
                down_q,
            });
        }

        // Per-layer PLE weights resident (small f32 matrices) for on-GPU injection.
        let mut ple = Vec::with_capacity(block_count);
        for lw in &cpu.layers {
            ple.push(
                if let (Some(ig), Some(pj), Some(pn)) = (
                    lw.ple_inp_gate.as_ref(),
                    lw.ple_proj.as_ref(),
                    lw.post_norm.as_ref(),
                ) {
                    Some(Gemma4LayerPleDev {
                        inp_gate: s.clone_htod(ig).map_err(cu)?,
                        proj: s.clone_htod(pj).map_err(cu)?,
                        post_norm: s.clone_htod(pn).map_err(cu)?,
                        output_scale: lw.ple_output_scale,
                    })
                } else {
                    None
                },
            );
        }

        // Per-owning-layer f16 KV caches sized to that layer's kv geometry.
        let mut cache_k = Vec::with_capacity(block_count);
        let mut cache_v = Vec::with_capacity(block_count);
        for p in &plan {
            if p.owns_kv {
                let n = p.kv_dim * max_positions;
                cache_k.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
                cache_v.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
            } else {
                cache_k.push(None);
                cache_v.push(None);
            }
        }

        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n.max(1));
        let alloc_i = |n: usize| s.alloc_zeros::<i8>(n.max(1));
        // SSER (M2): enable the per-(layer,expert) VRAM cache only when the model has
        // MoE layers AND the flag is set. Capacity defaults to ~1000 experts (the
        // measured hot set); each cached expert is ~2*n_ff_exp*(hidden/32)*18 +
        // hidden*(n_ff_exp/32)*18 bytes of Q4_0 wire (~3.3 MB on the 26B), so ~1000
        // experts ≈ ~3.3 GB — under the ~3.6 GB free after the resident set. Tunable
        // via CAMELID_SSER_CACHE_EXPERTS.
        let first_moe = cpu.layers.iter().find_map(|lw| lw.moe.as_ref());
        let sser = if let (Some(moe), true) =
            (first_moe, std::env::var_os("CAMELID_SSER_CACHE").is_some())
        {
            // Per-expert VRAM cost: the two Q4_0 slices this expert's GEMVs read.
            // gate_up = 2*n_ff_exp rows of hidden values; down = hidden rows of
            // n_ff_exp values; Q4_0 packs 32 values per 18-byte block.
            const WB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
            let two_nff = 2 * moe.n_ff_exp;
            let per_expert_bytes = two_nff * (hidden / 32) * WB + hidden * (moe.n_ff_exp / 32) * WB;
            // Budget: keep the cache under ~80% of the free VRAM after the resident set
            // (leaving headroom for the per-token scratch + the KV cache growth).
            let (free, _total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
            // Cache budget = free VRAM at load MINUS a fixed reserve for the transient
            // per-miss weight uploads (a few pooled ~6 MiB `clone_htod` buffers) and
            // driver slack. The KV caches and per-token scratch are already allocated
            // ABOVE (so `free` excludes them) — the only dynamic post-cache consumer is
            // those small transient buffers, whose need is ~constant, not proportional
            // to free VRAM. A fixed reserve therefore lets the cache claim far more of
            // the card than the old flat 0.80 factor did: on the 6 GB box this lifts
            // the cap ~690 -> ~820 experts, cutting the miss count and measuring
            // +~50% steady decode (miss-bound, capacity-limited). Reserve tunable via
            // CAMELID_SSER_CACHE_RESERVE_MIB; a hard 0.98 cap on the free fraction is a
            // final belt-and-suspenders against a pathologically small `free`.
            let reserve_mib = std::env::var("CAMELID_SSER_CACHE_RESERVE_MIB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(160);
            let reserve = reserve_mib * 1024 * 1024;
            let hard_cap = (free as f64 * 0.98) as usize;
            let budget = free.saturating_sub(reserve).min(hard_cap);
            let fit_cap = budget.checked_div(per_expert_bytes).unwrap_or(0);
            let req_cap = std::env::var("CAMELID_SSER_CACHE_EXPERTS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1000);
            // Honor the smaller of the requested capacity and what free VRAM allows.
            let cap = req_cap.min(fit_cap).max(1);
            eprintln!(
                "[sser] expert-residency cache ON: capacity {cap} experts ({} MiB each; requested {req_cap}, VRAM-fit {fit_cap}; {} MiB free)",
                per_expert_bytes / (1024 * 1024),
                free / (1024 * 1024),
            );
            Some(SserCache::new(cap))
        } else {
            None
        };
        let me = Self {
            norms,
            lweights,
            ple,
            block_count,
            heads,
            hidden,
            ple_dim,
            eps,
            vocab,
            max_positions,
            first_kv_shared,
            half_max: head_dim_max / 2,
            decode_graph: None,
            warmed: false,
            cache_k,
            cache_v,
            cached_tokens: Vec::new(),
            d_hidden: alloc_f(hidden).map_err(cu)?,
            d_normed: alloc_f(hidden).map_err(cu)?,
            d_inq: alloc_i(hidden).map_err(cu)?,
            d_ins: alloc_f(hidden / 32).map_err(cu)?,
            d_q: alloc_f(q_dim_max).map_err(cu)?,
            d_k: alloc_f(kv_dim_max).map_err(cu)?,
            d_v: alloc_f(kv_dim_max).map_err(cu)?,
            d_attn: alloc_f(q_dim_max).map_err(cu)?,
            d_attnq: alloc_i(q_dim_max).map_err(cu)?,
            d_attns: alloc_f(q_dim_max / 32).map_err(cu)?,
            d_o: alloc_f(hidden).map_err(cu)?,
            d_gate: alloc_f(ffn_max).map_err(cu)?,
            d_up: alloc_f(ffn_max).map_err(cu)?,
            d_geglu: alloc_f(ffn_max).map_err(cu)?,
            d_geglu_q: alloc_i(ffn_max).map_err(cu)?,
            d_geglu_s: alloc_f(ffn_max / 32).map_err(cu)?,
            d_ffn_out: alloc_f(hidden).map_err(cu)?,
            d_mlp: alloc_f(hidden).map_err(cu)?,
            d_cos_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_sin_all: alloc_f(block_count * (head_dim_max / 2)).map_err(cu)?,
            d_position: s.alloc_zeros::<i32>(1).map_err(cu)?,
            d_pli: alloc_f(block_count * ple_dim).map_err(cu)?,
            d_ple_gated: alloc_f(ple_dim).map_err(cu)?,
            d_ple_gated2: alloc_f(ple_dim).map_err(cu)?,
            d_ple_proj: alloc_f(hidden).map_err(cu)?,
            d_ple_normed: alloc_f(hidden).map_err(cu)?,
            sser: sser.map(std::cell::RefCell::new),
            plan,
            kernels,
            cap_stream,
            gpu_head,
            gpu_ple_ctx,
            cpu,
        };
        // Every device slice above was allocated + zeroed (`alloc_zeros`) on the DEFAULT
        // stream (`kernels.stream`), but the per-token forward runs on `cap_stream`. With
        // event-tracking disabled during load there is no automatic cross-stream ordering,
        // so the first forward's uploads on `cap_stream` (e.g. the RoPE cos/sin table) can
        // race the still-in-flight load-time memsets on the default stream — which then
        // clobber the just-uploaded values with zeros (observed: cos=0 at position 0 →
        // K zeroed → wrong tokens). Drain the default stream here so all load-time zeroing
        // is complete before any cap_stream work begins.
        me.kernels.stream.synchronize().map_err(cu)?;
        // Re-enable cudarc's auto event-tracking now that every gemma4 device slice is
        // allocated. Those slices were created while it was off, so they carry no
        // CudaEvents and the decode-graph capture stays clean; restoring it here keeps
        // multi-stream synchronization correct for any other model loaded into this
        // context afterwards (e.g. a later Llama reload in a serve process).
        unsafe { me.kernels.ctx.enable_event_tracking() };
        Ok(me)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.cpu.tokenizer
    }

    pub fn layer_plan(&self) -> &[crate::model::Gemma4LayerPlan] {
        &self.plan
    }

    /// SSER cache diagnostics: `(hits, misses, resident_experts, capacity)`.
    /// `None` when the cache is disabled (`CAMELID_SSER_CACHE` unset).
    pub fn sser_stats(&self) -> Option<(u64, u64, usize, usize)> {
        if std::env::var_os("CAMELID_SSER_PROFILE").is_some() {
            use std::sync::atomic::Ordering::Relaxed;
            let d = SSER_PROF_DENSE_NS.load(Relaxed) as f64 / 1e6;
            let r = SSER_PROF_ROUTER_NS.load(Relaxed) as f64 / 1e6;
            let e = SSER_PROF_EXPERT_NS.load(Relaxed) as f64 / 1e6;
            let hit = SSER_PROF_HIT_NS.load(Relaxed) as f64 / 1e6;
            let miss = SSER_PROF_MISS_NS.load(Relaxed) as f64 / 1e6;
            eprintln!(
                "[sser-profile] MoE CPU-side totals: dense-MLP {d:.0} ms, router {r:.0} ms, expert-loop {e:.0} ms (sum {:.0} ms)",
                d + r + e
            );
            eprintln!(
                "[sser-profile]   expert-loop split: hit-path {hit:.0} ms, miss-path {miss:.0} ms, rest(prep+dtoh+sync) {:.0} ms",
                e - hit - miss
            );
        }
        self.sser.as_ref().map(|c| {
            let c = c.borrow();
            (c.hits, c.misses, c.entries.len(), c.capacity)
        })
    }

    /// Reset the SSER hit/miss counters (keeps resident weights). Lets the harness
    /// separate warm-up misses from steady-state hit-rate. No-op when disabled.
    pub fn sser_reset_counters(&self) {
        if let Some(c) = self.sser.as_ref() {
            let mut c = c.borrow_mut();
            c.hits = 0;
            c.misses = 0;
        }
    }

    /// SSER (M2/M3/M4) sparse-expert branch of the MoE FFN. Runs the router on the
    /// CPU (tiny), then every selected expert's two GEMVs on the GPU — cached in VRAM
    /// (hit) or uploaded+promoted (miss) — accumulating each expert's weighted
    /// down-GEMV into an on-device buffer (`scaled_axpy`). Returns that device
    /// accumulator (the sparse expert sum, BEFORE `post_norm_2`); the caller composes
    /// it on-device with the GPU dense branch (M4). `attn_out` is the post-attention
    /// residual (already copied device->host by the caller for the router).
    ///
    /// The dense "shared expert" branch and the final compose+norms now run on the
    /// GPU in the layer loop (M4); this method owns ONLY the router + 8 expert GEMVs.
    ///
    /// Parity: the GPU `q4_0_gemv` is bit-identical to the CPU `q4_0_wire_row_dot`
    /// the miss path uses (proven in `q4_0_gemv_matches_oracle`), and the GPU GeGLU
    /// (`geglu_mul`) matches `gelu_tanh` within the accepted f16-KV/tanhf floor — so
    /// cached and uncached experts produce the same content tokens as M1.
    #[allow(clippy::too_many_lines)]
    fn moe_layer_ffn_cached(
        &self,
        li: usize,
        attn_out: &[f32],
    ) -> Result<cudarc::driver::CudaSlice<f32>> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let s = self.cap_stream.clone();
        let k = &self.kernels;
        let hidden = self.hidden;
        let eps = self.eps;
        let cpu = &self.cpu;
        let lw = &cpu.layers[li];
        let l = cpu.first_layer + li;
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn_cached called on a non-MoE layer");
        let sser = self
            .sser
            .as_ref()
            .expect("moe_layer_ffn_cached requires the SSER cache");

        let prof = std::env::var_os("CAMELID_SSER_PROFILE").is_some();
        let tp1 = std::time::Instant::now();

        // --- Router (CPU, identical). ---
        let mut r = rms_norm(attn_out, None, eps);
        let inv = 1.0f32 / (hidden as f32).sqrt();
        for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
            *rv = *rv * inv * sv;
        }
        let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
        let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let mut idx: Vec<usize> = (0..moe.n_expert).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(moe.n_expert_used);
        if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
            eprintln!("[route] l={l} e={idx:?}");
        }
        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);
        if prof {
            SSER_PROF_ROUTER_NS.fetch_add(
                tp1.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let tp2 = std::time::Instant::now();

        // --- Expert branch: quantize the shared input once (CPU), upload once. ---
        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = quantize_q8_0_blocks(&cur_moe);
        let two_nff = 2 * moe.n_ff_exp;
        let nff = moe.n_ff_exp;
        let gu_blocks = hidden / 32; // gate_up in_dim = hidden
        let down_blocks = nff / 32; // down in_dim = n_ff_exp
        let gu_row_bytes = gu_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
        let down_row_bytes = down_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;

        // Upload the shared Q8_0 expert input (scales + concatenated i8 quants) once —
        // every selected expert dots against the same activation. Device scratch is
        // allocated locally (keeps the hot path `&self`; batch-1 GEMVs are launch-bound
        // so the alloc cost is negligible next to the per-expert launch overhead).
        let in_scales: Vec<f32> = cur_moe_q.iter().map(|b| b.scale).collect();
        let mut in_quants = vec![0i8; gu_blocks * 32];
        for (b, blk) in cur_moe_q.iter().enumerate() {
            in_quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
        }
        let d_in_s = s.clone_htod(&in_scales).map_err(cu)?;
        let d_in_q = s.clone_htod(&in_quants).map_err(cu)?;
        let mut d_gate_up = s.alloc_zeros::<f32>(two_nff).map_err(cu)?;
        let mut d_geglu = s.alloc_zeros::<f32>(nff).map_err(cu)?;
        let mut d_geglu_q = s.alloc_zeros::<i8>(nff).map_err(cu)?;
        let mut d_geglu_s = s.alloc_zeros::<f32>(down_blocks).map_err(cu)?;
        let mut d_y = s.alloc_zeros::<f32>(hidden).map_err(cu)?;
        // M3/M4 on-device MoE accumulator: every selected expert (hit OR uploaded-miss)
        // folds its weighted down-GEMV output straight into this device buffer (one
        // scaled_axpy launch each). In M4 the buffer is RETURNED to the caller and
        // composed with the dense branch on-device — no per-layer dtoh at all.
        let mut d_moe_acc = s.alloc_zeros::<f32>(hidden).map_err(cu)?;

        let gate_up_bytes = moe.gate_up_exps.bytes();
        let down_bytes = moe.down_exps.bytes();

        for &e in &idx {
            let w = probs[e] / wsum;
            let scale = moe.down_exps_scale[e] * w;
            let key = (l as u16, e as u16);

            let cached = sser.borrow_mut().touch(key);
            let te = std::time::Instant::now();
            // On a MISS, upload the expert's two Q4_0 slices and insert them into the
            // VRAM cache (promotion) BEFORE running — then the GPU pipeline below reads
            // the freshly-resident slices exactly as it does for a hit. This moves the
            // expensive part of a miss (the ~1.8 ms CPU expert matvec, which profiling
            // showed was ~72% of all MoE time) onto the GPU: a miss now costs only the
            // ~6 MiB weight htod + the same tiny GEMV launches a hit already pays.
            if !cached {
                sser.borrow_mut().misses += 1;
                let gu_off = e * two_nff * gu_row_bytes;
                let down_off = e * hidden * down_row_bytes;
                let gu_slice = &gate_up_bytes[gu_off..gu_off + two_nff * gu_row_bytes];
                let down_slice = &down_bytes[down_off..down_off + hidden * down_row_bytes];
                let gu_dev = s.clone_htod(gu_slice).map_err(cu)?;
                let down_dev = s.clone_htod(down_slice).map_err(cu)?;
                let mut c = sser.borrow_mut();
                c.evict_if_full();
                c.clock += 1;
                let stamp = c.clock;
                c.entries.insert(
                    key,
                    SserExpertDev {
                        gate_up: gu_dev,
                        down: down_dev,
                        last_used: stamp,
                    },
                );
            } else {
                sser.borrow_mut().hits += 1;
            }

            // --- GPU pipeline (hit OR promoted-miss): fused gate‖up GEMV -> GeGLU ->
            // quantize -> down GEMV -> weighted on-device accumulate. Every expert now
            // takes the identical bit-exact GPU path, so the token stream no longer
            // depends on cache warmth (removes host/device path-divergence). Hold the
            // shared cache borrow for the whole launch sequence so the resident weight
            // views stay valid; `touch`/`insert` above made `key` the newest entry, so
            // it cannot be evicted while this borrow is live. ---
            {
                let c = sser.borrow();
                let ent = c.entries.get(&key).expect("hit or just-promoted miss");
                let gu_dev = ent.gate_up.slice(0..ent.gate_up.len());
                let down_dev = ent.down.slice(0..ent.down.len());
                // gate‖up: two_nff rows, gu_blocks blocks/row.
                crate::cuda_resident::launch_q4_0_gemv(
                    &s,
                    &k.q4_0_gemv,
                    &d_in_s,
                    &d_in_q,
                    &gu_dev,
                    two_nff,
                    gu_blocks,
                    &mut d_gate_up,
                    0,
                )
                .map_err(cu)?;
                // GeGLU: gelu_tanh(gate[o]) * up[o] where gate = out[0..nff], up = out[nff..2nff].
                {
                    let gate_v = d_gate_up.slice(0..nff);
                    let up_v = d_gate_up.slice(nff..two_nff);
                    let cfg = LaunchConfig {
                        grid_dim: ((nff as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let n_i = nff as i32;
                    let mut b = s.launch_builder(&k.geglu_mul);
                    b.arg(&gate_v).arg(&up_v).arg(&mut d_geglu).arg(&n_i);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &d_geglu,
                    &mut d_geglu_q,
                    &mut d_geglu_s,
                    down_blocks,
                )
                .map_err(cu)?;
                // down: hidden rows, down_blocks blocks/row.
                crate::cuda_resident::launch_q4_0_gemv(
                    &s,
                    &k.q4_0_gemv,
                    &d_geglu_s,
                    &d_geglu_q,
                    &down_dev,
                    hidden,
                    down_blocks,
                    &mut d_y,
                    0,
                )
                .map_err(cu)?;
            }
            // On-device weighted accumulate: d_moe_acc[i] += d_y[i] * scale (deferred to
            // one dtoh after the loop). scaled_axpy(acc, y, scale, n): acc += y*scale.
            {
                let cfg = LaunchConfig {
                    grid_dim: ((hidden as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_i = hidden as i32;
                let mut b = s.launch_builder(&k.scaled_axpy);
                b.arg(&mut d_moe_acc).arg(&d_y).arg(&scale).arg(&n_i);
                unsafe { b.launch(cfg) }.map_err(cu)?;
            }
            if prof {
                let ns = te.elapsed().as_nanos() as u64;
                use std::sync::atomic::Ordering::Relaxed;
                if cached {
                    SSER_PROF_HIT_NS.fetch_add(ns, Relaxed);
                } else {
                    SSER_PROF_MISS_NS.fetch_add(ns, Relaxed);
                }
            }
        }
        // Every selected expert (hit OR uploaded-miss) accumulated into `d_moe_acc` in
        // strict idx order, so the layer's expert sum is one left-to-right f32
        // accumulation identical to M1's single-buffer host loop. In M4 we return the
        // device buffer directly (no dtoh) — the caller applies post_norm_2, composes
        // with the dense branch, applies post_ffw_norm, and adds to the residual, all
        // on the GPU.
        if prof {
            SSER_PROF_EXPERT_NS.fetch_add(
                tp2.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Ok(d_moe_acc)
    }

    /// One token's forward; returns next-token logits. Mirrors the CPU
    /// `Gemma4Runtime::step_range` op order exactly (the parity oracle).
    fn forward_token(
        &mut self,
        token: u32,
        position: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        // Run on the capture-capable stream (not the default stream) so the layer
        // stack can be recorded into a CUDA graph.
        let s = self.cap_stream.clone();
        let hidden = self.hidden;
        let heads = self.heads;
        let ple_dim = self.ple_dim;
        let eps = self.eps;

        // ---- CPU: scaled embedding (small f32 gather); upload before the GPU PLE proj ----
        let h: Vec<f32> = self
            .cpu
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let ple_total = self.block_count * ple_dim;
        s.memcpy_htod(&h, &mut self.d_hidden).map_err(cu)?;
        s.memcpy_htod(&[position as i32], &mut self.d_position)
            .map_err(cu)?;
        // PLE per-layer inputs -> d_pli. GPU path: ctx = proj·h (f32_gemv) -> *proj_scale ->
        // per-layer rms_norm(proj_norm) -> + ti*embed_scale -> *1/sqrt(2), all on device
        // (the ~27.5M-mult matvec was the CPU prep bottleneck). The per_layer_token_embd
        // row `ti` is gathered on the CPU (that table is too big to reside). CPU fallback below.
        if let Some(ctxdev) = self.gpu_ple_ctx.as_mut() {
            let ti = self
                .cpu
                .per_layer_token_embd
                .as_ref()
                .expect("gpu_ple_ctx implies per_layer_token_embd")
                .dequantize_elements(token as usize * ctxdev.ple_total, ctxdev.ple_total)?;
            s.memcpy_htod(&ti, &mut ctxdev.ti).map_err(cu)?;
            crate::cuda_resident::launch_f32_gemv(
                &s,
                &self.kernels.f32_gemv,
                &ctxdev.proj,
                &self.d_hidden,
                &mut self.d_pli,
                hidden,
                ctxdev.ple_total,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut self.d_pli,
                ctxdev.ple_total,
                ctxdev.proj_scale,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_rms_norm_per_head(
                &s,
                &self.kernels.rms_norm_per_head,
                &mut self.d_pli,
                &ctxdev.proj_norm,
                self.block_count,
                ple_dim,
                eps,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut ctxdev.ti,
                ctxdev.ple_total,
                ctxdev.embed_scale,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_residual(
                &s,
                &self.kernels.residual_add,
                &mut self.d_pli,
                &ctxdev.ti,
                ctxdev.ple_total,
            )
            .map_err(cu)?;
            crate::cuda_resident::launch_scale(
                &s,
                &self.kernels.scale_f32,
                &mut self.d_pli,
                ctxdev.ple_total,
                std::f32::consts::FRAC_1_SQRT_2,
            )
            .map_err(cu)?;
        } else if let (Some(te), Some(proj), Some(pn)) = (
            self.cpu.per_layer_token_embd.as_ref(),
            self.cpu.per_layer_model_proj.as_ref(),
            self.cpu.per_layer_proj_norm.as_ref(),
        ) {
            let ti = te.dequantize_elements(token as usize * ple_total, ple_total)?;
            let ctx = f32_matvec(&proj[0..ple_total * hidden], hidden, ple_total, &h);
            let proj_scale = (hidden as f32).powf(-0.5);
            let ple_embed_scale = (ple_dim as f32).sqrt();
            let pli_flat: Vec<f32> = (0..self.block_count)
                .flat_map(|li| {
                    let ctx_l: Vec<f32> = (0..ple_dim)
                        .map(|d| ctx[li * ple_dim + d] * proj_scale)
                        .collect();
                    let ctx_n = rms_norm(&ctx_l, Some(pn), eps);
                    (0..ple_dim)
                        .map(|d| {
                            (ctx_n[d] + ti[li * ple_dim + d] * ple_embed_scale)
                                * std::f32::consts::FRAC_1_SQRT_2
                        })
                        .collect::<Vec<f32>>()
                })
                .collect();
            s.memcpy_htod(&pli_flat, &mut self.d_pli).map_err(cu)?;
        }
        // Precompute every layer's RoPE table for this position (slot li = li*half_max)
        // and upload once — so the per-layer loop has no in-loop memcpy (graph-capturable).
        {
            let half_max = self.half_max;
            let mut cos_all = vec![0f32; self.block_count * half_max];
            let mut sin_all = vec![0f32; self.block_count * half_max];
            for li in 0..self.block_count {
                let p = &self.plan[li];
                let hd = p.head_dim;
                let half = hd / 2;
                let theta = p.theta;
                let factors = if p.sliding {
                    None
                } else {
                    self.cpu.rope_factors.as_deref()
                };
                let base = li * half_max;
                for i in 0..half {
                    let mut freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                    if let Some(f) = factors {
                        freq /= f[i];
                    }
                    let (sn, cs) = (position as f32 * freq).sin_cos();
                    cos_all[base + i] = cs;
                    sin_all[base + i] = sn;
                }
            }
            s.memcpy_htod(&cos_all, &mut self.d_cos_all).map_err(cu)?;
            s.memcpy_htod(&sin_all, &mut self.d_sin_all).map_err(cu)?;
        }
        // Capture the per-token layer stack into a CUDA graph once, then replay it
        // (one launch instead of ~900). The loop reads device buffers only (weights
        // resident; pli/cos/position pre-uploaded above), so it is graph-capturable.
        // Record the graph only AFTER a warmup pass: a kernel's first launch does
        // lazy init (module/function load) which is not stream-capturable. The warmup
        // call runs the loop directly; the next call captures it; later calls replay.
        //
        // MoE (A4B/26B) rows compute their FFN on the CPU via a per-layer device<->host
        // round-trip (synchronize + memcpy). That CANNOT live inside a captured/replayed
        // graph, so disable capture entirely for any model with a MoE layer and always
        // run the explicit per-launch path. (Dense/E-series models keep the graph.)
        let has_moe = self.cpu.layers.iter().any(|lw| lw.moe.is_some());
        let do_capture = !has_moe && self.decode_graph.is_none() && self.warmed;
        if do_capture {
            use cudarc::driver::sys;
            s.begin_capture(sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(cu)?;
        }
        if self.decode_graph.is_none() {
            let k = &self.kernels;
            for li in 0..self.block_count {
                let p = self.plan[li].clone();
                let hd = p.head_dim;
                let half = hd / 2;
                let q_dim = p.q_dim;
                let kv_dim = p.kv_dim;
                let kv_heads = p.kv_heads;
                let ffn_dim = self.cpu.g.ffn_length_at(li) as usize;
                let lw = &self.cpu.layers[li];
                let nrm = &self.norms[li];
                let lwd = &self.lweights[li];

                // attention RMSNorm + Q8_0 quantize of the activation (shared by q/k/v).
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_hidden,
                    &nrm.attn_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_normed,
                    &mut self.d_inq,
                    &mut self.d_ins,
                    hidden / 32,
                )
                .map_err(cu)?;

                // Q projection -> per-head q-norm -> RoPE (split-half, dual-θ).
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.q_q,
                    &self.d_ins,
                    &self.d_inq,
                    &lwd.q.slice(0..lwd.q.len()),
                    q_dim,
                    hidden / 32,
                    &mut self.d_q,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_rms_norm_per_head(
                    &s,
                    &k.rms_norm_per_head,
                    &mut self.d_q,
                    &nrm.q_norm,
                    heads,
                    hd,
                    eps,
                )
                .map_err(cu)?;
                // RoPE q (split-half, dual-θ): read this layer's slot from d_cos_all/d_sin_all
                // (uploaded once before the loop). Inline launch (launch_rope takes &CudaSlice).
                let rope_off = li * self.half_max;
                {
                    let cos_v = self.d_cos_all.slice(rope_off..rope_off + half);
                    let sin_v = self.d_sin_all.slice(rope_off..rope_off + half);
                    let cfg = LaunchConfig {
                        grid_dim: (((heads * half) as u32).div_ceil(128).max(1), 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let (nh, hdi, rd, pr) = (heads as i32, hd as i32, hd as i32, 1i32);
                    let mut b = s.launch_builder(&k.rope);
                    b.arg(&mut self.d_q)
                        .arg(&cos_v)
                        .arg(&sin_v)
                        .arg(&nh)
                        .arg(&hdi)
                        .arg(&rd)
                        .arg(&pr);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }

                // K/V projection + norms + RoPE + cache scatter — owning layers only.
                if p.owns_kv {
                    {
                        let wk = lwd.k.as_ref().expect("owning layer has resident K");
                        gemma_proj_gemv(
                            &s,
                            k,
                            lwd.k_q,
                            &self.d_ins,
                            &self.d_inq,
                            &wk.slice(0..wk.len()),
                            kv_dim,
                            hidden / 32,
                            &mut self.d_k,
                        )
                        .map_err(cu)?;
                        match lwd.v.as_ref() {
                            Some(wv) => {
                                gemma_proj_gemv(
                                    &s,
                                    k,
                                    lwd.v_q,
                                    &self.d_ins,
                                    &self.d_inq,
                                    &wv.slice(0..wv.len()),
                                    kv_dim,
                                    hidden / 32,
                                    &mut self.d_v,
                                )
                                .map_err(cu)?;
                            }
                            // V-less layers: V = K projection.
                            None => {
                                gemma_proj_gemv(
                                    &s,
                                    k,
                                    lwd.k_q,
                                    &self.d_ins,
                                    &self.d_inq,
                                    &wk.slice(0..wk.len()),
                                    kv_dim,
                                    hidden / 32,
                                    &mut self.d_v,
                                )
                                .map_err(cu)?;
                            }
                        }
                    }
                    // k-norm (weighted) and v-norm (weightless), per kv head.
                    crate::cuda_resident::launch_rms_norm_per_head(
                        &s,
                        &k.rms_norm_per_head,
                        &mut self.d_k,
                        nrm.k_norm.as_ref().expect("owning layer binds attn_k_norm"),
                        kv_heads,
                        hd,
                        eps,
                    )
                    .map_err(cu)?;
                    {
                        // weightless V-norm (use_weight=0; weight ptr unused by the kernel).
                        let cfg = LaunchConfig {
                            grid_dim: (kv_heads as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: (hd as u32) * 4,
                        };
                        let (hdi, uw) = (hd as i32, 0i32);
                        let mut b = s.launch_builder(&k.rms_norm_per_head);
                        b.arg(&mut self.d_v)
                            .arg(&nrm.q_norm)
                            .arg(&hdi)
                            .arg(&eps)
                            .arg(&uw);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    {
                        let cos_v = self.d_cos_all.slice(rope_off..rope_off + half);
                        let sin_v = self.d_sin_all.slice(rope_off..rope_off + half);
                        let cfg = LaunchConfig {
                            grid_dim: (((kv_heads * half) as u32).div_ceil(128).max(1), 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let (nh, hdi, rd, pr) = (kv_heads as i32, hd as i32, hd as i32, 1i32);
                        let mut b = s.launch_builder(&k.rope);
                        b.arg(&mut self.d_k)
                            .arg(&cos_v)
                            .arg(&sin_v)
                            .arg(&nh)
                            .arg(&hdi)
                            .arg(&rd)
                            .arg(&pr);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    // Scatter K/V into this layer's cache at `position`.
                    let ck = self.cache_k[li].as_mut().expect("owning layer has K cache");
                    crate::cuda_resident::launch_kv_scatter(
                        &s,
                        &k.kv_scatter,
                        &self.d_k,
                        ck,
                        &self.d_position,
                        kv_heads,
                        hd,
                        self.max_positions,
                    )
                    .map_err(cu)?;
                    let cv = self.cache_v[li].as_mut().expect("owning layer has V cache");
                    crate::cuda_resident::launch_kv_scatter(
                        &s,
                        &k.kv_scatter,
                        &self.d_v,
                        cv,
                        &self.d_position,
                        kv_heads,
                        hd,
                        self.max_positions,
                    )
                    .map_err(cu)?;
                }

                // Attention against the source layer's cache (sliding window or full causal).
                let src = p.kv_source_layer;
                let window = p.window.map(|w| w as i32).unwrap_or(0);
                {
                    let ck = self.cache_k[src].as_ref().expect("KV source has K cache");
                    let cv = self.cache_v[src].as_ref().expect("KV source has V cache");
                    let cfg = LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: ((2 * hd + self.max_positions) as u32) * 4,
                    };
                    let (nh, nkv, hdi, mp) = (
                        heads as i32,
                        kv_heads as i32,
                        hd as i32,
                        self.max_positions as i32,
                    );
                    let scale = 1.0f32; // gemma folds the scale; attention uses no 1/sqrt(d).
                    let mut b = s.launch_builder(&k.attention_sw);
                    b.arg(&self.d_q)
                        .arg(ck)
                        .arg(cv)
                        .arg(&mut self.d_attn)
                        .arg(&nh)
                        .arg(&nkv)
                        .arg(&hdi)
                        .arg(&self.d_position)
                        .arg(&mp)
                        .arg(&scale)
                        .arg(&window);
                    unsafe { b.launch(cfg) }.map_err(cu)?;
                }

                // O projection (quantize attn output, in=q_dim) -> post-attn norm -> residual.
                crate::cuda_resident::launch_quantize(
                    &s,
                    &k.quantize,
                    &self.d_attn,
                    &mut self.d_attnq,
                    &mut self.d_attns,
                    q_dim / 32,
                )
                .map_err(cu)?;
                gemma_proj_gemv(
                    &s,
                    k,
                    lwd.o_q,
                    &self.d_attns,
                    &self.d_attnq,
                    &lwd.o.slice(0..lwd.o.len()),
                    hidden,
                    q_dim / 32,
                    &mut self.d_o,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_rmsnorm(
                    &s,
                    &k.rms_norm,
                    &self.d_o,
                    &nrm.post_attn_norm,
                    &mut self.d_normed,
                    hidden,
                    eps,
                )
                .map_err(cu)?;
                crate::cuda_resident::launch_residual(
                    &s,
                    &k.residual_add,
                    &mut self.d_hidden,
                    &self.d_normed,
                    hidden,
                )
                .map_err(cu)?;

                // FFN. MoE (A4B/26B) rows have TWO branches off the post-attention
                // residual `attn_out` (in `d_hidden`): (A) a dense "shared expert" MLP
                // and (B) the sparse 8-expert branch. With the SSER cache ON (M4) BOTH
                // branches run on the GPU and are composed on-device — the CPU only
                // runs the tiny router. With the cache OFF (M1) the whole two-branch
                // block runs on the CPU via the shared bit-exact `moe_layer_ffn`
                // helper. Either way `d_hidden` must be settled (attention done) before
                // reading it, so synchronize + dtoh first.
                if lw.moe.is_some() && self.sser.is_some() {
                    // --- M4: both MoE branches on the GPU. ---
                    // The router still runs on the CPU, so copy the post-attention
                    // residual to the host once (branch A + the experts read `d_hidden`
                    // on-device; only the top-8 pick needs the host copy).
                    s.synchronize().map_err(cu)?;
                    let mut attn_out_host = vec![0f32; hidden];
                    s.memcpy_dtoh(&self.d_hidden, &mut attn_out_host)
                        .map_err(cu)?;

                    // Branch A — dense shared-expert MLP, GPU (reuses the dense-row
                    // FFN kernels): rms_norm(attn_out, ffn_norm) -> quantize -> gate/up
                    // GEMV -> GeGLU -> quantize -> down GEMV -> rms_norm(_, post_norm_1)
                    // -> d_mlp. Differs from the dense-row path ONLY by post_norm_1 (vs
                    // post_ffw_norm) and by parking the result in d_mlp instead of
                    // folding straight into the residual.
                    let prof = std::env::var_os("CAMELID_SSER_PROFILE").is_some();
                    let ta = std::time::Instant::now();
                    let post_norm_1 = nrm
                        .moe_post_norm_1
                        .as_ref()
                        .expect("MoE layer binds post_norm_1");
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_hidden,
                        &nrm.ffn_norm,
                        &mut self.d_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_quantize(
                        &s,
                        &k.quantize,
                        &self.d_normed,
                        &mut self.d_inq,
                        &mut self.d_ins,
                        hidden / 32,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.gate_q,
                        &self.d_ins,
                        &self.d_inq,
                        &lwd.gate.slice(0..lwd.gate.len()),
                        ffn_dim,
                        hidden / 32,
                        &mut self.d_gate,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.up_q,
                        &self.d_ins,
                        &self.d_inq,
                        &lwd.up.slice(0..lwd.up.len()),
                        ffn_dim,
                        hidden / 32,
                        &mut self.d_up,
                    )
                    .map_err(cu)?;
                    {
                        let cfg = LaunchConfig {
                            grid_dim: ((ffn_dim as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let n_i = ffn_dim as i32;
                        let mut b = s.launch_builder(&k.geglu_mul);
                        b.arg(&self.d_gate)
                            .arg(&self.d_up)
                            .arg(&mut self.d_geglu)
                            .arg(&n_i);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    crate::cuda_resident::launch_quantize(
                        &s,
                        &k.quantize,
                        &self.d_geglu,
                        &mut self.d_geglu_q,
                        &mut self.d_geglu_s,
                        ffn_dim / 32,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.down_q,
                        &self.d_geglu_s,
                        &self.d_geglu_q,
                        &lwd.down.slice(0..lwd.down.len()),
                        hidden,
                        ffn_dim / 32,
                        &mut self.d_ffn_out,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_ffn_out,
                        post_norm_1,
                        &mut self.d_mlp,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    if prof {
                        SSER_PROF_DENSE_NS.fetch_add(
                            ta.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }

                    // Branch B — sparse expert sum on-device (returns d_moe_acc, the
                    // weighted expert accumulation BEFORE post_norm_2). Takes `&self`
                    // (LRU behind a RefCell), so it coexists with the loop-level
                    // `&self.kernels` borrow.
                    let d_moe_acc = self.moe_layer_ffn_cached(li, &attn_out_host)?;

                    // Compose on-device: rms_norm(moe_acc, post_norm_2) -> + d_mlp ->
                    // rms_norm(_, post_ffw_norm) -> add to the residual. Bit-identical
                    // op order to the CPU `moe_layer_ffn` tail.
                    let post_norm_2 = nrm
                        .moe_post_norm_2
                        .as_ref()
                        .expect("MoE layer binds post_norm_2");
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &d_moe_acc,
                        post_norm_2,
                        &mut self.d_ffn_out,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    // d_ffn_out (cur_moe) += d_mlp (dense branch).
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_ffn_out,
                        &self.d_mlp,
                        hidden,
                    )
                    .map_err(cu)?;
                    // rms_norm(combined, post_ffw_norm) -> d_normed -> + residual.
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_ffn_out,
                        &nrm.post_ffw_norm,
                        &mut self.d_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_normed,
                        hidden,
                    )
                    .map_err(cu)?;
                } else if lw.moe.is_some() {
                    // --- M1: whole two-branch MoE FFN on the CPU (cache OFF). ---
                    s.synchronize().map_err(cu)?;
                    let mut attn_out_host = vec![0f32; hidden];
                    s.memcpy_dtoh(&self.d_hidden, &mut attn_out_host)
                        .map_err(cu)?;
                    let ffn_out = self.cpu.moe_layer_ffn(li, &attn_out_host);
                    s.memcpy_htod(&ffn_out, &mut self.d_ffn_out).map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_ffn_out,
                        hidden,
                    )
                    .map_err(cu)?;
                } else {
                    // Dense row: norm + quantize -> gate/up -> GeGLU -> quantize ->
                    // down -> post-ffw norm -> residual, all on the GPU.
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_hidden,
                        &nrm.ffn_norm,
                        &mut self.d_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_quantize(
                        &s,
                        &k.quantize,
                        &self.d_normed,
                        &mut self.d_inq,
                        &mut self.d_ins,
                        hidden / 32,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.gate_q,
                        &self.d_ins,
                        &self.d_inq,
                        &lwd.gate.slice(0..lwd.gate.len()),
                        ffn_dim,
                        hidden / 32,
                        &mut self.d_gate,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.up_q,
                        &self.d_ins,
                        &self.d_inq,
                        &lwd.up.slice(0..lwd.up.len()),
                        ffn_dim,
                        hidden / 32,
                        &mut self.d_up,
                    )
                    .map_err(cu)?;
                    {
                        let cfg = LaunchConfig {
                            grid_dim: ((ffn_dim as u32).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let n_i = ffn_dim as i32;
                        let mut b = s.launch_builder(&k.geglu_mul);
                        b.arg(&self.d_gate)
                            .arg(&self.d_up)
                            .arg(&mut self.d_geglu)
                            .arg(&n_i);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    crate::cuda_resident::launch_quantize(
                        &s,
                        &k.quantize,
                        &self.d_geglu,
                        &mut self.d_geglu_q,
                        &mut self.d_geglu_s,
                        ffn_dim / 32,
                    )
                    .map_err(cu)?;
                    gemma_proj_gemv(
                        &s,
                        k,
                        lwd.down_q,
                        &self.d_geglu_s,
                        &self.d_geglu_q,
                        &lwd.down.slice(0..lwd.down.len()),
                        hidden,
                        ffn_dim / 32,
                        &mut self.d_ffn_out,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_ffn_out,
                        &nrm.post_ffw_norm,
                        &mut self.d_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_normed,
                        hidden,
                    )
                    .map_err(cu)?;
                }

                // PLE injection on the GPU (no host round-trip): gated = inp_gate·h ->
                // gelu_tanh(gated)*pli[li] -> proj·gated -> post_norm -> residual -> output_scale.
                if let Some(pd) = self.ple[li].as_ref() {
                    crate::cuda_resident::launch_f32_gemv(
                        &s,
                        &k.f32_gemv,
                        &pd.inp_gate,
                        &self.d_hidden,
                        &mut self.d_ple_gated,
                        hidden,
                        ple_dim,
                    )
                    .map_err(cu)?;
                    {
                        let off = li * ple_dim;
                        let pli_view = self.d_pli.slice(off..off + ple_dim);
                        let cfg = LaunchConfig {
                            grid_dim: ((ple_dim as u32).div_ceil(256).max(1), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let n_i = ple_dim as i32;
                        let mut b = s.launch_builder(&k.geglu_mul);
                        b.arg(&self.d_ple_gated)
                            .arg(&pli_view)
                            .arg(&mut self.d_ple_gated2)
                            .arg(&n_i);
                        unsafe { b.launch(cfg) }.map_err(cu)?;
                    }
                    crate::cuda_resident::launch_f32_gemv(
                        &s,
                        &k.f32_gemv,
                        &pd.proj,
                        &self.d_ple_gated2,
                        &mut self.d_ple_proj,
                        ple_dim,
                        hidden,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_rmsnorm(
                        &s,
                        &k.rms_norm,
                        &self.d_ple_proj,
                        &pd.post_norm,
                        &mut self.d_ple_normed,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_residual(
                        &s,
                        &k.residual_add,
                        &mut self.d_hidden,
                        &self.d_ple_normed,
                        hidden,
                    )
                    .map_err(cu)?;
                    if pd.output_scale != 1.0 {
                        crate::cuda_resident::launch_scale(
                            &s,
                            &k.scale_f32,
                            &mut self.d_hidden,
                            hidden,
                            pd.output_scale,
                        )
                        .map_err(cu)?;
                    }
                } else if lw.ple_output_scale != 1.0 {
                    crate::cuda_resident::launch_scale(
                        &s,
                        &k.scale_f32,
                        &mut self.d_hidden,
                        hidden,
                        lw.ple_output_scale,
                    )
                    .map_err(cu)?;
                }
            }
        }
        if do_capture {
            use cudarc::driver::sys;
            // Use a real enum variant (not transmute(0): the flags enum has no zero
            // variant, which trips the debug enum-validity check). USE_NODE_PRIORITY is
            // a no-op here (no node priorities are set), so instantiation is plain; the
            // graph is pre-uploaded explicitly via `g.upload()` below.
            let flags =
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
            match s.end_capture(flags).map_err(cu)? {
                Some(g) => {
                    g.upload().map_err(cu)?;
                    self.decode_graph = Some(SendGraph(g));
                }
                None => {
                    return Err(BackendError::InvalidModelMetadata(
                        "gemma4 cuda: decode graph capture produced no graph".into(),
                    ))
                }
            }
        }
        self.warmed = true;
        // Replay the captured graph when present. On the warmup call there is no graph
        // yet and the loop above already executed directly, so we skip the launch.
        if let Some(g) = self.decode_graph.as_ref() {
            g.0.launch().map_err(cu)?;
        }

        // Prefill tokens except the last only need their KV populated, not logits — skip
        // the ~10ms vocab head. The layers/graph already wrote KV on the capture stream,
        // and the next token's upload (a synchronous memcpy) orders after it, so no sync
        // is needed here.
        if !want_logits {
            return Ok(Vec::new());
        }

        // ---- Final norm + tied head + soft-cap. ----
        if let Some(head) = self.gpu_head.as_mut() {
            // GPU Q6_K head: fused rms_norm+Q8K-quant -> q6k_gemv over the vocab ->
            // soft-cap, on the capture stream; only the logits are copied back. This
            // replaces the ~1.2 s/token CPU Q6_K matvec that dominates decode.
            let wlen = head.weight.len();
            match head.lane {
                HeadLane::Q8_0 => {
                    crate::cuda_resident::launch_rmsnorm_quantize(
                        &s,
                        &self.kernels.rms_norm_quantize,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_gemv(
                        &s,
                        &self.kernels.gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                    )
                    .map_err(cu)?;
                }
                HeadLane::Q6K => {
                    crate::cuda_resident::launch_rmsnorm_quantize_q8k(
                        &s,
                        &self.kernels.rms_norm_quantize_q8k,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_q6k_gemv(
                        &s,
                        &self.kernels.q6k_gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                        0,
                    )
                    .map_err(cu)?;
                }
                HeadLane::Q4K => {
                    crate::cuda_resident::launch_rmsnorm_quantize_q8k(
                        &s,
                        &self.kernels.rms_norm_quantize_q8k,
                        &self.d_hidden,
                        &head.output_norm,
                        &mut head.inq,
                        &mut head.ins,
                        hidden,
                        eps,
                    )
                    .map_err(cu)?;
                    crate::cuda_resident::launch_q4k_gemv(
                        &s,
                        &self.kernels.q4k_gemv,
                        &head.ins,
                        &head.inq,
                        &head.weight.slice(0..wlen),
                        self.vocab,
                        head.blocks,
                        &mut head.logits,
                        0,
                    )
                    .map_err(cu)?;
                }
            }
            if head.softcap != 0.0 {
                let cfg = LaunchConfig {
                    grid_dim: ((self.vocab as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let (n_i, cap) = (self.vocab as i32, head.softcap);
                let mut b = s.launch_builder(&self.kernels.soft_cap);
                b.arg(&mut head.logits).arg(&n_i).arg(&cap);
                unsafe { b.launch(cfg) }.map_err(cu)?;
            }
            s.synchronize().map_err(cu)?;
            let mut logits = vec![0f32; self.vocab];
            s.memcpy_dtoh(&head.logits, &mut logits).map_err(cu)?;
            return Ok(logits);
        }
        // CPU head fallback (non-Q6_K head): final norm + tied matvec + soft-cap.
        s.synchronize().map_err(cu)?;
        let mut last = vec![0f32; hidden];
        s.memcpy_dtoh(&self.d_hidden, &mut last).map_err(cu)?;
        let normed = rms_norm(&last, Some(&self.cpu.output_norm), eps);
        let mut logits = self.cpu.token_embd.matvec(hidden, self.vocab, &normed);
        if let Some(cap) = self.cpu.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        Ok(logits)
    }

    /// Greedy-generate up to `max_new` tokens (mirrors the Metal runtime loop).
    /// Prefill `prompt_tokens`, reusing the longest prefix already present in the KV cache
    /// from the previous request (cross-request prefix cache) and only running
    /// `forward_token` for the new suffix. Returns the logits predicting the first new
    /// token. Output-equivalent to a full re-prefill: the KV for shared-prefix positions is
    /// identical (same tokens, same positions), so only redundant compute is skipped. The
    /// caller extends `cached_tokens` with any tokens it then generates. Disable with
    /// `CAMELID_GEMMA4_NO_PREFIX_CACHE=1`.
    fn prefill_reusing_cache(&mut self, prompt_tokens: &[u32]) -> Result<Vec<f32>> {
        let n = prompt_tokens.len();
        debug_assert!(n >= 1);
        // Hard cap: the prompt must leave at least one slot for a generated token, and the
        // KV cache is bounded by `max_positions`. Without this, prefilling past the cache
        // overflowed it and the generation silently produced nothing.
        if n >= self.max_positions {
            return Err(BackendError::InvalidModelMetadata(format!(
                "conversation is {n} tokens, which exceeds the gemma4 {}-token context \
                 window — please start a new chat",
                self.max_positions
            )));
        }
        let disabled = std::env::var("CAMELID_GEMMA4_NO_PREFIX_CACHE").is_ok_and(|v| v == "1");
        let mut p = 0usize;
        if !disabled {
            let cap = self.max_positions.min(n);
            while p < cap
                && p < self.cached_tokens.len()
                && prompt_tokens[p] == self.cached_tokens[p]
            {
                p += 1;
            }
        }
        // Always run at least the final prompt token to produce its logits.
        let start = p.min(n - 1);
        let last = n - 1;
        let mut logits = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for pos in start..n {
            logits = self.forward_token(prompt_tokens[pos], pos, pos == last)?;
        }
        self.cached_tokens.clear();
        self.cached_tokens.extend_from_slice(prompt_tokens);
        Ok(logits)
    }

    pub fn generate_greedy(&mut self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            logits = self.forward_token(next, pos, true)?;
        }
        // The cache now also holds the generated tokens — record them so the next request
        // can reuse this turn's full sequence as a prefix.
        self.cached_tokens.extend_from_slice(&generated);
        let text = self.cpu.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Greedy generate returning per-decode-token wall-clock times (seconds), for the
    /// SSER warm-up-curve measurement. `per_token[i]` is the time to produce
    /// `generated[i]` (the forward that emitted the NEXT logits), excluding prefill.
    pub fn generate_greedy_timed(
        &mut self,
        prompt: &str,
        max_new: usize,
    ) -> Result<(String, Vec<u32>, Vec<f64>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let mut per_token = Vec::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            let t = std::time::Instant::now();
            logits = self.forward_token(next, pos, true)?;
            per_token.push(t.elapsed().as_secs_f64());
        }
        self.cached_tokens.extend_from_slice(&generated);
        let text = self.cpu.tokenizer.decode(&generated, true)?;
        Ok((text, generated, per_token))
    }

    /// Greedy-generate emitting a per-token text delta (for SSE streaming): after
    /// each token the full output is re-decoded and the new suffix is handed to
    /// `on_delta` (robust to tokenizer spacing).
    pub fn generate_greedy_streaming<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
    ) -> Result<(String, Vec<u32>)> {
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        let mut generated = Vec::new();
        let mut prev_text = String::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            if eot.contains(&next) {
                break;
            }
            generated.push(next);
            let text = self.cpu.tokenizer.decode(&generated, true)?;
            if text.len() > prev_text.len() {
                on_delta(&text[prev_text.len()..]);
            }
            prev_text = text;
            logits = self.forward_token(next, pos, true)?;
        }
        self.cached_tokens.extend_from_slice(&generated);
        Ok((prev_text, generated))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_parity_tests {
    use super::*;

    // Greedy parity: the CUDA gemma4 forward must match the CPU Gemma4Runtime oracle
    // token-for-token on the E4B Q8_0 file (the oracle that the CPU runtime loads).
    // Weights stream from host per layer, so it fits the 6 GB card; kept short.
    #[test]
    #[ignore = "requires a CUDA device + the gemma4 E4B Q8_0 model"]
    fn gemma4_cuda_matches_cpu_greedy() {
        let path_s = match std::env::var("CAMELID_GEMMA4_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: set CAMELID_GEMMA4_GGUF to the gemma4 E4B Q8_0 gguf path");
                return;
            }
        };
        let path = std::path::Path::new(&path_s);
        if !path.exists() {
            eprintln!("skip: gemma4 model not found at {path_s}");
            return;
        }
        let prompt = "The capital of France is";
        let cpu = Gemma4Runtime::load(path).expect("cpu load");
        let (cpu_text, cpu_ids) = cpu.generate_greedy(prompt, 8).expect("cpu gen");
        let mut gpu = Gemma4CudaResident::load(path, 2048).expect("gpu load");
        let t0 = std::time::Instant::now();
        let (gpu_text, gpu_ids) = gpu.generate_greedy(prompt, 24).expect("gpu gen");
        let secs = t0.elapsed().as_secs_f64();
        eprintln!("CPU ids[..8] {cpu_ids:?} -> {cpu_text:?}");
        eprintln!("GPU ids       {gpu_ids:?} -> {gpu_text:?}");
        eprintln!(
            "GPU decode: {} tokens in {:.1}s = {:.2} tok/s",
            gpu_ids.len(),
            secs,
            gpu_ids.len() as f64 / secs.max(1e-9)
        );
        // Greedy-parity gate: the CUDA decode must match the CPU oracle's DETERMINISTIC
        // next-token argmax (the gemma4 lane's argmax-stability guarantee). Every
        // projection kernel is bit-exact vs its CPU oracle (q8/q4_0/q4_1/q4k/q6k unit
        // tests), but the attention online-softmax, PLE gelu (CUDA tanhf) and norm
        // reductions are fp-reassociated, so on coarse quant (Q4) a logit near-tie can
        // flip a LATER token — divergence past the first token is allowed. The shared
        // prefix length is logged so a deeper regression is still visible.
        let common = gpu_ids
            .iter()
            .zip(&cpu_ids)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "CPU/GPU greedy common prefix: {common}/{} tokens",
            cpu_ids.len()
        );
        assert_eq!(
            gpu_ids.first(),
            cpu_ids.first(),
            "gemma4 CUDA first-token argmax diverged from the CPU oracle"
        );
    }
}

#[cfg(test)]
mod q4_0_cpu_tests {
    use super::*;

    // Phase 1 gate (mission C): the CPU oracle must LOAD the mixed-quant Q4_0 file
    // (Q4_0 + Q4_1 ffn_down + Q4_K tied head + Q5_K per_layer_token_embd + BF16 proj)
    // and generate coherent greedy text. Set CAMELID_GEMMA4_Q4_GGUF to the file.
    #[test]
    #[ignore = "set CAMELID_GEMMA4_Q4_GGUF to the mixed Q4_0 gemma4 gguf"]
    fn cpu_loads_and_decodes_mixed_q4_0() {
        let path = match std::env::var("CAMELID_GEMMA4_Q4_GGUF") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: set CAMELID_GEMMA4_Q4_GGUF");
                return;
            }
        };
        let cpu = Gemma4Runtime::load(std::path::Path::new(&path)).expect("load mixed Q4_0");
        let (text, ids) = cpu
            .generate_greedy("The capital of France is", 16)
            .expect("cpu generate");
        eprintln!("Q4_0 CPU ids:  {ids:?}");
        eprintln!("Q4_0 CPU text: {text:?}");
        assert!(!ids.is_empty(), "generated no tokens");
    }
}

/// BASALT Phase 3: the gemma4 wire lane's NVFP4 seam — WireFormat constants,
/// WireQuant admission (incl. the D17/T5 sentinel refusal), and matvec/matmul
/// consistency with `nvfp4_wire_row_dot`. Fixture-anchored + deterministic; no
/// model loads (a bare temp file of wire bytes plus a hand-built descriptor is
/// all `WireQuant::new` consumes).
#[cfg(test)]
mod nvfp4_wire_tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufTensorDescriptor};

    /// Deterministic non-sentinel NVFP4 wire blocks: UE4M3 scale bytes drawn
    /// from a fixed safe set (0x00 zero through 0x7E max-normal; never
    /// 0x7F/0xFF), qs bytes from a small LCG-ish pattern.
    pub(super) fn synth_wire(superblocks: usize) -> Vec<u8> {
        const SAFE_SCALES: [u8; 8] = [0x00, 0x10, 0x2C, 0x38, 0x40, 0x51, 0x66, 0x7E];
        let mut wire = Vec::with_capacity(superblocks * 36);
        for b in 0..superblocks {
            for s in 0..4 {
                wire.push(SAFE_SCALES[(b + s) % SAFE_SCALES.len()]);
            }
            for j in 0..32 {
                wire.push(((b * 37 + j * 11 + 5) % 256) as u8);
            }
        }
        wire
    }

    pub(super) fn desc(
        name: &str,
        tensor_type: GgufTensorType,
        dims: &[u64],
        n_bytes: u64,
    ) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: dims.to_vec(),
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes,
        }
    }

    #[test]
    fn sidecar_check_refuses_nvfp4_with_scale_tensors() {
        // ModelOpt-converted shape: NVFP4 weight + its sidecar `.scale` /
        // `.input_scale` F32 tensors — the wire lane must refuse (D-B2).
        for sidecar_name in ["blk.0.attn_q.scale", "blk.0.attn_q.input_scale"] {
            let tensors = vec![
                desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 4], 144),
                desc(sidecar_name, GgufTensorType::F32, &[1], 4),
            ];
            let err = nvfp4_sidecar_check(&tensors).expect_err("sidecar must refuse");
            let msg = err.to_string();
            assert!(msg.contains(sidecar_name), "{msg}");
            assert!(msg.contains("D-B2"), "{msg}");
        }
    }

    #[test]
    fn sidecar_check_admits_pilot_shapes() {
        // The pilot's real `layer_output_scale.weight` name must NOT false-positive,
        // and sidecar-suffixed names without any NVFP4 tensor are out of scope.
        let pilot = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 4], 144),
            desc(
                "blk.0.layer_output_scale.weight",
                GgufTensorType::F32,
                &[1, 4],
                16,
            ),
        ];
        nvfp4_sidecar_check(&pilot).expect("pilot shape admits");

        let no_nvfp4 = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::Q8_0, &[64, 4], 136),
            desc("blk.0.attn_q.scale", GgufTensorType::F32, &[1], 4),
        ];
        nvfp4_sidecar_check(&no_nvfp4).expect("no NVFP4 -> check is out of scope");
    }

    /// Write `wire` to a temp file and wrap it in the two inputs WireQuant::new
    /// takes. The returned NamedTempFile keeps the mapping's backing file alive.
    pub(super) fn fixture(
        wire: &[u8],
        descs: Vec<GgufTensorDescriptor>,
    ) -> (tempfile::NamedTempFile, TensorStore, Arc<GgufWireMmap>) {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().expect("temp wire file");
        f.write_all(wire).expect("write wire bytes");
        f.flush().expect("flush wire bytes");
        let gguf = GgufFile {
            path: f.path().to_path_buf(),
            version: 3,
            tensor_count: descs.len() as i64,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::new(),
            tensors: descs,
        };
        let store = TensorStore::open(f.path(), &gguf);
        let mmap = GgufWireMmap::map(f.path()).expect("map wire file");
        (f, store, mmap)
    }

    #[test]
    fn wire_format_nvfp4_constants() {
        assert_eq!(WireFormat::Nvfp4.values_per_block(), 64);
        assert_eq!(WireFormat::Nvfp4.bytes_per_block(), 36);
        // 4 Q8_0 activation blocks = 128 values = 2 NVFP4 superblocks = 72 B...
        assert_eq!(WireFormat::Nvfp4.row_bytes_for_q8_blocks(4), 72);
        // ...while the 32-value formats keep their 1:1 block mapping.
        assert_eq!(WireFormat::Q8_0.row_bytes_for_q8_blocks(4), 4 * 34);
        assert_eq!(WireFormat::Q4_0.row_bytes_for_q8_blocks(4), 4 * 18);
    }

    #[test]
    fn wire_quant_new_admits_nvfp4_and_still_refuses_uncovered() {
        // 2 superblocks = 128 elements = 72 wire bytes, dims [64, 2].
        let wire = synth_wire(2);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![
                desc("blk.0.attn_q.weight", GgufTensorType::NVFP4, &[64, 2], 72),
                desc("blk.0.attn_k.weight", GgufTensorType::BF16, &[64, 2], 256),
            ],
        );

        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("NVFP4 admits");
        assert_eq!(wq.format, WireFormat::Nvfp4);
        assert_eq!(wq.element_count, 128);
        assert_eq!(wq.bytes(), &wire[..]);

        // An uncovered type keeps the fail-closed refusal, now naming NVFP4 as covered.
        // (WireQuant holds an Arc<GgufWireMmap> and derives no Debug, so match
        // instead of expect_err.)
        match WireQuant::new(&store, &mmap, "blk.0.attn_k.weight") {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("gemma4 wire load supports"), "msg: {msg}");
                assert!(msg.contains("NVFP4"), "covered list names NVFP4: {msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("BF16 must stay refused"),
        }
    }

    #[test]
    fn wire_quant_new_refuses_nan_sentinel_scale_bytes() {
        for sentinel in [0x7Fu8, 0xFFu8] {
            let mut wire = synth_wire(2);
            wire[36 + 2] = sentinel; // block 1, d[2]
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.ffn_up.weight",
                    GgufTensorType::NVFP4,
                    &[64, 2],
                    72,
                )],
            );
            match WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight") {
                Err(BackendError::InvalidTensorData(msg)) => {
                    assert!(msg.contains("NaN-sentinel"), "msg: {msg}");
                    assert!(msg.contains("block 1"), "first offending block: {msg}");
                }
                Err(other) => panic!("expected InvalidTensorData, got {other:?}"),
                Ok(_) => panic!("sentinel scale byte must refuse at load (D17/T5)"),
            }
        }
        // Zero scales admit (D17/T5: only the sentinel bytes refuse).
        let mut wire = synth_wire(2);
        for b in 0..2 {
            for s in 0..4 {
                wire[b * 36 + s] = 0x00;
            }
        }
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.ffn_up.weight",
                GgufTensorType::NVFP4,
                &[64, 2],
                72,
            )],
        );
        WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight").expect("zero scales admit");
    }

    #[test]
    fn metal_sentinel_check_refuses_nan_sentinel_nvfp4() {
        // GABBRO M3-followup: the Metal resident lane now RUNS NVFP4, reading wire raw
        // (bypassing WireQuant's scan), so nvfp4_metal_sentinel_check is the T5 guard —
        // a NaN-sentinel UE4M3 scale byte refuses at load, naming the tensor.
        for sentinel in [0x7Fu8, 0xFFu8] {
            let mut wire = synth_wire(2);
            wire[36 + 2] = sentinel; // block 1, d[2]
            let descs = vec![desc(
                "blk.7.ffn_down.weight",
                GgufTensorType::NVFP4,
                &[64, 2],
                72,
            )];
            let (_f, _store, mmap) = fixture(&wire, descs.clone());
            match nvfp4_metal_sentinel_check(&descs, &mmap) {
                Err(BackendError::InvalidTensorData(msg)) => {
                    assert!(msg.contains("NaN-sentinel"), "msg: {msg}");
                    assert!(
                        msg.contains("blk.7.ffn_down.weight"),
                        "names the tensor: {msg}"
                    );
                }
                Err(other) => panic!("expected InvalidTensorData, got {other:?}"),
                Ok(()) => panic!("sentinel-bearing NVFP4 must refuse on the Metal lane (D17/T5)"),
            }
        }
    }

    #[test]
    fn metal_sentinel_check_admits_clean_nvfp4_and_non_nvfp4() {
        // Clean NVFP4 admits (the lane runs it now); files without NVFP4 are out of scope.
        let wire = synth_wire(2);
        let descs = vec![desc(
            "blk.0.attn_q.weight",
            GgufTensorType::NVFP4,
            &[64, 2],
            72,
        )];
        let (_f, _store, mmap) = fixture(&wire, descs.clone());
        nvfp4_metal_sentinel_check(&descs, &mmap).expect("clean NVFP4 admits on the Metal lane");

        let descs2 = vec![desc(
            "blk.0.attn_q.weight",
            GgufTensorType::Q8_0,
            &[64, 2],
            136,
        )];
        let (_f2, _store2, mmap2) = fixture(&synth_wire(2), descs2.clone());
        nvfp4_metal_sentinel_check(&descs2, &mmap2).expect("non-NVFP4 files keep loading");
    }

    #[test]
    fn nvfp4_matvec_and_matmul_match_row_dot() {
        // 4 output rows x 128 inputs = 8 superblocks of wire.
        let (in_dim, out_dim) = (128usize, 4usize);
        let wire = synth_wire(8);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.attn_q.weight",
                GgufTensorType::NVFP4,
                &[in_dim as u64, out_dim as u64],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("load");

        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let xq = quantize_q8_0_blocks(&x);
        let row_bytes = WireFormat::Nvfp4.row_bytes_for_q8_blocks(xq.len());
        assert_eq!(row_bytes, 2 * 36, "two superblocks per 128-value row");

        // matvec (public dispatch) must equal the row dot on each wire row, bitwise.
        let out = wq.matvec(in_dim, out_dim, &x);
        for o in 0..out_dim {
            let want = nvfp4_wire_row_dot(&wire[o * row_bytes..(o + 1) * row_bytes], &xq);
            assert_eq!(
                out[o].to_bits(),
                want.to_bits(),
                "matvec row {o}: got {} want {want}",
                out[o]
            );
        }

        // matvec_q_rows: a row band must land on the same dots.
        let rows = wq.matvec_q_rows(1, 2, &xq);
        for (i, o) in (1..3).enumerate() {
            assert_eq!(rows[i].to_bits(), out[o].to_bits(), "row band offset {o}");
        }

        // Batched matmul_q over K activations == matvec_q per activation, bitwise
        // (the spec-verify shared-weight-read contract).
        let xs: Vec<Vec<f32>> = (0..3)
            .map(|k| {
                (0..in_dim)
                    .map(|i| ((i as f32) * 0.11 + k as f32 * 0.7).cos() * 2.0)
                    .collect()
            })
            .collect();
        let xqs: Vec<Vec<Q8_0Block>> = xs.iter().map(|x| quantize_q8_0_blocks(x)).collect();
        let batched = wq.matmul_q(out_dim, &xqs);
        for (k, xq) in xqs.iter().enumerate() {
            let single = wq.matvec_q(out_dim, xq);
            for o in 0..out_dim {
                assert_eq!(
                    batched[k][o].to_bits(),
                    single[o].to_bits(),
                    "matmul_q[{k}][{o}] != matvec_q"
                );
            }
        }
    }
}

/// BASALT Phase 3 SHA_E3 (§3 freeze-move crash fix) — K-quant LAYER-PROJECTION
/// routing. The per-layer projection call sites used to pre-quantize the shared
/// activation to Q8_0 and call `matvec_q` directly, which has no K-quant arms:
/// any gemma4 file with Q4_K/Q5_K/Q6_K projection matmuls panicked
/// `unreachable!` at forward time (latent pre-BASALT; probe-proven on the
/// campaign's Q4K-mm row). These tests pin the fixed dispatch three ways:
/// (1) K-quant projections route through the Q8_K family and land bit-equal to
/// the top-level [`WireQuant::matvec`] — the pre-existing, correct route — and
/// to the raw wire row dots; (2) the Q8_0-family dispatch stays byte-identical
/// to the direct Q8_0-activation path it replaced (NVFP4 non-disturbance at the
/// unit seam); (3) Q5_K matvec roles and Q4_1 gathers refuse TYPED, never panic
/// (invariant I-unknown-type, L2).
#[cfg(test)]
mod kquant_projection_tests {
    use super::nvfp4_wire_tests::{desc, fixture, synth_wire};
    use super::*;
    use crate::inference::{
        Q4_K_WIRE_BYTES_PER_BLOCK, Q5_K_WIRE_BYTES_PER_BLOCK, Q6_K_WIRE_BYTES_PER_BLOCK,
    };

    /// Deterministic K-quant wire: LCG byte fill, then tame f16 scale fields
    /// (per-block byte offsets in `f16_offs`) so no block scale is inf/NaN —
    /// the same recipe as the inference-layer K-quant dot tests.
    fn synth_kquant_wire(blocks: usize, bytes_per_block: usize, f16_offs: &[usize]) -> Vec<u8> {
        let mut wire = vec![0u8; blocks * bytes_per_block];
        for (i, b) in wire.iter_mut().enumerate() {
            *b = ((i * 131 + 17) % 256) as u8;
        }
        for blk in wire.chunks_exact_mut(bytes_per_block) {
            for (j, &off) in f16_offs.iter().enumerate() {
                let v = if j == 0 { 0.0173f32 } else { 0.0049 };
                blk[off..off + 2].copy_from_slice(&crate::tensor::f32_to_f16_bits(v).to_le_bytes());
            }
        }
        wire
    }

    fn activation(in_dim: usize, seed: f32) -> Vec<f32> {
        (0..in_dim)
            .map(|i| ((i as f32) * 0.37 + seed).sin() * 3.0)
            .collect()
    }

    #[test]
    fn kquant_projection_dispatch_matches_top_level_matvec_bitwise() {
        // 5 output rows x 512 inputs = 2 superblocks per row. Oracle #1 is the
        // top-level `matvec` (the route that was always correct for K-quants);
        // oracle #2 is the raw wire row dot on the same bytes.
        let (in_dim, out_dim) = (512usize, 5usize);
        let blocks_per_row = in_dim / 256;
        for (tt, bb, f16_offs) in [
            (
                GgufTensorType::Q4K,
                Q4_K_WIRE_BYTES_PER_BLOCK,
                vec![0usize, 2],
            ),
            (GgufTensorType::Q6K, Q6_K_WIRE_BYTES_PER_BLOCK, vec![208]),
        ] {
            let wire = synth_kquant_wire(blocks_per_row * out_dim, bb, &f16_offs);
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.attn_q.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("K-quant admits");

            let x = activation(in_dim, 0.0);
            let oracle = wq.matvec(in_dim, out_dim, &x);
            let sa = SharedActivation::new(&x);
            let got = wq.matvec_proj(out_dim, &sa);

            let xq = quantize_q8_k_blocks(&x);
            let row_bytes = blocks_per_row * bb;
            for o in 0..out_dim {
                assert_eq!(
                    got[o].to_bits(),
                    oracle[o].to_bits(),
                    "{tt:?} matvec_proj row {o} != top-level matvec"
                );
                let w_row = &wire[o * row_bytes..(o + 1) * row_bytes];
                let dot = match tt {
                    GgufTensorType::Q4K => q4_k_wire_row_dot(w_row, &xq),
                    _ => q6_k_wire_row_dot(w_row, &xq),
                };
                assert_eq!(
                    got[o].to_bits(),
                    dot.to_bits(),
                    "{tt:?} matvec_proj row {o} != wire row dot"
                );
            }
        }
    }

    #[test]
    fn kquant_batched_and_row_band_projections_match_single_dispatch() {
        // matmul_proj (the spec-verify chunk path) must equal matvec_proj per
        // activation, and matvec_rows_proj (the MoE expert-band path) must
        // land on the corresponding rows of the full matvec — all bitwise.
        let (in_dim, out_dim) = (256usize, 6usize);
        for (tt, bb, f16_offs) in [
            (
                GgufTensorType::Q4K,
                Q4_K_WIRE_BYTES_PER_BLOCK,
                vec![0usize, 2],
            ),
            (GgufTensorType::Q6K, Q6_K_WIRE_BYTES_PER_BLOCK, vec![208]),
        ] {
            let wire = synth_kquant_wire(out_dim, bb, &f16_offs);
            let (_f, store, mmap) = fixture(
                &wire,
                vec![desc(
                    "blk.0.ffn_up.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.ffn_up.weight").expect("K-quant admits");

            let xs: Vec<Vec<f32>> = (0..3).map(|k| activation(in_dim, k as f32 * 0.7)).collect();
            let xb = SharedActivationBatch::new(&xs);
            let batched = wq.matmul_proj(out_dim, &xb);
            for (k, x) in xs.iter().enumerate() {
                let sa = SharedActivation::new(x);
                let single = wq.matvec_proj(out_dim, &sa);
                for o in 0..out_dim {
                    assert_eq!(
                        batched[k][o].to_bits(),
                        single[o].to_bits(),
                        "{tt:?} matmul_proj[{k}][{o}] != matvec_proj"
                    );
                }
            }

            let sa = SharedActivation::new(&xs[0]);
            let full = wq.matvec_proj(out_dim, &sa);
            let band = wq.matvec_rows_proj(2, 3, &sa);
            for (i, o) in (2..5).enumerate() {
                assert_eq!(
                    band[i].to_bits(),
                    full[o].to_bits(),
                    "{tt:?} matvec_rows_proj row band offset {o}"
                );
            }
        }
    }

    #[test]
    fn q8_0_family_dispatch_is_byte_identical_to_the_direct_q8_0_path() {
        // NO behavior change for the matvec_q family (NVFP4/Q8_0 shown here;
        // they share the dispatch arm with Q4_0/Q4_1): the routed calls must
        // equal the pre-fix direct calls on the eagerly-quantized activation.
        // Q8_0: 2 blocks/row of 34 bytes (f16 scale at +0), 4 rows x 64 inputs.
        let (in_dim, out_dim) = (64usize, 4usize);
        let q8_wire =
            synth_kquant_wire((in_dim / 32) * out_dim, Q8_WIRE_BYTES_PER_BLOCK, &[0usize]);
        // NVFP4: the pilot format — 128 inputs = 2 superblocks/row, 4 rows.
        let nv_wire = synth_wire(8);
        let cases: [(GgufTensorType, usize, &[u8]); 2] = [
            (GgufTensorType::Q8_0, in_dim, &q8_wire),
            (GgufTensorType::NVFP4, 128, &nv_wire),
        ];
        for (tt, in_dim, wire) in cases {
            let (_f, store, mmap) = fixture(
                wire,
                vec![desc(
                    "blk.0.attn_q.weight",
                    tt,
                    &[in_dim as u64, out_dim as u64],
                    wire.len() as u64,
                )],
            );
            let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight").expect("admits");

            let xs: Vec<Vec<f32>> = (0..3).map(|k| activation(in_dim, k as f32 * 0.7)).collect();
            let xqs: Vec<Vec<Q8_0Block>> = xs.iter().map(|x| quantize_q8_0_blocks(x)).collect();

            let sa = SharedActivation::new(&xs[0]);
            let via_dispatch = wq.matvec_proj(out_dim, &sa);
            let direct = wq.matvec_q(out_dim, &xqs[0]);
            for o in 0..out_dim {
                assert_eq!(
                    via_dispatch[o].to_bits(),
                    direct[o].to_bits(),
                    "{tt:?} matvec_proj row {o} != direct matvec_q"
                );
            }
            let band_dispatch = wq.matvec_rows_proj(1, 2, &sa);
            let band_direct = wq.matvec_q_rows(1, 2, &xqs[0]);
            for i in 0..2 {
                assert_eq!(
                    band_dispatch[i].to_bits(),
                    band_direct[i].to_bits(),
                    "{tt:?} matvec_rows_proj row {i} != direct matvec_q_rows"
                );
            }
            let xb = SharedActivationBatch::new(&xs);
            let batch_dispatch = wq.matmul_proj(out_dim, &xb);
            let batch_direct = wq.matmul_q(out_dim, &xqs);
            for k in 0..xs.len() {
                for o in 0..out_dim {
                    assert_eq!(
                        batch_dispatch[k][o].to_bits(),
                        batch_direct[k][o].to_bits(),
                        "{tt:?} matmul_proj[{k}][{o}] != direct matmul_q"
                    );
                }
            }
        }
    }

    #[test]
    fn q5k_matvec_roles_refuse_typed_at_load() {
        // Q5_K stays admitted for gather (per_layer_token_embd) but must
        // refuse TYPED in any matvec role — pre-fix it loaded fine and
        // panicked `unreachable!` in the forward pass.
        let wire = synth_kquant_wire(2, Q5_K_WIRE_BYTES_PER_BLOCK, &[0, 2]);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.attn_q.weight",
                GgufTensorType::Q5K,
                &[256, 2],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.attn_q.weight")
            .expect("Q5_K admits for gather roles");
        assert_eq!(wq.format, WireFormat::Q5K);
        wq.dequantize_elements(0, 4)
            .expect("Q5_K gather stays served");
        match wq.require_matvec_capable("blk.0.attn_q.weight") {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("Q5_K"), "{msg}");
                assert!(msg.contains("gather-only"), "{msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("Q5_K must refuse matvec roles at load"),
        }
    }

    #[test]
    fn q4_1_gather_refuses_typed_instead_of_panicking() {
        // The sibling reachable-panic arm swept with the SHA_E3 fix: a Q4_1
        // embedding gather is not wired, so it must be a typed refusal.
        let wire = synth_kquant_wire(2, 20, &[0, 2]);
        let (_f, store, mmap) = fixture(
            &wire,
            vec![desc(
                "blk.0.ffn_down.weight",
                GgufTensorType::Q4_1,
                &[32, 2],
                wire.len() as u64,
            )],
        );
        let wq = WireQuant::new(&store, &mmap, "blk.0.ffn_down.weight").expect("Q4_1 admits");
        match wq.dequantize_elements(0, 4) {
            Err(BackendError::UnsupportedTensorType(msg)) => {
                assert!(msg.contains("Q4_1"), "{msg}");
            }
            Err(other) => panic!("expected UnsupportedTensorType, got {other:?}"),
            Ok(_) => panic!("Q4_1 gather must be a typed refusal"),
        }
    }
}

/// BASALT Amendment 3: the GPU-lane typed refusals and the §9 platform gate.
/// All helpers are cfg-independent, so these run on every host — no CUDA/Metal
/// hardware and no model loads (descriptor lists and raw [`WireFormat`]s only).
#[cfg(test)]
mod gpu_lane_refusal_tests {
    use super::*;
    use crate::gguf::GgufTensorDescriptor;

    fn desc(name: &str, tensor_type: GgufTensorType) -> GgufTensorDescriptor {
        GgufTensorDescriptor {
            name: name.into(),
            dimensions: vec![64, 1],
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 36,
        }
    }

    #[test]
    fn cuda_lane_check_admits_nvfp4_after_the_phase4_lift() {
        // BASALT Phase 4 (G4) inverted the pre-Phase-4 refusal: NVFP4 layer
        // projections now RESIDE on the CUDA lane (nvfp4_gemv), so an NVFP4
        // format in the projection set must ADMIT — a positive control that the
        // Phase-4 lift landed and the old "NVFP4 is Phase 4" refusal is gone.
        // (Regression guard: ratchet R3 requires this flip in the same PR that
        // closes the six L3 open:P4 cells.)
        nvfp4_cuda_lane_check([WireFormat::Q8_0, WireFormat::Nvfp4, WireFormat::Q4_0])
            .expect("NVFP4 projections now admit on the CUDA lane (Phase 4)");
    }

    #[test]
    fn cuda_lane_check_admits_the_supported_projection_formats() {
        // Every format from_wire actually supports must keep loading — Q8_0/Q4_0/
        // Q4_1 (pre-BASALT) plus NVFP4 (Phase 4). I-carveout boundary-preservation:
        // the K-quant refusal must not bleed onto the formats this lane serves.
        nvfp4_cuda_lane_check([
            WireFormat::Q8_0,
            WireFormat::Q4_0,
            WireFormat::Q4_1,
            WireFormat::Nvfp4,
        ])
        .expect("Q8_0/Q4_0/Q4_1/NVFP4 projections stay admitted");
        nvfp4_cuda_lane_check(std::iter::empty()).expect("no projections is vacuously fine");
    }

    #[test]
    fn cuda_lane_check_refuses_every_lane_uncovered_format_typed() {
        // SHA_E review finding #1: the campaign's own K-quant rows (Q4K-mm,
        // Q4_K_M-df/-im) load clean on the CPU wire lane but would hit the
        // from_wire repack panic on the CUDA lane. Every format outside the
        // lane's covered set must refuse TYPED and NAMED — never a panic
        // (invariant I-unknown-type, L3 cell).
        for uncovered in [WireFormat::Q4K, WireFormat::Q5K, WireFormat::Q6K] {
            match nvfp4_cuda_lane_check([WireFormat::Q8_0, uncovered]) {
                Err(BackendError::UnsupportedGguf(msg)) => {
                    assert!(
                        msg.contains(&format!("{uncovered:?}")),
                        "refusal must name the format: {msg}"
                    );
                    assert!(
                        msg.contains("covers Q8_0/Q4_0/Q4_1/NVFP4"),
                        "refusal must name the covered set: {msg}"
                    );
                }
                Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
                Ok(()) => panic!("{uncovered:?} projection must refuse in the CUDA lane"),
            }
        }
    }

    // GABBRO M3-followup: the blanket `nvfp4_metal_lane_check` refusal was lifted (the
    // Metal resident lane now RUNS NVFP4), replaced by `nvfp4_metal_sentinel_check` (the
    // T5 guard). Its tests need real wire bytes, so they live in the fixture-bearing mod:
    // `metal_sentinel_check_refuses_nan_sentinel_nvfp4` and
    // `metal_sentinel_check_admits_clean_nvfp4_and_non_nvfp4`.

    #[test]
    fn metal_layer_fmt_covers_q8_q4_nvfp4_refuses_others() {
        // I-unknown-type (L4): the Metal resident lane covers Q8_0/Q4_0/NVFP4 layer
        // projections; every other format refuses TYPED and NAMED, never a mis-bind.
        use crate::metal::GemmaWireFmt;
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::Q8_0).unwrap(),
            GemmaWireFmt::Q8_0
        );
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::Q4_0).unwrap(),
            GemmaWireFmt::Q4_0
        );
        assert_eq!(
            gemma4_metal_layer_fmt(GgufTensorType::NVFP4).unwrap(),
            GemmaWireFmt::Nvfp4
        );
        for uncovered in [
            GgufTensorType::Q6K,
            GgufTensorType::BF16,
            GgufTensorType::Q4K,
        ] {
            match gemma4_metal_layer_fmt(uncovered) {
                Err(BackendError::UnsupportedTensorType(msg)) => {
                    assert!(
                        msg.contains(&format!("{uncovered:?}")),
                        "names the format: {msg}"
                    );
                    assert!(
                        msg.contains("Q8_0/Q4_0/NVFP4"),
                        "names the covered set: {msg}"
                    );
                }
                other => panic!("uncovered format must refuse typed: {other:?}"),
            }
        }
    }

    #[test]
    fn windows_only_check_ignores_files_without_nvfp4() {
        // Platform-independent: the §9 gate only ever looks at NVFP4-bearing
        // files, so every other row is untouched on every OS.
        let tensors = vec![desc("blk.0.attn_q.weight", GgufTensorType::Q8_0)];
        nvfp4_windows_only_check(&tensors).expect("non-NVFP4 files admit everywhere");
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn windows_only_check_admits_nvfp4_on_windows() {
        // §9 twin (runs on the Windows leg): admission still works where the
        // release actually supports NVFP4.
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        nvfp4_windows_only_check(&tensors).expect("NVFP4 admits on Windows");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn windows_only_check_admits_nvfp4_on_macos() {
        // GABBRO M2 twin (runs on the macOS leg): NVFP4 now admits on macOS too,
        // once the Apple-Silicon CPU decode was proven bit-exact (Gate G-M1).
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        nvfp4_windows_only_check(&tensors).expect("NVFP4 admits on macOS (GABBRO M2)");
    }

    #[test]
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    fn windows_only_check_refuses_nvfp4_off_windows() {
        // §9 twin (runs on the linux leg — macOS now admits, GABBRO M2): the
        // named TK2 refusal on the still-unvalidated platforms.
        let tensors = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        match nvfp4_windows_only_check(&tensors) {
            Err(BackendError::UnsupportedGguf(msg)) => {
                assert_eq!(
                    msg,
                    "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
                );
            }
            Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
            Ok(()) => panic!("NVFP4 must refuse on unvalidated platforms (Amendment 3 §9)"),
        }
    }

    /// BASALT Phase 4 — L3 I-plat (cfg-twinned per §9.1): the shared §9 platform
    /// gate `nvfp4_windows_only_check` fires inside `Gemma4Runtime::load` (via
    /// `load_layer_range`), which is the FIRST act of `Gemma4CudaResident::load`,
    /// so it fronts the CUDA lane's entry before any CUDA initialization. This is
    /// the L3-native twin: the off-Windows legs assert the CUDA lane's shared
    /// entry gate yields the named TK2 refusal (no GPU needed to observe it —
    /// the gate is upstream of every CUDA call); the Windows leg asserts the pilot
    /// shape admits through the gate so the CUDA lane can bind (D-B3 carve-out).
    #[test]
    fn cuda_resident_platform_gate_fronts_the_cuda_lane_entry() {
        let pilot = vec![desc("blk.0.ffn_down.weight", GgufTensorType::NVFP4)];
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        {
            nvfp4_windows_only_check(&pilot).expect(
                "NVFP4 admits through the §9 gate on Windows/macOS so the resident lane binds",
            );
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            match nvfp4_windows_only_check(&pilot) {
                Err(BackendError::UnsupportedGguf(msg)) => assert_eq!(
                    msg, "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX",
                    "the CUDA lane's shared entry gate must yield the named TK2 refusal"
                ),
                Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
                Ok(()) => panic!(
                    "the §9 gate fronting Gemma4CudaResident::load must refuse NVFP4 on unvalidated platforms"
                ),
            }
        }
    }

    /// BASALT Phase 4 — L3 I-sidecar: `Gemma4CudaResident::load`'s first act is
    /// `Gemma4Runtime::load`, whose `nvfp4_sidecar_check` (D-B2) fires before any
    /// CUDA work, so the CUDA lane cannot bind a sidecar-bearing NVFP4 file — it
    /// inherits the shared refusal. Driven here on the shared helper (cfg-
    /// independent, no GPU/model); the end-to-end file trip on the same seam is
    /// `sidecar_fixture_trips_d_b2_end_to_end`. Also asserts the pilot shape (no
    /// sidecar) admits, so the guard the CUDA lane relies on is exact, not blanket.
    #[test]
    fn cuda_resident_load_inherits_shared_sidecar_refusal() {
        let sidecar = vec![
            desc("blk.0.attn_q.weight", GgufTensorType::NVFP4),
            desc("blk.0.attn_q.scale", GgufTensorType::F32),
        ];
        match nvfp4_sidecar_check(&sidecar) {
            Err(BackendError::UnsupportedGguf(msg)) => {
                assert!(
                    msg.contains("blk.0.attn_q.scale"),
                    "names the sidecar: {msg}"
                );
                assert!(msg.contains("D-B2"), "cites D-B2: {msg}");
            }
            Err(other) => panic!("expected UnsupportedGguf, got {other:?}"),
            Ok(()) => panic!("sidecar-bearing NVFP4 must refuse before the CUDA lane binds (D-B2)"),
        }
        let pilot = vec![desc("blk.0.attn_q.weight", GgufTensorType::NVFP4)];
        nvfp4_sidecar_check(&pilot).expect("pilot NVFP4 has no sidecar; the CUDA lane may bind");
    }
}

/// BASALT Amendment 3 review fix #4: the forced-decode step-boundary proof.
/// A scripted fake step fn stands in for the model: `predicted = 1000 + fed`,
/// prompt-end prediction 999 — so every observation uniquely identifies WHICH
/// token had been fed before it, and any off-by-one is unmissable.
#[cfg(test)]
mod forced_step_boundary_tests {
    use super::drive_forced_steps;

    #[test]
    fn observes_before_feeding_and_never_feeds_the_final_token() {
        let forced = [10u32, 20, 30];
        // One interleaved event log proves strict ordering, not just counts.
        // (RefCell: both closures append to the same log.)
        let events = std::cell::RefCell::new(Vec::<String>::new());
        let mut fed: Vec<u32> = Vec::new();
        drive_forced_steps::<u32, std::convert::Infallible>(
            &forced,
            999,
            |tok| {
                fed.push(tok);
                events.borrow_mut().push(format!("fed={tok}"));
                Ok(1000 + tok)
            },
            |i, &pred| events.borrow_mut().push(format!("obs{i}={pred}")),
        )
        .unwrap();
        let events = events.into_inner();

        // Step i's recorded prediction is the state from BEFORE forced[i] was
        // fed: obs0 sees the prompt-end prediction (999), obs1 sees 1000+forced[0],
        // obs2 sees 1000+forced[1]. If the loop fed first and observed second,
        // obs_i would read 1000+forced[i] instead.
        assert_eq!(
            events,
            vec!["obs0=999", "fed=10", "obs1=1010", "fed=20", "obs2=1020"]
        );
        // count == forced.len(): exactly 3 observations fired (asserted above by
        // the full event log), and the FINAL forced token (30) was never fed.
        assert_eq!(fed, vec![10, 20]);
    }

    #[test]
    fn single_forced_token_observes_once_and_feeds_nothing() {
        let mut observed = Vec::new();
        drive_forced_steps::<u32, std::convert::Infallible>(
            &[42],
            7,
            |_| panic!("a single forced token must never be fed"),
            |i, &pred| observed.push((i, pred)),
        )
        .unwrap();
        assert_eq!(observed, vec![(0, 7)]);
    }

    #[test]
    fn empty_forced_list_neither_observes_nor_feeds() {
        // The CLI refuses empty lists upstream; the construct itself is total.
        drive_forced_steps::<u32, std::convert::Infallible>(
            &[],
            0,
            |_| panic!("nothing to feed"),
            |_, _| panic!("nothing to observe"),
        )
        .unwrap();
    }

    #[test]
    fn step_errors_propagate_after_the_boundary_observation() {
        let mut observed = 0usize;
        let err = drive_forced_steps::<u32, &'static str>(
            &[1, 2],
            0,
            |_| Err("step failed"),
            |_, _| observed += 1,
        )
        .unwrap_err();
        assert_eq!(err, "step failed");
        // The step-0 observation (pre-feed) had already fired.
        assert_eq!(observed, 1);
    }
}
