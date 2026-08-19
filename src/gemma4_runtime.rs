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

use crate::gguf::{read_metadata, GgufFile, GgufTensorType};
#[cfg(feature = "cuda")]
use crate::ghost::GhostMoeMappedExpert;
use crate::ghost::{GhostFile, GhostMoeExpert, GhostMoeTensorView};
use crate::inference::gemma4::{gelu_tanh, soft_cap_in_place};
use crate::inference::{
    nvfp4_wire_block_dequant, nvfp4_wire_row_dot, q2_k_wire_row_dot, q4_0_wire_block_dequant,
    q4_0_wire_row_dot, q4_1_wire_row_dot, q4_k_wire_row_dot, q6_k_wire_block_dequant,
    q6_k_wire_row_dot, q8_0_wire_row_dot, quantize_q8_0_blocks, quantize_q8_k_blocks,
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

/// Result of a cooperatively-cancellable Gemma 4 generation.
///
/// Cancellation is not an inference failure: the HTTP owner went away, so the
/// runtime returns the number of tokens it had already committed and releases
/// its KV/expert state at the next forward boundary.  Keeping this distinct
/// from [`BackendError`] lets serving drop a disconnected request quietly while
/// still surfacing genuine model failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gemma4GenerationOutcome {
    Complete { text: String, token_ids: Vec<u32> },
    Cancelled { generated_tokens: usize },
}

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
    /// Q2_K joined the wire lane for the mixed Ghost-MoE expert artifact: the
    /// routed gate_up projections requantized to 2.625 bpw so the whole expert
    /// payload can be host/VRAM-resident on a 6 GiB + 16 GiB box. It rides the
    /// existing Q8_K-activation K-quant family (`q2_k_wire_row_dot`).
    Q2K,
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => 256,
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
            WireFormat::Q2K => crate::tensor::Q2_K_BLOCK_BYTES,
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
#[derive(Clone)]
enum WireBacking {
    Mmap {
        mmap: Arc<GgufWireMmap>,
        offset: u64,
    },
    /// Bounded routed-expert bytes read from a v2 `.cghost` group. `range`
    /// selects one of the two tensors while both share a single cache allocation.
    Owned {
        bytes: Arc<[u8]>,
        range: std::ops::Range<usize>,
    },
}

#[derive(Clone)]
struct WireQuant {
    backing: WireBacking,
    element_count: usize,
    format: WireFormat,
}

impl WireQuant {
    fn format_for_type(tensor_type: GgufTensorType, name: &str) -> Result<WireFormat> {
        match tensor_type {
            GgufTensorType::Q8_0 => Ok(WireFormat::Q8_0),
            GgufTensorType::Q4_0 => Ok(WireFormat::Q4_0),
            GgufTensorType::Q4_1 => Ok(WireFormat::Q4_1),
            GgufTensorType::Q2K => Ok(WireFormat::Q2K),
            GgufTensorType::Q4K => Ok(WireFormat::Q4K),
            GgufTensorType::Q5K => Ok(WireFormat::Q5K),
            GgufTensorType::Q6K => Ok(WireFormat::Q6K),
            GgufTensorType::NVFP4 => Ok(WireFormat::Nvfp4),
            other => Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {other:?}; gemma4 wire load supports Q8_0, Q4_0, Q4_1, Q2_K, Q4_K, Q5_K, Q6_K, and NVFP4"
            ))),
        }
    }

    fn new(store: &TensorStore, mmap: &Arc<GgufWireMmap>, name: &str) -> Result<Self> {
        let desc = store.descriptor(name)?;
        let format = Self::format_for_type(desc.tensor_type, name)?;
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
            backing: WireBacking::Mmap {
                mmap: mmap.clone(),
                offset: desc.absolute_offset,
            },
            element_count,
            format,
        })
    }

    fn from_ghost_tensor(
        expert: &GhostMoeExpert,
        view: &GhostMoeTensorView,
        name: &str,
    ) -> Result<Self> {
        let (bytes, range) = expert.tensor_backing(view);
        Self::from_owned_wire(bytes, range, view.dtype, &view.dims, name)
    }

    fn from_owned_wire(
        bytes: Arc<[u8]>,
        range: std::ops::Range<usize>,
        tensor_type: GgufTensorType,
        dims: &[u64],
        name: &str,
    ) -> Result<Self> {
        let format = Self::format_for_type(tensor_type, name)?;
        let element_count = dims
            .iter()
            .try_fold(1u64, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "ghost expert tensor {name} element count overflows"
                ))
            })? as usize;
        if !element_count.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "ghost expert tensor {name} element count {element_count} is not block-aligned"
            )));
        }
        let expected = element_count / format.values_per_block() * format.bytes_per_block();
        if range.start > range.end || range.end > bytes.len() || range.len() != expected {
            return Err(BackendError::InvalidTensorData(format!(
                "ghost expert tensor {name} has {} wire bytes; expected {expected}",
                range.len()
            )));
        }
        Ok(Self {
            backing: WireBacking::Owned { bytes, range },
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
        match &self.backing {
            WireBacking::Mmap { mmap, offset } => mmap
                .bytes(*offset, byte_len)
                .expect("wire quant range validated at load"),
            WireBacking::Owned { bytes, range } => &bytes[range.clone()],
        }
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q6K => {
                self.matvec_q8k(out_dim, &quantize_q8_k_blocks(x))
            }
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q6K => {
                self.matvec_q8k(out_dim, x.q8_k())
            }
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q6K => {
                self.matmul_q8k(out_dim, xs.q8_k())
            }
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q6K => {
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
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
            WireFormat::Q2K | WireFormat::Q4K | WireFormat::Q5K | WireFormat::Q6K => {
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
            WireFormat::Q2K => q2_k_wire_row_dot,
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
            WireFormat::Q2K => q2_k_wire_row_dot,
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
            WireFormat::Q2K => q2_k_wire_row_dot,
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

    /// Dequantize a contiguous element range [start, start+len) directly into `out`.
    fn dequantize_elements_into(&self, start: usize, len: usize, out: &mut [f32]) -> Result<()> {
        if out.len() < len {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "destination slice length {} too small for dequantize len {}",
                out.len(),
                len
            )));
        }
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
        match self.format {
            WireFormat::Q8_0 => {
                const BV: usize = Q8_VALUES_PER_BLOCK;
                const BB: usize = Q8_WIRE_BYTES_PER_BLOCK;
                for (i, e) in (start..end).enumerate() {
                    let block = e / BV;
                    let within = e % BV;
                    let scale = Self::block_scale(bytes, block);
                    let q = bytes[block * BB + 2 + within] as i8;
                    out[i] = scale * q as f32;
                }
            }
            WireFormat::Q4_0 => {
                const BB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; 32];
                for (i, e) in (start..end).enumerate() {
                    if e / 32 != block {
                        block = e / 32;
                        decoded = q4_0_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out[i] = decoded[e % 32];
                }
            }
            WireFormat::Q6K => {
                const BV: usize = crate::inference::Q6_K_VALUES_PER_BLOCK;
                const BB: usize = crate::inference::Q6_K_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for (i, e) in (start..end).enumerate() {
                    if e / BV != block {
                        block = e / BV;
                        decoded = q6_k_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out[i] = decoded[e % BV];
                }
            }
            WireFormat::Q2K => {
                const BV: usize = 256;
                let bb = self.format.bytes_per_block();
                let mut block = usize::MAX;
                let mut decoded = [0f32; 256];
                for (i, e) in (start..end).enumerate() {
                    if e / BV != block {
                        block = e / BV;
                        let sb: &[u8; crate::tensor::Q2_K_BLOCK_BYTES] = bytes
                            [block * bb..(block + 1) * bb]
                            .try_into()
                            .expect("Q2_K wire block span validated at load");
                        crate::tensor::Q2KBlock::from_bytes(sb).dequantize(&mut decoded);
                    }
                    out[i] = decoded[e % BV];
                }
            }
            WireFormat::Q4K | WireFormat::Q5K => {
                const BV: usize = 256;
                let bb = self.format.bytes_per_block();
                let mut block = usize::MAX;
                let mut decoded: Vec<f32> = Vec::new();
                for (i, e) in (start..end).enumerate() {
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
                    out[i] = decoded[e % BV];
                }
            }
            WireFormat::Q4_1 => {
                return Err(BackendError::UnsupportedTensorType(
                    "gemma4 wire lane cannot gather Q4_1 elements (Q4_1 is a \
                     matvec-only weight format here)"
                        .into(),
                ))
            }
            WireFormat::Nvfp4 => {
                const BV: usize = crate::tensor::NVFP4_VALUES_PER_BLOCK;
                const BB: usize = crate::tensor::NVFP4_WIRE_BYTES_PER_BLOCK;
                let mut block = usize::MAX;
                let mut decoded = [0f32; BV];
                for (i, e) in (start..end).enumerate() {
                    if e / BV != block {
                        block = e / BV;
                        decoded = nvfp4_wire_block_dequant(&bytes[block * BB..(block + 1) * BB]);
                    }
                    out[i] = decoded[e % BV];
                }
            }
        }
        Ok(())
    }

    /// Dequantize a contiguous element range [start, start+len) — used for
    /// row-major embedding lookups into vocab-major Q8 tables.
    fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; len];
        self.dequantize_elements_into(start, len, &mut out)?;
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

#[inline(always)]
pub(crate) fn f32_dot_neon(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        let n = a.len().min(b.len());
        let chunks = n / 16;
        let mut ap = a.as_ptr();
        let mut bp = b.as_ptr();
        for _ in 0..chunks {
            let a0 = vld1q_f32(ap);
            let b0 = vld1q_f32(bp);
            sum0 = vfmaq_f32(sum0, a0, b0);
            let a1 = vld1q_f32(ap.add(4));
            let b1 = vld1q_f32(bp.add(4));
            sum1 = vfmaq_f32(sum1, a1, b1);
            let a2 = vld1q_f32(ap.add(8));
            let b2 = vld1q_f32(bp.add(8));
            sum2 = vfmaq_f32(sum2, a2, b2);
            let a3 = vld1q_f32(ap.add(12));
            let b3 = vld1q_f32(bp.add(12));
            sum3 = vfmaq_f32(sum3, a3, b3);
            ap = ap.add(16);
            bp = bp.add(16);
        }
        let s01 = vaddq_f32(sum0, sum1);
        let s23 = vaddq_f32(sum2, sum3);
        let mut tot = vaddvq_f32(vaddq_f32(s01, s23));
        for i in (chunks * 16)..n {
            tot += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        tot
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}

pub(crate) fn f32_matvec(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32]) -> Vec<f32> {
    // Sequential scalar accumulation, matching the base (oracle-locked)
    // semantics. The 4-accumulator NEON FMA variant (`f32_dot_neon`) changes
    // the float summation order; this function feeds the PLE projection and
    // router logits, where a ULP-level difference flips near-tie argmax /
    // top-8 decisions and derails greedy decoding (measured: single-token
    // flip vs the llama.cpp oracle at position 15). Perf work on this dot
    // must preserve summation order or re-prove token parity end-to-end.
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

/// Observable state of the bounded Ghost-MoE expert cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GhostMoeCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub compulsory_misses: u64,
    pub capacity_misses: u64,
    pub evictions: u64,
    pub bytes_read: u64,
    pub resident_experts: usize,
    pub resident_bytes: usize,
    pub budget_bytes: usize,
}

struct GhostMoeCacheState {
    entries: std::collections::HashMap<(usize, usize), GhostMoeCacheEntry>,
    seen_keys: std::collections::HashSet<(usize, usize)>,
    /// Retained wire bytes per transformer layer. Each layer owns a hard slice
    /// of the global budget, so the layer-major forward pass cannot bulldoze
    /// another layer's hot experts before the next token reaches it.
    layer_bytes: Vec<usize>,
    /// Monotonic access stamp used as the LRU tie-break between equal-frequency
    /// entries. Frequency is periodically aged per layer (see `touch_layer`).
    clock: u64,
    layer_accesses: Vec<u64>,
    bytes: usize,
    hits: u64,
    misses: u64,
    compulsory_misses: u64,
    capacity_misses: u64,
    evictions: u64,
    bytes_read: u64,
}

struct GhostMoeCacheEntry {
    expert: Arc<GhostMoeExpert>,
    frequency: u16,
    last_used: u64,
}

impl GhostMoeCacheState {
    /// Advance one layer's LFU epoch. Aging prevents experts that were hot near
    /// the start of a long conversation from becoming permanently unevictable;
    /// LRU remains the deterministic tie-break after the inexpensive decay.
    fn touch_layer(&mut self, layer: usize) {
        self.clock = self.clock.saturating_add(1);
        let Some(accesses) = self.layer_accesses.get_mut(layer) else {
            return;
        };
        *accesses = accesses.saturating_add(1);
        if !(*accesses).is_multiple_of(256) {
            return;
        }
        for (&(entry_layer, _), entry) in &mut self.entries {
            if entry_layer == layer {
                entry.frequency = (entry.frequency / 2).max(1);
            }
        }
    }

    /// Remove the least-frequently used entry in `layer`, breaking frequency
    /// ties by oldest access. The forward accumulation never depends on this
    /// order; it only chooses which immutable wire record stays resident.
    fn evict_one_from_layer(&mut self, layer: usize) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(&(entry_layer, _), _)| entry_layer == layer)
            .min_by_key(|(_, entry)| (entry.frequency, entry.last_used))
            .map(|(&key, _)| key);
        let Some(victim) = victim else {
            return false;
        };
        let evicted = self
            .entries
            .remove(&victim)
            .expect("selected ghost MoE victim disappeared");
        let size = evicted.expert.byte_len();
        self.bytes = self.bytes.saturating_sub(size);
        self.layer_bytes[layer] = self.layer_bytes[layer].saturating_sub(size);
        self.evictions = self.evictions.saturating_add(1);
        true
    }
}

/// Below this budget the CUDA lane keeps faulting misses in from the `.cghost`
/// mapping instead of routing them through the host arena. A tier smaller than
/// one token's routed working set (240 records ≈ 806 MB on the 26B row) cannot
/// hold anything across a token boundary, so it would only add an allocation
/// and a copy to every miss. Above it the arena starts retaining real
/// residency and each retained record converts a storage read into a PCIe copy.
#[cfg(feature = "cuda")]
const GHOST_CUDA_HOST_TIER_MIN_BYTES: usize = 1024 * 1024 * 1024;

/// Physical RAM the Ghost-MoE CUDA host expert tier must leave for everything
/// else: the OS, the caller's desktop session, the sparse GGUF shadow's mapped
/// common core, and this process's own non-tier heap. Sizing an arena from
/// "available" RAM without a reserve is how a 16 GiB host starts swapping
/// mid-generation, which is far worse than a smaller tier. Override with
/// `CAMELID_GEMMA4_GHOST_HOST_TIER_RESERVE_MIB`.
#[cfg(feature = "cuda")]
const GHOST_CUDA_HOST_TIER_RESERVE_MIB: u64 = 3072;

/// Resolve the host expert-tier budget (MiB) for the Ghost-MoE CUDA lane.
///
/// The tier is all-or-nothing by measurement, not by taste. On the tracked box
/// a 7 GiB tier lifted steady decode 8.5 -> 12 tok/s, but a 1 GiB tier (an
/// auto-size taken during a transient RAM dip) ran at **0.0% hit rate** and
/// dragged decode to 4.85: the tier only ever sees the VRAM cache's miss tail,
/// whose LRU reuse distance far exceeds a small arena, so every miss paid a
/// fill + eviction on top of the storage read while the pinned RAM starved the
/// OS page cache that was previously absorbing those reads. A tier that cannot
/// hold a meaningful fraction of the routed payload must therefore refuse to
/// build and leave the page cache alone.
///
/// Explicit `CAMELID_GEMMA4_GHOST_HOST_TIER_MIB` always wins, including `0`
/// (forces the mapped path for A/B) and sub-viable sizes (measurement needs
/// them). A caller-requested `0` (`--expert-cache-mib 0` is documented as the
/// smallest application-owned footprint) disables auto-sizing entirely:
/// a minimal-memory request must not become gigabytes of page-locked RAM.
#[cfg(feature = "cuda")]
fn ghost_cuda_host_tier_mib(cghost: &Path, requested_mib: usize) -> usize {
    if let Some(explicit) = std::env::var("CAMELID_GEMMA4_GHOST_HOST_TIER_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return explicit;
    }
    if requested_mib == 0 {
        return 0;
    }
    let reserve_mib = std::env::var("CAMELID_GEMMA4_GHOST_HOST_TIER_RESERVE_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(GHOST_CUDA_HOST_TIER_RESERVE_MIB);
    let (_total, available) = crate::capability::host_ram_total_available_bytes();
    if available == 0 {
        // No probe on this platform: do not pin an arena sized from nothing.
        return 0;
    }
    let spare_mib = (available / (1024 * 1024)).saturating_sub(reserve_mib);
    // Never retain more than the routed payload itself; beyond that the arena
    // would reserve RAM it can never fill.
    let payload_mib = std::fs::metadata(cghost)
        .map(|m| m.len() / (1024 * 1024))
        .unwrap_or(0);
    if payload_mib == 0 {
        return 0;
    }
    let auto_mib = spare_mib.min(payload_mib);
    // Viability gate (see above): below a quarter of the payload the tier is
    // measured to be strictly worse than the page cache it displaces.
    if auto_mib < payload_mib / 4 {
        eprintln!(
            "[ghost] host expert tier skipped: {auto_mib} MiB spare would cover under a quarter of \
             the {payload_mib} MiB routed payload (a small tier measured 0% hits while starving \
             the OS page cache); set CAMELID_GEMMA4_GHOST_HOST_TIER_MIB to force one"
        );
        return 0;
    }
    usize::try_from(auto_mib).unwrap_or(usize::MAX)
}

/// One cache for the whole model, rather than one nominal budget per layer.
/// This is the memory-ceiling invariant: regardless of how many of Gemma 4's
/// 30×128 experts a session routes to, retained wire bytes never exceed the
/// configured global budget. A too-large entry remains usable for the current
/// layer but is not retained.
struct GhostMoeExpertCache {
    file: Arc<GhostFile>,
    budget_bytes: usize,
    /// One non-overlapping budget segment per model layer. Remainder bytes are
    /// assigned to the first layers; the sum is exactly `budget_bytes`.
    layer_budgets: Vec<usize>,
    /// Positioned reads for one routed top-k can be issued concurrently on
    /// SSD/NVMe without tying up Rayon compute workers. Set
    /// `CAMELID_GEMMA4_GHOST_READ_THREADS=1` for rotational or strictly serial
    /// storage. Windows' unbuffered reader is serialized internally, so it
    /// defaults to one thread there.
    read_pool: Option<rayon::ThreadPool>,
    state: std::sync::Mutex<GhostMoeCacheState>,
}

impl GhostMoeExpertCache {
    fn new(file: Arc<GhostFile>, budget_bytes: usize) -> Self {
        let layer_count = file.index.block_count.max(1);
        let base_layer_budget = budget_bytes / layer_count;
        let remainder = budget_bytes % layer_count;
        let layer_budgets = (0..layer_count)
            .map(|layer| base_layer_budget + usize::from(layer < remainder))
            .collect::<Vec<_>>();
        let default_read_threads = if cfg!(windows) { 1 } else { 4 };
        let read_threads = std::env::var("CAMELID_GEMMA4_GHOST_READ_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_read_threads)
            .clamp(1, 8);
        let read_pool = (read_threads > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(read_threads)
                    .thread_name(|index| format!("ghost-moe-read-{index}"))
                    .build()
                    .ok()
            })
            .flatten();
        Self {
            file,
            budget_bytes,
            layer_budgets,
            read_pool,
            state: std::sync::Mutex::new(GhostMoeCacheState {
                entries: std::collections::HashMap::new(),
                seen_keys: std::collections::HashSet::new(),
                layer_bytes: vec![0; layer_count],
                clock: 0,
                layer_accesses: vec![0; layer_count],
                bytes: 0,
                hits: 0,
                misses: 0,
                compulsory_misses: 0,
                capacity_misses: 0,
                evictions: 0,
                bytes_read: 0,
            }),
        }
    }

    #[cfg(test)]
    fn get(&self, layer: usize, expert: usize) -> Result<Arc<GhostMoeExpert>> {
        self.get_many(layer, &[expert]).map(|mut values| {
            values
                .pop()
                .expect("one requested ghost MoE expert must produce one result")
        })
    }

    /// Borrow an already-resident immutable record without changing cache
    /// frequency, recency, or observable hit/miss counters. The persistent
    /// Metal lane uses this only as a slot-fill source: routing remains owned by
    /// the normal caller, and a miss falls through to direct positioned I/O.
    #[cfg(any(target_os = "macos", test))]
    fn peek_resident(&self, layer: usize, expert: usize) -> Option<Arc<GhostMoeExpert>> {
        self.state
            .lock()
            .ok()?
            .entries
            .get(&(layer, expert))
            .map(|entry| Arc::clone(&entry.expert))
    }

    /// Resolve one layer's routed top-k as a batch. Cache hits are cloned under
    /// one short lock; misses are sorted by expert index (the v2 `.cghost`
    /// physical order) and read concurrently when a read pool is available.
    /// The returned vector is restored to `experts` order, so callers retain the
    /// router's exact floating-point accumulation order.
    fn get_many(&self, layer: usize, experts: &[usize]) -> Result<Vec<Arc<GhostMoeExpert>>> {
        if layer >= self.layer_budgets.len() {
            return Err(BackendError::InvalidModelMetadata(format!(
                "ghost MoE cache layer {layer} is outside its {}-layer layout",
                self.layer_budgets.len()
            )));
        }
        let mut resolved: Vec<Option<Arc<GhostMoeExpert>>> = vec![None; experts.len()];
        // Chunked prefill deliberately passes repeated route selections. Keep
        // their count and recency so an over-budget layer retains the experts
        // the prompt actually favored, rather than whichever numeric IDs were
        // inserted last after the physical reads were sorted.
        let mut missing_requests = std::collections::HashMap::<usize, (u16, u64)>::new();
        {
            let mut state = self.state.lock().expect("ghost MoE cache poisoned");
            for (slot, &expert) in experts.iter().enumerate() {
                let key = (layer, expert);
                state.touch_layer(layer);
                let now = state.clock;
                if let Some(entry) = state.entries.get_mut(&key) {
                    entry.frequency = entry.frequency.saturating_add(1);
                    entry.last_used = now;
                    resolved[slot] = Some(Arc::clone(&entry.expert));
                    state.hits = state.hits.saturating_add(1);
                } else {
                    state.misses = state.misses.saturating_add(1);
                    if state.seen_keys.insert(key) {
                        state.compulsory_misses = state.compulsory_misses.saturating_add(1);
                    } else {
                        state.capacity_misses = state.capacity_misses.saturating_add(1);
                    }
                    let request = missing_requests.entry(expert).or_insert((0, now));
                    request.0 = request.0.saturating_add(1);
                    request.1 = now;
                }
            }
        }

        // Expert groups are emitted in ascending expert order within a layer.
        // Sorting therefore gives the serial fallback monotonic file offsets;
        // par_iter preserves this indexed result order while allowing NVMe to
        // service a shallow queue of independent positioned reads.
        let mut missing: Vec<usize> = missing_requests.keys().copied().collect();
        missing.sort_unstable();
        let read_one = |&expert: &usize| -> Result<(usize, Arc<GhostMoeExpert>)> {
            Ok((expert, Arc::new(self.file.read_moe_expert(layer, expert)?)))
        };
        let mut loaded: Vec<(usize, Arc<GhostMoeExpert>)> = match &self.read_pool {
            Some(pool) if missing.len() > 1 => {
                pool.install(|| missing.par_iter().map(read_one).collect::<Result<Vec<_>>>())?
            }
            _ => missing.iter().map(read_one).collect::<Result<Vec<_>>>()?,
        };
        // I/O stays in physical order, but cache admission runs from the least
        // useful cold route to the most useful. Thus the final bounded segment
        // contains the highest-frequency, most-recent prompt experts even when
        // their numeric IDs were read first.
        loaded.sort_unstable_by_key(|(expert, _)| {
            missing_requests
                .get(expert)
                .copied()
                .expect("every loaded expert came from the missing request set")
        });

        let mut loaded_by_expert = std::collections::HashMap::with_capacity(loaded.len());
        {
            let mut state = self.state.lock().expect("ghost MoE cache poisoned");
            for (expert, loaded) in loaded {
                let key = (layer, expert);
                let size = loaded.byte_len();
                state.bytes_read = state.bytes_read.saturating_add(size as u64);

                // A second request can win the race while I/O is in flight. Use
                // its immutable entry rather than replacing it, but still report
                // the physical bytes this request actually read.
                let (request_frequency, request_last_used) = missing_requests
                    .get(&expert)
                    .copied()
                    .expect("every loaded expert came from the missing request set");
                if let Some(existing) = state.entries.get_mut(&key) {
                    existing.frequency = existing.frequency.saturating_add(request_frequency);
                    existing.last_used = existing.last_used.max(request_last_used);
                    loaded_by_expert.insert(expert, Arc::clone(&existing.expert));
                    continue;
                }

                let layer_budget = self.layer_budgets[layer];
                if size <= layer_budget {
                    while state.layer_bytes[layer].saturating_add(size) > layer_budget {
                        if !state.evict_one_from_layer(layer) {
                            break;
                        }
                    }
                    if state.layer_bytes[layer].saturating_add(size) <= layer_budget {
                        state.entries.insert(
                            key,
                            GhostMoeCacheEntry {
                                expert: Arc::clone(&loaded),
                                frequency: request_frequency,
                                last_used: request_last_used,
                            },
                        );
                        state.layer_bytes[layer] = state.layer_bytes[layer].saturating_add(size);
                        state.bytes = state.bytes.saturating_add(size);
                    }
                }
                loaded_by_expert.insert(expert, loaded);
            }
        }

        for (slot, &expert) in experts.iter().enumerate() {
            if resolved[slot].is_none() {
                resolved[slot] = loaded_by_expert.get(&expert).cloned();
            }
        }
        resolved
            .into_iter()
            .map(|value| {
                value.ok_or_else(|| {
                    BackendError::InvalidModelMetadata(
                        "ghost MoE batch read lost a requested expert".into(),
                    )
                })
            })
            .collect()
    }

    /// CUDA normal-cache mode uploads immutable expert ranges directly from
    /// the `.cghost` mapping. Strict cache mode has no mapping and deliberately
    /// falls back to the allocating positioned-reader cache.
    ///
    /// Only reached when the page-locked host tier (`SserHostTier`) is absent.
    /// Routing VRAM misses through the *pageable* `get_many` arena instead was
    /// measured on the tracked box and is a REGRESSION (6.22 -> 2.29-5.74 tok/s
    /// across 2-6 GiB budgets): the OS page cache is already an opportunistic
    /// host tier over this mapping, it uses all free RAM rather than a fixed
    /// reservation, and an owned arena competes with it for the same pages
    /// while still costing a staging memcpy per miss. The page-locked tier wins
    /// for a different reason — a hit there is a pure DMA with no CPU copy —
    /// so it supersedes this path rather than layering on it.
    #[cfg(feature = "cuda")]
    fn get_many_cuda(&self, layer: usize, experts: &[usize]) -> Result<Vec<GhostCudaExpertRecord>> {
        let mut mapped = Vec::with_capacity(experts.len());
        for &expert in experts {
            let Some(record) = self.file.mapped_moe_expert(layer, expert)? else {
                return self.get_many(layer, experts).map(|records| {
                    records
                        .into_iter()
                        .map(GhostCudaExpertRecord::Owned)
                        .collect()
                });
            };
            mapped.push(GhostCudaExpertRecord::Mapped(record));
        }
        Ok(mapped)
    }

    fn stats(&self) -> GhostMoeCacheStats {
        let state = self.state.lock().expect("ghost MoE cache poisoned");
        GhostMoeCacheStats {
            hits: state.hits,
            misses: state.misses,
            compulsory_misses: state.compulsory_misses,
            capacity_misses: state.capacity_misses,
            evictions: state.evictions,
            bytes_read: state.bytes_read,
            resident_experts: state.entries.len(),
            resident_bytes: state.bytes,
            budget_bytes: self.budget_bytes,
        }
    }
}

#[derive(Clone)]
struct GhostMoeLayer {
    layer_idx: usize,
    cache: Arc<GhostMoeExpertCache>,
}

/// Persistent routed-expert residency bounds. Gemma 4 routes eight experts per
/// token, so eight is the correctness floor. Sixteen preserves the established
/// default; larger opt-in slabs trade unified memory for fewer multi-megabyte
/// `.cghost` reads when routes churn across tokens.
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_MIN: usize = 8;
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_DEFAULT: usize = 16;
#[cfg(any(target_os = "macos", test))]
const GHOST_METAL_EXPERT_SLOTS_MAX: usize = 128;
/// Global overflow experts reused across layers. 24 resident + 18 covers the
/// measured K=8 unique max of 42. Two banks ping-pong so a predicted in-flight
/// command buffer can keep reading bank A while CPU fills bank B. Never ×30.
#[cfg(target_os = "macos")]
const GHOST_METAL_OVERFLOW_SLOTS: usize = crate::metal::GEMMA4_OVERFLOW_BANK_SLOTS;
/// Resident slots/layer at or above which the per-layer expert union can never
/// spill: K is capped at 8 candidate positions x 8 routes = 64 distinct experts
/// (measured worst case is 42). At or above this the overflow bank is allocated
/// as a single inert copy instead of one per layer.
#[cfg(target_os = "macos")]
const GHOST_METAL_OVERFLOW_COVER_SLOTS: usize = 64;

#[cfg(any(target_os = "macos", test))]
fn parse_ghost_metal_slots_per_layer(value: Option<&str>) -> usize {
    value
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(GHOST_METAL_EXPERT_SLOTS_DEFAULT)
        .clamp(GHOST_METAL_EXPERT_SLOTS_MIN, GHOST_METAL_EXPERT_SLOTS_MAX)
}

#[cfg(target_os = "macos")]
fn ghost_metal_slots_per_layer_from_env() -> usize {
    let raw = std::env::var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER").ok();
    let slots = parse_ghost_metal_slots_per_layer(raw.as_deref());
    if let Some(raw) = raw {
        match raw.trim().parse::<usize>() {
            Ok(requested) if requested != slots => eprintln!(
                "[gemma4-ghost-metal] requested {requested} slots/layer; clamped to supported range {GHOST_METAL_EXPERT_SLOTS_MIN}..={GHOST_METAL_EXPERT_SLOTS_MAX}: using {slots}"
            ),
            Err(_) => eprintln!(
                "[gemma4-ghost-metal] invalid CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER={raw:?}; using default {GHOST_METAL_EXPERT_SLOTS_DEFAULT}"
            ),
            _ => {}
        }
    }
    slots
}

#[cfg(any(target_os = "macos", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GhostMetalSlotEntry {
    expert: usize,
    frequency: u16,
    last_used: u64,
}

/// Deterministic per-layer LFU/LRU directory for the persistent Metal expert
/// slots. The directory deliberately knows nothing about Metal: the caller
/// supplies a loader that writes a missing expert directly into the selected
/// slot's shared storage. A mapping is committed only after that loader
/// succeeds, so a short read can never make partially initialized GPU bytes
/// addressable.
///
/// Slots selected by the current route are pinned until the whole route has
/// been resolved. Consequently a route with at most `entries.len()` distinct
/// experts cannot evict one of its own earlier selections while filling later
/// misses. Eviction chooses the least frequently used unpinned slot and uses
/// oldest access as its stable tie-break.
#[cfg(any(target_os = "macos", test))]
#[derive(Debug)]
struct GhostMetalSlotDirectory {
    entries: Vec<Option<GhostMetalSlotEntry>>,
    resident_slot_table: [i16; 128],
    pinned_hot_slots: usize,
    clock: u64,
    accesses: u64,
}

#[cfg(any(target_os = "macos", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GhostMetalSlotLoad {
    slot: usize,
    expert: usize,
    frequency: u16,
    last_used: u64,
}

#[cfg(any(target_os = "macos", test))]
#[derive(Debug, PartialEq, Eq)]
struct GhostMetalSlotPlan {
    /// Slot IDs in the router's original top-k order.
    route_slots: Vec<usize>,
    /// Distinct cache misses. Their slots have already been invalidated in the
    /// directory and become visible again only through `commit_load`.
    loads: Vec<GhostMetalSlotLoad>,
    /// Route entries served without another slot fill. Repeated experts within
    /// the same plan count as hits because they do not cause additional I/O.
    hits: usize,
    /// Resident entries invalidated to make room for this plan's loads.
    evictions: usize,
}

#[cfg(any(target_os = "macos", test))]
impl GhostMetalSlotDirectory {
    fn new(slot_count: usize) -> Self {
        // Plain LFU/LRU over every slot by default. Pinning slots 0..N to
        // experts 0..N-1 ("hot set") protects arbitrary expert IDs, not hot
        // ones, and leaves only ~6 slots to absorb the real routing
        // distribution; the measured K=1 lane with plain LFU reached 97.5%
        // hits at 80 slots/layer. Opt back in with
        // CAMELID_GEMMA4_GHOST_METAL_HOT_PIN=1 for A/B only.
        let pin_enabled = std::env::var("CAMELID_GEMMA4_GHOST_METAL_HOT_PIN")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        let pinned_hot_slots = if !pin_enabled {
            0
        } else if slot_count >= 128 {
            slot_count
        } else if slot_count >= 16 {
            slot_count.saturating_sub(6).max(slot_count * 3 / 4)
        } else {
            slot_count / 2
        };
        Self {
            entries: vec![None; slot_count],
            resident_slot_table: [-1; 128],
            pinned_hot_slots,
            clock: 0,
            accesses: 0,
        }
    }

    #[inline(always)]
    fn lookup_resident_slot(&self, expert_id: usize) -> Option<usize> {
        if expert_id < 128 {
            let slot = self.resident_slot_table[expert_id];
            if slot >= 0 {
                return Some(slot as usize);
            }
        }
        None
    }

    fn plan(&mut self, experts: &[usize]) -> Result<GhostMetalSlotPlan> {
        let distinct = experts
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if distinct > self.entries.len() {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Ghost Metal route has {distinct} distinct experts but only {} slots",
                self.entries.len()
            )));
        }

        let mut pinned = vec![false; self.entries.len()];
        let mut route_slots = Vec::with_capacity(experts.len());
        let mut loads = Vec::<GhostMetalSlotLoad>::new();
        let mut planned = std::collections::HashMap::<usize, usize>::new();
        let mut evictions = 0usize;

        for &expert in experts {
            self.clock = self.clock.saturating_add(1);
            self.accesses = self.accesses.saturating_add(1);
            if self.accesses.is_multiple_of(256) {
                for entry in self.entries.iter_mut().flatten() {
                    entry.frequency = (entry.frequency / 2).max(1);
                }
            }

            // 1. O(1) Resident Table Hit Lookup. The identity check guards
            // against a resident_slot_table/entries desync: a hit on a slot
            // that now holds a different expert must be treated as a miss
            // (and the stale table entry cleared), never served silently.
            if let Some(slot) = self.lookup_resident_slot(expert) {
                match self.entries[slot].as_mut() {
                    Some(entry) if entry.expert == expert => {
                        entry.frequency = entry.frequency.saturating_add(1);
                        entry.last_used = self.clock;
                        pinned[slot] = true;
                        route_slots.push(slot);
                        continue;
                    }
                    _ => {
                        if expert < 128 {
                            self.resident_slot_table[expert] = -1;
                        }
                    }
                }
            }

            if let Some(&load_idx) = planned.get(&expert) {
                let load = &mut loads[load_idx];
                load.frequency = load.frequency.saturating_add(1);
                load.last_used = self.clock;
                route_slots.push(load.slot);
                continue;
            }

            // 2. Select slot with Hot-Set Protection: consume free hot slots, then free transient, then evict transient
            let slot = if let Some(free_hot) = self.entries[..self.pinned_hot_slots]
                .iter()
                .enumerate()
                .position(|(s, e)| e.is_none() && !pinned[s])
            {
                free_hot
            } else if let Some(free_trans) = self.entries[self.pinned_hot_slots..]
                .iter()
                .enumerate()
                .position(|(s, e)| e.is_none() && !pinned[self.pinned_hot_slots + s])
            {
                self.pinned_hot_slots + free_trans
            } else {
                let transient_avail = self.entries[self.pinned_hot_slots..]
                    .iter()
                    .enumerate()
                    .any(|(s, _)| !pinned[self.pinned_hot_slots + s]);
                let range = if transient_avail {
                    self.pinned_hot_slots..self.entries.len()
                } else {
                    0..self.entries.len()
                };
                self.entries[range.clone()]
                    .iter()
                    .enumerate()
                    .filter(|(s, _)| !pinned[range.start + *s])
                    .min_by_key(|(s, entry)| match entry {
                        None => (0u8, 0u16, 0u64, range.start + *s),
                        Some(entry) => (1, entry.frequency, entry.last_used, range.start + *s),
                    })
                    .map(|(s, _)| range.start + s)
                    .expect("distinct expert count was checked against slot count")
            };

            if let Some(old) = self.entries[slot].take() {
                evictions += 1;
                if old.expert < 128 {
                    self.resident_slot_table[old.expert] = -1;
                }
            }
            self.entries[slot] = None;
            let load_idx = loads.len();
            loads.push(GhostMetalSlotLoad {
                slot,
                expert,
                frequency: 1,
                last_used: self.clock,
            });
            planned.insert(expert, load_idx);
            pinned[slot] = true;
            route_slots.push(slot);
        }
        let hits = experts.len().saturating_sub(loads.len());
        Ok(GhostMetalSlotPlan {
            route_slots,
            loads,
            hits,
            evictions,
        })
    }

    fn commit_load(&mut self, load: GhostMetalSlotLoad) {
        debug_assert!(self.entries.get(load.slot).is_some_and(Option::is_none));
        self.entries[load.slot] = Some(GhostMetalSlotEntry {
            expert: load.expert,
            frequency: load.frequency,
            last_used: load.last_used,
        });
        if load.expert < 128 {
            self.resident_slot_table[load.expert] = load.slot as i16;
        }
    }
}

/// Retain the hottest `limit` prompt experts while preserving the original
/// repeated route sequence for LFU/recency evidence. Frequency wins; the most
/// recent occurrence breaks ties; expert ID is the final deterministic key.
fn ghost_metal_prewarm_sequence(
    routed_experts: &[usize],
    expert_count: usize,
    limit: usize,
) -> Vec<usize> {
    if expert_count == 0 || limit == 0 {
        return Vec::new();
    }
    let mut frequency = vec![0usize; expert_count];
    let mut last_used = vec![0usize; expert_count];
    for (position, &expert) in routed_experts.iter().enumerate() {
        if expert < expert_count {
            frequency[expert] += 1;
            last_used[expert] = position;
        }
    }
    let mut ranked = (0..expert_count)
        .filter(|&expert| frequency[expert] > 0)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by_key(|&expert| {
        (
            std::cmp::Reverse(frequency[expert]),
            std::cmp::Reverse(last_used[expert]),
            expert,
        )
    });
    ranked.truncate(limit);
    let selected = ranked.into_iter().collect::<std::collections::HashSet<_>>();
    routed_experts
        .iter()
        .copied()
        .filter(|expert| selected.contains(expert))
        .collect()
}

/// Cumulative slot-directory and I/O telemetry. This lives under the existing
/// expert-runtime mutex, so the hot path needs no atomics or extra locking.
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct GhostMetalSlotStats {
    route_lookups: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    host_fills: u64,
    prewarm_copies: u64,
    direct_reads: u64,
    direct_read_bytes: u64,
    direct_read_failures: u64,
}

#[cfg(target_os = "macos")]
impl GhostMetalSlotStats {
    fn saturating_delta(self, earlier: Self) -> Self {
        Self {
            route_lookups: self.route_lookups.saturating_sub(earlier.route_lookups),
            hits: self.hits.saturating_sub(earlier.hits),
            misses: self.misses.saturating_sub(earlier.misses),
            evictions: self.evictions.saturating_sub(earlier.evictions),
            host_fills: self.host_fills.saturating_sub(earlier.host_fills),
            prewarm_copies: self.prewarm_copies.saturating_sub(earlier.prewarm_copies),
            direct_reads: self.direct_reads.saturating_sub(earlier.direct_reads),
            direct_read_bytes: self
                .direct_read_bytes
                .saturating_sub(earlier.direct_read_bytes),
            direct_read_failures: self
                .direct_read_failures
                .saturating_sub(earlier.direct_read_failures),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.route_lookups = self.route_lookups.saturating_add(other.route_lookups);
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.evictions = self.evictions.saturating_add(other.evictions);
        self.host_fills = self.host_fills.saturating_add(other.host_fills);
        self.prewarm_copies = self.prewarm_copies.saturating_add(other.prewarm_copies);
        self.direct_reads = self.direct_reads.saturating_add(other.direct_reads);
        self.direct_read_bytes = self
            .direct_read_bytes
            .saturating_add(other.direct_read_bytes);
        self.direct_read_failures = self
            .direct_read_failures
            .saturating_add(other.direct_read_failures);
    }
}

#[cfg(target_os = "macos")]
struct GhostMetalSharedBuffers {
    gate: metal::Buffer,
    up: metal::Buffer,
    down: metal::Buffer,
}

#[cfg(target_os = "macos")]
struct GhostMetalExpertLayer {
    directory: GhostMetalSlotDirectory,
    slots: crate::metal::Gemma4Q4ExpertSlots,
    stats: GhostMetalSlotStats,
    shared: Option<GhostMetalSharedBuffers>,
}

#[cfg(target_os = "macos")]
struct GhostMetalExpertRuntime {
    engine: crate::metal::Gemma4Q4ExpertMetal,
    layers: Vec<GhostMetalExpertLayer>,
    fused_fast: bool,
    common: Option<crate::metal::Gemma4GhostCommonMetal>,
    sequence_mode: GhostMetalSequenceMode,
    latest_routed_experts: Vec<Vec<usize>>,
    expert_decay_scores: Vec<Vec<f32>>,
    /// Eight 18-slot overflow slabs, reused across all 30 layers. Predicted
    /// rounds bind `overflow_bank[layer % copies]` so prior layers can stay
    /// in flight. Combined footprint is ~461 MiB, not 2.25 GiB.
    overflow_bank: Vec<crate::metal::Gemma4Q4ExpertSlots>,
    last_chained_k: Option<usize>,
    /// (start_pos, K, hash of the hidden rows) of the last successful chained
    /// round. Route prediction reuses the previous round's expert unions, which
    /// is only sound when the SAME chunk is re-verified at the SAME position
    /// (the verifier harness); a fresh chunk mispredicts by construction.
    last_chained_sig: Option<(usize, usize, u64)>,
    /// Set while retrying a refused predicted round without prediction.
    suppress_prediction: bool,
    /// Prefill chunks advance through fresh prompt segments; speculative
    /// slot fills from the previous round's unions are ~69% wrong there and
    /// evict live experts (measured 12.8 GB of fill reads across a 31-token
    /// prompt). Decode/verify rounds keep the speculative fill.
    prefill_round: bool,
}

/// K=1 decode lane selection. Default: the dedicated HEAD lane
/// (`try_ghost_common_step`) — verified oracle-exact 24/24 vs llama.cpp on
/// 2026-08-19 after fixing the misaligned-uchar4 UB in
/// `gemma4_q4_expert_block_term_simd` that had corrupted its expert GEMMs.
/// It pipelines per-layer fills against GPU work (the 17-20 tok/s design)
/// and is the token-parity reference on Metal. `CAMELID_GEMMA4_CHAINED_K1=1`
/// re-routes K=1 through the K>1 chained lane for A/B only (the chained
/// lane pays per-layer host synchronization with no batching win at K=1).
#[cfg(target_os = "macos")]
fn chained_k1_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_CHAINED_K1")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Identity of a chained round's input: position plus a hash of every hidden
/// row (bit patterns, so it is exact and cheap: K x hidden f32).
#[cfg(target_os = "macos")]
fn chained_round_signature(start_pos: usize, hidden_rows: &[Vec<f32>]) -> (usize, usize, u64) {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for row in hidden_rows {
        row.len().hash(&mut hasher);
        for value in row {
            value.to_bits().hash(&mut hasher);
        }
    }
    (start_pos, hidden_rows.len(), hasher.finish())
}

/// CAMELID_GEMMA4_TRACE_LANE=1: eprintln which forward lane each call takes.
fn lane_trace_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_TRACE_LANE")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[cfg(target_os = "macos")]
fn ghost_metal_timing_enabled() -> bool {
    std::env::var("CAMELID_GEMMA4_GHOST_METAL_TIMING")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Bring `selected_experts` into this layer's persistent Metal slots.
///
/// Directory-first: experts already resident in a slot cost nothing (no host
/// lookup, no I/O, no copy). Misses are sourced from an already-resident host
/// cache record when one exists (`peek_resident`: no I/O, no cache
/// accounting), otherwise read straight from the `.cghost` into the slot with
/// positioned reads (concurrent through the read pool). Routing this through
/// `get_many` for every routed expert — the previous behaviour — fetched all
/// eight experts per layer per token through the host cache even when the slot
/// already held them: with a 64 MiB cache that was eight 3.3 MB disk reads per
/// layer per token (115.9 GiB over 78 positions) and collapsed decode to
/// <1 tok/s. `nvme_*`/`demand_loads` now count only real slot loads from disk.
#[cfg(target_os = "macos")]
fn fill_metal_wave_slots_from_host_cache(
    layer: &mut GhostMetalExpertLayer,
    cache: &GhostMoeExpertCache,
    layer_idx: usize,
    selected_experts: &[usize],
    nvme_us: &std::sync::atomic::AtomicU64,
    nvme_bytes: &std::sync::atomic::AtomicU64,
    demand_loads: &std::sync::atomic::AtomicUsize,
) {
    let Ok(plan) = layer.directory.plan(selected_experts) else {
        eprintln!(
            "[gemma4-ghost-metal] layer {layer_idx} wave plan refused {} experts into {} slots",
            selected_experts.len(),
            layer.directory.entries.len()
        );
        return;
    };
    if plan.loads.is_empty() {
        return;
    }

    let record_bytes = layer.slots.slot_record_bytes();
    let mut direct_fallback = Vec::with_capacity(plan.loads.len());
    for load in plan.loads.iter().copied() {
        if let Some(record) = cache.peek_resident(layer_idx, load.expert) {
            if record.byte_len() == record_bytes {
                if let Some(dest) = layer.slots.slot_bytes_mut(load.slot) {
                    dest.copy_from_slice(record.record_bytes());
                    layer.directory.commit_load(load);
                    continue;
                }
            }
        }
        direct_fallback.push(load);
    }

    if !direct_fallback.is_empty() {
        let file = &cache.file;
        let t_disk = std::time::Instant::now();
        let results: Vec<(GhostMetalSlotLoad, Result<()>)> = if let Some(pool) = &cache.read_pool {
            if direct_fallback.len() == 1 {
                let load = direct_fallback[0];
                let result = layer
                    .slots
                    .slot_bytes_mut(load.slot)
                    .ok_or_else(|| {
                        BackendError::InvalidModelMetadata(format!(
                            "Ghost Metal slot {} is outside the layer slab",
                            load.slot
                        ))
                    })
                    .and_then(|destination| {
                        file.read_moe_expert_into(layer_idx, load.expert, destination)
                    });
                vec![(load, result)]
            } else {
                let slots_ref = &layer.slots;
                pool.install(|| {
                    direct_fallback
                        .par_iter()
                        .map(|&load| {
                            let res = match unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                                Some(destination) => {
                                    file.read_moe_expert_into(layer_idx, load.expert, destination)
                                }
                                None => Err(BackendError::InvalidModelMetadata(format!(
                                    "Ghost Metal slot {} is outside the layer slab",
                                    load.slot
                                ))),
                            };
                            (load, res)
                        })
                        .collect()
                })
            }
        } else {
            direct_fallback
                .iter()
                .map(|&load| {
                    let res = match layer.slots.slot_bytes_mut(load.slot) {
                        Some(destination) => {
                            file.read_moe_expert_into(layer_idx, load.expert, destination)
                        }
                        None => Err(BackendError::InvalidModelMetadata(format!(
                            "Ghost Metal slot {} is outside the layer slab",
                            load.slot
                        ))),
                    };
                    (load, res)
                })
                .collect()
        };
        let mut ok_reads = 0usize;
        for (load, result) in results {
            match result {
                Ok(()) => {
                    layer.directory.commit_load(load);
                    ok_reads += 1;
                }
                Err(err) => {
                    eprintln!(
                        "[gemma4-ghost-metal] layer {layer_idx} failed to fill slot {} with expert {}: {err}",
                        load.slot, load.expert
                    );
                }
            }
        }
        if ok_reads > 0 {
            nvme_us.fetch_add(
                (t_disk.elapsed().as_secs_f64() * 1_000_000.0) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            nvme_bytes.fetch_add(
                (ok_reads * record_bytes) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            demand_loads.fetch_add(ok_reads, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Pack `experts` into compact slots `[0, experts.len())` of `slots` from the
/// host Ghost cache. Does not touch a layer directory, so it is safe to fill a
/// ping-pong slab while GPU still reads a different buffer.
#[cfg(target_os = "macos")]
fn fill_compact_wave_into_slots(
    slots: &metal::Buffer,
    slot_count: usize,
    cache: &GhostMoeExpertCache,
    layer_idx: usize,
    experts: &[usize],
    updated_slots: &mut [u32; 128],
    nvme_us: &std::sync::atomic::AtomicU64,
    nvme_bytes: &std::sync::atomic::AtomicU64,
    demand_loads: &std::sync::atomic::AtomicUsize,
) {
    updated_slots.fill(0xFFFFFFFFu32);
    if experts.is_empty() {
        return;
    }
    let stats_before = cache.stats();
    let t_host = std::time::Instant::now();
    let records = match cache.get_many(layer_idx, experts) {
        Ok(records) => records,
        Err(err) => {
            eprintln!("[gemma4-ghost-metal] layer {layer_idx} compact get_many failed: {err}");
            return;
        }
    };
    let host_ms = t_host.elapsed().as_secs_f64() * 1000.0;
    let stats_after = cache.stats();
    let new_bytes = stats_after
        .bytes_read
        .saturating_sub(stats_before.bytes_read);
    let new_misses = stats_after.misses.saturating_sub(stats_before.misses);
    if new_bytes > 0 {
        nvme_us.fetch_add(
            (host_ms * 1000.0) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        nvme_bytes.fetch_add(new_bytes, std::sync::atomic::Ordering::Relaxed);
        demand_loads.fetch_add(new_misses as usize, std::sync::atomic::Ordering::Relaxed);
    }
    let record_bytes = crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES;
    let stride = crate::metal::GEMMA4_Q4_EXPERT_SLOT_STRIDE;
    use rayon::prelude::*;
    let base = slots.contents() as *mut u8;
    let base_raw = base as usize;
    let items: Vec<_> = experts
        .iter()
        .copied()
        .zip(records.into_iter())
        .enumerate()
        .collect();
    items.into_par_iter().for_each(|(i, (_expert, record))| {
        if i < slot_count && record.byte_len() == record_bytes {
            unsafe {
                let dst = (base_raw + i * stride) as *mut u8;
                std::ptr::copy_nonoverlapping(record.record_bytes().as_ptr(), dst, record_bytes);
            }
        }
    });
    for (i, &expert) in experts.iter().take(slot_count).enumerate() {
        if expert < 128 {
            updated_slots[expert] = i as u32;
        }
    }
}

#[cfg(target_os = "macos")]
fn ghost_metal_stats_enabled() -> bool {
    ghost_metal_timing_enabled()
        || std::env::var("CAMELID_GEMMA4_GHOST_METAL_STATS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostMetalSequenceMode {
    Idle,
    Cpu,
    /// The prompt is advancing the authoritative host KV cache in layer-major
    /// chunks. Decode may switch to Metal only after an atomic cache import.
    HybridPrefill,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostPrefillPlan {
    ScalarCpu,
    CpuChunk,
    ScalarMetal,
    HybridChunk,
}

fn select_ghost_prefill_plan(
    chunk_eligible: bool,
    hybrid_enabled: bool,
    prompt_len: usize,
    required_positions: usize,
    common_capacity: Option<usize>,
) -> GhostPrefillPlan {
    match common_capacity {
        Some(capacity) if required_positions <= capacity => {
            if chunk_eligible && hybrid_enabled && prompt_len > 1 {
                GhostPrefillPlan::HybridChunk
            } else {
                GhostPrefillPlan::ScalarMetal
            }
        }
        _ if chunk_eligible && prompt_len > 1 => GhostPrefillPlan::CpuChunk,
        _ => GhostPrefillPlan::ScalarCpu,
    }
}

/// A generation request owns the persistent common-core KV state. Resetting it
/// on every exit (success, error, or cancellation) prevents a later request from
/// inheriting a hybrid/import decision if it returns before another position-zero
/// scalar step can reselect the lane.
#[cfg(target_os = "macos")]
struct GhostMetalSequenceCleanup<'a> {
    lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
}

#[cfg(target_os = "macos")]
impl<'a> GhostMetalSequenceCleanup<'a> {
    fn new(lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>) -> Self {
        Self { lane }
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostMetalSequenceCleanup<'_> {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.lane.lock() {
            if let Some(runtime) = guard.as_mut() {
                runtime.sequence_mode = GhostMetalSequenceMode::Idle;
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
enum GhostMetalExpertAttempt {
    Output(Vec<f32>),
    /// A positioned read or directory preparation failed. The immutable CPU
    /// Ghost cache remains authoritative and should retry the route normally.
    CpuFallback,
    /// Metal dispatch failed after successful slot preparation. Drop the lane
    /// so subsequent layers do not repeatedly pay a known-bad GPU attempt.
    DisableMetal,
}

#[cfg(target_os = "macos")]
enum GhostMetalCommonAttempt {
    Complete,
    Pending(crate::metal::Gemma4Q4ExpertPending),
    CpuFallback,
    DisableMetal,
}

/// Result of one token on the persistent Ghost common-core lane. Prompt
/// prefill advances every non-final token without materializing the hidden
/// state or running the tied vocabulary head; decode (and the final prompt
/// token) requests logits normally.
#[cfg(target_os = "macos")]
enum GhostCommonStepOutput {
    Advanced,
    Logits(Vec<f32>),
}

/// Ensures an independently queued shared branch reaches a terminal command
/// state on every error/cancellation edge before its persistent scratch can be
/// reused by another request.
#[cfg(target_os = "macos")]
struct GhostCommonPendingGuard(Option<crate::metal::Gemma4GhostCommonPending>);

#[cfg(target_os = "macos")]
impl GhostCommonPendingGuard {
    fn new(pending: crate::metal::Gemma4GhostCommonPending) -> Self {
        Self(Some(pending))
    }

    fn finish(&mut self) -> bool {
        self.0
            .take()
            .and_then(crate::metal::Gemma4GhostCommonPending::wait)
            .is_some()
    }

    fn take(&mut self) -> Option<crate::metal::Gemma4GhostCommonPending> {
        self.0.take()
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostCommonPendingGuard {
    fn drop(&mut self) {
        if let Some(pending) = self.0.take() {
            let _ = pending.wait();
        }
    }
}

/// Owns both commands that finish a fused-fast layer. The expert+tail command
/// is later in the singleton Metal queue, so draining it first also proves the
/// shared branch has reached a terminal GPU state. Drop preserves that ordering
/// on every error and cancellation edge before persistent scratch is reused.
#[cfg(target_os = "macos")]
struct GhostLayerPendingGuard {
    shared: Option<crate::metal::Gemma4GhostCommonPending>,
    tail: Option<crate::metal::Gemma4Q4ExpertPending>,
}

#[cfg(target_os = "macos")]
impl GhostLayerPendingGuard {
    fn new(
        shared: crate::metal::Gemma4GhostCommonPending,
        tail: crate::metal::Gemma4Q4ExpertPending,
    ) -> Self {
        Self {
            shared: Some(shared),
            tail: Some(tail),
        }
    }

    fn finish(&mut self) -> bool {
        let tail_ok = self
            .tail
            .take()
            .and_then(crate::metal::Gemma4Q4ExpertPending::wait)
            .is_some();
        let shared_ok = self
            .shared
            .take()
            .and_then(crate::metal::Gemma4GhostCommonPending::wait)
            .is_some();
        tail_ok && shared_ok
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostLayerPendingGuard {
    fn drop(&mut self) {
        if let Some(tail) = self.tail.take() {
            let _ = tail.wait();
        }
        if let Some(shared) = self.shared.take() {
            let _ = shared.wait();
        }
    }
}

#[cfg(target_os = "macos")]
impl GhostMetalExpertRuntime {
    fn new(layer_count: usize, fused_fast: bool, slots_per_layer: usize) -> Option<Self> {
        if !(GHOST_METAL_EXPERT_SLOTS_MIN..=GHOST_METAL_EXPERT_SLOTS_MAX).contains(&slots_per_layer)
        {
            return None;
        }
        let engine = crate::metal::Gemma4Q4ExpertMetal::new()?;
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let slots = crate::metal::Gemma4Q4ExpertSlots::new(slots_per_layer)?;
            debug_assert_eq!(slots.slot_count(), slots_per_layer);
            layers.push(GhostMetalExpertLayer {
                directory: GhostMetalSlotDirectory::new(slots_per_layer),
                slots,
                stats: GhostMetalSlotStats::default(),
                shared: None,
            });
        }
        // The overflow bank only holds experts beyond the resident slots. When
        // `slots_per_layer` already covers the worst-case per-layer union
        // (observed max 42 for K=8), no layer ever spills, so `fill_pong` /
        // `fill_compact_wave_into_slots` are never called and a single copy is
        // enough. Allocating 30 copies × 24 slots = 2.25 GiB of unified memory
        // that is never written is pure RAM pressure (it swaps on 16 GB). When
        // slots may spill we still need one distinct copy per layer, because
        // all 30 layers' fills complete before the single GPU commit — sharing
        // a copy across layers would overwrite an earlier layer's overflow
        // experts before the GPU reads them.
        let bank_never_written = slots_per_layer >= GHOST_METAL_OVERFLOW_COVER_SLOTS;
        let bank_copies = if bank_never_written {
            1
        } else {
            crate::metal::GEMMA4_OVERFLOW_BANK_COPIES.max(layer_count)
        };
        let mut overflow_bank = Vec::with_capacity(bank_copies);
        for _ in 0..bank_copies {
            overflow_bank.push(crate::metal::Gemma4Q4ExpertSlots::new(
                GHOST_METAL_OVERFLOW_SLOTS,
            )?);
        }
        eprintln!(
            "[gemma4-ghost-metal] global overflow bank: {}×{} slots ({:.2} MiB total){}, reused across {} layers (not per-layer)",
            bank_copies,
            GHOST_METAL_OVERFLOW_SLOTS,
            (bank_copies
                * GHOST_METAL_OVERFLOW_SLOTS
                * crate::metal::GEMMA4_Q4_EXPERT_SLOT_STRIDE) as f64
                / (1024.0 * 1024.0),
            if bank_never_written { " [inactive: resident slots cover the union]" } else { "" },
            layer_count
        );
        Some(Self {
            engine,
            layers,
            fused_fast,
            common: None,
            sequence_mode: GhostMetalSequenceMode::Idle,
            latest_routed_experts: vec![Vec::new(); layer_count],
            expert_decay_scores: vec![vec![0.0f32; 128]; layer_count],
            overflow_bank,
            last_chained_k: None,
            last_chained_sig: None,
            suppress_prediction: false,
            prefill_round: false,
        })
    }

    pub(crate) fn last_chained_ledger(&self) -> crate::metal::ChainedRoundHostLedger {
        self.common
            .as_ref()
            .map(|c| c.last_chained_ledger())
            .unwrap_or_default()
    }

    fn overflow_slot_count(&self) -> usize {
        self.overflow_bank[0].slot_count()
    }

    pub(crate) fn record_layer_routes(&mut self, layer_idx: usize, experts: &[usize]) {
        if let Some(scores) = self.expert_decay_scores.get_mut(layer_idx) {
            for s in scores.iter_mut() {
                *s *= 0.85;
            }
            for &e in experts {
                if let Some(val) = scores.get_mut(e) {
                    *val += 1.0;
                }
            }
        }
    }

    pub(crate) fn top_speculative_candidates(&self, layer_idx: usize, top_n: usize) -> Vec<usize> {
        let Some(scores) = self.expert_decay_scores.get(layer_idx) else {
            return Vec::new();
        };
        let mut idx: Vec<(usize, f32)> = scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, s)| *s > 0.01)
            .collect();
        idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        idx.into_iter().take(top_n).map(|(e, _)| e).collect()
    }

    fn get_or_init_shared_buffers(
        &mut self,
        layer_idx: usize,
        gate_bytes: &[u8],
        up_bytes: &[u8],
        down_bytes: &[u8],
    ) -> Option<(&metal::Buffer, &metal::Buffer, &metal::Buffer)> {
        let layer = self.layers.get_mut(layer_idx)?;
        if layer.shared.is_none() {
            let kernel = crate::metal::metal_linear_kernel()?;
            let gate = kernel.device.new_buffer_with_data(
                gate_bytes.as_ptr() as *const _,
                gate_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let up = kernel.device.new_buffer_with_data(
                up_bytes.as_ptr() as *const _,
                up_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let down = kernel.device.new_buffer_with_data(
                down_bytes.as_ptr() as *const _,
                down_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            layer.shared = Some(GhostMetalSharedBuffers { gate, up, down });
        }
        let s = layer.shared.as_ref()?;
        Some((&s.gate, &s.up, &s.down))
    }

    fn execute_attention_chunk(
        &mut self,
        layer_idx: usize,
        hidden_rows: &[Vec<f32>],
        rope_freq_base: f32,
        rope_factors: Option<&[f32]>,
        start_pos: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let common = self.common.as_mut()?;
        common.execute_attention_chunk(
            layer_idx,
            hidden_rows,
            rope_freq_base,
            rope_factors,
            start_pos,
        )
    }

    fn execute_attention_chunk_into(
        &mut self,
        layer_idx: usize,
        hidden_rows: &[Vec<f32>],
        rope_freq_base: f32,
        rope_factors: Option<&[f32]>,
        start_pos: usize,
        out_rows: &mut [Vec<f32>],
    ) -> bool {
        let Some(common) = self.common.as_mut() else {
            return false;
        };
        common.execute_attention_chunk_into(
            layer_idx,
            hidden_rows,
            rope_freq_base,
            rope_factors,
            start_pos,
            out_rows,
        )
    }

    fn execute_attention_and_shared_chunk_into(
        &mut self,
        layer_idx: usize,
        hidden_rows: &[Vec<f32>],
        rope_freq_base: f32,
        rope_factors: Option<&[f32]>,
        start_pos: usize,
        out_rows: &mut [Vec<f32>],
        out_shared_mlp_rows: &mut [Vec<f32>],
        out_router_logits: Option<&mut [f32]>,
        is_first_layer: bool,
    ) -> bool {
        let Some(common) = self.common.as_mut() else {
            return false;
        };
        common.execute_attention_and_shared_chunk_into(
            layer_idx,
            hidden_rows,
            rope_freq_base,
            rope_factors,
            start_pos,
            out_rows,
            out_shared_mlp_rows,
            out_router_logits,
            is_first_layer,
        )
    }

    #[cfg(target_os = "macos")]
    fn resident_expert_input_buffers(&self) -> Option<(&metal::Buffer, &metal::Buffer)> {
        self.common
            .as_ref()
            .map(|c| c.resident_expert_input_buffers())
    }

    #[cfg(target_os = "macos")]
    fn resident_residual_buffers(
        &self,
    ) -> Option<(&metal::Buffer, &metal::Buffer, &metal::Buffer)> {
        self.common.as_ref().map(|c| c.resident_residual_buffers())
    }

    #[cfg(target_os = "macos")]
    fn resident_fused_tail_buffers(
        &self,
        layer_idx: usize,
    ) -> Option<(
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
        &metal::Buffer,
        f32,
    )> {
        self.common
            .as_ref()
            .and_then(|c| c.resident_fused_tail_buffers(layer_idx))
    }

    #[cfg(target_os = "macos")]
    fn read_slab_a_into(&self, k_tokens: usize, out_rows: &mut [Vec<f32>]) {
        if let Some(common) = self.common.as_ref() {
            common.read_slab_a_into(k_tokens, out_rows);
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn execute_chained_round_all_layers(
        &mut self,
        hidden_rows: &[Vec<f32>],
        theta_local: f32,
        theta_global: f32,
        rope_factors: Option<&[f32]>,
        start_pos: usize,
        ghost_cache: Option<&GhostMoeExpertCache>,
        out_rows: &mut [Vec<f32>],
    ) -> bool {
        // Last-round prefetch is background prediction, never the outer critical path.
        let prefetch_ms = 0.0;

        let t_setup = std::time::Instant::now();
        let mut slot_mappings = [[0xFFFFFFFFu32; 128]; 30];
        let mut num_slots_per_layer = [0usize; 30];
        let expert_slabs: Vec<metal::Buffer> = self
            .layers
            .iter()
            .take(30)
            .map(|l| l.slots.slab_buffer().clone())
            .collect();

        for (li, layer) in self.layers.iter().enumerate().take(30) {
            for e in 0..128 {
                let slot = layer.directory.resident_slot_table[e];
                if slot >= 0 {
                    slot_mappings[li][e] = slot as u32;
                }
            }
            num_slots_per_layer[li] = layer.directory.entries.len();
        }
        let setup_ms = t_setup.elapsed().as_secs_f64() * 1000.0;

        let k_tokens = hidden_rows.len();
        // Predicted round-level submit reuses the previous round's per-layer
        // expert unions. That is only sound when the previous chained round
        // verified the SAME chunk at the SAME position (identical hidden rows):
        // any new chunk routes differently, the post-hoc union check refuses
        // the round after a full GPU pass, and the CPU fallback cannot run
        // (the chained lane never appends the host K/V caches). So predict
        // only on an exact repeat, and retry unpredicted on a refusal.
        let chunk_sig = chained_round_signature(start_pos, hidden_rows);
        let allow_predicted = !self.suppress_prediction
            && k_tokens > 1
            && self.last_chained_k == Some(k_tokens)
            && self.last_chained_sig == Some(chunk_sig);
        let predicted_unions = if self.prefill_round {
            vec![Vec::new(); self.latest_routed_experts.len()]
        } else {
            self.latest_routed_experts.clone()
        };
        let overflow_slot_count = self.overflow_slot_count();
        let copies = self.overflow_bank.len().max(1);
        let pong_slab_buf = self.overflow_bank[0].slab_buffer().clone();
        let pong_slot_count = overflow_slot_count;
        let overflow_bufs: Vec<metal::Buffer> = self
            .overflow_bank
            .iter()
            .map(|b| b.slab_buffer().clone())
            .collect();
        let n_layers = self.layers.len();
        // Every layer's fill completes before the single GPU commit, so two
        // layers sharing an overflow copy would let the later layer overwrite
        // the earlier one's experts before the GPU reads them. Only bind the
        // bank when there is a distinct copy per layer.
        let wave1_bufs: Vec<metal::Buffer> = if allow_predicted && copies >= n_layers {
            (0..n_layers)
                .map(|i| overflow_bufs[i % copies].clone())
                .collect()
        } else {
            Vec::new()
        };
        let wave1_slot_count = overflow_slot_count;
        let wave1_gpu = wave1_bufs.clone();

        let mut collected_routes = vec![Vec::new(); self.layers.len()];
        let nvme_us = std::sync::atomic::AtomicU64::new(0);
        let nvme_bytes = std::sync::atomic::AtomicU64::new(0);
        let demand_loads = std::sync::atomic::AtomicUsize::new(0);

        if allow_predicted {
            if let Some(cache) = ghost_cache {
                use rayon::prelude::*;
                self.layers
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(layer_idx, layer)| {
                        let num_slots_fill = num_slots_per_layer
                            .get(layer_idx)
                            .copied()
                            .unwrap_or(128)
                            .min(128)
                            .max(1);
                        let pred = &predicted_unions[layer_idx];
                        let w0: Vec<usize> = pred.iter().copied().take(num_slots_fill).collect();
                        let rest: Vec<usize> = pred.iter().copied().skip(num_slots_fill).collect();
                        fill_metal_wave_slots_from_host_cache(
                            layer,
                            cache,
                            layer_idx,
                            &w0,
                            &nvme_us,
                            &nvme_bytes,
                            &demand_loads,
                        );
                        if !rest.is_empty() {
                            let (buf, nslots) = if let Some(buf) = wave1_bufs.get(layer_idx) {
                                (buf, wave1_slot_count)
                            } else {
                                (&pong_slab_buf, pong_slot_count)
                            };
                            let mut updated_slots = [0xFFFFFFFFu32; 128];
                            fill_compact_wave_into_slots(
                                buf,
                                nslots,
                                cache,
                                layer_idx,
                                &rest,
                                &mut updated_slots,
                                &nvme_us,
                                &nvme_bytes,
                                &demand_loads,
                            );
                        }
                    });
            }
        }

        let layers_ref = &mut self.layers;
        let mut slot_filler = if let Some(cache) = ghost_cache {
            Some(
                |layer_idx: usize,
                 router_logits: &[f32],
                 wave: Option<&[usize]>,
                 updated_slots: &mut [u32; 128],
                 union_out: &mut Vec<usize>| {
                    let layer = &mut layers_ref[layer_idx];
                    let n_tokens = k_tokens.min(crate::metal::GEMMA4_RESIDENT_MAX_BATCH);
                    let n_slots = layer.directory.entries.len();

                    let selected_experts: Vec<usize> = if let Some(wave) = wave {
                        wave.to_vec()
                    } else {
                        let mut selected_experts = Vec::with_capacity(n_tokens * 8);
                        for t in 0..n_tokens {
                            let logits = &router_logits[t * 128..(t + 1) * 128];
                            let mut maxl = f32::MIN;
                            for &v in logits {
                                if v > maxl {
                                    maxl = v;
                                }
                            }
                            let mut probs = [0.0f32; 128];
                            for e in 0..128 {
                                probs[e] = (logits[e] - maxl).exp();
                            }
                            let mut ranked: [usize; 128] = std::array::from_fn(|i| i);
                            ranked.sort_unstable_by(|&a, &b| {
                                probs[b]
                                    .partial_cmp(&probs[a])
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            for i in 0..8 {
                                let e = ranked[i];
                                if !selected_experts.contains(&e) {
                                    selected_experts.push(e);
                                }
                            }
                        }
                        union_out.clear();
                        union_out.extend_from_slice(&selected_experts);
                        collected_routes[layer_idx] = selected_experts.clone();
                        if selected_experts.len() > n_slots {
                            // Overflow becomes extra waves. Keep any speculative
                            // wave-0 slot table filled during the router wait.
                            return;
                        }
                        selected_experts
                    };

                    fill_metal_wave_slots_from_host_cache(
                        layer,
                        cache,
                        layer_idx,
                        &selected_experts,
                        &nvme_us,
                        &nvme_bytes,
                        &demand_loads,
                    );

                    for e in 0..128 {
                        let s = layer.directory.resident_slot_table[e];
                        updated_slots[e] = if s >= 0 { s as u32 } else { 0xFFFFFFFFu32 };
                    }
                },
            )
        } else {
            None
        };

        let pong_slab_for_gpu = pong_slab_buf.clone();
        let mut fill_pong = ghost_cache.map(|cache| {
            |layer_idx: usize, wave: &[usize], updated_slots: &mut [u32; 128]| {
                let (buf, nslots) = if let Some(buf) = wave1_bufs.get(layer_idx) {
                    (buf, wave1_slot_count)
                } else {
                    (&pong_slab_buf, pong_slot_count)
                };
                fill_compact_wave_into_slots(
                    buf,
                    nslots,
                    cache,
                    layer_idx,
                    wave,
                    updated_slots,
                    &nvme_us,
                    &nvme_bytes,
                    &demand_loads,
                );
            }
        });

        let mut slot_filler_fn: Option<
            &mut dyn FnMut(usize, &[f32], Option<&[usize]>, &mut [u32; 128], &mut Vec<usize>),
        > = match slot_filler.as_mut() {
            Some(f) => Some(f),
            None => None,
        };
        let mut fill_pong_fn: Option<&mut dyn FnMut(usize, &[usize], &mut [u32; 128])> =
            match fill_pong.as_mut() {
                Some(f) => Some(f),
                None => None,
            };

        let expert_slab_refs: Vec<&metal::Buffer> = expert_slabs.iter().collect();
        let wave1_refs: Vec<&metal::Buffer> = wave1_gpu.iter().collect();
        let slot_mapping_slices: Vec<&[u32; 128]> = slot_mappings.iter().collect();
        let ok = {
            let Some(common) = self.common.as_mut() else {
                return false;
            };
            common.update_resident_slot_tables(&slot_mapping_slices);
            common.execute_chained_round_all_layers(
                hidden_rows,
                start_pos,
                theta_local,
                theta_global,
                rope_factors,
                &expert_slab_refs,
                &num_slots_per_layer,
                out_rows,
                slot_filler_fn,
                Some(&pong_slab_for_gpu),
                &predicted_unions,
                fill_pong_fn,
                &wave1_refs,
            )
        };
        if !ok && allow_predicted {
            eprintln!(
                "[gemma4-ghost-metal] predicted chained round refused at start_pos={start_pos} K={k_tokens}; retrying without route prediction"
            );
            self.suppress_prediction = true;
            let retry = self.execute_chained_round_all_layers(
                hidden_rows,
                theta_local,
                theta_global,
                rope_factors,
                start_pos,
                ghost_cache,
                out_rows,
            );
            self.suppress_prediction = false;
            return retry;
        }
        if ok {
            self.last_chained_k = Some(k_tokens);
            self.last_chained_sig = Some(chunk_sig);
        }
        if let Some(common) = self.common.as_mut() {
            common.last_chained_ledger.nvme_ms =
                nvme_us.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
            common.last_chained_ledger.nvme_bytes =
                nvme_bytes.load(std::sync::atomic::Ordering::Relaxed);
            common.last_chained_ledger.demand_loads =
                demand_loads.load(std::sync::atomic::Ordering::Relaxed);
            common.last_chained_ledger.prefetch_ms = prefetch_ms;
            common.last_chained_ledger.setup_ms = setup_ms;
            if ghost_metal_timing_enabled() {
                let led = &common.last_chained_ledger;
                eprintln!(
                    "[metal chained ledger] start_pos={start_pos} K={k_tokens} ok={ok} predicted={allow_predicted} slot_wait={:.1}ms slot_filler={:.1}ms wave_load={:.1}ms final_wait={:.1}ms encode={:.1}ms gpu_busy(last_cb)={:.1}ms disk_loads={} disk_bytes={:.1}MiB disk_time={:.1}ms unique={}",
                    led.slot_wait_ms,
                    led.slot_filler_ms,
                    led.wave_load_ms,
                    led.final_wait_ms,
                    led.encode_ms,
                    led.gpu_busy_ms,
                    led.demand_loads,
                    led.nvme_bytes as f64 / (1024.0 * 1024.0),
                    led.nvme_ms,
                    led.unique_experts_sum,
                );
            }
        }
        for (i, experts) in collected_routes.into_iter().enumerate() {
            if !experts.is_empty() {
                self.record_layer_routes(i, &experts);
                if let Some(history) = self.latest_routed_experts.get_mut(i) {
                    *history = experts;
                }
            }
        }
        ok
    }

    fn truncate_sequence(&mut self, keep: usize) {
        if let Some(common) = self.common.as_mut() {
            common.truncate_sequence(keep);
        }
    }

    fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.slots.slot_count() * layer.slots.slot_stride_bytes())
            .sum()
    }

    fn slots_per_layer(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.slots.slot_count())
    }

    fn slot_stats(&self) -> GhostMetalSlotStats {
        self.layers
            .iter()
            .fold(GhostMetalSlotStats::default(), |mut total, layer| {
                total.add_assign(layer.stats);
                total
            })
    }

    /// Pre-warm the persistent pinned hot slots across all layers directly from the .cghost file
    /// during initialization so that initial decode steps have zero cold slot misses on hot experts.
    fn prewarm_hot_slots_direct(&mut self, ghost_file: &GhostFile) {
        let t0 = std::time::Instant::now();
        let mut total_prewarmed = 0usize;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Cold start: every slot is empty, so seeding each one costs a
            // read that would otherwise be paid as a decode-time miss. Slots
            // are seeded with experts 0..slot_count (no hotness is known yet);
            // plain LFU/LRU replaces them with the real hot set as routing
            // evidence accumulates. Independent of `pinned_hot_slots`, which
            // is an eviction policy, not a warm-up policy.
            let hot_count = layer.slots.slot_count().min(128);
            for slot in 0..hot_count {
                let expert = slot;
                if let Some(dest) = layer.slots.slot_bytes_mut(slot) {
                    if ghost_file
                        .read_moe_expert_into(layer_idx, expert, dest)
                        .is_ok()
                    {
                        // Frequency 1: prewarm is a cold-start guess, not
                        // routing evidence. Seeding at 50 made stale guessed
                        // experts out-rank live ones under LFU for ~200 tokens
                        // (halving every 256 accesses), pinning the measured
                        // hit rate at ~93% instead of the ~97.5% the routing
                        // distribution supports at 80 slots.
                        layer.directory.commit_load(GhostMetalSlotLoad {
                            slot,
                            expert,
                            frequency: 1,
                            last_used: 0,
                        });
                        total_prewarmed += 1;
                    }
                }
            }
        }
        if total_prewarmed > 0 {
            eprintln!(
                "[gemma4-ghost-metal] pre-warmed {} hot expert slots across {} layers in {:.1}ms (resident)",
                total_prewarmed,
                self.layers.len(),
                t0.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    /// Seed a layer's persistent slots from immutable expert records already
    /// fetched for chunked prompt prefill. `request_sequence` contains only the
    /// selected bounded working set but retains every occurrence in prompt route
    /// order, so the directory learns real frequency/recency rather than an
    /// arbitrary expert-ID order. This is a host-memory copy, never disk I/O.
    fn prewarm_layer_from_records(
        &mut self,
        layer_idx: usize,
        request_sequence: &[usize],
        records: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> bool {
        if request_sequence.is_empty() {
            return true;
        }
        let Some(layer) = self.layers.get_mut(layer_idx) else {
            return false;
        };
        let expected_bytes = layer.slots.slot_record_bytes();
        let sources_valid = request_sequence
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .all(|expert| {
                records
                    .get(&expert)
                    .is_some_and(|record| record.byte_len() == expected_bytes)
            });
        if !sources_valid {
            // Preflight before `plan`: bad/missing prompt metadata must not
            // invalidate an otherwise usable resident slot.
            return false;
        }
        let plan = match layer.directory.plan(request_sequence) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!("[gemma4-ghost-metal] prompt slot plan failed: {err}");
                return false;
            }
        };
        let started = std::time::Instant::now();
        let mut copied = 0usize;
        for load in plan.loads {
            let Some(expert) = records.get(&load.expert) else {
                return false;
            };
            let (bytes, _) = expert.tensor_backing(&expert.gate_up);
            if bytes.len() != layer.slots.slot_record_bytes() {
                return false;
            }
            let Some(destination) = layer.slots.slot_bytes_mut(load.slot) else {
                return false;
            };
            destination.copy_from_slice(&bytes);
            layer.directory.commit_load(load);
            copied += 1;
        }
        layer.stats.prewarm_copies = layer.stats.prewarm_copies.saturating_add(copied as u64);
        if copied > 0 && ghost_metal_timing_enabled() {
            eprintln!(
                "[gemma4-ghost-metal-fill] layer={layer_idx} prompt={} disk=0 bytes={:.2}MiB wall={}us",
                copied,
                copied * layer.slots.slot_record_bytes() / (1024 * 1024),
                started.elapsed().as_micros(),
            );
        }
        true
    }

    /// Fill this layer's missing fixed slots directly from `.cghost`. The read
    /// pool sees disjoint mutable chunks of one shared Metal slab, so up to eight
    /// cache misses become concurrent positioned reads with no intermediate copy.
    fn prepare_layer_routes(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> Option<[crate::metal::Gemma4Q4ExpertRoute; 8]> {
        if experts.len() != 8 || route_scales.len() != 8 {
            return None;
        }
        let layer = self.layers.get_mut(ghost.layer_idx)?;
        let plan = match layer.directory.plan(experts) {
            Ok(plan) => plan,
            Err(err) => {
                eprintln!("[gemma4-ghost-metal] slot plan failed: {err}");
                return None;
            }
        };
        let GhostMetalSlotPlan {
            route_slots,
            loads,
            hits,
            evictions,
        } = plan;
        layer.stats.route_lookups = layer
            .stats
            .route_lookups
            .saturating_add(experts.len() as u64);
        layer.stats.hits = layer.stats.hits.saturating_add(hits as u64);
        layer.stats.misses = layer.stats.misses.saturating_add(loads.len() as u64);
        layer.stats.evictions = layer.stats.evictions.saturating_add(evictions as u64);

        if !loads.is_empty() {
            let fill_started = std::time::Instant::now();
            let stride = layer.slots.slot_stride_bytes();
            let record_bytes = layer.slots.slot_record_bytes();
            debug_assert_eq!(record_bytes, crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES);
            let file = &ghost.cache.file;
            let mut host_fills = 0usize;
            let mut disk_loads = Vec::with_capacity(loads.len());
            for load in loads.iter().copied() {
                let Some((bytes, _)) = resident_sources
                    .get(&load.expert)
                    .map(|expert| expert.tensor_backing(&expert.gate_up))
                else {
                    disk_loads.push(load);
                    continue;
                };
                if bytes.len() != record_bytes {
                    disk_loads.push(load);
                    continue;
                }
                let Some(destination) = layer.slots.slot_bytes_mut(load.slot) else {
                    eprintln!(
                        "[gemma4-ghost-metal] host-cache fill selected invalid slot {}",
                        load.slot
                    );
                    return None;
                };
                destination.copy_from_slice(&bytes);
                layer.directory.commit_load(load);
                host_fills += 1;
            }
            layer.stats.host_fills = layer.stats.host_fills.saturating_add(host_fills as u64);

            let results: Vec<(GhostMetalSlotLoad, Result<()>)> = if disk_loads.is_empty() {
                Vec::new()
            } else if disk_loads.len() == 1 {
                let load = disk_loads[0];
                let result = layer
                    .slots
                    .slot_bytes_mut(load.slot)
                    .ok_or_else(|| {
                        BackendError::InvalidModelMetadata(format!(
                            "Ghost Metal slot {} is outside the layer slab",
                            load.slot
                        ))
                    })
                    .and_then(|destination| {
                        file.read_moe_expert_into(ghost.layer_idx, load.expert, destination)
                    });
                vec![(load, result)]
            } else if let Some(pool) = &ghost.cache.read_pool {
                let slots_ref = &layer.slots;
                pool.install(|| {
                    disk_loads
                        .par_iter()
                        .map(|&load| {
                            let res = match unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                                Some(destination) => file.read_moe_expert_into(
                                    ghost.layer_idx,
                                    load.expert,
                                    destination,
                                ),
                                None => Err(BackendError::InvalidModelMetadata(format!(
                                    "Ghost Metal slot {} is outside the layer slab",
                                    load.slot
                                ))),
                            };
                            (load, res)
                        })
                        .collect()
                })
            } else {
                disk_loads
                    .iter()
                    .map(|&load| {
                        let res = match layer.slots.slot_bytes_mut(load.slot) {
                            Some(destination) => {
                                file.read_moe_expert_into(ghost.layer_idx, load.expert, destination)
                            }
                            None => Err(BackendError::InvalidModelMetadata(format!(
                                "Ghost Metal slot {} is outside the layer slab",
                                load.slot
                            ))),
                        };
                        (load, res)
                    })
                    .collect()
            };

            let mut all_loaded = results.len() == disk_loads.len();
            let mut direct_reads = 0usize;
            let mut direct_read_failures = disk_loads.len().saturating_sub(results.len());
            for (load, result) in results {
                match result {
                    Ok(()) => {
                        layer.directory.commit_load(load);
                        direct_reads += 1;
                    }
                    Err(err) => {
                        all_loaded = false;
                        direct_read_failures += 1;
                        eprintln!(
                            "[gemma4-ghost-metal] layer {} expert {} direct slot read failed: {err}",
                            ghost.layer_idx, load.expert
                        );
                    }
                }
            }
            layer.stats.direct_reads = layer.stats.direct_reads.saturating_add(direct_reads as u64);
            layer.stats.direct_read_bytes = layer
                .stats
                .direct_read_bytes
                .saturating_add((direct_reads as u64).saturating_mul(record_bytes as u64));
            layer.stats.direct_read_failures = layer
                .stats
                .direct_read_failures
                .saturating_add(direct_read_failures as u64);
            if !all_loaded {
                return None;
            }
            if ghost_metal_timing_enabled() {
                eprintln!(
                    "[gemma4-ghost-metal-fill] layer={} host={} disk={} bytes={:.2}MiB wall={}us",
                    ghost.layer_idx,
                    host_fills,
                    disk_loads.len(),
                    loads.len() * record_bytes / (1024 * 1024),
                    fill_started.elapsed().as_micros(),
                );
            }
        }

        // Debug: verify each routed slot actually holds its expert's bytes
        // (first 16 bytes vs a fresh positioned read from the .cghost).
        if std::env::var("CAMELID_GEMMA4_VERIFY_SLOTS").is_ok_and(|v| v == "1") {
            let layer = self.layers.get_mut(ghost.layer_idx)?;
            for (rank, (&slot, &expert)) in route_slots.iter().zip(experts.iter()).enumerate() {
                let mut expected = vec![0u8; crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES];
                if ghost
                    .cache
                    .file
                    .read_moe_expert_into(ghost.layer_idx, expert, &mut expected)
                    .is_ok()
                {
                    if let Some(actual) = layer.slots.slot_bytes_mut(slot) {
                        let ok_head = actual[..16] == expected[..16];
                        let ok_tail = actual[crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES - 16..]
                            == expected[crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES - 16..];
                        if !ok_head || !ok_tail {
                            eprintln!(
                                "[slot-verify] layer {} rank {rank} expert {expert} slot {slot} MISMATCH head_ok={ok_head} tail_ok={ok_tail} slot[0..8]={:02x?} want[0..8]={:02x?}",
                                ghost.layer_idx, &actual[..8], &expected[..8]
                            );
                        } else if ghost.layer_idx == 0 {
                            eprintln!(
                                "[slot-verify] layer 0 rank {rank} expert {expert} slot {slot} ok scale={:.6}",
                                route_scales[rank]
                            );
                        }
                    }
                }
            }
        }
        Some(std::array::from_fn(|rank| {
            crate::metal::Gemma4Q4ExpertRoute {
                slot: route_slots[rank],
                scale: route_scales[rank],
            }
        }))
    }

    /// Asynchronously prefetch previous-round routed experts across all layers in parallel.
    /// This runs on background Rayon workers during drafter execution so that upcoming rounds
    /// hit resident Metal slots with zero exposed I/O latency.
    pub(crate) fn prefetch_temporal_routes(
        &mut self,
        file: &GhostFile,
        layer_routes: &[Vec<usize>],
        pool: &Option<rayon::ThreadPool>,
    ) {
        if layer_routes.is_empty() {
            return;
        }
        let prefetch_tasks: Vec<(usize, Vec<GhostMetalSlotLoad>)> = self
            .layers
            .iter_mut()
            .enumerate()
            .filter_map(|(layer_idx, layer)| {
                let routes = layer_routes.get(layer_idx)?;
                if routes.is_empty() {
                    return None;
                }
                let missing: Vec<usize> = routes
                    .iter()
                    .copied()
                    .filter(|&e| layer.directory.lookup_resident_slot(e).is_none())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                if missing.is_empty() {
                    return None;
                }
                let capped_missing: Vec<usize> =
                    missing.into_iter().take(layer.slots.slot_count()).collect();
                match layer.directory.plan(&capped_missing) {
                    Ok(plan) => {
                        if plan.loads.is_empty() {
                            None
                        } else {
                            Some((layer_idx, plan.loads))
                        }
                    }
                    Err(_) => None,
                }
            })
            .collect();

        if prefetch_tasks.is_empty() {
            return;
        }

        // Track per-load success: committing a failed read would mark a slot
        // resident with stale bytes, and every later route to that expert
        // would silently use them.
        let read_ok = |(layer_idx, loads): &(usize, Vec<GhostMetalSlotLoad>)| -> Vec<bool> {
            let layer = &self.layers[*layer_idx];
            let slots_ref = &layer.slots;
            loads
                .iter()
                .map(
                    |load| match unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                        Some(destination) => file
                            .read_moe_expert_into(*layer_idx, load.expert, destination)
                            .is_ok(),
                        None => false,
                    },
                )
                .collect()
        };

        let outcomes: Vec<Vec<bool>> = if let Some(pool) = pool {
            pool.install(|| prefetch_tasks.par_iter().map(read_ok).collect())
        } else {
            prefetch_tasks.iter().map(read_ok).collect()
        };

        for ((layer_idx, loads), oks) in prefetch_tasks.into_iter().zip(outcomes) {
            let layer = &mut self.layers[layer_idx];
            for (load, ok) in loads.into_iter().zip(oks) {
                if ok {
                    layer.directory.commit_load(load);
                }
            }
        }
    }

    /// Round-wide global asynchronous prefetch: merges previous round's routes with newly
    /// predicted candidate token routes across all 30 layers and loads all missing experts in
    /// parallel via Rayon before the forward pass verification begins.
    pub(crate) fn prefetch_round_wide_async(
        &mut self,
        predicted_routes_per_layer: &[Vec<usize>],
        file: &GhostFile,
        pool: &Option<rayon::ThreadPool>,
    ) {
        let mut union_routes: Vec<Vec<usize>> = Vec::with_capacity(self.layers.len());
        for layer_idx in 0..self.layers.len() {
            let mut set = std::collections::HashSet::new();
            if let Some(predicted) = predicted_routes_per_layer.get(layer_idx) {
                for &e in predicted {
                    set.insert(e);
                }
            }
            if set.is_empty() {
                if let Some(history) = self.latest_routed_experts.get(layer_idx) {
                    for &e in history {
                        set.insert(e);
                    }
                }
                for e in self.top_speculative_candidates(layer_idx, 8) {
                    set.insert(e);
                }
            }
            union_routes.push(set.into_iter().collect());
        }
        self.prefetch_temporal_routes(file, &union_routes, pool);
    }

    /// Pre-fetch temporal routes using the latest recorded routed experts from the prior round.
    pub(crate) fn prefetch_temporal_last_round(
        &mut self,
        file: &GhostFile,
        pool: &Option<rayon::ThreadPool>,
    ) {
        let routes = self.latest_routed_experts.clone();
        self.prefetch_temporal_routes(file, &routes, pool);
    }

    /// Asynchronously pre-fetch the next layer's missing experts during current layer execution.
    pub(crate) fn prefetch_next_layer_async(
        &mut self,
        current_layer: usize,
        file: &GhostFile,
        pool: &Option<rayon::ThreadPool>,
    ) {
        let next_layer = current_layer + 1;
        if next_layer >= self.layers.len() {
            return;
        }
        let routes = match self.latest_routed_experts.get(next_layer) {
            Some(r) if !r.is_empty() => r.clone(),
            _ => return,
        };
        let layer = &mut self.layers[next_layer];
        let missing: Vec<usize> = routes
            .iter()
            .copied()
            .filter(|&e| layer.directory.lookup_resident_slot(e).is_none())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        if missing.is_empty() {
            return;
        }
        if let Ok(plan) = layer.directory.plan(&missing) {
            if plan.loads.is_empty() {
                return;
            }
            let slots_ref = &layer.slots;
            let perform_read = |load: &GhostMetalSlotLoad| {
                if let Some(dest) = unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                    let _ = file.read_moe_expert_into(next_layer, load.expert, dest);
                }
            };
            if let Some(pool) = pool {
                pool.install(|| {
                    plan.loads.par_iter().for_each(perform_read);
                });
            } else {
                plan.loads.iter().for_each(perform_read);
            }
            for load in plan.loads {
                layer.directory.commit_load(load);
            }
        }
    }

    /// Asynchronously pre-fetch a specific target layer's predicted missing experts.
    pub(crate) fn prefetch_predicted_layer_routes(
        &mut self,
        layer_idx: usize,
        predicted_experts: &[usize],
        file: &GhostFile,
        pool: &Option<rayon::ThreadPool>,
    ) {
        if predicted_experts.is_empty() {
            return;
        }
        let Some(layer) = self.layers.get_mut(layer_idx) else {
            return;
        };
        let missing: Vec<usize> = predicted_experts
            .iter()
            .copied()
            .filter(|&e| layer.directory.lookup_resident_slot(e).is_none())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        if missing.is_empty() {
            return;
        }
        if let Ok(plan) = layer.directory.plan(&missing) {
            if plan.loads.is_empty() {
                return;
            }
            let slots_ref = &layer.slots;
            let perform_read = |load: &GhostMetalSlotLoad| {
                if let Some(dest) = unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                    let _ = file.read_moe_expert_into(layer_idx, load.expert, dest);
                }
            };
            if let Some(pool) = pool {
                pool.install(|| {
                    plan.loads.par_iter().for_each(perform_read);
                });
            } else {
                plan.loads.iter().for_each(perform_read);
            }
            for load in plan.loads {
                layer.directory.commit_load(load);
            }
        }
    }

    #[inline(always)]
    fn try_resolve_resident_chunk(
        &mut self,
        layer_idx: usize,
        unique_experts: &[usize],
    ) -> Option<(&metal::Buffer, Vec<usize>)> {
        if let Some(history) = self.latest_routed_experts.get_mut(layer_idx) {
            history.clear();
            history.extend_from_slice(unique_experts);
        }
        let layer = self.layers.get_mut(layer_idx)?;
        let mut slot_indices = Vec::with_capacity(unique_experts.len());
        for &e in unique_experts {
            let slot = layer.directory.lookup_resident_slot(e)?;
            slot_indices.push(slot);
        }
        layer.stats.route_lookups = layer
            .stats
            .route_lookups
            .saturating_add(unique_experts.len() as u64);
        layer.stats.hits = layer.stats.hits.saturating_add(unique_experts.len() as u64);
        Some((layer.slots.slab_buffer(), slot_indices))
    }

    fn prepare_chunk_slots(
        &mut self,
        ghost: &GhostMoeLayer,
        unique_experts: &[usize],
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> Option<(&metal::Buffer, Vec<usize>)> {
        if let Some(history) = self.latest_routed_experts.get_mut(ghost.layer_idx) {
            history.clear();
            history.extend_from_slice(unique_experts);
        }
        let layer = self.layers.get_mut(ghost.layer_idx)?;
        let plan = match layer.directory.plan(unique_experts) {
            Ok(plan) => plan,
            Err(_) => return None,
        };
        let GhostMetalSlotPlan {
            route_slots,
            loads,
            hits,
            evictions,
        } = plan;
        layer.stats.route_lookups = layer
            .stats
            .route_lookups
            .saturating_add(unique_experts.len() as u64);
        layer.stats.hits = layer.stats.hits.saturating_add(hits as u64);
        layer.stats.misses = layer.stats.misses.saturating_add(loads.len() as u64);
        layer.stats.evictions = layer.stats.evictions.saturating_add(evictions as u64);

        if !loads.is_empty() {
            let stride = layer.slots.slot_stride_bytes();
            let record_bytes = layer.slots.slot_record_bytes();
            let file = &ghost.cache.file;
            let mut host_fills = 0usize;
            let mut disk_loads = Vec::with_capacity(loads.len());
            for load in loads.iter().copied() {
                let Some((bytes, _)) = resident_sources
                    .get(&load.expert)
                    .map(|expert| expert.tensor_backing(&expert.gate_up))
                else {
                    disk_loads.push(load);
                    continue;
                };
                if bytes.len() != record_bytes {
                    disk_loads.push(load);
                    continue;
                }
                let Some(destination) = layer.slots.slot_bytes_mut(load.slot) else {
                    return None;
                };
                destination.copy_from_slice(&bytes);
                layer.directory.commit_load(load);
                host_fills += 1;
            }
            layer.stats.host_fills = layer.stats.host_fills.saturating_add(host_fills as u64);

            let results: Vec<(GhostMetalSlotLoad, Result<()>)> = if disk_loads.is_empty() {
                Vec::new()
            } else if disk_loads.len() == 1 {
                let load = disk_loads[0];
                let result = layer
                    .slots
                    .slot_bytes_mut(load.slot)
                    .ok_or_else(|| {
                        BackendError::InvalidModelMetadata(format!(
                            "Ghost Metal slot {} is outside the layer slab",
                            load.slot
                        ))
                    })
                    .and_then(|destination| {
                        file.read_moe_expert_into(ghost.layer_idx, load.expert, destination)
                    });
                vec![(load, result)]
            } else if let Some(pool) = &ghost.cache.read_pool {
                let slots_ref = &layer.slots;
                pool.install(|| {
                    disk_loads
                        .par_iter()
                        .map(|&load| {
                            let res = match unsafe { slots_ref.slot_bytes_mut_raw(load.slot) } {
                                Some(destination) => file.read_moe_expert_into(
                                    ghost.layer_idx,
                                    load.expert,
                                    destination,
                                ),
                                None => Err(BackendError::InvalidModelMetadata(format!(
                                    "Ghost Metal slot {} is outside the layer slab",
                                    load.slot
                                ))),
                            };
                            (load, res)
                        })
                        .collect()
                })
            } else {
                disk_loads
                    .iter()
                    .map(|&load| {
                        let res = match layer.slots.slot_bytes_mut(load.slot) {
                            Some(destination) => {
                                file.read_moe_expert_into(ghost.layer_idx, load.expert, destination)
                            }
                            None => Err(BackendError::InvalidModelMetadata(format!(
                                "Ghost Metal slot {} is outside the layer slab",
                                load.slot
                            ))),
                        };
                        (load, res)
                    })
                    .collect()
            };

            let mut all_loaded = results.len() == disk_loads.len();
            let mut direct_reads = 0usize;
            let mut direct_read_failures = disk_loads.len().saturating_sub(results.len());
            for (load, result) in results {
                match result {
                    Ok(()) => {
                        layer.directory.commit_load(load);
                        direct_reads += 1;
                    }
                    Err(_) => {
                        all_loaded = false;
                        direct_read_failures += 1;
                    }
                }
            }
            layer.stats.direct_reads = layer.stats.direct_reads.saturating_add(direct_reads as u64);
            layer.stats.direct_read_bytes = layer
                .stats
                .direct_read_bytes
                .saturating_add((direct_reads as u64).saturating_mul(record_bytes as u64));
            layer.stats.direct_read_failures = layer
                .stats
                .direct_read_failures
                .saturating_add(direct_read_failures as u64);
            if !all_loaded {
                return None;
            }
        }

        let slab_buffer = layer.slots.slab_buffer();
        Some((slab_buffer, route_slots))
    }

    /// Host-activation compatibility wrapper used by the established CPU
    /// common-core lane.
    fn run_layer(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        input: &[Q8_0Block],
        hidden: usize,
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> GhostMetalExpertAttempt {
        let Some(routes) =
            self.prepare_layer_routes(ghost, experts, route_scales, resident_sources)
        else {
            return GhostMetalExpertAttempt::CpuFallback;
        };
        let Some(layer) = self.layers.get(ghost.layer_idx) else {
            return GhostMetalExpertAttempt::CpuFallback;
        };
        let mut output = vec![0.0f32; hidden];
        let diagnostics = if self.fused_fast {
            self.engine
                .run_q8_into(input, &layer.slots, &routes, &mut output)
        } else {
            self.engine
                .run_q8_into_parity(input, &layer.slots, &routes, &mut output)
        };
        match diagnostics {
            Some(_) => GhostMetalExpertAttempt::Output(output),
            None => GhostMetalExpertAttempt::DisableMetal,
        }
    }

    /// Pure device-chain wrapper used by the persistent common core. Slot I/O
    /// runs while the already-enqueued shared branch consumes Metal bandwidth;
    /// expert reduce and the MoE tail then execute in queue order.
    fn run_layer_common(
        &mut self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        resident_sources: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) -> GhostMetalCommonAttempt {
        let Some(routes) =
            self.prepare_layer_routes(ghost, experts, route_scales, resident_sources)
        else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        let Some(layer) = self.layers.get(ghost.layer_idx) else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        let Some(common) = self.common.as_mut() else {
            return GhostMetalCommonAttempt::CpuFallback;
        };
        if self.fused_fast {
            match self.engine.enqueue_common_with_tail(
                common,
                ghost.layer_idx,
                &layer.slots,
                &routes,
            ) {
                Some(pending) => GhostMetalCommonAttempt::Pending(pending),
                None => GhostMetalCommonAttempt::DisableMetal,
            }
        } else {
            match self.engine.run_common_with_tail(
                common,
                ghost.layer_idx,
                &layer.slots,
                &routes,
                false,
            ) {
                Some(_) => GhostMetalCommonAttempt::Complete,
                None => GhostMetalCommonAttempt::DisableMetal,
            }
        }
    }
}

/// Timing-gated request delta reporter. Generation already serializes the
/// persistent common-core lane, so a start/end snapshot is sufficient and
/// costs only two short mutex acquisitions outside the layer hot path.
#[cfg(target_os = "macos")]
struct GhostMetalGenerationStatsGuard<'a> {
    lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
    start: Option<(GhostMetalSlotStats, usize, usize)>,
    started: std::time::Instant,
}

#[cfg(target_os = "macos")]
impl<'a> GhostMetalGenerationStatsGuard<'a> {
    fn new(lane: &'a std::sync::Mutex<Option<GhostMetalExpertRuntime>>) -> Self {
        let start = ghost_metal_stats_enabled()
            .then(|| {
                lane.lock().ok().and_then(|runtime| {
                    runtime.as_ref().map(|runtime| {
                        (
                            runtime.slot_stats(),
                            runtime.layers.len(),
                            runtime.slots_per_layer(),
                        )
                    })
                })
            })
            .flatten();
        Self {
            lane,
            start,
            started: std::time::Instant::now(),
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for GhostMetalGenerationStatsGuard<'_> {
    fn drop(&mut self) {
        let Some((start, layer_count, slots_per_layer)) = self.start else {
            return;
        };
        let Some(end) = self
            .lane
            .lock()
            .ok()
            .and_then(|runtime| runtime.as_ref().map(GhostMetalExpertRuntime::slot_stats))
        else {
            eprintln!(
                "[gemma4-ghost-metal-summary] slots lane became unavailable during generation"
            );
            return;
        };
        let delta = end.saturating_delta(start);
        let requests = delta.hits.saturating_add(delta.misses);
        let hit_rate = if requests == 0 {
            0.0
        } else {
            100.0 * delta.hits as f64 / requests as f64
        };
        let routed_positions = if layer_count == 0 {
            0
        } else {
            delta.route_lookups / (layer_count as u64 * 8)
        };
        let direct_mib = delta.direct_read_bytes as f64 / (1024.0 * 1024.0);
        let direct_mib_per_position = if routed_positions == 0 {
            0.0
        } else {
            direct_mib / routed_positions as f64
        };
        eprintln!(
            "[gemma4-ghost-metal-summary] layers={layer_count} slots/layer={slots_per_layer} routed_positions={routed_positions} lookups={} hits={} misses={} hit_rate={hit_rate:.1}% evictions={} host_fills={} prewarm_copies={} direct_reads={} direct_read_bytes={direct_mib:.1}MiB direct_read_per_position={direct_mib_per_position:.1}MiB read_failures={} wall={:.1}ms",
            delta.route_lookups,
            delta.hits,
            delta.misses,
            delta.evictions,
            delta.host_fills,
            delta.prewarm_copies,
            delta.direct_reads,
            delta.direct_read_failures,
            self.started.elapsed().as_secs_f64() * 1_000.0,
        );
    }
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

/// One policy gate for every Ghost-MoE Metal dispatch. The CLI/UI GPU switch is
/// live, so this must be evaluated at each use rather than latched when the model
/// loads. Deterministic mode remains authoritative even if the runtime switch is
/// subsequently turned back on.
#[cfg(any(target_os = "macos", test))]
#[inline]
#[cfg(any(target_os = "macos", test))]
fn ghost_metal_acceleration_allowed(deterministic: bool, runtime_gpu_enabled: bool) -> bool {
    // An explicit CAMELID_GEMMA4_GHOST_METAL=0 must mean OFF everywhere: with
    // the GPU present, `runtime_gpu_enabled` defaults true and previously
    // overrode the env, so "pure CPU" parity runs silently mixed GPU chained
    // rounds into their trajectories. Unset keeps the serve default.
    if std::env::var("CAMELID_GEMMA4_GHOST_METAL").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    }) {
        return false;
    }
    !deterministic && runtime_gpu_enabled
}

#[cfg(target_os = "macos")]
#[inline]
#[cfg(target_os = "macos")]
fn ghost_metal_acceleration_enabled() -> bool {
    ghost_metal_acceleration_allowed(
        crate::inference::deterministic_mode_enabled(),
        crate::cuda::gpu_accel_enabled(),
    )
}

/// Run one disk-paged Q4_0 expert projection on Metal while preserving the
/// CPU Ghost-MoE Q4_0 x Q8_0 row-dot contract. The expert remains bounded by
/// the host cache: its wire bytes are copied into one transient shared Metal
/// buffer for this projection and are not retained in an unbounded GPU cache.
///
/// Opt in with `CAMELID_GEMMA4_GHOST_METAL=1`. It remains off by default until
/// a real 26B sweep proves that transient expert uploads and command-buffer
/// count beat the CPU lane; the longer-lived fixed-slot runtime is the target.
fn ghost_metal_q4_matmul(
    weight: &WireQuant,
    rows: usize,
    inputs: &[&[Q8_0Block]],
) -> Option<Vec<Vec<f32>>> {
    #[cfg(target_os = "macos")]
    {
        // Explicit CAMELID_GEMMA4_GHOST_METAL=0 must mean OFF: "pure CPU"
        // parity runs are meaningless if expert GEMMs silently stay on Metal.
        // Unset keeps the serve default (on when a GPU is present).
        let enabled = match std::env::var("CAMELID_GEMMA4_GHOST_METAL") {
            Ok(value) => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            ),
            Err(_) => crate::cuda::gpu_accel_enabled(),
        } && !crate::inference::deterministic_mode_enabled();
        if !enabled || !ghost_metal_acceleration_enabled() || weight.format != WireFormat::Q4_0 {
            return None;
        }
        let output = crate::metal::try_gemma4_q4_0_matmul_q8_batch(inputs, weight.bytes(), rows)?;
        static ANNOUNCED: std::sync::Once = std::sync::Once::new();
        ANNOUNCED.call_once(|| {
            eprintln!(
                "[ghost-moe-metal] ordered Q4_0 expert GEMMs active (Metal; CPU fallback retained)"
            );
        });
        Some(output)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (weight, rows, inputs);
        None
    }
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
    /// Present only on the v2 `.cghost` lane. The mmap-backed expert tensors
    /// remain untouched; selected experts come from this bounded global cache.
    ghost: Option<GhostMoeLayer>,
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

/// A v3 Ghost index binds the GGUF by cryptographic source identity, so a
/// human-facing filename is only diagnostic and may legitimately change after
/// the splice was created. Legacy indexes have no such binding and retain the
/// old basename guard as their only source-model check.
fn ghost_source_filename_admitted(
    has_source_identity: bool,
    declared_source_model: &str,
    actual_filename: Option<&str>,
) -> bool {
    has_source_identity
        || declared_source_model.is_empty()
        || actual_filename.is_none_or(|actual| declared_source_model == actual)
}

#[cfg(target_os = "macos")]
fn build_ghost_common_metal(
    path: &Path,
    store: &TensorStore,
    binding: &Gemma4Binding,
    config: &LlamaModelConfig,
    g: &Gemma4Metadata,
    layers: &[LayerWeights],
    max_positions: usize,
) -> Result<Option<crate::metal::Gemma4GhostCommonMetal>> {
    let mut refusals = Vec::new();
    let expect = |refusals: &mut Vec<String>, admitted: bool, detail: String| {
        if !admitted {
            refusals.push(detail);
        }
    };
    expect(
        &mut refusals,
        config.block_count as usize == 30,
        format!("block_count={} (expected 30)", config.block_count),
    );
    expect(
        &mut refusals,
        config.embedding_length as usize == 2_816,
        format!(
            "embedding_length={} (expected 2816)",
            config.embedding_length
        ),
    );
    expect(
        &mut refusals,
        config.attention_head_count as usize == 16,
        format!(
            "attention_head_count={} (expected 16)",
            config.attention_head_count
        ),
    );
    expect(
        &mut refusals,
        g.sliding_window as usize == 1_024,
        format!("sliding_window={} (expected 1024)", g.sliding_window),
    );
    expect(
        &mut refusals,
        g.num_kv_shared_layers == 0,
        format!(
            "num_kv_shared_layers={} (expected 0)",
            g.num_kv_shared_layers
        ),
    );
    expect(&mut refusals, max_positions > 0, "max_positions=0".into());
    expect(
        &mut refusals,
        layers.len() == 30,
        format!("loaded layer count={} (expected 30)", layers.len()),
    );
    expect(
        &mut refusals,
        binding.layers.len() == 30,
        format!("bound layer count={} (expected 30)", binding.layers.len()),
    );
    match config.moe.as_ref() {
        Some(moe) => {
            expect(
                &mut refusals,
                moe.expert_count as usize == 128,
                format!("expert_count={} (expected 128)", moe.expert_count),
            );
            expect(
                &mut refusals,
                moe.expert_used_count as usize == 8,
                format!("expert_used_count={} (expected 8)", moe.expert_used_count),
            );
        }
        None => refusals.push("MoE metadata is absent".into()),
    }
    for (layer_idx, layer) in layers.iter().enumerate() {
        let mut layer_refusals = Vec::new();
        if layer.ple_inp_gate.is_some() {
            layer_refusals.push("PLE input gate is present");
        }
        if layer.ple_proj.is_some() {
            layer_refusals.push("PLE projection is present");
        }
        if layer.post_norm.is_some() {
            layer_refusals.push("PLE post norm is present");
        }
        // Gemma 4 26B carries a learned scalar on every layer. It is not PLE:
        // the reference applies it unconditionally after the layer, and the
        // Metal tail uploads/applies this exact value in `configure_moe`.
        if !layer.ple_output_scale.is_finite() {
            layer_refusals.push("layer output scale is non-finite");
        }
        let require_q4 =
            |refusals: &mut Vec<&'static str>, name: &'static str, format: WireFormat| {
                if format != WireFormat::Q4_0 {
                    refusals.push(name);
                }
            };
        require_q4(
            &mut layer_refusals,
            "attn_q is not Q4_0",
            layer.attn_q.format,
        );
        match layer.attn_k.as_ref() {
            Some(weight) => require_q4(&mut layer_refusals, "attn_k is not Q4_0", weight.format),
            None => layer_refusals.push("attn_k is absent"),
        }
        if let Some(weight) = layer.attn_v.as_ref() {
            require_q4(&mut layer_refusals, "attn_v is not Q4_0", weight.format);
        }
        require_q4(
            &mut layer_refusals,
            "attn_output is not Q4_0",
            layer.attn_output.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_gate is not Q4_0",
            layer.ffn_gate.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_up is not Q4_0",
            layer.ffn_up.format,
        );
        require_q4(
            &mut layer_refusals,
            "ffn_down is not Q4_0",
            layer.ffn_down.format,
        );
        match layer.moe.as_ref() {
            Some(moe) => {
                if moe.n_expert != 128 {
                    layer_refusals.push("MoE expert count is not 128");
                }
                if moe.n_expert_used != 8 {
                    layer_refusals.push("MoE top-k is not 8");
                }
                if moe.n_ff_exp != 704 {
                    layer_refusals.push("MoE expert FF width is not 704");
                }
                if moe.ghost.is_none() {
                    layer_refusals.push("MoE weights are not Ghost-backed");
                }
            }
            None => layer_refusals.push("MoE weights are absent"),
        }
        if !layer_refusals.is_empty() {
            refusals.push(format!("layer {layer_idx}: {}", layer_refusals.join(", ")));
        }
    }
    if !refusals.is_empty() {
        for refusal in refusals {
            eprintln!("[gemma4-ghost-common] admission refused: {refusal}");
        }
        return Ok(None);
    }
    let file = std::fs::File::open(path).map_err(|source| BackendError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let pages = |name: &str| -> Result<Arc<crate::wire_mmap::WirePages>> {
        let descriptor = store.descriptor(name)?;
        crate::wire_mmap::WirePages::read_from_file(
            &file,
            descriptor.absolute_offset,
            descriptor.n_bytes as usize,
        )
    };

    let mut resident_layers = Vec::with_capacity(layers.len());
    let mut post_norm_1 = Vec::with_capacity(layers.len());
    let mut moe_configs = Vec::with_capacity(layers.len());
    for (layer_idx, (layer, bound)) in layers.iter().zip(&binding.layers).enumerate() {
        let k_descriptor = bound.attn_k.as_ref().ok_or_else(|| {
            BackendError::InvalidModelMetadata(format!(
                "Ghost common Metal layer {layer_idx} omits attn_k"
            ))
        })?;
        let q_pages = pages(&bound.attn_q.name)?;
        let k_pages = pages(&k_descriptor.name)?;
        let v_pages = bound
            .attn_v
            .as_ref()
            .map(|descriptor| pages(&descriptor.name))
            .transpose()?;
        let resident = crate::metal::Gemma4ResidentLayer::from_wire_pages_owned(
            crate::metal::GemmaWireFmt::Q4_0,
            layer.attn_norm.clone(),
            layer.q_norm.clone(),
            layer.k_norm.clone().ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!(
                    "Ghost common Metal layer {layer_idx} omits attn_k_norm"
                ))
            })?,
            layer.post_attn_norm.clone(),
            layer.ffn_norm.clone(),
            layer.post_ffw_norm.clone(),
            &q_pages,
            &k_pages,
            v_pages.as_ref(),
            &pages(&bound.attn_output.name)?,
            &pages(&bound.ffn_gate.name)?,
            &pages(&bound.ffn_up.name)?,
            &pages(&bound.ffn_down.name)?,
            config.attention_head_count as usize,
            g.kv_heads_at(layer_idx) as usize,
            g.head_dim_at(layer_idx) as usize,
            g.ffn_length_at(layer_idx) as usize,
            config.rms_norm_epsilon,
        )
        .ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture(
                "Metal unavailable while constructing Ghost common core".into(),
            )
        })?;
        let moe = layer
            .moe
            .as_ref()
            .expect("exact Ghost common preflight requires MoE on every layer");
        resident_layers.push(resident);
        post_norm_1.push(moe.post_norm_1.clone());
        moe_configs.push(crate::metal::Gemma4GhostMoeLayerConfig {
            router: moe.gate_inp.clone(),
            gate_input_scale: moe.gate_inp_scale.clone(),
            pre_norm_2: moe.pre_norm_2.clone(),
            post_norm_2: moe.post_norm_2.clone(),
            down_exps_scale: moe.down_exps_scale.clone(),
            layer_output_scale: layer.ple_output_scale,
        });
    }
    let Some(mut common) =
        crate::metal::Gemma4GhostCommonMetal::new_26b(resident_layers, post_norm_1, max_positions)
    else {
        return Ok(None);
    };
    if !common.configure_moe(moe_configs) {
        return Ok(None);
    }
    Ok(Some(common))
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
    ghost_moe_cache: Option<Arc<GhostMoeExpertCache>>,
    /// Opt-in disk-paged Q4_0 expert engine. One reusable Metal executor serves
    /// a bounded, load-time-configured set of 16-KiB-aligned slots per layer.
    /// The inner `Option` is
    /// cleared after a Metal command failure so the established CPU Ghost lane
    /// remains the permanent fallback for the rest of the session.
    #[cfg(target_os = "macos")]
    metal_q4_experts: std::sync::Mutex<Option<GhostMetalExpertRuntime>>,
    /// The common-core KV cache is model-owned. Hold this for a complete public
    /// generation request so two callers cannot interleave position-zero resets
    /// and token steps on the same persistent Metal buffers.
    #[cfg(target_os = "macos")]
    ghost_common_generation: std::sync::Mutex<()>,
    /// Ghost-MoE keeps the decoder math on the correctness-first CPU lane for
    /// now, but the 605 MB Q6_K tied output table is already covered by Camelid's
    /// parity-tested Metal K-quant kernel. On macOS this optional no-copy head
    /// removes one full CPU sweep of that table per generated token; any Metal
    /// load/dispatch failure falls back to `token_embd.matvec` below.
    #[cfg(target_os = "macos")]
    metal_q6k_head: Option<crate::metal::Gemma4Q6KHead>,
}

/// Metal components constructed for a single-node Ghost-MoE runtime.
///
/// These are load/runtime-ownership facts only. The process-wide GPU switch
/// and deterministic-mode gate remain live policy and are applied by the API
/// health snapshot, so toggling acceleration updates the UI without reloading
/// the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4GhostMetalComponents {
    pub common: bool,
    pub experts: bool,
    pub head: bool,
}

/// CUDA components constructed for a single-node Ghost-MoE runtime. Kept as a
/// small copyable snapshot so API health never needs to lock the stateful CUDA
/// decoder while a generation owns it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gemma4GhostCudaComponents {
    pub common: bool,
    pub experts: bool,
    pub head: bool,
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

#[derive(Debug, Clone, Default)]
pub struct Gemma4ChunkRoundProfile {
    pub wall_clock_ms: f64,
    pub pure_gpu_ms: f64,
    pub layer0_moe_ms: f64,
    pub all_moe_layers_ms: f64,
    pub attention_core_ms: f64,
    pub ssd_cache_ms: f64,
    pub command_buffers: usize,
    pub cpu_waits: usize,
    pub layer0_gate_up_ms: f64,
    pub layer0_geglu_quant_ms: f64,
    pub layer0_down_ms: f64,
    pub layer0_weighted_reduce_ms: f64,
    pub layer0_commit_wait_ms: f64,
    pub layer0_buffer_prep_ms: f64,

    // Granular cache maintenance & common-core breakdown
    pub physical_ssd_reads_ms: f64,
    pub physical_ssd_bytes: u64,
    pub cache_lookup_ms: f64,
    pub slot_metadata_updates_ms: f64,
    pub expert_residency_validation_ms: f64,
    pub mapping_binding_ms: f64,
    pub memory_copies_ms: f64,
    pub synchronization_ms: f64,
    pub page_faults_overhead_ms: f64,
    pub eviction_bookkeeping_ms: f64,
    pub cpu_dense_shared_mlp_ms: f64,

    // Shared Expert Detailed Breakdown & Verification
    pub shared_expert_metal_calls: usize,
    pub shared_expert_cpu_calls: usize,
    pub shared_gate_up_gpu_ms: f64,
    pub shared_geglu_gpu_ms: f64,
    pub shared_down_gpu_ms: f64,
    pub shared_expert_gpu_busy_ms: f64,
    pub shared_expert_wall_ms: f64,
    pub shared_expert_cpu_fallback_ms: f64,

    // The 12 Non-Overlapping Critical Path Fields (ms):
    pub cp_attention_common_core_ms: f64,
    pub cp_routed_moe_gate_up_ms: f64,
    pub cp_routed_moe_quant_ms: f64,
    pub cp_routed_moe_down_ms: f64,
    pub cp_shared_expert_ms: f64,
    pub cp_router_topk_ms: f64,
    pub cp_kv_rope_ms: f64,
    pub cp_output_head_ms: f64,
    pub cp_cache_slot_lookup_ms: f64,
    pub cp_command_encoding_ms: f64,
    pub cp_gpu_waits_ms: f64,
    pub cp_other_ms: f64,

    // Exact Timestamped Interval Reconciliation (ms):
    pub cpu_only_exposed_ms: f64,
    pub gpu_only_exposed_ms: f64,
    pub cpu_gpu_overlapped_ms: f64,
    pub synchronization_gap_ms: f64,
    pub total_wall_clock_ms: f64,

    // Track B1: Prefetch Lead-Time Instrumentation
    pub prefetch_ready_early_count: usize,
    pub prefetch_just_in_time_count: usize,
    pub prefetch_late_count: usize,
    pub prefetch_never_used_count: usize,
    pub prefetch_avg_lead_early_ms: f64,
    pub prefetch_avg_lead_late_ms: f64,

    // Chained-round host ledger (timers only; nested GPU busy does not add to wall).
    pub gpu_chained_round_ok: bool,
    pub chained_upload_ms: f64,
    pub chained_rope_ms: f64,
    pub chained_download_ms: f64,
    pub chained_slot_wait_ms: f64,
    pub chained_final_wait_ms: f64,
    pub chained_gpu_busy_ms: f64,
    pub chained_demand_loads: usize,
    pub chained_host_sum_ms: f64,
    pub chained_prefetch_ms: f64,
    pub chained_setup_ms: f64,
    pub unique_experts_sum: u32,
    pub unique_experts_max: u32,
    pub unique_per_layer: [u16; 30],
    pub kv_capacity: u32,
    pub kv_bytes: u64,
    pub kv_filled: u32,
    pub overflow_slots: u32,
    pub overflow_bytes: u64,
    pub overflow_layers: u32,
    pub overflow_experts: u32,
    pub overflow_wait_ms: f64,
    pub expert_waves_sum: u32,
    pub expert_waves_max: u32,
    pub selected_experts_dropped: u32,
    pub missing_expert_failclose: u32,
    pub slot_capacity_overflow: u32,
    pub wave_load_ms: f64,
    pub wave_gpu_ms: f64,
    pub physical_nvme_mb: f64,
    pub gpu_qkv_o_ms: f64,
    pub gpu_attn_ms: f64,
    pub gpu_router_ms: f64,
    pub gpu_shared_ms: f64,
    pub gpu_gateup_ms: f64,
    pub gpu_down_ms: f64,
    pub gpu_resid_ms: f64,
}

impl Gemma4Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_layer_range(path, None)
    }

    /// Load Gemma 4 with routed experts supplied by a v2 expert-spliced
    /// `.cghost` file. Shared weights, router, embeddings/head, and norms stay
    /// on the existing GGUF wire path; only top-k expert blobs enter the bounded
    /// global cache.
    pub fn load_ghost_moe(
        path: &Path,
        cghost: &Path,
        cache_mib: usize,
        evict_page_cache: bool,
    ) -> Result<Self> {
        let budget_bytes = cache_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            BackendError::InvalidModelMetadata(format!(
                "ghost MoE cache size {cache_mib} MiB overflows usize"
            ))
        })?;
        let ghost = Arc::new(GhostFile::open_with_options(cghost, evict_page_cache)?);
        Self::load_layer_range_impl(path, None, Some((ghost, budget_bytes)))
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

/// Drive scalar prompt prefill while projecting the tied output head exactly
/// once, at the final prompt position. This tiny model-independent seam makes
/// the call-count contract testable without constructing a multi-gigabyte
/// Gemma runtime.
fn drive_scalar_prefill<T, E>(
    tokens: &[u32],
    mut step: impl FnMut(u32, usize, bool) -> std::result::Result<Option<T>, E>,
) -> std::result::Result<T, E> {
    let (&last_token, prefix) = tokens
        .split_last()
        .expect("prefill validates that the prompt is non-empty");
    for (pos, &token) in prefix.iter().enumerate() {
        let output = step(token, pos, false)?;
        debug_assert!(output.is_none());
    }
    Ok(step(last_token, prefix.len(), true)?
        .expect("the final scalar prefill step projects the output head"))
}

#[derive(Debug, Clone, Default)]
pub struct LayerStepProfile {
    pub layer: usize,
    pub selected_experts: Vec<usize>,
    pub router_us: u64,
    pub cache_and_io_us: u64,
    pub bytes_read: usize,
    pub shared_mlp_us: u64,
    pub expert_gemv_us: u64,
    pub attn_us: u64,
    pub ple_us: u64,
    pub total_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct TokenStepProfile {
    pub token: u32,
    pub total_us: u64,
    pub embed_us: u64,
    pub dense_attn_us: u64,
    pub router_us: u64,
    pub cache_and_io_us: u64,
    pub bytes_read: usize,
    pub shared_mlp_us: u64,
    pub expert_gemv_us: u64,
    pub ple_us: u64,
    pub head_us: u64,
    pub layers: Vec<LayerStepProfile>,
}

impl Gemma4Runtime {
    /// Merged byte spans of the wire tensors a `range` shard actually streams,
    /// for scoping the background `MADV_WILLNEED` warm-up.
    ///
    /// Readahead is bounded by device bandwidth, so advising the whole mapping
    /// spends it on bytes this shard never streams. A gemma4 GGUF's data section
    /// opens with `per_layer_token_embd` (2.5GB on E2B) — a *gather-only* table,
    /// one row per layer per token — so warming all of it front-loads the wrong
    /// bytes and, under memory pressure, evicts the layer weights the first step
    /// is actually blocked on. Every other non-layer tensor is either small or
    /// streamed whole each step (the tied head), so only the gather table is
    /// excluded.
    fn shard_warm_spans(
        gguf: &GgufFile,
        range: &std::ops::Range<usize>,
        exclude_routed_experts: bool,
    ) -> Vec<(usize, usize)> {
        let wanted = |name: &str| -> bool {
            if exclude_routed_experts
                && (name.ends_with(".ffn_gate_up_exps.weight")
                    || name.ends_with(".ffn_down_exps.weight"))
            {
                return false;
            }
            match name.strip_prefix("blk.") {
                Some(rest) => rest
                    .split_once('.')
                    .and_then(|(idx, _)| idx.parse::<usize>().ok())
                    .is_some_and(|layer| range.contains(&layer)),
                None => name != "per_layer_token_embd.weight",
            }
        };
        let mut spans: Vec<(usize, usize)> = gguf
            .tensors
            .iter()
            .filter(|t| t.n_bytes > 0 && wanted(&t.name))
            .map(|t| (t.absolute_offset as usize, t.n_bytes as usize))
            .collect();
        spans.sort_unstable();
        // Coalesce touching/overlapping spans so the kernel sees a few long
        // sequential runs rather than hundreds of small ones.
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
        for (offset, len) in spans {
            match merged.last_mut() {
                Some((m_off, m_len)) if offset <= m_off.saturating_add(*m_len) => {
                    *m_len = (offset + len).saturating_sub(*m_off).max(*m_len);
                }
                _ => merged.push((offset, len)),
            }
        }
        merged
    }

    /// Load only the given contiguous global layer range (None = all layers).
    /// Fails closed if the range would separate a KV-sharing layer from the
    /// cache it reads (the split must keep every shared layer on the same shard
    /// as its source layer).
    pub fn load_layer_range(path: &Path, range: Option<std::ops::Range<usize>>) -> Result<Self> {
        Self::load_layer_range_impl(path, range, None)
    }

    fn load_layer_range_impl(
        path: &Path,
        range: Option<std::ops::Range<usize>>,
        ghost_moe: Option<(Arc<GhostFile>, usize)>,
    ) -> Result<Self> {
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
        let ghost_moe_cache = match ghost_moe {
            Some((ghost, budget_bytes)) => {
                let moe = config.moe.as_ref().ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture(
                        "ghost MoE mode requires a Gemma 4 mixture-of-experts model".into(),
                    )
                })?;
                ghost.validate_moe_layout(
                    config.block_count as usize,
                    moe.expert_count as usize,
                    moe.expert_used_count as usize,
                )?;
                ghost.validate_moe_binding(&binding, moe.expert_count as usize)?;
                let filename = path.file_name().and_then(|name| name.to_str());
                if !ghost_source_filename_admitted(
                    ghost.index.source_identity.is_some(),
                    &ghost.index.source_model,
                    filename,
                ) {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "legacy .cghost source model {:?} does not match GGUF filename {:?}",
                        ghost.index.source_model,
                        filename.unwrap_or("<non-UTF-8>")
                    )));
                }
                ghost.validate_moe_source_identity(path, &binding, moe.expert_count as usize)?;
                Some(Arc::new(GhostMoeExpertCache::new(ghost, budget_bytes)))
            }
            None => None,
        };
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
            // Warm only the spans this shard streams, not all 5GB — see
            // `shard_warm_spans`. Still off the loading thread: the advisory
            // blocks on macOS over USB until the range is resident.
            let spans = Self::shard_warm_spans(&gguf, &range, ghost_moe_cache.is_some());
            std::thread::spawn(move || {
                for (offset, len) in spans {
                    mmap.advise_willneed_range(offset, len);
                }
            });
        }
        let q8 = |name: &str| WireQuant::new(&store, &mmap, name);
        // Matvec-role loads (projections, expert bands, the tied head) refuse
        // Q5_K typed at load — it is gather-only in this lane and would
        // otherwise panic at forward time (I-unknown-type, SHA_E3).
        let q8m = |name: &str| -> Result<WireQuant> { q8(name)?.require_matvec_capable(name) };
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };

        let mut layers = Vec::with_capacity(range.len());
        for (local_idx, l) in binding.layers[range.clone()].iter().enumerate() {
            let layer_idx = range.start + local_idx;
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
                        let pack_cache = if ghost_moe_cache.is_none()
                            && budget > 0
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
                            ghost: ghost_moe_cache.as_ref().map(|cache| GhostMoeLayer {
                                layer_idx,
                                cache: Arc::clone(cache),
                            }),
                        })
                    })
                    .transpose()?,
            });
        }

        #[cfg(target_os = "macos")]
        let metal_q4_experts = {
            let flag = |name: &str| {
                std::env::var(name).is_ok_and(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "on" | "yes"
                    )
                })
            };
            // Deterministic mode is process-pinned, so avoid compiling kernels or
            // allocating the persistent slot slab there. The live GPU toggle is instead
            // checked at every dispatch, allowing the UI to re-enable an already
            // loaded non-deterministic model without a reload.
            let enabled = (flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS")
                || crate::cuda::gpu_accel_enabled())
                && !crate::inference::deterministic_mode_enabled();
            let fused_fast =
                flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST") || crate::cuda::gpu_accel_enabled();
            let common_enabled = (flag("CAMELID_GEMMA4_GHOST_METAL_COMMON")
                || crate::cuda::gpu_accel_enabled())
                && !crate::inference::deterministic_mode_enabled();
            let slots_per_layer = if enabled {
                ghost_metal_slots_per_layer_from_env()
            } else {
                GHOST_METAL_EXPERT_SLOTS_DEFAULT
            };
            let moe_meta = config.moe.as_ref();
            let exact_geometry = ghost_moe_cache.is_some()
                && range.start == 0
                && range.end == block_count
                && config.embedding_length as usize == 2_816
                && moe_meta.is_some_and(|moe| {
                    moe.expert_count as usize == 128 && moe.expert_used_count as usize == 8
                })
                && layers.iter().all(|layer| {
                    layer.moe.as_ref().is_some_and(|moe| {
                        moe.n_ff_exp == 704
                            && moe.n_expert_used == 8
                            && moe.gate_up_exps.format == WireFormat::Q4_0
                            && moe.down_exps.format == WireFormat::Q4_0
                    })
                });
            let exact_records = if enabled && exact_geometry {
                let cache = ghost_moe_cache
                    .as_ref()
                    .expect("exact Ghost Metal geometry requires a Ghost cache");
                let expert_count = moe_meta
                    .expect("exact Ghost Metal geometry requires MoE metadata")
                    .expert_count as usize;
                match cache.file.validate_moe_expert_record_layouts(
                    block_count,
                    expert_count,
                    crate::metal::GEMMA4_Q4_EXPERT_RECORD_BYTES,
                ) {
                    Ok(()) => true,
                    Err(err) => {
                        eprintln!(
                            "[gemma4-ghost-metal] persistent slot record layout refused: {err}"
                        );
                        false
                    }
                }
            } else {
                false
            };
            let mut lane = if enabled && exact_geometry && exact_records {
                GhostMetalExpertRuntime::new(block_count, fused_fast, slots_per_layer)
            } else {
                None
            };
            if common_enabled {
                let max_positions = std::env::var("CAMELID_GEMMA4_GHOST_METAL_CONTEXT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(4_096)
                    .min(config.context_length as usize);
                if let Some(runtime) = lane.as_mut() {
                    match build_ghost_common_metal(
                        path,
                        &store,
                        &binding,
                        &config,
                        &g,
                        &layers,
                        max_positions,
                    ) {
                        Ok(Some(mut common)) => {
                            let q4_simd_fast = common.enable_fused_fast_q4(fused_fast);
                            let geometry = common.geometry();
                            eprintln!(
                                "[gemma4-ghost-common] ACTIVE: full Metal common core, context cap={} positions, allocated KV={} ({:.2}GiB of {:.2}GiB at cap), router/shared/expert/tail device-chained, mode={}, q4-row={}",
                                geometry.max_positions,
                                geometry.kv_capacity,
                                geometry.kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                                (geometry.max_positions as f64 / geometry.kv_capacity.max(1) as f64)
                                    * (geometry.kv_bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
                                if fused_fast { "fused-fast" } else { "CPU-GeGLU parity" },
                                if q4_simd_fast { "simdgroup-ordered" } else { "scalar-ordered" },
                            );
                            runtime.common = Some(common);
                        }
                        Ok(None) => eprintln!(
                            "[gemma4-ghost-common] requested but exact Gemma 4 26B Q4_0/no-PLE geometry was not admitted; CPU common core remains active"
                        ),
                        Err(error) => eprintln!(
                            "[gemma4-ghost-common] construction failed: {error}; CPU common core remains active"
                        ),
                    }
                } else {
                    eprintln!(
                        "[gemma4-ghost-common] requested but persistent expert slots are unavailable; CPU common core remains active"
                    );
                }
            }
            if let Some(lane) = lane.as_mut() {
                if let Some(cache) = ghost_moe_cache.as_ref() {
                    lane.prewarm_hot_slots_direct(&cache.file);
                }
                eprintln!(
                    "[gemma4-ghost-metal] persistent Q4_0 slots enabled: layers={} slots/layer={} resident={:.2}GiB mode={}",
                    block_count,
                    lane.slots_per_layer(),
                    lane.resident_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                    if fused_fast { "fused-fast" } else { "CPU-GeGLU parity" },
                );
            } else if enabled {
                eprintln!(
                    "[gemma4-ghost-metal] persistent slots unavailable or model geometry is not exact Gemma 4 26B Q4_0; using CPU Ghost experts"
                );
            }
            std::sync::Mutex::new(lane)
        };

        let first_kv_shared = config.block_count as usize - g.num_kv_shared_layers as usize;
        // Bind the common tied table once so the CPU fallback and the optional
        // no-copy Metal head share the exact same validated GGUF descriptor.
        let token_embd = q8m(&binding.token_embedding.name)?;
        let output_norm = f32t(&binding.output_norm.name)?;
        #[cfg(target_os = "macos")]
        let metal_q6k_head = {
            let explicitly_disabled = std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD")
                .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"));
            let eligible = ghost_moe_cache.is_some()
                && range.end == block_count
                && token_embd.format == WireFormat::Q6K
                && !explicitly_disabled
                && !crate::inference::deterministic_mode_enabled();
            let head = if eligible {
                match &token_embd.backing {
                    WireBacking::Mmap { mmap, offset } => crate::metal::Gemma4Q6KHead::new(
                        Arc::clone(mmap),
                        *offset,
                        token_embd.bytes().len(),
                        &output_norm,
                        token_embd.element_count / config.embedding_length as usize,
                        g.final_logit_softcapping.unwrap_or(0.0),
                        config.rms_norm_epsilon,
                    ),
                    WireBacking::Owned { .. } => None,
                }
            } else {
                None
            };
            if head.is_some() {
                eprintln!("[gemma4-ghost] Metal Q6_K tied head enabled (no-copy, file-backed)");
            } else if eligible {
                eprintln!("[gemma4-ghost] Metal Q6_K tied head unavailable; using CPU fallback");
            }
            head
        };
        Ok(Self {
            tokenizer,
            first_layer: range.start,
            // The tied head matvecs token_embd on the tail shard, so it takes
            // the matvec-role guard; per_layer_token_embd stays gather-only
            // (plain q8) — Q5_K is legitimate there.
            token_embd,
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
            output_norm,
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
            ghost_moe_cache,
            #[cfg(target_os = "macos")]
            metal_q4_experts,
            #[cfg(target_os = "macos")]
            ghost_common_generation: std::sync::Mutex::new(()),
            #[cfg(target_os = "macos")]
            metal_q6k_head,
            layers,
            config,
            g,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// `None` on the normal resident/mmap lane; live bounded-cache counters on
    /// the v2 Ghost-MoE lane.
    pub fn ghost_moe_cache_stats(&self) -> Option<GhostMoeCacheStats> {
        self.ghost_moe_cache.as_ref().map(|cache| cache.stats())
    }

    /// Metal components still owned by this Ghost runtime. The persistent
    /// expert lane can disable itself after a command failure, so this is read
    /// live rather than latched at model load.
    pub fn ghost_metal_components(&self) -> Gemma4GhostMetalComponents {
        #[cfg(target_os = "macos")]
        {
            let (experts, common) = self
                .metal_q4_experts
                .lock()
                .map(|guard| {
                    let experts = guard.is_some();
                    let common = guard
                        .as_ref()
                        .and_then(|runtime| runtime.common.as_ref())
                        .is_some_and(crate::metal::Gemma4GhostCommonMetal::moe_configured);
                    (experts, common)
                })
                .unwrap_or((false, false));
            Gemma4GhostMetalComponents {
                common,
                experts,
                head: self.metal_q6k_head.is_some(),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Gemma4GhostMetalComponents::default()
        }
    }

    /// Evaluates predicted routes for candidate tokens across all 30 layers and
    /// launches the round-wide asynchronous prefetch into the Metal resident slab buffers.
    /// Returns the number of distinct (layer, expert) pairs prefetched.
    pub fn prefetch_round_wide_chunk(&self, chunk: &[u32]) -> Result<usize> {
        self.prefetch_round_wide_chunk_top_n(chunk, 16)
    }

    /// Parameterized Top-N prefetch helper across all 30 layers.
    pub fn prefetch_round_wide_chunk_top_n(&self, chunk: &[u32], top_n: usize) -> Result<usize> {
        let predicted_routes = self.predict_all_layer_routes_for_chunk_top_n(chunk, top_n)?;
        let count = predicted_routes.iter().map(|r| r.len()).sum();
        #[cfg(target_os = "macos")]
        if let Some(cache) = self.ghost_moe_cache.as_ref() {
            use rayon::prelude::*;
            predicted_routes
                .par_iter()
                .enumerate()
                .for_each(|(layer_idx, routes)| {
                    if !routes.is_empty() {
                        let _ = cache.get_many(layer_idx, routes);
                    }
                });
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    lane.prefetch_round_wide_async(
                        &predicted_routes,
                        &cache.file,
                        &cache.read_pool,
                    );
                }
            }
        }
        Ok(count)
    }

    /// Aggregate hits and misses across all persistent Metal slot directories.
    pub fn ghost_metal_aggregate_slot_stats(&self) -> (u64, u64) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_ref() {
                    let stats = lane.slot_stats();
                    return (stats.hits, stats.misses);
                }
            }
        }
        (0, 0)
    }

    /// Truncate the resident sequence in the Ghost Metal common core to `keep` positions.
    pub fn truncate_sequence(&self, keep: usize) {
        #[cfg(target_os = "macos")]
        {
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    lane.truncate_sequence(keep);
                }
            }
        }
    }

    /// Backwards-compatible common-core construction probe used by the real
    /// fixture gate. Live GPU/deterministic policy is deliberately not folded
    /// into this model-owned state.
    pub fn ghost_common_metal_active(&self) -> bool {
        self.ghost_metal_components().common
    }

    /// Select one authoritative KV lane before prompt position zero. The budget
    /// covers every forward the request may need, so a configured 4K Metal cache
    /// never fails halfway through a longer request: that request stays on CPU
    /// from the start instead.
    fn prepare_ghost_prefill(
        &self,
        prompt_len: usize,
        future_forwards: usize,
    ) -> Result<GhostPrefillPlan> {
        let required_positions = prompt_len.checked_add(future_forwards).ok_or_else(|| {
            BackendError::InvalidModelMetadata(
                "Gemma 4 prompt plus decode position count overflows usize".into(),
            )
        })?;
        if required_positions > self.config.context_length as usize {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Gemma 4 request needs {required_positions} positions, exceeding the model context length {}",
                self.config.context_length
            )));
        }
        let chunk_eligible = self.ghost_moe_cache.is_some() && self.supports_chunk_forward();
        let hybrid_enabled = !std::env::var("CAMELID_GEMMA4_GHOST_HYBRID_PREFILL")
            .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"));

        #[cfg(target_os = "macos")]
        {
            let gpu_allowed = ghost_metal_acceleration_enabled();
            let mut guard = self.metal_q4_experts.lock().map_err(|_| {
                BackendError::InvalidModelMetadata("Ghost Metal runtime mutex is poisoned".into())
            })?;
            let Some(runtime) = guard.as_mut() else {
                return Ok(select_ghost_prefill_plan(
                    chunk_eligible,
                    hybrid_enabled,
                    prompt_len,
                    required_positions,
                    None,
                ));
            };
            let configured_capacity = runtime
                .common
                .as_ref()
                .filter(|common| common.moe_configured())
                .map(crate::metal::Gemma4GhostCommonMetal::max_positions);
            let common_capacity = gpu_allowed.then_some(configured_capacity).flatten();
            let plan = select_ghost_prefill_plan(
                chunk_eligible,
                hybrid_enabled,
                prompt_len,
                required_positions,
                common_capacity,
            );
            if let Some(common) = runtime.common.as_mut() {
                common.reset_sequence();
            }
            runtime.sequence_mode = match plan {
                GhostPrefillPlan::ScalarCpu | GhostPrefillPlan::CpuChunk => {
                    GhostMetalSequenceMode::Cpu
                }
                GhostPrefillPlan::ScalarMetal => GhostMetalSequenceMode::Metal,
                GhostPrefillPlan::HybridChunk => GhostMetalSequenceMode::HybridPrefill,
            };
            if let Some(capacity) = configured_capacity {
                if required_positions > capacity {
                    eprintln!(
                        "[gemma4-ghost-common] request needs {required_positions} positions but Metal capacity is {capacity}; using the CPU KV lane from position zero"
                    );
                }
            }
            Ok(plan)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(select_ghost_prefill_plan(
                chunk_eligible,
                hybrid_enabled,
                prompt_len,
                required_positions,
                None,
            ))
        }
    }

    /// Commit a completed CPU chunk prefill to the persistent common-core cache.
    /// Any refusal leaves the host cache authoritative and pins the rest of this
    /// request to CPU; host rows are released only after all Metal layers import.
    fn finish_ghost_hybrid_prefill(
        &self,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        positions: usize,
    ) -> Result<bool> {
        #[cfg(target_os = "macos")]
        {
            let started = std::time::Instant::now();
            let imported = {
                let mut guard = self.metal_q4_experts.lock().map_err(|_| {
                    BackendError::InvalidModelMetadata(
                        "Ghost Metal runtime mutex is poisoned".into(),
                    )
                })?;
                let Some(runtime) = guard.as_mut() else {
                    return Ok(false);
                };
                if runtime.sequence_mode != GhostMetalSequenceMode::HybridPrefill
                    || !ghost_metal_acceleration_enabled()
                {
                    runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                    if let Some(common) = runtime.common.as_mut() {
                        common.reset_sequence();
                    }
                    return Ok(false);
                }
                let Some(common) = runtime.common.as_mut() else {
                    runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                    return Ok(false);
                };
                if common.is_at_position(positions) {
                    runtime.sequence_mode = GhostMetalSequenceMode::Metal;
                    true
                } else {
                    match common.import_position_major_kv(kc, vc, positions) {
                        Ok(()) => {
                            runtime.sequence_mode = GhostMetalSequenceMode::Metal;
                            true
                        }
                        Err(error) => {
                            eprintln!(
                                "[gemma4-ghost-common] CPU prefill KV import refused: {error}; continuing this request on CPU"
                            );
                            common.reset_sequence();
                            runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                            false
                        }
                    }
                }
            };
            if imported {
                let keep_host_kv = std::env::var("CAMELID_SPEC_DECODE")
                    .map(|v| {
                        !matches!(
                            v.trim().to_ascii_lowercase().as_str(),
                            "off" | "0" | "false" | "none"
                        )
                    })
                    .unwrap_or(true);
                if !keep_host_kv {
                    // Drop the per-position allocations when pure scalar GPU decode is used.
                    for layer in kc.iter_mut().chain(vc.iter_mut()) {
                        layer.clear();
                        layer.shrink_to_fit();
                    }
                }
                if ghost_metal_timing_enabled() {
                    eprintln!(
                        "[gemma4-ghost-common] imported {positions} CPU-prefilled positions into Metal KV in {:.1}ms",
                        started.elapsed().as_secs_f64() * 1_000.0
                    );
                }
            }
            Ok(imported)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (kc, vc, positions);
            Ok(false)
        }
    }

    #[cfg(target_os = "macos")]
    fn lock_ghost_common_generation(&self) -> Result<std::sync::MutexGuard<'_, ()>> {
        self.ghost_common_generation.lock().map_err(|_| {
            BackendError::InvalidModelMetadata(
                "Ghost common Metal generation mutex is poisoned".into(),
            )
        })
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

    /// Final RMSNorm + tied vocabulary projection + logit soft-cap. Ghost-MoE
    /// on macOS first tries the persistent no-copy Q6_K Metal head; every other
    /// model/platform, and any soft Metal failure, executes the established CPU
    /// wire-dot path unchanged.
    fn project_logits(&self, hidden: &[f32]) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        if ghost_metal_acceleration_enabled() {
            if let Some(head) = self.metal_q6k_head.as_ref() {
                if let Some(logits) = head.forward(hidden) {
                    return logits;
                }
            }
        }
        self.project_logits_cpu(hidden)
    }

    fn project_logits_cpu(&self, hidden: &[f32]) -> Vec<f32> {
        let last = rms_norm(
            hidden,
            Some(&self.output_norm),
            self.config.rms_norm_epsilon,
        );
        let mut logits = self
            .token_embd
            .matvec(self.hidden_size(), self.vocab_size(), &last);
        if let Some(cap) = self.g.final_logit_softcapping {
            soft_cap_in_place(&mut logits, cap);
        }
        logits
    }

    fn project_logits_chunk_cpu(&self, hs: &[Vec<f32>], eps: f32, vocab: usize) -> Vec<Vec<f32>> {
        let lastq: Vec<Vec<f32>> = hs
            .iter()
            .map(|h| rms_norm(h, Some(&self.output_norm), eps))
            .collect();
        let lastb = SharedActivationBatch::new(&lastq);
        let mut logits_rows: Vec<Vec<f32>> = self.token_embd.matmul_proj(vocab, &lastb);
        if let Some(cap) = self.g.final_logit_softcapping {
            for logits in logits_rows.iter_mut() {
                soft_cap_in_place(logits, cap);
            }
        }
        logits_rows
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

    #[cfg(target_os = "macos")]
    fn try_ghost_common_step(
        &self,
        token: u32,
        pos: usize,
        project_head: bool,
    ) -> Result<Option<GhostCommonStepOutput>> {
        let gpu_allowed = ghost_metal_acceleration_enabled();
        let mut guard = self.metal_q4_experts.lock().map_err(|_| {
            BackendError::InvalidModelMetadata("Ghost Metal runtime mutex is poisoned".into())
        })?;
        let Some(runtime) = guard.as_mut() else {
            return Ok(None);
        };

        // Public generation can pin CPU/Hybrid before position zero. Preserve
        // those decisions; Idle direct callers and a stale/completed Metal
        // sequence still get the legacy position-zero reset.
        if pos == 0
            && matches!(
                runtime.sequence_mode,
                GhostMetalSequenceMode::Idle | GhostMetalSequenceMode::Metal
            )
        {
            runtime.sequence_mode = if gpu_allowed
                && runtime
                    .common
                    .as_ref()
                    .is_some_and(crate::metal::Gemma4GhostCommonMetal::moe_configured)
            {
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
                GhostMetalSequenceMode::Metal
            } else {
                GhostMetalSequenceMode::Cpu
            };
        }
        match runtime.sequence_mode {
            GhostMetalSequenceMode::Cpu
            | GhostMetalSequenceMode::HybridPrefill
            | GhostMetalSequenceMode::Idle => return Ok(None),
            GhostMetalSequenceMode::Metal if !gpu_allowed => {
                return Err(BackendError::UnsupportedModelArchitecture(
                    "Ghost common Metal was disabled during an active request; retry the request so Camelid can select one KV lane from position zero".into(),
                ));
            }
            GhostMetalSequenceMode::Metal => {}
        }

        let token_started = std::time::Instant::now();
        let forward = (|| -> Result<GhostCommonStepOutput> {
            let hidden = self.config.embedding_length as usize;
            let h0: Vec<f32> = self
                .token_embd
                .dequantize_elements(token as usize * hidden, hidden)?
                .iter()
                .map(|value| value * (hidden as f32).sqrt())
                .collect();
            let common = runtime.common.as_mut().ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture(
                    "Ghost common Metal state disappeared during an active request".into(),
                )
            })?;
            if pos >= common.max_positions() {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "Ghost common Metal context capacity {} is smaller than requested position {pos}; increase CAMELID_GEMMA4_GHOST_METAL_CONTEXT and reload",
                    common.max_positions()
                )));
            }
            if !common.write_hidden(&h0) {
                return Err(BackendError::InvalidTensorData(
                    "Ghost common Metal rejected the token embedding".into(),
                ));
            }

            let mut previous_layer_pending: Option<(usize, GhostLayerPendingGuard)> = None;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if !ghost_metal_acceleration_enabled() {
                    return Err(BackendError::UnsupportedModelArchitecture(
                        "GPU acceleration was disabled during a Ghost common token; retry the request from position zero".into(),
                    ));
                }
                let head_dim = self.g.head_dim_at(layer_idx) as usize;
                let theta = self.g.rope_freq_base_at(layer_idx);
                let factors = if self.g.is_sliding_layer(layer_idx) {
                    None
                } else {
                    self.rope_factors.as_deref()
                };
                let mut cos_t = vec![0.0f32; head_dim / 2];
                let mut sin_t = vec![0.0f32; head_dim / 2];
                for i in 0..head_dim / 2 {
                    let mut frequency = theta.powf(-(2.0 * i as f32) / head_dim as f32);
                    if let Some(factors) = factors {
                        frequency /= factors[i];
                    }
                    let (sin, cos) = (pos as f32 * frequency).sin_cos();
                    cos_t[i] = cos;
                    sin_t[i] = sin;
                }

                let attention_router_pending = runtime
                    .common
                    .as_mut()
                    .and_then(|common| {
                        common.enqueue_attention_router(layer_idx, &cos_t, &sin_t, pos)
                    })
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal attention/router failed to enqueue at layer {layer_idx} position {pos}"
                        ))
                    })?;
                // Queue shared immediately behind the fused attention/router
                // command. Waiting exposes only 128 logits while Metal has already
                // advanced into shared Q4 work; route selection and direct slot
                // reads overlap that work with no idle queue bubble.
                let mut shared_pending = GhostCommonPendingGuard::new(
                    runtime
                    .common
                    .as_mut()
                    .and_then(|common| common.enqueue_shared_branch(layer_idx))
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal shared branch failed to enqueue at layer {layer_idx}"
                        ))
                    })?,
                );
                if attention_router_pending.wait().is_none() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "Ghost common Metal attention/router failed at layer {layer_idx} position {pos}"
                    )));
                }
                // The singleton Metal queue completed this layer's
                // attention/router only after the preceding fused expert/tail.
                // Drain that older handle now: it is an immediate status check,
                // not another GPU synchronization point on the steady-state path.
                if let Some((pending_layer, mut pending)) = previous_layer_pending.take() {
                    if !pending.finish() {
                        return Err(BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal asynchronous expert/tail failed at layer {pending_layer}"
                        )));
                    }
                }
                let logits = runtime
                    .common
                    .as_ref()
                    .and_then(crate::metal::Gemma4GhostCommonMetal::read_router_logits)
                    .ok_or_else(|| {
                        BackendError::UnsupportedModelArchitecture(
                            "Ghost common Metal router logits were unavailable".into(),
                        )
                    })?;
                if logits.iter().any(|value| !value.is_finite()) {
                    return Err(BackendError::InvalidTensorData(format!(
                        "Ghost common Metal router produced non-finite logits at layer {layer_idx}"
                    )));
                }
                if std::env::var("CAMELID_GEMMA4_DUMP_ROUTER").is_ok_and(|v| v == "1") {
                    let l2 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!(
                        "[router-metal] pos {pos} layer {layer_idx} l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                        logits[0], logits[1], logits[2], logits[3]
                    );
                }
                let max_logit = logits.iter().copied().fold(f32::MIN, f32::max);
                let mut probabilities: Vec<f32> = logits
                    .iter()
                    .map(|value| (*value - max_logit).exp())
                    .collect();
                let probability_sum: f32 = probabilities.iter().sum();
                if !probability_sum.is_finite() || probability_sum <= 0.0 {
                    return Err(BackendError::InvalidTensorData(format!(
                        "Ghost common Metal router softmax failed at layer {layer_idx}"
                    )));
                }
                for probability in &mut probabilities {
                    *probability /= probability_sum;
                }
                let moe = layer.moe.as_ref().ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Ghost common Metal layer {layer_idx} has no MoE weights"
                    ))
                })?;
                let mut experts: Vec<usize> = (0..moe.n_expert).collect();
                experts.sort_unstable_by(|&a, &b| {
                    probabilities[b]
                        .partial_cmp(&probabilities[a])
                        .expect("finite router probabilities")
                        .then(a.cmp(&b))
                });
                experts.truncate(moe.n_expert_used);
                if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
                    eprintln!("[route-metal] l={layer_idx} e={experts:?}");
                }
                let selected_sum = experts
                    .iter()
                    .map(|&expert| probabilities[expert])
                    .sum::<f32>()
                    .max(6.103_515e-5);
                let route_scales: Vec<f32> = experts
                    .iter()
                    .map(|&expert| {
                        moe.down_exps_scale[expert] * (probabilities[expert] / selected_sum)
                    })
                    .collect();
                let ghost = moe.ghost.as_ref().ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Ghost common Metal layer {layer_idx} is not Ghost-backed"
                    ))
                })?;

                // Directory-first: `prepare_layer_routes` resolves slot hits
                // without any host copy and reads misses straight from the
                // .cghost into the slot. Only *already resident* host-cache
                // records are offered as a memcpy source (`peek_resident` never
                // performs I/O or touches cache accounting). Fetching all eight
                // routed experts through `get_many` here measured as 8 host
                // reads per layer per token even for slot hits (115.9 GiB over
                // 78 positions with a 64 MiB cache), collapsing decode to
                // <1 tok/s.
                let resident_sources = experts
                    .iter()
                    .filter_map(|&expert| {
                        ghost
                            .cache
                            .peek_resident(ghost.layer_idx, expert)
                            .map(|record| (expert, record))
                    })
                    .collect::<std::collections::HashMap<_, _>>();
                let expert_attempt =
                    runtime.run_layer_common(ghost, &experts, &route_scales, &resident_sources);
                match expert_attempt {
                    GhostMetalCommonAttempt::Pending(tail) => {
                        let shared = shared_pending.take().ok_or_else(|| {
                            BackendError::UnsupportedModelArchitecture(format!(
                                "Ghost common Metal lost the shared command at layer {layer_idx}"
                            ))
                        })?;
                        // Debug bisection: serialize the pipeline and dump the
                        // per-layer hidden L2 so the K=1 Metal lane can be
                        // compared layer-by-layer against the CPU reference.
                        if std::env::var("CAMELID_GEMMA4_DUMP_COMMON_LAYERS")
                            .is_ok_and(|v| v == "1")
                        {
                            let mut guard = GhostLayerPendingGuard::new(shared, tail);
                            if !guard.finish() {
                                return Err(BackendError::UnsupportedModelArchitecture(format!(
                                    "Ghost common Metal expert/tail failed at layer {layer_idx} (debug drain)"
                                )));
                            }
                            if let Some(common) = runtime.common.as_ref() {
                                let h = common.read_hidden();
                                let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                                eprintln!(
                                    "[h_common] pos {pos} layer {layer_idx} l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                                    h[0], h[1], h[2], h[3]
                                );
                            }
                        } else {
                            previous_layer_pending =
                                Some((layer_idx, GhostLayerPendingGuard::new(shared, tail)));
                        }
                    }
                    GhostMetalCommonAttempt::Complete => {
                        if !shared_pending.finish() {
                            return Err(BackendError::UnsupportedModelArchitecture(format!(
                                "Ghost common Metal shared branch failed at layer {layer_idx}"
                            )));
                        }
                    }
                    GhostMetalCommonAttempt::CpuFallback => {
                        return Err(BackendError::InvalidTensorData(format!(
                            "Ghost common Metal slot preparation failed at layer {layer_idx}"
                        )));
                    }
                    GhostMetalCommonAttempt::DisableMetal => {
                        return Err(BackendError::UnsupportedModelArchitecture(format!(
                            "Ghost common Metal expert/tail dispatch failed at layer {layer_idx}"
                        )));
                    }
                }
            }
            // Layer 29 has no following attention/router fence to imply its
            // completion. Drain it before the head reads hidden, and also before
            // a headless prefill step returns and permits the next token.
            if let Some((pending_layer, mut pending)) = previous_layer_pending.take() {
                if !pending.finish() {
                    return Err(BackendError::UnsupportedModelArchitecture(format!(
                        "Ghost common Metal asynchronous expert/tail failed at final layer {pending_layer}"
                    )));
                }
            }
            if !project_head {
                return Ok(GhostCommonStepOutput::Advanced);
            }
            let final_hidden = runtime
                .common
                .as_ref()
                .map(crate::metal::Gemma4GhostCommonMetal::read_hidden)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture(
                        "Ghost common Metal final hidden was unavailable".into(),
                    )
                })?;
            Ok(GhostCommonStepOutput::Logits(
                self.project_logits(&final_hidden),
            ))
        })();

        match forward {
            Ok(output) => {
                if std::env::var("CAMELID_GEMMA4_GHOST_COMMON_TIMING")
                    .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                {
                    eprintln!(
                        "[gemma4-ghost-common-token] pos={pos} layers={} head={} wall={}us effective={:.2}tok/s",
                        self.layers.len(),
                        if project_head { "on" } else { "off" },
                        token_started.elapsed().as_micros(),
                        1.0 / token_started.elapsed().as_secs_f64().max(f64::EPSILON),
                    );
                }
                Ok(Some(output))
            }
            Err(error) if pos == 0 => {
                eprintln!(
                    "[gemma4-ghost-common] first-position Metal attempt failed: {error}; restarting this request on the CPU lane"
                );
                runtime.sequence_mode = GhostMetalSequenceMode::Cpu;
                if let Some(common) = runtime.common.as_mut() {
                    common.reset_sequence();
                }
                Ok(None)
            }
            Err(error) => {
                runtime.sequence_mode = GhostMetalSequenceMode::Idle;
                Err(error)
            }
        }
    }

    /// Process one token at absolute `pos`, appending its K/V to the per-layer
    /// caches (`kc`/`vc`; only non-shared layers store entries — shared layers read
    /// the last same-type layer's cache, already updated this step). Returns the
    /// next-token logits.
    pub fn step(
        &self,
        token: u32,
        pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<f32>> {
        // K=1 decode: HEAD lane by default (see `chained_k1_enabled`);
        // the chained lane is the K>1 verifier.
        #[cfg(target_os = "macos")]
        if chained_k1_enabled() && self.supports_chunk_forward() {
            if lane_trace_enabled() {
                eprintln!("[lane] step pos={pos} -> chunk(K=1)");
            }
            let mut rows = self.step_chunk_with_head(&[token], pos, kc, vc, true, None)?;
            return Ok(rows.pop().expect("step_chunk_with_head emits 1 row"));
        }
        #[cfg(target_os = "macos")]
        if let Some(output) = self.try_ghost_common_step(token, pos, true)? {
            return match output {
                GhostCommonStepOutput::Logits(logits) => Ok(logits),
                GhostCommonStepOutput::Advanced => unreachable!("head was requested"),
            };
        }
        #[cfg(target_os = "macos")]
        self.guard_cpu_kv_lane(pos, kc)?;
        if lane_trace_enabled() {
            eprintln!("[lane] step pos={pos} -> step_range(scalar CPU)");
        }
        match self.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Logits(logits) => Ok(logits),
            Gemma4StepOutput::Hidden(_) => Err(BackendError::InvalidModelMetadata(
                "step() requires a runtime that owns the final layer; use step_range \
                 on interior shards"
                    .into(),
            )),
        }
    }

    /// Advance one scalar token's transformer/KV state without running the
    /// output vocabulary projection. Used for every non-final prompt token;
    /// decode still calls [`Self::step`] and therefore returns logits.
    fn step_without_head(
        &self,
        token: u32,
        pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if chained_k1_enabled() && self.supports_chunk_forward() {
            self.step_chunk_with_head(&[token], pos, kc, vc, false, None)?;
            return Ok(());
        }
        #[cfg(target_os = "macos")]
        if let Some(output) = self.try_ghost_common_step(token, pos, false)? {
            return match output {
                GhostCommonStepOutput::Advanced => Ok(()),
                GhostCommonStepOutput::Logits(_) => unreachable!("head was not requested"),
            };
        }
        #[cfg(target_os = "macos")]
        self.guard_cpu_kv_lane(pos, kc)?;
        match self.step_range(token, pos, None, kc, vc)? {
            Gemma4StepOutput::Hidden(_) | Gemma4StepOutput::Logits(_) => Ok(()),
        }
    }

    /// The chained Metal lane advances the resident Metal KV without appending
    /// to the host `kc`/`vc` caches. If a scalar step then falls through to the
    /// CPU `step_range` lane at a non-zero position with an empty host cache,
    /// the two KV lanes are inconsistent and the CPU forward would silently
    /// attend over nothing. Fail closed instead of producing wrong logits.
    #[cfg(target_os = "macos")]
    fn guard_cpu_kv_lane(&self, pos: usize, kc: &[Vec<Vec<f32>>]) -> Result<()> {
        if pos > 0 && self.ghost_metal_q4_is_enabled() && kc.iter().all(|layer| layer.is_empty()) {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Gemma 4 scalar step at position {pos} fell through to the CPU KV lane \
                 while the host K/V caches are empty (the Metal lane holds the sequence); \
                 refusing to run an inconsistent forward. Select the Metal lane from position \
                 zero (mode must be Metal) or prefill on the CPU lane."
            )));
        }
        Ok(())
    }

    /// Roll back on-device Metal sequence state to a prior position (e.g. during speculative rejection).
    pub fn rollback_sequence(&self, keep: usize) {
        #[cfg(target_os = "macos")]
        if let Ok(mut guard) = self.metal_q4_experts.lock() {
            if let Some(lane) = guard.as_mut() {
                lane.truncate_sequence(keep);
            }
        }
    }

    /// True when the batched [`Self::step_chunk`] forward is usable: single-node
    /// (this runtime owns every layer including the head), with either dense rows
    /// or Ghost-backed MoE rows. Mmap-backed MoE still uses the scalar lane: its
    /// packed-expert cache has different batching/lifetime tradeoffs, while the
    /// Ghost lane needs chunking to keep prompt prefill from rereading the same
    /// routed expert once per token.
    fn supports_chunk_forward(&self) -> bool {
        self.first_layer == 0
            && self.first_layer + self.layers.len() == self.config.block_count as usize
            && (self.ghost_moe_cache.is_some()
                || self
                    .layers
                    .iter()
                    .all(|lw| lw.moe.as_ref().is_none_or(|moe| moe.ghost.is_some())))
    }

    /// Return scaled embedding h0 for a token
    pub fn token_embedding(&self, token: u32) -> Result<Vec<f32>> {
        let hidden = self.config.embedding_length as usize;
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        Ok(h0)
    }

    /// Speculative decode operates on single-node dense models as well as Ghost-backed MoE models.
    fn supports_speculative_chunk_forward(&self) -> bool {
        self.supports_chunk_forward()
    }

    /// Batched forward over `tokens` at consecutive positions `start_pos +
    /// 0..tokens.len()`, appending all K K/V rows to the caches and returning the
    /// next-token logits at EACH position. Numerically identical to calling
    /// [`Self::step`] once per token (same dots, same order) — the only difference is
    /// that each weight matrix is read ONCE for the whole chunk via [`matmul_q`]
    /// Run the multi-expert batched forward on a candidate token chunk (e.g. K=5)
    /// instead of once per token, which is the speculative-decode verify win.
    /// Requires [`Self::supports_chunk_forward`]; caller guarantees it.
    #[allow(clippy::needless_range_loop)]
    pub fn step_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<f32>>> {
        self.step_chunk_with_head(tokens, start_pos, kc, vc, true, None)
    }

    /// Profiled sibling of [`Self::step_chunk`] that captures hardware and kernel timings.
    pub fn step_chunk_profiled(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<(Vec<Vec<f32>>, Gemma4ChunkRoundProfile)> {
        let mut prof = Gemma4ChunkRoundProfile::default();
        let logits = self.step_chunk_with_head(tokens, start_pos, kc, vc, true, Some(&mut prof))?;
        Ok((logits, prof))
    }

    /// Shared chunk body. Prompt prefill requests only the final row's tied
    /// head, while speculative verification and parity tests need every row.
    #[allow(clippy::needless_range_loop)]
    fn step_chunk_with_head(
        &self,
        tokens: &[u32],
        start_pos: usize,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        all_logits: bool,
        mut profile: Option<&mut Gemma4ChunkRoundProfile>,
    ) -> Result<Vec<Vec<f32>>> {
        let t_round_start = std::time::Instant::now();
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
        let t_init_start = std::time::Instant::now();
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
        let init_dur = t_init_start.elapsed();
        if let Some(prof) = profile.as_deref_mut() {
            prof.cp_other_ms += init_dur.as_secs_f64() * 1000.0;
        }

        let mut hs_buf: Vec<Vec<f32>> = (0..kk).map(|_| vec![0.0f32; hidden]).collect();
        let mut shared_mlp_buf: Vec<Vec<f32>> = (0..kk).map(|_| vec![0.0f32; hidden]).collect();
        let mut router_logits_buf = vec![0.0f32; kk * 128];

        let mut gpu_chained_round_ok = false;
        let t_gpu_round_start = std::time::Instant::now();

        #[cfg(target_os = "macos")]
        {
            if self.ghost_metal_q4_is_enabled() && n_local == self.layers.len() {
                let theta_local = (0..self.layers.len())
                    .find(|&l| self.g.is_sliding_layer(l))
                    .map(|l| self.g.rope_freq_base_at(l))
                    .unwrap_or(10000.0);
                let theta_global = (0..self.layers.len())
                    .find(|&l| !self.g.is_sliding_layer(l))
                    .map(|l| self.g.rope_freq_base_at(l))
                    .unwrap_or(theta_local);
                let rope_factors = self.rope_factors.as_deref();
                let ghost_cache = self.ghost_moe_cache.as_deref();
                if let Ok(mut guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_mut() {
                        lane.prefill_round = !all_logits;
                        gpu_chained_round_ok = lane.execute_chained_round_all_layers(
                            &hs,
                            theta_local,
                            theta_global,
                            rope_factors,
                            start_pos,
                            ghost_cache,
                            &mut hs_buf,
                        );
                    }
                }
            }
        }

        if lane_trace_enabled() {
            eprintln!("[lane] step_chunk kk={kk} start_pos={start_pos} all_logits={all_logits} gpu_chained_ok={gpu_chained_round_ok}");
        }
        if gpu_chained_round_ok {
            let round_dur = t_gpu_round_start.elapsed();
            std::mem::swap(&mut hs, &mut hs_buf);
            if let Some(dir) = std::env::var_os("CAMELID_GEMMA4_DUMP_DIR") {
                let _ = std::fs::write(
                    std::path::PathBuf::from(dir).join("gpu_chained_round.txt"),
                    "ok=1 fallback=0\n",
                );
            }
            if let Some(prof) = profile.as_deref_mut() {
                #[cfg(target_os = "macos")]
                {
                    if let Ok(guard) = self.metal_q4_experts.lock() {
                        if let Some(lane) = guard.as_ref() {
                            let led = lane.last_chained_ledger();
                            let filler_cpu =
                                (led.slot_filler_ms + led.wave_load_ms - led.nvme_ms).max(0.0);
                            prof.gpu_chained_round_ok = true;
                            prof.chained_upload_ms = led.upload_ms;
                            prof.chained_rope_ms = led.rope_ms;
                            prof.chained_download_ms = led.download_ms;
                            prof.chained_slot_wait_ms = led.slot_wait_ms;
                            prof.chained_final_wait_ms = led.final_wait_ms;
                            prof.chained_gpu_busy_ms = led.gpu_busy_ms;
                            prof.chained_demand_loads = led.demand_loads;
                            prof.chained_host_sum_ms = led.host_sum_ms();
                            prof.chained_prefetch_ms = led.prefetch_ms;
                            prof.chained_setup_ms = led.setup_ms;
                            prof.unique_experts_sum = led.unique_experts_sum;
                            prof.unique_experts_max = led.unique_experts_max;
                            prof.unique_per_layer = led.unique_per_layer;
                            prof.kv_capacity = led.kv_capacity;
                            prof.kv_bytes = led.kv_bytes;
                            prof.kv_filled = led.kv_filled;
                            prof.overflow_slots = led.overflow_slots;
                            prof.overflow_bytes = led.overflow_bytes;
                            prof.overflow_layers = led.overflow_layers;
                            prof.overflow_experts = led.overflow_experts;
                            prof.overflow_wait_ms = led.overflow_wait_ms;
                            prof.expert_waves_sum = led.expert_waves_sum;
                            prof.expert_waves_max = led.expert_waves_max;
                            prof.selected_experts_dropped = led.selected_experts_dropped;
                            prof.missing_expert_failclose = led.missing_expert_failclose;
                            prof.slot_capacity_overflow = led.slot_capacity_overflow;
                            prof.wave_load_ms = led.wave_load_ms;
                            prof.wave_gpu_ms = led.wave_gpu_ms;
                            prof.physical_nvme_mb = led.nvme_bytes as f64 / (1024.0 * 1024.0);
                            prof.gpu_qkv_o_ms = led.gpu_qkv_o_ms;
                            prof.gpu_attn_ms = led.gpu_attn_ms;
                            prof.gpu_router_ms = led.gpu_router_ms;
                            prof.gpu_shared_ms = led.gpu_shared_ms;
                            prof.gpu_gateup_ms = led.gpu_gateup_ms;
                            prof.gpu_down_ms = led.gpu_down_ms;
                            prof.gpu_resid_ms = led.gpu_resid_ms;
                            prof.cp_command_encoding_ms += led.encode_ms;
                            prof.cp_gpu_waits_ms += led.slot_wait_ms
                                + led.final_wait_ms
                                + led.wave_gpu_ms
                                + led.overflow_wait_ms;
                            prof.cp_cache_slot_lookup_ms += filler_cpu + led.setup_ms;
                            prof.physical_ssd_reads_ms += led.nvme_ms;
                            prof.physical_ssd_bytes += led.nvme_bytes;
                            prof.pure_gpu_ms += led.gpu_busy_ms;
                            prof.cp_other_ms += led.upload_ms + led.rope_ms + led.download_ms;
                            prof.prefetch_late_count += led.demand_loads;
                            let _ = round_dur;
                        }
                    }
                }
                #[cfg(not(target_os = "macos"))]
                {
                    prof.cp_attention_common_core_ms += round_dur.as_secs_f64() * 1000.0;
                    prof.pure_gpu_ms += round_dur.as_secs_f64() * 1000.0;
                }
            }
        } else {
            if let Some(dir) = std::env::var_os("CAMELID_GEMMA4_DUMP_DIR") {
                let _ = std::fs::write(
                    std::path::PathBuf::from(dir).join("gpu_chained_round.txt"),
                    "ok=0 fallback=1\n",
                );
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

                // --- attention projections, GPU-accelerated on Metal with CPU fallback ---
                let mut gpu_attn_ok = false;
                let mut gpu_shared_mlp_ok = false;
                let mut gpu_router_ok = false;
                let t_gpu_attn_start = std::time::Instant::now();

                #[cfg(target_os = "macos")]
                if self.ghost_metal_q4_is_enabled() {
                    if let Ok(mut guard) = self.metal_q4_experts.lock() {
                        if let Some(lane) = guard.as_mut() {
                            if lw.moe.is_some() {
                                gpu_shared_mlp_ok = lane.execute_attention_and_shared_chunk_into(
                                    li,
                                    &hs,
                                    theta,
                                    rope_factors,
                                    start_pos,
                                    &mut hs_buf,
                                    &mut shared_mlp_buf,
                                    Some(&mut router_logits_buf),
                                    li == 0,
                                );
                                gpu_attn_ok = gpu_shared_mlp_ok;
                                gpu_router_ok = gpu_shared_mlp_ok;
                            } else {
                                gpu_attn_ok = lane.execute_attention_chunk_into(
                                    li,
                                    &hs,
                                    theta,
                                    rope_factors,
                                    start_pos,
                                    &mut hs_buf,
                                );
                            }
                        }
                    }
                }

                if gpu_attn_ok {
                    let attn_dur = t_gpu_attn_start.elapsed();
                    std::mem::swap(&mut hs, &mut hs_buf);
                    if let Some(prof) = profile.as_deref_mut() {
                        prof.cp_attention_common_core_ms += attn_dur.as_secs_f64() * 1000.0;
                        prof.attention_core_ms += attn_dur.as_secs_f64() * 1000.0;
                    }
                } else {
                    let t_attn_proj_start = std::time::Instant::now();
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
                    let attn_q_dur = t_attn_proj_start.elapsed();

                    let t_rope_q_start = std::time::Instant::now();
                    for (i, q) in q_rows.iter_mut().enumerate() {
                        apply_rope(q, heads, head_dim, start_pos + i, theta, rope_factors);
                    }
                    let rope_q_dur = t_rope_q_start.elapsed();

                    let mut attn_kv_dur = std::time::Duration::ZERO;
                    let mut rope_kv_dur = std::time::Duration::ZERO;
                    if l < self.first_kv_shared {
                        let t_kv_proj_start = std::time::Instant::now();
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
                                    Some(lw.k_norm.as_deref().expect(
                                        "validate() guarantees owning layers bind attn_k_norm",
                                    )),
                                    eps,
                                ));
                                // Gemma 4 reference: V takes a weightless per-head
                                // RMS norm before caching (and never RoPE).
                                let sv = &mut v_rows[i][hh * head_dim..(hh + 1) * head_dim];
                                sv.copy_from_slice(&rms_norm(sv, None, eps));
                            }
                        }
                        attn_kv_dur = t_kv_proj_start.elapsed();

                        let t_rope_kv_start = std::time::Instant::now();
                        for (i, k) in k_rows.iter_mut().enumerate() {
                            apply_rope(k, kv_heads, head_dim, start_pos + i, theta, rope_factors);
                        }
                        rope_kv_dur = t_rope_kv_start.elapsed();

                        for i in 0..kk {
                            kc[li].push(k_rows[i].clone());
                            vc[li].push(v_rows[i].clone());
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
                    let t_attn_core_start = std::time::Instant::now();
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
                            if pos >= kc[src].len() {
                                panic!(
                                "[step_chunk crash] layer l={l} src={src} start_pos={start_pos} i={i} pos={pos} kk={kk} kc[src].len()={}",
                                kc[src].len()
                            );
                            }
                            // Gemma 4 attention scale is 1.0: the per-head QK
                            // RMS-norms replace the classic 1/sqrt(head_dim). The
                            // base (oracle-exact) chunk path has no score scale;
                            // adding head_dim^-0.5 here flattened every softmax.
                            let mut scores: Vec<f32> = (lo..=pos)
                                .map(|p| {
                                    let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                                    qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>()
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
                    let attn_core_dur = t_attn_core_start.elapsed();

                    if let Some(prof) = profile.as_deref_mut() {
                        prof.cp_attention_common_core_ms +=
                            (attn_q_dur + attn_kv_dur + attn_core_dur).as_secs_f64() * 1000.0;
                        prof.cp_kv_rope_ms += (rope_q_dur + rope_kv_dur).as_secs_f64() * 1000.0;
                        prof.attention_core_ms +=
                            (attn_q_dur + attn_kv_dur + attn_core_dur + rope_q_dur + rope_kv_dur)
                                .as_secs_f64()
                                * 1000.0;
                    }
                }

                // --- FFN, batched ---
                let t_moe_start = std::time::Instant::now();
                let ffn_out_rows = if lw.moe.is_some() {
                    let precomputed_shared = if gpu_shared_mlp_ok {
                        Some(shared_mlp_buf.as_slice())
                    } else {
                        None
                    };
                    let precomputed_router = if gpu_router_ok {
                        Some(router_logits_buf.as_slice())
                    } else {
                        None
                    };
                    self.moe_layer_ffn_chunk(
                        li,
                        &hs,
                        precomputed_shared,
                        precomputed_router,
                        profile.as_deref_mut(),
                    )?
                } else {
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
                    lw.ffn_down
                        .matmul_proj(hidden, &actq)
                        .into_iter()
                        .map(|mlp| rms_norm(&mlp, Some(&lw.post_ffw_norm), eps))
                        .collect()
                };
                let moe_dur = t_moe_start.elapsed();
                if let Some(prof) = profile.as_deref_mut() {
                    prof.all_moe_layers_ms += moe_dur.as_secs_f64() * 1000.0;
                    if li == 0 {
                        prof.layer0_moe_ms = moe_dur.as_secs_f64() * 1000.0;
                    }
                }

                let t_ple_start = std::time::Instant::now();
                if !gpu_shared_mlp_ok {
                    hs.iter_mut().zip(&ffn_out_rows).for_each(|(h, f)| {
                        for (a, b) in h.iter_mut().zip(f) {
                            *a += b;
                        }
                    });
                }
                for i in 0..kk {
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
                let ple_dur = t_ple_start.elapsed();
                if let Some(prof) = profile.as_deref_mut() {
                    prof.cp_other_ms += ple_dur.as_secs_f64() * 1000.0;
                }

                if std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                    for (i, h) in hs.iter().enumerate() {
                        let l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                        eprintln!(
                        "[h_chunk] tok {i} layer {li} l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                        h[0], h[1], h[2], h[3]
                    );
                    }
                }
            }
        }

        // --- tied head ---
        let t_head_start = std::time::Instant::now();
        let vocab = self.config.vocab_size.unwrap() as usize;
        let logits_res = if !all_logits {
            let logits =
                self.project_logits(hs.last().expect("non-empty chunk has a final hidden row"));
            vec![logits]
        } else {
            #[cfg(target_os = "macos")]
            {
                if ghost_metal_acceleration_enabled() {
                    if let Some(head) = self.metal_q6k_head.as_ref() {
                        if let Some(batched_logits) = head.forward_batch(&hs) {
                            batched_logits
                        } else {
                            hs.iter().map(|h| self.project_logits(h)).collect()
                        }
                    } else {
                        self.project_logits_chunk_cpu(&hs, eps, vocab)
                    }
                } else {
                    self.project_logits_chunk_cpu(&hs, eps, vocab)
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                self.project_logits_chunk_cpu(&hs, eps, vocab)
            }
        };
        let head_dur = t_head_start.elapsed();
        if let Some(prof) = profile {
            prof.cp_output_head_ms += head_dur.as_secs_f64() * 1000.0;
            prof.wall_clock_ms = t_round_start.elapsed().as_secs_f64() * 1000.0;
            prof.total_wall_clock_ms = prof.wall_clock_ms;
            if gpu_chained_round_ok {
                prof.cpu_only_exposed_ms = prof.cp_other_ms
                    + prof.cp_command_encoding_ms
                    + prof.cp_cache_slot_lookup_ms
                    + prof.cp_output_head_ms;
                prof.gpu_only_exposed_ms = 0.0;
                prof.cpu_gpu_overlapped_ms = 0.0;
                let accounted = prof.cpu_only_exposed_ms
                    + prof.physical_ssd_reads_ms
                    + prof.cp_gpu_waits_ms
                    + prof.chained_prefetch_ms;
                prof.synchronization_gap_ms = (prof.total_wall_clock_ms - accounted).max(0.0);
            } else {
                prof.cpu_only_exposed_ms = prof.cp_attention_common_core_ms
                    + prof.cp_router_topk_ms
                    + prof.cp_kv_rope_ms
                    + prof.cp_cache_slot_lookup_ms
                    + prof.cp_command_encoding_ms
                    + prof.cp_other_ms
                    + prof.cp_output_head_ms;
                prof.gpu_only_exposed_ms = prof.cp_routed_moe_gate_up_ms
                    + prof.cp_routed_moe_quant_ms
                    + prof.cp_routed_moe_down_ms
                    + prof.cp_shared_expert_ms;
                prof.cpu_gpu_overlapped_ms = 0.0;
                prof.synchronization_gap_ms = (prof.total_wall_clock_ms
                    - (prof.cpu_only_exposed_ms + prof.gpu_only_exposed_ms))
                    .max(0.0);
            }
        }
        Ok(logits_res)
    }

    fn ghost_metal_q4_is_enabled(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            ghost_metal_acceleration_enabled()
                && self
                    .metal_q4_experts
                    .lock()
                    .is_ok_and(|lane| lane.is_some())
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn ghost_metal_q4_slots_per_layer(&self) -> Option<usize> {
        #[cfg(target_os = "macos")]
        {
            if !ghost_metal_acceleration_enabled() {
                return None;
            }
            self.metal_q4_experts
                .lock()
                .ok()
                .and_then(|lane| lane.as_ref().map(GhostMetalExpertRuntime::slots_per_layer))
                .filter(|&slots| slots > 0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            None
        }
    }

    fn try_ghost_metal_q4_experts(
        &self,
        ghost: &GhostMoeLayer,
        experts: &[usize],
        route_scales: &[f32],
        input: &[Q8_0Block],
        hidden: usize,
    ) -> Option<Vec<f32>> {
        #[cfg(target_os = "macos")]
        {
            if !ghost_metal_acceleration_enabled() {
                return None;
            }
            let mut guard = self.metal_q4_experts.lock().ok()?;
            let lane = guard.as_mut()?;
            let empty_sources = std::collections::HashMap::new();
            match lane.run_layer(ghost, experts, route_scales, input, hidden, &empty_sources) {
                GhostMetalExpertAttempt::Output(output) => Some(output),
                GhostMetalExpertAttempt::CpuFallback => None,
                GhostMetalExpertAttempt::DisableMetal => {
                    eprintln!(
                        "[gemma4-ghost-metal] Metal expert dispatch failed; disabling persistent slots and using CPU Ghost experts"
                    );
                    *guard = None;
                    None
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (ghost, experts, route_scales, input, hidden);
            None
        }
    }

    fn prewarm_ghost_metal_q4(
        &self,
        layer_idx: usize,
        request_sequence: &[usize],
        records: &std::collections::HashMap<usize, Arc<GhostMoeExpert>>,
    ) {
        #[cfg(target_os = "macos")]
        if ghost_metal_acceleration_enabled() {
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    let _ = lane.prewarm_layer_from_records(layer_idx, request_sequence, records);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (layer_idx, request_sequence, records);
        }
    }

    /// Fast (<300us) prediction of all 30 layers' top-k routed experts for candidate tokens
    /// using token embeddings and per-layer router projections.
    pub fn predict_all_layer_routes_for_chunk(&self, tokens: &[u32]) -> Result<Vec<Vec<usize>>> {
        self.predict_all_layer_routes_for_chunk_top_n(tokens, 16)
    }

    /// Parameterized Top-N prediction of all 30 layers' routed experts.
    pub fn predict_all_layer_routes_for_chunk_top_n(
        &self,
        tokens: &[u32],
        top_n: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let hidden = self.config.embedding_length as usize;
        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let h0: Vec<f32> = self
                .token_embd
                .dequantize_elements(token as usize * hidden, hidden)?
                .iter()
                .map(|v| v * (hidden as f32).sqrt())
                .collect();
            hs.push(h0);
        }
        let mut predicted_per_layer = Vec::with_capacity(self.layers.len());
        for li in 0..self.layers.len() {
            predicted_per_layer.push(self.predict_next_layer_routes_top_k(li, &hs, top_n));
        }
        Ok(predicted_per_layer)
    }

    /// Fast (<5us) prediction of the next layer's top-k routed experts using
    /// current attention-residual approximation.
    fn predict_next_layer_routes(
        &self,
        next_li: usize,
        approx_hidden_rows: &[Vec<f32>],
    ) -> Vec<usize> {
        self.predict_next_layer_routes_top_k(next_li, approx_hidden_rows, 8)
    }

    fn predict_next_layer_routes_top_k(
        &self,
        next_li: usize,
        approx_hidden_rows: &[Vec<f32>],
        top_k: usize,
    ) -> Vec<usize> {
        let Some(lw) = self.layers.get(next_li) else {
            return Vec::new();
        };
        let Some(moe) = lw.moe.as_ref() else {
            return Vec::new();
        };
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let inv = 1.0f32 / (hidden as f32).sqrt();
        let mut selected = vec![false; moe.n_expert];
        for h in approx_hidden_rows {
            let mut r = rms_norm(h, None, eps);
            for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
                *rv = *rv * inv * sv;
            }
            let logits = f32_matvec(&moe.gate_inp, hidden, moe.n_expert, &r);
            let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
            let probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
            let mut idx: Vec<usize> = (0..moe.n_expert).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b)));
            idx.truncate(top_k.min(moe.n_expert));
            for &e in &idx {
                selected[e] = true;
            }
        }
        selected
            .iter()
            .enumerate()
            .filter_map(|(e, &is_selected)| is_selected.then_some(e))
            .collect()
    }

    /// Layer-major sibling of [`Self::moe_layer_ffn`] for Ghost prompt chunks.
    ///
    /// Routing is still computed independently for every token row. After all
    /// routes are known, the union of selected experts is fetched in one cache
    /// operation and each expert's immutable record is reused by every routed
    /// row. Expert projections are batched per expert, but each token's final
    /// mixture is accumulated in its original top-k route order. That last
    /// detail keeps the floating-point addition order identical to the scalar
    /// forward rather than making output depend on the union's expert order.
    fn moe_layer_ffn_chunk(
        &self,
        li: usize,
        attn_rows: &[Vec<f32>],
        precomputed_shared_mlp: Option<&[Vec<f32>]>,
        precomputed_router_logits: Option<&[f32]>,
        mut profile: Option<&mut Gemma4ChunkRoundProfile>,
    ) -> Result<Vec<Vec<f32>>> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn_chunk called on a non-MoE layer");
        let ghost = moe
            .ghost
            .as_ref()
            .expect("chunk forward admits only Ghost-backed MoE layers");
        let n_exp = moe.n_expert;
        let n_used = moe.n_expert_used;

        if let Some(logits_all) = precomputed_router_logits {
            if precomputed_shared_mlp.is_some() {
                let n_tokens = attn_rows.len().min(8);
                let mut route_indices = [[0usize; 8]; 8];
                let mut route_probs = [[0.0f32; 128]; 8];
                let mut route_wsums = [0.0f32; 8];
                let mut selected = [false; 128];
                let mut unique_experts_buf = [0usize; 128];
                let mut num_unique = 0;

                let t_router_start = std::time::Instant::now();
                for token_idx in 0..n_tokens {
                    let logits = &logits_all[token_idx * n_exp..(token_idx + 1) * n_exp];
                    let mut maxl = f32::MIN;
                    for &v in logits {
                        if v > maxl {
                            maxl = v;
                        }
                    }
                    let mut probs = [0.0f32; 128];
                    let mut sum = 0.0f32;
                    for e in 0..n_exp {
                        let p = (logits[e] - maxl).exp();
                        probs[e] = p;
                        sum += p;
                    }
                    let inv_sum = 1.0f32 / sum;
                    for e in 0..n_exp {
                        probs[e] *= inv_sum;
                    }
                    route_probs[token_idx][..n_exp].copy_from_slice(&probs[..n_exp]);

                    let mut idx = [0usize; 128];
                    for e in 0..n_exp {
                        idx[e] = e;
                    }
                    for i in 0..n_used {
                        let mut max_j = i;
                        let mut max_p = probs[idx[i]];
                        for j in (i + 1)..n_exp {
                            let p = probs[idx[j]];
                            if p > max_p {
                                max_p = p;
                                max_j = j;
                            }
                        }
                        idx.swap(i, max_j);
                    }

                    let mut wsum = 0.0f32;
                    for i in 0..n_used {
                        let e = idx[i];
                        route_indices[token_idx][i] = e;
                        wsum += probs[e];
                        selected[e] = true;
                    }
                    route_wsums[token_idx] = wsum.max(6.103_515e-5);
                }
                for e in 0..n_exp {
                    if selected[e] {
                        unique_experts_buf[num_unique] = e;
                        num_unique += 1;
                    }
                }
                let router_dur = t_router_start.elapsed();
                if let Some(prof) = profile.as_deref_mut() {
                    prof.cp_router_topk_ms += router_dur.as_secs_f64() * 1000.0;
                }

                let unique_experts = &unique_experts_buf[..num_unique];

                #[cfg(target_os = "macos")]
                {
                    let t_slot_start = std::time::Instant::now();
                    let mut slab_and_slots = None;
                    if let Ok(mut guard) = self.metal_q4_experts.lock() {
                        if let Some(lane) = guard.as_mut() {
                            if let Some((slab_buf, slot_indices)) =
                                lane.try_resolve_resident_chunk(ghost.layer_idx, unique_experts)
                            {
                                slab_and_slots =
                                    Some((slab_buf as *const metal::Buffer, slot_indices));
                            }
                            lane.record_layer_routes(ghost.layer_idx, unique_experts);
                        }
                    }
                    let slot_dur = t_slot_start.elapsed();
                    if let Some(prof) = profile.as_deref_mut() {
                        prof.cp_cache_slot_lookup_ms += slot_dur.as_secs_f64() * 1000.0;
                    }

                    if let Some((raw_slab, slot_indices)) = slab_and_slots {
                        let slab_buf = unsafe { &*raw_slab };
                        let mut expert_to_unique = [0u32; 128];
                        for (u, &e) in unique_experts.iter().enumerate() {
                            expert_to_unique[e] = u as u32;
                        }

                        let mut candidate_masks = [0u32; 128];
                        for r in 0..n_tokens {
                            let bit = 1u32 << r;
                            for i in 0..n_used {
                                let e = route_indices[r][i];
                                candidate_masks[e] |= bit;
                            }
                        }

                        let mut work_items = [crate::metal::Gemma4UniqueExpertWork::default(); 128];
                        for (u, &e) in unique_experts.iter().enumerate() {
                            let mask = candidate_masks[e];
                            let slot = slot_indices[u];
                            work_items[u] = crate::metal::Gemma4UniqueExpertWork {
                                candidate_mask: mask as u64,
                                expert_weight_offset: (slot
                                    * crate::metal::GEMMA4_Q4_EXPERT_SLOT_STRIDE)
                                    as u32,
                                slab_index: 0,
                            };
                        }

                        let mut route_entries =
                            [crate::metal::Gemma4CandidateRouteEntry::default(); 128];
                        let mut entry_idx = 0;
                        for r in 0..n_tokens {
                            for i in 0..n_used {
                                let e = route_indices[r][i];
                                let u = expert_to_unique[e];
                                let w =
                                    (route_probs[r][e] / route_wsums[r]) * moe.down_exps_scale[e];
                                route_entries[entry_idx] =
                                    crate::metal::Gemma4CandidateRouteEntry {
                                        unique_expert_idx: u,
                                        weight: w,
                                    };
                                entry_idx += 1;
                            }
                        }

                        let mut batch_times = crate::metal::Gemma4MultiExpertBatchTimes::default();
                        if let Ok(guard) = self.metal_q4_experts.lock() {
                            if let Some(lane) = guard.as_ref() {
                                if let Some((scales_buf, quants_buf)) =
                                    lane.resident_expert_input_buffers()
                                {
                                    let fused_tail =
                                        lane.resident_fused_tail_buffers(ghost.layer_idx);
                                    crate::metal::try_gemma4_q4_multi_expert_layer_chunk_with_gpu_quants(
                                        scales_buf,
                                        quants_buf,
                                        n_tokens,
                                        slab_buf,
                                        num_unique,
                                        &work_items[..num_unique],
                                        &route_entries[..entry_idx],
                                        &mut [],
                                        Some(&mut batch_times),
                                        fused_tail,
                                    );
                                }
                            }
                        }

                        if let Some(prof) = profile.as_deref_mut() {
                            prof.shared_expert_metal_calls += 1;
                            prof.cp_command_encoding_ms += batch_times.prep_us as f64 / 1000.0;
                            prof.cp_gpu_waits_ms += batch_times.commit_wait_us as f64 / 1000.0;
                            let moe_gpu_us = batch_times.gpu_busy_us as f64;
                            prof.cp_routed_moe_gate_up_ms += (moe_gpu_us * (2.0 / 3.0)) / 1000.0;
                            prof.cp_routed_moe_quant_ms += (moe_gpu_us * 0.05) / 1000.0;
                            prof.cp_routed_moe_down_ms +=
                                (moe_gpu_us * (1.0 / 3.0 - 0.05)) / 1000.0;
                        }

                        return Ok(Vec::new());
                    }
                }
            }
        }

        // Preserve the scalar router operation order row-for-row: norm, scale,
        // F32 matvec, all-expert softmax, probability sort, selected-weight sum.
        let t_router_start = std::time::Instant::now();
        let inv = 1.0f32 / (hidden as f32).sqrt();
        let n_exp = moe.n_expert;
        let n_used = moe.n_expert_used;

        let token_results: Vec<(Vec<usize>, Vec<f32>, f32)> = if let Some(logits_all) =
            precomputed_router_logits
        {
            (0..attn_rows.len())
                .map(|token_idx| {
                    let logits = &logits_all[token_idx * n_exp..(token_idx + 1) * n_exp];
                    let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
                    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                    let sum: f32 = probs.iter().sum();
                    let inv_sum = 1.0f32 / sum;
                    for p in probs.iter_mut() {
                        *p *= inv_sum;
                    }

                    let mut idx: Vec<usize> = (0..n_exp).collect();
                    for i in 0..n_used {
                        let mut max_j = i;
                        let mut max_p = probs[idx[i]];
                        for j in (i + 1)..n_exp {
                            let p = probs[idx[j]];
                            if p > max_p {
                                max_p = p;
                                max_j = j;
                            }
                        }
                        idx.swap(i, max_j);
                    }
                    idx.truncate(n_used);

                    let mut wsum = 0.0f32;
                    for &e in &idx {
                        wsum += probs[e];
                    }
                    wsum = wsum.max(6.103_515e-5);
                    (idx, probs, wsum)
                })
                .collect()
        } else {
            // Base (oracle-locked) router operation order, row-for-row:
            // norm, scale, F32 matvec (sequential sums), all-expert softmax
            // with `/= sum`, full stable sort with index tie-break. The fused
            // rms/scale factor + NEON dot + selection-sort variant changed
            // float ordering and flipped near-tie expert selections.
            attn_rows
                .iter()
                .map(|attn_out| {
                    let mut r = rms_norm(attn_out, None, eps);
                    for (rv, sv) in r.iter_mut().zip(&moe.gate_inp_scale) {
                        *rv = *rv * inv * sv;
                    }
                    let logits = f32_matvec(&moe.gate_inp, hidden, n_exp, &r);
                    if std::env::var("CAMELID_GEMMA4_DUMP_ROUTER").is_ok_and(|v| v == "1") {
                        let l2 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
                        eprintln!(
                            "[router-cpu] l2 {l2:.6} first4 [{:.6}, {:.6}, {:.6}, {:.6}]",
                            logits[0], logits[1], logits[2], logits[3]
                        );
                    }
                    let maxl = logits.iter().cloned().fold(f32::MIN, f32::max);
                    let mut probs: Vec<f32> = logits.iter().map(|&v| (v - maxl).exp()).collect();
                    let sum: f32 = probs.iter().sum();
                    for p in probs.iter_mut() {
                        *p /= sum;
                    }
                    let mut idx: Vec<usize> = (0..n_exp).collect();
                    idx.sort_unstable_by(|&a, &b| {
                        probs[b].partial_cmp(&probs[a]).unwrap().then(a.cmp(&b))
                    });
                    idx.truncate(n_used);
                    let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
                    wsum = wsum.max(6.103_515e-5);
                    (idx, probs, wsum)
                })
                .collect()
        };

        let mut route_indices: Vec<Vec<usize>> = Vec::with_capacity(token_results.len());
        let mut route_probs: Vec<Vec<f32>> = Vec::with_capacity(token_results.len());
        let mut route_wsums: Vec<f32> = Vec::with_capacity(token_results.len());
        let mut selected = vec![false; moe.n_expert];

        for (idx, probs, wsum) in token_results {
            if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
                eprintln!("[route] l={l} e={idx:?}");
            }
            for &e in &idx {
                selected[e] = true;
            }
            route_indices.push(idx);
            route_probs.push(probs);
            route_wsums.push(wsum);
        }
        let router_dur = t_router_start.elapsed();
        if let Some(prof) = profile.as_deref_mut() {
            prof.cp_router_topk_ms += router_dur.as_secs_f64() * 1000.0;
        }

        let unique_experts: Vec<usize> = selected
            .iter()
            .enumerate()
            .filter_map(|(e, &is_selected)| is_selected.then_some(e))
            .collect();
        let routed_experts: Vec<usize> = route_indices
            .iter()
            .flat_map(|indices| indices.iter().copied())
            .collect();
        let union_count = unique_experts.len();

        // Dense shared-expert branch. The batched projections use the exact same
        // row-dot kernels as the scalar lane, only reusing each weight row across
        // all prompt activations.
        let dense_mlp = || {
            let xn_rows: Vec<Vec<f32>> = attn_rows
                .iter()
                .map(|attn_out| rms_norm(attn_out, Some(&lw.ffn_norm), eps))
                .collect();
            let xnq = SharedActivationBatch::new(&xn_rows);
            let gate_rows = lw.ffn_gate.matmul_proj(ffn_dim, &xnq);
            let up_rows = lw.ffn_up.matmul_proj(ffn_dim, &xnq);
            let act_rows: Vec<Vec<f32>> = gate_rows
                .iter()
                .zip(&up_rows)
                .map(|(gate, up)| {
                    gate.iter()
                        .zip(up)
                        .map(|(g, u)| gelu_tanh(*g) * u)
                        .collect()
                })
                .collect();
            let actq = SharedActivationBatch::new(&act_rows);
            lw.ffn_down
                .matmul_proj(hidden, &actq)
                .into_iter()
                .map(|mlp| rms_norm(&mlp, Some(&moe.post_norm_1), eps))
                .collect::<Vec<_>>()
        };

        let gpu_shared_mlp = || -> Option<Vec<Vec<f32>>> {
            #[cfg(target_os = "macos")]
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    let (gate_buf, up_buf, down_buf) = lane.get_or_init_shared_buffers(
                        ghost.layer_idx,
                        lw.ffn_gate.bytes(),
                        lw.ffn_up.bytes(),
                        lw.ffn_down.bytes(),
                    )?;
                    let xn_rows: Vec<Vec<f32>> = attn_rows
                        .iter()
                        .map(|attn_out| rms_norm(attn_out, Some(&lw.ffn_norm), eps))
                        .collect();
                    let xnq = SharedActivationBatch::new(&xn_rows);
                    let input_q8_refs: Vec<&[Q8_0Block]> =
                        xnq.q8_0().iter().map(Vec::as_slice).collect();
                    let mut out_mlp = vec![vec![0.0f32; hidden]; attn_rows.len()];
                    crate::metal::try_gemma4_q4_shared_expert_chunk(
                        &input_q8_refs,
                        gate_buf,
                        up_buf,
                        down_buf,
                        &mut out_mlp,
                        None,
                    )?;
                    for row in out_mlp.iter_mut() {
                        *row = rms_norm(row, Some(&moe.post_norm_1), eps);
                    }
                    return Some(out_mlp);
                }
            }
            None
        };

        let compute_mlp = || {
            let t_start = std::time::Instant::now();
            if let Some(pre) = precomputed_shared_mlp {
                (pre.to_vec(), true, t_start.elapsed())
            } else if let Some(gpu_out) = gpu_shared_mlp() {
                (gpu_out, true, t_start.elapsed())
            } else {
                (dense_mlp(), false, t_start.elapsed())
            }
        };

        // 1. Fast Direct Resident Slot Table Resolution (Zero Mutex on Cache, Zero HashMap, Zero Arc clones)
        let t_slot_start = std::time::Instant::now();
        let mut direct_resolved = false;
        let mut slab_buf_slot_indices = None;

        #[cfg(target_os = "macos")]
        if let Ok(mut guard) = self.metal_q4_experts.lock() {
            if let Some(lane) = guard.as_mut() {
                if let Some((slab_buf, slot_indices)) =
                    lane.try_resolve_resident_chunk(ghost.layer_idx, &unique_experts)
                {
                    slab_buf_slot_indices = Some((slab_buf as *const metal::Buffer, slot_indices));
                    direct_resolved = true;
                } else {
                    // Direct Metal slab fill on miss without CPU double-hop
                    let empty_sources = std::collections::HashMap::new();
                    if let Some((slab_buf, slot_indices)) =
                        lane.prepare_chunk_slots(ghost, &unique_experts, &empty_sources)
                    {
                        slab_buf_slot_indices =
                            Some((slab_buf as *const metal::Buffer, slot_indices));
                        direct_resolved = true;
                    }
                }
                lane.record_layer_routes(ghost.layer_idx, &unique_experts);
            }
        }
        let slot_dur = t_slot_start.elapsed();
        if let Some(prof) = profile.as_deref_mut() {
            prof.cp_cache_slot_lookup_ms += slot_dur.as_secs_f64() * 1000.0;
        }

        let mut mlp_rows = Vec::new();
        let mut multi_expert_metal_result: Option<Vec<Vec<f32>>> = None;
        let mut shared_times = crate::metal::Gemma4SharedExpertBatchTimes::default();
        let mut batch_times = crate::metal::Gemma4MultiExpertBatchTimes::default();
        let mut expert_records: Vec<Option<Arc<GhostMoeExpert>>> = Vec::new();
        let mut prewarm = None;

        if direct_resolved {
            let (raw_slab, slot_indices) = slab_buf_slot_indices.as_ref().unwrap();
            let slab_buf = unsafe { &**raw_slab };

            let mut shared_q8_store = [[crate::tensor::Q8_0Block::default(); 88]; 8];
            let mut moe_q8_store = [[crate::tensor::Q8_0Block::default(); 88]; 8];

            let n_tokens = attn_rows.len().min(8);
            if precomputed_shared_mlp.is_none() {
                for r in 0..n_tokens {
                    let attn_out = &attn_rows[r];
                    let len = attn_out.len();
                    let mut ss = 0.0f32;
                    for &v in attn_out {
                        ss += v * v;
                    }
                    let inv = 1.0 / (ss / len as f32 + eps).sqrt();
                    let mut normed = [0.0f32; 2816];

                    // 1. Shared expert input: rms_norm(attn_out, lw.ffn_norm)
                    let w = &lw.ffn_norm;
                    for i in 0..len {
                        normed[i] = attn_out[i] * inv * w[i];
                    }
                    for (b, chunk) in normed[..len].chunks_exact(32).enumerate().take(88) {
                        shared_q8_store[r][b] = crate::inference::quantize_q8_0_block(chunk);
                    }

                    // 2. Routed MoE input: rms_norm(attn_out, moe.pre_norm_2)
                    let w_moe = &moe.pre_norm_2;
                    for i in 0..len {
                        normed[i] = attn_out[i] * inv * w_moe[i];
                    }
                    for (b, chunk) in normed[..len].chunks_exact(32).enumerate().take(88) {
                        moe_q8_store[r][b] = crate::inference::quantize_q8_0_block(chunk);
                    }
                }
            }

            let shared_q8_refs: Vec<&[crate::tensor::Q8_0Block]> =
                if precomputed_shared_mlp.is_none() {
                    (0..n_tokens)
                        .map(|r| &shared_q8_store[r][..attn_rows[r].len() / 32])
                        .collect()
                } else {
                    Vec::new()
                };
            let moe_q8_refs: Vec<&[crate::tensor::Q8_0Block]> = if precomputed_shared_mlp.is_none()
            {
                (0..n_tokens)
                    .map(|r| &moe_q8_store[r][..attn_rows[r].len() / 32])
                    .collect()
            } else {
                Vec::new()
            };

            let mut expert_to_unique = [0u32; 128];
            for (u, &e) in unique_experts.iter().enumerate() {
                expert_to_unique[e] = u as u32;
            }

            let mut candidate_masks = [0u32; 128];
            for (r, idx) in route_indices.iter().enumerate() {
                let bit = 1u32 << r;
                for &e in idx {
                    candidate_masks[e] |= bit;
                }
            }

            let mut work_items = [crate::metal::Gemma4UniqueExpertWork::default(); 128];
            for (u, &e) in unique_experts.iter().enumerate() {
                let mask = candidate_masks[e];
                let slot = slot_indices[u];
                work_items[u] = crate::metal::Gemma4UniqueExpertWork {
                    candidate_mask: mask as u64,
                    expert_weight_offset: (slot * crate::metal::GEMMA4_Q4_EXPERT_SLOT_STRIDE)
                        as u32,
                    slab_index: 0,
                };
            }

            let mut route_entries = [crate::metal::Gemma4CandidateRouteEntry::default(); 128];
            let mut entry_idx = 0;
            for r in 0..attn_rows.len() {
                for &e in &route_indices[r] {
                    let u = expert_to_unique[e];
                    let w = (route_probs[r][e] / route_wsums[r]) * moe.down_exps_scale[e];
                    route_entries[entry_idx] = crate::metal::Gemma4CandidateRouteEntry {
                        unique_expert_idx: u,
                        weight: w,
                    };
                    entry_idx += 1;
                }
            }

            let mut output_mlp = vec![vec![0.0f32; hidden]; attn_rows.len()];
            let mut output_moe = vec![vec![0.0f32; hidden]; attn_rows.len()];

            #[cfg(target_os = "macos")]
            let fused_ok = if precomputed_shared_mlp.is_some() {
                let mut dispatched = false;
                if let Ok(guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_ref() {
                        if let Some((scales_buf, quants_buf)) = lane.resident_expert_input_buffers()
                        {
                            let fused_tail = lane.resident_fused_tail_buffers(ghost.layer_idx);
                            dispatched = crate::metal::try_gemma4_q4_multi_expert_layer_chunk_with_gpu_quants(
                                scales_buf,
                                quants_buf,
                                attn_rows.len(),
                                slab_buf,
                                unique_experts.len(),
                                &work_items[..unique_experts.len()],
                                &route_entries[..entry_idx],
                                &mut output_moe,
                                Some(&mut batch_times),
                                fused_tail,
                            ).is_some();
                        }
                    }
                }
                if !dispatched {
                    crate::metal::try_gemma4_q4_multi_expert_layer_chunk_with_norm(
                        attn_rows,
                        &moe.pre_norm_2,
                        eps,
                        slab_buf,
                        unique_experts.len(),
                        &work_items[..unique_experts.len()],
                        &route_entries[..entry_idx],
                        &mut output_moe,
                        Some(&mut batch_times),
                    )
                    .is_some()
                } else {
                    true
                }
            } else if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    if let Some((gate_buf, up_buf, down_buf)) = lane.get_or_init_shared_buffers(
                        ghost.layer_idx,
                        lw.ffn_gate.bytes(),
                        lw.ffn_up.bytes(),
                        lw.ffn_down.bytes(),
                    ) {
                        crate::metal::try_gemma4_q4_fused_moe_layer_chunk(
                            &shared_q8_refs,
                            gate_buf,
                            up_buf,
                            down_buf,
                            &moe_q8_refs,
                            slab_buf,
                            unique_experts.len(),
                            &work_items[..unique_experts.len()],
                            &route_entries[..entry_idx],
                            &mut output_mlp,
                            &mut output_moe,
                            Some(&mut shared_times),
                            Some(&mut batch_times),
                        )
                        .is_some()
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            #[cfg(not(target_os = "macos"))]
            let fused_ok = false;

            if fused_ok {
                if let Some(pre) = precomputed_shared_mlp {
                    mlp_rows = pre.to_vec();
                } else {
                    for row in output_mlp.iter_mut() {
                        *row = rms_norm(row, Some(&moe.post_norm_1), eps);
                    }
                    mlp_rows = output_mlp;
                }
                multi_expert_metal_result = Some(output_moe);

                if let Some(prof) = profile.as_deref_mut() {
                    prof.shared_expert_metal_calls += 1;
                    prof.shared_gate_up_gpu_ms += (shared_times.gate_up_gpu_us as f64) / 1000.0;
                    prof.shared_geglu_gpu_ms += (shared_times.geglu_gpu_us as f64) / 1000.0;
                    prof.shared_down_gpu_ms += (shared_times.down_gpu_us as f64) / 1000.0;
                    prof.shared_expert_gpu_busy_ms += (shared_times.gpu_busy_us as f64) / 1000.0;
                    let shared_wall_ms = (shared_times.commit_wait_us as f64
                        + shared_times.prep_us as f64
                        + shared_times.readout_us as f64)
                        / 1000.0;
                    prof.shared_expert_wall_ms += shared_wall_ms;
                    prof.cpu_dense_shared_mlp_ms += shared_wall_ms;

                    // Non-overlapping critical-path metrics:
                    prof.cp_command_encoding_ms +=
                        (batch_times.prep_us as f64 + shared_times.prep_us as f64) / 1000.0;
                    prof.cp_gpu_waits_ms += (batch_times.commit_wait_us as f64) / 1000.0;
                    prof.cp_shared_expert_ms += (shared_times.gpu_busy_us as f64) / 1000.0;
                    let moe_gpu_us = (batch_times
                        .gpu_busy_us
                        .saturating_sub(shared_times.gpu_busy_us))
                        as f64;
                    prof.cp_routed_moe_gate_up_ms += (moe_gpu_us * (2.0 / 3.0)) / 1000.0;
                    prof.cp_routed_moe_quant_ms += (moe_gpu_us * 0.05) / 1000.0;
                    prof.cp_routed_moe_down_ms += (moe_gpu_us * (1.0 / 3.0 - 0.05)) / 1000.0;
                }
            } else {
                let (mlp, is_gpu, dense_dur) = compute_mlp();
                mlp_rows = mlp;
                if let Some(prof) = profile.as_deref_mut() {
                    if is_gpu {
                        prof.shared_expert_metal_calls += 1;
                    } else {
                        prof.shared_expert_cpu_calls += 1;
                        prof.shared_expert_cpu_fallback_ms += dense_dur.as_secs_f64() * 1000.0;
                    }
                    prof.cpu_dense_shared_mlp_ms += dense_dur.as_secs_f64() * 1000.0;
                }
            }
        } else {
            // Fallback for slot miss / cold route
            let t_join_start = std::time::Instant::now();
            let dense_mlp_timed = || compute_mlp();

            let t_cache_start = std::time::Instant::now();
            let cache_get_timed = || {
                let res = ghost.cache.get_many(ghost.layer_idx, &routed_experts);
                (res, t_cache_start.elapsed())
            };

            let ((paged_experts, cache_dur), (mlp, is_gpu, dense_dur)) =
                rayon::join(cache_get_timed, dense_mlp_timed);
            let join_dur = t_join_start.elapsed();

            if let Some(prof) = profile.as_deref_mut() {
                prof.ssd_cache_ms += join_dur.as_secs_f64() * 1000.0;
                prof.cpu_dense_shared_mlp_ms += dense_dur.as_secs_f64() * 1000.0;
                prof.cache_lookup_ms += cache_dur.as_secs_f64() * 1000.0;
                prof.synchronization_ms += (join_dur.as_secs_f64()
                    - cache_dur.as_secs_f64().max(dense_dur.as_secs_f64()))
                .max(0.0)
                    * 1000.0;
                if is_gpu {
                    prof.shared_expert_metal_calls += 1;
                } else {
                    prof.shared_expert_cpu_calls += 1;
                    prof.shared_expert_cpu_fallback_ms += dense_dur.as_secs_f64() * 1000.0;
                }
            }
            let paged_experts = paged_experts?;
            let mut records: Vec<Option<Arc<GhostMoeExpert>>> = vec![None; moe.n_expert];
            for (&e, expert) in routed_experts.iter().zip(paged_experts) {
                records[e] = Some(expert);
            }
            mlp_rows = mlp;
            expert_records = records;

            if let Some(slots_per_layer) = self.ghost_metal_q4_slots_per_layer() {
                let request_sequence =
                    ghost_metal_prewarm_sequence(&routed_experts, moe.n_expert, slots_per_layer);
                let records = request_sequence
                    .iter()
                    .copied()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .filter_map(|expert| {
                        expert_records
                            .get(expert)
                            .and_then(|rec| rec.as_ref())
                            .map(|r| (expert, Arc::clone(r)))
                    })
                    .collect::<std::collections::HashMap<_, _>>();
                prewarm = Some((request_sequence, records));
            }
        }

        if multi_expert_metal_result.is_none() {
            let cur_moe_rows: Vec<Vec<f32>> = attn_rows
                .iter()
                .map(|attn_out| rms_norm(attn_out, Some(&moe.pre_norm_2), eps))
                .collect();
            let xq = SharedActivationBatch::new(&cur_moe_rows);
            let input_q8_refs: Vec<&[Q8_0Block]> = xq.q8_0().iter().map(Vec::as_slice).collect();

            // Zero-copy resident slot execution if metal_q4_experts is active
            #[cfg(target_os = "macos")]
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    let slab_and_slots = if let Some((raw_slab, slots)) = slab_buf_slot_indices {
                        unsafe { Some((&*raw_slab, slots)) }
                    } else {
                        let resident_sources = unique_experts
                            .iter()
                            .filter_map(|&e| {
                                expert_records
                                    .get(e)
                                    .and_then(|rec| rec.as_ref())
                                    .map(|r| (e, Arc::clone(r)))
                            })
                            .collect::<std::collections::HashMap<_, _>>();
                        lane.prepare_chunk_slots(ghost, &unique_experts, &resident_sources)
                    };

                    if let Some((slab_buf, slot_indices)) = slab_and_slots {
                        let unique_map: std::collections::HashMap<usize, usize> = unique_experts
                            .iter()
                            .enumerate()
                            .map(|(u, &e)| (e, u))
                            .collect();

                        let mut work_items = Vec::with_capacity(unique_experts.len());
                        for (u, &e) in unique_experts.iter().enumerate() {
                            let mut mask = 0u32;
                            for (r, idx) in route_indices.iter().enumerate() {
                                if idx.contains(&e) {
                                    mask |= 1 << r;
                                }
                            }
                            let slot = slot_indices[u];
                            work_items.push(crate::metal::Gemma4UniqueExpertWork {
                                candidate_mask: mask as u64,
                                expert_weight_offset: (slot
                                    * crate::metal::GEMMA4_Q4_EXPERT_SLOT_STRIDE)
                                    as u32,
                                slab_index: 0,
                            });
                        }

                        let mut route_entries = Vec::with_capacity(attn_rows.len() * 8);
                        for r in 0..attn_rows.len() {
                            for &e in &route_indices[r] {
                                if let Some(&u) = unique_map.get(&e) {
                                    let w = (route_probs[r][e] / route_wsums[r])
                                        * moe.down_exps_scale[e];
                                    route_entries.push(crate::metal::Gemma4CandidateRouteEntry {
                                        unique_expert_idx: u as u32,
                                        weight: w,
                                    });
                                }
                            }
                        }

                        let mut output_acc = vec![vec![0.0f32; hidden]; attn_rows.len()];
                        if crate::metal::try_gemma4_q4_multi_expert_layer_chunk_with_buffer(
                            &input_q8_refs,
                            slab_buf,
                            unique_experts.len(),
                            &work_items,
                            &route_entries,
                            &mut output_acc,
                            Some(&mut batch_times),
                        )
                        .is_some()
                        {
                            multi_expert_metal_result = Some(output_acc);
                        }
                    }
                }
            }

            if multi_expert_metal_result.is_none() {
                // Fallback: copied unique expert records
                let mut unique_expert_records = Vec::with_capacity(unique_experts.len());
                for &e in &unique_experts {
                    if let Some(expert) = expert_records.get(e).and_then(|r| r.as_ref()) {
                        unique_expert_records.push(expert.record_bytes());
                    }
                }

                if unique_expert_records.len() == unique_experts.len() {
                    let unique_map: std::collections::HashMap<usize, usize> = unique_experts
                        .iter()
                        .enumerate()
                        .map(|(u, &e)| (e, u))
                        .collect();

                    let mut work_items = Vec::with_capacity(unique_experts.len());
                    for (u, &e) in unique_experts.iter().enumerate() {
                        let mut mask = 0u64;
                        for (r, idx) in route_indices.iter().enumerate() {
                            if idx.contains(&e) {
                                mask |= 1u64 << r;
                            }
                        }
                        work_items.push(crate::metal::Gemma4UniqueExpertWork {
                            candidate_mask: mask,
                            expert_weight_offset: (u * 3345408) as u32,
                            slab_index: 0,
                        });
                    }

                    let mut route_entries = Vec::with_capacity(attn_rows.len() * 8);
                    for r in 0..attn_rows.len() {
                        for &e in &route_indices[r] {
                            if let Some(&u) = unique_map.get(&e) {
                                let w =
                                    (route_probs[r][e] / route_wsums[r]) * moe.down_exps_scale[e];
                                route_entries.push(crate::metal::Gemma4CandidateRouteEntry {
                                    unique_expert_idx: u as u32,
                                    weight: w,
                                });
                            }
                        }
                    }

                    let mut output_acc = vec![vec![0.0f32; hidden]; attn_rows.len()];
                    if crate::metal::try_gemma4_q4_multi_expert_layer_chunk(
                        &input_q8_refs,
                        &unique_expert_records,
                        &work_items,
                        &route_entries,
                        &mut output_acc,
                        Some(&mut batch_times),
                    )
                    .is_some()
                    {
                        multi_expert_metal_result = Some(output_acc);
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(attn_rows.len());

        if let Some(output_acc) = multi_expert_metal_result {
            if let Some(prof) = profile.as_deref_mut() {
                prof.pure_gpu_ms += batch_times.gpu_busy_us as f64 / 1000.0;
                prof.command_buffers += 1;
                prof.cpu_waits += 1;
                if li == 0 {
                    prof.layer0_buffer_prep_ms = batch_times.prep_us as f64 / 1000.0;
                    prof.layer0_commit_wait_ms = batch_times.commit_wait_us as f64 / 1000.0;
                    prof.layer0_weighted_reduce_ms = batch_times.readout_us as f64 / 1000.0;
                    let gpu_ms = batch_times.gpu_busy_us as f64 / 1000.0;
                    prof.layer0_gate_up_ms = gpu_ms * (2.0 / 3.0);
                    prof.layer0_geglu_quant_ms = gpu_ms * 0.05;
                    prof.layer0_down_ms = gpu_ms * (1.0 / 3.0);
                }
            }

            if let Some((request_sequence, records)) = &prewarm {
                self.prewarm_ghost_metal_q4(ghost.layer_idx, request_sequence, records);
            }

            if precomputed_shared_mlp.is_none() {
                let w_post2 = &moe.post_norm_2;
                let w_ffw = &lw.post_ffw_norm;

                mlp_rows
                    .iter_mut()
                    .zip(&output_acc)
                    .for_each(|(d, moe_acc)| {
                        let len = moe_acc.len();
                        let mut ss = 0.0f32;
                        for &v in moe_acc {
                            ss += v * v;
                        }
                        let inv2 = (ss / len as f32 + eps).powf(-0.5);
                        for i in 0..len {
                            d[i] += moe_acc[i] * inv2 * w_post2[i];
                        }
                        let mut ss_ffw = 0.0f32;
                        for &v in d.iter() {
                            ss_ffw += v * v;
                        }
                        let inv_ffw = (ss_ffw / len as f32 + eps).powf(-0.5);
                        for i in 0..len {
                            d[i] = d[i] * inv_ffw * w_ffw[i];
                        }
                    });
            }
            out = mlp_rows;
        } else {
            let cur_moe_rows: Vec<Vec<f32>> = attn_rows
                .iter()
                .map(|attn_out| rms_norm(attn_out, Some(&moe.pre_norm_2), eps))
                .collect();
            let mut expert_outputs: Vec<Vec<Option<Vec<f32>>>> = route_indices
                .iter()
                .map(|idx| (0..idx.len()).map(|_| None).collect())
                .collect();
            let two_nff = 2 * moe.n_ff_exp;

            for &e in &unique_experts {
                let routed: Vec<(usize, usize)> = route_indices
                    .iter()
                    .enumerate()
                    .flat_map(|(row, idx)| {
                        idx.iter()
                            .enumerate()
                            .filter_map(move |(slot, &selected_e)| {
                                (selected_e == e).then_some((row, slot))
                            })
                    })
                    .collect();
                debug_assert!(!routed.is_empty());
                let x_rows: Vec<Vec<f32>> = routed
                    .iter()
                    .map(|&(row, _)| cur_moe_rows[row].clone())
                    .collect();
                let xq = SharedActivationBatch::new(&x_rows);
                let expert_vec = ghost.cache.get_many(ghost.layer_idx, &[e])?;
                let expert = expert_vec
                    .first()
                    .expect("unique Ghost expert record was not resolved");
                let gate_up =
                    WireQuant::from_ghost_tensor(expert, &expert.gate_up, "ghost gate_up_exps")?;
                let xq_refs: Vec<&[Q8_0Block]> = xq.q8_0().iter().map(Vec::as_slice).collect();
                let gate_up_rows = ghost_metal_q4_matmul(&gate_up, two_nff, &xq_refs)
                    .unwrap_or_else(|| gate_up.matmul_proj(two_nff, &xq));
                let act_rows: Vec<Vec<f32>> = gate_up_rows
                    .iter()
                    .map(|gate_up| {
                        (0..moe.n_ff_exp)
                            .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                            .collect()
                    })
                    .collect();
                let actq = SharedActivationBatch::new(&act_rows);
                let down = WireQuant::from_ghost_tensor(expert, &expert.down, "ghost down_exps")?;
                let actq_refs: Vec<&[Q8_0Block]> = actq.q8_0().iter().map(Vec::as_slice).collect();
                let y_rows = ghost_metal_q4_matmul(&down, hidden, &actq_refs)
                    .unwrap_or_else(|| down.matmul_proj(hidden, &actq));

                for ((row, slot), y) in routed.into_iter().zip(y_rows) {
                    expert_outputs[row][slot] = Some(y);
                }
            }

            if let Some((request_sequence, records)) = &prewarm {
                self.prewarm_ghost_metal_q4(ghost.layer_idx, request_sequence, records);
            }

            for row in 0..attn_rows.len() {
                let mut moe_acc = vec![0f32; hidden];
                for (slot, &e) in route_indices[row].iter().enumerate() {
                    let w = route_probs[row][e] / route_wsums[row];
                    let scale = moe.down_exps_scale[e] * w;
                    let y = expert_outputs[row][slot]
                        .take()
                        .expect("routed Ghost expert output was not computed");
                    for (a, yv) in moe_acc.iter_mut().zip(&y) {
                        *a += scale * yv;
                    }
                }
                if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                    let mlp_l2 = mlp_rows[row].iter().map(|v| v * v).sum::<f32>().sqrt();
                    let moe_l2 = moe_acc.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let moe_n = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);
                    let moe_n_l2 = moe_n.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!(
                        "[cpu chunk layer0] Mlp (Dn0) L2={mlp_l2:.6} first4={:?}",
                        &mlp_rows[row][..4]
                    );
                    eprintln!(
                        "[cpu chunk layer0] Moe0 raw L2={moe_l2:.6} first4={:?}",
                        &moe_acc[..4]
                    );
                    eprintln!(
                        "[cpu chunk layer0] Moe0 normed L2={moe_n_l2:.6} first4={:?}",
                        &moe_n[..4]
                    );
                }
                let moe_normed = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);
                for (m, d) in moe_normed.into_iter().zip(&mut mlp_rows[row]) {
                    *d += m;
                }
                let ffn_o = rms_norm(&mlp_rows[row], Some(&lw.post_ffw_norm), eps);
                if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                    let ffn_l2 = ffn_o.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!(
                        "[cpu chunk layer0] FFN out L2={ffn_l2:.6} first4={:?}",
                        &ffn_o[..4]
                    );
                }
                out.push(ffn_o);
            }
        }
        Ok(out)
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
    pub(crate) fn moe_layer_ffn(&self, li: usize, attn_out: &[f32]) -> Result<Vec<f32>> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw
            .moe
            .as_ref()
            .expect("moe_layer_ffn called on a non-MoE layer");

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
        if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
            let logit_l2 = logits.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!(
                "[cpu router layer0] Logits L2={logit_l2:.6} first4={:?}",
                &logits[..4]
            );
            eprintln!(
                "[cpu router layer0] Selected top-{}: {:?}",
                moe.n_expert_used, idx
            );
        }
        if std::env::var_os("CAMELID_GEMMA4_ROUTE_TRACE").is_some() {
            eprintln!("[route] l={l} e={idx:?}");
        }
        // sum-normalize the selected weights (clamped), w_scale=1.
        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);

        // Dense "shared expert" MLP branch: ffn_norm -> parallel GeGLU -> down.
        // On the Ghost-MoE lane its math is independent of the already-computed
        // router, so overlap it with the selected experts' positioned reads.
        // The two branches are still combined in the exact original order below.
        let dense_mlp = || {
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
            rms_norm(&mlp, Some(&moe.post_norm_1), eps)
        };
        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = SharedActivation::new(&cur_moe);
        // Materialize the tiny Q8_0 activation before `rayon::join`: the lazy
        // `SharedActivation` uses single-threaded OnceCell and is intentionally
        // not Sync, while its completed Vec of plain Q8 blocks is safe to share
        // with the independent Metal/read worker.
        let cur_moe_q8 = cur_moe_q.q8_0().to_vec();
        let route_scales: Vec<f32> = idx
            .iter()
            .map(|&e| moe.down_exps_scale[e] * (probs[e] / wsum))
            .collect();
        let (paged_experts, mlp, metal_moe_acc) = match &moe.ghost {
            Some(ghost) if self.ghost_metal_q4_is_enabled() => {
                // The Metal lane owns direct slot reads and both dominant expert
                // projections. Keep the independent shared-expert MLP on CPU at
                // the same time. If the opt-in lane soft-fails, retry through the
                // immutable host cache without changing the established result.
                let (metal, mlp) = rayon::join(
                    || {
                        self.try_ghost_metal_q4_experts(
                            ghost,
                            &idx,
                            &route_scales,
                            &cur_moe_q8,
                            hidden,
                        )
                    },
                    dense_mlp,
                );
                match metal {
                    Some(acc) => (None, mlp, Some(acc)),
                    None => (
                        Some(ghost.cache.get_many(ghost.layer_idx, &idx)?),
                        mlp,
                        None,
                    ),
                }
            }
            Some(ghost) => {
                let (experts, mlp) =
                    rayon::join(|| ghost.cache.get_many(ghost.layer_idx, &idx), dense_mlp);
                (Some(experts?), mlp, None)
            }
            None => (None, dense_mlp(), None),
        };

        let two_nff = 2 * moe.n_ff_exp;
        let moe_acc = if let Some(metal_moe_acc) = metal_moe_acc {
            metal_moe_acc
        } else {
            let mut moe_acc = vec![0f32; hidden];
            // Pre-packed (interleaved 8-row) expert matrices for the AVX2 GEMV,
            // packed once per expert per session and cached; `None` disables the
            // fast path. Paged Ghost experts deliberately use their wire record.
            for (route_slot, &e) in idx.iter().enumerate() {
                let paged = paged_experts
                    .as_ref()
                    .map(|experts| -> Result<(WireQuant, WireQuant)> {
                        let expert = &experts[route_slot];
                        Ok((
                            WireQuant::from_ghost_tensor(
                                expert,
                                &expert.gate_up,
                                "ghost gate_up_exps",
                            )?,
                            WireQuant::from_ghost_tensor(expert, &expert.down, "ghost down_exps")?,
                        ))
                    })
                    .transpose()?;
                let packed = if paged.is_none() {
                    moe.packed_expert(e, hidden, two_nff)
                } else {
                    None
                };
                // fused gate‖up for expert e: rows e*2nff .. +2nff,
                // in_dim=n_embd.
                let metal_gate_up = paged.as_ref().and_then(|(gate_up, _)| {
                    ghost_metal_q4_matmul(gate_up, two_nff, &[cur_moe_q.q8_0()])
                        .and_then(|mut rows| rows.pop())
                });
                let gate_up = metal_gate_up.unwrap_or_else(|| match (&paged, &packed) {
                    (Some((gate_up, _)), _) => gate_up.matvec_rows_proj(0, two_nff, &cur_moe_q),
                    (None, Some(p)) => packed_band_matvec(&p.gate_up, cur_moe_q.q8_0()),
                    (None, None) => {
                        moe.gate_up_exps
                            .matvec_rows_proj(e * two_nff, two_nff, &cur_moe_q)
                    }
                });
                let hexp: Vec<f32> = (0..moe.n_ff_exp)
                    .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                    .collect();
                let hexp_q = SharedActivation::new(&hexp);
                // down for expert e: rows e*n_embd .. +n_embd, in_dim=n_ff_exp.
                let metal_y = paged.as_ref().and_then(|(_, down)| {
                    ghost_metal_q4_matmul(down, hidden, &[hexp_q.q8_0()])
                        .and_then(|mut rows| rows.pop())
                });
                let y = metal_y.unwrap_or_else(|| match (&paged, &packed) {
                    (Some((_, down)), _) => down.matvec_rows_proj(0, hidden, &hexp_q),
                    (None, Some(p)) => packed_band_matvec(&p.down, hexp_q.q8_0()),
                    (None, None) => moe.down_exps.matvec_rows_proj(e * hidden, hidden, &hexp_q),
                });
                let scale = route_scales[route_slot];
                if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                    let gu_l2 = gate_up.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let hexp_l2 = hexp.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let y_l2 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
                    eprintln!("[cpu expert trace] slot={route_slot} e={e} prob={:.6} scale={:.6} gu_l2={gu_l2:.4} hexp_l2={hexp_l2:.4} y_l2={y_l2:.4}", probs[e], scale);
                }
                for (a, yv) in moe_acc.iter_mut().zip(&y) {
                    *a += yv * scale;
                }
            }
            moe_acc
        };
        let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);

        if std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
            let mlp_l2 = mlp.iter().map(|v| v * v).sum::<f32>().sqrt();
            let moe_l2 = moe_acc.iter().map(|v| v * v).sum::<f32>().sqrt();
            let cur_moe_l2 = cur_moe.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("[cpu layer {li}] Dn L2={mlp_l2:.6} Moe raw L2={moe_l2:.6} CurMoe L2={cur_moe_l2:.6} first4_moe={:?}", &moe_acc[..4]);
            if li == 0 {
                if let Some(dir) = std::env::var_os("CAMELID_GEMMA4_DUMP_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let ids = idx
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let ws = route_scales
                        .iter()
                        .map(|w| format!("{w:.8}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let meta =
                        format!("router_ids={ids}\nrouter_weights={ws}\nmoe_l2={moe_l2:.8}\n");
                    let _ = std::fs::write(dir.join("cpu_layer0_router.txt"), meta);
                    let mut raw = Vec::with_capacity(moe_acc.len() * 4);
                    for v in &moe_acc {
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                    let _ = std::fs::write(dir.join("cpu_layer0_moe.bin"), raw);
                }
            }
        }

        // combine the two branches, then the shared post_ffw_norm.
        let mut combined = mlp;
        for (c, m) in combined.iter_mut().zip(&cur_moe) {
            *c += m;
        }
        let ffn_out = rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
        if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
            let ffn_l2 = ffn_out.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!(
                "[cpu layer0] FFN out L2={ffn_l2:.6} first4={:?}",
                &ffn_out[..4]
            );
        }
        Ok(ffn_out)
    }

    /// Instrumented step forward that captures microsecond-level timing across all phases.
    pub fn step_range_profiled(
        &self,
        token: u32,
        pos: usize,
        h_in: Option<Vec<f32>>,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
    ) -> Result<(Gemma4StepOutput, TokenStepProfile)> {
        let hidden = self.config.embedding_length as usize;
        let heads = self.config.attention_head_count as usize;
        let ple_dim = self.g.per_layer_input_dim as usize;
        let eps = self.config.rms_norm_epsilon;
        let n_local = self.layers.len();
        let block_count = self.config.block_count as usize;
        let ple_total = block_count * ple_dim;
        let win = self.g.sliding_window as usize;
        let is_tail = self.first_layer + n_local == block_count;

        let t_total_start = std::time::Instant::now();
        let t_embed_start = std::time::Instant::now();

        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let mut h = match h_in {
            Some(h_in) => h_in,
            None => h0.clone(),
        };

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

        let embed_us = t_embed_start.elapsed().as_micros() as u64;
        let mut profile = TokenStepProfile {
            token,
            embed_us,
            ..Default::default()
        };

        for li in 0..n_local {
            let t_layer_start = std::time::Instant::now();
            let l = self.first_layer + li;
            let lw = &self.layers[li];
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;

            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            let t_attn_start = std::time::Instant::now();
            let xn = rms_norm(&h, Some(&lw.attn_norm), eps);
            let xnq = SharedActivation::new(&xn);
            let mut q = lw.attn_q.matvec_proj(q_dim, &xnq);
            for hh in 0..heads {
                let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&rms_norm(s, Some(&lw.q_norm), eps));
            }
            apply_rope(&mut q, heads, head_dim, pos, theta, rope_factors);

            if l < self.first_kv_shared {
                let mut k = lw
                    .attn_k
                    .as_ref()
                    .expect("owning layer binds attn_k")
                    .matvec_proj(kv_dim, &xnq);
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_proj(kv_dim, &xnq),
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&rms_norm(
                        s,
                        Some(lw.k_norm.as_deref().expect("k_norm")),
                        eps,
                    ));
                    // Weightless per-head V norm before caching (never RoPE).
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&rms_norm(sv, None, eps));
                }
                apply_rope(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                kc[li].push(k);
                vc[li].push(v);
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
            let lo = if sliding {
                (pos + 1).saturating_sub(win)
            } else {
                0
            };
            let mut attn = vec![0.0f32; heads * head_dim];
            // Attention scale 1.0 (per-head QK norms; matches the base
            // scalar path and the llama.cpp oracle).
            for h_idx in 0..heads {
                let kvh = h_idx / group;
                let qh = &q[h_idx * head_dim..(h_idx + 1) * head_dim];
                let out = &mut attn[h_idx * head_dim..(h_idx + 1) * head_dim];
                let mut scores = Vec::with_capacity(pos + 1 - lo);
                let mut max_s = f32::NEG_INFINITY;
                for p in lo..=pos {
                    let kp = &kc[src][p][kvh * head_dim..(kvh + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qh[d] * kp[d];
                    }
                    let s = dot;
                    if s > max_s {
                        max_s = s;
                    }
                    scores.push(s);
                }
                let mut den = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    den += *s;
                }
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
            if li == 0 && std::env::var("CAMELID_GEMMA4_DUMP_LAYERS").is_ok_and(|v| v == "1") {
                let o_l2 = o.iter().map(|v| v * v).sum::<f32>().sqrt();
                let on_l2 = on.iter().map(|v| v * v).sum::<f32>().sqrt();
                let h_l2 = h.iter().map(|v| v * v).sum::<f32>().sqrt();
                eprintln!("[cpu layer0] O L2={o_l2:.6} first4={:?}", &o[..4]);
                eprintln!(
                    "[cpu layer0] On (normed) L2={on_l2:.6} first4={:?}",
                    &on[..4]
                );
                eprintln!(
                    "[cpu layer0] SlabB (h_mid) L2={h_l2:.6} first4={:?}",
                    &h[..4]
                );
            }
            let attn_us = t_attn_start.elapsed().as_micros() as u64;

            let (ffn_out, mut l_prof) = if lw.moe.is_some() {
                self.moe_layer_ffn_profiled(li, &h)?
            } else {
                let t_dense = std::time::Instant::now();
                let ffn_dim = self.g.ffn_length_at(l) as usize;
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
                let out = rms_norm(&mlp, Some(&lw.post_ffw_norm), eps);
                (
                    out,
                    LayerStepProfile {
                        shared_mlp_us: t_dense.elapsed().as_micros() as u64,
                        ..Default::default()
                    },
                )
            };

            for (a, b) in h.iter_mut().zip(&ffn_out) {
                *a += b;
            }

            let t_ple_start = std::time::Instant::now();
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
            if lw.ple_output_scale != 1.0 {
                for v in h.iter_mut() {
                    *v *= lw.ple_output_scale;
                }
            }
            let ple_us = t_ple_start.elapsed().as_micros() as u64;

            l_prof.layer = l;
            l_prof.attn_us = attn_us;
            l_prof.ple_us = ple_us;
            l_prof.total_us = t_layer_start.elapsed().as_micros() as u64;

            profile.dense_attn_us += attn_us;
            profile.router_us += l_prof.router_us;
            profile.cache_and_io_us += l_prof.cache_and_io_us;
            profile.bytes_read += l_prof.bytes_read;
            profile.shared_mlp_us += l_prof.shared_mlp_us;
            profile.expert_gemv_us += l_prof.expert_gemv_us;
            profile.ple_us += ple_us;
            profile.layers.push(l_prof);
        }

        if !is_tail {
            profile.total_us = t_total_start.elapsed().as_micros() as u64;
            return Ok((Gemma4StepOutput::Hidden(h), profile));
        }

        let t_head = std::time::Instant::now();
        let logits = self.project_logits(&h);
        let head_us = t_head.elapsed().as_micros() as u64;
        profile.head_us = head_us;
        profile.total_us = t_total_start.elapsed().as_micros() as u64;

        Ok((Gemma4StepOutput::Logits(logits), profile))
    }

    /// Profiled MoE layer forward that captures router, cache/IO, shared MLP, and expert GEMV durations.
    fn moe_layer_ffn_profiled(
        &self,
        li: usize,
        attn_out: &[f32],
    ) -> Result<(Vec<f32>, LayerStepProfile)> {
        let hidden = self.config.embedding_length as usize;
        let eps = self.config.rms_norm_epsilon;
        let l = self.first_layer + li;
        let ffn_dim = self.g.ffn_length_at(l) as usize;
        let lw = &self.layers[li];
        let moe = lw.moe.as_ref().expect("moe layer");

        let mut prof = LayerStepProfile::default();

        let t_router = std::time::Instant::now();
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
        prof.selected_experts = idx.clone();
        prof.router_us = t_router.elapsed().as_micros() as u64;

        let mut wsum: f32 = idx.iter().map(|&e| probs[e]).sum();
        wsum = wsum.max(6.103_515e-5);

        let t_shared = std::time::Instant::now();
        let xn = rms_norm(attn_out, Some(&lw.ffn_norm), eps);
        let xnq = SharedActivation::new(&xn);
        let gate = lw.ffn_gate.matvec_proj(ffn_dim, &xnq);
        let up = lw.ffn_up.matvec_proj(ffn_dim, &xnq);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * u)
            .collect();
        let mlp_raw = lw.ffn_down.matvec(ffn_dim, hidden, &act);
        let mlp = rms_norm(&mlp_raw, Some(&moe.post_norm_1), eps);
        prof.shared_mlp_us = t_shared.elapsed().as_micros() as u64;

        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let cur_moe_q = SharedActivation::new(&cur_moe);
        let cur_moe_q8 = cur_moe_q.q8_0().to_vec();
        let route_scales: Vec<f32> = idx
            .iter()
            .map(|&e| moe.down_exps_scale[e] * (probs[e] / wsum))
            .collect();

        let t_io = std::time::Instant::now();
        let (paged_experts, metal_moe_acc) = match &moe.ghost {
            Some(ghost) if self.ghost_metal_q4_is_enabled() => {
                let bytes_before = ghost.cache.stats().bytes_read;
                let metal = self.try_ghost_metal_q4_experts(
                    ghost,
                    &idx,
                    &route_scales,
                    &cur_moe_q8,
                    hidden,
                );
                let bytes_after = ghost.cache.stats().bytes_read;
                prof.bytes_read = (bytes_after.saturating_sub(bytes_before)) as usize;
                match metal {
                    Some(acc) => (None, Some(acc)),
                    None => (Some(ghost.cache.get_many(ghost.layer_idx, &idx)?), None),
                }
            }
            Some(ghost) => {
                let bytes_before = ghost.cache.stats().bytes_read;
                let experts = ghost.cache.get_many(ghost.layer_idx, &idx)?;
                let bytes_after = ghost.cache.stats().bytes_read;
                prof.bytes_read = (bytes_after.saturating_sub(bytes_before)) as usize;
                (Some(experts), None)
            }
            None => (None, None),
        };
        prof.cache_and_io_us = t_io.elapsed().as_micros() as u64;

        let t_gemv = std::time::Instant::now();
        let two_nff = 2 * moe.n_ff_exp;
        let moe_acc = if let Some(metal_acc) = metal_moe_acc {
            metal_acc
        } else {
            let mut moe_acc = vec![0f32; hidden];
            for (route_slot, &e) in idx.iter().enumerate() {
                let paged = paged_experts
                    .as_ref()
                    .map(|experts| -> Result<(WireQuant, WireQuant)> {
                        let expert = &experts[route_slot];
                        Ok((
                            WireQuant::from_ghost_tensor(expert, &expert.gate_up, "ghost gate_up")?,
                            WireQuant::from_ghost_tensor(expert, &expert.down, "ghost down")?,
                        ))
                    })
                    .transpose()?;

                let gate_up = match &paged {
                    Some((gu, _)) => gu.matvec_rows_proj(0, two_nff, &cur_moe_q),
                    None => moe
                        .gate_up_exps
                        .matvec_rows_proj(e * two_nff, two_nff, &cur_moe_q),
                };
                let hexp: Vec<f32> = (0..moe.n_ff_exp)
                    .map(|o| gelu_tanh(gate_up[o]) * gate_up[o + moe.n_ff_exp])
                    .collect();
                let hexp_q = SharedActivation::new(&hexp);
                let y = match &paged {
                    Some((_, dn)) => dn.matvec_rows_proj(0, hidden, &hexp_q),
                    None => moe.down_exps.matvec_rows_proj(e * hidden, hidden, &hexp_q),
                };
                let scale = route_scales[route_slot];
                for (a, yv) in moe_acc.iter_mut().zip(&y) {
                    *a += yv * scale;
                }
            }
            moe_acc
        };
        prof.expert_gemv_us = t_gemv.elapsed().as_micros() as u64;

        let cur_moe = rms_norm(&moe_acc, Some(&moe.post_norm_2), eps);
        let mut combined = mlp;
        for (c, m) in combined.iter_mut().zip(&cur_moe) {
            *c += m;
        }
        let out = rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
        Ok((out, prof))
    }

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
        self.step_range_with_head(token, pos, h_in, kc, vc, true)
    }

    /// Internal scalar forward with an optional tied output head. Public shard
    /// callers retain the historical `step_range` behavior; prompt prefill can
    /// suppress the otherwise-unused 605 MB Q6_K projection on prefix tokens.
    fn step_range_with_head(
        &self,
        token: u32,
        pos: usize,
        h_in: Option<Vec<f32>>,
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        project_head: bool,
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
                    // V takes the usual weightless norm and never RoPE.
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
                self.moe_layer_ffn(li, &h)?
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
                if let Some(dir) = std::env::var_os("CAMELID_GEMMA4_DUMP_DIR") {
                    let path = std::path::PathBuf::from(dir).join("cpu_hidden.txt");
                    let line = format!(
                        "{l} {l2:.8} {:.8} {:.8} {:.8} {:.8}\n",
                        h[0], h[1], h[2], h[3]
                    );
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        let _ = f.write_all(line.as_bytes());
                    }
                }
            }
        }

        if !is_tail || !project_head {
            return Ok(Gemma4StepOutput::Hidden(h));
        }

        let t_out = std::time::Instant::now();
        // token_embd is vocab-major (row v = the v-th embedding). The helper
        // selects the no-copy Q6_K Metal head for Ghost-MoE on macOS and keeps
        // this exact CPU wire-dot implementation as its fallback.
        let logits = self.project_logits(&h);
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

    /// Prefill a freshly-created KV cache and return the logits at the final
    /// prompt position. Ghost-MoE uses bounded layer-major chunks so routes for
    /// several prompt tokens are known together and each unique expert record is
    /// read at most once per layer/chunk. Other runtimes retain the scalar path.
    pub fn prefill_tokens(
        &self,
        prompt_tokens: &[u32],
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        future_forwards: usize,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "Gemma 4 tokenizer produced an empty prompt".into(),
            ));
        }
        let plan = self.prepare_ghost_prefill(prompt_tokens.len(), future_forwards)?;
        if lane_trace_enabled() {
            eprintln!(
                "[lane] prefill plan={plan:?} prompt_len={}",
                prompt_tokens.len()
            );
        }
        if matches!(
            plan,
            GhostPrefillPlan::CpuChunk | GhostPrefillPlan::HybridChunk
        ) {
            // Bound transient routed-expert/head output memory. Sixteen covers
            // the complete short chat template in the common case; longer
            // prompts retain the same win independently in each chunk.
            // On the chained Metal lane a chunk is one chained round; one
            // token per round costs 30 host waits plus a full tied-head sweep
            // per prompt token. Eight is the K the chained lane is exercised
            // at (verifier harness); the CPU chunk lane keeps sixteen.
            let chained_lane = self.ghost_metal_q4_is_enabled();
            let chunk_size = std::env::var("CAMELID_GEMMA4_GHOST_PREFILL_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(if chained_lane { 8 } else { 16 })
                .min(if chained_lane {
                    crate::metal::GEMMA4_RESIDENT_MAX_BATCH
                } else {
                    64
                });
            let mut logits = Vec::new();
            for (chunk_idx, tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                let start_pos = chunk_idx * chunk_size;
                let mut rows = self.step_chunk_with_head(tokens, start_pos, kc, vc, false, None)?;
                logits = rows.pop().expect("non-empty prefill chunk has logits");
            }
            if plan == GhostPrefillPlan::HybridChunk {
                let _ = self.finish_ghost_hybrid_prefill(kc, vc, prompt_tokens.len())?;
            }
            #[cfg(target_os = "macos")]
            if let Some(cache) = self.ghost_moe_cache.as_ref() {
                if let Ok(mut guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_mut() {
                        lane.prefetch_temporal_last_round(&cache.file, &cache.read_pool);
                    }
                }
            }
            Ok(logits)
        } else {
            let logits = drive_scalar_prefill(prompt_tokens, |token, pos, project_head| {
                if project_head {
                    self.step(token, pos, kc, vc).map(Some)
                } else {
                    self.step_without_head(token, pos, kc, vc).map(|()| None)
                }
            })?;
            #[cfg(target_os = "macos")]
            if let Some(cache) = self.ghost_moe_cache.as_ref() {
                if let Ok(mut guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_mut() {
                        lane.prefetch_temporal_last_round(&cache.file, &cache.read_pool);
                    }
                }
            }
            Ok(logits)
        }
    }

    /// Cancellation-aware form of [`Self::prefill_tokens`] used by serving.
    /// A forward already submitted to CPU/Metal is allowed to finish, then the
    /// next chunk/token boundary observes the stop signal before touching any
    /// more model or Ghost-MoE state.
    fn prefill_tokens_cancellable<C: FnMut() -> bool>(
        &self,
        prompt_tokens: &[u32],
        kc: &mut [Vec<Vec<f32>>],
        vc: &mut [Vec<Vec<f32>>],
        future_forwards: usize,
        should_cancel: &mut C,
    ) -> Result<Option<Vec<f32>>> {
        if prompt_tokens.is_empty() {
            return Err(BackendError::InvalidModelMetadata(
                "Gemma 4 tokenizer produced an empty prompt".into(),
            ));
        }
        if should_cancel() {
            return Ok(None);
        }
        let plan = self.prepare_ghost_prefill(prompt_tokens.len(), future_forwards)?;
        if matches!(
            plan,
            GhostPrefillPlan::CpuChunk | GhostPrefillPlan::HybridChunk
        ) {
            // On the chained Metal lane a chunk is one chained round; one
            // token per round costs 30 host waits plus a full tied-head sweep
            // per prompt token. Eight is the K the chained lane is exercised
            // at (verifier harness); the CPU chunk lane keeps sixteen.
            let chained_lane = self.ghost_metal_q4_is_enabled();
            let chunk_size = std::env::var("CAMELID_GEMMA4_GHOST_PREFILL_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|&value| value > 0)
                .unwrap_or(if chained_lane { 8 } else { 16 })
                .min(if chained_lane {
                    crate::metal::GEMMA4_RESIDENT_MAX_BATCH
                } else {
                    64
                });
            let mut logits = Vec::new();
            for (chunk_idx, tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                if should_cancel() {
                    return Ok(None);
                }
                let start_pos = chunk_idx * chunk_size;
                let mut rows = self.step_chunk_with_head(tokens, start_pos, kc, vc, false, None)?;
                logits = rows.pop().expect("non-empty prefill chunk has logits");
            }
            if should_cancel() {
                return Ok(None);
            }
            if plan == GhostPrefillPlan::HybridChunk {
                let _ = self.finish_ghost_hybrid_prefill(kc, vc, prompt_tokens.len())?;
                // Import is synchronous but can copy a large context. Observe a
                // disconnect that arrived during it before starting decode; the
                // request cleanup guard resets the just-seeded Metal sequence.
                if should_cancel() {
                    return Ok(None);
                }
            }
            Ok(Some(logits))
        } else {
            let (&last_token, prefix) = prompt_tokens
                .split_last()
                .expect("prefill validates that the prompt is non-empty");
            for (pos, &token) in prefix.iter().enumerate() {
                if should_cancel() {
                    return Ok(None);
                }
                self.step_without_head(token, pos, kc, vc)?;
            }
            if should_cancel() {
                return Ok(None);
            }
            let logits = self.step(last_token, prefix.len(), kc, vc)?;
            if should_cancel() {
                Ok(None)
            } else {
                Ok(Some(logits))
            }
        }
    }

    /// Greedily generate up to `max_new` tokens from `prompt`, with an incremental
    /// KV cache (one forward step per token). Returns (decoded continuation, the
    /// generated token ids).
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index, not a count
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits =
            self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))?;
        // Lossless n-gram speculative decode (opt-in, single-node non-MoE rows): verify
        // a batch of drafted tokens in ONE weight pass via `step_chunk`. Output is
        // token-for-token identical to the greedy loop below — every committed token is
        // the target's own argmax — so it makes no support/parity claim, only speed.
        if std::env::var("CAMELID_GEMMA4_SPEC_DECODE").is_ok()
            && self.supports_speculative_chunk_forward()
        {
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
        for generated_index in 0..max_new {
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
            // The final allowed token is already known from `logits`. Feeding it
            // back through the whole model would only compute the prediction for
            // a token the caller did not request. That extra step is especially
            // expensive for Ghost-MoE (30 layers x 8 paged experts), so stop at
            // the exact generation boundary without changing any returned id.
            if generated_index + 1 < max_new {
                logits = if self.supports_speculative_chunk_forward() {
                    self.step_chunk(&[next], pos, &mut kc, &mut vc)?
                        .into_iter()
                        .next()
                        .expect("step_chunk row")
                } else {
                    self.step(next, pos, &mut kc, &mut vc)?
                };
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Serve-safe greedy generation.  The caller owns serialization; this
    /// method supplies the other half of that contract by relinquishing model
    /// state at the next prompt/decode forward boundary after cancellation.
    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_controlled(prompt, max_new, None::<fn(&str)>, should_cancel)
    }

    /// Streaming counterpart to [`Self::generate_greedy_cancellable`].
    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        on_delta: F,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_controlled(prompt, max_new, Some(on_delta), should_cancel)
    }

    fn generate_greedy_controlled<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: Option<F>,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let Some(mut logits) = self.prefill_tokens_cancellable(
            &prompt_tokens,
            &mut kc,
            &mut vc,
            max_new.saturating_sub(1),
            &mut should_cancel,
        )?
        else {
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        };
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();

        let use_speculative = self.supports_speculative_chunk_forward()
            && std::env::var("CAMELID_SPEC_DECODE")
                .map(|v| {
                    !matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "off" | "0" | "false" | "none"
                    )
                })
                .unwrap_or(true);

        if use_speculative {
            use crate::inference::speculative::{
                accepted_draft_prefix, NGramDrafter, DEFAULT_NGRAM_DRAFT_TOKENS,
            };
            let max_draft = std::env::var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS")
                .or_else(|_| std::env::var("CAMELID_SPEC_DRAFT_TOKENS"))
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
            let mut history = prompt_tokens.to_vec();
            let mut round_idx = 0usize;
            let mut round_records: Vec<crate::inference::speculative::SpeculativeRoundRecord> =
                Vec::new();

            while generated.len() < max_new {
                if should_cancel() {
                    return Ok(Gemma4GenerationOutcome::Cancelled {
                        generated_tokens: generated.len(),
                    });
                }
                round_idx += 1;
                let t_round_start = std::time::Instant::now();
                let t0 = argmax(&logits);
                if eot.contains(&t0) {
                    break;
                }
                generated.push(t0);
                if let Some(on_delta_fn) = on_delta.as_mut() {
                    let full = self.tokenizer.decode(&generated, true)?;
                    if let Some(delta) = full.strip_prefix(emitted.as_str()) {
                        if !delta.is_empty() {
                            on_delta_fn(delta);
                        }
                    }
                    emitted = full;
                }
                history.push(t0);
                if generated.len() >= max_new {
                    round_records.push(crate::inference::speculative::SpeculativeRoundRecord {
                        round: round_idx,
                        requested_k: 1,
                        draft_tokens_proposed: 0,
                        actual_verifier_batch_size: 1,
                        fallback_k1: false,
                        accepted_draft_prefix: 0,
                        tokens_committed: 1,
                        round_wall_ms: t_round_start.elapsed().as_secs_f64() * 1000.0,
                        gpu_ms: 0.0,
                        cpu_ms: 0.0,
                        io_wait_ms: 0.0,
                    });
                    break;
                }
                let budget = max_new - generated.len();
                let drafts = drafter.draft(&history, max_draft.min(budget));
                if drafts.is_empty() {
                    if should_cancel() {
                        return Ok(Gemma4GenerationOutcome::Cancelled {
                            generated_tokens: generated.len(),
                        });
                    }
                    let rows = self.step_chunk(&[t0], pos, &mut kc, &mut vc)?;
                    logits = rows.into_iter().next().expect("step_chunk row");
                    pos += 1;
                    round_records.push(crate::inference::speculative::SpeculativeRoundRecord {
                        round: round_idx,
                        requested_k: 1 + max_draft.min(budget),
                        draft_tokens_proposed: 0,
                        actual_verifier_batch_size: 1,
                        fallback_k1: true,
                        accepted_draft_prefix: 0,
                        tokens_committed: 1,
                        round_wall_ms: t_round_start.elapsed().as_secs_f64() * 1000.0,
                        gpu_ms: 0.0,
                        cpu_ms: 0.0,
                        io_wait_ms: 0.0,
                    });
                    continue;
                }
                let mut chunk = Vec::with_capacity(1 + drafts.len());
                chunk.push(t0);
                chunk.extend_from_slice(&drafts);
                if lane_trace_enabled() {
                    eprintln!("[lane] spec verify chunk len={} pos={pos}", chunk.len());
                }
                let rows = self.step_chunk(&chunk, pos, &mut kc, &mut vc)?;
                let preds: Vec<u32> = (0..drafts.len()).map(|i| argmax(&rows[i])).collect();
                let j = accepted_draft_prefix(&drafts, &preds);
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
                    if let Some(on_delta_fn) = on_delta.as_mut() {
                        let full = self.tokenizer.decode(&generated, true)?;
                        if let Some(delta) = full.strip_prefix(emitted.as_str()) {
                            if !delta.is_empty() {
                                on_delta_fn(delta);
                            }
                        }
                        emitted = full;
                    }
                    history.push(d);
                }
                let keep = pos + j + 1;
                for li in 0..kc.len() {
                    kc[li].truncate(keep);
                    vc[li].truncate(keep);
                }
                #[cfg(target_os = "macos")]
                if let Ok(mut guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_mut() {
                        lane.truncate_sequence(keep);
                    }
                }
                pos = keep;
                logits = rows.into_iter().nth(j).expect("rows[j] exists");
                round_records.push(crate::inference::speculative::SpeculativeRoundRecord {
                    round: round_idx,
                    requested_k: 1 + drafts.len(),
                    draft_tokens_proposed: drafts.len(),
                    actual_verifier_batch_size: 1 + drafts.len(),
                    fallback_k1: false,
                    accepted_draft_prefix: j,
                    tokens_committed: 1 + j,
                    round_wall_ms: t_round_start.elapsed().as_secs_f64() * 1000.0,
                    gpu_ms: 0.0,
                    cpu_ms: 0.0,
                    io_wait_ms: 0.0,
                });
                if stopped {
                    break;
                }
            }
            if std::env::var("CAMELID_SPEC_ACCOUNTING").is_ok() {
                crate::inference::speculative::report_speculative_accounting(&round_records);
            }
            if cpu_timing_enabled() {
                report_cpu_timing();
            }
            let text = if on_delta.is_some() {
                emitted
            } else {
                self.tokenizer.decode(&generated, true)?
            };
            return Ok(Gemma4GenerationOutcome::Complete {
                text,
                token_ids: generated,
            });
        }

        for generated_index in 0..max_new {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
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
            if let Some(on_delta) = on_delta.as_mut() {
                let full = self.tokenizer.decode(&generated, true)?;
                if let Some(delta) = full.strip_prefix(&emitted) {
                    if !delta.is_empty() {
                        on_delta(delta);
                    }
                }
                emitted = full;
            }
            // A dropped SSE receiver is discovered by `on_delta`; check again
            // before starting the forward for the next token.
            if generated_index + 1 < max_new {
                if should_cancel() {
                    return Ok(Gemma4GenerationOutcome::Cancelled {
                        generated_tokens: generated.len(),
                    });
                }
                logits = if self.supports_speculative_chunk_forward() {
                    self.step_chunk(&[next], pos, &mut kc, &mut vc)?
                        .into_iter()
                        .next()
                        .expect("step_chunk row")
                } else {
                    self.step(next, pos, &mut kc, &mut vc)?
                };
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
        }
        let text = if on_delta.is_some() {
            emitted
        } else {
            self.tokenizer.decode(&generated, true)?
        };
        Ok(Gemma4GenerationOutcome::Complete {
            text,
            token_ids: generated,
        })
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
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let logits = self.prefill_tokens(
            &prompt_tokens,
            &mut kc,
            &mut vc,
            forced.len().saturating_sub(1),
        )?;
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
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut logits = self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new)?;
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
        if !self.supports_speculative_chunk_forward() {
            return self.generate_greedy(prompt, max_new);
        }
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let logits = self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, 0)?;
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
            if spec_timing {
                eprintln!(
                    "[spec-debug] round pos={pos} chunk_len={} kc0_len={} drafts={:?}",
                    chunk.len(),
                    kc.first().map(|k| k.len()).unwrap_or(0),
                    drafts
                );
            }

            #[cfg(target_os = "macos")]
            if let Some(cache) = self.ghost_moe_cache.as_ref() {
                if let Ok(predicted_routes) = self.predict_all_layer_routes_for_chunk(&chunk) {
                    if let Ok(mut guard) = self.metal_q4_experts.lock() {
                        if let Some(lane) = guard.as_mut() {
                            lane.prefetch_round_wide_async(
                                &predicted_routes,
                                &cache.file,
                                &cache.read_pool,
                            );
                        }
                    }
                } else if let Ok(mut guard) = self.metal_q4_experts.lock() {
                    if let Some(lane) = guard.as_mut() {
                        lane.prefetch_temporal_last_round(&cache.file, &cache.read_pool);
                    }
                }
            }

            let rows = self.step_chunk(&chunk, pos, kc, vc)?;
            let preds: Vec<u32> = (0..drafts.len()).map(|i| argmax(&rows[i])).collect();
            let j = accepted_draft_prefix(&drafts, &preds);
            accepted_rounds += 1;
            accepted_drafts += j as u64;
            crate::metal::SPEC_VERIFY_ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::metal::SPEC_ACCEPTED_TOKENS
                .fetch_add(1 + j, std::sync::atomic::Ordering::Relaxed);
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
            #[cfg(target_os = "macos")]
            if let Ok(mut guard) = self.metal_q4_experts.lock() {
                if let Some(lane) = guard.as_mut() {
                    lane.truncate_sequence(keep);
                }
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
        #[cfg(target_os = "macos")]
        let _ghost_common_request = self.lock_ghost_common_generation()?;
        #[cfg(target_os = "macos")]
        let _ghost_metal_stats = GhostMetalGenerationStatsGuard::new(&self.metal_q4_experts);
        #[cfg(target_os = "macos")]
        let _ghost_sequence_cleanup = GhostMetalSequenceCleanup::new(&self.metal_q4_experts);
        let n_layers = self.layers.len();
        let mut kc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let mut vc: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layers];
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        if std::env::var("CAMELID_GEMMA4_DUMP_PROMPT_TOKENS").is_ok() {
            eprintln!("[prompt tokens] {prompt_tokens:?}");
        }
        let eot = gemma4_stop_token_ids(&self.tokenizer);

        let mut logits =
            self.prefill_tokens(&prompt_tokens, &mut kc, &mut vc, max_new.saturating_sub(1))?;
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut pos = prompt_tokens.len();
        for generated_index in 0..max_new {
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
            // Do not run a discarded full forward after the last requested
            // token. The sampled token/delta is already complete at this point.
            if generated_index + 1 < max_new {
                logits = self.step(next, pos, &mut kc, &mut vc)?;
                pos += 1;
            }
        }
        if cpu_timing_enabled() {
            report_cpu_timing();
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
    rope_inv_freqs: Vec<Vec<f32>>,
    _mmap: Arc<GgufWireMmap>,
    hidden: usize,
    ple_dim: usize,
    n_layers: usize,
    /// QAT hybrid lane: the tied head is Q6_K. If `q6k_gpu_head` is present, it executes
    /// on the Metal GPU in ~2.5ms; otherwise the CPU runs the fallback matvec.
    q6k_gpu_head: Option<crate::metal::Gemma4Q6KHead>,
    head_on_cpu: bool,
    /// Held for the CPU head (`head_on_cpu`): output RMS-norm weights + vocab.
    output_norm: Vec<f32>,
    vocab: usize,
    eps: f32,
    spec_h0_buf: std::cell::RefCell<Vec<f32>>,
    spec_ti_buf: std::cell::RefCell<Vec<f32>>,
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
        if config.moe.is_some() {
            return Err(BackendError::UnsupportedModelArchitecture(
                "Gemma 4 MoE models (e.g. 26B/A4B) are not supported by dense GPU resident runtime; use Ghost-MoE runtime".into(),
            ));
        }
        let g = config.gemma4.clone().ok_or_else(|| {
            BackendError::UnsupportedModelArchitecture("not a gemma4 model".into())
        })?;
        let binding = Gemma4Binding::bind(&gguf, &config)?;
        let store = TensorStore::open(path, &gguf);
        // The GPU-resident decode kernels run the layer projections as Q8_0 (34-byte
        // wire blocks), Q4_0 (18-byte QAT wire blocks), or NVFP4 (36-byte 64-value
        // superblocks; GABBRO M3) — all parity-gated GPU GEMVs. The tied head is read
        // separately: Q8_0 runs on the GPU (inside forward_token); Q6_K (the QAT tied
        // head) runs on the GPU via Gemma4Q6KHead or fallback CPU. Layer 0's attn_q
        // is representative of the projection format (the export quantizes every
        // layer's projections alike).
        let layer_fmt = gemma4_metal_layer_fmt(
            store
                .descriptor(&binding.layers[0].attn_q.name)?
                .tensor_type,
        )?;
        let head_on_cpu = match store.descriptor(&binding.token_embedding.name)?.tensor_type {
            GgufTensorType::Q8_0 => false, // GPU Q8 head
            GgufTensorType::Q6K => true,   // CPU/GPU Q6_K head (QAT tied head)
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

        let rope_factors = binding
            .rope_freqs
            .as_ref()
            .map(|d| f32t(&d.name))
            .transpose()?;
        let rope_inv_freqs: Vec<Vec<f32>> = (0..n_layers)
            .map(|l| {
                let hd = g.head_dim_at(l) as usize;
                let theta = g.rope_freq_base_at(l);
                let half = hd / 2;
                let factors = if g.is_sliding_layer(l) {
                    None
                } else {
                    rope_factors.as_deref()
                };
                (0..half)
                    .map(|i| {
                        let mut freq = theta.powf(-(2.0 * i as f32) / hd as f32);
                        if let Some(factors) = factors {
                            freq /= factors[i];
                        }
                        freq
                    })
                    .collect()
            })
            .collect();

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
            let layer = crate::metal::Gemma4ResidentLayer::from_wire_pages_with_rope(
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
                Some(&rope_inv_freqs[l]),
                max_positions,
                if g.is_sliding_layer(l) {
                    Some(g.sliding_window as usize)
                } else {
                    None
                },
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
        let q6k_gpu_head = if head_on_cpu {
            let desc = store.descriptor(&binding.token_embedding.name)?;
            crate::metal::Gemma4Q6KHead::new(
                mmap.clone(),
                desc.absolute_offset,
                desc.n_bytes as usize,
                &output_norm,
                vocab,
                softcap,
                eps,
            )
        } else {
            None
        };
        // QAT hybrid (Q6_K head on CPU/Gpu): don't hand the tied table to the GPU head — pass
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

        let ple_total = n_layers * ple_dim;
        Ok(Self {
            model,
            tokenizer,
            per_layer_token_embd: binding
                .per_layer_token_embd
                .as_ref()
                .map(|d| q8(&d.name))
                .transpose()?,
            rope_factors,
            rope_inv_freqs,
            token_embd,
            g,
            _mmap: mmap,
            hidden,
            ple_dim,
            n_layers,
            q6k_gpu_head,
            head_on_cpu,
            output_norm,
            vocab,
            eps,
            spec_h0_buf: std::cell::RefCell::new(Vec::with_capacity(32 * hidden)),
            spec_ti_buf: std::cell::RefCell::new(Vec::with_capacity(32 * ple_total)),
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
        let ti: Vec<f32> = if let Some(te) = self.per_layer_token_embd.as_ref() {
            let scale = (ple_dim as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            te.dequantize_elements(token as usize * ple_total, ple_total)?
                .iter()
                .map(|v| v * scale)
                .collect()
        } else {
            Vec::new()
        };
        let win = self.g.sliding_window as usize;
        let inputs: Vec<crate::metal::Gemma4TokenLayerInput> = (0..self.n_layers)
            .map(|l| {
                let half = self.rope_inv_freqs[l].len();
                let (mut cos_t, mut sin_t) = (vec![0f32; half], vec![0f32; half]);
                let pos_f = position as f32;
                for i in 0..half {
                    let (s, c) = (pos_f * self.rope_inv_freqs[l][i]).sin_cos();
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
                    pli: Vec::new(),
                    window_start,
                }
            })
            .collect();
        let prep_us = t_prep.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        // GPU Q6_K tied head executes directly on Metal in ~2.5ms; falls back to CPU if unavailable.
        let logits = if let Some(gpu_head) = &self.q6k_gpu_head {
            let last_hidden = self
                .model
                .forward_token_hidden(&h0, &inputs, &ti, position)
                .ok_or_else(|| {
                    BackendError::UnsupportedModelArchitecture("gpu forward failed".into())
                })?;
            gpu_head.forward(&last_hidden).unwrap_or_else(|| {
                let last = rms_norm(&last_hidden, Some(&self.output_norm), self.eps);
                let mut logits = self.token_embd.matvec(self.hidden, self.vocab, &last);
                if let Some(cap) = self.g.final_logit_softcapping {
                    soft_cap_in_place(&mut logits, cap);
                }
                logits
            })
        } else if self.head_on_cpu {
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

    /// Run one token's forward on the GPU and return the greedy argmax next-token ID
    /// directly with zero intermediate logits allocation and GPU-side reduction.
    fn forward_argmax(&self, token: u32, position: usize) -> Result<u32> {
        let t_prep = std::time::Instant::now();
        let hidden = self.hidden;
        let ple_dim = self.ple_dim;
        let ple_total = self.n_layers * ple_dim;
        let filled = position + 1;
        let h0: Vec<f32> = self
            .token_embd
            .dequantize_elements(token as usize * hidden, hidden)?
            .iter()
            .map(|v| v * (hidden as f32).sqrt())
            .collect();
        let ti: Vec<f32> = if let Some(te) = self.per_layer_token_embd.as_ref() {
            let scale = (ple_dim as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            te.dequantize_elements(token as usize * ple_total, ple_total)?
                .iter()
                .map(|v| v * scale)
                .collect()
        } else {
            Vec::new()
        };
        let win = self.g.sliding_window as usize;
        let inputs: Vec<crate::metal::Gemma4TokenLayerInput> = (0..self.n_layers)
            .map(|l| {
                let half = self.rope_inv_freqs[l].len();
                let (mut cos_t, mut sin_t) = (vec![0f32; half], vec![0f32; half]);
                let pos_f = position as f32;
                for i in 0..half {
                    let (s, c) = (pos_f * self.rope_inv_freqs[l][i]).sin_cos();
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
                    pli: Vec::new(),
                    window_start,
                }
            })
            .collect();
        let prep_us = t_prep.elapsed().as_micros();
        let t_gpu = std::time::Instant::now();
        let next_tok = if let Some(gpu_head) = &self.q6k_gpu_head {
            self.model
                .forward_token_fused_argmax(&h0, &inputs, &ti, position, gpu_head)
                .unwrap_or_else(|| {
                    let last_hidden = self
                        .model
                        .forward_token_hidden(&h0, &inputs, &ti, position)
                        .unwrap_or_default();
                    let last = rms_norm(&last_hidden, Some(&self.output_norm), self.eps);
                    let mut logits = self.token_embd.matvec(self.hidden, self.vocab, &last);
                    if let Some(cap) = self.g.final_logit_softcapping {
                        soft_cap_in_place(&mut logits, cap);
                    }
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
        } else {
            let logits = self.forward(token, position)?;
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        if std::env::var("CAMELID_GEMMA4_GPU_PROFILE").is_ok() && position == 0 {
            if let Some(gpu_head) = &self.q6k_gpu_head {
                self.model
                    .profile_subkernels(&h0, &inputs, &ti, position, gpu_head);
            }
        }
        let gpu_dur = t_gpu.elapsed().as_micros() as u64;
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok()
            || std::env::var("CAMELID_GEMMA4_GPU_PROFILE").is_ok()
        {
            PREP_US.fetch_add(prep_us as u64, std::sync::atomic::Ordering::Relaxed);
            GPU_US.fetch_add(gpu_dur, std::sync::atomic::Ordering::Relaxed);
            FWD_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Ok(mut lats) = STEP_LATENCIES_US.lock() {
                lats.push(gpu_dur);
            }
        }
        Ok(next_tok)
    }

    /// Verify a batch of K candidate tokens [base_position, base_position + K) on Metal GPU.
    fn verify_batch_argmax(
        &self,
        tokens: &[u32],
        base_position: usize,
    ) -> Result<(Vec<u32>, crate::metal::Gemma4VerifyTimings)> {
        let k_tokens = tokens.len();
        if k_tokens == 0 {
            return Ok((Vec::new(), crate::metal::Gemma4VerifyTimings::default()));
        }
        let Some(gpu_head) = &self.q6k_gpu_head else {
            let mut preds = Vec::with_capacity(k_tokens);
            for (i, &tok) in tokens.iter().enumerate() {
                preds.push(self.forward_argmax(tok, base_position + i)?);
            }
            return Ok((preds, crate::metal::Gemma4VerifyTimings::default()));
        };

        let hidden = self.hidden;
        let ple_dim = self.ple_dim;
        let ple_total = self.n_layers * ple_dim;

        let total_h0_len = k_tokens * hidden;
        let mut h0_borrow = self.spec_h0_buf.borrow_mut();
        if h0_borrow.len() < total_h0_len {
            h0_borrow.resize(total_h0_len, 0.0);
        }
        let h0_slice = &mut h0_borrow[..total_h0_len];

        let h0_scale = (hidden as f32).sqrt();
        for (i, &token) in tokens.iter().enumerate() {
            let out_chunk = &mut h0_slice[i * hidden..(i + 1) * hidden];
            self.token_embd
                .dequantize_elements_into(token as usize * hidden, hidden, out_chunk)?;
            for v in out_chunk.iter_mut() {
                *v *= h0_scale;
            }
        }

        let mut ti_borrow = self.spec_ti_buf.borrow_mut();
        let total_ti_len = k_tokens * ple_total;
        let has_ple = self.per_layer_token_embd.is_some();
        if has_ple && ti_borrow.len() < total_ti_len {
            ti_borrow.resize(total_ti_len, 0.0);
        }
        let ti_slice = if let Some(te) = self.per_layer_token_embd.as_ref() {
            let ti_slice = &mut ti_borrow[..total_ti_len];
            let scale = (ple_dim as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            for (i, &token) in tokens.iter().enumerate() {
                let out_chunk = &mut ti_slice[i * ple_total..(i + 1) * ple_total];
                te.dequantize_elements_into(token as usize * ple_total, ple_total, out_chunk)?;
                for v in out_chunk.iter_mut() {
                    *v *= scale;
                }
            }
            ti_slice
        } else {
            &mut ti_borrow[..0]
        };
        static PROFILED_VERIFY: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("CAMELID_GEMMA4_GPU_PROFILE").is_ok()
            && !PROFILED_VERIFY.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            self.model.profile_verify_batch_subkernels(
                h0_slice,
                ti_slice,
                base_position,
                k_tokens,
                gpu_head,
            );
        }

        self.model
            .verify_batch_fused_argmax_slab(h0_slice, ti_slice, base_position, k_tokens, gpu_head)
            .ok_or_else(|| {
                BackendError::UnsupportedModelArchitecture(
                    "verify_batch_fused_argmax_slab failed".into(),
                )
                .into()
            })
    }

    /// Speculative greedy generation on GPU with lossless target verification.
    pub fn generate_greedy_speculative_gpu(
        &self,
        prompt: &str,
        max_new: usize,
        max_draft: usize,
    ) -> Result<(String, Vec<u32>)> {
        use crate::inference::speculative::{accepted_draft_prefix, NGramDrafter};

        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut next_tok = 0u32;
        let t_prefill_start = std::time::Instant::now();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            next_tok = self.forward_argmax(tok, pos)?;
        }
        let prefill_duration = t_prefill_start.elapsed();

        let drafter = NGramDrafter::new(1, 6);
        let mut history = prompt_tokens.clone();
        let mut generated: Vec<u32> = Vec::new();
        let mut pos = prompt_tokens.len();

        let mut spec_rounds = 0u64;
        let mut total_drafts = 0u64;
        let mut total_accepted = 0u64;
        let mut total_emitted_spec = 0u64;
        let mut total_round_us = 0u128;
        let mut total_draft_us = 0u128;
        let mut total_verify_wall_us = 0u128;
        let mut total_cpu_encode_us = 0u128;
        let mut total_gpu_wait_us = 0u128;
        let mut total_gpu_hw_us = 0u128;
        let mut total_readback_us = 0u128;
        let mut total_bookkeeping_us = 0u128;

        let t_gen_start = std::time::Instant::now();
        while generated.len() < max_new {
            if eot.contains(&next_tok) {
                break;
            }
            generated.push(next_tok);
            history.push(next_tok);
            if generated.len() >= max_new {
                break;
            }

            // Start of full speculative round
            let t_round_start = std::time::Instant::now();

            let budget = (max_new - generated.len()).min(max_draft);
            let t_draft = std::time::Instant::now();
            let drafts = drafter.draft(&history, budget);
            let draft_us = t_draft.elapsed().as_micros();
            total_draft_us += draft_us;

            if drafts.is_empty() {
                // Standalone 1-token step
                next_tok = self.forward_argmax(next_tok, pos)?;
                pos += 1;
                let round_us = t_round_start.elapsed().as_micros();
                total_round_us += round_us;
                continue;
            }

            // Build chunk [t0, d1, d2, ...]
            let mut chunk = Vec::with_capacity(1 + drafts.len());
            chunk.push(next_tok);
            chunk.extend_from_slice(&drafts);

            let t_verify = std::time::Instant::now();
            let (preds, verify_timings) = self.verify_batch_argmax(&chunk, pos)?;
            let verify_wall_us = t_verify.elapsed().as_micros();
            total_verify_wall_us += verify_wall_us;
            total_cpu_encode_us += verify_timings.cpu_encode_us;
            total_gpu_wait_us += verify_timings.gpu_wait_us;
            total_gpu_hw_us += verify_timings.gpu_hw_us;
            total_readback_us += verify_timings.readback_us;

            // Check how many drafts match the target's predictions
            let t_bookkeeping = std::time::Instant::now();
            let accepted = accepted_draft_prefix(&drafts, &preds[..drafts.len()]);
            spec_rounds += 1;
            total_drafts += drafts.len() as u64;
            total_accepted += accepted as u64;
            let emitted_this_round = 1 + accepted as u64;
            total_emitted_spec += emitted_this_round;

            let mut stopped = false;
            for &d in &drafts[..accepted] {
                if generated.len() >= max_new {
                    stopped = true;
                    break;
                }
                generated.push(d);
                history.push(d);
                if eot.contains(&d) {
                    stopped = true;
                    break;
                }
            }

            pos += 1 + accepted;
            next_tok = preds[accepted];
            let bookkeeping_us = t_bookkeeping.elapsed().as_micros();
            total_bookkeeping_us += bookkeeping_us;

            let round_us = t_round_start.elapsed().as_micros();
            total_round_us += round_us;

            if stopped {
                break;
            }
        }

        let total_gen_duration = t_gen_start.elapsed();
        let gen_toks = generated.len().max(1);
        let actual_tok_s = gen_toks as f64 / total_gen_duration.as_secs_f64();
        let e2e_per_tok_ms = total_gen_duration.as_secs_f64() * 1000.0 / gen_toks as f64;

        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok()
            || std::env::var("CAMELID_GEMMA4_GPU_PROFILE").is_ok()
            || std::env::var("CAMELID_GEMMA4_SPEC_TIMING").is_ok()
        {
            let accept_pct = if total_drafts > 0 {
                100.0 * (total_accepted as f64) / (total_drafts as f64)
            } else {
                0.0
            };
            let mean_accepted_per_round = if spec_rounds > 0 {
                (total_accepted as f64) / (spec_rounds as f64)
            } else {
                0.0
            };
            let mean_emitted_per_round = if spec_rounds > 0 {
                (total_emitted_spec as f64) / (spec_rounds as f64)
            } else {
                0.0
            };
            let true_wall_ms_per_round = if spec_rounds > 0 {
                (total_round_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let pure_gpu_hw_ms_per_round = if spec_rounds > 0 {
                (total_gpu_hw_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let cpu_encode_ms_per_round = if spec_rounds > 0 {
                (total_cpu_encode_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let gpu_wait_ms_per_round = if spec_rounds > 0 {
                (total_gpu_wait_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let readback_ms_per_round = if spec_rounds > 0 {
                (total_readback_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let bookkeeping_ms_per_round = if spec_rounds > 0 {
                (total_bookkeeping_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let draft_ms_per_round = if spec_rounds > 0 {
                (total_draft_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let verify_wall_ms_per_round = if spec_rounds > 0 {
                (total_verify_wall_us as f64 / spec_rounds as f64) / 1000.0
            } else {
                0.0
            };
            let round_rate_tok_s = if true_wall_ms_per_round > 0.0 {
                mean_emitted_per_round / (true_wall_ms_per_round / 1000.0)
            } else {
                0.0
            };

            eprintln!("\n==========================================================================================");
            eprintln!("                  GEMMA 4 26B GPU SPECULATIVE VERIFICATION TELEMETRY                      ");
            eprintln!("==========================================================================================");
            eprintln!(
                "A. Sustained Generation Speed:           {:6.2} tok/s ({:6.2} ms/accepted token)",
                actual_tok_s, e2e_per_tok_ms
            );
            eprintln!("   Speculative Emitted Rate:             {:6.2} tok/s (emitted_tokens / true_wall_clock_round_time)", round_rate_tok_s);
            eprintln!("------------------------------------------------------------------------------------------");
            eprintln!("B. Speculative Acceptance Statistics:");
            eprintln!("   - Total Output Tokens:               {:6}", gen_toks);
            eprintln!("   - Speculative Verification Rounds:   {:6}", spec_rounds);
            eprintln!("   - Total Tokens Drafted:              {:6}", total_drafts);
            eprintln!(
                "   - Total Draft Tokens Accepted:       {:6}",
                total_accepted
            );
            eprintln!(
                "   - Draft Acceptance Rate:             {:6.1}%",
                accept_pct
            );
            eprintln!(
                "   - Mean Accepted Drafts/Round:        {:6.2}",
                mean_accepted_per_round
            );
            eprintln!(
                "   - Mean Emitted Tokens/Round:         {:6.2}",
                mean_emitted_per_round
            );
            eprintln!("------------------------------------------------------------------------------------------");
            eprintln!("C. Round Latency Breakdown (Per Speculative Round):");
            eprintln!(
                "   - true wall-clock ms/round:           {:6.2} ms (100.0%)",
                true_wall_ms_per_round
            );
            eprintln!(
                "     * pure GPU execution ms:            {:6.2} ms ({:4.1}%)",
                pure_gpu_hw_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * pure_gpu_hw_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * CPU command encoding ms:          {:6.2} ms ({:4.1}%)",
                cpu_encode_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * cpu_encode_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * GPU wait/synchronization ms:      {:6.2} ms ({:4.1}%)",
                gpu_wait_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * gpu_wait_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * argmax readback ms:               {:6.2} ms ({:4.1}%)",
                readback_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * readback_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * acceptance/rollback bookkeeping ms: {:4.2} ms ({:4.1}%)",
                bookkeeping_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * bookkeeping_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * drafting ms:                      {:6.3} ms ({:4.1}%)",
                draft_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * draft_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!(
                "     * (target verify wall-clock total): {:6.2} ms ({:4.1}%)",
                verify_wall_ms_per_round,
                if true_wall_ms_per_round > 0.0 {
                    100.0 * verify_wall_ms_per_round / true_wall_ms_per_round
                } else {
                    0.0
                }
            );
            eprintln!("------------------------------------------------------------------------------------------");
            eprintln!("D. Session Latency Summary:");
            eprintln!(
                "   - Prefill Duration (Prompt):         {:6.2} ms",
                prefill_duration.as_secs_f64() * 1000.0
            );
            eprintln!(
                "   - Net Wall-Clock Decode Duration:    {:6.2} s",
                total_gen_duration.as_secs_f64()
            );
            eprintln!("==========================================================================================\n");
        }

        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Greedy generate up to `max_new` tokens from `prompt` on the GPU.
    #[allow(clippy::explicit_counter_loop)] // `pos` is an absolute sequence index
    pub fn generate_greedy(&self, prompt: &str, max_new: usize) -> Result<(String, Vec<u32>)> {
        if let Ok(spec_str) = std::env::var("CAMELID_GEMMA4_SPEC_NGRAM") {
            if let Ok(max_draft) = spec_str.parse::<usize>() {
                if max_draft > 0 {
                    return self.generate_greedy_speculative_gpu(prompt, max_new, max_draft);
                }
            }
        }
        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut next_tok = 0u32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            next_tok = self.forward_argmax(tok, pos)?;
        }
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        let t_gen_start = std::time::Instant::now();
        for _ in 0..max_new {
            if eot.contains(&next_tok) {
                break;
            }
            generated.push(next_tok);
            next_tok = self.forward_argmax(next_tok, pos)?;
            pos += 1;
        }
        let total_gen_duration = t_gen_start.elapsed();
        if std::env::var("CAMELID_GEMMA4_GPU_TIMING").is_ok()
            || std::env::var("CAMELID_GEMMA4_GPU_PROFILE").is_ok()
        {
            use std::sync::atomic::Ordering::Relaxed;
            let n = FWD_N.load(Relaxed).max(1);
            let prep = PREP_US.load(Relaxed);
            let gpu = GPU_US.load(Relaxed);
            let gpu_hw = crate::metal::GPU_HW_US.load(Relaxed);
            let gen_toks = generated.len().max(1);
            let e2e_per_tok_ms = total_gen_duration.as_secs_f64() * 1000.0 / gen_toks as f64;
            let actual_tok_s = gen_toks as f64 / total_gen_duration.as_secs_f64();

            if let Ok(mut lats) = STEP_LATENCIES_US.lock() {
                if !lats.is_empty() {
                    lats.sort_unstable();
                    let count = lats.len();
                    let mean = gpu as f64 / (count as f64 * 1000.0);
                    let p50 = lats[count / 2] as f64 / 1000.0;
                    let p95 = lats[(count * 95) / 100] as f64 / 1000.0;
                    let min = lats[0] as f64 / 1000.0;
                    let max = lats[count - 1] as f64 / 1000.0;

                    let prep_ms = (prep as f64 / n as f64) / 1000.0;
                    let gpu_wait_ms = (gpu as f64 / n as f64) / 1000.0;
                    let gpu_hw_ms = (gpu_hw as f64 / n as f64) / 1000.0;
                    let queue_delay_ms = (gpu_wait_ms - gpu_hw_ms).max(0.0);
                    let loop_overhead_ms = (e2e_per_tok_ms - prep_ms - gpu_wait_ms).max(0.0);

                    eprintln!("\n==========================================================================================");
                    eprintln!("                           END-TO-END TIMING RECONCILIATION                               ");
                    eprintln!("==========================================================================================");
                    eprintln!("A. Actual End-to-End Generation Time:    {:6.2} ms/token ({:.2} tok/s sustained)", e2e_per_tok_ms, actual_tok_s);
                    eprintln!("B. Actual GPU Forward Latency (p50):     {:6.2} ms/token (mean = {:.2} ms, p95 = {:.2} ms)", p50, mean, p95);
                    eprintln!(
                        "   - CPU Preparation / Embedding / RoPE: {:6.3} ms/token",
                        prep_ms
                    );
                    eprintln!(
                        "   - Metal Command Queue / OS Delay:     {:6.3} ms/token",
                        queue_delay_ms
                    );
                    eprintln!(
                        "   - GPU Hardware On-Chip Execution:     {:6.3} ms/token",
                        gpu_hw_ms
                    );
                    eprintln!(
                        "   - CPU Loop Bookkeeping / Extraction:  {:6.3} ms/token",
                        loop_overhead_ms
                    );
                    eprintln!("------------------------------------------------------------------------------------------");
                    eprintln!("Forward Latency Distribution ({count} forwards):");
                    eprintln!(
                        "   Min: {:6.2} ms | p50: {:6.2} ms | p95: {:6.2} ms | Max: {:6.2} ms",
                        min, p50, p95, max
                    );
                    eprintln!("==========================================================================================\n");
                }
                lats.clear();
            }
        }
        let text = self.tokenizer.decode(&generated, true)?;
        Ok((text, generated))
    }

    /// Cancellable greedy generation on GPU with speculative acceleration.
    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        use crate::inference::speculative::{accepted_draft_prefix, NGramDrafter};

        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut next_tok = 0u32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: 0,
                });
            }
            next_tok = self.forward_argmax(tok, pos)?;
        }

        let drafter = NGramDrafter::new(1, 6);
        let mut history = prompt_tokens.clone();
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        let max_draft = 5usize;

        while generated.len() < max_new {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
            if eot.contains(&next_tok) {
                break;
            }
            generated.push(next_tok);
            history.push(next_tok);
            if generated.len() >= max_new {
                break;
            }

            let budget = (max_new - generated.len()).min(max_draft);
            let drafts = drafter.draft(&history, budget);
            if drafts.is_empty() {
                next_tok = self.forward_argmax(next_tok, pos)?;
                pos += 1;
                continue;
            }

            let mut chunk = Vec::with_capacity(1 + drafts.len());
            chunk.push(next_tok);
            chunk.extend_from_slice(&drafts);

            let (preds, _timings) = self.verify_batch_argmax(&chunk, pos)?;
            let accepted = accepted_draft_prefix(&drafts, &preds[..drafts.len()]);

            let mut stopped = false;
            for &d in &drafts[..accepted] {
                if should_cancel() {
                    return Ok(Gemma4GenerationOutcome::Cancelled {
                        generated_tokens: generated.len(),
                    });
                }
                if generated.len() >= max_new {
                    stopped = true;
                    break;
                }
                generated.push(d);
                history.push(d);
                if eot.contains(&d) {
                    stopped = true;
                    break;
                }
            }

            pos += 1 + accepted;
            next_tok = preds[accepted];
            if stopped {
                break;
            }
        }

        let text = self.tokenizer.decode(&generated, true)?;
        Ok(Gemma4GenerationOutcome::Complete {
            text,
            token_ids: generated,
        })
    }

    /// Streaming cancellable greedy generation on GPU with speculative acceleration.
    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        use crate::inference::speculative::{accepted_draft_prefix, NGramDrafter};

        let prompt_tokens = self.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.tokenizer);
        let mut next_tok = 0u32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: 0,
                });
            }
            next_tok = self.forward_argmax(tok, pos)?;
        }

        let drafter = NGramDrafter::new(1, 6);
        let mut history = prompt_tokens.clone();
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();
        let mut prev_text = String::new();
        let max_draft = 5usize;

        while generated.len() < max_new {
            if should_cancel() {
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
            if eot.contains(&next_tok) {
                break;
            }
            generated.push(next_tok);
            history.push(next_tok);

            let full_text = self.tokenizer.decode(&generated, true)?;
            if full_text.len() > prev_text.len() {
                on_delta(&full_text[prev_text.len()..]);
                prev_text = full_text;
            }

            if generated.len() >= max_new {
                break;
            }

            let budget = (max_new - generated.len()).min(max_draft);
            let drafts = drafter.draft(&history, budget);
            if drafts.is_empty() {
                next_tok = self.forward_argmax(next_tok, pos)?;
                pos += 1;
                continue;
            }

            let mut chunk = Vec::with_capacity(1 + drafts.len());
            chunk.push(next_tok);
            chunk.extend_from_slice(&drafts);

            let (preds, _timings) = self.verify_batch_argmax(&chunk, pos)?;
            let accepted = accepted_draft_prefix(&drafts, &preds[..drafts.len()]);

            let mut stopped = false;
            for &d in &drafts[..accepted] {
                if should_cancel() {
                    return Ok(Gemma4GenerationOutcome::Cancelled {
                        generated_tokens: generated.len(),
                    });
                }
                if generated.len() >= max_new {
                    stopped = true;
                    break;
                }
                generated.push(d);
                history.push(d);
                let full_text = self.tokenizer.decode(&generated, true)?;
                if full_text.len() > prev_text.len() {
                    on_delta(&full_text[prev_text.len()..]);
                    prev_text = full_text;
                }
                if eot.contains(&d) {
                    stopped = true;
                    break;
                }
            }

            pos += 1 + accepted;
            next_tok = preds[accepted];
            if stopped {
                break;
            }
        }

        let text = self.tokenizer.decode(&generated, true)?;
        Ok(Gemma4GenerationOutcome::Complete {
            text,
            token_ids: generated,
        })
    }
}

#[cfg(target_os = "macos")]
static PREP_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static GPU_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static FWD_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(target_os = "macos")]
static STEP_LATENCIES_US: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());

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

/// Quant lane of the GPU tied head: Q8_0 (`q8_gemv`, Q8_0 input), Q4_K (`q4k_gemv`,
/// Q8_K input) or Q6_K (`q6k_gemv`, Q8_K input). Each lane's GEMV reads a specific
/// GPU-side byte layout — see [`gemma4_head_upload`], which is the ONLY way the head
/// weight may reach VRAM.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HeadLane {
    Q8_0,
    Q4K,
    Q6K,
}

/// Convert the tied head's GGUF wire bytes into the GPU layout ITS GEMV reads.
///
/// None of these kernels reads the stock wire: `q8_gemv` wants the SoA split (quants
/// then f16 scales), `q4k_gemv` wants the quant-byte swizzle that makes each aux
/// lane's four stride-8 bytes one aligned i32, and `q6k_gemv` indexes super-blocks at
/// a 224-byte PADDED stride, not the 210-byte wire stride. This mirrors
/// `cuda_resident::repack_for_lane`, which is what every OTHER resident lane in the
/// tree already routes through.
///
/// Root cause of the gemma4 Q4_0 mis-decode: the Q4_K and Q6_K arms used to
/// `clone_htod` the raw wire while only the Q8_0 arm repacked. Since a Q4_0-quantized
/// gemma4 export carries a **Q4_K** `token_embd` (the E2B Q4_0 row does), that row ran
/// `q4k_gemv` over unswizzled bytes: every logit was formed from correctly-addressed
/// but wrongly-PAIRED nibbles, so the lane emitted fluent-looking nonsense instead of
/// refusing. Measured before the fix, "Name the capital of France in one word." →
/// "passe dép oficialmenteynam shalthapp lenghtynam" on CUDA vs "Paris" on the CPU
/// runtime. The Q6_K arm had the same defect one step worse — a 210-vs-224 stride
/// mismatch that also reads past the end of the allocation.
///
/// Head lane is chosen from `token_embd`'s format, which no admission check inspects,
/// so the only gemma4 row ever validated on this lane (E4B Q8_0) was the one whose
/// head happened to be Q8_0. Routing all three lanes through one function is what
/// keeps that class of bug from coming back.
#[cfg(feature = "cuda")]
fn gemma4_head_upload(lane: HeadLane, wire: &[u8]) -> Vec<u8> {
    match lane {
        HeadLane::Q8_0 => q8_wire_to_soa(wire),
        HeadLane::Q4K => crate::cuda_resident::swz_q4k_blocks(wire),
        HeadLane::Q6K => crate::cuda_resident::pad_q6k_blocks(wire),
    }
}

/// Resident GPU tied head. `weight` is the vocab-major projection in its lane's GPU
/// layout (always via [`gemma4_head_upload`]); input is quantized by the fused
/// rms_norm+quantize into `inq`/`ins`; `logits` is dtoh'd once per token. `blocks` is
/// blocks-per-row passed to the GEMV (`hidden/32` for Q8_0, `hidden/256` for K-quants).
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

/// One cached MoE expert's slot in the fixed gate/up and down Q4_0 GPU arenas.
/// `last_used` is the LRU recency stamp.
#[cfg(feature = "cuda")]
struct SserExpertDev {
    slot: usize,
    last_used: u64,
}

/// One page-locked host staging slot for a routed expert's two weight tensors.
/// A top-k-sized ring lets CPU mmap copies run ahead of the CUDA copy engine
/// without pinning the multi-gigabyte `.cghost` mapping itself.
#[cfg(feature = "cuda")]
struct SserTransferSlot {
    gate_up: cudarc::driver::PinnedHostSlice<u8>,
    down: cudarc::driver::PinnedHostSlice<u8>,
}

/// Page-locked host memory allocated with DEFAULT (cacheable) flags.
///
/// `CudaContext::alloc_pinned` hardcodes WRITE_COMBINED. On this platform's PCIe
/// link, cacheable pinned memory measured ~18% FASTER for host→device DMA (9.4 vs
/// 7.9 GB/s back to back — see the same finding at `cuda_resident.rs`'s
/// `CacheablePinned`). The routed-expert tier is read by DMA on every VRAM miss
/// and written only when a record is first admitted, so it wants the read-fast
/// flavour. The driver auto-detects the pinned pointer, so a plain slice view
/// drives the fast async DMA path with no staging copy.
#[cfg(feature = "cuda")]
struct SserPinnedSlab {
    ptr: *mut u8,
    len: usize,
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// SAFETY: `ptr` is a page-locked host allocation owned solely by this struct and
// freed on drop. The resident engine is only ever entered under the
// process-global resident-cache mutex (the same discipline that lets its
// `CudaGraph` be `Send`), so the pointer is never touched from two threads.
#[cfg(feature = "cuda")]
unsafe impl Send for SserPinnedSlab {}

#[cfg(feature = "cuda")]
impl SserPinnedSlab {
    fn new(ctx: &std::sync::Arc<cudarc::driver::CudaContext>, len: usize) -> Result<Self> {
        ctx.bind_to_thread().map_err(cu)?;
        // flags = 0 → cacheable (NOT write-combined). max(1) avoids a zero-size alloc.
        let ptr =
            unsafe { cudarc::driver::result::malloc_host(len.max(1), 0) }.map_err(cu)? as *mut u8;
        if ptr.is_null() {
            return Err(BackendError::InvalidModelMetadata(
                "Ghost-MoE CUDA host tier: malloc_host returned null".into(),
            ));
        }
        Ok(Self {
            ptr,
            len,
            ctx: ctx.clone(),
        })
    }

    /// SAFETY contract for both accessors: `range` must lie inside `0..len`, and
    /// the caller must not alias one slot mutably while a DMA from it is in
    /// flight. The tier enforces that by never recycling a slot that the current
    /// route selected (`protected`), mirroring the VRAM arena's own invariant.
    fn slice(&self, start: usize, len: usize) -> &[u8] {
        debug_assert!(start + len <= self.len);
        unsafe { std::slice::from_raw_parts(self.ptr.add(start), len) }
    }

    fn slice_mut(&mut self, start: usize, len: usize) -> &mut [u8] {
        debug_assert!(start + len <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }
}

#[cfg(feature = "cuda")]
impl Drop for SserPinnedSlab {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr as *mut std::ffi::c_void);
        }
    }
}

/// Tier 2 of the routed-expert hierarchy: a bounded, page-locked host arena of
/// whole `.cghost` expert records, sitting between the VRAM cache and storage.
///
/// The tier exists because storage cannot serve this workload. One decode step
/// routes 240 records (806 MB on the 26B row); the VRAM cache holds ~21% of the
/// 3840 slices, so ~73 records per token have to come from somewhere else. The
/// tracked box's NVMe delivers 1.3 GB/s buffered / 1.9 GB/s unbuffered and — the
/// decisive measurement — is FLAT from queue depth 1 to 16, so read parallelism
/// buys nothing and storage alone pins decode near 5 tok/s.
///
/// Records are held in fixed-stride slots inside ONE page-locked slab rather than
/// as individual heap allocations, which is what makes a tier hit cost a single
/// async DMA: the copy stream reads the slot directly and the CPU never touches
/// the bytes. Staging each miss through a pinned ring instead would add a
/// ~3.19 MiB host memcpy per miss — about 0.4 ms, or ~30 ms/token at 73 misses.
/// Page-locked records are allocated in chunks of this many slots rather than as
/// one slab. Measured on the tracked box: a single 7.9 GiB `malloc_host` fails
/// with `CUDA_ERROR_OUT_OF_MEMORY` even with 9.7 GiB free, while the same total
/// in ~800 MiB pieces does not have to find one contiguous locked range. Chunking
/// also makes the tier degrade gracefully — a request the host cannot fully honour
/// keeps the chunks it did get instead of collapsing to no tier at all, which is
/// the difference between a smaller tier and falling all the way back to storage.
#[cfg(feature = "cuda")]
const GHOST_CUDA_HOST_TIER_SLOTS_PER_CHUNK: usize = 256;

#[cfg(feature = "cuda")]
struct SserHostTier {
    /// Page-locked chunks, each holding `slots_per_chunk` whole records. A record
    /// never straddles a chunk boundary.
    chunks: Vec<SserPinnedSlab>,
    slots_per_chunk: usize,
    /// Byte stride of one whole expert record (gate_up ‖ down, as `.cghost`
    /// stores it). Validated against every record read.
    stride: usize,
    /// `gate_up` / `down` byte ranges within a record, learned once from a real
    /// record and re-validated on each admission.
    gu_off: usize,
    gu_len: usize,
    down_off: usize,
    down_len: usize,
    entries: std::collections::HashMap<(u16, u16), SserHostEntry>,
    free_slots: Vec<usize>,
    clock: u64,
    hits: u64,
    misses: u64,
    bytes_read: u64,
}

#[cfg(feature = "cuda")]
struct SserHostEntry {
    slot: usize,
    last_used: u64,
}

/// Resident per-layer scratch for the routed MoE branch.
///
/// These were allocated per layer per token. With a 95.9%-hit route — i.e. with
/// expert transfers all but removed — the expert loop still burned 26.7 ms/token
/// in prep/alloc, which at 30 layers is 0.89 ms of driver work per layer against
/// a 50 ms/token budget. Every buffer here is sized to the per-layer maximum at
/// load, so a layer uses a prefix and the kernels read byte-identical inputs.
#[cfg(feature = "cuda")]
struct MoeScratchDev {
    /// Q8_0 scales of the shared expert input (`hidden / 32`).
    in_s: cudarc::driver::CudaSlice<f32>,
    /// Q8_0 quants of the shared expert input (`hidden`).
    in_q: cudarc::driver::CudaSlice<i8>,
    /// Fused gate‖up GEMV output, `max_route * 2 * n_ff_exp`.
    gate_up: cudarc::driver::CudaSlice<f32>,
    /// GeGLU output requantized to Q8_0, `max_route * n_ff_exp`.
    geglu_q: cudarc::driver::CudaSlice<i8>,
    /// GeGLU Q8_0 scales, `max_route * n_ff_exp / 32`.
    geglu_s: cudarc::driver::CudaSlice<f32>,
    /// Per-expert down-GEMV outputs in router order, `max_route * hidden`.
    ///
    /// The three tiny index/scale uploads inside the routed path (`d_slots`,
    /// `d_routes`, `d_route_scales`, ≤32 bytes each) are deliberately NOT hoisted
    /// here: reaching them from `moe_layer_ffn_cached_routed` would need either
    /// three more parameters on an already 14-argument signature or a second
    /// `borrow_mut` of this cell while the caller still holds one, which panics.
    /// They are a small share of the allocation cost the large buffers dominate.
    y_all: cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
impl SserHostTier {
    /// Build the tier, or `Ok(None)` when it is disabled or cannot be page-locked.
    ///
    /// Failure here is never fatal: the caller keeps the storage-backed path, which
    /// is slower but correct. Page-locking is the one step that can legitimately
    /// fail on a loaded host, so it is the last thing done and its error is
    /// downgraded to a warning.
    fn new(
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        ghost: &GhostFile,
        budget_bytes: usize,
    ) -> Result<Option<Self>> {
        if budget_bytes == 0 {
            return Ok(None);
        }
        // Learn the record geometry from a real record rather than deriving it, so
        // a layout change in the repacker surfaces as a refusal instead of silent
        // garbage.
        let probe = ghost.read_moe_expert(0, 0)?;
        let stride = probe.byte_len();
        let gu = probe.gate_up.record_range();
        let down = probe.down.record_range();
        if stride == 0 || gu.end > stride || down.end > stride {
            return Err(BackendError::InvalidModelMetadata(format!(
                "Ghost-MoE CUDA host tier: record layout out of range (stride {stride}, gate_up {gu:?}, down {down:?})"
            )));
        }
        // The tier is a fixed-offset consumer: `views` applies the probe's
        // interior offsets to EVERY slot. The generic Ghost readers deliberately
        // accept any contiguous per-record tensor order, so probing one record
        // proves nothing about the other 3839 — a format-legal file with one
        // group serialized `[down][gate_up]` would pass every length and
        // identity check and DMA down-bytes as gate_up weights, silently. The
        // Metal persistent-slot lane already refuses such files through this
        // same canonical-layout gate; review caught that this lane skipped it.
        // Metadata-only (no I/O), so the cost is negligible.
        let layers = ghost.index.block_count;
        let expert_count = ghost.index.expert_count.unwrap_or(0);
        ghost.validate_moe_expert_record_layouts(layers, expert_count, stride)?;
        let want_slots = budget_bytes / stride;
        if want_slots == 0 {
            return Ok(None);
        }
        // Try the whole budget as ONE allocation first. Measured on the tracked
        // box, a single 7165 MiB `malloc_host` SUCCEEDS while the same total in
        // 817 MiB pieces stops at 6534 MiB — the driver's page-locking limit is
        // not simply additive, and chunking cost 198 records (2246 -> 2048) and
        // 6 points of tier hit rate. So chunks are the FALLBACK, not the default.
        if let Ok(chunk) = SserPinnedSlab::new(ctx, want_slots * stride) {
            return Ok(Some(Self {
                chunks: vec![chunk],
                slots_per_chunk: want_slots,
                stride,
                gu_off: gu.start,
                gu_len: gu.end - gu.start,
                down_off: down.start,
                down_len: down.end - down.start,
                entries: std::collections::HashMap::with_capacity(want_slots),
                free_slots: (0..want_slots).rev().collect(),
                clock: 0,
                hits: 0,
                misses: 0,
                bytes_read: 0,
            }));
        }
        // Otherwise take what the host will give, in chunks, and keep it. Stopping
        // at the first refusal (rather than erroring) is deliberate: a tier that is
        // smaller than requested still converts storage reads into PCIe copies,
        // whereas no tier at all costs ~2 ms per VRAM miss.
        let slots_per_chunk = GHOST_CUDA_HOST_TIER_SLOTS_PER_CHUNK.min(want_slots.max(1));
        let mut chunks = Vec::with_capacity(want_slots.div_ceil(slots_per_chunk));
        let mut slots = 0usize;
        let mut remaining = want_slots;
        while remaining > 0 {
            // The last chunk is sized to what is actually left — rounding it up
            // to a full chunk would pin up to `slots_per_chunk - 1` records
            // beyond the caller's budget. `slot_location` stays valid with a
            // short final chunk because slot ids never reach into its padding.
            let this_slots = slots_per_chunk.min(remaining);
            match SserPinnedSlab::new(ctx, this_slots * stride) {
                Ok(chunk) => {
                    chunks.push(chunk);
                    slots += this_slots;
                    remaining -= this_slots;
                }
                Err(e) => {
                    eprintln!(
                        "[ghost] host expert tier: host stopped page-locking after {} MiB of {} MiB \
                         requested ({e}); keeping the smaller tier",
                        (slots * stride) / (1024 * 1024),
                        (want_slots * stride) / (1024 * 1024),
                    );
                    break;
                }
            }
        }
        if chunks.is_empty() {
            eprintln!(
                "[ghost] host expert tier: could not page-lock even one {} MiB chunk; \
                 continuing on the storage-backed path",
                (slots_per_chunk * stride) / (1024 * 1024)
            );
            return Ok(None);
        }
        Ok(Some(Self {
            chunks,
            slots_per_chunk,
            stride,
            gu_off: gu.start,
            gu_len: gu.end - gu.start,
            down_off: down.start,
            down_len: down.end - down.start,
            entries: std::collections::HashMap::with_capacity(slots),
            free_slots: (0..slots).rev().collect(),
            clock: 0,
            hits: 0,
            misses: 0,
            bytes_read: 0,
        }))
    }

    /// Make `(layer, expert)` resident and return its slot.
    ///
    /// SLOT-REUSE SAFETY. The copy stream DMAs straight out of a slot, so
    /// overwriting a slot whose transfer is still in flight would silently
    /// corrupt an expert. Two things together make that unreachable, and BOTH
    /// are load-bearing:
    ///
    /// 1. Within a route, `protected` holds every key this route selected, so a
    ///    later miss in the same layer cannot recycle an earlier one's slot.
    /// 2. Across layers, `forward_token` runs a full `cap_stream.synchronize()`
    ///    at the top of each MoE layer, and the previous layer made `cap_stream`
    ///    wait on its `copy_done` event — so that synchronize transitively
    ///    retires the previous layer's copies before any slot here is recycled.
    ///
    /// If (2) is ever removed — dropping the per-layer drain is a standing
    /// optimization candidate, since it costs a WDDM flush 30x per token — this
    /// tier needs its own per-slot completion tracking (an event or generation
    /// counter checked before recycling). Do not remove it silently.
    fn ensure_resident(
        &mut self,
        ghost: &GhostFile,
        layer: usize,
        expert: usize,
        protected: &std::collections::HashSet<(u16, u16)>,
    ) -> Result<usize> {
        let key = (layer as u16, expert as u16);
        self.clock += 1;
        let stamp = self.clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = stamp;
            self.hits += 1;
            return Ok(entry.slot);
        }
        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None => {
                let victim = self
                    .entries
                    .iter()
                    .filter(|(k, _)| !protected.contains(*k))
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(k, _)| *k)
                    .ok_or_else(|| {
                        BackendError::InvalidModelMetadata(
                            "Ghost-MoE CUDA host tier has no recyclable slot for this route".into(),
                        )
                    })?;
                self.entries
                    .remove(&victim)
                    .expect("selected host-tier victim is resident")
                    .slot
            }
        };
        let stride = self.stride;
        // Reads land straight in the page-locked slot: no intermediate Vec, and
        // the copy stream can DMA from here without the CPU touching the bytes.
        let (chunk_idx, chunk_off) = self.slot_location(slot);
        ghost.read_moe_expert_into(
            layer,
            expert,
            self.chunks[chunk_idx].slice_mut(chunk_off, stride),
        )?;
        self.misses += 1;
        self.bytes_read = self.bytes_read.saturating_add(stride as u64);
        self.entries.insert(
            key,
            SserHostEntry {
                slot,
                last_used: stamp,
            },
        );
        Ok(slot)
    }

    /// Fill free slots with records at load, striped uniformly across layers.
    ///
    /// OPT-IN (`CAMELID_GEMMA4_GHOST_TIER_PREFILL=1`), and measured HONESTLY
    /// NEUTRAL-TO-NEGATIVE on the tracked 16 GiB box: it raised the tier hit
    /// rate 66.2% -> 74.9% and cut storage reads 26%, but steady decode moved
    /// 10.62 -> 9.71 tok/s and total wall time got worse — the 6.9 GiB prefill
    /// read evicts the OS page cache's copy of the same payload, so the misses
    /// that remain get colder by as much as the tier hits gain. It stays
    /// available because the trade flips when the tier can hold the entire
    /// routed payload (host RAM ≥ payload + reserve): one sequential read at
    /// load then eliminates storage from decode entirely. Records are stamped
    /// `last_used = 0`, so anything the session actually touches immediately
    /// outranks an unproven prefill for eviction.
    fn prefill(&mut self, ghost: &GhostFile) {
        let enabled = std::env::var("CAMELID_GEMMA4_GHOST_TIER_PREFILL")
            .ok()
            .is_some_and(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes" | "enabled"
                )
            });
        if !enabled {
            return;
        }
        let layers = ghost.index.block_count;
        let expert_count = ghost.index.expert_count.unwrap_or(0);
        if layers == 0 || expert_count == 0 {
            return;
        }
        let per_layer = (self.free_slots.len() / layers).min(expert_count);
        if per_layer == 0 {
            return;
        }
        let started = std::time::Instant::now();
        let mut filled = 0usize;
        'fill: for layer in 0..layers {
            for expert in 0..per_layer {
                let Some(slot) = self.free_slots.pop() else {
                    break 'fill;
                };
                let stride = self.stride;
                let (chunk_idx, chunk_off) = self.slot_location(slot);
                match ghost.read_moe_expert_into(
                    layer,
                    expert,
                    self.chunks[chunk_idx].slice_mut(chunk_off, stride),
                ) {
                    Ok(()) => {
                        self.entries.insert(
                            (layer as u16, expert as u16),
                            SserHostEntry { slot, last_used: 0 },
                        );
                        filled += 1;
                    }
                    Err(e) => {
                        // A read failure at prefill is not a load failure: give the
                        // slot back and let the per-miss path (which reports errors
                        // in context) deal with this record if routing ever wants it.
                        self.free_slots.push(slot);
                        eprintln!(
                            "[ghost] host tier prefill stopped at layer {layer} expert {expert}: {e}"
                        );
                        break 'fill;
                    }
                }
            }
        }
        eprintln!(
            "[ghost] host tier prefill: {filled} records ({:.2} GiB, {}/{} per layer) in {:.1}s",
            (filled * self.stride) as f64 / (1024.0 * 1024.0 * 1024.0),
            per_layer,
            expert_count,
            started.elapsed().as_secs_f32(),
        );
    }

    /// Map a slot to its page-locked chunk and byte offset within it. Records
    /// never straddle a chunk, so both projections of a record live together.
    fn slot_location(&self, slot: usize) -> (usize, usize) {
        (
            slot / self.slots_per_chunk,
            (slot % self.slots_per_chunk) * self.stride,
        )
    }

    /// The `gate_up` and `down` byte views of a resident slot, ready to DMA.
    fn views(&self, slot: usize) -> (&[u8], &[u8]) {
        let (chunk_idx, base) = self.slot_location(slot);
        let chunk = &self.chunks[chunk_idx];
        (
            chunk.slice(base + self.gu_off, self.gu_len),
            chunk.slice(base + self.down_off, self.down_len),
        )
    }

    fn stats(&self) -> (u64, u64, usize, usize) {
        (
            self.hits,
            self.misses,
            self.entries.len(),
            self.entries.len() + self.free_slots.len(),
        )
    }
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
enum GhostCudaExpertRecord {
    Owned(Arc<GhostMoeExpert>),
    Mapped(GhostMoeMappedExpert),
}

#[cfg(feature = "cuda")]
impl GhostCudaExpertRecord {
    fn gate_up(&self) -> Result<(GgufTensorType, &[u8])> {
        match self {
            Self::Owned(expert) => Ok((expert.gate_up.dtype, expert.tensor_bytes(&expert.gate_up))),
            Self::Mapped(expert) => {
                Ok((expert.gate_up.dtype, expert.tensor_bytes(&expert.gate_up)?))
            }
        }
    }

    fn down(&self) -> Result<(GgufTensorType, &[u8])> {
        match self {
            Self::Owned(expert) => Ok((expert.down.dtype, expert.tensor_bytes(&expert.down))),
            Self::Mapped(expert) => Ok((expert.down.dtype, expert.tensor_bytes(&expert.down)?)),
        }
    }
}

#[cfg(feature = "cuda")]
struct SserCache {
    entries: std::collections::HashMap<(u16, u16), SserExpertDev>,
    capacity: usize,
    clock: u64,
    gate_up_arena: cudarc::driver::CudaSlice<u8>,
    down_arena: cudarc::driver::CudaSlice<u8>,
    gate_up_stride: usize,
    down_stride: usize,
    free_slots: Vec<usize>,
    transfer_slots: Vec<SserTransferSlot>,
    // Diagnostics (per-generate; reset by the harness before each run).
    hits: u64,
    misses: u64,
}

#[cfg(feature = "cuda")]
impl SserCache {
    fn new(
        capacity: usize,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        gate_up_stride: usize,
        down_stride: usize,
        transfer_slot_count: usize,
    ) -> Result<Self> {
        let capacity = capacity.max(1);
        // SAFETY: a slot is initialized by HtoD before it is inserted into
        // `entries`, and only entries can be exposed to a GEMV.
        let gate_up_arena = unsafe { stream.alloc::<u8>(capacity * gate_up_stride) }.map_err(cu)?;
        // SAFETY: see above.
        let down_arena = unsafe { stream.alloc::<u8>(capacity * down_stride) }.map_err(cu)?;
        let mut transfer_slots = Vec::with_capacity(transfer_slot_count);
        for _ in 0..transfer_slot_count {
            transfer_slots.push(SserTransferSlot {
                gate_up: unsafe { stream.context().alloc_pinned::<u8>(gate_up_stride) }
                    .map_err(cu)?,
                down: unsafe { stream.context().alloc_pinned::<u8>(down_stride) }.map_err(cu)?,
            });
        }
        Ok(Self {
            entries: std::collections::HashMap::new(),
            capacity,
            clock: 0,
            gate_up_arena,
            down_arena,
            gate_up_stride,
            down_stride,
            free_slots: (0..capacity).rev().collect(),
            transfer_slots,
            hits: 0,
            misses: 0,
        })
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

    /// Return a free slot or remove and recycle the least-recently-used slot.
    fn slot_for_miss(&mut self) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            return slot;
        }
        let victim = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&key, _)| key)
            .expect("a full SSER cache has an LRU victim");
        self.entries
            .remove(&victim)
            .expect("the selected SSER victim remains resident")
            .slot
    }

    /// Reserve a slot without evicting any expert selected by the current
    /// route. Batched execution resolves all slots before launching, so routed
    /// hits must remain pinned until their GEMVs have consumed them.
    fn slot_for_miss_excluding(
        &mut self,
        protected: &std::collections::HashSet<(u16, u16)>,
    ) -> Option<usize> {
        if let Some(slot) = self.free_slots.pop() {
            return Some(slot);
        }
        let victim = self
            .entries
            .iter()
            .filter(|(key, _)| !protected.contains(key))
            .min_by_key(|(_, e)| e.last_used)
            .map(|(&key, _)| key)?;
        Some(
            self.entries
                .remove(&victim)
                .expect("the selected SSER victim remains resident")
                .slot,
        )
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
    /// Dedicated expert-weight upload stream. Routed hits execute on
    /// `cap_stream` while missing arena slots are populated by the copy engine.
    expert_copy_stream: std::sync::Arc<cudarc::driver::CudaStream>,
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
    /// Tier 2: page-locked host residency for routed expert records, between the
    /// VRAM cache and `.cghost` on storage. `None` keeps the storage-backed path.
    /// Same `RefCell` rationale as `sser`.
    host_tier: Option<std::cell::RefCell<SserHostTier>>,
    /// Resident routed-MoE scratch, replacing 7 stream-ordered allocations per
    /// layer per token. Same `RefCell` rationale as `sser`.
    moe_scratch: Option<std::cell::RefCell<MoeScratchDev>>,
}

/// Per-layer KV cache capacity (positions) for the CUDA-resident lane. A sliding
/// layer's `attention_decode_sw` read starts at `position_count - window`, so no key
/// older than `window` positions back is ever read: its cache is a ring of
/// `window + 1` slots (`kv_scatter` and the attention reads wrap with `% capacity`),
/// while global layers keep the full `max_positions`. On the 26B ghost row this
/// returns ~600 MiB of VRAM at the default 4096-position context (25 sliding layers
/// × K+V × kv_dim 2048 × (4096−1025) positions × 2 B) — headroom the SSER expert
/// cache's VRAM-fit sizing converts directly into more resident routed experts.
#[cfg(feature = "cuda")]
fn gemma4_kv_capacity(window: Option<usize>, max_positions: usize) -> usize {
    window.map_or(max_positions, |w| (w + 1).min(max_positions))
}

#[cfg(feature = "cuda")]
impl Gemma4CudaResident {
    /// Load the model (CPU runtime, weights mmap'd), bring up the CUDA kernels,
    /// upload per-layer norms, and allocate the KV caches + scratch. `max_positions`
    /// bounds the resident KV cache.
    pub fn load(path: &Path, max_positions: usize) -> Result<Self> {
        let cpu = Gemma4Runtime::load(path)?;
        Self::from_cpu_runtime(cpu, max_positions, 0)
    }

    /// Load the common Gemma 4 core from the sparse GGUF shadow and routed
    /// experts from `.cghost`, then execute through the existing CUDA common
    /// core and bounded SSER expert cache. A cache miss uploads exactly one
    /// validated expert record; routed expert holes in the shadow are never read.
    pub fn load_ghost_moe(
        path: &Path,
        cghost: &Path,
        cache_mib: usize,
        evict_page_cache: bool,
        max_positions: usize,
    ) -> Result<Self> {
        // Size the host expert tier from what this host can actually spare. The
        // caller's `--expert-cache-mib` is a floor, not a ceiling: its defaults
        // were tuned for the CPU/storage lane, where the tier is a small read
        // cache, whereas here it decides whether a VRAM miss costs a PCIe copy
        // or a storage read. The tier budget is deliberately SEPARATE from the
        // pageable `GhostMoeExpertCache` budget below — review caught that
        // passing the auto-sized figure to both let the serial-diagnostic +
        // strict-cache combination hold ~2x spare RAM (pinned tier plus a
        // pageable arena the batched path never reads).
        let tier_mib = ghost_cuda_host_tier_mib(cghost, cache_mib);
        let tier_bytes = tier_mib.saturating_mul(1024 * 1024);
        // Buffered reads stay ON. Forcing `FILE_FLAG_NO_BUFFERING` here to keep
        // the tier's residency "exclusive" was measured and is a REGRESSION: the
        // OS page cache is itself an opportunistic tier over the mapping, and
        // turning it off costs more on the tier's cold fills than the exclusivity
        // returns. The tier fills through the same buffered positioned reader, so
        // a fill that the page cache can serve stays at memcpy speed.
        let strict_reads = evict_page_cache;
        // KV context is a direct trade against routed-expert residency, but since
        // the sliding-layer caches became window+1-slot rings (gemma4_kv_capacity)
        // only the 5 global layers still scale with `max_positions`: the 26B row's
        // KV is ~280 MiB at 4096 positions where it used to be 880 MiB. Every
        // 3.19 MiB returned here buys one more resident expert, so the knob still
        // converts context into hit rate — it just has ~4x less lever arm than
        // when all 30 layers paid full context. Default is the caller's value, so
        // nothing changes unless asked. (The Metal ghost lane exposes the
        // equivalent knob.)
        let max_positions = std::env::var("CAMELID_GEMMA4_GHOST_CUDA_CONTEXT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v >= 512)
            .unwrap_or(max_positions);
        let cpu = Gemma4Runtime::load_ghost_moe(path, cghost, cache_mib, strict_reads)?;
        Self::from_cpu_runtime(cpu, max_positions, tier_bytes)
    }

    /// `host_tier_budget_bytes` sizes the page-locked expert tier; `0` means no
    /// tier (dense rows, non-ghost loads, or an explicit opt-out).
    fn from_cpu_runtime(
        cpu: Gemma4Runtime,
        max_positions: usize,
        host_tier_budget_bytes: usize,
    ) -> Result<Self> {
        let ghost_moe = cpu.ghost_moe_cache.is_some();
        // BASALT Amendment 3 review fix: refuse NVFP4 layer projections with a
        // typed error BEFORE the `GemmaLayerQuant::from_wire` catch-all (`upw`
        // below) can panic. The CPU wire lane serves NVFP4 in this release;
        // CUDA-resident NVFP4 is Phase 4 (BASALT).
        // Guard exactly the panic seam this check exists for: the DENSE layer
        // projections that `GemmaLayerQuant::from_wire` repacks. The routed MoE
        // expert tensors never pass through `from_wire` — they stay raw wire
        // bytes in the SSER arenas (GPU) or CPU-side `WireQuant`s — and their
        // formats are validated by the dedicated expert-residency gate below
        // (Q4_0, or the mixed Q2_K gate_up + Q4_0 down). Including them here
        // wrongly refused the mixed artifact this lane now serves.
        nvfp4_cuda_lane_check(cpu.layers.iter().flat_map(|lw| {
            [
                Some(lw.attn_q.format),
                lw.attn_k.as_ref().map(|w| w.format),
                lw.attn_v.as_ref().map(|w| w.format),
                Some(lw.attn_output.format),
                Some(lw.ffn_gate.format),
                Some(lw.ffn_up.format),
                Some(lw.ffn_down.format),
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
        let expert_copy_stream = kernels.ctx.new_stream().map_err(cu)?;
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
                        .clone_htod(&gemma4_head_upload(HeadLane::Q8_0, cpu.token_embd.bytes()))
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
                    weight: s
                        .clone_htod(&gemma4_head_upload(HeadLane::Q6K, cpu.token_embd.bytes()))
                        .map_err(cu)?,
                    output_norm: s.clone_htod(&cpu.output_norm).map_err(cu)?,
                    logits: s.alloc_zeros::<f32>(vocab).map_err(cu)?,
                    inq: s.alloc_zeros::<i8>(blocks * 256).map_err(cu)?,
                    ins: s.alloc_zeros::<f32>(blocks).map_err(cu)?,
                    blocks,
                    softcap,
                })
            }
            // Q4_K tied head (the format a Q4_0-quantized gemma4 export carries):
            // q4k_gemv over the SWIZZLED 144-byte super-blocks, Q8_K input.
            WireFormat::Q4K if hidden.is_multiple_of(256) => {
                let blocks = hidden / 256;
                Some(Gemma4HeadDev {
                    lane: HeadLane::Q4K,
                    weight: s
                        .clone_htod(&gemma4_head_upload(HeadLane::Q4K, cpu.token_embd.bytes()))
                        .map_err(cu)?,
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

        // Per-owning-layer f16 KV caches sized to that layer's kv geometry and
        // position capacity: sliding layers ring on window+1 slots, only global
        // layers pay for the full context (see gemma4_kv_capacity).
        let mut cache_k = Vec::with_capacity(block_count);
        let mut cache_v = Vec::with_capacity(block_count);
        for p in &plan {
            if p.owns_kv {
                let n = p.kv_dim * gemma4_kv_capacity(p.window, max_positions);
                cache_k.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
                cache_v.push(Some(s.alloc_zeros::<u16>(n).map_err(cu)?));
            } else {
                cache_k.push(None);
                cache_v.push(None);
            }
        }

        let alloc_f = |n: usize| s.alloc_zeros::<f32>(n.max(1));
        let alloc_i = |n: usize| s.alloc_zeros::<i8>(n.max(1));
        // SSER (M2): enable the per-(layer,expert) VRAM cache when requested for a
        // normal GGUF, and by default for Ghost-MoE. Capacity defaults to ~1000 experts (the
        // measured hot set); each cached expert is ~2*n_ff_exp*(hidden/32)*18 +
        // hidden*(n_ff_exp/32)*18 bytes of Q4_0 wire (~3.3 MB on the 26B), so ~1000
        // experts ≈ ~3.3 GB — under the ~3.6 GB free after the resident set. Tunable
        // via CAMELID_SSER_CACHE_EXPERTS.
        let first_moe = cpu.layers.iter().find_map(|lw| lw.moe.as_ref());
        let ghost_cuda_cache_enabled = std::env::var("CAMELID_GEMMA4_GHOST_CUDA_CACHE")
            .ok()
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no" | "disabled"
                )
            })
            .unwrap_or(true);
        let sser_requested = std::env::var_os("CAMELID_SSER_CACHE").is_some()
            || (ghost_moe && ghost_cuda_cache_enabled);
        // Residency admits Q4_0 experts (the original certified lane) plus the
        // mixed representation: Q2_K gate_up + Q4_0 down. The mix is not
        // arbitrary — gate_up's input dim (hidden, 2816 = 11×256) is K-quant
        // clean while down's (n_ff_exp, 704) is not divisible by 256, so Q2_K
        // down is impossible by format arithmetic, and keeping down Q4_0 also
        // keeps that half of the routed pipeline byte-identical to the
        // certified path. Each projection dispatches by its own wire format.
        let experts_supported = cpu
            .layers
            .iter()
            .filter_map(|layer| layer.moe.as_ref())
            .all(|moe| {
                matches!(moe.gate_up_exps.format, WireFormat::Q4_0 | WireFormat::Q2K)
                    && moe.down_exps.format == WireFormat::Q4_0
            });
        if ghost_moe && sser_requested && !experts_supported {
            return Err(BackendError::UnsupportedGguf(
                "Ghost-MoE CUDA expert residency requires Q4_0 (or Q2_K gate_up + Q4_0 down) expert records"
                    .into(),
            ));
        }
        let sser = if let (Some(moe), true) = (first_moe, sser_requested) {
            // Per-expert VRAM cost: the two wire slices this expert's GEMVs
            // read, derived from each projection's OWN format. gate_up =
            // 2*n_ff_exp rows of hidden values (Q4_0: 18 B/32 values; Q2_K:
            // 84 B/256 values — the mixed artifact); down = hidden rows of
            // n_ff_exp values, always Q4_0.
            const WB: usize = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
            let two_nff = 2 * moe.n_ff_exp;
            let gu_fmt = moe.gate_up_exps.format;
            let gate_up_row_bytes = hidden / gu_fmt.values_per_block() * gu_fmt.bytes_per_block();
            let per_expert_bytes = two_nff * gate_up_row_bytes + hidden * (moe.n_ff_exp / 32) * WB;
            // Budget: keep the cache under ~80% of the free VRAM after the resident set
            // (leaving headroom for the per-token scratch + the KV cache growth).
            let (free, _total) = cudarc::driver::result::mem_get_info().unwrap_or((0, 0));
            // Cache budget = free VRAM at load MINUS a fixed reserve for routed-batch
            // scratch and driver/WDDM slack. Expert weights upload directly into fixed
            // arenas; the top-k transfer ring is page-locked HOST memory. The KV caches
            // and common scratch are already allocated above, so `free` excludes them.
            // The remaining dynamic need is small and not proportional to free VRAM.
            // A fixed reserve therefore lets the cache claim far more of
            // the card than the old flat 0.80 factor did: on the 6 GB box this lifts
            // the cap ~690 -> ~820 experts, cutting the miss count and measuring
            // +~50% steady decode (miss-bound, capacity-limited). Reserve tunable via
            // CAMELID_SSER_CACHE_RESERVE_MIB; a hard 0.98 cap on the free fraction is a
            // final belt-and-suspenders against a pathologically small `free`.
            let reserve_mib = std::env::var("CAMELID_GEMMA4_GHOST_CUDA_RESERVE_MIB")
                .ok()
                .or_else(|| std::env::var("CAMELID_SSER_CACHE_RESERVE_MIB").ok())
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(160);
            let reserve = reserve_mib * 1024 * 1024;
            let hard_cap = (free as f64 * 0.98) as usize;
            let budget = free.saturating_sub(reserve).min(hard_cap);
            let fit_cap = budget.checked_div(per_expert_bytes).unwrap_or(0);
            let req_cap = std::env::var("CAMELID_GEMMA4_GHOST_CUDA_CACHE_EXPERTS")
                .ok()
                .or_else(|| std::env::var("CAMELID_SSER_CACHE_EXPERTS").ok())
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1000);
            if ghost_moe && fit_cap == 0 {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "gemma4 Ghost-MoE CUDA has only {} MiB free after common weights/KV/head; no routed expert fits while preserving the {} MiB reserve",
                    free / (1024 * 1024),
                    reserve_mib
                )));
            }
            // Honor the smaller of the requested capacity and what free VRAM allows.
            let cap = req_cap.min(fit_cap).max(1);
            eprintln!(
                "[sser] expert-residency cache ON: capacity {cap} experts ({} MiB each; requested {req_cap}, VRAM-fit {fit_cap}; {} MiB free)",
                per_expert_bytes / (1024 * 1024),
                free / (1024 * 1024),
            );
            let gate_up_stride = two_nff * gate_up_row_bytes;
            let down_stride = hidden * (moe.n_ff_exp / 32) * WB;
            let uniform_geometry = cpu
                .layers
                .iter()
                .filter_map(|layer| layer.moe.as_ref())
                .all(|layer_moe| {
                    layer_moe.gate_up_exps.format == gu_fmt
                        && 2 * layer_moe.n_ff_exp * gate_up_row_bytes == gate_up_stride
                        && hidden * (layer_moe.n_ff_exp / 32) * WB == down_stride
                });
            if !uniform_geometry {
                return Err(BackendError::UnsupportedGguf(
                    "Gemma 4 CUDA expert residency requires uniform MoE expert geometry".into(),
                ));
            }
            Some(SserCache::new(
                cap,
                &cap_stream,
                gate_up_stride,
                down_stride,
                moe.n_expert_used,
            )?)
        } else {
            None
        };
        // Tier 2: page-locked host residency for the VRAM cache's miss tail. Only
        // worth building alongside the VRAM cache and a `.cghost`; without one it
        // has nothing to catch, and below the minimum it cannot retain a record
        // across a token boundary.
        // The serial diagnostic path (`CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS=0`) has
        // no tier branch, so building (and possibly prefilling) one it will never
        // read would only pin RAM.
        let routed_batch_enabled = std::env::var("CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS")
            .ok()
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no" | "disabled"
                )
            })
            .unwrap_or(true);
        let host_tier = match (sser.as_ref(), cpu.ghost_moe_cache.as_ref()) {
            (Some(_), Some(ghost_cache))
                if routed_batch_enabled
                    && host_tier_budget_bytes >= GHOST_CUDA_HOST_TIER_MIN_BYTES =>
            {
                // Tier construction failure is never a load failure: the mapped
                // storage path is slower but correct, so downgrade to a warning.
                let mut tier = match SserHostTier::new(
                    cap_stream.context(),
                    &ghost_cache.file,
                    host_tier_budget_bytes,
                ) {
                    Ok(tier) => tier,
                    Err(e) => {
                        eprintln!(
                            "[ghost] host expert tier unavailable ({e}); continuing on the storage-backed path"
                        );
                        None
                    }
                };
                if let Some(tier) = tier.as_mut() {
                    let (_, _, _, slots) = tier.stats();
                    eprintln!(
                        "[ghost] host expert tier: {} slots ({} MiB page-locked, {:.1} MiB/record) covering the VRAM miss tail",
                        slots,
                        (slots * tier.stride) / (1024 * 1024),
                        tier.stride as f64 / (1024.0 * 1024.0),
                    );
                    tier.prefill(&ghost_cache.file);
                }
                tier.map(std::cell::RefCell::new)
            }
            (Some(_), Some(_))
                if host_tier_budget_bytes > 0
                    && host_tier_budget_bytes < GHOST_CUDA_HOST_TIER_MIN_BYTES =>
            {
                eprintln!(
                    "[ghost] host expert tier disabled: {} MiB is below the {} MiB minimum (a tier smaller than one routed working set only adds copies)",
                    host_tier_budget_bytes / (1024 * 1024),
                    GHOST_CUDA_HOST_TIER_MIN_BYTES / (1024 * 1024),
                );
                None
            }
            _ => None,
        };
        // Resident routed-MoE scratch, sized to the per-layer maxima across MoE rows.
        let moe_scratch = match cpu.layers.iter().filter_map(|lw| lw.moe.as_ref()).fold(
            None::<(usize, usize)>,
            |acc, m| {
                let (nff, used) = acc.unwrap_or((0, 0));
                Some((nff.max(m.n_ff_exp), used.max(m.n_expert_used)))
            },
        ) {
            Some((nff_max, route_max)) if nff_max > 0 && route_max > 0 => {
                Some(std::cell::RefCell::new(MoeScratchDev {
                    in_s: alloc_f(hidden / 32).map_err(cu)?,
                    in_q: alloc_i(hidden).map_err(cu)?,
                    gate_up: alloc_f(route_max * 2 * nff_max).map_err(cu)?,
                    geglu_q: alloc_i(route_max * nff_max).map_err(cu)?,
                    geglu_s: alloc_f(route_max * (nff_max / 32).max(1)).map_err(cu)?,
                    y_all: alloc_f(route_max * hidden).map_err(cu)?,
                }))
            }
            _ => None,
        };
        let me = Self {
            host_tier,
            moe_scratch,
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
            expert_copy_stream,
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
        if let Some(tier) = self.host_tier.as_ref() {
            let tier = tier.borrow();
            let (hits, misses, resident, slots) = tier.stats();
            let total = hits + misses;
            eprintln!(
                "[ghost] host tier: {hits} hits / {misses} storage reads = {:.1}% tier hit-rate; \
                 {resident}/{slots} records resident; {:.2} GiB read from .cghost",
                if total > 0 {
                    100.0 * hits as f64 / total as f64
                } else {
                    0.0
                },
                tier.bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        }
        self.sser.as_ref().map(|c| {
            let c = c.borrow();
            (c.hits, c.misses, c.entries.len(), c.capacity)
        })
    }

    /// Load-time Ghost CUDA ownership snapshot for lock-free API health.
    pub fn ghost_cuda_components(&self) -> Gemma4GhostCudaComponents {
        if self.cpu.ghost_moe_cache.is_none() {
            return Gemma4GhostCudaComponents::default();
        }
        Gemma4GhostCudaComponents {
            common: true,
            experts: self.sser.is_some(),
            head: self.gpu_head.is_some(),
        }
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

    /// Execute one top-k route as at most two CUDA batches: already-resident
    /// experts first, then misses after their copy-stream event. Both batches
    /// write into router-positioned scratch, and one final kernel performs the
    /// same left-to-right weighted sum as the serial launch path.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn moe_layer_ffn_cached_routed(
        &self,
        l: usize,
        moe: &MoeWeights,
        sser: &std::cell::RefCell<SserCache>,
        idx: &[usize],
        probs: &[f32],
        wsum: f32,
        mut ghost_records: Option<&mut std::collections::HashMap<usize, GhostCudaExpertRecord>>,
        d_in_s: &cudarc::driver::CudaSlice<f32>,
        d_in_q: &cudarc::driver::CudaSlice<i8>,
        d_gate_up: &mut cudarc::driver::CudaSlice<f32>,
        d_geglu_q: &mut cudarc::driver::CudaSlice<i8>,
        d_geglu_s: &mut cudarc::driver::CudaSlice<f32>,
        d_y_all: &mut cudarc::driver::CudaSlice<f32>,
        d_moe_acc: &mut cudarc::driver::CudaSlice<f32>,
    ) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};

        let s = self.cap_stream.clone();
        let k = &self.kernels;
        let hidden = self.hidden;
        let route_count = idx.len();
        let nff = moe.n_ff_exp;
        let two_nff = 2 * nff;
        let gu_fmt = moe.gate_up_exps.format;
        let gu_blocks = hidden / 32;
        let down_blocks = nff / 32;
        let gu_row_bytes = hidden / gu_fmt.values_per_block() * gu_fmt.bytes_per_block();
        let down_row_bytes = down_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
        let expected_gu = two_nff * gu_row_bytes;
        let expected_down = hidden * down_row_bytes;
        let gate_up_bytes = moe.gate_up_exps.bytes();
        let down_bytes = moe.down_exps.bytes();
        // The page-locked tier reads its own records, so it needs the `.cghost`
        // handle and this layer's index INTO that artifact (which is not `l` on a
        // sharded range — the same distinction `get_many_cuda` callers make).
        let ghost_layer = moe
            .ghost
            .as_ref()
            .map(|g| (g.cache.file.as_ref(), g.layer_idx));

        let protected = idx
            .iter()
            .map(|&expert| (l as u16, expert as u16))
            .collect::<std::collections::HashSet<_>>();
        // Same guarantee in the tier's own key space, which indexes the `.cghost`
        // layer rather than the model layer.
        let tier_protected = ghost_layer
            .map(|(_, layer_idx)| {
                idx.iter()
                    .map(|&expert| (layer_idx as u16, expert as u16))
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();
        let mut hit_slots = Vec::<i32>::with_capacity(route_count);
        let mut hit_routes = Vec::<i32>::with_capacity(route_count);
        let mut miss_slots = Vec::<i32>::with_capacity(route_count);
        let mut miss_routes = Vec::<i32>::with_capacity(route_count);
        let mut route_scales = Vec::<f32>::with_capacity(route_count);
        let pinned_transfers = std::env::var("CAMELID_GEMMA4_CUDA_PINNED_EXPERTS")
            .ok()
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no" | "disabled"
                )
            })
            .unwrap_or(true);

        // Resolve the complete route before any GEMV. `protected` prevents a miss
        // from recycling a slot that another selected resident expert still needs.
        for (route, &expert) in idx.iter().enumerate() {
            route_scales.push(moe.down_exps_scale[expert] * (probs[expert] / wsum));
            let key = (l as u16, expert as u16);
            if sser.borrow_mut().touch(key) {
                let slot = sser
                    .borrow()
                    .entries
                    .get(&key)
                    .expect("touched routed expert remains resident")
                    .slot;
                sser.borrow_mut().hits += 1;
                hit_slots.push(slot as i32);
                hit_routes.push(route as i32);
                continue;
            }

            // The page-locked tier owns its own record bytes, so skip resolving a
            // host slice entirely when it is active — reading `moe.*_exps` here
            // would read the sparse shadow's deliberately empty routed ranges.
            let tier_bytes = if self.host_tier.is_some() {
                (&[][..], &[][..])
            } else if let Some(records) = ghost_records.as_deref_mut() {
                let record = records.get(&expert).ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Ghost-MoE CUDA lost routed expert layer={l} expert={expert}"
                    ))
                })?;
                let (gu_dtype, gu_bytes) = record.gate_up()?;
                let (down_dtype, down_bytes) = record.down()?;
                // The record's gate_up must carry the SAME wire format the
                // model bound (Q4_0, or Q2_K on the mixed artifact); down is
                // always Q4_0. A mismatch means the .cghost does not pair with
                // this GGUF — refuse rather than feed the wrong kernel.
                let gu_expected = match gu_fmt {
                    WireFormat::Q2K => GgufTensorType::Q2K,
                    _ => GgufTensorType::Q4_0,
                };
                if gu_dtype != gu_expected || down_dtype != GgufTensorType::Q4_0 {
                    return Err(BackendError::UnsupportedGguf(format!(
                        "Ghost-MoE CUDA expert record format mismatch; layer={l} expert={expert} gate_up={gu_dtype:?} (expected {gu_expected:?}) down={down_dtype:?} (expected Q4_0)"
                    )));
                }
                (gu_bytes, down_bytes)
            } else {
                let gu_off = expert * expected_gu;
                let down_off = expert * expected_down;
                (
                    &gate_up_bytes[gu_off..gu_off + expected_gu],
                    &down_bytes[down_off..down_off + expected_down],
                )
            };
            let (gu_host, down_host) = tier_bytes;
            if self.host_tier.is_none()
                && (gu_host.len() != expected_gu || down_host.len() != expected_down)
            {
                return Err(BackendError::InvalidModelMetadata(format!(
                    "Gemma 4 CUDA expert record length mismatch at layer={l} expert={expert}: gate_up={} expected={expected_gu}, down={} expected={expected_down}",
                    gu_host.len(),
                    down_host.len()
                )));
            }

            let mut cache = sser.borrow_mut();
            let slot = cache.slot_for_miss_excluding(&protected).ok_or_else(|| {
                BackendError::InvalidModelMetadata(format!(
                    "Gemma 4 CUDA expert cache capacity {} cannot pin a top-{route_count} route",
                    cache.capacity
                ))
            })?;
            let gu_start = slot * cache.gate_up_stride;
            let down_start = slot * cache.down_stride;
            if let Some(tier) = self.host_tier.as_ref() {
                // Tier 2 hit: the record is already page-locked, so the copy
                // engine DMAs straight out of it and the CPU never touches the
                // bytes. That is the whole point of the tier — staging through
                // the ring instead would add a ~3.19 MiB host memcpy (~0.4 ms)
                // to every miss, ~30 ms/token at the measured 73 misses.
                let (ghost_file, ghost_layer_idx) = ghost_layer.ok_or_else(|| {
                    BackendError::InvalidModelMetadata(
                        "Ghost-MoE CUDA host tier requires the paired .cghost artifact".into(),
                    )
                })?;
                let mut tier = tier.borrow_mut();
                let tier_slot =
                    tier.ensure_resident(ghost_file, ghost_layer_idx, expert, &tier_protected)?;
                let (gu_pinned, down_pinned) = tier.views(tier_slot);
                if gu_pinned.len() != expected_gu || down_pinned.len() != expected_down {
                    return Err(BackendError::InvalidModelMetadata(format!(
                        "Ghost-MoE CUDA host tier record mismatch at layer={l} expert={expert}: gate_up={} expected={expected_gu}, down={} expected={expected_down}",
                        gu_pinned.len(),
                        down_pinned.len()
                    )));
                }
                let SserCache {
                    gate_up_arena,
                    down_arena,
                    ..
                } = &mut *cache;
                self.expert_copy_stream
                    .memcpy_htod(
                        gu_pinned,
                        &mut gate_up_arena.slice_mut(gu_start..gu_start + expected_gu),
                    )
                    .map_err(cu)?;
                self.expert_copy_stream
                    .memcpy_htod(
                        down_pinned,
                        &mut down_arena.slice_mut(down_start..down_start + expected_down),
                    )
                    .map_err(cu)?;
            } else if pinned_transfers {
                let transfer_index = miss_slots.len();
                let SserCache {
                    gate_up_arena,
                    down_arena,
                    transfer_slots,
                    ..
                } = &mut *cache;
                let transfer_slot_count = transfer_slots.len();
                let transfer = transfer_slots.get_mut(transfer_index).ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!(
                        "Gemma 4 CUDA transfer ring has {} slots for a top-{route_count} route",
                        transfer_slot_count
                    ))
                })?;
                transfer
                    .gate_up
                    .as_mut_slice()
                    .map_err(cu)?
                    .copy_from_slice(gu_host);
                transfer
                    .down
                    .as_mut_slice()
                    .map_err(cu)?
                    .copy_from_slice(down_host);
                self.expert_copy_stream
                    .memcpy_htod(
                        &transfer.gate_up,
                        &mut gate_up_arena.slice_mut(gu_start..gu_start + expected_gu),
                    )
                    .map_err(cu)?;
                self.expert_copy_stream
                    .memcpy_htod(
                        &transfer.down,
                        &mut down_arena.slice_mut(down_start..down_start + expected_down),
                    )
                    .map_err(cu)?;
            } else {
                self.expert_copy_stream
                    .memcpy_htod(
                        gu_host,
                        &mut cache
                            .gate_up_arena
                            .slice_mut(gu_start..gu_start + expected_gu),
                    )
                    .map_err(cu)?;
                self.expert_copy_stream
                    .memcpy_htod(
                        down_host,
                        &mut cache
                            .down_arena
                            .slice_mut(down_start..down_start + expected_down),
                    )
                    .map_err(cu)?;
            }
            cache.misses += 1;
            cache.clock += 1;
            let stamp = cache.clock;
            cache.entries.insert(
                key,
                SserExpertDev {
                    slot,
                    last_used: stamp,
                },
            );
            miss_slots.push(slot as i32);
            miss_routes.push(route as i32);
        }

        let copy_done = if miss_slots.is_empty() {
            None
        } else {
            Some(self.expert_copy_stream.record_event(None).map_err(cu)?)
        };

        let run_group = |slots: &[i32],
                         routes: &[i32],
                         d_gate_up: &mut cudarc::driver::CudaSlice<f32>,
                         d_geglu_q: &mut cudarc::driver::CudaSlice<i8>,
                         d_geglu_s: &mut cudarc::driver::CudaSlice<f32>,
                         d_y_all: &mut cudarc::driver::CudaSlice<f32>|
         -> Result<()> {
            if slots.is_empty() {
                return Ok(());
            }
            let d_slots = s.clone_htod(slots).map_err(cu)?;
            let d_routes = s.clone_htod(routes).map_err(cu)?;
            let experts_i = slots.len() as i32;
            let block = 256u32;
            let warps = block / 32;

            let launch_q4 = |weights: &cudarc::driver::CudaSlice<u8>,
                             stride: usize,
                             in_s: &cudarc::driver::CudaSlice<f32>,
                             in_q: &cudarc::driver::CudaSlice<i8>,
                             rows: usize,
                             blocks: usize,
                             out: &mut cudarc::driver::CudaSlice<f32>,
                             batched_input: i32|
             -> Result<()> {
                let cfg = LaunchConfig {
                    grid_dim: ((rows as u32).div_ceil(warps), slots.len() as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: blocks as u32 * 32
                        + blocks as u32 * 4
                        + warps * blocks as u32 * 4,
                };
                let stride_u64 = stride as u64;
                let rows_i = rows as i32;
                let blocks_i = blocks as i32;
                let mut builder = s.launch_builder(&k.q4_0_gemv_routed);
                builder
                    .arg(in_s)
                    .arg(in_q)
                    .arg(weights)
                    .arg(&d_slots)
                    .arg(&d_routes)
                    .arg(&stride_u64)
                    .arg(&rows_i)
                    .arg(&blocks_i)
                    .arg(out)
                    .arg(&experts_i)
                    .arg(&batched_input);
                unsafe { builder.launch(cfg) }.map_err(cu)?;
                Ok(())
            };

            {
                let cache = sser.borrow();
                match gu_fmt {
                    // Mixed artifact: Q2_K gate_up dots the shared Q8_K
                    // activation staged in d_in_s/d_in_q (hidden/256 f32 +
                    // hidden i8). The kernel's integer core and ordered fold
                    // are verbatim from the certified dense q2k_gemv.
                    WireFormat::Q2K => {
                        let n_sb = hidden / 256;
                        let cfg = LaunchConfig {
                            grid_dim: ((two_nff as u32).div_ceil(warps), slots.len() as u32, 1),
                            block_dim: (block, 1, 1),
                            // staged Q8_K input: n_sb*256 i8 + n_sb*4 f32;
                            // per-warp scratch: n_sb * 2 i32 (isum, summs).
                            shared_mem_bytes: n_sb as u32 * 256
                                + n_sb as u32 * 4
                                + warps * n_sb as u32 * 2 * 4,
                        };
                        let stride_u64 = cache.gate_up_stride as u64;
                        let rows_i = two_nff as i32;
                        let n_sb_i = n_sb as i32;
                        let mut builder = s.launch_builder(&k.q2k_gemv_routed);
                        builder
                            .arg(d_in_s)
                            .arg(d_in_q)
                            .arg(&cache.gate_up_arena)
                            .arg(&d_slots)
                            .arg(&d_routes)
                            .arg(&stride_u64)
                            .arg(&rows_i)
                            .arg(&n_sb_i)
                            .arg(&mut *d_gate_up)
                            .arg(&experts_i);
                        unsafe { builder.launch(cfg) }.map_err(cu)?;
                    }
                    _ => launch_q4(
                        &cache.gate_up_arena,
                        cache.gate_up_stride,
                        d_in_s,
                        d_in_q,
                        two_nff,
                        gu_blocks,
                        d_gate_up,
                        0,
                    )?,
                }
            }
            {
                let cfg = LaunchConfig {
                    grid_dim: ((down_blocks as u32).div_ceil(block), slots.len() as u32, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                };
                let nff_i = nff as i32;
                let blocks_i = down_blocks as i32;
                let mut builder = s.launch_builder(&k.geglu_quantize_routed);
                builder
                    .arg(d_gate_up)
                    .arg(&d_routes)
                    .arg(&mut *d_geglu_q)
                    .arg(&mut *d_geglu_s)
                    .arg(&nff_i)
                    .arg(&blocks_i)
                    .arg(&experts_i);
                unsafe { builder.launch(cfg) }.map_err(cu)?;
            }
            {
                let cache = sser.borrow();
                launch_q4(
                    &cache.down_arena,
                    cache.down_stride,
                    d_geglu_s,
                    d_geglu_q,
                    hidden,
                    down_blocks,
                    d_y_all,
                    1,
                )?;
            }
            Ok(())
        };

        let hit_started = std::time::Instant::now();
        run_group(
            &hit_slots,
            &hit_routes,
            d_gate_up,
            d_geglu_q,
            d_geglu_s,
            d_y_all,
        )?;
        if std::env::var_os("CAMELID_SSER_PROFILE").is_some() {
            SSER_PROF_HIT_NS.fetch_add(
                hit_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        if let Some(event) = copy_done.as_ref() {
            s.wait(event).map_err(cu)?;
        }
        let miss_started = std::time::Instant::now();
        run_group(
            &miss_slots,
            &miss_routes,
            d_gate_up,
            d_geglu_q,
            d_geglu_s,
            d_y_all,
        )?;
        if std::env::var_os("CAMELID_SSER_PROFILE").is_some() {
            SSER_PROF_MISS_NS.fetch_add(
                miss_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }

        let d_route_scales = s.clone_htod(&route_scales).map_err(cu)?;
        let cfg = LaunchConfig {
            grid_dim: ((hidden as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_i = hidden as i32;
        let experts_i = route_count as i32;
        let mut builder = s.launch_builder(&k.moe_weighted_sum_routed);
        builder
            .arg(d_y_all)
            .arg(&d_route_scales)
            .arg(d_moe_acc)
            .arg(&hidden_i)
            .arg(&experts_i);
        unsafe { builder.launch(cfg) }.map_err(cu)?;
        Ok(())
    }

    /// SSER (M2/M3/M4) sparse-expert branch of the MoE FFN. Runs the router on the
    /// CPU (tiny), then every selected expert's two GEMVs on the GPU — cached in VRAM
    /// (hit) or uploaded+promoted (miss) — accumulating each expert's weighted
    /// down-GEMV into router-positioned scratch. Hits run as one CUDA batch while
    /// a second stream stages misses; one ordered kernel returns the device
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
        // Ghost-MoE shadows intentionally contain zero holes for routed experts.
        // Resolve only SSER misses from the paired `.cghost`, batching positioned
        // reads in physical expert order before restoring router order.
        let routed_batch = std::env::var("CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS")
            .ok()
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no" | "disabled"
                )
            })
            .unwrap_or(true);
        // The serial diagnostic path launches the Q4_0 GEMV pipeline on
        // gate_up unconditionally; a Q2_K gate_up there would feed 84-byte
        // superblocks to an 18-byte-block kernel. Refuse typed rather than
        // compute garbage — the mixed artifact is batched-path only.
        if !routed_batch && moe.gate_up_exps.format != WireFormat::Q4_0 {
            return Err(BackendError::UnsupportedGguf(format!(
                "CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS=0 (serial diagnostic path) supports only \
                 Q4_0 experts; this artifact's gate_up is {:?}",
                moe.gate_up_exps.format
            )));
        }
        // The page-locked tier resolves its own records inside the ROUTED miss
        // path, so skip this pre-pass when both are active: mapping every miss
        // here as well would fault the same bytes in twice. The serial
        // diagnostic path (`CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS=0`) has no tier
        // branch and still reads from these records — without them it would fall
        // through to `moe.*_exps`, which on a Ghost-MoE shadow is a deliberately
        // empty routed range, i.e. silently wrong output rather than a failure.
        let skip_records = routed_batch && self.host_tier.is_some();
        let mut ghost_records = if let (Some(ghost), false) = (moe.ghost.as_ref(), skip_records) {
            let missing = {
                let cache = sser.borrow();
                idx.iter()
                    .copied()
                    .filter(|&expert| !cache.entries.contains_key(&(l as u16, expert as u16)))
                    .collect::<Vec<_>>()
            };
            let loaded = ghost.cache.get_many_cuda(ghost.layer_idx, &missing)?;
            Some(
                missing
                    .into_iter()
                    .zip(loaded)
                    .collect::<std::collections::HashMap<_, _>>(),
            )
        } else {
            None
        };
        if prof {
            SSER_PROF_ROUTER_NS.fetch_add(
                tp1.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let tp2 = std::time::Instant::now();

        // --- Expert branch: quantize the shared input once (CPU), upload once. ---
        let cur_moe = rms_norm(attn_out, Some(&moe.pre_norm_2), eps);
        let two_nff = 2 * moe.n_ff_exp;
        let nff = moe.n_ff_exp;
        let gu_fmt = moe.gate_up_exps.format;
        let gu_blocks = hidden / 32; // gate_up in_dim = hidden, in Q8_0 blocks
        let down_blocks = nff / 32; // down in_dim = n_ff_exp
        let gu_row_bytes = hidden / gu_fmt.values_per_block() * gu_fmt.bytes_per_block();
        let down_row_bytes = down_blocks * crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;

        // Upload the shared expert input once — every selected expert dots
        // against the same activation. The quantization family follows the
        // gate_up wire format: Q4_0 gate_up dots a Q8_0 activation (32-value
        // blocks), Q2_K gate_up dots a Q8_K activation (256-value superblocks)
        // — the SAME host quantizer the CPU lane's `SharedActivation` uses, so
        // the CUDA GEMV consumes bit-identical staged inputs to the CPU oracle.
        // Both layouts fit the resident scratch: `in_s` holds hidden/32 f32
        // (Q8_K needs only hidden/256) and `in_q` holds exactly `hidden` i8.
        let (in_scales, in_quants): (Vec<f32>, Vec<i8>) = match gu_fmt {
            WireFormat::Q2K => {
                let xq = quantize_q8_k_blocks(&cur_moe);
                let scales: Vec<f32> = xq.iter().map(|b| b.d).collect();
                let mut quants = vec![0i8; hidden];
                for (b, blk) in xq.iter().enumerate() {
                    quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
                }
                (scales, quants)
            }
            _ => {
                let cur_moe_q = quantize_q8_0_blocks(&cur_moe);
                let scales: Vec<f32> = cur_moe_q.iter().map(|b| b.scale).collect();
                let mut quants = vec![0i8; gu_blocks * 32];
                for (b, blk) in cur_moe_q.iter().enumerate() {
                    quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
                }
                (scales, quants)
            }
        };
        // Per-layer scratch is RESIDENT, not allocated per call. Measured on the
        // tracked box with a 95.9%-hit route (i.e. transfers almost entirely
        // removed), the expert loop still spent 26.7 ms/token in
        // "prep + dtoh + sync" — 0.89 ms per layer with barely any bytes moving.
        // Allocating here cost 7 stream-ordered allocations plus their
        // event-tracking teardown, 30x per token; `clone_htod` allocates as well
        // as copying. Reusing fixed buffers turns ~360 driver round-trips per
        // token into ~150 plain copies. The buffers are sized at load to the
        // per-layer maxima, so every layer's view is a prefix of the same
        // allocation and the kernels see byte-identical inputs.
        let mut scratch = self
            .moe_scratch
            .as_ref()
            .expect("moe_layer_ffn_cached requires resident MoE scratch")
            .borrow_mut();
        let MoeScratchDev {
            in_s: d_in_s,
            in_q: d_in_q,
            gate_up: d_gate_up,
            geglu_q: d_geglu_q,
            geglu_s: d_geglu_s,
            y_all: d_y_all,
        } = &mut *scratch;
        // Exact-length views: the Q8_K layout fills only a prefix of the scale
        // scratch (hidden/256 f32 vs the Q8_0 layout's hidden/32).
        {
            let mut in_s_view = d_in_s.slice_mut(0..in_scales.len());
            s.memcpy_htod(&in_scales, &mut in_s_view).map_err(cu)?;
            let mut in_q_view = d_in_q.slice_mut(0..in_quants.len());
            s.memcpy_htod(&in_quants, &mut in_q_view).map_err(cu)?;
        }
        // M3/M4 on-device MoE accumulator: every selected expert (hit OR uploaded-miss)
        // folds its weighted down-GEMV output straight into this device buffer (one
        // scaled_axpy launch each). In M4 the buffer is RETURNED to the caller and
        // composed with the dense branch on-device — no per-layer dtoh at all.
        let mut d_moe_acc = s.alloc_zeros::<f32>(hidden).map_err(cu)?;

        let gate_up_bytes = moe.gate_up_exps.bytes();
        let down_bytes = moe.down_exps.bytes();

        if !routed_batch {
            // Allocate fallback-only scratch lazily so the routed hot path pays
            // neither the device allocations nor their event-tracking teardown.
            let mut d_geglu = unsafe { s.alloc::<f32>(nff) }.map_err(cu)?;
            let mut d_y = unsafe { s.alloc::<f32>(hidden) }.map_err(cu)?;
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
                    let expected_gu = two_nff * gu_row_bytes;
                    let expected_down = hidden * down_row_bytes;
                    // Expert kernels and uploads share `cap_stream`, so overwriting an
                    // LRU arena slot is ordered after its preceding GEMVs without a host
                    // sync. The arenas were allocated while event tracking was disabled.
                    let upload = |gu_host: &[u8], down_host: &[u8]| -> Result<usize> {
                        let mut cache = sser.borrow_mut();
                        if gu_host.len() != cache.gate_up_stride
                            || down_host.len() != cache.down_stride
                        {
                            return Err(BackendError::InvalidModelMetadata(format!(
                            "Gemma 4 CUDA expert arena geometry changed at layer={l} expert={e}: gate_up={} expected={}, down={} expected={}",
                            gu_host.len(),
                            cache.gate_up_stride,
                            down_host.len(),
                            cache.down_stride,
                        )));
                        }
                        let slot = cache.slot_for_miss();
                        let gu_start = slot * cache.gate_up_stride;
                        let down_start = slot * cache.down_stride;
                        s.memcpy_htod(
                            gu_host,
                            &mut cache
                                .gate_up_arena
                                .slice_mut(gu_start..gu_start + gu_host.len()),
                        )
                        .map_err(cu)?;
                        s.memcpy_htod(
                            down_host,
                            &mut cache
                                .down_arena
                                .slice_mut(down_start..down_start + down_host.len()),
                        )
                        .map_err(cu)?;
                        Ok(slot)
                    };
                    let slot = if let Some(records) = ghost_records.as_mut() {
                        // A route that was resident when `missing` was captured can
                        // be evicted by an earlier miss in this same ordered top-k.
                        // Recover only that exact race from `.cghost`; processing
                        // order and weighted accumulation remain router-identical.
                        if let std::collections::hash_map::Entry::Vacant(slot) = records.entry(e) {
                            let ghost =
                                moe.ghost.as_ref().expect("Ghost records require Ghost MoE");
                            let record = ghost
                            .cache
                            .get_many_cuda(ghost.layer_idx, &[e])?
                            .into_iter()
                            .next()
                            .ok_or_else(|| {
                                BackendError::InvalidModelMetadata(format!(
                                    "Ghost-MoE CUDA could not recover evicted routed expert layer={l} expert={e}"
                                ))
                            })?;
                            slot.insert(record);
                        }
                        let record = records.get(&e).ok_or_else(|| {
                            BackendError::InvalidModelMetadata(format!(
                                "Ghost-MoE CUDA lost routed expert layer={l} expert={e}"
                            ))
                        })?;
                        let (gu_dtype, gu_bytes) = record.gate_up()?;
                        let (down_dtype, down_bytes) = record.down()?;
                        if gu_dtype != GgufTensorType::Q4_0 || down_dtype != GgufTensorType::Q4_0 {
                            return Err(BackendError::UnsupportedGguf(format!(
                            "Ghost-MoE CUDA requires Q4_0 expert records; layer={l} expert={e} gate_up={:?} down={:?}",
                            gu_dtype, down_dtype
                        )));
                        }
                        if gu_bytes.len() != expected_gu || down_bytes.len() != expected_down {
                            return Err(BackendError::InvalidModelMetadata(format!(
                            "Ghost-MoE CUDA expert record length mismatch at layer={l} expert={e}: gate_up={} expected={expected_gu}, down={} expected={expected_down}",
                            gu_bytes.len(),
                            down_bytes.len()
                        )));
                        }
                        upload(gu_bytes, down_bytes)?
                    } else {
                        let gu_off = e * expected_gu;
                        let down_off = e * expected_down;
                        upload(
                            &gate_up_bytes[gu_off..gu_off + expected_gu],
                            &down_bytes[down_off..down_off + expected_down],
                        )?
                    };
                    let mut c = sser.borrow_mut();
                    c.clock += 1;
                    let stamp = c.clock;
                    c.entries.insert(
                        key,
                        SserExpertDev {
                            slot,
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
                    let gu_start = ent.slot * c.gate_up_stride;
                    let down_start = ent.slot * c.down_stride;
                    let gu_dev = c.gate_up_arena.slice(gu_start..gu_start + c.gate_up_stride);
                    let down_dev = c.down_arena.slice(down_start..down_start + c.down_stride);
                    // gate‖up: two_nff rows, gu_blocks blocks/row.
                    crate::cuda_resident::launch_q4_0_gemv(
                        &s,
                        &k.q4_0_gemv,
                        d_in_s,
                        d_in_q,
                        &gu_dev,
                        two_nff,
                        gu_blocks,
                        d_gate_up,
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
                        d_geglu_q,
                        d_geglu_s,
                        down_blocks,
                    )
                    .map_err(cu)?;
                    // down: hidden rows, down_blocks blocks/row.
                    crate::cuda_resident::launch_q4_0_gemv(
                        &s,
                        &k.q4_0_gemv,
                        d_geglu_s,
                        d_geglu_q,
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
        } else {
            self.moe_layer_ffn_cached_routed(
                l,
                moe,
                sser,
                &idx,
                &probs,
                wsum,
                ghost_records.as_mut(),
                d_in_s,
                d_in_q,
                d_gate_up,
                d_geglu_q,
                d_geglu_s,
                d_y_all,
                &mut d_moe_acc,
            )?;
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
                    // Scatter K/V into this layer's cache at `position` (ring slot
                    // `position % capacity`; sliding layers keep only window+1 slots).
                    let kv_cap = gemma4_kv_capacity(p.window, self.max_positions);
                    let ck = self.cache_k[li].as_mut().expect("owning layer has K cache");
                    crate::cuda_resident::launch_kv_scatter(
                        &s,
                        &k.kv_scatter,
                        &self.d_k,
                        ck,
                        &self.d_position,
                        kv_heads,
                        hd,
                        kv_cap,
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
                        kv_cap,
                    )
                    .map_err(cu)?;
                }

                // Attention against the source layer's cache (sliding window or full causal).
                let src = p.kv_source_layer;
                let window = p.window.map(|w| w as i32).unwrap_or(0);
                // The source cache's position capacity (== this layer's: KV sharing is
                // same-type, enforced by the layer_plan test). Sliding caches ring on
                // window+1 slots; shared memory still spans the full context because
                // the kernel's scores[] is indexed by absolute position.
                let src_cap = gemma4_kv_capacity(self.plan[src].window, self.max_positions);
                {
                    let ck = self.cache_k[src].as_ref().expect("KV source has K cache");
                    let cv = self.cache_v[src].as_ref().expect("KV source has V cache");
                    let cfg = LaunchConfig {
                        grid_dim: (heads as u32, 1, 1),
                        block_dim: (hd as u32, 1, 1),
                        shared_mem_bytes: ((2 * hd + self.max_positions) as u32) * 4,
                    };
                    let (nh, nkv, hdi, mp) =
                        (heads as i32, kv_heads as i32, hd as i32, src_cap as i32);
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
                    let ffn_out = self.cpu.moe_layer_ffn(li, &attn_out_host)?;
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
            // Sliding-layer caches are rings of window+1 positions: writing position x
            // reclaims position x-(window+1)'s slot, so of the previous request only
            // the last window+1 positions still exist. Resuming at `start = p.min(n-1)`
            // reads sliding keys [start+1-window, start]; the oldest survivor is
            // cached_len-(window+1), so reuse is sound iff start + 2 >= cached_len
            // (pure extension, or regenerating the final token). A deeper rewind
            // (edited history / shortened prompt) re-prefills from zero — same output,
            // just no TTFT shortcut. While the cached sequence is still within
            // window+1 positions nothing has been overwritten and any p is fine.
            let win = self.cpu.g.sliding_window as usize;
            if self.cached_tokens.len() > win + 1 && p.min(n - 1) + 2 < self.cached_tokens.len() {
                p = 0;
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

    pub fn generate_greedy_cancellable<C: FnMut() -> bool>(
        &mut self,
        prompt: &str,
        max_new: usize,
        should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        self.generate_greedy_streaming_cancellable(prompt, max_new, |_| {}, should_cancel)
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

    pub fn generate_greedy_streaming_cancellable<F: FnMut(&str), C: FnMut() -> bool>(
        &mut self,
        prompt: &str,
        max_new: usize,
        mut on_delta: F,
        mut should_cancel: C,
    ) -> Result<Gemma4GenerationOutcome> {
        // The shared runtime GPU switch remains live after model load. When it
        // is disabled (or deterministic mode is active), use the already-loaded
        // CPU/Ghost runtime and discard CUDA prefix state so health and execution
        // agree without requiring a model reload.
        if !crate::cuda::gpu_accel_enabled() || crate::inference::deterministic_mode_enabled() {
            self.cached_tokens.clear();
            return self.cpu.generate_greedy_streaming_cancellable(
                prompt,
                max_new,
                on_delta,
                should_cancel,
            );
        }
        if should_cancel() {
            self.cached_tokens.clear();
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        }
        let prompt_tokens = self.cpu.tokenizer.encode(prompt, true, true)?;
        let eot = gemma4_stop_token_ids(&self.cpu.tokenizer);
        let mut logits = self.prefill_reusing_cache(&prompt_tokens)?;
        if should_cancel() {
            self.cached_tokens.clear();
            return Ok(Gemma4GenerationOutcome::Cancelled {
                generated_tokens: 0,
            });
        }
        let mut generated = Vec::new();
        let mut prev_text = String::new();
        let decode_end = (prompt_tokens.len() + max_new).min(self.max_positions);
        for pos in prompt_tokens.len()..decode_end {
            if should_cancel() {
                self.cached_tokens.clear();
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
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
            if should_cancel() {
                self.cached_tokens.clear();
                return Ok(Gemma4GenerationOutcome::Cancelled {
                    generated_tokens: generated.len(),
                });
            }
            logits = self.forward_token(next, pos, true)?;
        }
        self.cached_tokens.extend_from_slice(&generated);
        Ok(Gemma4GenerationOutcome::Complete {
            text: prev_text,
            token_ids: generated,
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_parity_tests {
    use super::*;

    /// Deterministic filler for the head-upload layout tests (no rand dep).
    fn lcg_bytes(n: usize, mut seed: u32) -> Vec<u8> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect()
    }

    // Root-cause regression for the gemma4 Q4_0 mis-decode. The tied-head GEMVs do
    // NOT read the stock GGUF wire: `q4k_gemv` indexes a SWIZZLED quant region and
    // `q6k_gemv` indexes super-blocks at a 224-byte PADDED stride. The Q4_K and Q6_K
    // head arms used to `clone_htod` the raw wire, so a Q4_0-quantized gemma4 export
    // (whose token_embd is Q4_K) computed every logit from wrongly-paired nibbles —
    // fluent-looking nonsense, no error. `gemma4_head_upload` is now the single upload
    // path; this pins each lane's layout so raw passthrough cannot come back.
    //
    // Asserted with explicit index arithmetic rather than by calling the repack
    // helpers back, so the test still fails if a helper is changed to agree with a
    // broken caller.
    #[test]
    fn gemma4_head_upload_matches_each_lane_gemv_layout() {
        // --- Q4_K: pure byte permutation of the quant region, header untouched. ---
        const Q4K_WIRE: usize = 144;
        let blocks = 5usize;
        let wire = lcg_bytes(blocks * Q4K_WIRE, 0x4b_4b_01);
        let up = gemma4_head_upload(HeadLane::Q4K, &wire);
        assert_eq!(up.len(), wire.len(), "the swizzle must not change size");
        assert!(
            up != wire,
            "raw passthrough is the defect: q4k_gemv reads swizzled quant bytes"
        );
        for b in 0..blocks {
            let (s, d) = (&wire[b * Q4K_WIRE..], &up[b * Q4K_WIRE..]);
            assert_eq!(
                &d[..16],
                &s[..16],
                "d/dmin/packed-scale header is untouched"
            );
            // The four stride-8 bytes an aux lane consumes must land contiguous.
            for g in 0..4 {
                for l in 0..8 {
                    for k in 0..4 {
                        assert_eq!(
                            d[16 + g * 32 + l * 4 + k],
                            s[16 + g * 32 + l + k * 8],
                            "q4k swizzle mismatch at block {b} group {g} lane {l} k {k}"
                        );
                    }
                }
            }
        }

        // --- Q6_K: 210-byte wire blocks padded to the 224-byte stride the kernel
        // indexes. Uploading raw here under-sizes the buffer AND mis-addresses every
        // block past the first, so this length check is the load-bearing assertion.
        const Q6K_WIRE: usize = 210;
        const Q6K_PADDED: usize = 224;
        let wire = lcg_bytes(blocks * Q6K_WIRE, 0x6b_6b_02);
        let up = gemma4_head_upload(HeadLane::Q6K, &wire);
        assert_eq!(
            up.len(),
            blocks * Q6K_PADDED,
            "q6k_gemv strides super-blocks by 224 B, not the 210 B wire"
        );
        for b in 0..blocks {
            assert_eq!(
                &up[b * Q6K_PADDED..b * Q6K_PADDED + Q6K_WIRE],
                &wire[b * Q6K_WIRE..(b + 1) * Q6K_WIRE],
                "q6k payload block {b} must survive the pad verbatim"
            );
        }

        // --- Q8_0: the SoA split q8_gemv reads (all quants, then all f16 scales). ---
        const Q8_WIRE: usize = 34;
        let wire = lcg_bytes(blocks * Q8_WIRE, 0x08_08_03);
        let up = gemma4_head_upload(HeadLane::Q8_0, &wire);
        assert_eq!(up.len(), wire.len());
        for b in 0..blocks {
            let src = &wire[b * Q8_WIRE..(b + 1) * Q8_WIRE];
            assert_eq!(
                &up[b * 32..b * 32 + 32],
                &src[2..34],
                "q8 quants are SoA-first"
            );
            assert_eq!(
                &up[blocks * 32 + b * 2..blocks * 32 + b * 2 + 2],
                &src[0..2],
                "q8 f16 scales trail the quant plane"
            );
        }
    }

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

#[cfg(test)]
mod scalar_prefill_head_tests {
    use super::drive_scalar_prefill;

    #[test]
    fn projects_only_the_final_prompt_position() {
        let calls = std::cell::RefCell::new(Vec::new());
        let logits = drive_scalar_prefill(&[10, 20, 30, 40], |token, pos, project_head| {
            calls.borrow_mut().push((token, pos, project_head));
            Ok::<Option<u32>, std::convert::Infallible>(project_head.then_some(token + 1))
        })
        .unwrap();

        assert_eq!(logits, 41);
        assert_eq!(
            calls.into_inner(),
            vec![
                (10, 0, false),
                (20, 1, false),
                (30, 2, false),
                (40, 3, true),
            ]
        );
    }
}

#[cfg(test)]
mod ghost_hybrid_prefill_plan_tests {
    use super::{select_ghost_prefill_plan, GhostPrefillPlan};

    #[test]
    fn multi_token_common_prefill_defaults_to_hybrid_but_has_a_scalar_kill_switch() {
        assert_eq!(
            select_ghost_prefill_plan(true, true, 16, 31, Some(4096)),
            GhostPrefillPlan::HybridChunk
        );
        assert_eq!(
            select_ghost_prefill_plan(true, false, 16, 31, Some(4096)),
            GhostPrefillPlan::ScalarMetal
        );
    }

    #[test]
    fn one_token_prompt_stays_scalar_metal_and_over_capacity_stays_cpu_from_zero() {
        assert_eq!(
            select_ghost_prefill_plan(true, true, 1, 8, Some(4096)),
            GhostPrefillPlan::ScalarMetal
        );
        assert_eq!(
            select_ghost_prefill_plan(true, true, 4000, 4127, Some(4096)),
            GhostPrefillPlan::CpuChunk
        );
        assert_eq!(
            select_ghost_prefill_plan(false, true, 1, 4127, Some(4096)),
            GhostPrefillPlan::ScalarCpu
        );
    }
}

#[cfg(test)]
mod ghost_moe_wire_tests {
    use super::*;
    use crate::ghost::{
        CghostGroup, CghostIndex, CghostLayout, CghostTensor, CGHOST_ALIGN, CGHOST_MAGIC,
    };
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn ghost_metal_dispatch_requires_runtime_gpu_and_non_deterministic_mode() {
        assert!(ghost_metal_acceleration_allowed(false, true));
        assert!(!ghost_metal_acceleration_allowed(false, false));
        assert!(!ghost_metal_acceleration_allowed(true, true));
        assert!(!ghost_metal_acceleration_allowed(true, false));
    }

    #[test]
    fn cryptographic_ghost_identity_survives_a_gguf_rename() {
        assert!(ghost_source_filename_admitted(
            true,
            "original.gguf",
            Some("renamed.gguf")
        ));
        assert!(ghost_source_filename_admitted(
            false,
            "original.gguf",
            Some("original.gguf")
        ));
        assert!(!ghost_source_filename_admitted(
            false,
            "original.gguf",
            Some("renamed.gguf")
        ));
        assert!(ghost_source_filename_admitted(
            false,
            "",
            Some("anything.gguf")
        ));
    }

    fn cache_fixture(
        block_count: usize,
        expert_count: usize,
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.cghost");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(CGHOST_MAGIC).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let mut cursor = (CGHOST_MAGIC.len() + 8) as u64;
        let mut groups = Vec::new();
        for layer in 0..block_count {
            for expert in 0..expert_count {
                let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
                file.write_all(&vec![0; (aligned - cursor) as usize])
                    .unwrap();
                cursor = aligned;
                let marker = (layer * expert_count + expert) as u8;
                file.write_all(&[marker; 4]).unwrap();
                groups.push(CghostGroup {
                    id: format!("blk.{layer}.exp.{expert}"),
                    tensors: vec![
                        CghostTensor {
                            name: "gate".into(),
                            role: "gate_up_exps".into(),
                            dtype: GgufTensorType::Q4_0,
                            dims: vec![32, 2],
                            offset: cursor,
                            len: 2,
                        },
                        CghostTensor {
                            name: "down".into(),
                            role: "down_exps".into(),
                            dtype: GgufTensorType::Q4_0,
                            dims: vec![32, 2],
                            offset: cursor + 2,
                            len: 2,
                        },
                    ],
                    source_sample_sha256: None,
                });
                cursor += 4;
            }
        }
        let index = CghostIndex {
            version: 2,
            layout: CghostLayout::MoeExperts,
            source_model: "cache.gguf".into(),
            block_count,
            tied_output: true,
            expert_count: Some(expert_count),
            expert_used_count: Some(1),
            source_identity: None,
            groups,
        };
        let index_bytes = serde_json::to_vec(&index).unwrap();
        file.write_all(&index_bytes).unwrap();
        file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
            .unwrap();
        file.write_all(&cursor.to_le_bytes()).unwrap();
        drop(file);
        (dir, path)
    }

    #[test]
    fn owned_expert_wire_matches_the_existing_q4_row_kernel_bitwise() {
        let rows = 2usize;
        let row_bytes = crate::inference::Q4_0_WIRE_BYTES_PER_BLOCK;
        let mut wire = vec![0u8; rows * row_bytes];
        for row in 0..rows {
            let base = row * row_bytes;
            let scale = crate::tensor::f32_to_f16_bits(0.125 + row as f32 * 0.0625);
            wire[base..base + 2].copy_from_slice(&scale.to_le_bytes());
            for (i, byte) in wire[base + 2..base + row_bytes].iter_mut().enumerate() {
                *byte = ((i * 17 + row * 29) & 0xff) as u8;
            }
        }

        // Prefix/suffix sentinels prove the owned view honors its range instead
        // of assuming the expert tensor begins at allocation offset zero.
        let prefix = 7usize;
        let mut allocation = vec![0xa5; prefix];
        allocation.extend_from_slice(&wire);
        allocation.extend_from_slice(&[0x5a; 11]);
        let allocation: Arc<[u8]> = allocation.into();
        let weight = WireQuant::from_owned_wire(
            allocation,
            prefix..prefix + wire.len(),
            GgufTensorType::Q4_0,
            &[32, rows as u64],
            "test ghost expert",
        )
        .unwrap();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.31).sin()).collect();
        let activation = SharedActivation::new(&x);
        let got = weight.matvec_rows_proj(0, rows, &activation);
        let xq = activation.q8_0();
        for row in 0..rows {
            let expected = q4_0_wire_row_dot(&wire[row * row_bytes..(row + 1) * row_bytes], xq);
            assert_eq!(got[row].to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn owned_expert_wire_rejects_a_truncated_view() {
        let bytes: Arc<[u8]> = vec![0u8; 17].into();
        let err = match WireQuant::from_owned_wire(
            bytes,
            0..17,
            GgufTensorType::Q4_0,
            &[32, 1],
            "truncated ghost expert",
        ) {
            Ok(_) => panic!("one Q4_0 row requires 18 bytes"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("expected 18"));
    }

    #[test]
    fn global_expert_cache_never_exceeds_its_byte_budget() {
        let (_dir, path) = cache_fixture(1, 2);

        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 4);
        cache.get(0, 0).unwrap();
        cache.get(0, 0).unwrap(); // hit
        cache.get(0, 1).unwrap(); // must evict expert 0
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_experts, 1);
        assert!(stats.resident_bytes <= stats.budget_bytes);
    }

    #[test]
    fn resident_peek_supplies_slot_bytes_without_changing_cache_stats() {
        let (_dir, path) = cache_fixture(1, 2);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 4);
        cache.get(0, 1).unwrap();
        let before = cache.stats();
        let resident = cache
            .peek_resident(0, 1)
            .expect("resident expert should be available as a slot-fill source");
        let (bytes, _) = resident.tensor_backing(&resident.gate_up);
        assert_eq!(bytes[0], 1);
        assert_eq!(cache.stats(), before);
        assert!(cache.peek_resident(0, 0).is_none());
        assert_eq!(cache.stats(), before);
    }

    #[test]
    fn batch_read_restores_router_order_after_sorted_parallel_io() {
        let (_dir, path) = cache_fixture(1, 3);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 12);
        let routed = cache.get_many(0, &[2, 0, 1]).unwrap();
        let markers = routed
            .iter()
            .map(|expert| {
                let (bytes, range) = expert.tensor_backing(&expert.gate_up);
                bytes[range.start]
            })
            .collect::<Vec<_>>();
        assert_eq!(markers, vec![2, 0, 1]);
        assert_eq!(cache.stats().misses, 3);
    }

    #[test]
    fn batch_read_warms_an_over_budget_segment_by_route_frequency() {
        let (_dir, path) = cache_fixture(1, 3);
        // The segment holds two experts. Expert 0 is physically read first but
        // requested most often, so admission must move it behind colder rows.
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get_many(0, &[0, 2, 0, 1, 0]).unwrap();
        {
            let state = cache.state.lock().unwrap();
            assert!(state.entries.contains_key(&(0, 0)));
            assert_eq!(state.entries.get(&(0, 0)).unwrap().frequency, 3);
        }
        cache.get(0, 0).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 5);
        assert_eq!(stats.resident_experts, 2);
    }

    #[test]
    fn layer_segments_prevent_cross_layer_scan_pollution() {
        let (_dir, path) = cache_fixture(2, 2);
        // Two four-byte segments: inserting a second layer-0 expert may evict
        // layer 0's old entry, but it must not bulldoze layer 1's resident hit.
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get(0, 0).unwrap();
        cache.get(1, 0).unwrap();
        cache.get(0, 1).unwrap();
        cache.get(1, 0).unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_experts, 2);
        assert!(stats.resident_bytes <= stats.budget_bytes);
    }

    #[test]
    fn segment_evicts_lfu_then_uses_lru_as_tie_break() {
        let (_dir, path) = cache_fixture(1, 3);
        let cache = GhostMoeExpertCache::new(Arc::new(GhostFile::open(&path).unwrap()), 8);
        cache.get(0, 0).unwrap();
        cache.get(0, 1).unwrap();
        cache.get(0, 0).unwrap(); // expert 0 frequency = 2
        cache.get(0, 2).unwrap(); // expert 1 is the LFU victim
        cache.get(0, 0).unwrap();

        let state = cache.state.lock().unwrap();
        assert!(state.entries.contains_key(&(0, 0)));
        assert!(!state.entries.contains_key(&(0, 1)));
        assert!(state.entries.contains_key(&(0, 2)));
        assert_eq!(state.hits, 2);
        assert_eq!(state.misses, 3);
        assert_eq!(state.evictions, 1);
    }

    #[test]
    fn metal_slot_plan_preserves_route_order_and_deduplicates_loads() {
        let mut directory = GhostMetalSlotDirectory::new(4);
        let plan = directory.plan(&[9, 2, 9, 4]).unwrap();
        assert_eq!(plan.route_slots, vec![0, 1, 0, 2]);
        assert_eq!(plan.hits, 1);
        assert_eq!(plan.evictions, 0);
        assert_eq!(
            plan.loads
                .iter()
                .map(|load| (load.slot, load.expert, load.frequency))
                .collect::<Vec<_>>(),
            vec![(0, 9, 2), (1, 2, 1), (2, 4, 1)]
        );
        for load in plan.loads {
            directory.commit_load(load);
        }

        let hits = directory.plan(&[4, 9]).unwrap();
        assert_eq!(hits.route_slots, vec![2, 0]);
        assert_eq!(hits.hits, 2);
        assert_eq!(hits.evictions, 0);
        assert!(hits.loads.is_empty());
    }

    #[test]
    fn metal_slot_count_config_defaults_and_clamps_to_safe_bounds() {
        assert_eq!(parse_ghost_metal_slots_per_layer(None), 16);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("invalid")), 16);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("0")), 8);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("8")), 8);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("24")), 24);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("32")), 32);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("96")), 96);
        assert_eq!(parse_ghost_metal_slots_per_layer(Some("4096")), 128);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_slot_stats_delta_tracks_churn_without_wraparound() {
        let before = GhostMetalSlotStats {
            route_lookups: 80,
            hits: 40,
            misses: 40,
            evictions: 24,
            direct_reads: 36,
            direct_read_bytes: 120_000,
            ..GhostMetalSlotStats::default()
        };
        let after = GhostMetalSlotStats {
            route_lookups: 96,
            hits: 50,
            misses: 46,
            evictions: 29,
            host_fills: 2,
            prewarm_copies: 3,
            direct_reads: 40,
            direct_read_bytes: 133_380,
            direct_read_failures: 1,
        };
        assert_eq!(
            after.saturating_delta(before),
            GhostMetalSlotStats {
                route_lookups: 16,
                hits: 10,
                misses: 6,
                evictions: 5,
                host_fills: 2,
                prewarm_copies: 3,
                direct_reads: 4,
                direct_read_bytes: 13_380,
                direct_read_failures: 1,
            }
        );
    }

    #[test]
    fn metal_slot_failed_read_never_publishes_partial_bytes() {
        let mut directory = GhostMetalSlotDirectory::new(2);
        let warm = directory.plan(&[1, 2]).unwrap();
        for load in warm.loads {
            directory.commit_load(load);
        }
        // Make expert 1 hotter so expert 2 is the deterministic victim.
        assert!(directory.plan(&[1]).unwrap().loads.is_empty());
        let failed = directory.plan(&[3]).unwrap();
        assert_eq!(failed.loads.len(), 1);
        assert_eq!(failed.loads[0].slot, 1);
        assert_eq!(failed.loads[0].expert, 3);
        // Deliberately do not commit: this models a failed positioned read.
        assert!(directory.entries[1].is_none());

        // Neither the evicted expert nor the failed replacement may hit. The
        // empty slot is safely reused and published only after commit.
        let retry = directory.plan(&[2]).unwrap();
        assert_eq!(retry.loads.len(), 1);
        assert_eq!(retry.loads[0].slot, 1);
        assert_eq!(retry.loads[0].expert, 2);
    }

    #[test]
    fn metal_slot_route_is_preflighted_before_any_eviction() {
        let mut directory = GhostMetalSlotDirectory::new(2);
        let warm = directory.plan(&[5, 6]).unwrap();
        for load in warm.loads {
            directory.commit_load(load);
        }
        let before = directory.entries.clone();
        let err = directory.plan(&[7, 8, 9]).unwrap_err();
        assert!(err.to_string().contains("3 distinct experts"));
        assert_eq!(directory.entries, before);
    }

    #[test]
    fn metal_prompt_prewarm_honors_configured_limit_and_preserves_route_evidence() {
        let broad_routes: Vec<usize> = (0..40).collect();
        assert_eq!(
            ghost_metal_prewarm_sequence(&broad_routes, 40, 8),
            (32..40).collect::<Vec<_>>()
        );
        assert_eq!(
            ghost_metal_prewarm_sequence(&broad_routes, 40, 32),
            (8..40).collect::<Vec<_>>()
        );
        let routed: Vec<usize> = (0..18).collect();
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 18, 16),
            (2..18).collect::<Vec<_>>(),
            "equal-frequency ties should retain the most recent experts"
        );

        let routed = vec![0, 1, 2, 0, 3, 1, 4];
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 5, 2),
            vec![0, 1, 0, 1],
            "filtered route must retain occurrence count and original order"
        );
        assert_eq!(
            ghost_metal_prewarm_sequence(&routed, 5, 1),
            vec![1, 1],
            "recency breaks equal-frequency ties"
        );
    }

    /// Opt-in production-row admission gate. Unlike the synthetic kernel
    /// fixtures, this proves the complete GGUF + expert-spliced `.cghost` load
    /// reaches an actually configured persistent common core. It intentionally
    /// stops before generation so admission regressions can be diagnosed
    /// without paying prompt/decode time.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_common_real_model_admits_when_fixture_is_configured() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost common admission: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost common admission: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost common admission: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let flag = |name: &str| {
            std::env::var(name).is_ok_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            })
        };
        if !flag("CAMELID_GEMMA4_GHOST_METAL_SLOTS") || !flag("CAMELID_GEMMA4_GHOST_METAL_COMMON") {
            eprintln!(
                "SKIP Ghost common admission: enable CAMELID_GEMMA4_GHOST_METAL_SLOTS=1 and CAMELID_GEMMA4_GHOST_METAL_COMMON=1"
            );
            return;
        }
        let slot_env = std::env::var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER").ok();
        let expected_slots = parse_ghost_metal_slots_per_layer(slot_env.as_deref());

        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 64, false)
            .expect("load production Ghost-MoE fixture");
        assert_eq!(runtime.layers.len(), 30);
        assert!(
            runtime
                .layers
                .iter()
                .all(|layer| layer.ple_output_scale.is_finite()),
            "all learned layer output scales must be finite"
        );
        assert!(
            runtime
                .layers
                .iter()
                .any(|layer| layer.ple_output_scale.to_bits() != 1.0f32.to_bits()),
            "production fixture must exercise learned non-unit layer scales"
        );
        assert!(
            runtime.ghost_common_metal_active(),
            "production Ghost-MoE fixture did not configure the persistent Metal common core"
        );
        let slot_guard = runtime
            .metal_q4_experts
            .lock()
            .expect("Ghost Metal runtime mutex poisoned");
        assert_eq!(
            slot_guard
                .as_ref()
                .expect("persistent slot lane is absent")
                .slots_per_layer(),
            expected_slots,
            "production slot slab did not honor the configured capacity"
        );
        drop(slot_guard);
        eprintln!(
            "[gemma4-ghost-common-test] ACTIVE: production GGUF/cghost pair admitted with 30 learned layer-output scales and {expected_slots} slots/layer"
        );
    }

    /// Opt-in real 26B tied-head gate. It compares the established CPU Q6_K
    /// projection with the no-copy Metal head on the exact local GGUF and prints
    /// cold/warm timings for performance diagnosis. No fixture is required in CI.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_metal_q6k_head_matches_cpu_argmax_when_fixture_is_configured() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost Metal head parity: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal head parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal head parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        if std::env::var("CAMELID_GEMMA4_GHOST_METAL_HEAD")
            .is_ok_and(|value| value == "0" || value.eq_ignore_ascii_case("false"))
        {
            eprintln!("SKIP Ghost Metal head parity: Metal head explicitly disabled");
            return;
        }
        let mut runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 64, false)
            .expect("load Ghost-MoE fixture");
        let head = runtime
            .metal_q6k_head
            .as_ref()
            .expect("real 26B Q6_K Ghost fixture should bind the Metal head");
        let hidden_size = runtime.hidden_size();
        // A real tied embedding row supplies a representative, deterministic
        // activation without paying for a full 30-layer Ghost forward.
        let hidden = runtime
            .token_embd
            .dequantize_elements(100 * hidden_size, hidden_size)
            .expect("gather representative hidden row");

        let cpu_started = std::time::Instant::now();
        let cpu = runtime.project_logits_cpu(&hidden);
        let cpu_elapsed = cpu_started.elapsed();
        let cold_started = std::time::Instant::now();
        let metal = head.forward(&hidden).expect("cold Metal head forward");
        let cold_elapsed = cold_started.elapsed();
        let warm_started = std::time::Instant::now();
        let metal_warm = head.forward(&hidden).expect("warm Metal head forward");
        let warm_elapsed = warm_started.elapsed();
        assert_eq!(metal, metal_warm, "reused Metal head must be deterministic");

        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap()
        };
        let max_abs = cpu
            .iter()
            .zip(&metal)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "Ghost Q6_K head: CPU={:.3}s Metal-cold={:.3}s Metal-warm={:.3}s max_abs={max_abs:.6}",
            cpu_elapsed.as_secs_f64(),
            cold_elapsed.as_secs_f64(),
            warm_elapsed.as_secs_f64(),
        );
        assert_eq!(
            argmax(&metal),
            argmax(&cpu),
            "Metal Q6_K head changed the real model's next-token argmax"
        );
        assert_eq!(
            metal, cpu,
            "strict ordered Metal Q6_K + CPU soft-cap must match the CPU head bit-for-bit"
        );

        // Natural hidden/token gate: run one real Ghost step twice with fresh KV,
        // first through Metal and then with only the head removed. Both paths use
        // the exact same decoder/runtime and must choose the same greedy token.
        let natural_started = std::time::Instant::now();
        let (metal_text, metal_ids) = runtime
            .generate_greedy("Hello", 1)
            .expect("natural Metal-head generation");
        let natural_metal_elapsed = natural_started.elapsed();
        let saved_head = runtime
            .metal_q6k_head
            .take()
            .expect("fixture bound a Metal head above");
        let natural_started = std::time::Instant::now();
        let (cpu_text, cpu_ids) = runtime
            .generate_greedy("Hello", 1)
            .expect("natural CPU-head generation");
        let natural_cpu_elapsed = natural_started.elapsed();
        runtime.metal_q6k_head = Some(saved_head);
        eprintln!(
            "Ghost natural one-token: Metal-head={:.3}s CPU-head={:.3}s ids={metal_ids:?}",
            natural_metal_elapsed.as_secs_f64(),
            natural_cpu_elapsed.as_secs_f64(),
        );
        assert_eq!(metal_ids, cpu_ids, "natural greedy token changed");
        assert_eq!(metal_text, cpu_text, "natural decoded token changed");
    }

    /// Opt-in real 26B parity and timing gate for the persistent Q4_0 expert
    /// slots. Run with the two fixture paths plus
    /// `CAMELID_GEMMA4_GHOST_METAL_SLOTS=1`. The first half isolates one natural
    /// MoE layer and requires every final FFN bit to match the CPU Ghost oracle;
    /// the second emits two tokens so token #2 is predicted by a full 30-layer
    /// decode through the persistent slot lane.
    #[cfg(target_os = "macos")]
    #[test]
    fn ghost_metal_q4_slots_match_real_layer_and_natural_decode() {
        if !crate::metal::detect_metal_device().available {
            eprintln!("SKIP Ghost Metal Q4 slots parity: no Metal device");
            return;
        }
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 256, false)
            .expect("load real Ghost-MoE fixture");
        if !runtime.ghost_metal_q4_is_enabled() {
            eprintln!("SKIP Ghost Metal Q4 slots parity: set CAMELID_GEMMA4_GHOST_METAL_SLOTS=1");
            return;
        }

        let hidden_size = runtime.hidden_size();
        let hidden = runtime
            .token_embd
            .dequantize_elements(100 * hidden_size, hidden_size)
            .expect("gather representative real hidden row");
        let metal_started = std::time::Instant::now();
        let metal_layer = runtime
            .moe_layer_ffn(0, &hidden)
            .expect("real layer-0 persistent-slot FFN");
        let metal_layer_elapsed = metal_started.elapsed();
        let saved_lane = runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned")
            .take()
            .expect("fixture bound persistent Metal expert slots");
        let cpu_started = std::time::Instant::now();
        let cpu_layer = runtime
            .moe_layer_ffn(0, &hidden)
            .expect("real layer-0 CPU Ghost FFN");
        let cpu_layer_elapsed = cpu_started.elapsed();
        *runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned") = Some(saved_lane);
        assert_eq!(
            metal_layer, cpu_layer,
            "persistent Q4_0 parity lane changed a real layer FFN bit"
        );

        let metal_started = std::time::Instant::now();
        let (metal_text, metal_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("natural persistent-slot generation");
        let metal_cold_decode_elapsed = metal_started.elapsed();
        // The first pass must allocate/fault and fill up to eight slots in every
        // layer. Repeat the identical decode before removing the lane so the
        // receipt separates that one-time cost from steady-state slot hits.
        let metal_started = std::time::Instant::now();
        let (metal_warm_text, metal_warm_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("warm persistent-slot generation");
        let metal_warm_decode_elapsed = metal_started.elapsed();
        assert_eq!(metal_warm_ids, metal_ids, "warm Metal ids changed");
        assert_eq!(metal_warm_text, metal_text, "warm Metal text changed");
        let saved_lane = runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned")
            .take()
            .expect("persistent Metal expert slots remained active");
        let cpu_started = std::time::Instant::now();
        let (cpu_text, cpu_ids) = runtime
            .generate_greedy("Hello", 2)
            .expect("natural CPU Ghost generation");
        let cpu_decode_elapsed = cpu_started.elapsed();
        *runtime
            .metal_q4_experts
            .lock()
            .expect("Metal expert mutex poisoned") = Some(saved_lane);

        eprintln!(
            "Ghost Q4 slots real parity: layer Metal={:.3}s CPU={:.3}s; two-token Metal-cold={:.3}s Metal-warm={:.3}s CPU={:.3}s ids={metal_ids:?}",
            metal_layer_elapsed.as_secs_f64(),
            cpu_layer_elapsed.as_secs_f64(),
            metal_cold_decode_elapsed.as_secs_f64(),
            metal_warm_decode_elapsed.as_secs_f64(),
            cpu_decode_elapsed.as_secs_f64(),
        );
        assert_eq!(metal_ids, cpu_ids, "persistent slots changed greedy ids");
        assert_eq!(
            metal_text, cpu_text,
            "persistent slots changed decoded text"
        );
    }

    /// Opt-in real-model parity gate for the layer-major Ghost prefill. The
    /// normal test suite has no 26B fixture, so this skips unless both paths are
    /// supplied. It compares every prompt-position logit bit, not just argmax,
    /// which also proves the expert-major compute schedule restores each row's
    /// original route-rank accumulation order.
    #[test]
    fn ghost_chunk_prefill_matches_scalar_step_bitwise_when_fixture_is_configured() {
        let Some(model) = std::env::var_os("CAMELID_GEMMA4_GGUF").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost chunk parity: set CAMELID_GEMMA4_GGUF");
            return;
        };
        let Some(cghost) =
            std::env::var_os("CAMELID_GEMMA4_GHOST_CGHOST").map(std::path::PathBuf::from)
        else {
            eprintln!("SKIP Ghost chunk parity: set CAMELID_GEMMA4_GHOST_CGHOST");
            return;
        };
        let runtime = Gemma4Runtime::load_ghost_moe(&model, &cghost, 1024, false)
            .expect("load Ghost-MoE fixture");
        assert!(runtime.supports_chunk_forward());
        let tokens = runtime
            .tokenizer
            .encode("Hello from chunked Ghost MoE.", true, true)
            .expect("tokenize parity prompt");

        let (mut scalar_k, mut scalar_v) = runtime.empty_kv_caches();
        let mut scalar_rows = Vec::with_capacity(tokens.len());
        for (pos, &token) in tokens.iter().enumerate() {
            scalar_rows.push(
                runtime
                    .step(token, pos, &mut scalar_k, &mut scalar_v)
                    .expect("scalar prompt step"),
            );
        }

        let (mut chunk_k, mut chunk_v) = runtime.empty_kv_caches();
        let chunk_rows = runtime
            .step_chunk(&tokens, 0, &mut chunk_k, &mut chunk_v)
            .expect("layer-major prompt chunk");
        assert_eq!(scalar_rows.len(), chunk_rows.len());
        for (position, (scalar, chunk)) in scalar_rows.iter().zip(&chunk_rows).enumerate() {
            assert_eq!(scalar.len(), chunk.len());
            for (token_id, (&a, &b)) in scalar.iter().zip(chunk).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "Ghost chunk diverged at position {position}, logit {token_id}"
                );
            }
        }

        let (mut final_k, mut final_v) = runtime.empty_kv_caches();
        let final_only = runtime
            .step_chunk_with_head(&tokens, 0, &mut final_k, &mut final_v, false, None)
            .expect("final-head-only prompt chunk");
        assert_eq!(final_only.len(), 1);
        for (token_id, (&a, &b)) in scalar_rows
            .last()
            .unwrap()
            .iter()
            .zip(&final_only[0])
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "final-head-only chunk diverged at logit {token_id}"
            );
        }

        let assert_kv_bits = |label: &str,
                              expected_k: &Gemma4KvCache,
                              expected_v: &Gemma4KvCache,
                              actual_k: &Gemma4KvCache,
                              actual_v: &Gemma4KvCache| {
            assert_eq!(expected_k.len(), actual_k.len(), "{label} K layers");
            assert_eq!(expected_v.len(), actual_v.len(), "{label} V layers");
            for layer in 0..expected_k.len() {
                assert_eq!(
                    expected_k[layer].len(),
                    actual_k[layer].len(),
                    "{label} K positions at layer {layer}"
                );
                assert_eq!(
                    expected_v[layer].len(),
                    actual_v[layer].len(),
                    "{label} V positions at layer {layer}"
                );
                for position in 0..expected_k[layer].len() {
                    for (index, (&expected, &actual)) in expected_k[layer][position]
                        .iter()
                        .zip(&actual_k[layer][position])
                        .enumerate()
                    {
                        assert_eq!(
                            expected.to_bits(),
                            actual.to_bits(),
                            "{label} K layer={layer} position={position} index={index}"
                        );
                    }
                    for (index, (&expected, &actual)) in expected_v[layer][position]
                        .iter()
                        .zip(&actual_v[layer][position])
                        .enumerate()
                    {
                        assert_eq!(
                            expected.to_bits(),
                            actual.to_bits(),
                            "{label} V layer={layer} position={position} index={index}"
                        );
                    }
                }
            }
        };
        assert_kv_bits("chunk-vs-scalar", &scalar_k, &scalar_v, &chunk_k, &chunk_v);
        assert_kv_bits(
            "final-only-vs-scalar",
            &scalar_k,
            &scalar_v,
            &final_k,
            &final_v,
        );
    }
}
