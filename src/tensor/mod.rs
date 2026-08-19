use std::{
    cell::Cell,
    collections::HashMap,
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    time::Instant,
};

const RETAIN_Q8_BLOCKS_ENV: &str = "CAMELID_RETAIN_Q8_0_BLOCKS";
const Q8_FILE_CACHE_BYTES_ENV: &str = "CAMELID_Q8_0_FILE_CACHE_BYTES";
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_VALUES: usize = 32;
// Keep lazy Q8_0 file reads memory-safe by default. The bounded chunk cache is an
// explicit diagnostic/performance probe until long-context prefill has row-specific evidence.
const DEFAULT_Q8_FILE_CACHE_BYTES: usize = 0;

use rayon::prelude::*;
use serde::Serialize;

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType},
    platform_fs::read_exact_at,
    BackendError, Result,
};

pub mod kv_quant;
pub mod wire_dequant;

#[cfg(target_os = "macos")]
pub(crate) fn disable_file_cache_best_effort(file: &File) {
    use std::{os::fd::AsRawFd, os::raw::c_int};

    const F_RDAHEAD: c_int = 45;
    const F_NOCACHE: c_int = 48;
    unsafe extern "C" {
        fn fcntl(fd: c_int, cmd: c_int, ...) -> c_int;
    }

    // Best-effort only: the lazy Q8 path streams model bytes repeatedly, and on macOS the
    // default file cache/readahead can consume free pages even when Camelid RSS stays low.
    // Keep both calls non-fatal: older kernels/filesystems may reject one knob but honor the other.
    let _ = unsafe { fcntl(file.as_raw_fd(), F_RDAHEAD, 0) };
    let _ = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1) };
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn disable_file_cache_best_effort(_file: &File) {}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn from_gguf_dims(dims: &[u64]) -> Result<Self> {
        let dims = dims
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    BackendError::InvalidTensorData(format!("dimension {dim} does not fit usize"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { dims })
    }

    pub fn element_count(&self) -> Result<usize> {
        self.dims.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                BackendError::InvalidTensorData("tensor element count overflow".to_string())
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDType {
    F32,
}

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Block {
    pub scale: f32,
    pub quants: [i8; 32],
}

#[allow(dead_code)]
pub(crate) fn q8_0_blocks_as_bytes(blocks: &[Q8_0Block]) -> &[u8] {
    debug_assert_eq!(std::mem::size_of::<Q8_0Block>(), 36);
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr().cast::<u8>(), std::mem::size_of_val(blocks))
    }
}

/// Cheap view into an immutable resident Q8_0 block allocation.
///
/// MoE expert packs are much larger than ordinary linears. Expert selection
/// uses this range view so each token can borrow one expert without cloning
/// hundreds of MiB of quantized blocks.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Q8_0SharedBlocks {
    // Keep the Vec allocation itself behind Arc. `Arc<[T]>::from(Vec<T>)`
    // allocates and moves into an Arc slice, which briefly doubles multi-GiB
    // MoE resident storage during promotion. Arc<Vec<T>> retains the loader's
    // existing allocation and still gives every expert view immutable sharing.
    pub(crate) blocks: std::sync::Arc<Vec<Q8_0Block>>,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl Q8_0SharedBlocks {
    pub(crate) fn as_slice(&self) -> &[Q8_0Block] {
        &self.blocks[self.start..self.start + self.len]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q8_0PackedRows4Interleave {
    I4,
    I8,
}

impl Q8_0PackedRows4Interleave {
    pub fn block_len(self) -> usize {
        match self {
            Self::I4 => 4,
            Self::I8 => 8,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::I4 => "4x4",
            Self::I8 => "4x8",
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0PackedRows4Block {
    pub scales: [f32; 4],
    pub quants: [i8; 128],
}

/// Q4_0 wire geometry (GGUF on-disk): 32 values per block, stored as a 2-byte
/// little-endian f16 scale followed by 16 nibble bytes (byte `j` low nibble is
/// value `j`, high nibble is value `j+16`; both unsigned with a -8 bias).
const Q4_0_WIRE_BYTES_PER_BLOCK: usize = 18;

/// Eight Q4_0 weight rows interleaved for the AVX2 8-row GEMV, mirroring
/// llama.cpp's `block_q4_0x8` (`d[8]` f16 scales + `qs[128]` nibble bytes).
///
/// Layout of `qs` (matches `ggml_gemv_q4_0_8x8_q8_0_generic`): for
/// `k in 0..2`, `row in 0..8`, `i in 0..8`, byte `qs[k*64 + row*8 + i]` holds
/// weight-nibbles for the 8 columns of one activation half — its LOW nibble is
/// the weight for activation index `k*8 + i`, its HIGH nibble the weight for
/// activation index `k*8 + i + 16` (both `-8`-biased). This is exactly the
/// scalar [`crate::inference::q4_0_wire_row_dot_scalar`] contract, re-laned so 8
/// rows can be dotted against one activation block in a single SIMD pass.
#[repr(C, align(32))]
#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0PackedRows8Block {
    pub scales: [f32; 8],
    pub qs: [u8; 128],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0PackedRows8 {
    pub rows: usize,
    pub blocks_per_row: usize,
    /// `(rows / 8) * blocks_per_row` interleaved blocks, row-group major then
    /// block major (group g, block b at index `g * blocks_per_row + b`).
    pub blocks: Vec<Q4_0PackedRows8Block>,
}

impl Q4_0PackedRows8 {
    /// Interleave `rows` (multiple of 8) Q4_0 weight rows read straight from the
    /// GGUF wire bytes into the 8-row layout. `q4_0_bytes` is the tensor's full
    /// wire slice, row-major, `blocks_per_row` Q4_0 blocks per row.
    pub fn from_q4_0_bytes(rows: usize, blocks_per_row: usize, q4_0_bytes: &[u8]) -> Result<Self> {
        let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::InvalidTensorData("q4_0 packed rows8 block count overflow".to_string())
        })?;
        let expected_bytes = expected_blocks
            .checked_mul(Q4_0_WIRE_BYTES_PER_BLOCK)
            .ok_or_else(|| {
                BackendError::InvalidTensorData("q4_0 packed rows8 byte count overflow".to_string())
            })?;
        if q4_0_bytes.len() != expected_bytes || !rows.is_multiple_of(8) {
            return Err(BackendError::InvalidTensorData(format!(
                "q4_0 packed rows8 expected GGUF Q4_0 bytes for rows multiple of 8; rows={rows}, blocks_per_row={blocks_per_row}, got {} bytes, expected {expected_bytes}",
                q4_0_bytes.len()
            )));
        }

        let mut blocks = Vec::with_capacity((rows / 8) * blocks_per_row);
        for row_group in (0..rows).step_by(8) {
            for block_idx in 0..blocks_per_row {
                let mut scales = [0.0_f32; 8];
                let mut qs = [0_u8; 128];
                for (lane, scale) in scales.iter_mut().enumerate() {
                    let source_block = (row_group + lane) * blocks_per_row + block_idx;
                    let source_start = source_block * Q4_0_WIRE_BYTES_PER_BLOCK;
                    *scale = f16_bits_to_f32(u16::from_le_bytes([
                        q4_0_bytes[source_start],
                        q4_0_bytes[source_start + 1],
                    ]));
                }
                // Re-lane the 16 wire nibble bytes of each of the 8 rows.
                // Wire byte `j` (0..16): low nibble = value `j`, high nibble =
                // value `j+16`. Interleaved byte `qs[k*64 + lane*8 + i]` must
                // carry value `k*8 + i` in its low nibble and value `k*8+i+16`
                // in its high nibble. Since wire byte `j` already pairs `j` with
                // `j+16`, the mapping is a straight copy: k*8+i == j (i.e. j in
                // 0..16 splits as k = j/8, i = j%8), so qs[k*64+lane*8+i] equals
                // wire byte (k*8+i).
                for lane in 0..8 {
                    let source_block = (row_group + lane) * blocks_per_row + block_idx;
                    let source_start = source_block * Q4_0_WIRE_BYTES_PER_BLOCK + 2;
                    for k in 0..2 {
                        for i in 0..8 {
                            let j = k * 8 + i;
                            qs[k * 64 + lane * 8 + i] = q4_0_bytes[source_start + j];
                        }
                    }
                }
                blocks.push(Q4_0PackedRows8Block { scales, qs });
            }
        }

        Ok(Self {
            rows,
            blocks_per_row,
            blocks,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q4_0PackedRows8Block>()
    }
}

/// Unpack one Q4_K superblock's 12 packed kmask scale bytes into the eight
/// 6-bit scales and eight 6-bit mins — the exact KMASK1/2/3 recombination the
/// per-cell kernels perform (shared here so the 8-row repack and the owner
/// kernels provably use the same unpack).
pub fn q4_k_unpack_kmask_scales(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let utmp0 = u32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
    let utmp1 = u32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]);
    let utmp2 = u32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]);
    let mins8 = [
        utmp1 & KMASK1,
        ((utmp2 >> 4) & KMASK2) | (((utmp1 >> 6) & KMASK3) << 4),
    ];
    let scales_w = [
        utmp0 & KMASK1,
        (utmp2 & KMASK2) | (((utmp0 >> 6) & KMASK3) << 4),
    ];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for g in 0..8 {
        scales[g] = ((scales_w[g / 4] >> (8 * (g % 4))) & 0xff) as u8;
        mins[g] = ((mins8[g / 4] >> (8 * (g % 4))) & 0xff) as u8;
    }
    (scales, mins)
}

/// STAMPEDE Lane B v5 — one 8-row group × one 256-column Q4_K superblock,
/// repacked for the AVX-512 VNNI 8-row prefill GEMM. Everything weight-side
/// is pre-hoisted at pack time: f16 super-scales pre-widened to f32, kmask
/// scales/mins pre-unpacked and stored GROUP-major (`[group][row]` — one
/// 8-byte load per group builds the per-row scale vector), and the nibble
/// bytes re-laned at 4-byte-quad granularity (below). A straight byte
/// permutation of wire data — zero numeric transformation.
///
/// Nibble layout: `qs[c*256 + qc*32 + r*4 + b]` = wire nibble byte
/// `(row r, 32-byte chunk c, quad qc, byte b)`, so one 64-byte load yields
/// quad-columns `qc, qc+1` for all 8 rows (lanes 0..7 = rows at `qc`, lanes
/// 8..15 = rows at `qc+1`) — each i32 lane is exactly one dpbusd operand.
#[repr(C, align(64))]
#[derive(Debug, Clone, PartialEq)]
pub struct Q4KPackedRows8Block {
    pub d: [f32; 8],
    pub dmin: [f32; 8],
    pub scales: [[u8; 8]; 8],
    pub mins: [[u8; 8]; 8],
    pub qs: [u8; 1024],
}

/// Global bytes currently reserved by live 8-row repacks against the
/// `CAMELID_X86_KQUANT_REPACK8_BUDGET_MIB` budget. Reserved by the builder
/// path in `inference`, RELEASED by [`Q4KPackedRows8`]'s `Drop` — so model
/// unload/reload returns budget instead of eroding it (review finding,
/// 2026-07-09).
pub static KQUANT_REPACK8_RESERVED_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Deliberately NOT `Clone` (a struct clone would double-release the budget
/// reservation on drop); share via `Arc`.
#[derive(Debug)]
pub struct Q4KPackedRows8 {
    pub rows: usize,
    pub superblocks_per_row: usize,
    /// `(rows / 8) * superblocks_per_row` blocks, row-group major then
    /// superblock major (group g, superblock i at `g * superblocks_per_row + i`).
    pub blocks: Vec<Q4KPackedRows8Block>,
    /// Bytes this pack reserved against the global budget (0 for packs built
    /// outside the budgeted path, e.g. tests). Returned on drop.
    pub reservation_bytes: usize,
}

impl Drop for Q4KPackedRows8 {
    fn drop(&mut self) {
        if self.reservation_bytes > 0 {
            KQUANT_REPACK8_RESERVED_BYTES
                .fetch_sub(self.reservation_bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Q4KPackedRows8 {
    /// Interleave `rows` (multiple of 8) Q4_K weight rows read straight from
    /// the GGUF wire bytes (144 B/superblock, row-major).
    pub fn from_q4_k_wire(rows: usize, superblocks_per_row: usize, wire: &[u8]) -> Result<Self> {
        const WIRE: usize = 144;
        let expected_bytes = rows
            .checked_mul(superblocks_per_row)
            .and_then(|blocks| blocks.checked_mul(WIRE))
            .ok_or_else(|| {
                BackendError::InvalidTensorData("q4_k packed rows8 size overflow".to_string())
            })?;
        if wire.len() != expected_bytes || !rows.is_multiple_of(8) || superblocks_per_row == 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "q4_k packed rows8 expects wire bytes for rows multiple of 8; rows={rows}, \
                 superblocks_per_row={superblocks_per_row}, got {} bytes, expected {expected_bytes}",
                wire.len()
            )));
        }
        let mut blocks = Vec::with_capacity((rows / 8) * superblocks_per_row);
        for row_group in (0..rows).step_by(8) {
            for sb in 0..superblocks_per_row {
                let mut block = Q4KPackedRows8Block {
                    d: [0.0; 8],
                    dmin: [0.0; 8],
                    scales: [[0; 8]; 8],
                    mins: [[0; 8]; 8],
                    qs: [0; 1024],
                };
                for r in 0..8 {
                    let src = ((row_group + r) * superblocks_per_row + sb) * WIRE;
                    block.d[r] = f16_bits_to_f32(u16::from_le_bytes([wire[src], wire[src + 1]]));
                    block.dmin[r] =
                        f16_bits_to_f32(u16::from_le_bytes([wire[src + 2], wire[src + 3]]));
                    let (scales, mins) = q4_k_unpack_kmask_scales(&wire[src + 4..src + 16]);
                    for g in 0..8 {
                        block.scales[g][r] = scales[g];
                        block.mins[g][r] = mins[g];
                    }
                    let qs_src = src + 16;
                    for c in 0..4 {
                        for qc in 0..8 {
                            for b in 0..4 {
                                block.qs[c * 256 + qc * 32 + r * 4 + b] =
                                    wire[qs_src + c * 32 + qc * 4 + b];
                            }
                        }
                    }
                }
                blocks.push(block);
            }
        }
        Ok(Self {
            rows,
            superblocks_per_row,
            blocks,
            reservation_bytes: 0,
        })
    }

    pub fn byte_len(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q4KPackedRows8Block>()
    }
}

/// Lazily-built 8-row repack cell carried on `CpuTensor`. Clones SHARE the
/// cell (the pack is a derived cache of the immutable wire bytes), and it is
/// deliberately identity-neutral in `PartialEq` — two tensors with equal wire
/// bytes are equal whether or not either has built its pack yet.
#[derive(Debug, Clone, Default)]
pub struct Q4KRepack8Cell(
    pub std::sync::Arc<std::sync::OnceLock<Option<std::sync::Arc<Q4KPackedRows8>>>>,
);

impl PartialEq for Q4KRepack8Cell {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

#[repr(C, align(64))]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0AmxPackedBlock {
    pub scales: [f32; 16],
    pub quants: [i8; 512],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0VnniTile16 {
    pub quants: [i8; 512],
    pub scale_f16: [u16; 16],
    pub scale_f32: [f32; 16],
    pub comp: [i32; 16],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0VnniPacked {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub tiles: Vec<Q8_0VnniTile16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0PackedRows4 {
    pub rows: usize,
    pub blocks_per_row: usize,
    pub interleave: Q8_0PackedRows4Interleave,
    pub blocks: Vec<Q8_0PackedRows4Block>,
    pub amx_blocks: Option<Vec<Q8_0AmxPackedBlock>>,
    pub vnni_packed: Option<Q8_0VnniPacked>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Q8_0RuntimeStorage {
    PackedRows4(Q8_0PackedRows4),
}

impl Q8_0PackedRows4 {
    pub fn from_rows(
        rows: usize,
        blocks_per_row: usize,
        interleave: Q8_0PackedRows4Interleave,
        row_major_blocks: &[Q8_0Block],
    ) -> Result<Self> {
        let expected = rows.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::InvalidTensorData("q8_0 packed rows4 block count overflow".to_string())
        })?;
        if row_major_blocks.len() != expected || !rows.is_multiple_of(4) {
            return Err(BackendError::InvalidTensorData(format!(
                "q8_0 packed rows4 expected row-major blocks for rows multiple of 4; rows={rows}, blocks_per_row={blocks_per_row}, got {} blocks",
                row_major_blocks.len()
            )));
        }

        let block_len = interleave.block_len();
        let chunks = 32 / block_len;
        let mut blocks = Vec::with_capacity((rows / 4) * blocks_per_row);
        for row_group in (0..rows).step_by(4) {
            for block_idx in 0..blocks_per_row {
                let mut scales = [0.0_f32; 4];
                let mut quants = [0_i8; 128];
                for lane in 0..4 {
                    let source = &row_major_blocks[(row_group + lane) * blocks_per_row + block_idx];
                    scales[lane] = source.scale;
                }
                for chunk in 0..chunks {
                    for lane in 0..4 {
                        let source =
                            &row_major_blocks[(row_group + lane) * blocks_per_row + block_idx];
                        let src_start = chunk * block_len;
                        let dst_start = chunk * 4 * block_len + lane * block_len;
                        quants[dst_start..dst_start + block_len]
                            .copy_from_slice(&source.quants[src_start..src_start + block_len]);
                    }
                }
                blocks.push(Q8_0PackedRows4Block { scales, quants });
            }
        }

        Ok(Self {
            rows,
            blocks_per_row,
            interleave,
            amx_blocks: q8_0_pack_rows4_amx16_if_enabled(rows, blocks_per_row, interleave, &blocks),
            vnni_packed: None,
            blocks,
        })
    }

    pub fn from_q8_0_bytes(
        rows: usize,
        blocks_per_row: usize,
        interleave: Q8_0PackedRows4Interleave,
        q8_0_bytes: &[u8],
    ) -> Result<Self> {
        let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::InvalidTensorData("q8_0 packed rows4 block count overflow".to_string())
        })?;
        let expected_bytes = expected_blocks
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or_else(|| {
                BackendError::InvalidTensorData("q8_0 packed rows4 byte count overflow".to_string())
            })?;
        if q8_0_bytes.len() != expected_bytes || !rows.is_multiple_of(4) {
            return Err(BackendError::InvalidTensorData(format!(
                "q8_0 packed rows4 expected GGUF Q8_0 bytes for rows multiple of 4; rows={rows}, blocks_per_row={blocks_per_row}, got {} bytes, expected {expected_bytes}",
                q8_0_bytes.len()
            )));
        }

        let block_len = interleave.block_len();
        let chunks = Q8_0_BLOCK_VALUES / block_len;
        let mut blocks = Vec::with_capacity((rows / 4) * blocks_per_row);
        for row_group in (0..rows).step_by(4) {
            for block_idx in 0..blocks_per_row {
                let mut scales = [0.0_f32; 4];
                let mut quants = [0_i8; 128];
                for (lane, scale) in scales.iter_mut().enumerate() {
                    let source_block = (row_group + lane) * blocks_per_row + block_idx;
                    let source_start = source_block * Q8_0_BLOCK_BYTES;
                    *scale = f16_bits_to_f32(u16::from_le_bytes([
                        q8_0_bytes[source_start],
                        q8_0_bytes[source_start + 1],
                    ]));
                }
                for chunk in 0..chunks {
                    for lane in 0..4 {
                        let source_block = (row_group + lane) * blocks_per_row + block_idx;
                        let source_start = source_block * Q8_0_BLOCK_BYTES + 2;
                        let src_start = source_start + chunk * block_len;
                        let dst_start = chunk * 4 * block_len + lane * block_len;
                        for (dst, src) in quants[dst_start..dst_start + block_len]
                            .iter_mut()
                            .zip(&q8_0_bytes[src_start..src_start + block_len])
                        {
                            *dst = *src as i8;
                        }
                    }
                }
                blocks.push(Q8_0PackedRows4Block { scales, quants });
            }
        }

        Ok(Self {
            rows,
            blocks_per_row,
            interleave,
            amx_blocks: q8_0_pack_rows4_amx16_if_enabled(rows, blocks_per_row, interleave, &blocks),
            vnni_packed: q8_0_pack_vnni16_if_enabled(rows, blocks_per_row, q8_0_bytes)?,
            blocks,
        })
    }

    #[allow(dead_code)]
    pub fn to_q8_0_bytes(&self) -> Vec<u8> {
        let block_len = self.interleave.block_len();
        let chunks = Q8_0_BLOCK_VALUES / block_len;
        let total_blocks = self.rows * self.blocks_per_row;
        let mut q8_0_bytes = vec![0u8; total_blocks * Q8_0_BLOCK_BYTES];
        let mut packed_idx = 0;
        for row_group in (0..self.rows).step_by(4) {
            for block_idx in 0..self.blocks_per_row {
                let packed_blk = &self.blocks[packed_idx];
                packed_idx += 1;
                for (lane, &scale) in packed_blk.scales.iter().enumerate() {
                    let target_block = (row_group + lane) * self.blocks_per_row + block_idx;
                    let target_start = target_block * Q8_0_BLOCK_BYTES;
                    let f16_bits = f32_to_f16_bits(scale);
                    q8_0_bytes[target_start..target_start + 2]
                        .copy_from_slice(&f16_bits.to_le_bytes());
                }
                for chunk in 0..chunks {
                    for lane in 0..4 {
                        let target_block = (row_group + lane) * self.blocks_per_row + block_idx;
                        let target_start = target_block * Q8_0_BLOCK_BYTES + 2;
                        let dst_start = target_start + chunk * block_len;
                        let src_start = chunk * 4 * block_len + lane * block_len;
                        for (dst, src) in q8_0_bytes[dst_start..dst_start + block_len]
                            .iter_mut()
                            .zip(&packed_blk.quants[src_start..src_start + block_len])
                        {
                            *dst = *src as u8;
                        }
                    }
                }
            }
        }
        q8_0_bytes
    }

    pub fn byte_len(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q8_0PackedRows4Block>()
    }
}

fn q8_0_pack_vnni16_if_enabled(
    rows: usize,
    blocks_per_row: usize,
    q8_0_bytes: &[u8],
) -> Result<Option<Q8_0VnniPacked>> {
    if !x86_q8_vnni_decode_repack_enabled() || !rows.is_multiple_of(16) {
        return Ok(None);
    }
    let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
        BackendError::InvalidTensorData("q8_0 VNNI packed block count overflow".to_string())
    })?;
    let expected_bytes = expected_blocks
        .checked_mul(Q8_0_BLOCK_BYTES)
        .ok_or_else(|| {
            BackendError::InvalidTensorData("q8_0 VNNI packed byte count overflow".to_string())
        })?;
    if q8_0_bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "q8_0 VNNI pack expected {expected_bytes} bytes, got {}",
            q8_0_bytes.len()
        )));
    }

    let mut tiles = Vec::with_capacity((rows / 16) * blocks_per_row);
    for row_tile in 0..rows / 16 {
        for block_idx in 0..blocks_per_row {
            let mut tile = Q8_0VnniTile16 {
                quants: [0; 512],
                scale_f16: [0; 16],
                scale_f32: [0.0; 16],
                comp: [0; 16],
            };
            for n in 0..16 {
                let source_block = (row_tile * 16 + n) * blocks_per_row + block_idx;
                let source_start = source_block * Q8_0_BLOCK_BYTES;
                tile.scale_f16[n] =
                    u16::from_le_bytes([q8_0_bytes[source_start], q8_0_bytes[source_start + 1]]);
                tile.scale_f32[n] = f16_bits_to_f32(tile.scale_f16[n]);
                let qs = &q8_0_bytes[source_start + 2..source_start + Q8_0_BLOCK_BYTES];
                let sum = qs
                    .iter()
                    .fold(0_i32, |acc, value| acc + i32::from(*value as i8));
                tile.comp[n] = 128 * sum;
                for g in 0..8 {
                    for r in 0..4 {
                        tile.quants[g * 64 + n * 4 + r] = qs[g * 4 + r] as i8;
                    }
                }
            }
            tiles.push(tile);
        }
    }
    Ok(Some(Q8_0VnniPacked {
        rows,
        blocks_per_row,
        tiles,
    }))
}

fn q8_0_pack_rows4_amx16_if_enabled(
    rows: usize,
    blocks_per_row: usize,
    interleave: Q8_0PackedRows4Interleave,
    rows4_blocks: &[Q8_0PackedRows4Block],
) -> Option<Vec<Q8_0AmxPackedBlock>> {
    if !x86_q8_amx_repack_enabled()
        || interleave != Q8_0PackedRows4Interleave::I8
        || !rows.is_multiple_of(16)
    {
        return None;
    }
    let expected = (rows / 4).checked_mul(blocks_per_row)?;
    if rows4_blocks.len() != expected {
        return None;
    }

    let mut amx_blocks = Vec::with_capacity((rows / 16) * blocks_per_row);
    for output_tile in 0..rows / 16 {
        let rows4_tile_base = output_tile * 4;
        for block_idx in 0..blocks_per_row {
            let mut packed = Q8_0AmxPackedBlock {
                scales: [0.0; 16],
                quants: [0; 512],
            };
            for n in 0..16 {
                let rows4_group = rows4_tile_base + n / 4;
                let lane = n % 4;
                let source = &rows4_blocks[rows4_group * blocks_per_row + block_idx];
                packed.scales[n] = source.scales[lane];
                for k_group in 0..8 {
                    for k_lane in 0..4 {
                        let k = k_group * 4 + k_lane;
                        let chunk = k / 8;
                        let offset_in_chunk = k % 8;
                        let src_idx = chunk * 32 + lane * 8 + offset_in_chunk;
                        let dst_idx = k_group * 64 + n * 4 + k_lane;
                        packed.quants[dst_idx] = source.quants[src_idx];
                    }
                }
            }
            amx_blocks.push(packed);
        }
    }
    Some(amx_blocks)
}

fn q8_0_pack_trace_enabled() -> bool {
    env_flag_enabled("CAMELID_Q8_0_PACK_TRACE")
}

fn env_flag_enabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_flag_disabled(name: &str) -> bool {
    env::var(name)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("dequantized")
                || value.eq_ignore_ascii_case("f32")
        })
        .unwrap_or(false)
}

fn mac_q8_repack_enabled() -> bool {
    env_flag_enabled("CAMELID_MAC_Q8_REPACK")
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_q8_repack_enabled() -> bool {
    env_flag_enabled("CAMELID_X86_Q8_REPACK")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn x86_q8_repack_enabled() -> bool {
    false
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn x86_q8_amx_repack_enabled() -> bool {
    env_flag_enabled("CAMELID_X86_Q8_AMX_REPACK")
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn x86_q8_amx_repack_enabled() -> bool {
    false
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn x86_q8_vnni_decode_repack_enabled() -> bool {
    env_flag_enabled("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE")
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn x86_q8_vnni_decode_repack_enabled() -> bool {
    false
}

fn q8_repack_tensor_enabled(name: &str) -> bool {
    q8_repack_tensor_enabled_for_flags(name, mac_q8_repack_enabled(), x86_q8_repack_enabled())
}

fn q8_repack_tensor_enabled_for_flags(name: &str, mac_enabled: bool, x86_enabled: bool) -> bool {
    (mac_enabled && q8_repack_mac_tensor_enabled(name))
        || (x86_enabled && q8_repack_x86_tensor_enabled(name))
}

fn q8_repack_mac_tensor_enabled(name: &str) -> bool {
    (name.starts_with("blk.")
        && (q8_repack_attention_tensor_enabled(name)
            || name.ends_with(".ffn_gate.weight")
            || name.ends_with(".ffn_up.weight")
            || name.ends_with(".ffn_down.weight")))
        || name == "output.weight"
}

fn q8_repack_x86_tensor_enabled(name: &str) -> bool {
    (name.starts_with("blk.")
        && (q8_repack_attention_tensor_enabled(name)
            || name.ends_with(".ffn_gate.weight")
            || name.ends_with(".ffn_up.weight")
            || name.ends_with(".ffn_down.weight")))
        || name == "output.weight"
}

fn q8_repack_attention_tensor_enabled(name: &str) -> bool {
    name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
}

fn q8_repack_linear_shape(name: &str, shape: &TensorShape) -> Option<(usize, usize)> {
    if !q8_repack_tensor_enabled(name) || shape.dims.len() != 2 {
        return None;
    }
    let rows = shape.dims[0];
    let cols = shape.dims[1];
    if name == "output.weight" {
        // Llama output projection commonly arrives as [hidden, vocab], while
        // Camelid's token-major runtime consumes rows as [vocab, hidden]. If a
        // GGUF already stores [vocab, hidden], keep it as-is; otherwise pack the
        // backend-owned runtime storage in the directly consumable token-row view.
        if rows < cols {
            Some((cols, rows))
        } else {
            Some((rows, cols))
        }
    } else if name.ends_with(".ffn_gate.weight")
        || name.ends_with(".ffn_up.weight")
        || name.ends_with(".ffn_down.weight")
        || name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
    {
        // Llama FFN and attention projection descriptors are stored as [input, output],
        // while Camelid's hot linear path consumes rows as [output, input]. Pack
        // backend-owned runtime storage in output-row order so optimized consumers
        // do not have to fall back to row-major f32 data that runtime-packed tensors
        // intentionally do not retain.
        Some((cols, rows))
    } else {
        Some((rows, cols))
    }
}

fn q8_0_packed_rows4_enabled_for_tensor(name: &str, interleave: Q8_0PackedRows4Interleave) -> bool {
    let _ = name;
    match interleave {
        Q8_0PackedRows4Interleave::I4 => env_flag_enabled("CAMELID_Q8_0_PACKED_4X4_DOT"),
        Q8_0PackedRows4Interleave::I8 => env_flag_enabled("CAMELID_Q8_0_PACKED_4X8_DOT"),
    }
}

fn q8_0_packed_rows4_shape(shape: &TensorShape) -> Option<(usize, usize)> {
    if shape.dims.len() != 2 {
        return None;
    }
    let rows = shape.dims[0];
    let cols = shape.dims[1];
    (rows.is_multiple_of(4) && cols.is_multiple_of(Q8_0_BLOCK_VALUES))
        .then_some((rows, cols / Q8_0_BLOCK_VALUES))
}

fn q8_0_runtime_packed_rows4_shape_for_tensor(
    name: &str,
    shape: &TensorShape,
) -> Option<(usize, usize)> {
    if env_flag_disabled("CAMELID_Q8_0_BLOCK_DOT") {
        return None;
    }
    let (rows, cols) = q8_repack_linear_shape(name, shape)?;
    (rows.is_multiple_of(4) && cols.is_multiple_of(Q8_0_BLOCK_VALUES))
        .then_some((rows, cols / Q8_0_BLOCK_VALUES))
}

fn q8_0_runtime_packed_rows4_for_tensor(
    name: &str,
    shape: &TensorShape,
    q8_0_bytes: &[u8],
) -> Result<Option<Q8_0RuntimeStorage>> {
    let Some((rows, blocks_per_row)) = q8_0_runtime_packed_rows4_shape_for_tensor(name, shape)
    else {
        return Ok(None);
    };
    let started = Instant::now();
    let packed = Q8_0PackedRows4::from_q8_0_bytes(
        rows,
        blocks_per_row,
        Q8_0PackedRows4Interleave::I8,
        q8_0_bytes,
    )?;
    if q8_0_pack_trace_enabled() {
        let cols = blocks_per_row * Q8_0_BLOCK_VALUES;
        eprintln!(
            "camelid_q8_pack tensor={name} owner=runtime layout={} rows={rows} cols={cols} blocks={} bytes={} micros={}",
            Q8_0PackedRows4Interleave::I8.label(),
            packed.blocks.len(),
            packed.byte_len(),
            started.elapsed().as_micros()
        );
    }
    Ok(Some(Q8_0RuntimeStorage::PackedRows4(packed)))
}

fn q8_0_packed_rows4_for_shape(
    name: &str,
    shape: &TensorShape,
    q8_0_blocks: Option<&[Q8_0Block]>,
    interleave: Q8_0PackedRows4Interleave,
) -> Result<Option<Q8_0PackedRows4>> {
    if !q8_0_packed_rows4_enabled_for_tensor(name, interleave) {
        return Ok(None);
    }
    let Some(blocks) = q8_0_blocks else {
        return Ok(None);
    };
    let Some((rows, blocks_per_row)) = q8_0_packed_rows4_shape(shape) else {
        return Ok(None);
    };
    let started = Instant::now();
    let packed = Q8_0PackedRows4::from_rows(rows, blocks_per_row, interleave, blocks)?;
    if q8_0_pack_trace_enabled() {
        let cols = blocks_per_row * Q8_0_BLOCK_VALUES;
        eprintln!(
            "camelid_q8_pack tensor={name} layout={} rows={rows} cols={cols} blocks={} bytes={} micros={}",
            interleave.label(),
            packed.blocks.len(),
            packed.byte_len(),
            started.elapsed().as_micros()
        );
    }
    Ok(Some(packed))
}

fn q8_0_payload_bytes(
    name: &str,
    block_count: usize,
    blocks_per_payload: usize,
    payload_size: usize,
    role: &str,
) -> Result<u64> {
    let payload_count = block_count / blocks_per_payload;
    u64::try_from(payload_count)
        .ok()
        .and_then(|count| count.checked_mul(payload_size as u64))
        .ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {name} {role} resident-byte estimate overflow"
            ))
        })
}

fn q8_0_shape_and_block_count(name: &str, gguf_dimensions: &[u64]) -> Result<(TensorShape, usize)> {
    let shape = TensorShape::from_gguf_dims(gguf_dimensions)?;
    let elements = shape.element_count()?;
    if !elements.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} Q8_0 element count {elements} is not block aligned"
        )));
    }
    Ok((shape, elements / Q8_0_BLOCK_VALUES))
}

/// Exact heap payload retained by `load_q8_0_block_backed_linear_as` for a
/// rank-2 Q8_0 tensor under the current repack/sidecar flags.
pub(crate) fn q8_0_block_backed_linear_retained_bytes(
    name: &str,
    gguf_dimensions: &[u64],
) -> Result<u64> {
    let (shape, block_count) = q8_0_shape_and_block_count(name, gguf_dimensions)?;
    if let Some((rows, _)) = q8_0_runtime_packed_rows4_shape_for_tensor(name, &shape) {
        let mut total = q8_0_payload_bytes(
            name,
            block_count,
            4,
            std::mem::size_of::<Q8_0PackedRows4Block>(),
            "runtime PackedRows4",
        )?;
        if rows.is_multiple_of(16) && x86_q8_amx_repack_enabled() {
            total = total
                .checked_add(q8_0_payload_bytes(
                    name,
                    block_count,
                    16,
                    std::mem::size_of::<Q8_0AmxPackedBlock>(),
                    "AMX repack",
                )?)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "tensor {name} Q8_0 resident-byte estimate overflow"
                    ))
                })?;
        }
        if rows.is_multiple_of(16) && x86_q8_vnni_decode_repack_enabled() {
            total = total
                .checked_add(q8_0_payload_bytes(
                    name,
                    block_count,
                    16,
                    std::mem::size_of::<Q8_0VnniTile16>(),
                    "VNNI repack",
                )?)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "tensor {name} Q8_0 resident-byte estimate overflow"
                    ))
                })?;
        }
        return Ok(total);
    }
    q8_0_retained_blocks_with_sidecars_bytes_for_shape(name, &shape, block_count)
}

fn q8_0_retained_blocks_with_sidecars_bytes_for_shape(
    name: &str,
    shape: &TensorShape,
    block_count: usize,
) -> Result<u64> {
    let mut total = q8_0_payload_bytes(
        name,
        block_count,
        1,
        std::mem::size_of::<Q8_0Block>(),
        "expanded blocks",
    )?;
    let rows4_eligible = q8_0_packed_rows4_shape(shape).is_some();
    for interleave in [Q8_0PackedRows4Interleave::I4, Q8_0PackedRows4Interleave::I8] {
        if !rows4_eligible || !q8_0_packed_rows4_enabled_for_tensor(name, interleave) {
            continue;
        }
        total = total
            .checked_add(q8_0_payload_bytes(
                name,
                block_count,
                4,
                std::mem::size_of::<Q8_0PackedRows4Block>(),
                "PackedRows4 sidecar",
            )?)
            .ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {name} Q8_0 resident-byte estimate overflow"
                ))
            })?;
        if interleave == Q8_0PackedRows4Interleave::I8
            && shape.dims[0].is_multiple_of(16)
            && x86_q8_amx_repack_enabled()
        {
            total = total
                .checked_add(q8_0_payload_bytes(
                    name,
                    block_count,
                    16,
                    std::mem::size_of::<Q8_0AmxPackedBlock>(),
                    "AMX sidecar repack",
                )?)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(format!(
                        "tensor {name} Q8_0 resident-byte estimate overflow"
                    ))
                })?;
        }
    }
    Ok(total)
}

/// Exact retained Q8 block/sidecar payload attached by the generic f32 loader
/// when CAMELID_RETAIN_Q8_0_BLOCKS is enabled.
pub(crate) fn q8_0_retained_blocks_with_sidecars_bytes(
    name: &str,
    gguf_dimensions: &[u64],
) -> Result<u64> {
    let (shape, block_count) = q8_0_shape_and_block_count(name, gguf_dimensions)?;
    q8_0_retained_blocks_with_sidecars_bytes_for_shape(name, &shape, block_count)
}

#[derive(Debug, Clone)]
pub struct Q8_0FileBacking {
    pub path: PathBuf,
    pub absolute_offset: u64,
    pub num_blocks: usize,
    file_handle: Arc<OnceLock<Arc<File>>>,
}

impl Q8_0FileBacking {
    pub fn new(path: PathBuf, absolute_offset: u64, num_blocks: usize) -> Self {
        Self {
            path,
            absolute_offset,
            num_blocks,
            file_handle: Arc::new(OnceLock::new()),
        }
    }

    pub fn clone_with_offset_and_blocks(&self, absolute_offset: u64, num_blocks: usize) -> Self {
        Self {
            path: self.path.clone(),
            absolute_offset,
            num_blocks,
            file_handle: self.file_handle.clone(),
        }
    }

    pub fn file(&self) -> Result<Arc<File>> {
        if let Some(file) = self.file_handle.get() {
            return Ok(file.clone());
        }
        let file = File::open(&self.path).map_err(|source| BackendError::Io {
            path: self.path.clone(),
            source,
        })?;
        disable_file_cache_best_effort(&file);
        let file = Arc::new(file);
        if self.file_handle.set(file.clone()).is_err() {
            return Ok(self
                .file_handle
                .get()
                .expect("q8_0 file handle must exist after OnceLock set race")
                .clone());
        }
        Ok(file)
    }

    pub fn file_handle_cached(&self) -> bool {
        self.file_handle.get().is_some()
    }

    pub fn storage_bytes(&self) -> u64 {
        const Q8_0_BLOCK_BYTES: u64 = 34;
        (self.num_blocks as u64).saturating_mul(Q8_0_BLOCK_BYTES)
    }

    pub fn f32_materialization_bytes(&self) -> u64 {
        const Q8_0_BLOCK_VALUES: u64 = 32;
        (self.num_blocks as u64)
            .saturating_mul(Q8_0_BLOCK_VALUES)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
    }

    pub fn retained_block_bytes(&self) -> u64 {
        (self.num_blocks as u64).saturating_mul(std::mem::size_of::<Q8_0Block>() as u64)
    }

    pub(crate) fn read_exact_at_cached(&self, out: &mut [u8], offset: u64) -> Result<()> {
        self.read_exact_at_cached_impl(out, offset, None)
            .map(|_| ())
    }

    pub(crate) fn read_exact_at_cached_with_q8_0_scales(
        &self,
        out: &mut [u8],
        offset: u64,
        scales: &mut [f32],
    ) -> Result<bool> {
        let expected_len = scales.len().checked_mul(Q8_0_BLOCK_BYTES).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "q8_0 cached scale read byte length overflow".to_string(),
            )
        })?;
        if out.len() != expected_len {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "q8_0 cached scale read expected {} bytes for {} scales, got {}",
                expected_len,
                scales.len(),
                out.len()
            )));
        }

        let scale_status = self.read_exact_at_cached_impl(out, offset, Some(&mut *scales))?;
        if !scale_status.scales_ready() {
            decode_q8_0_scales_from_bytes(out, scales);
            q8_file_cache_store_decoded_scales(&self.path, offset, scales);
        }
        if let Some(blocks) = scale_status.decoded_scale_hit_blocks() {
            record_q8_file_cache_decoded_scale_reuse(blocks);
        }
        Ok(scale_status.decoded_scales_reused())
    }

    fn read_exact_at_cached_impl(
        &self,
        out: &mut [u8],
        offset: u64,
        mut cached_scales: Option<&mut [f32]>,
    ) -> Result<Q8FileReadScaleStatus> {
        if out.is_empty() {
            return Ok(if cached_scales.is_some_and(|scales| scales.is_empty()) {
                Q8FileReadScaleStatus::DecodedScalesReused {
                    cache_hit_blocks: 0,
                }
            } else {
                Q8FileReadScaleStatus::NoScales
            });
        }
        let relative_start = offset.checked_sub(self.absolute_offset).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(format!(
                "q8_0 file-backed read offset {offset} is before backing offset {}",
                self.absolute_offset
            ))
        })?;
        let relative_end = relative_start
            .checked_add(out.len() as u64)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "q8_0 file-backed read byte range overflow".to_string(),
                )
            })?;
        let storage_bytes = self.storage_bytes();
        if relative_end > storage_bytes {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "q8_0 file-backed read offset {offset} length {} exceeds backing storage range {}..{} ({} bytes)",
                out.len(),
                self.absolute_offset,
                self.absolute_offset.saturating_add(storage_bytes),
                storage_bytes
            )));
        }
        let cache_decoded_q8_0_scales = cached_scales
            .as_ref()
            .and_then(|scales| scales.len().checked_mul(Q8_0_BLOCK_BYTES))
            .is_some_and(|scale_bytes| out.len() == scale_bytes);

        let cache_capacity = q8_file_cache_capacity_bytes();
        if cache_capacity == 0 {
            // The bounded Q8 chunk cache is disabled by default for 8B memory headroom.
            // Keep the default matmul reader on a straight pread path instead of building
            // cache-miss range bookkeeping for every streamed weight chunk.
            q8_file_cache_apply_capacity(0);
            let file = self.file()?;
            read_exact_at(&file, out, offset).map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
            record_q8_0_file_read(out.len());
            if cache_decoded_q8_0_scales {
                if let Some(scales) = &mut cached_scales {
                    decode_q8_0_scales_from_bytes(out, scales);
                    return Ok(Q8FileReadScaleStatus::DecodedScalesReady);
                }
            }
            return Ok(Q8FileReadScaleStatus::NoScales);
        }

        let (ranges, decoded_scales_reused, decoded_scale_hit_blocks) =
            match q8_file_cache_prepare_read(
                &self.path,
                offset,
                out,
                cached_scales.as_deref_mut(),
                cache_capacity,
            ) {
                Q8FileCacheRead::Hit {
                    decoded_scales_reused,
                    decoded_scale_hit_blocks,
                } => {
                    return Ok(if decoded_scales_reused {
                        Q8FileReadScaleStatus::DecodedScalesReused {
                            cache_hit_blocks: decoded_scale_hit_blocks,
                        }
                    } else {
                        Q8FileReadScaleStatus::NoScales
                    });
                }
                Q8FileCacheRead::Missing {
                    ranges,
                    decoded_scales_reused,
                    decoded_scale_hit_blocks,
                } => (ranges, decoded_scales_reused, decoded_scale_hit_blocks),
            };
        let file = self.file()?;
        for range in &ranges {
            let range_offset = offset.checked_add(range.out_start as u64).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "q8_0 file cache read offset overflow".to_string(),
                )
            })?;
            let out_end = range.out_start + range.len;
            read_exact_at(&file, &mut out[range.out_start..out_end], range_offset).map_err(
                |source| BackendError::Io {
                    path: self.path.clone(),
                    source,
                },
            )?;
            record_q8_0_file_read(range.len);
        }
        let mut scale_status = Q8FileReadScaleStatus::NoScales;
        let decoded_scales = if cache_decoded_q8_0_scales {
            if let Some(scales) = &mut cached_scales {
                let scales = &mut **scales;
                if decoded_scales_reused
                    && decode_q8_0_scales_from_byte_ranges(out, &ranges, scales)
                {
                    scale_status = Q8FileReadScaleStatus::DecodedScalesReused {
                        cache_hit_blocks: decoded_scale_hit_blocks,
                    };
                } else {
                    decode_q8_0_scales_from_bytes(out, scales);
                    scale_status = Q8FileReadScaleStatus::DecodedScalesReady;
                }
                Some(scales.to_vec())
            } else {
                decode_q8_0_scales_from_cache_bytes(out)
            }
        } else {
            None
        };
        q8_file_cache_insert_with_decoded_scales(self.path.clone(), offset, out, decoded_scales);
        Ok(scale_status)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Q8FileReadScaleStatus {
    NoScales,
    DecodedScalesReady,
    DecodedScalesReused { cache_hit_blocks: usize },
}

impl Q8FileReadScaleStatus {
    fn scales_ready(self) -> bool {
        matches!(
            self,
            Q8FileReadScaleStatus::DecodedScalesReady
                | Q8FileReadScaleStatus::DecodedScalesReused { .. }
        )
    }

    fn decoded_scales_reused(self) -> bool {
        matches!(self, Q8FileReadScaleStatus::DecodedScalesReused { .. })
    }

    fn decoded_scale_hit_blocks(self) -> Option<usize> {
        match self {
            Q8FileReadScaleStatus::DecodedScalesReused { cache_hit_blocks } => {
                (cache_hit_blocks > 0).then_some(cache_hit_blocks)
            }
            _ => None,
        }
    }
}

impl PartialEq for Q8_0FileBacking {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.absolute_offset == other.absolute_offset
            && self.num_blocks == other.num_blocks
    }
}

impl Eq for Q8_0FileBacking {}

/// File-backed raw Q4_0 wire storage for a rank-3 streamed MoE expert pack.
///
/// Unlike [`Q8_0FileBacking`], the shared handle keeps OS file caching
/// ENABLED (no `F_NOCACHE`/readahead-off): a fine-grained MoE's routed expert
/// set cannot be RAM-resident on the hosts this lane targets, and the page
/// cache retaining recently routed experts between tokens is the streamed
/// lane's entire performance model.
#[derive(Debug, Clone)]
pub struct Q4_0FileBacking {
    pub path: PathBuf,
    pub absolute_offset: u64,
    pub num_blocks: usize,
    /// 18 for Q4_0, 20 for Q4_1 — the streamed lane serves both nibble
    /// formats; the tensor's `source_type` picks the dequantizer.
    pub wire_bytes_per_block: usize,
    file_handle: Arc<OnceLock<Arc<File>>>,
}

impl Q4_0FileBacking {
    pub const WIRE_BYTES_PER_BLOCK: usize = 18;

    pub fn new(
        path: PathBuf,
        absolute_offset: u64,
        num_blocks: usize,
        wire_bytes_per_block: usize,
    ) -> Self {
        Self {
            path,
            absolute_offset,
            num_blocks,
            wire_bytes_per_block,
            file_handle: Arc::new(OnceLock::new()),
        }
    }

    fn file(&self) -> Result<Arc<File>> {
        if let Some(file) = self.file_handle.get() {
            return Ok(file.clone());
        }
        let file = File::open(&self.path).map_err(|source| BackendError::Io {
            path: self.path.clone(),
            source,
        })?;
        let file = Arc::new(file);
        if self.file_handle.set(file.clone()).is_err() {
            return Ok(self
                .file_handle
                .get()
                .expect("q4_0 file handle must exist after OnceLock set race")
                .clone());
        }
        Ok(file)
    }

    /// Read `block_count` contiguous 18-byte Q4_0 wire blocks starting
    /// `block_offset` blocks into the backing.
    pub fn read_wire_blocks(&self, block_offset: usize, block_count: usize) -> Result<Vec<u8>> {
        let end_block = block_offset.checked_add(block_count).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("q4_0 file-backed block range overflow".to_string())
        })?;
        if end_block > self.num_blocks {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "q4_0 file-backed read blocks {block_offset}..{end_block} exceed backing of {} blocks",
                self.num_blocks
            )));
        }
        let mut out = vec![0u8; block_count * self.wire_bytes_per_block];
        let file = self.file()?;
        let offset = self
            .absolute_offset
            .saturating_add((block_offset * self.wire_bytes_per_block) as u64);
        crate::platform_fs::read_exact_at(&file, &mut out, offset).map_err(|source| {
            BackendError::Io {
                path: self.path.clone(),
                source,
            }
        })?;
        Ok(out)
    }
}

impl PartialEq for Q4_0FileBacking {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
            && self.absolute_offset == other.absolute_offset
            && self.num_blocks == other.num_blocks
    }
}

impl Eq for Q4_0FileBacking {}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub source_type: Option<GgufTensorType>,
    pub q8_0_blocks: Option<Vec<Q8_0Block>>,
    pub(crate) q8_0_shared_blocks: Option<Q8_0SharedBlocks>,
    pub q8_0_packed_rows4_4x4: Option<Q8_0PackedRows4>,
    pub q8_0_packed_rows4_4x8: Option<Q8_0PackedRows4>,
    pub q8_0_runtime_storage: Option<Q8_0RuntimeStorage>,
    pub q8_0_file_backing: Option<Q8_0FileBacking>,
    pub q8_0_wire_mmap: Option<crate::wire_mmap::WireMmapTensor>,
    pub q8_0_wire_pages: Option<std::sync::Arc<crate::wire_mmap::WirePages>>,
    /// Page-aligned raw GGUF wire bytes for a rank-2 compact tensor. This is
    /// the no-copy Metal counterpart of the format-specific `*_wire_bytes`
    /// fields below: only one representation is populated. The tensor's
    /// `source_type` identifies the K-quant or Prism Q1/Q2 block format.
    pub kquant_wire_pages: Option<std::sync::Arc<crate::wire_mmap::WirePages>>,
    pub q8_0_split_file_backing: Option<Vec<Q8_0FileBacking>>,
    /// Buffered file backing for a rank-3 Q4_0 MoE expert pack. Per-expert
    /// wire slices are read and dequantized transiently at matvec time; see
    /// [`Q4_0FileBacking`] for why this backing deliberately keeps the OS
    /// page cache enabled.
    pub q4_0_file_backing: Option<Q4_0FileBacking>,
    /// Q4_K_M super-block wire bytes (144 bytes/super-block, row-major), retained
    /// when the tensor's `source_type` is `Q4K` so the GPU-resident decode path can
    /// repack them into the `q4k_gemv` SoA layout. Populated by the Q4_K load path;
    /// `None` for non-Q4_K tensors.
    pub q4_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// STAMPEDE Lane B v5: lazily-built 8-row interleaved repack of the Q4_K
    /// wire bytes for the batched prefill owner (budget-gated; `None` inside
    /// the cell = build denied/unpackable). Derived cache — see
    /// [`Q4KRepack8Cell`] for the clone/equality semantics.
    pub q4_k_repack8: Q4KRepack8Cell,
    /// Q5_K super-block wire bytes (176 bytes/super-block, row-major), retained when
    /// the tensor's `source_type` is `Q5K` so the CPU block-dot streams them via
    /// `q5_k_wire_row_dot` (and the GPU-resident decode path can feed the `q5k_gemv`
    /// kernel) with no f32 materialisation. `None` for non-Q5_K tensors.
    pub q5_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Q6_K super-block wire bytes (210 bytes/super-block, row-major), retained when
    /// the tensor's `source_type` is `Q6K` so the GPU-resident decode path can feed
    /// them straight to the `q6k_gemv` kernel (which reads the wire layout directly).
    /// Populated by the Q6_K load path; `None` for non-Q6_K tensors.
    pub q6_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Q2_K super-block wire bytes (84 bytes/super-block, row-major), retained when
    /// the tensor's `source_type` is `Q2K` so the GPU-resident decode path can feed
    /// them straight to the `q2k_gemv` kernel (which reads the wire layout directly).
    /// Populated by the Q2_K load path; `None` for non-Q2_K tensors.
    pub q2_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Q3_K super-block wire bytes (110 bytes/super-block, row-major), retained when
    /// the tensor's `source_type` is `Q3K` so the GPU-resident decode path can feed
    /// them straight to the `q3k_gemv` kernel (which reads the wire layout directly).
    /// Populated by the Q3_K load path; `None` for non-Q3_K tensors. (Q2_K models mix
    /// in Q3_K projections — typically attn_output / ffn_down.)
    pub q3_k_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// Low-bit row-major wire bytes. Originally introduced for mainstream TQ2_0,
    /// this backing also retains Prism Q1_0 and both Q2_0 dialects so native GPU
    /// kernels consume the packed file representation without a Q8/F32 expansion.
    /// `source_type` is the authoritative format discriminator.
    pub tq2_0_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    /// IQ4_XS wire bytes (136 bytes/256-weight super-block, row-major), retained when the
    /// tensor's `source_type` is `IQ4XS` so the CPU i-quant block-dot streams the quantized
    /// weights instead of materialising f32. Populated by `load_iq4_xs_wire_linear`; `None`
    /// otherwise.
    pub iq4_xs_wire_bytes: Option<std::sync::Arc<Vec<u8>>>,
    pub data: Vec<f32>,
}

impl CpuTensor {
    /// Raw Q4_K wire bytes irrespective of whether the loader retained an
    /// ordinary `Vec` (portable/CUDA path) or page-aligned storage (Metal
    /// no-copy path).
    pub fn q4_k_wire(&self) -> Option<&[u8]> {
        if self.source_type != Some(GgufTensorType::Q4K) {
            return None;
        }
        self.q4_k_wire_bytes
            .as_deref()
            .map(Vec::as_slice)
            .or_else(|| self.kquant_wire_pages.as_deref().map(|pages| pages.bytes()))
    }

    /// Raw Q6_K wire bytes across ordinary and page-aligned storage.
    pub fn q6_k_wire(&self) -> Option<&[u8]> {
        if self.source_type != Some(GgufTensorType::Q6K) {
            return None;
        }
        self.q6_k_wire_bytes
            .as_deref()
            .map(Vec::as_slice)
            .or_else(|| self.kquant_wire_pages.as_deref().map(|pages| pages.bytes()))
    }

    /// Raw packed bytes for TQ2_0 and the Prism Q1/Q2 family.
    pub fn low_bit_wire(&self) -> Option<&[u8]> {
        if !matches!(
            self.source_type,
            Some(
                GgufTensorType::Tq2_0
                    | GgufTensorType::Q1_0
                    | GgufTensorType::Q2_0G64
                    | GgufTensorType::Q2_0G128
                    | GgufTensorType::Pq2_0
            )
        ) {
            return None;
        }
        self.tq2_0_wire_bytes
            .as_deref()
            .map(Vec::as_slice)
            .or_else(|| self.kquant_wire_pages.as_deref().map(|pages| pages.bytes()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0TensorBlocks {
    pub name: String,
    pub shape: TensorShape,
    pub blocks: Vec<Q8_0Block>,
}

impl Q8_0TensorBlocks {
    pub fn element_count(&self) -> Result<usize> {
        self.shape.element_count()
    }

    pub fn byte_size_if_f32_materialized(&self) -> Result<usize> {
        self.element_count()?.checked_mul(4).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} f32 materialization byte size overflow",
                self.name
            ))
        })
    }

    pub fn dequantize_elements(&self, start: usize, len: usize) -> Result<Vec<f32>> {
        const BLOCK_VALUES: usize = 32;
        let end = start.checked_add(len).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} q8_0 dequant range overflows usize",
                self.name
            ))
        })?;
        let element_count = self.element_count()?;
        if end > element_count {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 dequant range {start}..{end} exceeds element count {element_count}",
                self.name
            )));
        }

        let mut out = Vec::with_capacity(len);
        for element_idx in start..end {
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for element {element_idx}",
                    self.name
                ))
            })?;
            out.push(block.scale * f32::from(block.quants[quant_idx]));
        }
        Ok(out)
    }

    pub fn dequantize_row(&self, row: usize) -> Result<Vec<f32>> {
        let (_rows, cols) = self.rank2_row_shape(row, "row dequant")?;
        self.dequantize_elements(row * cols, cols)
    }

    pub fn dot_row_f32(&self, row: usize, input: &[f32]) -> Result<f32> {
        const BLOCK_VALUES: usize = 32;
        let (_rows, cols) = self.rank2_row_shape(row, "row dot")?;
        if input.len() != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let row_start = row.checked_mul(cols).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "tensor {} q8_0 row dot offset overflows usize",
                self.name
            ))
        })?;
        let mut sum = 0.0f32;
        for (col, input_value) in input.iter().enumerate() {
            let element_idx = row_start + col;
            let block_idx = element_idx / BLOCK_VALUES;
            let quant_idx = element_idx % BLOCK_VALUES;
            let block = self.blocks.get(block_idx).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 block index {block_idx} missing for row {row} col {col}",
                    self.name
                ))
            })?;
            sum += (block.scale * f32::from(block.quants[quant_idx])) * input_value;
        }
        Ok(sum)
    }

    pub fn dot_all_rows_f32(&self, input: &[f32], name: impl Into<String>) -> Result<CpuTensor> {
        const BLOCK_VALUES: usize = 32;
        let (rows, cols) = self.rank2_shape("all-row dot")?;
        if input.len() != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 all-row dot expected input width {cols}, got {}",
                self.name,
                input.len()
            )));
        }

        let mut data = Vec::with_capacity(rows);
        if cols % BLOCK_VALUES == 0 {
            let blocks_per_row = cols / BLOCK_VALUES;
            let expected_blocks = rows.checked_mul(blocks_per_row).ok_or_else(|| {
                BackendError::InvalidTensorData(format!(
                    "tensor {} q8_0 all-row dot block count overflows usize",
                    self.name
                ))
            })?;
            if self.blocks.len() != expected_blocks {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "tensor {} q8_0 all-row dot expected {expected_blocks} blocks for shape {:?}, got {}",
                    self.name,
                    self.shape.dims,
                    self.blocks.len()
                )));
            }

            for row_blocks in self.blocks.chunks_exact(blocks_per_row) {
                let mut row_sum = 0.0_f32;
                for (block, input_block) in row_blocks.iter().zip(input.chunks_exact(BLOCK_VALUES))
                {
                    for (quant, input_value) in block.quants.iter().zip(input_block) {
                        row_sum += (block.scale * f32::from(*quant)) * input_value;
                    }
                }
                data.push(row_sum);
            }
        } else {
            for row in 0..rows {
                data.push(self.dot_row_f32(row, input)?);
            }
        }

        Ok(CpuTensor {
            name: name.into(),
            shape: TensorShape { dims: vec![rows] },
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }

    pub fn dot_single_input_row_f32(
        &self,
        input: &CpuTensor,
        name: impl Into<String>,
    ) -> Result<CpuTensor> {
        if input.shape.dims.len() != 2 || input.shape.dims[0] != 1 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 lazy linear expected single input row, got {:?}",
                self.name, input.shape.dims
            )));
        }
        let mut output = self.dot_all_rows_f32(&input.data, name)?;
        output.shape.dims.insert(0, 1);
        Ok(output)
    }

    fn rank2_shape(&self, op: &str) -> Result<(usize, usize)> {
        if self.shape.dims.len() != 2 {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 {op} requires rank-2 shape, got {:?}",
                self.name, self.shape.dims
            )));
        }
        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];
        Ok((rows, cols))
    }

    fn rank2_row_shape(&self, row: usize, op: &str) -> Result<(usize, usize)> {
        let (rows, cols) = self.rank2_shape(op)?;
        if row >= rows {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor {} q8_0 row {row} out of range for {rows} rows",
                self.name
            )));
        }
        Ok((rows, cols))
    }
}

impl CpuTensor {
    /// Decompose into the owned name, dims, and f32 data buffer so the
    /// decode scratch pool can recycle all three. Only meaningful for
    /// plain-F32 tensors; quantized side-storage (never present on decode
    /// intermediates) is dropped.
    pub(crate) fn into_parts(self) -> (String, Vec<usize>, Vec<f32>) {
        (self.name, self.shape.dims, self.data)
    }

    pub fn from_f32(name: impl Into<String>, dims: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let shape = TensorShape { dims };
        let expected = shape.element_count()?;
        if expected != data.len() {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "tensor data length {} does not match shape element count {expected}",
                data.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: None,
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }

    pub fn from_f32_with_source_type(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        source_type: Option<GgufTensorType>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = source_type;
        Ok(tensor)
    }

    pub fn from_f32_with_q8_0_blocks(
        name: impl Into<String>,
        dims: Vec<usize>,
        data: Vec<f32>,
        q8_0_blocks: Vec<Q8_0Block>,
    ) -> Result<Self> {
        let mut tensor = Self::from_f32(name, dims, data)?;
        tensor.source_type = Some(GgufTensorType::Q8_0);
        tensor.q8_0_blocks = Some(q8_0_blocks);
        tensor.q8_0_packed_rows4_4x4 = q8_0_packed_rows4_for_shape(
            &tensor.name,
            &tensor.shape,
            tensor.q8_0_blocks.as_deref(),
            Q8_0PackedRows4Interleave::I4,
        )?;
        tensor.q8_0_packed_rows4_4x8 = q8_0_packed_rows4_for_shape(
            &tensor.name,
            &tensor.shape,
            tensor.q8_0_blocks.as_deref(),
            Q8_0PackedRows4Interleave::I8,
        )?;
        Ok(tensor)
    }

    pub fn from_q8_0_blocks(
        name: impl Into<String>,
        shape: TensorShape,
        q8_0_blocks: Vec<Q8_0Block>,
    ) -> Result<Self> {
        let expected_elements = shape.element_count()?;
        if !expected_elements.is_multiple_of(32) {
            return Err(BackendError::InvalidTensorData(format!(
                "q8_0 block-backed tensor element count {expected_elements} is not block aligned"
            )));
        }
        let expected_blocks = expected_elements / 32;
        if q8_0_blocks.len() != expected_blocks {
            return Err(BackendError::InvalidTensorData(format!(
                "q8_0 block-backed tensor expected {expected_blocks} blocks, got {}",
                q8_0_blocks.len()
            )));
        }
        let name = name.into();
        let q8_0_packed_rows4_4x4 = q8_0_packed_rows4_for_shape(
            &name,
            &shape,
            Some(&q8_0_blocks),
            Q8_0PackedRows4Interleave::I4,
        )?;
        let q8_0_packed_rows4_4x8 = q8_0_packed_rows4_for_shape(
            &name,
            &shape,
            Some(&q8_0_blocks),
            Q8_0PackedRows4Interleave::I8,
        )?;
        Ok(Self {
            name,
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: Some(q8_0_blocks),
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4,
            q8_0_packed_rows4_4x8,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    pub(crate) fn from_q8_0_shared_blocks(
        name: impl Into<String>,
        shape: TensorShape,
        blocks: std::sync::Arc<Vec<Q8_0Block>>,
        start: usize,
        len: usize,
    ) -> Result<Self> {
        let expected_elements = shape.element_count()?;
        if !expected_elements.is_multiple_of(32) {
            return Err(BackendError::InvalidTensorData(format!(
                "shared q8_0 tensor element count {expected_elements} is not block aligned"
            )));
        }
        let expected_blocks = expected_elements / 32;
        if len != expected_blocks || start.saturating_add(len) > blocks.len() {
            return Err(BackendError::InvalidTensorData(format!(
                "shared q8_0 tensor expected {expected_blocks} blocks at {start}, got range length {len} over {} blocks",
                blocks.len()
            )));
        }
        Ok(Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: None,
            q8_0_shared_blocks: Some(Q8_0SharedBlocks { blocks, start, len }),
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    pub(crate) fn q8_0_block_slice(&self) -> Option<&[Q8_0Block]> {
        self.q8_0_blocks.as_deref().or_else(|| {
            self.q8_0_shared_blocks
                .as_ref()
                .map(Q8_0SharedBlocks::as_slice)
        })
    }

    #[allow(dead_code)]
    pub(crate) fn q8_0_raw_bytes(&self) -> Option<std::borrow::Cow<'_, [u8]>> {
        if let Some(blocks) = self.q8_0_block_slice() {
            return Some(std::borrow::Cow::Borrowed(q8_0_blocks_as_bytes(blocks)));
        }
        if let Some(Q8_0RuntimeStorage::PackedRows4(packed)) = &self.q8_0_runtime_storage {
            return Some(std::borrow::Cow::Owned(packed.to_q8_0_bytes()));
        }
        if let Some(wire) = &self.q8_0_wire_mmap {
            if let Ok(b) = wire.bytes() {
                return Some(std::borrow::Cow::Borrowed(b));
            }
        }
        if let Some(wire) = &self.q8_0_wire_pages {
            return Some(std::borrow::Cow::Borrowed(wire.bytes()));
        }
        None
    }

    pub fn with_q8_0_file_backing(mut self, backing: Q8_0FileBacking) -> Self {
        self.q8_0_file_backing = Some(backing);
        self
    }

    pub fn q8_0_file_backed_linear(
        name: impl Into<String>,
        shape: TensorShape,
        backing: Q8_0FileBacking,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: Some(backing),
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        }
    }

    /// Descriptor-only rank-3 Q4_0/Q4_1 MoE expert pack: no weight bytes are
    /// materialized at load; per-expert wire slices stream (buffered) from the
    /// backing at matvec time and are dequantized transiently.
    pub fn q4_0_file_backed_experts(
        name: impl Into<String>,
        shape: TensorShape,
        backing: Q4_0FileBacking,
        source_type: GgufTensorType,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(source_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: Some(backing),
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        }
    }

    pub fn q8_0_runtime_packed_rows4_linear(
        name: impl Into<String>,
        shape: TensorShape,
        packed: Q8_0PackedRows4,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: Some(Q8_0RuntimeStorage::PackedRows4(packed)),
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        }
    }

    pub fn q8_0_split_file_backed_tensor(
        name: impl Into<String>,
        shape: TensorShape,
        backings: Vec<Q8_0FileBacking>,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(GgufTensorType::Q8_0),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: Some(backings),
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.dims.len()
    }

    pub fn dim(&self, idx: usize) -> Result<usize> {
        self.shape.dims.get(idx).copied().ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(format!(
                "tensor {} rank {} has no dimension {idx}",
                self.name,
                self.rank()
            ))
        })
    }

    pub fn matmul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul lhs")?;
        require_rank(rhs, 2, "matmul rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let rhs_k = rhs.dim(0)?;
        let n = rhs.dim(1)?;
        if k != rhs_k {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "matmul shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                cblas_sgemm(
                    101, // CBLAS_ROW_MAJOR
                    111, // CBLAS_NO_TRANS
                    111, // CBLAS_NO_TRANS
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    self.data.as_ptr(),
                    k as i32,
                    rhs.data.as_ptr(),
                    n as i32,
                    0.0,
                    out.as_mut_ptr(),
                    n as i32,
                );
            }
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if should_parallelize_linear_output(n) {
                for row in 0..m {
                    let lhs_start = row * k;
                    let out_start = row * n;
                    let out_row = &mut out[out_start..out_start + n];
                    out_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(col, out_value)| {
                            let mut sum = 0.0;
                            for inner in 0..k {
                                let lhs_value = self.data[lhs_start + inner];
                                if lhs_value == 0.0 {
                                    continue;
                                }
                                sum += lhs_value * rhs.data[inner * n + col];
                            }
                            *out_value = sum;
                        });
                }
            } else if should_parallelize_linear_output(m * n) {
                out.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let lhs_start = row * k;
                        for inner in 0..k {
                            let lhs_value = self.data[lhs_start + inner];
                            if lhs_value == 0.0 {
                                continue;
                            }
                            let rhs_start = inner * n;
                            let rhs_row = &rhs.data[rhs_start..rhs_start + n];
                            for col in 0..n {
                                out_row[col] += lhs_value * rhs_row[col];
                            }
                        }
                    });
            } else {
                for row in 0..m {
                    let lhs_start = row * k;
                    let out_start = row * n;
                    let out_row = &mut out[out_start..out_start + n];
                    for inner in 0..k {
                        let lhs_value = self.data[lhs_start + inner];
                        if lhs_value == 0.0 {
                            continue;
                        }
                        let rhs_start = inner * n;
                        let rhs_row = &rhs.data[rhs_start..rhs_start + n];
                        for col in 0..n {
                            out_row[col] += lhs_value * rhs_row[col];
                        }
                    }
                }
            }
        }

        Self::from_f32(name, vec![m, n], out)
    }

    pub fn matmul_rhs_transposed(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "matmul rhs-transposed lhs")?;
        require_rank(rhs, 2, "matmul rhs-transposed rhs")?;
        rhs.require_row_major_f32_data("matmul rhs-transposed rhs")?;
        let m = self.dim(0)?;
        let k = self.dim(1)?;
        let n = rhs.dim(0)?;
        let rhs_k = rhs.dim(1)?;
        if k != rhs_k {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "matmul rhs-transposed shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; m * n];

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            unsafe {
                cblas_sgemm(
                    101, // CBLAS_ROW_MAJOR
                    111, // CBLAS_NO_TRANS
                    112, // CBLAS_TRANS
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    self.data.as_ptr(),
                    k as i32,
                    rhs.data.as_ptr(),
                    k as i32,
                    0.0,
                    out.as_mut_ptr(),
                    n as i32,
                );
            }
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if should_parallelize_linear_output(n) {
                for row in 0..m {
                    let lhs_start = row * k;
                    let lhs_row = &self.data[lhs_start..lhs_start + k];
                    let out_start = row * n;
                    let out_row = &mut out[out_start..out_start + n];
                    out_row
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(col, out_value)| {
                            let rhs_start = col * k;
                            let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                            *out_value = dot_product(lhs_row, rhs_row);
                        });
                }
            } else if should_parallelize_linear_output(m * n) {
                out.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let lhs_start = row * k;
                        let lhs_row = &self.data[lhs_start..lhs_start + k];
                        for (col, out_value) in out_row.iter_mut().enumerate() {
                            let rhs_start = col * k;
                            let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                            *out_value = dot_product(lhs_row, rhs_row);
                        }
                    });
            } else {
                for row in 0..m {
                    let lhs_start = row * k;
                    let lhs_row = &self.data[lhs_start..lhs_start + k];
                    let out_start = row * n;
                    let out_row = &mut out[out_start..out_start + n];
                    for (col, out_value) in out_row.iter_mut().enumerate() {
                        let rhs_start = col * k;
                        let rhs_row = &rhs.data[rhs_start..rhs_start + k];
                        *out_value = dot_product(lhs_row, rhs_row);
                    }
                }
            }
        }

        Self::from_f32(name, vec![m, n], out)
    }

    fn require_row_major_f32_data(&self, context: &str) -> Result<()> {
        let expected_len = self.shape.element_count()?;
        if self.data.len() == expected_len {
            return Ok(());
        }
        let storage = if self.q8_0_runtime_storage.is_some() {
            "runtime-packed-q8"
        } else if self.q8_0_blocks.is_some() {
            "retained-q8-blocks"
        } else if self.q8_0_file_backing.is_some() {
            "file-backed-q8"
        } else if self.data.is_empty() {
            "no-row-major-data"
        } else {
            "invalid-row-major-f32"
        };
        Err(BackendError::InvalidTensorData(format!(
            "{context} cannot read tensor {} as row-major f32: storage={storage}, shape={:?}, data_len={}, expected_len={expected_len}",
            self.name, self.shape.dims, self.data.len()
        )))
    }

    pub fn add(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        let mut out = vec![0.0; self.data.len()];
        self.add_into(rhs, &mut out)?;
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// The exact kernel of [`Self::add`], writing into a caller-provided
    /// buffer (same length as `self.data`). Shared by the allocating path
    /// above and the decode scratch-pool path so both are one numeric path.
    pub(crate) fn add_into(&self, rhs: &Self, out: &mut [f32]) -> Result<()> {
        if self.shape != rhs.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let len = self.data.len();
        debug_assert_eq!(out.len(), len);
        if should_parallelize_linear_output(len) {
            out.par_iter_mut()
                .zip(self.data.par_iter())
                .zip(rhs.data.par_iter())
                .for_each(|((o, &a), &b)| {
                    *o = a + b;
                });
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                use std::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
                let mut i = 0;
                unsafe {
                    while i + 4 <= len {
                        let va = vld1q_f32(self.data.as_ptr().add(i));
                        let vb = vld1q_f32(rhs.data.as_ptr().add(i));
                        let vout = vaddq_f32(va, vb);
                        vst1q_f32(out.as_mut_ptr().add(i), vout);
                        i += 4;
                    }
                    while i < len {
                        out[i] = self.data[i] + rhs.data[i];
                        i += 1;
                    }
                }
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                for (i, output) in out.iter_mut().enumerate().take(len) {
                    *output = self.data[i] + rhs.data[i];
                }
            }
        }
        Ok(())
    }

    pub fn mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let mut out = vec![0.0; self.data.len()];
        let len = self.data.len();
        if should_parallelize_linear_output(len) {
            out.par_iter_mut()
                .zip(self.data.par_iter())
                .zip(rhs.data.par_iter())
                .for_each(|((o, &a), &b)| {
                    *o = a * b;
                });
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                use std::arch::aarch64::{vld1q_f32, vmulq_f32, vst1q_f32};
                let mut i = 0;
                unsafe {
                    while i + 4 <= len {
                        let va = vld1q_f32(self.data.as_ptr().add(i));
                        let vb = vld1q_f32(rhs.data.as_ptr().add(i));
                        let vout = vmulq_f32(va, vb);
                        vst1q_f32(out.as_mut_ptr().add(i), vout);
                        i += 4;
                    }
                    while i < len {
                        out[i] = self.data[i] * rhs.data[i];
                        i += 1;
                    }
                }
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                for (i, output) in out.iter_mut().enumerate().take(len) {
                    *output = self.data[i] * rhs.data[i];
                }
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn silu_mul(&self, rhs: &Self, name: impl Into<String>) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        let len = self.data.len();
        let mut out = vec![0.0; len];
        if should_parallelize_linear_output(len) {
            out.par_iter_mut()
                .zip(self.data.par_iter())
                .zip(rhs.data.par_iter())
                .for_each(|((o, &a), &b)| {
                    *o = (a / (1.0 + (-a).exp())) * b;
                });
        } else {
            for (i, o) in out.iter_mut().enumerate().take(len) {
                let a = self.data[i];
                let b = rhs.data[i];
                *o = (a / (1.0 + (-a).exp())) * b;
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn silu(&self, name: impl Into<String>) -> Result<Self> {
        let len = self.data.len();
        let mut out = vec![0.0; len];
        if should_parallelize_linear_output(len) {
            out.par_iter_mut()
                .zip(self.data.par_iter())
                .for_each(|(o, &x)| {
                    *o = x / (1.0 + (-x).exp());
                });
        } else {
            for (i, o) in out.iter_mut().enumerate().take(len) {
                let x = self.data[i];
                *o = x / (1.0 + (-x).exp());
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[inline(always)]
    unsafe fn rms_norm_neon(input: &[f32], weight: &[f32], out: &mut [f32], cols: usize, eps: f32) {
        use std::arch::aarch64::{
            vaddq_f32, vdupq_n_f32, vget_high_f32, vget_lane_f32, vget_low_f32, vld1q_f32,
            vmulq_f32, vpadd_f32, vst1q_f32,
        };

        let mut sum_sq_vec = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= cols {
            let v = vld1q_f32(input.as_ptr().add(i));
            sum_sq_vec = vaddq_f32(sum_sq_vec, vmulq_f32(v, v));
            i += 4;
        }
        let low = vget_low_f32(sum_sq_vec);
        let high = vget_high_f32(sum_sq_vec);
        let sum_2 = vpadd_f32(low, high);
        let mut sum_sq = vget_lane_f32::<0>(sum_2) + vget_lane_f32::<1>(sum_2);
        while i < cols {
            let v = input[i];
            sum_sq += v * v;
            i += 1;
        }

        let mean_square = sum_sq / cols as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        let scale_vec = vdupq_n_f32(scale);

        i = 0;
        while i + 4 <= cols {
            let v_in = vld1q_f32(input.as_ptr().add(i));
            let v_w = vld1q_f32(weight.as_ptr().add(i));
            let v_out = vmulq_f32(vmulq_f32(v_in, scale_vec), v_w);
            vst1q_f32(out.as_mut_ptr().add(i), v_out);
            i += 4;
        }
        while i < cols {
            out[i] = input[i] * scale * weight[i];
            i += 1;
        }
    }

    pub fn rms_norm(&self, weight: &Self, eps: f32, name: impl Into<String>) -> Result<Self> {
        let mut out = vec![0.0; self.data.len()];
        self.rms_norm_into(weight, eps, &mut out)?;
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    /// The exact kernel of [`Self::rms_norm`], writing into a caller-provided
    /// buffer (same length as `self.data`). Shared by the allocating path
    /// above and the decode scratch-pool path so both are one numeric path
    /// (same reduction order, same parallel split).
    pub(crate) fn rms_norm_into(&self, weight: &Self, eps: f32, out: &mut [f32]) -> Result<()> {
        require_rank(self, 2, "rms_norm input")?;
        require_rank(weight, 1, "rms_norm weight")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        if weight.dim(0)? != cols {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "rms_norm weight shape {:?} does not match input shape {:?}",
                weight.shape.dims, self.shape.dims
            )));
        }
        debug_assert_eq!(out.len(), self.data.len());

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if should_parallelize_linear_output(self.data.len()) {
                out.par_chunks_mut(cols)
                    .zip(self.data.par_chunks(cols))
                    .for_each(|(out_row, in_row)| unsafe {
                        Self::rms_norm_neon(in_row, &weight.data, out_row, cols, eps);
                    });
            } else {
                for row in 0..rows {
                    let start = row * cols;
                    let in_row = &self.data[start..start + cols];
                    let out_row = &mut out[start..start + cols];
                    unsafe {
                        Self::rms_norm_neon(in_row, &weight.data, out_row, cols, eps);
                    }
                }
            }
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if should_parallelize_linear_output(self.data.len()) {
                out.par_chunks_mut(cols)
                    .zip(self.data.par_chunks(cols))
                    .for_each(|(out_row, in_row)| {
                        let mean_square = in_row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                        let scale = 1.0 / (mean_square + eps).sqrt();
                        for col in 0..cols {
                            out_row[col] = in_row[col] * scale * weight.data[col];
                        }
                    });
            } else {
                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    let mean_square =
                        self.data[start..end].iter().map(|v| v * v).sum::<f32>() / cols as f32;
                    let scale = 1.0 / (mean_square + eps).sqrt();
                    for col in 0..cols {
                        out[start + col] = self.data[start + col] * scale * weight.data[col];
                    }
                }
            }
        }

        Ok(())
    }

    /// Per-head RMSNorm (Qwen3 QK-norm). Treats each row of this `[rows, cols]`
    /// tensor as `head_count` contiguous heads of `head_dim = cols / head_count`
    /// and RMS-normalizes each head independently with the shared `[head_dim]`
    /// weight. This is what Qwen3 applies to the Q and K projections after
    /// reshape-to-heads and before RoPE.
    ///
    /// Because the data is row-major, the head slices of a `[rows, cols]` tensor
    /// are exactly the rows of a `[rows*head_count, head_dim]` tensor, so this
    /// reuses [`Self::rms_norm`] verbatim — same numeric path as every other RMS
    /// norm in the engine.
    pub fn per_head_rms_norm(
        &self,
        weight: &Self,
        head_count: usize,
        eps: f32,
        name: impl Into<String>,
    ) -> Result<Self> {
        require_rank(self, 2, "per_head_rms_norm input")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        if head_count == 0 || !cols.is_multiple_of(head_count) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "per_head_rms_norm width {cols} is not divisible by head count {head_count}"
            )));
        }
        let head_dim = cols / head_count;
        let name = name.into();
        let per_head = Self::from_f32(
            name.clone(),
            vec![rows * head_count, head_dim],
            self.data.clone(),
        )?;
        let normed = per_head.rms_norm(weight, eps, name.clone())?;
        Self::from_f32(name, vec![rows, cols], normed.data)
    }

    pub fn softmax_last_dim(&self, name: impl Into<String>) -> Result<Self> {
        if self.shape.dims.is_empty() {
            return Err(BackendError::RuntimeShapeMismatch(
                "softmax requires at least one dimension".to_string(),
            ));
        }
        let cols = *self.shape.dims.last().expect("non-empty dims");
        if cols == 0 || !self.data.len().is_multiple_of(cols) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "softmax invalid shape {:?} for data length {}",
                self.shape.dims,
                self.data.len()
            )));
        }
        let mut out = self.data.clone();
        if should_parallelize_linear_output(out.len()) {
            out.par_chunks_exact_mut(cols)
                .map(|row| {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    (row, sum)
                })
                .try_for_each(|(row, sum)| {
                    if sum == 0.0 || !sum.is_finite() {
                        return Err(BackendError::RuntimeShapeMismatch(
                            "softmax produced invalid normalization sum".to_string(),
                        ));
                    }
                    for v in row.iter_mut() {
                        *v /= sum;
                    }
                    Ok(())
                })?;
        } else {
            for row in out.chunks_exact_mut(cols) {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                if sum == 0.0 || !sum.is_finite() {
                    return Err(BackendError::RuntimeShapeMismatch(
                        "softmax produced invalid normalization sum".to_string(),
                    ));
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

    pub fn embedding_lookup(&self, token_ids: &[u32], name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "embedding weight")?;
        let vocab = self.dim(0)?;
        let width = self.dim(1)?;
        if let Some(pages) = self.q8_0_wire_pages.as_ref() {
            return self.embedding_lookup_q8_0_wire_pages(token_ids, name, vocab, width, pages);
        }
        if let Some(backing) = self.q8_0_file_backing.as_ref() {
            return self.embedding_lookup_q8_0_file_backed(token_ids, name, vocab, width, backing);
        }
        if let Some(blocks) = self.q8_0_blocks.as_deref() {
            return self.embedding_lookup_q8_0_block_backed(token_ids, name, vocab, width, blocks);
        }
        if let Some(wire) = self.low_bit_wire() {
            if matches!(
                self.source_type,
                Some(
                    GgufTensorType::Q1_0
                        | GgufTensorType::Q2_0G64
                        | GgufTensorType::Q2_0G128
                        | GgufTensorType::Pq2_0,
                )
            ) {
                return self.embedding_lookup_low_bit_wire(token_ids, name, vocab, width, wire);
            }
        }
        // K-quant token-embedding: the wire-only loader leaves `data` empty, so gather
        // each requested row by dequantizing its super-blocks straight from wire bytes.
        if let Some(wire) = self.q4_k_wire() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q4_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q4_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q4KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q5_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q5_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q5_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q5KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q6_k_wire() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q6_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q6_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q6KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q2_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q2_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q2_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q2KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.q3_k_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                Q3_K_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; Q3_K_BLOCK_BYTES] = b.try_into().unwrap();
                    Q3KBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        if let Some(wire) = self.iq4_xs_wire_bytes.as_deref() {
            return self.embedding_lookup_kquant_wire(
                token_ids,
                name,
                vocab,
                width,
                wire,
                IQ4_XS_BLOCK_BYTES,
                |b, out| {
                    let blk: &[u8; IQ4_XS_BLOCK_BYTES] = b.try_into().unwrap();
                    IQ4XSBlock::from_bytes(blk).dequantize(out);
                },
            );
        }
        let expected_dense_len = vocab.checked_mul(width).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("embedding dense element count overflow".to_string())
        })?;
        if self.data.len() != expected_dense_len {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "embedding tensor {} has no readable backing: dense elements {}, expected {}; source type {:?}",
                self.name,
                self.data.len(),
                expected_dense_len,
                self.source_type
            )));
        }
        let output_len = token_ids.len().checked_mul(width).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "embedding lookup output element count overflow".to_string(),
            )
        })?;
        let mut out = Vec::with_capacity(output_len);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let start = token_idx.checked_mul(width).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "embedding lookup row start overflow".to_string(),
                )
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch("embedding lookup row end overflow".to_string())
            })?;
            out.extend_from_slice(&self.data[start..end]);
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    /// Decode embedding rows straight from page-aligned wire bytes (34-byte
    /// f16-scale blocks) — the fast-load path's memory-speed CPU gather.
    fn embedding_lookup_q8_0_wire_pages(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        pages: &crate::wire_mmap::WirePages,
    ) -> Result<Self> {
        if !width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "embedding width {width} is not a multiple of {Q8_0_BLOCK_VALUES}"
            )));
        }
        let blocks_per_row = width / Q8_0_BLOCK_VALUES;
        let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
        let bytes = pages.bytes();
        if bytes.len() != vocab * row_bytes {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "embedding wire pages hold {} bytes, expected {} for [{vocab}, {width}]",
                bytes.len(),
                vocab * row_bytes
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len() * width);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let row = &bytes[token_idx * row_bytes..(token_idx + 1) * row_bytes];
            for block in row.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                out.extend(block[2..].iter().map(|&q| scale * f32::from(q as i8)));
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    fn embedding_lookup_q8_0_block_backed(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        blocks: &[Q8_0Block],
    ) -> Result<Self> {
        const Q8_0_BLOCK_VALUES: usize = 32;
        if self.source_type != Some(GgufTensorType::Q8_0) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "block-backed embedding {} must come from Q8_0 storage",
                self.name
            )));
        }
        if !width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "block-backed q8_0 embedding width {width} is not divisible by {Q8_0_BLOCK_VALUES}"
            )));
        }
        let blocks_per_row = width / Q8_0_BLOCK_VALUES;
        let expected_blocks = vocab.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "block-backed q8_0 embedding block count overflow".to_string(),
            )
        })?;
        if blocks.len() != expected_blocks {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "block-backed q8_0 embedding block count {} does not match expected {expected_blocks}",
                blocks.len()
            )));
        }
        let output_len = token_ids.len().checked_mul(width).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "block-backed q8_0 embedding output element count overflow".to_string(),
            )
        })?;
        let mut out = Vec::with_capacity(output_len);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let block_start = token_idx.checked_mul(blocks_per_row).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "block-backed q8_0 embedding row start overflow".to_string(),
                )
            })?;
            for block in &blocks[block_start..block_start + blocks_per_row] {
                out.extend(
                    block
                        .quants
                        .iter()
                        .map(|quant| block.scale * f32::from(*quant)),
                );
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    /// Gather K-quant (Q4_K / Q6_K) embedding rows straight from the super-block wire
    /// bytes (the wire-only loader leaves `data` empty). Only the few requested rows are
    /// dequantized via `dequant_block`, so this is cheap. `block_bytes` is the wire
    /// super-block size (144 for Q4_K, 210 for Q6_K); each super-block holds 256 values.
    #[allow(clippy::too_many_arguments)]
    fn embedding_lookup_kquant_wire(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        wire: &[u8],
        block_bytes: usize,
        dequant_block: impl Fn(&[u8], &mut [f32; QK_K_BLOCK_SIZE]),
    ) -> Result<Self> {
        if !width.is_multiple_of(QK_K_BLOCK_SIZE) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "K-quant embedding width {width} is not divisible by {QK_K_BLOCK_SIZE}"
            )));
        }
        let blocks_per_row = width / QK_K_BLOCK_SIZE;
        let row_bytes = blocks_per_row * block_bytes;
        let expected = vocab.checked_mul(row_bytes).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("K-quant embedding byte count overflow".to_string())
        })?;
        if wire.len() != expected {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "K-quant embedding wire bytes {} do not match expected {expected}",
                wire.len()
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len() * width);
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let row = &wire[token_idx * row_bytes..(token_idx + 1) * row_bytes];
            for b in 0..blocks_per_row {
                dequant_block(&row[b * block_bytes..(b + 1) * block_bytes], &mut values);
                out.extend_from_slice(&values);
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    fn embedding_lookup_low_bit_wire(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        wire: &[u8],
    ) -> Result<Self> {
        let tensor_type = self.source_type.ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("low-bit embedding has no source type".to_string())
        })?;
        let (block_elements, block_bytes) = tensor_type.layout().ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(format!(
                "low-bit embedding has unresolved layout {tensor_type:?}"
            ))
        })?;
        let block_elements = block_elements as usize;
        let block_bytes = block_bytes as usize;
        if !width.is_multiple_of(block_elements) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "{tensor_type:?} embedding width {width} is not divisible by {block_elements}"
            )));
        }
        let row_bytes = width / block_elements * block_bytes;
        let expected = vocab.checked_mul(row_bytes).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("low-bit embedding byte count overflow".to_string())
        })?;
        if wire.len() != expected {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "low-bit embedding wire bytes {} do not match expected {expected}",
                wire.len()
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len() * width);
        for &token_id in token_ids {
            let token_idx = usize::try_from(token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let row = &wire[token_idx * row_bytes..(token_idx + 1) * row_bytes];
            let decoded = match tensor_type {
                GgufTensorType::Q1_0 => decode_q1_0_tensor("embedding row", row, width)?,
                GgufTensorType::Q2_0G64 | GgufTensorType::Q2_0G128 | GgufTensorType::Pq2_0 => {
                    decode_q2_0_tensor("embedding row", row, width, tensor_type)?
                }
                other => {
                    return Err(BackendError::RuntimeShapeMismatch(format!(
                        "unsupported low-bit embedding type {other:?}"
                    )))
                }
            };
            out.extend_from_slice(&decoded);
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    fn embedding_lookup_q8_0_file_backed(
        &self,
        token_ids: &[u32],
        name: impl Into<String>,
        vocab: usize,
        width: usize,
        backing: &Q8_0FileBacking,
    ) -> Result<Self> {
        const Q8_0_BLOCK_VALUES: usize = 32;
        const Q8_0_BLOCK_BYTES: usize = 34;
        if self.source_type != Some(GgufTensorType::Q8_0) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed embedding {} must come from Q8_0 storage",
                self.name
            )));
        }
        if !width.is_multiple_of(Q8_0_BLOCK_VALUES) {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed q8_0 embedding width {width} is not divisible by {Q8_0_BLOCK_VALUES}"
            )));
        }
        let blocks_per_row = width / Q8_0_BLOCK_VALUES;
        let expected_blocks = vocab.checked_mul(blocks_per_row).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "file-backed q8_0 embedding block count overflow".to_string(),
            )
        })?;
        if backing.num_blocks != expected_blocks {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "file-backed q8_0 embedding block count {} does not match expected {expected_blocks}",
                backing.num_blocks
            )));
        }
        let row_bytes = blocks_per_row
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "file-backed q8_0 embedding row byte count overflow".to_string(),
                )
            })?;
        let output_len = token_ids.len().checked_mul(width).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch(
                "file-backed q8_0 embedding output element count overflow".to_string(),
            )
        })?;
        let mut row = vec![0_u8; row_bytes];
        let mut out = Vec::with_capacity(output_len);
        for token_id in token_ids {
            let token_idx = usize::try_from(*token_id).map_err(|_| {
                BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} does not fit usize"
                ))
            })?;
            if token_idx >= vocab {
                return Err(BackendError::RuntimeShapeMismatch(format!(
                    "token id {token_id} out of range for vocab size {vocab}"
                )));
            }
            let relative_offset = token_idx.checked_mul(row_bytes).ok_or_else(|| {
                BackendError::RuntimeShapeMismatch(
                    "file-backed q8_0 embedding row byte offset overflow".to_string(),
                )
            })?;
            let relative_offset = u64::try_from(relative_offset).map_err(|_| {
                BackendError::RuntimeShapeMismatch(
                    "file-backed q8_0 embedding row byte offset does not fit u64".to_string(),
                )
            })?;
            let offset = backing
                .absolute_offset
                .checked_add(relative_offset)
                .ok_or_else(|| {
                    BackendError::RuntimeShapeMismatch(
                        "file-backed q8_0 embedding absolute row byte offset overflow".to_string(),
                    )
                })?;
            backing.read_exact_at_cached(&mut row, offset)?;
            for block in row.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
                out.extend(block[2..].iter().map(|q| scale * f32::from(*q as i8)));
            }
        }
        Self::from_f32(name, vec![token_ids.len(), width], out)
    }

    pub fn transpose_2d(&self, name: impl Into<String>) -> Result<Self> {
        require_rank(self, 2, "transpose")?;
        let rows = self.dim(0)?;
        let cols = self.dim(1)?;
        let mut out = vec![0.0; self.data.len()];
        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = self.data[row * cols + col];
            }
        }
        Self::from_f32(name, vec![cols, rows], out)
    }

    #[allow(dead_code)]
    fn zip_same_shape(
        &self,
        rhs: &Self,
        name: impl Into<String>,
        f: impl Fn(f32, f32) -> f32,
    ) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "shape mismatch: lhs {:?}, rhs {:?}",
                self.shape.dims, rhs.shape.dims
            )));
        }
        Self::from_f32(
            name,
            self.shape.dims.clone(),
            self.data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| f(*a, *b))
                .collect(),
        )
    }
}

fn require_rank(tensor: &CpuTensor, rank: usize, op: &str) -> Result<()> {
    if tensor.rank() != rank {
        return Err(BackendError::RuntimeShapeMismatch(format!(
            "{op} expected rank {rank}, got shape {:?}",
            tensor.shape.dims
        )));
    }
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    use std::arch::aarch64::{
        vaddq_f32, vdupq_n_f32, vget_high_f32, vget_lane_f32, vget_low_f32, vld1q_f32, vmulq_f32,
        vpadd_f32,
    };
    let len = lhs.len();
    let mut idx = 0;
    unsafe {
        let mut sum_vec = vdupq_n_f32(0.0);
        // Unroll 4x (16 floats per iteration) for maximum instruction pipelining and data throughput
        while idx + 16 <= len {
            let l0 = vld1q_f32(lhs.as_ptr().add(idx));
            let r0 = vld1q_f32(rhs.as_ptr().add(idx));
            let l1 = vld1q_f32(lhs.as_ptr().add(idx + 4));
            let r1 = vld1q_f32(rhs.as_ptr().add(idx + 4));
            let l2 = vld1q_f32(lhs.as_ptr().add(idx + 8));
            let r2 = vld1q_f32(rhs.as_ptr().add(idx + 8));
            let l3 = vld1q_f32(lhs.as_ptr().add(idx + 12));
            let r3 = vld1q_f32(rhs.as_ptr().add(idx + 12));

            let m0 = vmulq_f32(l0, r0);
            let m1 = vmulq_f32(l1, r1);
            let m2 = vmulq_f32(l2, r2);
            let m3 = vmulq_f32(l3, r3);

            let s01 = vaddq_f32(m0, m1);
            let s23 = vaddq_f32(m2, m3);
            sum_vec = vaddq_f32(sum_vec, vaddq_f32(s01, s23));
            idx += 16;
        }
        while idx + 4 <= len {
            let l = vld1q_f32(lhs.as_ptr().add(idx));
            let r = vld1q_f32(rhs.as_ptr().add(idx));
            sum_vec = vaddq_f32(sum_vec, vmulq_f32(l, r));
            idx += 4;
        }
        let low = vget_low_f32(sum_vec);
        let high = vget_high_f32(sum_vec);
        let sum_2 = vpadd_f32(low, high);
        let mut sum = vget_lane_f32::<0>(sum_2) + vget_lane_f32::<1>(sum_2);
        while idx < len {
            sum += lhs[idx] * rhs[idx];
            idx += 1;
        }
        sum
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut sum = 0.0;
    let mut idx = 0;
    while idx + 4 <= lhs.len() {
        sum += lhs[idx] * rhs[idx];
        sum += lhs[idx + 1] * rhs[idx + 1];
        sum += lhs[idx + 2] * rhs[idx + 2];
        sum += lhs[idx + 3] * rhs[idx + 3];
        idx += 4;
    }
    while idx < lhs.len() {
        sum += lhs[idx] * rhs[idx];
        idx += 1;
    }
    sum
}
const DEFAULT_PARALLEL_LINEAR_MIN_OUTPUTS: usize = 1024;

static Q8_0_FILE_READ_CALLS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_READ_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_HIT_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MISS_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_INSERTS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_INSERT_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_EVICTIONS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_EVICTED_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MERGES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_MERGED_BYTES: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_DECODED_SCALE_HITS: AtomicU64 = AtomicU64::new(0);
static Q8_0_FILE_CACHE_DECODED_SCALE_HIT_BLOCKS: AtomicU64 = AtomicU64::new(0);
static Q8_FILE_CACHE: OnceLock<Mutex<Q8FileCache>> = OnceLock::new();

thread_local! {
    static Q8_FILE_CACHE_CAPACITY_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

#[derive(Debug, Default, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Q8_0FileReadStats {
    pub read_calls: u64,
    pub read_bytes: u64,
    pub cache_hits: u64,
    pub cache_hit_bytes: u64,
    pub cache_misses: u64,
    pub cache_miss_bytes: u64,
    pub cache_inserts: u64,
    pub cache_insert_bytes: u64,
    pub cache_evictions: u64,
    pub cache_evicted_bytes: u64,
    pub cache_merges: u64,
    pub cache_merged_bytes: u64,
    pub cache_decoded_scale_hits: u64,
    pub cache_decoded_scale_hit_blocks: u64,
    pub cache_entries: u64,
    pub cache_bytes: u64,
    pub cache_capacity_bytes: u64,
}

impl Q8_0FileReadStats {
    pub fn saturating_delta_since(self, start: Self) -> Self {
        Self {
            read_calls: self.read_calls.saturating_sub(start.read_calls),
            read_bytes: self.read_bytes.saturating_sub(start.read_bytes),
            cache_hits: self.cache_hits.saturating_sub(start.cache_hits),
            cache_hit_bytes: self.cache_hit_bytes.saturating_sub(start.cache_hit_bytes),
            cache_misses: self.cache_misses.saturating_sub(start.cache_misses),
            cache_miss_bytes: self.cache_miss_bytes.saturating_sub(start.cache_miss_bytes),
            cache_inserts: self.cache_inserts.saturating_sub(start.cache_inserts),
            cache_insert_bytes: self
                .cache_insert_bytes
                .saturating_sub(start.cache_insert_bytes),
            cache_evictions: self.cache_evictions.saturating_sub(start.cache_evictions),
            cache_evicted_bytes: self
                .cache_evicted_bytes
                .saturating_sub(start.cache_evicted_bytes),
            cache_merges: self.cache_merges.saturating_sub(start.cache_merges),
            cache_merged_bytes: self
                .cache_merged_bytes
                .saturating_sub(start.cache_merged_bytes),
            cache_decoded_scale_hits: self
                .cache_decoded_scale_hits
                .saturating_sub(start.cache_decoded_scale_hits),
            cache_decoded_scale_hit_blocks: self
                .cache_decoded_scale_hit_blocks
                .saturating_sub(start.cache_decoded_scale_hit_blocks),
            cache_entries: self.cache_entries,
            cache_bytes: self.cache_bytes,
            cache_capacity_bytes: self.cache_capacity_bytes,
        }
    }
}

pub(crate) fn record_q8_0_file_read(bytes: usize) {
    Q8_0_FILE_READ_CALLS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_READ_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn record_q8_file_cache_decoded_scale_reuse(blocks: usize) {
    if blocks == 0 {
        return;
    }
    Q8_0_FILE_CACHE_DECODED_SCALE_HITS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_DECODED_SCALE_HIT_BLOCKS.fetch_add(blocks as u64, Ordering::Relaxed);
}

/// Serializes every test that measures the process-global Q8 file-read counters
/// or mutates the process-global Q8 file cache.
///
/// Those tests span two modules (`tensor::tests` and `inference::tests`), so the
/// lock has to live beside the state it guards rather than inside either test
/// module. Concurrent runs corrupt each other's `saturating_delta_since` deltas —
/// typically an off-by-one read count — which made them intermittently red under
/// `cargo test`'s default parallelism.
///
/// Poisoning is deliberately ignored: one genuinely failing test must not cascade
/// into spurious failures across the rest of the family.
#[cfg(test)]
pub(crate) fn q8_stats_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn q8_0_file_read_stats() -> Q8_0FileReadStats {
    let cache_capacity_bytes = q8_file_cache_capacity_bytes();
    let (cache_entries, cache_bytes) = q8_file_cache_snapshot(cache_capacity_bytes);
    Q8_0FileReadStats {
        read_calls: Q8_0_FILE_READ_CALLS.load(Ordering::Relaxed),
        read_bytes: Q8_0_FILE_READ_BYTES.load(Ordering::Relaxed),
        cache_hits: Q8_0_FILE_CACHE_HITS.load(Ordering::Relaxed),
        cache_hit_bytes: Q8_0_FILE_CACHE_HIT_BYTES.load(Ordering::Relaxed),
        cache_misses: Q8_0_FILE_CACHE_MISSES.load(Ordering::Relaxed),
        cache_miss_bytes: Q8_0_FILE_CACHE_MISS_BYTES.load(Ordering::Relaxed),
        cache_inserts: Q8_0_FILE_CACHE_INSERTS.load(Ordering::Relaxed),
        cache_insert_bytes: Q8_0_FILE_CACHE_INSERT_BYTES.load(Ordering::Relaxed),
        cache_evictions: Q8_0_FILE_CACHE_EVICTIONS.load(Ordering::Relaxed),
        cache_evicted_bytes: Q8_0_FILE_CACHE_EVICTED_BYTES.load(Ordering::Relaxed),
        cache_merges: Q8_0_FILE_CACHE_MERGES.load(Ordering::Relaxed),
        cache_merged_bytes: Q8_0_FILE_CACHE_MERGED_BYTES.load(Ordering::Relaxed),
        cache_decoded_scale_hits: Q8_0_FILE_CACHE_DECODED_SCALE_HITS.load(Ordering::Relaxed),
        cache_decoded_scale_hit_blocks: Q8_0_FILE_CACHE_DECODED_SCALE_HIT_BLOCKS
            .load(Ordering::Relaxed),
        cache_entries,
        cache_bytes,
        cache_capacity_bytes: cache_capacity_bytes as u64,
    }
}

pub(crate) fn with_q8_file_cache_capacity_override<T>(
    capacity: Option<usize>,
    f: impl FnOnce() -> T,
) -> T {
    let Some(capacity) = capacity else {
        return f();
    };

    struct Q8FileCacheCapacityOverrideGuard {
        previous: Option<usize>,
    }

    impl Drop for Q8FileCacheCapacityOverrideGuard {
        fn drop(&mut self) {
            Q8_FILE_CACHE_CAPACITY_OVERRIDE.with(|cell| cell.set(self.previous));
            q8_file_cache_apply_capacity(q8_file_cache_capacity_bytes());
        }
    }

    let previous = Q8_FILE_CACHE_CAPACITY_OVERRIDE.with(|cell| {
        let previous = cell.get();
        cell.set(Some(capacity));
        previous
    });
    q8_file_cache_apply_capacity(q8_file_cache_capacity_bytes());
    let _guard = Q8FileCacheCapacityOverrideGuard { previous };
    f()
}

#[derive(Debug, Default)]
struct Q8FileCache {
    entries: Vec<Q8FileCacheEntry>,
    bytes: usize,
}

#[derive(Debug)]
struct Q8FileCacheEntry {
    path: PathBuf,
    offset: u64,
    bytes: Vec<u8>,
    decoded_q8_0_scales: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Q8FileCacheRead {
    Hit {
        decoded_scales_reused: bool,
        decoded_scale_hit_blocks: usize,
    },
    Missing {
        ranges: Vec<Q8FileCacheMissingRange>,
        decoded_scales_reused: bool,
        decoded_scale_hit_blocks: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Q8FileCacheMissingRange {
    out_start: usize,
    len: usize,
}

fn q8_file_cache_prepare_read(
    path: &Path,
    offset: u64,
    out: &mut [u8],
    mut cached_scales: Option<&mut [f32]>,
    capacity: usize,
) -> Q8FileCacheRead {
    let out_len = out.len();
    let mut decoded_scales_reused = cached_scales
        .as_ref()
        .and_then(|scales| scales.len().checked_mul(Q8_0_BLOCK_BYTES))
        .is_some_and(|scale_bytes| out_len == scale_bytes);
    let mut decoded_scale_hit_blocks = 0usize;
    debug_assert!(capacity > 0);
    let Some(request_end) = offset.checked_add(out_len as u64) else {
        record_q8_file_cache_miss(out_len);
        return q8_file_cache_missing_all(out_len);
    };
    let Some(cache) = Q8_FILE_CACHE.get() else {
        record_q8_file_cache_miss(out_len);
        return q8_file_cache_missing_all(out_len);
    };
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);

    let mut missing_ranges = vec![Q8FileCacheMissingRange {
        out_start: 0,
        len: out_len,
    }];
    let mut touched_indices = Vec::new();
    let mut hit_bytes = 0usize;

    for (idx, entry) in cache.entries.iter().enumerate().rev() {
        if entry.path != path {
            continue;
        }
        let Some(entry_end) = entry.offset.checked_add(entry.bytes.len() as u64) else {
            continue;
        };
        let overlap_start = entry.offset.max(offset);
        let overlap_end = entry_end.min(request_end);
        if overlap_start >= overlap_end {
            continue;
        }
        let overlap_out_start = (overlap_start - offset) as usize;
        let overlap_out_end = (overlap_end - offset) as usize;
        let mut next_missing = Vec::new();
        let mut touched = false;
        for missing in missing_ranges {
            let missing_end = missing.out_start + missing.len;
            let copy_start = missing.out_start.max(overlap_out_start);
            let copy_end = missing_end.min(overlap_out_end);
            if copy_start < copy_end {
                let entry_start = (offset + copy_start as u64 - entry.offset) as usize;
                let copy_len = copy_end - copy_start;
                out[copy_start..copy_end]
                    .copy_from_slice(&entry.bytes[entry_start..entry_start + copy_len]);
                if decoded_scales_reused {
                    let copied_scales = cached_scales.as_deref_mut().is_some_and(|scales| {
                        q8_file_cache_copy_decoded_scales(
                            entry,
                            entry_start,
                            copy_start,
                            copy_len,
                            scales,
                        )
                    });
                    if copied_scales {
                        decoded_scale_hit_blocks += copy_len / Q8_0_BLOCK_BYTES;
                    } else {
                        decoded_scales_reused = false;
                        decoded_scale_hit_blocks = 0;
                    }
                }
                hit_bytes += copy_len;
                touched = true;
                if missing.out_start < copy_start {
                    next_missing.push(Q8FileCacheMissingRange {
                        out_start: missing.out_start,
                        len: copy_start - missing.out_start,
                    });
                }
                if copy_end < missing_end {
                    next_missing.push(Q8FileCacheMissingRange {
                        out_start: copy_end,
                        len: missing_end - copy_end,
                    });
                }
            } else {
                next_missing.push(missing);
            }
        }
        missing_ranges = next_missing;
        if touched {
            touched_indices.push(idx);
        }
        if missing_ranges.is_empty() {
            break;
        }
    }

    if hit_bytes == 0 {
        record_q8_file_cache_miss(out_len);
        return q8_file_cache_missing_all(out_len);
    }
    q8_file_cache_mark_used(&mut cache, &touched_indices);
    Q8_0_FILE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_HIT_BYTES.fetch_add(hit_bytes as u64, Ordering::Relaxed);
    if missing_ranges.is_empty() {
        return Q8FileCacheRead::Hit {
            decoded_scales_reused,
            decoded_scale_hit_blocks,
        };
    }
    let miss_bytes = missing_ranges.iter().map(|range| range.len as u64).sum();
    Q8_0_FILE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_MISS_BYTES.fetch_add(miss_bytes, Ordering::Relaxed);
    Q8FileCacheRead::Missing {
        ranges: missing_ranges,
        decoded_scales_reused,
        decoded_scale_hit_blocks,
    }
}

fn q8_file_cache_missing_all(len: usize) -> Q8FileCacheRead {
    Q8FileCacheRead::Missing {
        ranges: vec![Q8FileCacheMissingRange { out_start: 0, len }],
        decoded_scales_reused: false,
        decoded_scale_hit_blocks: 0,
    }
}

fn q8_file_cache_mark_used(cache: &mut Q8FileCache, indices: &[usize]) {
    if indices.is_empty() {
        return;
    }
    let mut indices = indices.to_vec();
    indices.sort_unstable();
    indices.dedup();
    let mut entries = Vec::with_capacity(indices.len());
    for idx in indices.into_iter().rev() {
        entries.push(cache.entries.remove(idx));
    }
    entries.reverse();
    cache.entries.extend(entries);
}

fn q8_file_cache_copy_decoded_scales(
    entry: &Q8FileCacheEntry,
    entry_start: usize,
    out_start: usize,
    len: usize,
    out_scales: &mut [f32],
) -> bool {
    if !entry_start.is_multiple_of(Q8_0_BLOCK_BYTES)
        || !out_start.is_multiple_of(Q8_0_BLOCK_BYTES)
        || !len.is_multiple_of(Q8_0_BLOCK_BYTES)
    {
        return false;
    }
    let Some(entry_scales) = entry.decoded_q8_0_scales.as_ref() else {
        return false;
    };
    let entry_scale_start = entry_start / Q8_0_BLOCK_BYTES;
    let out_scale_start = out_start / Q8_0_BLOCK_BYTES;
    let scale_len = len / Q8_0_BLOCK_BYTES;
    let Some(entry_scale_end) = entry_scale_start.checked_add(scale_len) else {
        return false;
    };
    let Some(out_scale_end) = out_scale_start.checked_add(scale_len) else {
        return false;
    };
    if entry_scale_end > entry_scales.len() || out_scale_end > out_scales.len() {
        return false;
    }
    out_scales[out_scale_start..out_scale_end]
        .copy_from_slice(&entry_scales[entry_scale_start..entry_scale_end]);
    true
}

fn q8_file_cache_store_decoded_scales(path: &Path, offset: u64, scales: &[f32]) {
    let Some(byte_len) = scales.len().checked_mul(Q8_0_BLOCK_BYTES) else {
        return;
    };
    let capacity = q8_file_cache_capacity_bytes();
    if capacity == 0 {
        q8_file_cache_apply_capacity(0);
        return;
    }
    let Some(cache) = Q8_FILE_CACHE.get() else {
        return;
    };

    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    let Some(entry) = cache
        .entries
        .iter_mut()
        .rev()
        .find(|entry| q8_file_cache_entry_covers(entry, path, offset, byte_len))
    else {
        return;
    };
    if entry.path != path || !entry.bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return;
    }
    let Some(relative_start) = offset.checked_sub(entry.offset) else {
        return;
    };
    let Ok(relative_start) = usize::try_from(relative_start) else {
        return;
    };
    if !relative_start.is_multiple_of(Q8_0_BLOCK_BYTES) {
        return;
    }
    let scale_start = relative_start / Q8_0_BLOCK_BYTES;
    let Some(scale_end) = scale_start.checked_add(scales.len()) else {
        return;
    };
    let entry_scale_len = entry.bytes.len() / Q8_0_BLOCK_BYTES;
    if scale_end > entry_scale_len {
        return;
    }
    if entry
        .decoded_q8_0_scales
        .as_ref()
        .is_none_or(|entry_scales| entry_scales.len() != entry_scale_len)
    {
        let mut decoded_scales = vec![0.0_f32; entry_scale_len];
        decode_q8_0_scales_from_bytes(&entry.bytes, &mut decoded_scales);
        entry.decoded_q8_0_scales = Some(decoded_scales);
    }
    if let Some(entry_scales) = entry.decoded_q8_0_scales.as_mut() {
        entry_scales[scale_start..scale_end].copy_from_slice(scales);
    }
}

fn q8_file_cache_merge_decoded_scales(
    left: &Q8FileCacheEntry,
    right: &Q8FileCacheEntry,
    merged_len: usize,
    left_start: usize,
    right_start: usize,
) -> Option<Vec<f32>> {
    if !merged_len.is_multiple_of(Q8_0_BLOCK_BYTES)
        || !left_start.is_multiple_of(Q8_0_BLOCK_BYTES)
        || !right_start.is_multiple_of(Q8_0_BLOCK_BYTES)
        || !left.bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES)
        || !right.bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES)
    {
        return None;
    }
    let left_scales = left.decoded_q8_0_scales.as_ref()?;
    let right_scales = right.decoded_q8_0_scales.as_ref()?;
    let mut merged_scales = vec![0.0_f32; merged_len / Q8_0_BLOCK_BYTES];
    let left_scale_start = left_start / Q8_0_BLOCK_BYTES;
    let right_scale_start = right_start / Q8_0_BLOCK_BYTES;
    if left_scale_start + left_scales.len() > merged_scales.len()
        || right_scale_start + right_scales.len() > merged_scales.len()
    {
        return None;
    }
    merged_scales[left_scale_start..left_scale_start + left_scales.len()]
        .copy_from_slice(left_scales);
    // Let the newest read win for overlapping Q8 blocks, matching the byte merge.
    merged_scales[right_scale_start..right_scale_start + right_scales.len()]
        .copy_from_slice(right_scales);
    Some(merged_scales)
}

fn q8_file_cache_trim_decoded_scales(
    entry: &Q8FileCacheEntry,
    trim_start: usize,
    trim_end: usize,
) -> Option<Vec<f32>> {
    if !trim_start.is_multiple_of(Q8_0_BLOCK_BYTES) || !trim_end.is_multiple_of(Q8_0_BLOCK_BYTES) {
        return None;
    }
    let scales = entry.decoded_q8_0_scales.as_ref()?;
    let scale_start = trim_start / Q8_0_BLOCK_BYTES;
    let scale_end = trim_end / Q8_0_BLOCK_BYTES;
    Some(scales.get(scale_start..scale_end)?.to_vec())
}

fn decode_q8_0_scales_from_cache_bytes(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return None;
    }
    let mut scales = vec![0.0_f32; bytes.len() / Q8_0_BLOCK_BYTES];
    decode_q8_0_scales_from_bytes(bytes, &mut scales);
    Some(scales)
}

fn decode_q8_0_scales_from_bytes(bytes: &[u8], scales: &mut [f32]) {
    debug_assert_eq!(bytes.len(), scales.len() * Q8_0_BLOCK_BYTES);
    for (scale, block) in scales.iter_mut().zip(bytes.chunks_exact(Q8_0_BLOCK_BYTES)) {
        *scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    }
}

fn decode_q8_0_scales_from_byte_ranges(
    bytes: &[u8],
    ranges: &[Q8FileCacheMissingRange],
    scales: &mut [f32],
) -> bool {
    if bytes.len() != scales.len().saturating_mul(Q8_0_BLOCK_BYTES) {
        return false;
    }
    for range in ranges {
        if !range.out_start.is_multiple_of(Q8_0_BLOCK_BYTES)
            || !range.len.is_multiple_of(Q8_0_BLOCK_BYTES)
        {
            return false;
        }
        let Some(out_end) = range.out_start.checked_add(range.len) else {
            return false;
        };
        if out_end > bytes.len() {
            return false;
        }
        let scale_start = range.out_start / Q8_0_BLOCK_BYTES;
        let scale_len = range.len / Q8_0_BLOCK_BYTES;
        let Some(scale_end) = scale_start.checked_add(scale_len) else {
            return false;
        };
        if scale_end > scales.len() {
            return false;
        }
        decode_q8_0_scales_from_bytes(
            &bytes[range.out_start..out_end],
            &mut scales[scale_start..scale_end],
        );
    }
    true
}

#[cfg(test)]
fn q8_file_cache_get(path: &Path, offset: u64, out: &mut [u8]) -> bool {
    let capacity = q8_file_cache_capacity_bytes();
    if capacity == 0 {
        q8_file_cache_apply_capacity(0);
        return false;
    }
    let Some(cache) = Q8_FILE_CACHE.get() else {
        record_q8_file_cache_miss(out.len());
        return false;
    };
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    let Some(pos) = cache
        .entries
        .iter()
        .position(|entry| q8_file_cache_entry_covers(entry, path, offset, out.len()))
    else {
        record_q8_file_cache_miss(out.len());
        return false;
    };
    let entry = cache.entries.remove(pos);
    let start = (offset - entry.offset) as usize;
    out.copy_from_slice(&entry.bytes[start..start + out.len()]);
    cache.entries.push(entry);
    Q8_0_FILE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_HIT_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
    true
}

fn record_q8_file_cache_miss(bytes: usize) {
    Q8_0_FILE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    Q8_0_FILE_CACHE_MISS_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

fn q8_file_cache_entry_covers(
    entry: &Q8FileCacheEntry,
    path: &Path,
    offset: u64,
    len: usize,
) -> bool {
    let Some(request_end) = offset.checked_add(len as u64) else {
        return false;
    };
    let Some(entry_end) = entry.offset.checked_add(entry.bytes.len() as u64) else {
        return false;
    };
    entry.path == path && entry.offset <= offset && request_end <= entry_end
}

#[cfg(test)]
fn q8_file_cache_insert(path: PathBuf, offset: u64, bytes: &[u8]) {
    q8_file_cache_insert_with_decoded_scales(path, offset, bytes, None);
}

fn q8_file_cache_insert_with_decoded_scales(
    path: PathBuf,
    offset: u64,
    bytes: &[u8],
    decoded_q8_0_scales: Option<Vec<f32>>,
) {
    let capacity = q8_file_cache_capacity_bytes();
    if capacity == 0 || bytes.len() > capacity {
        if capacity == 0 {
            q8_file_cache_apply_capacity(0);
        }
        return;
    }
    let cache = Q8_FILE_CACHE.get_or_init(|| Mutex::new(Q8FileCache::default()));
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    cache.insert(path, offset, bytes.to_vec(), decoded_q8_0_scales, capacity);
}

fn q8_file_cache_capacity_bytes() -> usize {
    if let Some(capacity) = Q8_FILE_CACHE_CAPACITY_OVERRIDE.with(|cell| cell.get()) {
        return capacity;
    }
    env::var(Q8_FILE_CACHE_BYTES_ENV)
        .ok()
        .and_then(|value| parse_byte_count(&value))
        .unwrap_or(DEFAULT_Q8_FILE_CACHE_BYTES)
}

fn q8_file_cache_apply_capacity(capacity: usize) {
    if let Some(cache) = Q8_FILE_CACHE.get() {
        cache
            .lock()
            .expect("q8 file cache mutex poisoned")
            .apply_capacity(capacity);
    }
}

pub(crate) fn parse_byte_count_env(key: &str) -> Option<usize> {
    env::var(key)
        .ok()
        .and_then(|value| parse_byte_count(&value))
}

fn parse_byte_count(value: &str) -> Option<usize> {
    let normalized = value
        .trim()
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace() && *ch != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let digits_len = normalized
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if digits_len == 0 {
        return None;
    }

    let base = normalized[..digits_len].parse::<usize>().ok()?;
    let multiplier = match &normalized[digits_len..] {
        "" | "b" => 1usize,
        "k" | "kb" | "kib" => 1024usize,
        "m" | "mb" | "mib" => 1024usize.checked_mul(1024)?,
        "g" | "gb" | "gib" => 1024usize.checked_mul(1024)?.checked_mul(1024)?,
        _ => return None,
    };
    base.checked_mul(multiplier)
}

fn q8_file_cache_snapshot(capacity: usize) -> (u64, u64) {
    let Some(cache) = Q8_FILE_CACHE.get() else {
        return (0, 0);
    };
    let mut cache = cache.lock().expect("q8 file cache mutex poisoned");
    cache.apply_capacity(capacity);
    (cache.entries.len() as u64, cache.bytes as u64)
}

fn q8_file_cache_try_merge_entries(
    left: &Q8FileCacheEntry,
    right: &Q8FileCacheEntry,
    capacity: usize,
) -> Option<Q8FileCacheEntry> {
    if left.path != right.path {
        return None;
    }
    let left_end = left.offset.checked_add(left.bytes.len() as u64)?;
    let right_end = right.offset.checked_add(right.bytes.len() as u64)?;
    if left_end < right.offset || right_end < left.offset {
        return None;
    }
    let merged_offset = left.offset.min(right.offset);
    let merged_end = left_end.max(right_end);
    let merged_len = usize::try_from(merged_end.checked_sub(merged_offset)?).ok()?;

    let mut merged_bytes = vec![0u8; merged_len];
    let left_start = usize::try_from(left.offset.checked_sub(merged_offset)?).ok()?;
    merged_bytes[left_start..left_start + left.bytes.len()].copy_from_slice(&left.bytes);
    let right_start = usize::try_from(right.offset.checked_sub(merged_offset)?).ok()?;
    // Let the newest read win for overlapping bytes. The cache is only populated
    // from immutable GGUF payload reads, so equal bytes are expected; this keeps
    // the behavior deterministic for tests and any future synthetic cache probes.
    merged_bytes[right_start..right_start + right.bytes.len()].copy_from_slice(&right.bytes);

    let merged = Q8FileCacheEntry {
        path: left.path.clone(),
        offset: merged_offset,
        decoded_q8_0_scales: q8_file_cache_merge_decoded_scales(
            left,
            right,
            merged_len,
            left_start,
            right_start,
        ),
        bytes: merged_bytes,
    };
    Some(q8_file_cache_trim_merged_entry_to_capacity(
        merged,
        right.offset,
        right.bytes.len(),
        capacity,
    ))
}

fn q8_file_cache_trim_merged_entry_to_capacity(
    mut entry: Q8FileCacheEntry,
    newest_offset: u64,
    newest_len: usize,
    capacity: usize,
) -> Q8FileCacheEntry {
    if entry.bytes.len() <= capacity {
        return entry;
    }

    debug_assert!(newest_len <= capacity);
    let entry_end = entry.offset + entry.bytes.len() as u64;
    let newest_end = newest_offset + newest_len as u64;
    debug_assert!(entry.offset <= newest_offset);
    debug_assert!(newest_end <= entry_end);

    // Keep a contiguous cache window that retains the newest read. This matters for
    // sequential Q8 tensor streams where adjacent 32 MiB chunks can coalesce up to
    // the cache cap: when the next chunk arrives, dropping the whole old coalesced
    // entry would collapse a 320 MiB tail cache down to one chunk. Trimming preserves
    // the most recent contiguous window instead, which is the part most likely to be
    // reused by the next long-prefill chunk.
    let capacity_u64 = capacity as u64;
    let max_window_start = entry_end - capacity_u64;
    let lower_start = entry.offset.max(newest_end.saturating_sub(capacity_u64));
    let upper_start = newest_offset.min(max_window_start);
    let window_start = if lower_start <= upper_start {
        upper_start
    } else {
        lower_start.clamp(entry.offset, max_window_start)
    };
    let trim_start = (window_start - entry.offset) as usize;
    let trim_end = trim_start + capacity;
    entry.decoded_q8_0_scales = q8_file_cache_trim_decoded_scales(&entry, trim_start, trim_end);
    entry.bytes = entry.bytes[trim_start..trim_end].to_vec();
    entry.offset = window_start;
    entry
}

impl Q8FileCache {
    fn apply_capacity(&mut self, capacity: usize) {
        if capacity == 0 {
            self.entries.clear();
            self.bytes = 0;
            return;
        }
        while self.bytes > capacity {
            self.evict_oldest();
        }
    }

    fn insert(
        &mut self,
        path: PathBuf,
        offset: u64,
        bytes: Vec<u8>,
        decoded_q8_0_scales: Option<Vec<f32>>,
        capacity: usize,
    ) {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|entry| q8_file_cache_entry_covers(entry, &path, offset, bytes.len()))
        {
            let start = (offset - self.entries[pos].offset) as usize;
            if self.entries[pos].bytes[start..start + bytes.len()] == bytes {
                let entry = self.entries.remove(pos);
                self.entries.push(entry);
                return;
            }
        }

        let mut entry = Q8FileCacheEntry {
            path,
            offset,
            decoded_q8_0_scales,
            bytes,
        };
        let mut pos = 0usize;
        while pos < self.entries.len() {
            if let Some(merged) =
                q8_file_cache_try_merge_entries(&self.entries[pos], &entry, capacity)
            {
                let old = self.entries.remove(pos);
                self.bytes = self.bytes.saturating_sub(old.bytes.len());
                Q8_0_FILE_CACHE_MERGES.fetch_add(1, Ordering::Relaxed);
                Q8_0_FILE_CACHE_MERGED_BYTES
                    .fetch_add(merged.bytes.len() as u64, Ordering::Relaxed);
                entry = merged;
                pos = 0;
            } else {
                pos += 1;
            }
        }
        self.bytes = self.bytes.saturating_add(entry.bytes.len());
        Q8_0_FILE_CACHE_INSERTS.fetch_add(1, Ordering::Relaxed);
        Q8_0_FILE_CACHE_INSERT_BYTES.fetch_add(entry.bytes.len() as u64, Ordering::Relaxed);
        self.entries.push(entry);
        while self.bytes > capacity {
            self.evict_oldest();
        }
    }

    fn evict_oldest(&mut self) {
        if self.entries.is_empty() {
            self.bytes = 0;
            return;
        }
        let entry = self.entries.remove(0);
        self.bytes = self.bytes.saturating_sub(entry.bytes.len());
        Q8_0_FILE_CACHE_EVICTIONS.fetch_add(1, Ordering::Relaxed);
        Q8_0_FILE_CACHE_EVICTED_BYTES.fetch_add(entry.bytes.len() as u64, Ordering::Relaxed);
    }
}

pub(crate) fn should_parallelize_linear_output(output_width: usize) -> bool {
    parallel_linear_enabled()
        && output_width >= parallel_linear_min_outputs()
        && rayon::current_num_threads() > 1
}

fn parallel_linear_enabled() -> bool {
    // Read once per process (non-test): `should_parallelize_linear_output`
    // runs on every elementwise/matmul call in the decode hot loop, and env
    // reads allocate on Windows. Tests keep the uncached read.
    fn uncached() -> bool {
        match env::var("CAMELID_PARALLEL_LINEAR") {
            Ok(value) => matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes" | "enabled"
            ),
            Err(_) => false,
        }
    }
    #[cfg(test)]
    {
        uncached()
    }
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(uncached)
    }
}

fn parallel_linear_min_outputs() -> usize {
    fn uncached() -> usize {
        env::var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_PARALLEL_LINEAR_MIN_OUTPUTS)
    }
    #[cfg(test)]
    {
        uncached()
    }
    #[cfg(not(test))]
    {
        static MIN_OUTPUTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *MIN_OUTPUTS.get_or_init(uncached)
    }
}

pub struct TensorStore {
    path: PathBuf,
    descriptors: HashMap<String, GgufTensorDescriptor>,
}

impl TensorStore {
    pub fn open(path: impl AsRef<Path>, gguf: &GgufFile) -> Self {
        let descriptors = gguf
            .tensors
            .iter()
            .cloned()
            .map(|desc| (desc.name.clone(), desc))
            .collect();
        Self {
            path: path.as_ref().to_path_buf(),
            descriptors,
        }
    }

    pub fn descriptor(&self, name: &str) -> Result<&GgufTensorDescriptor> {
        self.descriptors
            .get(name)
            .ok_or_else(|| BackendError::TensorNotFound(name.to_string()))
    }

    /// Path backing this tensor store. Crate-internal format converters use this
    /// to copy bounded tensor ranges without materializing a whole (potentially
    /// multi-gigabyte) tensor in RAM.
    pub(crate) fn source_path(&self) -> &Path {
        &self.path
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let desc = self.descriptor(name)?;
        let len = usize::try_from(desc.n_bytes).map_err(|_| {
            BackendError::InvalidTensorData(format!("tensor {name} byte length does not fit usize"))
        })?;
        let mut file = File::open(&self.path).map_err(|source| BackendError::Io {
            path: self.path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(desc.absolute_offset))
            .map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)
            .map_err(|source| BackendError::Io {
                path: self.path.clone(),
                source,
            })?;
        Ok(bytes)
    }

    pub fn load_q8_0_blocks(&self, name: &str) -> Result<Q8_0TensorBlocks> {
        let desc = self.descriptor(name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} has storage type {:?}; q8_0 block-only load requires Q8_0",
                desc.tensor_type
            )));
        }
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let blocks = decode_q8_0_blocks(name, &bytes, expected_elements)?;
        Ok(Q8_0TensorBlocks {
            name: name.to_string(),
            shape,
            blocks,
        })
    }

    pub fn load_q8_0_file_backed_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return self.load_cpu_f32(name);
        }
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        self.load_q8_0_file_backed_tensor(name)
    }

    pub fn load_q8_0_block_backed_linear(&self, name: &str) -> Result<CpuTensor> {
        self.load_q8_0_block_backed_linear_as(name, name)
    }

    /// Load a Q8_0 tensor of any rank into resident quantized blocks.
    ///
    /// Unlike `load_q8_0_block_backed_linear`, this never materializes f32 and
    /// intentionally rejects non-Q8 storage. It exists for rank-3 MoE expert
    /// packs, where keeping the quantized bytes resident removes repeated disk
    /// reads without multiplying memory by four.
    pub fn load_q8_0_block_backed_tensor(&self, name: &str) -> Result<CpuTensor> {
        let tensor = self.load_q8_0_blocks(name)?;
        let len = tensor.blocks.len();
        CpuTensor::from_q8_0_shared_blocks(
            tensor.name,
            tensor.shape,
            std::sync::Arc::new(tensor.blocks),
            0,
            len,
        )
    }

    /// Fast-load: read the tensor's wire-format bytes once into a page-aligned
    /// allocation (page cache enabled, no decode) that the Metal stack wraps with
    /// an offset-0 NoCopy buffer — the only resident copy of the weight. The
    /// tensor also carries file backing so CPU fallback paths stay correct.
    pub fn load_q8_0_wire_pages_linear(&self, name: &str) -> Result<CpuTensor> {
        self.load_q8_0_wire_pages_linear_as(name, name)
    }

    pub fn load_q8_0_wire_pages_linear_as(
        &self,
        source_name: &str,
        tensor_name: &str,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(source_name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if desc.tensor_type != GgufTensorType::Q8_0 || shape.dims.len() != 2 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let expected_elements = shape.element_count()?;
        if expected_elements % 32 != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {source_name} Q8_0 element count {expected_elements} is not block aligned"
            )));
        }
        let wire_bytes = expected_elements / Q8_0_BLOCK_VALUES * Q8_0_BLOCK_BYTES;
        let mut tensor = self.load_q8_0_file_backed_tensor_as(source_name, tensor_name)?;
        let file = File::open(&self.path).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "wire pages open failed for {}: {err}",
                self.path.display()
            ))
        })?;
        tensor.q8_0_wire_pages = Some(crate::wire_mmap::WirePages::read_from_file(
            &file,
            desc.absolute_offset,
            wire_bytes,
        )?);
        Ok(tensor)
    }

    pub fn load_q8_0_block_backed_linear_as(
        &self,
        source_name: &str,
        tensor_name: &str,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(source_name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if shape.dims.len() != 2 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let expected_elements = shape.element_count()?;
        let bytes = self.tensor_bytes(source_name)?;
        if let Some(Q8_0RuntimeStorage::PackedRows4(packed)) =
            q8_0_runtime_packed_rows4_for_tensor(tensor_name, &shape, &bytes)?
        {
            return Ok(CpuTensor::q8_0_runtime_packed_rows4_linear(
                tensor_name,
                shape,
                packed,
            ));
        }
        let blocks = decode_q8_0_blocks(source_name, &bytes, expected_elements)?;
        CpuTensor::from_q8_0_blocks(tensor_name, shape, blocks)
    }

    /// Load a 2-D Q1_0 tensor by LOSSLESSLY re-encoding it into Q8_0 blocks.
    ///
    /// This is a re-encoding, NOT a re-quantization: Q1_0's representable set is
    /// exactly `{-d, +d}`, and a Q8_0 block represents both values EXACTLY as
    /// `qs = ±1` against the same scale `d`. `QK1_0 = 128` is `4 * 32`, so each
    /// Q1_0 block tiles into exactly 4 Q8_0 blocks that share its scale, with no
    /// remainder and no rounding anywhere. `q1_0_transcode_is_bit_exact` pins that
    /// the resulting blocks decode to values bit-identical to
    /// [`decode_q1_0_tensor`] on the same wire bytes.
    ///
    /// WHY: every GPU-resident lane in this engine is gated on
    /// `source_type == Q8_0` plus available blocks (`metal_resident_weight_eligible`).
    /// Re-encoding at load puts Q1_0 files on that lane with NO new Metal/CUDA
    /// kernels, and avoids the f32 materialization that the plain CPU loader would
    /// pay (a 1.7B model is ~6.9 GB decoded to f32 vs ~1.9 GB as blocks).
    ///
    /// COST, stated plainly: Q8_0 blocks are 36 bytes per 32 values (9 bpw) against
    /// Q1_0's 1.125 bpw, so this expands the weights ~8x in RAM and gives up the
    /// footprint that is the entire point of a 1-bit format. It buys correctness and
    /// the GPU lane today; a native Q1_0 resident kernel is what actually keeps the
    /// 1.125 bpw at runtime.
    ///
    /// TRUTHFULNESS: the returned tensor reports `source_type = Q8_0` because that
    /// describes its RUNTIME BLOCK REPRESENTATION, which is what every downstream
    /// consumer keys off. The FILE's quantization is unaffected and still reports
    /// Q1_0 everywhere it is disclosed — `general.file_type` (40) drives the receipt
    /// and execution-plan labels, and those read the GGUF descriptors, not this
    /// tensor. Nothing here may re-read the file bytes as Q8_0: the blocks are
    /// materialized in RAM and `q8_0_file_backing` / wire-page paths stay `None`
    /// (they would misread 18-byte Q1_0 blocks as 34-byte Q8_0 wire blocks).
    pub fn load_q1_0_as_q8_0_blocks_linear(&self, name: &str) -> Result<CpuTensor> {
        self.load_q1_0_as_q8_0_blocks_linear_as(name, name)
    }

    pub fn load_q1_0_as_q8_0_blocks_linear_as(
        &self,
        source_name: &str,
        tensor_name: &str,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(source_name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if desc.tensor_type != GgufTensorType::Q1_0 || shape.dims.len() != 2 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let expected_elements = shape.element_count()?;
        let bytes = self.tensor_bytes(source_name)?;
        let blocks = q1_0_to_q8_0_blocks(source_name, &bytes, expected_elements)?;
        CpuTensor::from_q8_0_blocks(tensor_name, shape, blocks)
    }

    /// Load a 2-D K-quant (Q4_K / Q6_K) linear retaining ONLY the raw super-block wire
    /// bytes — NO f32 `data` materialization. This mirrors `from_q8_0_blocks` (which
    /// leaves `data` empty): an 8B model fully decoded to f32 is ~32 GB and OOMs a
    /// 16 GB box, so the GPU-resident decode path (which reads the wire bytes via
    /// `q4k_gemv`/`q6k_gemv`) must not pay that cost. The tensor carries empty `data`,
    /// so the CPU dense-matmul fallback can NOT run it — callers take this path only
    /// when the resident path will own the forward. Q8_0 / non-K-quant / non-2D tensors
    /// fall back to the f32 loader unchanged.
    pub fn load_kquant_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let is_kquant = matches!(
            desc.tensor_type,
            GgufTensorType::Q4K
                | GgufTensorType::Q5K
                | GgufTensorType::Q6K
                | GgufTensorType::Q2K
                | GgufTensorType::Q3K
        );
        if !is_kquant || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let bytes = self.tensor_bytes(name)?;
        let mut q4_k_wire_bytes = None;
        let mut q5_k_wire_bytes = None;
        let mut q6_k_wire_bytes = None;
        let mut q2_k_wire_bytes = None;
        let mut q3_k_wire_bytes = None;
        match desc.tensor_type {
            GgufTensorType::Q4K => q4_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec())),
            GgufTensorType::Q5K => q5_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec())),
            GgufTensorType::Q6K => q6_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec())),
            GgufTensorType::Q2K => q2_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec())),
            GgufTensorType::Q3K => q3_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec())),
            _ => unreachable!(),
        }
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes,
            q6_k_wire_bytes,
            q2_k_wire_bytes,
            q3_k_wire_bytes,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Metal no-copy variant of [`load_kquant_wire_linear`]. The raw GGUF
    /// super-blocks are read once into page-aligned storage which Metal can
    /// wrap with `newBufferWithBytesNoCopy`; no second `Vec` or GPU upload is
    /// retained. CPU consumers use [`CpuTensor::q4_k_wire`] and
    /// [`CpuTensor::q6_k_wire`] to read the same pages when a resident path
    /// falls back.
    pub fn load_kquant_wire_pages_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let block_bytes = match desc.tensor_type {
            GgufTensorType::Q4K => 144,
            GgufTensorType::Q6K => 210,
            // The first Metal resident campaign covers the mixed Q4_K_M
            // pair. Other K-quants retain their existing Vec-backed loader
            // until they have device kernels and CPU page-backed consumers.
            GgufTensorType::Q2K | GgufTensorType::Q3K | GgufTensorType::Q5K => {
                return self.load_kquant_wire_linear(name);
            }
            _ => return self.load_cpu_f32(name),
        };
        if shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let elements = shape.element_count()?;
        const KQUANT_BLOCK_VALUES: usize = 256;
        if !elements.is_multiple_of(KQUANT_BLOCK_VALUES) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} {:?} element count {elements} is not 256-value super-block aligned",
                desc.tensor_type
            )));
        }
        let wire_bytes = elements / KQUANT_BLOCK_VALUES * block_bytes;
        let file = File::open(&self.path).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "K-quant wire pages open failed for {}: {err}",
                self.path.display()
            ))
        })?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: Some(crate::wire_mmap::WirePages::read_from_file(
                &file,
                desc.absolute_offset,
                wire_bytes,
            )?),
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Load a TQ2_0 (ternary) 2-D linear by retaining its raw wire bytes only — no f32
    /// materialisation. The CPU ternary block-dot streams these directly. Mirrors
    /// `load_kquant_wire_linear`. Falls back to f32 for non-TQ2_0 / non-2-D tensors.
    pub fn load_tq2_0_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if !matches!(desc.tensor_type, GgufTensorType::Tq2_0) || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let bytes = self.tensor_bytes(name)?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: Some(std::sync::Arc::new(bytes.to_vec())),
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Retain a Prism Q1_0/Q2_0 rank-2 tensor exactly as packed in the GGUF.
    /// This is the native Metal/CUDA handoff and deliberately carries no dense
    /// or Q8 representation, so a backend cannot accidentally claim the low-bit
    /// footprint while executing expanded weights.
    pub fn load_prism_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if !matches!(
            desc.tensor_type,
            GgufTensorType::Q1_0
                | GgufTensorType::Q2_0G64
                | GgufTensorType::Q2_0G128
                | GgufTensorType::Pq2_0
        ) || shape.dims.len() != 2
        {
            return self.load_cpu_f32(name);
        }
        let file = File::open(&self.path).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "Prism wire pages open failed for {}: {err}",
                self.path.display()
            ))
        })?;
        let wire_bytes = usize::try_from(desc.n_bytes).map_err(|_| {
            BackendError::InvalidTensorData(format!(
                "tensor {name} Prism wire length {} does not fit usize",
                desc.n_bytes
            ))
        })?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: Some(crate::wire_mmap::WirePages::read_from_file(
                &file,
                desc.absolute_offset,
                wire_bytes,
            )?),
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data: Vec::new(),
        })
    }

    /// Load an IQ4_XS (i-quant) 2-D linear by retaining its raw wire bytes only — no f32
    /// materialisation. The CPU i-quant block-dot streams these directly. Mirrors
    /// `load_tq2_0_wire_linear`. Falls back to f32 for non-IQ4_XS / non-2-D tensors.
    pub fn load_iq4_xs_wire_linear(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        if !matches!(desc.tensor_type, GgufTensorType::IQ4XS) || shape.dims.len() != 2 {
            return self.load_cpu_f32(name);
        }
        let bytes = self.tensor_bytes(name)?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks: None,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes: None,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes: None,
            q2_k_wire_bytes: None,
            q3_k_wire_bytes: None,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: Some(std::sync::Arc::new(bytes.to_vec())),
            data: Vec::new(),
        })
    }

    pub fn load_q8_0_split_file_backed_tensor(
        &self,
        name: impl Into<String>,
        dims: Vec<usize>,
        experts: &[GgufTensorDescriptor],
    ) -> Result<CpuTensor> {
        let name = name.into();
        let shape = TensorShape { dims };
        let expected_elements = shape.element_count()?;
        if expected_elements % 32 != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "split tensor {name} Q8_0 element count {expected_elements} is not block aligned"
            )));
        }
        let expert_count = experts.len();
        if expert_count == 0 {
            return Err(BackendError::InvalidTensorData(
                "split MoE tensor requires at least one expert".to_string(),
            ));
        }
        let per_expert_elements = expected_elements / expert_count;
        if !per_expert_elements.is_multiple_of(32) {
            return Err(BackendError::InvalidTensorData(
                "split MoE expert Q8_0 element count is not block aligned".to_string(),
            ));
        }
        let mut backings = Vec::with_capacity(expert_count);
        for desc in experts {
            if desc.tensor_type != GgufTensorType::Q8_0 {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "split MoE tensor {} has storage type {:?}; lazy split experts require Q8_0",
                    desc.name, desc.tensor_type
                )));
            }
            let expert_shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
            if expert_shape.element_count()? != per_expert_elements {
                return Err(BackendError::InvalidTensorData(format!(
                    "split MoE tensor {} has {} elements, expected {per_expert_elements}",
                    desc.name,
                    expert_shape.element_count()?
                )));
            }
            backings.push(Q8_0FileBacking::new(
                self.path.clone(),
                desc.absolute_offset,
                per_expert_elements / 32,
            ));
        }
        Ok(CpuTensor::q8_0_split_file_backed_tensor(
            name, shape, backings,
        ))
    }

    /// Join individually named Q8_0 experts into one resident rank-3 tensor.
    ///
    /// GGUF split-expert files store each expert contiguously. Appending their
    /// decoded block records in descriptor order therefore produces the same
    /// expert-major layout as a merged expert tensor.
    pub fn load_q8_0_split_block_backed_tensor(
        &self,
        name: impl Into<String>,
        dims: Vec<usize>,
        experts: &[GgufTensorDescriptor],
    ) -> Result<CpuTensor> {
        let name = name.into();
        let shape = TensorShape { dims };
        let expected_elements = shape.element_count()?;
        if expected_elements % Q8_0_BLOCK_VALUES != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "split tensor {name} Q8_0 element count {expected_elements} is not block aligned"
            )));
        }
        let expert_count = experts.len();
        if expert_count == 0 {
            return Err(BackendError::InvalidTensorData(
                "split MoE tensor requires at least one expert".to_string(),
            ));
        }
        if !expected_elements.is_multiple_of(expert_count) {
            return Err(BackendError::InvalidTensorData(format!(
                "split tensor {name} element count {expected_elements} is not divisible by {expert_count} experts"
            )));
        }
        let per_expert_elements = expected_elements / expert_count;
        let mut blocks = Vec::with_capacity(expected_elements / Q8_0_BLOCK_VALUES);
        for desc in experts {
            if desc.tensor_type != GgufTensorType::Q8_0 {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "split MoE tensor {} has storage type {:?}; resident experts require Q8_0",
                    desc.name, desc.tensor_type
                )));
            }
            let expert = self.load_q8_0_blocks(&desc.name)?;
            let actual_elements = expert.shape.element_count()?;
            if actual_elements != per_expert_elements {
                return Err(BackendError::InvalidTensorData(format!(
                    "split MoE tensor {} has {actual_elements} elements, expected {per_expert_elements}",
                    desc.name
                )));
            }
            blocks.extend(expert.blocks);
        }
        if blocks.len() != expected_elements / Q8_0_BLOCK_VALUES {
            return Err(BackendError::InvalidTensorData(format!(
                "split tensor {name} decoded {} Q8_0 blocks, expected {}",
                blocks.len(),
                expected_elements / Q8_0_BLOCK_VALUES
            )));
        }
        let len = blocks.len();
        CpuTensor::from_q8_0_shared_blocks(name, shape, std::sync::Arc::new(blocks), 0, len)
    }

    pub fn load_q8_0_file_backed_tensor(&self, name: &str) -> Result<CpuTensor> {
        self.load_q8_0_file_backed_tensor_as(name, name)
    }

    /// Load a rank-3 Q4_0/Q4_1 MoE expert pack as a descriptor-only streamed
    /// tensor. Refuses other sources rather than falling back to f32
    /// materialization: a fine-grained MoE pack materialized f32 is tens of
    /// GiB and must never happen silently.
    pub fn load_q4_0_file_backed_expert_tensor(&self, name: &str) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let wire_bytes_per_block = match desc.tensor_type {
            GgufTensorType::Q4_0 => Q4_0_BLOCK_BYTES,
            GgufTensorType::Q4_1 => Q4_1_BLOCK_BYTES,
            other => {
                return Err(BackendError::InvalidTensorData(format!(
                    "tensor {name} is {other:?}, not Q4_0/Q4_1; the streamed expert loader refuses fallback"
                )));
            }
        };
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        if expected_elements % 32 != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {name} nibble-block element count {expected_elements} is not block aligned"
            )));
        }
        Ok(CpuTensor::q4_0_file_backed_experts(
            name,
            shape,
            Q4_0FileBacking::new(
                self.path.clone(),
                desc.absolute_offset,
                expected_elements / 32,
                wire_bytes_per_block,
            ),
            desc.tensor_type,
        ))
    }

    /// Load Q8_0 as descriptor-only file backing, even when runtime repack
    /// flags are enabled. Callers select this path specifically to bound host
    /// residency (the dense lazy policy and the Metal wire-pages loader), so
    /// silently attaching PackedRows4 sidecars would violate that contract.
    pub fn load_q8_0_file_backed_tensor_as(
        &self,
        source_name: &str,
        tensor_name: &str,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(source_name)?.clone();
        if desc.tensor_type != GgufTensorType::Q8_0 {
            let mut tensor = self.load_cpu_f32(source_name)?;
            tensor.name = tensor_name.to_string();
            return Ok(tensor);
        }
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        if expected_elements % 32 != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {source_name} Q8_0 element count {expected_elements} is not block aligned"
            )));
        }
        Ok(CpuTensor::q8_0_file_backed_linear(
            tensor_name,
            shape,
            Q8_0FileBacking::new(
                self.path.clone(),
                desc.absolute_offset,
                expected_elements / 32,
            ),
        ))
    }

    pub fn load_cpu_f32(&self, name: &str) -> Result<CpuTensor> {
        let retain_q8_0_blocks = matches!(
            env::var(RETAIN_Q8_BLOCKS_ENV).as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        );
        self.load_cpu_f32_with_q8_0_block_retention(name, retain_q8_0_blocks)
    }

    pub fn load_cpu_f32_with_q8_0_block_retention(
        &self,
        name: &str,
        retain_q8_0_blocks: bool,
    ) -> Result<CpuTensor> {
        let desc = self.descriptor(name)?.clone();
        let bytes = self.tensor_bytes(name)?;
        let shape = TensorShape::from_gguf_dims(&desc.dimensions)?;
        let expected_elements = shape.element_count()?;
        let mut q8_0_blocks = None;
        let mut q8_0_file_backing = None;
        // Retain the raw K-quant super-block wire bytes so the GPU-resident decode path
        // can repack/feed them (q4k_gemv reads a SoA repack; q6k_gemv reads the wire
        // bytes directly). The CPU f32 `data` is still decoded below for the CPU path.
        let mut q4_k_wire_bytes = None;
        let mut q6_k_wire_bytes = None;
        let mut q2_k_wire_bytes = None;
        let mut q3_k_wire_bytes = None;
        let data = match desc.tensor_type {
            GgufTensorType::F32 => decode_f32_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::F16 => decode_f16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::BF16 => decode_bf16_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q8_0 => {
                let decoded = decode_q8_0_tensor(name, &bytes, expected_elements)?;
                if retain_q8_0_blocks {
                    q8_0_blocks = Some(decode_q8_0_blocks(name, &bytes, expected_elements)?);
                } else {
                    q8_0_file_backing = Some(Q8_0FileBacking::new(
                        self.path.clone(),
                        desc.absolute_offset,
                        expected_elements / 32,
                    ));
                }
                decoded
            }
            GgufTensorType::Q4_0 => decode_q4_0_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q4_1 => decode_q4_1_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q5_0 => decode_q5_0_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q5_1 => decode_q5_1_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q2K => {
                // 84-byte super-block wire layout — fed straight to the resident q2k GEMV.
                q2_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec()));
                decode_q2_k_tensor(name, &bytes, expected_elements)?
            }
            GgufTensorType::Q3K => {
                // 110-byte super-block wire layout — fed straight to the resident q3k GEMV.
                q3_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec()));
                decode_q3_k_tensor(name, &bytes, expected_elements)?
            }
            GgufTensorType::Q4K => {
                // The GGUF bytes ARE the 144-byte super-block wire layout the resident
                // q4k path repacks; keep them alongside the decoded f32.
                q4_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec()));
                decode_q4_k_tensor(name, &bytes, expected_elements)?
            }
            GgufTensorType::Q5K => decode_q5_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q6K => {
                // 210-byte super-block wire layout — fed straight to the resident q6k GEMV.
                q6_k_wire_bytes = Some(std::sync::Arc::new(bytes.to_vec()));
                decode_q6_k_tensor(name, &bytes, expected_elements)?
            }
            GgufTensorType::Q8K => decode_q8_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::IQ4NL => decode_iq4_nl_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::IQ4XS => decode_iq4_xs_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Tq1_0 => decode_tq1_0_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Tq2_0 => decode_tq2_0_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::I2S => decode_i2_s_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q1_0 => decode_q1_0_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q2_0G64
            | GgufTensorType::Q2_0G128
            | GgufTensorType::Pq2_0 => {
                decode_q2_0_tensor(name, &bytes, expected_elements, desc.tensor_type)?
            }
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} has unsupported storage type {other:?}; supported for CPU f32 load: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, I2_S, Q1_0, Q2_0G64, Q2_0G128, PQ2_0"
                )))
            }
        };
        let q8_0_packed_rows4_4x4 = q8_0_packed_rows4_for_shape(
            name,
            &shape,
            q8_0_blocks.as_deref(),
            Q8_0PackedRows4Interleave::I4,
        )?;
        let q8_0_packed_rows4_4x8 = q8_0_packed_rows4_for_shape(
            name,
            &shape,
            q8_0_blocks.as_deref(),
            Q8_0PackedRows4Interleave::I8,
        )?;
        Ok(CpuTensor {
            name: name.to_string(),
            shape,
            dtype: RuntimeDType::F32,
            source_type: Some(desc.tensor_type),
            q8_0_blocks,
            q8_0_shared_blocks: None,
            q8_0_packed_rows4_4x4,
            q8_0_packed_rows4_4x8,
            q8_0_runtime_storage: None,
            q8_0_file_backing,
            q8_0_wire_mmap: None,
            q8_0_wire_pages: None,
            kquant_wire_pages: None,
            q8_0_split_file_backing: None,
            q4_0_file_backing: None,
            q4_k_wire_bytes,
            q4_k_repack8: Q4KRepack8Cell::default(),
            q5_k_wire_bytes: None,
            q6_k_wire_bytes,
            q2_k_wire_bytes,
            q3_k_wire_bytes,
            tq2_0_wire_bytes: None,
            iq4_xs_wire_bytes: None,
            data,
        })
    }
}

/// Ghost (layer-streaming) mode support: materialize a `CpuTensor` from raw GGUF tensor
/// bytes that were already read from a `.cghost` layer group. 2-D Q8_0 linears come back as
/// plain RAM-resident blocks (the same storage the block-backed loader produces, so the
/// existing CPU forward path runs unchanged); float tensors decode to f32. Ghost v1 supports
/// the tensor types dense Llama models actually ship — anything else is a loud error, never
/// a silent fallback.
pub fn cpu_tensor_from_gguf_bytes(
    name: &str,
    tensor_type: GgufTensorType,
    dims: &[u64],
    bytes: &[u8],
) -> Result<CpuTensor> {
    let shape = TensorShape::from_gguf_dims(dims)?;
    let expected_elements = shape.element_count()?;
    match tensor_type {
        GgufTensorType::F32 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_f32_tensor(name, bytes, expected_elements)?,
        ),
        GgufTensorType::F16 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_f16_tensor(name, bytes, expected_elements)?,
        ),
        GgufTensorType::BF16 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_bf16_tensor(name, bytes, expected_elements)?,
        ),
        GgufTensorType::Q8_0 if shape.dims.len() == 2 => {
            let blocks = decode_q8_0_blocks(name, bytes, expected_elements)?;
            CpuTensor::from_q8_0_blocks(name, shape, blocks)
        }
        GgufTensorType::Q8_0 => CpuTensor::from_f32(
            name,
            shape.dims.clone(),
            decode_q8_0_tensor(name, bytes, expected_elements)?,
        ),
        other => Err(BackendError::UnsupportedTensorType(format!(
            "tensor {name} has storage type {other:?}; ghost v1 supports F32, F16, BF16, Q8_0"
        ))),
    }
}

pub(crate) fn decode_f32_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 4 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} f32 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 4
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("exact chunk length")))
        .collect())
}

pub(crate) fn decode_f16_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} f16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f16_bits_to_f32(u16::from_le_bytes(
                chunk.try_into().expect("exact chunk length"),
            ))
        })
        .collect())
}

pub(crate) fn decode_bf16_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if bytes.len() != expected_elements * 2 {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} bf16 byte length {} does not match expected {}",
            bytes.len(),
            expected_elements * 2
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            f32::from_bits(
                u32::from(u16::from_le_bytes(
                    chunk.try_into().expect("exact chunk length"),
                )) << 16,
            )
        })
        .collect())
}

pub(crate) fn decode_q8_0_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q8_0_blocks(name, bytes, expected_elements)?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        for q in block.quants {
            out.push(block.scale * f32::from(q));
        }
    }
    Ok(out)
}

fn decode_q8_0_blocks(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<Q8_0Block>> {
    const BLOCK_VALUES: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if !expected_elements.is_multiple_of(BLOCK_VALUES) {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} q8_0 element count {expected_elements} is not divisible by {BLOCK_VALUES}"
        )));
    }
    let expected_bytes = expected_elements / BLOCK_VALUES * BLOCK_BYTES;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "tensor {name} q8_0 byte length {} does not match expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut blocks = Vec::with_capacity(expected_elements / BLOCK_VALUES);
    for block in bytes.chunks_exact(BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut quants = [0_i8; BLOCK_VALUES];
        for (idx, q) in block[2..].iter().enumerate() {
            quants[idx] = *q as i8;
        }
        blocks.push(Q8_0Block { scale, quants });
    }
    Ok(blocks)
}

/// f32 -> IEEE f16 bits with round-to-nearest-even — the same conversion the
/// reference runtime's f16 KV cache applies on store (ARM `vcvt` semantics).
/// Kept for cache-precision experiments (the gemma4 KV cache is f32; parity
/// oracles pin the comparator to the plain-f32 path instead — see
/// `gemma4_runtime`).
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf / NaN (preserve a NaN payload bit so NaN stays NaN).
        let nan = if frac != 0 {
            0x0200 | ((frac >> 13) as u16 & 0x03ff)
        } else {
            0
        };
        return sign | 0x7c00 | nan;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00; // overflow -> +/-inf
    }
    if half_exp <= 0 {
        // Subnormal half (or zero): shift the implicit-1 mantissa down.
        if half_exp < -10 {
            return sign; // underflow -> +/-0
        }
        let mant = frac | 0x0080_0000;
        let shift = (14 - half_exp) as u32; // 14..=24
        let mut half = (mant >> shift) as u16;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = ((half_exp as u32) << 10) | (frac >> 13);
    let rem = frac & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1; // mantissa carry propagates into the exponent correctly
    }
    sign | half as u16
}

/// Round an f32 through f16 storage precision (f32 → f16 bits → f32) — the
/// effective value of an `ggml_half`-stored scale.
pub(crate) fn f16_round(value: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(value))
}

pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);

    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = u32::try_from(e + 127).expect("subnormal f16 exponent in range");
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

// Quantization Constants
pub const Q8_BLOCK_SIZE: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 2 + (Q8_BLOCK_SIZE / 2);
pub const Q4_1_BLOCK_BYTES: usize = 4 + (Q8_BLOCK_SIZE / 2);
pub const Q5_0_BLOCK_BYTES: usize = 2 + 4 + (Q8_BLOCK_SIZE / 2);
pub const Q5_1_BLOCK_BYTES: usize = 4 + 4 + (Q8_BLOCK_SIZE / 2);
pub const QK_K_BLOCK_SIZE: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = 16 + 64 + 4;
pub const Q3_K_BLOCK_BYTES: usize = 32 + 64 + 12 + 2;
pub const Q4_K_BLOCK_BYTES: usize = 4 + 12 + 128;
pub const Q5_K_BLOCK_BYTES: usize = 4 + 12 + 32 + 128;
pub const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
pub const Q8_K_BLOCK_BYTES: usize = 292;
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
// block_iq4_xs = f16 d(2) + scales_h u16(2) + scales_l[QK_K/64]=4 + qs[QK_K/2]=128 = 136 (4.25 bpw)
pub const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + (QK_K_BLOCK_SIZE / 64) + (QK_K_BLOCK_SIZE / 2);

/// ggml `kvalues_iq4nl`: the 16-entry non-linear codebook shared by the IQ4_NL and IQ4_XS
/// formats, as signed integers (the quantized weight magnitudes). Single source of truth.
pub(crate) const KVALUES_IQ4NL_I8: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// f32 view of [`KVALUES_IQ4NL_I8`], derived at compile time so the two can never diverge.
/// Used by the block decoders; the streaming integer dot uses the i8 table directly.
pub(crate) const KVALUES_IQ4NL: [f32; 16] = {
    let mut out = [0.0_f32; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = KVALUES_IQ4NL_I8[i] as f32;
        i += 1;
    }
    out
};

#[inline(always)]
pub fn fast_f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = u32::from(bits & 0x7c00) >> 10;
    let fraction = u32::from(bits & 0x03ff);

    if exponent == 0 {
        if fraction == 0 {
            return f32::from_bits(sign);
        }
        f16_bits_to_f32(bits)
    } else if exponent == 0x1f {
        f32::from_bits(sign | 0x7f80_0000 | (fraction << 13))
    } else {
        f32::from_bits(sign | ((exponent + 112) << 23) | (fraction << 13))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_0Block {
    scale_bits: u16,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q4_0Block {
    pub fn from_bytes(bytes: &[u8; Q4_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[2..]);
        Self { scale_bits, values }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn unpack_values(&self) -> [i8; Q8_BLOCK_SIZE] {
        let mut out = [0_i8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            out[idx] = ((byte & 0x0f) as i8) - 8;
            out[idx + 16] = ((byte >> 4) as i8) - 8;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_1Block {
    scale_bits: u16,
    min_bits: u16,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q4_1Block {
    pub fn from_bytes(bytes: &[u8; Q4_1_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[4..]);
        Self {
            scale_bits,
            min_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn unpack_values(&self) -> [u8; Q8_BLOCK_SIZE] {
        let mut out = [0_u8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            out[idx] = byte & 0x0f;
            out[idx + 16] = byte >> 4;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5_0Block {
    scale_bits: u16,
    high_bits: u32,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q5_0Block {
    pub fn from_bytes(bytes: &[u8; Q5_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let high_bits = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[6..]);
        Self {
            scale_bits,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn unpack_values(&self) -> [i8; Q8_BLOCK_SIZE] {
        let mut out = [0_i8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            let low_high = (((self.high_bits >> idx) & 1) as u8) << 4;
            let high_high = (((self.high_bits >> (idx + 16)) & 1) as u8) << 4;
            out[idx] = ((byte & 0x0f) | low_high) as i8 - 16;
            out[idx + 16] = ((byte >> 4) | high_high) as i8 - 16;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5_1Block {
    scale_bits: u16,
    min_bits: u16,
    high_bits: u32,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

impl Q5_1Block {
    pub fn from_bytes(bytes: &[u8; Q5_1_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let high_bits = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[8..]);
        Self {
            scale_bits,
            min_bits,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn unpack_values(&self) -> [u8; Q8_BLOCK_SIZE] {
        let mut out = [0_u8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            let low_high = (((self.high_bits >> idx) & 1) as u8) << 4;
            let high_high = (((self.high_bits >> (idx + 16)) & 1) as u8) << 4;
            out[idx] = (byte & 0x0f) | low_high;
            out[idx + 16] = (byte >> 4) | high_high;
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q2KBlock {
    scales: [u8; QK_K_BLOCK_SIZE / 16],
    values: [u8; QK_K_BLOCK_SIZE / 4],
    scale_bits: u16,
    min_bits: u16,
}

impl Q2KBlock {
    pub fn from_bytes(bytes: &[u8; Q2_K_BLOCK_BYTES]) -> Self {
        let mut scales = [0_u8; QK_K_BLOCK_SIZE / 16];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        scales.copy_from_slice(&bytes[0..16]);
        values.copy_from_slice(&bytes[16..80]);
        let scale_bits = u16::from_le_bytes([bytes[80], bytes[81]]);
        let min_bits = u16::from_le_bytes([bytes[82], bytes[83]]);
        Self {
            scales,
            values,
            scale_bits,
            min_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        let mut scale_idx = 0;

        for super_idx in 0..2 {
            let value_base = super_idx * 32;
            let out_base = super_idx * 128;
            let mut shift = 0;
            for group_idx in 0..4 {
                let low_scale = self.scales[scale_idx];
                scale_idx += 1;
                let low_d = d * (low_scale & 0x0f) as f32;
                let low_min = d_min * (low_scale >> 4) as f32;
                for l in 0..16 {
                    out[out_base + group_idx * 32 + l] =
                        low_d * ((self.values[value_base + l] >> shift) & 3) as f32 - low_min;
                }

                let high_scale = self.scales[scale_idx];
                scale_idx += 1;
                let high_d = d * (high_scale & 0x0f) as f32;
                let high_min = d_min * (high_scale >> 4) as f32;
                for l in 0..16 {
                    out[out_base + group_idx * 32 + 16 + l] = high_d
                        * ((self.values[value_base + 16 + l] >> shift) & 3) as f32
                        - high_min;
                }

                shift += 2;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q3KBlock {
    high_bits: [u8; QK_K_BLOCK_SIZE / 8],
    values: [u8; QK_K_BLOCK_SIZE / 4],
    scales: [u8; 12],
    scale_bits: u16,
}

impl Q3KBlock {
    pub fn from_bytes(bytes: &[u8; Q3_K_BLOCK_BYTES]) -> Self {
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        let mut scales = [0_u8; 12];
        high_bits.copy_from_slice(&bytes[0..32]);
        values.copy_from_slice(&bytes[32..96]);
        scales.copy_from_slice(&bytes[96..108]);
        let scale_bits = u16::from_le_bytes([bytes[108], bytes[109]]);
        Self {
            high_bits,
            values,
            scales,
            scale_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    fn expanded_scales(&self) -> [i8; 16] {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;

        let mut aux = [
            u32::from_le_bytes([
                self.scales[0],
                self.scales[1],
                self.scales[2],
                self.scales[3],
            ]),
            u32::from_le_bytes([
                self.scales[4],
                self.scales[5],
                self.scales[6],
                self.scales[7],
            ]),
            u32::from_le_bytes([
                self.scales[8],
                self.scales[9],
                self.scales[10],
                self.scales[11],
            ]),
            0,
        ];

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        let mut out = [0_i8; 16];
        for (chunk_idx, chunk) in aux.iter().enumerate() {
            for (byte_idx, byte) in chunk.to_le_bytes().iter().enumerate() {
                out[chunk_idx * 4 + byte_idx] = i8::from_le_bytes([*byte]);
            }
        }
        out
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let scales = self.expanded_scales();
        let mut scale_idx = 0;
        let mut high_mask = 1_u8;

        for super_idx in 0..2 {
            let value_base = super_idx * 32;
            let out_base = super_idx * 128;
            let mut shift = 0;
            for group_idx in 0..4 {
                let low_d = d * (scales[scale_idx] - 32) as f32;
                scale_idx += 1;
                for l in 0..16 {
                    let high = if self.high_bits[l] & high_mask != 0 {
                        0
                    } else {
                        4
                    };
                    let value = ((self.values[value_base + l] >> shift) & 3) as i8 - high;
                    out[out_base + group_idx * 32 + l] = low_d * value as f32;
                }

                let high_d = d * (scales[scale_idx] - 32) as f32;
                scale_idx += 1;
                for l in 0..16 {
                    let idx = 16 + l;
                    let high = if self.high_bits[idx] & high_mask != 0 {
                        0
                    } else {
                        4
                    };
                    let value = ((self.values[value_base + idx] >> shift) & 3) as i8 - high;
                    out[out_base + group_idx * 32 + 16 + l] = high_d * value as f32;
                }

                shift += 2;
                high_mask <<= 1;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4KBlock {
    scale_bits: u16,
    min_bits: u16,
    scales: [u8; 12],
    values: [u8; QK_K_BLOCK_SIZE / 2],
}

impl Q4KBlock {
    pub fn from_bytes(bytes: &[u8; Q4_K_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0_u8; 12];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        scales.copy_from_slice(&bytes[4..16]);
        values.copy_from_slice(&bytes[16..]);
        Self {
            scale_bits,
            min_bits,
            scales,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        for pair_idx in 0..4 {
            let low_scale_idx = pair_idx * 2;
            let high_scale_idx = low_scale_idx + 1;
            let (low_scale, low_min) = q4_k_scale_min(low_scale_idx, &self.scales);
            let (high_scale, high_min) = q4_k_scale_min(high_scale_idx, &self.scales);
            let low_scale = d * low_scale as f32;
            let high_scale = d * high_scale as f32;
            let low_min = d_min * low_min as f32;
            let high_min = d_min * high_min as f32;
            let value_base = pair_idx * 32;
            let out_base = pair_idx * 64;

            for l in 0..32 {
                let byte = self.values[value_base + l];
                out[out_base + l] = low_scale * (byte & 0x0f) as f32 - low_min;
                out[out_base + 32 + l] = high_scale * (byte >> 4) as f32 - high_min;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q5KBlock {
    scale_bits: u16,
    min_bits: u16,
    scales: [u8; 12],
    high_bits: [u8; QK_K_BLOCK_SIZE / 8],
    values: [u8; QK_K_BLOCK_SIZE / 2],
}

impl Q5KBlock {
    pub fn from_bytes(bytes: &[u8; Q5_K_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let min_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0_u8; 12];
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        scales.copy_from_slice(&bytes[4..16]);
        high_bits.copy_from_slice(&bytes[16..48]);
        values.copy_from_slice(&bytes[48..]);
        Self {
            scale_bits,
            min_bits,
            scales,
            high_bits,
            values,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn min_f32(&self) -> f32 {
        fast_f16_to_f32(self.min_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let d_min = self.min_f32();
        let mut u1 = 1_u8;
        let mut u2 = 2_u8;

        for pair_idx in 0..4 {
            let low_scale_idx = pair_idx * 2;
            let high_scale_idx = low_scale_idx + 1;
            let (low_scale, low_min) = q4_k_scale_min(low_scale_idx, &self.scales);
            let (high_scale, high_min) = q4_k_scale_min(high_scale_idx, &self.scales);
            let low_scale = d * low_scale as f32;
            let high_scale = d * high_scale as f32;
            let low_min = d_min * low_min as f32;
            let high_min = d_min * high_min as f32;
            let value_base = pair_idx * 32;
            let out_base = pair_idx * 64;

            for l in 0..32 {
                let byte = self.values[value_base + l];
                let qh = self.high_bits[l];
                let low = (byte & 0x0f) + if qh & u1 != 0 { 16 } else { 0 };
                let high = (byte >> 4) + if qh & u2 != 0 { 16 } else { 0 };
                out[out_base + l] = low_scale * low as f32 - low_min;
                out[out_base + 32 + l] = high_scale * high as f32 - high_min;
            }

            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

#[inline]
fn q4_k_scale_min(idx: usize, scales: &[u8; 12]) -> (u8, u8) {
    if idx < 4 {
        (scales[idx] & 63, scales[idx + 4] & 63)
    } else {
        (
            (scales[idx + 4] & 0x0f) | ((scales[idx - 4] >> 6) << 4),
            (scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q6KBlock {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    scale_bits: u16,
}

impl Q6KBlock {
    pub fn from_bytes(bytes: &[u8; Q6_K_BLOCK_BYTES]) -> Self {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        ql.copy_from_slice(&bytes[0..128]);
        qh.copy_from_slice(&bytes[128..192]);
        for (scale, &byte) in scales.iter_mut().zip(&bytes[192..208]) {
            *scale = i8::from_le_bytes([byte]);
        }
        let scale_bits = u16::from_le_bytes([bytes[208], bytes[209]]);
        Self {
            ql,
            qh,
            scales,
            scale_bits,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.scale_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut scale_offset = 0;

        for n in (0..QK_K_BLOCK_SIZE).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let qh = self.qh[qh_offset + l];
                let q1 = ((self.ql[ql_offset + l] & 0x0f) | ((qh & 0x03) << 4)) as i8 - 32;
                let q2 =
                    ((self.ql[ql_offset + l + 32] & 0x0f) | (((qh >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((self.ql[ql_offset + l] >> 4) | (((qh >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 =
                    ((self.ql[ql_offset + l + 32] >> 4) | (((qh >> 6) & 0x03) << 4)) as i8 - 32;

                out[n + l] = d * self.scales[scale_offset + is] as f32 * q1 as f32;
                out[n + l + 32] = d * self.scales[scale_offset + is + 2] as f32 * q2 as f32;
                out[n + l + 64] = d * self.scales[scale_offset + is + 4] as f32 * q3 as f32;
                out[n + l + 96] = d * self.scales[scale_offset + is + 6] as f32 * q4 as f32;
            }

            ql_offset += 64;
            qh_offset += 32;
            scale_offset += 8;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K_BLOCK_SIZE],
    bsums: [i16; QK_K_BLOCK_SIZE / 16],
}

impl Q8KBlock {
    pub fn from_bytes(bytes: &[u8; Q8_K_BLOCK_BYTES]) -> Self {
        let d = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let mut qs = [0_i8; 256];
        for (i, &byte) in qs.iter_mut().zip(&bytes[4..260]) {
            *i = byte as i8;
        }
        let mut bsums = [0_i16; 16];
        for (i, bsum) in bsums.iter_mut().enumerate() {
            let offset = 260 + i * 2;
            *bsum = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        }
        Self { d, qs, bsums }
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.d;
        for (out_value, &q) in out.iter_mut().zip(&self.qs) {
            *out_value = d * q as f32;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IQ4NLBlock {
    d: u16,
    qs: [u8; 16],
}

impl IQ4NLBlock {
    pub fn from_bytes(bytes: &[u8; IQ4_NL_BLOCK_BYTES]) -> Self {
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0_u8; 16];
        qs.copy_from_slice(&bytes[2..18]);
        Self { d, qs }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.d)
    }

    pub fn dequantize(&self, out: &mut [f32; 32]) {
        let d = self.scale_f32();
        for j in 0..16 {
            let byte = self.qs[j];
            out[j] = d * KVALUES_IQ4NL[(byte & 0x0F) as usize];
            out[j + 16] = d * KVALUES_IQ4NL[(byte >> 4) as usize];
        }
    }
}

/// IQ4_XS super-block (256 weights in 136 bytes, 4.25 bpw). One f16 super-block scale, eight
/// 6-bit sub-block scales (biased by -32, low nibble in `scales_l`, high 2 bits in `scales_h`),
/// and 128 bytes of 4-bit codebook indices into [`KVALUES_IQ4NL`]. Bit-for-bit with ggml's
/// `dequantize_row_iq4_xs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IQ4XSBlock {
    d: u16,
    scales_h: u16,
    scales_l: [u8; QK_K_BLOCK_SIZE / 64],
    qs: [u8; QK_K_BLOCK_SIZE / 2],
}

impl IQ4XSBlock {
    pub fn from_bytes(bytes: &[u8; IQ4_XS_BLOCK_BYTES]) -> Self {
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let scales_h = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales_l = [0_u8; QK_K_BLOCK_SIZE / 64];
        scales_l.copy_from_slice(&bytes[4..4 + QK_K_BLOCK_SIZE / 64]);
        let mut qs = [0_u8; QK_K_BLOCK_SIZE / 2];
        qs.copy_from_slice(&bytes[4 + QK_K_BLOCK_SIZE / 64..IQ4_XS_BLOCK_BYTES]);
        Self {
            d,
            scales_h,
            scales_l,
            qs,
        }
    }

    pub fn scale_f32(&self) -> f32 {
        fast_f16_to_f32(self.d)
    }

    /// Effective f32 scale of sub-block `ib` (0..8): the 6-bit scale is the low nibble from
    /// `scales_l[ib/2]` (even/odd nibble) OR'd with the high 2 bits from `scales_h`, biased -32.
    #[inline]
    fn sub_block_scale(&self, ib: usize) -> f32 {
        self.scale_f32() * self.sub_block_scale_int(ib) as f32
    }

    /// Integer part of sub-block `ib`'s scale: `ls - 32` (before the f16 super-scale). The
    /// streaming integer dot multiplies this by the super-scale and the Q8_K activation scale.
    #[inline]
    pub(crate) fn sub_block_scale_int(&self, ib: usize) -> i32 {
        let low = (self.scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F;
        let high = ((self.scales_h >> (2 * ib)) & 0x3) as u8;
        i32::from(low | (high << 4)) - 32
    }

    /// The 128 raw 4-bit codebook-index bytes (two indices per byte).
    #[inline]
    pub(crate) fn qs(&self) -> &[u8; QK_K_BLOCK_SIZE / 2] {
        &self.qs
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        for ib in 0..QK_K_BLOCK_SIZE / 32 {
            let dl = self.sub_block_scale(ib);
            let qs = &self.qs[ib * 16..ib * 16 + 16];
            let base = ib * 32;
            for j in 0..16 {
                out[base + j] = dl * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize];
                out[base + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize];
            }
        }
    }
}

// Decoding Block Helpers
pub fn decode_q4_0_blocks(bytes: &[u8]) -> Result<Vec<Q4_0Block>> {
    if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q4_0 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_0_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_0_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4_0Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q4_1_blocks(bytes: &[u8]) -> Result<Vec<Q4_1Block>> {
    if !bytes.len().is_multiple_of(Q4_1_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q4_1 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_1_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_1_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_1_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4_1Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_0_blocks(bytes: &[u8]) -> Result<Vec<Q5_0Block>> {
    if !bytes.len().is_multiple_of(Q5_0_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q5_0 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_0_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_0_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_0_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5_0Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_1_blocks(bytes: &[u8]) -> Result<Vec<Q5_1Block>> {
    if !bytes.len().is_multiple_of(Q5_1_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q5_1 byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_1_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_1_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_1_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5_1Block::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q2_k_blocks(bytes: &[u8]) -> Result<Vec<Q2KBlock>> {
    if !bytes.len().is_multiple_of(Q2_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q2_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q2_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q2_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q2_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q2KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q3_k_blocks(bytes: &[u8]) -> Result<Vec<Q3KBlock>> {
    if !bytes.len().is_multiple_of(Q3_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q3_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q3_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q3_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q3_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q3KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q4_k_blocks(bytes: &[u8]) -> Result<Vec<Q4KBlock>> {
    if !bytes.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q4_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q4_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q4_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q4_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q4KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q5_k_blocks(bytes: &[u8]) -> Result<Vec<Q5KBlock>> {
    if !bytes.len().is_multiple_of(Q5_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q5_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q5_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q5_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q5_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q5KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q6_k_blocks(bytes: &[u8]) -> Result<Vec<Q6KBlock>> {
    if !bytes.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q6_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q6_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q6_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q6_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q6KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_q8_k_blocks(bytes: &[u8]) -> Result<Vec<Q8KBlock>> {
    if !bytes.len().is_multiple_of(Q8_K_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "Q8_K byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            Q8_K_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(Q8_K_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; Q8_K_BLOCK_BYTES] = chunk.try_into().unwrap();
            Q8KBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_iq4_nl_blocks(bytes: &[u8]) -> Result<Vec<IQ4NLBlock>> {
    if !bytes.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "IQ4_NL byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            IQ4_NL_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(IQ4_NL_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; IQ4_NL_BLOCK_BYTES] = chunk.try_into().unwrap();
            IQ4NLBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

pub fn decode_iq4_xs_blocks(bytes: &[u8]) -> Result<Vec<IQ4XSBlock>> {
    if !bytes.len().is_multiple_of(IQ4_XS_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "IQ4_XS byte length {} is not aligned to {}-byte blocks",
            bytes.len(),
            IQ4_XS_BLOCK_BYTES
        )));
    }
    Ok(bytes
        .chunks_exact(IQ4_XS_BLOCK_BYTES)
        .map(|chunk| {
            let chunk_bytes: &[u8; IQ4_XS_BLOCK_BYTES] = chunk.try_into().unwrap();
            IQ4XSBlock::from_bytes(chunk_bytes)
        })
        .collect())
}

// Flat dequantization to f32 helpers
pub(crate) fn decode_q4_0_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q4_0_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

/// Decode BitNet's tensor-wide I2_S representation.
///
/// Packed bytes are grouped in 128-value tiles. Byte `gp` in each 32-byte tile
/// stores values `gp`, `32 + gp`, `64 + gp`, and `96 + gp` from most- to
/// least-significant two-bit field. Codes 0/1/2 map to -1/0/+1; code 3 is the
/// second zero spelling used by the reference decoder. One f32 scale follows
/// all packed bytes at the start of a 32-byte tensor trailer.
pub(crate) fn decode_i2_s_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if !expected_elements.is_multiple_of(128) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: I2_S element count {expected_elements} is not aligned to 128-value packing groups"
        )));
    }
    let packed_len = expected_elements / 4;
    let expected_bytes = packed_len.checked_add(32).ok_or_else(|| {
        BackendError::InvalidTensorData(format!("{name}: I2_S byte length overflow"))
    })?;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: I2_S byte length {} does not match packed payload {packed_len} + 32-byte trailer",
            bytes.len()
        )));
    }
    let scale = f32::from_le_bytes(
        bytes[packed_len..packed_len + 4]
            .try_into()
            .expect("four-byte I2_S scale"),
    );
    if !scale.is_finite() {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: I2_S tensor scale must be finite, got {scale}"
        )));
    }

    const CODE: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];
    let mut out = vec![0.0_f32; expected_elements];
    for (tile, packed) in bytes[..packed_len].chunks_exact(32).enumerate() {
        let base = tile * 128;
        for (gp, byte) in packed.iter().copied().enumerate() {
            out[base + gp] = CODE[((byte >> 6) & 3) as usize] * scale;
            out[base + 32 + gp] = CODE[((byte >> 4) & 3) as usize] * scale;
            out[base + 64 + gp] = CODE[((byte >> 2) & 3) as usize] * scale;
            out[base + 96 + gp] = CODE[(byte & 3) as usize] * scale;
        }
    }
    Ok(out)
}

// ---- Ternary (BitNet) TQ1_0 / TQ2_0 flat dequantization to f32 ----
// Faithful ports of ggml `dequantize_row_tq{1,2}_0` (llama.cpp ggml-quants.c). The
// element ORDER and the u8-truncating base-3 decode must match bit-for-bit so the
// dequantized weights reproduce llama.cpp's outputs for greedy parity.
const TQ1_0_BLOCK_BYTES: usize = 54; // qs[48] + qh[4] + f16 d  (1.69 bpw over 256 weights)
const TQ2_0_BLOCK_BYTES: usize = 66; // qs[64] + f16 d          (2.06 bpw over 256 weights)

pub(crate) fn decode_tq2_0_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(TQ2_0_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: TQ2_0 byte length {} is not aligned to {TQ2_0_BLOCK_BYTES}-byte blocks",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(TQ2_0_BLOCK_BYTES) {
        let qs = &block[0..64];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[64], block[65]]));
        // ggml: for j in {0,32}; for l in 0..4; for m in 0..32: q=(qs[j+m]>>(l*2))&3; (q-1)*d
        let mut j = 0usize;
        while j < 64 {
            for l in 0..4 {
                for m in 0..32 {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                    out.push((q - 1) as f32 * d);
                }
            }
            j += 32;
        }
    }
    Ok(out)
}

pub(crate) fn decode_tq1_0_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(TQ1_0_BLOCK_BYTES) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: TQ1_0 byte length {} is not aligned to {TQ1_0_BLOCK_BYTES}-byte blocks",
            bytes.len()
        )));
    }
    // pow3[n] for the base-3 digit extraction: trit_n = ((u8(qs*pow3[n]) * 3) >> 8) - 1.
    const POW3: [u32; 5] = [1, 3, 9, 27, 81];
    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(TQ1_0_BLOCK_BYTES) {
        let qs = &block[0..48];
        let qh = &block[48..52];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[52], block[53]]));
        // part 1: j=0 (qs[0..32]), 5 trit planes x 32
        for &pw in POW3.iter() {
            #[allow(clippy::needless_range_loop)]
            for m in 0..32 {
                let q = (qs[m] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
        // part 2: j=32 (qs[32..48]), 5 trit planes x 16
        for &pw in POW3.iter() {
            for m in 0..16 {
                let q = (qs[32 + m] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
        // part 3: qh (4 bytes), 4 trit planes x 4
        for &pw in POW3.iter().take(4) {
            #[allow(clippy::needless_range_loop)]
            for jj in 0..4 {
                let q = (qh[jj] as u32).wrapping_mul(pw) as u8;
                let xi = (((q as u16) * 3) >> 8) as i32;
                out.push((xi - 1) as f32 * d);
            }
        }
    }
    Ok(out)
}

// ---- Sign-only Q1_0 flat dequantization to f32 ----
// Q1_0 stores one SIGN bit per weight against a shared f16 scale, so the
// representable set is exactly {-d, +d} — there is no codebook, sub-scale or zero
// point. The element ORDER is the silent-corruption surface (as with the ternary
// and legacy split-half formats above): element `j` is bit `j % 8` of byte `j / 8`,
// LSB-first and sequential — NOT the split-half interleave the legacy Q4/Q5 formats
// use. A bit-order flip still dequantizes to plausible-looking weights, so this is
// pinned by `q1_0_dequant_matches_the_reference_layout` (hand-built blocks) and by
// `tests/q1_0_format.rs` (real blocks against the model vendor's own expansion of
// the same weights).
pub const Q1_0_BLOCK_ELEMENTS: usize = 128; // QK1_0
pub const Q1_0_BLOCK_BYTES: usize = 18; // f16 d(2) + qs[16]  (1.125 bpw over 128 weights)

pub const Q2_0_G64_BLOCK_ELEMENTS: usize = 64;
pub const Q2_0_G64_BLOCK_BYTES: usize = 18; // f16 d(2) + 16 packed 2-bit bytes
pub const Q2_0_G128_BLOCK_ELEMENTS: usize = 128;
pub const Q2_0_G128_BLOCK_BYTES: usize = 34; // f16 d(2) + 32 packed 2-bit bytes

pub fn decode_q1_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if !expected_elements.is_multiple_of(Q1_0_BLOCK_ELEMENTS) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: Q1_0 element count {expected_elements} is not a multiple of \
             {Q1_0_BLOCK_ELEMENTS}"
        )));
    }
    let blocks = expected_elements / Q1_0_BLOCK_ELEMENTS;
    let expected_bytes = blocks * Q1_0_BLOCK_BYTES;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: Q1_0 wire length {} != {blocks} blocks * {Q1_0_BLOCK_BYTES} bytes = \
             {expected_bytes}",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(Q1_0_BLOCK_BYTES) {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // Mirror the pin's `neg_d = -d` rather than multiplying by ±1.0: for a zero
        // scale this yields +0.0 for set bits and -0.0 for clear bits, and the sign
        // of zero is observable downstream.
        let neg_d = -d;
        let qs = &block[2..Q1_0_BLOCK_BYTES];
        for j in 0..Q1_0_BLOCK_ELEMENTS {
            let bit = (qs[j / 8] >> (j % 8)) & 1;
            out.push(if bit == 1 { d } else { neg_d });
        }
    }
    Ok(out)
}

/// Decode either currently deployed GGUF type-42 Q2_0 geometry.
///
/// Both dialects use the same code mapping (`00=-d`, `01=0`, `10=+d`,
/// `11=+2d`) and differ only in how many codes share one f16 scale. The GGUF
/// parser resolves the ambiguous type id into `Q2_0G64` or `Q2_0G128` before
/// this function is called.
pub fn decode_q2_0_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
    tensor_type: GgufTensorType,
) -> Result<Vec<f32>> {
    let (block_elements, block_bytes) = match tensor_type {
        GgufTensorType::Q2_0G64 => (Q2_0_G64_BLOCK_ELEMENTS, Q2_0_G64_BLOCK_BYTES),
        GgufTensorType::Q2_0G128 => (Q2_0_G128_BLOCK_ELEMENTS, Q2_0_G128_BLOCK_BYTES),
        GgufTensorType::Pq2_0 => (Q2_0_G128_BLOCK_ELEMENTS, Q2_0_G128_BLOCK_BYTES),
        GgufTensorType::Q2_0 => {
            return Err(BackendError::InvalidTensorData(format!(
                "{name}: unresolved Q2_0 dialect reached tensor decode"
            )))
        }
        other => {
            return Err(BackendError::InvalidTensorData(format!(
                "{name}: Q2_0 decoder received non-Q2 tensor type {other:?}"
            )))
        }
    };
    if !expected_elements.is_multiple_of(block_elements) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: {tensor_type:?} element count {expected_elements} is not a multiple of {block_elements}"
        )));
    }
    let blocks = expected_elements / block_elements;
    let expected_bytes = blocks.checked_mul(block_bytes).ok_or_else(|| {
        BackendError::InvalidTensorData(format!("{name}: Q2_0 byte length overflow"))
    })?;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: {tensor_type:?} wire length {} != {blocks} blocks * {block_bytes} bytes = {expected_bytes}",
            bytes.len()
        )));
    }

    let mut out = Vec::with_capacity(expected_elements);
    for block in bytes.chunks_exact(block_bytes) {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        for j in 0..block_elements {
            let q = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            out.push((i32::from(q) - 1) as f32 * d);
        }
    }
    Ok(out)
}

/// Losslessly re-encode Q1_0 wire bytes as Q8_0 blocks.
///
/// Each 128-value Q1_0 block becomes exactly 4 Q8_0 blocks of 32 that share its
/// scale, with `qs = +1` where the sign bit is set and `-1` where it is clear.
/// Q8_0 dequantizes as `qs as f32 * scale`, so `(+1) * d == +d` and
/// `(-1) * d == -d` reproduce Q1_0's `{-d, +d}` EXACTLY — no rounding, no
/// re-quantization, no error term. See `load_q1_0_as_q8_0_blocks_linear` for why
/// this exists and what it costs.
///
/// The scale is widened f16 -> f32 (exact), matching `Q8_0Block::scale`. A zero
/// scale stays zero in both encodings; the SIGN of a zero-scale value is the one
/// place the two representations can differ (`-1.0 * 0.0` is `-0.0`, as is the
/// decoder's `neg_d`), so they agree there too.
pub(crate) fn q1_0_to_q8_0_blocks(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<Q8_0Block>> {
    if !expected_elements.is_multiple_of(Q1_0_BLOCK_ELEMENTS) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: Q1_0 element count {expected_elements} is not a multiple of \
             {Q1_0_BLOCK_ELEMENTS}"
        )));
    }
    let q1_blocks = expected_elements / Q1_0_BLOCK_ELEMENTS;
    let expected_bytes = q1_blocks * Q1_0_BLOCK_BYTES;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: Q1_0 wire length {} != {q1_blocks} blocks * {Q1_0_BLOCK_BYTES} bytes = \
             {expected_bytes}",
            bytes.len()
        )));
    }
    // 128 values per Q1_0 block / 32 per Q8_0 block = 4, exactly.
    const Q8_PER_Q1: usize = Q1_0_BLOCK_ELEMENTS / 32;
    let mut out = Vec::with_capacity(q1_blocks * Q8_PER_Q1);
    for block in bytes.chunks_exact(Q1_0_BLOCK_BYTES) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..Q1_0_BLOCK_BYTES];
        for sub in 0..Q8_PER_Q1 {
            let mut quants = [0i8; 32];
            for (k, quant) in quants.iter_mut().enumerate() {
                let j = sub * 32 + k;
                let bit = (qs[j / 8] >> (j % 8)) & 1;
                *quant = if bit == 1 { 1 } else { -1 };
            }
            out.push(Q8_0Block { scale, quants });
        }
    }
    Ok(out)
}

pub(crate) fn decode_q4_1_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q4_1_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        let min = block.min_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale + min);
        }
    }
    Ok(out)
}

fn decode_q5_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_0_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

fn decode_q5_1_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_1_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        let min = block.min_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale + min);
        }
    }
    Ok(out)
}

pub(crate) fn decode_q2_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q2_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn decode_q3_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q3_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn decode_q4_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q4_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn decode_q5_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q5_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn decode_q6_k_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_q6_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q8_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q8_k_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_iq4_nl_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_iq4_nl_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; 32];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

pub(crate) fn decode_iq4_xs_tensor(
    name: &str,
    bytes: &[u8],
    expected_elements: usize,
) -> Result<Vec<f32>> {
    let blocks = decode_iq4_xs_blocks(bytes)
        .map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

// ---- NVFP4 (BASALT Phase 1) — pin-layout format core ------------------------------------
//
// Wire layout (DECISIONS.md D17 / D-B1: the pin's `block_nvfp4` byte-for-byte, llama.cpp
// acd79d603 / GGML_TYPE_NVFP4 = 40): 36 bytes per 64 elements — `d[4]` UE4M3 sub-block
// scales (one per 16 elements) FIRST, then `qs[32]` packed E2M1 nibbles. Sub-block `s`
// (0..3) owns `qs[s*8 .. s*8+7]`; the LOW nibble of byte `s*8+j` is element `s*16+j`, the
// HIGH nibble element `s*16+8+j` (the MXFP4-style half/half split, not adjacent pairing).
//
// WHY TWO DECODE SEMANTICS COEXIST (D17 / open item T5):
// - Per-block dequant ([`crate::inference::nvfp4_wire_block_dequant`] and the block loop
//   inside [`decode_nvfp4_tensor`]) is PIN-BITWISE: scale byte 0x7F is the pin's NaN
//   sentinel and FLUSHES to d = 0.0 silently, exactly like `dequantize_row_nvfp4`. This
//   keeps every decoded value bit-identical to the oracle for parity receipts, whatever
//   the bytes say.
// - The LOAD path ([`decode_nvfp4_tensor`]) FAILS CLOSED first: it scans every block's
//   `d[4]` and refuses tensors carrying a NaN-sentinel scale byte (0x7F or 0xFF) with a
//   machine-readable error, because a sentinel in a weight file means the quantizer saw
//   garbage and silently zeroing 16 weights per hit would be quiet model corruption.
//   Files that pass admission therefore never contain sentinels, so both semantics agree
//   on every tensor Camelid actually runs; the pin-bitwise flush only ever fires in
//   fixture/parity harnesses that feed crafted blocks below the load path.
//
// Sentinel subtlety the golden fixtures lock in (`nvfp4_ue4m3_table.json`): the pin's CPU
// decode checks the RAW byte (`x == 0x7F`), so 0xFF is NOT flushed — it decodes through
// exp/man extraction to 240.0. The pin's CUDA mirror flushes both 0x7F and 0xFF, which is
// exactly why the load path refuses both bytes: the two pin backends disagree on 0xFF, and
// refusing at admission keeps Camelid out of that ambiguity entirely (D17/T5).

/// NVFP4 values per wire super-block (pin `QK_NVFP4`).
pub const NVFP4_VALUES_PER_BLOCK: usize = 64;

/// NVFP4 wire bytes per super-block: `d[4]` UE4M3 scales then `qs[32]` nibbles
/// (pin `block_nvfp4`, `static_assert sizeof == 36`).
pub const NVFP4_WIRE_BYTES_PER_BLOCK: usize = 36;

/// NVFP4 sub-block width (pin `QK_NVFP4_SUB`): one UE4M3 scale byte per 16 elements.
pub const NVFP4_SUB_BLOCK_VALUES: usize = 16;

/// ggml `kvalues_mxfp4` (ggml-common.h): the E2M1 element magnitudes DOUBLED (true
/// magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6); nibble bit 3 selects the sign half.
/// THE PAIR RULE: this doubling is paired with the extra 0.5 factor baked into
/// [`UE4M3_TO_F32`] — the two conventions must always travel together, or every
/// decoded value is off by 2x or 0.5x.
pub const KVALUES_MXFP4_I8: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// f32 view of [`KVALUES_MXFP4_I8`], derived at compile time so the two can never
/// diverge (same idiom as [`KVALUES_IQ4NL`]).
pub const KVALUES_MXFP4: [f32; 16] = {
    let mut out = [0.0_f32; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = KVALUES_MXFP4_I8[i] as f32;
        i += 1;
    }
    out
};

/// One UE4M3 scale byte -> f32, mirroring the pin's `ggml_ue4m3_to_fp32`
/// (ggml-impl.h) bit-for-bit: raw bytes 0x00 and 0x7F return 0.0 (0x7F is the NaN
/// sentinel, FLUSHED — and the check is on the raw byte, so 0xFF is NOT flushed and
/// decodes to 240.0; see the module comment above). Otherwise exp = bits 6..3
/// (bias 7), man = bits 2..0; exp == 0 is subnormal `man * 2^-9`, else
/// `(1 + man/8) * 2^(exp-7)`; the result carries the extra 0.5 pair-rule factor.
/// Every step multiplies exact values by powers of two, so const evaluation cannot
/// round differently from the pin's `ldexpf` path.
const fn ue4m3_to_f32_const(byte: u8) -> f32 {
    if byte == 0x00 || byte == 0x7F {
        return 0.0;
    }
    let exp = ((byte >> 3) & 0xF) as i32;
    let man = (byte & 0x7) as f32;
    let raw = if exp == 0 {
        // subnormal: man * 2^-9
        man * f32::from_bits((127 - 9) << 23)
    } else {
        // normal: (1 + man/8) * 2^(exp-7); exp-7 in -6..=8 so the power is a
        // normal f32 built directly from its biased exponent
        (1.0 + man / 8.0) * f32::from_bits(((exp - 7 + 127) as u32) << 23)
    };
    raw * 0.5
}

/// Precomputed 256-entry UE4M3 decode table (see [`ue4m3_to_f32_const`]); anchored
/// bit-exactly to the pin-generated `tests/fixtures/dequant/nvfp4_ue4m3_table.json`.
pub const UE4M3_TO_F32: [f32; 256] = {
    let mut out = [0.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        out[b] = ue4m3_to_f32_const(b as u8);
        b += 1;
    }
    out
};

/// Scan NVFP4 wire bytes for NaN-sentinel UE4M3 scale bytes (0x7F / 0xFF) in any
/// block's `d[4]`, returning the FIRST offending block index. This is the single
/// definition of load-time sentinel refusal, shared by [`decode_nvfp4_tensor`] and
/// (Phase 2) runnable-lane admission. Scans whole 36-byte blocks only; callers
/// validate total length separately.
pub fn nvfp4_find_nan_scale(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(NVFP4_WIRE_BYTES_PER_BLOCK)
        .position(|block| block[..4].iter().any(|&b| b == 0x7F || b == 0xFF))
}

/// Decode one 36-byte NVFP4 wire block into 64 f32 values, pin-bitwise
/// (`dequantize_row_nvfp4` order): per sub-block `s`, `d = UE4M3_TO_F32[d[s]]`, low
/// nibble of `qs[s*8+j]` -> element `s*16+j`, high nibble -> element `s*16+8+j`,
/// value = `KVALUES_MXFP4[nibble] * d`. Negative codes (9..15) under a zero scale
/// produce -0.0 (the i8-derived f32 sign survives the multiply), matching the pin.
fn nvfp4_block_decode_into(out: &mut [f32], block: &[u8]) {
    debug_assert_eq!(block.len(), NVFP4_WIRE_BYTES_PER_BLOCK);
    debug_assert_eq!(out.len(), NVFP4_VALUES_PER_BLOCK);
    for s in 0..4 {
        let d = UE4M3_TO_F32[block[s] as usize];
        for j in 0..8 {
            let byte = block[4 + s * 8 + j];
            out[s * NVFP4_SUB_BLOCK_VALUES + j] = KVALUES_MXFP4[(byte & 0x0F) as usize] * d;
            out[s * NVFP4_SUB_BLOCK_VALUES + 8 + j] = KVALUES_MXFP4[(byte >> 4) as usize] * d;
        }
    }
}

/// Flat NVFP4 tensor dequantization for the LOAD path — mirrors
/// [`decode_q4_k_tensor`]'s shape, plus the D17/T5 fail-closed sentinel scan (see
/// the module comment above for why this deliberately diverges from the pin-bitwise
/// per-block seam on sentinel-bearing bytes).
pub fn decode_nvfp4_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    if !expected_elements.is_multiple_of(NVFP4_VALUES_PER_BLOCK) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: NVFP4 element count {expected_elements} is not a multiple of \
             {NVFP4_VALUES_PER_BLOCK}"
        )));
    }
    let blocks = expected_elements / NVFP4_VALUES_PER_BLOCK;
    let expected_bytes = blocks * NVFP4_WIRE_BYTES_PER_BLOCK;
    if bytes.len() != expected_bytes {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: NVFP4 wire length {} != {blocks} blocks * {NVFP4_WIRE_BYTES_PER_BLOCK} \
             bytes = {expected_bytes}",
            bytes.len()
        )));
    }
    // D17/T5 fail-closed: refuse NaN-sentinel scale bytes at load. A file that
    // admits never reaches the pin-bitwise flush below.
    if let Some(block_idx) = nvfp4_find_nan_scale(bytes) {
        return Err(BackendError::InvalidTensorData(format!(
            "{name}: NVFP4 block {block_idx} carries a NaN-sentinel UE4M3 scale byte \
             (0x7F/0xFF) — refusing per D17/T5 (fail closed at load; per-block dequant \
             stays pin-bitwise)"
        )));
    }
    let mut out = vec![0.0_f32; expected_elements];
    for (i, block) in bytes.chunks_exact(NVFP4_WIRE_BYTES_PER_BLOCK).enumerate() {
        nvfp4_block_decode_into(
            &mut out[i * NVFP4_VALUES_PER_BLOCK..(i + 1) * NVFP4_VALUES_PER_BLOCK],
            block,
        );
    }
    Ok(out)
}

/// Pin `ggml_fp32_to_ue4m3` port — TEST-ANCHORING ONLY. Quantizer ownership is
/// pin-tool-only for v1 (D17/D-B5); this exists so the encode golden vectors and
/// round-trip property tests can reproduce the pin's wire bytes, and is not a
/// Camelid quantizer surface. Semantics locked by `nvfp4_encode_vectors.json`:
/// NaN/<=0 -> 0x00; input domain clamps at 448.0; normal path rounds HALF-UP on the
/// 4th mantissa bit (with carry into the exponent); exp >= 15 saturates to 0x7E
/// (the encoder never emits 0x78..0x7D, 0x7F, or any byte with bit 7 set);
/// subnormal path rounds half-up via `(x * 512 + 0.5)` truncation.
#[cfg(test)]
pub(crate) fn fp32_to_ue4m3(x: f32) -> u8 {
    // The pin writes `if (!(x > 0.0f)) return 0;` — NaN and every x <= 0 land here.
    if x.is_nan() || x <= 0.0 {
        return 0;
    }
    let x = if x > 448.0 { 448.0 } else { x };
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue_exp = fp32_exp + 7;
    if ue_exp <= 0 {
        // subnormal: round-half-up on man * 512 (truncation of positive value)
        let man = ((x * 512.0 + 0.5) as i32).min(7);
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue_exp >= 15 {
        return 0x7E; // saturate to max finite code
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue_man = fp32_man + round_bit;
    if ue_man > 7 {
        ue_man = 0;
        ue_exp += 1;
        if ue_exp >= 15 {
            return 0x7E;
        }
    }
    ((ue_exp << 3) | ue_man) as u8
}

/// Pin `best_index_mxfp4` port — TEST-ANCHORING ONLY (see [`fp32_to_ue4m3`]).
/// Exhaustive nearest search over `KVALUES_MXFP4[i] * d`, strict `<` so the FIRST
/// index wins exact ties (scan order 0..15) — not IEEE round-nearest-even. NaN
/// inputs never beat the initial candidate, so they quantize to code 0.
#[cfg(test)]
fn nvfp4_best_index(x: f32, d: f32) -> u8 {
    let mut best_index = 0usize;
    let mut best_err = (KVALUES_MXFP4[0] * d - x).abs();
    for (i, kv) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (kv * d - x).abs();
        if err < best_err {
            best_index = i;
            best_err = err;
        }
    }
    best_index as u8
}

/// Encode one 64-element row into a 36-byte NVFP4 wire block — TEST-ANCHORING ONLY
/// (D17/D-B5: v1 is consume-side; the golden quantizer is the pin's tool). Mirrors
/// `quantize_row_nvfp4_ref` exactly: per sub-block amax via a NaN-insensitive `<`
/// comparison (all-NaN rows therefore encode to an all-zero wire, and +/-Inf rows
/// saturate the scale to 0x7E while `best_index` leaves every element at code 0),
/// scale byte = `fp32_to_ue4m3(amax / 6)`, elements quantized against the DECODED
/// scale via first-wins nearest-LUT search. The amax scan compares BIT PATTERNS,
/// not floats — see the comment in the loop; a plain `<` is miscompiled at `-O`.
#[cfg(test)]
pub(crate) fn encode_nvfp4_block(
    x: &[f32; NVFP4_VALUES_PER_BLOCK],
) -> [u8; NVFP4_WIRE_BYTES_PER_BLOCK] {
    let mut wire = [0u8; NVFP4_WIRE_BYTES_PER_BLOCK];
    for s in 0..4 {
        let xb = &x[s * NVFP4_SUB_BLOCK_VALUES..(s + 1) * NVFP4_SUB_BLOCK_VALUES];
        let mut amax = 0.0_f32;
        for &v in xb {
            let a = v.abs();
            // The pin's `if (amax < a) amax = a;` never fires on a NaN, so a NaN
            // element must leave `amax` untouched. Spelling that as the float `<`
            // is NOT enough: while `amax` is still the literal 0.0, LLVM peels the
            // first iteration and folds `select(0.0 < a, a, 0.0)` into a max
            // intrinsic — `fmaxnm` on aarch64 — and `fmaxnm` returns the QUIETED
            // operand when one input is a SIGNALING NaN (it returns the other
            // operand only for a quiet one). A leading sNaN therefore poisoned
            // `amax`, zeroing the sub-block scale, in optimized builds ONLY;
            // fixture row `path-snan-first` caught it, debug builds never did.
            // Here both operands are non-negative and non-NaN, and over that
            // domain unsigned bit-pattern order IS IEEE magnitude order — so this
            // is the same predicate, expressed as an integer compare that no
            // float fold can rewrite.
            if !a.is_nan() && a.to_bits() > amax.to_bits() {
                amax = a;
            }
        }
        let ue = fp32_to_ue4m3(amax / 6.0);
        wire[s] = ue;
        let d = UE4M3_TO_F32[ue as usize];
        for j in 0..8 {
            let lo = nvfp4_best_index(xb[j], d);
            let hi = nvfp4_best_index(xb[8 + j], d);
            wire[4 + s * 8 + j] = lo | (hi << 4);
        }
    }
    wire
}

/// BASALT Phase 1 encode-side anchoring + property loops. The decode-side golden
/// suites (ue4m3 table, 4096-pair decode table, nibble probes, 10k random + real
/// GGUF blocks, fail-closed seam) live in `tests/nvfp4_format.rs`; these unit tests
/// stay inline because [`encode_nvfp4_block`] / [`fp32_to_ue4m3`] are deliberately
/// not exported (D17/D-B5: v1 is consume-side, quantizer ownership is pin-tool-only).
#[cfg(test)]
mod nvfp4_tests {
    use super::{
        decode_nvfp4_tensor, encode_nvfp4_block, fp32_to_ue4m3, nvfp4_block_decode_into,
        KVALUES_MXFP4, NVFP4_SUB_BLOCK_VALUES, NVFP4_VALUES_PER_BLOCK, NVFP4_WIRE_BYTES_PER_BLOCK,
        UE4M3_TO_F32,
    };

    fn fixture_json(name: &str) -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dequant")
            .join(name);
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
        let v: serde_json::Value =
            serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name} parses: {e}"));
        assert_eq!(
            v["provenance"]["pin_sha"].as_str(),
            Some("acd79d603"),
            "{name}: fixture provenance pin mismatch"
        );
        v
    }

    fn hex_u32(h: &str) -> u32 {
        u32::from_str_radix(h, 16).unwrap_or_else(|e| panic!("bad hex u32 {h:?}: {e}"))
    }

    fn hex_row_bits(h: &str) -> Vec<u32> {
        assert!(h.len().is_multiple_of(8));
        (0..h.len())
            .step_by(8)
            .map(|i| hex_u32(&h[i..i + 8]))
            .collect()
    }

    /// Minimal RFC 4648 base64 decoder (fixtures only; no base64 dependency).
    fn b64_decode(s: &str) -> Vec<u8> {
        let mut table = [255u8; 256];
        for (i, c) in (b'A'..=b'Z').enumerate() {
            table[c as usize] = i as u8;
        }
        for (i, c) in (b'a'..=b'z').enumerate() {
            table[c as usize] = 26 + i as u8;
        }
        for (i, c) in (b'0'..=b'9').enumerate() {
            table[c as usize] = 52 + i as u8;
        }
        table[b'+' as usize] = 62;
        table[b'/' as usize] = 63;
        let bytes: Vec<u8> = s.bytes().filter(|&b| b != b'=').collect();
        let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
        for chunk in bytes.chunks(4) {
            let mut acc = 0u32;
            for (k, &b) in chunk.iter().enumerate() {
                let v = table[b as usize];
                assert_ne!(v, 255, "bad base64 byte {b}");
                acc |= u32::from(v) << (18 - 6 * k);
            }
            out.push((acc >> 16) as u8);
            if chunk.len() > 2 {
                out.push((acc >> 8) as u8);
            }
            if chunk.len() > 3 {
                out.push(acc as u8);
            }
        }
        out
    }

    fn decode_block_via_tensor_path(wire: &[u8]) -> [f32; NVFP4_VALUES_PER_BLOCK] {
        let out = decode_nvfp4_tensor("nvfp4-unit-test", wire, NVFP4_VALUES_PER_BLOCK)
            .expect("clean block decodes");
        let mut arr = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        arr.copy_from_slice(&out);
        arr
    }

    fn assert_bits(got: &[f32], want_bits: &[u32], ctx: &str) {
        assert_eq!(got.len(), want_bits.len(), "{ctx}: length");
        for (j, (g, w)) in got.iter().zip(want_bits.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                *w,
                "{ctx}: element {j} got {:#010x} want {w:#010x}",
                g.to_bits()
            );
        }
    }

    /// All 27 pin-generated encode vectors reproduce byte-exactly, including the
    /// pathological rows (golden truth, not judged): all-NaN input -> all-zero
    /// wire; +/-Inf rows -> scale 0x7E with every element code 0 (decode +0.0);
    /// exact LUT midpoints -> LOWER index; -0.0; subnormals; saturation.
    #[test]
    fn encode_vectors_reproduce_pin_wire_bytes_and_dequant() {
        let fx = fixture_json("nvfp4_encode_vectors.json");
        let vectors = fx["vectors"].as_array().expect("vectors");
        assert_eq!(vectors.len(), 27);
        let mut seen_spotlock_tags = std::collections::BTreeSet::new();
        for vec in vectors {
            let tag = vec["tag"].as_str().expect("tag");
            let input_hex = vec["input"].as_array().expect("input");
            assert_eq!(input_hex.len(), NVFP4_VALUES_PER_BLOCK, "{tag}: input len");
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for (j, h) in input_hex.iter().enumerate() {
                x[j] = f32::from_bits(hex_u32(h.as_str().expect("hex")));
            }
            let want_wire = b64_decode(vec["wire"].as_str().expect("wire"));
            assert_eq!(
                want_wire.len(),
                NVFP4_WIRE_BYTES_PER_BLOCK,
                "{tag}: wire len"
            );
            let got_wire = encode_nvfp4_block(&x);
            assert_eq!(
                got_wire.as_slice(),
                want_wire.as_slice(),
                "{tag}: wire bytes"
            );

            let want_bits = hex_row_bits(vec["dequant"].as_str().expect("dequant"));
            let got = decode_block_via_tensor_path(&got_wire);
            assert_bits(&got, &want_bits, tag);

            // Spot-lock the pathological semantics by tag so a future regression
            // fails with a readable message, not just a byte diff. Every tag the
            // arms name must actually occur in the fixture — a silent `_` arm
            // would let a fixture-tag rename disable these locks unnoticed.
            match tag {
                "path-all-qnan" | "path-all-neg-qnan" | "path-all-negzero" | "negzero-single" => {
                    assert_eq!(got_wire, [0u8; 36], "{tag}: expected all-zero wire");
                    seen_spotlock_tags.insert(tag.to_string());
                }
                "path-all-pinf" | "path-all-ninf" | "path-inf-alt" => {
                    assert!(
                        got_wire[..4].iter().all(|&b| b == 0x7E),
                        "{tag}: scale 0x7E"
                    );
                    assert!(got_wire[4..].iter().all(|&b| b == 0), "{tag}: all code 0");
                    seen_spotlock_tags.insert(tag.to_string());
                }
                "sat-exact-448" | "sat-448-plus-ulp" | "sat-1e4" | "sat-fltmax" => {
                    assert!(
                        got_wire[..4].iter().all(|&b| b == 0x7E),
                        "{tag}: scale 0x7E"
                    );
                    seen_spotlock_tags.insert(tag.to_string());
                }
                _ => {}
            }
        }
        for expected in [
            "path-all-qnan",
            "path-all-neg-qnan",
            "path-all-negzero",
            "negzero-single",
            "path-all-pinf",
            "path-all-ninf",
            "path-inf-alt",
            "sat-exact-448",
            "sat-448-plus-ulp",
            "sat-1e4",
            "sat-fltmax",
        ] {
            assert!(
                seen_spotlock_tags.contains(expected),
                "fixture is missing spot-lock tag {expected}: the semantic lock never ran"
            );
        }
    }

    /// OPTIMIZATION-SENSITIVITY LOCK for the amax scan (see [`encode_nvfp4_block`]).
    /// A NaN element must change NOTHING except its own nibble: not the sub-block
    /// scale, not its 63 neighbours' codes. Asserted against a NaN-free encode of
    /// the same row rather than a golden blob, so it keeps testing the property if
    /// the fixtures are ever regenerated — and swept over EVERY slot and all four
    /// NaN encodings, because the defect it guards was position- and
    /// signalling-specific: only slot 0 of a sub-block (where `amax` is still the
    /// literal 0.0, so LLVM peels the compare into `fmaxnm`) and only a SIGNALING
    /// NaN (which `fmaxnm` quiets INTO the result instead of discarding). That
    /// combination is one row of `nvfp4_encode_vectors.json` (`path-snan-first`)
    /// and it failed in release while every debug build passed.
    #[test]
    fn encode_amax_ignores_nan_in_every_slot_and_encoding() {
        let clean = encode_nvfp4_block(&[2.0_f32; NVFP4_VALUES_PER_BLOCK]);
        // Anchor the NaN-free baseline so a broken encoder cannot make this test
        // vacuous by agreeing with itself: amax 2.0 -> ue4m3(1/3) = 0x2B, and 2.0
        // sits nearest code 7 (12 * 0.171875 = 2.0625).
        assert_eq!(clean[..4], [0x2B; 4], "baseline scale bytes");
        assert!(clean[4..].iter().all(|&b| b == 0x77), "baseline codes");

        for slot in 0..NVFP4_VALUES_PER_BLOCK {
            for nan_bits in [
                0x7f80_0001_u32, // +sNaN (the regressing input)
                0xff80_0001,     // -sNaN
                0x7fc0_0000,     // +qNaN
                0xffc0_0000,     // -qNaN
            ] {
                let mut x = [2.0_f32; NVFP4_VALUES_PER_BLOCK];
                x[slot] = f32::from_bits(nan_bits);

                // Only the NaN's own nibble goes to code 0; `best_index` never lets
                // a NaN beat the initial candidate.
                let mut want = clean;
                let within = slot % NVFP4_SUB_BLOCK_VALUES;
                let byte = 4 + (slot / NVFP4_SUB_BLOCK_VALUES) * 8 + within % 8;
                want[byte] &= if within < 8 { 0xF0 } else { 0x0F };

                assert_eq!(
                    encode_nvfp4_block(&x),
                    want,
                    "NaN {nan_bits:#010x} in slot {slot} must not disturb the block"
                );
            }
        }
    }

    /// Every pin-quantized PRNG/edge input row reproduces the pin's wire bytes
    /// through the Rust encoder — 10031 diverse encode anchors on top of the 27
    /// curated vectors.
    #[test]
    fn random_blocks_encode_reproduces_pin_wire_bytes() {
        let fx = fixture_json("nvfp4_random_blocks.json");
        let blocks = fx["blocks"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 10031);
        for (i, blk) in blocks.iter().enumerate() {
            let tag = blk["tag"].as_str().expect("tag");
            let input = b64_decode(blk["i"].as_str().expect("input"));
            assert_eq!(input.len(), 256, "block {i} ({tag}): input bytes");
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for (j, chunk) in input.chunks_exact(4).enumerate() {
                x[j] = f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let want_wire = b64_decode(blk["w"].as_str().expect("wire"));
            let got_wire = encode_nvfp4_block(&x);
            assert_eq!(
                got_wire.as_slice(),
                want_wire.as_slice(),
                "block {i} ({tag}): wire bytes"
            );
        }
    }

    /// Representable-value round-trip: x = KVALUES_MXFP4[c] * UE4M3_TO_F32[s] for
    /// every scale byte with a nonzero decoded scale and all 16 codes. For every
    /// ENCODER-REACHABLE scale (masked 0x01..=0x77 and 0x7E) the round trip is
    /// bit-exact and the stored scale byte equals the masked input byte. Masked
    /// 0x78..0x7D (raw >= 256 saturates to 0x7E before mantissa rounding) and 0xFF
    /// (raw 480 exceeds the 448 input clamp) are unreachable from the pin encoder
    /// BY DESIGN — for those the encoder must emit 0x7E, and the quantized VALUE
    /// set must be a fixed point of a second quantize->dequantize pass. (The WIRE
    /// is deliberately not asserted stable: at masked 0x78 the pin itself re-tightens
    /// the scale on a second pass — amax of the first-pass output drops to 1344,
    /// whose amax/6 leaves the exp>=15 saturation region and re-encodes as 0x76 —
    /// while the decoded values stay bit-identical.)
    #[test]
    fn representable_value_round_trip_all_scales_all_codes() {
        for (s, &d) in UE4M3_TO_F32.iter().enumerate() {
            if d.to_bits() == 0 {
                continue; // 0x00 (zero), 0x7F (sentinel flush), 0x80 (masked zero)
            }
            let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
            for sub in 0..4 {
                for (c, kv) in KVALUES_MXFP4.iter().enumerate() {
                    x[sub * 16 + c] = kv * d;
                }
            }
            let wire = encode_nvfp4_block(&x);
            let masked = (s as u8) & 0x7F;
            let encodable = (0x01..=0x77).contains(&masked) || masked == 0x7E;
            if encodable {
                assert!(
                    wire[..4].iter().all(|&b| b == masked),
                    "scale {s:#04x}: stored scale byte {:#04x} != masked {masked:#04x}",
                    wire[0]
                );
                let y = decode_block_via_tensor_path(&wire);
                for (j, (got, want)) in y.iter().zip(x.iter()).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "scale {s:#04x} element {j}: round trip not bit-exact"
                    );
                }
            } else {
                assert!(
                    wire[..4].iter().all(|&b| b == 0x7E),
                    "scale {s:#04x}: unreachable scale must saturate to 0x7E, got {:#04x}",
                    wire[0]
                );
                // Value-level fixed point: re-quantizing the quantized values
                // reproduces them bit-exactly (even where the wire scale byte
                // legitimately re-tightens, e.g. masked 0x78).
                let y = decode_block_via_tensor_path(&wire);
                let y2 = decode_block_via_tensor_path(&encode_nvfp4_block(&y));
                for (j, (a, b)) in y.iter().zip(y2.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "scale {s:#04x} element {j}: quantized values not a fixed point"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_block_round_trip_and_negative_zero_encode() {
        // All +0.0: zero wire, all +0.0 back.
        let x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        let wire = encode_nvfp4_block(&x);
        assert_eq!(wire, [0u8; NVFP4_WIRE_BYTES_PER_BLOCK]);
        let y = decode_block_via_tensor_path(&wire);
        for v in &y {
            assert_eq!(v.to_bits(), 0x0000_0000);
        }
        // All -0.0 encodes IDENTICALLY (pin semantics: amax stays 0 because
        // `0.0 < |-0.0|` is false, and best_index's initial candidate 0 survives
        // every tie) — the -0.0 sign does NOT survive the encode side. Sign
        // survival on the DECODE side is covered below.
        let neg = [f32::from_bits(0x8000_0000); NVFP4_VALUES_PER_BLOCK];
        let wire = encode_nvfp4_block(&neg);
        assert_eq!(wire, [0u8; NVFP4_WIRE_BYTES_PER_BLOCK]);
    }

    /// Decode-side -0.0 sign survival: negative codes (9..15) under a ZERO decoded
    /// scale (byte 0x00, and the flushed sentinel 0x7F) multiply to -0.0
    /// (bit pattern 0x80000000), positive codes and code 8 to +0.0 — matching the
    /// golden decode-table rows bit-for-bit.
    #[test]
    fn negative_codes_times_zero_scale_decode_to_negative_zero() {
        let mut wire = [0u8; NVFP4_WIRE_BYTES_PER_BLOCK];
        wire[..4].copy_from_slice(&[0x00, 0x7F, 0x00, 0x7F]);
        // sub 0: code 9 everywhere; sub 1: code 15; sub 2: code 0; sub 3: code 8.
        wire[4..12].fill(0x99);
        wire[12..20].fill(0xFF);
        wire[20..28].fill(0x00);
        wire[28..36].fill(0x88);
        let mut out = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        nvfp4_block_decode_into(&mut out, &wire);
        for j in 0..16 {
            assert_eq!(out[j].to_bits(), 0x8000_0000, "sub 0 (code 9 x 0.0): -0.0");
            assert_eq!(
                out[16 + j].to_bits(),
                0x8000_0000,
                "sub 1 (code 15 x 0x7F): -0.0"
            );
            assert_eq!(out[32 + j].to_bits(), 0x0000_0000, "sub 2 (code 0): +0.0");
            assert_eq!(out[48 + j].to_bits(), 0x0000_0000, "sub 3 (code 8): +0.0");
        }
    }

    /// First-wins ties: at d = 0.5 (scale byte 0x38, anchored by a 6.0 element)
    /// every exact midpoint between adjacent representable magnitudes resolves to
    /// the LOWER LUT index, for both signs — strict `<` in the nearest search, not
    /// round-nearest-even. Expected codes are hand-derived from the LUT scan order
    /// and cross-checked against the pin via the `tie-mid-d0.5` golden vectors.
    #[test]
    fn exact_midpoint_ties_resolve_to_first_lut_index() {
        // Representable true values at d=0.5: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        let sub: [f32; 16] = [
            6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, // low nibbles
            -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.0, // high nibbles
        ];
        let mut x = [0.0_f32; NVFP4_VALUES_PER_BLOCK];
        for s in 0..4 {
            x[s * 16..(s + 1) * 16].copy_from_slice(&sub);
        }
        let wire = encode_nvfp4_block(&x);
        assert!(wire[..4].iter().all(|&b| b == 0x38), "anchor scale 0.5");
        // Expected codes: lows [7,0,1,2,3,4,5,6]; highs [0,9,10,11,12,13,14,0].
        // (-0.25 ties +0.0 at index 0 BEFORE -0.5 at index 9, so it goes positive-zero.)
        let expected_qs: [u8; 8] = [0x07, 0x90, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0x06];
        for s in 0..4 {
            assert_eq!(
                &wire[4 + s * 8..4 + (s + 1) * 8],
                &expected_qs,
                "sub-block {s} tie codes"
            );
        }
    }

    /// Scale saturation boundary around 448 x 6: the largest encoder-reachable
    /// sub-block scale is 0x7E (decoded 224, raw 448); the largest NON-saturating
    /// scale is 0x77 (decoded 120, raw 240). amax = 2688 = 12 x 224 = 448 x 6 hits
    /// 0x7E exactly and round-trips; one ULP either side stays at 0x7E (clamp /
    /// exponent saturation); amax = 6 x 248 carries past raw 240 into saturation.
    #[test]
    fn scale_saturation_boundary_at_448_by_6() {
        let cases: [(f32, u8); 5] = [
            (2688.0, 0x7E),                 // amax/6 == 448 exactly
            (2688.0_f32.next_up(), 0x7E),   // just over: clamps to 448
            (2688.0_f32.next_down(), 0x7E), // just under: exp path still saturates
            (6.0 * 240.0, 0x77),            // largest non-saturating: raw 240
            (6.0 * 248.0, 0x7E),            // round-half-up carry into exp 15
        ];
        for (amax, want_scale) in cases {
            let x = [amax; NVFP4_VALUES_PER_BLOCK];
            let wire = encode_nvfp4_block(&x);
            assert!(
                wire[..4].iter().all(|&b| b == want_scale),
                "amax {amax}: scale {:#04x} want {want_scale:#04x}",
                wire[0]
            );
        }
        // Exact round trip at the boundary: 2688 = code 7 x 224.
        let x = [2688.0_f32; NVFP4_VALUES_PER_BLOCK];
        let y = decode_block_via_tensor_path(&encode_nvfp4_block(&x));
        for v in &y {
            assert_eq!(v.to_bits(), 2688.0_f32.to_bits());
        }
        // 1440 = code 7 x 120 round-trips through the last non-saturating scale.
        let x = [1440.0_f32; NVFP4_VALUES_PER_BLOCK];
        let y = decode_block_via_tensor_path(&encode_nvfp4_block(&x));
        for v in &y {
            assert_eq!(v.to_bits(), 1440.0_f32.to_bits());
        }
    }

    /// The UE4M3 encoder in isolation: exact grid values, half-up rounding, the
    /// subnormal path, the 448 clamp, and the NaN/non-positive zero returns.
    #[test]
    fn fp32_to_ue4m3_semantics() {
        assert_eq!(fp32_to_ue4m3(f32::NAN), 0x00);
        assert_eq!(fp32_to_ue4m3(0.0), 0x00);
        assert_eq!(fp32_to_ue4m3(-1.0), 0x00);
        assert_eq!(fp32_to_ue4m3(f32::from_bits(0x8000_0000)), 0x00); // -0.0
        assert_eq!(fp32_to_ue4m3(f32::INFINITY), 0x7E); // clamp then saturate
        assert_eq!(fp32_to_ue4m3(448.0), 0x7E);
        assert_eq!(fp32_to_ue4m3(1.0), 0x38); // raw 1.0 -> exp 7, man 0
        assert_eq!(fp32_to_ue4m3(240.0), 0x77); // largest non-saturating grid point
        assert_eq!(fp32_to_ue4m3(248.0), 0x7E); // half-up carry into exp 15
        assert_eq!(fp32_to_ue4m3(1.0 / 512.0), 0x01); // subnormal grid
        assert_eq!(fp32_to_ue4m3(0.9 / 512.0), 0x01); // rounds half-up to man 1
        assert_eq!(fp32_to_ue4m3(0.4 / 512.0), 0x00); // below the subnormal floor
                                                      // Every encoder-reachable byte decodes back to a value that re-encodes to
                                                      // itself (grid fixed points).
        for b in 0x01..=0x77u8 {
            let raw = UE4M3_TO_F32[b as usize] * 2.0; // undo the pair-rule half
            assert_eq!(fp32_to_ue4m3(raw), b, "grid fixed point {b:#04x}");
        }
        let raw_7e = UE4M3_TO_F32[0x7E] * 2.0;
        assert_eq!(fp32_to_ue4m3(raw_7e), 0x7E);
    }
}

/// BASALT D-B6 (2026-07-17) — BF16 runnable-lane dequant-parity gate (M-B5 exit
/// condition (a)). bf16 -> f32 is the exact bit-widening `f32::from_bits(u32::from(u16)
/// << 16)`: bf16 stores the high 16 bits of the IEEE-754 f32 encoding, so widening
/// appends 16 zero low bits — lossless, no rounding, bit-deterministic, and
/// definitionally identical to the pin's (`llama.cpp acd79d603`) `ggml_bf16_to_fp32`.
/// The committed fixture `tests/fixtures/dequant/bf16_exact.json` carries the LE wire
/// bytes plus the reference f32 outputs as u32 bit patterns; every comparison here is
/// on `f32::to_bits()` (so +0.0/-0.0 and NaN payloads are distinguished exactly).
#[cfg(test)]
mod bf16_dequant_parity_tests {
    use super::decode_bf16_tensor;

    fn fixture() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dequant")
            .join("bf16_exact.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()));
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("bf16_exact.json parses: {e}"))
    }

    fn hex_u32(s: &str) -> u32 {
        u32::from_str_radix(s.trim_start_matches("0x"), 16)
            .unwrap_or_else(|e| panic!("hex {s:?}: {e}"))
    }

    fn hex_bytes(h: &str) -> Vec<u8> {
        assert!(h.len().is_multiple_of(2), "odd hex length {}", h.len());
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex byte"))
            .collect()
    }

    /// The fixture's provenance must state the definitional lossless-widening
    /// grounds (not a claim it was captured from a binary run), and the reference
    /// bits must BE the exact widening `bf16_u16 << 16` — a self-check so the golden
    /// can never silently encode a wrong value.
    #[test]
    fn bf16_fixture_reference_is_the_exact_widening() {
        let fx = fixture();
        assert_eq!(fx["qtype"].as_str(), Some("BF16"));
        let method = fx["provenance"]["method"]
            .as_str()
            .expect("provenance.method");
        assert!(
            method.contains("Lossless") || method.contains("lossless"),
            "provenance must state bf16->f32 is lossless: {method}"
        );
        assert!(
            fx["provenance"]["pin_equivalence"]
                .as_str()
                .expect("provenance.pin_equivalence")
                .contains("ggml_bf16_to_fp32"),
            "provenance must name the pin's bf16->f32 equivalence"
        );
        let u16s = fx["bf16_u16_hex"].as_array().expect("bf16_u16_hex");
        let refs = fx["ref_f32_bits"].as_array().expect("ref_f32_bits");
        assert_eq!(u16s.len(), refs.len(), "u16 vs ref length");
        for (u, r) in u16s.iter().zip(refs.iter()) {
            let bf16 = hex_u32(u.as_str().expect("u16 hex"));
            let want = hex_u32(r.as_str().expect("ref hex"));
            assert_eq!(
                bf16 << 16,
                want,
                "reference bits must be the exact widening (bf16 {bf16:#06x} << 16)"
            );
        }
    }

    /// `decode_bf16_tensor` reproduces the golden reference bit-for-bit.
    #[test]
    fn decode_bf16_tensor_matches_golden_bit_exact() {
        let fx = fixture();
        let n = fx["n_elements"].as_u64().expect("n_elements") as usize;
        let bytes = hex_bytes(fx["quant_hex"].as_str().expect("quant_hex"));
        assert_eq!(bytes.len(), n * 2, "wire byte length");
        let refs: Vec<u32> = fx["ref_f32_bits"]
            .as_array()
            .expect("ref_f32_bits")
            .iter()
            .map(|r| hex_u32(r.as_str().expect("ref hex")))
            .collect();
        assert_eq!(refs.len(), n, "ref count");

        let out =
            decode_bf16_tensor("fixture:bf16_exact", &bytes, n).expect("bf16 decode must succeed");
        assert_eq!(out.len(), n, "decoded length");
        for (i, (got, want)) in out.iter().zip(refs.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                *want,
                "element {i}: got {:#010x} want {want:#010x}",
                got.to_bits()
            );
        }
    }

    /// Wrong wire length fails closed (the lane never pads or truncates silently).
    #[test]
    fn decode_bf16_tensor_wrong_length_fails_closed() {
        let err = decode_bf16_tensor("t", &[0u8; 6], 2).expect_err("length mismatch must refuse");
        assert!(matches!(
            err,
            crate::error::BackendError::InvalidTensorData(_)
        ));
    }
}

#[cfg(test)]
mod tests {
    /// Serializes the `q8_file_*` tests against each other.
    ///
    /// They assert on DELTAS of the process-global `q8_0_file_read_stats()` counter
    /// and share the process-global Q8 file cache, so any two running concurrently
    /// corrupt each other's measurements — typically surfacing as an off-by-one
    /// read count (`left: 2, right: 1`). That made them intermittently red under
    /// `cargo test`'s default parallelism.
    ///
    /// The flakiness is latent and order-dependent, not random: adding unrelated
    /// tests to this module changed the scheduling enough to make it fire on almost
    /// every run, which is what motivated fixing it here rather than deferring.
    ///
    /// Poisoning is deliberately ignored — one genuinely failing test must not
    /// cascade into spurious failures across the rest of the family.
    fn q8_stats_test_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::tensor::q8_stats_test_lock()
    }

    /// Dequant-parity receipts for the four legacy block formats, each hand-derived
    /// from its `dequantize_row_*` in llama.cpp `ggml/src/ggml-quants.c`.
    ///
    /// The shared trap in all four is the SPLIT-HALF nibble order: the low nibble
    /// of `qs[j]` is element `j` and the high nibble is element `j + 16`, NOT
    /// `j*2` and `j*2+1`. A decoder that emits them adjacently produces the right
    /// multiset in the wrong order — plausible-looking, entirely wrong weights.
    /// Every case below therefore asserts an element from each half.
    #[test]
    fn legacy_quant_dequant_matches_the_reference_layout() {
        use super::{
            decode_iq4_nl_tensor, decode_q4_1_tensor, decode_q5_0_tensor, decode_q5_1_tensor,
        };

        // ---- Q4_1: d = 0.5 (f16 0x3800), m = -4.0 (f16 0xC400), value = q*d + m,
        // q unsigned 0..15 with no bias. qs sweeps low nibbles 0..15 forward while
        // the high nibbles sweep 15..0 backward.
        let mut q4_1 = vec![0x00u8, 0x38, 0x00, 0xC4];
        q4_1.extend([
            0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D,
            0x1E, 0x0F,
        ]);
        let out = decode_q4_1_tensor("q4_1", &q4_1, 32).unwrap();
        assert_eq!(out.len(), 32);
        // element 0 = low nibble 0 -> 0*0.5 - 4.0; element 16 = high nibble 15 -> 3.5
        assert_eq!(out[0], -4.0);
        assert_eq!(out[16], 3.5);
        assert_eq!(out[15], 3.5);
        assert_eq!(out[31], -4.0);

        // ---- Q5_0: d = 1.5 (f16 0x3E00), 5th bit from qh, value = (q - 16)*d.
        let mut q5_0 = vec![0x00u8, 0x3E, 0xAA, 0xAA, 0x53, 0xAA];
        q5_0.extend([
            0x00, 0xFF, 0x0F, 0xF0, 0x81, 0x18, 0x77, 0x8E, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC,
            0xDE, 0xF0,
        ]);
        let out = decode_q5_0_tensor("q5_0", &q5_0, 32).unwrap();
        assert_eq!(out.len(), 32);
        // element 0: low nibble 0, qh bit 0 of 0x53AAAAAA = 0 -> (0-16)*1.5
        assert_eq!(out[0], -24.0);
        // element 1: low nibble 15, qh bit 1 = 1 -> q = 31 -> (31-16)*1.5
        assert_eq!(out[1], 22.5);

        // ---- Q5_1: d = f16 0x2E66, m = -1.5 (f16 0xBE00), value = q*d + m with
        // NO -16 bias (that is Q5_0's). This is the format whose 24-byte block was
        // mis-sized as 22 in the reader; see the layout test in gguf::reader.
        let mut q5_1 = vec![0x66u8, 0x2E, 0x00, 0xBE, 0xAA, 0xAA, 0x55, 0x55];
        q5_1.extend([
            0xF0, 0x0F, 0x0F, 0xF0, 0xE1, 0x1E, 0x1E, 0xE1, 0xD2, 0x2D, 0x2D, 0xD2, 0xC3, 0x3C,
            0x3C, 0xC3,
        ]);
        assert_eq!(q5_1.len(), 24, "Q5_1 block is 24 bytes, not 22");
        let out = decode_q5_1_tensor("q5_1", &q5_1, 32).unwrap();
        assert_eq!(out.len(), 32);
        // element 0: low nibble 0, qh bit 0 = 0 -> q = 0 -> 0*d + m = -1.5
        assert_eq!(out[0], -1.5);
        // element 1: low nibble 15, qh bit 1 = 1 -> q = 31 -> 31*d - 1.5
        let d = super::f16_bits_to_f32(0x2E66);
        assert_eq!(out[1], 31.0 * d - 1.5);

        // ---- IQ4_NL: d = 0.25 (f16 0x3400); the nibble INDEXES kvalues_iq4nl
        // rather than scaling linearly, so a transposed or sign-flipped table is
        // the silent failure mode. kv = [-127,-104,-83,-65,-49,-35,-22,-10,
        //                                  1,  13, 25, 38, 53, 69, 89,113]
        let mut iq4_nl = vec![0x00u8, 0x34];
        iq4_nl.extend([
            0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D,
            0x1E, 0x0F,
        ]);
        let out = decode_iq4_nl_tensor("iq4_nl", &iq4_nl, 32).unwrap();
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], -127.0 * 0.25); // kv[0]
        assert_eq!(out[1], -104.0 * 0.25); // kv[1]
        assert_eq!(out[15], 113.0 * 0.25); // kv[15], last low nibble
        assert_eq!(out[16], 113.0 * 0.25); // kv[15], first high nibble
        assert_eq!(out[31], -127.0 * 0.25); // kv[0], last high nibble
    }

    /// Dequant-parity receipt for TQ2_0, hand-derived from llama.cpp
    /// `ggml/src/ggml-quants.c` `dequantize_row_tq2_0`:
    ///
    /// ```text
    /// for j in {0, 32}: for l in 0..4: for m in 0..32:
    ///     q = (qs[j + m] >> (l * 2)) & 3;  out = (q - 1) * d
    /// ```
    ///
    /// The ORDERING is the part most likely to be silently wrong: the four 2-bit
    /// codes packed into one byte do NOT decode to four adjacent outputs, they
    /// decode 32 apart. A decoder that emits the right multiset in the wrong
    /// order produces plausible-looking garbage, so this pins exact indices.
    #[test]
    fn tq2_0_dequant_matches_the_reference_layout() {
        use super::{decode_tq2_0_tensor, TQ2_0_BLOCK_BYTES};

        let mut block = vec![0x55u8; TQ2_0_BLOCK_BYTES]; // 0b01_01_01_01 -> all codes 1 -> 0.0
                                                         // 0xE4 = 0b11_10_01_00, so codes low->high are 0, 1, 2, 3.
        block[0] = 0xE4;
        // 0x1B = 0b00_01_10_11, so codes low->high are 3, 2, 1, 0.
        block[32] = 0x1B;
        // d = 2.0  (f16 0x4000), so (q - 1) * d yields -2, 0, 2, 4.
        block[64] = 0x00;
        block[65] = 0x40;

        let out = decode_tq2_0_tensor("t", &block, 256).unwrap();
        assert_eq!(
            out.len(),
            256,
            "one TQ2_0 block decodes QK_K = 256 elements"
        );

        // qs[0], planes l = 0..4, at stride 32 within the j = 0 half.
        assert_eq!([out[0], out[32], out[64], out[96]], [-2.0, 0.0, 2.0, 4.0]);
        // qs[32], same planes, in the j = 32 half which starts at element 128.
        assert_eq!(
            [out[128], out[160], out[192], out[224]],
            [4.0, 2.0, 0.0, -2.0]
        );

        // Every other byte is 0x55 -> code 1 -> exactly zero, including the last.
        assert_eq!(out[1], 0.0);
        assert_eq!(out[255], 0.0);
        assert_eq!(out.iter().filter(|v| **v != 0.0).count(), 6);
    }

    /// Dequant-parity receipt for TQ1_0, hand-derived from llama.cpp
    /// `dequantize_row_tq1_0`. TQ1_0 packs five base-3 trits per byte and
    /// recovers each with `xi = ((u8(qs * pow3[n]) as u16) * 3) >> 8`, then
    /// `(xi - 1) * d`. The `u8` truncation of `qs * pow3[n]` is load-bearing —
    /// doing that multiply in a wider type silently changes the trit.
    ///
    /// Layout: qs[0..32] -> 5 planes x 32 (elements 0..160), qs[32..48] -> 5
    /// planes x 16 (elements 160..240), qh[0..4] -> 4 planes x 4 (elements
    /// 240..256). Total 160 + 80 + 16 = 256 = QK_K.
    #[test]
    fn tq1_0_dequant_matches_the_reference_layout() {
        use super::{decode_tq1_0_tensor, TQ1_0_BLOCK_BYTES};

        let mut block = vec![0u8; TQ1_0_BLOCK_BYTES];
        block[0] = 200;
        block[1] = 121;
        // d = 1.0 (f16 0x3C00) so the output is the raw trit.
        block[52] = 0x00;
        block[53] = 0x3C;

        let out = decode_tq1_0_tensor("t", &block, 256).unwrap();
        assert_eq!(
            out.len(),
            256,
            "one TQ1_0 block decodes QK_K = 256 elements"
        );

        // qs[0] = 200. Per plane n, q = u8(200 * pow3[n]), xi = (q * 3) >> 8:
        //   n=0: 200      -> 600  >> 8 = 2 -> trit  1
        //   n=1: u8(600)  =  88 -> 264 >> 8 = 1 -> trit  0
        //   n=2: u8(1800) =   8 ->  24 >> 8 = 0 -> trit -1
        //   n=3: u8(5400) =  24 ->  72 >> 8 = 0 -> trit -1
        //   n=4: u8(16200) = 72 -> 216 >> 8 = 0 -> trit -1
        assert_eq!(
            [out[0], out[32], out[64], out[96], out[128]],
            [1.0, 0.0, -1.0, -1.0, -1.0]
        );

        // qs[1] = 121:
        //   n=0: 121      -> 363 >> 8 = 1 -> trit  0
        //   n=1: u8(363)  = 107 -> 321 >> 8 = 1 -> trit  0
        //   n=2: u8(1089) =  65 -> 195 >> 8 = 0 -> trit -1
        //   n=3: u8(3267) = 195 -> 585 >> 8 = 2 -> trit  1
        //   n=4: u8(9801) =  73 -> 219 >> 8 = 0 -> trit -1
        assert_eq!(
            [out[1], out[33], out[65], out[97], out[129]],
            [0.0, 0.0, -1.0, 1.0, -1.0]
        );

        // qs and qh bytes left at 0 decode to trit -1 (0 * 3 >> 8 = 0, minus 1),
        // so the tail planes are all -1 rather than 0 — a decoder that zero-fills
        // instead of decoding would pass a laxer test but fail this.
        assert_eq!(out[240], -1.0);
        assert_eq!(out[255], -1.0);
    }

    #[test]
    fn q1_0_dequant_matches_the_reference_layout() {
        use super::{decode_q1_0_tensor, Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMENTS};

        // d = 1.0 (f16 0x3C00) so each output is the raw sign.
        let mut block = vec![0u8; Q1_0_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C;
        // qs[0] = 0b0000_0001: only element 0 is set.
        block[2] = 0x01;
        // qs[1] = 0b1000_0000: only element 15 is set.
        block[3] = 0x80;
        // qs[2] = 0b0000_1111: elements 16..19 set, 20..23 clear.
        block[4] = 0x0F;

        let out = decode_q1_0_tensor("t", &block, Q1_0_BLOCK_ELEMENTS).unwrap();
        assert_eq!(
            out.len(),
            128,
            "one Q1_0 block decodes QK1_0 = 128 elements"
        );

        // Element j is bit (j % 8) of byte (j / 8), LSB-FIRST. This is the assertion
        // that fails on an MSB-first decoder: it would put the set bit of qs[0] at
        // element 7 instead of element 0, and the set bit of qs[1] at element 8
        // instead of element 15 — both still "plausible" weights, hence the pin.
        assert_eq!(out[0], 1.0, "qs[0] bit 0 -> element 0");
        assert_eq!(out[1..8], [-1.0; 7], "qs[0] bits 1..7 clear -> -d");
        assert_eq!(out[8..15], [-1.0; 7], "qs[1] bits 0..6 clear -> -d");
        assert_eq!(out[15], 1.0, "qs[1] bit 7 -> element 15");
        assert_eq!(out[16..20], [1.0; 4], "qs[2] low nibble set -> +d");
        assert_eq!(out[20..24], [-1.0; 4], "qs[2] high nibble clear -> -d");
        // Bytes left at 0 decode to -d, not 0.0 — a decoder that zero-fills the tail
        // instead of decoding it would pass a laxer test but fail this.
        assert_eq!(out[127], -1.0);

        // Sign-only means the representable set is exactly {-d, +d}: no code maps to
        // zero, so no element may be 0.0 under a non-zero scale.
        assert!(out.iter().all(|v| v.abs() == 1.0));

        // A negative f16 scale is simply mirrored (pin: `neg_d = -d`), it is not
        // normalised to a magnitude.
        let mut neg = vec![0u8; Q1_0_BLOCK_BYTES];
        neg[0] = 0x00;
        neg[1] = 0xBC; // f16 -1.0
        neg[2] = 0x01;
        let out_neg = decode_q1_0_tensor("t", &neg, Q1_0_BLOCK_ELEMENTS).unwrap();
        assert_eq!(
            out_neg[0], -1.0,
            "set bit under d = -1.0 yields -d's operand"
        );
        assert_eq!(out_neg[1], 1.0, "clear bit under d = -1.0 yields +1.0");

        // Zero scale: set bits give +0.0 and clear bits give -0.0, matching the pin's
        // `neg_d = -d` (rather than a `d * ±1.0` formulation collapsing the sign).
        let zero = vec![0u8; Q1_0_BLOCK_BYTES];
        let out_zero = decode_q1_0_tensor("t", &zero, Q1_0_BLOCK_ELEMENTS).unwrap();
        assert!(
            out_zero[0] == 0.0 && out_zero[0].is_sign_negative(),
            "clear bit under d = 0.0 must be -0.0, not +0.0"
        );
    }

    /// The load-time Q1_0 -> Q8_0 re-encoding must be LOSSLESS, because the whole
    /// GPU-resident lane for Q1_0 files rests on it. Q8_0's representable set is a
    /// superset of Q1_0's `{-d, +d}` (`qs = ±1` against the same scale), so this is
    /// a re-encoding and any difference at all is a bug — compared on `to_bits()`
    /// so a sign-of-zero divergence cannot hide.
    #[test]
    fn q1_0_transcode_is_bit_exact() {
        use super::{decode_q1_0_tensor, q1_0_to_q8_0_blocks, Q1_0_BLOCK_BYTES};

        // Three blocks with different scales and bit patterns, plus the zero-scale
        // block where +0.0 / -0.0 is observable.
        let mut wire = Vec::new();
        for (lo, hi) in [(0x00u8, 0x3Cu8), (0x55, 0x30), (0x00, 0xBC), (0x00, 0x00)] {
            let mut block = vec![0u8; Q1_0_BLOCK_BYTES];
            block[0] = lo;
            block[1] = hi;
            for (k, b) in block[2..].iter_mut().enumerate() {
                // A varied, asymmetric bit pattern so sub-block boundaries matter.
                *b = (k as u8).wrapping_mul(37).wrapping_add(0x5A);
            }
            wire.extend_from_slice(&block);
        }
        let elements = 4 * 128;

        let direct = decode_q1_0_tensor("t", &wire, elements).unwrap();
        let blocks = q1_0_to_q8_0_blocks("t", &wire, elements).unwrap();
        assert_eq!(blocks.len(), elements / 32, "128 values -> 4 Q8_0 blocks");

        // Dequantize the Q8_0 blocks exactly as the engine does: qs as f32 * scale.
        let via_q8: Vec<f32> = blocks
            .iter()
            .flat_map(|b| b.quants.iter().map(|q| *q as f32 * b.scale))
            .collect();

        assert_eq!(via_q8.len(), direct.len());
        for (i, (a, b)) in direct.iter().zip(via_q8.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: Q1_0 decode {a} != Q8_0 re-encode {b}"
            );
        }

        // Every quant must be exactly ±1: any other magnitude means the re-encoding
        // rescaled instead of re-encoding.
        assert!(blocks
            .iter()
            .all(|b| b.quants.iter().all(|q| *q == 1 || *q == -1)));
    }

    #[test]
    fn q1_0_dequant_refuses_a_length_mismatch() {
        use super::{decode_q1_0_tensor, Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMENTS};

        // Element counts must be whole blocks.
        let err = decode_q1_0_tensor("t", &[0u8; Q1_0_BLOCK_BYTES], 64).unwrap_err();
        assert!(
            format!("{err}").contains("not a multiple of"),
            "partial-block element counts refuse: {err}"
        );

        // A short buffer refuses rather than decoding a truncated tail. Without the
        // exact-length check this is the corruption the reader's contiguity check
        // reports against the NEXT tensor.
        let err =
            decode_q1_0_tensor("t", &[0u8; Q1_0_BLOCK_BYTES - 1], Q1_0_BLOCK_ELEMENTS).unwrap_err();
        assert!(
            format!("{err}").contains("Q1_0 wire length"),
            "short wire buffers refuse: {err}"
        );
    }

    #[test]
    fn f32_f16_roundtrip_matches_ieee_rne() {
        use super::{f16_bits_to_f32, f32_to_f16_bits};
        // Exact halves roundtrip exactly.
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.25, 65504.0, -65504.0] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(v)), v);
        }
        // Observed reference KV-cache roundings (f32 value -> f16-stored value).
        let cases = [
            (-0.2714f32, -0.27148438f32),
            (-0.6571, -0.65722656),
            (0.0809, 0.08087158),
        ];
        for (input, expect) in cases {
            let got = f16_bits_to_f32(f32_to_f16_bits(input));
            assert!(
                (got - expect).abs() < 2e-6,
                "{input} -> {got}, want {expect}"
            );
        }
        // Round-to-nearest-EVEN tie: 1 + 2^-11 is exactly halfway between
        // half(1.0) and half(1.0009766); RNE picks the even mantissa (1.0).
        let tie = 1.0f32 + 2.0f32.powi(-11);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(tie)), 1.0);
        // Just above the tie rounds up.
        let above = 1.0f32 + 2.0f32.powi(-11) + 2.0f32.powi(-20);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(above)), 1.0009766);
        // Overflow saturates to inf; tiny values flush toward subnormals/zero.
        assert_eq!(f32_to_f16_bits(1.0e6) & 0x7fff, 0x7c00);
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(1.0e-8)), 0.0);
    }

    /// Independent, spec-literal reference for IQ4_XS dequant (a second implementation used
    /// only to cross-check the optimized [`super::IQ4XSBlock`] decoder). Mirrors ggml's
    /// `dequantize_row_iq4_xs` field-for-field.
    fn iq4_xs_reference_dequant(block: &[u8; super::IQ4_XS_BLOCK_BYTES]) -> [f32; 256] {
        use super::{f16_bits_to_f32, KVALUES_IQ4NL};
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        let mut out = [0.0_f32; 256];
        for ib in 0..8usize {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 0x3) as u8;
            let ls = (low | (high << 4)) as i32;
            let dl = d * (ls - 32) as f32;
            for j in 0..16usize {
                let byte = qs[ib * 16 + j];
                out[ib * 32 + j] = dl * KVALUES_IQ4NL[(byte & 0x0F) as usize];
                out[ib * 32 + j + 16] = dl * KVALUES_IQ4NL[(byte >> 4) as usize];
            }
        }
        out
    }

    #[test]
    fn iq4_xs_block_dequant_matches_hand_computed_golden() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // d = 1.0 (f16 0x3C00). Sub-block scales chosen so dl = ls - 32 is exact per sub-block:
        //   ib:  0   1   2    3    4    5   6    7
        //   ls: 33  32  63    0   24    2  36   26
        //   dl:  1   0  31  -32   -8  -30   4   -6
        // scales_l nibbles (even ib -> low, odd ib -> high of scales_l[ib/2]):
        //   [0]=(ib0 low=1, ib1 high=0)=0x01  [1]=(ib2=15, ib3=0)=0x0F
        //   [2]=(ib4=8,  ib5 high=2)=0x28     [3]=(ib6=4,  ib7 high=10)=0xA4
        // scales_h (2 bits/sub-block): highs 2,2,3,0,1,0,2,1 -> 0x613A.
        let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x613Au16.to_le_bytes());
        bytes[4..8].copy_from_slice(&[0x01, 0x0F, 0x28, 0xA4]);
        // Every quant byte = 0x80: low nibble 0 -> kv[0]=-127, high nibble 8 -> kv[8]=1.
        for b in bytes[8..136].iter_mut() {
            *b = 0x80;
        }
        let mut out = [0.0_f32; 256];
        IQ4XSBlock::from_bytes(&bytes).dequantize(&mut out);

        let dl = [1.0, 0.0, 31.0, -32.0, -8.0, -30.0, 4.0, -6.0];
        for (ib, &d) in dl.iter().enumerate() {
            for j in 0..16 {
                assert_eq!(out[ib * 32 + j], d * -127.0, "ib{ib} j{j} low half");
                assert_eq!(out[ib * 32 + j + 16], d * 1.0, "ib{ib} j{j} high half");
            }
        }
        // And the optimized decoder equals the spec-literal reference on this block.
        assert_eq!(out, iq4_xs_reference_dequant(&bytes));
    }

    #[test]
    fn iq4_xs_block_dequant_matches_reference_over_deterministic_blocks() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // Deterministic LCG fills exercise every codebook index, scale split, and nibble.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..64 {
            let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
            for b in bytes.iter_mut() {
                *b = next();
            }
            let mut out = [0.0_f32; 256];
            IQ4XSBlock::from_bytes(&bytes).dequantize(&mut out);
            let reference = iq4_xs_reference_dequant(&bytes);
            // Bit-for-bit: random f16 `d` bytes can encode NaN/Inf, so compare raw bits
            // (NaN != NaN under `==`). The two implementations run identical float ops, so
            // every lane — finite or not — must match exactly.
            for i in 0..256 {
                assert_eq!(
                    out[i].to_bits(),
                    reference[i].to_bits(),
                    "lane {i} differs for block {bytes:02x?}"
                );
            }
        }
    }

    #[test]
    fn iq4_xs_sub_block_scale_unpacks_low_high_and_minus32_bias() {
        use super::{IQ4XSBlock, IQ4_XS_BLOCK_BYTES};
        // d = 1.0. ib0: scales_l low nibble 5 + scales_h high bits 3 -> ls = 5|48 = 53 -> dl = 21.
        // ib1: low 0 + high 0 -> ls = 0 -> dl = -32. qs byte 0x00 -> both nibbles index kv[0].
        let mut bytes = [0_u8; IQ4_XS_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&0x0003u16.to_le_bytes());
        bytes[4] = 0x05;
        bytes[8] = 0x00;
        let block = IQ4XSBlock::from_bytes(&bytes);
        assert_eq!(block.sub_block_scale(0), 21.0);
        assert_eq!(block.sub_block_scale(1), -32.0);
        let mut out = [0.0_f32; 256];
        block.dequantize(&mut out);
        assert_eq!(out[0], 21.0 * -127.0); // ib0, kv[0]
        assert_eq!(out[32], -32.0 * -127.0); // ib1, kv[0]
    }

    #[test]
    fn iq4_xs_tensor_decode_spans_multiple_blocks_and_rejects_misalignment() {
        use super::{decode_iq4_xs_tensor, IQ4_XS_BLOCK_BYTES};
        // Two full super-blocks of distinct constant bytes.
        let mut bytes = Vec::new();
        for fill in [0x11u8, 0x22u8] {
            let mut blk = vec![0u8; IQ4_XS_BLOCK_BYTES];
            blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
            for b in blk[2..].iter_mut() {
                *b = fill;
            }
            bytes.extend_from_slice(&blk);
        }
        let decoded = decode_iq4_xs_tensor("blk.iq4xs", &bytes, 512).unwrap();
        assert_eq!(decoded.len(), 512);
        let mut b0 = [0u8; IQ4_XS_BLOCK_BYTES];
        b0.copy_from_slice(&bytes[0..IQ4_XS_BLOCK_BYTES]);
        assert_eq!(&decoded[0..256], &iq4_xs_reference_dequant(&b0)[..]);

        // One byte short of a block boundary must fail closed, not truncate silently.
        let err = decode_iq4_xs_tensor("blk.bad", &bytes[..bytes.len() - 1], 512).unwrap_err();
        assert!(format!("{err}").contains("not aligned"), "got: {err}");
    }

    #[test]
    fn iq4_nl_and_iq4_xs_share_the_same_codebook() {
        use super::{IQ4NLBlock, IQ4_NL_BLOCK_BYTES, KVALUES_IQ4NL};
        // The shared const carries the exact ggml kvalues_iq4nl table.
        assert_eq!(
            KVALUES_IQ4NL,
            [
                -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0,
                53.0, 69.0, 89.0, 113.0
            ]
        );
        // IQ4_NL still indexes that same table (d = 1.0, qs byte 0xF0 -> kv[0] then kv[15]).
        let mut bytes = [0u8; IQ4_NL_BLOCK_BYTES];
        bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        bytes[2] = 0xF0;
        let mut out = [0.0_f32; 32];
        IQ4NLBlock::from_bytes(&bytes).dequantize(&mut out);
        assert_eq!(out[0], KVALUES_IQ4NL[0]);
        assert_eq!(out[16], KVALUES_IQ4NL[15]);
    }

    use super::{
        f16_bits_to_f32, parse_byte_count, q8_0_file_read_stats, q8_file_cache_get,
        q8_file_cache_insert, q8_repack_tensor_enabled_for_flags, q8_repack_x86_tensor_enabled,
        with_q8_file_cache_capacity_override, CpuTensor, Q8_0Block, Q8_0FileBacking,
        Q8_0PackedRows4, Q8_0PackedRows4Interleave, TensorShape, Q3_K_BLOCK_BYTES,
        Q8_0_BLOCK_BYTES,
    };
    use crate::gguf::GgufTensorType;
    use crate::test_support::env_lock;
    use std::sync::Arc;

    #[test]
    fn q8_file_cache_disabled_path_does_not_store_or_hit() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-disabled-{}", std::process::id()));

        let start = q8_0_file_read_stats();
        q8_file_cache_insert(path.clone(), 10, b"abcdefgh");
        let mut out = [0_u8; 8];
        assert!(!q8_file_cache_get(&path, 10, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_hit_bytes, 0);
        assert_eq!(stats.cache_entries, 0);
        assert_eq!(stats.cache_bytes, 0);
        assert_eq!(stats.cache_capacity_bytes, 0);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_disabled_scale_read_decodes_from_direct_read() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-disabled-scale-read-{}",
            std::process::id()
        ));
        let scale_bits = 0x3800_u16;
        let mut bytes = Vec::with_capacity(Q8_0_BLOCK_BYTES);
        bytes.extend_from_slice(&scale_bits.to_le_bytes());
        bytes.extend(0..32_u8);
        std::fs::write(&path, &bytes).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 0, 1);
        let mut out = [0_u8; Q8_0_BLOCK_BYTES];
        let mut scales = [0.0_f32; 1];

        let start = q8_0_file_read_stats();
        let reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut out, 0, &mut scales)
            .unwrap();
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert!(!reused);
        assert_eq!(out.as_slice(), bytes.as_slice());
        assert_eq!(scales, [f16_bits_to_f32(scale_bits)]);
        assert_eq!(stats.read_calls, 1);
        assert_eq!(stats.read_bytes, Q8_0_BLOCK_BYTES as u64);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert_eq!(stats.cache_entries, 0);
        assert_eq!(stats.cache_bytes, 0);
        assert_eq!(stats.cache_capacity_bytes, 0);
        let _ = std::fs::remove_file(path);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_backing_subviews_share_one_cached_file_handle() {
        let _q8_stats_guard = q8_stats_test_lock();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-shared-backing-handle-{}",
            std::process::id()
        ));
        let mut bytes = vec![0_u8; 3 * Q8_0_BLOCK_BYTES];
        bytes[Q8_0_BLOCK_BYTES..2 * Q8_0_BLOCK_BYTES].fill(0x5a);
        std::fs::write(&path, &bytes).unwrap();

        let parent = Q8_0FileBacking::new(path.clone(), 0, 3);
        let subview = parent.clone_with_offset_and_blocks(Q8_0_BLOCK_BYTES as u64, 1);
        assert!(!parent.file_handle_cached());
        assert!(!subview.file_handle_cached());

        let subview_file = subview.file().unwrap();
        assert!(parent.file_handle_cached());
        let parent_file = parent.file().unwrap();
        assert!(std::sync::Arc::ptr_eq(&parent_file, &subview_file));

        let mut out = [0_u8; Q8_0_BLOCK_BYTES];
        subview
            .read_exact_at_cached(&mut out, Q8_0_BLOCK_BYTES as u64)
            .unwrap();
        assert_eq!(out, [0x5a; Q8_0_BLOCK_BYTES]);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_backed_embedding_rejects_absolute_row_offset_overflow() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let tensor = CpuTensor::q8_0_file_backed_linear(
            "token_embd.weight",
            TensorShape { dims: vec![2, 32] },
            Q8_0FileBacking::new("unused.gguf".into(), u64::MAX - 16, 2),
        );

        let err = tensor.embedding_lookup(&[1], "embedding").unwrap_err();

        assert!(
            err.to_string()
                .contains("file-backed q8_0 embedding absolute row byte offset overflow"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn q8_block_backed_embedding_dequantizes_selected_rows() {
        let row0 = Q8_0Block {
            scale: 0.5,
            quants: [2; 32],
        };
        let row1 = Q8_0Block {
            scale: 0.25,
            quants: [-4; 32],
        };
        let tensor = CpuTensor::from_q8_0_blocks(
            "token_embd.weight",
            TensorShape { dims: vec![2, 32] },
            vec![row0, row1],
        )
        .unwrap();

        let embedding = tensor.embedding_lookup(&[1, 0], "embedding").unwrap();

        assert_eq!(embedding.shape.dims, vec![2, 32]);
        assert_eq!(&embedding.data[..32], &[-1.0; 32]);
        assert_eq!(&embedding.data[32..], &[1.0; 32]);
    }

    #[test]
    fn q3_k_wire_backed_embedding_dequantizes_without_dense_data() {
        let mut tensor =
            CpuTensor::from_f32("token_embd.weight", vec![2, 256], vec![0.0; 512]).unwrap();
        tensor.source_type = Some(GgufTensorType::Q3K);
        tensor.data.clear();
        tensor.q3_k_wire_bytes = Some(Arc::new(vec![0; 2 * Q3_K_BLOCK_BYTES]));

        let embedding = tensor.embedding_lookup(&[1], "embedding").unwrap();

        assert_eq!(embedding.shape.dims, vec![1, 256]);
        assert!(embedding.data.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn embedding_lookup_without_any_backing_returns_typed_error() {
        let mut tensor =
            CpuTensor::from_f32("token_embd.weight", vec![1, 256], vec![0.0; 256]).unwrap();
        tensor.source_type = Some(GgufTensorType::Q3K);
        tensor.data.clear();

        let error = tensor.embedding_lookup(&[0], "embedding").unwrap_err();

        assert!(
            error.to_string().contains("has no readable backing"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn q8_packed_rows4_sidecars_stay_opt_in_per_layout() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_MAC_Q8_REPACK");
        std::env::remove_var("CAMELID_X86_Q8_REPACK");
        std::env::remove_var("CAMELID_Q8_0_PACKED_4X4_DOT");
        std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");

        let make_weight = || {
            let rows = 4;
            let cols = 32;
            let blocks = (0..rows)
                .map(|row| Q8_0Block {
                    scale: 0.25 + row as f32 * 0.125,
                    quants: std::array::from_fn(|idx| (idx as i8 % 17) - 8),
                })
                .collect::<Vec<_>>();
            let data = blocks
                .iter()
                .flat_map(|block| block.quants.iter().map(|q| block.scale * f32::from(*q)))
                .collect::<Vec<_>>();

            CpuTensor::from_f32_with_q8_0_blocks(
                "blk.0.attn_q.weight",
                vec![rows, cols],
                data,
                blocks,
            )
            .unwrap()
        };

        let default_weight = make_weight();
        assert!(default_weight.q8_0_packed_rows4_4x4.is_none());
        assert!(default_weight.q8_0_packed_rows4_4x8.is_none());

        std::env::set_var("CAMELID_Q8_0_PACKED_4X4_DOT", "on");
        let packed_4x4_weight = make_weight();
        assert!(packed_4x4_weight.q8_0_packed_rows4_4x4.is_some());
        assert!(packed_4x4_weight.q8_0_packed_rows4_4x8.is_none());

        std::env::remove_var("CAMELID_Q8_0_PACKED_4X4_DOT");
        std::env::set_var("CAMELID_Q8_0_PACKED_4X8_DOT", "on");
        let packed_4x8_weight = make_weight();
        assert!(packed_4x8_weight.q8_0_packed_rows4_4x4.is_none());
        assert!(packed_4x8_weight.q8_0_packed_rows4_4x8.is_some());

        std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");
        std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
        let mac_repack_weight = make_weight();
        assert!(mac_repack_weight.q8_0_packed_rows4_4x4.is_none());
        assert!(mac_repack_weight.q8_0_packed_rows4_4x8.is_none());
        assert!(mac_repack_weight.q8_0_runtime_storage.is_none());

        let non_family_mac_repack_weight = CpuTensor::from_f32_with_q8_0_blocks(
            "blk.0.ffn_up.weight",
            vec![4, 32],
            vec![0.0; 128],
            vec![
                Q8_0Block {
                    scale: 1.0,
                    quants: [0; 32],
                };
                4
            ],
        )
        .unwrap();
        assert!(non_family_mac_repack_weight.q8_0_packed_rows4_4x4.is_none());
        assert!(non_family_mac_repack_weight.q8_0_packed_rows4_4x8.is_none());

        std::env::remove_var("CAMELID_MAC_Q8_REPACK");
        std::env::set_var("CAMELID_X86_Q8_REPACK", "on");
        let x86_repack_weight = make_weight();
        assert!(x86_repack_weight.q8_0_packed_rows4_4x4.is_none());
        assert!(x86_repack_weight.q8_0_packed_rows4_4x8.is_none());
        assert!(x86_repack_weight.q8_0_runtime_storage.is_none());

        std::env::remove_var("CAMELID_X86_Q8_REPACK");
    }

    #[test]
    fn q8_0_vnni_pack_requires_raw_q8_bytes_for_scale_bits() {
        let blocks = vec![
            Q8_0Block {
                scale: f16_bits_to_f32(0x3001),
                quants: [3; 32],
            };
            16
        ];
        let packed =
            Q8_0PackedRows4::from_rows(16, 1, Q8_0PackedRows4Interleave::I8, &blocks).unwrap();

        assert!(
            packed.vnni_packed.is_none(),
            "from_rows cannot prove original GGUF fp16 scale bits, so VNNI packing must be raw-byte only"
        );
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn q8_0_vnni_pack_from_q8_0_bytes_matches_llamacpp_tile16_layout() {
        let _env_guard = env_lock();
        std::env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        let rows = 16;
        let blocks_per_row = 2;
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * Q8_0_BLOCK_BYTES);
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let scale_bits = 0x3000_u16 + row as u16 * 17 + block as u16;
                bytes.extend_from_slice(&scale_bits.to_le_bytes());
                bytes.extend((0..32).map(|idx| {
                    (idx as i8)
                        .wrapping_mul(3)
                        .wrapping_add(row as i8 * 5)
                        .wrapping_sub(block as i8 * 7) as u8
                }));
            }
        }

        let packed = Q8_0PackedRows4::from_q8_0_bytes(
            rows,
            blocks_per_row,
            Q8_0PackedRows4Interleave::I8,
            &bytes,
        )
        .unwrap();
        let vnni = packed.vnni_packed.as_ref().expect("VNNI sidecar");
        assert_eq!(vnni.rows, rows);
        assert_eq!(vnni.blocks_per_row, blocks_per_row);
        assert_eq!(vnni.tiles.len(), blocks_per_row);

        for block in 0..blocks_per_row {
            let tile = &vnni.tiles[block];
            for n in 0..16 {
                let raw_start = (n * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
                assert_eq!(
                    tile.scale_f16[n],
                    u16::from_le_bytes([bytes[raw_start], bytes[raw_start + 1]])
                );
                assert_eq!(tile.scale_f32[n], f16_bits_to_f32(tile.scale_f16[n]));
                let qs = &bytes[raw_start + 2..raw_start + Q8_0_BLOCK_BYTES];
                let expected_comp = 128
                    * qs.iter()
                        .fold(0_i32, |acc, value| acc + i32::from(*value as i8));
                assert_eq!(tile.comp[n], expected_comp);
                for g in 0..8 {
                    for r in 0..4 {
                        assert_eq!(tile.quants[g * 64 + n * 4 + r], qs[g * 4 + r] as i8);
                    }
                }
            }
        }

        std::env::remove_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE");
    }

    #[test]
    fn q8_x86_repack_family_includes_output_projection_only() {
        assert!(q8_repack_x86_tensor_enabled("output.weight"));
        assert!(q8_repack_x86_tensor_enabled("blk.0.attn_output.weight"));
        assert!(q8_repack_x86_tensor_enabled("blk.0.ffn_down.weight"));
        assert!(!q8_repack_x86_tensor_enabled("token_embd.weight"));
        assert!(!q8_repack_x86_tensor_enabled("blk.0.attn_norm.weight"));
    }

    #[test]
    fn q8_runtime_repack_route_stays_default_off_and_family_scoped() {
        assert!(!q8_repack_tensor_enabled_for_flags(
            "output.weight",
            false,
            false
        ));
        assert!(!q8_repack_tensor_enabled_for_flags(
            "blk.0.attn_output.weight",
            false,
            false
        ));
        assert!(!q8_repack_tensor_enabled_for_flags(
            "token_embd.weight",
            true,
            true
        ));
        assert!(!q8_repack_tensor_enabled_for_flags(
            "blk.0.attn_norm.weight",
            true,
            true
        ));
        assert!(q8_repack_tensor_enabled_for_flags(
            "output.weight",
            true,
            false
        ));
        assert!(q8_repack_tensor_enabled_for_flags(
            "output.weight",
            false,
            true
        ));
        assert!(q8_repack_tensor_enabled_for_flags(
            "blk.0.ffn_down.weight",
            true,
            false
        ));
        assert!(q8_repack_tensor_enabled_for_flags(
            "blk.0.attn_q.weight",
            false,
            true
        ));
    }

    #[test]
    fn q8_runtime_repack_linear_shape_preserves_token_major_output_route() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_MAC_Q8_REPACK");
        std::env::remove_var("CAMELID_X86_Q8_REPACK");

        let hidden_vocab = TensorShape { dims: vec![32, 64] };
        let vocab_hidden = TensorShape { dims: vec![64, 32] };

        assert_eq!(
            super::q8_repack_linear_shape("output.weight", &hidden_vocab),
            None
        );

        std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
        assert_eq!(
            super::q8_repack_linear_shape("output.weight", &hidden_vocab),
            Some((64, 32))
        );
        assert_eq!(
            super::q8_repack_linear_shape("output.weight", &vocab_hidden),
            Some((64, 32))
        );
        assert_eq!(
            super::q8_repack_linear_shape("blk.0.attn_output.weight", &hidden_vocab),
            Some((64, 32))
        );
        assert_eq!(
            super::q8_repack_linear_shape("token_embd.weight", &vocab_hidden),
            None
        );

        std::env::remove_var("CAMELID_MAC_Q8_REPACK");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q8_x86_repack_includes_output_projection_runtime_storage() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_X86_Q8_REPACK");
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        let shape = TensorShape { dims: vec![32, 64] };
        let bytes = vec![0_u8; 64 * Q8_0_BLOCK_BYTES];

        assert!(
            super::q8_0_runtime_packed_rows4_for_tensor("output.weight", &shape, &bytes)
                .unwrap()
                .is_none()
        );

        std::env::set_var("CAMELID_X86_Q8_REPACK", "on");
        let Some(super::Q8_0RuntimeStorage::PackedRows4(packed)) =
            super::q8_0_runtime_packed_rows4_for_tensor("output.weight", &shape, &bytes).unwrap()
        else {
            panic!("expected x86 output projection Q8_0 runtime-packed rows4 storage");
        };
        assert_eq!(packed.rows, 64);
        assert_eq!(packed.blocks_per_row, 1);
        assert_eq!(packed.interleave, super::Q8_0PackedRows4Interleave::I8);

        let Some(super::Q8_0RuntimeStorage::PackedRows4(attn_output_packed)) =
            super::q8_0_runtime_packed_rows4_for_tensor("blk.0.attn_output.weight", &shape, &bytes)
                .unwrap()
        else {
            panic!("expected x86 attention output Q8_0 runtime-packed rows4 storage");
        };
        assert_eq!(attn_output_packed.rows, 64);
        assert_eq!(attn_output_packed.blocks_per_row, 1);
        assert_eq!(
            attn_output_packed.interleave,
            super::Q8_0PackedRows4Interleave::I8
        );

        std::env::remove_var("CAMELID_X86_Q8_REPACK");
    }

    #[test]
    fn wire_pages_linear_carries_wire_bytes_and_file_backing() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::remove_var("CAMELID_MAC_Q8_REPACK");

        let rows = 8;
        let blocks_per_row = 2;
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * Q8_0_BLOCK_BYTES);
        for block in 0..rows * blocks_per_row {
            bytes.extend_from_slice(&(0x3400_u16 + block as u16).to_le_bytes());
            bytes.extend((0..32).map(|idx| ((idx * 7 + block) % 251) as u8));
        }

        let path = std::env::temp_dir().join(format!(
            "camelid-wire-pages-linear-{}-{}.bin",
            std::process::id(),
            rows
        ));
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&bytes).unwrap();
        }
        let gguf = crate::gguf::GgufFile {
            path: path.clone(),
            version: 3,
            tensor_count: 1,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::new(),
            tensors: vec![crate::gguf::GgufTensorDescriptor {
                name: "blk.0.attn_q.weight".to_string(),
                dimensions: vec![(blocks_per_row * 32) as u64, rows as u64],
                tensor_type: crate::gguf::GgufTensorType::Q8_0,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: bytes.len() as u64,
            }],
        };
        let store = super::TensorStore::open(&path, &gguf);

        let tensor = store
            .load_q8_0_wire_pages_linear("blk.0.attn_q.weight")
            .unwrap();
        // Wire pages hold the file's exact wire bytes, page-aligned, and the tensor
        // keeps file backing for CPU fallback paths; nothing is materialized.
        let pages = tensor
            .q8_0_wire_pages
            .as_ref()
            .expect("wire pages attached");
        assert_eq!(pages.bytes(), &bytes[..]);
        assert_eq!(pages.base_ptr() as usize % crate::wire_mmap::page_size(), 0);
        assert!(tensor.q8_0_file_backing.is_some());
        assert!(tensor.q8_0_blocks.is_none());
        assert!(tensor.data.is_empty());

        // The embedding gather decodes rows straight from the wire pages.
        let mut shaped = tensor.clone();
        shaped.shape = TensorShape {
            dims: vec![rows, blocks_per_row * 32],
        };
        let row = shaped.embedding_lookup(&[3], "row").unwrap();
        let backed = store
            .load_q8_0_file_backed_tensor("blk.0.attn_q.weight")
            .unwrap();
        let mut backed_shaped = backed;
        backed_shaped.shape = TensorShape {
            dims: vec![rows, blocks_per_row * 32],
        };
        let expected = backed_shaped.embedding_lookup(&[3], "expected").unwrap();
        assert_eq!(row.data, expected.data);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn q8_file_backed_alias_stays_strict_when_repack_and_sidecars_are_enabled() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::set_var("CAMELID_MAC_Q8_REPACK", "on");
        std::env::set_var("CAMELID_X86_Q8_REPACK", "on");
        std::env::set_var("CAMELID_Q8_0_PACKED_4X4_DOT", "on");
        std::env::set_var("CAMELID_Q8_0_PACKED_4X8_DOT", "on");

        let source_shape = TensorShape { dims: vec![32, 64] };
        let rows = 64;
        let blocks_per_row = 1;
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * Q8_0_BLOCK_BYTES);
        for row in 0..rows {
            bytes.extend_from_slice(&(0x3000_u16 + row as u16).to_le_bytes());
            bytes.extend((0..32).map(|idx| (idx as i8).wrapping_sub(row as i8) as u8));
        }

        let path = std::env::temp_dir().join(format!(
            "camelid-tied-output-q8-{}-{}.bin",
            std::process::id(),
            rows
        ));
        {
            use std::io::Write;
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&bytes).unwrap();
        }
        let gguf = crate::gguf::GgufFile {
            path: path.clone(),
            version: 3,
            tensor_count: 1,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: std::collections::BTreeMap::new(),
            tensors: vec![crate::gguf::GgufTensorDescriptor {
                name: "token_embd.weight".to_string(),
                dimensions: source_shape.dims.iter().map(|dim| *dim as u64).collect(),
                tensor_type: crate::gguf::GgufTensorType::Q8_0,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: bytes.len() as u64,
            }],
        };
        let store = super::TensorStore::open(&path, &gguf);

        let embedding = store
            .load_q8_0_file_backed_tensor("token_embd.weight")
            .unwrap();
        assert_eq!(embedding.name, "token_embd.weight");
        assert!(embedding.q8_0_runtime_storage.is_none());
        assert!(embedding.q8_0_file_backing.is_some());

        let output = store
            .load_q8_0_file_backed_tensor_as("token_embd.weight", "output.weight")
            .unwrap();
        assert_eq!(output.name, "output.weight");
        assert_eq!(output.shape, source_shape);
        assert!(output.q8_0_file_backing.is_some());
        assert!(output.q8_0_runtime_storage.is_none());
        assert!(output.q8_0_packed_rows4_4x4.is_none());
        assert!(output.q8_0_packed_rows4_4x8.is_none());
        assert!(output.q8_0_blocks.is_none());
        assert!(output.data.is_empty());

        std::fs::remove_file(path).unwrap();
        std::env::remove_var("CAMELID_MAC_Q8_REPACK");
        std::env::remove_var("CAMELID_X86_Q8_REPACK");
        std::env::remove_var("CAMELID_Q8_0_PACKED_4X4_DOT");
        std::env::remove_var("CAMELID_Q8_0_PACKED_4X8_DOT");
    }

    #[test]
    fn q8_file_cache_zero_capacity_clears_retained_entries_on_use() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-zero-clear-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");

        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let mut disabled_out = [0_u8; 4];
        assert!(!q8_file_cache_get(&path, 100, &mut disabled_out));

        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let mut stale_out = [0_u8; 16];
        assert!(!q8_file_cache_get(&path, 100, &mut stale_out));
        let stats = q8_0_file_read_stats();
        assert_eq!(stats.cache_entries, 0);
        assert_eq!(stats.cache_bytes, 0);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_scoped_capacity_override_is_bounded_and_restored() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-scoped-{}", std::process::id()));

        let (hit, scoped_stats) = with_q8_file_cache_capacity_override(Some(8), || {
            q8_file_cache_insert(path.clone(), 10, b"abcdefgh");
            let mut out = [0_u8; 8];
            let start = q8_0_file_read_stats();
            let hit = q8_file_cache_get(&path, 10, &mut out);
            (hit, q8_0_file_read_stats().saturating_delta_since(start))
        });

        assert!(hit);
        assert_eq!(scoped_stats.cache_hits, 1);
        assert_eq!(scoped_stats.cache_hit_bytes, 8);
        assert_eq!(scoped_stats.cache_entries, 1);
        assert_eq!(scoped_stats.cache_bytes, 8);
        assert_eq!(scoped_stats.cache_capacity_bytes, 8);

        let restored_stats = q8_0_file_read_stats();
        assert_eq!(restored_stats.cache_capacity_bytes, 0);
        assert_eq!(restored_stats.cache_entries, 0);
        assert_eq!(restored_stats.cache_bytes, 0);
    }

    #[test]
    fn q8_byte_count_env_parser_accepts_binary_suffixes() {
        assert_eq!(parse_byte_count("1024"), Some(1024));
        assert_eq!(parse_byte_count("1 KiB"), Some(1024));
        assert_eq!(parse_byte_count("2_mib"), Some(2 * 1024 * 1024));
        assert_eq!(parse_byte_count("3GB"), Some(3 * 1024 * 1024 * 1024));
        assert_eq!(parse_byte_count(""), None);
        assert_eq!(parse_byte_count("1.5MiB"), None);
        assert_eq!(parse_byte_count("many"), None);
    }

    #[test]
    fn q8_file_cache_serves_matching_chunks_and_evicts_to_capacity() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "8");
        let first_path =
            std::env::temp_dir().join(format!("camelid-q8-cache-first-{}", std::process::id()));
        let second_path =
            std::env::temp_dir().join(format!("camelid-q8-cache-second-{}", std::process::id()));
        q8_file_cache_insert(first_path.clone(), 10, b"abcdefgh");
        let mut out = [0_u8; 8];
        let start = q8_0_file_read_stats();
        assert!(q8_file_cache_get(&first_path, 10, &mut out));
        assert_eq!(&out, b"abcdefgh");
        let after_first = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(after_first.cache_hits, 1);
        assert_eq!(after_first.cache_hit_bytes, 8);
        assert_eq!(after_first.cache_entries, 1);
        assert_eq!(after_first.cache_bytes, 8);
        assert_eq!(after_first.cache_capacity_bytes, 8);

        q8_file_cache_insert(second_path.clone(), 20, b"ijklmnop");
        let mut evicted = [0_u8; 8];
        assert!(!q8_file_cache_get(&first_path, 10, &mut evicted));
        assert!(q8_file_cache_get(&second_path, 20, &mut evicted));
        assert_eq!(&evicted, b"ijklmnop");
        let after_second = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(after_second.cache_hits, 2);
        assert_eq!(after_second.cache_hit_bytes, 16);
        assert_eq!(after_second.cache_entries, 1);
        assert_eq!(after_second.cache_bytes, 8);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_serves_subranges_from_retained_chunks() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-subrange-{}", std::process::id()));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 4];
        assert!(q8_file_cache_get(&path, 104, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"efgh");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 4);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_coalesces_adjacent_chunks_for_cross_boundary_reuse() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-adjacent-{}", std::process::id()));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 8];
        assert!(q8_file_cache_get(&path, 104, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"efghijkl");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 8);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_reports_miss_insert_merge_and_eviction_stats() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-stats-{}", std::process::id()));
        let other_path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-stats-other-{}",
            std::process::id()
        ));

        let start = q8_0_file_read_stats();
        let mut miss = [0_u8; 4];
        assert!(!q8_file_cache_get(&path, 100, &mut miss));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");
        let mut hit = [0_u8; 8];
        assert!(q8_file_cache_get(&path, 104, &mut hit));
        q8_file_cache_insert(other_path, 200, b"qrstuvwx");
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&hit, b"efghijkl");
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_miss_bytes, 4);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 8);
        assert_eq!(stats.cache_inserts, 3);
        assert_eq!(stats.cache_insert_bytes, 32);
        assert_eq!(stats.cache_merges, 1);
        assert_eq!(stats.cache_merged_bytes, 16);
        assert_eq!(stats.cache_evictions, 1);
        assert_eq!(stats.cache_evicted_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 8);
        assert_eq!(stats.cache_capacity_bytes, 16);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_trims_coalesced_stream_to_newest_capacity_window() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-trim-window-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 108, b"ijklmnop");
        q8_file_cache_insert(path.clone(), 116, b"qrstuvwx");

        let start = q8_0_file_read_stats();
        let mut evicted = [0_u8; 8];
        let mut retained = [0_u8; 16];
        assert!(!q8_file_cache_get(&path, 100, &mut evicted));
        assert!(q8_file_cache_get(&path, 108, &mut retained));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&retained, b"ijklmnopqrstuvwx");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_coalesces_overlapping_chunks_with_newest_bytes() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "12");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-overlap-{}", std::process::id()));
        q8_file_cache_insert(path.clone(), 100, b"abcdefgh");
        q8_file_cache_insert(path.clone(), 104, b"WXYZmnop");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 10];
        assert!(q8_file_cache_get(&path, 102, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"cdWXYZmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 10);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 12);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_skips_reinserting_identical_fully_covered_subranges() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-covered-{}", std::process::id()));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");
        q8_file_cache_insert(path.clone(), 104, b"efgh");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 16];
        assert!(q8_file_cache_get(&path, 100, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"abcdefghijklmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_keeps_newest_bytes_for_conflicting_covered_subranges() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-covered-conflict-{}",
            std::process::id()
        ));
        q8_file_cache_insert(path.clone(), 100, b"abcdefghijklmnop");
        q8_file_cache_insert(path.clone(), 104, b"WXYZ");

        let start = q8_0_file_read_stats();
        let mut out = [0_u8; 16];
        assert!(q8_file_cache_get(&path, 100, &mut out));
        let stats = q8_0_file_read_stats().saturating_delta_since(start);

        assert_eq!(&out, b"abcdWXYZijklmnop");
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_hit_bytes, 16);
        assert_eq!(stats.cache_entries, 1);
        assert_eq!(stats.cache_bytes, 16);
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn q8_file_cache_file_read_reuses_partial_overlap_and_reads_gaps() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-partial-file-read-{}",
            std::process::id()
        ));
        std::fs::write(&path, b"abcdefghijklmnopqrstuvwxyz").unwrap();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "32");
        let backing = Q8_0FileBacking::new(path.clone(), 0, 1);

        let start = q8_0_file_read_stats();
        let mut seed = [0_u8; 8];
        backing.read_exact_at_cached(&mut seed, 0).unwrap();
        let seed_stats = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(&seed, b"abcdefgh");
        assert_eq!(seed_stats.read_calls, 1);
        assert_eq!(seed_stats.read_bytes, 8);
        assert_eq!(seed_stats.cache_misses, 1);
        assert_eq!(seed_stats.cache_miss_bytes, 8);

        let after_seed = q8_0_file_read_stats();
        let mut partial = [0_u8; 16];
        backing.read_exact_at_cached(&mut partial, 4).unwrap();
        let partial_stats = q8_0_file_read_stats().saturating_delta_since(after_seed);
        assert_eq!(&partial, b"efghijklmnopqrst");
        assert_eq!(partial_stats.read_calls, 1);
        assert_eq!(partial_stats.read_bytes, 12);
        assert_eq!(partial_stats.cache_hits, 1);
        assert_eq!(partial_stats.cache_hit_bytes, 4);
        assert_eq!(partial_stats.cache_misses, 1);
        assert_eq!(partial_stats.cache_miss_bytes, 12);
        assert_eq!(partial_stats.cache_entries, 1);
        assert_eq!(partial_stats.cache_bytes, 20);

        let after_partial = q8_0_file_read_stats();
        let mut cached_again = [0_u8; 16];
        backing.read_exact_at_cached(&mut cached_again, 4).unwrap();
        let cached_stats = q8_0_file_read_stats().saturating_delta_since(after_partial);
        assert_eq!(&cached_again, b"efghijklmnopqrst");
        assert_eq!(cached_stats.read_calls, 0);
        assert_eq!(cached_stats.read_bytes, 0);
        assert_eq!(cached_stats.cache_hits, 1);
        assert_eq!(cached_stats.cache_hit_bytes, 16);
        assert_eq!(cached_stats.cache_misses, 0);
        assert_eq!(cached_stats.cache_miss_bytes, 0);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_cache_reuses_decoded_scales_on_full_block_hits() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "128");
        let _ = q8_0_file_read_stats();
        let path =
            std::env::temp_dir().join(format!("camelid-q8-cache-scales-{}", std::process::id()));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        bytes.extend_from_slice(&0x4000_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        std::fs::write(&path, &bytes).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 0, 2);

        let start = q8_0_file_read_stats();
        let mut first = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut first_scales = vec![0.0_f32; 2];
        let first_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut first, 0, &mut first_scales)
            .unwrap();
        let first_stats = q8_0_file_read_stats().saturating_delta_since(start);
        assert!(!first_reused);
        assert_eq!(first, bytes);
        assert_eq!(first_scales, vec![1.0, 2.0]);
        assert_eq!(first_stats.read_calls, 1);
        assert_eq!(first_stats.cache_misses, 1);
        assert_eq!(first_stats.cache_decoded_scale_hits, 0);
        assert_eq!(first_stats.cache_decoded_scale_hit_blocks, 0);

        let after_first = q8_0_file_read_stats();
        let mut second = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut second_scales = vec![-1.0_f32; 2];
        let second_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut second, 0, &mut second_scales)
            .unwrap();
        let second_stats = q8_0_file_read_stats().saturating_delta_since(after_first);

        assert!(second_reused);
        assert_eq!(second, bytes);
        assert_eq!(second_scales, vec![1.0, 2.0]);
        assert_eq!(second_stats.read_calls, 0);
        assert_eq!(second_stats.read_bytes, 0);
        assert_eq!(second_stats.cache_hits, 1);
        assert_eq!(second_stats.cache_hit_bytes, (Q8_0_BLOCK_BYTES * 2) as u64);
        assert_eq!(second_stats.cache_decoded_scale_hits, 1);
        assert_eq!(second_stats.cache_decoded_scale_hit_blocks, 2);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_cache_reuses_decoded_scales_on_partial_block_hits() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "256");
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-partial-scales-{}",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        for scale_bits in [0x3c00_u16, 0x4000, 0x4200, 0x4400] {
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        }
        std::fs::write(&path, &bytes).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 0, 4);

        let mut seed = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut seed_scales = vec![-1.0_f32; 2];
        let seed_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut seed, 0, &mut seed_scales)
            .unwrap();
        assert!(!seed_reused);
        assert_eq!(seed, bytes[..Q8_0_BLOCK_BYTES * 2]);
        assert_eq!(seed_scales, vec![1.0, 2.0]);

        let after_seed = q8_0_file_read_stats();
        let mut partial = vec![0_u8; Q8_0_BLOCK_BYTES * 3];
        let mut partial_scales = vec![-1.0_f32; 3];
        let partial_reused = backing
            .read_exact_at_cached_with_q8_0_scales(
                &mut partial,
                Q8_0_BLOCK_BYTES as u64,
                &mut partial_scales,
            )
            .unwrap();
        let partial_stats = q8_0_file_read_stats().saturating_delta_since(after_seed);

        assert!(partial_reused);
        assert_eq!(partial, bytes[Q8_0_BLOCK_BYTES..]);
        assert_eq!(partial_scales, vec![2.0, 3.0, 4.0]);
        assert_eq!(partial_stats.read_calls, 1);
        assert_eq!(partial_stats.read_bytes, (Q8_0_BLOCK_BYTES * 2) as u64);
        assert_eq!(partial_stats.cache_hits, 1);
        assert_eq!(partial_stats.cache_hit_bytes, Q8_0_BLOCK_BYTES as u64);
        assert_eq!(partial_stats.cache_misses, 1);
        assert_eq!(
            partial_stats.cache_miss_bytes,
            (Q8_0_BLOCK_BYTES * 2) as u64
        );
        assert_eq!(partial_stats.cache_decoded_scale_hits, 1);
        assert_eq!(partial_stats.cache_decoded_scale_hit_blocks, 1);

        let after_partial = q8_0_file_read_stats();
        let mut cached_again = vec![0_u8; Q8_0_BLOCK_BYTES * 3];
        let mut cached_again_scales = vec![-1.0_f32; 3];
        let cached_again_reused = backing
            .read_exact_at_cached_with_q8_0_scales(
                &mut cached_again,
                Q8_0_BLOCK_BYTES as u64,
                &mut cached_again_scales,
            )
            .unwrap();
        let cached_again_stats = q8_0_file_read_stats().saturating_delta_since(after_partial);

        assert!(cached_again_reused);
        assert_eq!(cached_again, bytes[Q8_0_BLOCK_BYTES..]);
        assert_eq!(cached_again_scales, vec![2.0, 3.0, 4.0]);
        assert_eq!(cached_again_stats.read_calls, 0);
        assert_eq!(cached_again_stats.read_bytes, 0);
        assert_eq!(cached_again_stats.cache_hits, 1);
        assert_eq!(
            cached_again_stats.cache_hit_bytes,
            (Q8_0_BLOCK_BYTES * 3) as u64
        );
        assert_eq!(cached_again_stats.cache_decoded_scale_hits, 1);
        assert_eq!(cached_again_stats.cache_decoded_scale_hit_blocks, 3);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_cache_retains_decoded_scales_after_coalesced_trim() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var(
            "CAMELID_Q8_0_FILE_CACHE_BYTES",
            (Q8_0_BLOCK_BYTES * 3).to_string(),
        );
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-scale-trim-{}",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        for scale_bits in [0x3c00_u16, 0x4000, 0x4200, 0x4400] {
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        }
        std::fs::write(&path, &bytes).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 0, 4);

        let mut first = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut first_scales = vec![-1.0_f32; 2];
        let first_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut first, 0, &mut first_scales)
            .unwrap();
        assert!(!first_reused);
        assert_eq!(first_scales, vec![1.0, 2.0]);

        let mut second = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut second_scales = vec![-1.0_f32; 2];
        let second_reused = backing
            .read_exact_at_cached_with_q8_0_scales(
                &mut second,
                (Q8_0_BLOCK_BYTES * 2) as u64,
                &mut second_scales,
            )
            .unwrap();
        assert!(!second_reused);
        assert_eq!(second_scales, vec![3.0, 4.0]);

        let after_trim = q8_0_file_read_stats();
        let mut retained = vec![0_u8; Q8_0_BLOCK_BYTES * 3];
        let mut retained_scales = vec![-1.0_f32; 3];
        let retained_reused = backing
            .read_exact_at_cached_with_q8_0_scales(
                &mut retained,
                Q8_0_BLOCK_BYTES as u64,
                &mut retained_scales,
            )
            .unwrap();
        let retained_stats = q8_0_file_read_stats().saturating_delta_since(after_trim);

        assert!(retained_reused);
        assert_eq!(retained, bytes[Q8_0_BLOCK_BYTES..]);
        assert_eq!(retained_scales, vec![2.0, 3.0, 4.0]);
        assert_eq!(retained_stats.read_calls, 0);
        assert_eq!(retained_stats.read_bytes, 0);
        assert_eq!(retained_stats.cache_hits, 1);
        assert_eq!(
            retained_stats.cache_hit_bytes,
            (Q8_0_BLOCK_BYTES * 3) as u64
        );
        assert_eq!(retained_stats.cache_entries, 1);
        assert_eq!(retained_stats.cache_bytes, (Q8_0_BLOCK_BYTES * 3) as u64);
        assert_eq!(retained_stats.cache_decoded_scale_hits, 1);
        assert_eq!(retained_stats.cache_decoded_scale_hit_blocks, 3);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_cache_promotes_decoded_scales_after_byte_only_hit() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "128");
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-cache-scale-upgrade-{}",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        bytes.extend_from_slice(&0x4000_u16.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0_u8, Q8_0_BLOCK_BYTES - 2));
        std::fs::write(&path, &bytes).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 0, 2);

        let start = q8_0_file_read_stats();
        let mut byte_only_seed = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        backing
            .read_exact_at_cached(&mut byte_only_seed, 0)
            .unwrap();
        let seed_stats = q8_0_file_read_stats().saturating_delta_since(start);
        assert_eq!(byte_only_seed, bytes);
        assert_eq!(seed_stats.read_calls, 1);
        assert_eq!(seed_stats.cache_misses, 1);

        let after_seed = q8_0_file_read_stats();
        let mut first_scale_hit = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut first_scales = vec![-1.0_f32; 2];
        let first_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut first_scale_hit, 0, &mut first_scales)
            .unwrap();
        let first_stats = q8_0_file_read_stats().saturating_delta_since(after_seed);
        assert!(!first_reused);
        assert_eq!(first_scale_hit, bytes);
        assert_eq!(first_scales, vec![1.0, 2.0]);
        assert_eq!(first_stats.read_calls, 0);
        assert_eq!(first_stats.read_bytes, 0);
        assert_eq!(first_stats.cache_hits, 1);
        assert_eq!(first_stats.cache_hit_bytes, (Q8_0_BLOCK_BYTES * 2) as u64);
        assert_eq!(first_stats.cache_decoded_scale_hits, 0);
        assert_eq!(first_stats.cache_decoded_scale_hit_blocks, 0);

        let after_upgrade = q8_0_file_read_stats();
        let mut second_scale_hit = vec![0_u8; Q8_0_BLOCK_BYTES * 2];
        let mut second_scales = vec![-1.0_f32; 2];
        let second_reused = backing
            .read_exact_at_cached_with_q8_0_scales(&mut second_scale_hit, 0, &mut second_scales)
            .unwrap();
        let second_stats = q8_0_file_read_stats().saturating_delta_since(after_upgrade);
        assert!(second_reused);
        assert_eq!(second_scale_hit, bytes);
        assert_eq!(second_scales, vec![1.0, 2.0]);
        assert_eq!(second_stats.read_calls, 0);
        assert_eq!(second_stats.read_bytes, 0);
        assert_eq!(second_stats.cache_hits, 1);
        assert_eq!(second_stats.cache_hit_bytes, (Q8_0_BLOCK_BYTES * 2) as u64);
        assert_eq!(second_stats.cache_decoded_scale_hits, 1);
        assert_eq!(second_stats.cache_decoded_scale_hit_blocks, 2);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_backing_rejects_reads_outside_declared_storage_before_file_io() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path =
            std::env::temp_dir().join(format!("camelid-q8-backing-bounds-{}", std::process::id()));
        std::fs::write(&path, (0_u8..64).collect::<Vec<_>>()).unwrap();
        let backing = Q8_0FileBacking::new(path.clone(), 8, 1);

        let mut valid = [0_u8; 34];
        backing.read_exact_at_cached(&mut valid, 8).unwrap();
        assert_eq!(&valid[..4], &[8, 9, 10, 11]);

        let after_valid = q8_0_file_read_stats();
        let mut before = [0_u8; 1];
        let before_err = backing.read_exact_at_cached(&mut before, 7).unwrap_err();
        let after_before_err = q8_0_file_read_stats().saturating_delta_since(after_valid);
        assert!(before_err.to_string().contains("before backing offset 8"));
        assert_eq!(after_before_err.read_calls, 0);
        assert_eq!(after_before_err.read_bytes, 0);

        let after_before_err_absolute = q8_0_file_read_stats();
        let mut beyond = [0_u8; 2];
        let beyond_err = backing
            .read_exact_at_cached(&mut beyond, 8 + 34 - 1)
            .unwrap_err();
        let after_beyond_err =
            q8_0_file_read_stats().saturating_delta_since(after_before_err_absolute);
        assert!(beyond_err
            .to_string()
            .contains("exceeds backing storage range"));
        assert_eq!(after_beyond_err.read_calls, 0);
        assert_eq!(after_beyond_err.read_bytes, 0);

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_file_backing_rejects_nonempty_zero_block_reads_before_file_io() {
        let _q8_stats_guard = q8_stats_test_lock();
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "32");
        let _ = q8_0_file_read_stats();
        let path = std::env::temp_dir().join(format!(
            "camelid-q8-zero-block-bounds-{}",
            std::process::id()
        ));
        let backing = Q8_0FileBacking::new(path.clone(), 128, 0);

        let mut empty = [];
        backing.read_exact_at_cached(&mut empty, 128).unwrap();
        assert!(!backing.file_handle_cached());

        let after_empty = q8_0_file_read_stats();
        let mut out = [0_u8; 1];
        let err = backing.read_exact_at_cached(&mut out, 128).unwrap_err();
        let stats = q8_0_file_read_stats().saturating_delta_since(after_empty);

        assert!(err.to_string().contains("exceeds backing storage range"));
        assert_eq!(stats.read_calls, 0);
        assert_eq!(stats.read_bytes, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
        assert!(!backing.file_handle_cached());

        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
    }

    #[test]
    fn matmul_rhs_transposed_handles_single_row_vectors() {
        let lhs = CpuTensor::from_f32("lhs", vec![1, 5], vec![1.0, -2.0, 3.0, 0.5, 4.0]).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![3, 5],
            vec![
                2.0, 0.0, -1.0, 4.0, 0.5, // first output row
                -3.0, 1.0, 0.0, 2.0, -0.5, // second output row
                1.0, 1.0, 1.0, 1.0, 1.0, // third output row
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![1, 3]);
        assert_eq!(actual.data, vec![3.0, -6.0, 6.5]);
    }

    #[test]
    fn matmul_rhs_transposed_handles_rectangular_batches() {
        let lhs = CpuTensor::from_f32(
            "lhs",
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
            ],
        )
        .unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![2, 3],
            vec![
                7.0, 8.0, 9.0, // output 0
                1.0, 0.0, -1.0, // output 1
            ],
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        assert_eq!(actual.shape.dims, vec![2, 2]);
        assert_eq!(actual.data, vec![50.0, -2.0, 122.0, -2.0]);
    }

    #[test]
    fn matmul_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..lhs_values.len() * output_width)
            .map(|idx| ((idx % 37) as f32 - 18.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs",
            vec![lhs_values.len(), output_width],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|col| {
                lhs_values
                    .iter()
                    .enumerate()
                    .map(|(inner, lhs_value)| lhs_value * rhs_values[inner * output_width + col])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        for (idx, &actual_val) in actual.data.iter().enumerate() {
            let expected_val = expected[idx];
            assert!(
                (actual_val - expected_val).abs() < 1e-4,
                "mismatch at index {idx}: actual {actual_val}, expected {expected_val}"
            );
        }
    }

    #[test]
    fn matmul_rhs_transposed_wide_output_matches_reference() {
        let lhs_values = vec![1.0, -2.0, 0.5, 3.0, -0.25];
        let output_width = 1031;
        let rhs_values = (0..output_width * lhs_values.len())
            .map(|idx| ((idx % 41) as f32 - 20.0) * 0.01)
            .collect::<Vec<_>>();
        let lhs =
            CpuTensor::from_f32("lhs", vec![1, lhs_values.len()], lhs_values.clone()).unwrap();
        let rhs = CpuTensor::from_f32(
            "rhs_t",
            vec![output_width, lhs_values.len()],
            rhs_values.clone(),
        )
        .unwrap();

        let actual = lhs.matmul_rhs_transposed(&rhs, "out").unwrap();

        let expected = (0..output_width)
            .map(|row| {
                let row_start = row * lhs_values.len();
                lhs_values
                    .iter()
                    .zip(&rhs_values[row_start..row_start + lhs_values.len()])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.shape.dims, vec![1, output_width]);
        for (idx, &actual_val) in actual.data.iter().enumerate() {
            let expected_val = expected[idx];
            assert!(
                (actual_val - expected_val).abs() < 1e-4,
                "mismatch at index {idx}: actual {actual_val}, expected {expected_val}"
            );
        }
    }

    #[test]
    fn converts_f16_bits_to_f32() {
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }
}
