use std::{
    cell::Cell,
    collections::HashMap,
    env,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
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
    BackendError, Result,
};

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

fn q8_0_runtime_packed_rows4_for_tensor(
    name: &str,
    shape: &TensorShape,
    q8_0_bytes: &[u8],
) -> Result<Option<Q8_0RuntimeStorage>> {
    if env_flag_disabled("CAMELID_Q8_0_BLOCK_DOT") {
        return Ok(None);
    }
    let Some((rows, cols)) = q8_repack_linear_shape(name, shape) else {
        return Ok(None);
    };
    if !rows.is_multiple_of(4) || !cols.is_multiple_of(Q8_0_BLOCK_VALUES) {
        return Ok(None);
    }
    let started = Instant::now();
    let packed = Q8_0PackedRows4::from_q8_0_bytes(
        rows,
        cols / Q8_0_BLOCK_VALUES,
        Q8_0PackedRows4Interleave::I8,
        q8_0_bytes,
    )?;
    if q8_0_pack_trace_enabled() {
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
    if shape.dims.len() != 2 {
        return Ok(None);
    }
    let rows = shape.dims[0];
    let cols = shape.dims[1];
    if !rows.is_multiple_of(4) || !cols.is_multiple_of(32) {
        return Ok(None);
    }
    let started = Instant::now();
    let packed = Q8_0PackedRows4::from_rows(rows, cols / 32, interleave, blocks)?;
    if q8_0_pack_trace_enabled() {
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
            file.read_exact_at(out, offset)
                .map_err(|source| BackendError::Io {
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
            file.read_exact_at(&mut out[range.out_start..out_end], range_offset)
                .map_err(|source| BackendError::Io {
                    path: self.path.clone(),
                    source,
                })?;
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

#[derive(Debug, Clone, PartialEq)]
pub struct CpuTensor {
    pub name: String,
    pub shape: TensorShape,
    pub dtype: RuntimeDType,
    pub source_type: Option<GgufTensorType>,
    pub q8_0_blocks: Option<Vec<Q8_0Block>>,
    pub q8_0_packed_rows4_4x4: Option<Q8_0PackedRows4>,
    pub q8_0_packed_rows4_4x8: Option<Q8_0PackedRows4>,
    pub q8_0_runtime_storage: Option<Q8_0RuntimeStorage>,
    pub q8_0_file_backing: Option<Q8_0FileBacking>,
    pub q8_0_split_file_backing: Option<Vec<Q8_0FileBacking>>,
    pub data: Vec<f32>,
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
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_split_file_backing: None,
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
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_split_file_backing: None,
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
            q8_0_packed_rows4_4x4,
            q8_0_packed_rows4_4x8,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_split_file_backing: None,
            data: Vec::new(),
        })
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
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: Some(backing),
            q8_0_split_file_backing: None,
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
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: Some(Q8_0RuntimeStorage::PackedRows4(packed)),
            q8_0_file_backing: None,
            q8_0_split_file_backing: None,
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
            q8_0_packed_rows4_4x4: None,
            q8_0_packed_rows4_4x8: None,
            q8_0_runtime_storage: None,
            q8_0_file_backing: None,
            q8_0_split_file_backing: Some(backings),
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
                    *o = a + b;
                });
        } else {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                use std::arch::aarch64::{vld1q_f32, vaddq_f32, vst1q_f32};
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
                for i in 0..len {
                    out[i] = self.data[i] + rhs.data[i];
                }
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
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
                for i in 0..len {
                    out[i] = self.data[i] * rhs.data[i];
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
            for i in 0..len {
                let a = self.data[i];
                let b = rhs.data[i];
                out[i] = (a / (1.0 + (-a).exp())) * b;
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
            for i in 0..len {
                let x = self.data[i];
                out[i] = x / (1.0 + (-x).exp());
            }
        }
        Self::from_f32(name, self.shape.dims.clone(), out)
    }

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn rms_norm_neon(input: &[f32], weight: &[f32], out: &mut [f32], cols: usize, eps: f32) {
    use std::arch::aarch64::{
        vld1q_f32, vmulq_f32, vaddq_f32, vdupq_n_f32, vst1q_f32, vget_low_f32, vget_high_f32, vpadd_f32, vget_lane_f32
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
        let mut out = vec![0.0; self.data.len()];

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if should_parallelize_linear_output(self.data.len()) {
                out.par_chunks_mut(cols)
                    .zip(self.data.par_chunks(cols))
                    .for_each(|(out_row, in_row)| {
                        unsafe {
                            Self::rms_norm_neon(in_row, &weight.data, out_row, cols, eps);
                        }
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

        Self::from_f32(name, self.shape.dims.clone(), out)
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
        if let Some(backing) = self.q8_0_file_backing.as_ref() {
            return self.embedding_lookup_q8_0_file_backed(token_ids, name, vocab, width, backing);
        }
        if let Some(blocks) = self.q8_0_blocks.as_deref() {
            return self.embedding_lookup_q8_0_block_backed(token_ids, name, vocab, width, blocks);
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
        vld1q_f32, vmulq_f32, vaddq_f32, vdupq_n_f32, vget_low_f32, vget_high_f32, vpadd_f32, vget_lane_f32
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
    match env::var("CAMELID_PARALLEL_LINEAR") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes" | "enabled"
        ),
        Err(_) => false,
    }
}

fn parallel_linear_min_outputs() -> usize {
    env::var("CAMELID_PARALLEL_LINEAR_MIN_OUTPUTS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PARALLEL_LINEAR_MIN_OUTPUTS)
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

    pub fn load_q8_0_file_backed_tensor(&self, name: &str) -> Result<CpuTensor> {
        self.load_q8_0_file_backed_tensor_as(name, name)
    }

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
        if q8_repack_tensor_enabled(tensor_name) {
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
        }
        let mut tensor = CpuTensor::q8_0_file_backed_linear(
            tensor_name,
            shape.clone(),
            Q8_0FileBacking::new(
                self.path.clone(),
                desc.absolute_offset,
                expected_elements / 32,
            ),
        );
        if q8_0_packed_rows4_enabled_for_tensor(tensor_name, Q8_0PackedRows4Interleave::I4)
            || q8_0_packed_rows4_enabled_for_tensor(tensor_name, Q8_0PackedRows4Interleave::I8)
        {
            let bytes = self.tensor_bytes(source_name)?;
            let blocks = decode_q8_0_blocks(source_name, &bytes, expected_elements)?;
            tensor.q8_0_packed_rows4_4x4 = q8_0_packed_rows4_for_shape(
                tensor_name,
                &shape,
                Some(&blocks),
                Q8_0PackedRows4Interleave::I4,
            )?;
            tensor.q8_0_packed_rows4_4x8 = q8_0_packed_rows4_for_shape(
                tensor_name,
                &shape,
                Some(&blocks),
                Q8_0PackedRows4Interleave::I8,
            )?;
        }
        Ok(tensor)
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
            GgufTensorType::Q2K => decode_q2_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q3K => decode_q3_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q4K => decode_q4_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q5K => decode_q5_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q6K => decode_q6_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::Q8K => decode_q8_k_tensor(name, &bytes, expected_elements)?,
            GgufTensorType::IQ4NL => decode_iq4_nl_tensor(name, &bytes, expected_elements)?,
            other => {
                return Err(BackendError::UnsupportedTensorType(format!(
                    "tensor {name} has unsupported storage type {other:?}; supported for CPU f32 load: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL"
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
            q8_0_packed_rows4_4x4,
            q8_0_packed_rows4_4x8,
            q8_0_runtime_storage: None,
            q8_0_file_backing,
            q8_0_split_file_backing: None,
            data,
        })
    }
}

fn decode_f32_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
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

fn decode_f16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
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

fn decode_bf16_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
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

fn decode_q8_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
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

fn f16_bits_to_f32(bits: u16) -> f32 {
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
        const KVALUES: [f32; 16] = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
            69.0, 89.0, 113.0,
        ];
        for j in 0..16 {
            let byte = self.qs[j];
            out[j] = d * KVALUES[(byte & 0x0F) as usize];
            out[j + 16] = d * KVALUES[(byte >> 4) as usize];
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

// Flat dequantization to f32 helpers
fn decode_q4_0_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_0_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let scale = block.scale_f32();
        for val in block.unpack_values() {
            out.push(val as f32 * scale);
        }
    }
    Ok(out)
}

fn decode_q4_1_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_1_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
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
    let blocks = decode_q5_0_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
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
    let blocks = decode_q5_1_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
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

fn decode_q2_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q2_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q3_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q3_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q4_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q4_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q5_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q5_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q6_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q6_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_q8_k_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_q8_k_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

fn decode_iq4_nl_tensor(name: &str, bytes: &[u8], expected_elements: usize) -> Result<Vec<f32>> {
    let blocks = decode_iq4_nl_blocks(bytes).map_err(|e| BackendError::InvalidTensorData(format!("{name}: {e}")))?;
    let mut out = Vec::with_capacity(expected_elements);
    for block in blocks {
        let mut values = [0.0_f32; 32];
        block.dequantize(&mut values);
        out.extend_from_slice(&values);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        f16_bits_to_f32, parse_byte_count, q8_0_file_read_stats, q8_file_cache_get,
        q8_file_cache_insert, q8_repack_tensor_enabled_for_flags, q8_repack_x86_tensor_enabled,
        with_q8_file_cache_capacity_override, CpuTensor, Q8_0Block, Q8_0FileBacking,
        Q8_0PackedRows4, Q8_0PackedRows4Interleave, TensorShape, Q8_0_BLOCK_BYTES,
    };
    use crate::test_support::env_lock;

    #[test]
    fn q8_file_cache_disabled_path_does_not_store_or_hit() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-disabled-{}",
            std::process::id()
        ));

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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-disabled-scale-read-{}",
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
    fn q8_file_backed_embedding_rejects_absolute_row_offset_overflow() {
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn q8_x86_repack_materializes_tied_embedding_as_output_runtime_storage() {
        let _env_guard = env_lock();
        std::env::remove_var("CAMELID_Q8_0_BLOCK_DOT");
        std::env::set_var("CAMELID_X86_Q8_REPACK", "on");

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
        assert!(output.q8_0_file_backing.is_none());
        let Some(super::Q8_0RuntimeStorage::PackedRows4(packed)) = output.q8_0_runtime_storage
        else {
            panic!("expected tied output alias to materialize output-compatible rows4 storage");
        };
        assert_eq!(packed.rows, rows);
        assert_eq!(packed.blocks_per_row, blocks_per_row);
        assert_eq!(packed.interleave, Q8_0PackedRows4Interleave::I8);

        std::fs::remove_file(path).unwrap();
        std::env::remove_var("CAMELID_X86_Q8_REPACK");
    }

    #[test]
    fn q8_file_cache_zero_capacity_clears_retained_entries_on_use() {
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-zero-clear-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::remove_var("CAMELID_Q8_0_FILE_CACHE_BYTES");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-scoped-{}",
            std::process::id()
        ));

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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "8");
        let first_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-first-{}",
            std::process::id()
        ));
        let second_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-second-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-subrange-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-adjacent-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-stats-{}",
            std::process::id()
        ));
        let other_path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-stats-other-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-trim-window-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "12");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-overlap-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-covered-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "16");
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-covered-conflict-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-partial-file-read-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "128");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-scales-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "256");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-partial-scales-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var(
            "CAMELID_Q8_0_FILE_CACHE_BYTES",
            (Q8_0_BLOCK_BYTES * 3).to_string(),
        );
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-scale-trim-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "128");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-cache-scale-upgrade-{}",
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "0");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-backing-bounds-{}",
            std::process::id()
        ));
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
        let _env_guard = env_lock();
        let _q8_guard = crate::test_support::q8_file_state_lock();
        std::env::set_var("CAMELID_Q8_0_FILE_CACHE_BYTES", "32");
        let _ = q8_0_file_read_stats();
        let path = std::path::PathBuf::from(format!(
            "/tmp/camelid-q8-zero-block-bounds-{}",
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
