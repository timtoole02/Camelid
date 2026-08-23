//! Read-only mmap of a GGUF file for zero-copy weight access.
//!
//! GGUF Q8_0 tensor data on disk is already in the exact 34-byte f16-scale wire
//! layout the Metal wire kernels consume (`CAMELID_METAL_WIRE`). Loading today
//! streams the file into 36-byte f32-scale CPU blocks and converts back to wire
//! on GPU upload — two copies and two conversions of bytes that never needed to
//! change. This module maps the file once and exposes page-aligned windows that
//! Metal can wrap with `newBufferWithBytesNoCopy`, so the file's own page-cache
//! pages back the GPU reads directly: no load-time read loop, no conversion, no
//! upload copy, and clean (file-backed, evictable) resident memory.
//!
//! Lifetime rule: a mapping must outlive every Metal buffer created over it.
//! Consumers hold the `Arc<GgufWireMmap>` alongside each derived buffer.

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::{platform_fs::read_exact_at, BackendError, Result};

/// System page size, used for window/buffer alignment.
#[cfg(unix)]
pub fn page_size() -> usize {
    // SAFETY: sysconf(_SC_PAGESIZE) has no preconditions.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// System page size, used for window/buffer alignment.
#[cfg(windows)]
pub fn page_size() -> usize {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: GetSystemInfo only writes into the provided SYSTEM_INFO.
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    unsafe { GetSystemInfo(&mut info) };
    info.dwPageSize as usize
}

/// Non-faulting physical-residency snapshot for a file-backed mapping.
/// `mincore` reports page-cache residency without reading any mapped byte.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireMmapResidencySnapshot {
    pub(crate) page_size_bytes: usize,
    pub(crate) mapped_bytes: usize,
    pub(crate) total_pages: usize,
    pub(crate) resident_pages: usize,
    pub(crate) resident_bytes: usize,
}

/// One validated page-aligned window inside a wire mapping. This geometry is
/// shared by non-faulting residency sampling and targeted cache-discard
/// advisories, so callers cannot accidentally round the end past the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireMmapAlignedRange {
    pub(crate) aligned_offset: usize,
    pub(crate) mapped_bytes: usize,
}

/// Result of one targeted `MADV_DONTNEED` advisory. The before/after snapshots
/// are both collected with `mincore`, which does not fault the advised pages.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WireMmapDiscardSnapshot {
    pub(crate) range: WireMmapAlignedRange,
    pub(crate) resident_before_bytes: usize,
    pub(crate) resident_after_bytes: usize,
    /// Resident bytes that `mincore` confirmed became nonresident across this
    /// advisory. This is an observed transition, not the requested capacity.
    pub(crate) confirmed_discarded_bytes: usize,
}

/// Round an exact byte range outwards to system-page boundaries while proving
/// that the resulting window remains inside `mapped_len`.
pub(crate) fn aligned_page_range(
    offset: u64,
    len: usize,
    mapped_len: usize,
    page_size_bytes: usize,
) -> Result<WireMmapAlignedRange> {
    if len == 0
        || mapped_len == 0
        || page_size_bytes == 0
        || !mapped_len.is_multiple_of(page_size_bytes)
    {
        return Err(BackendError::InvalidTensorData(format!(
            "invalid wire page-range geometry: offset={offset} len={len} mapped_len={mapped_len} page_size={page_size_bytes}"
        )));
    }
    let offset = usize::try_from(offset).map_err(|_| {
        BackendError::InvalidTensorData(format!(
            "wire page-range offset does not fit usize: {offset}"
        ))
    })?;
    let end = offset.checked_add(len).ok_or_else(|| {
        BackendError::InvalidTensorData(format!(
            "wire page-range overflow at offset={offset} len={len}"
        ))
    })?;
    if end > mapped_len {
        return Err(BackendError::InvalidTensorData(format!(
            "wire page range {offset}..{end} exceeds mapped length {mapped_len}"
        )));
    }
    let aligned_offset = offset - (offset % page_size_bytes);
    let aligned_end = end
        .div_ceil(page_size_bytes)
        .checked_mul(page_size_bytes)
        .ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "wire page-range aligned end overflowed for offset={offset} len={len} page_size={page_size_bytes}"
            ))
        })?;
    if aligned_end > mapped_len {
        return Err(BackendError::InvalidTensorData(format!(
            "wire aligned page range {aligned_offset}..{aligned_end} exceeds mapped length {mapped_len}"
        )));
    }
    Ok(WireMmapAlignedRange {
        aligned_offset,
        mapped_bytes: aligned_end - aligned_offset,
    })
}

/// Page-align and merge exact byte ranges. Overlapping and directly adjacent
/// windows coalesce, preventing a final cleanup pass from advising shared
/// tensor-boundary pages more than once.
pub(crate) fn merge_aligned_page_ranges(
    ranges: &[(u64, usize)],
    mapped_len: usize,
    page_size_bytes: usize,
) -> Result<Vec<WireMmapAlignedRange>> {
    let mut aligned = ranges
        .iter()
        .map(|&(offset, len)| aligned_page_range(offset, len, mapped_len, page_size_bytes))
        .collect::<Result<Vec<_>>>()?;
    aligned.sort_unstable_by_key(|range| range.aligned_offset);
    let mut merged: Vec<WireMmapAlignedRange> = Vec::with_capacity(aligned.len());
    for range in aligned {
        let range_end = range
            .aligned_offset
            .checked_add(range.mapped_bytes)
            .ok_or_else(|| {
                BackendError::InvalidTensorData(
                    "wire aligned page range overflowed while merging".into(),
                )
            })?;
        if let Some(previous) = merged.last_mut() {
            let previous_end = previous
                .aligned_offset
                .checked_add(previous.mapped_bytes)
                .ok_or_else(|| {
                    BackendError::InvalidTensorData(
                        "wire previous aligned page range overflowed while merging".into(),
                    )
                })?;
            if range.aligned_offset <= previous_end {
                previous.mapped_bytes = previous_end.max(range_end) - previous.aligned_offset;
                continue;
            }
        }
        merged.push(range);
    }
    Ok(merged)
}

/// A read-only, shared, page-cache-backed mapping of an entire GGUF file.
///
/// Unix maps the file with `mmap(PROT_READ, MAP_SHARED)`; Windows maps it with
/// `memmap2` (`CreateFileMapping`/`MapViewOfFile`). Both expose the same
/// immutable, byte-addressable, shareable view, so the file's own page cache
/// backs reads directly with no load-time copy. The public API is identical on
/// both platforms.
#[cfg(unix)]
#[derive(Debug)]
pub struct GgufWireMmap {
    ptr: *const u8,
    /// Mapped length: the file length rounded up to the page size by the kernel;
    /// bytes past EOF within the final page read as zero.
    mapped_len: usize,
    file_len: u64,
    path: PathBuf,
}

// SAFETY: the mapping is immutable (PROT_READ) for its entire lifetime and the
// underlying pages are managed by the kernel; concurrent reads are safe.
#[cfg(unix)]
unsafe impl Send for GgufWireMmap {}
#[cfg(unix)]
unsafe impl Sync for GgufWireMmap {}

#[cfg(unix)]
impl Drop for GgufWireMmap {
    fn drop(&mut self) {
        // SAFETY: ptr/mapped_len came from a successful mmap and are unmapped once.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.mapped_len);
        }
    }
}

#[cfg(unix)]
impl GgufWireMmap {
    /// Map `path` read-only. The mapping covers the whole file.
    pub fn map(path: &Path) -> Result<Arc<Self>> {
        let file = File::open(path).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "wire mmap open failed for {}: {err}",
                path.display()
            ))
        })?;
        Self::map_file(&file, path)
    }

    /// Map an already-open descriptor. Callers that also issue descriptor-
    /// scoped advisory I/O can thereby prove the mmap and advisory target the
    /// same vnode, even if the pathname is concurrently replaced.
    pub(crate) fn map_file(file: &File, path: &Path) -> Result<Arc<Self>> {
        use std::os::unix::io::AsRawFd;
        let file_len = file
            .metadata()
            .map_err(|err| {
                BackendError::InvalidTensorData(format!(
                    "wire mmap metadata failed for {}: {err}",
                    path.display()
                ))
            })?
            .len();
        if file_len == 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap refused for empty file {}",
                path.display()
            )));
        }
        let page = page_size();
        let mapped_len = (file_len as usize).div_ceil(page) * page;
        // SAFETY: fd is a valid open file; length is non-zero; PROT_READ +
        // MAP_SHARED of a regular file has no aliasing hazards for readers.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap failed for {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Arc::new(Self {
            ptr: ptr as *const u8,
            mapped_len,
            file_len,
            path: path.to_path_buf(),
        }))
    }

    /// Snapshot the mapping's page-cache residency without faulting pages in.
    pub(crate) fn residency_snapshot(&self) -> Result<WireMmapResidencySnapshot> {
        let page_size_bytes = page_size();
        if page_size_bytes == 0 || !self.mapped_len.is_multiple_of(page_size_bytes) {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap {} has invalid page geometry: mapped_len={} page_size={page_size_bytes}",
                self.path.display(),
                self.mapped_len
            )));
        }
        self.residency_snapshot_range(0, self.mapped_len)
    }

    /// Snapshot one page-aligned subrange without reading it. This is used to
    /// distinguish tied-head residency from unrelated pages in the same GGUF.
    pub(crate) fn residency_snapshot_range(
        &self,
        aligned_offset: usize,
        mapped_bytes: usize,
    ) -> Result<WireMmapResidencySnapshot> {
        let page_size_bytes = page_size();
        if page_size_bytes == 0
            || mapped_bytes == 0
            || !aligned_offset.is_multiple_of(page_size_bytes)
            || !mapped_bytes.is_multiple_of(page_size_bytes)
            || aligned_offset
                .checked_add(mapped_bytes)
                .is_none_or(|end| end > self.mapped_len)
        {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap {} has invalid residency range: offset={aligned_offset} len={mapped_bytes} mapped_len={} page_size={page_size_bytes}",
                self.path.display(),
                self.mapped_len
            )));
        }
        let total_pages = mapped_bytes / page_size_bytes;
        let mut status = vec![0u8; total_pages];
        // SAFETY: the queried range is this live mapping and `status` contains
        // exactly one output byte per mapped page. mincore does not fault pages.
        let result = unsafe {
            libc::mincore(
                self.ptr
                    .add(aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                mapped_bytes,
                status.as_mut_ptr().cast::<libc::c_char>(),
            )
        };
        if result != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "mincore failed for wire mmap {}: {}",
                self.path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let resident_pages = status.iter().filter(|entry| **entry & 1 != 0).count();
        Ok(WireMmapResidencySnapshot {
            page_size_bytes,
            mapped_bytes,
            total_pages,
            resident_pages,
            resident_bytes: resident_pages * page_size_bytes,
        })
    }

    /// Synchronously make every page in one aligned file-backed window
    /// required by issuing one volatile read per page. No readahead advisory or
    /// anonymous copy is involved.
    pub(crate) fn fault_pages_range(
        &self,
        aligned_offset: usize,
        mapped_bytes: usize,
    ) -> Result<()> {
        // Validate the page geometry through the non-faulting query first.
        let snapshot = self.residency_snapshot_range(aligned_offset, mapped_bytes)?;
        // Keep the explicit page-stride walk bounded to this window instead of
        // asking the kernel's sequential-fault heuristic to read ahead into an
        // adjacent tensor.
        let advise = unsafe {
            libc::madvise(
                self.ptr
                    .add(aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                mapped_bytes,
                libc::MADV_RANDOM,
            )
        };
        if advise != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "MADV_RANDOM failed for wire mmap {} range {aligned_offset}..{}: {}",
                self.path.display(),
                aligned_offset + mapped_bytes,
                std::io::Error::last_os_error()
            )));
        }
        let mut observed = 0u8;
        for relative in (0..mapped_bytes).step_by(snapshot.page_size_bytes) {
            // SAFETY: range validation above proved every page-stride address
            // lies in this live immutable mapping. Volatile prevents elision.
            observed ^= unsafe { std::ptr::read_volatile(self.ptr.add(aligned_offset + relative)) };
        }
        std::hint::black_box(observed);
        Ok(())
    }

    /// Advise that one exact immutable file range is no longer needed, rounded
    /// outwards to page boundaries. `MADV_DONTNEED` never changes file contents;
    /// a later CPU or GPU access simply faults the clean page back from disk.
    /// The call is advisory, so before/after `mincore` facts are returned and a
    /// higher-level load gate can fail closed if the kernel retained the pages.
    pub(crate) fn advise_dontneed_range(
        &self,
        offset: u64,
        len: usize,
    ) -> Result<WireMmapDiscardSnapshot> {
        let range = aligned_page_range(offset, len, self.mapped_len, page_size())?;
        self.advise_dontneed_aligned_range(range)
    }

    /// Aligned sibling used by a merged final cleanup pass.
    pub(crate) fn advise_dontneed_aligned_range(
        &self,
        range: WireMmapAlignedRange,
    ) -> Result<WireMmapDiscardSnapshot> {
        let before = self.residency_snapshot_range(range.aligned_offset, range.mapped_bytes)?;
        // SAFETY: `residency_snapshot_range` validated that this page-aligned
        // range lies wholly inside the live immutable mapping.
        let advise = unsafe {
            libc::madvise(
                self.ptr
                    .add(range.aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                range.mapped_bytes,
                libc::MADV_DONTNEED,
            )
        };
        if advise != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "MADV_DONTNEED failed for wire mmap {} range {}..{}: {}",
                self.path.display(),
                range.aligned_offset,
                range.aligned_offset + range.mapped_bytes,
                std::io::Error::last_os_error()
            )));
        }
        let after = self.residency_snapshot_range(range.aligned_offset, range.mapped_bytes)?;
        Ok(WireMmapDiscardSnapshot {
            range,
            resident_before_bytes: before.resident_bytes,
            resident_after_bytes: after.resident_bytes,
            confirmed_discarded_bytes: before.resident_bytes.saturating_sub(after.resident_bytes),
        })
    }

    /// Invalidate cached data for one aligned range of this immutable,
    /// read-only `MAP_SHARED` mapping while leaving the mapping live. Darwin's
    /// `MS_INVALIDATE` is stronger than the advisory `MADV_DONTNEED`; callers
    /// still receive before/after `mincore` facts so they can fail closed if
    /// the kernel retained any clean source pages.
    #[cfg(target_os = "macos")]
    pub(crate) fn invalidate_cached_aligned_range(
        &self,
        range: WireMmapAlignedRange,
    ) -> Result<WireMmapDiscardSnapshot> {
        let before = self.residency_snapshot_range(range.aligned_offset, range.mapped_bytes)?;
        // SAFETY: `residency_snapshot_range` validated that this page-aligned
        // range lies wholly inside the live immutable MAP_SHARED mapping.
        let invalidate = unsafe {
            libc::msync(
                self.ptr
                    .add(range.aligned_offset)
                    .cast_mut()
                    .cast::<libc::c_void>(),
                range.mapped_bytes,
                libc::MS_INVALIDATE,
            )
        };
        if invalidate != 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "MS_INVALIDATE failed for wire mmap {} range {}..{}: {}",
                self.path.display(),
                range.aligned_offset,
                range.aligned_offset + range.mapped_bytes,
                std::io::Error::last_os_error()
            )));
        }
        let after = self.residency_snapshot_range(range.aligned_offset, range.mapped_bytes)?;
        Ok(WireMmapDiscardSnapshot {
            range,
            resident_before_bytes: before.resident_bytes,
            resident_after_bytes: after.resident_bytes,
            confirmed_discarded_bytes: before.resident_bytes.saturating_sub(after.resident_bytes),
        })
    }

    /// Hint the kernel to read the file ahead sequentially (weight order is
    /// roughly file order, so the first forward pass streams predictably).
    pub fn advise_sequential(&self) {
        // SAFETY: the range is exactly this mapping.
        unsafe {
            libc::madvise(
                self.ptr as *mut libc::c_void,
                self.mapped_len,
                libc::MADV_SEQUENTIAL,
            );
        }
    }

    /// Kick off asynchronous population of the whole mapping (warm the page
    /// cache without blocking).
    pub fn advise_willneed(&self) -> bool {
        // SAFETY: the range is exactly this mapping.
        unsafe {
            libc::madvise(
                self.ptr as *mut libc::c_void,
                self.mapped_len,
                libc::MADV_WILLNEED,
            ) == 0
        }
    }

    /// Warm only `[offset, offset + len)` of the mapping.
    ///
    /// Prefer this over [`Self::advise_willneed`] when the caller owns a slice
    /// of the file: readahead is bounded by device bandwidth, so advising the
    /// whole mapping makes the kernel stream bytes the caller does not need
    /// before the ones it does. A gemma4 tail shard is the motivating case —
    /// the GGUF's data section opens with a 2.5GB `per_layer_token_embd` table
    /// that a layer-range worker never reads, and pulling it first both delays
    /// the shard's own pages and evicts them again under memory pressure.
    ///
    /// The range is clamped to the mapping and the start is rounded DOWN to a
    /// page boundary (`madvise` requires page-aligned addresses); a zero-length
    /// or out-of-bounds request is a no-op.
    pub fn advise_willneed_range(&self, offset: usize, len: usize) -> bool {
        if len == 0 || offset >= self.mapped_len {
            return false;
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page = if page > 0 { page as usize } else { 4096 };
        let start = offset - (offset % page);
        // Clamp against the mapping, not the requested end: `offset + len` can
        // overflow past `mapped_len` for a caller-computed tensor extent.
        let end = offset.saturating_add(len).min(self.mapped_len);
        let Some(span) = end.checked_sub(start).filter(|s| *s > 0) else {
            return false;
        };
        // SAFETY: `start` is page-aligned and `start + span <= mapped_len`, so
        // the range lies entirely within this mapping.
        unsafe {
            libc::madvise(
                self.ptr.add(start) as *mut libc::c_void,
                span,
                libc::MADV_WILLNEED,
            ) == 0
        }
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Base address of the mapping (page-aligned).
    pub fn base_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Mapped length (file length rounded up to a page multiple).
    pub fn mapped_len(&self) -> usize {
        self.mapped_len
    }

    /// Borrow file bytes at `offset..offset+len`.
    pub fn bytes(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let end = offset.checked_add(len as u64).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "wire mmap range overflow at offset {offset} len {len} in {}",
                self.path.display()
            ))
        })?;
        if end > self.file_len {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap range {offset}..{end} exceeds file length {} in {}",
                self.file_len,
                self.path.display()
            )));
        }
        // SAFETY: bounds-checked against file_len above; mapping is immutable.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset as usize), len) })
    }
}

/// A read-only mapping of an entire GGUF file, backed by `memmap2`
/// (`CreateFileMapping`/`MapViewOfFile`). `memmap2::Mmap` is already `Send +
/// Sync`, so this type is too without an explicit `unsafe impl`.
#[cfg(windows)]
#[derive(Debug)]
pub struct GgufWireMmap {
    mmap: memmap2::Mmap,
    file_len: u64,
    path: PathBuf,
}

#[cfg(windows)]
impl GgufWireMmap {
    /// Map `path` read-only. The mapping covers the whole file.
    pub fn map(path: &Path) -> Result<Arc<Self>> {
        let file = File::open(path).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "wire mmap open failed for {}: {err}",
                path.display()
            ))
        })?;
        Self::map_file(&file, path)
    }

    /// Windows counterpart of the descriptor-preserving mapper used by
    /// `GhostFile`; keeping one API avoids reopening a replaceable path.
    pub(crate) fn map_file(file: &File, path: &Path) -> Result<Arc<Self>> {
        let file_len = file
            .metadata()
            .map_err(|err| {
                BackendError::InvalidTensorData(format!(
                    "wire mmap metadata failed for {}: {err}",
                    path.display()
                ))
            })?
            .len();
        if file_len == 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap refused for empty file {}",
                path.display()
            )));
        }
        // SAFETY: the file is opened read-only and the mapping is treated as
        // immutable for its whole lifetime; no other handle here writes to it.
        let mmap = unsafe { memmap2::Mmap::map(file) }.map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "wire mmap failed for {}: {err}",
                path.display()
            ))
        })?;
        Ok(Arc::new(Self {
            mmap,
            file_len,
            path: path.to_path_buf(),
        }))
    }

    /// Sequential-access hint. `memmap2` exposes no portable advise on Windows;
    /// the OS prefetcher handles read-ahead, so this is a no-op.
    pub fn advise_sequential(&self) {}

    /// Population hint; a no-op on Windows (see `advise_sequential`).
    pub fn advise_willneed(&self) -> bool {
        false
    }

    /// Ranged population hint; a no-op on Windows (see `advise_sequential`).
    pub fn advise_willneed_range(&self, _offset: usize, _len: usize) -> bool {
        false
    }

    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Base address of the mapping.
    pub fn base_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Mapped length. `memmap2` maps exactly the file length.
    pub fn mapped_len(&self) -> usize {
        self.mmap.len()
    }

    /// Borrow file bytes at `offset..offset+len`.
    pub fn bytes(&self, offset: u64, len: usize) -> Result<&[u8]> {
        let end = offset.checked_add(len as u64).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "wire mmap range overflow at offset {offset} len {len} in {}",
                self.path.display()
            ))
        })?;
        if end > self.file_len {
            return Err(BackendError::InvalidTensorData(format!(
                "wire mmap range {offset}..{end} exceeds file length {} in {}",
                self.file_len,
                self.path.display()
            )));
        }
        Ok(&self.mmap[offset as usize..offset as usize + len])
    }
}

/// A page-aligned, heap-owned copy of one tensor's wire-format bytes, suitable
/// for an offset-0 `newBufferWithBytesNoCopy` Metal buffer: the GPU reads this
/// allocation in place, so it is the ONLY resident copy of the weight (no
/// 36-byte CPU decode, no GPU upload copy). Filled by one sequential read of
/// the tensor's file range with the page cache enabled, so reloading a model
/// runs at page-cache speed instead of re-streaming the disk.
#[derive(Debug)]
pub struct WirePages {
    ptr: *mut u8,
    /// Allocation length: `byte_len` rounded up to a page multiple
    /// (`newBufferWithBytesNoCopy` requires a page-multiple length).
    alloc_len: usize,
    /// Exact wire byte length of the tensor (rows * blocks_per_row * 34 for Q8_0).
    byte_len: usize,
}

// SAFETY: the allocation is written once during construction and immutable
// afterwards; concurrent reads are safe.
unsafe impl Send for WirePages {}
unsafe impl Sync for WirePages {}

impl Drop for WirePages {
    fn drop(&mut self) {
        // SAFETY: ptr/alloc_len describe the live allocation created in `read_from_file`.
        unsafe {
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align_unchecked(self.alloc_len, page_size()),
            );
        }
    }
}

impl WirePages {
    /// Allocate page-aligned storage and fill it with `byte_len` bytes read from
    /// `file` at `offset` (one sequential read, page cache enabled).
    pub fn read_from_file(file: &File, offset: u64, byte_len: usize) -> Result<Arc<Self>> {
        if byte_len == 0 {
            return Err(BackendError::InvalidTensorData(
                "wire pages refused for an empty tensor range".to_string(),
            ));
        }
        let page = page_size();
        let alloc_len = byte_len.div_ceil(page) * page;
        let layout = std::alloc::Layout::from_size_align(alloc_len, page).map_err(|err| {
            BackendError::InvalidTensorData(format!("wire pages layout error: {err}"))
        })?;
        // SAFETY: layout is non-zero and valid.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(BackendError::InvalidTensorData(format!(
                "wire pages allocation of {alloc_len} bytes failed"
            )));
        }
        let pages = Self {
            ptr,
            alloc_len,
            byte_len,
        };
        // SAFETY: the allocation is alloc_len >= byte_len bytes and exclusively owned here.
        let fill = unsafe { std::slice::from_raw_parts_mut(ptr, byte_len) };
        read_exact_at(file, fill, offset).map_err(|err| {
            BackendError::InvalidTensorData(format!(
                "wire pages read of {byte_len} bytes at offset {offset} failed: {err}"
            ))
        })?;
        // Zero the page-rounding tail so NoCopy buffer contents are deterministic.
        // SAFETY: byte_len..alloc_len is within the allocation.
        unsafe {
            std::ptr::write_bytes(ptr.add(byte_len), 0, alloc_len - byte_len);
        }
        Ok(Arc::new(pages))
    }

    /// The tensor's wire bytes (exact length, excluding the page-rounding tail).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: immutable after construction.
        unsafe { std::slice::from_raw_parts(self.ptr, self.byte_len) }
    }

    /// Page-aligned base pointer.
    pub fn base_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Page-multiple allocation length for the NoCopy buffer.
    pub fn alloc_len(&self) -> usize {
        self.alloc_len
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

impl PartialEq for WirePages {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.ptr, other.ptr)
    }
}

/// One tensor's data range inside a mapped GGUF file, in wire layout, plus the
/// page-aligned window a Metal NoCopy buffer wraps to reach it. Tensors that
/// share a window share the buffer.
///
/// Equality is identity of the mapped range (same mapping, same offsets) — the
/// mapping is immutable, so identical ranges are identical bytes.
#[derive(Debug, Clone)]
pub struct WireMmapTensor {
    pub mmap: Arc<GgufWireMmap>,
    /// Absolute byte offset of the tensor's data in the file.
    pub absolute_offset: u64,
    /// Tensor data length in bytes (rows * blocks_per_row * 34 for Q8_0).
    pub byte_len: usize,
    /// The page-aligned window containing this tensor's bytes.
    pub window: WireWindow,
    /// Byte offset of the tensor's data within its window.
    pub window_offset: usize,
}

impl PartialEq for WireMmapTensor {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.mmap, &other.mmap)
            && self.absolute_offset == other.absolute_offset
            && self.byte_len == other.byte_len
    }
}

impl WireMmapTensor {
    pub fn bytes(&self) -> Result<&[u8]> {
        self.mmap.bytes(self.absolute_offset, self.byte_len)
    }
}

/// Build a [`WireMmapTensor`] per input range, sharing windows planned by
/// [`plan_wire_windows`]. `ranges` are (absolute_offset, byte_len) pairs in any
/// order; results match the input order.
pub fn wire_mmap_tensors(
    mapping: &Arc<GgufWireMmap>,
    ranges: &[(u64, usize)],
    max_window_len: usize,
) -> Result<Vec<WireMmapTensor>> {
    let plan = plan_wire_windows(mapping, ranges, max_window_len)?;
    Ok(ranges
        .iter()
        .zip(plan.placements)
        .map(
            |(&(absolute_offset, byte_len), (window_index, window_offset))| WireMmapTensor {
                mmap: Arc::clone(mapping),
                absolute_offset,
                byte_len,
                window: plan.windows[window_index],
                window_offset,
            },
        )
        .collect())
}

/// A page-aligned window over the mapping, sized for one Metal buffer
/// (`newBufferWithBytesNoCopy` requires a page-aligned pointer and a
/// page-multiple length). Tensors reference a window plus a byte offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireWindow {
    /// Page-aligned offset of the window start within the file mapping.
    pub aligned_offset: u64,
    /// Page-multiple window length (clamped to the mapped length).
    pub len: usize,
}

/// Result of [`plan_wire_windows`]: the page-aligned windows plus, per input
/// range, its `(window index, byte offset within the window)`.
#[derive(Debug)]
pub struct WireWindowPlan {
    pub windows: Vec<WireWindow>,
    pub placements: Vec<(usize, usize)>,
}

/// Plan page-aligned windows covering `ranges` (absolute_offset, byte_len),
/// packing greedily so each window stays within `max_window_len` and no range
/// straddles a window boundary.
pub fn plan_wire_windows(
    mapping: &GgufWireMmap,
    ranges: &[(u64, usize)],
    max_window_len: usize,
) -> Result<WireWindowPlan> {
    let page = page_size() as u64;
    let mut sorted: Vec<(usize, u64, usize)> = ranges
        .iter()
        .enumerate()
        .map(|(idx, &(offset, len))| (idx, offset, len))
        .collect();
    sorted.sort_by_key(|&(_, offset, _)| offset);

    let mut windows: Vec<WireWindow> = Vec::new();
    let mut placements = vec![(usize::MAX, usize::MAX); ranges.len()];
    let mut current_start: Option<u64> = None;
    let mut current_end: u64 = 0;
    let mut pending: Vec<(usize, u64)> = Vec::new();

    let flush = |windows: &mut Vec<WireWindow>,
                 placements: &mut Vec<(usize, usize)>,
                 start: u64,
                 end: u64,
                 pending: &mut Vec<(usize, u64)>| {
        let aligned = start / page * page;
        let len = ((end - aligned) as usize).div_ceil(page as usize) * (page as usize);
        let len = len.min(mapping.mapped_len() - aligned as usize);
        let window_index = windows.len();
        windows.push(WireWindow {
            aligned_offset: aligned,
            len,
        });
        for (range_index, offset) in pending.drain(..) {
            placements[range_index] = (window_index, (offset - aligned) as usize);
        }
    };

    for (range_index, offset, len) in sorted {
        let end = offset.checked_add(len as u64).ok_or_else(|| {
            BackendError::InvalidTensorData(format!(
                "wire window range overflow at offset {offset} len {len}"
            ))
        })?;
        if end > mapping.file_len() {
            return Err(BackendError::InvalidTensorData(format!(
                "wire window range {offset}..{end} exceeds file length {}",
                mapping.file_len()
            )));
        }
        if len > max_window_len {
            return Err(BackendError::InvalidTensorData(format!(
                "wire window range of {len} bytes exceeds the max window length {max_window_len}"
            )));
        }
        match current_start {
            Some(start) => {
                let aligned = start / page * page;
                let prospective = (end - aligned) as usize;
                if prospective.div_ceil(page as usize) * (page as usize) > max_window_len {
                    flush(
                        &mut windows,
                        &mut placements,
                        start,
                        current_end,
                        &mut pending,
                    );
                    current_start = Some(offset);
                    current_end = end;
                } else {
                    current_end = current_end.max(end);
                }
            }
            None => {
                current_start = Some(offset);
                current_end = end;
            }
        }
        pending.push((range_index, offset));
    }
    if let Some(start) = current_start {
        flush(
            &mut windows,
            &mut placements,
            start,
            current_end,
            &mut pending,
        );
    }
    debug_assert!(placements.iter().all(|&(w, _)| w != usize::MAX));
    Ok(WireWindowPlan {
        windows,
        placements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "camelid-wire-mmap-test-{}-{}",
            std::process::id(),
            bytes.len()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn wire_pages_are_page_aligned_and_match_file_bytes() {
        let payload: Vec<u8> = (0..50_000usize).map(|i| (i % 199) as u8).collect();
        let path = write_temp(&payload);
        let file = File::open(&path).unwrap();
        let pages = WirePages::read_from_file(&file, 1234, 40_000).unwrap();
        assert_eq!(pages.base_ptr() as usize % page_size(), 0);
        assert_eq!(pages.alloc_len() % page_size(), 0);
        assert_eq!(pages.byte_len(), 40_000);
        assert_eq!(pages.bytes(), &payload[1234..1234 + 40_000]);
        // Page-rounding tail is zeroed.
        let tail = unsafe {
            std::slice::from_raw_parts(
                pages.base_ptr().add(pages.byte_len()),
                pages.alloc_len() - pages.byte_len(),
            )
        };
        assert!(tail.iter().all(|&b| b == 0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn maps_and_reads_exact_file_bytes() {
        let payload: Vec<u8> = (0..70_000usize).map(|i| (i % 251) as u8).collect();
        let path = write_temp(&payload);
        let mapping = GgufWireMmap::map(&path).unwrap();
        assert_eq!(mapping.file_len(), payload.len() as u64);
        assert_eq!(mapping.bytes(0, payload.len()).unwrap(), &payload[..]);
        assert_eq!(
            mapping.bytes(65_521, 100).unwrap(),
            &payload[65_521..65_621]
        );
        assert!(mapping.bytes(payload.len() as u64 - 10, 11).is_err());
        assert_eq!(mapping.base_ptr() as usize % page_size(), 0);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(unix)]
    #[test]
    fn map_file_uses_the_supplied_vnode_when_path_is_replaced() {
        let original = vec![0x3cu8; 32_768];
        let replacement = vec![0xa5u8; 32_768];
        let path = write_temp(&original);
        let retained = File::open(&path).unwrap();
        let moved = path.with_extension("retained-vnode");
        std::fs::rename(&path, &moved).unwrap();
        std::fs::write(&path, &replacement).unwrap();

        let mapping = GgufWireMmap::map_file(&retained, &path).unwrap();
        assert_eq!(mapping.bytes(0, original.len()).unwrap(), &original);

        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&moved).ok();
    }

    #[test]
    fn aligned_page_range_rounds_outward_without_crossing_mapping() {
        let page = 16_384;
        let mapped_len = page * 10;
        assert_eq!(
            aligned_page_range((page + 7136) as u64, page * 2, mapped_len, page).unwrap(),
            WireMmapAlignedRange {
                aligned_offset: page,
                mapped_bytes: page * 3,
            }
        );
        assert_eq!(
            aligned_page_range((page * 9 + 7) as u64, page - 7, mapped_len, page).unwrap(),
            WireMmapAlignedRange {
                aligned_offset: page * 9,
                mapped_bytes: page,
            }
        );
        assert!(aligned_page_range(0, 0, mapped_len, page).is_err());
        assert!(aligned_page_range((mapped_len - 1) as u64, 2, mapped_len, page).is_err());
        assert!(aligned_page_range(0, page, mapped_len - 1, page).is_err());
        assert!(aligned_page_range(u64::MAX, usize::MAX, mapped_len, page).is_err());
    }

    #[test]
    fn merged_page_ranges_coalesce_overlap_and_shared_boundaries() {
        let page = 16_384;
        let mapped_len = page * 20;
        let ranges = [
            ((page + 100) as u64, page),
            ((page * 2 + 50) as u64, page),
            ((page * 7) as u64, page),
            ((page * 8) as u64, page),
            ((page * 12 + 1) as u64, 32),
        ];
        assert_eq!(
            merge_aligned_page_ranges(&ranges, mapped_len, page).unwrap(),
            vec![
                WireMmapAlignedRange {
                    aligned_offset: page,
                    mapped_bytes: page * 3,
                },
                WireMmapAlignedRange {
                    aligned_offset: page * 7,
                    mapped_bytes: page * 2,
                },
                WireMmapAlignedRange {
                    aligned_offset: page * 12,
                    mapped_bytes: page,
                },
            ]
        );
        assert!(merge_aligned_page_ranges(&[(u64::MAX, 8)], mapped_len, page).is_err());
    }

    #[test]
    fn window_plan_covers_ranges_without_straddles() {
        let page = page_size();
        let payload = vec![7u8; page * 12 + 123];
        let path = write_temp(&payload);
        let mapping = GgufWireMmap::map(&path).unwrap();

        // Three tensors: two adjacent early, one far later; max window = 4 pages.
        let ranges = vec![
            (100u64, page),            // tensor 0
            (100 + page as u64, 500),  // tensor 1, adjacent
            ((page * 9) as u64, 2000), // tensor 2, far away
        ];
        let plan = plan_wire_windows(&mapping, &ranges, page * 4).unwrap();
        let (windows, placements) = (plan.windows, plan.placements);
        assert_eq!(windows.len(), 2);
        for (range_index, &(offset, len)) in ranges.iter().enumerate() {
            let (window_index, in_window) = placements[range_index];
            let window = windows[window_index];
            assert_eq!(window.aligned_offset % page as u64, 0);
            assert_eq!(window.len % page, 0);
            assert_eq!(window.aligned_offset + in_window as u64, offset);
            assert!(in_window + len <= window.len, "range fits its window");
            // Window bytes at the placement match the file bytes directly.
            let via_window = mapping
                .bytes(window.aligned_offset + in_window as u64, len)
                .unwrap();
            let direct = mapping.bytes(offset, len).unwrap();
            assert_eq!(via_window, direct);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn window_plan_splits_when_exceeding_max_window() {
        let page = page_size();
        let payload = vec![3u8; page * 32];
        let path = write_temp(&payload);
        let mapping = GgufWireMmap::map(&path).unwrap();
        // Eight 2-page tensors back to back; max window 4 pages -> 4+ windows.
        let ranges: Vec<(u64, usize)> = (0..8).map(|i| ((i * 2 * page) as u64, 2 * page)).collect();
        let plan = plan_wire_windows(&mapping, &ranges, page * 4).unwrap();
        let (windows, placements) = (plan.windows, plan.placements);
        assert!(windows.len() >= 4);
        for window in &windows {
            assert!(window.len <= page * 4);
        }
        for (range_index, &(offset, len)) in ranges.iter().enumerate() {
            let (window_index, in_window) = placements[range_index];
            assert_eq!(
                windows[window_index].aligned_offset + in_window as u64,
                offset
            );
            assert!(in_window + len <= windows[window_index].len);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_range_larger_than_max_window() {
        let page = page_size();
        let payload = vec![1u8; page * 8];
        let path = write_temp(&payload);
        let mapping = GgufWireMmap::map(&path).unwrap();
        let err = plan_wire_windows(&mapping, &[(0, page * 6)], page * 4);
        assert!(err.is_err());
        std::fs::remove_file(&path).ok();
    }
}
