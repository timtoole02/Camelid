//! Exact Gemma 4 12B QAT Q4_0 native-octet sidecar.
//!
//! The source GGUF remains immutable.  This module writes and validates a
//! same-payload-size projection pack whose eight-row tiles contain a scale
//! plane followed by a quant plane.  The Metal verifier can therefore issue
//! coalesced Q4 reads without retaining a second copy of the original Q4 wire
//! projections.  The path is deliberately pinned to one target SHA and is
//! opt-in at runtime; any identity, layout, index, size, or payload-integrity
//! mismatch is a hard refusal rather than a fallback to potentially misbound
//! bytes.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    gguf::{GgufFile, GgufTensorDescriptor, GgufTensorType},
    BackendError, Result,
};

pub const GEMMA4_12B_QAT_Q4_0_TARGET_SHA256: &str =
    "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b";
pub const GEMMA4_Q4_NATIVE_SIDECAR_ENV: &str = "CAMELID_GEMMA4_Q4_NATIVE_SIDECAR";

const SCHEMA: &str = "camelid.gemma4_q4_native_octets.v1";
const LAYOUT: &str = "octet_scale_plane_then_quant_plane.v1";
const MAGIC: &[u8; 16] = b"CAMELIDQ4OCTET1!";
const VERSION: u32 = 1;
const PREAMBLE_LEN: usize = 96;
const PAYLOAD_OFFSET: u64 = 1 << 20;
const WIRE_BLOCK_BYTES: usize = 18;
const NATIVE_OCTET_BLOCK_BYTES: usize = 144;
const PINNED_Q4_TENSOR_COUNT: usize = 328;
const PINNED_Q4_PAYLOAD_BYTES: u64 = 6_131_220_480;
const PINNED_Q4_SOURCE_START: u64 = 841_593_472;
const PINNED_Q4_SOURCE_END: u64 = 6_975_848_544;

fn invalid(detail: impl Into<String>) -> BackendError {
    BackendError::InvalidTensorData(format!("Gemma 4 native Q4 sidecar: {}", detail.into()))
}

fn io(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(target_os = "macos")]
fn install_no_replace(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};

    let temporary = CString::new(temporary.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let destination = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    // SAFETY: both C strings remain live for the call. RENAME_EXCL makes the
    // install atomic and refuses an existing destination instead of replacing it.
    let status =
        unsafe { libc::renamex_np(temporary.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
    if status == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "macos"))]
fn install_no_replace(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    // A same-directory hard link is an atomic create-if-absent on supported
    // filesystems. Refuse rather than falling back to an overwriting rename.
    std::fs::hard_link(temporary, destination)?;
    std::fs::remove_file(temporary)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gemma4Q4NativeEntry {
    pub name: String,
    pub source_offset: u64,
    pub source_len: u64,
    pub rows: u64,
    pub columns: u64,
    pub blocks_per_row: u64,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gemma4Q4NativeManifest {
    pub schema: String,
    pub layout: String,
    pub source_sha256: String,
    pub source_file_len: u64,
    pub source_q4_bytes: u64,
    pub payload_offset: u64,
    pub payload_len: u64,
    pub entries: Vec<Gemma4Q4NativeEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Q4Spec {
    name: String,
    source_offset: u64,
    source_len: u64,
    rows: usize,
    columns: usize,
    blocks_per_row: usize,
}

fn exact_q4_specs(gguf: &GgufFile, source_file_len: u64) -> Result<Vec<Q4Spec>> {
    if gguf.architecture() != Some("gemma4") {
        return Err(invalid(format!(
            "source architecture is {:?}, expected gemma4",
            gguf.architecture()
        )));
    }
    let mut specs = Vec::new();
    let mut names = BTreeSet::new();
    for descriptor in gguf
        .tensors
        .iter()
        .filter(|tensor| tensor.tensor_type == GgufTensorType::Q4_0)
    {
        if descriptor.dimensions.len() != 2 {
            return Err(invalid(format!(
                "Q4_0 tensor {} is rank {}, expected a dense rank-2 projection",
                descriptor.name,
                descriptor.dimensions.len()
            )));
        }
        if !names.insert(descriptor.name.clone()) {
            return Err(invalid(format!(
                "duplicate Q4_0 tensor name {}",
                descriptor.name
            )));
        }
        let columns = usize::try_from(descriptor.dimensions[0])
            .map_err(|_| invalid(format!("{} columns do not fit usize", descriptor.name)))?;
        let rows = usize::try_from(descriptor.dimensions[1])
            .map_err(|_| invalid(format!("{} rows do not fit usize", descriptor.name)))?;
        if columns == 0 || rows == 0 || !columns.is_multiple_of(32) {
            return Err(invalid(format!(
                "{} has invalid Q4_0 geometry rows={rows} columns={columns}",
                descriptor.name
            )));
        }
        // The packer supports a ragged final octet, but the pinned production
        // artifact must remain payload-neutral.  Refuse a source that would
        // need padding instead of silently growing the resident model.
        if !rows.is_multiple_of(8) {
            return Err(invalid(format!(
                "{} has {rows} rows; the exact target requires octet-aligned rows for zero payload growth",
                descriptor.name
            )));
        }
        let blocks_per_row = columns / 32;
        let expected = rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(WIRE_BLOCK_BYTES))
            .ok_or_else(|| invalid(format!("{} wire byte count overflow", descriptor.name)))?;
        if descriptor.n_bytes != expected as u64 {
            return Err(invalid(format!(
                "{} source size {} does not match rows={rows} bpr={blocks_per_row} Q4_0 size {expected}",
                descriptor.name, descriptor.n_bytes
            )));
        }
        let source_end = descriptor
            .absolute_offset
            .checked_add(descriptor.n_bytes)
            .ok_or_else(|| invalid(format!("{} source range overflow", descriptor.name)))?;
        if source_end > source_file_len {
            return Err(invalid(format!(
                "{} source range {}..{source_end} exceeds file length {source_file_len}",
                descriptor.name, descriptor.absolute_offset
            )));
        }
        specs.push(Q4Spec {
            name: descriptor.name.clone(),
            source_offset: descriptor.absolute_offset,
            source_len: descriptor.n_bytes,
            rows,
            columns,
            blocks_per_row,
        });
    }
    specs.sort_by_key(|spec| spec.source_offset);
    if specs.len() != PINNED_Q4_TENSOR_COUNT {
        return Err(invalid(format!(
            "source has {} Q4_0 projections, expected {PINNED_Q4_TENSOR_COUNT}",
            specs.len()
        )));
    }
    let total = specs.iter().try_fold(0u64, |sum, spec| {
        sum.checked_add(spec.source_len)
            .ok_or_else(|| invalid("source Q4_0 byte total overflow"))
    })?;
    if total != PINNED_Q4_PAYLOAD_BYTES {
        return Err(invalid(format!(
            "source Q4_0 payload is {total} bytes, expected {PINNED_Q4_PAYLOAD_BYTES}"
        )));
    }
    let first = specs
        .first()
        .ok_or_else(|| invalid("source has no Q4_0 projections"))?;
    let last = specs
        .last()
        .ok_or_else(|| invalid("source has no Q4_0 projections"))?;
    let last_end = last
        .source_offset
        .checked_add(last.source_len)
        .ok_or_else(|| invalid("last source Q4_0 range overflow"))?;
    if first.source_offset != PINNED_Q4_SOURCE_START || last_end != PINNED_Q4_SOURCE_END {
        return Err(invalid(format!(
            "source Q4_0 envelope is {}..{last_end}, expected {PINNED_Q4_SOURCE_START}..{PINNED_Q4_SOURCE_END}",
            first.source_offset
        )));
    }
    for pair in specs.windows(2) {
        let left_end = pair[0]
            .source_offset
            .checked_add(pair[0].source_len)
            .ok_or_else(|| invalid("source Q4_0 range overflow"))?;
        if left_end > pair[1].source_offset {
            return Err(invalid(format!(
                "source Q4_0 ranges overlap: {} ends at {left_end}, {} starts at {}",
                pair[0].name, pair[1].name, pair[1].source_offset
            )));
        }
    }
    Ok(specs)
}

/// Repack row-major 18-byte Q4_0 blocks into eight-row native tiles.
///
/// Each tile is `[block][row] f16` followed by
/// `[block][quarter][row] uchar4`.  Missing rows in a ragged final tile are
/// exact zeros: scale bits `0x0000` and packed nibbles `0x88`.
pub(crate) fn pack_native_q4_octets(
    wire: &[u8],
    rows: usize,
    blocks_per_row: usize,
) -> Result<Vec<u8>> {
    if rows == 0 || blocks_per_row == 0 {
        return Err(invalid("native Q4 pack refused empty geometry"));
    }
    let expected_wire = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(WIRE_BLOCK_BYTES))
        .ok_or_else(|| invalid("native Q4 source size overflow"))?;
    if wire.len() != expected_wire {
        return Err(invalid(format!(
            "native Q4 pack received {} bytes, expected {expected_wire}",
            wire.len()
        )));
    }
    let native_len = rows
        .div_ceil(8)
        .checked_mul(blocks_per_row)
        .and_then(|records| records.checked_mul(NATIVE_OCTET_BLOCK_BYTES))
        .ok_or_else(|| invalid("native Q4 destination size overflow"))?;
    let mut native = vec![0u8; native_len];
    for octet in 0..rows.div_ceil(8) {
        let tile_base = octet * blocks_per_row * NATIVE_OCTET_BLOCK_BYTES;
        let quant_base = tile_base + blocks_per_row * 16;
        native[quant_base..tile_base + blocks_per_row * NATIVE_OCTET_BLOCK_BYTES].fill(0x88);
        for block in 0..blocks_per_row {
            for tile_row in 0..8 {
                let row = octet * 8 + tile_row;
                if row >= rows {
                    continue;
                }
                let source = (row * blocks_per_row + block) * WIRE_BLOCK_BYTES;
                let scale_destination = tile_base + (block * 8 + tile_row) * 2;
                native[scale_destination..scale_destination + 2]
                    .copy_from_slice(&wire[source..source + 2]);
                for quarter in 0..4 {
                    let destination = quant_base + block * 128 + quarter * 32 + tile_row * 4;
                    let quant_source = source + 2 + quarter * 4;
                    native[destination..destination + 4]
                        .copy_from_slice(&wire[quant_source..quant_source + 4]);
                }
            }
        }
    }
    Ok(native)
}

fn encode_preamble(header_len: usize, header_sha256: &[u8; 32], payload_len: u64) -> [u8; 96] {
    let mut bytes = [0u8; PREAMBLE_LEN];
    bytes[..16].copy_from_slice(MAGIC);
    bytes[16..20].copy_from_slice(&VERSION.to_le_bytes());
    bytes[20..24].copy_from_slice(&(header_len as u32).to_le_bytes());
    bytes[24..32].copy_from_slice(&PAYLOAD_OFFSET.to_le_bytes());
    bytes[32..40].copy_from_slice(&payload_len.to_le_bytes());
    bytes[40..72].copy_from_slice(header_sha256);
    bytes
}

fn decode_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four-byte preamble field"))
}

fn decode_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight-byte preamble field"))
}

fn decode_preamble(bytes: &[u8; PREAMBLE_LEN]) -> Result<(usize, u64, [u8; 32])> {
    if &bytes[..16] != MAGIC {
        return Err(invalid("bad sidecar magic"));
    }
    let version = decode_u32(&bytes[16..20]);
    if version != VERSION {
        return Err(invalid(format!(
            "sidecar version {version} is not supported (expected {VERSION})"
        )));
    }
    let header_len = decode_u32(&bytes[20..24]) as usize;
    let payload_offset = decode_u64(&bytes[24..32]);
    let payload_len = decode_u64(&bytes[32..40]);
    if payload_offset != PAYLOAD_OFFSET
        || header_len == 0
        || PREAMBLE_LEN
            .checked_add(header_len)
            .is_none_or(|end| end as u64 > payload_offset)
    {
        return Err(invalid(format!(
            "invalid preamble header_len={header_len} payload_offset={payload_offset}"
        )));
    }
    let mut header_sha256 = [0u8; 32];
    header_sha256.copy_from_slice(&bytes[40..72]);
    if bytes[72..].iter().any(|byte| *byte != 0) {
        return Err(invalid("non-zero reserved preamble bytes"));
    }
    Ok((header_len, payload_len, header_sha256))
}

fn validate_manifest(
    manifest: &Gemma4Q4NativeManifest,
    specs: &[Q4Spec],
    source_sha256: &str,
    source_file_len: u64,
) -> Result<()> {
    if manifest.schema != SCHEMA
        || manifest.layout != LAYOUT
        || manifest.source_sha256 != source_sha256
        || manifest.source_sha256 != GEMMA4_12B_QAT_Q4_0_TARGET_SHA256
        || manifest.source_file_len != source_file_len
        || manifest.source_q4_bytes != PINNED_Q4_PAYLOAD_BYTES
        || manifest.payload_offset != PAYLOAD_OFFSET
        || manifest.payload_len != PINNED_Q4_PAYLOAD_BYTES
        || manifest.entries.len() != specs.len()
    {
        return Err(invalid(
            "manifest identity/layout/count/size fields do not match the exact target",
        ));
    }
    let mut next_payload = PAYLOAD_OFFSET;
    for (entry, spec) in manifest.entries.iter().zip(specs) {
        let expected_payload_len = spec
            .rows
            .div_ceil(8)
            .checked_mul(spec.blocks_per_row)
            .and_then(|records| records.checked_mul(NATIVE_OCTET_BLOCK_BYTES))
            .ok_or_else(|| invalid(format!("{} payload length overflow", spec.name)))?
            as u64;
        if entry.name != spec.name
            || entry.source_offset != spec.source_offset
            || entry.source_len != spec.source_len
            || entry.rows != spec.rows as u64
            || entry.columns != spec.columns as u64
            || entry.blocks_per_row != spec.blocks_per_row as u64
            || entry.payload_offset != next_payload
            || entry.payload_len != expected_payload_len
            || entry.payload_len != entry.source_len
            || entry.payload_sha256.len() != 64
            || !entry
                .payload_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(invalid(format!(
                "manifest entry {} does not match the exact source descriptor/native layout",
                spec.name
            )));
        }
        next_payload = next_payload
            .checked_add(entry.payload_len)
            .ok_or_else(|| invalid("manifest payload range overflow"))?;
    }
    if next_payload != PAYLOAD_OFFSET + manifest.payload_len {
        return Err(invalid(format!(
            "manifest payload ends at {next_payload}, expected {}",
            PAYLOAD_OFFSET + manifest.payload_len
        )));
    }
    Ok(())
}

/// Build the exact target's native Q4 sidecar.  Existing destinations are
/// never overwritten.  The completed file is installed only after all payload
/// hashes and the manifest have been written and synced.
pub fn repack_exact_gemma4_q4_sidecar(
    source: &Path,
    destination: &Path,
) -> Result<Gemma4Q4NativeManifest> {
    if source == destination {
        return Err(invalid("source and destination paths must differ"));
    }
    if destination.exists() {
        return Err(invalid(format!(
            "destination {} already exists; refusing to overwrite it",
            destination.display()
        )));
    }
    let gguf = crate::gguf::read_metadata(source)?;
    let source_file_len = std::fs::metadata(source)
        .map_err(|source_error| io(source, source_error))?
        .len();
    let source_sha256 = crate::receipt::sha256_file_hex(source)
        .map_err(|error| invalid(format!("hash source {}: {error}", source.display())))?;
    if source_sha256 != GEMMA4_12B_QAT_Q4_0_TARGET_SHA256 {
        return Err(invalid(format!(
            "source SHA-256 {source_sha256} does not match pinned target {}",
            GEMMA4_12B_QAT_Q4_0_TARGET_SHA256
        )));
    }
    let specs = exact_q4_specs(&gguf, source_file_len)?;
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let stem = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("gemma4-q4-native");
    let temporary = parent.join(format!(".{stem}.partial-{}", std::process::id()));
    if temporary.exists() {
        return Err(invalid(format!(
            "temporary output {} already exists",
            temporary.display()
        )));
    }

    let result = (|| -> Result<Gemma4Q4NativeManifest> {
        let mut source_file = File::open(source).map_err(|error| io(source, error))?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| io(&temporary, error))?;
        output
            .set_len(PAYLOAD_OFFSET)
            .map_err(|error| io(&temporary, error))?;
        output
            .seek(SeekFrom::Start(PAYLOAD_OFFSET))
            .map_err(|error| io(&temporary, error))?;

        let mut entries = Vec::with_capacity(specs.len());
        let mut payload_cursor = PAYLOAD_OFFSET;
        for spec in &specs {
            source_file
                .seek(SeekFrom::Start(spec.source_offset))
                .map_err(|error| io(source, error))?;
            let source_len = usize::try_from(spec.source_len).map_err(|_| {
                invalid(format!("{} source length does not fit usize", spec.name))
            })?;
            let mut wire = vec![0u8; source_len];
            source_file
                .read_exact(&mut wire)
                .map_err(|error| io(source, error))?;
            let native = pack_native_q4_octets(&wire, spec.rows, spec.blocks_per_row)?;
            if native.len() as u64 != spec.source_len {
                return Err(invalid(format!(
                    "{} native payload grew from {} to {} bytes",
                    spec.name,
                    spec.source_len,
                    native.len()
                )));
            }
            output
                .write_all(&native)
                .map_err(|error| io(&temporary, error))?;
            entries.push(Gemma4Q4NativeEntry {
                name: spec.name.clone(),
                source_offset: spec.source_offset,
                source_len: spec.source_len,
                rows: spec.rows as u64,
                columns: spec.columns as u64,
                blocks_per_row: spec.blocks_per_row as u64,
                payload_offset: payload_cursor,
                payload_len: native.len() as u64,
                payload_sha256: sha256_hex(&native),
            });
            payload_cursor = payload_cursor
                .checked_add(native.len() as u64)
                .ok_or_else(|| invalid("sidecar payload cursor overflow"))?;
        }
        let payload_len = payload_cursor - PAYLOAD_OFFSET;
        let manifest = Gemma4Q4NativeManifest {
            schema: SCHEMA.to_string(),
            layout: LAYOUT.to_string(),
            source_sha256: source_sha256.clone(),
            source_file_len,
            source_q4_bytes: PINNED_Q4_PAYLOAD_BYTES,
            payload_offset: PAYLOAD_OFFSET,
            payload_len,
            entries,
        };
        validate_manifest(&manifest, &specs, &source_sha256, source_file_len)?;
        let header = serde_json::to_vec(&manifest)
            .map_err(|error| invalid(format!("serialize manifest: {error}")))?;
        if PREAMBLE_LEN + header.len() > PAYLOAD_OFFSET as usize {
            return Err(invalid(format!(
                "manifest is {} bytes and exceeds the reserved index area",
                header.len()
            )));
        }
        let header_digest: [u8; 32] = Sha256::digest(&header).into();
        let preamble = encode_preamble(header.len(), &header_digest, payload_len);
        output
            .seek(SeekFrom::Start(0))
            .map_err(|error| io(&temporary, error))?;
        output
            .write_all(&preamble)
            .and_then(|_| output.write_all(&header))
            .map_err(|error| io(&temporary, error))?;
        output.sync_all().map_err(|error| io(&temporary, error))?;
        drop(output);
        install_no_replace(&temporary, destination).map_err(|error| io(destination, error))?;
        Ok(manifest)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

/// Validated reader for an exact native-Q4 sidecar.  Each requested matrix is
/// copied once into page-aligned `WirePages`, hashed in that final resident
/// allocation, and handed directly to Metal as a no-copy buffer.
#[derive(Debug)]
pub(crate) struct Gemma4Q4NativeSidecar {
    path: PathBuf,
    file: File,
    entries: BTreeMap<String, Gemma4Q4NativeEntry>,
    source_ranges: Vec<(usize, usize)>,
}

impl Gemma4Q4NativeSidecar {
    pub(crate) fn open(
        path: &Path,
        gguf: &GgufFile,
        source_file_len: u64,
        source_sha256: &str,
    ) -> Result<Self> {
        if source_sha256 != GEMMA4_12B_QAT_Q4_0_TARGET_SHA256 {
            return Err(invalid(format!(
                "loaded source SHA-256 {source_sha256} does not match pinned target {}",
                GEMMA4_12B_QAT_Q4_0_TARGET_SHA256
            )));
        }
        let specs = exact_q4_specs(gguf, source_file_len)?;
        let mut file = File::open(path).map_err(|error| io(path, error))?;
        // The payload is copied once into anonymous, page-aligned NoCopy
        // allocations. Keeping the sidecar's clean file-cache pages as well
        // would transiently recreate the very duplicate this layout replaces.
        crate::tensor::disable_file_cache_best_effort(&file);
        let sidecar_len = file.metadata().map_err(|error| io(path, error))?.len();
        let mut preamble = [0u8; PREAMBLE_LEN];
        file.read_exact(&mut preamble)
            .map_err(|error| io(path, error))?;
        let (header_len, payload_len, expected_header_sha) = decode_preamble(&preamble)?;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)
            .map_err(|error| io(path, error))?;
        let reserved_len = PAYLOAD_OFFSET as usize - PREAMBLE_LEN - header_len;
        let mut reserved = vec![0u8; reserved_len];
        file.read_exact(&mut reserved)
            .map_err(|error| io(path, error))?;
        if reserved.iter().any(|byte| *byte != 0) {
            return Err(invalid("non-zero reserved sidecar index padding"));
        }
        let actual_header_sha: [u8; 32] = Sha256::digest(&header).into();
        if actual_header_sha != expected_header_sha {
            return Err(invalid(
                "manifest SHA-256 does not match the sidecar preamble",
            ));
        }
        let manifest: Gemma4Q4NativeManifest = serde_json::from_slice(&header)
            .map_err(|error| invalid(format!("parse manifest JSON: {error}")))?;
        if payload_len != manifest.payload_len {
            return Err(invalid(format!(
                "preamble payload length {payload_len} != manifest {}",
                manifest.payload_len
            )));
        }
        validate_manifest(&manifest, &specs, source_sha256, source_file_len)?;
        let expected_sidecar_len = PAYLOAD_OFFSET
            .checked_add(payload_len)
            .ok_or_else(|| invalid("sidecar file length overflow"))?;
        if sidecar_len != expected_sidecar_len {
            return Err(invalid(format!(
                "sidecar file is {sidecar_len} bytes, expected exactly {expected_sidecar_len}"
            )));
        }
        let source_ranges = specs
            .iter()
            .map(|spec| {
                let offset = usize::try_from(spec.source_offset).map_err(|_| {
                    invalid(format!("{} source offset does not fit usize", spec.name))
                })?;
                let len = usize::try_from(spec.source_len).map_err(|_| {
                    invalid(format!("{} source length does not fit usize", spec.name))
                })?;
                Ok((offset, len))
            })
            .collect::<Result<Vec<_>>>()?;
        let entries = manifest
            .entries
            .into_iter()
            .map(|entry| (entry.name.clone(), entry))
            .collect();
        Ok(Self {
            path: path.to_path_buf(),
            file,
            entries,
            source_ranges,
        })
    }

    pub(crate) fn read_pages(
        &self,
        descriptor: &GgufTensorDescriptor,
    ) -> Result<Arc<crate::wire_mmap::WirePages>> {
        let entry = self.entries.get(&descriptor.name).ok_or_else(|| {
            invalid(format!(
                "validated index has no entry for projection {}",
                descriptor.name
            ))
        })?;
        if descriptor.tensor_type != GgufTensorType::Q4_0
            || descriptor.absolute_offset != entry.source_offset
            || descriptor.n_bytes != entry.source_len
        {
            return Err(invalid(format!(
                "runtime descriptor for {} no longer matches validated index",
                descriptor.name
            )));
        }
        let payload_len = usize::try_from(entry.payload_len)
            .map_err(|_| invalid(format!("{} payload length does not fit usize", entry.name)))?;
        let pages = crate::wire_mmap::WirePages::read_from_file(
            &self.file,
            entry.payload_offset,
            payload_len,
        )?;
        let actual_sha = sha256_hex(pages.bytes());
        if actual_sha != entry.payload_sha256 {
            return Err(invalid(format!(
                "{} resident payload SHA-256 {actual_sha} != index {} in {}",
                entry.name,
                entry.payload_sha256,
                self.path.display()
            )));
        }
        Ok(pages)
    }

    /// Read and hash every indexed projection before Metal owns any of them.
    /// This makes corrupt-payload refusal transactional with respect to the
    /// process-wide NoCopy buffer cache: a late bad entry cannot leave a
    /// partially resident native model pinned after load returns an error.
    pub(crate) fn read_all_pages(
        &self,
        gguf: &GgufFile,
    ) -> Result<BTreeMap<String, Arc<crate::wire_mmap::WirePages>>> {
        let descriptors = gguf
            .tensors
            .iter()
            .map(|descriptor| (descriptor.name.as_str(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut entries = self.entries.values().collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.payload_offset);
        let mut pages = BTreeMap::new();
        for entry in entries {
            let descriptor = descriptors.get(entry.name.as_str()).ok_or_else(|| {
                invalid(format!(
                    "runtime metadata has no descriptor for indexed projection {}",
                    entry.name
                ))
            })?;
            pages.insert(entry.name.clone(), self.read_pages(descriptor)?);
        }
        if pages.len() != self.entries.len() {
            return Err(invalid("resident native Q4 page index is incomplete"));
        }
        Ok(pages)
    }

    pub(crate) fn source_ranges(&self) -> &[(usize, usize)] {
        &self.source_ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "camelid-q4-sidecar-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn wire(rows: usize, blocks_per_row: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows * blocks_per_row * WIRE_BLOCK_BYTES);
        for row in 0..rows {
            for block in 0..blocks_per_row {
                bytes.extend_from_slice(&((row * 101 + block) as u16).to_le_bytes());
                for quant in 0..16 {
                    bytes.push((row * 31 + block * 17 + quant) as u8);
                }
            }
        }
        bytes
    }

    #[test]
    fn octet_pack_preserves_every_source_field_and_pads_ragged_rows() {
        let rows = 11;
        let blocks_per_row = 3;
        let source = wire(rows, blocks_per_row);
        let native = pack_native_q4_octets(&source, rows, blocks_per_row).unwrap();
        assert_eq!(
            native.len(),
            rows.div_ceil(8) * blocks_per_row * NATIVE_OCTET_BLOCK_BYTES
        );
        for row in 0..rows {
            let octet = row / 8;
            let tile_row = row % 8;
            let tile_base = octet * blocks_per_row * NATIVE_OCTET_BLOCK_BYTES;
            let quant_base = tile_base + blocks_per_row * 16;
            for block in 0..blocks_per_row {
                let source_offset = (row * blocks_per_row + block) * WIRE_BLOCK_BYTES;
                let scale_offset = tile_base + (block * 8 + tile_row) * 2;
                assert_eq!(
                    &native[scale_offset..scale_offset + 2],
                    &source[source_offset..source_offset + 2]
                );
                for quarter in 0..4 {
                    let destination = quant_base + block * 128 + quarter * 32 + tile_row * 4;
                    let quant_source = source_offset + 2 + quarter * 4;
                    assert_eq!(
                        &native[destination..destination + 4],
                        &source[quant_source..quant_source + 4]
                    );
                }
            }
        }
        let tail_tile = (rows / 8) * blocks_per_row * NATIVE_OCTET_BLOCK_BYTES;
        let tail_quant = tail_tile + blocks_per_row * 16;
        for tile_row in 3..8 {
            for block in 0..blocks_per_row {
                let scale = tail_tile + (block * 8 + tile_row) * 2;
                assert_eq!(&native[scale..scale + 2], &[0, 0]);
                for quarter in 0..4 {
                    let quant = tail_quant + block * 128 + quarter * 32 + tile_row * 4;
                    assert_eq!(&native[quant..quant + 4], &[0x88; 4]);
                }
            }
        }
    }

    #[test]
    fn aligned_octets_have_exactly_the_q4_wire_payload_size() {
        for rows in [8usize, 16, 3_840, 4_096] {
            let blocks_per_row = 3;
            let source = wire(rows, blocks_per_row);
            let native = pack_native_q4_octets(&source, rows, blocks_per_row).unwrap();
            assert_eq!(native.len(), source.len());
        }
    }

    #[test]
    fn preamble_round_trip_and_reserved_bytes_fail_closed() {
        let digest = [0x5au8; 32];
        let encoded = encode_preamble(12_345, &digest, PINNED_Q4_PAYLOAD_BYTES);
        assert_eq!(
            decode_preamble(&encoded).unwrap(),
            (12_345, PINNED_Q4_PAYLOAD_BYTES, digest)
        );
        let mut corrupt = encoded;
        corrupt[95] = 1;
        assert!(decode_preamble(&corrupt).is_err());
    }

    #[test]
    fn completed_sidecar_install_is_atomic_and_never_replaces_a_destination() {
        let temporary = temporary_path("install-partial");
        let collision = temporary_path("install-collision");
        let destination = temporary_path("install-final");
        std::fs::write(&temporary, b"complete-native-payload").unwrap();
        install_no_replace(&temporary, &destination).unwrap();
        assert!(!temporary.exists());
        assert_eq!(
            std::fs::read(&destination).unwrap(),
            b"complete-native-payload"
        );

        std::fs::write(&collision, b"must-not-win").unwrap();
        assert!(install_no_replace(&collision, &destination).is_err());
        assert_eq!(
            std::fs::read(&destination).unwrap(),
            b"complete-native-payload"
        );
        assert_eq!(std::fs::read(&collision).unwrap(), b"must-not-win");

        std::fs::remove_file(destination).unwrap();
        std::fs::remove_file(collision).unwrap();
    }
}
