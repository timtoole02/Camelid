//! Ghost (layer-streaming) mode: the `.cghost` container format and its reader/writer.
//!
//! Standard GGUF files scatter a transformer block's tensors across the file, which turns a
//! layer-by-layer streaming pass into random reads. A `.cghost` file is a pure re-layout of
//! a GGUF at **source quantization**: every tensor a block needs is contiguous on disk, so
//! streaming one layer is ONE sequential read. v1 deliberately does not requantize â€”
//! identical bytes mean the ghost path can be parity-gated against the resident path.
//!
//! Layout:
//! ```text
//! [magic "CGHOST1\0"][u64 index_offset][pad to 16 KiB]
//! [group 0 payload][pad][group 1 payload][pad]...
//! [index JSON]                                    <- at index_offset, runs to EOF
//! ```
//! Group payload offsets are 16 KiB aligned (Apple Silicon page size) so future no-copy /
//! madvise work operates on whole pages; tensors inside a group are packed back-to-back.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::platform_fs::read_exact_at;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{BackendError, Result};
use crate::gguf::{GgufTensorDescriptor, GgufTensorType};
use crate::inference::{DecodeLinearBindings, LlamaLayerWeights};
use crate::model::{
    Gemma4Binding, LlamaAttentionTensors, LlamaFfnTensors, LlamaModelConfig, LlamaTensorBinding,
};
use crate::tensor::{cpu_tensor_from_gguf_bytes, CpuTensor, TensorStore};
use crate::wire_mmap::GgufWireMmap;

pub const CGHOST_MAGIC: &[u8; 8] = b"CGHOST1\0";
pub const CGHOST_ALIGN: u64 = 16 * 1024;
const MOE_SOURCE_IDENTITY_SCHEME: &str = "gemma4-moe-sampled-sha256-v1";
pub(crate) const MOE_IDENTITY_SAMPLE_BYTES: u64 = 128;
const MOE_IDENTITY_SAMPLE_ALIGNMENT: u64 = 4096;
const ALLOW_LEGACY_SPARSE_IDENTITY_ENV: &str = "CAMELID_GHOST_ALLOW_LEGACY_SPARSE";

/// Physical organization of a `.cghost` payload. Old indexes predate this
/// discriminator and deserialize as [`DenseLayers`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CghostLayout {
    #[default]
    DenseLayers,
    MoeExperts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostTensor {
    pub name: String,
    /// Stable role inside the group ("attn_norm", "attn_q", ..., "token_embedding").
    pub role: String,
    pub dtype: GgufTensorType,
    pub dims: Vec<u64>,
    /// Absolute file offset of this tensor's raw GGUF bytes.
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostGroup {
    /// "pre" (embedding + rope), "blk.N" (one transformer block), "post" (output norm/proj).
    pub id: String,
    pub tensors: Vec<CghostTensor>,
    /// SHA-256 of bounded, deterministic samples from this expert record.
    /// Legacy indexes omit it. New MoE indexes use it both to bind the source
    /// GGUF/hot shadow and to validate bytes after each record read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sample_sha256: Option<String>,
}

/// Bounded source identity for a Ghost-MoE pair. This deliberately is not a
/// whole-file checksum: a 14+ GiB source must not be reread at every startup.
/// Instead, every routed expert contributes a deterministic sample through its
/// [`CghostGroup`] and every bound common tensor contributes one sample here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CghostSourceIdentity {
    pub scheme: String,
    pub logical_bytes: u64,
    pub sample_bytes: u32,
    pub common_sha256: String,
}

impl CghostGroup {
    /// Contiguous (start, len) span of this group's payload in the file.
    pub fn span(&self) -> (u64, u64) {
        let start = self.tensors.first().map(|t| t.offset).unwrap_or(0);
        let end = self
            .tensors
            .last()
            .map(|t| t.offset + t.len)
            .unwrap_or(start);
        (start, end - start)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostIndex {
    pub version: u32,
    #[serde(default)]
    pub layout: CghostLayout,
    pub source_model: String,
    pub block_count: usize,
    pub tied_output: bool,
    /// Present only for [`CghostLayout::MoeExperts`].
    #[serde(default)]
    pub expert_count: Option<usize>,
    /// Number of routed experts selected per token. Informational at the file
    /// layer, but validated by the Gemma 4 loader before generation.
    #[serde(default)]
    pub expert_used_count: Option<usize>,
    /// Present on newly repacked MoE artifacts. Optional so existing v1/v2
    /// `.cghost` files remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_identity: Option<CghostSourceIdentity>,
    pub groups: Vec<CghostGroup>,
}

/// The fixed tensor order inside a "blk.N" group. Decoding is positional-by-role, so the
/// writer and reader must agree on this list.
const LAYER_ROLES: [&str; 9] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

/// Queue an asynchronous read-ahead over a mapped byte range. Best-effort by
/// design: a failure just means the pages fault in on demand, as they did before.
#[cfg(all(feature = "cuda", windows))]
fn prefetch_range_best_effort(range: &[u8]) {
    use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if range.is_empty() {
        return;
    }
    let entry = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: range.as_ptr() as *mut core::ffi::c_void,
        NumberOfBytes: range.len(),
    };
    // SAFETY: `entry` describes a live, immutably-mapped range owned by the caller's
    // `GgufWireMmap`, which outlives this call. The API only hints the pager; it never
    // writes, and the return value is deliberately discarded.
    unsafe {
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
}

#[cfg(all(feature = "cuda", not(windows)))]
fn prefetch_range_best_effort(range: &[u8]) {
    if range.is_empty() {
        return;
    }
    // SAFETY: as above — an advisory call over a live read-only mapping.
    unsafe {
        libc::posix_madvise(
            range.as_ptr() as *mut libc::c_void,
            range.len(),
            libc::POSIX_MADV_WILLNEED,
        );
    }
}

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidModelMetadata(msg)
}

fn io_err(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn canonical_artifact_candidate(path: &Path) -> std::io::Result<std::path::PathBuf> {
    if path.exists() {
        return std::fs::canonicalize(path);
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = std::fs::canonicalize(parent)?;
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("artifact path {} has no file name", path.display()),
        )
    })?;
    Ok(parent.join(file_name))
}

#[cfg(unix)]
fn same_existing_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;

    fn identity(path: &Path) -> std::io::Result<Option<(u64, u64)>> {
        match std::fs::metadata(path) {
            Ok(metadata) => Ok(Some((metadata.dev(), metadata.ino()))),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    Ok(matches!(
        (identity(left)?, identity(right)?),
        (Some(left), Some(right)) if left == right
    ))
}

#[cfg(windows)]
fn same_existing_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    fn identity(path: &Path) -> std::io::Result<Option<(u32, u64)>> {
        let file = match File::open(path) {
            Ok(file) => file,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(err),
        };
        let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
        // SAFETY: `file` owns a live handle and `info` points to writable storage
        // for exactly one BY_HANDLE_FILE_INFORMATION value.
        if unsafe { GetFileInformationByHandle(file.as_raw_handle() as _, info.as_mut_ptr()) } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let info = unsafe { info.assume_init() };
        let file_index = ((info.nFileIndexHigh as u64) << 32) | info.nFileIndexLow as u64;
        Ok(Some((info.dwVolumeSerialNumber, file_index)))
    }

    Ok(matches!(
        (identity(left)?, identity(right)?),
        (Some(left), Some(right)) if left == right
    ))
}

#[cfg(not(any(unix, windows)))]
fn same_existing_file(_left: &Path, _right: &Path) -> std::io::Result<bool> {
    Ok(false)
}

/// Fail closed if any artifact path names the same destination.
///
/// Canonical paths catch spelling and symlink aliases. Existing files are also
/// compared by stable filesystem identity so distinct hard-link names cannot
/// turn a repack into an in-place truncate. Callers must invoke this before
/// creating or truncating any member of the set.
pub fn ensure_distinct_artifact_paths(artifacts: &[(&str, &Path)]) -> Result<()> {
    let canonical = artifacts
        .iter()
        .map(|(_, path)| canonical_artifact_candidate(path).map_err(|err| io_err(path, err)))
        .collect::<Result<Vec<_>>>()?;
    for left_idx in 0..artifacts.len() {
        for right_idx in left_idx + 1..artifacts.len() {
            let (left_label, left_path) = artifacts[left_idx];
            let (right_label, right_path) = artifacts[right_idx];
            let same_identity =
                same_existing_file(left_path, right_path).map_err(|err| io_err(right_path, err))?;
            if canonical[left_idx] == canonical[right_idx] || same_identity {
                return Err(invalid(format!(
                    "Ghost artifacts must be distinct files: {left_label} {} and {right_label} {} resolve to the same file",
                    left_path.display(),
                    right_path.display()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn source_is_sparse(path: &Path) -> Result<bool> {
    use std::os::unix::fs::MetadataExt;

    let metadata = std::fs::metadata(path).map_err(|err| io_err(path, err))?;
    Ok(metadata.len() > 0 && metadata.blocks().saturating_mul(512) < metadata.len())
}

#[cfg(windows)]
fn source_is_sparse(path: &Path) -> Result<bool> {
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_SPARSE_FILE;

    let metadata = std::fs::metadata(path).map_err(|err| io_err(path, err))?;
    Ok(metadata.file_attributes() & FILE_ATTRIBUTE_SPARSE_FILE != 0)
}

#[cfg(not(any(unix, windows)))]
fn source_is_sparse(_path: &Path) -> Result<bool> {
    Ok(false)
}

fn allow_legacy_sparse_identity() -> bool {
    std::env::var(ALLOW_LEGACY_SPARSE_IDENTITY_ENV)
        .ok()
        .is_some_and(|value| {
            value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
        })
}

fn sha256_hex(hasher: Sha256) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn valid_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn hash_bytes(hasher: &mut Sha256, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_tensor_geometry(
    hasher: &mut Sha256,
    role: &str,
    dtype: GgufTensorType,
    dims: &[u64],
    len: u64,
) {
    hash_bytes(hasher, role.as_bytes());
    hash_bytes(hasher, format!("{dtype:?}").as_bytes());
    hash_u64(hasher, dims.len() as u64);
    for dim in dims {
        hash_u64(hasher, *dim);
    }
    hash_u64(hasher, len);
}

fn bounded_sample_relative_offset(seed: &[u8], len: u64) -> u64 {
    let sample_len = len.min(MOE_IDENTITY_SAMPLE_BYTES);
    if sample_len == len {
        return 0;
    }
    let mut hasher = Sha256::new();
    hasher.update(MOE_SOURCE_IDENTITY_SCHEME.as_bytes());
    hasher.update(seed);
    hash_u64(&mut hasher, len);
    let digest = hasher.finalize();
    let random = u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix"));
    let slots = (len - sample_len) / MOE_IDENTITY_SAMPLE_ALIGNMENT + 1;
    (random % slots) * MOE_IDENTITY_SAMPLE_ALIGNMENT
}

/// Exact byte range retained inside a sparse expert slice for source identity.
/// The start is page-aligned whenever the slice is large enough, keeping sparse
/// shadows compact while making the plan independent of filesystem metadata.
pub(crate) fn moe_identity_sample_range(
    slice_start: u64,
    slice_len: u64,
    layer_idx: usize,
    expert_idx: usize,
    role: &str,
) -> std::ops::Range<u64> {
    let seed = format!("layer={layer_idx};expert={expert_idx};role={role}");
    let relative = bounded_sample_relative_offset(seed.as_bytes(), slice_len);
    let start = slice_start + relative;
    start..start + slice_len.min(MOE_IDENTITY_SAMPLE_BYTES)
}

fn begin_moe_group_fingerprint(layer_idx: usize, expert_idx: usize) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(MOE_SOURCE_IDENTITY_SCHEME.as_bytes());
    hasher.update(b"expert-group");
    hash_u64(&mut hasher, layer_idx as u64);
    hash_u64(&mut hasher, expert_idx as u64);
    hasher
}

fn update_moe_group_sample(
    hasher: &mut Sha256,
    role: &str,
    dtype: GgufTensorType,
    dims: &[u64],
    tensor_len: u64,
    sample_relative: u64,
    sample: &[u8],
) {
    hash_tensor_geometry(hasher, role, dtype, dims, tensor_len);
    hash_u64(hasher, sample_relative);
    hash_bytes(hasher, sample);
}

fn source_moe_group_fingerprint(
    source: &File,
    source_path: &Path,
    binding: &Gemma4Binding,
    expert_count: usize,
    layer_idx: usize,
    expert_idx: usize,
) -> Result<String> {
    let moe = binding.layers[layer_idx].moe.as_ref().ok_or_else(|| {
        invalid(format!(
            "model layer {layer_idx} has no routed-expert binding"
        ))
    })?;
    let mut hasher = begin_moe_group_fingerprint(layer_idx, expert_idx);
    for (role, desc) in [
        ("gate_up_exps", &moe.gate_up_exps),
        ("down_exps", &moe.down_exps),
    ] {
        if !desc.n_bytes.is_multiple_of(expert_count as u64) {
            return Err(invalid(format!(
                "model tensor {} byte length {} is not divisible by expert_count {expert_count}",
                desc.name, desc.n_bytes
            )));
        }
        let tensor_len = desc.n_bytes / expert_count as u64;
        let slice_start = desc.absolute_offset + expert_idx as u64 * tensor_len;
        let sample_range =
            moe_identity_sample_range(slice_start, tensor_len, layer_idx, expert_idx, role);
        let mut sample = vec![0u8; (sample_range.end - sample_range.start) as usize];
        read_exact_at(source, &mut sample, sample_range.start)
            .map_err(|err| io_err(source_path, err))?;
        let mut dims = desc.dimensions.clone();
        if dims.pop() != Some(expert_count as u64) {
            return Err(invalid(format!(
                "model tensor {} does not end in expert_count {expert_count}",
                desc.name
            )));
        }
        update_moe_group_sample(
            &mut hasher,
            role,
            desc.tensor_type,
            &dims,
            tensor_len,
            sample_range.start - slice_start,
            &sample,
        );
    }
    Ok(sha256_hex(hasher))
}

fn gemma4_common_descriptors(binding: &Gemma4Binding) -> Vec<&GgufTensorDescriptor> {
    let mut descriptors = vec![
        &binding.token_embedding,
        &binding.output_norm,
        &binding.output,
    ];
    descriptors.extend(binding.rope_freqs.iter());
    descriptors.extend(binding.per_layer_token_embd.iter());
    descriptors.extend(binding.per_layer_model_proj.iter());
    descriptors.extend(binding.per_layer_proj_norm.iter());
    for layer in &binding.layers {
        descriptors.extend([
            &layer.attn_norm,
            &layer.attn_q,
            &layer.attn_output,
            &layer.attn_q_norm,
            &layer.post_attention_norm,
            &layer.ffn_norm,
            &layer.post_ffw_norm,
            &layer.ffn_gate,
            &layer.ffn_up,
            &layer.ffn_down,
        ]);
        descriptors.extend(layer.attn_k.iter());
        descriptors.extend(layer.attn_v.iter());
        descriptors.extend(layer.attn_k_norm.iter());
        descriptors.extend(layer.post_norm.iter());
        descriptors.extend(layer.ple_inp_gate.iter());
        descriptors.extend(layer.ple_proj.iter());
        descriptors.extend(layer.ple_output_scale.iter());
        if let Some(moe) = &layer.moe {
            descriptors.extend([
                &moe.gate_inp,
                &moe.gate_inp_scale,
                &moe.down_exps_scale,
                &moe.pre_norm_2,
                &moe.post_norm_1,
                &moe.post_norm_2,
            ]);
        }
    }
    descriptors.sort_unstable_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.absolute_offset.cmp(&right.absolute_offset))
    });
    descriptors.dedup_by(|left, right| {
        left.name == right.name
            && left.absolute_offset == right.absolute_offset
            && left.n_bytes == right.n_bytes
    });
    descriptors
}

fn common_source_fingerprint(
    source: &File,
    source_path: &Path,
    binding: &Gemma4Binding,
) -> Result<String> {
    let descriptors = gemma4_common_descriptors(binding);
    let mut hasher = Sha256::new();
    hasher.update(MOE_SOURCE_IDENTITY_SCHEME.as_bytes());
    hasher.update(b"common-tensors");
    hash_u64(&mut hasher, descriptors.len() as u64);
    for desc in descriptors {
        hash_bytes(&mut hasher, desc.name.as_bytes());
        hash_tensor_geometry(
            &mut hasher,
            "common",
            desc.tensor_type,
            &desc.dimensions,
            desc.n_bytes,
        );
        hash_u64(&mut hasher, desc.absolute_offset);
        let seed = format!("common:{}", desc.name);
        let relative = bounded_sample_relative_offset(seed.as_bytes(), desc.n_bytes);
        let sample_len = desc.n_bytes.min(MOE_IDENTITY_SAMPLE_BYTES) as usize;
        let mut sample = vec![0u8; sample_len];
        read_exact_at(source, &mut sample, desc.absolute_offset + relative)
            .map_err(|err| io_err(source_path, err))?;
        hash_u64(&mut hasher, relative);
        hash_bytes(&mut hasher, &sample);
    }
    Ok(sha256_hex(hasher))
}

/// Write a `.cghost` re-layout of `store`'s GGUF. Dense models only â€” ghost v1 refuses MoE
/// (the streaming window assumes one fixed-size group per block).
///
/// `layer_range` selects a pipeline shard: only those "blk.N" groups are written, the "pre"
/// group (embedding) only when the range starts at layer 0, and the "post" group
/// (output norm/projection) only when it ends at the last layer â€” so each mesh node hosts
/// just its own half of the payload on local disk. `None` writes the whole model.
pub fn write_cghost(
    store: &TensorStore,
    binding: &LlamaTensorBinding,
    source_model: &str,
    out_path: &Path,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<CghostIndex> {
    ensure_distinct_artifact_paths(&[
        ("source GGUF", store.source_path()),
        (".cghost output", out_path),
    ])?;
    let total_layers = binding.layers.len();
    let range = layer_range.unwrap_or(0..total_layers);
    if range.start >= range.end || range.end > total_layers {
        return Err(invalid(format!(
            "layer range {range:?} is invalid for a {total_layers}-layer model"
        )));
    }
    // Plan the group contents (names + roles) first.
    let mut planned: Vec<(String, Vec<(String, String)>)> = Vec::new();
    if range.start == 0 {
        let mut pre = vec![(
            "token_embedding".to_string(),
            binding.token_embedding.name.clone(),
        )];
        if let Some(rope) = &binding.rope_freqs {
            pre.push(("rope_freqs".to_string(), rope.name.clone()));
        }
        planned.push(("pre".to_string(), pre));
    }
    for (layer_idx, layer) in binding.layers.iter().enumerate() {
        if !range.contains(&layer_idx) {
            continue;
        }
        let (gate, up, down) = match &layer.ffn {
            LlamaFfnTensors::Dense { gate, up, down } => (gate, up, down),
            _ => {
                return Err(invalid(format!(
                    "layer {layer_idx} is MoE; ghost v1 supports dense models only"
                )))
            }
        };
        let (q, k, v) = match &layer.attention {
            LlamaAttentionTensors::Standard {
                q,
                k,
                v,
                biases: None,
                ..
            } => (q, k, v),
            LlamaAttentionTensors::Standard {
                biases: Some(_), ..
            } => {
                return Err(invalid(format!(
                "layer {layer_idx} has attention projection biases; ghost v1 cannot preserve them"
            )))
            }
            _ => {
                return Err(invalid(format!(
                    "layer {layer_idx} is MLA; ghost v1 supports standard attention only"
                )))
            }
        };
        let tensors = vec![
            ("attn_norm".to_string(), layer.attention_norm.name.clone()),
            ("attn_q".to_string(), q.name.clone()),
            ("attn_k".to_string(), k.name.clone()),
            ("attn_v".to_string(), v.name.clone()),
            (
                "attn_output".to_string(),
                layer.attention_output.name.clone(),
            ),
            ("ffn_norm".to_string(), layer.ffn_norm.name.clone()),
            ("ffn_gate".to_string(), gate.name.clone()),
            ("ffn_up".to_string(), up.name.clone()),
            ("ffn_down".to_string(), down.name.clone()),
        ];
        planned.push((format!("blk.{layer_idx}"), tensors));
    }
    if range.end == total_layers {
        let mut post = vec![("output_norm".to_string(), binding.output_norm.name.clone())];
        if !binding.output_is_tied_embedding {
            post.push(("output".to_string(), binding.output.name.clone()));
        }
        planned.push(("post".to_string(), post));
    }

    // Stream the payload out group by group, page-aligning each group start.
    let mut file = File::create(out_path).map_err(|e| io_err(out_path, e))?;
    file.write_all(CGHOST_MAGIC)
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&0u64.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    let mut cursor: u64 = (CGHOST_MAGIC.len() + 8) as u64;
    let mut groups = Vec::with_capacity(planned.len());
    for (id, tensors) in planned {
        let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
        if aligned > cursor {
            file.write_all(&vec![0u8; (aligned - cursor) as usize])
                .map_err(|e| io_err(out_path, e))?;
            cursor = aligned;
        }
        let mut group = CghostGroup {
            id,
            tensors: Vec::with_capacity(tensors.len()),
            source_sample_sha256: None,
        };
        for (role, name) in tensors {
            let desc = store.descriptor(&name)?.clone();
            let bytes = store.tensor_bytes(&name)?;
            file.write_all(&bytes).map_err(|e| io_err(out_path, e))?;
            group.tensors.push(CghostTensor {
                name,
                role,
                dtype: desc.tensor_type,
                dims: desc.dimensions.clone(),
                offset: cursor,
                len: bytes.len() as u64,
            });
            cursor += bytes.len() as u64;
        }
        groups.push(group);
    }

    let index = CghostIndex {
        version: 1,
        layout: CghostLayout::DenseLayers,
        source_model: source_model.to_string(),
        block_count: binding.layers.len(),
        tied_output: binding.output_is_tied_embedding,
        expert_count: None,
        expert_used_count: None,
        source_identity: None,
        groups,
    };
    let index_json = serde_json::to_vec(&index)
        .map_err(|e| invalid(format!("failed to serialize .cghost index: {e}")))?;
    let index_offset = cursor;
    file.write_all(&index_json)
        .map_err(|e| io_err(out_path, e))?;
    file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&index_offset.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    file.sync_all().map_err(|e| io_err(out_path, e))?;
    Ok(index)
}

/// Write a Gemma 4 MoE `.cghost` payload. Shared tensors stay in the source
/// GGUF and are read through the normal wire runtime; only the two large routed
/// expert tensors are spliced into fixed `(layer, expert)` groups:
///
/// ```text
/// blk.0.exp.0: gate_up_exps slice + down_exps slice
/// blk.0.exp.1: gate_up_exps slice + down_exps slice
/// ...
/// ```
///
/// Each selected expert is therefore one sequential positioned read instead of
/// two distant reads into multi-gigabyte GGUF tensors. Copying is chunked, so
/// repacking never materializes a full expert tensor (or even a full expert) in
/// RAM.
pub fn write_cghost_moe(
    store: &TensorStore,
    binding: &Gemma4Binding,
    config: &LlamaModelConfig,
    source_model: &str,
    out_path: &Path,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<CghostIndex> {
    let moe = config.moe.as_ref().ok_or_else(|| {
        invalid("ghost MoE repack requires mixture-of-experts metadata".to_string())
    })?;
    write_cghost_moe_with_counts(
        store,
        binding,
        moe.expert_count as usize,
        moe.expert_used_count as usize,
        source_model,
        out_path,
        layer_range,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_cghost_moe_with_counts(
    store: &TensorStore,
    binding: &Gemma4Binding,
    expert_count: usize,
    expert_used_count: usize,
    source_model: &str,
    out_path: &Path,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<CghostIndex> {
    ensure_distinct_artifact_paths(&[
        ("source GGUF", store.source_path()),
        (".cghost output", out_path),
    ])?;
    // Refuse a sparse source. Repacking from the hot shadow (models/*.hot) instead of
    // the full GGUF is a one-character mistake that produces a SILENTLY WRONG artifact:
    // the shadow zeroes the routed-expert extents but deliberately retains a 128-byte
    // identity island at exactly the offset `validate_moe_expert_payload_identity`
    // samples, and this writer's verbatim copy preserves that byte at the same
    // record-relative offset. So a mostly-zero .cghost passes BOTH the source-identity
    // sweep and the per-record payload check, and the only symptom is degraded output.
    // `source_is_sparse` already existed but was wired only into the READ path.
    if source_is_sparse(store.source_path())? {
        return Err(invalid(format!(
            "{} is a sparse file — that is a hot shadow, not a full GGUF, and its routed-expert \
             ranges are holes. Repacking from it would write a .cghost of mostly zeros that still \
             passes every identity check. Repack from the full GGUF instead.",
            store.source_path().display()
        )));
    }
    let total_layers = binding.layers.len();
    let range = layer_range.unwrap_or(0..total_layers);
    if range.start >= range.end || range.end > total_layers {
        return Err(invalid(format!(
            "layer range {range:?} is invalid for a {total_layers}-layer model"
        )));
    }
    if range != (0..total_layers) {
        return Err(invalid(format!(
            "ghost MoE currently requires all {total_layers} layers; partial range {range:?} is not consumable by the single-node runner"
        )));
    }
    if expert_count == 0 || expert_used_count == 0 || expert_used_count > expert_count {
        return Err(invalid(format!(
            "invalid MoE expert counts: total={}, used={}",
            expert_count, expert_used_count
        )));
    }

    let source = File::open(store.source_path()).map_err(|e| io_err(store.source_path(), e))?;
    let source_len = source
        .metadata()
        .map_err(|e| io_err(store.source_path(), e))?
        .len();
    let mut file = File::create(out_path).map_err(|e| io_err(out_path, e))?;
    // Repacking a 26B artifact is a one-pass conversion, not a workload that
    // benefits from retaining either the source or destination expert pages.
    // Avoid filling unified memory immediately before the first Ghost-MoE run.
    crate::tensor::disable_file_cache_best_effort(&source);
    crate::tensor::disable_file_cache_best_effort(&file);
    file.write_all(CGHOST_MAGIC)
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&0u64.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    let mut cursor = (CGHOST_MAGIC.len() + 8) as u64;
    let mut groups = Vec::with_capacity(range.len() * expert_count);
    // A bounded scratch buffer keeps the conversion memory-independent of model size.
    let mut scratch = vec![0u8; 1024 * 1024];

    for layer_idx in range {
        let layer_moe = binding.layers[layer_idx].moe.as_ref().ok_or_else(|| {
            invalid(format!(
                "layer {layer_idx} has no routed-expert tensors; ghost MoE requires every selected layer to be MoE"
            ))
        })?;
        for desc in [&layer_moe.gate_up_exps, &layer_moe.down_exps] {
            if desc.dimensions.last().copied() != Some(expert_count as u64) {
                return Err(invalid(format!(
                    "tensor {} last dimension {:?} does not match expert_count {expert_count}",
                    desc.name,
                    desc.dimensions.last()
                )));
            }
            if !desc.n_bytes.is_multiple_of(expert_count as u64) {
                return Err(invalid(format!(
                    "tensor {} byte length {} is not divisible by expert_count {expert_count}",
                    desc.name, desc.n_bytes
                )));
            }
        }

        let gate_bytes = layer_moe.gate_up_exps.n_bytes / expert_count as u64;
        let down_bytes = layer_moe.down_exps.n_bytes / expert_count as u64;
        for expert_idx in 0..expert_count {
            let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
            if aligned > cursor {
                file.write_all(&vec![0u8; (aligned - cursor) as usize])
                    .map_err(|e| io_err(out_path, e))?;
                cursor = aligned;
            }
            let mut group = CghostGroup {
                id: format!("blk.{layer_idx}.exp.{expert_idx}"),
                tensors: Vec::with_capacity(2),
                source_sample_sha256: None,
            };
            for (role, desc, expert_len) in [
                ("gate_up_exps", &layer_moe.gate_up_exps, gate_bytes),
                ("down_exps", &layer_moe.down_exps, down_bytes),
            ] {
                let source_offset = desc.absolute_offset + expert_idx as u64 * expert_len;
                copy_range_chunked(
                    &source,
                    source_offset,
                    expert_len,
                    &mut file,
                    &mut scratch,
                    store.source_path(),
                    out_path,
                )?;
                let mut dims = desc.dimensions.clone();
                dims.pop();
                group.tensors.push(CghostTensor {
                    name: format!("{}#expert.{expert_idx}", desc.name),
                    role: role.to_string(),
                    dtype: desc.tensor_type,
                    dims,
                    offset: cursor,
                    len: expert_len,
                });
                cursor += expert_len;
            }
            group.source_sample_sha256 = Some(source_moe_group_fingerprint(
                &source,
                store.source_path(),
                binding,
                expert_count,
                layer_idx,
                expert_idx,
            )?);
            groups.push(group);
        }
    }

    let common_sha256 = common_source_fingerprint(&source, store.source_path(), binding)?;

    let index = CghostIndex {
        version: 2,
        layout: CghostLayout::MoeExperts,
        source_model: source_model.to_string(),
        block_count: total_layers,
        tied_output: binding.output_is_tied_embedding,
        expert_count: Some(expert_count),
        expert_used_count: Some(expert_used_count),
        source_identity: Some(CghostSourceIdentity {
            scheme: MOE_SOURCE_IDENTITY_SCHEME.to_string(),
            logical_bytes: source_len,
            sample_bytes: MOE_IDENTITY_SAMPLE_BYTES as u32,
            common_sha256,
        }),
        groups,
    };
    let index_json = serde_json::to_vec(&index)
        .map_err(|e| invalid(format!("failed to serialize .cghost index: {e}")))?;
    let index_offset = cursor;
    file.write_all(&index_json)
        .map_err(|e| io_err(out_path, e))?;
    file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&index_offset.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    file.sync_all().map_err(|e| io_err(out_path, e))?;
    Ok(index)
}

#[allow(clippy::too_many_arguments)]
fn copy_range_chunked(
    source: &File,
    source_offset: u64,
    len: u64,
    destination: &mut File,
    scratch: &mut [u8],
    source_path: &Path,
    destination_path: &Path,
) -> Result<()> {
    let mut copied = 0u64;
    while copied < len {
        let take = (len - copied).min(scratch.len() as u64) as usize;
        read_exact_at(source, &mut scratch[..take], source_offset + copied)
            .map_err(|e| io_err(source_path, e))?;
        destination
            .write_all(&scratch[..take])
            .map_err(|e| io_err(destination_path, e))?;
        copied += take as u64;
    }
    Ok(())
}

/// One selected routed expert read from a MoE `.cghost` file. Both projection
/// views share the same allocation, so a cache entry costs exactly the expert
/// record's wire bytes plus small metadata.
#[derive(Debug, Clone)]
pub struct GhostMoeExpert {
    bytes: Arc<[u8]>,
    pub gate_up: GhostMoeTensorView,
    pub down: GhostMoeTensorView,
}

/// Metadata-only description of the two projection views inside one contiguous
/// routed-expert record. Fixed host/device staging uses these role-aware ranges
/// instead of assuming the container stored gate/up before down.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GhostMoeExpertRecordLayout {
    pub(crate) byte_len: usize,
    pub(crate) gate_up: std::ops::Range<usize>,
    pub(crate) gate_up_dtype: GgufTensorType,
    pub(crate) down: std::ops::Range<usize>,
    pub(crate) down_dtype: GgufTensorType,
}

impl GhostMoeExpert {
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn tensor_backing(
        &self,
        view: &GhostMoeTensorView,
    ) -> (Arc<[u8]>, std::ops::Range<usize>) {
        (Arc::clone(&self.bytes), view.offset..view.offset + view.len)
    }

    /// Consumed by the CUDA lane's owned expert records; other targets read
    /// through [`Self::tensor_backing`].
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn tensor_bytes(&self, view: &GhostMoeTensorView) -> &[u8] {
        &self.bytes[view.offset..view.offset + view.len]
    }
}

/// A routed expert borrowed directly from the read-only `.cghost` mapping.
/// The mapping owns the file view; tensor slices allocate and copy nothing.
/// Consumed by the cuda expert-residency lane (and its unit tests); gated so
/// a default-features build — macOS CI's clippy shape — carries no dead code.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone)]
pub(crate) struct GhostMoeMappedExpert {
    mmap: Arc<GgufWireMmap>,
    record_start: u64,
    pub gate_up: GhostMoeTensorView,
    pub down: GhostMoeTensorView,
}

#[cfg(any(feature = "cuda", test))]
impl GhostMoeMappedExpert {
    pub(crate) fn tensor_bytes(&self, view: &GhostMoeTensorView) -> Result<&[u8]> {
        self.mmap
            .bytes(self.record_start + view.offset as u64, view.len)
    }
}

#[derive(Debug, Clone)]
pub struct GhostMoeTensorView {
    pub dtype: GgufTensorType,
    pub dims: Vec<u64>,
    offset: usize,
    len: usize,
}

/// Prevalidated coordinates for one routed-expert payload in [`CghostIndex::groups`].
///
/// A Gemma 4 26B artifact has thousands of expert groups. Looking up a selected expert by
/// formatting its name and scanning that whole vector put index JSON work on every cache miss.
/// This compact table is built once at open and turns `(layer, expert)` into the group and its
/// two tensor descriptors with direct array indexing. Optional tensor slots deliberately retain
/// malformed groups so the existing validation path can report the precise missing role.
#[derive(Debug, Clone, Copy)]
struct GhostMoeGroupAccess {
    group_idx: usize,
    gate_up_tensor_idx: Option<usize>,
    down_tensor_idx: Option<usize>,
}

#[derive(Debug)]
struct GhostMoeAccessIndex {
    expert_count: usize,
    groups: Vec<Option<GhostMoeGroupAccess>>,
}

impl GhostMoeAccessIndex {
    fn build(index: &CghostIndex) -> Result<Option<Self>> {
        if index.version != 2 || index.layout != CghostLayout::MoeExperts {
            return Ok(None);
        }
        let Some(expert_count) = index.expert_count else {
            // `validate_moe_layout` retains responsibility for reporting the model/index
            // mismatch. Without a declared stride there is no safe direct table to build.
            return Ok(None);
        };
        let slot_count = index
            .block_count
            .checked_mul(expert_count)
            .ok_or_else(|| invalid("ghost MoE group index dimensions overflow usize".into()))?;
        let mut groups = vec![None; slot_count];
        for (group_idx, group) in index.groups.iter().enumerate() {
            let Some((layer_idx, expert_idx)) = parse_moe_group_id(&group.id) else {
                continue;
            };
            if layer_idx >= index.block_count || expert_idx >= expert_count {
                continue;
            }
            let slot = layer_idx * expert_count + expert_idx;
            // Match the old linear `.find()` behavior for duplicate IDs: the first group wins.
            if groups[slot].is_some() {
                continue;
            }
            groups[slot] = Some(GhostMoeGroupAccess {
                group_idx,
                gate_up_tensor_idx: group
                    .tensors
                    .iter()
                    .position(|tensor| tensor.role == "gate_up_exps"),
                down_tensor_idx: group
                    .tensors
                    .iter()
                    .position(|tensor| tensor.role == "down_exps"),
            });
        }
        Ok(Some(Self {
            expert_count,
            groups,
        }))
    }

    fn get(&self, layer_idx: usize, expert_idx: usize) -> Option<GhostMoeGroupAccess> {
        if expert_idx >= self.expert_count {
            return None;
        }
        let slot = layer_idx
            .checked_mul(self.expert_count)?
            .checked_add(expert_idx)?;
        self.groups.get(slot).copied().flatten()
    }
}

/// Parse only the canonical writer spelling. In particular, `blk.00.exp.01` must not alias
/// `blk.0.exp.1`: the former was not found by the previous exact string lookup either.
fn parse_moe_group_id(id: &str) -> Option<(usize, usize)> {
    let rest = id.strip_prefix("blk.")?;
    let (layer, expert) = rest.split_once(".exp.")?;
    let layer_idx = layer.parse::<usize>().ok()?;
    let expert_idx = expert.parse::<usize>().ok()?;
    if layer != layer_idx.to_string() || expert != expert_idx.to_string() {
        return None;
    }
    Some((layer_idx, expert_idx))
}

/// Spans at or above this size take the explicit-read path when `bulk_reads` is set.
/// Well under one 3.19 MiB expert record and well above the few-hundred-byte identity
/// samples, so each lands on the path built for it.
const CGHOST_BULK_READ_MIN_BYTES: u64 = 256 * 1024;

/// Open `.cghost` reader: parses the index once, then serves page-aligned group reads.
pub struct GhostFile {
    pub index: CghostIndex,
    file: File,
    /// Raised around a cold bulk sweep (layer-major prefill), where records are read
    /// once and a map copy degenerates into ~816 demand faults per record.
    ///
    /// For DECODE the runtime decides per host: reuse makes the map copy a pure memcpy
    /// and the explicit read a measured 13% regression *when the page cache can serve*,
    /// but where free RAM cannot cover the routed payload those same misses fault cold
    /// and the map loses badly (measured 7.8 s -> 3.9 s of decode wall on the 26B-A4B row
    /// at 7.4 GiB available against a 12.4 GiB payload). See
    /// `Gemma4CudaResident::decode_bulk_reads` for the rule and
    /// `read_positioned_span_into` for both receipts.
    bulk_reads: AtomicBool,
    /// Set by `--ghost-strict-cache`: every read bypasses the page cache. Distinct from
    /// `bulk_reads`, which is a per-phase hint; both route to the same unbuffered handle.
    strict_cache: AtomicBool,
    /// Normal buffered mode maps the immutable payload so CUDA can upload a
    /// validated expert record without an intermediate Vec/read copy. Strict
    /// cache mode clears this mapping and retains positioned I/O semantics.
    mmap: Option<Arc<GgufWireMmap>>,
    /// Flat row-major `[layer][expert]` lookup for v2 MoE files. Dense/legacy files keep the
    /// original string lookup path and pay no extra allocation.
    moe_access: Option<GhostMoeAccessIndex>,
    /// Payload samples are checked lazily on first use, then never rehashed on
    /// the token-critical path. Atomic flags keep concurrent prefetch/direct
    /// readers race-safe; duplicate first checks are harmless.
    moe_payload_verified: Vec<AtomicBool>,
    /// Windows strict-ceiling mode: a second handle to the same file opened with
    /// `FILE_FLAG_NO_BUFFERING`, so streamed group reads bypass the OS page cache and hit
    /// the device — the equivalent of macOS `F_NOCACHE`, which Windows cannot toggle on an
    /// already-open handle. `None` unless `--evict-page-cache` requested it and the open
    /// succeeded; the read path falls back to the buffered `file` handle otherwise. Behind a
    /// `Mutex` because `GhostFile` is shared across the prefetch worker via `Arc` (needs
    /// `Sync`) and the aligned scratch buffer is interior-mutable.
    ///
    /// A POOL, not one reader. Each entry owns its own handle AND its own aligned scratch,
    /// so N record reads can be in flight at once. The previous single-`Mutex` form made the
    /// unbuffered path serial by construction: measured on this box with concurrent
    /// record-sized (3.19 MiB) reads, the drive delivers 1.04 GB/s at depth 1 but 2.42 at
    /// depth 4 and 3.03 at depth 28 — a 2.9x span that a single shared scratch cannot reach.
    /// The engine was getting exactly the depth-1 rate (1.13 GB/s profiled), and raising
    /// `CAMELID_GEMMA4_GHOST_READ_THREADS` made it WORSE (0.94 GB/s) because the extra rayon
    /// threads only contended on this lock. Pool size is
    /// `CAMELID_GEMMA4_GHOST_UNCACHED_READERS` (default 1 = the historical behaviour, so this
    /// is inert until measured); each extra reader costs one handle plus one aligned scratch
    /// of the largest group span.
    ///
    /// TWO THINGS MEASURED AND CLOSED (2026-08-24), so nobody re-derives them:
    /// 1. **The pool alone changes nothing**, because the caller is serial. The CUDA miss loop
    ///    reads each record "serially, with the GPU idle behind the fence"
    ///    (`gemma4_runtime.rs`, tier-0 hot path), so no two reads are ever outstanding and the
    ///    pool has nothing to overlap. Paired runs: 1.12 / 1.11 / 1.13 GB/s at 1 / 8 / 8
    ///    readers, token IDs identical.
    /// 2. **Splitting ONE record across the pool is a REGRESSION**, monotone in the split
    ///    factor: 1.41 GB/s unsplit -> 1.02-1.12 at 4 -> 0.88 at 8, prefill 14.84s -> 18.04s
    ///    (bit-exact throughout). The 2.9x depth win comes from INDEPENDENT records at
    ///    scattered offsets being in flight; chopping one contiguous 3.19 MiB span adds
    ///    per-request and fan-out cost with no seek to overlap. Implementation removed.
    ///
    /// The matching scalar-caller transaction now exists behind
    /// `CAMELID_GEMMA4_GHOST_BATCH_READS`: it reserves distinct page-locked tier slots and
    /// issues one whole-record read per miss. On the H40 scalar route it was exact but did
    /// not win (15.72 tok/s, or 15.82 with resident-hit overlap, against the adjacent 16.13
    /// control) because 899 decode reads over 48 tokens and 30 layers expose fewer than one
    /// disk miss per layer on average. The pool therefore stays at 1 by default. Its real
    /// consumer is the K-wide verifier, which can expose as many as 112 distinct assignments
    /// for a layer before issuing I/O; do not split individual records to manufacture depth.
    #[cfg(windows)]
    uncached: Vec<std::sync::Mutex<UncachedReader>>,
    /// Round-robin start index for pool selection. Relaxed: a stale value only changes which
    /// reader is tried first, never correctness.
    #[cfg(windows)]
    uncached_cursor: std::sync::atomic::AtomicUsize,
    /// Logical sector size of the volume holding `.cghost`, cached so the split-read planner
    /// can align chunk boundaries without taking a reader lock just to read it. Every pool
    /// entry opens the same file, so they all report the same value.
    #[cfg(windows)]
    uncached_sector: u64,
}

impl GhostFile {
    /// Open with optional strict-ceiling mode: `evict_page_cache` sets `F_NOCACHE` on the
    /// handle (macOS) so streamed reads bypass the page cache entirely. For models that fit
    /// in RAM the cache is a free win (leave this off); for the over-RAM models ghost mode
    /// targets, the cache can only thrash and the OS must not accumulate the file's pages.
    /// (`posix_madvise(DONTNEED)` does not apply here â€” that is for mmap'd ranges, and the
    /// streamer uses positioned reads.)
    /// (Windows: `evict_page_cache` opens a second `FILE_FLAG_NO_BUFFERING` handle so streamed
    /// group reads bypass the page cache and hit the device -- Windows cannot toggle
    /// no-buffering on an already-open handle, and the macOS `F_NOCACHE` fcntl is a no-op here.
    /// This makes cold-disk streaming cost MEASURABLE on a box where the model would otherwise
    /// cache in RAM. Falls back to buffered reads if the unbuffered open fails.)
    #[cfg_attr(not(windows), allow(unused_mut))]
    pub fn open_with_options(path: &Path, evict_page_cache: bool) -> Result<Self> {
        let mut this = Self::open(path)?;
        if evict_page_cache {
            this.mmap = None;
            this.strict_cache.store(true, Ordering::Relaxed);
            crate::tensor::disable_file_cache_best_effort(&this.file);
        }
        // Open the unbuffered handle regardless. Strict mode uses it for EVERY read;
        // normal mode uses it only for the cold bulk sweep (`bulk_reads`), where the
        // page cache has nothing to offer and this drive delivers ~1.9 GB/s unbuffered
        // against ~1.3 GB/s buffered. It costs one handle plus a sector-aligned scratch
        // buffer of one expert record.
        #[cfg(windows)]
        if this.uncached.is_empty() {
            // Default 1 reader = byte-for-byte the historical path. >1 opens independent
            // handles + scratches so a layer's expert-union records can be read concurrently;
            // see the field doc for the depth-vs-bandwidth receipts.
            let requested = std::env::var("CAMELID_GEMMA4_GHOST_UNCACHED_READERS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 32);
            let span = this.max_layer_span();
            for i in 0..requested {
                match UncachedReader::open(path, span) {
                    Ok(reader) => this.uncached.push(std::sync::Mutex::new(reader)),
                    Err(e) => {
                        // A partial pool is still correct — selection walks whatever exists,
                        // and an empty pool falls through to the buffered/mapped path.
                        eprintln!(
                            "[ghost] unbuffered (FILE_FLAG_NO_BUFFERING) open failed for reader \
                             {i} of {requested} ({e}); continuing with {} reader(s)",
                            this.uncached.len()
                        );
                        break;
                    }
                }
            }
            this.uncached_sector = this
                .uncached
                .first()
                .map(|r| r.lock().unwrap_or_else(|p| p.into_inner()).sector as u64)
                .unwrap_or(0);
            if this.uncached.len() > 1 {
                eprintln!(
                    "[ghost] unbuffered reader pool: {} readers (scratch {:.2} MiB each, sector {} B)",
                    this.uncached.len(),
                    span as f64 / (1024.0 * 1024.0),
                    this.uncached_sector
                );
            }
        }
        Ok(this)
    }

    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| io_err(path, e))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(|e| io_err(path, e))?;
        if &magic != CGHOST_MAGIC {
            return Err(invalid(format!(
                "{} is not a .cghost file (bad magic)",
                path.display()
            )));
        }
        let mut off = [0u8; 8];
        file.read_exact(&mut off).map_err(|e| io_err(path, e))?;
        let index_offset = u64::from_le_bytes(off);
        file.seek(SeekFrom::Start(index_offset))
            .map_err(|e| io_err(path, e))?;
        let mut index_json = Vec::new();
        file.read_to_end(&mut index_json)
            .map_err(|e| io_err(path, e))?;
        let index: CghostIndex = serde_json::from_slice(&index_json)
            .map_err(|e| invalid(format!("failed to parse .cghost index: {e}")))?;
        if !matches!(index.version, 1 | 2) {
            return Err(invalid(format!(
                "unsupported .cghost version {}",
                index.version
            )));
        }
        let moe_access = GhostMoeAccessIndex::build(&index)?;
        let moe_payload_verified = (0..index.groups.len())
            .map(|_| AtomicBool::new(false))
            .collect();
        Ok(Self {
            index,
            file,
            bulk_reads: AtomicBool::new(false),
            strict_cache: AtomicBool::new(false),
            mmap: Some(GgufWireMmap::map(path)?),
            moe_access,
            moe_payload_verified,
            #[cfg(windows)]
            uncached: Vec::new(),
            #[cfg(windows)]
            uncached_cursor: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(windows)]
            uncached_sector: 0,
        })
    }

    /// Select the read path for the phase that is about to run: `true` for a cold bulk
    /// sweep (prefill), `false` for reuse-dominated access (decode). Cheap and
    /// idempotent; the two paths return identical bytes.
    /// Size of the routed payload on disk, or 0 if it cannot be probed. Used to decide
    /// whether the page cache can plausibly serve decode misses; a stat per decision is
    /// nothing next to the reads the decision governs.
    #[cfg(feature = "cuda")]
    pub(crate) fn payload_bytes(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn set_bulk_reads(&self, on: bool) {
        self.bulk_reads.store(on, Ordering::Relaxed);
    }

    /// Validate the structural identity required by the Gemma 4 paged-expert
    /// runtime before any generation begins.
    pub fn validate_moe_layout(
        &self,
        block_count: usize,
        expert_count: usize,
        expert_used_count: usize,
    ) -> Result<()> {
        if self.index.layout != CghostLayout::MoeExperts || self.index.version != 2 {
            return Err(invalid(format!(
                "ghost MoE requires a v2 moe_experts .cghost; found v{} {:?}",
                self.index.version, self.index.layout
            )));
        }
        if self.index.block_count != block_count
            || self.index.expert_count != Some(expert_count)
            || self.index.expert_used_count != Some(expert_used_count)
        {
            return Err(invalid(format!(
                "ghost MoE index mismatch: file has blocks={}, experts={:?}, used={:?}; model requires blocks={block_count}, experts={expert_count}, used={expert_used_count}",
                self.index.block_count, self.index.expert_count, self.index.expert_used_count
            )));
        }
        for layer in 0..block_count {
            for expert in 0..expert_count {
                let (group, access) = self.moe_group_access(layer, expert)?;
                for role in ["gate_up_exps", "down_exps"] {
                    self.moe_group_tensor(group, access, role)?;
                }
            }
        }
        Ok(())
    }

    /// Bind every stored expert slice back to the GGUF descriptors it was
    /// derived from. Counts alone cannot catch a same-named but differently
    /// quantized checkpoint; dtype, per-expert dimensions, and byte lengths
    /// must all match before the runtime is allowed to consume a record.
    pub fn validate_moe_binding(&self, binding: &Gemma4Binding, expert_count: usize) -> Result<()> {
        if self.index.tied_output != binding.output_is_tied_embedding {
            return Err(invalid(format!(
                "ghost MoE tied-output flag {} does not match model {}",
                self.index.tied_output, binding.output_is_tied_embedding
            )));
        }
        for (layer_idx, layer) in binding.layers.iter().enumerate() {
            let moe = layer.moe.as_ref().ok_or_else(|| {
                invalid(format!(
                    "model layer {layer_idx} has no routed-expert binding"
                ))
            })?;
            for expert_idx in 0..expert_count {
                let (group, access) = self.moe_group_access(layer_idx, expert_idx)?;
                for (role, expected) in [
                    ("gate_up_exps", &moe.gate_up_exps),
                    ("down_exps", &moe.down_exps),
                ] {
                    let actual = self.moe_group_tensor(group, access, role)?;
                    let mut expected_dims = expected.dimensions.clone();
                    if expected_dims.pop() != Some(expert_count as u64) {
                        return Err(invalid(format!(
                            "model tensor {} does not end in expert_count {expert_count}",
                            expected.name
                        )));
                    }
                    if !expected.n_bytes.is_multiple_of(expert_count as u64) {
                        return Err(invalid(format!(
                            "model tensor {} byte length {} is not divisible by expert_count {expert_count}",
                            expected.name, expected.n_bytes
                        )));
                    }
                    let expected_len = expected.n_bytes / expert_count as u64;
                    if actual.dtype != expected.tensor_type
                        || actual.dims != expected_dims
                        || actual.len != expected_len
                    {
                        return Err(invalid(format!(
                            "group {} role {role} does not match model tensor {}: file dtype={:?} dims={:?} len={}, model dtype={:?} dims={:?} len={expected_len}",
                            group.id,
                            expected.name,
                            actual.dtype,
                            actual.dims,
                            actual.len,
                            expected.tensor_type,
                            expected_dims
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Prove that this `.cghost` was repacked from the supplied GGUF (or from
    /// the full GGUF underlying a sparse hot shadow) without hashing the whole
    /// model at startup. New artifacts sample every expert record and every
    /// bound common tensor. Legacy v2 indexes remain usable with a full GGUF,
    /// but are rejected for a recognized sparse hot shadow unless the operator
    /// deliberately enables the unsafe compatibility escape hatch.
    pub fn validate_moe_source_identity(
        &self,
        source_path: &Path,
        binding: &Gemma4Binding,
        expert_count: usize,
    ) -> Result<()> {
        let Some(identity) = self.index.source_identity.as_ref() else {
            if source_is_sparse(source_path)? && !allow_legacy_sparse_identity() {
                return Err(invalid(format!(
                    "legacy Ghost-MoE artifact has no strong sampled source identity and cannot be paired safely with sparse GGUF {}; repack the .cghost and hot shadow together, or set {ALLOW_LEGACY_SPARSE_IDENTITY_ENV}=1 only to accept shape-only pairing risk",
                    source_path.display()
                )));
            }
            eprintln!(
                "[ghost] WARNING: the legacy v2 .cghost paired with {} has no sampled source identity; source↔cghost pairing is shape-only.",
                source_path.display()
            );
            return Ok(());
        };
        if identity.scheme != MOE_SOURCE_IDENTITY_SCHEME
            || identity.sample_bytes != MOE_IDENTITY_SAMPLE_BYTES as u32
        {
            return Err(invalid(format!(
                "unsupported Ghost-MoE source identity scheme {:?} with sample_bytes={}; expected {:?} with sample_bytes={MOE_IDENTITY_SAMPLE_BYTES}",
                identity.scheme, identity.sample_bytes, MOE_SOURCE_IDENTITY_SCHEME
            )));
        }
        if !valid_sha256_hex(&identity.common_sha256) {
            return Err(invalid(
                "Ghost-MoE source identity contains a malformed common SHA-256".into(),
            ));
        }
        let source = File::open(source_path).map_err(|err| io_err(source_path, err))?;
        let actual_len = source
            .metadata()
            .map_err(|err| io_err(source_path, err))?
            .len();
        if actual_len != identity.logical_bytes {
            return Err(invalid(format!(
                "Ghost-MoE source identity length mismatch: .cghost expects {} bytes, {} has {actual_len} bytes",
                identity.logical_bytes,
                source_path.display()
            )));
        }
        let actual_common = common_source_fingerprint(&source, source_path, binding)?;
        if actual_common != identity.common_sha256 {
            return Err(invalid(format!(
                "Ghost-MoE common-core identity mismatch: {} was not the GGUF used to create this .cghost. Select the matching pair or repack both artifacts.",
                source_path.display()
            )));
        }
        for layer_idx in 0..binding.layers.len() {
            for expert_idx in 0..expert_count {
                let (group, _) = self.moe_group_access(layer_idx, expert_idx)?;
                let expected = group.source_sample_sha256.as_deref().ok_or_else(|| {
                    invalid(format!(
                        "Ghost-MoE source identity is incomplete: group {} has no sampled SHA-256; repack the .cghost",
                        group.id
                    ))
                })?;
                if !valid_sha256_hex(expected) {
                    return Err(invalid(format!(
                        "Ghost-MoE source identity for group {} is not a valid lowercase SHA-256",
                        group.id
                    )));
                }
                let actual = source_moe_group_fingerprint(
                    &source,
                    source_path,
                    binding,
                    expert_count,
                    layer_idx,
                    expert_idx,
                )?;
                if actual != expected {
                    return Err(invalid(format!(
                        "Ghost-MoE expert identity mismatch at {}: the source GGUF/hot shadow and .cghost are not a matching pair. If this is a sparse shadow created by an older Camelid, regenerate it to retain identity islands.",
                        group.id
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_moe_expert_payload_identity(
        &self,
        group: &CghostGroup,
        access: Option<GhostMoeGroupAccess>,
        record_start: u64,
        payload: &[u8],
    ) -> Result<()> {
        if let Some(access) = access {
            if self
                .moe_payload_verified
                .get(access.group_idx)
                .is_some_and(|verified| verified.load(Ordering::Acquire))
            {
                return Ok(());
            }
        }
        let Some(expected) = group.source_sample_sha256.as_deref() else {
            if self.index.source_identity.is_some() {
                return Err(invalid(format!(
                    "Ghost-MoE source identity is incomplete: group {} has no sampled SHA-256",
                    group.id
                )));
            }
            return Ok(());
        };
        if !valid_sha256_hex(expected) {
            return Err(invalid(format!(
                "Ghost-MoE source identity for group {} is not a valid lowercase SHA-256",
                group.id
            )));
        }
        let (layer_idx, expert_idx) = parse_moe_group_id(&group.id).ok_or_else(|| {
            invalid(format!(
                "Ghost-MoE group {} has a non-canonical identity key",
                group.id
            ))
        })?;
        let mut hasher = begin_moe_group_fingerprint(layer_idx, expert_idx);
        for role in ["gate_up_exps", "down_exps"] {
            let tensor = self.moe_group_tensor(group, access, role)?;
            let sample_range =
                moe_identity_sample_range(0, tensor.len, layer_idx, expert_idx, role);
            let tensor_in_record = tensor.offset.checked_sub(record_start).ok_or_else(|| {
                invalid(format!(
                    "group {} tensor {} begins before its record",
                    group.id, tensor.name
                ))
            })?;
            let sample_start = tensor_in_record
                .checked_add(sample_range.start)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| {
                    invalid(format!(
                        "group {} sampled tensor offset is not representable",
                        group.id
                    ))
                })?;
            let sample_end = sample_start
                .checked_add((sample_range.end - sample_range.start) as usize)
                .ok_or_else(|| invalid(format!("group {} sample range overflows", group.id)))?;
            let sample = payload.get(sample_start..sample_end).ok_or_else(|| {
                invalid(format!(
                    "group {} sample range {sample_start}..{sample_end} exceeds record length {}",
                    group.id,
                    payload.len()
                ))
            })?;
            update_moe_group_sample(
                &mut hasher,
                role,
                tensor.dtype,
                &tensor.dims,
                tensor.len,
                sample_range.start,
                sample,
            );
        }
        let actual = sha256_hex(hasher);
        if actual != expected {
            return Err(invalid(format!(
                "Ghost-MoE payload identity mismatch in {}: the .cghost expert record is corrupt or belongs to another source",
                group.id
            )));
        }
        if let Some(access) = access {
            if let Some(verified) = self.moe_payload_verified.get(access.group_idx) {
                verified.store(true, Ordering::Release);
            }
        }
        Ok(())
    }

    /// Borrow one routed expert directly from the read-only file mapping. The
    /// same payload-identity check as the positioned reader runs before bytes
    /// are exposed. Strict-cache mode returns `None` because it disables mmap.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn mapped_moe_expert(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<Option<GhostMoeMappedExpert>> {
        let Some(mmap) = self.mmap.as_ref() else {
            return Ok(None);
        };
        let (group, start, len) = self.checked_moe_expert_span(layer_idx, expert_idx)?;
        let (_, access) = self.moe_group_access(layer_idx, expert_idx)?;
        let payload = mmap.bytes(start, len)?;
        self.validate_moe_expert_payload_identity(group, access, start, payload)?;
        let view = |role: &str| -> Result<GhostMoeTensorView> {
            let tensor = self.moe_group_tensor(group, access, role)?;
            Ok(GhostMoeTensorView {
                dtype: tensor.dtype,
                dims: tensor.dims.clone(),
                offset: (tensor.offset - start) as usize,
                len: tensor.len as usize,
            })
        };
        Ok(Some(GhostMoeMappedExpert {
            mmap: Arc::clone(mmap),
            record_start: start,
            gate_up: view("gate_up_exps")?,
            down: view("down_exps")?,
        }))
    }

    /// Read one routed expert with one sequential positioned read. The file's
    /// fixed group layout makes both projection slices views of the same bounded
    /// allocation, ready for insertion into the global expert cache.
    pub fn read_moe_expert(&self, layer_idx: usize, expert_idx: usize) -> Result<GhostMoeExpert> {
        let (group, access) = self.moe_group_access(layer_idx, expert_idx)?;
        let mut payload = Vec::new();
        let start = self.read_group_payload_from(group, &mut payload)?;
        self.validate_moe_expert_payload_identity(group, access, start, &payload)?;
        let view = |role: &str| -> Result<GhostMoeTensorView> {
            let tensor = self.moe_group_tensor(group, access, role)?;
            Ok(GhostMoeTensorView {
                dtype: tensor.dtype,
                dims: tensor.dims.clone(),
                offset: (tensor.offset - start) as usize,
                len: tensor.len as usize,
            })
        };
        let gate_up = view("gate_up_exps")?;
        let down = view("down_exps")?;
        Ok(GhostMoeExpert {
            bytes: payload.into(),
            gate_up,
            down,
        })
    }

    /// Return the exact wire length of one routed-expert record.
    ///
    /// This resolves `(layer_idx, expert_idx)` through the same validated direct table used by
    /// [`Self::read_moe_expert`] and checks that the group's tensor ranges form one contiguous,
    /// representable record. Callers that own fixed storage (for example, a page-aligned Metal
    /// expert slot) can use this to size that storage before calling
    /// [`Self::read_moe_expert_into`].
    pub fn moe_expert_byte_len(&self, layer_idx: usize, expert_idx: usize) -> Result<usize> {
        let (_, _, len) = self.checked_moe_expert_span(layer_idx, expert_idx)?;
        Ok(len)
    }

    /// Resolve projection roles and byte ranges without reading the record.
    /// Generic `.cghost` files may store the two tensors in either contiguous
    /// order, so callers that split a caller-owned record must use this metadata
    /// rather than a hard-coded offset.
    #[cfg(feature = "cuda")]
    /// Ask the OS to begin faulting in one MoE layer's entire routed-expert span.
    ///
    /// The v2 layout stores experts layer-contiguous (`blk.L.exp.0 … blk.L.exp.E-1`),
    /// so a whole layer is ONE contiguous byte range — 408 MiB on the tracked 26B row.
    ///
    /// Prefetching the WHOLE layer rather than the routed subset is what makes this
    /// possible at all: layer L+1's routing is unknown until layer L has produced its
    /// output, but the layer's byte range is knowable immediately. It is also nearly
    /// free here — layer-major prefill was measured touching 121.1 of 128 experts per
    /// layer, so covering all 128 costs ~5% more bytes while turning ~121 scattered
    /// 3.19 MiB reads into one 408 MiB sequential read, which is the access pattern
    /// this NVMe is actually good at.
    ///
    /// Best-effort and asynchronous: `PrefetchVirtualMemory` queues the transfer and
    /// returns immediately, so there is no thread, no extra VRAM, and the on-demand
    /// mmap path underneath is untouched. Every failure is ignored — the pages simply
    /// fault in on first touch exactly as before.
    ///
    /// Prefill only. Do NOT call this per decoded token: a whole layer per token would
    /// be 408 MiB/token against the ~98 MiB the routed subset actually needs.
    pub(crate) fn prefetch_moe_layer(&self, layer_idx: usize, expert_count: usize) {
        if expert_count == 0 {
            return;
        }
        let Some(mmap) = self.mmap.as_ref() else {
            return;
        };
        let Ok((_, first_start, _)) = self.checked_moe_expert_span(layer_idx, 0) else {
            return;
        };
        let Ok((_, last_start, last_len)) =
            self.checked_moe_expert_span(layer_idx, expert_count - 1)
        else {
            return;
        };
        let Some(len) = last_start
            .checked_add(last_len as u64)
            .and_then(|end| end.checked_sub(first_start))
            .and_then(|len| usize::try_from(len).ok())
        else {
            return;
        };
        let Ok(range) = mmap.bytes(first_start, len) else {
            return;
        };
        prefetch_range_best_effort(range);
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn moe_expert_record_layout(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<GhostMoeExpertRecordLayout> {
        let (group, start, byte_len) = self.checked_moe_expert_span(layer_idx, expert_idx)?;
        let (_, access) = self.moe_group_access(layer_idx, expert_idx)?;
        let gate_up = self.moe_group_tensor(group, access, "gate_up_exps")?;
        let down = self.moe_group_tensor(group, access, "down_exps")?;
        let relative_range = |tensor: &CghostTensor| -> Result<std::ops::Range<usize>> {
            let begin = tensor
                .offset
                .checked_sub(start)
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| {
                    invalid(format!(
                        "group {} tensor {} begins before or outside its record",
                        group.id, tensor.name
                    ))
                })?;
            let len = usize::try_from(tensor.len).map_err(|_| {
                invalid(format!(
                    "group {} tensor {} length is not representable on this host",
                    group.id, tensor.name
                ))
            })?;
            let end = begin.checked_add(len).ok_or_else(|| {
                invalid(format!(
                    "group {} tensor {} relative range overflows usize",
                    group.id, tensor.name
                ))
            })?;
            if end > byte_len {
                return Err(invalid(format!(
                    "group {} tensor {} range {begin}..{end} exceeds record length {byte_len}",
                    group.id, tensor.name
                )));
            }
            Ok(begin..end)
        };
        Ok(GhostMoeExpertRecordLayout {
            byte_len,
            gate_up: relative_range(gate_up)?,
            gate_up_dtype: gate_up.dtype,
            down: relative_range(down)?,
            down_dtype: down.dtype,
        })
    }

    /// Validate the fixed record layout consumed by persistent routed-expert
    /// accelerators.
    ///
    /// The allocating Ghost reader follows each tensor's indexed view and can
    /// therefore consume any contiguous tensor order. Fixed-offset kernels do
    /// not have that freedom: they require exactly `[gate_up_exps][down_exps]`
    /// with no gap or extra tensor. Keep this stricter contract separate from
    /// [`Self::moe_expert_byte_len`] so generic readers retain their established
    /// behavior while fixed-layout consumers fail closed before touching bytes.
    pub fn validate_moe_expert_record_layout(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<usize> {
        let (group, _) = self.moe_group_access(layer_idx, expert_idx)?;
        if group.tensors.len() != 2 {
            return Err(invalid(format!(
                "group {} must contain exactly two tensors in canonical routed-expert order; found {}",
                group.id,
                group.tensors.len()
            )));
        }

        let gate_up = &group.tensors[0];
        let down = &group.tensors[1];
        if gate_up.role != "gate_up_exps" || down.role != "down_exps" {
            return Err(invalid(format!(
                "group {} has non-canonical routed-expert tensor order {:?}; expected [\"gate_up_exps\", \"down_exps\"]",
                group.id,
                group
                    .tensors
                    .iter()
                    .map(|tensor| tensor.role.as_str())
                    .collect::<Vec<_>>()
            )));
        }

        let record_start = gate_up.offset;
        let gate_up_end = gate_up.offset.checked_add(gate_up.len).ok_or_else(|| {
            invalid(format!(
                "group {} gate_up_exps byte range overflows u64",
                group.id
            ))
        })?;
        if down.offset != gate_up_end {
            return Err(invalid(format!(
                "group {} is not a canonical contiguous expert record: down_exps starts at {}, expected {gate_up_end}",
                group.id, down.offset
            )));
        }
        let record_end = down.offset.checked_add(down.len).ok_or_else(|| {
            invalid(format!(
                "group {} down_exps byte range overflows u64",
                group.id
            ))
        })?;
        let record_len = record_end
            .checked_sub(record_start)
            .and_then(|len| usize::try_from(len).ok())
            .ok_or_else(|| {
                invalid(format!(
                    "group {} canonical expert record length is not representable on this host",
                    group.id
                ))
            })?;
        Ok(record_len)
    }

    /// Validate every record in a fixed `[layer][expert]` layout and require one
    /// exact byte geometry. This is the load-time admission gate for kernels
    /// whose offsets are compiled constants.
    pub fn validate_moe_expert_record_layouts(
        &self,
        block_count: usize,
        expert_count: usize,
        expected_record_bytes: usize,
    ) -> Result<()> {
        for layer_idx in 0..block_count {
            for expert_idx in 0..expert_count {
                let actual = self.validate_moe_expert_record_layout(layer_idx, expert_idx)?;
                if actual != expected_record_bytes {
                    return Err(invalid(format!(
                        "group blk.{layer_idx}.exp.{expert_idx} canonical record length {actual} does not match fixed kernel geometry {expected_record_bytes}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Read one routed-expert record directly into caller-owned storage.
    ///
    /// `destination` must be exactly [`Self::moe_expert_byte_len`] bytes: undersized and
    /// oversized slices are rejected before any I/O. The positioned read writes into the
    /// supplied memory without first allocating a `Vec`/`Arc` or copying from an application
    /// scratch buffer. Strict-cache reads retain the same platform-specific behavior as the
    /// existing group reader (including the Windows unbuffered-handle path).
    pub fn read_moe_expert_into(
        &self,
        layer_idx: usize,
        expert_idx: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        let (group, start, expected_len) = self.checked_moe_expert_span(layer_idx, expert_idx)?;
        if destination.len() != expected_len {
            return Err(invalid(format!(
                "group {} expert destination length {} does not match record length {expected_len}",
                group.id,
                destination.len()
            )));
        }
        self.read_positioned_span_into(start, expected_len as u64, destination)?;
        let (_, access) = self.moe_group_access(layer_idx, expert_idx)?;
        self.validate_moe_expert_payload_identity(group, access, start, destination)
    }

    /// Resolve and validate the contiguous byte span consumed by the direct expert reader.
    fn checked_moe_expert_span(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<(&CghostGroup, u64, usize)> {
        let (group, access) = self.moe_group_access(layer_idx, expert_idx)?;
        // Validate the two required roles before touching caller memory. This also preserves the
        // precise malformed-index diagnostics provided by the allocating expert reader.
        self.moe_group_tensor(group, access, "gate_up_exps")?;
        self.moe_group_tensor(group, access, "down_exps")?;

        let first = group
            .tensors
            .first()
            .ok_or_else(|| invalid(format!("group {} has no tensors", group.id)))?;
        let start = first.offset;
        let mut cursor = start;
        for tensor in &group.tensors {
            if tensor.offset != cursor {
                return Err(invalid(format!(
                    "group {} is not a contiguous expert record: tensor {} starts at {}, expected {cursor}",
                    group.id, tensor.name, tensor.offset
                )));
            }
            cursor = tensor.offset.checked_add(tensor.len).ok_or_else(|| {
                invalid(format!(
                    "group {} tensor {} byte range overflows u64",
                    group.id, tensor.name
                ))
            })?;
        }
        let len = cursor
            .checked_sub(start)
            .and_then(|len| usize::try_from(len).ok())
            .ok_or_else(|| {
                invalid(format!(
                    "group {} expert record length is not representable on this host",
                    group.id
                ))
            })?;
        Ok((group, start, len))
    }

    /// Resolve a v2 MoE group through the direct table. The string-scan fallback exists only
    /// for malformed v2 metadata without `expert_count`, preserving the previous error surface
    /// until `validate_moe_layout` reports the mismatch.
    fn moe_group_access(
        &self,
        layer_idx: usize,
        expert_idx: usize,
    ) -> Result<(&CghostGroup, Option<GhostMoeGroupAccess>)> {
        if let Some(index) = &self.moe_access {
            let access = index.get(layer_idx, expert_idx).ok_or_else(|| {
                invalid(format!(
                    ".cghost file has no \"blk.{layer_idx}.exp.{expert_idx}\" group"
                ))
            })?;
            let group = self.index.groups.get(access.group_idx).ok_or_else(|| {
                invalid("ghost MoE direct group index is out of bounds".to_string())
            })?;
            return Ok((group, Some(access)));
        }
        let id = format!("blk.{layer_idx}.exp.{expert_idx}");
        Ok((self.group(&id)?, None))
    }

    fn moe_group_tensor<'a>(
        &self,
        group: &'a CghostGroup,
        access: Option<GhostMoeGroupAccess>,
        role: &str,
    ) -> Result<&'a CghostTensor> {
        let indexed = access.and_then(|access| match role {
            "gate_up_exps" => access.gate_up_tensor_idx,
            "down_exps" => access.down_tensor_idx,
            _ => None,
        });
        let tensor = if access.is_some() {
            indexed.and_then(|tensor_idx| group.tensors.get(tensor_idx))
        } else {
            group.tensors.iter().find(|tensor| tensor.role == role)
        };
        tensor.ok_or_else(|| invalid(format!("group {} is missing role \"{role}\"", group.id)))
    }

    fn group(&self, id: &str) -> Result<&CghostGroup> {
        self.index
            .groups
            .iter()
            .find(|g| g.id == id)
            .ok_or_else(|| invalid(format!(".cghost file has no \"{id}\" group")))
    }

    /// Read a whole group's payload with ONE sequential read into `buf` (reused across
    /// calls). Returns the group and the span start so tensor slices can be located.
    fn read_group_payload<'a>(
        &'a self,
        id: &str,
        buf: &mut Vec<u8>,
    ) -> Result<(&'a CghostGroup, u64)> {
        let group = self.group(id)?;
        let start = self.read_group_payload_from(group, buf)?;
        Ok((group, start))
    }

    /// Read a previously resolved group without performing another name lookup.
    fn read_group_payload_from(&self, group: &CghostGroup, buf: &mut Vec<u8>) -> Result<u64> {
        let (start, len) = group.span();
        buf.resize(len as usize, 0);
        self.read_positioned_span_into(start, len, buf)?;
        Ok(start)
    }

    /// Fill an already-sized byte span using the platform's positioned reader.
    fn read_positioned_span_into(&self, start: u64, len: u64, buf: &mut [u8]) -> Result<()> {
        if u64::try_from(buf.len()).ok() != Some(len) {
            return Err(invalid(format!(
                "positioned .cghost destination length {} does not match span length {len}",
                buf.len()
            )));
        }
        // Strict-ceiling mode (Windows): serve the read from the unbuffered handle so it
        // bypasses the page cache. read_into returns false when it declines (a group that is
        // not sector-aligned or whose aligned span would pass EOF -- never for a v1 .cghost's
        // 16 KiB-aligned blk groups), in which case we fall through to the buffered read.
        // Unbuffered path. Strict mode takes it for every read; normal mode only for the
        // cold bulk sweep, where the page cache has nothing to offer. `read_into` returns
        // false when it declines (span not sector-aligned, or the aligned span would pass
        // EOF), in which case we fall through.
        #[cfg(windows)]
        if !self.uncached.is_empty()
            && (self.strict_cache.load(Ordering::Relaxed)
                || (self.bulk_reads.load(Ordering::Relaxed) && len >= CGHOST_BULK_READ_MIN_BYTES))
        {
            // Pick a free reader so concurrent callers overlap on the device instead of
            // queueing on one scratch. With a single-entry pool this is exactly the old
            // path: one try_lock that always succeeds when uncontended.
            let n = self.uncached.len();
            let first = self.uncached_cursor.fetch_add(1, Ordering::Relaxed) % n;
            let mut picked = None;
            for k in 0..n {
                if let Ok(guard) = self.uncached[(first + k) % n].try_lock() {
                    picked = Some(guard);
                    break;
                }
            }
            let mut guard = match picked {
                Some(guard) => guard,
                // Every reader busy (or poisoned): block on the round-robin pick rather
                // than spin. Poison is recovered — the scratch is overwritten before use.
                None => self.uncached[first]
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()),
            };
            let served = guard
                .read_into(start, len, &mut buf[..])
                .map_err(|e| io_err(Path::new("<cghost>"), e))?;
            if served {
                return Ok(());
            }
        }
        // WHICH PATH IS FASTER DEPENDS ENTIRELY ON THE REGIME, and both regimes were
        // measured on this box:
        //
        // DECODE, WARM HOST (free RAM covers the payload): the map wins. Routing
        // record-sized spans to an explicit `read_exact_at` measured a 13.0% REGRESSION
        // (paired ABABAB, 3/3 pairs negative, token-ID and hit/miss identity PASS;
        // qa/perf/receipts/paired-mmap-vs-pread.json). With ~52% of decode misses being
        // re-fetches of evicted experts, most reads hit the page cache, where the map copy
        // is a pure memcpy and a positioned read pays a syscall plus that same copy. The
        // map arm also scaled with cache warmth (19.6 -> 23.0 steady across pairs) while
        // the positioned arm stayed flat.
        //
        // DECODE, STARVED HOST (free RAM well under the payload): the map loses, for the
        // same reason prefill does. The premise above is "most reads hit the page cache",
        // and it simply stops holding. Measured on the 26B-A4B row at 7.4 GiB available
        // against a 12.4 GiB payload: decode wall 7.8 s -> 3.9 s, steady +34.8% (3/3
        // pairs, sign consistent), demand faults 1.54M -> 0.45M, token ids identical.
        // `Gemma4CudaResident::decode_bulk_reads` picks between the two at load.
        //
        // PREFILL (cold sweep): the map loses badly. Layer-major prefill streams the whole
        // ~12 GiB payload once, so nearly every record is a genuine cold fault, and
        // `copy_from_slice` over an unresident 3.19 MiB record degenerates into ~816
        // synchronous 4 KiB demand faults. Profiled at **0.74 GB/s** against this drive's
        // flat 1.3-1.9 GB/s on explicit reads — 57% of prefill wall.
        //
        // So the choice is made per regime, not once. The runtime raises `bulk_reads`
        // around prefill. Byte-identical either way: same file, same offset, same length.
        // Fall through to the map on any error rather than failing the read.
        if self.bulk_reads.load(Ordering::Relaxed)
            && len >= CGHOST_BULK_READ_MIN_BYTES
            && read_exact_at(&self.file, buf, start).is_ok()
        {
            return Ok(());
        }
        if let Some(mmap) = &self.mmap {
            if let Ok(slice) = mmap.bytes(start, len as usize) {
                buf.copy_from_slice(slice);
                return Ok(());
            }
        }
        read_exact_at(&self.file, buf, start).map_err(|e| io_err(Path::new("<cghost>"), e))?;
        Ok(())
    }

    fn decode_group_tensor(
        tensor: &CghostTensor,
        span_start: u64,
        buf: &[u8],
    ) -> Result<CpuTensor> {
        let lo = (tensor.offset - span_start) as usize;
        let hi = lo + tensor.len as usize;
        cpu_tensor_from_gguf_bytes(&tensor.name, tensor.dtype, &tensor.dims, &buf[lo..hi])
    }

    /// Stream one transformer block's weights: one sequential read + decode to the same
    /// in-RAM storage the resident loader produces. Returns `(weights, bytes, read_us,
    /// decode_us)` — the read (positioned I/O, served from disk or page cache) and decode
    /// (dequant to the resident CpuTensor layout) times are split so a WRAITH Phase-0
    /// receipt can tell a disk-bound stall from a decode-bound one. Measurement-only: the
    /// two `Instant`s add no work to the streaming path.
    pub fn read_layer(
        &self,
        layer_idx: usize,
        buf: &mut Vec<u8>,
    ) -> Result<(LlamaLayerWeights, u64, u128, u128)> {
        let (span_len, read_us) = self.read_layer_bytes(layer_idx, buf)?;
        let (weights, _span, decode_us) = self.decode_layer(layer_idx, &buf[..])?;
        Ok((weights, span_len, read_us, decode_us))
    }

    /// The READ stage of `read_layer`, split out for the stage-split pipeline: one positioned
    /// read of the blk group's payload into `buf`. Returns `(span_bytes, read_us)`. Pair with
    /// `decode_layer` on the SAME `layer_idx` and the resulting `buf` to reconstruct exactly
    /// what `read_layer` produces (the two are byte-identical by construction).
    pub fn read_layer_bytes(&self, layer_idx: usize, buf: &mut Vec<u8>) -> Result<(u64, u128)> {
        let id = format!("blk.{layer_idx}");
        let read_started = std::time::Instant::now();
        let (group, _start) = self.read_group_payload(&id, buf)?;
        let read_us = read_started.elapsed().as_micros();
        let (_, span_len) = group.span();
        Ok((span_len, read_us))
    }

    /// The DECODE stage of `read_layer`: dequant the blk group payload already read into `buf`
    /// (by `read_layer_bytes` for the same `layer_idx`) into the resident `LlamaLayerWeights`
    /// layout. Returns `(weights, span_bytes, decode_us)`. Pure over `buf` — no file I/O — so a
    /// reader thread can be filling layer N+1's buffer while this runs on layer N.
    pub fn decode_layer(
        &self,
        layer_idx: usize,
        buf: &[u8],
    ) -> Result<(LlamaLayerWeights, u64, u128)> {
        let id = format!("blk.{layer_idx}");
        let group = self.group(&id)?;
        let (start, span_len) = group.span();
        let decode_started = std::time::Instant::now();
        let mut by_role: Vec<Option<CpuTensor>> = vec![None; LAYER_ROLES.len()];
        for tensor in &group.tensors {
            if let Some(slot) = LAYER_ROLES.iter().position(|r| *r == tensor.role) {
                by_role[slot] = Some(Self::decode_group_tensor(tensor, start, buf)?);
            }
        }
        let decode_us = decode_started.elapsed().as_micros();
        let mut take = |role: &str| -> Result<CpuTensor> {
            let slot = LAYER_ROLES.iter().position(|r| *r == role).unwrap();
            by_role[slot]
                .take()
                .ok_or_else(|| invalid(format!("group {id} is missing role \"{role}\"")))
        };
        Ok((
            LlamaLayerWeights {
                attention_norm: take("attn_norm")?,
                attention_q: take("attn_q")?,
                attention_k: take("attn_k")?,
                attention_v: take("attn_v")?,
                attention_output: take("attn_output")?,
                attention_biases: None,
                // The ghost (.cghost layer-streaming) format predates QK-norm and
                // carries no attn_q_norm/attn_k_norm roles, so ghost mode does not
                // support Qwen3-style models. Left None here; a Qwen3 ghost run is
                // not a supported configuration.
                attention_q_norm: None,
                attention_k_norm: None,
                post_attention_norm: None,
                post_ffw_norm: None,
                mla_q_a_proj: None,
                mla_q_a_layernorm: None,
                mla_q_b_proj: None,
                mla_kv_a_proj_with_mqa: None,
                mla_kv_a_layernorm: None,
                mla_kv_b_proj: None,
                ffn_norm: take("ffn_norm")?,
                ffn_gate: take("ffn_gate")?,
                ffn_up: take("ffn_up")?,
                ffn_down: take("ffn_down")?,
                moe_router: None, // ghost v1 only supports dense models
                moe_expert_bias: None,
                moe_shared_gate: None,
                moe_shared_up: None,
                moe_shared_down: None,
                decode_bindings: DecodeLinearBindings::default(),
            },
            span_len,
            decode_us,
        ))
    }

    /// Largest "blk.N" group payload in bytes â€” the size of the streaming read buffer.
    pub fn max_layer_span(&self) -> u64 {
        self.index
            .groups
            .iter()
            .filter(|g| g.id.starts_with("blk."))
            .map(|g| g.span().1)
            .max()
            .unwrap_or(0)
    }
}

/// Windows unbuffered (`FILE_FLAG_NO_BUFFERING`) reader for the strict-ceiling / cold-disk
/// measurement mode. Windows sets no-buffering at `CreateFile` time (not toggleable on an
/// open handle like macOS `F_NOCACHE`), so this owns a second handle to the same `.cghost`.
/// No-buffering imposes sector alignment on every read: the file offset, the transfer length,
/// and the destination buffer address must all be sector-aligned. `.cghost` blk-group offsets
/// are 16 KiB-aligned by the writer (a superset of the 4 KiB sector requirement), so only the
/// length (rounded up to a sector multiple, over-reading harmlessly into the following group)
/// and the scratch buffer (an aligned sub-slice) need handling here.
#[cfg(windows)]
struct UncachedReader {
    file: File,
    /// Logical sector size reported by the volume containing `.cghost`.
    sector: usize,
    /// Over-allocated so a sector-aligned sub-slice of the largest (rounded-up)
    /// group span always fits; sized once at open and never grown (growing would
    /// move the base pointer).
    scratch: Vec<u8>,
    file_len: u64,
}

#[cfg(windows)]
impl UncachedReader {
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

    fn volume_sector_size(path: &Path) -> std::io::Result<usize> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{GetDiskFreeSpaceW, GetVolumePathNameW};

        let absolute = std::fs::canonicalize(path)?;
        let wide = absolute
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let mut volume = vec![0u16; 32_768];
        // SAFETY: both UTF-16 buffers are live and NUL-terminated/capacity-sized
        // as required. The path names an existing file on the queried volume.
        if unsafe { GetVolumePathNameW(wide.as_ptr(), volume.as_mut_ptr(), volume.len() as u32) }
            == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let mut sectors_per_cluster = 0u32;
        let mut bytes_per_sector = 0u32;
        let mut free_clusters = 0u32;
        let mut total_clusters = 0u32;
        // SAFETY: volume contains the NUL-terminated root returned above and all
        // output pointers refer to initialized writable u32 storage.
        if unsafe {
            GetDiskFreeSpaceW(
                volume.as_ptr(),
                &mut sectors_per_cluster,
                &mut bytes_per_sector,
                &mut free_clusters,
                &mut total_clusters,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let sector = bytes_per_sector as usize;
        if sector == 0 || !sector.is_power_of_two() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("volume reported invalid sector size {sector}"),
            ));
        }
        Ok(sector)
    }

    fn open(path: &Path, max_layer_span: u64) -> std::io::Result<Self> {
        use std::os::windows::fs::OpenOptionsExt;
        let sector = Self::volume_sector_size(path)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(Self::FILE_FLAG_NO_BUFFERING)
            .open(path)?;
        let file_len = file.metadata()?.len();
        let cap = (max_layer_span as usize).next_multiple_of(sector) + sector;
        Ok(Self {
            file,
            sector,
            scratch: vec![0u8; cap],
            file_len,
        })
    }

    /// Fill `out` (already sized to `len`) with `file[start..start+len]` via cache-bypassing
    /// reads. Returns `Ok(true)` when served unbuffered, `Ok(false)` when it declines (offset
    /// not sector-aligned, or the aligned span would pass EOF, or the scratch can't hold it)
    /// so the caller uses the buffered handle instead. `Err` is a real I/O failure.
    fn read_into(&mut self, start: u64, len: u64, out: &mut [u8]) -> std::io::Result<bool> {
        use std::os::windows::fs::FileExt;
        let len = len as usize;
        let aligned_len = len.next_multiple_of(self.sector);
        // Locate a sector-aligned window inside the over-allocated scratch.
        let base = self.scratch.as_ptr() as usize;
        let pad = (self.sector - (base % self.sector)) % self.sector;
        if !start.is_multiple_of(self.sector as u64)
            || start + aligned_len as u64 > self.file_len
            || pad + aligned_len > self.scratch.len()
        {
            return Ok(false);
        }
        let region = &mut self.scratch[pad..pad + aligned_len];
        let mut filled = 0usize;
        while filled < aligned_len {
            // NO_BUFFERING transfers whole sectors, so `filled` stays sector-aligned and each
            // subsequent positioned read keeps its aligned-offset contract.
            let n = self
                .file
                .seek_read(&mut region[filled..], start + filled as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unbuffered .cghost read hit EOF before filling the sector-aligned span",
                ));
            }
            filled += n;
        }
        out.copy_from_slice(&self.scratch[pad..pad + len]);
        Ok(true)
    }
}

/// One layer's weights, read + decoded off the critical path by [`GhostPrefetcher`].
pub struct PrefetchedLayer {
    pub layer_idx: usize,
    pub weights: LlamaLayerWeights,
    /// Bytes read from disk for this layer's group.
    pub bytes: u64,
    /// Wall time the worker spent reading + decoding (overlapped with the main thread's
    /// forward, so it only shows up in token latency when it exceeds the compute time).
    pub worker_us: u128,
    /// Split of `worker_us`: positioned-read I/O time (disk or page cache) ...
    pub read_us: u128,
    /// ... and dequant-to-CpuTensor decode time. WRAITH Phase-0 uses the split to attribute
    /// the streaming stall to disk vs decode.
    pub decode_us: u128,
}

/// Double-buffered streaming: a background worker reads + decodes layer N+1 from the
/// `.cghost` file while the main thread runs layer N's forward. The result channel is a
/// rendezvous (capacity 0), so at most TWO layer windows exist at any instant â€” one in the
/// session being computed, one finished in the worker's hand awaiting handoff. The worker
/// fulfills requests strictly in order; the main thread queues each chunk's layer indices
/// ahead of consuming them (and primes the next chunk before the current one finishes, so
/// the disk is already rewinding to layer 0 of token N+1 during the last forwards of
/// token N).
pub struct GhostPrefetcher {
    request_tx: Option<std::sync::mpsc::Sender<usize>>,
    result_rx: Option<std::sync::mpsc::Receiver<Result<PrefetchedLayer>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl GhostPrefetcher {
    pub fn spawn(ghost: std::sync::Arc<GhostFile>) -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<usize>();
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<Result<PrefetchedLayer>>(0);
        let worker = std::thread::Builder::new()
            .name("ghost-prefetch".to_string())
            .spawn(move || {
                let mut buf: Vec<u8> = Vec::with_capacity(ghost.max_layer_span() as usize);
                while let Ok(layer_idx) = request_rx.recv() {
                    let started = std::time::Instant::now();
                    let result = ghost.read_layer(layer_idx, &mut buf).map(
                        |(weights, bytes, read_us, decode_us)| PrefetchedLayer {
                            layer_idx,
                            weights,
                            bytes,
                            worker_us: started.elapsed().as_micros(),
                            read_us,
                            decode_us,
                        },
                    );
                    // The receiver dropping mid-stream (generation ended) is a normal exit.
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn ghost-prefetch thread");
        Self {
            request_tx: Some(request_tx),
            result_rx: Some(result_rx),
            worker: Some(worker),
        }
    }

    /// Queue a layer to be read + decoded. Requests are fulfilled strictly in order.
    pub fn request(&self, layer_idx: usize) -> Result<()> {
        self.request_tx
            .as_ref()
            .expect("prefetcher closed")
            .send(layer_idx)
            .map_err(|_| invalid("ghost-prefetch worker exited unexpectedly".to_string()))
    }

    /// Block until the next requested layer is ready (instant when the worker finished
    /// while the previous layer was computing â€” the double-buffered steady state).
    pub fn next(&self) -> Result<PrefetchedLayer> {
        self.result_rx
            .as_ref()
            .expect("prefetcher closed")
            .recv()
            .map_err(|_| invalid("ghost-prefetch worker exited unexpectedly".to_string()))?
    }
}

impl Drop for GhostPrefetcher {
    fn drop(&mut self) {
        // Close BOTH channel ends before joining: dropping the request sender ends the
        // worker's request loop, and dropping the result receiver releases a worker that
        // is blocked mid-handoff on the rendezvous send (otherwise the join deadlocks).
        drop(self.request_tx.take());
        drop(self.result_rx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Two-stage streaming prefetcher (WRAITH Phase-2 read‖decode stage-split): a READER thread
/// fills layer N+1's raw buffer while a DECODER thread dequants layer N, so a layer's disk
/// read overlaps the previous layer's CPU dequant. The single-worker [`GhostPrefetcher`] does
/// read-then-decode serially, so its per-layer throughput is `read + decode`; this pipeline's
/// is `max(read, decode)` — the win is largest when read and decode are comparable (the
/// cold-NVMe regime). Same strict in-order handoff to the main thread and the same rendezvous
/// bound on decoded windows; the extra footprint is a small pool of raw byte buffers
/// (`read_ahead + 1` layer spans) that the ceiling accounting must include. Parity-identical
/// to the single-worker path by construction — same bytes read, same decode. Opt-in via
/// ghost-run `--stage-split`.
pub struct GhostPipelinePrefetcher {
    request_tx: Option<std::sync::mpsc::Sender<usize>>,
    result_rx: Option<std::sync::mpsc::Receiver<Result<PrefetchedLayer>>>,
    reader: Option<std::thread::JoinHandle<()>>,
    decoder: Option<std::thread::JoinHandle<()>>,
}

impl GhostPipelinePrefetcher {
    pub fn spawn(ghost: std::sync::Arc<GhostFile>, read_ahead: usize) -> Self {
        use std::sync::mpsc::{channel, sync_channel};
        let read_ahead = read_ahead.max(1);
        let (request_tx, request_rx) = channel::<usize>();
        // reader -> decoder: (layer_idx, filled buffer, Ok(read_us) | Err). Bounded so the
        // reader runs at most `read_ahead` layers ahead of the decoder (backpressure).
        let (stage_tx, stage_rx) = sync_channel::<(usize, Vec<u8>, Result<u128>)>(read_ahead);
        // decoder -> reader: drained buffers returned for reuse (bounds raw-buffer memory).
        let (free_tx, free_rx) = channel::<Vec<u8>>();
        // decoder -> main: rendezvous (capacity 0) — at most two decoded windows exist at once.
        let (result_tx, result_rx) = sync_channel::<Result<PrefetchedLayer>>(0);

        // Prime the raw-buffer pool (read_ahead + 1 spans) before moving free_tx to the decoder.
        let span_cap = ghost.max_layer_span() as usize;
        for _ in 0..(read_ahead + 1) {
            let _ = free_tx.send(Vec::with_capacity(span_cap));
        }

        let reader_ghost = std::sync::Arc::clone(&ghost);
        let reader = std::thread::Builder::new()
            .name("ghost-read".to_string())
            .spawn(move || {
                while let Ok(layer_idx) = request_rx.recv() {
                    // Reuse a pooled buffer; recv only blocks when the decoder is `read_ahead`
                    // layers behind (backpressure), or errors when the decoder has gone away.
                    let mut buf = match free_rx.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    };
                    let outcome = reader_ghost
                        .read_layer_bytes(layer_idx, &mut buf)
                        .map(|(_span, read_us)| read_us);
                    let read_failed = outcome.is_err();
                    if stage_tx.send((layer_idx, buf, outcome)).is_err() {
                        break;
                    }
                    if read_failed {
                        break; // terminal read error already forwarded to the decoder
                    }
                }
            })
            .expect("failed to spawn ghost-read thread");

        let decoder_ghost = ghost;
        let decoder = std::thread::Builder::new()
            .name("ghost-decode".to_string())
            .spawn(move || {
                while let Ok((layer_idx, buf, read_outcome)) = stage_rx.recv() {
                    let result = match read_outcome {
                        Ok(read_us) => decoder_ghost.decode_layer(layer_idx, &buf).map(
                            |(weights, span, decode_us)| PrefetchedLayer {
                                layer_idx,
                                weights,
                                bytes: span,
                                worker_us: read_us + decode_us,
                                read_us,
                                decode_us,
                            },
                        ),
                        Err(e) => Err(e),
                    };
                    let failed = result.is_err();
                    // Return the buffer to the pool BEFORE the (possibly blocking) rendezvous
                    // handoff, so the reader keeps working ahead while main consumes this layer.
                    let _ = free_tx.send(buf);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                    if failed {
                        break;
                    }
                }
            })
            .expect("failed to spawn ghost-decode thread");

        Self {
            request_tx: Some(request_tx),
            result_rx: Some(result_rx),
            reader: Some(reader),
            decoder: Some(decoder),
        }
    }

    /// Queue a layer for the reader; fulfilled strictly in order (matches [`GhostPrefetcher`]).
    pub fn request(&self, layer_idx: usize) -> Result<()> {
        self.request_tx
            .as_ref()
            .expect("pipeline prefetcher closed")
            .send(layer_idx)
            .map_err(|_| invalid("ghost stage-split reader exited unexpectedly".to_string()))
    }

    /// Block until the next layer is decoded and handed off.
    pub fn next(&self) -> Result<PrefetchedLayer> {
        self.result_rx
            .as_ref()
            .expect("pipeline prefetcher closed")
            .recv()
            .map_err(|_| invalid("ghost stage-split decoder exited unexpectedly".to_string()))?
    }
}

impl Drop for GhostPipelinePrefetcher {
    fn drop(&mut self) {
        // Cascade shutdown: dropping request_tx ends an idle reader; dropping result_rx frees a
        // decoder blocked on the rendezvous send. A decoder exit drops its stage_rx + free_tx,
        // releasing a reader blocked on stage_tx.send (full) or free_rx.recv (empty); a reader
        // exit drops stage_tx, releasing a decoder blocked on stage_rx.recv — so both unwind.
        drop(self.request_tx.take());
        drop(self.result_rx.take());
        if let Some(decoder) = self.decoder.take() {
            let _ = decoder.join();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, GgufTensorDescriptor};
    use crate::model::{Gemma4LayerTensors, Gemma4MoeLayerTensors};
    use std::collections::BTreeMap;

    #[test]
    fn artifact_guard_rejects_same_output_before_creation() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        let shared_output = dir.path().join("shared-output");
        std::fs::write(&source_path, b"source-must-survive").unwrap();

        let err = ensure_distinct_artifact_paths(&[
            ("source GGUF", &source_path),
            (".cghost output", &source_path),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert_eq!(std::fs::read(&source_path).unwrap(), b"source-must-survive");

        let err = ensure_distinct_artifact_paths(&[
            (".cghost output", &shared_output),
            ("hot-shadow output", &shared_output),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert!(!shared_output.exists());
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn artifact_guard_rejects_hard_link_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("source.gguf");
        let alias_path = dir.path().join("alias.cghost");
        std::fs::write(&source_path, b"source-must-survive").unwrap();
        std::fs::hard_link(&source_path, &alias_path).unwrap();

        let err = ensure_distinct_artifact_paths(&[
            ("source GGUF", &source_path),
            (".cghost output", &alias_path),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert_eq!(std::fs::read(&source_path).unwrap(), b"source-must-survive");
        assert_eq!(std::fs::read(&alias_path).unwrap(), b"source-must-survive");
    }

    fn write_tiny_moe_cghost(path: &Path) {
        let mut file = File::create(path).unwrap();
        file.write_all(CGHOST_MAGIC).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let mut cursor = (CGHOST_MAGIC.len() + 8) as u64;
        let mut groups = Vec::new();
        for expert in 0..2usize {
            let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
            file.write_all(&vec![0; (aligned - cursor) as usize])
                .unwrap();
            cursor = aligned;
            let payload = [
                expert as u8,
                10 + expert as u8,
                20 + expert as u8,
                30 + expert as u8,
            ];
            file.write_all(&payload).unwrap();
            groups.push(CghostGroup {
                id: format!("blk.0.exp.{expert}"),
                tensors: vec![
                    CghostTensor {
                        name: format!("gate#{expert}"),
                        role: "gate_up_exps".into(),
                        dtype: GgufTensorType::Q4_0,
                        dims: vec![32, 2],
                        offset: cursor,
                        len: 2,
                    },
                    CghostTensor {
                        name: format!("down#{expert}"),
                        role: "down_exps".into(),
                        dtype: GgufTensorType::Q4_0,
                        dims: vec![32, 2],
                        offset: cursor + 2,
                        len: 2,
                    },
                ],
                source_sample_sha256: None,
            });
            cursor += payload.len() as u64;
        }
        let index = CghostIndex {
            version: 2,
            layout: CghostLayout::MoeExperts,
            source_model: "tiny.gguf".into(),
            block_count: 1,
            tied_output: true,
            expert_count: Some(2),
            expert_used_count: Some(1),
            source_identity: None,
            groups,
        };
        let index_json = serde_json::to_vec(&index).unwrap();
        let index_offset = cursor;
        file.write_all(&index_json).unwrap();
        file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
            .unwrap();
        file.write_all(&index_offset.to_le_bytes()).unwrap();
    }

    #[test]
    fn v2_moe_index_validates_and_reads_one_contiguous_expert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();
        ghost.validate_moe_layout(1, 2, 1).unwrap();

        let direct = ghost.moe_access.as_ref().unwrap();
        assert_eq!(direct.groups.len(), 2);
        for expert_idx in 0..2 {
            let access = direct.get(0, expert_idx).unwrap();
            assert_eq!(
                ghost.index.groups[access.group_idx].id,
                format!("blk.0.exp.{expert_idx}")
            );
            assert_eq!(access.gate_up_tensor_idx, Some(0));
            assert_eq!(access.down_tensor_idx, Some(1));
        }

        let expert = ghost.read_moe_expert(0, 1).unwrap();
        assert_eq!(expert.byte_len(), 4);
        let (gate_bytes, gate_range) = expert.tensor_backing(&expert.gate_up);
        let (down_bytes, down_range) = expert.tensor_backing(&expert.down);
        assert!(Arc::ptr_eq(&gate_bytes, &down_bytes));
        assert_eq!(&gate_bytes[gate_range], &[1, 11]);
        assert_eq!(&down_bytes[down_range], &[21, 31]);

        let mapped = ghost.mapped_moe_expert(0, 1).unwrap().unwrap();
        assert_eq!(mapped.tensor_bytes(&mapped.gate_up).unwrap(), &[1, 11]);
        assert_eq!(mapped.tensor_bytes(&mapped.down).unwrap(), &[21, 31]);
    }

    #[test]
    fn direct_moe_expert_read_matches_allocating_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();

        for expert_idx in 0..2 {
            let expected = ghost.read_moe_expert(0, expert_idx).unwrap();
            let len = ghost.moe_expert_byte_len(0, expert_idx).unwrap();
            assert_eq!(len, expected.byte_len());
            let mut destination = vec![0xa5; len];
            ghost
                .read_moe_expert_into(0, expert_idx, &mut destination)
                .unwrap();
            assert_eq!(destination.as_slice(), expected.bytes.as_ref());
        }
    }

    #[test]
    fn expert_record_layout_resolves_projection_roles_in_either_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();

        let canonical = ghost.moe_expert_record_layout(0, 0).unwrap();
        assert_eq!(canonical.byte_len, 4);
        assert_eq!(canonical.gate_up, 0..2);
        assert_eq!(canonical.down, 2..4);

        let group = &mut ghost.index.groups[0];
        let start = group.tensors[0].offset;
        group.tensors.swap(0, 1);
        group.tensors[0].offset = start;
        group.tensors[1].offset = start + group.tensors[0].len;
        ghost.moe_access = GhostMoeAccessIndex::build(&ghost.index).unwrap();

        let reversed = ghost.moe_expert_record_layout(0, 0).unwrap();
        assert_eq!(reversed.byte_len, 4);
        assert_eq!(reversed.down, 0..2);
        assert_eq!(reversed.gate_up, 2..4);
    }

    #[test]
    fn direct_moe_expert_read_rejects_wrong_destination_lengths_before_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();
        let expected_len = ghost.moe_expert_byte_len(0, 0).unwrap();

        for actual_len in [expected_len - 1, expected_len + 1] {
            let mut destination = vec![0xa5; actual_len];
            let before = destination.clone();
            let err = ghost
                .read_moe_expert_into(0, 0, &mut destination)
                .unwrap_err();
            assert!(err.to_string().contains(&format!(
                "destination length {actual_len} does not match record length {expected_len}"
            )));
            assert_eq!(destination, before);
        }
    }

    #[test]
    fn direct_moe_expert_read_rejects_invalid_coordinates_before_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();

        for (layer_idx, expert_idx) in [(1, 0), (0, 2), (usize::MAX, usize::MAX)] {
            let mut destination = [0xa5; 4];
            let err = ghost
                .read_moe_expert_into(layer_idx, expert_idx, &mut destination)
                .unwrap_err();
            assert!(err
                .to_string()
                .contains(&format!("blk.{layer_idx}.exp.{expert_idx}")));
            assert_eq!(destination, [0xa5; 4]);
        }
    }

    #[test]
    fn direct_moe_expert_read_rejects_corrupt_noncontiguous_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();
        ghost.index.groups[0].tensors[1].offset += 1;

        let err = ghost.moe_expert_byte_len(0, 0).unwrap_err();
        assert!(err.to_string().contains("not a contiguous expert record"));
    }

    #[test]
    fn fixed_expert_layout_rejects_reversed_tensor_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();
        ghost.index.groups[0].tensors.swap(0, 1);
        ghost.moe_access = GhostMoeAccessIndex::build(&ghost.index).unwrap();

        let err = ghost.validate_moe_expert_record_layout(0, 0).unwrap_err();
        assert!(err
            .to_string()
            .contains("non-canonical routed-expert tensor order"));
    }

    #[test]
    fn fixed_expert_layout_rejects_extra_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();
        let group = &mut ghost.index.groups[0];
        let extra_offset = group.tensors[1].offset + group.tensors[1].len;
        group.tensors.push(CghostTensor {
            name: "unexpected".into(),
            role: "unexpected".into(),
            dtype: GgufTensorType::Q4_0,
            dims: vec![32, 1],
            offset: extra_offset,
            len: 0,
        });
        ghost.moe_access = GhostMoeAccessIndex::build(&ghost.index).unwrap();

        let err = ghost.validate_moe_expert_record_layout(0, 0).unwrap_err();
        assert!(err.to_string().contains("exactly two tensors"));
    }

    #[test]
    fn fixed_expert_layout_rejects_gap_between_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();
        ghost.index.groups[0].tensors[1].offset += 1;
        ghost.moe_access = GhostMoeAccessIndex::build(&ghost.index).unwrap();

        let err = ghost.validate_moe_expert_record_layout(0, 0).unwrap_err();
        assert!(err.to_string().contains("down_exps starts"));
    }

    #[test]
    fn fixed_expert_layout_validates_all_records_and_exact_geometry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();

        assert_eq!(ghost.validate_moe_expert_record_layout(0, 0).unwrap(), 4);
        ghost.validate_moe_expert_record_layouts(1, 2, 4).unwrap();
        let err = ghost
            .validate_moe_expert_record_layouts(1, 2, 5)
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("does not match fixed kernel geometry 5"));
    }

    #[test]
    fn direct_moe_expert_read_reports_truncated_backing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let mut ghost = GhostFile::open(&path).unwrap();
        let (group, _) = ghost.moe_group_access(0, 1).unwrap();
        let start = group.tensors[0].offset;
        let len = ghost.moe_expert_byte_len(0, 1).unwrap();

        // Windows refuses `set_len` on a file with a live user mapping
        // (ERROR_USER_MAPPED_FILE), so drop the mapping before truncating. The
        // positioned reader under test never reads through the mapping, so the
        // path being exercised is unchanged; POSIX hosts allowed the truncate
        // either way, which is why this only ever failed on Windows.
        ghost.mmap = None;

        // Keep the already-parsed index resident, then truncate the backing payload halfway
        // through the selected record. The positioned reader must report EOF, never success.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(start + len as u64 / 2)
            .unwrap();
        let mut destination = vec![0xa5; len];
        let err = ghost
            .read_moe_expert_into(0, 1, &mut destination)
            .unwrap_err();
        match err {
            BackendError::Io { source, .. } => {
                assert_eq!(source.kind(), std::io::ErrorKind::UnexpectedEof)
            }
            other => panic!("expected positioned-read I/O error, got {other}"),
        }
    }

    #[test]
    fn v2_moe_index_fails_closed_on_model_shape_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.cghost");
        write_tiny_moe_cghost(&path);
        let ghost = GhostFile::open(&path).unwrap();
        let err = ghost.validate_moe_layout(1, 128, 8).unwrap_err();
        assert!(err.to_string().contains("index mismatch"));
    }

    #[test]
    fn legacy_v1_index_defaults_to_dense_layout() {
        let json = r#"{
            "version":1,
            "source_model":"dense.gguf",
            "block_count":1,
            "tied_output":true,
            "groups":[]
        }"#;
        let index: CghostIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.layout, CghostLayout::DenseLayers);
        assert_eq!(index.expert_count, None);
        assert!(GhostMoeAccessIndex::build(&index).unwrap().is_none());
    }

    #[test]
    fn moe_group_parser_rejects_noncanonical_aliases() {
        assert_eq!(parse_moe_group_id("blk.29.exp.127"), Some((29, 127)));
        assert_eq!(parse_moe_group_id("blk.029.exp.127"), None);
        assert_eq!(parse_moe_group_id("blk.29.exp.0127"), None);
        assert_eq!(parse_moe_group_id("blk.29.expert.127"), None);
    }

    #[test]
    fn moe_repacker_splices_each_expert_without_materializing_whole_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let source_path = dir.path().join("tiny.gguf");
        let out_path = dir.path().join("tiny.cghost");
        let prefix = vec![0xcc; 64];
        let gate_payload: Vec<u8> = (10..18).collect();
        let down_payload: Vec<u8> = (20..28).collect();
        let mut source = File::create(&source_path).unwrap();
        source.write_all(&prefix).unwrap();
        source.write_all(&gate_payload).unwrap();
        source.write_all(&down_payload).unwrap();
        source.sync_all().unwrap();
        drop(source);

        let descriptor = |name: &str, offset: u64| GgufTensorDescriptor {
            name: name.into(),
            dimensions: vec![1, 1, 2],
            tensor_type: GgufTensorType::F32,
            relative_offset: offset,
            absolute_offset: offset,
            n_bytes: 8,
        };
        let gate = descriptor("blk.0.ffn_gate_up_exps.weight", 64);
        let down = descriptor("blk.0.ffn_down_exps.weight", 72);
        let dummy = GgufTensorDescriptor {
            name: "dummy".into(),
            dimensions: vec![1],
            tensor_type: GgufTensorType::F32,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: 4,
        };
        let gguf = GgufFile {
            path: source_path.clone(),
            version: 3,
            tensor_count: 2,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: BTreeMap::new(),
            tensors: vec![gate.clone(), down.clone()],
        };
        let store = TensorStore::open(&source_path, &gguf);
        let layer = Gemma4LayerTensors {
            attn_norm: dummy.clone(),
            attn_q: dummy.clone(),
            attn_k: Some(dummy.clone()),
            attn_v: Some(dummy.clone()),
            attn_output: dummy.clone(),
            attn_q_norm: dummy.clone(),
            attn_k_norm: Some(dummy.clone()),
            post_attention_norm: dummy.clone(),
            ffn_norm: dummy.clone(),
            post_ffw_norm: dummy.clone(),
            post_norm: None,
            ffn_gate: dummy.clone(),
            ffn_up: dummy.clone(),
            ffn_down: dummy.clone(),
            ple_inp_gate: None,
            ple_proj: None,
            ple_output_scale: None,
            moe: Some(Gemma4MoeLayerTensors {
                gate_inp: dummy.clone(),
                gate_inp_scale: dummy.clone(),
                gate_up_exps: gate,
                down_exps: down,
                down_exps_scale: dummy.clone(),
                pre_norm_2: dummy.clone(),
                post_norm_1: dummy.clone(),
                post_norm_2: dummy.clone(),
            }),
        };
        let binding = Gemma4Binding {
            token_embedding: dummy.clone(),
            output_norm: dummy.clone(),
            output: dummy,
            output_is_tied_embedding: true,
            rope_freqs: None,
            per_layer_token_embd: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            layers: vec![layer],
        };

        // Both the exact source spelling and a hard-link alias must fail
        // before File::create can truncate a single source byte.
        let source_before = std::fs::read(&source_path).unwrap();
        let err =
            write_cghost_moe_with_counts(&store, &binding, 2, 1, "tiny.gguf", &source_path, None)
                .unwrap_err();
        assert!(err.to_string().contains("distinct files"));
        assert_eq!(std::fs::read(&source_path).unwrap(), source_before);

        #[cfg(any(unix, windows))]
        {
            let alias_path = dir.path().join("source-hard-link.cghost");
            std::fs::hard_link(&source_path, &alias_path).unwrap();
            let err = write_cghost_moe_with_counts(
                &store,
                &binding,
                2,
                1,
                "tiny.gguf",
                &alias_path,
                None,
            )
            .unwrap_err();
            assert!(err.to_string().contains("distinct files"));
            assert_eq!(std::fs::read(&source_path).unwrap(), source_before);
            assert_eq!(std::fs::read(&alias_path).unwrap(), source_before);
            std::fs::remove_file(alias_path).unwrap();
        }

        let index =
            write_cghost_moe_with_counts(&store, &binding, 2, 1, "tiny.gguf", &out_path, None)
                .unwrap();
        assert_eq!(index.layout, CghostLayout::MoeExperts);
        assert_eq!(
            index.source_identity.as_ref().unwrap().scheme,
            MOE_SOURCE_IDENTITY_SCHEME
        );
        assert!(index
            .groups
            .iter()
            .all(|group| group.source_sample_sha256.is_some()));
        let ghost = GhostFile::open(&out_path).unwrap();
        ghost.validate_moe_binding(&binding, 2).unwrap();
        ghost
            .validate_moe_source_identity(&source_path, &binding, 2)
            .unwrap();

        // Legacy shape-only artifacts remain compatible with an ordinary full
        // GGUF, but a sparse same-shape source must fail closed because its
        // missing expert bytes could belong to any checkpoint.
        let mut legacy_ghost = GhostFile::open(&out_path).unwrap();
        legacy_ghost.index.source_identity = None;
        legacy_ghost
            .validate_moe_source_identity(&source_path, &binding, 2)
            .unwrap();
        #[cfg(unix)]
        {
            let sparse_path = dir.path().join("stale-sparse.gguf");
            let sparse = File::create(&sparse_path).unwrap();
            sparse.set_len(1024 * 1024).unwrap();
            drop(sparse);
            assert!(source_is_sparse(&sparse_path).unwrap());
            if !allow_legacy_sparse_identity() {
                let err = legacy_ghost
                    .validate_moe_source_identity(&sparse_path, &binding, 2)
                    .unwrap_err();
                assert!(err.to_string().contains("strong sampled source identity"));
                assert!(err.to_string().contains(ALLOW_LEGACY_SPARSE_IDENTITY_ENV));
            }
        }
        for (expert_idx, expected_gate, expected_down) in [
            (0, &gate_payload[0..4], &down_payload[0..4]),
            (1, &gate_payload[4..8], &down_payload[4..8]),
        ] {
            let expert = ghost.read_moe_expert(0, expert_idx).unwrap();
            let (bytes, range) = expert.tensor_backing(&expert.gate_up);
            assert_eq!(&bytes[range], expected_gate);
            let (bytes, range) = expert.tensor_backing(&expert.down);
            assert_eq!(&bytes[range], expected_down);
        }

        // Same shape and length are insufficient: a changed common byte and a
        // changed expert byte must each refuse the pair with a precise class.
        let mut wrong_source = std::fs::read(&source_path).unwrap();
        wrong_source[0] ^= 0x01;
        std::fs::write(&source_path, &wrong_source).unwrap();
        let err = ghost
            .validate_moe_source_identity(&source_path, &binding, 2)
            .unwrap_err();
        assert!(err.to_string().contains("common-core identity mismatch"));
        wrong_source[0] ^= 0x01;
        wrong_source[64] ^= 0x01;
        std::fs::write(&source_path, &wrong_source).unwrap();
        let err = ghost
            .validate_moe_source_identity(&source_path, &binding, 2)
            .unwrap_err();
        assert!(err.to_string().contains("expert identity mismatch"));
        wrong_source[64] ^= 0x01;
        std::fs::write(&source_path, &wrong_source).unwrap();

        // Corrupting a sampled byte in the `.cghost` payload is detected on
        // the normal record read without an additional full-record I/O pass.
        let corrupt_offset = ghost.index.groups[0].tensors[0].offset;
        drop(ghost);
        let mut corrupt = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        corrupt.seek(SeekFrom::Start(corrupt_offset)).unwrap();
        corrupt.write_all(&[0xff]).unwrap();
        corrupt.sync_all().unwrap();
        drop(corrupt);
        let ghost = GhostFile::open(&out_path).unwrap();
        let err = ghost.read_moe_expert(0, 0).unwrap_err();
        assert!(err.to_string().contains("payload identity mismatch"));
    }
}
