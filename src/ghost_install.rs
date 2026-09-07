//! Persistent catalog-managed Ghost-MoE installations.
//!
//! The Gemma 4 26B-A4B catalog row is larger than laptop VRAM, but its routed
//! experts do not need to be resident together. A user who opts into Ghost-MoE
//! downloads the canonical GGUF once; Camelid then repacks the routed experts
//! into a sibling `.cghost` and atomically replaces the full GGUF with a sparse,
//! offset-compatible common-core shadow. A tiny marker makes that choice durable
//! across API loads and process restarts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::capability::HardwareProfile;
use crate::fit::{FitInputs, FitVerdict};
use crate::gguf::read_metadata;
use crate::ghost::{write_cghost_moe, GhostFile};
use crate::ghost_hot::write_moe_hot_shadow;
use crate::model::{Gemma4Binding, LlamaModelConfig};
use crate::tensor::TensorStore;
use crate::{BackendError, Result};

pub const GEMMA4_26B_GHOST_CATALOG_ID: &str = "gemma4_26b_a4b_it_q4_0";
pub const GEMMA4_26B_GHOST_MODEL_FILENAME: &str = "gemma-4-26B_q4_0-it.gguf";
pub const GEMMA4_26B_GHOST_CGHOST_FILENAME: &str = "gemma-4-26B_q4_0-it.cghost";

/// Exact payload size produced by the v2 expert repack for the pinned 26B row.
pub const GEMMA4_26B_GHOST_CGHOST_BYTES: u64 = 12_899_030_749;
/// Conservative physical footprint of the sparse common-core shadow on Windows.
/// The measured NTFS allocation is 2,100,505,504 bytes; leave room for filesystem
/// allocation-granularity differences.
pub const GEMMA4_26B_GHOST_COMMON_DISK_BYTES: u64 = 2_250_000_000;
/// Runtime common weights plus bounded KV/scratch before the adaptive expert cache.
/// The expert cache consumes only VRAM left after this resident base.
pub const GEMMA4_26B_GHOST_RESIDENT_BYTES: u64 = 2_650_000_000;
pub const GEMMA4_26B_GHOST_CACHE_MIB: usize = 64;
/// REQUESTED routed-expert residency for the VRAM cache. The runtime takes
/// `min(requested, VRAM-fit)`, so this is deliberately above any fit this card
/// can reach: the free-VRAM probe is what binds. Requesting exactly the measured
/// fit (803 on the tracked 6 GiB box at the 4096-position KV) was a trap — it
/// silently turned every later VRAM-freeing improvement (e.g. a smaller KV via
/// `CAMELID_GEMMA4_GHOST_CUDA_CONTEXT`) into a no-op for the serve lane.
pub const GEMMA4_26B_GHOST_CACHE_EXPERTS: usize = 1600;

const MARKER_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct GhostMoeCatalogSupport {
    /// The exact row has a catalog-managed Ghost representation.
    pub available: bool,
    /// This binary/host combination can run the supported Windows CUDA lane.
    pub host_eligible: bool,
    /// Capacity verdict for the common resident set, not the full GGUF.
    pub fit: FitVerdict,
    /// Ghost is the preferred lane when the full model is not VRAM-resident.
    pub recommended: bool,
    /// Final physical storage estimate after the temporary full GGUF is reclaimed.
    pub installed_bytes: u64,
    /// Peak total storage while the downloaded full GGUF is being prepared.
    pub peak_disk_bytes: u64,
    pub cghost_filename: &'static str,
}

#[derive(Debug, Clone)]
pub struct GhostMoeRuntimeConfig {
    pub cghost: PathBuf,
    pub cache_mib: usize,
    pub strict_cache: bool,
    /// True only for the user-selected, catalog-managed supported lane. Ad-hoc
    /// environment-driven Ghost runs retain their explicit experimental gate.
    pub catalog_managed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct GhostInstallMarker {
    version: u32,
    catalog_id: String,
    model_filename: String,
    cghost_filename: String,
    cache_mib: usize,
}

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidModelMetadata(message.into())
}

pub fn is_catalog_row(catalog_id: &str) -> bool {
    catalog_id == GEMMA4_26B_GHOST_CATALOG_ID
}

pub fn is_catalog_model_path(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case(GEMMA4_26B_GHOST_MODEL_FILENAME))
}

pub fn resident_footprint() -> FitInputs {
    FitInputs {
        weight_bytes: GEMMA4_26B_GHOST_RESIDENT_BYTES,
        kv_bytes_at_ctx: 0,
    }
}

/// Whether the host has enough physical GPU capacity to prepare this lane.
/// `InsufficientFreeMemory` is intentionally allowed: preparation itself is a
/// disk operation, and the subsequent replace-load releases Camelid's resident
/// model before its authoritative live-memory check. `WontFit` still blocks a
/// GPU that is physically too small.
pub fn fit_allows_preparation(fit: FitVerdict) -> bool {
    matches!(
        fit,
        FitVerdict::FitsResident | FitVerdict::InsufficientFreeMemory
    )
}

pub fn host_eligible(hw: &HardwareProfile) -> bool {
    cfg!(all(target_os = "windows", feature = "cuda"))
        && hw.cuda_available
        && hw
            .cuda_compute_capability
            .is_some_and(|capability| capability >= (6, 1))
}

pub fn catalog_support(
    catalog_id: &str,
    full_fit: FitVerdict,
    hw: &HardwareProfile,
) -> Option<GhostMoeCatalogSupport> {
    if !is_catalog_row(catalog_id) {
        return None;
    }
    let eligible = host_eligible(hw);
    let fit = if eligible {
        crate::fit::assess_gpu_resident(hw, &resident_footprint())
    } else {
        FitVerdict::Unknown
    };
    Some(GhostMoeCatalogSupport {
        available: true,
        host_eligible: eligible,
        fit,
        recommended: eligible
            && fit == FitVerdict::FitsResident
            && full_fit != FitVerdict::FitsResident,
        installed_bytes: GEMMA4_26B_GHOST_CGHOST_BYTES
            .saturating_add(GEMMA4_26B_GHOST_COMMON_DISK_BYTES),
        peak_disk_bytes: 14_439_361_440_u64
            .saturating_add(GEMMA4_26B_GHOST_CGHOST_BYTES)
            .saturating_add(GEMMA4_26B_GHOST_COMMON_DISK_BYTES),
        cghost_filename: GEMMA4_26B_GHOST_CGHOST_FILENAME,
    })
}

pub fn marker_path(model_path: &Path) -> PathBuf {
    model_path.with_extension("ghost.json")
}

fn marker_for_model(_model_path: &Path) -> GhostInstallMarker {
    GhostInstallMarker {
        version: MARKER_VERSION,
        catalog_id: GEMMA4_26B_GHOST_CATALOG_ID.to_string(),
        model_filename: GEMMA4_26B_GHOST_MODEL_FILENAME.to_string(),
        cghost_filename: GEMMA4_26B_GHOST_CGHOST_FILENAME.to_string(),
        cache_mib: GEMMA4_26B_GHOST_CACHE_MIB,
    }
}

fn read_marker(model_path: &Path) -> std::result::Result<Option<GhostInstallMarker>, String> {
    if !is_catalog_model_path(model_path) {
        return Ok(None);
    }
    let path = marker_path(model_path);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("could not read {}: {error}", path.display())),
    };
    let marker: GhostInstallMarker = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid Ghost-MoE marker {}: {error}", path.display()))?;
    if marker.version != MARKER_VERSION
        || marker.catalog_id != GEMMA4_26B_GHOST_CATALOG_ID
        || marker.model_filename != GEMMA4_26B_GHOST_MODEL_FILENAME
        || marker.cghost_filename != GEMMA4_26B_GHOST_CGHOST_FILENAME
    {
        return Err(format!(
            "Ghost-MoE marker {} does not identify the supported Gemma 4 26B artifact",
            path.display()
        ));
    }
    Ok(Some(marker))
}

pub fn installed_runtime_config(
    model_path: &Path,
) -> std::result::Result<Option<GhostMoeRuntimeConfig>, String> {
    let Some(marker) = read_marker(model_path)? else {
        return Ok(None);
    };
    let cghost = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&marker.cghost_filename);
    if !cghost.is_file() {
        return Err(format!(
            "Ghost-MoE is enabled for {}, but {} is missing",
            model_path.display(),
            cghost.display()
        ));
    }
    Ok(Some(GhostMoeRuntimeConfig {
        cghost,
        cache_mib: marker.cache_mib,
        strict_cache: false,
        catalog_managed: true,
    }))
}

pub fn is_prepared(model_path: &Path) -> bool {
    installed_runtime_config(model_path).is_ok_and(|config| config.is_some())
}

/// Remove the durable marker and expert pack owned by a prepared catalog GGUF.
/// The caller deletes the identity-bound GGUF separately. Invalid/missing markers
/// never authorize touching a sidecar, and the fixed validated filename prevents
/// marker content from escaping the models directory.
pub fn remove_prepared_sidecars(model_path: &Path) -> std::io::Result<u64> {
    let Some(config) = read_marker(model_path)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?
    else {
        return Ok(0);
    };
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let cghost = parent.join(config.cghost_filename);
    let marker = marker_path(model_path);
    let mut removed = 0_u64;
    for path in [&cghost, &marker] {
        match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => {
                std::fs::remove_file(path)?;
                removed = removed.saturating_add(metadata.len());
            }
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{} is not a regular Ghost-MoE sidecar", path.display()),
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(removed)
}

fn validate_existing_cghost(
    path: &Path,
    source_path: &Path,
    binding: &Gemma4Binding,
    expert_count: usize,
) -> bool {
    GhostFile::open(path).is_ok_and(|ghost| {
        ghost.validate_moe_binding(binding, expert_count).is_ok()
            && ghost
                .validate_moe_source_identity(source_path, binding, expert_count)
                .is_ok()
    })
}

#[cfg(windows)]
fn replace_file_atomically(replaced: &Path, replacement: &Path, backup: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{ReplaceFileW, REPLACEFILE_WRITE_THROUGH};

    let wide = |path: &Path| {
        path.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>()
    };
    let replaced_wide = wide(replaced);
    let replacement_wide = wide(replacement);
    let backup_wide = wide(backup);
    // SAFETY: all three buffers are live, NUL-terminated UTF-16 paths. The
    // replacement and replaced files exist and the backup name is unique.
    let ok = unsafe {
        ReplaceFileW(
            replaced_wide.as_ptr(),
            replacement_wide.as_ptr(),
            backup_wide.as_ptr(),
            REPLACEFILE_WRITE_THROUGH,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(BackendError::Io {
            path: replaced.to_path_buf(),
            source: std::io::Error::last_os_error(),
        });
    }
    std::fs::remove_file(backup).map_err(|source| BackendError::Io {
        path: backup.to_path_buf(),
        source,
    })?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file_atomically(replaced: &Path, replacement: &Path, _backup: &Path) -> Result<()> {
    std::fs::rename(replacement, replaced).map_err(|source| BackendError::Io {
        path: replaced.to_path_buf(),
        source,
    })
}

fn promote_new_file(temp: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        let backup = destination.with_extension(format!("ghost-backup-{}", uuid::Uuid::new_v4()));
        replace_file_atomically(destination, temp, &backup)
    } else {
        std::fs::rename(temp, destination).map_err(|source| BackendError::Io {
            path: destination.to_path_buf(),
            source,
        })
    }
}

/// Prepare the exact catalog row and persist the opt-in. All large outputs are
/// written to unique temporary siblings. The marker is committed last, so a
/// crash can leave reclaimable temp files but can never advertise a partial pair.
pub fn prepare(model_path: &Path) -> Result<()> {
    if !is_catalog_model_path(model_path) {
        return Err(invalid(format!(
            "Ghost-MoE preparation is supported only for {GEMMA4_26B_GHOST_MODEL_FILENAME}"
        )));
    }
    if is_prepared(model_path) {
        return Ok(());
    }
    let gguf = read_metadata(model_path)?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let moe = config
        .moe
        .as_ref()
        .ok_or_else(|| invalid("Ghost-MoE preparation requires MoE metadata"))?;
    let binding = Gemma4Binding::bind(&gguf, &config)?;
    let store = TensorStore::open(model_path, &gguf);
    let parent = model_path.parent().unwrap_or_else(|| Path::new("."));
    let cghost = parent.join(GEMMA4_26B_GHOST_CGHOST_FILENAME);
    let nonce = uuid::Uuid::new_v4();
    let cghost_temp = parent.join(format!(".{GEMMA4_26B_GHOST_CGHOST_FILENAME}.{nonce}.part"));
    let hot_temp = parent.join(format!(
        ".{GEMMA4_26B_GHOST_MODEL_FILENAME}.{nonce}.hot.part"
    ));
    let marker = marker_path(model_path);
    let marker_temp = parent.join(format!(".gemma4-26b.{nonce}.ghost-marker.part"));
    let source_name = model_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| GEMMA4_26B_GHOST_MODEL_FILENAME.to_string());

    let reuse_cghost =
        validate_existing_cghost(&cghost, model_path, &binding, moe.expert_count as usize);
    let result = (|| {
        if !reuse_cghost {
            write_cghost_moe(&store, &binding, &config, &source_name, &cghost_temp, None)?;
            let ghost = GhostFile::open(&cghost_temp)?;
            ghost.validate_moe_binding(&binding, moe.expert_count as usize)?;
            ghost.validate_moe_source_identity(model_path, &binding, moe.expert_count as usize)?;
        }

        write_moe_hot_shadow(model_path, &hot_temp, &gguf, None)?;
        if !reuse_cghost {
            promote_new_file(&cghost_temp, &cghost)?;
        }

        // Replace the full source only after both derived artifacts validate.
        // Windows ReplaceFileW and Unix rename provide the atomic name swap.
        let backup = parent.join(format!(
            ".{GEMMA4_26B_GHOST_MODEL_FILENAME}.{nonce}.full-backup"
        ));
        replace_file_atomically(model_path, &hot_temp, &backup)?;

        let marker_bytes = serde_json::to_vec_pretty(&marker_for_model(model_path))
            .map_err(|error| invalid(format!("could not serialize Ghost-MoE marker: {error}")))?;
        std::fs::write(&marker_temp, marker_bytes).map_err(|source| BackendError::Io {
            path: marker_temp.clone(),
            source,
        })?;
        promote_new_file(&marker_temp, &marker)?;
        Ok(())
    })();

    if result.is_err() {
        std::fs::remove_file(&cghost_temp).ok();
        std::fs::remove_file(&hot_temp).ok();
        std::fs::remove_file(&marker_temp).ok();
    }
    result
}

/// Apply measured Windows CUDA defaults for a catalog-managed installation.
/// Explicit operator values always win.
pub fn apply_catalog_cuda_defaults() {
    for (key, value) in [
        ("CAMELID_GEMMA4_GHOST_CUDA_CACHE", "1"),
        ("CAMELID_GEMMA4_GHOST_CUDA_CACHE_EXPERTS", "1600"),
        ("CAMELID_GEMMA4_GHOST_CUDA_RESERVE_MIB", "160"),
        ("CAMELID_GEMMA4_CUDA_BATCHED_EXPERTS", "1"),
        ("CAMELID_GEMMA4_CUDA_PINNED_EXPERTS", "1"),
    ] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::SimdCaps;

    fn host(cuda: bool, vram: u64) -> HardwareProfile {
        HardwareProfile {
            metal_available: false,
            metal_device_name: None,
            metal_unified_memory: false,
            cuda_available: cuda,
            cuda_device_count: usize::from(cuda),
            cuda_device_name: cuda.then(|| "fixture".to_string()),
            cuda_compute_capability: cuda.then_some((8, 6)),
            cuda_tensor_cores: cuda,
            cuda_vram_total_bytes: vram,
            cuda_vram_free_bytes: vram,
            cpu_logical_cores: 8,
            host_ram_total_bytes: 32 * 1024 * 1024 * 1024,
            host_ram_free_bytes: 24 * 1024 * 1024 * 1024,
            host_ram_unevictable_bytes: 0,
            simd: SimdCaps::default(),
        }
    }

    #[test]
    fn only_the_exact_26b_row_advertises_ghost() {
        let hw = host(true, 6 * 1024 * 1024 * 1024);
        let support = catalog_support(GEMMA4_26B_GHOST_CATALOG_ID, FitVerdict::WontFit, &hw)
            .expect("exact row");
        assert!(support.available);
        #[cfg(all(target_os = "windows", feature = "cuda"))]
        {
            assert!(support.host_eligible);
            assert_eq!(support.fit, FitVerdict::FitsResident);
            assert!(support.recommended);
        }
        assert!(catalog_support("gemma4_e4b_it_q8_0", FitVerdict::WontFit, &hw).is_none());
    }

    #[test]
    fn busy_but_large_enough_gpu_can_prepare_ghost() {
        assert!(fit_allows_preparation(FitVerdict::FitsResident));
        assert!(fit_allows_preparation(FitVerdict::InsufficientFreeMemory));
        assert!(!fit_allows_preparation(FitVerdict::WontFit));
        assert!(!fit_allows_preparation(FitVerdict::Unknown));
    }

    #[test]
    fn marker_path_is_a_non_gguf_sibling() {
        assert_eq!(
            marker_path(Path::new(GEMMA4_26B_GHOST_MODEL_FILENAME)),
            PathBuf::from("gemma-4-26B_q4_0-it.ghost.json")
        );
    }

    #[test]
    fn deleting_a_prepared_row_removes_its_owned_sidecars() {
        let dir =
            std::env::temp_dir().join(format!("camelid-ghost-delete-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join(GEMMA4_26B_GHOST_MODEL_FILENAME);
        let marker = marker_path(&model);
        let cghost = dir.join(GEMMA4_26B_GHOST_CGHOST_FILENAME);
        std::fs::write(
            &marker,
            serde_json::to_vec(&marker_for_model(&model)).unwrap(),
        )
        .unwrap();
        std::fs::write(&cghost, b"ghost").unwrap();
        let expected = std::fs::metadata(&marker).unwrap().len() + 5;

        assert_eq!(remove_prepared_sidecars(&model).unwrap(), expected);
        assert!(!marker.exists());
        assert!(!cghost.exists());
        std::fs::remove_dir(&dir).unwrap();
    }
}
