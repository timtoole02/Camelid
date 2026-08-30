//! Memoized full-file GGUF SHA-256 for non-authoritative diagnostics.
//!
//! `build_loaded_model` hashes the whole GGUF so a receipt can name the lane
//! without re-hashing per request. That read is the dominant cost of loading a
//! large row: on a 2.5 GB Phi-4-mini artifact over an external SSD it is ~85 s
//! of pure I/O, paid on every process start, before the first request can be
//! served.
//!
//! The digest is unchanged — this caches it, it does not narrow it. A hit is
//! only accepted when the file still presents the exact same identity the
//! digest was recorded against: length, mtime to nanosecond precision, and the
//! filesystem file ID on Unix and Windows. Any ordinary rewrite — re-download,
//! re-quantize, `cp` over the top, editing in place — moves at least one of
//! those and misses the cache.
//!
//! The cache file is writable by the local user and its filesystem identity
//! key is an invalidation heuristic, not authentication. Consequently this
//! function must never authorize a capability, bind an agent request, select
//! an exact-artifact kernel, establish a distributed identity, or mint a
//! promotion receipt. Those paths use [`super::sha256_file_hex`] directly.
//!
//! Set `CAMELID_GGUF_HASH_CACHE=0` to always re-hash.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{sha256_file_hex, ReceiptError};

/// Entries kept before the oldest are pruned. A GGUF digest is ~150 bytes of
/// JSON, so this bounds the file at tens of kilobytes while comfortably
/// covering any realistic local model collection.
const MAX_ENTRIES: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    sha256: String,
    /// Unix seconds when this entry was written, used only to choose an
    /// eviction victim.
    stamped: u64,
}

/// Full-file SHA-256 of `path`, served from the on-disk cache when the file's
/// identity is unchanged since the digest was recorded.
///
/// Every failure mode here — unreadable cache, unwritable directory, corrupt
/// JSON, a filesystem that cannot report mtime — degrades to hashing the file,
/// never to an error and never to a wrong digest.
pub fn sha256_file_hex_cached(path: &Path) -> Result<String, ReceiptError> {
    cached_at(path, &cache_path(), enabled())
}

/// The whole of the caching logic, with the two pieces of ambient state — the
/// cache location and the opt-out — passed in. Tests drive this directly, so
/// they never mutate process-global env: `set_var` races every other test in
/// the binary, and this module's behaviour is entirely determined by these two
/// values anyway.
fn cached_at(path: &Path, cache_path: &Path, enabled: bool) -> Result<String, ReceiptError> {
    if !enabled {
        return sha256_file_hex(path);
    }
    let Some(key) = identity_key(path) else {
        return sha256_file_hex(path);
    };
    let mut entries = load(cache_path);
    if let Some(hit) = entries.get(&key) {
        return Ok(hit.sha256.clone());
    }
    // Miss: hash for real. Propagate this error — it means the GGUF itself
    // could not be read, which the caller must see.
    let digest = sha256_file_hex(path)?;
    entries.insert(
        key,
        CacheEntry {
            sha256: digest.clone(),
            stamped: now_unix(),
        },
    );
    prune(&mut entries);
    store(cache_path, &entries);
    Ok(digest)
}

fn enabled() -> bool {
    enabled_from(std::env::var("CAMELID_GGUF_HASH_CACHE").ok().as_deref())
}

/// On by default; off for the spellings someone reaching for an opt-out would
/// actually reach for. An unset or empty value is not an opt-out.
fn enabled_from(value: Option<&str>) -> bool {
    !matches!(value, Some("0") | Some("false") | Some("no"))
}

/// The file as the filesystem currently describes it. `None` whenever any
/// component is unavailable, which forces an honest re-hash rather than a key
/// that silently ignores a dimension.
fn identity_key(path: &Path) -> Option<String> {
    let canonical = std::fs::canonicalize(path).ok()?;
    let meta = std::fs::metadata(&canonical).ok()?;
    let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some(format!(
        "{}|{}|{}.{:09}{}",
        canonical.display(),
        meta.len(),
        modified.as_secs(),
        modified.subsec_nanos(),
        volume_identity(&canonical, &meta)?
    ))
}

/// The filesystem identity pins the file across a rename or a same-path
/// replacement that happens to preserve length and mtime. If the platform
/// cannot provide it, return `None` and force an honest re-hash.
#[cfg(unix)]
fn volume_identity(_path: &Path, meta: &std::fs::Metadata) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    Some(format!("|{}|{}", meta.dev(), meta.ino()))
}

#[cfg(windows)]
fn volume_identity(path: &Path, _meta: &std::fs::Metadata) -> Option<String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::{
        Foundation::{CloseHandle, INVALID_HANDLE_VALUE},
        Storage::FileSystem::{
            CreateFileW, GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
            FILE_ATTRIBUTE_NORMAL, FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ,
            FILE_SHARE_WRITE, OPEN_EXISTING,
        },
    };

    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return None;
    }

    let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
    let ok = unsafe { GetFileInformationByHandle(handle, info.as_mut_ptr()) } != 0;
    unsafe { CloseHandle(handle) };
    if !ok {
        return None;
    }
    let info = unsafe { info.assume_init() };
    Some(format!(
        "|{}|{}|{}|{}",
        info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow, info.nNumberOfLinks
    ))
}

#[cfg(not(any(unix, windows)))]
fn volume_identity(_path: &Path, _meta: &std::fs::Metadata) -> Option<String> {
    None
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

fn prune(entries: &mut BTreeMap<String, CacheEntry>) {
    while entries.len() > MAX_ENTRIES {
        let Some(oldest) = entries
            .iter()
            .min_by_key(|(key, entry)| (entry.stamped, (*key).clone()))
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        entries.remove(&oldest);
    }
}

fn load(path: &Path) -> BTreeMap<String, CacheEntry> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

/// Write via temp-file + rename so a concurrent reader sees either the old map
/// or the new one, never a half-written file. Two processes racing here is
/// benign: the loser's entry is dropped and simply re-hashed next time.
fn store(path: &Path, entries: &BTreeMap<String, CacheEntry>) {
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    set_private_permissions(parent);
    let Ok(serialized) = serde_json::to_string(entries) else {
        return;
    };
    let temp = parent.join(format!("gguf-sha256.{}.tmp", std::process::id()));
    if std::fs::write(&temp, serialized).is_err() {
        let _ = std::fs::remove_file(&temp);
        return;
    }
    if std::fs::rename(&temp, path).is_err() {
        let _ = std::fs::remove_file(&temp);
    }
}

#[cfg(unix)]
fn set_private_permissions(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn set_private_permissions(_dir: &Path) {}

/// Mirrors `responses_store::default_store_path`: explicit override first,
/// then the platform state directory, then a temp-dir fallback.
fn cache_path() -> PathBuf {
    if let Some(path) =
        std::env::var_os("CAMELID_GGUF_HASH_CACHE_PATH").filter(|value| !value.is_empty())
    {
        return PathBuf::from(path);
    }
    #[cfg(windows)]
    if let Some(base) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(base).join("Camelid").join("gguf-sha256.json");
    }
    if let Some(base) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(base).join("camelid").join("gguf-sha256.json");
    }
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("camelid")
            .join("gguf-sha256.json");
    }
    std::env::temp_dir()
        .join("camelid")
        .join("gguf-sha256.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ordinary rewrites should invalidate the performance hint. This is an
    /// optimization contract only; security-sensitive callers bypass it even
    /// when an identity key appears unchanged.
    #[test]
    fn cache_returns_the_true_digest_and_misses_when_the_file_changes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("cache").join("gguf-sha256.json");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"first contents").expect("write gguf");

        let truth = sha256_file_hex(&gguf).expect("hash");
        let cold = cached_at(&gguf, &cache, true).expect("cold");
        assert_eq!(cold, truth, "a cold miss must return the real digest");
        assert!(cache.exists(), "a miss must persist the entry");

        let warm = cached_at(&gguf, &cache, true).expect("warm");
        assert_eq!(warm, truth, "a hit must return the same digest");

        // Rewrite with different length + fresh mtime: the identity moves, so
        // the stale entry must not be served.
        std::fs::write(&gguf, b"second contents, longer").expect("rewrite");
        let after = cached_at(&gguf, &cache, true).expect("after rewrite");
        let truth_after = sha256_file_hex(&gguf).expect("rehash");
        assert_eq!(
            after, truth_after,
            "a changed file must not be served the previous digest"
        );
        assert_ne!(after, truth, "the digest must actually have moved");
    }

    /// The length-preserving rewrite is the case the mtime and inode
    /// components of the key exist for: content changed, size identical.
    #[test]
    fn same_length_rewrite_is_not_served_from_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("gguf-sha256.json");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"AAAAAAAAAAAAAAAA").expect("write gguf");

        let first = cached_at(&gguf, &cache, true).expect("cold");

        // Ensure the mtime actually advances even on a coarse-grained clock.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&gguf, b"BBBBBBBBBBBBBBBB").expect("rewrite same length");

        let second = cached_at(&gguf, &cache, true).expect("after rewrite");
        assert_eq!(
            second,
            sha256_file_hex(&gguf).expect("truth"),
            "a same-length rewrite must still miss the cache"
        );
        assert_ne!(first, second);
    }

    #[test]
    fn disabled_cache_still_returns_the_true_digest_and_writes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("cache").join("gguf-sha256.json");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"contents").expect("write gguf");

        let got = cached_at(&gguf, &cache, false).expect("hash");
        assert_eq!(got, sha256_file_hex(&gguf).expect("truth"));
        assert!(!cache.exists(), "the opt-out must not write a cache file");
    }

    #[test]
    fn authoritative_hash_ignores_a_forged_performance_cache_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("gguf-sha256.json");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"actual model bytes").expect("write gguf");
        let key = identity_key(&gguf).expect("filesystem identity");
        let forged = "00".repeat(32);
        let mut entries = BTreeMap::new();
        entries.insert(
            key,
            CacheEntry {
                sha256: forged.clone(),
                stamped: now_unix(),
            },
        );
        store(&cache, &entries);

        assert_eq!(
            cached_at(&gguf, &cache, true).expect("performance-cache hit"),
            forged,
            "the local cache is explicitly not an authentication boundary"
        );
        assert_eq!(
            super::sha256_file_hex(&gguf).expect("authoritative hash"),
            crate::receipt::sha256_hex(b"actual model bytes"),
            "security-sensitive hashing must read the artifact bytes"
        );
    }

    #[test]
    fn a_missing_gguf_still_reports_the_read_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("gguf-sha256.json");
        assert!(cached_at(&dir.path().join("absent.gguf"), &cache, true).is_err());
    }

    /// The opt-out reads as a disable for the spellings someone would try, and
    /// as *enabled* for everything else — including unset and empty, so a
    /// stray export cannot silently turn caching off.
    #[test]
    fn opt_out_spellings() {
        for value in [Some("0"), Some("false"), Some("no")] {
            assert!(!enabled_from(value), "{value:?} must disable the cache");
        }
        for value in [None, Some(""), Some("1"), Some("true"), Some("yes")] {
            assert!(enabled_from(value), "{value:?} must leave the cache on");
        }
    }

    #[test]
    fn prune_evicts_oldest_first_and_bounds_the_map() {
        let mut entries = BTreeMap::new();
        for i in 0..(MAX_ENTRIES + 10) {
            entries.insert(
                format!("key-{i:04}"),
                CacheEntry {
                    sha256: format!("{i:064x}"),
                    stamped: i as u64,
                },
            );
        }
        prune(&mut entries);
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert!(
            !entries.contains_key("key-0000"),
            "the oldest entry must be the first evicted"
        );
        assert!(
            entries.contains_key(&format!("key-{:04}", MAX_ENTRIES + 9)),
            "the newest entry must survive"
        );
    }

    /// End-to-end on a real multi-gigabyte artifact: the cached digest must
    /// equal the digest of the whole file, and the second call must not read
    /// it again. Opt-in because the artifact is multi-gigabyte.
    ///
    /// `CAMELID_GGUF_HASH_CACHE_TEST_SHA256` optionally pins the expected
    /// digest, so the run also proves the cache did not invent one.
    #[test]
    #[ignore = "set CAMELID_GGUF_HASH_CACHE_TEST_GGUF to a large GGUF"]
    fn real_artifact_is_hashed_once_and_served_from_cache() {
        let path = std::env::var("CAMELID_GGUF_HASH_CACHE_TEST_GGUF")
            .expect("set CAMELID_GGUF_HASH_CACHE_TEST_GGUF");
        let gguf = std::path::PathBuf::from(path);
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("gguf-sha256.json");

        let cold_start = std::time::Instant::now();
        let cold = cached_at(&gguf, &cache, true).expect("cold");
        let cold_ms = cold_start.elapsed().as_secs_f64() * 1e3;

        let warm_start = std::time::Instant::now();
        let warm = cached_at(&gguf, &cache, true).expect("warm");
        let warm_ms = warm_start.elapsed().as_secs_f64() * 1e3;

        eprintln!("cold_ms={cold_ms:.1} warm_ms={warm_ms:.1} sha256={cold}");
        assert_eq!(cold, warm, "the cache must serve the digest it recorded");
        assert_eq!(
            cold,
            sha256_file_hex(&gguf).expect("uncached truth"),
            "the cached digest must equal the whole-file digest"
        );
        if let Ok(expected) = std::env::var("CAMELID_GGUF_HASH_CACHE_TEST_SHA256") {
            assert_eq!(cold, expected, "digest must match the pinned value");
        }
        assert!(
            warm_ms < cold_ms / 10.0,
            "a hit must not re-read the file (cold {cold_ms:.1} ms, warm {warm_ms:.1} ms)"
        );
    }

    /// A corrupt or truncated cache file must degrade to hashing, not poison
    /// the result and not propagate a parse error to the load path.
    #[test]
    fn corrupt_cache_file_degrades_to_hashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("gguf-sha256.json");
        std::fs::write(&cache, b"{not json").expect("write corrupt cache");
        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"contents").expect("write gguf");

        std::env::set_var("CAMELID_GGUF_HASH_CACHE_PATH", &cache);
        std::env::remove_var("CAMELID_GGUF_HASH_CACHE");

        let got = sha256_file_hex_cached(&gguf).expect("hash despite corrupt cache");
        assert_eq!(got, sha256_file_hex(&gguf).expect("truth"));

        std::env::remove_var("CAMELID_GGUF_HASH_CACHE_PATH");
    }
}
